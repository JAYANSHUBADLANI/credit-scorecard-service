"""The monitoring loop: claim a window, compute the indices, apply the debounce, record.

This runs on a schedule and keeps running. That is the whole point of the project. A stability
index computed once at model build time tells you the model fitted the data it was fitted on.
It says nothing about month seven, which is when the population actually moves and when nobody
is looking any more.

Windows are claimed by scoring log id rather than by wall clock time. Ids are dense, ordered
and assigned by the database, so a window is always exactly `window_size` scored requests,
which keeps the sampling variance of the stability index constant from window to window. Time
based windows would vary in size with traffic, and a quiet overnight window would then produce
a large index for no reason other than having fewer records in it. A monitor that alerts every
night at 3am is a monitor that gets muted.

A partial window is not scored. The loop waits for a full one, and records that it waited, so
the run history distinguishes "nothing to do" from "nothing happened". The `--flush` flag
overrides that for the end of a finite replay, subject to `min_window_size`.
"""

from __future__ import annotations

import argparse
import json
import signal
import sys
import time
from typing import Any, Dict, List, Optional

import pandas as pd

from .alerting import (
    POPULATION_METRICS,
    DebounceConfig,
    evaluate_alert,
    is_attribution_only,
)
from .config import Config, load_config
from .drift import (
    METRIC_CSI,
    STATUS_ALERT,
    DriftMonitor,
    DriftThresholds,
    bin_indices_from_rows,
)
from .scoring import ScoringService
from .store import Store, utc_now

OUTCOME_SCORED = "scored"
OUTCOME_WAITING = "waiting"
OUTCOME_SKIPPED = "skipped"


class MonitorRunner:
    """Owns the artifact, the store and the thresholds for the life of the process."""

    def __init__(self, config: Optional[Config] = None):
        self.config = config or load_config()
        self.store = Store(self.config.path(self.config.service.db_path))
        self.scoring = ScoringService.from_path(self.config.path(self.config.service.model_path))

        monitoring = self.config.monitoring
        self.thresholds = DriftThresholds(
            psi_warn=monitoring.psi_warn,
            psi_alert=monitoring.psi_alert,
            csi_warn=monitoring.csi_warn,
            csi_alert=monitoring.csi_alert,
            prediction_psi_warn=monitoring.prediction_psi_warn,
            prediction_psi_alert=monitoring.prediction_psi_alert,
        )
        self.drift = DriftMonitor.from_artifact(self.scoring.artifact, self.thresholds)
        self.debounce = DebounceConfig(
            debounce_windows=monitoring.debounce_windows,
            cooldown_windows=monitoring.cooldown_windows,
        )

    def run_once(self, flush: bool = False) -> Dict[str, Any]:
        """Process at most one window. Returns what happened, for the run log and the caller."""
        monitoring = self.config.monitoring
        after_id = self.store.last_window_end_id()
        rows = self.store.fetch_scores_after(after_id, monitoring.window_size)

        if not rows:
            self.store.record_run(None, 0, OUTCOME_WAITING, "no new scored requests")
            return {"outcome": OUTCOME_WAITING, "n_records": 0}

        if len(rows) < monitoring.window_size:
            if not flush:
                self.store.record_run(
                    None,
                    len(rows),
                    OUTCOME_WAITING,
                    f"{len(rows)} of {monitoring.window_size} required, waiting for a full window",
                )
                return {"outcome": OUTCOME_WAITING, "n_records": len(rows)}
            if len(rows) < monitoring.min_window_size:
                self.store.record_run(
                    None,
                    len(rows),
                    OUTCOME_SKIPPED,
                    f"{len(rows)} below the minimum window of {monitoring.min_window_size}, "
                    "too few records to measure stability on",
                )
                return {"outcome": OUTCOME_SKIPPED, "n_records": len(rows)}

        window_id = self.store.last_window_id() + 1
        window_start_id = int(rows[0]["id"])
        window_end_id = int(rows[-1]["id"])

        scores = [float(row["score"]) for row in rows]
        probabilities = [float(row["probability"]) for row in rows]
        bands = [str(row["band"]) for row in rows]
        bins = bin_indices_from_rows(rows, self.drift.features)

        results = self.drift.evaluate(scores, bands, bins)
        summary = self.drift.summarise(scores, probabilities, bands)

        computed_at = utc_now()
        self.store.record_metrics(
            [
                {
                    **result.as_row(),
                    "window_id": window_id,
                    "computed_at": computed_at,
                    "window_start_id": window_start_id,
                    "window_end_id": window_end_id,
                    "n_records": summary.n_records,
                }
                for result in results
            ]
        )
        self.store.record_window_summary(
            {
                "window_id": window_id,
                "computed_at": computed_at,
                "window_start_id": window_start_id,
                "window_end_id": window_end_id,
                "first_scored_at": rows[0]["scored_at"],
                "last_scored_at": rows[-1]["scored_at"],
                **summary.as_dict(),
            }
        )

        fired = self.apply_debounce(results, window_id)

        detail = json.dumps(
            {
                "window_id": window_id,
                "breaching": [r.feature for r in results if r.status == STATUS_ALERT],
                "alerts_fired": len(fired),
                **summary.as_dict(),
            },
            separators=(",", ":"),
        )
        self.store.record_run(window_id, summary.n_records, OUTCOME_SCORED, detail)

        return {
            "outcome": OUTCOME_SCORED,
            "window_id": window_id,
            "n_records": summary.n_records,
            "metrics": [r.as_row() for r in results],
            "alerts_fired": fired,
            "summary": summary.as_dict(),
        }

    def apply_debounce(self, results, window_id: int) -> List[Dict[str, Any]]:
        """Run every breaching metric through the debounce and write the ones that survive.

        Two tiers. The population metrics alert in their own right. A characteristic that
        breaches while the population is also in breach is folded into that alert as
        attribution rather than raising its own, because it is explaining one event rather
        than reporting a second. A characteristic that breaches on its own still alerts.
        """
        population_in_breach = any(
            result.metric in POPULATION_METRICS and result.status == STATUS_ALERT
            for result in results
        )
        breaching_characteristics = [
            result.feature
            for result in results
            if result.metric == METRIC_CSI and result.status == STATUS_ALERT
        ]

        fired: List[Dict[str, Any]] = []
        for result in results:
            if is_attribution_only(result.metric, population_in_breach):
                continue
            history = self.store.fetch_metric_history(
                result.metric, result.feature, limit=max(self.debounce.debounce_windows * 4, 20)
            )
            history_rows = [
                {"status": row["status"], "window_id": int(row["window_id"])}
                for row in history
                if int(row["window_id"]) < window_id
            ]
            decision = evaluate_alert(
                result=result,
                window_id=window_id,
                history_newest_first=history_rows,
                config=self.debounce,
                last_alert_window=self.store.last_alert_window(result.metric, result.feature),
            )
            if not decision.should_fire:
                continue
            self.store.record_alert(
                window_id=window_id,
                metric=decision.metric,
                feature=decision.feature,
                value=decision.value,
                threshold=decision.threshold,
                severity="alert",
                consecutive_windows=decision.consecutive_windows,
                breach_window_ids=decision.breach_window_ids,
                message=decision.message(),
                attributed_features=(
                    breaching_characteristics
                    if decision.metric in POPULATION_METRICS
                    else []
                ),
            )
            fired.append(
                {
                    "metric": decision.metric,
                    "feature": decision.feature,
                    "value": round(decision.value, 4),
                    "threshold": decision.threshold,
                    "consecutive_windows": decision.consecutive_windows,
                    "attributed_features": (
                        breaching_characteristics
                        if decision.metric in POPULATION_METRICS
                        else []
                    ),
                    "message": decision.message(),
                }
            )
        return fired

    def drain(self, max_windows: int = 10000, verbose: bool = True) -> Dict[str, Any]:
        """Process every complete window available right now, then flush the remainder.

        This is what the reproducible end to end run uses. The scheduled loop is the same
        `run_once` on a timer, so draining a finite replay and monitoring a live service
        follow the identical code path.
        """
        processed = 0
        alerts = 0
        for _ in range(max_windows):
            outcome = self.run_once(flush=False)
            if outcome["outcome"] != OUTCOME_SCORED:
                break
            processed += 1
            alerts += len(outcome.get("alerts_fired", []))
            if verbose and outcome.get("alerts_fired"):
                for alert in outcome["alerts_fired"]:
                    print(f"  ALERT window {outcome['window_id']}: {alert['message']}", flush=True)

        final = self.run_once(flush=True)
        if final["outcome"] == OUTCOME_SCORED:
            processed += 1
            alerts += len(final.get("alerts_fired", []))
            if verbose and final.get("alerts_fired"):
                for alert in final["alerts_fired"]:
                    print(f"  ALERT window {final['window_id']}: {alert['message']}", flush=True)

        return {"windows_processed": processed, "alerts_fired": alerts}

    def loop(self, interval: float, verbose: bool = True) -> None:
        """Run on a schedule until the process is asked to stop."""
        stopping = {"flag": False}

        def handle_signal(signum, frame):
            stopping["flag"] = True
            if verbose:
                print(f"received signal {signum}, finishing the current window", flush=True)

        signal.signal(signal.SIGTERM, handle_signal)
        signal.signal(signal.SIGINT, handle_signal)

        if verbose:
            print(
                f"monitor started, window={self.config.monitoring.window_size} requests, "
                f"interval={interval}s, debounce={self.debounce.debounce_windows} windows",
                flush=True,
            )

        while not stopping["flag"]:
            try:
                outcome = self.run_once(flush=False)
                if verbose and outcome["outcome"] == "scored":
                    breaching = [
                        m["feature"] for m in outcome["metrics"] if m["status"] == STATUS_ALERT
                    ]
                    print(
                        f"window {outcome['window_id']}: n={outcome['n_records']} "
                        f"breaching={breaching or 'none'} "
                        f"alerts={len(outcome['alerts_fired'])}",
                        flush=True,
                    )
                    for alert in outcome["alerts_fired"]:
                        print(f"  ALERT: {alert['message']}", flush=True)
            except Exception as error:
                # A monitor that dies on one bad window stops monitoring everything else.
                print(f"monitor error, continuing: {error}", file=sys.stderr, flush=True)
                self.store.record_run(None, 0, "error", str(error)[:500])

            for _ in range(int(max(interval, 1))):
                if stopping["flag"]:
                    break
                time.sleep(1.0)

        if verbose:
            print("monitor stopped", flush=True)

    def export_reports(self) -> Dict[str, int]:
        """Write the monitoring history out as CSV so results survive the database."""
        reports = self.config.path("reports")
        reports.mkdir(parents=True, exist_ok=True)

        metrics = pd.DataFrame([dict(row) for row in self.store.fetch_all_metrics()])
        summaries = pd.DataFrame([dict(row) for row in self.store.fetch_window_summaries()])
        alerts = pd.DataFrame([dict(row) for row in self.store.fetch_alerts()])

        if not metrics.empty:
            metrics.to_csv(reports / "drift_metrics.csv", index=False)
        if not summaries.empty:
            summaries.to_csv(reports / "window_summary.csv", index=False)
        alerts.to_csv(reports / "alerts.csv", index=False)

        return {
            "metric_rows": len(metrics),
            "windows": len(summaries),
            "alerts": len(alerts),
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Rolling drift monitor for the scoring service")
    parser.add_argument("--loop", action="store_true", help="run continuously on an interval")
    parser.add_argument("--interval", type=float, default=30.0, help="seconds between windows")
    parser.add_argument("--drain", action="store_true", help="process every available window, then exit")
    parser.add_argument("--flush", action="store_true", help="allow a final partial window")
    parser.add_argument("--export", action="store_true", help="write the monitoring history to reports/")
    args = parser.parse_args()

    runner = MonitorRunner()

    if args.loop:
        runner.loop(interval=args.interval)
    elif args.drain:
        result = runner.drain()
        print(json.dumps(result, indent=2))
    else:
        result = runner.run_once(flush=args.flush)
        print(json.dumps(result, indent=2, default=str))

    if args.export or args.drain:
        print(json.dumps(runner.export_reports(), indent=2))


if __name__ == "__main__":
    main()
