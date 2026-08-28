"""The simulated scoring stream.

READ THIS BEFORE QUOTING ANY NUMBER THIS PRODUCES. This is a simulation. It is not production
traffic, and nothing in this repository is production traffic. What is real and what is
constructed:

Real: every application posted to the API is a genuine, unmodified row from the held out slice
of application_train.csv. No feature value is ever synthesised, perturbed or invented. The
scores that come back are produced by the served model through the same HTTP endpoint any
caller would use, and the API writes the scoring log itself.

Constructed: the arrival order, and the timing. The stream decides which real applications
arrive when.

The stream runs in two documented regimes:

  Regime 1, the first `stable_fraction` of requests. Applications are drawn from the held out
  slice with equal probability. This is the control. The population genuinely has not shifted,
  so the stability indices should stay near zero, and any alert raised here is a false
  positive. Getting this half right matters as much as the other half: it is what shows the
  monitor is not simply firing on noise.

  Regime 2, the remainder. The same pool of real applications is drawn with unequal
  probability, weighted toward lower external bureau scores and younger applicants, ramping up
  over the regime. The effect is a portfolio that is quietly taking on a riskier mix, which is
  the realistic version of this failure: not a broken feed, but an origination channel
  gradually changing who it sends.

The reason for injecting the shift is that the held out slice does not drift on its own. It is
a later slice of the same static Kaggle extract, so its distribution is close to the training
slice by construction. Waiting for natural drift in it would mean a monitoring system that was
never once observed to detect anything, which proves nothing. A known, deliberately introduced
shift is the only way to demonstrate that detection, attribution and the debounce all work.
The tradeoff is that the shift is one I chose, so it is evidence the mechanism works, not
evidence about how this population behaves in the field.

`reports/stream_manifest.json` records the exact parameters of every run.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import httpx
import numpy as np
import pandas as pd

from .config import Config, load_config
from .features import RAW_INPUTS

REGIME_STABLE = "stable"
REGIME_DRIFTED = "drifted"

# Lower is riskier for each of these, so the drifted regime weights toward the low end. Age is
# included with the same sign because younger applicants are the higher risk group here.
DRIFT_DIRECTION = {"EXT_SOURCE_1": -1.0, "EXT_SOURCE_2": -1.0, "EXT_SOURCE_3": -1.0, "AGE_YEARS": -1.0}


@dataclass
class StreamStats:
    sent: int = 0
    accepted: int = 0
    rejected: int = 0
    batches: int = 0
    elapsed_seconds: float = 0.0
    reject_examples: List[str] = field(default_factory=list)


def payload_from_row(row: pd.Series) -> Dict[str, Any]:
    """Turn a held out data row into a request body, with nulls where the data is null."""
    payload: Dict[str, Any] = {}
    for column in RAW_INPUTS:
        value = row[column]
        if pd.isna(value):
            payload[column] = None
        elif isinstance(value, str):
            payload[column] = value
        else:
            payload[column] = float(value)
    return payload


def drift_propensity(frame: pd.DataFrame, features: List[str]) -> np.ndarray:
    """A standardised, per record measure of how much the drifted regime favours it.

    Each named characteristic is standardised, signed so that the riskier direction is
    positive, and averaged. Missing values contribute nothing rather than dropping the record,
    so an applicant with no bureau score is neither favoured nor penalised by that field.
    """
    scores = []
    for feature in features:
        if feature == "AGE_YEARS":
            # Derived rather than raw: the holdout slice carries DAYS_BIRTH, not the feature.
            values = -pd.to_numeric(frame["DAYS_BIRTH"], errors="coerce") / 365.25
        else:
            values = pd.to_numeric(frame[feature], errors="coerce")
        mean = values.mean()
        std = values.std()
        if not np.isfinite(std) or std == 0:
            continue
        z = ((values - mean) / std).fillna(0.0).to_numpy()
        scores.append(DRIFT_DIRECTION.get(feature, -1.0) * z)
    if not scores:
        return np.zeros(len(frame), dtype="float64")
    return np.mean(scores, axis=0)


def batch_weights(propensity: np.ndarray, exponent: float) -> np.ndarray:
    """Sampling weights for one batch. An exponent of zero gives a uniform draw."""
    if exponent <= 0:
        return np.full(len(propensity), 1.0 / len(propensity))
    logits = exponent * propensity
    logits -= logits.max()
    weights = np.exp(logits)
    return weights / weights.sum()


def regime_for_batch(
    batch_index: int, n_batches: int, stable_fraction: float, ramp_fraction: float = 1.0
) -> Tuple[str, float]:
    """Which regime a batch belongs to, and how far into the drifted ramp it is.

    The ramp reaches full strength after `ramp_fraction` of the drifted regime and then holds
    there. Onset followed by a plateau is both the more realistic shape and the one that
    actually tests the debounce, since a shift that only peaks in the final batch never
    produces the consecutive breaching windows an alert requires.
    """
    progress = batch_index / max(n_batches - 1, 1)
    if progress < stable_fraction:
        return REGIME_STABLE, 0.0
    span = max(1.0 - stable_fraction, 1e-9)
    into_regime = (progress - stable_fraction) / span
    ramp = into_regime / max(ramp_fraction, 1e-9)
    return REGIME_DRIFTED, float(min(ramp, 1.0))


class StreamRunner:
    """Replays held out applications against the live API as ordinary HTTP requests."""

    def __init__(self, config: Config, api_url: str, verbose: bool = True):
        self.config = config
        self.api_url = api_url.rstrip("/")
        self.verbose = verbose

    def load_holdout(self) -> pd.DataFrame:
        path = self.config.path("data/processed/holdout.parquet")
        if not path.exists():
            raise FileNotFoundError(
                f"no held out slice at {path}. Run `make train` first, which writes it."
            )
        return pd.read_parquet(path)

    def wait_for_api(self, timeout: float = 60.0) -> None:
        deadline = time.time() + timeout
        last_error: Optional[str] = None
        while time.time() < deadline:
            try:
                response = httpx.get(f"{self.api_url}/health", timeout=5.0)
                if response.status_code == 200:
                    if self.verbose:
                        print(f"api ready at {self.api_url}: {response.json()}")
                    return
                last_error = f"status {response.status_code}"
            except Exception as error:  # the API may simply not be up yet
                last_error = str(error)
            time.sleep(1.0)
        raise RuntimeError(f"api at {self.api_url} not ready after {timeout}s: {last_error}")

    def run(self) -> Dict[str, Any]:
        stream = self.config.stream
        holdout = self.load_holdout()
        propensity = drift_propensity(holdout, stream.drift_features)
        rng = np.random.default_rng(stream.seed)

        n_batches = max(int(np.ceil(stream.total_requests / stream.requests_per_batch)), 1)
        stats = StreamStats()
        regime_counts = {REGIME_STABLE: 0, REGIME_DRIFTED: 0}
        started = time.perf_counter()

        with httpx.Client(base_url=self.api_url, timeout=30.0) as client:
            remaining = stream.total_requests
            for batch_index in range(n_batches):
                size = min(stream.requests_per_batch, remaining)
                if size <= 0:
                    break
                regime, ramp = regime_for_batch(
                    batch_index, n_batches, stream.stable_fraction, stream.drift_ramp_fraction
                )
                exponent = stream.drift_strength * ramp
                weights = batch_weights(propensity, exponent)

                picks = rng.choice(len(holdout), size=size, replace=False, p=weights)
                batch = holdout.iloc[picks]

                for _, row in batch.iterrows():
                    payload = payload_from_row(row)
                    response = client.post("/score", json=payload)
                    stats.sent += 1
                    if response.status_code == 200:
                        stats.accepted += 1
                    else:
                        stats.rejected += 1
                        if len(stats.reject_examples) < 5:
                            stats.reject_examples.append(
                                f"{response.status_code}: {response.text[:200]}"
                            )

                regime_counts[regime] += size
                remaining -= size
                stats.batches += 1
                if self.verbose and (batch_index % 10 == 0 or remaining <= 0):
                    print(
                        f"batch {batch_index + 1}/{n_batches} regime={regime} "
                        f"ramp={ramp:.2f} sent={stats.sent} rejected={stats.rejected}",
                        flush=True,
                    )

        stats.elapsed_seconds = time.perf_counter() - started

        manifest = {
            "note": (
                "Simulated scoring traffic. Every application is a real, unmodified row from "
                "the held out slice. The arrival order is constructed, and the drifted regime "
                "reweights which real applications arrive. No feature value is synthetic."
            ),
            "api_url": self.api_url,
            "holdout_rows_available": int(len(holdout)),
            "requests_sent": stats.sent,
            "requests_accepted": stats.accepted,
            "requests_rejected": stats.rejected,
            "reject_examples": stats.reject_examples,
            "batches": stats.batches,
            "requests_per_batch": stream.requests_per_batch,
            "elapsed_seconds": round(stats.elapsed_seconds, 2),
            "requests_per_second": round(stats.sent / max(stats.elapsed_seconds, 1e-9), 1),
            "regime_counts": regime_counts,
            "parameters": {
                "total_requests": stream.total_requests,
                "stable_fraction": stream.stable_fraction,
                "drift_strength": stream.drift_strength,
                "drift_ramp_fraction": stream.drift_ramp_fraction,
                "drift_features": list(stream.drift_features),
                "seed": stream.seed,
            },
        }

        reports = self.config.path("reports")
        reports.mkdir(parents=True, exist_ok=True)
        with open(reports / "stream_manifest.json", "w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2)

        return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--api-url", default="http://localhost:8000")
    parser.add_argument("--total", type=int, default=None, help="override total_requests")
    parser.add_argument("--wait", type=float, default=60.0, help="seconds to wait for the API")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    config = load_config()
    if args.total is not None:
        config.stream.total_requests = args.total

    runner = StreamRunner(config, args.api_url, verbose=not args.quiet)
    runner.wait_for_api(timeout=args.wait)
    manifest = runner.run()
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
