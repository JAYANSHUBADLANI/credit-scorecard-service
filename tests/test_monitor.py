"""Tests for window claiming and the monitor's integration with the store.

The unit tests in test_alerting.py check the debounce rule in isolation. These check that the
rule still holds once it is driven through the real store: that windows are claimed exactly
once, that a partial window waits rather than being scored on too little data, and that a
sustained breach produces exactly one alert rather than one per window.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from src.config import load_config
from src.monitor import OUTCOME_SCORED, OUTCOME_SKIPPED, OUTCOME_WAITING, MonitorRunner
from src.store import Store
from tests.conftest import requires_model

pytestmark = requires_model


@pytest.fixture
def runner(temp_config):
    config = load_config(temp_config)
    config.monitoring.window_size = 200
    config.monitoring.min_window_size = 50
    config.monitoring.debounce_windows = 3
    config.monitoring.cooldown_windows = 5
    return MonitorRunner(config)


def write_requests(
    runner: MonitorRunner,
    n: int,
    shift: float = 0.0,
    seed: int = 0,
    shift_features: list | None = None,
) -> None:
    """Append synthetic scored requests.

    `shift` of zero reproduces the training reference distribution, so the indices read as
    stable. Raising it moves mass into the low bins of every characteristic and into the
    decline band, which is what a riskier population looks like to the monitor.

    `shift_features` restricts the shift to named characteristics and leaves the score and
    band distributions at their reference. That is the localised case: one input has moved,
    and the population level metrics have not noticed.
    """
    rng = np.random.default_rng(seed)
    artifact = runner.scoring.artifact
    store = runner.store

    edges = artifact.reference_score_edges
    score_reference = np.asarray(artifact.reference_score_proportions, dtype="float64")
    band_reference = np.asarray(artifact.reference_band_proportions, dtype="float64")
    bands = ["decline", "refer", "approve"]

    def skew(proportions: np.ndarray) -> np.ndarray:
        if shift <= 0:
            return proportions
        weights = np.exp(-shift * np.arange(len(proportions)))
        shifted = proportions * weights
        return shifted / shifted.sum()

    localised = shift_features is not None
    population_skew = (lambda p: p) if localised else skew

    score_bins = rng.choice(len(score_reference), size=n, p=population_skew(score_reference))
    padded = np.concatenate([[edges[0] - 25.0], edges, [edges[-1] + 25.0]])
    scores = [float(padded[i] + 0.5 * (padded[i + 1] - padded[i])) for i in score_bins]
    chosen_bands = rng.choice(bands, size=n, p=population_skew(band_reference[::-1])[::-1])

    rows = []
    for i in range(n):
        bin_indices = {}
        for name, binning in artifact.transformer.bins.items():
            reference = np.asarray(binning.reference_proportions, dtype="float64")
            apply_shift = (not localised) or (name in shift_features)
            bin_indices[name] = int(
                rng.choice(len(reference), p=skew(reference) if apply_shift else reference)
            )
        rows.append({
            "request_id": f"synthetic-{shift}-{seed}-{i}",
            "model_version": artifact.model_version,
            "source": "test",
            "score": scores[i],
            "probability": 0.08,
            "band": str(chosen_bands[i]),
            "bin_indices": bin_indices,
        })
    store.log_scores_bulk(rows)


# Window claiming ---------------------------------------------------------------------

def test_no_traffic_reports_waiting(runner):
    outcome = runner.run_once()
    assert outcome["outcome"] == OUTCOME_WAITING
    assert outcome["n_records"] == 0


def test_partial_window_waits_rather_than_being_scored(runner):
    write_requests(runner, 150)
    outcome = runner.run_once()
    assert outcome["outcome"] == OUTCOME_WAITING
    assert outcome["n_records"] == 150
    assert runner.store.last_window_id() == 0


def test_full_window_is_scored(runner):
    write_requests(runner, 200)
    outcome = runner.run_once()
    assert outcome["outcome"] == OUTCOME_SCORED
    assert outcome["window_id"] == 1
    assert outcome["n_records"] == 200


def test_windows_advance_without_overlapping(runner):
    write_requests(runner, 600)
    first = runner.run_once()
    second = runner.run_once()
    third = runner.run_once()

    assert [first["window_id"], second["window_id"], third["window_id"]] == [1, 2, 3]

    metrics = runner.store.fetch_all_metrics()
    spans = {
        int(row["window_id"]): (int(row["window_start_id"]), int(row["window_end_id"]))
        for row in metrics
    }
    assert spans[1] == (1, 200)
    assert spans[2] == (201, 400)
    assert spans[3] == (401, 600)


def test_leftover_requests_wait_for_the_next_full_window(runner):
    write_requests(runner, 250)
    assert runner.run_once()["outcome"] == OUTCOME_SCORED
    assert runner.run_once()["outcome"] == OUTCOME_WAITING

    write_requests(runner, 150, seed=1)
    assert runner.run_once()["outcome"] == OUTCOME_SCORED


def test_flush_scores_a_partial_window(runner):
    write_requests(runner, 120)
    assert runner.run_once(flush=False)["outcome"] == OUTCOME_WAITING
    outcome = runner.run_once(flush=True)
    assert outcome["outcome"] == OUTCOME_SCORED
    assert outcome["n_records"] == 120


def test_flush_still_refuses_a_window_below_the_minimum(runner):
    """Too few records to measure stability on is a skip, not a noisy reading."""
    write_requests(runner, 30)
    outcome = runner.run_once(flush=True)
    assert outcome["outcome"] == OUTCOME_SKIPPED
    assert runner.store.last_window_id() == 0


def test_every_wake_up_is_recorded_even_when_idle(runner):
    """A stopped monitor and a quiet one must be distinguishable."""
    runner.run_once()
    write_requests(runner, 200)
    runner.run_once()

    runs = runner.store.fetch_runs()
    outcomes = [row["outcome"] for row in runs]
    assert OUTCOME_WAITING in outcomes
    assert OUTCOME_SCORED in outcomes


# Metrics reach the store ---------------------------------------------------------------

def test_stable_traffic_produces_no_alerts(runner):
    """The control case. Traffic drawn from the training distribution must not raise.

    Note the window here is 200 requests, well under the configured 2000. At that size the
    index has enough sampling variance to cross the warn line on a population that has not
    moved at all, which is measured in scripts/window_size_noise.py. Nothing reaches the alert
    threshold, and nothing fires. That is the property being asserted: noise can tint a
    window amber, it cannot page anybody.
    """
    for seed in range(6):
        write_requests(runner, 200, shift=0.0, seed=seed)
    drained = runner.drain(verbose=False)

    assert drained["windows_processed"] == 6
    assert drained["alerts_fired"] == 0
    assert runner.store.count_alerts() == 0

    metrics = runner.store.fetch_all_metrics()
    assert metrics
    breaching = [dict(row) for row in metrics if row["status"] == "alert"]
    assert not breaching, f"stable traffic reached the alert threshold: {breaching}"


def test_a_larger_window_has_a_lower_noise_floor(runner):
    """Why the configured window is 2000 requests and not 200.

    Both windows below are drawn from the reference distribution, so both are stable by
    construction and every index is pure sampling noise. The larger window reads lower.
    """
    small = []
    large = []
    for seed in range(3):
        write_requests(runner, 200, shift=0.0, seed=100 + seed)
        outcome = runner.run_once(flush=True)
        small.extend(m["value"] for m in outcome["metrics"])

    runner.config.monitoring.window_size = 2000
    for seed in range(3):
        write_requests(runner, 2000, shift=0.0, seed=200 + seed)
        outcome = runner.run_once(flush=True)
        large.extend(m["value"] for m in outcome["metrics"])

    assert np.median(large) < np.median(small)
    assert max(large) < 0.10, "a 2000 request window on stable traffic should stay quiet"


def test_window_summary_is_recorded_for_each_window(runner):
    write_requests(runner, 400)
    runner.drain(verbose=False)
    summaries = runner.store.fetch_window_summaries()
    assert len(summaries) == 2
    for row in summaries:
        assert row["n_records"] == 200
        assert 0.0 <= row["approval_rate"] <= 1.0
        assert json.loads(row["band_counts"])


# The debounce, driven through the store --------------------------------------------------

def test_sustained_breach_fires_once_per_metric_not_once_per_window(runner):
    """Six breaching windows, a debounce of three and a cooldown of five.

    Only the two population metrics alert, once each. Without the cooldown each would raise
    four times. Without the debounce, six. Without the attribution tier, every breaching
    characteristic would raise as well, turning one shift into sixteen notifications.
    """
    for seed in range(6):
        write_requests(runner, 200, shift=1.2, seed=seed)
    drained = runner.drain(verbose=False)

    assert drained["windows_processed"] == 6
    alerts = runner.store.fetch_alerts()
    assert {row["metric"] for row in alerts} == {"psi_score", "psi_band"}
    assert len(alerts) == 2, [dict(a) for a in alerts]
    for row in alerts:
        assert row["consecutive_windows"] == 3
        assert row["window_id"] == 3


def test_population_alert_carries_the_characteristics_that_moved(runner):
    """The attribution has to travel with the alert, or it is not actionable."""
    for seed in range(4):
        write_requests(runner, 200, shift=1.2, seed=seed)
    runner.drain(verbose=False)

    alert = runner.store.fetch_alerts()[0]
    attributed = json.loads(alert["attributed_features"])
    assert attributed, "a population alert should name the characteristics in breach"
    assert "csi" not in {row["metric"] for row in runner.store.fetch_alerts()}


def test_characteristic_alerts_fire_when_the_population_looks_stable(runner):
    """The localised case: one input moves, the score does not, and it still gets caught."""
    for seed in range(4):
        write_requests(
            runner, 200, shift=2.0, seed=seed, shift_features=["EXT_SOURCE_1"]
        )
    runner.drain(verbose=False)

    alerts = runner.store.fetch_alerts()
    assert alerts, "a characteristic breach on its own must still raise"
    assert {row["metric"] for row in alerts} == {"csi"}
    assert {row["feature"] for row in alerts} == {"EXT_SOURCE_1"}


def test_alert_records_a_usable_audit_trail(runner):
    for seed in range(4):
        write_requests(runner, 200, shift=1.2, seed=seed)
    runner.drain(verbose=False)

    alert = runner.store.fetch_alerts()[0]
    assert alert["fired_at"]
    assert alert["threshold"] == pytest.approx(0.25)
    assert alert["value"] >= alert["threshold"]
    assert json.loads(alert["breach_window_ids"]) == [1, 2, 3]
    assert str(alert["feature"]) in alert["message"]


def test_two_quiet_windows_before_a_breach_delay_the_alert(runner):
    for seed in range(2):
        write_requests(runner, 200, shift=0.0, seed=seed)
    for seed in range(2, 4):
        write_requests(runner, 200, shift=1.2, seed=seed)
    runner.drain(verbose=False)

    assert runner.store.count_alerts() == 0, "two breaching windows must not be enough"

    write_requests(runner, 200, shift=1.2, seed=9)
    runner.drain(verbose=False)
    alerts = runner.store.fetch_alerts()
    assert len(alerts) == 2
    assert {row["window_id"] for row in alerts} == {5}


def test_export_writes_the_history_to_reports(runner, tmp_path):
    write_requests(runner, 400)
    runner.drain(verbose=False)
    counts = runner.export_reports()
    assert counts["windows"] == 2
    assert counts["metric_rows"] > 0
    assert (runner.config.path("reports") / "drift_metrics.csv").exists()
