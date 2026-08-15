"""Tests for the stability index computation.

The index itself is checked against a hand computed value rather than against another
implementation, because "it matches what the code did last time" is not a test of a formula.
The rest covers the behaviours the monitoring layer depends on: an unchanged population reads
as zero, an empty bin stays finite, and the characteristic index actually points at the
characteristic that moved.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.drift import (
    EPSILON,
    METRIC_CSI,
    METRIC_PSI_BAND,
    METRIC_PSI_SCORE,
    STATUS_ALERT,
    STATUS_OK,
    STATUS_WARN,
    DriftMonitor,
    DriftThresholds,
    bin_scores,
    classify,
    counts_from_indices,
    proportions_from_counts,
    stability_index,
)

THRESHOLDS = DriftThresholds(
    psi_warn=0.10,
    psi_alert=0.25,
    csi_warn=0.10,
    csi_alert=0.25,
    prediction_psi_warn=0.10,
    prediction_psi_alert=0.25,
)


# The index -------------------------------------------------------------------------

def test_identical_distributions_give_zero():
    reference = [0.2, 0.3, 0.5]
    assert stability_index(reference, reference) == pytest.approx(0.0, abs=1e-12)


def test_matches_a_hand_computed_value():
    """Two bins, 50/50 expected against 60/40 actual.

    (0.6 - 0.5) * ln(0.6/0.5) + (0.4 - 0.5) * ln(0.4/0.5)
    """
    expected = (0.6 - 0.5) * math.log(0.6 / 0.5) + (0.4 - 0.5) * math.log(0.4 / 0.5)
    assert stability_index([0.6, 0.4], [0.5, 0.5]) == pytest.approx(expected, rel=1e-12)
    # 0.1 * ln(1.2) + (-0.1) * ln(0.8) = 0.0182322 + 0.0223144
    assert expected == pytest.approx(0.0405465, abs=1e-6)


def test_index_is_symmetric():
    a, b = [0.1, 0.4, 0.5], [0.3, 0.3, 0.4]
    assert stability_index(a, b) == pytest.approx(stability_index(b, a), rel=1e-12)


def test_index_grows_with_the_size_of_the_shift():
    reference = [0.25, 0.25, 0.25, 0.25]
    small = stability_index([0.30, 0.25, 0.25, 0.20], reference)
    large = stability_index([0.55, 0.20, 0.15, 0.10], reference)
    assert 0 < small < large


def test_empty_bin_stays_finite():
    """A bin nobody landed in is common and must not return an infinity."""
    value = stability_index([0.0, 0.5, 0.5], [0.33, 0.33, 0.34])
    assert math.isfinite(value)
    assert value > 0


def test_bin_absent_from_the_reference_is_also_finite():
    value = stability_index([0.4, 0.3, 0.3], [0.0, 0.5, 0.5])
    assert math.isfinite(value)


def test_flooring_uses_the_documented_epsilon():
    """Both sides floored at EPSILON, so an empty against empty bin contributes nothing."""
    assert stability_index([0.0, 1.0], [0.0, 1.0]) == pytest.approx(0.0, abs=1e-12)
    assert EPSILON > 0


def test_mismatched_bin_counts_raise():
    with pytest.raises(ValueError, match="same number of bins"):
        stability_index([0.5, 0.5], [0.3, 0.3, 0.4])


def test_empty_input_is_zero_not_an_error():
    assert stability_index([], []) == 0.0


# Helpers ---------------------------------------------------------------------------

def test_proportions_from_counts_handles_an_empty_window():
    assert proportions_from_counts([0, 0, 0]).tolist() == [0.0, 0.0, 0.0]
    assert proportions_from_counts([1, 3]).tolist() == [0.25, 0.75]


def test_counts_from_indices_pads_unused_bins():
    counts = counts_from_indices([0, 0, 2], n_bins=4)
    assert counts.tolist() == [2.0, 0.0, 1.0, 0.0]


def test_bin_scores_uses_right_closed_intervals():
    edges = np.array([500.0, 600.0])
    assert bin_scores([499.0, 500.0, 550.0, 600.0, 700.0], edges).tolist() == [0, 1, 1, 2, 2]


def test_classify_boundaries_are_inclusive():
    assert classify(0.09, 0.10, 0.25) == STATUS_OK
    assert classify(0.10, 0.10, 0.25) == STATUS_WARN
    assert classify(0.24, 0.10, 0.25) == STATUS_WARN
    assert classify(0.25, 0.10, 0.25) == STATUS_ALERT


# The monitor -----------------------------------------------------------------------

def build_monitor(features=("feature_a", "feature_b")):
    return DriftMonitor(
        reference_score_edges=np.array([500.0, 550.0, 600.0]),
        reference_score_proportions=np.array([0.25, 0.25, 0.25, 0.25]),
        reference_band_proportions=np.array([0.1, 0.2, 0.7]),
        reference_bin_proportions={name: np.array([0.5, 0.3, 0.2]) for name in features},
        thresholds=THRESHOLDS,
        features=list(features),
    )


def test_unchanged_population_reads_as_stable():
    monitor = build_monitor()
    scores = [480.0] * 25 + [520.0] * 25 + [570.0] * 25 + [650.0] * 25
    bands = ["decline"] * 10 + ["refer"] * 20 + ["approve"] * 70
    bins = {
        "feature_a": [0] * 50 + [1] * 30 + [2] * 20,
        "feature_b": [0] * 50 + [1] * 30 + [2] * 20,
    }
    results = monitor.evaluate(scores, bands, bins)
    for result in results:
        assert result.status == STATUS_OK, f"{result.feature} read as {result.status}"
        assert result.value < 0.01


def test_every_expected_metric_is_produced():
    monitor = build_monitor()
    results = monitor.evaluate([520.0] * 100, ["approve"] * 100, {"feature_a": [0] * 100})
    metrics = {(r.metric, r.feature) for r in results}
    assert (METRIC_PSI_SCORE, "score") in metrics
    assert (METRIC_PSI_BAND, "band") in metrics
    assert (METRIC_CSI, "feature_a") in metrics
    # A characteristic with no stored bins is skipped rather than scored on nothing.
    assert (METRIC_CSI, "feature_b") not in metrics


def test_characteristic_index_points_at_the_characteristic_that_moved():
    """The attribution property the whole design rests on."""
    monitor = build_monitor()
    scores = [480.0] * 25 + [520.0] * 25 + [570.0] * 25 + [650.0] * 25
    bands = ["decline"] * 10 + ["refer"] * 20 + ["approve"] * 70
    bins = {
        "feature_a": [0] * 95 + [1] * 3 + [2] * 2,   # shifted hard into the first bin
        "feature_b": [0] * 50 + [1] * 30 + [2] * 20,  # unchanged
    }
    results = {r.feature: r.value for r in monitor.evaluate(scores, bands, bins)}
    assert results["feature_a"] > results["feature_b"]
    assert results["feature_a"] > 0.25
    assert results["feature_b"] < 0.01


def test_score_shift_raises_the_population_index():
    monitor = build_monitor()
    stable = monitor.evaluate(
        [480.0] * 25 + [520.0] * 25 + [570.0] * 25 + [650.0] * 25,
        ["approve"] * 100,
        {},
    )
    shifted = monitor.evaluate([480.0] * 90 + [650.0] * 10, ["approve"] * 100, {})
    stable_psi = next(r.value for r in stable if r.metric == METRIC_PSI_SCORE)
    shifted_psi = next(r.value for r in shifted if r.metric == METRIC_PSI_SCORE)
    assert shifted_psi > stable_psi
    assert shifted_psi > 0.25


def test_band_index_reacts_to_an_approval_rate_shift():
    monitor = build_monitor()
    results = monitor.evaluate(
        [520.0] * 100, ["decline"] * 50 + ["refer"] * 30 + ["approve"] * 20, {}
    )
    band = next(r for r in results if r.metric == METRIC_PSI_BAND)
    assert band.status == STATUS_ALERT
    assert band.value > 0.25


def test_summary_reports_the_window_as_scored():
    monitor = build_monitor()
    summary = monitor.summarise(
        scores=[500.0, 600.0],
        probabilities=[0.2, 0.05],
        bands=["decline", "approve"],
    )
    assert summary.n_records == 2
    assert summary.mean_score == pytest.approx(550.0)
    assert summary.mean_probability == pytest.approx(0.125)
    assert summary.approval_rate == pytest.approx(0.5)
    assert summary.decline_rate == pytest.approx(0.5)
    assert summary.band_counts == {"decline": 1, "refer": 0, "approve": 1}
