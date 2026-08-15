"""Tests for weight of evidence binning.

Beyond the usual correctness checks, two properties are load bearing for the monitoring layer
and are tested explicitly here: bin assignment and weight of evidence must agree, since the
characteristic index is computed from the former and the score from the latter, and an unseen
category must land somewhere finite rather than raising at scoring time.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.binning import (
    MISSING_LABEL,
    OTHER_LABEL,
    BinningConfig,
    WOETransformer,
    fit_categorical,
    fit_numeric,
)

CONFIG = BinningConfig(max_prebins=10, min_bin_fraction=0.05, min_bin_bads=5)


@pytest.fixture
def monotone_data():
    """A clean signal: the probability of bad falls as the characteristic rises."""
    rng = np.random.default_rng(7)
    values = rng.uniform(0, 1, 4000)
    target = (rng.uniform(0, 1, 4000) < (0.6 - 0.5 * values)).astype("int64")
    return values, target


# Numeric binning ---------------------------------------------------------------------

def test_bad_rate_is_monotonic_across_bins(monotone_data):
    values, target = monotone_data
    binning = fit_numeric(values, target, "x", CONFIG)
    observed = binning.table[binning.table["bin"] != MISSING_LABEL]
    rates = observed["bad_rate"].to_numpy()
    differences = np.diff(rates)
    assert np.all(differences <= 1e-9) or np.all(differences >= -1e-9), rates


def test_information_value_detects_signal_and_its_absence(monotone_data):
    values, target = monotone_data
    signal = fit_numeric(values, target, "x", CONFIG)

    rng = np.random.default_rng(11)
    noise = rng.uniform(0, 1, 4000)
    noise_binning = fit_numeric(noise, target, "noise", CONFIG)

    assert signal.iv > 0.1
    assert noise_binning.iv < signal.iv


def test_every_bin_meets_the_minimum_population_share(monotone_data):
    values, target = monotone_data
    binning = fit_numeric(values, target, "x", CONFIG)
    observed = binning.table[binning.table["bin"] != MISSING_LABEL]
    assert (observed["count"] >= CONFIG.min_bin_fraction * len(values) * 0.999).all()


def test_missing_values_get_their_own_bin_with_their_own_weight(monotone_data):
    values, target = monotone_data
    values = values.copy()
    values[:400] = np.nan
    target = target.copy()
    target[:400] = 1  # missing is strongly bad here, so it must not be imputed away

    binning = fit_numeric(values, target, "x", CONFIG)
    missing_rows = binning.table[binning.table["bin"] == MISSING_LABEL]
    assert len(missing_rows) == 1
    assert int(missing_rows.iloc[0]["count"]) == 400
    assert binning.missing_woe != 0.0
    assert binning.transform([np.nan])[0] == pytest.approx(binning.missing_woe)


def test_missing_bin_exists_even_when_training_had_none(monotone_data):
    """Bin indices must stay stable between training and serving."""
    values, target = monotone_data
    binning = fit_numeric(values, target, "x", CONFIG)
    assert binning.bin_labels[-1] == MISSING_LABEL
    assert binning.missing_index == len(binning.woe)
    assert binning.bin_index([np.nan])[0] == binning.missing_index


def test_reference_proportions_form_a_distribution(monotone_data):
    values, target = monotone_data
    binning = fit_numeric(values, target, "x", CONFIG)
    assert binning.reference_proportions.sum() == pytest.approx(1.0)
    assert len(binning.reference_proportions) == binning.n_bins


def test_values_beyond_the_training_range_fall_in_the_end_bins(monotone_data):
    values, target = monotone_data
    binning = fit_numeric(values, target, "x", CONFIG)
    assert binning.bin_index([-500.0])[0] == 0
    assert binning.bin_index([500.0])[0] == len(binning.woe) - 1


def test_constant_characteristic_carries_no_information():
    """One value for everybody separates nobody, so the information value must be negligible.

    Not identically zero: the smoothing constant leaves a residue of about 1e-4 on a sample
    this small, which is orders of magnitude below any selection floor. What matters is that
    it does not crash and does not look like signal.
    """
    values = np.full(500, 3.0)
    target = np.zeros(500, dtype="int64")
    target[:50] = 1
    binning = fit_numeric(values, target, "flat", CONFIG)
    assert binning.iv < 1e-3
    assert np.isfinite(binning.transform([3.0])[0])


# The property the monitoring layer depends on -------------------------------------------

def test_bin_assignment_and_weight_of_evidence_agree(monotone_data):
    values, target = monotone_data
    binning = fit_numeric(values, target, "x", CONFIG)

    probe = np.array([0.05, 0.25, 0.5, 0.75, 0.95, np.nan])
    lookup = np.append(binning.woe, binning.missing_woe)
    assert np.allclose(binning.transform(probe), lookup[binning.bin_index(probe)])


# Categorical binning ---------------------------------------------------------------------

@pytest.fixture
def categorical_data():
    levels = ["common_a"] * 2000 + ["common_b"] * 1500 + ["rare"] * 10
    rng = np.random.default_rng(3)
    target = np.concatenate([
        (rng.uniform(size=2000) < 0.05).astype("int64"),
        (rng.uniform(size=1500) < 0.20).astype("int64"),
        (rng.uniform(size=10) < 0.50).astype("int64"),
    ])
    return pd.Series(levels), target


def test_frequent_levels_keep_their_own_bin(categorical_data):
    values, target = categorical_data
    binning = fit_categorical(values, target, "cat", BinningConfig(min_categorical_fraction=0.01))
    assert "common_a" in binning.level_to_index
    assert "common_b" in binning.level_to_index
    assert binning.level_to_index["common_a"] != binning.level_to_index["common_b"]


def test_rare_level_is_grouped_rather_than_given_a_bin_of_its_own(categorical_data):
    """A level below the population floor is not indexed in its own right, it falls back to
    the catch all through the same path an unseen level takes."""
    values, target = categorical_data
    binning = fit_categorical(values, target, "cat", BinningConfig(min_categorical_fraction=0.01))
    assert "rare" not in binning.level_to_index
    assert binning.bin_index(["rare"])[0] == binning.level_to_index[OTHER_LABEL]


def test_unseen_level_lands_in_the_catch_all_rather_than_raising(categorical_data):
    """The API rejects unknown levels, but the transformer must not be the thing that breaks."""
    values, target = categorical_data
    binning = fit_categorical(values, target, "cat", BinningConfig(min_categorical_fraction=0.01))
    index = binning.bin_index(["a_level_never_seen"])[0]
    assert index == binning.level_to_index[OTHER_LABEL]
    assert np.isfinite(binning.transform(["a_level_never_seen"])[0])


def test_categorical_weight_of_evidence_orders_by_risk(categorical_data):
    values, target = categorical_data
    binning = fit_categorical(values, target, "cat", BinningConfig(min_categorical_fraction=0.01))
    woe = {level: binning.transform([level])[0] for level in ["common_a", "common_b"]}
    # Higher weight of evidence means a higher share of bads under this sign convention.
    assert woe["common_b"] > woe["common_a"]


def test_categorical_missing_is_separate_from_the_catch_all(categorical_data):
    values, target = categorical_data
    values = values.copy()
    values.iloc[:100] = None
    binning = fit_categorical(values, target, "cat", BinningConfig(min_categorical_fraction=0.01))
    assert binning.bin_index([None])[0] == binning.missing_index
    assert binning.bin_index([None])[0] != binning.level_to_index[OTHER_LABEL]


# The transformer -------------------------------------------------------------------------

def test_transformer_fits_both_kinds_and_round_trips(monotone_data, categorical_data):
    values, target = monotone_data
    levels, _ = categorical_data
    frame = pd.DataFrame({
        "num": values,
        "cat": list(levels)[: len(values)] + ["common_a"] * max(0, len(values) - len(levels)),
    })
    frame = frame.iloc[: len(values)]

    transformer = WOETransformer(CONFIG).fit(frame, target, ["num"], ["cat"])
    woe = transformer.transform(frame)
    bins = transformer.bin_indices(frame)

    assert list(woe.columns) == ["num", "cat"]
    assert woe.shape == bins.shape
    assert np.isfinite(woe.to_numpy()).all()
    assert (bins.to_numpy() >= 0).all()

    iv_table = transformer.iv_table()
    assert set(iv_table["feature"]) == {"num", "cat"}
    assert (iv_table["iv"] >= 0).all()
