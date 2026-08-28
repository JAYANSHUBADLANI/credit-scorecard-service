"""Tests for the raw field to feature transformation.

The property that matters most is the last one in this file: a record transformed on its own
produces exactly what it produces inside a batch. That is what makes the API and the training
job agree, and it is the failure this module exists to prevent.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features import (
    DAYS_EMPLOYED_SENTINEL,
    RAW_INPUTS,
    build_features,
    frame_from_payload,
)


def base_row() -> dict:
    return {
        "EXT_SOURCE_1": 0.5,
        "EXT_SOURCE_2": 0.6,
        "EXT_SOURCE_3": 0.4,
        "DAYS_BIRTH": -14610.0,      # exactly 40 years at 365.25 days
        "DAYS_EMPLOYED": -3652.5,    # exactly 10 years
        "DAYS_ID_PUBLISH": -3650.0,
        "DAYS_LAST_PHONE_CHANGE": -900.0,
        "AMT_INCOME_TOTAL": 100000.0,
        "AMT_CREDIT": 400000.0,
        "AMT_ANNUITY": 20000.0,
        "AMT_GOODS_PRICE": 360000.0,
        "REGION_POPULATION_RELATIVE": 0.02,
        "NAME_EDUCATION_TYPE": "Higher education",
        "NAME_INCOME_TYPE": "Working",
        "NAME_FAMILY_STATUS": "Married",
        "NAME_CONTRACT_TYPE": "Cash loans",
    }


def build_one(**overrides) -> pd.Series:
    row = {**base_row(), **overrides}
    return build_features(pd.DataFrame([row])).iloc[0]


# Derivations -------------------------------------------------------------------------

def test_ages_and_durations_are_positive_years():
    features = build_one()
    assert features["AGE_YEARS"] == pytest.approx(40.0)
    assert features["EMPLOYED_YEARS"] == pytest.approx(10.0)
    assert features["ID_PUBLISH_YEARS"] == pytest.approx(3650.0 / 365.25)


def test_ratios_are_computed_as_documented():
    features = build_one()
    assert features["CREDIT_INCOME_RATIO"] == pytest.approx(4.0)
    assert features["ANNUITY_INCOME_RATIO"] == pytest.approx(0.2)
    assert features["CREDIT_TERM"] == pytest.approx(0.05)
    assert features["GOODS_CREDIT_RATIO"] == pytest.approx(0.9)


def test_employment_sentinel_becomes_missing():
    """365243 is a marker for no employment record, not a thousand years of service."""
    features = build_one(DAYS_EMPLOYED=float(DAYS_EMPLOYED_SENTINEL))
    assert np.isnan(features["EMPLOYED_YEARS"])


def test_ordinary_employment_value_is_not_treated_as_the_sentinel():
    features = build_one(DAYS_EMPLOYED=-7305.0)
    assert features["EMPLOYED_YEARS"] == pytest.approx(20.0)


def test_missing_inputs_propagate_as_missing_not_zero():
    features = build_one(EXT_SOURCE_1=None, AMT_ANNUITY=None, AMT_GOODS_PRICE=None)
    assert np.isnan(features["EXT_SOURCE_1"])
    assert np.isnan(features["AMT_ANNUITY"])
    assert np.isnan(features["ANNUITY_INCOME_RATIO"])
    assert np.isnan(features["GOODS_CREDIT_RATIO"])
    # A characteristic that did not depend on the missing input is unaffected.
    assert features["CREDIT_INCOME_RATIO"] == pytest.approx(4.0)


def test_zero_denominator_gives_missing_rather_than_infinity():
    """An infinity would sail through binning into the top bin and score confidently."""
    features = build_one(AMT_INCOME_TOTAL=0.0)
    assert np.isnan(features["CREDIT_INCOME_RATIO"])
    assert not np.isinf(features["CREDIT_INCOME_RATIO"])


def test_categoricals_pass_through_unchanged():
    features = build_one()
    assert features["NAME_EDUCATION_TYPE"] == "Higher education"
    assert features["NAME_INCOME_TYPE"] == "Working"


def test_gender_is_not_a_raw_input_at_all():
    """Excluded at the boundary, not merely left out of the model.

    A prohibited basis that the service still ingests is one refit away from being scored on.
    Keeping it out of RAW_INPUTS means the feature builder would have to be edited too.
    """
    assert "CODE_GENDER" not in RAW_INPUTS
    features = build_features(pd.DataFrame([{**base_row(), "CODE_GENDER": "F"}]))
    assert "CODE_GENDER" not in features.columns


def test_missing_raw_column_raises_rather_than_producing_a_silent_null():
    frame = pd.DataFrame([{k: v for k, v in base_row().items() if k != "AMT_CREDIT"}])
    with pytest.raises(KeyError, match="AMT_CREDIT"):
        build_features(frame)


def test_payload_frame_carries_every_raw_input(api_payload_keys=RAW_INPUTS):
    frame = frame_from_payload(base_row())
    assert list(frame.columns) == list(api_payload_keys)
    assert len(frame) == 1


# The property that prevents training and serving skew -----------------------------------

def test_single_record_matches_the_same_record_inside_a_batch():
    rows = [
        base_row(),
        {**base_row(), "EXT_SOURCE_1": None, "AMT_INCOME_TOTAL": 250000.0},
        {**base_row(), "DAYS_EMPLOYED": float(DAYS_EMPLOYED_SENTINEL)},
        {**base_row(), "AMT_GOODS_PRICE": None, "NAME_FAMILY_STATUS": "Single / not married"},
    ]
    batch = build_features(pd.DataFrame(rows))

    for i, row in enumerate(rows):
        single = build_features(pd.DataFrame([row])).iloc[0]
        for column in batch.columns:
            expected = batch.iloc[i][column]
            actual = single[column]
            if isinstance(expected, float) and np.isnan(expected):
                assert np.isnan(actual), f"{column} differs on row {i}"
            else:
                assert actual == expected, f"{column} differs on row {i}"
