"""Raw application fields to model features.

This module is the single transformation path. Training calls it on the full frame and the
API calls it on a one row frame built from the request body, so a feature can never be
derived one way offline and another way online. Training and serving skew in derived ratios
is one of the commonest causes of a model that validates well and then behaves oddly in
production, and keeping one implementation is the cheapest possible defence against it.

`DAYS_EMPLOYED` carries a sentinel of 365243 in this dataset, which is roughly a thousand
years of employment and marks a pensioner with no employment record. Left alone it would sit
in the top bin of an otherwise sensible characteristic. It is mapped to missing, which is what
it means, and the missing bin then carries its own weight of evidence.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

DAYS_EMPLOYED_SENTINEL = 365243
DAYS_PER_YEAR = 365.25

RAW_NUMERIC_INPUTS: List[str] = [
    "EXT_SOURCE_1",
    "EXT_SOURCE_2",
    "EXT_SOURCE_3",
    "DAYS_BIRTH",
    "DAYS_EMPLOYED",
    "DAYS_ID_PUBLISH",
    "DAYS_LAST_PHONE_CHANGE",
    "AMT_INCOME_TOTAL",
    "AMT_CREDIT",
    "AMT_ANNUITY",
    "AMT_GOODS_PRICE",
    "REGION_POPULATION_RELATIVE",
]

# CODE_GENDER is not here on purpose. Gender is a prohibited basis for a credit decision
# under ECOA and Regulation B, so it is excluded at the input boundary rather than merely
# left out of the model: a field the service never accepts cannot be reintroduced by a
# later refit that widens the feature list. See README, "Characteristics deliberately
# excluded".
RAW_CATEGORICAL_INPUTS: List[str] = [
    "NAME_EDUCATION_TYPE",
    "NAME_INCOME_TYPE",
    "NAME_FAMILY_STATUS",
    "NAME_CONTRACT_TYPE",
]

RAW_INPUTS: List[str] = RAW_NUMERIC_INPUTS + RAW_CATEGORICAL_INPUTS


def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Divide, returning missing rather than an infinity when the denominator is zero."""
    denom = denominator.replace(0, np.nan)
    return numerator / denom


def build_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Derive the model feature frame from raw application fields."""
    missing = [column for column in RAW_INPUTS if column not in frame.columns]
    if missing:
        raise KeyError(f"missing raw input columns: {sorted(missing)}")

    out = pd.DataFrame(index=frame.index)

    for column in ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]:
        out[column] = pd.to_numeric(frame[column], errors="coerce")

    days_employed = pd.to_numeric(frame["DAYS_EMPLOYED"], errors="coerce")
    days_employed = days_employed.where(days_employed != DAYS_EMPLOYED_SENTINEL, np.nan)

    out["AGE_YEARS"] = -pd.to_numeric(frame["DAYS_BIRTH"], errors="coerce") / DAYS_PER_YEAR
    out["EMPLOYED_YEARS"] = -days_employed / DAYS_PER_YEAR
    out["ID_PUBLISH_YEARS"] = (
        -pd.to_numeric(frame["DAYS_ID_PUBLISH"], errors="coerce") / DAYS_PER_YEAR
    )
    out["PHONE_CHANGE_YEARS"] = (
        -pd.to_numeric(frame["DAYS_LAST_PHONE_CHANGE"], errors="coerce") / DAYS_PER_YEAR
    )

    income = pd.to_numeric(frame["AMT_INCOME_TOTAL"], errors="coerce")
    credit = pd.to_numeric(frame["AMT_CREDIT"], errors="coerce")
    annuity = pd.to_numeric(frame["AMT_ANNUITY"], errors="coerce")
    goods = pd.to_numeric(frame["AMT_GOODS_PRICE"], errors="coerce")

    out["AMT_INCOME_TOTAL"] = income
    out["AMT_CREDIT"] = credit
    out["AMT_ANNUITY"] = annuity
    out["REGION_POPULATION_RELATIVE"] = pd.to_numeric(
        frame["REGION_POPULATION_RELATIVE"], errors="coerce"
    )

    out["CREDIT_INCOME_RATIO"] = _safe_ratio(credit, income)
    out["ANNUITY_INCOME_RATIO"] = _safe_ratio(annuity, income)
    out["CREDIT_TERM"] = _safe_ratio(annuity, credit)
    out["GOODS_CREDIT_RATIO"] = _safe_ratio(goods, credit)

    for column in RAW_CATEGORICAL_INPUTS:
        values = frame[column]
        out[column] = values.where(values.notna(), None)

    return out


def frame_from_payload(payload: Dict[str, object]) -> pd.DataFrame:
    """Build the one row raw frame the API scores, from an already validated payload."""
    return pd.DataFrame([{column: payload.get(column) for column in RAW_INPUTS}])
