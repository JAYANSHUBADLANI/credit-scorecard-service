"""Logistic scorecard on weight of evidence features, with points scaling.

Carried over unchanged from the application scorecard this service accompanies. Scaling
follows the usual convention: a higher score means a better applicant, and the score falls by
exactly `pdo` points every time the odds of going bad double. Points are allocated back to
individual characteristics so the card reads as a table rather than a set of coefficients,
which is what a credit committee actually reviews.

The model itself is deliberately not the interesting part of this project. It is here so that
there is something real and defensible to serve and to monitor.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from .binning import WOETransformer

BAND_DECLINE = "decline"
BAND_REFER = "refer"
BAND_APPROVE = "approve"


@dataclass
class ScalingConfig:
    base_score: float = 600.0
    base_odds: float = 50.0
    pdo: float = 20.0

    @property
    def factor(self) -> float:
        return self.pdo / np.log(2)

    @property
    def offset(self) -> float:
        return self.base_score + self.factor * np.log(1.0 / self.base_odds)


@dataclass
class BandCutoffs:
    """Absolute score cutoffs resolved from training percentiles and then frozen."""

    decline_below: float
    refer_below: float

    def assign(self, score: float) -> str:
        if score < self.decline_below:
            return BAND_DECLINE
        if score < self.refer_below:
            return BAND_REFER
        return BAND_APPROVE

    def assign_many(self, scores: np.ndarray) -> np.ndarray:
        out = np.full(len(scores), BAND_APPROVE, dtype=object)
        out[scores < self.refer_below] = BAND_REFER
        out[scores < self.decline_below] = BAND_DECLINE
        return out


def select_features(
    transformer: WOETransformer,
    woe_frame: pd.DataFrame,
    min_iv: float = 0.02,
    max_correlation: float = 0.75,
) -> List[str]:
    """Keep characteristics above an information value floor, dropping the weaker of any
    correlated pair. Correlated weight of evidence columns give unstable coefficients and a
    card whose points do not add up the way a reviewer expects."""
    ranked = transformer.iv_table()
    candidates = [f for f in ranked.loc[ranked["iv"] >= min_iv, "feature"] if f in woe_frame]
    if not candidates:
        return []

    correlations = woe_frame[candidates].corr().abs()
    kept: List[str] = []
    for feature in candidates:
        if all(correlations.loc[feature, other] < max_correlation for other in kept):
            kept.append(feature)
    return kept


class Scorecard:
    """Weight of evidence logistic regression plus the points transformation."""

    def __init__(self, scaling: Optional[ScalingConfig] = None, C: float = 1.0):
        self.scaling = scaling or ScalingConfig()
        self.model = LogisticRegression(C=C, max_iter=1000, solver="lbfgs")
        self.features: List[str] = []

    def fit(self, woe_frame: pd.DataFrame, target: np.ndarray, features: List[str]) -> "Scorecard":
        self.features = features
        self.model.fit(woe_frame[features].to_numpy(), target)
        return self

    def log_odds(self, woe_frame: pd.DataFrame) -> np.ndarray:
        return self.model.decision_function(woe_frame[self.features].to_numpy())

    def predict_proba(self, woe_frame: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(woe_frame[self.features].to_numpy())[:, 1]

    def score(self, woe_frame: pd.DataFrame) -> np.ndarray:
        return self.scaling.offset - self.scaling.factor * self.log_odds(woe_frame)

    def coefficient_table(self, transformer: WOETransformer) -> pd.DataFrame:
        rows = []
        for coefficient, feature in zip(self.model.coef_[0], self.features):
            rows.append({
                "feature": feature,
                "coefficient": float(coefficient),
                "iv": transformer.bins[feature].iv,
                "bins": transformer.bins[feature].n_bins,
            })
        table = pd.DataFrame(rows).sort_values("iv", ascending=False).reset_index(drop=True)
        table.attrs["intercept"] = float(self.model.intercept_[0])
        return table

    def scorecard_table(self, transformer: WOETransformer) -> pd.DataFrame:
        """The card itself: every bin of every retained characteristic with its points."""
        n = len(self.features)
        intercept = float(self.model.intercept_[0])
        coefficients = dict(zip(self.features, self.model.coef_[0]))
        rows = []
        for feature in self.features:
            table = transformer.bins[feature].table.copy()
            table["points"] = (
                -self.scaling.factor * (coefficients[feature] * table["woe"] + intercept / n)
                + self.scaling.offset / n
            ).round(1)
            rows.append(table)
        return pd.concat(rows, ignore_index=True)


@dataclass
class ScorecardArtifact:
    """Everything the service needs to score a request and to monitor itself.

    Persisted as one object so that the model, the bins it was fitted with, the band cutoffs
    and the training time reference distributions can never be loaded out of step with each
    other. A monitoring baseline that came from a different fit than the served model would
    report drift that is really just a version mismatch.
    """

    transformer: WOETransformer
    scorecard: Scorecard
    bands: BandCutoffs
    numeric_features: List[str]
    categorical_features: List[str]
    categorical_levels: Dict[str, List[str]]
    reference_score_edges: np.ndarray
    reference_score_proportions: np.ndarray
    reference_band_proportions: np.ndarray
    reference_mean_score: float
    reference_mean_probability: float
    trained_at: str
    training_rows: int
    training_bad_rate: float
    model_version: str

    def score_frame(self, feature_frame: pd.DataFrame) -> pd.DataFrame:
        """Score a feature frame, returning probability, score and band."""
        woe_frame = self.transformer.transform(feature_frame)
        probability = self.scorecard.predict_proba(woe_frame)
        score = self.scorecard.score(woe_frame)
        return pd.DataFrame(
            {
                "probability": probability,
                "score": score,
                "band": self.bands.assign_many(score),
            },
            index=feature_frame.index,
        )
