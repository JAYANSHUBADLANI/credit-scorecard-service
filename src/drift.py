"""Stability indices computed on a rolling window against the training time reference.

Three signals are produced per window, and they answer three different questions.

`psi_score` is the population stability index on the score distribution. It answers "has the
population the model is seeing changed". The reference is the training score cut into deciles,
so a stable population puts about ten percent in each bin and any departure is directly
readable.

`psi_band` is the same calculation over the three decision bands. It answers "has what the
model decides changed", which is the version a business owner cares about, because a shift
here is a shift in approval rate. It is coarser than `psi_score` on purpose.

`csi` per characteristic answers "which input moved". This is the attribution step. A score
PSI that fires on its own tells you something changed, and the characteristic indices tell you
where, which is the difference between an alert someone can act on and an alert someone
ignores.

One thing deliberately not computed: the population stability index on predicted probability.
Score is a monotone transform of log odds, so binning by score decile and by probability
decile assign identical records to identical bins, and the two numbers would be the same
quantity reported twice. Counting it as a separate signal would inflate the metric list
without adding information.

The thresholds are the conventional scorecard ones, 0.10 and 0.25. They are a stated
convention, not a calibration against observed incidents on this portfolio. See
docs/business_case.md.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

EPSILON = 1e-4

STATUS_OK = "ok"
STATUS_WARN = "warn"
STATUS_ALERT = "alert"

METRIC_PSI_SCORE = "psi_score"
METRIC_PSI_BAND = "psi_band"
METRIC_CSI = "csi"

BAND_ORDER = ["decline", "refer", "approve"]


def stability_index(
    actual_proportions: Sequence[float],
    reference_proportions: Sequence[float],
    epsilon: float = EPSILON,
) -> float:
    """The population stability index between two discrete distributions.

    Both sides are floored at `epsilon` before the log. A bin that is empty in one of the two
    distributions is a real and common case, an unused category or a score range no recent
    applicant reached, and without the floor it would return an infinity that then propagates
    into every downstream comparison. Flooring caps the contribution of an empty bin at a
    large but finite number, which is the behaviour wanted: loud, but still a number.
    """
    actual = np.asarray(actual_proportions, dtype="float64")
    reference = np.asarray(reference_proportions, dtype="float64")
    if actual.shape != reference.shape:
        raise ValueError(
            f"distributions must have the same number of bins, got {actual.shape} and "
            f"{reference.shape}"
        )
    if actual.size == 0:
        return 0.0

    actual = np.maximum(actual, epsilon)
    reference = np.maximum(reference, epsilon)
    return float(np.sum((actual - reference) * np.log(actual / reference)))


def proportions_from_counts(counts: Sequence[float]) -> np.ndarray:
    counts = np.asarray(counts, dtype="float64")
    total = counts.sum()
    if total <= 0:
        return np.zeros_like(counts)
    return counts / total


def bin_scores(scores: Sequence[float], edges: np.ndarray) -> np.ndarray:
    """Assign scores to the fixed training reference bins."""
    scores = np.asarray(scores, dtype="float64")
    if edges.size == 0:
        return np.zeros(len(scores), dtype="int64")
    return np.searchsorted(edges, scores, side="right")


def counts_from_indices(indices: Sequence[int], n_bins: int) -> np.ndarray:
    return np.bincount(np.asarray(indices, dtype="int64"), minlength=n_bins).astype("float64")[
        :n_bins
    ]


def classify(value: float, warn: float, alert: float) -> str:
    if value >= alert:
        return STATUS_ALERT
    if value >= warn:
        return STATUS_WARN
    return STATUS_OK


@dataclass
class MetricResult:
    metric: str
    feature: str
    value: float
    warn_threshold: float
    alert_threshold: float
    status: str

    def as_row(self) -> Dict[str, object]:
        return {
            "metric": self.metric,
            "feature": self.feature,
            "value": self.value,
            "warn_threshold": self.warn_threshold,
            "alert_threshold": self.alert_threshold,
            "status": self.status,
        }


@dataclass
class WindowSummary:
    """Descriptive statistics kept alongside the indices.

    These do not raise alerts. They exist so that when one does fire, the window it fired on
    can be described without re-reading the raw log.
    """

    n_records: int
    mean_score: float
    mean_probability: float
    approval_rate: float
    decline_rate: float
    band_counts: Dict[str, int] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, object]:
        return {
            "n_records": self.n_records,
            "mean_score": round(self.mean_score, 3),
            "mean_probability": round(self.mean_probability, 6),
            "approval_rate": round(self.approval_rate, 5),
            "decline_rate": round(self.decline_rate, 5),
            "band_counts": self.band_counts,
        }


@dataclass
class DriftThresholds:
    psi_warn: float
    psi_alert: float
    csi_warn: float
    csi_alert: float
    prediction_psi_warn: float
    prediction_psi_alert: float


class DriftMonitor:
    """Computes every stability index for one window of scored requests."""

    def __init__(
        self,
        reference_score_edges: np.ndarray,
        reference_score_proportions: np.ndarray,
        reference_band_proportions: np.ndarray,
        reference_bin_proportions: Dict[str, np.ndarray],
        thresholds: DriftThresholds,
        features: Optional[List[str]] = None,
    ):
        self.reference_score_edges = np.asarray(reference_score_edges, dtype="float64")
        self.reference_score_proportions = np.asarray(reference_score_proportions, dtype="float64")
        self.reference_band_proportions = np.asarray(reference_band_proportions, dtype="float64")
        self.reference_bin_proportions = {
            name: np.asarray(values, dtype="float64")
            for name, values in reference_bin_proportions.items()
        }
        self.thresholds = thresholds
        self.features = features or sorted(self.reference_bin_proportions)

    @classmethod
    def from_artifact(cls, artifact, thresholds: DriftThresholds, features=None) -> "DriftMonitor":
        reference_bins = {
            name: binning.reference_proportions
            for name, binning in artifact.transformer.bins.items()
        }
        return cls(
            reference_score_edges=artifact.reference_score_edges,
            reference_score_proportions=artifact.reference_score_proportions,
            reference_band_proportions=artifact.reference_band_proportions,
            reference_bin_proportions=reference_bins,
            thresholds=thresholds,
            features=features or list(artifact.scorecard.features),
        )

    def summarise(
        self, scores: Sequence[float], probabilities: Sequence[float], bands: Sequence[str]
    ) -> WindowSummary:
        scores = np.asarray(scores, dtype="float64")
        probabilities = np.asarray(probabilities, dtype="float64")
        band_array = np.asarray(bands, dtype=object)
        counts = {band: int((band_array == band).sum()) for band in BAND_ORDER}
        n = len(scores)
        return WindowSummary(
            n_records=n,
            mean_score=float(scores.mean()) if n else 0.0,
            mean_probability=float(probabilities.mean()) if n else 0.0,
            approval_rate=counts["approve"] / n if n else 0.0,
            decline_rate=counts["decline"] / n if n else 0.0,
            band_counts=counts,
        )

    def evaluate(
        self,
        scores: Sequence[float],
        bands: Sequence[str],
        bin_indices: Dict[str, Sequence[int]],
    ) -> List[MetricResult]:
        """Every index for one window, in the order they are worth reading."""
        results: List[MetricResult] = []

        score_bins = bin_scores(scores, self.reference_score_edges)
        score_counts = counts_from_indices(score_bins, len(self.reference_score_proportions))
        psi_score = stability_index(
            proportions_from_counts(score_counts), self.reference_score_proportions
        )
        results.append(
            MetricResult(
                metric=METRIC_PSI_SCORE,
                feature="score",
                value=psi_score,
                warn_threshold=self.thresholds.psi_warn,
                alert_threshold=self.thresholds.psi_alert,
                status=classify(psi_score, self.thresholds.psi_warn, self.thresholds.psi_alert),
            )
        )

        band_array = np.asarray(bands, dtype=object)
        band_counts = np.array(
            [float((band_array == band).sum()) for band in BAND_ORDER], dtype="float64"
        )
        psi_band = stability_index(
            proportions_from_counts(band_counts), self.reference_band_proportions
        )
        results.append(
            MetricResult(
                metric=METRIC_PSI_BAND,
                feature="band",
                value=psi_band,
                warn_threshold=self.thresholds.prediction_psi_warn,
                alert_threshold=self.thresholds.prediction_psi_alert,
                status=classify(
                    psi_band,
                    self.thresholds.prediction_psi_warn,
                    self.thresholds.prediction_psi_alert,
                ),
            )
        )

        for feature in self.features:
            reference = self.reference_bin_proportions.get(feature)
            observed = bin_indices.get(feature)
            if reference is None or observed is None:
                continue
            counts = counts_from_indices(observed, len(reference))
            csi = stability_index(proportions_from_counts(counts), reference)
            results.append(
                MetricResult(
                    metric=METRIC_CSI,
                    feature=feature,
                    value=csi,
                    warn_threshold=self.thresholds.csi_warn,
                    alert_threshold=self.thresholds.csi_alert,
                    status=classify(csi, self.thresholds.csi_warn, self.thresholds.csi_alert),
                )
            )

        return results


def bin_indices_from_rows(rows, features: List[str]) -> Dict[str, List[int]]:
    """Pull the stored per characteristic bin indices out of scoring log rows."""
    collected: Dict[str, List[int]] = {feature: [] for feature in features}
    for row in rows:
        payload = json.loads(row["bin_indices"])
        for feature in features:
            if feature in payload:
                collected[feature].append(int(payload[feature]))
    return {feature: values for feature, values in collected.items() if values}
