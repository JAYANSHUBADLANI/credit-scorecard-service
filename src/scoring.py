"""The scoring path itself, kept separate from the web layer.

Everything here is transport agnostic, which is what makes the endpoint testable without a
running server and lets the batch replay reuse exactly the code the API runs. The bin indices
come back with the score because they are what the characteristic stability index is computed
from later, and deriving them at scoring time is the only way to be sure the monitoring layer
sees the same bins the model used.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd

from .features import build_features
from .reasons import AssignedReason, ReasonCodeAssigner
from .scorecard import ScorecardArtifact
from .store import utc_now


def load_artifact(path: Path | str) -> ScorecardArtifact:
    """Load the frozen artifact, or say plainly that training has not been run.

    Separate from `ScoringService.from_path` because the reason code assigner is built from the
    artifact and the service is built from both, so a caller configuring reasons needs the
    artifact before it has a service to ask for one.
    """
    resolved = Path(path)
    if not resolved.exists():
        raise FileNotFoundError(
            f"no scorecard artifact at {resolved}. Run `make train` before starting the service."
        )
    return joblib.load(resolved)


@dataclass
class ScoringResult:
    request_id: str
    score: float
    probability: float
    band: str
    bin_indices: Dict[str, int]
    model_version: str
    scored_at: str
    # Empty unless the band is an adverse action. See src/reasons.py: a reason code on an
    # approved application is not a reason for anything.
    reason_codes: List[AssignedReason] = field(default_factory=list)


class ScoringService:
    """Loads the frozen artifact once and scores requests against it."""

    def __init__(
        self, artifact: ScorecardArtifact, reasons: Optional[ReasonCodeAssigner] = None
    ):
        self.artifact = artifact
        # Built here rather than per request: the per bin points and the reference profile are
        # fixed by the artifact, so recomputing them per decline would be work done 40,000
        # times to get the same answer.
        self.reasons = reasons or ReasonCodeAssigner(artifact)

    @classmethod
    def from_path(
        cls, path: Path | str, reasons: Optional[ReasonCodeAssigner] = None
    ) -> "ScoringService":
        return cls(load_artifact(path), reasons=reasons)

    @property
    def model_version(self) -> str:
        return self.artifact.model_version

    @property
    def model_features(self) -> List[str]:
        return list(self.artifact.scorecard.features)

    @property
    def monitored_features(self) -> List[str]:
        return list(self.artifact.transformer.bins.keys())

    def score_payload(self, payload: Dict[str, Any], request_id: Optional[str] = None) -> ScoringResult:
        """Score one validated payload."""
        return self.score_batch([payload], [request_id] if request_id else None)[0]

    def score_batch(
        self, payloads: List[Dict[str, Any]], request_ids: Optional[List[str]] = None
    ) -> List[ScoringResult]:
        """Score many validated payloads through the same path a single request takes."""
        if not payloads:
            return []
        if request_ids is not None and len(request_ids) != len(payloads):
            raise ValueError(
                f"got {len(request_ids)} request ids for {len(payloads)} payloads. Scoring "
                "them anyway would attach the wrong id to a decision, or raise several rows "
                "further in with nothing to say which caller it belonged to."
            )

        raw = pd.DataFrame(payloads)
        features = build_features(raw)
        woe = self.artifact.transformer.transform(features)
        probability = self.artifact.scorecard.predict_proba(woe)
        score = self.artifact.scorecard.score(woe)
        band = self.artifact.bands.assign_many(score)
        bins = self.artifact.transformer.bin_indices(features)

        ids = request_ids or [str(uuid.uuid4()) for _ in payloads]
        scored_at = utc_now()
        # One dict per row rather than a cell lookup per characteristic. The old form indexed
        # the frame len(payloads) * len(columns) times, which on a batch replay is the whole
        # cost of scoring spent on pandas rather than on the model.
        indices = [
            {column: int(value) for column, value in row.items()}
            for row in bins.to_dict(orient="records")
        ]
        return [
            ScoringResult(
                request_id=ids[i],
                score=float(score[i]),
                probability=float(probability[i]),
                band=str(band[i]),
                bin_indices=indices[i],
                model_version=self.artifact.model_version,
                scored_at=scored_at,
                reason_codes=self.reasons.assign(indices[i], str(band[i])),
            )
            for i in range(len(payloads))
        ]

    def reference_proportions(self, feature: str) -> np.ndarray:
        return self.artifact.transformer.bins[feature].reference_proportions

    def n_bins(self, feature: str) -> int:
        return self.artifact.transformer.bins[feature].n_bins
