"""FastAPI scoring service.

The endpoint does three things per request: validate, score, and append to the scoring log.
The logging is not incidental. It is the reason the monitoring layer has anything to look at,
and it happens on the serving path rather than in a batch job afterwards, because a
monitoring system that depends on someone remembering to export the scores is a monitoring
system that stops working in month three.

Validation failures return 422 with the offending field named, and nothing is written to the
scoring log, so a rejected request never contaminates the drift baseline. That matters: if
malformed requests were coerced and scored, they would show up later as a characteristic
shift and send someone hunting for a population change that was really a caller bug.
"""

from __future__ import annotations

import time
from contextlib import asynccontextmanager
from typing import Any, Dict, List

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from .config import Config, load_config
from .schemas import (
    CATEGORICAL_LEVELS,
    ErrorDetail,
    ErrorResponse,
    HealthResponse,
    ScoreRequest,
    ScoreResponse,
)
from .scoring import ScoringService
from .store import Store


class ServiceState:
    """Objects built once at startup and shared by every request."""

    def __init__(self) -> None:
        self.config: Config | None = None
        self.scoring: ScoringService | None = None
        self.store: Store | None = None

    def load(self, config: Config | None = None) -> None:
        self.config = config or load_config()
        self.scoring = ScoringService.from_path(self.config.path(self.config.service.model_path))
        self.store = Store(self.config.path(self.config.service.db_path))
        verify_contract(self.scoring)


state = ServiceState()


def verify_contract(scoring: ScoringService) -> None:
    """Fail startup if the published request contract and the fitted model disagree.

    A refit that introduces a new category level, or drops one, silently changes what the
    service should accept. Catching that at boot is far cheaper than discovering it when
    every application of some new employment type starts landing in the catch all bin.
    """
    fitted = scoring.artifact.categorical_levels
    problems: List[str] = []
    for field, declared in CATEGORICAL_LEVELS.items():
        actual = fitted.get(field)
        if actual is None:
            problems.append(f"{field}: declared in the API contract but not fitted in the model")
            continue
        if sorted(actual) != sorted(declared):
            problems.append(
                f"{field}: model fitted on {sorted(actual)} but the API contract declares "
                f"{sorted(declared)}"
            )
    if problems:
        raise RuntimeError(
            "request contract does not match the fitted artifact:\n  " + "\n  ".join(problems)
        )


@asynccontextmanager
async def lifespan(app: FastAPI):
    state.load()
    yield


app = FastAPI(
    title="Credit Scorecard Scoring Service",
    description=(
        "Serves an application credit scorecard and logs every scored request so that "
        "population and characteristic stability can be monitored continuously."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


@app.exception_handler(RequestValidationError)
async def validation_error_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    """Turn pydantic's nested errors into a flat, readable list naming each bad field."""
    details = []
    for error in exc.errors():
        location = [str(part) for part in error.get("loc", []) if part != "body"]
        details.append(
            ErrorDetail(
                field=".".join(location) if location else "body",
                message=error.get("msg", "invalid value"),
            )
        )
    payload = ErrorResponse(error="request validation failed", detail=details)
    return JSONResponse(status_code=422, content=payload.model_dump())


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    assert state.scoring is not None and state.store is not None
    return HealthResponse(
        status="ok",
        model_version=state.scoring.model_version,
        trained_at=state.scoring.artifact.trained_at,
        features=len(state.scoring.model_features),
        requests_scored=state.store.count_scores(),
    )


@app.get("/model")
def model_metadata() -> Dict[str, Any]:
    """What is actually deployed, which is the first question in any model review."""
    assert state.scoring is not None
    artifact = state.scoring.artifact
    return {
        "model_version": artifact.model_version,
        "trained_at": artifact.trained_at,
        "training_rows": artifact.training_rows,
        "training_bad_rate": round(artifact.training_bad_rate, 5),
        "model_features": state.scoring.model_features,
        "monitored_features": state.scoring.monitored_features,
        "band_cutoffs": {
            "decline_below": round(artifact.bands.decline_below, 2),
            "refer_below": round(artifact.bands.refer_below, 2),
        },
        "scaling": {
            "base_score": artifact.scorecard.scaling.base_score,
            "base_odds": artifact.scorecard.scaling.base_odds,
            "pdo": artifact.scorecard.scaling.pdo,
        },
        "accepted_categorical_levels": CATEGORICAL_LEVELS,
    }


@app.post(
    "/score",
    response_model=ScoreResponse,
    responses={422: {"model": ErrorResponse, "description": "The request failed validation"}},
)
def score(request: ScoreRequest) -> ScoreResponse:
    assert state.scoring is not None and state.store is not None

    started = time.perf_counter()
    result = state.scoring.score_payload(request.model_dump())
    latency_ms = (time.perf_counter() - started) * 1000.0

    state.store.log_score(
        request_id=result.request_id,
        model_version=result.model_version,
        source="api",
        score=result.score,
        probability=result.probability,
        band=result.band,
        bin_indices=result.bin_indices,
        latency_ms=latency_ms,
        scored_at=result.scored_at,
    )

    return ScoreResponse(
        request_id=result.request_id,
        score=round(result.score, 2),
        probability=round(result.probability, 6),
        band=result.band,
        model_version=result.model_version,
        scored_at=result.scored_at,
    )
