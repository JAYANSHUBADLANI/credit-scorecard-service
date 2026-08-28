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

import sys
import time
from contextlib import asynccontextmanager
from typing import Any, Dict, List

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from .binning import OTHER_LABEL
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


def loaded() -> tuple[ScoringService, Store]:
    """The objects startup built, or a clear error rather than a confusing one.

    These call sites used `assert`. Assertions are stripped under `python -O`, so on an
    optimised interpreter a startup that had not completed would not have produced this
    message at all, it would have produced `'NoneType' object has no attribute` from
    somewhere further down the call stack. A guard on request state belongs in an `if`.
    """
    if state.scoring is None or state.store is None:
        raise RuntimeError(
            "service state is not loaded, so startup did not complete. Check the logs for "
            "the artifact path and the contract check."
        )
    return state.scoring, state.store


def verify_contract(scoring: ScoringService) -> None:
    """Check the published request contract against the fitted artifact at startup.

    A refit that changes what the service should accept is the failure this guards against.
    The two directions of that disagreement are not symmetric, and the first version of this
    function was wrong to treat them as though they were.

    A level the model was fitted on that the contract does not declare is a hard failure. The
    API would answer 422 to an application the card was built to score, so real business is
    refused for as long as the mismatch stands, and nothing in the response says why.

    A level the contract declares that the fit cannot resolve is not a failure. It scores
    through the catch all bin, which is precisely what `fit_categorical` already does to every
    level below `min_categorical_fraction`, so no response changes. Refusing to boot over one
    would take the service down for a distinction the card cannot draw: on this data six of the
    declared levels share the catch all in a normal fit, two of them resting on two rows in
    215,000. Under a random split that turned a startup failure into a coin toss. It is
    worth a line on stderr, not an outage.
    """
    fitted = scoring.artifact.categorical_levels
    bins = scoring.artifact.transformer.bins

    problems: List[str] = []
    folded: List[str] = []

    for field, declared in CATEGORICAL_LEVELS.items():
        actual = fitted.get(field)
        if actual is None:
            problems.append(f"{field}: declared in the API contract but not fitted in the model")
            continue

        undeclared = sorted(set(actual) - set(declared))
        if undeclared:
            problems.append(
                f"{field}: the model was fitted on {undeclared}, which the API contract does "
                "not accept, so those applications would be refused rather than scored"
            )

        binning = bins.get(field)
        resolvable = set(binning.level_to_index) - {OTHER_LABEL} if binning else set()
        unresolved = sorted(set(declared) - resolvable)
        if unresolved:
            folded.append(f"{field}: {unresolved}")

    if problems:
        raise RuntimeError(
            "request contract does not match the fitted artifact:\n  " + "\n  ".join(problems)
        )

    if folded:
        print(
            "contract check: accepted levels with no bin of their own in this fit. They score "
            "through the catch all, which is a distinction the card does not draw:\n  "
            + "\n  ".join(folded),
            file=sys.stderr,
            flush=True,
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
    scoring, store = loaded()
    return HealthResponse(
        status="ok",
        model_version=scoring.model_version,
        trained_at=scoring.artifact.trained_at,
        features=len(scoring.model_features),
        requests_scored=store.count_scores(),
    )


@app.get("/model")
def model_metadata() -> Dict[str, Any]:
    """What is actually deployed, which is the first question in any model review."""
    scoring, _ = loaded()
    artifact = scoring.artifact
    return {
        "model_version": artifact.model_version,
        "trained_at": artifact.trained_at,
        "training_rows": artifact.training_rows,
        "training_bad_rate": round(artifact.training_bad_rate, 5),
        "model_features": scoring.model_features,
        "monitored_features": scoring.monitored_features,
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
    scoring, store = loaded()

    started = time.perf_counter()
    result = scoring.score_payload(request.model_dump())
    latency_ms = (time.perf_counter() - started) * 1000.0

    store.log_score(
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
