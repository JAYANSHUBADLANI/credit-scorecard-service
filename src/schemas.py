"""Request and response contracts for the scoring API.

The validation here is deliberately strict, and the reason is a credit specific one. A
scorecard maps every input into a bin, and every bin has a weight of evidence, including the
missing bin. That means a malformed input never fails loudly on its own: a string where a
number belongs, or an age of 4000, would quietly land in some bin and come back as a
confident looking score. The model has no way to signal that it was asked something absurd.
So the boundary has to.

Three rules follow from that:

1. `strict=True`, so a numeric field will not silently accept "0.35". A string arriving where
   a float belongs means the caller has a bug, and returning a score would hide it.
2. `extra="forbid"`, so a misspelled field name is rejected rather than ignored. Silently
   dropping `ext_source_2` and scoring the applicant with it missing is the worst case: a
   plausible score computed from a characteristic the caller believed they had sent.
3. Explicit ranges on every field, set from the observed range of the training data with room
   to spare, so out of range values are refused rather than clipped into an end bin.

Missing values are a separate matter and are allowed where the data genuinely has them. The
external bureau scores are absent for a large share of real applications, and the card has a
fitted missing bin for exactly that case. Absent is a legitimate value here. Malformed is not.
"""

from __future__ import annotations

from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .features import DAYS_EMPLOYED_SENTINEL

# There is no gender field and no marital status field. Neither is merely unused by the model:
# the request contract does not accept them, so a caller cannot send one and a later refit
# cannot quietly pick one up. `DAYS_BIRTH` is different and is still accepted, because an
# applicant has to be of age to contract and the bounds below are what check it. It is
# validated and never scored. See README, "Characteristics deliberately excluded".
EducationLiteral = Literal[
    "Academic degree",
    "Higher education",
    "Incomplete higher",
    "Lower secondary",
    "Secondary / secondary special",
]
IncomeTypeLiteral = Literal[
    "Businessman",
    "Commercial associate",
    "Maternity leave",
    "Pensioner",
    "State servant",
    "Student",
    "Unemployed",
    "Working",
]
ContractTypeLiteral = Literal["Cash loans", "Revolving loans"]

# Checked against the fitted artifact at startup so the published contract and the model
# cannot drift apart across a refit. See `api.verify_contract`.
CATEGORICAL_LEVELS: Dict[str, List[str]] = {
    "NAME_EDUCATION_TYPE": list(EducationLiteral.__args__),
    "NAME_INCOME_TYPE": list(IncomeTypeLiteral.__args__),
    "NAME_CONTRACT_TYPE": list(ContractTypeLiteral.__args__),
}

# Capacity to contract, not a model input. No characteristic is derived from DAYS_BIRTH.
MIN_DAYS_BIRTH = -36525.0   # 100 years old
MAX_DAYS_BIRTH = -6570.0    # 18 years old


class ScoreRequest(BaseModel):
    """One application presented for scoring, in the raw field names of the source data."""

    model_config = ConfigDict(strict=True, extra="forbid", populate_by_name=True)

    EXT_SOURCE_1: Optional[float] = Field(
        default=None, ge=0.0, le=1.0, description="External bureau score 1, absent for many applicants"
    )
    EXT_SOURCE_2: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    EXT_SOURCE_3: Optional[float] = Field(default=None, ge=0.0, le=1.0)

    DAYS_BIRTH: float = Field(
        ge=MIN_DAYS_BIRTH,
        le=MAX_DAYS_BIRTH,
        description="Days before application that the applicant was born, so always negative",
    )
    DAYS_EMPLOYED: float = Field(
        le=float(DAYS_EMPLOYED_SENTINEL),
        ge=-25000.0,
        description=(
            "Days before application that employment started, negative. "
            f"The value {DAYS_EMPLOYED_SENTINEL} is the source data sentinel for no "
            "employment record and is accepted, then treated as missing."
        ),
    )
    DAYS_ID_PUBLISH: float = Field(ge=-10000.0, le=0.0)
    # Absent for one applicant in the 307,511 row source. The card has a fitted missing bin
    # for it, so accepting the absence is more faithful than refusing a real application.
    DAYS_LAST_PHONE_CHANGE: Optional[float] = Field(default=None, ge=-10000.0, le=0.0)

    AMT_INCOME_TOTAL: float = Field(gt=0.0, le=1e9)
    AMT_CREDIT: float = Field(gt=0.0, le=1e8)
    AMT_ANNUITY: Optional[float] = Field(default=None, gt=0.0, le=1e7)
    AMT_GOODS_PRICE: Optional[float] = Field(default=None, gt=0.0, le=1e8)
    REGION_POPULATION_RELATIVE: float = Field(ge=0.0, le=1.0)

    NAME_EDUCATION_TYPE: EducationLiteral
    NAME_INCOME_TYPE: IncomeTypeLiteral
    NAME_CONTRACT_TYPE: ContractTypeLiteral

    @model_validator(mode="after")
    def check_employment_sentinel(self) -> "ScoreRequest":
        """Reject positive employment days other than the documented sentinel.

        Without this, a sign error in a caller turns 3000 days of employment into a value that
        bins as though it were the pensioner sentinel.
        """
        if 0 < self.DAYS_EMPLOYED < DAYS_EMPLOYED_SENTINEL:
            raise ValueError(
                "DAYS_EMPLOYED must be negative days before application, or exactly "
                f"{DAYS_EMPLOYED_SENTINEL} for no employment record"
            )
        return self

    @model_validator(mode="after")
    def check_annuity_against_credit(self) -> "ScoreRequest":
        """The annual instalment cannot exceed the amount advanced.

        This holds for every one of the 307,511 rows in the source data, so a violation is a
        caller error rather than an unusual applicant.
        """
        if self.AMT_ANNUITY is not None and self.AMT_ANNUITY > self.AMT_CREDIT:
            raise ValueError("AMT_ANNUITY cannot exceed AMT_CREDIT")
        return self


class ReasonCode(BaseModel):
    """One principal reason for an adverse action, with the points behind it.

    The arithmetic is published alongside the statement on purpose. A reason an applicant
    cannot check is a reason they cannot contest, and a reviewer handling a complaint needs to
    see that this characteristic cost this applicant these points against this reference.
    """

    rank: int = Field(ge=1, description="1 is the most costly characteristic")
    code: str = Field(description="Stable across refits, see src/reasons.STATEMENTS")
    feature: str
    statement: str = Field(description="The applicant facing reason")
    points: float = Field(description="Points this applicant earned on the characteristic")
    reference_points: float = Field(description="Points the reference profile earns on it")
    shortfall: float = Field(description="reference_points - points, what the reason ranks on")


class ScoreResponse(BaseModel):
    """The scoring decision, plus the identifiers a reviewer needs to trace it."""

    request_id: str
    score: float = Field(description="Points, higher is better, 20 points per doubling of odds")
    probability: float = Field(ge=0.0, le=1.0, description="Modelled probability of default")
    band: Literal["decline", "refer", "approve"]
    model_version: str
    scored_at: str
    # Empty for any band that is not an adverse action, which is every band but `decline`.
    # Regulation B attaches the disclosure to the decision to decline, and returning reasons on
    # an approval would invite them to be presented as though a decision had gone against the
    # applicant. See src/reasons.py.
    reason_codes: List[ReasonCode] = Field(
        default_factory=list,
        description="Principal reasons for the decision, most costly first. Declines only.",
    )


class HealthResponse(BaseModel):
    status: Literal["ok"]
    model_version: str
    trained_at: str
    features: int
    requests_scored: int


class ErrorDetail(BaseModel):
    field: str
    message: str


class ErrorResponse(BaseModel):
    """A rejection the caller can act on: which field, and what was wrong with it."""

    error: str
    detail: List[ErrorDetail]
