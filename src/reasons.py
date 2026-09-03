"""Adverse action reason codes for a declined application.

Regulation B requires that an applicant who is declined is told the specific principal reasons
for it. A points based weight of evidence scorecard can produce those almost for free, because
the score is already a sum of per characteristic points: the principal reasons are the
characteristics on which this applicant lost the most points against a reference profile. That
is the whole mechanism. What follows is mostly the decisions around it, because every one of
them changes what an applicant is told.

**The reference profile.** A shortfall only means something relative to something. Two bases
are conventional and both are implemented:

- `max_observed`. The reference is the highest scoring bin of that characteristic. A reason then
  reads "you scored below the best attainable value here".
- `population_mean`, the default. The reference is the training population's average points for
  that characteristic, from the same reference proportions the drift monitor uses. A reason then
  reads "you scored below a typical applicant here".

`max_observed` is the more common industry choice and was the default here first. It was changed
on evidence, not preference. Both are permissible under Regulation B, which is about comparison
to the best or to the average, but only one of them reliably names a *principal* reason for
*this* applicant:

- Under `max_observed`, a characteristic gets cited because its point range is wide, whether or
  not this applicant is unusual on it. Measured over the 9,010 held out declines, 1,417 of the
  36,040 reasons cited a characteristic the applicant sat at or **above** the population average
  on, putting at least one such reason on 15.3% of notices. A characteristic an applicant is
  above average on did nothing to distinguish them from the applicants who were approved, so
  calling it a principal reason for their decline is a stretch, even though "below the level
  required" is literally true of anyone who was declined.
- Under `population_mean` that cannot happen at all: a positive shortfall means below average by
  definition. It also spreads the citations across 12 characteristics rather than 7.

The cost of the change is that it rewrote the notice on 98.7% of declines, which is why it
wanted a measurement behind it rather than a preference. `scripts/adverse_action_audit.py`
reports both, so the choice stays checkable on the next card rather than inherited.

**The missing bin is excluded from the `max_observed` reference.** This is not a detail. A
characteristic that is almost never missing gets a smoothed weight of evidence for that bin
which is not a statement about anybody, and it can out score every observed bin. The card that
first exposed this carried `AGE_YEARS`, whose missing bin scored 53.93 against a best observed
bin of 42.46: including it would have put an 11 point age shortfall on every single applicant
and made age the leading reason code on the book, an artefact of a bin that a required field
made unreachable in the first place.

**Landing in the missing bin is a different reason from scoring badly.** "No external bureau
score on file" and "external bureau score below the level required" are both true things to say
and they are not the same thing, so each characteristic that can genuinely arrive missing
carries its own statement for that case. Sending the wrong one is a compliance defect, not a
wording preference.

**Reason codes are derived, not stored.** The scoring log already persists the bin indices, and
the bins plus the artifact determine the reasons exactly. Writing them to the log as well would
create a second copy that a later refit could put out of step with the first. `assign` is pure
given an artifact and a set of bin indices, so any decision in the log can be reconstructed
against the model version it was scored under.

**Prohibited bases are reported, never suppressed.** `STATEMENTS` marks the characteristics
that are prohibited bases under ECOA, and `verify_reason_coverage` reports any the card is
fitted on, at startup and in the audit. It does not drop them from the notice, because a reason
code that omits the characteristic which actually drove the decline is a false statement to the
applicant, and that is worse than an accurate statement about a characteristic that should not
be there. The fix belongs in the feature list, not here.

That is not hypothetical. Writing this module is what surfaced that the card was fitted on
`AGE_YEARS` and `NAME_FAMILY_STATUS`, and that its age treatment penalised applicants aged 62
or over by 6.46 points, which Regulation B does not permit. Both were removed from the feature
list and from the request contract, at a measured cost of 0.0003 of holdout AUC. The guard
stays because the next refit is the one to worry about. See README, "Characteristics
deliberately excluded", and `scripts/adverse_action_audit.py`.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .scorecard import BAND_DECLINE, ScorecardArtifact

BASIS_MAX_OBSERVED = "max_observed"
BASIS_POPULATION_MEAN = "population_mean"
VALID_BASES = (BASIS_MAX_OBSERVED, BASIS_POPULATION_MEAN)

ON_PROHIBITED_WARN = "warn"
ON_PROHIBITED_RAISE = "raise"

# The compliance list, and the single source of truth for it. Characteristic name to the basis
# it is prohibited on under ECOA. This is deliberately not a field on `ReasonStatement`: a
# characteristic can be retained by a fit without anybody having written an applicant facing
# statement for it, and that is exactly the case the fit time check has to catch. Two lists
# would eventually disagree, and the one that mattered would be whichever was consulted.
PROHIBITED_BASES: Dict[str, str] = {
    "AGE_YEARS": "age",
    "NAME_FAMILY_STATUS": "marital status",
    "CODE_GENDER": "sex",
}

# Fields that must not even be accepted at the request boundary, as opposed to characteristics
# that must not be scored on. Age is the distinction that makes this a separate list: an
# applicant has to be of age to contract, so `DAYS_BIRTH` is legitimately collected and range
# checked, and is simply never turned into a characteristic. Sex and marital status have no
# such justification here and the contract refuses them outright.
MUST_NOT_COLLECT = ("CODE_GENDER", "NAME_FAMILY_STATUS")


def prohibited_bases_in(features) -> Dict[str, str]:
    """Which of these characteristics are prohibited bases, name to basis.

    Name matching is the limit of what this can do. A characteristic derived from a prohibited
    basis under some other name, an age band called `LIFE_STAGE` say, passes it. The registry
    catches what it names, which is worth having and is not the same as proving a card is clean.
    """
    return {name: PROHIBITED_BASES[name] for name in features if name in PROHIBITED_BASES}


@dataclass(frozen=True)
class ReasonStatement:
    """What an applicant is told when a characteristic is cited.

    `code` is stable across refits by construction: it is written here against a characteristic
    name rather than derived from the fitted card's ordering. A code whose meaning changed when
    the model was refitted would make every previously issued notice unauditable.

    Whether a characteristic is a prohibited basis is not recorded here. It lives in
    `PROHIBITED_BASES`, because the fit time check has to answer that question for
    characteristics nobody has written a statement for yet.
    """

    code: str
    statement: str
    missing_statement: Optional[str] = None


# One entry per characteristic the card may retain. Phrased as the factor, in the manner of the
# model adverse action forms, rather than as advice about what the applicant should do.
STATEMENTS: Dict[str, ReasonStatement] = {
    "EXT_SOURCE_1": ReasonStatement(
        "AA01",
        "External credit bureau score (source 1) below the level required",
        missing_statement="No external credit bureau score (source 1) on file",
    ),
    "EXT_SOURCE_2": ReasonStatement(
        "AA02",
        "External credit bureau score (source 2) below the level required",
        missing_statement="No external credit bureau score (source 2) on file",
    ),
    "EXT_SOURCE_3": ReasonStatement(
        "AA03",
        "External credit bureau score (source 3) below the level required",
        missing_statement="No external credit bureau score (source 3) on file",
    ),
    "NAME_EDUCATION_TYPE": ReasonStatement(
        "AA04", "Level of education recorded on the application"
    ),
    "EMPLOYED_YEARS": ReasonStatement(
        "AA05",
        "Length of employment too short",
        # The source sentinel for "no employment record" marks a pensioner, so this statement
        # is strongly age correlated even though it names employment. Noted in the audit.
        missing_statement="No employment record on file",
    ),
    "GOODS_CREDIT_RATIO": ReasonStatement(
        "AA06",
        "Amount financed is high relative to the value of the goods purchased",
        missing_statement="Value of the goods purchased was not supplied",
    ),
    "CREDIT_TERM": ReasonStatement(
        "AA07",
        "Instalment is high relative to the amount of credit requested",
        missing_statement="Instalment amount was not supplied",
    ),
    "AMT_CREDIT": ReasonStatement("AA08", "Amount of credit requested"),
    "ID_PUBLISH_YEARS": ReasonStatement(
        "AA09", "Identity document has been on file for too short a time"
    ),
    "PHONE_CHANGE_YEARS": ReasonStatement(
        "AA10",
        "Telephone number on file was changed recently",
        missing_statement="No telephone number change date on file",
    ),
    "REGION_POPULATION_RELATIVE": ReasonStatement(
        "AA11", "Population density of the region of residence"
    ),
    "NAME_INCOME_TYPE": ReasonStatement("AA12", "Type of income recorded on the application"),
    "AGE_YEARS": ReasonStatement("AA13", "Age of applicant"),
    "NAME_FAMILY_STATUS": ReasonStatement("AA14", "Marital status"),
}


@dataclass
class AssignedReason:
    """One principal reason, with the arithmetic that produced it.

    The points are carried alongside the statement because a reason nobody can check is a
    reason nobody can contest. A reviewer handling a complaint needs to see that this
    characteristic cost this applicant this many points against this reference.
    """

    rank: int
    code: str
    feature: str
    statement: str
    points: float
    reference_points: float
    shortfall: float
    prohibited_basis: Optional[str] = None

    def as_dict(self) -> Dict[str, object]:
        """The published form, rounded so that the three numbers still add up.

        Rounding the shortfall independently of the two values it is the difference of leaves
        them disagreeing by a cent, which on a document telling somebody why they were refused
        credit is an invitation to challenge the whole notice. The shortfall is therefore
        derived from the rounded pair. Ranking upstream uses the unrounded value, so which
        reasons appear is unaffected by how they are displayed.
        """
        points = round(self.points, 2)
        reference = round(self.reference_points, 2)
        return {
            "rank": self.rank,
            "code": self.code,
            "feature": self.feature,
            "statement": self.statement,
            "points": points,
            "reference_points": reference,
            "shortfall": round(reference - points, 2),
        }


class ReasonCodeAssigner:
    """Turns a set of bin indices into the principal reasons for an adverse decision."""

    def __init__(
        self,
        artifact: ScorecardArtifact,
        basis: str = BASIS_POPULATION_MEAN,
        max_reasons: int = 4,
        min_shortfall_points: float = 0.05,
        adverse_bands: Sequence[str] = (BAND_DECLINE,),
    ):
        if basis not in VALID_BASES:
            raise ValueError(f"unknown reason code basis {basis!r}, expected one of {VALID_BASES}")
        if max_reasons < 1:
            raise ValueError("max_reasons must be at least 1")

        self.artifact = artifact
        self.basis = basis
        self.max_reasons = max_reasons
        self.min_shortfall_points = float(min_shortfall_points)
        self.adverse_bands = tuple(adverse_bands)

        self.features: List[str] = list(artifact.scorecard.features)
        self._points: Dict[str, np.ndarray] = artifact.scorecard.points_by_bin(artifact.transformer)
        self._missing_index: Dict[str, int] = {
            feature: artifact.transformer.bins[feature].missing_index for feature in self.features
        }
        self._reference: Dict[str, float] = {
            feature: self._reference_points(feature) for feature in self.features
        }

    def _reference_points(self, feature: str) -> float:
        points = self._points[feature]
        if self.basis == BASIS_MAX_OBSERVED:
            observed = np.delete(points, self._missing_index[feature])
            # A characteristic whose only bin is the missing bin cannot produce a shortfall.
            return float(observed.max()) if observed.size else float(points.max())

        proportions = self.artifact.transformer.bins[feature].reference_proportions
        if proportions.size != points.size or not np.isfinite(proportions).all():
            raise ValueError(
                f"cannot take a population mean reference for {feature!r}: the artifact holds "
                f"{proportions.size} reference proportions for {points.size} bins. Refit, or "
                f"use basis={BASIS_MAX_OBSERVED!r}."
            )
        return float(np.dot(proportions, points))

    def unstated_features(self) -> List[str]:
        """Retained characteristics with no applicant facing statement written for them.

        A refit that adds a characteristic must add its statement too. Without this check the
        service would decline an applicant and cite a raw column name at them.
        """
        return sorted(feature for feature in self.features if feature not in STATEMENTS)

    def prohibited_basis_features(self) -> Dict[str, str]:
        """Retained characteristics that are prohibited bases, name to basis."""
        return prohibited_bases_in(self.features)

    def is_adverse(self, band: str) -> bool:
        return band in self.adverse_bands

    def assign(self, bin_indices: Dict[str, int], band: str) -> List[AssignedReason]:
        """The principal reasons for one decision, most costly first.

        Returns nothing for a band that is not an adverse action. A reason code on an approved
        application is not a reason for anything, and returning one invites it to be presented
        as though it were.
        """
        if not self.is_adverse(band):
            return []

        scored: List[Tuple[float, str, int]] = []
        for feature in self.features:
            index = bin_indices.get(feature)
            if index is None:
                raise KeyError(
                    f"no bin index for retained characteristic {feature!r}. The reasons would "
                    "silently omit a characteristic the score was built from."
                )
            index = int(index)
            shortfall = self._reference[feature] - float(self._points[feature][index])
            if shortfall > self.min_shortfall_points:
                scored.append((shortfall, feature, index))

        # Descending shortfall, then feature name, so ties resolve the same way every time. Two
        # applications with identical characteristics have to produce identical notices.
        scored.sort(key=lambda row: (-row[0], row[1]))

        reasons: List[AssignedReason] = []
        for rank, (shortfall, feature, index) in enumerate(scored[: self.max_reasons], start=1):
            statement = STATEMENTS.get(feature)
            if statement is None:
                raise KeyError(
                    f"no adverse action statement written for characteristic {feature!r}. "
                    "Add one to reasons.STATEMENTS rather than sending a column name to an "
                    "applicant."
                )
            is_missing = index == self._missing_index[feature]
            text = statement.missing_statement if is_missing else statement.statement
            if text is None:
                # The characteristic reached its missing bin but was never expected to. Saying
                # "below the level required" about an absent value would be false.
                raise ValueError(
                    f"{feature!r} scored in its missing bin but has no statement for that case. "
                    "Either the request contract now admits a missing value it did not before, "
                    "or the bin is unreachable and this is a bug."
                )
            reasons.append(
                AssignedReason(
                    rank=rank,
                    code=statement.code,
                    feature=feature,
                    statement=text,
                    points=float(self._points[feature][index]),
                    reference_points=self._reference[feature],
                    shortfall=float(shortfall),
                    prohibited_basis=PROHIBITED_BASES.get(feature),
                )
            )
        return reasons


def verify_no_prohibited_basis(features, on_prohibited: str = ON_PROHIBITED_RAISE) -> None:
    """Refuse to fit a card that retains a prohibited basis.

    This is the check that should have existed first. `verify_reason_coverage` asks the same
    question at service startup and the audit asks it after the fact, and both are too late:
    by then the artifact exists, it has been evaluated, its numbers are in a README, and the
    cost of retracting is high enough to argue about. Beside feature selection the answer is
    free, because nothing has been built on the card yet.

    It raises by default, which is the opposite of the serving check. Refusing to produce an
    artifact costs a failed training run. Refusing to serve costs an outage on a card that is
    already deployed and already scoring, which is why that one warns instead.

    The honest limit is in `prohibited_bases_in`: this matches on characteristic names, so it
    catches the characteristics the registry names and nothing else.
    """
    prohibited = prohibited_bases_in(features)
    if not prohibited:
        return

    described = ", ".join(f"{name} ({basis})" for name, basis in sorted(prohibited.items()))
    message = (
        f"the fitted card retains prohibited bases under ECOA: {described}. A card that "
        "allocates points on these cannot be signed off, and an adverse action notice would "
        "have to disclose them. Remove them from `features` in the config, or set "
        "`adverse_action.on_prohibited_basis_at_fit: warn` if you are deliberately fitting one "
        "to measure what it costs."
    )
    if on_prohibited == ON_PROHIBITED_RAISE:
        raise RuntimeError(message)
    print(f"fit time {message}", file=sys.stderr, flush=True)


def verify_reason_coverage(
    assigner: ReasonCodeAssigner, on_prohibited: str = ON_PROHIBITED_WARN
) -> None:
    """Check at startup that every retained characteristic can be explained to an applicant.

    Two different findings, deliberately treated differently.

    A characteristic with no statement is a hard failure. The service would decline someone and
    have nothing to tell them, and there is no reading of Regulation B where that is acceptable.

    A characteristic that is a prohibited basis is not a failure of this module. The card was
    fitted on it, the score genuinely depends on it, and suppressing the reason would leave the
    applicant with a notice that does not name what actually drove the decision. It is reported
    rather than hidden, and the remedy is to refit without the characteristic. `raise` is
    available for a deployment that would rather not serve at all than serve this card.
    """
    unstated = assigner.unstated_features()
    if unstated:
        raise RuntimeError(
            "no adverse action statement for retained characteristics: "
            + ", ".join(unstated)
            + ". A declined applicant has to be told the reason, so this cannot be served."
        )

    prohibited = assigner.prohibited_basis_features()
    if not prohibited:
        return

    described = ", ".join(f"{feature} ({basis})" for feature, basis in sorted(prohibited.items()))
    message = (
        "adverse action check: the card allocates points on prohibited bases under ECOA, and a "
        f"reason code will disclose them: {described}. The remedy is a refit without these "
        "characteristics, not a suppressed reason code."
    )
    if on_prohibited == ON_PROHIBITED_RAISE:
        raise RuntimeError(message)
    print(message, file=sys.stderr, flush=True)
