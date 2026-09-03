"""Tests for the adverse action reason codes.

These exist because of what a reason code is. It is not a diagnostic, it is a statement sent to
a person about why they were refused credit, and every one of the tests below corresponds to a
way that statement could be false rather than merely unhelpful:

- points that do not sum to the score make every "this cost you N points" claim wrong,
- a missing value described as a low value is a false statement about the applicant's file,
- an unstable sort sends two identical applications two different notices,
- a reason on an approval is a reason for a decision that was never taken,
- an unstated characteristic sends a raw column name to an applicant.

The Regulation B age rule is tested on constructed cards rather than on the fitted one. What
this particular card does about age is a finding, and findings belong in `reports/` where the
audit writes them, not in an assertion that would have to be edited the day the card is fixed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features import build_features
from src.reasons import (
    BASIS_MAX_OBSERVED,
    BASIS_POPULATION_MEAN,
    ON_PROHIBITED_RAISE,
    PROHIBITED_BASES,
    STATEMENTS,
    ReasonCodeAssigner,
    prohibited_bases_in,
    verify_no_prohibited_basis,
    verify_reason_coverage,
)
from src.scoring import ScoringService, load_artifact
from tests.conftest import REAL_MODEL, requires_model

pytestmark = requires_model


@pytest.fixture(scope="module")
def artifact():
    return load_artifact(REAL_MODEL)


@pytest.fixture(scope="module")
def assigner(artifact):
    return ReasonCodeAssigner(artifact)


@pytest.fixture
def declined_payload(valid_payload) -> dict:
    """The shared payload weakened until it declines, keeping every value in contract range."""
    return dict(
        valid_payload,
        EXT_SOURCE_1=None,
        EXT_SOURCE_2=0.05,
        EXT_SOURCE_3=0.05,
        NAME_EDUCATION_TYPE="Lower secondary",
        DAYS_EMPLOYED=-200.0,
    )


def _bins(artifact, payload: dict) -> dict:
    frame = build_features(pd.DataFrame([payload]))
    return {k: int(v) for k, v in artifact.transformer.bin_indices(frame).iloc[0].items()}


# The arithmetic every reason rests on -------------------------------------------------

def test_points_sum_to_the_served_score(artifact, valid_payload, declined_payload):
    """A reason claims a characteristic cost N points. That is only true if the parts add up."""
    frame = build_features(pd.DataFrame([valid_payload, declined_payload]))
    points = artifact.scorecard.points_by_bin(artifact.transformer)
    indices = artifact.transformer.bin_indices(frame)

    total = np.zeros(len(frame))
    for feature in artifact.scorecard.features:
        total += points[feature][indices[feature].to_numpy()]

    assert np.allclose(total, artifact.score_frame(frame)["score"].to_numpy(), atol=1e-9)


def test_the_card_table_and_the_reason_points_agree(artifact):
    """Both read the same allocation, so a committee's card and an applicant's notice match."""
    table = artifact.scorecard.scorecard_table(artifact.transformer)
    points = artifact.scorecard.points_by_bin(artifact.transformer)
    for feature, group in table.groupby("feature"):
        if feature not in points:
            continue  # a monitored but unretained characteristic carries no points
        expected = points[feature][group["bin_index"].to_numpy()].round(1)
        assert np.allclose(group["points"].to_numpy(), expected)


# What gets returned, and when ---------------------------------------------------------

def test_reasons_are_returned_only_for_an_adverse_band(assigner, artifact, declined_payload):
    indices = _bins(artifact, declined_payload)
    assert assigner.assign(indices, "decline")
    # A referral is not a decision, and an approval is not adverse. Neither has reasons.
    assert assigner.assign(indices, "refer") == []
    assert assigner.assign(indices, "approve") == []


def test_reasons_are_ranked_by_shortfall_and_capped(assigner, artifact, declined_payload):
    reasons = assigner.assign(_bins(artifact, declined_payload), "decline")
    assert len(reasons) <= assigner.max_reasons
    assert [r.rank for r in reasons] == list(range(1, len(reasons) + 1))
    shortfalls = [r.shortfall for r in reasons]
    assert shortfalls == sorted(shortfalls, reverse=True)
    assert all(s > assigner.min_shortfall_points for s in shortfalls)


def test_the_same_application_always_produces_the_same_notice(assigner, artifact, declined_payload):
    """Ties are broken on feature name, so the order cannot depend on dict iteration."""
    indices = _bins(artifact, declined_payload)
    first = [(r.code, r.rank) for r in assigner.assign(indices, "decline")]
    second = [(r.code, r.rank) for r in assigner.assign(dict(reversed(list(indices.items()))), "decline")]
    assert first == second


def test_a_shortfall_is_the_reference_less_the_points(assigner, artifact, declined_payload):
    for reason in assigner.assign(_bins(artifact, declined_payload), "decline"):
        assert reason.shortfall == pytest.approx(reason.reference_points - reason.points)


# Missing values are a different statement ----------------------------------------------

def test_an_absent_characteristic_is_never_described_as_a_low_one(assigner, artifact, valid_payload):
    """A missing value and a bad value are different facts about a file, so they read differently.

    Asserted as an invariant over whatever the ranking happens to cite, rather than against a
    named characteristic. An earlier version of this test pinned `EXT_SOURCE_1` specifically and
    broke the moment the reference profile changed which characteristics reach the top four,
    which was a test of the ranking pretending to be a test of the wording.
    """
    thin_file = dict(
        valid_payload,
        EXT_SOURCE_1=None,
        EXT_SOURCE_3=None,
        EXT_SOURCE_2=0.05,
        NAME_EDUCATION_TYPE="Lower secondary",
        DAYS_EMPLOYED=-200.0,
    )
    indices = _bins(artifact, thin_file)
    reasons = assigner.assign(indices, "decline")
    assert reasons, "the thin file payload must decline for this test to mean anything"

    missing_cited = 0
    for reason in reasons:
        statement = STATEMENTS[reason.feature]
        if indices[reason.feature] == artifact.transformer.bins[reason.feature].missing_index:
            assert reason.statement == statement.missing_statement
            missing_cited += 1
        else:
            assert reason.statement == statement.statement

    assert missing_cited, "no absent characteristic was cited, so the wording was never exercised"


def test_every_optional_request_field_has_a_missing_statement(artifact):
    """A characteristic that can arrive missing needs wording for that case, or `assign` raises.

    The alternative is discovering it in production, on the one applicant whose file is thin.
    """
    can_be_missing = [
        "EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3",
        "EMPLOYED_YEARS", "PHONE_CHANGE_YEARS", "CREDIT_TERM", "GOODS_CREDIT_RATIO",
    ]
    for feature in can_be_missing:
        if feature in artifact.scorecard.features:
            assert STATEMENTS[feature].missing_statement, feature


# The reference profile -----------------------------------------------------------------

def test_the_missing_bin_is_excluded_from_the_max_reference(artifact):
    """Otherwise a bin nothing lands in becomes the reference and cites everyone.

    A sparse missing bin gets a smoothed weight of evidence that is not a statement about
    anybody, and it can easily out score every observed bin. Where a required field feeds the
    characteristic, that bin is unreachable too. Including it would put a shortfall on every
    applicant scored, on a characteristic none of them are actually weak on.
    """
    assigner = ReasonCodeAssigner(artifact, basis=BASIS_MAX_OBSERVED)
    all_points = artifact.scorecard.points_by_bin(artifact.transformer)
    candidates = []
    for feature in artifact.scorecard.features:
        points = all_points[feature]
        missing_index = artifact.transformer.bins[feature].missing_index
        observed_best = float(np.delete(points, missing_index).max())
        if points[missing_index] > observed_best:
            candidates.append((feature, observed_best))

    if not candidates:
        pytest.skip("no characteristic on this fit has a missing bin above its best observed")

    for feature, observed_best in candidates:
        assert assigner._reference[feature] == pytest.approx(observed_best)


def test_population_mean_is_a_real_average_of_the_training_population(artifact):
    other = ReasonCodeAssigner(artifact, basis=BASIS_POPULATION_MEAN)
    for feature in artifact.scorecard.features:
        points = artifact.scorecard.points_by_bin(artifact.transformer)[feature]
        assert points.min() <= other._reference[feature] <= points.max()


def test_the_two_bases_are_both_available_and_differ(artifact, declined_payload):
    """The choice is a real one: it changes which characteristics are named."""
    indices = _bins(artifact, declined_payload)
    by_max = ReasonCodeAssigner(artifact, basis=BASIS_MAX_OBSERVED).assign(indices, "decline")
    by_mean = ReasonCodeAssigner(artifact, basis=BASIS_POPULATION_MEAN).assign(indices, "decline")
    assert by_max and by_mean
    assert all(r.shortfall > 0 for r in by_max + by_mean)


def test_an_unknown_basis_is_refused(artifact):
    with pytest.raises(ValueError, match="unknown reason code basis"):
        ReasonCodeAssigner(artifact, basis="whatever_looks_best")


# Startup checks -------------------------------------------------------------------------

def test_a_characteristic_with_no_statement_cannot_be_served(artifact, monkeypatch):
    """A declined applicant has to be told something, so this is a hard failure, not a warning."""
    assigner = ReasonCodeAssigner(artifact)
    monkeypatch.setattr(assigner, "features", assigner.features + ["SOME_NEW_CHARACTERISTIC"])
    assert assigner.unstated_features() == ["SOME_NEW_CHARACTERISTIC"]
    with pytest.raises(RuntimeError, match="no adverse action statement"):
        verify_reason_coverage(assigner)


def test_the_fitted_card_allocates_no_points_on_a_prohibited_basis(assigner):
    """The point of the refit, asserted rather than described.

    `AGE_YEARS` and `NAME_FAMILY_STATUS` were both retained until the reason codes forced the
    question. Neither is on the card now, and neither is in the request contract, so this fails
    if either is reintroduced.
    """
    assert assigner.prohibited_basis_features() == {}
    assert "AGE_YEARS" not in assigner.features
    assert "NAME_FAMILY_STATUS" not in assigner.features


def _with_prohibited_basis(artifact, feature="AGE_YEARS"):
    """An assigner reporting a card that does allocate points on a prohibited basis."""
    assigner = ReasonCodeAssigner(artifact)
    assigner.features = assigner.features + [feature]
    return assigner


def test_a_deployment_can_refuse_to_serve_a_prohibited_basis_card(artifact):
    with pytest.raises(RuntimeError, match="prohibited bases under ECOA"):
        verify_reason_coverage(_with_prohibited_basis(artifact), on_prohibited=ON_PROHIBITED_RAISE)


def test_warning_is_the_default_and_does_not_stop_the_service(artifact, capsys):
    verify_reason_coverage(_with_prohibited_basis(artifact))
    err = capsys.readouterr().err
    assert "prohibited bases under ECOA" in err
    assert "AGE_YEARS" in err


def test_the_statements_still_flag_a_prohibited_basis_if_one_returns(artifact):
    """The guard has to survive the refit, or it protects nothing against the next one."""
    assert PROHIBITED_BASES["AGE_YEARS"] == "age"
    assert PROHIBITED_BASES["NAME_FAMILY_STATUS"] == "marital status"
    assert _with_prohibited_basis(artifact).prohibited_basis_features() == {"AGE_YEARS": "age"}


# The Regulation B age rule, on constructed cards -----------------------------------------

def _age_check(points, edges):
    """Run the audit's age rule against a constructed age characteristic."""
    from scripts.adverse_action_audit import check_age_rule

    class _Binning:
        def __init__(self):
            self.edges = np.array(edges)
            self.missing_index = len(points)

    class _Transformer:
        bins = {"AGE_YEARS": _Binning()}

    class _Scorecard:
        features = ["AGE_YEARS"]

        def points_by_bin(self, _transformer):
            return {"AGE_YEARS": np.append(np.array(points, dtype="float64"), 0.0)}

    class _Artifact:
        transformer = _Transformer()
        scorecard = _Scorecard()

    return check_age_rule(_Artifact())


def test_age_rule_fails_a_card_that_penalises_older_applicants():
    """Points falling with age means an applicant over 62 cannot reach the best age points."""
    result = _age_check(points=[50.0, 45.0, 40.0], edges=[30.0, 62.0])
    assert result["applicable"]
    assert not result["compliant"]
    assert result["max_penalty_vs_best_age"] == pytest.approx(10.0)


def test_age_rule_passes_a_card_that_treats_older_applicants_at_least_as_well():
    result = _age_check(points=[40.0, 45.0, 50.0], edges=[30.0, 62.0])
    assert result["compliant"]
    assert result["best_points_available_to_62_plus"] == pytest.approx(50.0)


def test_age_rule_is_skipped_when_age_is_not_on_the_card():
    from scripts.adverse_action_audit import check_age_rule

    class _Artifact:
        class transformer:
            bins = {}

    assert check_age_rule(_Artifact())["applicable"] is False


# Through the service --------------------------------------------------------------------

def test_the_scoring_service_attaches_reasons_to_a_decline(artifact, declined_payload, valid_payload):
    service = ScoringService(artifact)
    declined = service.score_payload(declined_payload)
    approved = service.score_payload(valid_payload)

    assert declined.band == "decline"
    assert declined.reason_codes
    assert approved.band != "decline"
    assert approved.reason_codes == []


# The fit time gate ------------------------------------------------------------------------

def test_the_registry_is_the_single_source_of_truth():
    """A statement is not required for the compliance list to know a characteristic is barred.

    That is the whole point of keeping them apart: the characteristic that slips through is the
    one nobody has written an applicant facing statement for yet.
    """
    assert prohibited_bases_in(["AGE_YEARS", "EXT_SOURCE_1"]) == {"AGE_YEARS": "age"}
    assert PROHIBITED_BASES["CODE_GENDER"] == "sex"
    assert "CODE_GENDER" not in STATEMENTS  # barred, and no notice wording for it


def test_a_clean_card_passes_the_fit_time_gate(artifact):
    verify_no_prohibited_basis(artifact.scorecard.features)


def test_the_fit_time_gate_refuses_to_produce_the_artifact():
    """Raising is the right default here, unlike at serving. Nothing is deployed yet."""
    with pytest.raises(RuntimeError, match="retains prohibited bases under ECOA"):
        verify_no_prohibited_basis(["EXT_SOURCE_1", "NAME_FAMILY_STATUS"])


def test_the_fit_time_gate_can_be_lowered_to_measure_a_cost(capsys):
    """Fitting one deliberately is how the cost of removing it was measured in the first place."""
    verify_no_prohibited_basis(["AGE_YEARS"], on_prohibited="warn")
    assert "AGE_YEARS (age)" in capsys.readouterr().err


def test_training_consults_the_gate_with_the_selected_characteristics(monkeypatch):
    """The gate has to be wired into training, not merely available to it.

    The defect this whole episode came from was a question nobody asked at fit time, so the
    assertion that matters is that `train` asks it, about the characteristics it actually kept.
    """
    from src import train as train_module

    seen = {}

    def _record(features, on_prohibited=None):
        seen["features"] = list(features)
        raise RuntimeError("stop here, the gate was reached")

    monkeypatch.setattr(train_module, "verify_no_prohibited_basis", _record)
    monkeypatch.setattr(
        train_module, "select_features", lambda *a, **k: ["EXT_SOURCE_3", "NAME_FAMILY_STATUS"]
    )
    with pytest.raises(RuntimeError, match="the gate was reached"):
        train_module.train()

    assert seen["features"] == ["EXT_SOURCE_3", "NAME_FAMILY_STATUS"]
