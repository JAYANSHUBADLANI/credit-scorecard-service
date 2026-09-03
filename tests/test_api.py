"""Tests for the scoring endpoint.

Two claims made elsewhere in the project are checked here rather than asserted in prose: that
malformed input is refused with the offending field named, and that a refused request is not
written to the scoring log. The second matters more than it looks. If rejected requests were
logged, every caller side bug would eventually surface as a characteristic shift and send
someone looking for a population change that never happened.
"""

from __future__ import annotations

import json

import pytest

from tests.conftest import requires_model

pytestmark = requires_model


# The service is up ------------------------------------------------------------------

def test_health_reports_the_loaded_model(api_client):
    body = api_client.get("/health").json()
    assert body["status"] == "ok"
    assert body["model_version"]
    assert body["features"] > 0
    assert body["requests_scored"] == 0


def test_model_metadata_describes_what_is_deployed(api_client):
    body = api_client.get("/model").json()
    assert body["model_features"]
    assert body["band_cutoffs"]["decline_below"] < body["band_cutoffs"]["refer_below"]
    assert body["scaling"]["pdo"] == 20.0
    assert "NAME_EDUCATION_TYPE" in body["accepted_categorical_levels"]
    assert "CODE_GENDER" not in body["accepted_categorical_levels"]
    assert "CODE_GENDER" not in body["model_features"]


# Scoring ----------------------------------------------------------------------------

def test_valid_request_is_scored(api_client, valid_payload):
    response = api_client.post("/score", json=valid_payload)
    assert response.status_code == 200

    body = response.json()
    assert 0.0 <= body["probability"] <= 1.0
    assert body["band"] in {"decline", "refer", "approve"}
    assert body["request_id"]
    assert body["scored_at"]
    assert 300 < body["score"] < 900


def test_scoring_is_deterministic(api_client, valid_payload):
    first = api_client.post("/score", json=valid_payload).json()
    second = api_client.post("/score", json=valid_payload).json()
    assert first["score"] == second["score"]
    assert first["probability"] == second["probability"]
    assert first["request_id"] != second["request_id"]


def test_absent_bureau_scores_are_accepted_not_rejected(api_client, valid_payload):
    """Missing is a legitimate value with its own fitted bin, unlike malformed."""
    payload = {**valid_payload, "EXT_SOURCE_1": None, "EXT_SOURCE_3": None}
    response = api_client.post("/score", json=payload)
    assert response.status_code == 200


def test_optional_fields_may_be_omitted_entirely(api_client, valid_payload):
    payload = {k: v for k, v in valid_payload.items() if k not in {"EXT_SOURCE_1", "AMT_ANNUITY"}}
    assert api_client.post("/score", json=payload).status_code == 200


def test_employment_sentinel_is_accepted(api_client, valid_payload):
    """365243 is the source data marker for no employment record, not a bad value."""
    response = api_client.post("/score", json={**valid_payload, "DAYS_EMPLOYED": 365243.0})
    assert response.status_code == 200


def test_a_weaker_application_scores_lower_than_a_stronger_one(api_client, valid_payload):
    """A sanity check on direction: the card must not be inverted."""
    strong = api_client.post(
        "/score",
        json={**valid_payload, "EXT_SOURCE_1": 0.85, "EXT_SOURCE_2": 0.80, "EXT_SOURCE_3": 0.78},
    ).json()
    weak = api_client.post(
        "/score",
        json={**valid_payload, "EXT_SOURCE_1": 0.10, "EXT_SOURCE_2": 0.08, "EXT_SOURCE_3": 0.12},
    ).json()
    assert strong["score"] > weak["score"]
    assert strong["probability"] < weak["probability"]


def test_bands_follow_the_published_cutoffs(api_client, valid_payload):
    cutoffs = api_client.get("/model").json()["band_cutoffs"]
    for payload in [
        valid_payload,
        {**valid_payload, "EXT_SOURCE_1": 0.05, "EXT_SOURCE_2": 0.05, "EXT_SOURCE_3": 0.05},
        {**valid_payload, "EXT_SOURCE_1": 0.95, "EXT_SOURCE_2": 0.85, "EXT_SOURCE_3": 0.90},
    ]:
        body = api_client.post("/score", json=payload).json()
        if body["score"] < cutoffs["decline_below"]:
            assert body["band"] == "decline"
        elif body["score"] < cutoffs["refer_below"]:
            assert body["band"] == "refer"
        else:
            assert body["band"] == "approve"


# Validation --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "field,value,expected_field",
    [
        ("EXT_SOURCE_2", "0.35", "EXT_SOURCE_2"),          # string, not silently coerced
        ("EXT_SOURCE_2", 1.4, "EXT_SOURCE_2"),             # above the valid range
        ("EXT_SOURCE_2", -0.1, "EXT_SOURCE_2"),            # below the valid range
        ("DAYS_BIRTH", 9461.0, "DAYS_BIRTH"),              # sign error
        ("DAYS_BIRTH", -1500.0, "DAYS_BIRTH"),             # four years old
        ("DAYS_BIRTH", -40000.0, "DAYS_BIRTH"),            # 109 years old
        ("AMT_INCOME_TOTAL", -5000.0, "AMT_INCOME_TOTAL"), # negative income
        ("AMT_INCOME_TOTAL", 0.0, "AMT_INCOME_TOTAL"),     # zero income divides the ratios
        ("AMT_CREDIT", 0.0, "AMT_CREDIT"),
        ("REGION_POPULATION_RELATIVE", 2.0, "REGION_POPULATION_RELATIVE"),
        ("NAME_EDUCATION_TYPE", "PhD", "NAME_EDUCATION_TYPE"),
        # Case matters on a categorical, it is a fitted level rather than free text.
        ("NAME_EDUCATION_TYPE", "higher education", "NAME_EDUCATION_TYPE"),
        # Marital status is a prohibited basis and the contract refuses the field itself.
        ("NAME_FAMILY_STATUS", "Married", "NAME_FAMILY_STATUS"),
        ("NAME_CONTRACT_TYPE", "Mortgage", "NAME_CONTRACT_TYPE"),
    ],
)
def test_out_of_range_or_malformed_values_are_rejected(
    api_client, valid_payload, field, value, expected_field
):
    response = api_client.post("/score", json={**valid_payload, field: value})
    assert response.status_code == 422

    body = response.json()
    assert body["error"] == "request validation failed"
    assert expected_field in {detail["field"] for detail in body["detail"]}
    assert all(detail["message"] for detail in body["detail"])


def test_unknown_field_is_rejected_rather_than_ignored(api_client, valid_payload):
    """A misspelled field silently dropped would score the applicant without it."""
    response = api_client.post("/score", json={**valid_payload, "EXT_SOURCE_TWO": 0.4})
    assert response.status_code == 422
    assert "EXT_SOURCE_TWO" in {d["field"] for d in response.json()["detail"]}


def test_gender_is_refused_rather_than_ignored(api_client, valid_payload):
    """Gender is a prohibited basis, so the service does not merely decline to model it.

    Accepting the field and dropping it would be worse than refusing: the caller would have
    no way to tell that the characteristic they sent played no part, and the field would sit
    in the request contract waiting for a later refit to pick it up.
    """
    response = api_client.post("/score", json={**valid_payload, "CODE_GENDER": "F"})
    assert response.status_code == 422
    assert "CODE_GENDER" in {d["field"] for d in response.json()["detail"]}


def test_missing_required_field_is_rejected(api_client, valid_payload):
    payload = {k: v for k, v in valid_payload.items() if k != "AMT_CREDIT"}
    response = api_client.post("/score", json=payload)
    assert response.status_code == 422
    assert "AMT_CREDIT" in {d["field"] for d in response.json()["detail"]}


def test_cross_field_rule_annuity_cannot_exceed_credit(api_client, valid_payload):
    response = api_client.post(
        "/score", json={**valid_payload, "AMT_CREDIT": 50000.0, "AMT_ANNUITY": 90000.0}
    )
    assert response.status_code == 422
    assert "AMT_ANNUITY" in json.dumps(response.json())


def test_positive_employment_days_other_than_the_sentinel_are_rejected(api_client, valid_payload):
    response = api_client.post("/score", json={**valid_payload, "DAYS_EMPLOYED": 3000.0})
    assert response.status_code == 422
    assert "DAYS_EMPLOYED" in json.dumps(response.json())


def test_empty_body_is_rejected_with_every_missing_field_named(api_client):
    response = api_client.post("/score", json={})
    assert response.status_code == 422
    assert len(response.json()["detail"]) >= 5


# The scoring log ----------------------------------------------------------------------

def test_scored_requests_are_logged(api_client, valid_payload):
    for _ in range(3):
        assert api_client.post("/score", json=valid_payload).status_code == 200
    assert api_client.get("/health").json()["requests_scored"] == 3


def test_rejected_requests_are_not_logged(api_client, valid_payload):
    """A refused request must not reach the drift baseline."""
    assert api_client.post("/score", json=valid_payload).status_code == 200
    assert api_client.post("/score", json={**valid_payload, "EXT_SOURCE_2": 5.0}).status_code == 422
    assert api_client.post("/score", json={}).status_code == 422
    assert api_client.get("/health").json()["requests_scored"] == 1


def test_log_records_the_bin_each_characteristic_fell_into(api_client, valid_payload, temp_config):
    """The characteristic index is computed from these, so they have to be stored per request."""
    from src.config import load_config
    from src.store import Store

    api_client.post("/score", json=valid_payload)

    config = load_config(temp_config)
    store = Store(config.path(config.service.db_path))
    row = store.fetch_recent_scores(limit=1)[0]
    bins = json.loads(row["bin_indices"])

    assert row["source"] == "api"
    assert row["latency_ms"] is not None
    assert len(bins) >= 15
    assert all(isinstance(index, int) for index in bins.values())
    assert "EXT_SOURCE_2" in bins


# Adverse action reason codes ---------------------------------------------------------

def test_a_decline_carries_its_principal_reasons(api_client, valid_payload):
    """Regulation B attaches the disclosure to the decline, so the endpoint has to carry it."""
    declined = dict(
        valid_payload,
        EXT_SOURCE_1=None,
        EXT_SOURCE_2=0.05,
        EXT_SOURCE_3=0.05,
        NAME_EDUCATION_TYPE="Lower secondary",
        DAYS_EMPLOYED=-200.0,
    )
    body = api_client.post("/score", json=declined).json()

    assert body["band"] == "decline"
    reasons = body["reason_codes"]
    assert reasons
    assert [r["rank"] for r in reasons] == list(range(1, len(reasons) + 1))
    # The arithmetic is published with the statement so the applicant can check it.
    for reason in reasons:
        assert reason["code"] and reason["statement"]
        assert reason["shortfall"] == pytest.approx(
            reason["reference_points"] - reason["points"], abs=0.01
        )


def test_an_approval_carries_no_reasons(api_client, valid_payload):
    """A reason code on an approval is a reason for a decision that was never taken."""
    body = api_client.post("/score", json=valid_payload).json()
    assert body["band"] == "approve"
    assert body["reason_codes"] == []
