"""Tests for the monitoring store.

The migration test is here because of a real failure. The `attributed_features` column was
added to the alerts table after a database had already been created, and `CREATE TABLE IF NOT
EXISTS` does nothing to a table that exists, so the next alert insert failed against the older
file. The monitoring store is meant to outlive any single deploy, which is precisely when this
happens, so the additive migration and this test exist to stop it happening again.
"""

from __future__ import annotations

import json
import sqlite3

import pytest

from src.store import MIGRATIONS, Store


@pytest.fixture
def store(tmp_path):
    return Store(tmp_path / "monitoring.db")


def sample_alert(store: Store, **overrides) -> int:
    payload = {
        "window_id": 3,
        "metric": "psi_score",
        "feature": "score",
        "value": 0.42,
        "threshold": 0.25,
        "severity": "alert",
        "consecutive_windows": 3,
        "breach_window_ids": [1, 2, 3],
        "message": "sustained breach",
    }
    payload.update(overrides)
    return store.record_alert(**payload)


# Schema and migration ------------------------------------------------------------------

def test_schema_is_created_on_first_open(store):
    with store.connect() as connection:
        tables = {
            row["name"]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
    for expected in ["scoring_log", "drift_metrics", "alerts", "window_summary", "monitor_runs"]:
        assert expected in tables


def test_opening_an_existing_database_is_idempotent(tmp_path):
    path = tmp_path / "monitoring.db"
    first = Store(path)
    sample_alert(first)
    second = Store(path)
    assert second.count_alerts() == 1


def test_missing_column_is_added_to_an_existing_database(tmp_path):
    """Reproduces the original failure: an alerts table created before the column existed."""
    path = tmp_path / "legacy.db"
    connection = sqlite3.connect(path)
    connection.execute(
        """
        CREATE TABLE alerts (
            id                  INTEGER PRIMARY KEY AUTOINCREMENT,
            fired_at            TEXT    NOT NULL,
            window_id           INTEGER NOT NULL,
            metric              TEXT    NOT NULL,
            feature             TEXT    NOT NULL,
            value               REAL    NOT NULL,
            threshold           REAL    NOT NULL,
            severity            TEXT    NOT NULL,
            consecutive_windows INTEGER NOT NULL,
            breach_window_ids   TEXT    NOT NULL,
            message             TEXT    NOT NULL
        )
        """
    )
    connection.commit()
    connection.close()

    store = Store(path)
    with store.connect() as connection:
        columns = {row["name"] for row in connection.execute("PRAGMA table_info(alerts)")}
    assert "attributed_features" in columns

    # And the insert that used to fail now succeeds.
    sample_alert(store, attributed_features=["EXT_SOURCE_2"])
    assert json.loads(store.fetch_alerts()[0]["attributed_features"]) == ["EXT_SOURCE_2"]


def test_migration_reports_what_it_changed_and_is_a_no_op_afterwards(tmp_path):
    store = Store(tmp_path / "monitoring.db")
    with store.connect() as connection:
        assert store._migrate(connection) == []
    assert "alerts" in MIGRATIONS


# Scoring log ------------------------------------------------------------------------

def test_scores_round_trip_with_their_bin_indices(store):
    store.log_score(
        request_id="r1", model_version="1.0.0", source="api", score=610.5,
        probability=0.031, band="approve", bin_indices={"EXT_SOURCE_2": 4}, latency_ms=1.2,
    )
    row = store.fetch_recent_scores(limit=1)[0]
    assert row["request_id"] == "r1"
    assert json.loads(row["bin_indices"]) == {"EXT_SOURCE_2": 4}
    assert store.count_scores() == 1


def test_bulk_insert_preserves_order(store):
    store.log_scores_bulk([
        {
            "request_id": f"r{i}", "model_version": "1.0.0", "score": 600.0 + i,
            "probability": 0.05, "band": "approve", "bin_indices": {"f": i},
        }
        for i in range(5)
    ])
    rows = store.fetch_scores_after(0, 10)
    assert [row["request_id"] for row in rows] == [f"r{i}" for i in range(5)]
    assert [int(row["id"]) for row in rows] == [1, 2, 3, 4, 5]


def test_fetch_after_claims_only_new_rows(store):
    store.log_scores_bulk([
        {"request_id": f"r{i}", "model_version": "1", "score": 600.0, "probability": 0.05,
         "band": "approve", "bin_indices": {}}
        for i in range(10)
    ])
    assert len(store.fetch_scores_after(0, 4)) == 4
    assert [int(r["id"]) for r in store.fetch_scores_after(4, 4)] == [5, 6, 7, 8]
    assert len(store.fetch_scores_after(8, 4)) == 2


def test_empty_bulk_insert_is_harmless(store):
    assert store.log_scores_bulk([]) == 0


# Metrics and alerts -------------------------------------------------------------------

def test_metrics_are_unique_per_window_metric_and_feature(store):
    row = {
        "window_id": 1, "window_start_id": 1, "window_end_id": 100, "n_records": 100,
        "metric": "csi", "feature": "AGE_YEARS", "value": 0.05,
        "warn_threshold": 0.1, "alert_threshold": 0.25, "status": "ok",
    }
    store.record_metrics([row])
    store.record_metrics([{**row, "value": 0.09}])
    metrics = store.fetch_all_metrics()
    assert len(metrics) == 1
    assert metrics[0]["value"] == pytest.approx(0.09)


def test_last_window_end_id_drives_the_next_claim(store):
    assert store.last_window_end_id() == 0
    store.record_metrics([{
        "window_id": 1, "window_start_id": 1, "window_end_id": 2000, "n_records": 2000,
        "metric": "psi_score", "feature": "score", "value": 0.01,
        "warn_threshold": 0.1, "alert_threshold": 0.25, "status": "ok",
    }])
    assert store.last_window_end_id() == 2000
    assert store.last_window_id() == 1


def test_alert_history_supports_the_cooldown_lookup(store):
    assert store.last_alert_window("psi_score", "score") is None
    sample_alert(store, window_id=3)
    sample_alert(store, window_id=9)
    assert store.last_alert_window("psi_score", "score") == 9
    assert store.last_alert_window("csi", "AGE_YEARS") is None


def test_reset_clears_state_and_restarts_ids(store):
    store.log_score(
        request_id="r1", model_version="1", source="api", score=600.0,
        probability=0.05, band="approve", bin_indices={},
    )
    sample_alert(store)
    store.reset()

    assert store.count_scores() == 0
    assert store.count_alerts() == 0
    store.log_score(
        request_id="r2", model_version="1", source="api", score=600.0,
        probability=0.05, band="approve", bin_indices={},
    )
    assert int(store.fetch_recent_scores(1)[0]["id"]) == 1


def test_runs_are_recorded_with_their_outcome(store):
    store.record_run(None, 0, "waiting", "no new scored requests")
    store.record_run(1, 2000, "scored", "{}")
    outcomes = [row["outcome"] for row in store.fetch_runs()]
    assert set(outcomes) == {"waiting", "scored"}
