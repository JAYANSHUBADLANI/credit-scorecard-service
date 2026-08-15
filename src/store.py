"""SQLite store for scoring history, drift metrics and alerts.

SQLite is chosen because the brief is a monitoring system, not a database exercise, and
because it keeps the whole stack to one file on a shared volume that three containers can
open. It is run in write ahead logging mode so the API can keep appending while the monitor
reads a window and the dashboard reads both, which is the actual concurrency pattern here.
That is enough for a few thousand requests a minute. It would not be enough for a real
production scorecard, and the README says so rather than implying otherwise.

Two design points worth stating, because both come up when this is walked through:

The scoring log stores the bin index each characteristic fell into, alongside the score. That
is deliberate. Recomputing bins later from stored raw values would mean the monitoring layer
could disagree with what the model actually did, and the whole point of the characteristic
stability index is to explain a score shift that already happened.

There is no table of debounce counters. Consecutive breaches are derived from the metric
history at evaluation time. State that only exists in a counter column is state that is wrong
after a restart, and it cannot be tested without stubbing the store.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional

SCHEMA = """
CREATE TABLE IF NOT EXISTS scoring_log (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    request_id      TEXT    NOT NULL,
    scored_at       TEXT    NOT NULL,
    model_version   TEXT    NOT NULL,
    source          TEXT    NOT NULL,
    score           REAL    NOT NULL,
    probability     REAL    NOT NULL,
    band            TEXT    NOT NULL,
    bin_indices     TEXT    NOT NULL,
    latency_ms      REAL
);
CREATE INDEX IF NOT EXISTS idx_scoring_log_scored_at ON scoring_log (scored_at);

CREATE TABLE IF NOT EXISTS drift_metrics (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    window_id       INTEGER NOT NULL,
    computed_at     TEXT    NOT NULL,
    window_start_id INTEGER NOT NULL,
    window_end_id   INTEGER NOT NULL,
    n_records       INTEGER NOT NULL,
    metric          TEXT    NOT NULL,
    feature         TEXT    NOT NULL,
    value           REAL    NOT NULL,
    warn_threshold  REAL    NOT NULL,
    alert_threshold REAL    NOT NULL,
    status          TEXT    NOT NULL,
    UNIQUE (window_id, metric, feature)
);
CREATE INDEX IF NOT EXISTS idx_drift_window ON drift_metrics (window_id);

CREATE TABLE IF NOT EXISTS alerts (
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
    message             TEXT    NOT NULL,
    attributed_features TEXT    NOT NULL DEFAULT '[]'
);
CREATE INDEX IF NOT EXISTS idx_alerts_window ON alerts (window_id);

CREATE TABLE IF NOT EXISTS window_summary (
    window_id       INTEGER PRIMARY KEY,
    computed_at     TEXT    NOT NULL,
    window_start_id INTEGER NOT NULL,
    window_end_id   INTEGER NOT NULL,
    n_records       INTEGER NOT NULL,
    mean_score      REAL    NOT NULL,
    mean_probability REAL   NOT NULL,
    approval_rate   REAL    NOT NULL,
    decline_rate    REAL    NOT NULL,
    band_counts     TEXT    NOT NULL,
    first_scored_at TEXT,
    last_scored_at  TEXT
);

CREATE TABLE IF NOT EXISTS monitor_runs (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    ran_at       TEXT    NOT NULL,
    window_id    INTEGER,
    n_records    INTEGER NOT NULL,
    outcome      TEXT    NOT NULL,
    detail       TEXT
);
"""


# Columns added after the first version of the schema. `CREATE TABLE IF NOT EXISTS` does
# nothing to a table that already exists, so a database created by an earlier version keeps
# its old shape and every insert naming a new column fails. The monitoring store is meant to
# be long lived and to survive redeploys, which is exactly the situation where this bites, so
# additive columns are applied explicitly at startup.
MIGRATIONS: Dict[str, Dict[str, str]] = {
    "alerts": {"attributed_features": "TEXT NOT NULL DEFAULT '[]'"},
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


class Store:
    """Thin wrapper over the SQLite file. One instance per process is enough."""

    def __init__(self, path: Path | str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path, timeout=30.0, isolation_level=None)
        connection.row_factory = sqlite3.Row
        try:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("PRAGMA synchronous=NORMAL")
            connection.execute("PRAGMA busy_timeout=30000")
            yield connection
        finally:
            connection.close()

    def _init_schema(self) -> None:
        with self.connect() as connection:
            connection.executescript(SCHEMA)
            self._migrate(connection)

    @staticmethod
    def _migrate(connection: sqlite3.Connection) -> List[str]:
        """Add any column introduced after the database was first created.

        Additive only, and deliberately so. Anything beyond adding a nullable or defaulted
        column needs a considered migration rather than something that runs silently on every
        process start. Returns what it changed, so the behaviour is testable.
        """
        applied: List[str] = []
        for table, columns in MIGRATIONS.items():
            present = {
                row["name"] for row in connection.execute(f"PRAGMA table_info({table})").fetchall()
            }
            if not present:
                continue
            for column, definition in columns.items():
                if column not in present:
                    connection.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")
                    applied.append(f"{table}.{column}")
        return applied

    # Scoring log ---------------------------------------------------------------

    def log_score(
        self,
        request_id: str,
        model_version: str,
        source: str,
        score: float,
        probability: float,
        band: str,
        bin_indices: Dict[str, int],
        latency_ms: Optional[float] = None,
        scored_at: Optional[str] = None,
    ) -> int:
        with self.connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO scoring_log
                    (request_id, scored_at, model_version, source, score, probability,
                     band, bin_indices, latency_ms)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    request_id,
                    scored_at or utc_now(),
                    model_version,
                    source,
                    float(score),
                    float(probability),
                    band,
                    json.dumps(bin_indices, separators=(",", ":")),
                    latency_ms,
                ),
            )
            return int(cursor.lastrowid)

    def log_scores_bulk(self, rows: Iterable[Dict[str, Any]]) -> int:
        """Append many scored records in one transaction, used by the stream replay."""
        payload = [
            (
                row["request_id"],
                row.get("scored_at") or utc_now(),
                row["model_version"],
                row.get("source", "stream"),
                float(row["score"]),
                float(row["probability"]),
                row["band"],
                json.dumps(row["bin_indices"], separators=(",", ":")),
                row.get("latency_ms"),
            )
            for row in rows
        ]
        if not payload:
            return 0
        with self.connect() as connection:
            connection.execute("BEGIN")
            connection.executemany(
                """
                INSERT INTO scoring_log
                    (request_id, scored_at, model_version, source, score, probability,
                     band, bin_indices, latency_ms)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                payload,
            )
            connection.execute("COMMIT")
        return len(payload)

    def count_scores(self) -> int:
        with self.connect() as connection:
            row = connection.execute("SELECT COUNT(*) AS n FROM scoring_log").fetchone()
            return int(row["n"])

    def max_scored_id(self) -> int:
        with self.connect() as connection:
            row = connection.execute("SELECT COALESCE(MAX(id), 0) AS n FROM scoring_log").fetchone()
            return int(row["n"])

    def fetch_scores_after(self, after_id: int, limit: int) -> List[sqlite3.Row]:
        """Rows appended since `after_id`, oldest first. This is how a window is claimed."""
        with self.connect() as connection:
            return connection.execute(
                "SELECT * FROM scoring_log WHERE id > ? ORDER BY id ASC LIMIT ?",
                (after_id, limit),
            ).fetchall()

    def fetch_recent_scores(self, limit: int = 5000) -> List[sqlite3.Row]:
        with self.connect() as connection:
            return connection.execute(
                "SELECT * FROM scoring_log ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()

    def fetch_latencies(self) -> List[float]:
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT latency_ms FROM scoring_log WHERE latency_ms IS NOT NULL"
            ).fetchall()
            return [row["latency_ms"] for row in rows]

    # Drift metrics -------------------------------------------------------------

    def record_metrics(self, rows: Iterable[Dict[str, Any]]) -> int:
        payload = [
            (
                int(row["window_id"]),
                row.get("computed_at") or utc_now(),
                int(row["window_start_id"]),
                int(row["window_end_id"]),
                int(row["n_records"]),
                row["metric"],
                row["feature"],
                float(row["value"]),
                float(row["warn_threshold"]),
                float(row["alert_threshold"]),
                row["status"],
            )
            for row in rows
        ]
        if not payload:
            return 0
        with self.connect() as connection:
            connection.execute("BEGIN")
            connection.executemany(
                """
                INSERT OR REPLACE INTO drift_metrics
                    (window_id, computed_at, window_start_id, window_end_id, n_records,
                     metric, feature, value, warn_threshold, alert_threshold, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                payload,
            )
            connection.execute("COMMIT")
        return len(payload)

    def last_window_id(self) -> int:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT COALESCE(MAX(window_id), 0) AS n FROM drift_metrics"
            ).fetchone()
            return int(row["n"])

    def last_window_end_id(self) -> int:
        """Highest scoring_log id already consumed by a monitoring window."""
        with self.connect() as connection:
            row = connection.execute(
                "SELECT COALESCE(MAX(window_end_id), 0) AS n FROM drift_metrics"
            ).fetchone()
            return int(row["n"])

    def fetch_metric_history(
        self, metric: str, feature: str, limit: int = 50
    ) -> List[sqlite3.Row]:
        """Most recent windows first, which is the order the debounce check walks."""
        with self.connect() as connection:
            return connection.execute(
                """
                SELECT * FROM drift_metrics
                WHERE metric = ? AND feature = ?
                ORDER BY window_id DESC LIMIT ?
                """,
                (metric, feature, limit),
            ).fetchall()

    def fetch_all_metrics(self) -> List[sqlite3.Row]:
        with self.connect() as connection:
            return connection.execute(
                "SELECT * FROM drift_metrics ORDER BY window_id ASC, metric, feature"
            ).fetchall()

    def record_window_summary(self, summary: Dict[str, Any]) -> None:
        with self.connect() as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO window_summary
                    (window_id, computed_at, window_start_id, window_end_id, n_records,
                     mean_score, mean_probability, approval_rate, decline_rate, band_counts,
                     first_scored_at, last_scored_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    int(summary["window_id"]),
                    summary.get("computed_at") or utc_now(),
                    int(summary["window_start_id"]),
                    int(summary["window_end_id"]),
                    int(summary["n_records"]),
                    float(summary["mean_score"]),
                    float(summary["mean_probability"]),
                    float(summary["approval_rate"]),
                    float(summary["decline_rate"]),
                    json.dumps(summary["band_counts"], separators=(",", ":")),
                    summary.get("first_scored_at"),
                    summary.get("last_scored_at"),
                ),
            )

    def fetch_window_summaries(self) -> List[sqlite3.Row]:
        with self.connect() as connection:
            return connection.execute(
                "SELECT * FROM window_summary ORDER BY window_id ASC"
            ).fetchall()

    # Alerts --------------------------------------------------------------------

    def record_alert(
        self,
        window_id: int,
        metric: str,
        feature: str,
        value: float,
        threshold: float,
        severity: str,
        consecutive_windows: int,
        breach_window_ids: List[int],
        message: str,
        fired_at: Optional[str] = None,
        attributed_features: Optional[List[str]] = None,
    ) -> int:
        with self.connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO alerts
                    (fired_at, window_id, metric, feature, value, threshold, severity,
                     consecutive_windows, breach_window_ids, message, attributed_features)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    fired_at or utc_now(),
                    int(window_id),
                    metric,
                    feature,
                    float(value),
                    float(threshold),
                    severity,
                    int(consecutive_windows),
                    json.dumps(breach_window_ids),
                    message,
                    json.dumps(attributed_features or []),
                ),
            )
            return int(cursor.lastrowid)

    def last_alert_window(self, metric: str, feature: str) -> Optional[int]:
        """Window id of the most recent alert for this metric, used for the cooldown."""
        with self.connect() as connection:
            row = connection.execute(
                """
                SELECT MAX(window_id) AS n FROM alerts WHERE metric = ? AND feature = ?
                """,
                (metric, feature),
            ).fetchone()
            return None if row["n"] is None else int(row["n"])

    def fetch_alerts(self) -> List[sqlite3.Row]:
        with self.connect() as connection:
            return connection.execute("SELECT * FROM alerts ORDER BY id ASC").fetchall()

    def count_alerts(self) -> int:
        with self.connect() as connection:
            return int(connection.execute("SELECT COUNT(*) AS n FROM alerts").fetchone()["n"])

    # Monitor runs --------------------------------------------------------------

    def record_run(
        self, window_id: Optional[int], n_records: int, outcome: str, detail: str = ""
    ) -> None:
        with self.connect() as connection:
            connection.execute(
                "INSERT INTO monitor_runs (ran_at, window_id, n_records, outcome, detail) "
                "VALUES (?, ?, ?, ?, ?)",
                (utc_now(), window_id, int(n_records), outcome, detail),
            )

    def fetch_runs(self, limit: int = 100) -> List[sqlite3.Row]:
        with self.connect() as connection:
            return connection.execute(
                "SELECT * FROM monitor_runs ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()

    def reset(self) -> None:
        """Drop all monitoring state. Used by the end to end run so a rerun is reproducible."""
        with self.connect() as connection:
            connection.executescript(
                "DELETE FROM scoring_log; DELETE FROM drift_metrics; DELETE FROM alerts; "
                "DELETE FROM monitor_runs; DELETE FROM window_summary; "
                "DELETE FROM sqlite_sequence WHERE name IN "
                "('scoring_log','drift_metrics','alerts','monitor_runs');"
            )
