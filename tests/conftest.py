"""Shared fixtures.

Tests that need the API point it at a throwaway SQLite file through a temporary config, so a
test run never writes into the monitoring store that the documented end to end run produced.
The fitted artifact is reused rather than refitted, because refitting on 215,000 rows per test
session would make the suite slow enough that it stops being run.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REAL_CONFIG = PROJECT_ROOT / "config" / "config.yaml"
REAL_MODEL = PROJECT_ROOT / "models" / "scorecard.joblib"

requires_model = pytest.mark.skipif(
    not REAL_MODEL.exists(),
    reason="no fitted artifact, run `make train` first",
)


@pytest.fixture
def temp_config(tmp_path: Path) -> Path:
    """A config identical to the real one except that it writes to a temporary database."""
    with open(REAL_CONFIG, "r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)

    raw["service"]["db_path"] = str(tmp_path / "test.db")
    raw["service"]["model_path"] = str(REAL_MODEL)

    config_dir = tmp_path / "config"
    config_dir.mkdir()
    path = config_dir / "config.yaml"
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(raw, handle)

    # load_config resolves the project root as the config file's grandparent, so the temporary
    # tree needs the same shape as the real one.
    for folder in ["models", "data", "reports"]:
        (tmp_path / folder).mkdir(exist_ok=True)
    return path


@pytest.fixture
def api_client(temp_config, monkeypatch):
    """A TestClient bound to the temporary config."""
    from fastapi.testclient import TestClient

    monkeypatch.setenv("CONFIG_PATH", str(temp_config))
    from src import api as api_module

    with TestClient(api_module.app) as client:
        yield client


@pytest.fixture
def valid_payload() -> dict:
    """A complete, in range application. Values are within the observed data ranges."""
    return {
        "EXT_SOURCE_1": 0.52,
        "EXT_SOURCE_2": 0.61,
        "EXT_SOURCE_3": 0.48,
        "DAYS_BIRTH": -14000.0,
        "DAYS_EMPLOYED": -2200.0,
        "DAYS_ID_PUBLISH": -3500.0,
        "DAYS_LAST_PHONE_CHANGE": -900.0,
        "AMT_INCOME_TOTAL": 180000.0,
        "AMT_CREDIT": 600000.0,
        "AMT_ANNUITY": 27000.0,
        "AMT_GOODS_PRICE": 540000.0,
        "REGION_POPULATION_RELATIVE": 0.018,
        "NAME_EDUCATION_TYPE": "Higher education",
        "NAME_INCOME_TYPE": "Working",
        "NAME_CONTRACT_TYPE": "Cash loans",
    }
