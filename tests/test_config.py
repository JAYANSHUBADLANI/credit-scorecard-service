"""Tests for how the config resolves its project root.

Every relative path in config.yaml, the SQLite store, the fitted artifact, the reports, is
resolved against a root that used to be inferred as the config file's grandparent without
checking anything. A config placed anywhere else therefore pointed the whole system at a tree
that did not exist, created it on first write, and reported nothing. Nothing failed, the run
just happened somewhere else, which is the hardest kind of wrong to notice.

These pin the three ways the root can now be settled, and pin that a layout it cannot read is
an error rather than a guess.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from src.config import load_config
from tests.conftest import REAL_CONFIG


def write_config(directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    raw = yaml.safe_load(REAL_CONFIG.read_text(encoding="utf-8"))
    path = directory / "config.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return path


# The documented layout ------------------------------------------------------------------

def test_the_documented_layout_resolves_to_its_own_tree(tmp_path):
    path = write_config(tmp_path / "config")
    config = load_config(path)
    assert config.root == tmp_path.resolve()
    assert config.path("data/monitoring.db") == tmp_path.resolve() / "data" / "monitoring.db"


def test_an_absolute_path_in_the_config_is_left_alone(tmp_path):
    path = write_config(tmp_path / "config")
    config = load_config(path)
    assert config.path("/var/lib/scorecard.db") == Path("/var/lib/scorecard.db")


# A layout it cannot read ------------------------------------------------------------------

def test_a_config_outside_a_config_directory_is_refused_rather_than_guessed(tmp_path):
    """The regression. This used to resolve silently against tmp_path's parent."""
    path = write_config(tmp_path)
    with pytest.raises(ValueError) as error:
        load_config(path)
    message = str(error.value)
    assert "project root" in message
    assert "root=" in message, "the error should name the way out, not just the problem"


# Stating it explicitly ---------------------------------------------------------------------

def test_an_explicit_root_overrides_the_layout(tmp_path):
    path = write_config(tmp_path / "somewhere")
    config = load_config(path, root=tmp_path)
    assert config.root == tmp_path.resolve()
    assert config.path("models/scorecard.joblib").parent.parent == tmp_path.resolve()


def test_the_project_root_environment_variable_is_honoured(tmp_path, monkeypatch):
    path = write_config(tmp_path / "somewhere")
    monkeypatch.setenv("PROJECT_ROOT", str(tmp_path))
    config = load_config(path)
    assert config.root == tmp_path.resolve()


def test_an_explicit_root_wins_over_the_environment(tmp_path, monkeypatch):
    other = tmp_path / "other"
    other.mkdir()
    path = write_config(tmp_path / "config")
    monkeypatch.setenv("PROJECT_ROOT", str(other))
    assert load_config(path, root=tmp_path).root == tmp_path.resolve()
