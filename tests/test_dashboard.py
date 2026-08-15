"""Tests that the dashboard script actually runs.

A dashboard that returns HTTP 200 has proved almost nothing, because Streamlit serves the page
shell before it ever executes the script. These use Streamlit's own test harness, which runs
the script the way a real session does and surfaces any exception it raises.

Both states are covered: with monitoring history present, and with an empty store, since the
empty case is what a reviewer sees if they open the dashboard before running the demo.
"""

from __future__ import annotations

import pytest

from src.config import load_config
from src.store import Store
from tests.conftest import PROJECT_ROOT, requires_model

pytestmark = requires_model

APP = str(PROJECT_ROOT / "dashboard" / "app.py")
TIMEOUT = 120


def run_app(monkeypatch, config_path):
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("CONFIG_PATH", str(config_path))
    app = AppTest.from_file(APP, default_timeout=TIMEOUT)
    app.run()
    return app


def test_dashboard_runs_against_an_empty_store(monkeypatch, temp_config):
    """Before the first run there is nothing to plot, and that must not be an error."""
    app = run_app(monkeypatch, temp_config)
    assert not app.exception, app.exception
    assert any("No monitoring windows yet" in str(item.value) for item in app.info)


def test_dashboard_renders_the_simulation_warning(monkeypatch, temp_config):
    app = run_app(monkeypatch, temp_config)
    assert not app.exception, app.exception
    warnings = " ".join(str(item.value) for item in app.warning)
    assert "Simulated traffic" in warnings
    assert "not production traffic" in warnings


def test_dashboard_runs_against_a_populated_store(monkeypatch, temp_config):
    """The real path: windows, metrics and a fired alert all present."""
    from src.monitor import MonitorRunner
    from tests.test_monitor import write_requests

    config = load_config(temp_config)
    config.monitoring.window_size = 200
    config.monitoring.min_window_size = 50
    runner = MonitorRunner(config)
    for seed in range(4):
        write_requests(runner, 200, shift=1.2, seed=seed)
    runner.drain(verbose=False)
    assert runner.store.count_alerts() > 0

    app = run_app(monkeypatch, temp_config)
    assert not app.exception, app.exception

    headings = " ".join(item.value for item in app.subheader)
    assert "Population and prediction stability" in headings
    assert "Characteristic stability" in headings
    assert "Alerts" in headings

    # The headline metrics should reflect the store rather than being placeholders.
    labels = {item.label: item.value for item in app.metric}
    assert labels["Monitoring windows"] == "4"
    assert int(labels["Alerts fired"]) > 0
    assert labels["Requests scored"] == "800"
