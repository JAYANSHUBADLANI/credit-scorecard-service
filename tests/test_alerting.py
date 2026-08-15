"""Tests for the debounce and cooldown rules.

This is the file to read first if you want to know whether the alerting actually behaves the
way the README claims. The rules under test are:

  1. A single breaching window does not fire.
  2. A breach that reverts before reaching the debounce length never fires.
  3. A sustained breach fires on exactly the window that completes the run, not before.
  4. A gap resets the run. Three breaches with a quiet window in between are not a sustained
     breach, and this is the case a naive "count breaches in the last N windows" gets wrong.
  5. After firing, the cooldown suppresses repeats, and the alert can fire again once it
     expires.
"""

from __future__ import annotations

import pytest

from src.alerting import (
    POPULATION_METRICS,
    AlertDecision,
    DebounceConfig,
    consecutive_breach_run,
    evaluate_alert,
    is_attribution_only,
)
from src.drift import (
    METRIC_CSI,
    METRIC_PSI_BAND,
    METRIC_PSI_SCORE,
    STATUS_ALERT,
    STATUS_OK,
    STATUS_WARN,
    MetricResult,
)

CONFIG = DebounceConfig(debounce_windows=3, cooldown_windows=5)


def breaching_metric(value: float = 0.40) -> MetricResult:
    return MetricResult(
        metric="psi_score",
        feature="score",
        value=value,
        warn_threshold=0.10,
        alert_threshold=0.25,
        status=STATUS_ALERT,
    )


def quiet_metric(value: float = 0.02) -> MetricResult:
    return MetricResult(
        metric="psi_score",
        feature="score",
        value=value,
        warn_threshold=0.10,
        alert_threshold=0.25,
        status=STATUS_OK,
    )


def history(statuses, first_window: int = 1):
    """Build a history newest first, the order the store returns and the checker expects."""
    rows = [
        {"status": status, "window_id": first_window + i} for i, status in enumerate(statuses)
    ]
    return list(reversed(rows))


# The run counter --------------------------------------------------------------------

def test_run_counts_only_the_unbroken_tail():
    assert consecutive_breach_run([]) == 0
    assert consecutive_breach_run([STATUS_ALERT]) == 1
    assert consecutive_breach_run([STATUS_ALERT, STATUS_ALERT]) == 2
    assert consecutive_breach_run([STATUS_ALERT, STATUS_OK, STATUS_ALERT]) == 1


def test_warn_does_not_count_as_a_breach():
    """A warn window sits between the thresholds and must break the run, not extend it."""
    assert consecutive_breach_run([STATUS_ALERT, STATUS_WARN, STATUS_ALERT]) == 1


# Debounce ---------------------------------------------------------------------------

def test_single_breach_does_not_fire():
    decision = evaluate_alert(breaching_metric(), 1, history([]), CONFIG)
    assert decision.should_fire is False
    assert decision.consecutive_windows == 1
    assert "1 of the 3" in decision.reason


def test_second_consecutive_breach_still_does_not_fire():
    decision = evaluate_alert(
        breaching_metric(), 2, history([STATUS_ALERT]), CONFIG
    )
    assert decision.should_fire is False
    assert decision.consecutive_windows == 2


def test_third_consecutive_breach_fires():
    decision = evaluate_alert(
        breaching_metric(0.42), 3, history([STATUS_ALERT, STATUS_ALERT]), CONFIG
    )
    assert decision.should_fire is True
    assert decision.consecutive_windows == 3
    assert decision.breach_window_ids == [1, 2, 3]
    assert "0.42" in decision.message()


def test_spike_that_reverts_never_fires():
    """The case the debounce exists for: one loud window surrounded by quiet ones."""
    first = evaluate_alert(breaching_metric(), 1, history([]), CONFIG)
    assert first.should_fire is False

    reverted = evaluate_alert(quiet_metric(), 2, history([STATUS_ALERT]), CONFIG)
    assert reverted.should_fire is False
    assert reverted.consecutive_windows == 0

    next_breach = evaluate_alert(
        breaching_metric(), 3, history([STATUS_ALERT, STATUS_OK]), CONFIG
    )
    assert next_breach.should_fire is False
    assert next_breach.consecutive_windows == 1


def test_gap_resets_the_run_even_with_three_breaches_in_four_windows():
    """Three of the last four windows breached, but not consecutively, so nothing fires.

    A monitor that counted breaches over a trailing period rather than requiring them to be
    consecutive would fire here. That is the distinction the rule is making.
    """
    decision = evaluate_alert(
        breaching_metric(),
        4,
        history([STATUS_ALERT, STATUS_OK, STATUS_ALERT]),
        CONFIG,
    )
    assert decision.should_fire is False
    assert decision.consecutive_windows == 2


def test_longer_debounce_requirement_delays_the_alert():
    strict = DebounceConfig(debounce_windows=5, cooldown_windows=5)
    statuses = [STATUS_ALERT] * 3
    decision = evaluate_alert(breaching_metric(), 4, history(statuses), strict)
    assert decision.should_fire is False
    assert decision.consecutive_windows == 4

    decision = evaluate_alert(
        breaching_metric(), 5, history([STATUS_ALERT] * 4), strict
    )
    assert decision.should_fire is True


def test_debounce_of_one_fires_immediately():
    """The degenerate configuration is still coherent: no debounce means fire on first breach."""
    immediate = DebounceConfig(debounce_windows=1, cooldown_windows=0)
    decision = evaluate_alert(breaching_metric(), 1, history([]), immediate)
    assert decision.should_fire is True


# Cooldown ---------------------------------------------------------------------------

def test_cooldown_suppresses_a_repeat_while_still_in_breach():
    decision = evaluate_alert(
        breaching_metric(),
        4,
        history([STATUS_ALERT] * 3),
        CONFIG,
        last_alert_window=3,
    )
    assert decision.should_fire is False
    assert "cooldown" in decision.reason
    assert decision.consecutive_windows == 4


def test_alert_fires_again_once_the_cooldown_expires():
    decision = evaluate_alert(
        breaching_metric(),
        8,
        history([STATUS_ALERT] * 7),
        CONFIG,
        last_alert_window=3,
    )
    assert decision.should_fire is True
    assert decision.consecutive_windows == 8


def test_cooldown_boundary_is_exclusive_on_the_last_suppressed_window():
    """Four windows after firing is still suppressed, five is not, with cooldown_windows=5."""
    suppressed = evaluate_alert(
        breaching_metric(), 7, history([STATUS_ALERT] * 6), CONFIG, last_alert_window=3
    )
    assert suppressed.should_fire is False

    released = evaluate_alert(
        breaching_metric(), 8, history([STATUS_ALERT] * 7), CONFIG, last_alert_window=3
    )
    assert released.should_fire is True


# The audit trail --------------------------------------------------------------------

def test_decision_records_the_windows_that_justified_it():
    """An alert has to be defensible after the fact, so it names the windows behind it."""
    decision = evaluate_alert(
        breaching_metric(0.31), 6, history([STATUS_ALERT, STATUS_ALERT], first_window=4), CONFIG
    )
    assert decision.should_fire is True
    assert decision.breach_window_ids == [4, 5, 6]
    message = decision.message()
    assert "psi_score" in message
    assert "3 consecutive windows" in message
    assert "0.25" in message


def test_quiet_window_reports_that_it_cleared_a_run():
    decision = evaluate_alert(quiet_metric(), 3, history([STATUS_ALERT, STATUS_ALERT]), CONFIG)
    assert decision.should_fire is False
    assert "clearing a run of 2" in decision.reason


# The attribution tier ------------------------------------------------------------------

def test_population_metrics_always_alert_in_their_own_right():
    assert POPULATION_METRICS == {METRIC_PSI_SCORE, METRIC_PSI_BAND}
    assert is_attribution_only(METRIC_PSI_SCORE, population_in_breach=True) is False
    assert is_attribution_only(METRIC_PSI_BAND, population_in_breach=True) is False


def test_characteristic_breach_is_attribution_while_the_population_is_also_breaching():
    """One shift moving fifteen characteristics is one event, not fifteen."""
    assert is_attribution_only(METRIC_CSI, population_in_breach=True) is True


def test_characteristic_breach_alerts_on_its_own_when_the_population_is_quiet():
    """The broken feed case: one input moves, the score barely does, and it still raises."""
    assert is_attribution_only(METRIC_CSI, population_in_breach=False) is False
