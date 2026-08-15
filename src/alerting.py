"""Threshold alerting with debounce.

The debounce is the part of this project I would most want to be asked about, so the
reasoning is written down rather than left in the code.

A single window crossing a threshold is not evidence of drift. With a window of a couple of
thousand requests, the stability index has real sampling variance, and a monitor that fires on
every crossing produces a stream of alerts that a risk function learns to ignore within a
month. An ignored alert is worse than no alert, because it still carries the implication that
someone is watching.

So an alert requires the metric to be in breach for `debounce_windows` consecutive windows.
A spike that reverts on the next window never fires. A genuine shift fires one window later
than it could have, and that delay is the price of the alert meaning something. The rule is
sustained breach, not average breach: taking a mean over the last three windows would let one
extreme window drag two quiet ones over the line, which is the failure the debounce exists to
prevent.

After firing, the same metric goes quiet for `cooldown_windows`. Drift does not repair
itself, so without a cooldown a sustained shift would re-fire every window for as long as it
lasted, which is the same alert fatigue arriving by a different route. The condition is still
recorded in the metric history throughout, so the dashboard shows it as continuously in
breach. What is suppressed is the repeat notification, not the observation.

State is never stored. Both the consecutive count and the cooldown are derived from the
metric and alert history each time, so a restarted monitor reaches the same conclusion as one
that has been up for a week, and the logic can be tested by writing rows rather than by
stubbing an object.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

from .drift import METRIC_CSI, METRIC_PSI_BAND, METRIC_PSI_SCORE, STATUS_ALERT, MetricResult

SEVERITY_ALERT = "alert"

# The population level metrics. These are the ones that raise an alert in their own right.
POPULATION_METRICS = {METRIC_PSI_SCORE, METRIC_PSI_BAND}


def is_attribution_only(metric: str, population_in_breach: bool) -> bool:
    """Whether a breaching characteristic should be recorded as attribution, not as an alert.

    When the population itself has shifted, the characteristics that moved are the explanation
    for that one event, not fifteen separate events. Firing an alert for each of them is the
    alert fatigue problem restated: a monitor that sends sixteen notifications for one shift
    gets muted just as fast as one that fires on every noisy window. So a characteristic
    breach is folded into the population alert as attribution while a population metric is
    also in breach.

    When the population metrics are quiet and a single characteristic has moved, that is a
    different situation and it does raise on its own. It is the shape a broken or stale feed
    for one input takes: the characteristic shifts hard, the score barely moves because the
    other fourteen dilute it, and nothing at the population level ever notices.
    """
    return metric == METRIC_CSI and population_in_breach


@dataclass
class DebounceConfig:
    debounce_windows: int = 3
    cooldown_windows: int = 5


@dataclass
class AlertDecision:
    """The outcome of evaluating one metric in one window, and why."""

    metric: str
    feature: str
    value: float
    threshold: float
    window_id: int
    should_fire: bool
    reason: str
    consecutive_windows: int
    breach_window_ids: List[int]

    def message(self) -> str:
        return (
            f"{self.metric} for '{self.feature}' held at or above {self.threshold:.2f} for "
            f"{self.consecutive_windows} consecutive windows, reaching {self.value:.4f} in "
            f"window {self.window_id}"
        )


def consecutive_breach_run(
    statuses_newest_first: Sequence[str], breaching: str = STATUS_ALERT
) -> int:
    """Length of the unbroken run of breaching windows ending at the most recent one.

    Counting stops at the first window that was not in breach. A gap resets the run, which is
    what makes this a sustained breach test rather than a count of breaches in a period.
    """
    run = 0
    for status in statuses_newest_first:
        if status != breaching:
            break
        run += 1
    return run


def evaluate_alert(
    result: MetricResult,
    window_id: int,
    history_newest_first: Sequence[Dict[str, object]],
    config: DebounceConfig,
    last_alert_window: Optional[int] = None,
) -> AlertDecision:
    """Decide whether one metric in one window should raise an alert.

    `history_newest_first` holds the previous windows for this same metric and feature, most
    recent first, each with at least `status` and `window_id`. The current window is not
    expected to be in it.
    """
    statuses = [str(row["status"]) for row in history_newest_first]
    window_ids = [int(row["window_id"]) for row in history_newest_first]

    if result.status != STATUS_ALERT:
        prior_run = consecutive_breach_run(statuses)
        return AlertDecision(
            metric=result.metric,
            feature=result.feature,
            value=result.value,
            threshold=result.alert_threshold,
            window_id=window_id,
            should_fire=False,
            reason=(
                f"window is {result.status}, below the alert threshold of "
                f"{result.alert_threshold:.2f}"
                + (f", clearing a run of {prior_run}" if prior_run else "")
            ),
            consecutive_windows=0,
            breach_window_ids=[],
        )

    prior_run = consecutive_breach_run(statuses)
    run = prior_run + 1
    breach_ids = [window_id] + window_ids[:prior_run]
    breach_ids.sort()

    if run < config.debounce_windows:
        return AlertDecision(
            metric=result.metric,
            feature=result.feature,
            value=result.value,
            threshold=result.alert_threshold,
            window_id=window_id,
            should_fire=False,
            reason=(
                f"in breach for {run} of the {config.debounce_windows} consecutive windows "
                "required, holding"
            ),
            consecutive_windows=run,
            breach_window_ids=breach_ids,
        )

    if last_alert_window is not None:
        windows_since = window_id - last_alert_window
        if windows_since < config.cooldown_windows:
            return AlertDecision(
                metric=result.metric,
                feature=result.feature,
                value=result.value,
                threshold=result.alert_threshold,
                window_id=window_id,
                should_fire=False,
                reason=(
                    f"still in breach, but suppressed by the cooldown: {windows_since} of "
                    f"{config.cooldown_windows} windows since the last alert in window "
                    f"{last_alert_window}"
                ),
                consecutive_windows=run,
                breach_window_ids=breach_ids,
            )

    return AlertDecision(
        metric=result.metric,
        feature=result.feature,
        value=result.value,
        threshold=result.alert_threshold,
        window_id=window_id,
        should_fire=True,
        reason=(
            f"sustained breach across {run} consecutive windows, meeting the debounce "
            f"requirement of {config.debounce_windows}"
        ),
        consecutive_windows=run,
        breach_window_ids=breach_ids,
    )
