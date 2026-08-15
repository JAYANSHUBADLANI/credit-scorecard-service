"""Monitoring dashboard.

The point of this view is that drift is something to watch happen. The stability indices are
plotted against their thresholds over time, the characteristic level breakdown says which
input moved, and the alert table shows what fired and on the strength of which windows.

The banner at the top is not decoration. Anyone looking at this screen needs to know within a
second that the traffic underneath it is simulated.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config  # noqa: E402
from src.store import Store  # noqa: E402

st.set_page_config(page_title="Scorecard Drift Monitor", page_icon="📉", layout="wide")

STATUS_COLOURS = {"ok": "#2e7d32", "warn": "#ed6c02", "alert": "#c62828"}


# The paths are passed in rather than read inside these functions so that they form part of
# the cache key. A cached loader keyed on nothing returns whatever it read the first time,
# whichever store it was pointed at, which is wrong the moment more than one config exists.
@st.cache_resource
def get_config(config_path: str):
    return load_config(config_path or None)


@st.cache_data(ttl=5)
def load_frames(db_path: str):
    store = Store(db_path)
    metrics = pd.DataFrame([dict(row) for row in store.fetch_all_metrics()])
    summaries = pd.DataFrame([dict(row) for row in store.fetch_window_summaries()])
    alerts = pd.DataFrame([dict(row) for row in store.fetch_alerts()])
    runs = pd.DataFrame([dict(row) for row in store.fetch_runs(limit=200)])
    return metrics, summaries, alerts, runs, store.count_scores()


@st.cache_data(ttl=30)
def load_reference(reference_path: str):
    path = Path(reference_path)
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def threshold_band(figure, warn: float, alert: float, x_range) -> None:
    figure.add_hline(
        y=warn, line_dash="dot", line_color=STATUS_COLOURS["warn"],
        annotation_text=f"warn {warn:.2f}", annotation_position="top left",
    )
    figure.add_hline(
        y=alert, line_dash="dash", line_color=STATUS_COLOURS["alert"],
        annotation_text=f"alert {alert:.2f}", annotation_position="top left",
    )


def main() -> None:
    config = get_config(os.environ.get("CONFIG_PATH", ""))
    metrics, summaries, alerts, runs, total_scored = load_frames(
        str(config.path(config.service.db_path))
    )
    reference = load_reference(str(config.path(config.service.reference_path)))

    st.title("Credit scorecard drift monitor")
    st.warning(
        "**Simulated traffic.** Every application below is a real, unmodified record from the "
        "held out slice of the Home Credit Default Risk data, scored by the live API. The "
        "arrival order is constructed, and the later part of the stream deliberately reweights "
        "which real applications arrive in order to introduce a known shift. This is a "
        "documented simulation, not production traffic.",
        icon="⚠️",
    )

    if metrics.empty:
        st.info(
            "No monitoring windows yet. Run `make demo` to fit the card, start the API, post "
            "the simulated stream and compute the windows."
        )
        st.stop()

    # Headline numbers ------------------------------------------------------------------
    latest_window = int(metrics["window_id"].max())
    latest = metrics[metrics["window_id"] == latest_window]
    breaching = latest[latest["status"] == "alert"]

    columns = st.columns(5)
    columns[0].metric("Requests scored", f"{total_scored:,}")
    columns[1].metric("Monitoring windows", latest_window)
    columns[2].metric("Alerts fired", len(alerts))
    columns[3].metric(
        "Characteristics in breach", len(breaching), help="In the most recent window"
    )
    if not summaries.empty:
        approval = summaries.iloc[-1]["approval_rate"]
        first_approval = summaries.iloc[0]["approval_rate"]
        columns[4].metric(
            "Approval rate", f"{approval:.1%}", delta=f"{approval - first_approval:+.1%}"
        )

    if reference:
        trained = reference.get("trained_at", "unknown")
        st.caption(
            f"Model version {reference.get('model_version', 'unknown')}, fitted "
            f"{trained} on {reference.get('training_rows', 0):,} applications. "
            f"Window size {config.monitoring.window_size} requests, debounce "
            f"{config.monitoring.debounce_windows} consecutive windows, cooldown "
            f"{config.monitoring.cooldown_windows} windows."
        )

    # Population and prediction stability -----------------------------------------------
    st.subheader("Population and prediction stability")
    st.caption(
        "Score PSI asks whether the population the model sees has changed. Band PSI asks "
        "whether what it decides has changed. Both are measured against the training "
        "distribution captured when the card was fitted, never against recent traffic."
    )

    headline = metrics[metrics["metric"].isin(["psi_score", "psi_band"])]
    figure = go.Figure()
    labels = {"psi_score": "Score PSI (population)", "psi_band": "Band PSI (decision mix)"}
    for metric_name, group in headline.groupby("metric"):
        group = group.sort_values("window_id")
        figure.add_trace(
            go.Scatter(
                x=group["window_id"], y=group["value"], mode="lines+markers",
                name=labels.get(metric_name, metric_name),
                hovertemplate="window %{x}<br>index %{y:.4f}<extra></extra>",
            )
        )
    threshold_band(figure, config.monitoring.psi_warn, config.monitoring.psi_alert, None)

    for _, alert in alerts.iterrows():
        figure.add_vline(
            x=int(alert["window_id"]), line_color=STATUS_COLOURS["alert"], line_width=1,
            opacity=0.35,
        )
    figure.update_layout(
        xaxis_title="Monitoring window", yaxis_title="Stability index",
        height=380, hovermode="x unified", margin=dict(t=30, b=40),
    )
    st.plotly_chart(figure, use_container_width=True)

    # Characteristic level attribution ---------------------------------------------------
    st.subheader("Characteristic stability, which input moved")
    st.caption(
        "A population index that fires on its own says something changed. These say where, "
        "which is the difference between an alert someone can act on and one they ignore."
    )

    csi = metrics[metrics["metric"] == "csi"]
    pivot = csi.pivot_table(index="window_id", columns="feature", values="value")
    ranked = pivot.iloc[-1].sort_values(ascending=False)

    left, right = st.columns([2, 1])
    with left:
        default = list(ranked.head(5).index)
        chosen = st.multiselect(
            "Characteristics", options=list(pivot.columns), default=default,
        )
        csi_figure = go.Figure()
        for feature in chosen:
            csi_figure.add_trace(
                go.Scatter(
                    x=pivot.index, y=pivot[feature], mode="lines+markers", name=feature,
                    hovertemplate=f"{feature}<br>window %{{x}}<br>CSI %{{y:.4f}}<extra></extra>",
                )
            )
        threshold_band(csi_figure, config.monitoring.csi_warn, config.monitoring.csi_alert, None)
        csi_figure.update_layout(
            xaxis_title="Monitoring window", yaxis_title="Characteristic stability index",
            height=380, hovermode="x unified", margin=dict(t=30, b=40),
        )
        st.plotly_chart(csi_figure, use_container_width=True)

    with right:
        st.markdown(f"**Ranked by the latest window ({latest_window})**")
        table = ranked.reset_index()
        table.columns = ["characteristic", "csi"]
        table["status"] = table["csi"].apply(
            lambda v: "alert" if v >= config.monitoring.csi_alert
            else ("warn" if v >= config.monitoring.csi_warn else "ok")
        )
        st.dataframe(
            table.style.format({"csi": "{:.4f}"}).map(
                lambda v: f"color: {STATUS_COLOURS.get(v, '')}", subset=["status"]
            ),
            hide_index=True, use_container_width=True, height=340,
        )

    # What the service is deciding --------------------------------------------------------
    st.subheader("What the service is deciding")
    if not summaries.empty:
        left, right = st.columns(2)

        approval_figure = go.Figure()
        approval_figure.add_trace(
            go.Scatter(
                x=summaries["window_id"], y=summaries["approval_rate"],
                mode="lines+markers", name="Approval rate",
            )
        )
        approval_figure.add_trace(
            go.Scatter(
                x=summaries["window_id"], y=summaries["decline_rate"],
                mode="lines+markers", name="Decline rate",
            )
        )
        if reference.get("band"):
            bands = reference["band"]["bands"]
            proportions = reference["band"]["reference_proportions"]
            approval_figure.add_hline(
                y=proportions[bands.index("approve")], line_dash="dot",
                annotation_text="training approval rate", annotation_position="bottom left",
            )
        approval_figure.update_layout(
            xaxis_title="Monitoring window", yaxis_title="Share of window",
            height=340, hovermode="x unified", margin=dict(t=30, b=40),
        )
        left.plotly_chart(approval_figure, use_container_width=True)

        pd_figure = go.Figure()
        pd_figure.add_trace(
            go.Scatter(
                x=summaries["window_id"], y=summaries["mean_probability"],
                mode="lines+markers", name="Mean predicted PD",
            )
        )
        if reference.get("score", {}).get("mean_probability"):
            pd_figure.add_hline(
                y=reference["score"]["mean_probability"], line_dash="dot",
                annotation_text="training mean PD", annotation_position="bottom left",
            )
        pd_figure.update_layout(
            xaxis_title="Monitoring window", yaxis_title="Mean predicted probability of default",
            height=340, hovermode="x unified", margin=dict(t=30, b=40),
        )
        right.plotly_chart(pd_figure, use_container_width=True)

    # Alerts ------------------------------------------------------------------------------
    st.subheader("Alerts")
    st.caption(
        f"An alert requires the metric to breach its threshold for "
        f"{config.monitoring.debounce_windows} consecutive windows, so a single noisy window "
        f"never fires one. After firing, the same metric is suppressed for "
        f"{config.monitoring.cooldown_windows} windows while the breach is still recorded."
    )

    if alerts.empty:
        st.success("No alerts have fired.")
    else:
        display = alerts.copy()
        display["breach_windows"] = display["breach_window_ids"].apply(
            lambda raw: ", ".join(str(v) for v in json.loads(raw))
        )
        st.dataframe(
            display[[
                "fired_at", "window_id", "metric", "feature", "value", "threshold",
                "consecutive_windows", "breach_windows", "message",
            ]].rename(columns={"window_id": "fired_on_window"}),
            hide_index=True, use_container_width=True,
        )

    # Run history --------------------------------------------------------------------------
    with st.expander("Monitor run history"):
        st.caption(
            "Every wake up, including the ones that found nothing to do. A monitor that has "
            "silently stopped running looks exactly like a monitor with no alerts, and this "
            "is how the two are told apart."
        )
        if runs.empty:
            st.write("No runs recorded.")
        else:
            st.dataframe(
                runs[["ran_at", "window_id", "n_records", "outcome", "detail"]],
                hide_index=True, use_container_width=True, height=300,
            )

    with st.expander("Window detail"):
        if not summaries.empty:
            st.dataframe(summaries, hide_index=True, use_container_width=True)


if __name__ == "__main__":
    main()
