"""How much of the stability index is sampling noise, as a function of window size.

This is the evidence behind two config values that would otherwise be assumptions:
`monitoring.window_size` and `monitoring.min_window_size`.

The method is to draw repeatedly from the training reference distribution itself, so the
population is stable by construction and every index computed is pure sampling noise. Whatever
comes out is the floor below which a reading means nothing.

Writes reports/window_size_noise.csv.
"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config  # noqa: E402
from src.drift import proportions_from_counts, stability_index  # noqa: E402

WINDOW_SIZES = [100, 200, 500, 1000, 2000, 5000]
TRIALS = 500


def main() -> None:
    config = load_config()
    artifact = joblib.load(config.path(config.service.model_path))
    rng = np.random.default_rng(config.stream.seed)

    references = {
        name: np.asarray(binning.reference_proportions, dtype="float64")
        for name, binning in artifact.transformer.bins.items()
        if name in artifact.scorecard.features
    }
    references["score"] = np.asarray(artifact.reference_score_proportions, dtype="float64")
    references["band"] = np.asarray(artifact.reference_band_proportions, dtype="float64")

    rows = []
    for name, reference in references.items():
        for size in WINDOW_SIZES:
            values = np.array([
                stability_index(proportions_from_counts(rng.multinomial(size, reference)), reference)
                for _ in range(TRIALS)
            ])
            rows.append({
                "feature": name,
                "bins": len(reference),
                "window_size": size,
                "mean": values.mean(),
                "p50": np.median(values),
                "p95": np.percentile(values, 95),
                "max": values.max(),
                "share_over_warn": float((values >= config.monitoring.psi_warn).mean()),
                "share_over_alert": float((values >= config.monitoring.psi_alert).mean()),
            })

    frame = pd.DataFrame(rows)
    reports = config.path("reports")
    reports.mkdir(parents=True, exist_ok=True)
    frame.to_csv(reports / "window_size_noise.csv", index=False)

    summary = frame.groupby("window_size").agg(
        worst_p95=("p95", "max"),
        worst_max=("max", "max"),
        share_over_warn=("share_over_warn", "max"),
        share_over_alert=("share_over_alert", "max"),
    ).reset_index()

    print("Stability index on a population that has not changed, by window size")
    print("(worst case across every monitored characteristic, "
          f"{TRIALS} trials each)\n")
    print(summary.round(4).to_string(index=False))
    print(
        f"\nConfigured window size: {config.monitoring.window_size}, "
        f"minimum: {config.monitoring.min_window_size}, "
        f"warn: {config.monitoring.psi_warn}, alert: {config.monitoring.psi_alert}"
    )


if __name__ == "__main__":
    main()
