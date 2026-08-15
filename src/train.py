"""Phase 1: fit the scorecard and freeze the artifact the service loads.

This is a refit of an already established card, not new modelling work. What matters here is
what comes out the other side: one versioned artifact holding the model, the bins it was
fitted with, the band cutoffs and the training time reference distributions that every later
monitoring window is compared against. Those reference distributions are captured at fit time
on purpose. A baseline recomputed later from recent traffic would drift along with the
traffic and would never raise anything.

The held out slice written here is never used in fitting. It is the raw material for the
simulated scoring stream in Phase 3.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from .binning import BinningConfig, WOETransformer
from .config import Config, load_config
from .features import RAW_CATEGORICAL_INPUTS, RAW_INPUTS, build_features
from .drift import BAND_ORDER
from .scorecard import BandCutoffs, ScalingConfig, Scorecard, ScorecardArtifact, select_features

MODEL_VERSION = "1.0.0"


def load_raw(config: Config) -> pd.DataFrame:
    path = config.path(config.data.raw_path)
    if not path.exists():
        raise FileNotFoundError(
            f"raw data not found at {path}. See README, 'Getting the data', for the "
            "Home Credit Default Risk download step."
        )
    columns = [config.data.id_column, config.data.target_column] + RAW_INPUTS
    return pd.read_csv(path, usecols=columns)


def split_by_sequence(frame: pd.DataFrame, config: Config) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split into a training slice and a genuinely held out later slice.

    The ordering column is a documented proxy for application sequence, not a real
    origination date. The dataset has no date column. What is true regardless of that proxy
    is that no row in the held out slice takes any part in fitting.
    """
    ordered = frame.sort_values(config.split.order_by).reset_index(drop=True)
    cut = int(len(ordered) * config.split.train_fraction)
    return ordered.iloc[:cut].copy(), ordered.iloc[cut:].copy()


def gini(y_true: np.ndarray, score: np.ndarray) -> float:
    return float(2.0 * roc_auc_score(y_true, score) - 1.0)


def ks_statistic(y_true: np.ndarray, probability: np.ndarray) -> float:
    order = np.argsort(probability)
    labels = np.asarray(y_true)[order]
    bads = np.cumsum(labels) / max(labels.sum(), 1)
    goods = np.cumsum(1 - labels) / max((1 - labels).sum(), 1)
    return float(np.max(np.abs(bads - goods)))


def reference_score_distribution(scores: np.ndarray, n_bins: int) -> Tuple[np.ndarray, np.ndarray]:
    """Cut the training score distribution into equal count bins and record their shares.

    Prediction drift is measured against these fixed edges. Because the edges are quantiles of
    the training scores, a stable population lands roughly 1/n_bins in each, and any departure
    from that is directly readable as a shift in what the model is producing.
    """
    quantiles = np.linspace(0, 1, n_bins + 1)[1:-1]
    edges = np.unique(np.quantile(scores, quantiles))
    index = np.searchsorted(edges, scores, side="right")
    counts = np.bincount(index, minlength=len(edges) + 1).astype("float64")
    return edges, counts / counts.sum()


def band_summary(frame: pd.DataFrame, target: np.ndarray) -> pd.DataFrame:
    """Population share and bad rate per decision band, which is what a credit policy
    discussion actually turns on."""
    table = frame.copy()
    table["target"] = target
    grouped = table.groupby("band", observed=True).agg(
        applications=("target", "size"),
        bads=("target", "sum"),
        mean_score=("score", "mean"),
        mean_probability=("probability", "mean"),
    )
    grouped["population_share"] = grouped["applications"] / grouped["applications"].sum()
    grouped["bad_rate"] = grouped["bads"] / grouped["applications"]
    order = ["decline", "refer", "approve"]
    return grouped.reindex([b for b in order if b in grouped.index]).reset_index()


def train(config: Config | None = None) -> Dict[str, object]:
    config = config or load_config()
    reports = config.path("reports")
    reports.mkdir(parents=True, exist_ok=True)
    processed = config.path("data/processed")
    processed.mkdir(parents=True, exist_ok=True)

    raw = load_raw(config)
    train_raw, holdout_raw = split_by_sequence(raw, config)

    target_column = config.data.target_column
    y_train = train_raw[target_column].to_numpy(dtype="int64")
    y_holdout = holdout_raw[target_column].to_numpy(dtype="int64")

    x_train = build_features(train_raw)
    x_holdout = build_features(holdout_raw)

    binning_config = BinningConfig(
        max_prebins=config.binning.max_prebins,
        min_bin_fraction=config.binning.min_bin_fraction,
        min_bin_bads=config.binning.min_bin_bads,
        enforce_monotonic=config.binning.enforce_monotonic,
        min_categorical_fraction=config.binning.min_categorical_fraction,
    )
    transformer = WOETransformer(binning_config).fit(
        x_train, y_train, config.features.numeric, config.features.categorical
    )

    woe_train = transformer.transform(x_train)
    selected = select_features(
        transformer, woe_train, config.selection.min_iv, config.selection.max_correlation
    )
    if not selected:
        raise RuntimeError("no characteristic cleared the information value floor")

    scaling = ScalingConfig(
        base_score=config.scaling.base_score,
        base_odds=config.scaling.base_odds,
        pdo=config.scaling.pdo,
    )
    card = Scorecard(scaling).fit(woe_train, y_train, selected)

    train_score = card.score(woe_train)
    decline_cut = float(np.percentile(train_score, config.bands.decline_below_percentile))
    refer_cut = float(np.percentile(train_score, config.bands.refer_below_percentile))
    bands = BandCutoffs(decline_below=decline_cut, refer_below=refer_cut)

    edges, proportions = reference_score_distribution(train_score, config.monitoring.reference_bins)

    train_bands = bands.assign_many(train_score)
    band_proportions = np.array(
        [float((train_bands == band).mean()) for band in BAND_ORDER], dtype="float64"
    )
    train_probability = card.predict_proba(woe_train)

    categorical_levels: Dict[str, List[str]] = {}
    for column in config.features.categorical:
        levels = sorted(
            str(value) for value in train_raw[column].dropna().unique()
        )
        categorical_levels[column] = levels

    artifact = ScorecardArtifact(
        transformer=transformer,
        scorecard=card,
        bands=bands,
        numeric_features=list(config.features.numeric),
        categorical_features=list(config.features.categorical),
        categorical_levels=categorical_levels,
        reference_score_edges=edges,
        reference_score_proportions=proportions,
        reference_band_proportions=band_proportions,
        reference_mean_score=float(train_score.mean()),
        reference_mean_probability=float(train_probability.mean()),
        trained_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        training_rows=int(len(train_raw)),
        training_bad_rate=float(y_train.mean()),
        model_version=MODEL_VERSION,
    )

    model_path = config.path(config.service.model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, model_path)

    woe_holdout = transformer.transform(x_holdout)
    scored_train = artifact.score_frame(x_train)
    scored_holdout = artifact.score_frame(x_holdout)

    performance = {
        "model_version": MODEL_VERSION,
        "trained_at": artifact.trained_at,
        "training_rows": int(len(train_raw)),
        "holdout_rows": int(len(holdout_raw)),
        "training_bad_rate": round(float(y_train.mean()), 5),
        "holdout_bad_rate": round(float(y_holdout.mean()), 5),
        "features_selected": selected,
        "train_auc": round(float(roc_auc_score(y_train, scored_train["probability"])), 4),
        "holdout_auc": round(float(roc_auc_score(y_holdout, scored_holdout["probability"])), 4),
        "train_gini": round(gini(y_train, scored_train["probability"].to_numpy()), 4),
        "holdout_gini": round(gini(y_holdout, scored_holdout["probability"].to_numpy()), 4),
        "train_ks": round(ks_statistic(y_train, scored_train["probability"].to_numpy()), 4),
        "holdout_ks": round(ks_statistic(y_holdout, scored_holdout["probability"].to_numpy()), 4),
        "band_cutoffs": {"decline_below": round(decline_cut, 2), "refer_below": round(refer_cut, 2)},
    }

    with open(reports / "model_performance.json", "w", encoding="utf-8") as handle:
        json.dump(performance, handle, indent=2)

    transformer.iv_table().to_csv(reports / "iv_table.csv", index=False)
    transformer.bin_table().to_csv(reports / "bins.csv", index=False)
    card.scorecard_table(transformer).to_csv(reports / "scorecard.csv", index=False)
    band_summary(scored_train, y_train).to_csv(reports / "band_summary_train.csv", index=False)
    band_summary(scored_holdout, y_holdout).to_csv(reports / "band_summary_holdout.csv", index=False)

    reference = {
        "model_version": MODEL_VERSION,
        "trained_at": artifact.trained_at,
        "training_rows": int(len(train_raw)),
        "features": {
            name: {
                "kind": binning.kind,
                "bin_labels": binning.bin_labels,
                "reference_proportions": [float(v) for v in binning.reference_proportions],
                "iv": float(binning.iv),
                "in_model": name in selected,
            }
            for name, binning in transformer.bins.items()
        },
        "score": {
            "edges": [float(v) for v in edges],
            "reference_proportions": [float(v) for v in proportions],
            "mean_score": round(float(train_score.mean()), 3),
            "mean_probability": round(float(train_probability.mean()), 6),
        },
        "band": {
            "bands": list(BAND_ORDER),
            "reference_proportions": [float(v) for v in band_proportions],
            "cutoffs": {"decline_below": round(decline_cut, 2), "refer_below": round(refer_cut, 2)},
        },
    }
    with open(config.path(config.service.reference_path), "w", encoding="utf-8") as handle:
        json.dump(reference, handle, indent=2)

    holdout_columns = [config.data.id_column, target_column] + RAW_INPUTS
    holdout_raw[holdout_columns].to_parquet(processed / "holdout.parquet", index=False)

    return performance


def main() -> None:
    performance = train()
    print(json.dumps(performance, indent=2))


if __name__ == "__main__":
    main()
