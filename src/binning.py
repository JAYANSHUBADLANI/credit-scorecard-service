"""Monotonic weight of evidence binning and information value.

This is the same binning approach I used on the application scorecard it accompanies, kept
deliberately unchanged: quantile prebins merged until every bin carries a minimum share of
the population and the bad rate moves in one direction. Monotonicity is not cosmetic. A card
that says a rising credit to income ratio is worse up to 4x and then better again cannot be
signed off by a credit committee, and it usually means the model is fitting sampling noise.

Missing values get their own bin rather than being imputed, because a null external score in
this data means the bureau returned nothing, which is itself informative.

Two things are added here that the offline scorecard did not need, both for the monitoring
layer rather than for the model:

1. `bin_index` exposes the bin each record falls into, because the characteristic stability
   index compares bin populations over time, not weight of evidence values.
2. `reference_proportions` records the training time population share of every bin, which is
   the baseline every later window is measured against.

Bin assignment is computed in exactly one place, so the weight of evidence applied at scoring
time and the bin counted by monitoring can never disagree.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

SMOOTHING = 0.5
MISSING_LABEL = "missing"
OTHER_LABEL = "__other__"


@dataclass
class BinningConfig:
    max_prebins: int = 20
    min_bin_fraction: float = 0.03
    min_bin_bads: int = 5
    enforce_monotonic: bool = True
    min_categorical_fraction: float = 0.01


@dataclass
class FeatureBinning:
    """Fitted bins for one feature.

    The missing bin is always present, even when the training data had no missing values,
    so that bin indices stay stable between training and serving.
    """

    name: str
    kind: str
    woe: np.ndarray
    missing_woe: float
    table: pd.DataFrame
    iv: float
    edges: np.ndarray = field(default_factory=lambda: np.array([]))
    level_to_index: Dict[str, int] = field(default_factory=dict)
    bin_labels: List[str] = field(default_factory=list)
    reference_proportions: np.ndarray = field(default_factory=lambda: np.array([]))

    @property
    def missing_index(self) -> int:
        return len(self.woe)

    @property
    def n_bins(self) -> int:
        return len(self.woe) + 1

    def bin_index(self, values) -> np.ndarray:
        """Map raw values onto bin indices. Missing and unknown levels go to their own bins."""
        if self.kind == "numeric":
            numeric = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype="float64")
            out = np.full(numeric.shape, self.missing_index, dtype="int64")
            observed = ~np.isnan(numeric)
            if observed.any():
                if self.edges.size:
                    position = np.searchsorted(self.edges, numeric[observed], side="right")
                else:
                    position = np.zeros(int(observed.sum()), dtype="int64")
                out[observed] = np.clip(position, 0, len(self.woe) - 1)
            return out

        series = pd.Series(values).astype("object")
        fallback = self.level_to_index.get(OTHER_LABEL, self.missing_index)
        out = np.empty(len(series), dtype="int64")
        for i, value in enumerate(series):
            if value is None or (isinstance(value, float) and np.isnan(value)):
                out[i] = self.missing_index
            else:
                out[i] = self.level_to_index.get(str(value), fallback)
        return out

    def transform(self, values) -> np.ndarray:
        """Weight of evidence for each value, derived from the same bin assignment."""
        index = self.bin_index(values)
        lookup = np.append(self.woe, self.missing_woe)
        return lookup[index].astype("float64")


def _woe_and_iv(bads: np.ndarray, goods: np.ndarray):
    """Weight of evidence per bin, and the information value over populated bins only.

    The smoothing constant keeps the log finite for a bin that holds only goods or only bads,
    and it is also what gives an empty bin a defined weight of evidence, which is needed
    because the missing bin is always present even when training saw no missing values. That
    same smoothing would otherwise let an empty bin contribute to information value, so bins
    with no observations are excluded from the sum. A bin nothing landed in carries no
    information about anything. The effect is small on a large sample, roughly 0.0001 on the
    215,257 row training slice here, and large enough to matter on a few hundred rows.
    """
    total_bad = bads.sum()
    total_good = goods.sum()
    if total_bad == 0 or total_good == 0:
        return np.zeros(len(bads), dtype="float64"), 0.0
    bad_rate = (bads + SMOOTHING) / (total_bad + SMOOTHING * len(bads))
    good_rate = (goods + SMOOTHING) / (total_good + SMOOTHING * len(goods))
    woe = np.log(bad_rate / good_rate)
    populated = (bads + goods) > 0
    iv = float(((bad_rate - good_rate) * woe)[populated].sum())
    return woe, iv


def _merge(counts: List[int], bads: List[int], edges: List[float], position: int) -> None:
    counts[position] += counts[position + 1]
    bads[position] += bads[position + 1]
    del counts[position + 1]
    del bads[position + 1]
    del edges[position]


def _enforce_minimums(counts, bads, edges, min_count, min_bads):
    changed = True
    while changed and len(counts) > 1:
        changed = False
        for i in range(len(counts)):
            goods = counts[i] - bads[i]
            too_small = counts[i] < min_count or bads[i] < min_bads or goods < 1
            if not too_small:
                continue
            position = i - 1 if i == len(counts) - 1 else i
            _merge(counts, bads, edges, position)
            changed = True
            break
    return counts, bads, edges


def _enforce_monotonic(counts, bads, edges):
    while len(counts) > 2:
        rates = np.array(bads, dtype="float64") / np.array(counts, dtype="float64")
        direction = np.sign(np.corrcoef(np.arange(len(rates)), rates)[0, 1])
        if np.isnan(direction) or direction == 0:
            direction = 1.0
        diffs = np.diff(rates) * direction
        violations = np.flatnonzero(diffs < 0)
        if violations.size == 0:
            break
        worst = int(violations[np.argmin(diffs[violations])])
        _merge(counts, bads, edges, worst)
    return counts, bads, edges


def _finalise(
    name: str,
    kind: str,
    labels: List[str],
    counts: np.ndarray,
    bads: np.ndarray,
    missing_count: int,
    missing_bads: int,
    edges: np.ndarray,
    level_to_index: Dict[str, int],
) -> FeatureBinning:
    """Shared tail of both fitting paths: weight of evidence, the bin table and the baseline."""
    goods = counts - bads
    all_bads = np.append(bads, missing_bads)
    all_goods = np.append(goods, missing_count - missing_bads)
    woe_all, iv = _woe_and_iv(all_bads.astype("float64"), all_goods.astype("float64"))
    woe = woe_all[:-1]
    missing_woe = float(woe_all[-1])

    bin_labels = list(labels) + [MISSING_LABEL]
    all_counts = np.append(counts, missing_count).astype("float64")
    total = all_counts.sum()
    reference = all_counts / total if total > 0 else np.zeros_like(all_counts)

    rows = []
    for i, label in enumerate(bin_labels):
        count = int(all_counts[i])
        bad = int(all_bads[i])
        rows.append({
            "feature": name,
            "bin_index": i,
            "bin": label,
            "count": count,
            "bads": bad,
            "bad_rate": float(bad / count) if count else 0.0,
            "population_share": float(reference[i]),
            "woe": float(woe_all[i]),
        })
    table = pd.DataFrame(rows)
    table["iv"] = iv

    return FeatureBinning(
        name=name,
        kind=kind,
        woe=woe.astype("float64"),
        missing_woe=missing_woe,
        table=table,
        iv=float(iv),
        edges=np.asarray(edges, dtype="float64"),
        level_to_index=level_to_index,
        bin_labels=bin_labels,
        reference_proportions=reference,
    )


def fit_numeric(values, target: np.ndarray, name: str, config: BinningConfig) -> FeatureBinning:
    """Fit monotonic weight of evidence bins for one numeric feature."""
    values = np.asarray(values, dtype="float64")
    target = np.asarray(target, dtype="int64")

    missing = np.isnan(values)
    missing_count = int(missing.sum())
    missing_bads = int(target[missing].sum())

    observed_values = values[~missing]
    observed_target = target[~missing]

    if observed_values.size == 0 or np.unique(observed_values).size < 2:
        return _finalise(
            name, "numeric", ["all"], np.array([observed_values.size]),
            np.array([int(observed_target.sum())]), missing_count, missing_bads,
            np.array([]), {},
        )

    quantiles = np.linspace(0, 1, config.max_prebins + 1)[1:-1]
    edges = list(np.unique(np.quantile(observed_values, quantiles)))
    index = np.searchsorted(np.array(edges), observed_values, side="right")
    n_bins = len(edges) + 1
    counts = list(np.bincount(index, minlength=n_bins).astype(int))
    bads = list(np.bincount(index, weights=observed_target, minlength=n_bins).astype(int))

    min_count = max(int(config.min_bin_fraction * len(values)), 1)
    counts, bads, edges = _enforce_minimums(counts, bads, edges, min_count, config.min_bin_bads)
    if config.enforce_monotonic:
        counts, bads, edges = _enforce_monotonic(counts, bads, edges)

    boundaries = [-np.inf] + list(edges) + [np.inf]
    labels = [f"({boundaries[i]:.4g}, {boundaries[i + 1]:.4g}]" for i in range(len(counts))]

    return _finalise(
        name, "numeric", labels, np.array(counts), np.array(bads),
        missing_count, missing_bads, np.array(edges), {},
    )


def fit_categorical(values, target: np.ndarray, name: str, config: BinningConfig) -> FeatureBinning:
    """Fit weight of evidence for one categorical feature.

    Every level keeps its own bin unless it falls below the minimum population share, in which
    case it joins a catch all group. That group also absorbs any level not seen in training,
    so the transformer cannot fail on an unexpected value even though the API rejects one.
    """
    series = pd.Series(values).astype("object")
    target = np.asarray(target, dtype="int64")

    missing_mask = series.isna().to_numpy()
    missing_count = int(missing_mask.sum())
    missing_bads = int(target[missing_mask].sum())

    observed = series[~missing_mask].astype(str)
    observed_target = target[~missing_mask]

    shares = observed.value_counts(normalize=True)
    frequent = [level for level, share in shares.items() if share >= config.min_categorical_fraction]
    frequent.sort()

    groups = frequent + [OTHER_LABEL]
    level_to_index = {level: i for i, level in enumerate(frequent)}
    other_index = len(frequent)
    level_to_index[OTHER_LABEL] = other_index

    assigned = observed.map(lambda level: level_to_index.get(level, other_index)).to_numpy()
    counts = np.bincount(assigned, minlength=len(groups)).astype(int)
    bads = np.bincount(assigned, weights=observed_target, minlength=len(groups)).astype(int)

    return _finalise(
        name, "categorical", groups, counts, bads,
        missing_count, missing_bads, np.array([]), level_to_index,
    )


class WOETransformer:
    """Fits and applies weight of evidence bins across a set of features."""

    def __init__(self, config: Optional[BinningConfig] = None):
        self.config = config or BinningConfig()
        self.bins: Dict[str, FeatureBinning] = {}

    def fit(
        self,
        frame: pd.DataFrame,
        target: np.ndarray,
        numeric: List[str],
        categorical: Optional[List[str]] = None,
    ) -> "WOETransformer":
        for column in numeric:
            self.bins[column] = fit_numeric(
                frame[column].to_numpy(dtype="float64"), target, column, self.config
            )
        for column in categorical or []:
            self.bins[column] = fit_categorical(frame[column], target, column, self.config)
        return self

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(
            {name: binning.transform(frame[name]) for name, binning in self.bins.items()},
            index=frame.index,
        )

    def bin_indices(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Bin index per feature, which is what the stability indices are computed from."""
        return pd.DataFrame(
            {name: binning.bin_index(frame[name]) for name, binning in self.bins.items()},
            index=frame.index,
        )

    def iv_table(self) -> pd.DataFrame:
        rows = [
            {"feature": name, "iv": binning.iv, "bins": binning.n_bins, "kind": binning.kind}
            for name, binning in self.bins.items()
        ]
        return pd.DataFrame(rows).sort_values("iv", ascending=False).reset_index(drop=True)

    def bin_table(self) -> pd.DataFrame:
        return pd.concat([b.table for b in self.bins.values()], ignore_index=True)
