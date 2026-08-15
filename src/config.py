"""Typed access to config/config.yaml.

Every module reads its settings through here so that thresholds and window lengths are
never hard coded next to the logic that uses them.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"


@dataclass
class SplitConfig:
    order_by: str
    train_fraction: float


@dataclass
class FeatureConfig:
    numeric: List[str]
    categorical: List[str]

    @property
    def all_features(self) -> List[str]:
        return list(self.numeric) + list(self.categorical)


@dataclass
class BinningConfig:
    max_prebins: int = 20
    min_bin_fraction: float = 0.03
    min_bin_bads: int = 5
    enforce_monotonic: bool = True
    min_categorical_fraction: float = 0.01


@dataclass
class SelectionConfig:
    min_iv: float = 0.02
    max_correlation: float = 0.75


@dataclass
class ScalingConfig:
    base_score: float = 600.0
    base_odds: float = 50.0
    pdo: float = 20.0


@dataclass
class BandConfig:
    decline_below_percentile: float
    refer_below_percentile: float


@dataclass
class MonitoringConfig:
    window_size: int
    min_window_size: int
    psi_warn: float
    psi_alert: float
    csi_warn: float
    csi_alert: float
    prediction_psi_warn: float
    prediction_psi_alert: float
    reference_bins: int
    debounce_windows: int
    cooldown_windows: int


@dataclass
class StreamConfig:
    total_requests: int
    requests_per_batch: int
    stable_fraction: float
    drift_strength: float
    drift_ramp_fraction: float
    drift_features: List[str]
    seed: int


@dataclass
class ServiceConfig:
    db_path: str
    model_path: str
    reference_path: str


@dataclass
class DataConfig:
    raw_path: str
    id_column: str
    target_column: str


@dataclass
class Config:
    data: DataConfig
    split: SplitConfig
    features: FeatureConfig
    binning: BinningConfig
    selection: SelectionConfig
    scaling: ScalingConfig
    bands: BandConfig
    monitoring: MonitoringConfig
    stream: StreamConfig
    service: ServiceConfig
    root: Path = field(default=PROJECT_ROOT)

    def path(self, relative: str) -> Path:
        """Resolve a config path against the project root unless it is already absolute."""
        candidate = Path(relative)
        return candidate if candidate.is_absolute() else self.root / candidate


def _as_dict(raw: Dict[str, Any], key: str) -> Dict[str, Any]:
    section = raw.get(key)
    if not isinstance(section, dict):
        raise ValueError(f"config section '{key}' is missing or malformed")
    return section


def load_config(path: Path | str | None = None) -> Config:
    """Read the YAML config. The CONFIG_PATH environment variable overrides the default."""
    resolved = Path(path or os.environ.get("CONFIG_PATH") or DEFAULT_CONFIG_PATH)
    with open(resolved, "r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)

    return Config(
        data=DataConfig(**_as_dict(raw, "data")),
        split=SplitConfig(**_as_dict(raw, "split")),
        features=FeatureConfig(**_as_dict(raw, "features")),
        binning=BinningConfig(**_as_dict(raw, "binning")),
        selection=SelectionConfig(**_as_dict(raw, "selection")),
        scaling=ScalingConfig(**_as_dict(raw, "scaling")),
        bands=BandConfig(**_as_dict(raw, "bands")),
        monitoring=MonitoringConfig(**_as_dict(raw, "monitoring")),
        stream=StreamConfig(**_as_dict(raw, "stream")),
        service=ServiceConfig(**_as_dict(raw, "service")),
        root=resolved.resolve().parents[1],
    )
