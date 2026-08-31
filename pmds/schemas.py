"""Shared data structures for the PMDS benchmark."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np
import pandas as pd


RESULT_COLUMNS = [
    "dataset",
    "series_id",
    "item_id",
    "origin",
    "cutoff_timestamp",
    "model",
    "repetition",
    "seed",
    "diagnostic_model",
    "forecast_distribution",
    "mae",
    "rmse",
    "smape",
    "mase",
    "wql",
    "wql_loss_sum",
    "wql_abs_target_sum",
    "rain_occurrence_error",
    "positive_mae",
    "actual_zero_fraction",
    "predicted_zero_fraction",
    "metric_notes",
    "duration_seconds",
    "error_type",
    "error",
]


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    family: str
    repo: str
    hf_configs: tuple[str, ...]
    split: str
    prediction_length: int
    seasonality: int
    max_series: int
    fallback_frequency: str
    target_column: str | None = None
    date_column: str | None = None
    exclude_columns: tuple[str, ...] = ()
    trust_remote_code: bool = False
    num_origins: int = 1
    origin_stride: int | None = None
    selection_strategy: str = "first"
    selection_seed: int = 0
    timestamp_unit: str | None = None
    resample_frequency: str | None = None
    resample_method: str = "mean"
    filter_column: str | None = None
    filter_values: tuple[str, ...] = ()
    zero_inflated: bool = False
    zero_threshold: float = 0.0
    source: str | None = None
    source_params: Mapping[str, Any] = field(default_factory=dict)
    snapshot_path: str | None = None
    revision: str | None = None

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "DatasetSpec":
        required = {
            "name",
            "family",
            "repo",
            "hf_configs",
            "split",
            "prediction_length",
            "seasonality",
            "max_series",
            "fallback_frequency",
        }
        missing = required - set(config)
        if missing:
            raise ValueError(f"Dataset config is missing fields: {sorted(missing)}")
        return cls(
            name=str(config["name"]),
            family=str(config["family"]),
            repo=str(config["repo"]),
            hf_configs=tuple(map(str, config["hf_configs"])),
            split=str(config["split"]),
            prediction_length=int(config["prediction_length"]),
            seasonality=int(config["seasonality"]),
            max_series=int(config["max_series"]),
            fallback_frequency=str(config["fallback_frequency"]),
            target_column=config.get("target_column"),
            date_column=config.get("date_column"),
            exclude_columns=tuple(map(str, config.get("exclude_columns", []))),
            trust_remote_code=bool(config.get("trust_remote_code", False)),
            num_origins=int(config.get("num_origins", 1)),
            origin_stride=(int(config["origin_stride"]) if config.get("origin_stride") is not None else None),
            selection_strategy=str(config.get("selection_strategy", "first")),
            selection_seed=int(config.get("selection_seed", 0)),
            timestamp_unit=(str(config["timestamp_unit"]) if config.get("timestamp_unit") is not None else None),
            resample_frequency=(
                str(config["resample_frequency"]) if config.get("resample_frequency") is not None else None
            ),
            resample_method=str(config.get("resample_method", "mean")),
            filter_column=(str(config["filter_column"]) if config.get("filter_column") is not None else None),
            filter_values=tuple(map(str, config.get("filter_values", []))),
            zero_inflated=bool(config.get("zero_inflated", False)),
            zero_threshold=float(config.get("zero_threshold", 0.0)),
            source=(str(config["source"]) if config.get("source") is not None else None),
            source_params=dict(config.get("source_params", {})),
            snapshot_path=(str(config["snapshot_path"]) if config.get("snapshot_path") is not None else None),
            revision=(str(config["revision"]) if config.get("revision") is not None else None),
        )


@dataclass(frozen=True)
class ForecastTask:
    dataset: str
    item_id: str
    context_timestamps: pd.DatetimeIndex
    future_timestamps: pd.DatetimeIndex
    context: np.ndarray
    future: np.ndarray
    prediction_length: int
    seasonality: int
    frequency: str
    series_id: str = ""
    origin: int = 0
    zero_inflated: bool = False
    zero_threshold: float = 0.0


@dataclass(frozen=True)
class ForecastOutput:
    mean: np.ndarray
    quantiles: np.ndarray  # shape: (prediction_length, num_quantiles)
    distribution: str = "unspecified"


@dataclass(frozen=True)
class ForecastResult:
    dataset: str
    item_id: str
    model: str
    output: ForecastOutput | None
    duration_seconds: float
    error_type: str = ""
    error: str = ""
    repetition: int = 0
    seed: int | None = None
