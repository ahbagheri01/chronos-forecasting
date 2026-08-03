"""Shared data structures for the PMDS benchmark."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import pandas as pd


RESULT_COLUMNS = [
    "dataset",
    "item_id",
    "model",
    "mae",
    "rmse",
    "smape",
    "mase",
    "wql",
    "wql_loss_sum",
    "wql_abs_target_sum",
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


@dataclass(frozen=True)
class ForecastOutput:
    mean: np.ndarray
    quantiles: np.ndarray  # shape: (prediction_length, num_quantiles)


@dataclass(frozen=True)
class ForecastResult:
    dataset: str
    item_id: str
    model: str
    output: ForecastOutput | None
    duration_seconds: float
    error_type: str = ""
    error: str = ""
