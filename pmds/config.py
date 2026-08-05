"""Configuration loading and validation for PMDS experiments."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np


def load_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Configuration file does not exist: {path}")
    with path.open(encoding="utf-8") as fp:
        config = json.load(fp)
    validate_config(config)
    return config


def validate_config(config: Mapping[str, Any]) -> None:
    required = {"run", "evaluation", "datasets", "models"}
    missing = required - set(config)
    if missing:
        raise ValueError(f"Top-level config is missing fields: {sorted(missing)}")

    run_required = {"name", "output_dir", "log_dir", "random_seed", "logging"}
    missing_run = run_required - set(config["run"])
    if missing_run:
        raise ValueError(f"run config is missing fields: {sorted(missing_run)}")

    evaluation = config["evaluation"]
    metrics = list(evaluation.get("metrics", []))
    supported_metrics = {
        "mae",
        "rmse",
        "smape",
        "mase",
        "wql",
        "rain_occurrence_error",
        "positive_mae",
    }
    unknown_metrics = set(metrics) - supported_metrics
    if unknown_metrics:
        raise ValueError(f"Unsupported metrics: {sorted(unknown_metrics)}")

    quantiles = np.asarray(evaluation.get("quantiles", []), dtype=float)
    if quantiles.ndim != 1 or len(quantiles) == 0:
        raise ValueError("evaluation.quantiles must be a non-empty list")
    if np.any(quantiles <= 0.0) or np.any(quantiles >= 1.0) or np.any(np.diff(quantiles) <= 0.0):
        raise ValueError("evaluation.quantiles must be strictly increasing and between 0 and 1")
    if evaluation.get("point_forecast") == "median" and not np.any(np.isclose(quantiles, 0.5)):
        raise ValueError("evaluation.quantiles must contain 0.5 when point_forecast='median'")
    if evaluation.get("point_forecast") not in {"mean", "median"}:
        raise ValueError("evaluation.point_forecast must be 'mean' or 'median'")
    if int(evaluation.get("stochastic_repetitions", 1)) < 1:
        raise ValueError("evaluation.stochastic_repetitions must be at least 1")
    if int(evaluation.get("forecast_context_multiplier", 3)) < 1:
        raise ValueError("evaluation.forecast_context_multiplier must be at least 1")

    enabled_datasets = [dataset for dataset in config["datasets"] if dataset.get("enabled", True)]
    dataset_names = [dataset.get("name") for dataset in enabled_datasets]
    if len(dataset_names) != len(set(dataset_names)):
        raise ValueError("Enabled dataset names must be unique")
    for dataset in enabled_datasets:
        name = dataset.get("name", "<unnamed>")
        if int(dataset.get("num_origins", 1)) < 1:
            raise ValueError(f"Dataset '{name}' num_origins must be at least 1")
        if dataset.get("origin_stride") is not None and int(dataset["origin_stride"]) < 1:
            raise ValueError(f"Dataset '{name}' origin_stride must be at least 1")
        if dataset.get("selection_strategy", "first") not in {"first", "evenly_spaced", "random"}:
            raise ValueError(
                f"Dataset '{name}' selection_strategy must be 'first', 'evenly_spaced', or 'random'"
            )
        if dataset.get("resample_method", "mean") not in {"mean", "median", "last"}:
            raise ValueError(f"Dataset '{name}' resample_method must be 'mean', 'median', or 'last'")
        filter_column = dataset.get("filter_column")
        filter_values = dataset.get("filter_values", [])
        if filter_column is not None and not filter_values:
            raise ValueError(f"Dataset '{name}' filter_values must be non-empty when filter_column is set")
        if filter_column is None and filter_values:
            raise ValueError(f"Dataset '{name}' filter_column is required when filter_values are set")
    model_names = [model.get("name") for model in config["models"] if model.get("enabled", True)]
    if len(model_names) != len(set(model_names)):
        raise ValueError("Enabled model names must be unique")
    unknown_scopes = {
        dataset_name
        for model in config["models"]
        if model.get("enabled", True)
        for dataset_name in model.get("datasets", [])
        if dataset_name not in dataset_names
    }
    if unknown_scopes:
        raise ValueError(f"Models reference unknown datasets: {sorted(unknown_scopes)}")



#  {
#       "name": "ar_2",
#       "type": "statsmodels_arima",
#       "enabled": false,
#       "params": {
#         "order": [2, 0, 0],
#         "seasonal_order": [0, 0, 0, 0],
#         "trend": "c",
#         "enforce_stationarity": false,
#         "enforce_invertibility": false,
#         "maxiter": 200
#       }
#     },
#     {
#       "name": "ma_2",
#       "type": "statsmodels_arima",
#       "enabled": false,
#       "params": {
#         "order": [0, 0, 2],
#         "seasonal_order": [0, 0, 0, 0],
#         "trend": "c",
#         "enforce_stationarity": false,
#         "enforce_invertibility": false,
#         "maxiter": 200
#       }
#     },