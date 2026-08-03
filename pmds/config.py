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
    supported_metrics = {"mae", "rmse", "smape", "mase", "wql"}
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

    dataset_names = [dataset.get("name") for dataset in config["datasets"] if dataset.get("enabled", True)]
    if len(dataset_names) != len(set(dataset_names)):
        raise ValueError("Enabled dataset names must be unique")
    model_names = [model.get("name") for model in config["models"] if model.get("enabled", True)]
    if len(model_names) != len(set(model_names)):
        raise ValueError("Enabled model names must be unique")
