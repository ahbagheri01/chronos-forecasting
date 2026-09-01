"""Result persistence helpers for PMDS benchmarks."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from pmds.metrics import result_frame, select_point_forecast, summarize
from pmds.schemas import ForecastOutput, ForecastResult, ForecastTask
from pmds.utils import resolve_path


LOGGER = logging.getLogger("pmds.compare")


def quantile_column(level: float) -> str:
    return f"q_{int(round(level * 100)):02d}"


def context_rows_for_task(task: ForecastTask, context_multiplier: int) -> list[dict[str, Any]]:
    history_points = min(
        len(task.context),
        max(task.prediction_length * context_multiplier, task.seasonality * 2),
    )
    timestamps = task.context_timestamps[-history_points:]
    values = task.context[-history_points:]
    return [
        {
            "dataset": task.dataset,
            "series_id": task.series_id or task.item_id,
            "item_id": task.item_id,
            "origin": task.origin,
            "step": index - history_points,
            "timestamp": str(timestamp),
            "actual": float(value),
        }
        for index, (timestamp, value) in enumerate(zip(timestamps, values))
    ]


def forecast_rows_for_result(
    task: ForecastTask,
    result: ForecastResult,
    quantiles: np.ndarray,
    point_method: str,
) -> list[dict[str, Any]]:
    if result.output is None:
        return []
    point = select_point_forecast(result.output, quantiles, point_method)
    rows: list[dict[str, Any]] = []
    for step, timestamp in enumerate(task.future_timestamps):
        row: dict[str, Any] = {
            "dataset": task.dataset,
            "series_id": task.series_id or task.item_id,
            "item_id": task.item_id,
            "origin": task.origin,
            "model": result.model,
            "repetition": result.repetition,
            "seed": result.seed,
            "forecast_distribution": result.output.distribution,
            "step": step,
            "timestamp": str(timestamp),
            "actual": float(task.future[step]),
            "point_forecast": float(point[step]),
            "mean_forecast": float(result.output.mean[step]),
        }
        for index, level in enumerate(quantiles):
            row[quantile_column(float(level))] = float(result.output.quantiles[step, index])
        rows.append(row)
    return rows


def atomic_write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def atomic_write_json(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as fp:
        json.dump(value, fp, indent=2)
    os.replace(temporary, path)


def persist_results(
    rows: Sequence[Mapping[str, Any]],
    statuses: Sequence[Mapping[str, Any]],
    run_config: Mapping[str, Any],
    metric_names: Sequence[str],
    forecast_rows: Sequence[Mapping[str, Any]] = (),
    context_rows: Sequence[Mapping[str, Any]] = (),
) -> tuple[Path, Path, Path, Path, Path]:
    output_dir = resolve_path(run_config["output_dir"])
    run_name = str(run_config["name"])
    detailed_path = output_dir / f"{run_name}_detailed.csv"
    summary_path = output_dir / f"{run_name}_summary.csv"
    status_path = output_dir / f"{run_name}_status.json"
    forecast_path = output_dir / f"{run_name}_forecasts.csv"
    context_path = output_dir / f"{run_name}_contexts.csv"
    detailed = result_frame(rows)
    summary = summarize(detailed, metric_names)
    atomic_write_csv(detailed, detailed_path)
    atomic_write_csv(summary, summary_path)
    atomic_write_json(list(statuses), status_path)
    atomic_write_csv(pd.DataFrame(forecast_rows), forecast_path)
    atomic_write_csv(pd.DataFrame(context_rows), context_path)
    LOGGER.info(
        "Results checkpoint saved | detailed=%s summary=%s status=%s forecasts=%s contexts=%s rows=%d",
        detailed_path,
        summary_path,
        status_path,
        forecast_path,
        context_path,
        len(detailed),
    )
    return detailed_path, summary_path, status_path, forecast_path, context_path


def dataset_failure_rows(
    dataset_name: str,
    runners: Mapping[str, Callable[[ForecastTask], ForecastOutput]],
    exc: Exception,
) -> list[dict[str, Any]]:
    return [
        {
            "dataset": dataset_name,
            "series_id": "__dataset_load__",
            "item_id": "__dataset_load__",
            "origin": 0,
            "cutoff_timestamp": "",
            "model": model_name,
            "repetition": 0,
            "seed": None,
            "diagnostic_model": False,
            "forecast_distribution": "",
            "mae": float("nan"),
            "rmse": float("nan"),
            "smape": float("nan"),
            "mase": float("nan"),
            "crps": float("nan"),
            "wql": float("nan"),
            "wql_loss_sum": float("nan"),
            "wql_abs_target_sum": float("nan"),
            "rain_occurrence_error": float("nan"),
            "positive_mae": float("nan"),
            "actual_zero_fraction": float("nan"),
            "predicted_zero_fraction": float("nan"),
            "metric_notes": "",
            "duration_seconds": 0.0,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        for model_name in runners
    ]
