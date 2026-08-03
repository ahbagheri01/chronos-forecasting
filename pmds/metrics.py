"""Metric computation and summarization for PMDS benchmark results."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from pmds.schemas import RESULT_COLUMNS, ForecastOutput, ForecastResult, ForecastTask
from pmds.utils import clean_numeric


def validate_output(output: ForecastOutput, task: ForecastTask, quantiles: np.ndarray) -> int:
    if output.mean.shape != (task.prediction_length,):
        raise ValueError(f"Mean forecast has shape {output.mean.shape}; expected {(task.prediction_length,)}")
    expected_quantile_shape = (task.prediction_length, len(quantiles))
    if output.quantiles.shape != expected_quantile_shape:
        raise ValueError(f"Quantile forecast has shape {output.quantiles.shape}; expected {expected_quantile_shape}")
    if not np.isfinite(output.mean).all() or not np.isfinite(output.quantiles).all():
        raise ValueError("Forecast contains non-finite values")
    return int(np.sum(np.diff(output.quantiles, axis=1) < 0.0))


def select_point_forecast(output: ForecastOutput, quantiles: np.ndarray, method: str) -> np.ndarray:
    if method == "mean":
        return output.mean
    median_index = int(np.flatnonzero(np.isclose(quantiles, 0.5))[0])
    return output.quantiles[:, median_index]


def mase(y_true: np.ndarray, y_pred: np.ndarray, insample: np.ndarray, seasonality: int) -> float:
    seasonality = seasonality if len(insample) > seasonality else 1
    differences = np.abs(insample[seasonality:] - insample[:-seasonality])
    scale = float(np.mean(differences)) if len(differences) else float("nan")
    if not np.isfinite(scale) or scale == 0.0:
        return float("nan")
    return float(np.mean(np.abs(y_true - y_pred)) / scale)


def weighted_quantile_loss_components(
    y_true: np.ndarray, quantile_values: np.ndarray, quantile_levels: np.ndarray
) -> tuple[float, float, float]:
    denominator = float(np.sum(np.abs(y_true)))
    if denominator == 0.0:
        return float("nan"), float("nan"), denominator
    quantile_loss_sums = []
    for index, level in enumerate(quantile_levels):
        prediction = quantile_values[:, index]
        error = prediction - y_true
        quantile_loss = 2.0 * np.abs(error * ((prediction >= y_true).astype(float) - level))
        quantile_loss_sums.append(float(np.sum(quantile_loss)))
    mean_loss_sum = float(np.mean(quantile_loss_sums))
    return mean_loss_sum / denominator, mean_loss_sum, denominator


def empty_metric_row(task: ForecastTask, result: ForecastResult) -> dict[str, Any]:
    return {
        "dataset": task.dataset,
        "item_id": task.item_id,
        "model": result.model,
        "mae": float("nan"),
        "rmse": float("nan"),
        "smape": float("nan"),
        "mase": float("nan"),
        "wql": float("nan"),
        "wql_loss_sum": float("nan"),
        "wql_abs_target_sum": float("nan"),
        "duration_seconds": result.duration_seconds,
        "error_type": result.error_type,
        "error": result.error,
    }


def compute_metrics(
    task: ForecastTask,
    result: ForecastResult,
    metric_names: Sequence[str],
    quantiles: np.ndarray,
    point_method: str,
) -> dict[str, Any]:
    if result.output is None:
        return empty_metric_row(task, result)

    y_true = task.future.astype(np.float32)
    y_pred = select_point_forecast(result.output, quantiles, point_method).astype(np.float32)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if not mask.all():
        raise ValueError(f"Non-finite values encountered while scoring {task.dataset}/{task.item_id}")

    row = empty_metric_row(task, result)
    row["error_type"] = ""
    row["error"] = ""
    if "mae" in metric_names:
        row["mae"] = float(np.mean(np.abs(y_true - y_pred)))
    if "rmse" in metric_names:
        row["rmse"] = float(np.sqrt(np.mean(np.square(y_true - y_pred))))
    if "smape" in metric_names:
        denominator = np.maximum(np.abs(y_true) + np.abs(y_pred), 1e-8)
        row["smape"] = float(np.mean(2.0 * np.abs(y_true - y_pred) / denominator))
    if "mase" in metric_names:
        row["mase"] = mase(y_true, y_pred, clean_numeric(task.context), task.seasonality)
    if "wql" in metric_names:
        row["wql"], row["wql_loss_sum"], row["wql_abs_target_sum"] = weighted_quantile_loss_components(
            y_true,
            result.output.quantiles,
            quantiles,
        )
    return row


def result_frame(rows: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=RESULT_COLUMNS)


def summarize(results: pd.DataFrame, metric_names: Sequence[str]) -> pd.DataFrame:
    if results.empty:
        return pd.DataFrame(columns=["dataset", "model", *metric_names, "successful_tasks", "failed_tasks"])
    grouped = results.groupby(["dataset", "model"], dropna=False)
    mean_metrics = [name for name in metric_names if name != "wql"]
    if mean_metrics:
        summary = grouped[mean_metrics].mean(numeric_only=True).reset_index()
    else:
        summary = grouped.size().rename("_size").reset_index().drop(columns="_size")
    if "wql" in metric_names:
        wql_components = grouped[["wql_loss_sum", "wql_abs_target_sum"]].sum(min_count=1).reset_index()
        wql_components["wql"] = wql_components["wql_loss_sum"] / wql_components["wql_abs_target_sum"]
        summary = summary.merge(wql_components[["dataset", "model", "wql"]], on=["dataset", "model"])
    counts = grouped["error"].agg(
        successful_tasks=lambda values: int((values == "").sum()),
        failed_tasks=lambda values: int((values != "").sum()),
    ).reset_index()
    durations = grouped["duration_seconds"].sum().rename("total_duration_seconds").reset_index()
    return (
        summary.merge(counts, on=["dataset", "model"])
        .merge(durations, on=["dataset", "model"])
        .sort_values(["dataset", metric_names[0]], na_position="last")
    )
