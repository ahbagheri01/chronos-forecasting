"""Metric computation and summarization for PMDS benchmark results."""

from __future__ import annotations

import json
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


def empty_metric_row(
    task: ForecastTask,
    result: ForecastResult,
    diagnostic_model: bool = False,
) -> dict[str, Any]:
    return {
        "dataset": task.dataset,
        "series_id": task.series_id or task.item_id,
        "item_id": task.item_id,
        "origin": task.origin,
        "cutoff_timestamp": str(task.future_timestamps[0]) if len(task.future_timestamps) else "",
        "model": result.model,
        "repetition": result.repetition,
        "seed": result.seed,
        "diagnostic_model": diagnostic_model,
        "forecast_distribution": result.output.distribution if result.output is not None else "",
        "mae": float("nan"),
        "rmse": float("nan"),
        "smape": float("nan"),
        "mase": float("nan"),
        "wql": float("nan"),
        "wql_loss_sum": float("nan"),
        "wql_abs_target_sum": float("nan"),
        "rain_occurrence_error": float("nan"),
        "positive_mae": float("nan"),
        "actual_zero_fraction": float("nan"),
        "predicted_zero_fraction": float("nan"),
        "metric_notes": "",
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
    diagnostic_model: bool = False,
) -> dict[str, Any]:
    if result.output is None:
        return empty_metric_row(task, result, diagnostic_model)

    y_true = task.future.astype(np.float32)
    y_pred = select_point_forecast(result.output, quantiles, point_method).astype(np.float32)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if not mask.all():
        raise ValueError(f"Non-finite values encountered while scoring {task.dataset}/{task.item_id}")

    row = empty_metric_row(task, result, diagnostic_model)
    row["error_type"] = ""
    row["error"] = ""
    threshold = task.zero_threshold
    actual_zero = np.abs(y_true) <= threshold
    predicted_zero = np.abs(y_pred) <= threshold
    row["actual_zero_fraction"] = float(np.mean(actual_zero))
    row["predicted_zero_fraction"] = float(np.mean(predicted_zero))
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
    if task.zero_inflated and "rain_occurrence_error" in metric_names:
        actual_occurrence = y_true > threshold
        predicted_occurrence = y_pred > threshold
        row["rain_occurrence_error"] = float(np.mean(actual_occurrence != predicted_occurrence))
    if task.zero_inflated and "positive_mae" in metric_names:
        positive_mask = y_true > threshold
        if positive_mask.any():
            row["positive_mae"] = float(np.mean(np.abs(y_true[positive_mask] - y_pred[positive_mask])))

    notes: dict[str, str] = {}
    if "mase" in metric_names and not np.isfinite(row["mase"]):
        notes["mase"] = "Undefined because the in-sample seasonal-naive scale is zero or unavailable."
    if "wql" in metric_names and row["wql_abs_target_sum"] == 0.0:
        notes["wql"] = "Undefined because every actual value in the forecast window is zero."
    if result.output.distribution == "degenerate" and "wql" in metric_names:
        notes["wql_distribution"] = (
            "Repeated point forecasts form a degenerate distribution; calibration is not measured."
        )
    if task.zero_inflated:
        notes["smape"] = "Zero-inflated target; interpret with rain occurrence and positive-only error."
        if "positive_mae" in metric_names and not np.isfinite(row["positive_mae"]):
            notes["positive_mae"] = "Undefined because the forecast window contains no positive actual values."
    row["metric_notes"] = json.dumps(notes, sort_keys=True) if notes else ""
    return row


def result_frame(rows: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=RESULT_COLUMNS)


def summarize(results: pd.DataFrame, metric_names: Sequence[str]) -> pd.DataFrame:
    if results.empty:
        return pd.DataFrame(
            columns=[
                "dataset",
                "model",
                *metric_names,
                "wql_macro",
                "wql_median",
                "evaluated_tasks",
                "successful_runs",
                "failed_runs",
            ]
        )
    grouped = results.groupby(["dataset", "model"], dropna=False)
    mean_metrics = [name for name in metric_names if name != "wql"]
    if mean_metrics:
        summary = grouped[mean_metrics].mean(numeric_only=True).reset_index()
    else:
        summary = grouped.size().rename("_size").reset_index().drop(columns="_size")
    if "wql" in metric_names:
        wql_components = grouped[["wql_loss_sum", "wql_abs_target_sum"]].sum(min_count=1).reset_index()
        wql_components["wql"] = wql_components["wql_loss_sum"] / wql_components["wql_abs_target_sum"]
        wql_macro = grouped["wql"].agg(wql_macro="mean", wql_median="median").reset_index()
        summary = summary.merge(
            wql_components[["dataset", "model", "wql"]],
            on=["dataset", "model"],
        ).merge(wql_macro, on=["dataset", "model"])
    run_counts = grouped.agg(
        successful_runs=("error", lambda values: int((values == "").sum())),
        failed_runs=("error", lambda values: int((values != "").sum())),
        diagnostic_model=("diagnostic_model", "max"),
    ).reset_index()
    task_counts = (
        results.assign(_successful=results["error"].eq(""))
        .groupby(["dataset", "model", "item_id"], dropna=False)["_successful"]
        .any()
        .groupby(["dataset", "model"])
        .agg(evaluated_tasks="size", successful_tasks="sum")
        .reset_index()
    )
    task_counts["failed_tasks"] = task_counts["evaluated_tasks"] - task_counts["successful_tasks"]
    counts = run_counts.merge(task_counts, on=["dataset", "model"])
    durations = grouped["duration_seconds"].sum().rename("total_duration_seconds").reset_index()
    return (
        summary.merge(counts, on=["dataset", "model"])
        .merge(durations, on=["dataset", "model"])
        .sort_values(["dataset", metric_names[0]], na_position="last")
    )
