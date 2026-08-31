"""Paired robustness evaluation for the PMDS forecasting benchmark.

This module intentionally sits beside the clean benchmark pipeline. It reuses
the existing dataset, model, prediction, and metric contracts while keeping the
four robustness interventions isolated from model implementations.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import logging
import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from pmds.config import load_config
from pmds.datasets import load_dataset_tasks
from pmds.metrics import compute_metrics, select_point_forecast
from pmds.models import build_model_runners
from pmds.outputs import atomic_write_csv, atomic_write_json
from pmds.pipeline import safe_predict
from pmds.runtime import apply_environment, seed_everything, setup_logging, stable_seed
from pmds.schemas import DatasetSpec, ForecastOutput, ForecastResult, ForecastTask
from pmds.utils import resolve_path


LOGGER = logging.getLogger("pmds.robustness")
SEVERITY_ORDER = ("mild", "moderate", "severe")
OUTLIER_SPECS = {
    "mild": (0.01, 3.0),
    "moderate": (0.03, 5.0),
    "severe": (0.05, 10.0),
}
OUTAGE_SPECS = {"mild": 0.25, "moderate": 0.5, "severe": 1.0}
BIAS_SPECS = {"mild": 0.25, "moderate": 1.0, "severe": 2.0}
SCALE_SPECS = {
    "mild": (0.5, 2.0),
    "moderate": (0.1, 10.0),
    "severe": (0.01, 100.0),
}


@dataclass(frozen=True)
class RobustnessScenario:
    test: str
    severity: str
    task: ForecastTask | None
    corruption_seed: int | None = None
    direction: str = ""
    factor: float | None = None
    robust_scale: float = float("nan")
    outage_length: int = 0
    selected_indices: tuple[int, ...] = ()
    selected_signs: tuple[int, ...] = ()
    not_applicable_reason: str = ""


def context_hash(task: ForecastTask) -> str:
    values = np.asarray(task.context, dtype="<f4")
    return hashlib.sha256(values.tobytes()).hexdigest()


def cap_context(task: ForecastTask, max_context: int) -> ForecastTask:
    if max_context < 1:
        raise ValueError("max_context must be positive")
    if len(task.context) <= max_context:
        return task
    return replace(
        task,
        context=task.context[-max_context:].copy(),
        context_timestamps=task.context_timestamps[-max_context:],
    )


def median_absolute_deviation(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if not len(values):
        return float("nan")
    median = float(np.median(values))
    return float(np.median(np.abs(values - median)))


def robust_scale(context: np.ndarray, seasonality: int) -> float:
    """Robust scale based on seasonal differences with documented fallbacks."""
    values = np.asarray(context, dtype=np.float64)
    lag = seasonality if seasonality > 0 and len(values) > seasonality else 1
    differences = values[lag:] - values[:-lag]
    scale = 1.4826 * median_absolute_deviation(differences)
    if not np.isfinite(scale) or scale == 0.0:
        scale = 1.4826 * median_absolute_deviation(values)
    if not np.isfinite(scale) or scale == 0.0:
        scale = float(np.std(values))
    if not np.isfinite(scale) or scale == 0.0:
        return float("nan")
    return scale


def mase_scale(context: np.ndarray, seasonality: int) -> float:
    values = np.asarray(context, dtype=np.float64)
    lag = seasonality if seasonality > 0 and len(values) > seasonality else 1
    differences = np.abs(values[lag:] - values[:-lag])
    if not len(differences):
        return float("nan")
    scale = float(np.mean(differences))
    return scale if np.isfinite(scale) and scale > 0.0 else float("nan")


def _scenario_task(task: ForecastTask, context: np.ndarray, future: np.ndarray | None = None) -> ForecastTask:
    return replace(
        task,
        context=np.asarray(context, dtype=np.float32),
        future=task.future if future is None else np.asarray(future, dtype=np.float32),
    )


def outlier_scenarios(
    task: ForecastTask,
    corruption_seeds: Sequence[int],
) -> list[RobustnessScenario]:
    scale = robust_scale(task.context, task.seasonality)
    scenarios: list[RobustnessScenario] = []
    window = min(2 * task.prediction_length, len(task.context))
    for severity, (fraction, amplitude) in OUTLIER_SPECS.items():
        for corruption_seed in corruption_seeds:
            if not np.isfinite(scale) or window == 0:
                scenarios.append(
                    RobustnessScenario(
                        "outlier",
                        severity,
                        None,
                        corruption_seed=int(corruption_seed),
                        robust_scale=scale,
                        not_applicable_reason="constant series or empty candidate window",
                    )
                )
                continue
            count = min(window, max(1, int(math.floor(fraction * window + 0.5))))
            rng = np.random.default_rng(
                stable_seed(
                    int(corruption_seed),
                    task.dataset,
                    task.series_id or task.item_id,
                    task.origin,
                    "outlier",
                    severity,
                )
            )
            candidates = np.arange(len(task.context) - window, len(task.context))
            indices = np.sort(rng.choice(candidates, size=count, replace=False))
            signs = rng.choice(np.asarray([-1, 1], dtype=int), size=count)
            context = task.context.astype(np.float64, copy=True)
            context[indices] += signs * amplitude * scale
            scenarios.append(
                RobustnessScenario(
                    "outlier",
                    severity,
                    _scenario_task(task, context),
                    corruption_seed=int(corruption_seed),
                    robust_scale=scale,
                    selected_indices=tuple(map(int, indices)),
                    selected_signs=tuple(map(int, signs)),
                )
            )
    return scenarios


def outage_scenarios(task: ForecastTask) -> list[RobustnessScenario]:
    scenarios: list[RobustnessScenario] = []
    horizon = task.prediction_length
    for severity, horizon_fraction in OUTAGE_SPECS.items():
        outage = int(math.ceil(horizon_fraction * horizon))
        if len(task.context) <= outage:
            scenarios.append(
                RobustnessScenario(
                    "outage",
                    severity,
                    None,
                    outage_length=outage,
                    not_applicable_reason="outage removes the complete context",
                )
            )
            continue
        forecast_timestamps = task.context_timestamps[-outage:].append(task.future_timestamps)
        forecast_values = np.concatenate([task.context[-outage:], task.future]).astype(np.float32)
        scenario_task = replace(
            task,
            context=task.context[:-outage].copy(),
            context_timestamps=task.context_timestamps[:-outage],
            future=forecast_values,
            future_timestamps=forecast_timestamps,
            prediction_length=outage + horizon,
        )
        scenarios.append(
            RobustnessScenario(
                "outage",
                severity,
                scenario_task,
                outage_length=outage,
            )
        )
    return scenarios


def bias_scenarios(task: ForecastTask) -> list[RobustnessScenario]:
    scale = robust_scale(task.context, task.seasonality)
    scenarios: list[RobustnessScenario] = []
    window = min(2 * task.prediction_length, len(task.context))
    for severity, magnitude in BIAS_SPECS.items():
        for sign, direction in ((1.0, "positive"), (-1.0, "negative")):
            if not np.isfinite(scale) or window == 0:
                scenarios.append(
                    RobustnessScenario(
                        "bias",
                        severity,
                        None,
                        direction=direction,
                        robust_scale=scale,
                        not_applicable_reason="constant series or empty candidate window",
                    )
                )
                continue
            context = task.context.astype(np.float64, copy=True)
            context[-window:] += sign * magnitude * scale
            scenarios.append(
                RobustnessScenario(
                    "bias",
                    severity,
                    _scenario_task(task, context),
                    direction=direction,
                    robust_scale=scale,
                )
            )
    return scenarios


def scale_scenarios(task: ForecastTask) -> list[RobustnessScenario]:
    return [
        RobustnessScenario(
            "scale",
            severity,
            _scenario_task(task, task.context * factor, task.future * factor),
            direction="down" if factor < 1.0 else "up",
            factor=factor,
        )
        for severity, factors in SCALE_SPECS.items()
        for factor in factors
    ]


def build_scenarios(task: ForecastTask, corruption_seeds: Sequence[int]) -> list[RobustnessScenario]:
    return [
        *outlier_scenarios(task, corruption_seeds),
        *outage_scenarios(task),
        *bias_scenarios(task),
        *scale_scenarios(task),
    ]


def normalize_scenario_result(
    result: ForecastResult,
    scenario: RobustnessScenario,
    clean_horizon: int,
) -> ForecastResult:
    """Return an H-step result in the original units for clean-task scoring."""
    if result.output is None:
        return result
    mean = result.output.mean
    quantiles = result.output.quantiles
    if scenario.test == "outage":
        start = scenario.outage_length
        mean = mean[start : start + clean_horizon]
        quantiles = quantiles[start : start + clean_horizon]
    elif scenario.test == "scale":
        if scenario.factor is None or scenario.factor == 0.0:
            raise ValueError("Scale scenario requires a non-zero factor")
        mean = mean / scenario.factor
        quantiles = quantiles / scenario.factor
    return replace(
        result,
        output=ForecastOutput(
            np.asarray(mean, dtype=np.float32),
            np.asarray(quantiles, dtype=np.float32),
            distribution=result.output.distribution,
        ),
    )


def scale_invariance_error(
    task: ForecastTask,
    clean_result: ForecastResult,
    stressed_result: ForecastResult,
    quantiles: np.ndarray,
    point_method: str,
) -> float:
    if clean_result.output is None or stressed_result.output is None:
        return float("nan")
    denominator = mase_scale(task.context, task.seasonality)
    if not np.isfinite(denominator):
        return float("nan")
    clean_point = select_point_forecast(clean_result.output, quantiles, point_method)
    stressed_point = select_point_forecast(stressed_result.output, quantiles, point_method)
    return float(np.mean(np.abs(clean_point - stressed_point)) / denominator)


def _model_family(model_config: Mapping[str, Any]) -> str:
    if model_config.get("family"):
        return str(model_config["family"])
    return "foundation" if model_config.get("type") in {"chronos", "timesfm", "moirai"} else "classical"


def _scenario_metadata(task: ForecastTask, scenario: RobustnessScenario) -> dict[str, Any]:
    return {
        "dataset": task.dataset,
        "series_id": task.series_id or task.item_id,
        "item_id": task.item_id,
        "origin": task.origin,
        "test": scenario.test,
        "severity": scenario.severity,
        "corruption_seed": scenario.corruption_seed,
        "direction": scenario.direction,
        "factor": scenario.factor,
        "robust_scale": scenario.robust_scale,
        "outage_length": scenario.outage_length,
        "selected_indices": json.dumps(scenario.selected_indices),
        "selected_signs": json.dumps(scenario.selected_signs),
        "context_hash": context_hash(task),
        "not_applicable_reason": scenario.not_applicable_reason,
    }


def _clean_row(
    task: ForecastTask,
    result: ForecastResult,
    metrics: Mapping[str, Any],
    model_family: str,
) -> dict[str, Any]:
    return {
        "dataset": task.dataset,
        "series_id": task.series_id or task.item_id,
        "item_id": task.item_id,
        "origin": task.origin,
        "model": result.model,
        "model_family": model_family,
        "model_seed": result.seed,
        "context_length": len(task.context),
        "forecast_horizon": task.prediction_length,
        "mase": metrics["mase"],
        "mae": metrics["mae"],
        "rmse": metrics["rmse"],
        "smape": metrics["smape"],
        "wql": (
            metrics["wql"]
            if result.output is not None and result.output.distribution != "degenerate"
            else float("nan")
        ),
        "runtime_seconds": result.duration_seconds,
        "status": "ok" if result.output is not None else "failed",
        "error_type": result.error_type,
        "error": result.error,
    }


def _robustness_row(
    task: ForecastTask,
    scenario: RobustnessScenario,
    model_name: str,
    model_family: str,
    model_seed: int,
    clean_result: ForecastResult,
    clean_metrics: Mapping[str, Any],
    stress_result: ForecastResult | None,
    stress_metrics: Mapping[str, Any] | None,
    quantiles: np.ndarray,
    point_method: str,
    ratio_epsilon: float,
) -> dict[str, Any]:
    row = _scenario_metadata(task, scenario)
    applicable = scenario.task is not None
    clean_mase = float(clean_metrics["mase"])
    stress_mase = float(stress_metrics["mase"]) if stress_metrics is not None else float("nan")
    ratio = (
        stress_mase / clean_mase
        if np.isfinite(clean_mase) and np.isfinite(stress_mase) and abs(clean_mase) > ratio_epsilon
        else float("nan")
    )
    invariance_error = float("nan")
    if scenario.test == "scale" and stress_result is not None:
        invariance_error = scale_invariance_error(
            task,
            clean_result,
            stress_result,
            quantiles,
            point_method,
        )
    if not applicable:
        status = "not_applicable"
        error_type = ""
        error = scenario.not_applicable_reason
    elif clean_result.output is None:
        status = "clean_failed"
        error_type = clean_result.error_type
        error = clean_result.error
    elif stress_result is None or stress_result.output is None:
        status = "stress_failed"
        error_type = stress_result.error_type if stress_result is not None else ""
        error = stress_result.error if stress_result is not None else ""
    else:
        status = "ok"
        error_type = ""
        error = ""
    row.update(
        {
            "model": model_name,
            "model_family": model_family,
            "model_seed": model_seed,
            "context_length": len(task.context),
            "forecast_horizon": task.prediction_length,
            "clean_mase": clean_mase,
            "stress_mase": stress_mase,
            "clean_mae": float(clean_metrics["mae"]),
            "stress_mae": float(stress_metrics["mae"]) if stress_metrics is not None else float("nan"),
            "robustness_ratio": ratio,
            "percentage_degradation": 100.0 * (ratio - 1.0) if np.isfinite(ratio) else float("nan"),
            "absolute_degradation": stress_mase - clean_mase,
            "scale_invariance_error": invariance_error,
            "clean_runtime_seconds": clean_result.duration_seconds,
            "stress_runtime_seconds": stress_result.duration_seconds if stress_result is not None else float("nan"),
            "runtime_seconds": (
                clean_result.duration_seconds + stress_result.duration_seconds
                if stress_result is not None
                else clean_result.duration_seconds
            ),
            "applicable": applicable,
            "status": status,
            "error_type": error_type,
            "error": error,
        }
    )
    return row


def _series_level(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    data = frame.copy()
    data["failed"] = data["applicable"].astype(bool) & data["status"].ne("ok")
    applicable = data[data["applicable"].astype(bool)]
    group = ["dataset", "series_id", "model", "model_family", "test", "severity"]
    metrics = [
        "clean_mase",
        "stress_mase",
        "robustness_ratio",
        "percentage_degradation",
        "absolute_degradation",
        "scale_invariance_error",
        "runtime_seconds",
    ]
    if applicable.empty:
        return pd.DataFrame(columns=[*group, *metrics, "failure_rate"])
    values = applicable.groupby(group, dropna=False)[metrics].mean().reset_index()
    failures = applicable.groupby(group, dropna=False)["failed"].mean().rename("failure_rate").reset_index()
    return values.merge(failures, on=group)


def _hierarchical_interval(
    group: pd.DataFrame,
    metric: str,
    samples: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    if samples <= 0 or group[metric].notna().sum() == 0:
        return float("nan"), float("nan")
    datasets = group["dataset"].drop_duplicates().to_numpy()
    estimates: list[float] = []
    for _ in range(samples):
        dataset_estimates: list[float] = []
        for dataset in rng.choice(datasets, size=len(datasets), replace=True):
            values = group.loc[group["dataset"] == dataset, metric].dropna().to_numpy(dtype=float)
            if len(values):
                dataset_estimates.append(float(np.mean(rng.choice(values, size=len(values), replace=True))))
        if dataset_estimates:
            estimates.append(float(np.mean(dataset_estimates)))
    if not estimates:
        return float("nan"), float("nan")
    return tuple(map(float, np.quantile(estimates, [0.025, 0.975])))


def aggregate_robustness(
    frame: pd.DataFrame,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    series = _series_level(frame)
    if series.empty:
        return pd.DataFrame()
    metrics = [
        "clean_mase",
        "stress_mase",
        "robustness_ratio",
        "percentage_degradation",
        "absolute_degradation",
        "scale_invariance_error",
        "runtime_seconds",
        "failure_rate",
    ]
    dataset_group = ["dataset", "model", "model_family", "test", "severity"]
    dataset_level = series.groupby(dataset_group, dropna=False)[metrics].mean().reset_index()
    summary_group = ["model", "model_family", "test", "severity"]
    summary = dataset_level.groupby(summary_group, dropna=False)[metrics].mean().reset_index()
    rng = np.random.default_rng(bootstrap_seed)
    interval_metrics = ("robustness_ratio", "absolute_degradation", "scale_invariance_error")
    interval_rows: list[dict[str, Any]] = []
    for keys, group in series.groupby(summary_group, dropna=False):
        row = dict(zip(summary_group, keys))
        for metric in interval_metrics:
            lower, upper = _hierarchical_interval(group, metric, bootstrap_samples, rng)
            row[f"{metric}_ci_lower"] = lower
            row[f"{metric}_ci_upper"] = upper
        interval_rows.append(row)
    summary = summary.merge(pd.DataFrame(interval_rows), on=summary_group, how="left")
    severity = pd.Categorical(summary["severity"], categories=SEVERITY_ORDER, ordered=True)
    return summary.assign(_severity=severity).sort_values(["test", "_severity", "model"]).drop(columns="_severity")


def plot_severity_curves(summary: pd.DataFrame, output_dir: Path) -> list[Path]:
    if summary.empty:
        return []
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        LOGGER.warning("matplotlib is unavailable; skipping robustness severity curves")
        return []
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for test, test_frame in summary.groupby("test"):
        metric = "scale_invariance_error" if test == "scale" else "robustness_ratio"
        lower_name = f"{metric}_ci_lower"
        upper_name = f"{metric}_ci_upper"
        figure, axis = plt.subplots(figsize=(8, 5))
        for model, model_frame in test_frame.groupby("model"):
            ordered = model_frame.set_index("severity").reindex(SEVERITY_ORDER)
            values = ordered[metric].to_numpy(dtype=float)
            axis.plot(SEVERITY_ORDER, values, marker="o", label=model)
            if lower_name in ordered and upper_name in ordered:
                axis.fill_between(
                    SEVERITY_ORDER,
                    ordered[lower_name].to_numpy(dtype=float),
                    ordered[upper_name].to_numpy(dtype=float),
                    alpha=0.12,
                )
        axis.axhline(0.0 if test == "scale" else 1.0, color="black", linestyle="--", linewidth=1)
        axis.set_title(f"{test.title()} robustness")
        axis.set_xlabel("Severity")
        axis.set_ylabel("Scale deviation" if test == "scale" else "Stressed MASE / clean MASE")
        axis.legend(fontsize="small")
        figure.tight_layout()
        path = plot_dir / f"{test}_severity.png"
        figure.savefig(path, dpi=160)
        plt.close(figure)
        paths.append(path)
    return paths


def _prepare_config(config: Mapping[str, Any], audit: bool) -> dict[str, Any]:
    prepared = copy.deepcopy(config)
    robustness = prepared["robustness"]
    max_series = int(robustness["max_series_per_dataset"])
    num_origins = int(robustness["num_origins"])
    enabled_seen = False
    for dataset in prepared["datasets"]:
        if not dataset.get("enabled", True):
            continue
        if audit and enabled_seen:
            dataset["enabled"] = False
            continue
        enabled_seen = True
        dataset["max_series"] = 1 if audit else min(int(dataset["max_series"]), max_series)
        dataset["num_origins"] = 1 if audit else num_origins
    output_name = "robustness_audit" if audit else "robustness"
    output_dir = Path(str(prepared["run"]["output_dir"])) / output_name
    prepared["run"]["name"] = output_name
    prepared["run"]["output_dir"] = str(output_dir)
    prepared["run"]["log_dir"] = str(output_dir / "logs")
    if audit:
        robustness["model_seeds"] = robustness["model_seeds"][:1]
        robustness["corruption_seeds"] = robustness["corruption_seeds"][:1]
        robustness["bootstrap_samples"] = 0
    return prepared


def run_robustness(config: Mapping[str, Any], config_path: Path, audit: bool = False) -> dict[str, Path]:
    prepared = _prepare_config(config, audit)
    run_config = prepared["run"]
    robustness = prepared["robustness"]
    evaluation = prepared["evaluation"]
    quantiles = np.asarray(evaluation["quantiles"], dtype=np.float32)
    metric_names = list(dict.fromkeys([*evaluation["metrics"], "mae", "rmse", "smape", "mase", "wql"]))
    point_method = str(evaluation["point_forecast"])
    model_seeds = list(map(int, robustness["model_seeds"]))
    corruption_seeds = list(map(int, robustness["corruption_seeds"]))
    max_context = int(robustness["max_context"])
    ratio_epsilon = float(robustness["ratio_epsilon"])

    log_path = setup_logging(run_config)
    apply_environment(run_config.get("environment", {}))
    LOGGER.info("Robustness experiment started | config=%s audit=%s log=%s", config_path, audit, log_path)

    model_configs = [
        value
        for value in prepared["models"]
        if value.get("enabled", True) and not value.get("diagnostic", False)
    ]
    runners = build_model_runners(model_configs, quantiles)
    model_config_by_name = {str(value["name"]): value for value in model_configs}
    dataset_specs = [
        DatasetSpec.from_config(value) for value in prepared["datasets"] if value.get("enabled", True)
    ]
    if not runners or not dataset_specs:
        raise ValueError("Robustness evaluation requires enabled non-diagnostic models and datasets")

    output_dir = resolve_path(run_config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    clean_path = output_dir / "clean.csv"
    long_path = output_dir / "robustness.csv"
    manifest_path = output_dir / "corruption_manifest.csv"
    status_path = output_dir / "status.json"
    summary_path = output_dir / "summary.csv"
    moderate_path = output_dir / "moderate.csv"
    severe_path = output_dir / "severe.csv"

    clean_rows: list[dict[str, Any]] = []
    robustness_rows: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []
    statuses: list[dict[str, Any]] = []

    for spec in dataset_specs:
        try:
            tasks = [cap_context(task, max_context) for task in load_dataset_tasks(spec)]
            if not tasks:
                raise RuntimeError(f"Dataset {spec.name} produced no valid robustness tasks")
            for task in tasks:
                scenarios = build_scenarios(task, corruption_seeds)
                manifest_rows.extend(_scenario_metadata(task, scenario) for scenario in scenarios)
                for model_name, runner in runners.items():
                    model_config = model_config_by_name[model_name]
                    scopes = set(map(str, model_config.get("datasets", [])))
                    if scopes and task.dataset not in scopes:
                        continue
                    seeds = model_seeds if model_config.get("stochastic", False) else model_seeds[:1]
                    family = _model_family(model_config)
                    for repetition, model_seed in enumerate(seeds):
                        seed_everything(model_seed)
                        clean_result = safe_predict(
                            model_name,
                            task,
                            runner,
                            quantiles,
                            repetition,
                            model_seed,
                        )
                        clean_metrics = compute_metrics(
                            task,
                            clean_result,
                            metric_names,
                            quantiles,
                            point_method,
                            bool(model_config.get("diagnostic", False)),
                        )
                        clean_rows.append(_clean_row(task, clean_result, clean_metrics, family))
                        for scenario in scenarios:
                            stress_result: ForecastResult | None = None
                            stress_metrics: Mapping[str, Any] | None = None
                            if scenario.task is not None:
                                seed_everything(model_seed)
                                raw_result = safe_predict(
                                    model_name,
                                    scenario.task,
                                    runner,
                                    quantiles,
                                    repetition,
                                    model_seed,
                                )
                                stress_result = normalize_scenario_result(
                                    raw_result,
                                    scenario,
                                    task.prediction_length,
                                )
                                stress_metrics = compute_metrics(
                                    task,
                                    stress_result,
                                    metric_names,
                                    quantiles,
                                    point_method,
                                    bool(model_config.get("diagnostic", False)),
                                )
                            robustness_rows.append(
                                _robustness_row(
                                    task,
                                    scenario,
                                    model_name,
                                    family,
                                    model_seed,
                                    clean_result,
                                    clean_metrics,
                                    stress_result,
                                    stress_metrics,
                                    quantiles,
                                    point_method,
                                    ratio_epsilon,
                                )
                            )
            dataset_rows = [row for row in robustness_rows if row["dataset"] == spec.name]
            failures = sum(row["status"] not in {"ok", "not_applicable"} for row in dataset_rows)
            statuses.append(
                {
                    "dataset": spec.name,
                    "status": "completed_with_errors" if failures else "completed",
                    "tasks": len(tasks),
                    "robustness_rows": len(dataset_rows),
                    "failed_rows": failures,
                }
            )
        except Exception as exc:
            LOGGER.exception("Robustness dataset failed | dataset=%s", spec.name)
            statuses.append(
                {
                    "dataset": spec.name,
                    "status": "failed",
                    "tasks": 0,
                    "robustness_rows": 0,
                    "failed_rows": 0,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
        atomic_write_csv(pd.DataFrame(clean_rows), clean_path)
        atomic_write_csv(pd.DataFrame(robustness_rows), long_path)
        manifest = pd.DataFrame(manifest_rows).drop_duplicates() if manifest_rows else pd.DataFrame()
        atomic_write_csv(manifest, manifest_path)
        atomic_write_json(statuses, status_path)

    long_frame = pd.DataFrame(robustness_rows)
    summary = aggregate_robustness(
        long_frame,
        int(robustness["bootstrap_samples"]),
        int(corruption_seeds[0]),
    )
    atomic_write_csv(summary, summary_path)
    atomic_write_csv(summary.loc[summary["severity"] == "moderate"] if not summary.empty else summary, moderate_path)
    atomic_write_csv(summary.loc[summary["severity"] == "severe"] if not summary.empty else summary, severe_path)
    plot_severity_curves(summary, output_dir)
    LOGGER.info("Robustness experiment completed | rows=%d output=%s", len(robustness_rows), output_dir)
    return {
        "clean": clean_path,
        "robustness": long_path,
        "manifest": manifest_path,
        "status": status_path,
        "summary": summary_path,
        "moderate": moderate_path,
        "severe": severe_path,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the paired PMDS robustness benchmark.")
    parser.add_argument("--config", type=Path, default=Path("pmds/config.json"))
    parser.add_argument(
        "--audit",
        action="store_true",
        help="Run the first enabled dataset, one series, one origin, and one seed.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = args.config.expanduser().resolve()
    paths = run_robustness(load_config(config_path), config_path, audit=args.audit)
    print("\nRobustness outputs")
    for name, path in paths.items():
        print(f"{name:>12}: {path}")


if __name__ == "__main__":
    main()
