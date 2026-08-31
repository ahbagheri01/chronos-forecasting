"""Experiment orchestration for PMDS benchmark runs."""

from __future__ import annotations

import json
import logging
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
import pandas as pd
import torch

from pmds.datasets import load_dataset_tasks
from pmds.metrics import compute_metrics, empty_metric_row, validate_output
from pmds.models import build_model_runners
from pmds.outputs import (
    context_rows_for_task,
    dataset_failure_rows,
    forecast_rows_for_result,
    persist_results,
)
from pmds.runtime import apply_environment, seed_everything, setup_logging, stable_seed
from pmds.schemas import DatasetSpec, ForecastOutput, ForecastResult, ForecastTask


LOGGER = logging.getLogger("pmds.compare")


def safe_predict(
    model_name: str,
    task: ForecastTask,
    runner: Callable[[ForecastTask], ForecastOutput],
    quantiles: np.ndarray,
    repetition: int,
    seed: int,
) -> ForecastResult:
    start = time.monotonic()
    LOGGER.info(
        "Model started | dataset=%s item=%s model=%s repetition=%d seed=%d",
        task.dataset,
        task.item_id,
        model_name,
        repetition,
        seed,
    )
    try:
        output = runner(task)
        crossing_count = validate_output(output, task, quantiles)
        if crossing_count:
            LOGGER.warning(
                "Quantile crossing detected | dataset=%s item=%s crossings=%d",
                task.dataset,
                task.item_id,
                crossing_count,
            )
        duration = time.monotonic() - start
        LOGGER.info(
            "Model completed | dataset=%s item=%s model=%s duration_seconds=%.3f",
            task.dataset,
            task.item_id,
            model_name,
            duration,
        )
        return ForecastResult(
            task.dataset,
            task.item_id,
            model_name,
            output,
            duration,
            repetition=repetition,
            seed=seed,
        )
    except Exception as exc:
        duration = time.monotonic() - start
        LOGGER.exception(
            "Model failed | dataset=%s item=%s model=%s duration_seconds=%.3f",
            task.dataset,
            task.item_id,
            model_name,
            duration,
        )
        return ForecastResult(
            dataset=task.dataset,
            item_id=task.item_id,
            model=model_name,
            output=None,
            duration_seconds=duration,
            error_type=type(exc).__name__,
            error=str(exc),
            repetition=repetition,
            seed=seed,
        )


def cap_task_context(task: ForecastTask, max_context: int) -> ForecastTask:
    if len(task.context) <= max_context:
        return task
    return replace(
        task,
        context=task.context[-max_context:].copy(),
        context_timestamps=task.context_timestamps[-max_context:],
    )


def run_experiment(config: Mapping[str, Any], config_path: Path) -> None:
    run_config = config["run"]
    evaluation_config = config["evaluation"]
    quantiles = np.asarray(evaluation_config["quantiles"], dtype=np.float32)
    metric_names = list(evaluation_config["metrics"])
    point_method = str(evaluation_config["point_forecast"])
    stochastic_repetitions = int(evaluation_config.get("stochastic_repetitions", 1))
    context_multiplier = int(evaluation_config.get("forecast_context_multiplier", 3))
    max_context = int(evaluation_config["max_context"])
    base_seed = int(run_config["random_seed"])

    log_path = setup_logging(run_config)
    apply_environment(run_config.get("environment", {}))
    seed_everything(base_seed)
    LOGGER.info("Experiment started | name=%s config=%s log=%s", run_config["name"], config_path, log_path)
    LOGGER.info(
        "Runtime | python=%s torch=%s torch_cuda=%s cuda_available=%s",
        sys.version.split()[0],
        torch.__version__,
        torch.version.cuda,
        torch.cuda.is_available(),
    )
    LOGGER.debug("Resolved config:\n%s", json.dumps(config, indent=2))

    runners = build_model_runners(config["models"], quantiles)
    if not runners:
        raise ValueError("No models are enabled in config.json")
    dataset_specs = [DatasetSpec.from_config(value) for value in config["datasets"] if value.get("enabled", True)]
    if not dataset_specs:
        raise ValueError("No datasets are enabled in config.json")
    model_configs = {
        str(value["name"]): value for value in config["models"] if value.get("enabled", True)
    }

    rows: list[dict[str, Any]] = []
    forecast_rows: list[dict[str, Any]] = []
    context_rows: list[dict[str, Any]] = []
    statuses: list[dict[str, Any]] = []
    detailed_path = summary_path = status_path = forecast_path = context_path = Path()

    for dataset_index, spec in enumerate(dataset_specs, start=1):
        dataset_start = time.monotonic()
        LOGGER.info(
            "Dataset started | dataset=%s index=%d total=%d",
            spec.name,
            dataset_index,
            len(dataset_specs),
        )
        try:
            tasks = [cap_task_context(task, max_context) for task in load_dataset_tasks(spec)]
            if not tasks:
                raise RuntimeError(f"Dataset {spec.name} produced no forecast tasks")
            LOGGER.info("Dataset loaded | dataset=%s tasks=%d", spec.name, len(tasks))
            for task_index, task in enumerate(tasks, start=1):
                context_rows.extend(context_rows_for_task(task, context_multiplier))
                LOGGER.info(
                    "Task started | dataset=%s item=%s task=%d/%d context=%d horizon=%d frequency=%s",
                    spec.name,
                    task.item_id,
                    task_index,
                    len(tasks),
                    len(task.context),
                    task.prediction_length,
                    task.frequency,
                )
                for model_name, runner in runners.items():
                    model_config = model_configs[model_name]
                    model_datasets = set(map(str, model_config.get("datasets", [])))
                    if model_datasets and task.dataset not in model_datasets:
                        continue
                    repetitions = stochastic_repetitions if model_config.get("stochastic", False) else 1
                    diagnostic_model = bool(model_config.get("diagnostic", False))
                    for repetition in range(repetitions):
                        seed = stable_seed(base_seed, task.dataset, task.item_id, model_name, repetition)
                        seed_everything(seed)
                        result = safe_predict(
                            model_name,
                            task,
                            runner,
                            quantiles,
                            repetition,
                            seed,
                        )
                        try:
                            rows.append(
                                compute_metrics(
                                    task,
                                    result,
                                    metric_names,
                                    quantiles,
                                    point_method,
                                    diagnostic_model,
                                )
                            )
                            forecast_rows.extend(
                                forecast_rows_for_result(task, result, quantiles, point_method)
                            )
                        except Exception as exc:
                            LOGGER.exception(
                                "Metric computation failed | dataset=%s item=%s model=%s repetition=%d",
                                task.dataset,
                                task.item_id,
                                model_name,
                                repetition,
                            )
                            metric_failure = ForecastResult(
                                dataset=task.dataset,
                                item_id=task.item_id,
                                model=model_name,
                                output=None,
                                duration_seconds=result.duration_seconds,
                                error_type=type(exc).__name__,
                                error=str(exc),
                                repetition=repetition,
                                seed=seed,
                            )
                            rows.append(empty_metric_row(task, metric_failure, diagnostic_model))

            failed_rows = sum(1 for row in rows if row["dataset"] == spec.name and row["error"])
            statuses.append(
                {
                    "dataset": spec.name,
                    "status": "completed_with_errors" if failed_rows else "completed",
                    "tasks": len(tasks),
                    "failed_model_tasks": failed_rows,
                    "duration_seconds": time.monotonic() - dataset_start,
                }
            )
            LOGGER.info(
                "Dataset completed | dataset=%s tasks=%d failures=%d duration_seconds=%.3f",
                spec.name,
                len(tasks),
                failed_rows,
                time.monotonic() - dataset_start,
            )
        except Exception as exc:
            LOGGER.exception("Dataset failed | dataset=%s", spec.name)
            applicable_runners = {
                name: runner
                for name, runner in runners.items()
                if not model_configs[name].get("datasets") or spec.name in model_configs[name]["datasets"]
            }
            rows.extend(dataset_failure_rows(spec.name, applicable_runners, exc))
            statuses.append(
                {
                    "dataset": spec.name,
                    "status": "failed",
                    "tasks": 0,
                    "failed_model_tasks": len(runners),
                    "duration_seconds": time.monotonic() - dataset_start,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )

        detailed_path, summary_path, status_path, forecast_path, context_path = persist_results(
            rows,
            statuses,
            run_config,
            metric_names,
            forecast_rows,
            context_rows,
        )

    summary = pd.read_csv(summary_path)
    LOGGER.info("Experiment completed | datasets=%d result_rows=%d", len(dataset_specs), len(rows))
    print("\nSummary")
    print(summary.to_string(index=False))
    print(f"\nDetailed results: {detailed_path}")
    print(f"Summary results:  {summary_path}")
    print(f"Dataset status:  {status_path}")
    print(f"Forecast values: {forecast_path}")
    print(f"Plot contexts:   {context_path}")
    print(f"Log file:        {log_path}")
