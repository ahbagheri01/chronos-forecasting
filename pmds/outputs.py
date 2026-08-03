"""Result persistence helpers for PMDS benchmarks."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import pandas as pd

from pmds.metrics import result_frame, summarize
from pmds.schemas import ForecastOutput, ForecastTask
from pmds.utils import resolve_path


LOGGER = logging.getLogger("pmds.compare")


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
) -> tuple[Path, Path, Path]:
    output_dir = resolve_path(run_config["output_dir"])
    run_name = str(run_config["name"])
    detailed_path = output_dir / f"{run_name}_detailed.csv"
    summary_path = output_dir / f"{run_name}_summary.csv"
    status_path = output_dir / f"{run_name}_status.json"
    detailed = result_frame(rows)
    summary = summarize(detailed, metric_names)
    atomic_write_csv(detailed, detailed_path)
    atomic_write_csv(summary, summary_path)
    atomic_write_json(list(statuses), status_path)
    LOGGER.info(
        "Results checkpoint saved | detailed=%s summary=%s status=%s rows=%d",
        detailed_path,
        summary_path,
        status_path,
        len(detailed),
    )
    return detailed_path, summary_path, status_path


def dataset_failure_rows(
    dataset_name: str,
    runners: Mapping[str, Callable[[ForecastTask], ForecastOutput]],
    exc: Exception,
) -> list[dict[str, Any]]:
    return [
        {
            "dataset": dataset_name,
            "item_id": "__dataset_load__",
            "model": model_name,
            "mae": float("nan"),
            "rmse": float("nan"),
            "smape": float("nan"),
            "mase": float("nan"),
            "wql": float("nan"),
            "wql_loss_sum": float("nan"),
            "wql_abs_target_sum": float("nan"),
            "duration_seconds": 0.0,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        for model_name in runners
    ]
