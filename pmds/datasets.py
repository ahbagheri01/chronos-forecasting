"""Dataset loading and task construction for PMDS benchmarks."""

from __future__ import annotations

import logging
from typing import Iterable

import numpy as np
import pandas as pd

from pmds.schemas import DatasetSpec, ForecastTask
from pmds.utils import clean_numeric


LOGGER = logging.getLogger("pmds.compare")


def import_datasets():
    try:
        import datasets
    except ImportError as exc:
        raise ImportError("Install datasets with: pip install datasets") from exc
    return datasets


def infer_frequency(timestamps: pd.DatetimeIndex, fallback: str) -> str:
    if len(timestamps) >= 3:
        frequency = pd.infer_freq(timestamps)
        if frequency is not None:
            return frequency
    return fallback


def align_timestamps(values: np.ndarray, timestamps: Iterable | None, fallback: str) -> pd.DatetimeIndex:
    if timestamps is not None:
        index = pd.DatetimeIndex(pd.to_datetime(timestamps))
        if len(index) == len(values):
            return index
        LOGGER.warning(
            "Timestamp/value length mismatch; using synthetic timestamps | values=%d timestamps=%d fallback=%s",
            len(values),
            len(index),
            fallback,
        )
    return pd.date_range("2000-01-01", periods=len(values), freq=fallback)


def load_hf_dataset(spec: DatasetSpec):
    datasets = import_datasets()
    last_error: Exception | None = None
    for hf_config in spec.hf_configs:
        try:
            LOGGER.info(
                "Loading Hugging Face dataset | dataset=%s repo=%s config=%s split=%s",
                spec.name,
                spec.repo,
                hf_config,
                spec.split,
            )
            return datasets.load_dataset(
                spec.repo,
                hf_config,
                split=spec.split,
                trust_remote_code=spec.trust_remote_code,
            )
        except Exception as exc:
            last_error = exc
            LOGGER.warning(
                "Dataset config failed | dataset=%s config=%s error=%s",
                spec.name,
                hf_config,
                exc,
                exc_info=True,
            )
    raise RuntimeError(f"Could not load {spec.name} from {spec.repo} with configs={spec.hf_configs}") from last_error


def infer_sequence_columns(hf_dataset) -> list[str]:
    datasets = import_datasets()
    return [
        column
        for column, feature in hf_dataset.features.items()
        if isinstance(feature, datasets.Sequence) and column != "timestamp"
    ]


def make_task(
    spec: DatasetSpec,
    item_id: str,
    values: Iterable,
    timestamps: Iterable | None,
) -> ForecastTask | None:
    series = clean_numeric(values)
    if len(series) <= spec.prediction_length + 1:
        LOGGER.warning(
            "Skipping short series | dataset=%s item=%s length=%d horizon=%d",
            spec.name,
            item_id,
            len(series),
            spec.prediction_length,
        )
        return None

    timestamp_index = align_timestamps(series, timestamps, fallback=spec.fallback_frequency)
    frequency = infer_frequency(timestamp_index, fallback=spec.fallback_frequency)
    return ForecastTask(
        dataset=spec.name,
        item_id=item_id,
        context_timestamps=timestamp_index[: -spec.prediction_length],
        future_timestamps=timestamp_index[-spec.prediction_length :],
        context=series[: -spec.prediction_length],
        future=series[-spec.prediction_length :],
        prediction_length=spec.prediction_length,
        seasonality=spec.seasonality,
        frequency=frequency,
    )


def load_chronos_tasks(spec: DatasetSpec) -> list[ForecastTask]:
    dataset = load_hf_dataset(spec)
    sequence_columns = infer_sequence_columns(dataset)
    if not sequence_columns:
        raise ValueError(f"No sequence target columns found in {spec.name}")

    tasks: list[ForecastTask] = []
    for row_index, row in enumerate(dataset):
        timestamps = row.get("timestamp")
        for field in sequence_columns:
            task = make_task(spec, f"{row_index}:{field}", row[field], timestamps)
            if task is not None:
                tasks.append(task)
            if len(tasks) >= spec.max_series:
                return tasks
    return tasks


def load_external_tasks(spec: DatasetSpec) -> list[ForecastTask]:
    dataset = load_hf_dataset(spec)
    frame = dataset.to_pandas()

    date_column = spec.date_column
    if date_column is None:
        date_column = next(
            (candidate for candidate in ("date", "timestamp", "time", "datetime") if candidate in frame.columns),
            None,
        )
    timestamps = frame[date_column] if date_column is not None and date_column in frame.columns else None
    numeric_columns = list(frame.select_dtypes(include=[np.number]).columns)
    if date_column in numeric_columns:
        numeric_columns.remove(date_column)
    numeric_columns = [column for column in numeric_columns if column not in spec.exclude_columns]
    if spec.target_column is not None:
        if spec.target_column not in numeric_columns:
            raise ValueError(f"Target column '{spec.target_column}' is not numeric or missing in {spec.name}")
        numeric_columns = [spec.target_column]
    if not numeric_columns:
        raise ValueError(f"No numeric target columns found in {spec.name}")

    tasks: list[ForecastTask] = []
    for column in numeric_columns:
        task = make_task(spec, str(column), frame[column], timestamps)
        if task is not None:
            tasks.append(task)
        if len(tasks) >= spec.max_series:
            break
    return tasks


def load_dataset_tasks(spec: DatasetSpec) -> list[ForecastTask]:
    if spec.family == "chronos":
        return load_chronos_tasks(spec)
    if spec.family == "external":
        return load_external_tasks(spec)
    raise ValueError(f"Unsupported dataset family: {spec.family}")
