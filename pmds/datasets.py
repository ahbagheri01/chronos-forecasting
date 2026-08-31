"""Dataset loading and task construction for PMDS benchmarks."""

from __future__ import annotations

import logging
from typing import Iterable, Sequence, TypeVar

import numpy as np
import pandas as pd

from pmds.schemas import DatasetSpec, ForecastTask
from pmds.utils import clean_numeric, numeric_array


LOGGER = logging.getLogger("pmds.compare")
T = TypeVar("T")


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
            kwargs = {
                "split": spec.split,
                "trust_remote_code": spec.trust_remote_code,
            }
            if spec.revision is not None:
                kwargs["revision"] = spec.revision
            return datasets.load_dataset(spec.repo, hf_config, **kwargs)
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


def select_representative_items(
    items: Sequence[T],
    max_items: int,
    strategy: str,
    seed: int,
) -> list[T]:
    candidates = list(items)
    if max_items <= 0 or max_items >= len(candidates):
        return candidates
    if strategy == "first":
        return candidates[:max_items]
    if strategy == "evenly_spaced":
        if max_items == 1:
            return [candidates[len(candidates) // 2]]
        indices = np.linspace(0, len(candidates) - 1, num=max_items, dtype=int)
        return [candidates[int(index)] for index in indices]
    if strategy == "random":
        rng = np.random.default_rng(seed)
        indices = sorted(rng.choice(len(candidates), size=max_items, replace=False).tolist())
        return [candidates[int(index)] for index in indices]
    raise ValueError(f"Unsupported selection strategy: {strategy}")


def make_tasks(
    spec: DatasetSpec,
    item_id: str,
    values: Iterable,
    timestamps: Iterable | None,
) -> list[ForecastTask]:
    series = numeric_array(values)
    stride = spec.origin_stride or spec.prediction_length
    required_length = required_series_length(spec)
    if len(series) < required_length:
        LOGGER.warning(
            "Series has fewer rolling origins than requested | dataset=%s item=%s length=%d required=%d",
            spec.name,
            item_id,
            len(series),
            required_length,
        )

    timestamp_index = align_timestamps(series, timestamps, fallback=spec.fallback_frequency)
    frequency = infer_frequency(timestamp_index, fallback=spec.fallback_frequency)
    tasks: list[ForecastTask] = []
    for origin in range(spec.num_origins):
        future_end = len(series) - origin * stride
        future_start = future_end - spec.prediction_length
        if future_start <= 1:
            continue
        raw_context = series[:future_start]
        future = series[future_start:future_end]
        missing_fraction = float(np.mean(~np.isfinite(raw_context)))
        if missing_fraction > 0.05:
            LOGGER.warning(
                "Skipping task with excessive context missingness | "
                "dataset=%s item=%s origin=%d missing_fraction=%.4f",
                spec.name,
                item_id,
                origin,
                missing_fraction,
            )
            continue
        if not np.isfinite(future).all():
            LOGGER.warning(
                "Skipping task with missing forecast target | dataset=%s item=%s origin=%d",
                spec.name,
                item_id,
                origin,
            )
            continue
        try:
            context = clean_numeric(raw_context)
        except ValueError as exc:
            LOGGER.warning(
                "Skipping task with invalid causal context | dataset=%s item=%s origin=%d error=%s",
                spec.name,
                item_id,
                origin,
                exc,
            )
            continue
        imputed_points = int(np.sum(~np.isfinite(raw_context)))
        if imputed_points:
            LOGGER.info(
                "Causally imputed context | dataset=%s item=%s origin=%d points=%d",
                spec.name,
                item_id,
                origin,
                imputed_points,
            )
        tasks.append(
            ForecastTask(
                dataset=spec.name,
                item_id=f"{item_id}::origin={origin}",
                context_timestamps=timestamp_index[:future_start],
                future_timestamps=timestamp_index[future_start:future_end],
                context=context,
                future=future.astype(np.float32),
                prediction_length=spec.prediction_length,
                seasonality=spec.seasonality,
                frequency=frequency,
                series_id=item_id,
                origin=origin,
                zero_inflated=spec.zero_inflated,
                zero_threshold=spec.zero_threshold,
            )
        )
    return tasks


def required_series_length(spec: DatasetSpec) -> int:
    stride = spec.origin_stride or spec.prediction_length
    minimum_context = max(8, spec.seasonality)
    return spec.prediction_length + (spec.num_origins - 1) * stride + minimum_context


def supports_requested_origins(values: Iterable, spec: DatasetSpec) -> bool:
    series = numeric_array(values)
    if len(series) < required_series_length(spec):
        return False
    stride = spec.origin_stride or spec.prediction_length
    for origin in range(spec.num_origins):
        future_end = len(series) - origin * stride
        future_start = future_end - spec.prediction_length
        if future_start <= 1:
            return False
        context = series[:future_start]
        future = series[future_start:future_end]
        if not np.isfinite(future).all() or not np.isfinite(context).any():
            return False
        if float(np.mean(~np.isfinite(context))) > 0.05:
            return False
    return True


def matching_row_indices(hf_dataset, spec: DatasetSpec) -> list[int]:
    if spec.filter_column is None:
        return list(range(len(hf_dataset)))
    if spec.filter_column not in hf_dataset.column_names:
        raise ValueError(f"Filter column '{spec.filter_column}' is missing from {spec.name}")
    allowed = set(spec.filter_values)
    indices = [
        index
        for index, value in enumerate(hf_dataset[spec.filter_column])
        if str(value) in allowed
    ]
    if not indices:
        raise ValueError(
            f"Dataset {spec.name} has no rows where {spec.filter_column} is one of {sorted(allowed)}"
        )
    LOGGER.info(
        "Dataset rows filtered | dataset=%s column=%s values=%s matched=%d total=%d",
        spec.name,
        spec.filter_column,
        sorted(allowed),
        len(indices),
        len(hf_dataset),
    )
    return indices


def make_task(
    spec: DatasetSpec,
    item_id: str,
    values: Iterable,
    timestamps: Iterable | None,
) -> ForecastTask | None:
    """Backward-compatible helper returning the most recent forecast origin."""
    tasks = make_tasks(spec, item_id, values, timestamps)
    return tasks[0] if tasks else None


def load_chronos_tasks(spec: DatasetSpec) -> list[ForecastTask]:
    dataset = load_hf_dataset(spec)
    sequence_columns = infer_sequence_columns(dataset)
    if not sequence_columns:
        raise ValueError(f"No sequence target columns found in {spec.name}")

    minimum_length = required_series_length(spec)
    row_indices = matching_row_indices(dataset, spec)
    candidates = [
        (row_index, field)
        for row_index in row_indices
        for field in sequence_columns
        if len(dataset[row_index][field]) >= minimum_length
        and supports_requested_origins(dataset[row_index][field], spec)
    ]
    selected = select_representative_items(
        candidates,
        spec.max_series,
        spec.selection_strategy,
        spec.selection_seed,
    )
    tasks: list[ForecastTask] = []
    for row_index, field in selected:
        row = dataset[row_index]
        group = f"{spec.filter_column}={row[spec.filter_column]}|" if spec.filter_column else ""
        tasks.extend(make_tasks(spec, f"{group}{row_index}:{field}", row[field], row.get("timestamp")))
    return tasks


def external_timestamps(frame: pd.DataFrame, spec: DatasetSpec) -> tuple[str | None, pd.DatetimeIndex | None]:
    date_column = spec.date_column
    if date_column is None:
        date_column = next(
            (candidate for candidate in ("date", "timestamp", "time", "datetime") if candidate in frame.columns),
            None,
        )
    if date_column is None or date_column not in frame.columns:
        return date_column, None
    values = frame[date_column]
    if spec.timestamp_unit is None:
        return date_column, pd.DatetimeIndex(pd.to_datetime(values))
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.isna().any():
        raise ValueError(f"Timestamp column '{date_column}' contains non-numeric values in {spec.name}")
    offsets = numeric - numeric.iloc[0]
    timestamps = pd.Timestamp("2000-01-01") + pd.to_timedelta(offsets, unit=spec.timestamp_unit)
    return date_column, pd.DatetimeIndex(timestamps)


def prepare_external_targets(
    frame: pd.DataFrame,
    spec: DatasetSpec,
) -> tuple[pd.DataFrame, pd.DatetimeIndex | None]:
    date_column, timestamps = external_timestamps(frame, spec)
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

    targets = frame[numeric_columns].copy()
    if spec.resample_frequency is not None:
        if timestamps is None:
            raise ValueError(f"Dataset {spec.name} requires timestamps before resampling")
        targets.index = timestamps
        targets = targets.sort_index()
        resampler = targets.resample(spec.resample_frequency)
        targets = getattr(resampler, spec.resample_method)().dropna(how="all")
        timestamps = pd.DatetimeIndex(targets.index)
    else:
        targets = targets.reset_index(drop=True)
    return targets, timestamps


def load_external_tasks(spec: DatasetSpec) -> list[ForecastTask]:
    dataset = load_hf_dataset(spec)
    frame = dataset.to_pandas()
    targets, timestamps = prepare_external_targets(frame, spec)
    selected_columns = select_representative_items(
        list(targets.columns),
        spec.max_series,
        spec.selection_strategy,
        spec.selection_seed,
    )
    tasks: list[ForecastTask] = []
    for column in selected_columns:
        tasks.extend(make_tasks(spec, str(column), targets[column], timestamps))
    return tasks


def load_dataset_tasks(spec: DatasetSpec) -> list[ForecastTask]:
    if spec.family == "chronos":
        return load_chronos_tasks(spec)
    if spec.family == "external":
        return load_external_tasks(spec)
    if spec.family == "official":
        from pmds.official_datasets import load_official_tasks

        return load_official_tasks(spec)
    raise ValueError(f"Unsupported dataset family: {spec.family}")
