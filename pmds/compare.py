"""
PMDS comparison benchmark.

Runs:
- smallest original Chronos model by default: amazon/chronos-t5-tiny
- 3 datasets from the Chronos Hugging Face dataset repository
- 3 datasets from outside the Chronos repository
- classical/statistical baselines plus optional DeepAR

The script is intentionally registry-like:
- add datasets by extending DATASETS
- add models by extending build_model_runners()
- all models return one point forecast per ForecastTask
"""

from __future__ import annotations

import argparse
import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np
import pandas as pd
import torch

from chronos import BaseChronosPipeline


DEFAULT_QUANTILES = [0.1, 0.5, 0.9]


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    family: str
    repo: str
    configs: tuple[str, ...]
    prediction_length: int
    seasonality: int
    max_series: int
    target_column: str | None = None
    date_column: str | None = None
    trust_remote_code: bool = False


@dataclass(frozen=True)
class ForecastTask:
    dataset: str
    item_id: str
    context_timestamps: pd.DatetimeIndex
    future_timestamps: pd.DatetimeIndex
    context: np.ndarray
    future: np.ndarray
    prediction_length: int
    seasonality: int
    frequency: str


@dataclass(frozen=True)
class ForecastResult:
    dataset: str
    item_id: str
    model: str
    predictions: np.ndarray
    error: str = ""


DATASETS: dict[str, DatasetSpec] = {
    # Chronos repository datasets.
    "chronos_m1_yearly": DatasetSpec(
        name="chronos_m1_yearly",
        family="chronos",
        repo="autogluon/chronos_datasets",
        configs=("monash_m1_yearly",),
        prediction_length=6,
        seasonality=1,
        max_series=20,
    ),
    "chronos_m4_hourly": DatasetSpec(
        name="chronos_m4_hourly",
        family="chronos",
        repo="autogluon/chronos_datasets",
        configs=("m4_hourly",),
        prediction_length=48,
        seasonality=24,
        max_series=20,
    ),
    "chronos_weather": DatasetSpec(
        name="chronos_weather",
        family="chronos",
        repo="autogluon/chronos_datasets",
        configs=("monash_weather",),
        prediction_length=30,
        seasonality=7,
        max_series=20,
    ),
    # External datasets. These are loaded from THUML's Time-Series-Library HF dataset,
    # not from autogluon/chronos_datasets.
    "external_national_illness": DatasetSpec(
        name="external_national_illness",
        family="external",
        repo="thuml/Time-Series-Library",
        configs=("national_illness",),
        prediction_length=24,
        seasonality=52,
        max_series=1,
        target_column="OT",
        date_column="date",
    ),
    "external_traffic": DatasetSpec(
        name="external_traffic",
        family="external",
        repo="thuml/Time-Series-Library",
        configs=("traffic",),
        prediction_length=24,
        seasonality=24,
        max_series=5,
    ),
    "external_psm": DatasetSpec(
        name="external_psm",
        family="external",
        repo="thuml/Time-Series-Library",
        configs=("PSM-data",),
        prediction_length=24,
        seasonality=24,
        max_series=5,
    ),
}


DEFAULT_DATASETS = [
    "chronos_m1_yearly",
    "chronos_m4_hourly",
    "chronos_weather",
    "external_national_illness",
    "external_traffic",
    "external_psm",
]


def import_datasets():
    try:
        import datasets
    except ImportError as exc:
        raise ImportError("Install datasets with: pip install datasets") from exc
    return datasets


def clean_numeric(values: Iterable) -> np.ndarray:
    series = pd.Series(np.asarray(values, dtype=np.float64))
    return series.interpolate(limit_direction="both").fillna(0.0).to_numpy(dtype=np.float32)


def infer_frequency(timestamps: pd.DatetimeIndex, fallback: str) -> str:
    if len(timestamps) >= 3:
        freq = pd.infer_freq(timestamps)
        if freq is not None:
            return freq
    return fallback


def align_timestamps(values: np.ndarray, timestamps: Iterable | None, fallback_freq: str) -> pd.DatetimeIndex:
    if timestamps is None:
        return pd.date_range("2000-01-01", periods=len(values), freq=fallback_freq)

    index = pd.DatetimeIndex(pd.to_datetime(timestamps))
    if len(index) == len(values):
        return index

    return pd.date_range("2000-01-01", periods=len(values), freq=fallback_freq)


def load_hf_dataset(spec: DatasetSpec):
    datasets = import_datasets()
    last_error: Exception | None = None
    for config in spec.configs:
        try:
            return datasets.load_dataset(
                spec.repo,
                config,
                split="train",
                trust_remote_code=spec.trust_remote_code,
            )
        except Exception as exc:
            last_error = exc

    try:
        available = datasets.get_dataset_config_names(spec.repo)
    except Exception:
        available = []
    raise RuntimeError(
        f"Could not load {spec.name} from {spec.repo} with configs={spec.configs}. "
        f"Available configs include: {available[:20]}"
    ) from last_error


def infer_sequence_columns(hf_dataset) -> list[str]:
    datasets = import_datasets()
    return [
        col
        for col, feature in hf_dataset.features.items()
        if isinstance(feature, datasets.Sequence) and col != "timestamp"
    ]


def make_task(
    spec: DatasetSpec,
    item_id: str,
    values: Iterable,
    timestamps: Iterable | None,
    fallback_freq: str,
) -> ForecastTask | None:
    series = clean_numeric(values)
    if len(series) <= spec.prediction_length + 1:
        return None

    timestamp_index = align_timestamps(series, timestamps, fallback_freq=fallback_freq)
    freq = infer_frequency(timestamp_index, fallback=fallback_freq)

    return ForecastTask(
        dataset=spec.name,
        item_id=item_id,
        context_timestamps=timestamp_index[: -spec.prediction_length],
        future_timestamps=timestamp_index[-spec.prediction_length :],
        context=series[: -spec.prediction_length],
        future=series[-spec.prediction_length :],
        prediction_length=spec.prediction_length,
        seasonality=spec.seasonality,
        frequency=freq,
    )


def load_chronos_tasks(spec: DatasetSpec) -> list[ForecastTask]:
    ds = load_hf_dataset(spec)
    sequence_cols = infer_sequence_columns(ds)
    if not sequence_cols:
        raise ValueError(f"No sequence target columns found in {spec.name}.")

    tasks: list[ForecastTask] = []
    for row_idx, row in enumerate(ds):
        timestamps = row["timestamp"] if "timestamp" in row else None
        for field in sequence_cols:
            task = make_task(
                spec=spec,
                item_id=f"{row_idx}:{field}",
                values=row[field],
                timestamps=timestamps,
                fallback_freq="D",
            )
            if task is not None:
                tasks.append(task)
            if len(tasks) >= spec.max_series:
                return tasks
    return tasks


def numeric_columns(df: pd.DataFrame, date_column: str | None) -> list[str]:
    exclude = {date_column} if date_column is not None else set()
    return [col for col in df.select_dtypes(include=[np.number]).columns if col not in exclude]


def load_external_tasks(spec: DatasetSpec) -> list[ForecastTask]:
    ds = load_hf_dataset(spec)
    df = ds.to_pandas()

    date_column = spec.date_column
    if date_column is None:
        for candidate in ("date", "timestamp", "time", "datetime"):
            if candidate in df.columns:
                date_column = candidate
                break

    timestamps = df[date_column] if date_column in df.columns else None
    columns = numeric_columns(df, date_column=date_column)
    if spec.target_column is not None and spec.target_column in columns:
        columns = [spec.target_column]

    if not columns:
        raise ValueError(f"No numeric target columns found in external dataset {spec.name}.")

    tasks: list[ForecastTask] = []
    for col in columns:
        task = make_task(
            spec=spec,
            item_id=col,
            values=df[col],
            timestamps=timestamps,
            fallback_freq="h",
        )
        if task is not None:
            tasks.append(task)
        if len(tasks) >= spec.max_series:
            break
    return tasks


def load_tasks(dataset_names: Sequence[str]) -> list[ForecastTask]:
    tasks: list[ForecastTask] = []
    for name in dataset_names:
        spec = DATASETS[name]
        try:
            if spec.family == "chronos":
                loaded = load_chronos_tasks(spec)
            else:
                loaded = load_external_tasks(spec)
            print(f"Loaded {len(loaded)} tasks from {name}")
            tasks.extend(loaded)
        except Exception as exc:
            print(f"WARNING: failed to load {name}: {exc}")
    return tasks


def seasonal_naive(task: ForecastTask) -> np.ndarray:
    context = clean_numeric(task.context)
    if len(context) == 0:
        return np.zeros(task.prediction_length, dtype=np.float32)
    if task.seasonality <= 1 or len(context) < task.seasonality:
        return np.repeat(context[-1], task.prediction_length).astype(np.float32)
    pattern = context[-task.seasonality :]
    return np.tile(pattern, math.ceil(task.prediction_length / len(pattern)))[: task.prediction_length].astype(
        np.float32
    )


def arma(task: ForecastTask) -> np.ndarray:
    try:
        from statsmodels.tsa.arima.model import ARIMA
    except ImportError as exc:
        raise ImportError("Install statsmodels with: pip install statsmodels") from exc

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fitted = ARIMA(clean_numeric(task.context).astype(np.float64), order=(2, 0, 2)).fit()
        forecast = fitted.forecast(steps=task.prediction_length)
    return np.asarray(forecast, dtype=np.float32)


def prophet(task: ForecastTask) -> np.ndarray:
    try:
        from prophet import Prophet
    except ImportError as exc:
        raise ImportError("Install Prophet with: pip install prophet") from exc

    df = pd.DataFrame({"ds": task.context_timestamps, "y": clean_numeric(task.context)})
    model = Prophet()
    model.fit(df)
    future = pd.DataFrame({"ds": task.future_timestamps})
    forecast = model.predict(future)
    return forecast["yhat"].to_numpy(dtype=np.float32)


def deepar(task: ForecastTask, max_epochs: int) -> np.ndarray:
    try:
        from gluonts.dataset.common import ListDataset
        from gluonts.torch.model.deepar import DeepAREstimator
    except ImportError as exc:
        raise ImportError("Install GluonTS torch support with: pip install 'gluonts[torch]' lightning") from exc

    start = pd.Period(task.context_timestamps[0], freq=task.frequency)
    train_ds = ListDataset(
        [{"start": start, "target": clean_numeric(task.context)}],
        freq=task.frequency,
    )
    estimator = DeepAREstimator(
        freq=task.frequency,
        prediction_length=task.prediction_length,
        context_length=min(len(task.context), max(task.prediction_length * 2, 8)),
        batch_size=16,
        num_batches_per_epoch=10,
        trainer_kwargs={
            "max_epochs": max_epochs,
            "enable_checkpointing": False,
            "logger": False,
            "enable_progress_bar": False,
        },
    )
    predictor = estimator.train(train_ds)
    forecast = next(iter(predictor.predict(train_ds)))
    return np.asarray(forecast.mean, dtype=np.float32)


class ChronosRunner:
    def __init__(self, model_id: str, device: str, torch_dtype: str, num_samples: int):
        dtype = getattr(torch, torch_dtype) if torch_dtype != "auto" else "auto"
        kwargs = {"device_map": device}
        if dtype != "auto":
            kwargs["torch_dtype"] = dtype

        self.model_id = model_id
        self.num_samples = num_samples
        self.pipeline = BaseChronosPipeline.from_pretrained(model_id, **kwargs)

    def __call__(self, task: ForecastTask) -> np.ndarray:
        context = torch.tensor(task.context, dtype=torch.float32)
        predict_kwargs = {}
        if self.pipeline.forecast_type.value == "samples":
            predict_kwargs["num_samples"] = self.num_samples

        _, mean = self.pipeline.predict_quantiles(
            [context],
            prediction_length=task.prediction_length,
            quantile_levels=DEFAULT_QUANTILES,
            **predict_kwargs,
        )
        if isinstance(mean, list):
            return mean[0].detach().cpu().numpy().reshape(-1).astype(np.float32)
        return mean[0].detach().cpu().numpy().reshape(-1).astype(np.float32)


def build_model_runners(args: argparse.Namespace) -> dict[str, Callable[[ForecastTask], np.ndarray]]:
    runners: dict[str, Callable[[ForecastTask], np.ndarray]] = {}
    for model_name in args.models:
        if model_name == "chronos":
            chronos_runner = ChronosRunner(args.chronos_model, args.device, args.torch_dtype, args.num_samples)
            runners[f"chronos:{args.chronos_model}"] = chronos_runner
        elif model_name == "seasonal_naive":
            runners["seasonal_naive"] = seasonal_naive
        elif model_name == "arma":
            runners["arma"] = arma
        elif model_name == "prophet":
            runners["prophet"] = prophet
        elif model_name == "deepar":
            runners["deepar"] = lambda task: deepar(task, max_epochs=args.deepar_epochs)
        else:
            raise ValueError(f"Unknown model: {model_name}")
    return runners


def safe_predict(model_name: str, task: ForecastTask, runner: Callable[[ForecastTask], np.ndarray]) -> ForecastResult:
    try:
        pred = np.asarray(runner(task), dtype=np.float32).reshape(-1)
        if len(pred) != task.prediction_length:
            raise ValueError(f"Expected {task.prediction_length} predictions, got {len(pred)}.")
        return ForecastResult(task.dataset, task.item_id, model_name, pred)
    except Exception as exc:
        return ForecastResult(task.dataset, task.item_id, model_name, np.array([], dtype=np.float32), str(exc))


def mase(y_true: np.ndarray, y_pred: np.ndarray, insample: np.ndarray, seasonality: int) -> float:
    insample = clean_numeric(insample)
    seasonality = seasonality if len(insample) > seasonality else 1
    diffs = np.abs(insample[seasonality:] - insample[:-seasonality])
    scale = float(np.mean(diffs)) if len(diffs) else float("nan")
    if not np.isfinite(scale) or scale == 0.0:
        return float("nan")
    return float(np.mean(np.abs(y_true - y_pred)) / scale)


def compute_metrics(task: ForecastTask, result: ForecastResult) -> dict[str, float | str]:
    if result.error:
        return {
            "dataset": task.dataset,
            "item_id": task.item_id,
            "model": result.model,
            "mae": float("nan"),
            "rmse": float("nan"),
            "smape": float("nan"),
            "mase": float("nan"),
            "error": result.error,
        }

    y_true = task.future.astype(np.float32)
    y_pred = result.predictions.astype(np.float32)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if not mask.any():
        error = "No finite target/prediction pairs available for metrics."
        return {
            "dataset": task.dataset,
            "item_id": task.item_id,
            "model": result.model,
            "mae": float("nan"),
            "rmse": float("nan"),
            "smape": float("nan"),
            "mase": float("nan"),
            "error": error,
        }

    y_true = y_true[mask]
    y_pred = y_pred[mask]
    denom = np.maximum(np.abs(y_true) + np.abs(y_pred), 1e-8)
    return {
        "dataset": task.dataset,
        "item_id": task.item_id,
        "model": result.model,
        "mae": float(np.mean(np.abs(y_true - y_pred))),
        "rmse": float(np.sqrt(np.mean(np.square(y_true - y_pred)))),
        "smape": float(np.mean(2.0 * np.abs(y_true - y_pred) / denom)),
        "mase": mase(y_true, y_pred, task.context, task.seasonality),
        "error": "",
    }


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    metric_cols = ["mae", "rmse", "smape", "mase"]
    return (
        results.groupby(["dataset", "model"], dropna=False)[metric_cols]
        .mean(numeric_only=True)
        .reset_index()
        .sort_values(["dataset", "mae"], na_position="last")
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare Chronos tiny against classical, Prophet, and DeepAR baselines.")
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS, choices=sorted(DATASETS))
    parser.add_argument(
        "--models",
        nargs="+",
        default=["chronos", "seasonal_naive", "arma", "prophet", "deepar"],
        choices=["chronos", "seasonal_naive", "arma", "prophet", "deepar"],
    )
    parser.add_argument("--chronos-model", default="amazon/chronos-t5-tiny")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--torch-dtype", default="float32", choices=["auto", "float32", "bfloat16"])
    parser.add_argument("--num-samples", type=int, default=20)
    parser.add_argument("--deepar-epochs", type=int, default=5)
    parser.add_argument("--output-dir", type=Path, default=Path("pmds/results"))
    parser.add_argument("--run-name", default="compare")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    runners = build_model_runners(args)

    tasks = load_tasks(args.datasets)
    if not tasks:
        raise RuntimeError("No tasks loaded. Check dataset configs or network access.")

    rows = []
    for task in tasks:
        for model_name, runner in runners.items():
            print(f"Running {model_name} on {task.dataset}/{task.item_id}")
            result = safe_predict(model_name, task, runner)
            rows.append(compute_metrics(task, result))

    detailed = pd.DataFrame(rows)
    summary = summarize(detailed)

    detailed_path = args.output_dir / f"{args.run_name}_detailed.csv"
    summary_path = args.output_dir / f"{args.run_name}_summary.csv"
    detailed.to_csv(detailed_path, index=False)
    summary.to_csv(summary_path, index=False)

    print("\nSummary")
    print(summary.to_string(index=False))
    print(f"\nSaved detailed results to: {detailed_path}")
    print(f"Saved summary results to: {summary_path}")


if __name__ == "__main__":
    main()
