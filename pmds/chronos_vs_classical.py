"""
PMDS baseline experiment: Chronos vs classical time-series models.

Default flow:
1. Run the smallest original Chronos model, amazon/chronos-t5-tiny.
2. Compare it against classical baselines on one Chronos dataset.
3. Optionally repeat the same comparison on external national_illness data.

This file is intentionally organized around small interfaces:
- dataset loaders return ForecastTask objects
- model runners consume ForecastTask and return point forecasts
- metrics are computed in one place

That should make it easy to add decoder-only or MoE models later without
rewriting the benchmark loop.
"""

from __future__ import annotations

import argparse
import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd
import torch

from chronos import BaseChronosPipeline


DEFAULT_QUANTILES = [0.1, 0.5, 0.9]


@dataclass(frozen=True)
class ForecastTask:
    dataset: str
    item_id: str
    timestamps: pd.DatetimeIndex
    context: np.ndarray
    future: np.ndarray
    prediction_length: int
    seasonality: int = 1


@dataclass(frozen=True)
class ForecastResult:
    dataset: str
    item_id: str
    model: str
    predictions: np.ndarray
    error: str | None = None


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    source: str
    repo: str
    config: str
    prediction_length: int
    max_series: int
    seasonality: int
    target_column: str | None = None


INTERNAL_CHRONOS_DATASET = DatasetSpec(
    name="chronos_monash_m1_yearly",
    source="chronos",
    repo="autogluon/chronos_datasets",
    config="monash_m1_yearly",
    prediction_length=6,
    max_series=20,
    seasonality=1,
)

EXTERNAL_NATIONAL_ILLNESS = DatasetSpec(
    name="external_national_illness",
    source="tslib",
    repo="thuml/Time-Series-Library",
    config="national_illness",
    prediction_length=24,
    max_series=1,
    seasonality=52,
    target_column="OT",
)


def import_datasets():
    try:
        import datasets
    except ImportError as exc:
        raise ImportError(
            "This experiment needs the Hugging Face datasets package. "
            "Install it with: pip install datasets"
        ) from exc
    return datasets


def numeric_array(values: Iterable) -> np.ndarray:
    return np.asarray(values, dtype=np.float32)


def clean_numeric(values: np.ndarray) -> np.ndarray:
    series = pd.Series(values.astype(np.float64))
    return series.interpolate(limit_direction="both").fillna(0.0).to_numpy(dtype=np.float32)


def infer_sequence_columns(hf_dataset) -> list[str]:
    datasets = import_datasets()
    sequence_cols = [
        col
        for col, feature in hf_dataset.features.items()
        if isinstance(feature, datasets.Sequence) and col != "timestamp"
    ]
    if not sequence_cols:
        raise ValueError("Could not find sequence-valued target columns in the Chronos dataset.")
    return sequence_cols


def load_chronos_dataset(spec: DatasetSpec) -> list[ForecastTask]:
    datasets = import_datasets()
    ds = datasets.load_dataset(spec.repo, spec.config, split="train")
    sequence_cols = infer_sequence_columns(ds)

    tasks: list[ForecastTask] = []
    for row_idx, row in enumerate(ds):
        timestamps = pd.DatetimeIndex(pd.to_datetime(row["timestamp"]))
        for field in sequence_cols:
            values = numeric_array(row[field])
            if len(values) <= spec.prediction_length + 1:
                continue
            tasks.append(
                ForecastTask(
                    dataset=spec.name,
                    item_id=f"{row_idx}:{field}",
                    timestamps=timestamps[-spec.prediction_length :],
                    context=values[: -spec.prediction_length],
                    future=values[-spec.prediction_length :],
                    prediction_length=spec.prediction_length,
                    seasonality=spec.seasonality,
                )
            )
            if len(tasks) >= spec.max_series:
                return tasks
    return tasks


def load_tslib_national_illness(spec: DatasetSpec) -> list[ForecastTask]:
    datasets = import_datasets()
    ds = datasets.load_dataset(spec.repo, spec.config, split="train")
    df = ds.to_pandas()

    timestamp_col = "date" if "date" in df.columns else df.columns[0]
    target_col = spec.target_column if spec.target_column in df.columns else None
    if target_col is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) == 0:
            raise ValueError("Could not find a numeric target column for national_illness.")
        target_col = numeric_cols[-1]

    values = numeric_array(df[target_col])
    timestamps = pd.DatetimeIndex(pd.to_datetime(df[timestamp_col]))
    if len(values) <= spec.prediction_length + 1:
        raise ValueError(f"{spec.name} is too short for prediction_length={spec.prediction_length}.")

    return [
        ForecastTask(
            dataset=spec.name,
            item_id=target_col,
            timestamps=timestamps[-spec.prediction_length :],
            context=values[: -spec.prediction_length],
            future=values[-spec.prediction_length :],
            prediction_length=spec.prediction_length,
            seasonality=spec.seasonality,
        )
    ]


def load_tasks(dataset: str) -> list[ForecastTask]:
    if dataset == "chronos":
        return load_chronos_dataset(INTERNAL_CHRONOS_DATASET)
    if dataset == "national_illness":
        return load_tslib_national_illness(EXTERNAL_NATIONAL_ILLNESS)
    if dataset == "all":
        return load_chronos_dataset(INTERNAL_CHRONOS_DATASET) + load_tslib_national_illness(EXTERNAL_NATIONAL_ILLNESS)
    raise ValueError(f"Unknown dataset choice: {dataset}")


def seasonal_naive_forecast(context: np.ndarray, prediction_length: int, seasonality: int) -> np.ndarray:
    context = clean_numeric(context)
    if len(context) == 0:
        return np.zeros(prediction_length, dtype=np.float32)
    if seasonality <= 1 or len(context) < seasonality:
        return np.repeat(context[-1], prediction_length).astype(np.float32)
    pattern = context[-seasonality:]
    repeats = math.ceil(prediction_length / len(pattern))
    return np.tile(pattern, repeats)[:prediction_length].astype(np.float32)


def arma_forecast(context: np.ndarray, prediction_length: int, seasonality: int) -> np.ndarray:
    del seasonality
    context = clean_numeric(context)
    try:
        from statsmodels.tsa.arima.model import ARIMA
    except ImportError as exc:
        raise ImportError("ARMA baseline requires statsmodels. Install with: pip install statsmodels") from exc

    order = (2, 0, 2)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = ARIMA(context.astype(np.float64), order=order)
        fitted = model.fit()
        forecast = fitted.forecast(steps=prediction_length)
    return np.asarray(forecast, dtype=np.float32)


def prophet_forecast(task: ForecastTask) -> np.ndarray:
    try:
        from prophet import Prophet
    except ImportError as exc:
        raise ImportError("Prophet baseline requires prophet. Install with: pip install prophet") from exc

    freq = pd.infer_freq(task.timestamps)
    if freq is None:
        freq = "W" if task.seasonality >= 52 else "D"

    offset = pd.tseries.frequencies.to_offset(freq)
    history_end = task.timestamps[0] - offset
    history_dates = pd.date_range(end=history_end, periods=len(task.context), freq=freq)
    df = pd.DataFrame({"ds": history_dates, "y": clean_numeric(task.context)})
    model = Prophet()
    model.fit(df)
    future = model.make_future_dataframe(periods=task.prediction_length, freq=freq, include_history=False)
    forecast = model.predict(future)
    return forecast["yhat"].to_numpy(dtype=np.float32)


class ChronosRunner:
    def __init__(self, model_id: str, device: str, torch_dtype: str, num_samples: int):
        dtype = getattr(torch, torch_dtype) if torch_dtype != "auto" else "auto"
        self.model_id = model_id
        self.num_samples = num_samples
        kwargs = {"device_map": device}
        if dtype != "auto":
            kwargs["torch_dtype"] = dtype
        self.pipeline = BaseChronosPipeline.from_pretrained(model_id, **kwargs)

    def predict(self, task: ForecastTask) -> np.ndarray:
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


def run_classical_model(name: str, task: ForecastTask) -> np.ndarray:
    if name == "seasonal_naive":
        return seasonal_naive_forecast(task.context, task.prediction_length, task.seasonality)
    if name == "arma":
        return arma_forecast(task.context, task.prediction_length, task.seasonality)
    if name == "prophet":
        return prophet_forecast(task)
    raise ValueError(f"Unknown classical model: {name}")


def mase(y_true: np.ndarray, y_pred: np.ndarray, insample: np.ndarray, seasonality: int) -> float:
    insample = clean_numeric(insample)
    if len(insample) <= seasonality:
        seasonality = 1
    diffs = np.abs(insample[seasonality:] - insample[:-seasonality])
    denom = float(np.mean(diffs)) if len(diffs) else float("nan")
    if not np.isfinite(denom) or denom == 0.0:
        return float("nan")
    return float(np.mean(np.abs(y_true - y_pred)) / denom)


def compute_metrics(task: ForecastTask, result: ForecastResult) -> dict[str, float | str]:
    if result.error is not None:
        return {
            "dataset": result.dataset,
            "item_id": result.item_id,
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
        return {
            "dataset": result.dataset,
            "item_id": result.item_id,
            "model": result.model,
            "mae": float("nan"),
            "rmse": float("nan"),
            "smape": float("nan"),
            "mase": float("nan"),
            "error": "No finite target/prediction pairs available for metrics.",
        }
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    denom = np.maximum(np.abs(y_true) + np.abs(y_pred), 1e-8)
    return {
        "dataset": result.dataset,
        "item_id": result.item_id,
        "model": result.model,
        "mae": float(np.mean(np.abs(y_true - y_pred))),
        "rmse": float(np.sqrt(np.mean(np.square(y_true - y_pred)))),
        "smape": float(np.mean(2.0 * np.abs(y_true - y_pred) / denom)),
        "mase": mase(y_true, y_pred, task.context, task.seasonality),
        "error": "",
    }


def safe_predict(model_name: str, task: ForecastTask, fn: Callable[[ForecastTask], np.ndarray]) -> ForecastResult:
    try:
        predictions = fn(task)
        if len(predictions) != task.prediction_length:
            raise ValueError(
                f"Expected {task.prediction_length} predictions, got {len(predictions)}."
            )
        return ForecastResult(task.dataset, task.item_id, model_name, predictions)
    except Exception as exc:
        return ForecastResult(task.dataset, task.item_id, model_name, np.array([]), error=str(exc))


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    metric_cols = ["mae", "rmse", "smape", "mase"]
    return (
        results.groupby(["dataset", "model"], dropna=False)[metric_cols]
        .mean(numeric_only=True)
        .reset_index()
        .sort_values(["dataset", "mae"])
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare Chronos against classical time-series baselines.")
    parser.add_argument("--dataset", choices=["chronos", "national_illness", "all"], default="chronos")
    parser.add_argument("--chronos-model", default="amazon/chronos-t5-tiny")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--torch-dtype", default="float32", choices=["auto", "float32", "bfloat16"])
    parser.add_argument("--num-samples", type=int, default=20)
    parser.add_argument(
        "--classical-models",
        nargs="+",
        default=["seasonal_naive", "arma"],
        choices=["seasonal_naive", "arma", "prophet"],
    )
    parser.add_argument("--output-dir", type=Path, default=Path("pmds/results"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    tasks = load_tasks(args.dataset)
    if not tasks:
        raise RuntimeError("No forecast tasks were loaded. Try a smaller horizon or a different dataset.")

    chronos = ChronosRunner(
        model_id=args.chronos_model,
        device=args.device,
        torch_dtype=args.torch_dtype,
        num_samples=args.num_samples,
    )

    rows = []
    for task in tasks:
        model_runs: list[tuple[str, Callable[[ForecastTask], np.ndarray]]] = [
            (f"chronos:{args.chronos_model}", chronos.predict)
        ]
        model_runs.extend(
            (model_name, lambda t, name=model_name: run_classical_model(name, t))
            for model_name in args.classical_models
        )

        for model_name, predict_fn in model_runs:
            result = safe_predict(model_name, task, predict_fn)
            rows.append(compute_metrics(task, result))

    detailed = pd.DataFrame(rows)
    summary = summarize(detailed)

    detailed_path = args.output_dir / f"{args.dataset}_detailed.csv"
    summary_path = args.output_dir / f"{args.dataset}_summary.csv"
    detailed.to_csv(detailed_path, index=False)
    summary.to_csv(summary_path, index=False)

    print("\nSummary")
    print(summary.to_string(index=False))
    print(f"\nSaved detailed results to: {detailed_path}")
    print(f"Saved summary results to: {summary_path}")


if __name__ == "__main__":
    main()
