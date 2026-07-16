"""Config-driven PMDS benchmark for Chronos and baseline forecasting models."""

from __future__ import annotations

import argparse
import json
import logging
import logging.handlers
import math
import os
import random
import sys
import tempfile
import time
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import torch

from chronos import BaseChronosPipeline


LOGGER = logging.getLogger("pmds.compare")
RESULT_COLUMNS = [
    "dataset",
    "item_id",
    "model",
    "mae",
    "rmse",
    "smape",
    "mase",
    "wql",
    "wql_loss_sum",
    "wql_abs_target_sum",
    "duration_seconds",
    "error_type",
    "error",
]


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    family: str
    repo: str
    hf_configs: tuple[str, ...]
    split: str
    prediction_length: int
    seasonality: int
    max_series: int
    fallback_frequency: str
    target_column: str | None = None
    date_column: str | None = None
    exclude_columns: tuple[str, ...] = ()
    trust_remote_code: bool = False

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "DatasetSpec":
        required = {
            "name",
            "family",
            "repo",
            "hf_configs",
            "split",
            "prediction_length",
            "seasonality",
            "max_series",
            "fallback_frequency",
        }
        missing = required - set(config)
        if missing:
            raise ValueError(f"Dataset config is missing fields: {sorted(missing)}")
        return cls(
            name=str(config["name"]),
            family=str(config["family"]),
            repo=str(config["repo"]),
            hf_configs=tuple(map(str, config["hf_configs"])),
            split=str(config["split"]),
            prediction_length=int(config["prediction_length"]),
            seasonality=int(config["seasonality"]),
            max_series=int(config["max_series"]),
            fallback_frequency=str(config["fallback_frequency"]),
            target_column=config.get("target_column"),
            date_column=config.get("date_column"),
            exclude_columns=tuple(map(str, config.get("exclude_columns", []))),
            trust_remote_code=bool(config.get("trust_remote_code", False)),
        )


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
class ForecastOutput:
    mean: np.ndarray
    quantiles: np.ndarray  # shape: (prediction_length, num_quantiles)


@dataclass(frozen=True)
class ForecastResult:
    dataset: str
    item_id: str
    model: str
    output: ForecastOutput | None
    duration_seconds: float
    error_type: str = ""
    error: str = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the config-driven PMDS forecast comparison.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("pmds/config.json"),
        help="Path to the JSON experiment configuration.",
    )
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Configuration file does not exist: {path}")
    with path.open(encoding="utf-8") as fp:
        config = json.load(fp)
    validate_config(config)
    return config


def validate_config(config: Mapping[str, Any]) -> None:
    required = {"run", "evaluation", "datasets", "models"}
    missing = required - set(config)
    if missing:
        raise ValueError(f"Top-level config is missing fields: {sorted(missing)}")

    run_required = {"name", "output_dir", "log_dir", "random_seed", "logging"}
    missing_run = run_required - set(config["run"])
    if missing_run:
        raise ValueError(f"run config is missing fields: {sorted(missing_run)}")

    evaluation = config["evaluation"]
    metrics = list(evaluation.get("metrics", []))
    supported_metrics = {"mae", "rmse", "smape", "mase", "wql"}
    unknown_metrics = set(metrics) - supported_metrics
    if unknown_metrics:
        raise ValueError(f"Unsupported metrics: {sorted(unknown_metrics)}")

    quantiles = np.asarray(evaluation.get("quantiles", []), dtype=float)
    if quantiles.ndim != 1 or len(quantiles) == 0:
        raise ValueError("evaluation.quantiles must be a non-empty list")
    if np.any(quantiles <= 0.0) or np.any(quantiles >= 1.0) or np.any(np.diff(quantiles) <= 0.0):
        raise ValueError("evaluation.quantiles must be strictly increasing and between 0 and 1")
    if evaluation.get("point_forecast") == "median" and not np.any(np.isclose(quantiles, 0.5)):
        raise ValueError("evaluation.quantiles must contain 0.5 when point_forecast='median'")
    if evaluation.get("point_forecast") not in {"mean", "median"}:
        raise ValueError("evaluation.point_forecast must be 'mean' or 'median'")

    dataset_names = [dataset.get("name") for dataset in config["datasets"] if dataset.get("enabled", True)]
    if len(dataset_names) != len(set(dataset_names)):
        raise ValueError("Enabled dataset names must be unique")
    model_names = [model.get("name") for model in config["models"] if model.get("enabled", True)]
    if len(model_names) != len(set(model_names)):
        raise ValueError("Enabled model names must be unique")


def resolve_path(path_value: str) -> Path:
    return Path(path_value).expanduser()


def setup_logging(run_config: Mapping[str, Any]) -> Path:
    log_config = run_config["logging"]
    log_dir = resolve_path(run_config["log_dir"])
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"{run_config['name']}_{timestamp}.log"

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.setLevel(logging.DEBUG)

    console = logging.StreamHandler(sys.stdout)
    console.setLevel(getattr(logging, str(log_config["console_level"]).upper()))
    console.setFormatter(formatter)
    root_logger.addHandler(console)

    file_handler = logging.handlers.RotatingFileHandler(
        log_path,
        maxBytes=int(log_config["max_bytes"]),
        backupCount=int(log_config["backup_count"]),
        encoding="utf-8",
    )
    file_handler.setLevel(getattr(logging, str(log_config["file_level"]).upper()))
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    logging.captureWarnings(True)
    return log_path


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def apply_environment(environment: Mapping[str, Any]) -> None:
    for name, value in environment.items():
        os.environ[str(name)] = str(value)
        if str(name).endswith("DIR"):
            Path(str(value)).expanduser().mkdir(parents=True, exist_ok=True)


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
        frequency = pd.infer_freq(timestamps)
        if frequency is not None:
            return frequency
    return fallback


def period_compatible_frequency(frequency: str) -> str:
    replacements = {
        "ME": "M",
        "QE-": "Q-",
        "YE-": "Y-",
        "YS-": "Y-",
    }
    for source, target in replacements.items():
        if frequency == source or frequency.startswith(source):
            return frequency.replace(source, target, 1)
    return frequency


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
    raise RuntimeError(
        f"Could not load {spec.name} from {spec.repo} with configs={spec.hf_configs}"
    ) from last_error


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


def repeated_quantiles(point: np.ndarray, num_quantiles: int) -> np.ndarray:
    return np.repeat(point[:, None], num_quantiles, axis=1).astype(np.float32)


def normal_quantiles(mean: np.ndarray, std: np.ndarray, quantile_levels: np.ndarray) -> np.ndarray:
    from scipy.stats import norm

    safe_std = np.maximum(np.asarray(std, dtype=np.float64), 1e-8)
    values = mean[:, None] + safe_std[:, None] * norm.ppf(quantile_levels)[None, :]
    return values.astype(np.float32)


class ChronosRunner:
    def __init__(self, params: Mapping[str, Any], quantiles: np.ndarray):
        self.params = dict(params)
        self.quantiles = quantiles
        self.pipeline = None
        self.load_error: Exception | None = None

    def _load(self):
        if self.load_error is not None:
            raise RuntimeError("Chronos model loading failed previously") from self.load_error
        if self.pipeline is None:
            try:
                dtype_name = str(self.params["torch_dtype"])
                kwargs: dict[str, Any] = {"device_map": self.params["device"]}
                if dtype_name != "auto":
                    kwargs["torch_dtype"] = getattr(torch, dtype_name)
                LOGGER.info(
                    "Loading Chronos model | model_id=%s device=%s dtype=%s",
                    self.params["model_id"],
                    self.params["device"],
                    dtype_name,
                )
                self.pipeline = BaseChronosPipeline.from_pretrained(self.params["model_id"], **kwargs)
            except Exception as exc:
                self.load_error = exc
                raise
        return self.pipeline

    def __call__(self, task: ForecastTask) -> ForecastOutput:
        pipeline = self._load()
        predict_kwargs: dict[str, Any] = {}
        if pipeline.forecast_type.value == "samples":
            predict_kwargs.update(
                num_samples=int(self.params["num_samples"]),
                temperature=self.params.get("temperature"),
                top_k=self.params.get("top_k"),
                top_p=self.params.get("top_p"),
            )
        quantiles, mean = pipeline.predict_quantiles(
            [torch.tensor(task.context, dtype=torch.float32)],
            prediction_length=task.prediction_length,
            quantile_levels=self.quantiles.tolist(),
            **predict_kwargs,
        )
        if isinstance(quantiles, list):
            quantile_values = quantiles[0].detach().cpu().numpy().reshape(task.prediction_length, -1)
            mean_values = mean[0].detach().cpu().numpy().reshape(-1)
        else:
            quantile_values = quantiles[0].detach().cpu().numpy()
            mean_values = mean[0].detach().cpu().numpy().reshape(-1)
        return ForecastOutput(mean_values.astype(np.float32), quantile_values.astype(np.float32))


def seasonal_naive_runner(params: Mapping[str, Any], quantiles: np.ndarray) -> Callable[[ForecastTask], ForecastOutput]:
    def run(task: ForecastTask) -> ForecastOutput:
        context = clean_numeric(task.context)
        use_dataset_seasonality = bool(params["use_dataset_seasonality"])
        period = task.seasonality if use_dataset_seasonality else int(params["seasonality"])
        if period <= 1 or len(context) < period:
            point = np.repeat(context[-1], task.prediction_length).astype(np.float32)
        else:
            pattern = context[-period:]
            point = np.tile(pattern, math.ceil(task.prediction_length / period))[: task.prediction_length]
        return ForecastOutput(point, repeated_quantiles(point, len(quantiles)))

    return run


def resolve_seasonal_order(value: Sequence[Any], task: ForecastTask) -> tuple[int, int, int, int]:
    if len(value) != 4:
        raise ValueError("seasonal_order must contain four values")
    period = task.seasonality if value[3] == "dataset_seasonality" else int(value[3])
    if period <= 1 and any(int(component) != 0 for component in value[:3]):
        LOGGER.warning(
            "Disabling seasonal ARIMA terms because dataset seasonality is <= 1 | dataset=%s item=%s",
            task.dataset,
            task.item_id,
        )
        return 0, 0, 0, 0
    return int(value[0]), int(value[1]), int(value[2]), period


def statsmodels_arima_runner(
    params: Mapping[str, Any], quantiles: np.ndarray
) -> Callable[[ForecastTask], ForecastOutput]:
    def run(task: ForecastTask) -> ForecastOutput:
        from statsmodels.tsa.arima.model import ARIMA

        order = tuple(map(int, params["order"]))
        seasonal_order = resolve_seasonal_order(params["seasonal_order"], task)
        model_kwargs = {
            "order": order,
            "seasonal_order": seasonal_order,
            "enforce_stationarity": bool(params["enforce_stationarity"]),
            "enforce_invertibility": bool(params["enforce_invertibility"]),
        }
        if params.get("trend") is not None:
            model_kwargs["trend"] = params["trend"]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = ARIMA(clean_numeric(task.context).astype(np.float64), **model_kwargs)
            fitted = model.fit(method_kwargs={"maxiter": int(params["maxiter"])})
            forecast = fitted.get_forecast(steps=task.prediction_length)
        mean = np.asarray(forecast.predicted_mean, dtype=np.float32)
        std = np.asarray(forecast.se_mean, dtype=np.float32)
        return ForecastOutput(mean, normal_quantiles(mean, std, quantiles))

    return run


def prophet_runner(params: Mapping[str, Any], quantiles: np.ndarray) -> Callable[[ForecastTask], ForecastOutput]:
    def run(task: ForecastTask) -> ForecastOutput:
        from prophet import Prophet

        model = Prophet(
            growth=str(params["growth"]),
            seasonality_mode=str(params["seasonality_mode"]),
            yearly_seasonality=params["yearly_seasonality"],
            weekly_seasonality=params["weekly_seasonality"],
            daily_seasonality=params["daily_seasonality"],
            changepoint_prior_scale=float(params["changepoint_prior_scale"]),
            seasonality_prior_scale=float(params["seasonality_prior_scale"]),
            interval_width=float(params["interval_width"]),
            uncertainty_samples=int(params["uncertainty_samples"]),
            mcmc_samples=int(params["mcmc_samples"]),
        )
        history = pd.DataFrame({"ds": task.context_timestamps, "y": clean_numeric(task.context)})
        future = pd.DataFrame({"ds": task.future_timestamps})
        model.fit(history)
        prediction_frame = model.predict(future)
        mean = prediction_frame["yhat"].to_numpy(dtype=np.float32)
        samples = np.asarray(model.predictive_samples(future)["yhat"], dtype=np.float32)
        if samples.shape[0] != task.prediction_length and samples.shape[1] == task.prediction_length:
            samples = samples.T
        if samples.shape[0] != task.prediction_length:
            raise ValueError(f"Unexpected Prophet sample shape: {samples.shape}")
        quantile_values = np.quantile(samples, quantiles, axis=1).T.astype(np.float32)
        return ForecastOutput(mean, quantile_values)

    return run


def deepar_runner(params: Mapping[str, Any], quantiles: np.ndarray) -> Callable[[ForecastTask], ForecastOutput]:
    def run(task: ForecastTask) -> ForecastOutput:
        from gluonts.dataset.common import ListDataset
        from gluonts.torch.model.deepar import DeepAREstimator

        frequency = period_compatible_frequency(task.frequency)
        start = pd.Period(task.context_timestamps[0], freq=frequency)
        train_dataset = ListDataset(
            [{"start": start, "target": clean_numeric(task.context)}],
            freq=frequency,
        )
        context_length = min(
            len(task.context),
            max(
                int(params["min_context_length"]),
                int(math.ceil(task.prediction_length * float(params["context_length_multiplier"]))),
            ),
        )
        with tempfile.TemporaryDirectory(
            prefix="pmds-deepar-",
            dir=str(params["temporary_directory"]),
        ) as temporary_directory:
            trainer_kwargs = dict(params["trainer"])
            trainer_kwargs["default_root_dir"] = temporary_directory
            estimator = DeepAREstimator(
                freq=frequency,
                prediction_length=task.prediction_length,
                context_length=context_length,
                hidden_size=int(params["hidden_size"]),
                num_layers=int(params["num_layers"]),
                dropout_rate=float(params["dropout_rate"]),
                lr=float(params["learning_rate"]),
                weight_decay=float(params["weight_decay"]),
                batch_size=int(params["batch_size"]),
                num_batches_per_epoch=int(params["num_batches_per_epoch"]),
                num_parallel_samples=int(params["num_parallel_samples"]),
                trainer_kwargs=trainer_kwargs,
            )
            predictor = estimator.train(train_dataset)
            forecast = next(iter(predictor.predict(train_dataset, num_samples=int(params["prediction_samples"]))))
            mean = np.asarray(forecast.mean, dtype=np.float32)
            quantile_values = np.stack([forecast.quantile(float(q)) for q in quantiles], axis=-1).astype(np.float32)
        return ForecastOutput(mean, quantile_values)

    return run


def build_model_runners(
    model_configs: Sequence[Mapping[str, Any]], quantiles: np.ndarray
) -> dict[str, Callable[[ForecastTask], ForecastOutput]]:
    builders: dict[str, Callable[[Mapping[str, Any], np.ndarray], Callable[[ForecastTask], ForecastOutput]]] = {
        "seasonal_naive": seasonal_naive_runner,
        "statsmodels_arima": statsmodels_arima_runner,
        "prophet": prophet_runner,
        "deepar": deepar_runner,
    }
    runners: dict[str, Callable[[ForecastTask], ForecastOutput]] = {}
    for model_config in model_configs:
        if not model_config.get("enabled", True):
            continue
        name = str(model_config["name"])
        model_type = str(model_config["type"])
        params = model_config.get("params", {})
        if model_type == "chronos":
            runners[name] = ChronosRunner(params, quantiles)
        elif model_type in builders:
            runners[name] = builders[model_type](params, quantiles)
        else:
            raise ValueError(f"Unsupported model type '{model_type}' for model '{name}'")
    return runners


def validate_output(output: ForecastOutput, task: ForecastTask, quantiles: np.ndarray) -> None:
    if output.mean.shape != (task.prediction_length,):
        raise ValueError(f"Mean forecast has shape {output.mean.shape}; expected {(task.prediction_length,)}")
    expected_quantile_shape = (task.prediction_length, len(quantiles))
    if output.quantiles.shape != expected_quantile_shape:
        raise ValueError(f"Quantile forecast has shape {output.quantiles.shape}; expected {expected_quantile_shape}")
    if not np.isfinite(output.mean).all() or not np.isfinite(output.quantiles).all():
        raise ValueError("Forecast contains non-finite values")
    crossing_count = int(np.sum(np.diff(output.quantiles, axis=1) < 0.0))
    if crossing_count:
        LOGGER.warning(
            "Quantile crossing detected | dataset=%s item=%s crossings=%d",
            task.dataset,
            task.item_id,
            crossing_count,
        )


def safe_predict(
    model_name: str,
    task: ForecastTask,
    runner: Callable[[ForecastTask], ForecastOutput],
    quantiles: np.ndarray,
) -> ForecastResult:
    start = time.monotonic()
    LOGGER.info("Model started | dataset=%s item=%s model=%s", task.dataset, task.item_id, model_name)
    try:
        output = runner(task)
        validate_output(output, task, quantiles)
        duration = time.monotonic() - start
        LOGGER.info(
            "Model completed | dataset=%s item=%s model=%s duration_seconds=%.3f",
            task.dataset,
            task.item_id,
            model_name,
            duration,
        )
        return ForecastResult(task.dataset, task.item_id, model_name, output, duration)
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
            task.dataset,
            task.item_id,
            model_name,
            None,
            duration,
            type(exc).__name__,
            str(exc),
        )


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


def run_experiment(config: Mapping[str, Any], config_path: Path) -> None:
    run_config = config["run"]
    evaluation_config = config["evaluation"]
    quantiles = np.asarray(evaluation_config["quantiles"], dtype=np.float32)
    metric_names = list(evaluation_config["metrics"])
    point_method = str(evaluation_config["point_forecast"])

    log_path = setup_logging(run_config)
    apply_environment(run_config.get("environment", {}))
    seed_everything(int(run_config["random_seed"]))
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

    rows: list[dict[str, Any]] = []
    statuses: list[dict[str, Any]] = []
    detailed_path = summary_path = status_path = Path()

    for dataset_index, spec in enumerate(dataset_specs, start=1):
        dataset_start = time.monotonic()
        LOGGER.info(
            "Dataset started | dataset=%s index=%d total=%d",
            spec.name,
            dataset_index,
            len(dataset_specs),
        )
        try:
            tasks = load_dataset_tasks(spec)
            if not tasks:
                raise RuntimeError(f"Dataset {spec.name} produced no forecast tasks")
            LOGGER.info("Dataset loaded | dataset=%s tasks=%d", spec.name, len(tasks))
            for task_index, task in enumerate(tasks, start=1):
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
                    result = safe_predict(model_name, task, runner, quantiles)
                    try:
                        rows.append(compute_metrics(task, result, metric_names, quantiles, point_method))
                    except Exception as exc:
                        LOGGER.exception(
                            "Metric computation failed | dataset=%s item=%s model=%s",
                            task.dataset,
                            task.item_id,
                            model_name,
                        )
                        metric_failure = ForecastResult(
                            task.dataset,
                            task.item_id,
                            model_name,
                            None,
                            result.duration_seconds,
                            type(exc).__name__,
                            str(exc),
                        )
                        rows.append(empty_metric_row(task, metric_failure))

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
            rows.extend(dataset_failure_rows(spec.name, runners, exc))
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

        detailed_path, summary_path, status_path = persist_results(rows, statuses, run_config, metric_names)

    summary = pd.read_csv(summary_path)
    LOGGER.info("Experiment completed | datasets=%d result_rows=%d", len(dataset_specs), len(rows))
    print("\nSummary")
    print(summary.to_string(index=False))
    print(f"\nDetailed results: {detailed_path}")
    print(f"Summary results:  {summary_path}")
    print(f"Dataset status:  {status_path}")
    print(f"Log file:        {log_path}")


def main() -> None:
    args = parse_args()
    config_path = args.config.expanduser().resolve()
    config = load_config(config_path)
    run_experiment(config, config_path)


if __name__ == "__main__":
    main()
