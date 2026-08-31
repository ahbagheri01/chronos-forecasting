"""Forecast model runners for PMDS benchmarks."""

from __future__ import annotations

import logging
import math
import tempfile
import warnings
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd
import torch

from pmds.schemas import ForecastOutput, ForecastTask
from pmds.utils import clean_numeric, period_compatible_frequency


LOGGER = logging.getLogger("pmds.compare")


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
                from chronos import BaseChronosPipeline

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
        return ForecastOutput(
            mean_values.astype(np.float32),
            quantile_values.astype(np.float32),
            distribution=f"chronos_{pipeline.forecast_type.value}",
        )


def _quantile_indices(requested: np.ndarray, available: Sequence[float], model_name: str) -> list[int]:
    available_array = np.asarray(available, dtype=float)
    indices: list[int] = []
    for level in requested:
        matches = np.flatnonzero(np.isclose(available_array, level, atol=1e-7, rtol=0.0))
        if len(matches) != 1:
            raise ValueError(f"{model_name} does not provide required quantile {level:g}; available={available}")
        indices.append(int(matches[0]))
    return indices


class TimesFMRunner:
    """Lazy adapter for the official TimesFM 2.5 PyTorch implementation."""

    OFFICIAL_QUANTILES = tuple(index / 10 for index in range(1, 10))
    INPUT_PATCH_LENGTH = 32

    def __init__(self, params: Mapping[str, Any], quantiles: np.ndarray):
        self.params = dict(params)
        self.quantiles = quantiles
        self.model = None
        self.load_error: Exception | None = None
        self.compiled_context: int | None = None
        self.quantile_indices = _quantile_indices(quantiles, self.OFFICIAL_QUANTILES, "TimesFM 2.5")

    def _load(self):
        if self.load_error is not None:
            raise RuntimeError("TimesFM model loading failed previously") from self.load_error
        if self.model is None:
            try:
                import timesfm

                kwargs: dict[str, Any] = {
                    "local_files_only": bool(self.params.get("local_files_only", False)),
                    "torch_compile": bool(self.params.get("torch_compile", False)),
                }
                if self.params.get("revision"):
                    kwargs["revision"] = self.params["revision"]
                LOGGER.info("Loading TimesFM model | model_id=%s", self.params["model_id"])
                model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(self.params["model_id"], **kwargs)
                self.model = model
            except ImportError as exc:
                self.load_error = exc
                raise ImportError("Install the PMDS foundation-model dependencies with: pip install -e '.[pmds]'") from exc
            except Exception as exc:
                self.load_error = exc
                raise
        return self.model

    def __call__(self, task: ForecastTask) -> ForecastOutput:
        context = clean_numeric(task.context)[-int(self.params.get("max_context", 512)) :]
        model = self._load()
        # Excess fully-masked leading patches can yield NaNs in the official
        # PyTorch decoder. Compile to the smallest patch-aligned context that
        # contains this task's real observations instead of always padding to
        # the experiment-wide cap.
        compiled_context = int(math.ceil(len(context) / self.INPUT_PATCH_LENGTH) * self.INPUT_PATCH_LENGTH)
        if compiled_context != self.compiled_context:
            import timesfm

            model.compile(
                timesfm.ForecastConfig(
                    max_context=compiled_context,
                    max_horizon=int(self.params.get("max_horizon", 256)),
                    normalize_inputs=bool(self.params.get("normalize_inputs", True)),
                    use_continuous_quantile_head=True,
                    fix_quantile_crossing=True,
                )
            )
            self.compiled_context = compiled_context
        point, probabilistic = model.forecast(horizon=task.prediction_length, inputs=[context])
        point_values = np.asarray(point, dtype=np.float32).reshape(1, task.prediction_length)[0]
        all_values = np.asarray(probabilistic, dtype=np.float32)
        if all_values.shape != (1, task.prediction_length, 10):
            raise ValueError(f"Unexpected TimesFM quantile shape: {all_values.shape}")
        # TimesFM places its mean in column 0 and q0.1,...,q0.9 in columns 1,...,9.
        mean = all_values[0, :, 0]
        quantile_values = np.take(all_values[0], np.asarray(self.quantile_indices) + 1, axis=1)
        if quantile_values.shape != (task.prediction_length, len(self.quantiles)):
            raise ValueError(f"Unexpected TimesFM selected quantile shape: {quantile_values.shape}")
        del point_values  # The official point output is the median; evaluation selects q0.5 explicitly.
        return ForecastOutput(mean, quantile_values, distribution="timesfm_quantiles")


class Moirai2Runner:
    """Lazy adapter for Salesforce Moirai 2.0 checkpoints."""

    def __init__(self, params: Mapping[str, Any], quantiles: np.ndarray):
        self.params = dict(params)
        self.quantiles = quantiles
        self.module = None
        self.forecast_class = None
        self.load_error: Exception | None = None
        self.quantile_indices: list[int] | None = None

    def _load(self):
        if self.load_error is not None:
            raise RuntimeError("Moirai model loading failed previously") from self.load_error
        if self.module is None:
            try:
                from uni2ts.model.moirai2 import Moirai2Forecast, Moirai2Module

                kwargs: dict[str, Any] = {}
                if self.params.get("revision"):
                    kwargs["revision"] = self.params["revision"]
                LOGGER.info("Loading Moirai model | model_id=%s device=%s", self.params["model_id"], self.params["device"])
                module = Moirai2Module.from_pretrained(self.params["model_id"], **kwargs)
                module = module.to(str(self.params.get("device", "cpu")))
                module.eval()
                available = [float(value) for value in module.quantile_levels]
                self.quantile_indices = _quantile_indices(self.quantiles, available, "Moirai 2.0")
                self.module = module
                self.forecast_class = Moirai2Forecast
            except ImportError as exc:
                self.load_error = exc
                raise ImportError("Install the PMDS foundation-model dependencies with: pip install -e '.[pmds]'") from exc
            except Exception as exc:
                self.load_error = exc
                raise
        return self.module

    def __call__(self, task: ForecastTask) -> ForecastOutput:
        module = self._load()
        context = clean_numeric(task.context)[-int(self.params.get("max_context", 512)) :]
        model = self.forecast_class(
            module=module,
            prediction_length=task.prediction_length,
            context_length=len(context),
            target_dim=1,
            feat_dynamic_real_dim=0,
            past_feat_dynamic_real_dim=0,
        ).to(str(self.params.get("device", "cpu")))
        with torch.inference_mode():
            prediction = model.predict([context])
        values = prediction.detach().cpu().numpy() if torch.is_tensor(prediction) else np.asarray(prediction)
        if values.ndim != 3 or values.shape[0] != 1 or values.shape[2] != task.prediction_length:
            raise ValueError(f"Unexpected Moirai quantile shape: {values.shape}")
        assert self.quantile_indices is not None
        quantile_values = values[0, self.quantile_indices, :].T.astype(np.float32)
        median_index = int(np.flatnonzero(np.isclose(self.quantiles, 0.5))[0])
        return ForecastOutput(
            quantile_values[:, median_index].copy(),
            quantile_values,
            distribution="moirai_quantiles_no_mean",
        )


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
        return ForecastOutput(point, repeated_quantiles(point, len(quantiles)), distribution="degenerate")

    return run


def zero_runner(params: Mapping[str, Any], quantiles: np.ndarray) -> Callable[[ForecastTask], ForecastOutput]:
    del params

    def run(task: ForecastTask) -> ForecastOutput:
        point = np.zeros(task.prediction_length, dtype=np.float32)
        return ForecastOutput(point, repeated_quantiles(point, len(quantiles)), distribution="degenerate")

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
        return ForecastOutput(mean, normal_quantiles(mean, std, quantiles), distribution="gaussian")

    return run


def _candidate_range(params: Mapping[str, Any], name: str) -> range:
    return range(int(params.get(f"min_{name}", 0)), int(params[f"max_{name}"]) + 1)


def _information_criterion(fitted, name: str, observations: int) -> float:
    criterion = name.lower()
    if criterion == "aic":
        return float(fitted.aic)
    if criterion == "bic":
        return float(fitted.bic)
    if criterion == "aicc":
        parameter_count = len(fitted.params)
        denominator = observations - parameter_count - 1
        if denominator <= 0:
            return float("inf")
        return float(fitted.aic + (2 * parameter_count * (parameter_count + 1)) / denominator)
    raise ValueError("information_criterion must be one of: aic, aicc, bic")


def _iter_auto_arima_orders(
    params: Mapping[str, Any], task: ForecastTask
) -> list[tuple[tuple[int, int, int], tuple[int, int, int, int]]]:
    max_order = int(params["max_order"])
    seasonal = bool(params["seasonal"])
    period = task.seasonality if bool(params["use_dataset_seasonality"]) else int(params["seasonality"])
    if period <= 1:
        seasonal = False

    orders = []
    seasonal_orders = [(0, 0, 0, 0)]
    if seasonal:
        seasonal_orders = [
            (seasonal_p, seasonal_d, seasonal_q, period)
            for seasonal_p in _candidate_range(params, "P")
            for seasonal_d in _candidate_range(params, "D")
            for seasonal_q in _candidate_range(params, "Q")
            if seasonal_p + seasonal_d + seasonal_q <= max_order
        ]

    for p in _candidate_range(params, "p"):
        for d in _candidate_range(params, "d"):
            for q in _candidate_range(params, "q"):
                if p + d + q > max_order:
                    continue
                for seasonal_order in seasonal_orders:
                    if p + d + q + sum(seasonal_order[:3]) <= max_order:
                        orders.append(((p, d, q), seasonal_order))
    if not orders:
        raise ValueError("AutoARIMA search space is empty")
    return orders


def auto_arima_runner(params: Mapping[str, Any], quantiles: np.ndarray) -> Callable[[ForecastTask], ForecastOutput]:
    def run(task: ForecastTask) -> ForecastOutput:
        from statsmodels.tsa.arima.model import ARIMA

        context = clean_numeric(task.context).astype(np.float64)
        selection_max_samples = int(params.get("selection_max_samples", 0))
        selection_context = context[-selection_max_samples:] if selection_max_samples > 0 else context
        if len(selection_context) <= task.prediction_length:
            selection_context = context
        best_fit = None
        best_order: tuple[int, int, int] | None = None
        best_seasonal_order: tuple[int, int, int, int] | None = None
        best_score = float("inf")
        failures: list[str] = []

        for order, seasonal_order in _iter_auto_arima_orders(params, task):
            model_kwargs = {
                "order": order,
                "seasonal_order": seasonal_order,
                "enforce_stationarity": bool(params["enforce_stationarity"]),
                "enforce_invertibility": bool(params["enforce_invertibility"]),
            }
            if params.get("trend") is not None and order[1] == 0:
                model_kwargs["trend"] = params["trend"]
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    fitted = ARIMA(selection_context, **model_kwargs).fit(
                        method_kwargs={"maxiter": int(params["maxiter"])}
                    )
                score = _information_criterion(fitted, str(params["information_criterion"]), len(selection_context))
                if np.isfinite(score) and score < best_score:
                    best_fit = fitted
                    best_order = order
                    best_seasonal_order = seasonal_order
                    best_score = score
            except Exception as exc:
                if len(failures) < 5:
                    failures.append(f"{order}/{seasonal_order}: {type(exc).__name__}: {exc}")

        if best_fit is None:
            detail = "; ".join(failures) if failures else "no candidate models were fitted"
            raise RuntimeError(f"AutoARIMA failed to fit any candidate for {task.dataset}/{task.item_id}: {detail}")

        LOGGER.info(
            "AutoARIMA selected | dataset=%s item=%s order=%s seasonal_order=%s %s=%.3f",
            task.dataset,
            task.item_id,
            best_order,
            best_seasonal_order,
            str(params["information_criterion"]).upper(),
            best_score,
        )
        if len(selection_context) != len(context):
            model_kwargs = {
                "order": best_order,
                "seasonal_order": best_seasonal_order,
                "enforce_stationarity": bool(params["enforce_stationarity"]),
                "enforce_invertibility": bool(params["enforce_invertibility"]),
            }
            if params.get("trend") is not None and best_order is not None and best_order[1] == 0:
                model_kwargs["trend"] = params["trend"]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                best_fit = ARIMA(context, **model_kwargs).fit(method_kwargs={"maxiter": int(params["maxiter"])})
        forecast = best_fit.get_forecast(steps=task.prediction_length)
        mean = np.asarray(forecast.predicted_mean, dtype=np.float32)
        std = np.asarray(forecast.se_mean, dtype=np.float32)
        return ForecastOutput(mean, normal_quantiles(mean, std, quantiles), distribution="gaussian")

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
        return ForecastOutput(mean, quantile_values, distribution="samples")

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
            if trainer_kwargs.get("enable_checkpointing") is False:
                LOGGER.warning(
                    "DeepAR requires checkpointing because GluonTS adds a ModelCheckpoint callback; enabling it."
                )
                trainer_kwargs["enable_checkpointing"] = True
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
        return ForecastOutput(mean, quantile_values, distribution="samples")

    return run


def build_model_runners(
    model_configs: Sequence[Mapping[str, Any]], quantiles: np.ndarray
) -> dict[str, Callable[[ForecastTask], ForecastOutput]]:
    builders: dict[str, Callable[[Mapping[str, Any], np.ndarray], Callable[[ForecastTask], ForecastOutput]]] = {
        "zero": zero_runner,
        "seasonal_naive": seasonal_naive_runner,
        "statsmodels_arima": statsmodels_arima_runner,
        "auto_arima": auto_arima_runner,
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
        elif model_type == "timesfm":
            runners[name] = TimesFMRunner(params, quantiles)
        elif model_type == "moirai2":
            runners[name] = Moirai2Runner(params, quantiles)
        elif model_type in builders:
            runners[name] = builders[model_type](params, quantiles)
        else:
            raise ValueError(f"Unsupported model type '{model_type}' for model '{name}'")
    return runners
