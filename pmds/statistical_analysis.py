#!/usr/bin/env python
"""Generate statistical diagnostics and model-comparison reports for PMDS results."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import warnings
from importlib.metadata import version
from pathlib import Path
from typing import Callable, Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyhomogeneity as hg
import pymannkendall as mk
from arch.bootstrap import MCS, SPA, StationaryBootstrap
from arch.unitroot import PhillipsPerron
from dieboldmariano import dm_test
from scipy.stats import binomtest, t as student_t, ttest_1samp, wilcoxon
from statsmodels.stats.diagnostic import acorr_ljungbox, breaks_cusumolsresid, het_arch
from statsmodels.stats.multitest import multipletests
from statsmodels.tsa.stattools import adfuller, kpss, zivot_andrews


plt.style.use("seaborn-v0_8-whitegrid")

DEFAULT_ALPHA = 0.05
DEFAULT_BOOTSTRAP_REPS = 5000
DEFAULT_SEED = 20260901
DEFAULT_PETTITT_SIMULATIONS = 0
CORE_METRICS = ("mae", "rmse", "smape", "mase", "wql_macro", "wql")


def stable_seed(base_seed: int, *parts: object) -> int:
    payload = "|".join([str(base_seed), *(str(part) for part in parts)]).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], byteorder="big", signed=False)


def successful_rows(frame: pd.DataFrame) -> pd.DataFrame:
    if "error_type" not in frame:
        return frame.copy()
    return frame.loc[frame["error_type"].fillna("").astype(str).eq("")].copy()


def numeric_values(values: Iterable[float]) -> np.ndarray:
    array = np.asarray(list(values), dtype=np.float64)
    return array[np.isfinite(array)]


def adjustment(
    frame: pd.DataFrame,
    pvalue_column: str,
    adjusted_column: str,
    alpha: float,
    method: str,
    group_columns: list[str] | None = None,
) -> pd.DataFrame:
    frame = frame.copy()
    frame[adjusted_column] = np.nan
    frame[f"{adjusted_column}_reject"] = False
    groups = [(None, frame)] if not group_columns else frame.groupby(group_columns, dropna=False, sort=False)
    for _, group in groups:
        valid = pd.to_numeric(group[pvalue_column], errors="coerce").notna()
        if not valid.any():
            continue
        indexes = group.index[valid]
        reject, adjusted, _, _ = multipletests(
            group.loc[indexes, pvalue_column].astype(float),
            alpha=alpha,
            method=method,
        )
        frame.loc[indexes, adjusted_column] = adjusted
        frame.loc[indexes, f"{adjusted_column}_reject"] = reject
    return frame


def observed_series(contexts: pd.DataFrame, forecasts: pd.DataFrame) -> pd.DataFrame:
    columns = ["dataset", "series_id", "timestamp", "actual"]
    combined = pd.concat([contexts[columns], forecasts[columns]], ignore_index=True)
    combined["timestamp"] = pd.to_datetime(combined["timestamp"], errors="coerce")
    combined["actual"] = pd.to_numeric(combined["actual"], errors="coerce")
    combined = combined.dropna(subset=["timestamp", "actual"])
    return (
        combined.groupby(["dataset", "series_id", "timestamp"], as_index=False, sort=True)["actual"]
        .first()
        .sort_values(["dataset", "series_id", "timestamp"])
    )


def stationarity_tests(
    contexts: pd.DataFrame,
    forecasts: pd.DataFrame,
    alpha: float,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (dataset, series_id), frame in observed_series(contexts, forecasts).groupby(
        ["dataset", "series_id"], sort=True
    ):
        frame = frame.sort_values("timestamp")
        values = numeric_values(frame["actual"])
        row: dict[str, object] = {
            "dataset": str(dataset),
            "series_id": str(series_id),
            "n_observations": len(values),
            "start_timestamp": str(frame["timestamp"].min()),
            "end_timestamp": str(frame["timestamp"].max()),
            "status": "ok",
            "error": "",
        }
        if len(values) < 12 or np.isclose(np.ptp(values), 0.0):
            row["status"] = "insufficient_or_constant_series"
            rows.append(row)
            continue
        errors: list[str] = []
        try:
            statistic, pvalue, lags, nobs, critical, _ = adfuller(values, regression="c", autolag="AIC")
            row.update(
                adf_statistic=float(statistic),
                adf_pvalue=float(pvalue),
                adf_lags=int(lags),
                adf_nobs=int(nobs),
                adf_critical_5pct=float(critical["5%"]),
            )
        except Exception as exc:
            errors.append(f"ADF: {type(exc).__name__}: {exc}")
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                statistic, pvalue, lags, critical = kpss(values, regression="c", nlags="auto")
            row.update(
                kpss_statistic=float(statistic),
                kpss_pvalue=float(pvalue),
                kpss_lags=int(lags),
                kpss_critical_5pct=float(critical["5%"]),
            )
        except Exception as exc:
            errors.append(f"KPSS: {type(exc).__name__}: {exc}")
        try:
            test = PhillipsPerron(values, trend="c", test_type="tau")
            row.update(
                pp_statistic=float(test.stat),
                pp_pvalue=float(test.pvalue),
                pp_lags=int(test.lags),
            )
        except Exception as exc:
            errors.append(f"PhillipsPerron: {type(exc).__name__}: {exc}")
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    statistic, pvalue, critical, lags, break_index = zivot_andrews(
                        values,
                        regression="c",
                        autolag="AIC",
                    )
                except (ValueError, np.linalg.LinAlgError):
                    statistic, pvalue, critical, lags, break_index = zivot_andrews(
                        values,
                        maxlag=min(5, max(0, len(values) // 10)),
                        regression="c",
                        autolag="AIC",
                    )
            row.update(
                zivot_andrews_statistic=float(statistic),
                zivot_andrews_pvalue=float(pvalue),
                zivot_andrews_lags=int(lags),
                zivot_andrews_critical_5pct=float(critical["5%"]),
                zivot_andrews_break_index=int(break_index),
                zivot_andrews_break_timestamp=str(frame.iloc[int(break_index)]["timestamp"]),
            )
        except Exception as exc:
            errors.append(f"ZivotAndrews: {type(exc).__name__}: {exc}")
        if errors:
            row["status"] = "partial"
            row["error"] = " | ".join(errors)
        rows.append(row)

    result = pd.DataFrame(rows)
    for name in ("adf", "kpss", "pp", "zivot_andrews"):
        pvalue_column = f"{name}_pvalue"
        if pvalue_column in result:
            result = adjustment(
                result,
                pvalue_column,
                f"{name}_pvalue_fdr",
                alpha,
                "fdr_bh",
                ["dataset"],
            )
    if {"adf_pvalue_fdr_reject", "kpss_pvalue_fdr_reject"}.issubset(result):
        conclusions: list[str] = []
        for _, row in result.iterrows():
            adf_reject = bool(row["adf_pvalue_fdr_reject"])
            kpss_reject = bool(row["kpss_pvalue_fdr_reject"])
            if adf_reject and not kpss_reject:
                conclusions.append("stationary")
            elif not adf_reject and kpss_reject:
                conclusions.append("nonstationary")
            elif adf_reject and kpss_reject:
                conclusions.append("conflicting_or_structural_break")
            else:
                conclusions.append("inconclusive")
        result["adf_kpss_fdr_conclusion"] = conclusions
    return result


def pettitt_result_row(
    values: np.ndarray,
    timestamps: pd.Series,
    alpha: float,
    simulations: int,
    seed: int,
) -> dict[str, object]:
    row: dict[str, object] = {
        "n_observations": len(values),
        "pettitt_simulations": simulations,
        "pettitt_pvalue_method": (
            "pyhomogeneity_analytic_approximation"
            if simulations == 0
            else "pyhomogeneity_monte_carlo"
        ),
        "status": "ok",
        "error": "",
    }
    if len(values) < 8 or np.isclose(np.ptp(values), 0.0):
        row["status"] = "insufficient_or_constant_series"
        return row
    try:
        # pyHomogeneity uses NumPy's global generator for its optional Monte Carlo p-value.
        # Preserve the caller's state so this report remains deterministic and side-effect free.
        random_state = np.random.get_state()
        try:
            np.random.seed(seed)
            test = hg.pettitt_test(values, alpha=alpha, sim=simulations)
        finally:
            np.random.set_state(random_state)
        change_after_observation = int(test.cp)
        change_before_index = max(0, change_after_observation - 1)
        change_after_index = min(change_after_observation, len(timestamps) - 1)
        pvalue = min(1.0, max(0.0, float(test.p)))
        row.update(
            pettitt_statistic=float(test.U),
            pettitt_pvalue=pvalue,
            pettitt_reject_raw=bool(pvalue < alpha),
            change_after_observation=change_after_observation,
            last_timestamp_before_change=str(timestamps.iloc[change_before_index]),
            first_timestamp_after_change=str(timestamps.iloc[change_after_index]),
            mean_before_change=float(test.avg.mu1),
            mean_after_change=float(test.avg.mu2),
            mean_change=float(test.avg.mu2 - test.avg.mu1),
        )
    except Exception as exc:
        row["status"] = "error"
        row["error"] = f"{type(exc).__name__}: {exc}"
    return row


def pettitt_change_points(
    contexts: pd.DataFrame,
    forecasts: pd.DataFrame,
    alpha: float,
    simulations: int,
    seed: int,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    targets = observed_series(contexts, forecasts)
    for (dataset, series_id), frame in targets.groupby(["dataset", "series_id"], sort=True):
        frame = frame.sort_values("timestamp").reset_index(drop=True)
        values = frame["actual"].to_numpy(dtype=float)
        rows.append(
            {
                "dataset": str(dataset),
                "model": "",
                "series_id": str(series_id),
                "series_type": "observed_target",
                **pettitt_result_row(
                    values,
                    frame["timestamp"],
                    alpha,
                    simulations,
                    stable_seed(seed, "pettitt", dataset, series_id, "target"),
                ),
            }
        )

    aggregate = aggregate_repetition_forecasts(forecasts)
    aggregate["absolute_error"] = np.abs(aggregate["actual"] - aggregate["point_forecast"])
    for (dataset, model, series_id), frame in aggregate.groupby(
        ["dataset", "model", "series_id"], sort=True
    ):
        frame = (
            frame.groupby("timestamp", as_index=False, sort=True)["absolute_error"]
            .mean()
            .sort_values("timestamp")
            .reset_index(drop=True)
        )
        values = frame["absolute_error"].to_numpy(dtype=float)
        rows.append(
            {
                "dataset": str(dataset),
                "model": str(model),
                "series_id": str(series_id),
                "series_type": "absolute_forecast_error_after_repetition_average",
                **pettitt_result_row(
                    values,
                    frame["timestamp"],
                    alpha,
                    simulations,
                    stable_seed(seed, "pettitt", dataset, model, series_id, "absolute_error"),
                ),
            }
        )

    result = pd.DataFrame(rows)
    result = adjustment(
        result,
        "pettitt_pvalue",
        "pettitt_pvalue_fdr",
        alpha,
        "fdr_bh",
        ["dataset", "series_type"],
    )
    result["inference_note"] = (
        "Pettitt detects one distributional change point; it does not compare forecast models. "
        "Use the FDR-adjusted decision within each dataset/series-type family. Serial dependence "
        "can affect its p-value, so interpret it with the residual autocorrelation diagnostics."
    )
    return result


def aggregate_repetition_forecasts(forecasts: pd.DataFrame) -> pd.DataFrame:
    quantiles = sorted(
        [column for column in forecasts if str(column).startswith("q_")],
        key=lambda value: int(str(value).split("_")[1]),
    )
    group = ["dataset", "series_id", "item_id", "origin", "step", "timestamp", "model"]
    aggregations: dict[str, str] = {
        "actual": "first",
        "point_forecast": "mean",
        "mean_forecast": "mean",
    }
    aggregations.update({column: "median" for column in quantiles})
    result = forecasts.groupby(group, as_index=False, sort=True).agg(aggregations)
    result["timestamp"] = pd.to_datetime(result["timestamp"], errors="coerce")
    return result


def diagnostic_lag(n_observations: int) -> int:
    return min(10, max(1, n_observations // 5))


def residual_reports(
    forecasts: pd.DataFrame,
    alpha: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    aggregate = aggregate_repetition_forecasts(forecasts)
    diagnostic_rows: list[dict[str, object]] = []
    bias_rows: list[dict[str, object]] = []
    trend_rows: list[dict[str, object]] = []
    for (dataset, model, series_id), frame in aggregate.groupby(
        ["dataset", "model", "series_id"], sort=True
    ):
        frame = frame.sort_values(["timestamp", "origin", "step"])
        residuals = numeric_values(frame["actual"] - frame["point_forecast"])
        nobs = len(residuals)
        lag = diagnostic_lag(nobs)
        base = {
            "dataset": str(dataset),
            "model": str(model),
            "series_id": str(series_id),
            "n_observations": nobs,
            "test_lag": lag,
            "status": "ok",
            "error": "",
        }
        diagnostics = dict(base)
        errors: list[str] = []
        if nobs < 8 or np.isclose(np.ptp(residuals), 0.0):
            diagnostics["status"] = "insufficient_or_constant_residuals"
        else:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    portmanteau = acorr_ljungbox(
                        residuals,
                        lags=[lag],
                        boxpierce=True,
                        model_df=0,
                        return_df=True,
                    ).iloc[0]
                diagnostics.update(
                    ljung_box_statistic=float(portmanteau["lb_stat"]),
                    ljung_box_pvalue=float(portmanteau["lb_pvalue"]),
                    box_pierce_statistic=float(portmanteau["bp_stat"]),
                    box_pierce_pvalue=float(portmanteau["bp_pvalue"]),
                )
            except Exception as exc:
                errors.append(f"Portmanteau: {type(exc).__name__}: {exc}")
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    lm_stat, lm_pvalue, f_stat, f_pvalue = het_arch(residuals, nlags=lag, ddof=0)
                diagnostics.update(
                    arch_lm_statistic=float(lm_stat),
                    arch_lm_pvalue=float(lm_pvalue),
                    arch_f_statistic=float(f_stat),
                    arch_f_pvalue=float(f_pvalue),
                )
            except Exception as exc:
                errors.append(f"ARCH-LM: {type(exc).__name__}: {exc}")
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    statistic, pvalue, _ = breaks_cusumolsresid(residuals - residuals.mean(), ddof=1)
                diagnostics.update(cusum_statistic=float(statistic), cusum_pvalue=float(pvalue))
            except Exception as exc:
                errors.append(f"CUSUM: {type(exc).__name__}: {exc}")
        if errors:
            diagnostics["status"] = "partial"
            diagnostics["error"] = " | ".join(errors)
        diagnostic_rows.append(diagnostics)

        bias = dict(base)
        bias.update(
            mean_signed_error=float(np.mean(residuals)) if nobs else np.nan,
            median_signed_error=float(np.median(residuals)) if nobs else np.nan,
            residual_mae=float(np.mean(np.abs(residuals))) if nobs else np.nan,
            residual_rmse=float(np.sqrt(np.mean(np.square(residuals)))) if nobs else np.nan,
        )
        if nobs >= 3 and np.allclose(residuals, 0.0):
            bias.update(
                status="constant_zero_residuals",
                bias_t_statistic=0.0,
                bias_t_pvalue=1.0,
                bias_wilcoxon_statistic=0.0,
                bias_wilcoxon_pvalue=1.0,
            )
        elif nobs >= 3:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                t_result = ttest_1samp(residuals, popmean=0.0, nan_policy="omit")
            bias.update(bias_t_statistic=float(t_result.statistic), bias_t_pvalue=float(t_result.pvalue))
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    w_result = wilcoxon(residuals, alternative="two-sided", zero_method="wilcox")
                bias.update(
                    bias_wilcoxon_statistic=float(w_result.statistic),
                    bias_wilcoxon_pvalue=float(w_result.pvalue),
                )
            except ValueError:
                bias.update(bias_wilcoxon_statistic=0.0, bias_wilcoxon_pvalue=1.0)
        else:
            bias["status"] = "insufficient_data"
        bias_rows.append(bias)

        for value_type, values in (
            ("signed_error", residuals),
            ("absolute_error", np.abs(residuals)),
        ):
            trend = dict(base)
            trend["value_type"] = value_type
            if nobs < 8 or np.isclose(np.ptp(values), 0.0):
                trend["status"] = "insufficient_or_constant_values"
            else:
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        original = mk.original_test(values, alpha=alpha)
                        modified = mk.hamed_rao_modification_test(values, alpha=alpha, lag=lag)
                    trend.update(
                        original_trend=str(original.trend),
                        original_pvalue=float(original.p),
                        original_z=float(original.z),
                        original_tau=float(original.Tau),
                        modified_trend=str(modified.trend),
                        modified_pvalue=float(modified.p),
                        modified_z=float(modified.z),
                        modified_tau=float(modified.Tau),
                        sens_slope=float(modified.slope),
                        sens_intercept=float(modified.intercept),
                    )
                except Exception as exc:
                    trend["status"] = "error"
                    trend["error"] = f"{type(exc).__name__}: {exc}"
            trend_rows.append(trend)

    diagnostics = pd.DataFrame(diagnostic_rows)
    for name in ("ljung_box", "box_pierce", "arch_lm", "cusum"):
        column = f"{name}_pvalue"
        if column in diagnostics:
            diagnostics = adjustment(
                diagnostics,
                column,
                f"{name}_pvalue_fdr",
                alpha,
                "fdr_bh",
                ["dataset"],
            )
    bias = pd.DataFrame(bias_rows)
    for name in ("bias_t", "bias_wilcoxon"):
        column = f"{name}_pvalue"
        if column in bias:
            bias = adjustment(bias, column, f"{name}_pvalue_fdr", alpha, "fdr_bh", ["dataset"])
    trends = pd.DataFrame(trend_rows)
    for name in ("original", "modified"):
        column = f"{name}_pvalue"
        if column in trends:
            trends = adjustment(
                trends,
                column,
                f"{name}_pvalue_fdr",
                alpha,
                "fdr_bh",
                ["dataset", "value_type"],
            )
    return diagnostics, bias, trends


def probabilistic_calibration(forecasts: pd.DataFrame, alpha: float) -> pd.DataFrame:
    aggregate = aggregate_repetition_forecasts(forecasts)
    quantile_columns = sorted(
        [column for column in aggregate if str(column).startswith("q_")],
        key=lambda value: int(str(value).split("_")[1]),
    )
    rows: list[dict[str, object]] = []
    for (dataset, model), frame in aggregate.groupby(["dataset", "model"], sort=True):
        actual = frame["actual"].to_numpy(dtype=float)
        common = {
            "dataset": str(dataset),
            "model": str(model),
            "n_observations": len(frame),
        }
        for column in quantile_columns:
            expected = int(column.split("_")[1]) / 100.0
            prediction = frame[column].to_numpy(dtype=float)
            valid = np.isfinite(actual) & np.isfinite(prediction)
            successes = int(np.sum(actual[valid] <= prediction[valid]))
            nobs = int(np.sum(valid))
            observed = successes / nobs if nobs else np.nan
            pvalue = float(binomtest(successes, nobs, expected).pvalue) if nobs else np.nan
            rows.append(
                {
                    **common,
                    "calibration_type": "quantile_coverage",
                    "level": column[2:],
                    "expected_coverage": expected,
                    "observed_coverage": observed,
                    "coverage_error": observed - expected if nobs else np.nan,
                    "absolute_coverage_error": abs(observed - expected) if nobs else np.nan,
                    "mean_interval_width": np.nan,
                    "binomial_pvalue": pvalue,
                    "inference_note": "Binomial p-value is diagnostic; serial dependence is not removed.",
                }
            )
        for low_index in range(len(quantile_columns) // 2):
            low_column = quantile_columns[low_index]
            high_column = quantile_columns[-low_index - 1]
            low_level = int(low_column.split("_")[1]) / 100.0
            high_level = int(high_column.split("_")[1]) / 100.0
            low = frame[low_column].to_numpy(dtype=float)
            high = frame[high_column].to_numpy(dtype=float)
            valid = np.isfinite(actual) & np.isfinite(low) & np.isfinite(high)
            covered = (actual[valid] >= low[valid]) & (actual[valid] <= high[valid])
            successes = int(np.sum(covered))
            nobs = int(np.sum(valid))
            expected = high_level - low_level
            observed = successes / nobs if nobs else np.nan
            pvalue = float(binomtest(successes, nobs, expected).pvalue) if nobs else np.nan
            rows.append(
                {
                    **common,
                    "calibration_type": "central_interval_coverage",
                    "level": f"{int(low_level * 100)}-{int(high_level * 100)}",
                    "expected_coverage": expected,
                    "observed_coverage": observed,
                    "coverage_error": observed - expected if nobs else np.nan,
                    "absolute_coverage_error": abs(observed - expected) if nobs else np.nan,
                    "mean_interval_width": float(np.mean(high[valid] - low[valid])) if nobs else np.nan,
                    "binomial_pvalue": pvalue,
                    "inference_note": "Binomial p-value is diagnostic; serial dependence is not removed.",
                }
            )
        quantile_values = frame[quantile_columns].to_numpy(dtype=float)
        adjacent = np.diff(quantile_values, axis=1)
        rows.append(
            {
                **common,
                "calibration_type": "quantile_crossing",
                "level": "all_adjacent_pairs",
                "expected_coverage": 0.0,
                "observed_coverage": float(np.mean(adjacent < 0.0)),
                "coverage_error": float(np.mean(adjacent < 0.0)),
                "absolute_coverage_error": float(np.mean(adjacent < 0.0)),
                "mean_interval_width": np.nan,
                "binomial_pvalue": np.nan,
                "inference_note": "Observed value is the adjacent-quantile crossing rate.",
            }
        )
    result = pd.DataFrame(rows)
    return adjustment(
        result,
        "binomial_pvalue",
        "binomial_pvalue_fdr",
        alpha,
        "fdr_bh",
        ["dataset", "calibration_type"],
    )


def task_loss_matrix(detailed: pd.DataFrame, dataset: str, metric: str) -> pd.DataFrame:
    frame = detailed.loc[detailed["dataset"].astype(str).eq(dataset)].copy()
    keys = ["series_id", "item_id", "origin"]
    group = [*keys, "model"]
    if metric == "wql":
        values = frame.groupby(group, sort=True)[["wql_loss_sum", "wql_abs_target_sum"]].mean()
        denominators = values["wql_abs_target_sum"].groupby(level=keys).first()
        denominators = denominators.loc[denominators.gt(0) & denominators.notna()]
        if denominators.empty:
            return pd.DataFrame()
        total_denominator = float(denominators.sum())
        values = values.loc[values.index.droplevel("model").isin(denominators.index)].copy()
        values["loss"] = len(denominators) * values["wql_loss_sum"] / total_denominator
        matrix = values["loss"].unstack("model")
    elif metric == "wql_macro":
        matrix = frame.groupby(group, sort=True)["wql"].mean().unstack("model")
    else:
        if metric not in frame or not frame[metric].notna().any():
            return pd.DataFrame()
        matrix = frame.groupby(group, sort=True)[metric].mean().unstack("model")
    matrix = matrix.dropna(axis=0, how="any")
    if matrix.empty:
        return matrix
    ordered = matrix.reset_index().sort_values(
        ["series_id", "origin", "item_id"],
        ascending=[True, False, True],
    )
    return ordered.set_index(keys)[matrix.columns]


def repetition_variability(
    detailed: pd.DataFrame,
    metrics: Iterable[str],
    alpha: float,
) -> pd.DataFrame:
    score_rows: list[dict[str, object]] = []
    group_columns = ["dataset", "model", "repetition"]
    for (dataset, model, repetition), frame in detailed.groupby(group_columns, sort=True):
        for metric in metrics:
            if metric == "wql":
                numerator = pd.to_numeric(frame["wql_loss_sum"], errors="coerce").sum(min_count=1)
                denominator = pd.to_numeric(frame["wql_abs_target_sum"], errors="coerce").sum(min_count=1)
                score = numerator / denominator if denominator > 0 else np.nan
            elif metric == "wql_macro":
                score = pd.to_numeric(frame["wql"], errors="coerce").mean()
            elif metric in frame:
                score = pd.to_numeric(frame[metric], errors="coerce").mean()
            else:
                score = np.nan
            if np.isfinite(score):
                score_rows.append(
                    {
                        "dataset": str(dataset),
                        "model": str(model),
                        "repetition": int(repetition),
                        "metric": str(metric),
                        "score": float(score),
                    }
                )

    rows: list[dict[str, object]] = []
    scores = pd.DataFrame(score_rows)
    for (dataset, metric, model), frame in scores.groupby(["dataset", "metric", "model"], sort=True):
        values = frame["score"].to_numpy(dtype=float)
        n_repetitions = len(values)
        mean_score = float(np.mean(values))
        sample_sd = float(np.std(values, ddof=1)) if n_repetitions >= 2 else np.nan
        standard_error = sample_sd / math.sqrt(n_repetitions) if n_repetitions >= 2 else np.nan
        if n_repetitions >= 2 and np.isfinite(standard_error):
            critical = float(student_t.ppf(1.0 - alpha / 2.0, df=n_repetitions - 1))
            ci_lower = mean_score - critical * standard_error
            ci_upper = mean_score + critical * standard_error
        else:
            ci_lower = np.nan
            ci_upper = np.nan
        rows.append(
            {
                "dataset": str(dataset),
                "metric": str(metric),
                "model": str(model),
                "n_repetitions": n_repetitions,
                "repetition_mean": mean_score,
                "repetition_sample_sd": sample_sd,
                "repetition_standard_error": standard_error,
                "repetition_t_ci_95_lower": ci_lower,
                "repetition_t_ci_95_upper": ci_upper,
                "coefficient_of_variation": (
                    sample_sd / abs(mean_score)
                    if np.isfinite(sample_sd) and not np.isclose(mean_score, 0.0)
                    else np.nan
                ),
                "variability_class": (
                    "identical_across_repetitions"
                    if np.isfinite(sample_sd) and np.isclose(sample_sd, 0.0)
                    else "seed_sensitive"
                    if np.isfinite(sample_sd)
                    else "insufficient_repetitions"
                ),
                "inference_note": (
                    "Descriptive seed variability only. With three repetitions, SD and its t interval "
                    "must not be used alone to claim a statistically significant winner."
                ),
            }
        )
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    result["empirical_rank_by_repetition_mean"] = result.groupby(
        ["dataset", "metric"], sort=False
    )["repetition_mean"].rank(method="min", ascending=True).astype(int)
    result["winner_by_repetition_mean"] = result["empirical_rank_by_repetition_mean"].eq(1)
    return result.sort_values(
        ["dataset", "metric", "empirical_rank_by_repetition_mean", "model"]
    ).reset_index(drop=True)


def add_repetition_variability_to_pairs(
    pairs: pd.DataFrame,
    variability: pd.DataFrame,
) -> pd.DataFrame:
    if pairs.empty:
        return pairs
    result = pairs.copy()
    lookup = variability.set_index(["dataset", "metric", "model"])
    rows: list[dict[str, object]] = []
    for _, pair in result.iterrows():
        dataset = str(pair["dataset"])
        metric = str(pair["metric"])
        winner = str(pair["winner"])
        competitor = str(pair["competitor"])

        def value(model: str, column: str) -> float:
            try:
                return float(lookup.loc[(dataset, metric, model), column])
            except (KeyError, TypeError, ValueError):
                return np.nan

        winner_mean = value(winner, "repetition_mean")
        competitor_mean = value(competitor, "repetition_mean")
        winner_sd = value(winner, "repetition_sample_sd")
        competitor_sd = value(competitor, "repetition_sample_sd")
        winner_repetitions = value(winner, "n_repetitions")
        competitor_repetitions = value(competitor, "n_repetitions")
        combined_sd = (
            math.sqrt(winner_sd**2 + competitor_sd**2)
            if np.isfinite(winner_sd) and np.isfinite(competitor_sd)
            else np.nan
        )
        mean_gap = competitor_mean - winner_mean
        if not all(np.isfinite(value_) for value_ in (winner_mean, competitor_mean, winner_sd, competitor_sd)):
            separation = "unavailable"
            bands_nonoverlap: object = np.nan
        elif np.isclose(winner_sd, 0.0) and np.isclose(competitor_sd, 0.0):
            separation = "both_models_identical_across_repetitions"
            bands_nonoverlap = bool(winner_mean < competitor_mean)
        else:
            bands_nonoverlap = bool(winner_mean + winner_sd < competitor_mean - competitor_sd)
            separation = "one_sd_bands_nonoverlap" if bands_nonoverlap else "one_sd_bands_overlap"
        rows.append(
            {
                "winner_repetition_mean": winner_mean,
                "winner_repetition_sample_sd": winner_sd,
                "winner_n_repetitions": winner_repetitions,
                "competitor_repetition_mean": competitor_mean,
                "competitor_repetition_sample_sd": competitor_sd,
                "competitor_n_repetitions": competitor_repetitions,
                "repetition_mean_gap": mean_gap,
                "combined_repetition_sd": combined_sd,
                "gap_over_combined_repetition_sd": (
                    mean_gap / combined_sd
                    if np.isfinite(combined_sd) and not np.isclose(combined_sd, 0.0)
                    else np.nan
                ),
                "one_sd_bands_nonoverlap": bands_nonoverlap,
                "repetition_descriptive_separation": separation,
                "repetition_inference_note": (
                    "Descriptive only; three repetitions estimate random-seed variability, not "
                    "dataset/task sampling uncertainty or winner significance."
                ),
            }
        )
    return pd.concat([result.reset_index(drop=True), pd.DataFrame(rows)], axis=1)


def precomputed_loss(_actual: float, loss_value: float) -> float:
    return loss_value


def diebold_mariano_comparisons(
    detailed: pd.DataFrame,
    metrics: Iterable[str],
    alpha: float,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    datasets = sorted(map(str, detailed["dataset"].dropna().unique()))
    for dataset in datasets:
        for metric in metrics:
            matrix = task_loss_matrix(detailed, dataset, metric)
            if matrix.empty or matrix.shape[1] < 2:
                continue
            model_means = matrix.mean(axis=0).sort_values()
            winner = str(model_means.index[0])
            winner_losses = matrix[winner].to_numpy(dtype=float)
            nobs = len(matrix)
            n_series = matrix.reset_index()["series_id"].nunique()
            quality = sample_quality(nobs, n_series)
            for competitor in model_means.index[1:]:
                competitor_name = str(competitor)
                competitor_losses = matrix[competitor_name].to_numpy(dtype=float)
                differences = winner_losses - competitor_losses
                status = "ok"
                error = ""
                statistic = np.nan
                one_sided_pvalue = np.nan
                two_sided_pvalue = np.nan
                if nobs < 8:
                    status = "not_run_fewer_than_8_ordered_tasks"
                elif np.isclose(np.var(differences), 0.0):
                    status = "not_run_zero_variance_loss_differential"
                else:
                    try:
                        actual = np.zeros(nobs, dtype=float).tolist()
                        winner_values = winner_losses.tolist()
                        competitor_values = competitor_losses.tolist()
                        statistic, one_sided_pvalue = dm_test(
                            actual,
                            winner_values,
                            competitor_values,
                            loss=precomputed_loss,
                            h=1,
                            one_sided=True,
                            harvey_correction=True,
                            variance_estimator="bartlett",
                        )
                        _, two_sided_pvalue = dm_test(
                            actual,
                            winner_values,
                            competitor_values,
                            loss=precomputed_loss,
                            h=1,
                            one_sided=False,
                            harvey_correction=True,
                            variance_estimator="bartlett",
                        )
                    except Exception as exc:
                        status = "error"
                        error = f"{type(exc).__name__}: {exc}"
                rows.append(
                    {
                        "dataset": dataset,
                        "metric": str(metric),
                        "winner": winner,
                        "competitor": competitor_name,
                        "winner_mean_loss": float(model_means[winner]),
                        "competitor_mean_loss": float(model_means[competitor_name]),
                        "winner_advantage": float(np.mean(competitor_losses - winner_losses)),
                        "dm_mean_loss_differential_winner_minus_competitor": float(np.mean(differences)),
                        "dm_statistic": float(statistic),
                        "dm_pvalue_one_sided_winner_better": float(one_sided_pvalue),
                        "dm_pvalue_two_sided": float(two_sided_pvalue),
                        "dm_h": 1,
                        "dm_harvey_correction": True,
                        "dm_variance_estimator": "bartlett",
                        "n_ordered_tasks": nobs,
                        "n_series": n_series,
                        "sample_quality": quality,
                        "sampling_unit": "forecast_task_after_averaging_three_repetitions",
                        "applicability": "exploratory_panel_task_sequence_not_fixed_horizon_time_series",
                        "status": status,
                        "error": error,
                    }
                )
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    result = adjustment(
        result,
        "dm_pvalue_one_sided_winner_better",
        "dm_pvalue_holm",
        alpha,
        "holm",
        ["dataset", "metric"],
    ).rename(columns={"dm_pvalue_holm_reject": "dm_significant_after_holm"})
    result["reportable_as_confirmatory"] = False
    decisions: list[str] = []
    for _, row in result.iterrows():
        if row["status"] != "ok":
            decisions.append(str(row["status"]))
        elif bool(row["dm_significant_after_holm"]):
            decisions.append("exploratory_dm_favors_winner")
        else:
            decisions.append("exploratory_dm_not_significant")
    result["inference_decision"] = decisions
    result["inference_note"] = (
        "The library HLN-DM calculation is exploratory because ordered task losses pool series and origins. "
        "Use SPA/MCS as the primary benchmark-level inference; a classical confirmatory DM design needs a "
        "long fixed-horizon out-of-sample loss-differential series."
    )
    return result


def bootstrap_block_size(n_observations: int) -> int:
    return min(n_observations, max(1, int(math.ceil(math.sqrt(n_observations)))))


def bootstrap_mean_interval(
    differences: np.ndarray,
    block_size: int,
    reps: int,
    seed: int,
) -> tuple[float, float]:
    bootstrap = StationaryBootstrap(block_size, differences, seed=seed)

    def mean_statistic(values: np.ndarray) -> np.ndarray:
        return np.asarray([np.mean(values)], dtype=float)

    interval = bootstrap.conf_int(
        mean_statistic,
        reps=reps,
        method="percentile",
        size=0.95,
    )
    return float(interval[0, 0]), float(interval[1, 0])


def sample_quality(n_observations: int, n_series: int) -> str:
    if n_observations < 8 or n_series <= 1:
        return "very_low_power"
    if n_observations < 20 or n_series < 5:
        return "low_power"
    return "adequate_with_dependence_caveat"


def model_comparisons(
    detailed: pd.DataFrame,
    metrics: list[str],
    alpha: float,
    bootstrap_reps: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    pairwise_rows: list[dict[str, object]] = []
    mcs_rows: list[dict[str, object]] = []
    datasets = sorted(map(str, detailed["dataset"].dropna().unique()))
    for dataset in datasets:
        for metric in metrics:
            matrix = task_loss_matrix(detailed, dataset, metric)
            if matrix.empty or matrix.shape[1] < 2:
                continue
            nobs = len(matrix)
            n_series = matrix.reset_index()["series_id"].nunique()
            block_size = bootstrap_block_size(nobs)
            model_means = matrix.mean(axis=0).sort_values()
            winner = str(model_means.index[0])
            quality = sample_quality(nobs, n_series)
            comparison_seed = stable_seed(seed, dataset, metric)
            included: set[str] = set()
            mcs_pvalues: dict[str, float] = {}
            mcs_status = "ok"
            mcs_error = ""
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    mcs = MCS(
                        matrix,
                        size=alpha,
                        reps=bootstrap_reps,
                        block_size=block_size,
                        method="R",
                        bootstrap="stationary",
                        seed=comparison_seed,
                    )
                    mcs.compute()
                included = set(map(str, mcs.included))
                mcs_pvalues = {
                    str(index): float(value)
                    for index, value in mcs.pvalues["Pvalue"].items()
                }
            except Exception as exc:
                mcs_status = "error"
                mcs_error = f"{type(exc).__name__}: {exc}"
            for rank, (model, mean_loss) in enumerate(model_means.items(), start=1):
                model_name = str(model)
                mcs_rows.append(
                    {
                        "dataset": dataset,
                        "metric": metric,
                        "model": model_name,
                        "mean_loss": float(mean_loss),
                        "empirical_rank": rank,
                        "empirical_winner": model_name == winner,
                        "included_in_95pct_mcs": model_name in included if mcs_status == "ok" else np.nan,
                        "mcs_elimination_pvalue": mcs_pvalues.get(model_name, np.nan),
                        "n_paired_tasks": nobs,
                        "n_series": n_series,
                        "bootstrap_block_size": block_size,
                        "bootstrap_reps": bootstrap_reps,
                        "sample_quality": quality,
                        "status": mcs_status,
                        "error": mcs_error,
                    }
                )
            winner_losses = matrix[winner].to_numpy(dtype=float)
            for competitor in model_means.index[1:]:
                competitor_name = str(competitor)
                competitor_losses = matrix[competitor_name].to_numpy(dtype=float)
                differences = competitor_losses - winner_losses
                effect = float(np.mean(differences))
                competitor_mean = float(np.mean(competitor_losses))
                pair_seed = stable_seed(comparison_seed, competitor_name)
                status = "ok"
                error = ""
                spa_pvalue = np.nan
                ci_low = np.nan
                ci_high = np.nan
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        spa = SPA(
                            benchmark=competitor_losses,
                            models=pd.DataFrame({winner: winner_losses}),
                            block_size=block_size,
                            reps=bootstrap_reps,
                            bootstrap="stationary",
                            studentize=True,
                            nested=False,
                            seed=pair_seed,
                        )
                        spa.compute()
                    spa_pvalue = float(spa.pvalues["consistent"])
                    ci_low, ci_high = bootstrap_mean_interval(
                        differences,
                        block_size,
                        bootstrap_reps,
                        pair_seed,
                    )
                except Exception as exc:
                    status = "error"
                    error = f"{type(exc).__name__}: {exc}"
                pairwise_rows.append(
                    {
                        "dataset": dataset,
                        "metric": metric,
                        "winner": winner,
                        "competitor": competitor_name,
                        "winner_mean_loss": float(model_means[winner]),
                        "competitor_mean_loss": competitor_mean,
                        "winner_advantage": effect,
                        "relative_improvement_percent": (
                            100.0 * effect / abs(competitor_mean)
                            if not np.isclose(competitor_mean, 0.0)
                            else np.nan
                        ),
                        "ci_95_lower": ci_low,
                        "ci_95_upper": ci_high,
                        "spa_pvalue": spa_pvalue,
                        "n_paired_tasks": nobs,
                        "n_series": n_series,
                        "bootstrap_block_size": block_size,
                        "bootstrap_reps": bootstrap_reps,
                        "sample_quality": quality,
                        "winner_selected_on_same_data": True,
                        "status": status,
                        "error": error,
                    }
                )
    pairwise = pd.DataFrame(pairwise_rows)
    if not pairwise.empty:
        pairwise = adjustment(
            pairwise,
            "spa_pvalue",
            "spa_pvalue_holm",
            alpha,
            "holm",
            ["dataset", "metric"],
        )
        pairwise = pairwise.rename(
            columns={"spa_pvalue_holm_reject": "significant_after_holm"}
        )
        pairwise["reportable_significance"] = (
            pairwise["significant_after_holm"]
            & pairwise["sample_quality"].eq("adequate_with_dependence_caveat")
        )
        decisions: list[str] = []
        for _, row in pairwise.iterrows():
            if row["sample_quality"] == "very_low_power":
                decisions.append("not_reportable_very_low_power")
            elif row["sample_quality"] == "low_power":
                decisions.append(
                    "exploratory_significance_low_power"
                    if bool(row["significant_after_holm"])
                    else "inconclusive_low_power"
                )
            elif bool(row["significant_after_holm"]):
                decisions.append("winner_significantly_better")
            else:
                decisions.append("not_significant")
        pairwise["inference_decision"] = decisions
    return pairwise, pd.DataFrame(mcs_rows)


def safe_filename(value: str) -> str:
    return "".join(character if character.isalnum() or character in "-_." else "_" for character in value)


def save_winner_plots(pairwise: pd.DataFrame, output_dir: Path) -> int:
    if pairwise.empty:
        return 0
    count = 0
    for (dataset, metric), frame in pairwise.groupby(["dataset", "metric"], sort=True):
        frame = frame.sort_values("winner_advantage", ascending=True).reset_index(drop=True)
        y = np.arange(len(frame))
        effects = frame["winner_advantage"].to_numpy(dtype=float)
        lower = frame["ci_95_lower"].to_numpy(dtype=float)
        upper = frame["ci_95_upper"].to_numpy(dtype=float)
        finite_interval = np.isfinite(lower) & np.isfinite(upper)
        lower_error = np.where(finite_interval, np.maximum(effects - lower, 0.0), 0.0)
        upper_error = np.where(finite_interval, np.maximum(upper - effects, 0.0), 0.0)
        significant = frame["significant_after_holm"].fillna(False).to_numpy(dtype=bool)
        reportable = frame["reportable_significance"].fillna(False).to_numpy(dtype=bool)
        colors = np.where(reportable, "#087E8B", np.where(significant, "#E09F3E", "#7A8793"))
        fig, axis = plt.subplots(figsize=(10.5, max(5.0, 0.52 * len(frame) + 1.8)))
        for index in range(len(frame)):
            axis.errorbar(
                effects[index],
                y[index],
                xerr=np.asarray([[lower_error[index]], [upper_error[index]]]),
                fmt="o",
                color=colors[index],
                ecolor=colors[index],
                capsize=4,
                markersize=7,
            )
        axis.axvline(0.0, color="#D95D39", linestyle="--", linewidth=1.1)
        axis.set_yticks(y, labels=frame["competitor"])
        axis.set_xlabel("Winner advantage: competitor loss - winner loss (95% block-bootstrap CI)")
        axis.set_ylabel("Competitor")
        winner = str(frame.iloc[0]["winner"])
        axis.set_title(
            f"{str(dataset).replace('_', ' ').title()} | {str(metric).upper()}\n"
            f"Winner: {winner} | teal = reportable; amber = Holm-significant but low-power",
            weight="bold",
        )
        axis.margins(x=0.12, y=0.08)
        fig.tight_layout()
        path = output_dir / "winner_vs_rest" / safe_filename(str(dataset)) / f"{safe_filename(str(metric))}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=220, bbox_inches="tight")
        plt.close(fig)
        count += 1
    return count


def save_mcs_membership_plots(mcs: pd.DataFrame, output_dir: Path) -> int:
    if mcs.empty:
        return 0
    count = 0
    for dataset, frame in mcs.groupby("dataset", sort=True):
        pivot = frame.pivot(index="model", columns="metric", values="included_in_95pct_mcs")
        if pivot.empty or not pivot.notna().any().any():
            continue
        order = (
            frame.groupby("model")["empirical_rank"]
            .mean()
            .sort_values()
            .index
        )
        pivot = pivot.reindex(order)
        values = pivot.astype(float).to_numpy()
        fig, axis = plt.subplots(
            figsize=(max(10.0, 1.65 * len(pivot.columns)), max(5.0, 0.5 * len(pivot.index)))
        )
        image = axis.imshow(values, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)
        metric_labels = [str(value).upper().replace("_", "\n") for value in pivot.columns]
        axis.set_xticks(range(len(pivot.columns)), labels=metric_labels, rotation=0)
        axis.set_yticks(range(len(pivot.index)), labels=pivot.index)
        for row_index in range(len(pivot.index)):
            for column_index in range(len(pivot.columns)):
                value = values[row_index, column_index]
                label = "Yes" if value == 1 else "No" if value == 0 else "NA"
                axis.text(column_index, row_index, label, ha="center", va="center", fontsize=8)
        axis.set_title(
            f"{str(dataset).replace('_', ' ').title()}\n95% Model Confidence Set membership",
            weight="bold",
        )
        fig.colorbar(image, ax=axis, ticks=[0, 1], label="Included")
        fig.tight_layout()
        path = output_dir / "model_confidence_set" / f"{safe_filename(str(dataset))}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=220, bbox_inches="tight")
        plt.close(fig)
        count += 1
    return count


def write_methodology(
    output_dir: Path,
    alpha: float,
    bootstrap_reps: int,
    pettitt_simulations: int,
    counts: dict[str, int],
) -> None:
    content = f"""# PMDS statistical diagnostics and model-comparison results

These reports use the completed `11_12_3_rep` benchmark outputs. No forecasting models are rerun.

## Established implementations

- `statsmodels`: ADF, KPSS, Zivot-Andrews, Box-Pierce, Ljung-Box, ARCH-LM, CUSUM, and p-value correction.
- `arch`: Phillips-Perron, Hansen SPA, Hansen-Lunde-Nason Model Confidence Set, and stationary bootstrap confidence intervals.
- `dieboldmariano`: Diebold-Mariano test with the Harvey-Leybourne-Newbold small-sample correction.
- `pyHomogeneity`: Pettitt's single-change-point test.
- `pymannkendall`: original and Hamed-Rao modified Mann-Kendall tests.
- `scipy`: signed-error bias tests and binomial calibration diagnostics.

## Settings

- Significance level: `{alpha}`.
- Bootstrap replications: `{bootstrap_reps}`.
- Pettitt Monte Carlo simulations: `{pettitt_simulations}` (`0` selects pyHomogeneity's analytic approximation).
- Three model repetitions are averaged within each forecast task before model comparison.
- Winner-versus-rest SPA p-values are Holm-adjusted within each dataset/metric family.
- Diagnostic p-values are Benjamini-Hochberg FDR-adjusted within dataset/test families.
- MCS and pairwise confidence intervals use stationary bootstraps with block length `ceil(sqrt(n_tasks))`.

## Interpretation

- ADF and Phillips-Perron have a unit-root null; KPSS has a stationarity null.
- Box-Pierce and Ljung-Box diagnose residual autocorrelation; ARCH-LM diagnoses changing residual variance.
- Mann-Kendall diagnoses monotonic drift and is not a model-superiority test.
- Pettitt diagnoses one change point in a target or absolute-error series and is not a model-superiority test.
- `winner_advantage = competitor loss - winner loss`; positive values favor the empirical winner.
- The Model Confidence Set is the primary protection against selecting a winner on the same data used for testing.
- DM results are exploratory here because their ordered loss series pools forecast tasks from multiple series and origins; a classical confirmatory DM design requires a long fixed-horizon loss-differential time series.
- Repetition SD describes sensitivity to random seeds. Three repetitions do not estimate dataset/task sampling uncertainty reliably, so SD or overlap of SD bands is not a significance test.
- `very_low_power` and `low_power` rows must not be presented as strong evidence even when a p-value is small.
- `reportable_significance` is true only for Holm-significant rows classified as `adequate_with_dependence_caveat`.
- Calibration binomial p-values are diagnostic because forecast observations are serially dependent.

## Generated records

{chr(10).join(f'- `{name}`: {value} rows' for name, value in counts.items())}
"""
    (output_dir / "README.md").write_text(content, encoding="utf-8")


def run_analysis(
    detailed_path: Path,
    forecast_path: Path,
    context_path: Path,
    config_path: Path,
    output_dir: Path,
    alpha: float,
    bootstrap_reps: int,
    pettitt_simulations: int,
    seed: int,
) -> dict[str, int]:
    with config_path.open(encoding="utf-8") as fp:
        config = json.load(fp)
    metrics = list(config.get("evaluation", {}).get("metrics", ["mae", "rmse", "smape", "mase", "wql"]))
    if "wql" in metrics and "wql_macro" not in metrics:
        metrics.append("wql_macro")
    core_metrics = [metric for metric in CORE_METRICS if metric in metrics]

    detailed = successful_rows(pd.read_csv(detailed_path))
    forecasts = pd.read_csv(forecast_path)
    contexts = pd.read_csv(context_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Running stationarity and structural-break tests...")
    stationarity = stationarity_tests(contexts, forecasts, alpha)
    stationarity.to_csv(output_dir / "dataset_stationarity.csv", index=False)

    print("Running Pettitt single-change-point diagnostics...")
    pettitt = pettitt_change_points(
        contexts,
        forecasts,
        alpha,
        pettitt_simulations,
        seed,
    )
    pettitt.to_csv(output_dir / "pettitt_change_points.csv", index=False)

    print("Running residual, bias, and Mann-Kendall diagnostics...")
    residuals, bias, trends = residual_reports(forecasts, alpha)
    residuals.to_csv(output_dir / "residual_diagnostics.csv", index=False)
    bias.to_csv(output_dir / "forecast_bias.csv", index=False)
    trends.to_csv(output_dir / "mann_kendall_trends.csv", index=False)

    print("Running probabilistic calibration diagnostics...")
    calibration = probabilistic_calibration(forecasts, alpha)
    calibration.to_csv(output_dir / "probabilistic_calibration.csv", index=False)

    print("Running SPA winner comparisons and Model Confidence Sets...")
    variability = repetition_variability(detailed, core_metrics, alpha)
    variability.to_csv(output_dir / "repetition_variability.csv", index=False)
    pairwise, mcs = model_comparisons(
        detailed,
        metrics,
        alpha,
        bootstrap_reps,
        seed,
    )
    pairwise = add_repetition_variability_to_pairs(pairwise, variability)
    pairwise.to_csv(output_dir / "winner_vs_rest.csv", index=False)
    mcs.to_csv(output_dir / "model_confidence_set.csv", index=False)

    print("Running exploratory Harvey-corrected Diebold-Mariano comparisons...")
    dm = diebold_mariano_comparisons(detailed, core_metrics, alpha)
    dm = add_repetition_variability_to_pairs(dm, variability)
    dm.to_csv(output_dir / "diebold_mariano_winner_vs_rest.csv", index=False)

    print("Generating hypothesis-test plots...")
    winner_plot_count = save_winner_plots(pairwise, output_dir / "plots")
    mcs_plot_count = save_mcs_membership_plots(mcs, output_dir / "plots")

    counts = {
        "dataset_stationarity.csv": len(stationarity),
        "pettitt_change_points.csv": len(pettitt),
        "residual_diagnostics.csv": len(residuals),
        "forecast_bias.csv": len(bias),
        "mann_kendall_trends.csv": len(trends),
        "probabilistic_calibration.csv": len(calibration),
        "winner_vs_rest.csv": len(pairwise),
        "model_confidence_set.csv": len(mcs),
        "diebold_mariano_winner_vs_rest.csv": len(dm),
        "repetition_variability.csv": len(variability),
        "winner_vs_rest_plots": winner_plot_count,
        "model_confidence_set_plots": mcs_plot_count,
    }
    write_methodology(output_dir, alpha, bootstrap_reps, pettitt_simulations, counts)
    status = {
        "alpha": alpha,
        "bootstrap_reps": bootstrap_reps,
        "pettitt_simulations": pettitt_simulations,
        "seed": seed,
        "input_files": {
            "detailed": str(detailed_path),
            "forecasts": str(forecast_path),
            "contexts": str(context_path),
            "config": str(config_path),
        },
        "library_versions": {
            package: version(package)
            for package in (
                "arch",
                "dieboldmariano",
                "pyhomogeneity",
                "pymannkendall",
                "scipy",
                "statsmodels",
                "pandas",
                "numpy",
            )
        },
        "counts": counts,
    }
    (output_dir / "analysis_status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate PMDS statistical diagnostics and hypothesis tests")
    parser.add_argument(
        "--detailed",
        type=Path,
        default=Path("pmds/results/11_12_3_rep/11_12_3_rep_detailed.csv"),
    )
    parser.add_argument(
        "--forecasts",
        type=Path,
        default=Path("pmds/results/11_12_3_rep/11_12_3_rep_forecasts.csv"),
    )
    parser.add_argument(
        "--contexts",
        type=Path,
        default=Path("pmds/results/11_12_3_rep/11_12_3_rep_contexts.csv"),
    )
    parser.add_argument("--config", type=Path, default=Path("pmds/config.json"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("pmds/results/11_12_3_rep/hypothesis_tests"),
    )
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    parser.add_argument("--bootstrap-reps", type=int, default=DEFAULT_BOOTSTRAP_REPS)
    parser.add_argument(
        "--pettitt-simulations",
        type=int,
        default=DEFAULT_PETTITT_SIMULATIONS,
        help="pyHomogeneity Monte Carlo replications; 0 uses its analytic Pettitt p-value",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()
    for path in (args.detailed, args.forecasts, args.contexts, args.config):
        if not path.exists():
            raise FileNotFoundError(path)
    if not 0.0 < args.alpha < 1.0:
        parser.error("--alpha must be between 0 and 1")
    if args.bootstrap_reps < 100:
        parser.error("--bootstrap-reps must be at least 100")
    if args.pettitt_simulations < 0:
        parser.error("--pettitt-simulations must be nonnegative")
    counts = run_analysis(
        args.detailed,
        args.forecasts,
        args.contexts,
        args.config,
        args.output,
        args.alpha,
        args.bootstrap_reps,
        args.pettitt_simulations,
        args.seed,
    )
    print(f"Reports saved under: {args.output}")
    for name, count in counts.items():
        print(f"  {name}: {count}")


if __name__ == "__main__":
    main()
