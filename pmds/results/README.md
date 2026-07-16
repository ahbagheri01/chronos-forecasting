# PMDS Benchmark Results

This directory contains benchmark outputs produced by `pmds/compare.py`. All experiment settings are read from `pmds/config.json`.

## Running the Benchmark

From the repository root:

```bash
python pmds/compare.py --config pmds/config.json
```

The configured run name is `compare`, so the runner writes:

- `compare_detailed.csv`: one row per dataset, series, and model.
- `compare_summary.csv`: dataset-level model metrics.
- `compare_status.json`: completion/failure status for every processed dataset.
- `logs/compare_<timestamp>.log`: full rotating log with stack traces and timing.

The CSV and status files are atomically overwritten after each dataset finishes. They therefore remain readable while a long benchmark is still running and contain all datasets completed so far.

## Configuration

`pmds/config.json` is the source of truth for:

- output and logging paths;
- random seed;
- metrics, point-forecast selection, and quantile levels;
- enabled datasets, horizons, seasonalities, and maximum series counts;
- Chronos model, device, dtype, sampling temperature, top-k, and top-p;
- AR, MA, ARMA, ARIMA, and SARIMA orders;
- Prophet priors and uncertainty samples;
- DeepAR architecture, optimization, sampling, and trainer settings.

Models and datasets have an `enabled` field. Set it to `false` to remove that entry from a run without editing Python code.

The default quantiles match the Chronos evaluation grid:

```text
[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
```

The default point forecast is the `0.5` quantile (median), matching the point forecast used for MASE in the Chronos evaluation path.

## Models

The current configuration includes:

- `chronos_t5_tiny`: `amazon/chronos-t5-tiny`.
- `seasonal_naive`: repeats the last dataset-specific seasonal pattern.
- `ar_2`: AR(2), represented by ARIMA order `(2, 0, 0)`.
- `ma_2`: MA(2), represented by ARIMA order `(0, 0, 2)`.
- `arma_2_2`: ARMA(2,2), represented by ARIMA order `(2, 0, 2)`.
- `arima_2_1_2`: ARIMA order `(2, 1, 2)`.
- `sarima_1_0_1`: configurable seasonal ARIMA example; disabled by default because it is slower.
- `prophet`: Prophet with configurable priors and predictive uncertainty samples.
- `deepar`: GluonTS Torch DeepAR trained independently for each forecast task.

All statistical orders and fitting options are JSON hyperparameters. Additional ARIMA-family configurations can be added by copying a `statsmodels_arima` model entry and changing `order` or `seasonal_order`.

## Metrics

Lower is better for all metrics.

### MAE

Mean Absolute Error:

```text
mean(abs(y_true - y_pred))
```

MAE is expressed in the target's original units and is scale-dependent.

### RMSE

Root Mean Squared Error:

```text
sqrt(mean((y_true - y_pred)^2))
```

RMSE penalizes large errors more strongly than MAE and is also scale-dependent.

### SMAPE

Symmetric Mean Absolute Percentage Error:

```text
mean(2 * abs(y_true - y_pred) / (abs(y_true) + abs(y_pred)))
```

SMAPE is scale-normalized but can be unstable when values are near zero.

### MASE

Mean Absolute Scaled Error:

```text
MAE(model) / mean(abs(y[t] - y[t - seasonality]))
```

MASE is scale-normalized. Values below `1.0` indicate performance better than the in-sample seasonal-naive scale.

### WQL

Mean Weighted Sum Quantile Loss is the probabilistic metric used by the Chronos benchmark. For each quantile `q`, the runner computes twice the pinball loss, sums it over the forecast horizon, and normalizes it by the sum of absolute target values. Dataset-level WQL is:

```text
mean_q(total_quantile_loss[q] / total_abs_target)
```

The summary combines loss numerators and target denominators across all series before taking the ratio. This matches the aggregation semantics of GluonTS `MeanWeightedSumQuantileLoss`, which is used by the Chronos evaluation code.

Quantile generation differs by model:

- Chronos uses its native sampled quantile forecasts.
- ARIMA-family models use the fitted forecast mean and standard error under a Gaussian approximation.
- Prophet uses predictive samples.
- DeepAR uses its native sample distribution.
- Seasonal naive is deterministic, so all requested quantiles equal its point forecast. Its WQL is valid as a deterministic baseline but does not represent calibrated uncertainty.

## Chronos Paper Alignment

The Chronos benchmark emphasizes:

- `MASE` for point forecasts;
- `WQL` for probabilistic forecasts.

Both are now included. MAE, RMSE, and SMAPE remain as supplementary diagnostics.

## Failure Diagnosis

For a model or dataset failure, inspect these in order:

1. `compare_status.json` to identify failed datasets.
2. `compare_detailed.csv`, especially the `error_type` and `error` columns, to identify failed model tasks.
3. The latest file under `logs/` for timestamps, model duration, warnings, and full Python stack traces.

One model failure does not stop other models or datasets. Dataset loading failures are also recorded in the CSV/status output and logged with stack traces.

## Existing Results

Any CSV files created before this config-driven version do not contain WQL. Running the new command overwrites `compare_detailed.csv` and `compare_summary.csv` incrementally with the new schema.

The previous run suggested that Prophet was strongest on M1 yearly and national illness, Chronos was competitive on M4 hourly, and seasonal naive was strong on weather. Those conclusions should be revisited after the new WQL-enabled run, especially because probabilistic quality can rank models differently from point-error metrics.
