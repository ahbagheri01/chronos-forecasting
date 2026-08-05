# PMDS Benchmark Results

This directory contains benchmark outputs produced by `pmds/compare.py`. All experiment settings are read from `pmds/config.json`.

## Current Result Status

The root `compare_*.csv` files are the historical single-origin run. They are retained for
traceability, but they predate the corrected PSM time scale, rolling origins, representative
selection, repeated stochastic runs, and weather diagnostics. Do not present their rankings as
results from the repaired configuration.

`forecast_audit/` is a completed visual smoke run using one representative series, the latest
origin, and one stochastic repetition per dataset. It completed all 55 dataset/model pairs with
zero errors and supplies the current forecast-versus-actual plots. Its scores are diagnostic and
are not a replacement for the full benchmark.

## Running the Benchmark

From the repository root:

```bash
python pmds/compare.py --config pmds/config.json
```

For the bounded visual audit:

```bash
python pmds/compare.py --config pmds/config.json --forecast-audit
```

The configured run name is `compare`, so the runner writes:

- `compare_detailed.csv`: one row per dataset, series, rolling origin, model, and repetition.
- `compare_summary.csv`: dataset-level model metrics.
- `compare_forecasts.csv`: timestamped actuals, point forecasts, means, and every configured quantile.
- `compare_contexts.csv`: recent pre-cutoff history used by forecast-versus-actual plots.
- `compare_status.json`: completion/failure status for every processed dataset.
- `logs/compare_<timestamp>.log`: full rotating log with stack traces and timing.

The CSV and status files are atomically overwritten after each dataset finishes. They therefore remain readable while a long benchmark is still running and contain all datasets completed so far.

## Configuration

`pmds/config.json` is the source of truth for:

- output and logging paths;
- random seed;
- metrics, point-forecast selection, and quantile levels;
- rolling origins and repeated stochastic runs;
- enabled datasets, horizons, seasonalities, timestamp handling, and maximum series counts;
- optional source-row filtering; weather is explicitly restricted to the `rain` subset;
- deterministic first, evenly-spaced, or seeded-random series selection;
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
- `weather_zero_baseline`: weather-only diagnostic that predicts zero at every horizon step.
- `ar_2`: AR(2), represented by ARIMA order `(2, 0, 0)`.
- `ma_2`: MA(2), represented by ARIMA order `(0, 0, 2)`.
- `arma_2_2`: ARMA(2,2), represented by ARIMA order `(2, 0, 2)`.
- `arima_2_1_2`: ARIMA order `(2, 1, 2)`.
- `auto_arima`: statsmodels-based ARIMA order search using the configured information criterion.
- `sarima_1_0_1`: configurable seasonal ARIMA example; disabled by default because it is slower.
- `prophet`: Prophet with configurable priors and predictive uncertainty samples.
- `deepar`: GluonTS Torch DeepAR trained independently for each forecast task.

All statistical orders and fitting options are JSON hyperparameters. Additional fixed ARIMA-family configurations can be added by copying a `statsmodels_arima` model entry and changing `order` or `seasonal_order`. AutoARIMA searches the configured order ranges and can use `selection_max_samples` to select an order on a tail window before refitting on the full context.

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

The summary also reports `wql_macro`, the unweighted mean of per-task WQL values, and
`wql_median`. These expose cases where one large series dominates official WQL.

Quantile generation differs by model:

- Chronos uses its native sampled quantile forecasts.
- ARIMA-family models use the fitted forecast mean and standard error under a Gaussian approximation.
- AutoARIMA uses the selected statsmodels ARIMA forecast mean and standard error under the same Gaussian approximation.
- Prophet uses predictive samples.
- DeepAR uses its native sample distribution.
- Seasonal naive is deterministic, so all requested quantiles equal its point forecast. Its WQL is valid as a deterministic baseline but does not represent calibrated uncertainty.

### Zero-Inflated Weather Diagnostics

Weather additionally reports:

- `rain_occurrence_error`: fraction of horizon steps where rain/no-rain classification is wrong;
- `positive_mae`: MAE only at steps with observed rainfall above the configured threshold;
- actual and predicted zero fractions in the detailed CSV.

These diagnostics prevent a nearly all-zero forecast from being selected only because ordinary
MAE or MASE is favorable on sparse rainfall.

## Forecast-Versus-Actual Plots

After a benchmark run, generate metric and trajectory plots with:

```bash
python pmds/plot_results.py \
  --csv pmds/results/compare_detailed.csv \
  --forecasts pmds/results/compare_forecasts.csv \
  --contexts pmds/results/compare_contexts.csv \
  --output pmds/results/plots \
  --config pmds/config.json
```

Each dataset/model combination receives one multi-panel image under:

```text
pmds/results/plots/<dataset>/forecast_vs_actual/<model>.png
```

Each panel shows recent history, the actual holdout, the point forecast, the outer configured
quantile interval, and the forecast cutoff. Multiple stochastic repetitions are aggregated by
their median for display. Metric bar plots use official WQL aggregation, while the comparison
views use within-metric ranks instead of outlier-sensitive min-max scaling.

To rebuild only the 55 visual-audit trajectories without replacing historical metric plots:

```bash
python pmds/plot_results.py \
  --csv pmds/results/forecast_audit/forecast_audit_detailed.csv \
  --forecasts pmds/results/forecast_audit/forecast_audit_forecasts.csv \
  --contexts pmds/results/forecast_audit/forecast_audit_contexts.csv \
  --output pmds/results/plots \
  --config pmds/config.json \
  --forecast-only
```

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

Running the full command overwrites the root `compare_*.csv` files incrementally with the new
schema. The historical rankings must be revisited after that run. In particular, the completed
visual audit confirms that Chronos can collapse to an all-zero rainfall median; on its sampled
window its WQL is `1.146`, worse than the explicit zero baseline's `1.000`.
