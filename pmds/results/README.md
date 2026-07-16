# PMDS Benchmark Results

This folder contains the outputs from `pmds/compare.py`.

Current files:

- `compare_summary.csv`: average metrics per dataset and model.
- `compare_detailed.csv`: per-series metrics and any model errors.

## Experiment Setup

The comparison uses the smallest original Chronos model:

```text
amazon/chronos-t5-tiny
```

The script is configured to compare Chronos against:

- `seasonal_naive`: repeats the last observed seasonal pattern.
- `arma`: ARMA(2,2), implemented as `statsmodels` `ARIMA(order=(2, 0, 2))`.
- `prophet`: Prophet with default configuration.
- `deepar`: GluonTS Torch DeepAR, trained per task for a small number of epochs.

The Chronos forecast call requests these quantiles:

```text
[0.1, 0.5, 0.9]
```

For this benchmark, all models are evaluated as point forecasters. Chronos point forecasts are taken from the mean returned by `predict_quantiles`.

## Dataset Configuration

The benchmark is designed for three datasets from the Chronos repository and three external datasets.

Chronos datasets:

| Dataset | Source config | Prediction length | Seasonality | Max series |
|---|---:|---:|---:|---:|
| `chronos_m1_yearly` | `autogluon/chronos_datasets:monash_m1_yearly` | 6 | 1 | 20 |
| `chronos_m4_hourly` | `autogluon/chronos_datasets:m4_hourly` | 48 | 24 | 20 |
| `chronos_weather` | `autogluon/chronos_datasets:monash_weather` | 30 | 7 | 20 |

External datasets:

| Dataset | Source config | Prediction length | Seasonality | Max series |
|---|---:|---:|---:|---:|
| `external_national_illness` | `thuml/Time-Series-Library:national_illness` | 24 | 52 | 1 |
| `external_traffic` | `thuml/Time-Series-Library:traffic` | 24 | 24 | 5 |
| `external_psm` | `thuml/Time-Series-Library:PSM-data` | 24 | 24 | 5 |

The current saved CSVs contain results for:

- `chronos_m1_yearly`
- `chronos_m4_hourly`
- `chronos_weather`
- `external_national_illness`
- `external_psm`

`external_traffic` is configured in `compare.py` for future full runs, but it is not present in the current saved summary.

## Metrics

Lower is better for all metrics.

### MAE

Mean Absolute Error:

```text
mean(abs(y_true - y_pred))
```

MAE is easy to interpret because it is in the same unit as the target. It is scale-dependent, so values cannot be compared fairly across datasets with different units.

### RMSE

Root Mean Squared Error:

```text
sqrt(mean((y_true - y_pred)^2))
```

RMSE penalizes large errors more strongly than MAE. Like MAE, it is scale-dependent.

### SMAPE

Symmetric Mean Absolute Percentage Error:

```text
mean(2 * abs(y_true - y_pred) / (abs(y_true) + abs(y_pred)))
```

SMAPE is scale-normalized, but it can behave strangely when targets or predictions are close to zero.

### MASE

Mean Absolute Scaled Error:

```text
MAE(model) / MAE(seasonal naive in-sample differences)
```

MASE is scale-normalized and is useful across datasets. A value below `1.0` usually means the model beats the seasonal naive scale.

## Are These Chronos Paper Metrics?

Partially.

The Chronos evaluation code and paper emphasize:

- `MASE`
- `WQL` / weighted quantile loss

This benchmark currently includes `MASE`, so one of the main Chronos metrics is covered.

This benchmark does **not** currently compute `WQL`. The reason is that the comparison is written as a point-forecast benchmark across Chronos, ARMA, Prophet, seasonal naive, and DeepAR. Chronos produces quantiles, but the classical models here mostly return point forecasts. To add paper-style probabilistic evaluation, each model should expose quantiles or samples, then `WQL` can be computed consistently.

## Current Result Summary

Based on `compare_summary.csv`:

### `chronos_m1_yearly`

Prophet is strongest overall on this dataset:

- best MAE
- best RMSE
- best SMAPE
- best MASE

Chronos is competitive with seasonal naive and ARMA, but Prophet clearly wins for this yearly dataset in the current run.

### `chronos_m4_hourly`

Chronos is strong on normalized metrics:

- best SMAPE
- best MASE

Seasonal naive has slightly better MAE and RMSE. This means Chronos has better scaled/relative behavior, while seasonal naive has lower raw absolute error on this sample.

### `chronos_weather`

Seasonal naive is best across all listed metrics. Chronos is second on MAE, RMSE, and MASE, but has poor SMAPE. This may be because weather targets can contain values close to zero, where SMAPE becomes unstable.

### `external_national_illness`

Prophet performs best:

- best MAE
- best RMSE
- best SMAPE
- best MASE

Chronos is second and beats seasonal naive and ARMA by a large margin. This is still useful evidence because `national_illness` is external to the Chronos repository.

### `external_psm`

The metrics disagree:

- Prophet has extremely low MAE/RMSE.
- Seasonal naive has the best MASE.
- SMAPE values are very small for seasonal naive and ARMA because the scale of this dataset makes percentage-style interpretation tricky.

This dataset should be interpreted carefully. PSM is anomaly/sensor data, and plain forecasting metrics may not fully reflect the intended task.

## DeepAR Status

DeepAR appears as `NaN` in the current summary because it failed for the detailed tasks rather than producing valid forecasts.

The detailed CSV shows errors such as:

```text
Invalid frequency: YE-DEC
```

So the current DeepAR rows should be treated as failed runs, not model performance.

## Conclusion

The benchmark is working end-to-end across Chronos datasets and external datasets. The result is already useful for comparison:

- Chronos tiny is competitive and often beats ARMA and seasonal naive.
- Prophet is very strong on yearly and illness-style datasets.
- Seasonal naive remains a strong baseline on weather and some sensor-style data.
- ARMA is generally weaker in this run.
- DeepAR needs a frequency-handling fix before its results can be interpreted.

For the next comparison iteration, the most important improvements are:

- Add `WQL` for paper-style probabilistic evaluation.
- Fix DeepAR frequency conversion.
- Run the external traffic dataset.
- Add the proposed decoder-only model as another runner using the same `ForecastTask -> predictions` interface.
