# PMDS statistical diagnostics and model-comparison results

These reports use the completed `11_12_3_rep` benchmark outputs. No forecasting models are rerun.

## Established implementations

- `statsmodels`: ADF, KPSS, Zivot-Andrews, Box-Pierce, Ljung-Box, ARCH-LM, CUSUM, and p-value correction.
- `arch`: Phillips-Perron, Hansen SPA, Hansen-Lunde-Nason Model Confidence Set, and stationary bootstrap confidence intervals.
- `dieboldmariano`: Diebold-Mariano test with the Harvey-Leybourne-Newbold small-sample correction.
- `pyHomogeneity`: Pettitt's single-change-point test.
- `pymannkendall`: original and Hamed-Rao modified Mann-Kendall tests.
- `scoringrules`: finite-quantile CRPS approximation from the saved q10-q90 forecasts.
- `scipy`: signed-error bias tests and binomial calibration diagnostics.

## Settings

- Significance level: `0.05`.
- Bootstrap replications: `5000`.
- Pettitt Monte Carlo simulations: `0` (`0` selects pyHomogeneity's analytic approximation).
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
- `repetition_paired_tests_core_metrics.csv` adds an exploratory seed-stability check for MASE, WQL, and CRPS only. It compares each empirical winner with the best model from the opposite family using a one-sided paired t-test across the three matched repetitions, followed by one Holm correction across all finite tests. A constant positive paired gap is labelled `seed_invariant_gap`; no artificial variance or p-value is assigned.
- `very_low_power` and `low_power` rows must not be presented as strong evidence even when a p-value is small.
- `reportable_significance` is true only for Holm-significant rows classified as `adequate_with_dependence_caveat`.
- Calibration binomial p-values are diagnostic because forecast observations are serially dependent.

## Generated records

- `dataset_stationarity.csv`: 140 rows
- `pettitt_change_points.csv`: 1680 rows
- `residual_diagnostics.csv`: 1540 rows
- `forecast_bias.csv`: 1540 rows
- `mann_kendall_trends.csv`: 3080 rows
- `probabilistic_calibration.csv`: 1848 rows
- `winner_vs_rest.csv`: 880 rows
- `model_confidence_set.csv`: 968 rows
- `diebold_mariano_winner_vs_rest.csv`: 840 rows
- `repetition_variability.csv`: 924 rows
- `repetition_paired_tests_core_metrics.csv`: 36 rows
- `winner_vs_rest_plots`: 88 rows
- `model_confidence_set_plots`: 12 rows
