# PMDS Forecasting Benchmark and Statistical Analysis Report

**Experiment:** `11_12_3_rep`  
**Final benchmark run:** 1 September 2026, 13:04–16:15 Europe/Berlin  
**Scope:** 11 models × 12 datasets × 3 repetitions  
**Primary significance level:** 0.05  
**Status:** completed with no failed model tasks

## Executive summary

This benchmark compared eleven forecasting models on twelve datasets. Every model–forecast-task pair was evaluated three times. The final dataset contains 396 forecast tasks, 13,068 detailed model runs, 257,202 forecast rows, and no failed runs.

The central result is that **TimesFM 2.5 200M is the best overall model** across the seven core metrics. It wins 40 of 84 dataset–metric combinations and has the best overall mean rank, 2.857. Its strongest results are on M3 Quarterly and NN5 Weekly:

- **M3 Quarterly:** TimesFM ranks first on all seven metrics and is the unique statistically supported winner for MAE, RMSE, and SMAPE.
- **NN5 Weekly:** TimesFM ranks first on all seven metrics, beats every competitor in the Holm-adjusted SPA comparisons on every metric, and is the sole member of the 95% Model Confidence Set for six metrics, including CRPS. Prophet remains alongside it only for MAE.

Other important results are:

- **Chronos T5 Base** ranks first on all seven M4 Hourly metrics, but large Model Confidence Sets for several metrics prevent a universal superiority claim.
- **Chronos T5 Small** ranks first on all seven PSM metrics, but the dataset supplies only 15 paired tasks, so the evidence is exploratory.
- **Moirai 2.0 Small** is the empirical leader for Car Parts and Traffic, although it is not cleanly separated from the other strong models.
- **ARIMA(2,1,2)** is the descriptive overall leader on M1 Yearly, but the winning model changes across metrics and no unique statistical winner is established.
- **Seasonal Naive** is the decisive SMAPE winner on both Car Parts and Weather, illustrating that a simple baseline can remain optimal for a specific error definition.
- National Illness, BLS Macro, Elexon Demand, PSM, Traffic, and USGS have too few independent forecast tasks for strong winner claims.

Only **12 of the 84 core dataset–metric families** satisfy the strictest criterion used in this report: the winner beats all ten competitors with reportable Holm-adjusted SPA evidence and is the only model in the 95% Model Confidence Set. Thirteen families beat all competitors using SPA, but NN5 MAE retains Prophet in its MCS.

## 1. How the final experiment was reached

### 1.1 Initial design

The target design was twelve datasets and twelve configured models, including Chronos T5 Large. Chronos T5 Large was retained in the configuration but disabled for the main benchmark because of its heavier resource requirements. This produced **eleven enabled models** for the final run.

All enabled models were marked for three repetitions. This ensured that every model had three recorded results per task. Marking a deterministic model as repeated does not make it stochastic; deterministic models frequently returned identical values and therefore zero repetition standard deviation.

### 1.2 Auto-ARIMA failure and correction

The first attempt, started at 12:40, used a broad Auto-ARIMA search:

- `max_p = 3`
- `max_d = 2`
- `max_q = 3`
- `max_order = 8`

That search could select unstable high-order models such as `(3,2,3)` on short or degenerate series. Some selected fits subsequently produced non-finite forecasts, which were correctly rejected by output validation. The attempt was stopped.

The safer final search was:

- `max_p = 2`
- `max_d = 1`
- `max_q = 2`
- `max_order = 5`
- non-seasonal search
- AIC model selection

This search still contains ARIMA `(2,1,2)` because `2 + 1 + 2 = 5`. The complete experiment was then rerun under the same `11_12_3_rep` name, replacing the partial outputs. The final run completed all twelve datasets with zero failed model tasks.

### 1.3 Chronos Large attempt

A separate `chronos_large_3_rep` run was started after the main benchmark. It began loading Chronos T5 Large but was stopped before producing result CSVs. It is therefore **not included anywhere in this report**.

## 2. Experiment design

### 2.1 Models

The eleven evaluated models were:

| Category | Models |
|---|---|
| Chronos foundation models | Chronos T5 Tiny, Mini, Small, Base |
| Other foundation models | TimesFM 2.5 200M, Moirai 2.0 Small |
| Statistical baselines | Seasonal Naive, ARIMA(2,1,2), Auto-ARIMA, Prophet |
| Trained neural baseline | DeepAR |

Chronos T5 Large was configured but disabled. Chronos Mini, Small, and Base used CUDA in the resolved experiment configuration; Tiny, Moirai, and DeepAR were configured on CPU. The run used Python 3.12.3, PyTorch 2.4.1+cu121, and an available CUDA GPU.

### 2.2 Datasets and forecast tasks

A forecast task is one series at one forecast origin. Most datasets used three origins; Car Parts used two.

| Dataset | Series | Origins | Horizon | Paired tasks | Inference quality |
|---|---:|---:|---:|---:|---|
| M1 Yearly | 20 | 3 | 6 | 60 | adequate, dependence caveat |
| M4 Hourly | 20 | 3 | 48 | 60 | adequate, dependence caveat |
| Weather/Rain | 20 | 3 | 30 | 60 | adequate, dependence caveat |
| National Illness | 1 | 3 | 24 | 3 | very low power |
| Traffic | 5 | 3 | 24 | 15 | low power |
| PSM | 5 | 3 | 24 | 15 | low power |
| M3 Quarterly | 20 | 3 | 8 | 60 | adequate, dependence caveat |
| Car Parts | 20 | 2 | 12 | 40 | adequate, dependence caveat |
| NN5 Weekly | 20 | 3 | 8 | 60 | adequate, dependence caveat |
| Elexon Demand | 1 | 3 | 30 | 3 | very low power |
| USGS Streamflow | 5 configured, 13 valid tasks | up to 3 | 30 | 13 | low power |
| BLS Macro | 3 configured, 7 valid tasks | up to 3 | 6 | 7 | very low power |

The power labels used by the analysis are deliberately conservative:

- `very_low_power`: fewer than 8 paired tasks or no cross-series replication;
- `low_power`: fewer than 20 paired tasks or fewer than 5 series;
- `adequate_with_dependence_caveat`: at least 20 tasks and at least 5 series.

The last label is not equivalent to independence. Forecast tasks from the same series and neighboring origins may remain dependent.

### 2.3 Repetitions and seeds

Each model was evaluated three times per task. The output contains exactly 4,356 rows for each repetition index, for a total of 13,068 detailed rows. All three repetitions succeeded.

The formal model-comparison tests first average the three repetitions within each forecast task. This prevents stochastic models from contributing three pseudo-independent copies of the same task.

### 2.4 Forecast outputs

- Point forecast: median.
- Quantiles: 0.1 through 0.9 in increments of 0.1.
- Maximum context: 512.
- Forecast origins are separated by the dataset horizon, reducing direct overlap between test blocks.

## 3. Evaluation metrics

All reported metrics are losses: **lower is better**.

| Metric | Meaning | Main sensitivity |
|---|---|---|
| MAE | Mean absolute error | Typical absolute point error |
| RMSE | Root mean squared error | Penalizes large errors more strongly |
| SMAPE | Symmetric mean absolute percentage error | Scale-free relative error; sensitive near zero |
| MASE | Mean absolute scaled error | Error relative to a seasonal-naive scale |
| CRPS | Continuous Ranked Probability Score | Full predictive-distribution accuracy and calibration/sharpness |
| WQL | Micro-weighted quantile loss | Probabilistic accuracy weighted by total target magnitude |
| WQL macro | Mean task-level WQL | Gives each forecast task equal weight |

Rain occurrence error and positive-value MAE were also produced for the zero-inflated Car Parts and Weather datasets. They appear in the descriptive plots and the broader winner table, but the main cross-dataset conclusions use the seven metrics above because those seven are available for every dataset.

CRPS was calculated after the benchmark from the saved q10–q90 forecasts using `scoringrules.crps_quantile`. It is therefore a nine-quantile approximation to CRPS, not the exact integral over a fully persisted forecast CDF or the ensemble-sample estimator. It is unnormalized and has the same units as the target. No model was rerun or changed. CRPS remains defined for all-zero target windows, unlike the normalized WQL used here.

WQL and WQL macro should not be treated as identical. WQL micro-aggregates numerators and denominators across tasks, so large-target tasks have more influence. WQL macro averages task scores, so each task has equal influence.

## 4. Descriptive plots and repetition standard deviation

The plot suite contains:

- per-metric model comparisons;
- normalized all-metric comparisons;
- forecast-versus-actual plots;
- one scatter plot with repetition standard deviation for every dataset and metric;
- winner-versus-rest confidence-interval plots;
- Model Confidence Set heatmaps.

Examples can be found in [`plots/chronos_m3_quarterly`](plots/chronos_m3_quarterly) and [`plots/chronos_nn5_weekly`](plots/chronos_nn5_weekly).

The repetition variability report covers 12 × 11 × 7 = **924 model–dataset–metric combinations**:

- 504 are classified as seed-sensitive;
- 420 are identical across repetitions.

The sample standard deviation is useful for describing random-seed stability, but it is **not a model-superiority test**:

1. Three repetitions give only two degrees of freedom for estimating variance.
2. Repetitions measure seed variation, not variation from selecting different datasets, series, origins, or real-world futures.
3. A deterministic model can have SD = 0 without being accurate.
4. Two non-overlapping mean ± SD bands are descriptive and are not a calibrated 5% hypothesis test.

The SD, standard error, descriptive t interval, coefficient of variation, and rank are recorded in [`hypothesis_tests/repetition_variability.csv`](hypothesis_tests/repetition_variability.csv). The winner and competitor SD fields are also included in [`hypothesis_tests/winner_vs_rest.csv`](hypothesis_tests/winner_vs_rest.csv).

## 5. Statistical methodology

### 5.1 Dataset-level diagnostics

| Test | Null hypothesis | Purpose |
|---|---|---|
| Augmented Dickey–Fuller | Unit root/nonstationarity | Tests whether differencing may be required |
| Phillips–Perron | Unit root/nonstationarity | Unit-root test robust to some serial correlation and heteroskedasticity |
| KPSS | Stationarity | Complements ADF/PP because its null is reversed |
| Zivot–Andrews | Unit root without a structural break | Tests against trend stationarity with one endogenous break |
| Pettitt | Homogeneous distribution/no change point | Detects one abrupt distributional change |

Using both ADF and KPSS gives four possible interpretations: stationary, nonstationary, conflicting/possible structural break, or inconclusive. Pettitt is also run on the absolute forecast errors for every model and series.

### 5.2 Residual diagnostics

| Test | Null hypothesis | Purpose |
|---|---|---|
| Ljung–Box | No residual autocorrelation through the selected lag | Checks remaining temporal structure |
| Box–Pierce | No residual autocorrelation through the selected lag | Classical portmanteau diagnostic |
| ARCH-LM | No autoregressive conditional heteroskedasticity | Checks changing error variance |
| CUSUM | Stable centered residual process | Checks instability/change in residual behavior |
| One-sample t test | Mean signed error is zero | Parametric bias diagnostic |
| Wilcoxon signed-rank | Signed errors are centered around zero | Non-parametric bias diagnostic |

### 5.3 Trend and probabilistic calibration

- Original Mann–Kendall tests signed and absolute errors for monotonic trends.
- Hamed–Rao modified Mann–Kendall adjusts the trend test for autocorrelation.
- Sen's slope describes the direction and magnitude of the trend.
- Quantile and central-interval coverage are compared with their nominal coverage using binomial diagnostics.
- Adjacent quantile crossing rates check whether predicted quantiles remain ordered.

These are model-adequacy diagnostics. They do not by themselves determine which model has the lowest forecast loss.

### 5.4 Winner-versus-rest inference

For every dataset and metric:

1. The three repetitions are averaged within each forecast task.
2. The empirical winner is the model with the lowest mean task loss.
3. The winner is compared separately with each of the other ten models.
4. Hansen's Superior Predictive Ability test uses a stationary bootstrap.
5. A 95% stationary-block-bootstrap confidence interval is calculated for `competitor loss − winner loss`.
6. The ten SPA p-values are Holm-adjusted within that dataset–metric family.
7. A Hansen–Lunde–Nason 95% Model Confidence Set is computed over all eleven models.

The bootstrap uses 5,000 replications and block length `ceil(sqrt(number of paired tasks))`.

`reportable_significance` is true only when the Holm-adjusted pairwise comparison is significant and the family has adequate sample quality. Low-power and very-low-power results remain explicitly exploratory even if their raw or adjusted p-values are small.

The Model Confidence Set is essential because the empirical winner was selected using the same observations later used for comparison. A singleton MCS is stronger evidence than merely having the smallest observed mean.

### 5.5 Diebold–Mariano test

The Harvey–Leybourne–Newbold corrected Diebold–Mariano test was added for the seven core metrics. Its null hypothesis is equal expected predictive loss; the one-sided alternative favors the empirical winner. Holm correction is again applied within each dataset–metric family.

There are 840 possible winner-versus-competitor comparisons:

- 630 were calculated;
- 210 were not run because fewer than eight ordered tasks were available;
- 302 calculated comparisons were exploratory Holm-significant;
- 328 calculated comparisons were not significant.

These DM results are **supplementary, not confirmatory**. Classical DM inference expects a long ordered fixed-horizon loss-differential time series. Here the ordered sequence pools tasks from multiple series and origins. A better confirmatory DM experiment would use many rolling origins and analyze each fixed lead time separately.

### 5.6 Multiple-testing control

- Winner-versus-rest SPA and DM tests: Holm family-wise error correction within each dataset–metric family.
- Dataset, residual, bias, trend, Pettitt, and calibration diagnostics: Benjamini–Hochberg false-discovery-rate correction within their documented dataset/test families.

## 6. Dataset properties and change points

All 140 observed target series produced stationarity diagnostics:

- 63 classified stationary by the joint FDR-adjusted ADF/KPSS interpretation;
- 40 classified nonstationary;
- 25 inconclusive;
- 12 conflicting or suggestive of a structural break.

Pettitt detected an FDR-significant target change point in 88 of 140 series. Structural change is therefore common and should be considered when interpreting both classical forecasting models and residual tests.

| Dataset | Stationary | Nonstationary | Conflict/break | Inconclusive | Pettitt target changes |
|---|---:|---:|---:|---:|---:|
| Car Parts | 15 | 0 | 0 | 5 | 3/20 |
| M1 Yearly | 2 | 12 | 1 | 5 | 15/20 |
| M3 Quarterly | 0 | 17 | 0 | 3 | 20/20 |
| M4 Hourly | 10 | 0 | 0 | 10 | 13/20 |
| NN5 Weekly | 9 | 7 | 4 | 0 | 17/20 |
| Weather | 19 | 0 | 0 | 1 | 7/20 |
| National Illness | 0 | 1 | 0 | 0 | 1/1 |
| PSM | 3 | 0 | 1 | 1 | 2/5 |
| Traffic | 5 | 0 | 0 | 0 | 1/5 |
| BLS Macro | 0 | 2 | 1 | 0 | 3/3 |
| Elexon Demand | 0 | 0 | 1 | 0 | 1/1 |
| USGS Streamflow | 0 | 1 | 4 | 0 | 5/5 |

The Zivot–Andrews test rejected its unit-root null in 89 of 137 valid series. Its interpretation is not the same as Pettitt: Zivot–Andrews tests a unit-root question while allowing one break; Pettitt directly tests homogeneity around a single abrupt change.

## 7. Aggregate adequacy diagnostics

The following are counts of FDR-adjusted rejections. A high rejection count means that the diagnostic assumption often fails; it does not mean that all forecasts are unusable.

| Diagnostic | Rejections | Valid tests | Interpretation of a rejection |
|---|---:|---:|---|
| Ljung–Box | 725 | 1,518 | residual autocorrelation remains |
| Box–Pierce | 693 | 1,518 | residual autocorrelation remains |
| ARCH-LM | 410 | 1,517 | residual variance changes over time |
| CUSUM | 445 | 1,518 | residual behavior is unstable |
| Mean-bias t test | 693 | 1,540 | nonzero mean signed error |
| Wilcoxon bias test | 763 | 1,540 | signed errors not centered on zero |
| Modified Mann–Kendall, signed error | 218 | 1,516 | monotonic signed-error trend |
| Modified Mann–Kendall, absolute error | 204 | 1,511 | monotonic accuracy trend |
| Pettitt, absolute error | 468 | 1,518 | abrupt accuracy change |

The modified Mann–Kendall rejection counts are substantially lower than those of the unmodified test, demonstrating why autocorrelation adjustment matters.

Probabilistic calibration was also challenging:

- 825 of 1,188 quantile-coverage tests rejected after FDR adjustment;
- 412 of 528 central-interval-coverage tests rejected;
- mean absolute quantile-coverage error was 0.144;
- mean absolute central-interval-coverage error was 0.195;
- adjacent quantile crossing was rare, with an average rate of 0.00133.

Thus the predicted quantiles are generally ordered, but their nominal coverage is often inaccurate. Good point accuracy should not be described as calibrated uncertainty without consulting the calibration report.

## 8. Model performance by dataset

The overall leader below is selected using average rank across the seven core metrics, supplemented by the number of metric wins. This avoids averaging values with incompatible units. It is still a descriptive aggregation: several metrics are related, so the seven votes are not independent.

The “strictly supported metrics” column requires both:

1. reportable Holm-adjusted SPA superiority over all ten competitors; and
2. a singleton 95% Model Confidence Set.

| Dataset | Descriptive overall leader | Metric wins | Mean rank | Strictly supported metric winners | Conclusion |
|---|---|---:|---:|---|---|
| Car Parts | Moirai 2.0 Small | 6/7 | 2.43 | SMAPE: Seasonal Naive | No universal winner; Moirai leads most metrics but remains in MCS groups of 7–8 models |
| M1 Yearly | ARIMA(2,1,2) | 4/7 | 2.57 | none | Metric-dependent and statistically inconclusive |
| M3 Quarterly | TimesFM 2.5 200M | 7/7 | 1.00 | MAE, RMSE, SMAPE: TimesFM | Strong overall TimesFM result; MASE, CRPS, and both WQL variants retain alternatives |
| M4 Hourly | Chronos T5 Base | 7/7 | 1.00 | none | Consistent empirical leader; statistical separation varies strongly by metric |
| NN5 Weekly | TimesFM 2.5 200M | 7/7 | 1.00 | CRPS, MASE, RMSE, SMAPE, WQL, WQL macro: TimesFM | Strongest overall result; MAE also beats all competitors but retains Prophet in the MCS |
| Weather | Chronos T5 Base by average rank | 2/7 | 2.86 | CRPS: Moirai; SMAPE: Seasonal Naive | No universal winner; different loss definitions select different models |
| National Illness | TimesFM 2.5 200M | 7/7 | 1.00 | none | Empirical only; 3 tasks and one series |
| PSM | Chronos T5 Small | 7/7 | 1.00 | none | Promising empirical result but low power with 15 tasks |
| Traffic | Moirai 2.0 Small | 6/7 | 1.86 | none | Empirical preference for Moirai; low power |
| BLS Macro | TimesFM 2.5 200M | 4/7 | 2.14 | none | No reliable overall winner with 7 tasks |
| Elexon Demand | TimesFM 2.5 200M | 7/7 | 1.00 | none | Empirical only; 3 tasks and one series |
| USGS Streamflow | TimesFM 2.5 200M | 5/7 | 1.43 | none | TimesFM leads overall; Moirai wins MASE and SMAPE; low power |

### 8.1 Car Parts

Moirai wins MAE, RMSE, MASE, CRPS, WQL, and WQL macro. However, it significantly beats only three or four competitors depending on the metric, and the MCS usually contains seven or eight models. Seasonal Naive is uniquely superior for SMAPE, beating every competitor and forming a singleton MCS. The correct conclusion is metric-specific, not that Moirai dominates every interpretation of error.

### 8.2 M1 Yearly

ARIMA(2,1,2) wins MAE, SMAPE, CRPS, and WQL. TimesFM wins MASE and WQL macro, while DeepAR wins RMSE. MCS groups are large and the pairwise evidence is weak. ARIMA(2,1,2) is the descriptive overall choice, but no model is statistically established as a general winner.

### 8.3 M3 Quarterly

TimesFM ranks first on every metric. It beats all competitors with reportable SPA evidence and has a singleton MCS for MAE, RMSE, and SMAPE. For MASE, CRPS, WQL, and WQL macro it beats nine of ten competitors, but alternatives remain in their MCS groups. This is a strong TimesFM dataset-level result without claiming that every metric separates it from every alternative.

### 8.4 M4 Hourly

Chronos T5 Base ranks first on all seven metrics. Nevertheless, MAE, RMSE, CRPS, and WQL have no reportable pairwise wins and their MCS contains all eleven models. SMAPE provides the strongest evidence: Chronos Base beats nine competitors and shares the MCS only with Chronos Small. The model is highly consistent in rank, but the evidence does not justify a universal superiority statement.

### 8.5 NN5 Weekly

TimesFM ranks first on all seven metrics and significantly beats all ten competitors on all seven. It is the sole MCS member for CRPS, MASE, RMSE, SMAPE, WQL, and WQL macro. The MAE MCS also contains Prophet. This is the cleanest and most defensible winner in the benchmark.

### 8.6 Weather

The best model depends on the metric:

- Moirai: MAE, CRPS, and WQL;
- TimesFM: RMSE;
- Chronos T5 Base: MASE and WQL macro;
- Seasonal Naive: SMAPE.

Seasonal Naive is a strict winner for SMAPE, and Moirai is a strict winner for CRPS. Moirai's WQL result is also strong—it beats nine competitors—but TimesFM remains in the MCS. Chronos Base has the best average rank, but the dataset does not support one universal winner.

### 8.7 Low-power datasets

- National Illness: TimesFM wins all seven observed metrics, but there are only three tasks from one series.
- PSM: Chronos T5 Small wins all seven, but fifteen tasks are insufficient for reportable inference under the stated safeguards.
- Traffic: Moirai wins six metrics and Chronos Base wins SMAPE; fifteen tasks leave the comparison underpowered.
- BLS Macro: TimesFM wins MAE, RMSE, CRPS, and WQL; Auto-ARIMA wins MASE and SMAPE; Chronos Base wins WQL macro. Seven tasks cannot distinguish these reliably.
- Elexon: TimesFM wins all seven, but the evidence consists of three origins from one series.
- USGS: TimesFM wins MAE, RMSE, CRPS, WQL, and WQL macro; Moirai wins MASE and SMAPE. Thirteen valid tasks give low power.

These observed winners are useful candidates for follow-up experiments, not confirmatory conclusions.

## 9. Overall model ranking

The following ranking treats every dataset–metric family equally and averages the empirical ranks. MCS inclusion counts are descriptive only: an underpowered dataset often includes many models, so more inclusions do not necessarily mean better performance.

| Model | Mean rank | Median rank | Metric wins | MCS inclusions out of 84 |
|---|---:|---:|---:|---:|
| TimesFM 2.5 200M | 2.857 | 2 | 40 | 74 |
| Chronos T5 Base | 4.107 | 3 | 11 | 66 |
| Chronos T5 Small | 4.464 | 4 | 7 | 50 |
| Chronos T5 Mini | 4.857 | 5 | 0 | 53 |
| Moirai 2.0 Small | 5.048 | 5 | 17 | 54 |
| Chronos T5 Tiny | 5.345 | 6 | 0 | 45 |
| ARIMA(2,1,2) | 7.357 | 8 | 4 | 32 |
| DeepAR | 7.536 | 8 | 1 | 31 |
| Auto-ARIMA | 7.571 | 8 | 2 | 28 |
| Seasonal Naive | 8.095 | 8 | 2 | 34 |
| Prophet | 8.762 | 10 | 0 | 25 |

TimesFM is the most consistently strong model. Moirai has more wins than several higher-ranked models because it behaves like a specialist: it wins particular datasets but ranks lower elsewhere. Chronos Mini never ranks first but maintains a better average rank than Moirai, reflecting consistency without outright wins.

## 10. Adequacy of the selected dataset leaders

Relative performance and model adequacy answer different questions. A model can be the best among those tested while still leaving autocorrelation, bias, instability, or calibration error in its residuals.

The table below shows FDR-adjusted rejection counts for each descriptive leader. Quantile calibration is counted across the nine nominal quantile levels.

| Dataset leader | Ljung–Box | Wilcoxon bias | Pettitt error change | Rejected quantile levels |
|---|---:|---:|---:|---:|
| Car Parts — Moirai | 2/20 | 8/20 | 9/20 | 9/9 |
| M1 — ARIMA(2,1,2) | 7/20 | 4/20 | 0/20 | 6/9 |
| M3 — TimesFM | 17/20 | 12/20 | 1/20 | 9/9 |
| M4 — Chronos Base | 19/20 | 8/20 | 15/20 | 7/9 |
| NN5 — TimesFM | 0/20 | 3/20 | 0/20 | 6/9 |
| Weather — Chronos Base | 3/20 | 13/20 | 10/20 | 8/9 |
| National Illness — TimesFM | 1/1 | 0/1 | 1/1 | 0/9 |
| PSM — Chronos Small | 5/5 | 1/5 | 2/5 | 3/9 |
| Traffic — Moirai | 4/5 | 3/5 | 4/5 | 5/9 |
| BLS — TimesFM | 2/3 | 2/3 | 0/3 | 0/9 |
| Elexon — TimesFM | 1/1 | 0/1 | 0/1 | 0/9 |
| USGS — TimesFM | 4/5 | 3/5 | 4/5 | 2/9 |

Two examples illustrate the distinction:

- TimesFM is the clear M3 accuracy winner, yet 17/20 M3 residual series retain significant autocorrelation and all nine quantile levels fail the calibration diagnostic.
- TimesFM is also the clear NN5 winner, but here none of the 20 residual series rejects the Ljung–Box test and no series shows a Pettitt change in absolute error. NN5 therefore provides both strong comparative evidence and much cleaner residual behavior.

## 11. What can and cannot be claimed

### Defensible claims

- TimesFM is the best overall model under equal weighting of the twelve datasets and seven core metrics.
- TimesFM is a strong general winner for M3 Quarterly and an especially strong winner for NN5 Weekly.
- Chronos T5 Base is the consistent empirical leader for M4 Hourly, although not uniquely superior on every metric.
- Several datasets are inherently metric-dependent; Car Parts, M1, Weather, BLS, and USGS should not be summarized with one winner without specifying the target metric.
- Repetition SD reveals seed sensitivity, while task-level SPA/MCS inference addresses model-loss differences across forecast tasks.
- Structural change, residual dependence, bias, and quantile miscalibration are common in the benchmark.

### Claims that are not supported

- A zero repetition SD does not prove that a model is statistically superior.
- A raw empirical win on a dataset with 3–15 tasks is not confirmatory evidence.
- Pettitt, ADF, KPSS, Box–Pierce, Ljung–Box, or Mann–Kendall cannot identify the best model by themselves.
- Exploratory DM significance from the pooled panel-task sequence should not be presented as classical fixed-horizon DM evidence.
- The best point-forecast model should not automatically be called the best probabilistic model; CRPS, WQL, and calibration must also be considered.
- Results from Chronos T5 Large cannot be inferred because that separate run was stopped before generating outputs.

## 12. Recommended next experiment

The most important limitation is the number of forecast origins, not the number of random seeds. For stronger publication-quality inference:

1. Increase to at least 20–30 rolling origins per dataset where history permits.
2. Keep the same origin timestamps for every model.
3. Preserve fixed forecast horizons and, for DM, analyze each lead time separately or use a defensible multivariate DM procedure.
4. Continue to use 3–5 model seeds for stochastic stability, but do not treat seeds as substitutes for out-of-sample forecast origins.
5. Predefine one primary metric or a formal metric hierarchy to avoid choosing a favorable metric after observing results.
6. Report MCS membership and pairwise adjusted confidence intervals, not only winner ranks.
7. Investigate calibration methods for models with strong point accuracy but poor interval coverage.
8. If hardware permits, rerun Chronos T5 Large as a separate complete benchmark and merge it only after all dataset–task pairs succeed.

## 13. Reproducibility and output index

### Benchmark inputs and raw outputs

- Configuration used by the experiment: [`../../config_all_models.json`](../../config_all_models.json)
- Final status: [`11_12_3_rep_status.json`](11_12_3_rep_status.json)
- Aggregated model metrics: [`11_12_3_rep_summary.csv`](11_12_3_rep_summary.csv)
- Per-task/per-repetition results: [`11_12_3_rep_detailed.csv`](11_12_3_rep_detailed.csv)
- Forecast values and quantiles: [`11_12_3_rep_forecasts.csv`](11_12_3_rep_forecasts.csv)
- Forecast contexts: [`11_12_3_rep_contexts.csv`](11_12_3_rep_contexts.csv)
- Completed-run logs: [`logs`](logs)

### Statistical outputs

- Methodology summary: [`hypothesis_tests/README.md`](hypothesis_tests/README.md)
- Dataset stationarity: [`hypothesis_tests/dataset_stationarity.csv`](hypothesis_tests/dataset_stationarity.csv)
- Pettitt change points: [`hypothesis_tests/pettitt_change_points.csv`](hypothesis_tests/pettitt_change_points.csv)
- Residual diagnostics: [`hypothesis_tests/residual_diagnostics.csv`](hypothesis_tests/residual_diagnostics.csv)
- Forecast bias: [`hypothesis_tests/forecast_bias.csv`](hypothesis_tests/forecast_bias.csv)
- Mann–Kendall trends: [`hypothesis_tests/mann_kendall_trends.csv`](hypothesis_tests/mann_kendall_trends.csv)
- Probabilistic calibration: [`hypothesis_tests/probabilistic_calibration.csv`](hypothesis_tests/probabilistic_calibration.csv)
- SPA winner comparisons and block-bootstrap intervals: [`hypothesis_tests/winner_vs_rest.csv`](hypothesis_tests/winner_vs_rest.csv)
- Model Confidence Sets: [`hypothesis_tests/model_confidence_set.csv`](hypothesis_tests/model_confidence_set.csv)
- Diebold–Mariano comparisons: [`hypothesis_tests/diebold_mariano_winner_vs_rest.csv`](hypothesis_tests/diebold_mariano_winner_vs_rest.csv)
- Repetition variability: [`hypothesis_tests/repetition_variability.csv`](hypothesis_tests/repetition_variability.csv)
- Winner/MCS hypothesis plots: [`hypothesis_tests/plots`](hypothesis_tests/plots)
- Descriptive and SD plots: [`plots`](plots)

### Record counts

| Output | Records |
|---|---:|
| Detailed model runs | 13,068 |
| Aggregated dataset–model rows | 132 |
| Forecast rows | 257,202 |
| Context rows | 33,013 |
| Dataset stationarity rows | 140 |
| Pettitt rows | 1,680 |
| Residual diagnostic rows | 1,540 |
| Bias rows | 1,540 |
| Mann–Kendall rows | 3,080 |
| Calibration rows | 1,848 |
| Winner-versus-rest rows, all available metrics | 880 |
| Model Confidence Set rows, all available metrics | 968 |
| Core-metric DM rows | 840 |
| Core-metric repetition-variability rows | 924 |
| PNG plots | 456 |

## 14. Supervisor-ready conclusion

> Eleven forecasting models were evaluated on twelve datasets using three repetitions, seven common accuracy and probabilistic metrics, probabilistic calibration diagnostics, and task-level statistical comparisons. CRPS was approximated from the persisted q10–q90 forecasts using the established quantile-score implementation. TimesFM 2.5 200M achieved the best overall rank and the most metric wins. Its superiority is most convincing on M3 Quarterly and NN5 Weekly, with NN5 providing the strongest combination of pairwise significance, singleton Model Confidence Sets, and clean residual behavior. Moirai is the strict CRPS winner on Weather. Chronos T5 Base was the consistent empirical leader on M4 Hourly, while Chronos T5 Small led PSM and Moirai led Car Parts and Traffic. However, several datasets had only 3–15 paired forecast tasks, so their observed winners remain exploratory. Random-seed standard deviations describe stability but cannot establish winner significance; the primary inference is based on Holm-adjusted SPA comparisons, stationary-block-bootstrap confidence intervals, and Model Confidence Sets. Dataset nonstationarity, structural breaks, residual dependence, and poor quantile calibration were common, showing that relative accuracy leadership does not imply complete model adequacy.

## References for the statistical methods

- Dickey, D. A., and Fuller, W. A. Augmented Dickey–Fuller unit-root testing.
- Kwiatkowski, D., Phillips, P. C. B., Schmidt, P., and Shin, Y. KPSS stationarity testing.
- Phillips, P. C. B., and Perron, P. Phillips–Perron unit-root testing.
- Zivot, E., and Andrews, D. W. K. Unit-root testing with an endogenous structural break.
- Pettitt, A. N. (1979). A non-parametric approach to the change-point problem.
- Box, G. E. P., and Pierce, D. A.; Ljung, G. M., and Box, G. E. P. Portmanteau residual tests.
- Mann, H. B.; Kendall, M. G.; Hamed, K. H., and Rao, A. R. Mann–Kendall trend testing and autocorrelation correction.
- Diebold, F. X., and Mariano, R. S. (1995). Comparing predictive accuracy.
- Harvey, D., Leybourne, S., and Newbold, P. (1997). Small-sample correction for predictive-accuracy testing.
- Hansen, P. R. Superior Predictive Ability testing.
- Hansen, P. R., Lunde, A., and Nason, J. M. Model Confidence Sets.
- Berrisch, J., and Ziel, F. (2023). CRPS learning and quantile-score approximation.
