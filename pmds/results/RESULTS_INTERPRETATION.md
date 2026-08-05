# PMDS Results Interpretation - Reviewed

Reviewed on 2026-08-03 from the current benchmark artifacts and scoring code.

> **Result provenance, updated 2026-08-04:** The numerical rankings below describe the
> historical `compare_*.csv` run. They do not represent the repaired configuration. PSM timestamp
> handling, rainfall scoping and diagnostics, rolling origins, representative sampling, repeated
> stochastic runs, official WQL plotting, and forecast persistence are now implemented. A full
> repaired benchmark rerun is still required before replacing the historical conclusions.

## Repair Verification

| Issue | Implemented correction | Verification |
| --- | --- | --- |
| PSM minute/hour mismatch | Parse `timestamp_(min)` and resample to hourly means; use horizon/lag 24 | Real audit plot spans 24 hourly steps |
| Weather target ambiguity | Explicitly filter `monash_weather` to `subset=rain` | 729 eligible rainfall rows; sampled holdout is 86.7% zero |
| Zero-like Chronos forecast | Add a zero baseline, occurrence error, positive-only MAE, and zero fractions | Audit Chronos predicts zero at 100% of steps; WQL 1.146 vs zero baseline 1.000 |
| One cutoff | Use three rolling origins per series | Task construction tests pass; full rerun pending |
| First-series bias | Select evenly spaced representative series/features | Deterministic selection tests pass |
| Stochastic uncertainty | Run Chronos, Prophet, and DeepAR with three deterministic seeds | Seed/repetition tests pass; full rerun pending |
| WQL plot mismatch | Aggregate WQL from total loss and target sums | Plot values match summary aggregation numerically |
| No trajectory audit | Persist contexts/forecasts and plot actual, point, and 10--90% interval | 55 dataset/model plots generated; zero failed audit rows |

The bounded `forecast_audit/` run uses one representative series, one latest origin, and one seed.
It validates data flow and plots, but its metric values are not full-benchmark rankings.

Authoritative inputs for this report are `compare_summary.csv`, `compare_detailed.csv`,
`compare_status.json`, `pmds/config.json`, and the metric/model implementation. The older
`all_summary.csv` and `all_detailed.csv` do not contain WQL and are not mixed into this
analysis. `MODEL_PERFORMANCE_ANALYSIS.md` and `plots/REPORT.md` describe an older July run;
their rankings and some metric descriptions are stale.

## Bottom Line

| Dataset | Point-forecast result | Probabilistic result | Reviewed conclusion |
| --- | --- | --- | --- |
| `chronos_m1_yearly` | DeepAR wins scale-sensitive MAE/RMSE; ARIMA(2,1,2) wins SMAPE/MASE | DeepAR wins official WQL | Objective-dependent. DeepAR's aggregate lead is heavily driven by the largest series; ARIMA is stronger on normalized errors. |
| `chronos_m4_hourly` | Seasonal naive wins MAE/RMSE; Chronos wins SMAPE/MASE | Chronos wins WQL | Chronos is the quality winner across more series; seasonal naive is the best speed/raw-error baseline. |
| `chronos_weather` | Chronos appears first on MAE/MASE, but exactly reproduces an all-zero point baseline; DeepAR wins RMSE | Chronos WQL is 0.984, only 1.6% better than an all-zero WQL of 1.0 | No clean model winner. The target is zero-inflated rainfall and the current metrics reward trivial zero forecasts. |
| `external_national_illness` | ARIMA(2,1,2) wins all four point metrics | Prophet wins WQL | ARIMA for point forecasts and Prophet for quantiles, but this is one series and one holdout window. |
| `external_psm` | ARIMA(2,1,2) wins all four point metrics | Chronos wins WQL | Numerically clear, but the configured time scale is wrong for the source data, so do not call this a 24-hour result. |
| `external_traffic` | DeepAR wins MAE/RMSE/MASE; Chronos wins SMAPE | DeepAR wins WQL | DeepAR is strongest on the five tested sensors; Chronos and seasonal naive are the useful alternatives. |

Across the 30 dataset-metric combinations, ARIMA(2,1,2) has the best mean rank (`2.93`),
followed by Chronos (`3.70`). This is only a descriptive rank: the five metrics are correlated,
dataset sample sizes differ, and the weather SMAPE result is not reliable.

## Critical Review Findings

1. The current CSV run is internally complete: 639 detailed rows, 54 summary rows, 71
   forecast tasks, nine models per task, no duplicate task/model rows, no failed rows, and no
   non-finite or negative metric values. RMSE is never below MAE, SMAPE is within `[0, 2]`,
   and the summary CSV recomputes exactly from the detailed components.

2. The original WQL plots did not use the official WQL aggregation. The plotter is now fixed and
   the PNGs have been regenerated from the historical detailed CSV. Use the summary CSV as the
   authoritative numeric source; the old mismatch described later in this report is retained as
   an audit finding, not a current code limitation.

3. Chronos weather SMAPE `2.0` is numerically valid but unusable. The 600 Chronos median
   forecasts were independently reproduced: every value is a tiny positive constant around
   `1.3e-8` to `4.6e-8`. The rainfall target is nonnegative and 72.67% zero. Those tiny values
   make SMAPE score every point as the maximum error, even though they are effectively zero.
   An exact-zero forecast has the same MAE, RMSE, and MASE as Chronos to floating-point
   precision, but SMAPE `0.5467` instead of `2.0`. The SMAPE discontinuity is a numerical
   artifact; the all-zero behavior is a genuine model limitation.

4. PSM is configured with hourly synthetic timestamps and seasonality 24, but its source
   column is `timestamp_(min)` and increments by one minute. The reported horizon is therefore
   24 source steps (apparently 24 minutes), not 24 hours, and MASE is scaled by lag 24 rather
   than a daily lag. PSM values can be compared within this run, but the operational time-scale
   interpretation is invalid until the configuration is corrected and rerun.

5. MA(2) M1 WQL `131.057` is not a software error, but it is not credible forecast quality.
   One series, `11:target`, has WQL `2282.352` and supplies 99.57% of MA(2)'s aggregate
   quantile loss. Its Gaussian forecast uncertainty has effectively exploded. Treat MA(2)'s
   probabilistic result on M1 as a failed forecast even though the row is finite.

6. This is a one-window benchmark over selected leading series/columns: 20 M1 series, 20 M4
   series, 20 weather series, one illness target, five of 862 traffic columns, and five of 25 PSM
   features. There are no rolling origins, repeated training runs, confidence intervals, or
   statistical significance tests. The rankings are evidence for these cases, not universal
   model rankings.

## What Each Metric Means Here

All metrics use the configured median forecast as the point forecast. Lower is better.

| Metric | Meaning in this implementation | Validity cautions |
| --- | --- | --- |
| MAE | Mean absolute median-forecast error in the target's original units. Dataset summary is the mean of per-task MAE. | Never compare MAE across datasets. Large-scale series dominate a mean of raw errors; this is severe for M1 and noticeable for PSM. |
| RMSE | Root mean squared median-forecast error in original units, computed per task and then averaged. It penalizes large misses more than MAE. | Also scale-dependent. The summary is a mean of per-task RMSE, not one global RMSE over all observations. |
| SMAPE | `mean(2*abs(y-p)/(abs(y)+abs(p)))`, reported as `0..2`, not `0..200%`. | Unstable at zero. Weather Chronos `2.0` must not be used for ranking. On nonzero positive datasets it is otherwise coherent. |
| MASE | Point MAE divided by the mean in-sample absolute lag-`seasonality` difference. Summary is an equal-task mean of ratios. | `MASE < 1` means better than the in-sample seasonal-naive scale, not necessarily better than the evaluated seasonal-naive model. A tiny denominator creates large but meaningful values, as in PSM. |
| WQL | Mean pinball loss over quantiles 0.1 through 0.9, normalized by total absolute target. It is dimensionless and unbounded above. | Official summary is target-magnitude weighted. Chronos/DeepAR use samples, Prophet uses predictive samples, ARIMA models assume Gaussian errors, and seasonal naive repeats one point at every quantile. Those uncertainty mechanisms are not equally expressive. |

Seasonal naive WQL is a valid deterministic-loss baseline, but it does not represent calibrated
uncertainty because all nine quantiles are identical. AR/MA/ARMA/ARIMA WQL is meaningful only
to the extent that the fitted Gaussian standard errors are stable. WQL greater than 1 is allowed;
it indicates quantile loss larger than the total absolute target scale, not an out-of-range value.

Runtime is also reported below, but it is elapsed model-call time, not a pure training-time or
inference-time benchmark. Chronos includes lazy model loading in its first call, DeepAR trains
per task, and AutoARIMA searches multiple orders.

## Dataset-Level Results

Values are followed by within-dataset rank. `#1` is best. Seconds are total over that dataset.

### chronos_m1_yearly

20 series, horizon 6, seasonality 1. Future series scales are extremely heterogeneous: absolute
target sums range from `54.13` to `208,952,496`, and the largest series supplies 76.14% of the
official WQL target weight.

| Model | MAE | RMSE | SMAPE | MASE | WQL | Seconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| deepar | 445885 (#1) | 543050 (#1) | 0.289257 (#3) | 6.79569 (#5) | 0.193286 (#1) | 8.642 |
| prophet | 738296 (#2) | 816324 (#2) | 0.280678 (#2) | 6.56468 (#3) | 0.305902 (#2) | 2.533 |
| arima_2_1_2 | 829497 (#3) | 914867 (#3) | 0.216332 (#1) | 4.76390 (#1) | 0.323587 (#3) | 0.808 |
| auto_arima | 924209 (#4) | 1.01959e6 (#4) | 0.298533 (#5) | 6.85136 (#6) | 0.365350 (#4) | 3.693 |
| arma_2_2 | 941433 (#5) | 1.03243e6 (#5) | 0.293543 (#4) | 6.20342 (#2) | 0.375589 (#5) | 0.935 |
| chronos_t5_tiny | 971053 (#6) | 1.05291e6 (#6) | 0.315926 (#7) | 6.97224 (#7) | 0.400216 (#7) | 4.697 |
| ar_2 | 976926 (#7) | 1.06358e6 (#7) | 0.315462 (#6) | 6.68432 (#4) | 0.392839 (#6) | 0.477 |
| seasonal_naive | 989134 (#8) | 1.07698e6 (#8) | 0.318782 (#8) | 7.33629 (#8) | 0.432513 (#8) | 0.008 |
| ma_2 | 1.62731e6 (#9) | 1.69579e6 (#9) | 0.751694 (#9) | 14.1945 (#9) | 131.057 (#9) | 0.499 |

DeepAR is best when high-volume absolute accuracy or target-weighted probabilistic accuracy is
the objective. This is not a broad 20-series victory: the largest-error M1 series contributes
58.1% of DeepAR's summed MAE and 76-89% for most other models. ARIMA(2,1,2) wins the
equal-series relative metrics, so it is the better choice when small and large series deserve
similar importance. It also wins MAE on four individual series versus three for DeepAR.

Prophet is the strongest compromise, ranking second in MAE, RMSE, SMAPE, and WQL. ARMA(2,2)
has good MASE but middling scale-sensitive metrics. AutoARIMA, AR(2), Chronos, and seasonal
naive form a lower middle group. Every model's average MASE is above 1, so none beats the
in-sample lag-1 naive scale on average. MA(2) is last everywhere and its WQL is pathological.

### chronos_m4_hourly

20 series, horizon 48, daily seasonality 24.

| Model | MAE | RMSE | SMAPE | MASE | WQL | Seconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| seasonal_naive | 473.329 (#1) | 590.714 (#1) | 0.048709 (#2) | 0.947608 (#2) | 0.033856 (#2) | 0.010 |
| chronos_t5_tiny | 525.415 (#2) | 660.944 (#2) | 0.045243 (#1) | 0.875271 (#1) | 0.028824 (#1) | 33.437 |
| prophet | 864.940 (#3) | 1000.32 (#3) | 0.070307 (#3) | 1.33811 (#3) | 0.049941 (#3) | 2.383 |
| arima_2_1_2 | 1116.55 (#4) | 1323.91 (#4) | 0.093648 (#4) | 1.86796 (#4) | 0.063255 (#4) | 5.234 |
| arma_2_2 | 1696.64 (#5) | 2017.73 (#5) | 0.126291 (#5) | 2.44264 (#5) | 0.093191 (#5) | 4.532 |
| auto_arima | 1738.28 (#6) | 2068.31 (#6) | 0.129847 (#6) | 2.55602 (#6) | 0.094626 (#6) | 29.172 |
| ar_2 | 1812.13 (#7) | 2153.16 (#7) | 0.140054 (#7) | 2.72247 (#7) | 0.098388 (#7) | 1.064 |
| deepar | 2232.55 (#8) | 2568.67 (#8) | 0.168199 (#8) | 3.34507 (#8) | 0.121651 (#8) | 25.337 |
| ma_2 | 2372.11 (#9) | 2740.15 (#9) | 0.175386 (#9) | 3.47202 (#9) | 0.134430 (#9) | 7.025 |

Seasonal naive has the best aggregate raw errors and is effectively free. Chronos has the best
SMAPE, MASE, and WQL, wins per-series MAE on 10 of 20 tasks and per-series WQL on 15 of 20,
and is therefore the better broad-quality choice. The raw-error disagreement occurs because
large-valued series influence MAE/RMSE more strongly. Both top models have MASE below 1.

Prophet is a clear third. Fixed ARIMA is fourth and gives no advantage over Prophet here.
ARMA, AutoARIMA, AR, DeepAR, and MA all have MASE well above 1; extra complexity did not
capture the daily pattern as well as direct seasonal repetition. DeepAR's poor result is specific
to this short training configuration and should not be generalized to all DeepAR setups.

### chronos_weather

20 rainfall series, horizon 30, configured weekly seasonality 7. Of the 600 future observations,
72.67% are zero and the median is zero.

| Model | MAE | RMSE | SMAPE | MASE | WQL | Seconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| chronos_t5_tiny | 0.806174 (#1) | 2.00862 (#2) | 2.00000 (#9) | 0.211428 (#1) | 0.984189 (#1) | 24.571 |
| deepar | 0.860946 (#2) | 1.96519 (#1) | 1.92805 (#8) | 0.224613 (#2) | 1.03084 (#2) | 44.187 |
| seasonal_naive | 0.897299 (#3) | 2.03209 (#3) | 0.631621 (#1) | 0.229829 (#3) | 1.11303 (#3) | 0.011 |
| prophet | 1.87430 (#4) | 2.37999 (#4) | 1.74980 (#7) | 0.512112 (#4) | 2.62941 (#4) | 35.154 |
| arima_2_1_2 | 2.19178 (#5) | 2.58660 (#5) | 1.72626 (#5) | 0.606297 (#5) | 2.67775 (#5) | 86.231 |
| arma_2_2 | 2.22640 (#6) | 2.61133 (#6) | 1.72635 (#6) | 0.615409 (#6) | 2.68792 (#6) | 121.215 |
| auto_arima | 2.24023 (#7) | 2.62140 (#7) | 1.72557 (#4) | 0.620143 (#7) | 2.69266 (#7) | 128.905 |
| ar_2 | 2.26432 (#8) | 2.63318 (#8) | 1.72513 (#2) | 0.628335 (#8) | 2.70008 (#8) | 8.337 |
| ma_2 | 2.26883 (#9) | 2.63690 (#9) | 1.72525 (#3) | 0.629549 (#9) | 2.70056 (#9) | 30.149 |

Chronos's apparent point lead is exactly the performance of forecasting zero everywhere. Its
WQL is only `0.01581` below the all-zero probabilistic baseline of `1.0`. It should not be
described as learning rainfall amount or occurrence from these metrics. Its SMAPE `2.0` is an
epsilon artifact and must be excluded from ranking, but simply replacing it with the exact-zero
SMAPE would make a trivial forecast look artificially strong.

DeepAR has the best RMSE, suggesting somewhat better treatment of large rain events, but its
MAE and WQL are worse than the zero-like Chronos forecast and its SMAPE is also near the
maximum. Seasonal naive is the most transparent nonlearned comparator and remains very close
on MAE/RMSE/MASE. Prophet and all ARIMA-family point forecasts are poorly matched to this
zero-inflated, nonnegative target; their high WQL values are finite and legal, but not competitive.

This dataset needs a zero baseline plus occurrence-sensitive evaluation: rain/no-rain precision
and recall or Brier score, error conditional on positive rainfall, and a probabilistic score suited
to mixed discrete-continuous targets. Nonnegative clipping should also be evaluated explicitly.

### external_national_illness

One `OT` series, horizon 24, configured yearly seasonality 52.

| Model | MAE | RMSE | SMAPE | MASE | WQL | Seconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| arima_2_1_2 | 49285.0 (#1) | 61576.0 (#1) | 0.032759 (#1) | 0.570783 (#1) | 0.033401 (#2) | 0.300 |
| prophet | 58627.0 (#2) | 73432.9 (#2) | 0.038782 (#2) | 0.678975 (#2) | 0.030373 (#1) | 0.120 |
| ar_2 | 68257.5 (#3) | 83249.8 (#3) | 0.046033 (#3) | 0.790509 (#3) | 0.038335 (#3) | 0.073 |
| auto_arima | 74947.4 (#4) | 95200.7 (#4) | 0.049126 (#4) | 0.867986 (#4) | 0.043038 (#4) | 0.619 |
| chronos_t5_tiny | 83596.6 (#5) | 114741 (#6) | 0.057087 (#5) | 0.968154 (#5) | 0.046262 (#6) | 1.132 |
| arma_2_2 | 97019.8 (#6) | 106807 (#5) | 0.065512 (#6) | 1.12361 (#6) | 0.045469 (#5) | 0.764 |
| deepar | 129746 (#7) | 144528 (#7) | 0.088631 (#7) | 1.50262 (#7) | 0.065613 (#7) | 1.022 |
| seasonal_naive | 153627 (#8) | 246659 (#8) | 0.116386 (#8) | 1.77920 (#8) | 0.101305 (#8) | <0.001 |
| ma_2 | 854579 (#9) | 863451 (#9) | 0.790935 (#9) | 9.89711 (#9) | 0.420612 (#9) | 0.144 |

ARIMA(2,1,2) is the unambiguous point winner. Prophet is second on all point metrics and has
the best WQL, about 9.1% below ARIMA's WQL. AR(2), AutoARIMA, and Chronos also have MASE
below 1; Chronos only narrowly does so. ARMA, DeepAR, seasonal naive, and especially MA(2)
are weak in this window. Because this is one target and one cutoff, even the clean ranking has
very low external validity.

### external_psm

Five of 25 features, horizon 24 source steps. The source index is minute-based; current hourly
labels and daily interpretation are invalid.

| Model | MAE | RMSE | SMAPE | MASE | WQL | Seconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| arima_2_1_2 | 0.002431 (#1) | 0.003084 (#1) | 0.005090 (#1) | 0.269591 (#1) | 0.004230 (#2) | 73.022 |
| auto_arima | 0.002772 (#2) | 0.003390 (#2) | 0.005793 (#2) | 0.357004 (#3) | 0.004361 (#4) | 149.102 |
| arma_2_2 | 0.002794 (#3) | 0.003417 (#3) | 0.005824 (#3) | 0.353257 (#2) | 0.004340 (#3) | 265.870 |
| seasonal_naive | 0.003150 (#4) | 0.003958 (#5) | 0.006550 (#5) | 0.365849 (#4) | 0.004616 (#5) | 0.005 |
| chronos_t5_tiny | 0.003234 (#5) | 0.003818 (#4) | 0.006453 (#4) | 0.655193 (#6) | 0.003751 (#1) | 7.848 |
| ar_2 | 0.003309 (#6) | 0.004008 (#6) | 0.006754 (#6) | 0.518466 (#5) | 0.005669 (#6) | 88.370 |
| deepar | 0.009545 (#7) | 0.012191 (#7) | 0.014929 (#7) | 7.37406 (#7) | 0.027157 (#7) | 93.556 |
| prophet | 0.029832 (#8) | 0.030756 (#8) | 0.048345 (#8) | 23.8420 (#8) | 0.034084 (#8) | 460.469 |
| ma_2 | 0.062626 (#9) | 0.063601 (#9) | 0.096880 (#9) | 53.2803 (#9) | 0.072213 (#9) | 234.451 |

Within the configured 24-step task, ARIMA(2,1,2) is first on every point metric. AutoARIMA
and ARMA are close but much slower, so their extra computation is not justified here. Chronos
is only fifth by MAE but first by WQL, indicating better quantile loss than median accuracy.
Seasonal naive is an unusually strong nearly free baseline.

The large Prophet and MA(2) MASE values are not impossible. Per-feature lag-24 scales range
from only `0.000604` to `0.021449`; errors that look small in raw units can therefore be tens of
times typical lag-24 movement. DeepAR is also poor by this relative standard. Feature 4 supplies
most raw error for the leading statistical models, so MASE is more informative than MAE for an
equal-feature decision. No operational model should be selected until minute timestamps,
horizon, and intended seasonal period are fixed and the benchmark is rerun.

### external_traffic

Five of 862 sensor columns, horizon 24 hours, daily seasonality 24.

| Model | MAE | RMSE | SMAPE | MASE | WQL | Seconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| deepar | 0.007646 (#1) | 0.010568 (#1) | 0.204877 (#2) | 0.378352 (#1) | 0.146575 (#1) | 16.744 |
| chronos_t5_tiny | 0.009495 (#2) | 0.014645 (#3) | 0.198002 (#1) | 0.489688 (#2) | 0.172265 (#2) | 5.422 |
| seasonal_naive | 0.010125 (#3) | 0.014302 (#2) | 0.254521 (#3) | 0.552300 (#3) | 0.229587 (#3) | 0.003 |
| arima_2_1_2 | 0.019555 (#4) | 0.026062 (#4) | 0.526470 (#4) | 1.11468 (#4) | 0.414573 (#4) | 29.638 |
| auto_arima | 0.021457 (#5) | 0.027027 (#5) | 0.567231 (#5) | 1.18162 (#5) | 0.430419 (#5) | 98.718 |
| arma_2_2 | 0.021849 (#6) | 0.027633 (#6) | 0.570536 (#6) | 1.19881 (#6) | 0.435451 (#6) | 73.694 |
| ar_2 | 0.023552 (#7) | 0.028892 (#7) | 0.588316 (#7) | 1.22047 (#7) | 0.459936 (#7) | 16.535 |
| ma_2 | 0.025757 (#8) | 0.031331 (#8) | 0.610427 (#8) | 1.31633 (#8) | 0.478789 (#8) | 38.029 |
| prophet | 0.027246 (#9) | 0.035888 (#9) | 0.923580 (#9) | 1.65358 (#9) | 0.490659 (#9) | 8.652 |

DeepAR wins four metrics and four of five sensors on MAE, RMSE, MASE, and WQL. Chronos wins
SMAPE and is the practical runner-up; seasonal naive has slightly better RMSE than Chronos and
is the speed baseline. All three have MASE below 1. Every ARIMA-family model and Prophet has
MASE above 1. Prophet is last on all five metrics. This ranking is coherent, but five leading
sensor columns are too small and selective a sample for a network-wide traffic conclusion.

## Model-Level Reading

| Model | Evidence-supported strengths | Main limitations in this run |
| --- | --- | --- |
| `chronos_t5_tiny` | Best scaled/probabilistic M4 result, best PSM WQL, strong traffic runner-up, fastest non-naive model overall (`77.1s`) | Weak M1; weather point forecast collapses to zero; probabilistic gains need task-specific checks |
| `deepar` | Best M1 scale-weighted result and strongest traffic model | Poor M4 and PSM; only five epochs and ten batches per epoch; stochastic single-run result |
| `arima_2_1_2` | Most consistent rank; best illness and PSM point forecasts; best normalized M1 point errors | Weak on M4, weather, and traffic; Gaussian WQL assumptions can be misspecified |
| `prophet` | Strong M1 compromise; best illness WQL | Weak traffic/PSM; slowest total runtime (`509.3s`), mostly due PSM |
| `seasonal_naive` | Exceptional speed; top-two M4; competitive weather, PSM, and traffic baseline | Deterministic quantiles are not calibrated uncertainty; cannot adapt trend well |
| `auto_arima` | Solid illness/PSM point model | Never wins a dataset metric and costs `410.2s`; fixed ARIMA is usually better here |
| `arma_2_2` | Good M1/PSM MASE among classical alternatives | No wins, slow (`467.0s`), weak on high-frequency seasonal sets |
| `ar_2` | Fast on short contexts; respectable illness point metrics | Generally lower-middle or weak; no metric wins |
| `ma_2` | None demonstrated | Last or near-last throughout; pathological M1 uncertainty; exclude from candidate set |

## Plot Review

The MAE, RMSE, SMAPE, and MASE PNGs agree with the corresponding simple-mean CSV summaries.
The original WQL PNGs did not agree with official aggregation. They have now been regenerated by
the corrected plotter; this table records the historical discrepancy that prompted the repair.

| Dataset | Official WQL leader | WQL PNG leader | Ranking consequence |
| --- | --- | --- | --- |
| `chronos_m1_yearly` | DeepAR (`0.193286`) | ARIMA(2,1,2) (`0.168744` plotted) | Materially different top three |
| `chronos_m4_hourly` | Chronos (`0.028824`) | Chronos (`0.035148` plotted) | Same winner, different value |
| `chronos_weather` | Chronos (`0.984189`) | Chronos (`0.988769` plotted) | Same winner; second place changes from DeepAR to seasonal naive |
| `external_national_illness` | Prophet (`0.030373`) | Prophet (`0.030373` plotted) | Identical because there is one task |
| `external_psm` | Chronos (`0.003751`) | Chronos (`0.005010` plotted) | Same winner, different value |
| `external_traffic` | DeepAR (`0.146575`) | DeepAR (`0.130116` plotted) | Same winner, different value |

The comparison views now use within-metric ranks, and large positive metric ranges use a log
axis where appropriate. This avoids the old min-max scaling and label-rounding problems. Use the
CSV tables for exact values and the plots for visual comparison.

## Recommended Next Actions

1. **Implemented:** parse PSM minute timestamps and resample to hourly means.
2. **Implemented:** use official WQL aggregation in plots and regenerate comparison figures.
3. **Implemented:** add a zero baseline, rainfall diagnostics, saved forecasts, and trajectories.
4. **Implemented in configuration:** three rolling origins and three seeded stochastic runs.
5. **Implemented:** evenly spaced representative series/features and explicit rainfall filtering.
6. **Pending:** run the full repaired benchmark and replace all historical model-selection claims.
