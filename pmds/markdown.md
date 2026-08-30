No implementation has been performed. The attached text file is empty, so this plan is based on your written request and the existing PMDS codebase.
1. Two additional foundation models
I recommend exactly these two:
Model	Why selected	Role in the benchmark
google/timesfm-2.5-200m-pytorch	TimesFM is one of the most prominent open time-series foundation-model families, originating from an ICML 2024 paper. Version 2.5 supports long contexts and direct quantile forecasts.	High-capacity reference: 200M parameters versus roughly 8M for Chronos-T5-Tiny.
Salesforce/moirai-2.0-R-small	Moirai is another established ICML 2024 foundation-model family. The current small model has 11.4M parameters and directly produces probabilistic quantiles.	Compact comparison close to Chronos-T5-Tiny’s size.


I would use TimesFM 2.5 rather than the newly released TimesFM 3.0 because 2.5 is more mature, ungated, and Apache-2.0 licensed. TimesFM 3.0 is currently gated and non-commercial. The TimesFM repository documents these differences.
Moirai 2.0 Small is research-only under CC-BY-NC-4.0, and its training data includes GIFT-Eval and non-leaking Chronos-derived data. This exposure must be documented from its official model card.
Common evaluation contract
For a fair Chronos–TimesFM–Moirai comparison, I would freeze these rules:
- Primary comparison is zero-shot, univariate, target-only forecasting.
- No fine-tuning, covariates, or dataset-specific post-processing.
- Use the same context, cutoff, timestamps, and horizon for every model.
- Use the median, q=0.5, as the point forecast.
- Change the comparable probabilistic grid to 0.1, 0.2, …, 0.9. TimesFM and Moirai do not natively supply the current PMDS 0.05 and 0.95 tails, and I would not fabricate them by interpolation.
- Use a shared context cap, initially the latest 512 observations. A separate experiment can allow each model its native maximum context.
- Pin checkpoint revision, dtype, package versions, and hardware.
- Run Moirai in a separate locked environment because Uni2TS dependencies may conflict with the current Chronos/GluonTS environment.
- Report two panels:
  - Compact: Chronos-T5-Tiny versus Moirai 2.0 Small.
  - High-capacity: add TimesFM 2.5, including its extra memory and latency cost.
2. Proposed 6+6 dataset portfolio
“Inside Chronos” will mean hosted in autogluon/chronos_datasets. It does not automatically mean that the data appeared in Chronos pretraining. The repository itself contains both in-domain and zero-shot datasets, as shown in the local [zero-shot configuration](/opt/Projects/PMDS/chronos-forecasting/scripts/evaluation/configs/zero-shot.yaml) and [in-domain configuration](/opt/Projects/PMDS/chronos-forecasting/scripts/evaluation/configs/in-domain.yaml).
Six datasets inside Chronos
Dataset/config	Decision	Frequency	Horizon	Seasonality	Main characteristic
monash_m1_yearly	Keep	Yearly	6	1	Short, heterogeneous economic/competition series
monash_m3_quarterly	Add	Quarterly	8	4	Medium-frequency competition data
monash_car_parts	Add	Monthly	12	12	Intermittent, zero-heavy spare-parts demand
monash_nn5_weekly	Add	Weekly	8	52	Banking/cash-withdrawal demand
monash_weather, subset=rain	Keep	Daily	30	7 primary	Zero-inflated environmental observations
m4_hourly	Keep	Hourly	48	24	Dense, high-frequency competition data


This produces yearly, quarterly, monthly, weekly, daily, and hourly coverage. It also deliberately includes smooth, volatile, seasonal, intermittent, and zero-inflated targets. The Monash archive documents the underlying dataset diversity and standard horizons. Monash Forecasting Archive
For weather, I would retain period 7 for continuity but explicitly test periods 1 and 365 as a seasonality-sensitivity experiment.
M4 Hourly was in-domain for the original Chronos-T5 experiment. The other five belong to its published zero-shot group, so M4 should be reported separately rather than hidden inside one combined “Chronos dataset” score.
Six datasets outside Chronos
For the research-quality external cohort, I recommend replacing the frozen TSLib files with reproducible, post-checkpoint target windows:
Dataset	Target	Frequency	Horizon	Seasonality	Series
CDC ILINet	Weighted ILI percentage, national plus HHS regions	Weekly	4	52	11
NYC TLC Yellow Taxi	Pickup count by pickup zone	Hourly	24	24	20 stratified zones
EIA-930	Electricity demand for CISO, ERCO, MISO, NYIS, PJM	Hourly	24	24	5
USGS Streamflow	Daily mean discharge from five fixed gauges	Daily	30	365	5
Wikimedia Pageviews	Aggregate views for ten language projects	Daily	30	7	10
FRED-MD fixed vintage	Seven macroeconomic indicators	Monthly	6	12	7


Primary sources: CDC FluView, NYC TLC records, EIA API, USGS Water Data API, Wikimedia Analytics API, and FRED-MD vintages.
I would freeze each external source to an immutable snapshot and score targets beginning after October 1, 2025, where sufficient data exists. That makes the targets post-release for Chronos-T5, Moirai 2.0, and TimesFM 2.5.
The current external datasets would remain as a legacy regression suite:
- external_national_illness is replaced in the primary suite by current CDC ILINet.
- external_traffic is not a strong independent external test because the same underlying Traffic benchmark also exists in the Chronos benchmark collection.
- external_psm is fundamentally an anomaly-detection dataset; I would move it into the robustness appendix rather than treat five arbitrary channels as a main forecasting dataset.
If minimal engineering were the overriding priority, the fallback would be to retain the existing three and add ETTm1, exchange_rate, and electricity. I would not use that as the primary research design because it provides weaker independence from common pretraining corpora.
3. Dataset construction protocol
Before running models, I would:
1. Freeze exact dataset revisions, raw files, queries, schema, selected series IDs, timestamps, and SHA-256 hashes.
2. Build a model-by-dataset exposure matrix:
   - confirmed pretraining exposure;
   - officially claimed unseen;
   - unknown exposure;
   - post-checkpoint target.
3. Select large-panel series using context-only stratification by scale, intermittency, volatility, length, and seasonal strength. Source row order or “first 20” would not be used.
4. Forecast every external multivariate panel channel independently in the primary benchmark. A native multivariate experiment would be a separate leaderboard.
5. Target five rolling origins where the series permits it, with at least three for normal datasets and two for short Car Parts series.
6. Use identical cutoffs and origins across every model.
Two PMDS issues must be resolved before robustness results would be trustworthy:
- [datasets.py (line 117)](/opt/Projects/PMDS/chronos-forecasting/pmds/datasets.py:117) cleans the complete series before splitting, while [utils.py (line 16)](/opt/Projects/PMDS/chronos-forecasting/pmds/utils.py:16) uses bidirectional interpolation. The later design must split first and use context-only causal imputation.
- [metrics.py (line 124)](/opt/Projects/PMDS/chronos-forecasting/pmds/metrics.py:124) calculates the MASE scale from the provided context. For corrupted contexts, the denominator must remain frozen from the pristine context; otherwise spikes and noise can artificially improve MASE.
4. Robustness evaluation
The core idea is a paired evaluation:
Same dataset + series + origin + model
        ├── clean context → clean loss
        └── corrupted context → stressed loss
The future target remains identical for observation-corruption tests. Every model receives the same pre-generated corruption mask and severity.
Core stress tests
Let H be horizon and r a robust scale derived from the clean context, such as the MAD of seasonal residuals.
Stress	Mild	Moderate	Severe
Random missing observations	5%	10%	20%
Trailing sensor outage	last 0.25H	0.5H	H
Gaussian sensor noise	0.05r	0.10r	0.20r
Outlier spikes	1% at 3r	3% at 5r	5% at 10r
Recent sensor bias	0.25r	1r	2r
Context reduction	retain long context	retain medium context	retain minimum seasonal context
Scale/unit change	×0.5/×2	×0.1/×10	×0.01/×100


Scale changes are invariance tests: transform the task, generate the forecast, inverse-transform the output, and score in the original units.
Missingness will have two tracks:
- shared causal preprocessing available to every model;
- native missing-value handling for models that officially support it.
Natural and extended shifts
Natural regime robustness will compare origins with low versus high:
- mean or level change;
- variance change;
- trend change;
- seasonal-strength change;
- autocorrelation change;
- zero/intermittency-rate change.
Extended tests will include:
- persistent level, trend, variance, and seasonal-amplitude changes;
- resampling or cadence changes while preserving the wall-clock horizon;
- combined faults such as missingness plus level shift;
- multivariate dependency fractures in a separate multivariate track;
- sparse black-box/adversarial perturbations;
- model-size, context-length, and sampling-parameter sensitivity.
This design follows the paired clean/corrupt idea used by the recent TS-Fault benchmark. Adversarial testing would be confirmatory rather than the main robustness definition, following concerns identified by the Microsoft TSFM robustness study.
5. Experimental sizing
I would divide execution into:
1. Full clean benchmark:
   - all selected series;
   - three to five rolling origins;
   - three inference/training seeds for stochastic models;
   - one run plus a determinism check for deterministic models.
2. Core robustness panel:
   - five representative series per multi-series dataset;
   - three origins;
   - moderate and severe levels;
   - all models;
   - three corruption seeds shared across models.
3. Extended robustness:
   - full three-level severity curves;
   - foundation models plus seasonal naive first;
   - expand to all models only for important effects;
   - ten-seed sensitivity on a smaller panel.
Model seeds and corruption seeds must be independent. Stochastic repetitions will be averaged before aggregating tasks so those models do not receive extra statistical weight.
6. Measurements and reporting
Primary clean metrics:
- MASE using the median;
- WQL using the common nine quantiles;
- calibration coverage and interval score.
Secondary metrics:
- MAE, RMSE, SMAPE;
- rain/zero occurrence error;
- positive-only MAE;
- runtime and peak memory.
Robustness outputs:
- robustness_ratio = stressed_loss / clean_loss;
- percentage degradation;
- area under the severity curve;
- worst-dataset degradation;
- clean-to-stress rank correlation;
- task success/failure rate;
- non-finite forecasts and quantile crossings;
- cold model-load time versus warm inference latency.
Aggregation will be:
1. seeds within a task;
2. origins within a series;
3. equal-weight series within a dataset;
4. equal-weight datasets in the final macro result.
Confidence intervals will use paired hierarchical bootstrap resampling over datasets and series. I would report clean quality, robustness, and computational cost separately as a Pareto comparison—not hide them inside one subjective “overall winner” score. This is consistent with the diverse-domain evaluation philosophy behind GIFT-Eval.
Planned order of work
If this plan is later approved for implementation, the order would be:
1. Write and freeze the experiment protocol and exposure matrix.
2. Freeze model revisions, environments, quantiles, and context rules.
3. Snapshot and validate all 12 datasets.
4. Add leakage and split-integrity tests.
5. Validate TimesFM and Moirai on one synthetic task.
6. Run a one-series, one-origin audit over all 12 datasets.
7. Run the full clean benchmark.
8. Freeze the robustness panel before looking at model rankings.
9. Run core and then extended robustness tests.
10. Produce separate accuracy, robustness, calibration, failure, and resource reports.
That would be the complete plan; no code or configuration should be changed until the protocol and dataset portfolio are explicitly approved.





Final Experimental Plan
Classical Time-Series Models vs Time-Series Foundation Models
1. Research Question

The benchmark should answer two separate questions:

Clean forecasting

How do existing classical time-series models compare with modern time-series foundation models under normal forecasting conditions?

Robustness

When the historical observations available at forecast time are imperfect, do classical models and foundation models degrade differently?

Do not combine accuracy and robustness into a single score.

The final comparison should show:

[
\boxed{\text{Clean Accuracy}}
]

and

[
\boxed{\text{Robustness}}
]

separately.

2. Models
2.1 Classical models

The project already has a predefined list of classical forecasting models.

Rule

Use exactly the existing classical-model list.

Do not:

add ARIMA because of this robustness study;
remove any existing classical model;
replace existing models;
introduce special robust versions;
change hyperparameters specifically for corrupted data.

The robustness framework must work automatically with every model already registered in the classical benchmark.

Conceptually every model should expose:

[
\texttt{forecast(context, horizon)}
\rightarrow
\hat y.
]

2.2 Existing foundation model

Keep the existing:

Chronos

Checkpoint:

amazon/chronos-t5-tiny

Parameters:

[
\approx 8\text{M}
]

Chronos-T5-Tiny is already in the project and should remain the compact reference foundation model.

2.3 New foundation models to add

Add exactly two models.

Model A — TimesFM 2.5

Checkpoint:

google/timesfm-2.5-200m-pytorch

Parameters:

[
\approx 200\text{M}
]

Role:

High-capacity foundation-model reference.

TimesFM 2.5 officially uses a 200M-parameter architecture and supports context lengths up to 16,384 observations; its checkpoint exposes quantiles from 0.1 to 0.9.

Model B — Moirai 2.0 Small

Checkpoint:

Salesforce/moirai-2.0-R-small

Parameters:

[
11.4\text{M}
]

Role:

Compact foundation model, much closer in scale to Chronos-T5-Tiny.

The official model card reports 11.4M parameters and notes that Moirai 2.0 uses quantile forecasting and was pretrained partly on non-leaking GIFT-Eval/Chronos-derived data.

2.4 Final model comparison

Therefore:

[
\boxed{
\text{Existing Classical Models}
}
]

versus

[
\boxed{
\text{Chronos-T5-Tiny}
}
]

[
\boxed{
\text{TimesFM 2.5}
}
]

[
\boxed{
\text{Moirai 2.0 Small}
}
]

We are therefore adding 2 models, not 3.

Chronos already exists.

3. Common Model Contract

The primary benchmark is:

zero-shot for foundation models;
univariate;
target-only;
no covariates;
no fine-tuning;
no dataset-specific postprocessing.

Use the same observations for every model.

Maximum common context:

[
L_{\max}=512.
]

Therefore:

context = latest min(series_length, 512) observations

even though TimesFM can accept substantially longer contexts.

This prevents the comparison from becoming:

model architecture + different amount of information.

Native-context experiments may later be added as a sensitivity analysis, but they are not part of the main benchmark.

4. Forecast Output

For point forecasting use:

[
q=0.5.
]

That is the median forecast.

For probabilistic models use the common grid:

[
Q=
{
0.1,0.2,\ldots,0.9
}.
]

Do not interpolate unsupported 0.05 or 0.95 quantiles.

TimesFM's published configuration includes exactly these nine quantiles.

5. Dataset Portfolio

The final primary benchmark contains:

[
\boxed{12\text{ datasets}}
]

split into:

[
6\text{ Chronos/Monash benchmark datasets}
]

and

[
6\text{ independent external datasets}.
]

6. Chronos/Monash Dataset Group

Use exactly these six.

Dataset	Status	Frequency	Horizon (H)	Seasonal period	Main property
monash_m1_yearly	Keep	Yearly	6	1	Short heterogeneous economic/competition series
monash_m3_quarterly	Add	Quarterly	8	4	Medium-frequency competition data
monash_car_parts	Add	Monthly	12	12	Intermittent / zero-heavy demand
monash_nn5_weekly	Add	Weekly	8	52	Banking/cash-demand series
monash_weather, subset=rain	Keep	Daily	30	7	Zero-inflated environmental data
m4_hourly	Keep	Hourly	48	24	High-frequency dense series

The Monash archive provides broad competition and real-world datasets across frequencies and domains.

Important

Being contained in the Chronos dataset repository does not mean a dataset necessarily appeared in Chronos pretraining.

Maintain a separate exposure table.

In particular, the existing project plan identifies M4 Hourly as in-domain for the original Chronos experiment and the other selected Monash datasets as belonging to its zero-shot evaluation group.

Report this explicitly.

7. New External Dataset Group

Add six external datasets.

These are important because comparison only on standard ML benchmark datasets gives weak evidence about true zero-shot generalization.

External target windows should start:

[
\boxed{\text{January 1, 2026 or later}}
]

where possible.

Reason:

TimesFM 2.5 was released September 15, 2025.
Moirai 2.0 was published November 12, 2025.

Targets from 2026 therefore provide a clean post-release evaluation window.

Historical context may of course extend before 2026.

7.1 CDC ILINet

Source:

CDC FluView / ILINet

Target:

Weighted ILI percentage

Frequency:

Weekly

Forecast horizon:

[
H=4
]

Seasonal period:

[
s=52
]

Series:

National
HHS Region 1
HHS Region 2
...
HHS Region 10

Total:

[
11\text{ series}.
]

CDC provides national and HHS-regional outpatient ILI data through FluView.

Use all 11 series.

7.2 NYC TLC Yellow Taxi

Source:

NYC Taxi & Limousine Commission
Yellow Taxi Trip Records

Target:

Hourly number of pickups per taxi zone

Frequency:

Hourly

Forecast horizon:

[
H=24.
]

Seasonal period:

[
s=24.
]

Use:

[
20\text{ taxi zones}.
]

The TLC records contain pickup timestamps and pickup locations, allowing hourly zone-level pickup counts to be constructed directly.

Zone selection

Do not use the first 20 zone IDs.

Using pre-2026 history only:

calculate mean hourly pickup volume for every eligible zone;
divide eligible zones into five volume quintiles;
select four zones per quintile;
use fixed seed:
2026
save the final 20 zone IDs permanently in the dataset manifest.

This creates:

[
5\times4=20
]

representative low-to-high-volume series.

7.3 EIA-930 Electricity Demand

Source:

U.S. Energy Information Administration
EIA-930

Target:

Actual electricity demand

Frequency:

Hourly

Forecast horizon:

[
H=24.
]

Seasonal period:

[
s=24.
]

Use exactly these balancing authorities:

CISO
ERCO
MISO
NYIS
PJM

Therefore:

[
5\text{ series}.
]

The EIA open-data API explicitly provides hourly operating data for balancing authorities including actual and forecast demand.

7.4 USGS Streamflow

Source:

USGS Water Data

Target:

Daily mean discharge

USGS parameter:

00060

Statistic:

00003

Frequency:

Daily

Forecast horizon:

[
H=30.
]

Primary seasonal period:

[
s=365.
]

USGS defines parameter 00060 as discharge and supports daily mean observations through its Daily Values service.

Use:

[
5\text{ gauges}.
]

Gauge selection

Eligible gauge:

active;
at least 10 years of history before 2026;
less than 1% missing observations in the evaluation context;
valid daily mean discharge.

Then:

calculate log median discharge using pre-2026 data;
divide eligible sites into five quintiles;
select one gauge per quintile using fixed seed 2026;
freeze the five USGS gauge IDs in the dataset manifest.

Do not select gauges based on forecasting performance.

7.5 Wikimedia Pageviews

Source:

Wikimedia Analytics API

Target:

Aggregate daily pageviews

Frequency:

Daily

Forecast horizon:

[
H=30.
]

Seasonal period:

[
s=7.
]

Use exactly these ten projects:

en.wikipedia
de.wikipedia
fr.wikipedia
es.wikipedia
it.wikipedia
ja.wikipedia
ru.wikipedia
zh.wikipedia
pt.wikipedia
ar.wikipedia

Use:

all-access
all-agents

The Wikimedia Analytics API provides aggregate pageview queries at project level.

Total:

[
10\text{ series}.
]

7.6 FRED-MD

Source:

Federal Reserve Bank of St. Louis
FRED-MD

Frequency:

Monthly

Forecast horizon:

[
H=6.
]

Seasonal period:

[
s=12.
]

Use these seven series:

RPI
INDPRO
PAYEMS
UNRATE
HOUST
CPIAUCSL
FEDFUNDS

These cover:

income;
industrial production;
employment;
unemployment;
housing;
inflation/prices;
interest rates.

Use a fixed FRED-MD vintage, not current.csv.

The St. Louis Fed publishes timestamped historical FRED-MD vintages specifically for reproducible research.

Freeze the exact selected vintage and its file hash.

8. What Happens to Existing External Datasets?

The current legacy external datasets should not remain in the primary leaderboard.

external_national_illness

Replace with:

CDC ILINet

because the CDC source is reproducible and current.

external_traffic

Remove from the primary external benchmark.

Reason:

The underlying Traffic benchmark already overlaps heavily with standard forecasting benchmark collections, weakening its value as an independent external dataset.

Keep it only for regression/testing if desired.

external_psm

Remove from the primary forecasting benchmark.

Reason:

PSM is fundamentally an anomaly-detection dataset.

It may remain in an appendix or software regression suite.

Do not treat arbitrary PSM channels as one of the main forecasting datasets.

9. Final Dataset Table

The primary benchmark is therefore:

#	Dataset	Frequency	H	Series type
1	M1 Yearly	Yearly	6	Monash
2	M3 Quarterly	Quarterly	8	Monash
3	Car Parts	Monthly	12	Monash
4	NN5 Weekly	Weekly	8	Monash
5	Weather Rain	Daily	30	Monash
6	M4 Hourly	Hourly	48	Monash
7	CDC ILINet	Weekly	4	External
8	NYC TLC Taxi	Hourly	24	External
9	EIA-930 Demand	Hourly	24	External
10	USGS Streamflow	Daily	30	External
11	Wikimedia Pageviews	Daily	30	External
12	FRED-MD	Monthly	6	External

This deliberately covers:

yearly
quarterly
monthly
weekly
daily
hourly

and includes:

smooth
seasonal
volatile
intermittent
zero-heavy
economic
energy
transport
health
environmental
web-traffic

series.

10. Dataset Freezing

Before running any model, save:

source
download/query
download date
raw file
series IDs
timestamps
frequency
horizon
seasonality
selected forecast origins
SHA-256

for every dataset.

Do not download "latest" data separately for different models.

Every model must use the identical frozen files.

11. Leakage / Exposure Matrix

Create:

model_dataset_exposure.csv

with:

model
dataset
status
evidence

Allowed statuses:

confirmed_pretraining
claimed_unseen
unknown
post_release_target

Never silently label an external dataset "unseen" unless that is documented.

For 2026 external targets use:

post_release_target

when appropriate.

Moirai's model card explicitly states that its pretraining contains portions of GIFT-Eval and Chronos-derived data, so exposure must be documented rather than assumed.

12. Data Preprocessing

This is critical.

Split before cleaning

Never clean the entire series and then split.

Correct:

raw series
    ↓
forecast origin
    ↓
context | future
    ↓
clean context only

Incorrect:

raw series
    ↓
bidirectional interpolation
    ↓
split

Future observations must never influence context preprocessing.

The existing PMDS plan already identified this potential leakage issue.

13. Missing Values in Clean Data

Use one shared causal preprocessing strategy where necessary.

Never use:

future observations;
backward interpolation from future target;
dataset-specific model-aware preprocessing.

Record exactly how many points were imputed.

If a context contains too many naturally missing values, mark the task invalid rather than performing aggressive reconstruction.

14. Clean Benchmark

Run the full clean benchmark before robustness.

For every:

dataset
series
origin
model

generate the clean forecast.

Use rolling origins.

Origin rule

Prefer:

[
5
]

non-overlapping rolling origins spaced by one forecast horizon:

[
H.
]

If insufficient history exists:

[
3
]

origins are acceptable.

For very short datasets such as Car Parts:

[
2
]

origins are acceptable if necessary.

Every model uses exactly the same origins.

15. Robustness Philosophy

The core experiment is paired:

same dataset
same series
same origin
same model
same target

clean context      -> clean prediction
stressed context   -> stressed prediction

For observation-corruption tests:

[
y_{\text{stress}}=y_{\text{clean}}.
]

Therefore changes in forecast quality are attributable to changes in the observed history.

16. Robustness Test 1 — Sparse Outlier Spikes
Question

Can the model tolerate a small number of severely incorrect recent observations?

Represents:

sensor glitches;
erroneous records;
extreme measurement errors.
Candidate window

Let:

[
L=\text{context length}
]

and

[
H=\text{forecast horizon}.
]

Define:

[
W=\min(2H,L).
]

Only the last (W) observations may be corrupted.

This is deliberate because recent observations generally matter more for the current forecast.

17. Robust Scale

Compute scale only from the pristine context.

Primary definition:

1.4826,
\operatorname{MAD}(e_t),
]

where (e_t) are seasonal differences:

[
e_t=x_t-x_{t-s}.
]

If insufficient seasonal history exists, use:

[
e_t=x_t-x_{t-1}.
]

If that MAD is zero, use raw-context MAD:

[
r=
1.4826,MAD(x).
]

If still zero, use:

[
r=\operatorname{std}(x).
]

If all are zero, the series is effectively constant; record the test as not applicable.

18. Outlier Construction

Select (k) observations from the recent (W):

[
k=
\max(1,\operatorname{round}(pW)).
]

For each selected point:

x_t+s_t a r,
]

where:

[
s_t\in{-1,+1}.
]

Severity
Severity	Fraction (p)	Amplitude
Mild	1%	(3r)
Moderate	3%	(5r)
Severe	5%	(10r)

Use corruption seeds:

2026
2027
2028

The corruption mask must be generated once and reused by every model.

19. Robustness Test 2 — Trailing Sensor Outage
Question

Can the model forecast when the latest observations disappear?

Represents:

API outage;
sensor outage;
reporting delay;
telemetry interruption.

Do not impute this outage.

Suppose normal forecasting is:

[
x_{1}
\rightarrow
y_{T+1+H}.
]

For an outage of (G):

give the model only

[
x_{1}.
]

Request:

[
G+H
]

predictions.

The model returns:

[
\hat y_{T-G+1+H}.
]

Discard the first (G) predictions.

Score only:

[
\hat y_{T+1+H}
]

against the original:

[
y_{T+1+H}.
]

Severity
Severity	Outage
Mild	(G=\lceil0.25H\rceil)
Moderate	(G=\lceil0.5H\rceil)
Severe	(G=H)

This test is deterministic.

No corruption seed is required.

20. Robustness Test 3 — Recent Sensor Bias
Question

Can the model tolerate systematic drift in recent measurements?

Represents:

calibration error;
biased sensor;
persistent measurement offset.

Unlike outliers, the corruption is not sparse.

Use:

[
W=\min(2H,L).
]

For every observation in the recent window:

[
\tilde x_t=x_t+b r.
]

The future remains unchanged.

Severity
Severity	Bias magnitude
Mild	(0.25r)
Moderate	(1r)
Severe	(2r)

Run both directions:

[
+b
]

and

[
-b.
]

Therefore results do not depend on whether the underlying series happens to react more strongly to an upward or downward shift.

Average the two directions.

21. Robustness Test 4 — Scale / Unit Invariance
Question

Does changing physical units change the underlying model behavior?

For example:

dollars -> cents
MW -> kW
meters -> centimeters

Given:

[
x,
]

transform:

[
x'=cx.
]

Forecast:

[
\hat y'=f(cx).
]

Return to original units:

\frac{\hat y'}{c}.
]

Scale factors
Severity	Factors
Mild	(0.5,;2)
Moderate	(0.1,;10)
Severe	(0.01,;100)

Run both factors at each severity.

22. Scale-Invariance Metric

For this test, do not rely only on forecasting error against the target.

Also measure whether the forecast itself changed.

Let:

[
\hat y
]

be the clean prediction.

Define:

\frac{
\frac1H
\sum_h
|
\hat y_h-
\hat y^{back}{h,c}
|
}{
S{\text{clean}}
},
]

where (S_{\text{clean}}) is the clean MASE scaling denominator.

Ideal:

[
\boxed{I_c=0}.
]

Also report normal stressed MASE after inverse scaling.

23. Tests We Are NOT Running

Do not currently implement:

random missing observations
Gaussian sensor noise
natural regime shifts
trend shifts
variance shifts
seasonality shifts
cadence changes
adversarial attacks
dependency fractures
combined faults

Reasons:

Random missingness

Requires deciding between imputation and native missing-value support, confounding model comparison.

Trailing outage gives a cleaner missing-information test.

Gaussian noise

Overlaps substantially with the observation-corruption question already covered by outliers and bias.

Natural regime shifts

They change the underlying data-generating process and future distribution, so they should be a separate distribution-shift study.

Adversarial attacks

White-box attacks cannot be applied equally to all classical and foundation models.

24. Optional Sensitivity Analysis

If budget remains, add:

context length

with:

512
256
128

or equivalent available history.

Do not call this one of the four robustness tests.

Call it:

Context-Length Sensitivity
25. Robustness Dataset Subset

Do not necessarily run every robustness transformation over every series in huge datasets.

For each dataset:

if it contains (\leq 5) series: use all;
otherwise select 5 representative series.

Use the same selected five for every model.

Selection must use context only

Representative-series features may include:

scale
zero/intermittency rate
volatility
series length
seasonal strength

Never select series based on model forecast performance.

Freeze selected IDs before running robustness models.

26. Robustness Origins

Use:

[
3
]

rolling origins per selected robustness series.

These should be the same origins for every model and every corruption.

27. Model Seeds

For deterministic models:

1 run
+
1 initial determinism check

If two identical runs produce the same result, run once thereafter.

For stochastic models:

3 model seeds

Use:

1001
1002
1003

These are independent from corruption seeds.

28. Corruption Seeds

For stochastic corruptions use:

2026
2027
2028

Never couple model randomness with corruption randomness.

29. Primary Clean Metric

Use:

[
\boxed{\text{MASE}}
]

for the across-all-model comparison.

For point prediction:

\frac{
\frac1H
\sum_{h=1}^{H}
|y_h-\hat y_h|
}{
S
}.
]

Compute (S) from the pristine clean context.

30. Critical MASE Rule

For the same:

dataset + series + origin

the denominator must satisfy:

S_{\text{outlier}}
S_{\text{outage}}
S_{\text{bias}}

S_{\text{scale}}.
]

Never recompute it from a corrupted context.

Otherwise, for example, outliers could increase the scaling denominator and artificially improve MASE.

The existing PMDS plan already identifies this issue.

31. Clean Probabilistic Metrics

For models that support probabilistic forecasts, also report:

[
\boxed{\text{WQL}}
]

using:

[
Q={0.1,\ldots,0.9}.
]

Also report:

calibration/coverage;
interval score.

If some existing classical models produce only point forecasts, do not fabricate probabilistic forecasts.

Therefore:

All-model leaderboard
MASE
Probabilistic-model leaderboard
WQL
coverage
interval score
32. Secondary Metrics

Store:

MAE
RMSE
SMAPE
runtime
peak memory
failure status

Do not make all of these primary.

33. Main Robustness Metric

For every corruption:

\frac{
\operatorname{MASE}{stress}
}{
\operatorname{MASE}{clean}
}.
]

Interpretation:

[
R=1
]

means no degradation.

[
R=1.2
]

means error increased by 20%.

[
R=2
]

means error doubled.

Also report:

[
D=100(R-1)
]

and

MASE_{\text{stress}}

MASE_{\text{clean}}.
]

34. Severity Curves

For each model produce separate severity curves for:

Outliers
Outage
Sensor Bias
Scale Invariance

Do not average these four tests into one robustness number.

Each measures a different failure mechanism.

35. Area Under Severity Curve

Optionally summarize each individual robustness test by its area under the severity curve.

For example:

[
AUC_{\text{outlier}}
]

and

[
AUC_{\text{outage}}.
]

But still keep the four tests separate.

Do not produce:

overall robustness score
36. Aggregation

Use the following hierarchy.

1. Model seeds

Average model seeds.

2. Corruption seeds

Average corruption seeds.

3. Origins

Average origins within each series.

4. Series

Equal-weight all selected series within a dataset.

5. Dataset

Each dataset contributes equal weight to the final macro result.

Therefore a dataset containing thousands of series cannot dominate a dataset containing five.

37. Confidence Intervals

Use paired hierarchical bootstrap.

Resample:

datasets
    ↓
series within dataset

while preserving clean/stress pairing.

Report:

[
95%\text{ confidence intervals}.
]

38. Required Result Tables
Table 1 — Clean forecasting
Model	Family	MASE	MAE	RMSE	Runtime
Table 2 — Robustness at Moderate Severity
Model	Family	Clean MASE	Outlier R	Outage R	Bias R	Scale deviation

This should be the main robustness table.

Table 3 — Severe robustness
Model	Outlier R	Outage R	Bias R	Scale deviation
Table 4 — Dataset-level results

Report each model separately across all 12 datasets.

This prevents the macro average from hiding dataset-specific failures.

39. Failure Reporting

Record:

fit failure
inference failure
NaN forecast
Inf forecast
quantile crossing
timeout
OOM

Do not silently exclude failed runs.

Also report:

[
\text{task failure rate}.
]

40. Required Long-Form Result File

Create a row for every execution:

model
model_family
dataset
series_id
origin
test
severity
corruption_seed
model_seed
context_length
forecast_horizon
clean_mase
stress_mase
clean_mae
stress_mae
robustness_ratio
percentage_degradation
absolute_degradation
scale_invariance_error
runtime_seconds
peak_memory
status
41. Reproducibility Layout

Suggested:

experiments/
    manifests/
        models.yaml
        datasets.yaml
        exposure.csv
        origins.csv
        robustness_series.csv

    corruptions/
        <dataset>/
            <series>/
                <origin>/
                    outlier_seed_2026.npz
                    outlier_seed_2027.npz
                    outlier_seed_2028.npz
                    bias.json

    results/
        clean.csv
        robustness.csv
        failures.csv
42. Pre-Generated Corruption Files

Each outlier corruption file should contain:

series_id
origin
selected_indices
signs
robust_scale_r
severity
fraction
amplitude
seed
context_hash

Every model must load this file.

Models must never independently generate their own corruption.

43. Implementation Order

Implement in this exact order.

Phase 1 — Freeze protocol
Freeze model list.
Pin all checkpoint revisions.
Freeze dataset list.
Freeze external-source snapshots.
Generate exposure matrix.
Freeze quantiles and context cap.
Phase 2 — Dataset work
Add the three missing Monash/Chronos configs:
M3 Quarterly;
Car Parts;
NN5 Weekly.
Add CDC ILINet.
Add NYC TLC.
Add EIA-930.
Add USGS streamflow.
Add Wikimedia pageviews.
Add FRED-MD.
Add split-before-cleaning validation tests.
Phase 3 — Model work
Add TimesFM 2.5 adapter.
Add Moirai 2.0 Small adapter.
Verify all existing classical-model adapters.
Validate all models on one synthetic forecasting task.
Phase 4 — Clean audit
Run:
1 model
×
1 series
×
1 origin

for every dataset.

Check indexing, timestamps, forecasts and metrics manually.
Then run all models on the same audit tasks.
Only after this passes, run the full clean benchmark.
Phase 5 — Robustness implementation
Implement robust-scale computation.
Implement outlier corruption.
Pre-generate outlier masks.
Implement trailing outage.
Test the G+H indexing carefully.
Implement recent sensor bias.
Implement scale/unit transformation.
Freeze MASE denominator from clean context.
Phase 6 — Robustness audit
Select robustness-series subset.
Freeze three robustness origins.
Run one dataset/series/origin through all four tests.
Plot clean and corrupted contexts.
Verify targets are identical.
Verify every model receives identical input corruption.
Phase 7 — Full experiments
Run full clean benchmark.
Run outlier benchmark.
Run outage benchmark.
Run sensor-bias benchmark.
Run scale-invariance benchmark.
Aggregate results.
Bootstrap confidence intervals.
Produce final tables and severity curves.
44. Final Scope
Models
Existing
All current classical models
Chronos-T5-Tiny
Add
TimesFM 2.5 200M
Moirai 2.0 Small
Primary datasets
Keep existing
M1 Yearly
Weather Rain
M4 Hourly
Add from Chronos/Monash
M3 Quarterly
Car Parts
NN5 Weekly
Add external
CDC ILINet
NYC TLC Yellow Taxi
EIA-930 Electricity Demand
USGS Streamflow
Wikimedia Pageviews
FRED-MD

Total:

[
\boxed{12\text{ primary datasets}}
]

Primary robustness tests
1. Sparse outlier spikes
2. Trailing sensor outage
3. Recent sensor bias
4. Scale/unit invariance

Each has:

mild
moderate
severe

severity.

45. Final Scientific Structure

The final paper/results should therefore have three distinct comparisons:

A. Clean forecasting quality

[
\text{Which models forecast best normally?}
]

B. Deployment robustness

[
\text{Outliers + Outage + Bias}
]

asks:

How much does performance degrade when observations are faulty or unavailable?

C. Representation/invariance robustness

[
\text{Scale transformation}
]

asks:

Does changing the numerical unit alter model behavior?

The key scientific question becomes:

Do foundation models improve clean forecasting accuracy relative to classical models, and if so, is that improvement accompanied by greater or smaller sensitivity to corrupted, unavailable, or rescaled historical observations?

That is the frozen first-version experimental plan.