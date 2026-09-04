# Sources and figure-selection notes

## Foundation-model papers

- Chronos: Ansari et al., *Chronos: Learning the Language of Time Series*,
  TMLR 2024, [arXiv:2403.07815](https://arxiv.org/abs/2403.07815).
  Figure 1 is used because it explains the full scale/quantize/tokenize/sample
  pipeline in one visual.
- TimesFM: Das et al., *A Decoder-Only Foundation Model for Time-Series
  Forecasting*, ICML 2024,
  [arXiv:2310.10688](https://arxiv.org/abs/2310.10688). Figure 1 is used because
  it makes the input-patch, causal Transformer, and output-patch mechanism
  explicit.
- Moirai: Woo et al., *Unified Training of Universal Time Series Forecasting
  Transformers*, ICML 2024,
  [arXiv:2402.02592](https://arxiv.org/abs/2402.02592). Figure 2 is used because
  it combines multi-patch tokenization, any-variate processing, masked forecast
  tokens, and probabilistic output.

The deck uses cropped PNGs so it remains portable and does not depend on PDF
page-cropping support. Source lines are deliberately kept out of the visual
slides; this file records their provenance. The original PDFs can be retrieved
from the arXiv links above and are not duplicated in the repository.

## Dataset metadata

- The exact enabled datasets, evaluated series counts, frequencies,
  seasonalities, horizons, and pinned revisions come from `../config.json`.
- The Monash panel counts and approximate lengths come from Godahewa et al.,
  *Monash Time Series Forecasting Archive*,
  [arXiv:2105.06643](https://arxiv.org/abs/2105.06643).
- National Illness, Traffic, and PSM dimensions and row counts were checked
  against the pinned Hugging Face dataset metadata in the local cache.
- Elexon, USGS, and BLS spans and dimensions come from the fixed snapshot
  metadata in `../data/official/`.

## Implementation-specific statements

Forecast-output behavior is based on `../models.py`:

- Seasonal Naive repeats its point forecast across quantiles.
- ARIMA and AutoARIMA return Gaussian quantiles.
- Prophet and DeepAR produce samples that are converted to quantiles.
- Chronos produces sample paths.
- TimesFM 2.5 uses its continuous q10--q90 quantile head.
- Moirai 2.0 Small directly returns q10--q90 forecasts; the q50 forecast is the
  point forecast in the benchmark.

Metric definitions and the q10--q90 CRPS approximation are taken from
`../metrics.py`.
