# PMDS 12-dataset benchmark setup

The primary config contains 12 enabled datasets: the original six, three additional
Chronos/Monash datasets, and three fixed official-source datasets. It compares the
existing classical models with Chronos-T5-Tiny, TimesFM 2.5, and Moirai 2.0 Small.

Install the optional foundation-model integrations from the pinned source revisions:

```bash
source /opt/Projects/PMDS/genai/bin/activate
pip install -e '.[pmds]'
```

The first official-data run needs EIA and FRED API keys. USGS requires no key:

```bash
export EIA_API_KEY='...'
export FRED_API_KEY='...'
```

Do not put keys in `pmds/config.json`. On first use, each official loader writes its
fixed JSON snapshot under `pmds/data/official/` together with a `.sha256` integrity
sidecar. Later runs reuse and verify that file rather than downloading the latest data.

Run a one-series forecast audit first:

```bash
python pmds/compare.py --config pmds/config.json --forecast-audit
```

Then run the clean benchmark and robustness benchmark separately:

```bash
python pmds/compare.py --config pmds/config.json
python pmds/robustness.py --config pmds/config.json
```

The common quantile grid is 0.1 through 0.9, the point forecast is the median, and
all model families receive at most the latest 512 context observations. Dataset
identities and exposure classifications are recorded in `pmds/manifests/`.

Moirai 2.0 Small's checkpoint is licensed CC-BY-NC-4.0; use it only where that
non-commercial restriction is acceptable.
