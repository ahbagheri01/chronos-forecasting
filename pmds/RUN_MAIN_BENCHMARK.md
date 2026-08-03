# PMDS Main Benchmark Runbook

This document explains how to run the main PMDS forecasting benchmark:

```bash
python pmds/compare.py --config pmds/config.json
```

This is the long run that benchmarks Chronos, statistical models, AutoARIMA, Prophet, DeepAR, and seasonal naive baselines across the configured datasets. On the current configuration it took about 40-45 minutes on CPU.

## 1. Where To Run It

Run the command from the repository root:

```bash
cd /opt/Projects/PMDS/chronos-forecasting
```

This matters because the default paths in the script and config are relative to this directory, especially:

```text
pmds/config.json
pmds/results/
pmds/results/logs/
```

## 2. Activate The Environment

Use the project environment from `/opt/Projects/PMDS`:

```bash
source /opt/Projects/PMDS/genai/bin/activate
```

You can confirm the Python executable:

```bash
which python
```

Expected:

```text
/opt/Projects/PMDS/genai/bin/python
```

## 3. Check The Hugging Face Cache

The current config runs Hugging Face in offline mode and points cache paths to `/tmp/pmds-hf`:

```json
"HF_HOME": "/tmp/pmds-hf",
"HF_DATASETS_CACHE": "/tmp/pmds-hf/datasets",
"TRANSFORMERS_CACHE": "/tmp/pmds-hf/hub",
"HF_HUB_OFFLINE": "1",
"HF_DATASETS_OFFLINE": "1",
"TRANSFORMERS_OFFLINE": "1"
```

Before running, check that the cache exists:

```bash
ls /tmp/pmds-hf
ls /tmp/pmds-hf/datasets
ls /tmp/pmds-hf/hub
```

The benchmark needs cached copies of:

```text
datasets--autogluon--chronos_datasets
datasets--thuml--Time-Series-Library
models--amazon--chronos-t5-tiny
```

If `/tmp/pmds-hf` is missing, the run may fail because the config is offline. If the user-level Hugging Face cache exists, recreate the writable cache like this:

```bash
mkdir -p /tmp/pmds-hf/hub
cp -a ~/.cache/huggingface/datasets /tmp/pmds-hf/
cp -a ~/.cache/huggingface/hub/datasets--autogluon--chronos_datasets /tmp/pmds-hf/hub/
cp -a ~/.cache/huggingface/hub/datasets--thuml--Time-Series-Library /tmp/pmds-hf/hub/
cp -a ~/.cache/huggingface/hub/models--amazon--chronos-t5-tiny /tmp/pmds-hf/hub/
```

Why this is needed: the original home cache may be readable but not writable for lock files in this environment, while `/tmp/pmds-hf` is writable.

## 4. Run The Benchmark

Basic command:

```bash
python pmds/compare.py --config pmds/config.json
```

Equivalent explicit command without relying on the activated shell:

```bash
/opt/Projects/PMDS/genai/bin/python pmds/compare.py --config pmds/config.json
```

The script also supports `--help`:

```bash
python pmds/compare.py --help
```

Current CLI:

```text
usage: compare.py [-h] [--config CONFIG]

Run the config-driven PMDS forecast comparison.

options:
  -h, --help       show this help message and exit
  --config CONFIG  Path to the JSON experiment configuration.
```

## 5. What The Script Runs

The benchmark is driven by `pmds/config.json`.

Enabled datasets:

```text
chronos_m1_yearly
chronos_m4_hourly
chronos_weather
external_national_illness
external_traffic
external_psm
```

Enabled models:

```text
chronos_t5_tiny
seasonal_naive
ar_2
ma_2
arma_2_2
arima_2_1_2
auto_arima
prophet
deepar
```

Current total workload:

```text
71 forecast tasks x 9 enabled models = 639 model results
```

Evaluation metrics:

```text
mae
rmse
smape
mase
wql
```

## 6. Expected Runtime

On the current CPU setup, the full run took about 40-45 minutes.

The slowest parts are:

```text
external_psm + Prophet
external_psm + ARMA/MA/AutoARIMA
chronos_weather + ARMA/ARIMA/AutoARIMA
DeepAR training loops on CPU
```

DeepAR is expected to print many Lightning and PyTorch messages because it trains a small model for each task.

## 7. Expected Output Files

The run writes results to:

```text
pmds/results/compare_detailed.csv
pmds/results/compare_summary.csv
pmds/results/compare_status.json
pmds/results/logs/compare_<timestamp>.log
```

For the verified run, the log file was:

```text
pmds/results/logs/compare_20260803_192617.log
```

The script also saves checkpoints after each dataset, so partial progress appears in the CSV/status files while the run is still active.

## 8. How To Know It Worked

At the end, the console should show:

```text
Experiment completed | datasets=6 result_rows=639
```

The verified run completed with:

```text
6 completed datasets
639 detailed rows
54 summary rows
0 failed tasks
0 error rows
```

You can verify the files with:

```bash
python - <<'PY'
import csv, json
from collections import Counter
from pathlib import Path

root = Path("pmds/results")
status = json.loads((root / "compare_status.json").read_text())

with (root / "compare_detailed.csv").open(newline="") as f:
    detailed = list(csv.DictReader(f))

with (root / "compare_summary.csv").open(newline="") as f:
    summary = list(csv.DictReader(f))

errors = [
    row for row in detailed
    if (row.get("error_type") or "").strip() or (row.get("error") or "").strip()
]

print("completed datasets:", sum(1 for item in status if item.get("status") == "completed"))
print("dataset failures:", sum(int(item.get("failures") or 0) for item in status))
print("detailed rows:", len(detailed))
print("summary rows:", len(summary))
print("error rows:", len(errors))
print("rows by model:", dict(sorted(Counter(row["model"] for row in detailed).items())))
PY
```

For a clean full run, expect:

```text
completed datasets: 6
dataset failures: 0
detailed rows: 639
summary rows: 54
error rows: 0
```

Each model should have 71 rows.

## 9. Common Warnings

These warnings appeared during the verified run and did not indicate failure:

```text
torch.cuda: Can't initialize NVML
```

This happens because the run is configured for CPU but PyTorch probes CUDA.

```text
GluonTS/PyTorch: Using a non-tuple sequence for multidimensional indexing is deprecated
```

This is a dependency deprecation warning from GluonTS/PyTorch.

```text
validation_step but have no val_dataloader
```

This is emitted by Lightning during DeepAR training and is expected for this setup.

```text
prophet.plot: Importing plotly failed. Interactive plots will not work.
```

This only affects Prophet interactive plotting, not the benchmark forecasts.

## 10. After The Run

Generate plots:

```bash
python pmds/plot_results.py \
  --csv pmds/results/compare_detailed.csv \
  --output pmds/results/plots \
  --config pmds/config.json
```

Generate the static dataset/model/metric report:

```bash
python pmds/generate_report.py
```

Generated report files:

```text
pmds/results/plots/REPORT.md
pmds/results/plots/REPORT.html
```

## 11. Quick Troubleshooting

If imports fail, confirm the environment:

```bash
source /opt/Projects/PMDS/genai/bin/activate
which python
python -m py_compile pmds/*.py
```

If Hugging Face dataset/model loading fails, check `/tmp/pmds-hf` and the offline flags in `pmds/config.json`.

If the run is very slow, check which dataset/model is active in the console log. Long waits are normal on `external_psm` for Prophet, MA, ARMA, and AutoARIMA.

If the process exits early, inspect the newest log:

```bash
ls -t pmds/results/logs | head
```

Then search it:

```bash
rg -n "ERROR|Traceback|Model failed|Dataset failed|Exception" pmds/results/logs/<log-file>
```

## 12. Main Files

```text
pmds/compare.py       CLI entrypoint
pmds/config.json      Experiment configuration
pmds/pipeline.py      Benchmark orchestration
pmds/models.py        Model runners
pmds/metrics.py       Metric calculation
pmds/datasets.py      Dataset loading and task creation
pmds/outputs.py       CSV/status output writing
pmds/runtime.py       Logging, environment, seeding
```
