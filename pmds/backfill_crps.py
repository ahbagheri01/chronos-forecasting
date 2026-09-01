#!/usr/bin/env python
"""Backfill finite-quantile CRPS into a completed PMDS experiment."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import scoringrules as sr

from pmds.config import load_config
from pmds.metrics import summarize
from pmds.outputs import atomic_write_csv


KEY_COLUMNS = ["dataset", "series_id", "item_id", "origin", "model", "repetition", "seed"]
CRPS_NOTE = (
    "Nine-quantile approximation from q10-q90 via scoringrules.crps_quantile; "
    "original samples/full CDF were not persisted."
)


def quantile_columns(frame: pd.DataFrame) -> tuple[list[str], np.ndarray]:
    pairs = []
    for column in frame.columns:
        match = re.fullmatch(r"q_(\d+)", str(column))
        if match:
            pairs.append((int(match.group(1)) / 100.0, str(column)))
    pairs.sort()
    if not pairs:
        raise ValueError("Forecast file does not contain q_NN quantile columns")
    return [column for _, column in pairs], np.asarray([level for level, _ in pairs], dtype=np.float64)


def add_note(value: object) -> str:
    if isinstance(value, str) and value.strip():
        notes = json.loads(value)
    else:
        notes = {}
    notes["crps"] = CRPS_NOTE
    return json.dumps(notes, sort_keys=True)


def backfill(
    detailed_path: Path,
    forecast_path: Path,
    summary_path: Path,
    config_path: Path,
) -> tuple[int, float]:
    config = load_config(config_path)
    metrics = list(config["evaluation"]["metrics"])
    if "crps" not in metrics:
        raise ValueError(f"CRPS is not enabled in {config_path}")

    detailed = pd.read_csv(detailed_path)
    forecasts = pd.read_csv(forecast_path)
    q_columns, q_levels = quantile_columns(forecasts)
    configured = np.asarray(config["evaluation"]["quantiles"], dtype=np.float64)
    if not np.allclose(q_levels, configured):
        raise ValueError(f"Saved quantiles {q_levels.tolist()} do not match config {configured.tolist()}")

    forecasts["_crps"] = np.asarray(
        sr.crps_quantile(
            forecasts["actual"].to_numpy(dtype=np.float64),
            forecasts[q_columns].to_numpy(dtype=np.float64),
            q_levels,
            backend="numpy",
        ),
        dtype=np.float64,
    )
    task_crps = forecasts.groupby(KEY_COLUMNS, dropna=False, as_index=False)["_crps"].mean()
    task_crps = task_crps.rename(columns={"_crps": "crps"})

    detailed = detailed.drop(columns="crps", errors="ignore").merge(
        task_crps,
        on=KEY_COLUMNS,
        how="left",
        validate="one_to_one",
    )
    successful = detailed["error_type"].fillna("").astype(str).eq("")
    if detailed.loc[successful, "crps"].isna().any():
        missing = int(detailed.loc[successful, "crps"].isna().sum())
        raise ValueError(f"CRPS could not be matched to {missing} successful task rows")

    forecast_counts = forecasts.groupby(KEY_COLUMNS, dropna=False).size().rename("_horizon").reset_index()
    check = detailed.merge(forecast_counts, on=KEY_COLUMNS, how="left", validate="one_to_one")
    validation_rows = successful & check["wql_loss_sum"].notna() & check["_horizon"].notna()
    expected = check.loc[validation_rows, "wql_loss_sum"] / check.loc[validation_rows, "_horizon"]
    observed = check.loc[validation_rows, "crps"]
    max_difference = float(np.nanmax(np.abs(observed.to_numpy() - expected.to_numpy())))
    if not np.allclose(observed, expected, rtol=1e-6, atol=1e-8, equal_nan=True):
        raise ValueError(f"CRPS/WQL identity validation failed (maximum difference {max_difference:.6g})")

    detailed["metric_notes"] = detailed["metric_notes"].map(add_note)
    metric_position = detailed.columns.get_loc("mase") + 1
    crps = detailed.pop("crps")
    detailed.insert(metric_position, "crps", crps)
    detailed["error_type"] = detailed["error_type"].fillna("")
    detailed["error"] = detailed["error"].fillna("")

    summary = summarize(detailed, metrics)
    atomic_write_csv(detailed, detailed_path)
    atomic_write_csv(summary, summary_path)
    return len(detailed), max_difference


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--detailed", type=Path, required=True)
    parser.add_argument("--forecasts", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    rows, difference = backfill(args.detailed, args.forecasts, args.summary, args.config)
    print(f"Backfilled CRPS for {rows} detailed rows; max CRPS/WQL identity difference={difference:.6g}")


if __name__ == "__main__":
    main()
