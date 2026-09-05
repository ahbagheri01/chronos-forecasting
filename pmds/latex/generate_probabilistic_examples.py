"""Generate WQL and CRPS examples with predictive intervals."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


LATEX_DIR = Path(__file__).resolve().parent
RESULTS_DIR = LATEX_DIR.parent / "results" / "11_12_3_rep"

MODEL_LABELS = {
    "arima_2_1_2": "ARIMA(2,1,2)",
    "auto_arima": "Auto-ARIMA",
    "chronos_t5_base": "Chronos Base",
    "chronos_t5_mini": "Chronos Mini",
    "chronos_t5_small": "Chronos Small",
    "chronos_t5_tiny": "Chronos Tiny",
    "deepar": "DeepAR",
    "moirai_2_0_small": "Moirai",
    "prophet": "Prophet",
    "seasonal_naive": "Seasonal Naive",
    "timesfm_2_5_200m": "TimesFM",
}

FOUNDATION_MODELS = {
    "chronos_t5_base", "chronos_t5_mini", "chronos_t5_small",
    "chronos_t5_tiny", "moirai_2_0_small", "timesfm_2_5_200m",
}

CASES = (
    {
        "dataset": "chronos_m4_hourly",
        "dataset_title": "M4 Hourly",
        "metric": "crps",
        "metric_label": "CRPS",
        "winner": "chronos_t5_base",
        "comparator": "prophet",
        "item_id": "304:target::origin=1",
        "output": "m4-crps-separated-case.png",
        "history": 48,
        "gap_text": "38.1% lower",
    },
    {
        "dataset": "chronos_m3_quarterly",
        "dataset_title": "M3 Quarterly",
        "metric": "wql",
        "metric_label": "WQL",
        "winner": "timesfm_2_5_200m",
        "comparator": "auto_arima",
        "item_id": "337:target::origin=1",
        "output": "m3-wql-close-case.png",
        "history": 24,
        "gap_text": "8.74% lower",
    },
)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def forecast_values(contexts, forecasts, case):
    dataset, item_id = case["dataset"], case["item_id"]
    selected = {case["winner"], case["comparator"]}
    context = [
        (int(row["step"]), float(row["actual"]))
        for row in contexts
        if row["dataset"] == dataset and row["item_id"] == item_id
    ]
    actual = {}
    values = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for row in forecasts:
        if row["dataset"] == dataset and row["item_id"] == item_id and row["model"] in selected:
            step = int(row["step"])
            actual[step] = float(row["actual"])
            for field in ("point_forecast", "q_10", "q_90"):
                values[row["model"]][field][step].append(float(row[field]))
    averaged = {
        model: {
            field: {step: float(np.mean(samples)) for step, samples in steps.items()}
            for field, steps in fields.items()
        }
        for model, fields in values.items()
    }
    return context, actual, averaged


def plot_case(case, repetitions, contexts, forecasts):
    ink, muted = "#17212B", "#65717D"
    teal, purple, gold, line = "#087E8B", "#68558A", "#B67D0D", "#D7E0E6"
    rows = sorted(
        (
            row for row in repetitions
            if row["dataset"] == case["dataset"] and row["metric"] == case["metric"]
        ),
        key=lambda row: float(row["repetition_mean"]),
    )
    context, actual, predictions = forecast_values(contexts, forecasts, case)
    winner, comparator = case["winner"], case["comparator"]

    fig, (metric_ax, forecast_ax) = plt.subplots(
        1, 2, figsize=(13.2, 4.3), gridspec_kw={"width_ratios": [1.12, 1.25], "wspace": 0.27}
    )
    positions = np.arange(len(rows))
    for x, row in zip(positions, rows):
        model = row["model"]
        mean, sd = float(row["repetition_mean"]), float(row["repetition_sample_sd"])
        if model == winner:
            color, size, zorder = teal, 8, 4
        elif model == comparator:
            color, size, zorder = gold, 8, 4
        elif model in FOUNDATION_MODELS:
            color, size, zorder = purple, 6, 3
        else:
            color, size, zorder = muted, 6, 2
        metric_ax.errorbar(
            x, mean, yerr=sd, fmt="o", color=color, ecolor=color,
            markersize=size, capsize=3, linewidth=1.5, zorder=zorder,
        )
        if model in {winner, comparator}:
            metric_ax.annotate(
                f"{mean:.3g}", (x, mean), xytext=(0, 8), textcoords="offset points",
                ha="center", fontsize=8, color=color, weight="bold",
            )
    metric_ax.set_xticks(positions, [MODEL_LABELS[row["model"]] for row in rows])
    metric_ax.tick_params(axis="x", rotation=48, labelsize=7.5, colors=ink)
    for label in metric_ax.get_xticklabels():
        label.set_ha("right")
    metric_ax.set_ylabel(f"Aggregate {case['metric_label']} (lower is better)", color=ink)
    metric_ax.set_title(
        f"{case['dataset_title']}: {case['metric_label']} by model",
        loc="left", color=ink, weight="bold",
    )
    metric_ax.grid(axis="y", color=line, linewidth=0.8)
    metric_ax.set_axisbelow(True)
    metric_ax.tick_params(axis="y", colors=ink, labelsize=9)
    metric_ax.text(
        0.98, 0.94, case["gap_text"], transform=metric_ax.transAxes,
        ha="right", va="top", fontsize=9, color=teal, weight="bold",
    )

    context = sorted(context)[-case["history"]:]
    context_x = [step for step, _ in context]
    context_y = [value for _, value in context]
    future_x = sorted(actual)
    forecast_ax.plot(context_x, context_y, color="#9AA6B2", marker="o", markersize=3, label="History")
    forecast_ax.plot(
        future_x, [actual[step] for step in future_x], color=ink,
        marker="o", linewidth=2.2, label="Actual",
    )
    for model, color, marker in ((winner, teal, "o"), (comparator, gold, "s")):
        point = [predictions[model]["point_forecast"][step] for step in future_x]
        lower = [predictions[model]["q_10"][step] for step in future_x]
        upper = [predictions[model]["q_90"][step] for step in future_x]
        forecast_ax.fill_between(future_x, lower, upper, color=color, alpha=0.10)
        forecast_ax.plot(
            future_x, point, color=color, marker=marker, linewidth=2,
            label=MODEL_LABELS[model],
        )
    forecast_ax.axvline(-0.5, color="#D95D39", linestyle="--", linewidth=1.2)
    forecast_ax.axvspan(-0.5, max(future_x) + 0.5, color="#F5F8FA", zorder=0)
    forecast_ax.set_title("Representative probabilistic forecast", loc="left", color=ink, weight="bold")
    forecast_ax.set_xlabel("Step relative to forecast origin", color=ink)
    forecast_ax.set_ylabel("Observed value / forecast", color=ink)
    forecast_ax.grid(color=line, linewidth=0.8)
    forecast_ax.set_axisbelow(True)
    forecast_ax.tick_params(axis="both", colors=ink, labelsize=9)
    forecast_ax.legend(frameon=False, fontsize=8.5, ncol=4, loc="upper left")
    for axis in (metric_ax, forecast_ax):
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.spines["left"].set_color(line)
        axis.spines["bottom"].set_color(line)
    fig.patch.set_facecolor("white")
    fig.savefig(LATEX_DIR / "assets" / case["output"], dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    repetitions = read_rows(RESULTS_DIR / "hypothesis_tests" / "repetition_variability.csv")
    contexts = read_rows(RESULTS_DIR / "11_12_3_rep_contexts.csv")
    forecasts = read_rows(RESULTS_DIR / "11_12_3_rep_forecasts.csv")
    for case in CASES:
        plot_case(case, repetitions, contexts, forecasts)


if __name__ == "__main__":
    main()
