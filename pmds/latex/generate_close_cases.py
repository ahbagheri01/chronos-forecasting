"""Generate model-metric and forecast panels for the three closest contests."""

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
    "chronos_t5_base",
    "chronos_t5_mini",
    "chronos_t5_small",
    "chronos_t5_tiny",
    "moirai_2_0_small",
    "timesfm_2_5_200m",
}

CASES = (
    {
        "dataset": "external_psm",
        "title": "PSM",
        "winner": "chronos_t5_small",
        "comparator": "arima_2_1_2",
        "item_id": "feature_18::origin=2",
        "output": "psm-mase-separated-case.png",
        "history": 36,
        "gap_text": "34.2% lower",
    },
    {
        "dataset": "chronos_m1_yearly",
        "title": "M1 Yearly",
        "winner": "timesfm_2_5_200m",
        "comparator": "prophet",
        "item_id": "156:target::origin=2",
        "output": "m1-mase-close-case.png",
        "history": 18,
    },
)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def metric_rows(all_rows: list[dict[str, str]], dataset: str) -> list[dict[str, str]]:
    return sorted(
        (row for row in all_rows if row["dataset"] == dataset and row["metric"] == "mase"),
        key=lambda row: float(row["repetition_mean"]),
    )


def forecast_example(
    contexts: list[dict[str, str]],
    forecasts: list[dict[str, str]],
    case: dict[str, object],
) -> tuple[list[tuple[int, float]], dict[int, float], dict[str, dict[int, float]]]:
    dataset = str(case["dataset"])
    item_id = str(case["item_id"])
    selected_models = {str(case["winner"]), str(case["comparator"])}

    context = [
        (int(row["step"]), float(row["actual"]))
        for row in contexts
        if row["dataset"] == dataset and row["item_id"] == item_id
    ]
    actual: dict[int, float] = {}
    repetitions: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in forecasts:
        if (
            row["dataset"] == dataset
            and row["item_id"] == item_id
            and row["model"] in selected_models
        ):
            step = int(row["step"])
            actual[step] = float(row["actual"])
            repetitions[row["model"]][step].append(float(row["point_forecast"]))

    averaged = {
        model: {step: float(np.mean(values)) for step, values in steps.items()}
        for model, steps in repetitions.items()
    }
    return context, actual, averaged


def plot_case(
    case: dict[str, object],
    all_metric_rows: list[dict[str, str]],
    contexts: list[dict[str, str]],
    forecasts: list[dict[str, str]],
) -> None:
    ink = "#17212B"
    muted = "#65717D"
    teal = "#087E8B"
    purple = "#68558A"
    gold = "#B67D0D"
    line = "#D7E0E6"

    rows = metric_rows(all_metric_rows, str(case["dataset"]))
    context, actual, predictions = forecast_example(contexts, forecasts, case)
    winner = str(case["winner"])
    comparator = str(case["comparator"])
    winner_mean = next(float(row["repetition_mean"]) for row in rows if row["model"] == winner)
    comparator_mean = next(float(row["repetition_mean"]) for row in rows if row["model"] == comparator)
    gap = (comparator_mean - winner_mean) / winner_mean * 100

    fig, (metric_ax, forecast_ax) = plt.subplots(
        1,
        2,
        figsize=(13.2, 4.3),
        gridspec_kw={"width_ratios": [1.12, 1.25], "wspace": 0.27},
    )

    x_positions = np.arange(len(rows))
    for x, row in zip(x_positions, rows):
        model = row["model"]
        mean = float(row["repetition_mean"])
        sd = float(row["repetition_sample_sd"])
        if model == winner:
            color, size, zorder = teal, 8, 4
        elif model == comparator:
            color, size, zorder = gold, 8, 4
        elif model in FOUNDATION_MODELS:
            color, size, zorder = purple, 6, 3
        else:
            color, size, zorder = muted, 6, 2
        metric_ax.errorbar(
            x,
            mean,
            yerr=sd,
            fmt="o",
            color=color,
            ecolor=color,
            markersize=size,
            capsize=3,
            linewidth=1.5,
            zorder=zorder,
        )
        if model in {winner, comparator}:
            metric_ax.annotate(
                f"{mean:.3g}",
                (x, mean),
                xytext=(0, 8),
                textcoords="offset points",
                ha="center",
                fontsize=8,
                color=color,
                weight="bold",
            )

    metric_ax.set_xticks(x_positions, [MODEL_LABELS[row["model"]] for row in rows])
    metric_ax.tick_params(axis="x", rotation=48, labelsize=7.5, colors=ink)
    for label in metric_ax.get_xticklabels():
        label.set_ha("right")
    metric_ax.set_ylabel("Aggregate MASE (lower is better)", color=ink)
    metric_ax.set_title(f"{case['title']}: MASE by model", loc="left", color=ink, weight="bold")
    metric_ax.grid(axis="y", color=line, linewidth=0.8)
    metric_ax.set_axisbelow(True)
    metric_ax.tick_params(axis="y", colors=ink, labelsize=9)
    metric_ax.text(
        0.98,
        0.94,
        str(case.get("gap_text", f"{gap:.2f}% gap")),
        transform=metric_ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        color=teal,
        weight="bold",
    )

    context = sorted(context)[-int(case["history"]):]
    context_x = [step for step, _ in context]
    context_y = [value for _, value in context]
    future_x = sorted(actual)
    actual_y = [actual[step] for step in future_x]
    winner_y = [predictions[winner][step] for step in future_x]
    comparator_y = [predictions[comparator][step] for step in future_x]

    forecast_ax.plot(context_x, context_y, color="#9AA6B2", marker="o", markersize=3, label="History")
    forecast_ax.plot(future_x, actual_y, color=ink, marker="o", linewidth=2.2, label="Actual")
    forecast_ax.plot(
        future_x, winner_y, color=teal, marker="o", linewidth=2, label=MODEL_LABELS[winner]
    )
    forecast_ax.plot(
        future_x,
        comparator_y,
        color=gold,
        marker="s",
        linewidth=2,
        label=MODEL_LABELS[comparator],
    )
    forecast_ax.axvline(-0.5, color="#D95D39", linestyle="--", linewidth=1.2)
    forecast_ax.axvspan(-0.5, max(future_x) + 0.5, color="#F5F8FA", zorder=0)
    forecast_ax.set_title("Representative forecast", loc="left", color=ink, weight="bold")
    forecast_ax.set_xlabel("Step relative to forecast origin", color=ink)
    forecast_ax.set_ylabel("Observed value / point forecast", color=ink)
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
    fig.savefig(LATEX_DIR / "assets" / str(case["output"]), dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    all_metric_rows = read_rows(RESULTS_DIR / "hypothesis_tests" / "repetition_variability.csv")
    contexts = read_rows(RESULTS_DIR / "11_12_3_rep_contexts.csv")
    forecasts = read_rows(RESULTS_DIR / "11_12_3_rep_forecasts.csv")
    for case in CASES:
        plot_case(case, all_metric_rows, contexts, forecasts)


if __name__ == "__main__":
    main()
