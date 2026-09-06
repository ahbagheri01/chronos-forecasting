"""Generate forecast and MCS panels for four presentation case studies.

The figures use only the saved forecasts and hypothesis-test outputs from the
11_12_3_rep experiment; no model inference or resampling is performed here.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter
import numpy as np


LATEX_DIR = Path(__file__).resolve().parent
RESULTS_DIR = LATEX_DIR.parent / "results" / "11_12_3_rep"
ASSET_DIR = LATEX_DIR / "assets"

INK = "#17212B"
MUTED = "#65717D"
HISTORY = "#9AA6B2"
LINE = "#D7E0E6"
PAPER = "#F5F8FA"
TEAL = "#087E8B"
CORAL = "#D95D39"
GOLD = "#B67D0D"
GREEN = "#2E7D5B"
PURPLE = "#68558A"

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

MODEL_COLORS = {
    "arima_2_1_2": GOLD,
    "chronos_t5_base": PURPLE,
    "deepar": CORAL,
    "moirai_2_0_small": CORAL,
    "prophet": MUTED,
    "seasonal_naive": GOLD,
    "timesfm_2_5_200m": TEAL,
}

CASES = (
    {
        "slug": "nn5-crps-mcs",
        "dataset": "chronos_nn5_weekly",
        "metric": "crps",
        "metric_label": "CRPS",
        "item_id": "28:target::origin=2",
        "history": 36,
        "models": (
            "timesfm_2_5_200m",
            "chronos_t5_base",
            "moirai_2_0_small",
            "seasonal_naive",
        ),
    },
    {
        "slug": "m1-wql-mcs",
        "dataset": "chronos_m1_yearly",
        "metric": "wql",
        "metric_label": "WQL",
        "item_id": "180:target::origin=2",
        "history": 10,
        "models": (
            "arima_2_1_2",
            "deepar",
            "timesfm_2_5_200m",
            "prophet",
        ),
    },
    {
        "slug": "elexon-wql-mcs",
        "dataset": "official_elexon_demand",
        "metric": "wql",
        "metric_label": "WQL",
        "item_id": "GB_NATIONAL_DEMAND::origin=2",
        "history": 48,
        "models": (
            "timesfm_2_5_200m",
            "chronos_t5_base",
            "moirai_2_0_small",
            "seasonal_naive",
        ),
    },
    {
        "slug": "traffic-mase-mcs",
        "dataset": "external_traffic",
        "metric": "mase",
        "metric_label": "MASE",
        "item_id": "0::origin=2",
        "history": 48,
        "models": (
            "moirai_2_0_small",
            "chronos_t5_base",
            "seasonal_naive",
            "timesfm_2_5_200m",
        ),
    },
)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def load_forecast_examples():
    selected = {(case["dataset"], case["item_id"]): case for case in CASES}
    contexts: dict[tuple[str, str], list[tuple[int, float]]] = defaultdict(list)
    with (RESULTS_DIR / "11_12_3_rep_contexts.csv").open(newline="") as handle:
        for row in csv.DictReader(handle):
            key = (row["dataset"], row["item_id"])
            if key in selected:
                contexts[key].append((int(row["step"]), float(row["actual"])))

    actual: dict[tuple[str, str], dict[int, float]] = defaultdict(dict)
    values = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))
    with (RESULTS_DIR / "11_12_3_rep_forecasts.csv").open(newline="") as handle:
        for row in csv.DictReader(handle):
            key = (row["dataset"], row["item_id"])
            case = selected.get(key)
            if case is None or row["model"] not in case["models"]:
                continue
            step = int(row["step"])
            actual[key][step] = float(row["actual"])
            for field in ("point_forecast", "q_10", "q_90"):
                values[key][row["model"]][field][step].append(float(row[field]))

    averaged = {}
    for key, model_values in values.items():
        averaged[key] = {
            model: {
                field: {step: float(np.mean(samples)) for step, samples in steps.items()}
                for field, steps in fields.items()
            }
            for model, fields in model_values.items()
        }
    return contexts, actual, averaged


def style_axis(axis) -> None:
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_color(LINE)
    axis.spines["bottom"].set_color(LINE)
    axis.tick_params(axis="both", colors=INK)
    axis.grid(color=LINE, linewidth=0.75)
    axis.set_axisbelow(True)


def compact_number(value: float, _position: int) -> str:
    absolute = abs(value)
    if absolute >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    if absolute >= 1_000:
        decimals = 1 if absolute < 10_000 else 0
        return f"{value / 1_000:.{decimals}f}k"
    if 0 < absolute < 0.01:
        return f"{value:.3f}"
    return f"{value:g}"


def plot_forecast(case, contexts, actual, predictions, empirical_best: str) -> None:
    key = (case["dataset"], case["item_id"])
    context = sorted(contexts[key])[-case["history"] :]
    future_steps = sorted(actual[key])

    figure, axis = plt.subplots(figsize=(6.6, 3.85))
    axis.axvspan(-0.5, max(future_steps) + 0.5, color=PAPER, zorder=0)
    axis.plot(
        [step for step, _ in context],
        [value for _, value in context],
        color=HISTORY,
        linewidth=1.7,
        label="History",
        zorder=2,
    )
    axis.plot(
        future_steps,
        [actual[key][step] for step in future_steps],
        color=INK,
        marker="o",
        markersize=3.2,
        linewidth=2.4,
        label="Actual",
        zorder=5,
    )

    if case["metric"] in {"wql", "crps"}:
        lower = [predictions[key][empirical_best]["q_10"][step] for step in future_steps]
        upper = [predictions[key][empirical_best]["q_90"][step] for step in future_steps]
        axis.fill_between(
            future_steps,
            lower,
            upper,
            color=MODEL_COLORS[empirical_best],
            alpha=0.11,
            linewidth=0,
            zorder=1,
        )

    markers = ("s", "D", "^", "v")
    for model, marker in zip(case["models"], markers):
        axis.plot(
            future_steps,
            [predictions[key][model]["point_forecast"][step] for step in future_steps],
            color=MODEL_COLORS[model],
            marker=marker,
            markersize=2.8,
            linewidth=1.7,
            label=MODEL_LABELS[model],
            zorder=3,
        )

    axis.axvline(-0.5, color=CORAL, linestyle="--", linewidth=1.1)
    axis.set_title("Representative forecast task", loc="left", color=INK, weight="bold", fontsize=13)
    axis.set_xlabel("Step relative to forecast origin", color=INK, fontsize=9.5)
    axis.set_ylabel("Observed value / point forecast", color=INK, fontsize=9.5)
    axis.yaxis.set_major_formatter(FuncFormatter(compact_number))
    axis.legend(
        frameon=False,
        fontsize=8.2,
        ncol=3,
        loc="upper left",
        columnspacing=0.9,
        handlelength=1.5,
        handletextpad=0.4,
    )
    if case["metric"] in {"wql", "crps"}:
        axis.text(
            0.99,
            0.02,
            f"Shading: 10--90% interval for {MODEL_LABELS[empirical_best]}",
            transform=axis.transAxes,
            ha="right",
            va="bottom",
            color=MUTED,
            fontsize=7.8,
        )
    style_axis(axis)
    axis.tick_params(axis="both", labelsize=8.8)
    figure.patch.set_facecolor("white")
    figure.tight_layout(pad=0.5)
    figure.savefig(ASSET_DIR / f"{case['slug']}-forecast.png", dpi=230, bbox_inches="tight")
    plt.close(figure)


def plot_mcs(case, rows: list[dict[str, str]]) -> None:
    rows = sorted(rows, key=lambda row: float(row["mean_loss"]))
    best = float(rows[0]["mean_loss"])
    relative_loss = np.array([(float(row["mean_loss"]) / best - 1) * 100 for row in rows])
    included = np.array([row["included_in_95pct_mcs"].lower() == "true" for row in rows])
    positions = np.arange(len(rows))

    figure, axis = plt.subplots(figsize=(4.35, 3.95))
    for y, x, keep in zip(positions, relative_loss, included):
        axis.hlines(y, 0, x, color=LINE, linewidth=1.2, zorder=1)
        if keep:
            axis.scatter(x, y, s=55, marker="o", color=GREEN, linewidth=1.7, zorder=3)
        else:
            axis.scatter(x, y, s=46, marker="x", color=CORAL, linewidth=1.8, zorder=3)

    axis.set_yticks(positions, [MODEL_LABELS[row["model"]] for row in rows])
    axis.invert_yaxis()
    axis.set_xlabel("Loss above the empirical best (%)", color=INK, fontsize=9.5)
    axis.set_title(
        f"95% MCS for {case['metric_label']}",
        loc="left",
        color=INK,
        weight="bold",
        fontsize=13.5,
        pad=21,
    )
    retained = int(included.sum())
    tasks = rows[0]["n_paired_tasks"]
    axis.text(
        0,
        1.005,
        f"{retained} of 11 models retained  |  {tasks} paired tasks",
        transform=axis.transAxes,
        ha="left",
        va="bottom",
        color=MUTED,
        fontsize=9.0,
    )
    legend_handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=GREEN,
               markeredgecolor=GREEN, markersize=7.5, label="Retained")
    ]
    if not included.all():
        legend_handles.append(
            Line2D([0], [0], marker="x", color=CORAL, linestyle="none",
                   markersize=7.5, label="Eliminated")
        )
    axis.legend(
        handles=legend_handles,
        frameon=False,
        fontsize=8.3,
        ncol=len(legend_handles),
        loc="upper right",
        columnspacing=0.9,
        handletextpad=0.35,
    )
    axis.grid(axis="x", color=LINE, linewidth=0.75)
    axis.grid(axis="y", visible=False)
    axis.tick_params(axis="y", length=0, labelsize=10.2, pad=4)
    axis.tick_params(axis="x", labelsize=8.5)
    style_axis(axis)
    axis.grid(axis="y", visible=False)
    figure.patch.set_facecolor("white")
    figure.tight_layout(pad=0.5)
    figure.savefig(ASSET_DIR / f"{case['slug']}-mcs.png", dpi=230, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    contexts, actual, predictions = load_forecast_examples()
    all_mcs_rows = read_rows(RESULTS_DIR / "hypothesis_tests" / "model_confidence_set.csv")
    for case in CASES:
        rows = [
            row
            for row in all_mcs_rows
            if row["dataset"] == case["dataset"] and row["metric"] == case["metric"]
        ]
        if len(rows) != 11:
            raise RuntimeError(f"Expected 11 MCS rows for {case['slug']}, found {len(rows)}")
        empirical_best = min(rows, key=lambda row: float(row["mean_loss"]))["model"]
        plot_forecast(case, contexts, actual, predictions, empirical_best)
        plot_mcs(case, rows)


if __name__ == "__main__":
    main()
