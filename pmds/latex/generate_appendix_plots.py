"""Generate appendix figures from the saved 11_12_3_rep results.

The script only summarizes existing CSV outputs. It does not rerun forecasts,
bootstrap the MCS, or perform any additional simulation.
"""

from __future__ import annotations

import csv
import os
from collections import defaultdict
from pathlib import Path
from statistics import mean, median

os.environ.setdefault("MPLCONFIGDIR", "/tmp/pmds-matplotlib-cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from matplotlib.ticker import FuncFormatter
import numpy as np


LATEX_DIR = Path(__file__).resolve().parent
RESULTS_DIR = LATEX_DIR.parent / "results" / "11_12_3_rep"
ASSET_DIR = LATEX_DIR / "assets"

INK = "#17212B"
MUTED = "#65717D"
LINE = "#D7E0E6"
PAPER = "#F5F8FA"
TEAL = "#087E8B"
TEAL_LIGHT = "#DCEFF0"
CORAL = "#D95D39"
CORAL_LIGHT = "#F8E4DE"
GOLD = "#B67D0D"
GOLD_LIGHT = "#F6ECD2"
GREEN = "#2E7D5B"
GREEN_LIGHT = "#DDEEE6"
PURPLE = "#68558A"
PURPLE_LIGHT = "#E9E3F2"

METRICS = ("mase", "wql", "crps")
METRIC_LABELS = {"mase": "MASE", "wql": "WQL", "crps": "CRPS"}
METRIC_COLORS = {"mase": TEAL, "wql": PURPLE, "crps": GREEN}

FOUNDATION_MODELS = {
    "chronos_t5_tiny",
    "chronos_t5_mini",
    "chronos_t5_small",
    "chronos_t5_base",
    "timesfm_2_5_200m",
    "moirai_2_0_small",
}

MODEL_ORDER = (
    "timesfm_2_5_200m",
    "chronos_t5_small",
    "chronos_t5_base",
    "moirai_2_0_small",
    "chronos_t5_mini",
    "chronos_t5_tiny",
    "arima_2_1_2",
    "auto_arima",
    "deepar",
    "seasonal_naive",
    "prophet",
)

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

DATASET_ORDER = (
    "chronos_m1_yearly",
    "chronos_m4_hourly",
    "chronos_weather",
    "chronos_m3_quarterly",
    "chronos_car_parts",
    "chronos_nn5_weekly",
    "external_national_illness",
    "external_traffic",
    "external_psm",
    "official_elexon_demand",
    "official_usgs_streamflow",
    "official_bls_macro",
)

DATASET_LABELS = {
    "chronos_car_parts": "Car Parts",
    "chronos_m1_yearly": "M1 Yearly",
    "chronos_m3_quarterly": "M3 Quarterly",
    "chronos_m4_hourly": "M4 Hourly",
    "chronos_nn5_weekly": "NN5 Weekly",
    "chronos_weather": "Weather",
    "external_national_illness": "National Illness",
    "external_psm": "PSM",
    "external_traffic": "Traffic",
    "official_bls_macro": "BLS Macro",
    "official_elexon_demand": "Elexon Demand",
    "official_usgs_streamflow": "USGS Streamflow",
}


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def style_axis(axis, *, grid_axis: str = "x") -> None:
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_color(LINE)
    axis.spines["bottom"].set_color(LINE)
    axis.tick_params(axis="both", colors=INK, labelsize=9)
    axis.grid(axis=grid_axis, color=LINE, linewidth=0.8)
    axis.set_axisbelow(True)


def save(figure: plt.Figure, name: str) -> None:
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        ASSET_DIR / name,
        dpi=220,
        bbox_inches="tight",
        facecolor="white",
        pad_inches=0.08,
    )
    plt.close(figure)


def grouped_summary(rows: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["dataset"]].append(row)
    return grouped


def plot_family_gap(summary_rows: list[dict[str, str]]) -> None:
    grouped = grouped_summary(summary_rows)
    figure, axis = plt.subplots(figsize=(10.8, 5.35))
    axis.axvspan(-35, 0, color=GOLD_LIGHT, alpha=0.72, zorder=0)
    axis.axvspan(0, 75, color=TEAL_LIGHT, alpha=0.60, zorder=0)
    offsets = {"mase": -0.22, "wql": 0.0, "crps": 0.22}
    markers = {"mase": "o", "wql": "s", "crps": "D"}

    for row_index, dataset in enumerate(DATASET_ORDER):
        group = grouped[dataset]
        advantages = []
        for metric in METRICS:
            best_foundation = min(
                float(row[metric]) for row in group if row["model"] in FOUNDATION_MODELS
            )
            best_baseline = min(
                float(row[metric]) for row in group if row["model"] not in FOUNDATION_MODELS
            )
            advantage = 100.0 * (best_baseline - best_foundation) / best_baseline
            advantages.append(advantage)
            axis.scatter(
                advantage,
                row_index + offsets[metric],
                s=52,
                marker=markers[metric],
                color=METRIC_COLORS[metric],
                edgecolor="white",
                linewidth=0.7,
                zorder=3,
            )
        axis.plot(advantages, [row_index] * len(advantages), color=LINE, linewidth=1.2, zorder=1)

    axis.axvline(0, color=INK, linewidth=1.2)
    axis.set_xlim(-35, 75)
    axis.set_ylim(-0.7, len(DATASET_ORDER) - 0.3)
    axis.invert_yaxis()
    axis.set_yticks(range(len(DATASET_ORDER)), [DATASET_LABELS[d] for d in DATASET_ORDER])
    axis.set_xlabel(
        "Advantage of the best foundation model over the best baseline (%)",
        color=INK,
        fontsize=10,
    )
    axis.text(-17.5, -0.95, "baseline lead", ha="center", color=GOLD, fontsize=9, weight="bold")
    axis.text(37.5, -0.95, "foundation-model lead", ha="center", color=TEAL, fontsize=9, weight="bold")
    axis.legend(
        handles=[
            Line2D([0], [0], marker="o", color="none", markerfacecolor=TEAL, markeredgecolor="white", label="MASE", markersize=7),
            Line2D([0], [0], marker="s", color="none", markerfacecolor=PURPLE, markeredgecolor="white", label="WQL", markersize=7),
            Line2D([0], [0], marker="D", color="none", markerfacecolor=GREEN, markeredgecolor="white", label="CRPS", markersize=6.5),
        ],
        loc="lower right",
        frameon=False,
        ncol=3,
        fontsize=9,
    )
    style_axis(axis)
    figure.tight_layout()
    save(figure, "appendix-family-loss-gap.png")


def rank_values(summary_rows: list[dict[str, str]]) -> dict[str, list[float]]:
    grouped = grouped_summary(summary_rows)
    ranks: dict[str, list[float]] = defaultdict(list)
    for dataset in DATASET_ORDER:
        group = grouped[dataset]
        for metric in METRICS:
            ordered = sorted(group, key=lambda row: float(row[metric]))
            for rank, row in enumerate(ordered, start=1):
                ranks[row["model"]].append(float(rank))
    return ranks


def plot_rank_distribution(summary_rows: list[dict[str, str]]) -> None:
    ranks = rank_values(summary_rows)
    order = sorted(MODEL_ORDER, key=lambda model: mean(ranks[model]))
    figure, axis = plt.subplots(figsize=(10.8, 5.35))
    axis.axvspan(0.5, 3.5, color=TEAL_LIGHT, alpha=0.65, zorder=0)

    for row_index, model in enumerate(order):
        values = np.asarray(ranks[model])
        family_color = PURPLE if model in FOUNDATION_MODELS else GOLD
        box = axis.boxplot(
            values,
            positions=[row_index],
            vert=False,
            widths=0.52,
            patch_artist=True,
            showfliers=False,
            medianprops={"color": INK, "linewidth": 1.6},
            whiskerprops={"color": MUTED, "linewidth": 1.0},
            capprops={"color": MUTED, "linewidth": 1.0},
            boxprops={"facecolor": family_color, "alpha": 0.18, "edgecolor": family_color, "linewidth": 1.2},
        )
        _ = box
        jitter = np.linspace(-0.17, 0.17, len(values))
        axis.scatter(values, row_index + jitter, color=family_color, alpha=0.46, s=14, linewidth=0, zorder=2)
        average = mean(values)
        axis.scatter(average, row_index, marker="D", s=48, color=family_color, edgecolor="white", linewidth=0.8, zorder=4)
        axis.text(11.25, row_index, f"{average:.1f}", va="center", ha="left", color=INK, fontsize=8.5, weight="bold")

    axis.set_xlim(0.5, 11.8)
    axis.set_xticks(range(1, 12))
    axis.set_ylim(-0.7, len(order) - 0.3)
    axis.invert_yaxis()
    axis.set_yticks(range(len(order)), [MODEL_LABELS[m] for m in order])
    axis.set_xlabel("Empirical rank in each dataset-loss comparison (1 = best)", color=INK, fontsize=10)
    axis.text(11.25, -0.75, "mean", ha="left", color=MUTED, fontsize=8.5, weight="bold")
    axis.legend(
        handles=[
            Patch(facecolor=PURPLE_LIGHT, edgecolor=PURPLE, label="Foundation model"),
            Patch(facecolor=GOLD_LIGHT, edgecolor=GOLD, label="Baseline"),
            Line2D([0], [0], marker="D", color="none", markerfacecolor=INK, label="Mean rank", markersize=6),
        ],
        loc="lower right",
        frameon=False,
        ncol=3,
        fontsize=8.8,
    )
    style_axis(axis)
    figure.tight_layout()
    save(figure, "appendix-rank-distribution.png")


def plot_mcs_heatmap(mcs_rows: list[dict[str, str]]) -> None:
    selected = [
        row
        for row in mcs_rows
        if row["metric"] in METRICS and row["status"] == "ok"
    ]
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in selected:
        grouped[(row["dataset"], row["metric"])].append(row)

    sizes = np.zeros((len(DATASET_ORDER), len(METRICS)))
    separated = np.zeros_like(sizes, dtype=bool)
    for row_index, dataset in enumerate(DATASET_ORDER):
        for column_index, metric in enumerate(METRICS):
            group = grouped[(dataset, metric)]
            sizes[row_index, column_index] = sum(row["included_in_95pct_mcs"] == "True" for row in group)
            winner = next(row for row in group if row["empirical_winner"] == "True")
            winner_is_foundation = winner["model"] in FOUNDATION_MODELS
            other = [row for row in group if (row["model"] in FOUNDATION_MODELS) != winner_is_foundation]
            best_other = min(other, key=lambda row: float(row["mean_loss"]))
            separated[row_index, column_index] = best_other["included_in_95pct_mcs"] != "True"

    cmap = LinearSegmentedColormap.from_list("mcs_size", [GREEN_LIGHT, TEAL_LIGHT, PURPLE_LIGHT, CORAL_LIGHT])
    figure, axis = plt.subplots(figsize=(7.8, 5.45))
    image = axis.imshow(sizes, cmap=cmap, vmin=1, vmax=11, aspect="auto")
    for row_index in range(sizes.shape[0]):
        for column_index in range(sizes.shape[1]):
            axis.text(column_index, row_index, f"{int(sizes[row_index, column_index])}", ha="center", va="center", color=INK, fontsize=10, weight="bold")
            if separated[row_index, column_index]:
                axis.add_patch(
                    Rectangle(
                        (column_index - 0.47, row_index - 0.47),
                        0.94,
                        0.94,
                        fill=False,
                        edgecolor=TEAL,
                        linewidth=2.4,
                    )
                )

    axis.set_xticks(range(len(METRICS)), [METRIC_LABELS[m] for m in METRICS], fontsize=10, weight="bold")
    axis.set_yticks(range(len(DATASET_ORDER)), [DATASET_LABELS[d] for d in DATASET_ORDER], fontsize=9)
    axis.tick_params(length=0, colors=INK)
    for spine in axis.spines.values():
        spine.set_visible(False)
    colorbar = figure.colorbar(image, ax=axis, fraction=0.045, pad=0.04)
    colorbar.set_label("Models retained in the 95% MCS", color=INK, fontsize=9)
    colorbar.set_ticks([1, 3, 5, 7, 9, 11])
    colorbar.ax.tick_params(labelsize=8, colors=INK)
    axis.legend(
        handles=[Patch(facecolor="none", edgecolor=TEAL, linewidth=2.2, label="Best other-family model excluded")],
        loc="lower center",
        bbox_to_anchor=(0.5, -0.12),
        frameon=False,
        fontsize=8.8,
    )
    figure.tight_layout()
    save(figure, "appendix-mcs-size-heatmap.png")


def plot_calibration(calibration_rows: list[dict[str, str]]) -> None:
    errors: dict[str, list[float]] = defaultdict(list)
    for row in calibration_rows:
        if (
            row["calibration_type"] == "central_interval_coverage"
            and row["level"] == "10-90"
            and row["absolute_coverage_error"]
        ):
            errors[row["model"]].append(100.0 * float(row["absolute_coverage_error"]))

    order = sorted(MODEL_ORDER, key=lambda model: median(errors[model]))
    figure, axis = plt.subplots(figsize=(10.8, 5.35))
    axis.axvspan(0, 5, color=GREEN_LIGHT, alpha=0.70, zorder=0)
    for row_index, model in enumerate(order):
        values = np.asarray(errors[model])
        family_color = PURPLE if model in FOUNDATION_MODELS else GOLD
        axis.boxplot(
            values,
            positions=[row_index],
            vert=False,
            widths=0.52,
            patch_artist=True,
            showfliers=False,
            medianprops={"color": INK, "linewidth": 1.6},
            whiskerprops={"color": MUTED, "linewidth": 1.0},
            capprops={"color": MUTED, "linewidth": 1.0},
            boxprops={"facecolor": family_color, "alpha": 0.18, "edgecolor": family_color, "linewidth": 1.2},
        )
        jitter = np.linspace(-0.16, 0.16, len(values))
        axis.scatter(values, row_index + jitter, color=family_color, alpha=0.48, s=18, linewidth=0, zorder=2)
        middle = median(values)
        axis.scatter(middle, row_index, marker="D", s=48, color=family_color, edgecolor="white", linewidth=0.8, zorder=4)
        axis.text(82.0, row_index, f"{middle:.1f}", va="center", ha="right", color=INK, fontsize=8.5, weight="bold")

    axis.set_xlim(-1, 84)
    axis.set_ylim(-0.7, len(order) - 0.3)
    axis.invert_yaxis()
    axis.set_yticks(range(len(order)), [MODEL_LABELS[m] for m in order])
    axis.set_xlabel("Absolute error of observed 80% interval coverage (percentage points)", color=INK, fontsize=10)
    axis.xaxis.set_major_formatter(FuncFormatter(lambda value, _position: f"{value:.0f}"))
    axis.text(82.0, -0.75, "median", ha="right", color=MUTED, fontsize=8.5, weight="bold")
    axis.legend(
        handles=[
            Patch(facecolor=PURPLE_LIGHT, edgecolor=PURPLE, label="Foundation model"),
            Patch(facecolor=GOLD_LIGHT, edgecolor=GOLD, label="Baseline"),
            Line2D([0], [0], marker="D", color="none", markerfacecolor=INK, label="Median across datasets", markersize=6),
        ],
        loc="lower right",
        frameon=False,
        ncol=3,
        fontsize=8.8,
    )
    style_axis(axis)
    figure.tight_layout()
    save(figure, "appendix-calibration-distribution.png")


def plot_speed_accuracy(summary_rows: list[dict[str, str]]) -> None:
    ranks = rank_values(summary_rows)
    elapsed: dict[str, list[float]] = defaultdict(list)
    for row in summary_rows:
        tasks = float(row["successful_tasks"])
        if tasks > 0:
            elapsed[row["model"]].append(float(row["total_duration_seconds"]) / tasks)

    figure, axis = plt.subplots(figsize=(10.8, 5.35))
    label_offsets = {
        "timesfm_2_5_200m": (7, -13),
        "chronos_t5_small": (7, -5),
        "chronos_t5_base": (7, 8),
        "moirai_2_0_small": (7, -12),
        "chronos_t5_mini": (7, 8),
        "chronos_t5_tiny": (7, 8),
        "arima_2_1_2": (7, -11),
        "auto_arima": (7, 8),
        "deepar": (-7, -12),
        "seasonal_naive": (7, -11),
        "prophet": (7, 8),
    }
    for model in MODEL_ORDER:
        x_value = median(elapsed[model])
        y_value = mean(ranks[model])
        family_color = PURPLE if model in FOUNDATION_MODELS else GOLD
        marker = "o" if model in FOUNDATION_MODELS else "s"
        axis.scatter(x_value, y_value, s=92, marker=marker, color=family_color, edgecolor="white", linewidth=0.9, zorder=3)
        dx, dy = label_offsets[model]
        axis.annotate(
            MODEL_LABELS[model],
            (x_value, y_value),
            xytext=(dx, dy),
            textcoords="offset points",
            ha="right" if dx < 0 else "left",
            va="center",
            color=INK,
            fontsize=8.4,
        )

    axis.set_xscale("log")
    axis.set_xlim(0.0008, 55)
    axis.set_ylim(9.6, 2.1)
    axis.set_xlabel("Median recorded runtime per successful task (seconds, log scale)", color=INK, fontsize=10)
    axis.set_ylabel("Mean empirical rank over 36 comparisons (lower is better)", color=INK, fontsize=10)
    axis.axhspan(2.1, 4.0, color=TEAL_LIGHT, alpha=0.55, zorder=0)
    axis.legend(
        handles=[
            Line2D([0], [0], marker="o", color="none", markerfacecolor=PURPLE, label="Foundation model", markersize=8),
            Line2D([0], [0], marker="s", color="none", markerfacecolor=GOLD, label="Baseline", markersize=7),
        ],
        loc="upper right",
        frameon=False,
        fontsize=8.8,
    )
    style_axis(axis, grid_axis="both")
    figure.tight_layout()
    save(figure, "appendix-speed-accuracy.png")


def main() -> None:
    summary_rows = read_rows(RESULTS_DIR / "11_12_3_rep_summary.csv")
    mcs_rows = read_rows(RESULTS_DIR / "hypothesis_tests" / "model_confidence_set.csv")
    calibration_rows = read_rows(RESULTS_DIR / "hypothesis_tests" / "probabilistic_calibration.csv")
    plot_family_gap(summary_rows)
    plot_rank_distribution(summary_rows)
    plot_mcs_heatmap(mcs_rows)
    plot_calibration(calibration_rows)
    plot_speed_accuracy(summary_rows)


if __name__ == "__main__":
    main()
