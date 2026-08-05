#!/usr/bin/env python
"""Generate metric and forecast-audit plots from PMDS result CSVs."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


plt.style.use("seaborn-v0_8-whitegrid")

MODEL_COLOR = "#4F86A8"
FORECAST_COLOR = "#087E8B"
ACTUAL_COLOR = "#17212B"
CONTEXT_COLOR = "#7A8793"
BAND_COLOR = "#9FD5D9"


def load_config(config_path: Path) -> dict:
    with config_path.open(encoding="utf-8") as fp:
        return json.load(fp)


def successful_rows(frame: pd.DataFrame) -> pd.DataFrame:
    if "error_type" not in frame.columns:
        return frame.copy()
    errors = frame["error_type"].fillna("").astype(str)
    return frame.loc[errors.eq("")].copy()


def aggregate_model_metrics(dataset_frame: pd.DataFrame, metrics: Iterable[str]) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    for model, model_frame in dataset_frame.groupby("model", sort=True):
        row: dict[str, float | str] = {"model": str(model)}
        for metric in metrics:
            if metric not in model_frame.columns:
                row[metric] = float("nan")
                continue
            if metric == "wql" and {"wql_loss_sum", "wql_abs_target_sum"}.issubset(model_frame.columns):
                loss = model_frame["wql_loss_sum"].sum(min_count=1)
                target = model_frame["wql_abs_target_sum"].sum(min_count=1)
                row[metric] = float(loss / target) if pd.notna(target) and target != 0 else float("nan")
                row["wql_macro"] = float(model_frame["wql"].mean())
            else:
                row[metric] = float(model_frame[metric].mean())
        rows.append(row)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).set_index("model")


def metric_label(value: float) -> str:
    if not np.isfinite(value):
        return "NA"
    magnitude = abs(value)
    if magnitude >= 1000:
        return f"{value:.3g}"
    if 0 < magnitude < 0.01:
        return f"{value:.4g}"
    return f"{value:.3f}"


def save_metric_bar(data: pd.Series, dataset: str, metric: str, path: Path) -> None:
    data = data.dropna().sort_values()
    if data.empty:
        return
    width = max(9.0, 0.9 * len(data))
    fig, ax = plt.subplots(figsize=(width, 5.4))
    bars = ax.bar(data.index, data.values, color=MODEL_COLOR, edgecolor="#24485F", alpha=0.88)
    ax.set_title(f"{dataset.replace('_', ' ').title()}\n{metric.upper()} by model", weight="bold")
    ax.set_xlabel("Model")
    ax.set_ylabel(metric.upper())
    positive = data.loc[data.gt(0)]
    use_log_scale = (
        len(positive) == len(data)
        and not positive.empty
        and float(positive.max() / positive.min()) >= 100.0
    )
    if use_log_scale:
        ax.set_yscale("log")
        ax.set_ylabel(f"{metric.upper()} (log scale)")
    ax.tick_params(axis="x", rotation=38)
    for label in ax.get_xticklabels():
        label.set_ha("right")
    for bar, value in zip(bars, data.values):
        ax.annotate(
            metric_label(float(value)),
            (bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    ax.text(
        0.99,
        0.98,
        "Lower is better" + (" | log scale" if use_log_scale else ""),
        transform=ax.transAxes,
        ha="right",
        va="top",
        color="#66717D",
        fontsize=9,
    )
    ax.margins(y=0.16)
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_rank_views(model_stats: pd.DataFrame, dataset: str, metrics: list[str], output_dir: Path) -> None:
    available = [metric for metric in metrics if metric in model_stats and model_stats[metric].notna().any()]
    if len(available) < 2:
        return
    ranks = model_stats[available].rank(axis=0, method="average", ascending=True, na_option="bottom")

    fig, ax = plt.subplots(figsize=(max(10, 1.2 * len(ranks)), 5.8))
    ranks.plot(kind="bar", ax=ax, width=0.82)
    ax.set_title(
        f"{dataset.replace('_', ' ').title()}\nWithin-metric model ranks",
        weight="bold",
    )
    ax.set_xlabel("Model")
    ax.set_ylabel("Rank (1 = best)")
    ax.tick_params(axis="x", rotation=38)
    for label in ax.get_xticklabels():
        label.set_ha("right")
    ax.legend(title="Metric", bbox_to_anchor=(1.02, 1), loc="upper left")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(output_dir / "all_metrics_normalized.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    ordered = ranks.mean(axis=1).sort_values().index
    heatmap = ranks.loc[ordered]
    fig, ax = plt.subplots(figsize=(max(7, 1.25 * len(available)), max(4.5, 0.52 * len(heatmap))))
    image = ax.imshow(heatmap.to_numpy(), cmap="YlGnBu", aspect="auto", vmin=1)
    ax.set_xticks(range(len(available)), labels=[metric.upper() for metric in available])
    ax.set_yticks(range(len(heatmap)), labels=heatmap.index)
    for row_index in range(len(heatmap)):
        for column_index in range(len(available)):
            value = heatmap.iat[row_index, column_index]
            ax.text(column_index, row_index, f"{value:.1f}", ha="center", va="center", fontsize=8)
    ax.set_title(
        f"{dataset.replace('_', ' ').title()}\nRank heatmap (1 = best)",
        weight="bold",
    )
    fig.colorbar(image, ax=ax, label="Rank")
    fig.tight_layout()
    fig.savefig(output_dir / "all_metrics_dot_comparison.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def safe_filename(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "model"


def quantile_columns(frame: pd.DataFrame) -> list[str]:
    columns = [column for column in frame.columns if re.fullmatch(r"q_\d+", str(column))]
    return sorted(columns, key=lambda value: int(value.split("_")[1]))


def _aggregate_forecast_repetitions(frame: pd.DataFrame, q_columns: list[str]) -> pd.DataFrame:
    aggregations: dict[str, str] = {
        "actual": "first",
        "point_forecast": "median",
        "mean_forecast": "mean",
    }
    aggregations.update({column: "median" for column in q_columns})
    return frame.groupby("timestamp", as_index=False).agg(aggregations).sort_values("timestamp")


def save_forecast_comparison(
    dataset: str,
    model: str,
    forecasts: pd.DataFrame,
    contexts: pd.DataFrame | None,
    output_path: Path,
) -> None:
    latest = forecasts.loc[forecasts["origin"].eq(forecasts["origin"].min())].copy()
    series_ids = list(map(str, latest["series_id"].drop_duplicates()))
    if not series_ids:
        return
    columns = 2 if len(series_ids) <= 8 else 4
    rows = math.ceil(len(series_ids) / columns)
    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(5.2 * columns, 3.2 * rows + 1.2),
        squeeze=False,
        sharex=False,
    )
    q_columns = quantile_columns(latest)
    low_column = q_columns[0] if q_columns else None
    high_column = q_columns[-1] if q_columns else None

    for axis, series_id in zip(axes.flat, series_ids):
        series_forecasts = latest.loc[latest["series_id"].astype(str).eq(series_id)].copy()
        series_forecasts["timestamp"] = pd.to_datetime(series_forecasts["timestamp"])
        aggregate = _aggregate_forecast_repetitions(series_forecasts, q_columns)

        if contexts is not None and not contexts.empty:
            series_context = contexts.loc[
                contexts["series_id"].astype(str).eq(series_id)
                & contexts["origin"].eq(latest["origin"].min())
            ].copy()
            if not series_context.empty:
                series_context["timestamp"] = pd.to_datetime(series_context["timestamp"])
                series_context = series_context.sort_values("timestamp")
                axis.plot(
                    series_context["timestamp"],
                    series_context["actual"],
                    color=CONTEXT_COLOR,
                    linewidth=1.1,
                    label="History",
                )

        axis.plot(
            aggregate["timestamp"],
            aggregate["actual"],
            color=ACTUAL_COLOR,
            linewidth=1.8,
            label="Actual",
        )
        axis.plot(
            aggregate["timestamp"],
            aggregate["point_forecast"],
            color=FORECAST_COLOR,
            linewidth=1.7,
            label="Point forecast",
        )
        if low_column and high_column:
            axis.fill_between(
                aggregate["timestamp"],
                aggregate[low_column].astype(float),
                aggregate[high_column].astype(float),
                color=BAND_COLOR,
                alpha=0.38,
                label=f"{low_column[2:]}-{high_column[2:]}% interval",
            )
        cutoff = aggregate["timestamp"].min()
        axis.axvline(cutoff, color="#D95D39", linewidth=1, linestyle="--")
        axis.set_title(series_id, fontsize=10, weight="bold")
        axis.tick_params(axis="x", rotation=25, labelsize=8)
        axis.tick_params(axis="y", labelsize=8)

    for axis in axes.flat[len(series_ids) :]:
        axis.set_visible(False)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.89),
        ncol=min(4, len(labels)),
        frameon=False,
    )
    fig.suptitle(
        f"{dataset.replace('_', ' ').title()} | {model}\nForecast versus actual (latest rolling origin)",
        fontsize=15,
        weight="bold",
        y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.82))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def generate_forecast_plots(forecast_path: Path, context_path: Path, output_base: Path) -> int:
    if not forecast_path.exists():
        print(f"Forecast CSV not found; skipping forecast plots: {forecast_path}")
        return 0
    forecasts = pd.read_csv(forecast_path)
    if forecasts.empty:
        print(f"Forecast CSV is empty; skipping forecast plots: {forecast_path}")
        return 0
    contexts = pd.read_csv(context_path) if context_path.exists() else None
    count = 0
    for (dataset, model), frame in forecasts.groupby(["dataset", "model"], sort=True):
        dataset_contexts = None
        if contexts is not None and not contexts.empty:
            dataset_contexts = contexts.loc[contexts["dataset"].eq(dataset)]
        output_path = output_base / str(dataset) / "forecast_vs_actual" / f"{safe_filename(str(model))}.png"
        save_forecast_comparison(str(dataset), str(model), frame, dataset_contexts, output_path)
        print(f"  Saved: {output_path}")
        count += 1
    return count


def plot_results(
    csv_path: Path,
    output_base: Path,
    config_path: Path,
    forecast_path: Path | None = None,
    context_path: Path | None = None,
    forecast_only: bool = False,
) -> None:
    config = load_config(config_path)
    metrics = list(config.get("evaluation", {}).get("metrics", ["mae", "rmse", "smape", "mase", "wql"]))
    detailed = successful_rows(pd.read_csv(csv_path))
    output_base.mkdir(parents=True, exist_ok=True)

    if not forecast_only:
        datasets = list(map(str, detailed["dataset"].dropna().unique()))
        print(f"Found {len(datasets)} datasets")
        for dataset in datasets:
            print(f"\nProcessing dataset: {dataset}")
            dataset_frame = detailed.loc[detailed["dataset"].eq(dataset)]
            dataset_dir = output_base / dataset
            dataset_dir.mkdir(parents=True, exist_ok=True)
            model_stats = aggregate_model_metrics(dataset_frame, metrics)
            print(f"  Models: {', '.join(map(str, model_stats.index))}")
            for metric in metrics:
                if metric not in model_stats or not model_stats[metric].notna().any():
                    continue
                path = dataset_dir / f"{metric}.png"
                save_metric_bar(model_stats[metric], dataset, metric, path)
                print(f"  Saved: {path}")
            if "wql_macro" in model_stats and model_stats["wql_macro"].notna().any():
                path = dataset_dir / "wql_macro.png"
                save_metric_bar(model_stats["wql_macro"], dataset, "wql_macro", path)
                print(f"  Saved: {path}")
            save_rank_views(model_stats, dataset, metrics, dataset_dir)

    if forecast_path is None:
        forecast_path = csv_path.with_name(csv_path.name.replace("_detailed.csv", "_forecasts.csv"))
    if context_path is None:
        context_path = csv_path.with_name(csv_path.name.replace("_detailed.csv", "_contexts.csv"))
    forecast_count = generate_forecast_plots(forecast_path, context_path, output_base)
    print(f"\nMetric plots saved under: {output_base}")
    print(f"Forecast-vs-actual plots generated: {forecast_count}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate PMDS metric and forecast-audit plots")
    parser.add_argument("--csv", type=Path, default=Path("pmds/results/compare_detailed.csv"))
    parser.add_argument("--output", type=Path, default=Path("pmds/results/plots"))
    parser.add_argument("--config", type=Path, default=Path("pmds/config.json"))
    parser.add_argument("--forecasts", type=Path)
    parser.add_argument("--contexts", type=Path)
    parser.add_argument(
        "--forecast-only",
        action="store_true",
        help="Generate forecast-versus-actual plots without replacing metric plots.",
    )
    args = parser.parse_args()
    if not args.csv.exists():
        raise FileNotFoundError(f"Detailed results CSV not found: {args.csv}")
    plot_results(
        args.csv,
        args.output,
        args.config,
        args.forecasts,
        args.contexts,
        forecast_only=args.forecast_only,
    )


if __name__ == "__main__":
    main()
