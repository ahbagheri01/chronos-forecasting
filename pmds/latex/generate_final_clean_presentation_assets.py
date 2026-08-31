#!/usr/bin/env python3
"""Generate data-driven result and appendix slides for the final clean deck."""

from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RESULT_ROOT = ROOT / "pmds" / "results" / "final_clean_results"
PLOT_ROOT = RESULT_ROOT / "plots"
SUMMARY_PATH = RESULT_ROOT / "final_clean_summary.csv"
LATEX_ROOT = Path(__file__).resolve().parent

DATASET_ORDER = [
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
]

CHRONOS_DATASETS = set(DATASET_ORDER[:6])

DATASET_NAMES = {
    "chronos_m1_yearly": "M1 Yearly",
    "chronos_m4_hourly": "M4 Hourly",
    "chronos_weather": "Weather",
    "chronos_m3_quarterly": "M3 Quarterly",
    "chronos_car_parts": "Car Parts",
    "chronos_nn5_weekly": "NN5 Weekly",
    "external_national_illness": "National Illness",
    "external_traffic": "Traffic",
    "external_psm": "PSM",
    "official_elexon_demand": "Elexon Electricity Demand",
    "official_usgs_streamflow": "USGS Streamflow",
    "official_bls_macro": "BLS Macroeconomic Indicators",
}

# Shorter labels keep the longest forecast-vs-actual appendix titles on one line.
APPENDIX_DATASET_NAMES = {
    **DATASET_NAMES,
    "official_elexon_demand": "Elexon Demand",
    "official_bls_macro": "BLS Macro Indicators",
}

MODEL_ORDER = [
    "chronos_t5_tiny",
    "timesfm_2_5_200m",
    "moirai_2_0_small",
    "seasonal_naive",
    "arima_2_1_2",
    "auto_arima",
    "prophet",
    "deepar",
    "chronos_t5_mini",
    "chronos_t5_small",
    "chronos_t5_base",
    "chronos_t5_large",
]

MODEL_NAMES = {
    "chronos_t5_tiny": "Chronos Tiny",
    "chronos_t5_mini": "Chronos Mini",
    "chronos_t5_small": "Chronos Small",
    "chronos_t5_base": "Chronos Base",
    "chronos_t5_large": "Chronos Large",
    "timesfm_2_5_200m": "TimesFM 2.5",
    "moirai_2_0_small": "Moirai 2.0",
    "seasonal_naive": "Seasonal Naive",
    "arima_2_1_2": "ARIMA(2,1,2)",
    "auto_arima": "AutoARIMA",
    "prophet": "Prophet",
    "deepar": "DeepAR",
}

SHORT_MODEL_NAMES = {
    **MODEL_NAMES,
    "timesfm_2_5_200m": "TimesFM",
    "moirai_2_0_small": "Moirai",
    "seasonal_naive": "S. Naive",
    "arima_2_1_2": "ARIMA",
}

MODEL_COLORS = {
    "chronos_t5_tiny": "TealLight",
    "chronos_t5_mini": "TealLight",
    "chronos_t5_small": "TealLight",
    "chronos_t5_base": "TealLight",
    "chronos_t5_large": "TealLight",
    "timesfm_2_5_200m": "GreenLight",
    "moirai_2_0_small": "GoldLight",
    "seasonal_naive": "GrayLight",
    "arima_2_1_2": "CoralLight",
    "auto_arima": "CoralLight",
    "prophet": "CoralLight",
    "deepar": "PurpleLight",
}

CORE_METRICS = ["mae", "rmse", "smape", "mase", "wql"]
METRIC_LABELS = {
    "mae": "MAE",
    "rmse": "RMSE",
    "smape": "sMAPE",
    "mase": "MASE",
    "wql": "WQL",
    "wql_macro": "Macro WQL",
    "rain_occurrence_error": "Rain occurrence error",
    "positive_mae": "Positive-value MAE",
    "all_metrics_normalized": "Within-metric ranks",
    "all_metrics_dot_comparison": "Rank heatmap",
}

METRIC_PLOT_ORDER = [
    "mae",
    "rmse",
    "smape",
    "mase",
    "wql",
    "wql_macro",
    "rain_occurrence_error",
    "positive_mae",
    "all_metrics_normalized",
    "all_metrics_dot_comparison",
]


def tex(value: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "_": r"\_",
        "&": r"\&",
        "%": r"\%",
        "#": r"\#",
    }
    return "".join(replacements.get(char, char) for char in str(value))


def metric_value(value: float) -> str:
    magnitude = abs(float(value))
    if magnitude >= 100_000:
        return f"{value:,.0f}"
    if magnitude >= 1_000:
        return f"{value:,.1f}"
    if magnitude >= 1:
        return f"{value:.3f}"
    if magnitude >= 0.01:
        return f"{value:.4f}"
    return f"{value:.3g}"


def canonical_summary() -> list[dict[str, str]]:
    with SUMMARY_PATH.open(newline="", encoding="utf-8") as handle:
        summary = list(csv.DictReader(handle))
    selected = [
        row
        for row in summary
        if row["source_run"] == "heavier_chronos"
        or (row["source_run"] == "keyless" and row["dataset"] in CHRONOS_DATASETS)
        or (
            row["source_run"] == "external_keyless"
            and row["dataset"] not in CHRONOS_DATASETS
        )
    ]
    keys = [(row["dataset"], row["model"]) for row in selected]
    duplicates = len(keys) - len(set(keys))
    if duplicates:
        raise RuntimeError(
            f"Canonical summary contains {duplicates} duplicate dataset-model rows"
        )
    expected = len(DATASET_ORDER) * len(MODEL_ORDER)
    if len(selected) != expected:
        raise RuntimeError(f"Expected {expected} canonical rows, found {len(selected)}")
    return selected


def winner_rows(summary: list[dict[str, str]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for dataset in DATASET_ORDER:
        frame = [row for row in summary if row["dataset"] == dataset]
        row: dict[str, object] = {"dataset": dataset}
        for metric in CORE_METRICS:
            winner = min(frame, key=lambda item: float(item[metric]))
            row[metric] = str(winner["model"])
            row[f"{metric}_value"] = float(winner[metric])
        rows.append(row)
    return rows


def result_slides(winners: list[dict[str, object]]) -> str:
    output: list[str] = []
    for record in winners:
        dataset = str(record["dataset"])
        name = DATASET_NAMES[dataset]
        output.extend(
            [
                "% ------------------------------------------------------------",
                rf"\begin{{frame}}{{{tex(name)}: performance by metric}}",
                r"  \vspace{-0.06cm}",
                r"  \begin{columns}[T,totalwidth=\linewidth]",
                r"    \begin{column}{0.49\linewidth}",
                rf"      \includegraphics[width=\linewidth,height=0.48\textheight,keepaspectratio]{{{dataset}/mase.png}}",
                r"    \end{column}",
                r"    \begin{column}{0.49\linewidth}",
                rf"      \includegraphics[width=\linewidth,height=0.48\textheight,keepaspectratio]{{{dataset}/wql.png}}",
                r"    \end{column}",
                r"  \end{columns}",
                r"  \vspace{0.06cm}",
                r"  \centering",
                r"  \scriptsize",
                r"  \setlength{\tabcolsep}{2pt}",
                r"  \begin{tabular}{C{0.19\linewidth} C{0.19\linewidth} C{0.19\linewidth} C{0.19\linewidth} C{0.19\linewidth}}",
                r"    \textbf{MAE} & \textbf{RMSE} & \textbf{sMAPE} & \textbf{MASE} & \textbf{WQL} \\[2pt]",
            ]
        )
        cells = []
        for metric in CORE_METRICS:
            model = str(record[metric])
            label = tex(SHORT_MODEL_NAMES[model])
            value = tex(metric_value(float(record[f"{metric}_value"])))
            cells.append(rf"\winnerbox{{{MODEL_COLORS[model]}}}{{{label}\\{value}}}")
        output.append("    " + " & ".join(cells) + r" \\")
        output.extend(
            [
                r"  \end{tabular}",
                r"  \vspace{0.05cm}",
                r"  {\tiny\color{Muted}Lower is better. Values are canonical dataset-level aggregates from the final clean runs.}",
                r"\end{frame}",
                "",
            ]
        )

    output.extend(winner_matrix_slide(winners))
    output.extend(winner_count_slide(winners))
    return "\n".join(output)


def winner_matrix_slide(winners: list[dict[str, object]]) -> list[str]:
    lines = [
        "% ------------------------------------------------------------",
        r"\begin{frame}{Winner matrix: model with the lowest aggregate metric}",
        r"  \vspace{-0.10cm}",
        r"  \centering",
        r"  \tiny",
        r"  \renewcommand{\arraystretch}{1.28}",
        r"  \setlength{\tabcolsep}{2.5pt}",
        r"  \begin{tabular}{L{0.27\linewidth} C{0.13\linewidth} C{0.13\linewidth} C{0.13\linewidth} C{0.13\linewidth} C{0.13\linewidth}}",
        r"    \toprule",
        r"    \textbf{Dataset} & \textbf{MAE} & \textbf{RMSE} & \textbf{sMAPE} & \textbf{MASE} & \textbf{WQL} \\",
        r"    \midrule",
    ]
    for record in winners:
        cells = [tex(DATASET_NAMES[str(record["dataset"])])]
        for metric in CORE_METRICS:
            model = str(record[metric])
            cells.append(
                rf"\cellcolor{{{MODEL_COLORS[model]}}}\textbf{{{tex(SHORT_MODEL_NAMES[model])}}}"
            )
        lines.append("    " + " & ".join(cells) + r" \\")
    lines.extend(
        [
            r"    \bottomrule",
            r"  \end{tabular}",
            r"  \vspace{0.16cm}",
            r"  {\scriptsize\color{Muted}Each cell reports the minimum among the 12 evaluated models for that dataset and metric.}",
            r"\end{frame}",
            "",
        ]
    )
    return lines


def winner_count_slide(winners: list[dict[str, object]]) -> list[str]:
    counts = Counter(
        str(record[metric]) for record in winners for metric in CORE_METRICS
    )
    lines = [
        "% ------------------------------------------------------------",
        r"\begin{frame}{How often each model wins a dataset--metric cell}",
        r"  \vspace{0.08cm}",
        r"  \begin{columns}[T,totalwidth=0.94\linewidth]",
        r"    \begin{column}{0.48\linewidth}",
        r"      \small",
        r"      \renewcommand{\arraystretch}{1.30}",
        r"      \setlength{\tabcolsep}{5pt}",
        r"      \begin{tabular}{L{0.66\linewidth} C{0.20\linewidth}}",
        r"        \toprule",
        r"        \textbf{Model} & \textbf{Wins} \\",
        r"        \midrule",
    ]
    for model, count in counts.most_common():
        lines.append(
            rf"        \cellcolor{{{MODEL_COLORS[str(model)]}}}{tex(MODEL_NAMES[str(model)])} & \textbf{{{int(count)}}} \\"
        )
    lines.extend(
        [
            r"        \bottomrule",
            r"      \end{tabular}",
            r"    \end{column}",
            r"    \begin{column}{0.40\linewidth}",
            r"      \small",
            r"      \tagbox{GreenLight}{\textbf{60 winner cells}}",
            r"      \vspace{0.22cm}",
            r"      \begin{itemize}",
            r"        \item 12 datasets $\times$ 5 core metrics.",
            r"        \item A win is the lowest aggregate metric value.",
            r"        \item Counts summarize breadth, not statistical significance.",
            r"        \item Metrics are correlated, so five wins on one dataset are not five independent experiments.",
            r"      \end{itemize}",
            r"    \end{column}",
            r"  \end{columns}",
            r"\end{frame}",
            "",
        ]
    )
    return lines


def appendix_slides() -> str:
    lines = [
        r"\appendix",
        r"\section{Appendix}",
        "% ------------------------------------------------------------",
        r"\begin{frame}[plain]",
        r"  \vspace{1.25cm}",
        r"  {\color{Teal}\rule{1.25cm}{3pt}}",
        r"  \vspace{0.45cm}",
        r"  {\fontsize{28}{33}\selectfont\bfseries Appendix\par}",
        r"  \vspace{0.30cm}",
        r"  {\Large\color{Muted}Complete metric and forecast-trajectory plots\par}",
        r"\end{frame}",
        "",
    ]

    for dataset in DATASET_ORDER:
        dataset_dir = PLOT_ROOT / dataset
        available = {path.stem: path for path in dataset_dir.glob("*.png")}
        for metric in METRIC_PLOT_ORDER:
            if metric not in available:
                continue
            title = f"{APPENDIX_DATASET_NAMES[dataset]} --- {METRIC_LABELS[metric]}"
            rel = f"{dataset}/{available[metric].name}"
            lines.extend(single_image_slide(title, rel))

    for dataset in DATASET_ORDER:
        forecast_dir = PLOT_ROOT / dataset / "forecast_vs_actual"
        available = {path.stem: path for path in forecast_dir.glob("*.png")}
        for model in MODEL_ORDER:
            if model not in available:
                raise RuntimeError(f"Missing forecast plot for {dataset}/{model}")
            title = f"{APPENDIX_DATASET_NAMES[dataset]} --- forecast vs. actual --- {MODEL_NAMES[model]}"
            rel = f"{dataset}/forecast_vs_actual/{available[model].name}"
            lines.extend(single_image_slide(title, rel))
    return "\n".join(lines)


def single_image_slide(title: str, relative_path: str) -> list[str]:
    return [
        "% ------------------------------------------------------------",
        rf"\begin{{frame}}{{{tex(title)}}}",
        r"  \centering",
        rf"  \includegraphics[width=\linewidth,height=0.79\textheight,keepaspectratio]{{{relative_path}}}",
        r"\end{frame}",
        "",
    ]


def main() -> None:
    summary = canonical_summary()
    winners = winner_rows(summary)
    result_path = LATEX_ROOT / "final_clean_results_slides.tex"
    appendix_path = LATEX_ROOT / "final_clean_appendix.tex"
    result_path.write_text(result_slides(winners), encoding="utf-8")
    appendix_path.write_text(appendix_slides(), encoding="utf-8")
    print(f"Wrote {result_path}")
    print(f"Wrote {appendix_path}")
    print(f"Canonical rows: {len(summary)}")
    print(f"Winner cells: {len(winners) * len(CORE_METRICS)}")
    print(f"Appendix images: {len(list(PLOT_ROOT.rglob('*.png')))}")


if __name__ == "__main__":
    main()
