"""Paired repetition tests for MASE, WQL, and CRPS.

This analysis measures stability across the three matched benchmark repetitions.
It is complementary to task-level SPA, MCS, and DM analyses and must not be
interpreted as a replacement for them.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t as student_t


CORE_METRICS = ("mase", "wql", "crps")
FOUNDATION_MODELS = {
    "chronos_t5_base",
    "chronos_t5_mini",
    "chronos_t5_small",
    "chronos_t5_tiny",
    "moirai_2_0_small",
    "timesfm_2_5_200m",
}


def repetition_scores(detailed: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (dataset, model, repetition), frame in detailed.groupby(
        ["dataset", "model", "repetition"], sort=True
    ):
        for metric in CORE_METRICS:
            if metric == "wql":
                numerator = pd.to_numeric(frame["wql_loss_sum"], errors="coerce").sum(min_count=1)
                denominator = pd.to_numeric(frame["wql_abs_target_sum"], errors="coerce").sum(min_count=1)
                score = numerator / denominator if denominator > 0 else np.nan
            else:
                score = pd.to_numeric(frame[metric], errors="coerce").mean()
            if np.isfinite(score):
                rows.append(
                    {
                        "dataset": str(dataset),
                        "metric": metric,
                        "model": str(model),
                        "repetition": int(repetition),
                        "score": float(score),
                    }
                )
    return pd.DataFrame(rows)


def holm_adjust(p_values: list[float]) -> list[float]:
    values = np.asarray(p_values, dtype=float)
    order = np.argsort(values)
    adjusted = np.empty_like(values)
    running = 0.0
    m = len(values)
    for rank, index in enumerate(order):
        candidate = (m - rank) * values[index]
        running = max(running, candidate)
        adjusted[index] = min(1.0, running)
    return adjusted.tolist()


def paired_tests(scores: pd.DataFrame, alpha: float = 0.05) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for dataset in sorted(scores["dataset"].unique()):
        for metric in CORE_METRICS:
            frame = scores.loc[
                scores["dataset"].eq(dataset) & scores["metric"].eq(metric)
            ].copy()
            means = frame.groupby("model", sort=True)["score"].mean().sort_values()
            winner = str(means.index[0])
            winner_is_foundation = winner in FOUNDATION_MODELS
            opposite = [
                str(model)
                for model in means.index
                if (str(model) in FOUNDATION_MODELS) != winner_is_foundation
            ]
            comparator = opposite[0]
            pivot = frame.loc[frame["model"].isin([winner, comparator])].pivot(
                index="repetition", columns="model", values="score"
            ).dropna()
            winner_values = pivot[winner].to_numpy(dtype=float)
            comparator_values = pivot[comparator].to_numpy(dtype=float)
            differences = comparator_values - winner_values
            n = len(differences)
            mean_difference = float(np.mean(differences))
            difference_sd = float(np.std(differences, ddof=1)) if n >= 2 else np.nan
            winner_mean = float(np.mean(winner_values))
            comparator_mean = float(np.mean(comparator_values))
            winner_sd = float(np.std(winner_values, ddof=1)) if n >= 2 else np.nan
            comparator_sd = float(np.std(comparator_values, ddof=1)) if n >= 2 else np.nan

            if n < 2:
                status = "insufficient_repetitions"
                statistic = np.nan
                raw_p = np.nan
            elif np.isclose(difference_sd, 0.0, atol=1e-15, rtol=1e-12):
                status = "seed_invariant_gap" if mean_difference > 0 else "exact_tie"
                statistic = np.nan
                raw_p = np.nan
            else:
                status = "paired_t_test"
                statistic = mean_difference / (difference_sd / math.sqrt(n))
                raw_p = float(student_t.sf(statistic, df=n - 1))

            row: dict[str, object] = {
                "dataset": dataset,
                "metric": metric,
                "winner": winner,
                "winner_family": "foundation" if winner_is_foundation else "comparator",
                "best_opposite_family_model": comparator,
                "n_matched_repetitions": n,
                "winner_repetition_mean": winner_mean,
                "winner_repetition_sample_sd": winner_sd,
                "opposite_repetition_mean": comparator_mean,
                "opposite_repetition_sample_sd": comparator_sd,
                "mean_paired_gap": mean_difference,
                "paired_gap_sample_sd": difference_sd,
                "relative_loss_reduction_percent": (
                    100.0 * mean_difference / comparator_mean
                    if not np.isclose(comparator_mean, 0.0)
                    else np.nan
                ),
                "test": "one_sided_paired_t",
                "alternative": "mean_paired_gap_gt_0",
                "degrees_of_freedom": n - 1 if n >= 2 else np.nan,
                "t_statistic": statistic,
                "raw_p_value": raw_p,
                "test_status": status,
            }
            for index in range(n):
                repetition = int(pivot.index[index])
                row[f"winner_score_rep_{repetition}"] = winner_values[index]
                row[f"opposite_score_rep_{repetition}"] = comparator_values[index]
                row[f"paired_gap_rep_{repetition}"] = differences[index]
            rows.append(row)

    result = pd.DataFrame(rows)
    valid = result["raw_p_value"].notna()
    adjusted = holm_adjust(result.loc[valid, "raw_p_value"].astype(float).tolist())
    result["holm_adjusted_p_value"] = np.nan
    result.loc[valid, "holm_adjusted_p_value"] = adjusted
    result["holm_family_size"] = int(valid.sum())
    result["holm_reject_0_05"] = pd.NA
    result.loc[valid, "holm_reject_0_05"] = (
        result.loc[valid, "holm_adjusted_p_value"].astype(float) < alpha
    )
    result["interpretation"] = np.select(
        [
            result["test_status"].eq("seed_invariant_gap"),
            result["test_status"].eq("exact_tie"),
            result["holm_reject_0_05"].eq(True).fillna(False),
            result["test_status"].eq("paired_t_test"),
        ],
        [
            "same positive gap in all three repetitions; no finite t statistic",
            "same score in all three repetitions",
            "gap exceeds repetition variability after Holm correction",
            "not resolved with three repetitions after Holm correction",
        ],
        default="insufficient repetitions",
    )
    result["inference_scope"] = (
        "Exploratory seed-stability test only. Repetitions measure random-seed variation; "
        "SPA, MCS, and DM provide the separate task-level model-comparison evidence."
    )
    return result.sort_values(["dataset", "metric"]).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "results" / "11_12_3_rep",
    )
    args = parser.parse_args()
    detailed_path = args.results_dir / "11_12_3_rep_detailed.csv"
    output_path = args.results_dir / "hypothesis_tests" / "repetition_paired_tests_core_metrics.csv"
    detailed = pd.read_csv(detailed_path, low_memory=False)
    result = paired_tests(repetition_scores(detailed))
    result.to_csv(output_path, index=False)
    tested = int(result["raw_p_value"].notna().sum())
    invariant = int(result["test_status"].eq("seed_invariant_gap").sum())
    significant = int(result["holm_reject_0_05"].eq(True).fillna(False).sum())
    print(f"wrote {len(result)} rows to {output_path}")
    print(f"paired t tests: {tested}; seed-invariant gaps: {invariant}; Holm significant: {significant}")


if __name__ == "__main__":
    main()
