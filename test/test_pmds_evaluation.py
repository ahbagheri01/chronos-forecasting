from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from pmds.compare import forecast_audit_config
from pmds.datasets import (
    make_tasks,
    matching_row_indices,
    prepare_external_targets,
    select_representative_items,
)
from pmds.metrics import compute_metrics, summarize
from pmds.outputs import context_rows_for_task, forecast_rows_for_result
from pmds.plot_results import aggregate_model_metrics, plot_results
from pmds.robustness import (
    aggregate_robustness,
    bias_scenarios,
    cap_context,
    normalize_scenario_result,
    outage_scenarios,
    outlier_scenarios,
    robust_scale,
    scale_invariance_error,
    scale_scenarios,
    run_robustness,
)
from pmds.runtime import stable_seed
from pmds.schemas import DatasetSpec, ForecastOutput, ForecastResult, ForecastTask


def dataset_spec(**overrides) -> DatasetSpec:
    values = {
        "name": "example",
        "family": "external",
        "repo": "repo",
        "hf_configs": ("config",),
        "split": "train",
        "prediction_length": 3,
        "seasonality": 2,
        "max_series": 3,
        "fallback_frequency": "D",
    }
    values.update(overrides)
    return DatasetSpec(**values)


class PmdsEvaluationTest(unittest.TestCase):
    def test_evenly_spaced_selection_is_representative_and_deterministic(self) -> None:
        values = list(range(10))
        self.assertEqual(select_representative_items(values, 3, "evenly_spaced", 0), [0, 4, 9])
        self.assertEqual(select_representative_items(values, 1, "evenly_spaced", 0), [5])
        self.assertEqual(
            select_representative_items(values, 4, "random", 17),
            select_representative_items(values, 4, "random", 17),
        )

    def test_chronos_row_filter_keeps_weather_metric_scope_homogeneous(self) -> None:
        class FakeDataset:
            column_names = ["subset"]

            def __len__(self) -> int:
                return 4

            def __getitem__(self, key):
                if key == "subset":
                    return ["rain", "mintemp", "rain", "solar"]
                raise KeyError(key)

        spec = dataset_spec(filter_column="subset", filter_values=("rain",), zero_inflated=True)
        self.assertEqual(matching_row_indices(FakeDataset(), spec), [0, 2])

    def test_forecast_audit_override_is_small_and_does_not_mutate_source(self) -> None:
        source = {
            "run": {"name": "compare", "output_dir": "results", "log_dir": "results/logs"},
            "evaluation": {"stochastic_repetitions": 3},
            "datasets": [{"enabled": True, "max_series": 20, "num_origins": 3}],
        }
        audit = forecast_audit_config(source)

        self.assertEqual(source["datasets"][0]["max_series"], 20)
        self.assertEqual(audit["run"]["output_dir"], "results/forecast_audit")
        self.assertEqual(audit["evaluation"]["stochastic_repetitions"], 1)
        self.assertEqual(audit["datasets"][0]["max_series"], 1)
        self.assertEqual(audit["datasets"][0]["num_origins"], 1)

    def test_make_tasks_builds_rolling_origins(self) -> None:
        spec = dataset_spec(num_origins=3, origin_stride=3)
        tasks = make_tasks(spec, "sensor", np.arange(20), None)

        self.assertEqual(len(tasks), 3)
        self.assertEqual([task.origin for task in tasks], [0, 1, 2])
        self.assertEqual([task.series_id for task in tasks], ["sensor", "sensor", "sensor"])
        np.testing.assert_array_equal(tasks[0].future, [17, 18, 19])
        np.testing.assert_array_equal(tasks[1].future, [14, 15, 16])
        np.testing.assert_array_equal(tasks[2].future, [11, 12, 13])

    def test_make_tasks_splits_before_causal_context_cleaning(self) -> None:
        spec = dataset_spec(prediction_length=2, seasonality=1, num_origins=1)
        values = [*np.arange(1.0, 20.0), np.nan, 100.0, 200.0]
        tasks = make_tasks(spec, "sensor", values, None)

        self.assertEqual(len(tasks), 1)
        np.testing.assert_array_equal(tasks[0].context, [*np.arange(1.0, 20.0), 19.0])
        np.testing.assert_array_equal(tasks[0].future, [100.0, 200.0])

    def test_context_cap_keeps_only_latest_observations(self) -> None:
        task = self._robustness_task()
        capped = cap_context(task, 5)

        np.testing.assert_array_equal(capped.context, task.context[-5:])
        self.assertEqual(list(capped.context_timestamps), list(task.context_timestamps[-5:]))
        np.testing.assert_array_equal(capped.future, task.future)

    def test_outlier_corruption_is_deterministic_shared_and_recent(self) -> None:
        task = self._robustness_task()
        first = outlier_scenarios(task, [2026])
        second = outlier_scenarios(task, [2026])

        self.assertEqual(len(first), 3)
        for left, right in zip(first, second):
            self.assertEqual(left.selected_indices, right.selected_indices)
            self.assertEqual(left.selected_signs, right.selected_signs)
            np.testing.assert_array_equal(left.task.context, right.task.context)
            np.testing.assert_array_equal(left.task.future, task.future)
            self.assertTrue(
                all(
                    index >= len(task.context) - 2 * task.prediction_length
                    for index in left.selected_indices
                )
            )

    def test_outage_requests_g_plus_h_and_discards_gap_predictions(self) -> None:
        task = self._robustness_task()
        moderate = next(scenario for scenario in outage_scenarios(task) if scenario.severity == "moderate")
        self.assertEqual(moderate.outage_length, 2)
        self.assertEqual(moderate.task.prediction_length, 6)
        np.testing.assert_array_equal(moderate.task.context, task.context[:-2])

        quantiles = np.array([0.1, 0.5, 0.9], dtype=np.float32)
        output = ForecastOutput(
            mean=np.arange(6, dtype=np.float32),
            quantiles=np.repeat(np.arange(6, dtype=np.float32)[:, None], 3, axis=1),
            distribution="samples",
        )
        result = ForecastResult(task.dataset, task.item_id, "model", output, 0.1, seed=1001)
        normalized = normalize_scenario_result(result, moderate, task.prediction_length)

        np.testing.assert_array_equal(normalized.output.mean, [2.0, 3.0, 4.0, 5.0])
        self.assertEqual(normalized.output.quantiles.shape, (4, 3))

    def test_bias_and_scale_preserve_the_clean_target_for_scoring(self) -> None:
        task = self._robustness_task()
        positive_bias = next(
            scenario
            for scenario in bias_scenarios(task)
            if scenario.severity == "moderate" and scenario.direction == "positive"
        )
        scale_up = next(
            scenario
            for scenario in scale_scenarios(task)
            if scenario.severity == "moderate" and scenario.factor == 10.0
        )

        np.testing.assert_array_equal(positive_bias.task.future, task.future)
        np.testing.assert_array_equal(scale_up.task.future, task.future * 10.0)

        quantiles = np.array([0.1, 0.5, 0.9], dtype=np.float32)
        scaled_output = ForecastOutput(
            mean=task.future * 10.0,
            quantiles=np.repeat((task.future * 10.0)[:, None], 3, axis=1),
            distribution="samples",
        )
        scaled_result = ForecastResult(task.dataset, task.item_id, "model", scaled_output, 0.1, seed=1001)
        normalized = normalize_scenario_result(scaled_result, scale_up, task.prediction_length)
        np.testing.assert_array_equal(normalized.output.mean, task.future)

    def test_scale_invariance_uses_clean_mase_denominator(self) -> None:
        task = self._robustness_task()
        quantiles = np.array([0.1, 0.5, 0.9], dtype=np.float32)
        clean_values = task.future.copy()
        shifted_values = task.future + 1.0
        clean = ForecastResult(
            task.dataset,
            task.item_id,
            "model",
            ForecastOutput(clean_values, np.repeat(clean_values[:, None], 3, axis=1), "samples"),
            0.1,
        )
        shifted = ForecastResult(
            task.dataset,
            task.item_id,
            "model",
            ForecastOutput(shifted_values, np.repeat(shifted_values[:, None], 3, axis=1), "samples"),
            0.1,
        )

        self.assertAlmostEqual(scale_invariance_error(task, clean, shifted, quantiles, "median"), 1.0)

    def test_hierarchical_summary_equal_weights_datasets(self) -> None:
        rows = []
        for dataset, ratios in {"small": [1.0], "large": [3.0, 3.0, 3.0]}.items():
            for index, ratio in enumerate(ratios):
                rows.append(
                    {
                        "dataset": dataset,
                        "series_id": f"s{index}",
                        "model": "m",
                        "model_family": "classical",
                        "test": "outlier",
                        "severity": "moderate",
                        "clean_mase": 1.0,
                        "stress_mase": ratio,
                        "robustness_ratio": ratio,
                        "percentage_degradation": 100.0 * (ratio - 1.0),
                        "absolute_degradation": ratio - 1.0,
                        "scale_invariance_error": np.nan,
                        "runtime_seconds": 1.0,
                        "applicable": True,
                        "status": "ok",
                    }
                )
        summary = aggregate_robustness(pd.DataFrame(rows), bootstrap_samples=0, bootstrap_seed=2026)

        self.assertAlmostEqual(summary.iloc[0]["robustness_ratio"], 2.0)

    def test_empty_robustness_results_produce_an_empty_summary(self) -> None:
        summary = aggregate_robustness(pd.DataFrame(), bootstrap_samples=10, bootstrap_seed=2026)
        self.assertTrue(summary.empty)

    def test_robust_scale_is_nan_for_constant_series(self) -> None:
        self.assertTrue(np.isnan(robust_scale(np.ones(20), seasonality=4)))

    def test_robustness_audit_runner_writes_paired_outputs(self) -> None:
        task = self._robustness_task()
        with tempfile.TemporaryDirectory() as directory:
            config = {
                "run": {
                    "name": "test",
                    "output_dir": directory,
                    "log_dir": str(Path(directory) / "logs"),
                    "random_seed": 42,
                    "environment": {"MPLCONFIGDIR": str(Path(directory) / "mpl")},
                    "logging": {
                        "console_level": "ERROR",
                        "file_level": "ERROR",
                        "max_bytes": 100000,
                        "backup_count": 1,
                    },
                },
                "evaluation": {
                    "metrics": ["mae", "rmse", "smape", "mase", "wql"],
                    "point_forecast": "median",
                    "quantiles": [0.1, 0.5, 0.9],
                },
                "robustness": {
                    "max_context": 512,
                    "max_series_per_dataset": 5,
                    "num_origins": 3,
                    "model_seeds": [1001, 1002, 1003],
                    "corruption_seeds": [2026, 2027, 2028],
                    "bootstrap_samples": 10,
                    "ratio_epsilon": 1e-8,
                },
                "datasets": [
                    {
                        "name": "d",
                        "family": "external",
                        "repo": "unused",
                        "hf_configs": ["unused"],
                        "split": "train",
                        "prediction_length": 4,
                        "seasonality": 1,
                        "max_series": 5,
                        "fallback_frequency": "D",
                    }
                ],
                "models": [
                    {
                        "name": "seasonal_naive",
                        "type": "seasonal_naive",
                        "params": {"use_dataset_seasonality": True, "seasonality": 1},
                    }
                ],
            }
            with patch("pmds.robustness.load_dataset_tasks", return_value=[task]):
                paths = run_robustness(config, Path("config.json"), audit=True)

            robustness = pd.read_csv(paths["robustness"])
            manifest = pd.read_csv(paths["manifest"])
            summary = pd.read_csv(paths["summary"])
            self.assertEqual(len(robustness), 18)
            self.assertEqual(len(manifest), 18)
            self.assertEqual(len(summary), 12)
            self.assertTrue((robustness["status"] == "ok").all())
            self.assertEqual(set(robustness["test"]), {"outlier", "outage", "bias", "scale"})

    @staticmethod
    def _robustness_task() -> ForecastTask:
        timestamps = pd.date_range("2024-01-01", periods=24, freq="D")
        return ForecastTask(
            dataset="d",
            item_id="s::origin=0",
            context_timestamps=timestamps[:20],
            future_timestamps=timestamps[20:],
            context=np.arange(1, 21, dtype=np.float32),
            future=np.arange(21, 25, dtype=np.float32),
            prediction_length=4,
            seasonality=1,
            frequency="D",
            series_id="s",
        )

    def test_psm_minute_timestamps_are_resampled_to_hourly_means(self) -> None:
        frame = pd.DataFrame(
            {
                "timestamp_(min)": np.arange(120),
                "sensor": np.arange(120, dtype=float),
            }
        )
        spec = dataset_spec(
            date_column="timestamp_(min)",
            timestamp_unit="m",
            resample_frequency="h",
            resample_method="mean",
            fallback_frequency="h",
        )

        targets, timestamps = prepare_external_targets(frame, spec)

        self.assertIsNotNone(timestamps)
        self.assertEqual(
            list(timestamps),
            [pd.Timestamp("2000-01-01 00:00:00"), pd.Timestamp("2000-01-01 01:00:00")],
        )
        np.testing.assert_allclose(targets["sensor"].to_numpy(), [29.5, 89.5])

    def test_official_and_macro_wql_are_both_reported(self) -> None:
        detailed = pd.DataFrame(
            [
                {
                    "dataset": "d",
                    "model": "m",
                    "item_id": "small",
                    "wql": 1.0,
                    "wql_loss_sum": 1.0,
                    "wql_abs_target_sum": 1.0,
                    "error": "",
                    "diagnostic_model": False,
                    "duration_seconds": 1.0,
                },
                {
                    "dataset": "d",
                    "model": "m",
                    "item_id": "large",
                    "wql": 9.0 / 99.0,
                    "wql_loss_sum": 9.0,
                    "wql_abs_target_sum": 99.0,
                    "error": "",
                    "diagnostic_model": False,
                    "duration_seconds": 1.0,
                },
            ]
        )

        summary = summarize(detailed, ["wql"]).iloc[0]
        self.assertAlmostEqual(summary["wql"], 0.1)
        self.assertAlmostEqual(summary["wql_macro"], (1.0 + 9.0 / 99.0) / 2.0)
        self.assertEqual(summary["evaluated_tasks"], 2)
        self.assertEqual(summary["successful_tasks"], 2)

    def test_weather_diagnostics_expose_zero_forecast_behavior(self) -> None:
        timestamps = pd.date_range("2024-01-01", periods=6, freq="D")
        task = ForecastTask(
            dataset="weather",
            item_id="rain::origin=0",
            context_timestamps=timestamps[:4],
            future_timestamps=timestamps[4:],
            context=np.array([0.0, 1.0, 0.0, 2.0], dtype=np.float32),
            future=np.array([0.0, 2.0], dtype=np.float32),
            prediction_length=2,
            seasonality=1,
            frequency="D",
            series_id="rain",
            zero_inflated=True,
            zero_threshold=1e-6,
        )
        quantiles = np.array([0.1, 0.5, 0.9], dtype=np.float32)
        output = ForecastOutput(
            mean=np.zeros(2, dtype=np.float32),
            quantiles=np.zeros((2, 3), dtype=np.float32),
            distribution="degenerate",
        )
        result = ForecastResult("weather", task.item_id, "zero", output, 0.1, repetition=0, seed=7)

        row = compute_metrics(
            task,
            result,
            ["mae", "smape", "wql", "rain_occurrence_error", "positive_mae"],
            quantiles,
            "median",
            diagnostic_model=True,
        )

        self.assertAlmostEqual(row["rain_occurrence_error"], 0.5)
        self.assertAlmostEqual(row["positive_mae"], 2.0)
        self.assertAlmostEqual(row["actual_zero_fraction"], 0.5)
        self.assertAlmostEqual(row["predicted_zero_fraction"], 1.0)
        self.assertIn("degenerate distribution", row["metric_notes"])

    def test_forecast_and_context_rows_preserve_auditable_values(self) -> None:
        timestamps = pd.date_range("2024-01-01", periods=6, freq="D")
        task = ForecastTask(
            dataset="d",
            item_id="s::origin=0",
            context_timestamps=timestamps[:4],
            future_timestamps=timestamps[4:],
            context=np.array([1, 2, 3, 4], dtype=np.float32),
            future=np.array([5, 6], dtype=np.float32),
            prediction_length=2,
            seasonality=1,
            frequency="D",
            series_id="s",
        )
        quantiles = np.array([0.1, 0.5, 0.9], dtype=np.float32)
        output = ForecastOutput(
            mean=np.array([4.5, 5.5], dtype=np.float32),
            quantiles=np.array([[4, 4.5, 5], [5, 5.5, 6]], dtype=np.float32),
            distribution="samples",
        )
        result = ForecastResult("d", task.item_id, "model", output, 0.2, repetition=1, seed=11)

        forecasts = forecast_rows_for_result(task, result, quantiles, "median")
        contexts = context_rows_for_task(task, 2)

        self.assertEqual(forecasts[0]["actual"], 5.0)
        self.assertEqual(forecasts[0]["point_forecast"], 4.5)
        self.assertEqual(forecasts[0]["q_10"], 4.0)
        self.assertEqual(forecasts[0]["repetition"], 1)
        self.assertEqual(contexts[-1]["actual"], 4.0)

    def test_plot_aggregation_and_forecast_plot_generation(self) -> None:
        detailed = pd.DataFrame(
            [
                {
                    "dataset": "d",
                    "item_id": "s::origin=0",
                    "model": model,
                    "mae": mae,
                    "wql": loss / target,
                    "wql_loss_sum": loss,
                    "wql_abs_target_sum": target,
                    "error_type": "",
                }
                for model, mae, loss, target in [("a", 1.0, 2.0, 10.0), ("b", 2.0, 4.0, 10.0)]
            ]
        )
        stats = aggregate_model_metrics(detailed, ["mae", "wql"])
        self.assertAlmostEqual(stats.loc["a", "wql"], 0.2)

        forecasts = pd.DataFrame(
            [
                {
                    "dataset": "d",
                    "series_id": "s",
                    "item_id": "s::origin=0",
                    "origin": 0,
                    "model": model,
                    "repetition": 0,
                    "timestamp": str(timestamp),
                    "actual": actual,
                    "point_forecast": actual + offset,
                    "mean_forecast": actual + offset,
                    "q_10": actual + offset - 0.5,
                    "q_90": actual + offset + 0.5,
                }
                for model, offset in [("a", 0.0), ("b", 1.0)]
                for timestamp, actual in zip(pd.date_range("2024-01-05", periods=2, freq="D"), [5.0, 6.0])
            ]
        )
        contexts = pd.DataFrame(
            [
                {
                    "dataset": "d",
                    "series_id": "s",
                    "item_id": "s::origin=0",
                    "origin": 0,
                    "timestamp": str(timestamp),
                    "actual": actual,
                }
                for timestamp, actual in zip(pd.date_range("2024-01-01", periods=4, freq="D"), [1, 2, 3, 4])
            ]
        )
        config = {"evaluation": {"metrics": ["mae", "wql"]}}
        with tempfile.TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            detailed_path = tmp_path / "run_detailed.csv"
            forecast_path = tmp_path / "run_forecasts.csv"
            context_path = tmp_path / "run_contexts.csv"
            config_path = tmp_path / "config.json"
            output_path = tmp_path / "plots"
            detailed.to_csv(detailed_path, index=False)
            forecasts.to_csv(forecast_path, index=False)
            contexts.to_csv(context_path, index=False)
            config_path.write_text(json.dumps(config), encoding="utf-8")

            plot_results(detailed_path, output_path, config_path, forecast_path, context_path)

            self.assertTrue((output_path / "d" / "wql.png").exists())
            self.assertTrue((output_path / "d" / "forecast_vs_actual" / "a.png").exists())
            self.assertTrue((output_path / "d" / "forecast_vs_actual" / "b.png").exists())

    def test_stable_seed_is_order_independent(self) -> None:
        self.assertEqual(
            stable_seed(42, "d", "item", "model", 0),
            stable_seed(42, "d", "item", "model", 0),
        )
        self.assertNotEqual(
            stable_seed(42, "d", "item", "model", 0),
            stable_seed(42, "d", "item", "model", 1),
        )


if __name__ == "__main__":
    unittest.main()
