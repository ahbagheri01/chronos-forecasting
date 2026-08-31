from __future__ import annotations

import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from pmds.config import load_config
from pmds.datasets import supports_requested_origins
from pmds.models import Moirai2Runner, TimesFMRunner, build_model_runners
from pmds.official_datasets import (
    _download_bls_macro,
    _download_elexon_demand,
    _write_snapshot,
    load_official_tasks,
)
from pmds.pipeline import cap_task_context
from pmds.schemas import DatasetSpec, ForecastTask
from pmds.utils import period_compatible_frequency


QUANTILES = np.arange(0.1, 1.0, 0.1, dtype=np.float32)


def example_task(context_length: int = 20, horizon: int = 3) -> ForecastTask:
    timestamps = pd.date_range("2020-01-01", periods=context_length + horizon, freq="D")
    return ForecastTask(
        dataset="example",
        item_id="series::origin=0",
        context_timestamps=timestamps[:context_length],
        future_timestamps=timestamps[context_length:],
        context=np.arange(context_length, dtype=np.float32),
        future=np.arange(horizon, dtype=np.float32),
        prediction_length=horizon,
        seasonality=7,
        frequency="D",
    )


class PmdsExpansionTest(unittest.TestCase):
    def test_primary_config_has_twelve_datasets_and_three_foundation_models(self) -> None:
        config = load_config(Path("pmds/config.json"))
        datasets = [item for item in config["datasets"] if item.get("enabled", True)]
        foundation_models = [
            item for item in config["models"] if item.get("enabled", True) and item.get("foundation_model", False)
        ]

        self.assertEqual(len(datasets), 12)
        self.assertEqual(
            {item["name"] for item in foundation_models},
            {"chronos_t5_tiny", "timesfm_2_5_200m", "moirai_2_0_small"},
        )
        self.assertEqual(
            {item["name"] for item in datasets if item["family"] == "official"},
            {"official_elexon_demand", "official_usgs_streamflow", "official_bls_macro"},
        )

    def test_external_keyless_profile_selects_only_six_non_chronos_datasets(self) -> None:
        config = load_config(Path("pmds/config_external_keyless.json"))
        enabled = [item for item in config["datasets"] if item.get("enabled", True)]

        self.assertEqual(config["run"]["name"], "external_keyless_clean")
        self.assertEqual(config["run"]["output_dir"], "pmds/results/external_keyless")
        self.assertEqual(len(enabled), 6)
        self.assertTrue(all(item["family"] != "chronos" for item in enabled))
        self.assertEqual(
            {item["name"] for item in enabled},
            {
                "external_national_illness",
                "external_traffic",
                "external_psm",
                "official_elexon_demand",
                "official_usgs_streamflow",
                "official_bls_macro",
            },
        )

    def test_heavier_chronos_profile_runs_all_datasets_with_four_requested_models(self) -> None:
        config = load_config(Path("pmds/config_heavier_chronos.json"))
        enabled_datasets = [item for item in config["datasets"] if item.get("enabled", True)]
        enabled_models = [item for item in config["models"] if item.get("enabled", True)]

        self.assertEqual(config["run"]["name"], "heavier_chronos_clean")
        self.assertEqual(config["run"]["output_dir"], "pmds/results/heavier_chronos")
        self.assertEqual(len(enabled_datasets), 12)
        self.assertEqual(
            {item["name"] for item in enabled_models},
            {"chronos_t5_mini", "chronos_t5_small", "chronos_t5_base", "chronos_t5_large"},
        )
        self.assertTrue(all(item["params"]["device"] == "cuda" for item in enabled_models))
        self.assertTrue(all(item["params"]["torch_dtype"] == "bfloat16" for item in enabled_models))

    def test_keyless_official_downloaders_normalize_elexon_and_bls(self) -> None:
        elexon = DatasetSpec(
            name="elexon",
            family="official",
            repo="",
            hf_configs=(),
            split="snapshot",
            prediction_length=1,
            seasonality=1,
            max_series=1,
            fallback_frequency="D",
            source="elexon_demand",
            source_params={"start": "2026-07-01", "end": "2026-07-03"},
        )
        with patch(
            "pmds.official_datasets._get_json",
            return_value=[
                {"settlementDate": "2026-07-03", "demand": 30},
                {"settlementDate": "2026-07-01", "demand": 10},
                {"settlementDate": "2026-07-02", "demand": 20},
            ],
        ):
            payload = _download_elexon_demand(elexon)
        self.assertEqual(payload["series"]["GB_NATIONAL_DEMAND"]["values"], [10.0, 20.0, 30.0])

        bls = DatasetSpec(
            name="bls",
            family="official",
            repo="",
            hf_configs=(),
            split="snapshot",
            prediction_length=1,
            seasonality=1,
            max_series=1,
            fallback_frequency="MS",
            source="bls_macro",
            source_params={
                "series_ids": ["SERIES"],
                "start": "2026-01-01",
                "end": "2026-03-31",
            },
        )
        response = {
            "status": "REQUEST_SUCCEEDED",
            "Results": {
                "series": [
                    {
                        "seriesID": "SERIES",
                        "data": [
                            {"year": "2026", "period": "M03", "value": "3"},
                            {"year": "2026", "period": "M13", "value": "2"},
                            {"year": "2026", "period": "M01", "value": "1"},
                        ],
                    }
                ]
            },
        }
        with patch("pmds.official_datasets._post_json", return_value=response):
            payload = _download_bls_macro(bls)
        self.assertEqual(payload["series"]["SERIES"]["values"], [1.0, None, 3.0])

    def test_all_model_runners_are_registered_without_loading_optional_packages(self) -> None:
        configs = [
            {"name": "chronos", "type": "chronos", "params": {}},
            {"name": "timesfm", "type": "timesfm", "params": {}},
            {"name": "moirai", "type": "moirai2", "params": {}},
        ]
        runners = build_model_runners(configs, QUANTILES)

        self.assertEqual(set(runners), {"chronos", "timesfm", "moirai"})
        self.assertIsInstance(runners["timesfm"], TimesFMRunner)
        self.assertIsInstance(runners["moirai"], Moirai2Runner)

    def test_timesfm_adapter_maps_official_quantile_axis(self) -> None:
        class FakeModel:
            compiled_config = None

            @classmethod
            def from_pretrained(cls, model_id, **kwargs):
                self = cls()
                self.model_id = model_id
                self.load_kwargs = kwargs
                return self

            def compile(self, config):
                self.compiled_config = config

            def forecast(self, horizon, inputs):
                self.inputs = inputs
                values = np.zeros((1, horizon, 10), dtype=np.float32)
                values[:, :, 0] = 99.0
                for index in range(1, 10):
                    values[:, :, index] = float(index)
                return np.full((1, horizon), 5.0, dtype=np.float32), values

        class FakeForecastConfig:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        fake_timesfm = types.ModuleType("timesfm")
        fake_timesfm.TimesFM_2p5_200M_torch = FakeModel
        fake_timesfm.ForecastConfig = FakeForecastConfig
        runner = TimesFMRunner(
            {
                "model_id": "google/timesfm-2.5-200m-pytorch",
                "max_context": 4,
                "max_horizon": 256,
            },
            QUANTILES,
        )
        with patch.dict(sys.modules, {"timesfm": fake_timesfm}):
            output = runner(example_task(context_length=8))

        np.testing.assert_array_equal(output.mean, [99.0, 99.0, 99.0])
        np.testing.assert_array_equal(output.quantiles[0], np.arange(1.0, 10.0))
        self.assertEqual(len(runner.model.inputs[0]), 4)
        self.assertEqual(runner.model.compiled_config.kwargs["max_context"], 32)

    def test_timesfm_adapter_recompiles_for_the_observed_context_bucket(self) -> None:
        class FakeModel:
            @classmethod
            def from_pretrained(cls, model_id, **kwargs):
                self = cls()
                self.compiled_contexts = []
                return self

            def compile(self, config):
                self.compiled_contexts.append(config.kwargs["max_context"])

            def forecast(self, horizon, inputs):
                values = np.ones((1, horizon, 10), dtype=np.float32)
                return np.ones((1, horizon), dtype=np.float32), values

        class FakeForecastConfig:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        fake_timesfm = types.ModuleType("timesfm")
        fake_timesfm.TimesFM_2p5_200M_torch = FakeModel
        fake_timesfm.ForecastConfig = FakeForecastConfig
        runner = TimesFMRunner(
            {"model_id": "timesfm", "max_context": 512, "max_horizon": 256}, QUANTILES
        )
        with patch.dict(sys.modules, {"timesfm": fake_timesfm}):
            runner(example_task(context_length=11))
            runner(example_task(context_length=20))
            runner(example_task(context_length=44))

        self.assertEqual(runner.model.compiled_contexts, [32, 64])

    def test_period_frequency_conversion_supports_month_start(self) -> None:
        self.assertEqual(period_compatible_frequency("MS"), "M")

    def test_moirai_adapter_maps_quantiles_and_uses_median_as_point(self) -> None:
        class FakeModule:
            quantile_levels = [index / 10 for index in range(1, 10)]

            @classmethod
            def from_pretrained(cls, model_id, **kwargs):
                return cls()

            def to(self, device):
                return self

            def eval(self):
                return self

        class FakeForecast:
            def __init__(self, **kwargs):
                self.horizon = kwargs["prediction_length"]

            def to(self, device):
                return self

            def predict(self, inputs):
                values = np.stack(
                    [np.full(self.horizon, index, dtype=np.float32) for index in range(1, 10)], axis=0
                )
                return torch.tensor(values[None, :, :])

        uni2ts = types.ModuleType("uni2ts")
        model_package = types.ModuleType("uni2ts.model")
        moirai2 = types.ModuleType("uni2ts.model.moirai2")
        moirai2.Moirai2Module = FakeModule
        moirai2.Moirai2Forecast = FakeForecast
        runner = Moirai2Runner(
            {"model_id": "Salesforce/moirai-2.0-R-small", "device": "cpu", "max_context": 4}, QUANTILES
        )
        with patch.dict(
            sys.modules,
            {"uni2ts": uni2ts, "uni2ts.model": model_package, "uni2ts.model.moirai2": moirai2},
        ):
            output = runner(example_task(context_length=8))

        np.testing.assert_array_equal(output.quantiles[0], np.arange(1.0, 10.0))
        np.testing.assert_array_equal(output.mean, [5.0, 5.0, 5.0])

    def test_common_context_cap_preserves_latest_observations(self) -> None:
        task = example_task(context_length=20)
        capped = cap_task_context(task, 8)

        np.testing.assert_array_equal(capped.context, task.context[-8:])
        self.assertEqual(list(capped.context_timestamps), list(task.context_timestamps[-8:]))
        self.assertEqual(len(task.context), 20)

    def test_dataset_selection_rejects_missing_forecast_targets(self) -> None:
        spec = DatasetSpec(
            name="selection",
            family="chronos",
            repo="repo",
            hf_configs=("config",),
            split="train",
            prediction_length=2,
            seasonality=4,
            max_series=1,
            fallback_frequency="D",
            num_origins=2,
            origin_stride=2,
        )
        self.assertTrue(supports_requested_origins(np.arange(20, dtype=float), spec))
        with_missing_tail = np.arange(20, dtype=float)
        with_missing_tail[-1] = np.nan
        self.assertFalse(supports_requested_origins(with_missing_tail, spec))

    def test_official_snapshot_is_reused_and_checksum_verified(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            snapshot_path = Path(directory) / "snapshot.json"
            timestamps = pd.date_range("2020-01-01", periods=30, freq="D")
            payload = {
                "dataset": "fixture",
                "downloaded_at": "2026-01-01T00:00:00Z",
                "source": "fixture",
                "query": {},
                "series": {
                    "gauge": {
                        "timestamps": [value.isoformat() for value in timestamps],
                        "values": list(map(float, range(30))),
                    }
                },
            }
            _write_snapshot(snapshot_path, payload)
            spec = DatasetSpec(
                name="official_fixture",
                family="official",
                repo="",
                hf_configs=(),
                split="snapshot",
                prediction_length=2,
                seasonality=1,
                max_series=1,
                fallback_frequency="D",
                num_origins=2,
                source="usgs_streamflow",
                snapshot_path=str(snapshot_path),
            )

            with patch("pmds.official_datasets._download_usgs_streamflow") as downloader:
                tasks = load_official_tasks(spec)
            self.assertEqual(len(tasks), 2)
            downloader.assert_not_called()

            raw = json.loads(snapshot_path.read_text(encoding="utf-8"))
            raw["series"]["gauge"]["values"][0] = 123.0
            snapshot_path.write_text(json.dumps(raw), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                load_official_tasks(spec)


if __name__ == "__main__":
    unittest.main()
