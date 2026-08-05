"""CLI entrypoint for the config-driven PMDS forecast comparison."""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path
from typing import Any, Mapping


if __package__ in {None, ""}:
    script_dir = str(Path(__file__).resolve().parent)
    sys.path = [path for path in sys.path if path != script_dir]
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the config-driven PMDS forecast comparison.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("pmds/config.json"),
        help="Path to the JSON experiment configuration.",
    )
    parser.add_argument(
        "--forecast-audit",
        action="store_true",
        help=(
            "Run one representative series, one origin, and one repetition per dataset for "
            "forecast-plot inspection. Its metrics are diagnostic, not the full benchmark."
        ),
    )
    return parser.parse_args()


def forecast_audit_config(config: Mapping[str, Any]) -> dict[str, Any]:
    audit = copy.deepcopy(config)
    output_dir = Path(str(audit["run"]["output_dir"])) / "forecast_audit"
    audit["run"]["name"] = "forecast_audit"
    audit["run"]["output_dir"] = str(output_dir)
    audit["run"]["log_dir"] = str(output_dir / "logs")
    audit["evaluation"]["stochastic_repetitions"] = 1
    for dataset in audit["datasets"]:
        if not dataset.get("enabled", True):
            continue
        dataset["max_series"] = 1
        dataset["num_origins"] = 1
        dataset["selection_strategy"] = "evenly_spaced"
    return audit


def main() -> None:
    args = parse_args()
    from pmds.config import load_config, validate_config
    from pmds.pipeline import run_experiment

    config_path = args.config.expanduser().resolve()
    config = load_config(config_path)
    if args.forecast_audit:
        config = forecast_audit_config(config)
        validate_config(config)
    run_experiment(config, config_path)


if __name__ == "__main__":
    main()
