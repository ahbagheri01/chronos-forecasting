"""CLI entrypoint for the config-driven PMDS forecast comparison."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from pmds.config import load_config
    from pmds.pipeline import run_experiment

    config_path = args.config.expanduser().resolve()
    config = load_config(config_path)
    run_experiment(config, config_path)


if __name__ == "__main__":
    main()
