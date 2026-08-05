"""Runtime setup for PMDS experiment runs."""

from __future__ import annotations

import hashlib
import logging
import logging.handlers
import os
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from pmds.utils import resolve_path


def setup_logging(run_config: Mapping[str, Any]) -> Path:
    log_config = run_config["logging"]
    log_dir = resolve_path(run_config["log_dir"])
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"{run_config['name']}_{timestamp}.log"

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.setLevel(logging.DEBUG)

    console = logging.StreamHandler(sys.stdout)
    console.setLevel(getattr(logging, str(log_config["console_level"]).upper()))
    console.setFormatter(formatter)
    root_logger.addHandler(console)

    file_handler = logging.handlers.RotatingFileHandler(
        log_path,
        maxBytes=int(log_config["max_bytes"]),
        backupCount=int(log_config["backup_count"]),
        encoding="utf-8",
    )
    file_handler.setLevel(getattr(logging, str(log_config["file_level"]).upper()))
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    logging.captureWarnings(True)
    return log_path


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def stable_seed(base_seed: int, *parts: object) -> int:
    payload = "|".join([str(base_seed), *(str(part) for part in parts)]).encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:4], byteorder="big", signed=False)


def apply_environment(environment: Mapping[str, Any]) -> None:
    for name, value in environment.items():
        os.environ[str(name)] = str(value)
        if str(name).endswith("DIR"):
            Path(str(value)).expanduser().mkdir(parents=True, exist_ok=True)
