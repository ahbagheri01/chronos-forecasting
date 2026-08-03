"""Small shared helpers for the PMDS benchmark."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


def resolve_path(path_value: str) -> Path:
    return Path(path_value).expanduser()


def clean_numeric(values: Iterable) -> np.ndarray:
    series = pd.Series(np.asarray(values, dtype=np.float64))
    return series.interpolate(limit_direction="both").fillna(0.0).to_numpy(dtype=np.float32)


def period_compatible_frequency(frequency: str) -> str:
    replacements = {
        "ME": "M",
        "QE-": "Q-",
        "YE-": "Y-",
        "YS-": "Y-",
    }
    for source, target in replacements.items():
        if frequency == source or frequency.startswith(source):
            return frequency.replace(source, target, 1)
    return frequency
