"""Small shared helpers for the PMDS benchmark."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


def resolve_path(path_value: str) -> Path:
    return Path(path_value).expanduser()


def numeric_array(values: Iterable) -> np.ndarray:
    """Convert values to a numeric array without filling missing observations."""
    return np.asarray(values, dtype=np.float64)


def clean_numeric(values: Iterable) -> np.ndarray:
    """Causally fill an already-split forecast context.

    Forward filling never borrows information from the forecast target. Leading
    missing values remain invalid because filling them would require looking
    ahead or inventing a dataset-specific value.
    """
    series = pd.Series(numeric_array(values)).replace([np.inf, -np.inf], np.nan).ffill()
    if series.isna().any():
        raise ValueError("Context contains leading or unfillable missing values")
    return series.to_numpy(dtype=np.float32)


def period_compatible_frequency(frequency: str) -> str:
    replacements = {
        "MS": "M",
        "ME": "M",
        "QS-": "Q-",
        "QE-": "Q-",
        "YE-": "Y-",
        "YS-": "Y-",
    }
    for source, target in replacements.items():
        if frequency == source or frequency.startswith(source):
            return frequency.replace(source, target, 1)
    return frequency
