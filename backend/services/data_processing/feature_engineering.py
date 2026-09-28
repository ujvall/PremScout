"""Feature engineering utilities for PremAI datasets."""

import pandas as pd


def calculate_per90(stat_total: float | int | None, minutes: float | int | None, round_digits: int = 2) -> float:
    """Safely calculate per-90 minute metrics.

    Formula: stat_per90 = (stat_total / minutes) * 90.0

    Rules:
    - If minutes <= 0 or stat is null/None -> returns 0.0
    - Prevents division-by-zero errors.
    """
    if stat_total is None or minutes is None:
        return 0.0
    try:
        m = float(minutes)
        s = float(stat_total)
        if m <= 0.0 or pd.isna(m) or pd.isna(s):
            return 0.0
        return round((s / m) * 90.0, round_digits)
    except (ValueError, TypeError, ZeroDivisionError):
        return 0.0


def assign_chronological_split(season_str: str) -> str:
    """Assign chronological train/val/test split group metadata.

    Target Split Scheme:
    - 2017-18 through 2021-22 -> 'train' (5 seasons)
    - 2022-23 -> 'val' (1 season)
    - 2023-24 -> 'test' (1 season)
    """
    s = str(season_str).strip()
    if s in ["2017-18", "2018-19", "2019-20", "2020-21", "2021-22", "2017", "2018", "2019", "2020", "2021"]:
        return "train"
    elif s in ["2022-23", "2022"]:
        return "val"
    elif s in ["2023-24", "2023"]:
        return "test"
    return "train"
