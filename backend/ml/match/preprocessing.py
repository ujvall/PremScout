"""Preprocessing pipeline and feature definitions for the Match Outcome Predictor."""

from typing import List, Dict
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# Pre-kickoff features only. Metadata, match identifiers, dates, team names, and post-match stats are excluded.
MATCH_FEATURES: List[str] = [
    "home_rest_days",
    "away_rest_days",
    "home_rolling_points_5",
    "away_rolling_points_5",
    "home_rolling_gf_5",
    "away_rolling_gf_5",
    "home_rolling_ga_5",
    "away_rolling_ga_5",
    "home_rolling_shots_5",
    "away_rolling_shots_5",
    "home_rolling_sot_5",
    "away_rolling_sot_5",
    "home_rolling_sot_conceded_5",
    "away_rolling_sot_conceded_5",
    "home_rolling_corners_5",
    "away_rolling_corners_5",
    "home_rolling_win_pct_5",
    "away_rolling_win_pct_5",
    "home_league_rank",
    "away_league_rank",
    "rank_diff",
    "home_ppg",
    "away_ppg",
    "h2h_matches_prior",
    "h2h_home_win_pct",
]

TARGET_COL: str = "match_result"

# Class label mappings
# Internal integer encoding: 0 = H, 1 = D, 2 = A
CLASS_LABELS: List[str] = ["H", "D", "A"]
CLASS_TO_INT: Dict[str, int] = {"H": 0, "D": 1, "A": 2}
INT_TO_CLASS: Dict[int, str] = {0: "H", 1: "D", 2: "A"}

CLASS_HUMAN_NAMES: Dict[str, str] = {
    "H": "Home Win",
    "D": "Draw",
    "A": "Away Win",
}

INT_TO_HUMAN: Dict[int, str] = {
    0: "Home Win",
    1: "Draw",
    2: "Away Win",
}


def build_preprocessor() -> Pipeline:
    """Build an sklearn preprocessing pipeline for numerical pre-match features.

    Imputes missing values using the median learned from training data,
    and scales features using StandardScaler.
    """
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
