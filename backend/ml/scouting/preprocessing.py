"""Preprocessing pipeline and position definitions for the Player Scouting & Replacement module."""

from typing import List, Dict, Set, Optional, Tuple
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# 16 Core Football Performance Features (per-90 and rate metrics, playing-time normalized)
SCOUTING_FEATURES: List[str] = [
    # Attacking & Goal Threat
    "goals_per90",
    "assists_per90",
    "xG_per90",
    "npxG_per90",
    "xAG_per90",
    "shots_per90",
    "key_passes_per90",
    # Passing & Ball Progression
    "pass_completion_pct",
    "progressive_passes_per90",
    "progressive_carries_per90",
    "successful_takeons_per90",
    # Defending & Aerial Dominance
    "tackles_won_per90",
    "interceptions_per90",
    "blocks_per90",
    "clearances_per90",
    "aerial_won_pct",
]

METADATA_COLS: List[str] = [
    "fbref_id",
    "tm_player_id",
    "player_name",
    "norm_name",
    "team",
    "season",
    "season_end_year",
    "position",
    "tm_position",
    "age",
    "minutes_played",
    "market_value_eur",
]

# Position Grouping Hierarchy mapped from raw FBref/Transfermarkt labels
POSITION_GROUP_MAP: Dict[str, str] = {
    "GK": "GK",
    "DF": "DF",
    "DF,MF": "DM/WB",
    "MF,DF": "DM/WB",
    "DF,FW": "DM/WB",
    "FW,DF": "DM/WB",
    "MF": "CM",
    "MF,FW": "AM/W",
    "FW,MF": "AM/W",
    "FW": "FW",
}

# Human-readable labels
POSITION_DISPLAY_NAMES: Dict[str, str] = {
    "GK": "Goalkeeper (GK)",
    "DF": "Defender (DF)",
    "DM/WB": "Defensive Utility / Wing-Back (DM/WB)",
    "CM": "Central Midfield (CM)",
    "AM/W": "Attacking Midfield / Winger (AM/W)",
    "FW": "Forward / Striker (FW)",
}

# Positional Compatibility Rules:
# Goalkeepers are strictly non-interchangeable.
# Outfield positions allow tactically adjacent transitions.
POSITION_COMPATIBILITY: Dict[str, Set[str]] = {
    "GK": {"GK"},
    "DF": {"DF", "DM/WB"},
    "DM/WB": {"DM/WB", "DF", "CM"},
    "CM": {"CM", "DM/WB", "AM/W"},
    "AM/W": {"AM/W", "CM", "FW"},
    "FW": {"FW", "AM/W"},
}


def normalize_position(raw_position: Optional[str]) -> str:
    """Normalize raw position string to a canonical scouting position group."""
    if not isinstance(raw_position, str):
        return "Unknown"
    return POSITION_GROUP_MAP.get(raw_position.strip(), "Unknown")


def check_position_compatibility(target_pos: str, candidate_pos: str) -> Optional[str]:
    """Check relationship between target position and candidate position.

    Returns:
        'Same Position' if exact match,
        'Compatible Position' if in tactical compatibility set,
        None if positions are incompatible.
    """
    if target_pos == candidate_pos:
        return "Same Position"
    if candidate_pos in POSITION_COMPATIBILITY.get(target_pos, set()):
        return "Compatible Position"
    return None


def build_scouting_preprocessor() -> Pipeline:
    """Build an sklearn preprocessing pipeline for football performance metrics.

    Applies median imputation for missing values and StandardScaler to ensure
    all metrics contribute evenly to geometric similarity.
    """
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
