"""Player Scouting & Replacement module for PremAI."""

from .preprocessing import (
    SCOUTING_FEATURES,
    METADATA_COLS,
    POSITION_GROUP_MAP,
    POSITION_COMPATIBILITY,
    POSITION_DISPLAY_NAMES,
    normalize_position,
    check_position_compatibility,
)
from .similarity import ScoutingSimilarityIndex
from .predict import find_replacements, get_similar_players, load_scouting_index

__all__ = [
    "SCOUTING_FEATURES",
    "METADATA_COLS",
    "POSITION_GROUP_MAP",
    "POSITION_COMPATIBILITY",
    "POSITION_DISPLAY_NAMES",
    "normalize_position",
    "check_position_compatibility",
    "ScoutingSimilarityIndex",
    "find_replacements",
    "get_similar_players",
    "load_scouting_index",
]
