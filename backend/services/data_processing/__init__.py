"""Data processing and feature engineering package for PremAI."""

from .clean_matches import build_match_outcome_dataset
from .clean_scouting import build_player_scouting_dataset
from .clean_transfer import build_transfer_value_dataset
from .entity_resolution import get_canonical_team_name, normalize_player_name

__all__ = [
    "build_match_outcome_dataset",
    "build_player_scouting_dataset",
    "build_transfer_value_dataset",
    "get_canonical_team_name",
    "normalize_player_name",
]
