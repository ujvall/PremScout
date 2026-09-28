"""Services package for the PremAI backend."""

from .transfer_service import (
    predict_transfer_value_service,
    get_transfer_model_info_service,
    check_transfer_model_health,
)
from .match_service import (
    predict_match_outcome_service,
    get_match_model_info_service,
    check_match_model_health,
)
from .scouting_service import (
    get_similar_players_service,
    find_replacements_service,
    search_players_service,
    get_scouting_model_info_service,
    check_scouting_model_health,
)

__all__ = [
    "predict_transfer_value_service",
    "get_transfer_model_info_service",
    "check_transfer_model_health",
    "predict_match_outcome_service",
    "get_match_model_info_service",
    "check_match_model_health",
    "get_similar_players_service",
    "find_replacements_service",
    "search_players_service",
    "get_scouting_model_info_service",
    "check_scouting_model_health",
]
