"""API routes for Player Scouting & Replacement module."""

from typing import List, Optional
from fastapi import APIRouter, Query, Path
from backend.api.schemas.scouting import (
    ScoutingReplacementRequest,
    ReplacementCandidate,
    PlayerSearchItem,
    ScoutingModelInfoResponse,
)
from backend.services.scouting_service import (
    get_similar_players_service,
    find_replacements_service,
    search_players_service,
    get_scouting_model_info_service,
)

router = APIRouter(prefix="/scouting", tags=["Player Scouting & Replacement"])


@router.get(
    "/similar/{player_name}",
    response_model=List[ReplacementCandidate],
    summary="Find Statistically Similar Players",
    description="Retrieves the top N most similar players based on multi-dimensional football performance vectors across compatible positions.",
)
def get_similar_players_endpoint(
    player_name: str = Path(..., description="Target player name"),
    top_n: int = Query(10, ge=1, le=50, description="Number of similar players to retrieve"),
    include_free_agents: bool = Query(False, description="Whether to include verified free agents in candidate pool"),
) -> List[ReplacementCandidate]:
    """Retrieve top N statistically similar players."""
    return get_similar_players_service(player_name=player_name, top_n=top_n, include_free_agents=include_free_agents)


@router.post(
    "/replacements",
    response_model=List[ReplacementCandidate],
    summary="Find Replacement Candidates Under Scouting Constraints",
    description="Identifies ranked replacement candidates under recruitment filters including position compatibility, maximum market valuation, age limit, and minimum playing time.",
)
def find_replacements_endpoint(request: ScoutingReplacementRequest) -> List[ReplacementCandidate]:
    """Find ranked replacement candidates subject to recruitment filters."""
    return find_replacements_service(request)


@router.get(
    "/players",
    response_model=List[PlayerSearchItem],
    summary="Search Available Players Catalog",
    description="Returns a lightweight list of available players from the scouting database for frontend search dropdowns and autocomplete.",
)
def search_players_endpoint(
    search: Optional[str] = Query(None, description="Optional search substring to filter by player name or club"),
    limit: int = Query(50, ge=1, le=100, description="Maximum number of player records to return"),
    include_free_agents: bool = Query(False, description="Whether to include verified free agents in search universe"),
) -> List[PlayerSearchItem]:
    """Search and browse players available in the scouting index."""
    return search_players_service(search=search, limit=limit, include_free_agents=include_free_agents)


@router.get(
    "/model-info",
    response_model=ScoutingModelInfoResponse,
    summary="Scouting Model Metadata & Retrieval Evaluation",
    description="Returns feature definitions, position mapping, scaling method, and retrieval metrics (Top-5 same position rate, MRR).",
)
def get_scouting_model_info_endpoint() -> ScoutingModelInfoResponse:
    """Retrieve scouting vector similarity engine metadata and evaluation metrics."""
    return get_scouting_model_info_service()
