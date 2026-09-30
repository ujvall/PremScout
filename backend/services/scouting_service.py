"""Business service layer for Player Scouting & Replacement module."""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import pandas as pd
from fastapi import HTTPException

from backend.api.schemas.scouting import (
    ScoutingReplacementRequest,
    ReplacementCandidate,
    CandidateStats,
    PlayerSearchItem,
    ScoutingModelInfoResponse,
)
from backend.ml.scouting.predict import (
    find_replacements,
    get_similar_players,
    load_scouting_index,
)
from backend.ml.scouting.preprocessing import normalize_position

logger = logging.getLogger(__name__)

SCOUTING_MODEL_PATH = Path("models/scouting/scouting_model.joblib")
SCOUTING_METADATA_PATH = Path("models/scouting/model_metadata.json")

CURRENT_PLAYERS_PATH = Path("data/current/players/current_players.csv")

# In-memory cached player search catalog
_CACHED_PLAYER_CATALOG: Optional[List[Dict[str, Any]]] = None


def _normalize_search_string(name: str) -> str:
    """Robust Unicode name normalization handling accents, hyphens, and whitespace."""
    import unicodedata
    if not isinstance(name, str):
        return ""
    norm = unicodedata.normalize("NFKD", name)
    norm = "".join(c for c in norm if not unicodedata.combining(c))
    norm = norm.replace("'", "").replace("-", " ").replace(".", "")
    norm = norm.replace("\xad", "").replace("\xa0", " ").replace("\ufffd", "")
    return " ".join(norm.lower().split())


def check_scouting_model_health() -> bool:
    """Verify that the Scouting & Replacement model artifact is present and loadable."""
    try:
        if not SCOUTING_MODEL_PATH.exists():
            return False
        index = load_scouting_index(SCOUTING_MODEL_PATH)
        return index is not None and len(index.df) > 0
    except Exception as e:
        logger.error(f"Scouting model health check failed: {e}")
        return False


def _get_player_catalog() -> List[Dict[str, Any]]:
    """Build and cache a lightweight search catalog of all unique players from latest available data.
    
    Prioritizes data/current/players/current_players.csv (2025/26 - 2026/27).
    Falls back to historical model dataset if current data layer is not present.
    """
    global _CACHED_PLAYER_CATALOG
    if _CACHED_PLAYER_CATALOG is None:
        catalog = []
        if CURRENT_PLAYERS_PATH.exists():
            df_curr = pd.read_csv(CURRENT_PLAYERS_PATH, encoding="utf-8")
            for _, row in df_curr.iterrows():
                p_name = str(row["player_name"])
                norm_name = str(row.get("normalized_name") or _normalize_search_string(p_name))
                c_club = str(row.get("current_club") or row["club"])
                c_league = str(row.get("current_league") or "Premier League")
                elig = str(row.get("eligibility_status") or "BIG5_ACTIVE")
                is_b5 = bool(row.get("is_big5_current", True))
                is_fa = bool(row.get("is_free_agent", False))
                sc_elig = bool(row.get("scouting_eligible", is_b5))

                catalog.append({
                    "player_name": p_name,
                    "normalized_name": norm_name,
                    "player_id": str(row["player_id"]) if pd.notna(row.get("player_id")) else None,
                    "position": str(row.get("position", "Midfield")),
                    "normalized_position": str(row.get("position", "Midfield")),
                    "club": c_club,
                    "season": str(row["season"]),
                    "age": int(row["age"]) if pd.notna(row.get("age")) else None,
                    "market_value_eur": float(row["market_value_eur"]) if pd.notna(row.get("market_value_eur")) else None,
                    "data_as_of": str(row.get("data_as_of", "2026-09-20")),
                    "has_scouting_data": bool(row.get("has_scouting_data", False)),
                    "is_current": True,
                    # Registration & Eligibility
                    "current_club": c_club,
                    "current_league": c_league,
                    "eligibility_status": elig,
                    "is_big5_current": is_b5,
                    "is_current_big5": is_b5,
                    "is_free_agent": is_fa,
                    "scouting_eligible": sc_elig,
                })
        else:
            # Fallback to model's historical index dataframe
            index = load_scouting_index(SCOUTING_MODEL_PATH)
            df = index.df.copy()
            df_latest = df.sort_values(by="season_end_year", ascending=False).drop_duplicates(subset=["player_name"])
            for _, row in df_latest.iterrows():
                p_name = str(row["player_name"])
                norm_name = _normalize_search_string(p_name)
                catalog.append({
                    "player_name": p_name,
                    "normalized_name": norm_name,
                    "player_id": str(row["fbref_id"]) if pd.notna(row.get("fbref_id")) else None,
                    "position": str(row["position"]),
                    "normalized_position": str(row["norm_pos"]),
                    "club": str(row["team"]),
                    "season": str(row["season"]),
                    "age": int(row["age"]) if pd.notna(row.get("age")) else None,
                    "market_value_eur": float(row["market_value_eur"]) if pd.notna(row.get("market_value_eur")) else None,
                    "data_as_of": "2024-05-19",
                    "has_scouting_data": True,
                    "is_current": False,
                    "current_club": str(row["team"]),
                    "current_league": "Premier League",
                    "eligibility_status": "BIG5_ACTIVE",
                    "is_big5_current": True,
                    "is_current_big5": True,
                    "is_free_agent": False,
                    "scouting_eligible": True,
                })
        _CACHED_PLAYER_CATALOG = catalog

    return _CACHED_PLAYER_CATALOG


PREMIER_LEAGUE_CLUBS = {
    "Arsenal", "Aston Villa", "Bournemouth", "Brentford", "Brighton",
    "Chelsea", "Crystal Palace", "Everton", "Fulham", "Ipswich Town",
    "Leicester City", "Liverpool", "Manchester City", "Manchester United",
    "Newcastle United", "Nottingham Forest", "Southampton", "Tottenham Hotspur",
    "West Ham United", "Wolverhampton Wanderers"
}


def search_players_service(
    search: Optional[str] = None,
    limit: int = 50,
    include_free_agents: bool = False,
) -> List[PlayerSearchItem]:
    """Search available players in the scouting catalog by name or club substring.

    Dynamically restricts search results to the eligible scouting universe:
    - Default: BIG5_ACTIVE only.
    - With include_free_agents=True: BIG5_ACTIVE + FREE_AGENT.
    - NON_BIG5 and UNKNOWN are strictly excluded.

    Args:
        search: Optional search query string (case-insensitive).
        limit: Maximum number of records to return (capped at 100).
        include_free_agents: Whether to include verified free agents in search universe.

    Returns:
        List of lightweight PlayerSearchItem records.
    """
    limit = max(1, min(limit, 100))
    full_catalog = _get_player_catalog()

    # Filter catalog strictly to eligible scouting universe:
    # Default: BIG5_ACTIVE only
    # With toggle ON: BIG5_ACTIVE + FREE_AGENT
    # NON_BIG5 and UNKNOWN are strictly excluded
    if not include_free_agents:
        catalog = [item for item in full_catalog if item.get("eligibility_status") == "BIG5_ACTIVE"]
    else:
        catalog = [item for item in full_catalog if item.get("eligibility_status") in ("BIG5_ACTIVE", "FREE_AGENT")]

    def sort_key(x):
        return (
            x.get("is_big5_current", False),
            x.get("club") in PREMIER_LEAGUE_CLUBS,
            x.get("has_scouting_data", False),
            x.get("market_value_eur") or 0.0,
        )

    if not search or not search.strip():
        # Return top players prioritizing current season stars with scouting profiles
        sorted_top = sorted(catalog, key=sort_key, reverse=True)
        return [
            PlayerSearchItem(**{k: v for k, v in item.items() if k in PlayerSearchItem.model_fields})
            for item in sorted_top[:limit]
        ]

    raw_query = search.strip().lower()
    q_norm = _normalize_search_string(search)

    # Multi-tier match prioritizing exact names, token/surname matches, prefixes, and substrings
    exact_matches = []
    token_matches = []
    prefix_matches = []
    substring_matches = []

    for item in catalog:
        p_name_raw = item["player_name"].lower()
        p_norm = item.get("normalized_name") or _normalize_search_string(item["player_name"])
        c_name = item["club"].lower()
        tokens = p_norm.split()

        is_exact = (q_norm == p_norm or raw_query == p_name_raw)
        is_token = (q_norm in tokens)
        is_prefix = any(t.startswith(q_norm) for t in tokens) or p_norm.startswith(q_norm)
        is_sub = (q_norm in p_norm or raw_query in p_name_raw or q_norm in c_name or raw_query in c_name)

        if is_exact:
            exact_matches.append(item)
        elif is_token:
            token_matches.append(item)
        elif is_prefix:
            prefix_matches.append(item)
        elif is_sub:
            substring_matches.append(item)

    exact_matches.sort(key=sort_key, reverse=True)
    token_matches.sort(key=sort_key, reverse=True)
    prefix_matches.sort(key=sort_key, reverse=True)
    substring_matches.sort(key=sort_key, reverse=True)

    ordered = exact_matches + token_matches + prefix_matches + substring_matches
    results = [
        PlayerSearchItem(**{k: v for k, v in item.items() if k in PlayerSearchItem.model_fields})
        for item in ordered[:limit]
    ]

    return results


def get_similar_players_service(player_name: str, top_n: int = 10, include_free_agents: bool = False) -> List[ReplacementCandidate]:
    """Retrieve top N statistically similar players across compatible positions.

    Args:
        player_name: Target player name.
        top_n: Number of candidates to return.
        include_free_agents: Whether to include verified free agents.

    Returns:
        List of ReplacementCandidate models.
    """
    try:
        raw_candidates = get_similar_players(
            player_name=player_name,
            top_n=top_n,
            include_free_agents=include_free_agents,
            model_path=SCOUTING_MODEL_PATH,
        )
        return [_map_candidate(c) for c in raw_candidates]
    except ValueError as e:
        err_msg = str(e)
        if "not found in scouting database" in err_msg.lower():
            raise HTTPException(status_code=404, detail=f"Player '{player_name}' was not found.")
        raise HTTPException(status_code=422, detail=err_msg)
    except FileNotFoundError:
        raise HTTPException(status_code=503, detail="Player Scouting model is currently unavailable.")
    except Exception as e:
        logger.error(f"Error retrieving similar players for {player_name}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to retrieve similar players: {str(e)}")


def find_replacements_service(request: ScoutingReplacementRequest) -> List[ReplacementCandidate]:
    """Find ranked replacement candidates for a player under recruitment constraints.

    Args:
        request: Validated Pydantic request model.

    Returns:
        List of ReplacementCandidate models.
    """
    try:
        raw_candidates = find_replacements(
            player_name=request.player_name,
            top_n=request.top_n,
            same_position_only=request.same_position_only,
            max_market_value=request.max_market_value,
            max_age=request.max_age,
            min_minutes=request.min_minutes,
            include_free_agents=request.include_free_agents,
            model_path=SCOUTING_MODEL_PATH,
        )
        return [_map_candidate(c) for c in raw_candidates]
    except ValueError as e:
        err_msg = str(e)
        if "not found in scouting database" in err_msg.lower():
            raise HTTPException(status_code=404, detail=f"Player '{request.player_name}' was not found.")
        if "no eligible replacement candidates found" in err_msg.lower():
            logger.info(f"Zero candidates found for {request.player_name} with strict filters.")
            return []
        raise HTTPException(status_code=422, detail=err_msg)
    except FileNotFoundError:
        raise HTTPException(status_code=503, detail="Player Scouting model is currently unavailable.")
    except Exception as e:
        logger.error(f"Error finding replacements for {request.player_name}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to find replacements: {str(e)}")


def get_scouting_model_info_service() -> ScoutingModelInfoResponse:
    """Retrieve metadata, feature list, and evaluation metrics for the scouting model."""
    if not SCOUTING_METADATA_PATH.exists():
        raise HTTPException(status_code=404, detail="Scouting model metadata not found.")

    try:
        with open(SCOUTING_METADATA_PATH, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        return ScoutingModelInfoResponse(**metadata)
    except Exception as e:
        logger.error(f"Error loading scouting model metadata: {e}")
        raise HTTPException(status_code=500, detail="Failed to load scouting model metadata.")


def _map_candidate(c: Dict[str, Any]) -> ReplacementCandidate:
    """Map dictionary output from similarity engine to Pydantic ReplacementCandidate."""
    stats = c.get("key_stats", {})
    return ReplacementCandidate(
        target_player=c["target_player"],
        target_team=c["target_team"],
        target_position=c["target_position"],
        target_season=c["target_season"],
        candidate_player=c["candidate_player"],
        candidate_club=c["candidate_club"],
        candidate_position=c["candidate_position"],
        candidate_raw_position=c["candidate_raw_position"],
        candidate_season=c["candidate_season"],
        # Required Concept B & Concept C fields
        current_club=c.get("current_club", c["candidate_club"]),
        current_league=c.get("current_league", "Premier League"),
        current_country=c.get("current_country", "England"),
        current_season=c.get("current_season", "2026-27"),
        performance_club=c.get("performance_club", c["candidate_club"]),
        performance_league=c.get("performance_league", c.get("current_league", "Premier League")),
        performance_season=c.get("performance_season", "2025-26"),
        is_current_big5=bool(c.get("is_current_big5", True)),
        is_free_agent=bool(c.get("is_free_agent", False)),
        similarity_score=c["similarity_score"],
        cosine_similarity=c["cosine_similarity"],
        position_compatibility=c["position_compatibility"],
        age=c.get("age"),
        market_value_eur=c.get("market_value_eur"),
        minutes=c["minutes"],
        key_stats=CandidateStats(
            goals_per90=stats.get("goals_per90", 0.0),
            assists_per90=stats.get("assists_per90", 0.0),
            npxG_per90=stats.get("npxG_per90", 0.0),
            xAG_per90=stats.get("xAG_per90", 0.0),
            key_passes_per90=stats.get("key_passes_per90", 0.0),
            pass_completion_pct=stats.get("pass_completion_pct", 0.0),
            progressive_passes_per90=stats.get("progressive_passes_per90", 0.0),
            progressive_carries_per90=stats.get("progressive_carries_per90", 0.0),
            tackles_won_per90=stats.get("tackles_won_per90", 0.0),
            interceptions_per90=stats.get("interceptions_per90", 0.0),
        ),
        recent_form=c.get("recent_form"),
        data_as_of=c.get("data_as_of", "2026-09-20"),
    )
