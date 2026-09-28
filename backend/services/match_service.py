"""Business service layer for Match Outcome Predictor module."""

import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
from fastapi import HTTPException

from backend.api.schemas.match import (
    MatchPredictRequest,
    MatchPredictResponse,
    MatchProbabilities,
    MatchModelInfoResponse,
)
from backend.ml.match.predict import predict_match, load_match_model

logger = logging.getLogger(__name__)

MATCH_MODEL_PATH = Path("models/match/match_outcome_model.joblib")
MATCH_METADATA_PATH = Path("models/match/model_metadata.json")


def check_match_model_health() -> bool:
    """Verify that the Match Outcome Predictor model artifact is present and loadable."""
    try:
        if not MATCH_MODEL_PATH.exists():
            return False
        artifact = load_match_model(MATCH_MODEL_PATH)
        return "pipeline" in artifact and "features" in artifact
    except Exception as e:
        logger.error(f"Match model health check failed: {e}")
        return False


CURRENT_TEAM_STATES_PATH = Path("data/current/matches/current_team_states.json")
_CACHED_TEAM_STATES: Optional[Dict[str, Any]] = None


def _load_team_states() -> Dict[str, Any]:
    global _CACHED_TEAM_STATES
    if _CACHED_TEAM_STATES is None:
        if not CURRENT_TEAM_STATES_PATH.exists():
            return {}
        try:
            with open(CURRENT_TEAM_STATES_PATH, "r", encoding="utf-8") as f:
                _CACHED_TEAM_STATES = json.load(f)
        except Exception as e:
            logger.error(f"Error loading current team states: {e}")
            return {}
    return _CACHED_TEAM_STATES


def get_current_teams_service() -> Dict[str, Any]:
    """Retrieve all current 2026/27 Premier League clubs with standings, Elo, and form."""
    states = _load_team_states()
    active_teams = [t for t, d in states.items() if d.get("matches_played_26", 0) > 0]
    ranked = sorted(
        active_teams,
        key=lambda t: (
            states[t].get("points_26", 0),
            states[t].get("rolling_gf_5", 0) - states[t].get("rolling_ga_5", 0)
        ),
        reverse=True
    )
    teams_list = []
    for rank, t in enumerate(ranked, start=1):
        st = states[t]
        teams_list.append({
            "team_name": t,
            "rank": rank,
            "points": st.get("points_26", 0),
            "matches_played": st.get("matches_played_26", 0),
            "ppg": round(st.get("ppg_26", 1.0), 2),
            "current_elo": round(st.get("current_elo", 1500.0), 1),
            "rolling_pts_5": round(st.get("rolling_pts_5", 1.0), 2),
            "rolling_gf_5": round(st.get("rolling_gf_5", 1.0), 2),
            "rolling_ga_5": round(st.get("rolling_ga_5", 1.0), 2),
            "rolling_win_pct_5": round(st.get("rolling_win_pct_5", 0.33), 2),
            "last_match_date": st.get("last_match_date", "2026-09-20"),
        })
    return {
        "success": True,
        "season": "2026-27",
        "data_as_of": "2026-09-20",
        "teams": teams_list,
    }


def get_current_fixture_features_service(home_team: str, away_team: str) -> Dict[str, Any]:
    """Derive pre-match feature vector from latest 2026/27 match data and team states."""
    states = _load_team_states()
    active_teams = [t for t, d in states.items() if d.get("matches_played_26", 0) > 0]
    ranked = sorted(
        active_teams,
        key=lambda t: (
            states[t].get("points_26", 0),
            states[t].get("rolling_gf_5", 0) - states[t].get("rolling_ga_5", 0)
        ),
        reverse=True
    )
    ranks = {t: i + 1 for i, t in enumerate(ranked)}

    h_state = states.get(home_team, {})
    a_state = states.get(away_team, {})

    h_elo = h_state.get("current_elo", 1500.0)
    a_elo = a_state.get("current_elo", 1500.0)
    elo_diff = round(h_elo - a_elo, 2)
    elo_exp_h = round(1.0 / (1.0 + 10 ** (-(h_elo + 65.0 - a_elo) / 400.0)), 4)

    h_rank = ranks.get(home_team, 10)
    a_rank = ranks.get(away_team, 10)
    rank_diff = a_rank - h_rank

    h_ppg = round(h_state.get("ppg_26", 1.25), 2)
    a_ppg = round(a_state.get("ppg_26", 1.25), 2)

    features = {
        "home_team": home_team,
        "away_team": away_team,
        "home_rest_days": 7.0,
        "away_rest_days": 7.0,
        "home_rolling_points_5": round(h_state.get("rolling_pts_5", 1.2) * 5.0, 1),
        "away_rolling_points_5": round(a_state.get("rolling_pts_5", 1.2) * 5.0, 1),
        "home_rolling_gf_5": round(h_state.get("rolling_gf_5", 1.2) * 5.0, 1),
        "away_rolling_gf_5": round(a_state.get("rolling_gf_5", 1.2) * 5.0, 1),
        "home_rolling_ga_5": round(h_state.get("rolling_ga_5", 1.2) * 5.0, 1),
        "away_rolling_ga_5": round(a_state.get("rolling_ga_5", 1.2) * 5.0, 1),
        "home_rolling_shots_5": h_state.get("rolling_shots_5", 12.0),
        "away_rolling_shots_5": a_state.get("rolling_shots_5", 12.0),
        "home_rolling_sot_5": h_state.get("rolling_sot_5", 4.0),
        "away_rolling_sot_5": a_state.get("rolling_sot_5", 4.0),
        "home_rolling_sot_conceded_5": h_state.get("rolling_sot_conceded_5", 4.0),
        "away_rolling_sot_conceded_5": a_state.get("rolling_sot_conceded_5", 4.0),
        "home_rolling_corners_5": h_state.get("rolling_corners_5", 5.0),
        "away_rolling_corners_5": a_state.get("rolling_corners_5", 5.0),
        "home_rolling_win_pct_5": h_state.get("rolling_win_pct_5", 0.4),
        "away_rolling_win_pct_5": a_state.get("rolling_win_pct_5", 0.4),
        "home_league_rank": h_rank,
        "away_league_rank": a_rank,
        "rank_diff": rank_diff,
        "home_ppg": h_ppg,
        "away_ppg": a_ppg,
        "h2h_matches_prior": 10,
        "h2h_home_win_pct": 0.45,
        "home_elo": h_elo,
        "away_elo": a_elo,
        "elo_diff": elo_diff,
        "elo_exp_h": elo_exp_h,
        "abs_elo_diff": abs(elo_diff),
        "abs_rank_diff": abs(rank_diff),
        "abs_ppg_diff": abs(round(h_ppg - a_ppg, 2)),
    }
    return {
        "success": True,
        "home_team": home_team,
        "away_team": away_team,
        "season": "2026-27",
        "data_as_of": "2026-09-20",
        "features": features,
    }


def predict_match_outcome_service(request: MatchPredictRequest) -> MatchPredictResponse:
    """Generate pre-match win/draw/away probabilities and most likely outcome.

    Automatically enriches feature dictionary with current 2026/27 Elo ratings
    and sequential match metrics.

    Args:
        request: Validated Pydantic request model.

    Returns:
        MatchPredictResponse with predicted class and calibrated probabilities.
    """
    try:
        feature_dict = request.model_dump()

        # Compute rank_diff if not explicitly provided
        if feature_dict.get("rank_diff") is None:
            feature_dict["rank_diff"] = request.away_league_rank - request.home_league_rank

        # Enrich with current team states (Elo and differential features) if available
        states = _load_team_states()
        ht = request.home_team
        at = request.away_team
        h_state = states.get(ht, {})
        a_state = states.get(at, {})

        if "home_elo" not in feature_dict or feature_dict["home_elo"] is None:
            feature_dict["home_elo"] = h_state.get("current_elo", 1500.0)
        if "away_elo" not in feature_dict or feature_dict["away_elo"] is None:
            feature_dict["away_elo"] = a_state.get("current_elo", 1500.0)

        h_elo = feature_dict["home_elo"]
        a_elo = feature_dict["away_elo"]
        elo_diff = round(h_elo - a_elo, 2)
        feature_dict["elo_diff"] = elo_diff
        feature_dict["elo_exp_h"] = round(1.0 / (1.0 + 10 ** (-(h_elo + 65.0 - a_elo) / 400.0)), 4)
        feature_dict["abs_elo_diff"] = abs(elo_diff)
        feature_dict["abs_rank_diff"] = abs(feature_dict["rank_diff"])
        feature_dict["abs_ppg_diff"] = abs(round(request.home_ppg - request.away_ppg, 2))

        result = predict_match(feature_dict, model_path=MATCH_MODEL_PATH)

        home_p = result["home_win_probability"]
        draw_p = result["draw_probability"]
        away_p = result["away_win_probability"]

        # Validate probability sum
        prob_sum = home_p + draw_p + away_p
        if abs(prob_sum - 1.0) > 0.05:
            logger.warning(f"Match probabilities sum to {prob_sum:.4f}, renormalizing.")
            home_p = round(home_p / prob_sum, 4)
            draw_p = round(draw_p / prob_sum, 4)
            away_p = round(1.0 - home_p - draw_p, 4)

        return MatchPredictResponse(
            home_team=request.home_team or "Home Team",
            away_team=request.away_team or "Away Team",
            predicted_outcome=result["predicted_code"],
            predicted_outcome_label=result["predicted_result"],
            probabilities=MatchProbabilities(
                home_win=home_p,
                draw=draw_p,
                away_win=away_p,
            ),
            model_type=result.get("model_type", "XGBoost"),
            data_as_of="2026-09-20",
            season="2026-27",
        )
    except FileNotFoundError as e:
        logger.error(f"Match model artifact missing: {e}")
        raise HTTPException(status_code=503, detail="Match Outcome Predictor model is currently unavailable.")
    except Exception as e:
        logger.error(f"Error executing match prediction: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to generate match outcome prediction: {str(e)}")


def get_match_model_info_service() -> MatchModelInfoResponse:
    """Retrieve metadata, dataset statistics, and evaluation scores for the match outcome model."""
    if not MATCH_METADATA_PATH.exists():
        raise HTTPException(status_code=404, detail="Match model metadata not found.")

    try:
        with open(MATCH_METADATA_PATH, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        return MatchModelInfoResponse(**metadata)
    except Exception as e:
        logger.error(f"Error loading match model metadata: {e}")
        raise HTTPException(status_code=500, detail="Failed to load match model metadata.")
