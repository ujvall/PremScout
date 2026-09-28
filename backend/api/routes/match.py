"""API routes for Match Outcome Predictor module."""

from fastapi import APIRouter
from backend.api.schemas.match import (
    MatchPredictRequest,
    MatchPredictResponse,
    MatchModelInfoResponse,
)
from backend.services.match_service import (
    predict_match_outcome_service,
    get_match_model_info_service,
)

router = APIRouter(prefix="/match", tags=["Match Outcome Predictor"])


@router.post(
    "/predict",
    response_model=MatchPredictResponse,
    summary="Predict Pre-Match Outcome Probabilities",
    description="Forecasts probabilities of Home Win, Draw, and Away Win from pre-kickoff rolling statistics, rest days, and league positions.",
)
def predict_match_endpoint(request: MatchPredictRequest) -> MatchPredictResponse:
    """Generate match outcome probabilities and most likely result."""
    return predict_match_outcome_service(request)


@router.get(
    "/model-info",
    response_model=MatchModelInfoResponse,
    summary="Match Outcome Model Metadata & Performance",
    description="Returns training seasons, feature count, validation metrics, and test accuracy/log loss for the match model.",
)
def get_match_model_info_endpoint() -> MatchModelInfoResponse:
    """Retrieve match outcome model training metadata and validation scores."""
    return get_match_model_info_service()


@router.get(
    "/current-teams",
    summary="Current Premier League Teams & Standings",
    description="Returns the active 2026/27 Premier League clubs with chronological sequential Elo ratings, league table rankings, and recent form metrics.",
)
def get_current_teams_endpoint():
    """Retrieve current Premier League clubs and standings."""
    from backend.services.match_service import get_current_teams_service
    return get_current_teams_service()


@router.get(
    "/current-fixture-features",
    summary="Current Pre-Match Fixture Features",
    description="Calculates the full pre-match feature vector for a matchup directly from current 2026/27 match data, Elo ratings, and rolling form.",
)
def get_current_fixture_features_endpoint(home_team: str, away_team: str):
    """Retrieve pre-match features for a specific head-to-head pairing."""
    from backend.services.match_service import get_current_fixture_features_service
    return get_current_fixture_features_service(home_team, away_team)

