"""Pydantic schemas for the Match Outcome Predictor API."""

from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field


class MatchPredictRequest(BaseModel):
    """Pre-match statistical indicators required for outcome forecasting."""
    home_team: Optional[str] = Field("Home Team", description="Home club name")
    away_team: Optional[str] = Field("Away Team", description="Away club name")
    home_rest_days: float = Field(7.0, ge=0, le=60, description="Rest days since home team's last competitive match")
    away_rest_days: float = Field(7.0, ge=0, le=60, description="Rest days since away team's last competitive match")
    home_rolling_points_5: float = Field(..., ge=0, le=15, description="Cumulative points earned in home team's last 5 matches")
    away_rolling_points_5: float = Field(..., ge=0, le=15, description="Cumulative points earned in away team's last 5 matches")
    home_rolling_gf_5: float = Field(..., ge=0, description="Goals scored in home team's last 5 matches")
    away_rolling_gf_5: float = Field(..., ge=0, description="Goals scored in away team's last 5 matches")
    home_rolling_ga_5: float = Field(..., ge=0, description="Goals conceded in home team's last 5 matches")
    away_rolling_ga_5: float = Field(..., ge=0, description="Goals conceded in away team's last 5 matches")
    home_rolling_shots_5: float = Field(..., ge=0, description="Average shots taken per match in last 5 matches")
    away_rolling_shots_5: float = Field(..., ge=0, description="Average shots taken per match in last 5 matches")
    home_rolling_sot_5: float = Field(..., ge=0, description="Average shots on target per match in last 5 matches")
    away_rolling_sot_5: float = Field(..., ge=0, description="Average shots on target per match in last 5 matches")
    home_rolling_sot_conceded_5: float = Field(..., ge=0, description="Average shots on target conceded in last 5 matches")
    away_rolling_sot_conceded_5: float = Field(..., ge=0, description="Average shots on target conceded in last 5 matches")
    home_rolling_corners_5: float = Field(..., ge=0, description="Average corner kicks earned per match in last 5 matches")
    away_rolling_corners_5: float = Field(..., ge=0, description="Average corner kicks earned per match in last 5 matches")
    home_rolling_win_pct_5: float = Field(..., ge=0, le=1.0, description="Win rate over last 5 matches (0.0 to 1.0)")
    away_rolling_win_pct_5: float = Field(..., ge=0, le=1.0, description="Win rate over last 5 matches (0.0 to 1.0)")
    home_league_rank: int = Field(..., ge=1, le=20, description="Home team table position before kickoff (1-20)")
    away_league_rank: int = Field(..., ge=1, le=20, description="Away team table position before kickoff (1-20)")
    rank_diff: Optional[int] = Field(None, description="away_league_rank - home_league_rank (computed automatically if omitted)")
    home_ppg: float = Field(..., ge=0, le=3.0, description="Home points per game in current season")
    away_ppg: float = Field(..., ge=0, le=3.0, description="Away points per game in current season")
    h2h_matches_prior: Optional[int] = Field(0, ge=0, description="Number of prior historical matches between these clubs")
    h2h_home_win_pct: Optional[float] = Field(0.45, ge=0, le=1.0, description="Historical home win rate in head-to-head fixtures")

    model_config = {
        "json_schema_extra": {
            "example": {
                "home_team": "Arsenal",
                "away_team": "Chelsea",
                "home_rest_days": 7.0,
                "away_rest_days": 4.0,
                "home_rolling_points_5": 12.0,
                "away_rolling_points_5": 7.0,
                "home_rolling_gf_5": 11.0,
                "away_rolling_gf_5": 6.0,
                "home_rolling_ga_5": 3.0,
                "away_rolling_ga_5": 8.0,
                "home_rolling_shots_5": 16.4,
                "away_rolling_shots_5": 12.2,
                "home_rolling_sot_5": 6.2,
                "away_rolling_sot_5": 4.0,
                "home_rolling_sot_conceded_5": 2.6,
                "away_rolling_sot_conceded_5": 5.4,
                "home_rolling_corners_5": 7.2,
                "away_rolling_corners_5": 4.6,
                "home_rolling_win_pct_5": 0.80,
                "away_rolling_win_pct_5": 0.40,
                "home_league_rank": 2,
                "away_league_rank": 9,
                "rank_diff": 7,
                "home_ppg": 2.25,
                "away_ppg": 1.45,
                "h2h_matches_prior": 12,
                "h2h_home_win_pct": 0.58
            }
        }
    }


class MatchProbabilities(BaseModel):
    """Probability distribution over the three mutually exclusive match outcomes."""
    home_win: float = Field(..., description="Estimated probability of Home Win (0.0 to 1.0)")
    draw: float = Field(..., description="Estimated probability of Draw (0.0 to 1.0)")
    away_win: float = Field(..., description="Estimated probability of Away Win (0.0 to 1.0)")


class MatchPredictResponse(BaseModel):
    """Output payload for pre-match outcome prediction."""
    success: bool = Field(True, description="Request execution status")
    home_team: str = Field(..., description="Home club name")
    away_team: str = Field(..., description="Away club name")
    predicted_outcome: str = Field(..., description="Predicted outcome code: 'H', 'D', or 'A'")
    predicted_outcome_label: str = Field(..., description="Human-readable outcome label: 'Home Win', 'Draw', or 'Away Win'")
    probabilities: MatchProbabilities = Field(..., description="Calibrated probabilities summing approximately to 1.0")
    model_type: str = Field(..., description="ML classification model architecture")
    data_as_of: Optional[str] = Field("2026-09-20", description="Latest match history timestamp")
    season: Optional[str] = Field("2026-27", description="Active Premier League campaign")


class TeamStateItem(BaseModel):
    """Current team state containing Elo rating, league table standing, and form."""
    team_name: str
    rank: int
    points: int
    matches_played: int
    ppg: float
    current_elo: float
    rolling_pts_5: float
    rolling_gf_5: float
    rolling_ga_5: float
    rolling_win_pct_5: float
    last_match_date: str


class CurrentTeamsResponse(BaseModel):
    """Envelope response for current Premier League teams and standings."""
    success: bool = True
    season: str = "2026-27"
    data_as_of: str = "2026-09-20"
    teams: List[TeamStateItem]


class CurrentFixtureFeaturesResponse(BaseModel):
    """Pre-computed pre-match feature vector for a matchup derived from current match data."""
    success: bool = True
    home_team: str
    away_team: str
    season: str = "2026-27"
    data_as_of: str = "2026-09-20"
    features: Dict[str, Any]



class MatchModelInfoResponse(BaseModel):
    """Match outcome model training metadata and validation performance."""
    model_type: str
    model_version: Optional[str] = "V1-Improved"
    architecture: Optional[str] = "XGBoost + Sequential Elo"
    feature_count: int
    features: List[str]
    train_seasons: List[str]
    validation_season: str
    test_season: str
    split_counts: Dict[str, int]
    class_labels: List[str]
    validation_metrics: Dict[str, Any]
    test_metrics: Dict[str, Any]
    training_timestamp: str
