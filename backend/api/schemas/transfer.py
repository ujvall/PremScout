"""Pydantic schemas for the Transfer Value Predictor API."""

from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field


class TransferPredictRequest(BaseModel):
    """Input features for player transfer valuation regression."""
    player_name: Optional[str] = Field("Unknown Player", description="Player's full name")
    age: float = Field(..., ge=15, le=45, description="Player age in years")
    height_cm: Optional[float] = Field(None, ge=150, le=220, description="Height in centimeters")
    contract_years_left: Optional[float] = Field(None, ge=0, le=10, description="Years remaining on current contract")
    club_market_value_eur: Optional[float] = Field(None, ge=0, description="Total squad valuation of player's club in EUR")
    minutes_played: Optional[float] = Field(None, ge=0, description="Total minutes played across the season")
    appearances: Optional[float] = Field(None, ge=0, description="Total match appearances across the season")
    goals: Optional[float] = Field(0.0, ge=0, description="Total goals scored")
    assists: Optional[float] = Field(0.0, ge=0, description="Total assists provided")
    yellow_cards: Optional[float] = Field(0.0, ge=0, description="Total yellow cards received")
    red_cards: Optional[float] = Field(0.0, ge=0, description="Total red cards received")
    goals_per90: Optional[float] = Field(None, ge=0, description="Goals scored per 90 minutes")
    assists_per90: Optional[float] = Field(None, ge=0, description="Assists per 90 minutes")
    shots_per90: Optional[float] = Field(None, ge=0, description="Total shots attempted per 90 minutes")
    npxg_per90: Optional[float] = Field(None, ge=0, description="Non-penalty expected goals (npxG) per 90 minutes")
    prev_market_value_eur: Optional[float] = Field(None, ge=0, description="Player's market valuation at end of prior season in EUR")
    primary_position: Optional[str] = Field("Attack", description="Primary position category: Attack, Midfield, Defender, or Goalkeeper")
    sub_position: Optional[str] = Field(None, description="Specific tactical sub-position (e.g. Centre-Forward, Centre-Back)")
    preferred_foot: Optional[str] = Field("Right", description="Preferred kicking foot: Right, Left, or Both")

    model_config = {
        "json_schema_extra": {
            "example": {
                "player_name": "Declan Rice",
                "age": 25.0,
                "height_cm": 188.0,
                "contract_years_left": 4.0,
                "club_market_value_eur": 1040000000.0,
                "minutes_played": 3200.0,
                "appearances": 38.0,
                "goals": 7.0,
                "assists": 8.0,
                "yellow_cards": 5.0,
                "red_cards": 0.0,
                "goals_per90": 0.20,
                "assists_per90": 0.23,
                "shots_per90": 1.35,
                "npxg_per90": 0.15,
                "prev_market_value_eur": 90000000.0,
                "primary_position": "Midfield",
                "sub_position": "Defensive Midfield",
                "preferred_foot": "Right"
            }
        }
    }


class TransferPredictResponse(BaseModel):
    """Output payload for transfer market value prediction."""
    success: bool = Field(True, description="Request execution status")
    player_name: str = Field(..., description="Player's full name")
    predicted_market_value_eur: float = Field(..., description="Predicted continuous market valuation in EUR")
    predicted_market_value_formatted: str = Field(..., description="Formatted valuation string in millions of EUR (€M)")
    model_type: str = Field(..., description="ML model architecture used for inference")
    target_transformation: str = Field(..., description="Target scale transformation applied during model training")


class TransferModelInfoResponse(BaseModel):
    """Model training metadata and performance specifications."""
    model_type: str
    model_version: Optional[str] = "V2"
    architecture: Optional[str] = "XGBoost + HistGradientBoosting Ensemble"
    target_transformation: str
    training_seasons: List[str]
    validation_season: str
    test_season: str
    dataset_rows: Dict[str, int]
    feature_count: int
    features: List[str]
    validation_metrics: Dict[str, Any]
    test_metrics: Dict[str, Any]
    training_timestamp: str


class PlayerTransferFeaturesResponse(BaseModel):
    """Response containing latest current transfer features for a player."""
    success: bool = Field(True, description="Request execution status")
    player_name: str
    club: str
    season: str
    data_as_of: str = "2026-09-20"
    features: Dict[str, Any]


class TransferPresetItem(BaseModel):
    """Dynamically generated preset profile based on current data."""
    id: str
    name: str
    club: str
    position: str
    description: str
    season: str = "2025-26"
    data_as_of: str = "2026-09-20"
    features: Dict[str, Any]


class TransferPresetsResponse(BaseModel):
    """Envelope response for current player presets."""
    success: bool = True
    data_as_of: str = "2026-09-20"
    presets: List[TransferPresetItem]

