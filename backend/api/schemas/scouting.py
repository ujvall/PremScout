"""Pydantic schemas for the Player Scouting & Replacement API."""

from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field


class ScoutingReplacementRequest(BaseModel):
    """Input parameters for finding replacement candidates for a player."""
    player_name: str = Field(..., min_length=1, description="Target player name to find replacements for")
    top_n: int = Field(10, ge=1, le=50, description="Number of candidate recommendations to return (1-50)")
    same_position_only: bool = Field(True, description="Restrict strictly to exact normalized position group")
    max_market_value: Optional[float] = Field(None, gt=0, description="Maximum affordable market valuation in EUR")
    max_age: Optional[int] = Field(None, ge=15, le=45, description="Maximum candidate age in years (15-45)")
    min_minutes: Optional[int] = Field(None, ge=0, description="Minimum seasonal minutes played")

    model_config = {
        "json_schema_extra": {
            "example": {
                "player_name": "Declan Rice",
                "top_n": 10,
                "same_position_only": True,
                "max_market_value": 50000000.0,
                "max_age": 28,
                "min_minutes": 900
            }
        }
    }


class CandidateStats(BaseModel):
    """Core per-90 comparative metrics for a recommended player candidate."""
    goals_per90: float
    assists_per90: float
    npxG_per90: float
    xAG_per90: float
    key_passes_per90: float
    pass_completion_pct: float
    progressive_passes_per90: float
    progressive_carries_per90: float
    tackles_won_per90: float
    interceptions_per90: float


class ReplacementCandidate(BaseModel):
    """Structured recommendation item representing a similar football player."""
    target_player: str
    target_team: str
    target_position: str
    target_season: str
    candidate_player: str
    candidate_club: str
    candidate_position: str
    candidate_raw_position: str
    candidate_season: str
    similarity_score: float = Field(..., description="Normalized similarity index on a 0.0 to 100.0 scale")
    cosine_similarity: float = Field(..., description="Raw cosine similarity in standardized feature space")
    position_compatibility: str = Field(..., description="'Same Position' or 'Compatible Position'")
    age: Optional[int]
    market_value_eur: Optional[float]
    minutes: int
    key_stats: CandidateStats
    data_as_of: Optional[str] = "2026-09-20"


class PlayerSearchItem(BaseModel):
    """Lightweight player summary item for frontend autocomplete search dropdowns."""
    player_name: str
    player_id: Optional[str] = None
    position: str
    normalized_position: str
    club: str
    season: str
    age: Optional[int] = None
    market_value_eur: Optional[float] = None
    data_as_of: Optional[str] = "2026-09-20"
    has_scouting_data: Optional[bool] = False
    is_current: Optional[bool] = True


class ScoutingModelInfoResponse(BaseModel):
    """Scouting vector similarity engine metadata and evaluation metrics."""
    model_type: str = Field("VectorSimilarityEngine", description="Model architecture")
    model_version: Optional[str] = Field("V2", description="Production model version")
    architecture: Optional[str] = Field("Archetype-Weighted Cosine Similarity", description="Human-readable model architecture")
    dataset_path: str
    dataset_row_count: int
    distinct_players_count: int
    features_used: List[str]
    feature_count: int
    position_mapping: Dict[str, str]
    preprocessing_method: str
    similarity_method: str
    scaling_method: str
    filters_supported: List[str]
    evaluation_metrics: Dict[str, Any]
    training_timestamp: str


class ScoutingReplacementResponse(BaseModel):
    """Envelope response for replacement recommendations."""
    success: bool = Field(True, description="Request execution status")
    target_player: str
    count: int
    data: List[ReplacementCandidate]


class PlayerSearchResponse(BaseModel):
    """Envelope response for player search catalog."""
    success: bool = Field(True, description="Request execution status")
    count: int
    data: List[PlayerSearchItem]
