"""Pydantic schemas package for the PremAI API."""

from .common import APIResponse, RootResponse, HealthResponse
from .transfer import (
    TransferPredictRequest,
    TransferPredictResponse,
    TransferModelInfoResponse,
)
from .match import (
    MatchPredictRequest,
    MatchProbabilities,
    MatchPredictResponse,
    MatchModelInfoResponse,
)
from .scouting import (
    ScoutingReplacementRequest,
    CandidateStats,
    ReplacementCandidate,
    PlayerSearchItem,
    ScoutingModelInfoResponse,
)

__all__ = [
    "APIResponse",
    "RootResponse",
    "HealthResponse",
    "TransferPredictRequest",
    "TransferPredictResponse",
    "TransferModelInfoResponse",
    "MatchPredictRequest",
    "MatchProbabilities",
    "MatchPredictResponse",
    "MatchModelInfoResponse",
    "ScoutingReplacementRequest",
    "CandidateStats",
    "ReplacementCandidate",
    "PlayerSearchItem",
    "ScoutingModelInfoResponse",
]
