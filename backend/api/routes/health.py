"""Health check router for the PremAI API."""

from fastapi import APIRouter
from backend.api.schemas.common import HealthResponse, APIResponse
from backend.services.transfer_service import check_transfer_model_health
from backend.services.match_service import check_match_model_health
from backend.services.scouting_service import check_scouting_model_health

router = APIRouter(prefix="/health", tags=["Health"])


@router.get(
    "",
    response_model=HealthResponse,
    summary="System Health & Model Artifact Availability",
    description="Actively verifies that the three serialized ML model artifacts (Transfer, Match, Scouting) exist and are loadable into memory.",
)
def get_health() -> HealthResponse:
    """Check availability of each machine learning module artifact."""
    transfer_healthy = check_transfer_model_health()
    match_healthy = check_match_model_health()
    scouting_healthy = check_scouting_model_health()

    all_healthy = transfer_healthy and match_healthy and scouting_healthy
    status_str = "healthy" if all_healthy else "degraded"

    return HealthResponse(
        status=status_str,
        models={
            "transfer": transfer_healthy,
            "match": match_healthy,
            "scouting": scouting_healthy,
        },
        data_as_of="2026-09-20",
        current_data_layer=True,
    )
