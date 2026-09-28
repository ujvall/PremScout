"""API route routers package for PremAI."""

import json
from pathlib import Path
from fastapi import APIRouter
from .health import router as health_router
from .transfer import router as transfer_router
from .match import router as match_router
from .scouting import router as scouting_router

api_router = APIRouter(prefix="/api")
api_router.include_router(health_router)
api_router.include_router(transfer_router)
api_router.include_router(match_router)
api_router.include_router(scouting_router)


@api_router.get(
    "/data-status",
    summary="Current Data Freshness Status",
    description="Returns data freshness timestamps, coverage dates, and data source metadata.",
    tags=["Metadata"],
)
def get_data_status():
    """Retrieve timestamps and coverage bounds for the current data layer."""
    status_path = Path("data/current/metadata/current_data_status.json")
    if not status_path.exists():
        return {
            "status": "historical_only",
            "data_as_of": "2024-05-19",
            "message": "Current data layer not initialized",
        }
    with open(status_path, "r", encoding="utf-8") as f:
        return json.load(f)


__all__ = [
    "api_router",
    "health_router",
    "transfer_router",
    "match_router",
    "scouting_router",
]
