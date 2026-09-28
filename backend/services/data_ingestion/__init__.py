"""Data ingestion package for Football Intelligence Platform (PremAI)."""

from .download_matches import download_all_matches
from .download_transfer_data import download_all_transfer_data
from .download_scouting_data import download_all_scouting_data

__all__ = [
    "download_all_matches",
    "download_all_transfer_data",
    "download_all_scouting_data",
]
