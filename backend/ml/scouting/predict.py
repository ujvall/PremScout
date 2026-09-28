"""Inference and candidate retrieval service for the Player Scouting & Replacement module."""

from pathlib import Path
from typing import List, Dict, Any, Optional
import joblib
import pandas as pd

from .similarity import ScoutingSimilarityIndex

DEFAULT_MODEL_PATH = Path("models/scouting/scouting_model.joblib")

_CACHED_INDEX: Optional[ScoutingSimilarityIndex] = None
_CACHED_ARTIFACT: Optional[Dict[str, Any]] = None


def load_scouting_index(model_path: Path = DEFAULT_MODEL_PATH) -> ScoutingSimilarityIndex:
    """Load and cache the trained Player Scouting similarity index artifact."""
    global _CACHED_INDEX, _CACHED_ARTIFACT
    if _CACHED_INDEX is None:
        if not model_path.exists():
            raise FileNotFoundError(
                f"Scouting model artifact not found at {model_path}. "
                "Run 'python -m backend.ml.scouting.train' first."
            )
        _CACHED_ARTIFACT = joblib.load(model_path)
        _CACHED_INDEX = ScoutingSimilarityIndex(
            player_df=_CACHED_ARTIFACT["player_df"],
            embeddings=_CACHED_ARTIFACT["embeddings"],
            features=_CACHED_ARTIFACT["features"],
            scaler=_CACHED_ARTIFACT.get("scaler"),
            position_weights=_CACHED_ARTIFACT.get("position_weights"),
        )
    return _CACHED_INDEX


def get_similar_players(
    player_name: str,
    top_n: int = 10,
    model_path: Path = DEFAULT_MODEL_PATH,
) -> List[Dict[str, Any]]:
    """Retrieve the top N most similar players based on football performance profile.

    Args:
        player_name: Name of target player.
        top_n: Number of candidates to return (default: 10).
        model_path: Path to serialized scouting model artifact.

    Returns:
        List of similar player dictionaries with similarity scores (0-100) and profiles.
    """
    return find_replacements(
        player_name=player_name,
        top_n=top_n,
        same_position_only=False,  # allows compatible positions
        max_market_value=None,
        max_age=None,
        min_minutes=None,
        model_path=model_path,
    )


def find_replacements(
    player_name: str,
    top_n: int = 10,
    same_position_only: bool = True,
    max_market_value: Optional[float] = None,
    max_age: Optional[int] = None,
    min_minutes: Optional[int] = None,
    season: Optional[str] = None,
    model_path: Path = DEFAULT_MODEL_PATH,
) -> List[Dict[str, Any]]:
    """Find ranked replacement candidates for a target player subject to scouting constraints.

    Args:
        player_name: Target player name to replace.
        top_n: Number of candidate recommendations (default: 10).
        same_position_only: Restrict strictly to exact position group (default: True).
        max_market_value: Maximum affordable market valuation in EUR (optional).
        max_age: Maximum player age in years (optional).
        min_minutes: Minimum minutes played in season (optional).
        season: Specific season to match target against (e.g. '2023-24'). Defaults to latest.
        model_path: Path to model artifact.

    Returns:
        List of candidate dictionaries ordered by similarity score descending.

    Raises:
        ValueError: If player_name is invalid, unknown, or if filter values are negative/invalid.
    """
    if not player_name or not isinstance(player_name, str):
        raise ValueError("Target player_name must be a non-empty string.")

    if top_n <= 0:
        raise ValueError(f"top_n must be a positive integer, got {top_n}")

    if max_market_value is not None and max_market_value <= 0:
        raise ValueError(f"max_market_value must be positive, got {max_market_value}")

    if max_age is not None and (max_age < 15 or max_age > 50):
        raise ValueError(f"max_age must be a reasonable football age (15-50), got {max_age}")

    if min_minutes is not None and min_minutes < 0:
        raise ValueError(f"min_minutes cannot be negative, got {min_minutes}")

    index = load_scouting_index(model_path)

    results = index.find_replacements(
        player_name=player_name,
        top_n=top_n,
        same_position_only=same_position_only,
        max_market_value=max_market_value,
        max_age=max_age,
        min_minutes=min_minutes,
        season=season,
    )

    if not results:
        # Provide helpful diagnostic error if candidate pool was overly restricted
        raise ValueError(
            f"No eligible replacement candidates found for '{player_name}' with the given filters "
            f"(same_position_only={same_position_only}, max_market_value={max_market_value}, "
            f"max_age={max_age}, min_minutes={min_minutes}). Try loosening constraints."
        )

    return results
