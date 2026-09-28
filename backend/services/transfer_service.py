"""Business service layer for Transfer Value Predictor module."""

import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
import numpy as np
import pandas as pd
from fastapi import HTTPException

from backend.api.schemas.transfer import (
    TransferPredictRequest,
    TransferPredictResponse,
    TransferModelInfoResponse,
)
from backend.ml.transfer.predict import predict_transfer_value, load_transfer_model

logger = logging.getLogger(__name__)

TRANSFER_MODEL_PATH = Path("models/transfer/transfer_value_model.joblib")
TRANSFER_METADATA_PATH = Path("models/transfer/model_metadata.json")


def check_transfer_model_health() -> bool:
    """Verify that the Transfer Value Predictor model artifact is present and loadable."""
    try:
        if not TRANSFER_MODEL_PATH.exists():
            return False
        artifact = load_transfer_model(TRANSFER_MODEL_PATH)
        return ("model" in artifact or "xgb_model" in artifact) and "preprocessor" in artifact
    except Exception as e:
        logger.error(f"Transfer model health check failed: {e}")
        return False


def predict_transfer_value_service(request: TransferPredictRequest) -> TransferPredictResponse:
    """Generate transfer market valuation prediction from player features.

    Args:
        request: Validated Pydantic request model.

    Returns:
        TransferPredictResponse with predicted EUR valuation and formatted string.
    """
    try:
        # Convert request to dictionary matching model features
        feature_dict = request.model_dump()
        result = predict_transfer_value(feature_dict, model_path=TRANSFER_MODEL_PATH)

        return TransferPredictResponse(
            player_name=request.player_name or "Unknown Player",
            predicted_market_value_eur=result["predicted_market_value_eur"],
            predicted_market_value_formatted=result["formatted_market_value"],
            model_type=result.get("model_type", "XGBoost"),
            target_transformation=result.get("target_transformation", "log1p"),
        )
    except FileNotFoundError as e:
        logger.error(f"Transfer model artifact missing: {e}")
        raise HTTPException(status_code=503, detail="Transfer Value Predictor model is currently unavailable.")
    except Exception as e:
        logger.error(f"Error executing transfer prediction: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to generate transfer valuation: {str(e)}")


def get_transfer_model_info_service() -> TransferModelInfoResponse:
    """Retrieve metadata, dataset statistics, and evaluation scores for the transfer model."""
    if not TRANSFER_METADATA_PATH.exists():
        raise HTTPException(status_code=404, detail="Transfer model metadata not found.")

    try:
        with open(TRANSFER_METADATA_PATH, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        return TransferModelInfoResponse(**metadata)
    except Exception as e:
        logger.error(f"Error loading transfer model metadata: {e}")
        raise HTTPException(status_code=500, detail="Failed to load transfer model metadata.")


CURRENT_TRANSFER_FEATURES_PATH = Path("data/current/players/current_transfer_features.csv")
_CACHED_TRANSFER_FEATURES: Optional[pd.DataFrame] = None


def _load_current_transfer_features() -> pd.DataFrame:
    """Load and cache current transfer feature matrix."""
    global _CACHED_TRANSFER_FEATURES
    if _CACHED_TRANSFER_FEATURES is None:
        if not CURRENT_TRANSFER_FEATURES_PATH.exists():
            raise HTTPException(status_code=503, detail="Current transfer features dataset not available.")
        import pandas as pd
        _CACHED_TRANSFER_FEATURES = pd.read_csv(CURRENT_TRANSFER_FEATURES_PATH, encoding="utf-8")
    return _CACHED_TRANSFER_FEATURES


def get_player_transfer_features_service(player_name: str) -> Dict[str, Any]:
    """Look up latest available current transfer model input features for a player."""
    import unicodedata
    def norm_s(s):
        if not isinstance(s, str): return ""
        n = unicodedata.normalize("NFKD", s)
        n = "".join(c for c in n if not unicodedata.combining(c)).replace("'", "").replace("-", " ").replace(".", "")
        return " ".join(n.lower().split())

    df = _load_current_transfer_features()
    q_norm = norm_s(player_name)

    # Search for matching player
    matches = df[df["normalized_name"] == q_norm]
    if matches.empty:
        # Try token match or substring match
        matches = df[df["normalized_name"].str.contains(q_norm, na=False, regex=False)]
    if matches.empty:
        raise HTTPException(status_code=404, detail=f"Player '{player_name}' not found in current transfer database.")

    row = matches.iloc[0]

    feature_cols = [
        "age", "height_cm", "contract_years_left", "club_market_value_eur",
        "minutes_played", "appearances", "goals", "assists",
        "yellow_cards", "red_cards", "goals_per90", "assists_per90",
        "shots_per90", "npxg_per90", "prev_market_value_eur",
        "primary_position", "sub_position", "preferred_foot"
    ]

    features = {}
    for col in feature_cols:
        val = row.get(col)
        if pd.isna(val):
            val = None
        elif isinstance(val, (np.floating, float)):
            val = round(float(val), 2)
        elif isinstance(val, (np.integer, int)):
            val = int(val)
        features[col] = val

    return {
        "success": True,
        "player_name": row["player_name"],
        "club": row.get("club", "Unknown"),
        "season": row.get("season", "2025-26"),
        "data_as_of": "2026-09-20",
        "features": features,
    }


def get_transfer_presets_service() -> Dict[str, Any]:
    """Retrieve dynamically generated presets for key current Premier League players."""
    target_players = [
        ("rayan cherki", "cherki", "Creative Playmaker • Man City (Age 23)"),
        ("erling haaland", "haaland", "Generational Striker • Man City (Age 26)"),
        ("bukayo saka", "saka", "Elite Winger • Arsenal (Age 25)"),
        ("declan rice", "rice", "Box-to-Box Midfielder • Arsenal (Age 27)"),
        ("virgil van dijk", "van_dijk", "Veteran Centre-Back • Liverpool (Age 35)"),
        ("emiliano martinez", "martinez", "World Cup Winner GK • Aston Villa (Age 34)"),
    ]

    presets = []
    for norm_name, preset_id, desc in target_players:
        try:
            player_info = get_player_transfer_features_service(norm_name)
            p_features = player_info["features"].copy()
            p_features["player_name"] = player_info["player_name"]
            presets.append({
                "id": preset_id,
                "name": player_info["player_name"],
                "club": player_info["club"],
                "position": p_features.get("primary_position", "Attack"),
                "description": desc,
                "season": player_info["season"],
                "data_as_of": "2026-09-20",
                "features": p_features,
            })
        except Exception as e:
            logger.warning(f"Could not build dynamic preset for {norm_name}: {e}")

    return {
        "success": True,
        "data_as_of": "2026-09-20",
        "presets": presets,
    }
