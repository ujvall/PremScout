"""Inference service for the Transfer Value Predictor module."""

from pathlib import Path
import joblib
import numpy as np
import pandas as pd

from .evaluate import format_eur
from .preprocessing import ALL_MODEL_FEATURES

DEFAULT_MODEL_PATH = Path("models/transfer/transfer_value_model.joblib")

# Cache model in memory
_CACHED_ARTIFACT = None


def load_transfer_model(model_path: Path = DEFAULT_MODEL_PATH) -> dict:
    """Load and cache the trained Transfer Value Predictor model artifact."""
    global _CACHED_ARTIFACT
    if _CACHED_ARTIFACT is None:
        if not model_path.exists():
            raise FileNotFoundError(
                f"Transfer model artifact not found at {model_path}. "
                "Run 'python -m backend.ml.transfer.train' first."
            )
        _CACHED_ARTIFACT = joblib.load(model_path)
    return _CACHED_ARTIFACT


def predict_transfer_value(
    player_features: dict | pd.DataFrame,
    model_path: Path = DEFAULT_MODEL_PATH,
) -> dict | list[dict]:
    """Predict realistic market/transfer value for a football player.

    Args:
        player_features: dict of features for a single player or DataFrame for multiple players.
        model_path: Path to serialized model artifact.

    Returns:
        Dictionary (or list of dicts) with predicted valuation in EUR and human-readable format.
    """
    artifact = load_transfer_model(model_path)
    preprocessor = artifact["preprocessor"]
    target_transform = artifact.get("target_transform") or artifact.get("target_transformation", "raw")
    model_name = "XGBoost"

    is_single = isinstance(player_features, dict)
    if is_single:
        df = pd.DataFrame([player_features])
    else:
        df = player_features.copy()

    model_features = artifact.get("features", ALL_MODEL_FEATURES)

    # Ensure all required features are present, computing engineered features if necessary
    if "minutes_per_appearance" in model_features and "minutes_per_appearance" not in df.columns:
        mins = df["minutes_played"].fillna(0.0)
        apps = df["appearances"].fillna(1.0)
        df["minutes_per_appearance"] = mins / (apps + 1e-5)
    if "cards_per90" in model_features and "cards_per90" not in df.columns:
        mins = df["minutes_played"].fillna(0.0)
        df["cards_per90"] = (df["yellow_cards"].fillna(0) + 2 * df["red_cards"].fillna(0)) / (mins / 90.0 + 1e-5)
    if "age_x_minutes" in model_features and "age_x_minutes" not in df.columns:
        df["age_x_minutes"] = (df["age"] * df["minutes_played"].fillna(0.0)) / 1000.0
    if "age_x_goals" in model_features and "age_x_goals" not in df.columns:
        df["age_x_goals"] = df["age"] * df["goals_per90"].fillna(0.0)
    if "contract_x_prev_val" in model_features and "contract_x_prev_val" not in df.columns:
        df["contract_x_prev_val"] = (df["contract_years_left"].fillna(2.0) * df["prev_market_value_eur"].fillna(0.0)) / 1e6
    if "youth_starter_premium" in model_features and "youth_starter_premium" not in df.columns:
        mpa = df.get("minutes_per_appearance", pd.Series([0.0] * len(df)))
        df["youth_starter_premium"] = np.where((df["age"] <= 23) & (mpa >= 65), 1.0, 0.0)
    if "veteran_depreciation" in model_features and "veteran_depreciation" not in df.columns:
        df["veteran_depreciation"] = np.maximum(0.0, df["age"] - 30.0)
    if "val_growth_pct" in model_features and "val_growth_pct" not in df.columns:
        df["val_growth_pct"] = 0.0

    # Fill any missing required feature columns with defaults
    for col in model_features:
        if col not in df.columns:
            if col in artifact.get("numerical_features", []):
                df[col] = np.nan
            else:
                df[col] = "Unknown"

    X_subset = df[model_features]

    # Preprocess
    X_proc = preprocessor.transform(X_subset)

    # Predict with single model or ensemble
    if "model" in artifact:
        raw_preds = artifact["model"].predict(X_proc)
    elif "xgb_model" in artifact and "hgb_model" in artifact:
        weights = artifact.get("ensemble_weights", {"xgb": 0.526, "hgb": 0.474})
        p_xgb = artifact["xgb_model"].predict(X_proc)
        p_hgb = artifact["hgb_model"].predict(X_proc)
        raw_preds = weights["xgb"] * p_xgb + weights["hgb"] * p_hgb
    else:
        raise KeyError("Artifact contains neither 'model' nor ('xgb_model', 'hgb_model')")

    # Invert log1p if model was trained on log target
    if target_transform == "log1p":
        preds_eur = np.expm1(raw_preds)
    else:
        preds_eur = raw_preds

    # Clip to realistic non-negative floor
    preds_eur = np.clip(preds_eur, a_min=25_000, a_max=350_000_000)

    results = []
    for val in preds_eur:
        rounded_val = float(np.round(val, -3))  # round to nearest thousand
        results.append({
            "predicted_market_value_eur": rounded_val,
            "formatted_market_value": format_eur(rounded_val),
            "model_type": model_name,
            "target_transformation": target_transform,
        })

    return results[0] if is_single else results
