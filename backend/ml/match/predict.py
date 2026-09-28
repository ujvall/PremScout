"""Inference service for the Match Outcome Predictor module."""

from pathlib import Path
from typing import Dict, Any, Union, List
import joblib
import numpy as np
import pandas as pd

from .preprocessing import (
    CLASS_LABELS,
    CLASS_TO_INT,
    INT_TO_CLASS,
    INT_TO_HUMAN,
    MATCH_FEATURES,
)

DEFAULT_MODEL_PATH = Path("models/match/match_outcome_model.joblib")

_CACHED_ARTIFACT = None


def load_match_model(model_path: Path = DEFAULT_MODEL_PATH) -> Dict[str, Any]:
    """Load and cache the trained Match Outcome Predictor model artifact."""
    global _CACHED_ARTIFACT
    if _CACHED_ARTIFACT is None:
        if not model_path.exists():
            raise FileNotFoundError(
                f"Match model artifact not found at {model_path}. "
                "Run 'python -m backend.ml.match.train' first."
            )
        _CACHED_ARTIFACT = joblib.load(model_path)
    return _CACHED_ARTIFACT


def predict_match(
    features: Union[Dict[str, Any], pd.DataFrame],
    model_path: Path = DEFAULT_MODEL_PATH,
) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
    """Predict probabilities and likely match outcome from pre-match features.

    Args:
        features: Dictionary of pre-match features for a single match or DataFrame for multiple matches.
        model_path: Path to serialized model artifact.

    Returns:
        Dictionary (or list of dicts) containing:
            - predicted_result: "Home Win", "Draw", or "Away Win"
            - predicted_code: "H", "D", or "A"
            - home_win_probability: float in [0, 1]
            - draw_probability: float in [0, 1]
            - away_win_probability: float in [0, 1]
            - model_type: name of the model architecture
    """
    artifact = load_match_model(model_path)
    pipeline = artifact["pipeline"]
    model_name = "XGBoost"

    is_single = isinstance(features, dict)
    if is_single:
        df = pd.DataFrame([features])
    else:
        df = features.copy()

    model_features = artifact.get("features", MATCH_FEATURES)
    team_elos = artifact.get("team_elo_ratings", {})
    home_adv = artifact.get("home_advantage_elo", 65.0)

    # Compute Elo and parity features if not present
    for idx, row in df.iterrows():
        h_team = row.get("home_team", "")
        a_team = row.get("away_team", "")
        h_elo = row.get("home_elo")
        if pd.isna(h_elo) or h_elo is None:
            h_elo = team_elos.get(h_team, 1500.0)
            df.loc[idx, "home_elo"] = h_elo
        a_elo = row.get("away_elo")
        if pd.isna(a_elo) or a_elo is None:
            a_elo = team_elos.get(a_team, 1500.0)
            df.loc[idx, "away_elo"] = a_elo

        diff = (float(h_elo) + home_adv) - float(a_elo)
        df.loc[idx, "elo_diff"] = diff
        df.loc[idx, "elo_exp_h"] = 1.0 / (1.0 + 10.0 ** (-diff / 400.0))
        df.loc[idx, "abs_elo_diff"] = abs(diff)

        rank_diff = row.get("rank_diff")
        if pd.isna(rank_diff) or rank_diff is None:
            h_rank = row.get("home_league_rank", 10)
            a_rank = row.get("away_league_rank", 10)
            rank_diff = float(a_rank) - float(h_rank)
        df.loc[idx, "abs_rank_diff"] = abs(float(rank_diff))

        h_ppg = row.get("home_ppg", 1.3)
        a_ppg = row.get("away_ppg", 1.3)
        df.loc[idx, "abs_ppg_diff"] = abs(float(h_ppg) - float(a_ppg))

    # Fill any missing required feature columns with NaN (handled by imputer in pipeline)
    for col in model_features:
        if col not in df.columns:
            df[col] = np.nan

    X_subset = df[model_features]

    # Predict probabilities (shape: N, 3 corresponding to class indices 0: H, 1: D, 2: A)
    probs = pipeline.predict_proba(X_subset)
    preds = pipeline.predict(X_subset)

    results = []
    for pred_idx, prob_row in zip(preds, probs):
        # Normalize slightly to guarantee sum == 1.0
        norm_prob = prob_row / np.sum(prob_row)
        home_p = round(float(norm_prob[0]), 4)
        draw_p = round(float(norm_prob[1]), 4)
        away_p = round(float(norm_prob[2]), 4)

        predicted_code = INT_TO_CLASS[int(pred_idx)]
        predicted_human = INT_TO_HUMAN[int(pred_idx)]

        results.append({
            "predicted_result": predicted_human,
            "predicted_code": predicted_code,
            "home_win_probability": home_p,
            "draw_probability": draw_p,
            "away_win_probability": away_p,
            "model_type": model_name,
        })

    return results[0] if is_single else results
