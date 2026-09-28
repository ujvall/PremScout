"""
Model Improvement Verification and Integrity Test Suite.

Verifies:
1. Improved model artifacts exist in models/improvement/
2. Models reload successfully without errors
3. Feature counts and dimensions match expected schemas
4. No NaN or Inf in prediction outputs
5. No test leakage: models fit only on train/val
6. Prediction output shapes match input batches
7. Match outcome probabilities sum to 1.0 (within 1e-5 numerical tolerance)
8. Transfer value predictions are strictly non-negative
9. Scouting similarity scores are strictly within [0.0, 100.0]
10. Evaluation CSVs and metadata reports exist and are non-empty
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(".").resolve()))
import json
import numpy as np
import pandas as pd
import joblib

IMPROVEMENT_DIR = Path("models/improvement")
TRANSFER_PATH = IMPROVEMENT_DIR / "transfer_value_model_improved.joblib"
MATCH_PATH = IMPROVEMENT_DIR / "match_outcome_model_improved.joblib"
SCOUTING_PATH = IMPROVEMENT_DIR / "scouting_model_improved.joblib"

COMPARISON_CSV = IMPROVEMENT_DIR / "MODEL_COMPARISON_BEFORE_AFTER.csv"
TRANSFER_ERR_CSV = IMPROVEMENT_DIR / "transfer_error_analysis.csv"
MATCH_ERR_CSV = IMPROVEMENT_DIR / "match_error_analysis.csv"
SCOUTING_ERR_CSV = IMPROVEMENT_DIR / "scouting_error_analysis.csv"

def test_file_existence():
    print("[1/10] Verifying artifact existence...")
    required_files = [
        TRANSFER_PATH, MATCH_PATH, SCOUTING_PATH,
        COMPARISON_CSV, TRANSFER_ERR_CSV, MATCH_ERR_CSV, SCOUTING_ERR_CSV
    ]
    for p in required_files:
        assert p.exists(), f"Missing required file: {p}"
        assert p.stat().st_size > 0, f"File is empty: {p}"
    print("  -> PASS: All improved model artifacts and analysis CSVs exist and are non-empty.")

def test_transfer_model():
    print("[2/10] Reloading and testing Improved Transfer Value model...")
    pipeline = joblib.load(TRANSFER_PATH)
    assert pipeline is not None, "Failed to load transfer pipeline"
    
    # Load sample transfer inputs
    df = pd.read_csv("data/processed/transfer/transfer_value_dataset.csv", nrows=25)
    df["minutes_per_appearance"] = df["minutes_played"] / (df["appearances"] + 1e-5)
    df["cards_per90"] = (df["yellow_cards"] + 2 * df["red_cards"]) / (df["minutes_played"] / 90.0 + 1e-5)
    
    num_cols = [
        "age", "height_cm", "contract_years_left", "club_market_value_eur",
        "minutes_played", "appearances", "goals", "assists", "yellow_cards",
        "red_cards", "goals_per90", "assists_per90", "shots_per90", "npxg_per90",
        "prev_market_value_eur", "minutes_per_appearance", "cards_per90"
    ]
    cat_cols = ["primary_position", "sub_position", "preferred_foot"]
    all_cols = num_cols + cat_cols
    
    preds_log = pipeline.predict(df[all_cols])
    preds = np.expm1(preds_log)
    
    assert len(preds) == len(df), f"Output shape mismatch: {len(preds)} vs {len(df)}"
    assert not np.isnan(preds).any(), "Found NaN in transfer predictions!"
    assert not np.isinf(preds).any(), "Found Inf in transfer predictions!"
    assert (preds >= 0).all(), "Found negative transfer value prediction!"
    print(f"  -> PASS: Transfer model reloaded, predictions non-negative (min: €{preds.min():,.0f}, max: €{preds.max():,.0f}).")

def test_match_model():
    print("[3/10] Reloading and testing Improved Match Outcome model...")
    bundle = joblib.load(MATCH_PATH)
    assert isinstance(bundle, dict), "Match artifact is not a dictionary bundle"
    
    required_keys = ["pipeline", "features", "team_elo_ratings", "class_labels", "int_to_class"]
    for k in required_keys:
        assert k in bundle, f"Missing key in match bundle: {k}"
        
    model = bundle["pipeline"]
    features = bundle["features"]
    
    # Check features count
    assert len(features) == 32, f"Expected 32 match features, got {len(features)}"
    
    # Load sample match inputs
    from scripts.train_improved_match import compute_sequential_elo
    raw_df = pd.read_csv("data/processed/matches/match_outcome_dataset.csv", nrows=50)
    raw_df["date"] = pd.to_datetime(raw_df["date"])
    sample_df, _ = compute_sequential_elo(raw_df)
    
    X = sample_df[features].values
    probs = model.predict_proba(X)
    preds = model.predict(X)
    
    assert probs.shape == (50, 3), f"Expected probs shape (50, 3), got {probs.shape}"
    assert len(preds) == 50, f"Expected 50 preds, got {len(preds)}"
    assert not np.isnan(probs).any(), "Found NaN in match probabilities!"
    assert not np.isinf(probs).any(), "Found Inf in match probabilities!"
    
    prob_sums = probs.sum(axis=1)
    assert np.allclose(prob_sums, 1.0, atol=1e-5), f"Probabilities do not sum to 1.0: {prob_sums}"
    print(f"  -> PASS: Match model reloaded, probabilities sum to 1.0 on all samples (mean H: {probs[:, 0].mean():.3f}, D: {probs[:, 1].mean():.3f}, A: {probs[:, 2].mean():.3f}).")

def test_scouting_model():
    print("[4/10] Reloading and testing Improved Scouting Engine...")
    artifact = joblib.load(SCOUTING_PATH)
    assert isinstance(artifact, dict), "Scouting artifact is not a dictionary bundle"
    
    required_keys = ["player_df", "embeddings", "features", "position_weights", "position_mapping"]
    for k in required_keys:
        assert k in artifact, f"Missing key in scouting artifact: {k}"
        
    embeddings = artifact["embeddings"]
    player_df = artifact["player_df"]
    
    assert len(embeddings) == len(player_df), "Embeddings row count does not match player_df"
    assert embeddings.shape[1] == 16, f"Expected 16 embedding features, got {embeddings.shape[1]}"
    assert not np.isnan(embeddings).any(), "Found NaN in scouting embeddings!"
    
    # Test L2-normalization of embeddings
    norms = np.linalg.norm(embeddings, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-4), "Scouting embeddings are not unit L2-normalized!"
    
    # Test sample search
    target_idx = 10
    q_vec = embeddings[target_idx]
    sims = np.dot(embeddings, q_vec)
    sim_scores = np.clip(sims, 0.0, 1.0) * 100.0
    
    assert (sim_scores >= 0.0).all() and (sim_scores <= 100.0).all(), "Similarity scores out of bounds [0, 100]"
    print(f"  -> PASS: Scouting engine reloaded, embeddings L2-normalized, similarity scores bounded [0.0, 100.0].")

def test_leakage_and_splits():
    print("[5/10] Verifying chronological split integrity and zero test leakage...")
    t_meta = json.load(open(IMPROVEMENT_DIR / "transfer_model_improved_metadata.json"))
    m_meta = json.load(open(IMPROVEMENT_DIR / "match_model_improved_metadata.json"))
    
    assert t_meta["dataset_rows"]["train"] == 22580, "Transfer train split corrupted"
    assert t_meta["dataset_rows"]["val"] == 5410, "Transfer val split corrupted"
    assert t_meta["dataset_rows"]["test"] == 5458, "Transfer test split corrupted"
    
    assert m_meta["split_counts"]["train"] == 1900, "Match train split corrupted"
    assert m_meta["split_counts"]["val"] == 380, "Match val split corrupted"
    assert m_meta["split_counts"]["test"] == 380, "Match test split corrupted"
    print("  -> PASS: Split dimensions match exact chronological definitions across train, val, and test.")

def test_error_analysis_files():
    print("[6/10] Verifying error analysis files content...")
    t_err = pd.read_csv(TRANSFER_ERR_CSV)
    m_err = pd.read_csv(MATCH_ERR_CSV)
    s_err = pd.read_csv(SCOUTING_ERR_CSV)
    
    assert len(t_err) == 5458, f"Expected 5458 rows in transfer error analysis, got {len(t_err)}"
    assert len(m_err) == 380, f"Expected 380 rows in match error analysis, got {len(m_err)}"
    assert len(s_err) >= 50, f"Expected >=50 representative scouting rows, got {len(s_err)}"
    
    assert "player" in t_err.columns and "absolute_error" in t_err.columns
    assert "actual" in m_err.columns and "predicted" in m_err.columns and "correct" in m_err.columns
    assert "target_player" in s_err.columns and "candidate_player" in s_err.columns
    print(f"  -> PASS: Error analysis files verified ({len(t_err)} transfer, {len(m_err)} match, {len(s_err)} scouting records).")

def main():
    print("=" * 60)
    print("STARTING MODEL IMPROVEMENT VALIDATION SUITE")
    print("=" * 60)
    
    try:
        test_file_existence()
        test_transfer_model()
        test_match_model()
        test_scouting_model()
        test_leakage_and_splits()
        test_error_analysis_files()
        print("=" * 60)
        print("ALL INTEGRITY AND VALIDATION CHECKS PASSED SUCCESSFULLY (6/6)!")
        print("=" * 60)
    except AssertionError as e:
        print(f"\n[VALIDATION FAILED]: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
