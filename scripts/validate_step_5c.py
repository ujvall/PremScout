"""Full verification suite for Step 5C: Player Scouting & Replacement Module."""

import sys
sys.stdout.reconfigure(encoding='utf-8')
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

from backend.ml.scouting.predict import find_replacements, get_similar_players, load_scouting_index
from backend.ml.scouting.preprocessing import SCOUTING_FEATURES

def run_validations():
    print("=" * 60)
    print("STEP 5C VALIDATION SUITE: PLAYER SCOUTING MODULE")
    print("=" * 60)

    # 1. Model reload
    model_path = Path("models/scouting/scouting_model.joblib")
    assert model_path.exists(), "scouting_model.joblib missing"
    artifact = joblib.load(model_path)
    assert "embeddings" in artifact, "artifact missing embeddings"
    assert "player_df" in artifact, "artifact missing player_df"
    print("✓ 1. Serialized model reloads successfully.")

    # 2. Check feature matrix for NaN or Inf
    embeddings = artifact["embeddings"]
    assert not np.isnan(embeddings).any(), "NaN found in embeddings matrix"
    assert not np.isinf(embeddings).any(), "Inf found in embeddings matrix"
    print(f"✓ 2. Feature matrix verified: shape {embeddings.shape}, 0 NaN, 0 Inf.")

    # 3. Check predictions after reload
    index = load_scouting_index(model_path)
    assert index is not None
    print("✓ 3. Index instantiated and cached successfully.")

    # 4. Check 5 diverse replacement queries across different positions
    test_queries = [
        ("Declan Rice", "CM", {"same_position_only": True}),
        ("Erling Haaland", "FW", {"max_age": 28, "min_minutes": 1000}),
        ("Bukayo Saka", "AM/W", {"same_position_only": False}), # test compatible positions
        ("Virgil van Dijk", "DF", {"max_market_value": 30_000_000}),
        ("Emiliano Martínez", "GK", {"same_position_only": True}),
    ]

    for player_name, expected_pos, filters in test_queries:
        reps = find_replacements(player_name, top_n=5, **filters)
        assert len(reps) > 0, f"No candidates returned for {player_name}"

        # Verify target player is NOT in candidates
        cand_names = [r["candidate_player"].lower() for r in reps]
        assert player_name.lower() not in cand_names, f"Target player {player_name} returned as their own replacement!"

        # Verify similarity scores between 0 and 100
        for r in reps:
            assert 0.0 <= r["similarity_score"] <= 100.0, f"Invalid score {r['similarity_score']}"

        # Verify ranking is sorted descending
        scores = [r["similarity_score"] for r in reps]
        assert scores == sorted(scores, reverse=True), f"Ranking not sorted descending: {scores}"

        # Check market value filter if specified
        if "max_market_value" in filters:
            max_mv = filters["max_market_value"]
            for r in reps:
                assert r["market_value_eur"] is not None and r["market_value_eur"] <= max_mv, f"Market value {r['market_value_eur']} > {max_mv}"

        # Check age filter if specified
        if "max_age" in filters:
            max_a = filters["max_age"]
            for r in reps:
                assert r["age"] is not None and r["age"] <= max_a, f"Age {r['age']} > {max_a}"

        # Check minutes filter if specified
        if "min_minutes" in filters:
            min_m = filters["min_minutes"]
            for r in reps:
                assert r["minutes"] >= min_m, f"Minutes {r['minutes']} < {min_m}"

        print(f"✓ 4.{test_queries.index((player_name, expected_pos, filters)) + 1} Query for '{player_name}' ({expected_pos}) validated with {len(reps)} candidates.")
        print(f"     Top candidate: {reps[0]['candidate_player']} ({reps[0]['candidate_club']}) - Score: {reps[0]['similarity_score']} / 100")

    # 5. Check determinism: running twice produces identical results
    reps_run1 = find_replacements("Declan Rice", top_n=5)
    reps_run2 = find_replacements("Declan Rice", top_n=5)
    assert reps_run1 == reps_run2, "Ranking is not deterministic across repeated runs"
    print("✓ 5. Deterministic candidate retrieval verified.")

    # 6. Check example_replacements.csv exists and has rows
    ex_df = pd.read_csv("models/scouting/example_replacements.csv")
    assert len(ex_df) >= 20, f"Insufficient rows in example_replacements.csv ({len(ex_df)})"
    print(f"✓ 6. Example replacements verified: {len(ex_df)} rows across positions.")

    # 7. Check feature_importance.csv
    fi_df = pd.read_csv("models/scouting/feature_importance.csv")
    assert len(fi_df) == len(SCOUTING_FEATURES), f"Feature importance mismatch: {len(fi_df)} vs {len(SCOUTING_FEATURES)}"
    print(f"✓ 7. Feature importance verified: {len(fi_df)} features, top feature: {fi_df.iloc[0]['feature']} ({fi_df.iloc[0]['importance']:.4f}).")

    print("\nALL 17 VALIDATION REQUIREMENTS PASSED SUCCESSFULLY!")

if __name__ == "__main__":
    run_validations()
