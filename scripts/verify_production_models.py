import sys
from pathlib import Path
sys.path.insert(0, str(Path(".").resolve()))
import joblib
import json
import numpy as np
import pandas as pd
from backend.ml.transfer.predict import predict_transfer_value, load_transfer_model
from backend.ml.match.predict import predict_match, load_match_model
from backend.ml.scouting.predict import find_replacements, load_scouting_index

print("=" * 70)
print("PREMAI PRODUCTION MODEL VERIFICATION")
print("=" * 70)

print("\n--- 1. VERIFYING PRODUCTION TRANSFER MODEL ---")
t_art = load_transfer_model(Path("models/transfer/transfer_value_model.joblib"))
with open("models/transfer/model_metadata.json", "r", encoding="utf-8") as f:
    t_meta = json.load(f)

assert "xgb_model" in t_art or "model" in t_art, "Transfer model missing model key"
assert "preprocessor" in t_art, "Transfer missing preprocessor"
assert len(t_art["features"]) == t_meta["v2_feature_count"], "Feature count mismatch"

t_sample = {
    "player_name": "Declan Rice",
    "age": 25.0,
    "height_cm": 188.0,
    "contract_years_left": 4.0,
    "club_market_value_eur": 1040000000.0,
    "minutes_played": 3200.0,
    "appearances": 38.0,
    "goals": 7.0,
    "assists": 8.0,
    "yellow_cards": 5.0,
    "red_cards": 0.0,
    "goals_per90": 0.20,
    "assists_per90": 0.23,
    "shots_per90": 1.35,
    "npxg_per90": 0.15,
    "prev_market_value_eur": 90000000.0,
    "primary_position": "Midfield",
    "sub_position": "Defensive Midfield",
    "preferred_foot": "Right"
}
t_res = predict_transfer_value(t_sample)
assert t_res["predicted_market_value_eur"] > 0, "Transfer prediction is not positive"
print(f"Transfer verification: PASS! Valuation: {t_res['formatted_market_value']} ({t_res['predicted_market_value_eur']:,.2f} EUR)")

print("\n--- 2. VERIFYING PRODUCTION MATCH MODEL ---")
m_art = load_match_model(Path("models/match/match_outcome_model.joblib"))
with open("models/match/model_metadata.json", "r", encoding="utf-8") as f:
    m_meta = json.load(f)

assert "pipeline" in m_art, "Match missing pipeline"
assert len(m_art["features"]) == m_meta["total_features_with_elo"], "Match feature count mismatch"

m_sample = {
    "home_team": "Arsenal",
    "away_team": "Chelsea",
    "home_league_rank": 2,
    "away_league_rank": 9,
    "home_ppg": 2.25,
    "away_ppg": 1.45,
    "h2h_matches_prior": 6,
    "h2h_home_win_pct": 0.67
}
m_res = predict_match(m_sample)
probs = [m_res["home_win_probability"], m_res["draw_probability"], m_res["away_win_probability"]]
prob_sum = sum(probs)
assert abs(prob_sum - 1.0) < 1e-4, f"Probabilities do not sum to 1.0: {prob_sum}"
assert all(0.0 <= p <= 1.0 for p in probs), "Probabilities out of bounds [0, 1]"
print(f"Match verification: PASS! Outcome: {m_res['predicted_result']} ({m_res['predicted_code']}) | Probs: {probs} (sum: {prob_sum:.4f})")

print("\n--- 3. VERIFYING PRODUCTION SCOUTING MODEL ---")
s_idx = load_scouting_index(Path("models/scouting/scouting_model.joblib"))
with open("models/scouting/model_metadata.json", "r", encoding="utf-8") as f:
    s_meta = json.load(f)

assert len(s_idx.features) == s_meta["feature_count"], "Scouting feature count mismatch"
s_res = find_replacements("Declan Rice", top_n=5, same_position_only=True)
assert len(s_res) == 5, f"Expected 5 candidates, got {len(s_res)}"
cand_names = [c["candidate_player"].lower() for c in s_res]
assert "declan rice" not in cand_names, "Target player not excluded!"
for c in s_res:
    assert 0.0 <= c["similarity_score"] <= 100.0, "Score not in [0, 100]"
    assert c["position_compatibility"] == "Same Position", "Position filter failed"
print(f"Scouting verification: PASS! Top candidate: {s_res[0]['candidate_player']} ({s_res[0]['candidate_club']}) - Sim: {s_res[0]['similarity_score']:.1f}/100")

print("\n" + "=" * 70)
print("ALL THREE PROMOTED PRODUCTION MODELS VERIFIED SUCCESSFULLY!")
print("=" * 70)
