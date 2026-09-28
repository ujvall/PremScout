import sys
from pathlib import Path
sys.path.insert(0, str(Path(".").resolve()))
import joblib
import pandas as pd
import numpy as np

print("=" * 60)
print("TESTING API COMPATIBILITY WITH IMPROVED MODELS")
print("=" * 60)

# 1. SCOUTING API COMPATIBILITY
print("\n--- 1. Testing Scouting API with Improved Model ---")
from backend.ml.scouting.predict import find_replacements

try:
    scout_reps = find_replacements(
        player_name="Kevin De Bruyne",
        top_n=5,
        same_position_only=False,
        model_path=Path("models/improvement/scouting_model_improved.joblib")
    )
    print(f"Scouting returned {len(scout_reps)} candidates for Kevin De Bruyne:")
    for c in scout_reps:
        print(f"  -> {c['candidate_player']:20s} ({c['candidate_club']:18s}) | Pos: {c['candidate_position']:6s} | Score: {c['similarity_score']}%")
    assert len(scout_reps) == 5
    assert all(0.0 <= c["similarity_score"] <= 100.0 for c in scout_reps)
    print("Scouting API Compatibility: PASS!")
except Exception as e:
    print(f"Scouting API Compatibility FAILED: {e}")

# 2. TRANSFER API COMPATIBILITY
print("\n--- 2. Testing Transfer API Input with Improved Model ---")
impr_pipe = joblib.load("models/improvement/transfer_value_model_improved.joblib")

sample_transfer_input = {
    "age": 24.5,
    "height_cm": 185.0,
    "contract_years_left": 3.0,
    "club_market_value_eur": 450_000_000.0,
    "minutes_played": 2700,
    "appearances": 32,
    "goals": 14,
    "assists": 8,
    "yellow_cards": 3,
    "red_cards": 0,
    "goals_per90": 0.47,
    "assists_per90": 0.27,
    "shots_per90": 3.1,
    "npxg_per90": 0.42,
    "prev_market_value_eur": 40_000_000.0,
    "primary_position": "Attack",
    "sub_position": "Centre-Forward",
    "preferred_foot": "Right",
}

# The improved model includes minutes_per_appearance and cards_per90
# We verify if we can compute them automatically from standard transfer input:
df_t = pd.DataFrame([sample_transfer_input])
if "minutes_per_appearance" not in df_t.columns or pd.isna(df_t["minutes_per_appearance"].iloc[0]):
    df_t["minutes_per_appearance"] = df_t["minutes_played"] / (df_t["appearances"] + 1e-5)
if "cards_per90" not in df_t.columns or pd.isna(df_t["cards_per90"].iloc[0]):
    df_t["cards_per90"] = (df_t["yellow_cards"] + 2 * df_t["red_cards"]) / (df_t["minutes_played"] / 90.0 + 1e-5)

pred_log = impr_pipe.predict(df_t)
pred_eur = np.clip(np.expm1(pred_log)[0], 25_000, 350_000_000)
print(f"Transfer Input: Sample 24yo forward with EUR 40M prior value")
print(f"Predicted Market Value: EUR {pred_eur:,.0f} (EUR {pred_eur/1e6:.2f}M)")
assert pred_eur > 0
print("Transfer Model Input Compatibility: PASS!")

# 3. MATCH API COMPATIBILITY
print("\n--- 3. Testing Match API Input with Improved Model ---")
match_bundle = joblib.load("models/improvement/match_outcome_model_improved.joblib")
m_pipeline = match_bundle["pipeline"]
m_features = match_bundle["features"]
team_elos = match_bundle["team_elo_ratings"]
home_adv = match_bundle["home_advantage_elo"]

sample_match_input = {
    "home_rest_days": 7.0,
    "away_rest_days": 7.0,
    "home_rolling_points_5": 10.0,
    "away_rolling_points_5": 8.0,
    "home_rolling_gf_5": 8.0,
    "away_rolling_gf_5": 6.0,
    "home_rolling_ga_5": 4.0,
    "away_rolling_ga_5": 5.0,
    "home_rolling_shots_5": 14.0,
    "away_rolling_shots_5": 11.0,
    "home_rolling_sot_5": 5.5,
    "away_rolling_sot_5": 4.0,
    "home_rolling_sot_conceded_5": 3.5,
    "away_rolling_sot_conceded_5": 4.5,
    "home_rolling_corners_5": 6.0,
    "away_rolling_corners_5": 4.5,
    "home_rolling_win_pct_5": 0.6,
    "away_rolling_win_pct_5": 0.4,
    "home_league_rank": 3.0,
    "away_league_rank": 7.0,
    "rank_diff": -4.0,
    "home_ppg": 2.1,
    "away_ppg": 1.6,
    "h2h_matches_prior": 5.0,
    "h2h_home_win_pct": 0.6,
    # Teams
    "home_team": "Arsenal",
    "away_team": "Chelsea"
}

df_m = pd.DataFrame([sample_match_input])

# If Elo features not provided in raw request, populate from team_elo_ratings:
h_elo = team_elos.get(sample_match_input["home_team"], 1500.0)
a_elo = team_elos.get(sample_match_input["away_team"], 1500.0)
diff = (h_elo + home_adv) - a_elo
exp_h = 1.0 / (1.0 + 10.0 ** (-diff / 400.0))

df_m["home_elo"] = h_elo
df_m["away_elo"] = a_elo
df_m["elo_diff"] = diff
df_m["elo_exp_h"] = exp_h
df_m["abs_elo_diff"] = abs(diff)
df_m["abs_rank_diff"] = abs(sample_match_input["rank_diff"])
df_m["abs_ppg_diff"] = abs(sample_match_input["home_ppg"] - sample_match_input["away_ppg"])

probs = m_pipeline.predict_proba(df_m[m_features].values)[0]
preds = m_pipeline.predict(df_m[m_features].values)[0]

int_to_human = {0: "Home Win", 1: "Draw", 2: "Away Win"}
print(f"Match: {sample_match_input['home_team']} (Elo: {h_elo:.1f}) vs {sample_match_input['away_team']} (Elo: {a_elo:.1f})")
print(f"Probabilities: Home={probs[0]*100:.1f}%, Draw={probs[1]*100:.1f}%, Away={probs[2]*100:.1f}% (Sum={probs.sum():.4f})")
print(f"Predicted Outcome: {int_to_human[preds]}")
assert np.isclose(probs.sum(), 1.0, atol=1e-5)
print("Match Model Input Compatibility: PASS!")
