import sys, json
sys.path.insert(0, ".")
from pathlib import Path

with open("data/current/matches/current_team_states.json", "r", encoding="utf-8") as f:
    states = json.load(f)

# Compute 2026/27 league table
active_teams = [t for t, d in states.items() if d.get("matches_played_26", 0) > 0]
ranked_teams = sorted(active_teams, key=lambda t: (states[t]["points_26"], states[t]["rolling_gf_5"] - states[t]["rolling_ga_5"]), reverse=True)
ranks = {t: i + 1 for i, t in enumerate(ranked_teams)}

print(f"Active 2026/27 PL teams ({len(ranked_teams)}):")
for t in ranked_teams[:10]:
    print(f"  Rank {ranks[t]:2d}: {t:22} | Pts: {states[t]['points_26']:2d} | Elo: {states[t]['current_elo']:.1f} | PPG: {states[t]['ppg_26']:.2f}")

# Test fixture Arsenal vs Chelsea
ht = "Arsenal"
at = "Chelsea"
h_state = states[ht]
a_state = states[at]
h_elo = h_state["current_elo"]
a_elo = a_state["current_elo"]
elo_diff = round(h_elo - a_elo, 2)
elo_exp_h = round(1.0 / (1.0 + 10 ** (-(h_elo + 65.0 - a_elo) / 400.0)), 4)
h_rank = ranks.get(ht, 10)
a_rank = ranks.get(at, 10)
rank_diff = a_rank - h_rank

features = {
    "home_rest_days": 7.0,
    "away_rest_days": 7.0,
    "home_rolling_points_5": h_state["rolling_pts_5"] * 5.0,
    "away_rolling_points_5": a_state["rolling_pts_5"] * 5.0,
    "home_rolling_gf_5": h_state["rolling_gf_5"] * 5.0,
    "away_rolling_gf_5": a_state["rolling_gf_5"] * 5.0,
    "home_rolling_ga_5": h_state["rolling_ga_5"] * 5.0,
    "away_rolling_ga_5": a_state["rolling_ga_5"] * 5.0,
    "home_rolling_shots_5": h_state["rolling_shots_5"],
    "away_rolling_shots_5": a_state["rolling_shots_5"],
    "home_rolling_sot_5": h_state["rolling_sot_5"],
    "away_rolling_sot_5": a_state["rolling_sot_5"],
    "home_rolling_sot_conceded_5": h_state["rolling_sot_conceded_5"],
    "away_rolling_sot_conceded_5": a_state["rolling_sot_conceded_5"],
    "home_rolling_corners_5": h_state["rolling_corners_5"],
    "away_rolling_corners_5": a_state["rolling_corners_5"],
    "home_rolling_win_pct_5": h_state["rolling_win_pct_5"],
    "away_rolling_win_pct_5": a_state["rolling_win_pct_5"],
    "home_league_rank": h_rank,
    "away_league_rank": a_rank,
    "rank_diff": rank_diff,
    "home_ppg": h_state["ppg_26"],
    "away_ppg": a_state["ppg_26"],
    "h2h_matches_prior": 12,
    "h2h_home_win_pct": 0.50,
    "home_elo": h_elo,
    "away_elo": a_elo,
    "elo_diff": elo_diff,
    "elo_exp_h": elo_exp_h,
    "abs_elo_diff": abs(elo_diff),
    "abs_rank_diff": abs(rank_diff),
    "abs_ppg_diff": abs(round(h_state["ppg_26"] - a_state["ppg_26"], 2)),
}

from backend.ml.match.predict import predict_match
pred = predict_match(features)
print(f"\nPrediction for {ht} vs {at}:")
print(f"  Outcome: {pred['predicted_result']} (H:{pred['home_win_probability']:.1%}, D:{pred['draw_probability']:.1%}, A:{pred['away_win_probability']:.1%})")
