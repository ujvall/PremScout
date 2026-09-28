import json
import sys
from pathlib import Path
from backend.ml.match.predict import predict_match

state_path = Path("data/current/matches/current_team_states.json")
with open(state_path, "r", encoding="utf-8") as f:
    team_states = json.load(f)

fixtures = [
    ("Arsenal", "Chelsea"),
    ("Manchester City", "Liverpool"),
    ("Aston Villa", "Tottenham Hotspur")
]

for ht, at in fixtures:
    h_state = team_states.get(ht, {})
    a_state = team_states.get(at, {})
    
    h_elo = h_state.get("current_elo", 1700.0)
    a_elo = a_state.get("current_elo", 1600.0)
    
    # Standings mock rank based on points_26
    feature_dict = {
        "home_team": ht,
        "away_team": at,
        "home_rest_days": 7,
        "away_rest_days": 6,
        "home_rolling_points_5": h_state.get("rolling_pts_5", 2.0),
        "away_rolling_points_5": a_state.get("rolling_pts_5", 1.4),
        "home_rolling_gf_5": h_state.get("rolling_gf_5", 1.8),
        "away_rolling_gf_5": a_state.get("rolling_gf_5", 1.2),
        "home_rolling_ga_5": h_state.get("rolling_ga_5", 0.8),
        "away_rolling_ga_5": a_state.get("rolling_ga_5", 1.4),
        "home_rolling_shots_5": h_state.get("rolling_shots_5", 15.0),
        "away_rolling_shots_5": a_state.get("rolling_shots_5", 12.0),
        "home_rolling_sot_5": h_state.get("rolling_sot_5", 5.5),
        "away_rolling_sot_5": a_state.get("rolling_sot_5", 4.0),
        "home_rolling_sot_conceded_5": h_state.get("rolling_sot_conceded_5", 3.0),
        "away_rolling_sot_conceded_5": a_state.get("rolling_sot_conceded_5", 4.5),
        "home_rolling_corners_5": h_state.get("rolling_corners_5", 6.5),
        "away_rolling_corners_5": a_state.get("rolling_corners_5", 4.5),
        "home_rolling_win_pct_5": h_state.get("rolling_win_pct_5", 0.60),
        "away_rolling_win_pct_5": a_state.get("rolling_win_pct_5", 0.40),
        "home_league_rank": 2,
        "away_league_rank": 8,
        "rank_diff": 6,
        "home_ppg": h_state.get("ppg_26", 2.0),
        "away_ppg": a_state.get("ppg_26", 1.4),
        "h2h_matches_prior": 14,
        "h2h_home_win_pct": 0.55,
        "home_elo": h_elo,
        "away_elo": a_elo,
    }
    res = predict_match(feature_dict)
    print(f"{ht} (Elo {h_elo}) vs {at} (Elo {a_elo}) -> Pred: {res['predicted_result']} (H: {res['home_win_probability']*100:.1f}%, D: {res['draw_probability']*100:.1f}%, A: {res['away_win_probability']*100:.1f}%)")
