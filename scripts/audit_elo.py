import sys
from pathlib import Path
sys.path.insert(0, str(Path(".").resolve()))
import pandas as pd
import numpy as np
from scripts.train_improved_match import compute_sequential_elo, HOME_ADVANTAGE, K_FACTOR

df = pd.read_csv("data/processed/matches/match_outcome_dataset.csv")
df["date"] = pd.to_datetime(df["date"])

df_elo, final_elos = compute_sequential_elo(df)

print("=== 10 SEQUENTIAL MATCHES ELO AUDIT TRACE ===")
trace_matches = df_elo.head(10)
for idx, row in trace_matches.iterrows():
    h_team = row["home_team"]
    a_team = row["away_team"]
    h_elo_before = row["home_elo"]
    a_elo_before = row["away_elo"]
    diff = row["elo_diff"]
    exp_h = row["elo_exp_h"]
    res = row["match_result"]
    s_h = 1.0 if res == "H" else (0.5 if res == "D" else 0.0)
    s_a = 1.0 - s_h
    h_elo_after = h_elo_before + K_FACTOR * (s_h - exp_h)
    a_elo_after = a_elo_before + K_FACTOR * (s_a - (1.0 - exp_h))
    date_str = str(row["date"])[:10]
    print(f"Match {idx+1:2d} | {date_str} | {h_team:20s} ({h_elo_before:.1f}) vs {a_team:20s} ({a_elo_before:.1f}) | Diff: {diff:+.1f} | Exp_H: {exp_h:.3f} | Res: {res} -> Post: {h_team[:10]}={h_elo_after:.1f}, {a_team[:10]}={a_elo_after:.1f}")

arsenal_matches = df_elo[(df_elo["home_team"] == "Arsenal") | (df_elo["away_team"] == "Arsenal")].head(4)
print("\n=== ARSENAL SEQUENTIAL CONTINUITY CHECK ===")
for idx, row in arsenal_matches.iterrows():
    is_home = (row["home_team"] == "Arsenal")
    elo_pre = row["home_elo"] if is_home else row["away_elo"]
    opp = row["away_team"] if is_home else row["home_team"]
    role = "HOME" if is_home else "AWAY"
    date_str = str(row["date"])[:10]
    res = row["match_result"]
    print(f"Date: {date_str} | Role: {role:4s} vs {opp:18s} | Arsenal Pre-Match Elo: {elo_pre:.2f} | Result: {res}")
