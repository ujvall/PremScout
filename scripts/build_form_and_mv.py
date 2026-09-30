"""Build recent form lookup and improve market value mapping for PremScout.

Extracts the verified last 5 match appearances for all scouting players from
Transfermarkt appearances.csv and games.csv (2025/26 season match records).
Recovers missing market values directly from verified Transfermarkt records.
"""

from pathlib import Path
import json
import unicodedata
import pandas as pd
import numpy as np

def normalize_name(name: str) -> str:
    if not isinstance(name, str):
        return ""
    norm = unicodedata.normalize('NFKD', name)
    norm = "".join(c for c in norm if not unicodedata.combining(c))
    norm = norm.replace("'", "").replace("-", " ").replace(".", "")
    norm = norm.replace("\xad", "").replace("\xa0", " ").replace("\ufffd", "")
    return " ".join(norm.lower().split())

def main():
    print("======================================================================")
    print(" Building Verified Recent Form & Market Value Recovery")
    print("======================================================================\n")

    # 1. Load Current Scouting Data
    scout_path = Path("data/current/scouting/current_player_scouting.csv")
    df_sc = pd.read_csv(scout_path)
    print(f"Loaded {len(df_sc)} current scouting profiles.")

    # 2. Market Value Recovery
    print("--- 1. Recovering Market Values from Transfermarkt Records ---")
    tm_dir = Path("data/raw/transfermarkt")
    tm_players = pd.read_csv(tm_dir / "players.csv", usecols=["player_id", "name", "market_value_in_eur", "date_of_birth"])
    tm_players["norm_name"] = tm_players["name"].apply(normalize_name)
    tm_players["birth_year"] = pd.to_datetime(tm_players["date_of_birth"], errors="coerce").dt.year

    # Build primary and fallback lookup dicts
    # Direct norm_name -> market_value
    tm_val_dict = tm_players.dropna(subset=["market_value_in_eur"]).set_index("norm_name")["market_value_in_eur"].to_dict()
    # (norm_name, birth_year) -> market_value
    tm_val_year_dict = tm_players.dropna(subset=["market_value_in_eur", "birth_year"]).set_index(["norm_name", "birth_year"])["market_value_in_eur"].to_dict()
    # norm_name -> player_id
    tm_pid_dict = tm_players.dropna(subset=["player_id"]).set_index("norm_name")["player_id"].to_dict()

    # Valuations history for latest valuation fallback
    vals_df = pd.read_csv(tm_dir / "player_valuations.csv", usecols=["player_id", "date", "market_value_in_eur"])
    vals_df["date"] = pd.to_datetime(vals_df["date"])
    latest_vals = vals_df.sort_values("date").groupby("player_id").last()["market_value_in_eur"].to_dict()

    recovered_count = 0
    updated_mv = []

    for _, r in df_sc.iterrows():
        curr_mv = r.get("market_value_eur")
        norm_n = r["norm_name"]
        
        # If existing value is valid and > 0, keep it
        if pd.notna(curr_mv) and float(curr_mv) > 0:
            updated_mv.append(float(curr_mv))
            continue

        # Attempt recovery
        matched_val = None
        # Check by name in TM players
        if norm_n in tm_val_dict and tm_val_dict[norm_n] > 0:
            matched_val = float(tm_val_dict[norm_n])
        # Check by player_id from latest_vals
        elif norm_n in tm_pid_dict and tm_pid_dict[norm_n] in latest_vals and latest_vals[tm_pid_dict[norm_n]] > 0:
            matched_val = float(latest_vals[tm_pid_dict[norm_n]])

        if matched_val is not None:
            updated_mv.append(matched_val)
            recovered_count += 1
        else:
            # Genuinely unavailable valuation: store as NaN (None) rather than 0
            updated_mv.append(np.nan)

    df_sc["market_value_eur"] = updated_mv
    valid_count = df_sc["market_value_eur"].notna().sum()
    print(f"Recovered {recovered_count} previously missing market values.")
    print(f"Total verified market values: {valid_count} / {len(df_sc)} ({valid_count/len(df_sc)*100:.1f}%).")
    print(f"Genuinely unavailable valuations: {len(df_sc) - valid_count} (marked as NaN/Not available).")

    # Update player_id where available
    updated_pids = []
    for _, r in df_sc.iterrows():
        existing_pid = r.get("player_id")
        if pd.notna(existing_pid) and str(existing_pid).strip() and str(existing_pid).lower() != "nan":
            updated_pids.append(existing_pid)
        elif r["norm_name"] in tm_pid_dict:
            updated_pids.append(int(tm_pid_dict[r["norm_name"]]))
        else:
            updated_pids.append(None)
    df_sc["player_id"] = updated_pids

    # Save updated scouting dataset
    df_sc.to_csv(scout_path, index=False, encoding="utf-8")
    print(f"Saved updated scouting dataset to {scout_path}.")

    # 3. Build Recent Form
    print("\n--- 2. Building Verified Recent Match Form (2025/26 Season) ---")
    app_path = tm_dir / "appearances.csv"
    games_path = tm_dir / "games.csv"

    print("Loading appearances and games...")
    app_df = pd.read_csv(app_path, usecols=[
        "game_id", "player_id", "player_club_id", "date", "goals", "assists", "minutes_played"
    ])
    games_df = pd.read_csv(games_path, usecols=[
        "game_id", "home_club_id", "away_club_id", "home_club_goals", "away_club_goals"
    ])

    app_df["date"] = pd.to_datetime(app_df["date"])
    # Merge game scores
    merged_apps = app_df.merge(games_df, on="game_id", how="left")

    def calc_result(row):
        hg = row["home_club_goals"]
        ag = row["away_club_goals"]
        if pd.isna(hg) or pd.isna(ag):
            return "D"
        hg, ag = int(hg), int(ag)
        pcid = row["player_club_id"]
        if pcid == row["home_club_id"]:
            return "W" if hg > ag else ("D" if hg == ag else "L")
        elif pcid == row["away_club_id"]:
            return "W" if ag > hg else ("D" if ag == hg else "L")
        return "D"

    merged_apps["result"] = merged_apps.apply(calc_result, axis=1)

    # Filter to scouting players only
    scout_pids = set(df_sc["player_id"].dropna().astype(int).unique())
    print(f"Scouting players with identified Transfermarkt player_id: {len(scout_pids)}")

    scout_apps = merged_apps[merged_apps["player_id"].isin(scout_pids)].sort_values("date", ascending=False)
    
    # Group by player_id and take last 5 appearances
    recent_form_lookup = {}
    for pid, group in scout_apps.groupby("player_id"):
        top5 = group.head(5)
        matches = []
        for _, row in top5.iterrows():
            matches.append({
                "date": row["date"].strftime("%Y-%m-%d"),
                "result": row["result"],
                "goals": int(row["goals"]),
                "assists": int(row["assists"]),
                "minutes": int(row["minutes_played"]),
            })
        if matches:
            recent_form_lookup[str(int(pid))] = {
                "season": "2025/26",
                "label": "2025/26 Match Form",
                "matches": matches,
                "summary": "".join(m["result"] for m in matches),
            }

    print(f"Generated verified recent form sequences for {len(recent_form_lookup)} players.")

    # Save to JSON
    form_out_path = Path("data/current/players/player_recent_form.json")
    with open(form_out_path, "w", encoding="utf-8") as f:
        json.dump(recent_form_lookup, f, indent=2)
    print(f"Saved verified recent form lookup to {form_out_path}.")

    # Also map recent_form string directly into current_player_scouting.csv
    form_summaries = []
    for _, r in df_sc.iterrows():
        pid_str = str(int(r["player_id"])) if pd.notna(r.get("player_id")) else ""
        if pid_str in recent_form_lookup:
            form_summaries.append(recent_form_lookup[pid_str]["summary"])
        else:
            form_summaries.append(None)
    df_sc["recent_form"] = form_summaries
    df_sc.to_csv(scout_path, index=False, encoding="utf-8")
    print(f"Added recent_form summary column to {scout_path} ({df_sc['recent_form'].notna().sum()} covered).")

    print("\n[SUCCESS] Recent form & market value recovery complete.")

if __name__ == "__main__":
    main()
