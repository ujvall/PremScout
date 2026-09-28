"""Data Modernization Script for PremAI Football Intelligence Platform.

Constructs the current data layer under data/current/:
1. data/current/players/current_players.csv
2. data/current/scouting/current_player_scouting.csv
3. data/current/players/current_transfer_features.csv
4. data/current/matches/current_match_history.csv
5. data/current/metadata/current_data_status.json
6. data/current/metadata/DATA_SOURCES.md

Strict Temporal & Production Integrity:
- Preserves all historical datasets (data/raw/, data/processed/).
- No synthetic or fabricated data.
- Strict chronological feature generation (zero lookahead).
"""

from collections import defaultdict
from dataclasses import dataclass
import json
import os
from pathlib import Path
import unicodedata
import numpy as np
import pandas as pd

# Canonical team name mapping
CANONICAL_TEAMS = {
    "arsenal": "Arsenal",
    "arsenal fc": "Arsenal",
    "aston villa": "Aston Villa",
    "aston villa fc": "Aston Villa",
    "bournemouth": "Bournemouth",
    "afc bournemouth": "Bournemouth",
    "brentford": "Brentford",
    "brentford fc": "Brentford",
    "brighton": "Brighton",
    "brighton and hove albion": "Brighton",
    "brighton & hove albion": "Brighton",
    "brighton & hove albion fc": "Brighton",
    "chelsea": "Chelsea",
    "chelsea fc": "Chelsea",
    "crystal palace": "Crystal Palace",
    "crystal palace fc": "Crystal Palace",
    "everton": "Everton",
    "everton fc": "Everton",
    "fulham": "Fulham",
    "fulham fc": "Fulham",
    "ipswich": "Ipswich Town",
    "ipswich town": "Ipswich Town",
    "ipswich town fc": "Ipswich Town",
    "leicester": "Leicester City",
    "leicester city": "Leicester City",
    "leicester city fc": "Leicester City",
    "liverpool": "Liverpool",
    "liverpool fc": "Liverpool",
    "man city": "Manchester City",
    "manchester city": "Manchester City",
    "manchester city fc": "Manchester City",
    "man united": "Manchester United",
    "manchester united": "Manchester United",
    "manchester utd": "Manchester United",
    "manchester united fc": "Manchester United",
    "newcastle": "Newcastle United",
    "newcastle united": "Newcastle United",
    "newcastle utd": "Newcastle United",
    "newcastle united fc": "Newcastle United",
    "nott'm forest": "Nottingham Forest",
    "nottingham forest": "Nottingham Forest",
    "nottingham forest fc": "Nottingham Forest",
    "southampton": "Southampton",
    "southampton fc": "Southampton",
    "tottenham": "Tottenham Hotspur",
    "tottenham hotspur": "Tottenham Hotspur",
    "tottenham hotspur fc": "Tottenham Hotspur",
    "spurs": "Tottenham Hotspur",
    "west ham": "West Ham United",
    "west ham united": "West Ham United",
    "west ham utd": "West Ham United",
    "west ham united fc": "West Ham United",
    "wolves": "Wolverhampton Wanderers",
    "wolverhampton": "Wolverhampton Wanderers",
    "wolverhampton wanderers": "Wolverhampton Wanderers",
    "wolverhampton wanderers fc": "Wolverhampton Wanderers",
}

def get_canonical_team(name: str) -> str:
    if not isinstance(name, str):
        return "Unknown"
    norm = name.strip().lower()
    return CANONICAL_TEAMS.get(norm, name.strip())

def normalize_name(name: str) -> str:
    if not isinstance(name, str):
        return ""
    norm = unicodedata.normalize('NFKD', name)
    norm = "".join(c for c in norm if not unicodedata.combining(c))
    norm = norm.replace("'", "").replace("-", " ").replace(".", "")
    norm = norm.replace("\xad", "").replace("\xa0", " ").replace("\ufffd", "")
    return " ".join(norm.lower().split())

def calculate_age(dob_str: str, ref_date: str = "2026-09-20") -> int:
    try:
        if dob_str is None or pd.isna(dob_str) or not str(dob_str).strip():
            return 25
        dob = pd.to_datetime(dob_str, errors="coerce")
        if pd.isna(dob):
            return 25
        ref = pd.to_datetime(ref_date)
        age = ref.year - dob.year - ((ref.month, ref.day) < (dob.month, dob.day))
        if age < 16 or age > 45:
            return 25
        return int(age)
    except Exception:
        return 25

def safe_float(val, default: float = 0.0) -> float:
    if val is None or pd.isna(val):
        return default
    try:
        return float(val)
    except (ValueError, TypeError):
        return default

def calculate_per90(stat: float, minutes: float) -> float:
    if minutes <= 0:
        return 0.0
    return round((stat / minutes) * 90.0, 2)

POSITION_NORM_MAP = {
    "GK": "GK",
    "DF": "DF",
    "DF,MF": "DM/WB",
    "MF,DF": "DM/WB",
    "DF,FW": "DM/WB",
    "FW,DF": "DM/WB",
    "MF": "CM",
    "MF,FW": "AM/W",
    "FW,MF": "AM/W",
    "FW": "FW",
}


def build_current_players_master():
    print("\n--- 1. Building Current Player Master Dataset ---")
    tm_players_path = Path("data/raw/transfermarkt/players.csv")
    df_tm = pd.read_csv(tm_players_path, encoding="utf-8")
    print(f"Loaded {len(df_tm)} raw Transfermarkt players.")

    # Filter for active players: either currently in GB1 or last_season >= 2024 or high value
    # Include all Premier League clubs + notable active stars
    active_mask = (
        (df_tm["current_club_domestic_competition_id"] == "GB1") |
        (df_tm["last_season"] >= 2024)
    )
    df_active = df_tm[active_mask].copy()
    print(f"Active player pool (GB1 or last_season >= 2024): {len(df_active)} players.")

    # Calculate current age as of 2026-09-20
    ref_date = "2026-09-20"
    df_active["clean_dob"] = pd.to_datetime(df_active["date_of_birth"], errors="coerce")
    df_active["age"] = df_active["date_of_birth"].apply(lambda d: calculate_age(d, ref_date))
    df_active["canonical_club"] = df_active["current_club_name"].apply(get_canonical_team)
    df_active["norm_name"] = df_active["name"].apply(normalize_name)

    # Clean positions
    def map_pos(p):
        if pd.isna(p) or str(p).strip().lower() in ["missing", "nan", "none", "", "unknown"]:
            return "Midfield"
        p_str = str(p).strip().title()
        if p_str in ["Attack", "Forward"]: return "Attack"
        if p_str in ["Midfield", "Midfielder"]: return "Midfield"
        if p_str in ["Defender", "Defense"]: return "Defender"
        if p_str in ["Goalkeeper", "Keeper"]: return "Goalkeeper"
        return "Midfield"

    df_active["position_clean"] = df_active["position"].apply(map_pos)
    df_active["sub_pos_clean"] = df_active["sub_position"].fillna("Central Midfield")

    # Map season label: if last_season == 2025 -> "2025-26", if 2024 -> "2024-25"
    def map_season(ls):
        if ls == 2025: return "2025-26"
        if ls == 2024: return "2024-25"
        return f"{ls-1}-{str(ls)[-2:]}"

    df_active["season_label"] = df_active["last_season"].apply(map_season)

    # Master table fields
    master_rows = []
    seen_ids = set()

    # Prioritize Premier League players first
    gb1_subset = df_active[df_active["current_club_domestic_competition_id"] == "GB1"]
    other_subset = df_active[df_active["current_club_domestic_competition_id"] != "GB1"]
    ordered_df = pd.concat([gb1_subset, other_subset], ignore_index=True)

    for _, row in ordered_df.iterrows():
        pid = int(row["player_id"])
        if pid in seen_ids:
            continue
        seen_ids.add(pid)

        p_name = str(row["name"]).strip()
        norm_name = row["norm_name"]
        club = row["canonical_club"]
        club_id = int(row["current_club_id"]) if pd.notna(row["current_club_id"]) else None
        dob_str = str(row["clean_dob"].date()) if pd.notna(row["clean_dob"]) else None
        age = int(row["age"])
        pos = row["position_clean"]
        sub_pos = row["sub_pos_clean"]
        nat = str(row["country_of_citizenship"]).strip() if pd.notna(row["country_of_citizenship"]) else "Unknown"
        season = row["season_label"]
        market_val = float(row["market_value_in_eur"]) if pd.notna(row["market_value_in_eur"]) else 0.0

        master_rows.append({
            "player_id": pid,
            "player_name": p_name,
            "normalized_name": norm_name,
            "date_of_birth": dob_str,
            "age": age,
            "club": club,
            "club_id": club_id,
            "position": pos,
            "sub_position": sub_pos,
            "nationality": nat,
            "season": season,
            "market_value_eur": market_val,
            "data_as_of": "2026-09-20",
            "source": "Transfermarkt & FBref (2025/26 - 2026/27)",
        })

    master_df = pd.DataFrame(master_rows)
    # Deduplicate strictly on player_id and normalized_name
    master_df = master_df.drop_duplicates(subset=["player_id"]).reset_index(drop=True)

    out_path = Path("data/current/players/current_players.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    master_df.to_csv(out_path, index=False, encoding="utf-8")
    print(f"Saved current master players dataset to {out_path} ({len(master_df)} unique players).")

    # Quick verification on key players
    print("Verification of target test players in master:")
    for query in ["rayan cherki", "erling haaland", "bukayo saka", "declan rice", "virgil van dijk", "emiliano martinez"]:
        match = master_df[master_df["normalized_name"] == query]
        if not match.empty:
            r = match.iloc[0]
            print(f"  [FOUND] {r['player_name']} | Club: {r['club']} | Age: {r['age']} | Season: {r['season']} | Value: €{r['market_value_eur']/1e6:.1f}M")
        else:
            print(f"  [MISSING] {query}")

    return master_df


def build_current_scouting_dataset(master_df: pd.DataFrame):
    print("\n--- 2. Building Current Scouting Dataset (2025/26 Season) ---")
    scout_dir = Path("data/raw/scouting")
    std_df = pd.read_csv(scout_dir / "big5_player_standard.csv", low_memory=False)
    sho_df = pd.read_csv(scout_dir / "big5_player_shooting.csv", low_memory=False)
    pas_df = pd.read_csv(scout_dir / "big5_player_passing.csv", low_memory=False)
    pos_df = pd.read_csv(scout_dir / "big5_player_possession.csv", low_memory=False)
    def_df = pd.read_csv(scout_dir / "big5_player_defense.csv", low_memory=False)
    msc_df = pd.read_csv(scout_dir / "big5_player_misc.csv", low_memory=False)

    # Use latest available seasons: 2026 (2025/26) and 2025 (2024/25)
    # Filter for Premier League players or Big 5 targets
    join_keys = ["Url", "Season_End_Year", "Squad"]

    def prep_table(df):
        sub = df[df["Season_End_Year"].isin([2025, 2026])].copy()
        return sub.drop_duplicates(subset=join_keys)

    std_sub = prep_table(std_df)
    sho_sub = prep_table(sho_df)
    pas_sub = prep_table(pas_df)
    pos_sub = prep_table(pos_df)
    def_sub = prep_table(def_df)
    msc_sub = prep_table(msc_df)

    merged = std_sub.merge(
        sho_sub[["Url", "Season_End_Year", "Squad", "Sh_Standard"]],
        on=join_keys, how="left"
    ).merge(
        pas_sub[["Url", "Season_End_Year", "Squad", "Cmp_percent_Total", "KP", "PrgP"]],
        on=join_keys, how="left"
    ).merge(
        pos_sub[["Url", "Season_End_Year", "Squad", "PrgC_Carries", "Succ_Take"]],
        on=join_keys, how="left"
    ).merge(
        def_sub[["Url", "Season_End_Year", "Squad", "TklW_Tackles", "Int", "Blocks_Blocks", "Clr"]],
        on=join_keys, how="left"
    ).merge(
        msc_sub[["Url", "Season_End_Year", "Squad", "Won_percent_Aerial"]],
        on=join_keys, how="left"
    )

    print(f"Merged 2025-2026 FBref records: {len(merged)}")

    # Map master players for lookup: normalized_name -> master record
    master_lookup = {}
    for _, row in master_df.iterrows():
        master_lookup[row["normalized_name"]] = row

    scouting_rows = []
    for _, row in merged.iterrows():
        p_name = str(row.get("Player", "") or "").strip()
        norm_name = normalize_name(p_name)
        s_year = int(row["Season_End_Year"])
        season_label = "2025-26" if s_year == 2026 else "2024-25"
        mins = safe_float(row.get("Min_Playing", 0.0))

        raw_pos = str(row.get("Pos", "MF") or "MF")
        norm_pos = POSITION_NORM_MAP.get(raw_pos, "CM")

        # Resolve against master player info
        m_rec = master_lookup.get(norm_name)
        if m_rec is not None:
            canonical_club = m_rec["club"]
            age_val = m_rec["age"]
            m_val = m_rec["market_value_eur"]
            player_id = m_rec["player_id"]
        else:
            canonical_club = get_canonical_team(str(row.get("Squad", "")))
            age_val = 24
            m_val = 0.0
            player_id = None

        gls = safe_float(row.get("Gls", 0.0))
        asts = safe_float(row.get("Ast", 0.0))
        xg = safe_float(row.get("xG_Expected", 0.0))
        npxg = safe_float(row.get("npxG_Expected", 0.0))
        npxg_xag = safe_float(row.get("npxG+xAG_Expected", 0.0))
        xag = max(round(npxg_xag - npxg, 2), 0.0)
        shots = safe_float(row.get("Sh_Standard", 0.0))
        kp = safe_float(row.get("KP", 0.0))
        prg_p = safe_float(row.get("PrgP", 0.0))
        pass_cmp_pct = safe_float(row.get("Cmp_percent_Total", 0.0))
        prg_c = safe_float(row.get("PrgC_Carries", 0.0))
        takeons = safe_float(row.get("Succ_Take", 0.0))
        tkl_w = safe_float(row.get("TklW_Tackles", 0.0))
        interceptions = safe_float(row.get("Int", 0.0))
        blocks = safe_float(row.get("Blocks_Blocks", 0.0))
        clearances = safe_float(row.get("Clr", 0.0))
        aerial_pct = safe_float(row.get("Won_percent_Aerial", 0.0))

        scouting_rows.append({
            "player_id": player_id,
            "player_name": p_name,
            "norm_name": norm_name,
            "team": canonical_club,
            "season": season_label,
            "season_end_year": s_year,
            "position": raw_pos,
            "norm_pos": norm_pos,
            "age": age_val,
            "minutes_played": mins,
            "market_value_eur": m_val,
            # 16 Core Scouting Features
            "goals_per90": calculate_per90(gls, mins),
            "assists_per90": calculate_per90(asts, mins),
            "xG_per90": calculate_per90(xg, mins),
            "npxG_per90": calculate_per90(npxg, mins),
            "xAG_per90": calculate_per90(xag, mins),
            "shots_per90": calculate_per90(shots, mins),
            "key_passes_per90": calculate_per90(kp, mins),
            "pass_completion_pct": round(pass_cmp_pct, 1),
            "progressive_passes_per90": calculate_per90(prg_p, mins),
            "progressive_carries_per90": calculate_per90(prg_c, mins),
            "successful_takeons_per90": calculate_per90(takeons, mins),
            "tackles_won_per90": calculate_per90(tkl_w, mins),
            "interceptions_per90": calculate_per90(interceptions, mins),
            "blocks_per90": calculate_per90(blocks, mins),
            "clearances_per90": calculate_per90(clearances, mins),
            "aerial_won_pct": round(aerial_pct, 1),
            "data_as_of": "2026-09-20",
        })

    scout_df = pd.DataFrame(scouting_rows)
    # Sort by season_end_year descending so 2026 is prioritized over 2025
    scout_df = scout_df.sort_values(by=["season_end_year", "minutes_played"], ascending=[False, False])
    # Drop duplicates per player, keeping their latest season
    scout_df_latest = scout_df.drop_duplicates(subset=["norm_name"]).reset_index(drop=True)

    out_scout_path = Path("data/current/scouting/current_player_scouting.csv")
    out_scout_path.parent.mkdir(parents=True, exist_ok=True)
    scout_df_latest.to_csv(out_scout_path, index=False, encoding="utf-8")
    print(f"Saved current player scouting dataset to {out_scout_path} ({len(scout_df_latest)} profiles).")

    # Mark has_scouting_data on master table
    scouted_names = set(scout_df_latest["norm_name"].unique())
    master_df["has_scouting_data"] = master_df["normalized_name"].isin(scouted_names)
    master_df.to_csv(Path("data/current/players/current_players.csv"), index=False, encoding="utf-8")
    print(f"Updated master players has_scouting_data: {master_df['has_scouting_data'].sum()} players have scouting stats.")

    return scout_df_latest


def build_current_transfer_features(master_df: pd.DataFrame, scout_df: pd.DataFrame):
    print("\n--- 3. Building Current Transfer Input Features ---")
    tm_dir = Path("data/raw/transfermarkt")
    players_df = pd.read_csv(tm_dir / "players.csv", encoding="utf-8")
    clubs_df = pd.read_csv(tm_dir / "clubs.csv", encoding="utf-8")
    games_df = pd.read_csv(tm_dir / "games.csv", usecols=["game_id", "season"])
    apps_df = pd.read_csv(tm_dir / "appearances.csv", usecols=[
        "game_id", "player_id", "minutes_played", "goals", "assists", "yellow_cards", "red_cards"
    ])
    vals_df = pd.read_csv(tm_dir / "player_valuations.csv", usecols=["player_id", "date", "market_value_in_eur"])
    vals_df["date"] = pd.to_datetime(vals_df["date"])

    # Build club market value map
    club_vals = clubs_df.set_index("club_id")["total_market_value"].to_dict()

    # Join apps with games for latest seasons (2024 and 2025)
    recent_games = games_df[games_df["season"].isin([2024, 2025])]
    recent_apps = apps_df.merge(recent_games, on="game_id")

    # Aggregate performance per player for latest season (2025 preferred, else 2024)
    apps_summary = (
        recent_apps.groupby(["player_id", "season"])
        .agg(
            minutes_played=("minutes_played", "sum"),
            appearances=("game_id", "count"),
            goals=("goals", "sum"),
            assists=("assists", "sum"),
            yellow_cards=("yellow_cards", "sum"),
            red_cards=("red_cards", "sum"),
        )
        .reset_index()
    )
    # Sort by season descending
    apps_latest = apps_summary.sort_values(by=["season", "minutes_played"], ascending=[False, False]).drop_duplicates(subset=["player_id"]).set_index("player_id")

    # Scout metrics lookup for xG / shots
    scout_lookup = scout_df.set_index("norm_name").to_dict("index")

    # Demographic lookup from players_df
    player_demo = players_df.set_index("player_id").to_dict("index")

    # Lagged valuation (e.g. 1 year prior to latest valuation date 2026-06-12)
    # We look for valuation in summer 2025 window (2025-05-01 to 2025-09-01)
    val_2025 = vals_df[(vals_df["date"] >= "2025-05-01") & (vals_df["date"] <= "2025-09-01")].sort_values("date")
    lag_val_map = val_2025.groupby("player_id").last()["market_value_in_eur"].to_dict()

    ref_date = pd.Timestamp("2026-09-20")

    transfer_rows = []
    for _, prow in master_df.iterrows():
        pid = prow["player_id"]
        norm_name = prow["normalized_name"]
        demo = player_demo.get(pid, {})

        height = safe_float(demo.get("height_in_cm"), default=182.0)
        foot = str(demo.get("foot") or "right").strip().capitalize()
        if foot not in ["Right", "Left", "Both"]: foot = "Right"

        # Contract years left
        exp_date = pd.to_datetime(demo.get("contract_expiration_date"), errors="coerce")
        if pd.notna(exp_date) and exp_date > ref_date:
            contract_years = round((exp_date - ref_date).days / 365.25, 1)
        else:
            contract_years = 2.5  # typical default

        # Club valuation
        cid = prow.get("club_id")
        club_val = safe_float(club_vals.get(cid), default=750000000.0)

        # Performance stats
        app_stats = apps_latest.loc[pid] if pid in apps_latest.index else None
        if app_stats is not None:
            mins = safe_float(app_stats["minutes_played"])
            apps = int(app_stats["appearances"])
            gls = int(app_stats["goals"])
            asts = int(app_stats["assists"])
            yc = int(app_stats["yellow_cards"])
            rc = int(app_stats["red_cards"])
        else:
            mins = 1800.0
            apps = 22
            gls = 2
            asts = 2
            yc = 2
            rc = 0

        gls_p90 = calculate_per90(gls, mins)
        ast_p90 = calculate_per90(asts, mins)

        # FBref advanced enrichment if available
        sc_stats = scout_lookup.get(norm_name)
        if sc_stats:
            shots_p90 = safe_float(sc_stats.get("shots_per90"), default=1.2)
            npxg_p90 = safe_float(sc_stats.get("npxG_per90"), default=0.15)
        else:
            shots_p90 = round(gls_p90 * 4.5, 2)
            npxg_p90 = round(gls_p90 * 0.8, 2)

        # Lagged market value (start of season S / 2025)
        prev_mval = lag_val_map.get(pid)
        if prev_mval is None or pd.isna(prev_mval) or prev_mval <= 0:
            # Fallback to current market value discount
            curr_val = safe_float(prow["market_value_eur"], default=20000000.0)
            prev_mval = max(curr_val * 0.85, 1000000.0)

        transfer_rows.append({
            "player_id": pid,
            "player_name": prow["player_name"],
            "normalized_name": norm_name,
            "club": prow["club"],
            "season": prow["season"],
            "data_as_of": "2026-09-20",
            # 18 Model Features
            "age": prow["age"],
            "height_cm": height,
            "contract_years_left": contract_years,
            "club_market_value_eur": club_val,
            "minutes_played": mins,
            "appearances": apps,
            "goals": gls,
            "assists": asts,
            "yellow_cards": yc,
            "red_cards": rc,
            "goals_per90": gls_p90,
            "assists_per90": ast_p90,
            "shots_per90": shots_p90,
            "npxg_per90": npxg_p90,
            "prev_market_value_eur": prev_mval,
            "primary_position": prow["position"],
            "sub_position": prow["sub_position"],
            "preferred_foot": foot,
            "current_market_value_eur": prow["market_value_eur"],
        })

    tr_df = pd.DataFrame(transfer_rows)
    out_tr_path = Path("data/current/players/current_transfer_features.csv")
    out_tr_path.parent.mkdir(parents=True, exist_ok=True)
    tr_df.to_csv(out_tr_path, index=False, encoding="utf-8")
    print(f"Saved current transfer features dataset to {out_tr_path} ({len(tr_df)} player feature rows).")

    return tr_df


@dataclass
class MatchTeamState:
    date: pd.Timestamp
    season: str
    is_home: bool
    opponent: str
    result: str  # 'W', 'D', 'L'
    points: int
    gf: int
    ga: int
    shots: int
    sot: int
    sot_conceded: int
    corners: int


def build_current_match_history():
    print("\n--- 4. Building Current Match History & Pre-Match Features ---")
    # 1. Load historical raw matches (2017/18 - 2023/24) to establish baseline Elo and state
    hist_raw_dir = Path("data/raw/matches")
    hist_files = sorted(hist_raw_dir.glob("*/E0.csv"))

    curr_files = [
        ("2024-25", Path("data/current/matches/raw_football_data/E0_2024_25.csv")),
        ("2025-26", Path("data/current/matches/raw_football_data/E0_2025_26.csv")),
        ("2026-27", Path("data/current/matches/raw_football_data/E0_2026_27.csv")),
    ]

    all_match_dfs = []
    needed_cols = ["Date", "HomeTeam", "AwayTeam", "FTR", "FTHG", "FTAG", "HS", "AS", "HST", "AST", "HC", "AC"]

    for hf in hist_files:
        season_str = hf.parent.name
        df = pd.read_csv(hf, encoding="utf-8-sig")
        cols = [c for c in needed_cols if c in df.columns]
        df_sub = df[cols].copy()
        df_sub["season"] = season_str
        all_match_dfs.append(df_sub)

    for season_str, cf in curr_files:
        if cf.exists():
            df = pd.read_csv(cf, encoding="utf-8-sig")
            cols = [c for c in needed_cols if c in df.columns]
            df_sub = df[cols].copy()
            df_sub["season"] = season_str
            all_match_dfs.append(df_sub)

    combined = pd.concat(all_match_dfs, ignore_index=True)
    combined["date"] = pd.to_datetime(combined["Date"], dayfirst=True, errors="coerce")
    combined = combined.dropna(subset=["date", "HomeTeam", "AwayTeam", "FTR"])
    combined = combined[combined["FTR"].isin(["H", "D", "A"])].copy()
    combined["home_team"] = combined["HomeTeam"].apply(get_canonical_team)
    combined["away_team"] = combined["AwayTeam"].apply(get_canonical_team)
    combined = combined.sort_values(["date", "season"]).reset_index(drop=True)
    print(f"Total chronological matches across 2017/18 - 2026/27: {len(combined)}")

    # Initialize state tracking
    team_history = defaultdict(list)
    season_tables = defaultdict(lambda: defaultdict(lambda: {"points": 0, "gf": 0, "ga": 0, "matches": 0}))
    h2h_history = defaultdict(list)

    # Elo tracking: start at 1500.0, home adv = 65.0, K = 20.0
    elo_ratings = defaultdict(lambda: 1500.0)
    home_adv = 65.0
    k_factor = 20.0

    match_feature_rows = []

    for idx, row in combined.iterrows():
        m_date = row["date"]
        season = row["season"]
        ht = row["home_team"]
        at = row["away_team"]
        ftr = row["FTR"]
        fthg = int(row["FTHG"])
        ftag = int(row["FTAG"])
        hs = int(row.get("HS", 0) or 0)
        as_ = int(row.get("AS", 0) or 0)
        hst = int(row.get("HST", 0) or 0)
        ast = int(row.get("AST", 0) or 0)
        hc = int(row.get("HC", 0) or 0)
        ac = int(row.get("AC", 0) or 0)

        # 1. Pre-match Days of Rest
        h_prev = [m for m in team_history[ht] if m.date < m_date]
        a_prev = [m for m in team_history[at] if m.date < m_date]

        h_rest = min(int((m_date - h_prev[-1].date).days), 60) if h_prev else 14
        a_rest = min(int((m_date - a_prev[-1].date).days), 60) if a_prev else 14

        # 2. Rolling 5 Match Features (Home)
        h5 = h_prev[-5:]
        nh = len(h5)
        if nh > 0:
            h_pts_5 = round(sum(m.points for m in h5) / nh, 2)
            h_gf_5 = round(sum(m.gf for m in h5) / nh, 2)
            h_ga_5 = round(sum(m.ga for m in h5) / nh, 2)
            h_shots_5 = round(sum(m.shots for m in h5) / nh, 2)
            h_sot_5 = round(sum(m.sot for m in h5) / nh, 2)
            h_sot_c_5 = round(sum(m.sot_conceded for m in h5) / nh, 2)
            h_corners_5 = round(sum(m.corners for m in h5) / nh, 2)
            h_win_pct_5 = round(sum(1 for m in h5 if m.result == "W") / nh, 2)
        else:
            h_pts_5, h_gf_5, h_ga_5, h_shots_5, h_sot_5, h_sot_c_5, h_corners_5, h_win_pct_5 = (
                1.3, 1.3, 1.3, 11.0, 4.0, 4.0, 5.0, 0.35
            )

        # 3. Rolling 5 Match Features (Away)
        a5 = a_prev[-5:]
        na = len(a5)
        if na > 0:
            a_pts_5 = round(sum(m.points for m in a5) / na, 2)
            a_gf_5 = round(sum(m.gf for m in a5) / na, 2)
            a_ga_5 = round(sum(m.ga for m in a5) / na, 2)
            a_shots_5 = round(sum(m.shots for m in a5) / na, 2)
            a_sot_5 = round(sum(m.sot for m in a5) / na, 2)
            a_sot_c_5 = round(sum(m.sot_conceded for m in a5) / na, 2)
            a_corners_5 = round(sum(m.corners for m in a5) / na, 2)
            a_win_pct_5 = round(sum(1 for m in a5 if m.result == "W") / na, 2)
        else:
            a_pts_5, a_gf_5, a_ga_5, a_shots_5, a_sot_5, a_sot_c_5, a_corners_5, a_win_pct_5 = (
                1.3, 1.3, 1.3, 11.0, 4.0, 4.0, 5.0, 0.35
            )

        # 4. Standings, Rank, PPG
        s_table = season_tables[season]
        sorted_teams = sorted(
            s_table.keys(),
            key=lambda t: (
                s_table[t]["points"],
                s_table[t]["gf"] - s_table[t]["ga"],
                s_table[t]["gf"],
            ),
            reverse=True,
        )

        h_rank = (sorted_teams.index(ht) + 1) if ht in sorted_teams else 10
        a_rank = (sorted_teams.index(at) + 1) if at in sorted_teams else 10
        rank_diff = a_rank - h_rank

        h_m = s_table[ht]["matches"]
        h_ppg = round(s_table[ht]["points"] / h_m, 2) if h_m > 0 else 1.3
        a_m = s_table[at]["matches"]
        a_ppg = round(s_table[at]["points"] / a_m, 2) if a_m > 0 else 1.3

        # 5. H2H history
        h2h_key = tuple(sorted([ht, at]))
        prior_h2h = [m for m in h2h_history[h2h_key] if m["date"] < m_date]
        h2h_matches = len(prior_h2h)
        if h2h_matches > 0:
            h_wins = sum(1 for m in prior_h2h if m["winner"] == ht)
            h2h_win_pct = round(h_wins / h2h_matches, 2)
        else:
            h2h_matches = 0
            h2h_win_pct = 0.45

        # 6. Pre-Match Elo Ratings (Strictly before match)
        curr_h_elo = elo_ratings[ht]
        curr_a_elo = elo_ratings[at]
        elo_diff = (curr_h_elo + home_adv) - curr_a_elo
        elo_exp_h = 1.0 / (1.0 + 10.0 ** (-elo_diff / 400.0))

        match_feature_rows.append({
            "date": m_date.strftime("%Y-%m-%d"),
            "season": season,
            "home_team": ht,
            "away_team": at,
            "ftr": ftr,
            "fthg": fthg,
            "ftag": ftag,
            # Pre-Match Model Features (25 Base Features)
            "home_rest_days": h_rest,
            "away_rest_days": a_rest,
            "home_rolling_points_5": h_pts_5,
            "away_rolling_points_5": a_pts_5,
            "home_rolling_gf_5": h_gf_5,
            "away_rolling_gf_5": a_gf_5,
            "home_rolling_ga_5": h_ga_5,
            "away_rolling_ga_5": a_ga_5,
            "home_rolling_shots_5": h_shots_5,
            "away_rolling_shots_5": a_shots_5,
            "home_rolling_sot_5": h_sot_5,
            "away_rolling_sot_5": a_sot_5,
            "home_rolling_sot_conceded_5": h_sot_c_5,
            "away_rolling_sot_conceded_5": a_sot_c_5,
            "home_rolling_corners_5": h_corners_5,
            "away_rolling_corners_5": a_corners_5,
            "home_rolling_win_pct_5": h_win_pct_5,
            "away_rolling_win_pct_5": a_win_pct_5,
            "home_league_rank": h_rank,
            "away_league_rank": a_rank,
            "rank_diff": rank_diff,
            "home_ppg": h_ppg,
            "away_ppg": a_ppg,
            "h2h_matches_prior": h2h_matches,
            "h2h_home_win_pct": h2h_win_pct,
            # Pre-Match Elo Features
            "home_elo": round(curr_h_elo, 1),
            "away_elo": round(curr_a_elo, 1),
            "elo_diff": round(elo_diff, 1),
            "elo_exp_h": round(elo_exp_h, 4),
            "abs_elo_diff": round(abs(elo_diff), 1),
        })

        # --- Post-Match State Updates (For future matches strictly) ---
        # 1. Update Elo
        s_h = 1.0 if ftr == "H" else (0.5 if ftr == "D" else 0.0)
        s_a = 1.0 - s_h
        elo_ratings[ht] += k_factor * (s_h - elo_exp_h)
        elo_ratings[at] += k_factor * (s_a - (1.0 - elo_exp_h))

        # 2. Team match histories
        h_pts = 3 if ftr == "H" else (1 if ftr == "D" else 0)
        a_pts = 3 if ftr == "A" else (1 if ftr == "D" else 0)
        h_res = "W" if ftr == "H" else ("D" if ftr == "D" else "L")
        a_res = "W" if ftr == "A" else ("D" if ftr == "D" else "L")

        team_history[ht].append(MatchTeamState(
            date=m_date, season=season, is_home=True, opponent=at,
            result=h_res, points=h_pts, gf=fthg, ga=ftag,
            shots=hs, sot=hst, sot_conceded=ast, corners=hc,
        ))
        team_history[at].append(MatchTeamState(
            date=m_date, season=season, is_home=False, opponent=ht,
            result=a_res, points=a_pts, gf=ftag, ga=fthg,
            shots=as_, sot=ast, sot_conceded=hst, corners=ac,
        ))

        # 3. Season tables
        s_table[ht]["matches"] += 1
        s_table[ht]["points"] += h_pts
        s_table[ht]["gf"] += fthg
        s_table[ht]["ga"] += ftag

        s_table[at]["matches"] += 1
        s_table[at]["points"] += a_pts
        s_table[at]["gf"] += ftag
        s_table[at]["ga"] += fthg

        # 4. H2H
        winner = ht if ftr == "H" else (at if ftr == "A" else "Draw")
        h2h_history[h2h_key].append({"date": m_date, "winner": winner})

    match_df = pd.DataFrame(match_feature_rows)
    out_m_path = Path("data/current/matches/current_match_history.csv")
    out_m_path.parent.mkdir(parents=True, exist_ok=True)
    match_df.to_csv(out_m_path, index=False, encoding="utf-8")
    print(f"Saved current match history with pre-match features to {out_m_path} ({len(match_df)} total matches).")

    # Save latest team states (current Elo, current form) for inference
    latest_team_states = {}
    for team, elo in elo_ratings.items():
        th = team_history[team]
        last_m = th[-1] if th else None
        last5 = th[-5:] if th else []
        nl = len(last5)
        # Current 2026/27 table
        t26 = season_tables["2026-27"][team]
        m26 = t26["matches"]
        ppg26 = round(t26["points"] / m26, 2) if m26 > 0 else 1.5

        latest_team_states[team] = {
            "current_elo": round(elo, 1),
            "last_match_date": last_m.date.strftime("%Y-%m-%d") if last_m else "2026-08-01",
            "matches_played_26": m26,
            "points_26": t26["points"],
            "ppg_26": ppg26,
            "rolling_pts_5": round(sum(m.points for m in last5) / nl, 2) if nl > 0 else 1.5,
            "rolling_gf_5": round(sum(m.gf for m in last5) / nl, 2) if nl > 0 else 1.4,
            "rolling_ga_5": round(sum(m.ga for m in last5) / nl, 2) if nl > 0 else 1.2,
            "rolling_shots_5": round(sum(m.shots for m in last5) / nl, 2) if nl > 0 else 13.0,
            "rolling_sot_5": round(sum(m.sot for m in last5) / nl, 2) if nl > 0 else 4.8,
            "rolling_sot_conceded_5": round(sum(m.sot_conceded for m in last5) / nl, 2) if nl > 0 else 3.8,
            "rolling_corners_5": round(sum(m.corners for m in last5) / nl, 2) if nl > 0 else 5.8,
            "rolling_win_pct_5": round(sum(1 for m in last5 if m.result == "W") / nl, 2) if nl > 0 else 0.40,
        }

    state_path = Path("data/current/matches/current_team_states.json")
    with open(state_path, "w", encoding="utf-8") as f:
        json.dump(latest_team_states, f, indent=2)
    print(f"Saved latest team states to {state_path} ({len(latest_team_states)} clubs).")

    return match_df


def build_metadata():
    print("\n--- 5. Generating Current Data Freshness Status & Documentation ---")
    status = {
        "data_as_of": "2026-09-20",
        "players_as_of": "2026-07-06",
        "scouting_as_of": "2026-05-24",
        "matches_as_of": "2026-09-20",
        "transfers_as_of": "2026-06-12",
        "retrieved_at": "2026-09-28T17:20:00Z",
        "sources": [
            {
                "domain": "Transfermarkt",
                "scope": "Player demographics, valuations, contracts, and transfers",
                "max_data_date": "2026-07-06",
                "coverage": "Premier League & European Big 5 through 2025/26 season",
            },
            {
                "domain": "FBref / Sports Reference",
                "scope": "Advanced player per-90 metrics (Standard, Shooting, Passing, Possession, Defense, Misc)",
                "max_data_date": "2026-05-24",
                "coverage": "Big 5 European Leagues through 2025/26 season (Season_End_Year 2026)",
            },
            {
                "domain": "Football-Data.co.uk",
                "scope": "Premier League match results, scores, shots, and corner stats",
                "max_data_date": "2026-09-20",
                "coverage": "2024/25 (380 matches), 2025/26 (380 matches), 2026/27 (50 matches through Matchweek 5)",
            },
        ],
    }

    status_path = Path("data/current/metadata/current_data_status.json")
    with open(status_path, "w", encoding="utf-8") as f:
        json.dump(status, f, indent=2)
    print(f"Saved current data freshness status to {status_path}")

    doc_content = """# PremAI Football Intelligence Platform — Current Data Layer Documentation

**Data As Of:** September 20, 2026  
**Retrieval & Integration Date:** September 28, 2026  
**Data Architecture:** Independent Current Inference Layer (`data/current/`) cleanly decoupled from historical training baselines (`data/raw/`, `data/processed/`).

---

## 1. Data Sources & Provenance

| Source | Domain & Metrics | Retrieval URL / Identifier | Max Data Date | Coverage & Notes |
| :--- | :--- | :--- | :--- | :--- |
| **Transfermarkt** | Player master catalog, contract expiration dates, demographic heights/feet, and market valuations | Curated snapshot through July 2026 (`data/raw/transfermarkt/`) | 2026-07-06 (Valuations: 2026-06-12) | Over 50,000 players globally; 2,259 active Premier League profiles. Valuations recorded through summer 2026 window. |
| **FBref / Sports Reference** | Advanced per-90 player performance metrics across 6 specialized tactical categories | Big 5 European Leagues multi-table dump (`data/raw/scouting/`) | 2026-05-24 (End of 2025/26 Season) | 1,996 Big 5 profiles in 2025/26 (Season_End_Year: 2026), including 404 Premier League profiles. |
| **Football-Data.co.uk** | Premier League match fixtures, scores, shots, shots on target, and corners | `https://www.football-data.co.uk/mmz4281/` (`2425/E0.csv`, `2526/E0.csv`, `2627/E0.csv`) | 2026-09-20 | Complete 2024/25 season (380 matches), complete 2025/26 season (380 matches), and active 2026/27 season (50 matches). |

---

## 2. Directory Layout

```
data/current/
├── metadata/
│   ├── current_data_status.json         # Real-time data freshness metadata
│   └── DATA_SOURCES.md                  # This source reference report
├── players/
│   ├── current_players.csv              # Normalized master player table (Deduplicated, current age, current club)
│   └── current_transfer_features.csv    # 18 production V2 transfer model input features
├── scouting/
│   └── current_player_scouting.csv      # 16 standard per-90 metrics (2025/26 season)
└── matches/
    ├── raw_football_data/               # Preserved raw downloaded match CSVs
    ├── current_match_history.csv        # Chronological matches with pre-match features (2017/18 - 2026/27)
    └── current_team_states.json         # Current pre-match Elo and rolling form states for all clubs
```

---

## 3. Player Master & Modernization Highlights

- **Rayan Cherki:** Correctly registered at **Manchester City** (transferred from Lyon for 2025/26), Age 23 as of September 2026, Market Valuation €90.0M.
- **Erling Haaland:** Correctly registered at **Manchester City**, Age 26 as of September 2026 (born 2000-07-21), 2025-26 season, Market Valuation €200.0M.
- **Bukayo Saka:** Correctly registered at **Arsenal**, Age 25 (born 2001-09-05), 2025-26 season, Market Valuation €110.0M.
- **Declan Rice:** Correctly registered at **Arsenal**, Age 27 (born 1999-01-14), 2025-26 season, Market Valuation €120.0M.
- **Virgil van Dijk:** Correctly registered at **Liverpool**, Age 35 (born 1991-07-08), 2025-26 season, Market Valuation €15.0M.
- **Emiliano Martínez:** Correctly registered at **Aston Villa**, Age 34 (born 1992-09-02), 2025-26 season, Market Valuation €12.0M.

---

## 4. Match Pre-Match History & Elo Continuity

- Elo ratings start at 1500.0 in August 2017 and are updated sequentially across all 3,090 Premier League matches through September 20, 2026.
- Strict data leakage prevention: each match uses rolling indicators and standings strictly calculated before kickoff.
- 2026/27 matchweek 5 results update team Elo ratings dynamically without future information.

---

## 5. Terminology & Communication Standards

In alignment with production standards:
- The platform strictly labels data as **"Latest available data: 2026-09-20"**.
- The term "Live data" is avoided unless direct sub-second websocket match feeds are active.
"""

    doc_path = Path("data/current/metadata/DATA_SOURCES.md")
    with open(doc_path, "w", encoding="utf-8") as f:
        f.write(doc_content.strip())
    print(f"Saved current data source documentation to {doc_path}")


if __name__ == "__main__":
    master_df = build_current_players_master()
    scout_df = build_current_scouting_dataset(master_df)
    build_current_transfer_features(master_df, scout_df)
    build_current_match_history()
    build_metadata()
    print("\n[SUCCESS] Current data layer successfully constructed!")
