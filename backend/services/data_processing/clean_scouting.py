"""Player scouting dataset preprocessor and feature generator.

Combines FBref advanced player metrics (Standard, Shooting, Passing, Possession, Defense, Misc)
with Transfermarkt market valuations using validated entity resolution.
"""

from pathlib import Path
import numpy as np
import pandas as pd

from .entity_resolution import (
    extract_fbref_id,
    get_canonical_team_name,
    match_player_candidate,
    normalize_player_name,
)
from .feature_engineering import calculate_per90

DEFAULT_SCOUT_DIR = Path("data/raw/scouting")
DEFAULT_TM_DIR = Path("data/raw/transfermarkt")
DEFAULT_OUTPUT_PATH = Path("data/processed/scouting/player_scouting_dataset.csv")
UNFILTERED_OUTPUT_PATH = Path("data/processed/scouting/player_scouting_unfiltered.csv")

TARGET_SEASON_END_YEARS = [2018, 2019, 2020, 2021, 2022, 2023, 2024]


def safe_float(val, default: float = 0.0) -> float:
    """Safely convert string or numeric to float."""
    if val is None or pd.isna(val):
        return default
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


def parse_fbref_age(val) -> float:
    """Parse FBref age format (e.g. '31-324' -> 31.9 or 24 -> 24.0)."""
    if val is None or pd.isna(val):
        return 0.0
    val_str = str(val).strip()
    if "-" in val_str:
        parts = val_str.split("-")
        try:
            return round(float(parts[0]) + float(parts[1]) / 365.25, 1)
        except Exception:
            return float(parts[0])
    try:
        return float(val_str)
    except Exception:
        return 0.0


def build_player_scouting_dataset(
    scout_dir: Path = DEFAULT_SCOUT_DIR,
    tm_dir: Path = DEFAULT_TM_DIR,
    output_path: Path = DEFAULT_OUTPUT_PATH,
    min_minutes_threshold: int = 900,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build the clean Player Scouting / Replacement dataset.

    Args:
        scout_dir: Directory with FBref CSVs
        tm_dir: Directory with Transfermarkt CSVs
        output_path: Destination for the filtered ML-ready scouting dataset
        min_minutes_threshold: Minimum minutes threshold (default 900 mins)

    Returns:
        (filtered_df, unfiltered_df)
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print("Building Player Scouting Dataset...")

    # 1. Load the 6 FBref tables
    std_df = pd.read_csv(scout_dir / "big5_player_standard.csv", low_memory=False)
    sho_df = pd.read_csv(scout_dir / "big5_player_shooting.csv", low_memory=False)
    pas_df = pd.read_csv(scout_dir / "big5_player_passing.csv", low_memory=False)
    pos_df = pd.read_csv(scout_dir / "big5_player_possession.csv", low_memory=False)
    def_df = pd.read_csv(scout_dir / "big5_player_defense.csv", low_memory=False)
    msc_df = pd.read_csv(scout_dir / "big5_player_misc.csv", low_memory=False)

    # Filter for Premier League and Target Seasons
    join_keys = ["Url", "Season_End_Year", "Squad"]

    def filter_pl(df: pd.DataFrame) -> pd.DataFrame:
        sub = df[(df["Comp"] == "Premier League") & (df["Season_End_Year"].isin(TARGET_SEASON_END_YEARS))].copy()
        return sub.drop_duplicates(subset=join_keys)

    std_pl = filter_pl(std_df)
    sho_pl = filter_pl(sho_df)
    pas_pl = filter_pl(pas_df)
    pos_pl = filter_pl(pos_df)
    def_pl = filter_pl(def_df)
    msc_pl = filter_pl(msc_df)

    # 2. Sequential Left Joins on join_keys
    merged = std_pl.merge(
        sho_pl[["Url", "Season_End_Year", "Squad", "Sh_Standard"]],
        on=join_keys,
        how="left",
    )
    merged = merged.merge(
        pas_pl[["Url", "Season_End_Year", "Squad", "Cmp_percent_Total", "KP", "PrgP"]],
        on=join_keys,
        how="left",
    )
    merged = merged.merge(
        pos_pl[["Url", "Season_End_Year", "Squad", "PrgC_Carries", "Succ_Take"]],
        on=join_keys,
        how="left",
    )
    merged = merged.merge(
        def_pl[["Url", "Season_End_Year", "Squad", "TklW_Tackles", "Int", "Blocks_Blocks", "Clr"]],
        on=join_keys,
        how="left",
    )
    merged = merged.merge(
        msc_pl[["Url", "Season_End_Year", "Squad", "Won_percent_Aerial"]],
        on=join_keys,
        how="left",
    )

    # 3. Load Transfermarkt entities for Valuation & Demographic matching
    tm_players_path = tm_dir / "players.csv"
    tm_vals_path = tm_dir / "player_valuations.csv"

    tm_candidates = []
    tm_val_lookup = {}  # (player_id, season_end_year) -> valuation

    if tm_players_path.exists() and tm_vals_path.exists():
        tm_players = pd.read_csv(tm_players_path, usecols=["player_id", "name", "date_of_birth", "current_club_name", "position", "sub_position"])
        tm_players["norm_name"] = tm_players["name"].apply(normalize_player_name)
        tm_players["birth_year"] = pd.to_datetime(tm_players["date_of_birth"], errors="coerce").dt.year
        tm_players["canonical_team"] = tm_players["current_club_name"].apply(get_canonical_team_name)

        tm_candidates = tm_players[["player_id", "norm_name", "birth_year", "canonical_team", "position", "sub_position"]].dropna(subset=["norm_name"]).to_dict("records")

        # Load valuations
        vals_df = pd.read_csv(tm_vals_path, usecols=["player_id", "date", "market_value_in_eur"])
        vals_df["date"] = pd.to_datetime(vals_df["date"])

        # Index valuations by summer window of season_end_year (e.g. May-Sep of season_end_year)
        for s_year in TARGET_SEASON_END_YEARS:
            start_w = pd.Timestamp(f"{s_year}-05-01")
            end_w = pd.Timestamp(f"{s_year}-09-01")
            w_vals = vals_df[(vals_df["date"] >= start_w) & (vals_df["date"] <= end_w)].sort_values("date")
            latest_in_window = w_vals.groupby("player_id").last()["market_value_in_eur"].to_dict()
            for pid, m_val in latest_in_window.items():
                tm_val_lookup[(pid, s_year)] = m_val

    # 4. Construct player-season feature rows
    rows = []
    matched_count = 0
    total_players = len(merged)

    # Group candidate dict by birth year for faster indexing
    by_year_index = {}
    for c in tm_candidates:
        by = c.get("birth_year")
        if by not in by_year_index:
            by_year_index[by] = []
        by_year_index[by].append(c)

    for _, row in merged.iterrows():
        raw_name = str(row.get("Player", "") or "")
        norm_name = normalize_player_name(raw_name)
        s_year = int(row["Season_End_Year"])
        season_label = f"{s_year - 1}-{str(s_year)[-2:]}"
        squad_raw = str(row.get("Squad", "") or "")
        canonical_team = get_canonical_team_name(squad_raw)
        fbref_id = extract_fbref_id(row.get("Url"))

        # Age and Birth Year
        birth_year = None
        if pd.notna(row.get("Born")):
            try:
                birth_year = int(float(row["Born"]))
            except (ValueError, TypeError):
                birth_year = None

        age_val = parse_fbref_age(row.get("Age"))
        pos_val = str(row.get("Pos", "Unknown") or "Unknown")
        mins = safe_float(row.get("Min_Playing", 0.0))

        # Entity Resolution to Transfermarkt
        matched_tm = None
        if birth_year and birth_year in by_year_index:
            pool = by_year_index[birth_year]
            matched_tm = match_player_candidate(norm_name, birth_year, canonical_team, pool)
        if not matched_tm and tm_candidates:
            # Fallback across all candidates
            matched_tm = match_player_candidate(norm_name, birth_year, canonical_team, tm_candidates)

        tm_player_id = None
        market_val = np.nan
        tm_pos = None

        if matched_tm:
            matched_count += 1
            tm_player_id = matched_tm["player_id"]
            tm_pos = matched_tm.get("position")
            market_val = tm_val_lookup.get((tm_player_id, s_year), np.nan)

        # Compute per-90 metrics safely
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

        rows.append({
            "fbref_id": fbref_id,
            "tm_player_id": tm_player_id,
            "player_name": raw_name,
            "norm_name": norm_name,
            "team": canonical_team,
            "season": season_label,
            "season_end_year": s_year,
            "position": pos_val,
            "tm_position": tm_pos,
            "age": age_val,
            "minutes_played": mins,
            "market_value_eur": market_val,
            # Attacking Metrics
            "goals_per90": calculate_per90(gls, mins),
            "assists_per90": calculate_per90(asts, mins),
            "xG_per90": calculate_per90(xg, mins),
            "npxG_per90": calculate_per90(npxg, mins),
            "xAG_per90": calculate_per90(xag, mins),
            "shots_per90": calculate_per90(shots, mins),
            # Creation Metrics
            "key_passes_per90": calculate_per90(kp, mins),
            # Passing & Progression
            "pass_completion_pct": round(pass_cmp_pct, 1),
            "progressive_passes_per90": calculate_per90(prg_p, mins),
            "progressive_carries_per90": calculate_per90(prg_c, mins),
            "successful_takeons_per90": calculate_per90(takeons, mins),
            # Defending Metrics
            "tackles_won_per90": calculate_per90(tkl_w, mins),
            "interceptions_per90": calculate_per90(interceptions, mins),
            "blocks_per90": calculate_per90(blocks, mins),
            "clearances_per90": calculate_per90(clearances, mins),
            # Physical Metrics
            "aerial_won_pct": round(aerial_pct, 1),
        })

    full_df = pd.DataFrame(rows)
    # Save unfiltered dataset
    full_df.to_csv(UNFILTERED_OUTPUT_PATH, index=False, encoding="utf-8")
    print(f"[Scouting] Saved unfiltered dataset: {UNFILTERED_OUTPUT_PATH} ({len(full_df):,} records)")

    # Filter for main scouting similarity pool
    filtered_df = full_df[full_df["minutes_played"] >= min_minutes_threshold].copy().reset_index(drop=True)
    filtered_df.to_csv(output_path, index=False, encoding="utf-8")
    print(f"[Scouting] Saved filtered dataset (mins >= {min_minutes_threshold}): {output_path} ({len(filtered_df):,} records, {len(filtered_df.columns)} columns)")

    match_rate = (matched_count / total_players * 100) if total_players > 0 else 0.0
    print(f"[Scouting] TM Entity Match Rate: {matched_count}/{total_players} ({match_rate:.1f}%)")

    return filtered_df, full_df


if __name__ == "__main__":
    build_player_scouting_dataset()
