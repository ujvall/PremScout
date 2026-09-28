"""Transfer value dataset preprocessor and feature generator.

Temporal Integrity Rules:
1. All performance statistics and demographic indicators strictly reflect season S.
2. The target variable (market_value_eur) is the valuation recorded at the END of season S (summer).
3. The lag feature (prev_market_value_eur) is the valuation recorded at the START of season S (prior summer).
4. No future valuation or post-valuation performance is used as an input feature.
"""

from pathlib import Path
import numpy as np
import pandas as pd

from .entity_resolution import get_canonical_team_name, normalize_player_name
from .feature_engineering import assign_chronological_split, calculate_per90

DEFAULT_TM_RAW_DIR = Path("data/raw/transfermarkt")
DEFAULT_SCOUT_RAW_DIR = Path("data/raw/scouting")
DEFAULT_OUTPUT_PATH = Path("data/processed/transfer/transfer_value_dataset.csv")

TARGET_SEASONS = [2017, 2018, 2019, 2020, 2021, 2022, 2023]


def build_transfer_value_dataset(
    tm_dir: Path = DEFAULT_TM_RAW_DIR,
    scout_dir: Path = DEFAULT_SCOUT_RAW_DIR,
    output_path: Path = DEFAULT_OUTPUT_PATH,
    min_minutes_threshold: int = 270,
) -> pd.DataFrame:
    """Build the clean, leak-free Transfer Value Predictor dataset.

    Args:
        tm_dir: Directory containing Transfermarkt raw CSVs.
        scout_dir: Directory containing scouting raw CSVs.
        output_path: Destination path for processed CSV.
        min_minutes_threshold: Minimum competitive minutes played in season (default 270 mins).

    Returns:
        Processed DataFrame.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print("Building Transfer Value Dataset...")

    # 1. Load players demographic metadata
    players_df = pd.read_csv(
        tm_dir / "players.csv",
        usecols=[
            "player_id",
            "name",
            "date_of_birth",
            "sub_position",
            "position",
            "foot",
            "height_in_cm",
            "contract_expiration_date",
            "current_club_id",
            "current_club_name",
        ],
    )
    players_df["norm_name"] = players_df["name"].apply(normalize_player_name)
    players_df["birth_date"] = pd.to_datetime(players_df["date_of_birth"], errors="coerce")
    players_df["contract_exp_date"] = pd.to_datetime(players_df["contract_expiration_date"], errors="coerce")
    players_df = players_df.drop_duplicates(subset=["player_id"])

    # 2. Load clubs for club valuation
    clubs_df = pd.read_csv(tm_dir / "clubs.csv", usecols=["club_id", "name", "total_market_value"])
    clubs_df["canonical_club_name"] = clubs_df["name"].apply(get_canonical_team_name)
    club_val_map = clubs_df.set_index("club_id")["total_market_value"].to_dict()
    club_name_map = clubs_df.set_index("club_id")["canonical_club_name"].to_dict()

    # 3. Load games to map game_id -> season
    games_df = pd.read_csv(tm_dir / "games.csv", usecols=["game_id", "season"])
    games_df = games_df[games_df["season"].isin(TARGET_SEASONS)]

    # 4. Load appearances and aggregate per (player_id, season)
    apps_df = pd.read_csv(
        tm_dir / "appearances.csv",
        usecols=[
            "game_id",
            "player_id",
            "player_club_id",
            "minutes_played",
            "goals",
            "assists",
            "yellow_cards",
            "red_cards",
        ],
    )
    merged_apps = apps_df.merge(games_df, on="game_id", how="inner")

    # Group by (player_id, season)
    season_stats = (
        merged_apps.groupby(["player_id", "season"])
        .agg(
            minutes_played=("minutes_played", "sum"),
            appearances=("game_id", "count"),
            goals=("goals", "sum"),
            assists=("assists", "sum"),
            yellow_cards=("yellow_cards", "sum"),
            red_cards=("red_cards", "sum"),
            primary_club_id=("player_club_id", lambda s: s.mode()[0] if not s.empty else np.nan),
        )
        .reset_index()
    )

    # Filter min minutes
    season_stats = season_stats[season_stats["minutes_played"] >= min_minutes_threshold].copy()

    # 5. Load historical valuations
    vals_df = pd.read_csv(
        tm_dir / "player_valuations.csv",
        usecols=["player_id", "date", "market_value_in_eur"],
    )
    vals_df["date"] = pd.to_datetime(vals_df["date"])
    vals_df = vals_df.sort_values("date")

    # 6. Load FBref advanced stats for optional enrichment
    fbref_map = {}
    fbref_path = scout_dir / "big5_player_standard.csv"
    if fbref_path.exists():
        fb_df = pd.read_csv(fbref_path, low_memory=False)
        # Standardize season
        fb_df["season_year"] = pd.to_numeric(fb_df["Season_End_Year"], errors="coerce")
        fb_df["norm_name"] = fb_df["Player"].apply(normalize_player_name)
        fb_df["birth_year"] = pd.to_numeric(fb_df["Born"], errors="coerce")

        # Key metrics where present
        for _, fb_row in fb_df.iterrows():
            key = (fb_row["norm_name"], fb_row["season_year"])
            fbref_map[key] = {
                "shots_per90": float(fb_row.get("Sh_per_90_Standard", 0.0) or 0.0) if "Sh_per_90_Standard" in fb_row else 0.0,
                "xG": float(fb_row.get("xG_Expected", 0.0) or 0.0) if "xG_Expected" in fb_row else 0.0,
                "npxG": float(fb_row.get("npxG_Expected", 0.0) or 0.0) if "npxG_Expected" in fb_row else 0.0,
            }

    # 7. Build season-by-season records
    records = []

    for season_year in TARGET_SEASONS:
        season_label = f"{season_year}-{str(season_year + 1)[-2:]}"
        season_subset = season_stats[season_stats["season"] == season_year]

        # Prior valuation window: latest valuation on or before Sep 1 of season_year
        prior_val_cutoff = pd.Timestamp(f"{season_year}-09-01")
        # Target valuation window: latest valuation between May 1 and Sep 1 of season_year + 1
        target_val_start = pd.Timestamp(f"{season_year + 1}-05-01")
        target_val_end = pd.Timestamp(f"{season_year + 1}-09-01")

        # Compute prior valuations for this season
        prior_vals = (
            vals_df[vals_df["date"] <= prior_val_cutoff]
            .groupby("player_id")
            .last()["market_value_in_eur"]
            .to_dict()
        )

        # Compute dynamic club squad market valuation prior to season
        club_priors = {}
        for _, s_row in season_subset.iterrows():
            c_id = s_row["primary_club_id"]
            if pd.notna(c_id):
                club_priors[c_id] = club_priors.get(c_id, 0.0) + prior_vals.get(s_row["player_id"], 0.0)

        # Compute target valuations for this season
        target_vals = (
            vals_df[(vals_df["date"] >= target_val_start) & (vals_df["date"] <= target_val_end)]
            .groupby("player_id")
            .last()["market_value_in_eur"]
            .to_dict()
        )

        # Merge with player demographics
        merged_season = season_subset.merge(players_df, on="player_id", how="inner")

        for _, row in merged_season.iterrows():
            p_id = row["player_id"]
            target_val = target_vals.get(p_id)
            prev_val = prior_vals.get(p_id)

            # Both previous and target valuations must be valid and positive
            if not target_val or target_val <= 0 or not prev_val or prev_val <= 0:
                continue

            mins = int(row["minutes_played"])
            gls = int(row["goals"])
            asts = int(row["assists"])

            # Age at season kickoff
            season_start_date = pd.Timestamp(f"{season_year}-08-01")
            b_date = row["birth_date"]
            age = round((season_start_date - b_date).days / 365.25, 1) if pd.notna(b_date) else np.nan

            # Contract remaining
            c_exp = row["contract_exp_date"]
            season_end_date = pd.Timestamp(f"{season_year + 1}-06-30")
            if pd.notna(c_exp):
                contract_years = max(round((c_exp - season_end_date).days / 365.25, 2), 0.0)
            else:
                contract_years = np.nan

            # Club info
            p_club_id = row["primary_club_id"]
            club_val = club_priors.get(p_club_id, np.nan)
            if club_val is not None and (pd.isna(club_val) or club_val <= 0):
                club_val = np.nan
            club_name = club_name_map.get(p_club_id, row.get("current_club_name", "Unknown"))

            # Per 90 metrics
            g_per90 = calculate_per90(gls, mins)
            a_per90 = calculate_per90(asts, mins)

            # Check FBref metrics
            fb_info = fbref_map.get((row["norm_name"], season_year + 1), {})
            shots_per90 = fb_info.get("shots_per90", 0.0)
            npxg_val = fb_info.get("npxG", 0.0)
            npxg_per90 = calculate_per90(npxg_val, mins)

            split_group = assign_chronological_split(season_label)

            records.append({
                "player_id": p_id,
                "player_name": row["name"],
                "season": season_label,
                "age": age,
                "primary_position": row["position"] if pd.notna(row["position"]) else "Unknown",
                "sub_position": row["sub_position"] if pd.notna(row["sub_position"]) else "Unknown",
                "height_cm": float(row["height_in_cm"]) if pd.notna(row["height_in_cm"]) else np.nan,
                "preferred_foot": row["foot"] if pd.notna(row["foot"]) else "Unknown",
                "contract_years_left": contract_years,
                "club_name": club_name,
                "club_market_value_eur": club_val,
                "minutes_played": mins,
                "appearances": int(row["appearances"]),
                "goals": gls,
                "assists": asts,
                "yellow_cards": int(row["yellow_cards"]),
                "red_cards": int(row["red_cards"]),
                "goals_per90": g_per90,
                "assists_per90": a_per90,
                "shots_per90": shots_per90,
                "npxg_per90": npxg_per90,
                "prev_market_value_eur": float(prev_val),
                "split_group": split_group,
                "market_value_eur": float(target_val),  # Target Variable
            })

    out_df = pd.DataFrame(records)
    # Deduplicate in case of duplicate player-season pairs
    out_df = out_df.drop_duplicates(subset=["player_id", "season"]).reset_index(drop=True)
    out_df.to_csv(output_path, index=False, encoding="utf-8")
    print(f"[Transfer Dataset] Successfully built: {output_path} ({len(out_df):,} rows, {len(out_df.columns)} columns)")
    return out_df


if __name__ == "__main__":
    build_transfer_value_dataset()
