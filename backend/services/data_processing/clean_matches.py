"""Match outcome dataset preprocessor and feature generator.

Strict Data Leakage Rules:
1. Every feature for match T is calculated strictly using matches played BEFORE date T.
2. Zero same-match post-kickoff statistics (goals, shots, corners, fouls, cards) are used as features.
3. Pre-match league table positions are calculated dynamically as of kickoff, not post-season.
4. Bookmaker odds are intentionally excluded from the core ML dataset to ensure the model
   learns intrinsic football dynamics.
"""

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
import pandas as pd

from .entity_resolution import get_canonical_team_name
from .feature_engineering import assign_chronological_split

DEFAULT_MATCH_RAW_DIR = Path("data/raw/matches")
DEFAULT_PROCESSED_PATH = Path("data/processed/matches/match_outcome_dataset.csv")


@dataclass
class TeamMatchRecord:
    date: pd.Timestamp
    season: str
    is_home: bool
    opponent: str
    result: str  # 'W', 'D', 'L' from this team's perspective
    points: int
    gf: int
    ga: int
    shots: int
    sot: int
    sot_conceded: int
    corners: int


def build_match_outcome_dataset(
    raw_dir: Path = DEFAULT_MATCH_RAW_DIR,
    output_path: Path = DEFAULT_PROCESSED_PATH,
) -> pd.DataFrame:
    """Build the clean, leak-free Match Outcome Predictor dataset.

    Args:
        raw_dir: Directory containing season subfolders with E0.csv
        output_path: Destination path for the processed CSV

    Returns:
        Processed DataFrame ready for ML modeling.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. Load and sort all matches chronologically
    dfs = []
    needed_cols = ["Date", "HomeTeam", "AwayTeam", "FTR", "FTHG", "FTAG", "HS", "AS", "HST", "AST", "HC", "AC"]
    for csv_file in sorted(raw_dir.glob("*/E0.csv")):
        season = csv_file.parent.name
        df_season = pd.read_csv(csv_file, encoding="utf-8-sig")
        cols_present = [c for c in needed_cols if c in df_season.columns]
        df_sub = df_season[cols_present].copy()
        df_sub["season"] = season
        dfs.append(df_sub)

    if not dfs:
        raise FileNotFoundError(f"No match CSV files found under {raw_dir}")

    raw_matches = pd.concat(dfs, ignore_index=True)

    # Clean dates and targets
    raw_matches["date"] = pd.to_datetime(raw_matches["Date"], dayfirst=True, errors="coerce")
    raw_matches = raw_matches.dropna(subset=["date", "HomeTeam", "AwayTeam", "FTR"])
    raw_matches = raw_matches[raw_matches["FTR"].isin(["H", "D", "A"])].copy()

    # Map teams to canonical names
    raw_matches["home_team"] = raw_matches["HomeTeam"].apply(get_canonical_team_name)
    raw_matches["away_team"] = raw_matches["AwayTeam"].apply(get_canonical_team_name)

    # Sort strictly chronologically
    raw_matches = raw_matches.sort_values(["date", "season"]).reset_index(drop=True)

    # State tracking structures for leakage-free feature calculation
    team_history: dict[str, list[TeamMatchRecord]] = defaultdict(list)
    # Season table: season -> team -> {'points': 0, 'gf': 0, 'ga': 0, 'matches': 0}
    season_tables: dict[str, dict[str, dict[str, int]]] = defaultdict(lambda: defaultdict(lambda: {"points": 0, "gf": 0, "ga": 0, "matches": 0}))
    # H2H history: (team_a, team_b) sorted tuple -> list of prior results
    h2h_history: dict[tuple[str, str], list[dict]] = defaultdict(list)

    rows = []

    for idx, row in raw_matches.iterrows():
        match_date = row["date"]
        season = row["season"]
        home_t = row["home_team"]
        away_t = row["away_team"]
        ftr = row["FTR"]  # Target: 'H', 'D', 'A'
        fthg = int(row["FTHG"])
        ftag = int(row["FTAG"])
        hs = int(row.get("HS", 0) or 0)
        as_ = int(row.get("AS", 0) or 0)
        hst = int(row.get("HST", 0) or 0)
        ast = int(row.get("AST", 0) or 0)
        hc = int(row.get("HC", 0) or 0)
        ac = int(row.get("AC", 0) or 0)

        # -------------------------------------------------------------
        # Feature 1: Days of Rest (prior to kickoff)
        # -------------------------------------------------------------
        home_prev_matches = [m for m in team_history[home_t] if m.date < match_date]
        away_prev_matches = [m for m in team_history[away_t] if m.date < match_date]

        if home_prev_matches:
            home_rest_days = min(int((match_date - home_prev_matches[-1].date).days), 60)
        else:
            home_rest_days = 14  # Sensible default for season opener

        if away_prev_matches:
            away_rest_days = min(int((match_date - away_prev_matches[-1].date).days), 60)
        else:
            away_rest_days = 14

        # -------------------------------------------------------------
        # Feature 2: Rolling 5 Match Indicators (Home Team)
        # -------------------------------------------------------------
        h_last5 = home_prev_matches[-5:]
        n_h5 = len(h_last5)
        if n_h5 > 0:
            h_roll_pts_5 = round(sum(m.points for m in h_last5) / n_h5, 2)
            h_roll_gf_5 = round(sum(m.gf for m in h_last5) / n_h5, 2)
            h_roll_ga_5 = round(sum(m.ga for m in h_last5) / n_h5, 2)
            h_roll_shots_5 = round(sum(m.shots for m in h_last5) / n_h5, 2)
            h_roll_sot_5 = round(sum(m.sot for m in h_last5) / n_h5, 2)
            h_roll_sot_conceded_5 = round(sum(m.sot_conceded for m in h_last5) / n_h5, 2)
            h_roll_corners_5 = round(sum(m.corners for m in h_last5) / n_h5, 2)
            h_roll_win_pct_5 = round(sum(1 for m in h_last5 if m.result == "W") / n_h5, 2)
        else:
            h_roll_pts_5 = 1.3
            h_roll_gf_5 = 1.3
            h_roll_ga_5 = 1.3
            h_roll_shots_5 = 11.0
            h_roll_sot_5 = 4.0
            h_roll_sot_conceded_5 = 4.0
            h_roll_corners_5 = 5.0
            h_roll_win_pct_5 = 0.35

        # -------------------------------------------------------------
        # Feature 3: Rolling 5 Match Indicators (Away Team)
        # -------------------------------------------------------------
        a_last5 = away_prev_matches[-5:]
        n_a5 = len(a_last5)
        if n_a5 > 0:
            a_roll_pts_5 = round(sum(m.points for m in a_last5) / n_a5, 2)
            a_roll_gf_5 = round(sum(m.gf for m in a_last5) / n_a5, 2)
            a_roll_ga_5 = round(sum(m.ga for m in a_last5) / n_a5, 2)
            a_roll_shots_5 = round(sum(m.shots for m in a_last5) / n_a5, 2)
            a_roll_sot_5 = round(sum(m.sot for m in a_last5) / n_a5, 2)
            a_roll_sot_conceded_5 = round(sum(m.sot_conceded for m in a_last5) / n_a5, 2)
            a_roll_corners_5 = round(sum(m.corners for m in a_last5) / n_a5, 2)
            a_roll_win_pct_5 = round(sum(1 for m in a_last5 if m.result == "W") / n_a5, 2)
        else:
            a_roll_pts_5 = 1.3
            a_roll_gf_5 = 1.3
            a_roll_ga_5 = 1.3
            a_roll_shots_5 = 11.0
            a_roll_sot_5 = 4.0
            a_roll_sot_conceded_5 = 4.0
            a_roll_corners_5 = 5.0
            a_roll_win_pct_5 = 0.35

        # -------------------------------------------------------------
        # Feature 4: Dynamic Pre-Match League Table & Rank
        # -------------------------------------------------------------
        current_table = season_tables[season]
        # Rank teams by points -> GD -> GF
        team_stats_list = []
        for t_name, stats in current_table.items():
            gd = stats["gf"] - stats["ga"]
            team_stats_list.append((t_name, stats["points"], gd, stats["gf"]))

        # Sort descending
        team_stats_list.sort(key=lambda x: (x[1], x[2], x[3]), reverse=True)
        rankings = {item[0]: rank + 1 for rank, item in enumerate(team_stats_list)}

        # Default rank is 10 for teams that have not yet played in the season
        home_rank = rankings.get(home_t, 10)
        away_rank = rankings.get(away_t, 10)
        rank_diff = away_rank - home_rank  # Positive = home team ranked higher (e.g. 2nd vs 8th -> +6)

        home_season_pts = current_table[home_t]["points"]
        away_season_pts = current_table[away_t]["points"]
        home_season_matches = current_table[home_t]["matches"]
        away_season_matches = current_table[away_t]["matches"]

        # Points per game in current season before this match
        home_ppg = round(home_season_pts / home_season_matches, 2) if home_season_matches > 0 else 1.3
        away_ppg = round(away_season_pts / away_season_matches, 2) if away_season_matches > 0 else 1.3

        # -------------------------------------------------------------
        # Feature 5: Prior Head-to-Head Record
        # -------------------------------------------------------------
        h2h_key = tuple(sorted([home_t, away_t]))
        prior_meetings = [m for m in h2h_history[h2h_key] if m["date"] < match_date]
        h2h_total = len(prior_meetings)
        if h2h_total > 0:
            h2h_h_wins = sum(1 for m in prior_meetings if (m["home_team"] == home_t and m["ftr"] == "H") or (m["away_team"] == home_t and m["ftr"] == "A"))
            h2h_home_win_pct = round(h2h_h_wins / h2h_total, 2)
        else:
            h2h_home_win_pct = 0.45  # Historical baseline home win percentage

        # Split group metadata
        split_group = assign_chronological_split(season)

        # Append feature row
        rows.append({
            "match_id": f"{season}_{idx:04d}",
            "date": match_date.strftime("%Y-%m-%d"),
            "season": season,
            "home_team": home_t,
            "away_team": away_t,
            "home_rest_days": home_rest_days,
            "away_rest_days": away_rest_days,
            "home_rolling_points_5": h_roll_pts_5,
            "away_rolling_points_5": a_roll_pts_5,
            "home_rolling_gf_5": h_roll_gf_5,
            "away_rolling_gf_5": a_roll_gf_5,
            "home_rolling_ga_5": h_roll_ga_5,
            "away_rolling_ga_5": a_roll_ga_5,
            "home_rolling_shots_5": h_roll_shots_5,
            "away_rolling_shots_5": a_roll_shots_5,
            "home_rolling_sot_5": h_roll_sot_5,
            "away_rolling_sot_5": a_roll_sot_5,
            "home_rolling_sot_conceded_5": h_roll_sot_conceded_5,
            "away_rolling_sot_conceded_5": a_roll_sot_conceded_5,
            "home_rolling_corners_5": h_roll_corners_5,
            "away_rolling_corners_5": a_roll_corners_5,
            "home_rolling_win_pct_5": h_roll_win_pct_5,
            "away_rolling_win_pct_5": a_roll_win_pct_5,
            "home_league_rank": home_rank,
            "away_league_rank": away_rank,
            "rank_diff": rank_diff,
            "home_ppg": home_ppg,
            "away_ppg": away_ppg,
            "h2h_matches_prior": h2h_total,
            "h2h_home_win_pct": h2h_home_win_pct,
            "split_group": split_group,
            "match_result": ftr,  # Target variable
        })

        # -------------------------------------------------------------
        # Post-Match State Update (Only done AFTER recording pre-match features)
        # -------------------------------------------------------------
        # Home points and result
        if ftr == "H":
            h_pts, a_pts = 3, 0
            h_res, a_res = "W", "L"
        elif ftr == "D":
            h_pts, a_pts = 1, 1
            h_res, a_res = "D", "D"
        else:
            h_pts, a_pts = 0, 3
            h_res, a_res = "L", "W"

        # Update Home Team History
        team_history[home_t].append(TeamMatchRecord(
            date=match_date,
            season=season,
            is_home=True,
            opponent=away_t,
            result=h_res,
            points=h_pts,
            gf=fthg,
            ga=ftag,
            shots=hs,
            sot=hst,
            sot_conceded=ast,
            corners=hc,
        ))

        # Update Away Team History
        team_history[away_t].append(TeamMatchRecord(
            date=match_date,
            season=season,
            is_home=False,
            opponent=home_t,
            result=a_res,
            points=a_pts,
            gf=ftag,
            ga=fthg,
            shots=as_,
            sot=ast,
            sot_conceded=hst,
            corners=ac,
        ))

        # Update League Table
        current_table[home_t]["points"] += h_pts
        current_table[home_t]["gf"] += fthg
        current_table[home_t]["ga"] += ftag
        current_table[home_t]["matches"] += 1

        current_table[away_t]["points"] += a_pts
        current_table[away_t]["gf"] += ftag
        current_table[away_t]["ga"] += fthg
        current_table[away_t]["matches"] += 1

        # Update H2H History
        h2h_history[h2h_key].append({
            "date": match_date,
            "home_team": home_t,
            "away_team": away_t,
            "ftr": ftr,
        })

    processed_df = pd.DataFrame(rows)
    processed_df.to_csv(output_path, index=False, encoding="utf-8")
    print(f"[Match Dataset] Successfully built: {output_path} ({len(processed_df):,} matches, {len(processed_df.columns)} columns)")
    return processed_df


if __name__ == "__main__":
    build_match_outcome_dataset()
