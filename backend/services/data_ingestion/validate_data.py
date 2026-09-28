"""Data validation service for Football Intelligence Platform (PremAI).

Validates downloaded raw files for:
1. Premier League match files (matches/2017-18 to 2023-24)
2. Transfermarkt relational tables (transfermarkt/)
3. Scouting datasets (scouting/ - FBref and FPL)
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import pandas as pd


@dataclass
class DatasetReport:
    category: str
    filename: str
    file_path: str
    file_size_bytes: int
    row_count: int
    col_count: int
    columns: list[str]
    missing_value_counts: dict[str, int]
    duplicate_rows: int
    min_date: str | None = None
    max_date: str | None = None
    unique_players: int | None = None
    unique_teams: int | None = None
    extra_checks: dict[str, Any] = field(default_factory=dict)


def validate_match_csv(file_path: Path, season_label: str) -> DatasetReport:
    """Validate a single season Premier League match CSV."""
    df = pd.read_csv(file_path, encoding="utf-8-sig")

    # Match-specific column detection
    date_col = next((c for c in ["Date", "date"] if c in df.columns), None)
    home_col = next((c for c in ["HomeTeam", "Home"] if c in df.columns), None)
    away_col = next((c for c in ["AwayTeam", "Away"] if c in df.columns), None)
    res_col = next((c for c in ["FTR", "Result"] if c in df.columns), None)
    fthg_col = next((c for c in ["FTHG", "HomeGoals"] if c in df.columns), None)
    ftag_col = next((c for c in ["FTAG", "AwayGoals"] if c in df.columns), None)

    # Date range
    min_d, max_d = None, None
    if date_col and date_col in df.columns:
        dates = pd.to_datetime(df[date_col], dayfirst=True, errors="coerce").dropna()
        if not dates.empty:
            min_d = str(dates.min().date())
            max_d = str(dates.max().date())

    # Unique teams
    teams = set()
    if home_col:
        teams.update(df[home_col].dropna().unique())
    if away_col:
        teams.update(df[away_col].dropna().unique())

    # Extra checks
    extra = {}
    if res_col and res_col in df.columns:
        extra["result_distribution"] = df[res_col].value_counts().to_dict()

    # Invalid scores
    if fthg_col and ftag_col and fthg_col in df.columns and ftag_col in df.columns:
        invalid_scores = df[(df[fthg_col] < 0) | (df[ftag_col] < 0) | (df[fthg_col] > 20) | (df[ftag_col] > 20)]
        extra["invalid_scores_count"] = len(invalid_scores)

    # Duplicate fixtures (HomeTeam vs AwayTeam in same season)
    if home_col and away_col and home_col in df.columns and away_col in df.columns:
        fixture_dupes = df.duplicated(subset=[home_col, away_col]).sum()
        extra["duplicate_fixtures"] = int(fixture_dupes)

    extra["matches_per_season_check"] = "PASS (~380 matches)" if 370 <= len(df) <= 390 else f"CHECK: {len(df)} matches"

    missing = {k: int(v) for k, v in df.isnull().sum().items() if v > 0}

    return DatasetReport(
        category=f"Matches ({season_label})",
        filename=file_path.name,
        file_path=str(file_path),
        file_size_bytes=file_path.stat().st_size,
        row_count=len(df),
        col_count=len(df.columns),
        columns=list(df.columns),
        missing_value_counts=missing,
        duplicate_rows=int(df.duplicated().sum()),
        min_date=min_d,
        max_date=max_d,
        unique_players=None,
        unique_teams=len(teams),
        extra_checks=extra,
    )


def validate_player_or_transfer_csv(file_path: Path, category: str) -> DatasetReport:
    """Validate a player or transfer CSV table."""
    # Read sample or full table
    df = pd.read_csv(file_path, low_memory=False, encoding="utf-8-sig")

    date_col = next((c for c in ["date", "Date", "transfer_date", "contract_expiration_date"] if c in df.columns), None)
    min_d, max_d = None, None
    if date_col and date_col in df.columns:
        parsed_dates = pd.to_datetime(df[date_col], errors="coerce").dropna()
        if not parsed_dates.empty:
            min_d = str(parsed_dates.min().date())
            max_d = str(parsed_dates.max().date())

    # Player and team detection
    player_id_col = next((c for c in ["player_id", "Player_ID", "id"] if c in df.columns), None)
    player_name_col = next((c for c in ["name", "player_name", "Player", "second_name"] if c in df.columns), None)
    team_col = next((c for c in ["club_name", "current_club_name", "Squad", "team_name", "from_club_name"] if c in df.columns), None)
    pos_col = next((c for c in ["position", "sub_position", "Pos", "element_type"] if c in df.columns), None)
    mv_col = next((c for c in ["market_value_in_eur", "market_value", "now_cost"] if c in df.columns), None)
    min_col = next((c for c in ["minutes_played", "minutes", "Min_Playing", "Mins_Per_90"] if c in df.columns), None)

    unique_players = None
    if player_id_col and player_id_col in df.columns:
        unique_players = int(df[player_id_col].nunique())
    elif player_name_col and player_name_col in df.columns:
        unique_players = int(df[player_name_col].nunique())

    unique_teams = None
    if team_col and team_col in df.columns:
        unique_teams = int(df[team_col].nunique())

    extra = {}
    if player_id_col and player_id_col in df.columns and file_path.stem == "players":
        extra["duplicate_player_ids"] = int(df.duplicated(subset=[player_id_col]).sum())
    if player_name_col and player_name_col in df.columns:
        extra["missing_player_names"] = int(df[player_name_col].isnull().sum())
    if pos_col and pos_col in df.columns:
        extra["missing_positions"] = int(df[pos_col].isnull().sum())
    if mv_col and mv_col in df.columns:
        extra["missing_market_values"] = int(df[mv_col].isnull().sum())
    if min_col and min_col in df.columns:
        extra["missing_minutes"] = int(df[min_col].isnull().sum())

    season_col = next((c for c in ["season", "Season_End_Year", "transfer_season", "last_season"] if c in df.columns), None)
    if season_col and season_col in df.columns:
        extra["season_coverage"] = sorted([str(s) for s in df[season_col].dropna().unique()])

    missing = {k: int(v) for k, v in df.isnull().sum().items() if v > 0}

    return DatasetReport(
        category=category,
        filename=file_path.name,
        file_path=str(file_path),
        file_size_bytes=file_path.stat().st_size,
        row_count=len(df),
        col_count=len(df.columns),
        columns=list(df.columns),
        missing_value_counts=missing,
        duplicate_rows=int(df.duplicated().sum()),
        min_date=min_d,
        max_date=max_d,
        unique_players=unique_players,
        unique_teams=unique_teams,
        extra_checks=extra,
    )


def run_all_validations(base_data_dir: Path = Path("data/raw")) -> list[DatasetReport]:
    """Inspect and validate all raw datasets."""
    reports: list[DatasetReport] = []

    # 1. Match datasets
    matches_dir = base_data_dir / "matches"
    if matches_dir.exists():
        for season_dir in sorted(matches_dir.iterdir()):
            csv_file = season_dir / "E0.csv"
            if csv_file.exists():
                rep = validate_match_csv(csv_file, season_label=season_dir.name)
                reports.append(rep)

    # 2. Transfermarkt datasets
    tm_dir = base_data_dir / "transfermarkt"
    if tm_dir.exists():
        for csv_file in sorted(tm_dir.glob("*.csv")):
            rep = validate_player_or_transfer_csv(csv_file, category="Transfermarkt")
            reports.append(rep)

    # 3. Scouting datasets
    scout_dir = base_data_dir / "scouting"
    if scout_dir.exists():
        for csv_file in sorted(scout_dir.glob("*.csv")):
            rep = validate_player_or_transfer_csv(csv_file, category="Scouting")
            reports.append(rep)

    return reports


def print_validation_summary(reports: list[DatasetReport]) -> None:
    """Print a clean CLI summary of validation results."""
    print("\n" + "=" * 90)
    print("DATA VALIDATION SUMMARY REPORT")
    print("=" * 90)
    for r in reports:
        print(f"\n[{r.category}] -> {r.filename}")
        print(f"  Path: {r.file_path}")
        print(f"  Size: {r.file_size_bytes:,} bytes | Rows: {r.row_count:,} | Cols: {r.col_count:,} | Dupe Rows: {r.duplicate_rows}")
        if r.min_date and r.max_date:
            print(f"  Date Range: {r.min_date} to {r.max_date}")
        if r.unique_players is not None:
            print(f"  Unique Players: {r.unique_players:,}")
        if r.unique_teams is not None:
            print(f"  Unique Teams: {r.unique_teams:,}")
        if r.extra_checks:
            print(f"  Checks: {r.extra_checks}")
        null_count = sum(r.missing_value_counts.values())
        print(f"  Total Missing Values across all cols: {null_count:,}")
    print("\n" + "=" * 90 + "\n")


if __name__ == "__main__":
    reps = run_all_validations()
    print_validation_summary(reps)
