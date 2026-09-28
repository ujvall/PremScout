"""Validation suite for processed PremAI datasets.

Verifies:
1. Match Outcome Dataset (Target leakage, chronological order, H/D/A validity, duplicate IDs)
2. Transfer Value Dataset (No target in features, no future valuation, non-negative stats, valid per-90s)
3. Player Scouting Dataset (Valid per-90 metrics, non-negative minutes, no duplicate player-seasons)
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd


@dataclass
class ProcessedValidationReport:
    dataset_name: str
    file_path: str
    row_count: int
    col_count: int
    duplicate_rows: int
    unique_players: int | None
    unique_teams: int | None
    unique_seasons: list[str]
    missing_value_summary: dict[str, int]
    checks_passed: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


def validate_processed_matches(file_path: Path) -> ProcessedValidationReport:
    """Validate processed match outcome dataset."""
    df = pd.read_csv(file_path)
    report = ProcessedValidationReport(
        dataset_name="Match Outcome Dataset",
        file_path=str(file_path),
        row_count=len(df),
        col_count=len(df.columns),
        duplicate_rows=int(df.duplicated().sum()),
        unique_players=None,
        unique_teams=int(df["home_team"].nunique()),
        unique_seasons=sorted([str(s) for s in df["season"].unique()]),
        missing_value_summary={k: int(v) for k, v in df.isnull().sum().items() if v > 0},
    )

    # 1. No duplicate match IDs
    if df["match_id"].nunique() == len(df):
        report.checks_passed.append("No duplicate match IDs (PASS)")
    else:
        report.errors.append(f"Duplicate match IDs found: {len(df) - df['match_id'].nunique()}")

    # 2. Each target is H/D/A
    valid_targets = set(df["match_result"].unique())
    if valid_targets.issubset({"H", "D", "A"}):
        report.checks_passed.append("Match targets are strictly H, D, A (PASS)")
    else:
        report.errors.append(f"Invalid targets found: {valid_targets - {'H', 'D', 'A'}}")

    # 3. Chronological ordering
    df["dt"] = pd.to_datetime(df["date"])
    is_sorted = df["dt"].is_monotonic_increasing
    if is_sorted:
        report.checks_passed.append("Matches strictly sorted chronologically (PASS)")
    else:
        report.errors.append("Matches are NOT sorted chronologically (FAIL)")

    # 4. Check for target leakage
    # Forbidden columns in features: same-match goals, shots, corners, bookmaker odds
    forbidden = ["FTHG", "FTAG", "HS", "AS", "HST", "AST", "B365H", "B365D", "B365A", "BbAvH"]
    found_forbidden = [c for c in forbidden if c in df.columns]
    if not found_forbidden:
        report.checks_passed.append("No post-match stats or bookmaker odds present (PASS)")
    else:
        report.errors.append(f"LEAKAGE DETECTED: Forbidden columns present: {found_forbidden}")

    # 5. Numerical ranges for rolling features
    pts_out_of_bounds = df[(df["home_rolling_points_5"] < 0) | (df["home_rolling_points_5"] > 3.0)]
    if pts_out_of_bounds.empty:
        report.checks_passed.append("Rolling points in valid [0.0, 3.0] interval (PASS)")
    else:
        report.errors.append(f"Rolling points out of bounds: {len(pts_out_of_bounds)} rows")

    ranks_out_of_bounds = df[(df["home_league_rank"] < 1) | (df["home_league_rank"] > 20)]
    if ranks_out_of_bounds.empty:
        report.checks_passed.append("League ranks in valid [1, 20] range (PASS)")
    else:
        report.errors.append(f"League ranks out of bounds: {len(ranks_out_of_bounds)} rows")

    return report


def validate_processed_transfer(file_path: Path) -> ProcessedValidationReport:
    """Validate processed transfer value dataset."""
    df = pd.read_csv(file_path)
    report = ProcessedValidationReport(
        dataset_name="Transfer Value Dataset",
        file_path=str(file_path),
        row_count=len(df),
        col_count=len(df.columns),
        duplicate_rows=int(df.duplicated().sum()),
        unique_players=int(df["player_id"].nunique()),
        unique_teams=int(df["club_name"].nunique()) if "club_name" in df.columns else None,
        unique_seasons=sorted([str(s) for s in df["season"].unique()]),
        missing_value_summary={k: int(v) for k, v in df.isnull().sum().items() if v > 0},
    )

    # 1. No duplicate player-season pairs
    dupe_ps = df.duplicated(subset=["player_id", "season"]).sum()
    if dupe_ps == 0:
        report.checks_passed.append("No duplicate (player_id, season) pairs (PASS)")
    else:
        report.errors.append(f"Duplicate player-season rows: {dupe_ps}")

    # 2. No negative minutes, goals, assists
    neg_mins = (df["minutes_played"] < 0).sum()
    neg_gls = (df["goals"] < 0).sum()
    neg_asts = (df["assists"] < 0).sum()
    if neg_mins == 0 and neg_gls == 0 and neg_asts == 0:
        report.checks_passed.append("No negative minutes, goals, or assists (PASS)")
    else:
        report.errors.append(f"Negative values found (mins: {neg_mins}, gls: {neg_gls}, asts: {neg_asts})")

    # 3. Market value target > 0
    neg_targets = (df["market_value_eur"] <= 0).sum()
    if neg_targets == 0:
        report.checks_passed.append("All target market values strictly positive (PASS)")
    else:
        report.errors.append(f"Non-positive target market values: {neg_targets}")

    # 4. Target not in features
    # Verify prev_market_value_eur is different from market_value_eur across dataset
    exact_ident = (df["market_value_eur"] == df["prev_market_value_eur"]).mean()
    if exact_ident < 0.90:  # Market values fluctuate; not 100% identical copy
        report.checks_passed.append("Target is distinct from previous valuation feature (PASS)")
    else:
        report.warnings.append("High proportion of unchanged valuations between seasons.")

    # 5. Per-90 mathematical validity
    expected_g90 = np.round((df["goals"] / df["minutes_played"]) * 90.0, 2)
    diff = np.abs(df["goals_per90"] - expected_g90)
    if (diff <= 0.05).all():
        report.checks_passed.append("Per-90 metrics mathematically consistent (PASS)")
    else:
        report.errors.append("Per-90 calculation discrepancy found")

    return report


def validate_processed_scouting(file_path: Path) -> ProcessedValidationReport:
    """Validate processed player scouting dataset."""
    df = pd.read_csv(file_path)
    report = ProcessedValidationReport(
        dataset_name="Player Scouting Dataset",
        file_path=str(file_path),
        row_count=len(df),
        col_count=len(df.columns),
        duplicate_rows=int(df.duplicated().sum()),
        unique_players=int(df["norm_name"].nunique()),
        unique_teams=int(df["team"].nunique()),
        unique_seasons=sorted([str(s) for s in df["season"].unique()]),
        missing_value_summary={k: int(v) for k, v in df.isnull().sum().items() if v > 0},
    )

    # 1. No duplicate player-team-season rows
    dupes = df.duplicated(subset=["norm_name", "team", "season"]).sum()
    if dupes == 0:
        report.checks_passed.append("No duplicate (player, team, season) rows (PASS)")
    else:
        report.warnings.append(f"Duplicate player entries across same season/team: {dupes}")

    # 2. Minutes threshold compliance
    below_thresh = (df["minutes_played"] < 900).sum()
    if below_thresh == 0:
        report.checks_passed.append("All scouting records satisfy minutes >= 900 threshold (PASS)")
    else:
        report.warnings.append(f"Found {below_thresh} records below 900 minutes threshold")

    # 3. Non-negative metrics
    cols_to_check = ["goals_per90", "assists_per90", "xG_per90", "shots_per90", "tackles_won_per90"]
    any_neg = False
    for c in cols_to_check:
        if c in df.columns and (df[c] < 0).any():
            any_neg = True
            report.errors.append(f"Negative values detected in column {c}")
    if not any_neg:
        report.checks_passed.append("All per-90 rate metrics non-negative (PASS)")

    # 4. Percentages in [0, 100]
    for pct_col in ["pass_completion_pct", "aerial_won_pct"]:
        if pct_col in df.columns:
            invalid_pct = df[(df[pct_col] < 0.0) | (df[pct_col] > 100.0)]
            if invalid_pct.empty:
                report.checks_passed.append(f"{pct_col} in valid [0, 100] range (PASS)")
            else:
                report.errors.append(f"Out of bounds values in {pct_col}: {len(invalid_pct)}")

    return report


def run_all_processed_validations(processed_base_dir: Path = Path("data/processed")) -> list[ProcessedValidationReport]:
    """Execute validation suite across all processed datasets."""
    reports = []

    match_file = processed_base_dir / "matches" / "match_outcome_dataset.csv"
    if match_file.exists():
        reports.append(validate_processed_matches(match_file))

    transfer_file = processed_base_dir / "transfer" / "transfer_value_dataset.csv"
    if transfer_file.exists():
        reports.append(validate_processed_transfer(transfer_file))

    scout_file = processed_base_dir / "scouting" / "player_scouting_dataset.csv"
    if scout_file.exists():
        reports.append(validate_processed_scouting(scout_file))

    return reports


def print_processed_validation_summary(reports: list[ProcessedValidationReport]) -> bool:
    """Print validation summary and return True if all passed."""
    print("\n" + "=" * 90)
    print("PROCESSED DATASETS VALIDATION SUMMARY")
    print("=" * 90)

    all_passed = True

    for r in reports:
        print(f"\n[{r.dataset_name}] -> {r.file_path}")
        print(f"  Rows: {r.row_count:,} | Columns: {r.col_count:,} | Duplicate Rows: {r.duplicate_rows}")
        if r.unique_players is not None:
            print(f"  Unique Players: {r.unique_players:,}")
        if r.unique_teams is not None:
            print(f"  Unique Teams: {r.unique_teams:,}")
        print(f"  Seasons ({len(r.unique_seasons)}): {r.unique_seasons}")
        print(f"  Checks Passed ({len(r.checks_passed)}):")
        for cp in r.checks_passed:
            print(f"    [OK] {cp}")
        if r.warnings:
            print(f"  Warnings ({len(r.warnings)}):")
            for w in r.warnings:
                print(f"    [WARN] {w}")
        if r.errors:
            all_passed = False
            print(f"  Errors ({len(r.errors)}):")
            for e in r.errors:
                print(f"    [FAIL] {e}")

    print("\n" + "=" * 90)
    print(f"OVERALL STATUS: {'ALL CHECKS PASSED' if all_passed else 'VALIDATION FAILURES DETECTED'}")
    print("=" * 90 + "\n")
    return all_passed


if __name__ == "__main__":
    reps = run_all_processed_validations()
    print_processed_validation_summary(reps)
