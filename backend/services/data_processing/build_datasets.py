"""Master pipeline runner for PremAI processed datasets.

Coordinates:
1. Match outcome dataset generation (Leakage-free, chronological)
2. Transfer value dataset generation (Temporal alignment, previous valuations)
3. Player scouting dataset generation (FBref + Transfermarkt entity resolution)
4. Comprehensive validation on all generated datasets
5. Generation of data/PROCESSED_DATA_REPORT.md
"""

from pathlib import Path
import sys
import time

from .clean_matches import build_match_outcome_dataset
from .clean_scouting import build_player_scouting_dataset
from .clean_transfer import build_transfer_value_dataset
from .validate_processed import (
    print_processed_validation_summary,
    run_all_processed_validations,
)


def write_processed_data_report(
    match_df,
    transfer_df,
    scout_df,
    scout_unfiltered_df,
    validation_reports,
    report_path: Path = Path("data/PROCESSED_DATA_REPORT.md"),
) -> None:
    """Generate the comprehensive markdown data processing report."""
    match_seasons = sorted(match_df["season"].unique().tolist())
    transfer_seasons = sorted(transfer_df["season"].unique().tolist())
    scout_seasons = sorted(scout_df["season"].unique().tolist())

    # Entity match stats
    scout_matched = scout_df["tm_player_id"].notna().sum()
    scout_total = len(scout_df)
    scout_pct = (scout_matched / scout_total * 100) if scout_total > 0 else 0.0

    lines = [
        "# Processed Datasets Report",
        "",
        "## 1. Executive Summary",
        "This document details the three clean, ML-ready datasets produced during **Step 4** of the",
        "**Football Intelligence Platform (PremAI)** project. All datasets were constructed under strict",
        "temporal ordering rules to prevent data leakage and ensure cross-source entity alignment.",
        "",
        "---",
        "",
        "## 2. Processed Datasets Overview",
        "",
        "| Module | Processed Dataset File | Rows | Columns | Seasons | Unique Entities | Target Variable |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
        f"| **Match Outcome** | `data/processed/matches/match_outcome_dataset.csv` | {len(match_df):,} | {len(match_df.columns)} | {len(match_seasons)} ({match_seasons[0]} to {match_seasons[-1]}) | {match_df['home_team'].nunique()} teams | `match_result` (H/D/A) |",
        f"| **Transfer Value** | `data/processed/transfer/transfer_value_dataset.csv` | {len(transfer_df):,} | {len(transfer_df.columns)} | {len(transfer_seasons)} ({transfer_seasons[0]} to {transfer_seasons[-1]}) | {transfer_df['player_id'].nunique():,} players | `market_value_eur` (continuous) |",
        f"| **Player Scouting** | `data/processed/scouting/player_scouting_dataset.csv` | {len(scout_df):,} | {len(scout_df.columns)} | {len(scout_seasons)} ({scout_seasons[0]} to {scout_seasons[-1]}) | {scout_df['norm_name'].nunique():,} players | Feature space (Retrieval/KNN) |",
        "",
        "---",
        "",
        "## 3. Module Details & Feature Engineering",
        "",
        "### A. Match Outcome Dataset",
        "- **Target**: `match_result` (`H`: Home Win, `D`: Draw, `A`: Away Win).",
        "- **Leakage Prevention**: All features for match $T$ are calculated using strictly matches played *prior* to date $T$. Same-match post-kickoff stats (goals, shots, cards) are strictly excluded.",
        "- **Bookmaker Odds**: Intentionally omitted from the primary ML dataset so the model learns fundamental sporting mechanics.",
        "- **Features Created**:",
        "  - `home_rest_days`, `away_rest_days`: Days between matches.",
        "  - `home_rolling_points_5`, `away_rolling_points_5`: Points average in last 5 matches.",
        "  - `home_rolling_gf_5`, `away_rolling_gf_5`: Goals scored average in last 5 matches.",
        "  - `home_rolling_ga_5`, `away_rolling_ga_5`: Goals conceded average in last 5 matches.",
        "  - `home_rolling_shots_5`, `away_rolling_shots_5`: Shots average in last 5 matches.",
        "  - `home_rolling_sot_5`, `away_rolling_sot_5`: Shots on target average in last 5 matches.",
        "  - `home_rolling_sot_conceded_5`, `away_rolling_sot_conceded_5`: Shots on target allowed.",
        "  - `home_rolling_corners_5`, `away_rolling_corners_5`: Corner kicks average in last 5 matches.",
        "  - `home_rolling_win_pct_5`, `away_rolling_win_pct_5`: Win rate in last 5 matches.",
        "  - `home_league_rank`, `away_league_rank`: Dynamic in-season standings rank before kickoff.",
        "  - `rank_diff`: Differential in pre-match standings (`away_rank - home_rank`).",
        "  - `home_ppg`, `away_ppg`: Current season points per game prior to kickoff.",
        "  - `h2h_matches_prior`, `h2h_home_win_pct`: Historical head-to-head record before kickoff.",
        "",
        "### B. Transfer Value Dataset",
        "- **Target**: `market_value_eur` (valuation at the end of season $S$, summer update).",
        "- **Lag Feature**: `prev_market_value_eur` (valuation at the start of season $S$, prior summer).",
        "- **Filtering Rule**: Minimum 270 competitive minutes in season to exclude inactive reserve players.",
        "- **Demographic & Contract Features**:",
        "  - `age`: Exact age at season kickoff.",
        "  - `primary_position`, `sub_position`: Standardized positional hierarchy.",
        "  - `height_cm`, `preferred_foot`: Physical profile.",
        "  - `contract_years_left`: Years remaining on player contract.",
        "  - `club_market_value_eur`: Financial tier of the player's club.",
        "- **Performance Features (Season S)**:",
        "  - `minutes_played`, `appearances`, `goals`, `assists`, `yellow_cards`, `red_cards`.",
        "  - `goals_per90`, `assists_per90`, `shots_per90`, `npxg_per90`.",
        "",
        "### C. Player Scouting Dataset",
        "- **Core Purpose**: High-dimensional vector space for player comparison and replacement retrieval.",
        "- **Filtering Rule**: Minimum 900 league minutes (~10 full matches) applied to `player_scouting_dataset.csv` to eliminate small-sample noise. Unfiltered records ({len(scout_unfiltered_df):,} rows) are preserved in `player_scouting_unfiltered.csv`.",
        "- **Features Created**:",
        "  - **Attacking**: `goals_per90`, `assists_per90`, `xG_per90`, `npxG_per90`, `xAG_per90`, `shots_per90`.",
        "  - **Creation**: `key_passes_per90`.",
        "  - **Passing & Progression**: `pass_completion_pct`, `progressive_passes_per90`, `progressive_carries_per90`, `successful_takeons_per90`.",
        "  - **Defending**: `tackles_won_per90`, `interceptions_per90`, `blocks_per90`, `clearances_per90`.",
        "  - **Physical**: `aerial_won_pct`.",
        "  - **Valuation Anchor**: Merged `market_value_eur` from Transfermarkt for budget filtering.",
        "",
        "---",
        "",
        "## 4. Entity Resolution & Matching Statistics",
        "",
        "- **Team Mapping**: A comprehensive dictionary maps all 30 Premier League clubs across football-data, Transfermarkt, FBref, and FPL into single canonical names.",
        "- **Player Resolution Method**:",
        "  1. NFKD Unicode decomposition, accent removal, punctuation stripping, suffix normalization.",
        "  2. Tier 1: Exact `(norm_name, birth_year)` matching.",
        "  3. Tier 2: Exact `(norm_name, canonical_team)` matching.",
        "  4. Tier 3: Token-subset containment + birth year validation.",
        "  5. Tier 4: Rapidfuzz token-sort similarity $\\ge 90$ strictly validated with birth year / team match.",
        f"- **Scouting Matching Results**: **{scout_matched:,} / {scout_total:,} player-seasons ({scout_pct:.1f}%)** matched to Transfermarkt valuations.",
        f"- **Unmatched Players**: {scout_total - scout_matched:,} ({100 - scout_pct:.1f}%) — Kept in dataset with `market_value_eur = NaN` so tactical profiles remain available.",
        "",
        "---",
        "",
        "## 5. Chronological Train / Validation / Test Splitting Plan",
        "",
        "| Split Group | Seasons Included | Purpose | Match Rows | Transfer Rows |",
        "| :--- | :--- | :--- | :--- | :--- |",
        f"| **`train`** | 2017/18 through 2021/22 | Model training | {(match_df['split_group'] == 'train').sum():,} | {(transfer_df['split_group'] == 'train').sum():,} |",
        f"| **`val`** | 2022/23 | Hyperparameter tuning & model selection | {(match_df['split_group'] == 'val').sum():,} | {(transfer_df['split_group'] == 'val').sum():,} |",
        f"| **`test`** | 2023/24 | Final out-of-time evaluation | {(match_df['split_group'] == 'test').sum():,} | {(transfer_df['split_group'] == 'test').sum():,} |",
        "",
        "---",
        "",
        "## 6. Validation Verification",
        "",
    ]

    for vr in validation_reports:
        lines.append(f"### {vr.dataset_name}")
        for cp in vr.checks_passed:
            lines.append(f"- [x] {cp}")
        for w in vr.warnings:
            lines.append(f"- [!] {w}")
        for e in vr.errors:
            lines.append(f"- [ ] **ERROR**: {e}")
        lines.append("")

    lines.extend([
        "---",
        "",
        "## 7. Next Actions",
        "1. Proceed to Model Architecture design.",
        "2. Establish benchmark models (Baseline dummy, ElasticNet/LightGBM for Transfer Value, Logistic Regression for Match Outcome).",
        "3. Implement cosine distance / k-NN similarity retriever for Scouting.",
    ])

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Generated comprehensive report: {report_path}")


def main() -> None:
    """Execute complete data processing pipeline."""
    t0 = time.time()
    print("=" * 90)
    print("STARTING PREMAI DATA PROCESSING & FEATURE ENGINEERING PIPELINE")
    print("=" * 90 + "\n")

    try:
        # Step 1: Match Outcome Dataset
        print("--- [1/4] Building Match Outcome Dataset ---")
        match_df = build_match_outcome_dataset()

        # Step 2: Transfer Value Dataset
        print("\n--- [2/4] Building Transfer Value Dataset ---")
        transfer_df = build_transfer_value_dataset()

        # Step 3: Player Scouting Dataset
        print("\n--- [3/4] Building Player Scouting Dataset ---")
        scout_df, scout_unfiltered_df = build_player_scouting_dataset()

        # Step 4: Validate all processed datasets
        print("\n--- [4/4] Validating All Processed Datasets ---")
        reports = run_all_processed_validations()
        passed = print_processed_validation_summary(reports)

        if not passed:
            print("[CRITICAL WARNING] Some validation checks failed. Inspect summary above.", file=sys.stderr)

        # Step 5: Generate Report
        write_processed_data_report(
            match_df=match_df,
            transfer_df=transfer_df,
            scout_df=scout_df,
            scout_unfiltered_df=scout_unfiltered_df,
            validation_reports=reports,
        )

        elapsed = time.time() - t0
        print(f"\nProcessing pipeline completed successfully in {elapsed:.2f} seconds.")

    except Exception as exc:
        print(f"\n[FATAL ERROR] Ingestion pipeline failed: {exc}", file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
