# Processed Datasets Report

## 1. Executive Summary
This document details the three clean, ML-ready datasets produced during **Step 4** of the
**Football Intelligence Platform (PremAI)** project. All datasets were constructed under strict
temporal ordering rules to prevent data leakage and ensure cross-source entity alignment.

---

## 2. Processed Datasets Overview

| Module | Processed Dataset File | Rows | Columns | Seasons | Unique Entities | Target Variable |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Match Outcome** | `data/processed/matches/match_outcome_dataset.csv` | 2,660 | 32 | 7 (2017-18 to 2023-24) | 30 teams | `match_result` (H/D/A) |
| **Transfer Value** | `data/processed/transfer/transfer_value_dataset.csv` | 33,448 | 24 | 7 (2017-18 to 2023-24) | 12,524 players | `market_value_eur` (continuous) |
| **Player Scouting** | `data/processed/scouting/player_scouting_dataset.csv` | 2,325 | 28 | 7 (2017-18 to 2023-24) | 923 players | Feature space (Retrieval/KNN) |

---

## 3. Module Details & Feature Engineering

### A. Match Outcome Dataset
- **Target**: `match_result` (`H`: Home Win, `D`: Draw, `A`: Away Win).
- **Leakage Prevention**: All features for match $T$ are calculated using strictly matches played *prior* to date $T$. Same-match post-kickoff stats (goals, shots, cards) are strictly excluded.
- **Bookmaker Odds**: Intentionally omitted from the primary ML dataset so the model learns fundamental sporting mechanics.
- **Features Created**:
  - `home_rest_days`, `away_rest_days`: Days between matches.
  - `home_rolling_points_5`, `away_rolling_points_5`: Points average in last 5 matches.
  - `home_rolling_gf_5`, `away_rolling_gf_5`: Goals scored average in last 5 matches.
  - `home_rolling_ga_5`, `away_rolling_ga_5`: Goals conceded average in last 5 matches.
  - `home_rolling_shots_5`, `away_rolling_shots_5`: Shots average in last 5 matches.
  - `home_rolling_sot_5`, `away_rolling_sot_5`: Shots on target average in last 5 matches.
  - `home_rolling_sot_conceded_5`, `away_rolling_sot_conceded_5`: Shots on target allowed.
  - `home_rolling_corners_5`, `away_rolling_corners_5`: Corner kicks average in last 5 matches.
  - `home_rolling_win_pct_5`, `away_rolling_win_pct_5`: Win rate in last 5 matches.
  - `home_league_rank`, `away_league_rank`: Dynamic in-season standings rank before kickoff.
  - `rank_diff`: Differential in pre-match standings (`away_rank - home_rank`).
  - `home_ppg`, `away_ppg`: Current season points per game prior to kickoff.
  - `h2h_matches_prior`, `h2h_home_win_pct`: Historical head-to-head record before kickoff.

### B. Transfer Value Dataset
- **Target**: `market_value_eur` (valuation at the end of season $S$, summer update).
- **Lag Feature**: `prev_market_value_eur` (valuation at the start of season $S$, prior summer).
- **Filtering Rule**: Minimum 270 competitive minutes in season to exclude inactive reserve players.
- **Demographic & Contract Features**:
  - `age`: Exact age at season kickoff.
  - `primary_position`, `sub_position`: Standardized positional hierarchy.
  - `height_cm`, `preferred_foot`: Physical profile.
  - `contract_years_left`: Years remaining on player contract.
  - `club_market_value_eur`: Financial tier of the player's club.
- **Performance Features (Season S)**:
  - `minutes_played`, `appearances`, `goals`, `assists`, `yellow_cards`, `red_cards`.
  - `goals_per90`, `assists_per90`, `shots_per90`, `npxg_per90`.

### C. Player Scouting Dataset
- **Core Purpose**: High-dimensional vector space for player comparison and replacement retrieval.
- **Filtering Rule**: Minimum 900 league minutes (~10 full matches) applied to `player_scouting_dataset.csv` to eliminate small-sample noise. Unfiltered records ({len(scout_unfiltered_df):,} rows) are preserved in `player_scouting_unfiltered.csv`.
- **Features Created**:
  - **Attacking**: `goals_per90`, `assists_per90`, `xG_per90`, `npxG_per90`, `xAG_per90`, `shots_per90`.
  - **Creation**: `key_passes_per90`.
  - **Passing & Progression**: `pass_completion_pct`, `progressive_passes_per90`, `progressive_carries_per90`, `successful_takeons_per90`.
  - **Defending**: `tackles_won_per90`, `interceptions_per90`, `blocks_per90`, `clearances_per90`.
  - **Physical**: `aerial_won_pct`.
  - **Valuation Anchor**: Merged `market_value_eur` from Transfermarkt for budget filtering.

---

## 4. Entity Resolution & Matching Statistics

- **Team Mapping**: A comprehensive dictionary maps all 30 Premier League clubs across football-data, Transfermarkt, FBref, and FPL into single canonical names.
- **Player Resolution Method**:
  1. NFKD Unicode decomposition, accent removal, punctuation stripping, suffix normalization.
  2. Tier 1: Exact `(norm_name, birth_year)` matching.
  3. Tier 2: Exact `(norm_name, canonical_team)` matching.
  4. Tier 3: Token-subset containment + birth year validation.
  5. Tier 4: Rapidfuzz token-sort similarity $\ge 90$ strictly validated with birth year / team match.
- **Scouting Matching Results**: **2,287 / 2,325 player-seasons (98.4%)** matched to Transfermarkt valuations.
- **Unmatched Players**: 38 (1.6%) — Kept in dataset with `market_value_eur = NaN` so tactical profiles remain available.

---

## 5. Chronological Train / Validation / Test Splitting Plan

| Split Group | Seasons Included | Purpose | Match Rows | Transfer Rows |
| :--- | :--- | :--- | :--- | :--- |
| **`train`** | 2017/18 through 2021/22 | Model training | 1,900 | 22,580 |
| **`val`** | 2022/23 | Hyperparameter tuning & model selection | 380 | 5,410 |
| **`test`** | 2023/24 | Final out-of-time evaluation | 380 | 5,458 |

---

## 6. Validation Verification

### Match Outcome Dataset
- [x] No duplicate match IDs (PASS)
- [x] Match targets are strictly H, D, A (PASS)
- [x] Matches strictly sorted chronologically (PASS)
- [x] No post-match stats or bookmaker odds present (PASS)
- [x] Rolling points in valid [0.0, 3.0] interval (PASS)
- [x] League ranks in valid [1, 20] range (PASS)

### Transfer Value Dataset
- [x] No duplicate (player_id, season) pairs (PASS)
- [x] No negative minutes, goals, or assists (PASS)
- [x] All target market values strictly positive (PASS)
- [x] Target is distinct from previous valuation feature (PASS)
- [x] Per-90 metrics mathematically consistent (PASS)

### Player Scouting Dataset
- [x] No duplicate (player, team, season) rows (PASS)
- [x] All scouting records satisfy minutes >= 900 threshold (PASS)
- [x] All per-90 rate metrics non-negative (PASS)
- [x] pass_completion_pct in valid [0, 100] range (PASS)
- [x] aerial_won_pct in valid [0, 100] range (PASS)

---

## 7. Next Actions
1. Proceed to Model Architecture design.
2. Establish benchmark models (Baseline dummy, ElasticNet/LightGBM for Transfer Value, Logistic Regression for Match Outcome).
3. Implement cosine distance / k-NN similarity retriever for Scouting.