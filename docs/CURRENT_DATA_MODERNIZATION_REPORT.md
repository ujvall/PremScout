# PremAI Platform — Data Modernization Engineering Report

**Execution Timestamp:** 2026-09-28  
**Reference As-Of Date:** 2026-09-20  
**Status:** ALL PRODUCTION VALIDATION CRITERIA SATISFIED (100% PASS)  
**Strict Modeling Rule Enforced:** Existing validated production models (`Transfer: V2`, `Match: V1-Improved`, `Scouting: V2`) were consumed directly. **Zero models were retrained, no weights were modified, and historical training datasets were preserved completely intact.**

---

## Executive Summary

During product quality assurance, a data freshness boundary was identified: while the three core machine learning models were validated and promoted to production, the underlying search and feature pipelines were constrained by historical datasets ending at the 2023/24 season boundary. Consequently, players transferred after 2024 (e.g., Rayan Cherki) were missing from player discovery, and established stars displayed stale historical metadata (e.g., Erling Haaland displayed as "Manchester City • 2023-24 • Age 23").

This Data Modernization Pass successfully implemented a **Current Data Layer** (`data/current/`) that provides authentic, up-to-date football intelligence through September 2026, while preserving all historical datasets (`data/raw/`, `data/processed/`) and maintaining the integrity of the promoted production models.

---

## 1. Old Data Coverage vs. New Data Coverage

| Dimension | Old Historical Pipeline (Preserved) | New Current Data Layer (`data/current/`) |
| :--- | :--- | :--- |
| **Historical Range** | 2017/18 – 2023/24 | **2017/18 – 2026/27 (Active Matchweek 5)** |
| **Data Boundary Date** | May 19, 2024 | **September 20, 2026** |
| **Player Roster Coverage** | Frozen at 2023/24 Premier League squads | **29,653 unique players across Premier League & European Big 5** |
| **Scouting Performance Profiles** | 2,828 player-seasons ending in 2023/24 | **3,169 performance profiles from 2025/26 season** |
| **Premier League Match History** | 2,660 matches (2017/18 through 2023/24) | **3,470 chronological matches (adding 2024/25, 2025/26, 2026/27)** |
| **Sequential Team Elo** | Frozen at May 2024 ratings | **Continuously updated through September 20, 2026 fixtures** |
| **Player Valuations** | Historical Transfermarkt valuations (2024) | **Valuations through July 2026 (e.g. Haaland €200M, Cherki €90M)** |

> [!IMPORTANT]
> **Separation of Concerns:**  
> - **Historical Data (`data/raw/`, `data/processed/`):** Retained 100% untouched for model research, backtesting, and reproducibility.  
> - **Current Inference Data (`data/current/`):** Dedicated production inference layer supplying current vectors to validated models.

---

## 2. Data Sources & Retrieval Methodology

All data ingested into `data/current/` originates from authoritative football intelligence sources without synthetic fabrication. Documented formally in [`data/current/metadata/DATA_SOURCES.md`](file:///c:/Users/ujval/Desktop/Projects/PremAI/data/current/metadata/DATA_SOURCES.md):

1. **Transfermarkt Player & Valuation Database:**
   - **Scope:** Player demographics, contract expiration dates, tactical sub-positions, club affiliations, and market valuations.
   - **Maximum Data Date:** 2026-07-06 (Appearances) / 2026-06-12 (Valuations).
   - **Retrieval Date:** 2026-09-28.
   - **Coverage:** Complete Premier League rosters and active European first-tier players.

2. **FBref / Sports Reference Advanced Statistics:**
   - **Scope:** Complete 2025/26 seasonal per-90 metrics across Standard, Shooting, Passing, Possession, Defense, and Miscellaneous modules.
   - **Maximum Data Date:** 2026-05-24 (End of 2025/26 European league campaigns).
   - **Season End Year:** 2026 (2025/26 season).
   - **Coverage:** 3,169 advanced profiles across European Big 5 leagues.

3. **Football-Data.co.uk Match Results:**
   - **Scope:** Premier League fixtures, half-time and full-time scores, shots, shots on target, and corner kicks.
   - **Seasons Ingested:** 2024/25 (380 matches), 2025/26 (380 matches), 2026/27 (50 matches through Matchweek 5).
   - **Maximum Data Date:** 2026-09-20.
   - **Sequential Elo Integrity:** Strict chronological sorting prior to feature computation. Zero future leakage.

---

## 3. Current Player Master Dataset

Created at [`data/current/players/current_players.csv`](file:///c:/Users/ujval/Desktop/Projects/PremAI/data/current/players/current_players.csv) containing 29,653 deduplicated, normalized player profiles.

### Schema:
`player_id, player_name, normalized_name, date_of_birth, age, club, club_id, position, sub_position, nationality, season, market_value_eur, data_as_of, source, has_scouting_data`

### Normalization Logic:
Robust Unicode normalizer handling:
- Accents & diacritics (`Martínez` → `martinez`, `Lúcio` → `lucio`)
- Soft hyphens (`\xad`) and non-breaking spaces (`\xa0`)
- Punctuation & apostrophes (`N'Golo` → `ngolo`, `van Dijk` → `van dijk`)
- Whitespace stripping and lowercase unification.

### Test Targets Verification:

| Player Name | Current Club | Current Age (as of 2026-09-20) | Active Season | Market Valuation | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Rayan Cherki** | Manchester City | **23** | **2025-26** | **€90.0M** | **PASS** |
| **Erling Haaland** | Manchester City | **26** | **2025-26** | **€200.0M** | **PASS** |
| **Bukayo Saka** | Arsenal | **25** | **2025-26** | **€110.0M** | **PASS** |
| **Declan Rice** | Arsenal | **27** | **2025-26** | **€120.0M** | **PASS** |
| **Virgil van Dijk** | Liverpool | **35** | **2025-26** | **€15.0M** | **PASS** |
| **Emiliano Martínez** | Aston Villa | **34** | **2025-26** | **€12.0M** | **PASS** |

---

## 4. Current Scouting Dataset & Feature Compatibility

Created at [`data/current/scouting/current_player_scouting.csv`](file:///c:/Users/ujval/Desktop/Projects/PremAI/data/current/scouting/current_player_scouting.csv) containing 3,169 seasonal profiles from the 2025/26 campaign.

### Feature Compatibility Audit:
The 16 core features expected by the production V2 scouting model (`models/scouting/scouting_model.joblib`) were compared against the 2025/26 FBref dataset:

| Feature Name | V2 Model Definition | 2025/26 Current Data Source | Compatibility Assessment |
| :--- | :--- | :--- | :--- |
| `goals_per90` | Goals scored / 90 mins | FBref Standard `Gls / 90` | **Identical (100%)** |
| `assists_per90` | Assists / 90 mins | FBref Standard `Ast / 90` | **Identical (100%)** |
| `xG_per90` | Expected goals / 90 mins | FBref Standard `xG / 90` | **Identical (100%)** |
| `npxG_per90` | Non-penalty xG / 90 mins | FBref Standard `npxG / 90` | **Identical (100%)** |
| `xAG_per90` | Expected assisted goals / 90 | FBref Standard `xAG / 90` | **Identical (100%)** |
| `shots_per90` | Shots attempted / 90 mins | FBref Shooting `Sh / 90` | **Identical (100%)** |
| `key_passes_per90` | Key passes / 90 mins | FBref Passing `KP / 90` | **Identical (100%)** |
| `pass_completion_pct` | Pass accuracy percentage | FBref Passing `Cmp%` | **Identical (100%)** |
| `progressive_passes_per90` | Passes moving ball 10m+ forward | FBref Passing `PrgP / 90` | **Identical (100%)** |
| `progressive_carries_per90` | Carries moving ball 10m+ forward | FBref Possession `PrgC / 90` | **Identical (100%)** |
| `successful_takeons_per90` | Dribbles completed / 90 mins | FBref Possession `Succ / 90` | **Identical (100%)** |
| `tackles_won_per90` | Tackles won / 90 mins | FBref Defense `TklW / 90` | **Identical (100%)** |
| `interceptions_per90` | Interceptions / 90 mins | FBref Defense `Int / 90` | **Identical (100%)** |
| `blocks_per90` | Shots + passes blocked / 90 | FBref Defense `Blocks / 90` | **Identical (100%)** |
| `clearances_per90` | Clearances / 90 mins | FBref Defense `Clr / 90` | **Identical (100%)** |
| `aerial_won_pct` | Aerial duels won percentage | FBref Misc `Won%` | **Identical (100%)** |

**Conclusion:** The feature space is 100% mathematically and semantically compatible. Current 2025/26 vectors are standardized via the fitted V2 `StandardScaler` and projected using V2 archetype position weights without retraining or modifying weights.

---

## 5. Verification of Key Stars

### Rayan Cherki Verification:
- **Player Master Record:** Rayan Cherki, player_id `607223`, Club: Manchester City, Age: 23, Season: 2025-26, Market Value: €90,000,000, Data as of: 2026-09-20.
- **Scouting Profile:** Present in 2025/26 FBref statistics (1.27 goals/90, 7.61 progressive passes/90).
- **V2 Similarity Engine Output:** Successfully projected into V2 vector space. Top statistical matches:
  1. Roberto Firmino (Liverpool) — 67.7%
  2. Andriy Yarmolenko (West Ham United) — 67.0%
  3. Son Heung-min (Tottenham Hotspur) — 64.3%
- **Transfer Valuation Output:** Production V2 Model predicted market valuation: **€58.66M**.

### Erling Haaland Verification:
- **Player Master Record:** Erling Haaland, player_id `418560`, Club: Manchester City, Age: 26, Season: 2025-26, Market Value: €200,000,000, Data as of: 2026-09-20.
- **Stale Data Resolved:** Completely removed stale "2023-24 • Age 23" label. UI displays current age (26) and latest season (2025-26).
- **V2 Similarity Engine Output:** Top statistical matches:
  1. Gabriel Jesus (Manchester City) — 98.0%
  2. Tammy Abraham (Chelsea) — 97.7%
  3. Pierre-Emerick Aubameyang (Arsenal) — 97.0%
- **Transfer Valuation Output:** Production V2 Model predicted market valuation: **€140.32M**.

---

## 6. Current Match Data & Prediction Service

Created at [`data/current/matches/current_match_history.csv`](file:///c:/Users/ujval/Desktop/Projects/PremAI/data/current/matches/current_match_history.csv) and [`data/current/matches/current_team_states.json`](file:///c:/Users/ujval/Desktop/Projects/PremAI/data/current/matches/current_team_states.json):
- **3,470 total chronological matches** from August 2017 through September 20, 2026.
- **Strict Chronological Elo:** Sequential Elo starts at 1500.0 and updates match-by-match with zero future lookahead.
- Current 2026/27 Elo ratings:
  - Manchester City: **1772.4** (Rank 1, 15 pts, 3.00 PPG)
  - Arsenal: **1774.3** (Rank 2, 12 pts, 2.40 PPG)
  - Liverpool: **1664.0** (Rank 7, 9 pts, 1.80 PPG)
  - Chelsea: **1586.8** (Rank 10, 7 pts, 1.40 PPG)
- **V1-Improved XGBoost + Sequential Elo Model Execution:**
  - Consumes live calculated pre-match features from `current_team_states.json`.
  - For Arsenal vs. Chelsea (Matchweek 6): Predicts **Home Win (74.3% Home, 9.4% Draw, 16.3% Away)**.

---

## 7. Data Freshness Status Metadata

Created at [`data/current/metadata/current_data_status.json`](file:///c:/Users/ujval/Desktop/Projects/PremAI/data/current/metadata/current_data_status.json):
```json
{
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
      "coverage": "Premier League & European Big 5 through 2025/26 season"
    },
    {
      "domain": "FBref / Sports Reference",
      "scope": "Advanced player per-90 metrics (Standard, Shooting, Passing, Possession, Defense, Misc)",
      "max_data_date": "2026-05-24",
      "coverage": "Big 5 European Leagues through 2025/26 season (Season_End_Year 2026)"
    },
    {
      "domain": "Football-Data.co.uk",
      "scope": "Premier League match results, scores, shots, and corner stats",
      "max_data_date": "2026-09-20",
      "coverage": "2024/25 (380 matches), 2025/26 (380 matches), 2026/27 (50 matches through Matchweek 5)"
    }
  ]
}
```

---

## 8. Data Validation Results (`scripts/validate_current_data.py`)

All 25 automated integrity tests passed:
- **0 duplicate player IDs** across 29,653 records.
- **0 missing player names** or clubs.
- **Ages strictly verified** in realistic athletic window (16–45 years).
- **16 core scouting features** have zero negative values.
- **3,470 matches strictly sorted chronologically** with valid Elo range (1307.5 to 1836.2).
- Explicit target players verified (Cherki, Haaland, Saka, Rice, Van Dijk, Martínez).

---

## 9. API & Frontend Verification

1. **API Endpoints:**
   - `GET /api/health` — Returns healthy, `data_as_of: 2026-09-20`, `current_data_layer: true`.
   - `GET /api/data-status` — Serves current freshness metadata.
   - `GET /api/scouting/players?search=...` — Searches 29,653 current players with Unicode normalization.
   - `GET /api/scouting/similar/{player_name}` — Runs current vector through V2 model.
   - `GET /api/transfer/presets` — Dynamically generates presets for Cherki, Haaland, Saka, Rice, etc.
   - `GET /api/transfer/player-features/{player_name}` — Returns 18 features from current player dataset.
   - `GET /api/match/current-teams` — Lists active 2026/27 Premier League teams with current standings and Elo.
   - `GET /api/match/current-fixture-features` — Generates pre-match features from latest 2026/27 team states.
   - **Pytest Suite:** 23 / 23 tests PASSED (`tests/api/test_api.py`).

2. **Frontend UI Integration:**
   - Global Header displays badge: **`Latest available data: 2026-09-20`**.
   - Player Search and Detail cards display: Current Club, Current Age, Latest Season, and **`Data as of: 2026-09-20`**.
   - Transfer page contains dynamic presets populated from current 2025/26 features.
   - Match predictor dropdowns populated with active 2026/27 Premier League clubs and current matchday presets.
   - Existing ML badges preserved: `Transfer V2`, `Match V1-Improved`, `Scouting V2`.
   - Production Build: `tsc -b && vite build` built cleanly with 0 errors.

---

## 10. Known Source Limitations

1. **Mid-Season Boundary:**
   - 2026/27 Premier League season is active through Matchweek 5 (50 matches played as of 2026-09-20).
   - In-season match sample size for newly promoted clubs (e.g., Hull, Leeds) incorporates prior historical form and initial matchday performances.
2. **FBref Per-90 Boundary:**
   - Detailed multi-module per-90 metrics (tackles won, progressive carries, take-ons) reflect the completed 2025/26 season (`Season_End_Year: 2026`). Players who made their senior professional debut only in Autumn 2026 have master profiles in Transfermarkt but are marked `has_scouting_data = False`.
3. **No Live Claim:**
   - In accordance with prompt instructions, the platform explicitly labels data as **"Latest available data: 2026-09-20"** and avoids claiming "live streaming data".
