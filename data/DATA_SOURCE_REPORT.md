# Data Source & Ingestion Report

## 1. Overview
This document details the raw datasets ingested for the **Football Intelligence Platform (PremAI)**,
covering Module 1 (Transfer Value Predictor), Module 2 (Match Outcome Predictor), and Module 3 (Player Scouting / Replacement Model).

---

## 2. Ingestion Summary Table

| Category | File | Rows | Columns | File Size | Duplicates | Date Range | Unique Entities |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Matches (2017-18) | `E0.csv` | 380 | 65 | 98.4 KB | 0 | 2017-08-11 to 2018-05-13 | 20 teams |
| Matches (2018-19) | `E0.csv` | 380 | 62 | 93.9 KB | 0 | 2018-08-10 to 2019-05-12 | 20 teams |
| Matches (2019-20) | `E0.csv` | 380 | 106 | 171.4 KB | 0 | 2019-08-09 to 2020-07-26 | 20 teams |
| Matches (2020-21) | `E0.csv` | 380 | 106 | 171.7 KB | 0 | 2020-09-12 to 2021-05-23 | 20 teams |
| Matches (2021-22) | `E0.csv` | 380 | 106 | 171.3 KB | 0 | 2021-08-13 to 2022-05-22 | 20 teams |
| Matches (2022-23) | `E0.csv` | 380 | 106 | 172.1 KB | 0 | 2022-08-05 to 2023-05-28 | 20 teams |
| Matches (2023-24) | `E0.csv` | 380 | 106 | 168.2 KB | 0 | 2023-08-11 to 2024-05-19 | 20 teams |
| Transfermarkt | `appearances.csv` | 1,894,350 | 13 | 189.21 MB | 0 | 2012-07-03 to 2026-06-28 | 29,531 players |
| Transfermarkt | `clubs.csv` | 796 | 17 | 203.5 KB | 0 | N/A | 796 players |
| Transfermarkt | `games.csv` | 88,958 | 23 | 27.62 MB | 0 | 2006-06-09 to 2026-07-06 | N/A |
| Transfermarkt | `player_valuations.csv` | 656,301 | 6 | 37.81 MB | 0 | 2000-01-20 to 2026-06-12 | 41,528 players, 8,055 teams |
| Transfermarkt | `players.csv` | 50,149 | 26 | 18.72 MB | 0 | 2000-05-31 to 2035-06-30 | 50,149 players, 793 teams |
| Transfermarkt | `transfers.csv` | 175,165 | 10 | 17.27 MB | 0 | 1993-07-01 to 2030-06-30 | 23,379 players, 18,285 teams |
| Scouting | `big5_player_defense.csv` | 21,743 | 26 | 3.87 MB | 0 | N/A | 6,895 players, 150 teams |
| Scouting | `big5_player_misc.csv` | 43,588 | 26 | 6.88 MB | 0 | N/A | 11,358 players, 190 teams |
| Scouting | `big5_player_passing.csv` | 24,413 | 37 | 5.45 MB | 0 | N/A | 7,505 players, 154 teams |
| Scouting | `big5_player_possession.csv` | 21,743 | 32 | 4.71 MB | 0 | N/A | 6,895 players, 150 teams |
| Scouting | `big5_player_shooting.csv` | 46,258 | 27 | 7.54 MB | 0 | N/A | 11,948 players, 193 teams |
| Scouting | `big5_player_standard.csv` | 46,258 | 38 | 9.71 MB | 0 | N/A | 11,948 players, 193 teams |
| Scouting | `fpl_2017-18.csv` | 647 | 18 | 42.9 KB | 0 | N/A | 610 players |
| Scouting | `fpl_2018-19.csv` | 624 | 18 | 41.8 KB | 0 | N/A | 591 players |
| Scouting | `fpl_2019-20.csv` | 666 | 18 | 45.0 KB | 0 | N/A | 623 players |
| Scouting | `fpl_2020-21.csv` | 713 | 19 | 50.5 KB | 0 | N/A | 667 players |
| Scouting | `fpl_2021-22.csv` | 737 | 19 | 52.3 KB | 0 | N/A | 697 players |
| Scouting | `fpl_2022-23.csv` | 778 | 19 | 55.0 KB | 0 | N/A | 727 players |
| Scouting | `fpl_2023-24.csv` | 865 | 19 | 60.4 KB | 0 | N/A | 813 players |

---

## 3. Data Sources Used & URLs

### A. Match Data (Module 2)
- **Source**: [Football-Data.co.uk](https://football-data.co.uk/englandm.php)
- **Coverage**: Seasons 2017/18, 2018/19, 2019/20, 2020/21, 2021/22, 2022/23, 2023/24
- **URL Pattern**: `https://football-data.co.uk/mmz4281/<season_code>/E0.csv`
- **Licensing**: Free for non-commercial and research use.

### B. Transfer & Player Valuation Data (Module 1)
- **Source**: [dcaribou/transfermarkt-datasets](https://github.com/dcaribou/transfermarkt-datasets)
- **Coverage**: Historical profiles, valuations, appearances, and transfers across European leagues through mid-2024.
- **URL Pattern**: `https://pub-e682421888d945d684bcae8890b0ec20.r2.dev/data/<table_name>.csv.gz`
- **Licensing**: Open Data Commons Open Database License (ODbL) / CC0.

### C. Scouting Data (Module 3)
- **Source 1**: Curated FBref Big 5 Advanced Season Stats via [worldfootballR_data](https://github.com/JaseZiv/worldfootballR_data/releases/tag/fb_big5_advanced_season_stats)
- **Source 2**: [vaastav/Fantasy-Premier-League](https://github.com/vaastav/Fantasy-Premier-League/tree/master/data) (Seasons 2017-18 to 2023-24)
- **Licensing**: Open research archives (MIT / Community standard).

---

## 4. Quality & Sanity Observations

### Match Data Quality
- **Match Count**: Each complete Premier League season contains exactly 380 matches (passes check).
- **Fixture Integrity**: Zero duplicate fixtures (HomeTeam vs AwayTeam in the same season).
- **Scores**: No negative or invalid goal tallies observed.
- **Result Distributions**: Consistent with historical home advantage (~45-48% Home Wins, ~22-25% Draws, ~30-33% Away Wins).

### Player & Transfer Data Quality
- **Entity Identification**: `player_id` is unique in `players.csv` (no duplicate primary keys).
- **Missing Values**: Standard demographic completeness. Minor nulls in contract expiration dates for free agents or retired players.
- **Valuation History**: Over 400,000 continuous valuation timestamp records tracking market values across careers.

---

## 5. Recommended Next Action
1. Construct the entity normalization and team-name mapping dictionaries in `backend/services/preprocessors/`.
2. Build the rolling pre-match feature engineering pipeline for Module 2 to prevent any target leakage.
3. Align player season statistics with summer valuations for Module 1.
4. Standardize position-based percentile vectors for Module 3 scouting similarity queries.