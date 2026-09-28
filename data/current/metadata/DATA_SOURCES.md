# PremAI Football Intelligence Platform — Current Data Layer Documentation

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