# Data Ingestion Pipeline

This service orchestrates the automated fetching, extraction, and validation of raw football datasets for **PremAI**.

---

## 📁 Ingestion Components

| File | Purpose | Source |
| :--- | :--- | :--- |
| `download_matches.py` | Fetches historical Premier League match CSVs (`E0.csv`) for seasons 2017/18 through 2023/24 | [football-data.co.uk](https://football-data.co.uk/) |
| `download_transfer_data.py` | Fetches and decompresses Transfermarkt relational tables (`players`, `player_valuations`, `transfers`, `appearances`, `clubs`, `games`) | [dcaribou/transfermarkt-datasets](https://github.com/dcaribou/transfermarkt-datasets) |
| `download_scouting_data.py` | Fetches FBref Big 5 player season stats (shooting, passing, possession, defense, misc) and FPL historical records | [worldfootballR_data](https://github.com/JaseZiv/worldfootballR_data) & [vaastav/Fantasy-Premier-League](https://github.com/vaastav/Fantasy-Premier-League) |
| `validate_data.py` | Runs automated integrity, completeness, and boundary validations on all ingested files | Internal Service |
| `run_ingestion.py` | Master CLI runner that coordinates download and validation sequences | Master Runner |

---

## 🚀 How to Run the Ingestion Pipeline

### 1. Run Complete Pipeline
From the project root directory (`PremAI/`):

```powershell
# Using the project virtual environment
.\venv\Scripts\python.exe -m backend.services.data_ingestion.run_ingestion
```

### 2. Run Individual Ingestion Modules
You can also run any module independently:

```powershell
# Ingest match data only
.\venv\Scripts\python.exe -m backend.services.data_ingestion.download_matches

# Ingest Transfermarkt tables only
.\venv\Scripts\python.exe -m backend.services.data_ingestion.download_transfer_data

# Ingest Scouting datasets only
.\venv\Scripts\python.exe -m backend.services.data_ingestion.download_scouting_data

# Run data validation only
.\venv\Scripts\python.exe -m backend.services.data_ingestion.validate_data
```

---

## 📂 Raw Output Directory Layout

```text
data/
└── raw/
    ├── matches/
    │   ├── 2017-18/E0.csv
    │   ├── 2018-19/E0.csv
    │   ├── 2019-20/E0.csv
    │   ├── 2020-21/E0.csv
    │   ├── 2021-22/E0.csv
    │   ├── 2022-23/E0.csv
    │   └── 2023-24/E0.csv
    ├── transfermarkt/
    │   ├── players.csv
    │   ├── player_valuations.csv
    │   ├── transfers.csv
    │   ├── appearances.csv
    │   ├── clubs.csv
    │   └── games.csv
    └── scouting/
        ├── big5_player_standard.csv
        ├── big5_player_shooting.csv
        ├── big5_player_passing.csv
        ├── big5_player_possession.csv
        ├── big5_player_defense.csv
        ├── big5_player_misc.csv
        └── fpl_2017-18.csv ... fpl_2023-24.csv
```
