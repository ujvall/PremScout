# Data Processing & Feature Engineering Service

This package cleans, resolves entities across, and feature-engineers raw football data into three leak-free ML-ready datasets for **PremAI**.

---

## 📁 Modules

| File | Purpose |
| :--- | :--- |
| `entity_resolution.py` | Canonical team mapping, NFKD player name normalization, deterministic ID extraction, and validated fuzzy matching |
| `feature_engineering.py` | Safe per-90 rate metric calculations and chronological train/val/test splitting metadata |
| `clean_matches.py` | Generates leak-free match prediction features (rolling 5-match indicators, dynamic pre-match table ranks, days of rest, H2H) |
| `clean_transfer.py` | Generates player-season transfer value records with strict temporal alignment (`prev_market_value_eur` $\to$ `market_value_eur`) |
| `clean_scouting.py` | Combines 6 FBref statistical tables with Transfermarkt valuations for player scouting and retrieval |
| `validate_processed.py` | Comprehensive verification of leakage rules, target validity, mathematical consistency, and ranges |
| `build_datasets.py` | Master CLI orchestrator to build all three datasets and generate `data/PROCESSED_DATA_REPORT.md` |

---

## 🚀 How to Run the Processing Pipeline

### 1. Build All Processed Datasets
From the project root:

```powershell
.\venv\Scripts\python.exe -m backend.services.data_processing.build_datasets
```

### 2. Run Validation Separately
```powershell
.\venv\Scripts\python.exe -m backend.services.data_processing.validate_processed
```

---

## 📂 Output Datasets

```text
data/
└── processed/
    ├── matches/
    │   └── match_outcome_dataset.csv
    ├── transfer/
    │   └── transfer_value_dataset.csv
    └── scouting/
        ├── player_scouting_dataset.csv         (Filtered: minutes >= 900)
        └── player_scouting_unfiltered.csv      (All player-seasons)
```
