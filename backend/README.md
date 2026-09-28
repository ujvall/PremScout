# PremAI: FastAPI Backend Integration

The **PremAI Backend** is a high-performance RESTful API built with **FastAPI** that serves inference pipelines for three specialized football machine learning modules:
1. **Transfer Value Predictor**: Continuous player transfer valuation regression.
2. **Match Outcome Predictor**: Pre-match 3-class probabilistic outcome forecasting.
3. **Player Scouting & Replacement**: Vector similarity retrieval with position-aware recruitment filters.

---

## 1. Architecture Overview

The backend is organized into a clean, layered architecture separating routes, validation schemas, and business services from the underlying ML model artifacts:

```
PremAI/
├── backend/
│   ├── main.py                     # FastAPI application entrypoint & middleware configuration
│   ├── api/
│   │   ├── routes/
│   │   │   ├── health.py           # Health check endpoint actively verifying ML model availability
│   │   │   ├── transfer.py         # Transfer valuation endpoints (predict & model-info)
│   │   │   ├── match.py            # Match outcome forecasting endpoints (predict & model-info)
│   │   │   └── scouting.py         # Player similarity, replacement search & catalog endpoints
│   │   └── schemas/
│   │       ├── common.py           # Standardized envelopes & health responses
│   │       ├── transfer.py         # Pydantic schemas for transfer predictor
│   │       ├── match.py            # Pydantic schemas for match predictor
│   │       └── scouting.py         # Pydantic schemas for scouting engine
│   ├── services/
│   │   ├── transfer_service.py     # Business logic & inference wrapper for transfer model
│   │   ├── match_service.py        # Business logic & probability normalization for match model
│   │   └── scouting_service.py     # In-memory candidate search & similarity filtering
│   └── ml/
│       ├── transfer/               # Serialized transfer model pipeline & preprocessor
│       ├── match/                  # Serialized match outcome classifier & preprocessor
│       └── scouting/               # Serialized vector embeddings & similarity engine
└── tests/
    └── api/
        └── test_api.py             # Integration test suite using FastAPI TestClient
```

---

## 2. Environment Setup & Starting the Backend

### Prerequisites
- Python 3.11+
- Virtual environment activated (`.\venv\Scripts\activate`)

### Running the API Server:
From the project root directory, run:
```bash
python -m uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
```

Once running:
- **Root Service Discovery:** `http://127.0.0.1:8000/`
- **Interactive Swagger UI:** `http://127.0.0.1:8000/docs`
- **Alternative ReDoc:** `http://127.0.0.1:8000/redoc`
- **OpenAPI JSON Schema:** `http://127.0.0.1:8000/openapi.json`

---

## 3. Model Loading & In-Memory Caching

- Model artifacts are stored in `models/` as serialized `.joblib` files.
- Each service layer (`transfer_service`, `match_service`, `scouting_service`) initializes and caches the preloaded model in memory upon first request. Subsequent requests execute inference with zero disk-read overhead.
- The health check endpoint (`GET /api/health`) actively confirms the loadability of all three model artifacts rather than hardcoding status flags.

---

## 4. CORS Configuration

CORS middleware is configured in `backend/main.py` to allow local development with React/Vite frontends:
- `http://localhost:3000` (Create React App default)
- `http://localhost:5173` (Vite default)
- `http://127.0.0.1:3000`
- `http://127.0.0.1:5173`

---

## 5. Endpoints Reference

### A. Root & Health Check
| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/` | API status, title, version, and documentation link. |
| `GET` | `/api/health` | Active verification of all 3 model artifacts. |

### B. Transfer Value Predictor
| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `POST` | `/api/transfer/predict` | Predict player transfer valuation in EUR from stats. |
| `GET` | `/api/transfer/model-info` | Training dataset size, feature list, and evaluation metrics. |

### C. Match Outcome Predictor
| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `POST` | `/api/match/predict` | Forecast Home Win, Draw, Away Win probabilities. |
| `GET` | `/api/match/model-info` | Training seasons, feature list, and test accuracy/log loss. |

### D. Player Scouting & Replacement
| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/api/scouting/similar/{player_name}` | Retrieve top N statistically similar players. |
| `POST` | `/api/scouting/replacements` | Find candidates under budget, age, and position filters. |
| `GET` | `/api/scouting/players` | Lightweight player catalog for search dropdowns. |
| `GET` | `/api/scouting/model-info` | Feature list, position mapping, and retrieval metrics. |

---

## 6. Example Requests & Responses

### 1. Transfer Value Prediction
**Request (`POST /api/transfer/predict`):**
```json
{
  "player_name": "Declan Rice",
  "age": 25.0,
  "height_cm": 188.0,
  "contract_years_left": 4.0,
  "club_market_value_eur": 1040000000.0,
  "minutes_played": 3200.0,
  "appearances": 38.0,
  "goals": 7.0,
  "assists": 8.0,
  "yellow_cards": 5.0,
  "red_cards": 0.0,
  "goals_per90": 0.20,
  "assists_per90": 0.23,
  "shots_per90": 1.35,
  "npxg_per90": 0.15,
  "prev_market_value_eur": 90000000.0,
  "primary_position": "Midfield",
  "sub_position": "Defensive Midfield",
  "preferred_foot": "Right"
}
```

**Response (`200 OK`):**
```json
{
  "success": true,
  "player_name": "Declan Rice",
  "predicted_market_value_eur": 87450000.0,
  "predicted_market_value_formatted": "€87.45M",
  "model_type": "XGBoost",
  "target_transformation": "log1p"
}
```

---

### 2. Match Outcome Prediction
**Request (`POST /api/match/predict`):**
```json
{
  "home_team": "Arsenal",
  "away_team": "Chelsea",
  "home_rest_days": 7.0,
  "away_rest_days": 4.0,
  "home_rolling_points_5": 12.0,
  "away_rolling_points_5": 7.0,
  "home_rolling_gf_5": 11.0,
  "away_rolling_gf_5": 6.0,
  "home_rolling_ga_5": 3.0,
  "away_rolling_ga_5": 8.0,
  "home_rolling_shots_5": 16.4,
  "away_rolling_shots_5": 12.2,
  "home_rolling_sot_5": 6.2,
  "away_rolling_sot_5": 4.0,
  "home_rolling_sot_conceded_5": 2.6,
  "away_rolling_sot_conceded_5": 5.4,
  "home_rolling_corners_5": 7.2,
  "away_rolling_corners_5": 4.6,
  "home_rolling_win_pct_5": 0.80,
  "away_rolling_win_pct_5": 0.40,
  "home_league_rank": 2,
  "away_league_rank": 9,
  "home_ppg": 2.25,
  "away_ppg": 1.45,
  "h2h_matches_prior": 12,
  "h2h_home_win_pct": 0.58
}
```

**Response (`200 OK`):**
```json
{
  "success": true,
  "home_team": "Arsenal",
  "away_team": "Chelsea",
  "predicted_outcome": "H",
  "predicted_outcome_label": "Home Win",
  "probabilities": {
    "home_win": 0.5982,
    "draw": 0.2145,
    "away_win": 0.1873
  },
  "model_type": "XGBoost"
}
```

---

### 3. Replacement Search Under Scouting Constraints
**Request (`POST /api/scouting/replacements`):**
```json
{
  "player_name": "Declan Rice",
  "top_n": 3,
  "same_position_only": true,
  "max_market_value": 50000000.0,
  "max_age": 26,
  "min_minutes": 1000
}
```

**Response (`200 OK`):**
```json
[
  {
    "target_player": "Declan Rice",
    "target_team": "Arsenal",
    "target_position": "CM",
    "target_season": "2023-24",
    "candidate_player": "Douglas Luiz",
    "candidate_club": "Aston Villa",
    "candidate_position": "CM",
    "candidate_raw_position": "MF",
    "candidate_season": "2022-23",
    "similarity_score": 92.9,
    "cosine_similarity": 0.9287,
    "position_compatibility": "Same Position",
    "age": 24,
    "market_value_eur": 40000000.0,
    "minutes": 2927,
    "key_stats": {
      "goals_per90": 0.18,
      "assists_per90": 0.18,
      "npxG_per90": 0.11,
      "xAG_per90": 0.16,
      "key_passes_per90": 1.63,
      "pass_completion_pct": 85.2,
      "progressive_passes_per90": 1.94,
      "progressive_carries_per90": 1.48,
      "tackles_won_per90": 1.23,
      "interceptions_per90": 1.14
    }
  },
  {
    "target_player": "Declan Rice",
    "target_team": "Arsenal",
    "target_position": "CM",
    "target_season": "2023-24",
    "candidate_player": "Pape Matar Sarr",
    "candidate_club": "Tottenham Hotspur",
    "candidate_position": "CM",
    "candidate_raw_position": "MF",
    "candidate_season": "2023-24",
    "similarity_score": 92.1,
    "cosine_similarity": 0.9209,
    "position_compatibility": "Same Position",
    "age": 21,
    "market_value_eur": 45000000.0,
    "minutes": 2069,
    "key_stats": {
      "goals_per90": 0.13,
      "assists_per90": 0.13,
      "npxG_per90": 0.15,
      "xAG_per90": 0.13,
      "key_passes_per90": 1.13,
      "pass_completion_pct": 87.4,
      "progressive_passes_per90": 1.96,
      "progressive_carries_per90": 1.57,
      "tackles_won_per90": 0.91,
      "interceptions_per90": 0.78
    }
  }
]
```

---

## 7. Running the Test Suite

Run all backend API integration tests via `pytest`:
```bash
python -m pytest tests/api/test_api.py -v
```
All 14 tests validate root routing, active model health checks, transfer inference, match probability calibration, vector similarity retrieval, error handling (404 and 422), and player catalog search.
