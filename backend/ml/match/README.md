# PremAI: Match Outcome Predictor

The **Match Outcome Predictor** module predicts the probabilities and most likely outcome of a Premier League football match (**Home Win**, **Draw**, **Away Win**) using strictly pre-match historical information.

---

## 1. Problem Formulation

- **Target:** `match_result`
  - `H`: Home Win (Class 0)
  - `D`: Draw (Class 1)
  - `A`: Away Win (Class 2)
- **Task Type:** 3-Class Multiclass Probabilistic Classification
- **Outputs:**
  - `home_win_probability` $\in [0, 1]$
  - `draw_probability` $\in [0, 1]$
  - `away_win_probability` $\in [0, 1]$
  - `predicted_result`: "Home Win", "Draw", or "Away Win"

---

## 2. Features (25 Pre-Kickoff Indicators)

The model relies strictly on signals available before referee kickoff:
1. **Rest & Congestion:** `home_rest_days`, `away_rest_days`
2. **Rolling 5-Match Form (Points & Goals):** `home_rolling_points_5`, `away_rolling_points_5`, `home_rolling_gf_5`, `away_rolling_gf_5`, `home_rolling_ga_5`, `away_rolling_ga_5`, `home_rolling_win_pct_5`, `away_rolling_win_pct_5`
3. **Underlying Rolling Dominance (Shots & Corners):** `home_rolling_shots_5`, `away_rolling_shots_5`, `home_rolling_sot_5`, `away_rolling_sot_5`, `home_rolling_sot_conceded_5`, `away_rolling_sot_conceded_5`, `home_rolling_corners_5`, `away_rolling_corners_5`
4. **League Standings Context:** `home_league_rank`, `away_league_rank`, `rank_diff`, `home_ppg`, `away_ppg`
5. **Head-to-Head History:** `h2h_matches_prior`, `h2h_home_win_pct`

**Strict Leakage Prevention:** Same-match final scores, live match statistics, goals, shots, fouls, and betting market odds are strictly excluded.

---

## 3. Chronological Splits

- **Train:** 2017/18 through 2021/22 (1,900 matches, 5 seasons)
- **Validation:** 2022/23 (380 matches, 1 season) — used exclusively for model selection
- **Test:** 2023/24 (380 matches, 1 season) — held out untouched until final evaluation

---

## 4. Pipeline Execution & Usage

### Training the Model:
```bash
python -m backend.ml.match.train
```

### Python Inference:
```python
from backend.ml.match.predict import predict_match

sample_fixture = {
    "home_rest_days": 7,
    "away_rest_days": 4,
    "home_rolling_points_5": 12.0,
    "away_rolling_points_5": 4.0,
    "home_rolling_gf_5": 13.0,
    "away_rolling_gf_5": 4.0,
    "home_rolling_ga_5": 3.0,
    "away_rolling_ga_5": 9.0,
    "home_rolling_shots_5": 18.2,
    "away_rolling_shots_5": 9.4,
    "home_rolling_sot_5": 6.8,
    "away_rolling_sot_5": 3.0,
    "home_rolling_sot_conceded_5": 2.2,
    "away_rolling_sot_conceded_5": 6.0,
    "home_rolling_corners_5": 7.4,
    "away_rolling_corners_5": 3.2,
    "home_rolling_win_pct_5": 0.8,
    "away_rolling_win_pct_5": 0.2,
    "home_league_rank": 2,
    "away_league_rank": 16,
    "rank_diff": -14,
    "home_ppg": 2.35,
    "away_ppg": 0.85,
    "h2h_matches_prior": 10,
    "h2h_home_win_pct": 0.70,
}

prediction = predict_match(sample_fixture)
print(prediction)
# Output:
# {
#   "predicted_result": "Home Win",
#   "predicted_code": "H",
#   "home_win_probability": 0.6842,
#   "draw_probability": 0.2015,
#   "away_win_probability": 0.1143,
#   "model_type": "XGBoost"
# }
```
