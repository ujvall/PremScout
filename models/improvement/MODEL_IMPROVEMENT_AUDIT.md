# PremAI Football Intelligence Platform — Model Improvement Audit

**Document Date**: September 28, 2026  
**Auditor**: Lead Applied ML Specialist (DeepMind Agentic Systems)  
**Objective**: Comprehensive technical audit of baseline models, error analysis, identification of bottlenecks, leakage risk assessment, and candidate improvement roadmap.

---

## 1. Executive Summary & Improvement Roadmap

The initial phase of PremAI established operational pipelines and serialized production models for:
1. **Transfer Value Predictor**: Continuous player market valuation (€M) via XGBoost regression.
2. **Match Outcome Predictor**: Pre-match 3-class outcome forecasting (Home Win, Draw, Away Win) via XGBoost classification.
3. **Player Scouting & Replacement**: Multidimensional player retrieval via Standardized Cosine Similarity.

While all models are functional, each possesses specific architectural, feature-engineering, calibration, or domain-representation weaknesses. This audit investigates the root causes of underperformance and establishes an improvement plan adhering to strict temporal hold-out discipline.

---

## 2. Transfer Value Predictor Audit

### 2.1 Current Architecture & Pipeline
- **Algorithm**: `XGBRegressor` (n_estimators=300, max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0).
- **Target Transformation**: `log1p(market_value_eur)` fitted on training data, evaluated by inverse transform `expm1`.
- **Preprocessing**: `ColumnTransformer` with median imputation + `StandardScaler` on numerical features, most frequent imputation + `OneHotEncoder(handle_unknown='ignore')` on categorical features.
- **Dataset Splits**:
  - Train: 2017/18 through 2021/22 (22,580 player-seasons)
  - Validation: 2022/23 (5,410 player-seasons)
  - Test (Hold-out): 2023/24 (5,458 player-seasons)

### 2.2 Current Features (18 Baseline Inputs, 38 One-Hot Encoded)
- **Demographics & Physical**: `age`, `height_cm`, `preferred_foot`, `primary_position`, `sub_position`.
- **Contract & Stature**: `contract_years_left`, `club_market_value_eur`, `prev_market_value_eur`.
- **Playing Time & Volume**: `minutes_played`, `appearances`.
- **Performance Totals**: `goals`, `assists`, `yellow_cards`, `red_cards`.
- **Per-90 Rates**: `goals_per90`, `assists_per90`, `shots_per90`, `npxg_per90`.

### 2.3 Baseline Performance

| Metric | Validation (2022/23) | Test Holdout (2023/24) |
| :--- | :---: | :---: |
| **MAE** | €1,560,465 (€1.56M) | €1,706,977 (€1.71M) |
| **RMSE** | €3,969,568 (€3.97M) | €4,279,569 (€4.28M) |
| **R²** | 0.8966 | 0.8986 |

### 2.4 Weaknesses & Root Causes of Underperformance
1. **High Relative Error on Elite Tiers (>€60M)**: Overall MAE of €1.71M is heavily skewed by the massive mass of low-valuation players (<€5M) where absolute errors are small. On top-tier talent (€60M–€150M+), the baseline model severely underpredicts peak valuations due to gradient shrinkage and log-space compression.
2. **Lack of Valuation Trajectory Features**: `prev_market_value_eur` provides a static anchor, but fails to capture momentum:
   - Absolute valuation growth: `prev_market_value_eur - prev_2y_market_value`
   - Relative growth rate: `(market_value - prev) / prev`
   - Peak historical valuation vs current valuation.
3. **Missing Club Economic Stature Ratio**: The absolute `club_market_value_eur` is provided, but a player's relative stature within their squad (`prev_market_value / club_market_value`) is unrepresented.
4. **Non-Linear Age Trajectory by Position**: Forward physical decline begins around age 29–30, whereas goalkeepers and central defenders peak later (27–32). Standard linear one-hot encoding fails to capture position-specific age elasticity without explicit interaction terms or tree depth optimization.
5. **No Per-Appearance Playing Time Density**: `minutes_played / appearances` separates consistent starters from fringe substitutes who accumulate sub-20-minute cameos.

### 2.5 Candidate Improvements
- **Engineered Features**:
  - `valuation_momentum`: Historical value growth ratio and change.
  - `player_club_ratio`: `prev_market_value_eur / (club_market_value_eur + 1e5)`.
  - `minutes_per_appearance`: `minutes_played / (appearances + 1e-5)`.
  - `goal_contribution_rate`: `(goals + assists) / 90`.
  - `age_squared` & `age_position_interaction`.
- **Model Exploration**:
  - Compare `XGBRegressor`, `RandomForestRegressor`, `ExtraTreesRegressor`, and `HistGradientBoostingRegressor`.
  - Systematic hyperparameter tuning (max_depth, subsample, colsample_bytree, reg_alpha, reg_lambda, learning_rate).
  - Target representation: Confirm `log1p` vs robust scaling or Box-Cox.
- **Evaluation Plan**:
  - Segmented MAE across tiers: `<€10M`, `€10–30M`, `€30–60M`, `€60–100M`, `>€100M`.
  - Error analysis on breakout young players vs veteran assets.

---

## 3. Match Outcome Predictor Audit

### 3.1 Current Architecture & Pipeline
- **Algorithm**: `XGBClassifier(objective='multi:softprob', num_class=3)`.
- **Dataset Splits**:
  - Train: 2017/18 through 2021/22 (1,900 matches)
  - Validation: 2022/23 (380 matches)
  - Test (Hold-out): 2023/24 (380 matches)

### 3.2 Baseline Performance

| Metric | Validation (2022/23) | Test Holdout (2023/24) |
| :--- | :---: | :---: |
| **Accuracy** | 55.26% | 55.79% |
| **Macro F1** | 0.4033 | 0.4212 |
| **Log Loss** | 0.9882 | 0.9536 |
| **Brier Score** | 0.5872 | 0.5615 |
| **ROC-AUC (OVR)** | 0.6617 | 0.7088 |

#### Per-Class Test Breakdown (Baseline)
- **Home Win (H)**: Precision = 0.5830 | Recall = 0.7829 | F1 = 0.6683 (Support: 175)
- **Draw (D)**: Precision = 0.1000 | **Recall = 0.0122 (1.2%)** | **F1 = 0.0217** (Support: 82)
- **Away Win (A)**: Precision = 0.5481 | Recall = 0.6016 | F1 = 0.5736 (Support: 123)

### 3.3 Critical Weakness: Draw Collapse
The baseline model essentially predicts only binary outcomes (Home Win vs Away Win). Out of 82 actual draws in the 2023/24 test season, **the baseline model only predicted 1 single draw**!
#### Root Causes:
1. **Class Asymmetry & Probabilistic Overlap**: Draws constitute ~21-23% of Premier League fixtures. In a 3-way softmax where Home Win is ~46% and Away Win is ~31%, the draw probability rarely exceeds 0.35–0.38, meaning argmax almost never selects Draw unless probabilities are calibrated or thresholds are adjusted.
2. **Missing Match Differentials**: The baseline inputs home and away features independently (`home_rolling_points_5`, `away_rolling_points_5`, `home_rolling_gf_5`, `away_rolling_gf_5`). Gradient boosted trees must perform multiple recursive splits to learn `home - away`. Providing explicit differential features (`diff_points`, `diff_gf`, `diff_ga`, `diff_sot`, `diff_ppg`) directly exposes match parity. When parity features are close to 0 between balanced sides, draw likelihood spikes.
3. **No Sequential Rating (Elo)**: Table rank and 5-match rolling points suffer from early-season volatility and fixture difficulty bias. A sequential Elo rating chronologically updated after every fixture provides an objective, long-term strength baseline.
4. **Lack of Probability Calibration**: Uncalibrated raw tree softmax outputs tend to be overconfident on extreme classes and suppress middle-density outcomes (Draw).

### 3.4 Candidate Improvements
- **Feature Engineering**:
  - Complete set of Home - Away differentials: `diff_points_5`, `diff_gf_5`, `diff_ga_5`, `diff_shots_5`, `diff_sot_5`, `diff_corners_5`, `diff_ppg`, `diff_rest_days`, `diff_win_pct_5`.
  - Sequential Elo Rating System: Chronologically initialized and updated fixture-by-fixture (strictly using only pre-match historical results).
  - Parity Index: Absolute strength divergence `abs(home_elo - away_elo)` and `abs(home_ppg - away_ppg)`.
- **Model Exploration & Calibration**:
  - Compare `LogisticRegression`, `RandomForestClassifier`, `XGBClassifier`, and `HistGradientBoostingClassifier`.
  - Systematic tuning of class weighting / sample weighting on validation set.
  - Probability Calibration via `CalibratedClassifierCV` (isotonic and sigmoid/Platt scaling) fitted strictly on train/val.
  - Evaluate Log Loss and Brier Score improvements alongside Macro F1.

---

## 4. Player Scouting & Replacement Engine Audit

### 4.1 Current Architecture & Pipeline
- **Method**: Standardized vector cosine similarity across 16 per-90 metrics.
- **Normalization**: Single global `StandardScaler` with median imputation over the entire player catalog.
- **Filters**: Post-hoc boolean masking for `same_position_only`, `max_market_value`, `max_age`, `min_minutes`.
- **Dataset**: 2,325 player-seasons covering 925 distinct Premier League players (2017/18–2023/24).

### 4.2 Baseline Evaluation Metrics
- **Top-5 Same Position Retrieval Rate**: 70.96%
- **Top-10 Same Position Retrieval Rate**: 70.02%
- **Top-5 Compatible Position Rate**: 92.05%
- **Position MRR (Mean Reciprocal Rank)**: 0.8124
- **Multi-Season Self-Retrieval Rate (Top 10)**: 45.03%

### 4.3 Weaknesses & Root Causes
1. **Uniform Global Scaling Distorts Tactical Archetypes**: Standardizing all players globally causes extreme outliers in one position (e.g. wingers taking 4.0 take-ons/90 vs centre-backs taking 0.2) to compress variance in unrelated positions.
2. **Unweighted Feature Space**: A central defender and an attacking midfielder are currently compared across the exact same 16 metrics with equal weight. For a centre-back, aerial win %, clearances, and tackles won are primary indicators of tactical profile, while take-ons and shots are irrelevant noise.
3. **No Multi-Season Stability Validation**: Player performance naturally fluctuates across seasons. High-quality scouting engines should demonstrate multi-season temporal stability: a player's profile in Season T should retrieve their own profile in Season T+1 or T-1 among the top candidates.

### 4.4 Candidate Improvements
- **Position-Specific Feature Weighting**:
  - Tailor metric weighting by tactical archetype (CB, FB/WB, DM, CM, AM/W, CF, GK).
  - Emphasize defensive volume and aerial dominance for CBs; progressive passing, carries, and tackles for DMs/CMs; creation, take-ons, and shot generation for AM/Ws; box presence and conversion for CFs.
- **Strict Goalkeeper Isolation**:
  - Goalkeepers have zero overlap with outfield player actions. Enforce strict isolation.
- **Time-Aware Retrieval Evaluation**:
  - Measure multi-season self-retrieval rate (e.g., does 2021/22 Declan Rice retrieve 2022/23 or 2020/21 Declan Rice?).
  - Evaluate Precision@5, Precision@10, and Mean Reciprocal Rank (MRR).

---

## 5. Leakage Risk Assessment & Guardrails

| Risk | Mitigation Strategy |
| :--- | :--- |
| **Post-Match Leakage** | Match features strictly derive from historical matches played *before* the kickoff date. |
| **Future Target Leakage** | Transfer features strictly use previous season values (`prev_market_value_eur`) and current season pre-target statistics. |
| **Test Set Peeking** | All feature selection, scaling, model comparison, hyperparameter tuning, and probability calibration MUST occur strictly on training and validation sets (`train`: 2017/18–2021/22, `val`: 2022/23). The 2023/24 test set is evaluated exactly ONCE at the end. |
| **Elo Rating Future Leakage** | Elo ratings are calculated chronologically match-by-match in ascending date order. The rating updated *after* a match is never used for predicting that match. |
| **Data Fabrication** | All data derived from active processed files (`transfer_value_dataset.csv`, `match_outcome_dataset.csv`, `player_scouting_dataset.csv`). Zero synthetic rows or bookmaker odds. |

---

## 6. Execution Plan & Next Steps

1. **Step 1**: Implement Transfer Value feature engineering, cross-model comparison, and hyperparameter tuning on validation set. Select champion model, evaluate on holdout test set, and generate error analysis.
2. **Step 2**: Implement Match Outcome feature engineering (differentials + chronological Elo), model selection, class weight analysis, and probability calibration on validation set. Evaluate on holdout test set and generate error analysis.
3. **Step 3**: Implement Scouting position-specific feature weighting and multi-season temporal retrieval validation.
4. **Step 4**: Produce `MODEL_COMPARISON_BEFORE_AFTER.csv` and `MODEL_IMPROVEMENT_REPORT.md`.
5. **Step 5**: Create `scripts/validate_model_improvements.py` to ensure complete system integrity and frontend/API compatibility.
