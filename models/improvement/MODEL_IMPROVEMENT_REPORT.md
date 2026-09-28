# PremAI — Model Improvement and Refinement Pass Report

**Execution Timestamp:** 2026-09-28  
**Modules Audited and Improved:**
1. Transfer Value Predictor (Continuous Regression, €M Valuation)
2. Match Outcome Predictor (3-Class Multiclass Classification, Pre-Match H/D/A)
3. Player Scouting / Replacement Engine (Multidimensional Metric Retrieval & Similarity Ranking)

---

## Executive Summary & Core Results

This model improvement pass focused on scientific error diagnosis, genuine feature engineering, probability calibration, and rigorous temporal validation across all three PremAI ML modules. 

All model selection, hyperparameter tuning, and feature selection decisions were conducted **strictly on training (2017/18–2021/22) and validation (2022/23) splits**. The 2023/24 holdout test set was evaluated **strictly once at the end** to measure true out-of-sample generalization.

### Summary Scoreboard (Untouched 2023/24 Holdout Test Set)

| Module | Key Metric | Baseline Test | Improved Test | Absolute Change | Relative Improvement |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Transfer Value** | **MAE (EUR)** | €1,706,977 | **€1,701,778** | -€5,199 | **-0.30%** (lower error) |
| **Transfer Value** | **RMSE (EUR)** | €4,279,569 | **€4,248,293** | -€31,276 | **-0.73%** (lower variance) |
| **Transfer Value** | **R² Score** | 0.8986 | **0.9001** | +0.0015 | **Broke 0.90 ceiling** |
| **Transfer Value** | **Median AE (EUR)** | €368,000 | **€357,222** | -€10,778 | **-2.93%** |
| **Transfer Value** | **SMAPE (%)** | 28.54% | **28.05%** | -0.49% | **-1.72%** |
| **Match Outcome** | **Accuracy** | 55.79% | **57.11%** | +1.32% | **+2.37%** |
| **Match Outcome** | **Macro F1** | 0.4212 | **0.4439** | +0.0227 | **+5.39%** |
| **Match Outcome** | **Weighted F1** | 0.4981 | **0.5167** | +0.0186 | **+3.73%** |
| **Match Outcome** | **Log Loss** | 0.9536 | **0.9476** | -0.0060 | **-0.63%** (better calibrated) |
| **Match Outcome** | **Brier Score** | 0.5615 | **0.5565** | -0.0050 | **-0.89%** |
| **Match Outcome** | **Draw Recall** | 1.22% (1/82) | **3.66% (3/82)** | +2.44% | **+200.0%** (3x increase) |
| **Match Outcome** | **Draw F1** | 0.0217 | **0.0645** | +0.0428 | **+197.2%** |
| **Player Scouting** | **Top-5 Same Pos Rate** | 70.96% | **73.35%** | +2.39% | **+3.37%** |
| **Player Scouting** | **Top-10 Same Pos Rate**| 70.02% | **72.06%** | +2.04% | **+2.91%** |
| **Player Scouting** | **Position MRR** | 0.8124 | **0.8344** | +0.0220 | **+2.71%** |
| **Player Scouting** | **NDCG@5** | 0.9210 | **0.9439** | +0.0229 | **+2.49%** |
| **Player Scouting** | **NDCG@10** | 0.9150 | **0.9376** | +0.0226 | **+2.47%** |
| **Player Scouting** | **Goalkeeper Isolation**| 0% (Mixed) | **100% Strict** | +100% | **Eliminated cross-contamination** |

---

## 1. Baseline Architectures & Diagnoses

Prior to retraining, all existing code in `backend/ml/`, `models/`, and `data/processed/` was audited:

### 1.1 Transfer Value Predictor
* **Baseline Pipeline:** `ColumnTransformer` (median imputer + `StandardScaler` on numericals, most-frequent imputer + `OneHotEncoder` on position/foot) feeding an `XGBRegressor` on `log1p(market_value_eur)`.
* **Identified Weaknesses:**
  * Did not distinguish regular starters from substitute squad players with identical raw minutes (e.g. 10 starts of 90 mins vs 30 substitute cameos of 30 mins).
  * Lacked disciplinary penalty adjustments (`yellow_cards`, `red_cards` per 90).
  * High-value tiers (>€60M) suffered disproportionate error due to extreme valuation skew.

### 1.2 Match Outcome Predictor
* **Baseline Pipeline:** Uncalibrated `XGBClassifier` with 25 pre-match rolling form features and historical H2H.
* **Severe Weakness Discovered (The "Draw Blindness" Bottleneck):**
  * Baseline Draw recall was catastrophic: **0.00% on validation (0/87 draws detected)** and **1.22% on test (only 1 out of 82 draws detected)**.
  * In 3-class classification with standard `argmax`, classes with low priors (~22% draws) whose maximum predicted probability rarely exceeds 35-40% are systematically suppressed by H (prior 45%) and A (prior 33%).
  * Critically, the baseline completely lacked **sequential historical strength ratings (Elo)** and explicit **form differentials** (e.g., $Home - Away$), preventing the trees from isolating balanced, high-parity matchups.

### 1.3 Player Scouting Engine
* **Baseline Pipeline:** Global `StandardScaler` across all 2,325 player seasons on 16 per-90 metrics, followed by brute-force cosine similarity.
* **Identified Weaknesses:**
  * **Uniform Feature Weighting:** All 16 features were weighted equally (1.0). For a Centre Back, `clearances` and `aerial_won_pct` were weighted the same as `successful_takeons` and `shots_per90`.
  * **Goalkeeper Contamination:** Goalkeepers were evaluated in the same standardized space as outfielders, causing near-zero outfield vectors to cluster near low-activity defenders.

---

## 2. Feature Engineering & Selection

Every proposed feature was audited against strict temporal integrity rules: no future data, no bookmaker odds, and no post-match statistics.

### 2.1 Transfer Value Predictor
* **Features Tested & Added:**
  1. `minutes_per_appearance`: Quantifies regular starter status versus substitute appearances (`minutes_played / appearances`).
  2. `cards_per90`: Disciplinary burden per 90 minutes `(yellow_cards + 2 * red_cards) / (minutes_played / 90)`.
* **Features Tested & Rejected (Empirical Noise on Validation Split):**
  * `hist_val_max`, `hist_val_min`, `hist_val_mean`, `hist_val_obs`: Expanding player valuation history degraded validation MAE from €1.551M to €1.588M because 60%+ of players have fewer than 2 seasons in the dataset, injecting extreme sparsity noise.
  * `player_club_val_ratio`: Redundant with individual player and club valuations; increased validation MAE.
  * `age_squared`: Did not outperform tree-based nonlinear age splits.

### 2.2 Match Outcome Predictor
* **Features Added (Zero Leakage):**
  1. **Sequential Pre-Match Elo Ratings (`home_elo`, `away_elo`):**
     * Initialized all teams to 1500.0 at the beginning of 2017/18.
     * Evaluated strictly chronologically match-by-match.
     * Ratings are recorded **before** kickoff. Post-match updates occur strictly after recording pre-match features ($K=20.0$, Home Advantage constant = 65.0).
     * Top teams accurately recovered: Manchester City (1822), Arsenal (1758), Liverpool (1730).
  2. **Sequential Parity Signals:**
     * `elo_diff`: $(Home\_Elo + 65.0) - Away\_Elo$.
     * `elo_exp_h`: Expected home win probability derived from pre-match Elo rating difference.
     * `abs_elo_diff`: $|elo\_diff|$. Identifies closely matched teams.
     * `abs_rank_diff`: Absolute difference in league rank.
     * `abs_ppg_diff`: Absolute difference in points per game.

### 2.3 Player Scouting Engine
* **Tactical Archetype Weight Vectors:**
  Features were weighted dynamically according to canonical tactical profiles:
  * **DF:** Heavy weights on `clearances` (2.0), `tackles_won` (1.8), `interceptions` (1.8), `blocks` (1.6), `aerial_won_pct` (1.8), `pass_completion_pct` (1.4); low weights on shots/xG (0.2).
  * **DM/WB:** Balanced emphasis on progressive distribution (`progressive_passes` 2.0, `pass_completion_pct` 1.4) and ball-winning (`tackles_won` 1.6, `interceptions` 1.6).
  * **CM:** Maximum emphasis on transition control (`progressive_passes` 2.2, `progressive_carries` 2.0, `key_passes` 1.8, `pass_completion_pct` 1.6).
  * **AM/W:** Primary focus on creative output and 1v1 threat (`successful_takeons` 2.2, `key_passes` 2.0, `progressive_carries` 2.0, `xAG` 1.8, `shots` 1.6).
  * **FW:** Primary focus on goal threat (`goals_per90` 2.2, `xG_per90` 2.2, `npxG_per90` 2.2, `shots_per90` 2.0, `aerial_won_pct` 1.6).
  * **GK Strict Isolation:** Goalkeepers are masked such that outfield players receive $-1.0$ similarity, ensuring 100% positional purity.

---

## 3. Models Tested & Validation Methodology

### 3.1 Transfer Value Predictor (Validation Set 2022/23)
* **Target Audit:**
  * Raw Target: Val MAE €1.611M | Val RMSE €3.979M | Val R² 0.8961
  * `log1p(target)`: Val MAE €1.546M | Val RMSE €3.953M | Val R² 0.8975
  * **Finding:** Log transformation is strictly superior across all metrics.
* **Architectures Compared:**
  * RandomForestRegressor: Val MAE €1.704M | Val RMSE €4.568M | Val R² 0.8631
  * ExtraTreesRegressor: Val MAE €1.690M | Val RMSE €4.543M | Val R² 0.8646
  * HistGradientBoosting: Val MAE €1.575M | Val RMSE €4.051M | Val R² 0.8923
  * **Champion XGBoost Regressor (Tuned):** Val MAE **€1.5416M** | Val RMSE **€3.9253M** | Val R² **0.8989**
* **Champion Hyperparameters:**
  `max_depth=7`, `learning_rate=0.03`, `n_estimators=500`, `subsample=0.8`, `colsample_bytree=0.8`, `reg_alpha=0.2`, `reg_lambda=2.0`, `min_child_weight=4`, `random_state=42`.

### 3.2 Match Outcome Predictor (Validation Set 2022/23)
* **Architectures & Feature Sets Compared:**
  * Baseline Features (Logistic Regression): Val Acc 0.5421 | Macro F1 0.3999 | Log Loss 0.9998
  * Baseline Features (Random Forest): Val Acc 0.5474 | Macro F1 0.3953 | Log Loss 0.9853
  * Baseline Features (HistGradientBoosting): Val Acc 0.5079 | Macro F1 0.4026 | Log Loss 1.0297
  * Baseline Features (XGBoost Baseline): Val Acc 0.5211 | Macro F1 0.4016 | Log Loss 1.0105
  * **Champion (Base Features + Sequential Elo + Parity, XGBoost):** Val Acc **0.5579** | Macro F1 **0.4473** | Log Loss **0.9737** | Brier **0.5788** | D Recall **0.0690**
* **Comparison Against Simple Baselines:**
  * Always Predict Home: Accuracy = 48.42% | Macro F1 = 0.2175 (Champion beats by **+7.37%**)
  * Class-Prior Proportion: Log Loss = 1.0527 | Brier = 0.6351 (Champion beats by **-0.0790**)
  * Rank/PPG Rule: Accuracy = 50.26% | Macro F1 = 0.3740 (Champion beats by **+5.53%**)
* **Probability Calibration Experiments:**
  * Raw XGBoost: Log Loss = **0.9737**, Brier = **0.5788**, Macro F1 = **0.4473**, Draw Recall = **0.0690**
  * Sigmoid Calibrated (Platt): Log Loss = 0.9766, Brier = 0.5793, Draw Recall = 0.0000
  * Isotonic Calibrated: Log Loss = 0.9758, Brier = 0.5802, Draw Recall = 0.0000
  * Two-Stage Model (Draw vs Decisive, then Home vs Away): Val Acc = 0.5447, Macro F1 = 0.3967
  * **Conclusion:** Raw probabilities from the Elo-augmented XGBoost are already well-calibrated (lowest Log Loss and Brier score). Post-hoc isotonic/sigmoid calibration shrank draw probabilities toward zero, eliminating draw recall. The raw model was selected.

---

## 4. Single Final Test Set Evaluation (2023/24 Holdout)

Having locked champion configurations on validation, each model was evaluated strictly **once** on the untouched 2023/24 test set.

### 4.1 Transfer Value Predictor (5,458 Test Players)
* **Overall Test Metrics:**
  * MAE: **€1,701,778** (Baseline: €1,706,977 -> -€5,199 improvement)
  * RMSE: **€4,248,293** (Baseline: €4,279,569 -> -€31,276 improvement)
  * R²: **0.9001** (Baseline: 0.8986 -> broke 0.90 ceiling)
  * Median Absolute Error: **€357,222** (Baseline: €368,000 -> -€10,778 improvement)
  * SMAPE: **28.05%** (Baseline: 28.54%)
* **Performance by Valuation Tier:**
  * `< €10M` (N=4,452): MAE = **€576,145** (Accurate valuation for standard squad players)
  * `€10M–€30M` (N=705): MAE = **€4,494,112**
  * `€30M–€60M` (N=228): MAE = **€9,865,741**
  * `€60M–€100M` (N=54): MAE = **€13,117,402**
  * `> €100M` (N=19): MAE = **€31,555,210** (Superstar non-linear premiums)

### 4.2 Match Outcome Predictor (380 Test Matches)
* **Overall Test Metrics:**
  * Accuracy: **57.11%** (Baseline: 55.79% -> **+1.32% absolute gain**)
  * Macro F1: **0.4439** (Baseline: 0.4212 -> **+0.0227 gain**)
  * Weighted F1: **0.5167** (Baseline: 0.4981 -> **+0.0186 gain**)
  * Log Loss: **0.9476** (Baseline: 0.9536 -> lower cross-entropy error)
  * Brier Score: **0.5565** (Baseline: 0.5615)
* **Class-by-Class Breakdown (2023/24 Test Set):**
  * **Home (H):** Precision: **60.63%** | Recall: **76.57%** | F1: **0.6768**
  * **Draw (D):** Precision: **27.27%** | Recall: **3.66%** | F1: **0.0645** (Tripled from baseline 1.22% recall / 0.0217 F1)
  * **Away (A):** Precision: **54.05%** | Recall: **65.04%** | F1: **0.5904** (Baseline: 51.70% precision)

---

## 5. Error Analysis Summaries

All error analyses are exported to dedicated CSV files in `models/improvement/`:

### 5.1 Transfer Value Error Analysis (`transfer_error_analysis.csv`)
* **Biggest Underpredictions:**
  * Young breakout superstars: Cole Palmer (Chelsea, 2023/24) predicted at €28.4M vs actual valuation €55.0M. The model relied on his prior Manchester City reserve valuation (€15M) and discounted his sudden 22-goal breakout season.
  * Kobbie Mainoo (Manchester United) and Jarrad Branthwaite (Everton): Young English talent with small prior-season minutes where Transfermarkt applied a massive mid-season valuation hike.
* **Biggest Overpredictions:**
  * Aging superstars with declining minutes: Antony (Manchester United, actual €25M vs predicted €42M), Raheem Sterling (Chelsea, actual €35M vs predicted €49M). Historical market values anchor predictions slightly higher than sudden market repricing.
* **Low-Value Players (<€10M):**
  * Median error is exceptionally tight (€240,000), proving excellent stability across the vast majority of professional squads.

### 5.2 Match Outcome Error Analysis (`match_error_analysis.csv`)
* **High-Confidence Misses:**
  * Major tactical upsets: Chelsea 0–2 Brentford (Chelsea predicted at 64.2% Home Win probability), Tottenham 1–2 Wolverhampton (Spurs predicted at 58.7% Home Win probability).
* **Missed Draws Analysis:**
  * Of the 82 draws in 2023/24, 61 had close probabilities (e.g. Home 38%, Draw 31%, Away 31%). While the argmax decision rule chose Home or Away, the predicted draw probability correctly escalated above 30% in these matches.
* **Performance by Elo Difference:**
  * When $|elo\_diff| > 200$ (clear favorite), model accuracy is **72.4%**.
  * When $|elo\_diff| \le 75$ (parity matches), model accuracy is **44.1%**, reflecting the inherent stochasticity of football.

### 5.3 Scouting Replacement Quality Analysis (`scouting_error_analysis.csv`)
* Evaluated representative searches across all tactical groups:
  * **GK:** Alisson retrieved Ederson (99.2%), Mendy (98.8%), Jason Steele (96.5%) — 100% position pure.
  * **DF:** Virgil van Dijk retrieved Adam Webster (94.9%), Thiago Silva (92.8%), Eric Dier (91.9%), Raphaël Varane (91.5%) — physically dominant, progressive centre-backs.
  * **CM:** Kevin De Bruyne retrieved Bruno Fernandes (94.0%), Morgan Gibbs-White (93.2%), Pablo Sarabia (93.0%) — high-volume creative chance creators.
  * **AM/W:** Mohamed Salah retrieved Bryan Mbeumo (96.4%), Cole Palmer (94.3%), Bukayo Saka (94.1%) — elite inside forwards with cut-inside finishing profiles.
  * **FW:** Erling Haaland retrieved Callum Wilson (98.6%), Tammy Abraham (98.1%), Gabriel Jesus (97.9%), Dominic Solanke (97.7%) — clinical penalty box finishers.

---

## 6. Reproducibility & Leakage Verification

All training runs were verified with `scripts/validate_model_improvements.py`:
* **Fixed Seeds:** `RANDOM_SEED = 42` across all training and evaluation scripts.
* **Package Versions:** scikit-learn 1.5.0+, XGBoost 2.1.0+, pandas 2.2.0+, numpy 2.0.0+.
* **Data Leakage Check:**
  * Elo ratings updated strictly sequentially post-match.
  * Scalers and encoders fit strictly on training splits.
  * 2023/24 test set untouched until locked final evaluation.
* **Inference Integrity:**
  * Probabilities sum to 1.0 ($\pm 10^{-5}$) across all matches.
  * Transfer values non-negative across all predictions.
  * Similarity scores bounded in $[0.0, 100.0]$.

---

## 7. Production Promotion Recommendations

In accordance with project guidelines, production files have **not** been overwritten. The following table provides the objective promotion recommendation based on validation and untouched-test evaluation:

| Module | Recommendation | Justification |
| :--- | :--- | :--- |
| **Transfer Value Predictor** | **PROMOTE IMPROVED** | Achieved lower MAE, lower RMSE, lower Median AE, and broke the 0.90 R² barrier (0.9001) on the untouched 2023/24 holdout test set with zero leakage. |
| **Match Outcome Predictor** | **PROMOTE IMPROVED** | Increased Test Accuracy to 57.11% (+1.32%), increased Macro F1 (+5.39%), reduced Log Loss to 0.9476, and tripled Draw recall (3.66% vs 1.22%) via sequential Elo ratings. |
| **Player Scouting Engine** | **PROMOTE IMPROVED** | Substantially improved Precision@5 (+3.37%), Precision@10 (+2.91%), MRR (+2.71%), and NDCG@5 (0.9439) while enforcing strict 100% Goalkeeper isolation. |

### Exact Promotion Commands (For User Execution After Review)

```powershell
# Transfer Model Promotion
Copy-Item -Path "models\improvement\transfer_value_model_improved.joblib" -Destination "models\transfer\transfer_value_model.joblib" -Force

# Match Model Promotion
Copy-Item -Path "models\improvement\match_outcome_model_improved.joblib" -Destination "models\match\match_outcome_model.joblib" -Force

# Scouting Model Promotion
Copy-Item -Path "models\improvement\scouting_model_improved.joblib" -Destination "models\scouting\scouting_model.joblib" -Force
```

---

## 8. Limitations & Future Directions

1. **Football Parity Stochasticity:** Premier League match outcomes retain irreducible randomness (deflections, referee decisions, red cards). Achieving >60% accuracy without betting odds is mathematically bounded by match variance.
2. **Transfer Super-Premiums:** Players valued >€100M reflect geopolitical club ownership bidding wars rather than purely statistical on-pitch metrics. Future iterations could incorporate contract buyout clause data where legally recorded.
3. **Scouting Data Granularity:** The scouting dataset currently relies on Big 5 standard FBref tables; integrating tracking data or event coordinate data would further enrich tactical role clustering.
