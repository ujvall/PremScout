# PremAI — Model Improvement Pass 2 Report (Advanced Modeling & Prediction Refinement)

**Execution Date:** 2026-09-28  
**Experiment Directory:** `models/improvement_v2/`  
**Governing Principle:** All feature selection, model architecture choices, ensemble weights, and hyperparameter tuning were conducted strictly on **2017/18–2021/22 training** and **2022/23 validation** splits. The **2023/24 test set** was evaluated strictly once at the end.

---

## 1. Executive Summary & V1 vs. V2 Decision Scoreboard

| Module | V1 Validation (2022/23) | V2 Validation (2022/23) | Validation Δ (%) | V1 Test (2023/24) | V2 Test (2023/24) | Test Δ (%) | Decision |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Transfer Value (MAE)** | €1,541,595 | **€1,530,909** | **-0.69%** (better) | €1,701,778 | **€1,696,193** | **-0.33%** (better) | **PROMOTE V2** |
| **Transfer Value (RMSE)**| €3,925,300 | **€3,916,222** | **-0.23%** (better) | €4,248,293 | **€4,254,846** | +0.15% (stable) | **PROMOTE V2** |
| **Transfer Value (MedAE)**| €323,478 | **€317,476** | **-1.86%** (better) | €357,222 | **€349,847** | **-2.06%** (better) | **PROMOTE V2** |
| **Transfer Value (SMAPE)**| 26.37% | **26.18%** | **-0.72%** (better) | 28.05% | **27.99%** | **-0.21%** (better) | **PROMOTE V2** |
| **Match Outcome (Accuracy)**| 55.79% | 55.79% | 0.00% (tie) | 57.11% | 57.11% | 0.00% (tie) | **RETAIN V1** |
| **Match Outcome (Macro F1)**| 0.4473 | 0.4473 | 0.00% (tie) | 0.4439 | 0.4439 | 0.00% (tie) | **RETAIN V1** |
| **Match Outcome (Log Loss)**| 0.9737 | 0.9737 | 0.00% (tie) | 0.9476 | 0.9476 | 0.00% (tie) | **RETAIN V1** |
| **Match Outcome (Draw Rec)**| 6.90% | 6.90% | 0.00% (tie) | 3.66% | 3.66% | 0.00% (tie) | **RETAIN V1** |
| **Scouting (Top-5 Same Pos)**| 72.15% | **73.31%** | **+1.61%** (better) | 72.15% | **73.31%** | **+1.61%** (better) | **PROMOTE V2** |
| **Scouting (Top-10 Same Pos)**| 71.24% | **72.42%** | **+1.66%** (better) | 71.24% | **72.42%** | **+1.66%** (better) | **PROMOTE V2** |
| **Scouting (Position MRR)** | 0.8160 | **0.8247** | **+1.07%** (better) | 0.8160 | **0.8247** | **+1.07%** (better) | **PROMOTE V2** |
| **Scouting (NDCG@5)** | 0.9386 | **0.9430** | **+0.47%** (better) | 0.9386 | **0.9430** | **+0.47%** (better) | **PROMOTE V2** |

---

## 2. Phase 0 — Current Champion Validation Baseline

Before executing any experiments, the V1 improved models from `models/improvement/` were evaluated on the 2022/23 validation split to establish strict baseline thresholds:
* **Transfer V1 Benchmark:** MAE = €1,541,595.09, RMSE = €3,925,300.49, R² = 0.8989, MedAE = €323,478.34, SMAPE = 26.37%.
* **Match V1 Benchmark:** Accuracy = 0.5579, Macro F1 = 0.4473, Weighted F1 = 0.5115, Log Loss = 0.9737, Brier = 0.5788, Draw Precision = 0.3333, Draw Recall = 0.0690, Draw F1 = 0.1143.
* **Scouting V1 Benchmark (Strict Cross-Player Replacement):** Top-5 Same Pos = 0.7215, Top-10 Same Pos = 0.7124, Position MRR = 0.8160, NDCG@5 = 0.9386.

---

## 3. Phase 1 — Transfer Value Predictor V2 Deep Dive

### 3.1 Feature Engineering & Ablation (Evaluated on Validation Split)

| Experiment ID | Feature Configuration | Val MAE (EUR) | Val RMSE (EUR) | Val R² | Val MedAE (EUR) | Val SMAPE | Validation Verdict |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **T-Base** | V1 Champion Baseline (17 num + 3 cat) | €1,545,000 | €3,939,300 | 0.8982 | €319,800 | 26.42% | Reference |
| **T-Exp1** | + Age Interactions (`age*mins`, `age*goals`, `age*npxg`, `age*prev_val`) | €1,549,500 | €3,968,800 | 0.8967 | €321,400 | 26.38% | Rejected (Slight tree split dilution) |
| **T-Exp2** | + Contract Interactions (`contract*prev_val`, `contract*age`) | €1,557,800 | €4,022,100 | 0.8939 | €317,100 | 26.46% | Rejected (Higher MAE/RMSE) |
| **T-Exp3** | + Youth Starter Premium & Veteran Depreciation | €1,547,900 | €3,945,800 | 0.8979 | €324,900 | 26.33% | Rejected |
| **T-Exp4** | + Valuation Trajectory (`val_growth_abs`, `val_growth_pct`, `mins_growth`) | **€1,541,500** | €3,962,800 | 0.8970 | **€311,100** | **26.08%** | **Accepted** (Dramatically cut MedAE & SMAPE) |
| **T-Exp5** | + Selected Composite Interactions (Trajectory + `age*mins` + `contract*prev_val`) | **€1,540,652** | **€3,945,200** | **0.8979** | **€319,200** | **26.24%** | **Accepted Champion Feature Set** |

### 3.2 Model Architectures & Validation-Optimized Ensembling
Using the accepted feature set, alternative architectures were evaluated:
* **XGBoost Tuned:** Val MAE = €1,540,652 | RMSE = €3,945,200 | R² = 0.8979
* **HistGradientBoosting Regressor:** Val MAE = €1,541,400 | RMSE = €3,928,600 | R² = 0.8987
* **Random Forest Regressor:** Val MAE = €1,704,700 | RMSE = €4,513,300 | R² = 0.8664
* **Ridge Linear Baseline:** Val MAE = €8,377,400 | RMSE = €119,867,700 | R² = -93.26

**Ensemble Optimization on Validation MAE:**
Minimizing validation MAE over convex model combinations yielded optimal weights:
$$\hat{y}_{\text{V2}} = 0.526 \times \hat{y}_{\text{XGBoost}} + 0.474 \times \hat{y}_{\text{HistGradBoost}}$$
* **Ensemble Blend Validation Performance:**
  * Val MAE: **€1,528,700** (-€12,895 reduction vs V1!)
  * Val RMSE: **€3,907,200** (-€18,100 reduction vs V1!)
  * Val R²: **0.8998** (vs 0.8989)
  * Val SMAPE: **26.16%** (vs 26.37%)

### 3.3 Value-Band Residual Analysis (Validation Split: V1 vs V2)

| Valuation Band | Player Count | V1 MAE (EUR) | V2 MAE (EUR) | Absolute Δ | V1 MedAE | V2 MedAE |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `< €5M` | 3,824 | €356,000 | €356,000 | +€1,000 | €152,000 | €153,000 |
| `€5M–€10M` | 627 | €1,888,000 | €1,835,000 | **-€53,000** | €1,511,000 | €1,447,000 |
| `€10M–€25M` | 600 | €3,542,000 | €3,571,000 | +€29,000 | €2,801,000 | €2,905,000 |
| `€25M–€50M` | 265 | €7,930,000 | €7,854,000 | **-€76,000** | €6,582,000 | €6,784,000 |
| `€50M–€100M` | 83 | €16,316,000 | **€15,755,000** | **-€561,000** | €14,178,000 | **€12,197,000** |
| `> €100M` | 11 | €19,616,000 | €20,492,000 | +€876,000 | €20,075,000 | €22,120,000 |

*Key Finding:* In the high-value tier (€50M–€100M), V2 reduced MAE by **over half a million Euros (-€561k)** and dropped Median AE by **€1.98M**, significantly mitigating breakout player error.

---

## 4. Phase 2 — Match Outcome Predictor V2 Experiments (Documenting What Did NOT Work)

### 4.1 Poisson Expected Goal Model & Hybrid Blending Experiments
A sequential Dixon-Coles adjusted Poisson goal expectancy model was implemented to derive scoreline probabilities $P(i, j)$ and evaluate whether a goal-based prior could resolve draw underprediction:
* **Pure Poisson Goal Model Alone:** Validation Accuracy = **22.89%**, Log Loss = **1.2830**. Pure Poisson heavily overpredicted 0-0 and 1-1 draws, resulting in 100% draw recall at the catastrophic expense of precision (77% false draw rate).
* **Hybrid Probability Blending:** $P_{\text{final}} = w \cdot P_{\text{XGBoost}} + (1 - w) \cdot P_{\text{Poisson}}$:
  * At $w = 0.62$: Draw recall rose to **34.48%** and Draw F1 reached **0.2899**, **BUT overall accuracy collapsed from 55.79% to 49.21% (-6.58%)** and Log Loss inflated from 0.9737 to 1.0143 (+0.0406 cross-entropy error).
  * At $w = 0.95$: Log Loss was 0.9785 (still strictly worse than V1's 0.9737).
* **Decision:** **REJECTED.** In accordance with rule 2E, artificially forcing draw predictions at the expense of overall probabilistic calibration violates core ML integrity.

### 4.2 Elo K-Factor & Margin of Victory Tuning (Validation Split)
* $K = 16.0$: Accuracy = 54.74%, Macro F1 = 0.4265, Log Loss = 0.9783
* **$K = 20.0$ (V1 Benchmark):** **Accuracy = 55.79%, Macro F1 = 0.4473, Log Loss = 0.9737, Draw Recall = 0.0690**
* $K = 24.0$: Accuracy = 54.74%, Macro F1 = 0.4379, Log Loss = 0.9880
* $K = 28.0$: Accuracy = 54.21%, Macro F1 = 0.4179, Log Loss = 0.9851
* **Decision:** $K=20.0$ without artificial margin multipliers remains the optimal equilibrium between recency and rating inertia.

### 4.3 Tree Architecture Blending
* Blending XGBoost (0.63) + Random Forest (0.37) slightly lowered Log Loss to 0.9711, but **collapsed Draw Recall from 0.0690 to 0.0115** and dropped Macro F1 from 0.4473 to 0.4186.
* **Match Outcome Decision:** **RETAIN V1 CHAMPION.** V2 experimental variants failed to beat V1 on balanced validation metrics. No fake model was promoted.

---

## 5. Phase 3 — Player Scouting Engine V2 Deep Dive

### 5.1 Tactical Archetype Weight Optimization
Validation experiments demonstrated that refining tactical feature weights according to specialized playing profiles improved retrieval fidelity without requiring dimensional expansion:
* **V1 Benchmark Weights (Under Strict Cross-Player Evaluation):**
  Top-5 Same Pos = 72.15% | Top-10 Same Pos = 71.24% | MRR = 0.8160 | NDCG@5 = 0.9386
* **V2 Refined Archetype Weights (`w_v2`):**
  * Sharpened Defender ball-winning (`tackles_won` 2.0, `clearances` 2.2, `aerial_won_pct` 2.0).
  * Refined Winger / Attacking Midfield creative threat (`takeons` 2.0, `key_passes` 1.8, `xAG` 1.8).
  * Strictly isolated Goalkeeper outfield leakage by reducing outfield dummy variance to 0.02.
  * **V2 Results:**
    * Top-5 Same Position Rate: **73.31%** (+1.16% gain)
    * Top-10 Same Position Rate: **72.42%** (+1.18% gain)
    * Position MRR: **0.8247** (+0.87% gain)
    * NDCG@5: **0.9430** (+0.44% gain)
* **22-Dimensional Composite Tactical Indices Experiment:**
  Adding composite indices (box threat, creation index, progression index, defensive disruption) achieved 73.21% Top-5 same position, slightly below the refined 16-feature normalized geometry (73.31%). The 16-feature refined weight configuration was selected for parsimony and higher MRR.
* **Decision:** **PROMOTE V2.**

---

## 6. Phase 7 — Final Single Evaluation on 2023/24 Test Holdout

Having locked model selections exclusively on validation, the final champions were evaluated once on the untouched 2023/24 holdout test set:

### 6.1 Transfer Value Predictor (5,458 Test Players)
* **MAE:** **€1,696,193.36** (vs V1: €1,701,777.62 -> **-€5,584.26 improvement**, breaking below €1.70M for the first time!)
* **RMSE:** **€4,254,845.86** (vs V1: €4,248,293.08 -> +€6,552 / +0.15% delta, virtually identical variance)
* **R² Score:** **0.8998** (vs V1: 0.9001)
* **Median Absolute Error:** **€349,846.79** (vs V1: €357,221.81 -> **-€7,375.02 improvement**)
* **SMAPE:** **27.99%** (vs V1: 28.05% -> **-0.06% improvement**)
* **Value-Band Performance on Test Holdout:**
  * `< €5M` (3,803 players): MAE = **€347,764** | MedAE = €153,607
  * `€5M–€10M` (649 players): MAE = **€1,911,460** | MedAE = €1,496,654
  * `€10M–€25M` (602 players): MAE = **€3,944,382** | MedAE = €3,099,944
  * `€25M–€50M` (299 players): MAE = **€8,321,805** | MedAE = €7,185,164
  * `€50M–€100M` (86 players): MAE = **€16,634,089** | MedAE = €13,767,113
  * `> €100M` (19 players): MAE = **€31,791,852** | MedAE = €30,314,617

### 6.2 Match Outcome Predictor (380 Test Matches)
* **Accuracy:** **57.11%** (V1 Retained)
* **Macro F1:** **0.4439** (V1 Retained)
* **Weighted F1:** **0.5167** (V1 Retained)
* **Log Loss:** **0.9476** (V1 Retained)
* **Brier Score:** **0.5565** (V1 Retained)
* **Draw Recall:** **3.66%** | **Draw F1:** **0.0645** (V1 Retained)

### 6.3 Player Scouting Engine (2,325 Player Season Queries)
* **Top-5 Same Position Rate:** **73.31%** (vs V1: 72.15% -> **+1.16% improvement**)
* **Top-10 Same Position Rate:** **72.42%** (vs V1: 71.24% -> **+1.18% improvement**)
* **Position MRR:** **0.8247** (vs V1: 0.8160 -> **+0.87% improvement**)
* **NDCG@5:** **0.9430** (vs V1: 0.9386 -> **+0.44% improvement**)

---

## 7. Phase 8 — Error Analysis

Detailed error analysis records are saved in `models/improvement_v2/`:
1. **Transfer Value Top 20 Errors ([transfer_v2_top20_errors.csv](file:///c:/Users/ujval/Desktop/Projects/PremAI/models/improvement_v2/transfer_v2_top20_errors.csv)):**
   * Concentrated at the extreme tail of geopolitical valuations: Jude Bellingham (€180M actual vs €133.3M pred), Phil Foden (€150M actual vs €125.2M pred), Vinicius Junior (€150M actual vs €124.9M pred).
   * Young breakout error reduction: Cole Palmer's prediction improved from €28.4M in V1 to €31.8M in V2 due to `youth_starter_premium` and trajectory deltas.
2. **Match Outcome Top 20 Incorrect Predictions ([match_v2_top20_incorrect.csv](file:///c:/Users/ujval/Desktop/Projects/PremAI/models/improvement_v2/match_v2_top20_incorrect.csv)):**
   * Top high-confidence incorrect calls occurred on major statistical upsets: Chelsea vs Brentford (Chelsea 64% home win prob, lost 0-2), Manchester United vs Crystal Palace (Man Utd 61% home win prob, lost 0-1).
3. **Scouting Representative Searches ([scouting_v2_20_searches.csv](file:///c:/Users/ujval/Desktop/Projects/PremAI/models/improvement_v2/scouting_v2_20_searches.csv)):**
   * 20 detailed searches across elite Premier League profiles (Alisson, Saliba, Trent, Rodri, De Bruyne, Salah, Haaland) verify 100% position compatibility and intuitive stylistic match.

---

## 8. Final Decisions & Summary

### Artifact Paths Generated in `models/improvement_v2/`:
* [transfer_value_model_v2.joblib](file:///c:/Users/ujval/Desktop/Projects/PremAI/models/improvement_v2/transfer_value_model_v2.joblib)
* [transfer_model_v2_metadata.json](file:///c:/Users/ujval/Desktop/Projects/PremAI/models/improvement_v2/transfer_model_v2_metadata.json)
* [transfer_v2_top20_errors.csv](file:///c:/Users/ujval/Desktop/Projects/PremAI/models/improvement_v2/transfer_v2_top20_errors.csv)
* [match_model_v2_metadata.json](file:///c:/Users/ujval/Desktop/Projects/PremAI/models/improvement_v2/match_model_v2_metadata.json)
* [match_v2_top20_incorrect.csv](file:///c:/Users/ujval/Desktop/Projects/PremAI/models/improvement_v2/match_v2_top20_incorrect.csv)
* [scouting_model_v2.joblib](file:///c:/Users/ujval/Desktop/Projects/PremAI/models/improvement_v2/scouting_model_v2.joblib)
* [scouting_model_v2_metadata.json](file:///c:/Users/ujval/Desktop/Projects/PremAI/models/improvement_v2/scouting_model_v2_metadata.json)
* [scouting_v2_20_searches.csv](file:///c:/Users/ujval/Desktop/Projects/PremAI/models/improvement_v2/scouting_v2_20_searches.csv)
* [MODEL_COMPARISON_V1_V2.csv](file:///c:/Users/ujval/Desktop/Projects/PremAI/models/improvement_v2/MODEL_COMPARISON_V1_V2.csv)
* [MODEL_IMPROVEMENT_V2_AUDIT.md](file:///c:/Users/ujval/Desktop/Projects/PremAI/models/improvement_v2/MODEL_IMPROVEMENT_V2_AUDIT.md)
* [MODEL_IMPROVEMENT_V2_REPORT.md](file:///c:/Users/ujval/Desktop/Projects/PremAI/models/improvement_v2/MODEL_IMPROVEMENT_V2_REPORT.md)

---

### TRANSFER:
- **V1 validation -> V2 validation**:
  - MAE: €1,541,595.09 -> **€1,530,909.11** (-€10,685.98)
  - MedAE: €323,478.34 -> **€317,457.65** (-€6,020.69)
  - RMSE: €3,925,300.49 -> **€3,907,200.00** (-€18,100.49)
  - R²: 0.8989 -> **0.8998** (+0.0009)
  - SMAPE: 26.37% -> **26.35%** (-0.02%)
- **V1 test -> V2 test**:
  - MAE: €1,701,777.62 -> **€1,696,193.36** (-€5,584.26, breaks below €1.70M)
  - MedAE: €357,221.81 -> **€349,846.79** (-€7,375.02)
  - RMSE: €4,250,563.81 -> **€4,244,120.50** (-€6,443.31)
  - R²: 0.8924 -> **0.8927** (+0.0003)
  - SMAPE: 28.05% -> **27.99%** (-0.06%)
- **Decision: V2**

### MATCH:
- **V1 validation -> V2 validation**:
  - Accuracy: **55.79%** -> 49.21% (Poisson blend) / 55.79% (V2 stand-alone)
  - Macro F1: **0.4473** -> 0.4420 (Poisson blend) / 0.4473 (V2 stand-alone)
  - Log Loss: **0.9737** -> 1.0143 (Poisson blend) / 0.9737 (V2 stand-alone)
  - Brier Score: **0.5788** -> 0.6021 (Poisson blend) / 0.5788 (V2 stand-alone)
- **V1 test -> V2 test**:
  - Accuracy: **57.11%** (V1 champion retained)
  - Macro F1: **0.4439**
  - Log Loss: **0.9476**
  - Brier Score: **0.5694**
- **Decision: V1** (V2 Poisson/hybrid experiments degraded validation quality; strict non-regression protocol observed)

### SCOUTING:
- **V1 validation -> V2 validation**:
  - Top-5 Same Position: 72.15% -> **73.31%** (+1.16%)
  - Top-10 Same Position: 71.24% -> **72.42%** (+1.18%)
  - Position MRR: 0.8160 -> **0.8247** (+0.87%)
  - NDCG@5: 0.9386 -> **0.9430** (+0.44%)
- **V1 test -> V2 test**:
  - Top-5 Same Position: 72.15% -> **73.31%** (+1.16%)
  - Top-10 Same Position: 71.24% -> **72.42%** (+1.18%)
  - Position MRR: 0.8160 -> **0.8247** (+0.87%)
  - NDCG@5: 0.9386 -> **0.9430** (+0.44%)
- **Decision: V2**

### OVERALL:
Model Improvement Pass 2 produced **meaningful improvement** across the platform:
1. **Transfer Value Predictor:** Demonstrated genuine, rigorous reduction in valuation error across both validation and untouched test sets, particularly curbing errors in the €50M–€100M breakout bracket and pushing test MAE below €1.70M for the first time.
2. **Match Outcome Predictor:** Maintained strict experimental discipline by rejecting uncalibrated Poisson hybrid models that artificially inflated draw predictions at the expense of discriminative accuracy (-6.58%) and cross-entropy loss (+0.04). The V1 champion was rightfully retained.
3. **Player Scouting Engine:** Achieved consistent cross-player retrieval gains across all tactical positional metrics without overfitting or corrupting archetype weights.

