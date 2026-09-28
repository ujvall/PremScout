# PremAI - Transfer Value Predictor: Model Report & Evaluation

**Module:** Machine Learning Module 1 - Transfer Value Predictor  
**Project:** Football Intelligence Platform (PremAI)  
**Date:** September 2026  
**Status:** Completed & Validated  
**Model Artifact:** `models/transfer/transfer_value_model.joblib`  
**Metadata:** `models/transfer/model_metadata.json`  

---

## 1. Problem Definition

In modern European football, player market valuations and transfer fees are critical economic assets. However, player pricing is notoriously opaque, driven by asymmetric negotiations, agent influence, and speculative inflation. 

The goal of the **Transfer Value Predictor** is to establish an objective, data-driven regression model that estimates a football player's realistic market valuation (`market_value_eur` in EUR) based on:
- **Demographics & Physical Profile:** Age, height, preferred foot.
- **Contractual & Financial Context:** Contract duration remaining, club financial stature (club squad valuation).
- **Historical Valuation Anchor:** Previous season valuation (`prev_market_value_eur`).
- **On-Pitch Performance & Workload:** Appearances, minutes played, goals, assists, discipline, shots per 90, and non-penalty expected goals per 90 (`npxg_per90`).

This is formulated as a continuous regression problem on $y \in [\text{€25,000}, \text{€200,000,000}+]$.

---

## 2. Dataset Size & Breakdown

The model is trained and evaluated on the validated dataset:
`data/processed/transfer/transfer_value_dataset.csv`

| Partition | Seasons Included | Row Count | Percentage |
| :--- | :--- | :--- | :--- |
| **Train** | 2017/18, 2018/19, 2019/20, 2020/21, 2021/22 | **22,580** | 67.5% |
| **Validation** | 2022/23 | **5,410** | 16.2% |
| **Test** | 2023/24 | **5,458** | 16.3% |
| **Total** | **7 Seasons** | **33,448** | **100.0%** |

---

## 3. Features Used

To ensure generalizability and eliminate data leakage, player identities and target leakages were strictly excluded.

### Predictor Features (18 total):
1. **Demographics (2):** `age`, `height_cm`
2. **Financial & Contractual (3):** `contract_years_left`, `club_market_value_eur`, `prev_market_value_eur`
3. **Playing Volume & Discipline (4):** `minutes_played`, `appearances`, `yellow_cards`, `red_cards`
4. **Performance & Goal Threat (5):** `goals`, `assists`, `goals_per90`, `assists_per90`, `shots_per90`, `npxg_per90`
5. **Categoricals (3):** `primary_position` (Attack, Midfield, Defender, Goalkeeper), `sub_position` (14 specific roles), `preferred_foot` (Left, Right, Both)

### Explicitly Excluded Columns:
- `player_id`, `player_name`: Excluded to prevent the model from memorizing individual identities.
- `club_name`: Excluded to prevent club-specific memorization; represented objectively via `club_market_value_eur`.
- `market_value_eur`: Excluded from feature matrix $X$ (target variable).
- Future season information: Excluded to prevent lookahead bias.

---

## 4. Train / Validation / Test Split Strategy

- **No Random Splitting:** Football valuation data has strong temporal autocorrelation. A random split (e.g. 80/20 train_test_split) leaks contemporary transfer inflation and player career progressions between folds, yielding overly optimistic, invalid metrics.
- **Strict Out-of-Time Chronological Split:**
  - **Train (2017/18 – 2021/22):** 5 historical seasons capturing long-term market trends.
  - **Validation (2022/23):** 1 full season used exclusively for model selection and comparing target transformations.
  - **Test (2023/24):** 1 full season held strictly untouched until final evaluation.

---

## 5. Preprocessing Architecture

All transformations are encapsulated into a Scikit-Learn `ColumnTransformer` fitted strictly on training data:

```
[Input Raw Features]
        │
        ├── Numerical Features (15) ──> [SimpleImputer (median)] ──> [StandardScaler]
        │
        └── Categorical Features (3) ─> [SimpleImputer (most_frequent)] ─> [OneHotEncoder (handle_unknown='ignore')]
        │
[Preprocessed Feature Matrix X_proc (38 features)]
```

- **Missing Values:** Numerical nulls (e.g. `npxg_per90` for lower-tier matches or missing height) are imputed using the median of the training distribution. Categorical nulls are imputed with `'Unknown'`.
- **Scaling:** Features are standardized with zero mean and unit variance for linear models and normalized convergence.
- **Categorical Encoding:** One-Hot Encoding creates 38 total transformed features, ignoring unknown categories at test/inference time.
- **Zero Leakage Guarantee:** Imputation medians and scaling parameters are learned solely on the training partition (`split_group == 'train'`).

---

## 6. Models Tested

We benchmarked three distinct model families:
1. **Linear Regression (OLS Baseline):** Standard ordinary least squares regression.
2. **Random Forest Regressor:** Bagged ensemble of 100 decision trees (`max_depth=16`, `min_samples_split=5`, `random_state=42`).
3. **XGBoost Regressor:** Gradient boosted decision trees (`n_estimators=150`, `learning_rate=0.08`, `max_depth=6`, `subsample=0.85`, `colsample_bytree=0.85`, `random_state=42`).

---

## 7. Raw vs. Log Target Transformation Analysis

The distribution of `market_value_eur` is heavily right-skewed (skewness = 5.25), spanning several orders of magnitude. We tested both:
- **Approach A (Raw Target):** Model trains directly on `market_value_eur`.
- **Approach B (Log1p Target):** Model trains on $z = \log(1 + y)$. Predictions are inverted at evaluation/inference time via $\hat{y} = \exp(z) - 1$.

### Comparative Results:

| Model | Target Transform | Train MAE | Val MAE | Test MAE | Test RMSE | Test R² |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **XGBoost** | **log1p** | **€1.19M** | **€1.56M** | **€1.71M** | **€4.28M** | **0.8986** |
| XGBoost | raw | €1.08M | €1.63M | €1.70M | €4.23M | 0.9010 |
| Random Forest | log1p | €720.8K | €1.68M | €1.78M | €4.61M | 0.8825 |
| Random Forest | raw | €729.0K | €1.72M | €1.74M | €4.33M | 0.8960 |
| Linear Regression | raw | €2.23M | €2.26M | €2.19M | €4.77M | 0.8742 |
| Linear Regression | log1p | €12.86M | €14.87M | €27.18M | €974.70M | -5259.0354 |

### Findings on Target Transformation:
1. **Tree Models Benefit from log1p:** For both XGBoost and Random Forest, `log1p` yielded lower Validation MAE (**€1.56M vs. €1.63M** for XGBoost, **€1.68M vs. €1.72M** for Random Forest). Because tree models split on absolute residual error, raw targets over-prioritize the ultra-high valuation tier (e.g. €100M+ players) at the expense of accuracy on the 95% of players valued under €15M. `log1p` stabilizes error penalties across all valuation tiers.
2. **Linear Regression Failure with log1p:** OLS Linear Regression is unconstrained and extrapolated large positive outputs for certain feature combinations. When passed through $\exp(z)$, these produced extreme positive predictions (€10B+), causing catastrophic test set failure ($R^2 < 0$).

---

## 8. Validation Results & Model Selection

In strict compliance with machine learning best practices:
> **Rule:** The final production model was selected purely based on **Validation MAE** (Season 2022/23). The 2023/24 test set was NOT consulted during model selection.

- **Rank 1:** **XGBoost (log1p)** — Val MAE: **€1,560,465 (€1.56M)**
- **Rank 2:** XGBoost (raw) — Val MAE: €1,626,819 (€1.63M)
- **Rank 3:** Random Forest (log1p) — Val MAE: €1,684,210 (€1.68M)
- **Rank 4:** Random Forest (raw) — Val MAE: €1,720,115 (€1.72M)
- **Rank 5:** Linear Regression (raw) — Val MAE: €2,260,340 (€2.26M)
- **Rank 6:** Linear Regression (log1p) — Val MAE: €14.87M

**Selected Model:** **XGBoost Regressor with log1p target transformation**.

---

## 9. Final Test Performance (Season 2023/24)

After selecting the model based on the validation set, it was evaluated **once** on the untouched 2023/24 test set (5,458 player seasons):

| Metric | Score (Raw) | Score (Formatted) | Interpretation |
| :--- | :--- | :--- | :--- |
| **MAE** | €1,706,977 | **€1.71M** | Average absolute valuation error across all 5,458 players |
| **RMSE** | €4,279,569 | **€4.28M** | Penalizes large outlier deviations |
| **R²** | 0.898597 | **0.8986 (~89.9%)** | Explains ~90% of transfer valuation variance |
| **MAPE** | 38.4% | **38.4%** | Average relative percentage error |

The model demonstrates excellent generalization, with test MAE (€1.71M) remaining close to validation MAE (€1.56M).

---

## 10. Feature Importance

Feature importances extracted from the final XGBoost model:

| Rank | Feature | Importance | Cumulative | Domain Rationale |
| :--- | :--- | :--- | :--- | :--- |
| 1 | `prev_market_value_eur` | **57.79%** | 57.79% | Foundational anchor; player market price is heavily autoregressive. |
| 2 | `club_market_value_eur` | **15.88%** | 73.67% | Club financial tier; elite clubs command higher valuation multiples. |
| 3 | `age` | **3.55%** | 77.22% | Age curve; younger players retain peak future resale value. |
| 4 | `minutes_played` | **3.36%** | 80.58% | Workload & trust; indicates regular starter status. |
| 5 | `goals` | **3.29%** | 83.87% | Offensive end-product; goals command the highest transfer premium. |
| 6 | `appearances` | **2.82%** | 86.69% | Match involvement and fitness reliability. |
| 7 | `npxg_per90` | **2.37%** | 89.06% | Underlying attacking quality independent of penalties. |
| 8 | `contract_years_left` | **1.71%** | 90.77% | Contract leverage; expiring contracts force lower transfer fees. |
| 9 | `assists` | **1.43%** | 92.20% | Playmaking contribution and chance creation. |
| 10 | `yellow_cards` | **0.68%** | 92.88% | Defensive aggression / disciplinary record. |

**Observation:** Demographics and club financial stature anchor ~75% of the valuation, while on-pitch performance metrics (`minutes`, `goals`, `appearances`, `npxg`) provide the dynamic adjustment factor.

---

## 11. Error Analysis & Real Player Predictions

Test predictions from `models/transfer/test_predictions.csv` highlight key behavioral patterns:

### A. Spot-on Predictions (Established Premier League Stars)
- **Erling Haaland (Manchester City):** Actual: **€180.00M** | Predicted: **€171.60M** | Error: **€8.40M (4.7%)**
- **William Saliba (Arsenal):** Actual: **€80.00M** | Predicted: **€79.68M** | Error: **€322.0K (0.4%)**
- **Bukayo Saka (Arsenal):** Actual: **€140.00M** | Predicted: **€126.27M** | Error: **€13.73M (9.8%)**
- **Alexander Isak (Newcastle):** Actual: **€75.00M** | Predicted: **€88.23M** | Error: **€13.23M (17.6%)**

### B. Largest Absolute Underpredictions (Actual > Predicted)
- **Declan Rice (Arsenal):** Actual: **€120.00M** | Predicted: **€82.13M** | Error: **€37.87M (31.6%)**
  - *Cause:* Record €105M summer transfer fee and immediate adaptation to title-contending Arsenal inflated his market valuation faster than historical statistical models predict.
- **Cole Palmer (Chelsea):** Actual: **€80.00M** | Predicted: **€48.75M** | Error: **€31.25M (39.1%)**
  - *Cause:* Breakout season phenomenon. In 2022/23, Palmer was a Manchester City bench youth player (valuation ~€15M). His explosion to 22 Premier League goals in 2023/24 caused Transfermarkt to upgrade him to €80M in a single year. The model partially anchors on his prior valuation.
- **Kobbie Mainoo (Manchester United):** Actual: **€55.00M** | Predicted: **€30.30M** | Error: **€24.70M (44.9%)**
  - *Cause:* Zero prior first-team valuation baseline.

### C. Largest Absolute Overpredictions (Predicted > Actual)
- **Richarlison (Tottenham Hotspur):** Actual: **€38.00M** | Predicted: **€63.15M** | Overvalued by **€25.15M**
  - *Cause:* High historical prior valuation and high club market valuation maintained high predictions despite intermittent injuries.
- **Antony (Manchester United):** Actual: **€25.00M** | Predicted: **€48.20M** | Overvalued by **€23.20M**
  - *Cause:* €75M prior valuation and Man United squad valuation created inertia against sharp downward market depreciation.

---

## 12. Visualizations

Generated plots are saved under `models/transfer/plots/`:
1. `actual_vs_predicted.png`: Scatter plot comparing actual vs. predicted values with ideal 45-degree reference line.
2. `residual_distribution.png`: Histogram and KDE of prediction residuals centered around zero.
3. `feature_importance.png`: Ranked horizontal bar chart of the top 15 predictive signals.
4. `test_predictions_scatter.png`: Full out-of-time test set evaluation scatter plot.

---

## 13. Limitations

1. **Breakout Player Lag:** Because previous valuation is the strongest predictor (57.8%), meteoric breakouts (e.g. Cole Palmer, Kobbie Mainoo) will see a 1-season lag before fully reaching consensus human valuations.
2. **Qualitative & Contract Nuance:** The model lacks data on contract buyout/release clauses, agent negotiations, wage structures, and non-disclosure transfer clauses.
3. **League Reputation Bias:** Players in lower leagues or non-top-5 European leagues suffer from lower club market valuation baselines even if their individual statistics are world-class.

---

## 14. Application Integration Architecture

The module is structured for seamless integration into the upcoming FastAPI backend and React frontend:

```
[FastAPI Endpoint: POST /api/v1/transfer/predict]
                    │
                    ▼
  [backend.ml.transfer.predict.predict_transfer_value(player_features)]
                    │
                    ├── Validates input features against ALL_MODEL_FEATURES
                    ├── Loads cached artifact (transfer_value_model.joblib)
                    ├── Executes Scikit-Learn Pipeline (Impute + Scale + OHE)
                    ├── Computes XGBoost log1p prediction
                    ├── Inverts target via expm1() and clips floor (€25,000)
                    └── Formats response (€M / €K) + metadata
                    │
                    ▼
[Client Response JSON: { "predicted_market_value_eur": 88230000, "formatted": "€88.23M" }]
```

The model artifact is self-contained, stateless, and ready for production serving.
