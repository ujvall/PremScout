# PremAI — Model Improvement Pass 2 Audit & Experiment Plan

**Date:** 2026-09-28  
**Scope:** Advanced Modeling & Prediction Refinement (Phase 2 Research Pass)  
**Governing Rule:** All model selection, feature selection, ensemble weights, and hyperparameter tuning MUST occur exclusively on the **2017/18–2021/22 training** and **2022/23 validation** splits. The 2023/24 test holdout remains untouched until final validation.

---

## 1. Current Champion Baseline (V1 Benchmark on 2022/23 Validation Split)

### Transfer Value Predictor (V1 Benchmark)
* **MAE:** €1,541,595.09 (€1.542M)
* **RMSE:** €3,925,300.49 (€3.925M)
* **R² Score:** 0.8989
* **Median Absolute Error:** €323,478.34
* **SMAPE:** 26.37%
* **Identified Bottlenecks:**
  * Young breakout players (e.g., Cole Palmer, Kobbie Mainoo) and rapid movers undergo massive non-linear valuation adjustments that are underestimated because previous valuation anchors predictions too heavily.
  * Lack of explicit age interaction features (e.g., peak curve penalty for veterans, potential multiplier for U21 with high starter minutes).
  * Lack of trajectory features (season-over-season change in minutes, goals per 90, appearances).

### Match Outcome Predictor (V1 Benchmark)
* **Accuracy:** 0.5579 (55.79%)
* **Macro F1:** 0.4473
* **Weighted F1:** 0.5115
* **Log Loss:** 0.9737
* **Brier Score:** 0.5788
* **Draw Performance:** Precision = 0.3333, Recall = 0.0690, F1 = 0.1143
* **Identified Bottlenecks:**
  * Draw recall remains low (only 6 draws predicted on validation out of 87). Parity between opponents is signaled by Elo difference, but attack/defence interactions (e.g. strong defence vs weak attack) are not explicitly modeled.
  * Absence of a goal-based statistical prior (e.g. bivariate Poisson or Dixon-Coles goal model) to provide principled Poisson probability distributions for 0-0, 1-1, and 2-2 draws.
  * Rolling form windows are fixed at 5 matches; exponential weighting and multi-window views (3, 5, 10 matches) could better capture short-term form momentum.

### Player Scouting Engine (V1 Benchmark)
* **Top-5 Same Position Rate:** 0.7335 (73.35%)
* **Top-10 Same Position Rate:** 0.7206 (72.06%)
* **Top-5 Compatible Position Rate:** 0.9268 (92.68%)
* **Position MRR:** 0.8344
* **NDCG@5:** 0.9439
* **NDCG@10:** 0.9376
* **Identified Bottlenecks:**
  * The similarity engine still relies on a single fixed tactical weight per coarse position group (`DF`, `DM/WB`, `CM`, `AM/W`, `FW`, `GK`).
  * Sub-position nuances (e.g., Wingback vs pure Centre Back, Inverted Winger vs Traditional Centre Forward) are bundled into broad positional buckets.
  * Lack of dedicated tactical archetype sub-dimensions (ball progression, chance creation, ball-winning, pressing proxies).

---

## 2. Hypotheses & Experimentation Roadmap

### Transfer Value Predictor V2 Experiments
* **Exp-T1 (Age & Contract Interaction Features):**
  * *Hypothesis:* Interacting `age` with `minutes_played`, `goals_per90`, and `prev_market_value_eur`, along with `contract_years_left * prev_market_value_eur`, will allow tree models to separate high-ceiling young breakout stars from depreciating veterans with high past values.
* **Exp-T2 (Performance Trajectory & Lag Deltas):**
  * *Hypothesis:* Explicit season-over-season deltas computed strictly from prior seasons (`delta_minutes`, `delta_goals_per90`, `delta_starter_share`) will supply momentum signals for breakout players.
* **Exp-T3 (Model Architecture Exploration & Blending):**
  * *Hypothesis:* An ensemble blending tuned XGBoost, HistGradientBoosting, and Random Forest regressors (weights optimized on validation MAE) will reduce variance on high-value outliers.
* **Exp-T4 (Value-Band Error Segmentation):**
  * *Hypothesis:* Analyzing residuals across <€5M, €5M–€10M, €10M–€25M, €25M–€50M, €50M–€100M, and >€100M will pinpoint where non-linear transformations or tuning adjustments are needed.

### Match Outcome Predictor V2 Experiments
* **Exp-M1 (Advanced Goal & Form Features):**
  * *Hypothesis:* Incorporating multi-window rolling form (last 3, 5, 10 matches) and exponentially weighted form, alongside rolling attack/defense ratings, will improve pre-match strength representations.
* **Exp-M2 (Dynamic & Margin-Adjusted Elo):**
  * *Hypothesis:* Adjusting Elo updates with a margin-of-victory multiplier ($MOV = \ln(|GD| + 1)$) and tuning $K$-factor on the training period will create more discriminative pre-match ratings.
* **Exp-M3 (Poisson Goal Expectancy Statistical Model):**
  * *Hypothesis:* Fitting an independent Poisson goal model on pre-match rolling expected home/away goals to calculate exact scoreline probabilities $P(i, j)$, then extracting $P(H), P(D), P(A)$, will provide an objective draw prior.
* **Exp-M4 (XGBoost + Poisson Hybrid Ensemble):**
  * *Hypothesis:* Blending XGBoost discriminative probabilities with Poisson statistical probabilities (weights tuned strictly on validation Log Loss) will reduce Log Loss and increase Draw F1.

### Player Scouting Engine V2 Experiments
* **Exp-S1 (Sub-Position Tactical Archetypes):**
  * *Hypothesis:* Refining tactical weight profiles using granular sub-positions (`Centre-Back`, `Full-Back`, `Defensive Midfield`, `Central Midfield`, `Attacking Midfield`, `Winger`, `Centre-Forward`) will elevate same-role retrieval fidelity.
* **Exp-S2 (Composite Tactical Dimensions):**
  * *Hypothesis:* Constructing normalized composite tactical archetype indices (Ball Progression, Chance Creation, Box Threat, Ball Winning, Pressing/Disruption) will yield higher stylistic stability across seasons.
* **Exp-S3 (Validation-Optimized Weighting):**
  * *Hypothesis:* Optimizing archetype dimension weights using validation retrieval NDCG will improve recruitment match quality.

---

## 3. Strict Validation & Test Isolation Rules

1. All models will be fit strictly on `train` data (2017/18–2021/22).
2. All feature choices, hyperparameter choices, and ensemble weights will be selected strictly by observing `val` performance (2022/23).
3. Under no circumstances will 2023/24 test metrics be consulted during development.
4. Only if a V2 champion definitively outperforms V1 on the 2022/23 validation set will it be evaluated on the 2023/24 holdout test set.
