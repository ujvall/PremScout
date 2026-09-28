# PremAI - Player Scouting & Replacement Engine: Model Report & Evaluation

**Module:** Machine Learning Module 3 - Player Scouting & Replacement  
**Project:** Football Intelligence Platform (PremAI)  
**Date:** September 2026  
**Status:** Completed & Validated  
**Model Artifact:** `models/scouting/scouting_model.joblib`  
**Embeddings:** `models/scouting/player_embeddings.csv`  
**Metadata:** `models/scouting/model_metadata.json`  

---

> **CRITICAL DISCLAIMER:**  
> This system is an algorithmic **player style similarity and replacement identification engine**. A high similarity score between two players indicates that they exhibit statistically comparable per-90 event profiles across attacking, passing, defending, and aerial metrics. **It is NOT a claim that one player is objectively better than, or an exact 1:1 substitute for, another.** Real-world football recruitment requires complementary qualitative assessment, video scouting, personality evaluation, medical history checks, and tactical coaching alignment.

---

## 1. Dataset Overview & Size

The model is built on the validated dataset:
`data/processed/scouting/player_scouting_dataset.csv`

The dataset covers **7 consecutive Premier League seasons** (2017/18 through 2023/24) with a minimum seasonal threshold of **450 minutes played** to eliminate statistical noise from brief substitute appearances.

- **Total Player-Season Records:** **2,325**
- **Unique Individual Players:** **925**
- **Seasons Included:** 2017/18, 2018/19, 2019/20, 2020/21, 2021/22, 2022/23, 2023/24
- **Features in Dataset:** 28 total columns (16 performance metrics, 12 demographic and identifier fields)
- **Missing Values in Core Performance Metrics:** **0 (100% complete)**

---

## 2. Features Used (16 Core Football Performance Dimensions)

To eliminate playing-time bias, all 16 metrics are expressed as **per-90-minute averages** or **rate percentages**:

| Category | Feature Name | Description | Units |
| :--- | :--- | :--- | :--- |
| **Attacking** | `goals_per90` | Non-penalty and penalty goals scored | Goals / 90 |
| **Attacking** | `assists_per90` | Direct goal assists | Assists / 90 |
| **Attacking** | `xG_per90` | Expected goals | xG / 90 |
| **Attacking** | `npxG_per90` | Non-penalty expected goals | npxG / 90 |
| **Attacking** | `xAG_per90` | Expected assisted goals | xAG / 90 |
| **Attacking** | `shots_per90` | Total shots attempted | Shots / 90 |
| **Attacking** | `key_passes_per90` | Passes directly leading to a shot | Key Passes / 90 |
| **Passing & Progression** | `pass_completion_pct`| Percentage of completed passes | % (0 - 100) |
| **Passing & Progression** | `progressive_passes_per90` | Completed passes advancing $\ge 10$ yards | Passes / 90 |
| **Passing & Progression** | `progressive_carries_per90`| Carries advancing ball $\ge 10$ yards | Carries / 90 |
| **Passing & Progression** | `successful_takeons_per90` | Successful 1v1 dribbles | Dribbles / 90 |
| **Defending & Aerial** | `tackles_won_per90` | Tackles won in all thirds | Tackles / 90 |
| **Defending & Aerial** | `interceptions_per90`| Interceptions executed | Interceptions / 90 |
| **Defending & Aerial** | `blocks_per90` | Shots and passes blocked | Blocks / 90 |
| **Defending & Aerial** | `clearances_per90` | Defensive clearances | Clearances / 90 |
| **Defending & Aerial** | `aerial_won_pct` | Percentage of aerial duels won | % (0 - 100) |

### Deliberate Exclusion of Market Value from Feature Vectors:
`market_value_eur` is intentionally **excluded** from the similarity feature space. Football performance similarity represents how a player operates tactically on the pitch, not their economic price tag. Market value is utilized exclusively as an optional recruitment filter (e.g. finding affordable alternatives under €40M).

---

## 3. Position Grouping & Tactical Compatibility

Raw position tags from FBref and Transfermarkt are mapped to 6 canonical position groups:

| Raw Tag(s) | Normalized Group | Description | Population Count |
| :--- | :--- | :--- | :--- |
| `DF` | **DF** | Central Defenders / Fullbacks | 803 (34.5%) |
| `MF` | **CM** | Central Midfielders | 492 (21.2%) |
| `MF,FW`, `FW,MF` | **AM/W** | Attacking Midfielders / Wingers | 393 (16.9%) |
| `FW` | **FW** | Centre-Forwards / Strikers | 314 (13.5%) |
| `GK` | **GK** | Goalkeepers | 172 (7.4%) |
| `DF,MF`, `MF,DF`, `DF,FW`, `FW,DF` | **DM/WB** | Defensive Utility / Wing-Backs | 151 (6.5%) |

### Positional Compatibility Rules:
- **Goalkeepers (`GK`):** Strictly non-interchangeable (only compatible with `GK`). A goalkeeper will never be recommended for an outfield player.
- **Defenders (`DF`):** Compatible with `DF` and `DM/WB`.
- **Defensive Utility (`DM/WB`):** Compatible with `DM/WB`, `DF`, and `CM`.
- **Central Midfielders (`CM`):** Compatible with `CM`, `DM/WB`, and `AM/W`.
- **Attacking Midfielders / Wingers (`AM/W`):** Compatible with `AM/W`, `CM`, and `FW`.
- **Forwards (`FW`):** Compatible with `FW` and `AM/W`.

Each recommendation is labeled with `position_compatibility`:
- `"Same Position"`: Exact match in normalized group.
- `"Compatible Position"`: Valid tactical adjacent group.

---

## 4. Preprocessing & Standardization

Because performance metrics operate on vastly different scales (e.g. `pass_completion_pct` ~ 78% vs `goals_per90` ~ 0.13), standardizing features is essential to prevent percentage metrics from dominating geometric distance.

1. **Imputation:** `SimpleImputer(strategy='median')` ensures zero nulls.
2. **Scaling:** `StandardScaler()` shifts each feature to zero mean and unit variance:
   $$z_{ij} = \frac{x_{ij} - \mu_j}{\sigma_j}$$
3. **L2 Normalization:** Each standardized player vector $\vec{z}_i$ is normalized to unit Euclidean length:
   $$\hat{z}_i = \frac{\vec{z}_i}{\|\vec{z}_i\|_2}$$
   This enables cosine similarity to be computed via dot products.

---

## 5. Similarity & Scoring Methodology

### Cosine Similarity:
Similarity between target player $\vec{u}$ and candidate player $\vec{v}$ is calculated as:
$$\text{cos\_sim}(\vec{u}, \vec{v}) = \frac{\vec{u} \cdot \vec{v}}{\|\vec{u}\|_2 \|\vec{v}\|_2}$$

### Normalization to 0–100 Scale:
Raw cosine similarity on standardized features typically ranges from $-0.5$ (opposite style) to $+1.0$ (identical style). For intuitive user interpretation:
$$\text{similarity\_score} = \text{round}\Big(\max\big(0.0, \text{cos\_sim}(\vec{u}, \vec{v})\big) \times 100,\ 1\Big)$$

- **Score $\ge 90.0$:** Very high statistical similarity (e.g. Rodri vs Declan Rice: **93.7 / 100**).
- **Score $80.0 - 89.9$:** Strong functional similarity with minor tactical nuances.
- **Score $< 70.0$:** Moderate to low similarity.
- **Score $\le 50.0$:** Disparate playing styles.

---

## 6. Practical Scouting Filters

The retrieval service `find_replacements()` in `backend.ml.scouting.predict` supports parameterized recruitment queries:

- `same_position_only` (bool): Restrict candidates to exact normalized position group (True) or include tactically compatible positions (False).
- `max_market_value` (float, optional): Maximum affordable transfer value in EUR.
- `max_age` (int, optional): Maximum candidate age in years for youth scouting.
- `min_minutes` (int, optional): Minimum playing time filter to ensure sample reliability.
- `season` (str, optional): Target player's historical season profile (defaults to latest available).
- **Candidate Deduplication:** Ensures distinct individual players (retaining their most similar season profile).
- **Self-Exclusion:** The target player themselves is excluded across all historical seasons.

---

## 7. Retrieval Evaluation Methodology & Results

Because scouting similarity is an unsupervised recommendation task, classification accuracy is not applicable. The system was evaluated using **legitimate information retrieval and multi-season stability metrics** across all 2,325 player seasons:

| Metric | Measured Value | Benchmark / Interpretation |
| :--- | :--- | :--- |
| **Top-5 Same-Position Retrieval Rate** | **70.96%** | Pure performance stats naturally cluster identical positions |
| **Top-10 Same-Position Retrieval Rate** | **70.02%** | Sustained high positional purity across deep rankings |
| **Top-5 Compatible Position Rate** | **92.05%** | Over 92% of nearest neighbors fall in tactically valid positions |
| **Top-10 Compatible Position Rate** | **91.00%** | Virtually zero goalkeeper/outfield cross-contamination |
| **Position Mean Reciprocal Rank (MRR)** | **0.8124** | The first matching position player appears at average rank 1.2 |
| **Top-10 Multi-Season Self-Retrieval Rate** | **45.03%** | 45% of multi-season players retrieve themselves across different years |
| **Pairwise Cosine Similarity Mean** | **0.0095** | Distribution is well-centered near orthogonal ($0.0$) |
| **Pairwise Cosine Similarity Median** | **-0.0062** | Balanced geometric representation |

Artifact saved: `models/scouting/similarity_evaluation.csv`.

---

## 8. Feature Contribution Analysis

Cosine similarity does not have traditional supervised tree feature importance. Instead, we compute the **mathematical elementwise contribution to the inner product among nearest neighbors**:
$$\text{contribution}_j = \mathbb{E}_{(i, k) \in \text{Top-5}} [\hat{z}_{ij} \cdot \hat{z}_{kj}]$$

| Rank | Feature Dimension | Relative Contribution | Tactical Domain |
| :--- | :--- | :--- | :--- |
| 1 | `progressive_passes_per90` | **7.83%** | Ball progression and vertical passing |
| 2 | `clearances_per90` | **7.78%** | Last-line box defending |
| 3 | `tackles_won_per90` | **7.24%** | Ground dueling and ball winning |
| 4 | `interceptions_per90` | **7.16%** | Anticipation and defensive positioning |
| 5 | `pass_completion_pct` | **6.38%** | Possession security and passing accuracy |
| 6 | `blocks_per90` | **6.33%** | Defensive disruption |
| 7 | `key_passes_per90` | **6.23%** | Chance creation and vision |
| 8 | `shots_per90` | **6.06%** | Direct goal threat volume |
| 9 | `progressive_carries_per90` | **6.02%** | Dribbling progression |
| 10 | `aerial_won_pct` | **5.88%** | Aerial dominance |

All 16 features exhibit balanced contributions between **5.1% and 7.8%**, confirming that no single feature dominates the embedding space.

Artifact saved: `models/scouting/feature_importance.csv`.

---

## 9. Example Replacement Searches

Generated from real Premier League records in `models/scouting/example_replacements.csv`:

### A. Central Midfield: Declan Rice (Target: CM, Market Value: €120M)
| Rank | Candidate | Club | Age | Valuation | Similarity Score | Compatibility |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | **Rodri** | Manchester City | 27 | €130M | **93.7 / 100** | Same Position |
| 2 | **Douglas Luiz** | Aston Villa | 24 | €40M | **92.9 / 100** | Same Position |
| 3 | **Pape Matar Sarr** | Tottenham Hotspur | 21 | €45M | **92.1 / 100** | Same Position |
| 4 | **Conor Gallagher** | Chelsea | 24 | €50M | **90.2 / 100** | Same Position |
| 5 | **Bruno Guimarães** | Newcastle United | 24 | €70M | **89.5 / 100** | Same Position |

### B. Elite Striker: Erling Haaland (Target: FW, Market Value: €180M)
| Rank | Candidate | Club | Age | Valuation | Similarity Score | Compatibility |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | **Callum Wilson** | Newcastle United | 30 | €16M | **98.0 / 100** | Same Position |
| 2 | **Tammy Abraham** | Chelsea | 21 | N/A | **97.4 / 100** | Same Position |
| 3 | **Sergio Agüero** | Manchester City | 31 | N/A | **96.9 / 100** | Same Position |
| 4 | **Pierre-Emerick Aubameyang** | Arsenal | 29 | €70M | **96.6 / 100** | Same Position |
| 5 | **Dominic Solanke** | Bournemouth | 26 | €40M | **96.4 / 100** | Same Position |

### C. Wide Attacker: Bukayo Saka (Target: FW/Winger, Market Value: €140M)
| Rank | Candidate | Club | Age | Valuation | Similarity Score | Compatibility |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | **Anthony Gordon** | Newcastle United | 23 | €60M | **93.7 / 100** | Same Position |
| 2 | **Gabriel Martinelli** | Arsenal | 23 | €70M | **92.7 / 100** | Same Position |
| 3 | **Mohamed Salah** | Liverpool | 32 | €55M | **92.0 / 100** | Same Position |
| 4 | **Phil Foden** | Manchester City | 22 | €110M | **91.3 / 100** | Same Position |
| 5 | **Bryan Mbeumo** | Brentford | 24 | €40M | **89.3 / 100** | Same Position |

### D. Dominant Centre-Back: Virgil van Dijk (Target: DF, Market Value: €80M)
| Rank | Candidate | Club | Age | Valuation | Similarity Score | Compatibility |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | **Adam Webster** | Brighton | 29 | €20M | **93.2 / 100** | Same Position |
| 2 | **Lewis Dunk** | Brighton | 32 | €12M | **92.8 / 100** | Same Position |
| 3 | **Thiago Silva** | Chelsea | 37 | €2M | **90.9 / 100** | Same Position |
| 4 | **Raphaël Varane** | Manchester United | 31 | €20M | **89.4 / 100** | Same Position |
| 5 | **Cristian Romero** | Tottenham Hotspur | 26 | €65M | **88.4 / 100** | Same Position |

---

## 10. System Limitations

1. **Statistical Profile $\neq$ Absolute Quality:** Two players with identical progressive passing and pressing statistics may play in very different tactical systems. A 94% similarity score reflects style alignment, not equivalent individual quality.
2. **Team Dominance & Possession Bias:** Players on dominant teams (e.g. Manchester City, Arsenal) naturally record higher possession and passing volume, while defensive players on lower-table clubs record higher tackle and clearance volume due to sustained opposition pressure.
3. **Physical & Medical Absence:** The system does not incorporate tracking data (top sprinting speed, acceleration, high-intensity distance) or medical injury histories.
4. **Sample Size for Young Players:** A minimum seasonal cutoff of 450 minutes is enforced; players with fewer minutes are excluded to prevent small-sample distortion.

---

## 11. Reproduction Instructions

To reproduce the embeddings, evaluation tables, example searches, and plots:
```bash
python -m backend.ml.scouting.train
```

To run interactive replacement queries:
```python
from backend.ml.scouting.predict import find_replacements

# Find candidates for any player
candidates = find_replacements(
    player_name="Declan Rice",
    top_n=5,
    max_market_value=50_000_000,
    max_age=25
)
```
