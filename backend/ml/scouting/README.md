# PremAI: Player Scouting & Replacement Module

The **Player Scouting & Replacement** module provides a vector similarity retrieval engine for discovering statistically similar football players and identifying suitable replacement candidates under real-world scouting constraints.

---

## 1. Core Purpose

Given a target player (e.g. *Declan Rice*, *Erling Haaland*, or *Bukayo Saka*), the module:
1. Embeds the player into a standardized 16-dimensional football performance space.
2. Identifies players exhibiting statistically equivalent on-pitch profiles.
3. Enforces tactical position awareness (e.g. separating Strikers, Central Midfielders, and Goalkeepers).
4. Applies practical recruitment filters (maximum affordable transfer valuation, age ceilings, playing time).
5. Produces normalized similarity scores ($0 - 100$) and comparative metrics.

---

## 2. Features (16 Football Performance Dimensions)

All metrics are playing-time normalized (per 90 minutes or rate percentages) from FBref match logs:

- **Attacking:** `goals_per90`, `assists_per90`, `xG_per90`, `npxG_per90`, `xAG_per90`, `shots_per90`, `key_passes_per90`
- **Passing & Progression:** `pass_completion_pct`, `progressive_passes_per90`, `progressive_carries_per90`, `successful_takeons_per90`
- **Defending & Aerial:** `tackles_won_per90`, `interceptions_per90`, `blocks_per90`, `clearances_per90`, `aerial_won_pct`

**Important Design Choice on Market Value:**  
`market_value_eur` is intentionally **excluded** from the similarity feature space. Football performance similarity represents how a player plays, not their price tag. Market valuation is applied exclusively as an optional recruitment constraint (e.g. finding a replacement valued under €30M).

---

## 3. Position Grouping & Compatibility Rules

Raw position labels from FBref and Transfermarkt are mapped to 6 canonical tactical groups:
- `GK`: Goalkeeper
- `DF`: Defender (Centre-Back / Full-Back)
- `DM/WB`: Defensive Utility / Wing-Back
- `CM`: Central Midfield
- `AM/W`: Attacking Midfield / Winger
- `FW`: Forward / Striker

### Positional Compatibility:
- `GK` $\rightarrow$ strictly `GK`
- `DF` $\rightarrow$ `DF`, `DM/WB`
- `DM/WB` $\rightarrow$ `DM/WB`, `DF`, `CM`
- `CM` $\rightarrow$ `CM`, `DM/WB`, `AM/W`
- `AM/W` $\rightarrow$ `AM/W`, `CM`, `FW`
- `FW` $\rightarrow$ `FW`, `AM/W`

---

## 4. Execution & Usage

### Run Training & Indexing Pipeline:
```bash
python -m backend.ml.scouting.train
```

### Python Inference:
```python
from backend.ml.scouting.predict import find_replacements, get_similar_players

# Find budget-friendly replacements for Declan Rice (under €40M, max age 26)
replacements = find_replacements(
    player_name="Declan Rice",
    top_n=5,
    same_position_only=True,
    max_market_value=40_000_000,
    max_age=26,
    min_minutes=900,
)

for cand in replacements:
    print(f"{cand['candidate_player']} ({cand['candidate_club']}) - Score: {cand['similarity_score']} / 100")
```
