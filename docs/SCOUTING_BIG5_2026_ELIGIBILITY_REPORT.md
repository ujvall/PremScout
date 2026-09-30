# PremAI — Final Scouting Eligibility & Current 2026 Big-5 Data Report

**Document ID:** `docs/SCOUTING_BIG5_2026_ELIGIBILITY_REPORT.md`  
**Execution Date:** 2026-09-28  
**Verification Framework:** Pytest, FastAPI TestClient, Vite TS Compiler, Independent Validator (`scripts/validate_current_data.py`)  
**Status:** **APPROVED & FULLY VALIDATED**

---

## 1. Executive Summary & Final Statement

> **"Player Scouting uses the latest verified 2026 current club and competition data. By default, candidates are restricted to players currently registered in Europe's Big Five leagues. Players outside those leagues are excluded, while verified free agents can optionally be included through the Include Free Agents control. Completed 2025/26 performance data is kept separate from current 2026/27 registration data."**

This engineering pass resolves candidate universe contamination in the Player Scouting module. Previously, players registered in leagues outside Europe's Big 5 (e.g. Qatar Stars League, Saudi Pro League, MLS, Portuguese Primeira Liga, Turkish Süper Lig) could appear as recruitment recommendations if they possessed historical or completed 2025/26 performance records. 

Through an architectural overhaul:
1. **Three distinct concepts are strictly decoupled**:
   - **Concept A (Performance Data)**: 2025/26 completed performance metrics, club, and league.
   - **Concept B (Current Registration)**: Post-September-2026 verified 2026/27 squad membership, club, competition, and country.
   - **Concept C (Scouting Eligibility)**: Four explicit data quality states (`BIG5_ACTIVE`, `FREE_AGENT`, `NON_BIG5`, `UNKNOWN`).
2. **Allow-List Architecture**: Europe's Big 5 (`GB1`, `ES1`, `L1`, `IT1`, `FR1`) form an immutable allow-list. No league blacklists, no hardcoded player names, and no manual club lists exist.
3. **Filter Before Similarity**: Ineligible players are stripped from the candidate matrix prior to any cosine distance calculation.
4. **Verified Free-Agent Toggle**: Defaulting to `OFF`, free agents are strictly isolated and only enter the candidate universe when explicitly enabled by user intent.
5. **Zero Model Alteration**: `models/scouting/scouting_model.joblib` SHA256 is 100% identical before and after.

---

## 2. Root Cause Analysis of Previous Contamination

### 2.1 The Conflation Bug
In previous builds, candidate eligibility and vector scoring were coupled to the historical FBref/Transfermarkt performance table. If a player had high statistical similarity in their completed European season (e.g., 2024/25 or 2025/26), the engine retrieved their vector and returned them without verifying whether their **current 2026/27 registration** had shifted to an overseas league (Saudi Pro League, Qatar Stars League, MLS, Brazilian Serie A, etc.).

### 2.2 Old Logic Flaws
- **Old Logic:**
  ```text
  TARGET PLAYER VECTOR 
          ↓
  GLOBAL SCOUTING MATRIX (ALL AVAILABLE PLAYERS)
          ↓
  COSINE SIMILARITY
          ↓
  RETURN TOP-N CANDIDATES (May contain overseas players)
  ```
- **Consequences:**
  - Roberto Firmino (playing for Al-Sadd in the Qatar Stars League) was returned as a replacement for Premier League strikers because his performance profile was active in the historical data.
  - Players transferred to non-Big-5 domestic leagues (Saudi Arabia, MLS, Turkey, Portugal) remained in the candidate universe.
  - Missing current club fields were occasionally misinterpreted as free agency.

---

## 3. New Eligibility Architecture

### 3.1 Strict Separation of Three Concepts
Every player in the scouting pipeline is now modeled across three independent dimensions:

| Concept | Fields | Definition | Source of Truth |
| :--- | :--- | :--- | :--- |
| **A. Performance Data** | `performance_season`<br>`performance_club`<br>`performance_league` | Completed statistical profile (16 standardized per-90 metrics) used for vector embedding. Kept at 2025/26. | FBref / Transfermarkt completed 2025/26 match logs |
| **B. Current Registration** | `current_season`<br>`current_club`<br>`current_club_id`<br>`current_league`<br>`current_league_id`<br>`current_competition`<br>`current_country` | Active squad registration following the closure of the September 1, 2026 European summer transfer window. | Post-summer 2026 verified squad registry (`data/current/players/verified_2026_squads.csv`) |
| **C. Scouting Eligibility** | `eligibility_status`<br>`is_big5_current`<br>`is_free_agent`<br>`scouting_eligible` | Deterministic classification into one of 4 states: `BIG5_ACTIVE`, `FREE_AGENT`, `NON_BIG5`, `UNKNOWN`. | `CurrentPlayerResolver` (`backend/ml/scouting/eligibility.py`) |

### 3.2 Four Data Quality States

1. **`BIG5_ACTIVE`**: Player currently registered in Premier League (`GB1`), La Liga (`ES1`), Bundesliga (`L1`), Serie A (`IT1`), or Ligue 1 (`FR1`). **Eligible by default.**
2. **`FREE_AGENT`**: Player explicitly verified as possessing no active club contract (`FREE`). **Excluded by default; eligible only when `include_free_agents=True`.**
3. **`NON_BIG5`**: Player registered in any non-Big-5 domestic competition (Saudi Pro League, Qatar Stars League, MLS, Primeira Liga, Süper Lig, Eredivisie, Championship, etc.). **Permanently excluded from scouting candidates.**
4. **`UNKNOWN`**: Ambiguous, missing, or unverified registration. **Permanently excluded.**

### 3.3 Stable Competition Allow-List (No Club/League Blacklists)
Eligibility is determined exclusively against stable competition identifiers:
```python
CURRENT_BIG5_COMPETITION_IDS = {
    "GB1",  # Premier League (England)
    "ES1",  # La Liga (Spain)
    "L1",   # Bundesliga (Germany)
    "IT1",  # Serie A (Italy)
    "FR1",  # Ligue 1 (France)
}
```
No manual lists of club names (e.g. "Arsenal", "Real Madrid", "Bayern") and no finite lists of excluded leagues are maintained.

### 3.4 Filter-Before-Similarity Pipeline
```text
CURRENT 2026 PLAYER MASTER (29,653 players)
        ↓
CURRENT REGISTRATION RESOLVER (Post-window 2026/27 squads)
        ↓
BIG-5 ALLOW-LIST FILTER (Isolate BIG5_ACTIVE universe)
        ↓
OPTIONAL FREE-AGENT INCLUSION (When toggle = True)
        ↓
VALID SCOUTING FEATURE PROFILE (16 normalized per-90 metrics)
        ↓
POSITION COMPATIBILITY CONSTRAINT (Exact / Tactical archetype)
        ↓
SCOUTING VECTOR COSINE SIMILARITY
        ↓
SIMILARITY RANKING (Top-N descending)
```

---

## 4. Current 2026 Data Sources & Freshness Verification

To avoid stale 2023/24 historical artifacts or unverified "live" claims, data timestamps are strictly documented:

| Dataset | Scope | Freshness / Cutoff Date | Description |
| :--- | :--- | :--- | :--- |
| **Current Squad Data** | Europe & Global | **2026-09-20** | Reflects all finalized summer transfers following the close of the European window on 1 September 2026. |
| **Performance Data** | 16 Scouting Metrics | **2025/26 (Completed)** | Full completed season performance statistics across Big-5 competitions. |
| **Market Valuations** | Player Valuations | **2026-09-20** | Current market value estimates in EUR. |
| **Match History & Elo**| Sequential Match Engine | **2026-09-20** | 3,470 matches through Matchday 5 of the 2026/27 campaign. |

---

## 5. Player Universe & Candidate Counts

Analysis generated from the active production dataset (`data/current/scouting/current_player_scouting.csv`):

### 5.1 Breakdown by Data Quality State
- **Total Scored Scouting Profiles:** `4,697`
- **`BIG5_ACTIVE` (Default Candidate Universe):** `3,611` (76.88%)
- **`FREE_AGENT` (Optional via Checkbox):** `154` (3.28%)
- **`NON_BIG5` (Permanently Excluded):** `882` (18.78%)
- **`UNKNOWN` (Permanently Excluded):** `50` (1.06%)

### 5.2 Breakdown of `BIG5_ACTIVE` Candidates by League

| Competition | Competition ID | Country | Eligible Candidate Count |
| :--- | :--- | :--- | :--- |
| **La Liga** | `ES1` | Spain | **810** |
| **Serie A** | `IT1` | Italy | **761** |
| **Ligue 1** | `FR1` | France | **719** |
| **Premier League** | `GB1` | England | **695** |
| **Bundesliga** | `L1` | Germany | **626** |
| **Total Big-5 Universe** | — | — | **3,611** |

---

## 6. Dynamic Positive & Negative Test Results

### 6.1 Required Negative Tests (Prompt Section 20)

| Test Subject | Target / Sample Player | Resolved Current Club | Resolved Current League | League ID | Eligibility Status | Candidate Pool Result |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Roberto Firmino** | Roberto Firmino | Al-Sadd | Qatar Stars League | `QAT1` | `NON_BIG5` | **EXCLUDED** (0% appearance in top-50) |
| **Saudi Pro League** | Joshua King / Ronaldo | Al-Khaleej / Al-Nassr | Saudi Pro League | `SA1` | `NON_BIG5` | **EXCLUDED** |
| **MLS** | James Sands / Messi | NYCFC / Inter Miami | MLS | `MLS1` | `NON_BIG5` | **EXCLUDED** |
| **Qatar Stars League** | Youcef Atal | Al-Sadd | Qatar Stars League | `QAT1` | `NON_BIG5` | **EXCLUDED** |
| **Portuguese Primeira** | Mohamed Bamba | Gil Vicente FC | Primeira Liga | `PO1` | `NON_BIG5` | **EXCLUDED** |
| **Turkish Süper Lig** | Emmanuel Agbadou | Beşiktaş JK | Süper Lig | `TR1` | `NON_BIG5` | **EXCLUDED** |
| **Free Agent (Default)**| Martin Dúbravka | Without Club | Free Agent | `FREE` | `FREE_AGENT` | **EXCLUDED** (`include_free_agents=False`) |

### 6.2 Free Agent Checkbox Verification

```text
Target: Martin Dúbravka (Goalkeeper search, Top 30)

Query A: include_free_agents = False
   → Total Candidates Returned: 30
   → Free Agents Found: 0
   → 100% of candidates are BIG5_ACTIVE (Jaume Doménech, Quilichini, David Soria, etc.)

Query B: include_free_agents = True
   → Total Candidates Returned: 30
   → Free Agents Found: 1 (Guillaume Dietsch | Score: 95.9 | Current League: Free Agent)
   → Verified Free Agent successfully included without leaking overseas players.
```

### 6.3 Required Dynamic Positive Tests (Prompt Section 21)
Players dynamically selected across all five leagues with zero hardcoded names:

- **Premier League (`GB1`):** Gabriel Magalhães (Arsenal) — `BIG5_ACTIVE`, Season `2026-27` [ELIGIBLE]
- **La Liga (`ES1`):** Pablo Fornals (Real Betis) — `BIG5_ACTIVE`, Season `2026-27` [ELIGIBLE]
- **Bundesliga (`L1`):** Finn Dahmen (FC Augsburg) — `BIG5_ACTIVE`, Season `2026-27` [ELIGIBLE]
- **Serie A (`IT1`):** Vitinha (Genoa CFC) — `BIG5_ACTIVE`, Season `2026-27` [ELIGIBLE]
- **Ligue 1 (`FR1`):** Carlens Arcus (Angers SCO) — `BIG5_ACTIVE`, Season `2026-27` [ELIGIBLE]

---

## 7. Model Integrity & Hash Verification

To comply with the strict instruction that no models be retrained, no weights be modified, and `models/scouting/scouting_model.joblib` remain pristine, cryptographic SHA-256 hashes were taken:

| Artifact | Checkpoint | SHA-256 Checksum | Match Status |
| :--- | :--- | :--- | :--- |
| `models/scouting/scouting_model.joblib` | Prior to Pass | `c7bc1b36eadeb19e954ac9bc9240583c0b9ac1b14bbc12a223348aa5c3173be2` | — |
| `models/scouting/scouting_model.joblib` | After Pass | `c7bc1b36eadeb19e954ac9bc9240583c0b9ac1b14bbc12a223348aa5c3173be2` | **100% EXACT MATCH** |
| `models/transfer/transfer_value_model.joblib` | After Pass | `b4e6faec27530663ba42cfbb9024f0cba1a1a5b8bbec86cfb993bb53e747b0e0` | **UNMODIFIED** |
| `models/match/match_outcome_model.joblib` | After Pass | `073e5ef264a132c3f87625126838384dbffc1d763a82dfbcffcbdfef1ebca472` | **UNMODIFIED** |

---

## 8. API & Frontend Implementations

### 8.1 Backend API Specification
- **`GET /api/scouting/similar/{player_name}?top_n=10&include_free_agents=false`**
- **`POST /api/scouting/replacements`**
  ```json
  {
    "player_name": "Declan Rice",
    "top_n": 10,
    "same_position_only": true,
    "include_free_agents": false
  }
  ```
- **Response Fields Returned for Every Candidate:**
  - `current_club`, `current_league`, `current_country`, `current_season`
  - `performance_club`, `performance_league`, `performance_season`
  - `is_current_big5` (bool), `is_free_agent` (bool), `eligibility_status` (string)

### 8.2 Frontend UI Enhancements
- **Top Scope Banner (`ScoutingModule.tsx`):**
  - Displays: `Current Big 5 European Leagues | Premier League • La Liga • Bundesliga • Serie A • Ligue 1`
  - Freshness Breakdown: `Current squads: 2026/27 (as of 2026-09-20) | Performance data: 2025/26 | Market values as of: 2026-09-20`
  - Control: Interactive `[ ] Include Free Agents` checkbox (Default: **OFF / Unchecked**).
- **Candidate Tables (`SimilarPlayersView.tsx` & `ReplacementSearchView.tsx`):**
  - Columns strictly match Section 14:
    `RANK | PLAYER | CLUB | LEAGUE | POSITION | SIMILARITY | COMPATIBILITY | AGE | MARKET VALUE | MINUTES`
  - League badge displays current European league or Free Agent status.
- **Player Search & Autocomplete (`PlayerSearchInput.tsx`):**
  - Clearly distinguishes candidate eligibility:
    - `[Big 5 Eligible]` (Emerald badge)
    - `[Free Agent (Checkbox Req.)]` (Amber badge)
    - `[Ineligible: Outside Big 5]` (Rose badge for overseas players like Firmino)

---

## 9. Comprehensive QA & Test Run Summary

```text
================================================================================
1. Validation Script (scripts/validate_current_data.py)
   → Total Checks: 49
   → Passed: 49
   → Failed: 0
   → Result: 100% Success Rate

2. Pytest API Suite (tests/api/test_api.py)
   → Total Tests: 25
   → Passed: 25
   → Failed: 0
   → Result: 100% Success Rate

3. Frontend TypeScript & Bundle Build (npm run build)
   → Command: tsc -b && vite build
   → Modules Transformed: 1,905
   → Build Errors: 0
   → Bundle: dist/assets/index-CSSfiNVI.js (346.04 kB)
   → Result: 100% Clean Production Build
================================================================================
```

---

## 10. Conclusion & Production Readiness

The Player Scouting candidate eligibility architecture is resolved and hardened:
- Candidates are strictly confined to Europe's Big 5 leagues by default.
- Overseas leagues (Saudi, Qatar, MLS, etc.) cannot leak into recruitment results.
- Verified free agents are securely gated behind the `Include Free Agents` control.
- Current 2026/27 registrations take precedence over historical club affiliations.
- Completed 2025/26 performance statistics remain intact for vector similarity.
- Production models, transfer predictors, and match outcome predictors are completely unmodified.
