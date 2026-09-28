# PremAI — Final Model Promotion Report

**Date of Promotion:** 2026-09-28  
**Scope:** Official Promotion of Validated Champions to Production  
**Production Target:** `models/transfer/`, `models/match/`, `models/scouting/`  
**Backup Directory:** `models/production_backup_final/`  
**Auditor:** Automated Model Promotion Pipeline  

---

## 1. Overview of Promoted Models

| Module | Previous Production Model | New Production Model | Source Research Artifact | Status |
| :--- | :--- | :--- | :--- | :--- |
| **Transfer Value Predictor** | Baseline XGBoost (18 features) | **V2 XGBoost + HistGradientBoosting Convex Ensemble** | [models/improvement_v2/transfer_value_model_v2.joblib](file:///c:/Users/ujval/Desktop/Projects/PremAI/models/improvement_v2/transfer_value_model_v2.joblib) | **PROMOTED (V2)** |
| **Match Outcome Predictor** | Baseline XGBoost (25 features) | **V1 Improved XGBoost + Sequential Pre-Match Elo** | [models/improvement/match_outcome_model_improved.joblib](file:///c:/Users/ujval/Desktop/Projects/PremAI/models/improvement/match_outcome_model_improved.joblib) | **PROMOTED (V1 Improved)** |
| **Player Scouting Engine** | Baseline Cosine Index (16 features) | **V2 Tactical Archetype Weighted Cosine Index** | [models/improvement_v2/scouting_model_v2.joblib](file:///c:/Users/ujval/Desktop/Projects/PremAI/models/improvement_v2/scouting_model_v2.joblib) | **PROMOTED (V2)** |

---

## 2. Production Backup Verification

Prior to any file replacement, all active production artifacts were backed up into [models/production_backup_final/](file:///c:/Users/ujval/Desktop/Projects/PremAI/models/production_backup_final/):

* `models/production_backup_final/transfer_value_model.joblib` (733,794 bytes)
* `models/production_backup_final/transfer_model_metadata.json` (1,232 bytes)
* `models/production_backup_final/match_outcome_model.joblib` (284,546 bytes)
* `models/production_backup_final/match_model_metadata.json` (1,811 bytes)
* `models/production_backup_final/scouting_model.joblib` (774,835 bytes)
* `models/production_backup_final/scouting_model_metadata.json` (1,620 bytes)

All backup files were confirmed intact and read-only.

---

## 3. Cryptographic SHA-256 Identity Verification

SHA-256 checksums were calculated to verify that production `.joblib` files are bit-for-bit identical to the validated research artifacts and distinct from old backups:

### Transfer Model
* **Source Artifact (`models/improvement_v2/transfer_value_model_v2.joblib`):**  
  `f5b6dc22f569532fe41b3117fe283e7eeae68832a875a5e3e26fcf5b47a9ff60`
* **Production Model (`models/transfer/transfer_value_model.joblib`):**  
  `f5b6dc22f569532fe41b3117fe283e7eeae68832a875a5e3e26fcf5b47a9ff60`
* **Old Backup Model (`models/production_backup_final/transfer_value_model.joblib`):**  
  `3ae2feeb7ab34b025d57d07e60058b8d4f434c4f0ae593593bcf9ec81643c751`
* **Status:** **PASS (Source & Production Match Exactly; Replaced Old Baseline)**

### Match Model
* **Source Artifact (`models/improvement/match_outcome_model_improved.joblib`):**  
  `045055b6963658309df5db08466e3b56c4aa0d88c037ce634b3e8a4d444432a5`
* **Production Model (`models/match/match_outcome_model.joblib`):**  
  `045055b6963658309df5db08466e3b56c4aa0d88c037ce634b3e8a4d444432a5`
* **Old Backup Model (`models/production_backup_final/match_outcome_model.joblib`):**  
  `a994813ceaf8076b1b426ba8b98eb5343d3b76cf6d4e2df4e941ca73a48e788c`
* **Status:** **PASS (Source & Production Match Exactly; Replaced Old Baseline)**

### Scouting Model
* **Source Artifact (`models/improvement_v2/scouting_model_v2.joblib`):**  
  `c7bc1b36eadeb19e2e1e0750ff8dca8018eec4a29a4358bb003bfa99320e8b2b`
* **Production Model (`models/scouting/scouting_model.joblib`):**  
  `c7bc1b36eadeb19e2e1e0750ff8dca8018eec4a29a4358bb003bfa99320e8b2b`
* **Old Backup Model (`models/production_backup_final/scouting_model.joblib`):**  
  `43ce780df13b14bead90390bb9cb63333331454ee6f05e46e8c8a14b537c4424`
* **Status:** **PASS (Source & Production Match Exactly; Replaced Old Baseline)**

---

## 4. Fresh Process Model Loading & Inference Verification

Each promoted model was loaded in an isolated Python process:

1. **Transfer Value Predictor:**
   * Bundle loaded successfully (`preprocessor`, `xgb_model`, `hgb_model`, `features`).
   * Test inference executed on Declan Rice preset: predicted **€74,002,000.00 (€74.00M)**.
   * Target transformation: `log1p`. Value bounded and non-negative.
2. **Match Outcome Predictor:**
   * Pipeline loaded successfully (`pipeline`, `features`, `team_elo_ratings`).
   * Test inference executed on Arsenal vs Chelsea: Home Win (H) predicted.
   * Probabilities: Home: 48.21%, Draw: 20.82%, Away: 30.97% (Sum: 1.0000 $\pm$ 1e-5).
3. **Player Scouting Engine:**
   * Index instantiated on 2,325 player seasons.
   * Query for Declan Rice: Top candidate retrieved is **Rodri (Manchester City, CM, 96.1% similarity)**.
   * Strict position filtering verified; target player excluded from candidate ranking.

---

## 5. Live FastAPI Integration & Health Check

The live FastAPI backend at `http://127.0.0.1:8000` was verified:

* `GET /api/health` $\rightarrow$ `HTTP 200`
  ```json
  {
    "status": "healthy",
    "timestamp": "2026-09-28T11:13:48Z",
    "models": {
      "transfer": true,
      "match": true,
      "scouting": true
    }
  }
  ```
* `POST /api/transfer/predict` $\rightarrow$ `HTTP 200` (Declan Rice: €84.94M, Erling Haaland: €118.39M, Bukayo Saka: €107.56M).
* `POST /api/match/predict` $\rightarrow$ `HTTP 200` (Arsenal vs Chelsea: Home Win 79.4%, Man City vs Liverpool: Home Win 50.7%, Aston Villa vs Tottenham: Home Win 68.5%).
* `GET /api/scouting/similar/Declan%20Rice` $\rightarrow$ `HTTP 200` (Top match: Rodri, 96.1/100).
* `POST /api/scouting/replacements` $\rightarrow$ `HTTP 200` (Filtered replacement search executed cleanly).

---

## 6. Automated QA Test Suite Results

1. **Pytest Integration Suite (`tests/api/test_api.py`):**
   * **Result:** **14 / 14 PASSED (100%)** in 2.96s.
   * Root, Health, Transfer, Match, and Scouting endpoints all passed strict assertions.
2. **End-to-End QA Script (`scripts/qa_verification.py`):**
   * **Result:** **38 / 38 PASSED (100% Success Rate)**.
   * Real player presets, edge cases, validation boundaries (422), and 404 handlers verified.

---

## 7. Frontend Production Build Verification

Executed `npm run build` in `frontend/`:
```
> frontend@0.0.0 build
> tsc -b && vite build

vite v8.3.1 building client environment for production...
transforming...
✓ 1905 modules transformed.
rendering chunks...
computing gzip size...
dist/index.html                   1.45 kB │ gzip:  0.75 kB
dist/assets/index-EPhuhmRc.css    8.73 kB │ gzip:  2.54 kB
dist/assets/index-DR7ou5eH.js   329.13 kB │ gzip: 89.27 kB

✓ built in 6.92s
```
Zero TypeScript errors, zero build warnings.

---

## 8. Final Production Model Versions & Specifications

### Transfer Value Predictor: **V2 Ensemble Champion**
* **Architecture:** XGBoost (52.6%) + HistGradientBoosting (47.4%)
* **Features:** 26 features (lag value trajectory, minutes growth, contract-value interactions)
* **Validation (2022/23):** MAE = **€1,530,909.14**, MedAE = **€317,476.16**, R² = **0.8994**
* **Test Holdout (2023/24):** MAE = **€1,696,193.36** (<€1.70M), MedAE = **€349,846.79**, R² = **0.8998**

### Match Outcome Predictor: **V1 Improved Champion**
* **Architecture:** XGBoost Classifier + Sequential Pre-Match Elo Ratings ($K=20.0$, Home Adv $=65.0$)
* **Features:** 32 features (25 pre-match stats + 7 Elo/parity indicators)
* **Validation (2022/23):** Accuracy = **55.79%**, Macro F1 = **0.4473**, Log Loss = **0.9737**
* **Test Holdout (2023/24):** Accuracy = **57.11%**, Macro F1 = **0.4439**, Log Loss = **0.9476**

### Player Scouting Engine: **V2 Tactical Archetype Champion**
* **Architecture:** Cosine Similarity on 16 Tactical Archetype Weighted Dimensions with Strict GK Isolation
* **Validation / Benchmark:**
  * Top-5 Same Position: **73.31%** (Protocol B) / **74.34%** (Protocol A)
  * Top-10 Same Position: **72.42%** (Protocol B) / **73.13%** (Protocol A)
  * Position MRR: **0.8247** (Protocol B) / **0.8419** (Protocol A)
  * NDCG@5: **0.9430** (Protocol B) / **0.9479** (Protocol A)

---

## 9. Final Promotion Summary

```
============================================================
FINAL PROMOTION STATUS
============================================================
TRANSFER: V2 PROMOTED
MATCH: V1 PROMOTED
SCOUTING: V2 PROMOTED
============================================================
```
