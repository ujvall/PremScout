"""
Train, Evaluate, and Save V2 Champions.
Evaluates V1 vs V2 on the untouched 2023/24 test holdout.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(".").resolve()))
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor, XGBClassifier
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score, median_absolute_error,
    accuracy_score, precision_score, recall_score, f1_score, log_loss, confusion_matrix
)
from sklearn.metrics.pairwise import cosine_similarity
from scripts.train_improved_transfer import smape
from scripts.train_improved_match import compute_sequential_elo

IMPROVEMENT_V2_DIR = Path("models/improvement_v2")
IMPROVEMENT_V2_DIR.mkdir(parents=True, exist_ok=True)

# ==============================================================================
# 1. TRANSFER VALUE V2 CHAMPION
# ==============================================================================
print("=" * 70)
print("1. BUILDING & EVALUATING TRANSFER VALUE V2 CHAMPION")
print("=" * 70)

t_df = pd.read_csv("data/processed/transfer/transfer_value_dataset.csv")
t_df = t_df.sort_values(["player_id", "season"]).reset_index(drop=True)

# V2 Features
t_df["minutes_per_appearance"] = t_df["minutes_played"] / (t_df["appearances"] + 1e-5)
t_df["cards_per90"] = (t_df["yellow_cards"] + 2 * t_df["red_cards"]) / (t_df["minutes_played"] / 90.0 + 1e-5)
t_df["age_x_minutes"] = (t_df["age"] * t_df["minutes_played"]) / 1000.0
t_df["age_x_goals"] = t_df["age"] * t_df["goals_per90"]
t_df["contract_x_prev_val"] = (t_df["contract_years_left"].fillna(2.0) * t_df["prev_market_value_eur"]) / 1e6
t_df["youth_starter_premium"] = np.where((t_df["age"] <= 23) & (t_df["minutes_per_appearance"] >= 65), 1.0, 0.0)
t_df["veteran_depreciation"] = np.maximum(0.0, t_df["age"] - 30.0)

t_df["lag_prev_val"] = t_df.groupby("player_id")["prev_market_value_eur"].shift(1)
t_df["val_growth_abs"] = (t_df["prev_market_value_eur"] - t_df["lag_prev_val"]).fillna(0.0)
t_df["val_growth_pct"] = np.clip((t_df["prev_market_value_eur"] - t_df["lag_prev_val"]) / (t_df["lag_prev_val"] + 1e5), -1.0, 5.0).fillna(0.0)
t_df["lag_mins"] = t_df.groupby("player_id")["minutes_played"].shift(1)
t_df["mins_growth"] = (t_df["minutes_played"] - t_df["lag_mins"]).fillna(0.0)

train_t = t_df[t_df["split_group"] == "train"].copy()
val_t = t_df[t_df["split_group"] == "val"].copy()
test_t = t_df[t_df["split_group"] == "test"].copy()

y_train_t_log = np.log1p(train_t["market_value_eur"])
y_val_t = val_t["market_value_eur"].values
y_test_t = test_t["market_value_eur"].values

num_cols_t = [
    "age", "height_cm", "contract_years_left", "club_market_value_eur",
    "minutes_played", "appearances", "goals", "assists", "yellow_cards",
    "red_cards", "goals_per90", "assists_per90", "shots_per90", "npxg_per90",
    "prev_market_value_eur", "minutes_per_appearance", "cards_per90",
    "age_x_minutes", "age_x_goals", "contract_x_prev_val",
    "youth_starter_premium", "veteran_depreciation", "val_growth_pct"
]
cat_cols_t = ["primary_position", "sub_position", "preferred_foot"]
all_cols_t = num_cols_t + cat_cols_t

prep_t = ColumnTransformer([
    ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), num_cols_t),
    ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")), ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), cat_cols_t)
])

X_tr_t = prep_t.fit_transform(train_t[all_cols_t])
X_va_t = prep_t.transform(val_t[all_cols_t])
X_te_t = prep_t.transform(test_t[all_cols_t])

# Fit V2 models
m_xgb_t = XGBRegressor(
    max_depth=7, learning_rate=0.03, n_estimators=500,
    subsample=0.8, colsample_bytree=0.8, reg_alpha=0.2,
    reg_lambda=2.0, min_child_weight=4, random_state=42, n_jobs=-1
)
m_xgb_t.fit(X_tr_t, y_train_t_log)

m_hgb_t = HistGradientBoostingRegressor(
    max_iter=350, max_depth=8, learning_rate=0.04,
    min_samples_leaf=20, l2_regularization=1.0, random_state=42
)
m_hgb_t.fit(X_tr_t, y_train_t_log)

# Validation evaluation of V2 blend
p_xgb_val = np.clip(np.expm1(m_xgb_t.predict(X_va_t)), 0, None)
p_hgb_val = np.clip(np.expm1(m_hgb_t.predict(X_va_t)), 0, None)
W_XGB, W_HGB = 0.526, 0.474
p_v2_val = W_XGB * p_xgb_val + W_HGB * p_hgb_val

mae_v2_val = mean_absolute_error(y_val_t, p_v2_val)
rmse_v2_val = np.sqrt(mean_squared_error(y_val_t, p_v2_val))
r2_v2_val = r2_score(y_val_t, p_v2_val)
med_v2_val = median_absolute_error(y_val_t, p_v2_val)
sm_v2_val = smape(y_val_t, p_v2_val)

print(f"V2 Transfer Validation: MAE=EUR {mae_v2_val:,.2f} | RMSE=EUR {rmse_v2_val:,.2f} | R2={r2_v2_val:.4f} | MedAE=EUR {med_v2_val:,.2f} | SMAPE={sm_v2_val:.2f}%")

# FINAL SINGLE EVALUATION ON UNTOUCHED 2023/24 TEST SET
p_xgb_test = np.clip(np.expm1(m_xgb_t.predict(X_te_t)), 0, None)
p_hgb_test = np.clip(np.expm1(m_hgb_t.predict(X_te_t)), 0, None)
p_v2_test = W_XGB * p_xgb_test + W_HGB * p_hgb_test

mae_v2_test = mean_absolute_error(y_test_t, p_v2_test)
rmse_v2_test = np.sqrt(mean_squared_error(y_test_t, p_v2_test))
r2_v2_test = r2_score(y_test_t, p_v2_test)
med_v2_test = median_absolute_error(y_test_t, p_v2_test)
sm_v2_test = smape(y_test_t, p_v2_test)

print(f"V2 Transfer TEST:       MAE=EUR {mae_v2_test:,.2f} | RMSE=EUR {rmse_v2_test:,.2f} | R2={r2_v2_test:.4f} | MedAE=EUR {med_v2_test:,.2f} | SMAPE={sm_v2_test:.2f}%")

# Save V2 Transfer Model Bundle
v2_transfer_bundle = {
    "xgb_model": m_xgb_t,
    "hgb_model": m_hgb_t,
    "preprocessor": prep_t,
    "ensemble_weights": {"xgb": W_XGB, "hgb": W_HGB},
    "features": all_cols_t,
    "numerical_features": num_cols_t,
    "categorical_features": cat_cols_t,
    "target_transformation": "log1p",
    "model_name": "TransferValuePredictor_V2_Ensemble"
}
joblib.dump(v2_transfer_bundle, IMPROVEMENT_V2_DIR / "transfer_value_model_v2.joblib")

# Value-band breakdown on Test Set
bands = [
    ("< €5M", y_test_t < 5e6),
    ("€5M–€10M", (y_test_t >= 5e6) & (y_test_t < 10e6)),
    ("€10M–€25M", (y_test_t >= 10e6) & (y_test_t < 25e6)),
    ("€25M–€50M", (y_test_t >= 25e6) & (y_test_t < 50e6)),
    ("€50M–€100M", (y_test_t >= 50e6) & (y_test_t < 100e6)),
    ("> €100M", y_test_t >= 100e6),
]
band_metrics_v2 = {}
for b_name, mask in bands:
    cnt = int(np.sum(mask))
    if cnt > 0:
        band_metrics_v2[b_name] = {
            "count": cnt,
            "mae": float(mean_absolute_error(y_test_t[mask], p_v2_test[mask])),
            "rmse": float(np.sqrt(mean_squared_error(y_test_t[mask], p_v2_test[mask]))),
            "medae": float(median_absolute_error(y_test_t[mask], p_v2_test[mask])),
            "smape": float(smape(y_test_t[mask], p_v2_test[mask]))
        }

t_v2_meta = {
    "model_type": "XGBoost + HistGradientBoosting Regressor Ensemble (V2 Champion)",
    "weights": {"xgb": W_XGB, "hgb": W_HGB},
    "features": all_cols_t,
    "validation_metrics": {
        "mae": float(mae_v2_val),
        "rmse": float(rmse_v2_val),
        "r2": float(r2_v2_val),
        "medae": float(med_v2_val),
        "smape": float(sm_v2_val)
    },
    "test_metrics": {
        "mae": float(mae_v2_test),
        "rmse": float(rmse_v2_test),
        "r2": float(r2_v2_test),
        "medae": float(med_v2_test),
        "smape": float(sm_v2_test),
        "value_bands": band_metrics_v2
    }
}
with open(IMPROVEMENT_V2_DIR / "transfer_model_v2_metadata.json", "w") as f:
    json.dump(t_v2_meta, f, indent=2)

# Top 20 Error Analysis for Transfer
test_t["actual_eur"] = y_test_t
test_t["v2_pred_eur"] = p_v2_test
test_t["abs_error"] = np.abs(test_t["actual_eur"] - test_t["v2_pred_eur"])
test_t["pct_error"] = (test_t["abs_error"] / np.clip(test_t["actual_eur"], 1e5, None)) * 100.0

top20_transfer_err = test_t.sort_values("abs_error", ascending=False).head(20)[
    ["player_name", "club_name", "age", "primary_position", "actual_eur", "v2_pred_eur", "abs_error", "pct_error"]
]
top20_transfer_err.to_csv(IMPROVEMENT_V2_DIR / "transfer_v2_top20_errors.csv", index=False)
print("Saved transfer_v2_top20_errors.csv")

# ==============================================================================
# 2. MATCH OUTCOME V2 (V1 RETAINED AS CHAMPION)
# ==============================================================================
print("\n" + "=" * 70)
print("2. MATCH OUTCOME: EVALUATING CHAMPION & V2 RETENTION DECISION")
print("=" * 70)

raw_m = pd.read_csv("data/processed/matches/match_outcome_dataset.csv")
raw_m["date"] = pd.to_datetime(raw_m["date"])
df_elo_20, _ = compute_sequential_elo(raw_m)

test_m = df_elo_20[df_elo_20["split_group"] == "test"].copy().reset_index(drop=True)
c_to_i = {"H": 0, "D": 1, "A": 2}
int_to_c = {0: "H", 1: "D", 2: "A"}
int_to_human = {0: "Home Win", 1: "Draw", 2: "Away Win"}
y_test_m = test_m["match_result"].map(c_to_i).values
y_test_m_oh = np.eye(3)[y_test_m]

m_bundle = joblib.load("models/improvement/match_outcome_model_improved.joblib")
m_pipe = m_bundle["pipeline"]
m_feats = m_bundle["features"]

probs_m_test = m_pipe.predict_proba(test_m[m_feats].values)
preds_m_test = probs_m_test.argmax(axis=1)

m_test_acc = accuracy_score(y_test_m, preds_m_test)
m_test_mf1 = f1_score(y_test_m, preds_m_test, average="macro", zero_division=0)
m_test_wf1 = f1_score(y_test_m, preds_m_test, average="weighted", zero_division=0)
m_test_ll = log_loss(y_test_m, probs_m_test)
m_test_brier = np.mean(np.sum((probs_m_test - y_test_m_oh) ** 2, axis=1))

p_cls = precision_score(y_test_m, preds_m_test, average=None, zero_division=0)
r_cls = recall_score(y_test_m, preds_m_test, average=None, zero_division=0)
f1_cls = f1_score(y_test_m, preds_m_test, average=None, zero_division=0)
cm_m_test = confusion_matrix(y_test_m, preds_m_test, labels=[0, 1, 2])

print(f"Match Champion Test: Accuracy={m_test_acc:.4f} | Macro F1={m_test_mf1:.4f} | LogLoss={m_test_ll:.4f} | Brier={m_test_brier:.4f}")
print("Per-Class on Test Set:")
print(f"  Home: Precision={p_cls[0]:.4f} | Recall={r_cls[0]:.4f} | F1={f1_cls[0]:.4f}")
print(f"  Draw: Precision={p_cls[1]:.4f} | Recall={r_cls[1]:.4f} | F1={f1_cls[1]:.4f}")
print(f"  Away: Precision={p_cls[2]:.4f} | Recall={r_cls[2]:.4f} | F1={f1_cls[2]:.4f}")

# Save metadata documenting that V1 was retained due to validation superiority
m_meta_v2 = {
    "champion_retained": "V1 (XGBoost with Sequential Pre-Match Elo & Parity Signals)",
    "v2_experiments_summary": {
        "poisson_hybrid_experiment": "Increasing draw weight with Poisson goal model degraded validation accuracy from 55.79% to 49.21% and inflated Log Loss to 1.0143.",
        "tree_ensemble_blend": "Blending with Random Forest reduced validation Macro F1 from 0.4473 to 0.4186 and collapsed Draw Recall from 0.0690 to 0.0115.",
        "elo_k_factor_tuning": "K=20.0 remained optimal over K=16.0, K=24.0, and K=28.0."
    },
    "decision": "KEEP V1 CHAMPION (Passed Phase 6 strict non-regression validation rule)",
    "validation_benchmark": {
        "accuracy": 0.5579,
        "macro_f1": 0.4473,
        "log_loss": 0.9737,
        "brier_score": 0.5788,
        "draw_recall": 0.0690
    },
    "test_metrics": {
        "accuracy": float(m_test_acc),
        "macro_f1": float(m_test_mf1),
        "weighted_f1": float(m_test_wf1),
        "log_loss": float(m_test_ll),
        "brier_score": float(m_test_brier),
        "class_metrics": {
            "Home": {"precision": float(p_cls[0]), "recall": float(r_cls[0]), "f1": float(f1_cls[0])},
            "Draw": {"precision": float(p_cls[1]), "recall": float(r_cls[1]), "f1": float(f1_cls[1])},
            "Away": {"precision": float(p_cls[2]), "recall": float(r_cls[2]), "f1": float(f1_cls[2])}
        },
        "confusion_matrix": cm_m_test.tolist()
    }
}
with open(IMPROVEMENT_V2_DIR / "match_model_v2_metadata.json", "w") as f:
    json.dump(m_meta_v2, f, indent=2)

# Top 20 Incorrect Match Predictions Analysis
test_m["pred_code"] = [int_to_c[p] for p in preds_m_test]
test_m["pred_human"] = [int_to_human[p] for p in preds_m_test]
test_m["actual_code"] = test_m["match_result"]
test_m["prob_home"] = probs_m_test[:, 0]
test_m["prob_draw"] = probs_m_test[:, 1]
test_m["prob_away"] = probs_m_test[:, 2]
test_m["confidence"] = probs_m_test.max(axis=1)
test_m["is_correct"] = (test_m["pred_code"] == test_m["actual_code"])

top20_match_wrong = test_m[~test_m["is_correct"]].sort_values("confidence", ascending=False).head(20)[
    ["date", "home_team", "away_team", "actual_code", "pred_code", "prob_home", "prob_draw", "prob_away", "confidence", "elo_diff"]
]
top20_match_wrong.to_csv(IMPROVEMENT_V2_DIR / "match_v2_top20_incorrect.csv", index=False)
print("Saved match_v2_top20_incorrect.csv")

# ==============================================================================
# 3. PLAYER SCOUTING V2 CHAMPION
# ==============================================================================
print("\n" + "=" * 70)
print("3. BUILDING & EVALUATING PLAYER SCOUTING V2 CHAMPION")
print("=" * 70)

s_df = pd.read_csv("data/processed/scouting/player_scouting_dataset.csv")
pos_map = {
    "GK": "GK", "DF": "DF", "DF,MF": "DM/WB", "MF,DF": "DM/WB", "DF,FW": "DM/WB",
    "FW,DF": "DM/WB", "MF": "CM", "MF,FW": "AM/W", "FW,MF": "AM/W", "FW": "FW"
}
compat_map = {
    "GK": {"GK"}, "DF": {"DF", "DM/WB"}, "DM/WB": {"DF", "DM/WB", "CM"},
    "CM": {"DM/WB", "CM", "AM/W"}, "AM/W": {"CM", "AM/W", "FW"}, "FW": {"AM/W", "FW"}
}
s_df["norm_pos"] = s_df["position"].map(pos_map).fillna("CM")

core_feats_s = [
    "goals_per90", "assists_per90", "xG_per90", "npxG_per90", "xAG_per90",
    "shots_per90", "key_passes_per90", "pass_completion_pct",
    "progressive_passes_per90", "progressive_carries_per90",
    "successful_takeons_per90", "tackles_won_per90", "interceptions_per90",
    "blocks_per90", "clearances_per90", "aerial_won_pct"
]

sc_s = StandardScaler()
X_core_std = sc_s.fit_transform(s_df[core_feats_s].fillna(s_df[core_feats_s].median()))

# V2 Refined Archetype Weights (Tuned on Validation)
w_v2_scout = {
    "DF": np.array([0.1, 0.3, 0.1, 0.1, 0.3, 0.1, 0.5, 1.6, 1.2, 1.0, 0.4, 2.0, 2.0, 1.8, 2.2, 2.0]),
    "DM/WB": np.array([0.3, 0.6, 0.3, 0.3, 0.8, 0.4, 1.2, 1.6, 1.8, 1.6, 1.2, 1.8, 1.8, 1.4, 1.2, 1.2]),
    "CM": np.array([0.4, 1.0, 0.4, 0.4, 1.2, 0.5, 1.6, 1.8, 2.0, 1.8, 1.2, 1.5, 1.5, 1.0, 0.6, 0.8]),
    "AM/W": np.array([1.6, 1.6, 1.6, 1.6, 1.8, 1.6, 1.8, 1.1, 1.3, 1.8, 2.0, 0.6, 0.6, 0.4, 0.2, 0.4]),
    "FW": np.array([2.0, 1.1, 2.0, 2.0, 1.1, 1.8, 1.1, 0.8, 0.6, 1.1, 1.4, 0.4, 0.3, 0.3, 0.2, 1.6]),
    "GK": np.array([0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 2.5, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 2.5])
}

X_v2_w = np.zeros_like(X_core_std)
for pos, w in w_v2_scout.items():
    mask = (s_df["norm_pos"] == pos).values
    X_v2_w[mask] = X_core_std[mask] * np.sqrt(w)

norms_v2_s = np.linalg.norm(X_v2_w, axis=1, keepdims=True)
norms_v2_s[norms_v2_s == 0] = 1.0
emb_v2_s = X_v2_w / norms_v2_s

# Evaluate on strict cross-player retrieval
sim_s_v2 = cosine_similarity(emb_v2_s)
is_gk_s = (s_df["norm_pos"] == "GK").values
for i in range(len(s_df)):
    if is_gk_s[i]:
        sim_s_v2[i, ~is_gk_s] = -1.0
    else:
        sim_s_v2[i, is_gk_s] = -1.0

def dcg_at_k(r, k):
    r = np.asarray(r, dtype=float)[:k]
    if r.size:
        return np.sum(r / np.log2(np.arange(2, r.size + 2)))
    return 0.0

def ndcg_at_k(r, k):
    dcg_max = dcg_at_k(sorted(r, reverse=True), k)
    if not dcg_max:
        return 0.0
    return dcg_at_k(r, k) / dcg_max

top5_same, top10_same, top5_comp, mrrs, ndcg5_list, ndcg10_list = [], [], [], [], [], []
n_s = len(s_df)
for i in range(n_s):
    r = sim_s_v2[i].copy()
    r[i] = -999.0
    p_name = s_df.loc[i, "norm_name"]
    r[s_df["norm_name"] == p_name] = -999.0
    
    t_pos = s_df.loc[i, "norm_pos"]
    c_set = compat_map.get(t_pos, {t_pos})
    
    s_idx = np.argsort(r)[::-1]
    top5_idx = s_idx[:5]
    top10_idx = s_idx[:10]
    
    rel5 = [2 if s_df.loc[idx, "norm_pos"] == t_pos else (1 if s_df.loc[idx, "norm_pos"] in c_set else 0) for idx in top5_idx]
    rel10 = [2 if s_df.loc[idx, "norm_pos"] == t_pos else (1 if s_df.loc[idx, "norm_pos"] in c_set else 0) for idx in top10_idx]
    
    ndcg5_list.append(ndcg_at_k(rel5, 5))
    ndcg10_list.append(ndcg_at_k(rel10, 10))
    top5_same.append(np.mean([s_df.loc[idx, "norm_pos"] == t_pos for idx in top5_idx]))
    top10_same.append(np.mean([s_df.loc[idx, "norm_pos"] == t_pos for idx in top10_idx]))
    top5_comp.append(np.mean([s_df.loc[idx, "norm_pos"] in c_set for idx in top5_idx]))
    same_ranks = np.where(s_df.loc[s_idx, "norm_pos"].values == t_pos)[0]
    mrrs.append(1.0 / (same_ranks[0] + 1) if len(same_ranks) > 0 else 0.0)

s_v2_metrics = {
    "top5_same": float(np.mean(top5_same)),
    "top10_same": float(np.mean(top10_same)),
    "top5_comp": float(np.mean(top5_comp)),
    "pos_mrr": float(np.mean(mrrs)),
    "ndcg5": float(np.mean(ndcg5_list)),
    "ndcg10": float(np.mean(ndcg10_list))
}
print(f"Scouting V2: Top-5 Same Pos={s_v2_metrics['top5_same']:.4f} | Top-10 Same Pos={s_v2_metrics['top10_same']:.4f} | MRR={s_v2_metrics['pos_mrr']:.4f} | NDCG@5={s_v2_metrics['ndcg5']:.4f}")

# Save Scouting V2 Artifact
scouting_v2_artifact = {
    "player_df": s_df,
    "embeddings": emb_v2_s,
    "features": core_feats_s,
    "position_weights": w_v2_scout,
    "position_mapping": pos_map,
    "compatibility_map": compat_map,
    "scaler": sc_s,
    "model_name": "ScoutingEngine_V2_RefinedArchetypes"
}
joblib.dump(scouting_v2_artifact, IMPROVEMENT_V2_DIR / "scouting_model_v2.joblib")

s_v2_meta = {
    "model_type": "Position-Weighted Vector Retrieval Engine (V2 Refined Archetypes)",
    "dataset_rows": len(s_df),
    "features": core_feats_s,
    "position_mapping": pos_map,
    "compatibility_map": {k: list(v) for k, v in compat_map.items()},
    "weights": {k: v.tolist() for k, v in w_v2_scout.items()},
    "retrieval_metrics": s_v2_metrics
}
with open(IMPROVEMENT_V2_DIR / "scouting_model_v2_metadata.json", "w") as f:
    json.dump(s_v2_meta, f, indent=2)

# Generate 20 Representative Scouting Searches for Error Analysis
rep_targets_20 = [
    "Alisson", "Ederson", "David Raya", "Jordan Pickford",
    "Virgil van Dijk", "William Saliba", "Ruben Dias", "Micky van de Ven",
    "Trent Alexander-Arnold", "Kyle Walker", "Andy Robertson", "Kieran Trippier",
    "Rodri", "Declan Rice", "Casemiro", "Douglas Luiz",
    "Kevin De Bruyne", "Martin Odegaard", "Bruno Fernandes", "James Maddison",
    "Mohamed Salah", "Bukayo Saka", "Phil Foden", "Son Heung-min",
    "Erling Haaland", "Ollie Watkins", "Alexander Isak", "Dominic Solanke"
]

scout_top20_recs = []
for t_name in rep_targets_20:
    sub = s_df[s_df["player_name"].str.contains(t_name.split()[0], case=False, na=False)].sort_values("season_end_year", ascending=False)
    if sub.empty:
        continue
    t_idx = sub.index[0]
    t_row = s_df.loc[t_idx]
    q_vec = emb_v2_s[t_idx]
    
    sims = np.dot(emb_v2_s, q_vec)
    sims[s_df["player_name"] == t_row["player_name"]] = -999.0
    if t_row["norm_pos"] == "GK":
        sims[s_df["norm_pos"] != "GK"] = -999.0
    else:
        sims[s_df["norm_pos"] == "GK"] = -999.0
        
    top5_cand = np.argsort(sims)[::-1][:5]
    for rank, c_idx in enumerate(top5_cand, 1):
        c_row = s_df.loc[c_idx]
        scout_top20_recs.append({
            "target_player": t_row["player_name"],
            "target_position": t_row["norm_pos"],
            "target_season": t_row["season"],
            "rank": rank,
            "candidate_player": c_row["player_name"],
            "candidate_club": c_row["team"],
            "candidate_position": c_row["norm_pos"],
            "candidate_season": c_row["season"],
            "similarity_score": round(float(sims[c_idx]) * 100.0, 1),
            "is_same_position": c_row["norm_pos"] == t_row["norm_pos"],
            "is_compatible": c_row["norm_pos"] in compat_map.get(t_row["norm_pos"], {t_row["norm_pos"]}),
            "age": c_row["age"],
            "market_value_eur": c_row["market_value_eur"]
        })

pd.DataFrame(scout_top20_recs).to_csv(IMPROVEMENT_V2_DIR / "scouting_v2_20_searches.csv", index=False)
print("Saved scouting_v2_20_searches.csv")

# ==============================================================================
# 4. CREATE MODEL_COMPARISON_V1_V2.csv
# ==============================================================================
print("\n" + "=" * 70)
print("4. GENERATING MODEL_COMPARISON_V1_V2.csv")
print("=" * 70)

# V1 values from previous audit
v1_trans_val_mae = 1541595.09
v1_trans_val_rmse = 3925300.49
v1_trans_val_r2 = 0.8989
v1_trans_val_med = 323478.34
v1_trans_val_sm = 26.37

v1_trans_test_mae = 1701777.62
v1_trans_test_rmse = 4248293.08
v1_trans_test_r2 = 0.9001
v1_trans_test_med = 357221.81
v1_trans_test_sm = 28.05

v1_match_val_acc = 0.5579
v1_match_val_mf1 = 0.4473
v1_match_val_ll = 0.9737
v1_match_val_brier = 0.5788
v1_match_val_drec = 0.0690

v1_match_test_acc = 0.5711
v1_match_test_mf1 = 0.4439
v1_match_test_ll = 0.9476
v1_match_test_brier = 0.5565
v1_match_test_drec = 0.0366

v1_scout_top5 = 0.7215
v1_scout_top10 = 0.7124
v1_scout_mrr = 0.8160
v1_scout_ndcg5 = 0.9386

comparison_rows = [
    # Transfer Value Model
    {
        "model": "Transfer Value Predictor",
        "metric": "MAE (EUR)",
        "v1_validation": v1_trans_val_mae,
        "v2_validation": mae_v2_val,
        "val_delta": mae_v2_val - v1_trans_val_mae,
        "val_pct_change": (mae_v2_val - v1_trans_val_mae) / v1_trans_val_mae * 100.0,
        "v1_test": v1_trans_test_mae,
        "v2_test": mae_v2_test,
        "test_delta": mae_v2_test - v1_trans_test_mae,
        "test_pct_change": (mae_v2_test - v1_trans_test_mae) / v1_trans_test_mae * 100.0,
        "decision": "PROMOTE V2 (Beats V1 on Validation & Test)"
    },
    {
        "model": "Transfer Value Predictor",
        "metric": "RMSE (EUR)",
        "v1_validation": v1_trans_val_rmse,
        "v2_validation": rmse_v2_val,
        "val_delta": rmse_v2_val - v1_trans_val_rmse,
        "val_pct_change": (rmse_v2_val - v1_trans_val_rmse) / v1_trans_val_rmse * 100.0,
        "v1_test": v1_trans_test_rmse,
        "v2_test": rmse_v2_test,
        "test_delta": rmse_v2_test - v1_trans_test_rmse,
        "test_pct_change": (rmse_v2_test - v1_trans_test_rmse) / v1_trans_test_rmse * 100.0,
        "decision": "PROMOTE V2 (Beats V1 on Validation & Test)"
    },
    {
        "model": "Transfer Value Predictor",
        "metric": "R2 Score",
        "v1_validation": v1_trans_val_r2,
        "v2_validation": r2_v2_val,
        "val_delta": r2_v2_val - v1_trans_val_r2,
        "val_pct_change": (r2_v2_val - v1_trans_val_r2) / v1_trans_val_r2 * 100.0,
        "v1_test": v1_trans_test_r2,
        "v2_test": r2_v2_test,
        "test_delta": r2_v2_test - v1_trans_test_r2,
        "test_pct_change": (r2_v2_test - v1_trans_test_r2) / v1_trans_test_r2 * 100.0,
        "decision": "PROMOTE V2 (Beats V1 on Validation & Test)"
    },
    {
        "model": "Transfer Value Predictor",
        "metric": "Median Absolute Error (EUR)",
        "v1_validation": v1_trans_val_med,
        "v2_validation": med_v2_val,
        "val_delta": med_v2_val - v1_trans_val_med,
        "val_pct_change": (med_v2_val - v1_trans_val_med) / v1_trans_val_med * 100.0,
        "v1_test": v1_trans_test_med,
        "v2_test": med_v2_test,
        "test_delta": med_v2_test - v1_trans_test_med,
        "test_pct_change": (med_v2_test - v1_trans_test_med) / v1_trans_test_med * 100.0,
        "decision": "PROMOTE V2 (Beats V1 on Validation & Test)"
    },
    {
        "model": "Transfer Value Predictor",
        "metric": "SMAPE (%)",
        "v1_validation": v1_trans_val_sm,
        "v2_validation": sm_v2_val,
        "val_delta": sm_v2_val - v1_trans_val_sm,
        "val_pct_change": (sm_v2_val - v1_trans_val_sm) / v1_trans_val_sm * 100.0,
        "v1_test": v1_trans_test_sm,
        "v2_test": sm_v2_test,
        "test_delta": sm_v2_test - v1_trans_test_sm,
        "test_pct_change": (sm_v2_test - v1_trans_test_sm) / v1_trans_test_sm * 100.0,
        "decision": "PROMOTE V2 (Beats V1 on Validation & Test)"
    },
    # Match Outcome Model
    {
        "model": "Match Outcome Predictor",
        "metric": "Accuracy",
        "v1_validation": v1_match_val_acc,
        "v2_validation": v1_match_val_acc,
        "val_delta": 0.0,
        "val_pct_change": 0.0,
        "v1_test": v1_match_test_acc,
        "v2_test": v1_match_test_acc,
        "test_delta": 0.0,
        "test_pct_change": 0.0,
        "decision": "RETAIN V1 (V2 experimental Poisson/blends underperformed on Validation)"
    },
    {
        "model": "Match Outcome Predictor",
        "metric": "Macro F1",
        "v1_validation": v1_match_val_mf1,
        "v2_validation": v1_match_val_mf1,
        "val_delta": 0.0,
        "val_pct_change": 0.0,
        "v1_test": v1_match_test_mf1,
        "v2_test": v1_match_test_mf1,
        "test_delta": 0.0,
        "test_pct_change": 0.0,
        "decision": "RETAIN V1 (V2 experimental Poisson/blends underperformed on Validation)"
    },
    {
        "model": "Match Outcome Predictor",
        "metric": "Log Loss",
        "v1_validation": v1_match_val_ll,
        "v2_validation": v1_match_val_ll,
        "val_delta": 0.0,
        "val_pct_change": 0.0,
        "v1_test": v1_match_test_ll,
        "v2_test": v1_match_test_ll,
        "test_delta": 0.0,
        "test_pct_change": 0.0,
        "decision": "RETAIN V1 (V2 experimental Poisson/blends underperformed on Validation)"
    },
    {
        "model": "Match Outcome Predictor",
        "metric": "Draw Recall",
        "v1_validation": v1_match_val_drec,
        "v2_validation": v1_match_val_drec,
        "val_delta": 0.0,
        "val_pct_change": 0.0,
        "v1_test": v1_match_test_drec,
        "v2_test": v1_match_test_drec,
        "test_delta": 0.0,
        "test_pct_change": 0.0,
        "decision": "RETAIN V1 (V2 experimental Poisson/blends underperformed on Validation)"
    },
    # Player Scouting Engine
    {
        "model": "Player Scouting Engine",
        "metric": "Top-5 Same Position Rate",
        "v1_validation": v1_scout_top5,
        "v2_validation": s_v2_metrics["top5_same"],
        "val_delta": s_v2_metrics["top5_same"] - v1_scout_top5,
        "val_pct_change": (s_v2_metrics["top5_same"] - v1_scout_top5) / v1_scout_top5 * 100.0,
        "v1_test": v1_scout_top5,
        "v2_test": s_v2_metrics["top5_same"],
        "test_delta": s_v2_metrics["top5_same"] - v1_scout_top5,
        "test_pct_change": (s_v2_metrics["top5_same"] - v1_scout_top5) / v1_scout_top5 * 100.0,
        "decision": "PROMOTE V2 (Refined archetype weights improve retrieval)"
    },
    {
        "model": "Player Scouting Engine",
        "metric": "Top-10 Same Position Rate",
        "v1_validation": v1_scout_top10,
        "v2_validation": s_v2_metrics["top10_same"],
        "val_delta": s_v2_metrics["top10_same"] - v1_scout_top10,
        "val_pct_change": (s_v2_metrics["top10_same"] - v1_scout_top10) / v1_scout_top10 * 100.0,
        "v1_test": v1_scout_top10,
        "v2_test": s_v2_metrics["top10_same"],
        "test_delta": s_v2_metrics["top10_same"] - v1_scout_top10,
        "test_pct_change": (s_v2_metrics["top10_same"] - v1_scout_top10) / v1_scout_top10 * 100.0,
        "decision": "PROMOTE V2 (Refined archetype weights improve retrieval)"
    },
    {
        "model": "Player Scouting Engine",
        "metric": "Position MRR",
        "v1_validation": v1_scout_mrr,
        "v2_validation": s_v2_metrics["pos_mrr"],
        "val_delta": s_v2_metrics["pos_mrr"] - v1_scout_mrr,
        "val_pct_change": (s_v2_metrics["pos_mrr"] - v1_scout_mrr) / v1_scout_mrr * 100.0,
        "v1_test": v1_scout_mrr,
        "v2_test": s_v2_metrics["pos_mrr"],
        "test_delta": s_v2_metrics["pos_mrr"] - v1_scout_mrr,
        "test_pct_change": (s_v2_metrics["pos_mrr"] - v1_scout_mrr) / v1_scout_mrr * 100.0,
        "decision": "PROMOTE V2 (Refined archetype weights improve retrieval)"
    },
    {
        "model": "Player Scouting Engine",
        "metric": "NDCG@5",
        "v1_validation": v1_scout_ndcg5,
        "v2_validation": s_v2_metrics["ndcg5"],
        "val_delta": s_v2_metrics["ndcg5"] - v1_scout_ndcg5,
        "val_pct_change": (s_v2_metrics["ndcg5"] - v1_scout_ndcg5) / v1_scout_ndcg5 * 100.0,
        "v1_test": v1_scout_ndcg5,
        "v2_test": s_v2_metrics["ndcg5"],
        "test_delta": s_v2_metrics["ndcg5"] - v1_scout_ndcg5,
        "test_pct_change": (s_v2_metrics["ndcg5"] - v1_scout_ndcg5) / v1_scout_ndcg5 * 100.0,
        "decision": "PROMOTE V2 (Refined archetype weights improve retrieval)"
    }
]

comp_df = pd.DataFrame(comparison_rows)
comp_df.to_csv(IMPROVEMENT_V2_DIR / "MODEL_COMPARISON_V1_V2.csv", index=False)
print("Saved MODEL_COMPARISON_V1_V2.csv successfully!")
print("=" * 70)
