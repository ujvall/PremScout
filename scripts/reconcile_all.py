import sys
from pathlib import Path
sys.path.insert(0, str(Path(".").resolve()))
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score, median_absolute_error,
    accuracy_score, f1_score, log_loss
)
from sklearn.metrics.pairwise import cosine_similarity
from scripts.train_improved_transfer import smape
from scripts.train_improved_match import compute_sequential_elo

print("=" * 80)
print("PREMAI V2 METRIC RECONCILIATION SCRIPT")
print("=" * 80)

# ==============================================================================
# 1. TRANSFER VALUE PREDICTOR RECONCILIATION
# ==============================================================================
print("\n--- 1. TRANSFER VALUE PREDICTOR ---")
df_t = pd.read_csv("data/processed/transfer/transfer_value_dataset.csv")
df_t = df_t.sort_values(["player_id", "season"]).reset_index(drop=True)

# Engineer features
df_t["minutes_per_appearance"] = df_t["minutes_played"] / (df_t["appearances"] + 1e-5)
df_t["cards_per90"] = (df_t["yellow_cards"] + 2 * df_t["red_cards"]) / (df_t["minutes_played"] / 90.0 + 1e-5)
df_t["age_x_minutes"] = (df_t["age"] * df_t["minutes_played"]) / 1000.0
df_t["age_x_goals"] = df_t["age"] * df_t["goals_per90"]
df_t["contract_x_prev_val"] = (df_t["contract_years_left"].fillna(2.0) * df_t["prev_market_value_eur"]) / 1e6
df_t["youth_starter_premium"] = np.where((df_t["age"] <= 23) & (df_t["minutes_per_appearance"] >= 65), 1.0, 0.0)
df_t["veteran_depreciation"] = np.maximum(0.0, df_t["age"] - 30.0)

df_t["lag_prev_val"] = df_t.groupby("player_id")["prev_market_value_eur"].shift(1)
df_t["val_growth_abs"] = (df_t["prev_market_value_eur"] - df_t["lag_prev_val"]).fillna(0.0)
df_t["val_growth_pct"] = np.clip((df_t["prev_market_value_eur"] - df_t["lag_prev_val"]) / (df_t["lag_prev_val"] + 1e5), -1.0, 5.0).fillna(0.0)

val_t = df_t[df_t["split_group"] == "val"].copy()
test_t = df_t[df_t["split_group"] == "test"].copy()

y_val_t = val_t["market_value_eur"].values
y_test_t = test_t["market_value_eur"].values

# V1 Transfer
v1_bundle_t = joblib.load("models/improvement/transfer_value_model_improved.joblib")
v1_cols_num = [
    "age", "height_cm", "contract_years_left", "club_market_value_eur",
    "minutes_played", "appearances", "goals", "assists", "yellow_cards",
    "red_cards", "goals_per90", "assists_per90", "shots_per90", "npxg_per90",
    "prev_market_value_eur", "minutes_per_appearance", "cards_per90"
]
v1_cols_cat = ["primary_position", "sub_position", "preferred_foot"]
v1_cols_t = v1_cols_num + v1_cols_cat

v1_pred_val_t = np.clip(np.expm1(v1_bundle_t.predict(val_t[v1_cols_t])), 0, None)
v1_pred_test_t = np.clip(np.expm1(v1_bundle_t.predict(test_t[v1_cols_t])), 0, None)

# V2 Transfer
v2_bundle_t = joblib.load("models/improvement_v2/transfer_value_model_v2.joblib")
prep_t = v2_bundle_t["preprocessor"]
xgb_t = v2_bundle_t["xgb_model"]
hgb_t = v2_bundle_t["hgb_model"]
w_t = v2_bundle_t["ensemble_weights"]
cols_v2_t = v2_bundle_t["features"]

X_val_t = prep_t.transform(val_t[cols_v2_t])
X_test_t = prep_t.transform(test_t[cols_v2_t])

v2_pred_val_t = w_t["xgb"] * np.clip(np.expm1(xgb_t.predict(X_val_t)), 0, None) + w_t["hgb"] * np.clip(np.expm1(hgb_t.predict(X_val_t)), 0, None)
v2_pred_test_t = w_t["xgb"] * np.clip(np.expm1(xgb_t.predict(X_test_t)), 0, None) + w_t["hgb"] * np.clip(np.expm1(hgb_t.predict(X_test_t)), 0, None)

print("TRANSFER VALIDATION (2022/23):")
print(f"  V1: MAE = €{mean_absolute_error(y_val_t, v1_pred_val_t):,.2f} | RMSE = €{np.sqrt(mean_squared_error(y_val_t, v1_pred_val_t)):,.2f} | R² = {r2_score(y_val_t, v1_pred_val_t):.4f} | MedAE = €{median_absolute_error(y_val_t, v1_pred_val_t):,.2f} | SMAPE = {smape(y_val_t, v1_pred_val_t):.2f}%")
print(f"  V2: MAE = €{mean_absolute_error(y_val_t, v2_pred_val_t):,.2f} | RMSE = €{np.sqrt(mean_squared_error(y_val_t, v2_pred_val_t)):,.2f} | R² = {r2_score(y_val_t, v2_pred_val_t):.4f} | MedAE = €{median_absolute_error(y_val_t, v2_pred_val_t):,.2f} | SMAPE = {smape(y_val_t, v2_pred_val_t):.2f}%")

print("TRANSFER TEST (2023/24 Holdout):")
print(f"  V1: MAE = €{mean_absolute_error(y_test_t, v1_pred_test_t):,.2f} | RMSE = €{np.sqrt(mean_squared_error(y_test_t, v1_pred_test_t)):,.2f} | R² = {r2_score(y_test_t, v1_pred_test_t):.4f} | MedAE = €{median_absolute_error(y_test_t, v1_pred_test_t):,.2f} | SMAPE = {smape(y_test_t, v1_pred_test_t):.2f}%")
print(f"  V2: MAE = €{mean_absolute_error(y_test_t, v2_pred_test_t):,.2f} | RMSE = €{np.sqrt(mean_squared_error(y_test_t, v2_pred_test_t)):,.2f} | R² = {r2_score(y_test_t, v2_pred_test_t):.4f} | MedAE = €{median_absolute_error(y_test_t, v2_pred_test_t):,.2f} | SMAPE = {smape(y_test_t, v2_pred_test_t):.2f}%")


# ==============================================================================
# 2. MATCH OUTCOME PREDICTOR RECONCILIATION
# ==============================================================================
print("\n--- 2. MATCH OUTCOME PREDICTOR ---")
raw_m = pd.read_csv("data/processed/matches/match_outcome_dataset.csv")
raw_m["date"] = pd.to_datetime(raw_m["date"])
elo_m, _ = compute_sequential_elo(raw_m)

val_m = elo_m[elo_m["split_group"] == "val"].copy()
test_m = elo_m[elo_m["split_group"] == "test"].copy()

c_to_i = {"H": 0, "D": 1, "A": 2}
y_val_m = val_m["match_result"].map(c_to_i).values
y_test_m = test_m["match_result"].map(c_to_i).values

v1_bundle_m = joblib.load("models/improvement/match_outcome_model_improved.joblib")
pipe_m = v1_bundle_m["pipeline"]
cols_m = v1_bundle_m["features"]

val_m_probs = pipe_m.predict_proba(val_m[cols_m])
val_m_preds = pipe_m.predict(val_m[cols_m])

test_m_probs = pipe_m.predict_proba(test_m[cols_m])
test_m_preds = pipe_m.predict(test_m[cols_m])

print("MATCH VALIDATION (2022/23):")
print(f"  V1: Acc = {accuracy_score(y_val_m, val_m_preds)*100:.2f}% | Macro F1 = {f1_score(y_val_m, val_m_preds, average='macro'):.4f} | Log Loss = {log_loss(y_val_m, val_m_probs):.4f}")
print("MATCH TEST (2023/24 Holdout):")
print(f"  V1: Acc = {accuracy_score(y_test_m, test_m_preds)*100:.2f}% | Macro F1 = {f1_score(y_test_m, test_m_preds, average='macro'):.4f} | Log Loss = {log_loss(y_test_m, test_m_probs):.4f}")


# ==============================================================================
# 3. PLAYER SCOUTING ENGINE RECONCILIATION
# ==============================================================================
print("\n--- 3. PLAYER SCOUTING ENGINE ---")

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

v1_art_s = joblib.load("models/improvement/scouting_model_improved.joblib")
v2_art_s = joblib.load("models/improvement_v2/scouting_model_v2.joblib")

df_s = v1_art_s["player_df"].copy().reset_index(drop=True)
pos_map_s = v1_art_s["position_mapping"]
compat_map_s = v1_art_s["compatibility_map"]
df_s["norm_pos"] = df_s["position"].map(pos_map_s).fillna("CM")

emb_v1_s = v1_art_s["embeddings"]
emb_v2_s = v2_art_s["embeddings"]

def run_scouting_eval(emb, exclude_mode):
    sim = cosine_similarity(emb)
    is_gk = (df_s["norm_pos"] == "GK").values
    for i in range(len(df_s)):
        if is_gk[i]:
            sim[i, ~is_gk] = -1.0
        else:
            sim[i, is_gk] = -1.0
            
    top5_same, top10_same, mrrs, ndcg5_l = [], [], [], []
    n = len(df_s)
    for i in range(n):
        r = sim[i].copy()
        r[i] = -999.0
        if exclude_mode == "player":
            p_name = df_s.loc[i, "norm_name"]
            r[df_s["norm_name"] == p_name] = -999.0
            
        t_pos = df_s.loc[i, "norm_pos"]
        c_set = compat_map_s.get(t_pos, {t_pos})
        
        s_idx = np.argsort(r)[::-1]
        top5_idx = s_idx[:5]
        top10_idx = s_idx[:10]
        
        rel5 = [2 if df_s.loc[idx, "norm_pos"] == t_pos else (1 if df_s.loc[idx, "norm_pos"] in c_set else 0) for idx in top5_idx]
        
        ndcg5_l.append(ndcg_at_k(rel5, 5))
        top5_same.append(np.mean([df_s.loc[idx, "norm_pos"] == t_pos for idx in top5_idx]))
        top10_same.append(np.mean([df_s.loc[idx, "norm_pos"] == t_pos for idx in top10_idx]))
        
        same_ranks = np.where(df_s.loc[s_idx, "norm_pos"].values == t_pos)[0]
        mrrs.append(1.0 / (same_ranks[0] + 1) if len(same_ranks) > 0 else 0.0)
        
    return {
        "top5_same": np.mean(top5_same) * 100,
        "top10_same": np.mean(top10_same) * 100,
        "mrr": np.mean(mrrs),
        "ndcg5": np.mean(ndcg5_l)
    }

print("SCOUTING PROTOCOL A (Self Row Excluded Only, Allows other seasons of same player):")
s_v1_row = run_scouting_eval(emb_v1_s, "row")
s_v2_row = run_scouting_eval(emb_v2_s, "row")
print(f"  V1: Top-5 Same = {s_v1_row['top5_same']:.2f}% | Top-10 Same = {s_v1_row['top10_same']:.2f}% | MRR = {s_v1_row['mrr']:.4f} | NDCG@5 = {s_v1_row['ndcg5']:.4f}")
print(f"  V2: Top-5 Same = {s_v2_row['top5_same']:.2f}% | Top-10 Same = {s_v2_row['top10_same']:.2f}% | MRR = {s_v2_row['mrr']:.4f} | NDCG@5 = {s_v2_row['ndcg5']:.4f}")

print("\nSCOUTING PROTOCOL B (Strict Cross-Player Replacement, All seasons of query player excluded):")
s_v1_plr = run_scouting_eval(emb_v1_s, "player")
s_v2_plr = run_scouting_eval(emb_v2_s, "player")
print(f"  V1: Top-5 Same = {s_v1_plr['top5_same']:.2f}% | Top-10 Same = {s_v1_plr['top10_same']:.2f}% | MRR = {s_v1_plr['mrr']:.4f} | NDCG@5 = {s_v1_plr['ndcg5']:.4f}")
print(f"  V2: Top-5 Same = {s_v2_plr['top5_same']:.2f}% | Top-10 Same = {s_v2_plr['top10_same']:.2f}% | MRR = {s_v2_plr['mrr']:.4f} | NDCG@5 = {s_v2_plr['ndcg5']:.4f}")

print("\n" + "=" * 80)
