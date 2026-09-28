import sys
from pathlib import Path
sys.path.insert(0, str(Path(".").resolve()))
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score, median_absolute_error,
    accuracy_score, precision_score, recall_score, f1_score, log_loss, confusion_matrix
)
from sklearn.metrics.pairwise import cosine_similarity
from scripts.train_improved_transfer import smape
from scripts.train_improved_match import compute_sequential_elo

print("=" * 70)
print("PHASE 0: ESTABLISH V1 CHAMPION VALIDATION BENCHMARKS (2022/23)")
print("=" * 70)

# 1. TRANSFER VALUE V1 MODEL ON VALIDATION (2022/23)
print("\n--- 1. Transfer Value Model V1 on Validation Split ---")
t_df = pd.read_csv("data/processed/transfer/transfer_value_dataset.csv")
t_df["minutes_per_appearance"] = t_df["minutes_played"] / (t_df["appearances"] + 1e-5)
t_df["cards_per90"] = (t_df["yellow_cards"] + 2 * t_df["red_cards"]) / (t_df["minutes_played"] / 90.0 + 1e-5)

val_t = t_df[t_df["split_group"] == "val"].copy().reset_index(drop=True)
y_val_t = val_t["market_value_eur"].values

impr_t_pipe = joblib.load("models/improvement/transfer_value_model_improved.joblib")
impr_t_cols = [
    "age", "height_cm", "contract_years_left", "club_market_value_eur",
    "minutes_played", "appearances", "goals", "assists", "yellow_cards",
    "red_cards", "goals_per90", "assists_per90", "shots_per90", "npxg_per90",
    "prev_market_value_eur", "minutes_per_appearance", "cards_per90",
    "primary_position", "sub_position", "preferred_foot"
]

preds_val_t = np.clip(np.expm1(impr_t_pipe.predict(val_t[impr_t_cols])), 0, None)

val_t_mae = mean_absolute_error(y_val_t, preds_val_t)
val_t_rmse = np.sqrt(mean_squared_error(y_val_t, preds_val_t))
val_t_r2 = r2_score(y_val_t, preds_val_t)
val_t_medae = median_absolute_error(y_val_t, preds_val_t)
val_t_smape = smape(y_val_t, preds_val_t)

transfer_v1_val_bench = {
    "MAE": float(val_t_mae),
    "RMSE": float(val_t_rmse),
    "R2": float(val_t_r2),
    "Median_AE": float(val_t_medae),
    "SMAPE": float(val_t_smape)
}
print(f"MAE:       EUR {val_t_mae:,.2f} (EUR {val_t_mae/1e6:.3f}M)")
print(f"RMSE:      EUR {val_t_rmse:,.2f} (EUR {val_t_rmse/1e6:.3f}M)")
print(f"R2:        {val_t_r2:.4f}")
print(f"Median AE: EUR {val_t_medae:,.2f} (EUR {val_t_medae/1e6:.3f}M)")
print(f"SMAPE:     {val_t_smape:.2f}%")

# 2. MATCH OUTCOME V1 MODEL ON VALIDATION (2022/23)
print("\n--- 2. Match Outcome Model V1 on Validation Split ---")
m_df = pd.read_csv("data/processed/matches/match_outcome_dataset.csv")
m_df["date"] = pd.to_datetime(m_df["date"])
df_elo, _ = compute_sequential_elo(m_df)
val_m = df_elo[df_elo["split_group"] == "val"].copy().reset_index(drop=True)

c_to_i = {"H": 0, "D": 1, "A": 2}
y_val_m = val_m["match_result"].map(c_to_i).values
y_val_m_oh = np.eye(3)[y_val_m]

impr_m_bundle = joblib.load("models/improvement/match_outcome_model_improved.joblib")
m_pipeline = impr_m_bundle["pipeline"]
m_features = impr_m_bundle["features"]

probs_val_m = m_pipeline.predict_proba(val_m[m_features].values)
preds_val_m = m_pipeline.predict(val_m[m_features].values)

val_m_acc = accuracy_score(y_val_m, preds_val_m)
val_m_mf1 = f1_score(y_val_m, preds_val_m, average="macro", zero_division=0)
val_m_wf1 = f1_score(y_val_m, preds_val_m, average="weighted", zero_division=0)
val_m_ll = log_loss(y_val_m, probs_val_m)
val_m_brier = np.mean(np.sum((probs_val_m - y_val_m_oh) ** 2, axis=1))

p_class = precision_score(y_val_m, preds_val_m, average=None, zero_division=0)
r_class = recall_score(y_val_m, preds_val_m, average=None, zero_division=0)
f1_class = f1_score(y_val_m, preds_val_m, average=None, zero_division=0)
val_m_cm = confusion_matrix(y_val_m, preds_val_m, labels=[0, 1, 2])

match_v1_val_bench = {
    "Accuracy": float(val_m_acc),
    "Macro_F1": float(val_m_mf1),
    "Weighted_F1": float(val_m_wf1),
    "Log_Loss": float(val_m_ll),
    "Brier_Score": float(val_m_brier),
    "Home": {"Precision": float(p_class[0]), "Recall": float(r_class[0]), "F1": float(f1_class[0])},
    "Draw": {"Precision": float(p_class[1]), "Recall": float(r_class[1]), "F1": float(f1_class[1])},
    "Away": {"Precision": float(p_class[2]), "Recall": float(r_class[2]), "F1": float(f1_class[2])},
    "Confusion_Matrix": val_m_cm.tolist()
}
print(f"Accuracy:    {val_m_acc:.4f}")
print(f"Macro F1:    {val_m_mf1:.4f}")
print(f"Weighted F1: {val_m_wf1:.4f}")
print(f"Log Loss:    {val_m_ll:.4f}")
print(f"Brier Score: {val_m_brier:.4f}")
print("Per-class:")
print(f"  Home: Precision={p_class[0]:.4f}, Recall={r_class[0]:.4f}, F1={f1_class[0]:.4f}")
print(f"  Draw: Precision={p_class[1]:.4f}, Recall={r_class[1]:.4f}, F1={f1_class[1]:.4f}")
print(f"  Away: Precision={p_class[2]:.4f}, Recall={r_class[2]:.4f}, F1={f1_class[2]:.4f}")
print("Confusion Matrix:\n", val_m_cm)

# 3. SCOUTING V1 MODEL ON VALIDATION / REFERENCE
print("\n--- 3. Player Scouting Model V1 Benchmark ---")
s_art = joblib.load("models/improvement/scouting_model_improved.joblib")
s_df = s_art["player_df"].copy().reset_index(drop=True)
s_emb = s_art["embeddings"]
pos_map = s_art["position_mapping"]
compat_map = s_art["compatibility_map"]
s_df["norm_pos"] = s_df["position"].map(pos_map).fillna("CM")

sim_s = cosine_similarity(s_emb)
is_gk = (s_df["norm_pos"] == "GK").values
for i in range(len(s_df)):
    if is_gk[i]:
        sim_s[i, ~is_gk] = -1.0
    else:
        sim_s[i, is_gk] = -1.0

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

top5_same, top10_same, top5_comp, top10_comp, mrrs, ndcg5_list, ndcg10_list = [], [], [], [], [], [], []
n_s = len(s_df)
for i in range(n_s):
    r = sim_s[i].copy()
    r[i] = -999.0
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
    top10_comp.append(np.mean([s_df.loc[idx, "norm_pos"] in c_set for idx in top10_idx]))
    
    same_ranks = np.where(s_df.loc[s_idx, "norm_pos"].values == t_pos)[0]
    mrrs.append(1.0 / (same_ranks[0] + 1) if len(same_ranks) > 0 else 0.0)

scouting_v1_bench = {
    "Top5_Same_Position_Rate": float(np.mean(top5_same)),
    "Top10_Same_Position_Rate": float(np.mean(top10_same)),
    "Top5_Compatible_Position_Rate": float(np.mean(top5_comp)),
    "Top10_Compatible_Position_Rate": float(np.mean(top10_comp)),
    "Position_MRR": float(np.mean(mrrs)),
    "NDCG@5": float(np.mean(ndcg5_list)),
    "NDCG@10": float(np.mean(ndcg10_list)),
    "Goalkeeper_Isolation": "100% Strict"
}

print(f"Top-5 Same Pos Rate:       {scouting_v1_bench['Top5_Same_Position_Rate']:.4f}")
print(f"Top-10 Same Pos Rate:      {scouting_v1_bench['Top10_Same_Position_Rate']:.4f}")
print(f"Top-5 Compatible Pos Rate: {scouting_v1_bench['Top5_Compatible_Position_Rate']:.4f}")
print(f"Top-10 Compatible Pos Rate:{scouting_v1_bench['Top10_Compatible_Position_Rate']:.4f}")
print(f"Position MRR:              {scouting_v1_bench['Position_MRR']:.4f}")
print(f"NDCG@5:                    {scouting_v1_bench['NDCG@5']:.4f}")
print(f"NDCG@10:                   {scouting_v1_bench['NDCG@10']:.4f}")

benchmark_payload = {
    "transfer": transfer_v1_val_bench,
    "match": match_v1_val_bench,
    "scouting": scouting_v1_bench
}
with open("models/improvement_v2/v1_validation_benchmarks.json", "w") as f:
    json.dump(benchmark_payload, f, indent=2)

print("\nSaved models/improvement_v2/v1_validation_benchmarks.json successfully!")
print("=" * 70)
