import sys
from pathlib import Path
sys.path.insert(0, str(Path(".").resolve()))
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

df = pd.read_csv("data/processed/scouting/player_scouting_dataset.csv")

pos_map = {
    "GK": "GK",
    "DF": "DF",
    "DF,MF": "DM/WB",
    "MF,DF": "DM/WB",
    "DF,FW": "DM/WB",
    "FW,DF": "DM/WB",
    "MF": "CM",
    "MF,FW": "AM/W",
    "FW,MF": "AM/W",
    "FW": "FW"
}
compat_map = {
    "GK": {"GK"},
    "DF": {"DF", "DM/WB"},
    "DM/WB": {"DF", "DM/WB", "CM"},
    "CM": {"DM/WB", "CM", "AM/W"},
    "AM/W": {"CM", "AM/W", "FW"},
    "FW": {"AM/W", "FW"}
}
df["norm_pos"] = df["position"].map(pos_map).fillna("CM")

core_feats = [
    "goals_per90", "assists_per90", "xG_per90", "npxG_per90", "xAG_per90",
    "shots_per90", "key_passes_per90", "pass_completion_pct",
    "progressive_passes_per90", "progressive_carries_per90",
    "successful_takeons_per90", "tackles_won_per90", "interceptions_per90",
    "blocks_per90", "clearances_per90", "aerial_won_pct"
]

scaler = StandardScaler()
X_core_std = scaler.fit_transform(df[core_feats].fillna(df[core_feats].median()))

# Tactical Archetype Indices
# 0: goals, 1: assists, 2: xG, 3: npxG, 4: xAG, 5: shots, 6: key_passes, 7: pass_completion,
# 8: prog_passes, 9: prog_carries, 10: takeons, 11: tackles, 12: interceptions, 13: blocks, 14: clearances, 15: aerial
box_threat = (X_core_std[:, 0] + X_core_std[:, 2] + 0.5 * X_core_std[:, 5]) / 2.5
creation = (X_core_std[:, 1] + X_core_std[:, 4] + X_core_std[:, 6]) / 3.0
progression = (X_core_std[:, 8] + X_core_std[:, 9] + 0.5 * X_core_std[:, 10]) / 2.5
def_disruption = (X_core_std[:, 11] + X_core_std[:, 12] + 0.5 * X_core_std[:, 13]) / 2.5
aerial_command = (X_core_std[:, 15] + 0.5 * X_core_std[:, 14]) / 1.5
ball_security = X_core_std[:, 7]

archetype_indices = np.column_stack([box_threat, creation, progression, def_disruption, aerial_command, ball_security])
X_expanded = np.column_stack([X_core_std, archetype_indices])

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

def evaluate_embeddings(emb_matrix):
    sim = cosine_similarity(emb_matrix)
    is_gk = (df["norm_pos"] == "GK").values
    for i in range(len(df)):
        if is_gk[i]:
            sim[i, ~is_gk] = -1.0
        else:
            sim[i, is_gk] = -1.0
            
    top5_same, top10_same = [], []
    top5_comp, top10_comp = [], []
    mrrs = []
    ndcg5_list, ndcg10_list = [], []
    
    n = len(df)
    for i in range(n):
        r = sim[i].copy()
        r[i] = -999.0
        
        # Exclude self player
        p_name = df.loc[i, "norm_name"]
        r[df["norm_name"] == p_name] = -999.0
        
        t_pos = df.loc[i, "norm_pos"]
        c_set = compat_map.get(t_pos, {t_pos})
        
        s_idx = np.argsort(r)[::-1]
        top5_idx = s_idx[:5]
        top10_idx = s_idx[:10]
        
        rel5 = [2 if df.loc[idx, "norm_pos"] == t_pos else (1 if df.loc[idx, "norm_pos"] in c_set else 0) for idx in top5_idx]
        rel10 = [2 if df.loc[idx, "norm_pos"] == t_pos else (1 if df.loc[idx, "norm_pos"] in c_set else 0) for idx in top10_idx]
        
        ndcg5_list.append(ndcg_at_k(rel5, 5))
        ndcg10_list.append(ndcg_at_k(rel10, 10))
        top5_same.append(np.mean([df.loc[idx, "norm_pos"] == t_pos for idx in top5_idx]))
        top10_same.append(np.mean([df.loc[idx, "norm_pos"] == t_pos for idx in top10_idx]))
        top5_comp.append(np.mean([df.loc[idx, "norm_pos"] in c_set for idx in top5_idx]))
        top10_comp.append(np.mean([df.loc[idx, "norm_pos"] in c_set for idx in top10_idx]))
        same_ranks = np.where(df.loc[s_idx, "norm_pos"].values == t_pos)[0]
        mrrs.append(1.0 / (same_ranks[0] + 1) if len(same_ranks) > 0 else 0.0)
        
    return {
        "top5_same": float(np.mean(top5_same)),
        "top10_same": float(np.mean(top10_same)),
        "top5_comp": float(np.mean(top5_comp)),
        "pos_mrr": float(np.mean(mrrs)),
        "ndcg5": float(np.mean(ndcg5_list)),
        "ndcg10": float(np.mean(ndcg10_list))
    }

# 1. V1 Champion Weights
w_v1 = {
    "DF": np.array([0.2, 0.4, 0.2, 0.2, 0.4, 0.2, 0.6, 1.4, 1.1, 0.9, 0.5, 1.8, 1.8, 1.6, 2.0, 1.8]),
    "DM/WB": np.array([0.4, 0.6, 0.4, 0.4, 0.7, 0.5, 1.1, 1.4, 1.6, 1.4, 1.1, 1.6, 1.6, 1.3, 1.1, 1.1]),
    "CM": np.array([0.5, 0.9, 0.5, 0.5, 1.1, 0.6, 1.5, 1.6, 1.7, 1.6, 1.1, 1.4, 1.4, 1.0, 0.7, 0.9]),
    "AM/W": np.array([1.5, 1.5, 1.5, 1.5, 1.6, 1.5, 1.6, 1.1, 1.2, 1.6, 1.8, 0.7, 0.7, 0.5, 0.3, 0.5]),
    "FW": np.array([1.8, 1.1, 1.8, 1.8, 1.1, 1.7, 1.1, 0.9, 0.7, 1.1, 1.3, 0.5, 0.4, 0.4, 0.3, 1.5]),
    "GK": np.array([0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 2.0, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 2.0])
}
X_v1_w = np.zeros_like(X_core_std)
for pos, w in w_v1.items():
    mask = (df["norm_pos"] == pos).values
    X_v1_w[mask] = X_core_std[mask] * np.sqrt(w)
norms_v1 = np.linalg.norm(X_v1_w, axis=1, keepdims=True)
norms_v1[norms_v1 == 0] = 1.0
emb_v1 = X_v1_w / norms_v1

res_v1 = evaluate_embeddings(emb_v1)
print("=== V1 SCOUTING CHAMPION BENCHMARK ===")
print(f"Top-5 Same Pos: {res_v1['top5_same']:.4f} | Top-10 Same Pos: {res_v1['top10_same']:.4f} | MRR: {res_v1['pos_mrr']:.4f} | NDCG@5: {res_v1['ndcg5']:.4f}")

# 2. V2 Refined Archetype Weights (Optimized for tactical roles)
w_v2 = {
    "DF": np.array([0.1, 0.3, 0.1, 0.1, 0.3, 0.1, 0.5, 1.6, 1.2, 1.0, 0.4, 2.0, 2.0, 1.8, 2.2, 2.0]),
    "DM/WB": np.array([0.3, 0.6, 0.3, 0.3, 0.8, 0.4, 1.2, 1.6, 1.8, 1.6, 1.2, 1.8, 1.8, 1.4, 1.2, 1.2]),
    "CM": np.array([0.4, 1.0, 0.4, 0.4, 1.2, 0.5, 1.6, 1.8, 2.0, 1.8, 1.2, 1.5, 1.5, 1.0, 0.6, 0.8]),
    "AM/W": np.array([1.6, 1.6, 1.6, 1.6, 1.8, 1.6, 1.8, 1.1, 1.3, 1.8, 2.0, 0.6, 0.6, 0.4, 0.2, 0.4]),
    "FW": np.array([2.0, 1.1, 2.0, 2.0, 1.1, 1.8, 1.1, 0.8, 0.6, 1.1, 1.4, 0.4, 0.3, 0.3, 0.2, 1.6]),
    "GK": np.array([0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 2.5, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 2.5])
}
X_v2_w = np.zeros_like(X_core_std)
for pos, w in w_v2.items():
    mask = (df["norm_pos"] == pos).values
    X_v2_w[mask] = X_core_std[mask] * np.sqrt(w)
norms_v2 = np.linalg.norm(X_v2_w, axis=1, keepdims=True)
norms_v2[norms_v2 == 0] = 1.0
emb_v2 = X_v2_w / norms_v2

res_v2 = evaluate_embeddings(emb_v2)
print("\n=== V2 REFINED ARCHETYPE WEIGHTS ===")
print(f"Top-5 Same Pos: {res_v2['top5_same']:.4f} | Top-10 Same Pos: {res_v2['top10_same']:.4f} | MRR: {res_v2['pos_mrr']:.4f} | NDCG@5: {res_v2['ndcg5']:.4f}")

# 3. V2 with Composite Tactical Indices (22 dimensions)
arch_weights = {
    "DF": np.array([0.1, 0.5, 1.0, 2.0, 2.0, 1.4]),
    "DM/WB": np.array([0.3, 0.8, 1.8, 1.8, 1.2, 1.5]),
    "CM": np.array([0.4, 1.4, 2.0, 1.4, 0.8, 1.6]),
    "AM/W": np.array([1.6, 1.8, 1.8, 0.6, 0.4, 1.2]),
    "FW": np.array([2.0, 1.2, 1.1, 0.3, 1.6, 0.8]),
    "GK": np.array([0.0, 0.0, 0.0, 0.0, 2.0, 2.0])
}
X_exp_w = np.zeros_like(X_expanded)
for pos, w in w_v2.items():
    mask = (df["norm_pos"] == pos).values
    core_w = np.sqrt(w)
    idx_w = np.sqrt(arch_weights[pos])
    total_w = np.concatenate([core_w, idx_w])
    X_exp_w[mask] = X_expanded[mask] * total_w

norms_exp = np.linalg.norm(X_exp_w, axis=1, keepdims=True)
norms_exp[norms_exp == 0] = 1.0
emb_exp = X_exp_w / norms_exp

res_exp = evaluate_embeddings(emb_exp)
print("\n=== V2 WITH 22-DIM TACTICAL COMPOSITE INDICES ===")
print(f"Top-5 Same Pos: {res_exp['top5_same']:.4f} | Top-10 Same Pos: {res_exp['top10_same']:.4f} | MRR: {res_exp['pos_mrr']:.4f} | NDCG@5: {res_exp['ndcg5']:.4f}")
