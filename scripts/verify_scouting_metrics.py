import sys
from pathlib import Path
sys.path.insert(0, str(Path(".").resolve()))
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# 1. Load artifacts
base_art = joblib.load("models/baseline/scouting/scouting_model.joblib")
impr_art = joblib.load("models/improvement/scouting_model_improved.joblib")

df = impr_art["player_df"].copy().reset_index(drop=True)
base_emb = base_art["embeddings"]
impr_emb = impr_art["embeddings"]

pos_map = impr_art["position_mapping"]
compat_map = impr_art["compatibility_map"]

df["norm_pos"] = df["position"].map(pos_map).fillna("CM")

print(f"Scouting total rows: {len(df)}")
print(f"Base embeddings shape: {base_emb.shape}")
print(f"Impr embeddings shape: {impr_emb.shape}")

# Precompute cosine similarities
sim_base_raw = cosine_similarity(base_emb)
sim_impr_raw = cosine_similarity(impr_emb)

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

def evaluate_retrieval(sim_mat, apply_gk_isolation=False):
    top5_same, top10_same = [], []
    top5_comp, top10_comp = [], []
    mrrs = []
    ndcg5_list, ndcg10_list = [], []
    
    n = len(df)
    for i in range(n):
        r = sim_mat[i].copy()
        r[i] = -999.0  # exclude self
        
        # Exclude other seasons of the same player to measure player replacement retrieval, not self-retrieval
        p_name = df.loc[i, "norm_name"]
        same_player_mask = (df["norm_name"] == p_name).values
        r[same_player_mask] = -999.0
        
        target_pos = df.loc[i, "norm_pos"]
        comp_set = compat_map.get(target_pos, {target_pos})
        
        if apply_gk_isolation:
            if target_pos == "GK":
                r[df["norm_pos"] != "GK"] = -999.0
            else:
                r[df["norm_pos"] == "GK"] = -999.0
                
        sorted_idx = np.argsort(r)[::-1]
        top5_idx = sorted_idx[:5]
        top10_idx = sorted_idx[:10]
        
        rel5 = [2 if df.loc[idx, "norm_pos"] == target_pos else (1 if df.loc[idx, "norm_pos"] in comp_set else 0) for idx in top5_idx]
        rel10 = [2 if df.loc[idx, "norm_pos"] == target_pos else (1 if df.loc[idx, "norm_pos"] in comp_set else 0) for idx in top10_idx]
        
        ndcg5_list.append(ndcg_at_k(rel5, 5))
        ndcg10_list.append(ndcg_at_k(rel10, 10))
        
        top5_same.append(np.mean([df.loc[idx, "norm_pos"] == target_pos for idx in top5_idx]))
        top10_same.append(np.mean([df.loc[idx, "norm_pos"] == target_pos for idx in top10_idx]))
        top5_comp.append(np.mean([df.loc[idx, "norm_pos"] in comp_set for idx in top5_idx]))
        top10_comp.append(np.mean([df.loc[idx, "norm_pos"] in comp_set for idx in top10_idx]))
        
        same_ranks = np.where(df.loc[sorted_idx, "norm_pos"].values == target_pos)[0]
        mrrs.append(1.0 / (same_ranks[0] + 1) if len(same_ranks) > 0 else 0.0)
        
    return {
        "top5_same": float(np.mean(top5_same)),
        "top10_same": float(np.mean(top10_same)),
        "top5_comp": float(np.mean(top5_comp)),
        "top10_comp": float(np.mean(top10_comp)),
        "pos_mrr": float(np.mean(mrrs)),
        "ndcg5": float(np.mean(ndcg5_list)),
        "ndcg10": float(np.mean(ndcg10_list)),
    }

print("\n--- ABLATION STUDY: SEPARATING EMBEDDING IMPROVEMENT FROM FILTERING ---")
# 1. Baseline embeddings without GK isolation
m_base_nofilt = evaluate_retrieval(sim_base_raw, apply_gk_isolation=False)

# 2. Baseline embeddings WITH GK isolation
m_base_filt = evaluate_retrieval(sim_base_raw, apply_gk_isolation=True)

# 3. Improved position-weighted embeddings without GK isolation
m_impr_nofilt = evaluate_retrieval(sim_impr_raw, apply_gk_isolation=False)

# 4. Improved position-weighted embeddings WITH GK isolation
m_impr_filt = evaluate_retrieval(sim_impr_raw, apply_gk_isolation=True)

print(f"{'Condition':45s} | {'Top5 Same':10s} | {'Top10 Same':10s} | {'Top5 Comp':10s} | {'MRR':8s} | {'NDCG@5':8s}")
print("-" * 105)
print(f"{'1. Baseline Embeddings (No GK Isolation)':45s} | {m_base_nofilt['top5_same']:.4f}     | {m_base_nofilt['top10_same']:.4f}      | {m_base_nofilt['top5_comp']:.4f}     | {m_base_nofilt['pos_mrr']:.4f}   | {m_base_nofilt['ndcg5']:.4f}")
print(f"{'2. Baseline Embeddings + GK Isolation':45s} | {m_base_filt['top5_same']:.4f}     | {m_base_filt['top10_same']:.4f}      | {m_base_filt['top5_comp']:.4f}     | {m_base_filt['pos_mrr']:.4f}   | {m_base_filt['ndcg5']:.4f}")
print(f"{'3. Improved Embeddings (No GK Isolation)':45s} | {m_impr_nofilt['top5_same']:.4f}     | {m_impr_nofilt['top10_same']:.4f}      | {m_impr_nofilt['top5_comp']:.4f}     | {m_impr_nofilt['pos_mrr']:.4f}   | {m_impr_nofilt['ndcg5']:.4f}")
print(f"{'4. Improved Embeddings + GK Isolation':45s} | {m_impr_filt['top5_same']:.4f}     | {m_impr_filt['top10_same']:.4f}      | {m_impr_filt['top5_comp']:.4f}     | {m_impr_filt['pos_mrr']:.4f}   | {m_impr_filt['ndcg5']:.4f}")
