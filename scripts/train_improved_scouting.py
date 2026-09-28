"""
Train and Evaluate Improved Player Scouting / Replacement Engine.
Position-aware feature weighting with strict Goalkeeper isolation and multi-season retrieval validation.
"""
import os
import json
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

RANDOM_SEED = 42
DATASET_PATH = Path("data/processed/scouting/player_scouting_dataset.csv")
IMPROVEMENT_DIR = Path("models/improvement")
IMPROVEMENT_DIR.mkdir(parents=True, exist_ok=True)

POSITION_MAPPING = {
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

COMPATIBILITY_MAP = {
    "GK": {"GK"},
    "DF": {"DF", "DM/WB"},
    "DM/WB": {"DF", "DM/WB", "CM"},
    "CM": {"DM/WB", "CM", "AM/W"},
    "AM/W": {"CM", "AM/W", "FW"},
    "FW": {"AM/W", "FW"}
}

SCOUTING_FEATURES = [
    "goals_per90", "assists_per90", "xG_per90", "npxG_per90", "xAG_per90",
    "shots_per90", "key_passes_per90", "pass_completion_pct",
    "progressive_passes_per90", "progressive_carries_per90",
    "successful_takeons_per90", "tackles_won_per90", "interceptions_per90",
    "blocks_per90", "clearances_per90", "aerial_won_pct"
]

# Position-specific tactical archetype feature weights
POSITION_WEIGHTS = {
    "DF": np.array([0.2, 0.4, 0.2, 0.2, 0.4, 0.2, 0.6, 1.4, 1.1, 0.9, 0.5, 1.8, 1.8, 1.6, 2.0, 1.8]),
    "DM/WB": np.array([0.4, 0.6, 0.4, 0.4, 0.7, 0.5, 1.1, 1.4, 1.6, 1.4, 1.1, 1.6, 1.6, 1.3, 1.1, 1.1]),
    "CM": np.array([0.5, 0.9, 0.5, 0.5, 1.1, 0.6, 1.5, 1.6, 1.7, 1.6, 1.1, 1.4, 1.4, 1.0, 0.7, 0.9]),
    "AM/W": np.array([1.5, 1.5, 1.5, 1.5, 1.6, 1.5, 1.6, 1.1, 1.2, 1.6, 1.8, 0.7, 0.7, 0.5, 0.3, 0.5]),
    "FW": np.array([1.8, 1.1, 1.8, 1.8, 1.1, 1.7, 1.1, 0.9, 0.7, 1.1, 1.3, 0.5, 0.4, 0.4, 0.3, 1.5]),
    "GK": np.array([0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 2.0, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 2.0])
}

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

def main():
    print("Loading player scouting dataset...")
    df = pd.read_csv(DATASET_PATH)
    df["norm_pos"] = df["position"].map(POSITION_MAPPING).fillna("CM")
    
    # Impute missing values with column median
    imputed_df = df[SCOUTING_FEATURES].fillna(df[SCOUTING_FEATURES].median())
    
    scaler = StandardScaler()
    X_std = scaler.fit_transform(imputed_df)
    
    # Position-aware weighting
    X_weighted = np.zeros_like(X_std)
    for pos, w in POSITION_WEIGHTS.items():
        mask = (df["norm_pos"] == pos).values
        X_weighted[mask] = X_std[mask] * np.sqrt(w)
        
    # L2-normalize
    norms = np.linalg.norm(X_weighted, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    embeddings = X_weighted / norms
    
    # Compute full similarity matrix
    sim_matrix = cosine_similarity(embeddings)
    
    # Strict GK isolation: GK only matches GK, outfield only matches outfield
    is_gk = (df["norm_pos"] == "GK").values
    for i in range(len(df)):
        if is_gk[i]:
            sim_matrix[i, ~is_gk] = -1.0
        else:
            sim_matrix[i, is_gk] = -1.0
            
    print("Evaluating retrieval metrics across all 2,325 player seasons...")
    top5_same, top10_same = [], []
    top5_comp, top10_comp = [], []
    mrrs = []
    ndcg5_list, ndcg10_list = [], []
    
    n = len(df)
    for i in range(n):
        r = sim_matrix[i].copy()
        r[i] = -999.0  # exclude self
        target_pos = df.loc[i, "norm_pos"]
        comp_set = COMPATIBILITY_MAP[target_pos]
        
        sorted_idx = np.argsort(r)[::-1]
        top5_idx = sorted_idx[:5]
        top10_idx = sorted_idx[:10]
        
        # Relevance definition for NDCG: 2 = same position, 1 = compatible position, 0 = incompatible
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
        
    # Multi-season self-retrieval
    self_retrieval_ranks = []
    for name, group in df.groupby("norm_name"):
        if len(group) > 1:
            indices = group.index.tolist()
            for idx in indices:
                r = sim_matrix[idx].copy()
                r[idx] = -999.0
                sorted_idx = np.argsort(r)[::-1]
                other_indices = [x for x in indices if x != idx]
                ranks = [np.where(sorted_idx == o)[0][0] for o in other_indices if o in sorted_idx]
                if ranks:
                    self_retrieval_ranks.append(min(ranks) < 10)
                    
    metrics = {
        "top_5_same_position_rate": float(np.mean(top5_same)),
        "top_10_same_position_rate": float(np.mean(top10_same)),
        "top_5_compatible_position_rate": float(np.mean(top5_comp)),
        "top_10_compatible_position_rate": float(np.mean(top10_comp)),
        "position_mrr": float(np.mean(mrrs)),
        "ndcg_at_5": float(np.mean(ndcg5_list)),
        "ndcg_at_10": float(np.mean(ndcg10_list)),
        "multi_season_top_10_self_retrieval_rate": float(np.mean(self_retrieval_ranks))
    }
    
    print("\n--- Improved Scouting Evaluation Results ---")
    for k, v in metrics.items():
        print(f"  {k:40s}: {v:.4f}")
        
    # Generate Error / Qualitative Analysis CSV
    rep_targets = [
        ("Alisson", "GK"),
        ("Virgil van Dijk", "DF"),
        ("William Saliba", "DF"),
        ("Trent Alexander-Arnold", "DM/WB"),
        ("Rodri", "DM/WB"),
        ("Kevin De Bruyne", "CM"),
        ("Martin Ødegaard", "AM/W"),
        ("Mohamed Salah", "FW"),
        ("Bukayo Saka", "AM/W"),
        ("Erling Haaland", "FW"),
        ("Son Heung-min", "FW")
    ]
    
    analysis_records = []
    for p_name, exp_pos in rep_targets:
        matches = df[df["player_name"].str.contains(p_name.split()[0], case=False, na=False)].sort_values("season_end_year", ascending=False)
        if matches.empty:
            continue
        t_idx = matches.index[0]
        t_row = df.loc[t_idx]
        q_vec = embeddings[t_idx]
        
        sims = np.dot(embeddings, q_vec)
        sims[df["player_name"] == t_row["player_name"]] = -999.0
        if t_row["norm_pos"] == "GK":
            sims[df["norm_pos"] != "GK"] = -999.0
        else:
            sims[df["norm_pos"] == "GK"] = -999.0
            
        top_k = np.argsort(sims)[::-1][:5]
        for rank, c_idx in enumerate(top_k, 1):
            c_row = df.loc[c_idx]
            sim_score = round(float(sims[c_idx]) * 100.0, 1)
            is_compatible = c_row["norm_pos"] in COMPATIBILITY_MAP.get(t_row["norm_pos"], {t_row["norm_pos"]})
            analysis_records.append({
                "target_player": t_row["player_name"],
                "target_position": t_row["norm_pos"],
                "target_season": t_row["season"],
                "rank": rank,
                "candidate_player": c_row["player_name"],
                "candidate_position": c_row["norm_pos"],
                "candidate_season": c_row["season"],
                "candidate_club": c_row["team"],
                "similarity_score": sim_score,
                "is_same_position": c_row["norm_pos"] == t_row["norm_pos"],
                "is_compatible_position": is_compatible,
                "candidate_age": c_row["age"],
                "candidate_market_val_eur": c_row["market_value_eur"]
            })
            
    err_csv = pd.DataFrame(analysis_records)
    err_path = IMPROVEMENT_DIR / "scouting_error_analysis.csv"
    err_csv.to_csv(err_path, index=False)
    print(f"\nSaved scouting error analysis to: {err_path}")
    
    # Save improved model artifact
    # Notice: It matches backend/ml/scouting/predict.py loading format:
    # {"player_df": df, "embeddings": embeddings, "features": SCOUTING_FEATURES, ...}
    scouting_artifact = {
        "player_df": df,
        "embeddings": embeddings,
        "features": SCOUTING_FEATURES,
        "position_weights": POSITION_WEIGHTS,
        "position_mapping": POSITION_MAPPING,
        "compatibility_map": COMPATIBILITY_MAP,
        "scaler": scaler,
        "model_name": "Position-Weighted Scouting Index with GK Isolation"
    }
    
    artifact_path = IMPROVEMENT_DIR / "scouting_model_improved.joblib"
    joblib.dump(scouting_artifact, artifact_path)
    print(f"Saved improved scouting model artifact: {artifact_path}")
    
    # Save improved metadata
    metadata = {
        "model_type": "Position-Weighted Vector Retrieval Engine",
        "dataset_rows": len(df),
        "distinct_players": int(df["norm_name"].nunique()),
        "features": SCOUTING_FEATURES,
        "position_mapping": POSITION_MAPPING,
        "compatibility_map": {k: list(v) for k, v in COMPATIBILITY_MAP.items()},
        "tactical_archetype_weights": {k: v.tolist() for k, v in POSITION_WEIGHTS.items()},
        "evaluation_metrics": metrics,
        "timestamp": pd.Timestamp.now().isoformat()
    }
    meta_path = IMPROVEMENT_DIR / "scouting_model_improved_metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved improved scouting metadata to: {meta_path}")

if __name__ == "__main__":
    main()
