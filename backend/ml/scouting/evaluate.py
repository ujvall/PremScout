"""Evaluation metrics and feature contribution analysis for the Player Scouting & Replacement module."""

from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from .preprocessing import (
    SCOUTING_FEATURES,
    POSITION_COMPATIBILITY,
    normalize_position,
)


def evaluate_scouting_retrieval(
    player_df: pd.DataFrame,
    embeddings: np.ndarray,
    features: List[str] = SCOUTING_FEATURES,
) -> Tuple[Dict[str, Any], pd.DataFrame, pd.DataFrame]:
    """Evaluate scouting similarity engine using information retrieval and stability metrics.

    Args:
        player_df: DataFrame of player metadata and metrics.
        embeddings: Standardized feature matrix (N, D).
        features: List of feature names.

    Returns:
        Tuple of:
            - metrics_dict: Retrieval precision rates, self-retrieval rate, and similarity distributions.
            - evaluation_table_df: Summary metrics formatted for reporting.
            - feature_importance_df: Elementwise inner-product contribution per feature.
    """
    df = player_df.copy().reset_index(drop=True)
    if "norm_pos" not in df.columns:
        df["norm_pos"] = df["position"].apply(normalize_position)

    # L2-normalize embeddings for cosine similarity
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    norm_embeddings = embeddings / norms

    sim_matrix = np.dot(norm_embeddings, norm_embeddings.T)
    np.fill_diagonal(sim_matrix, -1.0)  # Exclude self-match

    n = len(df)
    k_vals = [5, 10]

    # 1. Position Retrieval Consistency (Precision@K)
    same_pos_at_5 = []
    same_pos_at_10 = []
    compat_pos_at_5 = []
    compat_pos_at_10 = []
    reciprocal_ranks = []

    for i in range(n):
        target_pos = df.loc[i, "norm_pos"]
        top_indices = np.argsort(sim_matrix[i])[::-1][:10]
        top_pos = df.loc[top_indices, "norm_pos"].values

        compat_set = {target_pos} | POSITION_COMPATIBILITY.get(target_pos, set())

        # Top 5
        same_pos_at_5.append(sum(p == target_pos for p in top_pos[:5]) / 5.0)
        compat_pos_at_5.append(sum(p in compat_set for p in top_pos[:5]) / 5.0)

        # Top 10
        same_pos_at_10.append(sum(p == target_pos for p in top_pos[:10]) / 10.0)
        compat_pos_at_10.append(sum(p in compat_set for p in top_pos[:10]) / 10.0)

        # Reciprocal Rank of first same-position match
        rr = 0.0
        for rank_idx, p in enumerate(top_pos, start=1):
            if p == target_pos:
                rr = 1.0 / rank_idx
                break
        reciprocal_ranks.append(rr)

    # 2. Multi-Season Self-Retrieval Rate
    # For players appearing in > 1 season, how often is their other season in top 10?
    player_counts = df["player_name"].value_counts()
    multi_season_names = set(player_counts[player_counts > 1].index)

    self_hits_5 = 0
    self_hits_10 = 0
    multi_season_total = 0

    for i in range(n):
        name = df.loc[i, "player_name"]
        if name not in multi_season_names:
            continue
        multi_season_total += 1
        top_indices = np.argsort(sim_matrix[i])[::-1][:10]
        top_names = df.loc[top_indices, "player_name"].values

        if name in top_names[:5]:
            self_hits_5 += 1
        if name in top_names[:10]:
            self_hits_10 += 1

    self_retrieval_5 = (self_hits_5 / multi_season_total) if multi_season_total > 0 else 0.0
    self_retrieval_10 = (self_hits_10 / multi_season_total) if multi_season_total > 0 else 0.0

    # 3. Overall Pairwise Similarity Distribution
    # Sample 100,000 off-diagonal elements for distribution summary
    triu_indices = np.triu_indices(n, k=1)
    sample_sims = sim_matrix[triu_indices]

    sim_dist = {
        "min": float(np.min(sample_sims)),
        "p05": float(np.percentile(sample_sims, 5)),
        "p25": float(np.percentile(sample_sims, 25)),
        "median": float(np.median(sample_sims)),
        "mean": float(np.mean(sample_sims)),
        "p75": float(np.percentile(sample_sims, 75)),
        "p95": float(np.percentile(sample_sims, 95)),
        "max": float(np.max(sample_sims)),
    }

    # 4. Mathematical Feature Contribution (Elementwise Inner Product among top-5 nearest neighbors)
    feature_contributions = np.zeros(len(features))
    for i in range(n):
        top5_idx = np.argsort(sim_matrix[i])[::-1][:5]
        for idx in top5_idx:
            feature_contributions += norm_embeddings[i] * norm_embeddings[idx]

    feature_contributions /= (n * 5.0)
    total_contrib = np.sum(feature_contributions)
    if total_contrib > 0:
        normalized_contrib = feature_contributions / total_contrib
    else:
        normalized_contrib = feature_contributions

    feature_importance_df = pd.DataFrame({
        "feature": features,
        "importance": normalized_contrib,
    }).sort_values(by="importance", ascending=False).reset_index(drop=True)

    metrics_dict = {
        "dataset_size": n,
        "multi_season_profiles_evaluated": multi_season_total,
        "top_5_same_position_rate": float(np.mean(same_pos_at_5)),
        "top_10_same_position_rate": float(np.mean(same_pos_at_10)),
        "top_5_compatible_position_rate": float(np.mean(compat_pos_at_5)),
        "top_10_compatible_position_rate": float(np.mean(compat_pos_at_10)),
        "mean_reciprocal_rank_position": float(np.mean(reciprocal_ranks)),
        "top_5_self_retrieval_rate": float(self_retrieval_5),
        "top_10_self_retrieval_rate": float(self_retrieval_10),
        "similarity_distribution": sim_dist,
    }

    eval_table = pd.DataFrame([
        {"Metric": "Top-5 Same-Position Retrieval Rate", "Value": f"{metrics_dict['top_5_same_position_rate']*100:.2f}%", "Domain Benchmark": "High positional clustering"},
        {"Metric": "Top-10 Same-Position Retrieval Rate", "Value": f"{metrics_dict['top_10_same_position_rate']*100:.2f}%", "Domain Benchmark": "High positional clustering"},
        {"Metric": "Top-5 Compatible Position Rate", "Value": f"{metrics_dict['top_5_compatible_position_rate']*100:.2f}%", "Domain Benchmark": "Tactical adjacency preserved"},
        {"Metric": "Top-10 Compatible Position Rate", "Value": f"{metrics_dict['top_10_compatible_position_rate']*100:.2f}%", "Domain Benchmark": "Tactical adjacency preserved"},
        {"Metric": "Position Mean Reciprocal Rank (MRR)", "Value": f"{metrics_dict['mean_reciprocal_rank_position']:.4f}", "Domain Benchmark": "First match at rank ~1"},
        {"Metric": "Top-10 Temporal Self-Retrieval Rate", "Value": f"{metrics_dict['top_10_self_retrieval_rate']*100:.2f}%", "Domain Benchmark": "Multi-season signature persistence"},
        {"Metric": "Pairwise Cosine Similarity Mean", "Value": f"{sim_dist['mean']:.4f}", "Domain Benchmark": "Centered near orthogonal (0.0)"},
        {"Metric": "Pairwise Cosine Similarity Median", "Value": f"{sim_dist['median']:.4f}", "Domain Benchmark": "Centered near orthogonal (0.0)"},
    ])

    return metrics_dict, eval_table, feature_importance_df
