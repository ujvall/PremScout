"""Training and indexing pipeline for the Player Scouting & Replacement module."""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .evaluate import evaluate_scouting_retrieval
from .preprocessing import (
    METADATA_COLS,
    POSITION_COMPATIBILITY,
    POSITION_DISPLAY_NAMES,
    POSITION_GROUP_MAP,
    SCOUTING_FEATURES,
    build_scouting_preprocessor,
    normalize_position,
)
from .similarity import ScoutingSimilarityIndex

DATA_PATH = Path("data/processed/scouting/player_scouting_dataset.csv")
MODELS_DIR = Path("models/scouting")
PLOTS_DIR = MODELS_DIR / "plots"


def plot_position_distribution(df: pd.DataFrame, output_path: Path):
    """Plot distribution of players across normalized scouting position groups."""
    fig, ax = plt.subplots(figsize=(9, 5))
    pos_counts = df["norm_pos"].value_counts()
    colors = ["#2b5c8f", "#4575b4", "#74add1", "#abd9e9", "#fdae61", "#f46d43"]

    bars = ax.bar(pos_counts.index, pos_counts.values, color=colors[:len(pos_counts)], edgecolor="black", alpha=0.85)
    for bar in bars:
        h = bar.get_height()
        ax.annotate(f"{h:,}\n({h/len(df)*100:.1f}%)", xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9)

    ax.set_title("Scouting Dataset Player Distribution by Position Group", fontsize=13, fontweight="bold")
    ax.set_ylabel("Player Seasons Count", fontsize=11)
    ax.set_xlabel("Normalized Position Group", fontsize=11)
    ax.set_ylim(0, max(pos_counts.values) * 1.18)
    ax.grid(True, linestyle="--", alpha=0.5, axis="y")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_similarity_distribution(metrics_dict: Dict[str, Any], sim_samples: np.ndarray, output_path: Path):
    """Plot distribution of pairwise cosine similarities across player population."""
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(sim_samples, bins=50, color="#2b5c8f", edgecolor="black", alpha=0.75, density=True)
    ax.axvline(0.0, color="red", linestyle="--", linewidth=1.5, label="Orthogonal (0.0)")
    ax.axvline(metrics_dict["similarity_distribution"]["median"], color="orange", linestyle="-", linewidth=1.8, label=f"Median ({metrics_dict['similarity_distribution']['median']:.3f})")
    ax.axvline(metrics_dict["similarity_distribution"]["p95"], color="green", linestyle=":", linewidth=1.8, label=f"95th Pct ({metrics_dict['similarity_distribution']['p95']:.3f})")

    ax.set_title("Pairwise Player Similarity Distribution (Standardized Cosine Space)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Cosine Similarity", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.legend(frameon=True)
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_feature_distributions(df: pd.DataFrame, features: List[str], output_path: Path):
    """Plot distribution grid for key football performance metrics."""
    fig, axes = plt.subplots(4, 4, figsize=(16, 12))
    axes = axes.flatten()

    for idx, feat in enumerate(features):
        ax = axes[idx]
        ax.hist(df[feat].dropna(), bins=25, color="#4575b4", edgecolor="black", alpha=0.8)
        clean_title = feat.replace("_per90", " /90").replace("_pct", " %").replace("_", " ").title()
        ax.set_title(clean_title, fontsize=10, fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.tick_params(axis="both", which="major", labelsize=8)

    plt.suptitle("Distributions of 16 Core Football Performance Metrics", fontsize=14, fontweight="bold", y=0.995)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_example_replacements(example_df: pd.DataFrame, output_path: Path):
    """Plot visual comparison of top replacement candidates for featured query."""
    # Take first target player from example_df
    target_name = example_df["target_player"].iloc[0]
    sub = example_df[example_df["target_player"] == target_name].head(8)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    y_pos = np.arange(len(sub))
    scores = sub["similarity_score"].values
    labels = [f"{row['candidate_player']} ({row['candidate_club']})" for _, row in sub.iterrows()]

    bars = ax.barh(y_pos[::-1], scores, color="#2b5c8f", edgecolor="black", alpha=0.85)
    for bar in bars:
        w = bar.get_width()
        ax.annotate(f"{w:.1f} / 100", xy=(w, bar.get_y() + bar.get_height() / 2),
                    xytext=(5, 0), textcoords="offset points", ha="left", va="center", fontsize=9, fontweight="bold")

    ax.set_yticks(y_pos[::-1])
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel("Similarity Score (0 - 100)", fontsize=11)
    ax.set_xlim(0, 115)
    ax.set_title(f"Top Replacement Candidates for {target_name} ({sub['target_position'].iloc[0]})", fontsize=13, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.5, axis="x")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def run_scouting_pipeline():
    """Execute complete training, embedding generation, evaluation, and serialization for scouting module."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading scouting dataset...")
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Scouting dataset not found at {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(df):,} player seasons across {df['season'].nunique()} seasons.")

    # 1. Normalize position
    df["norm_pos"] = df["position"].apply(normalize_position)
    print(f"Position breakdown:\n{df['norm_pos'].value_counts()}")

    # 2. Build and fit preprocessing pipeline on football performance features
    print(f"\nFitting preprocessing pipeline on {len(SCOUTING_FEATURES)} performance features...")
    preprocessor = build_scouting_preprocessor()
    X_scaled = preprocessor.fit_transform(df[SCOUTING_FEATURES])

    # 3. Save player embeddings (metadata + scaled features)
    embed_df = df[METADATA_COLS + ["norm_pos"]].copy()
    for idx, feat in enumerate(SCOUTING_FEATURES):
        embed_df[f"emb_{feat}"] = np.round(X_scaled[:, idx], 4)

    embed_path = MODELS_DIR / "player_embeddings.csv"
    embed_df.to_csv(embed_path, index=False)
    print(f"Saved player embeddings ({len(embed_df)} rows) to {embed_path}")

    # 4. Initialize similarity index
    index = ScoutingSimilarityIndex(df, X_scaled, SCOUTING_FEATURES)

    # 5. Evaluate similarity retrieval performance
    print("\nEvaluating scouting retrieval metrics...")
    metrics_dict, eval_table_df, feat_imp_df = evaluate_scouting_retrieval(df, X_scaled, SCOUTING_FEATURES)

    eval_csv_path = MODELS_DIR / "similarity_evaluation.csv"
    eval_table_df.to_csv(eval_csv_path, index=False)
    print(f"Saved similarity evaluation to {eval_csv_path}")

    feat_imp_csv_path = MODELS_DIR / "feature_importance.csv"
    feat_imp_df.to_csv(feat_imp_csv_path, index=False)
    print(f"Saved feature importance contribution to {feat_imp_csv_path}")

    print("\nRetrieval Evaluation Summary:")
    for _, row in eval_table_df.iterrows():
        print(f"  {row['Metric']:<40}: {row['Value']}")

    # 6. Generate example replacements for diverse key players
    print("\nGenerating example replacement searches across positions...")
    example_targets = [
        ("Declan Rice", "CM"),
        ("Erling Haaland", "FW"),
        ("Bukayo Saka", "AM/W"),
        ("Virgil van Dijk", "DF"),
        ("Emiliano Martínez", "GK"),
    ]

    all_example_rows = []
    for player_name, expected_pos in example_targets:
        try:
            replacements = index.find_replacements(
                player_name=player_name,
                top_n=5,
                same_position_only=True,
                min_minutes=900,
            )
            for cand in replacements:
                all_example_rows.append({
                    "target_player": cand["target_player"],
                    "target_position": cand["target_position"],
                    "candidate_player": cand["candidate_player"],
                    "candidate_club": cand["candidate_club"],
                    "candidate_position": cand["candidate_position"],
                    "similarity_score": cand["similarity_score"],
                    "position_compatibility": cand["position_compatibility"],
                    "age": cand["age"],
                    "market_value_eur": cand["market_value_eur"],
                    "minutes": cand["minutes"],
                })
        except Exception as e:
            print(f"Warning generating replacements for {player_name}: {e}")

    example_df = pd.DataFrame(all_example_rows)
    example_csv_path = MODELS_DIR / "example_replacements.csv"
    example_df.to_csv(example_csv_path, index=False)
    print(f"Saved {len(example_df)} example replacement rows to {example_csv_path}")

    # 7. Generate Visualizations
    print("\nGenerating visualizations...")
    plot_position_distribution(df, PLOTS_DIR / "position_distribution.png")

    # Sample similarities for distribution plot
    triu_idx = np.triu_indices(len(df), k=1)
    norms = np.linalg.norm(X_scaled, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    norm_X = X_scaled / norms
    sample_sims = np.dot(norm_X, norm_X.T)[triu_idx]
    plot_similarity_distribution(metrics_dict, sample_sims, PLOTS_DIR / "similarity_distribution.png")

    plot_feature_distributions(df, SCOUTING_FEATURES, PLOTS_DIR / "feature_distributions.png")
    if not example_df.empty:
        plot_example_replacements(example_df, PLOTS_DIR / "example_replacements.png")

    # 8. Save serialized model artifact
    artifact_path = MODELS_DIR / "scouting_model.joblib"
    artifact = {
        "preprocessor": preprocessor,
        "features": SCOUTING_FEATURES,
        "metadata_cols": METADATA_COLS,
        "player_df": df[METADATA_COLS + ["norm_pos"] + SCOUTING_FEATURES].copy(),
        "embeddings": X_scaled,
        "position_group_map": POSITION_GROUP_MAP,
        "position_compatibility": POSITION_COMPATIBILITY,
        "position_display_names": POSITION_DISPLAY_NAMES,
    }
    joblib.dump(artifact, artifact_path)
    print(f"Saved serialized model artifact to {artifact_path}")

    # 9. Save metadata JSON
    metadata = {
        "dataset_path": str(DATA_PATH),
        "dataset_row_count": len(df),
        "distinct_players_count": int(df["player_name"].nunique()),
        "features_used": SCOUTING_FEATURES,
        "feature_count": len(SCOUTING_FEATURES),
        "position_mapping": POSITION_GROUP_MAP,
        "preprocessing_method": "StandardScaler with median imputation",
        "similarity_method": "Cosine similarity on standardized performance feature vectors",
        "scaling_method": "StandardScaler (zero mean, unit variance)",
        "filters_supported": [
            "same_position_only",
            "compatible_positions",
            "max_market_value",
            "max_age",
            "min_minutes",
        ],
        "evaluation_metrics": {
            "top_5_same_position_rate": metrics_dict["top_5_same_position_rate"],
            "top_10_same_position_rate": metrics_dict["top_10_same_position_rate"],
            "top_5_compatible_position_rate": metrics_dict["top_5_compatible_position_rate"],
            "top_10_compatible_position_rate": metrics_dict["top_10_compatible_position_rate"],
            "position_mrr": metrics_dict["mean_reciprocal_rank_position"],
            "multi_season_top_10_self_retrieval_rate": metrics_dict["top_10_self_retrieval_rate"],
        },
        "training_timestamp": datetime.now(timezone.utc).isoformat(),
    }
    metadata_path = MODELS_DIR / "model_metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved model metadata to {metadata_path}")


if __name__ == "__main__":
    run_scouting_pipeline()
