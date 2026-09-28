"""
Train and Evaluate Improved Match Outcome Predictor.
Champion Model: XGBoost Classifier with sequential pre-match Elo ratings and parity features.
Validation set (2022/23) used for selection; final single evaluation on test set (2023/24).
"""
import os
import json
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    log_loss, confusion_matrix, classification_report, roc_auc_score
)

RANDOM_SEED = 42
DATASET_PATH = Path("data/processed/matches/match_outcome_dataset.csv")
IMPROVEMENT_DIR = Path("models/improvement")
IMPROVEMENT_DIR.mkdir(parents=True, exist_ok=True)

HOME_ADVANTAGE = 65.0
K_FACTOR = 20.0

def compute_sequential_elo(df: pd.DataFrame):
    """Compute chronological pre-match Elo ratings without data leakage."""
    df_sorted = df.sort_values(["date", "match_id"]).copy().reset_index(drop=True)
    
    elo_ratings = {}
    home_elos, away_elos, elo_diffs, elo_exp_h = [], [], [], []
    
    for _, row in df_sorted.iterrows():
        h_team, a_team = row["home_team"], row["away_team"]
        h_elo = elo_ratings.get(h_team, 1500.0)
        a_elo = elo_ratings.get(a_team, 1500.0)
        
        home_elos.append(h_elo)
        away_elos.append(a_elo)
        
        diff = (h_elo + HOME_ADVANTAGE) - a_elo
        exp_h = 1.0 / (1.0 + 10.0 ** (-diff / 400.0))
        elo_diffs.append(diff)
        elo_exp_h.append(exp_h)
        
        # Post-match update (only after pre-match features are recorded)
        res = row["match_result"]
        s_h = 1.0 if res == "H" else (0.5 if res == "D" else 0.0)
        s_a = 1.0 - s_h
        elo_ratings[h_team] = h_elo + K_FACTOR * (s_h - exp_h)
        elo_ratings[a_team] = a_elo + K_FACTOR * (s_a - (1.0 - exp_h))
        
    df_sorted["home_elo"] = home_elos
    df_sorted["away_elo"] = away_elos
    df_sorted["elo_diff"] = elo_diffs
    df_sorted["elo_exp_h"] = elo_exp_h
    df_sorted["abs_elo_diff"] = np.abs(df_sorted["elo_diff"])
    df_sorted["abs_rank_diff"] = np.abs(df_sorted["rank_diff"])
    df_sorted["abs_ppg_diff"] = np.abs(df_sorted["home_ppg"] - df_sorted["away_ppg"])
    
    return df_sorted, elo_ratings

def main():
    print("Loading match outcome dataset...")
    raw_df = pd.read_csv(DATASET_PATH)
    raw_df["date"] = pd.to_datetime(raw_df["date"])
    
    df, final_elo_ratings = compute_sequential_elo(raw_df)
    
    base_features = [
        "home_rest_days", "away_rest_days", "home_rolling_points_5", "away_rolling_points_5",
        "home_rolling_gf_5", "away_rolling_gf_5", "home_rolling_ga_5", "away_rolling_ga_5",
        "home_rolling_shots_5", "away_rolling_shots_5", "home_rolling_sot_5", "away_rolling_sot_5",
        "home_rolling_sot_conceded_5", "away_rolling_sot_conceded_5", "home_rolling_corners_5",
        "away_rolling_corners_5", "home_rolling_win_pct_5", "away_rolling_win_pct_5",
        "home_league_rank", "away_league_rank", "rank_diff", "home_ppg", "away_ppg",
        "h2h_matches_prior", "h2h_home_win_pct"
    ]
    elo_features = [
        "home_elo", "away_elo", "elo_diff", "elo_exp_h",
        "abs_elo_diff", "abs_rank_diff", "abs_ppg_diff"
    ]
    all_features = base_features + elo_features
    
    class_labels = ["H", "D", "A"]
    class_to_int = {"H": 0, "D": 1, "A": 2}
    int_to_class = {0: "H", 1: "D", 2: "A"}
    int_to_human = {0: "Home Win", 1: "Draw", 2: "Away Win"}
    df["target"] = df["match_result"].map(class_to_int)
    
    train_df = df[df["split_group"] == "train"].copy()
    val_df = df[df["split_group"] == "val"].copy()
    test_df = df[df["split_group"] == "test"].copy()
    
    print(f"Dataset splits: Train={len(train_df):,} | Val={len(val_df):,} | Test={len(test_df):,}")
    
    X_train = train_df[all_features].values
    y_train = train_df["target"].values
    
    model = XGBClassifier(
        n_estimators=150,
        max_depth=4,
        learning_rate=0.03,
        random_state=RANDOM_SEED,
        eval_metric="mlogloss",
        n_jobs=-1
    )
    
    print("Fitting model on training split (2017/18 - 2021/22)...")
    model.fit(X_train, y_train)
    
    def evaluate_split(split_df, split_name):
        X = split_df[all_features].values
        y_true = split_df["target"].values
        probs = model.predict_proba(X)
        preds = model.predict(X)
        
        acc = accuracy_score(y_true, preds)
        p_macro = precision_score(y_true, preds, average="macro", zero_division=0)
        r_macro = recall_score(y_true, preds, average="macro", zero_division=0)
        f1_macro = f1_score(y_true, preds, average="macro", zero_division=0)
        f1_weighted = f1_score(y_true, preds, average="weighted", zero_division=0)
        ll = log_loss(y_true, probs)
        
        y_true_oh = np.eye(3)[y_true]
        brier = np.mean(np.sum((probs - y_true_oh) ** 2, axis=1))
        
        try:
            auc = roc_auc_score(y_true_oh, probs, average="macro", multi_class="ovr")
        except Exception:
            auc = None
            
        cm = confusion_matrix(y_true, preds, labels=[0, 1, 2])
        
        # Per class metrics
        p_class = precision_score(y_true, preds, average=None, zero_division=0)
        r_class = recall_score(y_true, preds, average=None, zero_division=0)
        f1_c = f1_score(y_true, preds, average=None, zero_division=0)
        
        print(f"\n--- {split_name} Evaluation ---")
        print(f"Accuracy:    {acc:.4f} ({acc*100:.2f}%)")
        print(f"Macro F1:    {f1_macro:.4f}")
        print(f"Weighted F1: {f1_weighted:.4f}")
        print(f"Log Loss:    {ll:.4f}")
        print(f"Brier Score: {brier:.4f}")
        if auc:
            print(f"ROC-AUC OvR: {auc:.4f}")
        print("Confusion Matrix [H, D, A]:")
        print(cm)
        print("Class-Level Performance:")
        for i, c_name in enumerate(["Home (H)", "Draw (D)", "Away (A)"]):
            print(f"  {c_name:10s} - Precision: {p_class[i]:.4f} | Recall: {r_class[i]:.4f} | F1: {f1_c[i]:.4f}")
            
        return {
            "accuracy": float(acc),
            "precision_macro": float(p_macro),
            "recall_macro": float(r_macro),
            "f1_macro": float(f1_macro),
            "f1_weighted": float(f1_weighted),
            "log_loss": float(ll),
            "brier_score": float(brier),
            "roc_auc_ovr": float(auc) if auc else None,
            "confusion_matrix": cm.tolist(),
            "class_metrics": {
                "H": {"precision": float(p_class[0]), "recall": float(r_class[0]), "f1": float(f1_c[0])},
                "D": {"precision": float(p_class[1]), "recall": float(r_class[1]), "f1": float(f1_c[1])},
                "A": {"precision": float(p_class[2]), "recall": float(r_class[2]), "f1": float(f1_c[2])},
            },
            "probs": probs,
            "preds": preds
        }

    val_res = evaluate_split(val_df, "Validation Split (2022/23)")
    test_res = evaluate_split(test_df, "Test Split (2023/24 - Final Holdout)")
    
    # Save Error Analysis CSV on the holdout test set
    test_df_out = test_df.copy()
    test_df_out["predicted"] = [int_to_class[p] for p in test_res["preds"]]
    test_df_out["home_probability"] = test_res["probs"][:, 0]
    test_df_out["draw_probability"] = test_res["probs"][:, 1]
    test_df_out["away_probability"] = test_res["probs"][:, 2]
    test_df_out["confidence"] = test_res["probs"].max(axis=1)
    test_df_out["correct"] = (test_df_out["predicted"] == test_df_out["match_result"])
    
    analysis_cols = [
        "date", "home_team", "away_team", "match_result", "predicted",
        "home_probability", "draw_probability", "away_probability",
        "confidence", "correct", "home_elo", "away_elo", "elo_diff"
    ]
    err_csv = test_df_out[analysis_cols].rename(columns={"match_result": "actual"}).sort_values("date")
    err_path = IMPROVEMENT_DIR / "match_error_analysis.csv"
    err_csv.to_csv(err_path, index=False)
    print(f"\nSaved match error analysis to: {err_path}")
    
    # Save improved model artifact compatible with backend
    # Notice: In backend/ml/match/predict.py, it loads pipeline and features
    model_bundle = {
        "pipeline": model,
        "features": all_features,
        "base_features": base_features,
        "elo_features": elo_features,
        "model_name": "XGBoost Classifier with Sequential Elo",
        "class_labels": class_labels,
        "class_to_int": class_to_int,
        "int_to_class": int_to_class,
        "int_to_human": int_to_human,
        "team_elo_ratings": final_elo_ratings,
        "home_advantage_elo": HOME_ADVANTAGE
    }
    
    artifact_path = IMPROVEMENT_DIR / "match_outcome_model_improved.joblib"
    joblib.dump(model_bundle, artifact_path)
    print(f"Saved improved match model artifact: {artifact_path}")
    
    # Save improved metadata
    metadata = {
        "model_type": "XGBoost Classifier (Improved Champion with Sequential Elo)",
        "random_seed": RANDOM_SEED,
        "hyperparameters": {
            "n_estimators": 150,
            "max_depth": 4,
            "learning_rate": 0.03,
            "eval_metric": "mlogloss"
        },
        "features_count": len(all_features),
        "features": all_features,
        "engineered_elo_features": elo_features,
        "split_counts": {
            "train": len(train_df),
            "val": len(val_df),
            "test": len(test_df)
        },
        "validation_metrics": {k: v for k, v in val_res.items() if k not in ["probs", "preds"]},
        "test_metrics": {k: v for k, v in test_res.items() if k not in ["probs", "preds"]},
        "timestamp": pd.Timestamp.now().isoformat()
    }
    
    meta_path = IMPROVEMENT_DIR / "match_model_improved_metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved improved match metadata to: {meta_path}")

if __name__ == "__main__":
    main()
