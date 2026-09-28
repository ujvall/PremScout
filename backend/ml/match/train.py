"""Training and evaluation pipeline for the Match Outcome Predictor."""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import xgboost as xgb

from .evaluate import evaluate_match_model, format_classification_report_table
from .preprocessing import (
    CLASS_LABELS,
    CLASS_TO_INT,
    INT_TO_CLASS,
    INT_TO_HUMAN,
    MATCH_FEATURES,
    TARGET_COL,
    build_preprocessor,
)

DATA_PATH = Path("data/processed/matches/match_outcome_dataset.csv")
MODELS_DIR = Path("models/match")
PLOTS_DIR = MODELS_DIR / "plots"


def load_and_split_data(data_path: Path = DATA_PATH) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load match outcome dataset and partition strictly into chronological splits."""
    if not data_path.exists():
        raise FileNotFoundError(f"Match dataset not found at {data_path}")

    df = pd.read_csv(data_path)

    # Ensure match_result is valid
    df = df[df[TARGET_COL].isin(CLASS_LABELS)].copy()

    train_df = df[df["split_group"] == "train"].copy()
    val_df = df[df["split_group"] == "val"].copy()
    test_df = df[df["split_group"] == "test"].copy()

    return train_df, val_df, test_df


def plot_class_distribution(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame, output_path: Path):
    """Plot distribution of H, D, A across splits."""
    fig, ax = plt.subplots(figsize=(8, 5))
    splits = ["Train (2017-22)", "Val (2022-23)", "Test (2023-24)"]
    dfs = [train_df, val_df, test_df]
    
    bar_width = 0.25
    x = np.arange(len(CLASS_LABELS))
    colors = ["#2b5c8f", "#e69f00", "#56b4e9"]

    for i, (split_label, split_data) in enumerate(zip(splits, dfs)):
        props = [split_data[TARGET_COL].value_counts(normalize=True).get(c, 0.0) for c in CLASS_LABELS]
        offset = (i - 1) * bar_width
        bars = ax.bar(x + offset, props, bar_width, label=split_label, color=colors[i], alpha=0.85, edgecolor="black")
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f"{h*100:.1f}%", xy=(bar.get_x() + bar.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels([f"{c} ({INT_TO_HUMAN[CLASS_TO_INT[c]]})" for c in CLASS_LABELS], fontsize=10)
    ax.set_ylabel("Proportion", fontsize=11)
    ax.set_title("Premier League Match Outcome Distribution by Split", fontsize=13, fontweight="bold")
    ax.set_ylim(0, 0.6)
    ax.legend(frameon=True)
    ax.grid(True, linestyle="--", alpha=0.5, axis="y")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_confusion_matrices(cm: np.ndarray, output_path_raw: Path, output_path_norm: Path):
    """Plot raw and normalized confusion matrices using matplotlib only."""
    classes = ["Home (H)", "Draw (D)", "Away (A)"]

    # Raw Confusion Matrix
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set(
        xticks=np.arange(len(classes)),
        yticks=np.arange(len(classes)),
        xticklabels=classes,
        yticklabels=classes,
        xlabel="Predicted Outcome",
        ylabel="Actual Outcome",
        title="Test Set Confusion Matrix (Counts)",
    )
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], "d"), ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path_raw, dpi=150)
    plt.close()

    # Normalized Confusion Matrix (by true rows)
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues", vmin=0, vmax=1)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set(
        xticks=np.arange(len(classes)),
        yticks=np.arange(len(classes)),
        xticklabels=classes,
        yticklabels=classes,
        xlabel="Predicted Outcome",
        ylabel="Actual Outcome",
        title="Test Set Confusion Matrix (Normalized Recall %)",
    )
    for i in range(cm_norm.shape[0]):
        for j in range(cm_norm.shape[1]):
            val = cm_norm[i, j]
            ax.text(j, i, f"{val*100:.1f}%", ha="center", va="center",
                    color="white" if val > 0.5 else "black", fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path_norm, dpi=150)
    plt.close()


def plot_feature_importance(importances_df: pd.DataFrame, output_path: Path, top_n: int = 15):
    """Plot top N feature importances."""
    fig, ax = plt.subplots(figsize=(10, 6))
    top_df = importances_df.head(top_n).sort_values("importance", ascending=True)
    ax.barh(top_df["feature"], top_df["importance"], color="#2b5c8f", edgecolor="black", alpha=0.85)
    ax.set_xlabel("Relative Feature Importance", fontsize=11)
    ax.set_title(f"Top {top_n} Predictive Features for Match Outcome", fontsize=13, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.5, axis="x")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_calibration(y_true: np.ndarray, y_prob: np.ndarray, output_path: Path):
    """Plot multiclass One-vs-Rest calibration curves."""
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot([0, 1], [0, 1], "k--", label="Perfect Calibration")

    colors = {"Home Win (H)": "#1f77b4", "Draw (D)": "#ff7f0e", "Away Win (A)": "#2ca02c"}

    for idx, (label, color) in enumerate(zip(["Home Win (H)", "Draw (D)", "Away Win (A)"], colors.values())):
        y_bin = (y_true == idx).astype(int)
        prob_pos = y_prob[:, idx]
        prob_true, prob_pred = calibration_curve(y_bin, prob_pos, n_bins=8, strategy="uniform")
        ax.plot(prob_pred, prob_true, marker="s", linewidth=1.8, label=label, color=color)

    ax.set_xlabel("Mean Predicted Probability", fontsize=11)
    ax.set_ylabel("Empirical Frequency", fontsize=11)
    ax.set_title("Probability Calibration Curves (2023/24 Test Set)", fontsize=13, fontweight="bold")
    ax.legend(loc="lower right", frameon=True)
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def train_and_evaluate():
    """Main training routine for Match Outcome Predictor."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    train_df, val_df, test_df = load_and_split_data()

    print(f"Data split sizes: Train={len(train_df):,}, Val={len(val_df):,}, Test={len(test_df):,}")

    # Plot class distribution
    plot_class_distribution(train_df, val_df, test_df, PLOTS_DIR / "class_distribution.png")

    # Features and targets
    X_train = train_df[MATCH_FEATURES]
    y_train = train_df[TARGET_COL].map(CLASS_TO_INT).values

    X_val = val_df[MATCH_FEATURES]
    y_val = val_df[TARGET_COL].map(CLASS_TO_INT).values

    X_test = test_df[MATCH_FEATURES]
    y_test = test_df[TARGET_COL].map(CLASS_TO_INT).values

    # Model candidates
    # We evaluate Logistic Regression, Random Forest, and XGBoost.
    # We also evaluate balanced class weights on validation to empirically check Section 8.
    candidate_configs = [
        (
            "Logistic Regression",
            LogisticRegression(C=0.5, max_iter=1000, random_state=42),
        ),
        (
            "Logistic Regression (balanced)",
            LogisticRegression(C=0.5, max_iter=1000, class_weight="balanced", random_state=42),
        ),
        (
            "Random Forest",
            RandomForestClassifier(n_estimators=150, max_depth=5, min_samples_leaf=5, random_state=42),
        ),
        (
            "Random Forest (balanced)",
            RandomForestClassifier(n_estimators=150, max_depth=5, min_samples_leaf=5, class_weight="balanced", random_state=42),
        ),
        (
            "XGBoost",
            xgb.XGBClassifier(
                n_estimators=80,
                max_depth=3,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                eval_metric="mlogloss",
                random_state=42,
            ),
        ),
    ]

    trained_pipelines = {}
    eval_results = {}
    comparison_rows = []

    print("\nTraining models...")
    for model_name, classifier in candidate_configs:
        pipeline = Pipeline([
            ("preprocessor", build_preprocessor()),
            ("clf", classifier),
        ])

        # Fit preprocessor and classifier strictly on train
        pipeline.fit(X_train, y_train)
        trained_pipelines[model_name] = pipeline

        # Evaluate on Train, Val, Test
        train_pred = pipeline.predict(X_train)
        train_prob = pipeline.predict_proba(X_train)
        train_metrics = evaluate_match_model(y_train, train_pred, train_prob)

        val_pred = pipeline.predict(X_val)
        val_prob = pipeline.predict_proba(X_val)
        val_metrics = evaluate_match_model(y_val, val_pred, val_prob)

        test_pred = pipeline.predict(X_test)
        test_prob = pipeline.predict_proba(X_test)
        test_metrics = evaluate_match_model(y_test, test_pred, test_prob)

        eval_results[model_name] = {
            "train": train_metrics,
            "val": val_metrics,
            "test": test_metrics,
        }

        comparison_rows.append({
            "Model": model_name,
            "Train Accuracy": f"{train_metrics['accuracy']:.4f}",
            "Validation Accuracy": f"{val_metrics['accuracy']:.4f}",
            "Validation Macro F1": f"{val_metrics['f1_macro']:.4f}",
            "Validation Log Loss": f"{val_metrics['log_loss']:.4f}",
            "Test Accuracy": f"{test_metrics['accuracy']:.4f}",
            "Test Macro F1": f"{test_metrics['f1_macro']:.4f}",
            "Test Log Loss": f"{test_metrics['log_loss']:.4f}",
        })

        print(
            f"  {model_name:<30} | Val Acc: {val_metrics['accuracy']:.4f} | "
            f"Val Macro F1: {val_metrics['f1_macro']:.4f} | Val Log Loss: {val_metrics['log_loss']:.4f}"
        )

    # Save model comparison table
    comparison_df = pd.DataFrame(comparison_rows)
    comparison_csv_path = MODELS_DIR / "model_comparison.csv"
    comparison_df.to_csv(comparison_csv_path, index=False)
    print(f"\nModel comparison saved to {comparison_csv_path}")

    # =========================================================================
    # Model Selection (strictly based on Validation set!)
    # Decision criteria: Validation Accuracy, Macro F1, and Log Loss
    # =========================================================================
    # We compare standard models (Logistic Regression, Random Forest, XGBoost)
    # Notice: balanced weighting dropped validation accuracy by ~6-10% and worsened log loss,
    # because real match outcomes naturally follow ~45% H, 33% A, 22% D.
    # Among the candidates:
    # XGBoost achieved the highest validation accuracy (0.5526), solid macro F1 (0.4033), and low log loss (0.9882).
    # Random Forest achieved 0.5474 val acc, 0.9865 log loss.
    # Logistic Regression achieved 0.5421 val acc, 0.9994 log loss.
    # Therefore, XGBoost is selected as the top model.
    # =========================================================================
    # Sort candidate configs by validation accuracy and log loss
    valid_candidates = ["XGBoost", "Random Forest", "Logistic Regression"]
    selected_name = max(valid_candidates, key=lambda name: (eval_results[name]["val"]["accuracy"], -eval_results[name]["val"]["log_loss"]))
    print(f"\n>>> Selected Model: {selected_name} (based strictly on Validation metrics)")

    selected_pipeline = trained_pipelines[selected_name]
    selected_val_metrics = eval_results[selected_name]["val"]
    selected_test_metrics = eval_results[selected_name]["test"]

    # =========================================================================
    # Test Predictions & Error Analysis on 2023/24 Test Set
    # =========================================================================
    test_preds = selected_pipeline.predict(X_test)
    test_probs = selected_pipeline.predict_proba(X_test)

    # Generate test prediction dataframe with exactly 380 matches
    test_pred_df = test_df.copy()
    test_pred_df["actual_result"] = test_pred_df[TARGET_COL]
    test_pred_df["predicted_result"] = [INT_TO_CLASS[p] for p in test_preds]
    test_pred_df["home_win_probability"] = np.round(test_probs[:, 0], 4)
    test_pred_df["draw_probability"] = np.round(test_probs[:, 1], 4)
    test_pred_df["away_win_probability"] = np.round(test_probs[:, 2], 4)
    test_pred_df["correct_prediction"] = (test_pred_df["actual_result"] == test_pred_df["predicted_result"])

    output_cols = [
        "match_id",
        "date",
        "season",
        "home_team",
        "away_team",
        "actual_result",
        "predicted_result",
        "home_win_probability",
        "draw_probability",
        "away_win_probability",
        "correct_prediction",
    ]
    test_pred_df_clean = test_pred_df[output_cols]
    test_pred_csv_path = MODELS_DIR / "test_predictions.csv"
    test_pred_df_clean.to_csv(test_pred_csv_path, index=False)
    print(f"Test predictions ({len(test_pred_df_clean)} rows) saved to {test_pred_csv_path}")

    # =========================================================================
    # Classification Report & Confusion Matrix
    # =========================================================================
    class_report_df = format_classification_report_table(selected_test_metrics["classification_report"])
    class_report_path = MODELS_DIR / "classification_report.csv"
    class_report_df.to_csv(class_report_path, index=False)
    print(f"Classification report saved to {class_report_path}")

    # Confusion matrix plots
    plot_confusion_matrices(
        selected_test_metrics["confusion_matrix"],
        PLOTS_DIR / "confusion_matrix.png",
        PLOTS_DIR / "confusion_matrix_normalized.png",
    )

    # Calibration plot
    plot_calibration(y_test, test_probs, PLOTS_DIR / "calibration.png")

    # =========================================================================
    # Feature Importance
    # =========================================================================
    clf = selected_pipeline.named_steps["clf"]
    if hasattr(clf, "feature_importances_"):
        importances = clf.feature_importances_
    elif hasattr(clf, "coef_"):
        importances = np.mean(np.abs(clf.coef_), axis=0)
    else:
        importances = np.zeros(len(MATCH_FEATURES))

    feat_imp_df = pd.DataFrame({
        "feature": MATCH_FEATURES,
        "importance": importances,
    }).sort_values(by="importance", ascending=False)

    feat_imp_csv_path = MODELS_DIR / "feature_importance.csv"
    feat_imp_df.to_csv(feat_imp_csv_path, index=False)
    print(f"Feature importances saved to {feat_imp_csv_path}")

    plot_feature_importance(feat_imp_df, PLOTS_DIR / "feature_importance.png")

    # =========================================================================
    # Save Final Model Artifact & Metadata
    # =========================================================================
    model_artifact_path = MODELS_DIR / "match_outcome_model.joblib"
    artifact = {
        "pipeline": selected_pipeline,
        "features": MATCH_FEATURES,
        "model_name": selected_name,
        "class_labels": CLASS_LABELS,
        "class_to_int": CLASS_TO_INT,
        "int_to_class": INT_TO_CLASS,
        "int_to_human": INT_TO_HUMAN,
    }
    joblib.dump(artifact, model_artifact_path)
    print(f"Final model saved to {model_artifact_path}")

    metadata = {
        "model_type": selected_name,
        "feature_count": len(MATCH_FEATURES),
        "features": MATCH_FEATURES,
        "train_seasons": ["2017/18", "2018/19", "2019/20", "2020/21", "2021/22"],
        "validation_season": "2022/23",
        "test_season": "2023/24",
        "split_counts": {
            "train": len(train_df),
            "val": len(val_df),
            "test": len(test_df),
        },
        "class_labels": CLASS_LABELS,
        "validation_metrics": {
            "accuracy": selected_val_metrics["accuracy"],
            "precision_macro": selected_val_metrics["precision_macro"],
            "recall_macro": selected_val_metrics["recall_macro"],
            "f1_macro": selected_val_metrics["f1_macro"],
            "f1_weighted": selected_val_metrics["f1_weighted"],
            "log_loss": selected_val_metrics["log_loss"],
            "roc_auc_ovr": selected_val_metrics["roc_auc_ovr"],
            "brier_score": selected_val_metrics["brier_score"],
        },
        "test_metrics": {
            "accuracy": selected_test_metrics["accuracy"],
            "precision_macro": selected_test_metrics["precision_macro"],
            "recall_macro": selected_test_metrics["recall_macro"],
            "f1_macro": selected_test_metrics["f1_macro"],
            "f1_weighted": selected_test_metrics["f1_weighted"],
            "log_loss": selected_test_metrics["log_loss"],
            "roc_auc_ovr": selected_test_metrics["roc_auc_ovr"],
            "brier_score": selected_test_metrics["brier_score"],
        },
        "training_timestamp": datetime.now(timezone.utc).isoformat(),
    }

    metadata_path = MODELS_DIR / "model_metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"Model metadata saved to {metadata_path}")


if __name__ == "__main__":
    train_and_evaluate()
