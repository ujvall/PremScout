"""Training pipeline for the Transfer Value Predictor module.

Trains and evaluates:
1. LinearRegression
2. RandomForestRegressor
3. XGBRegressor
across both:
A. Raw target (market_value_eur)
B. Log1p target (log1p(market_value_eur))

Selects the best model strictly using the Validation set (2022/23),
evaluates once on the untouched Test set (2023/24),
and saves model artifacts, comparison tables, error analysis, and visualization plots.
"""

from datetime import datetime
import json
from pathlib import Path
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor

from .evaluate import calculate_metrics, format_eur
from .preprocessing import (
    ALL_MODEL_FEATURES,
    CATEGORICAL_FEATURES,
    NUMERICAL_FEATURES,
    TARGET_COLUMN,
    get_transformed_feature_names,
    get_transfer_preprocessor,
)

DATA_PATH = Path("data/processed/transfer/transfer_value_dataset.csv")
MODELS_DIR = Path("models/transfer")
PLOTS_DIR = MODELS_DIR / "plots"


def train_and_evaluate_transfer_models(
    data_path: Path = DATA_PATH,
    output_dir: Path = MODELS_DIR,
) -> dict:
    """Train all 3 model architectures under raw and log1p targets and select the best model."""
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 90)
    print("STARTING TRANSFER VALUE PREDICTOR TRAINING PIPELINE")
    print("=" * 90)

    # 1. Load data
    print(f"\n[1/7] Loading dataset from {data_path}...")
    df = pd.read_csv(data_path)

    # 2. Chronological splits
    train_df = df[df["split_group"] == "train"].copy()
    val_df = df[df["split_group"] == "val"].copy()
    test_df = df[df["split_group"] == "test"].copy()

    print(f"  Training Set   (2017/18 - 2021/22): {len(train_df):,} rows")
    print(f"  Validation Set (2022/23)          : {len(val_df):,} rows")
    print(f"  Test Set       (2023/24)          : {len(test_df):,} rows")

    X_train_raw = train_df[ALL_MODEL_FEATURES]
    y_train_raw = train_df[TARGET_COLUMN].values

    X_val_raw = val_df[ALL_MODEL_FEATURES]
    y_val_raw = val_df[TARGET_COLUMN].values

    X_test_raw = test_df[ALL_MODEL_FEATURES]
    y_test_raw = test_df[TARGET_COLUMN].values

    # 3. Fit Preprocessing ONLY on training data
    print("\n[2/7] Fitting preprocessor on training data...")
    preprocessor = get_transfer_preprocessor()
    X_train_proc = preprocessor.fit_transform(X_train_raw)
    X_val_proc = preprocessor.transform(X_val_raw)
    X_test_proc = preprocessor.transform(X_test_raw)

    feature_names = get_transformed_feature_names(preprocessor)
    print(f"  Transformed feature space: {X_train_proc.shape[1]} features (including one-hot encodings)")

    # Target transformations
    y_train_log = np.log1p(y_train_raw)
    y_val_log = np.log1p(y_val_raw)
    y_test_log = np.log1p(y_test_raw)

    # 4. Model candidates
    model_definitions = {
        "Linear Regression": lambda: LinearRegression(),
        "Random Forest": lambda: RandomForestRegressor(
            n_estimators=120,
            max_depth=16,
            min_samples_split=4,
            random_state=42,
            n_jobs=-1,
        ),
        "XGBoost": lambda: XGBRegressor(
            n_estimators=160,
            max_depth=6,
            learning_rate=0.07,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
        ),
    }

    # 5. Experiment execution: 3 models x 2 target versions = 6 experiments
    print("\n[3/7] Training and evaluating 6 model configurations...")
    experiments = []

    for model_name, model_fn in model_definitions.items():
        for target_transform in ["raw", "log1p"]:
            config_label = f"{model_name} ({target_transform} target)"
            print(f"  --> Training: {config_label}...")

            model = model_fn()
            y_tr = y_train_log if target_transform == "log1p" else y_train_raw
            model.fit(X_train_proc, y_tr)

            # Predictions
            pred_train = model.predict(X_train_proc)
            pred_val = model.predict(X_val_proc)
            pred_test = model.predict(X_test_proc)

            # Invert log transform if applicable
            if target_transform == "log1p":
                pred_train = np.expm1(pred_train)
                pred_val = np.expm1(pred_val)
                pred_test = np.expm1(pred_test)

            # Calculate metrics
            m_train = calculate_metrics(y_train_raw, pred_train)
            m_val = calculate_metrics(y_val_raw, pred_val)
            m_test = calculate_metrics(y_test_raw, pred_test)

            experiments.append({
                "model_name": model_name,
                "target_transform": target_transform,
                "model_instance": model,
                "pred_test": pred_test,
                "train_mae": m_train["mae"],
                "val_mae": m_val["mae"],
                "test_mae": m_test["mae"],
                "train_rmse": m_train["rmse"],
                "val_rmse": m_val["rmse"],
                "test_rmse": m_test["rmse"],
                "train_r2": m_train["r2"],
                "val_r2": m_val["r2"],
                "test_r2": m_test["r2"],
                "train_mae_eur": m_train["mae_formatted"],
                "val_mae_eur": m_val["mae_formatted"],
                "test_mae_eur": m_test["mae_formatted"],
                "test_rmse_eur": m_test["rmse_formatted"],
                "test_r2_str": m_test["r2_formatted"],
            })

    # 6. Save comparison table
    print("\n[4/7] Compiling model comparison table...")
    comp_rows = []
    for exp in experiments:
        comp_rows.append({
            "Model": exp["model_name"],
            "Target Transform": exp["target_transform"],
            "Train MAE": exp["train_mae_eur"],
            "Val MAE": exp["val_mae_eur"],
            "Test MAE": exp["test_mae_eur"],
            "Test RMSE": exp["test_rmse_eur"],
            "Test R²": exp["test_r2_str"],
            "val_mae_numeric": exp["val_mae"],
        })

    comp_df = pd.DataFrame(comp_rows)
    comp_df_sorted = comp_df.sort_values("val_mae_numeric").drop(columns=["val_mae_numeric"])
    comparison_csv_path = output_dir / "model_comparison.csv"
    comp_df_sorted.to_csv(comparison_csv_path, index=False)
    print(f"  Saved comparison table: {comparison_csv_path}")
    print("\n" + comp_df_sorted.to_string(index=False) + "\n")

    # 7. Model Selection based strictly on Validation MAE
    best_exp = min(experiments, key=lambda x: x["val_mae"])
    best_model_name = best_exp["model_name"]
    best_target_transform = best_exp["target_transform"]
    best_model = best_exp["model_instance"]
    test_preds = best_exp["pred_test"]

    print("=" * 90)
    print(f"BEST MODEL SELECTED (by Validation MAE): {best_model_name} with {best_target_transform} target")
    print(f"  Validation MAE: {best_exp['val_mae_eur']}")
    print(f"  Test MAE (Out-of-time 2023/24): {best_exp['test_mae_eur']}")
    print(f"  Test RMSE: {best_exp['test_rmse_eur']}")
    print(f"  Test R²: {best_exp['test_r2_str']}")
    print("=" * 90)

    # 8. Save Selected Model Artifact & Metadata
    print("\n[5/7] Saving final model artifacts...")
    model_save_path = output_dir / "transfer_value_model.joblib"
    artifact = {
        "model": best_model,
        "preprocessor": preprocessor,
        "target_transform": best_target_transform,
        "model_name": best_model_name,
        "features": ALL_MODEL_FEATURES,
        "numerical_features": NUMERICAL_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "trained_date": datetime.utcnow().isoformat(),
    }
    joblib.dump(artifact, model_save_path)
    print(f"  Saved model artifact: {model_save_path}")

    # Metadata JSON
    metadata = {
        "model_type": best_model_name,
        "target_transformation": best_target_transform,
        "training_seasons": ["2017/18", "2018/19", "2019/20", "2020/21", "2021/22"],
        "validation_season": "2022/23",
        "test_season": "2023/24",
        "dataset_rows": {
            "train": len(train_df),
            "val": len(val_df),
            "test": len(test_df),
        },
        "feature_count": len(ALL_MODEL_FEATURES),
        "transformed_feature_count": len(feature_names),
        "features": ALL_MODEL_FEATURES,
        "validation_metrics": {
            "mae": best_exp["val_mae"],
            "mae_formatted": best_exp["val_mae_eur"],
            "rmse": best_exp["val_rmse"],
            "r2": best_exp["val_r2"],
        },
        "test_metrics": {
            "mae": best_exp["test_mae"],
            "mae_formatted": best_exp["test_mae_eur"],
            "rmse": best_exp["test_rmse"],
            "rmse_formatted": best_exp["test_rmse_eur"],
            "r2": best_exp["test_r2"],
            "r2_formatted": best_exp["test_r2_str"],
        },
        "training_timestamp": datetime.utcnow().isoformat() + "Z",
    }
    metadata_path = output_dir / "model_metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"  Saved model metadata: {metadata_path}")

    # 9. Error Analysis on Test Set
    print("\n[6/7] Performing Error Analysis on 2023/24 Test Set...")
    abs_errors = np.abs(y_test_raw - test_preds)
    pct_errors = np.where(y_test_raw > 0, (abs_errors / y_test_raw) * 100.0, 0.0)

    test_pred_df = pd.DataFrame({
        "player_id": test_df["player_id"].values,
        "player_name": test_df["player_name"].values,
        "season": test_df["season"].values,
        "actual_market_value_eur": y_test_raw,
        "predicted_market_value_eur": np.round(test_preds, 2),
        "absolute_error_eur": np.round(abs_errors, 2),
        "percentage_error": np.round(pct_errors, 2),
    })
    test_pred_path = output_dir / "test_predictions.csv"
    test_pred_df.to_csv(test_pred_path, index=False)
    print(f"  Saved test predictions: {test_pred_path}")

    # Error analysis insights
    top_abs_errors = test_pred_df.sort_values("absolute_error_eur", ascending=False).head(5)
    print("\nTop 5 Largest Absolute Errors:")
    for _, r in top_abs_errors.iterrows():
        print(f"  - {r['player_name']}: Actual {format_eur(r['actual_market_value_eur'])}, Predicted {format_eur(r['predicted_market_value_eur'])}, Error {format_eur(r['absolute_error_eur'])}")

    # 10. Feature Importance
    print("\n[7/7] Extracting Feature Importance and Generating Plots...")
    fi_list = []
    if hasattr(best_model, "feature_importances_"):
        raw_importances = best_model.feature_importances_
        for f_name, imp in zip(feature_names, raw_importances):
            fi_list.append({"feature": f_name, "importance": float(imp)})
    elif hasattr(best_model, "coef_"):
        raw_coefs = np.abs(best_model.coef_)
        for f_name, imp in zip(feature_names, raw_coefs):
            fi_list.append({"feature": f_name, "importance": float(imp)})

    fi_df = pd.DataFrame(fi_list).sort_values("importance", ascending=False).reset_index(drop=True)
    fi_path = output_dir / "feature_importance.csv"
    fi_df.to_csv(fi_path, index=False)
    print(f"  Saved feature importance: {fi_path}")

    print("\nTop 10 Important Features:")
    for idx, r in fi_df.head(10).iterrows():
        print(f"  {idx + 1:2d}. {r['feature']:<30} : {r['importance']:.4f}")

    # 11. Visualizations with Matplotlib
    # Plot 1: Actual vs Predicted
    plt.figure(figsize=(8, 6), dpi=150)
    plt.scatter(y_test_raw / 1_000_000, test_preds / 1_000_000, alpha=0.35, color="#1f77b4", edgecolor="none", s=25)
    max_val = max(y_test_raw.max(), test_preds.max()) / 1_000_000
    plt.plot([0, max_val], [0, max_val], color="#d62728", linestyle="--", linewidth=1.5, label="Perfect Fit (y = x)")
    plt.title(f"Actual vs Predicted Market Value (Test Set 2023/24) - {best_model_name}", fontsize=12, fontweight="bold")
    plt.xlabel("Actual Market Value (€ Millions)", fontsize=10)
    plt.ylabel("Predicted Market Value (€ Millions)", fontsize=10)
    plt.xlim(0, min(max_val * 1.05, 200))
    plt.ylim(0, min(max_val * 1.05, 200))
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(frameon=True)
    plt.tight_layout()
    plot1_path = plots_dir / "actual_vs_predicted.png"
    plt.savefig(plot1_path)
    plt.close()

    # Plot 2: Residual / Error Distribution
    residuals = y_test_raw - test_preds
    plt.figure(figsize=(8, 5), dpi=150)
    # Clip residuals for clean visualization between -€25M and +€25M
    res_m = residuals / 1_000_000
    res_clipped = np.clip(res_m, -25, 25)
    plt.hist(res_clipped, bins=50, color="#2ca02c", edgecolor="black", alpha=0.7)
    plt.axvline(0, color="#d62728", linestyle="--", linewidth=1.5, label="Zero Error")
    plt.title(f"Residual Distribution ({best_model_name})", fontsize=12, fontweight="bold")
    plt.xlabel("Residual (Actual - Predicted) in € Millions", fontsize=10)
    plt.ylabel("Player Count", fontsize=10)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plot2_path = plots_dir / "residual_distribution.png"
    plt.savefig(plot2_path)
    plt.close()

    # Plot 3: Feature Importance Bar Chart
    top15_fi = fi_df.head(15).iloc[::-1]  # reverse for horizontal bar chart
    plt.figure(figsize=(9, 6), dpi=150)
    plt.barh(top15_fi["feature"], top15_fi["importance"], color="#3b528b", edgecolor="black", alpha=0.85)
    plt.title(f"Top 15 Feature Importances ({best_model_name})", fontsize=12, fontweight="bold")
    plt.xlabel("Relative Importance Score", fontsize=10)
    plt.grid(True, linestyle=":", alpha=0.5, axis="x")
    plt.tight_layout()
    plot3_path = plots_dir / "feature_importance.png"
    plt.savefig(plot3_path)
    plt.close()

    # Plot 4: Test Predictions vs Actual (Log-Scale Scatter)
    plt.figure(figsize=(8, 6), dpi=150)
    plt.scatter(y_test_raw, test_preds, alpha=0.35, color="#9467bd", edgecolor="none", s=25)
    plt.xscale("log")
    plt.yscale("log")
    min_log = min(y_test_raw[y_test_raw > 0].min(), test_preds[test_preds > 0].min())
    max_log = max(y_test_raw.max(), test_preds.max())
    plt.plot([min_log, max_log], [min_log, max_log], color="#d62728", linestyle="--", label="y = x")
    plt.title(f"Predicted vs Actual Market Value (Log Scale) - {best_model_name}", fontsize=12, fontweight="bold")
    plt.xlabel("Actual Market Value (EUR, log scale)", fontsize=10)
    plt.ylabel("Predicted Market Value (EUR, log scale)", fontsize=10)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plot4_path = plots_dir / "test_predictions_scatter.png"
    plt.savefig(plot4_path)
    plt.close()

    print(f"  Saved 4 visualization plots to: {plots_dir}")
    print("\n" + "=" * 90)
    print("TRANSFER VALUE MODEL TRAINING COMPLETE")
    print("=" * 90 + "\n")

    return {
        "best_model_name": best_model_name,
        "best_target_transform": best_target_transform,
        "validation_mae": best_exp["val_mae_eur"],
        "test_mae": best_exp["test_mae_eur"],
        "test_rmse": best_exp["test_rmse_eur"],
        "test_r2": best_exp["test_r2_str"],
        "top_features": fi_df.head(10).to_dict("records"),
    }


if __name__ == "__main__":
    train_and_evaluate_transfer_models()
