"""
Train and Evaluate Improved Transfer Value Predictor.
Champion model: XGBoost Regressor with log1p target and verified engineered features.
Validation set (2022/23) used for selection; final single evaluation on test set (2023/24).
"""
import os
import json
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, median_absolute_error

RANDOM_SEED = 42
DATASET_PATH = Path("data/processed/transfer/transfer_value_dataset.csv")
IMPROVEMENT_DIR = Path("models/improvement")
IMPROVEMENT_DIR.mkdir(parents=True, exist_ok=True)

def smape(y_true, y_pred):
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    denom = np.where(denom == 0, 1e-6, denom)
    return np.mean(np.abs(y_pred - y_true) / denom) * 100.0

def main():
    print("Loading transfer value dataset...")
    df = pd.read_csv(DATASET_PATH)

    # Feature engineering (validated on validation split)
    df["minutes_per_appearance"] = df["minutes_played"] / (df["appearances"] + 1e-5)
    df["cards_per90"] = (df["yellow_cards"] + 2 * df["red_cards"]) / (df["minutes_played"] / 90.0 + 1e-5)

    num_cols = [
        "age", "height_cm", "contract_years_left", "club_market_value_eur",
        "minutes_played", "appearances", "goals", "assists", "yellow_cards",
        "red_cards", "goals_per90", "assists_per90", "shots_per90", "npxg_per90",
        "prev_market_value_eur", "minutes_per_appearance", "cards_per90"
    ]
    cat_cols = ["primary_position", "sub_position", "preferred_foot"]
    all_feature_cols = num_cols + cat_cols

    train_df = df[df["split_group"] == "train"].copy()
    val_df = df[df["split_group"] == "val"].copy()
    test_df = df[df["split_group"] == "test"].copy()

    print(f"Dataset splits: Train={len(train_df):,} | Val={len(val_df):,} | Test={len(test_df):,}")

    preprocessor = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), num_cols),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")), ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), cat_cols)
    ])

    model = XGBRegressor(
        max_depth=7,
        learning_rate=0.03,
        n_estimators=500,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.2,
        reg_lambda=2.0,
        min_child_weight=4,
        random_state=RANDOM_SEED,
        n_jobs=-1
    )

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", model)
    ])

    print("Fitting model on training split (2017/18 - 2021/22)...")
    y_train_log = np.log1p(train_df["market_value_eur"])
    pipeline.fit(train_df[all_feature_cols], y_train_log)

    def evaluate_split(split_df, split_name):
        X = split_df[all_feature_cols]
        y_true = split_df["market_value_eur"].values
        preds_log = pipeline.predict(X)
        preds = np.clip(np.expm1(preds_log), 0, None)

        mae = mean_absolute_error(y_true, preds)
        rmse = np.sqrt(mean_squared_error(y_true, preds))
        r2 = r2_score(y_true, preds)
        medae = median_absolute_error(y_true, preds)
        sm = smape(y_true, preds)

        # Bucketed MAE
        buckets = [
            ("< €10M", y_true < 10e6),
            ("€10–30M", (y_true >= 10e6) & (y_true < 30e6)),
            ("€30–60M", (y_true >= 30e6) & (y_true < 60e6)),
            ("€60–100M", (y_true >= 60e6) & (y_true < 100e6)),
            ("> €100M", y_true >= 100e6),
        ]
        bucket_results = {}
        for b_name, mask in buckets:
            if np.sum(mask) > 0:
                bucket_results[b_name] = {
                    "count": int(np.sum(mask)),
                    "mae": float(mean_absolute_error(y_true[mask], preds[mask]))
                }

        # Age group MAE
        age = split_df["age"].values
        age_buckets = [
            ("U21", age < 21),
            ("21-25", (age >= 21) & (age <= 25)),
            ("26-30", (age > 25) & (age <= 30)),
            ("31+", age > 30),
        ]
        age_results = {}
        for a_name, mask in age_buckets:
            if np.sum(mask) > 0:
                age_results[a_name] = {
                    "count": int(np.sum(mask)),
                    "mae": float(mean_absolute_error(y_true[mask], preds[mask]))
                }

        # Position MAE
        pos_results = {}
        for pos in split_df["primary_position"].unique():
            mask = (split_df["primary_position"] == pos).values
            if np.sum(mask) > 0:
                pos_results[pos] = {
                    "count": int(np.sum(mask)),
                    "mae": float(mean_absolute_error(y_true[mask], preds[mask]))
                }

        print(f"\n--- {split_name} Evaluation ---")
        print(f"MAE: €{mae/1e6:.4f}M ({mae:,.0f})")
        print(f"RMSE: €{rmse/1e6:.4f}M ({rmse:,.0f})")
        print(f"R²: {r2:.4f}")
        print(f"MedAE: €{medae/1e6:.4f}M ({medae:,.0f})")
        print(f"SMAPE: {sm:.2f}%")
        print("Value Buckets MAE:")
        for b_name, b_info in bucket_results.items():
            print(f"  {b_name:10s} (N={b_info['count']:4d}): €{b_info['mae']/1e6:.3f}M")

        return {
            "mae": float(mae),
            "mae_formatted": f"€{mae/1e6:.3f}M",
            "rmse": float(rmse),
            "rmse_formatted": f"€{rmse/1e6:.3f}M",
            "r2": float(r2),
            "median_absolute_error": float(medae),
            "smape": float(sm),
            "bucket_results": bucket_results,
            "age_results": age_results,
            "position_results": pos_results,
            "preds": preds
        }

    val_res = evaluate_split(val_df, "Validation Split (2022/23)")
    test_res = evaluate_split(test_df, "Test Split (2023/24 - Final Holdout)")

    # Error analysis CSV
    test_df_out = test_df.copy()
    test_df_out["predicted_value"] = test_res["preds"]
    test_df_out["actual_value"] = test_df_out["market_value_eur"]
    test_df_out["absolute_error"] = np.abs(test_df_out["actual_value"] - test_df_out["predicted_value"])
    test_df_out["percentage_error"] = (test_df_out["absolute_error"] / np.clip(test_df_out["actual_value"], 1e5, None)) * 100.0

    analysis_cols = [
        "player_name", "actual_value", "predicted_value", "absolute_error",
        "percentage_error", "age", "primary_position", "sub_position",
        "prev_market_value_eur", "club_name"
    ]
    err_csv = test_df_out[analysis_cols].rename(columns={
        "player_name": "player",
        "primary_position": "position",
        "prev_market_value_eur": "previous_value"
    }).sort_values("absolute_error", ascending=False)

    err_analysis_path = IMPROVEMENT_DIR / "transfer_error_analysis.csv"
    err_csv.to_csv(err_analysis_path, index=False)
    print(f"\nSaved transfer error analysis to: {err_analysis_path}")

    # Save improved model
    model_artifact_path = IMPROVEMENT_DIR / "transfer_value_model_improved.joblib"
    joblib.dump(pipeline, model_artifact_path)
    print(f"Saved improved transfer model artifact: {model_artifact_path}")

    # Save improved metadata
    metadata = {
        "model_type": "XGBoost Regressor (Improved Champion)",
        "target_transformation": "log1p",
        "random_seed": RANDOM_SEED,
        "hyperparameters": {
            "max_depth": 7,
            "learning_rate": 0.03,
            "n_estimators": 500,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.2,
            "reg_lambda": 2.0,
            "min_child_weight": 4
        },
        "features": {
            "numerical": num_cols,
            "categorical": cat_cols,
            "engineered": ["minutes_per_appearance", "cards_per90"]
        },
        "dataset_rows": {
            "train": len(train_df),
            "val": len(val_df),
            "test": len(test_df)
        },
        "validation_metrics": {k: v for k, v in val_res.items() if k != "preds"},
        "test_metrics": {k: v for k, v in test_res.items() if k != "preds"},
        "timestamp": pd.Timestamp.now().isoformat()
    }

    meta_path = IMPROVEMENT_DIR / "transfer_model_improved_metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved improved transfer metadata to: {meta_path}")

if __name__ == "__main__":
    main()
