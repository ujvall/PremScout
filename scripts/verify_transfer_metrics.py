import sys
from pathlib import Path
sys.path.insert(0, str(Path(".").resolve()))
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, median_absolute_error
from scripts.train_improved_transfer import smape

# 1. Load dataset
df = pd.read_csv("data/processed/transfer/transfer_value_dataset.csv")

# Engineer improved features
df["minutes_per_appearance"] = df["minutes_played"] / (df["appearances"] + 1e-5)
df["cards_per90"] = (df["yellow_cards"] + 2 * df["red_cards"]) / (df["minutes_played"] / 90.0 + 1e-5)

test_df = df[df["split_group"] == "test"].copy()
print(f"Transfer Test rows: {len(test_df)}")

y_true = test_df["market_value_eur"].values

# 2. Reload Baseline Model
print("\n=== EVALUATING BASELINE TRANSFER MODEL ON TEST SET ===")
base_bundle = joblib.load("models/baseline/transfer/transfer_value_model.joblib")
base_model = base_bundle["model"]
base_preprocessor = base_bundle["preprocessor"]
base_cols = base_bundle["features"]

X_test_base = base_preprocessor.transform(test_df[base_cols])
base_preds_log = base_model.predict(X_test_base)
base_preds = np.clip(np.expm1(base_preds_log), 0, None)

base_mae = mean_absolute_error(y_true, base_preds)
base_rmse = np.sqrt(mean_squared_error(y_true, base_preds))
base_r2 = r2_score(y_true, base_preds)
base_medae = median_absolute_error(y_true, base_preds)
base_smape = smape(y_true, base_preds)

print(f"Baseline MAE:    €{base_mae:,.2f} (€{base_mae/1e6:.3f}M)")
print(f"Baseline RMSE:   €{base_rmse:,.2f} (€{base_rmse/1e6:.3f}M)")
print(f"Baseline R²:     {base_r2:.4f}")
print(f"Baseline MedAE:  €{base_medae:,.2f} (€{base_medae/1e6:.3f}M)")
print(f"Baseline SMAPE:  {base_smape:.2f}%")

# 3. Reload Improved Model
print("\n=== EVALUATING IMPROVED TRANSFER MODEL ON TEST SET ===")
impr_pipeline = joblib.load("models/improvement/transfer_value_model_improved.joblib")

impr_num = [
    "age", "height_cm", "contract_years_left", "club_market_value_eur",
    "minutes_played", "appearances", "goals", "assists", "yellow_cards",
    "red_cards", "goals_per90", "assists_per90", "shots_per90", "npxg_per90",
    "prev_market_value_eur", "minutes_per_appearance", "cards_per90"
]
impr_cat = ["primary_position", "sub_position", "preferred_foot"]
impr_cols = impr_num + impr_cat

impr_preds_log = impr_pipeline.predict(test_df[impr_cols])
impr_preds = np.clip(np.expm1(impr_preds_log), 0, None)

impr_mae = mean_absolute_error(y_true, impr_preds)
impr_rmse = np.sqrt(mean_squared_error(y_true, impr_preds))
impr_r2 = r2_score(y_true, impr_preds)
impr_medae = median_absolute_error(y_true, impr_preds)
impr_smape = smape(y_true, impr_preds)

print(f"Improved MAE:    €{impr_mae:,.2f} (€{impr_mae/1e6:.3f}M)")
print(f"Improved RMSE:   €{impr_rmse:,.2f} (€{impr_rmse/1e6:.3f}M)")
print(f"Improved R²:     {impr_r2:.4f}")
print(f"Improved MedAE:  €{impr_medae:,.2f} (€{impr_medae/1e6:.3f}M)")
print(f"Improved SMAPE:  {impr_smape:.2f}%")

diff_mae = impr_mae - base_mae
diff_rmse = impr_rmse - base_rmse
diff_r2 = impr_r2 - base_r2
diff_medae = impr_medae - base_medae
diff_smape = impr_smape - base_smape

print("\n=== DELTA (IMPROVED - BASELINE) ===")
print(f"Delta MAE:   EUR {diff_mae:+,.2f} ({diff_mae/base_mae*100:+.2f}%)")
print(f"Delta RMSE:  EUR {diff_rmse:+,.2f} ({diff_rmse/base_rmse*100:+.2f}%)")
print(f"Delta R2:    {diff_r2:+.4f} ({diff_r2/base_r2*100:+.2f}%)")
print(f"Delta MedAE: EUR {diff_medae:+,.2f} ({diff_medae/base_medae*100:+.2f}%)")
print(f"Delta SMAPE: {diff_smape:+.2f}% ({diff_smape/base_smape*100:+.2f}%)")

# Compare against reported CSV
print("\n=== CHECKING AGAINST REPORTED CSV ===")
comp_df = pd.read_csv("models/improvement/MODEL_COMPARISON_BEFORE_AFTER.csv")
trans_comp = comp_df[comp_df["model"] == "Transfer Value Predictor"]

checks = [
    ("MAE (EUR)", base_mae, impr_mae),
    ("RMSE (EUR)", base_rmse, impr_rmse),
    ("R2 Score", base_r2, impr_r2),
    ("SMAPE (%)", base_smape, impr_smape)
]

for metric_name, b_val, i_val in checks:
    row = trans_comp[trans_comp["metric"] == metric_name]
    if not row.empty:
        rep_base = float(row["baseline_test"].values[0])
        rep_impr = float(row["improved_test"].values[0])
        print(f"Metric: {metric_name:15s} | Calc Base: {b_val:,.2f} (Rep: {rep_base:,.2f}) | Calc Impr: {i_val:,.2f} (Rep: {rep_impr:,.2f})")
