import sys
from pathlib import Path
sys.path.insert(0, str(Path(".").resolve()))
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    log_loss, confusion_matrix
)
from scripts.train_improved_match import compute_sequential_elo

# 1. Load data
raw_df = pd.read_csv("data/processed/matches/match_outcome_dataset.csv")
raw_df["date"] = pd.to_datetime(raw_df["date"])

# For improved model, compute sequential Elo
df_elo, _ = compute_sequential_elo(raw_df)

# Test split (2023/24)
test_raw = raw_df[raw_df["split_group"] == "test"].copy()
test_elo = df_elo[df_elo["split_group"] == "test"].copy()

print(f"Test rows count: {len(test_raw)} (raw) / {len(test_elo)} (elo)")

# Ground truth
c_to_i = {"H": 0, "D": 1, "A": 2}
y_test = test_raw["match_result"].map(c_to_i).values
y_test_oh = np.eye(3)[y_test]

# 2. Reload Baseline Model
print("\n=== EVALUATING BASELINE MATCH MODEL ON TEST SET ===")
base_bundle = joblib.load("models/baseline/match/match_outcome_model.joblib")
base_pipeline = base_bundle["pipeline"]
base_feats = base_bundle["features"]

X_test_base = test_raw[base_feats].values
base_probs = base_pipeline.predict_proba(X_test_base)
base_preds = base_pipeline.predict(X_test_base)

base_acc = accuracy_score(y_test, base_preds)
base_mf1 = f1_score(y_test, base_preds, average="macro", zero_division=0)
base_wf1 = f1_score(y_test, base_preds, average="weighted", zero_division=0)
base_ll = log_loss(y_test, base_probs)
base_brier = np.mean(np.sum((base_probs - y_test_oh) ** 2, axis=1))

base_p = precision_score(y_test, base_preds, average=None, zero_division=0)
base_r = recall_score(y_test, base_preds, average=None, zero_division=0)
base_f1 = f1_score(y_test, base_preds, average=None, zero_division=0)
base_cm = confusion_matrix(y_test, base_preds, labels=[0, 1, 2])

print(f"Accuracy:    {base_acc:.4f}")
print(f"Macro F1:    {base_mf1:.4f}")
print(f"Weighted F1: {base_wf1:.4f}")
print(f"Log Loss:    {base_ll:.4f}")
print(f"Brier Score: {base_brier:.4f}")
print("Per-class [H, D, A]:")
for i, c in enumerate(["Home (H)", "Draw (D)", "Away (A)"]):
    print(f"  {c:10s} - P: {base_p[i]:.4f} | R: {base_r[i]:.4f} | F1: {base_f1[i]:.4f}")
print("Confusion Matrix:")
print(base_cm)

# 3. Reload Improved Model
print("\n=== EVALUATING IMPROVED MATCH MODEL ON TEST SET ===")
impr_bundle = joblib.load("models/improvement/match_outcome_model_improved.joblib")
impr_pipeline = impr_bundle["pipeline"]
impr_feats = impr_bundle["features"]

X_test_impr = test_elo[impr_feats].values
impr_probs = impr_pipeline.predict_proba(X_test_impr)
impr_preds = impr_pipeline.predict(X_test_impr)

impr_acc = accuracy_score(y_test, impr_preds)
impr_mf1 = f1_score(y_test, impr_preds, average="macro", zero_division=0)
impr_wf1 = f1_score(y_test, impr_preds, average="weighted", zero_division=0)
impr_ll = log_loss(y_test, impr_probs)
impr_brier = np.mean(np.sum((impr_probs - y_test_oh) ** 2, axis=1))

impr_p = precision_score(y_test, impr_preds, average=None, zero_division=0)
impr_r = recall_score(y_test, impr_preds, average=None, zero_division=0)
impr_f1 = f1_score(y_test, impr_preds, average=None, zero_division=0)
impr_cm = confusion_matrix(y_test, impr_preds, labels=[0, 1, 2])

print(f"Accuracy:    {impr_acc:.4f}")
print(f"Macro F1:    {impr_mf1:.4f}")
print(f"Weighted F1: {impr_wf1:.4f}")
print(f"Log Loss:    {impr_ll:.4f}")
print(f"Brier Score: {impr_brier:.4f}")
print("Per-class [H, D, A]:")
for i, c in enumerate(["Home (H)", "Draw (D)", "Away (A)"]):
    print(f"  {c:10s} - P: {impr_p[i]:.4f} | R: {impr_r[i]:.4f} | F1: {impr_f1[i]:.4f}")
print("Confusion Matrix:")
print(impr_cm)

# 4. Compare against MODEL_COMPARISON_BEFORE_AFTER.csv
print("\n=== CHECKING DISCREPANCIES AGAINST REPORTED CSV ===")
comp_df = pd.read_csv("models/improvement/MODEL_COMPARISON_BEFORE_AFTER.csv")
match_comp = comp_df[comp_df["model"] == "Match Outcome Predictor"]

checks = [
    ("Accuracy", base_acc, impr_acc),
    ("Macro F1", base_mf1, impr_mf1),
    ("Weighted F1", base_wf1, impr_wf1),
    ("Log Loss", base_ll, impr_ll),
    ("Brier Score", base_brier, impr_brier),
    ("Draw Recall", base_r[1], impr_r[1]),
    ("Draw F1", base_f1[1], impr_f1[1]),
]

for metric_name, b_val, i_val in checks:
    row = match_comp[match_comp["metric"] == metric_name]
    if not row.empty:
        rep_base = float(row["baseline_test"].values[0])
        rep_impr = float(row["improved_test"].values[0])
        diff_base = abs(b_val - rep_base)
        diff_impr = abs(i_val - rep_impr)
        status = "PASS" if (diff_base <= 0.001 and diff_impr <= 0.001) else "DISCREPANCY"
        print(f"Metric: {metric_name:15s} | Calc Base: {b_val:.4f} (Rep: {rep_base:.4f}, diff={diff_base:.5f}) | Calc Impr: {i_val:.4f} (Rep: {rep_impr:.4f}, diff={diff_impr:.5f}) | Status: {status}")
        assert diff_base <= 0.001 and diff_impr <= 0.001, f"Discrepancy > 0.001 in {metric_name}!"
