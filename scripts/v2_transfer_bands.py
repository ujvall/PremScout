import sys
from pathlib import Path
sys.path.insert(0, str(Path(".").resolve()))
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error
from scripts.train_improved_transfer import smape

# Load data and models
df = pd.read_csv("data/processed/transfer/transfer_value_dataset.csv")
df = df.sort_values(["player_id", "season"]).reset_index(drop=True)

df["minutes_per_appearance"] = df["minutes_played"] / (df["appearances"] + 1e-5)
df["cards_per90"] = (df["yellow_cards"] + 2 * df["red_cards"]) / (df["minutes_played"] / 90.0 + 1e-5)
df["age_x_minutes"] = (df["age"] * df["minutes_played"]) / 1000.0
df["age_x_goals"] = df["age"] * df["goals_per90"]
df["contract_x_prev_val"] = (df["contract_years_left"].fillna(2.0) * df["prev_market_value_eur"]) / 1e6
df["youth_starter_premium"] = np.where((df["age"] <= 23) & (df["minutes_per_appearance"] >= 65), 1.0, 0.0)
df["veteran_depreciation"] = np.maximum(0.0, df["age"] - 30.0)

df["lag_prev_val"] = df.groupby("player_id")["prev_market_value_eur"].shift(1)
df["val_growth_abs"] = (df["prev_market_value_eur"] - df["lag_prev_val"]).fillna(0.0)
df["val_growth_pct"] = np.clip((df["prev_market_value_eur"] - df["lag_prev_val"]) / (df["lag_prev_val"] + 1e5), -1.0, 5.0).fillna(0.0)
df["lag_mins"] = df.groupby("player_id")["minutes_played"].shift(1)
df["mins_growth"] = (df["minutes_played"] - df["lag_mins"]).fillna(0.0)

train_df = df[df["split_group"] == "train"].copy()
val_df = df[df["split_group"] == "val"].copy()

y_train_log = np.log1p(train_df["market_value_eur"])
y_val = val_df["market_value_eur"].values

# V1 predictions
v1_pipe = joblib.load("models/improvement/transfer_value_model_improved.joblib")
v1_cols = [
    "age", "height_cm", "contract_years_left", "club_market_value_eur",
    "minutes_played", "appearances", "goals", "assists", "yellow_cards",
    "red_cards", "goals_per90", "assists_per90", "shots_per90", "npxg_per90",
    "prev_market_value_eur", "minutes_per_appearance", "cards_per90",
    "primary_position", "sub_position", "preferred_foot"
]
p_v1 = np.clip(np.expm1(v1_pipe.predict(val_df[v1_cols])), 0, None)

# V2 predictions (Ensemble of XGBoost + HistGradientBoosting)
from scripts.v2_transfer_models import models, prep, num_cols, cat_cols, opt_w

X_tr = prep.fit_transform(train_df[num_cols + cat_cols])
X_v = prep.transform(val_df[num_cols + cat_cols])

m_xgb = models["XGBoost_Tuned"]
m_xgb.fit(X_tr, y_train_log)
p_xgb = np.clip(np.expm1(m_xgb.predict(X_v)), 0, None)

m_hgb = models["HistGradBoost"]
m_hgb.fit(X_tr, y_train_log)
p_hgb = np.clip(np.expm1(m_hgb.predict(X_v)), 0, None)

p_v2 = opt_w[0] * p_xgb + opt_w[1] * p_hgb

bands = [
    ("< €5M", y_val < 5e6),
    ("€5M–€10M", (y_val >= 5e6) & (y_val < 10e6)),
    ("€10M–€25M", (y_val >= 10e6) & (y_val < 25e6)),
    ("€25M–€50M", (y_val >= 25e6) & (y_val < 50e6)),
    ("€50M–€100M", (y_val >= 50e6) & (y_val < 100e6)),
    ("> €100M", y_val >= 100e6),
]

print("=== VALUE-BAND VALIDATION ANALYSIS (V1 vs V2) ===")
print(f"{'Band':12s} | {'Count':6s} | {'V1 MAE':12s} | {'V2 MAE':12s} | {'Diff MAE':10s} | {'V1 MedAE':12s} | {'V2 MedAE':12s}")
print("-" * 88)

for b_name, mask in bands:
    cnt = np.sum(mask)
    if cnt == 0:
        continue
    y_sub = y_val[mask]
    mae_1 = mean_absolute_error(y_sub, p_v1[mask])
    mae_2 = mean_absolute_error(y_sub, p_v2[mask])
    diff_mae = mae_2 - mae_1
    med_1 = median_absolute_error(y_sub, p_v1[mask])
    med_2 = median_absolute_error(y_sub, p_v2[mask])
    print(f"{b_name:12s} | {cnt:6d} | EUR {mae_1/1e6:.3f}M   | EUR {mae_2/1e6:.3f}M   | EUR {diff_mae/1e6:+.3f}M | EUR {med_1/1e6:.3f}M   | EUR {med_2/1e6:.3f}M")
