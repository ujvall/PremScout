import sys
from pathlib import Path
sys.path.insert(0, str(Path(".").resolve()))
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import Ridge, ElasticNet
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, median_absolute_error
from scripts.train_improved_transfer import smape

# 1. Load data
df = pd.read_csv("data/processed/transfer/transfer_value_dataset.csv")
df = df.sort_values(["player_id", "season"]).reset_index(drop=True)

# Base improved features
df["minutes_per_appearance"] = df["minutes_played"] / (df["appearances"] + 1e-5)
df["cards_per90"] = (df["yellow_cards"] + 2 * df["red_cards"]) / (df["minutes_played"] / 90.0 + 1e-5)

# --- 1A. Candidate Feature Engineering ---
# Interaction features
df["age_x_minutes"] = (df["age"] * df["minutes_played"]) / 1000.0
df["age_x_goals"] = df["age"] * df["goals_per90"]
df["age_x_npxg"] = df["age"] * df["npxg_per90"]
df["age_x_prev_val"] = (df["age"] * df["prev_market_value_eur"]) / 1e6
df["contract_x_prev_val"] = (df["contract_years_left"].fillna(2.0) * df["prev_market_value_eur"]) / 1e6
df["contract_x_age"] = df["contract_years_left"].fillna(2.0) * df["age"]
df["youth_starter_premium"] = np.where((df["age"] <= 23) & (df["minutes_per_appearance"] >= 65), 1.0, 0.0)
df["veteran_depreciation"] = np.maximum(0.0, df["age"] - 30.0)

# Performance trajectory (Strictly past seasons lag differences per player)
df["lag_prev_val"] = df.groupby("player_id")["prev_market_value_eur"].shift(1)
df["val_growth_abs"] = (df["prev_market_value_eur"] - df["lag_prev_val"]).fillna(0.0)
df["val_growth_pct"] = np.clip((df["prev_market_value_eur"] - df["lag_prev_val"]) / (df["lag_prev_val"] + 1e5), -1.0, 5.0).fillna(0.0)

df["lag_mins"] = df.groupby("player_id")["minutes_played"].shift(1)
df["mins_growth"] = (df["minutes_played"] - df["lag_mins"]).fillna(0.0)

train_df = df[df["split_group"] == "train"].copy()
val_df = df[df["split_group"] == "val"].copy()

y_train_log = np.log1p(train_df["market_value_eur"])
y_val = val_df["market_value_eur"].values

v1_num = [
    "age", "height_cm", "contract_years_left", "club_market_value_eur",
    "minutes_played", "appearances", "goals", "assists", "yellow_cards",
    "red_cards", "goals_per90", "assists_per90", "shots_per90", "npxg_per90",
    "prev_market_value_eur", "minutes_per_appearance", "cards_per90"
]
cat_cols = ["primary_position", "sub_position", "preferred_foot"]

feature_candidates = {
    "V1_Champion": v1_num,
    "+ Age_Interactions": v1_num + ["age_x_minutes", "age_x_goals", "age_x_npxg", "age_x_prev_val"],
    "+ Contract_Interactions": v1_num + ["contract_x_prev_val", "contract_x_age"],
    "+ Youth_Veteran_Flags": v1_num + ["youth_starter_premium", "veteran_depreciation"],
    "+ Valuation_Trajectory": v1_num + ["val_growth_abs", "val_growth_pct", "mins_growth"],
    "+ All_Interactions": v1_num + [
        "age_x_minutes", "age_x_goals", "contract_x_prev_val",
        "youth_starter_premium", "veteran_depreciation", "val_growth_pct"
    ]
}

print(f"{'Feature Set':30s} | {'Val MAE':14s} | {'Val RMSE':14s} | {'Val R2':8s} | {'Val MedAE':14s} | {'Val SMAPE':10s}")
print("-" * 105)

best_features = v1_num
best_mae = 1541595.09
best_f_name = "V1_Champion"

for f_name, num_cols in feature_candidates.items():
    prep = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), num_cols),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")), ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), cat_cols)
    ])
    
    X_tr = prep.fit_transform(train_df[num_cols + cat_cols])
    X_v = prep.transform(val_df[num_cols + cat_cols])
    
    m = XGBRegressor(
        max_depth=7,
        learning_rate=0.03,
        n_estimators=500,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.2,
        reg_lambda=2.0,
        min_child_weight=4,
        random_state=42,
        n_jobs=-1
    )
    m.fit(X_tr, y_train_log)
    preds = np.clip(np.expm1(m.predict(X_v)), 0, None)
    
    mae = mean_absolute_error(y_val, preds)
    rmse = np.sqrt(mean_squared_error(y_val, preds))
    r2 = r2_score(y_val, preds)
    medae = median_absolute_error(y_val, preds)
    sm = smape(y_val, preds)
    
    print(f"{f_name:30s} | EUR {mae/1e6:.4f}M     | EUR {rmse/1e6:.4f}M     | {r2:.4f}   | EUR {medae/1e6:.4f}M     | {sm:.2f}%")
    if mae < best_mae:
        best_mae = mae
        best_features = num_cols
        best_f_name = f_name

print(f"\nBest Feature Candidate: {best_f_name} (Val MAE: EUR {best_mae:,.2f})")
