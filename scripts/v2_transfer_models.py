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
from scipy.optimize import minimize
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, median_absolute_error
from scripts.train_improved_transfer import smape

# 1. Load data
df = pd.read_csv("data/processed/transfer/transfer_value_dataset.csv")
df = df.sort_values(["player_id", "season"]).reset_index(drop=True)

# Feature engineering
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

num_cols = [
    "age", "height_cm", "contract_years_left", "club_market_value_eur",
    "minutes_played", "appearances", "goals", "assists", "yellow_cards",
    "red_cards", "goals_per90", "assists_per90", "shots_per90", "npxg_per90",
    "prev_market_value_eur", "minutes_per_appearance", "cards_per90",
    "age_x_minutes", "age_x_goals", "contract_x_prev_val",
    "youth_starter_premium", "veteran_depreciation", "val_growth_pct"
]
cat_cols = ["primary_position", "sub_position", "preferred_foot"]

prep = ColumnTransformer([
    ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), num_cols),
    ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")), ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), cat_cols)
])

X_tr = prep.fit_transform(train_df[num_cols + cat_cols])
X_v = prep.transform(val_df[num_cols + cat_cols])

models = {
    "XGBoost_Tuned": XGBRegressor(
        max_depth=7, learning_rate=0.03, n_estimators=500,
        subsample=0.8, colsample_bytree=0.8, reg_alpha=0.2,
        reg_lambda=2.0, min_child_weight=4, random_state=42, n_jobs=-1
    ),
    "HistGradBoost": HistGradientBoostingRegressor(
        max_iter=350, max_depth=8, learning_rate=0.04,
        min_samples_leaf=20, l2_regularization=1.0, random_state=42
    ),
    "RandomForest": RandomForestRegressor(
        n_estimators=200, max_depth=16, min_samples_leaf=4,
        random_state=42, n_jobs=-1
    ),
    "Ridge_Baseline": Ridge(alpha=10.0, random_state=42)
}

preds_dict = {}

print("=== EVALUATING MODEL ARCHITECTURES ON VALIDATION SPLIT (2022/23) ===")
print(f"{'Model':20s} | {'Val MAE':14s} | {'Val RMSE':14s} | {'Val R2':8s} | {'Val MedAE':14s} | {'Val SMAPE':10s}")
print("-" * 95)

for m_name, model in models.items():
    model.fit(X_tr, y_train_log)
    p_log = model.predict(X_v)
    p_eur = np.clip(np.expm1(p_log), 0, None)
    preds_dict[m_name] = p_eur
    
    mae = mean_absolute_error(y_val, p_eur)
    rmse = np.sqrt(mean_squared_error(y_val, p_eur))
    r2 = r2_score(y_val, p_eur)
    medae = median_absolute_error(y_val, p_eur)
    sm = smape(y_val, p_eur)
    print(f"{m_name:20s} | EUR {mae/1e6:.4f}M     | EUR {rmse/1e6:.4f}M     | {r2:.4f}   | EUR {medae/1e6:.4f}M     | {sm:.2f}%")

# Test blending on validation set
print("\n=== OPTIMIZING ENSEMBLE BLEND ON VALIDATION SET ===")
# Optimization objective: minimize validation MAE over convex combination of predictions
p_xgb = preds_dict["XGBoost_Tuned"]
p_hgb = preds_dict["HistGradBoost"]
p_rf = preds_dict["RandomForest"]

def loss_func(weights):
    w1, w2, w3 = weights
    blend = w1 * p_xgb + w2 * p_hgb + w3 * p_rf
    return mean_absolute_error(y_val, blend)

res = minimize(
    loss_func,
    x0=[0.6, 0.3, 0.1],
    bounds=[(0, 1), (0, 1), (0, 1)],
    constraints={"type": "eq", "fun": lambda w: sum(w) - 1.0}
)
opt_w = res.x
print(f"Optimal validation weights: XGB={opt_w[0]:.3f}, HGB={opt_w[1]:.3f}, RF={opt_w[2]:.3f}")

p_blend = opt_w[0] * p_xgb + opt_w[1] * p_hgb + opt_w[2] * p_rf
mae_b = mean_absolute_error(y_val, p_blend)
rmse_b = np.sqrt(mean_squared_error(y_val, p_blend))
r2_b = r2_score(y_val, p_blend)
medae_b = median_absolute_error(y_val, p_blend)
sm_b = smape(y_val, p_blend)

print(f"{'Ensemble_Blend':20s} | EUR {mae_b/1e6:.4f}M     | EUR {rmse_b/1e6:.4f}M     | {r2_b:.4f}   | EUR {medae_b/1e6:.4f}M     | {sm_b:.2f}%")
print(f"V1 Champion Benchmark: EUR 1.5416M MAE | EUR 3.9253M RMSE | 0.8989 R2 | EUR 0.3235M MedAE | 26.37% SMAPE")
