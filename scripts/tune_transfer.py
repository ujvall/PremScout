"""
Systematic Hyperparameter Tuning for Transfer Value Predictor on Validation Set (2022/23).
Does NOT touch test set (2023/24).
"""
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

df = pd.read_csv("data/processed/transfer/transfer_value_dataset.csv")
df = df.sort_values(["player_id", "season"]).reset_index(drop=True)

# Historical momentum (prior season lag relative change)
df["prev2_val"] = df.groupby("player_id")["prev_market_value_eur"].shift(1).fillna(df["prev_market_value_eur"])
df["val_momentum"] = (df["prev_market_value_eur"] - df["prev2_val"]) / (df["prev2_val"] + 1e5)
df["player_club_val_ratio"] = df["prev_market_value_eur"] / (df["club_market_value_eur"] + 1e5)
df["minutes_per_appearance"] = df["minutes_played"] / (df["appearances"] + 1e-5)
df["goal_contribution_rate"] = df["goals_per90"] + df["assists_per90"]
df["age_squared"] = (df["age"] - 26.0) ** 2
df["contract_security"] = df["contract_years_left"].fillna(2.0) * df["minutes_played"] / 1000.0

train = df[df["split_group"] == "train"].copy()
val = df[df["split_group"] == "val"].copy()

num_cols = [
    "age", "height_cm", "contract_years_left", "club_market_value_eur",
    "minutes_played", "appearances", "goals", "assists", "yellow_cards",
    "red_cards", "goals_per90", "assists_per90", "shots_per90", "npxg_per90",
    "prev_market_value_eur", "player_club_val_ratio", "minutes_per_appearance",
    "goal_contribution_rate", "age_squared", "contract_security", "val_momentum"
]
cat_cols = ["primary_position", "sub_position", "preferred_foot"]

preprocessor = ColumnTransformer([
    ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), num_cols),
    ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")), ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), cat_cols)
])

X_tr = preprocessor.fit_transform(train[num_cols + cat_cols])
y_tr_log = np.log1p(train["market_value_eur"])

X_v = preprocessor.transform(val[num_cols + cat_cols])
y_v = val["market_value_eur"].values

configs = [
    {"name": "Cfg1_D6_LR05", "max_depth": 6, "learning_rate": 0.05, "n_estimators": 350, "subsample": 0.8, "colsample_bytree": 0.8, "reg_alpha": 0.1, "reg_lambda": 1.0, "min_child_weight": 3},
    {"name": "Cfg2_D6_LR04", "max_depth": 6, "learning_rate": 0.04, "n_estimators": 450, "subsample": 0.85, "colsample_bytree": 0.85, "reg_alpha": 0.05, "reg_lambda": 1.5, "min_child_weight": 2},
    {"name": "Cfg3_D7_LR03", "max_depth": 7, "learning_rate": 0.03, "n_estimators": 500, "subsample": 0.8, "colsample_bytree": 0.8, "reg_alpha": 0.2, "reg_lambda": 2.0, "min_child_weight": 4},
    {"name": "Cfg4_D5_LR05", "max_depth": 5, "learning_rate": 0.05, "n_estimators": 400, "subsample": 0.85, "colsample_bytree": 0.8, "reg_alpha": 0.1, "reg_lambda": 1.0, "min_child_weight": 3},
    {"name": "Cfg5_D7_LR04", "max_depth": 7, "learning_rate": 0.04, "n_estimators": 450, "subsample": 0.85, "colsample_bytree": 0.85, "reg_alpha": 0.1, "reg_lambda": 1.0, "min_child_weight": 2},
    {"name": "Cfg6_D6_LR035", "max_depth": 6, "learning_rate": 0.035, "n_estimators": 550, "subsample": 0.85, "colsample_bytree": 0.8, "reg_alpha": 0.15, "reg_lambda": 1.2, "min_child_weight": 3},
]

print(f"{'Config':16s} | {'Val MAE':12s} | {'Val RMSE':12s} | {'Val R2':8s}")
print("-" * 55)

best_cfg = None
best_mae = float("inf")

for cfg in configs:
    params = {k: v for k, v in cfg.items() if k != "name"}
    m = XGBRegressor(**params, random_state=42, n_jobs=-1)
    m.fit(X_tr, y_tr_log)
    preds = np.clip(np.expm1(m.predict(X_v)), 0, None)
    mae = mean_absolute_error(y_v, preds)
    rmse = np.sqrt(mean_squared_error(y_v, preds))
    r2 = r2_score(y_v, preds)
    print(f"{cfg['name']:16s} | €{mae/1e6:.3f}M      | €{rmse/1e6:.3f}M      | {r2:.4f}")
    if mae < best_mae:
        best_mae = mae
        best_cfg = cfg

print("\nBest Configuration on Validation Set:")
print(best_cfg, f"Val MAE: €{best_mae/1e6:.3f}M")
