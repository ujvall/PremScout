import sys
from pathlib import Path
sys.path.insert(0, str(Path(".").resolve()))
import pandas as pd
import numpy as np
from scipy.optimize import minimize
from xgboost import XGBClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, log_loss
from scripts.train_improved_match import compute_sequential_elo

raw_df = pd.read_csv("data/processed/matches/match_outcome_dataset.csv")
raw_df["date"] = pd.to_datetime(raw_df["date"])

# Test different K factors and MOV in sequential Elo
print("=== TESTING ELO PARAMETER TUNING ON VALIDATION SPLIT ===")
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

c_to_i = {"H": 0, "D": 1, "A": 2}
raw_df["target"] = raw_df["match_result"].map(c_to_i)

for k_val in [16.0, 20.0, 24.0, 28.0]:
    # Custom elo
    df_sorted = raw_df.sort_values(["date", "match_id"]).copy().reset_index(drop=True)
    elo_ratings = {}
    home_elos, away_elos, elo_diffs, elo_exp_h = [], [], [], []
    for _, row in df_sorted.iterrows():
        h_team, a_team = row["home_team"], row["away_team"]
        h_elo = elo_ratings.get(h_team, 1500.0)
        a_elo = elo_ratings.get(a_team, 1500.0)
        home_elos.append(h_elo)
        away_elos.append(a_elo)
        diff = (h_elo + 65.0) - a_elo
        exp_h = 1.0 / (1.0 + 10.0 ** (-diff / 400.0))
        elo_diffs.append(diff)
        elo_exp_h.append(exp_h)
        res = row["match_result"]
        s_h = 1.0 if res == "H" else (0.5 if res == "D" else 0.0)
        s_a = 1.0 - s_h
        elo_ratings[h_team] = h_elo + k_val * (s_h - exp_h)
        elo_ratings[a_team] = a_elo + k_val * (s_a - (1.0 - exp_h))
        
    df_sorted["home_elo"] = home_elos
    df_sorted["away_elo"] = away_elos
    df_sorted["elo_diff"] = elo_diffs
    df_sorted["elo_exp_h"] = elo_exp_h
    df_sorted["abs_elo_diff"] = np.abs(df_sorted["elo_diff"])
    df_sorted["abs_rank_diff"] = np.abs(df_sorted["rank_diff"])
    df_sorted["abs_ppg_diff"] = np.abs(df_sorted["home_ppg"] - df_sorted["away_ppg"])
    
    tr = df_sorted[df_sorted["split_group"] == "train"]
    va = df_sorted[df_sorted["split_group"] == "val"]
    
    m = XGBClassifier(n_estimators=150, max_depth=4, learning_rate=0.03, random_state=42, eval_metric="mlogloss", n_jobs=-1)
    m.fit(tr[all_features].values, tr["target"].values)
    probs = m.predict_proba(va[all_features].values)
    preds = probs.argmax(axis=1)
    y_v = va["target"].values
    
    acc = accuracy_score(y_v, preds)
    mf1 = f1_score(y_v, preds, average="macro")
    ll = log_loss(y_v, probs)
    d_rec = np.mean((preds == 1)[y_v == 1])
    print(f"K={k_val:.1f} | Acc={acc:.4f} | Macro F1={mf1:.4f} | LogLoss={ll:.4f} | Draw Rec={d_rec:.4f}")

# Model ensembling on K=20.0
print("\n=== MODEL ARCHITECTURE COMPARISON & ENSEMBLE ON VALIDATION SPLIT ===")
df_elo_20, _ = compute_sequential_elo(raw_df)
tr_20 = df_elo_20[df_elo_20["split_group"] == "train"]
va_20 = df_elo_20[df_elo_20["split_group"] == "val"]
y_tr = tr_20["target"].values
y_v = va_20["target"].values

m_xgb = XGBClassifier(n_estimators=150, max_depth=4, learning_rate=0.03, random_state=42, eval_metric="mlogloss", n_jobs=-1)
m_xgb.fit(tr_20[all_features].values, y_tr)
p_xgb = m_xgb.predict_proba(va_20[all_features].values)

m_rf = RandomForestClassifier(n_estimators=250, max_depth=6, min_samples_leaf=4, random_state=42, n_jobs=-1)
m_rf.fit(tr_20[all_features].values, y_tr)
p_rf = m_rf.predict_proba(va_20[all_features].values)

m_hgb = HistGradientBoostingClassifier(max_iter=120, max_depth=4, learning_rate=0.03, min_samples_leaf=15, random_state=42)
m_hgb.fit(tr_20[all_features].values, y_tr)
p_hgb = m_hgb.predict_proba(va_20[all_features].values)

for name, p in [("XGBoost", p_xgb), ("RandomForest", p_rf), ("HistGradBoost", p_hgb)]:
    pred = p.argmax(axis=1)
    acc = accuracy_score(y_v, pred)
    mf1 = f1_score(y_v, pred, average="macro")
    ll = log_loss(y_v, p)
    print(f"{name:16s} | Acc={acc:.4f} | Macro F1={mf1:.4f} | LogLoss={ll:.4f}")

# Optimized probability blend of tree models
def blend_loss(weights):
    w1, w2, w3 = weights
    pb = w1 * p_xgb + w2 * p_rf + w3 * p_hgb
    return log_loss(y_v, pb)

res = minimize(blend_loss, x0=[0.6, 0.2, 0.2], bounds=[(0, 1), (0, 1), (0, 1)], constraints={"type": "eq", "fun": lambda w: sum(w) - 1.0})
opt_w = res.x
print(f"\nOptimal Tree Blend Weights: XGB={opt_w[0]:.3f}, RF={opt_w[1]:.3f}, HGB={opt_w[2]:.3f}")
p_tree_blend = opt_w[0] * p_xgb + opt_w[1] * p_rf + opt_w[2] * p_hgb
pred_b = p_tree_blend.argmax(axis=1)
acc_b = accuracy_score(y_v, pred_b)
mf1_b = f1_score(y_v, pred_b, average="macro")
ll_b = log_loss(y_v, p_tree_blend)
d_rec_b = np.mean((pred_b == 1)[y_v == 1])
d_f1_b = f1_score(y_v, pred_b, average=None, zero_division=0)[1]

print(f"Tree Ensemble Blend: Acc={acc_b:.4f} | Macro F1={mf1_b:.4f} | LogLoss={ll_b:.4f} | Draw Rec={d_rec_b:.4f} | Draw F1={d_f1_b:.4f}")
print(f"V1 Benchmark:        Acc=0.5579 | Macro F1=0.4473 | LogLoss=0.9737 | Draw Rec=0.0690 | Draw F1=0.1143")
