import sys
from pathlib import Path
sys.path.insert(0, str(Path(".").resolve()))
import pandas as pd
import numpy as np
from scipy.stats import poisson
from scipy.optimize import minimize
from xgboost import XGBClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, log_loss, confusion_matrix

# 1. Load match dataset
df = pd.read_csv("data/processed/matches/match_outcome_dataset.csv")
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values(["date", "match_id"]).reset_index(drop=True)

# --- 2A & 2B. Advanced Pre-Match Features & Margin-of-Victory Elo ---
def compute_v2_match_features(raw_df: pd.DataFrame, elo_k=20.0, use_mov=True):
    df_sorted = raw_df.sort_values(["date", "match_id"]).copy().reset_index(drop=True)
    
    # Track sequential team states
    elo_ratings = {}
    team_history = {} # list of match dicts strictly before current match
    
    home_elos, away_elos, elo_diffs, elo_exp_h = [], [], [], []
    poisson_p_h, poisson_p_d, poisson_p_a = [], [], []
    form3_h_pts, form3_a_pts = [], []
    form10_h_pts, form10_a_pts = [], []
    h_att_strength, a_att_strength = [], []
    h_def_strength, a_def_strength = [], []
    
    HOME_ADVANTAGE = 65.0
    
    for idx, row in df_sorted.iterrows():
        h_team, a_team = row["home_team"], row["away_team"]
        
        # 1. Pre-match Elo
        h_elo = elo_ratings.get(h_team, 1500.0)
        a_elo = elo_ratings.get(a_team, 1500.0)
        diff = (h_elo + HOME_ADVANTAGE) - a_elo
        exp_h = 1.0 / (1.0 + 10.0 ** (-diff / 400.0))
        
        home_elos.append(h_elo)
        away_elos.append(a_elo)
        elo_diffs.append(diff)
        elo_exp_h.append(exp_h)
        
        # 2. Sequential Multi-Window Form & Poisson Goal Model
        h_hist = team_history.get(h_team, [])
        a_hist = team_history.get(a_team, [])
        
        # Form last 3 matches
        h_pts_3 = np.mean([m["pts"] for m in h_hist[-3:]]) if len(h_hist) >= 3 else (row["home_rolling_points_5"] / 5.0 * 3.0)
        a_pts_3 = np.mean([m["pts"] for m in a_hist[-3:]]) if len(a_hist) >= 3 else (row["away_rolling_points_5"] / 5.0 * 3.0)
        form3_h_pts.append(h_pts_3)
        form3_a_pts.append(a_pts_3)
        
        # Form last 10 matches
        h_pts_10 = np.mean([m["pts"] for m in h_hist[-10:]]) if len(h_hist) >= 5 else (row["home_rolling_points_5"] * 2.0)
        a_pts_10 = np.mean([m["pts"] for m in a_hist[-10:]]) if len(a_hist) >= 5 else (row["away_rolling_points_5"] * 2.0)
        form10_h_pts.append(h_pts_10)
        form10_a_pts.append(a_pts_10)
        
        # Poisson goal lambdas from rolling goals (prior 5-10 matches)
        # Home league average ~ 1.5, Away ~ 1.2
        h_gf_avg = row["home_rolling_gf_5"] / 5.0
        h_ga_avg = row["home_rolling_ga_5"] / 5.0
        a_gf_avg = row["away_rolling_gf_5"] / 5.0
        a_ga_avg = row["away_rolling_ga_5"] / 5.0
        
        lambda_h = np.clip(1.35 * (h_gf_avg / 1.35 + 1e-4) * (a_ga_avg / 1.35 + 1e-4), 0.4, 3.5)
        lambda_a = np.clip(1.15 * (a_gf_avg / 1.15 + 1e-4) * (h_ga_avg / 1.15 + 1e-4), 0.3, 3.0)
        
        # Derive Poisson probabilities
        max_g = 6
        p_matrix = np.zeros((max_g, max_g))
        for g_h in range(max_g):
            for g_a in range(max_g):
                p_matrix[g_h, g_a] = poisson.pmf(g_h, lambda_h) * poisson.pmf(g_a, lambda_a)
        
        # Normalize
        p_matrix = p_matrix / p_matrix.sum()
        p_h = float(np.sum(np.tril(p_matrix, -1).T)) # g_h > g_a
        p_d = float(np.sum(np.diag(p_matrix)))        # g_h == g_a
        p_a = float(np.sum(np.triu(p_matrix, 1).T))  # g_h < g_a
        
        # Dixon-Coles adjustment for low scores (0-0, 1-0, 0-1, 1-1)
        rho = -0.05
        adj_00 = max(0.0, 1.0 - lambda_h * lambda_a * rho)
        adj_10 = max(0.0, 1.0 + lambda_a * rho)
        adj_01 = max(0.0, 1.0 + lambda_h * rho)
        adj_11 = max(0.0, 1.0 - rho)
        p_matrix[0, 0] *= adj_00
        p_matrix[1, 0] *= adj_10
        p_matrix[0, 1] *= adj_01
        p_matrix[1, 1] *= adj_11
        p_matrix = p_matrix / p_matrix.sum()
        
        p_h = float(np.sum([p_matrix[i, j] for i in range(max_g) for j in range(max_g) if i > j]))
        p_d = float(np.sum([p_matrix[i, i] for i in range(max_g)]))
        p_a = float(np.sum([p_matrix[i, j] for i in range(max_g) for j in range(max_g) if i < j]))
        
        poisson_p_h.append(p_h)
        poisson_p_d.append(p_d)
        poisson_p_a.append(p_a)
        
        h_att_strength.append(h_gf_avg)
        h_def_strength.append(h_ga_avg)
        a_att_strength.append(a_gf_avg)
        a_def_strength.append(a_ga_avg)
        
        # 3. Post-match Elo update (strictly after pre-match features recorded)
        res = row["match_result"]
        s_h = 1.0 if res == "H" else (0.5 if res == "D" else 0.0)
        s_a = 1.0 - s_h
        
        # Post-match goal difference for MOV multiplier if enabled
        k_eff = elo_k
        if use_mov:
            # Synthetic or default margin approximation from result: H/A win avg GD ~ 1.5, Draw GD = 0
            mov_mult = 1.0 if res == "D" else 1.2
            k_eff = elo_k * mov_mult
            
        elo_ratings[h_team] = h_elo + k_eff * (s_h - exp_h)
        elo_ratings[a_team] = a_elo + k_eff * (s_a - (1.0 - exp_h))
        
        # Update team history
        pts_h = 3 if res == "H" else (1 if res == "D" else 0)
        pts_a = 3 if res == "A" else (1 if res == "D" else 0)
        if h_team not in team_history:
            team_history[h_team] = []
        team_history[h_team].append({"pts": pts_h})
        if a_team not in team_history:
            team_history[a_team] = []
        team_history[a_team].append({"pts": pts_a})
        
    df_sorted["home_elo"] = home_elos
    df_sorted["away_elo"] = away_elos
    df_sorted["elo_diff"] = elo_diffs
    df_sorted["elo_exp_h"] = elo_exp_h
    df_sorted["abs_elo_diff"] = np.abs(df_sorted["elo_diff"])
    df_sorted["abs_rank_diff"] = np.abs(df_sorted["rank_diff"])
    df_sorted["abs_ppg_diff"] = np.abs(df_sorted["home_ppg"] - df_sorted["away_ppg"])
    
    df_sorted["poisson_p_h"] = poisson_p_h
    df_sorted["poisson_p_d"] = poisson_p_d
    df_sorted["poisson_p_a"] = poisson_p_a
    df_sorted["form3_h_pts"] = form3_h_pts
    df_sorted["form3_a_pts"] = form3_a_pts
    df_sorted["form10_h_pts"] = form10_h_pts
    df_sorted["form10_a_pts"] = form10_a_pts
    df_sorted["h_att_strength"] = h_att_strength
    df_sorted["h_def_strength"] = h_def_strength
    df_sorted["a_att_strength"] = a_att_strength
    df_sorted["a_def_strength"] = a_def_strength
    
    # Parity index: closeness in attack and defence
    df_sorted["att_def_parity"] = np.abs(df_sorted["h_att_strength"] - df_sorted["a_att_strength"]) + np.abs(df_sorted["h_def_strength"] - df_sorted["a_def_strength"])
    
    return df_sorted, elo_ratings

print("Building candidate match features...")
df_v2, final_elos = compute_v2_match_features(df, elo_k=22.0, use_mov=True)

c_to_i = {"H": 0, "D": 1, "A": 2}
df_v2["target"] = df_v2["match_result"].map(c_to_i)

train_m = df_v2[df_v2["split_group"] == "train"].copy()
val_m = df_v2[df_v2["split_group"] == "val"].copy()

y_tr = train_m["target"].values
y_v = val_m["target"].values
y_v_oh = np.eye(3)[y_v]

# Feature sets
base_feats = [
    "home_rest_days", "away_rest_days", "home_rolling_points_5", "away_rolling_points_5",
    "home_rolling_gf_5", "away_rolling_gf_5", "home_rolling_ga_5", "away_rolling_ga_5",
    "home_rolling_shots_5", "away_rolling_shots_5", "home_rolling_sot_5", "away_rolling_sot_5",
    "home_rolling_sot_conceded_5", "away_rolling_sot_conceded_5", "home_rolling_corners_5",
    "away_rolling_corners_5", "home_rolling_win_pct_5", "away_rolling_win_pct_5",
    "home_league_rank", "away_league_rank", "rank_diff", "home_ppg", "away_ppg",
    "h2h_matches_prior", "h2h_home_win_pct"
]

v1_feats = base_feats + [
    "home_elo", "away_elo", "elo_diff", "elo_exp_h", "abs_elo_diff", "abs_rank_diff", "abs_ppg_diff"
]

v2_extra_feats = [
    "poisson_p_h", "poisson_p_d", "poisson_p_a",
    "form3_h_pts", "form3_a_pts", "form10_h_pts", "form10_a_pts",
    "att_def_parity"
]

v2_all_feats = v1_feats + v2_extra_feats

# Evaluate V1 feature set with XGBoost
xgb_v1 = XGBClassifier(n_estimators=150, max_depth=4, learning_rate=0.03, random_state=42, eval_metric="mlogloss", n_jobs=-1)
xgb_v1.fit(train_m[v1_feats].values, y_tr)
probs_v1 = xgb_v1.predict_proba(val_m[v1_feats].values)
preds_v1 = probs_v1.argmax(axis=1)

print("=== V1 CHAMPION ON VALIDATION SPLIT ===")
print(f"Accuracy: {accuracy_score(y_v, preds_v1):.4f} | Macro F1: {f1_score(y_v, preds_v1, average='macro'):.4f} | LogLoss: {log_loss(y_v, probs_v1):.4f} | Draw Recall: {np.mean((preds_v1==1)[y_v==1]):.4f}")

# Evaluate V2 feature set with XGBoost
xgb_v2 = XGBClassifier(n_estimators=140, max_depth=4, learning_rate=0.03, subsample=0.85, colsample_bytree=0.8, reg_alpha=0.2, random_state=42, eval_metric="mlogloss", n_jobs=-1)
xgb_v2.fit(train_m[v2_all_feats].values, y_tr)
probs_xgb_v2 = xgb_v2.predict_proba(val_m[v2_all_feats].values)
preds_xgb_v2 = probs_xgb_v2.argmax(axis=1)

print("\n=== V2 XGBOOST WITH POISSON & FORM EXPANSION ON VALIDATION SPLIT ===")
print(f"Accuracy: {accuracy_score(y_v, preds_xgb_v2):.4f} | Macro F1: {f1_score(y_v, preds_xgb_v2, average='macro'):.4f} | LogLoss: {log_loss(y_v, probs_xgb_v2):.4f} | Draw Recall: {np.mean((preds_xgb_v2==1)[y_v==1]):.4f}")

# Pure Poisson model baseline
poisson_val_probs = val_m[["poisson_p_h", "poisson_p_d", "poisson_p_a"]].values
poisson_val_preds = poisson_val_probs.argmax(axis=1)
print("\n=== PURE POISSON GOAL MODEL BASELINE ON VALIDATION ===")
print(f"Accuracy: {accuracy_score(y_v, poisson_val_preds):.4f} | Macro F1: {f1_score(y_v, poisson_val_preds, average='macro'):.4f} | LogLoss: {log_loss(y_v, poisson_val_probs):.4f} | Draw Recall: {np.mean((poisson_val_preds==1)[y_v==1]):.4f}")

# Test Hybrid Probability Blending: w * XGBoost_probs + (1 - w) * Poisson_probs
print("\n=== HYBRID PROBABILITY BLENDING ON VALIDATION SPLIT ===")
best_w = 1.0
best_ll = float("inf")
for w in np.linspace(0.5, 1.0, 21):
    p_blend = w * probs_xgb_v2 + (1.0 - w) * poisson_val_probs
    p_blend = p_blend / p_blend.sum(axis=1, keepdims=True)
    ll = log_loss(y_v, p_blend)
    acc = accuracy_score(y_v, p_blend.argmax(axis=1))
    mf1 = f1_score(y_v, p_blend.argmax(axis=1), average="macro")
    d_rec = np.mean((p_blend.argmax(axis=1) == 1)[y_v == 1])
    d_f1 = f1_score(y_v, p_blend.argmax(axis=1), average=None, zero_division=0)[1]
    if ll < best_ll:
        best_ll = ll
        best_w = w
    print(f"w_xgb={w:.2f}: Acc={acc:.4f} | Macro F1={mf1:.4f} | LogLoss={ll:.4f} | Draw Rec={d_rec:.4f} | Draw F1={d_f1:.4f}")

print(f"\nBest Hybrid Blend: w_xgb = {best_w:.2f} (LogLoss: {best_ll:.4f})")
