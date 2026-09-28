import sys
from pathlib import Path
sys.path.insert(0, str(Path(".").resolve()))
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scripts.train_improved_match import compute_sequential_elo

print("=" * 60)
print("1. TRANSFER PREDICTION DIFFERENCES ON TEST SET (2023/24)")
print("=" * 60)
t_df = pd.read_csv("data/processed/transfer/transfer_value_dataset.csv")
t_df["minutes_per_appearance"] = t_df["minutes_played"] / (t_df["appearances"] + 1e-5)
t_df["cards_per90"] = (t_df["yellow_cards"] + 2 * t_df["red_cards"]) / (t_df["minutes_played"] / 90.0 + 1e-5)
test_t = t_df[t_df["split_group"] == "test"].copy().reset_index(drop=True)

base_t_bundle = joblib.load("models/baseline/transfer/transfer_value_model.joblib")
impr_t_pipe = joblib.load("models/improvement/transfer_value_model_improved.joblib")

X_base_t = base_t_bundle["preprocessor"].transform(test_t[base_t_bundle["features"]])
preds_base_t = np.clip(np.expm1(base_t_bundle["model"].predict(X_base_t)), 0, None)

impr_cols = [
    "age", "height_cm", "contract_years_left", "club_market_value_eur",
    "minutes_played", "appearances", "goals", "assists", "yellow_cards",
    "red_cards", "goals_per90", "assists_per90", "shots_per90", "npxg_per90",
    "prev_market_value_eur", "minutes_per_appearance", "cards_per90",
    "primary_position", "sub_position", "preferred_foot"
]
preds_impr_t = np.clip(np.expm1(impr_t_pipe.predict(test_t[impr_cols])), 0, None)

abs_diff_t = np.abs(preds_impr_t - preds_base_t)
pct_diff_t = (abs_diff_t / np.clip(preds_base_t, 1e5, None)) * 100.0

# How many changed by at least 1%
changed_t_1pct = (pct_diff_t >= 1.0).mean() * 100.0
changed_t_any = (abs_diff_t > 1.0).mean() * 100.0

print(f"Total test players: {len(test_t)}")
print(f"Percentage of predictions changed (>1% relative): {changed_t_1pct:.2f}%")
print(f"Percentage of predictions changed (any non-zero diff): {changed_t_any:.2f}%")
print(f"Mean absolute change: EUR {abs_diff_t.mean():,.2f}")
print(f"Median absolute change: EUR {np.median(abs_diff_t):,.2f}")
print(f"Max absolute change: EUR {abs_diff_t.max():,.2f}")

test_t["base_pred"] = preds_base_t
test_t["impr_pred"] = preds_impr_t
test_t["abs_diff"] = abs_diff_t
test_t["actual"] = test_t["market_value_eur"]

top_changes_t = test_t.sort_values("abs_diff", ascending=False).head(5)
print("\nTop 5 Largest Transfer Valuation Changes:")
for _, r in top_changes_t.iterrows():
    p_name = r['player_name'].encode('ascii', 'replace').decode('ascii')
    c_name = r['club_name'].encode('ascii', 'replace').decode('ascii')
    print(f"  {p_name:22s} ({c_name:18s}) | Actual: EUR {r['actual']/1e6:6.1f}M | Base: EUR {r['base_pred']/1e6:6.1f}M -> Impr: EUR {r['impr_pred']/1e6:6.1f}M | Diff: EUR {r['abs_diff']/1e6:+.2f}M")

print("\n" + "=" * 60)
print("2. MATCH PREDICTION DIFFERENCES ON TEST SET (2023/24)")
print("=" * 60)
raw_m = pd.read_csv("data/processed/matches/match_outcome_dataset.csv")
raw_m["date"] = pd.to_datetime(raw_m["date"])
df_elo, _ = compute_sequential_elo(raw_m)
test_m_raw = raw_m[raw_m["split_group"] == "test"].copy().reset_index(drop=True)
test_m_elo = df_elo[df_elo["split_group"] == "test"].copy().reset_index(drop=True)

base_m_bundle = joblib.load("models/baseline/match/match_outcome_model.joblib")
impr_m_bundle = joblib.load("models/improvement/match_outcome_model_improved.joblib")

preds_base_m = base_m_bundle["pipeline"].predict(test_m_raw[base_m_bundle["features"]].values)
probs_base_m = base_m_bundle["pipeline"].predict_proba(test_m_raw[base_m_bundle["features"]].values)

preds_impr_m = impr_m_bundle["pipeline"].predict(test_m_elo[impr_m_bundle["features"]].values)
probs_impr_m = impr_m_bundle["pipeline"].predict_proba(test_m_elo[impr_m_bundle["features"]].values)

class_diff = (preds_base_m != preds_impr_m)
prob_l1_diff = np.abs(probs_impr_m - probs_base_m).sum(axis=1) / 2.0  # Total variation distance

int_to_c = {0: "H", 1: "D", 2: "A"}
c_to_i = {"H": 0, "D": 1, "A": 2}
y_true_m = test_m_raw["match_result"].map(c_to_i).values

base_draws = (preds_base_m == 1).sum()
impr_draws = (preds_impr_m == 1).sum()

print(f"Total test matches: {len(test_m_raw)}")
print(f"Percentage of predicted classes changed: {class_diff.mean()*100:.2f}% ({class_diff.sum()} matches)")
print(f"Mean probability shift (Total Variation Distance): {prob_l1_diff.mean()*100:.2f}%")
print(f"Predicted draws: Baseline = {base_draws} -> Improved = {impr_draws} (+{impr_draws - base_draws} new draw predictions)")

# Find interesting examples where improved model was correct while baseline was wrong
test_m_raw["base_pred"] = [int_to_c[p] for p in preds_base_m]
test_m_raw["impr_pred"] = [int_to_c[p] for p in preds_impr_m]
test_m_raw["actual"] = test_m_raw["match_result"]
test_m_raw["prob_diff"] = prob_l1_diff

flip_to_correct = test_m_raw[(test_m_raw["base_pred"] != test_m_raw["actual"]) & (test_m_raw["impr_pred"] == test_m_raw["actual"])]
print(f"\nMatches where Improved model FLIPPED to CORRECT outcome ({len(flip_to_correct)} fixtures):")
for _, r in flip_to_correct.head(5).iterrows():
    print(f"  {str(r['date'])[:10]} | {r['home_team']:18s} vs {r['away_team']:18s} | Actual: {r['actual']} | Base: {r['base_pred']} -> Impr: {r['impr_pred']}")

print("\n" + "=" * 60)
print("3. SCOUTING PREDICTION DIFFERENCES ACROSS ALL PLAYERS")
print("=" * 60)
base_s_art = joblib.load("models/baseline/scouting/scouting_model.joblib")
impr_s_art = joblib.load("models/improvement/scouting_model_improved.joblib")

s_df = impr_s_art["player_df"].reset_index(drop=True)
base_s_emb = base_s_art["embeddings"]
impr_s_emb = impr_s_art["embeddings"]

sim_base_all = cosine_similarity(base_s_emb)
sim_impr_all = cosine_similarity(impr_s_emb)

# Mask self and same player
top5_changed_count = 0
overlaps = []
n_s = len(s_df)

for i in range(n_s):
    p_name = s_df.loc[i, "norm_name"]
    same_p = (s_df["norm_name"] == p_name).values
    
    r_b = sim_base_all[i].copy()
    r_b[same_p] = -999.0
    top5_b = set(np.argsort(r_b)[-5:])
    
    r_i = sim_impr_all[i].copy()
    r_i[same_p] = -999.0
    if s_df.loc[i, "norm_pos"] == "GK":
        r_i[s_df["norm_pos"] != "GK"] = -999.0
    else:
        r_i[s_df["norm_pos"] == "GK"] = -999.0
    top5_i = set(np.argsort(r_i)[-5:])
    
    overlap = len(top5_b.intersection(top5_i)) / 5.0
    overlaps.append(overlap)
    if overlap < 1.0:
        top5_changed_count += 1

print(f"Total player-season queries: {n_s}")
print(f"Percentage of Top-5 lists changed: {top5_changed_count / n_s * 100:.2f}%")
print(f"Average Top-5 Jaccard/List Overlap: {np.mean(overlaps)*100:.2f}%")
