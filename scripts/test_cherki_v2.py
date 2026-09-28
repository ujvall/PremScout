import joblib
import pandas as pd
import numpy as np

artifact = joblib.load('models/scouting/scouting_model.joblib')
scaler = artifact['scaler']
features = artifact['features']
pos_weights = artifact.get('position_weights', {})

curr_scout = pd.read_csv('data/current/scouting/current_player_scouting.csv')
cherki = curr_scout[curr_scout['norm_name'] == 'rayan cherki']

raw_df = cherki[features]
scaled_vec = scaler.transform(raw_df)
norm_pos = cherki['norm_pos'].values[0]

if norm_pos in pos_weights:
    w = pos_weights[norm_pos]
    scaled_vec = scaled_vec * w

query_vec = scaled_vec / np.linalg.norm(scaled_vec)

hist_emb = artifact['embeddings']
hist_norms = np.linalg.norm(hist_emb, axis=1, keepdims=True)
hist_norms[hist_norms == 0] = 1.0
hist_norm_emb = hist_emb / hist_norms

sims = np.dot(hist_norm_emb, query_vec.T).flatten()
hist_df = artifact['player_df']
top_indices = np.argsort(sims)[::-1][:10]

print("Top 5 similar players for Rayan Cherki via production V2 model:")
for i in top_indices[:5]:
    row = hist_df.iloc[i]
    score = round(float(np.clip(sims[i], 0, 1)) * 100, 1)
    print(f"  {row['player_name']} ({row['team']}) | Pos: {row['position']} | Season: {row['season']} | Sim: {score}/100")
