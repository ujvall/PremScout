import pandas as pd
from backend.ml.transfer.predict import predict_transfer_value

df_tr = pd.read_csv('data/current/players/current_transfer_features.csv')
for name in ['rayan cherki', 'erling haaland', 'bukayo saka', 'declan rice']:
    match = df_tr[df_tr['normalized_name'] == name]
    if match.empty:
        print("Missing:", name)
        continue
    row = match.iloc[0]
    feat = {k: row[k] for k in [
        'age', 'height_cm', 'contract_years_left', 'club_market_value_eur',
        'minutes_played', 'appearances', 'goals', 'assists', 'yellow_cards',
        'red_cards', 'goals_per90', 'assists_per90', 'shots_per90', 'npxg_per90',
        'prev_market_value_eur', 'primary_position', 'sub_position', 'preferred_foot'
    ]}
    res = predict_transfer_value(feat)
    print(f"{row['player_name']} ({row['club']}, Age {row['age']}, {row['season']}) -> Pred: {res['formatted_market_value']} (EUR {res['predicted_market_value_eur']:,.2f})")
