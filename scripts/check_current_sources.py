import pandas as pd
import unicodedata

def normalize_name(name):
    if not isinstance(name, str):
        return ""
    norm = unicodedata.normalize('NFKD', name)
    norm = "".join(c for c in norm if not unicodedata.combining(c))
    return norm.strip().lower()

df_p = pd.read_csv('data/raw/transfermarkt/players.csv')
gb1_players = df_p[df_p['current_club_domestic_competition_id'] == 'GB1']
print(f'Total TM players: {len(df_p)}')
print(f'GB1 current players: {len(gb1_players)}')
print(f'GB1 clubs count: {gb1_players["current_club_name"].nunique()}')

test_names = ['Cherki', 'Haaland', 'Saka', 'Rice', 'van Dijk', 'Martínez']
print("\n--- Test Players Check in TM ---")
for t in test_names:
    norm_t = normalize_name(t)
    matches = df_p[df_p['name'].apply(normalize_name).str.contains(norm_t, na=False)]
    for idx, row in matches.iterrows():
        if row['current_club_domestic_competition_id'] == 'GB1' or row['name'] in ['Rayan Cherki', 'Erling Haaland', 'Bukayo Saka', 'Declan Rice', 'Virgil van Dijk', 'Emiliano Martínez']:
            print(f"Match: {row['name']} (ID: {row['player_id']}) | Club: {row['current_club_name']} | DoB: {row['date_of_birth']} | Last Season: {row['last_season']} | Pos: {row['position']}/{row['sub_position']} | Market Value: €{row['market_value_in_eur']}")

print("\n--- FBref Big 5 2025/26 Scouting Check ---")
df_sc = pd.read_csv('data/raw/scouting/big5_player_standard.csv')
sc_2026 = df_sc[df_sc['Season_End_Year'] == 2026]
print(f"FBref 2025/26 total player rows: {len(sc_2026)}")
print(f"FBref 2025/26 Premier League rows: {len(sc_2026[sc_2026['Comp'] == 'Premier League'])}")

for t in test_names:
    norm_t = normalize_name(t)
    matches = sc_2026[sc_2026['Player'].apply(normalize_name).str.contains(norm_t, na=False)]
    for idx, row in matches.iterrows():
        print(f"FBref 2026: {row['Player']} | Squad: {row['Squad']} | Comp: {row['Comp']} | Pos: {row['Pos']} | Age: {row['Age']} | Min: {row['Min_Playing']} | Gls: {row['Gls']} | Ast: {row['Ast']}")
