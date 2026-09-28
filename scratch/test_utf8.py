import pandas as pd
import unicodedata

def normalize_name(name: str) -> str:
    if not isinstance(name, str):
        return ""
    norm = unicodedata.normalize('NFKD', name)
    norm = "".join(c for c in norm if not unicodedata.combining(c))
    norm = norm.replace("'", "").replace("-", " ").replace(".", "")
    return " ".join(norm.lower().split())

df = pd.read_csv('data/raw/transfermarkt/players.csv', encoding='utf-8')
row = df[df['player_id'] == 111873].iloc[0]
print("Name:", row['name'])
print("Normalized:", normalize_name(row['name']))
print("Equal?", normalize_name(row['name']) == "emiliano martinez")
