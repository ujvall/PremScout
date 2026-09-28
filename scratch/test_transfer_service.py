import sys
sys.path.insert(0, ".")
from backend.services.transfer_service import get_player_transfer_features_service, get_transfer_presets_service
for name in ['Rayan Cherki', 'Erling Haaland', 'Bukayo Saka', 'Declan Rice']:
    info = get_player_transfer_features_service(name)
    f = info['features']
    print(f"{name:15} -> Age: {f['age']}, ClubVal: €{f['club_market_value_eur']/1e6:.0f}M, PrevVal: €{f['prev_market_value_eur']/1e6:.0f}M, Pos: {f['primary_position']}/{f['sub_position']}")

presets = get_transfer_presets_service()
print("Presets count:", len(presets["presets"]))
for p in presets["presets"]:
    print(f"Preset: {p['name']} ({p['club']}) - {p['description']}")
