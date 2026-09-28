import urllib.request, json

presets_data = json.loads(urllib.request.urlopen("http://127.0.0.1:8000/api/transfer/presets").read().decode())
for p in presets_data["presets"]:
    req = urllib.request.Request(
        "http://127.0.0.1:8000/api/transfer/predict",
        data=json.dumps(p["features"]).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )
    res = json.loads(urllib.request.urlopen(req).read().decode())
    print(f"{res['player_name']:18} -> Predicted: {res['predicted_market_value_formatted']} ({res['model_type']})")
