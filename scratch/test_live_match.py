import urllib.request, json

# 1. Fetch fixture features
url = "http://127.0.0.1:8000/api/match/current-fixture-features?home_team=Arsenal&away_team=Chelsea"
feat_resp = json.loads(urllib.request.urlopen(url).read().decode())
print("Fixture features success:", feat_resp["success"])
print("Home Elo:", feat_resp["features"]["home_elo"], "Away Elo:", feat_resp["features"]["away_elo"])

# 2. Predict match outcome
req = urllib.request.Request(
    "http://127.0.0.1:8000/api/match/predict",
    data=json.dumps(feat_resp["features"]).encode("utf-8"),
    headers={"Content-Type": "application/json"}
)
res = json.loads(urllib.request.urlopen(req).read().decode())
print(f"Match: {res['home_team']} vs {res['away_team']} -> {res['predicted_outcome_label']} (H:{res['probabilities']['home_win']:.1%}, D:{res['probabilities']['draw']:.1%}, A:{res['probabilities']['away_win']:.1%}) | Data As-Of: {res['data_as_of']}")
