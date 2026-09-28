import urllib.request
import urllib.parse
import json

players = ['Declan Rice', 'Erling Haaland', 'Bukayo Saka', 'Virgil van Dijk', 'Emiliano Martínez']

print("--- 1. Testing Similar Players Endpoint ---")
for p in players:
    url = f"http://127.0.0.1:8000/api/scouting/similar/{urllib.parse.quote(p)}?top_n=5"
    res = json.loads(urllib.request.urlopen(url).read().decode())
    target = p.lower()
    candidate_names = [c['candidate_player'].lower() for c in res]
    assert target not in candidate_names, f"Target {p} found in candidates!"
    print(f"=== {p} ===")
    for c in res[:3]:
        print(f"  {c['candidate_player']} ({c['candidate_club']}) | pos: {c['candidate_position']} | score: {c['similarity_score']:.1f}")
        assert 0 <= c['similarity_score'] <= 100, f"Score out of bounds: {c['similarity_score']}"
    if 'Martínez' in p:
        for c in res:
            assert c['candidate_position'] == 'GK', f"Goalkeeper cross contamination: {c}"
        print("  -> Strict GK Isolation verified! 100% of candidates are GKs.")

print("\n--- 2. Testing Replacement Search with Recruitment Filters ---")
# Declan Rice replacement search with budget cap €40M, age <= 26, min_minutes 1000, same_position_only True
payload = {
    "player_name": "Declan Rice",
    "top_n": 5,
    "same_position_only": True,
    "max_market_value": 40000000.0,
    "max_age": 26,
    "min_minutes": 1000
}
req = urllib.request.Request(
    "http://127.0.0.1:8000/api/scouting/replacements",
    data=json.dumps(payload).encode("utf-8"),
    headers={"Content-Type": "application/json"}
)
res = json.loads(urllib.request.urlopen(req).read().decode())
assert len(res) > 0, "No replacements returned!"
print(f"Replacement search for Declan Rice (Budget <= €40M, Age <= 26, Min mins 1000):")
for r in res:
    print(f"  {r['candidate_player']} ({r['candidate_club']}) | age: {r['age']} | val: €{r['market_value_eur']/1e6:.1f}M | mins: {r['minutes']} | sim: {r['similarity_score']:.1f}")
    assert r['age'] <= 26, f"Age filter failed: {r['age']}"
    assert r['market_value_eur'] <= 40000000.0, f"Budget filter failed: {r['market_value_eur']}"
    assert r['minutes'] >= 1000, f"Minutes filter failed: {r['minutes']}"
    assert r['position_compatibility'] == "Same Position", f"Position filter failed: {r['position_compatibility']}"

print("\nAll scouting checks PASSED successfully!")
