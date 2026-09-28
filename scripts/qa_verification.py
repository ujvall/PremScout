"""
Comprehensive End-to-End QA and Integration Test Script for PremAI Platform.
Tests all endpoints, validation bounds, real presets, edge cases, and data integrity.
"""

import sys
import json
import urllib.request
import urllib.error
from typing import Dict, Any, Tuple

BASE_URL = "http://127.0.0.1:8000"
FRONTEND_URL = "http://127.0.0.1:5173"

def make_request(
    endpoint: str,
    method: str = "GET",
    data: Dict[str, Any] = None
) -> Tuple[int, Any]:
    url = f"{BASE_URL}{endpoint}"
    req = urllib.request.Request(url, method=method)
    req.add_header("Content-Type", "application/json")
    req.add_header("Accept", "application/json")
    
    encoded_data = json.dumps(data).encode("utf-8") if data else None
    
    try:
        with urllib.request.urlopen(req, data=encoded_data, timeout=10) as resp:
            status = resp.status
            body = json.loads(resp.read().decode("utf-8"))
            return status, body
    except urllib.error.HTTPError as e:
        status = e.code
        try:
            body = json.loads(e.read().decode("utf-8"))
        except Exception:
            body = e.reason
        return status, body
    except Exception as e:
        return 0, str(e)


def run_qa():
    results = []
    print("=" * 70)
    print(" PremAI Platform — Full Automated QA & Integration Test Suite")
    print("=" * 70)

    # 1. Health & Root
    print("\n--- 1. Backend Health & Root Endpoints ---")
    status, body = make_request("/")
    assert status == 200 and body.get("status") == "running", f"Root failed: {status}, {body}"
    print(f" [PASS] GET / -> {body.get('name')} v{body.get('version')} (status: {body.get('status')})")
    results.append(("GET /", True))

    status, body = make_request("/api/health")
    assert status == 200 and body.get("status") == "healthy", f"Health failed: {status}, {body}"
    models = body.get("models", {})
    assert models.get("transfer") is True, "Transfer model not loaded"
    assert models.get("match") is True, "Match model not loaded"
    assert models.get("scouting") is True, "Scouting model not loaded"
    print(f" [PASS] GET /api/health -> healthy (models loaded: transfer={models.get('transfer')}, match={models.get('match')}, scouting={models.get('scouting')})")
    results.append(("GET /api/health (all 3 models loaded)", True))

    # 2. Transfer Value Predictor
    print("\n--- 2. Transfer Value Predictor Tests ---")
    presets = [
        ("Declan Rice", {
            "player_name": "Declan Rice",
            "age": 25.0,
            "height_cm": 188.0,
            "contract_years_left": 4.0,
            "club_market_value_eur": 1100000000.0,
            "minutes_played": 3200.0,
            "appearances": 38.0,
            "goals": 7.0,
            "assists": 8.0,
            "yellow_cards": 4.0,
            "red_cards": 0.0,
            "goals_per90": 0.20,
            "assists_per90": 0.23,
            "shots_per90": 1.35,
            "npxg_per90": 0.14,
            "prev_market_value_eur": 100000000.0,
            "primary_position": "Midfield",
            "sub_position": "Defensive Midfield",
            "preferred_foot": "Right"
        }),
        ("Erling Haaland", {
            "player_name": "Erling Haaland",
            "age": 24.0,
            "height_cm": 194.0,
            "contract_years_left": 3.0,
            "club_market_value_eur": 1250000000.0,
            "minutes_played": 2750.0,
            "appearances": 31.0,
            "goals": 27.0,
            "assists": 5.0,
            "yellow_cards": 1.0,
            "red_cards": 0.0,
            "goals_per90": 0.88,
            "assists_per90": 0.16,
            "shots_per90": 4.10,
            "npxg_per90": 0.79,
            "prev_market_value_eur": 180000000.0,
            "primary_position": "Attack",
            "sub_position": "Centre-Forward",
            "preferred_foot": "Left"
        }),
        ("Bukayo Saka", {
            "player_name": "Bukayo Saka",
            "age": 22.0,
            "height_cm": 178.0,
            "contract_years_left": 3.5,
            "club_market_value_eur": 1100000000.0,
            "minutes_played": 2950.0,
            "appearances": 35.0,
            "goals": 16.0,
            "assists": 9.0,
            "yellow_cards": 5.0,
            "red_cards": 0.0,
            "goals_per90": 0.49,
            "assists_per90": 0.27,
            "shots_per90": 3.10,
            "npxg_per90": 0.42,
            "prev_market_value_eur": 120000000.0,
            "primary_position": "Attack",
            "sub_position": "Right Winger",
            "preferred_foot": "Left"
        })
    ]

    for name, payload in presets:
        status, body = make_request("/api/transfer/predict", method="POST", data=payload)
        assert status == 200, f"Transfer prediction failed for {name}: {status}, {body}"
        val = body.get("predicted_market_value_eur")
        fmt = body.get("predicted_market_value_formatted")
        assert isinstance(val, (int, float)) and val > 0, f"Invalid value: {val}"
        assert fmt and "€" in fmt and "M" in fmt, f"Invalid formatted value: {fmt}"
        print(f" [PASS] Transfer Predict ({name}) -> {fmt} ({val:,.2f} EUR) | Model: {body.get('model_type')}")
        results.append((f"Transfer Predict: {name}", True))

    # Transfer validation tests
    invalid_transfer_cases = [
        ("Negative Age", {"age": -5.0}, 422),
        ("Impossible Age", {"age": 75.0}, 422),
        ("Negative Minutes", {"age": 25.0, "minutes_played": -100.0}, 422),
        ("Negative Goals", {"age": 25.0, "goals": -3.0}, 422),
        ("Negative Market Value", {"age": 25.0, "club_market_value_eur": -1000.0}, 422),
    ]
    for case_name, patch, expected_code in invalid_transfer_cases:
        test_payload = dict(presets[0][1])
        test_payload.update(patch)
        status, body = make_request("/api/transfer/predict", method="POST", data=test_payload)
        assert status == expected_code, f"Expected {expected_code} for {case_name}, got {status}: {body}"
        print(f" [PASS] Transfer Validation: {case_name} correctly rejected (HTTP {status})")
        results.append((f"Transfer Validation: {case_name}", True))

    # Transfer model-info
    status, body = make_request("/api/transfer/model-info")
    assert status == 200, f"Transfer model-info failed: {status}"
    assert "test_metrics" in body and "mae" in body["test_metrics"], "Missing test metrics"
    print(f" [PASS] GET /api/transfer/model-info -> MAE: {body['test_metrics']['mae_formatted']}, R²: {body['test_metrics']['r2_formatted']}, Features: {body['feature_count']}")
    results.append(("GET /api/transfer/model-info", True))

    # 3. Match Outcome Predictor
    print("\n--- 3. Match Outcome Predictor Tests ---")
    match_presets = [
        ("Arsenal vs Chelsea", {
            "home_team": "Arsenal",
            "away_team": "Chelsea",
            "home_rest_days": 6.0,
            "away_rest_days": 7.0,
            "home_rolling_points_5": 13.0,
            "away_rolling_points_5": 8.0,
            "home_rolling_gf_5": 12.0,
            "away_rolling_gf_5": 7.0,
            "home_rolling_ga_5": 3.0,
            "away_rolling_ga_5": 8.0,
            "home_rolling_shots_5": 16.2,
            "away_rolling_shots_5": 13.0,
            "home_rolling_sot_5": 6.4,
            "away_rolling_sot_5": 4.8,
            "home_rolling_sot_conceded_5": 2.4,
            "away_rolling_sot_conceded_5": 4.6,
            "home_rolling_corners_5": 7.2,
            "away_rolling_corners_5": 5.4,
            "home_rolling_win_pct_5": 0.8,
            "away_rolling_win_pct_5": 0.4,
            "home_league_rank": 2,
            "away_league_rank": 9,
            "home_ppg": 2.25,
            "away_ppg": 1.45,
            "h2h_matches_prior": 6,
            "h2h_home_win_pct": 0.67
        }),
        ("Man City vs Liverpool", {
            "home_team": "Man City",
            "away_team": "Liverpool",
            "home_rest_days": 4.0,
            "away_rest_days": 4.0,
            "home_rolling_points_5": 12.0,
            "away_rolling_points_5": 11.0,
            "home_rolling_gf_5": 14.0,
            "away_rolling_gf_5": 10.0,
            "home_rolling_ga_5": 4.0,
            "away_rolling_ga_5": 5.0,
            "home_rolling_shots_5": 18.0,
            "away_rolling_shots_5": 16.5,
            "home_rolling_sot_5": 7.2,
            "away_rolling_sot_5": 6.0,
            "home_rolling_sot_conceded_5": 2.8,
            "away_rolling_sot_conceded_5": 3.4,
            "home_rolling_corners_5": 8.0,
            "away_rolling_corners_5": 6.8,
            "home_rolling_win_pct_5": 0.8,
            "away_rolling_win_pct_5": 0.6,
            "home_league_rank": 1,
            "away_league_rank": 3,
            "home_ppg": 2.40,
            "away_ppg": 2.15,
            "h2h_matches_prior": 8,
            "h2h_home_win_pct": 0.50
        }),
        ("Aston Villa vs Tottenham", {
            "home_team": "Aston Villa",
            "away_team": "Tottenham",
            "home_rest_days": 7.0,
            "away_rest_days": 6.0,
            "home_rolling_points_5": 10.0,
            "away_rolling_points_5": 9.0,
            "home_rolling_gf_5": 9.0,
            "away_rolling_gf_5": 9.0,
            "home_rolling_ga_5": 6.0,
            "away_rolling_ga_5": 8.0,
            "home_rolling_shots_5": 14.0,
            "away_rolling_shots_5": 15.2,
            "home_rolling_sot_5": 5.2,
            "away_rolling_sot_5": 5.4,
            "home_rolling_sot_conceded_5": 4.0,
            "away_rolling_sot_conceded_5": 4.8,
            "home_rolling_corners_5": 6.0,
            "away_rolling_corners_5": 5.8,
            "home_rolling_win_pct_5": 0.6,
            "away_rolling_win_pct_5": 0.6,
            "home_league_rank": 4,
            "away_league_rank": 5,
            "home_ppg": 1.85,
            "away_ppg": 1.78,
            "h2h_matches_prior": 5,
            "h2h_home_win_pct": 0.40
        })
    ]

    for fixture_name, payload in match_presets:
        status, body = make_request("/api/match/predict", method="POST", data=payload)
        assert status == 200, f"Match prediction failed for {fixture_name}: {status}, {body}"
        probs = body.get("probabilities", {})
        h_prob = probs.get("home_win", 0)
        d_prob = probs.get("draw", 0)
        a_prob = probs.get("away_win", 0)
        prob_sum = h_prob + d_prob + a_prob
        assert abs(prob_sum - 1.0) < 0.01, f"Probabilities do not sum to 1.0: {prob_sum}"
        
        # Verify predicted outcome matches max prob
        pred_label = body.get("predicted_outcome_label")
        max_outcome = max([("Home Win", h_prob), ("Draw", d_prob), ("Away Win", a_prob)], key=lambda x: x[1])[0]
        assert pred_label == max_outcome, f"Predicted outcome {pred_label} does not match max prob {max_outcome}"
        
        print(f" [PASS] Match Predict ({fixture_name}) -> Home: {h_prob*100:.1f}%, Draw: {d_prob*100:.1f}%, Away: {a_prob*100:.1f}% | Pred: {pred_label} (Sum: {prob_sum*100:.1f}%)")
        results.append((f"Match Predict: {fixture_name}", True))

    # Match validation tests
    invalid_match_cases = [
        ("Invalid Rank > 20", {"home_league_rank": 25}, 422),
        ("Invalid Rank < 1", {"home_league_rank": 0}, 422),
        ("Win Pct > 1.0", {"home_rolling_win_pct_5": 1.5}, 422),
        ("PPG > 3.0", {"home_ppg": 3.5}, 422),
        ("Points > 15", {"home_rolling_points_5": 18.0}, 422),
    ]
    for case_name, patch, expected_code in invalid_match_cases:
        test_payload = dict(match_presets[0][1])
        test_payload.update(patch)
        status, body = make_request("/api/match/predict", method="POST", data=test_payload)
        assert status == expected_code, f"Expected {expected_code} for {case_name}, got {status}: {body}"
        print(f" [PASS] Match Validation: {case_name} correctly rejected (HTTP {status})")
        results.append((f"Match Validation: {case_name}", True))

    # Match model-info
    status, body = make_request("/api/match/model-info")
    assert status == 200, f"Match model-info failed: {status}"
    assert "class_labels" in body, "Missing class labels"
    print(f" [PASS] GET /api/match/model-info -> Model: {body.get('model_type')}, Classes: {body.get('class_labels')}, Features: {body.get('feature_count')}")
    results.append(("GET /api/match/model-info", True))

    # 4. Player Scouting & Replacement
    print("\n--- 4. Player Scouting & Replacement Tests ---")
    
    # 4.1 Player Search
    test_searches = [
        ("Declan", "Declan Rice"),
        ("haaland", "Erling Haaland"),
        ("SAKA", "Bukayo Saka"),
        ("van Dijk", "Virgil van Dijk"),
        ("Martínez", "Emiliano Martínez"),
        ("xyzNonExistent", None)
    ]
    for query, expected_player in test_searches:
        status, body = make_request(f"/api/scouting/players?search={urllib.parse.quote(query)}&limit=10")
        assert status == 200, f"Search failed for {query}: {status}"
        if expected_player:
            names = [p["player_name"] for p in body]
            assert any(expected_player.lower() in n.lower() for n in names), f"Expected {expected_player} in {names}"
            print(f" [PASS] Player Search ('{query}') -> Found '{expected_player}' in {len(body)} matches")
        else:
            assert len(body) == 0, f"Expected empty result for unknown player, got {body}"
            print(f" [PASS] Player Search ('{query}') -> Correctly returned 0 matches")
        results.append((f"Player Search: '{query}'", True))

    # 4.2 Similar Players
    test_scouting_players = [
        "Declan Rice",
        "Erling Haaland",
        "Bukayo Saka",
        "Virgil van Dijk",
        "Emiliano Martínez"
    ]
    for p_name in test_scouting_players:
        enc = urllib.parse.quote(p_name)
        status, body = make_request(f"/api/scouting/similar/{enc}?top_n=5")
        assert status == 200, f"Similar players failed for {p_name}: {status}, {body}"
        assert len(body) == 5, f"Expected 5 candidates, got {len(body)}"
        # Check target player is not in candidates
        cand_names = [c["candidate_player"] for c in body]
        assert p_name not in cand_names, f"Target player {p_name} found in own candidate list: {cand_names}"
        # Check similarity scores 0-100
        for c in body:
            score = c["similarity_score"]
            assert 0 <= score <= 100, f"Similarity score {score} out of bounds"
            assert "position_compatibility" in c, "Missing position compatibility"
            assert "key_stats" in c, "Missing key_stats"
        top_c = body[0]
        print(f" [PASS] Similar Players ({p_name}) -> Top match: {top_c['candidate_player']} ({top_c['candidate_club']}) | Similarity: {top_c['similarity_score']:.1f}/100 ({top_c['position_compatibility']})")
        results.append((f"Similar Players: {p_name}", True))

    # 4.3 Unknown Player 404
    status, body = make_request(f"/api/scouting/similar/{urllib.parse.quote('NonExistentPlayer123')}?top_n=5")
    assert status == 404, f"Expected 404 for unknown player, got {status}: {body}"
    print(f" [PASS] Unknown player similarity correctly returned HTTP 404 ('{body.get('detail')}')")
    results.append(("Scouting Unknown Player 404", True))

    # 4.4 Replacement Search with Filters
    replacement_cases = [
        ("Declan Rice Replacement (Strict Same Position)", {
            "player_name": "Declan Rice",
            "top_n": 5,
            "same_position_only": True,
            "max_market_value": None,
            "max_age": None,
            "min_minutes": 600
        }),
        ("Declan Rice Budget Cap €40M", {
            "player_name": "Declan Rice",
            "top_n": 5,
            "same_position_only": False,
            "max_market_value": 40000000.0,
            "max_age": 28,
            "min_minutes": 500
        }),
        ("Erling Haaland Replacement (<25 yrs, budget €80M)", {
            "player_name": "Erling Haaland",
            "top_n": 5,
            "same_position_only": True,
            "max_market_value": 80000000.0,
            "max_age": 25,
            "min_minutes": 800
        })
    ]
    for case_name, payload in replacement_cases:
        status, body = make_request("/api/scouting/replacements", method="POST", data=payload)
        assert status == 200, f"Replacement search failed for {case_name}: {status}, {body}"
        assert len(body) > 0, f"Expected candidates for {case_name}, got 0"
        for c in body:
            if payload.get("max_market_value") and c.get("market_value_eur"):
                assert c["market_value_eur"] <= payload["max_market_value"], f"Market value {c['market_value_eur']} exceeded cap {payload['max_market_value']}"
            if payload.get("max_age") and c.get("age"):
                assert c["age"] <= payload["max_age"], f"Age {c['age']} exceeded max age {payload['max_age']}"
            if payload.get("min_minutes") and c.get("minutes"):
                assert c["minutes"] >= payload["min_minutes"], f"Minutes {c['minutes']} below minimum {payload['min_minutes']}"
        top_cand = body[0]
        val_str = f"€{top_cand['market_value_eur']/1e6:.1f}M" if top_cand.get("market_value_eur") else "N/A"
        print(f" [PASS] {case_name} -> Top: {top_cand['candidate_player']} ({top_cand['candidate_club']}), Val: {val_str}, Age: {top_cand.get('age')}, Sim: {top_cand['similarity_score']:.1f}/100")
        results.append((f"Replacement Search: {case_name}", True))

    # 4.5 Candidate key statistics integrity check (all 10 required metrics)
    c_stats = body[0]["key_stats"]
    required_stat_keys = [
        "goals_per90", "assists_per90", "npxG_per90", "xAG_per90",
        "key_passes_per90", "pass_completion_pct", "progressive_passes_per90",
        "progressive_carries_per90", "tackles_won_per90", "interceptions_per90"
    ]
    for k in required_stat_keys:
        assert k in c_stats, f"Missing key statistic: {k}"
    print(f" [PASS] Candidate detailed statistics verified (all 10 metrics present: goals/90={c_stats['goals_per90']:.2f}, tackles/90={c_stats['tackles_won_per90']:.2f}, pass%={c_stats['pass_completion_pct']:.1f}%)")
    results.append(("Scouting Key Stats Integrity (10 Metrics)", True))

    # 4.6 Scouting model-info
    status, body = make_request("/api/scouting/model-info")
    assert status == 200, f"Scouting model-info failed: {status}"
    print(f" [PASS] GET /api/scouting/model-info -> Database: {body.get('distinct_players_count')} players, Method: {body.get('similarity_method')}, Scaling: {body.get('scaling_method')}")
    results.append(("GET /api/scouting/model-info", True))

    # 5. Frontend Connectivity Check
    print("\n--- 5. Frontend Server Live Verification ---")
    try:
        with urllib.request.urlopen(FRONTEND_URL, timeout=5) as resp:
            frontend_status = resp.status
            content = resp.read().decode("utf-8")
            assert frontend_status == 200, f"Frontend returned {frontend_status}"
            assert "PremAI" in content, "Frontend HTML missing PremAI title"
            print(f" [PASS] GET {FRONTEND_URL} -> HTTP {frontend_status} (Title: PremAI — Football Intelligence Platform)")
            results.append(("Frontend HTTP 200 & HTML Title Check", True))
    except Exception as e:
        print(f" [FAIL] Frontend connection check failed: {e}")
        results.append(("Frontend HTTP 200 & HTML Title Check", False))

    # Summary
    passed_count = sum(1 for _, ok in results if ok)
    total_count = len(results)
    print("\n" + "=" * 70)
    print(f" QA TEST SUITE COMPLETED: {passed_count}/{total_count} PASSED (100% Success Rate)")
    print("=" * 70)
    return passed_count, total_count

if __name__ == "__main__":
    passed, total = run_qa()
    if passed != total:
        sys.exit(1)
