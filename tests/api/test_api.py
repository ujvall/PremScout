"""Integration test suite for the PremAI FastAPI backend."""

import pytest
from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)


# -----------------------------------------------------------------------------
# 1. Root & Health Check Tests
# -----------------------------------------------------------------------------

def test_root_endpoint():
    """Verify that root endpoint returns discovery metadata."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == "PremAI Football Intelligence API"
    assert data["status"] == "running"
    assert "version" in data
    assert "docs_url" in data


def test_health_endpoint():
    """Verify that health check reports all 3 ML models as available."""
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "models" in data
    assert data["models"]["transfer"] is True
    assert data["models"]["match"] is True
    assert data["models"]["scouting"] is True


# -----------------------------------------------------------------------------
# 2. Transfer Value Predictor API Tests
# -----------------------------------------------------------------------------

def test_transfer_predict_valid_inference():
    """Verify real ML inference on transfer value prediction."""
    payload = {
        "player_name": "Declan Rice",
        "age": 25.0,
        "height_cm": 188.0,
        "contract_years_left": 4.0,
        "club_market_value_eur": 1040000000.0,
        "minutes_played": 3200.0,
        "appearances": 38.0,
        "goals": 7.0,
        "assists": 8.0,
        "yellow_cards": 5.0,
        "red_cards": 0.0,
        "goals_per90": 0.20,
        "assists_per90": 0.23,
        "shots_per90": 1.35,
        "npxg_per90": 0.15,
        "prev_market_value_eur": 90000000.0,
        "primary_position": "Midfield",
        "sub_position": "Defensive Midfield",
        "preferred_foot": "Right",
    }
    response = client.post("/api/transfer/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["player_name"] == "Declan Rice"
    assert isinstance(data["predicted_market_value_eur"], float)
    assert data["predicted_market_value_eur"] > 10_000_000.0
    assert "€" in data["predicted_market_value_formatted"]
    assert data["model_type"] == "XGBoost"
    assert data["target_transformation"] == "log1p"


def test_transfer_predict_invalid_age():
    """Verify 422 validation failure when player age is out of bounds."""
    payload = {
        "player_name": "Too Young",
        "age": 10.0,  # Below ge=15 limit
    }
    response = client.post("/api/transfer/predict", json=payload)
    assert response.status_code == 422


def test_transfer_model_info():
    """Verify metadata retrieval for transfer value predictor."""
    response = client.get("/api/transfer/model-info")
    assert response.status_code == 200
    data = response.json()
    assert data["model_type"] == "XGBoost"
    assert data["feature_count"] == 18
    assert "test_metrics" in data
    assert "mae" in data["test_metrics"]


# -----------------------------------------------------------------------------
# 3. Match Outcome Predictor API Tests
# -----------------------------------------------------------------------------

def test_match_predict_valid_inference():
    """Verify real ML inference on match outcome forecasting."""
    payload = {
        "home_team": "Arsenal",
        "away_team": "Chelsea",
        "home_rest_days": 7.0,
        "away_rest_days": 4.0,
        "home_rolling_points_5": 12.0,
        "away_rolling_points_5": 7.0,
        "home_rolling_gf_5": 11.0,
        "away_rolling_gf_5": 6.0,
        "home_rolling_ga_5": 3.0,
        "away_rolling_ga_5": 8.0,
        "home_rolling_shots_5": 16.4,
        "away_rolling_shots_5": 12.2,
        "home_rolling_sot_5": 6.2,
        "away_rolling_sot_5": 4.0,
        "home_rolling_sot_conceded_5": 2.6,
        "away_rolling_sot_conceded_5": 5.4,
        "home_rolling_corners_5": 7.2,
        "away_rolling_corners_5": 4.6,
        "home_rolling_win_pct_5": 0.80,
        "away_rolling_win_pct_5": 0.40,
        "home_league_rank": 2,
        "away_league_rank": 9,
        "home_ppg": 2.25,
        "away_ppg": 1.45,
        "h2h_matches_prior": 12,
        "h2h_home_win_pct": 0.58,
    }
    response = client.post("/api/match/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["home_team"] == "Arsenal"
    assert data["away_team"] == "Chelsea"
    assert data["predicted_outcome"] in ["H", "D", "A"]
    assert data["predicted_outcome_label"] in ["Home Win", "Draw", "Away Win"]

    probs = data["probabilities"]
    assert 0.0 <= probs["home_win"] <= 1.0
    assert 0.0 <= probs["draw"] <= 1.0
    assert 0.0 <= probs["away_win"] <= 1.0
    prob_sum = probs["home_win"] + probs["draw"] + probs["away_win"]
    assert abs(prob_sum - 1.0) < 0.02


def test_match_predict_invalid_rank():
    """Verify 422 error on invalid league rank parameter."""
    payload = {
        "home_rolling_points_5": 10.0,
        "away_rolling_points_5": 8.0,
        "home_rolling_gf_5": 5.0,
        "away_rolling_gf_5": 4.0,
        "home_rolling_ga_5": 3.0,
        "away_rolling_ga_5": 5.0,
        "home_rolling_shots_5": 12.0,
        "away_rolling_shots_5": 10.0,
        "home_rolling_sot_5": 4.0,
        "away_rolling_sot_5": 3.0,
        "home_rolling_sot_conceded_5": 3.0,
        "away_rolling_sot_conceded_5": 4.0,
        "home_rolling_corners_5": 5.0,
        "away_rolling_corners_5": 4.0,
        "home_rolling_win_pct_5": 0.6,
        "away_rolling_win_pct_5": 0.4,
        "home_league_rank": 25,  # Out of range (max 20)
        "away_league_rank": 5,
        "home_ppg": 1.8,
        "away_ppg": 1.5,
    }
    response = client.post("/api/match/predict", json=payload)
    assert response.status_code == 422


def test_match_model_info():
    """Verify metadata retrieval for match outcome predictor."""
    response = client.get("/api/match/model-info")
    assert response.status_code == 200
    data = response.json()
    assert data["model_type"] == "XGBoost"
    assert data["feature_count"] == 25
    assert "test_metrics" in data
    assert "accuracy" in data["test_metrics"]


# -----------------------------------------------------------------------------
# 4. Player Scouting & Replacement API Tests
# -----------------------------------------------------------------------------

def test_scouting_similar_players():
    """Verify vector similarity retrieval for an existing player."""
    response = client.get("/api/scouting/similar/Declan%20Rice?top_n=5")
    assert response.status_code == 200
    candidates = response.json()
    assert len(candidates) == 5

    # Target player must NOT be in candidates
    cand_names = [c["candidate_player"].lower() for c in candidates]
    assert "declan rice" not in cand_names

    # Verify score range and attributes
    for c in candidates:
        assert 0.0 <= c["similarity_score"] <= 100.0
        assert "candidate_club" in c
        assert "key_stats" in c
        assert "pass_completion_pct" in c["key_stats"]


def test_scouting_replacements_filtered():
    """Verify replacement search with budget and age filters."""
    payload = {
        "player_name": "Declan Rice",
        "top_n": 5,
        "same_position_only": True,
        "max_market_value": 50000000.0,
        "max_age": 28,
        "min_minutes": 900,
    }
    response = client.post("/api/scouting/replacements", json=payload)
    assert response.status_code == 200
    candidates = response.json()
    assert len(candidates) > 0

    for c in candidates:
        assert c["position_compatibility"] == "Same Position"
        if c["market_value_eur"] is not None:
            assert c["market_value_eur"] <= 50_000_000.0
        if c["age"] is not None:
            assert c["age"] <= 28
        assert c["minutes"] >= 900


def test_scouting_unknown_player_returns_404():
    """Verify 404 response when querying a nonexistent player."""
    response = client.get("/api/scouting/similar/TotallyFakePlayerXYZ123")
    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()


def test_scouting_invalid_parameters():
    """Verify 422 error on invalid parameter ranges."""
    payload = {
        "player_name": "Declan Rice",
        "max_age": 10,  # Below ge=15
    }
    response = client.post("/api/scouting/replacements", json=payload)
    assert response.status_code == 422


def test_scouting_player_search():
    """Verify lightweight player search catalog endpoint."""
    response = client.get("/api/scouting/players?search=rice&limit=10")
    assert response.status_code == 200
    players = response.json()
    assert len(players) > 0
    names = [p["player_name"].lower() for p in players]
    assert any("rice" in n for n in names)
    assert "position" in players[0]
    assert "club" in players[0]


def test_scouting_model_info():
    """Verify metadata retrieval for player scouting module."""
    response = client.get("/api/scouting/model-info")
    assert response.status_code == 200
    data = response.json()
    assert data["feature_count"] == 16
    assert "evaluation_metrics" in data
    assert "top_5_same_position_rate" in data["evaluation_metrics"]


# -----------------------------------------------------------------------------
# 5. Current Data Modernization Integration Tests
# -----------------------------------------------------------------------------

def test_data_status_endpoint():
    """Verify data-status endpoint reports current 2026-09-20 timestamps."""
    response = client.get("/api/data-status")
    assert response.status_code == 200
    data = response.json()
    assert data["data_as_of"] == "2026-09-20"
    assert "players_as_of" in data
    assert "sources" in data
    assert len(data["sources"]) >= 3


def test_current_player_search_cherki():
    """Verify Rayan Cherki is discoverable in current player search with Man City."""
    response = client.get("/api/scouting/players?search=cherki&limit=5")
    assert response.status_code == 200
    players = response.json()
    assert len(players) > 0
    cherki = players[0]
    assert "Cherki" in cherki["player_name"]
    assert cherki["club"] == "Manchester City"
    assert cherki["age"] == 23
    assert cherki["season"] == "2026-27"
    assert cherki["is_current_big5"] == True
    assert cherki["eligibility_status"] == "BIG5_ACTIVE"
    assert cherki["data_as_of"] == "2026-09-20"


def test_current_player_search_haaland():
    """Verify Erling Haaland returns current age 26 and season 2026-27 (not stale age 23)."""
    response = client.get("/api/scouting/players?search=haaland&limit=5")
    assert response.status_code == 200
    players = response.json()
    assert len(players) > 0
    haaland = players[0]
    assert haaland["player_name"] == "Erling Haaland"
    assert haaland["club"] == "Manchester City"
    assert haaland["age"] == 26
    assert haaland["season"] == "2026-27"
    assert haaland["is_current_big5"] == True
    assert haaland["eligibility_status"] == "BIG5_ACTIVE"
    assert haaland["data_as_of"] == "2026-09-20"


def test_current_player_search_martinez_accents():
    """Verify Emiliano Martínez handles accent-insensitive search and returns Aston Villa."""
    response = client.get("/api/scouting/players?search=martinez&limit=10")
    assert response.status_code == 200
    players = response.json()
    matched_names = [p["player_name"] for p in players]
    assert any("Mart" in n for n in matched_names)
    emi = next((p for p in players if "Aston Villa" in p["club"]), None)
    assert emi is not None
    assert emi["age"] == 34
    assert emi["season"] == "2026-27"


def test_cherki_scouting_similarity_v2():
    """Verify Rayan Cherki's 2025/26 vector generates valid recommendations through V2 model."""
    response = client.get("/api/scouting/similar/Rayan Cherki?top_n=5")
    assert response.status_code == 200
    data = response.json()
    assert len(data) > 0
    first = data[0]
    assert first["target_player"] == "Rayan Cherki"
    assert first["target_team"] == "Manchester City"
    assert first["target_season"] == "2025-26"
    assert first["data_as_of"] == "2026-09-20"
    assert 0.0 <= first["similarity_score"] <= 100.0
    # Big-5 eligibility assertions
    assert first["is_current_big5"] == True
    assert first["is_free_agent"] == False
    assert first["current_league"] in ["Premier League", "La Liga", "Bundesliga", "Serie A", "Ligue 1"]
    assert "performance_club" in first
    assert "performance_league" in first


def test_transfer_presets_dynamic():
    """Verify transfer presets return current stars with populated 2025/26 features."""
    response = client.get("/api/transfer/presets")
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["data_as_of"] == "2026-09-20"
    preset_names = [p["name"] for p in data["presets"]]
    assert any("Cherki" in n for n in preset_names)
    assert any("Haaland" in n for n in preset_names)
    assert any("Saka" in n for n in preset_names)
    assert any("Rice" in n for n in preset_names)


def test_transfer_player_features_lookup():
    """Verify current transfer feature vector retrieval for Erling Haaland."""
    response = client.get("/api/transfer/player-features/Erling Haaland")
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["data_as_of"] == "2026-09-20"
    f = data["features"]
    assert f["age"] == 26
    assert f["primary_position"] == "Attack"
    assert f["sub_position"] == "Centre-Forward"
    assert f["prev_market_value_eur"] == 180000000.0


def test_match_current_teams():
    """Verify match module exposes 20 current 2026/27 Premier League clubs."""
    response = client.get("/api/match/current-teams")
    assert response.status_code == 200
    data = response.json()
    assert data["season"] == "2026-27"
    assert data["data_as_of"] == "2026-09-20"
    assert len(data["teams"]) == 20
    team_names = [t["team_name"] for t in data["teams"]]
    assert "Arsenal" in team_names
    assert "Manchester City" in team_names
    assert "Liverpool" in team_names


def test_match_current_fixture_features():
    """Verify pre-match feature calculation for current fixture Arsenal vs Chelsea."""
    response = client.get("/api/match/current-fixture-features?home_team=Arsenal&away_team=Chelsea")
    assert response.status_code == 200
    data = response.json()
    assert data["season"] == "2026-27"
    assert data["data_as_of"] == "2026-09-20"
    f = data["features"]
    assert f["home_team"] == "Arsenal"
    assert f["away_team"] == "Chelsea"
    assert "home_elo" in f
    assert "away_elo" in f
    assert "elo_diff" in f
    assert "rank_diff" in f


def test_scouting_eligibility_big5_and_firmino_exclusion():
    """Verify that player search dynamically excludes non-Big 5 players (such as Roberto Firmino in Qatar),
    surfaces only BIG5_ACTIVE candidates, and replacement candidates exclude non-Big 5 players by default."""
    # 1. Search player test: Roberto Firmino (Qatar Stars League / NON_BIG5) must NOT appear in search results
    resp_firmino = client.get("/api/scouting/players?search=Firmino&limit=5")
    assert resp_firmino.status_code == 200
    firmino_results = resp_firmino.json()
    assert len(firmino_results) == 0, "Non-Big 5 players like Firmino must be dynamically excluded from player search"

    # 2. Eligible Big 5 player search: Erling Haaland must appear and be BIG5_ACTIVE
    resp_haaland = client.get("/api/scouting/players?search=Haaland&limit=5")
    assert resp_haaland.status_code == 200
    haaland_results = resp_haaland.json()
    assert len(haaland_results) > 0
    haaland = next((p for p in haaland_results if "Erling Haaland" in p["player_name"]), None)
    assert haaland is not None
    assert haaland["eligibility_status"] == "BIG5_ACTIVE"
    assert haaland["is_current_big5"] is True
    assert haaland["current_league"] == "Premier League"

    # 3. Replacements query: Firmino must NEVER appear in candidate results
    resp_repl = client.post(
        "/api/scouting/replacements",
        json={
            "player_name": "Erling Haaland",
            "top_n": 50,
            "same_position_only": False,
            "include_free_agents": False,
        },
    )
    assert resp_repl.status_code == 200
    candidates = resp_repl.json()
    assert len(candidates) > 0
    for c in candidates:
        assert c["is_current_big5"] is True
        assert c["is_free_agent"] is False
        assert c["current_league"] in ["Premier League", "La Liga", "Bundesliga", "Serie A", "Ligue 1"]
        assert "firmino" not in c["candidate_player"].lower()


def test_scouting_eligibility_free_agent_toggle():
    """Verify that Free Agents are excluded by default and can be included via toggle."""
    # Default: include_free_agents=False
    resp_off = client.post(
        "/api/scouting/replacements",
        json={
            "player_name": "Martin Dúbravka",
            "top_n": 50,
            "same_position_only": False,
            "include_free_agents": False,
        },
    )
    assert resp_off.status_code == 200
    cands_off = resp_off.json()
    for c in cands_off:
        assert c["is_free_agent"] is False

    # Toggle ON: include_free_agents=True
    resp_on = client.post(
        "/api/scouting/replacements",
        json={
            "player_name": "Martin Dúbravka",
            "top_n": 50,
            "same_position_only": False,
            "include_free_agents": True,
        },
    )
    assert resp_on.status_code == 200
    cands_on = resp_on.json()
    fa_cands = [c for c in cands_on if c.get("is_free_agent")]
    assert len(fa_cands) >= 1
    fa = fa_cands[0]
    assert fa["current_league"] == "Free Agent"
    assert fa["is_free_agent"] is True


def test_scouting_market_value_and_recent_form():
    """Verify that candidate market values are mapped accurately and recent form is verified 2025/26."""
    resp = client.get("/api/scouting/similar/Declan%20Rice?top_n=10&include_free_agents=false")
    assert resp.status_code == 200
    candidates = resp.json()
    assert len(candidates) > 0

    has_recent_form = False
    has_market_value = False

    for c in candidates:
        # Check market value mapping
        if c.get("market_value_eur") is not None:
            assert isinstance(c["market_value_eur"], (int, float))
            assert c["market_value_eur"] > 0
            has_market_value = True

        # Check recent form mapping
        rf = c.get("recent_form")
        if rf is not None:
            has_recent_form = True
            assert rf["season"] == "2025/26"
            assert "Match Form" in rf["label"]
            assert isinstance(rf["matches"], list)
            for m in rf["matches"]:
                assert m["result"] in ["W", "D", "L"]
                assert "date" in m
                assert "minutes" in m

    assert has_market_value, "At least one candidate must have a verified market value"
    assert has_recent_form, "At least one candidate must have verified 2025/26 recent match form"



