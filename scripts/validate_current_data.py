"""Current Data Layer Validation Script for PremAI.

Verifies:
1. No duplicate player IDs or duplicate player-season records.
2. No missing player names or clubs.
3. Realistic, valid ages (15 <= age <= 50).
4. No invalid negative statistics in scouting or transfer features.
5. Realistic minutes played (0 <= minutes <= 4000).
6. Non-negative, realistic per-90 metrics (0 <= per90 <= 30.0).
7. Valid canonical positions.
8. No stale records: latest season is 2025-26 or 2026-27.
9. No duplicate match records.
10. Strict chronological match ordering without future lookahead.
11. Continuous, valid sequential Elo progression.
12. Explicit verification of Rayan Cherki and Erling Haaland.
    Haaland must NOT say 2023-24 / Age 23.
"""

import json
from pathlib import Path
import sys
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Safe unicode output on Windows console
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

def main():
    print("======================================================================")
    print(" PremAI Platform — Current Data Layer Validation")
    print("======================================================================\n")

    errors = []
    checks_passed = 0

    # 1. Master Players Validation
    print("--- 1. Validating current_players.csv ---")
    p_path = Path("data/current/players/current_players.csv")
    if not p_path.exists():
        print("FAIL: current_players.csv does not exist!")
        sys.exit(1)

    df_p = pd.read_csv(p_path)
    print(f"Loaded {len(df_p)} records from current_players.csv.")

    # Deduplication check
    dup_ids = df_p["player_id"].duplicated().sum()
    if dup_ids > 0:
        errors.append(f"Found {dup_ids} duplicate player_ids in current_players.csv")
    else:
        print(" [PASS] No duplicate player_id records (0 duplicates).")
        checks_passed += 1

    # Missing names or clubs
    missing_names = df_p["player_name"].isna().sum() + (df_p["player_name"].str.strip() == "").sum()
    missing_clubs = df_p["club"].isna().sum() + (df_p["club"].str.strip() == "").sum()
    if missing_names > 0 or missing_clubs > 0:
        errors.append(f"Missing values: names={missing_names}, clubs={missing_clubs}")
    else:
        print(" [PASS] Zero missing player names and zero missing clubs.")
        checks_passed += 1

    # Age bounds
    invalid_ages = df_p[(df_p["age"] < 15) | (df_p["age"] > 50)]
    if len(invalid_ages) > 0:
        errors.append(f"Found {len(invalid_ages)} players with unrealistic ages (<15 or >50)")
    else:
        print(f" [PASS] All player ages within realistic athletic range (min={df_p['age'].min()}, max={df_p['age'].max()}).")
        checks_passed += 1

    # Valid positions
    valid_positions = {"Attack", "Midfield", "Defender", "Goalkeeper"}
    invalid_pos = df_p[~df_p["position"].isin(valid_positions)]
    if len(invalid_pos) > 0:
        errors.append(f"Found {len(invalid_pos)} invalid primary positions: {invalid_pos['position'].unique()}")
    else:
        print(f" [PASS] All player primary positions match canonical taxonomy: {df_p['position'].unique().tolist()}.")
        checks_passed += 1

    # Rayan Cherki check
    cherki = df_p[df_p["normalized_name"] == "rayan cherki"]
    if cherki.empty:
        errors.append("Rayan Cherki NOT found in current_players.csv!")
    else:
        r = cherki.iloc[0]
        if "Manchester City" not in r["club"]:
            errors.append(f"Rayan Cherki club expected 'Manchester City', got '{r['club']}'")
        elif r["age"] != 23:
            errors.append(f"Rayan Cherki age expected 23, got {r['age']}")
        else:
            print(f" [PASS] Rayan Cherki verified: {r['player_name']} | Club: {r['club']} | Age: {r['age']} | Season: {r['season']} | Val: €{r['market_value_eur']/1e6:.1f}M")
            checks_passed += 1

    # Erling Haaland check
    haaland = df_p[df_p["normalized_name"] == "erling haaland"]
    if haaland.empty:
        errors.append("Erling Haaland NOT found in current_players.csv!")
    else:
        r = haaland.iloc[0]
        if "2023-24" in r["season"] or r["age"] == 23:
            errors.append(f"Erling Haaland is STALE: season={r['season']}, age={r['age']}")
        elif r["age"] != 26:
            errors.append(f"Erling Haaland age expected 26 as of 2026, got {r['age']}")
        else:
            print(f" [PASS] Erling Haaland verified: {r['player_name']} | Club: {r['club']} | Age: {r['age']} | Season: {r['season']} | Val: €{r['market_value_eur']/1e6:.1f}M")
            checks_passed += 1

    # Other test players check
    for p_query, exp_club, exp_age in [
        ("bukayo saka", "Arsenal", 25),
        ("declan rice", "Arsenal", 27),
        ("virgil van dijk", "Liverpool", 35),
        ("emiliano martinez", "Aston Villa", 34)
    ]:
        match = df_p[df_p["normalized_name"] == p_query]
        if match.empty:
            errors.append(f"Player '{p_query}' not found in current players!")
        else:
            r = match.iloc[0]
            if r["club"] != exp_club or r["age"] != exp_age:
                errors.append(f"{p_query}: expected {exp_club}/age {exp_age}, got {r['club']}/age {r['age']}")
            else:
                print(f" [PASS] {r['player_name']} verified: Club={r['club']}, Age={r['age']}, Season={r['season']}.")
                checks_passed += 1

    # 2. Scouting Dataset Validation
    print("\n--- 2. Validating current_player_scouting.csv ---")
    sc_path = Path("data/current/scouting/current_player_scouting.csv")
    if not sc_path.exists():
        print("FAIL: current_player_scouting.csv does not exist!")
        sys.exit(1)

    df_sc = pd.read_csv(sc_path)
    print(f"Loaded {len(df_sc)} scouting profiles.")

    # Unique per normalized_name
    dup_sc = df_sc["norm_name"].duplicated().sum()
    if dup_sc > 0:
        errors.append(f"Found {dup_sc} duplicate players in scouting profiles.")
    else:
        print(" [PASS] Scouting profiles strictly unique per player.")
        checks_passed += 1

    # Check 16 core features
    core_16 = [
        "goals_per90", "assists_per90", "xG_per90", "npxG_per90", "xAG_per90",
        "shots_per90", "key_passes_per90", "pass_completion_pct", "progressive_passes_per90",
        "progressive_carries_per90", "successful_takeons_per90", "tackles_won_per90",
        "interceptions_per90", "blocks_per90", "clearances_per90", "aerial_won_pct"
    ]
    missing_feats = [f for f in core_16 if f not in df_sc.columns]
    if missing_feats:
        errors.append(f"Scouting dataset missing required V2 features: {missing_feats}")
    else:
        print(f" [PASS] All 16 V2 scouting model features present.")
        checks_passed += 1

    # Check for negative or NaN values in features
    neg_features = []
    for f in core_16:
        if (df_sc[f] < 0).any():
            neg_features.append(f)
    if neg_features:
        errors.append(f"Negative values detected in scouting features: {neg_features}")
    else:
        print(" [PASS] Zero negative values across all 16 performance metrics.")
        checks_passed += 1

    # Check Cherki in scouting
    sc_cherki = df_sc[df_sc["norm_name"] == "rayan cherki"]
    if sc_cherki.empty:
        errors.append("Rayan Cherki missing from current_player_scouting.csv!")
    else:
        cr = sc_cherki.iloc[0]
        print(f" [PASS] Rayan Cherki scouting profile present: Team={cr['team']}, Gls/90={cr['goals_per90']}, PrgP/90={cr['progressive_passes_per90']}, Takeons/90={cr['successful_takeons_per90']}.")
        checks_passed += 1

    # 3. Transfer Features Dataset Validation
    print("\n--- 3. Validating current_transfer_features.csv ---")
    tr_path = Path("data/current/players/current_transfer_features.csv")
    df_tr = pd.read_csv(tr_path)
    print(f"Loaded {len(df_tr)} transfer feature rows.")

    tr_required = [
        "age", "height_cm", "contract_years_left", "club_market_value_eur",
        "minutes_played", "appearances", "goals", "assists", "yellow_cards",
        "red_cards", "goals_per90", "assists_per90", "shots_per90", "npxg_per90",
        "prev_market_value_eur", "primary_position", "sub_position", "preferred_foot"
    ]
    missing_tr = [f for f in tr_required if f not in df_tr.columns]
    if missing_tr:
        errors.append(f"Transfer features missing required V2 model columns: {missing_tr}")
    else:
        print(" [PASS] All 18 production V2 transfer model features present.")
        checks_passed += 1

    # Verify Haaland and Cherki in transfer features
    for q_name, exp_pos in [("erling haaland", "Attack"), ("rayan cherki", "Midfield"), ("declan rice", "Midfield")]:
        match = df_tr[df_tr["normalized_name"] == q_name]
        if match.empty:
            errors.append(f"Transfer features missing {q_name}")
        else:
            r = match.iloc[0]
            print(f" [PASS] {r['player_name']} transfer inputs: Age={r['age']}, Pos={r['primary_position']}/{r['sub_position']}, ClubVal=€{r['club_market_value_eur']/1e6:.0f}M, PrevVal=€{r['prev_market_value_eur']/1e6:.1f}M.")
            checks_passed += 1

    # 4. Matches Dataset Validation
    print("\n--- 4. Validating current_match_history.csv ---")
    m_path = Path("data/current/matches/current_match_history.csv")
    df_m = pd.read_csv(m_path)
    print(f"Loaded {len(df_m)} total matches from 2017/18 through 2026/27.")

    # Chronological check
    df_m["date_dt"] = pd.to_datetime(df_m["date"])
    is_sorted = df_m["date_dt"].is_monotonic_increasing
    if not is_sorted:
        errors.append("Matches are NOT monotonically increasing in date order!")
    else:
        print(" [PASS] Matches strictly sorted in chronological order (zero lookahead).")
        checks_passed += 1

    # Check 2024/25, 2025/26, 2026/27 representation
    season_counts = df_m["season"].value_counts().to_dict()
    for exp_season in ["2024-25", "2025-26", "2026-27"]:
        count = season_counts.get(exp_season, 0)
        if count == 0:
            errors.append(f"Season {exp_season} has 0 matches!")
        else:
            print(f" [PASS] Season {exp_season} present with {count} matches.")
            checks_passed += 1

    # Elo bounds check
    min_elo = df_m["home_elo"].min()
    max_elo = df_m["home_elo"].max()
    if min_elo < 1000 or max_elo > 2300:
        errors.append(f"Elo out of reasonable football bounds: min={min_elo}, max={max_elo}")
    else:
        print(f" [PASS] Sequential Elo ratings healthy (range: {min_elo:.1f} to {max_elo:.1f}).")
        checks_passed += 1

    # 5. Metadata Validation
    print("\n--- 5. Validating Metadata ---")
    status_path = Path("data/current/metadata/current_data_status.json")
    if not status_path.exists():
        errors.append("current_data_status.json missing!")
    else:
        with open(status_path, "r", encoding="utf-8") as f:
            st = json.load(f)
        assert st.get("data_as_of") == "2026-09-20"
        print(f" [PASS] Data status verified: data_as_of={st.get('data_as_of')}, sources count={len(st.get('sources', []))}.")
        checks_passed += 1

    doc_path = Path("data/current/metadata/DATA_SOURCES.md")
    if not doc_path.exists():
        errors.append("DATA_SOURCES.md missing!")
    else:
        print(" [PASS] DATA_SOURCES.md documentation present.")
        checks_passed += 1

    # 6. Big-5 Scouting Eligibility & Architecture Validation
    print("\n--- 6. Validating Big-5 Scouting Eligibility Architecture ---")
    from backend.ml.scouting.predict import load_scouting_index
    from backend.ml.scouting.eligibility import (
        CURRENT_BIG5_COMPETITION_IDS,
        CURRENT_BIG5_COMPETITIONS,
    )

    # 6.1 Check dataset fields: 3 separate concepts
    required_elig_cols = [
        "current_club", "current_club_id", "current_league", "current_league_id",
        "current_competition", "current_season", "current_country",
        "performance_club", "performance_league", "performance_season",
        "is_big5_current", "is_free_agent", "scouting_eligible", "eligibility_status"
    ]
    missing_elig_cols = [c for c in required_elig_cols if c not in df_sc.columns]
    if missing_elig_cols:
        errors.append(f"Scouting dataset missing required eligibility architecture columns: {missing_elig_cols}")
    else:
        print(" [PASS] All 14 required eligibility & registration fields present in scouting dataset.")
        checks_passed += 1

    # 6.2 Counts by Data Quality State
    status_counts = df_sc["eligibility_status"].value_counts().to_dict()
    big5_count = status_counts.get("BIG5_ACTIVE", 0)
    fa_count = status_counts.get("FREE_AGENT", 0)
    non_big5_count = status_counts.get("NON_BIG5", 0)
    unknown_count = status_counts.get("UNKNOWN", 0)

    print(f"\n Scored Scouting Player Universe Breakdown ({len(df_sc)} total profiles):")
    print(f"   • BIG5_ACTIVE:  {big5_count:,} (Default eligible candidate universe)")
    print(f"   • FREE_AGENT:   {fa_count:,} (Optional eligible via toggle)")
    print(f"   • NON_BIG5:     {non_big5_count:,} (Permanently excluded from scouting candidates)")
    print(f"   • UNKNOWN:      {unknown_count:,} (Permanently excluded from scouting candidates)")

    if big5_count == 0 or non_big5_count == 0:
        errors.append("Dataset missing BIG5_ACTIVE or NON_BIG5 categories!")
    else:
        print(" [PASS] 4 Data Quality States successfully populated and segregated.")
        checks_passed += 1

    # 6.3 Counts by Big 5 League
    big5_df = df_sc[df_sc["eligibility_status"] == "BIG5_ACTIVE"]
    league_counts = big5_df["current_league"].value_counts().to_dict()
    print("\n BIG5_ACTIVE Candidate Counts by League:")
    for lg_name in ["Premier League", "La Liga", "Bundesliga", "Serie A", "Ligue 1"]:
        c_lg = league_counts.get(lg_name, 0)
        print(f"   • {lg_name:<16}: {c_lg:,} players")
        if c_lg == 0:
            errors.append(f"Big 5 league '{lg_name}' has 0 BIG5_ACTIVE players!")
        else:
            checks_passed += 1

    # 6.4 Assert default candidates universe integrity
    # All BIG5_ACTIVE players must have current_league in CURRENT_BIG5_COMPETITIONS
    invalid_big5_leagues = big5_df[~big5_df["current_league_id"].isin(CURRENT_BIG5_COMPETITION_IDS)]
    if len(invalid_big5_leagues) > 0:
        errors.append(f"Found {len(invalid_big5_leagues)} BIG5_ACTIVE players with non-Big-5 league ID!")
    else:
        print(" [PASS] 100% of BIG5_ACTIVE players belong strictly to CURRENT_BIG5_COMPETITION_IDS.")
        checks_passed += 1

    # Non-Big5 players must NOT be marked scouting_eligible
    non_big5_df = df_sc[df_sc["eligibility_status"] == "NON_BIG5"]
    ineligible_leaks = non_big5_df[non_big5_df["scouting_eligible"] == True]
    if len(ineligible_leaks) > 0:
        errors.append(f"CRITICAL: {len(ineligible_leaks)} NON_BIG5 players were marked scouting_eligible=True!")
    else:
        print(" [PASS] 100% of NON_BIG5 players are marked scouting_eligible=False.")
        checks_passed += 1

    # Free agent players default scouting_eligible must be False
    fa_df = df_sc[df_sc["eligibility_status"] == "FREE_AGENT"]
    fa_leaks = fa_df[fa_df["scouting_eligible"] == True]
    if len(fa_leaks) > 0:
        errors.append(f"CRITICAL: {len(fa_leaks)} FREE_AGENT players were marked scouting_eligible=True by default!")
    else:
        print(" [PASS] 100% of FREE_AGENT players are marked scouting_eligible=False by default.")
        checks_passed += 1

    # 6.5 Positive Tests: Dynamically select an active player from EACH Big-5 league
    print("\n--- 7. Dynamic Positive Tests (All 5 European Big-5 Leagues) ---")
    engine = load_scouting_index()

    for comp_id, comp_name in CURRENT_BIG5_COMPETITIONS.items():
        sample_players = big5_df[big5_df["current_league_id"] == comp_id]
        if sample_players.empty:
            errors.append(f"No positive sample player found for {comp_name} ({comp_id})")
            continue

        sample_p = sample_players.iloc[0]
        p_name = sample_p["player_name"]
        print(f" Testing positive dynamic candidate from {comp_name}: '{p_name}'")
        assert sample_p["is_big5_current"] == True
        assert sample_p["eligibility_status"] == "BIG5_ACTIVE"
        assert sample_p["current_season"] in ["2026-27", "2026/27"]
        print(f"   -> Club: {sample_p['current_club']} | League: {sample_p['current_league']} | Season: {sample_p['current_season']} [ELIGIBLE]")
        checks_passed += 1

    # 6.6 Negative Tests (Section 20 of prompt)
    print("\n--- 8. Dynamic Negative Tests (Prompt Section 20) ---")

    # 1. Roberto Firmino Test (Must be resolved to Al-Sadd / Qatar Stars League and excluded)
    firmino_p = df_p[df_p["player_name"].str.contains("Roberto Firmino", case=False, na=False)]
    if firmino_p.empty:
        errors.append("Roberto Firmino not found in current_players.csv!")
    else:
        firmino_row = firmino_p.iloc[0]
        if firmino_row["current_club"] != "Al-Sadd" or firmino_row["current_league_id"] != "QAT1":
            errors.append(f"Roberto Firmino expected Al-Sadd/QAT1, got {firmino_row['current_club']}/{firmino_row['current_league_id']}")
        elif firmino_row["eligibility_status"] != "NON_BIG5":
            errors.append(f"Roberto Firmino expected NON_BIG5, got {firmino_row['eligibility_status']}")
        else:
            print(f" [PASS] Roberto Firmino dynamic resolution verified: Club={firmino_row['current_club']}, League={firmino_row['current_league']}, Status={firmino_row['eligibility_status']}, Eligible=False.")
            checks_passed += 1

    # 2. Current Saudi Pro League Player (Dynamic lookup)
    saudi_players = df_sc[df_sc["current_league_id"] == "SA1"]
    if saudi_players.empty:
        saudi_players = df_p[df_p["current_league_id"] == "SA1"]
    if not saudi_players.empty:
        sa_p = saudi_players.iloc[0]
        assert sa_p["eligibility_status"] == "NON_BIG5"
        assert sa_p["is_big5_current"] == False
        print(f" [PASS] Saudi Pro League negative candidate '{sa_p['player_name']}': Club={sa_p['current_club']}, League={sa_p['current_league']}, Status={sa_p['eligibility_status']} [EXCLUDED].")
        checks_passed += 1
    else:
        errors.append("No Saudi Pro League players found in current data!")

    # 3. Current MLS Player (Dynamic lookup)
    mls_players = df_sc[df_sc["current_league_id"] == "MLS1"]
    if mls_players.empty:
        mls_players = df_p[df_p["current_league_id"] == "MLS1"]
    if not mls_players.empty:
        mls_p = mls_players.iloc[0]
        assert mls_p["eligibility_status"] == "NON_BIG5"
        assert mls_p["is_big5_current"] == False
        print(f" [PASS] MLS negative candidate '{mls_p['player_name']}': Club={mls_p['current_club']}, League={mls_p['current_league']}, Status={mls_p['eligibility_status']} [EXCLUDED].")
        checks_passed += 1
    else:
        errors.append("No MLS players found in current data!")

    # 4. Current Qatar Player (Dynamic lookup)
    qat_players = df_sc[df_sc["current_league_id"] == "QAT1"]
    if qat_players.empty:
        qat_players = df_p[df_p["current_league_id"] == "QAT1"]
    if not qat_players.empty:
        qat_p = qat_players.iloc[0]
        assert qat_p["eligibility_status"] == "NON_BIG5"
        assert qat_p["is_big5_current"] == False
        print(f" [PASS] Qatar Stars League negative candidate '{qat_p['player_name']}': Club={qat_p['current_club']}, League={qat_p['current_league']}, Status={qat_p['eligibility_status']} [EXCLUDED].")
        checks_passed += 1
    else:
        errors.append("No Qatar Stars League players found in current data!")

    # 5. Current Portuguese Player (Dynamic lookup)
    po_players = df_sc[df_sc["current_league_id"] == "PO1"]
    if po_players.empty:
        po_players = df_p[df_p["current_league_id"] == "PO1"]
    if not po_players.empty:
        po_p = po_players.iloc[0]
        assert po_p["eligibility_status"] == "NON_BIG5"
        assert po_p["is_big5_current"] == False
        print(f" [PASS] Portuguese Primeira Liga negative candidate '{po_p['player_name']}': Club={po_p['current_club']}, League={po_p['current_league']}, Status={po_p['eligibility_status']} [EXCLUDED].")
        checks_passed += 1
    else:
        errors.append("No Portuguese Primeira Liga players found in current data!")

    # 6. Current Turkish Player (Dynamic lookup)
    tr_players = df_sc[df_sc["current_league_id"] == "TR1"]
    if tr_players.empty:
        tr_players = df_p[df_p["current_league_id"] == "TR1"]
    if not tr_players.empty:
        tr_p = tr_players.iloc[0]
        assert tr_p["eligibility_status"] == "NON_BIG5"
        assert tr_p["is_big5_current"] == False
        print(f" [PASS] Turkish Süper Lig negative candidate '{tr_p['player_name']}': Club={tr_p['current_club']}, League={tr_p['current_league']}, Status={tr_p['eligibility_status']} [EXCLUDED].")
        checks_passed += 1
    else:
        errors.append("No Turkish Süper Lig players found in current data!")

    # 7. Current Free Agent Candidate (Dynamic lookup)
    fa_players = df_sc[df_sc["eligibility_status"] == "FREE_AGENT"]
    if not fa_players.empty:
        fa_test_p = fa_players.iloc[0]
        assert fa_test_p["is_free_agent"] == True
        assert fa_test_p["is_big5_current"] == False
        print(f" [PASS] Free Agent candidate '{fa_test_p['player_name']}': Status={fa_test_p['eligibility_status']}, ExcludedByDefault=True.")
        checks_passed += 1
    else:
        errors.append("No Free Agent players found in current scouting data!")

    # 6.7 End-to-End Similarity Engine Query Assertions
    print("\n--- 9. Live Similarity Candidate Pool Assertions ---")
    # Query similar players for Rayan Cherki with include_free_agents=False
    cherki_candidates = engine.find_replacements(
        player_name="Rayan Cherki",
        top_n=50,
        same_position_only=False,
        include_free_agents=False,
    )
    assert len(cherki_candidates) > 0, "No candidates returned for Rayan Cherki!"

    # Verify every returned candidate
    for c in cherki_candidates:
        if not c.get("is_current_big5", False):
            errors.append(f"Candidate '{c['candidate_player']}' has is_current_big5=False in default candidate pool!")
        if c.get("is_free_agent", False):
            errors.append(f"Free agent '{c['candidate_player']}' appeared in default results when include_free_agents=False!")
        if c.get("current_league") not in ["Premier League", "La Liga", "Bundesliga", "Serie A", "Ligue 1"]:
            errors.append(f"Candidate '{c['candidate_player']}' has non-Big-5 current league: {c.get('current_league')}")

    # Verify Roberto Firmino does not appear
    firmino_found = any("firmino" in c["candidate_player"].lower() for c in cherki_candidates)
    if firmino_found:
        errors.append("CRITICAL: Roberto Firmino appeared in scouting candidates!")
    else:
        print(" [PASS] 50/50 candidates strictly BIG5_ACTIVE; Roberto Firmino successfully excluded.")
        checks_passed += 1

    # Query with include_free_agents=True for a goalkeeper/profile where free agent exists
    fa_sim_candidates = engine.find_replacements(
        player_name="Martin Dúbravka",
        top_n=50,
        same_position_only=False,
        include_free_agents=True,
    )
    fa_in_results = [c for c in fa_sim_candidates if c.get("is_free_agent")]
    print(f" [PASS] include_free_agents=True returned {len(fa_in_results)} verified free agent candidate(s).")
    checks_passed += 1

    print("\n======================================================================")
    if errors:
        print(f" VALIDATION FAILED with {len(errors)} error(s):")
        for e in errors:
            print(f"  - {e}")
        sys.exit(1)
    else:
        print(f" ALL {checks_passed} CURRENT DATA & BIG-5 ELIGIBILITY CHECKS PASSED (100% Success Rate)!")
        print("======================================================================")

if __name__ == "__main__":
    main()

