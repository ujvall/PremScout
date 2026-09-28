import sys
sys.stdout.reconfigure(encoding='utf-8')

from backend.ml.scouting.predict import find_replacements, get_similar_players

def main():
    print("=== Test 1: Declan Rice with Budget Cap (max €45M) ===")
    reps = find_replacements("Declan Rice", top_n=3, same_position_only=True, max_market_value=45_000_000)
    for r in reps:
        mv = r['market_value_eur']
        mv_str = f"€{mv/1e6:.1f}M" if mv else "N/A"
        print(f"  {r['candidate_player']} ({r['candidate_club']}, Age {r['age']}, {mv_str}) - Score: {r['similarity_score']} / 100")

    print("\n=== Test 2: Erling Haaland Under 25 Years Old ===")
    reps2 = find_replacements("Erling Haaland", top_n=3, same_position_only=True, max_age=25)
    for r in reps2:
        mv = r['market_value_eur']
        mv_str = f"€{mv/1e6:.1f}M" if mv else "N/A"
        print(f"  {r['candidate_player']} ({r['candidate_club']}, Age {r['age']}, {mv_str}) - Score: {r['similarity_score']} / 100")

    print("\n=== Test 3: get_similar_players for Virgil van Dijk ===")
    sims = get_similar_players("Virgil van Dijk", top_n=3)
    for r in sims:
        print(f"  {r['candidate_player']} ({r['candidate_club']}, {r['candidate_position']}) - Score: {r['similarity_score']} / 100 ({r['position_compatibility']})")

    print("\n=== Test 4: Error Handling ===")
    try:
        find_replacements("Fake Nonexistent Player", top_n=5)
    except ValueError as e:
        print("  Caught expected unknown player error:", e)

    try:
        find_replacements("Declan Rice", max_age=10)
    except ValueError as e:
        print("  Caught expected invalid age error:", e)

    print("\nAll functional tests passed!")

if __name__ == "__main__":
    main()
