"""Entity resolution and name normalization utilities for PremAI.

Provides:
1. Canonical Premier League team-name mappings across football-data, Transfermarkt, FBref, and FPL.
2. Standardized Unicode-based player name normalization.
3. FBref player ID extraction.
4. Deterministic entity resolution with validated rapidfuzz fallback.
"""

from collections.abc import Iterable
import re
import unicodedata
from rapidfuzz import fuzz

# =====================================================================
# Canonical Premier League Team-Name Mapping
# =====================================================================

CANONICAL_TEAMS = {
    # Arsenal
    "arsenal": "Arsenal",
    "arsenal fc": "Arsenal",
    # Aston Villa
    "aston villa": "Aston Villa",
    "aston villa fc": "Aston Villa",
    # Bournemouth
    "bournemouth": "Bournemouth",
    "afc bournemouth": "Bournemouth",
    # Brentford
    "brentford": "Brentford",
    "brentford fc": "Brentford",
    # Brighton
    "brighton": "Brighton",
    "brighton & hove albion": "Brighton",
    "brighton and hove albion": "Brighton",
    "brighton & hove albion fc": "Brighton",
    # Burnley
    "burnley": "Burnley",
    "burnley fc": "Burnley",
    # Cardiff City
    "cardiff": "Cardiff City",
    "cardiff city": "Cardiff City",
    "cardiff city fc": "Cardiff City",
    # Chelsea
    "chelsea": "Chelsea",
    "chelsea fc": "Chelsea",
    # Crystal Palace
    "crystal palace": "Crystal Palace",
    "crystal palace fc": "Crystal Palace",
    # Everton
    "everton": "Everton",
    "everton fc": "Everton",
    # Fulham
    "fulham": "Fulham",
    "fulham fc": "Fulham",
    # Huddersfield Town
    "huddersfield": "Huddersfield Town",
    "huddersfield town": "Huddersfield Town",
    "huddersfield town fc": "Huddersfield Town",
    # Hull City
    "hull": "Hull City",
    "hull city": "Hull City",
    # Ipswich Town
    "ipswich": "Ipswich Town",
    "ipswich town": "Ipswich Town",
    # Leeds United
    "leeds": "Leeds United",
    "leeds united": "Leeds United",
    "leeds united fc": "Leeds United",
    # Leicester City
    "leicester": "Leicester City",
    "leicester city": "Leicester City",
    "leicester city fc": "Leicester City",
    # Liverpool
    "liverpool": "Liverpool",
    "liverpool fc": "Liverpool",
    # Luton Town
    "luton": "Luton Town",
    "luton town": "Luton Town",
    "luton town fc": "Luton Town",
    # Manchester City
    "man city": "Manchester City",
    "manchester city": "Manchester City",
    "manchester city fc": "Manchester City",
    # Manchester United
    "man united": "Manchester United",
    "manchester united": "Manchester United",
    "manchester utd": "Manchester United",
    "manchester united fc": "Manchester United",
    # Middlesbrough
    "middlesbrough": "Middlesbrough",
    "middlesbrough fc": "Middlesbrough",
    # Newcastle United
    "newcastle": "Newcastle United",
    "newcastle united": "Newcastle United",
    "newcastle utd": "Newcastle United",
    "newcastle united fc": "Newcastle United",
    # Norwich City
    "norwich": "Norwich City",
    "norwich city": "Norwich City",
    "norwich city fc": "Norwich City",
    # Nottingham Forest
    "nott'm forest": "Nottingham Forest",
    "nott'ham forest": "Nottingham Forest",
    "nottingham forest": "Nottingham Forest",
    "nottingham forest fc": "Nottingham Forest",
    # Queens Park Rangers
    "qpr": "Queens Park Rangers",
    "queens park rangers": "Queens Park Rangers",
    # Reading
    "reading": "Reading",
    "reading fc": "Reading",
    # Sheffield United
    "sheffield united": "Sheffield United",
    "sheffield utd": "Sheffield United",
    "sheffield united fc": "Sheffield United",
    # Southampton
    "southampton": "Southampton",
    "southampton fc": "Southampton",
    # Stoke City
    "stoke": "Stoke City",
    "stoke city": "Stoke City",
    "stoke city fc": "Stoke City",
    # Sunderland
    "sunderland": "Sunderland",
    "sunderland afc": "Sunderland",
    # Swansea City
    "swansea": "Swansea City",
    "swansea city": "Swansea City",
    "swansea city fc": "Swansea City",
    # Tottenham Hotspur
    "tottenham": "Tottenham Hotspur",
    "tottenham hotspur": "Tottenham Hotspur",
    "spurs": "Tottenham Hotspur",
    "tottenham hotspur fc": "Tottenham Hotspur",
    # Watford
    "watford": "Watford",
    "watford fc": "Watford",
    # West Bromwich Albion
    "west brom": "West Bromwich Albion",
    "west bromwich albion": "West Bromwich Albion",
    "west bromwich": "West Bromwich Albion",
    "west bromwich albion fc": "West Bromwich Albion",
    # West Ham United
    "west ham": "West Ham United",
    "west ham united": "West Ham United",
    "west ham united fc": "West Ham United",
    # Wigan Athletic
    "wigan": "Wigan Athletic",
    "wigan athletic": "Wigan Athletic",
    # Wolverhampton Wanderers
    "wolves": "Wolverhampton Wanderers",
    "wolverhampton wanderers": "Wolverhampton Wanderers",
    "wolverhampton": "Wolverhampton Wanderers",
    "wolverhampton wanderers fc": "Wolverhampton Wanderers",
}


def get_canonical_team_name(name: str | None) -> str:
    """Map any raw team string to its canonical Premier League name."""
    if not name or not isinstance(name, str):
        return "Unknown"

    cleaned = name.strip().lower()
    if cleaned in CANONICAL_TEAMS:
        return CANONICAL_TEAMS[cleaned]

    # Secondary lookup stripping FC / AFC
    no_fc = re.sub(r"\b(fc|afc)\b", "", cleaned).strip()
    if no_fc in CANONICAL_TEAMS:
        return CANONICAL_TEAMS[no_fc]

    # Return title-cased cleaned version if not in dictionary
    return name.strip()


# =====================================================================
# Player Name Normalization
# =====================================================================

COMMON_SUFFIXES = re.compile(r"\b(jr\.?|sr\.?|ii|iii|iv)\b", re.IGNORECASE)


def normalize_player_name(name: str | None) -> str:
    """Normalize player names for reliable cross-dataset entity matching.

    1. Unicode NFKD decomposition
    2. Remove combining diacritical marks (accents, umlauts, cedillas)
    3. Lowercase
    4. Remove punctuation and special characters
    5. Strip common generational suffixes (Jr, Sr, III)
    6. Normalize internal and trailing whitespace
    """
    if not name or not isinstance(name, str):
        return ""

    # Unicode NFKD decomposition
    nfkd = unicodedata.normalize("NFKD", name)
    # Strip combining diacritics
    stripped = "".join(c for c in nfkd if not unicodedata.combining(c))

    # Lowercase
    cleaned = stripped.lower()

    # Strip generational suffixes
    cleaned = COMMON_SUFFIXES.sub("", cleaned)

    # Remove non-letter characters (replace hyphens/apostrophes with space)
    cleaned = re.sub(r"[^a-z\s]", " ", cleaned)

    # Normalize whitespace
    return " ".join(cleaned.split())


# =====================================================================
# Deterministic Identifiers & Multi-Tier Matching
# =====================================================================

FBREF_URL_PATTERN = re.compile(r"/players/([a-zA-Z0-9]{8})/")


def extract_fbref_id(url: str | None) -> str | None:
    """Extract deterministic 8-character FBref player hash from URL."""
    if not url or not isinstance(url, str):
        return None
    match = FBREF_URL_PATTERN.search(url)
    return match.group(1) if match else None


def match_player_candidate(
    query_name: str,
    query_birth_year: int | None,
    query_team: str | None,
    candidates: Iterable[dict],
    min_fuzzy_score: float = 90.0,
) -> dict | None:
    """Find the best matching candidate using deterministic tiers + rapidfuzz validation.

    Each candidate dict is expected to contain:
    {'player_id': ..., 'norm_name': ..., 'birth_year': ..., 'canonical_team': ...}

    Matching Tiers:
    Tier 1: Exact normalized name + exact birth year
    Tier 2: Exact normalized name + exact canonical team
    Tier 3: Token set matching (e.g. "bruno fernandes" in "bruno miguel borges fernandes")
            validating with exact birth year
    Tier 4: Rapidfuzz token-sort similarity >= min_fuzzy_score
            VALIDATED by exact birth year match (or query_birth_year is None and team matches)

    Returns None if no candidate reliably satisfies the validation rules.
    """
    if not query_name:
        return None

    # Tier 1: Exact Name + Exact Birth Year
    if query_birth_year is not None:
        for c in candidates:
            if c.get("norm_name") == query_name and c.get("birth_year") == query_birth_year:
                return c

    # Tier 2: Exact Name + Exact Team
    if query_team and query_team != "Unknown":
        for c in candidates:
            if c.get("norm_name") == query_name and c.get("canonical_team") == query_team:
                return c

    # Tier 3: Substring / Token Containment + Birth Year
    if query_birth_year is not None:
        query_tokens = set(query_name.split())
        for c in candidates:
            c_name = c.get("norm_name", "")
            c_tokens = set(c_name.split())
            if c.get("birth_year") == query_birth_year and (
                query_tokens.issubset(c_tokens) or c_tokens.issubset(query_tokens)
            ):
                return c

    # Tier 4: Rapidfuzz Fallback with Strict Validation
    best_candidate = None
    highest_score = 0.0

    for c in candidates:
        c_name = c.get("norm_name", "")
        # Require birth year match to even consider fuzzy matching
        if query_birth_year is not None and c.get("birth_year") is not None:
            if abs(query_birth_year - c["birth_year"]) > 1:
                continue

        score = fuzz.token_sort_ratio(query_name, c_name)
        if score > highest_score and score >= min_fuzzy_score:
            # Validate: must match birth year OR canonical team
            by_match = query_birth_year is not None and c.get("birth_year") == query_birth_year
            team_match = query_team and c.get("canonical_team") == query_team
            if by_match or team_match:
                highest_score = score
                best_candidate = c

    return best_candidate
