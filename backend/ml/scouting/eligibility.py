"""Scouting eligibility architecture and current 2026/27 player registration resolver.

Separates three distinct concepts:
A. PERFORMANCE DATA: performance_season, performance_club, performance_league
B. CURRENT REGISTRATION: current_season, current_club, current_club_id, current_league, current_league_id, current_competition, current_country
C. SCOUTING ELIGIBILITY: is_big5_current, is_free_agent, scouting_eligible, eligibility_status

Eligibility Taxonomy:
- BIG5_ACTIVE: Currently registered in a club competing in Europe's Big 5 leagues (Premier League, La Liga, Bundesliga, Serie A, Ligue 1) for 2026/27.
- FREE_AGENT: Verified explicit free agent (registration_status == 'free_agent'). Excluded by default; optionally included via 'Include Free Agents' toggle.
- NON_BIG5: Currently playing outside Europe's Big 5 (e.g. Saudi Pro League, Qatar Stars League, MLS, Liga Portugal, Süper Lig, etc.). ALWAYS excluded from replacement candidates.
- UNKNOWN: Missing, ambiguous, or unverifiable current registration status. ALWAYS excluded to prevent data corruption.
"""

from typing import Dict, Any, Optional, Set
from pathlib import Path
import unicodedata
import pandas as pd

# Stable Big 5 Domestic Competition Identifiers (Source Standards)
CURRENT_BIG5_COMPETITIONS: Dict[str, Dict[str, Any]] = {
    "GB1": {"league": "Premier League", "country": "England", "is_big5": True},
    "ES1": {"league": "La Liga", "country": "Spain", "is_big5": True},
    "L1": {"league": "Bundesliga", "country": "Germany", "is_big5": True},
    "IT1": {"league": "Serie A", "country": "Italy", "is_big5": True},
    "FR1": {"league": "Ligue 1", "country": "France", "is_big5": True},
}

CURRENT_BIG5_COMPETITION_IDS: Set[str] = frozenset(CURRENT_BIG5_COMPETITIONS.keys())

# Non-Big 5 Domestic Competitions Metadata
COMPETITION_METADATA: Dict[str, Dict[str, Any]] = {
    # Big 5
    **CURRENT_BIG5_COMPETITIONS,
    # Non-Big 5 domestic leagues
    "SA1": {"league": "Saudi Pro League", "country": "Saudi Arabia", "is_big5": False},
    "QAT1": {"league": "Qatar Stars League", "country": "Qatar", "is_big5": False},
    "MLS1": {"league": "MLS", "country": "United States", "is_big5": False},
    "PO1": {"league": "Primeira Liga", "country": "Portugal", "is_big5": False},
    "TR1": {"league": "Süper Lig", "country": "Turkey", "is_big5": False},
    "NL1": {"league": "Eredivisie", "country": "Netherlands", "is_big5": False},
    "BE1": {"league": "Belgian Pro League", "country": "Belgium", "is_big5": False},
    "BRA1": {"league": "Brazilian Serie A", "country": "Brazil", "is_big5": False},
    "ARG1": {"league": "Argentine Primera División", "country": "Argentina", "is_big5": False},
    "RU1": {"league": "Russian Premier League", "country": "Russia", "is_big5": False},
    "UKR1": {"league": "Ukrainian Premier League", "country": "Ukraine", "is_big5": False},
    "GR1": {"league": "Super League Greece", "country": "Greece", "is_big5": False},
    "SC1": {"league": "Scottish Premiership", "country": "Scotland", "is_big5": False},
    "DK1": {"league": "Danish Superliga", "country": "Denmark", "is_big5": False},
    "GB2": {"league": "EFL Championship", "country": "England", "is_big5": False},
    "A1": {"league": "Austrian Bundesliga", "country": "Austria", "is_big5": False},
    "AUS1": {"league": "A-League", "country": "Australia", "is_big5": False},
    "C1": {"league": "Swiss Super League", "country": "Switzerland", "is_big5": False},
    "COL1": {"league": "Categoría Primera A", "country": "Colombia", "is_big5": False},
    "JAP1": {"league": "J1 League", "country": "Japan", "is_big5": False},
    "KR1": {"league": "K League 1", "country": "South Korea", "is_big5": False},
    "MEX1": {"league": "Liga MX", "country": "Mexico", "is_big5": False},
    "NO1": {"league": "Eliteserien", "country": "Norway", "is_big5": False},
    "PL1": {"league": "Ekstraklasa", "country": "Poland", "is_big5": False},
    "RO1": {"league": "Liga I", "country": "Romania", "is_big5": False},
    "RSK1": {"league": "Czech First League", "country": "Czech Republic", "is_big5": False},
    "SE1": {"league": "Allsvenskan", "country": "Sweden", "is_big5": False},
    "SER1": {"league": "Serbian SuperLiga", "country": "Serbia", "is_big5": False},
    "TS1": {"league": "Slovak First League", "country": "Slovakia", "is_big5": False},
    # Free Agent State
    "FREE": {"league": "Free Agent", "country": "None", "is_big5": False},
}

# Mapping of normalized league names back to competition IDs
LEAGUE_NAME_TO_COMP_ID: Dict[str, str] = {
    "premier league": "GB1",
    "la liga": "ES1",
    "bundesliga": "L1",
    "serie a": "IT1",
    "ligue 1": "FR1",
    "saudi pro league": "SA1",
    "qatar stars league": "QAT1",
    "mls": "MLS1",
    "primeira liga": "PO1",
    "super lig": "TR1",
    "eredivisie": "NL1",
    "belgian pro league": "BE1",
    "brazilian serie a": "BRA1",
    "free agent": "FREE",
}

# Explicit aliases for standard Big 5 European squads across data sources
BIG5_SQUAD_ALIASES: Dict[str, Dict[str, Any]] = {
    "athletic club": {"name": "Athletic Bilbao", "cid": 621, "comp": "ES1"},
    "atletico madrid": {"name": "Atlético de Madrid", "cid": 13, "comp": "ES1"},
    "atletico de madrid": {"name": "Atlético de Madrid", "cid": 13, "comp": "ES1"},
    "celta vigo": {"name": "RC Celta de Vigo", "cid": 940, "comp": "ES1"},
    "rc celta": {"name": "RC Celta de Vigo", "cid": 940, "comp": "ES1"},
    "eint frankfurt": {"name": "Eintracht Frankfurt", "cid": 24, "comp": "L1"},
    "eintracht frankfurt": {"name": "Eintracht Frankfurt", "cid": 24, "comp": "L1"},
    "manchester utd": {"name": "Manchester United", "cid": 985, "comp": "GB1"},
    "manchester united": {"name": "Manchester United", "cid": 985, "comp": "GB1"},
    "newcastle utd": {"name": "Newcastle United", "cid": 762, "comp": "GB1"},
    "newcastle united": {"name": "Newcastle United", "cid": 762, "comp": "GB1"},
    "nottingham forest": {"name": "Nottingham Forest", "cid": 703, "comp": "GB1"},
    "nottm forest": {"name": "Nottingham Forest", "cid": 703, "comp": "GB1"},
    "paris s g": {"name": "Paris Saint-Germain", "cid": 583, "comp": "FR1"},
    "paris saint germain": {"name": "Paris Saint-Germain", "cid": 583, "comp": "FR1"},
    "psg": {"name": "Paris Saint-Germain", "cid": 583, "comp": "FR1"},
    "rennes": {"name": "Stade Rennais FC", "cid": 273, "comp": "FR1"},
    "stade rennais": {"name": "Stade Rennais FC", "cid": 273, "comp": "FR1"},
    "wolves": {"name": "Wolverhampton Wanderers", "cid": 543, "comp": "GB1"},
    "wolverhampton wanderers": {"name": "Wolverhampton Wanderers", "cid": 543, "comp": "GB1"},
    "betis": {"name": "Real Betis Balompié", "cid": 150, "comp": "ES1"},
    "real betis": {"name": "Real Betis Balompié", "cid": 150, "comp": "ES1"},
    "alaves": {"name": "Deportivo Alavés", "cid": 1108, "comp": "ES1"},
    "deportivo alaves": {"name": "Deportivo Alavés", "cid": 1108, "comp": "ES1"},
    "atalanta": {"name": "Atalanta BC", "cid": 800, "comp": "IT1"},
    "augsburg": {"name": "FC Augsburg", "cid": 167, "comp": "L1"},
    "fc augsburg": {"name": "FC Augsburg", "cid": 167, "comp": "L1"},
    "bayer leverkusen": {"name": "Bayer 04 Leverkusen", "cid": 15, "comp": "L1"},
    "leverkusen": {"name": "Bayer 04 Leverkusen", "cid": 15, "comp": "L1"},
    "bayern munich": {"name": "Bayern Munich", "cid": 27, "comp": "L1"},
    "fc bayern": {"name": "Bayern Munich", "cid": 27, "comp": "L1"},
    "bologna": {"name": "Bologna Football Club 1909", "cid": 1025, "comp": "IT1"},
    "inter": {"name": "Inter Milan", "cid": 46, "comp": "IT1"},
    "internazionale": {"name": "Inter Milan", "cid": 46, "comp": "IT1"},
    "milan": {"name": "AC Milan", "cid": 5, "comp": "IT1"},
    "ac milan": {"name": "AC Milan", "cid": 5, "comp": "IT1"},
    "roma": {"name": "AS Roma", "cid": 12, "comp": "IT1"},
    "as roma": {"name": "AS Roma", "cid": 12, "comp": "IT1"},
    "juventus": {"name": "Juventus FC", "cid": 506, "comp": "IT1"},
    "real madrid": {"name": "Real Madrid", "cid": 418, "comp": "ES1"},
    "barcelona": {"name": "FC Barcelona", "cid": 131, "comp": "ES1"},
    "fc barcelona": {"name": "FC Barcelona", "cid": 131, "comp": "ES1"},
    "dortmund": {"name": "Borussia Dortmund", "cid": 16, "comp": "L1"},
    "borussia dortmund": {"name": "Borussia Dortmund", "cid": 16, "comp": "L1"},
    "rb leipzig": {"name": "RB Leipzig", "cid": 23826, "comp": "L1"},
    "leipzig": {"name": "RB Leipzig", "cid": 23826, "comp": "L1"},
    "marseille": {"name": "Olympique Marseille", "cid": 244, "comp": "FR1"},
    "lyon": {"name": "Olympique Lyon", "cid": 1041, "comp": "FR1"},
    "monaco": {"name": "AS Monaco", "cid": 162, "comp": "FR1"},
    "lille": {"name": "LOSC Lille", "cid": 1082, "comp": "FR1"},
}

# Data quality states
class EligibilityState:
    BIG5_ACTIVE = "BIG5_ACTIVE"
    FREE_AGENT = "FREE_AGENT"
    NON_BIG5 = "NON_BIG5"
    UNKNOWN = "UNKNOWN"


def normalize_string(val: str) -> str:
    """Robust Unicode string normalization."""
    if not isinstance(val, str):
        return ""
    norm = unicodedata.normalize("NFKD", val)
    norm = "".join(c for c in norm if not unicodedata.combining(c))
    norm = norm.replace("'", "").replace("-", " ").replace(".", "")
    norm = norm.replace("\xad", "").replace("\xa0", " ").replace("\ufffd", "")
    return " ".join(norm.lower().split())


class CurrentPlayerResolver:
    """Single reusable current-player resolver for Europe's Big 5 and 2026/27 squad status.
    
    Resolution precedence:
    1. Verified post-summer-2026 squad overlay (by stable player_id)
    2. Summer 2026 transfers to Without Club / club 515 (free agent)
    3. Stable club ID in clubs master (club_id -> domestic_competition_id)
    4. Stable competition ID in player record
    5. Fallback name normalization
    """

    _instance: Optional["CurrentPlayerResolver"] = None

    def __init__(self, data_dir: Optional[Path] = None):
        self.base_dir = data_dir or Path("data")
        self.verified_players: Dict[int, Dict[str, Any]] = {}
        self.club_registry: Dict[int, Dict[str, Any]] = {}
        self.free_agent_ids: Set[int] = set()
        self._load_reference_data()

    @classmethod
    def get_instance(cls, data_dir: Optional[Path] = None) -> "CurrentPlayerResolver":
        if cls._instance is None:
            cls._instance = cls(data_dir=data_dir)
        return cls._instance

    def _load_reference_data(self):
        """Load clubs master, verified 2026 squad overlay, and free agent transfers."""
        # 1. Clubs master
        clubs_path = self.base_dir / "raw" / "transfermarkt" / "clubs.csv"
        if clubs_path.exists():
            try:
                clubs_df = pd.read_csv(clubs_path)
                for _, row in clubs_df.iterrows():
                    cid = int(row["club_id"])
                    comp_id = str(row.get("domestic_competition_id") or "").strip()
                    cname = str(row.get("name") or "").strip()
                    self.club_registry[cid] = {
                        "name": cname,
                        "competition_id": comp_id,
                    }
            except Exception:
                pass

        # 2. Known clubs outside clubs.csv (e.g. Al-Sadd is club 656 in Qatar)
        self.club_registry[656] = {"name": "Al-Sadd", "competition_id": "QAT1"}
        self.club_registry[515] = {"name": "Without Club", "competition_id": "FREE"}

        # 3. Verified 2026/27 squad overlay
        overlay_path = self.base_dir / "current" / "players" / "verified_2026_squads.csv"
        if overlay_path.exists():
            try:
                overlay_df = pd.read_csv(overlay_path)
                for _, row in overlay_df.iterrows():
                    pid = int(row["player_id"])
                    cid = int(row["current_club_id"]) if pd.notna(row.get("current_club_id")) else None
                    self.verified_players[pid] = {
                        "player_id": pid,
                        "player_name": str(row["player_name"]),
                        "current_club_id": cid,
                        "current_club_name": str(row["current_club_name"]),
                        "current_competition_id": str(row["current_competition_id"]),
                        "registration_status": str(row.get("registration_status", "registered")),
                        "verified_date": str(row.get("verified_date", "2026-09-01")),
                    }
                    if row.get("registration_status") == "free_agent" or cid == 515:
                        self.free_agent_ids.add(pid)
            except Exception:
                pass

        # 4. Summer 2026 transfers to Without Club (club 515)
        transfers_path = self.base_dir / "raw" / "transfermarkt" / "transfers.csv"
        if transfers_path.exists():
            try:
                t_df = pd.read_csv(transfers_path, usecols=["player_id", "to_club_id", "to_club_name", "transfer_date"])
                fa_26 = t_df[(t_df["to_club_name"] == "Without Club") & (t_df["transfer_date"].str.startswith("2026", na=False))]
                for pid in fa_26["player_id"].dropna().astype(int).unique():
                    self.free_agent_ids.add(pid)
            except Exception:
                pass

    def resolve(
        self,
        player_id: Optional[int] = None,
        player_name: Optional[str] = None,
        raw_club_id: Optional[int] = None,
        raw_club_name: Optional[str] = None,
        raw_competition_id: Optional[str] = None,
        raw_season: Optional[str] = None,
        registration_status_hint: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Resolve a player's verified 2026/27 registration, competition, and Big-5 eligibility.
        
        Returns:
            Dictionary containing:
            - player_id: int or None
            - player_name: str
            - current_season: '2026-27'
            - current_club: str
            - current_club_id: int or None
            - current_league: str (e.g. 'Premier League', 'La Liga', 'Saudi Pro League')
            - current_league_id: str (e.g. 'GB1', 'ES1', 'SA1', 'FREE')
            - current_competition: str
            - current_country: str
            - registration_status: 'registered' | 'free_agent' | 'unknown'
            - eligibility_status: 'BIG5_ACTIVE' | 'FREE_AGENT' | 'NON_BIG5' | 'UNKNOWN'
            - is_big5_current: bool
            - is_free_agent: bool
            - scouting_eligible: bool (default eligibility without free agents toggle)
        """
        pid = int(player_id) if player_id is not None and pd.notna(player_id) else None
        pname = str(player_name or "").strip()

        # Step 1: Check verified post-summer-2026 squad overlay by stable player_id
        if pid is not None and pid in self.verified_players:
            v = self.verified_players[pid]
            cid = v["current_club_id"]
            cname = v["current_club_name"]
            comp_id = v["current_competition_id"]
            reg_status = v["registration_status"]
            return self._build_result(pid, pname or v["player_name"], cid, cname, comp_id, reg_status)

        # Step 2: Check explicit Free Agent status
        if (pid is not None and pid in self.free_agent_ids) or registration_status_hint == "free_agent" or raw_club_id == 515:
            return self._build_result(pid, pname, 515, "Without Club", "FREE", "free_agent")

        # Step 3: Resolve using stable club_id
        cid = int(raw_club_id) if raw_club_id is not None and pd.notna(raw_club_id) else None
        if cid is not None and cid in self.club_registry:
            club_info = self.club_registry[cid]
            cname = raw_club_name or club_info["name"]
            comp_id = club_info["competition_id"] or raw_competition_id
            return self._build_result(pid, pname, cid, cname, comp_id, "registered")

        # Step 4: Resolve using stable competition_id if provided
        comp_id = str(raw_competition_id).strip() if raw_competition_id and pd.notna(raw_competition_id) else None
        cname = str(raw_club_name or "").strip()
        if comp_id and comp_id in COMPETITION_METADATA and cname and cname.lower() not in ["nan", "unknown", "none", ""]:
            return self._build_result(pid, pname, cid, cname, comp_id, "registered")

        # Step 5: Check if club name matches BIG5_SQUAD_ALIASES or club registry
        if cname and cname.lower() not in ["nan", "unknown", "none", ""]:
            norm_cname = normalize_string(cname)
            # Check explicit Big 5 aliases
            if norm_cname in BIG5_SQUAD_ALIASES:
                sq_match = BIG5_SQUAD_ALIASES[norm_cname]
                return self._build_result(pid, pname, sq_match["cid"], sq_match["name"], sq_match["comp"], "registered")

            # Try to match club name in club_registry
            for reg_cid, reg_info in self.club_registry.items():
                if normalize_string(reg_info["name"]) == norm_cname:
                    return self._build_result(pid, pname, reg_cid, reg_info["name"], reg_info["competition_id"], "registered")

        # Step 6: Unknown or unverified status
        return self._build_result(pid, pname, cid, cname or "Unknown", "UNKNOWN", "unknown")

    def _build_result(
        self,
        player_id: Optional[int],
        player_name: str,
        club_id: Optional[int],
        club_name: str,
        comp_id: Optional[str],
        reg_status: str,
    ) -> Dict[str, Any]:
        """Construct normalized registration and eligibility record."""
        current_season = "2026-27"
        comp_id_clean = str(comp_id or "").strip().upper()

        if reg_status == "free_agent" or comp_id_clean == "FREE" or club_id == 515:
            elig_status = EligibilityState.FREE_AGENT
            is_big5 = False
            is_fa = True
            scouting_eligible = False  # default is false until include_free_agents enabled
            league_name = "Free Agent"
            country = "None"
            comp_id_final = "FREE"
            club_final = "Without Club"
            cid_final = 515
        elif comp_id_clean in CURRENT_BIG5_COMPETITION_IDS:
            elig_status = EligibilityState.BIG5_ACTIVE
            is_big5 = True
            is_fa = False
            scouting_eligible = True
            comp_meta = CURRENT_BIG5_COMPETITIONS[comp_id_clean]
            league_name = comp_meta["league"]
            country = comp_meta["country"]
            comp_id_final = comp_id_clean
            club_final = club_name
            cid_final = club_id
        elif comp_id_clean in COMPETITION_METADATA:
            elig_status = EligibilityState.NON_BIG5
            is_big5 = False
            is_fa = False
            scouting_eligible = False
            comp_meta = COMPETITION_METADATA[comp_id_clean]
            league_name = comp_meta["league"]
            country = comp_meta["country"]
            comp_id_final = comp_id_clean
            club_final = club_name
            cid_final = club_id
        else:
            # Unverified or missing
            elig_status = EligibilityState.UNKNOWN
            is_big5 = False
            is_fa = False
            scouting_eligible = False
            league_name = "Unknown"
            country = "Unknown"
            comp_id_final = "UNKNOWN"
            club_final = club_name or "Unknown"
            cid_final = club_id

        return {
            "player_id": player_id,
            "player_name": player_name,
            "current_season": current_season,
            "current_club": club_final,
            "current_club_id": cid_final,
            "current_league": league_name,
            "current_league_id": comp_id_final,
            "current_competition": league_name,
            "current_country": country,
            "registration_status": reg_status,
            "eligibility_status": elig_status,
            "is_big5_current": is_big5,
            "is_free_agent": is_fa,
            "scouting_eligible": scouting_eligible,
        }
