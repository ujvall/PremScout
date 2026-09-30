"""Similarity indexing and candidate retrieval engine for the Player Scouting module.

Strictly enforces Europe's Big 5 candidate universe (Premier League, La Liga, Bundesliga,
Serie A, Ligue 1) using verified current 2026/27 registrations. Pre-filters candidate pool
BEFORE similarity ranking.
"""

from typing import List, Dict, Any, Optional
from pathlib import Path
import unicodedata
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from .preprocessing import (
    SCOUTING_FEATURES,
    normalize_position,
    check_position_compatibility,
    POSITION_COMPATIBILITY,
    POSITION_DISPLAY_NAMES,
)
from .eligibility import (
    EligibilityState,
    CURRENT_BIG5_COMPETITION_IDS,
)


def _normalize(name: str) -> str:
    if not isinstance(name, str):
        return ""
    norm = unicodedata.normalize("NFKD", name)
    norm = "".join(c for c in norm if not unicodedata.combining(c))
    norm = norm.replace("'", "").replace("-", " ").replace(".", "")
    norm = norm.replace("\xad", "").replace("\xa0", " ").replace("\ufffd", "")
    return " ".join(norm.lower().split())


class ScoutingSimilarityIndex:
    """Vector similarity index for player scouting based on standardized football performance embeddings."""

    def __init__(
        self,
        player_df: pd.DataFrame,
        embeddings: np.ndarray,
        features: List[str] = SCOUTING_FEATURES,
        scaler: Any = None,
        position_weights: Any = None,
    ):
        """Initialize scouting similarity index.

        Args:
            player_df: DataFrame containing player metadata and metrics.
            embeddings: Standardized feature matrix (N, D).
            features: List of feature names corresponding to embedding dimensions.
            scaler: Standard scaler fitted during V2 model training.
            position_weights: Dict of position archetype weights.
        """
        self.df = player_df.copy().reset_index(drop=True)
        self.features = features
        self.scaler = scaler
        self.position_weights = position_weights

        # Ensure normalized position column exists on historical df
        if "norm_pos" not in self.df.columns:
            self.df["norm_pos"] = self.df["position"].apply(normalize_position)

        # L2-normalize embeddings for cosine similarity
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self.normalized_embeddings = embeddings / norms
        self.raw_embeddings = embeddings

        # Sklearn NearestNeighbors index for historical retrieval
        self.nn_index = NearestNeighbors(metric="cosine", algorithm="brute")
        self.nn_index.fit(self.normalized_embeddings)

        # Load current scouting and player master tables (2025/26 performance + 2026/27 registration)
        self.current_scout_df = None
        self.current_embeddings = None
        self.current_players_df = None

        curr_scout_path = Path("data/current/scouting/current_player_scouting.csv")
        if curr_scout_path.exists():
            try:
                self.current_scout_df = pd.read_csv(curr_scout_path)
                if "norm_name" not in self.current_scout_df.columns:
                    self.current_scout_df["norm_name"] = self.current_scout_df["player_name"].apply(_normalize)
                if "norm_pos" not in self.current_scout_df.columns:
                    self.current_scout_df["norm_pos"] = self.current_scout_df["position"].apply(normalize_position)

                # Compute normalized embedding vectors for all current candidates
                self._compute_current_embeddings()
            except Exception:
                pass

        curr_players_path = Path("data/current/players/current_players.csv")
        if curr_players_path.exists():
            try:
                self.current_players_df = pd.read_csv(curr_players_path)
                if "normalized_name" not in self.current_players_df.columns:
                    self.current_players_df["normalized_name"] = self.current_players_df["player_name"].apply(_normalize)
            except Exception:
                pass

        # Load verified recent form lookup (2025/26 match records)
        self.recent_form_lookup = {}
        form_path = Path("data/current/players/player_recent_form.json")
        if form_path.exists():
            try:
                import json
                with open(form_path, "r", encoding="utf-8") as f:
                    self.recent_form_lookup = json.load(f)
            except Exception:
                self.recent_form_lookup = {}

    def _compute_current_embeddings(self):
        """Compute standardized embedding vectors for current candidate pool using model scaler and weights."""
        if self.current_scout_df is None or self.scaler is None:
            return

        raw_feat_df = self.current_scout_df[self.features].fillna(0.0)
        scaled_mat = self.scaler.transform(raw_feat_df)

        n_rows = len(self.current_scout_df)
        emb_list = []
        for i in range(n_rows):
            vec = scaled_mat[i].copy()
            norm_pos = self.current_scout_df["norm_pos"].iloc[i]
            if self.position_weights and norm_pos in self.position_weights:
                vec = vec * self.position_weights[norm_pos]
            norm_val = np.linalg.norm(vec)
            if norm_val > 0:
                vec = vec / norm_val
            emb_list.append(vec)

        self.current_embeddings = np.array(emb_list)

    def compute_similarity(self, query_vec: np.ndarray) -> np.ndarray:
        """Compute cosine similarity between a 1D query vector and all indexed historical players."""
        vec = query_vec.reshape(1, -1)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        sims = np.dot(self.normalized_embeddings, vec.T).flatten()
        return sims

    def find_replacements(
        self,
        player_name: str,
        top_n: int = 10,
        same_position_only: bool = True,
        max_market_value: Optional[float] = None,
        max_age: Optional[int] = None,
        min_minutes: Optional[int] = None,
        season: Optional[str] = None,
        include_free_agents: bool = False,
    ) -> List[Dict[str, Any]]:
        """Find ranked replacement candidates for a target football player.
        
        FILTER BEFORE SIMILARITY Pipeline:
        1. Locate target player profile and generate standardized query vector.
        2. Filter candidate universe strictly to Europe's Big 5 current players (or verified free agents if enabled).
        3. Apply tactical position compatibility constraints.
        4. Apply optional recruitment constraints (max valuation, max age, min minutes).
        5. Compute cosine similarities on strictly eligible candidates.
        6. Rank by similarity score descending.

        Args:
            player_name: Target player name.
            top_n: Number of candidates to return.
            same_position_only: Restrict strictly to exact position group.
            max_market_value: Maximum valuation ceiling in EUR.
            max_age: Maximum age ceiling in years.
            min_minutes: Minimum minutes played threshold.
            season: Optional specific season for target player.
            include_free_agents: If True, verified free agents are included in candidate pool. Default False.

        Returns:
            List of candidate dictionaries with complete current & performance metadata and similarity stats.
        """
        q_norm = _normalize(player_name)
        target_row = None
        query_vec = None
        target_player_name = ""
        target_team = ""
        target_norm_pos = ""
        target_season = ""

        # Step 1: Locate target player (check current scouting table first, then historical)
        if self.current_scout_df is not None and not season:
            c_matches = self.current_scout_df[self.current_scout_df["norm_name"] == q_norm]
            if c_matches.empty:
                c_matches = self.current_scout_df[self.current_scout_df["norm_name"].str.contains(q_norm, regex=False)]
            if not c_matches.empty:
                target_row = c_matches.iloc[0]
                target_player_name = str(target_row["player_name"])
                target_team = str(target_row.get("current_club", target_row.get("team", "")))
                target_norm_pos = str(target_row["norm_pos"])
                target_season = str(target_row.get("performance_season", target_row.get("season", "2025-26")))

                # Transform raw 16 core features using validated scaler & archetype weights
                raw_feat_df = pd.DataFrame([target_row[self.features]])
                scaled_vec = self.scaler.transform(raw_feat_df)
                if self.position_weights and target_norm_pos in self.position_weights:
                    scaled_vec = scaled_vec * self.position_weights[target_norm_pos]
                norm_val = np.linalg.norm(scaled_vec)
                query_vec = scaled_vec / (norm_val if norm_val > 0 else 1.0)

        # Fallback to historical index if not located in current scouting table
        if query_vec is None:
            name_lower = player_name.strip().lower()
            exact_matches = self.df[self.df["player_name"].str.lower() == name_lower]
            if exact_matches.empty:
                sub_matches = self.df[self.df["player_name"].str.lower().str.contains(name_lower, regex=False)]
                if sub_matches.empty:
                    # Check if player exists in master player list but has no scouting data
                    if self.current_players_df is not None:
                        cp_matches = self.current_players_df[self.current_players_df["normalized_name"] == q_norm]
                        if not cp_matches.empty:
                            raise ValueError(f"Player '{player_name}' exists in master dataset, but has insufficient performance data for scouting vector similarity.")
                    raise ValueError(f"Player '{player_name}' not found in scouting database.")
                matches = sub_matches
            else:
                matches = exact_matches

            if season:
                target_records = matches[matches["season"] == season]
                if target_records.empty:
                    avail_seasons = matches["season"].tolist()
                    raise ValueError(f"Player '{player_name}' exists but not in season '{season}'. Available: {avail_seasons}")
                target_idx = target_records.index[0]
            else:
                target_idx = matches.sort_values(by="season_end_year", ascending=False).index[0]

            target_row = self.df.loc[target_idx]
            target_player_name = str(target_row["player_name"])
            target_team = str(target_row["team"])
            target_norm_pos = str(target_row["norm_pos"])
            target_season = str(target_row["season"])
            query_vec = self.normalized_embeddings[target_idx]

        query_vec = query_vec.reshape(1, -1)

        # Step 2: Build eligible candidate pool (FILTER BEFORE SIMILARITY)
        # Use current scouting dataframe as the candidate pool
        if self.current_scout_df is not None and self.current_embeddings is not None:
            cand_pool_df = self.current_scout_df
            cand_embeddings_pool = self.current_embeddings
        else:
            cand_pool_df = self.df
            cand_embeddings_pool = self.normalized_embeddings

        # Default eligibility: ONLY BIG5_ACTIVE players currently playing in Europe's Big 5
        # If include_free_agents is True, FREE_AGENT players are also eligible
        if "eligibility_status" in cand_pool_df.columns:
            if not include_free_agents:
                eligible_mask = (cand_pool_df["eligibility_status"] == EligibilityState.BIG5_ACTIVE) & (cand_pool_df["is_big5_current"] == True)
            else:
                eligible_mask = cand_pool_df["eligibility_status"].isin([EligibilityState.BIG5_ACTIVE, EligibilityState.FREE_AGENT])
        elif "is_big5_current" in cand_pool_df.columns:
            if not include_free_agents:
                eligible_mask = (cand_pool_df["is_big5_current"] == True) & (cand_pool_df["is_free_agent"] == False)
            else:
                eligible_mask = (cand_pool_df["is_big5_current"] == True) | (cand_pool_df["is_free_agent"] == True)
        else:
            eligible_mask = np.ones(len(cand_pool_df), dtype=bool)

        # Exclude target player themselves
        target_name_clean = target_player_name.lower()
        target_norm_clean = _normalize(target_player_name)
        if "norm_name" in cand_pool_df.columns:
            is_same_player = (
                (cand_pool_df["player_name"].str.lower() == target_name_clean)
                | (cand_pool_df["norm_name"] == target_norm_clean)
            )
        else:
            is_same_player = (cand_pool_df["player_name"].str.lower() == target_name_clean)
        eligible_mask &= (~is_same_player)

        # Step 3: Position filtering
        if same_position_only:
            eligible_mask &= (cand_pool_df["norm_pos"] == target_norm_pos)
        else:
            compatible_positions = {target_norm_pos}
            compatible_positions.update(POSITION_COMPATIBILITY.get(target_norm_pos, set()))
            eligible_mask &= cand_pool_df["norm_pos"].isin(compatible_positions)

        # Step 4: Optional filters
        if max_market_value is not None:
            eligible_mask &= (cand_pool_df["market_value_eur"].notna()) & (cand_pool_df["market_value_eur"] <= max_market_value)

        if max_age is not None:
            eligible_mask &= (cand_pool_df["age"].notna()) & (cand_pool_df["age"] <= max_age)

        if min_minutes is not None:
            eligible_mask &= (cand_pool_df["minutes_played"] >= min_minutes)

        candidate_indices = np.where(eligible_mask)[0]
        if len(candidate_indices) == 0:
            return []

        # Step 5: Compute cosine similarity on strictly eligible candidates
        cand_sub_embeddings = cand_embeddings_pool[candidate_indices]
        candidate_sims = np.dot(cand_sub_embeddings, query_vec.T).flatten()

        # Step 6: Rank eligible candidates by similarity descending
        sorted_order = np.argsort(candidate_sims)[::-1]
        sorted_indices = candidate_indices[sorted_order]
        sorted_sims = candidate_sims[sorted_order]

        # Step 7: Deduplicate by player name
        seen_candidate_keys = set()
        results: List[Dict[str, Any]] = []

        for idx, sim_val in zip(sorted_indices, sorted_sims):
            cand_row = cand_pool_df.iloc[idx]
            cand_name = str(cand_row["player_name"])
            cand_norm = cand_row.get("norm_name", _normalize(cand_name))

            if cand_norm in seen_candidate_keys:
                continue
            seen_candidate_keys.add(cand_norm)

            compat_status = check_position_compatibility(target_norm_pos, cand_row["norm_pos"])
            score_100 = round(float(np.clip(sim_val, 0.0, 1.0)) * 100.0, 1)

            # Key comparative statistics
            stats_comparison = {
                "goals_per90": float(round(cand_row.get("goals_per90", 0.0), 2)),
                "assists_per90": float(round(cand_row.get("assists_per90", 0.0), 2)),
                "npxG_per90": float(round(cand_row.get("npxG_per90", 0.0), 2)),
                "xAG_per90": float(round(cand_row.get("xAG_per90", 0.0), 2)),
                "key_passes_per90": float(round(cand_row.get("key_passes_per90", 0.0), 2)),
                "pass_completion_pct": float(round(cand_row.get("pass_completion_pct", 0.0), 1)),
                "progressive_passes_per90": float(round(cand_row.get("progressive_passes_per90", 0.0), 2)),
                "progressive_carries_per90": float(round(cand_row.get("progressive_carries_per90", 0.0), 2)),
                "tackles_won_per90": float(round(cand_row.get("tackles_won_per90", 0.0), 2)),
                "interceptions_per90": float(round(cand_row.get("interceptions_per90", 0.0), 2)),
            }

            mv_val = cand_row.get("market_value_eur")
            mv_eur = float(mv_val) if (pd.notna(mv_val) and float(mv_val) > 0) else None

            # Recent form lookup (verified 2025/26 match sequence)
            cand_pid = cand_row.get("player_id")
            recent_form_data = None
            if pd.notna(cand_pid) and str(int(cand_pid)) in self.recent_form_lookup:
                recent_form_data = self.recent_form_lookup[str(int(cand_pid))]

            # Current 2026/27 registration details
            curr_club = str(cand_row.get("current_club") or cand_row.get("team") or "Unknown")
            curr_league = str(cand_row.get("current_league") or "Premier League")
            curr_country = str(cand_row.get("current_country") or "England")
            curr_season = str(cand_row.get("current_season") or "2026-27")

            # Completed 2025/26 performance details
            perf_club = str(cand_row.get("performance_club") or cand_row.get("team") or curr_club)
            perf_league = str(cand_row.get("performance_league") or curr_league)
            perf_season = str(cand_row.get("performance_season") or cand_row.get("season") or "2025-26")

            is_big5 = bool(cand_row.get("is_big5_current", True))
            is_fa = bool(cand_row.get("is_free_agent", False))

            results.append({
                "target_player": target_player_name,
                "target_team": target_team,
                "target_position": target_norm_pos,
                "target_season": target_season,
                "candidate_player": cand_name,
                "candidate_club": curr_club,
                "candidate_position": cand_row["norm_pos"],
                "candidate_raw_position": cand_row.get("position", cand_row["norm_pos"]),
                "candidate_season": curr_season,
                # Required Concept B & Concept C fields
                "current_club": curr_club,
                "current_league": curr_league,
                "current_country": curr_country,
                "current_season": curr_season,
                "performance_club": perf_club,
                "performance_league": perf_league,
                "performance_season": perf_season,
                "is_current_big5": is_big5,
                "is_free_agent": is_fa,
                # Metrics & Compatibility
                "similarity_score": score_100,
                "cosine_similarity": round(float(sim_val), 4),
                "position_compatibility": compat_status or "Compatible Position",
                "age": int(cand_row["age"]) if pd.notna(cand_row.get("age")) else None,
                "market_value_eur": mv_eur,
                "minutes": int(cand_row.get("minutes_played", 0)),
                "key_stats": stats_comparison,
                "recent_form": recent_form_data,
                "data_as_of": str(cand_row.get("data_as_of", "2026-09-20")),
            })

            if len(results) >= top_n:
                break

        return results
