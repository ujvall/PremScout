"""Similarity indexing and candidate retrieval engine for the Player Scouting module."""

from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from .preprocessing import (
    SCOUTING_FEATURES,
    normalize_position,
    check_position_compatibility,
    POSITION_DISPLAY_NAMES,
)


from pathlib import Path
import unicodedata

def _normalize(name: str) -> str:
    if not isinstance(name, str):
        return ""
    norm = unicodedata.normalize('NFKD', name)
    norm = "".join(c for c in norm if not unicodedata.combining(c))
    norm = norm.replace("'", "").replace("-", " ").replace(".", "")
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

        # Ensure normalized position column exists
        if "norm_pos" not in self.df.columns:
            self.df["norm_pos"] = self.df["position"].apply(normalize_position)

        # L2-normalize embeddings for cosine similarity
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self.normalized_embeddings = embeddings / norms
        self.raw_embeddings = embeddings

        # Sklearn NearestNeighbors index for efficient retrieval
        self.nn_index = NearestNeighbors(metric="cosine", algorithm="brute")
        self.nn_index.fit(self.normalized_embeddings)

        # Load current scouting and player master tables if present
        self.current_scout_df = None
        self.current_players_df = None
        curr_scout_path = Path("data/current/scouting/current_player_scouting.csv")
        if curr_scout_path.exists():
            try:
                self.current_scout_df = pd.read_csv(curr_scout_path)
                if "norm_name" not in self.current_scout_df.columns:
                    self.current_scout_df["norm_name"] = self.current_scout_df["player_name"].apply(_normalize)
            except Exception:
                pass

        curr_players_path = Path("data/current/players/current_players.csv")
        if curr_players_path.exists():
            try:
                self.current_players_df = pd.read_csv(curr_players_path)
            except Exception:
                pass

    def compute_similarity(self, query_vec: np.ndarray) -> np.ndarray:
        """Compute cosine similarity between a 1D query vector and all indexed players.

        Args:
            query_vec: 1D or 2D feature vector.

        Returns:
            1D array of cosine similarity scores in [-1.0, 1.0].
        """
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
    ) -> List[Dict[str, Any]]:
        """Find ranked replacement candidates for a target football player.

        Args:
            player_name: Name of target player (case-insensitive substring or exact match).
            top_n: Number of candidate recommendations to return.
            same_position_only: If True, candidate must have exact same normalized position.
                                If False, candidates in tactically compatible positions are allowed.
            max_market_value: Optional upper ceiling on candidate market value in EUR.
            max_age: Optional upper ceiling on candidate age in years.
            min_minutes: Optional lower threshold on candidate seasonal minutes played.
            season: Optional specific season for target player (e.g. '2023-24'). If None, uses latest.

        Returns:
            List of dictionaries containing candidate player profiles, similarity scores (0-100),
            position compatibility, and comparative statistics.
        """
        # 1. Locate target player records (check current 2025/26 scouting dataset first)
        q_norm = _normalize(player_name)
        target_from_current = False
        target_row = None

        if self.current_scout_df is not None and not season:
            c_matches = self.current_scout_df[self.current_scout_df["norm_name"] == q_norm]
            if c_matches.empty:
                c_matches = self.current_scout_df[self.current_scout_df["norm_name"].str.contains(q_norm, regex=False)]
            if not c_matches.empty:
                target_from_current = True
                target_row = c_matches.iloc[0]

        if target_from_current and target_row is not None and self.scaler is not None:
            target_player_name = str(target_row["player_name"])
            target_team = str(target_row["team"])
            target_norm_pos = str(target_row["norm_pos"])
            target_season = str(target_row["season"])

            # Transform raw 16 core features using validated V2 scaler & archetype weights
            raw_feat_df = pd.DataFrame([target_row[self.features]])
            scaled_vec = self.scaler.transform(raw_feat_df)
            if self.position_weights and target_norm_pos in self.position_weights:
                scaled_vec = scaled_vec * self.position_weights[target_norm_pos]
            norm_val = np.linalg.norm(scaled_vec)
            query_vec = scaled_vec / (norm_val if norm_val > 0 else 1.0)
            sims = self.compute_similarity(query_vec)
        else:
            name_lower = player_name.strip().lower()
            exact_matches = self.df[self.df["player_name"].str.lower() == name_lower]
            if exact_matches.empty:
                sub_matches = self.df[self.df["player_name"].str.lower().str.contains(name_lower, regex=False)]
                if sub_matches.empty:
                    # Check if player exists in master player list but has no scouting
                    if self.current_players_df is not None:
                        cp_matches = self.current_players_df[self.current_players_df["normalized_name"] == q_norm]
                        if not cp_matches.empty:
                            raise ValueError(f"Player '{player_name}' exists in master dataset, but has insufficient performance data for scouting vector similarity.")
                    raise ValueError(f"Player '{player_name}' not found in scouting database.")
                matches = sub_matches
            else:
                matches = exact_matches

            # If specific season requested, filter; otherwise select most recent season
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

            # 2. Build target feature vector & compute similarity
            query_vec = self.normalized_embeddings[target_idx]
            sims = self.compute_similarity(query_vec)

        # 3. Filter candidates
        # Must exclude target player itself (all historical seasons of target player)
        target_name_clean = target_player_name.lower()
        target_norm_clean = _normalize(target_player_name)
        is_same_player = (
            (self.df["player_name"].str.lower() == target_name_clean)
            | (self.df["player_name"].apply(_normalize) == target_norm_clean)
        )
        eligible_mask = ~is_same_player

        # Position filtering
        if same_position_only:
            eligible_mask &= (self.df["norm_pos"] == target_norm_pos)
        else:
            # Must be either same or compatible position
            compatible_positions = {target_norm_pos}
            from .preprocessing import POSITION_COMPATIBILITY
            compatible_positions.update(POSITION_COMPATIBILITY.get(target_norm_pos, set()))
            eligible_mask &= self.df["norm_pos"].isin(compatible_positions)

        # Optional filters
        if max_market_value is not None:
            # Keep records where market value <= max_market_value (exclude NaN market values when filtering)
            eligible_mask &= (self.df["market_value_eur"].notna()) & (self.df["market_value_eur"] <= max_market_value)

        if max_age is not None:
            eligible_mask &= (self.df["age"].notna()) & (self.df["age"] <= max_age)

        if min_minutes is not None:
            eligible_mask &= (self.df["minutes_played"] >= min_minutes)

        candidate_indices = np.where(eligible_mask)[0]
        if len(candidate_indices) == 0:
            return []

        # 4. Sort eligible candidates by similarity descending
        candidate_sims = sims[candidate_indices]
        sorted_order = np.argsort(candidate_sims)[::-1]
        sorted_indices = candidate_indices[sorted_order]
        sorted_sims = candidate_sims[sorted_order]

        # 5. Deduplicate candidates by player name (keep their best/highest-similarity season profile)
        seen_candidate_ids = set()
        results: List[Dict[str, Any]] = []

        for idx, sim_val in zip(sorted_indices, sorted_sims):
            cand_row = self.df.loc[idx]
            cand_name = cand_row["player_name"]
            cand_id = cand_row.get("fbref_id")
            cand_key = cand_id if pd.notna(cand_id) else cand_name.lower()

            if cand_key in seen_candidate_ids:
                continue
            seen_candidate_ids.add(cand_key)

            compat_status = check_position_compatibility(target_norm_pos, cand_row["norm_pos"])
            # Normalize score to 0-100 scale (clip negative similarities to 0)
            score_100 = round(float(np.clip(sim_val, 0.0, 1.0)) * 100.0, 1)

            # Key comparative statistics
            stats_comparison = {
                "goals_per90": float(round(cand_row["goals_per90"], 2)),
                "assists_per90": float(round(cand_row["assists_per90"], 2)),
                "npxG_per90": float(round(cand_row["npxG_per90"], 2)),
                "xAG_per90": float(round(cand_row["xAG_per90"], 2)),
                "key_passes_per90": float(round(cand_row["key_passes_per90"], 2)),
                "pass_completion_pct": float(round(cand_row["pass_completion_pct"], 1)),
                "progressive_passes_per90": float(round(cand_row["progressive_passes_per90"], 2)),
                "progressive_carries_per90": float(round(cand_row["progressive_carries_per90"], 2)),
                "tackles_won_per90": float(round(cand_row["tackles_won_per90"], 2)),
                "interceptions_per90": float(round(cand_row["interceptions_per90"], 2)),
            }

            mv_val = cand_row["market_value_eur"]
            mv_eur = float(mv_val) if pd.notna(mv_val) else None

            results.append({
                "target_player": target_row["player_name"],
                "target_team": target_row["team"],
                "target_position": target_norm_pos,
                "target_season": target_row["season"],
                "candidate_player": cand_name,
                "candidate_club": cand_row["team"],
                "candidate_position": cand_row["norm_pos"],
                "candidate_raw_position": cand_row["position"],
                "candidate_season": cand_row["season"],
                "similarity_score": score_100,
                "cosine_similarity": round(float(sim_val), 4),
                "position_compatibility": compat_status or "Compatible Position",
                "age": int(cand_row["age"]) if pd.notna(cand_row["age"]) else None,
                "market_value_eur": mv_eur,
                "minutes": int(cand_row["minutes_played"]),
                "key_stats": stats_comparison,
            })

            if len(results) >= top_n:
                break

        return results
