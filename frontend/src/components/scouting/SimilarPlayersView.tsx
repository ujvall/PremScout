import React, { useState, useRef } from "react";
import type { PlayerSearchItem, ReplacementCandidate } from "../../types/api";
import { fetchSimilarPlayers } from "../../api";
import { PlayerSearchInput } from "./PlayerSearchInput";
import { PlayerDetailCard } from "./PlayerDetailCard";
import { TableSkeleton } from "../common/Skeleton";
import { Users, Search, AlertCircle, ArrowDown } from "lucide-react";

interface SimilarPlayersViewProps {
  includeFreeAgents?: boolean;
  setIncludeFreeAgents?: (val: boolean) => void;
}

export const SimilarPlayersView: React.FC<SimilarPlayersViewProps> = ({
  includeFreeAgents = false,
}) => {
  const [selectedPlayer, setSelectedPlayer] = useState<PlayerSearchItem | null>(null);
  const [topN, setTopN] = useState<number>(10);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [candidates, setCandidates] = useState<ReplacementCandidate[]>([]);
  const [activeCandidate, setActiveCandidate] = useState<ReplacementCandidate | null>(null);
  const [hasSearched, setHasSearched] = useState<boolean>(false);
  const detailRef = useRef<HTMLDivElement>(null);

  const handleSearch = async () => {
    if (!selectedPlayer) {
      setError("Please select a player to find similar candidates.");
      return;
    }

    setLoading(true);
    setError(null);
    setActiveCandidate(null);
    setHasSearched(true);

    try {
      const results = await fetchSimilarPlayers(selectedPlayer.player_name, topN, includeFreeAgents);
      setCandidates(results);
      if (results.length > 0) {
        setActiveCandidate(results[0]);
      }
    } catch (err: unknown) {
      const msg =
        err instanceof Error ? err.message : "Failed to retrieve similar players.";
      setError(msg);
      setCandidates([]);
    } finally {
      setLoading(false);
    }
  };

  const handleSelectCandidate = (c: ReplacementCandidate) => {
    setActiveCandidate(c);
    // Smooth scroll down to detail card if user clicks a candidate
    setTimeout(() => {
      detailRef.current?.scrollIntoView({ behavior: "smooth", block: "nearest" });
    }, 50);
  };

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "1.5rem" }}>
      {/* 1. Target Player Search Controls */}
      <div className="card">
        <div className="card-header">
          <div>
            <h4 className="card-title">
              <Users size={17} color="var(--accent-tea-green)" />
              Target Player &amp; Similarity Configuration
            </h4>
            <p className="card-subtitle">
              Calculates multidimensional cosine similarity vectors across normalized European league metrics
            </p>
          </div>
        </div>

        <div style={{ display: "flex", flexDirection: "column", gap: "1rem" }}>
          <PlayerSearchInput
            selectedPlayer={selectedPlayer}
            onSelectPlayer={setSelectedPlayer}
            placeholder="Type player name to search (e.g. Declan Rice, Rodri, Martin Odegaard, Mo Salah)..."
            includeFreeAgents={includeFreeAgents}
          />

          <div
            style={{
              display: "flex",
              alignItems: "center",
              justifyContent: "space-between",
              flexWrap: "wrap",
              gap: "1rem",
              paddingTop: "0.25rem",
            }}
          >
            {/* Top N Candidate Selector */}
            <div style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
              <span style={{ fontSize: "0.8125rem", color: "var(--text-secondary)" }}>
                Return Top Candidates:
              </span>
              <div className="segmented-nav">
                {[5, 10, 15, 20].map((n) => (
                  <button
                    key={n}
                    type="button"
                    className={`segmented-nav-item ${topN === n ? "active" : ""}`}
                    style={{ padding: "0.25rem 0.625rem", fontSize: "0.8125rem" }}
                    onClick={() => setTopN(n)}
                  >
                    {n}
                  </button>
                ))}
              </div>
            </div>

            {/* Action button */}
            <button
              type="button"
              disabled={loading || !selectedPlayer}
              onClick={handleSearch}
              className="btn btn-primary"
            >
              {loading ? (
                <>
                  <div
                    className="animate-spin"
                    style={{
                      width: "14px",
                      height: "14px",
                      border: "2px solid #151515",
                      borderTopColor: "transparent",
                      borderRadius: "50%",
                    }}
                  />
                  <span>Finding Similar Players...</span>
                </>
              ) : (
                <>
                  <Search size={15} />
                  <span>Find Similar Players</span>
                </>
              )}
            </button>
          </div>
        </div>
      </div>

      {error && (
        <div
          style={{
            display: "flex",
            alignItems: "center",
            gap: "0.75rem",
            padding: "0.875rem 1.25rem",
            borderRadius: "var(--radius-md)",
            backgroundColor: "var(--accent-rose-subtle)",
            border: "1px solid rgba(244, 63, 94, 0.3)",
            color: "#fb7185",
            fontSize: "0.875rem",
          }}
        >
          <AlertCircle size={18} />
          <span>{error}</span>
        </div>
      )}

      {/* Loading Skeleton */}
      {loading && <TableSkeleton rows={topN} columns={10} />}

      {/* 2. Full-Width Candidates Results Table */}
      {!loading && candidates.length > 0 && (
        <div className="scouting-results-container">
          <div className="card" style={{ padding: "1.25rem" }}>
            <div
              style={{
                display: "flex",
                alignItems: "center",
                justifyContent: "space-between",
                marginBottom: "1rem",
                flexWrap: "wrap",
                gap: "0.5rem",
              }}
            >
              <div>
                <span style={{ fontSize: "0.9375rem", fontWeight: 700, color: "var(--text-primary)" }}>
                  Ranked Similar Candidates
                </span>
                <span
                  style={{
                    fontSize: "0.8125rem",
                    color: "var(--text-secondary)",
                    marginLeft: "0.5rem",
                  }}
                >
                  ({candidates.length} profiles calculated)
                </span>
              </div>
              <span style={{ fontSize: "0.75rem", color: "var(--text-muted)", display: "flex", alignItems: "center", gap: "0.3rem" }}>
                <span>Select any row to inspect profile below</span>
                <ArrowDown size={13} />
              </span>
            </div>

            <div className="table-container">
              <table className="data-table">
                <thead>
                  <tr>
                    <th style={{ width: "45px" }}>Rank</th>
                    <th>Player</th>
                    <th>Club</th>
                    <th>League</th>
                    <th>Position</th>
                    <th style={{ textAlign: "right" }}>Similarity</th>
                    <th>Compatibility</th>
                    <th>Age</th>
                    <th>Market Value</th>
                    <th style={{ textAlign: "right" }}>Minutes</th>
                  </tr>
                </thead>
                <tbody>
                  {candidates.map((c, idx) => {
                    const isSelected = activeCandidate?.candidate_player === c.candidate_player;
                    const clubDisplay = c.current_club || c.candidate_club;
                    const leagueDisplay = c.is_free_agent
                      ? "Free Agent"
                      : c.current_league || (c.is_current_big5 ? "Big 5 Active" : "—");
                    const mvDisplay =
                      c.market_value_eur != null && !isNaN(c.market_value_eur) && c.market_value_eur > 0
                        ? `€${(c.market_value_eur / 1e6).toFixed(1)}M`
                        : "Not available";

                    return (
                      <tr
                        key={`${c.candidate_player}-${c.candidate_season}`}
                        onClick={() => handleSelectCandidate(c)}
                        className={isSelected ? "selected" : ""}
                      >
                        <td style={{ fontWeight: 600, color: "var(--text-muted)", width: "45px" }}>
                          #{idx + 1}
                        </td>
                        <td style={{ fontWeight: 600, color: "var(--text-primary)" }}>
                          {c.candidate_player}
                        </td>
                        <td style={{ color: "var(--text-secondary)" }}>
                          {clubDisplay}
                        </td>
                        <td>
                          {c.is_free_agent ? (
                            <span className="badge badge-amber" style={{ fontSize: "0.71875rem" }}>
                              Free Agent
                            </span>
                          ) : (
                            <span style={{ color: "var(--text-secondary)", fontSize: "0.8125rem" }}>
                              {leagueDisplay}
                            </span>
                          )}
                        </td>
                        <td>
                          <span className="tag-position">
                            {c.candidate_position || c.candidate_raw_position}
                          </span>
                        </td>
                        <td style={{ textAlign: "right" }}>
                          <span
                            style={{
                              fontSize: "1rem",
                              fontWeight: 700,
                              fontFamily: "var(--font-mono)",
                              color: "var(--accent-tea-green)",
                            }}
                          >
                            {c.similarity_score.toFixed(1)}
                          </span>
                          <span style={{ fontSize: "0.6875rem", color: "var(--text-muted)", marginLeft: "2px" }}>
                            / 100
                          </span>
                        </td>
                        <td>
                          <span
                            className={`badge ${
                              c.position_compatibility === "Same Position"
                                ? "badge-emerald"
                                : "badge-blue"
                            }`}
                          >
                            {c.position_compatibility}
                          </span>
                        </td>
                        <td style={{ color: "var(--text-secondary)" }}>{c.age ?? "—"}</td>
                        <td>
                          <span
                            style={{
                              fontWeight: mvDisplay === "Not available" ? 400 : 600,
                              color: mvDisplay === "Not available" ? "var(--text-muted)" : "var(--text-primary)",
                              fontFamily: mvDisplay === "Not available" ? "inherit" : "var(--font-mono)",
                              fontSize: "0.8125rem",
                            }}
                          >
                            {mvDisplay}
                          </span>
                        </td>
                        <td style={{ textAlign: "right", color: "var(--text-muted)", fontFamily: "var(--font-mono)", fontSize: "0.8125rem" }}>
                          {c.minutes.toLocaleString()}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>

          {/* 3. Selected Candidate Profile Section — Renders Responsively Below Table */}
          {activeCandidate && (
            <div ref={detailRef} className="scouting-detail-section">
              <div
                style={{
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "space-between",
                  marginBottom: "0.75rem",
                }}
              >
                <h4 style={{ fontSize: "1rem", fontWeight: 700, color: "var(--text-secondary)", margin: 0 }}>
                  Selected Candidate Scouting Profile
                </h4>
              </div>
              <PlayerDetailCard
                candidate={activeCandidate}
                onClose={() => setActiveCandidate(null)}
              />
            </div>
          )}
        </div>
      )}

      {/* Empty State before search */}
      {!loading && candidates.length === 0 && !error && (
        <div
          className="card"
          style={{
            padding: "3.5rem 1.5rem",
            textAlign: "center",
            borderStyle: "dashed",
            borderColor: "var(--border-subtle)",
          }}
        >
          <div
            style={{
              width: "44px",
              height: "44px",
              borderRadius: "var(--radius-md)",
              backgroundColor: "var(--bg-elevated)",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              margin: "0 auto 1rem",
              color: "var(--text-muted)",
            }}
          >
            <Users size={20} />
          </div>
          <h4 style={{ fontSize: "0.9375rem", color: "var(--text-secondary)", marginBottom: "0.35rem" }}>
            {hasSearched ? "No similar candidates found" : "No target player selected"}
          </h4>
          <p style={{ fontSize: "0.8125rem", color: "var(--text-muted)", maxWidth: "380px", margin: "0 auto" }}>
            {hasSearched
              ? "No matching players met the statistical similarity threshold with the active filters."
              : "Search and choose an eligible Big 5 player to begin cosine similarity analysis and recruitment exploration."}
          </p>
        </div>
      )}
    </div>
  );
};
