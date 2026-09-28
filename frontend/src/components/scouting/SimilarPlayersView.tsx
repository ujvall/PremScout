import React, { useState } from "react";
import type { PlayerSearchItem, ReplacementCandidate } from "../../types/api";
import { fetchSimilarPlayers } from "../../api";
import { PlayerSearchInput } from "./PlayerSearchInput";
import { PlayerDetailCard } from "./PlayerDetailCard";
import { Users, Search, AlertCircle } from "lucide-react";

export const SimilarPlayersView: React.FC = () => {
  const [selectedPlayer, setSelectedPlayer] = useState<PlayerSearchItem | null>(null);
  const [topN, setTopN] = useState<number>(10);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [candidates, setCandidates] = useState<ReplacementCandidate[]>([]);
  const [activeCandidate, setActiveCandidate] = useState<ReplacementCandidate | null>(null);
  const [hasSearched, setHasSearched] = useState<boolean>(false);

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
      const results = await fetchSimilarPlayers(selectedPlayer.player_name, topN);
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

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "1.5rem" }}>
      {/* Controls Card */}
      <div className="card">
        <div className="card-header">
          <div>
            <h4 className="card-title">
              <Users size={18} color="var(--accent-emerald)" />
              Select Target Player for Similarity Analysis
            </h4>
            <p className="card-subtitle">
              Calculates cosine similarity vectors across normalized football performance metrics
            </p>
          </div>
        </div>

        <div style={{ display: "flex", flexDirection: "column", gap: "1rem" }}>
          <PlayerSearchInput
            selectedPlayer={selectedPlayer}
            onSelectPlayer={setSelectedPlayer}
            placeholder="Type player name to search (e.g. Declan Rice, Rodri, Martin Odegaard, Mo Salah)..."
          />

          <div
            style={{
              display: "flex",
              alignItems: "center",
              justifyContent: "space-between",
              flexWrap: "wrap",
              gap: "1rem",
              paddingTop: "0.5rem",
            }}
          >
            {/* Top N Selector */}
            <div style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
              <span style={{ fontSize: "0.8125rem", color: "var(--text-secondary)" }}>
                Return Top N Candidates:
              </span>
              <div className="pill-nav">
                {[5, 10, 15, 20].map((n) => (
                  <button
                    key={n}
                    type="button"
                    className={`pill-item ${topN === n ? "active" : ""}`}
                    style={{ padding: "0.25rem 0.75rem", fontSize: "0.8125rem" }}
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
              style={{ padding: "0.625rem 1.5rem" }}
            >
              {loading ? (
                <>
                  <div
                    className="animate-spin"
                    style={{
                      width: "14px",
                      height: "14px",
                      border: "2px solid #06281e",
                      borderTopColor: "transparent",
                      borderRadius: "50%",
                    }}
                  />
                  Finding Similar Players...
                </>
              ) : (
                <>
                  <Search size={16} />
                  Find Similar Players
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
            padding: "1rem 1.25rem",
            borderRadius: "var(--radius-md)",
            backgroundColor: "rgba(244, 63, 94, 0.12)",
            border: "1px solid rgba(244, 63, 94, 0.3)",
            color: "#fb7185",
            fontSize: "0.875rem",
          }}
        >
          <AlertCircle size={18} />
          <span>{error}</span>
        </div>
      )}

      {/* Results View */}
      {candidates.length > 0 && (
        <div className="grid-layout-with-sidebar animate-fade-in">
          {/* Candidates Table Left */}
          <div>
            <div className="card" style={{ padding: "1rem" }}>
              <div
                style={{
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "space-between",
                  marginBottom: "0.75rem",
                }}
              >
                <span style={{ fontSize: "0.875rem", fontWeight: 600, color: "var(--text-secondary)" }}>
                  Ranked Similar Candidates ({candidates.length} matches)
                </span>
                <span style={{ fontSize: "0.75rem", color: "var(--text-muted)" }}>
                  Click row to view full statistical radar
                </span>
              </div>

              <div className="table-container">
                <table className="data-table">
                  <thead>
                    <tr>
                      <th style={{ width: "50px" }}>Rank</th>
                      <th>Player</th>
                      <th>Club</th>
                      <th>Position</th>
                      <th style={{ textAlign: "right" }}>Similarity</th>
                      <th>Compatibility</th>
                      <th>Age</th>
                      <th>Market Value</th>
                      <th>Minutes</th>
                    </tr>
                  </thead>
                  <tbody>
                    {candidates.map((c, idx) => {
                      const isSelected = activeCandidate?.candidate_player === c.candidate_player;
                      return (
                        <tr
                          key={`${c.candidate_player}-${c.candidate_season}`}
                          onClick={() => setActiveCandidate(c)}
                          className={isSelected ? "selected" : ""}
                        >
                          <td style={{ fontWeight: 700, color: "var(--text-muted)", width: "50px" }}>
                            #{idx + 1}
                          </td>
                          <td style={{ fontWeight: 600, color: "#ffffff" }}>
                            {c.candidate_player}
                          </td>
                          <td style={{ color: "var(--text-secondary)" }}>
                            {c.candidate_club}
                          </td>
                          <td>
                            <span className="badge badge-blue">
                              {c.candidate_position || c.candidate_raw_position}
                            </span>
                          </td>
                          <td style={{ textAlign: "right" }}>
                            <span
                              style={{
                                fontSize: "1.05rem",
                                fontWeight: 800,
                                fontFamily: "var(--font-mono)",
                                color: "var(--accent-emerald-light)",
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
                          <td>{c.age ?? "—"}</td>
                          <td style={{ fontFamily: "var(--font-mono)" }}>
                            {c.market_value_eur
                              ? `€${(c.market_value_eur / 1e6).toFixed(1)}M`
                              : "—"}
                          </td>
                          <td>{c.minutes}</td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </div>
          </div>

          {/* Active Candidate Detail Panel Right */}
          <div>
            <div style={{ position: "sticky", top: "5.5rem" }}>
              <PlayerDetailCard
                candidate={activeCandidate}
                onClose={() => setActiveCandidate(null)}
              />
            </div>
          </div>
        </div>
      )}

      {/* Empty State before search */}
      {!loading && candidates.length === 0 && !error && (
        <div
          className="card"
          style={{
            padding: "3rem 1.5rem",
            textAlign: "center",
            borderStyle: "dashed",
            borderColor: "var(--border-subtle)",
          }}
        >
          <div
            style={{
              width: "48px",
              height: "48px",
              borderRadius: "50%",
              backgroundColor: "var(--bg-elevated)",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              margin: "0 auto 1rem",
              color: "var(--text-muted)",
            }}
          >
            <Users size={22} />
          </div>
          <h4 style={{ fontSize: "1rem", color: "var(--text-secondary)", marginBottom: "0.35rem" }}>
            {hasSearched ? "No similar candidates found" : "No player selected"}
          </h4>
          <p style={{ fontSize: "0.8125rem", color: "var(--text-muted)", maxWidth: "340px", margin: "0 auto" }}>
            {hasSearched
              ? "No matching players met the statistical similarity threshold."
              : "Choose a target player to begin statistical cosine similarity analysis and recruitment exploration."}
          </p>
        </div>
      )}
    </div>
  );
};
