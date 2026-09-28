import React, { useState } from "react";
import type {
  PlayerSearchItem,
  ReplacementCandidate,
  ScoutingReplacementRequest,
} from "../../types/api";
import { findReplacements } from "../../api";
import { PlayerSearchInput } from "./PlayerSearchInput";
import { PlayerDetailCard } from "./PlayerDetailCard";
import {
  Filter,
  Search,
  AlertCircle,
  RotateCcw,
  CheckCircle2,
  ArrowUpDown,
} from "lucide-react";

type SortField = "similarity_score" | "age" | "market_value_eur" | "minutes";
type SortOrder = "asc" | "desc";

export const ReplacementSearchView: React.FC = () => {
  const [selectedPlayer, setSelectedPlayer] = useState<PlayerSearchItem | null>(null);
  const [topN, setTopN] = useState<number>(10);
  const [samePositionOnly, setSamePositionOnly] = useState<boolean>(true);
  const [maxMarketValueM, setMaxMarketValueM] = useState<number | "">("");
  const [maxAge, setMaxAge] = useState<number | "">("");
  const [minMinutes, setMinMinutes] = useState<number | "">(600);

  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [candidates, setCandidates] = useState<ReplacementCandidate[]>([]);
  const [activeCandidate, setActiveCandidate] = useState<ReplacementCandidate | null>(null);
  const [hasSearched, setHasSearched] = useState<boolean>(false);

  // Sorting
  const [sortField, setSortField] = useState<SortField>("similarity_score");
  const [sortOrder, setSortOrder] = useState<SortOrder>("desc");

  const handleSearch = async () => {
    if (!selectedPlayer) {
      setError("Please select a target player to find replacement candidates.");
      return;
    }

    setLoading(true);
    setError(null);
    setActiveCandidate(null);
    setHasSearched(true);

    const payload: ScoutingReplacementRequest = {
      player_name: selectedPlayer.player_name,
      top_n: topN,
      same_position_only: samePositionOnly,
      max_market_value: maxMarketValueM !== "" && Number(maxMarketValueM) > 0 ? Number(maxMarketValueM) * 1e6 : null,
      max_age: maxAge !== "" && Number(maxAge) > 0 ? Number(maxAge) : null,
      min_minutes: minMinutes !== "" && Number(minMinutes) >= 0 ? Number(minMinutes) : null,
    };

    try {
      const results = await findReplacements(payload);
      setCandidates(results);
      if (results.length > 0) {
        setActiveCandidate(results[0]);
      }
    } catch (err: unknown) {
      const msg =
        err instanceof Error ? err.message : "Failed to find replacement candidates.";
      setError(msg);
      setCandidates([]);
    } finally {
      setLoading(false);
    }
  };

  const handleResetFilters = () => {
    setTopN(10);
    setSamePositionOnly(true);
    setMaxMarketValueM("");
    setMaxAge("");
    setMinMinutes(600);
  };

  const handleSort = (field: SortField) => {
    if (sortField === field) {
      setSortOrder((prev) => (prev === "asc" ? "desc" : "asc"));
    } else {
      setSortField(field);
      setSortOrder("desc");
    }
  };

  const sortedCandidates = [...candidates].sort((a, b) => {
    let aVal = a[sortField] ?? (sortOrder === "asc" ? Infinity : -Infinity);
    let bVal = b[sortField] ?? (sortOrder === "asc" ? Infinity : -Infinity);
    if (sortOrder === "asc") {
      return aVal > bVal ? 1 : -1;
    } else {
      return aVal < bVal ? 1 : -1;
    }
  });

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "1.5rem" }}>
      {/* Search & Filters Card */}
      <div className="card">
        <div className="card-header">
          <div>
            <h4 className="card-title">
              <Filter size={18} color="var(--accent-blue)" />
              Target Player &amp; Recruitment Filters
            </h4>
            <p className="card-subtitle">
              Filter replacements by position compatibility, maximum budget, age cap, and minimum playing time
            </p>
          </div>
          <button
            type="button"
            onClick={handleResetFilters}
            className="btn btn-ghost"
            style={{ padding: "0.25rem 0.5rem", fontSize: "0.75rem", gap: "0.25rem" }}
          >
            <RotateCcw size={12} />
            <span>Reset Filters</span>
          </button>
        </div>

        <div style={{ display: "flex", flexDirection: "column", gap: "1.25rem" }}>
          {/* Target Player Autocomplete */}
          <PlayerSearchInput
            selectedPlayer={selectedPlayer}
            onSelectPlayer={setSelectedPlayer}
            placeholder="Type player to replace (e.g. Harry Kane, Granit Xhaka, Declan Rice, Mason Mount)..."
          />

          {/* Filter Controls Grid */}
          <div className="grid-4" style={{ paddingTop: "0.5rem" }}>
            {/* Same Position Only */}
            <div className="form-group">
              <label className="form-label" htmlFor="scout-same-pos">Position Constraint</label>
              <div
                style={{
                  display: "flex",
                  alignItems: "center",
                  gap: "0.5rem",
                  padding: "0.625rem 0.875rem",
                  backgroundColor: "var(--bg-input)",
                  border: "1px solid var(--border-subtle)",
                  borderRadius: "var(--radius-md)",
                  height: "38px",
                }}
              >
                <input
                  type="checkbox"
                  id="scout-same-pos"
                  checked={samePositionOnly}
                  onChange={(e) => setSamePositionOnly(e.target.checked)}
                  style={{ accentColor: "var(--accent-emerald)", cursor: "pointer" }}
                />
                <label
                  htmlFor="scout-same-pos"
                  style={{ fontSize: "0.8125rem", cursor: "pointer", color: "var(--text-primary)" }}
                >
                  Strict Position Match
                </label>
              </div>
            </div>

            {/* Max Market Value */}
            <div className="form-group">
              <label className="form-label" htmlFor="scout-max-val">Max Valuation (€M)</label>
              <input
                id="scout-max-val"
                type="number"
                min="0"
                step="5"
                className="form-input"
                placeholder="No budget limit"
                value={maxMarketValueM}
                onChange={(e) => setMaxMarketValueM(e.target.value === "" ? "" : Number(e.target.value))}
              />
            </div>

            {/* Max Age */}
            <div className="form-group">
              <label className="form-label" htmlFor="scout-max-age">Max Age</label>
              <input
                id="scout-max-age"
                type="number"
                min="16"
                max="40"
                className="form-input"
                placeholder="No age limit"
                value={maxAge}
                onChange={(e) => setMaxAge(e.target.value === "" ? "" : Number(e.target.value))}
              />
            </div>

            {/* Min Minutes */}
            <div className="form-group">
              <label className="form-label" htmlFor="scout-min-mins">Min Minutes</label>
              <input
                id="scout-min-mins"
                type="number"
                min="0"
                step="100"
                className="form-input"
                placeholder="e.g. 600"
                value={minMinutes}
                onChange={(e) => setMinMinutes(e.target.value === "" ? "" : Number(e.target.value))}
              />
            </div>
          </div>

          {/* Bottom Bar: Top N & Find Button */}
          <div
            style={{
              display: "flex",
              alignItems: "center",
              justifyContent: "space-between",
              flexWrap: "wrap",
              gap: "1rem",
              borderTop: "1px solid var(--border-subtle)",
              paddingTop: "1rem",
            }}
          >
            <div style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
              <span style={{ fontSize: "0.8125rem", color: "var(--text-secondary)" }}>
                Max Candidates:
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
                  Finding Replacements...
                </>
              ) : (
                <>
                  <Search size={16} />
                  Find Replacement
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
          {/* Scouting Table Left */}
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
                  Suitable Candidates ({candidates.length} profiles matching filters)
                </span>
                <span style={{ fontSize: "0.75rem", color: "var(--text-muted)" }}>
                  Click row to view detailed per-90 stats
                </span>
              </div>

              <div className="table-container">
                <table className="data-table">
                  <thead>
                    <tr>
                      <th>Player</th>
                      <th>Club</th>
                      <th>Position</th>
                      <th>Compatibility</th>
                      <th
                        onClick={() => handleSort("similarity_score")}
                        style={{ cursor: "pointer", textAlign: "right" }}
                        title="Sort by similarity score"
                      >
                        <div style={{ display: "inline-flex", alignItems: "center", gap: "0.3rem" }}>
                          Similarity
                          <ArrowUpDown size={12} />
                        </div>
                      </th>
                      <th
                        onClick={() => handleSort("age")}
                        style={{ cursor: "pointer" }}
                        title="Sort by age"
                      >
                        <div style={{ display: "inline-flex", alignItems: "center", gap: "0.3rem" }}>
                          Age
                          <ArrowUpDown size={12} />
                        </div>
                      </th>
                      <th
                        onClick={() => handleSort("market_value_eur")}
                        style={{ cursor: "pointer" }}
                        title="Sort by market value"
                      >
                        <div style={{ display: "inline-flex", alignItems: "center", gap: "0.3rem" }}>
                          Market Value
                          <ArrowUpDown size={12} />
                        </div>
                      </th>
                      <th
                        onClick={() => handleSort("minutes")}
                        style={{ cursor: "pointer" }}
                        title="Sort by minutes played"
                      >
                        <div style={{ display: "inline-flex", alignItems: "center", gap: "0.3rem" }}>
                          Minutes
                          <ArrowUpDown size={12} />
                        </div>
                      </th>
                    </tr>
                  </thead>
                  <tbody>
                    {sortedCandidates.map((c) => {
                      const isSelected = activeCandidate?.candidate_player === c.candidate_player;
                      return (
                        <tr
                          key={`${c.candidate_player}-${c.candidate_season}`}
                          onClick={() => setActiveCandidate(c)}
                          className={isSelected ? "selected" : ""}
                        >
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
                          <td>
                            <span
                              className={`badge ${
                                c.position_compatibility === "Same Position"
                                  ? "badge-emerald"
                                  : "badge-blue"
                              }`}
                            >
                              <CheckCircle2 size={11} />
                              {c.position_compatibility}
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
            <Filter size={22} />
          </div>
          <h4 style={{ fontSize: "1rem", color: "var(--text-secondary)", marginBottom: "0.35rem" }}>
            {hasSearched ? "No candidates found" : "No player selected"}
          </h4>
          <p style={{ fontSize: "0.8125rem", color: "var(--text-muted)", maxWidth: "340px", margin: "0 auto" }}>
            {hasSearched
              ? "Try relaxing the scouting filters (e.g. increase max valuation or lower minimum minutes)."
              : "Choose a player to begin scouting and exploring replacement candidates."}
          </p>
        </div>
      )}
    </div>
  );
};
