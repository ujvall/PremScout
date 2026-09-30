import React, { useState, useRef } from "react";
import type {
  PlayerSearchItem,
  ReplacementCandidate,
  ScoutingReplacementRequest,
} from "../../types/api";
import { findReplacements } from "../../api";
import { PlayerSearchInput } from "./PlayerSearchInput";
import { PlayerDetailCard } from "./PlayerDetailCard";
import { TableSkeleton } from "../common/Skeleton";
import {
  Filter,
  Search,
  AlertCircle,
  RotateCcw,
  ArrowUpDown,
  ArrowDown,
} from "lucide-react";

type SortField = "similarity_score" | "age" | "market_value_eur" | "minutes";
type SortOrder = "asc" | "desc";

interface ReplacementSearchViewProps {
  includeFreeAgents?: boolean;
  setIncludeFreeAgents?: (val: boolean) => void;
}

export const ReplacementSearchView: React.FC<ReplacementSearchViewProps> = ({
  includeFreeAgents = false,
}) => {
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
  const detailRef = useRef<HTMLDivElement>(null);

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
      include_free_agents: includeFreeAgents,
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

  const handleSelectCandidate = (c: ReplacementCandidate) => {
    setActiveCandidate(c);
    setTimeout(() => {
      detailRef.current?.scrollIntoView({ behavior: "smooth", block: "nearest" });
    }, 50);
  };

  const sortedCandidates = [...candidates].sort((a, b) => {
    const aVal = a[sortField] ?? (sortOrder === "asc" ? Infinity : -Infinity);
    const bVal = b[sortField] ?? (sortOrder === "asc" ? Infinity : -Infinity);
    if (sortOrder === "asc") {
      return aVal > bVal ? 1 : -1;
    } else {
      return aVal < bVal ? 1 : -1;
    }
  });

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "1.5rem" }}>
      {/* 1. Search & Filters Card */}
      <div className="card">
        <div className="card-header">
          <div>
            <h4 className="card-title">
              <Filter size={17} color="var(--accent-blue)" />
              Target Player &amp; Recruitment Filters
            </h4>
            <p className="card-subtitle">
              Filter replacements by position constraint, transfer budget cap, age limit, and minimum playing time
            </p>
          </div>
          <button
            type="button"
            onClick={handleResetFilters}
            className="btn btn-ghost btn-sm"
            style={{ gap: "0.3rem" }}
          >
            <RotateCcw size={13} />
            <span>Reset Filters</span>
          </button>
        </div>

        <div style={{ display: "flex", flexDirection: "column", gap: "1.25rem" }}>
          {/* Target Player Autocomplete */}
          <PlayerSearchInput
            selectedPlayer={selectedPlayer}
            onSelectPlayer={setSelectedPlayer}
            placeholder="Type player to replace (e.g. Harry Kane, Granit Xhaka, Declan Rice, Mason Mount)..."
            includeFreeAgents={includeFreeAgents}
          />

          {/* Filter Controls Grid */}
          <div className="grid-4" style={{ paddingTop: "0.25rem" }}>
            {/* Same Position Only */}
            <div className="form-group">
              <label className="form-label" htmlFor="scout-same-pos">Position Constraint</label>
              <div
                style={{
                  display: "flex",
                  alignItems: "center",
                  gap: "0.5rem",
                  padding: "0 0.875rem",
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
                  style={{ accentColor: "var(--accent-tea-green)", cursor: "pointer" }}
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
                type="number"
                id="scout-max-val"
                className="form-input"
                placeholder="No limit"
                min={0}
                step={5}
                value={maxMarketValueM}
                onChange={(e) =>
                  setMaxMarketValueM(e.target.value === "" ? "" : Number(e.target.value))
                }
              />
            </div>

            {/* Max Age */}
            <div className="form-group">
              <label className="form-label" htmlFor="scout-max-age">Age Ceiling</label>
              <input
                type="number"
                id="scout-max-age"
                className="form-input"
                placeholder="No limit (e.g. 26)"
                min={16}
                max={42}
                value={maxAge}
                onChange={(e) =>
                  setMaxAge(e.target.value === "" ? "" : Number(e.target.value))
                }
              />
            </div>

            {/* Min Minutes */}
            <div className="form-group">
              <label className="form-label" htmlFor="scout-min-min">Min Scouted Minutes</label>
              <input
                type="number"
                id="scout-min-min"
                className="form-input"
                placeholder="e.g. 600"
                min={0}
                step={100}
                value={minMinutes}
                onChange={(e) =>
                  setMinMinutes(e.target.value === "" ? "" : Number(e.target.value))
                }
              />
            </div>
          </div>

          {/* Sub-bar: Top N and Search Submit */}
          <div
            style={{
              display: "flex",
              alignItems: "center",
              justifyContent: "space-between",
              flexWrap: "wrap",
              gap: "1rem",
              paddingTop: "0.25rem",
              borderTop: "1px solid var(--border-subtle)",
            }}
          >
            <div style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
              <span style={{ fontSize: "0.8125rem", color: "var(--text-secondary)" }}>
                Max Candidate Count:
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
                  <span>Finding Replacements...</span>
                </>
              ) : (
                <>
                  <Search size={15} />
                  <span>Find Replacements</span>
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
                  Suitable Replacement Candidates
                </span>
                <span
                  style={{
                    fontSize: "0.8125rem",
                    color: "var(--text-secondary)",
                    marginLeft: "0.5rem",
                  }}
                >
                  ({candidates.length} profiles matching recruitment constraints)
                </span>
              </div>
              <span style={{ fontSize: "0.75rem", color: "var(--text-muted)", display: "flex", alignItems: "center", gap: "0.3rem" }}>
                <span>Select row to inspect radar below</span>
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
                    <th
                      onClick={() => handleSort("similarity_score")}
                      style={{ cursor: "pointer", textAlign: "right" }}
                      title="Sort by similarity score"
                    >
                      <div style={{ display: "inline-flex", alignItems: "center", gap: "0.3rem" }}>
                        Similarity
                        <ArrowUpDown size={11} />
                      </div>
                    </th>
                    <th>Compatibility</th>
                    <th
                      onClick={() => handleSort("age")}
                      style={{ cursor: "pointer" }}
                      title="Sort by age"
                    >
                      <div style={{ display: "inline-flex", alignItems: "center", gap: "0.3rem" }}>
                        Age
                        <ArrowUpDown size={11} />
                      </div>
                    </th>
                    <th
                      onClick={() => handleSort("market_value_eur")}
                      style={{ cursor: "pointer" }}
                      title="Sort by market value"
                    >
                      <div style={{ display: "inline-flex", alignItems: "center", gap: "0.3rem" }}>
                        Market Value
                        <ArrowUpDown size={11} />
                      </div>
                    </th>
                    <th
                      onClick={() => handleSort("minutes")}
                      style={{ cursor: "pointer", textAlign: "right" }}
                      title="Sort by minutes played"
                    >
                      <div style={{ display: "inline-flex", alignItems: "center", gap: "0.3rem" }}>
                        Minutes
                        <ArrowUpDown size={11} />
                      </div>
                    </th>
                  </tr>
                </thead>
                <tbody>
                  {sortedCandidates.map((c, idx) => {
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
                  Selected Replacement Profile
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
            <Filter size={20} />
          </div>
          <h4 style={{ fontSize: "0.9375rem", color: "var(--text-secondary)", marginBottom: "0.35rem" }}>
            {hasSearched ? "No replacement candidates match filters" : "No target player selected"}
          </h4>
          <p style={{ fontSize: "0.8125rem", color: "var(--text-muted)", maxWidth: "380px", margin: "0 auto" }}>
            {hasSearched
              ? "Try adjusting budget ceiling, minutes threshold, or unchecking strict position match."
              : "Select a player to replace and apply transfer recruitment constraints to explore viable market alternatives."}
          </p>
        </div>
      )}
    </div>
  );
};
