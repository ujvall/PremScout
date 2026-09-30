import React, { useState } from "react";
import { SimilarPlayersView } from "./SimilarPlayersView";
import { ReplacementSearchView } from "./ReplacementSearchView";
import { Users, UserSearch } from "lucide-react";

export const ScoutingModule: React.FC = () => {
  const [internalTab, setInternalTab] = useState<"similar" | "replacement">("similar");
  const [includeFreeAgents, setIncludeFreeAgents] = useState<boolean>(false);

  return (
    <div className="animate-fade-in">
      {/* Header Banner */}
      <div
        style={{
          display: "flex",
          alignItems: "flex-end",
          justifyContent: "space-between",
          flexWrap: "wrap",
          gap: "1rem",
          marginBottom: "1.25rem",
        }}
      >
        <div>
          <div style={{ display: "flex", alignItems: "center", gap: "0.5rem", marginBottom: "0.25rem" }}>
            <h2 style={{ fontSize: "1.625rem", fontWeight: 800, letterSpacing: "-0.03em", margin: 0 }}>
              Player Scouting &amp; Recruitment
            </h2>
            <span
              style={{
                fontSize: "0.6875rem",
                fontWeight: 700,
                padding: "0.1rem 0.45rem",
                borderRadius: "var(--radius-xs)",
                backgroundColor: "var(--bg-elevated)",
                border: "1px solid var(--border-medium)",
                color: "var(--text-secondary)",
              }}
            >
              V2 Engine
            </span>
          </div>
          <p style={{ color: "var(--text-secondary)", fontSize: "0.875rem", margin: 0 }}>
            Find statistically similar players and replacement options across Europe's top 5 domestic leagues.
          </p>
        </div>

        {/* Internal Tabs — Squircle Segmented Control */}
        <div className="segmented-nav" role="tablist" aria-label="Scouting Sub-tabs">
          <button
            role="tab"
            aria-selected={internalTab === "similar"}
            className={`segmented-nav-item ${internalTab === "similar" ? "active" : ""}`}
            onClick={() => setInternalTab("similar")}
          >
            <Users size={14} />
            <span>Similar Players</span>
          </button>
          <button
            role="tab"
            aria-selected={internalTab === "replacement"}
            className={`segmented-nav-item ${internalTab === "replacement" ? "active" : ""}`}
            onClick={() => setInternalTab("replacement")}
          >
            <UserSearch size={14} />
            <span>Find Replacement</span>
          </button>
        </div>
      </div>

      {/* Scope, Eligibility & Freshness Bar */}
      <div
        className="card"
        style={{
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          flexWrap: "wrap",
          gap: "1rem",
          padding: "0.75rem 1.25rem",
          marginBottom: "1.5rem",
          backgroundColor: "var(--bg-surface)",
          border: "1px solid var(--border-subtle)",
        }}
      >
        <div style={{ display: "flex", flexDirection: "column", gap: "0.25rem" }}>
          <div style={{ display: "flex", alignItems: "center", gap: "0.5rem", flexWrap: "wrap" }}>
            <span style={{ fontSize: "0.875rem", fontWeight: 600, color: "var(--text-primary)" }}>
              European Big 5 League Universe
            </span>
            <span style={{ fontSize: "0.75rem", color: "var(--text-secondary)" }}>
              Premier League • La Liga • Bundesliga • Serie A • Ligue 1
            </span>
          </div>
          <div
            style={{
              display: "flex",
              alignItems: "center",
              gap: "0.75rem",
              flexWrap: "wrap",
              fontSize: "0.75rem",
              color: "var(--text-muted)",
            }}
          >
            <span>
              Current Squads: <strong style={{ color: "var(--text-secondary)" }}>2026/27</strong>
            </span>
            <span>•</span>
            <span>
              Performance Baseline: <strong style={{ color: "var(--text-secondary)" }}>2025/26 Season</strong>
            </span>
            <span>•</span>
            <span>Valuations verified through Transfermarkt</span>
          </div>
        </div>

        {/* Include Free Agents Checkbox */}
        <label
          htmlFor="include-free-agents-toggle"
          style={{
            display: "flex",
            alignItems: "center",
            gap: "0.5rem",
            fontSize: "0.8125rem",
            fontWeight: 500,
            cursor: "pointer",
            padding: "0.35rem 0.75rem",
            borderRadius: "var(--radius-btn)",
            backgroundColor: includeFreeAgents ? "var(--bg-elevated)" : "var(--bg-input)",
            border: `1px solid ${includeFreeAgents ? "var(--border-medium)" : "var(--border-subtle)"}`,
            color: includeFreeAgents ? "var(--text-primary)" : "var(--text-secondary)",
            userSelect: "none",
            transition: "all 0.15s ease",
          }}
        >
          <input
            type="checkbox"
            id="include-free-agents-toggle"
            checked={includeFreeAgents}
            onChange={(e) => setIncludeFreeAgents(e.target.checked)}
            style={{ accentColor: "var(--accent-emerald)", cursor: "pointer", width: "14px", height: "14px" }}
          />
          <span>Include Free Agents</span>
        </label>
      </div>

      {/* Internal Sub-View */}
      {internalTab === "similar" ? (
        <SimilarPlayersView includeFreeAgents={includeFreeAgents} setIncludeFreeAgents={setIncludeFreeAgents} />
      ) : (
        <ReplacementSearchView includeFreeAgents={includeFreeAgents} setIncludeFreeAgents={setIncludeFreeAgents} />
      )}
    </div>
  );
};
