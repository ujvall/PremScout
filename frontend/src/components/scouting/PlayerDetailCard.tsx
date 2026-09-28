import React from "react";
import type { ReplacementCandidate } from "../../types/api";
import { X, User, CheckCircle2, Activity } from "lucide-react";

interface PlayerDetailCardProps {
  candidate: ReplacementCandidate | null;
  onClose: () => void;
}

export const PlayerDetailCard: React.FC<PlayerDetailCardProps> = ({
  candidate,
  onClose,
}) => {
  if (!candidate) return null;

  const stats = candidate.key_stats;

  const metrics = [
    { label: "Goals / 90", value: stats.goals_per90?.toFixed(2) ?? "—", unit: "rate" },
    { label: "Assists / 90", value: stats.assists_per90?.toFixed(2) ?? "—", unit: "rate" },
    { label: "npxG / 90", value: stats.npxG_per90?.toFixed(2) ?? "—", unit: "rate" },
    { label: "xAG / 90", value: stats.xAG_per90?.toFixed(2) ?? "—", unit: "rate" },
    { label: "Key Passes / 90", value: stats.key_passes_per90?.toFixed(2) ?? "—", unit: "rate" },
    {
      label: "Pass Completion",
      value: stats.pass_completion_pct != null ? `${stats.pass_completion_pct.toFixed(1)}%` : "—",
      unit: "pct",
    },
    { label: "Prog. Passes / 90", value: stats.progressive_passes_per90?.toFixed(2) ?? "—", unit: "rate" },
    { label: "Prog. Carries / 90", value: stats.progressive_carries_per90?.toFixed(2) ?? "—", unit: "rate" },
    { label: "Tackles Won / 90", value: stats.tackles_won_per90?.toFixed(2) ?? "—", unit: "rate" },
    { label: "Interceptions / 90", value: stats.interceptions_per90?.toFixed(2) ?? "—", unit: "rate" },
  ];

  return (
    <div className="card animate-fade-in" style={{ padding: "1.25rem", border: "1px solid var(--border-medium)" }}>
      {/* Header */}
      <div
        style={{
          display: "flex",
          alignItems: "flex-start",
          justifyContent: "space-between",
          paddingBottom: "1rem",
          borderBottom: "1px solid var(--border-subtle)",
          marginBottom: "1rem",
        }}
      >
        <div style={{ display: "flex", alignItems: "center", gap: "0.75rem" }}>
          <div
            style={{
              width: "40px",
              height: "40px",
              borderRadius: "50%",
              backgroundColor: "var(--bg-elevated)",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              color: "var(--accent-emerald)",
            }}
          >
            <User size={20} />
          </div>
          <div>
            <div style={{ display: "flex", alignItems: "center", gap: "0.5rem", flexWrap: "wrap" }}>
              <h4 style={{ fontSize: "1.1rem", fontWeight: 700, margin: 0, color: "#ffffff" }}>
                {candidate.candidate_player}
              </h4>
              <span className="badge badge-emerald">
                {candidate.candidate_position || candidate.candidate_raw_position}
              </span>
              <span
                style={{
                  fontSize: "0.6875rem",
                  color: "#38bdf8",
                  backgroundColor: "rgba(56, 189, 248, 0.12)",
                  border: "1px solid rgba(56, 189, 248, 0.3)",
                  borderRadius: "9999px",
                  padding: "0.1rem 0.45rem",
                  fontWeight: 600,
                }}
              >
                Data as of: {candidate.data_as_of || "2026-09-20"}
              </span>
            </div>
            <p style={{ fontSize: "0.75rem", color: "var(--text-muted)", margin: "0.2rem 0 0" }}>
              Current Club: <strong style={{ color: "var(--text-secondary)" }}>{candidate.candidate_club}</strong> |
              Latest Season: {candidate.candidate_season}
              {candidate.age ? ` | Current Age: ${candidate.age}` : ""}
              {candidate.minutes ? ` | Minutes: ${candidate.minutes}` : ""}
            </p>
          </div>
        </div>

        <button
          onClick={onClose}
          className="btn btn-ghost"
          style={{ padding: "0.3rem" }}
          aria-label="Close detail panel"
        >
          <X size={16} />
        </button>
      </div>

      {/* Comparison & Similarity Summary */}
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "1fr 1fr",
          gap: "0.75rem",
          padding: "0.875rem",
          borderRadius: "var(--radius-md)",
          backgroundColor: "var(--bg-surface)",
          border: "1px solid var(--border-subtle)",
          marginBottom: "1.25rem",
        }}
      >
        <div>
          <span style={{ fontSize: "0.6875rem", color: "var(--text-muted)", textTransform: "uppercase" }}>
            Similarity to {candidate.target_player}
          </span>
          <div style={{ display: "flex", alignItems: "baseline", gap: "0.25rem", marginTop: "0.2rem" }}>
            <span
              style={{
                fontSize: "1.35rem",
                fontWeight: 800,
                color: "var(--accent-emerald-light)",
                fontFamily: "var(--font-mono)",
              }}
            >
              {candidate.similarity_score.toFixed(1)}
            </span>
            <span style={{ fontSize: "0.75rem", color: "var(--text-muted)" }}>/ 100</span>
          </div>
        </div>

        <div>
          <span style={{ fontSize: "0.6875rem", color: "var(--text-muted)", textTransform: "uppercase" }}>
            Compatibility
          </span>
          <div style={{ marginTop: "0.35rem" }}>
            <span
              className={`badge ${
                candidate.position_compatibility === "Same Position"
                  ? "badge-emerald"
                  : "badge-blue"
              }`}
            >
              <CheckCircle2 size={11} />
              {candidate.position_compatibility}
            </span>
          </div>
        </div>
      </div>

      {/* Key Per-90 Statistics Grid */}
      <div>
        <div style={{ display: "flex", alignItems: "center", gap: "0.4rem", marginBottom: "0.75rem" }}>
          <Activity size={14} color="var(--accent-emerald)" />
          <span style={{ fontSize: "0.8125rem", fontWeight: 600, color: "var(--text-secondary)" }}>
            Verified Performance Metrics (FBref / Transfermarkt)
          </span>
        </div>

        <div className="grid-2">
          {metrics.map((m) => (
            <div
              key={m.label}
              style={{
                padding: "0.5rem 0.75rem",
                borderRadius: "var(--radius-sm)",
                backgroundColor: "var(--bg-input)",
                border: "1px solid var(--border-subtle)",
                display: "flex",
                alignItems: "center",
                justifyContent: "space-between",
              }}
            >
              <span style={{ fontSize: "0.75rem", color: "var(--text-muted)" }}>{m.label}</span>
              <span
                style={{
                  fontSize: "0.875rem",
                  fontWeight: 700,
                  fontFamily: "var(--font-mono)",
                  color: "var(--text-primary)",
                }}
              >
                {m.value}
              </span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};
