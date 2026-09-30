import React from "react";
import type { ReplacementCandidate } from "../../types/api";
import { X, Activity } from "lucide-react";

interface PlayerDetailCardProps {
  candidate: ReplacementCandidate | null;
  onClose?: () => void;
}

export const PlayerDetailCard: React.FC<PlayerDetailCardProps> = ({
  candidate,
  onClose,
}) => {
  if (!candidate) return null;

  const stats = candidate.key_stats;

  const metrics = [
    { label: "Goals / 90", value: stats.goals_per90?.toFixed(2) ?? "—" },
    { label: "Assists / 90", value: stats.assists_per90?.toFixed(2) ?? "—" },
    { label: "npxG / 90", value: stats.npxG_per90?.toFixed(2) ?? "—" },
    { label: "xAG / 90", value: stats.xAG_per90?.toFixed(2) ?? "—" },
    { label: "Key Passes / 90", value: stats.key_passes_per90?.toFixed(2) ?? "—" },
    {
      label: "Pass Completion",
      value: stats.pass_completion_pct != null ? `${stats.pass_completion_pct.toFixed(1)}%` : "—",
    },
    { label: "Prog. Passes / 90", value: stats.progressive_passes_per90?.toFixed(2) ?? "—" },
    { label: "Prog. Carries / 90", value: stats.progressive_carries_per90?.toFixed(2) ?? "—" },
    { label: "Tackles Won / 90", value: stats.tackles_won_per90?.toFixed(2) ?? "—" },
    { label: "Interceptions / 90", value: stats.interceptions_per90?.toFixed(2) ?? "—" },
  ];

  const currentClub = candidate.current_club || candidate.candidate_club;
  const currentLeague = candidate.is_free_agent
    ? "Free Agent"
    : candidate.current_league || (candidate.is_current_big5 ? "Big 5 Active" : "Non-Big 5");
  const currentSeason = candidate.current_season || "2026/27";

  const perfClub = candidate.performance_club || candidate.candidate_club;
  const perfLeague = candidate.performance_league || "—";
  const perfSeason = candidate.performance_season || "2025/26";

  const marketValueText =
    candidate.market_value_eur != null && !isNaN(candidate.market_value_eur) && candidate.market_value_eur > 0
      ? `€${(candidate.market_value_eur / 1e6).toFixed(1)}M`
      : "Not available";

  const recentMatches = candidate.recent_form?.matches || [];

  return (
    <div className="card animate-fade-in" style={{ padding: "1.5rem" }}>
      {/* 1. Header & Primary Identification */}
      <div
        style={{
          display: "flex",
          alignItems: "flex-start",
          justifyContent: "space-between",
          borderBottom: "1px solid var(--border-subtle)",
          paddingBottom: "1.25rem",
          marginBottom: "1.25rem",
          gap: "1rem",
        }}
      >
        <div style={{ flex: 1 }}>
          <div style={{ display: "flex", alignItems: "baseline", gap: "0.75rem", flexWrap: "wrap" }}>
            <h3
              style={{
                fontSize: "1.5rem",
                fontWeight: 800,
                color: "var(--text-primary)",
                letterSpacing: "-0.03em",
                margin: 0,
              }}
            >
              {candidate.candidate_player}
            </h3>
            <span className="tag-position">
              {candidate.candidate_position || candidate.candidate_raw_position}
            </span>
            {candidate.age && (
              <span style={{ fontSize: "0.875rem", color: "var(--text-secondary)" }}>
                Age {candidate.age}
              </span>
            )}
          </div>

          <div
            style={{
              display: "flex",
              alignItems: "center",
              gap: "0.5rem",
              marginTop: "0.35rem",
              fontSize: "0.875rem",
              color: "var(--text-secondary)",
              flexWrap: "wrap",
            }}
          >
            <span style={{ fontWeight: 600, color: "var(--text-primary)" }}>{currentClub}</span>
            <span style={{ color: "var(--border-medium)" }}>•</span>
            <span>{currentLeague}</span>
            {candidate.is_free_agent && (
              <span className="badge badge-amber" style={{ fontSize: "0.6875rem" }}>
                Free Agent
              </span>
            )}
          </div>
        </div>

        {onClose && (
          <button
            onClick={onClose}
            className="btn btn-ghost btn-sm"
            style={{ padding: "0.35rem", color: "var(--text-muted)" }}
            aria-label="Close player detail"
          >
            <X size={16} />
          </button>
        )}
      </div>

      {/* 2. Key Metrics Row: Similarity, Compatibility, Valuation, Minutes, Form */}
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(auto-fit, minmax(140px, 1fr))",
          gap: "0.75rem",
          marginBottom: "1.5rem",
        }}
      >
        {/* Similarity */}
        <div
          style={{
            padding: "0.75rem",
            borderRadius: "var(--radius-md)",
            backgroundColor: "var(--bg-surface)",
            border: "1px solid var(--border-subtle)",
          }}
        >
          <div style={{ fontSize: "0.6875rem", color: "var(--text-muted)", textTransform: "uppercase" }}>
            Similarity
          </div>
          <div style={{ display: "flex", alignItems: "baseline", gap: "0.25rem", marginTop: "0.25rem" }}>
            <span
              style={{
                fontSize: "1.375rem",
                fontWeight: 800,
                color: "var(--accent-tea-green)",
                fontFamily: "var(--font-mono)",
              }}
            >
              {candidate.similarity_score.toFixed(1)}
            </span>
            <span style={{ fontSize: "0.75rem", color: "var(--text-muted)" }}>/ 100</span>
          </div>
        </div>

        {/* Compatibility */}
        <div
          style={{
            padding: "0.75rem",
            borderRadius: "var(--radius-md)",
            backgroundColor: "var(--bg-surface)",
            border: "1px solid var(--border-subtle)",
          }}
        >
          <div style={{ fontSize: "0.6875rem", color: "var(--text-muted)", textTransform: "uppercase" }}>
            Compatibility
          </div>
          <div style={{ marginTop: "0.35rem" }}>
            <span
              className={`badge ${
                candidate.position_compatibility === "Same Position"
                  ? "badge-emerald"
                  : "badge-blue"
              }`}
            >
              {candidate.position_compatibility}
            </span>
          </div>
        </div>

        {/* Market Value */}
        <div
          style={{
            padding: "0.75rem",
            borderRadius: "var(--radius-md)",
            backgroundColor: "var(--bg-surface)",
            border: "1px solid var(--border-subtle)",
          }}
        >
          <div style={{ fontSize: "0.6875rem", color: "var(--text-muted)", textTransform: "uppercase" }}>
            Market Value
          </div>
          <div
            style={{
              fontSize: "1.0625rem",
              fontWeight: 700,
              color: marketValueText === "Not available" ? "var(--text-muted)" : "var(--text-primary)",
              fontFamily: marketValueText === "Not available" ? "inherit" : "var(--font-mono)",
              marginTop: "0.35rem",
            }}
          >
            {marketValueText}
          </div>
        </div>

        {/* Scouted Minutes */}
        <div
          style={{
            padding: "0.75rem",
            borderRadius: "var(--radius-md)",
            backgroundColor: "var(--bg-surface)",
            border: "1px solid var(--border-subtle)",
          }}
        >
          <div style={{ fontSize: "0.6875rem", color: "var(--text-muted)", textTransform: "uppercase" }}>
            Scouted Minutes
          </div>
          <div
            style={{
              fontSize: "1.0625rem",
              fontWeight: 700,
              color: "var(--text-primary)",
              fontFamily: "var(--font-mono)",
              marginTop: "0.35rem",
            }}
          >
            {candidate.minutes.toLocaleString()}
          </div>
        </div>

        {/* Recent Form */}
        <div
          style={{
            padding: "0.75rem",
            borderRadius: "var(--radius-md)",
            backgroundColor: "var(--bg-surface)",
            border: "1px solid var(--border-subtle)",
            minWidth: "150px",
          }}
        >
          <div style={{ fontSize: "0.6875rem", color: "var(--text-muted)", textTransform: "uppercase" }}>
            Recent Form (2025/26)
          </div>
          <div style={{ display: "flex", alignItems: "center", gap: "0.35rem", marginTop: "0.45rem" }}>
            {recentMatches.length > 0 ? (
              recentMatches.map((m, idx) => {
                const isWin = m.result === "W";
                const isDraw = m.result === "D";
                const bg = isWin
                  ? "rgba(193, 237, 204, 0.16)"
                  : isDraw
                  ? "rgba(176, 192, 188, 0.15)"
                  : "rgba(244, 63, 94, 0.18)";
                const color = isWin
                  ? "var(--accent-tea-green)"
                  : isDraw
                  ? "var(--text-secondary)"
                  : "#fb7185";
                const border = isWin
                  ? "rgba(193, 237, 204, 0.35)"
                  : isDraw
                  ? "rgba(176, 192, 188, 0.3)"
                  : "rgba(244, 63, 94, 0.35)";

                const tooltip = `${m.date} • ${m.result} • ${m.minutes} mins${
                  m.goals ? ` • ${m.goals}G` : ""
                }${m.assists ? ` • ${m.assists}A` : ""}`;

                return (
                  <span
                    key={`${m.date}-${idx}`}
                    title={tooltip}
                    style={{
                      width: "22px",
                      height: "22px",
                      borderRadius: "var(--radius-xs)",
                      backgroundColor: bg,
                      color: color,
                      border: `1px solid ${border}`,
                      fontSize: "0.6875rem",
                      fontWeight: 700,
                      display: "inline-flex",
                      alignItems: "center",
                      justifyContent: "center",
                      fontFamily: "var(--font-mono)",
                      cursor: "default",
                    }}
                  >
                    {m.result}
                  </span>
                );
              })
            ) : (
              <span style={{ fontSize: "0.75rem", color: "var(--text-muted)" }}>
                Form data not available
              </span>
            )}
          </div>
        </div>
      </div>

      {/* 3. Registration vs Historical Performance Separation */}
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(auto-fit, minmax(240px, 1fr))",
          gap: "1rem",
          padding: "1rem",
          borderRadius: "var(--radius-md)",
          backgroundColor: "var(--bg-surface)",
          border: "1px solid var(--border-subtle)",
          marginBottom: "1.5rem",
          fontSize: "0.8125rem",
        }}
      >
        <div>
          <div
            style={{
              fontSize: "0.6875rem",
              fontWeight: 600,
              textTransform: "uppercase",
              letterSpacing: "0.04em",
              color: "var(--accent-tea-green)",
              marginBottom: "0.4rem",
            }}
          >
            Current Registration ({currentSeason})
          </div>
          <div style={{ color: "var(--text-secondary)", lineHeight: 1.5 }}>
            Club: <strong style={{ color: "var(--text-primary)" }}>{currentClub}</strong>
            <br />
            League: <strong style={{ color: "var(--text-primary)" }}>{currentLeague}</strong>
          </div>
        </div>

        <div>
          <div
            style={{
              fontSize: "0.6875rem",
              fontWeight: 600,
              textTransform: "uppercase",
              letterSpacing: "0.04em",
              color: "var(--accent-blue)",
              marginBottom: "0.4rem",
            }}
          >
            Scouted Performance ({perfSeason})
          </div>
          <div style={{ color: "var(--text-secondary)", lineHeight: 1.5 }}>
            Club: <strong style={{ color: "var(--text-primary)" }}>{perfClub}</strong>
            <br />
            League: <strong style={{ color: "var(--text-primary)" }}>{perfLeague}</strong>
          </div>
        </div>
      </div>

      {/* 4. Performance Profile Metrics */}
      <div>
        <div
          style={{
            display: "flex",
            alignItems: "center",
            gap: "0.45rem",
            marginBottom: "0.875rem",
          }}
        >
          <Activity size={15} color="var(--accent-tea-green)" />
          <span style={{ fontSize: "0.875rem", fontWeight: 600, color: "var(--text-primary)" }}>
            Normalized Performance Profile (Per 90 Metrics)
          </span>
        </div>

        <div
          style={{
            display: "grid",
            gridTemplateColumns: "repeat(auto-fit, minmax(180px, 1fr))",
            gap: "0.5rem",
          }}
        >
          {metrics.map((m) => (
            <div
              key={m.label}
              style={{
                padding: "0.625rem 0.875rem",
                borderRadius: "var(--radius-sm)",
                backgroundColor: "var(--bg-input)",
                border: "1px solid var(--border-subtle)",
                display: "flex",
                alignItems: "center",
                justifyContent: "space-between",
              }}
            >
              <span style={{ fontSize: "0.75rem", color: "var(--text-secondary)" }}>{m.label}</span>
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
