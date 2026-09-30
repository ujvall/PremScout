import React from "react";
import type { MatchPredictResponse } from "../../types/api";
import { Swords, Award, Clock } from "lucide-react";

interface MatchResultProps {
  result: MatchPredictResponse | null;
  loading: boolean;
}

export const MatchResult: React.FC<MatchResultProps> = ({ result, loading }) => {
  if (loading) {
    return (
      <div className="card" style={{ padding: "2.5rem 1.5rem", textAlign: "center" }}>
        <div className="animate-spin" style={{ display: "inline-block", marginBottom: "0.75rem" }}>
          <Swords size={28} color="var(--accent-blue)" />
        </div>
        <h4 style={{ fontSize: "1.125rem", color: "var(--text-primary)", marginBottom: "0.25rem" }}>
          Simulating Match Outcome...
        </h4>
        <p style={{ fontSize: "0.8125rem", color: "var(--text-muted)" }}>
          Calculating 3-way probabilistic distribution across pre-match feature vectors
        </p>
      </div>
    );
  }

  if (!result) {
    return (
      <div
        className="card"
        style={{
          padding: "2.5rem 1.5rem",
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
          <Swords size={22} />
        </div>
        <h4 style={{ fontSize: "1rem", color: "var(--text-secondary)", marginBottom: "0.35rem" }}>
          No prediction yet
        </h4>
        <p style={{ fontSize: "0.8125rem", color: "var(--text-muted)", maxWidth: "280px", margin: "0 auto" }}>
          Enter match information and run the model to view probabilistic forecasts.
        </p>
      </div>
    );
  }

  const { home_team, away_team, predicted_outcome_label, probabilities } = result;

  // Convert decimal probabilities to percentages (e.g. 0.598 -> 59.8)
  const homePct = (probabilities.home_win * 100);
  const drawPct = (probabilities.draw * 100);
  const awayPct = (probabilities.away_win * 100);

  const isHomeWinner = result.predicted_outcome === "H";
  const isDrawWinner = result.predicted_outcome === "D";
  const isAwayWinner = result.predicted_outcome === "A";

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "1.25rem" }} className="animate-fade-in">
      {/* Large Match Fixture Card */}
      <div className="card" style={{ padding: "1.5rem", position: "relative" }}>
        <div
          style={{
            display: "flex",
            alignItems: "center",
            justifyContent: "space-between",
            marginBottom: "1rem",
          }}
        >
          <span className="badge badge-blue">
            <Clock size={12} />
            Pre-match model
          </span>
          <span style={{ fontSize: "0.75rem", color: "var(--text-muted)" }}>
            Algorithm: {result.model_type}
          </span>
        </div>

        {/* Fixture Banner */}
        <div
          style={{
            display: "flex",
            alignItems: "center",
            justifyContent: "space-around",
            padding: "1rem 0",
            borderBottom: "1px solid var(--border-subtle)",
          }}
        >
          <div style={{ textAlign: "center", flex: 1 }}>
            <h3
              style={{
                fontSize: "1.25rem",
                fontWeight: 800,
                color: isHomeWinner ? "var(--accent-tea-green)" : "var(--text-primary)",
                margin: 0,
              }}
            >
              {home_team || "Home Team"}
            </h3>
            <span style={{ fontSize: "0.75rem", color: "var(--text-muted)", textTransform: "uppercase" }}>
              Home
            </span>
          </div>

          <div
            style={{
              padding: "0.25rem 0.625rem",
              borderRadius: "var(--radius-xs)",
              backgroundColor: "var(--bg-elevated)",
              border: "1px solid var(--border-subtle)",
              fontSize: "0.75rem",
              fontWeight: 700,
              color: "var(--text-muted)",
              letterSpacing: "0.05em",
            }}
          >
            VS
          </div>

          <div style={{ textAlign: "center", flex: 1 }}>
            <h3
              style={{
                fontSize: "1.25rem",
                fontWeight: 800,
                color: isAwayWinner ? "var(--accent-blue)" : "var(--text-primary)",
                margin: 0,
              }}
            >
              {away_team || "Away Team"}
            </h3>
            <span style={{ fontSize: "0.75rem", color: "var(--text-muted)", textTransform: "uppercase" }}>
              Away
            </span>
          </div>
        </div>

        {/* Segmented Horizontal Probability Bar (Sums to 100%) */}
        <div style={{ marginTop: "1.25rem" }}>
          <div
            style={{
              display: "flex",
              justifyContent: "space-between",
              fontSize: "0.75rem",
              color: "var(--text-muted)",
              marginBottom: "0.35rem",
            }}
          >
            <span>Home: {homePct.toFixed(1)}%</span>
            <span>Draw: {drawPct.toFixed(1)}%</span>
            <span>Away: {awayPct.toFixed(1)}%</span>
          </div>

          <div className="probability-bar-track" aria-label="Outcome probabilities bar">
            <div
              className="prob-segment-home"
              style={{ width: `${homePct}%` }}
              title={`Home Win: ${homePct.toFixed(1)}%`}
            />
            <div
              className="prob-segment-draw"
              style={{ width: `${drawPct}%` }}
              title={`Draw: ${drawPct.toFixed(1)}%`}
            />
            <div
              className="prob-segment-away"
              style={{ width: `${awayPct}%` }}
              title={`Away Win: ${awayPct.toFixed(1)}%`}
            />
          </div>
        </div>

        {/* Predicted Outcome Highlight */}
        <div
          style={{
            marginTop: "1.25rem",
            padding: "0.875rem 1rem",
            borderRadius: "var(--radius-md)",
            backgroundColor: "var(--bg-elevated)",
            display: "flex",
            alignItems: "center",
            justifyContent: "space-between",
          }}
        >
          <div style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
            <Award size={18} color="var(--accent-tea-green)" />
            <span style={{ fontSize: "0.8125rem", color: "var(--text-muted)" }}>Predicted Outcome</span>
          </div>
          <span
            style={{
              fontWeight: 800,
              fontSize: "1rem",
              color: isHomeWinner
                ? "var(--accent-tea-green)"
                : isDrawWinner
                ? "var(--accent-amber)"
                : "var(--accent-blue)",
            }}
          >
            {predicted_outcome_label}
          </span>
        </div>
      </div>

      {/* 3 Outcome Probability Cards */}
      <div className="grid-3">
        {/* Home Win */}
        <div
          className="card"
          style={{
            padding: "1rem",
            textAlign: "center",
            borderColor: isHomeWinner ? "rgba(193, 237, 204, 0.4)" : "var(--border-subtle)",
            backgroundColor: isHomeWinner ? "rgba(193, 237, 204, 0.05)" : "var(--bg-card)",
          }}
        >
          <span
            style={{
              fontSize: "0.6875rem",
              textTransform: "uppercase",
              fontWeight: 700,
              letterSpacing: "0.05em",
              color: "var(--accent-tea-green)",
            }}
          >
            HOME WIN
          </span>
          <p
            style={{
              fontSize: "1.75rem",
              fontWeight: 800,
              fontFamily: "var(--font-mono)",
              color: "#ffffff",
              margin: "0.35rem 0 0",
            }}
          >
            {homePct.toFixed(1)}%
          </p>
        </div>

        {/* Draw */}
        <div
          className="card"
          style={{
            padding: "1rem",
            textAlign: "center",
            borderColor: isDrawWinner ? "rgba(245, 158, 11, 0.4)" : "var(--border-subtle)",
            backgroundColor: isDrawWinner ? "rgba(245, 158, 11, 0.05)" : "var(--bg-card)",
          }}
        >
          <span
            style={{
              fontSize: "0.6875rem",
              textTransform: "uppercase",
              fontWeight: 700,
              letterSpacing: "0.05em",
              color: "var(--accent-amber)",
            }}
          >
            DRAW
          </span>
          <p
            style={{
              fontSize: "1.75rem",
              fontWeight: 800,
              fontFamily: "var(--font-mono)",
              color: "#ffffff",
              margin: "0.35rem 0 0",
            }}
          >
            {drawPct.toFixed(1)}%
          </p>
        </div>

        {/* Away Win */}
        <div
          className="card"
          style={{
            padding: "1rem",
            textAlign: "center",
            borderColor: isAwayWinner ? "rgba(56, 189, 248, 0.4)" : "var(--border-subtle)",
            backgroundColor: isAwayWinner ? "rgba(56, 189, 248, 0.05)" : "var(--bg-card)",
          }}
        >
          <span
            style={{
              fontSize: "0.6875rem",
              textTransform: "uppercase",
              fontWeight: 700,
              letterSpacing: "0.05em",
              color: "var(--accent-blue)",
            }}
          >
            AWAY WIN
          </span>
          <p
            style={{
              fontSize: "1.75rem",
              fontWeight: 800,
              fontFamily: "var(--font-mono)",
              color: "#ffffff",
              margin: "0.35rem 0 0",
            }}
          >
            {awayPct.toFixed(1)}%
          </p>
        </div>
      </div>
    </div>
  );
};
