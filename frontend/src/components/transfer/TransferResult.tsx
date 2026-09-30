import React from "react";
import type { TransferPredictResponse } from "../../types/api";
import { TrendingUp, Info, Cpu, CheckCircle } from "lucide-react";

interface TransferResultProps {
  result: TransferPredictResponse | null;
  loading: boolean;
}

export const TransferResult: React.FC<TransferResultProps> = ({
  result,
  loading,
}) => {
  if (loading) {
    return (
      <div className="card" style={{ padding: "2rem", textAlign: "center" }}>
        <div className="animate-spin" style={{ display: "inline-block", marginBottom: "0.75rem" }}>
          <TrendingUp size={28} color="var(--accent-tea-green)" />
        </div>
        <h4 style={{ fontSize: "1.125rem", color: "var(--text-primary)", marginBottom: "0.25rem" }}>
          Evaluating Valuation Model...
        </h4>
        <p style={{ fontSize: "0.8125rem", color: "var(--text-muted)" }}>
          Processing player metrics through gradient boosting regression pipeline
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
          <TrendingUp size={22} />
        </div>
        <h4 style={{ fontSize: "1rem", color: "var(--text-secondary)", marginBottom: "0.35rem" }}>
          No valuation generated yet
        </h4>
        <p style={{ fontSize: "0.8125rem", color: "var(--text-muted)", maxWidth: "280px", margin: "0 auto" }}>
          Fill in the player context and performance stats, or select a preset, then click Predict.
        </p>
      </div>
    );
  }

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "1.25rem" }} className="animate-fade-in">
      {/* Prominent Value Card */}
      <div className="valuation-display">
        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between" }}>
          <span
            style={{
              fontSize: "0.75rem",
              fontWeight: 700,
              textTransform: "uppercase",
              letterSpacing: "0.08em",
              color: "var(--accent-tea-green)",
            }}
          >
            Estimated Market Value
          </span>
          <span className="badge badge-emerald">
            <CheckCircle size={12} />
            Computed
          </span>
        </div>

        <div className="valuation-amount">
          {result.predicted_market_value_formatted}
        </div>

        <p style={{ fontSize: "0.875rem", color: "var(--text-secondary)", margin: 0 }}>
          Player: <strong style={{ color: "var(--text-primary)" }}>{result.player_name}</strong>
        </p>

        {/* Model and transformation metadata */}
        <div
          style={{
            display: "grid",
            gridTemplateColumns: "1fr 1fr",
            gap: "0.75rem",
            marginTop: "1.25rem",
            paddingTop: "1rem",
            borderTop: "1px solid rgba(255, 255, 255, 0.08)",
          }}
        >
          <div>
            <span style={{ fontSize: "0.75rem", color: "var(--text-muted)" }}>Model</span>
            <div style={{ display: "flex", alignItems: "center", gap: "0.35rem", marginTop: "0.2rem" }}>
              <Cpu size={14} color="var(--accent-tea-green)" />
              <span style={{ fontSize: "0.875rem", fontWeight: 600 }}>{result.model_type}</span>
            </div>
          </div>
          <div>
            <span style={{ fontSize: "0.75rem", color: "var(--text-muted)" }}>Transformation</span>
            <div style={{ marginTop: "0.2rem" }}>
              <span
                style={{
                  fontSize: "0.8125rem",
                  fontFamily: "var(--font-mono)",
                  color: "var(--text-secondary)",
                }}
              >
                {result.target_transformation}
              </span>
            </div>
          </div>
        </div>
      </div>

      {/* Explanatory Section */}
      <div className="card" style={{ padding: "1.25rem" }}>
        <div style={{ display: "flex", alignItems: "center", gap: "0.5rem", marginBottom: "0.5rem" }}>
          <Info size={16} color="var(--accent-blue)" />
          <h5 style={{ fontSize: "0.875rem", fontWeight: 600, color: "var(--text-primary)" }}>
            How this estimate is generated
          </h5>
        </div>
        <p style={{ fontSize: "0.8125rem", color: "var(--text-muted)", lineHeight: 1.5, margin: 0 }}>
          This valuation is produced using a machine learning model trained on historical Premier League
          and European transfer valuations. It synthesizes player context (age, contract duration, club
          stature) with comprehensive attacking and defensive performance metrics. It represents an
          objective statistical valuation benchmark, not a guaranteed contract negotiation price.
        </p>
      </div>
    </div>
  );
};
