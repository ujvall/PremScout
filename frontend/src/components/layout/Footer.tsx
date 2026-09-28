import React from "react";
import { Activity, ShieldCheck, Database, GitBranch } from "lucide-react";

export const Footer: React.FC = () => {
  return (
    <footer
      style={{
        borderTop: "1px solid var(--border-subtle)",
        backgroundColor: "var(--bg-app)",
        padding: "2.5rem 1.5rem 3rem",
        marginTop: "auto",
      }}
    >
      <div
        style={{
          maxWidth: "1400px",
          margin: "0 auto",
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          flexWrap: "wrap",
          gap: "1.5rem",
        }}
      >
        <div>
          <div style={{ display: "flex", alignItems: "center", gap: "0.5rem", marginBottom: "0.4rem" }}>
            <Activity size={16} color="var(--accent-emerald)" />
            <span style={{ fontWeight: 700, fontSize: "0.9375rem", letterSpacing: "-0.02em" }}>
              PremAI Football Intelligence
            </span>
          </div>
          <p style={{ fontSize: "0.8125rem", color: "var(--text-muted)", maxWidth: "520px", margin: 0 }}>
            Advanced analytical suite combining transfer market valuation, pre-match probabilistic
            forecasting, and multidimensional player similarity scouting.
          </p>
        </div>

        <div style={{ display: "flex", alignItems: "center", gap: "1.5rem", flexWrap: "wrap" }}>
          <div style={{ display: "flex", alignItems: "center", gap: "0.4rem", color: "var(--text-muted)", fontSize: "0.8125rem" }}>
            <Database size={14} />
            <span>Transfermarkt &amp; FBref Data</span>
          </div>
          <div style={{ display: "flex", alignItems: "center", gap: "0.4rem", color: "var(--text-muted)", fontSize: "0.8125rem" }}>
            <ShieldCheck size={14} />
            <span>FastAPI ML Engine</span>
          </div>
          <div style={{ display: "flex", alignItems: "center", gap: "0.4rem", color: "var(--text-muted)", fontSize: "0.8125rem" }}>
            <GitBranch size={14} />
            <span>v1.0.0</span>
          </div>
        </div>
      </div>
    </footer>
  );
};
