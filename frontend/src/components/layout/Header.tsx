import React from "react";
import type { ActiveModule, HealthResponse } from "../../types/api";
import {
  TrendingUp,
  Swords,
  Users,
  RefreshCw,
  Activity,
  HelpCircle,
} from "lucide-react";

interface HeaderProps {
  activeModule: ActiveModule;
  onSelectModule: (module: ActiveModule) => void;
  health: HealthResponse | null;
  healthLoading: boolean;
  onRefreshHealth: () => void;
  onOpenModelInfo: () => void;
}

export const Header: React.FC<HeaderProps> = ({
  activeModule,
  onSelectModule,
  health,
  healthLoading,
  onRefreshHealth,
  onOpenModelInfo,
}) => {
  const isHealthy = health?.status === "healthy";

  return (
    <header
      style={{
        borderBottom: "1px solid var(--border-subtle)",
        backgroundColor: "rgba(9, 12, 16, 0.85)",
        backdropFilter: "blur(12px)",
        position: "sticky",
        top: 0,
        zIndex: 50,
        padding: "0.875rem 1.5rem",
      }}
    >
      <div
        style={{
          maxWidth: "1400px",
          margin: "0 auto",
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          flexWrap: "wrap",
          gap: "1rem",
        }}
      >
        {/* Brand Left */}
        <div style={{ display: "flex", alignItems: "center", gap: "0.75rem" }}>
          <div
            style={{
              width: "36px",
              height: "36px",
              borderRadius: "10px",
              background: "linear-gradient(135deg, #10b981 0%, #047857 100%)",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              boxShadow: "0 0 15px rgba(16, 185, 129, 0.35)",
            }}
          >
            <Activity size={20} color="#06281e" strokeWidth={2.5} />
          </div>
          <div>
            <div style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
              <span
                style={{
                  fontSize: "1.1875rem",
                  fontWeight: 800,
                  letterSpacing: "-0.03em",
                  color: "#ffffff",
                }}
              >
                PremAI
              </span>
              <span
                style={{
                  fontSize: "0.6875rem",
                  fontWeight: 700,
                  backgroundColor: "rgba(16, 185, 129, 0.15)",
                  color: "var(--accent-emerald-light)",
                  border: "1px solid rgba(16, 185, 129, 0.3)",
                  padding: "0.15rem 0.45rem",
                  borderRadius: "9999px",
                  textTransform: "uppercase",
                  letterSpacing: "0.05em",
                }}
              >
                Intelligence
              </span>
            </div>
            <p
              style={{
                fontSize: "0.75rem",
                color: "var(--text-muted)",
                margin: 0,
                lineHeight: 1.2,
              }}
            >
              Premier League Predictive &amp; Scouting Engine
            </p>
          </div>
        </div>

        {/* Center: Segmented Pill Module Selector */}
        <div className="pill-nav" role="tablist" aria-label="Main Module Selector">
          <button
            role="tab"
            aria-selected={activeModule === "transfer"}
            className={`pill-item ${activeModule === "transfer" ? "active" : ""}`}
            onClick={() => onSelectModule("transfer")}
          >
            <TrendingUp size={16} />
            <span>Transfer Value</span>
          </button>
          <button
            role="tab"
            aria-selected={activeModule === "match"}
            className={`pill-item ${activeModule === "match" ? "active" : ""}`}
            onClick={() => onSelectModule("match")}
          >
            <Swords size={16} />
            <span>Match Predictor</span>
          </button>
          <button
            role="tab"
            aria-selected={activeModule === "scouting"}
            className={`pill-item ${activeModule === "scouting" ? "active" : ""}`}
            onClick={() => onSelectModule("scouting")}
          >
            <Users size={16} />
            <span>Player Scouting</span>
          </button>
        </div>

        {/* Right: API Status & Info */}
        <div style={{ display: "flex", alignItems: "center", gap: "0.75rem" }}>
          {/* About Model Button */}
          <button
            onClick={onOpenModelInfo}
            className="btn btn-ghost"
            style={{
              padding: "0.4rem 0.75rem",
              fontSize: "0.8125rem",
              gap: "0.35rem",
            }}
            title="Inspect active model specifications and training metrics"
          >
            <HelpCircle size={15} />
            <span className="hidden-mobile">Model Specs</span>
          </button>

          {/* Latest Available Data Badge */}
          <div
            style={{
              display: "flex",
              alignItems: "center",
              gap: "0.45rem",
              padding: "0.35rem 0.75rem",
              borderRadius: "9999px",
              backgroundColor: "rgba(56, 189, 248, 0.1)",
              border: "1px solid rgba(56, 189, 248, 0.25)",
            }}
            title="Player valuations, squad rosters, and match histories are synchronized through September 20, 2026."
          >
            <div
              style={{
                width: "7px",
                height: "7px",
                borderRadius: "50%",
                backgroundColor: "#38bdf8",
                boxShadow: "0 0 8px #38bdf8",
              }}
            />
            <span
              style={{
                fontSize: "0.75rem",
                fontWeight: 600,
                color: "#38bdf8",
                letterSpacing: "0.01em",
              }}
            >
              Latest available data: 2026-09-20
            </span>
          </div>

          {/* Live Health Status */}
          <div
            style={{
              display: "flex",
              alignItems: "center",
              gap: "0.5rem",
              padding: "0.35rem 0.75rem",
              borderRadius: "9999px",
              backgroundColor: isHealthy
                ? "rgba(16, 185, 129, 0.1)"
                : "rgba(244, 63, 94, 0.1)",
              border: `1px solid ${
                isHealthy
                  ? "rgba(16, 185, 129, 0.3)"
                  : "rgba(244, 63, 94, 0.3)"
              }`,
            }}
          >
            <div
              style={{
                width: "8px",
                height: "8px",
                borderRadius: "50%",
                backgroundColor: isHealthy ? "#10b981" : "#f43f5e",
                boxShadow: isHealthy
                  ? "0 0 8px #10b981"
                  : "0 0 8px #f43f5e",
              }}
              className={healthLoading ? "animate-pulse-slow" : ""}
            />
            <span
              style={{
                fontSize: "0.75rem",
                fontWeight: 600,
                color: isHealthy ? "#34d399" : "#fb7185",
              }}
            >
              {healthLoading
                ? "Checking..."
                : isHealthy
                ? "API Connected"
                : "API Offline"}
            </span>

            <button
              onClick={onRefreshHealth}
              disabled={healthLoading}
              title="Refresh API health status"
              style={{
                background: "transparent",
                border: "none",
                color: "var(--text-muted)",
                cursor: "pointer",
                padding: "2px",
                display: "flex",
                alignItems: "center",
              }}
            >
              <RefreshCw
                size={12}
                className={healthLoading ? "animate-spin" : ""}
              />
            </button>
          </div>
        </div>
      </div>
    </header>
  );
};
