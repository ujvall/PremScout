import React from "react";
import type { ActiveModule, HealthResponse } from "../../types/api";
import {
  TrendingUp,
  Swords,
  Users,
  RefreshCw,
  Activity,
  Sliders,
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
        backgroundColor: "rgba(18, 19, 22, 0.92)",
        backdropFilter: "blur(12px)",
        position: "sticky",
        top: 0,
        zIndex: 50,
        padding: "0.625rem 1.5rem",
      }}
    >
      <div
        style={{
          maxWidth: "1440px",
          margin: "0 auto",
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          flexWrap: "wrap",
          gap: "0.75rem",
        }}
      >
        {/* Brand Left */}
        <div style={{ display: "flex", alignItems: "center", gap: "0.75rem" }}>
          <div
            style={{
              width: "32px",
              height: "32px",
              borderRadius: "var(--radius-md)",
              backgroundColor: "var(--bg-elevated)",
              border: "1px solid var(--border-medium)",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              color: "var(--accent-tea-green)",
            }}
          >
            <Activity size={18} strokeWidth={2.2} />
          </div>
          <div>
            <div style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
              <span
                style={{
                  fontSize: "1.0625rem",
                  fontWeight: 700,
                  letterSpacing: "-0.025em",
                  color: "var(--text-primary)",
                }}
              >
                PremAI
              </span>
              <span
                style={{
                  fontSize: "0.6875rem",
                  fontWeight: 600,
                  color: "var(--text-secondary)",
                  backgroundColor: "rgba(176, 192, 188, 0.08)",
                  border: "1px solid var(--border-subtle)",
                  padding: "0.1rem 0.4rem",
                  borderRadius: "var(--radius-xs)",
                  letterSpacing: "0.03em",
                }}
              >
                SCOUT
              </span>
            </div>
            <p
              style={{
                fontSize: "0.6875rem",
                color: "var(--text-muted)",
                margin: 0,
                lineHeight: 1.1,
              }}
            >
              Premier League Predictive &amp; Scouting Engine
            </p>
          </div>
        </div>

        {/* Center: Segmented Navigation Taskbar */}
        <nav
          className="segmented-nav"
          role="tablist"
          aria-label="Main Module Selector"
        >
          <button
            role="tab"
            aria-selected={activeModule === "transfer"}
            className={`segmented-nav-item ${
              activeModule === "transfer" ? "active" : ""
            }`}
            onClick={() => onSelectModule("transfer")}
          >
            <TrendingUp size={15} />
            <span>Transfer Value</span>
          </button>
          <button
            role="tab"
            aria-selected={activeModule === "match"}
            className={`segmented-nav-item ${
              activeModule === "match" ? "active" : ""
            }`}
            onClick={() => onSelectModule("match")}
          >
            <Swords size={15} />
            <span>Match Predictor</span>
          </button>
          <button
            role="tab"
            aria-selected={activeModule === "scouting"}
            className={`segmented-nav-item ${
              activeModule === "scouting" ? "active" : ""
            }`}
            onClick={() => onSelectModule("scouting")}
          >
            <Users size={15} />
            <span>Player Scouting</span>
          </button>
        </nav>

        {/* Right: Cohesive Utility Bar */}
        <div style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
          {/* Model Specs Secondary Action */}
          <button
            onClick={onOpenModelInfo}
            className="btn btn-secondary btn-sm"
            style={{ gap: "0.35rem" }}
            title="Inspect active ML models and performance metrics"
          >
            <Sliders size={13} />
            <span className="hidden-mobile">Model Specs</span>
          </button>

          {/* Latest Data Metadata */}
          <div
            style={{
              display: "flex",
              alignItems: "center",
              gap: "0.4rem",
              height: "32px",
              padding: "0 0.625rem",
              borderRadius: "var(--radius-btn)",
              backgroundColor: "var(--bg-surface)",
              border: "1px solid var(--border-subtle)",
              fontSize: "0.75rem",
              color: "var(--text-secondary)",
            }}
            title="Valuations, rosters, and match histories synchronized as of September 20, 2026"
          >
            <div
              style={{
                width: "6px",
                height: "6px",
                borderRadius: "50%",
                backgroundColor: "var(--accent-blue)",
              }}
            />
            <span className="hidden-mobile">Data: </span>
            <span style={{ fontWeight: 600, color: "var(--text-primary)" }}>
              2026-09-20
            </span>
          </div>

          {/* Live API Health Status */}
          <div
            style={{
              display: "flex",
              alignItems: "center",
              gap: "0.4rem",
              height: "32px",
              padding: "0 0.625rem",
              borderRadius: "var(--radius-btn)",
              backgroundColor: "var(--bg-surface)",
              border: "1px solid var(--border-subtle)",
              fontSize: "0.75rem",
            }}
          >
            <div
              style={{
                width: "6px",
                height: "6px",
                borderRadius: "50%",
                backgroundColor: isHealthy ? "var(--success)" : "var(--error)",
              }}
              className={healthLoading ? "animate-pulse-slow" : ""}
            />
            <span
              style={{
                fontWeight: 600,
                color: isHealthy ? "var(--success)" : "var(--error)",
              }}
            >
              {healthLoading
                ? "Connecting..."
                : isHealthy
                ? "API Connected"
                : "Offline"}
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
                marginLeft: "2px",
              }}
              aria-label="Refresh API connection status"
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
