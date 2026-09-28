import React, { useEffect, useState } from "react";
import type {
  TransferModelInfo,
  MatchModelInfo,
  ScoutingModelInfo,
} from "../../types/api";
import {
  fetchTransferModelInfo,
  fetchMatchModelInfo,
  fetchScoutingModelInfo,
} from "../../api";
import { X, Cpu, CheckCircle2, Layers } from "lucide-react";

interface ModelInfoModalProps {
  isOpen: boolean;
  onClose: () => void;
  initialTab?: "transfer" | "match" | "scouting";
}

export const ModelInfoModal: React.FC<ModelInfoModalProps> = ({
  isOpen,
  onClose,
  initialTab = "transfer",
}) => {
  const [activeTab, setActiveTab] = useState<"transfer" | "match" | "scouting">(
    initialTab
  );
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  const [transferInfo, setTransferInfo] = useState<TransferModelInfo | null>(null);
  const [matchInfo, setMatchInfo] = useState<MatchModelInfo | null>(null);
  const [scoutingInfo, setScoutingInfo] = useState<ScoutingModelInfo | null>(null);

  useEffect(() => {
    setActiveTab(initialTab);
  }, [initialTab]);

  useEffect(() => {
    if (!isOpen) return;

    const loadData = async () => {
      setLoading(true);
      setError(null);
      try {
        if (activeTab === "transfer" && !transferInfo) {
          const data = await fetchTransferModelInfo();
          setTransferInfo(data);
        } else if (activeTab === "match" && !matchInfo) {
          const data = await fetchMatchModelInfo();
          setMatchInfo(data);
        } else if (activeTab === "scouting" && !scoutingInfo) {
          const data = await fetchScoutingModelInfo();
          setScoutingInfo(data);
        }
      } catch (err: unknown) {
        const msg = err instanceof Error ? err.message : "Failed to load model specs";
        setError(msg);
      } finally {
        setLoading(false);
      }
    };

    loadData();
  }, [isOpen, activeTab, transferInfo, matchInfo, scoutingInfo]);

  useEffect(() => {
    if (!isOpen) return;
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        onClose();
      }
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [isOpen, onClose]);

  if (!isOpen) return null;

  return (
    <div className="modal-overlay" onClick={onClose} role="dialog" aria-modal="true">
      <div
        className="modal-content animate-fade-in"
        onClick={(e) => e.stopPropagation()}
        style={{ padding: "1.75rem" }}
      >
        {/* Header */}
        <div
          style={{
            display: "flex",
            alignItems: "center",
            justifyContent: "space-between",
            borderBottom: "1px solid var(--border-subtle)",
            paddingBottom: "1rem",
            marginBottom: "1.25rem",
          }}
        >
          <div style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
            <Cpu size={20} color="var(--accent-emerald)" />
            <h3 style={{ fontSize: "1.25rem", margin: 0 }}>Active Model Specifications</h3>
          </div>
          <button
            onClick={onClose}
            className="btn btn-ghost"
            style={{ padding: "0.35rem" }}
            aria-label="Close modal"
          >
            <X size={18} />
          </button>
        </div>

        {/* Tab switch */}
        <div className="pill-nav" style={{ marginBottom: "1.5rem" }}>
          <button
            className={`pill-item ${activeTab === "transfer" ? "active" : ""}`}
            onClick={() => setActiveTab("transfer")}
          >
            Transfer Valuation
          </button>
          <button
            className={`pill-item ${activeTab === "match" ? "active" : ""}`}
            onClick={() => setActiveTab("match")}
          >
            Match Predictor
          </button>
          <button
            className={`pill-item ${activeTab === "scouting" ? "active" : ""}`}
            onClick={() => setActiveTab("scouting")}
          >
            Player Scouting
          </button>
        </div>

        {/* Content */}
        {loading && (
          <div style={{ padding: "2.5rem", textAlign: "center", color: "var(--text-muted)" }}>
            <div className="animate-spin" style={{ display: "inline-block", marginBottom: "0.5rem" }}>
              <Layers size={24} color="var(--accent-emerald)" />
            </div>
            <p>Fetching verified backend model metadata...</p>
          </div>
        )}

        {error && (
          <div
            style={{
              padding: "1rem",
              borderRadius: "var(--radius-md)",
              backgroundColor: "rgba(244, 63, 94, 0.1)",
              border: "1px solid rgba(244, 63, 94, 0.3)",
              color: "#fb7185",
              fontSize: "0.875rem",
              marginBottom: "1rem",
            }}
          >
            {error}
          </div>
        )}

        {!loading && activeTab === "transfer" && transferInfo && (
          <div className="animate-fade-in" style={{ display: "flex", flexDirection: "column", gap: "1.25rem" }}>
            <div className="grid-3">
              <div className="card" style={{ padding: "1rem" }}>
                <span style={{ fontSize: "0.75rem", color: "var(--text-muted)" }}>Model Version &amp; Architecture</span>
                <div style={{ display: "flex", alignItems: "center", gap: "0.5rem", marginTop: "0.25rem" }}>
                  <span className="badge badge-emerald" style={{ fontWeight: 700 }}>
                    {transferInfo.model_version || "V2"}
                  </span>
                  <span style={{ fontWeight: 600, fontSize: "0.9375rem" }}>
                    {transferInfo.architecture || "XGBoost + HistGradientBoosting Ensemble"}
                  </span>
                </div>
              </div>
              <div className="card" style={{ padding: "1rem" }}>
                <span style={{ fontSize: "0.75rem", color: "var(--text-muted)" }}>Target Transformation</span>
                <p style={{ fontWeight: 700, fontSize: "1.1rem", marginTop: "0.25rem", fontFamily: "var(--font-mono)" }}>
                  {transferInfo.target_transformation}
                </p>
              </div>
              <div className="card" style={{ padding: "1rem" }}>
                <span style={{ fontSize: "0.75rem", color: "var(--text-muted)" }}>Features Count</span>
                <p style={{ fontWeight: 700, fontSize: "1.1rem", marginTop: "0.25rem" }}>
                  {transferInfo.feature_count} base / {(transferInfo as any).v2_feature_count || 26} engineered
                </p>
              </div>
            </div>

            {/* Validation Metrics */}
            <div className="card" style={{ padding: "1.25rem" }}>
              <h4 style={{ fontSize: "0.875rem", marginBottom: "0.75rem", color: "var(--text-secondary)" }}>
                Validation Evaluation (Season: {transferInfo.validation_season})
              </h4>
              <div className="grid-3">
                <div>
                  <span style={{ fontSize: "0.75rem", color: "var(--text-muted)" }}>Val MAE</span>
                  <p style={{ fontWeight: 700, fontSize: "1.25rem", color: "var(--accent-emerald)" }}>
                    {transferInfo.validation_metrics?.mae_formatted || `€${((transferInfo.validation_metrics?.mae || 0) / 1e6).toFixed(2)}M`}
                  </p>
                </div>
                <div>
                  <span style={{ fontSize: "0.75rem", color: "var(--text-muted)" }}>Val RMSE</span>
                  <p style={{ fontWeight: 700, fontSize: "1.25rem" }}>
                    {transferInfo.validation_metrics?.rmse_formatted || `€${((transferInfo.validation_metrics?.rmse || 0) / 1e6).toFixed(2)}M`}
                  </p>
                </div>
                <div>
                  <span style={{ fontSize: "0.75rem", color: "var(--text-muted)" }}>Val R²</span>
                  <p style={{ fontWeight: 700, fontSize: "1.25rem" }}>
                    {transferInfo.validation_metrics?.r2_formatted || (transferInfo.validation_metrics?.r2 ? transferInfo.validation_metrics.r2.toFixed(4) : "N/A")}
                  </p>
                </div>
              </div>
            </div>

            {/* Hold-out Test Metrics */}
            <div className="card" style={{ padding: "1.25rem" }}>
              <h4 style={{ fontSize: "0.875rem", marginBottom: "0.75rem", color: "var(--text-secondary)" }}>
                Hold-out Test Evaluation (Test Season: {transferInfo.test_season})
              </h4>
              <div className="grid-3">
                <div>
                  <span style={{ fontSize: "0.75rem", color: "var(--text-muted)" }}>Test MAE</span>
                  <p style={{ fontWeight: 700, fontSize: "1.25rem", color: "var(--accent-emerald)" }}>
                    {transferInfo.test_metrics?.mae_formatted || `€${((transferInfo.test_metrics?.mae || 0) / 1e6).toFixed(2)}M`}
                  </p>
                </div>
                <div>
                  <span style={{ fontSize: "0.75rem", color: "var(--text-muted)" }}>Test RMSE</span>
                  <p style={{ fontWeight: 700, fontSize: "1.25rem" }}>
                    {transferInfo.test_metrics?.rmse_formatted || `€${((transferInfo.test_metrics?.rmse || 0) / 1e6).toFixed(2)}M`}
                  </p>
                </div>
                <div>
                  <span style={{ fontSize: "0.75rem", color: "var(--text-muted)" }}>Test R²</span>
                  <p style={{ fontWeight: 700, fontSize: "1.25rem" }}>
                    {transferInfo.test_metrics?.r2_formatted || (transferInfo.test_metrics?.r2 ? transferInfo.test_metrics.r2.toFixed(4) : "N/A")}
                  </p>
                </div>
              </div>
            </div>

            <div>
              <span style={{ fontSize: "0.75rem", color: "var(--text-muted)" }}>Model Features ({transferInfo.features.length})</span>
              <div style={{ display: "flex", flexWrap: "wrap", gap: "0.35rem", marginTop: "0.4rem" }}>
                {transferInfo.features.map((feat) => (
                  <span
                    key={feat}
                    style={{
                      fontSize: "0.75rem",
                      padding: "0.2rem 0.5rem",
                      borderRadius: "var(--radius-sm)",
                      backgroundColor: "var(--bg-elevated)",
                      border: "1px solid var(--border-subtle)",
                      fontFamily: "var(--font-mono)",
                    }}
                  >
                    {feat}
                  </span>
                ))}
              </div>
            </div>
          </div>
        )}

        {!loading && activeTab === "match" && matchInfo && (
          <div className="animate-fade-in" style={{ display: "flex", flexDirection: "column", gap: "1.25rem" }}>
            <div className="grid-3">
              <div className="card" style={{ padding: "1rem" }}>
                <span style={{ fontSize: "0.75rem", color: "var(--text-muted)" }}>Model Version &amp; Architecture</span>
                <div style={{ display: "flex", alignItems: "center", gap: "0.5rem", marginTop: "0.25rem" }}>
                  <span className="badge badge-blue" style={{ fontWeight: 700 }}>
                    {matchInfo.model_version || "V1-Improved"}
                  </span>
                  <span style={{ fontWeight: 600, fontSize: "0.9375rem" }}>
                    {matchInfo.architecture || "XGBoost + Sequential Elo"}
                  </span>
                </div>
              </div>
              <div className="card" style={{ padding: "1rem" }}>
                <span style={{ fontSize: "0.75rem", color: "var(--text-muted)" }}>Class Labels</span>
                <p style={{ fontWeight: 700, fontSize: "1.05rem", marginTop: "0.25rem" }}>
                  Home (H) / Draw (D) / Away (A)
                </p>
              </div>
              <div className="card" style={{ padding: "1rem" }}>
                <span style={{ fontSize: "0.75rem", color: "var(--text-muted)" }}>Features Count</span>
                <p style={{ fontWeight: 700, fontSize: "1.1rem", marginTop: "0.25rem" }}>
                  {matchInfo.feature_count} base / {(matchInfo as any).total_features_with_elo || 32} with Elo
                </p>
              </div>
            </div>

            {/* Validation Metrics */}
            <div className="card" style={{ padding: "1.25rem" }}>
              <h4 style={{ fontSize: "0.875rem", marginBottom: "0.75rem", color: "var(--text-secondary)" }}>
                Validation Evaluation (Season: {matchInfo.validation_season})
              </h4>
              <div className="grid-4" style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: "1rem" }}>
                <div>
                  <span style={{ fontSize: "0.75rem", color: "var(--text-muted)" }}>Accuracy</span>
                  <p style={{ fontWeight: 700, fontSize: "1.15rem", color: "var(--accent-emerald)" }}>
                    {matchInfo.validation_metrics?.accuracy !== undefined
                      ? `${(matchInfo.validation_metrics.accuracy * 100).toFixed(1)}%`
                      : "N/A"}
                  </p>
                </div>
                <div>
                  <span style={{ fontSize: "0.75rem", color: "var(--text-muted)" }}>Macro F1</span>
                  <p style={{ fontWeight: 700, fontSize: "1.15rem" }}>
                    {matchInfo.validation_metrics?.f1_macro !== undefined
                      ? matchInfo.validation_metrics.f1_macro.toFixed(4)
                      : "N/A"}
                  </p>
                </div>
                <div>
                  <span style={{ fontSize: "0.75rem", color: "var(--text-muted)" }}>Log Loss</span>
                  <p style={{ fontWeight: 700, fontSize: "1.15rem" }}>
                    {matchInfo.validation_metrics?.log_loss !== undefined
                      ? matchInfo.validation_metrics.log_loss.toFixed(4)
                      : "N/A"}
                  </p>
                </div>
                <div>
                  <span style={{ fontSize: "0.75rem", color: "var(--text-muted)" }}>Brier Score</span>
                  <p style={{ fontWeight: 700, fontSize: "1.15rem" }}>
                    {matchInfo.validation_metrics?.brier_score !== undefined
                      ? matchInfo.validation_metrics.brier_score.toFixed(4)
                      : "N/A"}
                  </p>
                </div>
              </div>
            </div>

            {/* Hold-out Test Metrics */}
            <div className="card" style={{ padding: "1.25rem" }}>
              <h4 style={{ fontSize: "0.875rem", marginBottom: "0.75rem", color: "var(--text-secondary)" }}>
                Hold-out Test Evaluation (Test Season: {matchInfo.test_season})
              </h4>
              <div className="grid-4" style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: "1rem" }}>
                <div>
                  <span style={{ fontSize: "0.75rem", color: "var(--text-muted)" }}>Test Accuracy</span>
                  <p style={{ fontWeight: 700, fontSize: "1.15rem", color: "var(--accent-emerald)" }}>
                    {matchInfo.test_metrics?.accuracy !== undefined
                      ? `${(matchInfo.test_metrics.accuracy * 100).toFixed(1)}%`
                      : "N/A"}
                  </p>
                </div>
                <div>
                  <span style={{ fontSize: "0.75rem", color: "var(--text-muted)" }}>Macro F1</span>
                  <p style={{ fontWeight: 700, fontSize: "1.15rem" }}>
                    {matchInfo.test_metrics?.f1_macro !== undefined
                      ? matchInfo.test_metrics.f1_macro.toFixed(4)
                      : "N/A"}
                  </p>
                </div>
                <div>
                  <span style={{ fontSize: "0.75rem", color: "var(--text-muted)" }}>Log Loss</span>
                  <p style={{ fontWeight: 700, fontSize: "1.15rem" }}>
                    {matchInfo.test_metrics?.log_loss !== undefined
                      ? matchInfo.test_metrics.log_loss.toFixed(4)
                      : "N/A"}
                  </p>
                </div>
                <div>
                  <span style={{ fontSize: "0.75rem", color: "var(--text-muted)" }}>Brier Score</span>
                  <p style={{ fontWeight: 700, fontSize: "1.15rem" }}>
                    {matchInfo.test_metrics?.brier_score !== undefined
                      ? matchInfo.test_metrics.brier_score.toFixed(4)
                      : "N/A"}
                  </p>
                </div>
              </div>
            </div>

            <div className="card" style={{ padding: "1rem" }}>
              <h4 style={{ fontSize: "0.875rem", marginBottom: "0.35rem", color: "var(--text-secondary)" }}>
                Partitions &amp; Chronological Split
              </h4>
              <p style={{ fontSize: "0.8125rem", color: "var(--text-muted)", margin: 0 }}>
                Train: {matchInfo.split_counts?.train} matches | Val ({matchInfo.validation_season}):{" "}
                {matchInfo.split_counts?.val} matches | Test ({matchInfo.test_season}):{" "}
                {matchInfo.split_counts?.test} matches
              </p>
            </div>

            <div>
              <span style={{ fontSize: "0.75rem", color: "var(--text-muted)" }}>Pre-Match Features ({matchInfo.features.length})</span>
              <div style={{ display: "flex", flexWrap: "wrap", gap: "0.35rem", marginTop: "0.4rem" }}>
                {matchInfo.features.map((feat) => (
                  <span
                    key={feat}
                    style={{
                      fontSize: "0.75rem",
                      padding: "0.2rem 0.5rem",
                      borderRadius: "var(--radius-sm)",
                      backgroundColor: "var(--bg-elevated)",
                      border: "1px solid var(--border-subtle)",
                      fontFamily: "var(--font-mono)",
                    }}
                  >
                    {feat}
                  </span>
                ))}
              </div>
            </div>
          </div>
        )}

        {!loading && activeTab === "scouting" && scoutingInfo && (
          <div className="animate-fade-in" style={{ display: "flex", flexDirection: "column", gap: "1.25rem" }}>
            <div className="grid-3">
              <div className="card" style={{ padding: "1rem" }}>
                <span style={{ fontSize: "0.75rem", color: "var(--text-muted)" }}>Model Version &amp; Architecture</span>
                <div style={{ display: "flex", alignItems: "center", gap: "0.5rem", marginTop: "0.25rem" }}>
                  <span className="badge badge-emerald" style={{ fontWeight: 700 }}>
                    {scoutingInfo.model_version || "V2"}
                  </span>
                  <span style={{ fontWeight: 600, fontSize: "0.9375rem" }}>
                    {scoutingInfo.architecture || "Archetype-Weighted Cosine Similarity"}
                  </span>
                </div>
              </div>
              <div className="card" style={{ padding: "1rem" }}>
                <span style={{ fontSize: "0.75rem", color: "var(--text-muted)" }}>Pre-Processing &amp; Scaling</span>
                <p style={{ fontWeight: 600, fontSize: "0.9375rem", marginTop: "0.25rem" }}>
                  {scoutingInfo.scaling_method}
                </p>
              </div>
              <div className="card" style={{ padding: "1rem" }}>
                <span style={{ fontSize: "0.75rem", color: "var(--text-muted)" }}>Player Database</span>
                <p style={{ fontWeight: 700, fontSize: "1.1rem", marginTop: "0.25rem" }}>
                  {scoutingInfo.distinct_players_count} distinct profiles ({scoutingInfo.dataset_row_count} player-seasons)
                </p>
              </div>
            </div>

            {/* Strict Goalkeeper Isolation Badge / Notice */}
            <div
              style={{
                display: "flex",
                alignItems: "center",
                gap: "0.75rem",
                padding: "0.875rem 1rem",
                borderRadius: "var(--radius-md)",
                backgroundColor: "rgba(16, 185, 129, 0.08)",
                border: "1px solid rgba(16, 185, 129, 0.25)",
              }}
            >
              <span className="badge badge-emerald" style={{ fontWeight: 700 }}>
                Strict Goalkeeper Isolation: Active
              </span>
              <span style={{ fontSize: "0.8125rem", color: "var(--text-secondary)" }}>
                Goalkeepers are evaluated strictly against other goalkeepers using specialized metrics with 0% outfield cross-contamination.
              </span>
            </div>

            {/* Retrieval & Ranking Evaluation Metrics */}
            <div className="card" style={{ padding: "1.25rem" }}>
              <h4 style={{ fontSize: "0.875rem", marginBottom: "0.75rem", color: "var(--text-secondary)" }}>
                Retrieval &amp; Ranking Performance Metrics
              </h4>
              <div className="grid-4" style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: "1rem" }}>
                <div>
                  <span style={{ fontSize: "0.75rem", color: "var(--text-muted)" }}>Top-5 Same Position</span>
                  <p style={{ fontWeight: 700, fontSize: "1.25rem", color: "var(--accent-emerald)" }}>
                    {scoutingInfo.evaluation_metrics?.top_5_same_position_rate !== undefined
                      ? `${(scoutingInfo.evaluation_metrics.top_5_same_position_rate * 100).toFixed(1)}%`
                      : "N/A"}
                  </p>
                </div>
                <div>
                  <span style={{ fontSize: "0.75rem", color: "var(--text-muted)" }}>Top-10 Same Position</span>
                  <p style={{ fontWeight: 700, fontSize: "1.25rem", color: "var(--accent-emerald)" }}>
                    {scoutingInfo.evaluation_metrics?.top_10_same_position_rate !== undefined
                      ? `${(scoutingInfo.evaluation_metrics.top_10_same_position_rate * 100).toFixed(1)}%`
                      : "N/A"}
                  </p>
                </div>
                <div>
                  <span style={{ fontSize: "0.75rem", color: "var(--text-muted)" }}>Position MRR</span>
                  <p style={{ fontWeight: 700, fontSize: "1.25rem" }}>
                    {scoutingInfo.evaluation_metrics?.position_mrr !== undefined
                      ? Number(scoutingInfo.evaluation_metrics.position_mrr).toFixed(4)
                      : "N/A"}
                  </p>
                </div>
                <div>
                  <span style={{ fontSize: "0.75rem", color: "var(--text-muted)" }}>NDCG@5</span>
                  <p style={{ fontWeight: 700, fontSize: "1.25rem" }}>
                    {scoutingInfo.evaluation_metrics?.ndcg_5 !== undefined
                      ? Number(scoutingInfo.evaluation_metrics.ndcg_5).toFixed(4)
                      : "N/A"}
                  </p>
                </div>
              </div>
            </div>

            <div className="card" style={{ padding: "1.25rem" }}>
              <h4 style={{ fontSize: "0.875rem", marginBottom: "0.5rem", color: "var(--text-secondary)" }}>
                Recruitment Filters Supported
              </h4>
              <div style={{ display: "flex", flexWrap: "wrap", gap: "0.5rem" }}>
                {scoutingInfo.filters_supported.map((filter) => (
                  <span key={filter} className="badge badge-emerald">
                    <CheckCircle2 size={12} />
                    {filter}
                  </span>
                ))}
              </div>
            </div>

            <div>
              <span style={{ fontSize: "0.75rem", color: "var(--text-muted)" }}>
                Scouted Statistical Features ({scoutingInfo.feature_count})
              </span>
              <div style={{ display: "flex", flexWrap: "wrap", gap: "0.35rem", marginTop: "0.4rem" }}>
                {scoutingInfo.features_used.map((feat) => (
                  <span
                    key={feat}
                    style={{
                      fontSize: "0.75rem",
                      padding: "0.2rem 0.5rem",
                      borderRadius: "var(--radius-sm)",
                      backgroundColor: "var(--bg-elevated)",
                      border: "1px solid var(--border-subtle)",
                      fontFamily: "var(--font-mono)",
                    }}
                  >
                    {feat}
                  </span>
                ))}
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};
