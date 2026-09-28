import React, { useState } from "react";
import type { TransferPredictRequest, TransferPredictResponse } from "../../types/api";
import { predictTransferValue } from "../../api";
import { TransferForm } from "./TransferForm";
import { TransferResult } from "./TransferResult";
import { AlertCircle } from "lucide-react";

export const TransferModule: React.FC = () => {
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<TransferPredictResponse | null>(null);

  const handleSubmit = async (formData: TransferPredictRequest) => {
    setLoading(true);
    setError(null);
    try {
      const response = await predictTransferValue(formData);
      setResult(response);
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : "Something went wrong while generating the prediction.";
      setError(msg);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="animate-fade-in">
      {/* Header Banner */}
      <div style={{ marginBottom: "2rem" }}>
        <div style={{ display: "flex", alignItems: "center", gap: "0.625rem", marginBottom: "0.35rem" }}>
          <h2 style={{ fontSize: "1.75rem", fontWeight: 800, letterSpacing: "-0.03em", margin: 0 }}>
            Transfer Value Predictor
          </h2>
          <span className="badge badge-emerald" style={{ fontSize: "0.75rem", fontWeight: 700, padding: "0.15rem 0.5rem" }}>
            V2
          </span>
        </div>
        <p style={{ color: "var(--text-secondary)", fontSize: "0.9375rem", margin: 0 }}>
          Estimate a player's market value from football performance and player context.
        </p>
      </div>

      {error && (
        <div
          style={{
            display: "flex",
            alignItems: "center",
            gap: "0.75rem",
            padding: "1rem 1.25rem",
            borderRadius: "var(--radius-md)",
            backgroundColor: "rgba(244, 63, 94, 0.12)",
            border: "1px solid rgba(244, 63, 94, 0.3)",
            color: "#fb7185",
            fontSize: "0.875rem",
            marginBottom: "1.5rem",
          }}
        >
          <AlertCircle size={18} />
          <span>{error}</span>
        </div>
      )}

      {/* Main Grid: Form Left, Result Right */}
      <div className="grid-layout-with-sidebar">
        <div>
          <TransferForm onSubmit={handleSubmit} loading={loading} />
        </div>
        <div>
          <div style={{ position: "sticky", top: "5.5rem" }}>
            <TransferResult result={result} loading={loading} />
          </div>
        </div>
      </div>
    </div>
  );
};
