import { useState, useEffect, useCallback } from "react";
import type { ActiveModule, HealthResponse } from "./types/api";
import { fetchHealth } from "./api";
import { Header } from "./components/layout/Header";
import { Footer } from "./components/layout/Footer";
import { TransferModule } from "./components/transfer/TransferModule";
import { MatchModule } from "./components/match/MatchModule";
import { ScoutingModule } from "./components/scouting/ScoutingModule";
import { ModelInfoModal } from "./components/common/ModelInfoModal";
import { AlertCircle, RefreshCw } from "lucide-react";

export function App() {
  const [activeModule, setActiveModule] = useState<ActiveModule>("transfer");
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [healthLoading, setHealthLoading] = useState<boolean>(true);
  const [healthError, setHealthError] = useState<string | null>(null);
  const [isModelInfoOpen, setIsModelInfoOpen] = useState<boolean>(false);

  // Poll API health on mount and on user refresh
  const checkHealth = useCallback(async () => {
    setHealthLoading(true);
    setHealthError(null);
    try {
      const data = await fetchHealth();
      setHealth(data);
    } catch {
      setHealth(null);
      setHealthError("Unable to connect to the PremAI backend. Ensure FastAPI server is running on port 8000.");
    } finally {
      setHealthLoading(false);
    }
  }, []);

  useEffect(() => {
    checkHealth();
  }, [checkHealth]);

  return (
    <div style={{ display: "flex", flexDirection: "column", minHeight: "100vh" }}>
      {/* Persistent Navigation Header */}
      <Header
        activeModule={activeModule}
        onSelectModule={setActiveModule}
        health={health}
        healthLoading={healthLoading}
        onRefreshHealth={checkHealth}
        onOpenModelInfo={() => setIsModelInfoOpen(true)}
      />

      {/* Main Content Area */}
      <main className="app-container" style={{ flex: 1, paddingTop: "2rem" }}>
        {/* Offline Warning Banner if health check failed */}
        {!healthLoading && (!health || health.status !== "healthy") && (
          <div
            style={{
              display: "flex",
              alignItems: "center",
              justifyContent: "space-between",
              padding: "0.875rem 1.25rem",
              borderRadius: "var(--radius-md)",
              backgroundColor: "rgba(244, 63, 94, 0.12)",
              border: "1px solid rgba(244, 63, 94, 0.3)",
              color: "#fb7185",
              fontSize: "0.875rem",
              marginBottom: "1.75rem",
            }}
          >
            <div style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
              <AlertCircle size={18} />
              <span>
                {healthError || "Unable to connect to the PremAI backend. Please check the backend connection."}
              </span>
            </div>
            <button
              onClick={checkHealth}
              disabled={healthLoading}
              className="btn btn-secondary"
              style={{ padding: "0.25rem 0.625rem", fontSize: "0.75rem", gap: "0.3rem" }}
            >
              <RefreshCw size={12} className={healthLoading ? "animate-spin" : ""} />
              <span>Retry</span>
            </button>
          </div>
        )}

        {/* Dynamic Module Rendering */}
        {activeModule === "transfer" && <TransferModule />}
        {activeModule === "match" && <MatchModule />}
        {activeModule === "scouting" && <ScoutingModule />}
      </main>

      {/* Model Information Modal */}
      <ModelInfoModal
        isOpen={isModelInfoOpen}
        onClose={() => setIsModelInfoOpen(false)}
        initialTab={activeModule}
      />

      {/* Persistent Footer */}
      <Footer />
    </div>
  );
}

export default App;
