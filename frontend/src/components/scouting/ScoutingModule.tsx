import React, { useState } from "react";
import { SimilarPlayersView } from "./SimilarPlayersView";
import { ReplacementSearchView } from "./ReplacementSearchView";
import { Users, UserSearch } from "lucide-react";

export const ScoutingModule: React.FC = () => {
  const [internalTab, setInternalTab] = useState<"similar" | "replacement">("similar");

  return (
    <div className="animate-fade-in">
      {/* Header Banner */}
      <div
        style={{
          display: "flex",
          alignItems: "flex-end",
          justifyContent: "space-between",
          flexWrap: "wrap",
          gap: "1rem",
          marginBottom: "1.75rem",
        }}
      >
        <div>
          <div style={{ display: "flex", alignItems: "center", gap: "0.625rem", marginBottom: "0.35rem" }}>
            <h2 style={{ fontSize: "1.75rem", fontWeight: 800, letterSpacing: "-0.03em", margin: 0 }}>
              Player Scouting
            </h2>
            <span className="badge badge-emerald" style={{ fontSize: "0.75rem", fontWeight: 700, padding: "0.15rem 0.5rem" }}>
              V2
            </span>
          </div>
          <p style={{ color: "var(--text-secondary)", fontSize: "0.9375rem", margin: 0 }}>
            Find statistically similar players and potential replacements.
          </p>
        </div>

        {/* Internal Tabs */}
        <div className="pill-nav" role="tablist" aria-label="Scouting Sub-tabs">
          <button
            role="tab"
            aria-selected={internalTab === "similar"}
            className={`pill-item ${internalTab === "similar" ? "active" : ""}`}
            onClick={() => setInternalTab("similar")}
          >
            <Users size={15} />
            <span>Similar Players</span>
          </button>
          <button
            role="tab"
            aria-selected={internalTab === "replacement"}
            className={`pill-item ${internalTab === "replacement" ? "active" : ""}`}
            onClick={() => setInternalTab("replacement")}
          >
            <UserSearch size={15} />
            <span>Find Replacement</span>
          </button>
        </div>
      </div>

      {/* Internal Sub-View */}
      {internalTab === "similar" ? <SimilarPlayersView /> : <ReplacementSearchView />}
    </div>
  );
};
