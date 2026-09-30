import React, { useState, useEffect } from "react";
import type { TransferPredictRequest } from "../../types/api";
import { Sparkles, User, Building, Activity, RotateCcw, AlertCircle } from "lucide-react";
import { fetchTransferPresets } from "../../api/transfer";

interface TransferFormProps {
  onSubmit: (data: TransferPredictRequest) => void;
  loading: boolean;
}

const PRESETS: Record<string, TransferPredictRequest> = {
  "Rayan Cherki": {
    player_name: "Rayan Cherki",
    age: 23,
    height_cm: 177,
    primary_position: "Midfield",
    sub_position: "Attacking Midfield",
    preferred_foot: "Both",
    contract_years_left: 3.8,
    club_market_value_eur: 750000000,
    prev_market_value_eur: 45000000,
    minutes_played: 2979,
    appearances: 58,
    goals: 11,
    assists: 16,
    yellow_cards: 2,
    red_cards: 0,
    goals_per90: 0.33,
    assists_per90: 0.48,
    shots_per90: 1.27,
    npxg_per90: 0.13,
  },
  "Erling Haaland": {
    player_name: "Erling Haaland",
    age: 26,
    height_cm: 195,
    primary_position: "Attack",
    sub_position: "Centre-Forward",
    preferred_foot: "Left",
    contract_years_left: 7.8,
    club_market_value_eur: 750000000,
    prev_market_value_eur: 180000000,
    minutes_played: 4491,
    appearances: 57,
    goals: 45,
    assists: 10,
    yellow_cards: 2,
    red_cards: 0,
    goals_per90: 0.90,
    assists_per90: 0.20,
    shots_per90: 5.06,
    npxg_per90: 1.52,
  },
  "Bukayo Saka": {
    player_name: "Bukayo Saka",
    age: 25,
    height_cm: 178,
    primary_position: "Attack",
    sub_position: "Right Winger",
    preferred_foot: "Left",
    contract_years_left: 3.8,
    club_market_value_eur: 750000000,
    prev_market_value_eur: 150000000,
    minutes_played: 3279,
    appearances: 51,
    goals: 11,
    assists: 11,
    yellow_cards: 2,
    red_cards: 0,
    goals_per90: 0.30,
    assists_per90: 0.30,
    shots_per90: 1.27,
    npxg_per90: 0.19,
  },
  "Declan Rice": {
    player_name: "Declan Rice",
    age: 27,
    height_cm: 188,
    primary_position: "Midfield",
    sub_position: "Central Midfield",
    preferred_foot: "Right",
    contract_years_left: 1.8,
    club_market_value_eur: 750000000,
    prev_market_value_eur: 120000000,
    minutes_played: 4498,
    appearances: 56,
    goals: 5,
    assists: 12,
    yellow_cards: 7,
    red_cards: 0,
    goals_per90: 0.10,
    assists_per90: 0.24,
    shots_per90: 1.26,
    npxg_per90: 0.03,
  },
  "Virgil van Dijk": {
    player_name: "Virgil van Dijk",
    age: 35,
    height_cm: 195,
    primary_position: "Defender",
    sub_position: "Centre-Back",
    preferred_foot: "Right",
    contract_years_left: 0.8,
    club_market_value_eur: 750000000,
    prev_market_value_eur: 23000000,
    minutes_played: 5211,
    appearances: 58,
    goals: 9,
    assists: 4,
    yellow_cards: 6,
    red_cards: 0,
    goals_per90: 0.16,
    assists_per90: 0.07,
    shots_per90: 0.50,
    npxg_per90: 0.05,
  },
  "Emiliano Martínez": {
    player_name: "Emiliano Martínez",
    age: 34,
    height_cm: 195,
    primary_position: "Goalkeeper",
    sub_position: "Goalkeeper",
    preferred_foot: "Right",
    contract_years_left: 2.8,
    club_market_value_eur: 750000000,
    prev_market_value_eur: 20000000,
    minutes_played: 4050,
    appearances: 46,
    goals: 0,
    assists: 1,
    yellow_cards: 3,
    red_cards: 0,
    goals_per90: 0.0,
    assists_per90: 0.02,
    shots_per90: 0.0,
    npxg_per90: 0.0,
  },
};

const DEFAULT_FORM: TransferPredictRequest = {
  player_name: "",
  age: 24,
  height_cm: 182,
  primary_position: "Attack",
  sub_position: "Centre-Forward",
  preferred_foot: "Right",
  contract_years_left: 3.0,
  club_market_value_eur: 600000000,
  prev_market_value_eur: 30000000,
  minutes_played: 2200,
  appearances: 28,
  goals: 8,
  assists: 4,
  yellow_cards: 3,
  red_cards: 0,
  goals_per90: 0.33,
  assists_per90: 0.16,
  shots_per90: 2.2,
  npxg_per90: 0.28,
};

export const TransferForm: React.FC<TransferFormProps> = ({ onSubmit, loading }) => {
  const [formData, setFormData] = useState<TransferPredictRequest>(DEFAULT_FORM);
  const [presets, setPresets] = useState<Record<string, TransferPredictRequest>>(PRESETS);
  const [errors, setErrors] = useState<Record<string, string>>({});

  useEffect(() => {
    let isCurrent = true;
    fetchTransferPresets()
      .then((res) => {
        if (isCurrent && res?.presets?.length > 0) {
          const dynamicPresets: Record<string, TransferPredictRequest> = {};
          res.presets.forEach((p) => {
            dynamicPresets[p.name] = p.features;
          });
          setPresets(dynamicPresets);
        }
      })
      .catch((err) => {
        console.warn("Using local presets:", err);
      });
    return () => {
      isCurrent = false;
    };
  }, []);

  const validate = (): boolean => {
    const errs: Record<string, string> = {};

    if (formData.age === undefined || formData.age === null || isNaN(formData.age)) {
      errs.age = "Age is required";
    } else if (formData.age < 15 || formData.age > 45) {
      errs.age = "Age must be between 15 and 45";
    }

    if (formData.height_cm != null && !isNaN(formData.height_cm)) {
      if (formData.height_cm < 150 || formData.height_cm > 220) {
        errs.height_cm = "Height must be between 150 and 220 cm";
      }
    }

    if (formData.contract_years_left != null && !isNaN(formData.contract_years_left)) {
      if (formData.contract_years_left < 0 || formData.contract_years_left > 10) {
        errs.contract_years_left = "Contract years left must be between 0 and 10";
      }
    }

    if (formData.club_market_value_eur != null && formData.club_market_value_eur < 0) {
      errs.club_market_value_eur = "Club market value cannot be negative";
    }

    if (formData.prev_market_value_eur != null && formData.prev_market_value_eur < 0) {
      errs.prev_market_value_eur = "Previous market value cannot be negative";
    }

    if (formData.minutes_played != null && formData.minutes_played < 0) {
      errs.minutes_played = "Minutes played cannot be negative";
    }

    if (formData.appearances != null && formData.appearances < 0) {
      errs.appearances = "Appearances cannot be negative";
    }

    if (formData.goals != null && formData.goals < 0) {
      errs.goals = "Goals cannot be negative";
    }

    if (formData.assists != null && formData.assists < 0) {
      errs.assists = "Assists cannot be negative";
    }

    if (formData.yellow_cards != null && formData.yellow_cards < 0) {
      errs.yellow_cards = "Yellow cards cannot be negative";
    }

    if (formData.red_cards != null && formData.red_cards < 0) {
      errs.red_cards = "Red cards cannot be negative";
    }

    if (formData.shots_per90 != null && formData.shots_per90 < 0) {
      errs.shots_per90 = "Shots / 90 cannot be negative";
    }

    if (formData.npxg_per90 != null && formData.npxg_per90 < 0) {
      errs.npxg_per90 = "npxG / 90 cannot be negative";
    }

    setErrors(errs);
    return Object.keys(errs).length === 0;
  };

  const handleChange = (
    field: keyof TransferPredictRequest,
    value: string | number | null
  ) => {
    // Clear error on change
    if (errors[field]) {
      setErrors((prev) => {
        const next = { ...prev };
        delete next[field];
        return next;
      });
    }

    setFormData((prev) => {
      const updated = { ...prev, [field]: value };
      // Auto-compute per90 if goals/minutes changed and fields not manually overridden
      if (field === "goals" || field === "minutes_played") {
        const goals = field === "goals" ? Number(value) : prev.goals || 0;
        const mins = field === "minutes_played" ? Number(value) : prev.minutes_played || 0;
        if (mins > 0) {
          updated.goals_per90 = Number(((goals / mins) * 90).toFixed(2));
        }
      }
      if (field === "assists" || field === "minutes_played") {
        const assists = field === "assists" ? Number(value) : prev.assists || 0;
        const mins = field === "minutes_played" ? Number(value) : prev.minutes_played || 0;
        if (mins > 0) {
          updated.assists_per90 = Number(((assists / mins) * 90).toFixed(2));
        }
      }
      return updated;
    });
  };

  const handleApplyPreset = (presetName: string) => {
    if (presets[presetName]) {
      setFormData({ ...presets[presetName] });
      setErrors({});
    }
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!validate()) {
      return;
    }
    onSubmit(formData);
  };

  const hasErrors = Object.keys(errors).length > 0;

  return (
    <form onSubmit={handleSubmit} style={{ display: "flex", flexDirection: "column", gap: "1.5rem" }} noValidate>
      {/* Quick Presets Bar */}
      <div
        style={{
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          flexWrap: "wrap",
          gap: "0.5rem",
          padding: "0.625rem 1rem",
          backgroundColor: "var(--bg-surface)",
          border: "1px solid var(--border-subtle)",
          borderRadius: "var(--radius-md)",
        }}
      >
        <div style={{ display: "flex", alignItems: "center", gap: "0.6rem", flexWrap: "wrap" }}>
          <div style={{ display: "flex", alignItems: "center", gap: "0.4rem", fontSize: "0.8125rem", color: "var(--text-secondary)" }}>
            <Sparkles size={14} color="var(--accent-tea-green)" />
            <span>Current Player Presets:</span>
          </div>
          <span
            style={{
              fontSize: "0.6875rem",
              color: "var(--accent-blue)",
              backgroundColor: "var(--accent-blue-subtle)",
              border: "1px solid rgba(56, 189, 248, 0.25)",
              borderRadius: "var(--radius-xs)",
              padding: "0.1rem 0.45rem",
              fontWeight: 600,
            }}
          >
            Data: 2026-09-20
          </span>
        </div>
        <div style={{ display: "flex", gap: "0.5rem", flexWrap: "wrap" }}>
          {Object.keys(presets).map((name) => (
            <button
              key={name}
              type="button"
              onClick={() => handleApplyPreset(name)}
              className="btn btn-secondary"
              style={{ padding: "0.25rem 0.625rem", fontSize: "0.75rem" }}
            >
              {name}
            </button>
          ))}
          <button
            type="button"
            onClick={() => {
              setFormData(DEFAULT_FORM);
              setErrors({});
            }}
            className="btn btn-ghost"
            style={{ padding: "0.25rem 0.625rem", fontSize: "0.75rem", gap: "0.25rem" }}
            title="Reset form"
          >
            <RotateCcw size={12} />
            <span>Reset</span>
          </button>
        </div>
      </div>

      {/* Validation Alert */}
      {hasErrors && (
        <div
          role="alert"
          style={{
            display: "flex",
            alignItems: "center",
            gap: "0.75rem",
            padding: "0.875rem 1rem",
            borderRadius: "var(--radius-md)",
            backgroundColor: "rgba(244, 63, 94, 0.12)",
            border: "1px solid rgba(244, 63, 94, 0.3)",
            color: "#fb7185",
            fontSize: "0.875rem",
          }}
        >
          <AlertCircle size={16} />
          <span>Please check the highlighted fields below before predicting.</span>
        </div>
      )}

      {/* Section 1: Player Context */}
      <div className="card">
        <div className="card-header">
          <div>
            <h4 className="card-title">
              <User size={18} color="var(--accent-tea-green)" />
              Player Demographics &amp; Role
            </h4>
            <p className="card-subtitle">Primary profile attributes and physical dimensions</p>
          </div>
        </div>

        <div className="grid-3">
          <div className="form-group">
            <label className="form-label" htmlFor="tv-player-name">Player Name</label>
            <input
              id="tv-player-name"
              type="text"
              className="form-input"
              placeholder="e.g. Cole Palmer"
              value={formData.player_name || ""}
              onChange={(e) => handleChange("player_name", e.target.value)}
            />
          </div>

          <div className="form-group">
            <label className="form-label" htmlFor="tv-age">Age (years) *</label>
            <input
              id="tv-age"
              type="number"
              min={15}
              max={45}
              required
              className="form-input"
              style={errors.age ? { borderColor: "var(--accent-rose)" } : undefined}
              value={formData.age}
              onChange={(e) => handleChange("age", Number(e.target.value))}
            />
            {errors.age && (
              <span style={{ fontSize: "0.75rem", color: "#fb7185" }}>{errors.age}</span>
            )}
          </div>

          <div className="form-group">
            <label className="form-label" htmlFor="tv-height">Height (cm)</label>
            <input
              id="tv-height"
              type="number"
              min={150}
              max={220}
              className="form-input"
              style={errors.height_cm ? { borderColor: "var(--accent-rose)" } : undefined}
              value={formData.height_cm || ""}
              onChange={(e) => handleChange("height_cm", Number(e.target.value) || null)}
            />
            {errors.height_cm && (
              <span style={{ fontSize: "0.75rem", color: "#fb7185" }}>{errors.height_cm}</span>
            )}
          </div>
        </div>

        <div className="grid-3" style={{ marginTop: "0.5rem" }}>
          <div className="form-group">
            <label className="form-label" htmlFor="tv-pos">Primary Position</label>
            <select
              id="tv-pos"
              className="form-select"
              value={formData.primary_position || "Attack"}
              onChange={(e) => handleChange("primary_position", e.target.value)}
            >
              <option value="Attack">Attack</option>
              <option value="Midfield">Midfield</option>
              <option value="Defender">Defender</option>
              <option value="Goalkeeper">Goalkeeper</option>
            </select>
          </div>

          <div className="form-group">
            <label className="form-label" htmlFor="tv-sub-pos">Sub-Position</label>
            <select
              id="tv-sub-pos"
              className="form-select"
              value={formData.sub_position || "Centre-Forward"}
              onChange={(e) => handleChange("sub_position", e.target.value)}
            >
              <option value="Centre-Forward">Centre-Forward</option>
              <option value="Second Striker">Second Striker</option>
              <option value="Left Winger">Left Winger</option>
              <option value="Right Winger">Right Winger</option>
              <option value="Attacking Midfield">Attacking Midfield</option>
              <option value="Central Midfield">Central Midfield</option>
              <option value="Defensive Midfield">Defensive Midfield</option>
              <option value="Centre-Back">Centre-Back</option>
              <option value="Left-Back">Left-Back</option>
              <option value="Right-Back">Right-Back</option>
              <option value="Goalkeeper">Goalkeeper</option>
            </select>
          </div>

          <div className="form-group">
            <label className="form-label" htmlFor="tv-foot">Preferred Foot</label>
            <select
              id="tv-foot"
              className="form-select"
              value={formData.preferred_foot || "Right"}
              onChange={(e) => handleChange("preferred_foot", e.target.value)}
            >
              <option value="Right">Right</option>
              <option value="Left">Left</option>
              <option value="Both">Both</option>
            </select>
          </div>
        </div>
      </div>

      {/* Section 2: Contract / Club */}
      <div className="card">
        <div className="card-header">
          <div>
            <h4 className="card-title">
              <Building size={18} color="var(--accent-blue)" />
              Contract &amp; Club Stature
            </h4>
            <p className="card-subtitle">Economic factors, remaining tenure, and prior valuation</p>
          </div>
        </div>

        <div className="grid-3">
          <div className="form-group">
            <label className="form-label" htmlFor="tv-contract">Contract Years Left</label>
            <input
              id="tv-contract"
              type="number"
              step="0.5"
              min="0"
              max="10"
              className="form-input"
              style={errors.contract_years_left ? { borderColor: "var(--accent-rose)" } : undefined}
              value={formData.contract_years_left ?? ""}
              onChange={(e) => handleChange("contract_years_left", Number(e.target.value))}
            />
            {errors.contract_years_left && (
              <span style={{ fontSize: "0.75rem", color: "#fb7185" }}>{errors.contract_years_left}</span>
            )}
          </div>

          <div className="form-group">
            <label className="form-label" htmlFor="tv-club-val">Club Market Value (€)</label>
            <input
              id="tv-club-val"
              type="number"
              step="10000000"
              className="form-input"
              style={errors.club_market_value_eur ? { borderColor: "var(--accent-rose)" } : undefined}
              placeholder="e.g. 800000000"
              value={formData.club_market_value_eur ?? ""}
              onChange={(e) => handleChange("club_market_value_eur", Number(e.target.value))}
            />
            {errors.club_market_value_eur ? (
              <span style={{ fontSize: "0.75rem", color: "#fb7185" }}>{errors.club_market_value_eur}</span>
            ) : (
              <span style={{ fontSize: "0.75rem", color: "var(--text-muted)", marginTop: "2px" }}>
                {formData.club_market_value_eur
                  ? `€${(formData.club_market_value_eur / 1e6).toFixed(1)}M Squad Value`
                  : "Total squad Transfermarkt valuation"}
              </span>
            )}
          </div>

          <div className="form-group">
            <label className="form-label" htmlFor="tv-prev-val">Previous Market Value (€)</label>
            <input
              id="tv-prev-val"
              type="number"
              step="1000000"
              className="form-input"
              style={errors.prev_market_value_eur ? { borderColor: "var(--accent-rose)" } : undefined}
              placeholder="e.g. 50000000"
              value={formData.prev_market_value_eur ?? ""}
              onChange={(e) => handleChange("prev_market_value_eur", Number(e.target.value))}
            />
            {errors.prev_market_value_eur ? (
              <span style={{ fontSize: "0.75rem", color: "#fb7185" }}>{errors.prev_market_value_eur}</span>
            ) : (
              <span style={{ fontSize: "0.75rem", color: "var(--text-muted)", marginTop: "2px" }}>
                {formData.prev_market_value_eur
                  ? `€${(formData.prev_market_value_eur / 1e6).toFixed(1)}M Prior Valuation`
                  : "Baseline historical value"}
              </span>
            )}
          </div>
        </div>
      </div>

      {/* Section 3: Performance Metrics */}
      <div className="card">
        <div className="card-header">
          <div>
            <h4 className="card-title">
              <Activity size={18} color="var(--accent-amber)" />
              Performance &amp; Volume Metrics
            </h4>
            <p className="card-subtitle">Season playing time, goal contributions, and per-90 rates</p>
          </div>
        </div>

        <div className="grid-4">
          <div className="form-group">
            <label className="form-label" htmlFor="tv-minutes">Minutes Played</label>
            <input
              id="tv-minutes"
              type="number"
              min="0"
              max="5000"
              className="form-input"
              style={errors.minutes_played ? { borderColor: "var(--accent-rose)" } : undefined}
              value={formData.minutes_played ?? ""}
              onChange={(e) => handleChange("minutes_played", Number(e.target.value))}
            />
            {errors.minutes_played && (
              <span style={{ fontSize: "0.75rem", color: "#fb7185" }}>{errors.minutes_played}</span>
            )}
          </div>

          <div className="form-group">
            <label className="form-label" htmlFor="tv-appearances">Appearances</label>
            <input
              id="tv-appearances"
              type="number"
              min="0"
              max="65"
              className="form-input"
              style={errors.appearances ? { borderColor: "var(--accent-rose)" } : undefined}
              value={formData.appearances ?? ""}
              onChange={(e) => handleChange("appearances", Number(e.target.value))}
            />
            {errors.appearances && (
              <span style={{ fontSize: "0.75rem", color: "#fb7185" }}>{errors.appearances}</span>
            )}
          </div>

          <div className="form-group">
            <label className="form-label" htmlFor="tv-goals">Goals</label>
            <input
              id="tv-goals"
              type="number"
              min="0"
              max="80"
              className="form-input"
              style={errors.goals ? { borderColor: "var(--accent-rose)" } : undefined}
              value={formData.goals ?? 0}
              onChange={(e) => handleChange("goals", Number(e.target.value))}
            />
            {errors.goals && (
              <span style={{ fontSize: "0.75rem", color: "#fb7185" }}>{errors.goals}</span>
            )}
          </div>

          <div className="form-group">
            <label className="form-label" htmlFor="tv-assists">Assists</label>
            <input
              id="tv-assists"
              type="number"
              min="0"
              max="50"
              className="form-input"
              style={errors.assists ? { borderColor: "var(--accent-rose)" } : undefined}
              value={formData.assists ?? 0}
              onChange={(e) => handleChange("assists", Number(e.target.value))}
            />
            {errors.assists && (
              <span style={{ fontSize: "0.75rem", color: "#fb7185" }}>{errors.assists}</span>
            )}
          </div>
        </div>

        <div className="grid-4" style={{ marginTop: "0.5rem" }}>
          <div className="form-group">
            <label className="form-label" htmlFor="tv-goals90">Goals / 90</label>
            <input
              id="tv-goals90"
              type="number"
              step="0.01"
              min="0"
              className="form-input"
              value={formData.goals_per90 ?? ""}
              onChange={(e) => handleChange("goals_per90", Number(e.target.value))}
            />
          </div>

          <div className="form-group">
            <label className="form-label" htmlFor="tv-assists90">Assists / 90</label>
            <input
              id="tv-assists90"
              type="number"
              step="0.01"
              min="0"
              className="form-input"
              value={formData.assists_per90 ?? ""}
              onChange={(e) => handleChange("assists_per90", Number(e.target.value))}
            />
          </div>

          <div className="form-group">
            <label className="form-label" htmlFor="tv-shots90">Shots / 90</label>
            <input
              id="tv-shots90"
              type="number"
              step="0.01"
              min="0"
              className="form-input"
              style={errors.shots_per90 ? { borderColor: "var(--accent-rose)" } : undefined}
              value={formData.shots_per90 ?? ""}
              onChange={(e) => handleChange("shots_per90", Number(e.target.value))}
            />
            {errors.shots_per90 && (
              <span style={{ fontSize: "0.75rem", color: "#fb7185" }}>{errors.shots_per90}</span>
            )}
          </div>

          <div className="form-group">
            <label className="form-label" htmlFor="tv-npxg90">npxG / 90</label>
            <input
              id="tv-npxg90"
              type="number"
              step="0.01"
              min="0"
              className="form-input"
              style={errors.npxg_per90 ? { borderColor: "var(--accent-rose)" } : undefined}
              value={formData.npxg_per90 ?? ""}
              onChange={(e) => handleChange("npxg_per90", Number(e.target.value))}
            />
            {errors.npxg_per90 && (
              <span style={{ fontSize: "0.75rem", color: "#fb7185" }}>{errors.npxg_per90}</span>
            )}
          </div>
        </div>

        <div className="grid-2" style={{ marginTop: "0.5rem" }}>
          <div className="form-group">
            <label className="form-label" htmlFor="tv-yc">Yellow Cards</label>
            <input
              id="tv-yc"
              type="number"
              min="0"
              className="form-input"
              style={errors.yellow_cards ? { borderColor: "var(--accent-rose)" } : undefined}
              value={formData.yellow_cards ?? 0}
              onChange={(e) => handleChange("yellow_cards", Number(e.target.value))}
            />
            {errors.yellow_cards && (
              <span style={{ fontSize: "0.75rem", color: "#fb7185" }}>{errors.yellow_cards}</span>
            )}
          </div>

          <div className="form-group">
            <label className="form-label" htmlFor="tv-rc">Red Cards</label>
            <input
              id="tv-rc"
              type="number"
              min="0"
              className="form-input"
              style={errors.red_cards ? { borderColor: "var(--accent-rose)" } : undefined}
              value={formData.red_cards ?? 0}
              onChange={(e) => handleChange("red_cards", Number(e.target.value))}
            />
            {errors.red_cards && (
              <span style={{ fontSize: "0.75rem", color: "#fb7185" }}>{errors.red_cards}</span>
            )}
          </div>
        </div>
      </div>

      {/* Submit Button */}
      <div style={{ display: "flex", justifyContent: "flex-end" }}>
        <button
          type="submit"
          disabled={loading}
          className="btn btn-primary"
          style={{ padding: "0.875rem 2rem", fontSize: "0.9375rem" }}
          aria-busy={loading}
        >
          {loading ? (
            <>
              <div className="animate-spin" style={{ width: "16px", height: "16px", border: "2px solid #151515", borderTopColor: "transparent", borderRadius: "50%" }} />
              Predicting...
            </>
          ) : (
            "Predict Market Value"
          )}
        </button>
      </div>
    </form>
  );
};
