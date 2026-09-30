import React, { useState } from "react";
import type { MatchPredictRequest } from "../../types/api";
import {
  Swords,
  Sparkles,
  RotateCcw,
  ArrowRightLeft,
  TrendingUp,
  Calendar,
  History,
  Trophy,
  AlertCircle,
} from "lucide-react";

interface MatchFormProps {
  onSubmit: (data: MatchPredictRequest) => void;
  loading: boolean;
}

const PREMIER_LEAGUE_TEAMS = [
  "Arsenal",
  "Aston Villa",
  "Bournemouth",
  "Brentford",
  "Brighton",
  "Chelsea",
  "Crystal Palace",
  "Everton",
  "Fulham",
  "Hull",
  "Ipswich Town",
  "Leeds",
  "Leicester City",
  "Liverpool",
  "Manchester City",
  "Manchester United",
  "Newcastle United",
  "Nottingham Forest",
  "Southampton",
  "Tottenham Hotspur",
  "West Ham United",
  "Wolverhampton Wanderers",
];

const PRESETS: Record<string, MatchPredictRequest> = {
  "Arsenal vs Chelsea": {
    home_team: "Arsenal",
    away_team: "Chelsea",
    home_rest_days: 7,
    away_rest_days: 7,
    home_rolling_points_5: 12.0,
    away_rolling_points_5: 7.0,
    home_rolling_gf_5: 8.0,
    away_rolling_gf_5: 6.0,
    home_rolling_ga_5: 4.0,
    away_rolling_ga_5: 8.0,
    home_rolling_shots_5: 13.0,
    away_rolling_shots_5: 13.5,
    home_rolling_sot_5: 4.8,
    away_rolling_sot_5: 4.5,
    home_rolling_sot_conceded_5: 2.8,
    away_rolling_sot_conceded_5: 4.2,
    home_rolling_corners_5: 5.0,
    away_rolling_corners_5: 5.2,
    home_rolling_win_pct_5: 0.80,
    away_rolling_win_pct_5: 0.40,
    home_league_rank: 2,
    away_league_rank: 10,
    home_ppg: 2.40,
    away_ppg: 1.40,
    h2h_matches_prior: 10,
    h2h_home_win_pct: 0.60,
  },
  "Man City vs Liverpool": {
    home_team: "Manchester City",
    away_team: "Liverpool",
    home_rest_days: 7,
    away_rest_days: 7,
    home_rolling_points_5: 15.0,
    away_rolling_points_5: 9.0,
    home_rolling_gf_5: 14.0,
    away_rolling_gf_5: 9.0,
    home_rolling_ga_5: 3.0,
    away_rolling_ga_5: 6.0,
    home_rolling_shots_5: 18.2,
    away_rolling_shots_5: 16.0,
    home_rolling_sot_5: 7.4,
    away_rolling_sot_5: 5.8,
    home_rolling_sot_conceded_5: 2.4,
    away_rolling_sot_conceded_5: 3.6,
    home_rolling_corners_5: 7.8,
    away_rolling_corners_5: 6.2,
    home_rolling_win_pct_5: 1.0,
    away_rolling_win_pct_5: 0.60,
    home_league_rank: 1,
    away_league_rank: 7,
    home_ppg: 3.00,
    away_ppg: 1.80,
    h2h_matches_prior: 12,
    h2h_home_win_pct: 0.50,
  },
  "Aston Villa vs Tottenham": {
    home_team: "Aston Villa",
    away_team: "Tottenham Hotspur",
    home_rest_days: 7,
    away_rest_days: 7,
    home_rolling_points_5: 8.0,
    away_rolling_points_5: 7.0,
    home_rolling_gf_5: 7.0,
    away_rolling_gf_5: 8.0,
    home_rolling_ga_5: 6.0,
    away_rolling_ga_5: 7.0,
    home_rolling_shots_5: 14.0,
    away_rolling_shots_5: 14.8,
    home_rolling_sot_5: 5.0,
    away_rolling_sot_5: 5.2,
    home_rolling_sot_conceded_5: 3.8,
    away_rolling_sot_conceded_5: 4.4,
    home_rolling_corners_5: 5.8,
    away_rolling_corners_5: 5.6,
    home_rolling_win_pct_5: 0.50,
    away_rolling_win_pct_5: 0.40,
    home_league_rank: 11,
    away_league_rank: 12,
    home_ppg: 1.60,
    away_ppg: 1.40,
    h2h_matches_prior: 8,
    h2h_home_win_pct: 0.45,
  },
};

const DEFAULT_MATCH_FORM: MatchPredictRequest = {
  home_team: "Arsenal",
  away_team: "Chelsea",
  home_rest_days: 6,
  away_rest_days: 7,
  home_rolling_points_5: 12,
  away_rolling_points_5: 8,
  home_rolling_gf_5: 10,
  away_rolling_gf_5: 6,
  home_rolling_ga_5: 4,
  away_rolling_ga_5: 7,
  home_rolling_shots_5: 15.4,
  away_rolling_shots_5: 12.8,
  home_rolling_sot_5: 5.8,
  away_rolling_sot_5: 4.2,
  home_rolling_sot_conceded_5: 3.0,
  away_rolling_sot_conceded_5: 4.4,
  home_rolling_corners_5: 6.8,
  away_rolling_corners_5: 5.0,
  home_rolling_win_pct_5: 0.8,
  away_rolling_win_pct_5: 0.4,
  home_league_rank: 2,
  away_league_rank: 8,
  home_ppg: 2.15,
  away_ppg: 1.50,
  h2h_matches_prior: 6,
  h2h_home_win_pct: 0.50,
};

export const MatchForm: React.FC<MatchFormProps> = ({ onSubmit, loading }) => {
  const [formData, setFormData] = useState<MatchPredictRequest>(DEFAULT_MATCH_FORM);
  const [errors, setErrors] = useState<Record<string, string>>({});

  const validate = (): boolean => {
    const errs: Record<string, string> = {};

    if (!formData.home_team || !formData.away_team) {
      errs.teams = "Both home and away teams are required";
    } else if (formData.home_team === formData.away_team) {
      errs.teams = "Home and away clubs must be different teams";
    }

    if (formData.home_rest_days < 0 || formData.home_rest_days > 60) {
      errs.home_rest_days = "Home rest days must be between 0 and 60";
    }
    if (formData.away_rest_days < 0 || formData.away_rest_days > 60) {
      errs.away_rest_days = "Away rest days must be between 0 and 60";
    }

    if (formData.home_league_rank < 1 || formData.home_league_rank > 20) {
      errs.home_league_rank = "Home league rank must be between 1 and 20";
    }
    if (formData.away_league_rank < 1 || formData.away_league_rank > 20) {
      errs.away_league_rank = "Away league rank must be between 1 and 20";
    }

    if (formData.home_ppg < 0 || formData.home_ppg > 3.0) {
      errs.home_ppg = "Home PPG must be between 0.0 and 3.0";
    }
    if (formData.away_ppg < 0 || formData.away_ppg > 3.0) {
      errs.away_ppg = "Away PPG must be between 0.0 and 3.0";
    }

    if (formData.home_rolling_win_pct_5 < 0 || formData.home_rolling_win_pct_5 > 1.0) {
      errs.home_rolling_win_pct_5 = "Win % must be between 0.0 and 1.0";
    }
    if (formData.away_rolling_win_pct_5 < 0 || formData.away_rolling_win_pct_5 > 1.0) {
      errs.away_rolling_win_pct_5 = "Win % must be between 0.0 and 1.0";
    }

    if (formData.home_rolling_points_5 < 0 || formData.home_rolling_points_5 > 15) {
      errs.home_rolling_points_5 = "Points (last 5) must be between 0 and 15";
    }
    if (formData.away_rolling_points_5 < 0 || formData.away_rolling_points_5 > 15) {
      errs.away_rolling_points_5 = "Points (last 5) must be between 0 and 15";
    }

    if (formData.h2h_matches_prior != null && formData.h2h_matches_prior < 0) {
      errs.h2h_matches_prior = "Prior H2H matches cannot be negative";
    }
    if (formData.h2h_home_win_pct != null && (formData.h2h_home_win_pct < 0 || formData.h2h_home_win_pct > 1.0)) {
      errs.h2h_home_win_pct = "H2H Home Win % must be between 0.0 and 1.0";
    }

    // Non-negative checks
    if (formData.home_rolling_gf_5 < 0) errs.home_rolling_gf_5 = "Cannot be negative";
    if (formData.away_rolling_gf_5 < 0) errs.away_rolling_gf_5 = "Cannot be negative";
    if (formData.home_rolling_ga_5 < 0) errs.home_rolling_ga_5 = "Cannot be negative";
    if (formData.away_rolling_ga_5 < 0) errs.away_rolling_ga_5 = "Cannot be negative";
    if (formData.home_rolling_shots_5 < 0) errs.home_rolling_shots_5 = "Cannot be negative";
    if (formData.away_rolling_shots_5 < 0) errs.away_rolling_shots_5 = "Cannot be negative";
    if (formData.home_rolling_sot_5 < 0) errs.home_rolling_sot_5 = "Cannot be negative";
    if (formData.away_rolling_sot_5 < 0) errs.away_rolling_sot_5 = "Cannot be negative";
    if (formData.home_rolling_sot_conceded_5 < 0) errs.home_rolling_sot_conceded_5 = "Cannot be negative";
    if (formData.away_rolling_sot_conceded_5 < 0) errs.away_rolling_sot_conceded_5 = "Cannot be negative";
    if (formData.home_rolling_corners_5 < 0) errs.home_rolling_corners_5 = "Cannot be negative";
    if (formData.away_rolling_corners_5 < 0) errs.away_rolling_corners_5 = "Cannot be negative";

    setErrors(errs);
    return Object.keys(errs).length === 0;
  };

  const handleChange = (
    field: keyof MatchPredictRequest,
    value: string | number
  ) => {
    if (errors[field] || (errors.teams && (field === "home_team" || field === "away_team"))) {
      setErrors((prev) => {
        const next = { ...prev };
        delete next[field];
        if (field === "home_team" || field === "away_team") delete next.teams;
        return next;
      });
    }

    setFormData((prev) => ({
      ...prev,
      [field]: value,
    }));
  };

  const handleSwapTeams = () => {
    setFormData((prev) => ({
      ...prev,
      home_team: prev.away_team,
      away_team: prev.home_team,
      home_rest_days: prev.away_rest_days,
      away_rest_days: prev.home_rest_days,
      home_rolling_points_5: prev.away_rolling_points_5,
      away_rolling_points_5: prev.home_rolling_points_5,
      home_rolling_gf_5: prev.away_rolling_gf_5,
      away_rolling_gf_5: prev.home_rolling_gf_5,
      home_rolling_ga_5: prev.away_rolling_ga_5,
      away_rolling_ga_5: prev.home_rolling_ga_5,
      home_rolling_shots_5: prev.away_rolling_shots_5,
      away_rolling_shots_5: prev.home_rolling_shots_5,
      home_rolling_sot_5: prev.away_rolling_sot_5,
      away_rolling_sot_5: prev.home_rolling_sot_5,
      home_rolling_sot_conceded_5: prev.away_rolling_sot_conceded_5,
      away_rolling_sot_conceded_5: prev.home_rolling_sot_conceded_5,
      home_rolling_corners_5: prev.away_rolling_corners_5,
      away_rolling_corners_5: prev.home_rolling_corners_5,
      home_rolling_win_pct_5: prev.away_rolling_win_pct_5,
      away_rolling_win_pct_5: prev.home_rolling_win_pct_5,
      home_league_rank: prev.away_league_rank,
      away_league_rank: prev.home_league_rank,
      home_ppg: prev.away_ppg,
      away_ppg: prev.home_ppg,
      h2h_home_win_pct: prev.h2h_home_win_pct ? 1 - prev.h2h_home_win_pct : 0.5,
    }));
    setErrors({});
  };

  const handleApplyPreset = (name: string) => {
    if (PRESETS[name]) {
      setFormData({ ...PRESETS[name] });
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
            <Sparkles size={14} color="var(--accent-blue)" />
            <span>2026/27 Matchday Presets:</span>
          </div>
          <span
            style={{
              fontSize: "0.6875rem",
              color: "#38bdf8",
              backgroundColor: "rgba(56, 189, 248, 0.12)",
              border: "1px solid rgba(56, 189, 248, 0.3)",
              borderRadius: "9999px",
              padding: "0.1rem 0.45rem",
              fontWeight: 600,
            }}
          >
            Data as of: 2026-09-20
          </span>
        </div>
        <div style={{ display: "flex", gap: "0.5rem", flexWrap: "wrap" }}>
          {Object.keys(PRESETS).map((name) => (
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
              setFormData(DEFAULT_MATCH_FORM);
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
          <span>
            {errors.teams || "Please correct the highlighted inputs before predicting."}
          </span>
        </div>
      )}

      {/* Section 1: Match Teams */}
      <div className="card">
        <div className="card-header">
          <div>
            <h4 className="card-title">
              <Swords size={18} color="var(--accent-blue)" />
              Match Fixture
            </h4>
            <p className="card-subtitle">Select home and away clubs for the match simulation</p>
          </div>
          <button
            type="button"
            onClick={handleSwapTeams}
            className="btn btn-secondary"
            style={{ padding: "0.35rem 0.75rem", fontSize: "0.75rem", gap: "0.35rem" }}
            title="Swap Home and Away teams"
          >
            <ArrowRightLeft size={13} />
            <span>Swap Venue</span>
          </button>
        </div>

        <div className="grid-2">
          <div className="form-group">
            <label className="form-label" htmlFor="m-home-team">Home Team</label>
            <select
              id="m-home-team"
              className="form-select"
              style={errors.teams ? { borderColor: "var(--accent-rose)" } : undefined}
              value={formData.home_team || "Arsenal"}
              onChange={(e) => handleChange("home_team", e.target.value)}
            >
              {PREMIER_LEAGUE_TEAMS.map((team) => (
                <option key={team} value={team}>{team}</option>
              ))}
            </select>
          </div>

          <div className="form-group">
            <label className="form-label" htmlFor="m-away-team">Away Team</label>
            <select
              id="m-away-team"
              className="form-select"
              style={errors.teams ? { borderColor: "var(--accent-rose)" } : undefined}
              value={formData.away_team || "Chelsea"}
              onChange={(e) => handleChange("away_team", e.target.value)}
            >
              {PREMIER_LEAGUE_TEAMS.map((team) => (
                <option key={team} value={team}>{team}</option>
              ))}
            </select>
          </div>
        </div>
        {errors.teams && (
          <span style={{ fontSize: "0.75rem", color: "#fb7185", marginTop: "0.25rem", display: "block" }}>
            {errors.teams}
          </span>
        )}
      </div>

      {/* Section 2: Recent Form (5-match rolling) */}
      <div className="card">
        <div className="card-header">
          <div>
            <h4 className="card-title">
              <TrendingUp size={18} color="var(--accent-tea-green)" />
              Recent Form (5-Match Rolling Statistics)
            </h4>
            <p className="card-subtitle">Performance volume and momentum leading into the fixture</p>
          </div>
        </div>

        {/* Home & Away Rolling Columns */}
        <div className="grid-2">
          {/* Home 5-Match Form */}
          <div
            style={{
              padding: "1rem",
              borderRadius: "var(--radius-md)",
              backgroundColor: "rgba(193, 237, 204, 0.05)",
              border: "1px solid rgba(193, 237, 204, 0.2)",
            }}
          >
            <h5 style={{ fontSize: "0.875rem", color: "var(--accent-tea-green)", marginBottom: "0.75rem" }}>
              {formData.home_team} (Home Form)
            </h5>

            <div className="grid-2">
              <div className="form-group">
                <label className="form-label" htmlFor="m-h-pts">Points (Last 5)</label>
                <input
                  id="m-h-pts"
                  type="number"
                  min="0"
                  max="15"
                  className="form-input"
                  style={errors.home_rolling_points_5 ? { borderColor: "var(--accent-rose)" } : undefined}
                  value={formData.home_rolling_points_5}
                  onChange={(e) => handleChange("home_rolling_points_5", Number(e.target.value))}
                />
                {errors.home_rolling_points_5 && (
                  <span style={{ fontSize: "0.75rem", color: "#fb7185" }}>{errors.home_rolling_points_5}</span>
                )}
              </div>
              <div className="form-group">
                <label className="form-label" htmlFor="m-h-winpct">Win Rate (0 - 1.0)</label>
                <input
                  id="m-h-winpct"
                  type="number"
                  step="0.05"
                  min="0"
                  max="1"
                  className="form-input"
                  style={errors.home_rolling_win_pct_5 ? { borderColor: "var(--accent-rose)" } : undefined}
                  value={formData.home_rolling_win_pct_5}
                  onChange={(e) => handleChange("home_rolling_win_pct_5", Number(e.target.value))}
                />
                {errors.home_rolling_win_pct_5 && (
                  <span style={{ fontSize: "0.75rem", color: "#fb7185" }}>{errors.home_rolling_win_pct_5}</span>
                )}
              </div>
            </div>

            <div className="grid-2">
              <div className="form-group">
                <label className="form-label" htmlFor="m-h-gf">Goals For / 5</label>
                <input
                  id="m-h-gf"
                  type="number"
                  min="0"
                  className="form-input"
                  style={errors.home_rolling_gf_5 ? { borderColor: "var(--accent-rose)" } : undefined}
                  value={formData.home_rolling_gf_5}
                  onChange={(e) => handleChange("home_rolling_gf_5", Number(e.target.value))}
                />
              </div>
              <div className="form-group">
                <label className="form-label" htmlFor="m-h-ga">Goals Against / 5</label>
                <input
                  id="m-h-ga"
                  type="number"
                  min="0"
                  className="form-input"
                  style={errors.home_rolling_ga_5 ? { borderColor: "var(--accent-rose)" } : undefined}
                  value={formData.home_rolling_ga_5}
                  onChange={(e) => handleChange("home_rolling_ga_5", Number(e.target.value))}
                />
              </div>
            </div>

            <div className="grid-2">
              <div className="form-group">
                <label className="form-label" htmlFor="m-h-shots">Shots / Match</label>
                <input
                  id="m-h-shots"
                  type="number"
                  step="0.1"
                  min="0"
                  className="form-input"
                  value={formData.home_rolling_shots_5}
                  onChange={(e) => handleChange("home_rolling_shots_5", Number(e.target.value))}
                />
              </div>
              <div className="form-group">
                <label className="form-label" htmlFor="m-h-sot">Shots on Target</label>
                <input
                  id="m-h-sot"
                  type="number"
                  step="0.1"
                  min="0"
                  className="form-input"
                  value={formData.home_rolling_sot_5}
                  onChange={(e) => handleChange("home_rolling_sot_5", Number(e.target.value))}
                />
              </div>
            </div>

            <div className="grid-2">
              <div className="form-group">
                <label className="form-label" htmlFor="m-h-sot-conc">SoT Conceded</label>
                <input
                  id="m-h-sot-conc"
                  type="number"
                  step="0.1"
                  min="0"
                  className="form-input"
                  value={formData.home_rolling_sot_conceded_5}
                  onChange={(e) => handleChange("home_rolling_sot_conceded_5", Number(e.target.value))}
                />
              </div>
              <div className="form-group">
                <label className="form-label" htmlFor="m-h-corners">Corners / Match</label>
                <input
                  id="m-h-corners"
                  type="number"
                  step="0.1"
                  min="0"
                  className="form-input"
                  value={formData.home_rolling_corners_5}
                  onChange={(e) => handleChange("home_rolling_corners_5", Number(e.target.value))}
                />
              </div>
            </div>
          </div>

          {/* Away 5-Match Form */}
          <div
            style={{
              padding: "1rem",
              borderRadius: "var(--radius-md)",
              backgroundColor: "rgba(56, 189, 248, 0.04)",
              border: "1px solid rgba(56, 189, 248, 0.15)",
            }}
          >
            <h5 style={{ fontSize: "0.875rem", color: "var(--accent-blue)", marginBottom: "0.75rem" }}>
              {formData.away_team} (Away Form)
            </h5>

            <div className="grid-2">
              <div className="form-group">
                <label className="form-label" htmlFor="m-a-pts">Points (Last 5)</label>
                <input
                  id="m-a-pts"
                  type="number"
                  min="0"
                  max="15"
                  className="form-input"
                  style={errors.away_rolling_points_5 ? { borderColor: "var(--accent-rose)" } : undefined}
                  value={formData.away_rolling_points_5}
                  onChange={(e) => handleChange("away_rolling_points_5", Number(e.target.value))}
                />
                {errors.away_rolling_points_5 && (
                  <span style={{ fontSize: "0.75rem", color: "#fb7185" }}>{errors.away_rolling_points_5}</span>
                )}
              </div>
              <div className="form-group">
                <label className="form-label" htmlFor="m-a-winpct">Win Rate (0 - 1.0)</label>
                <input
                  id="m-a-winpct"
                  type="number"
                  step="0.05"
                  min="0"
                  max="1"
                  className="form-input"
                  style={errors.away_rolling_win_pct_5 ? { borderColor: "var(--accent-rose)" } : undefined}
                  value={formData.away_rolling_win_pct_5}
                  onChange={(e) => handleChange("away_rolling_win_pct_5", Number(e.target.value))}
                />
                {errors.away_rolling_win_pct_5 && (
                  <span style={{ fontSize: "0.75rem", color: "#fb7185" }}>{errors.away_rolling_win_pct_5}</span>
                )}
              </div>
            </div>

            <div className="grid-2">
              <div className="form-group">
                <label className="form-label" htmlFor="m-a-gf">Goals For / 5</label>
                <input
                  id="m-a-gf"
                  type="number"
                  min="0"
                  className="form-input"
                  style={errors.away_rolling_gf_5 ? { borderColor: "var(--accent-rose)" } : undefined}
                  value={formData.away_rolling_gf_5}
                  onChange={(e) => handleChange("away_rolling_gf_5", Number(e.target.value))}
                />
              </div>
              <div className="form-group">
                <label className="form-label" htmlFor="m-a-ga">Goals Against / 5</label>
                <input
                  id="m-a-ga"
                  type="number"
                  min="0"
                  className="form-input"
                  style={errors.away_rolling_ga_5 ? { borderColor: "var(--accent-rose)" } : undefined}
                  value={formData.away_rolling_ga_5}
                  onChange={(e) => handleChange("away_rolling_ga_5", Number(e.target.value))}
                />
              </div>
            </div>

            <div className="grid-2">
              <div className="form-group">
                <label className="form-label" htmlFor="m-a-shots">Shots / Match</label>
                <input
                  id="m-a-shots"
                  type="number"
                  step="0.1"
                  min="0"
                  className="form-input"
                  value={formData.away_rolling_shots_5}
                  onChange={(e) => handleChange("away_rolling_shots_5", Number(e.target.value))}
                />
              </div>
              <div className="form-group">
                <label className="form-label" htmlFor="m-a-sot">Shots on Target</label>
                <input
                  id="m-a-sot"
                  type="number"
                  step="0.1"
                  min="0"
                  className="form-input"
                  value={formData.away_rolling_sot_5}
                  onChange={(e) => handleChange("away_rolling_sot_5", Number(e.target.value))}
                />
              </div>
            </div>

            <div className="grid-2">
              <div className="form-group">
                <label className="form-label" htmlFor="m-a-sot-conc">SoT Conceded</label>
                <input
                  id="m-a-sot-conc"
                  type="number"
                  step="0.1"
                  min="0"
                  className="form-input"
                  value={formData.away_rolling_sot_conceded_5}
                  onChange={(e) => handleChange("away_rolling_sot_conceded_5", Number(e.target.value))}
                />
              </div>
              <div className="form-group">
                <label className="form-label" htmlFor="m-a-corners">Corners / Match</label>
                <input
                  id="m-a-corners"
                  type="number"
                  step="0.1"
                  min="0"
                  className="form-input"
                  value={formData.away_rolling_corners_5}
                  onChange={(e) => handleChange("away_rolling_corners_5", Number(e.target.value))}
                />
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Section 3: League Strength */}
      <div className="card">
        <div className="card-header">
          <div>
            <h4 className="card-title">
              <Trophy size={18} color="var(--accent-amber)" />
              League Table Standing &amp; Quality
            </h4>
            <p className="card-subtitle">Current league rank (1-20) and points per game rate</p>
          </div>
        </div>

        <div className="grid-4">
          <div className="form-group">
            <label className="form-label" htmlFor="m-h-rank">Home League Rank (1-20) *</label>
            <input
              id="m-h-rank"
              type="number"
              min="1"
              max="20"
              required
              className="form-input"
              style={errors.home_league_rank ? { borderColor: "var(--accent-rose)" } : undefined}
              value={formData.home_league_rank}
              onChange={(e) => handleChange("home_league_rank", Number(e.target.value))}
            />
            {errors.home_league_rank && (
              <span style={{ fontSize: "0.75rem", color: "#fb7185" }}>{errors.home_league_rank}</span>
            )}
          </div>

          <div className="form-group">
            <label className="form-label" htmlFor="m-h-ppg">Home PPG (0.0 - 3.0) *</label>
            <input
              id="m-h-ppg"
              type="number"
              step="0.01"
              min="0"
              max="3"
              required
              className="form-input"
              style={errors.home_ppg ? { borderColor: "var(--accent-rose)" } : undefined}
              value={formData.home_ppg}
              onChange={(e) => handleChange("home_ppg", Number(e.target.value))}
            />
            {errors.home_ppg && (
              <span style={{ fontSize: "0.75rem", color: "#fb7185" }}>{errors.home_ppg}</span>
            )}
          </div>

          <div className="form-group">
            <label className="form-label" htmlFor="m-a-rank">Away League Rank (1-20) *</label>
            <input
              id="m-a-rank"
              type="number"
              min="1"
              max="20"
              required
              className="form-input"
              style={errors.away_league_rank ? { borderColor: "var(--accent-rose)" } : undefined}
              value={formData.away_league_rank}
              onChange={(e) => handleChange("away_league_rank", Number(e.target.value))}
            />
            {errors.away_league_rank && (
              <span style={{ fontSize: "0.75rem", color: "#fb7185" }}>{errors.away_league_rank}</span>
            )}
          </div>

          <div className="form-group">
            <label className="form-label" htmlFor="m-a-ppg">Away PPG (0.0 - 3.0) *</label>
            <input
              id="m-a-ppg"
              type="number"
              step="0.01"
              min="0"
              max="3"
              required
              className="form-input"
              style={errors.away_ppg ? { borderColor: "var(--accent-rose)" } : undefined}
              value={formData.away_ppg}
              onChange={(e) => handleChange("away_ppg", Number(e.target.value))}
            />
            {errors.away_ppg && (
              <span style={{ fontSize: "0.75rem", color: "#fb7185" }}>{errors.away_ppg}</span>
            )}
          </div>
        </div>
      </div>

      {/* Section 4 & 5: Historical Context & Rest */}
      <div className="grid-2">
        {/* Historical Context */}
        <div className="card">
          <div className="card-header">
            <div>
              <h4 className="card-title">
                <History size={18} color="var(--accent-blue)" />
                Historical Context (H2H)
              </h4>
              <p className="card-subtitle">Prior head-to-head encounters and home win rate</p>
            </div>
          </div>

          <div className="grid-2">
            <div className="form-group">
              <label className="form-label" htmlFor="m-h2h-matches">Prior H2H Matches</label>
              <input
                id="m-h2h-matches"
                type="number"
                min="0"
                max="30"
                className="form-input"
                style={errors.h2h_matches_prior ? { borderColor: "var(--accent-rose)" } : undefined}
                value={formData.h2h_matches_prior ?? 6}
                onChange={(e) => handleChange("h2h_matches_prior", Number(e.target.value))}
              />
              {errors.h2h_matches_prior && (
                <span style={{ fontSize: "0.75rem", color: "#fb7185" }}>{errors.h2h_matches_prior}</span>
              )}
            </div>

            <div className="form-group">
              <label className="form-label" htmlFor="m-h2h-winpct">H2H Home Win Rate (0 - 1.0)</label>
              <input
                id="m-h2h-winpct"
                type="number"
                step="0.05"
                min="0"
                max="1"
                className="form-input"
                style={errors.h2h_home_win_pct ? { borderColor: "var(--accent-rose)" } : undefined}
                value={formData.h2h_home_win_pct ?? 0.5}
                onChange={(e) => handleChange("h2h_home_win_pct", Number(e.target.value))}
              />
              {errors.h2h_home_win_pct && (
                <span style={{ fontSize: "0.75rem", color: "#fb7185" }}>{errors.h2h_home_win_pct}</span>
              )}
            </div>
          </div>
        </div>

        {/* Rest Days */}
        <div className="card">
          <div className="card-header">
            <div>
              <h4 className="card-title">
                <Calendar size={18} color="var(--accent-tea-green)" />
                Rest &amp; Recovery
              </h4>
              <p className="card-subtitle">Calendar days elapsed since each club's last match (0 - 60)</p>
            </div>
          </div>

          <div className="grid-2">
            <div className="form-group">
              <label className="form-label" htmlFor="m-h-rest">Home Rest Days (0-60)</label>
              <input
                id="m-h-rest"
                type="number"
                min="0"
                max="60"
                className="form-input"
                style={errors.home_rest_days ? { borderColor: "var(--accent-rose)" } : undefined}
                value={formData.home_rest_days}
                onChange={(e) => handleChange("home_rest_days", Number(e.target.value))}
              />
              {errors.home_rest_days && (
                <span style={{ fontSize: "0.75rem", color: "#fb7185" }}>{errors.home_rest_days}</span>
              )}
            </div>

            <div className="form-group">
              <label className="form-label" htmlFor="m-a-rest">Away Rest Days (0-60)</label>
              <input
                id="m-a-rest"
                type="number"
                min="0"
                max="60"
                className="form-input"
                style={errors.away_rest_days ? { borderColor: "var(--accent-rose)" } : undefined}
                value={formData.away_rest_days}
                onChange={(e) => handleChange("away_rest_days", Number(e.target.value))}
              />
              {errors.away_rest_days && (
                <span style={{ fontSize: "0.75rem", color: "#fb7185" }}>{errors.away_rest_days}</span>
              )}
            </div>
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
            "Predict Match"
          )}
        </button>
      </div>
    </form>
  );
};
