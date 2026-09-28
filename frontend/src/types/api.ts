/**
 * TypeScript API schema interfaces for PremAI Football Intelligence Platform.
 * Kept strictly synchronized with backend FastAPI Pydantic schemas.
 */

export interface HealthResponse {
  status: string;
  models: {
    transfer: boolean;
    match: boolean;
    scouting: boolean;
  };
}

export interface RootResponse {
  name: string;
  status: string;
  version: string;
  docs_url: string;
}

// ----------------------------------------------------------------------------
// Transfer Value Predictor
// ----------------------------------------------------------------------------

export interface TransferPredictRequest {
  player_name?: string;
  age: number;
  height_cm?: number | null;
  contract_years_left?: number | null;
  club_market_value_eur?: number | null;
  minutes_played?: number | null;
  appearances?: number | null;
  goals?: number;
  assists?: number;
  yellow_cards?: number;
  red_cards?: number;
  goals_per90?: number | null;
  assists_per90?: number | null;
  shots_per90?: number | null;
  npxg_per90?: number | null;
  prev_market_value_eur?: number | null;
  primary_position?: string;
  sub_position?: string | null;
  preferred_foot?: string;
}

export interface TransferPredictResponse {
  success: boolean;
  player_name: string;
  predicted_market_value_eur: number;
  predicted_market_value_formatted: string;
  model_type: string;
  target_transformation: string;
}

export interface TransferModelInfo {
  model_type: string;
  model_version?: string;
  architecture?: string;
  target_transformation: string;
  training_seasons: string[];
  validation_season: string;
  test_season: string;
  dataset_rows: {
    train: number;
    val: number;
    test: number;
  };
  feature_count: number;
  features: string[];
  validation_metrics: {
    mae: number;
    mae_formatted: string;
    rmse: number;
    rmse_formatted?: string;
    r2: number;
    r2_formatted?: string;
    medae?: number;
    medae_formatted?: string;
    smape?: number;
  };
  test_metrics: {
    mae: number;
    mae_formatted: string;
    rmse: number;
    rmse_formatted: string;
    r2: number;
    r2_formatted: string;
    medae?: number;
    medae_formatted?: string;
    smape?: number;
  };
  training_timestamp: string;
}

// ----------------------------------------------------------------------------
// Match Outcome Predictor
// ----------------------------------------------------------------------------

export interface MatchPredictRequest {
  home_team?: string;
  away_team?: string;
  home_rest_days: number;
  away_rest_days: number;
  home_rolling_points_5: number;
  away_rolling_points_5: number;
  home_rolling_gf_5: number;
  away_rolling_gf_5: number;
  home_rolling_ga_5: number;
  away_rolling_ga_5: number;
  home_rolling_shots_5: number;
  away_rolling_shots_5: number;
  home_rolling_sot_5: number;
  away_rolling_sot_5: number;
  home_rolling_sot_conceded_5: number;
  away_rolling_sot_conceded_5: number;
  home_rolling_corners_5: number;
  away_rolling_corners_5: number;
  home_rolling_win_pct_5: number;
  away_rolling_win_pct_5: number;
  home_league_rank: number;
  away_league_rank: number;
  rank_diff?: number | null;
  home_ppg: number;
  away_ppg: number;
  h2h_matches_prior?: number;
  h2h_home_win_pct?: number;
}

export interface MatchProbabilities {
  home_win: number;
  draw: number;
  away_win: number;
}

export interface MatchPredictResponse {
  success: boolean;
  home_team: string;
  away_team: string;
  predicted_outcome: "H" | "D" | "A" | string;
  predicted_outcome_label: string;
  probabilities: MatchProbabilities;
  model_type: string;
  data_as_of?: string;
  season?: string;
}

export interface MatchModelInfo {
  model_type: string;
  model_version?: string;
  architecture?: string;
  feature_count: number;
  features: string[];
  train_seasons: string[];
  validation_season: string;
  test_season: string;
  split_counts: {
    train: number;
    val: number;
    test: number;
  };
  class_labels: string[];
  validation_metrics: Record<string, number>;
  test_metrics: Record<string, number>;
  training_timestamp: string;
}

// ----------------------------------------------------------------------------
// Player Scouting & Replacement
// ----------------------------------------------------------------------------

export interface CandidateStats {
  goals_per90: number;
  assists_per90: number;
  npxG_per90: number;
  xAG_per90: number;
  key_passes_per90: number;
  pass_completion_pct: number;
  progressive_passes_per90: number;
  progressive_carries_per90: number;
  tackles_won_per90: number;
  interceptions_per90: number;
}

export interface ReplacementCandidate {
  target_player: string;
  target_team: string;
  target_position: string;
  target_season: string;
  candidate_player: string;
  candidate_club: string;
  candidate_position: string;
  candidate_raw_position: string;
  candidate_season: string;
  similarity_score: number;
  cosine_similarity: number;
  position_compatibility: "Same Position" | "Compatible Position" | string;
  age?: number | null;
  market_value_eur?: number | null;
  minutes: number;
  key_stats: CandidateStats;
  data_as_of?: string;
}

export interface ScoutingReplacementRequest {
  player_name: string;
  top_n?: number;
  same_position_only?: boolean;
  max_market_value?: number | null;
  max_age?: number | null;
  min_minutes?: number | null;
}

export interface PlayerSearchItem {
  player_name: string;
  player_id?: string | null;
  position: string;
  normalized_position: string;
  club: string;
  season: string;
  age?: number | null;
  market_value_eur?: number | null;
  data_as_of?: string;
  has_scouting_data?: boolean;
  is_current?: boolean;
}

export interface ScoutingModelInfo {
  model_type: string;
  model_version?: string;
  architecture?: string;
  dataset_path: string;
  dataset_row_count: number;
  distinct_players_count: number;
  features_used: string[];
  feature_count: number;
  position_mapping: Record<string, string>;
  preprocessing_method: string;
  similarity_method: string;
  scaling_method: string;
  filters_supported: string[];
  evaluation_metrics: Record<string, any>;
  training_timestamp: string;
}

export interface TransferPresetItem {
  id: string;
  name: string;
  club: string;
  position: string;
  description: string;
  season?: string;
  data_as_of?: string;
  features: TransferPredictRequest;
}

export interface DataStatusResponse {
  data_as_of: string;
  players_as_of: string;
  scouting_as_of: string;
  matches_as_of: string;
  transfers_as_of: string;
  retrieved_at: string;
  sources: Array<{
    domain: string;
    scope: string;
    max_data_date: string;
    coverage: string;
  }>;
}

export type ActiveModule = "transfer" | "match" | "scouting";

