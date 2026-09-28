import { apiClient } from "./client";
import type {
  MatchPredictRequest,
  MatchPredictResponse,
  MatchModelInfo,
} from "../types/api";

export async function predictMatchOutcome(
  data: MatchPredictRequest
): Promise<MatchPredictResponse> {
  return apiClient<MatchPredictResponse>("/api/match/predict", {
    method: "POST",
    body: JSON.stringify(data),
  });
}

export async function fetchMatchModelInfo(): Promise<MatchModelInfo> {
  return apiClient<MatchModelInfo>("/api/match/model-info");
}
