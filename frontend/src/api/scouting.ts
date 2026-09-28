import { apiClient } from "./client";
import type {
  ReplacementCandidate,
  ScoutingReplacementRequest,
  PlayerSearchItem,
  ScoutingModelInfo,
} from "../types/api";

export async function fetchSimilarPlayers(
  playerName: string,
  topN: number = 10
): Promise<ReplacementCandidate[]> {
  const encName = encodeURIComponent(playerName);
  return apiClient<ReplacementCandidate[]>(
    `/api/scouting/similar/${encName}?top_n=${topN}`
  );
}

export async function findReplacements(
  params: ScoutingReplacementRequest
): Promise<ReplacementCandidate[]> {
  return apiClient<ReplacementCandidate[]>("/api/scouting/replacements", {
    method: "POST",
    body: JSON.stringify(params),
  });
}

export async function searchPlayers(
  query: string = "",
  limit: number = 50
): Promise<PlayerSearchItem[]> {
  const searchParam = query ? `&search=${encodeURIComponent(query)}` : "";
  return apiClient<PlayerSearchItem[]>(
    `/api/scouting/players?limit=${limit}${searchParam}`
  );
}

export async function fetchScoutingModelInfo(): Promise<ScoutingModelInfo> {
  return apiClient<ScoutingModelInfo>("/api/scouting/model-info");
}
