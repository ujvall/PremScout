import { apiClient } from "./client";
import type {
  ReplacementCandidate,
  ScoutingReplacementRequest,
  PlayerSearchItem,
  ScoutingModelInfo,
} from "../types/api";

export async function fetchSimilarPlayers(
  playerName: string,
  topN: number = 10,
  includeFreeAgents: boolean = false
): Promise<ReplacementCandidate[]> {
  const encName = encodeURIComponent(playerName);
  const faParam = includeFreeAgents ? "&include_free_agents=true" : "&include_free_agents=false";
  return apiClient<ReplacementCandidate[]>(
    `/api/scouting/similar/${encName}?top_n=${topN}${faParam}`
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
  limit: number = 50,
  includeFreeAgents: boolean = false
): Promise<PlayerSearchItem[]> {
  const searchParam = query ? `&search=${encodeURIComponent(query)}` : "";
  const faParam = includeFreeAgents ? "&include_free_agents=true" : "&include_free_agents=false";
  return apiClient<PlayerSearchItem[]>(
    `/api/scouting/players?limit=${limit}${searchParam}${faParam}`
  );
}

export async function fetchScoutingModelInfo(): Promise<ScoutingModelInfo> {
  return apiClient<ScoutingModelInfo>("/api/scouting/model-info");
}
