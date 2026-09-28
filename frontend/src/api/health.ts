import { apiClient } from "./client";
import type { HealthResponse, RootResponse } from "../types/api";

export async function fetchHealth(): Promise<HealthResponse> {
  return apiClient<HealthResponse>("/api/health");
}

export async function fetchRoot(): Promise<RootResponse> {
  return apiClient<RootResponse>("/");
}
