import { apiClient } from "./client";
import type {
  TransferPredictRequest,
  TransferPredictResponse,
  TransferModelInfo,
} from "../types/api";

export async function predictTransferValue(
  data: TransferPredictRequest
): Promise<TransferPredictResponse> {
  return apiClient<TransferPredictResponse>("/api/transfer/predict", {
    method: "POST",
    body: JSON.stringify(data),
  });
}

export async function fetchTransferModelInfo(): Promise<TransferModelInfo> {
  return apiClient<TransferModelInfo>("/api/transfer/model-info");
}

export interface TransferPresetsResponse {
  success: boolean;
  data_as_of: string;
  presets: Array<{
    id: string;
    name: string;
    club: string;
    position: string;
    description: string;
    season: string;
    data_as_of: string;
    features: TransferPredictRequest;
  }>;
}

export async function fetchTransferPresets(): Promise<TransferPresetsResponse> {
  return apiClient<TransferPresetsResponse>("/api/transfer/presets");
}

