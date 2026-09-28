/**
 * Centralized HTTP Client for PremAI REST API.
 */

export const API_BASE_URL = (
  import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000"
).replace(/\/+$/, "");

export class ApiError extends Error {
  status: number;
  data?: unknown;

  constructor(status: number, message: string, data?: unknown) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.data = data;
  }
}

/**
 * Reusable JSON request wrapper with structured error parsing.
 */
export async function apiClient<T>(endpoint: string, options?: RequestInit): Promise<T> {
  const url = `${API_BASE_URL}${endpoint.startsWith("/") ? "" : "/"}${endpoint}`;

  const headers = new Headers(options?.headers);
  if (!headers.has("Content-Type") && options?.body && typeof options.body === "string") {
    headers.set("Content-Type", "application/json");
  }

  try {
    const res = await fetch(url, {
      ...options,
      headers,
    });

    if (!res.ok) {
      let errorMessage = `Request failed with status ${res.status}`;
      let errorBody: unknown = null;

      try {
        errorBody = await res.json();
        const errObj = errorBody as { detail?: string | Array<{ loc?: string[]; msg?: string }> };
        if (errObj && errObj.detail) {
          if (typeof errObj.detail === "string") {
            errorMessage = errObj.detail;
          } else if (Array.isArray(errObj.detail)) {
            errorMessage = errObj.detail
              .map((err) => `${err.loc ? err.loc.slice(-1) : "field"}: ${err.msg}`)
              .join("; ");
          }
        }
      } catch {
        // Response was not JSON
      }

      throw new ApiError(res.status, errorMessage, errorBody);
    }

    return (await res.json()) as T;
  } catch (err: unknown) {
    if (err instanceof ApiError) {
      throw err;
    }
    // Network errors (e.g. backend offline / connection refused)
    throw new ApiError(0, "Unable to connect to the PremAI backend. Please ensure the API is running at " + API_BASE_URL);
  }
}
