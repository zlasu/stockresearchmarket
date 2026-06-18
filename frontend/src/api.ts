import type {
  ComparePayload,
  ExperimentDetail,
  ExperimentSummary,
  ScatterPayload,
  SeriesPoint,
  TablePayload,
  VariantSummary,
} from "./types";

const API_BASE = import.meta.env.VITE_API_BASE ?? "http://127.0.0.1:8000";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, init);
  if (!response.ok) {
    const text = await response.text();
    throw new Error(text || `Request failed: ${response.status}`);
  }
  return (await response.json()) as T;
}

export const api = {
  apiBase: API_BASE,
  health: () => request<{ status: string; experiments: number; root: string }>("/api/health"),
  refresh: () => request<{ status: string; experiments: number }>("/api/index/refresh", { method: "POST" }),
  experiments: (params: Record<string, string | number | undefined> = {}) => {
    const search = new URLSearchParams();
    Object.entries(params).forEach(([key, value]) => {
      if (value !== undefined && value !== "") search.set(key, String(value));
    });
    const suffix = search.toString() ? `?${search}` : "";
    return request<{ total: number; items: ExperimentSummary[] }>(`/api/experiments${suffix}`);
  },
  experiment: (id: string) => request<ExperimentDetail>(`/api/experiments/${encodeURIComponent(id)}`),
  variant: (experimentId: string, variantId: string) =>
    request<VariantSummary>(`/api/experiments/${encodeURIComponent(experimentId)}/variants/${encodeURIComponent(variantId)}`),
  scatter: (experimentId: string) => request<ScatterPayload>(`/api/experiments/${encodeURIComponent(experimentId)}/scatter`),
  compare: (experimentId: string, variantIds: string[]) => {
    const query = new URLSearchParams({ variant_ids: variantIds.join(",") });
    return request<ComparePayload>(`/api/experiments/${encodeURIComponent(experimentId)}/compare?${query}`);
  },
  series: (experimentId: string, variantId: string, series: string) => {
    const query = new URLSearchParams({ variant_id: variantId });
    return request<{ variant_id: string; series: string; rows: SeriesPoint[] }>(
      `/api/experiments/${encodeURIComponent(experimentId)}/series/${series}?${query}`,
    );
  },
  table: (experimentId: string, table: string, variantId?: string) => {
    const query = new URLSearchParams();
    if (variantId) query.set("variant_id", variantId);
    return request<TablePayload>(`/api/experiments/${encodeURIComponent(experimentId)}/tables/${table}?${query}`);
  },
  plotUrl: (experimentId: string, plot: string) =>
    `${API_BASE}/api/experiments/${encodeURIComponent(experimentId)}/plots/${plot}.png`,
};

