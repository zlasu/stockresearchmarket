export type MetricValue = number | string | boolean | null | undefined;

export interface Metrics {
  total_return?: number | null;
  cagr?: number | null;
  volatility?: number | null;
  sharpe?: number | null;
  sortino?: number | null;
  max_drawdown?: number | null;
  drawdown_magnitude?: number | null;
  calmar?: number | null;
  trades?: number | null;
  avg_annual_turnover?: number | null;
  alpha_total_return?: number | null;
  benchmark_total_return?: number | null;
  capital?: number | null;
  research_score?: number | null;
  [key: string]: MetricValue;
}

export interface ExperimentSummary {
  id: string;
  name: string;
  kind: string;
  created_at?: string | null;
  variant_count: number;
  best_variant_id?: string | null;
  best_variant_name?: string | null;
  metrics: Metrics;
  path: string;
  caveat_count: number;
}

export interface VariantSummary {
  id: string;
  name: string;
  kind: string;
  role?: string | null;
  rank?: number | null;
  source?: string | null;
  metrics: Metrics;
  original_metrics: Record<string, MetricValue>;
  params: Record<string, MetricValue>;
  computed_score?: number | null;
  path: string;
  available_tables: string[];
  available_series: string[];
  series_columns: Record<string, string>;
  artifacts: Record<string, string>;
  start?: string | null;
  end?: string | null;
}

export interface Recommendation {
  title: string;
  variant_id: string;
  variant_name: string;
  score?: number | null;
  body: string;
}

export interface ExperimentDetail extends ExperimentSummary {
  metadata: Record<string, unknown>;
  caveats: string[];
  recommendations: Recommendation[];
  available_tables: string[];
  available_series: string[];
  artifacts: Record<string, string>;
  variants: VariantSummary[];
}

export interface SeriesPoint {
  date: string;
  value: number;
}

export interface ComparePayload {
  metrics: Array<{
    id: string;
    name: string;
    role?: string | null;
    metrics: Metrics;
    params: Record<string, MetricValue>;
    computed_score?: number | null;
  }>;
  series: Record<string, Record<string, SeriesPoint[]>>;
}

export interface ScatterPayload {
  x: string;
  y: string;
  rows: Array<{
    variant_id: string;
    variant_name: string;
    x: number;
    y: number;
    color?: string | null;
    size?: number | null;
    metrics: Metrics;
  }>;
}

export interface TablePayload {
  columns: Array<{ key: string; label: string }>;
  rows: Array<Record<string, MetricValue>>;
  row_count: number;
  offset: number;
  limit: number;
  truncated: boolean;
}

