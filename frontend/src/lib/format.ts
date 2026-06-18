import type { Metrics, MetricValue } from "../types";

export const percentMetrics = new Set([
  "total_return",
  "cagr",
  "volatility",
  "max_drawdown",
  "drawdown_magnitude",
  "alpha_total_return",
  "benchmark_total_return",
  "win_rate",
]);

export function formatMetric(key: string, value: MetricValue): string {
  if (value === null || value === undefined || value === "") return "n/a";
  if (typeof value === "boolean") return value ? "yes" : "no";
  if (typeof value !== "number") return String(value);
  if (!Number.isFinite(value)) return "n/a";
  if (percentMetrics.has(key)) {
    return `${(value * 100).toFixed(Math.abs(value) < 0.1 ? 2 : 1)}%`;
  }
  if (key.includes("turnover")) return value.toFixed(2);
  if (key === "trades" || key.endsWith("_count")) return value.toLocaleString();
  if (Math.abs(value) >= 1000) return value.toLocaleString(undefined, { maximumFractionDigits: 0 });
  return value.toFixed(Math.abs(value) < 10 ? 2 : 1);
}

export function metricNumber(metrics: Metrics, key: string): number | null {
  const value = metrics[key];
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

export function shortDate(value?: string | null): string {
  if (!value) return "n/a";
  const parsed = new Date(value);
  if (Number.isNaN(parsed.getTime())) return value.slice(0, 16);
  return parsed.toLocaleDateString(undefined, { year: "numeric", month: "short", day: "2-digit" });
}

export function compactLabel(value: string, max = 42): string {
  if (value.length <= max) return value;
  return `${value.slice(0, max - 1)}...`;
}

