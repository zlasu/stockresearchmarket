import type { ComparePayload, ExperimentSummary, ScatterPayload, SeriesPoint } from "./types";

const palette = ["#0F766E", "#F0986E", "#5477C4", "#B8A037", "#BD569B", "#386411", "#804126"];
export type ChartOption = Record<string, unknown>;

export function experimentScatterOption(experiments: ExperimentSummary[]): ChartOption {
  const data = experiments
    .filter((item) => typeof item.metrics.drawdown_magnitude === "number" && typeof item.metrics.cagr === "number")
    .map((item, index) => ({
      value: [item.metrics.drawdown_magnitude, item.metrics.cagr, item.variant_count],
      name: item.name,
      itemStyle: { color: palette[index % palette.length] },
      experimentId: item.id,
    }));
  return scatterBase("Experiment risk map", "Max drawdown", "CAGR", data);
}

export function variantScatterOption(scatter: ScatterPayload): ChartOption {
  const data = scatter.rows.map((row, index) => ({
    value: [row.x, row.y, row.size ?? 1],
    name: row.variant_name,
    itemStyle: { color: palette[index % palette.length] },
    variantId: row.variant_id,
  }));
  return scatterBase("Capital versus drawdown", scatter.x, scatter.y, data);
}

export function compareLineOption(compare: ComparePayload, series: string, title: string): ChartOption {
  const byVariant = compare.series[series] ?? {};
  return {
    color: palette,
    grid: { left: 54, right: 22, top: 52, bottom: 52 },
    tooltip: { trigger: "axis" },
    legend: { top: 8, type: "scroll" },
    xAxis: { type: "category", boundaryGap: false },
    yAxis: { type: "value", scale: true },
    dataZoom: [{ type: "inside" }, { type: "slider", height: 20, bottom: 12 }],
    title: { text: title, left: 8, top: 6, textStyle: { fontSize: 13, fontWeight: 650 } },
    series: Object.entries(byVariant).map(([variantId, rows]) => ({
      name: labelFor(compare, variantId),
      type: "line",
      showSymbol: false,
      smooth: false,
      data: rows.map((row) => [row.date, row.value]),
    })),
  };
}

export function singleLineOption(rows: SeriesPoint[], title: string): ChartOption {
  return {
    color: [palette[0]],
    grid: { left: 54, right: 22, top: 52, bottom: 52 },
    tooltip: { trigger: "axis" },
    xAxis: { type: "category", boundaryGap: false },
    yAxis: { type: "value", scale: true },
    dataZoom: [{ type: "inside" }, { type: "slider", height: 20, bottom: 12 }],
    title: { text: title, left: 8, top: 6, textStyle: { fontSize: 13, fontWeight: 650 } },
    series: [{ type: "line", showSymbol: false, data: rows.map((row) => [row.date, row.value]) }],
  };
}

function scatterBase(title: string, xName: string, yName: string, data: unknown[]): ChartOption {
  return {
    grid: { left: 56, right: 24, top: 52, bottom: 46 },
    tooltip: {
      trigger: "item",
      formatter: (params: { name: string; value: number[] }) =>
        `${params.name}<br/>${xName}: ${(params.value[0] * 100).toFixed(1)}%<br/>${yName}: ${(params.value[1] * 100).toFixed(1)}%`,
    },
    title: { text: title, left: 8, top: 6, textStyle: { fontSize: 13, fontWeight: 650 } },
    xAxis: { type: "value", name: xName, axisLabel: { formatter: (value: number) => `${(value * 100).toFixed(0)}%` } },
    yAxis: { type: "value", name: yName, axisLabel: { formatter: (value: number) => `${(value * 100).toFixed(0)}%` } },
    series: [
      {
        type: "scatter",
        symbolSize: (value: number[]) => Math.max(8, Math.min(34, 8 + Math.sqrt(Math.max(value[2] ?? 1, 1)) * 2)),
        data,
      },
    ],
  };
}

function labelFor(compare: ComparePayload, variantId: string): string {
  return compare.metrics.find((row) => row.id === variantId)?.name ?? variantId;
}
