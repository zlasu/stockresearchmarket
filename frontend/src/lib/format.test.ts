import { describe, expect, it } from "vitest";

import { formatMetric, metricNumber, shortDate } from "./format";

describe("format helpers", () => {
  it("formats percent metrics", () => {
    expect(formatMetric("cagr", 0.1234)).toBe("12.3%");
    expect(formatMetric("drawdown_magnitude", 0.0456)).toBe("4.56%");
  });

  it("formats missing metrics", () => {
    expect(formatMetric("sharpe", null)).toBe("n/a");
  });

  it("returns numeric metric values safely", () => {
    expect(metricNumber({ sharpe: 1.2 }, "sharpe")).toBe(1.2);
    expect(metricNumber({ sharpe: "bad" } as never, "sharpe")).toBeNull();
  });

  it("formats dates compactly", () => {
    expect(shortDate("2026-06-16T10:00:00")).toContain("2026");
  });
});
