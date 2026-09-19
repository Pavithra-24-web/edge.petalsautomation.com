"use client";

import type { MetricRow } from "@/types/model-testing";

interface MetricsTableProps {
  metrics: MetricRow[];
  title?: string;
}

// Metrics that are image counts, not scores. Everything else in this table is
// a ratio or a rate and reads correctly at 2 decimal places; "3.00 background
// images" does not.
const COUNT_METRICS = new Set([
  "background_images",
  "background_images_with_fp",
]);

function formatValue(v: number | null | undefined, metricName?: string): string {
  if (v === null || v === undefined || !Number.isFinite(v)) return "—";
  if (metricName && COUNT_METRICS.has(metricName)) return String(Math.round(v));
  return v.toFixed(2);
}

export default function MetricsTable({
  metrics,
  title = "Metrics for Object detection",
}: MetricsTableProps) {
  return (
    <div>
      {title && (
        <div className="flex items-center justify-between mb-3">
          <h4 className="text-sm font-semibold text-[color:var(--app-text)]">{title}</h4>
        </div>
      )}

      <table className="w-full text-sm">
        <thead>
          <tr style={{ borderBottom: "1px solid var(--app-border)" }}>
            <th className="text-left text-[10px] font-semibold uppercase tracking-wider pb-2 text-[color:var(--app-text-muted)]">
              Metric
            </th>
            <th className="text-right text-[10px] font-semibold uppercase tracking-wider pb-2 text-[color:var(--app-text-muted)]">
              Value
            </th>
          </tr>
        </thead>
        <tbody>
          {metrics.map((row, idx) => (
            <tr
              key={row.metric_name ?? idx}
              style={{ borderBottom: "1px solid var(--app-border)" }}
              className="last:border-0"
            >
              <td className="py-2.5 pr-4">
                <span className="text-sm text-[color:var(--app-text)]">
                  {row.metric_display_name ?? row.metric_name ?? "—"}
                </span>
              </td>
              <td
                className="py-2.5 text-right text-sm font-semibold tabular-nums"
                style={{ color: "#6d28d9" }}
              >
                {formatValue(row.metric_value, row.metric_name)}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
