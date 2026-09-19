import type { SampleSignal } from "./sampleSignalCache";

// Pure chart-data helpers for MotionSignalViewer (Motion Phase 2 §5, Batch 4
// — AC1 "every axis renders" / AC2 "per-axis toggles show and hide
// individual channels"). Split out of the component so they're unit
// testable without mounting recharts/React: this repo's jest config runs a
// plain "node" test environment, not jsdom (see labelingStatus.test.ts for
// the established pure-function-test pattern used throughout this codebase).

export interface SignalChartPoint {
  t: number;
  [axisKey: string]: number;
}

/**
 * Builds one chart-ready point per (decimated) row, keyed `axis0..axisN-1`
 * so every decoded axis carries a value on every point — AC1 depends on
 * every row carrying every axis, independent of which axes are currently
 * toggled visible (visibility is applied separately, at render time, via
 * `toggleAxisVisibility`). `t` is elapsed time in ms when the sample has a
 * positive `frequency_hz`, otherwise the raw sample index — the same
 * decoded `{axes, values, frequency_hz}` shape serves both a device
 * recording's JSON envelope and a CSV upload, since `GET /samples/{id}/signal`
 * (Batch 3) already normalizes both formats to it before the frontend ever
 * sees them.
 */
export function buildSignalChartData(
  signal: Pick<SampleSignal, "values" | "frequency_hz">,
  maxPoints: number,
): SignalChartPoint[] {
  const { values, frequency_hz } = signal;
  if (!values.length) return [];
  const step = Math.max(1, Math.floor(values.length / maxPoints));
  const dt = frequency_hz > 0 ? 1000 / frequency_hz : null;
  const rows: SignalChartPoint[] = [];
  for (let i = 0; i < values.length; i += step) {
    const row = values[i];
    const point: SignalChartPoint = { t: dt ? Number((i * dt).toFixed(1)) : i };
    row.forEach((v, ch) => { point[`axis${ch}`] = v; });
    rows.push(point);
  }
  return rows;
}

/**
 * Toggles one axis index in/out of the hidden set (AC2). Pure — returns a
 * new Set rather than mutating `hidden`, so React state updates see a new
 * reference.
 */
export function toggleAxisVisibility(hidden: ReadonlySet<number>, index: number): Set<number> {
  const next = new Set(hidden);
  if (next.has(index)) next.delete(index); else next.add(index);
  return next;
}
