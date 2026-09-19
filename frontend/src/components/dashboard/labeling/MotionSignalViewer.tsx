"use client";
import { useEffect, useMemo, useState } from "react";
import { createPortal } from "react-dom";
import {
  LineChart, Line, XAxis, YAxis, Tooltip as RTooltip, CartesianGrid, ResponsiveContainer,
  ReferenceLine,
} from "recharts";
import { X, Activity, AlertTriangle, Loader2 } from "lucide-react";
import { useAppStore, type Impulse } from "@/store/appStore";
import { formatDate, formatDuration, formatFrequency, splitLabel } from "./LabelingShared";
import { fetchSampleSignal, type SampleSignal } from "./sampleSignalCache";
import { buildSignalChartData, toggleAxisVisibility } from "./motionSignalChart";
import type { LabelOption } from "./LabelFilterSelect";

// Full, interactive multi-axis Signal Viewer — Motion Phase 2 §5 / Batch 4.
// Opened from the Raw Data detail panel's expand action (MotionLabeling.tsx);
// it is an additional, intentional expanded view, not a replacement for the
// compact WaveformThumb preview that stays mounted underneath.
//
// Renders the same decoded {axes, values, ...} payload WaveformThumb and the
// detail panel already fetch through sampleSignalCache (a cache hit here,
// not a second decode/API call) as a real recharts line chart with per-axis
// show/hide toggles, instead of the thumb's normalized SVG sparkline. Axis
// colors and chart chrome (grid/tick/tooltip styling) mirror
// SignalPreview.tsx's conventions (Phase 1's live ring-buffer chart) — not
// its buffering/staleness logic, which has no meaning for a fixed decoded
// array.

const AXIS_COLORS = ["#60a5fa", "#f472b6", "#34d399", "#fbbf24", "#a78bfa", "#fb7185"];
const AXIS_TICK = { fontSize: 11, fill: "var(--app-text-soft)" };
const GRID_STROKE = "var(--app-border)";
const TOOLTIP_STYLE = {
  background: "var(--app-surface)",
  border: "1px solid var(--app-border)",
  borderRadius: 8,
  fontSize: 12,
  color: "var(--app-text)",
} as const;

// Cap for chart responsiveness on long recordings — well above what a
// viewport can usefully distinguish, but keeps a very long capture from
// stalling recharts. WaveformThumb's own cap (80 points) is a thumbnail
// concern and unrelated to this one.
const MAX_CHART_POINTS = 2000;

export interface MotionSignalViewerSample {
  id: string;
  filename?: string | null;
  frequency_hz?: number | null;
  duration_ms?: number | null;
  num_channels?: number | null;
  sample_type?: string | null;
  created_at?: string | null;
  label_id?: string | null;
}

export interface MotionSignalViewerProps {
  sample: MotionSignalViewerSample;
  labels: LabelOption[];
  busy?: boolean;
  onAssignLabel: (sampleId: string, labelId: string) => void;
  onClose: () => void;
}

type ViewerState = "loading" | "ready" | "empty" | "error";

// ── Window boundary overlay (Motion Phase 3 §4.2.2 / Batch 9) ───────────────
// Client-side only: mirrors backend/app/motion/dsp/windowing.py's
// count_windows arithmetic exactly, against data this component already has
// on screen (no new endpoint, no second decode). The one piece Batch 9 needed
// that wasn't already here is the *impulse* config (window_size_ms,
// window_increase_ms, zero_pad_allowed) — Samples carry no impulse_id (a
// project's dataset isn't owned by any one impulse) and this modal's own
// props/parent (MotionLabeling.tsx) never touch impulse data, so there is no
// per-sample "correct impulse" to fetch even if a new prop were allowed.
// What *is* reliable, and already the established resolution for "the
// impulse currently relevant to this project" (Sidebar.tsx's own
// `sidebarImpulse`, Batch 8's dataset-summary refresh key): the store's
// `savedActiveImpulse`/`activeImpulse`, guarded to the active project so a
// leftover selection from a different project is never used. `savedActiveImpulse`
// (persisted) is preferred over `activeImpulse` (possibly-unsaved draft) so the
// overlay reflects what a real feature-generation run would actually window
// with, matching Batch 8's own "keyed off savedActiveImpulse" precedent. When
// neither resolves (no impulse selected yet, e.g. a project with none created),
// no overlay is drawn — the signal renders exactly as before, unchanged.
function useResolvedWindowingImpulse(): Impulse | null {
  const { activeProject, activeImpulse, savedActiveImpulse } = useAppStore();
  return useMemo(() => {
    if (!activeProject) return null;
    const candidate =
      savedActiveImpulse?.project_id === activeProject.id
        ? savedActiveImpulse
        : activeImpulse?.project_id === activeProject.id
          ? activeImpulse
          : null;
    if (!candidate || candidate.input_type !== "time-series") return null;
    return candidate as Impulse;
  }, [activeProject?.id, activeImpulse, savedActiveImpulse]);
}

interface WindowBoundary {
  index: number;
  t: number;
  padded: boolean;
}

/** Snaps a target chart-x value to the nearest point actually plotted —
 *  required because recharts' default category XAxis only positions a
 *  ReferenceLine at an x that exists in `data`; long recordings decimate
 *  (MAX_CHART_POINTS), so a window boundary's raw sample index rarely lands
 *  on a kept point exactly. `points` is sorted ascending by `t`. */
function snapToChartT(points: { t: number }[], target: number): number {
  let lo = 0, hi = points.length - 1;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (points[mid].t < target) lo = mid + 1; else hi = mid;
  }
  if (lo > 0 && Math.abs(points[lo - 1].t - target) <= Math.abs(points[lo].t - target)) {
    return points[lo - 1].t;
  }
  return points[lo].t;
}

/** Mirrors windowing.py's `_window_geometry` + `count_windows` exactly:
 *  window_length = round(window_size_ms * freq / 1000),
 *  stride = round(window_increase_ms * freq / 1000). `freqHz` here must be
 *  the same fallback chain `iter_payloads` uses in production
 *  (`sample.frequency_hz or impulse.frequency_hz or 100.0`), not the
 *  possibly-different frequency the chart derived the recording's own decoded
 *  `frequency_hz` from — the two can disagree (e.g. a JSON device recording's
 *  `interval_ms` vs. the Sample row's stored `frequency_hz`), and windowing
 *  math must match what dsp_worker.py will actually run at feature-generation
 *  time, not what the chart's x-axis happens to be scaled by. */
function computeWindowBoundaries(
  numSamples: number,
  freqHz: number,
  windowSizeMs: number,
  windowIncreaseMs: number,
  zeroPadAllowed: boolean,
  chartT: (sampleIndex: number) => number,
  chartPoints: { t: number }[],
): WindowBoundary[] {
  if (!(numSamples > 0) || !(freqHz > 0) || !chartPoints.length) return [];

  const windowLength = Math.round((windowSizeMs * freqHz) / 1000);
  const stride = Math.round((windowIncreaseMs * freqHz) / 1000);
  if (!(windowLength > 0) || !(stride > 0)) return [];

  let numWindows: number;
  if (numSamples < windowLength) {
    numWindows = zeroPadAllowed ? 1 : 0;
  } else {
    numWindows = 1 + Math.floor((numSamples - windowLength) / stride);
  }
  if (numWindows <= 0) return [];

  const boundaries: WindowBoundary[] = [];
  for (let i = 0; i < numWindows; i++) {
    const startIdx = i * stride;
    boundaries.push({
      index: i,
      t: snapToChartT(chartPoints, chartT(startIdx)),
      padded: startIdx + windowLength > numSamples,
    });
  }
  return boundaries;
}

export default function MotionSignalViewer({
  sample, labels, busy, onAssignLabel, onClose,
}: MotionSignalViewerProps) {
  const [state, setState] = useState<ViewerState>("loading");
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [signal, setSignal] = useState<SampleSignal | null>(null);
  const [hiddenAxes, setHiddenAxes] = useState<Set<number>>(new Set());
  const windowingImpulse = useResolvedWindowingImpulse();

  useEffect(() => {
    let cancelled = false;
    setState("loading");
    setErrorMessage(null);
    fetchSampleSignal(sample.id)
      .then(data => {
        if (cancelled) return;
        if (!data.axes?.length || !data.values?.length) {
          setState("empty");
          return;
        }
        setSignal(data);
        setHiddenAxes(new Set());
        setState("ready");
      })
      .catch(e => {
        if (cancelled) return;
        setErrorMessage(e?.response?.data?.detail || "Failed to load signal data for this recording.");
        setState("error");
      });
    return () => { cancelled = true; };
  }, [sample.id]);

  useEffect(() => {
    function onKeyDown(e: KeyboardEvent) {
      if (e.key === "Escape") onClose();
    }
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [onClose]);

  const chartData = useMemo(
    () => (signal ? buildSignalChartData(signal, MAX_CHART_POINTS) : []),
    [signal],
  );

  const windowBoundaries = useMemo(() => {
    if (!signal || !windowingImpulse || !chartData.length) return [];
    const numSamples = signal.num_samples > 0 ? signal.num_samples : signal.values.length;
    const freqHz = sample.frequency_hz || windowingImpulse.frequency_hz || 0;
    const chartDt = signal.frequency_hz > 0 ? 1000 / signal.frequency_hz : null;
    const chartT = (sampleIndex: number) => (chartDt ? sampleIndex * chartDt : sampleIndex);
    return computeWindowBoundaries(
      numSamples,
      freqHz,
      windowingImpulse.window_size_ms,
      windowingImpulse.window_increase_ms,
      windowingImpulse.zero_pad_allowed !== false,
      chartT,
      chartData,
    );
  }, [signal, windowingImpulse, chartData, sample.frequency_hz]);

  function toggleAxis(index: number) {
    setHiddenAxes(prev => toggleAxisVisibility(prev, index));
  }

  const baseName = sample.filename?.split("/").pop()?.split("\\").pop() || sample.id;
  const usesTime = (signal?.frequency_hz ?? 0) > 0;

  if (typeof document === "undefined") return null;

  return createPortal(
    <div
      className="overlay-modal fixed inset-0 z-[100] flex items-center justify-center px-4 py-6"
      onClick={onClose}
    >
      <div
        className="surface-raised rounded-2xl w-full flex flex-col"
        style={{ maxWidth: 920, maxHeight: "92vh" }}
        onClick={e => e.stopPropagation()}
        role="dialog"
        aria-modal="true"
        aria-label={`Signal viewer — ${baseName}`}
      >
        <div
          className="flex items-center justify-between gap-3 px-6 py-4 shrink-0"
          style={{ borderBottom: "1px solid var(--app-border)" }}
        >
          <div className="min-w-0">
            <p className="text-[10px] font-semibold uppercase tracking-wider" style={{ color: "var(--app-text-soft)" }}>
              Signal viewer
            </p>
            <h3
              className="text-base font-semibold truncate"
              style={{ color: "var(--app-text)" }}
              title={baseName}
            >
              {baseName}
            </h3>
          </div>
          <button
            type="button"
            onClick={onClose}
            className="opacity-60 hover:opacity-100 transition-opacity shrink-0"
            style={{ color: "var(--app-text-soft)" }}
            aria-label="Close signal viewer"
          >
            <X size={16} />
          </button>
        </div>

        <div className="flex-1 overflow-y-auto px-6 py-5 space-y-4">
          <div className="flex items-center gap-2">
            <label
              htmlFor="signal-viewer-label"
              className="text-xs font-medium shrink-0"
              style={{ color: "var(--app-text-soft)" }}
            >
              Gesture label
            </label>
            <select
              id="signal-viewer-label"
              className="input"
              style={{ padding: "4px 8px", fontSize: "12px", maxWidth: 240 }}
              value={sample.label_id ?? ""}
              disabled={busy}
              onChange={e => onAssignLabel(sample.id, e.target.value)}
            >
              <option value="">Unlabeled</option>
              {labels.map(l => (
                <option key={l.id} value={l.id}>{l.name}</option>
              ))}
            </select>
          </div>

          <div className="ds-detail-meta">
            <div className="ds-detail-meta-row">
              <span className="ds-detail-meta-label">Frequency</span>
              <span className="ds-detail-meta-value">{formatFrequency(sample.frequency_hz)}</span>
            </div>
            <div className="ds-detail-meta-row">
              <span className="ds-detail-meta-label">Duration</span>
              <span className="ds-detail-meta-value">{formatDuration(sample.duration_ms)}</span>
            </div>
            <div className="ds-detail-meta-row">
              <span className="ds-detail-meta-label">Channels</span>
              <span className="ds-detail-meta-value">{sample.num_channels ?? "—"}</span>
            </div>
            <div className="ds-detail-meta-row">
              <span className="ds-detail-meta-label">Split</span>
              <span className="ds-detail-meta-value">{splitLabel(sample.sample_type)}</span>
            </div>
            <div className="ds-detail-meta-row">
              <span className="ds-detail-meta-label">Collected</span>
              <span className="ds-detail-meta-value">{formatDate(sample.created_at)}</span>
            </div>
          </div>

          <div className="ds-raw-data-chart">
            {state === "loading" && (
              <div className="flex flex-col items-center justify-center py-16 gap-2">
                <Loader2 size={18} className="animate-spin" style={{ color: "var(--app-text-soft)" }} />
                <p className="text-xs" style={{ color: "var(--app-text-muted)" }}>Loading signal…</p>
              </div>
            )}

            {state === "empty" && (
              <div className="flex flex-col items-center justify-center py-16 gap-2 text-center">
                <Activity size={20} style={{ color: "var(--app-text-soft)" }} />
                <p className="text-xs" style={{ color: "var(--app-text-muted)" }}>
                  No signal data is available for this recording.
                </p>
              </div>
            )}

            {state === "error" && (
              <div className="flex flex-col items-center justify-center py-16 gap-2 text-center">
                <AlertTriangle size={20} className="text-amber-400" />
                <p className="text-xs" style={{ color: "var(--app-text-muted)" }}>{errorMessage}</p>
              </div>
            )}

            {state === "ready" && signal && (
              <>
                <div style={{ width: "100%", height: 340 }}>
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={chartData} margin={{ top: 8, right: 16, left: 0, bottom: 4 }}>
                      <CartesianGrid stroke={GRID_STROKE} vertical={false} />
                      <XAxis
                        dataKey="t"
                        tick={AXIS_TICK}
                        axisLine={{ stroke: GRID_STROKE }}
                        label={{
                          value: usesTime ? "Time (ms)" : "Sample index",
                          position: "insideBottom",
                          offset: -2,
                          fontSize: 11,
                          fill: "var(--app-text-soft)",
                        }}
                      />
                      <YAxis tick={AXIS_TICK} width={48} domain={["auto", "auto"]} />
                      <RTooltip
                        contentStyle={TOOLTIP_STYLE}
                        formatter={(value: any, name: any) => [Number(value).toFixed(3), name]}
                        labelFormatter={(t: any) => (usesTime ? `${t} ms` : `sample ${t}`)}
                      />
                      {signal.axes.map((axis, i) => {
                        if (hiddenAxes.has(i)) return null;
                        return (
                          <Line
                            key={axis}
                            type="monotone"
                            dataKey={`axis${i}`}
                            name={axis}
                            stroke={AXIS_COLORS[i % AXIS_COLORS.length]}
                            strokeWidth={1.75}
                            dot={false}
                            isAnimationActive={false}
                          />
                        );
                      })}
                      {windowBoundaries.map(b => (
                        <ReferenceLine
                          key={b.index}
                          x={b.t}
                          stroke={b.padded ? "#f59e0b" : "#94a3b8"}
                          strokeDasharray={b.padded ? "2 3" : "4 4"}
                          strokeWidth={1}
                          ifOverflow="extendDomain"
                        />
                      ))}
                    </LineChart>
                  </ResponsiveContainer>
                </div>
                {windowBoundaries.length > 0 && windowingImpulse && (
                  <p className="text-[10px]" style={{ color: "var(--app-text-soft)" }}>
                    {windowBoundaries.length} window boundar{windowBoundaries.length === 1 ? "y" : "ies"}
                    {" · "}{windowingImpulse.window_size_ms}ms window / {windowingImpulse.window_increase_ms}ms step
                    {windowBoundaries.some(b => b.padded) ? " · zero-padded (amber)" : ""}
                  </p>
                )}
                <div className="ds-raw-data-legend">
                  {signal.axes.map((axis, i) => {
                    const active = !hiddenAxes.has(i);
                    return (
                      <button
                        key={axis}
                        type="button"
                        onClick={() => toggleAxis(i)}
                        className="ds-raw-data-legend-item"
                        style={{ background: "none", border: "none", padding: 0, opacity: active ? 1 : 0.4 }}
                        aria-pressed={active}
                        aria-label={`Toggle ${axis} axis ${active ? "off" : "on"}`}
                      >
                        <span
                          className="ds-raw-data-legend-dot"
                          style={{ background: AXIS_COLORS[i % AXIS_COLORS.length] }}
                          aria-hidden="true"
                        />
                        {axis}
                      </button>
                    );
                  })}
                </div>
              </>
            )}
          </div>
        </div>
      </div>
    </div>,
    document.body,
  );
}
