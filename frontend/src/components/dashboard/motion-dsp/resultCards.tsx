"use client";
import { ReactNode } from "react";
import { Box, Tag, Cpu, AlertTriangle } from "lucide-react";
import { CopyButton, HelpTooltip, FeatureCountBadge } from "./common";
import { AXIS_PALETTE } from "./RawDataGraph";

/** Discriminated shape of `/dsp/preview`'s `raw_preview` field (motion_phase4_5.md
 *  G3, closed by Phase 4.5 Batch 3.5). `time_series` carries the same
 *  `{axes, values, frequency_hz, duration_ms, num_channels, num_samples}`
 *  shape as `GET /samples/{id}/signal` (R41), so it can be charted the same
 *  way `RawDataGraph` charts the whole recording. */
export type RawPreview =
  | {
      type: "time_series";
      axes: string[];
      values: number[][];
      frequency_hz: number;
      duration_ms: number | null;
      num_channels: number;
      num_samples: number;
    }
  | { type: "image"; data_uri: string | null; width: number; height: number }
  | { type: "unavailable"; reason: string };

/** Raw features strip — real per-axis sensor values for motion recordings
 *  (motion_phase4_5.md G3, closed by Phase 4.5 Batch 3.5). Sourced from
 *  `/dsp/preview`'s `raw_preview` field: a compact multi-axis chart, a
 *  truncated numeric value list per axis, and available metadata (sample
 *  count, sampling frequency, channel names, duration). Falls back to an
 *  honest unavailable/error state rather than fabricating values when the
 *  backend could not decode the recording. */
export function RawFeaturesStrip({
  rawPreview,
  loading,
  error,
}: {
  rawPreview: RawPreview | null;
  loading: boolean;
  error: string | null;
}) {
  const body = (() => {
    if (loading) {
      return <div className="pe-dsp-raw-block" style={{ fontStyle: "italic", opacity: 0.7 }}>Computing preview…</div>;
    }
    if (error) {
      return <div className="pe-dsp-raw-block" style={{ color: "#dc2626" }}>{error}</div>;
    }
    if (!rawPreview) {
      return (
        <div className="pe-dsp-raw-block" style={{ fontStyle: "italic", opacity: 0.7 }}>
          Select a sample to preview its raw features.
        </div>
      );
    }
    if (rawPreview.type === "unavailable") {
      return <div className="pe-dsp-raw-block" style={{ fontStyle: "italic", opacity: 0.7 }}>{rawPreview.reason}</div>;
    }
    if (rawPreview.type === "image") {
      // Not reached from any motion DSP page today (blockType is always
      // raw/spectral_analysis/flatten), kept only so the type is exhaustive.
      return <div className="pe-dsp-raw-block" style={{ fontStyle: "italic", opacity: 0.7 }}>Image-shaped raw preview is not rendered here.</div>;
    }

    const { axes, values, frequency_hz, duration_ms, num_channels, num_samples } = rawPreview;
    return (
      <div className="pe-dsp-card-body">
        <div className="pe-dsp-raw-block" style={{ padding: "0.5rem 0" }}>
          {axes.map((axis, i) => {
            const columnValues = values.map((row) => row[i]);
            const truncated = columnValues.slice(0, 30);
            return (
              <div key={axis} style={{ marginBottom: 4 }}>
                <span style={{ color: AXIS_PALETTE[i % AXIS_PALETTE.length] }}>{axis}: </span>
                [{truncated.map((n) => n.toFixed(4)).join(", ")}
                {columnValues.length > 30 ? " ..." : ""}]
              </div>
            );
          })}
        </div>
        <div className="flex items-center justify-between flex-wrap gap-2" style={{ marginTop: 4 }}>
          <div className="flex items-center gap-3 flex-wrap">
            {axes.map((axis, i) => (
              <span key={axis} className="text-xs font-medium inline-flex items-center gap-1.5" style={{ color: "var(--app-text-muted)" }}>
                <span style={{ width: 8, height: 8, borderRadius: 999, background: AXIS_PALETTE[i % AXIS_PALETTE.length], display: "inline-block" }} />
                {axis}
              </span>
            ))}
          </div>
          <span className="text-[10px]" style={{ color: "var(--app-text-soft)" }}>
            {num_channels} channel{num_channels === 1 ? "" : "s"} · {num_samples} samples
            {duration_ms != null ? ` · ${duration_ms} ms` : ""}
            {frequency_hz ? ` · ${frequency_hz.toFixed(1)} Hz` : ""}
          </span>
        </div>
      </div>
    );
  })();

  return (
    <div className="pe-dsp-card">
      <div className="pe-dsp-card-head" style={{ justifyContent: "space-between" }}>
        <h3 className="head-title">Raw features</h3>
        <div className="flex items-center gap-2">
          {rawPreview?.type === "time_series" && (
            <CopyButton getText={() => JSON.stringify({ axes: rawPreview.axes, values: rawPreview.values })} />
          )}
          <HelpTooltip text="The recording's decoded sensor values, before this block's processing is applied." />
        </div>
      </div>
      {body}
    </div>
  );
}

export function LabelCard({ label }: { label: string | null }) {
  return (
    <div className="pe-dsp-card">
      <div className="pe-dsp-card-head" style={{ justifyContent: "space-between" }}>
        <div className="flex items-center gap-2">
          <span className="head-icon icon-emerald"><Tag size={14} /></span>
          <h3 className="head-title">Label</h3>
        </div>
        <HelpTooltip text="The label assigned to the currently previewed recording." />
      </div>
      <div className="pe-dsp-card-body">
        <p className="text-sm font-medium" style={{ color: "var(--app-text)" }}>{label || "Unlabeled"}</p>
      </div>
    </div>
  );
}

/** Right-hand `DSP result` card — block-specific graphs (or their degraded
 *  placeholders) via `children`, then the live processed-features strip and
 *  window-aware feature count. */
export function DspResultCard({
  blockType,
  features,
  loading,
  axisCount,
  children,
}: {
  blockType: string;
  features: number[] | null;
  loading: boolean;
  axisCount: number;
  children?: ReactNode;
}) {
  return (
    <div className="pe-dsp-card">
      <div className="pe-dsp-card-head">
        <span className="head-icon icon-purple"><Box size={14} /></span>
        <h3 className="head-title">DSP result</h3>
      </div>
      <div className="pe-dsp-card-body space-y-3">
        {children}
        <div>
          <div className="flex items-center justify-between mb-1">
            <h4 className="text-xs font-bold" style={{ color: "var(--app-text)" }}>Processed features</h4>
            {features && <CopyButton getText={() => JSON.stringify(features)} />}
          </div>
          <div className="pe-dsp-raw-block">
            {loading
              ? "Computing preview…"
              : features
                ? `[${features.slice(0, 30).map((n) => n.toFixed(4)).join(", ")}${features.length > 30 ? " ..." : ""}]`
                : "Select a sample to preview processed features."}
          </div>
        </div>
        <FeatureCountBadge blockType={blockType} featureCount={features ? features.length : null} axisCount={axisCount} />
      </div>
    </div>
  );
}

/** `On-device performance` card — honest-unavailable state (G8): no
 *  processing-time/RAM estimation exists anywhere in the backend, so this
 *  never fabricates a number. */
export function PerformanceCards() {
  return (
    <div className="pe-dsp-card">
      <div className="pe-dsp-card-head" style={{ justifyContent: "space-between" }}>
        <div className="flex items-center gap-2">
          <span className="head-icon icon-blue"><Cpu size={14} /></span>
          <h3 className="head-title">On-device performance</h3>
        </div>
        <HelpTooltip text="Processing-time and memory estimation for a target device is not implemented yet." />
      </div>
      <div className="pe-dsp-perf-body">
        <div className="pe-dsp-perf-stat">
          <span className="stat-label"><AlertTriangle size={12} /> Processing time</span>
          <span className="stat-value" style={{ fontSize: "0.95rem", opacity: 0.6 }}>Not available</span>
        </div>
        <div className="pe-dsp-perf-stat">
          <span className="stat-label"><AlertTriangle size={12} /> Peak RAM usage</span>
          <span className="stat-value" style={{ fontSize: "0.95rem", opacity: 0.6 }}>Not available</span>
        </div>
      </div>
      <div className="pe-dsp-perf-note">
        On-device performance estimation is not available for this target yet.
      </div>
    </div>
  );
}
