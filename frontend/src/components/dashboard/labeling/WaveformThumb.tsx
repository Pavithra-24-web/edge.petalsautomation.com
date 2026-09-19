"use client";
import { useEffect, useState } from "react";
import { Activity } from "lucide-react";
import { PREMIUM_SOFT_PALETTE } from "./LabelingShared";
import { fetchSampleSignal } from "./sampleSignalCache";

// Compact multi-axis sparkline — the motion equivalent of the image dataset's
// thumbnail cell (Motion Phase 2 §4 task 2). Decodes a recording's signal via
// GET /samples/{id}/signal (Batch 3), then draws one polyline per axis,
// each independently min/max-normalized to the thumb's height so a small
// motion on one axis is still visible next to a large motion on another.
// Used in both the list-layout thumbnail column and the grid/detailed cards.

export interface WaveformThumbProps {
  sampleId: string;
  width?: number;
  height?: number;
  className?: string;
}

const MAX_POINTS = 80; // enough resolution for a thumb-sized sparkline, decimated below

export default function WaveformThumb({ sampleId, width = 96, height = 32, className }: WaveformThumbProps) {
  const [state, setState] = useState<"loading" | "ready" | "error">("loading");
  const [axes, setAxes] = useState<string[]>([]);
  const [values, setValues] = useState<number[][]>([]);

  useEffect(() => {
    let cancelled = false;
    setState("loading");
    fetchSampleSignal(sampleId)
      .then(data => {
        if (cancelled) return;
        setAxes(data.axes ?? []);
        setValues(data.values ?? []);
        setState("ready");
      })
      .catch(() => { if (!cancelled) setState("error"); });
    return () => { cancelled = true; };
  }, [sampleId]);

  if (state === "loading") {
    return (
      <div
        className={className}
        style={{ width, height, borderRadius: 6, background: "var(--app-surface-2)" }}
        aria-label="Loading waveform"
      >
        <div className="h-full w-full animate-pulse" style={{ borderRadius: 6, background: "var(--app-surface-2)" }} />
      </div>
    );
  }

  if (state === "error" || values.length === 0 || axes.length === 0) {
    return (
      <div
        className={className}
        style={{
          width, height, borderRadius: 6, display: "flex", alignItems: "center", justifyContent: "center",
          background: "var(--app-surface-2)", color: "var(--app-text-soft)",
        }}
        aria-label="Waveform unavailable"
      >
        <Activity size={Math.min(width, height) * 0.5} strokeWidth={1.5} />
      </div>
    );
  }

  // Decimate to MAX_POINTS via simple striding — a thumb this size can't
  // usefully render every sample of a multi-second recording anyway.
  const step = Math.max(1, Math.floor(values.length / MAX_POINTS));
  const rows = values.filter((_, i) => i % step === 0);
  const numChannels = axes.length;
  const padY = 2;

  const paths = Array.from({ length: numChannels }, (_, ch) => {
    const series = rows.map(r => r[ch]).filter((v): v is number => typeof v === "number" && Number.isFinite(v));
    if (series.length === 0) return "";
    const min = Math.min(...series);
    const max = Math.max(...series);
    const range = max - min || 1;
    return rows
      .map((r, i) => {
        const x = rows.length > 1 ? (i / (rows.length - 1)) * width : width / 2;
        const v = r[ch];
        const norm = typeof v === "number" && Number.isFinite(v) ? (v - min) / range : 0.5;
        const y = height - padY - norm * (height - padY * 2);
        return `${x.toFixed(2)},${y.toFixed(2)}`;
      })
      .join(" ");
  });

  return (
    <svg
      className={className}
      width={width}
      height={height}
      viewBox={`0 0 ${width} ${height}`}
      role="img"
      aria-label={`Waveform preview across ${numChannels} axis${numChannels !== 1 ? "es" : ""}`}
      style={{ borderRadius: 6, background: "var(--app-surface-2)" }}
    >
      {paths.map((points, ch) => points && (
        <polyline
          key={axes[ch] ?? ch}
          points={points}
          fill="none"
          stroke={PREMIUM_SOFT_PALETTE[(ch * 2) % PREMIUM_SOFT_PALETTE.length]}
          strokeWidth={1.2}
          strokeLinejoin="round"
          strokeLinecap="round"
        />
      ))}
    </svg>
  );
}
