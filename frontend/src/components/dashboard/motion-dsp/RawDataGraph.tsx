"use client";
import { useEffect, useMemo, useState } from "react";
import { Activity } from "lucide-react";
import { samplesApi } from "@/utils/api";
import { LabelSelector, SampleSelector } from "./selectors";

export const AXIS_PALETTE = ["#6366f1", "#f59e0b", "#10b981", "#ec4899", "#06b6d4", "#a855f7"];

type SignalData = {
  axes: string[];
  values: number[][];
  frequency_hz: number;
  duration_ms: number;
  num_channels: number;
  num_samples: number;
};

/** Multi-axis waveform panel over `/samples/{id}/signal` (R41) — the whole
 *  recording, real data, no window scoping (G4 remains open, so there is
 *  no drag-to-select overlay here; adding one would imply a backend effect
 *  that doesn't exist yet). Shared by all three blocks' Parameters pages —
 *  visualizing the input signal is independent of which processing block is
 *  being configured. */
export default function RawDataGraph({
  selectedSample,
  setSelectedSample,
  selectedLabel,
  setSelectedLabel,
  samples,
  filteredSamples,
  availableLabels,
  loadingSamples,
}: {
  selectedSample: string;
  setSelectedSample: (v: string) => void;
  selectedLabel: string;
  setSelectedLabel: (v: string) => void;
  samples: any[];
  filteredSamples: any[];
  availableLabels: string[];
  loadingSamples: boolean;
}) {
  const [signal, setSignal] = useState<SignalData | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!selectedSample) {
      setSignal(null);
      return;
    }
    let cancelled = false;
    setLoading(true);
    setError(null);
    samplesApi
      .signal(selectedSample)
      .then(({ data }) => {
        if (!cancelled) setSignal(data);
      })
      .catch((e) => {
        if (!cancelled) {
          setSignal(null);
          setError(e?.response?.data?.detail || "Could not decode this recording's signal");
        }
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [selectedSample]);

  const chart = useMemo(() => {
    if (!signal || !signal.values?.length) return null;
    const width = 1000;
    const height = 140;
    const padding = { top: 10, bottom: 18, left: 4, right: 4 };
    const plotW = width - padding.left - padding.right;
    const plotH = height - padding.top - padding.bottom;

    let min = Infinity;
    let max = -Infinity;
    for (const row of signal.values) {
      for (const v of row) {
        if (v < min) min = v;
        if (v > max) max = v;
      }
    }
    if (!isFinite(min) || !isFinite(max)) return null;
    if (min === max) {
      min -= 1;
      max += 1;
    }
    const span = max - min;

    const n = signal.values.length;
    const paths = signal.axes.map((_axis, axisIdx) => {
      const pts = signal.values.map((row, i) => {
        const x = padding.left + (n > 1 ? (i / (n - 1)) * plotW : 0);
        const y = padding.top + plotH - ((row[axisIdx] - min) / span) * plotH;
        return `${x.toFixed(1)},${y.toFixed(1)}`;
      });
      return pts.join(" ");
    });

    return { width, height, paths };
  }, [signal]);

  return (
    <div className="pe-dsp-card">
      <div className="pe-dsp-card-head" style={{ justifyContent: "space-between", flexWrap: "wrap", gap: 12 }}>
        <div className="flex items-center gap-2">
          <span className="head-icon icon-blue"><Activity size={14} /></span>
          <h3 className="head-title">Raw data</h3>
        </div>
        <div style={{ display: "flex", alignItems: "flex-end", gap: 12, flexWrap: "wrap" }}>
          <LabelSelector value={selectedLabel} onChange={setSelectedLabel} labels={availableLabels} disabled={loadingSamples} />
          <SampleSelector value={selectedSample} onChange={setSelectedSample} samples={filteredSamples} loading={loadingSamples} />
        </div>
      </div>
      <div className="pe-dsp-card-body">
        {!loadingSamples && samples.length === 0 ? (
          <div className="w-full rounded-2xl flex items-center justify-center"
            style={{ minHeight: 200, border: "2px dashed var(--app-border)", background: "var(--app-surface)" }}>
            <p className="text-sm font-medium" style={{ color: "var(--app-text-muted)" }}>
              No samples available in your training set
            </p>
          </div>
        ) : loading ? (
          <p className="text-sm font-medium" style={{ color: "var(--app-text-muted)" }}>Loading signal…</p>
        ) : error ? (
          <p className="text-sm font-medium" style={{ color: "#dc2626" }}>{error}</p>
        ) : !chart || !signal ? (
          <p className="text-sm font-medium" style={{ color: "var(--app-text-muted)" }}>Select a recording to preview its signal</p>
        ) : (
          <>
            <svg
              viewBox={`0 0 ${chart.width} ${chart.height}`}
              preserveAspectRatio="none"
              width="100%"
              height={chart.height}
              style={{ display: "block" }}
            >
              {chart.paths.map((points, i) => (
                <polyline
                  key={i}
                  points={points}
                  fill="none"
                  stroke={AXIS_PALETTE[i % AXIS_PALETTE.length]}
                  strokeWidth={1.3}
                  vectorEffect="non-scaling-stroke"
                />
              ))}
            </svg>
            <div className="flex items-center justify-between flex-wrap gap-2" style={{ marginTop: 8 }}>
              <div className="flex items-center gap-3 flex-wrap">
                {signal.axes.map((axis, i) => (
                  <span key={axis} className="text-xs font-medium inline-flex items-center gap-1.5" style={{ color: "var(--app-text-muted)" }}>
                    <span style={{ width: 8, height: 8, borderRadius: 999, background: AXIS_PALETTE[i % AXIS_PALETTE.length], display: "inline-block" }} />
                    {axis}
                  </span>
                ))}
              </div>
              <span className="text-[10px]" style={{ color: "var(--app-text-soft)" }}>
                {signal.duration_ms} ms · {signal.frequency_hz.toFixed(1)} Hz · {signal.num_samples} samples
              </span>
            </div>
          </>
        )}
      </div>
    </div>
  );
}
