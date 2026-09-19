"use client";
import { Wand2 } from "lucide-react";
import { HelpTooltip } from "./common";

/** Shared shell for the four Spectral Analysis DSP-result graphs — all four
 *  are blocked on G20 (no filter-response/wavelet computation exists on the
 *  backend). Rendered visibly-unavailable rather than omitted, per the
 *  honest-degradation rule (§12.6). */
function UnavailableGraph({ title }: { title: string }) {
  return (
    <div>
      <h4 className="text-xs font-bold mb-1.5" style={{ color: "var(--app-text)" }}>{title}</h4>
      <div
        className="rounded-xl flex items-center justify-center text-xs font-medium"
        style={{ minHeight: 90, border: "1px dashed var(--app-border)", background: "var(--app-surface)", color: "var(--app-text-soft)" }}
      >
        Not available yet
      </div>
    </div>
  );
}

export function FilterResponseGraph() {
  return <UnavailableGraph title="Filter response" />;
}

export function AfterFilterGraph() {
  return <UnavailableGraph title="After filter" />;
}

export function WaveletFunctionGraph() {
  return <UnavailableGraph title="Wavelet function" />;
}

export function WaveletApproximationGraph() {
  return <UnavailableGraph title="Wavelet approximation" />;
}

/** `Autotune parameters` — blocked on G21 (no autotuning implementation
 *  anywhere in the backend). Visible, disabled, with an explanatory tooltip
 *  rather than hidden entirely, matching the other Spectral degraded rows. */
export function AutotuneButton() {
  return (
    <button
      type="button"
      disabled
      style={{
        opacity: 0.5,
        cursor: "not-allowed",
        display: "inline-flex",
        alignItems: "center",
        gap: 6,
        padding: "0.4rem 0.75rem",
        fontSize: "0.78rem",
        fontWeight: 600,
        borderRadius: "0.5rem",
        border: "1px solid #6366f1",
        color: "#6366f1",
        background: "transparent",
      }}
      title="Parameter autotuning is not implemented yet."
    >
      <Wand2 size={13} /> Autotune parameters
    </button>
  );
}

/** Flatten's `State` row — blocked on G24 (no such field exists in the
 *  `flatten` block's DSP-result output). */
export function BlockStateCard() {
  return (
    <div>
      <div className="flex items-center gap-1.5 mb-1">
        <h4 className="text-xs font-bold" style={{ color: "var(--app-text)" }}>State</h4>
        <HelpTooltip text="Per-window statistic state is not available for the Flatten block yet (backend gap G24)." />
      </div>
      <div className="pe-dsp-raw-block" style={{ fontStyle: "italic", opacity: 0.7 }}>
        None for these settings
      </div>
    </div>
  );
}
