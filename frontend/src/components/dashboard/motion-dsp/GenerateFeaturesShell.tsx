"use client";
import { useEffect, useMemo, useState } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import toast from "react-hot-toast";
import { useAppStore } from "@/store/appStore";
import { impulsesApi, dspApi, DatasetSummaryResponse } from "@/utils/api";
import { DspTabs, HelpTooltip } from "./common";
import { PerformanceCards } from "./resultCards";
import { Layers, MapIcon, Play, AlertTriangle } from "lucide-react";
import type { BlockType } from "./DspBlockLayout";

/** `Generate features` tab header — title, block context, tab pair. */
export function GenerateFeaturesHeader({
  blockType,
  title,
  impulseName,
  impulseId,
  parametersSaved,
}: {
  blockType: BlockType;
  title: string;
  impulseName?: string;
  impulseId: string | null;
  parametersSaved: boolean;
}) {
  return (
    <>
      <DspTabs blockType={blockType} impulseId={impulseId} active="generate-features" parametersSaved={parametersSaved} />
      <div className="px-5 pt-4">
        <h2 className="text-sm font-semibold" style={{ color: "var(--app-text)" }}>{title} — Generate features</h2>
        {impulseName && (
          <p className="text-xs mt-0.5" style={{ color: "var(--app-text-soft)" }}>{impulseName}</p>
        )}
      </div>
    </>
  );
}

type SummaryState =
  | { status: "loading" }
  | { status: "error"; message: string }
  | { status: "ready"; data: DatasetSummaryResponse };

/** `Training set` summary card (motion_phase4_5.md §11.1, Batch 4). Renders
 *  real `Classes`/`Training windows` from `GET /dsp/dataset-summary` (R25) —
 *  no fabricated or estimated values. `Data in training set` (duration) has
 *  no backend field to source from (G12, still open) and stays permanently
 *  degraded with a stated reason rather than showing a real number.
 *  `Calculate feature importance` / `Normalize features` / `Generate
 *  features` are out of this batch's scope (Batches 5–7) and stay as
 *  not-implemented-yet placeholders. */
export function TrainingSetCard({ impulseId }: { impulseId: string }) {
  const [state, setState] = useState<SummaryState>({ status: "loading" });
  // Cross-check only (§14 Batch 4 note): `/dsp/input-size`'s `window_count`
  // describes one representative sample, while `/dsp/dataset-summary`'s
  // describes the whole training subset — they are only expected to agree
  // when the project has exactly one usable training sample. Fetched
  // independently and compared defensively; never blocks the primary render.
  const [inputSizeWindowCount, setInputSizeWindowCount] = useState<number | null>(null);

  useEffect(() => {
    let cancelled = false;
    setState({ status: "loading" });
    setInputSizeWindowCount(null);

    dspApi.datasetSummary(impulseId)
      .then(({ data }) => {
        if (cancelled) return;
        setState({ status: "ready", data });
      })
      .catch((e: any) => {
        if (cancelled) return;
        const message = e?.response?.data?.detail || e?.message || "Failed to load the dataset summary.";
        setState({ status: "error", message });
      });

    dspApi.inputSize(impulseId)
      .then(({ data }) => {
        if (cancelled) return;
        setInputSizeWindowCount(typeof data.window_count === "number" ? data.window_count : null);
      })
      .catch(() => {
        // Cross-check only — a failure here must never affect the card's
        // primary state, which is sourced from dataset-summary alone.
      });

    return () => { cancelled = true; };
  }, [impulseId]);

  useEffect(() => {
    if (state.status !== "ready" || inputSizeWindowCount == null) return;
    const { total_samples, test_samples, window_count } = state.data;
    const trainingSampleCount = (total_samples ?? 0) + (test_samples ?? 0);
    if (trainingSampleCount === 1 && typeof window_count === "number" && window_count !== inputSizeWindowCount) {
      console.warn(
        `[TrainingSetCard] window_count mismatch for impulse ${impulseId} with a single sample: ` +
        `/dsp/dataset-summary reports ${window_count}, /dsp/input-size reports ${inputSizeWindowCount}.`
      );
    }
  }, [state, inputSizeWindowCount, impulseId]);

  const isEmpty = state.status === "ready" && (state.data.total_samples ?? 0) === 0;

  return (
    <div className="pe-dsp-card">
      <div className="pe-dsp-card-head" style={{ justifyContent: "space-between" }}>
        <div className="flex items-center gap-2">
          <span className="head-icon icon-blue"><Layers size={14} /></span>
          <h3 className="head-title">Training set</h3>
        </div>
        <HelpTooltip text="A summary of the data this impulse will train on." />
      </div>
      <div className="pe-dsp-card-body">
        {state.status === "loading" && (
          <div className="pe-features-info">
            {["Data in training set", "Classes", "Training windows"].map((label) => (
              <div className="pe-features-info-row" key={label}>
                <span className="pe-features-info-label">{label}</span>
                <span className="pe-features-info-value" style={{ opacity: 0.55, fontStyle: "italic" }}>Loading…</span>
              </div>
            ))}
          </div>
        )}

        {state.status === "error" && (
          <div
            className="flex items-start gap-2"
            style={{ marginBottom: "1.4rem", color: "#dc2626", fontSize: "0.85rem" }}
          >
            <AlertTriangle size={15} style={{ flexShrink: 0, marginTop: 2 }} />
            <span>Could not load the dataset summary: {state.message}</span>
          </div>
        )}

        {state.status === "ready" && isEmpty && (
          <div
            className="flex items-start gap-2"
            style={{
              marginBottom: "1.4rem",
              padding: "0.75rem 0.9rem",
              borderRadius: 8,
              background: "var(--app-surface-2)",
              border: "1px solid var(--app-border)",
              color: "var(--app-text-muted)",
              fontSize: "0.83rem",
              lineHeight: 1.5,
            }}
          >
            <AlertTriangle size={15} style={{ flexShrink: 0, marginTop: 2, color: "var(--app-text-soft)" }} />
            <span>
              No samples in the training set yet. Upload and label recordings for this project,
              then return here to generate features.
            </span>
          </div>
        )}

        {state.status === "ready" && !isEmpty && (
          <div className="pe-features-info">
            <div className="pe-features-info-row">
              <span className="pe-features-info-label flex items-center gap-1">
                Data in training set
                <HelpTooltip text="Total recording duration is not computed by the backend yet (motion_phase4_5.md gap G12) — the sample count below is shown instead." />
              </span>
              <span className="pe-features-info-value">
                {state.data.total_samples} sample{state.data.total_samples === 1 ? "" : "s"}
                <span style={{ display: "block", fontWeight: 400, fontSize: "0.72rem", opacity: 0.6, fontStyle: "italic" }}>
                  Duration unavailable (G12)
                </span>
              </span>
            </div>
            <div className="pe-features-info-row">
              <span className="pe-features-info-label">Classes</span>
              <span className="pe-features-info-value">
                {state.data.num_classes}
                {state.data.class_names && state.data.class_names.length > 0
                  ? ` (${state.data.class_names.join(", ")})`
                  : ""}
              </span>
            </div>
            <div className="pe-features-info-row">
              <span className="pe-features-info-label">Training windows</span>
              <span className="pe-features-info-value">
                {typeof state.data.window_count === "number" ? state.data.window_count : "Not available"}
                {typeof state.data.skipped_too_short === "number" && state.data.skipped_too_short > 0 && (
                  <span style={{ display: "block", fontWeight: 400, fontSize: "0.72rem", opacity: 0.6, fontStyle: "italic" }}>
                    {state.data.skipped_too_short} recording{state.data.skipped_too_short === 1 ? "" : "s"} too short for a window
                  </span>
                )}
              </span>
            </div>
          </div>
        )}

        <div className="pe-features-info">
          <div className="pe-features-info-row">
            <span className="pe-features-info-label">Calculate feature importance</span>
            <span className="pe-features-info-value" style={{ opacity: 0.55, fontStyle: "italic" }}>Not implemented yet</span>
          </div>
          <div className="pe-features-info-row">
            <span className="pe-features-info-label">Normalize features</span>
            <span className="pe-features-info-value" style={{ opacity: 0.55, fontStyle: "italic" }}>Not implemented yet</span>
          </div>
        </div>

        <button
          type="button"
          disabled
          className="data-acq-btn-primary"
          style={{ opacity: 0.5, cursor: "not-allowed", marginTop: 12 }}
          title={
            isEmpty
              ? "Generate features is disabled: no samples in the training set yet."
              : "Feature generation dispatch is not implemented in this batch."
          }
        >
          <Play size={14} fill="currentColor" />
          Generate features
        </button>
      </div>
    </div>
  );
}

/** `Feature generation output` — a status-area shell. No job can be
 *  dispatched from this batch, so there is nothing real to log; this states
 *  that plainly rather than fabricating progress lines. */
export function FeatureGenerationStatusPlaceholder() {
  return (
    <div className="pe-dsp-card flex flex-col">
      <div className="pe-dsp-card-head">
        <h3 className="head-title">Feature generation output</h3>
      </div>
      <div className="pe-features-log">
        <div style={{ color: "var(--app-text-soft)" }}>
          Feature generation is not available yet. This panel will show live progress once
          generation is implemented.
        </div>
      </div>
    </div>
  );
}

function FeatureExplorerPlaceholder() {
  return (
    <div className="pe-dsp-card pe-features-explorer" style={{ flex: 1 }}>
      <div className="pe-dsp-card-head">
        <div className="flex items-center gap-2">
          <span className="head-icon icon-blue"><MapIcon size={14} /></span>
          <h3 className="head-title">Feature explorer</h3>
        </div>
      </div>
      <div className="explorer-chart">
        <div className="flex flex-col items-center justify-center p-8 text-center">
          <p className="text-sm font-bold" style={{ color: "var(--app-text)" }}>No features yet</p>
          <p className="text-xs mt-2 max-w-[300px] leading-relaxed" style={{ color: "var(--app-text-muted)" }}>
            The feature explorer will appear here once feature generation is implemented and a run
            has completed.
          </p>
        </div>
      </div>
    </div>
  );
}

/** Shared Generate-features tab shell (§11, Batch 3 scope) — structurally
 *  identical across all three motion DSP blocks. Renders every required
 *  section (header, training-set summary, generate action, status area,
 *  feature explorer, on-device performance) but never calls
 *  `dspApi.generateFeatures` — dispatch/polling/cancel is explicitly out of
 *  scope for this batch (see motion_phase4_5.md §14 Batch 7+). */
export default function GenerateFeaturesShell({
  blockType,
  title,
}: {
  blockType: BlockType;
  title: string;
}) {
  const searchParams = useSearchParams();
  const router = useRouter();
  const impulseId = searchParams.get("impulseId");
  const { activeImpulse, savedActiveImpulse, setActiveImpulse, setSavedActiveImpulse, hasHydrated } = useAppStore();

  const currentImpulse = useMemo(
    () => (impulseId ? (activeImpulse?.id === impulseId ? activeImpulse : null) : activeImpulse),
    [activeImpulse, impulseId]
  );

  const parametersSaved = useMemo(() => !!savedActiveImpulse?.dsp_params_saved_at, [savedActiveImpulse]);

  useEffect(() => {
    if (!impulseId || activeImpulse?.id === impulseId) return;
    let cancelled = false;
    impulsesApi.get(impulseId).then(({ data }) => {
      if (!cancelled) {
        setActiveImpulse(data);
        setSavedActiveImpulse(data);
      }
    }).catch(() => {});
    return () => { cancelled = true; };
  }, [impulseId, activeImpulse?.id, setActiveImpulse, setSavedActiveImpulse]);

  // Gate: unsaved parameters redirect back with ?needsSave=1 (R13, mirrors
  // the image page's pattern, R36).
  useEffect(() => {
    if (!hasHydrated || !currentImpulse) return;
    if (savedActiveImpulse?.id !== currentImpulse.id) return;
    if (parametersSaved) return;
    const qs = impulseId ? `?impulseId=${impulseId}&needsSave=1` : "?needsSave=1";
    router.replace(`/dashboard/impulse/${blockType}/parameters${qs}`);
  }, [hasHydrated, currentImpulse, savedActiveImpulse, parametersSaved, impulseId, router, blockType]);

  if (!currentImpulse) {
    return <div className="p-6 text-gray-400">Loading impulse...</div>;
  }

  return (
    <div className="pe-dsp motion-dsp-page flex flex-col min-h-full">
      <GenerateFeaturesHeader
        blockType={blockType}
        title={title}
        impulseName={currentImpulse.name}
        impulseId={impulseId}
        parametersSaved={parametersSaved}
      />

      <div className="pe-features-body">
        <div className="pe-features-col">
          <TrainingSetCard impulseId={currentImpulse.id} />
          <FeatureGenerationStatusPlaceholder />
        </div>
        <div className="pe-features-col">
          <FeatureExplorerPlaceholder />
          <PerformanceCards />
        </div>
      </div>
    </div>
  );
}
