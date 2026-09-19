"use client";
import { useEffect, useState, useCallback } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import { useAppStore } from "@/store/appStore";
import { impulsesApi, samplesApi, dspApi } from "@/utils/api";
import {
  Plus, Trash2, Zap, RefreshCw, Settings,
  CheckCircle, Activity, Database, FlaskConical,
  X, AlertCircle,
  Rocket, User, ArrowRight,
} from "lucide-react";
import toast from "react-hot-toast";
import PremiumSaveButton from "@/components/PremiumSaveButton";
import ImpulseNotReady from "@/components/dashboard/ImpulseNotReady";
import {
  Arrow, SadFolderIcon, ProcessingBlockModal, BlockPickerModal,
  PETAL_EDGE_AUTHOR, normalizeLearningAuthor,
  BLOCK_ACCENT, PROCESSING_BLOCK_BADGE, DEFAULT_PROCESSING_BADGE, LEARNING_BLOCK_BADGE,
} from "./ImpulseShared";

const OBJECT_DETECTION_LEARNING_LABEL = "Object Detection (Images)";

function isObjectDetectionLearningBlock(block: any) {
  const type = String(block?.type ?? "").toLowerCase();
  const name = String(block?.name ?? "").toLowerCase();
  return type === "object_detection" || name === OBJECT_DETECTION_LEARNING_LABEL.toLowerCase();
}

/* ─────────────────────────────────────────────────────────────────────────────
   New-impulse default payload — sent straight to POST /impulses/. The backend
   assigns the canonical "Impulse N" name from its monotonic per-project
   counter, so we deliberately omit `name` here.
───────────────────────────────────────────────────────────────────────────── */
function buildEmptyImpulsePayload(projectId: string) {
  return {
    project_id: projectId,
    window_size_ms: 1000,
    window_increase_ms: 500,
    frequency_hz: 100,
    input_type: "image",
    input_axes: ["image"],
    sensor_type: "camera",
    image_width: 96,
    image_height: 96,
    resize_mode: "Fit shortest axis",
    train_subset_percent: 100,
    dsp_blocks: [],
    ml_blocks: [],
  };
}

/* ─────────────────────────────────────────────────────────────────────────────
   Resize-mode preview icon — 👤 → 👤 pair with axis-specific "after" shape.
   Single source of truth for both the Input block's Resize-mode picker and the
   Processing — Image card's empty-slot illustration.
───────────────────────────────────────────────────────────────────────────── */
function ResizeModePreview({
  mode,
  scale = 1,
  showTip = true,
}: {
  mode: string;
  scale?: number;
  showTip?: boolean;
}) {
  // Source box: a portrait silhouette in a square frame — same for every mode.
  const beforeSize = 32 * scale;
  const beforeIconSize = 20 * scale;
  const arrowSize = 14 * scale;

  // Each mode renders a visually distinct destination. The destination box
  // is the same size for all modes (so the layout doesn't jump when the mode
  // changes), and the icon inside is what differs:
  //   • Squash             → icon stretched horizontally (full-bleed, distorted)
  //   • Fit shortest axis  → icon larger than the box (content cropped, only
  //                          the upper torso/head shows after centering)
  //   • Fit longest axis   → icon smaller than the box (whole figure fits with
  //                          visible padding — letterbox effect)
  const destSize = 32 * scale;
  let destIcon: { size: number; style?: React.CSSProperties };
  if (mode === "Fit shortest axis") {
    destIcon = {
      size: 34 * scale,
      // Pull the figure down a touch so the crop visibly slices the head/torso
      // instead of just clipping symmetrically (mirrors how center-crop on a
      // tall portrait keeps the face roughly in frame).
      style: { transform: `translateY(${5 * scale}px)` },
    };
  } else if (mode === "Fit longest axis") {
    destIcon = { size: 13 * scale };
  } else {
    // Squash (default). Lucide renders a viewBox-locked SVG, so scaleX
    // visibly squashes the silhouette wider while keeping height.
    destIcon = {
      size: 20 * scale,
      style: { transform: "scaleX(1.55)" },
    };
  }

  const tip =
    mode === "Fit longest axis"
      ? "'Fit longest axis' will resize the image so that the longest axis fits, and then applies padding."
      : mode === "Fit shortest axis"
        ? "'Fit shortest axis' will resize the image so that the shortest axis fits, and then crops the image to the center."
        : "'Squash' will resize the image and ignore aspect ratio.";

  return (
    <div
      role="img"
      aria-label={`${mode} preview`}
      className={`relative group/rp inline-flex items-center gap-1 px-1.5 py-1 rounded flex-shrink-0 ${showTip ? "cursor-help" : ""}`}
    >
      <span
        className="inline-flex items-center justify-center rounded-md bg-gray-200 text-gray-500 overflow-hidden"
        style={{ width: beforeSize, height: beforeSize }}
      >
        <User size={beforeIconSize} strokeWidth={1.75} fill="currentColor" />
      </span>
      <ArrowRight size={arrowSize} strokeWidth={2.25} className="text-gray-300" />
      <span
        className="inline-flex items-center justify-center rounded-md bg-gray-200 text-gray-500 overflow-hidden"
        style={{ width: destSize, height: destSize }}
      >
        <User
          size={destIcon.size}
          strokeWidth={1.75}
          fill="currentColor"
          style={destIcon.style}
        />
      </span>
      {showTip && (
        <div role="tooltip" className="pointer-events-none absolute right-0 top-full mt-1 z-[100] w-[280px] max-w-[90vw] rounded-md bg-slate-800 text-white text-[11px] leading-relaxed p-2.5 shadow-xl opacity-0 group-hover/rp:opacity-100 transition-opacity whitespace-normal break-words">
          {tip}
        </div>
      )}
    </div>
  );
}

/* ─────────────────────────────────────────────────────────────────────────────
   "Add a learning block" modal — matches EI screenshot exactly. Object
   Detection only ever offers the one architecture entry here; picking a
   different base model is a later step (see the dormant "Choose a different
   model" modal below, kept as-is — not wired into any button today).
───────────────────────────────────────────────────────────────────────────── */
function LearningBlockModal({
  impulseId, projectType, onAdd, onClose,
}: {
  impulseId: string;
  projectType?: string;
  onAdd: (block: any) => void;
  onClose: () => void;
}) {
  const [blocks, setBlocks] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);

  const load = useCallback(async () => {
    setLoading(true);
    try {
      const { data } = await dspApi.learningBlocks(impulseId, false, projectType);
      const filteredBlocks = (data.blocks || [])
        .filter((block: any) => isObjectDetectionLearningBlock(block))
        .map((block: any) => ({
          ...block,
          name: OBJECT_DETECTION_LEARNING_LABEL,
          author: normalizeLearningAuthor(block.author),
        }));
      setBlocks(filteredBlocks);
    } catch { toast.error("Failed to load learning blocks"); }
    finally { setLoading(false); }
  }, [impulseId, projectType]);

  useEffect(() => { load(); }, [load]);

  return (
    <BlockPickerModal
      titleIcon={FlaskConical}
      title="Add a learning block"
      loading={loading}
      blocks={blocks}
      emptyMessage="No learning blocks are available for this impulse."
      showFooterCancel
      onAdd={onAdd}
      onClose={onClose}
    />
  );
}

/* ─────────────────────────────────────────────────────────────────────────────
   "Choose a different model" modal — matches EI screenshot exactly
───────────────────────────────────────────────────────────────────────────── */
function ChooseModelModal({
  onAdd, onClose,
}: {
  onAdd: (model: any) => void;
  onClose: () => void;
}) {
  const models = [
    {
      id: "mobilenet_v2_ssd_fpn_lite",
      name: "EdgeDetect Lite (320x320 only)",
      author: PETAL_EDGE_AUTHOR,
      description: "EdgeDetect Lite for object detection. Good for devices with > 512KB RAM or Linux.",
      official: true,
      preview: false,
      unavailable: true,
    },
    {
      id: "fomo_mobilenetv2_0_1",
      name: "NanoVision MobileNetV2 0.35",
      author: PETAL_EDGE_AUTHOR,
      description: "NanoVision is a novel object detection architecture that's specifically designed to run on constrained devices.",
      official: true,
      preview: false,
    },
    {
      id: "yolo_pro",
      name: "Vision Pro",
      author: "Ultralytics",
      description: "Vision Pro state of the art object detection. Good for Linux devices or constrained edge devices with NPUs.",
      official: false,
      preview: true,
    }
  ];

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm">
      <div className="bg-white rounded-2xl shadow-2xl w-[680px] max-h-[85vh] flex flex-col overflow-hidden">

        {/* Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-gray-200">
          <div className="flex items-center gap-2">
            <FlaskConical size={17} className="text-indigo-600" />
            <h2 className="text-[15px] font-semibold text-gray-900">Choose a different model</h2>
          </div>
          <button onClick={onClose} className="text-gray-400 hover:text-gray-600">
            <X size={17} />
          </button>
        </div>

        {/* Column headers */}
        <div className="grid grid-cols-[1fr_130px_100px] px-6 py-2 bg-gray-50 border-b border-gray-200">
          <span className="text-[11px] font-semibold text-gray-400 uppercase tracking-wider">Description</span>
          <span className="text-[11px] font-semibold text-gray-400 uppercase tracking-wider">Author</span>
          <span className="text-[11px] font-semibold text-gray-400 uppercase tracking-wider text-right pr-2"></span>
        </div>

        {/* Block rows */}
        <div className="flex-1 overflow-y-auto divide-y divide-gray-100">
          {models.map(model => (
            <div key={model.id}
              className={`grid grid-cols-[1fr_130px_100px] items-center px-6 py-4 transition-colors ${model.unavailable ? "opacity-60" : "hover:bg-gray-50"}`}>
              <div className="pr-4">
                <div className="flex items-center flex-wrap gap-2 mb-1">
                  <span className="text-sm font-semibold text-gray-900">{model.name}</span>
                  {model.official && (
                    <span className="px-1.5 py-px rounded-full bg-gray-100 text-gray-500 text-[10px] font-semibold uppercase tracking-wide border border-gray-200">
                      OFFICIALLY SUPPORTED
                    </span>
                  )}
                  {model.preview && (
                    <span className="px-1.5 py-px rounded-full bg-purple-50 text-purple-600 text-[10px] font-semibold uppercase tracking-wide border border-purple-200">
                      DEVELOPER PREVIEW
                    </span>
                  )}
                  {model.unavailable && (
                    <span className="px-1.5 py-px rounded-full bg-gray-100 text-gray-500 text-[10px] font-semibold uppercase tracking-wide border border-gray-200">
                      UNAVAILABLE
                    </span>
                  )}
                </div>
                <p className="text-sm text-indigo-700 leading-snug">{model.description}</p>
              </div>
              <span className="text-sm text-gray-600">{model.author}</span>
              <div className="flex items-center justify-end gap-2 pr-1">
                <button
                  disabled={model.unavailable}
                  onClick={() => { onAdd(model); onClose(); }}
                  className="px-4 py-1.5 text-sm font-medium text-indigo-700 border border-indigo-300 rounded-lg transition-colors hover:bg-indigo-50 disabled:cursor-not-allowed disabled:border-gray-200 disabled:text-gray-400 disabled:hover:bg-transparent"
                >
                  {model.unavailable ? "Unavailable" : "Add"}
                </button>
              </div>
            </div>
          ))}
        </div>

        {/* Upgrade banner */}
        <div className="mx-6 my-2 p-3 bg-indigo-50 border border-indigo-100 rounded-xl flex items-center gap-2 text-sm text-gray-700">
          <Zap size={13} className="text-indigo-500 flex-shrink-0" />
          Want access to all enterprise base models?{" "}
          <a href="#" className="text-indigo-600 hover:underline font-medium">Upgrade now.</a>
        </div>

        {/* Footer */}
        <div className="px-6 py-3 border-t border-gray-200 bg-gray-50">
          <div className="flex justify-end">
            <button onClick={onClose}
              className="px-4 py-2 text-sm font-medium text-gray-700 bg-white border border-gray-300 rounded-lg hover:bg-gray-50 transition-colors">
              Cancel
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

/* ─────────────────────────────────────────────────────────────────────────────
   Main page
───────────────────────────────────────────────────────────────────────────── */
export default function ObjectDetectionImpulse() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const { activeProject, activeImpulse, savedActiveImpulse, setActiveImpulse, setSavedActiveImpulse, hasHydrated } = useAppStore();
  // Generate features is unlocked only when the backend has confirmed the
  // Parameters save (dsp_params_saved_at). Field-presence checks are NOT
  // sufficient — defaults populated at impulse creation would unlock it.
  const generateFeaturesEnabled = (() => {
    if (!activeImpulse || savedActiveImpulse?.id !== activeImpulse.id) return false;
    return !!savedActiveImpulse?.dsp_params_saved_at;
  })();

  const [impulses, setImpulses] = useState<any[]>([]);
  const [saving, setSaving] = useState(false);

  const [showDSPModal, setShowDSPModal] = useState(false);
  const [showMLModal, setShowMLModal] = useState(false);
  const [activeDspIdx, setActiveDspIdx] = useState<number | null>(null);

  const [validationErrors, setValidationErrors] = useState<string[]>([]);

  // Local string state for the "Train on data subset" input so the user can
  // briefly hold an empty value while backspacing without us snapping it to a
  // number. Committed (clamped int) value lives on activeImpulse.
  const [subsetDraft, setSubsetDraft] = useState<string | null>(null);

  // null = readiness check in flight, true = at least one sample exists,
  // false = project has no uploaded training data yet → empty-data screen.
  const [hasData, setHasData] = useState<boolean | null>(null);

  // Output labels — distinct *assigned* labels across the project's samples.
  // Server-side aggregated via `samplesApi.labelsSummary`; the previous
  // client-side scan pulled up to 10k samples with full metadata and stalled
  // both Output features cards for seconds. `loading` lets us paint a subtle
  // skeleton instead of "0 (no labels assigned)" while the fetch is in flight.
  const [outputLabels, setOutputLabels] = useState<{ count: number; names: string[] }>({ count: 0, names: [] });
  const [outputLabelsLoading, setOutputLabelsLoading] = useState(false);

  /* ── Bootstrap ─────────────────────────────────────────────────── */
  // Gate on hasHydrated: without this, the effect can fire on the first
  // render with activeImpulse=null (zustand persist is async) and then fall
  // through to `setActiveImpulse(data[0])` — clobbering a draft that the
  // store will hand back a tick later. Waiting for hydration means the
  // "is it a draft?" check inside loadImpulses sees the real value.
  useEffect(() => {
    if (hasHydrated && activeProject) loadImpulses();
  }, [hasHydrated, activeProject]);

  /* ── Detect empty dataset (no samples uploaded for this project) ─ */
  useEffect(() => {
    if (!activeProject?.id) {
      setHasData(null);
      return;
    }
    let cancelled = false;
    setHasData(null);
    samplesApi
      .list(activeProject.id, { limit: 1 })
      .then(({ data }) => {
        if (cancelled) return;
        const count = Array.isArray(data)
          ? data.length
          : Array.isArray(data?.items)
            ? data.items.length
            : Number(data?.total ?? data?.count ?? 0);
        setHasData(count > 0);
      })
      .catch(() => {
        if (!cancelled) setHasData(false);
      });
    return () => {
      cancelled = true;
    };
  }, [activeProject?.id]);

  /* ── Load distinct assigned labels for Output features cards ─────
     Single fetch keyed by project id; both the Learning block card and the
     Output features card read from the same `outputLabels` state, so the
     network round-trip happens once per project switch — never twice, never
     on every re-render. */
  useEffect(() => {
    if (!activeProject?.id) {
      setOutputLabels({ count: 0, names: [] });
      setOutputLabelsLoading(false);
      return;
    }
    let cancelled = false;
    setOutputLabelsLoading(true);
    samplesApi
      .labelsSummary(activeProject.id)
      .then(({ data }) => {
        if (cancelled) return;
        const names: string[] = Array.isArray(data?.names) ? data.names : [];
        const count: number = typeof data?.count === "number" ? data.count : names.length;
        setOutputLabels({ count, names });
      })
      .catch(() => {
        if (!cancelled) setOutputLabels({ count: 0, names: [] });
      })
      .finally(() => {
        if (!cancelled) setOutputLabelsLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [activeProject?.id]);

  /* ── Loaders ────────────────────────────────────────────────────── */
  async function loadImpulses() {
    if (!activeProject) return;

    try {
      const { data } = await impulsesApi.list(activeProject.id);
      setImpulses(data);

      if (data.length === 0) return;

      // Priority order:
      // 1. impulseId from URL  → fetch it and use it
      // 2. existing activeImpulse that belongs to this project → keep it
      // 3. first impulse in list → fallback only
      const urlId = searchParams.get("impulseId");
      if (urlId) {
        const match = data.find((i: any) => i.id === urlId);
        if (match) {
          setActiveImpulse(match);
          setSavedActiveImpulse(match);
          return;
        }
        // Not in list — fetch directly (e.g. cross-project deep link)
        try {
          const { data: fetched } = await impulsesApi.get(urlId);
          setActiveImpulse(fetched);
          setSavedActiveImpulse(fetched);
          return;
        } catch { /* fall through */ }
      }
      if (activeImpulse && data.some((i: any) => i.id === activeImpulse.id)) {
        // Store has a valid impulse for this project — leave it alone
        return;
      }
      setActiveImpulse(data[0]);
      setSavedActiveImpulse(data[0]);
    } catch (err) {
      console.error("Failed to load impulses:", err);
      toast.error("Failed to load impulses");
    }
  }

  /* ── Impulse CRUD ───────────────────────────────────────────────── */
  async function createImpulse() {
    if (!activeProject) return;
    // Persist immediately. The new impulse appears in the sidebar dropdown
    // and Manage Impulses table right away; the builder shows an empty
    // pipeline that the user fills in over time. Subsequent edits round-trip
    // through Save Impulse (PUT), not another POST.
    try {
      const { data } = await impulsesApi.create(buildEmptyImpulsePayload(activeProject.id));
      setActiveImpulse(data);
      setSavedActiveImpulse(data);
      setImpulses((rows) => (rows.some((r) => r.id === data.id) ? rows : [...rows, data]));
      setValidationErrors([]);
      setActiveDspIdx(null);
      window.dispatchEvent(new Event("impulses:changed"));
      router.replace(`/dashboard/impulse?impulseId=${data.id}`);
    } catch (e: any) {
      toast.error(e?.response?.data?.detail || "Failed to create impulse");
    }
  }

  async function saveImpulse() {
    if (!activeImpulse) return;
    setSaving(true);

    // Keep the root input type aligned with the current DSP pipeline so the
    // backend never saves an image block under a time-series impulse.
    const payload: any = { ...activeImpulse };
    const hasImageDSP = (payload.dsp_blocks || []).some((block: any) => block?.type === "image");
    payload.input_type = hasImageDSP || payload.input_type === "image" ? "image" : "time-series";

    // Auto-sync root image values to the image DSP block before saving.
    if (payload.input_type === "image" && payload.dsp_blocks?.length > 0) {
      payload.dsp_blocks = payload.dsp_blocks.map((block: any) => (
        block?.type === "image"
          ? {
            ...block,
            params: {
              ...block.params,
              image_width: payload.image_width,
              image_height: payload.image_height,
            },
          }
          : block
      ));
    }

    try {
      const { data } = await impulsesApi.update(payload.id, payload);
      setActiveImpulse(data);
      setSavedActiveImpulse(data);
      setImpulses((rows) => {
        const idx = rows.findIndex((imp) => imp.id === data.id);
        if (idx >= 0) {
          const next = rows.slice();
          next[idx] = data;
          return next;
        }
        return [...rows, data];
      });
    } catch (e: any) {
      toast.error(e.message || "Save failed");
      throw e;
    } finally { setSaving(false); }
  }

  /* ── Block management — local edits only, no per-edit API calls.
     All pipeline mutations stay in activeImpulse state; Save Impulse is the
     single commit point that sends the consolidated config to the backend. ── */
  function addDSPBlock(block: any) {
    if (!activeImpulse) return;
    const initialParams: Record<string, any> =
      block.type === "image"
        ? {
          image_width: activeImpulse.image_width ?? 96,
          image_height: activeImpulse.image_height ?? 96,
        }
        : {};
    const newBlock = { type: block.type, name: block.name, params: initialParams };
    const nextBlocks = [...(activeImpulse.dsp_blocks || []), newBlock];
    const nextImpulse: any = { ...activeImpulse, dsp_blocks: nextBlocks };
    if (block.type === "image") nextImpulse.input_type = "image";
    setActiveImpulse(nextImpulse);
    if (block.type === "image") {
      setActiveDspIdx(nextBlocks.length - 1);
      toast.success("Image block added — configure dimensions below");
    }
  }

  function removeDSPBlock(idx: number) {
    if (!activeImpulse) return;
    const nextBlocks = (activeImpulse.dsp_blocks || []).filter((_: any, i: number) => i !== idx);
    setActiveImpulse({ ...activeImpulse, dsp_blocks: nextBlocks });
    if (activeDspIdx === idx) setActiveDspIdx(null);
  }

  function addMLBlock(block: any) {
    if (!activeImpulse) return;
    // EI allows exactly one learning block — replace any existing entry.
    const newBlock = { type: block.type, name: block.name, params: {} };
    setActiveImpulse({ ...activeImpulse, ml_blocks: [newBlock] });
  }

  function removeMLBlock(idx: number) {
    if (!activeImpulse) return;
    const nextBlocks = (activeImpulse.ml_blocks || []).filter((_: any, i: number) => i !== idx);
    setActiveImpulse({ ...activeImpulse, ml_blocks: nextBlocks });
  }

  function updateField(key: string, value: any) {
    if (!activeImpulse) return;

    const usesFomo = (activeImpulse.ml_blocks || []).some(
      (block: any) => (block?.type ?? block?.architecture) === "fomo_mobilenetv2_0_1"
    );
    const nextValue =
      (key === "image_width" || key === "image_height") && usesFomo
        ? Math.max(Number(value) || 0, 96)
        : value;

    let nextImpulse = { ...activeImpulse, [key]: nextValue };

    // Sync image width/height to any existing image DSP block in local state
    if ((key === "image_width" || key === "image_height") && nextImpulse.dsp_blocks) {
      nextImpulse.dsp_blocks = nextImpulse.dsp_blocks.map((block: any) => (
        block?.type === "image"
          ? { ...block, params: { ...block.params, [key]: nextValue } }
          : block
      ));
    }

    if (key === "input_type") {
      if (nextValue === "image") {
        nextImpulse.input_axes = ["image"];
        nextImpulse.sensor_type = "camera";
      }
    }

    setActiveImpulse(nextImpulse);
  }

  /* ── Derived ────────────────────────────────────────────────────── */
  const dspBlocks = activeImpulse?.dsp_blocks || [];
  const mlBlocks = activeImpulse?.ml_blocks || [];
  const isFomoImpulse = mlBlocks.some((block: any) => (block?.type ?? block?.architecture) === "fomo_mobilenetv2_0_1");
  const outputReady = dspBlocks.length > 0 && mlBlocks.length > 0;
  // Shared "<count> (<names>)" string consumed by both the Learning block
  // (Object Detection) card and the green Output features card.
  const outputLabelsSummary = outputLabels.count > 0
    ? `${outputLabels.count} (${outputLabels.names.join(", ")})`
    : null;

  /* ── Render ─────────────────────────────────────────────────────── */
  return (
    <div className="pe-impulse impulse-page min-h-screen">

      {/* Header */}
      <div className="mb-8">
        <div className="flex items-start justify-between gap-6 flex-wrap">
          <div className="flex items-start gap-4 min-w-0">
            <div className="pe-impulse-header-icon flex-shrink-0">
              <Activity size={22} strokeWidth={2.2} />
            </div>
            <div className="min-w-0">
              <h1 className="pe-impulse-title">{activeImpulse?.name || "Impulse Design"}</h1>
              <p className="pe-impulse-sub">
                An impulse takes raw data, uses signal processing to extract features,
                and then uses a learning block to classify new data.
              </p>
            </div>
          </div>
          <div className="flex items-center gap-3 flex-shrink-0">
            <button onClick={() => router.push("/dashboard/impulse/manage")}
              className="pe-impulse-toolbar-btn">
              <Settings size={14} /> Manage impulse
            </button>
            <button onClick={createImpulse} className="pe-impulse-toolbar-btn">
              <Plus size={14} /> New impulse
            </button>
            {activeImpulse && (
              <PremiumSaveButton
                onSave={saveImpulse}
                onSuccess={() => toast.success("Impulse saved")}
                label="Save Impulse"
                loadingLabel="Saving..."
                disabled={saving}
              />
            )}
          </div>
        </div>
      </div>

      {activeProject && hasData === null ? (
        /* ── Readiness check in flight ─────────────────────────────────────
           Until the sample-count check resolves we can't tell whether to show
           the "No data collected yet" warning or the main pipeline. Render a
           loading state first so the warning is the first non-loading frame
           for empty projects. */
        <div className="flex flex-col items-center justify-center py-32">
          <span className="inline-flex items-center gap-2 text-sm text-[var(--app-text-soft)]">
            <RefreshCw size={14} className="animate-spin" />
            Checking project data…
          </span>
        </div>
      ) : hasData === false ? (
        /* ── No uploaded data: takes priority over the "create impulse" state ─
           A project without samples can't produce a meaningful impulse, so
           steer the user to data acquisition first regardless of whether an
           impulse has been created. */
        <ImpulseNotReady
          title="No data collected yet"
          description="You'll need some training data to design your first impulse."
          icon={<SadFolderIcon />}
          actions={
            <button
              type="button"
              onClick={() => router.push("/dashboard/data")}
              className="pe-warn-cta"
            >
              <Rocket size={15} /> Go to data acquisition
            </button>
          }
        />
      ) : !activeImpulse ? (
        <div className="flex flex-col items-center justify-center py-32 border-2 border-dashed border-gray-700 rounded-2xl">
          <Zap size={36} className="text-gray-600 mb-4" />
          <p className="text-gray-400 text-sm mb-4">No impulse yet — create one to build your pipeline</p>
          <button onClick={createImpulse}
            className="flex items-center gap-2 px-4 py-2 text-sm font-medium text-white bg-brand-600 hover:bg-brand-500 rounded-lg transition-colors">
            <Plus size={14} /> Create impulse
          </button>
        </div>
      ) : (
        <>
          {/* ── Pipeline canvas ──────────────────────────────────────── */}
          <div className="impulse-canvas">
            <div className="overflow-x-auto pb-2">
              <div className="flex min-w-max items-start justify-center gap-0 xl:min-w-full">

                {/* Input block */}
                <div className="impulse-input-block w-[clamp(17rem,22vw,20rem)] min-h-[26rem] border border-gray-700 rounded-xl flex-shrink-0 overflow-visible relative">
                  <div className="impulse-input-block-header bg-gradient-to-br from-red-500 to-orange-500 px-5 py-4 flex items-center justify-between rounded-t-xl">
                    <div>
                      <p className="text-base font-semibold text-white">
                        {activeImpulse.input_type === "image" ? "Image data" : "Time-series data"}
                      </p>
                    </div>
                    <div className="impulse-input-icon h-10 w-10 bg-black/20 rounded-full flex items-center justify-center">
                      <Database size={20} className="text-white" />
                    </div>
                  </div>
                  <div className="p-4 sm:p-5 space-y-3">
                    {activeImpulse.input_type === "image" ? (
                      <>
                        <div>
                          <p className="text-[10px] text-gray-500 mb-0.5">Input axes</p>
                          <input type="text" disabled className="impulse-field w-full rounded px-2 py-1 text-xs cursor-not-allowed focus:outline-none"
                            value="image" />
                        </div>
                        <div className="grid grid-cols-2 gap-2 mt-2">
                          <div>
                            <div className="relative group/iw inline-flex items-center gap-1 cursor-help mb-0.5">
                              <p className="text-[10px] text-gray-500">Width</p>
                              <button
                                type="button"
                                aria-label="About image width"
                                aria-describedby="tip-image-width"
                                className="text-[9px] text-gray-500 leading-none focus:outline-none focus-visible:ring-1 focus-visible:ring-brand-400 rounded-full"
                              >
                                &#9432;
                              </button>
                              <div
                                id="tip-image-width"
                                role="tooltip"
                                className="pointer-events-none absolute left-0 top-full mt-1 z-[100] w-[240px] max-w-[90vw] rounded-md bg-slate-800 text-white text-[11px] leading-relaxed p-2.5 shadow-xl opacity-0 group-hover/iw:opacity-100 group-focus-within/iw:opacity-100 transition-opacity whitespace-normal break-words"
                              >
                                We resize all images to equal dimensions before training. This is the width all images are resized to.
                              </div>
                            </div>
                            <input type="number" min={isFomoImpulse ? 96 : 1} aria-describedby="tip-image-width" className="impulse-field w-full rounded px-2 py-1 text-xs focus:border-brand-500 focus:outline-none"
                              value={activeImpulse.image_width} onChange={e => updateField("image_width", Number(e.target.value))} />
                          </div>
                          <div>
                            <div className="relative group/ih inline-flex items-center gap-1 cursor-help mb-0.5">
                              <p className="text-[10px] text-gray-500">Height</p>
                              <button
                                type="button"
                                aria-label="About image height"
                                aria-describedby="tip-image-height"
                                className="text-[9px] text-gray-500 leading-none focus:outline-none focus-visible:ring-1 focus-visible:ring-brand-400 rounded-full"
                              >
                                &#9432;
                              </button>
                              <div
                                id="tip-image-height"
                                role="tooltip"
                                className="pointer-events-none absolute left-0 top-full mt-1 z-[100] w-[240px] max-w-[90vw] rounded-md bg-slate-800 text-white text-[11px] leading-relaxed p-2.5 shadow-xl opacity-0 group-hover/ih:opacity-100 group-focus-within/ih:opacity-100 transition-opacity whitespace-normal break-words"
                              >
                                We resize all images to equal dimensions before training. This is the height all images are resized to.
                              </div>
                            </div>
                            <input type="number" min={isFomoImpulse ? 96 : 1} aria-describedby="tip-image-height" className="impulse-field w-full rounded px-2 py-1 text-xs focus:border-brand-500 focus:outline-none"
                              value={activeImpulse.image_height} onChange={e => updateField("image_height", Number(e.target.value))} />
                          </div>
                        </div>
                        <div>
                          <div className="relative group/rm inline-flex items-center gap-1 cursor-help mb-0.5">
                            <p className="text-[10px] text-gray-500">Resize mode</p>
                            <span className="text-[9px] text-gray-500 leading-none">&#9432;</span>
                            <div role="tooltip" className="pointer-events-none absolute left-0 top-full mt-1 z-[100] w-[300px] max-w-[90vw] rounded-md bg-slate-800 text-white text-[11px] leading-relaxed p-2.5 shadow-xl opacity-0 group-hover/rm:opacity-100 transition-opacity whitespace-normal break-words">
                              {"'Squash' will resize via interpolation, using the entire image. The longer axis will look squashed. 'Fit shortest axis' will first crop the outsides of the longer axis to the desired aspect ratio, then interpolate to desired size. 'Fit longest' will 'letterbox' the short axis to the desired aspect ratio, then interpolate."}
                            </div>
                          </div>
                          <div className="flex items-center gap-2">
                            <div className="flex-1 min-w-0">
                              <select className="impulse-field w-full rounded px-2 py-1 text-xs focus:border-brand-500 focus:outline-none"
                                value={activeImpulse.resize_mode} onChange={e => updateField("resize_mode", e.target.value)}>
                                {["Fit shortest axis", "Fit longest axis", "Squash"].map(m => <option key={m}>{m}</option>)}
                              </select>
                            </div>
                            <ResizeModePreview mode={activeImpulse.resize_mode} />

                          </div>
                        </div>
                      </>
                    ) : (
                      <>
                        <div>
                          <p className="text-[10px] text-gray-500 mb-0.5">Input axes</p>
                          <input type="text" className="impulse-field w-full rounded px-2 py-1 text-xs focus:border-brand-500 focus:outline-none"
                            value={(activeImpulse.input_axes || []).join(", ")}
                            onChange={e => updateField("input_axes", e.target.value.split(',').map(s => s.trim()).filter(Boolean))} />
                        </div>
                        <div className="grid grid-cols-2 gap-2 mt-2">
                          <div>
                            <p className="text-[10px] text-gray-500 mb-0.5">Window (ms)</p>
                            <input type="number" className="impulse-field w-full rounded px-2 py-1 text-xs focus:border-brand-500 focus:outline-none"
                              value={activeImpulse.window_size_ms} onChange={e => updateField("window_size_ms", Number(e.target.value))} />
                          </div>
                          <div>
                            <p className="text-[10px] text-gray-500 mb-0.5">Increase (ms)</p>
                            <input type="number" className="impulse-field w-full rounded px-2 py-1 text-xs focus:border-brand-500 focus:outline-none"
                              value={activeImpulse.window_increase_ms} onChange={e => updateField("window_increase_ms", Number(e.target.value))} />
                          </div>
                        </div>
                        <div>
                          <p className="text-[10px] text-gray-500 mb-0.5">Frequency (Hz)</p>
                          <input type="number" className="impulse-field w-full rounded px-2 py-1 text-xs focus:border-brand-500 focus:outline-none"
                            value={activeImpulse.frequency_hz} onChange={e => updateField("frequency_hz", Number(e.target.value))} />
                        </div>
                      </>
                    )}
                    <div>
                      <div className="relative group/ts inline-flex items-center gap-1 cursor-help mb-0.5">
                        <p className="text-[10px] text-gray-500">Train on data subset</p>
                        <button
                          type="button"
                          aria-label="About training data subset"
                          aria-describedby="tip-train-subset"
                          className="text-[9px] text-gray-500 leading-none focus:outline-none focus-visible:ring-1 focus-visible:ring-brand-400 rounded-full"
                        >
                          &#9432;
                        </button>
                        <div
                          id="tip-train-subset"
                          role="tooltip"
                          className="pointer-events-none absolute left-0 top-full mt-1 z-[100] w-[260px] max-w-[90vw] rounded-md bg-slate-800 text-white text-[11px] leading-relaxed p-2.5 shadow-xl opacity-0 group-hover/ts:opacity-100 group-focus-within/ts:opacity-100 transition-opacity whitespace-normal break-words"
                        >
                          Train using a subset percentage of your dataset to accelerate feature generation and training.
                        </div>
                      </div>
                      <div className="flex items-center gap-1.5">
                        <input
                          type="number"
                          min={1}
                          max={100}
                          step={1}
                          inputMode="numeric"
                          aria-describedby="tip-train-subset"
                          className="impulse-field w-20 rounded px-2 py-1 text-xs focus:border-brand-500 focus:outline-none"
                          value={subsetDraft ?? (activeImpulse.train_subset_percent ?? 100)}
                          onKeyDown={e => {
                            if ([".", ",", "e", "E", "+", "-"].includes(e.key)) e.preventDefault();
                          }}
                          onPaste={e => {
                            e.preventDefault();
                            const raw = e.clipboardData.getData("text").replace(/\D/g, "");
                            if (raw === "") return;
                            const n = Math.min(100, Math.max(1, parseInt(raw, 10)));
                            setSubsetDraft(null);
                            updateField("train_subset_percent", n);
                          }}
                          onChange={e => {
                            const raw = e.target.value.replace(/\D/g, "");
                            if (raw === "") {
                              setSubsetDraft("");
                              return;
                            }
                            // Show what the user typed (digits only), but the
                            // committed store value stays clamped to [1, 100].
                            setSubsetDraft(raw);
                            const n = Math.min(100, Math.max(1, parseInt(raw, 10)));
                            updateField("train_subset_percent", n);
                          }}
                          onBlur={() => {
                            if (subsetDraft === "" || subsetDraft === null) {
                              const fallback = activeImpulse.train_subset_percent ?? 100;
                              updateField("train_subset_percent", fallback);
                            }
                            // Drop the draft so display falls back to the
                            // clamped committed value (e.g. typed "150" → "100").
                            setSubsetDraft(null);
                          }}
                        />
                        <span className="text-xs text-gray-500 select-none">%</span>
                      </div>
                    </div>
                  </div>
                </div>

                <Arrow />

                {/* Processing block stack — DSP blocks + Add placeholder, stacked vertically */}
                <div className="flex flex-col items-stretch gap-5 w-[clamp(17rem,22vw,20rem)] flex-shrink-0">
                  {dspBlocks.map((block: any, idx: number) => {
                    const isImage = block?.type === "image";
                    const badge = PROCESSING_BLOCK_BADGE[block.type] || DEFAULT_PROCESSING_BADGE;
                    const BadgeIcon = badge.icon;
                    return (<div key={idx} data-block-type={block.type} className={`impulse-processing-card relative w-full min-h-[26rem] flex flex-col border rounded-xl overflow-hidden transition-all ${BLOCK_ACCENT[block.type] || "border-gray-600/50 bg-gray-800/40"
                      }`}>
                      <button
                        type="button"
                        aria-label="Delete processing block"
                        onClick={() => removeDSPBlock(idx)}
                        className="absolute bottom-2 right-2 z-10 p-1.5 rounded text-gray-500 hover:text-red-400 hover:bg-red-500/10 transition-colors"
                      >
                        <Trash2 size={13} />
                      </button>
                      <div className="px-5 py-4 flex items-center justify-between border-b border-white/5">
                        <div className="flex-1 min-w-0 pr-3">
                          <p className="text-[9px] text-gray-400 uppercase tracking-wider mb-0.5">Processing</p>
                          <p className="text-base font-semibold text-gray-100">{block.name || (block?.type ?? "").replace(/_/g, " ")}</p>
                        </div>
                        <div className={`h-10 w-10 rounded-full flex items-center justify-center shadow-md flex-shrink-0 ${badge.bg}`}>
                          <BadgeIcon size={20} className={badge.fg} strokeWidth={2.5} />
                        </div>
                      </div>
                      <div className="flex flex-1 flex-col justify-center p-4 space-y-2">
                        {isImage ? (
                          /* Image block — dims + color mode, centered. The
                             negative margin compensates for the card header
                             so the text aligns vertically with the Arrow
                             connector (which is centered in the 21rem block,
                             not in just the content area below the header). */
                          <div className="flex flex-col items-center justify-center text-center -mt-10">
                            <p className="text-base font-semibold text-orange-300">
                              {activeImpulse?.image_width ?? "?"} × {activeImpulse?.image_height ?? "?"}
                            </p>
                            <p className="text-[10px] text-gray-500 mt-1">
                              {(block.params?.grayscale ? "Grayscale" : "RGB")}
                            </p>
                          </div>
                        ) : (
                          /* Other block types — read-only param preview */
                          Object.entries(block.params || {}).slice(0, 3).map(([key, val]) => (
                            <div key={key}>
                              <p className="text-[9px] text-gray-500 capitalize">{(typeof key === "string" ? key : "").replace(/_/g, " ")}</p>
                              <input type="number" readOnly
                                className="impulse-inline-readonly w-full rounded px-2 py-1 text-[11px]"
                                value={String(val)} />
                            </div>
                          ))
                        )}
                      </div>
                    </div>
                    );
                  })}

                  {/* Add processing block placeholder — only when no processing block exists (one per impulse) */}
                  {dspBlocks.length === 0 && (
                    <button
                      onClick={() => setShowDSPModal(true)}
                      className="impulse-add-card w-full min-h-[26rem] border-2 border-dashed border-gray-600 rounded-xl flex flex-col items-center justify-center gap-2 px-4 py-5 hover:border-indigo-500/60 hover:bg-indigo-900/10 transition-all group"
                    >
                      <div className="impulse-add-card-icon w-11 h-11 rounded-full border-2 border-dashed border-gray-600 group-hover:border-indigo-400 flex items-center justify-center transition-colors">
                        <Zap size={15} className="text-gray-500 group-hover:text-indigo-400 transition-colors" />
                      </div>
                      <span className="text-sm text-gray-500 group-hover:text-gray-300 transition-colors text-center px-4 leading-relaxed">
                        Add a processing block
                      </span>
                    </button>
                  )}
                </div>

                <Arrow />

                {/* Learning lane — ML blocks + Add placeholder, stacked vertically */}
                <div className="flex flex-col items-stretch gap-5 w-[clamp(17rem,22vw,20rem)] flex-shrink-0">
                  {mlBlocks.map((block: any, idx: number) => {
                    const isOD = isObjectDetectionLearningBlock(block);
                    const LearningBadgeIcon = LEARNING_BLOCK_BADGE.icon;
                    return (
                      <div key={idx} className="impulse-learning-card relative w-full min-h-[26rem] border border-indigo-500/50 bg-indigo-900/20 rounded-xl overflow-hidden flex flex-col">
                        <button
                          type="button"
                          aria-label="Delete learning block"
                          onClick={() => removeMLBlock(idx)}
                          className="absolute bottom-2 right-2 z-10 p-1.5 rounded text-gray-500 hover:text-red-400 hover:bg-red-500/10 transition-colors"
                        >
                          <Trash2 size={13} />
                        </button>
                        <div className="px-5 py-4 flex items-center justify-between border-b border-indigo-400/10">
                          <div className="flex-1 min-w-0 pr-3">
                            <p className="text-[9px] text-indigo-400 uppercase tracking-wider mb-0.5">Learning</p>
                            <p className="text-base font-semibold text-gray-100">{block.name || (block?.type ?? "").replace(/_/g, " ")}</p>
                          </div>
                          <div className={`h-10 w-10 rounded-full flex items-center justify-center shadow-md flex-shrink-0 ${LEARNING_BLOCK_BADGE.bg}`}>
                            <LearningBadgeIcon size={20} className={LEARNING_BLOCK_BADGE.fg} strokeWidth={2.5} />
                          </div>
                        </div>
                        {/* Body — two zones: Change block (center) + Output features (bottom).
                          Header above is the top zone. flex-1 + items-center anchors the
                          button to the card's vertical middle; the section row sits at the
                          bottom naturally as the last flow child. */}
                        <div className="flex flex-col p-4 flex-1">
                          <div className="flex-1 flex items-center justify-center w-full">
                            <button onClick={() => setShowMLModal(true)}
                              className="impulse-learning-action w-full text-sm border rounded-lg py-2 transition-colors">
                              Change block
                            </button>
                          </div>
                          {isOD && (
                            <div className="impulse-learning-section mt-1 mb-6">
                              <p className="impulse-learning-section-label text-[10px] uppercase tracking-wider text-indigo-300/80 mb-1">Output features</p>
                              {outputLabelsLoading && !outputLabelsSummary ? (
                                <span
                                  aria-hidden="true"
                                  className="inline-block h-4 w-32 rounded bg-gray-700/60 animate-pulse"
                                />
                              ) : (
                                <p className="impulse-learning-section-value text-sm text-gray-100">
                                  {outputLabelsSummary ?? "0 (no labels assigned)"}
                                </p>
                              )}
                            </div>
                          )}
                        </div>
                      </div>
                    );
                  })}

                  {/* Add learning block placeholder — only when no learning block exists (one per impulse) */}
                  {mlBlocks.length === 0 && (
                    <button
                      onClick={() => setShowMLModal(true)}
                      className="impulse-add-card w-full min-h-[26rem] border-2 border-dashed border-gray-600 rounded-xl flex flex-col items-center justify-center gap-2 px-4 py-5 hover:border-indigo-500/60 hover:bg-indigo-900/10 transition-all group"
                    >
                      <div className="impulse-add-card-icon w-11 h-11 rounded-full border-2 border-dashed border-gray-600 group-hover:border-indigo-400 flex items-center justify-center transition-colors">
                        <FlaskConical size={15} className="text-gray-500 group-hover:text-indigo-400 transition-colors" />
                      </div>
                      <span className="text-sm text-gray-500 group-hover:text-gray-300 transition-colors text-center px-4 leading-relaxed">
                        Add a learning block
                      </span>
                    </button>
                  )}
                </div>

                <Arrow />

                {/* Output features */}
                <div className={`impulse-output-card w-[clamp(17rem,22vw,20rem)] min-h-[21rem] border rounded-xl flex-shrink-0 flex flex-col items-center justify-center px-4 ${outputReady ? "border-emerald-600/50 bg-emerald-900/10" : "border-gray-700/50 bg-gray-800/20"
                  }`}>
                  <div className={`w-12 h-12 rounded-full flex items-center justify-center mb-3 ${outputReady ? "bg-emerald-500" : "bg-gray-700"}`}>
                    {outputReady ? <CheckCircle size={20} className="text-white" /> : <Settings size={15} className="text-gray-400" />}
                  </div>
                  <p className={`text-sm font-semibold text-center ${outputReady ? "text-emerald-300" : "text-gray-500"}`}>
                    Output features
                  </p>
                  {outputReady && (
                    outputLabelsLoading && !outputLabelsSummary ? (
                      <span
                        aria-hidden="true"
                        className="mt-1 inline-block h-4 w-32 rounded bg-gray-700/60 animate-pulse"
                      />
                    ) : (
                      <p className="impulse-output-card-summary text-[12px] text-center text-gray-100 mt-1 px-2">
                        {outputLabelsSummary ?? "0 (no labels assigned)"}
                      </p>
                    )
                  )}
                </div>

              </div>
            </div>
          </div>

          {/* Validation errors */}
          {validationErrors.length > 0 && (
            <div className="mt-4 p-3 bg-red-900/20 border border-red-800/50 rounded-xl space-y-1">
              {validationErrors.map((e, i) => (
                <div key={i} className="flex items-center gap-2 text-sm text-red-400">
                  <AlertCircle size={13} className="flex-shrink-0" /> {e}
                </div>
              ))}
            </div>
          )}

        </>
      )}

      {/* Modals */}
      {showDSPModal && activeImpulse && (
        <ProcessingBlockModal
          inputType={activeImpulse.input_type || ""}
          sensorType={activeImpulse.sensor_type || ""}
          projectType={activeProject?.project_type}
          onAdd={addDSPBlock}
          onClose={() => setShowDSPModal(false)}
        />
      )}
      {showMLModal && activeImpulse && (
        <LearningBlockModal
          impulseId={activeImpulse.id}
          projectType={activeProject?.project_type}
          onAdd={addMLBlock}
          onClose={() => setShowMLModal(false)}
        />
      )}
    </div>
  );
}
