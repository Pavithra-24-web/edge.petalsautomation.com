"use client";
import { useEffect, useMemo, useState } from "react";
import { createPortal } from "react-dom";
import { useRouter } from "next/navigation";
import { useAppStore } from "@/store/appStore";
import { impulsesApi, trainingApi } from "@/utils/api";
import {
  Trash2, FlaskConical,
  Square, CheckSquare, RefreshCw, LayoutList, MoreVertical,
  Eye, Pencil, RotateCw,
} from "lucide-react";
import toast from "react-hot-toast";
import {
  EDGE_DETECT_LITE,
  NANO_VISION_MNV2_035,
  NANO_VISION_V1,
  VISION_PRO,
} from "@/lib/model-display-names";

/* ─────────────────────────────────────────────────────────────────────────────
   Helpers
───────────────────────────────────────────────────────────────────────────── */
function dspBlockLabel(type: string): string {
  const map: Record<string, string> = {
    image: "Image",
    spectral_analysis: "Spectral Analysis",
    mfcc: "MFCC",
    spectrogram: "Spectrogram",
    raw: "Raw Data",
    flatten: "Flatten",
    eeg: "EEG",
  };
  return map[type] ?? type.replace(/_/g, " ");
}

/**
 * Display name for an architecture string returned by
 * `trainingApi.impulseStatus → latest_model.architecture`. These are the raw
 * keys the backend stamps on the trained TFLite model_metadata; we map them
 * to the user-facing names from the model picker.
 *
 * Returns `null` if the input is empty/unknown so callers can render "—".
 */
function trainedArchitectureName(arch: string | null | undefined): string | null {
  if (!arch) return null;
  const a = String(arch).toLowerCase().trim();
  if (!a) return null;
  // FOMO family
  if (a === "fomo_mobilenetv2_0_1") return NANO_VISION_MNV2_035;
  if (a === "fomo_v1" || a === "fomo") return NANO_VISION_V1;
  if (a.startsWith("fomo")) return NANO_VISION_MNV2_035;
  // YOLO family (yolo_pro, yolo_pro_s, yolo_pro_v2, yolo_pro_detection, …)
  if (a.startsWith("yolo_pro") || a === "yolo-pro") return VISION_PRO;
  // MobileNetV2 SSD FPN-Lite (a few stamped variants in the wild)
  if (
    a === "mobilenetv2_ssd_fpnlite_320x320" ||
    a === "mobilenet_v2_ssd_fpn_lite" ||
    a === "ssd_detection" ||
    a === "object_detection"
  ) {
    return EDGE_DETECT_LITE;
  }
  if (a === "mobilenet_v2" || a === "mobilenet") return "MobileNetV2";
  // Classification / regression / anomaly heads
  if (a === "dense" || a === "conv1d" || a === "classification") return "Classification";
  if (a === "regression") return "Regression";
  if (a === "anomaly_detection" || a === "anomaly_gmm") return "Anomaly Detection";
  // Fallback: humanise unknown architecture strings rather than dropping them.
  return arch.replace(/_/g, " ");
}

/**
 * Architecture / model name for a learn block — used as a fallback when the
 * API hasn't returned a `latest_model.architecture` yet (e.g. response in
 * flight). Returns `null` if we can't confidently identify a concrete model;
 * callers should render an em dash.
 */
function learnBlockArchitecture(block: any): string | null {
  const rawType = String(block?.type ?? block?.architecture ?? "").toLowerCase();
  const params = (block?.params ?? {}) as Record<string, any>;

  // Generic "object_detection" block — the real model MUST be in params.model.
  // If it isn't, we can't claim a specific architecture; show "—" instead of
  // guessing (the previous "default to fomo" behaviour labelled every
  // never-configured impulse as the v1 centroid model).
  if (rawType === "object_detection") {
    const modelRaw = params.model;
    if (modelRaw == null || String(modelRaw).trim() === "") return null;
    const model = String(modelRaw).toLowerCase();
    if (model === "fomo" || model === "fomo_mobilenetv2_0_1") {
      const version = Number(params.fomo_version);
      if (Number.isFinite(version) && version >= 2) return NANO_VISION_MNV2_035;
      if (Number.isFinite(version) && version === 1) return NANO_VISION_V1;
      return null; // model is fomo but version unknown — don't fabricate
    }
    if (model === "yolo_pro" || model === "yolo-pro") return VISION_PRO;
    if (model === "mobilenet_v2_ssd_fpn_lite") return EDGE_DETECT_LITE;
    return model.replace(/_/g, " ");
  }

  // Specific learn block types — the type IS the architecture.
  if (rawType === "fomo_mobilenetv2_0_1") {
    const version = Number(params.fomo_version);
    if (Number.isFinite(version) && version >= 2) return NANO_VISION_MNV2_035;
    if (Number.isFinite(version) && version === 1) return NANO_VISION_V1;
    return NANO_VISION_MNV2_035; // type carries enough info
  }
  const map: Record<string, string> = {
    mobilenet_v2_ssd_fpn_lite: EDGE_DETECT_LITE,
    "fomo_v2_0.35": NANO_VISION_MNV2_035,
    fomo_v2_035: NANO_VISION_MNV2_035,
    fomo_v1: NANO_VISION_V1,
    yolo_pro: VISION_PRO,
    classification: "Classification",
    regression: "Regression",
    anomaly_detection: "Anomaly Detection",
  };
  if (rawType && map[rawType]) return map[rawType];

  const name = typeof block?.name === "string" ? block.name.trim() : "";
  if (name) return name;
  return null;
}

function DeleteImpulseModal({
  count,
  deleting,
  onClose,
  onConfirm,
}: {
  count: number;
  deleting: boolean;
  onClose: () => void;
  onConfirm: () => void;
}) {
  const plural = count === 1 ? "impulse" : "impulses";

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/55 px-4 backdrop-blur-[2px]">
      <div className="w-full max-w-[512px] rounded-xl bg-white shadow-[0_24px_80px_rgba(15,23,42,0.35)]">
        <div className="flex flex-col items-center px-8 pb-6 pt-10 text-center">
          <div className="mb-8 flex h-24 w-24 items-center justify-center rounded-full border-4 border-rose-500 text-5xl font-light leading-none text-rose-500">
            ?
          </div>

          <h2 className="text-[20px] font-medium text-slate-700">
            Delete {count === 1 ? "impulse" : "multiple impulses"}
          </h2>

          <p className="mt-4 max-w-[360px] text-[15px] leading-6 text-slate-500">
            Are you sure you want to delete the selected {count} {plural}? This will remove the
            data for all blocks in {count === 1 ? "this impulse." : "these impulses."}
          </p>

          <div className="mt-8 flex items-center gap-3">
            <button
              onClick={onClose}
              disabled={deleting}
              className="rounded-lg bg-slate-100 px-7 py-3 text-[15px] font-semibold text-slate-700 transition hover:bg-slate-200 disabled:cursor-not-allowed disabled:opacity-60"
            >
              Cancel
            </button>
            <button
              onClick={onConfirm}
              disabled={deleting}
              className="rounded-lg bg-rose-500 px-7 py-3 text-[15px] font-semibold text-white transition hover:bg-rose-600 disabled:cursor-wait disabled:opacity-70"
            >
              {deleting ? "Deleting..." : "Yes, delete"}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

/* ─────────────────────────────────────────────────────────────────────────────
   Row column grid — keep header + every row in sync via one template.
   Columns: checkbox / name+dot / input / dsp / learn / kebab
───────────────────────────────────────────────────────────────────────────── */
const ROW_GRID =
  "grid items-center gap-4 px-5 " +
  "grid-cols-[32px_minmax(180px,1.6fr)_minmax(100px,0.8fr)_minmax(140px,1fr)_minmax(220px,1.6fr)_44px]";

/* ─────────────────────────────────────────────────────────────────────────────
   Single impulse row
───────────────────────────────────────────────────────────────────────────── */
function ImpulseRow({
  impulse,
  isActive,
  hasTrainedModel,
  trainedArchitecture,
  onOpen,
  isSelected,
  onToggle,
  isMenuOpen,
  onRequestMenu,
  onCloseMenu,
}: {
  impulse: any;
  isActive: boolean;
  hasTrainedModel: boolean | null;
  /** Raw architecture string from `latest_model.architecture`, e.g.
   *  "fomo_mobilenetv2_0_1", "yolo_pro_s". Null until status fetched. */
  trainedArchitecture: string | null;
  onOpen: () => void;
  isSelected?: boolean;
  onToggle?: () => void;
  isMenuOpen: boolean;
  onRequestMenu: (anchor: HTMLElement) => void;
  onCloseMenu: () => void;
}) {
  const dspBlocks: any[] = impulse.dsp_blocks ?? [];
  const mlBlocks: any[] = impulse.ml_blocks ?? [];

  // Input cell: an impulse without any DSP block has no concrete input
  // configuration to display yet — show an em-dash placeholder so brand-new
  // empty impulses read as "not configured" instead of "96 × 96 default".
  const hasDsp = dspBlocks.length > 0;
  const inputDisplay = hasDsp
    ? impulse.input_type === "image"
      ? `${impulse.image_width ?? "?"} × ${impulse.image_height ?? "?"}`
      : "TIME-SERIES"
    : null;

  // Status dot tone — drives both fill + soft glow ring.
  const dotTone =
    hasTrainedModel === true ? "trained" : hasTrainedModel === false ? "untrained" : "unknown";
  const dotTitle =
    hasTrainedModel === true
      ? "Model trained"
      : hasTrainedModel === false
        ? "Not trained yet"
        : "Checking training status…";

  return (
    <div
      className={`manage-impulse-row ${ROW_GRID} h-14 transition-colors ${
        isActive ? "is-active" : ""
      }`}
    >
      {/* Selection checkbox */}
      <button
        onClick={(e) => { e.stopPropagation(); onToggle?.(); }}
        className="manage-impulse-check flex-shrink-0 p-1 rounded transition-colors"
        aria-label={isSelected ? "Deselect impulse" : "Select impulse"}
      >
        {isSelected ? (
          <CheckSquare size={16} className="text-indigo-500" />
        ) : (
          <Square size={16} />
        )}
      </button>

      {/* Name + status dot + ACTIVE badge */}
      <div className="flex items-center gap-2.5 min-w-0">
        <span
          className={`manage-impulse-dot manage-impulse-dot--${dotTone} flex-shrink-0`}
          title={dotTitle}
          aria-label={dotTitle}
        />
        <button
          onClick={onOpen}
          className={`manage-impulse-name text-[13.5px] font-semibold tracking-tight hover:underline truncate ${
            isActive ? "is-active-name" : ""
          }`}
          title={impulse.name}
        >
          {impulse.name}
        </button>
        <span
          className="text-[10px] font-mono text-gray-500 truncate flex-shrink-0"
          title={impulse.id}
        >
          {impulse.id ? String(impulse.id).slice(0, 8) : "—"}
        </span>
        {isActive && (
          <span className="manage-impulse-badge text-[9px] font-bold uppercase tracking-widest px-2 py-[3px] rounded-full flex-shrink-0">
            ACTIVE
          </span>
        )}
      </div>

      {/* Input — image: "96 × 96", time-series: "TIME-SERIES", em-dash when
          the impulse has no processing block yet (empty placeholder). */}
      <div className="manage-impulse-input-cell text-[12.5px] font-medium truncate">
        {inputDisplay ?? <span className="manage-impulse-empty text-[12.5px]">—</span>}
      </div>

      {/* DSP Blocks */}
      <div className="flex flex-col gap-0.5 min-w-0">
        {dspBlocks.length === 0 ? (
          <span className="manage-impulse-empty text-[12.5px]">—</span>
        ) : (
          dspBlocks.map((b: any, i: number) => (
            <span
              key={i}
              className="manage-impulse-dsp-text text-[12.5px] font-medium truncate"
              title={dspBlockLabel(b.type)}
            >
              {dspBlockLabel(b.type)}
            </span>
          ))
        )}
      </div>

      {/* Learn Blocks — the exact model the impulse was last trained with
          (from `latest_model.architecture`). Falls back to a best-effort name
          parsed from the block config; "—" only when truly unknown. */}
      <div className="flex flex-col gap-0.5 min-w-0">
        {(() => {
          const trainedName = trainedArchitectureName(trainedArchitecture);
          if (trainedName) {
            return (
              <div className="flex items-center gap-1.5 min-w-0">
                <FlaskConical size={11} className="manage-impulse-learn-icon flex-shrink-0" />
                <span
                  className="manage-impulse-learn-text text-[12.5px] font-medium truncate"
                  title={trainedName}
                >
                  {trainedName}
                </span>
              </div>
            );
          }

          // No trained architecture available yet — fall back to whatever the
          // block config implies. Render "—" when neither is conclusive.
          if (mlBlocks.length === 0) {
            return <span className="manage-impulse-empty text-[12.5px]">—</span>;
          }
          return mlBlocks.map((b: any, i: number) => {
            const fromBlock = learnBlockArchitecture(b);
            if (!fromBlock) {
              return (
                <span key={i} className="manage-impulse-empty text-[12.5px]">—</span>
              );
            }
            return (
              <div key={i} className="flex items-center gap-1.5 min-w-0">
                <FlaskConical size={11} className="manage-impulse-learn-icon flex-shrink-0" />
                <span
                  className="manage-impulse-learn-text text-[12.5px] font-medium truncate"
                  title={fromBlock}
                >
                  {fromBlock}
                </span>
              </div>
            );
          });
        })()}
      </div>

      {/* 3-dot menu (ghost icon button) */}
      <div className="flex justify-end">
        <button
          onClick={(e) => {
            e.stopPropagation();
            if (isMenuOpen) onCloseMenu();
            else onRequestMenu(e.currentTarget);
          }}
          className={`manage-impulse-kebab ${isMenuOpen ? "is-open" : ""}`}
          aria-label="Row actions"
          title="More actions"
        >
          <MoreVertical size={16} />
        </button>
      </div>
    </div>
  );
}

/* ─────────────────────────────────────────────────────────────────────────────
   Kebab menu (portal-rendered so it escapes any clipped ancestor)
───────────────────────────────────────────────────────────────────────────── */
function RowMenu({
  pos,
  trained,
  onClose,
  onView,
  onRename,
  onRetrain,
  onTest,
  onDelete,
}: {
  pos: { top: number; left: number };
  trained: boolean;
  onClose: () => void;
  onView: () => void;
  onRename: () => void;
  onRetrain: () => void;
  onTest: () => void;
  onDelete: () => void;
}) {
  useEffect(() => {
    function onDoc(e: MouseEvent) {
      const target = e.target as HTMLElement;
      if (!target.closest("[data-row-menu]")) onClose();
    }
    document.addEventListener("mousedown", onDoc);
    return () => document.removeEventListener("mousedown", onDoc);
  }, [onClose]);

  return createPortal(
    <div
      data-row-menu
      className="manage-impulse-menu"
      style={{ top: pos.top, left: pos.left }}
      onClick={(e) => e.stopPropagation()}
      role="menu"
    >
      <button
        className="manage-impulse-menu-item"
        onClick={() => { onClose(); onView(); }}
        role="menuitem"
      >
        <Eye size={13} /> View impulse
      </button>
      <button
        className="manage-impulse-menu-item"
        onClick={() => { onClose(); onRename(); }}
        role="menuitem"
      >
        <Pencil size={13} /> Rename
      </button>
      <button
        className="manage-impulse-menu-item"
        disabled={!trained}
        title={trained ? undefined : "Train the model first to enable retrain"}
        onClick={() => { if (!trained) return; onClose(); onRetrain(); }}
        role="menuitem"
      >
        <RotateCw size={13} /> Retrain
      </button>
      <button
        className="manage-impulse-menu-item"
        disabled={!trained}
        title={trained ? undefined : "Train the model first to enable testing"}
        onClick={() => { if (!trained) return; onClose(); onTest(); }}
        role="menuitem"
      >
        <FlaskConical size={13} /> Test
      </button>
      <div className="manage-impulse-menu-divider" />
      <button
        className="manage-impulse-menu-item is-destructive"
        onClick={() => { onClose(); onDelete(); }}
        role="menuitem"
      >
        <Trash2 size={13} /> Delete impulse
      </button>
    </div>,
    document.body,
  );
}

/* ─────────────────────────────────────────────────────────────────────────────
   Page
───────────────────────────────────────────────────────────────────────────── */
export default function ManageImpulsePage() {
  const router = useRouter();
  const { activeProject, activeImpulse, setActiveImpulse, setSavedActiveImpulse } = useAppStore();

  const [impulses, setImpulses] = useState<any[]>([]);

  // Display order: newest first. Backend returns `created_at` as ISO string
  // (see impulses.py `_imp`); we fall back to id-compare when timestamps
  // collide or are missing so the order stays stable.
  const sortedImpulses = useMemo(() => {
    const ts = (imp: any) => {
      const t = Date.parse(imp?.created_at ?? "");
      return Number.isFinite(t) ? t : 0;
    };
    return [...impulses].sort((a, b) => {
      const diff = ts(b) - ts(a);
      if (diff !== 0) return diff;
      return String(b?.id ?? "").localeCompare(String(a?.id ?? ""));
    });
  }, [impulses]);

  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selectedIds, setSelectedIds] = useState<Set<string>>(new Set());
  const [deletingBulk, setDeletingBulk] = useState(false);
  const [deletingSingleId, setDeletingSingleId] = useState<string | null>(null);
  const [deleteModal, setDeleteModal] = useState<
    | { mode: "single"; impulse: any; count: 1 }
    | { mode: "bulk"; count: number }
    | null
  >(null);

  // Per-impulse training status, mirrors the gate the retrain/warning page
  // uses (`has_trained_model` from `trainingApi.impulseStatus`). We also
  // capture `latest_model.architecture` so the Learn Blocks column can show
  // the exact model that was trained (instead of guessing from the block).
  // `null` until fetched — Retrain/Test stay disabled while unknown.
  type ImpulseStatus = { trained: boolean; architecture: string | null };
  const [statusMap, setStatusMap] = useState<Record<string, ImpulseStatus | null>>({});

  // Row kebab menu state
  const [menuOpenId, setMenuOpenId] = useState<string | null>(null);
  const [menuPos, setMenuPos] = useState<{ top: number; left: number } | null>(null);

  const deletingFromModal =
    deleteModal?.mode === "bulk"
      ? deletingBulk
      : deleteModal?.mode === "single" && deletingSingleId === deleteModal.impulse.id;

  useEffect(() => {
    if (activeProject) load();
  }, [activeProject]);

  async function load() {
    if (!activeProject) return;
    setLoading(true);
    setError(null);
    try {
      const { data } = await impulsesApi.list(activeProject.id);
      setImpulses(data);
      if (data.length === 0) {
        setActiveImpulse(null);
        setSavedActiveImpulse(null);
        setSelectedIds(new Set());
      } else if (activeImpulse && !data.some((i: any) => i.id === activeImpulse.id)) {
        setActiveImpulse(null);
        setSavedActiveImpulse(null);
      }
      // Fetch trained-model status in parallel. Same endpoint the retrain
      // page uses for its readiness gate — keeps Retrain/Test logic
      // consistent — and also returns `latest_model.architecture` so the
      // Learn Blocks column can show the exact trained model name.
      const results = await Promise.allSettled(
        (data as any[]).map((i: any) => trainingApi.impulseStatus(i.id)),
      );
      const next: Record<string, ImpulseStatus | null> = {};
      (data as any[]).forEach((i: any, idx: number) => {
        const r = results[idx];
        if (r.status === "fulfilled") {
          const payload = (r.value as any).data ?? {};
          next[i.id] = {
            trained: payload.has_trained_model === true,
            architecture: payload.latest_model?.architecture ?? null,
          };
        } else {
          next[i.id] = { trained: false, architecture: null };
        }
      });
      setStatusMap(next);
    } catch {
      setError("Failed to load impulses.");
      toast.error("Failed to load impulses");
    } finally {
      setLoading(false);
    }
  }

  async function renameImpulse(impulse: any) {
    // Native prompt — matches the lightweight rename pattern used elsewhere
    // (e.g. dataset sample rename) without a new modal component.
    const next = window.prompt("Rename impulse", impulse.name)?.trim();
    if (!next || next === impulse.name) return;
    try {
      const { data } = await impulsesApi.patch(impulse.id, { name: next });
      setImpulses((prev) => prev.map((i) => (i.id === impulse.id ? { ...i, name: data?.name ?? next } : i)));
      if (activeImpulse?.id === impulse.id) {
        setActiveImpulse({ ...activeImpulse, name: data?.name ?? next });
      }
      window.dispatchEvent(new Event("impulses:changed"));
      toast.success("Impulse renamed");
    } catch (e: any) {
      toast.error(e?.response?.data?.detail || "Failed to rename impulse");
    }
  }

  function openRowMenu(impulseId: string, anchor: HTMLElement) {
    const r = anchor.getBoundingClientRect();
    const MENU_W = 224; // matches w-56
    const MENU_H = 260;
    const margin = 8;
    let left = r.right - MENU_W;
    if (left < margin) left = margin;
    if (left + MENU_W > window.innerWidth - margin) {
      left = window.innerWidth - MENU_W - margin;
    }
    let top = r.bottom + 4;
    if (top + MENU_H > window.innerHeight - margin) {
      top = Math.max(margin, r.top - MENU_H - 4);
    }
    setMenuPos({ top, left });
    setMenuOpenId(impulseId);
  }
  function closeRowMenu() {
    setMenuOpenId(null);
    setMenuPos(null);
  }

  async function openEditor(impulse: any) {
    try {
      // Always fetch fresh so we have the canonical server copy
      const { data } = await impulsesApi.get(impulse.id);
      setActiveImpulse(data);
      setSavedActiveImpulse(data);
      router.push(`/dashboard/impulse?impulseId=${data.id}`);
    } catch {
      toast.error("Could not load impulse — please try again.");
    }
  }

  async function deleteImpulse(impulse: any) {
    setDeleteModal({ mode: "single", impulse, count: 1 });
  }

  async function confirmSingleDelete(impulse: any) {
    setDeletingSingleId(impulse.id);
    try {
      await impulsesApi.delete(impulse.id);
      await load();
      if (activeImpulse?.id === impulse.id) {
        setActiveImpulse(null);
        setSavedActiveImpulse(null);
      }
      
      const next = new Set(selectedIds);
      next.delete(impulse.id);
      setSelectedIds(next);

      window.dispatchEvent(new Event("impulses:changed"));
      
      setDeleteModal(null);
      toast.success("Impulse deleted");
    } catch (error: any) {
      toast.error(error?.response?.data?.detail || "Failed to delete impulse");
    } finally {
      setDeletingSingleId(null);
    }
  }

  function toggleSelect(id: string) {
    const next = new Set(selectedIds);
    if (next.has(id)) next.delete(id);
    else next.add(id);
    setSelectedIds(next);
  }

  function toggleAll() {
    if (selectedIds.size === impulses.length) {
      setSelectedIds(new Set());
    } else {
      setSelectedIds(new Set(impulses.map(i => i.id)));
    }
  }

  async function bulkDelete() {
    const count = selectedIds.size;
    if (count === 0) return;
    setDeleteModal({ mode: "bulk", count });
  }

  async function confirmBulkDelete() {
    const count = selectedIds.size;
    if (count === 0) {
      setDeleteModal(null);
      return;
    }
    setDeletingBulk(true);
    try {
      const ids = Array.from(selectedIds);
      const { data } = await impulsesApi.bulkDelete(ids);
      const failed = new Set<string>(data?.failed || []);
      const deletedIds = ids.filter(id => !failed.has(id));

      await load();

      if (activeImpulse && deletedIds.includes(activeImpulse.id)) {
        setActiveImpulse(null);
        setSavedActiveImpulse(null);
      }
      window.dispatchEvent(new Event("impulses:changed"));
      setDeleteModal(null);

      if (failed.size === 0) {
        setSelectedIds(new Set());
        toast.success(`Deleted ${deletedIds.length} impulses`);
      } else if (deletedIds.length > 0) {
        setSelectedIds(failed);
        toast.error(`${failed.size} impulse(s) could not be deleted`);
      } else {
        setSelectedIds(failed);
        toast.error("No selected impulses were deleted");
      }
    } catch (error: any) {
      toast.error(error?.response?.data?.detail || "Failed to delete some impulses");
    } finally {
      setDeletingBulk(false);
    }
  }

  return (
    <div className="min-h-screen manage-impulses-page">

      {/* Page header */}
      <div className="mb-6 flex items-start justify-between">
        <div>
          <div className="flex items-center gap-2 mb-0.5">
            <LayoutList size={18} className="text-indigo-400" />
            <h1 className="manage-impulses-title text-xl font-bold tracking-tight">Manage Impulses</h1>
          </div>
          <p className="manage-impulses-subtitle text-sm">
            Select an impulse to open its pipeline, training, and deployment settings.
          </p>
        </div>
        <button
          onClick={() => router.push("/dashboard/impulse")}
          className="manage-impulses-back flex items-center gap-1.5 px-3 py-2 text-sm rounded-lg transition-colors flex-shrink-0"
        >
          ← Back to Editor
        </button>
      </div>

      {/* Table card — overflow-clip preserves rounded corners without
          breaking the sticky column header. */}
      <div className="manage-impulses-card rounded-xl overflow-clip">

        {/* Top bar */}
        <div className="manage-impulses-toolbar flex items-center justify-between px-5 py-3">
          <div className="flex items-center gap-4">
            <button
              onClick={toggleAll}
              className="manage-impulses-select-all flex items-center gap-2 text-sm transition-colors"
            >
              {selectedIds.size > 0 && selectedIds.size === impulses.length ? (
                <CheckSquare size={14} className="text-indigo-500" />
              ) : (
                <Square size={14} />
              )}
              Select All
            </button>

            {selectedIds.size > 0 && (
              <div className="manage-impulses-bulk flex items-center gap-3 pl-4">
                <span className="manage-impulses-selected-count text-xs font-medium">
                  {selectedIds.size} selected
                </span>
                <button
                  onClick={bulkDelete}
                  disabled={deletingBulk}
                  className="manage-impulses-delete flex items-center gap-1.5 px-2 py-1 text-[11px] font-bold uppercase tracking-wider rounded transition-colors"
                >
                  <Trash2 size={12} />
                  Delete Selected
                </button>
              </div>
            )}
          </div>
          <span className="manage-impulses-total text-sm">
            TOTAL:{" "}
            <span className="manage-impulses-total-num font-semibold ml-1">{impulses.length}</span>
          </span>
        </div>

        {/* Loading */}
        {loading && (
          <div className="py-24 text-center text-sm text-gray-500">
            <RefreshCw size={18} className="animate-spin mx-auto mb-3 text-gray-600" />
            Loading impulses…
          </div>
        )}

        {/* Error */}
        {!loading && error && (
          <div className="py-24 text-center text-sm text-red-400">
            {error}{" "}
            <button onClick={load} className="underline hover:no-underline ml-1">
              Retry
            </button>
          </div>
        )}

        {/* Empty */}
        {!loading && !error && impulses.length === 0 && (
          <div className="py-24 text-center text-sm text-gray-500">
            No impulses yet.{" "}
            <button
              onClick={() => router.push("/dashboard/impulse")}
              className="text-indigo-400 hover:underline"
            >
              Create one
            </button>
          </div>
        )}

        {/* Column header — sticky so it stays in view while scrolling */}
        {!loading && !error && impulses.length > 0 && (
          <div
            className={`manage-impulses-header-row ${ROW_GRID} py-2.5 text-[10.5px] font-semibold uppercase tracking-[0.08em]`}
          >
            <span />
            <span>Name</span>
            <span>Input</span>
            <span>DSP Blocks</span>
            <span>Learn Blocks</span>
            <span />
          </div>
        )}

        {/* Rows */}
        {!loading &&
          !error &&
          sortedImpulses.map(imp => (
            <ImpulseRow
              key={imp.id}
              impulse={imp}
              isActive={activeImpulse?.id === imp.id}
              hasTrainedModel={statusMap[imp.id] ? statusMap[imp.id]!.trained : null}
              trainedArchitecture={statusMap[imp.id]?.architecture ?? null}
              onOpen={() => openEditor(imp)}
              isSelected={selectedIds.has(imp.id)}
              onToggle={() => toggleSelect(imp.id)}
              isMenuOpen={menuOpenId === imp.id}
              onRequestMenu={(anchor) => openRowMenu(imp.id, anchor)}
              onCloseMenu={closeRowMenu}
            />
          ))}

        {/* Portal-rendered kebab menu for whichever row is active */}
        {menuOpenId && menuPos && (() => {
          const imp = impulses.find((i) => i.id === menuOpenId);
          if (!imp) return null;
          return (
            <RowMenu
              pos={menuPos}
              trained={statusMap[imp.id]?.trained === true}
              onClose={closeRowMenu}
              onView={() => openEditor(imp)}
              onRename={() => renameImpulse(imp)}
              onRetrain={() => router.push(`/dashboard/impulse/retrain?impulseId=${imp.id}`)}
              onTest={() => router.push(`/dashboard/impulse/model-testing?impulseId=${imp.id}`)}
              onDelete={() => deleteImpulse(imp)}
            />
          );
        })()}

        {/* Footer */}
        {!loading && !error && impulses.length > 0 && (
          <div className="manage-impulses-footer flex items-center justify-between px-5 py-2.5 text-[11px]">
            <span>
              Scroll to see more impulses •{" "}
              <strong className="manage-impulses-footer-strong">{impulses.length} PIPELINES TOTAL</strong>
            </span>
            <span className="flex items-center gap-1.5">
              <span className="manage-impulses-footer-dot inline-block" />
              All pipelines synced
            </span>
          </div>
        )}
      </div>

      {deleteModal && (
        <DeleteImpulseModal
          count={deleteModal.count}
          deleting={Boolean(deletingFromModal)}
          onClose={() => {
            if (!deletingFromModal) setDeleteModal(null);
          }}
          onConfirm={() => {
            if (deleteModal.mode === "single") {
              void confirmSingleDelete(deleteModal.impulse);
            } else {
              void confirmBulkDelete();
            }
          }}
        />
      )}
    </div>
  );
}
