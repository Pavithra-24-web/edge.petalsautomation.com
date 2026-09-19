"use client";
import { useCallback, useEffect, useState } from "react";
import { dspApi } from "@/utils/api";
import { Zap, Activity, Database, ChevronRight, Star, X, FlaskConical, Trash2 } from "lucide-react";
import toast from "react-hot-toast";

// ─── Pieces shared by ObjectDetectionImpulse and MotionImpulse ────────────────
// Everything here is modality-agnostic: it reads the impulse's own
// input_type/sensor_type, plus an optional `projectType` threaded through by
// each caller (§0.5) to narrow the processing-block catalog server-side.

export const PETAL_EDGE_AUTHOR = "Petal Edge";

export function normalizeLearningAuthor(author?: string) {
  return author === "Edge Impulse" ? PETAL_EDGE_AUTHOR : (author || PETAL_EDGE_AUTHOR);
}

export const BLOCK_ACCENT: Record<string, string> = {
  spectral_analysis: "border-blue-400/60 bg-blue-950/40",
  mfcc: "border-purple-400/60 bg-purple-950/40",
  spectrogram: "border-teal-400/60 bg-teal-950/40",
  image: "border-orange-400/60 bg-orange-950/40",
  raw: "border-gray-500/60 bg-gray-800/60",
  flatten: "border-yellow-400/60 bg-yellow-950/40",
  eeg: "border-pink-400/60 bg-pink-950/40",
};

// Top-right corner badge style per processing block type. The icon component
// is a Lucide icon; `bg` is the badge fill, `fg` is the icon stroke color.
export const PROCESSING_BLOCK_BADGE: Record<string, { bg: string; fg: string; icon: any }> = {
  image: { bg: "bg-orange-500", fg: "text-white", icon: Zap },
  spectral_analysis: { bg: "bg-blue-500", fg: "text-white", icon: Activity },
  mfcc: { bg: "bg-purple-500", fg: "text-white", icon: Activity },
  spectrogram: { bg: "bg-teal-500", fg: "text-white", icon: Activity },
  raw: { bg: "bg-gray-500", fg: "text-white", icon: Database },
  flatten: { bg: "bg-yellow-500", fg: "text-white", icon: Zap },
  eeg: { bg: "bg-pink-500", fg: "text-white", icon: Activity },
};
export const DEFAULT_PROCESSING_BADGE = { bg: "bg-indigo-500", fg: "text-white", icon: Zap };
export const LEARNING_BLOCK_BADGE = { bg: "bg-white", fg: "text-indigo-700", icon: FlaskConical };

/* ── Arrow connector ──────────────────────────────────────────────────────── */
// `self-center` (not a fixed height + items-center trick) so this centers on
// whatever the row's tallest sibling actually is, whether every card in the
// row is a uniform height (unchanged result — self-center in a fixed-height
// row lands at the same midpoint as before) or the cards vary independently.
export function Arrow() {
  return (
    <div className="flex items-center self-center flex-shrink-0 px-1 sm:px-2">
      <div className="h-px w-6 bg-gray-600 sm:w-8 xl:w-10" />
      <ChevronRight size={13} className="text-gray-500 -ml-0.5 flex-shrink-0" />
    </div>
  );
}

/* ── "Sad folder" icon for the empty-data state ──────────────────────────── */
export function SadFolderIcon() {
  return (
    <svg
      width="56"
      height="56"
      viewBox="0 0 32 32"
      fill="none"
      stroke="url(#pe-warn-folder-grad)"
      strokeWidth="1.8"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      <defs>
        <linearGradient id="pe-warn-folder-grad" x1="0" y1="0" x2="32" y2="32" gradientUnits="userSpaceOnUse">
          <stop offset="0" stopColor="#a855f7" />
          <stop offset="0.5" stopColor="#ec4899" />
          <stop offset="1" stopColor="#6366f1" />
        </linearGradient>
      </defs>
      <path d="M4 10v14a2 2 0 0 0 2 2h20a2 2 0 0 0 2-2V12a2 2 0 0 0-2-2H16l-2-3H6a2 2 0 0 0-2 2Z" />
      <circle cx="13" cy="18" r="0.9" fill="url(#pe-warn-folder-grad)" stroke="none" />
      <circle cx="19" cy="18" r="0.9" fill="url(#pe-warn-folder-grad)" stroke="none" />
      <path d="M13.5 22.5c.7-.8 1.6-1.2 2.5-1.2s1.8.4 2.5 1.2" />
    </svg>
  );
}

/** Mirror of backend `_filter_blocks` (dsp.py): keep a block when its
 *  `input_types`/`sensor_types` contain the active impulse's input/sensor,
 *  with empty strings treated as "no constraint". The fallback (returning the
 *  full list when nothing matches) matches the backend so the modal is never
 *  empty for unusual configs. */
export function filterBlocksForImpulse(blocks: any[], inputType: string, sensorType: string): any[] {
  const result = blocks.filter((b: any) => {
    const typeMatch = !inputType || (b.input_types || []).includes(inputType);
    const sensorMatch = !sensorType || (b.sensor_types || []).includes(sensorType);
    return typeMatch || sensorMatch;
  });
  return result.length ? result : blocks;
}

/* ─────────────────────────────────────────────────────────────────────────────
   Generic block-picker modal shell — the table-of-blocks UI shared by every
   "Add a processing/learning block" dialog. Callers own fetching/filtering;
   this owns layout, loading state and the row markup only.
───────────────────────────────────────────────────────────────────────────── */
interface BlockPickerModalProps {
  titleIcon: any;
  title: string;
  loading: boolean;
  blocks: any[];
  emptyMessage?: string;
  showFooterCancel?: boolean;
  onAdd: (block: any) => void;
  onClose: () => void;
}

export function BlockPickerModal({
  titleIcon: TitleIcon, title, loading, blocks, emptyMessage, showFooterCancel, onAdd, onClose,
}: BlockPickerModalProps) {
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm">
      <div className="bg-white rounded-2xl shadow-2xl w-[680px] max-h-[85vh] flex flex-col overflow-hidden">

        {/* Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-gray-200">
          <div className="flex items-center gap-2">
            <TitleIcon size={17} className="text-indigo-600" />
            <h2 className="text-[15px] font-semibold text-gray-900">{title}</h2>
          </div>
          <button onClick={onClose} className="text-gray-400 hover:text-gray-600">
            <X size={17} />
          </button>
        </div>

        {/* Column headers */}
        <div className="grid grid-cols-[1fr_130px_100px] px-6 py-2 bg-gray-50 border-b border-gray-200">
          <span className="text-[11px] font-semibold text-gray-400 uppercase tracking-wider">Description</span>
          <span className="text-[11px] font-semibold text-gray-400 uppercase tracking-wider">Author</span>
          <span className="text-[11px] font-semibold text-gray-400 uppercase tracking-wider text-right pr-2">Recommended</span>
        </div>

        {/* Block rows */}
        <div className="flex-1 overflow-y-auto divide-y divide-gray-100">
          {loading && <div className="py-14 text-center text-sm text-gray-400">Loading…</div>}
          {!loading && blocks.map(block => (
            <div key={block.type}
              className="grid grid-cols-[1fr_130px_100px] items-center px-6 py-4 hover:bg-gray-50 transition-colors">
              <div className="pr-4">
                <div className="flex items-center gap-2 mb-1">
                  <span className="text-sm font-semibold text-gray-900">{block.name}</span>
                  {block.official && (
                    <span className="px-1.5 py-px rounded-full bg-gray-100 text-gray-500 text-[10px] font-semibold uppercase tracking-wide border border-gray-200">
                      Officially Supported
                    </span>
                  )}
                </div>
                <p className="text-sm text-indigo-700 leading-snug">{block.description}</p>
              </div>
              <span className="text-sm text-gray-600">{block.author}</span>
              <div className="flex items-center justify-end gap-2 pr-1">
                {block.recommended && <Star size={15} className="text-yellow-400 fill-yellow-400 flex-shrink-0" />}
                <button
                  onClick={() => { onAdd(block); onClose(); }}
                  className="px-4 py-1.5 text-sm font-medium text-indigo-700 border border-indigo-300 rounded-lg hover:bg-indigo-50 transition-colors"
                >
                  Add
                </button>
              </div>
            </div>
          ))}
          {!loading && blocks.length === 0 && emptyMessage && (
            <div className="py-14 text-center text-sm text-gray-400">{emptyMessage}</div>
          )}
        </div>

        {showFooterCancel && (
          <div className="px-6 py-3 border-t border-gray-200 bg-gray-50">
            <div className="flex justify-end">
              <button onClick={onClose}
                className="px-4 py-2 text-sm font-medium text-gray-700 bg-white border border-gray-300 rounded-lg hover:bg-gray-50 transition-colors">
                Cancel
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

/* ─────────────────────────────────────────────────────────────────────────────
   "Add a processing block" modal — type-agnostic: fetches the project-type-
   filtered catalog and additionally filters client-side by the impulse's own
   input/sensor type. Used unchanged by both content components.
───────────────────────────────────────────────────────────────────────────── */
export function ProcessingBlockModal({
  inputType, sensorType, projectType, onAdd, onClose,
}: {
  inputType: string;
  sensorType: string;
  projectType?: string;
  onAdd: (block: any) => void;
  onClose: () => void;
}) {
  const [blocks, setBlocks] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);

  const load = useCallback(async () => {
    setLoading(true);
    try {
      // Request the project-type-filtered catalog (§0.5) and then filter
      // client-side by input/sensor type. The catalog route filters by
      // impulse_id, but unsaved drafts have no backend row to look up —
      // without the local filter, a draft image impulse would show
      // MFCC/Spectrogram options that don't apply. show_all is no longer
      // sent (T1): project_type does the narrowing this used to skip.
      const { data } = await dspApi.processingBlocks(undefined, false, projectType);
      const allBlocks = (data.blocks || []).map((block: any) => ({
        ...block,
        author: normalizeLearningAuthor(block.author),
      }));
      const visible = filterBlocksForImpulse(allBlocks, inputType, sensorType);
      setBlocks(visible);
    } catch { toast.error("Failed to load processing blocks"); }
    finally { setLoading(false); }
  }, [inputType, sensorType, projectType]);

  useEffect(() => { load(); }, [load]);

  return (
    <BlockPickerModal
      titleIcon={Zap}
      title="Add a processing block"
      loading={loading}
      blocks={blocks}
      onAdd={onAdd}
      onClose={onClose}
    />
  );
}

/* ─────────────────────────────────────────────────────────────────────────────
   ProcessingBlockBase — the one processing-block card shell. Every block
   (Spectral Analysis, Flatten, MFCC, Raw Data, ...) renders through this;
   it owns the title, icon badge, delete button and the axis selector, and
   takes only block-specific params as `children`. It never computes which
   features are available — that's the impulse graph's job (see
   featurePipeline.ts) — it only renders whatever `availableFeatures` /
   `selectedFeatures` it's handed and reports toggles back up.
───────────────────────────────────────────────────────────────────────────── */
export function ProcessingBlockBase({
  block, badge, accentClass, availableFeatures, selectedFeatures,
  onAxesChange, onDelete, children,
}: {
  block: any;
  badge: { bg: string; fg: string; icon: any };
  accentClass: string;
  availableFeatures: string[];
  selectedFeatures: string[];
  onAxesChange: (axes: string[]) => void;
  onDelete: () => void;
  children?: React.ReactNode;
}) {
  const BadgeIcon = badge.icon;

  function toggleAxis(axis: string) {
    const next = selectedFeatures.includes(axis)
      ? selectedFeatures.filter((a) => a !== axis)
      : [...selectedFeatures, axis];
    onAxesChange(next);
  }

  return (
    <div
      data-block-type={block.type}
      className={`impulse-processing-card w-full min-h-[26rem] border rounded-xl flex-shrink-0 overflow-visible relative transition-all ${accentClass}`}
    >
      <div className="px-5 py-4 flex items-center justify-between rounded-t-xl border-b border-white/5">
        <div className="flex-1 min-w-0 pr-3">
          <p className="text-[9px] text-gray-400 uppercase tracking-wider mb-0.5">Processing</p>
          <p className="text-base font-semibold text-gray-100 truncate">
            {block.name || (block?.type ?? "").replace(/_/g, " ")}
          </p>
        </div>
        <div className="flex items-center gap-2 flex-shrink-0">
          <button
            type="button"
            aria-label="Delete processing block"
            onClick={onDelete}
            className="p-1.5 rounded text-gray-500 hover:text-red-400 hover:bg-red-500/10 transition-colors"
          >
            <Trash2 size={13} />
          </button>
          <div className={`h-10 w-10 rounded-full flex items-center justify-center shadow-md ${badge.bg}`}>
            <BadgeIcon size={20} className={badge.fg} strokeWidth={2.5} />
          </div>
        </div>
      </div>

      <div className="p-4 sm:p-5 space-y-3">
        <div>
          <p className="text-[10px] text-gray-500 mb-0.5">
            Input axes ({selectedFeatures.length})
          </p>
          <div className="impulse-axis-list max-h-64 overflow-y-auto rounded border border-white/10 divide-y divide-white/5">
            {availableFeatures.length === 0 && (
              <p className="text-[10px] text-gray-500 px-2 py-1.5">No input features available</p>
            )}
            {availableFeatures.map((axis) => (
              <label
                key={axis}
                className="flex items-center gap-1.5 px-2 py-1 text-[11px] text-gray-300 hover:bg-white/5 cursor-pointer"
              >
                <input
                  type="checkbox"
                  checked={selectedFeatures.includes(axis)}
                  onChange={() => toggleAxis(axis)}
                  className="accent-indigo-500 h-3 w-3 flex-shrink-0"
                />
                <span className="truncate">{axis}</span>
              </label>
            ))}
          </div>
        </div>

        {children}
      </div>
    </div>
  );
}
