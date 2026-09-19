"use client";
import { useEffect, useMemo, useLayoutEffect, useState, useRef } from "react";
import { createPortal } from "react-dom";
import { useAppStore } from "@/store/appStore";
import { samplesApi, labelsApi, aiLabelingApi, projectsApi } from "@/utils/api";
import {
  Upload, Trash2, RefreshCw, XCircle,
  FileText, Music, Image as ImageIcon, Video, Database, ArrowLeft, MoreVertical, Check, Cpu,
  Plus, Play, X, Edit, Eye, Settings, Zap, BarChart3, CheckCircle, AlertTriangle, Loader2, ClipboardList, HelpCircle,
  Home, ChevronRight, Sparkles, Smartphone, Maximize2, List, Columns3, Download,
  DownloadCloudIcon,
  UploadCloud,
  HardDrive,
  Wand2,
} from "lucide-react";
import {
  BarChart, Bar, XAxis, YAxis, Tooltip as RTooltip, CartesianGrid, ResponsiveContainer, Cell,
} from "recharts";
import toast from "react-hot-toast";
import Link from "next/link";
import { useRouter } from "next/navigation";
import useImage from 'use-image';
import { v4 as uuidv4 } from 'uuid';
import { useExportProject } from "@/hooks/useExportProject";
import { useUnlabeledCount } from "@/hooks/useUnlabeledCount";
import { SplitKey, PieSegment, PREMIUM_SOFT_PALETTE, PieChart, EmptyFolderIllustration } from "./LabelingShared";
import { AnnotationEditor } from "./AnnotationEditor";
import { fetchSampleUrl, prefetchSample } from "./sampleImageCache";
import { buildLabelColorMap, colorForLabelFromMap, pillStyleFromMap } from "./labelColors";
import { notifySamplesChanged } from "./labelingEvents";

// ─── Helpers ──────────────────────────────────────────────────────────────────

function fileIcon(filename: string) {
  const ext = filename.split(".").pop()?.toLowerCase() ?? "";
  if (["jpg", "jpeg", "png"].includes(ext)) return <ImageIcon size={13} />;
  if (["wav", "mp3"].includes(ext)) return <Music size={13} />;
  if (["avi", "mp4"].includes(ext)) return <Video size={13} />;
  if (["csv", "json", "cbor", "parquet"].includes(ext)) return <Database size={13} />;
  return <FileText size={13} />;
}

function isVideoFile(filename: string) {
  const ext = filename.split(".").pop()?.toLowerCase() ?? "";
  return ["avi", "mp4"].includes(ext);
}

// ─── Train/Test split health icon ────────────────────────────────────────────
// Single icon slot rendered next to the inline ratio. Swaps the chart icon
// for a red warning triangle when overall health is unhealthy; the tooltip
// content swaps in lockstep. The tooltip is portalled and positioned with
// fixed coordinates so the card's overflow / stacking context can't clip
// it (the Train/Test card lives inside the stat-grid which has its own
// rounded clip).
function SplitHealthIconButton({
  healthy,
  onClick,
  buttonRef,
}: {
  healthy: boolean;
  onClick: () => void;
  buttonRef: React.RefObject<HTMLButtonElement>;
}) {
  const [open, setOpen] = useState(false);
  const tooltipRef = useRef<HTMLDivElement | null>(null);
  const [pos, setPos] = useState<{
    top: number;
    left: number;
    placement: "top" | "bottom";
    arrowLeft: number;
  } | null>(null);

  const message = healthy
    ? "See the dataset split per label in your dataset."
    : "One or more of the labels in your dataset have a poor dataset split. Click to learn how to rebalance your dataset.";

  // Position the tooltip after each open. Prefers above the icon; flips
  // below when there's not enough room (the card sits near the top of the
  // page so this happens routinely). Horizontal anchor is the icon center,
  // clamped inside the viewport so the 260px-wide bubble never escapes
  // the visible area; the arrow follows the icon center even after clamp.
  useLayoutEffect(() => {
    if (!open) return;
    const tip = tooltipRef.current;
    const btn = buttonRef.current;
    if (!tip || !btn) return;

    const compute = () => {
      const btnRect = btn.getBoundingClientRect();
      const tipRect = tip.getBoundingClientRect();
      const margin = 8;
      const gap = 8;

      let placement: "top" | "bottom" = "top";
      let top = btnRect.top - tipRect.height - gap;
      if (top < margin) {
        placement = "bottom";
        top = btnRect.bottom + gap;
      }

      const btnCenterX = btnRect.left + btnRect.width / 2;
      let left = btnCenterX - tipRect.width / 2;
      const minLeft = margin;
      const maxLeft = window.innerWidth - tipRect.width - margin;
      if (maxLeft > minLeft) {
        left = Math.max(minLeft, Math.min(left, maxLeft));
      } else {
        left = minLeft;
      }

      const arrowLeft = Math.max(8, Math.min(tipRect.width - 8, btnCenterX - left));
      setPos({ top, left, placement, arrowLeft });
    };

    compute();
    window.addEventListener("resize", compute);
    window.addEventListener("scroll", compute, true);
    return () => {
      window.removeEventListener("resize", compute);
      window.removeEventListener("scroll", compute, true);
    };
  }, [open, buttonRef, message]);

  // Esc dismisses the tooltip without affecting the modal. The modal's own
  // Esc handler only activates while it is open, so the two don't fight.
  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => { if (e.key === "Escape") setOpen(false); };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open]);

  // Clear stale position when the tooltip closes so the next open starts
  // from a fresh measure (otherwise re-opening near a different edge would
  // briefly flash the old coordinates).
  useEffect(() => {
    if (!open) setPos(null);
  }, [open]);

  return (
    <>
      <button
        ref={buttonRef}
        type="button"
        onClick={onClick}
        onMouseEnter={() => setOpen(true)}
        onMouseLeave={() => setOpen(false)}
        onFocus={() => setOpen(true)}
        onBlur={() => setOpen(false)}
        aria-label={healthy
          ? "View split details"
          : "One or more labels have a poor dataset split"}
        className={`ds-split-icon-btn ${healthy ? "is-healthy" : "is-unhealthy"}`}
      >
        {healthy
          ? <BarChart3 size={14} />
          : <AlertTriangle size={12} strokeWidth={2.4} />}
      </button>
      {open && typeof document !== "undefined" && createPortal(
        <div
          ref={tooltipRef}
          role="tooltip"
          className={`ds-split-icon-tip ${pos?.placement === "bottom" ? "is-below" : "is-above"}`}
          style={{
            top: pos ? pos.top : -9999,
            left: pos ? pos.left : -9999,
            opacity: pos ? 1 : 0,
          }}
        >
          {message}
          <span
            className="ds-split-icon-tip-arrow"
            style={{ left: pos ? pos.arrowLeft : "50%" }}
          />
        </div>,
        document.body
      )}
    </>
  );
}
// ─── Train/Test split modal ───────────────────────────────────────────────────
// Per-label and overall health come from the /dataset-health endpoint.
// Rows are ordered unhealthy-first → alphabetical so the user reads the
// problem classes before the healthy ones; bar color follows label.healthy
// (red for unhealthy, green otherwise) per the reference design.

type LabelHealth = {
  name: string;
  training: number;
  testing: number;
  healthy: boolean;
  reason: string | null;
};

interface SplitRatioModalProps {
  open: boolean;
  onClose: () => void;
  labels: LabelHealth[];
  returnFocusRef?: { readonly current: HTMLElement | null };
}

function SplitRatioModal({ open, onClose, labels, returnFocusRef }: SplitRatioModalProps) {
  const [helpOpen, setHelpOpen] = useState(false);

  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => { if (e.key === "Escape") onClose(); };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open, onClose]);

  // Return focus to the inline trigger when the modal closes so keyboard
  // users don't lose their place.
  useEffect(() => {
    if (!open && returnFocusRef?.current) {
      returnFocusRef.current.focus();
    }
  }, [open, returnFocusRef]);

  if (!open) return null;

  const rows = [...labels].sort((a, b) => {
    if (a.healthy !== b.healthy) return a.healthy ? 1 : -1;
    return a.name.toLowerCase().localeCompare(b.name.toLowerCase());
  });

  const unhealthyNames = rows.filter(r => !r.healthy).map(r => r.name);

  return (
    <div className="ds-split-modal-overlay" onClick={onClose} role="dialog" aria-modal="true">
      <div className="ds-split-modal" onClick={(e) => e.stopPropagation()}>
        <div className="ds-split-modal-head">
          <h3 className="ds-split-modal-title">
            <span className="ds-split-modal-title-icon" aria-hidden="true">
              <ClipboardList size={16} strokeWidth={2.1} />
            </span>
            Dataset train / test split ratio
          </h3>
          <button type="button" onClick={onClose} className="ds-split-modal-close" aria-label="Close">
            <X size={16} />
          </button>
        </div>

        <p className="ds-split-modal-intro">
          <strong>Training data</strong> is used to train your model, and <strong>testing data</strong> is
          used to test your model&apos;s accuracy after training. We recommend an approximate 80/20
          train/test split ratio for your data for every class (or label) in your dataset, although
          especially large datasets may require less testing data.
        </p>

        <div className="ds-split-row">
          <div className="ds-split-row-head">
            <span className="ds-split-suggest-pill">SUGGESTED TRAIN / TEST SPLIT</span>
            <span className="ds-split-row-pct">80% / 20%</span>
          </div>
          <div className="ds-split-bar">
            <div className="ds-split-bar-fill" style={{ width: "80%" }} />
          </div>
        </div>

        <div className="ds-split-section-head">
          <span className="ds-split-section-title">Labels in your dataset</span>
          <span className="ds-split-help-wrap">
            <button
              type="button"
              className="ds-split-help-btn"
              onMouseEnter={() => setHelpOpen(true)}
              onMouseLeave={() => setHelpOpen(false)}
              onFocus={() => setHelpOpen(true)}
              onBlur={() => setHelpOpen(false)}
              aria-label="What's shown here"
            >
              <HelpCircle size={14} />
            </button>
            {helpOpen && (
              <span className="ds-split-help-tip" role="tooltip">
                Per-class breakdown of training vs. testing samples.
              </span>
            )}
          </span>
        </div>

        {unhealthyNames.length > 0 && (
          <p className="ds-split-unhealthy-subtext">
            Some classes have a poor train/test split ratio: {unhealthyNames.join(", ")}.
            To fix this, add or move samples to the training or testing data.
          </p>
        )}

        <div className="ds-split-rows">
          {rows.length === 0 ? (
            <p className="ds-split-empty">No labeled training data yet.</p>
          ) : rows.map((row) => {
            const t = row.training + row.testing;
            const trainPct = t > 0 ? Math.round((row.training / t) * 100) : 0;
            const testPct = t > 0 ? 100 - trainPct : 0;
            return (
              <div
                key={row.name}
                className="ds-split-row"
                title={!row.healthy && row.reason ? row.reason : undefined}
              >
                <div className="ds-split-row-head">
                  <span className="ds-split-label-pill">{row.name.toUpperCase()}</span>
                  <span className="ds-split-row-pct">
                    {trainPct}% / {testPct}% ({row.training} / {row.testing})
                  </span>
                </div>
                <div className="ds-split-bar">
                  <div
                    className={`ds-split-bar-fill ${row.healthy ? "" : "is-unhealthy"}`}
                    style={{ width: `${trainPct}%` }}
                  />
                </div>
              </div>
            );
          })}
        </div>

        <div className="ds-split-modal-foot">
          <button type="button" onClick={onClose} className="ds-split-dismiss-btn">Dismiss</button>
        </div>
      </div>
    </div>
  );
}

// ─── Data Distribution modal (Training) ──────────────────────────────────────
interface DataDistributionModalProps {
  open: boolean;
  onClose: () => void;
  perLabel: Record<string, { training: number; testing: number }>;
  colorForLabel: (label: string) => string;
}

function DataDistributionModal({ open, onClose, perLabel, colorForLabel }: DataDistributionModalProps) {
  const [fullscreen, setFullscreen] = useState(false);

  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => { if (e.key === "Escape") onClose(); };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open, onClose]);

  useEffect(() => { if (!open) setFullscreen(false); }, [open]);

  const data = useMemo(() => {
    return Object.entries(perLabel)
      .filter(([label, v]) => label !== "Unlabeled" && v.training > 0)
      .map(([label, v]) => ({ label, value: v.training, color: colorForLabel(label) }))
      .sort((a, b) => a.label.localeCompare(b.label));
  }, [perLabel, colorForLabel]);

  if (!open) return null;

  return (
    <div className="ds-dist-modal-overlay" onClick={onClose} role="dialog" aria-modal="true">
      <div
        className={`ds-dist-modal ${fullscreen ? "is-fullscreen" : ""}`}
        onClick={(e) => e.stopPropagation()}
      >
        <div className="ds-dist-modal-head">
          <h3 className="ds-dist-modal-title">
            <BarChart3 size={16} strokeWidth={2.1} />
            Data distribution <span className="ds-dist-modal-title-sub">(Training)</span>
          </h3>
          <div className="ds-dist-modal-head-actions">
            <button
              type="button"
              onClick={() => setFullscreen(f => !f)}
              className="ds-dist-icon-btn"
              aria-label={fullscreen ? "Exit fullscreen" : "Fullscreen"}
            >
              <Maximize2 size={14} />
            </button>
            <button type="button" onClick={onClose} className="ds-dist-icon-btn" aria-label="Close">
              <X size={16} />
            </button>
          </div>
        </div>

        <div className="ds-dist-modal-body">
          {data.length === 0 ? (
            <div className="ds-dist-empty">No training data available.</div>
          ) : (
            <div className="ds-dist-chart-wrap" style={{ height: fullscreen ? 560 : 420 }}>
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={data} margin={{ top: 16, right: 24, left: 8, bottom: 32 }}>
                  <CartesianGrid stroke="#E5E7EB" vertical={false} />
                  <XAxis
                    dataKey="label"
                    tick={{ fontSize: 12, fill: "#6B7280" }}
                    label={{ value: "Label", position: "insideBottom", offset: -16, fill: "#6B7280", fontSize: 12 }}
                  />
                  <YAxis
                    tick={{ fontSize: 12, fill: "#6B7280" }}
                    label={{ value: "Items", angle: -90, position: "insideLeft", fill: "#6B7280", fontSize: 12 }}
                    allowDecimals={false}
                  />
                  <RTooltip
                    cursor={{ fill: "rgba(107,114,128,0.08)" }}
                    contentStyle={{ background: "#ffffff", border: "1px solid #E5E7EB", borderRadius: 8, fontSize: 12, color: "#111827" }}
                    formatter={(v: any) => [v, "Items"]}
                  />
                  <Bar
                    dataKey="value"
                    radius={[2, 2, 0, 0]}
                    animationDuration={1400}
                    animationEasing="ease-out"
                    isAnimationActive
                  >
                    {data.map((d) => <Cell key={d.label} fill={d.color} />)}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

// ─── Grid / Detailed view ─────────────────────────────────────────────────────
// Card grid that displays each sample as an image tile with bounding-box
// overlays, colored label tags, and the sample ID. Powers both the in-card
// "Switch to grid view" mode and the full-width "Show detailed view" mode.

interface DetailedSampleGridProps {
  samples: any[];
  columns: number;
  itemsPerPage: number;
  page: number;
  setPage: (p: number) => void;
  inlineEdit: boolean;
  labels: any[];
  usedLabelKeys: Set<string>;
  pillStyleForLabel: (label: string) => React.CSSProperties;
  normalizeBoxLabel: (value: any) => string;
  onSelectSample: (s: any) => void;
  onUpdateBoxLabel: (sampleId: string, boxIdx: number, newLabel: string) => void;
  onSaveBoxes: (sampleId: string, newBoxes: any[]) => Promise<boolean>;
  onCreateLabel: (name: string) => Promise<{ id: string; name: string } | null>;
  colorForLabel: (label: string) => string;
  renderRowMenu?: (s: any) => React.ReactNode;
}

function DetailedSampleGrid({
  samples, columns, itemsPerPage, page, setPage, inlineEdit,
  labels, usedLabelKeys, pillStyleForLabel, normalizeBoxLabel,
  onSelectSample, onUpdateBoxLabel, onSaveBoxes, onCreateLabel, colorForLabel,
  renderRowMenu,
}: DetailedSampleGridProps) {
  const totalPages = Math.max(1, Math.ceil(samples.length / itemsPerPage));
  const currentPage = Math.min(page, totalPages);
  const start = (currentPage - 1) * itemsPerPage;
  const visible = samples.slice(start, start + itemsPerPage);

  return (
    <div className="w-full flex-1 overflow-y-auto min-h-0 px-1">
      <div
        className="grid gap-4 pb-3 items-stretch"
        style={{
          gridTemplateColumns: `repeat(${columns}, minmax(0, 1fr))`,
          gridAutoRows: "1fr",
        }}
      >
        {visible.map(s => (
          <DetailedSampleCard
            key={s.id}
            sample={s}
            inlineEdit={inlineEdit}
            labels={labels}
            usedLabelKeys={usedLabelKeys}
            pillStyleForLabel={pillStyleForLabel}
            normalizeBoxLabel={normalizeBoxLabel}
            onClick={() => onSelectSample(s)}
            onUpdateBoxLabel={onUpdateBoxLabel}
            onSaveBoxes={onSaveBoxes}
            onCreateLabel={onCreateLabel}
            colorForLabel={colorForLabel}
            renderRowMenu={renderRowMenu}
          />
        ))}
      </div>

      {totalPages > 1 && (
        <div
          className="flex items-center justify-between py-3 text-xs"
          style={{ color: "var(--app-text-soft)" }}
        >
          <span>
            Showing {start + 1}–{Math.min(start + itemsPerPage, samples.length)} of {samples.length}
          </span>
          <div className="flex items-center gap-2">
            <button
              className="px-2.5 py-1 rounded-md border"
              style={{
                background: "var(--app-surface)",
                borderColor: "var(--app-border)",
                color: "var(--app-text)",
                opacity: currentPage === 1 ? 0.45 : 1,
                cursor: currentPage === 1 ? "not-allowed" : "pointer",
              }}
              onClick={() => currentPage > 1 && setPage(currentPage - 1)}
              disabled={currentPage === 1}
            >
              Previous
            </button>
            <span style={{ color: "var(--app-text)" }}>
              Page {currentPage} / {totalPages}
            </span>
            <button
              className="px-2.5 py-1 rounded-md border"
              style={{
                background: "var(--app-surface)",
                borderColor: "var(--app-border)",
                color: "var(--app-text)",
                opacity: currentPage === totalPages ? 0.45 : 1,
                cursor: currentPage === totalPages ? "not-allowed" : "pointer",
              }}
              onClick={() => currentPage < totalPages && setPage(currentPage + 1)}
              disabled={currentPage === totalPages}
            >
              Next
            </button>
          </div>
        </div>
      )}
    </div>
  );
}

interface DetailedSampleCardProps {
  sample: any;
  inlineEdit: boolean;
  labels: any[];
  usedLabelKeys: Set<string>;
  pillStyleForLabel: (label: string) => React.CSSProperties;
  normalizeBoxLabel: (value: any) => string;
  onClick: () => void;
  onUpdateBoxLabel: (sampleId: string, boxIdx: number, newLabel: string) => void;
  onSaveBoxes: (sampleId: string, newBoxes: any[]) => Promise<boolean>;
  // Threaded through to the shared AnnotationEditor used in inline-edit mode.
  onCreateLabel: (name: string) => Promise<{ id: string; name: string } | null>;
  colorForLabel: (label: string) => string;
  // Shared kebab menu (rename/labels/split/disable/download/delete) rendered by
  // the parent so the grid card exposes the exact same actions as the list rows.
  renderRowMenu?: (s: any) => React.ReactNode;
}

// Read the stored box geometry in canonical {x, y, w, h} form. Storage may use
// `width/height` aliases (see normalizeBoxGeometryShape used by the main editor).
function readBox(b: any) {
  return {
    x: Number(b?.x ?? 0),
    y: Number(b?.y ?? 0),
    w: Number(b?.w ?? b?.width ?? 0),
    h: Number(b?.h ?? b?.height ?? 0),
  };
}

function DetailedSampleCard({
  sample, inlineEdit, labels,
  pillStyleForLabel, normalizeBoxLabel, onClick, onSaveBoxes,
  onCreateLabel, colorForLabel, renderRowMenu,
}: DetailedSampleCardProps) {
  const [url, setUrl] = useState<string | null>(null);
  const [nat, setNat] = useState<{ w: number; h: number } | null>(null);
  const imgRef = useRef<HTMLImageElement>(null);
  // Guards the <img onError> path so a genuinely-broken URL retries at most once
  // (force-refreshing the presigned URL) instead of looping on repeated errors.
  const imgRetriedRef = useRef(false);
  const ext = (sample.filename?.split(".").pop() || "").toLowerCase();
  const isImg = ["jpg", "jpeg", "png"].includes(ext);

  // Presigned URL for the sample image (shared cache keeps repeat views cheap).
  useEffect(() => {
    if (!isImg) { setUrl(null); return; }
    let cancelled = false;
    imgRetriedRef.current = false; // allow one onError retry per (re)loaded sample
    fetchSampleUrl(sample.id).then(u => {
      if (!cancelled) setUrl(u);
    }).catch(err => {
      // Surface instead of swallowing — this is the path that previously left
      // cards silently blank when the download() call failed under load.
      if (!cancelled) console.warn(`[dataset] thumbnail URL fetch failed for sample ${sample.id}`, err);
    });
    return () => { cancelled = true; };
  }, [sample.id, isImg]);

  // Decoded bitmap for the shared Konva AnnotationEditor (inline-edit mode
  // only). The read-only thumbnail path renders the <img> directly, so the
  // decode is only paid when the editor actually needs it.
  const [konvaImage] = useImage(inlineEdit && url ? url : "", "anonymous");

  // Boxes stored on the sample, in canonical shape with stable ids. Read-only
  // overlays render straight from this; the editor snapshots it via resetKey.
  // Fingerprinted so unrelated re-renders don't churn ids / remount overlays,
  // and so an edit made in the right-hand panel re-seeds the overlays here.
  const sampleBoxesKey = useMemo(
    () => JSON.stringify(sample.extra_metadata?.boundingBoxes ?? []),
    [sample.extra_metadata?.boundingBoxes]
  );
  const boxes: any[] = useMemo(
    () => (sample.extra_metadata?.boundingBoxes || []).map((b: any) => ({
      ...b,
      id: b.id || uuidv4(),
    })),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [sampleBoxesKey]
  );

  const baseName = sample.filename?.split("/").pop()?.split("\\").pop() || sample.id;
  const shortName = baseName.replace(/\.[^.]+$/, "");
  const longId = String(sample.id || "").replace(/-/g, "").slice(0, 10);

  // With object-contain the whole image is shown letterboxed inside a fixed
  // aspect-ratio container. Compute the displayed image rect (as fractions of
  // the container) so bbox %s and drag deltas map to the fitted image rather
  // than the raw container.
  // CARD_ASPECT must match the aspect-[X/Y] applied to the image container.
  const CARD_ASPECT = 4 / 3;
  function imageFit() {
    if (!nat) return null;
    const imageAspect = nat.w / nat.h;
    if (imageAspect > CARD_ASPECT) {
      // Wider than container → fits to width, letterboxed top/bottom.
      const fracW = 1;
      const fracH = CARD_ASPECT / imageAspect;
      return { fracW, fracH, offX: 0, offY: (1 - fracH) / 2 };
    } else {
      // Taller than (or matches) container → fits to height, pillarboxed.
      const fracW = imageAspect / CARD_ASPECT;
      const fracH = 1;
      return { fracW, fracH, offX: (1 - fracW) / 2, offY: 0 };
    }
  }

  return (
    <div
      onClick={onClick}
      className="flex h-full flex-col overflow-hidden rounded-xl border cursor-pointer transition-colors"
      style={{
        background: "var(--app-surface)",
        borderColor: "var(--app-border)",
      }}
    >
      <div
        className="shrink-0 flex items-center gap-2 px-3 h-9 text-xs font-medium"
        style={{ color: "var(--app-text)", borderBottom: "1px solid var(--app-border)" }}
        title={baseName}
      >
        <span className="truncate">{shortName}</span>
        {renderRowMenu && (
          // Same kebab menu as the list rows. Stop clicks bubbling to the card's
          // onClick (which opens the right-hand preview) — the trigger button
          // already calls stopPropagation, this guards the wrapper too.
          <span
            className="ml-auto shrink-0 relative text-right"
            onClick={e => e.stopPropagation()}
          >
            {renderRowMenu(sample)}
          </span>
        )}
      </div>
      <div
        className="relative w-full shrink-0 overflow-hidden flex flex-col"
        style={{ background: "var(--app-surface-2)", aspectRatio: "4 / 3" }}
        // In inline-edit mode the Konva AnnotationEditor owns pointer/click
        // handling; stop those from bubbling to the card's onClick (which opens
        // the right-hand panel) so drawing/selecting doesn't also navigate away.
        onClick={inlineEdit ? (e => e.stopPropagation()) : undefined}
      >
        {isImg && url ? (
          inlineEdit ? (
            // Inline-edit mode → the SAME shared Konva AnnotationEditor the
            // right-hand panel uses, so draw / nested-label / move / resize /
            // delete behave identically and can no longer drift between views.
            // It persists on every change via onSaveBoxes (no manual Save).
            konvaImage ? (
              <AnnotationEditor
                image={konvaImage}
                initialBoxes={boxes}
                resetKey={`gridcard:${sample.id}:${sample._ts ?? ""}`}
                labels={labels}
                onChange={(next) => { onSaveBoxes(sample.id, next); }}
                onCreateLabel={onCreateLabel}
                colorForLabel={colorForLabel}
              />
            ) : (
              <div className="absolute inset-0 flex items-center justify-center" style={{ color: "var(--app-text-soft)" }}>
                <Loader2 size={18} className="animate-spin opacity-70" />
              </div>
            )
          ) : (
            <>
              <img
                ref={imgRef}
                src={url}
                alt={baseName}
                draggable={false}
                onLoad={e => {
                  const t = e.currentTarget;
                  setNat({ w: t.naturalWidth, h: t.naturalHeight });
                }}
                onError={() => {
                  // The download() call succeeded but the presigned GET failed
                  // (e.g. an expired/stale cached URL). Force a fresh URL once.
                  if (imgRetriedRef.current) {
                    console.warn(`[dataset] thumbnail image failed to load for sample ${sample.id}`);
                    return;
                  }
                  imgRetriedRef.current = true;
                  fetchSampleUrl(sample.id, true).then(setUrl).catch(() => { });
                }}
                style={{ display: "block", width: "100%", height: "100%", objectFit: "contain" }}
              />
              {(() => {
                const fit = imageFit();
                return fit && boxes.map((b, i) => {
                  const g = readBox(b);
                  const left = ((g.x / nat!.w) * fit.fracW + fit.offX) * 100;
                  const top = ((g.y / nat!.h) * fit.fracH + fit.offY) * 100;
                  const width = (g.w / nat!.w) * fit.fracW * 100;
                  const height = (g.h / nat!.h) * fit.fracH * 100;
                  const label = normalizeBoxLabel(b.label) || "unlabeled";
                  const pill = pillStyleForLabel(label);
                  const color = pill.color as string;
                  return (
                    <div
                      key={b.id || i}
                      className="absolute"
                      style={{
                        left: `${left}%`,
                        top: `${top}%`,
                        width: `${width}%`,
                        height: `${height}%`,
                        border: `2px solid ${color}`,
                        boxShadow: "0 0 0 1px rgba(0,0,0,0.25)",
                        pointerEvents: "none",
                      }}
                    >
                      {/* Label tag (read-only in thumbnail mode) */}
                      <span
                        className="absolute text-[10px] font-semibold px-1.5 py-0.5 leading-tight whitespace-nowrap"
                        style={{
                          top: -18, left: -1,
                          background: color,
                          color: "#fff",
                          borderTopLeftRadius: 4,
                          borderTopRightRadius: 4,
                          pointerEvents: "none",
                        }}
                      >
                        {label}
                      </span>
                    </div>
                  );
                });
              })()}
            </>
          )
        ) : isImg && !url ? (
          <div className="absolute inset-0 flex items-center justify-center" style={{ color: "var(--app-text-soft)" }}>
            <Loader2 size={18} className="animate-spin opacity-70" />
          </div>
        ) : isVideoFile(sample.filename) ? (
          // Reuse the right preview panel's "Video samples can't be previewed
          // here" icon (Video / text-violet-400/70 / strokeWidth 1.6) so the
          // empty state matches across views — see line 3009-area panel.
          <div className="absolute inset-0 flex items-center justify-center">
            <Video size={40} className="text-violet-400/70" strokeWidth={1.6} />
          </div>
        ) : (
          <div className="absolute inset-0 flex items-center justify-center text-xs" style={{ color: "var(--app-text-soft)" }}>
            No preview
          </div>
        )}
      </div>
      <div
        className="mt-auto shrink-0 flex items-center gap-2 px-3 h-10 text-[11px]"
        style={{ color: "var(--app-text-soft)", borderTop: "1px solid var(--app-border)" }}
      >
        <span>ID: {longId}</span>
        {sample.is_background && <BackgroundBadge />}
      </div>
    </div>
  );
}

// Marks a deliberate training negative — an image the user asserts contains
// none of the project's classes. Distinct from "Unlabeled", which means nobody
// has looked at it yet; the two are the same on disk apart from the explicit
// `is_background` marker, so the badge is the only thing that tells them apart
// in the grid and list.
function BackgroundBadge() {
  return (
    <span
      className="data-acq-pill shrink-0"
      title="Background image — trained as a negative (contains no objects)"
      style={{
        background: "rgba(148, 163, 184, 0.16)",
        color: "var(--app-text-soft)",
        border: "1px dashed var(--app-border)",
      }}
    >
      Background
    </span>
  );
}

// ─── Component ────────────────────────────────────────────────────────────────

export default function ObjectDetectionLabeling() {
  const { activeProject } = useAppStore();
  const router = useRouter();
  const SAMPLE_LIST_LIMIT = 1000;
  const { exportProject, busy: exportBusy } = useExportProject();
  const { count: unlabeledCount } = useUnlabeledCount(activeProject?.id);

  const [samples, setSamples] = useState<any[]>([]);
  const [labels, setLabels] = useState<any[]>([]);
  const [loading, setLoading] = useState(false);
  // Header Refresh button spinner. Distinct from `loading` (which also flips
  // on filter changes) so the icon only spins for the user-initiated refresh
  // click and stops the moment the refresh promise resolves.
  const [refreshing, setRefreshing] = useState(false);
  const [filterLabel, setFilterLabel] = useState("");
  const [filterType, setFilterType] = useState<SplitKey>("training");
  // ── Dataset view-settings (toolbar) ──────────────────────────────────────
  // Three layout modes:
  //   list     → table view; right panel visible; toolbar offers grid switch + rows-per-page.
  //   grid     → card grid view confined to the dataset card width; right panel still visible.
  //   detailed → card grid view that spans the row full-width; right panel hidden.
  // Maximize2 button toggles between `detailed` and the user's most-recent
  // non-detailed mode (list or grid); the popover offers list↔grid swaps.
  const [layoutMode, setLayoutMode] = useState<"list" | "grid" | "detailed">(() => {
    if (typeof window === "undefined") return "list";
    const v = window.localStorage.getItem("dataset:layoutMode");
    return v === "grid" || v === "detailed" || v === "list" ? v : "list";
  });
  const [lastNonDetailedMode, setLastNonDetailedMode] = useState<"list" | "grid">(() => {
    if (typeof window === "undefined") return "list";
    const v = window.localStorage.getItem("dataset:lastNonDetailedMode");
    return v === "grid" || v === "list" ? v : "list";
  });
  const [gridColumns, setGridColumns] = useState(3);
  const [itemsPerPage, setItemsPerPage] = useState(12);   // grid/detailed
  const [listRowsPerPage, setListRowsPerPage] = useState(12); // list
  const [inlineEditBoxLabels, setInlineEditBoxLabels] = useState(false);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [gridPage, setGridPage] = useState(1);
  const [listPage, setListPage] = useState(1);
  const settingsRef = useRef<HTMLDivElement>(null);

  const switchLayoutMode = (mode: "list" | "grid" | "detailed") => {
    setLayoutMode(mode);
    if (typeof window !== "undefined") window.localStorage.setItem("dataset:layoutMode", mode);
    if (mode !== "detailed") {
      setLastNonDetailedMode(mode);
      if (typeof window !== "undefined") window.localStorage.setItem("dataset:lastNonDetailedMode", mode);
    }
  };
  // "detailed" only changes container width + right-panel visibility — the
  // actual table-vs-grid choice carries over from whichever non-detailed mode
  // the user was in last, so toggling Maximize2 in list view expands the table
  // and toggling it in grid view expands the grid.
  const effectiveLayout: "list" | "grid" =
    layoutMode === "detailed" ? lastNonDetailedMode : layoutMode;
  const [splitCounts, setSplitCounts] = useState({ training: 0, testing: 0, postprocessing: 0 });
  const [usedLabelKeys, setUsedLabelKeys] = useState<Set<string>>(new Set());
  const [labelCounts, setLabelCounts] = useState<Record<string, number>>({});
  const [labelSplit, setLabelSplit] = useState<Record<string, { training: number; testing: number }>>({});
  const [showSplitModal, setShowSplitModal] = useState(false);
  const [showDistributionModal, setShowDistributionModal] = useState(false);
  const [datasetHealth, setDatasetHealth] = useState<{
    overall: { training: number; testing: number; healthy: boolean; reason: string | null };
    labels: { name: string; training: number; testing: number; healthy: boolean; reason: string | null }[];
  } | null>(null);
  const splitTriggerRef = useRef<HTMLButtonElement | null>(null);
  const [menuOpen, setMenuOpen] = useState<string | null>(null);
  const [menuPos, setMenuPos] = useState<{ top: number; left: number } | null>(null);
  const [selectedIds, setSelectedIds] = useState<Set<string>>(new Set());

  // Build a stable label → color map using evenly-spaced HSL hues.
  // Labels are sorted alphabetically (excluding "Unlabeled") so the same label
  // always gets the same index — and thus the same hue — regardless of count
  // changes or render order. This guarantees unique colors for every label and
  // consistent colors across pie slices, legend dots, pills, and bbox overlays.
  const labelColorMap = useMemo(() => buildLabelColorMap(labelCounts), [labelCounts]);

  const colorForLabel = (label: string): string => colorForLabelFromMap(labelColorMap, label);

  const pillStyleForLabel = (label: string): React.CSSProperties => pillStyleFromMap(labelColorMap, label);

  // View mode: dataset vs ai-labeling
  const [viewMode, setViewMode] = useState<"dataset" | "ai-labeling">("dataset");

  // ── AI Labeling State ──────────────────────────────────────────────────────
  const [aiActions, setAiActions] = useState<any[]>([]);
  const [aiJobs, setAiJobs] = useState<any[]>([]);
  const [aiPredictions, setAiPredictions] = useState<any[]>([]);
  const [aiPredictionTotal, setAiPredictionTotal] = useState(0);
  const [activeJob, setActiveJob] = useState<any>(null);
  const [activePrediction, setActivePrediction] = useState<any>(null);
  const [selectedPreds, setSelectedPreds] = useState<Set<string>>(new Set());
  const [aiFilterStatus, setAiFilterStatus] = useState<string>("all");
  const [showActionModal, setShowActionModal] = useState(false);
  const [editingAction, setEditingAction] = useState<any>(null);
  const [actionForm, setActionForm] = useState({ name: "", prompt: "", label_names: "", model_type: "detection" });
  const [aiRunning, setAiRunning] = useState(false);
  const aiPollRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const [selectedActionId, setSelectedActionId] = useState<string>("");
  const [aiSampleLimit, setAiSampleLimit] = useState(50);
  const [aiSkipLabeled, setAiSkipLabeled] = useState(true);
  // AI labeling preview image (canvas fit/zoom now handled inside AnnotationEditor)
  const [aiPreviewUrl, setAiPreviewUrl] = useState<string | null>(null);
  const [aiPreviewImage] = useImage(aiPreviewUrl || "", "anonymous");

  // ── Preview data panel ─────────────────────────────────────────────────────
  const [previewItems, setPreviewItems] = useState(6);          // # samples to preview
  const [previewCols, setPreviewCols] = useState(3);            // grid columns
  const [showPreviewSettings, setShowPreviewSettings] = useState(false);
  const [previewSampleIds, setPreviewSampleIds] = useState<string[]>([]);
  const [previewUrls, setPreviewUrls] = useState<Record<string, string>>({});
  const [previewFilenames, setPreviewFilenames] = useState<Record<string, string>>({});
  // natural image dims (px) per sample — bounding boxes are stored in absolute px
  const [previewDims, setPreviewDims] = useState<Record<string, { w: number; h: number }>>({});
  const [previewLoading, setPreviewLoading] = useState(false);
  const [previewRunning, setPreviewRunning] = useState(false);
  // sample_id → prediction, populated after "Label preview data" completes
  const [previewPreds, setPreviewPreds] = useState<Record<string, any>>({});
  const previewPollRef = useRef<ReturnType<typeof setInterval> | null>(null);
  // dataset totals for the "Total results" readout under the filter select
  const [sampleCounts, setSampleCounts] = useState({ all: 0, unlabeled: 0 });
  const totalResults = aiSkipLabeled ? sampleCounts.unlabeled : sampleCounts.all;

  // Open directly in AI Labeling when arrived via ?view=ai-labeling
  // (e.g. the "Back" action from the Job records page).
  useEffect(() => {
    if (typeof window === "undefined") return;
    const params = new URLSearchParams(window.location.search);
    if (params.get("view") === "ai-labeling") setViewMode("ai-labeling");
  }, []);

  // Counterpart to the effect above: the sidebar's "Data labeling" link
  // points at this same bare route, so clicking it while already here is a
  // same-URL no-op for Next (no remount, no re-run of the effect above) and
  // would otherwise leave viewMode stuck on "ai-labeling". Sidebar.tsx
  // broadcasts this event on click so the reset happens regardless of
  // whether a route transition actually occurs.
  useEffect(() => {
    const goHome = () => setViewMode("dataset");
    window.addEventListener("data-labeling:go-home", goHome);
    return () => window.removeEventListener("data-labeling:go-home", goHome);
  }, []);

  // Load AI data when switching to AI labeling mode
  useEffect(() => {
    if (viewMode === "ai-labeling" && activeProject) {
      loadAiActions();
      loadAiJobs();
    }
  }, [viewMode, activeProject]);

  // Load predictions when selecting a job
  useEffect(() => {
    if (activeJob) loadAiPredictions(activeJob.id);
  }, [activeJob]);

  // Load AI preview image when prediction is selected
  useEffect(() => {
    if (activePrediction && activeProject) {
      setAiPreviewUrl(null);
      const sid = activePrediction.sample_id;
      samplesApi.download(sid).then(({ data }) => {
        setAiPreviewUrl(data.url);
      }).catch(() => setAiPreviewUrl(null));
    } else {
      setAiPreviewUrl(null);
    }
  }, [activePrediction?.id, activePrediction?._ts]);

  async function loadAiActions() {
    if (!activeProject) return;
    try {
      const { data } = await aiLabelingApi.listActions(activeProject.id);
      setAiActions(data);
    } catch { /* ignore */ }
  }

  async function loadAiJobs() {
    if (!activeProject) return;
    try {
      const { data } = await aiLabelingApi.listJobs(activeProject.id);
      setAiJobs(data);
      if (data.length > 0 && !activeJob) setActiveJob(data[0]);
      // Restore running job: if any job is still running on the backend,
      // resume the running indicator + polling so refresh/navigation doesn't lose state.
      const running = (data || []).find((j: any) => j.status === "running");
      if (running && !aiPollRef.current) {
        setActiveJob(running);
        setAiRunning(true);
        pollAiJob(running.id);
      }
    } catch { /* ignore */ }
  }

  function pollAiJob(jobId: string) {
    if (aiPollRef.current) clearInterval(aiPollRef.current);
    aiPollRef.current = setInterval(async () => {
      try {
        const { data: updated } = await aiLabelingApi.getJob(jobId);
        setActiveJob((prev: any) => (prev && prev.id === updated.id ? updated : prev));
        setAiJobs((prev: any[]) => prev.map(j => (j.id === updated.id ? updated : j)));
        if (updated.status === "completed" || updated.status === "failed") {
          if (aiPollRef.current) { clearInterval(aiPollRef.current); aiPollRef.current = null; }
          setAiRunning(false);
          loadAiJobs();
          loadAiPredictions(updated.id);
          if (updated.status === "completed") {
            toast.success("AI labeling complete!");
            mergeFetchSamples();
            notifySamplesChanged();
          } else {
            toast.error("AI labeling failed");
          }
        }
      } catch {
        if (aiPollRef.current) { clearInterval(aiPollRef.current); aiPollRef.current = null; }
        setAiRunning(false);
      }
    }, 2000);
  }

  // Clean up the polling interval on unmount
  useEffect(() => {
    return () => {
      if (aiPollRef.current) { clearInterval(aiPollRef.current); aiPollRef.current = null; }
    };
  }, []);

  async function loadAiPredictions(jobId: string) {
    try {
      const pageSize = 250;
      const allItems: any[] = [];
      let total = 0;
      let skip = 0;

      while (true) {
        const { data } = await aiLabelingApi.getPredictions(jobId, { skip, limit: pageSize });
        const items = data?.items || (Array.isArray(data) ? data : []);
        total = data?.total || items.length;
        allItems.push(...items);
        if (items.length < pageSize || allItems.length >= total) break;
        skip += pageSize;
      }

      setAiPredictions(allItems);
      setAiPredictionTotal(total || allItems.length);
      setSelectedPreds(new Set());
    } catch {
      setAiPredictions([]);
      setAiPredictionTotal(0);
    }
  }

  // ── Preview data ───────────────────────────────────────────────────────────
  // Pick the first N samples and prefetch their presigned image URLs.
  async function loadPreviewSamples() {
    if (!activeProject) return;
    setPreviewLoading(true);
    try {
      const res = await samplesApi.list(activeProject.id, { limit: previewItems });
      const items = res.data.items || res.data || [];
      const chosen = items.slice(0, previewItems);
      const ids = chosen.map((s: any) => s.id);
      setPreviewSampleIds(ids);
      setPreviewFilenames(Object.fromEntries(chosen.map((s: any) => [s.id, s.filename])));
      // Reset any stale suggestions/urls for a clean refresh.
      setPreviewPreds({});
      const urls: Record<string, string> = {};
      await Promise.all(ids.map(async (id: string) => {
        try {
          const { data } = await samplesApi.download(id);
          if (data?.url) urls[id] = data.url;
        } catch { /* skip unresolved image */ }
      }));
      setPreviewUrls(urls);
    } catch {
      toast.error("Failed to load preview samples");
    } finally {
      setPreviewLoading(false);
    }
  }

  function pollPreviewJob(jobId: string) {
    if (previewPollRef.current) clearInterval(previewPollRef.current);
    previewPollRef.current = setInterval(async () => {
      try {
        const { data: job } = await aiLabelingApi.getJob(jobId);
        if (job.status === "completed" || job.status === "failed") {
          if (previewPollRef.current) { clearInterval(previewPollRef.current); previewPollRef.current = null; }
          setPreviewRunning(false);
          if (job.status === "completed") {
            const { data } = await aiLabelingApi.getPredictions(jobId, { skip: 0, limit: 250 });
            const preds = data?.items || (Array.isArray(data) ? data : []);
            const bySample: Record<string, any> = {};
            for (const p of preds) bySample[p.sample_id] = p;
            setPreviewPreds(bySample);
            toast.success("Preview labeled");
          } else {
            toast.error("Preview labeling failed");
          }
        }
      } catch {
        if (previewPollRef.current) { clearInterval(previewPollRef.current); previewPollRef.current = null; }
        setPreviewRunning(false);
      }
    }, 2000);
  }

  // Run AI labeling over just the preview subset, then show suggestions per card.
  async function runPreviewLabeling() {
    if (!activeProject) return;
    if (!selectedActionId) { toast.error("Select an action first"); return; }
    if (previewSampleIds.length === 0) { toast.error("No preview samples — refresh first"); return; }
    if (previewRunning) return;
    setPreviewRunning(true);
    try {
      const { data: job } = await aiLabelingApi.run({
        action_id: selectedActionId,
        project_id: activeProject.id,
        sample_ids: previewSampleIds,
      });
      pollPreviewJob(job.id);
    } catch (err: any) {
      toast.error(err.response?.data?.detail || "Failed to label preview data");
      setPreviewRunning(false);
    }
  }

  // Local reset of preview suggestions (no dedicated backend endpoint).
  function clearPreviewChanges() {
    setPreviewPreds({});
    setShowPreviewSettings(false);
    toast.success("Preview changes cleared");
  }

  // Auto-load preview samples when entering AI labeling mode or changing item count.
  useEffect(() => {
    if (viewMode === "ai-labeling" && activeProject) loadPreviewSamples();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [viewMode, activeProject, previewItems]);

  // Compute dataset totals (all / unlabeled) for the "Total results" readout.
  // Backed by the same labeling-status-summary endpoint as the Labeling Queue
  // button count, so the two numbers can never disagree (Phase 5.D).
  useEffect(() => {
    if (viewMode !== "ai-labeling" || !activeProject) return;
    (async () => {
      try {
        const { data } = await samplesApi.labelingStatusSummary(activeProject.id);
        setSampleCounts({
          all: data?.total ?? 0,
          unlabeled: data?.unlabeled ?? 0,
        });
      } catch { setSampleCounts({ all: 0, unlabeled: 0 }); }
    })();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [viewMode, activeProject]);

  // Clean up preview polling on unmount.
  useEffect(() => {
    return () => { if (previewPollRef.current) clearInterval(previewPollRef.current); };
  }, []);

  async function saveAiAction() {
    if (!activeProject) return;

    if (actionForm.label_names && actionForm.prompt) {
      const promptFragments = actionForm.prompt.split(/,|\bor\b|\band\b/i).map(s => s.trim()).filter(s => s.length > 0 && s !== 'and' && s !== 'or');
      const labelNames = actionForm.label_names.split(',').map(s => s.trim()).filter(s => s.length > 0);
      if (labelNames.length > 0 && labelNames.length !== promptFragments.length) {
        toast.error(`Warning: Provided ${labelNames.length} label names but prompt has ${promptFragments.length} fragments. They must match.`);
        return;
      }
    }

    const payload = { ...actionForm, project_id: activeProject.id };
    try {
      if (editingAction) {
        await aiLabelingApi.updateAction(editingAction.id, payload);
        toast.success("Action updated");
      } else {
        await aiLabelingApi.createAction(payload);
        toast.success("Action created");
      }
      loadAiActions();
      setShowActionModal(false);
      setEditingAction(null);
      setActionForm({ name: "", prompt: "", label_names: "", model_type: "detection" });
    } catch { toast.error("Failed to save action"); }
  }

  async function deleteAiAction(id: string) {
    try {
      await aiLabelingApi.deleteAction(id);
      loadAiActions();
      toast.success("Deleted");
    } catch { toast.error("Delete failed"); }
  }

  async function runAiLabeling() {
    if (!activeProject || !selectedActionId) return;
    if (aiRunning || aiPollRef.current) {
      toast.error("A labeling job is already running");
      return;
    }
    // Double-check the backend in case state was lost on refresh.
    try {
      const { data: existing } = await aiLabelingApi.listJobs(activeProject.id);
      const running = (existing || []).find((j: any) => j.status === "running");
      if (running) {
        setAiJobs(existing);
        setActiveJob(running);
        setAiRunning(true);
        pollAiJob(running.id);
        toast.error("A labeling job is already running");
        return;
      }
    } catch { /* ignore — fall through and attempt to start */ }
    setAiRunning(true);
    try {
      // Label the full selected dataset: leave sample_ids empty so the backend
      // selects every matching sample (honouring skip_labeled) — no client-side cap.
      const { data: job } = await aiLabelingApi.run({
        action_id: selectedActionId,
        project_id: activeProject.id,
        sample_ids: [],
        skip_labeled: aiSkipLabeled,
      });
      toast.success(`AI labeling started for ${job.total_samples ?? "all"} samples`);
      setActiveJob(job);
      loadAiJobs();
      pollAiJob(job.id);
    } catch (err: any) {
      toast.error(err.response?.data?.detail || "Failed to start AI labeling");
      setAiRunning(false);
    }
  }

  async function applyAiPredictions(predIds: string[]) {
    if (!activeProject) return;
    try {
      await aiLabelingApi.apply({ prediction_ids: predIds, project_id: activeProject.id });
      toast.success(`Applied ${predIds.length} prediction(s)`);
      if (activeJob) loadAiPredictions(activeJob.id);
      mergeFetchSamples(); // Merge instead of replacing to avoid clearing state
      notifySamplesChanged();
    } catch { toast.error("Apply failed"); }
  }

  async function rejectAiPredictions(predIds: string[]) {
    try {
      await aiLabelingApi.reject({ prediction_ids: predIds });
      toast.success("Rejected");
      if (activeJob) loadAiPredictions(activeJob.id);
    } catch { toast.error("Reject failed"); }
  }

  function togglePred(id: string) {
    setSelectedPreds(prev => {
      const next = new Set(prev);
      next.has(id) ? next.delete(id) : next.add(id);
      return next;
    });
  }

  const filteredPredictions = (Array.isArray(aiPredictions) ? aiPredictions : []).filter((p: any) =>
    aiFilterStatus === "all" ? true : p.status === aiFilterStatus
  );

  // Pending predictions from the current job that can be committed to samples.
  const pendingPredictionIds = (Array.isArray(aiPredictions) ? aiPredictions : [])
    .filter((p: any) => p.status === "pending" && p.predicted_label)
    .map((p: any) => p.id);

  function confidenceColor(c: number) {
    if (c >= 0.8) return "text-green-400";
    if (c >= 0.5) return "text-yellow-400";
    return "text-red-400";
  }

  function normalizeBoxLabel(value: any) {
    const text = String(value ?? "").trim();
    if (!text) return "";
    const lowered = text.toLowerCase();
    if (lowered === "unlabeled" || lowered === "unlabelled" || lowered === "unknown") {
      return "";
    }
    return text;
  }

  /** Translate width/height aliases to the canonical w/h keys expected by the renderer. */
  function normalizeBoxGeometry(b: any): any {
    if (!b || typeof b !== "object") return b;
    const out = { ...b };
    if (out.w === undefined && out.width !== undefined) { out.w = out.width; delete out.width; }
    if (out.h === undefined && out.height !== undefined) { out.h = out.height; delete out.height; }
    return out;
  }

  // Split View & Component States
  const [activeSample, setActiveSample] = useState<any>(null);
  const [imageUrl, setImageUrl] = useState<string | null>(null);
  const [image] = useImage(imageUrl || "", 'anonymous');

  // Inline video preview (Post-processing tab). `videoStatus` drives the
  // panel: 'loading' from the moment a video is selected (so we never flash
  // empty → placeholder → player), 'ready' once the browser has resolved
  // metadata, 'error' whenever an actual failure signal fires — the presigned
  // URL fetch rejects, or the <video> element itself fires `error` (bad
  // format, missing/expired file, non-streamable source). Never gated on a
  // format allowlist.
  const [videoUrl, setVideoUrl] = useState<string | null>(null);
  const [videoStatus, setVideoStatus] = useState<"idle" | "loading" | "ready" | "error">("idle");

  useEffect(() => {
    const handleOutsideClick = () => { setMenuOpen(null); setMenuPos(null); };
    window.addEventListener("click", handleOutsideClick);
    return () => window.removeEventListener("click", handleOutsideClick);
  }, []);

  // Close the row menu on scroll / resize so it doesn't float away from its row.
  useEffect(() => {
    if (!menuOpen) return;
    const close = () => { setMenuOpen(null); setMenuPos(null); };
    window.addEventListener("scroll", close, true);
    window.addEventListener("resize", close);
    return () => {
      window.removeEventListener("scroll", close, true);
      window.removeEventListener("resize", close);
    };
  }, [menuOpen]);

  // Clear selections when switching tabs
  useEffect(() => {
    setSelectedIds(new Set());
    setActiveSample(null);
    setGridPage(1);
    setListPage(1);
  }, [filterType]);

  // Reset to first page when the label filter changes
  useEffect(() => { setGridPage(1); setListPage(1); }, [filterLabel]);

  // Close the view-settings popover when clicking outside it
  useEffect(() => {
    if (!settingsOpen) return;
    const close = (e: MouseEvent) => {
      if (settingsRef.current && !settingsRef.current.contains(e.target as Node)) {
        setSettingsOpen(false);
      }
    };
    window.addEventListener("mousedown", close);
    return () => window.removeEventListener("mousedown", close);
  }, [settingsOpen]);

  async function updateBoxLabelInSamples(sampleId: string, boxIdx: number, newLabel: string) {
    const sample = samples.find(x => x.id === sampleId);
    if (!sample) return;
    const boxes = (sample.extra_metadata?.boundingBoxes || []) as any[];
    const newBoxes = boxes.map((b, i) => i === boxIdx ? { ...b, label: newLabel } : b);
    const meta = { ...(sample.extra_metadata || {}), boundingBoxes: newBoxes };
    try {
      await samplesApi.update(sampleId, { extra_metadata: meta });
      setSamples(prev => prev.map(s => s.id === sampleId ? { ...s, extra_metadata: meta } : s));
      // Right-preview pane reads from activeSample, a separate state slice
      // from `samples`. Without this mirror, changing a box label via the
      // grid card's inline dropdown leaves the preview's AnnotationEditor
      // showing the old label until the user re-selects the sample. The
      // _ts bump invalidates AnnotationEditor's resetKey so it reseeds
      // from the new boundingBoxes. Mirrors persistSampleBoxes / saveSampleBoxes.
      setActiveSample((prev: any) =>
        prev?.id === sampleId
          ? { ...prev, extra_metadata: meta, _ts: Date.now() }
          : prev
      );
      // Same invalidation pathway loadSamples().finally uses — refresh the
      // summary cards (counts) and per-label health (warning triangle, modal)
      // without re-fetching the sample list (which would skeleton-flash the
      // grid the user just edited).
      loadCounts();
      loadDatasetHealth();
      notifySamplesChanged();
    } catch {
      toast.error("Failed to update label");
    }
  }

  // Reuses the same persistence pattern as `saveSampleBoxes` (the
  // AnnotationEditor onChange handler) but for any sample passed by id.
  // The grid cards call this on Save.
  async function persistSampleBoxes(sampleId: string, newBoxes: any[]) {
    const sample = samples.find(x => x.id === sampleId);
    if (!sample) return false;
    const normalized = newBoxes.map((box: any) => {
      const lbl = normalizeBoxLabel(box?.label);
      return {
        ...box,
        label: lbl || null,
        label_id: lbl ? box?.label_id : null,
      };
    });
    const meta = { ...(sample.extra_metadata || {}), boundingBoxes: normalized };
    try {
      await samplesApi.update(sampleId, { extra_metadata: meta });
      setSamples(prev => prev.map(s => s.id === sampleId ? { ...s, extra_metadata: meta } : s));
      // Mirror saveSampleBoxes: keep the right-side preview in sync. Bump _ts
      // so AnnotationEditor's resetKey changes and it re-seeds from the new
      // boundingBoxes instead of holding the stale ones.
      setActiveSample((prev: any) =>
        prev?.id === sampleId
          ? { ...prev, extra_metadata: meta, _ts: Date.now() }
          : prev
      );
      // Refresh summary cards + per-label health so the Train/Test ratio,
      // chart icon, and "no labels" gate react to the just-saved annotation.
      loadCounts();
      loadDatasetHealth();
      notifySamplesChanged();
      toast.success("Annotations saved", { duration: 1500 });
      return true;
    } catch {
      toast.error("Backend sync failed");
      return false;
    }
  }

  // Load image when activeSample changes
  useEffect(() => {
    if (!activeSample) {
      setImageUrl(null);
      return;
    }
    const ext = activeSample.filename.split(".").pop()?.toLowerCase() ?? "";
    if (["jpg", "jpeg", "png"].includes(ext)) {
      setImageUrl(null);
      fetchSampleUrl(activeSample.id).then(u => {
        setImageUrl(u);
      }).catch(err => {
        console.warn(`[dataset] preview URL fetch failed for sample ${activeSample.id}`, err);
        setImageUrl(null);
      });
    } else {
      setImageUrl(null);
    }
  }, [activeSample?.id, activeSample?._ts]);

  // Load video preview URL when activeSample changes. Reuses the same
  // presigned-download endpoint as the image preview — no dedicated video
  // route. `videoStatus` starts at 'loading' immediately so the panel shows
  // a spinner rather than the placeholder while the URL (and later, the
  // browser's metadata probe) resolves.
  useEffect(() => {
    if (!activeSample || !isVideoFile(activeSample.filename)) {
      setVideoUrl(null);
      setVideoStatus("idle");
      return;
    }
    setVideoUrl(null);
    setVideoStatus("loading");
    fetchSampleUrl(activeSample.id).then(u => {
      setVideoUrl(u);
    }).catch(err => {
      console.warn(`[dataset] video preview URL fetch failed for sample ${activeSample.id}`, err);
      setVideoStatus("error");
    });
  }, [activeSample?.id, activeSample?._ts]);

  // Prefetch the images adjacent to the current selection so stepping through
  // the dataset feels instant. Neighbours are taken from `samples` (the same
  // ordered list the grid/list render), and we warm both the next and previous
  // image samples — users step in either direction — skipping non-image files.
  // Each prefetch warms the presigned URL cache and decodes the bitmap into the
  // browser cache, so the on-select `fetchSampleUrl` + `useImage` both hit warm
  // caches and the "Loading preview…" state is skipped in the common case.
  useEffect(() => {
    if (!activeSample) return;
    const idx = samples.findIndex(s => s.id === activeSample.id);
    if (idx === -1) return;
    const isImage = (s: any) =>
      ["jpg", "jpeg", "png"].includes(
        (s?.filename?.split(".").pop() || "").toLowerCase()
      );
    for (const neighbor of [samples[idx + 1], samples[idx - 1]]) {
      if (neighbor && isImage(neighbor)) prefetchSample(neighbor.id);
    }
  }, [activeSample?.id, samples]);

  // Initial boxes passed to the editor for the active sample
  const activeSampleInitialBoxes = useMemo(() => {
    if (!activeSample) return [];
    return (activeSample.extra_metadata?.boundingBoxes || []).map((b: any) => ({
      ...normalizeBoxGeometry(b),
      id: b.id || uuidv4(),
    }));
  }, [activeSample?.id, activeSample?._ts]);

  // Initial boxes passed to the editor for the active AI prediction
  const activePredictionInitialBoxes = useMemo(() => {
    if (!activePrediction) return [];
    return (activePrediction.bounding_boxes || []).map((b: any) => ({
      ...normalizeBoxGeometry(b),
      label: normalizeBoxLabel(b.label || b.className),
      id: b.id || uuidv4(),
    }));
  }, [activePrediction?.id, activePrediction?._ts]);

  // Create a project label on demand (used by the editor's label modal)
  async function createProjectLabel(name: string): Promise<{ id: string; name: string } | null> {
    try {
      const { data } = await labelsApi.create({ project_id: activeProject?.id, name });
      setLabels(prev => [...prev, data]);
      return { id: data.id, name: data.name };
    } catch {
      toast.error("Failed to create new label");
      return null;
    }
  }

  // Persist edits to the AI prediction's bounding_boxes (used in AI Labeling)
  async function savePredictionBoxes(newBoxes: any[]) {
    if (!activePrediction) return;
    const normalized = newBoxes.map((b: any) => {
      const lbl = normalizeBoxLabel(b?.label);
      return { ...b, label: lbl || null, label_id: lbl ? b?.label_id : null };
    });
    // Keep the prediction's top-level label in sync with the boxes so the
    // table's "Predicted Label" column, Apply workflow, and DB all agree.
    // Use the most-frequent non-empty box label; fall back to the existing one.
    const counts = new Map<string, number>();
    for (const b of normalized) {
      const lbl = (b.label || "").toString();
      if (!lbl) continue;
      counts.set(lbl, (counts.get(lbl) || 0) + 1);
    }
    let dominantLabel: string | null = activePrediction.predicted_label ?? null;
    if (counts.size > 0) {
      dominantLabel = Array.from(counts.entries()).sort((a, b) => b[1] - a[1])[0][0];
    } else if (normalized.length === 0) {
      dominantLabel = null;
    }
    try {
      await aiLabelingApi.updatePrediction(activePrediction.id, {
        bounding_boxes: normalized,
        predicted_label: dominantLabel,
      });
      const updated = { ...activePrediction, bounding_boxes: normalized, predicted_label: dominantLabel };
      setActivePrediction(updated);
      setAiPredictions(prev => prev.map((p: any) =>
        p.id === updated.id ? { ...p, bounding_boxes: normalized, predicted_label: dominantLabel } : p
      ));
    } catch {
      toast.error("Failed to save prediction edits");
    }
  }



  // ── Quietly refresh samples for current tab (no loading flash) ────────────
  // Filter-aware (same params as loadSamples) so newly-labeled rows correctly
  // drop out of a label-filtered view (e.g. "unlabeled") instead of lingering
  // as stale rows. Each sample's _ts is bumped so grid cards / the
  // AnnotationEditor resetKey reseed their bounding boxes. Counts, labels, and
  // dataset health are refreshed too so the summary cards and label sidebar
  // stay in sync — without the setLoading flash loadSamples would cause.
  async function mergeFetchSamples() {
    if (!activeProject) return;
    try {
      const params: any = { sample_type: filterType, limit: SAMPLE_LIST_LIMIT };
      if (filterLabel === "__unlabeled__") params.unlabeled_only = true;
      else if (filterLabel) params.label_id = filterLabel;
      const { data } = await samplesApi.list(activeProject.id, params);
      const fetched = data?.items ?? (Array.isArray(data) ? data : []);
      const ts = Date.now();
      setSamples(fetched.map((s: any) => ({ ...s, _ts: ts })));
    } catch { /* ignore */ } finally {
      loadCounts();
      loadLabels();
      loadDatasetHealth();
    }
  }

  async function loadSamples() {
    if (!activeProject) return;
    setLoading(true);
    try {
      const params: any = { sample_type: filterType, limit: SAMPLE_LIST_LIMIT };
      if (filterLabel === "__unlabeled__") params.unlabeled_only = true;
      else if (filterLabel) params.label_id = filterLabel;
      const { data } = await samplesApi.list(activeProject.id, params);
      setSamples(data?.items ?? []);
    } catch {
      setSamples([]);
    } finally {
      setLoading(false);
      loadCounts(); // keep summary cards in sync after every load
      loadDatasetHealth(); // and the per-label health driving the modal/warning
    }
  }

  async function loadLabels() {
    if (!activeProject) return;
    try {
      const { data } = await labelsApi.list(activeProject.id);
      setLabels(data ?? []);
    } catch { /* ignore */ }
  }

  // Fetch per-split item counts for the summary cards (no split filter = all samples).
  async function loadCounts() {
    if (!activeProject) return;
    try {
      const { data } = await samplesApi.list(activeProject.id, { limit: 10000 });
      const all: any[] = data?.items ?? (Array.isArray(data) ? data : []);
      const c = { training: 0, testing: 0, postprocessing: 0 };
      const used = new Set<string>();
      const perLabel: Record<string, number> = {};
      const perLabelSplit: Record<string, { training: number; testing: number }> = {};
      for (const s of all) {
        const t: string = s.sample_type ?? "";
        if (t === "training") c.training++;
        else if (t === "testing") c.testing++;
        else if (t === "postprocessing") c.postprocessing++;

        if (s.label_id) used.add(`id:${s.label_id}`);
        if (s.label_name) used.add(`name:${String(s.label_name).toLowerCase()}`);
        const boxes = s?.extra_metadata?.boundingBoxes || [];
        for (const b of boxes) {
          if (b?.label_id) used.add(`id:${b.label_id}`);
          const lbl = normalizeBoxLabel(b?.label);
          if (lbl) used.add(`name:${lbl.toLowerCase()}`);
        }

        // Multi-label: a sample can carry several labels (its classification
        // label plus one per bounding box), so count it once for EVERY distinct
        // label it has. Per-label counts therefore do NOT sum to the total
        // sample count — that's expected here. Deduplicated case-insensitively,
        // keeping the first-seen casing as the display name.
        const sampleLabels = new Map<string, string>();
        const addLabel = (value: any) => {
          const lbl = normalizeBoxLabel(value);
          if (lbl && !sampleLabels.has(lbl.toLowerCase())) sampleLabels.set(lbl.toLowerCase(), lbl);
        };
        addLabel(s.label_name);
        for (const b of boxes) addLabel(b?.label);
        const keys = sampleLabels.size > 0 ? Array.from(sampleLabels.values()) : ["Unlabeled"];

        for (const key of keys) {
          perLabel[key] = (perLabel[key] ?? 0) + 1;

          if (t === "training" || t === "testing") {
            if (!perLabelSplit[key]) perLabelSplit[key] = { training: 0, testing: 0 };
            perLabelSplit[key][t]++;
          }
        }
      }
      setSplitCounts(c);
      setUsedLabelKeys(used);
      setLabelCounts(perLabel);
      setLabelSplit(perLabelSplit);
    } catch { /* ignore */ }
  }

  // Per-label + overall split health. Drives both the inline warning
  // triangle on the Train/Test stat card and the per-label colors inside
  // the Dataset train / test split ratio modal. Fetched once per project
  // change (and after any sample-mutation that triggers loadCounts).
  async function loadDatasetHealth() {
    if (!activeProject) return;
    try {
      const { data } = await projectsApi.datasetHealth(activeProject.id);
      setDatasetHealth(data);
    } catch {
      setDatasetHealth(null);
    }
  }

  // Mount & project change → load labels + counts once independently
  useEffect(() => {
    if (activeProject) {
      loadLabels();
      loadCounts();
      loadDatasetHealth();
    }
  }, [activeProject]);

  // Tab or label filter change → reload the sample list only
  useEffect(() => {
    if (activeProject) loadSamples();
  }, [activeProject, filterType, filterLabel]);

  // ── Helpers ───────────────────────────────────────────────────────────────
  async function deleteSample(id: string) {
    try {
      await samplesApi.delete(id);
      setSamples(s => s.filter(x => x.id !== id));
      // Clear the right preview if it was showing the deleted sample so the
      // pane doesn't render a ghost (image URL was already fetched for the
      // now-deleted id).
      setActiveSample((prev: any) => prev?.id === id ? null : prev);
      // Drop the deleted id from the multi-select Set so the bulk-action bar
      // count stays honest if the user had this row checked.
      setSelectedIds(prev => {
        if (!prev.has(id)) return prev;
        const next = new Set(prev);
        next.delete(id);
        return next;
      });
      // Same invalidation as the annotation save handlers — refresh summary
      // cards and per-label health, but skip loadSamples() to avoid the
      // skeleton flash on the grid the user just acted on.
      loadCounts();
      loadDatasetHealth();
      notifySamplesChanged();
      toast.success("Deleted");
    } catch { toast.error("Delete failed"); }
  }

  function applyRowLabelUpdate(sample: any, labelId: string) {
    const labelName = labels.find(l => l.id === labelId)?.name;
    // Clearing the label (empty labelId) also clears the drawn bounding boxes:
    // for image samples the boxes ARE the labels, so "no label" means "no
    // annotations". Relabeling the boxes to null (the old behavior) left them on
    // screen. assignLabel persists the empty set so this sticks after reload.
    const boxes = !labelId
      ? []
      : Array.isArray(sample?.extra_metadata?.boundingBoxes)
        ? sample.extra_metadata.boundingBoxes.map((box: any) => {
          const updated = normalizeBoxGeometry(box);
          return {
            ...updated,
            label: labelName ?? null,
            label_id: labelId,
          };
        })
        : null;

    return {
      ...sample,
      label_id: labelId || null,
      label_name: labelId ? labelName : null,
      extra_metadata: boxes
        ? { ...(sample.extra_metadata || {}), boundingBoxes: boxes }
        : sample.extra_metadata,
    };
  }

  async function assignLabel(sampleId: string, labelId: string) {
    try {
      await samplesApi.assignLabel(sampleId, labelId);
      // Clearing the label must also drop the bounding boxes from extra_metadata
      // so they don't reappear on reload. samplesApi.assignLabel only touches the
      // sample's label_id, never the stored boxes — persist an empty set here.
      //
      // The `boundingBoxes: []` written here means "the annotation was removed",
      // NOT "this image is background". Those are different assertions and the
      // backend keeps them apart with an explicit `is_background` key; nothing
      // infers background from an empty box array. Clearing the label also drops
      // any existing background marker, because the user has just said this
      // sample carries no annotation state at all — leaving the marker would
      // silently keep feeding it to training as a negative.
      if (!labelId) {
        const sample = samples.find(x => x.id === sampleId);
        if (sample?.extra_metadata?.boundingBoxes?.length || sample?.is_background) {
          const meta = { ...(sample?.extra_metadata || {}), boundingBoxes: [] };
          delete (meta as any).is_background;
          await samplesApi.update(sampleId, { extra_metadata: meta });
        }
      }
      setSamples(s => s.map(x => x.id === sampleId ? applyRowLabelUpdate(x, labelId) : x));
      // Bump _ts so the right-hand preview's AnnotationEditor resetKey changes
      // and it re-seeds from the updated boxes — otherwise clearing labels leaves
      // the old boxes on the preview until a manual refresh.
      setActiveSample((prev: any) =>
        prev?.id === sampleId ? { ...applyRowLabelUpdate(prev, labelId), _ts: Date.now() } : prev
      );
      await loadSamples();
      notifySamplesChanged();
    } catch { toast.error("Label update failed"); }
  }

  async function changeSplit(sampleId: string, newType: string) {
    try {
      await samplesApi.updateSplit(sampleId, newType);
      setSamples(prev => prev.filter(x => x.id !== sampleId));
      await loadSamples();
      notifySamplesChanged();
    } catch { toast.error("Split change failed"); }
  }

  async function renameSample(s: any) {
    const newName = window.prompt("Enter new file name:", s.filename);
    if (newName && newName !== s.filename) {
      try {
        await samplesApi.update(s.id, { filename: newName });
        setSamples(prev => prev.map(x => x.id === s.id ? { ...x, filename: newName } : x));
        toast.success("Renamed");
      } catch { toast.error("Rename failed"); }
    }
  }

  async function toggleDisable(s: any) {
    try {
      await samplesApi.update(s.id, { is_disabled: !s.is_disabled });
      setSamples(prev => prev.map(x => x.id === s.id ? { ...x, is_disabled: !s.is_disabled } : x));
    } catch { toast.error("Update failed"); }
  }

  async function downloadSample(id: string) {
    try {
      const { data } = await samplesApi.download(id);
      if (data.url) window.location.href = data.url;
    } catch { toast.error("Download failed"); }
  }

  // Persist annotation edits for the active sample (called from AnnotationEditor onChange)
  async function saveSampleBoxes(newBoxes: any[], primary?: { id?: string; name?: string } | null) {
    if (!activeSample) return;
    const patchLabelId = primary?.id;
    const patchLabelName = primary?.name;
    const normalizedBoxes = newBoxes.map((box: any) => {
      const normalizedLabel = normalizeBoxLabel(box?.label);
      return {
        ...box,
        label: normalizedLabel || null,
        label_id: normalizedLabel ? box?.label_id : null,
      };
    });
    const meta = { ...(activeSample.extra_metadata || {}), boundingBoxes: normalizedBoxes };
    try {
      if (patchLabelId && activeSample.label_id !== patchLabelId) {
        await samplesApi.assignLabel(activeSample.id, patchLabelId);
      }
      await samplesApi.update(activeSample.id, { extra_metadata: meta });
      const updated = {
        ...activeSample,
        label_id: patchLabelId || activeSample.label_id,
        label_name: patchLabelName || activeSample.label_name,
        extra_metadata: meta,
      };
      setActiveSample(updated);
      setSamples(prev => prev.map(s => s.id === activeSample.id ? updated : s));
      // Same as persistSampleBoxes / updateBoxLabelInSamples — sync the
      // summary cards and dataset-health off the existing helpers.
      loadCounts();
      loadDatasetHealth();
      notifySamplesChanged();
      toast.success("Annotations saved", { duration: 1500 });
    } catch {
      toast.error("Backend sync failed");
    }
  }

  // ── Bulk Actions ──────────────────────────────────────────────────────────
  function toggleRow(id: string) {
    const next = new Set(selectedIds);
    if (next.has(id)) next.delete(id);
    else next.add(id);
    setSelectedIds(next);
  }

  function toggleAll() {
    if (selectedIds.size === samples.length) setSelectedIds(new Set());
    else setSelectedIds(new Set(samples.map(s => s.id)));
  }

  async function handleBulkAction(action: string, value?: any) {
    if (!action) return;
    const ids = Array.from(selectedIds);
    if (ids.length === 0) return;

    try {
      let res: any;
      if (action === "delete") {
        if (!window.confirm(`Delete ${ids.length} samples?`)) return;
        res = await samplesApi.bulkDelete(ids);
        // Only locally drop the ids the server actually deleted. The bulk
        // endpoint may report partial failures in res.data.failed; those
        // samples still exist on the server and must stay in the grid so
        // the user sees the warning toast lines up with what's visible.
        // The failed entries can be ids or {id,...} objects depending on
        // the endpoint shape — handle both.
        const failedRaw: any[] = res.data?.failed ?? [];
        const failedIds = new Set<string>(
          failedRaw
            .map((f: any) => (typeof f === "string" ? f : f?.id))
            .filter(Boolean)
        );
        const deletedSet = new Set(ids.filter(i => !failedIds.has(i)));
        setSamples(prev => prev.filter(s => !deletedSet.has(s.id)));
        setActiveSample((prev: any) => prev && deletedSet.has(prev.id) ? null : prev);
        setSelectedIds(new Set());
        // Same invalidation pathway the save handlers use. No loadSamples()
        // here so the grid doesn't skeleton-flash after the local removal.
        loadCounts();
        loadDatasetHealth();
        notifySamplesChanged();
      } else {
        // metadata_add takes an object server-side. The "Add metadata" menu
        // item collects a "key:value" string from a prompt() and is parsed
        // here; callers that already have the object (e.g. the background
        // marker, which must stay a real JSON boolean, not the string "true")
        // pass it straight through.
        if (action === "metadata_add" && typeof value === "string") {
          const parts = value.split(":");
          if (parts.length < 2) return toast.error("Format must be key:value");
          value = { [parts[0].trim()]: parts.slice(1).join(":").trim() };
        }
        if (action === "metadata_add" && (value === null || typeof value !== "object")) return;
        res = await samplesApi.bulkUpdate(ids, action, value);
        // Bulk *update* paths (label change, split move, metadata) still
        // need the full sample-list refetch — labels/types may have moved
        // rows between tabs, and local mutation can't infer the new server
        // shape. Spec carve-out only applies to delete.
        setSelectedIds(new Set());
        await loadSamples();
        notifySamplesChanged();
      }

      // Read partial success/failures
      const failedCount = res.data?.failed?.length || 0;
      if (failedCount > 0) {
        toast(`Bulk action applied, but ${failedCount} item(s) failed.`, { icon: '⚠️' });
      } else {
        toast.success("Bulk action applied successfully");
      }
    } catch (err: any) {
      toast.error(err.response?.data?.detail || "Bulk action failed due to network or server error");
    }
  }

  // ── Guard ────────────────────────────────────────────────────────────────
  if (!activeProject) {
    return (
      <div className="flex items-center justify-center h-64">
        <p className="text-gray-500">Select a project from the sidebar to get started.</p>
      </div>
    );
  }

  const tabs: { key: SplitKey; label: string }[] = [
    { key: "training", label: "Training" },
    { key: "testing", label: "Test" },
    { key: "postprocessing", label: "Post-processing" },
  ];

  // Row/card actions kebab menu. Shared verbatim between the list-view rows and
  // the grid-view cards so both surfaces expose identical options + handlers.
  const renderSampleMenu = (s: any) => (
    <>
      <button
        onClick={(e) => {
          e.stopPropagation();
          if (menuOpen === s.id) {
            setMenuOpen(null);
            setMenuPos(null);
            return;
          }
          const btn = e.currentTarget as HTMLElement;
          const r = btn.getBoundingClientRect();
          const MENU_W = 224;   // matches w-56 (14rem)
          const MENU_H = 360;   // approx; used only for flip check
          const margin = 8;
          let left = r.right - MENU_W;
          if (left < margin) left = margin;
          if (left + MENU_W > window.innerWidth - margin) {
            left = window.innerWidth - MENU_W - margin;
          }
          let top = r.bottom + 4;
          if (top + MENU_H > window.innerHeight - margin) {
            // Not enough room below → flip above the button.
            top = Math.max(margin, r.top - MENU_H - 4);
          }
          setMenuPos({ top, left });
          setMenuOpen(s.id);
        }}
        className="row-action"
        aria-label="Row actions"
      >
        <MoreVertical size={14} />
      </button>

      {menuOpen === s.id && menuPos && createPortal(
        <div
          className="fixed w-56 bg-gray-900 border border-gray-700 rounded-md shadow-2xl z-[1000] py-1 text-left"
          style={{ top: menuPos.top, left: menuPos.left }}
          onClick={e => e.stopPropagation()}
        >
          <button className="w-full text-left px-4 py-2.5 text-xs text-gray-300 hover:bg-gray-800" onClick={() => { setMenuOpen(null); renameSample(s); }}>
            Rename
          </button>

          {(s.extra_metadata?.boundingBoxes || []).length > 0 && (
            <>
              <div className="px-4 py-2 hover:bg-gray-800">
                <span className="text-[10px] text-gray-500 uppercase tracking-wider mb-1.5 block">Edit labels</span>
                <select
                  className="bg-gray-800 border border-gray-600 rounded px-2 py-1.5 flex-1 text-xs text-gray-300 w-full"
                  value={s.label_id || ""}
                  onChange={e => { setMenuOpen(null); assignLabel(s.id, e.target.value); }}
                >
                  <option value="">Unlabeled...</option>
                  {labels
                    .filter(l =>
                      l.id === s.label_id ||
                      usedLabelKeys.has(`id:${l.id}`) ||
                      usedLabelKeys.has(`name:${String(l.name).toLowerCase()}`)
                    )
                    .map(l => <option key={l.id} value={l.id}>{l.name}</option>)}
                </select>
              </div>

              {s.label_id && (
                <button className="w-full text-left px-4 py-2.5 text-xs text-gray-300 hover:bg-gray-800" onClick={() => { setMenuOpen(null); assignLabel(s.id, ""); }}>
                  Clear labels
                </button>
              )}
            </>
          )}

          <div className="my-1 border-t border-gray-800" />

          {filterType !== 'testing' && <button className="w-full text-left px-4 py-2.5 text-xs text-gray-300 hover:bg-gray-800" onClick={() => { setMenuOpen(null); changeSplit(s.id, 'testing'); }}>Move to test set</button>}
          {filterType !== 'training' && <button className="w-full text-left px-4 py-2.5 text-xs text-gray-300 hover:bg-gray-800" onClick={() => { setMenuOpen(null); changeSplit(s.id, 'training'); }}>Move to training set</button>}
          {filterType !== 'postprocessing' && <button className="w-full text-left px-4 py-2.5 text-xs text-gray-300 hover:bg-gray-800" onClick={() => { setMenuOpen(null); changeSplit(s.id, 'postprocessing'); }}>Move to post-processing set</button>}

          <div className="my-1 border-t border-gray-800" />

          <button className="w-full text-left px-4 py-2.5 text-xs text-gray-300 hover:bg-gray-800" onClick={() => { setMenuOpen(null); toggleDisable(s); }}>
            {s.is_disabled ? "Enable" : "Disable"}
          </button>

          <button className="w-full text-left px-4 py-2.5 text-xs text-gray-300 hover:bg-gray-800" onClick={() => { setMenuOpen(null); downloadSample(s.id); }}>
            Download
          </button>

          <button className="w-full text-left px-4 py-2.5 text-xs text-red-500 hover:bg-red-900/30" onClick={() => { setMenuOpen(null); deleteSample(s.id); }}>
            Delete
          </button>
        </div>,
        document.body
      )}
    </>
  );

  // ─── Render ─────────────────────────────────────────────────────────────
  return (
    <div className="dataset-page data-acq-page w-full space-y-6">

      <SplitRatioModal
        open={showSplitModal}
        onClose={() => setShowSplitModal(false)}
        labels={datasetHealth?.labels ?? []}
        returnFocusRef={splitTriggerRef}
      />

      <DataDistributionModal
        open={showDistributionModal}
        onClose={() => setShowDistributionModal(false)}
        perLabel={labelSplit}
        colorForLabel={colorForLabel}
      />

      {/* Header — breadcrumb / title / search + actions */}
      <div className="flex items-start justify-between gap-6 flex-wrap">
        <div className="min-w-0 flex-1">
          <div className="data-acq-header-top">
            {viewMode === "ai-labeling" && (
              <>
                <button
                  type="button"
                  onClick={() => setViewMode("dataset")}
                  className="data-acq-back-btn"
                  aria-label="Back to Data Labeling"
                >
                  <ArrowLeft size={13} strokeWidth={2.25} />
                  <span>Back</span>
                </button>
                <span className="header-sep" aria-hidden="true" />
              </>
            )}
            <nav className="data-acq-breadcrumb" aria-label="Breadcrumb">
              <Home size={13} aria-hidden="true" />
              <ChevronRight size={12} aria-hidden="true" />
              {viewMode === "ai-labeling" ? (
                <>
                  <button
                    type="button"
                    onClick={() => setViewMode("dataset")}
                    className="crumb-link"
                  >
                    Data Labeling
                  </button>
                  <ChevronRight size={12} aria-hidden="true" />
                  <span className="crumb-current">AI Labeling</span>
                </>
              ) : (
                <span className="crumb-current">Data Labeling</span>
              )}
            </nav>
          </div>
          <h1 className="data-acq-title">
            {viewMode === "ai-labeling" ? "AI Labeling" : "Data Labeling"}
          </h1>
          <p className="data-acq-subtitle">
            {viewMode === "ai-labeling"
              ? "Use AI-powered actions to auto-label your dataset samples."
              : "Review, label and curate samples in your dataset."}
          </p>
        </div>

        <div className="flex items-center gap-3 flex-shrink-0">
          {viewMode !== "ai-labeling" && (
            <>
              <button
                onClick={() => router.push("/dashboard/data/dataset/synthetic")}
                className="data-acq-btn-primary"
              >
                <Wand2 size={14} /> Synthetic Data
              </button>
              <button
                onClick={() => router.push("/dashboard/data/dataset/queue")}
                className="data-acq-btn-primary"
                disabled={unlabeledCount === 0}
                title={unlabeledCount === 0
                  ? "No unlabeled images"
                  : `Label ${unlabeledCount} unlabeled image${unlabeledCount === 1 ? "" : "s"}`}
              >
                <ClipboardList size={14} />
                Labeling Queue{unlabeledCount > 0 ? ` (${unlabeledCount})` : ""}
              </button>
              <button
                onClick={() => setViewMode("ai-labeling")}
                className="data-acq-btn-primary"
              >
                <Sparkles size={14} /> AI Labeling
              </button>
            </>
          )}
          <button
            onClick={async () => {
              if (refreshing) return;
              setRefreshing(true);
              try {
                await loadSamples();
              } finally {
                setRefreshing(false);
              }
            }}
            disabled={refreshing}
            className="data-acq-btn-primary"
          >
            <RefreshCw size={14} className={refreshing ? "animate-spin" : ""} /> Refresh
          </button>
        </div>
      </div>
      {viewMode === "dataset" && (
        <>

          {/* ── Summary cards row ────────────────────────────────────────────── */}
          {(() => {
            const trainTestTotal = splitCounts.training + splitCounts.testing;
            const totalCount = trainTestTotal + splitCounts.postprocessing;
            const trainPct = trainTestTotal > 0 ? Math.round((splitCounts.training / trainTestTotal) * 100) : 0;
            const testPct = trainTestTotal > 0 ? 100 - trainPct : 0;
            const hasLabels = datasetHealth?.labels?.some(l => l.name !== "Unlabeled") ?? false;


            const totalSegments: PieSegment[] = Object.entries(labelCounts)
              .sort((a, b) => b[1] - a[1])
              .map(([label, value]) => ({
                key: label,
                label,
                value,
                color: colorForLabel(label),
              }));

            // Derive the active pie key (label name) from the current filterLabel
            // (a label ID, "__unlabeled__", or ""). Keeps the chart in sync with
            // the toolbar filter dropdown without any separate state.
            const activeLabelName = filterLabel === "__unlabeled__"
              ? "Unlabeled"
              : filterLabel
                ? labels.find(l => l.id === filterLabel)?.name ?? null
                : null;

            // Click handler: clicking a slice sets filterLabel to that label's ID
            // (same value the <select> dropdown uses). Clicking the active slice
            // clears the filter (toggle). Uses "__unlabeled__" for Unlabeled slices.
            const handleSliceClick = (sliceKey: string) => {
              if (sliceKey === "Unlabeled") {
                setFilterLabel(prev => prev === "__unlabeled__" ? "" : "__unlabeled__");
              } else {
                const labelId = labels.find(l => l.name === sliceKey)?.id ?? "";
                setFilterLabel(prev => prev === labelId ? "" : labelId);
              }
            };

            const splitSegments: PieSegment[] = [
              { key: "train", label: "Training", value: splitCounts.training, color: "#7c3aed" },
              { key: "test", label: "Test", value: splitCounts.testing, color: "#2563eb" },
            ].filter(s => s.value > 0);

            return (
              <div className="grid grid-cols-1 md:grid-cols-3 gap-5">

                {/* Data collected */}
                <div className="data-acq-stat-card data-acq-stat-card--total">
                  <div className="min-w-0">
                    <p className="stat-eyebrow">Data collected</p>
                    <p className="stat-value inline-flex items-center gap-2">
                      <span>
                        {totalCount.toLocaleString()}
                        <span className="stat-unit">Items</span>
                      </span>
                      {hasLabels && (
                        <button
                          type="button"
                          onClick={() => setShowDistributionModal(true)}
                          className="ds-split-trigger"
                          title="See data distribution charts for the training category"
                          aria-label="See data distribution charts for the training category"
                        >
                          <BarChart3 size={14} />
                        </button>
                      )}
                    </p>
                  </div>
                  {hasLabels && (
                    <div className="data-acq-stat-donut">
                      <PieChart
                        segments={totalSegments}
                        onSliceClick={handleSliceClick}
                        activeKey={activeLabelName}
                      />
                    </div>
                  )}
                </div>

                {/* Train / Test split */}
                <div className="data-acq-stat-card">
                  <div className="min-w-0">
                    <p className="stat-eyebrow">Train / Test Split</p>
                    {hasLabels ? (

                      <>
                        <p className="stat-value inline-flex items-center gap-2">
                          <button
                            type="button"
                            onClick={() => setShowSplitModal(true)}
                            className="ds-split-ratio-trigger"
                            aria-label="Open dataset train / test split ratio"
                          >
                            {trainPct}% <span className="stat-unit">/ {testPct}%</span>
                          </button>
                          <SplitHealthIconButton
                            healthy={datasetHealth?.overall.healthy ?? true}
                            onClick={() => setShowSplitModal(true)}
                            buttonRef={splitTriggerRef}
                          />
                        </p>
                        <div className="stat-meta">
                          <span className="inline-flex items-center gap-1.5"><span className="stat-dot train" /> Train</span>
                          <span className="inline-flex items-center gap-1.5"><span className="stat-dot test" /> Test</span>
                        </div>
                      </>
                    ) : (
                      <p className="stat-value">-</p>

                    )}
                  </div>
                  {hasLabels && (
                    <div className="data-acq-stat-donut">
                      <PieChart segments={splitSegments} />
                    </div>
                  )}
                </div>

                {/* Collect data */}
                <div className="data-acq-stat-card data-acq-stat-card--collect">
                  <div className="min-w-0">
                    <p className="stat-eyebrow">Collect Data</p>
                    <p className="stat-body">
                      To start building your dataset{" "}
                      <Link href="/dashboard/devices">Connect a device.</Link>
                    </p>
                  </div>
                  <div className="data-acq-stat-tile">
                    <Smartphone size={22} />
                  </div>
                </div>

              </div>
            );
          })()}

          {/* ── Dataset Split View Container ─────────────────────────────────────
          Height-capped to the visible viewport so the right panel doesn't
          turn into a downward-extending section. Both columns share this
          fixed height and scroll internally. */}
          <div className={`data-acq-split grid grid-cols-1 gap-5 items-stretch w-full ${layoutMode !== "detailed" ? "lg:grid-cols-[1.6fr_1fr]" : ""}`}>

            {/* Left: Table Section */}
            <div className="dataset-main-panel data-acq-table-card flex flex-col h-full min-h-0 min-w-0 overflow-hidden">

              {/* Tab bar with accurate per-split counts */}
              <div className="data-acq-tabs">
                {tabs.map(tab => (
                  <button
                    key={tab.key}
                    onClick={() => setFilterType(tab.key)}
                    className={`tab ${filterType === tab.key ? "is-active" : ""}`}
                  >
                    {tab.label}
                    <span className="tab-count">({splitCounts[tab.key].toLocaleString()})</span>
                  </button>
                ))}
                <div className="data-acq-tab-tools">
                  <Link
                    href="/dashboard/data"
                    className="tt-icon-btn"
                    aria-label="Go to Data acquisition"
                    title="Go to Data"
                  >
                    <Database size={16} />
                  </Link>
                  {(splitCounts.training + splitCounts.testing + splitCounts.postprocessing) > 0 && activeProject && (
                    <button
                      className="tt-icon-btn"
                      aria-label="Export data"
                      title="Export data"
                      disabled={exportBusy}
                      onClick={() => exportProject(activeProject.id, activeProject.name)}
                    >
                      {exportBusy
                        ? <Loader2 size={16} className="animate-spin" />
                        : <DownloadCloudIcon size={16} />}
                    </button>
                  )}
                  <div className="relative" ref={settingsRef}>
                    <button
                      className="tt-icon-btn"
                      aria-label="View settings"
                      title="View settings"
                      onClick={() => setSettingsOpen(o => !o)}
                    >
                      <span className="relative inline-flex items-center justify-center" style={{ width: 16, height: 16 }}>
                        <Settings size={16} />
                        <Eye
                          size={11}
                          className="absolute"
                          style={{
                            right: -3,
                            bottom: -3,
                            background: "var(--app-surface)",
                            borderRadius: 999,
                            padding: 1,
                          }}
                        />
                      </span>
                    </button>
                    {settingsOpen && (
                      <div
                        className="absolute right-0 top-full mt-2 w-72 z-50 rounded-xl shadow-xl p-1.5"
                        style={{
                          background: "var(--app-surface)",
                          border: "1px solid var(--app-border)",
                          boxShadow: "var(--app-shadow)",
                        }}
                        role="menu"
                      >
                        {effectiveLayout === "list" ? (
                          <>
                            <button
                              className="w-full flex items-center gap-2.5 px-3 py-2 text-sm rounded-lg hover:bg-[var(--app-surface-2)] transition-colors"
                              style={{ color: "var(--app-text)" }}
                              onClick={() => { switchLayoutMode("grid"); setSettingsOpen(false); }}
                            >
                              <Columns3 size={14} className="opacity-70" />
                              <span>Switch to grid view</span>
                            </button>
                            <div
                              className="w-full flex items-center justify-between gap-2 px-3 py-2 text-sm rounded-lg"
                              style={{ color: "var(--app-text)" }}
                            >
                              <span className="inline-flex items-center gap-2.5">
                                <Database size={14} className="opacity-70" />
                                Rows per page
                              </span>
                              <select
                                value={listRowsPerPage}
                                onChange={e => { setListRowsPerPage(Number(e.target.value)); setListPage(1); }}
                                className="text-xs rounded px-1.5 py-1"
                                style={{
                                  background: "var(--app-surface-2)",
                                  border: "1px solid var(--app-border)",
                                  color: "var(--app-text)",
                                }}
                              >
                                {[6, 12, 24, 48, 100].map(n => <option key={n} value={n}>{n}</option>)}
                              </select>
                            </div>
                          </>
                        ) : (
                          <>
                            <button
                              className="w-full flex items-center gap-2.5 px-3 py-2 text-sm rounded-lg hover:bg-[var(--app-surface-2)] transition-colors"
                              style={{ color: "var(--app-text)" }}
                              onClick={() => { switchLayoutMode("list"); setSettingsOpen(false); }}
                            >
                              <List size={14} className="opacity-70" />
                              <span>Switch to list view</span>
                            </button>
                            <label
                              className="w-full flex items-center justify-between gap-2 px-3 py-2 text-sm rounded-lg hover:bg-[var(--app-surface-2)] cursor-pointer transition-colors"
                              style={{ color: "var(--app-text)" }}
                            >
                              <span className="inline-flex items-center gap-2.5">
                                <ImageIcon size={14} className="opacity-70" />
                                Inline edit bounding box labels
                              </span>
                              <input
                                type="checkbox"
                                checked={inlineEditBoxLabels}
                                onChange={e => setInlineEditBoxLabels(e.target.checked)}
                                style={{ accentColor: "var(--brand-500, #8b5cf6)" }}
                              />
                            </label>
                            <div
                              className="w-full flex items-center justify-between gap-2 px-3 py-2 text-sm rounded-lg"
                              style={{ color: "var(--app-text)" }}
                            >
                              <span className="inline-flex items-center gap-2.5">
                                <Database size={14} className="opacity-70" />
                                Items per page
                              </span>
                              <select
                                value={itemsPerPage}
                                onChange={e => { setItemsPerPage(Number(e.target.value)); setGridPage(1); }}
                                className="text-xs rounded px-1.5 py-1"
                                style={{
                                  background: "var(--app-surface-2)",
                                  border: "1px solid var(--app-border)",
                                  color: "var(--app-text)",
                                }}
                              >
                                {[6, 12, 24, 48].map(n => <option key={n} value={n}>{n}</option>)}
                              </select>
                            </div>
                            <div
                              className="w-full flex items-center justify-between gap-2 px-3 py-2 text-sm rounded-lg"
                              style={{ color: "var(--app-text)" }}
                            >
                              <span className="inline-flex items-center gap-2.5">
                                <Columns3 size={14} className="opacity-70" />
                                Number of columns
                              </span>
                              <select
                                value={gridColumns}
                                onChange={e => setGridColumns(Number(e.target.value))}
                                className="text-xs rounded px-1.5 py-1"
                                style={{
                                  background: "var(--app-surface-2)",
                                  border: "1px solid var(--app-border)",
                                  color: "var(--app-text)",
                                }}
                              >
                                {[2, 3, 4, 5, 6].map(n => <option key={n} value={n}>{n}</option>)}
                              </select>
                            </div>
                          </>
                        )}
                      </div>
                    )}
                  </div>
                  <button
                    className="tt-icon-btn"
                    aria-label="Show detailed view"
                    title="Show detailed view"
                    onClick={() => switchLayoutMode(layoutMode === "detailed" ? lastNonDetailedMode : "detailed")}
                    style={layoutMode === "detailed" ? { color: "var(--app-text)", background: "var(--app-surface-2)" } : undefined}
                  >
                    <Maximize2 size={14} />
                  </button>
                  <select value={filterLabel} onChange={e => setFilterLabel(e.target.value)} aria-label="Filter by label">
                    <option value="">All labels</option>
                    {labels
                      .filter(l => usedLabelKeys.has(`id:${l.id}`) || usedLabelKeys.has(`name:${String(l.name).toLowerCase()}`))
                      .map(l => <option key={l.id} value={l.id}>{l.name}</option>)}
                    {(labelCounts["Unlabeled"] ?? 0) > 0 && (
                      <option value="__unlabeled__">Unlabeled</option>
                    )}
                  </select>
                  {(() => {
                    const orphans = labels.filter(
                      l =>
                        !usedLabelKeys.has(`id:${l.id}`) &&
                        !usedLabelKeys.has(`name:${String(l.name).toLowerCase()}`)
                    );
                    if (orphans.length === 0) return null;
                    return (
                      <button
                        className="btn-ghost text-xs py-1"
                        title={`Unused: ${orphans.map(o => o.name).join(", ")}`}
                        onClick={async () => {
                          if (!activeProject) return;
                          const ok = window.confirm(
                            `Delete ${orphans.length} unused label(s)?\n\n${orphans.map(o => `• ${o.name}`).join("\n")}`
                          );
                          if (!ok) return;
                          try {
                            await labelsApi.pruneOrphans(activeProject.id);
                            await loadLabels();
                            await loadCounts();
                          } catch (e) {
                            console.error("Failed to prune unused labels", e);
                          }
                        }}
                      >
                        Delete {orphans.length} unused
                      </button>
                    );
                  })()}
                </div>
              </div>

              {/* Content */}
              {loading ? (
                <div className="space-y-2">
                  {[1, 2, 3, 4].map(i => <div key={i} className="h-12 bg-gray-800 rounded animate-pulse" />)}
                </div>

              ) : filterType === "postprocessing" && samples.length === 0 ? (
                /* Post-processing empty state — Edge Impulse exact match */
                <div className="flex flex-col items-center justify-center py-20 gap-4">
                  <h3 className="text-xl font-bold text-gray-200">Add data</h3>
                  <p className="text-gray-400 text-sm text-center max-w-xs">
                    Start building your dataset by adding some data.
                  </p>
                  <Link href="/dashboard/data" className="btn-primary mt-2 flex items-center gap-2">
                    <span className="text-base font-bold leading-none">+</span> Add data
                  </Link>
                </div>

              ) : samples.length === 0 ? (
                <div className="text-center py-12">
                  <Upload size={32} className="mx-auto text-gray-700 mb-2" />
                  <p className="text-gray-500 text-sm">
                    No {filterType} samples yet —{" "}
                    <Link href="/dashboard/data" className="text-brand-400 hover:underline">upload some data</Link>
                  </p>
                </div>

              ) : (
                <div className="overflow-x-auto space-y-3">
                  {/* Bulk Actions Bar */}
                  {selectedIds.size > 0 && (
                    <div className="bg-brand-900/30 border border-brand-500/50 rounded-md p-2 flex items-center justify-between animate-fade-in">
                      <span className="text-sm font-medium text-brand-300 ml-2">
                        {selectedIds.size} selected
                      </span>
                      <div className="flex items-center gap-2">
                        <button className="btn-ghost text-xs py-1 hover:text-red-400" onClick={() => handleBulkAction("delete")}>
                          <Trash2 size={12} className="inline mr-1 mb-0.5" /> Delete
                        </button>

                        <select
                          className="bg-gray-800 border-gray-700 text-xs py-1 rounded"
                          onChange={e => { e.target.value && handleBulkAction("label", e.target.value); e.target.value = ""; }}
                        >
                          <option value="">Edit labels...</option>
                          {labels
                            .filter(l =>
                              usedLabelKeys.has(`id:${l.id}`) ||
                              usedLabelKeys.has(`name:${String(l.name).toLowerCase()}`)
                            )
                            .map(l => <option value={l.id} key={l.id}>{l.name}</option>)}
                        </select>
                        <button className="btn-ghost text-xs py-1" onClick={() => handleBulkAction("label", "")}>
                          Clear labels
                        </button>

                        <select
                          className="bg-gray-800 border-gray-700 text-xs py-1 rounded"
                          onChange={e => { e.target.value && handleBulkAction("split", e.target.value); e.target.value = ""; }}
                        >
                          <option value="">Move to...</option>
                          {filterType !== 'training' && <option value="training">Training</option>}
                          {filterType !== 'testing' && <option value="testing">Testing</option>}
                          {filterType !== 'postprocessing' && <option value="postprocessing">Post-processing</option>}
                        </select>

                        <button className="btn-ghost text-xs py-1 border-l border-gray-800 pl-3" onClick={() => handleBulkAction("enable")}>Enable</button>
                        <button className="btn-ghost text-xs py-1 border-r border-gray-800 pr-3" onClick={() => handleBulkAction("disable")}>Disable</button>

                        {/* Background ("negative") images: the only way to reach
                            this state from the studio. The annotation editor
                            persists on box mutation, so an image that is never
                            annotated fires no event and can never be marked
                            there — hence a bulk action rather than an editor
                            mode. Rides the existing generic metadata_add /
                            metadata_clear_key actions; no new endpoint. */}
                        <select
                          className="bg-gray-800 border-gray-700 text-xs py-1 rounded"
                          onChange={e => {
                            if (e.target.value === "mark") handleBulkAction("metadata_add", { is_background: true });
                            else if (e.target.value === "clear") handleBulkAction("metadata_clear_key", "is_background");
                            e.target.value = "";
                          }}
                        >
                          <option value="">Background...</option>
                          <option value="mark">Mark as background</option>
                          <option value="clear">Clear background</option>
                        </select>

                        <select
                          className="bg-gray-800 border-gray-700 text-xs py-1 rounded"
                          onChange={e => {
                            if (e.target.value) {
                              const val = ["metadata_add", "metadata_clear_key"].includes(e.target.value) ? window.prompt(e.target.value === "metadata_add" ? "Enter 'key:value':" : "Enter exact key to clear:") : null;
                              handleBulkAction(e.target.value, val);
                              e.target.value = "";
                            }
                          }}
                        >
                          <option value="">Metadata...</option>
                          <option value="metadata_add">Add metadata</option>
                          <option value="metadata_clear_key">Clear metadata by key</option>
                          <option value="metadata_clear_all">Clear all metadata</option>
                        </select>
                      </div>
                    </div>
                  )}

                  {effectiveLayout === "grid" ? (
                    <DetailedSampleGrid
                      samples={samples}
                      columns={gridColumns}
                      itemsPerPage={itemsPerPage}
                      page={gridPage}
                      setPage={setGridPage}
                      inlineEdit={inlineEditBoxLabels}
                      labels={labels}
                      usedLabelKeys={usedLabelKeys}
                      pillStyleForLabel={pillStyleForLabel}
                      normalizeBoxLabel={normalizeBoxLabel}
                      onSelectSample={(s) => setActiveSample({ ...s, _ts: Date.now() })}
                      onUpdateBoxLabel={updateBoxLabelInSamples}
                      onSaveBoxes={persistSampleBoxes}
                      onCreateLabel={createProjectLabel}
                      colorForLabel={colorForLabel}
                      renderRowMenu={renderSampleMenu}
                    />
                  ) : (
                    <div className="w-full flex-1 overflow-x-auto overflow-y-auto min-h-0">
                      <table className="w-full">
                        <thead className="sticky top-0 z-10">
                          <tr>
                            <th style={{ width: 40 }}>
                              <input
                                type="checkbox"
                                className="row-check"
                                checked={samples.length > 0 && selectedIds.size === samples.length}
                                onChange={toggleAll}
                              />
                            </th>
                            <th>Sample name</th>
                            <th>Labels</th>
                            <th>Added</th>
                            <th style={{ width: 60 }}></th>
                          </tr>
                        </thead>
                        <tbody>
                          {samples
                            .slice((listPage - 1) * listRowsPerPage, listPage * listRowsPerPage)
                            .map((s, rowIdx) => {
                              return (
                                <tr
                                  key={s.id}
                                  className={`cursor-pointer ${activeSample?.id === s.id ? "is-active" : ""}`}
                                  onClick={() => setActiveSample({ ...s, _ts: Date.now() })}
                                >
                                  <td onClick={e => e.stopPropagation()}>
                                    <input
                                      type="checkbox"
                                      className="row-check"
                                      checked={selectedIds.has(s.id)}
                                      onChange={() => toggleRow(s.id)}
                                    />
                                  </td>
                                  <td>
                                    <span className={`inline-flex items-center gap-2.5 min-w-0 ${s.is_disabled ? "opacity-50 line-through" : ""}`}>
                                      <span className="sample-icon">{fileIcon(s.filename)}</span>
                                      <span className="sample-name truncate" title={s.filename}>
                                        {s.filename.split("/").pop()?.split("\\").pop()}
                                      </span>
                                    </span>
                                  </td>
                                  <td>
                                    {(() => {
                                      const isImg = ["jpg", "jpeg", "png"].includes(s.filename.split('.').pop()?.toLowerCase() || "");
                                      if (isImg) {
                                        const boxes = s.extra_metadata?.boundingBoxes || [];
                                        if (boxes.length > 0) {
                                          const uniqueLabels = Array.from(
                                            new Set(boxes.map((b: any) => normalizeBoxLabel(b.label)).filter(Boolean))
                                          ) as string[];
                                          if (uniqueLabels.length === 0) {
                                            return <span className="data-acq-pill data-acq-pill--gray">Unlabeled</span>;
                                          }
                                          return (
                                            <div className="flex flex-wrap gap-1.5">
                                              {uniqueLabels.map(l => (
                                                <span key={l} className="data-acq-pill" style={pillStyleForLabel(l)}>{l}</span>
                                              ))}
                                            </div>
                                          );
                                        }
                                        return s.is_background
                                          ? <BackgroundBadge />
                                          : <span className="data-acq-pill data-acq-pill--gray">Unlabeled</span>;
                                      }
                                      return s.label_id ? (
                                        <span className="data-acq-pill" style={pillStyleForLabel(s.label_name || s.label_id)}>{s.label_name}</span>
                                      ) : (
                                        <span className="data-acq-pill data-acq-pill--gray">Unlabeled</span>
                                      );
                                    })()}
                                  </td>
                                  <td>
                                    {new Date(s.created_at).toLocaleDateString("en-US", { month: "short", day: "numeric", year: "numeric" })}
                                  </td>
                                  <td className="text-right relative">
                                    {renderSampleMenu(s)}
                                  </td>
                                </tr>
                              );
                            })}
                        </tbody>
                      </table>
                      {samples.length > listRowsPerPage && (
                        <div
                          className="flex items-center justify-between py-2 px-1 text-xs"
                          style={{ color: "var(--app-text-soft)" }}
                        >
                          <span>
                            Showing {(listPage - 1) * listRowsPerPage + 1}–{Math.min(listPage * listRowsPerPage, samples.length)} of {samples.length}
                          </span>
                          <div className="flex items-center gap-2">
                            <button
                              className="px-2.5 py-1 rounded-md border"
                              style={{
                                background: "var(--app-surface)",
                                borderColor: "var(--app-border)",
                                color: "var(--app-text)",
                                opacity: listPage === 1 ? 0.45 : 1,
                                cursor: listPage === 1 ? "not-allowed" : "pointer",
                              }}
                              onClick={() => listPage > 1 && setListPage(listPage - 1)}
                              disabled={listPage === 1}
                            >
                              Previous
                            </button>
                            <span style={{ color: "var(--app-text)" }}>
                              Page {listPage} / {Math.max(1, Math.ceil(samples.length / listRowsPerPage))}
                            </span>
                            <button
                              className="px-2.5 py-1 rounded-md border"
                              style={{
                                background: "var(--app-surface)",
                                borderColor: "var(--app-border)",
                                color: "var(--app-text)",
                                opacity: listPage >= Math.ceil(samples.length / listRowsPerPage) ? 0.45 : 1,
                                cursor: listPage >= Math.ceil(samples.length / listRowsPerPage) ? "not-allowed" : "pointer",
                              }}
                              onClick={() => listPage < Math.ceil(samples.length / listRowsPerPage) && setListPage(listPage + 1)}
                              disabled={listPage >= Math.ceil(samples.length / listRowsPerPage)}
                            >
                              Next
                            </button>
                          </div>
                        </div>
                      )}
                    </div>
                  )}
                </div>
              )}
            </div>

            {/* Right: Annotation Panel — single, deterministic render path so the
            content is always top-aligned and never shifts between selections.
            Height is capped to the split container so this panel never extends
            the page; it scrolls internally if its content overflows. Hidden in
            detailed-grid view, which spans the full row width. */}
            {layoutMode !== "detailed" && (
              <div className="data-acq-side-card flex flex-col h-full min-h-0 overflow-hidden">
                {(() => {
                  // No selection → empty state pinned to the top of the panel.
                  if (!activeSample) {
                    return (
                      <div className="data-acq-empty-inner">
                        <EmptyFolderIllustration />
                        <p className="ec-title">No data selected</p>
                        <p className="ec-sub">Select a sample from the list to view its details and preview.</p>
                      </div>
                    );
                  }

                  const ext = (activeSample.filename?.split(".").pop() || "").toLowerCase();
                  const isImageExt = ["jpg", "jpeg", "png"].includes(ext);

                  // Image selected but still downloading — render a full-size loader
                  // in the same top position so the panel never collapses.
                  if (isImageExt && !(imageUrl && image)) {
                    return (
                      <div className="flex-1 flex flex-col min-h-0 w-full">
                        <div className="shrink-0 flex items-center justify-between border-b border-gray-800 px-3 py-2 bg-gray-900">
                          <span className="text-sm font-semibold text-gray-200 truncate" title={activeSample.filename}>
                            {activeSample.filename.split("/").pop()?.split("\\").pop()}
                          </span>
                          <button className="text-gray-500 hover:text-gray-200" onClick={() => setActiveSample(null)}>
                            <XCircle size={18} strokeWidth={2} />
                          </button>
                        </div>
                        <div className="flex-1 min-h-0 flex flex-col items-center justify-center gap-3 text-gray-500">
                          <Loader2 size={28} className="animate-spin opacity-70" />
                          <p className="text-xs">Loading preview…</p>
                        </div>
                      </div>
                    );
                  }

                  // Image loaded — the Konva viewer below this IIFE will render it.
                  if (isImageExt && imageUrl && image) {
                    return null;
                  }

                  // Video sample header, shared across the loading / player /
                  // fallback states below so it never shifts position.
                  const isVideo = isVideoFile(activeSample.filename);
                  const videoHeader = (
                    <div className="shrink-0 flex items-center justify-between border-b border-gray-800 px-3 py-2 bg-gray-900">
                      <span className="text-sm font-semibold text-gray-200 truncate">{activeSample.filename}</span>
                      <button className="text-gray-500 hover:text-gray-200" onClick={() => setActiveSample(null)}>
                        <XCircle size={18} strokeWidth={2} />
                      </button>
                    </div>
                  );

                  // Original unsupported-preview placeholder — reused verbatim
                  // as the video fallback whenever playback isn't possible
                  // (presigned URL fetch failed, or the <video> element itself
                  // fired an `error` event: bad format, missing/expired file,
                  // non-streamable source).
                  const unsupportedPlaceholder = (
                    <div className="flex-1 min-h-0 flex flex-col items-center justify-center px-6 text-center text-gray-500">
                      {isVideo ? (
                        <>
                          <Video size={40} className="mb-3 text-violet-400/70" strokeWidth={1.6} />
                          <p className="text-sm text-gray-300 mb-1">Video samples can&apos;t be previewed here.</p>
                          <p className="text-xs text-gray-500 mb-4">
                            Render and inspect this video on the post-processing page.
                          </p>
                          <Link
                            href="/dashboard/impulse/post-processing"
                            className="data-acq-btn-primary"
                          >
                            Open Post-processing
                          </Link>

                        </>
                      ) : (
                        <>
                          <ImageIcon size={28} className="mb-3 opacity-20" />
                          <p className="text-xs">
                            Cannot preview <strong>.{activeSample.filename.split('.').pop()}</strong> files.
                          </p>
                        </>
                      )}
                    </div>
                  );

                  if (isVideo) {
                    if (videoStatus === "error") {
                      return (
                        <div className="flex-1 flex flex-col min-h-0 w-full">
                          {videoHeader}
                          {unsupportedPlaceholder}
                        </div>
                      );
                    }
                    // 'loading' (URL still resolving or browser still probing
                    // metadata) and 'ready' both mount the <video> element so
                    // its loadedmetadata/error events can fire; only the
                    // spinner overlay and Open Post-processing button toggle.
                    return (
                      <div className="flex-1 flex flex-col min-h-0 w-full">
                        {videoHeader}
                        <div className="flex-1 min-h-0 flex flex-col items-center justify-center px-4 py-4 gap-3 text-gray-500">
                          {videoStatus === "loading" && (
                            <div className="flex flex-col items-center gap-3">
                              <Loader2 size={28} className="animate-spin opacity-70" />
                              <p className="text-xs">Loading preview…</p>
                            </div>
                          )}
                          {videoUrl && (
                            <video
                              key={`video:${activeSample.id}`}
                              src={videoUrl}
                              controls
                              controlsList="nodownload noremoteplayback"
                              disablePictureInPicture
                              disableRemotePlayback
                              preload="metadata"
                              style={{
                                display: videoStatus === "ready" ? "block" : "none",
                                maxWidth: "100%",
                                maxHeight: "min(60vh, 420px)",
                                width: "100%",
                                borderRadius: 12,
                                background: "#000",
                              }}
                              onLoadedMetadata={() => setVideoStatus("ready")}
                              onError={() => setVideoStatus("error")}
                            />
                          )}
                          {videoStatus === "ready" && (
                            <div className="flex flex-col items-center gap-2.5 shrink-0">
                              <Link
                                href="/dashboard/impulse/post-processing"
                                className="data-acq-btn-primary"
                              >
                                Open Post-processing
                              </Link>
                              <p className="text-xs text-gray-500">
                                Run inference on this video and review the model&apos;s predictions.
                              </p>
                            </div>
                          )}
                        </div>
                      </div>
                    );
                  }

                  // Non-image, non-video sample — unsupported-preview fallback.
                  return (
                    <div className="flex-1 flex flex-col min-h-0 w-full">
                      {videoHeader}
                      {unsupportedPlaceholder}
                    </div>
                  );
                })()}

                {activeSample && imageUrl && image ? (
                  /* ── Image viewer (flex column: header / canvas / nothing else) ── */
                  <div className="flex-1 flex flex-col min-h-0 w-full">

                    {/* Header - fixed height */}
                    <div className="data-acq-editor-header shrink-0 flex items-center justify-between px-3.5 py-2.5">
                      <div className="flex flex-col min-w-0 pr-3">
                        <span className="file-name text-sm truncate" title={activeSample.filename}>
                          {activeSample.filename.split("/").pop()?.split("\\").pop()}
                        </span>
                        <span className="file-meta text-[10px]">
                          Added: {new Date(activeSample.created_at).toLocaleString()}
                        </span>
                      </div>
                      <button className="close-btn shrink-0" onClick={() => setActiveSample(null)} aria-label="Close preview">
                        <XCircle size={16} strokeWidth={2} />
                      </button>
                    </div>

                    {/* Annotation editor — shared with AI Labeling */}
                    <AnnotationEditor
                      image={image}
                      initialBoxes={activeSampleInitialBoxes}
                      resetKey={`sample:${activeSample?.id}:${activeSample?._ts ?? ""}`}
                      labels={labels}
                      onChange={(boxes, primary) => saveSampleBoxes(boxes, primary)}
                      onCreateLabel={createProjectLabel}
                      colorForLabel={colorForLabel}
                    />

                  </div>

                ) : null}

              </div>
            )}

          </div>

          {/* Labeling modal now lives inside <AnnotationEditor /> */}
        </>
      )}

      {/* ═══════════════════════════════════════════════════════════════════════ */}
      {/* ═══ AI LABELING VIEW ═══════════════════════════════════════════════ */}
      {/* ═══════════════════════════════════════════════════════════════════════ */}
      {viewMode === "ai-labeling" && (
        <>
          <div className="flex flex-col lg:flex-row items-stretch gap-4 lg:h-[calc(100vh-160px)] w-full min-h-0">

            {/* ── LEFT SIDEBAR: control cards ─────────────────────────────────── */}
            <aside className="w-full lg:w-[480px] lg:shrink-0 flex flex-col gap-4 lg:overflow-y-auto">

              {/* Action card */}
              <section className="dataset-main-panel card !p-0 overflow-hidden shrink-0">
                <div className="flex items-center gap-2 px-4 py-3 border-b border-gray-800">
                  <Cpu size={14} className="text-brand-400" />
                  <h3 className="text-sm font-semibold text-gray-200">Action</h3>
                </div>
                <div className="p-4 space-y-3">
                  <select
                    className="input text-sm py-2 w-full"
                    value={selectedActionId}
                    onChange={e => setSelectedActionId(e.target.value)}
                  >
                    <option value="">Select action...</option>
                    {aiActions.map((a: any) => (
                      <option key={a.id} value={a.id}>{a.name}</option>
                    ))}
                  </select>
                  <div className="flex items-center gap-2">
                    <button
                      onClick={() => { setEditingAction(null); setActionForm({ name: "", prompt: "", label_names: "", model_type: "detection" }); setShowActionModal(true); }}
                      className="btn-primary text-xs px-3 py-2 flex items-center justify-center gap-1 flex-1"
                      title="Create a new action"
                    >
                      <Plus size={12} /> New action
                    </button>
                    {selectedActionId && (() => {
                      const a = aiActions.find((x: any) => x.id === selectedActionId);
                      if (!a) return null;
                      return (
                        <>
                          <button
                            className="p-2 rounded-lg bg-gray-800 border border-gray-700 text-gray-400 hover:text-blue-400 shrink-0"
                            onClick={() => { setEditingAction(a); setActionForm({ name: a.name, prompt: a.prompt, label_names: a.label_names || "", model_type: a.model_type }); setShowActionModal(true); }}
                            title="Edit action"
                          >
                            <Edit size={13} />
                          </button>
                          <button
                            className="p-2 rounded-lg bg-gray-800 border border-gray-700 text-gray-400 hover:text-red-400 shrink-0"
                            onClick={() => deleteAiAction(a.id)}
                            title="Delete action"
                          >
                            <Trash2 size={13} />
                          </button>
                        </>
                      );
                    })()}
                  </div>
                  {aiActions.length === 0 && (
                    <p className="text-xs text-gray-500">No actions yet — create one to start AI labeling.</p>
                  )}
                </div>
              </section>

              {/* Select data to label card */}
              <section className="dataset-main-panel card !p-0 overflow-hidden shrink-0">
                <div className="px-4 py-3 border-b border-gray-800">
                  <h3 className="text-sm font-semibold text-gray-200">Select data to label</h3>
                </div>
                <div className="p-4 space-y-3">
                  <div>
                    <label className="block text-xs font-medium text-gray-400 mb-1.5">Run AI labeling on</label>
                    <select
                      className="input text-sm py-2 w-full"
                      value={aiSkipLabeled ? "unlabeled" : "all"}
                      onChange={e => setAiSkipLabeled(e.target.value === "unlabeled")}
                    >
                      <option value="all">All data</option>
                      <option value="unlabeled">Data without labels</option>
                    </select>
                  </div>
                  <div className="flex items-center justify-between rounded-lg bg-gray-900/50 border border-gray-800 px-3 py-2">
                    <span className="text-xs text-gray-400">Total results</span>
                    <span className="text-sm font-semibold text-gray-200">
                      {totalResults} <span className="text-xs font-normal text-gray-500">samples</span>
                    </span>
                  </div>
                </div>
              </section>

            </aside>

            {/* ── MAIN CONTENT: preview + predictions (or editor) ──────────────── */}
            <div className="flex-1 min-w-0 flex flex-col gap-4 min-h-0">

              {activePrediction ? (
                /* ── Prediction editor ── */
                <div className="dataset-preview-panel card !p-0 overflow-hidden flex flex-col flex-1 min-h-0 bg-gray-950">
                  {aiPreviewUrl && aiPreviewImage ? (
                    <div className="flex flex-col h-full w-full">
                      {/* Header */}
                      <div className="shrink-0 flex items-center justify-between border-b border-gray-800 px-3 py-2 bg-gray-900">
                        <div className="flex flex-col min-w-0 pr-3">
                          <span className="text-sm font-semibold text-gray-200 truncate" title={activePrediction.sample_filename}>
                            {activePrediction.sample_filename?.split("/").pop()?.split("\\").pop()}
                          </span>
                          <span className="text-[10px] text-gray-400 flex items-center gap-2">
                            AI Output — {activePrediction.predicted_label || "Unlabelled"} ({(activePrediction.confidence * 100).toFixed(0)}%)
                            {activePrediction.low_confidence && (
                              <span className="text-yellow-500 font-medium flex items-center gap-1">
                                <AlertTriangle size={10} /> Low Confidence
                              </span>
                            )}
                          </span>
                        </div>
                        <div className="flex items-center gap-1.5">
                          {activePrediction.status === "pending" && (
                            <>
                              <button
                                className="px-2.5 py-1 text-xs rounded-md bg-green-900/30 text-green-400 border border-green-500/40 hover:bg-green-800/40 transition-colors flex items-center gap-1"
                                onClick={async () => {
                                  await applyAiPredictions([activePrediction.id]);
                                  setActivePrediction({ ...activePrediction, status: "approved" });
                                }}
                              >
                                <Check size={12} /> Apply
                              </button>
                              <button
                                className="px-2.5 py-1 text-xs rounded-md bg-red-900/30 text-red-400 border border-red-500/40 hover:bg-red-800/40 transition-colors flex items-center gap-1"
                                onClick={async () => {
                                  await rejectAiPredictions([activePrediction.id]);
                                  setActivePrediction({ ...activePrediction, status: "rejected" });
                                }}
                              >
                                <X size={12} /> Reject
                              </button>
                            </>
                          )}
                          {activePrediction.status === "approved" && (
                            <span className="text-[10px] px-2 py-0.5 rounded-full bg-green-900/30 text-green-400 border border-green-500/40 font-medium">✓ Applied</span>
                          )}
                          {activePrediction.status === "rejected" && (
                            <span className="text-[10px] px-2 py-0.5 rounded-full bg-red-900/30 text-red-400 border border-red-500/40 font-medium">✗ Rejected</span>
                          )}
                          <button className="shrink-0 text-gray-500 hover:text-gray-200 transition-colors ml-1" onClick={() => setActivePrediction(null)}>
                            <XCircle size={18} strokeWidth={2} />
                          </button>
                        </div>
                      </div>

                      {/* Annotation editor — shared with Data Labeling */}
                      <AnnotationEditor
                        image={aiPreviewImage}
                        initialBoxes={activePredictionInitialBoxes}
                        resetKey={`pred:${activePrediction?.id}:${activePrediction?._ts ?? ""}`}
                        labels={labels}
                        onChange={(boxes) => savePredictionBoxes(boxes)}
                        onCreateLabel={createProjectLabel}
                        colorForLabel={colorForLabel}
                        readOnly={activePrediction?.status === "approved"}
                      />
                    </div>
                  ) : (
                    <div className="flex flex-col items-center justify-center h-full text-gray-500">
                      <Loader2 size={24} className="animate-spin opacity-30 mb-3" />
                      <p className="text-xs">Loading preview…</p>
                    </div>
                  )}
                </div>
              ) : (
                <>
                  {/* ── Preview data card ── */}
                  <section className="dataset-main-panel card !p-0 overflow-hidden flex flex-col min-h-0 lg:flex-1">
                    <div className="flex items-center gap-2 px-4 py-3 border-b border-gray-800">
                      <h3 className="text-sm font-semibold text-gray-200">Preview data</h3>
                      <button
                        onClick={loadPreviewSamples}
                        disabled={previewLoading}
                        className="p-1 rounded hover:bg-gray-800 text-gray-400 hover:text-gray-200 disabled:opacity-50"
                        title="Refresh preview data"
                      >
                        <RefreshCw size={13} className={previewLoading ? "animate-spin" : ""} />
                      </button>
                      <div className="ml-auto flex items-center gap-2">
                        <Link
                          href={`/dashboard/data/dataset/job-records${activeJob?.id ? `?job=${activeJob.id}` : ""}`}
                          className="btn-ghost text-xs px-3 py-1.5 flex items-center gap-2"
                          title="View AI labeling job records"
                        >
                          <ClipboardList size={13} /> Job records
                        </Link>
                        <button
                          className="btn-primary text-xs px-3 py-1.5 flex items-center gap-2 disabled:opacity-50"
                          disabled={!selectedActionId || previewRunning || previewSampleIds.length === 0}
                          onClick={runPreviewLabeling}
                        >
                          {previewRunning ? (<><Loader2 size={13} className="animate-spin" /> Labeling...</>) : (<><Zap size={13} /> Label preview data</>)}
                        </button>
                        <div className="relative">
                          <button
                            onClick={() => setShowPreviewSettings(v => !v)}
                            className="p-1.5 rounded-lg bg-gray-800 border border-gray-700 text-gray-400 hover:text-gray-200"
                            title="Preview settings"
                          >
                            <Settings size={14} />
                          </button>
                          {showPreviewSettings && (
                            <>
                              <div className="fixed inset-0 z-10" onClick={() => setShowPreviewSettings(false)} />
                              <div className="absolute right-0 mt-1 w-64 z-20 card !p-0 bg-gray-900 border border-gray-700 rounded-lg shadow-xl overflow-hidden">
                                <div className="flex items-center justify-between px-3 py-2.5 border-b border-gray-800">
                                  <span className="text-xs text-gray-300">Items to preview</span>
                                  <input
                                    type="number" min={1} max={100}
                                    className="input w-16 text-xs py-1 text-right"
                                    value={previewItems}
                                    onChange={e => setPreviewItems(Math.max(1, parseInt(e.target.value) || 1))}
                                  />
                                </div>
                                <div className="flex items-center justify-between px-3 py-2.5 border-b border-gray-800">
                                  <span className="text-xs text-gray-300">Number of columns</span>
                                  <input
                                    type="number" min={1} max={6}
                                    className="input w-16 text-xs py-1 text-right"
                                    value={previewCols}
                                    onChange={e => setPreviewCols(Math.min(6, Math.max(1, parseInt(e.target.value) || 1)))}
                                  />
                                </div>
                                <button
                                  onClick={clearPreviewChanges}
                                  className="w-full flex items-center gap-2 px-3 py-2.5 text-xs text-gray-300 hover:bg-gray-800 text-left"
                                >
                                  <Trash2 size={13} /> Clear preview changes
                                </button>
                              </div>
                            </>
                          )}
                        </div>
                      </div>
                    </div>

                    <div className="p-4 overflow-y-auto min-h-0">
                      {previewSampleIds.length === 0 ? (
                        <p className="text-gray-500 text-xs py-4 text-center">
                          {previewLoading ? "Loading preview data…" : "No samples to preview"}
                        </p>
                      ) : (
                        <div className="grid gap-3" style={{ gridTemplateColumns: `repeat(${previewCols}, minmax(0, 1fr))` }}>
                          {previewSampleIds.map((sid) => {
                            const pred = previewPreds[sid];
                            const name = (previewFilenames[sid] || sid).split("/").pop()?.split("\\").pop() || sid.slice(0, 8);
                            return (
                              <div key={sid} className="rounded-lg border border-gray-800 bg-gray-900/40 overflow-hidden">
                                <div className="px-2.5 py-1.5 text-[11px] font-mono text-gray-400 truncate">{name}</div>
                                <div className="relative bg-gray-950 min-h-[80px]">
                                  {previewUrls[sid] ? (
                                    // eslint-disable-next-line @next/next/no-img-element
                                    <img
                                      src={previewUrls[sid]}
                                      alt={name}
                                      className="block w-full h-auto"
                                      onLoad={e => {
                                        const t = e.currentTarget;
                                        setPreviewDims(d => ({ ...d, [sid]: { w: t.naturalWidth, h: t.naturalHeight } }));
                                      }}
                                    />
                                  ) : (
                                    <div className="w-full h-32 flex items-center justify-center text-gray-700 text-xs">No image</div>
                                  )}
                                  {previewUrls[sid] && previewDims[sid] && (pred?.bounding_boxes || []).map((b: any, i: number) => (
                                    <div
                                      key={i}
                                      className="absolute border-2 border-green-500"
                                      style={{
                                        left: `${((b.x ?? 0) / previewDims[sid].w) * 100}%`,
                                        top: `${((b.y ?? 0) / previewDims[sid].h) * 100}%`,
                                        width: `${((b.w ?? b.width ?? 0) / previewDims[sid].w) * 100}%`,
                                        height: `${((b.h ?? b.height ?? 0) / previewDims[sid].h) * 100}%`,
                                      }}
                                    >
                                      {(pred?.predicted_label || b.label) && (
                                        <span className="absolute -top-0.5 left-0 -translate-y-full bg-green-500 text-white text-[9px] px-1 rounded-t">
                                          {pred?.predicted_label || b.label}
                                        </span>
                                      )}
                                    </div>
                                  ))}
                                </div>
                                <div className="px-2.5 py-2">
                                  <p className="text-[11px] font-semibold text-gray-300 mb-1">Suggested labels</p>
                                  {previewRunning ? (
                                    <span className="text-[11px] text-gray-500 flex items-center gap-1">
                                      <Loader2 size={11} className="animate-spin" /> Labeling…
                                    </span>
                                  ) : pred ? (
                                    <div className="flex items-center gap-1.5">
                                      <span className="badge-blue text-[11px]">{pred.predicted_label || "Unlabelled"}</span>
                                      {typeof pred.confidence === "number" && (
                                        <span className="text-[10px] text-gray-500 font-mono">{(pred.confidence * 100).toFixed(0)}%</span>
                                      )}
                                      <div className="ml-auto flex items-center gap-1">
                                        {pred.status === "approved" ? (
                                          <span className="text-[10px] px-1.5 py-0.5 rounded-full bg-green-900/30 text-green-400 border border-green-500/40">✓ Applied</span>
                                        ) : pred.status === "rejected" ? (
                                          <span className="text-[10px] px-1.5 py-0.5 rounded-full bg-red-900/30 text-red-400 border border-red-500/40">✗ Rejected</span>
                                        ) : (
                                          <>
                                            <button
                                              className="p-1 rounded hover:bg-green-900/30 text-gray-500 hover:text-green-400"
                                              title="Approve label"
                                              onClick={async () => { await applyAiPredictions([pred.id]); setPreviewPreds(prev => ({ ...prev, [sid]: { ...prev[sid], status: "approved" } })); }}
                                            >
                                              <Check size={13} />
                                            </button>
                                            <button
                                              className="p-1 rounded hover:bg-red-900/30 text-gray-500 hover:text-red-400"
                                              title="Reject label"
                                              onClick={async () => { await rejectAiPredictions([pred.id]); setPreviewPreds(prev => ({ ...prev, [sid]: { ...prev[sid], status: "rejected" } })); }}
                                            >
                                              <X size={13} />
                                            </button>
                                          </>
                                        )}
                                      </div>
                                    </div>
                                  ) : (
                                    <span className="text-[11px] text-gray-500">Run &apos;label preview data&apos; first.</span>
                                  )}
                                </div>
                              </div>
                            );
                          })}
                        </div>
                      )}
                    </div>
                  </section>
                </>
              )}
            </div>
          </div>

          {/* ── Bottom action bar: run over the whole dataset + commit results ── */}
          <div className="mt-4 flex flex-col sm:flex-row items-stretch sm:items-center sm:justify-end gap-2 border-t border-gray-800 pt-4">
            {!aiRunning && pendingPredictionIds.length > 0 && (
              <button
                onClick={() => applyAiPredictions(pendingPredictionIds)}
                className="px-4 py-2 text-sm rounded-lg bg-green-900/30 text-green-400 border border-green-500/40 hover:bg-green-800/40 transition-colors flex items-center justify-center gap-2 font-medium"
              >
                <Check size={15} /> Approve label ({pendingPredictionIds.length})
              </button>
            )}
            <button
              onClick={runAiLabeling}
              disabled={!selectedActionId || aiRunning}
              className="btn-primary text-sm px-4 py-2 flex items-center justify-center gap-2 disabled:opacity-50"
              title={!selectedActionId ? "Select an action first" : "Run AI labeling on the whole dataset"}
            >
              {aiRunning
                ? (<><Loader2 size={15} className="animate-spin" /> Labeling all data…</>)
                : (<><Zap size={15} /> Label all data</>)}
            </button>
          </div>
        </>
      )}

      {/* Action Modal */}
      {showActionModal && (
        <div className="fixed inset-0 bg-black/70 flex items-center justify-center z-50 p-4">
          <div className="bg-gray-900 border border-gray-700 rounded-xl w-full max-w-md overflow-hidden shadow-2xl animate-fade-in">
            <div className="px-5 py-4 border-b border-gray-800 flex justify-between items-center">
              <h3 className="font-bold text-gray-100">{editingAction ? "Edit Action" : "New Labeling Action"}</h3>
              <button className="text-gray-500 hover:text-gray-200 transition-colors" onClick={() => { setShowActionModal(false); setEditingAction(null); }}>
                <XCircle size={16} />
              </button>
            </div>
            <div className="p-5 space-y-4">
              <div>
                <label className="block text-xs text-gray-400 mb-1">Name</label>
                <input className="input w-full" placeholder="e.g. Detect defects" value={actionForm.name} onChange={e => setActionForm({ ...actionForm, name: e.target.value })} />
              </div>
              <div>
                <label className="block text-xs text-gray-400 mb-1">Prompt</label>
                <textarea className="input w-full h-24 resize-none" placeholder="e.g. Classify as cat, dog, or bird" value={actionForm.prompt} onChange={e => setActionForm({ ...actionForm, prompt: e.target.value })} />
              </div>
              <div>
                <label className="block text-xs text-gray-400 mb-1">Label Names (comma-separated, must match prompt order)</label>
                <input className="input w-full" placeholder="e.g. Cat, Dog" value={actionForm.label_names} onChange={e => setActionForm({ ...actionForm, label_names: e.target.value })} />
              </div>
              <div>
                <label className="block text-xs text-gray-400 mb-1">Model Type</label>
                <select className="input w-full" value={actionForm.model_type} onChange={e => setActionForm({ ...actionForm, model_type: e.target.value })}>
                  <option value="detection">Detection (OWL-ViT)</option>
                  <option value="classification">Classification</option>
                </select>
              </div>
            </div>
            <div className="px-5 py-4 bg-gray-950/50 border-t border-gray-800 flex justify-end gap-3">
              <button onClick={() => { setShowActionModal(false); setEditingAction(null); }} className="btn-ghost">Cancel</button>
              <button onClick={saveAiAction} className="btn-primary" disabled={!actionForm.name.trim() || !actionForm.prompt.trim()}>Save</button>
            </div>
          </div>
        </div>
      )}

    </div>
  );
}
