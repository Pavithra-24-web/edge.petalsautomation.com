"use client";
import { useEffect, useRef, useState } from "react";
import { MoreVertical, Check } from "lucide-react";
import type {
  LiveClassificationDetection,
  LiveClassificationGroundTruthBox,
  LiveClassificationRunResult,
  LiveClassificationSampleOption,
} from "@/types/live-classification";
import {
  colorForLabel,
  isImageSample,
  resultMode,
} from "@/lib/live-classification-helpers";

interface Props {
  result: LiveClassificationRunResult;
  sampleMeta: LiveClassificationSampleOption | null;
  imageUrl: string | null;
}

type ViewMode = "side" | "overlay";
type Visibility = "all" | "none";

export default function LiveClassificationResultPanel({
  result,
  sampleMeta,
  imageUrl,
}: Props) {
  const mode = resultMode(result);
  const showImage = isImageSample(sampleMeta?.sensor_type ?? null);

  const [viewMode, setViewMode] = useState<ViewMode>("side");
  const [menuOpen, setMenuOpen] = useState(false);
  const [expandedView, setExpandedView] = useState(false);
  const [groundTruthVis, setGroundTruthVis] = useState<Visibility>("all");
  const [predictionsVis, setPredictionsVis] = useState<Visibility>("all");
  const [showLabels, setShowLabels] = useState(true);
  const menuRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!menuOpen) return;
    const onClick = (e: MouseEvent) => {
      if (menuRef.current && !menuRef.current.contains(e.target as Node)) {
        setMenuOpen(false);
      }
    };
    window.addEventListener("mousedown", onClick);
    return () => window.removeEventListener("mousedown", onClick);
  }, [menuOpen]);

  // ── Annotation sources ────────────────────────────────────────────────
  // Ground truth = annotations authored in Data Labeling (boxes for detection
  // samples, label-only for classification samples).
  // Predictions  = the model's output for this sample.
  const gtBoxes: LiveClassificationGroundTruthBox[] = result.ground_truth_boxes ?? [];
  const predDetections: LiveClassificationDetection[] =
    mode === "detection" ? (result.detections ?? []) : [];
  const predClassificationLabel = mode === "classification" ? result.label : undefined;

  // ── Visibility filters from the bottom card ───────────────────────────
  const gtVisible = groundTruthVis === "all";
  const predVisible = predictionsVis === "all";

  // ── Counts for "Show all (N)" / "Hide all (N)" ────────────────────────
  // Ground truth count: boxes if present, otherwise 1 when a GT label exists
  // (classification sample with a folder label).
  const gtCount = gtBoxes.length || (result.ground_truth_label ? 1 : 0);
  const predCount = mode === "detection"
    ? predDetections.length
    : (predClassificationLabel ? 1 : 0);

  // ── Frame builders ────────────────────────────────────────────────────
  // Raw data = image + GT annotations.
  const rawFrame = (
    <ImageWithOverlays
      imageUrl={imageUrl}
      sensorType={sampleMeta?.sensor_type ?? null}
      showLabels={showLabels}
      groundTruthBoxes={gtVisible ? gtBoxes : []}
      groundTruthLabel={gtVisible ? (result.ground_truth_label ?? null) : null}
    />
  );

  // Classification result = image + prediction annotations.
  const predFrame = (
    <ImageWithOverlays
      imageUrl={imageUrl}
      sensorType={sampleMeta?.sensor_type ?? null}
      showLabels={showLabels}
      predictionDetections={predVisible ? predDetections : []}
      predictionLabel={predVisible ? predClassificationLabel : undefined}
    />
  );

  // Overlay = one image with both GT and prediction annotations layered.
  // GT renders dashed, predictions render solid, so they're distinguishable
  // even when boxes overlap.
  const overlayFrame = (
    <ImageWithOverlays
      imageUrl={imageUrl}
      sensorType={sampleMeta?.sensor_type ?? null}
      showLabels={showLabels}
      groundTruthBoxes={gtVisible ? gtBoxes : []}
      groundTruthLabel={gtVisible ? (result.ground_truth_label ?? null) : null}
      predictionDetections={predVisible ? predDetections : []}
      predictionLabel={predVisible ? predClassificationLabel : undefined}
    />
  );

  return (
    <div className="pe-lc-result-stack">
      {/* ── Top card: Raw data / Classification result ───────────────── */}
      <div className="pe-lc-rd-card">
        <div className="pe-lc-rd-head">
          <div className="min-w-0">
            <p className="pe-lc-rd-overline">RAW DATA / CLASSIFICATION RESULT</p>
            <p className="pe-lc-rd-id" title={result.sample_id}>
              {shortSampleId(result.sample_id, result.sample_name)}
            </p>
          </div>

          <div className="pe-lc-rd-head-actions">
            <select
              className="pe-lc-rd-view-select"
              value={viewMode}
              onChange={e => setViewMode(e.target.value as ViewMode)}
              aria-label="View mode"
            >
              <option value="side">Side by side</option>
              <option value="overlay">Overlay</option>
            </select>

            <div className="pe-lc-rd-menu-wrap" ref={menuRef}>
              <button
                type="button"
                className="pe-lc-rd-menu-btn"
                onClick={() => setMenuOpen(v => !v)}
                aria-label="More settings"
                aria-haspopup="menu"
                aria-expanded={menuOpen}
              >
                <MoreVertical size={16} />
              </button>
              {menuOpen && (
                <div className="pe-lc-rd-menu" role="menu">
                  <p className="pe-lc-rd-menu-head">SETTINGS</p>
                  <label className="pe-lc-rd-menu-row">
                    <span>Show expanded view</span>
                    <input
                      type="checkbox"
                      checked={expandedView}
                      onChange={e => setExpandedView(e.target.checked)}
                    />
                  </label>
                </div>
              )}
            </div>
          </div>
        </div>

        <div className={`pe-lc-rd-body ${expandedView ? "is-expanded" : ""}`}>
          {!showImage ? (
            <NonImagePreview sampleMeta={sampleMeta} />
          ) : viewMode === "side" ? (
            <div className="pe-lc-rd-side">
              <div className="pe-lc-rd-frame">
                <p className="pe-lc-rd-frame-label">Raw data</p>
                {rawFrame}
              </div>
              <div className="pe-lc-rd-frame">
                <p className="pe-lc-rd-frame-label">Classification result</p>
                {predFrame}
              </div>
            </div>
          ) : (
            <div className="pe-lc-rd-overlay">
              <div className="pe-lc-rd-frame">
                <p className="pe-lc-rd-frame-label">
                  Overlay <span className="pe-lc-rd-frame-legend">(dashed = ground truth, solid = prediction)</span>
                </p>
                {overlayFrame}
              </div>
            </div>
          )}
        </div>

      </div>

      {/* ── Bottom card: View controls ───────────────────────────────── */}
      <div className="pe-lc-rd-view-card">
        <h4 className="pe-lc-rd-view-title">View</h4>
        <div className="pe-lc-rd-view-grid">
          <label className="pe-lc-rd-view-field">
            <span>Ground truth</span>
            <select
              value={groundTruthVis}
              onChange={e => setGroundTruthVis(e.target.value as Visibility)}
            >
              <option value="all">Show all ({gtCount})</option>
              <option value="none">Hide all ({gtCount})</option>
            </select>
          </label>
          <label className="pe-lc-rd-view-field">
            <span>Predictions</span>
            <select
              value={predictionsVis}
              onChange={e => setPredictionsVis(e.target.value as Visibility)}
            >
              <option value="all">Show all ({predCount})</option>
              <option value="none">Hide all ({predCount})</option>
            </select>
          </label>
        </div>
        <label className="pe-lc-rd-show-labels">
          <input
            type="checkbox"
            checked={showLabels}
            onChange={e => setShowLabels(e.target.checked)}
          />
          <span>Show labels</span>
          {showLabels && <Check size={12} className="pe-lc-rd-show-labels-check" aria-hidden="true" />}
        </label>
      </div>
    </div>
  );
}

// ── Helpers ─────────────────────────────────────────────────────────────────

function shortSampleId(sampleId: string, fallback: string) {
  // Render the leading 10 chars of the sample id when it looks like a uuid /
  // hash so the header reads compact like the reference. Falls back to the
  // sample name when the id is missing or non-hex (e.g. uploaded filenames).
  const id = (sampleId || "").replace(/-/g, "");
  if (/^[a-f0-9]{16,}$/i.test(id)) return id.slice(0, 10);
  return fallback || sampleId || "—";
}

function NonImagePreview({
  sampleMeta,
}: {
  sampleMeta: LiveClassificationSampleOption | null;
}) {
  return (
    <div className="pe-lc-rd-nonimage">
      <p className="pe-lc-rd-nonimage-row">
        <span>Sensor type</span>
        <span>{sampleMeta?.sensor_type ?? "Unknown"}</span>
      </p>
      {sampleMeta?.duration_ms != null && (
        <p className="pe-lc-rd-nonimage-row">
          <span>Duration</span>
          <span>{sampleMeta.duration_ms} ms</span>
        </p>
      )}
      <p className="pe-lc-rd-nonimage-hint">
        Visual preview is available for image samples only.
      </p>
    </div>
  );
}

interface ImageOverlayProps {
  imageUrl: string | null;
  sensorType: string | null;
  showLabels: boolean;
  // Ground-truth annotations
  groundTruthBoxes?: LiveClassificationGroundTruthBox[];
  groundTruthLabel?: string | null;
  // Prediction annotations
  predictionDetections?: LiveClassificationDetection[];
  predictionLabel?: string;
}

function ImageWithOverlays({
  imageUrl,
  sensorType,
  showLabels,
  groundTruthBoxes,
  groundTruthLabel,
  predictionDetections,
  predictionLabel,
}: ImageOverlayProps) {
  const [imgError, setImgError] = useState(false);
  // GT boxes come from the backend in pixel coordinates; we normalize against
  // the natural dimensions of the loaded image so the SVG (viewBox 0..1) lays
  // them on top of the rendered img at any display size.
  const [naturalDims, setNaturalDims] = useState<{ w: number; h: number } | null>(null);

  if (!imageUrl || imgError) {
    return (
      <div className="pe-lc-rd-empty">
        <p>No visual preview available</p>
        {sensorType && <p className="pe-lc-rd-empty-sub">Sensor: {sensorType}</p>}
      </div>
    );
  }

  const gtBoxes = groundTruthBoxes ?? [];
  const predBoxes = predictionDetections ?? [];
  const hasGtBoxes = gtBoxes.length > 0 && naturalDims != null;
  const hasPredBoxes = predBoxes.length > 0;
  const hasAnyBoxes = hasGtBoxes || hasPredBoxes;

  // Classification chips render only when the corresponding mode has no
  // boxes; otherwise the box's own label chip carries the info.
  const showGtChip = !hasAnyBoxes && showLabels && !!groundTruthLabel;
  const showPredChip = !hasAnyBoxes && showLabels && !!predictionLabel;

  return (
    <div className="pe-lc-rd-image">
      <img
        src={imageUrl}
        alt="Sample"
        onError={() => setImgError(true)}
        onLoad={e => {
          const img = e.currentTarget;
          if (img.naturalWidth > 0 && img.naturalHeight > 0) {
            setNaturalDims({ w: img.naturalWidth, h: img.naturalHeight });
          }
        }}
      />

      {showPredChip && (
        <span
          className="pe-lc-rd-chip pe-lc-rd-chip--top"
          style={{ background: colorForLabel(predictionLabel!) }}
        >
          {predictionLabel}
        </span>
      )}
      {showGtChip && (
        <span
          className="pe-lc-rd-chip pe-lc-rd-chip--gt"
          style={{ background: colorForLabel(groundTruthLabel!) }}
        >
          GT: {groundTruthLabel}
        </span>
      )}

      {hasAnyBoxes && (
        <svg
          className="pe-lc-rd-svg"
          viewBox="0 0 1 1"
          preserveAspectRatio="none"
          aria-hidden="true"
        >
          {/* Ground-truth boxes — dashed stroke so they read as
              "annotation" rather than "prediction" when both modes are
              shown in overlay view. */}
          {hasGtBoxes && gtBoxes.map((b, i) => {
            const color = colorForLabel(b.label);
            const x = b.x / naturalDims!.w;
            const y = b.y / naturalDims!.h;
            const w = b.w / naturalDims!.w;
            const h = b.h / naturalDims!.h;
            return (
              <g key={`gt-${i}`}>
                <rect
                  x={x} y={y} width={w} height={h}
                  fill="none"
                  stroke={color}
                  strokeWidth="0.008"
                  strokeDasharray="0.022 0.012"
                />
                {showLabels && (
                  <>
                    <rect
                      x={x}
                      y={Math.max(0, y - 0.045)}
                      width={Math.min(0.4, Math.max(0.12, b.label.length * 0.018))}
                      height={0.045}
                      fill={color}
                    />
                    <text
                      x={x + 0.008}
                      y={Math.max(0.034, y - 0.012)}
                      fontSize="0.032"
                      fill="white"
                      fontWeight="600"
                    >
                      {b.label}
                    </text>
                  </>
                )}
              </g>
            );
          })}

          {/* Prediction boxes — solid stroke. */}
          {predBoxes.map((det, i) => {
            const color = colorForLabel(det.label);
            const { x1, y1, x2, y2 } = det.bbox;
            const w = x2 - x1;
            const h = y2 - y1;
            return (
              <g key={`pred-${i}`}>
                <rect
                  x={x1}
                  y={y1}
                  width={w}
                  height={h}
                  fill="none"
                  stroke={color}
                  strokeWidth="0.008"
                />
                {showLabels && (
                  <>
                    <rect
                      x={x1}
                      y={Math.max(0, y1 - 0.045)}
                      width={Math.min(0.4, Math.max(0.12, det.label.length * 0.018))}
                      height={0.045}
                      fill={color}
                    />
                    <text
                      x={x1 + 0.008}
                      y={Math.max(0.034, y1 - 0.012)}
                      fontSize="0.032"
                      fill="white"
                      fontWeight="600"
                    >
                      {det.label}
                    </text>
                  </>
                )}
              </g>
            );
          })}
        </svg>
      )}
    </div>
  );
}
