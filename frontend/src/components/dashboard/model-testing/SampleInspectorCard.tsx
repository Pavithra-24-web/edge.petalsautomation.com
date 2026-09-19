"use client";

import { useEffect, useState } from "react";
import { ImageOff, MousePointerClick, ScanLine } from "lucide-react";
import { samplesApi } from "@/utils/api";
import type {
  DetStats,
  GtBox,
  PredBox,
  TestResultStatus,
  TestSampleRow,
} from "@/types/model-testing";

type BoxView = "gt" | "pred" | "both";

const PALETTE = [
  "#22c55e", "#3b82f6", "#f59e0b", "#ef4444",
  "#8b5cf6", "#ec4899", "#14b8a6", "#f97316",
];

function labelColor(label: string): string {
  let h = 0;
  for (let i = 0; i < label.length; i++) {
    h = ((h << 5) - h + label.charCodeAt(i)) & 0x7fffffff;
  }
  return PALETTE[h % PALETTE.length];
}

function predToPixel(b: PredBox, W: number, H: number) {
  return { x: b.x * W, y: b.y * H, w: (b.x2 - b.x) * W, h: (b.y2 - b.y) * H };
}

function gtToPixel(b: GtBox, W: number, H: number) {
  const maxVal = Math.max(Math.abs(b.x), Math.abs(b.y), Math.abs(b.w), Math.abs(b.h));
  if (maxVal <= 1.0) {
    return { x: b.x * W, y: b.y * H, w: b.w * W, h: b.h * H };
  }
  return { x: b.x, y: b.y, w: b.w, h: b.h };
}

function gtLabel(b: GtBox): string {
  return (b.label || b.label_id || "").toString().trim();
}

interface LabelTagProps {
  bx: number; by: number; bw: number;
  text: string;
  fill: string; textFill: string;
}

function LabelTag({ bx, by, bw, text, fill, textFill }: LabelTagProps) {
  const charW = 6.5;
  const tagW = Math.min(bw, text.length * charW + 8);
  const tagH = 15;
  const tagY = Math.max(0, by - tagH);
  return (
    <g>
      <rect x={bx} y={tagY} width={tagW} height={tagH} fill={fill} opacity={0.92} rx={2} />
      <text
        x={bx + 3} y={tagY + tagH - 4}
        fontSize={10} fill={textFill}
        fontFamily="ui-monospace,monospace" fontWeight="700"
      >
        {text}
      </text>
    </g>
  );
}

interface ViewerProps {
  sampleId: string | null;
  predBoxes: PredBox[];
  gtBoxes: GtBox[];
  view: BoxView;
  onChangeView: (v: BoxView) => void;
}

function DetectionViewer({ sampleId, predBoxes, gtBoxes, view, onChangeView }: ViewerProps) {
  const [url, setUrl] = useState<string | null>(null);
  const [urlError, setUrlError] = useState(false);
  const [imgError, setImgError] = useState(false);
  const [naturalSize, setNaturalSize] = useState<{ w: number; h: number } | null>(null);

  useEffect(() => {
    setUrl(null);
    setUrlError(false);
    setImgError(false);
    setNaturalSize(null);
    if (!sampleId) return;

    let cancelled = false;
    samplesApi
      .download(sampleId)
      .then((res) => { if (!cancelled) setUrl((res as any)?.data?.url ?? null); })
      .catch(() => { if (!cancelled) setUrlError(true); });
    return () => { cancelled = true; };
  }, [sampleId]);

  const hasGt = gtBoxes.length > 0;
  const hasPred = predBoxes.length > 0;
  const showGt = (view === "gt" || view === "both") && hasGt;
  const showPred = (view === "pred" || view === "both") && hasPred;
  const { w: W = 1, h: H = 1 } = naturalSize ?? {};

  // Note: the toggle row is now rendered by the parent SampleInspectorCard so it
  // always sits in the header area regardless of viewer state.
  void onChangeView;

  if (!sampleId || urlError) {
    return (
      <div className="pe-mt-image flex flex-col items-center justify-center h-48 gap-2">
        <ImageOff size={22} className="text-slate-400" />
        <span className="text-xs text-slate-300">No preview</span>
      </div>
    );
  }

  if (!url) {
    return <div className="pe-mt-image h-48 animate-pulse" />;
  }

  return (
    <div className="pe-mt-image">
      <div className="relative overflow-hidden">
        <img
          src={url}
          alt="Sample"
          className="w-full h-auto block"
          onLoad={(e) => {
            const t = e.currentTarget;
            setNaturalSize({ w: t.naturalWidth, h: t.naturalHeight });
          }}
          onError={() => setImgError(true)}
        />

        {imgError && (
          <div className="absolute inset-0 flex flex-col items-center justify-center gap-1.5 bg-slate-900">
            <ImageOff size={22} className="text-slate-400" />
            <span className="text-xs text-slate-300">Not an image</span>
          </div>
        )}

        {naturalSize && !imgError && (
          <svg
            className="absolute inset-0 w-full h-full pointer-events-none"
            viewBox={`0 0 ${W} ${H}`}
            preserveAspectRatio="none"
          >
            {showGt && gtBoxes.map((box, i) => {
              const { x, y, w, h } = gtToPixel(box, W, H);
              const lbl = gtLabel(box);
              return (
                <g key={`gt-${i}`}>
                  <rect
                    x={x} y={y} width={w} height={h}
                    fill="none"
                    stroke="white" strokeWidth={2} strokeDasharray="8 4"
                    vectorEffect="non-scaling-stroke"
                  />
                  {lbl && (
                    <LabelTag
                      bx={x} by={y} bw={w}
                      text={lbl}
                      fill="rgba(0,0,0,0.65)" textFill="white"
                    />
                  )}
                </g>
              );
            })}
            {showPred && predBoxes.map((box, i) => {
              const color = labelColor(box.label);
              const { x, y, w, h } = predToPixel(box, W, H);
              const conf = `${box.label} ${Math.round(box.score * 100)}%`;
              return (
                <g key={`pred-${i}`}>
                  <rect
                    x={x} y={y} width={w} height={h}
                    fill={color} fillOpacity={0.08}
                    stroke={color} strokeWidth={2.5}
                    vectorEffect="non-scaling-stroke"
                  />
                  <LabelTag
                    bx={x} by={y} bw={w}
                    text={conf}
                    fill={color} textFill="white"
                  />
                </g>
              );
            })}
          </svg>
        )}
      </div>
    </div>
  );
}

function StatusBadge({ status, detStats }: { status: TestResultStatus; detStats?: DetStats | null }) {
  const isDetPassWithFP =
    status === "uncertain" && detStats != null && (detStats.fn ?? 1) === 0;
  const display = isDetPassWithFP ? "pass" : status;
  const label =
    display === "pass" ? "Pass" :
    display === "fail" ? "Fail" :
    display === "uncertain" ? "Mixed" : "Pending";
  const cls =
    display === "pass" ? "pe-mt-chip-pass" :
    display === "fail" ? "pe-mt-chip-fail" :
    display === "uncertain" ? "pe-mt-chip-mixed" : "pe-mt-chip-pending";
  return <span className={cls}>{label}</span>;
}

function DetStatsStrip({ stats }: { stats: DetStats }) {
  const { tp, fp, fn } = stats;
  if (tp === null && fp === null && fn === null) return null;
  const hasExtraFP = fp !== null && fp > 0;
  const hasMissedFN = fn !== null && fn > 0;
  return (
    <div className="pe-mt-detbar">
      <span className="pe-mt-detbar-chip pe-mt-detbar-chip--tp" title="True Positives — expected objects found">
        TP {tp ?? "—"}
      </span>
      <span className="pe-mt-detbar-chip pe-mt-detbar-chip--fp" title="False Positives — extra predictions not in ground truth">
        FP {fp ?? "—"}
      </span>
      <span className="pe-mt-detbar-chip pe-mt-detbar-chip--fn" title="False Negatives — expected objects missed">
        FN {fn ?? "—"}
      </span>
      {hasExtraFP && !hasMissedFN && (
        <p className="pe-mt-detbar-note">Correct object detected, but extra predictions were also made.</p>
      )}
      {hasExtraFP && hasMissedFN && (
        <p className="pe-mt-detbar-note">Some expected objects were missed and extra false predictions also exist.</p>
      )}
      {!hasExtraFP && hasMissedFN && (
        <p className="pe-mt-detbar-note">Expected object was not detected.</p>
      )}
    </div>
  );
}

function MetaRow({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="pe-mt-meta-row">
      <span className="pe-mt-meta-label">{label}</span>
      <span className="pe-mt-meta-value">{children}</span>
    </div>
  );
}

function EmptyState() {
  return (
    <div className="flex flex-col items-center justify-center py-14 px-6 text-center gap-3">
      <div className="w-12 h-12 rounded-full flex items-center justify-center
                      bg-violet-100 text-violet-600 dark:bg-violet-500/10 dark:text-violet-300">
        <MousePointerClick size={18} />
      </div>
      <div>
        <p className="text-sm font-semibold text-[color:var(--app-text)]">No sample selected</p>
        <p className="text-xs text-[color:var(--app-text-muted)] mt-0.5">
          Click any row in the table to inspect it
        </p>
      </div>
    </div>
  );
}

interface SampleInspectorCardProps {
  sample: TestSampleRow | null;
}

export default function SampleInspectorCard({ sample }: SampleInspectorCardProps) {
  const score = sample?.precision_score ?? sample?.f1_score ?? null;
  const predBoxes = sample?.predicted_boxes ?? [];
  const gtBoxes = sample?.gt_boxes ?? [];
  const detStats = sample?.det_stats ?? null;
  const scoreLabel = detStats ? "Match score" : "Score";

  const [view, setView] = useState<BoxView>("both");

  // Reset view when switching samples so disabled toggles never get stuck
  useEffect(() => { setView("both"); }, [sample?.id]);

  const hasGt = gtBoxes.length > 0;
  const hasPred = predBoxes.length > 0;
  const showToggle = !!sample && (hasGt || hasPred);

  return (
    <div className="pe-mt-card">
      <div className="pe-mt-inspector-head">
        <div className="pe-mt-inspector-title">
          <div className="pe-mt-card-icon" aria-hidden="true">
            <ScanLine size={18} strokeWidth={2.2} />
          </div>
          <h2 className="pe-mt-card-title">Sample inspector</h2>
        </div>
        {sample && (
          <span className="pe-mt-filename-pill" title={sample.sample_name}>
            <span className="pe-mt-filename-pill-text">{sample.sample_name}</span>
          </span>
        )}
      </div>

      {showToggle && (
        <div className="pe-mt-toggle-row">
          <div className="pe-mt-toggle-legend">
            {hasGt && (
              <span className="pe-mt-toggle-legend-item">
                <svg width="18" height="10" aria-hidden="true">
                  <rect x="1" y="2" width="16" height="7" fill="none"
                    stroke="currentColor" strokeWidth="1.5" strokeDasharray="4 2" />
                </svg>
                Ground truth
              </span>
            )}
            {hasPred && (
              <span className="pe-mt-toggle-legend-item">
                <svg width="18" height="10" aria-hidden="true">
                  <rect x="1" y="2" width="16" height="7" fill="none"
                    stroke="#22c55e" strokeWidth="2" />
                </svg>
                Prediction
              </span>
            )}
          </div>
          <div className="pe-mt-toggle-pills" role="group" aria-label="overlay-view-toggle">
            {(["gt", "pred", "both"] as BoxView[]).map((v) => (
              <button
                key={v}
                type="button"
                onClick={() => setView(v)}
                disabled={(v === "gt" && !hasGt) || (v === "pred" && !hasPred)}
                className={`pe-mt-toggle-pill ${view === v ? "is-active" : ""}`}
                aria-pressed={view === v}
              >
                {v === "gt" ? "GT" : v === "pred" ? "Pred" : "Both"}
              </button>
            ))}
          </div>
        </div>
      )}

      {!sample ? (
        <EmptyState />
      ) : (
        <>
          <DetectionViewer
            sampleId={sample.sample_id}
            predBoxes={predBoxes}
            gtBoxes={gtBoxes}
            view={view}
            onChangeView={setView}
          />

          {detStats && <DetStatsStrip stats={detStats} />}

          <div className="pe-mt-meta">
            <MetaRow label="File">{sample.sample_name}</MetaRow>
            {sample.sample_id && (
              <MetaRow label="ID">{sample.sample_id}</MetaRow>
            )}
            <MetaRow label="Expected">
              {sample.expected_outcome === "-"
                ? <span className="text-[color:var(--app-text-muted)]">—</span>
                : sample.expected_outcome}
            </MetaRow>
            <MetaRow label="Predicted">
              {sample.predicted_class
                ? sample.predicted_class
                : <span className="text-[color:var(--app-text-muted)]">—</span>}
            </MetaRow>
            <MetaRow label={scoreLabel}>
              {score !== null
                ? <span className="pe-mt-score">{score}%</span>
                : <span className="text-[color:var(--app-text-muted)]">—</span>}
            </MetaRow>
            <div className="pe-mt-meta-row">
              <span className="pe-mt-meta-label">Result</span>
              <span className="pe-mt-meta-value">
                <StatusBadge status={sample.result_status} detStats={detStats} />
              </span>
            </div>
          </div>
        </>
      )}
    </div>
  );
}
