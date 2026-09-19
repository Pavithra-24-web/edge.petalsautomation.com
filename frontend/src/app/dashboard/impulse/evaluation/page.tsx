"use client";

import { type ReactNode, useCallback, useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { useAppStore } from "@/store/appStore";
import { evaluationApi, trainedModelsApi } from "@/utils/api";
import { useTrainingValidity } from "@/hooks/useTrainingValidity";
import {
  RefreshCw,
  AlertTriangle,
  CheckCircle,
  Activity,
  TrendingUp,
  TrendingDown,
  HelpCircle,
  BarChart3,
  X,
  Rocket,
  HardDrive,
  Cpu,
  Clock,
  Lightbulb,
} from "lucide-react";
import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import toast from "react-hot-toast";
import ImpulseNotReady from "@/components/dashboard/ImpulseNotReady";
import MotionPhasePending from "@/components/dashboard/MotionPhasePending";

// Export-only variants that must never be offered or named on this page.
// decoded_float32 exists purely for internal consumers — PXE deployment
// (deployment_worker.py) and post-processing (post-processing-helpers.ts) —
// and is never independently scored/labeled for end users.
const HIDDEN_MODEL_VARIANTS = new Set(["decoded_float32"]);

type EpochPoint = { epoch: number; train_loss?: number | null; val_loss?: number | null };

function extractEpochSeries(training: any): EpochPoint[] {
  if (!training) return [];
  const direct = training.epoch_metrics || training.per_epoch_metrics;
  if (Array.isArray(direct) && direct.length) {
    return direct.map((p: any, i: number) => ({
      epoch: p.epoch ?? i + 1,
      train_loss: p.train_loss ?? p.loss ?? null,
      val_loss: p.val_loss ?? p.validation_loss ?? null,
    }));
  }
  const trainArr: number[] = training.loss_history || training.train_loss_history || [];
  const valArr: number[] = training.val_loss_history || training.validation_loss_history || [];
  const n = Math.max(trainArr.length, valArr.length);
  if (!n) return [];
  return Array.from({ length: n }, (_, i) => ({
    epoch: i + 1,
    train_loss: trainArr[i] ?? null,
    val_loss: valArr[i] ?? null,
  }));
}

function TrainingGraphsModal({ training, onClose }: { training: any; onClose: () => void }) {
  const data = extractEpochSeries(training);
  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm p-4"
      onClick={onClose}
    >
      <div
        className="w-full max-w-3xl overflow-hidden rounded-2xl bg-white shadow-2xl"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-center justify-between px-6 py-4">
          <h3 className="text-lg font-semibold text-gray-900">Training graphs</h3>
          <button
            onClick={onClose}
            className="rounded p-1 text-gray-400 hover:bg-gray-100 hover:text-gray-700"
            aria-label="Close"
          >
            <X size={20} />
          </button>
        </div>
        <div className="bg-[#1a1b3a] p-6">
          <p className="mb-2 inline-flex w-full items-center justify-center gap-1.5 text-center text-xs font-semibold uppercase tracking-wider text-gray-300">
            EPOCH_LOSS
            <InfoTooltip text="A plot showing how model loss changes as the model trains, for both validation and train datasets." />
          </p>
          {data.length === 0 ? (
            <div className="flex h-72 items-center justify-center text-sm text-gray-400">
              No per-epoch metrics available for this training run.
            </div>
          ) : (
            <ResponsiveContainer width="100%" height={360}>
              <LineChart data={data} margin={{ top: 10, right: 28, left: 8, bottom: 56 }}>
                <CartesianGrid stroke="#2a2b55" strokeDasharray="3 3" />
                <XAxis
                  dataKey="epoch"
                  stroke="#9ca3af"
                  tick={{ fontSize: 10, fill: "#9ca3af" }}
                  interval={1}
                  height={40}
                  label={{ value: "Epoch", position: "insideBottom", offset: 0, fill: "#cbd5e1", fontSize: 12 }}
                />
                <YAxis
                  stroke="#9ca3af"
                  tick={{ fontSize: 10, fill: "#9ca3af" }}
                  label={{ value: "Loss", angle: -90, position: "insideLeft", fill: "#cbd5e1", fontSize: 12 }}
                />
                <Tooltip
                  contentStyle={{
                    background: "#1f1f4a",
                    border: "1px solid #373768",
                    borderRadius: 8,
                    fontSize: 12,
                    color: "#e5e7eb",
                  }}
                  labelStyle={{ color: "#e5e7eb" }}
                  labelFormatter={(v: any) => `Epoch ${v}`}
                  formatter={(value: any, name: any) => [
                    typeof value === "number" && Number.isFinite(value) ? value.toFixed(4) : value,
                    name,
                  ]}
                />
                <Legend
                  iconType="circle"
                  verticalAlign="bottom"
                  align="center"
                  wrapperStyle={{ fontSize: 13, bottom: 0, color: "#e5e7eb" }}
                  formatter={(v) => <span style={{ color: "#e5e7eb", marginLeft: 4, marginRight: 12 }}>{v}</span>}
                />
                <Line type="monotone" dataKey="val_loss" name="Validation loss" stroke="#f97316" dot={false} strokeWidth={2} />
                <Line type="monotone" dataKey="train_loss" name="Train loss" stroke="#14b8a6" dot={false} strokeWidth={2} />
              </LineChart>
            </ResponsiveContainer>
          )}
        </div>
      </div>
    </div>
  );
}

function PanelConfusionMatrix({
  matrix,
  labels,
}: {
  matrix: number[][];
  labels: string[];
}) {
  if (!matrix?.length || !labels?.length) return null;

  const rowTotals = matrix.map((row) => row.reduce((sum, value) => sum + value, 0));

  return (
    <div className="overflow-x-auto">
      <table className="w-full text-xs">
        <thead>
          <tr>
            <th className="p-2 text-left text-gray-600" />
            {labels.map((label) => (
              <th
                key={label}
                className="p-2 text-center text-[10px] font-semibold uppercase tracking-wider text-gray-400"
              >
                {label}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {matrix.map((row, rowIndex) => (
            <tr key={`${labels[rowIndex] ?? rowIndex}`}>
              <td className="whitespace-nowrap p-2 text-[10px] font-semibold uppercase tracking-wider text-gray-400">
                {labels[rowIndex] || `Row ${rowIndex + 1}`}
              </td>
              {row.map((value, colIndex) => {
                const total = rowTotals[rowIndex];
                const percent = total > 0 ? Math.round((value / total) * 100) : 0;
                const isDiagonal = rowIndex === colIndex;
                const tone = isDiagonal
                  ? percent > 0
                    ? "bg-green-500/80 text-white"
                    : "bg-green-500/20 text-green-400"
                  : percent > 0
                    ? "bg-red-500/60 text-white"
                    : "text-gray-600";

                return (
                  <td key={`${rowIndex}-${colIndex}`} className="p-1">
                    <div className={`flex h-9 items-center justify-center rounded text-xs font-medium ${tone}`}>
                      {percent}%
                    </div>
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

/**
 * `placement` exists because .pe-eval-panel sets overflow:hidden for its rounded
 * corners — a tooltip anchored in the panel header escapes the top edge and gets
 * clipped. Anchors near the top of the card should pass "bottom" so the bubble
 * opens downward, into the card.
 */
function InfoTooltip({
  text,
  placement = "top",
}: {
  text: string;
  placement?: "top" | "bottom";
}) {
  const isBottom = placement === "bottom";
  return (
    <span className="group relative inline-flex items-center align-middle">
      <HelpCircle
        size={12}
        className="cursor-help text-gray-500 transition-colors group-hover:text-gray-300"
        aria-label={text}
        tabIndex={0}
      />
      <span
        role="tooltip"
        className={`pointer-events-none absolute left-1/2 z-50 w-72 -translate-x-1/2 rounded-xl bg-[#373768] px-4 py-3 text-left text-xs font-normal normal-case leading-relaxed tracking-normal text-white opacity-0 shadow-2xl transition-opacity duration-150 group-hover:opacity-100 group-focus-within:opacity-100 ${
          isBottom ? "top-full mt-2" : "bottom-full mb-2"
        }`}
      >
        {text}
        <span
          aria-hidden="true"
          className={`absolute left-1/2 -translate-x-1/2 rotate-45 h-2 w-2 bg-[#373768] ${
            isBottom ? "bottom-full translate-y-1/2" : "top-full -translate-y-1/2"
          }`}
        />
      </span>
    </span>
  );
}

function MetricCard({ label, value, info }: { label: string; value: string; info?: string }) {
  return (
    <div className="pe-eval-metric-card">
      <p className="pe-eval-metric-card-label inline-flex items-center gap-1">
        {label}
        {info && <InfoTooltip text={info} />}
      </p>
      <p className="pe-eval-metric-card-value">{value}</p>
    </div>
  );
}

const MODEL_VERSION_TOOLTIP =
  "Compare the performance of optimized variants of the model.";
const MAP50_TOOLTIP =
  "Mean Average Precision [IoU=50]. Represents object detection precision, from 0–1. Also known as PASCAL VOC mAP, mAP@0.5, or AP₅₀.";
const F1_TOOLTIP = "Represents the F1 score; a mix of precision and recall.";
const Bestval_loss_TOOLTIP = "Lowest validation loss reached during training. Lower values generally indicate better performance on unseen data.";

const Precision_TOOLTIP = "Precision (non-background). Classification precision for all NanoVision cells that are not the implied background class";
const Recall_TOOLTIP = "Recall (non-background). Classification recall for all NanoVision cells that are not the implied background class.";
const F1_SCORE_TOOLTIP = "F1 Score (non-background). Classification F1 score for all NanoVision cells that are not the implied background class.";



function SectionLabel({ children }: { children: ReactNode }) {
  return (
    <h4 className="pe-eval-section-label">
      {children}
    </h4>
  );
}

function MetricRow({ label, value }: { label: string; value: string }) {
  return (
    <div className="pe-eval-metric-row">
      <span className="pe-eval-metric-row-label">{label}</span>
      <span className="pe-eval-metric-row-value">{value}</span>
    </div>
  );
}

function DeviceMetric({
  icon,
  label,
  value,
  note,
  overBudget,
}: {
  icon: ReactNode;
  label: string;
  value: string;
  note?: string;
  overBudget?: boolean;
}) {
  return (
    <div className="flex items-start gap-3">
      <div className="flex h-10 w-10 flex-shrink-0 items-center justify-center rounded-full bg-gray-800 text-gray-300">
        {icon}
      </div>
      <div className="min-w-0">
        <p className="text-[10px] uppercase tracking-wider text-gray-500">{label}</p>
        <p className="text-sm font-bold text-gray-200">
          {value}
          {overBudget && (
            <span className="ml-2 text-[10px] font-semibold text-red-400">OVER BUDGET</span>
          )}
        </p>
        {note && <p className="mt-0.5 text-[11px] leading-snug text-gray-500">{note}</p>}
      </div>
    </div>
  );
}

// Backend Phase 5 estimates return raw bytes/ms + a `unit` field — formatting
// for display is the UI's job (DeviceSpecification's own doc comment says the
// same of its canonical-unit columns).
function formatDeviceBytes(n: number | null | undefined): string {
  if (n === null || n === undefined) return "N/A";
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
  return `${(n / (1024 * 1024)).toFixed(1)} MB`;
}

function formatDeviceMs(n: number | null | undefined): string {
  if (n === null || n === undefined) return "N/A";
  return `${n < 10 ? n.toFixed(2) : n.toFixed(1)} ms`;
}

// One flash/RAM/latency figure from `device_performance` (Target Device
// Phase 5, docs/target_device_phase5.md). `metric` is a MetricEstimate dict:
// {value, unit, source, budget, budget_source, margin, fits, reason_code,
// note, confidence}. Never rendered as a plain number — the note (and
// confidence, for estimated figures) always travels with it.
function DevicePerformanceMetric({
  icon,
  label,
  metric,
  format,
}: {
  icon: ReactNode;
  label: string;
  metric: any;
  format: (n: number | null | undefined) => string;
}) {
  if (!metric) return null;
  const budgetSuffix =
    metric.budget != null ? ` of ${format(metric.budget)} budget` : "";
  const noteParts = [metric.note, metric.confidence].filter(Boolean);
  return (
    <DeviceMetric
      icon={icon}
      label={label}
      value={`${format(metric.value)}${budgetSuffix}`}
      note={noteParts.join(" ")}
      overBudget={metric.fits === false}
    />
  );
}

// Phase 7 optimization advice (Recommendation dicts: {constraint, severity,
// code, title, detail, expected_gain}) rendered under the on-device metrics
// above — visibly secondary, never affecting whether a build can happen.
// `reason` is always a real answer ("fits comfortably" / "cannot be
// assessed" / "incompatible") when `items` is empty, so this never renders
// blank.
function DeviceRecommendations({ items, reason }: { items: any[] | undefined; reason: string | null | undefined }) {
  if (!items || items.length === 0) {
    if (!reason) return null;
    return <p className="mt-3 text-xs text-gray-500">{reason}</p>;
  }
  return (
    <div className="mt-3 space-y-2">
      {items.map((rec, i) => (
        <div
          key={`${rec.code}-${rec.constraint}-${i}`}
          className="flex items-start gap-2 rounded-lg border border-dashed border-gray-700 bg-gray-800/50 p-3"
        >
          <Lightbulb size={14} className="mt-0.5 flex-shrink-0 text-gray-400" aria-hidden="true" />
          <div className="min-w-0">
            <p className="text-sm font-semibold text-gray-200">
              {rec.title}
              <span
                className={`ml-2 rounded-full px-1.5 py-0.5 text-[10px] font-semibold uppercase tracking-wide ${
                  rec.severity === "over_budget" ? "bg-amber-400/10 text-amber-400" : "bg-gray-700 text-gray-400"
                }`}
              >
                {rec.constraint}
              </span>
            </p>
            <p className="mt-0.5 text-xs leading-snug text-gray-400">
              {rec.detail} Expected gain: {rec.expected_gain}.
            </p>
          </div>
        </div>
      ))}
    </div>
  );
}

function ModelPanel({ impulseId }: { impulseId: string }) {
  const [panel, setPanel] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [selectedVariant, setSelectedVariant] = useState("int8");
  const [selectedEngine, setSelectedEngine] = useState("tflite");
  const [showGraphs, setShowGraphs] = useState(false);

  const loadPanel = useCallback(async (variant: string, engine: string) => {
    setLoading(true);
    try {
      const { data } = await trainedModelsApi.panel(impulseId, variant, engine);
      const selected = data.selected_model_version?.variant;
      if (selected && HIDDEN_MODEL_VARIANTS.has(selected)) {
        // The API is free to select a hidden variant (e.g. no int8/float32
        // export is scored yet) — steer the page to a visible one instead of
        // rendering a pill whose value has no matching dropdown option.
        const visible = (data.available_model_versions || []).filter(
          (v: any) => !HIDDEN_MODEL_VARIANTS.has(v.variant)
        );
        const fallback =
          visible.find((v: any) => v.variant === "int8") ||
          visible.find((v: any) => v.variant === "float32") ||
          visible[0];
        if (fallback && fallback.variant !== variant) {
          await loadPanel(fallback.variant, engine);
          return;
        }
      }
      setPanel(data);
      setSelectedVariant(selected && !HIDDEN_MODEL_VARIANTS.has(selected) ? selected : variant);
      setSelectedEngine(data.selected_engine?.id || engine);
    } catch {
      setPanel(null);
    } finally {
      setLoading(false);
    }
  }, [impulseId]);

  useEffect(() => {
    loadPanel(selectedVariant, selectedEngine);
  }, [impulseId]);

  if (loading && !panel) {
    return (
      <div className="eval-panel rounded-xl p-8 text-center">
        <RefreshCw size={16} className="mx-auto animate-spin text-gray-500" />
        <p className="mt-2 text-xs text-gray-500">Loading model panel...</p>
      </div>
    );
  }

  if (!panel) {
    return (
      <div className="eval-panel rounded-xl p-8 text-center">
        <p className="text-sm text-gray-500">No trained model found for this impulse.</p>
        <p className="mt-1 text-xs text-gray-600">Train a model first to see performance results here.</p>
      </div>
    );
  }

  const training = panel.training_performance || {};
  const metrics = panel.metrics || {};
  const versions = (panel.available_model_versions || []).filter(
    (v: any) => !HIDDEN_MODEL_VARIANTS.has(v.variant)
  );
  const engines = panel.available_engines || [];
  const selVersion = panel.selected_model_version || {};
  const selEngine = panel.selected_engine || {};
  // loadPanel steers away from hidden variants before they ever reach state,
  // but never let one leak into rendered text even if that coercion path is
  // bypassed (e.g. a stale panel response).
  const safeSelectedVariant =
    selVersion.variant && !HIDDEN_MODEL_VARIANTS.has(selVersion.variant)
      ? selVersion.variant
      : selectedVariant;

  const isYoloPro = panel.is_yolo_pro === true;
  const isFomo = panel.is_fomo === true;
  const fomoEvalStatus = metrics.fomo_eval_status ?? null;
  const fomoEvalError = metrics.fomo_eval_error ?? null;
  const yoloEvalStatus = training.yolo_pro_eval_status ?? metrics.yolo_pro_eval_status ?? null;
  const yoloEvalError = training.yolo_pro_eval_error ?? metrics.yolo_pro_eval_error ?? null;
  // Metrics are stored per variant. When the selected variant has none of its
  // own, every derived number below must read "N/A" — showing the job-wide
  // (float32) figures under an int8 selection is exactly the mismatch this
  // panel is meant to expose.
  const variantScored = selVersion.metrics_available === true;
  const hasRealFomoMetrics = isFomo && fomoEvalStatus === "success" && variantScored;
  const hasRealYoloMetrics = isYoloPro && yoloEvalStatus === "success" && variantScored;

  const currentVersionLabel =
    versions.find((item: any) => item.variant === safeSelectedVariant && item.available)?.label ||
    safeSelectedVariant ||
    "Current";
  const currentEngineLabel =
    engines.find((item: any) => item.id === selEngine.id)?.label ||
    selEngine.id ||
    "Engine";

  const headlineLabel = isYoloPro
    ? hasRealYoloMetrics ? "mAP@50" : "Best val_loss"
    : isFomo
      ? hasRealFomoMetrics ? "F1 score" : "Best val_loss"
      : training.headline_metric?.name || "Metric";

  const headlineValue = isYoloPro
    ? hasRealYoloMetrics && training.map50 != null
      ? `${(training.map50 * 100).toFixed(1)}%`
      : training.best_loss != null
        ? training.best_loss.toFixed(4)
        : "N/A"
    : isFomo
      ? hasRealFomoMetrics && training.f1_score != null
        ? `${(training.f1_score * 100).toFixed(1)}%`
        : training.best_loss != null
          ? training.best_loss.toFixed(4)
          : "N/A"
      : training.headline_metric?.value != null
        ? `${training.headline_metric.value}%`
        : "N/A";

  const auxLabel = "Best val_loss";
  // Only show the Best val_loss tile when the model actually reports a real
  // scalar val_loss. Object-detection / SSD (e.g. MobileNetV2 SSD FPN-Lite)
  // don't produce a meaningful scalar val_loss, so "N/A" here is misleading.
  const hasAuxValLoss = Number.isFinite(training.best_loss);
  const auxValue = hasAuxValLoss ? training.best_loss.toFixed(4) : "N/A";
  const detailedMetrics = metrics.aggregate || metrics.detailed_metrics || {};
  const metricsRows = metrics.rows || [];
  const yoloSummaryRows = !variantScored ? [] : [
    { label: "mAP", value: detailedMetrics.map ?? training.map },
    { label: "mAP@[IoU=50]", value: detailedMetrics.map50 ?? training.map50 },
    { label: "mAP@[IoU=75]", value: detailedMetrics.map75 ?? training.map75 },
    { label: "mAP@[area=small]", value: detailedMetrics.map_small },
    { label: "mAP@[area=medium]", value: detailedMetrics.map_medium },
    { label: "mAP@[area=large]", value: detailedMetrics.map_large },
    { label: "Recall@[max_detections=1]", value: detailedMetrics.recall_max1 },
    { label: "Recall@[max_detections=10]", value: detailedMetrics.recall_max10 },
    { label: "Recall@[max_detections=100]", value: detailedMetrics.recall_max100 },
    { label: "Recall@[area=small]", value: detailedMetrics.recall_small },
    { label: "Recall@[area=medium]", value: detailedMetrics.recall_medium },
    { label: "Recall@[area=large]", value: detailedMetrics.recall_large },
    { label: "Precision score (legacy)", value: detailedMetrics.precision_legacy ?? training.precision },
  ];
  const fomoSummaryRows = [
    { label: "Precision (non-background)", value: metrics.aggregate?.precision },
    { label: "Recall (non-background)", value: metrics.aggregate?.recall },
    { label: "F1 Score (non-background)", value: metrics.aggregate?.f1_score },
  ].filter((row) => row.value !== undefined && row.value !== null);

  // Background ("negative") images: mAP and precision/recall are near-blind to
  // them by construction, so a project that added negatives needs its own
  // readout to tell whether they helped. Rendered only when the evaluated
  // split actually contained background images — a raw count, not a ratio,
  // because "0.3 false positives per background image" is the number a user
  // acts on, unlike a percentage of nothing.
  const backgroundImages = metrics.aggregate?.background_images;
  const backgroundRows = !backgroundImages ? [] : [
    { label: "Background images evaluated", value: backgroundImages, kind: "count" },
    {
      label: "Background FP rate (per image)",
      value: metrics.aggregate?.background_fp_rate,
      kind: "score",
    },
    {
      label: "Background images with a false positive",
      value: metrics.aggregate?.background_images_with_fp,
      kind: "count",
    },
  ].filter((row) => row.value !== undefined && row.value !== null);

  function fmtRatio(value: any) {
    if (value === null || value === undefined) return "N/A";
    // COCO eval uses -1 as the "no ground-truth in this bucket" sentinel
    // (e.g. mAP@[area=large] when no large objects exist) — not a real score.
    if (Number(value) < 0) return "N/A";
    return `${(Number(value) * 100).toFixed(1)}%`;
  }

  function fmtScore(value: any) {
    if (value === null || value === undefined) return "N/A";
    return Number(value).toFixed(4);
  }

  return (
    <div className="pe-eval-panel">
      <div className="pe-eval-panel-header">
        <div>
          <h3 className="pe-eval-panel-title">Model</h3>
          <p className="pe-eval-panel-subtitle">
            {currentVersionLabel} using {currentEngineLabel}
          </p>
        </div>

        <div className="flex flex-wrap items-center gap-2">
          {/* Model Version picker. Switching re-fetches the panel for that
              variant so every number below belongs to the selected variant —
              metrics are stored per variant, not shared across them. */}
          {/* Tooltip sits beside the <label>, not inside it — nesting it would
              make clicking the help icon activate the select. */}
          <span className="inline-flex items-center gap-1">
            <label
              htmlFor="pe-model-version"
              className="text-sm font-medium text-gray-400"
            >
              Model version:
            </label>
            <InfoTooltip text={MODEL_VERSION_TOOLTIP} placement="bottom" />
          </span>
          <select
            id="pe-model-version"
            className="pe-eval-chip"
            value={safeSelectedVariant}
            disabled={loading}
            onChange={(e) => {
              const next = e.target.value;
              setSelectedVariant(next);
              loadPanel(next, selectedEngine);
            }}
          >
            {/* Option text carries the label ONLY. A native select sizes itself
                to its widest option, so a state suffix here would stretch the
                collapsed pill by text that is never visible in it. Unavailable
                variants read as disabled, and the selected variant's state is
                spelled out by the notice below. */}
            {versions.map((v: any) => (
              <option
                key={v.variant}
                value={v.variant}
                disabled={!v.available}
                title={
                  !v.available
                    ? v.unavailable_reason || "Unavailable"
                    : !v.metrics_available
                      ? v.metrics_unavailable_reason || "Not scored"
                      : undefined
                }
              >
                {v.label}
              </option>
            ))}
          </select>
          {loading && <RefreshCw size={13} className="animate-spin text-gray-500" />}
        </div>
      </div>

      {/* Explicit unavailable state. The selected variant's own metrics are
          either missing (older job), failed, or the export itself failed —
          in every case say so and say why, rather than rendering some other
          variant's numbers in its place. */}
      {!selVersion.metrics_available && (
        <div className="pe-eval-variant-notice" role="status">
          <AlertTriangle size={14} aria-hidden="true" />
          <span>
            <strong>No metrics for {safeSelectedVariant}.</strong>{" "}
            {selVersion.metrics_unavailable_reason ||
              "This variant has not been scored independently."}
          </span>
        </div>
      )}

      <div className="pe-eval-panel-body">
        <div className="pe-eval-top-grid">
          <div>
            <div className="mb-4 flex items-center gap-2">
              <h4 className="pe-eval-section-label" style={{ marginBottom: 0 }}>
                Last training performance <span className="normal-case font-normal"></span>
              </h4>
              <button
                type="button"
                onClick={() => setShowGraphs(true)}
                className="flex h-5 w-5 items-center justify-center rounded text-gray-400 transition-colors hover:bg-gray-800 hover:text-teal-300"
                aria-label="View training graphs"
                title="Training graphs"
              >
                <BarChart3 size={13} />
              </button>
            </div>
            <div className="pe-eval-headline-row">
              <div className="pe-eval-headline-icon pe-eval-headline-icon--up">
                <TrendingUp size={18} />
              </div>
              <div>
                <p className="pe-eval-headline-label inline-flex items-center gap-1">
                  {headlineLabel}
                  {headlineLabel === "mAP@50" && <InfoTooltip text={MAP50_TOOLTIP} />}
                  {(headlineLabel === "F1 score" || headlineLabel === "F1 SCORE") && <InfoTooltip text={F1_TOOLTIP} />}
                  {headlineLabel === "Best val_loss" && <InfoTooltip text={Bestval_loss_TOOLTIP} />}
                </p>
                <p className="pe-eval-headline-value">{headlineValue}</p>
              </div>
            </div>
            {hasAuxValLoss && (
              <div className="pe-eval-headline-row">
                <div className="pe-eval-headline-icon pe-eval-headline-icon--down">
                  <TrendingDown size={18} />
                </div>
                <div>
                  <p className="pe-eval-headline-label inline-flex items-center gap-1">
                    {auxLabel}
                    {auxLabel === "Best val_loss" && <InfoTooltip text={Bestval_loss_TOOLTIP} />}
                  </p>
                  <p className="pe-eval-headline-value pe-eval-headline-value--sm">{auxValue}</p>
                </div>
              </div>
            )}
          </div>
        </div>

        {selVersion.is_fallback && selVersion.fallback_reason && (
          <div className="flex items-start gap-2 rounded-lg border border-amber-800/50 bg-amber-900/20 p-3">
            <AlertTriangle size={14} className="mt-0.5 flex-shrink-0 text-amber-400" />
            <p className="text-xs text-amber-400">{selVersion.fallback_reason}</p>
          </div>
        )}

        {selEngine.is_fallback && selEngine.fallback_reason && (
          <div className="flex items-start gap-2 rounded-lg border border-amber-800/50 bg-amber-900/20 p-3">
            <AlertTriangle size={14} className="mt-0.5 flex-shrink-0 text-amber-400" />
            <p className="text-xs text-amber-400">{selEngine.fallback_reason}</p>
          </div>
        )}

        <div>
          <SectionLabel>
            Metrics <span className="normal-case font-normal"></span>
          </SectionLabel>
          {hasRealYoloMetrics && (
            <>
              <div className="space-y-0">
                {yoloSummaryRows.map((row) => (
                  <MetricRow key={row.label} label={row.label} value={fmtRatio(row.value)} />
                ))}
              </div>
              {metricsRows.length > 0 && (
                <div className="mt-5 overflow-x-auto">
                  <table className="w-full text-xs">
                    <thead>
                      <tr className="border-b border-gray-700">
                        <th className="py-2 text-left uppercase tracking-wider text-gray-500 text-[10px]">Class</th>
                        <th className="py-2 text-right uppercase tracking-wider text-gray-500 text-[10px]">AP</th>
                        <th className="py-2 text-right uppercase tracking-wider text-gray-500 text-[10px]">Prec.</th>
                        <th className="py-2 text-right uppercase tracking-wider text-gray-500 text-[10px]">Recall</th>
                      </tr>
                    </thead>
                    <tbody>
                      {metricsRows.map((row: any) => (
                        <tr key={row.label} className="border-b border-gray-800/50">
                          <td className="py-2.5 text-gray-300">{row.label}</td>
                          <td className="py-2.5 text-right text-gray-400">{fmtRatio(row.ap)}</td>
                          <td className="py-2.5 text-right text-gray-400">{fmtRatio(row.precision)}</td>
                          <td className="py-2.5 text-right text-gray-400">{fmtRatio(row.recall)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                  <p className="mt-2 text-[10px] text-gray-600">
                    COCO-style evaluation · IoU 0.50:0.95 · area buckets: small&lt;32² medium&lt;96² large≥96² px
                  </p>
                </div>
              )}
            </>
          )}

          {hasRealFomoMetrics && (
            <>
              <div className="space-y-0">
                {fomoSummaryRows.map((row) => (
                  <MetricRow key={row.label} label={row.label} value={fmtRatio(row.value)} />
                ))}
              </div>
              {metricsRows.length > 0 && (
                <div className="mt-5 overflow-x-auto">
                  <table className="w-full text-xs">
                    <thead>
                      <tr className="border-b border-gray-700">
                        <th className="py-2 text-left uppercase tracking-wider text-gray-500 text-[10px]">Class</th>
                        <th className="py-2 text-right uppercase tracking-wider text-gray-500 text-[10px]">Prec.</th>
                        <th className="py-2 text-right uppercase tracking-wider text-gray-500 text-[10px]">Recall</th>
                        <th className="py-2 text-right uppercase tracking-wider text-gray-500 text-[10px]">F1</th>
                        <th className="py-2 text-right uppercase tracking-wider text-gray-500 text-[10px]">TP/FP/FN</th>
                      </tr>
                    </thead>
                    <tbody>
                      {metricsRows.map((row: any) => (
                        <tr key={row.label} className="border-b border-gray-800/50">
                          <td className="py-2.5 text-gray-300">{row.label}</td>
                          <td className="py-2.5 text-right text-gray-400">{fmtRatio(row.precision)}</td>
                          <td className="py-2.5 text-right text-gray-400">{fmtRatio(row.recall)}</td>
                          <td className="py-2.5 text-right text-gray-400">{fmtRatio(row.f1_score)}</td>
                          <td className="py-2.5 text-right text-gray-500">
                            {row.tp != null ? `${row.tp}/${row.fp}/${row.fn}` : "N/A"}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </>
          )}

          {backgroundRows.length > 0 && (
            <div className="mt-5">
              <SectionLabel>
                Background images <span className="normal-case font-normal"></span>
              </SectionLabel>
              <div className="space-y-0">
                {backgroundRows.map((row) => (
                  <MetricRow
                    key={row.label}
                    label={row.label}
                    value={row.kind === "count" ? String(row.value) : fmtScore(row.value)}
                  />
                ))}
              </div>
              <p className="mt-2 text-[10px] text-gray-600">
                Images marked as containing no objects. mAP and precision/recall
                cannot see these — this is the readout that tells you whether
                adding negatives reduced false positives.
              </p>
            </div>
          )}

          {showGraphs && (
            <TrainingGraphsModal training={training} onClose={() => setShowGraphs(false)} />
          )}

          {(isYoloPro || isFomo) && !hasRealYoloMetrics && !hasRealFomoMetrics && (
            <div
              className={`flex items-start gap-2 rounded-lg p-3 ${yoloEvalStatus === "inference_failed" || fomoEvalStatus === "inference_failed"
                ? "border border-red-800/40 bg-red-900/20"
                : "border border-gray-700 bg-gray-800/50"
                }`}
            >
              {yoloEvalStatus === "success" || fomoEvalStatus === "success" ? (
                <CheckCircle size={14} className="mt-0.5 flex-shrink-0 text-teal-400" />
              ) : (
                <AlertTriangle size={14} className="mt-0.5 flex-shrink-0 text-amber-400" />
              )}
              <div>
                <p className="text-xs font-medium text-gray-300">
                  {isYoloPro ? "Vision Pro results" : "NanoVision results"}
                </p>
                <p className="mt-0.5 text-[11px] text-gray-500">
                  {isYoloPro
                    ? yoloEvalError ||
                    (yoloEvalStatus === "no_data"
                      ? "No boxed test data was available when this model was evaluated."
                      : "Detailed Vision Pro metrics are not available for this model yet.")
                    : fomoEvalError ||
                    (fomoEvalStatus === "no_data"
                      ? "No labelled test samples were available when this model was evaluated."
                      : "Detailed NanoVision metrics are not available for this model yet.")}
                </p>
              </div>
            </div>
          )}
        </div>

        {panel.device_performance && (
          <div className="mt-6">
            <SectionLabel>On-device performance</SectionLabel>
            {panel.device_performance.available ? (
              <div className="grid grid-cols-1 gap-4 sm:grid-cols-3">
                <DevicePerformanceMetric
                  icon={<HardDrive size={16} />}
                  label="Flash usage"
                  metric={panel.device_performance.flash_usage}
                  format={formatDeviceBytes}
                />
                <DevicePerformanceMetric
                  icon={<Cpu size={16} />}
                  label="Peak RAM"
                  metric={panel.device_performance.ram_usage}
                  format={formatDeviceBytes}
                />
                <DevicePerformanceMetric
                  icon={<Clock size={16} />}
                  label="Inference latency"
                  metric={panel.device_performance.inferencing_time}
                  format={formatDeviceMs}
                />
              </div>
            ) : (
              <div className="flex items-start gap-2 rounded-lg border border-gray-700 bg-gray-800/50 p-3">
                <AlertTriangle size={14} className="mt-0.5 flex-shrink-0 text-amber-400" />
                <p className="text-xs text-gray-400">{panel.device_performance.unavailable_reason}</p>
              </div>
            )}
            <DeviceRecommendations
              items={panel.device_performance.recommendations}
              reason={panel.device_performance.recommendations_reason}
            />
          </div>
        )}
      </div>
    </div>
  );
}

export default function EvaluationPage() {
  const router = useRouter();
  const { activeProject, activeImpulse: storeActiveImpulse } = useAppStore();
  const currentImpulse =
    storeActiveImpulse?.project_id === activeProject?.id ? storeActiveImpulse : null;

  const {
    hasValidTrainingOutput,
    loading: trainingValidityLoading,
  } = useTrainingValidity(currentImpulse?.id ?? null);

  const [evalData, setEvalData] = useState<any>(null);
  const [loading, setLoading] = useState(false);

  const loadEvaluation = useCallback(async (impulseId: string) => {
    setEvalData(null);
    setLoading(true);
    try {
      const { data } = await evaluationApi.forImpulse(impulseId);
      setEvalData(data);
    } catch (error: any) {
      console.error("Failed to load evaluation data:", error);
      setEvalData(null);
      const status = error?.response?.status ?? 0;
      if (status >= 500) {
        toast.error("Failed to load evaluation");
      }
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    if (!activeProject || !currentImpulse?.id) {
      setEvalData(null);
      setLoading(false);
      return;
    }
    // Gate the artifact fetch on the single source of truth — never display
    // results for an older completed run if the latest run is in a non-completed
    // state.
    if (!hasValidTrainingOutput) {
      setEvalData(null);
      setLoading(false);
      return;
    }
    loadEvaluation(currentImpulse.id);
  }, [activeProject?.id, currentImpulse?.id, hasValidTrainingOutput, loadEvaluation]);

  const history = evalData?.job?.training_history || {};
  const isYoloPro =
    history?.is_yolo_pro === true ||
    evalData?.job?.training_history?.is_yolo_pro === true ||
    evalData?.architecture === "yolo_pro";
  const isFomo =
    !isYoloPro &&
    (evalData?.is_detection === true ||
      history?.is_fomo === true ||
      (evalData?.job?.architecture || evalData?.architecture || "").toLowerCase().includes("fomo"));
  // Until training validity resolves, we can't tell whether to render the
  // main UI or the "Almost there!" warning. Render a loading state first so
  // the warning is the first non-loading frame for untrained impulses.
  const isInitialLoad =
    !!currentImpulse?.id && (trainingValidityLoading || (hasValidTrainingOutput && loading));
  const notReady = !trainingValidityLoading && !!currentImpulse?.id && !hasValidTrainingOutput;

  if (activeProject?.project_type === "motion") {
    return (
      <MotionPhasePending
        feature="Evaluation"
        description="Once motion evaluation is wired up, this page will show accuracy, a confusion matrix and a training summary for your trained motion model."
        tip="Evaluation for motion models lands in a later phase — for now, design and train your impulse."
        icon={Activity}
      />
    );
  }

  if (!currentImpulse) {
    return (
      <ImpulseNotReady
        description="No impulse to evaluate yet. Create an impulse and train a model before you can evaluate it."
        tip="Head to Impulse Design to create one, then come back here once training finishes."
        actions={
          <button
            type="button"
            onClick={() => router.push("/dashboard/impulse")}
            className="pe-warn-cta"
          >
            <Rocket size={15} /> Go to impulse design
          </button>
        }
      />
    );
  }

  if (isInitialLoad) {
    return (
      <div className="flex h-[calc(100vh-theme('spacing.16'))] items-center justify-center pt-2">
        <span className="inline-flex items-center gap-2 text-sm text-[var(--app-text-soft)]">
          <RefreshCw size={14} className="animate-spin" />
          Verifying model status...
        </span>
      </div>
    );
  }

  if (notReady) {
    return (
      <ImpulseNotReady
        description="Your impulse is not fully trained. Use the items in the navigation bar to configure and train your model before you can evaluate it."
        tip="Train your model first — evaluation metrics need a completed training run to compare against."
      />
    );
  }

  return (
    <div className="pe-evaluation evaluation-page mx-auto max-w-7xl space-y-6">
      <div className="pe-evaluation-header">
        <div className="pe-evaluation-header-icon" aria-hidden="true">
          <Activity size={22} strokeWidth={2.2} />
        </div>
        <div>
          <h2 className="pe-evaluation-title">Evaluation</h2>
          <p className="pe-evaluation-sub">Evaluate your trained model against labeled samples</p>
        </div>
      </div>

      {loading && (
        <div className="pe-eval-panel pe-eval-panel--loading">
          <p>Loading evaluation results...</p>
        </div>
      )}

      {currentImpulse?.id && !loading && <ModelPanel impulseId={currentImpulse.id} />}

    </div>
  );
}
