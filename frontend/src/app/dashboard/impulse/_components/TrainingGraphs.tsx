"use client";
import {
  LineChart, Line, XAxis, YAxis, Tooltip as RTooltip, CartesianGrid, ResponsiveContainer, Legend, ReferenceLine,
} from "recharts";
import { LineChart as LineChartIcon, HelpCircle } from "lucide-react";

function InfoIcon({ text }: { text: string }) {
  return (
    <span className="group relative inline-flex items-center align-middle">
      <HelpCircle
        size={12}
        className="cursor-help text-[var(--app-text-soft)] transition-colors group-hover:text-[var(--app-text)]"
        aria-label={text}
        tabIndex={0}
      />
      <span
        role="tooltip"
        className="pointer-events-none absolute bottom-full left-0 z-50 mb-2 w-64 max-w-[calc(100vw-2rem)] rounded-xl bg-[#373768] px-3 py-2 text-left text-[11px] font-normal normal-case leading-relaxed tracking-normal text-white opacity-0 shadow-2xl transition-opacity duration-150 group-hover:opacity-100 group-focus-within:opacity-100"
      >
        {text}
        <span
          aria-hidden="true"
          className="absolute left-1.5 top-full -translate-x-1/2 -translate-y-1/2 rotate-45 h-2 w-2 bg-[#373768]"
        />
      </span>
    </span>
  );
}

const LOSS_INFO =
  "A plot showing how model loss changes as the model trains, for both validation and train datasets.";
const ACCURACY_INFO =
  "A plot showing how model accuracy changes as the model trains, for both validation and train datasets.";

/**
 * One persisted metric object per completed epoch. This mirrors the
 * architecture-agnostic `training_performance.epoch_metrics` API contract
 * exactly — the component must never branch on worker or model type.
 * `train_loss`/`val_loss` are always present; accuracy fields are optional and
 * omitted (not nulled) by architectures that don't compute them.
 */
export interface EpochMetric {
  epoch: number;
  train_loss?: number;
  val_loss?: number;
  train_accuracy?: number;
  val_accuracy?: number;
}

interface TrainingGraphsProps {
  /** The persisted epoch_metrics array (live or historical — same shape). */
  epochMetrics?: EpochMetric[] | null;
  /** When the run is still training, the empty state invites the first epoch. */
  isRunning?: boolean;
}

const AXIS_TICK = { fontSize: 11, fill: "var(--app-text-soft)" };
const GRID_STROKE = "var(--app-border)";

const TOOLTIP_STYLE = {
  background: "var(--app-surface)",
  border: "1px solid var(--app-border)",
  borderRadius: 8,
  fontSize: 12,
  color: "var(--app-text)",
} as const;

/** Epoch of the best (minimum) value of `key`, or null if the series is empty. */
function bestEpochByMin(data: EpochMetric[], key: keyof EpochMetric): number | null {
  let best: { epoch: number; v: number } | null = null;
  for (const m of data) {
    const v = m[key];
    if (typeof v !== "number" || !Number.isFinite(v)) continue;
    if (best === null || v < best.v) best = { epoch: m.epoch, v };
  }
  return best ? best.epoch : null;
}

/** Epoch of the best (maximum) value of `key`, or null if the series is empty. */
function bestEpochByMax(data: EpochMetric[], key: keyof EpochMetric): number | null {
  let best: { epoch: number; v: number } | null = null;
  for (const m of data) {
    const v = m[key];
    if (typeof v !== "number" || !Number.isFinite(v)) continue;
    if (best === null || v > best.v) best = { epoch: m.epoch, v };
  }
  return best ? best.epoch : null;
}

function MetricChart({
  data,
  series,
  yLabel,
  bestEpoch,
  bestLabel,
}: {
  data: EpochMetric[];
  series: { key: keyof EpochMetric; name: string; color: string }[];
  yLabel: string;
  bestEpoch?: number | null;
  bestLabel?: string;
}) {
  return (
    <ResponsiveContainer width="100%" height="100%">
      <LineChart data={data} margin={{ top: 16, right: 20, left: 4, bottom: 40 }}>
        <CartesianGrid stroke={GRID_STROKE} vertical={false} />
        <XAxis
          dataKey="epoch"
          tick={AXIS_TICK}
          allowDecimals={false}
          height={36}
          label={{ value: "Epoch", position: "insideBottom", offset: 0, fill: "var(--app-text-soft)", fontSize: 11 }}
        />
        <YAxis
          tick={AXIS_TICK}
          width={48}
          label={{ value: yLabel, angle: -90, position: "insideLeft", fill: "var(--app-text-soft)", fontSize: 11 }}
        />
        <RTooltip
          contentStyle={TOOLTIP_STYLE}
          labelFormatter={(v: any) => `Epoch ${v}`}
          formatter={(value: any, name: any) => [Number(value).toFixed(4), name]}
        />
        <Legend
          verticalAlign="bottom"
          align="center"
          wrapperStyle={{ fontSize: 11, bottom: 20, color: "var(--app-text-soft)" }}
          formatter={(v) => <span style={{ marginLeft: 4, marginRight: 12, color: "var(--app-text-soft)" }}>{v}</span>}
        />
        {/* Mark the best validation epoch — derived from the series itself so it
            stays architecture-agnostic (no branching on worker/model type). */}
        {bestEpoch != null && data.length > 1 && (
          <ReferenceLine
            x={bestEpoch}
            stroke="var(--app-text-soft)"
            strokeDasharray="4 4"
            label={{ value: bestLabel ?? "Best", position: "top", fill: "var(--app-text-soft)", fontSize: 10 }}
          />
        )}
        {series.map((s) => (
          <Line
            key={String(s.key)}
            type="monotone"
            dataKey={s.key as string}
            name={s.name}
            stroke={s.color}
            strokeWidth={2}
            dot={{ r: 2, strokeWidth: 0, fill: s.color }}
            activeDot={{ r: 4 }}
            connectNulls
            isAnimationActive={false}
          />
        ))}
      </LineChart>
    </ResponsiveContainer>
  );
}

export function TrainingGraphs({ epochMetrics, isRunning }: TrainingGraphsProps) {
  const metrics = Array.isArray(epochMetrics) ? epochMetrics : [];

  // Empty state only when the array is missing or empty — never on loading.
  if (metrics.length === 0) {
    return (
      <div className="rounded-xl border border-[var(--app-border)] bg-[var(--app-surface)] px-5 py-8 text-center">
        <LineChartIcon size={18} className="mx-auto text-[var(--app-text-soft)]" />
        <p className="mt-2 text-sm font-medium text-[var(--app-text)]">No training graph available</p>
        <p className="mt-1 text-[12px] text-[var(--app-text-soft)]">
          {isRunning
            ? "Loss curves appear here once the first epoch completes."
            : "This run did not record per-epoch metrics."}
        </p>
      </div>
    );
  }

  // Render the accuracy chart only when the payload carries accuracy at all —
  // detection architectures (FOMO / YOLO-Pro / SSD) omit it entirely.
  const hasAccuracy = metrics.some(
    (m) => m.train_accuracy != null || m.val_accuracy != null
  );

  const bestLossEpoch = bestEpochByMin(metrics, "val_loss");
  const bestAccEpoch = bestEpochByMax(metrics, "val_accuracy");

  return (
    <div className="space-y-5">
      <div>
        <h3 className="inline-flex items-center gap-1.5 text-[13px] font-semibold text-[var(--app-text)]">
          Loss
          <InfoIcon text={LOSS_INFO} />
        </h3>
        <div style={{ height: 240 }}>
          <MetricChart
            data={metrics}
            yLabel="Loss"
            bestEpoch={bestLossEpoch}
            bestLabel="Best val_loss"
            series={[
              { key: "train_loss", name: "Train loss", color: "#6366f1" },
              { key: "val_loss", name: "Validation loss", color: "#f59e0b" },
            ]}
          />
        </div>
      </div>

      {hasAccuracy && (
        <div>
          <h3 className="inline-flex items-center gap-1.5 text-[13px] font-semibold text-[var(--app-text)]">
            Accuracy
            <InfoIcon text={ACCURACY_INFO} />
          </h3>
          <div style={{ height: 240 }}>
            <MetricChart
              data={metrics}
              yLabel="Accuracy"
              bestEpoch={bestAccEpoch}
              bestLabel="Best val_acc"
              series={[
                { key: "train_accuracy", name: "Train accuracy", color: "#10b981" },
                { key: "val_accuracy", name: "Validation accuracy", color: "#14b8a6" },
              ]}
            />
          </div>
        </div>
      )}
    </div>
  );
}
