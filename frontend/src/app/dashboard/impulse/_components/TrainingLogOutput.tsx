"use client";
import Link from "next/link";
import { Check, Zap, SlidersHorizontal } from "lucide-react";
import { mapModelNamesInText } from "@/lib/model-display-names";

// First line of the backend's FeatureResolutionMismatchError message
// (_feature_resolution_mismatch_message in training_worker.py).  Matching on it
// lets a failed job offer the one action that actually fixes it — no extra
// field on the job payload, no second error channel.
const FEATURE_RESOLUTION_MISMATCH_TITLE = "Feature resolution mismatch";

interface TrainingLogOutputProps {
  job: any;
  isTraining: boolean;
  logEndRef: React.RefObject<HTMLDivElement>;
  fallbackEpochs?: number;
  emptyMessage?: string;
}

export function TrainingLogOutput({
  job,
  isTraining,
  logEndRef,
  fallbackEpochs = 60,
  emptyMessage = "No recent jobs. Click Start training to begin.",
}: TrainingLogOutputProps) {
  if (!job) {
    return (
      <div className="font-mono text-[11px] text-gray-500 italic mt-4">
        {emptyMessage}
      </div>
    );
  }

  const th = job.training_history || {};
  const architecture = (job.architecture || "").toLowerCase();
  const isFomo: boolean = th.is_fomo === true || architecture.includes("fomo");
  const isSsd: boolean = th.is_ssd === true || architecture.includes("ssd");
  const isYoloPro: boolean = th.is_yolo_pro === true || architecture.includes("yolo_pro");
  const isLossOnlyModel: boolean = isFomo || isYoloPro || isSsd;

  const reqEpochs: number =
    job.requested_epochs ?? th.requested_epochs ?? job.epochs ?? fallbackEpochs;
  const actualEpochs: number | null = job.actual_epochs ?? th.actual_epochs ?? null;

  const fomoWarmupEpochs =
    (th.warmup_epochs as number | undefined) ??
    (Number(job.fomo_version) === 2 ? 16 : 15);
  const fomoPatience = Math.max(1, Math.ceil((reqEpochs - fomoWarmupEpochs) / 2));
  const earlyStopInfo = isFomo
    ? `INFO: Early stopping patience: ${fomoPatience} epochs (val_f1 up, phase 2 only)`
    : isYoloPro
      ? `INFO: Early stopping patience: ${th.patience ?? "dynamic"} epochs (val_loss down)`
      : isSsd
        ? (th.early_stopping === true
          ? `INFO: Early stopping patience: ${th.patience ?? "dynamic"} epochs (val_loss down)`
          : "INFO: Early stopping: disabled")
        : "INFO: Early stopping patience: 20 epochs (val_accuracy up)";

  const logLines: string[] = (th.log_lines as string[] | undefined) ?? [];
  const logLinesComplete: boolean = th.log_lines_complete === true;

  const useLiveLogLines =
    logLines.length > 0 &&
    (job.status === "running" || job.status === "pending" || logLinesComplete);

  const epochSource: number[] = isLossOnlyModel
    ? (th.loss as number[]) || []
    : (th.accuracy as number[]) || [];

  // A resolution mismatch is fixed by editing the DSP image block and
  // regenerating features, so the failure gets a direct link to that page.
  const isFeatureResolutionMismatch: boolean = String(job.error_message || "")
    .trimStart()
    .startsWith(FEATURE_RESOLUTION_MISMATCH_TITLE);

  const stoppedEarlyExplicit: boolean | null = job.stopped_early ?? th.stopped_early ?? null;
  const stoppedEarlyByEpochCount: boolean =
    actualEpochs !== null && actualEpochs < reqEpochs;
  const stoppedEarly: boolean =
    actualEpochs !== null
      ? stoppedEarlyByEpochCount
      : (stoppedEarlyExplicit ?? false);

  return (
    <div className="font-mono text-[11px] text-gray-300 whitespace-pre-wrap leading-relaxed space-y-1">
      <div>Creating job... OK (ID: {job.id?.substring(0, 8) ?? "..."})</div>

      {job.status !== "pending" && (
        <div className="flex items-center gap-1">
          <Check size={12} className="text-gray-400" />
          Job scheduled at {new Date(job.created_at).toLocaleString()}
        </div>
      )}

      {(job.status === "running" ||
        job.status === "completed" ||
        job.status === "failed") && (
          <div className="flex items-center gap-1">
            <Check size={12} className="text-gray-400" />
            Job started at {new Date(job.started_at || job.created_at).toLocaleString()}
          </div>
        )}

      {job.status !== "pending" && (
        <>
          <br />
          <div>INFO: Maximum training cycles requested: {reqEpochs}</div>
          <div>{earlyStopInfo}</div>
          <div>Copying features from processing blocks...</div>
          <div>Copying features from DSP block OK</div>
          <div>Copying features from processing blocks OK</div>
          <br />
        </>
      )}

      {useLiveLogLines ? (
        logLines.map((line: string, i: number) => (
          // Display-only rename: the log line itself is untouched.
          <div key={`ll-${i}`}>{mapModelNamesInText(line)}</div>
        ))
      ) : (
        <>
          {epochSource.map((_: number, idx: number) => {
            const loss = th.loss?.[idx];
            const vLoss = th.val_loss?.[idx];
            if (isFomo) {
              const valF1 = (th.val_f1 as number[] | undefined)?.[idx];
              const phaseBreak =
                idx === fomoWarmupEpochs &&
                fomoWarmupEpochs > 0 &&
                fomoWarmupEpochs < epochSource.length;
              return (
                <div key={idx}>
                  {phaseBreak && (
                    <div className="text-brand-400 my-1">─── Phase 2: fine-tuning ───</div>
                  )}
                  <div>
                    Epoch {idx + 1}/{reqEpochs}
                    {" "}- loss: {loss?.toFixed(4) ?? "-"}
                    {" "}- val_loss: {vLoss?.toFixed(4) ?? "-"}
                    {valF1 != null ? ` - val_f1: ${valF1.toFixed(4)}` : ""}
                  </div>
                </div>
              );
            }
            if (isYoloPro || isSsd) {
              return (
                <div key={idx}>
                  Epoch {idx + 1}/{reqEpochs} - loss: {loss?.toFixed(4) ?? "-"}
                  {vLoss != null ? ` - val_loss: ${vLoss.toFixed(4)}` : ""}
                </div>
              );
            }
            const acc = th.accuracy?.[idx];
            const vAcc = th.val_accuracy?.[idx];
            return (
              <div key={idx}>
                Epoch {idx + 1}/{reqEpochs} - loss: {loss?.toFixed(4) ?? "-"} - accuracy:{" "}
                {acc?.toFixed(4) ?? "-"} - val_loss: {vLoss?.toFixed(4) ?? "-"} - val_accuracy:{" "}
                {vAcc?.toFixed(4) ?? "-"}
              </div>
            );
          })}

          {isFomo && epochSource.length > 0 && (
            <>
              {th.best_threshold != null && (
                <div className="mt-1 text-gray-400">
                  Best threshold: {Number(th.best_threshold).toFixed(2)}
                  {th.best_val_f1 != null
                    ? `  best_val_F1: ${Number(th.best_val_f1).toFixed(4)}`
                    : ""}
                  {th.best_val_f1_epoch != null
                    ? `  at epoch ${th.best_val_f1_epoch}`
                    : ""}
                </div>
              )}
              {th.best_predicted_class_distribution &&
                Object.keys(th.best_predicted_class_distribution).length > 0 && (
                  <div className="text-gray-400">
                    Predicted class distribution:{" "}
                    {Object.entries(
                      th.best_predicted_class_distribution as Record<string, number>
                    )
                      .map(([cls, cnt]) => `${cls}=${cnt}`)
                      .join("  ")}
                  </div>
                )}
            </>
          )}
        </>
      )}

      {job.status === "completed" && stoppedEarly && (() => {
        const stopReason =
          job.stop_reason ||
          (isFomo
            ? `Stopped at epoch ${actualEpochs ?? "?"}/${reqEpochs} because val_f1 did not improve.`
            : isYoloPro
              ? `Stopped at epoch ${actualEpochs ?? "?"}/${reqEpochs} because validation loss did not improve.`
              : `Stopped at epoch ${actualEpochs ?? "?"}/${reqEpochs} because validation accuracy did not improve.`);
        return (
          <div className="mt-3 p-3 rounded-lg" style={{ background: "var(--app-surface)", border: "1px solid var(--app-border)" }}>
            <p className="text-xs font-medium flex items-center gap-1.5" style={{ color: "var(--app-text)" }}>
              <Zap size={12} /> Training stopped early
            </p>
            <p className="text-[11px] mt-1 leading-relaxed" style={{ color: "var(--app-text-soft)" }}>{mapModelNamesInText(String(stopReason))}</p>
          </div>
        );
      })()}

      {job.status === "completed" && (
        <div className="flex items-center gap-1 text-emerald-400 mt-2">
          <Check size={12} /> Training complete
        </div>
      )}

      {job.status === "failed" && (
        <div className="mt-2">
          <div className="text-red-400">Training failed: {mapModelNamesInText(String(job.error_message ?? ""))}</div>
          {isFeatureResolutionMismatch && job.impulse_id && (
            <Link
              href={`/dashboard/impulse/image/parameters?impulseId=${job.impulse_id}`}
              className="inline-flex items-center gap-1.5 mt-2 px-2.5 py-1.5 rounded-md font-medium no-underline"
              style={{
                background: "var(--app-surface)",
                border: "1px solid var(--app-border)",
                color: "var(--app-text)",
              }}
            >
              <SlidersHorizontal size={12} /> Image Parameters
            </Link>
          )}
        </div>
      )}

      {job.status === "cancelled" && (
        <div className="mt-3 p-3 rounded-lg" style={{ background: "var(--app-surface)", border: "1px solid var(--app-border)" }}>
          <p className="text-xs font-medium" style={{ color: "var(--app-text)" }}>Training cancelled</p>
          <p className="text-[11px] mt-1 leading-relaxed" style={{ color: "var(--app-text-soft)" }}>
            {job.error_message || "Cancelled by user"}
          </p>
        </div>
      )}

      {isTraining && (
        <div className="animate-pulse mt-2 text-gray-400">Training...</div>
      )}

      <div ref={logEndRef} />
    </div>
  );
}
