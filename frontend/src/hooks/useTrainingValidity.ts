/**
 * Single source of truth for whether an impulse's Active Model artifacts
 * should be displayed. Every training-dependent surface (deployment,
 * evaluation, model testing, live classification, post-processing, retrain)
 * derives its gate from this hook so the Training Output panel and the rest
 * of the UI can never disagree.
 *
 * Backend contract:
 *   - impulse.active_model_run_id is set ONLY on successful completion (fresh
 *     or retrain). It is NEVER cleared on start, cancel, or fail.
 *   - current_job is the latest run regardless of status, with `run_kind`
 *     ("fresh" | "retrain") and `status`.
 *
 * UI derivation (do not relax):
 *
 *   Let job        = current_job        (may be null)
 *   Let activeExists = active_model_run_id != null
 *
 *   If job is null:
 *     hasValidTrainingOutput = activeExists
 *
 *   Else if job.run_kind === "retrain":
 *     hasValidTrainingOutput = activeExists
 *     // retrain in ANY state is non-destructive — the Active Model stays
 *     // visible. A completed retrain has already moved the pointer on the
 *     // server, so this branch keeps working uniformly across statuses.
 *
 *   Else  // job.run_kind === "fresh"
 *     If job.status === "completed":
 *       hasValidTrainingOutput = activeExists  // (true; the just-completed run is now active)
 *     Else  // running | queued | cancelled | failed
 *       hasValidTrainingOutput = false
 *       // The user signaled intent to replace the model. Hide outputs until
 *       // the new run completes. active_model_run_id in the DB is untouched
 *       // (recoverability), but the UI does not surface it.
 *
 * Invalidation reuses the same `window`-event pattern the sidebar uses for
 * "impulses:changed" — no React Query, no polling, no new event bus. Callers
 * invoke `invalidateTrainingStatus()` only at the points the spec calls out
 * per page (see hook callers).
 */
import { useCallback, useEffect, useRef, useState } from "react";
import { trainingApi } from "@/utils/api";

const EVENT_NAME = "training:status-changed";

type RunKind = "fresh" | "retrain";

export interface TrainingValidity {
  hasValidTrainingOutput: boolean;
  /** The latest run regardless of status. */
  currentJob: any | null;
  /** Pointer to the run whose artifacts are the Active Model, or null. */
  activeModelRunId: string | null;
  loading: boolean;
  refresh: () => Promise<void>;
}

function deriveValidity(
  currentJob: any | null,
  activeModelRunId: string | null,
): boolean {
  const activeExists = activeModelRunId != null;

  if (currentJob == null) return activeExists;

  const runKind: RunKind = (currentJob.run_kind === "retrain" ? "retrain" : "fresh");

  if (runKind === "retrain") {
    return activeExists;
  }

  // run_kind === "fresh"
  if (currentJob.status === "completed") {
    return activeExists;
  }
  return false;
}

export function useTrainingValidity(impulseId: string | null | undefined): TrainingValidity {
  const [currentJob, setCurrentJob] = useState<any | null>(null);
  const [activeModelRunId, setActiveModelRunId] = useState<string | null>(null);
  const [loading, setLoading] = useState<boolean>(!!impulseId);
  const inflight = useRef<string | null>(null);

  const fetchStatus = useCallback(async (impId: string) => {
    if (inflight.current === impId) return;
    inflight.current = impId;
    setLoading(true);
    try {
      const { data } = await trainingApi.impulseStatus(impId);
      const job = data?.current_job ?? data?.latest_job ?? null;
      const pointer = data?.active_model_run_id ?? null;
      setCurrentJob(job);
      setActiveModelRunId(pointer);
    } catch {
      setCurrentJob(null);
      setActiveModelRunId(null);
    } finally {
      if (inflight.current === impId) inflight.current = null;
      setLoading(false);
    }
  }, []);

  const refresh = useCallback(async () => {
    if (!impulseId) return;
    inflight.current = null;
    await fetchStatus(impulseId);
  }, [impulseId, fetchStatus]);

  useEffect(() => {
    if (!impulseId) {
      setCurrentJob(null);
      setActiveModelRunId(null);
      setLoading(false);
      return;
    }
    fetchStatus(impulseId);
  }, [impulseId, fetchStatus]);

  useEffect(() => {
    if (!impulseId) return;
    const onChange = (event: Event) => {
      const detail = (event as CustomEvent<{ impulseId?: string }>).detail;
      // Refetch when the event is for our impulse, or has no impulse scope
      // (a broadcast invalidation).
      if (!detail?.impulseId || detail.impulseId === impulseId) {
        inflight.current = null;
        fetchStatus(impulseId);
      }
    };
    window.addEventListener(EVENT_NAME, onChange as EventListener);
    return () => window.removeEventListener(EVENT_NAME, onChange as EventListener);
  }, [impulseId, fetchStatus]);

  const hasValidTrainingOutput = deriveValidity(currentJob, activeModelRunId);

  return { hasValidTrainingOutput, currentJob, activeModelRunId, loading, refresh };
}

/**
 * Broadcast that training state for an impulse has changed. Subscribed pages
 * re-fetch and re-derive the gate.
 *
 * Per the spec, this is fired:
 *   - From /training (fresh): start, cancel, fail, terminal-status observation.
 *   - From /retrain: ONLY on successful completion. Never on start/cancel/fail —
 *     a retrain run in flight or aborted must not clear the Active Model from
 *     downstream pages.
 */
export function invalidateTrainingStatus(impulseId?: string) {
  if (typeof window === "undefined") return;
  window.dispatchEvent(
    new CustomEvent(EVENT_NAME, { detail: impulseId ? { impulseId } : {} }),
  );
}
