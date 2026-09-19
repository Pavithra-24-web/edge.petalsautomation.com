import { useCallback, useEffect, useState } from "react";
import { samplesApi } from "@/utils/api";
import { SAMPLES_CHANGED } from "@/components/dashboard/labeling/labelingEvents";

/**
 * Live "unlabeled" count for the Dataset page's Labeling Queue button.
 * Backed by the Phase 1 `labeling-status-summary` endpoint, which evaluates
 * the shared `is_sample_unlabeled` predicate server-side — the same
 * definition the queue and AI Labeling's skip_labeled use, so this number
 * can't disagree with them.
 */
export function useUnlabeledCount(projectId?: string) {
  const [count, setCount] = useState(0);

  const refresh = useCallback(async () => {
    if (!projectId) return setCount(0);
    try {
      const { data } = await samplesApi.labelingStatusSummary(projectId);
      setCount(data?.unlabeled ?? 0);
    } catch {
      // Keep the last known value — never render a wrong 0 on a transient failure.
    }
  }, [projectId]);

  useEffect(() => { refresh(); }, [refresh]);

  useEffect(() => {
    window.addEventListener(SAMPLES_CHANGED, refresh);
    return () => window.removeEventListener(SAMPLES_CHANGED, refresh);
  }, [refresh]);

  useEffect(() => {
    const onVisible = () => { if (!document.hidden) refresh(); };
    document.addEventListener("visibilitychange", onVisible);
    window.addEventListener("focus", refresh);
    return () => {
      document.removeEventListener("visibilitychange", onVisible);
      window.removeEventListener("focus", refresh);
    };
  }, [refresh]);

  return { count, refresh };
}
