"use client";
import { useCallback, useEffect, useReducer, useRef } from "react";
import { usePathname, useRouter, useSearchParams } from "next/navigation";
import { samplesApi } from "@/utils/api";

// ─── Labeling Queue controller state ───────────────────────────────────────
// Labeling Queue plan, Phase 4 §4.B/§4.C. Holds one ordered snapshot of
// `unlabeled-queue` ids for the session; every mutation removes/updates an
// entry in place (never re-sorts), so a completed sample never reshuffles
// the remaining items (§2.3).

export type QueueSplit = "training" | "testing" | "postprocessing";
export type QueueItemStatus = "remaining" | "completed" | "missing";

export type QueueItem = {
  id: string;
  filename: string;
  sample_type: QueueSplit;
  created_at: string;
  status: QueueItemStatus;
};

export type QueueStatus = "idle" | "loading" | "ready" | "error";

export type QueueState = {
  status: QueueStatus;
  items: QueueItem[];
  index: number;
  error: string | null;
  totalSamplesInProject: number;
  pendingNewItems: QueueItem[];
};

export type Action =
  | { type: "LOAD_START" }
  | { type: "LOAD_SNAPSHOT"; items: QueueItem[]; resumeId: string | null; totalSamplesInProject: number }
  | { type: "LOAD_ERROR"; error: string }
  | { type: "GOTO"; index: number }
  | { type: "NEXT" }
  | { type: "PREV" }
  | { type: "MARK_SAVED"; id: string }
  | { type: "MARK_MISSING"; id: string }
  | { type: "RECONCILE"; liveItems: QueueItem[] }
  | { type: "APPEND_PENDING" }
  | { type: "DISMISS_PENDING" };

export const initialState: QueueState = {
  status: "idle",
  items: [],
  index: 0,
  error: null,
  totalSamplesInProject: 0,
  pendingNewItems: [],
};

/** First "remaining" item strictly after `from`, wrapping once to the start
 *  (§4.C) — a user who jumps backward to fix something is still carried to
 *  outstanding work rather than dead-ending on a completed item. */
export function firstRemainingAfter(items: QueueItem[], from: number): number {
  for (let i = from + 1; i < items.length; i++) if (items[i].status === "remaining") return i;
  for (let i = 0; i <= from && i < items.length; i++) if (items[i].status === "remaining") return i;
  return -1;
}

export function queueReducer(state: QueueState, action: Action): QueueState {
  switch (action.type) {
    case "LOAD_START":
      return { ...initialState, status: "loading" };
    case "LOAD_SNAPSHOT": {
      let index = 0;
      if (action.resumeId) {
        const idx = action.items.findIndex(i => i.id === action.resumeId);
        if (idx !== -1) index = idx;
      }
      return {
        ...state,
        status: "ready",
        items: action.items,
        index,
        error: null,
        totalSamplesInProject: action.totalSamplesInProject,
        pendingNewItems: [],
      };
    }
    case "LOAD_ERROR":
      return { ...state, status: "error", error: action.error };
    case "GOTO":
      if (action.index < 0 || action.index >= state.items.length) return state;
      return { ...state, index: action.index };
    case "NEXT":
      return state.index < state.items.length - 1 ? { ...state, index: state.index + 1 } : state;
    case "PREV":
      return state.index > 0 ? { ...state, index: state.index - 1 } : state;
    case "MARK_SAVED": {
      const items = state.items.map(i => (i.id === action.id ? { ...i, status: "completed" as const } : i));
      const next = firstRemainingAfter(items, state.index);
      return { ...state, items, index: next === -1 ? state.index : next };
    }
    case "MARK_MISSING": {
      // Status only — the controller advances the pointer itself after a
      // short delay so the "this image was deleted" message is visible
      // before the queue moves on (Appendix A).
      const items = state.items.map(i => (i.id === action.id ? { ...i, status: "missing" as const } : i));
      return { ...state, items };
    }
    case "RECONCILE": {
      const liveIds = new Set(action.liveItems.map(i => i.id));
      const existingIds = new Set(state.items.map(i => i.id));
      const items = state.items.map(i => {
        if (i.status === "missing") return i;
        return liveIds.has(i.id) ? { ...i, status: "remaining" as const } : { ...i, status: "completed" as const };
      });
      const pendingNewItems = action.liveItems.filter(i => !existingIds.has(i.id));
      return { ...state, items, pendingNewItems };
    }
    case "APPEND_PENDING": {
      if (state.pendingNewItems.length === 0) return state;
      return { ...state, items: [...state.items, ...state.pendingNewItems], pendingNewItems: [] };
    }
    case "DISMISS_PENDING":
      return { ...state, pendingNewItems: [] };
    default:
      return state;
  }
}

const RESUME_TTL_MS = 24 * 60 * 60 * 1000;

function resumeKey(projectId: string) {
  return `labelingQueue:${projectId}`;
}

function readResumePointer(projectId: string): string | null {
  try {
    const raw = sessionStorage.getItem(resumeKey(projectId));
    if (!raw) return null;
    const parsed = JSON.parse(raw);
    if (!parsed?.sampleId || !parsed?.savedAt) return null;
    if (Date.now() - parsed.savedAt > RESUME_TTL_MS) return null;
    return String(parsed.sampleId);
  } catch {
    return null;
  }
}

function writeResumePointer(projectId: string, sampleId: string) {
  try {
    sessionStorage.setItem(resumeKey(projectId), JSON.stringify({ sampleId, savedAt: Date.now() }));
  } catch {
    /* best-effort — a lost pointer just falls back to item 0 */
  }
}

export function useLabelingQueue(projectId: string | undefined) {
  const [state, dispatch] = useReducer(queueReducer, initialState);
  const router = useRouter();
  const pathname = usePathname();
  const searchParams = useSearchParams();
  const loadTokenRef = useRef(0);
  // The URL/sessionStorage pointer should only be read once, when the
  // snapshot is first built — after that the queue's own pointer is the
  // source of truth, so a stale `?sample=` left over from a previous visit
  // never fights the reducer on every render. Set just before each
  // `buildSnapshot` call (see the effect below).
  const initialResumeRef = useRef<string | null>(null);

  const buildSnapshot = useCallback(async (pid: string) => {
    const token = ++loadTokenRef.current;
    dispatch({ type: "LOAD_START" });
    try {
      const [{ data: queueData }, { data: summaryData }] = await Promise.all([
        samplesApi.unlabeledQueue(pid),
        samplesApi.labelingStatusSummary(pid),
      ]);
      if (loadTokenRef.current !== token) return;
      const seen = new Set<string>();
      const items: QueueItem[] = [];
      for (const it of queueData.items || []) {
        if (seen.has(it.id)) continue;
        seen.add(it.id);
        items.push({
          id: it.id,
          filename: it.filename,
          sample_type: it.sample_type,
          created_at: it.created_at,
          status: "remaining",
        });
      }
      dispatch({
        type: "LOAD_SNAPSHOT",
        items,
        resumeId: initialResumeRef.current || null,
        totalSamplesInProject: summaryData?.total ?? items.length,
      });
    } catch {
      if (loadTokenRef.current !== token) return;
      dispatch({ type: "LOAD_ERROR", error: "Couldn't load the labeling queue. Check your connection and retry." });
    }
  }, []);

  useEffect(() => {
    initialResumeRef.current = null;
    if (!projectId) return;
    initialResumeRef.current = searchParams.get("sample") || readResumePointer(projectId) || "";
    buildSnapshot(projectId);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [projectId, buildSnapshot]);

  const current = state.items[state.index] || null;

  // Mirror the current item into sessionStorage + the URL (§2.6) so a
  // refresh, a navigation away-and-back, or a browser restart in the same
  // tab resumes here instead of restarting from item 0.
  useEffect(() => {
    if (!projectId || !current) return;
    writeResumePointer(projectId, current.id);
    if (searchParams.get("sample") !== current.id) {
      const params = new URLSearchParams(searchParams.toString());
      params.set("sample", current.id);
      router.replace(`${pathname}?${params.toString()}`, { scroll: false });
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [projectId, current?.id]);

  const reconcile = useCallback(async () => {
    if (!projectId) return;
    try {
      const { data } = await samplesApi.unlabeledQueue(projectId);
      const liveItems: QueueItem[] = (data.items || []).map((it: any) => ({
        id: it.id,
        filename: it.filename,
        sample_type: it.sample_type,
        created_at: it.created_at,
        status: "remaining" as const,
      }));
      dispatch({ type: "RECONCILE", liveItems });
    } catch {
      /* keep the local snapshot; the next focus/visibility trigger retries */
    }
  }, [projectId]);

  // Multi-tab / multi-session reconciliation (§2.7): each tab keeps its own
  // pointer, but self-corrects against live server state on focus/visibility
  // return, so a sample labeled in another tab flips to "completed" here
  // without a manual refresh.
  useEffect(() => {
    if (state.status !== "ready") return;
    const onFocusish = () => {
      if (!document.hidden) reconcile();
    };
    window.addEventListener("focus", onFocusish);
    document.addEventListener("visibilitychange", onFocusish);
    return () => {
      window.removeEventListener("focus", onFocusish);
      document.removeEventListener("visibilitychange", onFocusish);
    };
  }, [state.status, reconcile]);

  const remaining = state.items.filter(i => i.status === "remaining");
  const completed = state.items.filter(i => i.status === "completed");

  // Stable identities (dispatch never changes) so consumers' useCallback/
  // useEffect dependency arrays don't churn on every unrelated re-render.
  const goto = useCallback((index: number) => dispatch({ type: "GOTO", index }), []);
  const next = useCallback(() => dispatch({ type: "NEXT" }), []);
  const prev = useCallback(() => dispatch({ type: "PREV" }), []);
  const markSaved = useCallback((id: string) => dispatch({ type: "MARK_SAVED", id }), []);
  const markMissing = useCallback((id: string) => dispatch({ type: "MARK_MISSING", id }), []);
  const addPendingSamples = useCallback(() => dispatch({ type: "APPEND_PENDING" }), []);
  const dismissPendingSamples = useCallback(() => dispatch({ type: "DISMISS_PENDING" }), []);
  const reload = useCallback(() => { if (projectId) buildSnapshot(projectId); }, [projectId, buildSnapshot]);

  return {
    status: state.status,
    error: state.error,
    items: state.items,
    index: state.index,
    current,
    remaining,
    completed,
    isEmpty: state.status === "ready" && state.items.length === 0,
    isFinished: state.status === "ready" && state.items.length > 0 && remaining.length === 0,
    totalSamplesInProject: state.totalSamplesInProject,
    pendingNewItems: state.pendingNewItems,
    goto,
    next,
    prev,
    markSaved,
    markMissing,
    addPendingSamples,
    dismissPendingSamples,
    reconcile,
    reload,
  };
}

export type UseLabelingQueueReturn = ReturnType<typeof useLabelingQueue>;
