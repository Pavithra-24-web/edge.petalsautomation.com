"use client";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useRouter } from "next/navigation";
import toast from "react-hot-toast";
import { v4 as uuidv4 } from "uuid";
import { AlertTriangle, Loader2, X } from "lucide-react";
import { useAppStore } from "@/store/appStore";
import { samplesApi, labelsApi } from "@/utils/api";
import { useLabelingQueue } from "@/hooks/useLabelingQueue";
import { fetchSampleUrl, prefetchSample } from "../sampleImageCache";
import { buildLabelColorMap, colorForLabelFromMap } from "../labelColors";
import { isSampleUnlabeled } from "../labelingStatus";
import QueueHeader from "./QueueHeader";
import QueueSidebar from "./QueueSidebar";
import QueueCanvas from "./QueueCanvas";
import QueueToolbar from "./QueueToolbar";
import QueueEmptyState from "./QueueEmptyState";

// Labeling Queue plan, Phase 4 — the controller. Owns the queue snapshot
// (via useLabelingQueue), the current sample's full record, the presigned
// image URL, and the save flow (§2.2). <AnnotationEditor/> (via
// <QueueCanvas/>) does all of the drawing/shortcut/undo/label-modal work
// unmodified — this component only supplies its props and persists onChange.

function normalizeBoxLabel(value: any) {
  const text = String(value ?? "").trim();
  if (!text) return "";
  const lowered = text.toLowerCase();
  if (lowered === "unlabeled" || lowered === "unlabelled" || lowered === "unknown") return "";
  return text;
}

function normalizeBoxGeometry(b: any): any {
  if (!b || typeof b !== "object") return b;
  const out = { ...b };
  if (out.w === undefined && out.width !== undefined) { out.w = out.width; delete out.width; }
  if (out.h === undefined && out.height !== undefined) { out.h = out.height; delete out.height; }
  return out;
}

function baseName(filename: string) {
  return filename.split("/").pop()?.split("\\").pop() || filename;
}

export default function LabelingQueue() {
  const { activeProject } = useAppStore();
  const projectId = activeProject?.id;
  const router = useRouter();
  const queue = useLabelingQueue(projectId);
  const current = queue.current;

  const [labels, setLabels] = useState<any[]>([]);
  const [currentSample, setCurrentSample] = useState<any>(null);
  const [sampleLoading, setSampleLoading] = useState(false);
  const [sampleLoadError, setSampleLoadError] = useState(false);
  const [imageUrl, setImageUrl] = useState<string | null>(null);
  const [reloadNonce, setReloadNonce] = useState(0);
  const [saveLabelsBusy, setSaveLabelsBusy] = useState(false);
  const [saveError, setSaveError] = useState<string | null>(null);
  const [blockedMessage, setBlockedMessage] = useState<string | null>(null);
  const [confirmExitOpen, setConfirmExitOpen] = useState(false);

  const currentSampleRef = useRef<any>(null);
  const saveErrorRef = useRef<string | null>(null);
  const saveLabelsBusyRef = useRef(false);
  const pendingSaveRef = useRef<Promise<any> | null>(null);
  const lastEditRef = useRef<{ boxes: any[]; primary?: { id?: string; name?: string } | null } | null>(null);
  const missingTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  // Tracks which sample id is on screen right now, so a save that resolves
  // after the user has already navigated elsewhere (Previous/Next don't
  // wait for a save — §2.10) never overwrites the *new* current sample's
  // state with a stale response.
  const currentIdRef = useRef<string | null>(null);
  useEffect(() => { currentIdRef.current = current?.id ?? null; }, [current?.id]);

  const setSaveErrorBoth = useCallback((msg: string | null) => {
    saveErrorRef.current = msg;
    setSaveError(msg);
  }, []);

  // ── Project labels (also feeds the editor's on-demand label creation) ──
  useEffect(() => {
    if (!projectId) return;
    labelsApi.list(projectId).then(({ data }) => setLabels(data)).catch(() => setLabels([]));
  }, [projectId]);

  const labelColorMap = useMemo(() => {
    const counts: Record<string, number> = {};
    for (const l of labels) counts[l.name] = 1;
    return buildLabelColorMap(counts);
  }, [labels]);
  const colorForLabel = useCallback((label: string) => colorForLabelFromMap(labelColorMap, label), [labelColorMap]);

  const createProjectLabel = useCallback(async (name: string) => {
    if (!projectId) return null;
    try {
      const { data } = await labelsApi.create({ project_id: projectId, name });
      setLabels(prev => [...prev, data]);
      return { id: data.id, name: data.name };
    } catch {
      toast.error("Failed to create new label");
      return null;
    }
  }, [projectId]);

  // ── Load the current sample's full record on demand (§3.C) ─────────────
  useEffect(() => {
    if (missingTimerRef.current) { clearTimeout(missingTimerRef.current); missingTimerRef.current = null; }
    setBlockedMessage(null);
    setSaveErrorBoth(null);
    if (!current) { setCurrentSample(null); currentSampleRef.current = null; return; }
    let cancelled = false;
    setSampleLoading(true);
    setSampleLoadError(false);
    samplesApi.get(current.id).then(({ data }) => {
      if (cancelled) return;
      setCurrentSample(data);
      currentSampleRef.current = data;
    }).catch((err: any) => {
      if (cancelled) return;
      if (err?.response?.status === 404) {
        queue.markMissing(current.id);
        missingTimerRef.current = setTimeout(() => queue.next(), 1500);
      } else {
        setSampleLoadError(true);
      }
    }).finally(() => { if (!cancelled) setSampleLoading(false); });
    return () => { cancelled = true; };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [current?.id, reloadNonce]);

  // ── Presigned image URL + neighbor prefetch (§3.C) ─────────────────────
  useEffect(() => {
    if (!current) { setImageUrl(null); return; }
    let cancelled = false;
    setImageUrl(null);
    fetchSampleUrl(current.id).then(u => { if (!cancelled) setImageUrl(u); }).catch(() => { if (!cancelled) setImageUrl(null); });
    return () => { cancelled = true; };
  }, [current?.id]);

  useEffect(() => {
    const next = queue.items[queue.index + 1];
    const prev = queue.items[queue.index - 1];
    for (const neighbor of [next, prev]) {
      if (neighbor && neighbor.status !== "missing") prefetchSample(neighbor.id);
    }
  }, [queue.index, queue.items]);

  // ── Save (§2.2) — onChange persists quietly; "Save Labels" flushes,
  //    validates against the shared predicate, then commits-and-advances ──
  const persistBoxes = useCallback((sampleId: string, boxes: any[], primary?: { id?: string; name?: string } | null) => {
    const normalizedBoxes = boxes.map((box: any) => {
      const normalizedLabel = normalizeBoxLabel(box?.label);
      return { ...box, label: normalizedLabel || null, label_id: normalizedLabel ? box?.label_id : null };
    });
    const base = currentSampleRef.current;
    const meta = { ...(base?.extra_metadata || {}), boundingBoxes: normalizedBoxes };
    const run = (async () => {
      if (primary?.id && base?.label_id !== primary.id) {
        await samplesApi.assignLabel(sampleId, primary.id);
      }
      await samplesApi.update(sampleId, { extra_metadata: meta });
      return {
        ...base,
        label_id: primary?.id || base?.label_id,
        label_name: primary?.name || base?.label_name,
        extra_metadata: meta,
      };
    })()
      .then(updated => {
        // Only touch shared state if the user hasn't since moved on to a
        // different sample — Previous/Next don't wait for a save (§2.10),
        // so a slow response can land well after the pointer has moved.
        if (currentIdRef.current === sampleId) {
          currentSampleRef.current = updated;
          setCurrentSample(updated);
          setSaveErrorBoth(null);
        }
        return updated;
      })
      .catch(err => {
        if (currentIdRef.current === sampleId) {
          setSaveErrorBoth("Backend sync failed. Your last edit may not be saved.");
          // Reseed from the server (§4.D) — never leave the editor showing
          // local boxes that didn't actually land.
          samplesApi.get(sampleId).then(({ data }) => {
            if (currentIdRef.current !== sampleId) return;
            currentSampleRef.current = data;
            setCurrentSample(data);
            setReloadNonce(n => n + 1);
          }).catch(() => { /* keep showing the last known-good state */ });
        }
        throw err;
      })
      .finally(() => {
        if (pendingSaveRef.current === run) pendingSaveRef.current = null;
      });
    pendingSaveRef.current = run;
    return run;
  }, [setSaveErrorBoth]);

  const handleEditorChange = useCallback((boxes: any[], primary?: { id?: string; name?: string } | null) => {
    if (!current) return;
    lastEditRef.current = { boxes, primary };
    persistBoxes(current.id, boxes, primary).catch(() => { /* surfaced via the error banner */ });
  }, [current, persistBoxes]);

  const handleRetry = useCallback(() => {
    if (!current || !lastEditRef.current) { setSaveErrorBoth(null); return; }
    persistBoxes(current.id, lastEditRef.current.boxes, lastEditRef.current.primary).catch(() => {});
  }, [current, persistBoxes, setSaveErrorBoth]);

  const handleSkip = useCallback(() => {
    setSaveErrorBoth(null);
    queue.next();
  }, [queue.next, setSaveErrorBoth]);

  const handleSaveLabels = useCallback(async () => {
    if (!current || saveLabelsBusyRef.current) return;
    saveLabelsBusyRef.current = true;
    setSaveLabelsBusy(true);
    setBlockedMessage(null);
    try {
      if (pendingSaveRef.current) {
        try { await pendingSaveRef.current; } catch { return; }
      }
      if (saveErrorRef.current) return;
      const sampleToCheck = currentSampleRef.current;
      if (!sampleToCheck || isSampleUnlabeled(sampleToCheck)) {
        setBlockedMessage("Draw at least one box, or mark this image as Background from the Dataset page.");
        return;
      }
      queue.markSaved(current.id);
    } finally {
      saveLabelsBusyRef.current = false;
      setSaveLabelsBusy(false);
    }
  }, [current, queue.markSaved]);

  // ── Exit-queue confirmation (§2.9) ──────────────────────────────────────
  const handleExitClick = useCallback(() => {
    if (queue.isFinished) router.push("/dashboard/data/dataset");
    else setConfirmExitOpen(true);
  }, [queue.isFinished, router]);
  const handleStay = useCallback(() => setConfirmExitOpen(false), []);
  const handleConfirmExit = useCallback(() => {
    setConfirmExitOpen(false);
    router.push("/dashboard/data/dataset");
  }, [router]);

  // ── Keyboard shortcuts (§2.10) — inert while typing (label modal input,
  //    or any other focused field) and while there's nothing to edit ──────
  const readyToEdit = queue.status === "ready" && !queue.isEmpty && !queue.isFinished;
  useEffect(() => {
    if (!readyToEdit) return;
    function isTypingTarget(el: Element | null) {
      if (!el) return false;
      const tag = el.tagName;
      return tag === "INPUT" || tag === "TEXTAREA" || (el as HTMLElement).isContentEditable;
    }
    function onKeyDown(e: KeyboardEvent) {
      if (isTypingTarget(document.activeElement)) return;
      if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "s") {
        e.preventDefault();
        handleSaveLabels();
      } else if (e.key === "ArrowRight") {
        queue.next();
      } else if (e.key === "ArrowLeft") {
        queue.prev();
      } else if (e.key === "Escape") {
        setConfirmExitOpen(true);
      }
    }
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [readyToEdit, handleSaveLabels, queue.next, queue.prev]);

  const initialBoxes = useMemo(() => {
    if (!currentSample) return [];
    return (currentSample.extra_metadata?.boundingBoxes || []).map((b: any) => ({
      ...normalizeBoxGeometry(b),
      id: b.id || uuidv4(),
    }));
  }, [currentSample]);

  const resetKey = current ? `sample:${current.id}:${reloadNonce}` : "";

  // ── Render ───────────────────────────────────────────────────────────
  if (!projectId) return null;

  if (queue.status === "loading" || queue.status === "idle") {
    return (
      <div className="labeling-queue-page-status">
        <Loader2 size={22} className="animate-spin" />
        <p>Loading the labeling queue…</p>
      </div>
    );
  }

  if (queue.status === "error") {
    return (
      <div className="labeling-queue-page-status">
        <AlertTriangle size={22} className="text-red-500" />
        <p>{queue.error}</p>
        <button type="button" className="data-acq-btn-primary" onClick={queue.reload}>Retry</button>
      </div>
    );
  }

  if (queue.isEmpty) {
    return (
      <div className="dataset-page data-acq-page w-full space-y-6">
        <QueueEmptyState variant={queue.totalSamplesInProject === 0 ? "no-samples" : "nothing-to-label"} />
      </div>
    );
  }

  if (queue.isFinished) {
    const splitTotals = { training: 0, testing: 0, postprocessing: 0 };
    for (const item of queue.completed) splitTotals[item.sample_type]++;
    return (
      <div className="dataset-page data-acq-page w-full space-y-6">
        <QueueEmptyState
          variant="session-complete"
          labeledThisSession={queue.completed.length}
          splitTotals={splitTotals}
        />
      </div>
    );
  }

  return (
    <div className="dataset-page data-acq-page w-full space-y-4">
      <QueueHeader
        filename={current ? baseName(current.filename) : null}
        split={current?.sample_type ?? null}
        position={queue.index + 1}
        total={queue.items.length}
        remainingCount={queue.remaining.length}
        progressPct={queue.items.length ? (queue.completed.length / queue.items.length) * 100 : 0}
        confirmOpen={confirmExitOpen}
        isFinished={queue.isFinished}
        onExitClick={handleExitClick}
        onStay={handleStay}
        onConfirmExit={handleConfirmExit}
      />

      {queue.pendingNewItems.length > 0 && (
        <div className="labeling-queue-new-banner">
          <span>
            {queue.pendingNewItems.length} new unlabeled sample{queue.pendingNewItems.length === 1 ? "" : "s"} found
          </span>
          <div className="labeling-queue-new-banner-actions">
            <button type="button" onClick={queue.addPendingSamples}>Add to queue</button>
            <button type="button" onClick={queue.dismissPendingSamples} aria-label="Dismiss">
              <X size={14} />
            </button>
          </div>
        </div>
      )}

      <div className="flex flex-col lg:flex-row items-stretch gap-4 lg:h-[calc(100vh-220px)] w-full min-h-0">
        <QueueSidebar items={queue.items} currentIndex={queue.index} onSelect={queue.goto} />

        <section className="dataset-main-panel card !p-0 overflow-hidden flex flex-col min-h-0 lg:flex-1">
          <div className="flex-1 flex flex-col min-h-0">
            {current && (
              <QueueCanvas
                filename={current.filename}
                imageUrl={imageUrl}
                loading={sampleLoading}
                loadError={sampleLoadError}
                missing={current.status === "missing"}
                initialBoxes={initialBoxes}
                resetKey={resetKey}
                labels={labels}
                onChange={handleEditorChange}
                onCreateLabel={createProjectLabel}
                colorForLabel={colorForLabel}
              />
            )}
          </div>
          <QueueToolbar
            onPrev={queue.prev}
            onNext={queue.next}
            onSave={handleSaveLabels}
            canPrev={queue.index > 0}
            canNext={queue.index < queue.items.length - 1}
            saving={saveLabelsBusy}
            blockedMessage={blockedMessage}
            saveError={saveError}
            onRetry={handleRetry}
            onSkip={handleSkip}
          />
        </section>
      </div>
    </div>
  );
}
