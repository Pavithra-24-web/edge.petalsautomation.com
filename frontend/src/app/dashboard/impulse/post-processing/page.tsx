"use client";
import { useCallback, useEffect, useLayoutEffect, useRef, useState } from "react";
import { useAppStore } from "@/store/appStore";
import { postProcessingApi, samplesApi, trainedModelsApi, dspApi } from "@/utils/api";
import { useTrainingValidity } from "@/hooks/useTrainingValidity";
import {
  DEFAULT_SETTINGS,
  INITIAL_VIDEO_STATE,
  extractApiError,
  mergeWithDefaults,
  shouldStopPolling,
  serializeForPersist,
  shouldRestoreAsInProgress,
  resolvePostProcessingModelSupport,
  type PPSettings,
  type VideoJobState,
} from "@/lib/post-processing-helpers";
import {
  AlertCircle,
  ChevronDown,
  ChevronUp,
  Filter as FilterIcon,
  HelpCircle,
  PartyPopper,
  PlayCircle,
  RefreshCw,
  Save,
  ScanLine,
  Sliders,
  Upload,
  Video,
  XCircle,
} from "lucide-react";
import toast from "react-hot-toast";
import ImpulseNotReady from "@/components/dashboard/ImpulseNotReady";
import MotionPhasePending from "@/components/dashboard/MotionPhasePending";

const _VIDEO_EXTS = [".mp4", ".avi", ".mov"];

function isVideoSample(filename: string): boolean {
  const dot = filename.lastIndexOf(".");
  return dot !== -1 && _VIDEO_EXTS.includes(filename.slice(dot).toLowerCase());
}

function HelpTooltip({ text }: { text: string }) {
  const triggerRef = useRef<HTMLButtonElement>(null);
  const [open, setOpen] = useState(false);
  const [position, setPosition] = useState({ top: 0, left: 0, arrowLeft: 0 });

  const updatePosition = useCallback(() => {
    const trigger = triggerRef.current;
    if (!trigger) return;

    const rect = trigger.getBoundingClientRect();
    const tooltipWidth = 240;
    const gutter = 16;
    const idealLeft = rect.left + rect.width / 2 - tooltipWidth / 2;
    const left = Math.min(
      Math.max(gutter, idealLeft),
      window.innerWidth - tooltipWidth - gutter
    );

    setPosition({
      top: rect.top - 12,
      left,
      arrowLeft: rect.left + rect.width / 2 - left,
    });
  }, []);

  useLayoutEffect(() => {
    if (!open) return;

    updatePosition();
    window.addEventListener("resize", updatePosition);
    window.addEventListener("scroll", updatePosition, true);
    return () => {
      window.removeEventListener("resize", updatePosition);
      window.removeEventListener("scroll", updatePosition, true);
    };
  }, [open, updatePosition]);

  return (
    <span className="relative inline-flex items-center">
      <button
        ref={triggerRef}
        type="button"
        tabIndex={0}
        aria-label={text}
        onMouseEnter={() => setOpen(true)}
        onMouseLeave={() => setOpen(false)}
        onFocus={() => setOpen(true)}
        onBlur={() => setOpen(false)}
        className="ml-1 inline-flex cursor-help items-center justify-center text-gray-500 transition-colors hover:text-gray-300 focus:text-gray-300 focus:outline-none"
      >
        <HelpCircle size={12} />
      </button>
      {open && (
        <span
          role="tooltip"
          className="pointer-events-none fixed z-50 w-60 -translate-y-full rounded-lg bg-[#373768] px-3 py-2 text-left text-xs leading-relaxed text-white shadow-2xl"
          style={{
            top: position.top,
            left: position.left,
          }}
        >
          {text}
          <span
            className="absolute top-full h-2.5 w-2.5 -translate-x-1/2 -translate-y-1/2 rotate-45 bg-[#373768]"
            style={{ left: position.arrowLeft }}
          />
        </span>
      )}
    </span>
  );
}

export default function PostProcessingPage() {
  const { activeProject, activeImpulse, ppPage, setPPPage, clearPPPage } = useAppStore();
  const {
    hasValidTrainingOutput,
    loading: trainingValidityLoading,
  } = useTrainingValidity(activeImpulse?.id ?? null);

  const [settings, setSettings] = useState<PPSettings>(DEFAULT_SETTINGS);
  const [loading, setLoading] = useState(false);
  const [saving, setSaving] = useState(false);
  // null = still checking; final state is derived from the latest trained artifact.
  const [modelSupportStatus, setModelSupportStatus] = useState<"ready" | "no_model" | "unsupported" | null>(null);
  const [modelSupportMessage, setModelSupportMessage] = useState<string | null>(null);

  const [labels, setLabels] = useState<{ id: string; name: string }[]>([]);
  const [labelsLoading, setLabelsLoading] = useState(false);

  // Project video samples (Phase 7)
  const [projectVideos, setProjectVideos] = useState<{ id: string; filename: string }[]>([]);
  const [videosLoading, setVideosLoading] = useState(false);
  const [selectedVideoId, setSelectedVideoId] = useState<string | null>(null);

  // Video job
  const [videoJob, setVideoJob] = useState<VideoJobState>(INITIAL_VIDEO_STATE);
  const [cancelling, setCancelling] = useState(false);
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const [videoPlayError, setVideoPlayError] = useState(false);

  // ── Restore / save guards ────────────────────────────────────────────────────
  // hasRestoredRef gates the save effect so initial render never wipes persisted data.
  // restoredScopeRef tracks which project/impulse page state was last restored.
  const hasRestoredRef = useRef(false);
  const restoredScopeRef = useRef<string | null>(null);
  const activeImpulseId = activeImpulse?.id ?? null;
  const ppPageKey = activeProject
    ? `${activeProject.id}:${activeImpulseId ?? "no-impulse"}`
    : null;

  // ── Bootstrap ────────────────────────────────────────────────────────────────
  useEffect(() => {
    if (!activeProject || !ppPageKey) return;

    // Reset restore gate when the user switches project or impulse
    if (restoredScopeRef.current !== ppPageKey) {
      restoredScopeRef.current = ppPageKey;
      hasRestoredRef.current = false;
      stopPolling();
      setVideoJob(INITIAL_VIDEO_STATE);
      setSelectedVideoId(null);
      setModelSupportStatus(null);
      setModelSupportMessage(null);
    }

    loadSettings();
    loadLabels();
    loadProjectVideos();
  }, [activeProject, ppPageKey]); // eslint-disable-line react-hooks/exhaustive-deps

  // Refresh model-support status whenever the latest training run's status
  // changes (cancel/fail/complete/new-run-start). Keeps modelSupportStatus
  // aligned with the artifacts that actually belong to the latest run.
  useEffect(() => {
    if (!activeProject || !activeImpulseId) return;
    loadSettings();
  }, [hasValidTrainingOutput]); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => () => { stopPolling(); }, []); // eslint-disable-line react-hooks/exhaustive-deps

  // ── Persist state whenever it changes (gated: only after restore is done) ───
  useEffect(() => {
    if (!activeProject || !ppPageKey || !hasRestoredRef.current) return;
    setPPPage(ppPageKey, serializeForPersist(selectedVideoId, videoJob));
  }, [ // eslint-disable-line react-hooks/exhaustive-deps
    selectedVideoId,
    videoJob.status,
    videoJob.jobId,
    videoJob.outputUrl,
    videoJob.errorMessage,
    ppPageKey,
  ]);

  // ── Data loaders ─────────────────────────────────────────────────────────────
  async function loadSettings() {
    if (!activeProject) return;
    setLoading(true);
    setModelSupportStatus(null);
    setModelSupportMessage(null);
    try {
      // Run settings fetch and artifact check in parallel.
      // listAllForImpulse returns [{job, models[]}] — checking models.length > 0
      // confirms real TrainedModel artifacts exist, independent of job status.
      // A later failed/cancelled retrain does not hide artifacts from a prior run.
      const [settingsRes, artifactsRes] = await Promise.allSettled([
        postProcessingApi.getSettings(activeProject.id, activeImpulseId ?? undefined),
        activeImpulseId
          ? trainedModelsApi.listAllForImpulse(activeImpulseId)
          : Promise.resolve(null),
      ]);

      if (settingsRes.status === "fulfilled") {
        setSettings(mergeWithDefaults(settingsRes.value.data));
      } else {
        toast.error(extractApiError((settingsRes as PromiseRejectedResult).reason) || "Failed to load post-processing settings");
      }

      const entries =
        artifactsRes.status === "fulfilled" && artifactsRes.value !== null
          ? ((artifactsRes.value as any).data ?? [])
          : [];
      if (!activeImpulseId) {
        setModelSupportStatus("no_model");
        setModelSupportMessage("Select an impulse before using video post-processing preview.");
      } else {
        const support = resolvePostProcessingModelSupport(entries);
        setModelSupportStatus(support.status);
        setModelSupportMessage(support.message);
      }
    } finally {
      setLoading(false);
    }
  }

  async function loadLabels() {
    if (!activeProject) return;
    // No impulse selected yet — leave class filter empty rather than showing
    // every project label (incl. unused/folder labels). The dataset summary
    // endpoint is what computes the active class set.
    if (!activeImpulseId) {
      setLabels([]);
      return;
    }
    setLabelsLoading(true);
    try {
      const { data } = await dspApi.datasetSummary(activeImpulseId);
      const classNames: string[] = Array.isArray(data?.class_names) ? data.class_names : [];
      setLabels(classNames.map((name) => ({ id: name, name })));
    } catch {
      setLabels([]);
    } finally {
      setLabelsLoading(false);
    }
  }

  async function loadProjectVideos() {
    if (!activeProject) return;
    setVideosLoading(true);
    try {
      const { data } = await samplesApi.list(activeProject.id, { limit: 1000 });
      const videos = (data?.items ?? []).filter((s: { filename: string }) =>
        isVideoSample(s.filename),
      );
      setProjectVideos(videos);

      // Restore persisted page state once per project mount
      if (!hasRestoredRef.current) {
        await restorePersistedState(activeProject.id, ppPageKey!, videos);
        hasRestoredRef.current = true;
      }
    } catch {
      setProjectVideos([]);
      hasRestoredRef.current = true; // allow saving even if video load fails
    } finally {
      setVideosLoading(false);
    }
  }

  // ── Restore persisted page state ─────────────────────────────────────────────
  async function restorePersistedState(
    projectId: string,
    pageKey: string,
    videos: { id: string; filename: string }[],
  ) {
    const saved = ppPage[pageKey];
    if (!saved) return;

    // 1. Restore selected video (clear silently if sample no longer exists)
    if (saved.selectedSampleId) {
      if (videos.some(v => v.id === saved.selectedSampleId)) {
        setSelectedVideoId(saved.selectedSampleId);
      } else {
        setPPPage(pageKey, { ...saved, selectedSampleId: null });
      }
    }

    // 2. Restore job state
    const { jobId, status } = saved;
    if (!jobId || !status) return;

    if (status === "failed") {
      setVideoJob({ status: "failed", jobId, outputUrl: null, errorMessage: saved.errorMessage });
      return;
    }

    if (status === "cancelled") {
      setVideoJob({ status: "cancelled", jobId, outputUrl: null, errorMessage: null });
      return;
    }

    // Complete (any savedOutputUrl) and in-progress both go through the backend.
    // Presigned URLs are short-lived and must never be used directly from storage;
    // complete jobs re-fetch a fresh URL on every restore. This also handles
    // complete+null-URL (partial state) which would otherwise fall through silently.
    if (status === "complete" || shouldRestoreAsInProgress(status as any)) {
      setVideoJob({ status: "processing", jobId, outputUrl: null, errorMessage: null });
      try {
        const { data } = await postProcessingApi.getProcessingJob(projectId, jobId);
        if (shouldStopPolling(data.status)) {
          if (data.status === "complete" && data.output_ready) {
            const { data: urlData } = await postProcessingApi.getProcessingJobVideoUrl(projectId, jobId);
            setVideoJob({ status: "complete", jobId, outputUrl: urlData.url, errorMessage: null });
          } else if (data.status === "cancelled") {
            setVideoJob({ status: "cancelled", jobId, outputUrl: null, errorMessage: null });
          } else {
            setVideoJob({
              status: "failed",
              jobId,
              outputUrl: null,
              errorMessage: data.error_message || "Processing failed",
            });
          }
        } else {
          // Still running — resume polling (startPolling calls stopPolling first, no duplicates)
          startPolling(projectId, jobId);
        }
      } catch {
        // Job not found or wrong project — clear persisted job silently
        clearPPPage(pageKey);
        setVideoJob(INITIAL_VIDEO_STATE);
      }
    }
  }

  // ── Settings helpers ─────────────────────────────────────────────────────────
  function setSetting<K extends keyof PPSettings>(key: K, value: PPSettings[K]) {
    setSettings(s => ({ ...s, [key]: value }));
  }

  function toggleClassFilter(name: string) {
    setSettings(s => ({
      ...s,
      class_filter: s.class_filter.includes(name)
        ? s.class_filter.filter(n => n !== name)
        : [...s.class_filter, name],
    }));
  }

  async function saveSettings(silent = false): Promise<boolean> {
    if (!activeProject) return false;
    setSaving(true);
    try {
      await postProcessingApi.updateSettings(
        activeProject.id,
        settings,
        activeImpulseId ?? undefined,
      );
      if (!silent) toast.success("Post-processing settings saved");
      return true;
    } catch (e: any) {
      toast.error(extractApiError(e) || "Failed to save settings");
      return false;
    } finally {
      setSaving(false);
    }
  }

  // ── Video job ────────────────────────────────────────────────────────────────
  function stopPolling() {
    if (pollRef.current != null) {
      clearInterval(pollRef.current);
      pollRef.current = null;
    }
  }

  function startPolling(projectId: string, jobId: string) {
    stopPolling();
    pollRef.current = setInterval(async () => {
      try {
        const { data } = await postProcessingApi.getProcessingJob(projectId, jobId);
        if (shouldStopPolling(data.status)) {
          stopPolling();
          if (data.status === "complete" && data.output_ready) {
            const { data: urlData } = await postProcessingApi.getProcessingJobVideoUrl(projectId, jobId);
            setVideoJob({ status: "complete", jobId, outputUrl: urlData.url, errorMessage: null });
          } else if (data.status === "cancelled") {
            setVideoJob(prev => ({ ...prev, status: "cancelled", errorMessage: null }));
          } else {
            setVideoJob(prev => ({
              ...prev,
              status: "failed",
              errorMessage: data.error_message || "Processing failed",
            }));
          }
        }
      } catch (e: any) {
        stopPolling();
        setVideoJob(prev => ({ ...prev, status: "failed", errorMessage: extractApiError(e) }));
      }
    }, 3000);
  }

  async function renderFromSample() {
    if (!activeProject || !selectedVideoId) return;
    stopPolling();
    setVideoPlayError(false);

    // Persist the current settings to DB before the worker reads them.
    // Without this the worker may load a stale class_filter from a previous save.
    const saved = await saveSettings(true);
    if (!saved) return;

    setVideoJob({ status: "processing", jobId: null, outputUrl: null, errorMessage: null });
    try {
      const { data } = await postProcessingApi.triggerPostProcessingFromSample(
        activeProject.id, selectedVideoId, activeImpulse?.id ?? undefined,
      );
      setVideoJob(prev => ({ ...prev, jobId: data.id }));
      startPolling(activeProject.id, data.id);
    } catch (e: any) {
      setVideoJob({ status: "failed", jobId: null, outputUrl: null, errorMessage: extractApiError(e) });
    }
  }

  function resetVideo() {
    cancelActiveJobIfAny();
    stopPolling();
    setVideoJob(INITIAL_VIDEO_STATE);
  }

  // Best-effort server-side cancel of whatever job is currently tracked, so
  // switching videos (or any other reset) mid-render doesn't orphan the
  // Celery task — it would otherwise keep running to completion unseen,
  // since the frontend is about to forget its jobId.
  function cancelActiveJobIfAny() {
    if (activeProject && videoJob.jobId && videoJob.status === "processing") {
      postProcessingApi.cancelProcessingJob(activeProject.id, videoJob.jobId).catch(() => { });
    }
  }

  async function cancelRender() {
    if (!activeProject || !videoJob.jobId || cancelling) return;
    setCancelling(true);
    try {
      await postProcessingApi.cancelProcessingJob(activeProject.id, videoJob.jobId);
      stopPolling();
      setVideoJob(prev => ({ ...prev, status: "cancelled", errorMessage: null }));
    } catch (e: any) {
      toast.error(extractApiError(e) || "Failed to cancel render");
    } finally {
      setCancelling(false);
    }
  }

  const selectedVideo = projectVideos.find(v => v.id === selectedVideoId);

  // ── Shared video selector — dashed dropzone with native select overlay ────
  function VideoDropzone() {
    const label = selectedVideo?.filename ?? "Select a video…";
    return (
      <div className="pe-pp-dropzone relative">
        <span className="pe-pp-dropzone-icon" aria-hidden="true">
          <PlayCircle size={20} strokeWidth={2.2} />
        </span>
        <div className="pe-pp-dropzone-text">
          <span className="pe-pp-dropzone-title" title={label}>{label}</span>
          <span className="pe-pp-dropzone-sub">MP4, MOV, AVI up to 1GB</span>
        </div>
        <span className="pe-pp-dropzone-btn" aria-hidden="true">
          <Upload size={14} /> Select Video
        </span>
        {/* Transparent native <select> overlays the entire row; preserves a11y */}
        {projectVideos.length > 0 ? (
          <select
            aria-label="Select a project video"
            value={selectedVideoId ?? ""}
            onChange={e => {
              const id = e.target.value || null;
              setSelectedVideoId(id);
              if (id) {
                // Auto-trigger render when picking a new video
                cancelActiveJobIfAny();
                stopPolling();
                setVideoJob(INITIAL_VIDEO_STATE);
              }
            }}
          >
            <option value="">Select a video…</option>
            {projectVideos.map(v => (
              <option key={v.id} value={v.id}>{v.filename}</option>
            ))}
          </select>
        ) : null}
      </div>
    );
  }

  // ── Small toggle switch ─────────────────────────────────────────────────────
  function Toggle({
    checked,
    onChange,
    ariaLabel,
  }: { checked: boolean; onChange: (v: boolean) => void; ariaLabel: string }) {
    return (
      <label className="pe-pp-switch" aria-label={ariaLabel}>
        <input
          type="checkbox"
          checked={checked}
          onChange={e => onChange(e.target.checked)}
        />
        <span className="pe-pp-switch-slider" />
      </label>
    );
  }

  // ── Number stepper input ─────────────────────────────────────────────────────
  function NumberStepper({
    value,
    onChange,
    min = 0,
    max,
    step = 1,
  }: {
    value: number;
    onChange: (v: number) => void;
    min?: number;
    max?: number;
    step?: number;
  }) {
    const clamp = (n: number) => {
      let next = n;
      if (typeof max === "number") next = Math.min(max, next);
      next = Math.max(min, next);
      return next;
    };
    return (
      <div className="pe-pp-num">
        <input
          type="number"
          min={min}
          max={max}
          step={step}
          value={Number.isFinite(value) ? value : ""}
          onChange={e => {
            const v = parseFloat(e.target.value);
            if (!isNaN(v)) onChange(clamp(v));
          }}
        />
        <div className="pe-pp-num-spin">
          <button type="button" aria-label="increment" onClick={() => onChange(clamp(value + step))}>
            <ChevronUp size={12} />
          </button>
          <button type="button" aria-label="decrement" onClick={() => onChange(clamp(value - step))}>
            <ChevronDown size={12} />
          </button>
        </div>
      </div>
    );
  }

  // ── Render ───────────────────────────────────────────────────────────────────

  // Until modelSupportStatus AND training-validity are committed, we can't tell
  // whether to render the main UI or the "Almost there!" warning. Render the
  // loading card first so the warning is the first non-loading frame for draft
  // or non-completed-latest-run impulses.
  const isInitialLoad =
    !!activeProject && (loading || modelSupportStatus === null || trainingValidityLoading);
  // The latest training run is the single source of truth. Even if an older
  // completed run produced compatible artifacts, we suppress them whenever the
  // latest run is cancelled/failed/running/queued.
  const notReady =
    !isInitialLoad && !!activeProject && (!hasValidTrainingOutput || modelSupportStatus === "no_model");
  const unsupportedModel =
    !isInitialLoad && !!activeProject && hasValidTrainingOutput && modelSupportStatus === "unsupported";

  if (activeProject?.project_type === "motion") {
    return (
      <MotionPhasePending
        feature="Post-processing"
        description="Once motion post-processing is wired up, this page will let you tune thresholds and filtering on top of your trained motion model."
        tip="Post-processing tunes a working model — design and train your impulse first."
        icon={Sliders}
      />
    );
  }

  if (isInitialLoad) {
    return (
      <div className="pe-pp-page mx-auto max-w-[88rem] space-y-6">
        <div className="pe-pp-card text-center py-12">
          <p className="text-sm text-[color:var(--app-text-muted)]">Loading settings…</p>
        </div>
      </div>
    );
  }

  if (notReady) {
    return (
      <ImpulseNotReady
        description="Your impulse is not fully trained. Use the items in the navigation bar to configure and train your model before you can configure postprocessing."
        tip="Configure and train your model first — post-processing tunes thresholds on top of a working detector."
      />
    );
  }

  if (unsupportedModel) {
    return (
      <ImpulseNotReady
        description={modelSupportMessage || "The latest trained model for this impulse is not compatible with video post-processing preview."}
        tip="Train a Vision Pro, EdgeDetect Lite, or NanoVision object-detection model for this impulse, then return here to render video previews."
      />
    );
  }

  const renderingPreview = videoJob.status === "processing";

  return (
    <div className="pe-pp-page mx-auto max-w-[88rem] space-y-6">
      <div>
        <h2 className="pe-pp-title">Post-processing for object detection models</h2>
        <p className="pe-pp-sub">
          Configure tracking and filtering settings, then select a project video to see your
          trained detection model run on real footage with the full post-processing
          pipeline applied.
        </p>
      </div>

      {loading && (
        <div className="pe-pp-card text-center py-12">
          <p className="text-sm text-[color:var(--app-text-muted)]">Loading settings…</p>
        </div>
      )}

      {!loading && !activeProject && (
        <div className="pe-pp-card text-center py-16">
          <Sliders size={36} className="mx-auto mb-3 text-[color:var(--app-text-muted)]" />
          <p className="text-[color:var(--app-text-muted)]">
            Select a project to configure post-processing.
          </p>
        </div>
      )}

      {!loading && activeProject && (
        <>
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
            {/* ── LEFT column ─────────────────────────────────────────── */}
            <div className="lg:col-span-5 space-y-5">
              {/* Object tracking settings */}
              <div className="pe-pp-card">
                <div className="pe-pp-card-head">
                  <span className="pe-pp-card-icon" aria-hidden="true">
                    <ScanLine size={18} strokeWidth={2.2} />
                  </span>
                  <h3 className="pe-pp-card-title">Object tracking settings</h3>
                </div>

                <div className="space-y-5">
                  <div className="pe-pp-row">
                    <div className="pe-pp-field">
                      <span className="pe-pp-field-label">Enable post-processing</span>
                      <span className="pe-pp-field-help">
                        Apply threshold filtering and tracking to model output
                      </span>
                    </div>
                    <Toggle
                      ariaLabel="Enable post-processing"
                      checked={settings.enabled}
                      onChange={v => setSetting("enabled", v)}
                    />
                  </div>

                  <div className="pe-pp-field">
                    <span className="pe-pp-field-label">
                      Detection threshold
                      <HelpTooltip text="Intersection over Union threshold to decide if a bounding box is the same object." />
                    </span>
                    <span className="pe-pp-field-help">Filter out weak detections</span>
                    <NumberStepper
                      value={settings.threshold}
                      onChange={v => setSetting("threshold", Math.min(1, Math.max(0, v)))}
                      min={0}
                      max={1}
                      step={0.05}
                    />
                  </div>

                  <div className="pe-pp-row">
                    <div className="pe-pp-field">
                      <span className="pe-pp-field-label">
                        Enable tracking
                        <HelpTooltip text="Enables object tracking module with fixed parameters (can override them via thresholds)" />
                      </span>
                      <span className="pe-pp-field-help">
                        Assign stable IDs to detections across frames
                      </span>
                    </div>
                    <Toggle
                      ariaLabel="Enable tracking"
                      checked={settings.tracking_enabled}
                      onChange={v => setSetting("tracking_enabled", v)}
                    />
                  </div>

                  <div className="grid grid-cols-2 gap-4">
                    <div className="pe-pp-field">
                      <span className="pe-pp-field-label">
                        Keep grace frames
                        <HelpTooltip text="How many frames an object is kept if it disappears." />
                      </span>
                      <span className="pe-pp-field-help">Temporarily keep lost tracks</span>
                      <NumberStepper
                        value={settings.keep_grace}
                        onChange={v => setSetting("keep_grace", Math.max(0, Math.round(v)))}
                        min={0}
                        max={100}
                        step={1}
                      />
                    </div>
                    <div className="pe-pp-field">
                      <span className="pe-pp-field-label">
                        Max observations
                        <HelpTooltip text="The maximum number of observations to match for stable tracking." />
                      </span>
                      <span className="pe-pp-field-help">Confirm tracks with min hits</span>
                      <NumberStepper
                        value={settings.max_observations}
                        onChange={v => setSetting("max_observations", Math.max(1, Math.round(v)))}
                        min={1}
                        max={10000}
                        step={1}
                      />
                    </div>
                  </div>
                </div>
              </div>

              {/* Class filter */}
              <div className="pe-pp-card">
                <div className="pe-pp-card-head">
                  <span className="pe-pp-card-icon" aria-hidden="true">
                    <FilterIcon size={18} strokeWidth={2.2} />
                  </span>
                  <h3 className="pe-pp-card-title">
                    Class filter
                    <HelpTooltip text="When set, only shows traces for these classes." />
                  </h3>
                </div>
                <p className="pe-pp-field-help mb-3">
                  Check classes to include. Leave all unchecked to pass every class.
                </p>
                {labelsLoading ? (
                  <p className="pe-pp-field-help">Loading classes…</p>
                ) : labels.length === 0 ? (
                  <p className="pe-pp-field-help">
                    No classes found. Add labels in the Data section to enable class filtering.
                  </p>
                ) : (
                  <div className="pe-pp-check-grid">
                    {labels.map(lbl => {
                      const checked = settings.class_filter.includes(lbl.name);
                      return (
                        <label
                          key={lbl.id}
                          className={`pe-pp-check-row ${checked ? "is-checked" : ""}`}
                        >
                          <input
                            type="checkbox"
                            checked={checked}
                            onChange={() => toggleClassFilter(lbl.name)}
                          />
                          <span className="pe-pp-check-box" aria-hidden="true">
                            <svg width="11" height="11" viewBox="0 0 14 14" fill="none">
                              <path
                                d="M3 7.2 L6 10 L11 4"
                                stroke="currentColor"
                                strokeWidth="2.2"
                                strokeLinecap="round"
                                strokeLinejoin="round"
                              />
                            </svg>
                          </span>
                          <span className="pe-pp-check-label">{lbl.name}</span>
                        </label>
                      );
                    })}
                  </div>
                )}
              </div>

              {/* Save */}
              <button
                type="button"
                onClick={() => { void saveSettings(); }}
                disabled={saving}
                className="pe-pp-save"
              >
                <Save size={16} /> {saving ? "Saving…" : "Save settings"}
              </button>
            </div>

            {/* ── RIGHT column ────────────────────────────────────────── */}
            <div className="lg:col-span-7">
              <div className="pe-pp-card">
                <div className="pe-pp-card-head" style={{ marginBottom: "0.4rem" }}>
                  <span className="pe-pp-card-icon" aria-hidden="true">
                    <PartyPopper size={18} strokeWidth={2.2} />
                  </span>
                  <div>
                    <h3 className="pe-pp-card-title">Render a preview</h3>
                    <p className="pe-pp-field-help" style={{ marginTop: 2 }}>
                      Select a project video to run your trained model with post-processing applied.
                    </p>
                  </div>
                </div>

                <div className="mt-4">
                  <VideoDropzone />
                </div>

                {/* Preview area — varies by job status */}
                {videoJob.status === "idle" && (
                  <div className="pe-pp-preview-empty">
                    <Video size={32} />
                    <p className="text-sm font-medium text-[color:var(--app-text)]">
                      Pick a video to render a preview
                    </p>
                    <p className="text-xs text-[color:var(--app-text-muted)] max-w-sm text-center px-4">
                      Once selected, your trained model runs with the full post-processing
                      pipeline and the result appears here.
                    </p>
                  </div>
                )}

                {renderingPreview && (
                  <div className="pe-pp-preview-empty pe-pp-preview-empty--processing">
                    <div className="pe-pp-processing-shimmer" aria-hidden="true" />
                    <div className="pe-pp-processing-icon">
                      <Video size={24} strokeWidth={2} />
                      <span className="pe-pp-processing-icon-ring" aria-hidden="true" />
                      <span className="pe-pp-processing-icon-ring pe-pp-processing-icon-ring--delay" aria-hidden="true" />
                    </div>
                    <p className="pe-pp-processing-title">
                      Running inference and post-processing
                      <span className="pe-pp-processing-dots" aria-hidden="true">
                        <span />
                        <span />
                        <span />
                      </span>
                    </p>
                    <div className="pe-pp-processing-bar" aria-hidden="true">
                      <div className="pe-pp-processing-bar-fill" />
                    </div>
                    {selectedVideo && (
                      <p className="pe-pp-processing-meta">{selectedVideo.filename}</p>
                    )}
                    {videoJob.jobId && (
                      <p className="pe-pp-processing-meta font-mono">
                        job {videoJob.jobId.slice(0, 8)}…
                      </p>
                    )}
                  </div>
                )}

                {videoJob.status === "complete" && videoJob.outputUrl && (
                  <>
                    <div className="pe-pp-preview">
                      <video
                        ref={videoRef}
                        src={videoJob.outputUrl}
                        controls
                        className="w-full h-auto block"
                        style={{ maxHeight: "560px" }}
                        onError={() => setVideoPlayError(true)}
                      />
                    </div>
                    {videoPlayError && (
                      <p className="mt-2 text-xs text-[color:var(--app-text-muted)] text-center">
                        Video could not be played in this browser.
                      </p>
                    )}
                  </>
                )}

                {videoJob.status === "failed" && (
                  <div className="pe-pp-preview-empty" style={{ color: "#dc2626" }}>
                    <AlertCircle size={28} />
                    <p className="text-sm font-medium">Processing failed</p>
                    <p className="text-xs text-[color:var(--app-text-muted)] max-w-sm text-center px-4">
                      {videoJob.errorMessage || "An unexpected error occurred."}
                    </p>
                  </div>
                )}

                {videoJob.status === "cancelled" && (
                  <div className="pe-pp-preview-empty">
                    <XCircle size={28} />
                    <p className="text-sm font-medium text-[color:var(--app-text)]">Render cancelled</p>
                    <p className="text-xs text-[color:var(--app-text-muted)] max-w-sm text-center px-4">
                      You cancelled this render. Pick a video and try again.
                    </p>
                  </div>
                )}

                {/* Actions row */}
                <div className="pe-pp-preview-actions">
                  <button
                    type="button"
                    className="pe-pp-action-btn"
                    disabled={videoJob.status !== "complete"}
                    onClick={() => videoRef.current?.play()}
                  >
                    <PlayCircle size={14} /> Play
                  </button>
                  {renderingPreview && (
                    <button
                      type="button"
                      className="pe-pp-action-btn"
                      disabled={!videoJob.jobId || cancelling}
                      onClick={cancelRender}
                    >
                      <XCircle size={14} /> {cancelling ? "Cancelling…" : "Cancel"}
                    </button>
                  )}
                  <button
                    type="button"
                    className="pe-pp-action-btn"
                    disabled={!selectedVideoId || renderingPreview}
                    onClick={renderFromSample}
                  >
                    <RefreshCw size={14} /> Render preview
                  </button>
                </div>
              </div>
            </div>
          </div>

        </>
      )}
    </div>
  );
}
