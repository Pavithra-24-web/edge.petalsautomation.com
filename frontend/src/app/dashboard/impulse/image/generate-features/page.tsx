"use client";
import { useState, useEffect, useRef, useMemo } from "react";
import { useAppStore } from "@/store/appStore";
import { dspApi, impulsesApi } from "@/utils/api";
import Link from "next/link";
import { useSearchParams, useRouter } from "next/navigation";
import toast from "react-hot-toast";
import {
  Play, Clock, Maximize2,
  Settings, Layers, RefreshCw, AlertCircle,
  Image as ImageIcon, ExternalLink, Code,
  ChevronDown, ChevronUp,
  BadgeAlertIcon,
  CpuIcon,
  MapIcon,
  InfoIcon
} from "lucide-react";
import FeatureExplorer from "@/components/dashboard/FeatureExplorer";
import { samplesApi } from "@/utils/api";

export default function GenerateFeaturesPage() {
  const searchParams = useSearchParams();
  const router = useRouter();
  const {
    hasHydrated,
    activeImpulse,
    savedActiveImpulse,
    setActiveImpulse,
    setSavedActiveImpulse,
    featureGenerationPage,
    setFeatureGenerationPage,
  } = useAppStore();
  const impulseId = searchParams.get("impulseId");
  const currentImpulse = useMemo(
    () => (impulseId ? (activeImpulse?.id === impulseId ? activeImpulse : null) : activeImpulse),
    [activeImpulse, impulseId]
  );

  // Gate the route: parameters must be saved before features can be generated.
  // Source of truth is the backend-stamped `dsp_params_saved_at` field — it
  // is null until the user submits the Parameters form successfully and the
  // server confirms the save. Checking populated fields (image_width etc.)
  // would let defaults from impulse creation unlock the page.
  const parametersSaved = useMemo(
    () => !!savedActiveImpulse?.dsp_params_saved_at,
    [savedActiveImpulse]
  );

  useEffect(() => {
    if (!hasHydrated) return;
    if (!currentImpulse) return;
    // Wait for the saved snapshot to catch up with the active impulse before
    // judging save-state; otherwise we'd bounce users away during the brief
    // window after URL-driven impulse fetch but before setSavedActiveImpulse.
    if (savedActiveImpulse?.id !== currentImpulse.id) return;
    if (parametersSaved) return;
    const blockType = currentImpulse.dsp_blocks?.[0]?.type || "image";
    const qs = impulseId ? `?impulseId=${impulseId}&needsSave=1` : "?needsSave=1";
    router.replace(`/dashboard/impulse/${blockType}/parameters${qs}`);
  }, [hasHydrated, currentImpulse, savedActiveImpulse, parametersSaved, impulseId, router]);

  // If URL specifies an impulse that differs from the store, fetch and sync it
  useEffect(() => {
    if (!impulseId || activeImpulse?.id === impulseId) return;
    let cancelled = false;
    impulsesApi.get(impulseId).then(({ data }) => {
      if (!cancelled) {
        setActiveImpulse(data);
        setSavedActiveImpulse(data);
      }
    }).catch(() => { });
    return () => { cancelled = true; };
  }, [impulseId, activeImpulse?.id, setActiveImpulse, setSavedActiveImpulse]);

  const [summary, setSummary] = useState({ total_samples: 0, test_samples: 0, num_classes: 0, class_names: [] });
  const [job, setJob] = useState<{ id: string | null; status: string; progress: number }>({ id: null, status: "idle", progress: 0 });
  const [featurePoints, setFeaturePoints] = useState<any[]>([]);
  const [isLoadingFeatures, setIsLoadingFeatures] = useState(false);
  const [featureError, setFeatureError] = useState<string | null>(null);

  // Selection states
  const [selectedSampleId, setSelectedSampleId] = useState<string | null>(null);
  const [selectedSampleData, setSelectedSampleData] = useState<any>(null);
  const [selectedSampleUrl, setSelectedSampleUrl] = useState<string | null>(null);
  const [isLoadingSample, setIsLoadingSample] = useState(false);
  const [imgDims, setImgDims] = useState({ w: 0, h: 0 });
  const [showFeatures, setShowFeatures] = useState(false);
  const [logsCollapsed, setLogsCollapsed] = useState(false);

  const [logs, setLogs] = useState<string[]>([
    "Ready to generate features.",
    "Click 'Generate features' to start the DSP pipeline."
  ]);
  const defaultLogs = useMemo(
    () => [
      "Ready to generate features.",
      "Click 'Generate features' to start the DSP pipeline."
    ],
    []
  );
  const persistedOutput = currentImpulse?.id ? featureGenerationPage[currentImpulse.id] : null;
  // Keep a ref so the initialization effect can read the latest persistedOutput
  // without listing it as a dependency (it writes to it, so listing it causes a loop).
  const persistedOutputRef = useRef(persistedOutput);
  persistedOutputRef.current = persistedOutput;

  const normalizeBoxLabel = (value: any) => {
    const text = String(value ?? "").trim();
    if (!text) return "";
    const lowered = text.toLowerCase();
    if (lowered === "unlabeled" || lowered === "unlabelled" || lowered === "unknown") {
      return "";
    }
    return text;
  };

  const getDisplayLabel = (sample: any) => {
    const boxes = sample?.extra_metadata?.boundingBoxes;
    if (Array.isArray(boxes)) {
      for (const box of boxes) {
        const label = normalizeBoxLabel(box?.label);
        if (label) return label;
      }
    }
    return normalizeBoxLabel(sample?.label_name);
  };

  const logsEndRef = useRef<HTMLDivElement>(null);
  const pollingRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const isSubmittingRef = useRef(false);
  // Monotonic version stamp for `loadFeatureData` calls. Any time the user
  // starts a new generation (or hops impulses) we bump this; in-flight
  // `loadFeatureData` responses compare their captured version against the
  // ref and bail out if it has advanced. Without this guard, a stale
  // dataset-load response that resolves AFTER the user clicked "Generate"
  // overwrites the optimistic `pending` status back to `completed`, which
  // is exactly what made the action feel like it needed two clicks.
  const loadVersionRef = useRef(0);

  const sameLogs = (a: string[], b: string[]) =>
    a.length === b.length && a.every((line, index) => line === b[index]);

  const appendLog = (message: string) => {
    setLogs((current) => (current[current.length - 1] === message ? current : [...current, message]));
  };

  const loadFeatureData = async (impulseId: string) => {
    const myVersion = ++loadVersionRef.current;
    setIsLoadingFeatures(true);
    setFeatureError(null);
    try {
      const res = await dspApi.getFeatures(impulseId);
      // If the user clicked "Generate features" (or switched impulse) while
      // this request was in flight, abandon the response so it can't undo
      // the optimistic pending state or restore stale feature points.
      if (myVersion !== loadVersionRef.current) return;
      if (res.data.points) {
        setFeaturePoints(res.data.points);
        if (res.data.points.length > 0) {
          setLogs((current) => {
            const stillDefault =
              current.length === defaultLogs.length &&
              current.every((line, index) => line === defaultLogs[index]);
            if (!stillDefault) return current;
            return [
              "Features already generated for this impulse.",
              `Loaded ${res.data.points.length} feature points from saved feature data.`,
            ];
          });
          setJob((current) => {
            // Never overwrite an active or optimistic run. Only mark
            // "completed" when we're truly idle with no in-flight job.
            if (current.id) return current;
            if (current.status === "pending" || current.status === "running") return current;
            return { ...current, status: "completed", progress: 100 };
          });
        }
      }
      // Backend now returns a clean empty `{points: []}` when no features
      // exist yet (NoSuchKey) and a generic friendly `message` only on real
      // failures — surface it as an error rather than logs so we don't echo
      // any backend text into the activity log.
      if (res.data.message && res.data.points.length === 0) {
        setFeatureError("We couldn't load the feature data. Please try regenerating features.");
      }
    } catch (e: any) {
      if (myVersion !== loadVersionRef.current) return;
      setFeatureError("We couldn't load the feature data. Please try regenerating features.");
      console.error(e);
    } finally {
      if (myVersion === loadVersionRef.current) {
        setIsLoadingFeatures(false);
      }
    }
  };

  const handlePointClick = async (point: any) => {
    if (!point.sample_id) return;
    setSelectedSampleId(point.sample_id);
    setIsLoadingSample(true);
    setSelectedSampleData(null);
    setSelectedSampleUrl(null);
    setImgDims({ w: 0, h: 0 });

    try {
      const { data: sample } = await samplesApi.get(point.sample_id);
      setSelectedSampleData(sample);

      const { data: download } = await samplesApi.download(point.sample_id);
      setSelectedSampleUrl(download.url);
    } catch (e) {
      toast.error("Failed to load sample details");
    } finally {
      setIsLoadingSample(false);
    }
  };

  useEffect(() => {
    if (currentImpulse) {
      dspApi.datasetSummary(currentImpulse.id)
        .then(res => setSummary(res.data))
        .catch(e => console.error("Failed to load summary", e));

      loadFeatureData(currentImpulse.id);
    }
  }, [currentImpulse?.id]);

  useEffect(() => {
    if (logsEndRef.current) {
      logsEndRef.current.scrollIntoView({ behavior: "smooth" });
    }
  }, [logs]);

  useEffect(() => {
    if (!hasHydrated || !currentImpulse?.id) return;
    // Read via ref — we intentionally exclude persistedOutput from deps because
    // Effect 2 writes to it, and including it here creates an infinite loop.
    const po = persistedOutputRef.current;
    if (po) {
      const nextLogs = po.logs?.length ? po.logs : defaultLogs;
      const nextJob = {
        id: po.jobId ?? null,
        status: po.status ?? "idle",
        progress: Number(po.progress ?? 0),
      };

      setLogs((current) => (sameLogs(current, nextLogs) ? current : nextLogs));
      setJob((current) =>
        current.id === nextJob.id &&
          current.status === nextJob.status &&
          current.progress === nextJob.progress
          ? current
          : nextJob
      );
      return;
    }
    setLogs((current) => (sameLogs(current, defaultLogs) ? current : defaultLogs));
    setJob((current) =>
      current.id === null && current.status === "idle" && current.progress === 0
        ? current
        : { id: null, status: "idle", progress: 0 }
    );
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [hasHydrated, currentImpulse?.id, defaultLogs]);

  useEffect(() => {
    if (!hasHydrated || !currentImpulse?.id) return;
    if (
      persistedOutput &&
      sameLogs(persistedOutput.logs ?? [], logs) &&
      (persistedOutput.jobId ?? null) === job.id &&
      (persistedOutput.status ?? "idle") === job.status &&
      Number(persistedOutput.progress ?? 0) === job.progress
    ) {
      return;
    }
    setFeatureGenerationPage(currentImpulse.id, {
      logs,
      jobId: job.id,
      status: job.status,
      progress: job.progress,
    });
  }, [hasHydrated, currentImpulse?.id, persistedOutput, logs, job, setFeatureGenerationPage]);

  useEffect(() => {
    if (!job.id || !["pending", "running"].includes(job.status)) {
      if (pollingRef.current) {
        clearInterval(pollingRef.current);
        pollingRef.current = null;
      }
      return;
    }

    const poll = async () => {
      try {
        const statusRes = await dspApi.jobStatus(job.id!);
        const sData = statusRes.data;
        const nextStatus = sData.status || "pending";
        const nextProgress = Number(sData.progress ?? 0);

        setJob({ id: job.id, status: nextStatus, progress: nextProgress });

        if (nextStatus === "pending") {
          appendLog("Waiting for worker to pick up the task...");
          return;
        }

        if (nextStatus === "running") {
          const phase = sData.meta?.status || "processing";
          setLogs((current) => {
            const msg = `Running feature extraction... ${nextProgress}% (${phase})`;
            return current[current.length - 1]?.startsWith("Running feature extraction...")
              ? [...current.slice(0, -1), msg]
              : [...current, msg];
          });
          return;
        }

        if (pollingRef.current) {
          clearInterval(pollingRef.current);
          pollingRef.current = null;
        }

        if (nextStatus === "completed") {
          appendLog("Done.");
          appendLog(`Features saved to ${sData.result?.storage_key || "storage"}`);
          toast.success("Features generated!");
          if (currentImpulse?.id) {
            loadFeatureData(currentImpulse.id);
          }
          return;
        }

        if (nextStatus === "cancelled") {
          appendLog("Feature generation cancelled.");
          toast.success("Feature generation cancelled");
          return;
        }

        if (nextStatus === "failed") {
          appendLog(`Job failed: ${sData.error || "Unknown error"}`);
          toast.error("Failed to generate features");
        }
      } catch (e: any) {
        if (pollingRef.current) {
          clearInterval(pollingRef.current);
          pollingRef.current = null;
        }
        setJob((current) => ({ ...current, status: "failed" }));
        appendLog(`Failed to poll job status: ${e?.response?.data?.error || e?.message || "Unknown error"}`);
        toast.error("Status check failed");
      }
    };

    poll();
    pollingRef.current = setInterval(poll, 2000);

    return () => {
      if (pollingRef.current) {
        clearInterval(pollingRef.current);
        pollingRef.current = null;
      }
    };
  }, [job.id, job.status, currentImpulse?.id]);

  const handleCancel = async () => {
    if (!job.id) return;
    try {
      await dspApi.cancel(job.id);
      if (pollingRef.current) {
        clearInterval(pollingRef.current);
        pollingRef.current = null;
      }
      setJob({ id: null, status: "idle", progress: 0 });
      appendLog("Feature generation cancelled.");
      toast.success("Feature generation cancelled");
    } catch (e: any) {
      toast.error("Failed to cancel job");
    }
  };

  const handleGenerate = async () => {
    if (!currentImpulse) return;
    if (isSubmittingRef.current) return;
    isSubmittingRef.current = true;

    // Invalidate any in-flight loadFeatureData response so it cannot land
    // after this click and overwrite our optimistic state.
    loadVersionRef.current++;
    setIsLoadingFeatures(false);

    // Flip to a pending state synchronously so the button immediately shows
    // "Generating features..." on the first click, before the queue API
    // responds. Polling stays off until a real job.id arrives.
    setJob({ id: null, status: "pending", progress: 0 });
    setLogsCollapsed(false);
    setLogs(["Starting feature extraction pipeline..."]);
    setFeaturePoints([]);
    setSelectedSampleId(null);
    setSelectedSampleData(null);
    setSelectedSampleUrl(null);
    setFeatureError(null);

    try {
      const res = await dspApi.generateFeatures({ impulse_id: currentImpulse.id });
      const data = res.data;

      setJob({ id: data.job_id, status: "pending", progress: 0 });
      appendLog(`Job dispatched (ID: ${data.job_id})`);
    } catch (e: any) {
      const detail = e?.response?.data?.detail || e?.message || "Failed to queue feature generation";
      toast.error(detail);
      appendLog(`Error: ${detail}`);
      setJob({ id: null, status: "failed", progress: 0 });
    } finally {
      isSubmittingRef.current = false;
    }
  };

  if (!currentImpulse) return <div className="p-6 text-gray-400">Loading impulse...</div>;

  const isRunning = job.status === "running" || job.status === "pending";
  const blockType = currentImpulse.dsp_blocks?.[0]?.type || "image";
  const selectedDisplayLabel = getDisplayLabel(selectedSampleData);
  const selectedPointData = selectedSampleId
    ? featurePoints.find((point) => point.sample_id === selectedSampleId)
    : null;

  return (
    <div className="pe-dsp image-dsp-page image-dsp-features-page flex flex-col min-h-full">

      {/* Tabs */}
      <div className="pe-dsp-tabs">
        <Link
          href={`/dashboard/impulse/${blockType}/parameters${impulseId ? `?impulseId=${impulseId}` : ""}`}
          className="pe-dsp-tab"
        >
          Parameters
        </Link>
        {parametersSaved ? (
          <Link
            href={`/dashboard/impulse/${blockType}/generate-features${impulseId ? `?impulseId=${impulseId}` : ""}`}
            className="pe-dsp-tab is-active"
          >
            Generate features
          </Link>
        ) : (
          <button
            type="button"
            disabled
            aria-disabled="true"
            title="Save parameters before generating features."
            className="pe-dsp-tab is-active is-disabled"
            style={{ opacity: 0.5, cursor: "not-allowed" }}
            onClick={(e) => e.preventDefault()}
          >
            Generate features
          </button>
        )}
      </div>

      <div className="pe-features-body">

        {/* Left column */}
        <div className="pe-features-col">
          {/* Generate Features card */}
          <div className="pe-dsp-card">
            <div className="pe-dsp-card-head">
              <span className="head-icon icon-blue"><Layers size={14} /></span>
              <h3 className="head-title">Generate Features</h3>
            </div>
            <div className="pe-dsp-card-body">
              <div className="pe-features-info">
                <div className="pe-features-info-row">
                  <span className="pe-features-info-label">Training samples</span>
                  <span className="pe-features-info-value">
                    {summary.total_samples} items
                  </span>
                </div>
                {Math.round(Number(currentImpulse?.train_subset_percent ?? 100)) < 100 && (
                  <div className="pe-features-info-row">
                    <span className="pe-features-info-label">Subset percentage</span>
                    <span className="pe-features-info-value">
                      {Math.round(Number(currentImpulse?.train_subset_percent ?? 100))}%
                    </span>
                  </div>
                )}
                <div className="pe-features-info-row">
                  <span className="pe-features-info-label">Classes</span>
                  <span className="pe-features-info-value">
                    {summary.num_classes}
                    {summary.class_names && summary.class_names.length > 0
                      ? ` (${summary.class_names.join(", ")})`
                      : ""}
                  </span>
                </div>
              </div>

              <div className="flex items-center gap-3">
                <button
                  onClick={handleGenerate}
                  disabled={isRunning}
                  className="data-acq-btn-primary disabled:opacity-60 disabled:cursor-not-allowed"
                >
                  {isRunning ? <RefreshCw size={14} className="animate-spin" /> : <Play size={14} fill="currentColor" />}
                  {isRunning ? "Generating features..." : "Generate features"}
                </button>
                {isRunning && job.id && (
                  <button
                    onClick={handleCancel}
                    className="rounded-lg bg-brand-600 px-3 py-1.5 text-white shadow-sm transition-colors hover:bg-brand-500"
                  >
                    Cancel
                  </button>
                )}
              </div>
            </div>
          </div>

          {/* Feature generation output card */}
          <div className="pe-dsp-card flex flex-col">
            <div className="pe-dsp-card-head" style={{ justifyContent: "space-between" }}>
              <div className="flex items-center gap-2">
                <span className="head-icon icon-blue"><InfoIcon size={14} /></span>
                <h3 className="head-title">Feature generation output</h3>
              </div>
              <button
                type="button"
                className="tt-icon-btn"
                aria-label={logsCollapsed ? "Expand output" : "Collapse output"}
                aria-expanded={!logsCollapsed}
                title={logsCollapsed ? "Expand" : "Collapse"}
                onClick={() => setLogsCollapsed(v => !v)}
              >
                {logsCollapsed ? <ChevronDown size={14} /> : <ChevronUp size={14} />}
              </button>
            </div>
            {!logsCollapsed && (
              <div className="pe-features-log">
                {logs.length === 0 ? (
                  <div style={{ color: "var(--app-text-soft)" }}>Waiting for output…</div>
                ) : (
                  logs.map((log, i) => <div key={i}>{log}</div>)
                )}
                <div ref={logsEndRef} />
              </div>
            )}
          </div>
        </div>

        {/* Right column */}
        <div className="pe-features-col">
          {/* Feature explorer — stretches to fill the right column now that
              the On-device performance card has been removed, so the column
              heights stay balanced against the left column's two cards. */}
          <div className="pe-dsp-card pe-features-explorer" style={{ flex: 1 }}>
            <div className="pe-dsp-card-head" style={{ justifyContent: "space-between" }}>
              <div className="flex items-center gap-2">
                <span className="head-icon icon-blue"><MapIcon size={14} /></span>
                <h3 className="head-title">Feature explorer</h3>
              </div>
              {selectedSampleId && (
                <button
                  onClick={() => { setSelectedSampleId(null); setSelectedSampleData(null); setSelectedSampleUrl(null); }}
                  className="clear-link"
                >
                  Clear selection
                </button>
              )}
            </div>

            <div className="explorer-chart">
              {isLoadingFeatures ? (
                <div className="flex flex-col items-center gap-3">
                  <div className="w-12 h-12 rounded-full border-2 animate-spin flex items-center justify-center"
                    style={{ borderColor: "rgba(99, 102, 241, 0.2)", borderTopColor: "#6366f1" }}>
                    <Layers size={20} style={{ color: "rgba(99, 102, 241, 0.5)" }} />
                  </div>
                  <p className="text-xs font-medium animate-pulse" style={{ color: "var(--app-text-muted)" }}>
                    Mapping feature space...
                  </p>
                </div>
              ) : featurePoints.length > 0 ? (
                <FeatureExplorer
                  points={featurePoints}
                  selectedPointId={selectedSampleId}
                  onPointClick={handlePointClick}
                />
              ) : (
                <div className="flex flex-col items-center justify-center p-8 text-center">
                  <div className="w-16 h-16 rounded-full border-2 border-dashed flex items-center justify-center mb-4"
                    style={{ borderColor: "var(--app-border)", background: "var(--app-surface-2)", color: "var(--app-text-soft)" }}>
                    <Maximize2 size={22} />
                  </div>
                  {featureError ? (
                    <>
                      <p className="text-sm font-semibold" style={{ color: "#dc2626" }}>{featureError}</p>
                      <p className="text-[10px] uppercase tracking-wider font-semibold mt-1" style={{ color: "var(--app-text-soft)" }}>
                        Please check your impulse configuration
                      </p>
                    </>
                  ) : (
                    <>
                      <p className="text-sm font-bold" style={{ color: "var(--app-text)" }}>No features yet</p>
                      <p className="text-xs mt-2 max-w-[300px] leading-relaxed" style={{ color: "var(--app-text-muted)" }}>
                        No features have been generated yet. Click &apos;Generate Features&apos; to create features for your dataset.
                      </p>
                    </>
                  )}
                </div>
              )}
            </div>

            {/* Legend (only when we have data) */}
            {featurePoints.length > 0 && (() => {
              const classes = Array.from(new Set(featurePoints.map(p => getDisplayLabel(p as any)).filter(Boolean))) as string[];
              const palette = ["#8b5cf6", "#22c55e", "#f59e0b", "#ef4444", "#06b6d4", "#ec4899"];
              return (
                <div className="explorer-legend">
                  {classes.slice(0, 6).map((c, i) => (
                    <span key={c} className="inline-flex items-center">
                      <span className="legend-dot" style={{ background: palette[i % palette.length] }} />
                      {c}
                    </span>
                  ))}
                </div>
              );
            })()}

            {/* Sample details / empty state */}
            {(!selectedSampleId && !isLoadingSample) ? (
              <div className="pe-features-detail-empty">
                <div className="icon"><ImageIcon size={20} /></div>
                <p className="label">No selection</p>
                <p className="desc">Select a point on the map to inspect metadata and labels.</p>
              </div>
            ) : isLoadingSample ? (
              <div className="pe-features-detail-empty">
                <RefreshCw className="animate-spin mx-auto mb-2" size={22} style={{ color: "#6366f1" }} />
                <p className="label">Fetching metadata…</p>
              </div>
            ) : selectedSampleData ? (
              <>
                <div className="pe-features-detail">
                  <div className="min-w-0">
                    <p className="det-eyebrow">Sample details</p>
                    <p className="det-filename">{selectedSampleData.filename.split('/').pop()}</p>
                    <div className="det-chips">
                      {selectedDisplayLabel && (
                        <span className="pe-features-pill">
                          <span className="pill-dot" /> {selectedDisplayLabel}
                        </span>
                      )}
                      <span className="pe-features-pill pe-features-pill--mono">
                        {selectedSampleId?.slice(0, 8)}
                      </span>
                    </div>
                    <div className="det-actions">
                      <Link href={`/dashboard/data/dataset?id=${selectedSampleId}`} className="det-action">
                        <span className="lhs"><ExternalLink size={14} /> View sample</span>
                      </Link>
                      <button
                        type="button"
                        onClick={() => setShowFeatures(!showFeatures)}
                        className={`det-action ${showFeatures ? "is-on" : ""}`}
                      >
                        <span className="lhs"><Code size={14} /> View features</span>
                        <span className="rhs-dot" />
                      </button>
                    </div>
                  </div>
                  <div className="det-thumb">
                    {selectedDisplayLabel && <span className="det-thumb-tag">{selectedDisplayLabel}</span>}
                    {selectedSampleUrl ? (
                      <>
                        <img
                          src={selectedSampleUrl}
                          alt="Preview"
                          onLoad={(e) => {
                            const img = e.currentTarget;
                            setImgDims({ w: img.naturalWidth, h: img.naturalHeight });
                          }}
                        />
                        {imgDims.w > 0 && selectedSampleData.extra_metadata?.boundingBoxes?.map((box: any, i: number) => (
                          normalizeBoxLabel(box.label || selectedDisplayLabel) ? (
                            <div
                              key={i}
                              className="absolute pointer-events-none"
                              style={{
                                left: `${(box.x / imgDims.w) * 100}%`,
                                top: `${(box.y / imgDims.h) * 100}%`,
                                width: `${(box.w / imgDims.w) * 100}%`,
                                height: `${(box.h / imgDims.h) * 100}%`,
                                border: "2px solid #6366f1",
                                background: "rgba(99, 102, 241, 0.12)",
                              }}
                            />
                          ) : null
                        ))}
                      </>
                    ) : (
                      <ImageIcon size={28} style={{ color: "var(--app-text-soft)" }} />
                    )}
                  </div>
                </div>

                {showFeatures && (
                  <div className="px-5 py-3 border-t" style={{ borderColor: "var(--app-border)", background: "var(--app-surface-2)" }}>
                    <div className="flex items-center justify-between mb-2">
                      <span className="text-[10px] font-bold uppercase tracking-widest" style={{ color: "var(--app-text-soft)" }}>
                        Projection data (PCA)
                      </span>
                      <button onClick={() => setShowFeatures(false)} className="text-xs font-semibold" style={{ color: "#4f46e5" }}>
                        Close
                      </button>
                    </div>
                    <div className="font-mono text-[11px] leading-relaxed break-all p-2 rounded"
                      style={{ background: "var(--app-surface)", border: "1px solid var(--app-border)", color: "var(--app-text-muted)", maxHeight: 140, overflowY: "auto" }}>
                      <pre className="whitespace-pre-wrap">
                        {JSON.stringify(selectedPointData, null, 2)}
                      </pre>
                    </div>
                  </div>
                )}
              </>
            ) : null}
          </div>
        </div>
      </div>
    </div>
  );
}
