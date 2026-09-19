"use client";
import { useEffect, useRef, useState } from "react";
import { useRouter } from "next/navigation";
import { useAppStore } from "@/store/appStore";
import { trainingApi } from "@/utils/api";
import {
  Bell,
  ArrowRight,
  RefreshCcw,
  Layers,
  Target,
  CheckCircle,
  Box as BoxIcon,
  Sparkles,
  CheckCircle2,
  XCircle,
  AlertTriangle,
  Clock,
  Hash,
  Activity,
  CircleDashed,
  Terminal,
} from "lucide-react";
import { TrainingLogOutput } from "../_components/TrainingLogOutput";
import ImpulseNotReady from "@/components/dashboard/ImpulseNotReady";
import MotionPhasePending from "@/components/dashboard/MotionPhasePending";
import { useTrainingValidity, invalidateTrainingStatus } from "@/hooks/useTrainingValidity";
import toast from "react-hot-toast";
import { ChevronDown, ChevronUp } from "lucide-react";

const POLL_MS = 2500;

type JobStatus = "pending" | "running" | "completed" | "failed" | "cancelled";

const STATUS_META: Record<
  JobStatus,
  { label: string; sub: string; icon: React.ComponentType<any> }
> = {
  pending: { label: "Queued", sub: "Waiting in the build queue…", icon: CircleDashed },
  running: { label: "Running", sub: "Training in progress", icon: Activity },
  completed: { label: "Completed", sub: "Model build finished", icon: CheckCircle2 },
  failed: { label: "Failed", sub: "Model build failed", icon: XCircle },
  cancelled: { label: "Cancelled", sub: "Build was cancelled", icon: AlertTriangle },
};

function formatDuration(ms: number): string {
  if (!Number.isFinite(ms) || ms < 0) return "—";
  const s = Math.floor(ms / 1000);
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  const sec = s % 60;
  if (h > 0) return `${h}h ${m}m ${sec}s`;
  if (m > 0) return `${m}m ${sec}s`;
  return `${sec}s`;
}

function jobDurationMs(job: any): number {
  const start = job?.started_at || job?.created_at;
  if (!start) return 0;
  const startMs = new Date(start).getTime();
  const endMs = job?.completed_at ? new Date(job.completed_at).getTime() : Date.now();
  return endMs - startMs;
}

function currentEpoch(job: any): number {
  const th = job?.training_history || {};
  return (
    th.actual_epochs ??
    th.loss?.length ??
    th.accuracy?.length ??
    job?.actual_epochs ??
    0
  );
}

function requestedEpochs(job: any): number {
  const th = job?.training_history || {};
  return job?.requested_epochs ?? th.requested_epochs ?? job?.epochs ?? 0;
}

function buildConfigSteps(impulse: any): string[] {
  const steps: string[] = [];
  for (const block of impulse?.dsp_blocks ?? []) {
    steps.push(block.name || block.type || "Processing block");
  }
  for (const block of impulse?.ml_blocks ?? []) {
    steps.push(block.name || block.type || "Learning block");
  }
  steps.push("Model testing");
  return steps.length > 1 ? steps : ["Image", "Object Detection (Images)", "Model testing"];
}

function stepIcon(label: string): React.ComponentType<any> {
  const s = label.toLowerCase();
  if (s.includes("test")) return CheckCircle;
  if (s.includes("detect") || s.includes("object") || s.includes("yolo") || s.includes("fomo"))
    return Target;
  if (s.includes("image") || s.includes("dsp") || s.includes("processing")) return Layers;
  return Layers;
}

export default function RetrainModelPage() {
  const { activeProject, activeImpulse } = useAppStore();
  const router = useRouter();
  const {
    hasValidTrainingOutput,
    loading: trainingValidityLoading,
  } = useTrainingValidity(activeImpulse?.id ?? null);

  const [isTraining, setIsTraining] = useState(false);
  const [activeJob, setActiveJob] = useState<any>(null);
  const logEndRef = useRef<HTMLDivElement>(null);

  const configSteps = buildConfigSteps(activeImpulse);

  async function loadStatus(impulseId: string) {
    try {
      const { data: jobsData } = await trainingApi.listJobs(impulseId);
      const retrainJob =
        (jobsData as any[]).find((job) => job?.launch_mode === "retrain") ?? null;
      setActiveJob(retrainJob);
      setIsTraining(retrainJob?.status === "running" || retrainJob?.status === "pending");
    } catch (e) {
      console.error(e);
      setActiveJob(null);
      setIsTraining(false);
    }
  }

  useEffect(() => {
    if (!activeImpulse?.id) {
      setActiveJob(null);
      setIsTraining(false);
      return;
    }
    loadStatus(activeImpulse.id);
  }, [activeImpulse?.id]);

  // Poll while job is running
  useEffect(() => {
    if (!isTraining || !activeJob?.id) return;
    const interval = setInterval(async () => {
      try {
        const { data } = await trainingApi.getJob(activeJob.id);
        setActiveJob(data);
        if (
          data.status === "completed" ||
          data.status === "failed" ||
          data.status === "cancelled"
        ) {
          setIsTraining(false);
          clearInterval(interval);
          // Retrain is non-destructive. Broadcast ONLY on success: completion
          // moves the Active Model pointer on the server, so downstream pages
          // must refetch. A cancelled / failed retrain leaves the previous
          // Active Model intact, so no invalidation is fired.
          if (data.status === "completed") {
            invalidateTrainingStatus(activeImpulse?.id);
            toast.success("Retraining completed");
          } else if (data.status === "cancelled") {
            toast.success("Retraining cancelled");
          } else {
            toast.error("Retraining failed: " + (data.error_message || "unknown error"));
          }
        }
      } catch (e) {
        console.error(e);
      }
    }, POLL_MS);
    return () => clearInterval(interval);
  }, [isTraining, activeJob?.id, activeImpulse?.id]);

  // Auto-scroll log output
  useEffect(() => {
    logEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [activeJob?.training_history?.log_lines?.length, activeJob?.status]);

  async function handleRetrain() {
    if (!activeImpulse) {
      toast.error("No active impulse — create one first");
      return;
    }
    setIsTraining(true);
    try {
      const { data } = await trainingApi.retrain({ impulse_id: activeImpulse.id });
      // `data` may be the job directly or wrapped; fall back to loadStatus if no id
      if (data?.id) {
        setActiveJob(data);
      } else {
        await loadStatus(activeImpulse.id);
      }
      // Retrain is non-destructive: downstream pages must keep showing the
      // current Active Model while the new run is in flight. We deliberately
      // do NOT invalidate here — the broadcast fires only on successful
      // completion (see the poll effect above).
      toast.success("Retraining started");
    } catch (e: any) {
      toast.error(e.response?.data?.detail || "Failed to start retraining");
      setIsTraining(false);
    }
  }

  async function handleCancel() {
    if (!activeJob?.id) return;
    try {
      await trainingApi.cancelJob(activeJob.id);
      toast.success("Job cancelled");
      setActiveJob((prev: any) => prev ? ({
        ...prev,
        status: "cancelled",
        error_message: "Cancelled by user",
        completed_at: new Date().toISOString(),
      }) : prev);
      setIsTraining(false);
      // Cancelled retrain is non-destructive — the previous Active Model
      // remains the source of truth, so we deliberately do NOT broadcast
      // here. Downstream pages should not flicker.
    } catch {
      toast.error("Failed to cancel job");
    }
  }

  // Until training validity resolves, we can't tell whether to render the main
  // retrain UI or the "Almost there!" warning. Render a loading state first so
  // the warning is the first non-loading frame for untrained impulses.
  const isInitialLoad = !!activeImpulse && trainingValidityLoading;
  const notReady = !isInitialLoad && !!activeImpulse && !hasValidTrainingOutput;

  if (activeProject?.project_type === "motion") {
    return (
      <MotionPhasePending
        feature="Retrain"
        description="Once motion retraining is wired up, you'll be able to relaunch training with your impulse's current configuration from here."
        tip="Retraining reuses your impulse's saved processing and learning blocks — for now, use Training to run a first pass."
        icon={RefreshCcw}
      />
    );
  }

  if (!activeImpulse) {
    return (
      <ImpulseNotReady
        description="No impulse to retrain yet. Create an impulse and train a model before you can retrain it."
        tip="Head to Impulse Design to create one, configure your data, and run an initial training pass."
      />
    );
  }

  if (isInitialLoad) {
    return (
      <div className="flex h-[calc(100vh-theme('spacing.16'))] items-center justify-center pt-2">
        <span className="inline-flex items-center gap-2 text-sm text-[var(--app-text-soft)]">
          <RefreshCcw size={14} className="animate-spin" />
          Verifying model status...
        </span>
      </div>
    );
  }

  return (
    <div className="retrain-page h-[calc(100vh-theme('spacing.16'))] flex flex-col pt-2">
      {notReady ? (
        <ImpulseNotReady
          description="Your impulse is not fully trained. Use the items in the navigation bar to configure and train your model before you can retrain the model."
          tip="Configure your data, label it, and train your model to enable retraining."
        />
      ) : (
        /* ── Normal retrain layout ───────────────────────────────────── */
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 h-full pb-6">

          {/* ── Left panel: config ─────────────────────────────────── */}
          <div className="retrain-card">
            <div className="retrain-head">
              <div className="retrain-head__left">
                <span className="retrain-head__tile">
                  <RefreshCcw size={18} />
                </span>
                <h2 className="retrain-head__title">
                  Retrain model with known parameters
                </h2>
              </div>
            </div>

            <div className="retrain-body">
              <ul className="retrain-steps">
                {configSteps.map((step) => {
                  const Icon = stepIcon(step);
                  return (
                    <li key={step} className="retrain-step">
                      <span className="retrain-step__icon">
                        <Icon size={15} />
                      </span>
                      <span className="retrain-step__text">{step}</span>
                    </li>
                  );
                })}
              </ul>

              <div className="retrain-action">
                <button
                  onClick={handleRetrain}
                  disabled={isTraining || !activeImpulse}
                  className={`pe-save-btn disabled:opacity-60 disabled:cursor-not-allowed ${isTraining ? "is-loading" : ""
                    }`}
                >
                  {isTraining ? (
                    <span className="pe-save-spinner" aria-hidden="true" />
                  ) : (
                    <Sparkles size={16} aria-hidden="true" />
                  )}
                  <span>
                    {isTraining
                      ? activeJob?.status === "pending"
                        ? "Queued…"
                        : "Training in progress…"
                      : "Retrain model"}
                  </span>
                </button>

                {!activeImpulse && (
                  <p className="text-xs text-gray-500 text-center">
                    No active impulse.{" "}
                    <button
                      onClick={() => router.push("/dashboard/impulse")}
                      className="text-brand-400 hover:underline"
                    >
                      Create one first
                    </button>
                  </p>
                )}

                {activeJob?.status === "completed" && (
                  <button
                    onClick={() =>
                      router.push(
                        `/dashboard/impulse/training?impulseId=${activeImpulse?.id}`
                      )
                    }
                    className="flex items-center gap-1.5 text-xs text-gray-500 hover:text-gray-700 transition-colors"
                  >
                    View full training details
                    <ArrowRight size={12} />
                  </button>
                )}
              </div>
            </div>
          </div>

          {/* ── Right panel: build output ───────────────────────────── */}
          <BuildOutputPanel
            job={activeJob}
            isTraining={isTraining}
            onCancel={handleCancel}
            logEndRef={logEndRef}
          />

        </div>
      )}
    </div>
  );
}

/* ────────────────────────────────────────────────────────────────────────
 * Build output panel — premium status strip + progress + outcome + logs.
 * Uses real job data; falls back to the existing empty-state when no job.
 * ──────────────────────────────────────────────────────────────────────── */

function BuildOutputPanel({
  job,
  isTraining,
  onCancel,
  logEndRef,
}: {
  job: any;
  isTraining: boolean;
  onCancel: () => void;
  logEndRef: React.RefObject<HTMLDivElement>;
}) {
  const [logCollapsed, setLogCollapsed] = useState(false);
  const status: JobStatus | null = job?.status ?? null;
  const meta = status ? STATUS_META[status] : null;

  const reqEpochs = requestedEpochs(job);
  const curEpoch = currentEpoch(job);
  const progressPct =
    reqEpochs > 0
      ? Math.max(2, Math.min(100, (curEpoch / reqEpochs) * 100))
      : isTraining
        ? 8
        : 0;

  const durationMs = job ? jobDurationMs(job) : 0;
  const jobIdShort = job?.id ? String(job.id).substring(0, 8) : null;
  const isLive = status === "running" || status === "pending";

  const accentClass = status
    ? `build-output__accent build-output__accent--${status}`
    : "build-output__accent";
  const chipClass = status ? `bo-chip bo-chip--${status}` : "bo-chip bo-chip--running";

  /* ── Clean empty state — matches the target mockup ───────────────── */
  if (!job) {
    return (
      <div className={`retrain-card${logCollapsed ? " self-start" : ""}`}>
        <div className="retrain-head">
          <div className="retrain-head__left">
            <span className="retrain-head__tile">
              <BoxIcon size={18} />
            </span>
            <h2 className="retrain-head__title">Build output</h2>

          </div>
          <button
            type="button"
            onClick={() => setLogCollapsed(v => !v)}
            aria-label={logCollapsed ? "Expand training output" : "Collapse training output"}
            aria-expanded={!logCollapsed}
            title={logCollapsed ? "Expand" : "Collapse"}
            className="rounded-lg p-1.5 text-[var(--app-text-soft)] transition-colors hover:bg-white/10 hover:text-[var(--app-text)]"
          >
            {logCollapsed ? <ChevronDown size={16} /> : <ChevronUp size={16} />}
          </button>
        </div>

        <div className="retrain-empty">
          <div className="retrain-empty__art" aria-hidden="true">
            <NoActiveJobArt />
          </div>
          <h3 className="retrain-empty__heading">No active job</h3>
          <p className="retrain-empty__hint">
            Click <span className="kw-violet">Retrain</span>{" "}
            <span className="kw-pink">model</span> to begin.
          </p>
        </div>
      </div>
    );
  }

  /* ── Active / completed / failed / cancelled job ─────────────────── */
  return (
    <div className={`retrain-card build-output${logCollapsed ? " self-start" : ""}`}>
      <div aria-hidden="true" className={accentClass} />

      {/* Hero header */}
      <div className="retrain-head">
        <div className="retrain-head__left">
          <span className="retrain-head__tile">
            <BoxIcon size={18} />
          </span>
          <div>
            <h2 className="retrain-head__title">Build output</h2>
            <div className="build-output__title-sub">
              {meta ? meta.sub : "Live build, logs, and outcome details"}
            </div>
          </div>
        </div>
        <div className="build-output__hero-right">
          {meta && (
            <span className={`${chipClass}${isLive ? "" : " bo-chip--static"}`}>
              <span className="bo-chip__dot" />
              {meta.label}
            </span>
          )}
          {isTraining && (
            <button onClick={onCancel} className="build-output__cancel">
              Cancel
            </button>
          )}
          <button
            type="button"
            onClick={() => setLogCollapsed(v => !v)}
            aria-label={logCollapsed ? "Expand training output" : "Collapse training output"}
            aria-expanded={!logCollapsed}
            title={logCollapsed ? "Expand" : "Collapse"}
            className="rounded-lg p-1.5 text-[var(--app-text-soft)] transition-colors hover:bg-white/10 hover:text-[var(--app-text)]"
          >
            {logCollapsed ? <ChevronDown size={16} /> : <ChevronUp size={16} />}
          </button>
        </div>
      </div>

      {/* Log frame — code-editor look */}
      {!logCollapsed && (
        <div className="build-output__log-wrap">
          <span className="build-output__log-label">
            <Terminal size={12} />
            Output log
          </span>
          <div className="bo-log-frame">
            <div className="bo-log-frame__bar">
              <span className="bo-log-frame__dot bo-log-frame__dot--r" />
              <span className="bo-log-frame__dot bo-log-frame__dot--y" />
              <span className="bo-log-frame__dot bo-log-frame__dot--g" />
              <span className="bo-log-frame__name">
                {job ? `job-${jobIdShort ?? "—"}.log` : "build.log"}
              </span>
            </div>
            <div className="bo-log-frame__body train-log-area">
              <TrainingLogOutput
                job={job}
                isTraining={isTraining}
                logEndRef={logEndRef}
                emptyMessage="No active job. Click Retrain model to begin."
              />
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

function OutcomeBanner({
  variant,
  title,
  subtitle,
  icon,
  metrics,
}: {
  variant: "completed" | "failed" | "cancelled";
  title: string;
  subtitle: string;
  icon: React.ReactNode;
  metrics?: { label: string; value: string }[];
}) {
  return (
    <div className={`bo-banner bo-banner--${variant}`}>
      <span className="bo-banner__icon">{icon}</span>
      <div className="min-w-0 flex-1">
        <p className="bo-banner__title">{title}</p>
        <p className="bo-banner__sub">{subtitle}</p>
        {metrics && metrics.length > 0 && (
          <div className="bo-metrics">
            {metrics.map((m) => (
              <span key={m.label} className="bo-metric">
                <span className="bo-metric__label">{m.label}</span>
                <span className="bo-metric__value">{m.value}</span>
              </span>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

/* ────────────────────────────────────────────────────────────────────────
 * NoActiveJobArt — purple/lavender illustration shown in the empty Build
 * output panel. Stylized "build artifact" on a circular plinth with sparkles.
 * ──────────────────────────────────────────────────────────────────────── */
function NoActiveJobArt() {
  return (
    <svg
      viewBox="0 0 240 200"
      width="100%"
      height="100%"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
      role="img"
      aria-hidden="true"
    >
      <defs>
        <linearGradient id="naja-plinth" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor="#ddd6fe" />
          <stop offset="100%" stopColor="#c7d2fe" stopOpacity="0.0" />
        </linearGradient>
        <linearGradient id="naja-box" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor="#a78bfa" />
          <stop offset="100%" stopColor="#7c3aed" />
        </linearGradient>
        <linearGradient id="naja-box-side" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor="#8b5cf6" />
          <stop offset="100%" stopColor="#6d28d9" />
        </linearGradient>
        <linearGradient id="naja-wave" x1="0" y1="0" x2="1" y2="0">
          <stop offset="0%" stopColor="#ffffff" stopOpacity="0.9" />
          <stop offset="100%" stopColor="#e9d5ff" stopOpacity="0.9" />
        </linearGradient>
      </defs>

      {/* Plinth */}
      <ellipse cx="120" cy="168" rx="86" ry="14" fill="url(#naja-plinth)" />

      {/* Dotted halo behind */}
      <g opacity="0.45">
        <circle cx="42" cy="40" r="2" fill="#a78bfa" />
        <circle cx="60" cy="22" r="1.5" fill="#a78bfa" />
        <circle cx="200" cy="35" r="2" fill="#a78bfa" />
        <circle cx="216" cy="60" r="1.5" fill="#a78bfa" />
        <circle cx="218" cy="100" r="1.6" fill="#a78bfa" />
        <circle cx="30" cy="92" r="1.6" fill="#a78bfa" />
        <circle cx="22" cy="70" r="1.4" fill="#a78bfa" />
      </g>

      {/* Sparkles */}
      <g fill="#c4b5fd">
        <path d="M178 36 l3 7 7 3 -7 3 -3 7 -3 -7 -7 -3 7 -3z" opacity="0.9" />
        <path d="M50 124 l2 5 5 2 -5 2 -2 5 -2 -5 -5 -2 5 -2z" opacity="0.85" />
        <path d="M196 130 l2 4 4 2 -4 2 -2 4 -2 -4 -4 -2 4 -2z" opacity="0.8" />
      </g>

      {/* Folder/box body */}
      <g>
        <rect x="78" y="74" width="84" height="74" rx="10" fill="url(#naja-box-side)" />
        <rect x="74" y="86" width="92" height="62" rx="10" fill="url(#naja-box)" />
        <path
          d="M74 96 Q120 80 166 96"
          stroke="#ffffff"
          strokeOpacity="0.55"
          strokeWidth="1.2"
          fill="none"
        />
        <path
          d="M86 120 L98 120 L104 108 L112 132 L120 112 L128 128 L136 116 L146 120 L154 120"
          stroke="url(#naja-wave)"
          strokeWidth="2.4"
          strokeLinecap="round"
          strokeLinejoin="round"
          fill="none"
        />
      </g>

      {/* Top tab/cap */}
      <rect x="98" y="68" width="44" height="14" rx="4" fill="#8b5cf6" />
      <rect x="100" y="70" width="40" height="3" rx="1.5" fill="#ffffff" opacity="0.4" />
    </svg>
  );
}
