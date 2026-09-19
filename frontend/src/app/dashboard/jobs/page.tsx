"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { createPortal } from "react-dom";
import {
  RefreshCw,
  ChevronLeft,
  ChevronRight,
  History,
  FileDownIcon,
  FileText,
  X,
  CheckCircle2,
  XCircle,
  Loader2,
  Ban,
  Clock,
  ListChecks,
} from "lucide-react";
import toast from "react-hot-toast";

import { useAppStore } from "@/store/appStore";
import { jobsApi } from "@/utils/api";
// Reuse the exact owner-avatar initials logic from the Projects page so the
// Started By cell renders identically — same helper, same `pe-projects-row-avatar` class.
import { avatarInitial } from "@/utils/avatarInitial";

type JobRow = {
  id: string;
  type: "training" | "retraining" | "deployment" | "model_test" | "ai_labeling" | string;
  type_label: string;
  status: string;
  started_by?: string | null;
  created_at?: string | null;
  completed_at?: string | null;
  duration_seconds: number;
  source_table: string;
  source_id: string;
  // training/retraining only:
  impulse_id?: string;
  job_id?: string;
  launch_mode?: "start" | "retrain";
  // deployment only:
  download_url?: string | null;
};

type JobsResponse = {
  items: JobRow[];
  total: number;
  page: number;
  page_size: number;
};

const PAGE_SIZE = 20;

// Frontend job `type` → backend log endpoint `job_type`. Only job types that
// persist a log stream appear here; everything else renders no View Logs
// action. The frontend stays out of worker/model internals — it only knows
// this type→action mapping.
const LOG_JOB_TYPE: Record<string, "training" | "retraining"> = {
  training: "training",
  retraining: "retraining",
};

// ── Formatters ────────────────────────────────────────────────────────────────

function shortId(id: string): string {
  return id ? id.slice(0, 8) : "—";
}

function formatDate(iso?: string | null): string {
  if (!iso) return "—";
  const d = new Date(iso);
  if (isNaN(d.getTime())) return "—";
  const now = new Date();
  const sameDay =
    d.getFullYear() === now.getFullYear() &&
    d.getMonth() === now.getMonth() &&
    d.getDate() === now.getDate();
  const time = d.toLocaleTimeString(undefined, { hour: "2-digit", minute: "2-digit", second: "2-digit", hour12: false });
  if (sameDay) return `Today, ${time}`;
  const date = d.toLocaleDateString(undefined, { month: "short", day: "numeric", year: "numeric" });
  return `${date}, ${time}`;
}

function formatDuration(seconds: number): string {
  if (!seconds || seconds <= 0) return "—";
  const s = Math.floor(seconds);
  if (s < 60) return `${s}s`;
  const m = Math.floor(s / 60);
  const r = s % 60;
  if (m < 60) return r ? `${m}m ${r}s` : `${m}m`;
  const h = Math.floor(m / 60);
  const rm = m % 60;
  return rm ? `${h}h ${rm}m` : `${h}h`;
}

// Status → pill class. The unified vocabulary comes from the backend; anything
// else falls through to "muted" and is displayed as-is so we never silently
// hide an unmapped status.
function statusPillClass(status: string): string {
  switch (status) {
    case "completed": return "pe-dep2-pill pe-dep2-pill--success";
    case "running": return "pe-dep2-pill pe-dep2-pill--info";
    case "failed": return "pe-dep2-pill pe-dep2-pill--danger";
    case "cancelled": return "pe-dep2-pill pe-dep2-pill--warn";
    case "pending": return "pe-dep2-pill pe-dep2-pill--muted";
    default: return "pe-dep2-pill pe-dep2-pill--muted";
  }
}

const STATUS_LABELS: Record<string, string> = {
  completed: "Completed",
  running: "Running",
  failed: "Failed",
  cancelled: "Cancelled",
  pending: "Pending",
  unknown: "Unknown",
};

function statusLabel(status: string): string {
  return STATUS_LABELS[status] ?? status.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());
}

function StatusIcon({ status }: { status: string }) {
  switch (status) {
    case "completed": return <CheckCircle2 size={12} aria-hidden="true" />;
    case "failed": return <XCircle size={12} aria-hidden="true" />;
    case "running": return <Loader2 size={12} aria-hidden="true" className="animate-spin" />;
    case "cancelled": return <Ban size={12} aria-hidden="true" />;
    case "pending": return <Clock size={12} aria-hidden="true" />;
    default: return <Clock size={12} aria-hidden="true" />;
  }
}

// ── Page ──────────────────────────────────────────────────────────────────────

export default function JobsPage() {
  const { activeProject, user, profileExtras } = useAppStore();

  // Identity for the "Started By" cell. Jobs are always owned by the current
  // user (project-ownership check on the backend), so we render the same
  // display name + initial the Projects page derives from the store — keeping
  // the avatar bit-for-bit identical instead of showing the raw owner email.
  const startedByName =
    ((user?.id && profileExtras[user.id]?.name?.trim()) || "") ||
    user?.username ||
    (user?.email ? user.email.split("@")[0] : "User");
  const startedByInitial = avatarInitial(
    (user?.id && profileExtras[user.id]?.name?.trim()) || user?.username,
    user?.email,
  );

  const [page, setPage] = useState(1);
  const [data, setData] = useState<JobsResponse | null>(null);
  const [loading, setLoading] = useState(false);

  // ── Log viewer modal ──────────────────────────────────────────────────────
  // `logJob` is the row whose logs are open (null = closed). `logLines` is
  // null until the fetch resolves, then the verbatim array (possibly empty).
  const [logJob, setLogJob] = useState<JobRow | null>(null);
  const [logLines, setLogLines] = useState<string[] | null>(null);
  const [logLoading, setLogLoading] = useState(false);
  const [logError, setLogError] = useState<string | null>(null);
  const logScrollRef = useRef<HTMLDivElement>(null);

  const openLogs = useCallback(
    async (row: JobRow) => {
      const jobType = LOG_JOB_TYPE[row.type];
      if (!jobType || !activeProject?.id) return;
      setLogJob(row);
      setLogLines(null);
      setLogError(null);
      setLogLoading(true);
      try {
        const { data: resp } = await jobsApi.logs(activeProject.id, jobType, row.id);
        const lines: string[] = resp?.log_lines ?? [];
        setLogLines(lines);
        // Auto-scroll to the bottom only for a still-running job; for
        // completed/failed/cancelled jobs we keep the user at the top.
        if (resp?.status === "running") {
          requestAnimationFrame(() => {
            const el = logScrollRef.current;
            if (el) el.scrollTop = el.scrollHeight;
          });
        }
      } catch (e: any) {
        const detail = e?.response?.data?.detail || e?.message || "Failed to load logs";
        setLogError(detail);
        setLogLines([]);
      } finally {
        setLogLoading(false);
      }
    },
    [activeProject?.id],
  );

  const closeLogs = useCallback(() => {
    setLogJob(null);
    setLogLines(null);
    setLogError(null);
    setLogLoading(false);
  }, []);

  // Close the modal on Esc.
  useEffect(() => {
    if (!logJob) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") closeLogs();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [logJob, closeLogs]);

  const fetchPage = useCallback(async (projectId: string, p: number) => {
    setLoading(true);
    try {
      const { data: resp } = await jobsApi.list(projectId, p, PAGE_SIZE);
      setData(resp);
    } catch (e: any) {
      const detail = e?.response?.data?.detail || e?.message || "Failed to load jobs";
      toast.error(detail);
    } finally {
      setLoading(false);
    }
  }, []);

  // Reset to page 1 when the active project changes.
  useEffect(() => {
    if (!activeProject?.id) return;
    setPage(1);
  }, [activeProject?.id]);

  useEffect(() => {
    if (!activeProject?.id) return;
    fetchPage(activeProject.id, page);
  }, [activeProject?.id, page, fetchPage]);

  const totalPages = useMemo(() => {
    if (!data || data.page_size <= 0) return 1;
    return Math.max(1, Math.ceil(data.total / data.page_size));
  }, [data]);

  const handleRefresh = () => {
    if (!activeProject?.id) return;
    fetchPage(activeProject.id, page);
  };

  // The Jobs page is informational for everything except deployment
  // downloads. Training/retraining/test/labeling rows render an empty
  // action cell — no deep-link, no Open button.
  const handleRowAction = (row: JobRow) => {
    if (row.type === "deployment" && row.download_url) {
      window.open(row.download_url, "_blank", "noopener,noreferrer");
    }
  };

  if (!activeProject) {
    return (
      <div className="w-full">
        <div className="pe-dep2-surface p-8 text-center">
          <p className="text-sm" style={{ color: "var(--app-text-muted)" }}>
            Select a project to view its jobs.
          </p>
        </div>
      </div>
    );
  }

  const items = data?.items ?? [];
  const total = data?.total ?? 0;

  return (
    <div className="w-full space-y-5">
      <section className="pe-dep2-surface" style={{ padding: "1.5rem 1.8rem" }}>
        <div className="flex items-start justify-between gap-4">
          <div className="min-w-0">
            <h1 className="pe-dep2-hero-title" style={{ display: "flex", alignItems: "center", gap: "0.6rem" }}>
              <span>Jobs</span>
            </h1>
            <p className="pe-dep2-hero-sub">
              All jobs for {user?.username ?? "your account"} / {activeProject.name}
            </p>
          </div>
          <button
            type="button"
            onClick={handleRefresh}
            disabled={loading}
            className="data-acq-btn-primary"
            title="Refresh current page"
          >
            <RefreshCw size={14} className={loading ? "animate-spin" : ""} aria-hidden="true" />
            Refresh
          </button>
        </div>
      </section>

      <section className="pe-dep2-surface" style={{ padding: "1rem 1.2rem" }}>
        <div className="overflow-x-auto">
          <table className="w-full text-sm" style={{ borderCollapse: "collapse" }}>
            <thead>
              <tr style={{ textAlign: "left", color: "var(--app-text-muted)" }}>
                <th className="py-2 pr-4 font-medium" style={{ fontSize: "0.72rem", letterSpacing: "0.04em", textTransform: "uppercase" }}>Job ID</th>
                <th className="py-2 pr-4 font-medium" style={{ fontSize: "0.72rem", letterSpacing: "0.04em", textTransform: "uppercase" }}>Type</th>
                <th className="py-2 pr-4 font-medium" style={{ fontSize: "0.72rem", letterSpacing: "0.04em", textTransform: "uppercase" }}>Started By</th>
                <th className="py-2 pr-4 font-medium" style={{ fontSize: "0.72rem", letterSpacing: "0.04em", textTransform: "uppercase" }}>Created</th>
                <th className="py-2 pr-4 font-medium" style={{ fontSize: "0.72rem", letterSpacing: "0.04em", textTransform: "uppercase" }}>Finished</th>
                <th className="py-2 pr-4 font-medium" style={{ fontSize: "0.72rem", letterSpacing: "0.04em", textTransform: "uppercase" }}>Status</th>
                <th className="py-2 pr-4 font-medium" style={{ fontSize: "0.72rem", letterSpacing: "0.04em", textTransform: "uppercase" }}>length</th>
                <th className="py-2 font-medium" style={{ fontSize: "0.72rem", letterSpacing: "0.04em", textTransform: "uppercase", textAlign: "right" }}>Action</th>
              </tr>
            </thead>
            <tbody>
              {items.length === 0 && !loading && (
                <tr>
                  <td colSpan={8} className="py-8 text-center" style={{ color: "var(--app-text-muted)" }}>
                    No jobs yet for this project.
                  </td>
                </tr>
              )}
              {loading && items.length === 0 && (
                <tr>
                  <td colSpan={8} className="py-8 text-center" style={{ color: "var(--app-text-muted)" }}>
                    <span className="inline-flex items-center gap-2">
                      <RefreshCw size={14} className="animate-spin" />
                      Loading jobs…
                    </span>
                  </td>
                </tr>
              )}
              {items.map((row) => {
                // Deployment rows with a download_url get a Download action;
                // training/retraining/feature-gen rows get a View Logs action.
                // Everything else renders an empty action cell.
                const actionable = row.type === "deployment" && !!row.download_url;
                const loggable = !!LOG_JOB_TYPE[row.type];
                return (
                  <tr
                    key={`${row.source_table}:${row.id}`}
                    style={{ borderTop: "1px solid rgba(148, 163, 184, 0.12)" }}
                  >
                    <td className="py-3 pr-4 font-mono" style={{ color: "var(--app-text)" }} title={row.id}>
                      {shortId(row.id)}
                    </td>
                    <td className="py-3 pr-4" style={{ color: "var(--app-text)" }}>
                      {row.type_label}
                    </td>
                    <td className="py-3 pr-4">
                      <div className="flex items-center gap-3">
                        <span className="pe-projects-row-avatar" aria-hidden="true">
                          {startedByInitial}
                        </span>
                        <span style={{ color: "var(--app-text)" }}>{startedByName}</span>
                      </div>
                    </td>
                    <td className="py-3 pr-4" style={{ color: "var(--app-text-muted)" }}>
                      {formatDate(row.created_at)}
                    </td>
                    <td className="py-3 pr-4" style={{ color: "var(--app-text-muted)" }}>
                      {formatDate(row.completed_at)}
                    </td>
                    <td className="py-3 pr-4">
                      <span className={statusPillClass(row.status)}>
                        <StatusIcon status={row.status} />
                        {statusLabel(row.status)}
                      </span>
                    </td>
                    <td className="py-3 pr-4" style={{ color: "var(--app-text-muted)" }}>
                      {formatDuration(row.duration_seconds)}
                    </td>
                    <td className="py-3 text-right">
                      <div className="inline-flex items-center justify-end gap-1.5">
                        {actionable && (
                          <button
                            type="button"
                            onClick={() => handleRowAction(row)}
                            className="inline-flex items-center gap-1 rounded-md px-2 py-1 text-xs font-medium transition-colors"
                            style={{
                              color: " #777777",
                              background: "rgba(167, 139, 250, 0.10)",
                              border: "1px solid rgba(167, 139, 250, 0.30)",
                            }}
                            title="Download package"
                          >
                            <FileDownIcon size={12} />
                          </button>
                        )}
                        {loggable && (
                          <button
                            type="button"
                            onClick={() => openLogs(row)}
                            className="inline-flex items-center gap-1 rounded-md px-2 py-1 text-xs font-medium transition-colors"
                            style={{
                              color: "#777777",
                              background: "rgba(167, 139, 250, 0.10)",
                              border: "1px solid rgba(167, 139, 250, 0.30)",
                            }}
                            title="View logs"
                          >
                            <FileText size={12} />
                          </button>
                        )}
                      </div>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>

        {/* Pagination */}
        <div className="mt-4 flex items-center justify-between text-sm" style={{ color: "var(--app-text-muted)" }}>
          <div>
            {total > 0 ? (
              <span>
                Page {data?.page ?? page} of {totalPages} · {total} job{total === 1 ? "" : "s"} total
              </span>
            ) : (
              <span>0 jobs</span>
            )}
          </div>
          <div className="flex items-center gap-2">
            <button
              type="button"
              onClick={() => setPage((p) => Math.max(1, p - 1))}
              disabled={loading || page <= 1}
              className="inline-flex items-center gap-1 rounded-md px-2 py-1 text-xs font-medium transition-colors disabled:cursor-not-allowed disabled:opacity-40"
              style={{
                color: "var(--app-text)",
                background: "rgba(148, 163, 184, 0.10)",
                border: "1px solid rgba(148, 163, 184, 0.28)",
              }}
            >
              <ChevronLeft size={12} />
              Prev
            </button>
            <button
              type="button"
              onClick={() => setPage((p) => Math.min(totalPages, p + 1))}
              disabled={loading || page >= totalPages}
              className="inline-flex items-center gap-1 rounded-md px-2 py-1 text-xs font-medium transition-colors disabled:cursor-not-allowed disabled:opacity-40"
              style={{
                color: "var(--app-text)",
                background: "rgba(148, 163, 184, 0.10)",
                border: "1px solid rgba(148, 163, 184, 0.28)",
              }}
            >
              Next
              <ChevronRight size={12} />
            </button>
          </div>
        </div>
      </section>

      {/* ── Log viewer modal ─────────────────────────────────────────────── */}
      {/* Portaled to <body> so the dashboard layout's containing block (the
          scrollable <main>) can't trap the `fixed` overlay below the TopBar —
          same pattern the TopBar avatar dropdown uses. */}
      {logJob && typeof document !== "undefined" && createPortal(
        <div
          onClick={closeLogs}
          className="fixed inset-0 z-[100] flex items-center justify-center p-4"
          style={{ background: "rgba(2, 6, 23, 0.66)" }}
          role="dialog"
          aria-modal="true"
        >
          <div
            onClick={(e) => e.stopPropagation()}
            className="flex w-full max-w-3xl flex-col"
            style={{
              maxHeight: "82vh",
              padding: "1.25rem 1.4rem",
              // Opaque solid surface — the glass `pe-dep2-surface` is meant to
              // sit on the page background; as a floating modal its blur lets
              // the table behind bleed through, so use a solid theme surface.
              background: "var(--app-surface)",
              border: "1px solid var(--app-border-strong)",
              borderRadius: 16,
              boxShadow: "0 24px 60px rgba(2, 6, 23, 0.45)",
            }}
          >
            <div className="mb-3 flex items-start justify-between gap-4">
              <h2 className="text-sm font-semibold" style={{ color: "var(--app-text)" }}>
                Showing logs for job{" "}
                <span className="font-mono" style={{ color: "var(--app-text)" }}>{logJob.id}</span>
              </h2>
              <button
                type="button"
                onClick={closeLogs}
                className="inline-flex items-center justify-center rounded-md p-1 transition-colors"
                style={{ color: "var(--app-text-muted)" }}
                title="Close"
                aria-label="Close"
              >
                <X size={16} />
              </button>
            </div>

            <div
              ref={logScrollRef}
              className="overflow-y-auto whitespace-pre-wrap rounded-md font-mono text-[11px] leading-relaxed"
              style={{
                height: "420px",
                padding: "0.85rem 1rem",
                // Theme surface so the global (theme-matched) scrollbar blends
                // instead of rendering a light strip against a dark box.
                color: "var(--app-text)",
                background: "var(--app-surface-2)",
                border: "1px solid var(--app-border)",
              }}
            >
              {logLoading && (
                <span className="inline-flex items-center gap-2" style={{ color: "var(--app-text-muted)" }}>
                  <Loader2 size={12} className="animate-spin" />
                  Loading logs…
                </span>
              )}
              {!logLoading && logError && (
                <span style={{ color: "#fca5a5" }}>{logError}</span>
              )}
              {!logLoading && !logError && logLines && logLines.length === 0 && (
                <span className="italic" style={{ color: "var(--app-text-muted)" }}>
                  No logs recorded for this job.
                </span>
              )}
              {!logLoading && !logError && logLines && logLines.length > 0 &&
                logLines.map((line, i) => <div key={`log-${i}`}>{line}</div>)}
            </div>
          </div>
        </div>,
        document.body,
      )}
    </div>
  );
}

