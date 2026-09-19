"use client";
import { Suspense, useEffect, useState } from "react";
import Link from "next/link";
import { useSearchParams } from "next/navigation";
import { useAppStore } from "@/store/appStore";
import { aiLabelingApi } from "@/utils/api";
import {
  Home, ChevronRight, ArrowLeft, Cpu, RefreshCw, Loader2, AlertTriangle,
} from "lucide-react";

function confidenceStyle(c: number) {
  if (c >= 0.8) return { bar: "#22c55e", text: "#16a34a" };
  if (c >= 0.5) return { bar: "#eab308", text: "#a16207" };
  return { bar: "#ef4444", text: "#dc2626" };
}

function statusBadgeClass(status: string) {
  if (status === "approved") return "badge-green";
  if (status === "rejected") return "badge-red";
  return "badge-yellow"; // pending / unknown
}

function JobRecordsContent() {
  const { activeProject } = useAppStore();
  const searchParams = useSearchParams();
  const initialJobId = searchParams.get("job") || "";

  const [jobs, setJobs] = useState<any[]>([]);
  const [activeJob, setActiveJob] = useState<any>(null);
  const [predictions, setPredictions] = useState<any[]>([]);
  const [predictionTotal, setPredictionTotal] = useState(0);
  const [filterStatus, setFilterStatus] = useState<string>("all");
  const [loading, setLoading] = useState(false);

  // Load jobs for the active project
  useEffect(() => {
    if (!activeProject) return;
    (async () => {
      try {
        const { data } = await aiLabelingApi.listJobs(activeProject.id);
        setJobs(data);
        const preferred = data.find((j: any) => j.id === initialJobId);
        if (preferred) setActiveJob(preferred);
        else if (data.length > 0) setActiveJob(data[0]);
      } catch { /* ignore */ }
    })();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [activeProject]);

  // Load predictions when the selected job changes
  useEffect(() => {
    if (activeJob) loadPredictions(activeJob.id);
    else { setPredictions([]); setPredictionTotal(0); }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [activeJob]);

  async function loadPredictions(jobId: string) {
    setLoading(true);
    try {
      const pageSize = 250;
      const allItems: any[] = [];
      let total = 0;
      let skip = 0;
      while (true) {
        const { data } = await aiLabelingApi.getPredictions(jobId, { skip, limit: pageSize });
        const items = data?.items || (Array.isArray(data) ? data : []);
        total = data?.total || items.length;
        allItems.push(...items);
        if (items.length < pageSize || allItems.length >= total) break;
        skip += pageSize;
      }
      setPredictions(allItems);
      setPredictionTotal(total || allItems.length);
    } catch {
      setPredictions([]);
      setPredictionTotal(0);
    } finally {
      setLoading(false);
    }
  }

  const filteredPredictions = (Array.isArray(predictions) ? predictions : []).filter((p: any) =>
    filterStatus === "all" ? true : p.status === filterStatus
  );

  return (
    <div className="flex flex-col gap-4 min-h-0">
      {/* Header — breadcrumb / title */}
      <div className="flex items-start justify-between gap-6 flex-wrap">
        <div className="min-w-0 flex-1">
          <div className="data-acq-header-top">
            <Link
              href="/dashboard/data/dataset/?view=ai-labeling"
              className="data-acq-back-btn"
              aria-label="Back to AI Labeling"
            >
              <ArrowLeft size={13} strokeWidth={2.25} />
              <span>Back</span>
            </Link>
            <span className="header-sep" aria-hidden="true" />
            <nav className="data-acq-breadcrumb" aria-label="Breadcrumb">
              <Home size={13} aria-hidden="true" />
              <ChevronRight size={12} aria-hidden="true" />
              <Link href="/dashboard/data/dataset/?view=ai-labeling" className="crumb-link">AI Labeling</Link>
              <ChevronRight size={12} aria-hidden="true" />
              <span className="crumb-current">Job records</span>
            </nav>
          </div>
          <h1 className="data-acq-title">Job records</h1>
          <p className="data-acq-subtitle">
            Review the sample details produced by an AI labeling job.
          </p>
        </div>
      </div>

      {/* Records card */}
      <section className="dataset-main-panel card !p-0 overflow-hidden flex flex-col min-h-0 lg:h-[calc(100vh-220px)]">

        {/* Job selector + filter */}
        <div
          className="shrink-0 flex flex-wrap items-center gap-3 px-4 py-3"
          style={{ borderBottom: "1px solid var(--app-border)" }}
        >
          <select
            className="input text-xs py-1.5 flex-1 min-w-[220px]"
            value={activeJob?.id || ""}
            onChange={e => { const j = jobs.find((j: any) => j.id === e.target.value); setActiveJob(j || null); }}
          >
            <option value="">Select job...</option>
            {jobs.map((j: any) => (
              <option key={j.id} value={j.id}>[{j.status}] {new Date(j.created_at).toLocaleString()} — {j.processed || 0} samples</option>
            ))}
          </select>
          <select className="input text-xs py-1.5 w-32" value={filterStatus} onChange={e => setFilterStatus(e.target.value)}>
            <option value="all">All statuses</option>
            <option value="pending">Pending</option>
            <option value="approved">Approved</option>
            <option value="rejected">Rejected</option>
          </select>
          {activeJob && (
            <div className="flex items-center gap-2 ml-auto">
              <span className="text-[11px] whitespace-nowrap tabular-nums" style={{ color: "var(--app-text-soft)" }}>
                Showing{" "}
                <span className="font-semibold" style={{ color: "var(--app-text-muted)" }}>{filteredPredictions.length}</span>
                {" "}of {predictionTotal || activeJob.processed || 0}
              </span>
              <button
                onClick={() => loadPredictions(activeJob.id)}
                disabled={loading}
                className="p-1.5 rounded-md transition-colors disabled:opacity-50 hover:bg-[color:var(--app-surface-2)]"
                style={{ color: "var(--app-text-muted)" }}
                title="Refresh records"
              >
                <RefreshCw size={13} className={loading ? "animate-spin" : ""} />
              </button>
            </div>
          )}
        </div>

        {/* Records table */}
        {filteredPredictions.length === 0 ? (
          <div className="flex-1 flex flex-col items-center justify-center py-16" style={{ color: "var(--app-text-soft)" }}>
            <Cpu size={28} className="mb-3 opacity-20" />
            <p className="text-sm">{activeJob ? "No records match the filter" : "No AI labeling jobs yet"}</p>
          </div>
        ) : (
          <div className="w-full flex-1 overflow-x-auto overflow-y-auto min-h-0">
            <table className="w-full table-fixed">
              <thead
                className="sticky top-0 z-10"
                style={{ background: "var(--app-surface)", boxShadow: "inset 0 -1px 0 var(--app-border)" }}
              >
                <tr className="text-[11px] uppercase tracking-wider font-semibold" style={{ color: "var(--app-text-soft)" }}>
                  <th className="text-left px-4 py-2.5">Sample</th>
                  <th className="text-left px-4 py-2.5">Predicted Label</th>
                  <th className="text-left px-4 py-2.5 w-44">Confidence</th>
                  <th className="text-left px-4 py-2.5 w-28">Status</th>
                </tr>
              </thead>
              <tbody>
                {filteredPredictions.map((p: any) => {
                  const conf = typeof p.confidence === "number" ? p.confidence : 0;
                  const pct = Math.round(conf * 100);
                  const cs = confidenceStyle(conf);
                  return (
                    <tr
                      key={p.id}
                      className="transition-colors hover:bg-[color:var(--app-surface-2)]"
                      style={{ borderTop: "1px solid var(--app-border)" }}
                    >
                      <td className="px-4 py-3 align-middle">
                        <span className="text-xs font-mono truncate block" style={{ color: "var(--app-text)" }}>
                          {(p.sample_filename || p.sample_id)?.split("/").pop()?.split("\\").pop() || p.sample_id?.slice(0, 8)}
                        </span>
                      </td>
                      <td className="px-4 py-3 align-middle">
                        <div className="flex items-center gap-2 flex-wrap">
                          <span className="badge-blue text-xs">
                            {p.predicted_label || "Unlabelled"}
                          </span>
                          {p.low_confidence && (
                            <span className="inline-flex items-center gap-1 text-[10px] font-medium text-yellow-500" title="Model confidence was below threshold, but this was the best available guess.">
                              <AlertTriangle size={10} /> Low conf
                            </span>
                          )}
                          {p.bounding_boxes?.length > 0 && (
                            <span className="text-[10px]" style={{ color: "var(--app-text-soft)" }}>
                              {p.bounding_boxes.length} box{p.bounding_boxes.length !== 1 ? "es" : ""}
                            </span>
                          )}
                        </div>
                      </td>
                      <td className="px-4 py-3 align-middle">
                        <div className="flex items-center gap-2.5">
                          <div className="flex-1 h-1.5 rounded-full overflow-hidden" style={{ background: "var(--app-surface-2)" }}>
                            <div className="h-full rounded-full" style={{ width: `${pct}%`, background: cs.bar }} />
                          </div>
                          <span className="text-xs font-mono tabular-nums w-9 text-right shrink-0" style={{ color: cs.text }}>
                            {pct}%
                          </span>
                        </div>
                      </td>
                      <td className="px-4 py-3 align-middle">
                        <span className={`${statusBadgeClass(p.status)} capitalize`}>
                          {p.status}
                        </span>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </section>
    </div>
  );
}

export default function JobRecordsPage() {
  return (
    <Suspense fallback={
      <div className="flex items-center justify-center py-16" style={{ color: "var(--app-text-soft)" }}>
        <Loader2 size={20} className="animate-spin mr-2" /> Loading…
      </div>
    }>
      <JobRecordsContent />
    </Suspense>
  );
}
