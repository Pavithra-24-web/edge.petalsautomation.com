"use client";

import { useEffect, useMemo, useState } from "react";
import { ChevronLeft, ChevronRight, MoreVertical } from "lucide-react";
import type { TestSampleRow } from "@/types/model-testing";
import SampleThumbnail from "./SampleThumbnail";

interface TestDataTableProps {
  samples: TestSampleRow[];
  onUpdateSample?: (id: string, expected_outcome: string) => void;
  selectedSampleId?: string | null;
  onSelectSample?: (sample: TestSampleRow) => void;
}

const PAGE_SIZE = 6;

export default function TestDataTable({
  samples,
  selectedSampleId,
  onSelectSample,
}: TestDataTableProps) {
  const [openMenu, setOpenMenu] = useState<string | null>(null);
  const [page, setPage] = useState(1);

  const scoreLabel = samples.some((s) => s.det_stats !== null) ? "Match score" : "F1 score";

  const isDetPassWithFP = (s: TestSampleRow) =>
    s.result_status === "uncertain" && s.det_stats !== null && (s.det_stats.fn ?? 1) === 0;

  const total = samples.length;
  const totalPages = Math.max(1, Math.ceil(total / PAGE_SIZE));

  // Keep page valid if sample list shrinks
  useEffect(() => {
    if (page > totalPages) setPage(totalPages);
  }, [page, totalPages]);

  // Auto-jump to the page containing the selected sample
  useEffect(() => {
    if (!selectedSampleId) return;
    const idx = samples.findIndex((s) => s.id === selectedSampleId);
    if (idx < 0) return;
    const targetPage = Math.floor(idx / PAGE_SIZE) + 1;
    if (targetPage !== page) setPage(targetPage);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedSampleId]);

  const pageStart = (page - 1) * PAGE_SIZE;
  const pageItems = useMemo(
    () => samples.slice(pageStart, pageStart + PAGE_SIZE),
    [samples, pageStart],
  );

  const showingFrom = total === 0 ? 0 : pageStart + 1;
  const showingTo = Math.min(total, pageStart + PAGE_SIZE);

  const rowClassFor = (s: TestSampleRow): string => {
    const score = s.precision_score ?? s.f1_score;
    if (selectedSampleId === s.id) return "pe-mt-row-selected";
    if (s.result_status === "pending" || score === null) return "";
    if (isDetPassWithFP(s) || s.result_status === "pass") return "pe-mt-row-pass";
    if (s.result_status === "uncertain") return "pe-mt-row-uncertain";
    return "pe-mt-row-fail";
  };

  const renderResult = (s: TestSampleRow) => {
    if (isDetPassWithFP(s)) {
      return (
        <span className="inline-flex items-center gap-1.5">
          <span className="pe-mt-chip-pass">Pass</span>
          <span
            className="pe-mt-chip-mixed"
            title={`${s.det_stats!.fp} extra prediction${s.det_stats!.fp !== 1 ? "s" : ""} not matched to ground truth`}
          >
            +{s.det_stats!.fp} FP
          </span>
        </span>
      );
    }
    if (s.result_status === "pass") return <span className="pe-mt-chip-pass">Pass</span>;
    if (s.result_status === "fail") return <span className="pe-mt-chip-fail">Fail</span>;
    if (s.result_status === "uncertain") {
      return (
        <span className="pe-mt-chip-mixed" title="Correct prediction but confidence is below threshold">
          Mixed
        </span>
      );
    }
    return <span className="pe-mt-chip-pending">—</span>;
  };

  const renderScore = (s: TestSampleRow) => {
    const v = s.precision_score ?? s.f1_score;
    if (v === null) return <span className="text-[color:var(--app-text-muted)]">-</span>;
    return <span className="pe-mt-score">{v}%</span>;
  };

  // Build page-number sequence with ellipsis
  const pageList = useMemo(() => buildPageList(page, totalPages), [page, totalPages]);

  return (
    <>
      <div className="pe-mt-table-wrap">
        <table className="pe-mt-table">
          <thead>
            <tr>
              <th style={{ width: 60 }}>Preview</th>
              <th>Sample name</th>
              <th>Expected outcome</th>
              <th>{scoreLabel}</th>
              <th>Result</th>
              <th style={{ width: 40 }} aria-label="actions" />
            </tr>
          </thead>
          <tbody>
            {pageItems.map((sample) => (
              <tr
                key={sample.id}
                className={rowClassFor(sample)}
                onClick={() => onSelectSample?.(sample)}
              >
                <td>
                  <SampleThumbnail sampleId={sample.sample_id} size="sm" />
                </td>
                <td>
                  <span className="pe-mt-name">{sample.sample_name}</span>
                </td>
                <td>
                  {sample.expected_outcome && sample.expected_outcome !== "-" ? (
                    <span className="pe-mt-chip-label">{sample.expected_outcome}</span>
                  ) : (
                    <span className="text-[color:var(--app-text-muted)]">-</span>
                  )}
                </td>
                <td>{renderScore(sample)}</td>
                <td>{renderResult(sample)}</td>
                <td style={{ position: "relative" }}>
                  <button
                    type="button"
                    className="pe-mt-kebab"
                    aria-label="sample-actions"
                    onClick={(e) => {
                      e.stopPropagation();
                      setOpenMenu(openMenu === sample.id ? null : sample.id);
                    }}
                  >
                    <MoreVertical size={15} />
                  </button>
                  {openMenu === sample.id && (
                    <>
                      <div
                        className="fixed inset-0 z-40"
                        onClick={() => setOpenMenu(null)}
                      />
                      <div
                        className="absolute right-2 top-full z-50 min-w-[140px] rounded-lg border py-1 shadow-lg"
                        style={{
                          background: "var(--app-surface)",
                          borderColor: "var(--app-border)",
                        }}
                      >
                        {["View sample", "Classify", "Delete"].map((action) => (
                          <button
                            key={action}
                            type="button"
                            onClick={(e) => {
                              e.stopPropagation();
                              setOpenMenu(null);
                            }}
                            className="w-full text-left px-4 py-2 text-sm hover:bg-violet-50 dark:hover:bg-slate-800"
                            style={{ color: "var(--app-text)" }}
                          >
                            {action}
                          </button>
                        ))}
                      </div>
                    </>
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div className="pe-mt-pagination">
        <span className="pe-mt-pagination-count">
          Showing {showingFrom}–{showingTo} of {total} samples
        </span>
        <div className="pe-mt-pagination-controls">
          <button
            type="button"
            className="pe-mt-page-btn"
            disabled={page <= 1}
            onClick={() => setPage((p) => Math.max(1, p - 1))}
            aria-label="previous-page"
          >
            <ChevronLeft size={15} />
          </button>
          {pageList.map((entry, i) =>
            entry === "..." ? (
              <span key={`e-${i}`} className="pe-mt-page-ellipsis">…</span>
            ) : (
              <button
                key={entry}
                type="button"
                className={`pe-mt-page-btn ${entry === page ? "is-active" : ""}`}
                onClick={() => setPage(entry)}
              >
                {entry}
              </button>
            ),
          )}
          <button
            type="button"
            className="pe-mt-page-btn"
            disabled={page >= totalPages}
            onClick={() => setPage((p) => Math.min(totalPages, p + 1))}
            aria-label="next-page"
          >
            <ChevronRight size={15} />
          </button>
        </div>
      </div>
    </>
  );
}

function buildPageList(current: number, total: number): (number | "...")[] {
  if (total <= 7) return Array.from({ length: total }, (_, i) => i + 1);
  const out: (number | "...")[] = [];
  const push = (v: number | "...") => {
    if (out[out.length - 1] === v) return;
    out.push(v);
  };
  push(1);
  if (current > 3) push("...");
  for (let p = Math.max(2, current - 1); p <= Math.min(total - 1, current + 1); p++) {
    push(p);
  }
  if (current < total - 2) push("...");
  push(total);
  return out;
}
