"use client";

// Shared pager for the dataset explorer's grid/detailed page (DetailedSampleGrid)
// and its list-table page. Moved out of ObjectDetectionLabeling verbatim — same
// "Showing X–Y of Z" + Previous/Page N of M/Next markup both call sites used
// independently before this extraction. `className` lets each call site keep
// its own (slightly different) wrapper spacing.

export interface PaginationProps {
  page: number;
  pageSize: number;
  totalItems: number;
  onPageChange: (page: number) => void;
  className?: string;
}

export function Pagination({
  page, pageSize, totalItems, onPageChange,
  className = "flex items-center justify-between py-3 text-xs",
}: PaginationProps) {
  const totalPages = Math.max(1, Math.ceil(totalItems / pageSize));
  const currentPage = Math.min(page, totalPages);
  const start = (currentPage - 1) * pageSize;

  if (totalPages <= 1) return null;

  return (
    <div className={className} style={{ color: "var(--app-text-soft)" }}>
      <span>
        Showing {start + 1}–{Math.min(start + pageSize, totalItems)} of {totalItems}
      </span>
      <div className="flex items-center gap-2">
        <button
          className="px-2.5 py-1 rounded-md border"
          style={{
            background: "var(--app-surface)",
            borderColor: "var(--app-border)",
            color: "var(--app-text)",
            opacity: currentPage === 1 ? 0.45 : 1,
            cursor: currentPage === 1 ? "not-allowed" : "pointer",
          }}
          onClick={() => currentPage > 1 && onPageChange(currentPage - 1)}
          disabled={currentPage === 1}
        >
          Previous
        </button>
        <span style={{ color: "var(--app-text)" }}>
          Page {currentPage} / {totalPages}
        </span>
        <button
          className="px-2.5 py-1 rounded-md border"
          style={{
            background: "var(--app-surface)",
            borderColor: "var(--app-border)",
            color: "var(--app-text)",
            opacity: currentPage === totalPages ? 0.45 : 1,
            cursor: currentPage === totalPages ? "not-allowed" : "pointer",
          }}
          onClick={() => currentPage < totalPages && onPageChange(currentPage + 1)}
          disabled={currentPage === totalPages}
        >
          Next
        </button>
      </div>
    </div>
  );
}
