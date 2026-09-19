"use client";

import { CheckCircle, Database, Layers, Loader2 } from "lucide-react";
import TestDataTable from "./TestDataTable";
import type { TestSampleRow } from "@/types/model-testing";

interface TestDataCardProps {
  samples:          TestSampleRow[];
  loading:          boolean;
  classifying:      boolean;
  selectedSampleId?: string | null;
  onClassifyAll:    () => void;
  onSelectSample?:  (sample: TestSampleRow) => void;
  onUpdateSample?:  (id: string, expected_outcome: string) => void;
}

function TableSkeleton() {
  return (
    <div className="px-6 py-4 space-y-3" aria-label="test-data-skeleton">
      {Array.from({ length: 5 }).map((_, i) => (
        <div key={i} className="flex items-center gap-4 animate-pulse">
          <div className="h-11 w-11 bg-gray-200 dark:bg-slate-800 rounded-lg" />
          <div className="h-3 flex-1 bg-gray-200 dark:bg-slate-800 rounded" />
          <div className="h-3 w-16 bg-gray-200 dark:bg-slate-800 rounded" />
          <div className="h-3 w-12 bg-gray-200 dark:bg-slate-800 rounded" />
          <div className="h-3 w-12 bg-gray-200 dark:bg-slate-800 rounded" />
        </div>
      ))}
    </div>
  );
}

function EmptyState() {
  return (
    <div className="flex flex-col items-center justify-center py-14 px-6 text-center">
      <div className="w-12 h-12 rounded-full flex items-center justify-center mb-3
                      bg-violet-100 text-violet-600 dark:bg-violet-500/10 dark:text-violet-300">
        <Database size={20} />
      </div>
      <p className="text-sm font-semibold text-[color:var(--app-text)] mb-1">
        No test samples yet
      </p>
      <p className="text-xs text-[color:var(--app-text-muted)] max-w-xs leading-relaxed">
        Upload samples via Data acquisition and assign them to the test split to see them here.
      </p>
    </div>
  );
}

export default function TestDataCard({
  samples,
  loading,
  classifying,
  selectedSampleId,
  onClassifyAll,
  onSelectSample,
  onUpdateSample,
}: TestDataCardProps) {
  return (
    <div className="pe-mt-card">
      <div className="pe-mt-card-head">
        <div className="pe-mt-card-head-left">
          <div className="pe-mt-card-icon" aria-hidden="true">
            <Layers size={18} strokeWidth={2.2} />
          </div>
          <div>
            <h2 className="pe-mt-card-title">Test data</h2>
          </div>
        </div>

        <div className="pe-mt-actions">
          <button
            type="button"
            onClick={onClassifyAll}
            disabled={classifying || loading}
            className="pe-mt-btn-classify"
            aria-label="classify-all"
          >
            {classifying ? (
              <Loader2 size={13} className="animate-spin" />
            ) : (
              <CheckCircle size={13} />
            )}
            {classifying ? "Classifying…" : "Classify all"}
          </button>
        </div>
      </div>

      <p className="pe-mt-helper">
        Click a row to inspect the sample. Set the &apos;expected outcome&apos; for each sample
        to automatically score the impulse.
      </p>

      {loading ? (
        <TableSkeleton />
      ) : samples.length === 0 ? (
        <EmptyState />
      ) : (
        <TestDataTable
          samples={samples}
          selectedSampleId={selectedSampleId}
          onSelectSample={onSelectSample}
          onUpdateSample={onUpdateSample}
        />
      )}
    </div>
  );
}
