"use client";
import Link from "next/link";
import { ChevronRight, Home, LogOut } from "lucide-react";
import type { QueueSplit } from "@/hooks/useLabelingQueue";

// Labeling Queue plan, Phase 4 §3/§4.A/§2.9/§2.11.

const SPLIT_LABEL: Record<QueueSplit, string> = {
  training: "Training",
  testing: "Test",
  postprocessing: "Post-processing",
};

const SPLIT_PILL_CLASS: Record<QueueSplit, string> = {
  training: "data-acq-pill--purple",
  testing: "data-acq-pill--blue",
  postprocessing: "data-acq-pill--green",
};

type QueueHeaderProps = {
  filename: string | null;
  split: QueueSplit | null;
  position: number;
  total: number;
  remainingCount: number;
  progressPct: number;
  confirmOpen: boolean;
  isFinished: boolean;
  onExitClick: () => void;
  onStay: () => void;
  onConfirmExit: () => void;
};

export default function QueueHeader({
  filename, split, position, total, remainingCount, progressPct,
  confirmOpen, isFinished, onExitClick, onStay, onConfirmExit,
}: QueueHeaderProps) {
  return (
    <div className="flex items-start justify-between gap-6 flex-wrap">
      <div className="min-w-0 flex-1">
        <nav className="data-acq-breadcrumb" aria-label="Breadcrumb">
          <Home size={13} aria-hidden="true" />
          <ChevronRight size={12} aria-hidden="true" />
          <Link href="/dashboard/data/dataset" className="crumb-link">Data Labeling</Link>
          <ChevronRight size={12} aria-hidden="true" />
          <span className="crumb-current">Labeling Queue</span>
        </nav>
        <h1 className="data-acq-title">Labeling Queue</h1>
        <p className="data-acq-subtitle labeling-queue-subtitle">
          {filename && (
            <>
              <span className="labeling-queue-subtitle-file truncate" title={filename}>{filename}</span>
              {split && <span className={`data-acq-pill ${SPLIT_PILL_CLASS[split]}`}>{SPLIT_LABEL[split]}</span>}
              <span className="labeling-queue-subtitle-sep" aria-hidden="true">·</span>
            </>
          )}
          {total > 0 ? (
            <span>Image {position} of {total} · {remainingCount} remaining</span>
          ) : (
            <span>Review, label and curate every unlabeled sample in this project.</span>
          )}
        </p>
      </div>

      <div className="flex flex-col items-end gap-2 flex-shrink-0 labeling-queue-header-right">
        {total > 0 && (
          <div className="labeling-queue-progress" role="progressbar" aria-valuenow={Math.round(progressPct)} aria-valuemin={0} aria-valuemax={100}>
            <div className="labeling-queue-progress-fill" style={{ width: `${progressPct}%` }} />
          </div>
        )}
        <button type="button" className="labeling-queue-btn-secondary" onClick={onExitClick}>
          <LogOut size={13} /> Exit Queue
        </button>
      </div>

      {confirmOpen && (
        <div className="labeling-queue-dialog-overlay" role="dialog" aria-modal="true" onClick={onStay}>
          <div className="labeling-queue-dialog" onClick={e => e.stopPropagation()}>
            <h3 className="labeling-queue-dialog-title">
              {isFinished ? "Leave the labeling queue?" : "Leave the labeling queue?"}
            </h3>
            <p className="labeling-queue-dialog-body">
              Your progress is saved — every image you&apos;ve labeled is already stored, and
              the rest will still be waiting here next time you open the queue.
            </p>
            <div className="labeling-queue-dialog-actions">
              <button type="button" className="labeling-queue-btn-secondary" onClick={onStay}>Stay</button>
              <button type="button" className="data-acq-btn-primary" onClick={onConfirmExit}>Exit to Dataset</button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
