"use client";
import Link from "next/link";
import { CheckCircle2, ImageOff, PartyPopper, Sparkles } from "lucide-react";

// Labeling Queue plan, Phase 4 §2.8/§4.A — the three non-editor screens the
// queue can render instead of <AnnotationEditor/>. The editor is never
// mounted for any of these variants.
type Variant = "no-samples" | "nothing-to-label" | "session-complete";

type SplitTotals = { training: number; testing: number; postprocessing: number };

type QueueEmptyStateProps = {
  variant: Variant;
  labeledThisSession?: number;
  splitTotals?: SplitTotals;
  onStartAiLabeling?: () => void;
};

export default function QueueEmptyState({ variant, labeledThisSession = 0, splitTotals, onStartAiLabeling }: QueueEmptyStateProps) {
  if (variant === "no-samples") {
    return (
      <div className="labeling-queue-empty">
        <div className="labeling-queue-empty-icon labeling-queue-empty-icon--muted">
          <ImageOff size={26} strokeWidth={1.75} />
        </div>
        <h2 className="labeling-queue-empty-title">This project has no images yet</h2>
        <p className="labeling-queue-empty-sub">
          Upload some images before starting the labeling queue.
        </p>
        <Link href="/dashboard/data" className="data-acq-btn-primary">
          Go to Data acquisition
        </Link>
      </div>
    );
  }

  if (variant === "nothing-to-label") {
    return (
      <div className="labeling-queue-empty">
        <div className="labeling-queue-empty-icon labeling-queue-empty-icon--success">
          <CheckCircle2 size={26} strokeWidth={1.75} />
        </div>
        <h2 className="labeling-queue-empty-title">All images are labeled</h2>
        <p className="labeling-queue-empty-sub">
          There&apos;s nothing left to label in this project right now.
        </p>
        <Link href="/dashboard/data/dataset" className="data-acq-btn-primary">
          Back to Dataset
        </Link>
      </div>
    );
  }

  const total = (splitTotals?.training ?? 0) + (splitTotals?.testing ?? 0) + (splitTotals?.postprocessing ?? 0);

  return (
    <div className="labeling-queue-empty">
      <div className="labeling-queue-empty-icon labeling-queue-empty-icon--celebrate">
        <PartyPopper size={26} strokeWidth={1.75} />
      </div>
      <h2 className="labeling-queue-empty-title">Queue complete</h2>
      <p className="labeling-queue-empty-sub">
        You labeled <strong>{labeledThisSession}</strong> image{labeledThisSession === 1 ? "" : "s"} this session.
        Remaining: <strong>0</strong>.
      </p>
      {splitTotals && total > 0 && (
        <p className="labeling-queue-empty-splits">
          Training {splitTotals.training} · Test {splitTotals.testing} · Post-processing {splitTotals.postprocessing}
        </p>
      )}
      <div className="labeling-queue-empty-actions">
        <Link href="/dashboard/data/dataset" className="data-acq-btn-primary">
          Return to Dataset
        </Link>
        <Link
          href="/dashboard/data/dataset?view=ai-labeling"
          className="labeling-queue-btn-secondary"
          onClick={onStartAiLabeling}
        >
          <Sparkles size={13} /> Start AI Labeling
        </Link>
      </div>
    </div>
  );
}
