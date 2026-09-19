"use client";
import { AlertTriangle, ChevronLeft, ChevronRight, Loader2 } from "lucide-react";

// Labeling Queue plan, Phase 4 §3/§4.A/§2.2/§4.D.

type QueueToolbarProps = {
  onPrev: () => void;
  onNext: () => void;
  onSave: () => void;
  canPrev: boolean;
  canNext: boolean;
  saving: boolean;
  blockedMessage: string | null;
  saveError: string | null;
  onRetry: () => void;
  onSkip: () => void;
};

export default function QueueToolbar({
  onPrev, onNext, onSave, canPrev, canNext, saving,
  blockedMessage, saveError, onRetry, onSkip,
}: QueueToolbarProps) {
  return (
    <div className="labeling-queue-toolbar">
      {(blockedMessage || saveError) && (
        <div className="labeling-queue-toolbar-banner" role="alert">
          <AlertTriangle size={14} />
          <span>{saveError || blockedMessage}</span>
          {saveError && (
            <span className="labeling-queue-toolbar-banner-actions">
              <button type="button" onClick={onRetry}>Retry</button>
              <button type="button" onClick={onSkip}>Skip for now</button>
            </span>
          )}
        </div>
      )}
      <div className="labeling-queue-toolbar-row">
        <button type="button" className="labeling-queue-btn-secondary" onClick={onPrev} disabled={!canPrev}>
          <ChevronLeft size={14} /> Previous
        </button>
        <button type="button" className="data-acq-btn-primary labeling-queue-save-btn" onClick={onSave} disabled={saving}>
          {saving ? <Loader2 size={14} className="animate-spin" /> : null}
          Save Labels
        </button>
        <button type="button" className="labeling-queue-btn-secondary" onClick={onNext} disabled={!canNext}>
          Next <ChevronRight size={14} />
        </button>
      </div>
      <p className="labeling-queue-toolbar-hint">Auto-advance: on · Ctrl/Cmd+S to save · ← → to navigate</p>
    </div>
  );
}
