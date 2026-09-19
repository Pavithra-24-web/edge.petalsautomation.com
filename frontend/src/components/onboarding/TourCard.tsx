"use client";
/**
 * Presentational card used inside both the centered modal and the anchored
 * spotlight tooltip. Renders the step copy, a progress indicator, and the
 * Back / Next / Skip controls. All behaviour comes in via props so the same
 * markup serves every step.
 */
import { X } from "lucide-react";

type TourCardProps = {
  title: string;
  body: string;
  stepIndex: number;
  total: number;
  isFirst: boolean;
  isLast: boolean;
  onNext: () => void;
  onBack: () => void;
  onSkip: () => void;
  onGoTo: (index: number) => void;
};

export default function TourCard({
  title,
  body,
  stepIndex,
  total,
  isFirst,
  isLast,
  onNext,
  onBack,
  onSkip,
  onGoTo,
}: TourCardProps) {
  return (
    <div className="pe-tour-card">
      <button
        type="button"
        className="pe-tour-close"
        aria-label="Skip tutorial"
        onClick={onSkip}
      >
        <X size={16} />
      </button>

      <div className="pe-tour-step-count">
        Step {stepIndex + 1} of {total}
      </div>
      <h3 className="pe-tour-title">{title}</h3>
      <p className="pe-tour-body">{body}</p>

      <div className="pe-tour-dots" role="tablist" aria-label="Tutorial progress">
        {Array.from({ length: total }).map((_, i) => (
          <button
            key={i}
            type="button"
            aria-label={`Go to step ${i + 1}`}
            aria-selected={i === stepIndex}
            className={`pe-tour-dot ${i === stepIndex ? "is-active" : ""} ${
              i < stepIndex ? "is-done" : ""
            }`}
            onClick={() => onGoTo(i)}
          />
        ))}
      </div>

      <div className="pe-tour-actions">
        <button type="button" className="pe-tour-skip" onClick={onSkip}>
          Skip tour
        </button>
        <div className="pe-tour-nav">
          {!isFirst && (
            <button type="button" className="btn-secondary pe-tour-btn" onClick={onBack}>
              Back
            </button>
          )}
          <button type="button" className="btn-primary pe-tour-btn" onClick={onNext}>
            {isLast ? "Finish" : "Next"}
          </button>
        </div>
      </div>
    </div>
  );
}
