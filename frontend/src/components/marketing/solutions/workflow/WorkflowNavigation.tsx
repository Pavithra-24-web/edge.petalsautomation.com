"use client";

import { ChevronLeft, ChevronRight, Pause, Play } from "lucide-react";

/* ---------------------------------------------------------------------------
   WorkflowNavigation — the control cluster beside the section heading.

   Three controls, one job each: step back, step forward, and hold the pipeline
   where it is. The pause control is not optional polish — content that starts
   moving on its own needs a way to stop it, and hovering is not a mechanism a
   keyboard or touch reader can use.

   The counter reads as position in a sequence, not as decoration, so it uses
   tabular figures and never reflows as the number changes.
--------------------------------------------------------------------------- */
export default function WorkflowNavigation({
  index,
  total,
  paused,
  onPrev,
  onNext,
  onTogglePause,
}: {
  index: number;
  total: number;
  paused: boolean;
  onPrev: () => void;
  onNext: () => void;
  onTogglePause: () => void;
}) {
  return (
    <div className="pe-wf-nav">
      <p className="pe-wf-counter">
        <span className="pe-wf-counter-now">{String(index + 1).padStart(2, "0")}</span>
        <span className="pe-wf-counter-sep" aria-hidden="true">/</span>
        <span className="pe-wf-counter-all">{String(total).padStart(2, "0")}</span>
      </p>

      <div className="pe-wf-nav-btns">
        <button
          type="button"
          className="pe-wf-nav-btn"
          onClick={onTogglePause}
          aria-pressed={paused}
        >
          {paused ? <Play /> : <Pause />}
          <span className="pe-sr-only">
            {paused ? "Play the walkthrough" : "Pause the walkthrough"}
          </span>
        </button>
        <button type="button" className="pe-wf-nav-btn" onClick={onPrev}>
          <ChevronLeft />
          <span className="pe-sr-only">Previous stage</span>
        </button>
        <button type="button" className="pe-wf-nav-btn" onClick={onNext}>
          <ChevronRight />
          <span className="pe-sr-only">Next stage</span>
        </button>
      </div>
    </div>
  );
}
