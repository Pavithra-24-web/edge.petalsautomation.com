"use client";

import { forwardRef } from "react";
import { stageAccent } from "./palette";
import type { WorkflowStage } from "@/lib/solutions.config";

/* ---------------------------------------------------------------------------
   WorkflowProgress — the pipeline rail.

   This is the progress indicator and the diagram at once. A node per stage sits
   on a hairline; the fill travelling that hairline is the autoplay clock, so
   the reader watches the pipeline advance rather than a bar unrelated to the
   content. The fill's colour walks the brand gradient across the run, matching
   each slide's own accent.

   Position comes from two custom properties the carousel writes directly:
   --wf-i (whole slides completed) and --wf-p (fraction of the current slide,
   updated per animation frame without a React render).

   Structure is a tablist: each node is a tab controlling its slide, with roving
   tabindex and the arrow / Home / End keys the pattern expects.
--------------------------------------------------------------------------- */
const WorkflowProgress = forwardRef<
  HTMLDivElement,
  {
    stages: WorkflowStage[];
    index: number;
    idBase: string;
    label: string;
    onSelect: (i: number) => void;
  }
>(function WorkflowProgress({ stages, index, idBase, label, onSelect }, ref) {
  const total = stages.length;

  const onKeyDown = (e: React.KeyboardEvent<HTMLDivElement>) => {
    const move =
      e.key === "ArrowRight" ? index + 1
      : e.key === "ArrowLeft" ? index - 1
      : e.key === "Home" ? 0
      : e.key === "End" ? total - 1
      : null;
    if (move === null) return;

    e.preventDefault();
    const next = ((move % total) + total) % total;
    onSelect(next);
    // Selection follows focus in this pattern, so the focus ring has to travel
    // with it or the next arrow key would move from the old node.
    const el = e.currentTarget.querySelector<HTMLButtonElement>(
      `#${idBase}-tab-${next}`
    );
    el?.focus();
  };

  return (
    <div
      ref={ref}
      className="pe-wf-rail"
      style={{
        ["--wf-n" as string]: total,
        ["--wf-den" as string]: Math.max(total - 1, 1),
        ["--wf-i" as string]: index,
        // Half a column, so the hairline starts and stops at the outer dot
        // centres instead of running past them. Resolved here rather than as
        // calc(50% / var(--wf-n)) — a plain value needs no browser to agree
        // about dividing a percentage by a custom property.
        ["--wf-half" as string]: `${50 / total}%`,
      }}
      role="tablist"
      aria-label={label}
      aria-orientation="horizontal"
      onKeyDown={onKeyDown}
    >
      <span className="pe-wf-rail-line" aria-hidden="true">
        <span className="pe-wf-rail-fill" />
      </span>

      {stages.map((stage, i) => (
        <button
          key={stage.name}
          type="button"
          id={`${idBase}-tab-${i}`}
          className={`pe-wf-node ${i === index ? "is-active" : ""} ${i < index ? "is-done" : ""}`}
          style={{ ["--c" as string]: stageAccent(i, total) }}
          role="tab"
          aria-selected={i === index}
          aria-controls={`${idBase}-panel-${i}`}
          tabIndex={i === index ? 0 : -1}
          onClick={() => onSelect(i)}
        >
          <span className="pe-wf-node-dot" aria-hidden="true" />
          <span className="pe-wf-node-label">{stage.name}</span>
        </button>
      ))}
    </div>
  );
});

export default WorkflowProgress;
