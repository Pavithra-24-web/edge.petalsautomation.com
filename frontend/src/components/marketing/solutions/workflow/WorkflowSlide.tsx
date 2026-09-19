"use client";

import WorkflowMedia from "./WorkflowMedia";
import { stageAccent } from "./palette";
import type { WorkflowStage } from "@/lib/solutions.config";

/* ---------------------------------------------------------------------------
   WorkflowSlide — one stage of the pipeline.

   Copy leads, media dominates. The stage accent arrives as --c and lights the
   media frame, the step rule and the ordinal, so a slide's colour states where
   in the pipeline the reader is without a legend.

   The panel is a tabpanel in the rail's tablist. Inactive panels sit off-screen
   inside the track and hold no focusable content, so aria-hidden on them is
   safe and keeps them out of the accessibility tree entirely.
--------------------------------------------------------------------------- */
export default function WorkflowSlide({
  stage,
  index,
  total,
  active,
  idBase,
}: {
  stage: WorkflowStage;
  index: number;
  total: number;
  active: boolean;
  /** Shared id prefix, so tab ↔ panel wiring survives multiple variants. */
  idBase: string;
}) {
  return (
    <li
      className={`pe-wf-slide ${active ? "is-active" : ""}`}
      style={{ ["--c" as string]: stageAccent(index, total) }}
      role="tabpanel"
      id={`${idBase}-panel-${index}`}
      aria-labelledby={`${idBase}-tab-${index}`}
      aria-hidden={!active}
    >
      <div className="pe-wf-copy">
        <p className="pe-wf-step">
          <span className="pe-wf-step-num">{String(index + 1).padStart(2, "0")}</span>
          <span className="pe-wf-step-rule" aria-hidden="true" />
          <span className="pe-wf-step-of">Stage {index + 1} of {total}</span>
        </p>

        <h3 className="pe-wf-title">{stage.name}</h3>
        <p className="pe-wf-desc">{stage.desc}</p>
        <p className="pe-wf-where">{stage.where}</p>
      </div>

      <div className="pe-wf-frame">
        <span className="pe-wf-frame-glow" aria-hidden="true" />
        <WorkflowMedia stage={stage} step={index + 1} active={active} />
      </div>
    </li>
  );
}
