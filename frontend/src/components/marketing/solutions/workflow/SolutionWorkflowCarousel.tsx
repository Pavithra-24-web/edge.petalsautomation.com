"use client";

import { useCallback, useEffect, useId, useRef, useState } from "react";
import { Cable } from "lucide-react";
import { Reveal } from "../../primitives";
import WorkflowSlide from "./WorkflowSlide";
import WorkflowProgress from "./WorkflowProgress";
import WorkflowNavigation from "./WorkflowNavigation";
import {
  useInView,
  usePageVisible,
  useReducedMotion,
  useWorkflowCarousel,
} from "./useWorkflowCarousel";
import { WORKFLOW_VARIANTS, type Solution } from "@/lib/solutions.config";

/* ---------------------------------------------------------------------------
   SolutionWorkflowCarousel — the product walkthrough.

   Replaces the stage-card grid. The pipeline is a sequence, so it is presented
   as one: a single stage at a time, advancing on a clock, with the rail beneath
   showing where in the run the reader is and how far the current stage has to
   go. Nothing here is imported from the homepage and no asset is shared with
   it; every slide carries its own media through solutions.config.

   The three ways to move through it are the same movement underneath:
     · autoplay, which holds while the pointer rests, focus is inside, the tab
       is hidden, the section is off screen, or the reader has paused it;
     · the rail, arrows and arrow keys;
     · a drag on touch and pen, which follows the finger and settles either way.

   Pages with two pipelines (Manufacturing runs vision and sensor) get a switch
   above the stage rather than a second carousel.
--------------------------------------------------------------------------- */

/** How long one stage holds before the clock advances. */
const STAGE_MS = 6500;
/** Drag distance, in px, that commits to the neighbouring stage. */
const SWIPE_PX = 48;

export default function SolutionWorkflowCarousel({
  solution,
  embedded = false,
}: {
  solution: Solution;
  /**
   * The carousel is sitting inside a section that someone else owns — today the
   * homepage's #platform, which brings its own heading and landmark. One flag
   * rather than two because it is one fact: not owning the section means not
   * owning the <section>/id, and not owning the heading means not rendering the
   * eyebrow/h2/lede either. Defaults to false, so /solutions/computer-vision
   * passes nothing and renders exactly as before.
   */
  embedded?: boolean;
}) {
  const variants = solution.workflow.map((id) => WORKFLOW_VARIANTS[id]);
  const [variantIndex, setVariantIndex] = useState(0);
  const variant = variants[Math.min(variantIndex, variants.length - 1)];
  const stages = variant.stages;
  const total = stages.length;

  const idBase = `wf${useId().replace(/[:]/g, "")}`;
  const sectionRef = useRef<HTMLDivElement | null>(null);
  const railRef = useRef<HTMLDivElement | null>(null);
  const trackRef = useRef<HTMLUListElement | null>(null);

  const [held, setHeld] = useState(false); // pointer resting or focus inside
  const [paused, setPaused] = useState(false); // reader's own choice
  const [dragging, setDragging] = useState(false);

  const inView = useInView(sectionRef);
  const pageVisible = usePageVisible();
  const reduced = useReducedMotion();

  const writeProgress = useCallback((p: number) => {
    railRef.current?.style.setProperty("--wf-p", String(p));
  }, []);

  const { index, go, next, prev } = useWorkflowCarousel({
    count: total,
    intervalMs: STAGE_MS,
    enabled: inView && pageVisible && !held && !paused && !dragging && !reduced,
    onProgress: writeProgress,
  });

  // Reduced motion has no clock to fill the rail, so the fill snaps to the
  // stage boundary instead of sitting permanently short of it.
  useEffect(() => {
    if (reduced) writeProgress(1);
  }, [reduced, index, writeProgress]);

  // Switching pipelines restarts the run — stage 4 of vision has nothing to do
  // with stage 4 of sensor.
  const selectVariant = (i: number) => {
    setVariantIndex(i);
    go(0);
  };

  /* ---- Drag to advance --------------------------------------------------
     Touch and pen only. A mouse has the arrows, the rail and the keyboard, and
     dragging one across a screenshot fights text selection and image ghosting
     for nothing. The track follows the pointer through --wf-drag so the
     gesture reads as direct manipulation rather than a delayed jump.        */
  const drag = useRef({ id: -1, x0: 0, dx: 0 });

  const onPointerDown = (e: React.PointerEvent) => {
    if (e.pointerType === "mouse" || total < 2) return;
    drag.current = { id: e.pointerId, x0: e.clientX, dx: 0 };
    setDragging(true);
  };

  const onPointerMove = (e: React.PointerEvent) => {
    if (drag.current.id !== e.pointerId) return;
    drag.current.dx = e.clientX - drag.current.x0;
    trackRef.current?.style.setProperty("--wf-drag", `${drag.current.dx}px`);
  };

  const endDrag = (e: React.PointerEvent) => {
    if (drag.current.id !== e.pointerId) return;
    const { dx } = drag.current;
    drag.current.id = -1;
    trackRef.current?.style.removeProperty("--wf-drag");
    setDragging(false);
    if (dx <= -SWIPE_PX) next();
    else if (dx >= SWIPE_PX) prev();
  };

  /* Same tree either way; only the element that carries it changes. Embedded it
     is a plain <div>, so the host section keeps the sole landmark and the sole
     id, and .pe-section's vertical padding is not applied twice. */
  const body = (
    <div className="pe-container">
      <div className="pe-wf-head">
        {!embedded && (
          <div className="pe-wf-head-copy">
            <Reveal as="span" className="pe-eyebrow"><Cable /> How it works</Reveal>
            <Reveal as="h2" className="pe-h2" delay={60}>
              The whole pipeline, <span className="pe-grad-text">one stage at a time</span>
            </Reveal>
            <Reveal as="p" className="pe-lede" delay={120}>
              Every stage below is a screen in the platform, not a diagram of something
              you still have to build.
            </Reveal>
          </div>
        )}

        <Reveal className="pe-wf-head-nav" delay={180}>
          <WorkflowNavigation
            index={index}
            total={total}
            paused={paused}
            onPrev={prev}
            onNext={next}
            onTogglePause={() => setPaused((p) => !p)}
          />
        </Reveal>
      </div>

      {variants.length > 1 && (
        <Reveal className="pe-wf-variants" delay={60}>
          {variants.map((v, i) => (
            <button
              key={v.label}
              type="button"
              className={`pe-wf-variant ${i === variantIndex ? "is-active" : ""}`}
              onClick={() => selectVariant(i)}
              aria-pressed={i === variantIndex}
            >
              {v.label}
            </button>
          ))}
        </Reveal>
      )}

      <Reveal
        className="pe-wf-stage"
        delay={120}
        onMouseEnter={() => setHeld(true)}
        onMouseLeave={() => setHeld(false)}
        onFocusCapture={() => setHeld(true)}
        onBlurCapture={() => setHeld(false)}
      >
        <div
          className={`pe-wf-viewport ${dragging ? "is-dragging" : ""}`}
          onPointerDown={onPointerDown}
          onPointerMove={onPointerMove}
          onPointerUp={endDrag}
          onPointerCancel={endDrag}
        >
          <ul
            ref={trackRef}
            className="pe-wf-track"
            style={{ ["--wf-x" as string]: `${-index * 100}%` }}
          >
            {stages.map((stage, i) => (
              <WorkflowSlide
                key={`${variant.label}-${stage.name}`}
                stage={stage}
                index={i}
                total={total}
                active={i === index}
                idBase={idBase}
              />
            ))}
          </ul>
        </div>

        <WorkflowProgress
          ref={railRef}
          stages={stages}
          index={index}
          idBase={idBase}
          label={`${variant.label} stages`}
          onSelect={go}
        />
      </Reveal>

      {/* Only says something when there are two pipelines to tell apart. On a
          single-pipeline page it would restate the lede above, so it is not
          rendered at all rather than padded out. */}
      {variants.length > 1 && <p className="pe-wf-note">{variant.desc}</p>}
    </div>
  );

  return embedded ? (
    <div className="pe-wf pe-wf-embedded" ref={sectionRef}>
      {body}
    </div>
  ) : (
    <section className="pe-section pe-wf" id="workflow" ref={sectionRef}>
      {body}
    </section>
  );
}
