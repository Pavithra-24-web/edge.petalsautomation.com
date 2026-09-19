"use client";

import { useEffect, useRef } from "react";
import { SolIcon } from "../icons";
import { useReducedMotion } from "./useWorkflowCarousel";
import type { MediaAsset, WorkflowStage } from "@/lib/solutions.config";

/* ---------------------------------------------------------------------------
   WorkflowMedia — the media column of a workflow slide.

   The frame is a product window: a title bar naming the screen this stage
   happens on, then the capture below it. Whether the capture is a screenshot or
   a clip is data, not markup — a stage carries an optional MediaAsset and this
   component renders the matching element:

     stage.media = { kind: "image", src: "/workflow/label.png", alt: "…" }
     stage.media = { kind: "video", src: "/workflow/train.mp4", poster: "…" }

   (`kind`, not `type` — MediaAsset is the shape already used by the hero and
   section slots in solutions.config.ts, and one project should have one shape.)

   With no asset, the same window renders a composed field: the stage
   accent as light, a faint measure grid, and the stage's own icon. It is the
   window a capture will sit in, at the exact size it will occupy, so dropping
   real files into the config changes nothing else.

   Captures vary in shape, the window does not: every asset is fitted inside a
   fixed-ratio frame and letterboxed against the frame ground, so the slide is
   the same height on stage 01 and on stage 10 and the rail below never moves.

   Video costs nothing until its slide is reached: the element preloads no
   bytes and paints its poster, and only the active slide plays. The rest are
   paused and rewound, so a ten-stage pipeline never decodes ten clips at once
   and an off-screen clip never fetches one.
--------------------------------------------------------------------------- */
export default function WorkflowMedia({
  stage,
  step,
  active,
}: {
  stage: WorkflowStage;
  /** 1-based position, shown in the window bar. */
  step: number;
  active: boolean;
}) {
  const media: MediaAsset | undefined = stage.media;
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const reduced = useReducedMotion();

  useEffect(() => {
    const el = videoRef.current;
    if (!el) return;
    // Reduced motion: the poster is the whole presentation. Never start the
    // clip, and stop it if the preference flips while it is on stage.
    if (active && !reduced) {
      // Autoplay can reject (power saving, no user gesture). Nothing to
      // recover — the poster stays up, which is the right fallback.
      void el.play().catch(() => {});
    } else {
      el.pause();
      el.currentTime = 0;
    }
  }, [active, reduced]);

  return (
    <figure className="pe-wf-media">
      <div className="pe-wf-media-bar">
        <span className="pe-wf-media-screen">
          <span className="pe-wf-media-mark" aria-hidden="true" />
          {stage.where}
        </span>
        <span className="pe-wf-media-step">{String(step).padStart(2, "0")}</span>
      </div>

      <div className="pe-wf-media-body">
        {media?.kind === "video" ? (
          <video
            ref={videoRef}
            src={media.src}
            poster={media.poster}
            muted
            loop
            playsInline
            /* Not "metadata": the clip is tens of megabytes and lives on the
               last slide. Nothing is fetched until the effect above plays it. */
            preload="none"
            disablePictureInPicture
            disableRemotePlayback
            aria-label={media.alt}
          />
        ) : media ? (
          <img
            src={media.src}
            alt={media.alt}
            /* Only the opening slide is on screen at load; every capture
               behind it waits until the reader walks to it. */
            loading={step === 1 ? "eager" : "lazy"}
            fetchPriority={step === 1 ? "high" : "low"}
            decoding="async"
          />
        ) : (
          <span className="pe-wf-media-field" aria-hidden="true">
            <span className="pe-wf-media-grid" />
            <span className="pe-wf-media-light" />
            <span className="pe-wf-media-glyph">
              <SolIcon name={stage.icon} />
            </span>
          </span>
        )}
      </div>

      {media?.caption && <figcaption>{media.caption}</figcaption>}
    </figure>
  );
}
