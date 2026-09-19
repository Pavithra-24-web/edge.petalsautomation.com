"use client";

import { useCallback, useRef } from "react";

/* ---------------------------------------------------------------------------
   InfiniteTicker — the marquee engine.

   One continuous belt, not a carousel: there is no page, no index and no snap
   point. The children are rendered three times inside a flex track, and CSS
   translates the track by exactly one copy width, forever. Because copy 2 is
   already in the viewport when copy 1 leaves it, the wrap is invisible.

   Why three copies and not two. The belt has two independent offsets — the CSS
   animation and the reader's drag — and they add up. With the drag normalised
   into one copy width, the pair can reach two copy widths of travel, so a third
   copy is what keeps the right edge covered at the extreme. Copies 2 and 3 are
   aria-hidden, so the row is announced once.

   Motion runs on `transform` only (no layout, no paint) and lives on two
   elements per row rather than on every card, so a page with five rows still
   composites on the GPU.

   Pausing is CSS: `:hover` and `:focus-within` on the viewport set
   animation-play-state. This component only owns the drag.
--------------------------------------------------------------------------- */
export default function InfiniteTicker({
  children,
  /** Seconds for one full copy of the belt to pass. Higher = slower. */
  duration,
  /** "left" is the default flow; "right" plays the same keyframes reversed. */
  direction = "left",
  label,
}: {
  children: React.ReactNode;
  duration: number;
  direction?: "left" | "right";
  label?: string;
}) {
  const viewRef = useRef<HTMLDivElement | null>(null);
  const shiftRef = useRef<HTMLDivElement | null>(null);
  const trackRef = useRef<HTMLDivElement | null>(null);

  /* Drag offset in px, always folded into (-copyWidth, 0]. Kept in a ref and
     written straight to a custom property — dragging must not re-render 100
     cards a frame. */
  const dragRef = useRef(0);
  const lastX = useRef(0);
  const dragging = useRef(false);

  const setDrag = (px: number) => {
    const track = trackRef.current;
    const shift = shiftRef.current;
    if (!track || !shift) return;
    const copy = track.scrollWidth / 3;
    // Fold rather than clamp: the belt is a loop, so an offset past one copy
    // width is the same offset one copy earlier.
    const folded = copy > 0 ? ((px % copy) - copy) % copy : px;
    dragRef.current = folded;
    shift.style.setProperty("--dx", `${folded}px`);
  };

  const onPointerDown = useCallback((e: React.PointerEvent<HTMLDivElement>) => {
    // Let the browser keep vertical scrolling and let links/buttons work.
    if (e.pointerType === "mouse" && e.button !== 0) return;
    dragging.current = true;
    lastX.current = e.clientX;
    viewRef.current?.classList.add("is-dragging");
    e.currentTarget.setPointerCapture(e.pointerId);
  }, []);

  const onPointerMove = useCallback((e: React.PointerEvent<HTMLDivElement>) => {
    if (!dragging.current) return;
    const dx = e.clientX - lastX.current;
    lastX.current = e.clientX;
    setDrag(dragRef.current + dx);
  }, []);

  const endDrag = useCallback((e: React.PointerEvent<HTMLDivElement>) => {
    if (!dragging.current) return;
    dragging.current = false;
    viewRef.current?.classList.remove("is-dragging");
    if (e.currentTarget.hasPointerCapture(e.pointerId)) {
      e.currentTarget.releasePointerCapture(e.pointerId);
    }
  }, []);

  return (
    <div
      ref={viewRef}
      className="pe-mq-view"
      role="group"
      aria-label={label}
      onPointerDown={onPointerDown}
      onPointerMove={onPointerMove}
      onPointerUp={endDrag}
      onPointerCancel={endDrag}
    >
      <div ref={shiftRef} className="pe-mq-shift">
        <div
          ref={trackRef}
          className="pe-mq-track"
          style={{
            ["--dur" as string]: `${duration}s`,
            ["--dir" as string]: direction === "right" ? "reverse" : "normal",
          }}
        >
          <div className="pe-mq-copy">{children}</div>
          <div className="pe-mq-copy pe-mq-copy-dup" aria-hidden="true">{children}</div>
          <div className="pe-mq-copy pe-mq-copy-dup" aria-hidden="true">{children}</div>
        </div>
      </div>
    </div>
  );
}
