"use client";

import { useCallback, useEffect, useRef, useState } from "react";

/* ---------------------------------------------------------------------------
   useWorkflowCarousel — index state + a real-time autoplay clock.

   The clock is a single requestAnimationFrame loop that reports elapsed
   fraction of the current slide (0 → 1) through `onProgress`. The caller writes
   that straight onto a CSS custom property, so the progress rail fills smoothly
   without a React render per frame.

   Autoplay only runs while every gate agrees: the section is on screen, the tab
   is visible, the pointer is not resting on the carousel, focus is not inside
   it, the reader has not asked for reduced motion, and the reader has not
   paused it. Elapsed time is held across pauses, so resuming picks up where it
   stopped rather than restarting the slide.
--------------------------------------------------------------------------- */

export type CarouselClock = {
  index: number;
  /** Jump to a slide. Wraps in both directions. */
  go: (next: number) => void;
  next: () => void;
  prev: () => void;
  /** True while the clock is actually advancing. */
  running: boolean;
};

export function useWorkflowCarousel({
  count,
  intervalMs = 6500,
  enabled,
  onProgress,
}: {
  count: number;
  intervalMs?: number;
  /** All external gates, already ANDed together by the caller. */
  enabled: boolean;
  onProgress: (fraction: number) => void;
}): CarouselClock {
  const [index, setIndex] = useState(0);
  const elapsed = useRef(0);
  const frame = useRef(0);
  const report = useRef(onProgress);
  report.current = onProgress;

  const running = enabled && count > 1;

  const go = useCallback(
    (next: number) => {
      if (count < 1) return;
      elapsed.current = 0;
      report.current(0);
      setIndex(((next % count) + count) % count);
    },
    [count]
  );

  // A shorter variant can leave the index past the end — snap back rather than
  // rendering an empty track.
  useEffect(() => {
    if (index > count - 1) go(0);
  }, [count, index, go]);

  useEffect(() => {
    if (!running) return;
    let last = performance.now();

    const tick = (now: number) => {
      elapsed.current += now - last;
      last = now;

      if (elapsed.current >= intervalMs) {
        elapsed.current = 0;
        report.current(0);
        setIndex((i) => (i + 1) % count);
      } else {
        report.current(elapsed.current / intervalMs);
      }
      frame.current = requestAnimationFrame(tick);
    };

    frame.current = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(frame.current);
    // `index` is deliberately absent: advancing a slide must not restart the
    // loop, only reset the elapsed counter (which the tick already did).
  }, [running, intervalMs, count]);

  return {
    index,
    go,
    next: useCallback(() => go(index + 1), [go, index]),
    prev: useCallback(() => go(index - 1), [go, index]),
    running,
  };
}

/* ---------------------------------------------------------------------------
   Gates. Each returns a boolean the carousel ANDs into `enabled`.
--------------------------------------------------------------------------- */

/** True once the element has entered the viewport, false when it leaves. */
export function useInView<T extends HTMLElement>(ref: React.RefObject<T>) {
  const [inView, setInView] = useState(false);

  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const io = new IntersectionObserver(
      ([entry]) => setInView(entry.isIntersecting),
      { threshold: 0.25 }
    );
    io.observe(el);
    return () => io.disconnect();
  }, [ref]);

  return inView;
}

/** True while the document is the foreground tab. */
export function usePageVisible() {
  const [visible, setVisible] = useState(true);

  useEffect(() => {
    const sync = () => setVisible(document.visibilityState === "visible");
    sync();
    document.addEventListener("visibilitychange", sync);
    return () => document.removeEventListener("visibilitychange", sync);
  }, []);

  return visible;
}

/** True when the reader has asked the system for reduced motion. */
export function useReducedMotion() {
  const [reduced, setReduced] = useState(false);

  useEffect(() => {
    const mq = window.matchMedia("(prefers-reduced-motion: reduce)");
    const sync = () => setReduced(mq.matches);
    sync();
    mq.addEventListener("change", sync);
    return () => mq.removeEventListener("change", sync);
  }, []);

  return reduced;
}
