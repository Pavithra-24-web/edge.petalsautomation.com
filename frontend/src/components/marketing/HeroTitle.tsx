"use client";

import { useEffect, useRef, useState } from "react";

/* ---------------------------------------------------------------------------
   HeroTitle — the headline's one-time premium entrance.

   Two balanced lines. Each rises + fades in phrase-by-phrase on a ~200ms
   stagger (Apple-keynote / Stripe-launch cadence); the final gradient phrase
   adds a soft left-to-right mask wipe. Runs exactly once via
   IntersectionObserver, uses GPU transforms + opacity + a CSS mask only (no
   layout shift), and collapses to an instant, static state under
   prefers-reduced-motion. All visual styling lives in .pe-hero-title /
   .pe-ht-* in marketing.css so the headline's typography is untouched.
--------------------------------------------------------------------------- */
export default function HeroTitle() {
  const ref = useRef<HTMLHeadingElement | null>(null);
  const [seen, setSeen] = useState(false);

  useEffect(() => {
    const el = ref.current;
    if (!el || seen) return;

    // Reduced motion → skip the sequence, show the final state immediately.
    if (window.matchMedia?.("(prefers-reduced-motion: reduce)").matches) {
      setSeen(true);
      return;
    }

    const io = new IntersectionObserver(
      (entries) => {
        entries.forEach((e) => {
          if (e.isIntersecting) {
            setSeen(true);
            io.disconnect();
          }
        });
      },
      { threshold: 0.25, rootMargin: "0px 0px -8% 0px" }
    );
    io.observe(el);
    return () => io.disconnect();
  }, [seen]);

  return (
    <h1 ref={ref} className={`pe-h1 pe-hero-title ${seen ? "in" : ""}`}>
      <span className="pe-ht-line">
        <span className="pe-ht-word" style={{ "--d": "0ms" } as React.CSSProperties}>
          Build,
        </span>{" "}
        <span className="pe-ht-word" style={{ "--d": "200ms" } as React.CSSProperties}>
          Train &amp;
        </span>{" "}
        <span className="pe-ht-word" style={{ "--d": "400ms" } as React.CSSProperties}>
          Deploy
        </span>
      </span>
      <span className="pe-ht-line">
        <span className="pe-ht-word" style={{ "--d": "600ms" } as React.CSSProperties}>
          AI for
        </span>{" "}
        <span
          className="pe-ht-word pe-ht-wipe"
          style={{ "--d": "800ms" } as React.CSSProperties}
        >
          <span className="pe-grad-text">Every Edge Device</span>
        </span>
      </span>
    </h1>
  );
}
