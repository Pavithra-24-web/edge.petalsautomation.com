"use client";

import { useEffect, useRef, useState } from "react";

/* ---------------------------------------------------------------------------
   useHomeTheme — self-contained light/dark for the marketing page, independent
   of the app-wide ThemeController. Persists to localStorage("pe-home-theme").
   Defaults to "dark" so first paint matches the app shell (no flash).
--------------------------------------------------------------------------- */
export type HomeTheme = "dark" | "light";

export function useHomeTheme(): [HomeTheme, () => void] {
  const [theme, setTheme] = useState<HomeTheme>("dark");
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setMounted(true);
    try {
      const saved = window.localStorage.getItem("pe-home-theme") as HomeTheme | null;
      if (saved === "light" || saved === "dark") setTheme(saved);
      else if (window.matchMedia?.("(prefers-color-scheme: light)").matches) setTheme("light");
    } catch {
      /* localStorage unavailable — keep default */
    }
  }, []);

  const toggle = () => {
    setTheme((prev) => {
      const next: HomeTheme = prev === "dark" ? "light" : "dark";
      try {
        window.localStorage.setItem("pe-home-theme", next);
      } catch {
        /* ignore */
      }
      return next;
    });
  };

  // Before mount we always report "dark" to keep SSR + first client render in sync.
  return [mounted ? theme : "dark", toggle];
}

/* ---------------------------------------------------------------------------
   Reveal — fade-up on scroll via IntersectionObserver. Adds `.in` once visible.
--------------------------------------------------------------------------- */
export function Reveal({
  children,
  as: Tag = "div",
  className = "",
  delay = 0,
  style,
  ...rest
}: {
  children: React.ReactNode;
  as?: any;
  className?: string;
  delay?: number;
} & React.HTMLAttributes<HTMLElement>) {
  const ref = useRef<HTMLElement | null>(null);
  const [seen, setSeen] = useState(false);

  useEffect(() => {
    const el = ref.current;
    if (!el || seen) return;
    const io = new IntersectionObserver(
      (entries) => {
        entries.forEach((e) => {
          if (e.isIntersecting) {
            setSeen(true);
            io.disconnect();
          }
        });
      },
      { threshold: 0.12, rootMargin: "0px 0px -8% 0px" }
    );
    io.observe(el);
    return () => io.disconnect();
  }, [seen]);

  return (
    <Tag
      ref={ref}
      className={`pe-reveal ${seen ? "in" : ""} ${className}`}
      style={{ transitionDelay: seen ? `${delay}ms` : "0ms", ...style }}
      {...rest}
    >
      {children}
    </Tag>
  );
}

/* ---------------------------------------------------------------------------
   Counter — animates from 0 to `value` when scrolled into view.
--------------------------------------------------------------------------- */
export function Counter({
  value,
  suffix = "",
  prefix = "",
  decimals = 0,
  duration = 1600,
}: {
  value: number;
  suffix?: string;
  prefix?: string;
  decimals?: number;
  duration?: number;
}) {
  const ref = useRef<HTMLSpanElement | null>(null);
  const [display, setDisplay] = useState(0);
  const started = useRef(false);

  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const io = new IntersectionObserver((entries) => {
      entries.forEach((e) => {
        if (e.isIntersecting && !started.current) {
          started.current = true;
          const reduce = window.matchMedia?.("(prefers-reduced-motion: reduce)").matches;
          if (reduce) {
            setDisplay(value);
            return;
          }
          const start = performance.now();
          const tick = (now: number) => {
            const p = Math.min((now - start) / duration, 1);
            const eased = 1 - Math.pow(1 - p, 3);
            setDisplay(value * eased);
            if (p < 1) requestAnimationFrame(tick);
          };
          requestAnimationFrame(tick);
        }
      });
    }, { threshold: 0.4 });
    io.observe(el);
    return () => io.disconnect();
  }, [value, duration]);

  const formatted =
    decimals > 0
      ? display.toFixed(decimals)
      : Math.round(display).toLocaleString("en-US");

  return (
    <span ref={ref}>
      {prefix}
      {formatted}
      {suffix}
    </span>
  );
}
