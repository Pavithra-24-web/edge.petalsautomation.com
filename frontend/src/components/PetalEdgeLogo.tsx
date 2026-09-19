"use client";

import { useId } from "react";

interface PetalEdgeLogoProps {
  variant?: "sidebar" | "login" | "icon";
  size?: number;
  className?: string;
  showWordmark?: boolean;
  label?: string;
}

function LogoMark({
  size = 40,
  id,
}: {
  size: number;
  id: string;
}) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 64 64"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
      aria-hidden="true"
      focusable="false"
      style={{ overflow: "visible", display: "block" }}
    >
      <defs>
        {/* Four petal gradients — magenta NW, cyan NE, violet SW, blue SE.
            Each gradient runs along the petal's own long axis (tip → centre
            after the rotation transform), so the brightest stop sits at the
            outer tip and the deepest stop sits near the centre, matching
            the reference. */}
        <linearGradient id={`${id}-petal-nw`} x1="32" y1="6" x2="32" y2="32" gradientUnits="userSpaceOnUse">
          <stop stopColor="#F0ABFC" />
          <stop offset="0.55" stopColor="#D946EF" />
          <stop offset="1" stopColor="#7C3AED" />
        </linearGradient>
        <linearGradient id={`${id}-petal-ne`} x1="32" y1="6" x2="32" y2="32" gradientUnits="userSpaceOnUse">
          <stop stopColor="#93C5FD" />
          <stop offset="0.55" stopColor="#38BDF8" />
          <stop offset="1" stopColor="#2563EB" />
        </linearGradient>
        <linearGradient id={`${id}-petal-sw`} x1="32" y1="6" x2="32" y2="32" gradientUnits="userSpaceOnUse">
          <stop stopColor="#A78BFA" />
          <stop offset="0.55" stopColor="#7C3AED" />
          <stop offset="1" stopColor="#4338CA" />
        </linearGradient>
        <linearGradient id={`${id}-petal-se`} x1="32" y1="6" x2="32" y2="32" gradientUnits="userSpaceOnUse">
          <stop stopColor="#60A5FA" />
          <stop offset="0.55" stopColor="#3B82F6" />
          <stop offset="1" stopColor="#1D4ED8" />
        </linearGradient>

        <linearGradient id={`${id}-bolt`} x1="28" y1="16" x2="40" y2="46" gradientUnits="userSpaceOnUse">
          <stop stopColor="#FFFFFF" stopOpacity="1" />
          <stop offset="0.6" stopColor="#F5F3FF" />
          <stop offset="1" stopColor="#DDD6FE" />
        </linearGradient>

        <filter id={`${id}-glow`} x="-30%" y="-30%" width="160%" height="160%">
          <feGaussianBlur in="SourceGraphic" stdDeviation="1.6" result="blur" />
          <feColorMatrix
            in="blur"
            type="matrix"
            values="0.5 0 0.8 0 0.1  0 0 0.6 0 0.2  0 0 0.6 0 0.8  0 0 0 0.65 0"
            result="coloredBlur"
          />
          <feMerge>
            <feMergeNode in="coloredBlur" />
            <feMergeNode in="SourceGraphic" />
          </feMerge>
        </filter>

        <filter id={`${id}-bolt-glow`} x="-30%" y="-20%" width="160%" height="140%">
          <feGaussianBlur in="SourceGraphic" stdDeviation="1.0" result="blur" />
          <feMerge>
            <feMergeNode in="blur" />
            <feMergeNode in="SourceGraphic" />
          </feMerge>
        </filter>

        {/* One petal shape, reused for each of the four positions by
            rotating around (32, 32). Teardrop with tip at (32, 4) and base
            at the centre. */}
        <path
          id={`${id}-petal-shape`}
          d="M32 4 C 19 8, 13 21, 17 31 C 22 28, 28 28, 32 32 C 36 28, 42 28, 47 31 C 51 21, 45 8, 32 4 Z"
        />
      </defs>

      {/* Petals — NW, NE, SE, SW (rotated 45° increments around centre) */}
      <g filter={`url(#${id}-glow)`}>
        <use href={`#${id}-petal-shape`} fill={`url(#${id}-petal-nw)`} transform="rotate(-45 32 32)" />
        <use href={`#${id}-petal-shape`} fill={`url(#${id}-petal-ne)`} transform="rotate(45 32 32)" />
        <use href={`#${id}-petal-shape`} fill={`url(#${id}-petal-se)`} transform="rotate(135 32 32)" />
        <use href={`#${id}-petal-shape`} fill={`url(#${id}-petal-sw)`} transform="rotate(-135 32 32)" />
      </g>

      {/* Circuit traces — one short stem with two end dots per petal.
          Stems run along each petal's long axis so they read as
          electronic veins on top of the petals. */}
      <g stroke="white" strokeWidth="1.3" strokeOpacity="0.95" strokeLinecap="round" fill="white" fillOpacity="0.96">
        {/* NW petal — trace going to upper-left */}
        <line x1="32" y1="32" x2="17.5" y2="17.5" />
        <circle cx="17.5" cy="17.5" r="1.8" />
        <circle cx="24" cy="24" r="1.4" />

        {/* NE petal — upper-right */}
        <line x1="32" y1="32" x2="46.5" y2="17.5" />
        <circle cx="46.5" cy="17.5" r="1.8" />
        <circle cx="40" cy="24" r="1.4" />

        {/* SW petal — lower-left */}
        <line x1="32" y1="32" x2="17.5" y2="46.5" />
        <circle cx="17.5" cy="46.5" r="1.8" />
        <circle cx="24" cy="40" r="1.4" />

        {/* SE petal — lower-right */}
        <line x1="32" y1="32" x2="46.5" y2="46.5" />
        <circle cx="46.5" cy="46.5" r="1.8" />
        <circle cx="40" cy="40" r="1.4" />
      </g>

      {/* Centred lightning bolt — sized to sit inside the petal cross */}
      <path
        d="M34.4 13L24.6 30.8L31.5 30.0L27.8 51L41.2 32.8L34.6 33.6L37.4 22.4L33.6 22.9L34.4 13Z"
        fill={`url(#${id}-bolt)`}
        filter={`url(#${id}-bolt-glow)`}
      />
    </svg>
  );
}

/**
 * Compact "PE" monogram in a rounded dark tile — the premium app-icon
 * style mark used in the top-bar row. Pairs with <Wordmark /> for the
 * sidebar variant. Independent from <LogoMark /> so the login screen's
 * detailed petal flower remains untouched.
 */
function PEMark({ size = 40, id = "pe" }: { size?: number; id?: string }) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 100 100"
      xmlns="http://www.w3.org/2000/svg"
      fill="none"
    >
      <defs>
        <linearGradient
          id={`${id}-grad`}
          x1="20"
          y1="20"
          x2="80"
          y2="80"
          gradientUnits="userSpaceOnUse"
        >
          <stop offset="0%" stopColor="#3FA9FF" />
          <stop offset="60%" stopColor="#386BFF" />
          <stop offset="100%" stopColor="#582CFF" />
        </linearGradient>
      </defs>

      <g
        stroke={`url(#${id}-grad)`}
        strokeWidth="10"
        strokeLinecap="square"
        strokeLinejoin="miter"
      >
        {/* 1. The 'P' Shape */}
        <path d="M30 25 H55 C70 25, 78 33, 70 47 C65 55, 55 55, 30 55 V80" />

        {/* 2. The Interlocking 'C' Shape */}
        {/* Top chunk: completely above the middle crossover bar */}
        <path d="M48 37 H76 V45" />

        {/* Bottom chunk: completely below the middle crossover bar */}
        <path d="M76 63 V71 H44 V58" />
      </g>
    </svg>
  );
}
function Wordmark({
  compact = false,
}: {
  compact?: boolean;
}) {
  return (
    <span
      className="pe-wordmark"
      style={{
        display: "inline-flex",
        alignItems: "baseline",
        gap: 0,
        fontWeight: 800,
        letterSpacing: compact ? "-0.04em" : "-0.045em",
        // line-height 1 clipped the descender of the "g" in "Edge" because
        // background-clip:text crops to the line box. 1.18 leaves enough
        // room without changing visual baseline.
        lineHeight: 1.18,
        // Tiny breathing room under the wordmark so the gradient text
        // tail of "g" is never clipped by an outer overflow:hidden parent.
        paddingBottom: 1,
        userSelect: "none",
        whiteSpace: "nowrap",
      }}
    >
      <span
        // Theme-aware color comes from .pe-wordmark-main in globals.css:
        //   default (dark mode): #e5eefb (near-white, readable on dark)
        //   .theme-light:        #0f172a (dark navy, readable on light)
        // No inline `color` override — that's what made "Petal" invisible
        // in dark mode previously.
        className="pe-wordmark-main"
      >
        Petal
      </span>
      <span
        className="pe-wordmark-accent"
        style={{
          // Keep the inline gradient as a fallback for places that load
          // the component before the stylesheet; theme-specific overrides
          // in globals.css take precedence when classes apply.
          background: "linear-gradient(90deg, #7c3aed 0%, #4f46e5 45%, #2563eb 100%)",
          WebkitBackgroundClip: "text",
          backgroundClip: "text",
          color: "transparent",
        }}
      >
        Edge
      </span>
    </span>
  );
}

export default function PetalEdgeLogo({
  variant = "sidebar",
  size,
  className = "",
  showWordmark = true,
  label = "PetalEdge",
}: PetalEdgeLogoProps) {
  const uid = useId().replace(/:/g, "");
  const markSize =
    size ?? (variant === "login" ? 168 : variant === "icon" ? 42 : 34);

  if (variant === "icon") {
    return (
      <span
        className={className}
        aria-label={label}
        role="img"
        style={{ display: "inline-flex", alignItems: "center" }}
      >
        <LogoMark size={markSize} id={`pe-${uid}`} />
      </span>
    );
  }

  if (variant === "login") {
    return (
      <div
        className={className}
        aria-label={label}
        role="img"
        style={{
          display: "inline-flex",
          flexDirection: "column",
          alignItems: "center",
          gap: 14,
        }}
      >
        <div style={{ position: "relative" }}>
          <div
            aria-hidden="true"
            style={{
              position: "absolute",
              inset: -10,
              borderRadius: 999,
              background:
                "radial-gradient(circle, rgba(124,58,237,0.25) 0%, rgba(37,99,235,0.12) 42%, rgba(255,255,255,0) 72%)",
              filter: "blur(10px)",
            }}
          />
          <LogoMark size={markSize} id={`pe-${uid}`} />
        </div>

        {showWordmark ? <Wordmark compact={false} /> : null}
      </div>
    );
  }

  // Sidebar (top-bar row) variant — premium "PE" tile + wordmark. The
  // login screen keeps the detailed petal mark via the "login" branch above.
  return (
    <div
      className={className}
      aria-label={label}
      role="img"
      style={{
        display: "inline-flex",
        alignItems: "center",
        gap: 10,
        whiteSpace: "nowrap",
      }}
    >
      <PEMark size={markSize} id={`pe-${uid}`} />
      {showWordmark ? <Wordmark compact /> : null}
    </div>
  );
}
