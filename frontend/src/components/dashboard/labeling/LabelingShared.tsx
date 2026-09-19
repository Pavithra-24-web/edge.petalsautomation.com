"use client";
import { useState, useRef, useEffect } from "react";
import { createPortal } from "react-dom";

// Pieces shared by ObjectDetectionLabeling and MotionLabeling — a label
// distribution donut and the split-key type they both filter samples by.
// Moved out of the Labeling page verbatim; nothing here reads project_type
// or anything image-specific.

export type SplitKey = "training" | "testing" | "postprocessing";

// ─── Donut chart (SVG, no extra deps) ─────────────────────────────────────────
// 50-color premium palette — light & dark tones strictly interleaved
// (light → dark → light → dark …) so alphabetically-adjacent labels always
// contrast in brightness. Same 50 hues as before, just alternated.
export interface PieSegment { key: string; label: string; value: number; color: string }
export const PREMIUM_SOFT_PALETTE = [
  "#60a5fa", "#2c469c",  // blue 400        / blue 800     — blue family
  "#fbbf24", "#d97706",  // soft amber      / burnt orange
  "#f472b6", "#db2777",  // soft pink       / deep pink
  "#fb7185", "#dc2626",  // soft rose       / deep red
  "#34d399", "#065f46",  // emerald 400     / emerald 900  — emerald family
  "#22d3ee", "#155e75",  // cyan 400        / cyan 900     — cyan family
  "#2dd4bf", "#0d9488",  // teal            / dark teal
  "#34d399", "#059669",  // soft emerald    / forest green
  "#86efac", "#16a34a",  // mint green      / deep green
  "#a3e635", "#65a30d",  // soft lime       / olive
  "#facc15", "#ca8a04",  // soft yellow     / dark amber
  "#e879f9", "#c026d3",  // soft fuchsia    / deep fuchsia
  "#67e8f9", "#6d28d9",  // aqua            / plum
  "#fda4af", "#0369a1",  // blush           / petrol
  "#86bfff", "#b45309",  // periwinkle      / caramel
  "#d8b4fe", "#9f1239",  // lilac           / burgundy
  "#7dd3fc", "#86198f",  // sky blue        / deep magenta
  "#9333ea", "#3730a3",  // pale mint       / midnight indigo
];

export function PieChart({
  segments,
  size = 64,
  thickness = 8,
  onSliceClick,
  activeKey,
}: {
  segments: PieSegment[];
  size?: number;
  thickness?: number;
  onSliceClick?: (key: string) => void;
  activeKey?: string | null;
}) {
  const [hoverIdx, setHoverIdx] = useState<number | null>(null);
  const [mounted, setMounted] = useState(false);
  const wrapRef = useRef<HTMLDivElement | null>(null);
  useEffect(() => { setMounted(true); }, []);

  const total = segments.reduce((s, x) => s + x.value, 0);
  // Active slice uses a larger effective radius to visually "pop" it out.
  // We achieve this by bumping the stroke-width on the active arc so it
  // protrudes beyond the track on both the inner and outer edges.
  const ACTIVE_BUMP = 4;  // px added to strokeWidth on the active slice
  const r = size / 2 - thickness / 2;
  const cx = size / 2;
  const cy = size / 2;
  const circ = 2 * Math.PI * r;

  if (total === 0) {
    return (
      <div ref={wrapRef} className="ds-pie-wrap" style={{ width: size, height: size }}>
        <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`} aria-hidden="true">
          <circle cx={cx} cy={cy} r={r} fill="none"
            stroke="rgba(148,163,184,0.22)" strokeWidth={thickness} />
        </svg>
      </div>
    );
  }

  let acc = 0;
  const hoverSeg = hoverIdx != null ? segments[hoverIdx] : null;
  const pct = hoverSeg ? Math.round((hoverSeg.value / total) * 100) : 0;

  // Anchor the portal-rendered tooltip above the pie wrapper's bounding rect.
  const rect = mounted && hoverSeg ? wrapRef.current?.getBoundingClientRect() : null;
  const tooltipPos = rect
    ? { top: rect.top + window.scrollY - 8, left: rect.left + window.scrollX + rect.width / 2 }
    : null;

  // Pre-compute segment arc data so we can render the active ring overlay
  // as a second pass above all other arcs (avoiding z-order clipping).
  const arcData: { seg: PieSegment; len: number; dashOffset: number }[] = [];
  let accPre = 0;
  for (const seg of segments) {
    const len = (seg.value / total) * circ;
    arcData.push({ seg, len, dashOffset: -accPre });
    accPre += len;
  }

  const isClickable = !!onSliceClick;

  return (
    <div ref={wrapRef} className="ds-pie-wrap" style={{ width: size, height: size }}>
      <svg
        width={size} height={size} viewBox={`0 0 ${size} ${size}`}
        style={{ transform: "rotate(-90deg)", overflow: "visible" }}
        onMouseLeave={() => setHoverIdx(null)}
        role={isClickable ? "group" : undefined}
        aria-label={isClickable ? "Label distribution chart — click a slice to filter" : undefined}
      >
        {/* Track */}
        <circle cx={cx} cy={cy} r={r} fill="none"
          stroke="rgba(148,163,184,0.18)" strokeWidth={thickness} />

        {/* Slices */}
        {arcData.map(({ seg, len, dashOffset }, i) => {
          const isHover = hoverIdx === i;
          const isActive = activeKey != null && seg.key === activeKey;
          const sw = isActive ? thickness + ACTIVE_BUMP : (isHover ? thickness + 2 : thickness);
          return (
            <circle
              key={seg.key}
              cx={cx} cy={cy} r={r} fill="none"
              stroke={seg.color}
              strokeWidth={sw}
              strokeDasharray={`${len} ${circ - len}`}
              strokeDashoffset={dashOffset}
              onMouseEnter={() => setHoverIdx(i)}
              onClick={isClickable ? () => onSliceClick(seg.key) : undefined}
              onKeyDown={isClickable ? (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); onSliceClick(seg.key); } } : undefined}
              tabIndex={isClickable ? 0 : undefined}
              role={isClickable ? "button" : undefined}
              aria-label={isClickable ? `${seg.label}: ${seg.value} item${seg.value !== 1 ? "s" : ""}${isActive ? " (active filter — click to clear)" : " — click to filter"}` : undefined}
              aria-pressed={isClickable ? isActive : undefined}
              style={{ cursor: isClickable ? "pointer" : undefined, transition: "stroke-width 120ms", outline: "none" }}
            />
          );
        })}

        {/* Active-slice ring — rendered above all other arcs so it's never
            clipped by adjacent slices. A thin white/contrast overlay on the
            arc edges gives a crisp "selected" halo. */}
        {activeKey != null && arcData.map(({ seg, len, dashOffset }) => {
          if (seg.key !== activeKey) return null;
          return (
            <circle
              key={`ring-${seg.key}`}
              cx={cx} cy={cy} r={r} fill="none"
              stroke={seg.color}
              strokeWidth={thickness + ACTIVE_BUMP + 2}
              strokeOpacity={0.32}
              strokeDasharray={`${len} ${circ - len}`}
              strokeDashoffset={dashOffset}
              style={{ pointerEvents: "none" }}
            />
          );
        })}
      </svg>
      {hoverSeg && tooltipPos && createPortal(
        <div
          className="ds-pie-tooltip"
          role="tooltip"
          style={{ top: tooltipPos.top, left: tooltipPos.left }}
        >
          <div className="ds-pie-tooltip-label">{hoverSeg.label}</div>
          <div className="ds-pie-tooltip-row">
            <span className="ds-pie-tooltip-dot" style={{ background: hoverSeg.color }} />
            <span>{hoverSeg.value.toLocaleString()} item{hoverSeg.value !== 1 ? "s" : ""}</span>
            <span className="ds-pie-tooltip-pct">· {pct}%</span>
          </div>
        </div>,
        document.body
      )}
    </div>
  );
}
// ─── Folder empty-state illustration ──────────────────────────────────────────
export function EmptyFolderIllustration() {
  return (
    <svg width="156" height="120" viewBox="0 0 156 120" fill="none" xmlns="http://www.w3.org/2000/svg" aria-hidden="true" className="illu-empty-folder">
      {/* Animations: folder floats with a soft glow, sparkles twinkle on a
          staggered loop. Matches the style used on /dashboard/data and
          /dashboard/devices. Transform/opacity/filter only — theme-agnostic. */}
      <style>{`
        @keyframes pe-folder-float {
          0%   { transform: translateY(0)    rotate(0deg);   }
          25%  { transform: translateY(-4px) rotate(-1.5deg); }
          50%  { transform: translateY(-7px) rotate(0deg);   }
          75%  { transform: translateY(-4px) rotate(1.5deg);  }
          100% { transform: translateY(0)    rotate(0deg);   }
        }
        @keyframes pe-folder-glow {
          0%, 100% { filter: drop-shadow(0 3px 6px rgba(99, 102, 241, 0.18)); }
          50%      { filter: drop-shadow(0 10px 22px rgba(139, 92, 246, 0.50)); }
        }
        @keyframes pe-folder-sparkle {
          0%, 100% { opacity: 0.2;  transform: scale(0.6)  rotate(0deg);   }
          40%      { opacity: 1;    transform: scale(1.35) rotate(180deg); }
          50%      { opacity: 1;    transform: scale(1.4)  rotate(180deg); }
          60%      { opacity: 1;    transform: scale(1.35) rotate(180deg); }
        }
        .illu-empty-folder .illu-folder {
          animation: pe-folder-float 3.6s cubic-bezier(0.45, 0, 0.55, 1) infinite,
                     pe-folder-glow  3.6s ease-in-out infinite;
          transform-box: fill-box;
          transform-origin: center;
          will-change: transform, filter;
        }
        .illu-empty-folder .illu-sparkle {
          animation: pe-folder-sparkle 2.2s ease-in-out infinite;
          transform-box: fill-box;
          transform-origin: center;
          will-change: transform, opacity;
        }
        .illu-empty-folder .illu-sparkle:nth-of-type(1) { animation-delay: 0s;    }
        .illu-empty-folder .illu-sparkle:nth-of-type(2) { animation-delay: 0.45s; }
        .illu-empty-folder .illu-sparkle:nth-of-type(3) { animation-delay: 0.9s;  }
        .illu-empty-folder .illu-sparkle:nth-of-type(4) { animation-delay: 1.35s; }
        .illu-empty-folder .illu-sparkle:nth-of-type(5) { animation-delay: 1.8s;  }
        @media (prefers-reduced-motion: reduce) {
          .illu-empty-folder .illu-folder,
          .illu-empty-folder .illu-sparkle { animation: none; }
        }
      `}</style>
      <defs>
        <linearGradient id="folder-back" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor="#c7d2fe" />
          <stop offset="100%" stopColor="#a5b4fc" />
        </linearGradient>
        <linearGradient id="folder-front" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor="#ede9fe" />
          <stop offset="100%" stopColor="#c4b5fd" />
        </linearGradient>
        <linearGradient id="page-grad" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor="#ffffff" />
          <stop offset="100%" stopColor="#eef2ff" />
        </linearGradient>
      </defs>
      {/* sparkles */}
      <g opacity="0.85">
        <path className="illu-sparkle" d="M22 30l1.4 3 3 1.4-3 1.4L22 39l-1.4-3.2-3-1.4 3-1.4z" fill="#a5b4fc" />
        <path className="illu-sparkle" d="M138 24l1 2 2 1-2 1-1 2-1-2-2-1 2-1z" fill="#8b5cf6" />
        <circle className="illu-sparkle" cx="130" cy="60" r="2" fill="#c7d2fe" />
        <circle className="illu-sparkle" cx="20" cy="80" r="1.6" fill="#c4b5fd" />
        <path className="illu-sparkle" d="M142 88l1 2 2 1-2 1-1 2-1-2-2-1 2-1z" fill="#a78bfa" />
      </g>
      {/* folder + pages group — floats together */}
      <g className="illu-folder">
        {/* back folder */}
        <path d="M40 38h28l8 8h44a8 8 0 018 8v44a8 8 0 01-8 8H40a8 8 0 01-8-8V46a8 8 0 018-8z" fill="url(#folder-back)" />
        {/* white pages peeking out */}
        <rect x="52" y="50" width="58" height="36" rx="4" fill="url(#page-grad)" stroke="#c7d2fe" />
        <rect x="58" y="58" width="36" height="3" rx="1.5" fill="#c7d2fe" />
        <rect x="58" y="65" width="24" height="3" rx="1.5" fill="#ddd6fe" />
        {/* front folder */}
        <path d="M36 56h84a6 6 0 016 6v34a8 8 0 01-8 8H38a8 8 0 01-8-8V62a6 6 0 016-6z" fill="url(#folder-front)" />
        <path d="M36 56h84a6 6 0 016 6v3H30v-3a6 6 0 016-6z" fill="#a5b4fc" opacity="0.45" />
      </g>
    </svg>
  );
}
