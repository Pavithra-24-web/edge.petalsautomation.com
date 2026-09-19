"use client";

import type { ReactNode } from "react";
import { Lightbulb } from "lucide-react";

interface ImpulseNotReadyProps {
  /** Override the title. Defaults to a two-tone "Almost there!". */
  title?: ReactNode;
  description: ReactNode;
  /** Optional override for the floating circular icon. */
  icon?: ReactNode;
  /** Page-specific tip text rendered in the bottom pill. */
  tip?: ReactNode;
  /** Optional CTA(s) rendered inside the card below the description. */
  actions?: ReactNode;
}

const DEFAULT_TITLE = (
  <>
    Almost <span className="pe-warn-title-accent">there!</span>
  </>
);

function DefaultIcon() {
  return (
    <svg
      width="56"
      height="56"
      viewBox="0 0 24 24"
      fill="none"
      stroke="url(#pe-warn-icon-grad)"
      strokeWidth="2.4"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      <defs>
        <linearGradient id="pe-warn-icon-grad" x1="0" y1="0" x2="24" y2="24" gradientUnits="userSpaceOnUse">
          <stop offset="0" stopColor="#a855f7" />
          <stop offset="0.5" stopColor="#ec4899" />
          <stop offset="1" stopColor="#3b82f6" />
        </linearGradient>
      </defs>
      <path d="M22 12h-4l-3 9L9 3l-3 9H2" />
    </svg>
  );
}

/**
 * Shared "not ready" / blocked state used across the impulse flow.
 * Renders an ambient surface with a centered card, a floating circular
 * icon, and an optional page-specific tip pill underneath.
 */
export default function ImpulseNotReady({
  title = DEFAULT_TITLE,
  description,
  icon,
  tip,
  actions,
}: ImpulseNotReadyProps) {
  return (
    <div className="pe-warn-page">
      <div className="pe-warn-bg" aria-hidden="true">
        <span className="pe-warn-bg-dots pe-warn-bg-dots--tl" />
        <span className="pe-warn-bg-dots pe-warn-bg-dots--br" />
        <span className="pe-warn-bg-blob pe-warn-bg-blob--tl" />
        <span className="pe-warn-bg-blob pe-warn-bg-blob--br" />
        <span className="pe-warn-sparkle pe-warn-sparkle--a" />
        <span className="pe-warn-sparkle pe-warn-sparkle--b" />
        <span className="pe-warn-sparkle pe-warn-sparkle--c" />
        <span className="pe-warn-ring" />
      </div>

      <div className="pe-warn-shell">
        <div className="pe-warn-icon-wrap" aria-hidden="true">
          <span className="pe-warn-icon-halo" />
          <span className="pe-warn-icon">{icon ?? <DefaultIcon />}</span>
          <span className="pe-warn-icon-spark pe-warn-icon-spark--a" />
          <span className="pe-warn-icon-spark pe-warn-icon-spark--b" />
          <span className="pe-warn-icon-spark pe-warn-icon-spark--c" />
        </div>

        <div className="pe-warn-card" role="status" aria-live="polite">
          <h2 className="pe-warn-title">{title}</h2>
          <p className="pe-warn-desc">{description}</p>
          {actions && <div className="pe-warn-actions">{actions}</div>}
        </div>

        {tip && (
          <div className="pe-warn-tip" role="note">
            <span className="pe-warn-tip-icon" aria-hidden="true">
              <Lightbulb size={14} />
            </span>
            <span className="pe-warn-tip-text">
              <strong>Tip:</strong> {tip}
            </span>
          </div>
        )}
      </div>
    </div>
  );
}
