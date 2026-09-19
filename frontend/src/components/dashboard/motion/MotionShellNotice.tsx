"use client";
import type { ReactNode } from "react";
import type { LucideIcon } from "lucide-react";

interface MotionShellNoticeProps {
  icon: LucideIcon;
  title: string;
  description: ReactNode;
}

/**
 * Inline "this arrives in a later phase" card for a Motion surface that has
 * no functional build yet — Phase 0 renders shells, not motion DSP, feature
 * generation or training (see motion_phase0.md's scope boundary). Reused
 * across the Motion content components wherever a sub-section (signal
 * preview, live progress curves, a deployment build) isn't wired up yet, so
 * every "not built yet" moment in the motion workflow reads as one
 * consistent, on-brand card rather than five different ad-hoc ones.
 *
 * Visually distinct from `ImpulseNotReady` (a full-page blocked state) —
 * this sits inside a card as a section body.
 */
export default function MotionShellNotice({ icon: Icon, title, description }: MotionShellNoticeProps) {
  return (
    <div className="flex flex-col items-center justify-center py-14 px-6 text-center">
      <div className="w-12 h-12 rounded-full flex items-center justify-center mb-3
                      bg-violet-100 text-violet-600 dark:bg-violet-500/10 dark:text-violet-300">
        <Icon size={20} />
      </div>
      <p className="text-sm font-semibold text-[color:var(--app-text)] mb-1">
        {title}
      </p>
      <p className="text-xs text-[color:var(--app-text-muted)] max-w-sm leading-relaxed">
        {description}
      </p>
    </div>
  );
}
