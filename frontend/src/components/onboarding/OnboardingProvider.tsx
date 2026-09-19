"use client";
/**
 * Drives the first-time onboarding tour. Mounted once in the dashboard layout
 * so it survives navigation between routes (the step index lives in the
 * persisted store, not in page state).
 *
 * Responsibilities:
 *  - Wait for store hydration so we never render onboarding on the server or
 *    flash it before the persisted status is known.
 *  - Navigate to a step's `route` when needed (only ever the dashboard, which
 *    is safe for a brand-new account).
 *  - Render the active step as a spotlight (anchored) or a centered card
 *    (modal / missing-target fallback).
 */
import { useEffect } from "react";
import { usePathname, useRouter } from "next/navigation";
import { useAppStore } from "@/store/appStore";
import { useOnboarding } from "@/hooks/useOnboarding";
import TourSpotlight from "./TourSpotlight";

export default function OnboardingProvider() {
  const hasHydrated = useAppStore((s) => s.hasHydrated);
  const router = useRouter();
  const pathname = usePathname();
  const {
    isActive,
    step,
    stepIndex,
    total,
    isFirst,
    isLast,
    next,
    back,
    skip,
    goTo,
  } = useOnboarding();

  // Navigate to the step's route if we're not already there. Keyed on the step
  // id so it fires once per step, not on every pathname change.
  useEffect(() => {
    if (!isActive || !step?.route) return;
    if (pathname !== step.route) {
      router.push(step.route);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isActive, step?.id]);

  if (!hasHydrated || !isActive || !step) return null;

  return (
    <TourSpotlight
      step={step}
      stepIndex={stepIndex}
      total={total}
      isFirst={isFirst}
      isLast={isLast}
      onNext={next}
      onBack={back}
      onSkip={skip}
      onGoTo={goTo}
    />
  );
}
