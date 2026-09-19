"use client";
/**
 * Thin selector over the onboarding slice of the app store, scoped to the
 * currently signed-in user. Centralises the next/back/skip/finish transitions
 * and clamps the step index against ONBOARDING_STEPS so callers never go out
 * of range.
 */
import { useAppStore } from "@/store/appStore";
import { ONBOARDING_STEPS } from "@/components/onboarding/steps";

export function useOnboarding() {
  const user = useAppStore((s) => s.user);
  const onboarding = useAppStore((s) => s.onboarding);
  const setOnboardingStep = useAppStore((s) => s.setOnboardingStep);
  const finishOnboarding = useAppStore((s) => s.finishOnboarding);
  const skipOnboarding = useAppStore((s) => s.skipOnboarding);
  const restartOnboarding = useAppStore((s) => s.restartOnboarding);

  const userId = user?.id ?? null;
  const record = userId ? onboarding[userId] : undefined;
  const isActive = record?.status === "active";
  const rawStep = record?.step ?? 0;
  const stepIndex = Math.min(Math.max(rawStep, 0), ONBOARDING_STEPS.length - 1);
  const step = ONBOARDING_STEPS[stepIndex];
  const total = ONBOARDING_STEPS.length;
  const isLast = stepIndex >= total - 1;
  const isFirst = stepIndex <= 0;

  function next() {
    if (!userId) return;
    if (isLast) {
      finishOnboarding(userId);
    } else {
      setOnboardingStep(userId, stepIndex + 1);
    }
  }

  function back() {
    if (!userId || isFirst) return;
    setOnboardingStep(userId, stepIndex - 1);
  }

  function goTo(index: number) {
    if (!userId) return;
    setOnboardingStep(userId, Math.min(Math.max(index, 0), total - 1));
  }

  function skip() {
    if (!userId) return;
    skipOnboarding(userId);
  }

  function finish() {
    if (!userId) return;
    finishOnboarding(userId);
  }

  function restart() {
    if (!userId) return;
    restartOnboarding(userId);
  }

  return {
    userId,
    isActive,
    step,
    stepIndex,
    total,
    isFirst,
    isLast,
    next,
    back,
    goTo,
    skip,
    finish,
    restart,
  };
}
