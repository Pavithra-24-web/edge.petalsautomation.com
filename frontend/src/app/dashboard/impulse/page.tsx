"use client";
import { useAppStore } from "@/store/appStore";
import ObjectDetectionImpulse from "@/components/dashboard/impulse/ObjectDetectionImpulse";
import MotionImpulse from "@/components/dashboard/impulse/MotionImpulse";

/**
 * Route: /dashboard/impulse
 *
 * Delegates its body to one of two components by `project_type`. See
 * docs/Motion recognition/motion_phase0.md §4, §6.3.
 */
export default function ImpulsePage() {
  const { activeProject } = useAppStore();

  return activeProject?.project_type === "motion"
    ? <MotionImpulse />
    : <ObjectDetectionImpulse />;
}
