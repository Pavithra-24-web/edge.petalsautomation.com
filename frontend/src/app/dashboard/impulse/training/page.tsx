"use client";
import { useAppStore } from "@/store/appStore";
import ObjectDetectionTraining from "@/components/dashboard/training/ObjectDetectionTraining";
import MotionTraining from "@/components/dashboard/training/MotionTraining";

/**
 * Route: /dashboard/impulse/training
 *
 * Delegates its body to one of two components by `project_type`. See
 * docs/Motion recognition/motion_phase0.md §4, §6.4.
 */
export default function TrainModelPage() {
  const { activeProject } = useAppStore();

  return activeProject?.project_type === "motion"
    ? <MotionTraining />
    : <ObjectDetectionTraining />;
}
