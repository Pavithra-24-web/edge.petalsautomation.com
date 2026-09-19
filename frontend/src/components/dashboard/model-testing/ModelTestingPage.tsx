"use client";
import { useAppStore } from "@/store/appStore";
import ObjectDetectionModelTesting from "./ObjectDetectionModelTesting";
import MotionModelTesting from "./MotionModelTesting";

/**
 * Model Testing — already the extracted page component for
 * /dashboard/impulse/model-testing (a 36-line route file delegates here).
 * Delegates its body to one of two components by `project_type`. See
 * docs/Motion recognition/motion_phase0.md §4, §6.5.
 */
export default function ModelTestingPage() {
  const { activeProject } = useAppStore();

  return activeProject?.project_type === "motion"
    ? <MotionModelTesting />
    : <ObjectDetectionModelTesting />;
}
