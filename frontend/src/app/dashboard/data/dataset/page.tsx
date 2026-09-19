"use client";
import { useAppStore } from "@/store/appStore";
import ObjectDetectionLabeling from "@/components/dashboard/labeling/ObjectDetectionLabeling";
import MotionLabeling from "@/components/dashboard/labeling/MotionLabeling";

/**
 * Route: /dashboard/data/dataset (Data labeling)
 *
 * Delegates its body to one of two components by `project_type`. See
 * docs/Motion recognition/motion_phase0.md §4, §6.2.
 */
export default function DatasetPage() {
  const { activeProject } = useAppStore();

  return activeProject?.project_type === "motion"
    ? <MotionLabeling />
    : <ObjectDetectionLabeling />;
}
