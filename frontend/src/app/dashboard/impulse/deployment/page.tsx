"use client";
import { useAppStore } from "@/store/appStore";
import ObjectDetectionDeployment from "@/components/dashboard/deployment/ObjectDetectionDeployment";
import MotionDeployment from "@/components/dashboard/deployment/MotionDeployment";

/**
 * Route: /dashboard/impulse/deployment
 *
 * Delegates its body to one of two components by `project_type`. See
 * docs/Motion recognition/motion_phase0.md §4, §6.6.
 */
export default function DeploymentPage() {
  const { activeProject } = useAppStore();

  return activeProject?.project_type === "motion"
    ? <MotionDeployment />
    : <ObjectDetectionDeployment />;
}
