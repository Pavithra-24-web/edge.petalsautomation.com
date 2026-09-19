"use client";
import { useAppStore } from "@/store/appStore";
import ObjectDetectionDataset from "@/components/dashboard/data/ObjectDetectionDataset";
import MotionDataset from "@/components/dashboard/data/MotionDataset";

/**
 * Route: /dashboard/data (Data acquisition)
 *
 * This page owns routing, the active-project guard and layout; the content
 * body is delegated to one of two components by `project_type`. See
 * docs/Motion recognition/motion_phase0.md §4.
 */
export default function DataPage() {
  const { activeProject } = useAppStore();

  if (!activeProject) {
    return (
      <div className="flex items-center justify-center h-64">
        <p className="text-gray-500">Select a project from the sidebar to get started.</p>
      </div>
    );
  }

  return activeProject.project_type === "motion"
    ? <MotionDataset />
    : <ObjectDetectionDataset />;
}
