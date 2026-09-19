"use client";
import { Suspense, useEffect } from "react";
import { useRouter } from "next/navigation";
import { useAppStore } from "@/store/appStore";
import LabelingQueue from "@/components/dashboard/labeling/queue/LabelingQueue";

/**
 * Route: /dashboard/data/dataset/queue (Labeling Queue)
 *
 * A real route rather than a third `ObjectDetectionLabeling` viewMode (see
 * Labeling Queue plan §2.1) — owns the active-project guard and the
 * motion-project redirect; all queue behaviour lives in `LabelingQueue`.
 */
function LabelingQueueGuard() {
  const { activeProject } = useAppStore();
  const router = useRouter();

  const isMotion = activeProject?.project_type === "motion";
  useEffect(() => {
    if (isMotion) router.replace("/dashboard/data/dataset");
  }, [isMotion, router]);

  if (!activeProject) {
    return (
      <div className="flex items-center justify-center h-64">
        <p className="text-gray-500">Select a project from the sidebar to get started.</p>
      </div>
    );
  }

  if (isMotion) return null;

  return <LabelingQueue />;
}

export default function LabelingQueuePage() {
  return (
    <Suspense fallback={
      <div className="flex items-center justify-center h-64">
        <p className="text-gray-500">Loading…</p>
      </div>
    }>
      <LabelingQueueGuard />
    </Suspense>
  );
}
