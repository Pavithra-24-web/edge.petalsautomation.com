"use client";
import { Suspense, useEffect } from "react";
import { useRouter } from "next/navigation";
import { useAppStore } from "@/store/appStore";
import SyntheticData from "@/components/dashboard/synthetic/SyntheticData";

/**
 * Route: /dashboard/data/dataset/synthetic (Synthetic Data)
 *
 * Mirrors `dataset/queue/page.tsx`'s guard: active-project placeholder and
 * a motion-project redirect (plan §5.2). All behaviour lives in
 * `SyntheticData`.
 */
function SyntheticDataGuard() {
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

  return <SyntheticData />;
}

export default function SyntheticDataPage() {
  return (
    <Suspense fallback={
      <div className="flex items-center justify-center h-64">
        <p className="text-gray-500">Loading…</p>
      </div>
    }>
      <SyntheticDataGuard />
    </Suspense>
  );
}
