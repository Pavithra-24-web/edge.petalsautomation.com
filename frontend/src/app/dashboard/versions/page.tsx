"use client";

import { useEffect, useState } from "react";
import { History } from "lucide-react";
import { useAppStore } from "@/store/appStore";
import { impulsesApi, samplesApi } from "@/utils/api";
import ProjectVersionTable from "@/components/dashboard/versions/ProjectVersionTable";

/**
 * Route: /dashboard/versions
 *
 * Versioning (C2) — a standalone, project-level management surface, not a
 * step in the training pipeline. It reads no impulse state at all: switching
 * the active impulse in the sidebar never reloads or re-filters this page
 * (docs/Action/parityfix.md §1.0, §7.1).
 */
export default function ProjectVersioningPage() {
  const { activeProject } = useAppStore();
  const [captureSummary, setCaptureSummary] = useState({ sampleCount: 0, classCount: 0, impulseCount: 0 });

  useEffect(() => {
    if (!activeProject) return;
    let cancelled = false;
    async function loadSummary() {
      try {
        const [impulsesRes, samplesRes, labelsRes] = await Promise.all([
          impulsesApi.list(activeProject!.id),
          samplesApi.list(activeProject!.id, { limit: 1 }),
          samplesApi.labelsSummary(activeProject!.id),
        ]);
        if (cancelled) return;
        setCaptureSummary({
          impulseCount: Array.isArray(impulsesRes.data) ? impulsesRes.data.length : 0,
          sampleCount: samplesRes.data?.total ?? 0,
          classCount: labelsRes.data?.count ?? 0,
        });
      } catch {
        // Non-fatal — the dialog just shows zeros if this fails.
      }
    }
    void loadSummary();
    return () => {
      cancelled = true;
    };
  }, [activeProject?.id]);

  if (!activeProject) {
    return (
      <div className="flex items-center justify-center h-64">
        <p className="text-gray-500">Select a project from the sidebar to get started.</p>
      </div>
    );
  }

  return (
    <div className="pe-versions mx-auto max-w-6xl space-y-6">
      <div className="pe-versions-header">
        <div className="pe-versions-header-icon" aria-hidden="true">
          <History size={22} strokeWidth={2.2} />
        </div>
        <div>
          <h2 className="pe-versions-title">Versioning</h2>
          <p className="pe-versions-sub">
            Store a complete snapshot of <strong>{activeProject.name}</strong> — dataset, impulses,
            models and settings. Versions are only ever created when you store one.
          </p>
        </div>
      </div>

      <ProjectVersionTable projectId={activeProject.id} captureSummary={captureSummary} />
    </div>
  );
}
