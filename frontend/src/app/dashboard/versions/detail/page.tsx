"use client";

import Link from "next/link";
import { useSearchParams } from "next/navigation";
import { ChevronLeft, History } from "lucide-react";
import ProjectVersionDetailPanel from "@/components/dashboard/versions/ProjectVersionDetailPanel";

/**
 * Route: /dashboard/versions/detail?versionId=
 *
 * The version detail view (parityfix.md §7.2, §7.3) — reached from the
 * history table's row name. A query param, not a `[versionId]` path segment
 * — this app builds with `output: "export"` (next.config.js), where a
 * dynamic path segment needs `generateStaticParams()` known at build time
 * (see src/app/solutions/[slug]/page.tsx for that pattern), which doesn't
 * fit an open-ended set of version ids. Every other per-resource deep link
 * in this app already uses a query param for the same reason (e.g.
 * `/dashboard/data/dataset?id=`, `?impulseId=`).
 */
export default function ProjectVersionDetailPage() {
  const searchParams = useSearchParams();
  const versionId = searchParams.get("versionId") || "";

  return (
    <div className="pe-versions mx-auto max-w-6xl space-y-6">
      <Link href="/dashboard/versions" className="pe-pver-back-link">
        <ChevronLeft size={15} /> Back to version history
      </Link>

      <div className="pe-versions-header">
        <div className="pe-versions-header-icon" aria-hidden="true">
          <History size={22} strokeWidth={2.2} />
        </div>
        <div>
          <h2 className="pe-versions-title">Version detail</h2>
          <p className="pe-versions-sub">
            Everything captured in this snapshot, and how the project has drifted since.
          </p>
        </div>
      </div>

      {versionId ? (
        <ProjectVersionDetailPanel versionId={versionId} />
      ) : (
        <p className="pe-version-empty-note">No version selected.</p>
      )}
    </div>
  );
}
