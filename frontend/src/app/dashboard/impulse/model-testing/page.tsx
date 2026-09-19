"use client";

import { Suspense } from "react";
import ModelTestingPage from "@/components/dashboard/model-testing/ModelTestingPage";

/**
 * Route: /dashboard/impulse/model-testing
 *
 * Suspense is required because ModelTestingPage uses useSearchParams(),
 * which needs a Suspense boundary in Next.js 14 App Router.
 *
 * The fallback renders a premium shell so there's no layout flash between
 * the fallback and the real page.
 */
function PageShellFallback() {
  return (
    <div className="pe-mt-page mx-auto max-w-[96rem] space-y-6">
      <div className="pe-mt-hero">
        <div className="h-4 w-48 bg-white/25 rounded animate-pulse" />
        <div className="mt-4 h-3 w-3/4 bg-white/15 rounded animate-pulse" />
      </div>
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-5 items-start">
        <div className="lg:col-span-7 pe-mt-card h-96 animate-pulse" />
        <div className="lg:col-span-5 pe-mt-card h-96 animate-pulse" />
      </div>
    </div>
  );
}

export default function ModelTestingRoute() {
  return (
    <Suspense fallback={<PageShellFallback />}>
      <ModelTestingPage />
    </Suspense>
  );
}
