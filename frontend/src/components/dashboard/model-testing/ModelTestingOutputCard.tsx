"use client";

import { useState } from "react";
import { BarChart2, ChevronDown } from "lucide-react";
import AccuracySummary from "./AccuracySummary";
import MetricsTable from "./MetricsTable";
import type { MetricRow, ModelVersion } from "@/types/model-testing";

interface ModelTestingOutputCardProps {
  loading:       boolean;
  accuracy:      number | null;
  metrics:       MetricRow[];
  modelVersions: ModelVersion[];
  activeVersion: ModelVersion | null;
  onVersionChange?: (version: ModelVersion) => void;
}

function CardSkeleton() {
  return (
    <div className="px-6 py-6 space-y-4 animate-pulse">
      <div className="flex items-center gap-3">
        <div className="w-12 h-12 rounded-full bg-gray-200 dark:bg-slate-800" />
        <div className="space-y-1.5">
          <div className="h-2.5 w-16 bg-gray-200 dark:bg-slate-800 rounded" />
          <div className="h-5  w-20 bg-gray-200 dark:bg-slate-800 rounded" />
        </div>
      </div>
      <div className="border-t border-[color:var(--app-border)] pt-4 space-y-2.5">
        {[1, 2, 3].map((i) => (
          <div key={i} className="flex justify-between">
            <div className="h-2.5 w-40 bg-gray-200 dark:bg-slate-800 rounded" />
            <div className="h-2.5 w-8  bg-gray-200 dark:bg-slate-800 rounded" />
          </div>
        ))}
      </div>
    </div>
  );
}

function EmptyMetrics() {
  return (
    <div className="flex flex-col items-center justify-center py-10 px-4 text-center">
      <div className="w-10 h-10 rounded-full flex items-center justify-center mb-2
                      bg-violet-100 text-violet-600 dark:bg-violet-500/10 dark:text-violet-300">
        <BarChart2 size={18} />
      </div>
      <p className="text-xs font-semibold text-[color:var(--app-text)] mb-1">No results yet</p>
      <p className="text-xs text-[color:var(--app-text-muted)]">
        Click &ldquo;Classify all&rdquo; in the Test data card to run classification.
      </p>
    </div>
  );
}

export default function ModelTestingOutputCard({
  loading,
  accuracy,
  metrics,
  modelVersions,
  activeVersion,
  onVersionChange,
}: ModelTestingOutputCardProps) {
  const [collapsed,    setCollapsed]    = useState(false);
  const [versionOpen,  setVersionOpen]  = useState(false);
  const [localVersion, setLocalVersion] = useState<ModelVersion | null>(null);

  const displayVersion = localVersion ?? activeVersion;

  const handleVersionSelect = (v: ModelVersion) => {
    setLocalVersion(v);
    setVersionOpen(false);
    onVersionChange?.(v);
  };

  const versionLabel = displayVersion?.name ?? "Quantized (int8)";

  const versionOptions: ModelVersion[] =
    modelVersions.length > 0
      ? modelVersions
      : displayVersion
        ? [displayVersion]
        : [];

  const hasMetrics = metrics.length > 0;
  const hasAccuracy = accuracy !== null;

  return (
    <div className="pe-mt-card">
      <div className="flex items-center justify-between px-6 py-4 border-b"
           style={{ borderColor: "var(--app-border)" }}>
        <h2 className="pe-mt-card-title">Model testing output</h2>
        <button
          type="button"
          onClick={() => setCollapsed(!collapsed)}
          className="pe-mt-icon-btn"
          aria-label="toggle-output-card"
        >
          <ChevronDown
            size={16}
            className={`transition-transform duration-200 ${collapsed ? "-rotate-90" : ""}`}
          />
        </button>
      </div>

      {!collapsed && (
        loading ? (
          <CardSkeleton />
        ) : (
          <div className="px-6 py-6 space-y-6">
            <div>
              <div className="flex items-center justify-between mb-3">
                <h3 className="text-sm font-semibold text-[color:var(--app-text)]">Results</h3>

                <div className="relative">
                  <div className="flex items-center gap-1.5">
                    <span className="text-xs text-[color:var(--app-text-muted)]">Model version:</span>
                    <button
                      type="button"
                      onClick={() => setVersionOpen(!versionOpen)}
                      className="inline-flex items-center gap-1 px-2.5 py-1 rounded-lg text-xs font-medium transition-colors"
                      style={{
                        background: "rgba(139, 92, 246, 0.10)",
                        border: "1px solid rgba(139, 92, 246, 0.28)",
                        color: "#6d28d9",
                      }}
                    >
                      {versionLabel}
                      <ChevronDown size={11} />
                    </button>
                  </div>

                  {versionOpen && (
                    <>
                      <div className="fixed inset-0 z-40" onClick={() => setVersionOpen(false)} />
                      <div
                        className="absolute right-0 top-full mt-1 z-50 min-w-[200px] rounded-lg shadow-lg py-1 border"
                        style={{
                          background: "var(--app-surface)",
                          borderColor: "var(--app-border)",
                        }}
                      >
                        {versionOptions.length > 0 ? (
                          versionOptions.map((v) => (
                            <button
                              key={v.id}
                              type="button"
                              onClick={() => handleVersionSelect(v)}
                              className="w-full text-left px-4 py-2 text-sm transition-colors hover:bg-violet-50 dark:hover:bg-slate-800"
                              style={{
                                color: v.id === displayVersion?.id ? "#6d28d9" : "var(--app-text)",
                                fontWeight: v.id === displayVersion?.id ? 600 : 400,
                              }}
                            >
                              {v.name}
                            </button>
                          ))
                        ) : (
                          <p className="px-4 py-2 text-xs text-[color:var(--app-text-muted)]">
                            No versions available
                          </p>
                        )}
                      </div>
                    </>
                  )}
                </div>
              </div>

              {hasAccuracy ? (
                <AccuracySummary accuracy={accuracy!} />
              ) : (
                <EmptyMetrics />
              )}
            </div>

            <div className="border-t" style={{ borderColor: "var(--app-border)" }} />
            <div>
              <div className="mb-3">
                <h4 className="text-sm font-semibold text-[color:var(--app-text)]">
                  Metrics for Object detection
                </h4>
              </div>
              {hasMetrics ? (
                <MetricsTable metrics={metrics} title="" />
              ) : (
                <p className="text-xs text-[color:var(--app-text-muted)] italic">
                  Run &ldquo;Classify all&rdquo; to generate metrics.
                </p>
              )}
            </div>

            <div className="border-t" style={{ borderColor: "var(--app-border)" }} />
            <div>
              <div className="mb-2">
                <h4 className="text-sm font-semibold text-[color:var(--app-text)]">Feature explorer</h4>
              </div>
              <p className="text-sm text-[color:var(--app-text-muted)] italic">
                Label the data in your test set to render the feature explorer.
              </p>
            </div>
          </div>
        )
      )}
    </div>
  );
}
