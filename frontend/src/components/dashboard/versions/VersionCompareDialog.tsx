"use client";

import { useEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { GitCompare, X, Loader2, Check } from "lucide-react";
import { projectVersionsApi } from "@/utils/api";
import type { VersionCompareResponse } from "./projectVersionTypes";

interface VersionCompareDialogProps {
  versionIds: [string, string];
  onClose: () => void;
}

type Stage = "loading" | "ready" | "error";

function formatDateTime(iso: string) {
  const d = new Date(iso);
  return d.toLocaleDateString(undefined, { month: "short", day: "numeric", year: "numeric" }) +
    ", " + d.toLocaleTimeString(undefined, { hour: "numeric", minute: "2-digit" });
}

/**
 * Compact per-category summary of two project versions (parityfix.md §4.5,
 * §7.3) — four lines (Project settings, Dataset, Impulses, Deployment), not
 * a full impulse-by-impulse listing. Calls GET /project-versions/{id}/compare/{id}.
 */
export default function VersionCompareDialog({ versionIds, onClose }: VersionCompareDialogProps) {
  const [stage, setStage] = useState<Stage>("loading");
  const [data, setData] = useState<VersionCompareResponse | null>(null);
  const modalRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    let cancelled = false;
    setStage("loading");
    projectVersionsApi
      .compare(versionIds[0], versionIds[1])
      .then(({ data }) => {
        if (cancelled) return;
        setData(data);
        setStage("ready");
      })
      .catch(() => {
        if (!cancelled) setStage("error");
      });
    return () => {
      cancelled = true;
    };
  }, [versionIds]);

  useEffect(() => {
    modalRef.current?.focus();
    function handleKey(e: KeyboardEvent) {
      if (e.key === "Escape") onClose();
    }
    document.addEventListener("keydown", handleKey);
    return () => document.removeEventListener("keydown", handleKey);
  }, [onClose]);

  if (typeof document === "undefined") return null;

  return createPortal(
    <div className="pe-version-overlay" onClick={onClose} role="dialog" aria-modal="true" aria-labelledby="pe-pver-compare-title">
      <div
        ref={modalRef}
        className="pe-version-modal"
        tabIndex={-1}
        onClick={(e) => e.stopPropagation()}
      >
        <div className="pe-version-modal-head">
          <h3 id="pe-pver-compare-title" className="pe-version-modal-title">
            <span className="pe-version-modal-title-icon" aria-hidden="true">
              <GitCompare size={16} strokeWidth={2.1} />
            </span>
            Compare versions
          </h3>
          <button type="button" onClick={onClose} className="pe-version-modal-close" aria-label="Close">
            <X size={16} />
          </button>
        </div>

        <div className="pe-version-modal-body">
          {stage === "loading" && (
            <div className="pe-version-loading">
              <Loader2 size={16} className="pe-version-spin" /> Loading comparison…
            </div>
          )}

          {stage === "error" && (
            <div className="pe-version-loading pe-version-loading--error">
              Couldn&apos;t load this comparison.
            </div>
          )}

          {stage === "ready" && data && (
            <>
              <div className="pe-pver-compare-heads">
                <div className="pe-pver-compare-head">
                  <span className="pe-pver-compare-head-num">v{data.version_a.version_number}</span>
                  <span className="pe-pver-compare-head-name">{data.version_a.name || "Untitled snapshot"}</span>
                  <span className="pe-pver-compare-head-date">{formatDateTime(data.version_a.created_at)}</span>
                </div>
                <span className="pe-pver-compare-vs" aria-hidden="true">vs</span>
                <div className="pe-pver-compare-head">
                  <span className="pe-pver-compare-head-num">v{data.version_b.version_number}</span>
                  <span className="pe-pver-compare-head-name">{data.version_b.name || "Untitled snapshot"}</span>
                  <span className="pe-pver-compare-head-date">{formatDateTime(data.version_b.created_at)}</span>
                </div>
              </div>

              <div className="pe-pver-compare-summary">
                <div className="pe-pver-compare-row">
                  <span className="pe-pver-compare-row-label">Project Settings</span>
                  <span className={`pe-version-diff-chip ${data.project_config_same ? "" : "is-changed"}`}>
                    {data.project_config_same ? <><Check size={12} /> Same</> : "Changed"}
                  </span>
                </div>

                <div className="pe-pver-compare-row">
                  <span className="pe-pver-compare-row-label">Dataset</span>
                  <span className="pe-pver-compare-row-detail">
                    Samples: {data.sample_count_a.toLocaleString()} → {data.sample_count_b.toLocaleString()}
                    {" · "}
                    Classes: {data.class_count_a} → {data.class_count_b}
                  </span>
                </div>

                <div className="pe-pver-compare-row">
                  <span className="pe-pver-compare-row-label">Impulses</span>
                  <span className="pe-pver-compare-row-detail">
                    Total: {data.impulse_total_a} → {data.impulse_total_b}
                    {" · "}
                    Added: {data.impulses_added}
                    {" · "}
                    Removed: {data.impulses_removed}
                    {" · "}
                    Modified: {data.impulses_modified}
                  </span>
                </div>

                <div className="pe-pver-compare-row">
                  <span className="pe-pver-compare-row-label">Deployment</span>
                  <span className={`pe-version-diff-chip ${data.deployment_same ? "" : "is-changed"}`}>
                    {data.deployment_same ? <><Check size={12} /> Same</> : "Changed"}
                  </span>
                </div>
              </div>
            </>
          )}
        </div>

        <div className="pe-version-modal-foot">
          <button type="button" onClick={onClose} className="pe-version-btn-ghost">
            Close
          </button>
        </div>
      </div>
    </div>,
    document.body,
  );
}
