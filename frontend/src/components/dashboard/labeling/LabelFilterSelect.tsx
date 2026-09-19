"use client";

import { useEffect, useRef, useState } from "react";
import { ChevronDown } from "lucide-react";

// Shared "filter by label" dropdown + orphaned-label prune button. Moved out
// of ObjectDetectionLabeling verbatim. The prune confirm/pruneOrphans/reload
// side effects stay with the caller (project-scoped); this piece only decides
// what's rendered and hands back which labels are unused.
//
// Rendered as a custom button+menu (not a native <select>) so the open menu
// matches the app's dark theme — a native select's popup is OS-rendered and
// can't be themed or positioned via CSS.

export interface LabelOption {
  id: string;
  name: string;
}

export interface LabelFilterSelectProps {
  labels: LabelOption[];
  usedLabelKeys: Set<string>;
  labelCounts: Record<string, number>;
  filterLabel: string;
  onChange: (value: string) => void;
  onPruneOrphans: (orphans: LabelOption[]) => void;
}

export function LabelFilterSelect({
  labels, usedLabelKeys, labelCounts, filterLabel, onChange, onPruneOrphans,
}: LabelFilterSelectProps) {
  const [open, setOpen] = useState(false);
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!open) return;
    function onPointerDown(e: MouseEvent) {
      if (containerRef.current && !containerRef.current.contains(e.target as Node)) setOpen(false);
    }
    function onKeyDown(e: KeyboardEvent) {
      if (e.key === "Escape") setOpen(false);
    }
    document.addEventListener("mousedown", onPointerDown);
    document.addEventListener("keydown", onKeyDown);
    return () => {
      document.removeEventListener("mousedown", onPointerDown);
      document.removeEventListener("keydown", onKeyDown);
    };
  }, [open]);

  const orphans = labels.filter(
    l =>
      !usedLabelKeys.has(`id:${l.id}`) &&
      !usedLabelKeys.has(`name:${String(l.name).toLowerCase()}`)
  );

  const usedLabels = labels.filter(
    l => usedLabelKeys.has(`id:${l.id}`) || usedLabelKeys.has(`name:${String(l.name).toLowerCase()}`)
  );

  const hasUnlabeled = (labelCounts["Unlabeled"] ?? 0) > 0;

  const selectedLabel =
    filterLabel === "__unlabeled__"
      ? "Unlabeled"
      : filterLabel
        ? usedLabels.find(l => l.id === filterLabel)?.name ?? "All labels"
        : "All labels";

  function select(value: string) {
    onChange(value);
    setOpen(false);
  }

  return (
    <>
      <div className={`ds-label-select ${open ? "is-open" : ""}`} ref={containerRef}>
        <button
          type="button"
          className="ds-label-select-btn"
          onClick={() => setOpen(o => !o)}
          aria-haspopup="listbox"
          aria-expanded={open}
          aria-label="Filter by label"
        >
          <span className="truncate">{selectedLabel}</span>
          <ChevronDown size={14} className="chevron" aria-hidden="true" />
        </button>
        {open && (
          <div className="ds-label-select-menu" role="listbox" aria-label="Filter by label">
            <button
              type="button"
              role="option"
              aria-selected={filterLabel === ""}
              className={`ds-label-select-item ${filterLabel === "" ? "is-selected" : ""}`}
              onClick={() => select("")}
            >
              All labels
            </button>
            {usedLabels.map(l => (
              <button
                key={l.id}
                type="button"
                role="option"
                aria-selected={filterLabel === l.id}
                className={`ds-label-select-item ${filterLabel === l.id ? "is-selected" : ""}`}
                onClick={() => select(l.id)}
              >
                {l.name}
              </button>
            ))}
            {hasUnlabeled && (
              <button
                type="button"
                role="option"
                aria-selected={filterLabel === "__unlabeled__"}
                className={`ds-label-select-item ${filterLabel === "__unlabeled__" ? "is-selected" : ""}`}
                onClick={() => select("__unlabeled__")}
              >
                Unlabeled
              </button>
            )}
          </div>
        )}
      </div>
      {orphans.length > 0 && (
        <button
          className="btn-ghost text-xs py-1"
          title={`Unused: ${orphans.map(o => o.name).join(", ")}`}
          onClick={() => onPruneOrphans(orphans)}
        >
          Delete {orphans.length} unused
        </button>
      )}
    </>
  );
}
