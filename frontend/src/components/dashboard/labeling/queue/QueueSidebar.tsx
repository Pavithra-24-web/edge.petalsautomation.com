"use client";
import { Check, Circle, AlertTriangle } from "lucide-react";
import type { QueueItem, QueueSplit } from "@/hooks/useLabelingQueue";

// Labeling Queue plan, Phase 4 §3/§4.A. Current / Completed / Remaining
// lists, grouped/badged by split for orientation only — never the sort
// order (§2.3). Rows stay clickable so a completed sample can be reopened.

const SPLIT_DOT_COLOR: Record<QueueSplit, string> = {
  training: "#7c3aed",
  testing: "#2563eb",
  postprocessing: "#059669",
};

const MAX_RENDERED_ROWS = 200;

type QueueSidebarProps = {
  items: QueueItem[];
  currentIndex: number;
  onSelect: (index: number) => void;
};

function displayName(filename: string) {
  return filename.split("/").pop()?.split("\\").pop() || filename;
}

export default function QueueSidebar({ items, currentIndex, onSelect }: QueueSidebarProps) {
  const current = items[currentIndex] || null;
  const completed = items
    .map((item, index) => ({ item, index }))
    .filter(({ item }) => item.status === "completed");
  const remaining = items
    .map((item, index) => ({ item, index }))
    .filter(({ item, index }) => item.status === "remaining" && index !== currentIndex);

  return (
    <aside className="labeling-queue-sidebar">
      {current && (
        <div className="labeling-queue-sidebar-group">
          <h3 className="labeling-queue-sidebar-heading">Current</h3>
          <button type="button" className="labeling-queue-row labeling-queue-row--current" disabled>
            <span className="labeling-queue-row-dot" style={{ background: SPLIT_DOT_COLOR[current.sample_type] }} aria-hidden="true" />
            <span className="labeling-queue-row-name truncate" title={current.filename}>{displayName(current.filename)}</span>
          </button>
        </div>
      )}

      <div className="labeling-queue-sidebar-group">
        <h3 className="labeling-queue-sidebar-heading">Completed ({completed.length})</h3>
        {completed.length === 0 ? (
          <p className="labeling-queue-sidebar-empty">Nothing labeled yet this session.</p>
        ) : (
          <div className="labeling-queue-sidebar-list">
            {completed.slice(0, MAX_RENDERED_ROWS).map(({ item, index }) => (
              <button
                key={item.id}
                type="button"
                className="labeling-queue-row labeling-queue-row--completed"
                onClick={() => onSelect(index)}
                title={item.filename}
              >
                <Check size={12} className="labeling-queue-row-check" aria-hidden="true" />
                <span className="labeling-queue-row-name truncate">{displayName(item.filename)}</span>
              </button>
            ))}
            {completed.length > MAX_RENDERED_ROWS && (
              <p className="labeling-queue-sidebar-more">+{completed.length - MAX_RENDERED_ROWS} more</p>
            )}
          </div>
        )}
      </div>

      <div className="labeling-queue-sidebar-group">
        <h3 className="labeling-queue-sidebar-heading">Remaining ({remaining.length})</h3>
        {remaining.length === 0 ? (
          <p className="labeling-queue-sidebar-empty">Nothing else queued.</p>
        ) : (
          <div className="labeling-queue-sidebar-list">
            {remaining.slice(0, MAX_RENDERED_ROWS).map(({ item, index }) => (
              <button
                key={item.id}
                type="button"
                className="labeling-queue-row"
                onClick={() => onSelect(index)}
                title={item.filename}
              >
                <Circle size={8} className="labeling-queue-row-circle" style={{ color: SPLIT_DOT_COLOR[item.sample_type] }} aria-hidden="true" />
                <span className="labeling-queue-row-name truncate">{displayName(item.filename)}</span>
              </button>
            ))}
            {remaining.length > MAX_RENDERED_ROWS && (
              <p className="labeling-queue-sidebar-more">+{remaining.length - MAX_RENDERED_ROWS} more</p>
            )}
          </div>
        )}
      </div>

      {items.some(i => i.status === "missing") && (
        <div className="labeling-queue-sidebar-group">
          <h3 className="labeling-queue-sidebar-heading">Unavailable</h3>
          <div className="labeling-queue-sidebar-list">
            {items
              .map((item, index) => ({ item, index }))
              .filter(({ item }) => item.status === "missing")
              .map(({ item }) => (
                <span key={item.id} className="labeling-queue-row labeling-queue-row--missing">
                  <AlertTriangle size={12} aria-hidden="true" />
                  <span className="labeling-queue-row-name truncate">{displayName(item.filename)}</span>
                </span>
              ))}
          </div>
        </div>
      )}
    </aside>
  );
}
