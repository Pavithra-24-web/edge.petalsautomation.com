"use client";
import { Trash2 } from "lucide-react";

// Shared bulk-action-bar shell: selected count + the always-present Delete
// button + a container for whatever else the caller wants to offer (label
// edit, split move, enable/disable, background marker, metadata — all of
// which stay OD-specific and are passed in as children). Moved out of
// ObjectDetectionLabeling verbatim.

export interface BulkActionBarProps {
  count: number;
  onDelete: () => void;
  children?: React.ReactNode;
}

export function BulkActionBar({ count, onDelete, children }: BulkActionBarProps) {
  if (count === 0) return null;
  return (
    <div className="bg-brand-900/30 border border-brand-500/50 rounded-md p-2 flex items-center justify-between animate-fade-in">
      <span className="text-sm font-medium text-brand-300 ml-2">
        {count} selected
      </span>
      <div className="flex items-center gap-2">
        <button className="btn-ghost text-xs py-1 hover:text-red-400" onClick={onDelete}>
          <Trash2 size={12} className="inline mr-1 mb-0.5" /> Delete
        </button>
        {children}
      </div>
    </div>
  );
}
