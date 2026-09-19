"use client";
import { EmptyFolderIllustration } from "./LabelingShared";

// Shared right-hand detail-panel shell: the card container plus the
// "nothing selected" empty state. Moved out of ObjectDetectionLabeling
// verbatim — the actual preview body (image / video / Konva editor for OD;
// a waveform player for Motion) stays entirely with the caller as `children`,
// rendered only once something is selected.

export interface DetailPanelHostProps {
  show: boolean;
  hasSelection: boolean;
  emptyTitle?: string;
  emptySubtitle?: string;
  children?: React.ReactNode;
}

export function DetailPanelHost({
  show,
  hasSelection,
  emptyTitle = "No data selected",
  emptySubtitle = "Select a sample from the list to view its details and preview.",
  children,
}: DetailPanelHostProps) {
  if (!show) return null;
  return (
    <div className="data-acq-side-card flex flex-col h-full min-h-0 overflow-hidden">
      {hasSelection ? children : (
        <div className="data-acq-empty-inner">
          <EmptyFolderIllustration />
          <p className="ec-title">{emptyTitle}</p>
          <p className="ec-sub">{emptySubtitle}</p>
        </div>
      )}
    </div>
  );
}
