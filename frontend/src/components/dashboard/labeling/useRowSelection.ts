import { useState } from "react";

// Shared multi-row selection model for the dataset explorer's list/grid table.
// Moved out of ObjectDetectionLabeling verbatim (toggleRow/toggleAll read the
// hook's own `selectedIds` directly rather than a functional updater — same
// non-functional-update semantics the original inline handlers had).

export function useRowSelection(allIds: string[]) {
  const [selectedIds, setSelectedIds] = useState<Set<string>>(new Set());

  function toggleRow(id: string) {
    const next = new Set(selectedIds);
    if (next.has(id)) next.delete(id);
    else next.add(id);
    setSelectedIds(next);
  }

  function toggleAll() {
    if (selectedIds.size === allIds.length) setSelectedIds(new Set());
    else setSelectedIds(new Set(allIds));
  }

  return { selectedIds, setSelectedIds, toggleRow, toggleAll };
}
