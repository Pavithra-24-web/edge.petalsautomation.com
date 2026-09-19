"use client";
import { Settings, Eye, Columns3, Database, List, Maximize2 } from "lucide-react";

// Shared layout-mode switcher: the toolbar's "View settings" gear (popover
// offering list↔grid swap + rows/items-per-page + grid-column count) and the
// separate "Show detailed view" Maximize2 toggle. Moved out of
// ObjectDetectionLabeling verbatim.
//
// `extraGridMenuItem` is a slot rendered inside the grid-mode popover, right
// after "Switch to list view" — exactly where OD's "Inline edit bounding box
// labels" checkbox sat originally. It stays a caller-supplied slot because
// that checkbox is Konva/bounding-box specific, not chrome; a non-OD caller
// (e.g. a future Motion explorer) can simply omit it.

export interface ViewSettingsMenuProps {
  open: boolean;
  setOpen: (open: boolean) => void;
  containerRef: React.RefObject<HTMLDivElement>;
  effectiveLayout: "list" | "grid";
  onSwitchLayout: (mode: "list" | "grid") => void;

  listRowsPerPage: number;
  onListRowsPerPageChange: (n: number) => void;
  listRowsPerPageOptions?: number[];

  itemsPerPage: number;
  onItemsPerPageChange: (n: number) => void;
  itemsPerPageOptions?: number[];

  gridColumns: number;
  onGridColumnsChange: (n: number) => void;
  gridColumnsOptions?: number[];

  extraGridMenuItem?: React.ReactNode;
}

export function ViewSettingsMenu({
  open, setOpen, containerRef, effectiveLayout, onSwitchLayout,
  listRowsPerPage, onListRowsPerPageChange, listRowsPerPageOptions = [6, 12, 24, 48, 100],
  itemsPerPage, onItemsPerPageChange, itemsPerPageOptions = [6, 12, 24, 48],
  gridColumns, onGridColumnsChange, gridColumnsOptions = [2, 3, 4, 5, 6],
  extraGridMenuItem,
}: ViewSettingsMenuProps) {
  return (
    <div className="relative" ref={containerRef}>
      <button
        className="tt-icon-btn"
        aria-label="View settings"
        title="View settings"
        onClick={() => setOpen(!open)}
      >
        <span className="relative inline-flex items-center justify-center" style={{ width: 16, height: 16 }}>
          <Settings size={16} />
          <Eye
            size={11}
            className="absolute"
            style={{
              right: -3,
              bottom: -3,
              background: "var(--app-surface)",
              borderRadius: 999,
              padding: 1,
            }}
          />
        </span>
      </button>
      {open && (
        <div
          className="absolute right-0 top-full mt-2 w-72 z-50 rounded-xl shadow-xl p-1.5"
          style={{
            background: "var(--app-surface)",
            border: "1px solid var(--app-border)",
            boxShadow: "var(--app-shadow)",
          }}
          role="menu"
        >
          {effectiveLayout === "list" ? (
            <>
              <button
                className="w-full flex items-center gap-2.5 px-3 py-2 text-sm rounded-lg hover:bg-[var(--app-surface-2)] transition-colors"
                style={{ color: "var(--app-text)" }}
                onClick={() => { onSwitchLayout("grid"); setOpen(false); }}
              >
                <Columns3 size={14} className="opacity-70" />
                <span>Switch to grid view</span>
              </button>
              <div
                className="w-full flex items-center justify-between gap-2 px-3 py-2 text-sm rounded-lg"
                style={{ color: "var(--app-text)" }}
              >
                <span className="inline-flex items-center gap-2.5">
                  <Database size={14} className="opacity-70" />
                  Rows per page
                </span>
                <select
                  value={listRowsPerPage}
                  onChange={e => onListRowsPerPageChange(Number(e.target.value))}
                  className="text-xs rounded px-1.5 py-1"
                  style={{
                    background: "var(--app-surface-2)",
                    border: "1px solid var(--app-border)",
                    color: "var(--app-text)",
                  }}
                >
                  {listRowsPerPageOptions.map(n => <option key={n} value={n}>{n}</option>)}
                </select>
              </div>
            </>
          ) : (
            <>
              <button
                className="w-full flex items-center gap-2.5 px-3 py-2 text-sm rounded-lg hover:bg-[var(--app-surface-2)] transition-colors"
                style={{ color: "var(--app-text)" }}
                onClick={() => { onSwitchLayout("list"); setOpen(false); }}
              >
                <List size={14} className="opacity-70" />
                <span>Switch to list view</span>
              </button>
              {extraGridMenuItem}
              <div
                className="w-full flex items-center justify-between gap-2 px-3 py-2 text-sm rounded-lg"
                style={{ color: "var(--app-text)" }}
              >
                <span className="inline-flex items-center gap-2.5">
                  <Database size={14} className="opacity-70" />
                  Items per page
                </span>
                <select
                  value={itemsPerPage}
                  onChange={e => onItemsPerPageChange(Number(e.target.value))}
                  className="text-xs rounded px-1.5 py-1"
                  style={{
                    background: "var(--app-surface-2)",
                    border: "1px solid var(--app-border)",
                    color: "var(--app-text)",
                  }}
                >
                  {itemsPerPageOptions.map(n => <option key={n} value={n}>{n}</option>)}
                </select>
              </div>
              <div
                className="w-full flex items-center justify-between gap-2 px-3 py-2 text-sm rounded-lg"
                style={{ color: "var(--app-text)" }}
              >
                <span className="inline-flex items-center gap-2.5">
                  <Columns3 size={14} className="opacity-70" />
                  Number of columns
                </span>
                <select
                  value={gridColumns}
                  onChange={e => onGridColumnsChange(Number(e.target.value))}
                  className="text-xs rounded px-1.5 py-1"
                  style={{
                    background: "var(--app-surface-2)",
                    border: "1px solid var(--app-border)",
                    color: "var(--app-text)",
                  }}
                >
                  {gridColumnsOptions.map(n => <option key={n} value={n}>{n}</option>)}
                </select>
              </div>
            </>
          )}
        </div>
      )}
    </div>
  );
}

export interface DetailedViewToggleProps {
  layoutMode: "list" | "grid" | "detailed";
  lastNonDetailedMode: "list" | "grid";
  onToggle: (mode: "list" | "grid" | "detailed") => void;
}

export function DetailedViewToggle({ layoutMode, lastNonDetailedMode, onToggle }: DetailedViewToggleProps) {
  return (
    <button
      className="tt-icon-btn"
      aria-label="Show detailed view"
      title="Show detailed view"
      onClick={() => onToggle(layoutMode === "detailed" ? lastNonDetailedMode : "detailed")}
      style={layoutMode === "detailed" ? { color: "var(--app-text)", background: "var(--app-surface-2)" } : undefined}
    >
      <Maximize2 size={14} />
    </button>
  );
}
