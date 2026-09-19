"use client";
import { SearchX, RefreshCw, Usb } from "lucide-react";
import { SUPPORTED_BOARDS, SUPPORTED_BOARD_ORDER } from "./boards";

export default function EmptyState({
  onRescan,
  rescanning,
  onRequestDevice,
  requesting,
}: {
  onRescan: () => void;
  rescanning: boolean;
  /** Open the browser's native device picker — the only way to authorize a
   *  board never granted access before. Already-authorized boards surface
   *  via Rescan, so this is a distinct action, not a duplicate of it. */
  onRequestDevice?: () => void;
  requesting?: boolean;
}) {
  return (
    <div
      className="flex flex-col items-center text-center gap-3 rounded-xl px-6 py-10"
      style={{ background: "var(--app-bg)", border: "1px dashed var(--app-border-strong)" }}
    >
      <span
        className="grid place-items-center w-12 h-12 rounded-full"
        style={{ background: "var(--app-surface-2)", border: "1px solid var(--app-border)" }}
      >
        <SearchX size={20} style={{ color: "var(--app-text-soft)" }} />
      </span>
      <div>
        <p className="text-sm font-semibold" style={{ color: "var(--app-text)" }}>
          No compatible devices found.
        </p>
        <p className="text-xs mt-1 max-w-xs" style={{ color: "var(--app-text-muted)" }}>
          Plug a supported board into this computer over USB, then rescan.
        </p>
      </div>

      <div className="flex items-center gap-2 mt-1">
        {SUPPORTED_BOARD_ORDER.map((id) => {
          const board = SUPPORTED_BOARDS[id];
          const Icon = board.icon;
          return (
            <span
              key={id}
              className="inline-flex items-center gap-1.5 text-[11px] font-medium px-2.5 py-1 rounded-full"
              style={{ background: "var(--app-surface-2)", border: "1px solid var(--app-border)", color: "var(--app-text-soft)" }}
            >
              <Icon size={12} />
              {board.label}
            </span>
          );
        })}
      </div>

      <div className="flex items-center gap-2 mt-2">
        <button
          type="button"
          onClick={onRescan}
          disabled={rescanning}
          className="btn-secondary"
          style={{ fontSize: "12px" }}
        >
          <RefreshCw size={13} className={rescanning ? "animate-spin" : ""} />
          {rescanning ? "Scanning…" : "Rescan"}
        </button>
        {onRequestDevice && (
          <button
            type="button"
            onClick={onRequestDevice}
            disabled={requesting}
            className="btn-primary"
            style={{ fontSize: "12px" }}
          >
            <Usb size={13} />
            {requesting ? "Waiting for selection…" : "Select a device…"}
          </button>
        )}
      </div>
    </div>
  );
}
