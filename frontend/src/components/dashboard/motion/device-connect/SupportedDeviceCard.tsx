"use client";
import { ChevronRight, Loader2, Usb } from "lucide-react";
import { SUPPORTED_BOARDS } from "./boards";
import type { DiscoveredDevice } from "./types";

const BRAND = "var(--app-brand, #6366f1)";

export default function SupportedDeviceCard({
  device,
  connecting,
  disabled,
  onConnect,
}: {
  device: DiscoveredDevice;
  connecting: boolean;
  disabled?: boolean;
  onConnect: () => void;
}) {
  const board = SUPPORTED_BOARDS[device.boardId];
  const Icon = board.icon;

  return (
    <button
      type="button"
      onClick={onConnect}
      disabled={disabled || connecting}
      className="group w-full flex items-center gap-3 rounded-xl px-4 py-3.5 text-left transition-all duration-150 disabled:cursor-not-allowed"
      style={{
        background: "var(--app-surface)",
        border: "1px solid var(--app-border)",
        boxShadow: "var(--app-shadow)",
        opacity: disabled && !connecting ? 0.55 : 1,
      }}
      onMouseEnter={(e) => { if (!disabled && !connecting) e.currentTarget.style.borderColor = BRAND; }}
      onMouseLeave={(e) => { e.currentTarget.style.borderColor = "var(--app-border)"; }}
    >
      <span
        className="flex-shrink-0 grid place-items-center w-10 h-10 rounded-lg"
        style={{ background: "var(--app-surface-2)", border: "1px solid var(--app-border)" }}
      >
        <Icon size={19} style={{ color: BRAND }} />
      </span>
      <div className="min-w-0 flex-1">
        <p className="text-sm font-semibold" style={{ color: "var(--app-text)" }}>{device.suggestedName}</p>
        <p className="text-[11px] mt-0.5 font-mono" style={{ color: "var(--app-text-soft)" }}>
          {device.port} · VID {device.vendorId} · PID {device.productId}
        </p>
      </div>
      {connecting ? (
        <span className="flex-shrink-0 inline-flex items-center gap-1.5 text-xs font-medium" style={{ color: BRAND }}>
          <Loader2 size={14} className="animate-spin" /> Connecting…
        </span>
      ) : (
        <span
          className="flex-shrink-0 inline-flex items-center gap-1 text-xs font-medium transition-colors"
          style={{ color: "var(--app-text-soft)" }}
        >
          <Usb size={13} className="opacity-0 group-hover:opacity-100 transition-opacity" />
          Connect
          <ChevronRight size={14} />
        </span>
      )}
    </button>
  );
}
