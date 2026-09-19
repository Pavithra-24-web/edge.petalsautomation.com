"use client";
import { CheckCircle2, Loader2, XCircle, Circle } from "lucide-react";

export type ConnectionStatus = "idle" | "scanning" | "connecting" | "connected" | "error";

const STATUS_META: Record<ConnectionStatus, { label: string; color: string; icon: typeof Circle }> = {
  idle: { label: "Not connected", color: "var(--app-text-soft)", icon: Circle },
  scanning: { label: "Scanning…", color: "#f59e0b", icon: Loader2 },
  connecting: { label: "Connecting…", color: "#f59e0b", icon: Loader2 },
  connected: { label: "Connected", color: "#22c55e", icon: CheckCircle2 },
  error: { label: "Connection failed", color: "#ef4444", icon: XCircle },
};

export default function ConnectionStatusBadge({
  status,
  label,
}: {
  status: ConnectionStatus;
  /** Override the default copy (e.g. show a device name once connected). */
  label?: string;
}) {
  const meta = STATUS_META[status];
  const Icon = meta.icon;
  const spinning = status === "scanning" || status === "connecting";

  return (
    <span
      className="inline-flex items-center gap-1.5 text-xs font-medium px-2.5 py-1 rounded-full"
      style={{
        color: meta.color,
        background: `color-mix(in srgb, ${meta.color} 12%, transparent)`,
        border: `1px solid color-mix(in srgb, ${meta.color} 30%, transparent)`,
      }}
    >
      <Icon size={12} className={spinning ? "animate-spin" : ""} />
      {label ?? meta.label}
    </span>
  );
}
