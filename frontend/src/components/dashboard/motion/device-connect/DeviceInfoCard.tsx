"use client";
import { Loader2, Radio, Unplug } from "lucide-react";
import { SUPPORTED_BOARDS } from "./boards";
import ConnectionStatusBadge from "./ConnectionStatusBadge";
import type { ConnectedDeviceInfo } from "./types";

const BRAND = "var(--app-brand, #6366f1)";

export default function DeviceInfoCard({
  device,
  onDisconnect,
  disconnecting,
  compact,
}: {
  device: ConnectedDeviceInfo;
  onDisconnect?: () => void;
  disconnecting?: boolean;
  /** Denser layout for embedding inline on the page vs. the dialog's success step. */
  compact?: boolean;
}) {
  const board = SUPPORTED_BOARDS[device.boardId];
  const Icon = board.icon;

  return (
    <div
      className="rounded-xl"
      style={{ background: "var(--app-surface)", border: "1px solid var(--app-border)", boxShadow: "var(--app-shadow)" }}
    >
      <div className={`flex items-start justify-between gap-4 ${compact ? "p-4" : "p-5"}`}>
        <div className="flex items-start gap-3 min-w-0">
          <span
            className="flex-shrink-0 grid place-items-center rounded-lg"
            style={{
              width: compact ? 40 : 48,
              height: compact ? 40 : 48,
              background: "color-mix(in srgb, var(--app-brand, #6366f1) 10%, var(--app-surface-2))",
              border: "1px solid var(--app-border)",
            }}
          >
            <Icon size={compact ? 19 : 22} style={{ color: BRAND }} />
          </span>
          <div className="min-w-0">
            <div className="flex items-center gap-2 flex-wrap">
              <p className="text-sm font-semibold truncate" style={{ color: "var(--app-text)" }}>{device.deviceName}</p>
              <ConnectionStatusBadge status="connected" />
            </div>
            <p className="text-xs mt-0.5" style={{ color: "var(--app-text-soft)" }}>{board.label}</p>
          </div>
        </div>

        {onDisconnect && (
          <button
            type="button"
            onClick={onDisconnect}
            disabled={disconnecting}
            className="btn-secondary flex-shrink-0"
            style={{ fontSize: "12px", padding: "6px 10px" }}
          >
            {disconnecting ? <Loader2 size={13} className="animate-spin" /> : <Unplug size={13} />}
            Disconnect
          </button>
        )}
      </div>

      <div
        className={`grid grid-cols-2 sm:grid-cols-4 gap-x-4 gap-y-3 ${compact ? "px-4 pb-4" : "px-5 pb-5"}`}
        style={{ borderTop: "1px solid var(--app-border)", paddingTop: "1rem" }}
      >
        <Field label="Device ID" value={device.deviceId} mono />
        <Field label="USB port" value={device.port} mono />
        <Field label="VID / PID" value={`${device.vendorId} / ${device.productId}`} mono />
        <Field label="Firmware" value={device.firmwareVersion ?? "Unknown"} />
      </div>

      <div className={`${compact ? "px-4 pb-4" : "px-5 pb-5"}`} style={{ borderTop: "1px solid var(--app-border)", paddingTop: "0.75rem" }}>
        <div className="flex items-center gap-1.5 mb-2">
          <Radio size={13} style={{ color: "var(--app-text-soft)" }} />
          <p className="text-[11px] font-semibold uppercase tracking-wide" style={{ color: "var(--app-text-soft)" }}>
            Available sensors
          </p>
        </div>
        {device.sensors.length === 0 ? (
          <p className="text-xs" style={{ color: "var(--app-text-muted)" }}>No sensors reported.</p>
        ) : (
          <div className="flex flex-wrap gap-1.5">
            {device.sensors.map((s) => (
              <span
                key={s.name}
                className="inline-flex items-center gap-1 text-[11px] font-medium px-2 py-1 rounded-md"
                style={{ background: "var(--app-surface-2)", border: "1px solid var(--app-border)", color: "var(--app-text)" }}
              >
                {s.name}
                {s.freqHz && <span style={{ color: "var(--app-text-soft)" }}>· {s.freqHz}Hz</span>}
              </span>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

function Field({ label, value, mono }: { label: string; value: string; mono?: boolean }) {
  return (
    <div className="min-w-0">
      <p className="text-[10px] uppercase tracking-wide" style={{ color: "var(--app-text-soft)" }}>{label}</p>
      <p className={`text-xs mt-0.5 truncate ${mono ? "font-mono" : ""}`} style={{ color: "var(--app-text)" }} title={value}>
        {value}
      </p>
    </div>
  );
}
