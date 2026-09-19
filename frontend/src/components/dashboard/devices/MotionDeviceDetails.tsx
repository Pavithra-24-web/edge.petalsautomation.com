"use client";
import type { Device } from "@/types/devices";
import { Activity, Cable, Cpu } from "lucide-react";

interface NormalisedSensor {
  name: string;
  type?: string;
  freq_hz?: number;
  axes: string[];
}

/**
 * Tolerant, display-only counterpart to the backend's
 * app.motion.services.sensor_inventory.normalise — Device.sensors is
 * free-form JSON, and rows written by the CSV/serial forwarder path carry
 * bare strings instead of `{name, type, freq_hz, axes}` dicts (same
 * tolerance CollectFromDevice.tsx's sensorNames() already applies).
 */
function normaliseSensors(raw: any[] | undefined): NormalisedSensor[] {
  if (!Array.isArray(raw)) return [];
  const out: NormalisedSensor[] = [];
  for (const entry of raw) {
    if (typeof entry === "string") {
      if (entry) out.push({ name: entry, axes: [] });
      continue;
    }
    if (entry && typeof entry === "object" && typeof entry.name === "string" && entry.name) {
      out.push({
        name: entry.name,
        type: typeof entry.type === "string" ? entry.type : undefined,
        freq_hz: typeof entry.freq_hz === "number" ? entry.freq_hz : undefined,
        axes: Array.isArray(entry.axes) ? entry.axes.filter((a: any) => typeof a === "string") : [],
      });
    }
  }
  return out;
}

function StatusPill({ label, tone }: { label: string; tone: "idle" | "active" | "error" }) {
  const colors = {
    idle: { color: "var(--app-text-soft)", background: "var(--app-surface-2)", border: "var(--app-border)" },
    active: { color: "#a78bfa", background: "rgba(139, 92, 246, 0.12)", border: "rgba(139, 92, 246, 0.35)" },
    error: { color: "#f87171", background: "rgba(248, 113, 113, 0.12)", border: "rgba(248, 113, 113, 0.35)" },
  }[tone];
  return (
    <span
      className="text-[11px] px-2 py-0.5 rounded-full font-medium"
      style={{ color: colors.color, background: colors.background, border: `1px solid ${colors.border}` }}
    >
      {label}
    </span>
  );
}

/**
 * Motion device body — identity and sensor inventory only. The USB
 * connection flow lives entirely on the Motion Dataset page
 * (`components/dashboard/motion/device-connect/`) — this panel is
 * read-only, showing what a device last reported, and never initiates or
 * manages a connection itself (single-workflow requirement).
 */
export default function MotionDeviceDetails({ dev }: { dev: Device }) {
  const sensors = normaliseSensors(dev.sensors);

  return (
    <div className="mt-4 space-y-4 border-t border-gray-800 pt-4">
      {/* Identity */}
      <div>
        <p className="text-xs text-gray-500 mb-1 font-medium uppercase tracking-wide">Identity</p>
        <div className="flex flex-wrap items-center gap-2">
          <span className="inline-flex items-center gap-1.5 text-xs" style={{ color: "var(--app-text)" }}>
            <Cpu size={13} className="text-violet-400" /> {dev.device_type || "Unknown board"}
          </span>
          {dev.firmware_version && (
            <span className="text-xs font-mono" style={{ color: "var(--app-text-soft)" }}>
              fw {dev.firmware_version}
            </span>
          )}
          {dev.connection && (
            <span className="inline-flex items-center gap-1 text-xs" style={{ color: "var(--app-text-soft)" }}>
              <Cable size={12} /> {dev.connection}
            </span>
          )}
          <StatusPill label={dev.is_online ? "Online" : "Offline"} tone={dev.is_online ? "active" : "idle"} />
        </div>
      </div>

      {/* Sensors */}
      <div>
        <p className="text-xs text-gray-500 mb-2 font-medium uppercase tracking-wide">
          Sensors {sensors.length > 0 && `(${sensors.length})`}
        </p>
        {sensors.length === 0 ? (
          <p className="text-xs" style={{ color: "var(--app-text-soft)" }}>
            No sensors reported yet.
          </p>
        ) : (
          <div className="space-y-1.5">
            {sensors.map((s) => (
              <div
                key={s.name}
                className="flex flex-wrap items-center gap-2 rounded-lg px-3 py-2"
                style={{ background: "var(--app-surface-2)", border: "1px solid var(--app-border)" }}
              >
                <Activity size={13} className="text-violet-400 flex-shrink-0" />
                <span className="text-xs font-medium" style={{ color: "var(--app-text)" }}>{s.name}</span>
                {s.type && <span className="text-[11px]" style={{ color: "var(--app-text-soft)" }}>{s.type}</span>}
                {s.freq_hz != null && (
                  <span className="text-[11px] font-mono" style={{ color: "var(--app-text-soft)" }}>{s.freq_hz} Hz</span>
                )}
                {s.axes.length > 0 && (
                  <span className="text-[11px] font-mono" style={{ color: "var(--app-text-soft)" }}>
                    [{s.axes.join(", ")}]
                  </span>
                )}
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
