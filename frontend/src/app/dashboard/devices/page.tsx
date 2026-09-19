"use client";
import { useEffect, useState } from "react";
import Link from "next/link";
import { useAppStore } from "@/store/appStore";
import { devicesApi } from "@/utils/api";
import { useStudioWS } from "@/hooks/useStudioWS";
import type { Device, StudioEvent, UpdateStatus, DeviceKey, DeviceKeySecret } from "@/types/devices";
import { applyStudioEvent } from "@/utils/devicesReducer";
import CameraDeviceDetails from "@/components/dashboard/devices/CameraDeviceDetails";
import MotionDeviceDetails from "@/components/dashboard/devices/MotionDeviceDetails";
import {
  Smartphone, Wifi, WifiOff, Plus, RefreshCw, Copy, CheckCircle,
  ChevronDown, ChevronUp, Loader2, Key, Trash2, AlertTriangle,
} from "lucide-react";
import toast from "react-hot-toast";

const BACKEND_HOST = process.env.NEXT_PUBLIC_API_URL || "http://<backend-host>:8010";

const DEVICE_TYPES = [
  { id: "arduino", label: "Arduino", icon: "🔧" },
  { id: "raspberry_pi", label: "Raspberry Pi", icon: "🍓" },
];

// ─── Expanded device panel ────────────────────────────────────────────────────
// Branches its body on the active project's type (motion_phase1.md §5 C2):
// registration, keys, list, WS wiring and the empty state above stay one
// shared page — only this region inside the panel differs. The camera body
// is CameraDeviceDetails, extracted verbatim so it stays byte-identical to
// before this branch existed (Phase 1.1 acceptance criterion 7).
function DevicePanel({
  dev,
  projectType,
  snapFrame,
  infResults,
  onUpdateStatusChange,
}: {
  dev: Device;
  projectType?: "object_detection" | "motion";
  snapFrame: any;
  infResults: any[];
  onUpdateStatusChange: (deviceId: string, status: UpdateStatus | null) => void;
}) {
  if (projectType === "motion") {
    return <MotionDeviceDetails dev={dev} />;
  }
  return (
    <CameraDeviceDetails
      dev={dev}
      snapFrame={snapFrame}
      infResults={infResults}
      onUpdateStatusChange={onUpdateStatusChange}
    />
  );
}

// ─── Device keys section ──────────────────────────────────────────────────────
function DeviceKeysSection({ projectId }: { projectId: string }) {
  const [keys, setKeys] = useState<DeviceKey[]>([]);
  const [loading, setLoading] = useState(false);
  const [name, setName] = useState("");
  const [creating, setCreating] = useState(false);
  const [secret, setSecret] = useState<DeviceKeySecret | null>(null);
  const [copiedId, setCopiedId] = useState<string | null>(null);

  async function loadKeys() {
    setLoading(true);
    try {
      const { data } = await devicesApi.listKeys(projectId);
      setKeys(data);
    } catch {
      toast.error("Failed to load device keys");
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    loadKeys();
    // Reset any previously-shown secret when switching projects.
    setSecret(null);
    setName("");
  }, [projectId]);

  async function createKey() {
    if (creating) return;
    setCreating(true);
    try {
      const { data } = await devicesApi.createKey(projectId, name.trim());
      setSecret(data);
      setName("");
      toast.success("Device key created — copy the secret now");
      loadKeys();
    } catch {
      toast.error("Failed to create device key");
    } finally {
      setCreating(false);
    }
  }

  async function revokeKey(keyId: string) {
    try {
      await devicesApi.revokeKey(projectId, keyId);
      toast.success("Device key revoked");
      loadKeys();
    } catch {
      toast.error("Failed to revoke device key");
    }
  }

  function copyToClipboard(text: string, id: string) {
    navigator.clipboard.writeText(text);
    setCopiedId(id);
    setTimeout(() => setCopiedId(null), 2000);
    toast.success("Copied!");
  }

  const snippet =
    `PETAL_HOST=${BACKEND_HOST} \\\n` +
    `PETAL_API_KEY=${secret?.api_key ?? ""} \\\n` +
    `PETAL_DEVICE_ID=my-device-01 \\\n` +
    `python client.py`;

  return (
    <div className="card">
      <div className="flex items-center gap-2 mb-1">
        <Key size={16} className="text-gray-400" />
        <h3 className="text-sm font-semibold text-gray-300">Device keys</h3>
      </div>
      <p className="text-xs text-gray-500 mb-4">
        Mint an <span className="font-mono">ef_…</span> key for a device to authenticate its
        connection. The full secret is shown only once, at creation.
      </p>

      {/* Create form */}
      <div className="flex flex-wrap gap-2 items-end mb-4">
        <div className="flex-1 min-w-[180px]">
          <label className="label">Key name</label>
          <input
            className="input"
            placeholder="e.g. Production fleet"
            value={name}
            onChange={e => setName(e.target.value)}
            onKeyDown={e => { if (e.key === "Enter") createKey(); }}
          />
        </div>
        <button onClick={createKey} disabled={creating} className="btn-primary">
          {creating ? <Loader2 size={14} className="animate-spin" /> : <Plus size={14} />}
          Create device key
        </button>
      </div>

      {/* Freshly-created secret — shown once */}
      {secret && (
        <div className="mb-4 rounded-lg border border-yellow-600/50 bg-yellow-900/10 p-3">
          <div className="flex items-center gap-2 mb-2 text-yellow-400">
            <AlertTriangle size={14} />
            <span className="text-xs font-semibold">
              Copy these now — they will not be shown again.
            </span>
          </div>

          {/* api_key */}
          <label className="label">API key</label>
          <div className="flex items-center gap-2 bg-gray-800 rounded-lg px-3 py-2 mb-2">
            <span className="text-xs text-gray-300 font-mono truncate flex-1">{secret.api_key}</span>
            <button
              onClick={() => copyToClipboard(secret.api_key, "secret-api")}
              className="text-gray-500 hover:text-gray-300 flex-shrink-0"
            >
              {copiedId === "secret-api" ? <CheckCircle size={13} className="text-green-400" /> : <Copy size={13} />}
            </button>
          </div>

          {/* hmac_key */}
          <label className="label">HMAC key</label>
          <div className="flex items-center gap-2 bg-gray-800 rounded-lg px-3 py-2 mb-3">
            <span className="text-xs text-gray-300 font-mono truncate flex-1">{secret.hmac_key}</span>
            <button
              onClick={() => copyToClipboard(secret.hmac_key, "secret-hmac")}
              className="text-gray-500 hover:text-gray-300 flex-shrink-0"
            >
              {copiedId === "secret-hmac" ? <CheckCircle size={13} className="text-green-400" /> : <Copy size={13} />}
            </button>
          </div>

          {/* Ready-to-run connection snippet */}
          <label className="label">Connect a device (device_client/client.py)</label>
          <div className="relative bg-gray-950 border border-gray-800 rounded-lg p-3">
            <button
              onClick={() => copyToClipboard(snippet, "secret-snippet")}
              className="absolute top-2 right-2 text-gray-500 hover:text-gray-300"
              aria-label="Copy connection snippet"
            >
              {copiedId === "secret-snippet" ? <CheckCircle size={13} className="text-green-400" /> : <Copy size={13} />}
            </button>
            <pre className="text-xs text-gray-400 font-mono overflow-x-auto whitespace-pre pr-8">{snippet}</pre>
          </div>
        </div>
      )}

      {/* Existing keys list */}
      {loading ? (
        <div className="space-y-2">
          {[1, 2].map(i => <div key={i} className="h-10 bg-gray-900 border border-gray-800 rounded-lg animate-pulse" />)}
        </div>
      ) : keys.length === 0 ? (
        <p className="text-xs text-gray-500">No device keys yet.</p>
      ) : (
        <div className="space-y-2">
          {keys.map(k => (
            <div key={k.id} className="flex items-center gap-3 bg-gray-800/50 rounded-lg px-3 py-2">
              <div className="min-w-0 flex-1">
                <p className="text-xs text-gray-300 truncate">{k.name || <span className="text-gray-500 italic">unnamed</span>}</p>
                <p className="text-[11px] text-gray-500 font-mono">
                  {k.api_key_prefix ? `${k.api_key_prefix}…` : "—"}
                  {k.created_at && <span> · {new Date(k.created_at).toLocaleDateString()}</span>}
                </p>
              </div>
              <button
                onClick={() => revokeKey(k.id)}
                className="btn-secondary flex-shrink-0"
                style={{ padding: "2px 8px", fontSize: "11px" }}
              >
                <Trash2 size={12} /> Revoke
              </button>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

// ─── Main page ────────────────────────────────────────────────────────────────
export default function DevicesPage() {
  const { activeProject } = useAppStore();
  const [devices, setDevices] = useState<Device[]>([]);
  const [loading, setLoading] = useState(false);
  const [showRegister, setShowRegister] = useState(false);
  const [form, setForm] = useState({ name: "", device_type: "esp32" });
  const [copiedId, setCopiedId] = useState<string | null>(null);

  // WS-augmented runtime state
  const [liveDevices, setLiveDevices] = useState<Record<string, Partial<Device>>>({});
  const [expanded, setExpanded] = useState<string | null>(null);
  const [snapFrames, setSnapFrames] = useState<Record<string, any>>({});
  const [infResults, setInfResults] = useState<Record<string, any[]>>({});
  const [updateStatuses, setUpdateStatuses] = useState<Record<string, any>>({});

  function handleEvent(e: StudioEvent) {
    if (e.type === "snapshot.frame" && e.device_id) {
      setSnapFrames(m => ({ ...m, [e.device_id!]: e.payload }));
      return;
    }
    if (e.type === "inference.result" && e.device_id) {
      setInfResults(m => ({ ...m, [e.device_id!]: [e.payload, ...(m[e.device_id!] ?? [])].slice(0, 20) }));
      return;
    }
    if (e.type === "stream.failed" && e.device_id) {
      toast.error(`Stream failed for device ${e.device_id}`);
      setSnapFrames(m => { const n = { ...m }; delete n[e.device_id!]; return n; });
      setInfResults(m => { const n = { ...m }; delete n[e.device_id!]; return n; });
      return;
    }
    if (e.type.startsWith("update.") && e.device_id) {
      setUpdateStatuses(m => ({ ...m, [e.device_id!]: { status: e.type.replace("update.", ""), ...e.payload } }));
    }
    // Re-fetch REST state on connect/disconnect so last_seen and is_online stay accurate.
    if (e.type === "device.connected" || e.type === "device.disconnected") {
      loadDevices();
    }
    setLiveDevices(prev => applyStudioEvent(prev, e));
  }

  const { connected: wsConnected } = useStudioWS(activeProject?.id ?? null, handleEvent);

  // Merge REST + live WS overrides
  const merged = devices.map(d => ({ ...d, ...liveDevices[d.device_id] }));
  const onlineCount = merged.filter(d => d.is_online).length;
  const offlineCount = merged.length - onlineCount;

  useEffect(() => {
    if (!activeProject) return;
    loadDevices();
    // Periodic fallback refresh so online status stays accurate if SSE events are missed.
    const interval = setInterval(loadDevices, 30_000);
    return () => clearInterval(interval);
  }, [activeProject]);

  async function loadDevices() {
    if (!activeProject) return;
    setLoading(true);
    try {
      const { data } = await devicesApi.list(activeProject.id);
      setDevices(data);
    } finally { setLoading(false); }
  }

  async function registerDevice() {
    if (!form.name.trim() || !activeProject) return;
    try {
      const { data } = await devicesApi.register({ ...form, project_id: activeProject.id });
      setDevices(d => [...d, data]);
      setShowRegister(false);
      setForm({ name: "", device_type: "esp32" });
      toast.success("Device registered");
    } catch { toast.error("Registration failed"); }
  }

  function copyToClipboard(text: string, id: string) {
    navigator.clipboard.writeText(text);
    setCopiedId(id);
    setTimeout(() => setCopiedId(null), 2000);
    toast.success("Copied!");
  }

  return (
    <div className="pe-devices max-w-6xl mx-auto space-y-6">

      {/* Empty-state animations: phone floats with a soft glow, sparkles
          twinkle on a staggered loop. Transform/opacity/filter only —
          theme-agnostic. Honors prefers-reduced-motion. */}
      <style>{`
        @keyframes pe-device-float {
          0%   { transform: translateY(0)    rotate(0deg);   }
          25%  { transform: translateY(-4px) rotate(-2deg);   }
          50%  { transform: translateY(-7px) rotate(0deg);   }
          75%  { transform: translateY(-4px) rotate(2deg);    }
          100% { transform: translateY(0)    rotate(0deg);   }
        }
        @keyframes pe-device-glow {
          0%, 100% { filter: drop-shadow(0 3px 6px rgba(99, 102, 241, 0.18)); }
          50%      { filter: drop-shadow(0 10px 22px rgba(139, 92, 246, 0.50)); }
        }
        @keyframes pe-sparkle-twinkle {
          0%, 100% { opacity: 0.2;  transform: scale(0.6)  rotate(0deg);   }
          40%      { opacity: 1;    transform: scale(1.35) rotate(180deg); }
          50%      { opacity: 1;    transform: scale(1.4)  rotate(180deg); }
          60%      { opacity: 1;    transform: scale(1.35) rotate(180deg); }
        }
        .pe-devices-illu .illu-device {
          animation: pe-device-float 3.6s cubic-bezier(0.45, 0, 0.55, 1) infinite,
                     pe-device-glow  3.6s ease-in-out infinite;
          will-change: transform, filter;
          transform-origin: center bottom;
        }
        .pe-devices-illu .illu-sparkle {
          animation: pe-sparkle-twinkle 2.2s ease-in-out infinite;
          will-change: opacity, transform;
          display: inline-block;
        }
        .pe-devices-illu .illu-sparkle:nth-of-type(1) { animation-delay: 0s;    }
        .pe-devices-illu .illu-sparkle:nth-of-type(2) { animation-delay: 0.55s; }
        .pe-devices-illu .illu-sparkle:nth-of-type(3) { animation-delay: 1.1s;  }
        .pe-devices-illu .illu-sparkle:nth-of-type(4) { animation-delay: 1.65s; }
        @media (prefers-reduced-motion: reduce) {
          .pe-devices-illu .illu-device,
          .pe-devices-illu .illu-sparkle { animation: none; }
        }
      `}</style>

      {/* Header */}
      <div className="flex items-start justify-between gap-6 flex-wrap">
        <div className="min-w-0">
          <div className="flex items-center gap-2">
            <h1 className="pe-devices-title">Devices</h1>
            {!wsConnected && (
              <span className="text-[11px] px-2 py-0.5 rounded-full" style={{ color: "var(--app-text-muted)", background: "var(--app-surface-2)", border: "1px solid var(--app-border)" }}>
                Live updates disconnected
              </span>
            )}
          </div>
          <p className="pe-devices-sub">
            {onlineCount > 0 && <span style={{ color: "#059669" }}>{onlineCount} online</span>}
            {onlineCount > 0 && offlineCount > 0 && " · "}
            {offlineCount > 0 && <span>{offlineCount} offline</span>}
            {devices.length === 0 && "No devices registered"}
          </p>
          {activeProject?.project_type === "motion" && (
            <p className="text-xs mt-1" style={{ color: "var(--app-text-soft)" }}>
              Connect a Motion device over USB from the{" "}
              <Link href="/dashboard/data/dataset" className="underline" style={{ color: "var(--app-brand, #6366f1)" }}>
                Data Labeling
              </Link>{" "}
              page.
            </p>
          )}
        </div>
        <div className="flex items-center gap-3">
          <button
            onClick={loadDevices}
            className="data-acq-btn-primary"
            aria-label="Refresh devices"
            disabled={loading}
          >
            <RefreshCw size={14} className={loading ? "animate-spin" : ""} /> Refresh
          </button>
          <Link href="/dashboard/devices/catalog" className="data-acq-btn-primary">
            <Smartphone size={14} /> Device catalog
          </Link>
          <Link href="/dashboard/devices/connect" className="data-acq-btn-primary">
            <Plus size={14} /> Connect Device
          </Link>
          <button onClick={() => setShowRegister(true)} className="data-acq-btn-primary">
            <Plus size={14} /> Register device
          </button>
        </div>
      </div>

      {/* Register form */}
      {showRegister && (
        <div className="card border-brand-700">
          <h3 className="text-sm font-semibold text-gray-300 mb-4">Register new device</h3>
          <div className="grid grid-cols-2 gap-4 mb-4">
            <div>
              <label className="label">Device name</label>
              <input
                autoFocus
                className="input"
                placeholder="e.g. ESP32 Dev Board #1"
                value={form.name}
                onChange={e => setForm(f => ({ ...f, name: e.target.value }))}
              />
            </div>
            <div>
              <label className="label">Device type</label>
              <select
                className="input"
                value={form.device_type}
                onChange={e => setForm(f => ({ ...f, device_type: e.target.value }))}
              >
                {DEVICE_TYPES.map(t => (
                  <option key={t.id} value={t.id}>{t.icon} {t.label}</option>
                ))}
              </select>
            </div>
          </div>
          <div className="flex gap-2">
            <button onClick={registerDevice} className="btn-primary">Register</button>
            <button onClick={() => setShowRegister(false)} className="btn-secondary">Cancel</button>
          </div>
        </div>
      )}

      {/* Device keys */}
      {activeProject && <DeviceKeysSection projectId={activeProject.id} />}

      {/* Device grid */}
      {loading ? (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {[1, 2, 3].map(i => (
            <div key={i} className="h-40 bg-gray-900 border border-gray-800 rounded-xl animate-pulse" />
          ))}
        </div>
      ) : devices.length === 0 ? (
        <div className="pe-devices-card">
          <div className="pe-devices-inner">
            <div className="pe-devices-illu" aria-hidden="true">
              <span className="illu-sparkle" style={{ top: 18, left: 26 }}>✦</span>
              <span className="illu-sparkle" style={{ top: 12, right: 22 }}>✦</span>
              <span className="illu-sparkle" style={{ bottom: 20, left: 18 }}>✦</span>
              <span className="illu-sparkle" style={{ bottom: 28, right: 24, fontSize: 10 }}>✦</span>
              <Smartphone size={52} strokeWidth={1.6} className="illu-device" />
            </div>
            <p className="pe-devices-empty-title">No devices yet</p>
            <p className="pe-devices-empty-sub">
              Register a device using the button above to connect automatically
            </p>
          </div>
        </div>
      ) : (
        <div className="space-y-3">
          {merged.map(dev => {
            const dtype = DEVICE_TYPES.find(t => t.id === dev.device_type);
            const isExpanded = expanded === dev.id;

            return (
              <div key={dev.id} className="card hover:border-gray-700 transition-colors">
                {/* Card header */}
                <div className="flex items-start justify-between">
                  <div className="flex items-center gap-3 flex-1 min-w-0">
                    <div className="text-2xl flex-shrink-0">{dtype?.icon || "🔩"}</div>
                    <div className="min-w-0">
                      <p className="font-medium text-gray-200 text-sm">{dev.name}</p>
                      <p className="text-xs text-gray-500">{dtype?.label || dev.device_type}</p>
                    </div>
                  </div>
                  <div className="flex items-center gap-3 flex-shrink-0">
                    <div className={`flex items-center gap-1.5 text-xs font-medium ${dev.is_online ? "text-green-400" : "text-gray-500"
                      }`}>
                      {dev.is_online
                        ? <><Wifi size={12} /> Online</>
                        : <><WifiOff size={12} /> Offline</>}
                    </div>
                    <button
                      onClick={() => setExpanded(isExpanded ? null : dev.id)}
                      className="btn-ghost p-1"
                    >
                      {isExpanded ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
                    </button>
                  </div>
                </div>

                {/* Inline meta */}
                <div className="mt-3 space-y-1.5 text-xs">
                  {dev.firmware_version && (
                    <div className="flex justify-between">
                      <span className="text-gray-500">Firmware</span>
                      <span className="text-gray-300 font-mono">{dev.firmware_version}</span>
                    </div>
                  )}
                  {dev.ip_address && (
                    <div className="flex justify-between">
                      <span className="text-gray-500">IP</span>
                      <span className="text-gray-300 font-mono">{dev.ip_address}</span>
                    </div>
                  )}
                  {dev.last_seen && (
                    <div className="flex justify-between">
                      <span className="text-gray-500">Last seen</span>
                      <span className="text-gray-400">{new Date(dev.last_seen).toLocaleString()}</span>
                    </div>
                  )}
                  {dev.device_metadata?.current_model_id && (
                    <div className="flex justify-between">
                      <span className="text-gray-500">Model</span>
                      <span className="text-gray-300 font-mono text-xs truncate max-w-[200px]">
                        {dev.device_metadata.current_model_id}
                      </span>
                    </div>
                  )}
                  {dev.device_metadata?.diagnostics && (() => {
                    const diag = dev.device_metadata.diagnostics;
                    const metrics: string[] = [];
                    if (typeof diag.cpu_percent === "number")  metrics.push(`CPU ${diag.cpu_percent}%`);
                    if (typeof diag.mem_percent === "number")  metrics.push(`RAM ${diag.mem_percent}%`);
                    if (typeof diag.temp_c === "number")       metrics.push(`${diag.temp_c}°C`);
                    if (typeof diag.disk_percent === "number") metrics.push(`Disk ${diag.disk_percent}%`);
                    const sensors = diag.sensors;
                    const hasHealth = metrics.length > 0 || (sensors && (typeof sensors.imu === "boolean" || typeof sensors.camera === "boolean"));
                    if (!hasHealth) return null;
                    return (
                      <div className="flex justify-between items-start gap-3">
                        <span className="text-gray-500">Health</span>
                        <div className="flex flex-wrap justify-end gap-1.5 max-w-[220px]">
                          {metrics.map(m => (
                            <span key={m} className="text-gray-300 font-mono px-1.5 py-0.5 rounded bg-gray-800">{m}</span>
                          ))}
                          {sensors && typeof sensors.camera === "boolean" && (
                            <span className={`px-1.5 py-0.5 rounded ${sensors.camera ? "bg-green-900/40 text-green-400" : "bg-gray-800 text-gray-500"}`}>Cam</span>
                          )}
                          {sensors && typeof sensors.imu === "boolean" && (
                            <span className={`px-1.5 py-0.5 rounded ${sensors.imu ? "bg-green-900/40 text-green-400" : "bg-gray-800 text-gray-500"}`}>IMU</span>
                          )}
                        </div>
                      </div>
                    );
                  })()}
                </div>

                {/* Device ID copy row */}
                <div className="mt-3 flex items-center gap-2 bg-gray-800 rounded-lg px-3 py-2">
                  <span className="text-xs text-gray-500 font-mono truncate flex-1">{dev.id}</span>
                  <button
                    onClick={() => copyToClipboard(dev.id, dev.id)}
                    className="text-gray-500 hover:text-gray-300 flex-shrink-0"
                  >
                    {copiedId === dev.id
                      ? <CheckCircle size={13} className="text-green-400" />
                      : <Copy size={13} />}
                  </button>
                </div>

                {/* Expanded panel */}
                {isExpanded && activeProject && (
                  <DevicePanel
                    dev={dev}
                    projectType={activeProject.project_type}
                    snapFrame={snapFrames[dev.device_id]}
                    infResults={infResults[dev.device_id] ?? []}
                    onUpdateStatusChange={(deviceId, status) =>
                      setUpdateStatuses(m => ({ ...m, [deviceId]: status }))
                    }
                  />
                )}
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
