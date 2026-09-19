"use client";
import { useEffect, useMemo, useRef, useState } from "react";
import { samplesApi, devicesApi } from "@/utils/api";
import { useStudioWS } from "@/hooks/useStudioWS";
import { applyStudioEvent } from "@/utils/devicesReducer";
import type { Device, StudioEvent } from "@/types/devices";
import { Smartphone, RefreshCw, Radio, Play, Loader2, Database } from "lucide-react";
import toast from "react-hot-toast";
import Link from "next/link";

// ─── Collect from device ──────────────────────────────────────────────────────
// Live device sampling straight from Data acquisition — mirrors the flow on the
// Devices page but scoped to data collection: pick an online device → sensor →
// label → Start sampling. Device presence and sample lifecycle stay live via
// the same Studio WebSocket (useStudioWS) the Devices page uses. When a sample
// finishes it lands in the project dataset via the device's ingestion upload;
// we reload the recent-samples list (the same samplesApi.list path the dataset
// view uses) so the new sample surfaces without visiting the Devices page.
//
// Shared by ObjectDetectionDataset and MotionDataset — sensor selection is
// driven entirely by what the connected device reports, so nothing here is
// modality-specific.

interface RecentSample {
  id: string;
  filename?: string;
  label_name?: string | null;
  sample_type?: string;
  created_at?: string;
}

/** Distinct sensor names reported by a device. Each entry is normally a
 *  `{ name, type?, units? }` dict but we tolerate bare strings too. */
function sensorNames(dev: Device | null | undefined): string[] {
  const raw = dev?.sensors;
  if (!Array.isArray(raw)) return [];
  const names = raw
    .map((s: any) => (typeof s === "string" ? s : s?.name))
    .filter((n: any): n is string => typeof n === "string" && n.length > 0);
  return Array.from(new Set(names));
}

export function CollectFromDevice({ projectId }: { projectId: string }) {
  const [devices, setDevices] = useState<Device[]>([]);
  const [live, setLive] = useState<Record<string, Partial<Device>>>({});
  const [selectedPk, setSelectedPk] = useState<string | null>(null);

  const [sensor, setSensor] = useState("");
  const [category, setCategory] = useState<"training" | "testing">("training");
  const [label, setLabel] = useState("");
  const [lengthMs, setLengthMs] = useState(5000);
  const [frequency, setFrequency] = useState(100);

  const [sampling, setSampling] = useState(false);
  const [phase, setPhase] = useState<string | null>(null);
  const [recent, setRecent] = useState<RecentSample[]>([]);
  const [loadingDevices, setLoadingDevices] = useState(false);

  // The WS handler runs off a stable closure (useStudioWS keeps the latest
  // callback), so read the current selection through a ref rather than state.
  const selectedDeviceIdRef = useRef<string | null>(null);
  const doneTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  // Merge REST rows with live WS overrides (is_online / mode), same as Devices.
  const merged = useMemo(
    () => devices.map(d => ({ ...d, ...live[d.device_id] })),
    [devices, live],
  );
  const online = useMemo(() => merged.filter(d => d.is_online), [merged]);
  const selected = merged.find(d => d.id === selectedPk) ?? null;
  const availableSensors = sensorNames(selected);

  useEffect(() => {
    selectedDeviceIdRef.current = selected?.device_id ?? null;
  }, [selected?.device_id]);

  function clearDoneTimer() {
    if (doneTimerRef.current) { clearTimeout(doneTimerRef.current); doneTimerRef.current = null; }
  }

  async function loadDevices() {
    setLoadingDevices(true);
    try {
      const { data } = await devicesApi.list(projectId);
      setDevices(Array.isArray(data) ? data : []);
    } catch { /* ignore — badge shows disconnected */ } finally {
      setLoadingDevices(false);
    }
  }

  async function refreshRecent() {
    try {
      const { data } = await samplesApi.list(projectId, { limit: 6 });
      setRecent(data?.items ?? (Array.isArray(data) ? data : []));
    } catch { /* ignore */ }
  }

  function finishSampling(ok: boolean, msg?: string) {
    clearDoneTimer();
    setSampling(false);
    setPhase(null);
    if (ok) { if (msg) toast.success(msg); refreshRecent(); }
    else if (msg) { toast.error(msg); }
  }

  function handleSampleEvent(e: StudioEvent) {
    switch (e.type) {
      case "sample.requested": setPhase("Requesting…"); break;
      case "sample.acked":     setPhase("Acknowledged"); break;
      case "sample.started":   setPhase("Recording…"); break;
      case "sample.stopped":   finishSampling(true, "Sample collected — added to dataset"); break;
      case "sample.rejected":  finishSampling(false, "Device rejected the sample"); break;
      case "sample.failed":
        finishSampling(false, `Sampling failed${e.payload?.error ? `: ${e.payload.error}` : ""}`);
        break;
    }
  }

  function handleEvent(e: StudioEvent) {
    if (e.type.startsWith("sample.") && e.device_id && e.device_id === selectedDeviceIdRef.current) {
      handleSampleEvent(e);
    }
    // Keep the device list fresh on connect/disconnect (mirrors Devices page).
    if (e.type === "device.connected" || e.type === "device.disconnected") {
      loadDevices();
    }
    setLive(prev => applyStudioEvent(prev, e));
  }

  const { connected: wsConnected } = useStudioWS(projectId, handleEvent);

  // Initial load + reset on project change.
  useEffect(() => {
    setSelectedPk(null);
    setLive({});
    loadDevices();
    refreshRecent();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [projectId]);

  // Auto-select the first online device and keep the selection valid.
  const onlineIds = online.map(d => d.id).join(",");
  useEffect(() => {
    if (online.length === 0) return;
    if (!selectedPk || !online.some(d => d.id === selectedPk)) {
      setSelectedPk(online[0].id);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [onlineIds, selectedPk]);

  // Default the sensor to the selected device's first reported sensor.
  useEffect(() => {
    const names = sensorNames(selected);
    if (names.length > 0 && !names.includes(sensor)) setSensor(names[0]);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selected?.id]);

  useEffect(() => () => clearDoneTimer(), []);

  const canStart =
    !!selected && !!selected.is_online && !sampling &&
    label.trim().length > 0 && lengthMs > 0 && frequency > 0;

  async function startSampling() {
    if (!selected) return;
    if (!selected.is_online) { toast.error("Device is offline"); return; }

    setSampling(true);
    setPhase("Starting…");
    clearDoneTimer();
    // Safety net: if a sample.stopped event is missed, release the UI a little
    // past the requested duration and refresh so it never stays stuck.
    doneTimerRef.current = setTimeout(() => {
      setSampling(false);
      setPhase(null);
      refreshRecent();
    }, Math.min(lengthMs, 120000) + 5000);

    try {
      await devicesApi.startSampling(selected.id, {
        label: label.trim(),
        length_ms: lengthMs,
        frequency,
        sensor: sensor.trim() || undefined,
        category,
      });
      setPhase("Recording…");
      toast.success("Sampling started");
    } catch (err: any) {
      const status = err?.response?.status;
      const detail = err?.response?.data?.detail;
      const msg =
        detail ??
        (status === 408 ? "Device did not acknowledge in time"
          : status === 409 ? "Device is busy or offline"
          : "Failed to start sampling");
      finishSampling(false, msg);
    }
  }

  return (
    <div className="card">
      {/* Header */}
      <div className="flex items-start justify-between gap-3 mb-4">
        <div className="flex items-center gap-2">
          <Smartphone size={16} className="text-gray-400" />
          <h3 className="text-sm font-semibold text-gray-300">Collect from device</h3>
          {!wsConnected && (
            <span className="text-[11px] px-2 py-0.5 rounded-full"
              style={{ color: "var(--app-text-muted)", background: "var(--app-surface-2)", border: "1px solid var(--app-border)" }}>
              Live updates disconnected
            </span>
          )}
        </div>
        <button
          type="button"
          onClick={loadDevices}
          disabled={loadingDevices}
          className="btn-secondary"
          style={{ padding: "4px 10px", fontSize: "12px" }}
          aria-label="Refresh devices"
        >
          <RefreshCw size={13} className={loadingDevices ? "animate-spin" : ""} /> Refresh
        </button>
      </div>

      {online.length === 0 ? (
        <div className="text-xs text-gray-500 flex flex-wrap items-center gap-1">
          <span>No online devices for this project.</span>
          <Link href="/dashboard/devices" className="text-brand-400 hover:underline" style={{ color: "#8b5cf6" }}>
            Register or connect a device →
          </Link>
        </div>
      ) : (
        <>
          {/* Device selector */}
          <div className="mb-4">
            <label className="label">Device</label>
            <select
              className="input"
              value={selectedPk ?? ""}
              onChange={e => setSelectedPk(e.target.value || null)}
            >
              {online.map(d => (
                <option key={d.id} value={d.id}>
                  {d.name}{d.mode && d.mode !== "idle" ? ` · ${d.mode}` : ""}
                </option>
              ))}
            </select>
          </div>

          {/* Sampling form — driven by the selected device */}
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
            {/* Sensor */}
            <div>
              <label className="label">Sensor</label>
              {availableSensors.length > 0 ? (
                <select className="input" value={sensor} onChange={e => setSensor(e.target.value)}>
                  {availableSensors.map(s => <option key={s} value={s}>{s}</option>)}
                </select>
              ) : (
                <input
                  className="input"
                  placeholder="e.g. accelerometer"
                  value={sensor}
                  onChange={e => setSensor(e.target.value)}
                />
              )}
            </div>

            {/* Category */}
            <div>
              <label className="label">Category</label>
              <div className="flex gap-2">
                {(["training", "testing"] as const).map(c => (
                  <button
                    key={c}
                    type="button"
                    onClick={() => setCategory(c)}
                    className={`flex-1 rounded-lg px-3 py-2 text-xs font-medium capitalize border transition-colors ${category === c ? "text-white" : ""}`}
                    style={category === c
                      ? { background: "linear-gradient(90deg, #6366f1, #8b5cf6)", borderColor: "transparent" }
                      : { background: "var(--app-surface-2)", borderColor: "var(--app-border)", color: "var(--app-text-soft)" }}
                    aria-pressed={category === c}
                  >
                    {c}
                  </button>
                ))}
              </div>
            </div>

            {/* Label */}
            <div>
              <label className="label">Label</label>
              <input
                className="input"
                placeholder="e.g. walking"
                value={label}
                onChange={e => setLabel(e.target.value)}
              />
            </div>

            {/* Sample length + frequency */}
            <div className="grid grid-cols-2 gap-3">
              <div>
                <label className="label">Length (ms)</label>
                <input
                  type="number"
                  min={100}
                  className="input"
                  value={lengthMs}
                  onChange={e => setLengthMs(Number(e.target.value))}
                />
              </div>
              <div>
                <label className="label">Frequency (Hz)</label>
                <input
                  type="number"
                  min={1}
                  className="input"
                  value={frequency}
                  onChange={e => setFrequency(Number(e.target.value))}
                />
              </div>
            </div>
          </div>

          {/* Start + live phase */}
          <div className="flex items-center gap-3 mt-4">
            <button
              type="button"
              onClick={startSampling}
              disabled={!canStart}
              className="btn-primary disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {sampling
                ? <><Loader2 size={14} className="animate-spin" /> Sampling…</>
                : <><Play size={14} /> Start sampling</>}
            </button>
            {phase && (
              <span className="flex items-center gap-1.5 text-xs text-blue-300">
                <Radio size={13} className="animate-pulse" /> {phase}
              </span>
            )}
            {selected && !selected.is_online && (
              <span className="text-xs text-gray-500">Selected device went offline</span>
            )}
          </div>
        </>
      )}

      {/* Recently collected — reuses the dataset list path so a new sample shows up */}
      {recent.length > 0 && (
        <div className="mt-5 border-t border-gray-800 pt-4">
          <div className="flex items-center justify-between mb-2">
            <p className="text-[11px] font-semibold uppercase tracking-wider" style={{ color: "var(--app-text-soft)" }}>
              Recently collected
            </p>
            <Link href="/dashboard/data/dataset" className="text-xs hover:underline" style={{ color: "#8b5cf6" }}>
              View dataset →
            </Link>
          </div>
          <div className="space-y-1.5">
            {recent.map(s => (
              <div key={s.id}
                className="flex items-center gap-2 rounded-lg px-3 py-2 text-xs"
                style={{ background: "var(--app-surface-2)", border: "1px solid var(--app-border)" }}>
                <Database size={13} style={{ color: "var(--app-text-soft)" }} />
                <span className="font-mono truncate flex-1" style={{ color: "var(--app-text)" }}>
                  {s.filename?.split("/").pop()?.split("\\").pop() ?? s.id}
                </span>
                {s.label_name && (
                  <span className="px-1.5 py-0.5 rounded" style={{ background: "var(--app-surface)", color: "var(--app-text-soft)" }}>
                    {s.label_name}
                  </span>
                )}
                {s.sample_type && <span style={{ color: "var(--app-text-muted)" }}>{s.sample_type}</span>}
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
