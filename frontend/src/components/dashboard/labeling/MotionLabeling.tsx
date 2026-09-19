"use client";
import { useEffect, useMemo, useRef, useState } from "react";
import { useAppStore } from "@/store/appStore";
import { samplesApi, labelsApi } from "@/utils/api";
import {
  Home, ChevronRight, RefreshCw, Plus, X, Trash2, Activity,
  Hand, Database, Loader2, LineChart, Usb,
} from "lucide-react";
import toast from "react-hot-toast";
import {
  type SplitKey, type PieSegment, PREMIUM_SOFT_PALETTE, PieChart, EmptyFolderIllustration,
} from "./LabelingShared";
import MotionShellNotice from "../motion/MotionShellNotice";
import USBConnectionDialog from "../motion/device-connect/USBConnectionDialog";
import DeviceInfoCard from "../motion/device-connect/DeviceInfoCard";
import { createWebSerialMotionConnectionService, isWebSerialSupported } from "../motion/device-connect/webSerialTransport";
import type { MotionConnectionService } from "../motion/device-connect/connectionService";
import type { ConnectedDeviceInfo } from "../motion/device-connect/types";

const SPLIT_TABS: { value: SplitKey; label: string }[] = [
  { value: "training", label: "Training" },
  { value: "testing", label: "Testing" },
  { value: "postprocessing", label: "Post-processing" },
];

function formatDate(value?: string | null) {
  if (!value) return "—";
  const d = new Date(value);
  if (Number.isNaN(d.getTime())) return "—";
  return d.toLocaleDateString(undefined, { month: "short", day: "numeric", year: "numeric" });
}

/**
 * Labeling — Motion content component.
 *
 * §6.2: gesture labels, recording sessions and signal preview, no bounding
 * boxes / image annotation. Gesture labels and recording sessions are real
 * here — both are plain CRUD over the existing generic `labelsApi` /
 * `samplesApi` (label taxonomy and sample rows carry no image-specific
 * shape). The signal preview / recording timeline is a shell: rendering a
 * per-axis waveform needs raw sample content read + charting that doesn't
 * exist yet anywhere in the app (see MotionDataset's identical note) — out
 * of Phase 0's scope per motion_phase0.md's scope boundary.
 */
export default function MotionLabeling() {
  const { activeProject } = useAppStore();

  const [samples, setSamples] = useState<any[]>([]);
  const [labels, setLabels] = useState<any[]>([]);
  const [loading, setLoading] = useState(false);
  const [refreshing, setRefreshing] = useState(false);
  const [filterType, setFilterType] = useState<SplitKey>("training");
  const [newLabelName, setNewLabelName] = useState("");
  const [creatingLabel, setCreatingLabel] = useState(false);
  const [deletingLabelId, setDeletingLabelId] = useState<string | null>(null);
  const [busySampleId, setBusySampleId] = useState<string | null>(null);

  // USB device connection — see device-connect/. One MotionConnectionService
  // instance per active project, real Web Serial transport (Batch 2 of
  // motion_phase1.md); per Architecture Principle 2 nothing else in the app
  // is allowed to talk to navigator.serial or the device WebSocket directly.
  const webSerialSupported = useMemo(() => isWebSerialSupported(), []);
  const motionConnectionService = useMemo(
    () => (activeProject ? createWebSerialMotionConnectionService(activeProject.id) : null),
    [activeProject?.id],
  );
  const [usbDialogOpen, setUsbDialogOpen] = useState(false);
  const [connectedDevice, setConnectedDevice] = useState<ConnectedDeviceInfo | null>(null);
  const [disconnecting, setDisconnecting] = useState(false);

  // Tear down the live connection (WS session + heartbeat timer + serial
  // port) on unmount or on switching projects — otherwise the heartbeat
  // interval and the open port outlive the component that owns them.
  const connectionRef = useRef<{ device: ConnectedDeviceInfo | null; service: MotionConnectionService | null }>({
    device: null,
    service: null,
  });
  useEffect(() => {
    connectionRef.current = { device: connectedDevice, service: motionConnectionService };
  });
  useEffect(() => {
    return () => {
      const { device, service } = connectionRef.current;
      if (device && service) service.disconnect(device.deviceId).catch(() => {});
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [activeProject?.id]);

  const labelColorMap = useMemo(() => {
    const sorted = [...labels].map(l => l.name as string).sort((a, b) => a.localeCompare(b));
    const m = new Map<string, string>();
    sorted.forEach((name, i) => m.set(name.toLowerCase(), PREMIUM_SOFT_PALETTE[i % PREMIUM_SOFT_PALETTE.length]));
    return m;
  }, [labels]);

  const colorForLabel = (name: string): string => {
    if (!name) return "rgba(148,163,184,0.55)";
    return labelColorMap.get(name.toLowerCase()) ?? PREMIUM_SOFT_PALETTE[0];
  };

  async function loadLabels() {
    if (!activeProject) return;
    try {
      const { data } = await labelsApi.list(activeProject.id);
      setLabels(data ?? []);
    } catch { /* ignore */ }
  }

  async function loadSamples() {
    if (!activeProject) return;
    setLoading(true);
    try {
      const { data } = await samplesApi.list(activeProject.id, { sample_type: filterType, limit: 1000 });
      setSamples(data?.items ?? (Array.isArray(data) ? data : []));
    } catch {
      setSamples([]);
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    if (!activeProject) return;
    loadLabels();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [activeProject?.id]);

  // A connection belongs to the project it was opened under — switching
  // projects gets a fresh MotionConnectionService (above), so any previously
  // connected device no longer applies to what's on screen.
  useEffect(() => {
    setConnectedDevice(null);
  }, [activeProject?.id]);

  useEffect(() => {
    loadSamples();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [activeProject?.id, filterType]);

  const labelCounts = useMemo(() => {
    const counts: Record<string, number> = {};
    for (const s of samples) {
      const name = s.label_name || "Unlabeled";
      counts[name] = (counts[name] ?? 0) + 1;
    }
    return counts;
  }, [samples]);

  const segments: PieSegment[] = useMemo(
    () => Object.entries(labelCounts)
      .sort((a, b) => b[1] - a[1])
      .map(([name, value]) => ({
        key: name,
        label: name,
        value,
        color: name === "Unlabeled" ? "rgba(148,163,184,0.55)" : colorForLabel(name),
      })),
    [labelCounts, labelColorMap], // eslint-disable-line react-hooks/exhaustive-deps
  );

  async function createLabel() {
    const name = newLabelName.trim();
    if (!name || !activeProject) return;
    setCreatingLabel(true);
    try {
      const { data } = await labelsApi.create({ project_id: activeProject.id, name });
      setLabels(prev => [...prev, data]);
      setNewLabelName("");
      toast.success(`Gesture "${name}" added`);
    } catch (e: any) {
      toast.error(e?.response?.data?.detail || "Failed to add gesture label");
    } finally {
      setCreatingLabel(false);
    }
  }

  async function deleteLabel(id: string, name: string) {
    setDeletingLabelId(id);
    try {
      await labelsApi.delete(id);
      setLabels(prev => prev.filter(l => l.id !== id));
      toast.success(`Gesture "${name}" removed`);
    } catch (e: any) {
      toast.error(e?.response?.data?.detail || "Failed to remove gesture label");
    } finally {
      setDeletingLabelId(null);
    }
  }

  async function assignLabel(sampleId: string, labelId: string) {
    setBusySampleId(sampleId);
    try {
      if (labelId) {
        await samplesApi.assignLabel(sampleId, labelId);
      }
      await loadSamples();
    } catch (e: any) {
      toast.error(e?.response?.data?.detail || "Failed to update label");
    } finally {
      setBusySampleId(null);
    }
  }

  async function deleteRecording(sampleId: string) {
    setBusySampleId(sampleId);
    try {
      await samplesApi.delete(sampleId);
      setSamples(prev => prev.filter(s => s.id !== sampleId));
      toast.success("Recording deleted");
    } catch (e: any) {
      toast.error(e?.response?.data?.detail || "Failed to delete recording");
    } finally {
      setBusySampleId(null);
    }
  }

  async function disconnectDevice() {
    if (!connectedDevice || !motionConnectionService) return;
    setDisconnecting(true);
    try {
      await motionConnectionService.disconnect(connectedDevice.deviceId);
      setConnectedDevice(null);
      toast.success("Device disconnected");
    } finally {
      setDisconnecting(false);
    }
  }

  if (!activeProject) {
    return (
      <div className="flex items-center justify-center h-64">
        <p className="text-gray-500">Select a project from the sidebar to get started.</p>
      </div>
    );
  }

  const hasLabels = labels.length > 0;

  return (
    <div className="dataset-page data-acq-page w-full space-y-6">
      {/* Header */}
      <div className="flex items-start justify-between gap-6 flex-wrap">
        <div className="min-w-0 flex-1">
          <div className="data-acq-header-top">
            <nav className="data-acq-breadcrumb" aria-label="Breadcrumb">
              <Home size={13} aria-hidden="true" />
              <ChevronRight size={12} aria-hidden="true" />
              <span className="crumb-current">Data Labeling</span>
            </nav>
          </div>
          <h1 className="data-acq-title">Data Labeling</h1>
          <p className="data-acq-subtitle">Manage gesture labels and review recording sessions.</p>
        </div>
        <div className="flex items-center gap-3 flex-shrink-0">
          <button
            onClick={() => setUsbDialogOpen(true)}
            className="data-acq-btn-primary"
            disabled={!!connectedDevice || !webSerialSupported}
            title={webSerialSupported ? undefined : "Connect Device requires Chrome, Edge, or another Web Serial–capable browser."}
          >
            <Usb size={14} /> {connectedDevice ? "Device connected" : "Connect Device"}
          </button>
          <button
            onClick={async () => {
              if (refreshing) return;
              setRefreshing(true);
              try { await Promise.all([loadSamples(), loadLabels()]); } finally { setRefreshing(false); }
            }}
            disabled={refreshing}
            className="data-acq-btn-primary"
          >
            <RefreshCw size={14} className={refreshing ? "animate-spin" : ""} /> Refresh
          </button>
        </div>
      </div>

      {/* Connected device — auto-populated identity, no typed fields anywhere
          in this flow (device-connect/). Backed by a real Web Serial
          connection and the /ws/device hello/heartbeat contract. */}
      {connectedDevice && (
        <DeviceInfoCard device={connectedDevice} onDisconnect={disconnectDevice} disconnecting={disconnecting} />
      )}

      {/* Summary */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-5">
        <div className="data-acq-stat-card data-acq-stat-card--total">
          <div className="min-w-0">
            <p className="stat-eyebrow">Recordings collected</p>
            <p className="stat-value">
              {samples.length.toLocaleString()}
              <span className="stat-unit">Items</span>
            </p>
          </div>
          {hasLabels && samples.length > 0 && (
            <div className="data-acq-stat-donut">
              <PieChart segments={segments} />
            </div>
          )}
        </div>
        <div className="data-acq-stat-card">
          <div className="min-w-0">
            <p className="stat-eyebrow">Gesture labels</p>
            <p className="stat-value">{labels.length.toLocaleString()}</p>
          </div>
        </div>
        <div className="data-acq-stat-card">
          <div className="min-w-0">
            <p className="stat-eyebrow">Viewing split</p>
            <p className="stat-value" style={{ fontSize: "1.1rem" }}>
              {SPLIT_TABS.find(t => t.value === filterType)?.label}
            </p>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-[19rem_1fr] gap-5 items-start">
        {/* Gesture labels panel */}
        <div className="card">
          <div className="flex items-center gap-2 mb-4">
            <Hand size={16} className="text-gray-400" />
            <h3 className="text-sm font-semibold text-gray-300">Gesture labels</h3>
          </div>
          <div className="flex gap-2 mb-4">
            <input
              className="input flex-1"
              placeholder="e.g. wave"
              value={newLabelName}
              onChange={e => setNewLabelName(e.target.value)}
              onKeyDown={e => { if (e.key === "Enter") createLabel(); }}
            />
            <button
              type="button"
              onClick={createLabel}
              disabled={creatingLabel || !newLabelName.trim()}
              className="btn-primary disabled:opacity-50 disabled:cursor-not-allowed"
              aria-label="Add gesture label"
            >
              {creatingLabel ? <Loader2 size={14} className="animate-spin" /> : <Plus size={14} />}
            </button>
          </div>
          {labels.length === 0 ? (
            <p className="text-xs" style={{ color: "var(--app-text-soft)" }}>
              No gesture labels yet — add one above.
            </p>
          ) : (
            <div className="space-y-1.5">
              {labels.map(l => (
                <div
                  key={l.id}
                  className="flex items-center gap-2 rounded-lg px-3 py-2 text-xs"
                  style={{ background: "var(--app-surface-2)", border: "1px solid var(--app-border)" }}
                >
                  <span
                    className="h-2.5 w-2.5 rounded-full shrink-0"
                    style={{ background: colorForLabel(l.name) }}
                    aria-hidden="true"
                  />
                  <span className="flex-1 truncate font-medium" style={{ color: "var(--app-text)" }}>
                    {l.name}
                  </span>
                  <span style={{ color: "var(--app-text-muted)" }}>
                    {labelCounts[l.name] ?? 0}
                  </span>
                  <button
                    type="button"
                    onClick={() => deleteLabel(l.id, l.name)}
                    disabled={deletingLabelId === l.id}
                    className="shrink-0 opacity-60 hover:opacity-100 transition-opacity"
                    style={{ color: "var(--app-text-soft)" }}
                    aria-label={`Remove ${l.name}`}
                  >
                    {deletingLabelId === l.id
                      ? <Loader2 size={13} className="animate-spin" />
                      : <X size={13} />}
                  </button>
                </div>
              ))}
            </div>
          )}
        </div>

        {/* Recording sessions */}
        <div className="card">
          <div className="flex items-center justify-between gap-3 mb-4 flex-wrap">
            <div className="flex items-center gap-2">
              <Database size={16} className="text-gray-400" />
              <h3 className="text-sm font-semibold text-gray-300">Recording sessions</h3>
            </div>
            <div className="flex gap-1.5">
              {SPLIT_TABS.map(t => (
                <button
                  key={t.value}
                  type="button"
                  onClick={() => setFilterType(t.value)}
                  className={`rounded-lg px-3 py-1.5 text-xs font-medium transition-colors border ${filterType === t.value ? "text-white" : ""}`}
                  style={filterType === t.value
                    ? { background: "linear-gradient(90deg, #7c3aed, #a855f7)", borderColor: "transparent" }
                    : { background: "var(--app-surface-2)", borderColor: "var(--app-border)", color: "var(--app-text-soft)" }}
                >
                  {t.label}
                </button>
              ))}
            </div>
          </div>

          {loading ? (
            <div className="flex items-center justify-center py-14">
              <Loader2 size={20} className="animate-spin" style={{ color: "var(--app-text-soft)" }} />
            </div>
          ) : samples.length === 0 ? (
            <div className="flex flex-col items-center justify-center py-10 text-center gap-3">
              <EmptyFolderIllustration />
              <div>
                <p className="text-sm font-semibold" style={{ color: "var(--app-text)" }}>
                  No recordings in this split yet
                </p>
                <p className="text-xs mt-0.5" style={{ color: "var(--app-text-muted)" }}>
                  Upload a CSV or record live from Data acquisition.
                </p>
              </div>
            </div>
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full text-xs">
                <thead>
                  <tr style={{ color: "var(--app-text-soft)", borderBottom: "1px solid var(--app-border)" }}>
                    <th className="text-left font-medium py-2 pr-3">Recording</th>
                    <th className="text-left font-medium py-2 pr-3">Gesture</th>
                    <th className="text-left font-medium py-2 pr-3">Collected</th>
                    <th className="text-right font-medium py-2 pl-3">Actions</th>
                  </tr>
                </thead>
                <tbody>
                  {samples.map(s => {
                    const baseName = s.filename?.split("/").pop()?.split("\\").pop() || s.id;
                    const busy = busySampleId === s.id;
                    return (
                      <tr key={s.id} style={{ borderBottom: "1px solid var(--app-border)" }}>
                        <td className="py-2.5 pr-3">
                          <span className="font-mono truncate block max-w-[16rem]" style={{ color: "var(--app-text)" }} title={baseName}>
                            {baseName}
                          </span>
                        </td>
                        <td className="py-2.5 pr-3">
                          <select
                            className="input"
                            style={{ padding: "4px 8px", fontSize: "12px", width: "auto" }}
                            value={s.label_id ?? ""}
                            disabled={busy}
                            onChange={e => assignLabel(s.id, e.target.value)}
                          >
                            <option value="">Unlabeled</option>
                            {labels.map(l => (
                              <option key={l.id} value={l.id}>{l.name}</option>
                            ))}
                          </select>
                        </td>
                        <td className="py-2.5 pr-3" style={{ color: "var(--app-text-soft)" }}>
                          {formatDate(s.created_at)}
                        </td>
                        <td className="py-2.5 pl-3 text-right">
                          <button
                            type="button"
                            onClick={() => deleteRecording(s.id)}
                            disabled={busy}
                            className="opacity-60 hover:opacity-100 hover:text-red-400 transition-opacity"
                            style={{ color: "var(--app-text-soft)" }}
                            aria-label={`Delete ${baseName}`}
                          >
                            {busy ? <Loader2 size={13} className="animate-spin" /> : <Trash2 size={13} />}
                          </button>
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          )}
        </div>
      </div>

      {/* Signal preview / recording timeline — arrives with motion feature generation */}
      <div className="card">
        <div className="flex items-center gap-2 mb-1">
          <LineChart size={16} className="text-gray-400" />
          <h3 className="text-sm font-semibold text-gray-300">Signal preview &amp; timeline</h3>
        </div>
        <MotionShellNotice
          icon={Activity}
          title="Signal preview arrives with motion processing"
          description="Select a recording once this is wired up to see its per-axis waveform and where each gesture falls along the timeline."
        />
      </div>

      {motionConnectionService && (
        <USBConnectionDialog
          open={usbDialogOpen}
          onClose={() => setUsbDialogOpen(false)}
          onConnected={setConnectedDevice}
          service={motionConnectionService}
        />
      )}
    </div>
  );
}
