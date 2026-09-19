"use client";
import { useEffect, useRef, useState, type ReactNode } from "react";
import { devicesApi } from "@/utils/api";
import type { Device, CompatibleDeployment, UpdateStatus } from "@/types/devices";
import { Loader2 } from "lucide-react";
import toast from "react-hot-toast";

const UPDATE_STATUS_COLORS: Record<string, string> = {
  requested: "text-yellow-400",
  downloading: "text-blue-400",
  flashing: "text-purple-400",
  done: "text-green-400",
  failed: "text-red-400",
};

/**
 * The pre-Phase-1 `DevicePanel` body, extracted verbatim so `DevicePanel`
 * can branch on `project_type` (motion_phase1.md §5 C2 — the Devices page
 * branches one region, not the whole page body). Byte-for-byte the same
 * markup and logic as before the extraction; only the function name and
 * imports changed to make it a standalone component.
 */
export default function CameraDeviceDetails({
  dev,
  snapFrame,
  infResults,
  onUpdateStatusChange,
}: {
  dev: Device;
  snapFrame: any;
  infResults: any[];
  onUpdateStatusChange: (deviceId: string, status: UpdateStatus | null) => void;
}) {
  const [compat, setCompat] = useState<CompatibleDeployment | null | "none" | "loading">("loading");
  const [updateStatus, setUpdateStatus] = useState<UpdateStatus | null>(null);
  const [samplingForm, setSamplingForm] = useState({ label: "", length_ms: 2000, frequency: 62.5 });
  const [activeStream, setActiveStream] = useState<"snapshot" | "inference" | null>(null);
  const [fomoThreshold, setFomoThreshold] = useState<string>("");  // "" = use model metadata
  const keepaliveRef = useRef<ReturnType<typeof setInterval> | null>(null);

  function clearKeepalive() {
    if (keepaliveRef.current) { clearInterval(keepaliveRef.current); keepaliveRef.current = null; }
  }

  useEffect(() => {
    devicesApi.compatibleDeployment(dev.id)
      .then(({ data }) => setCompat(data ?? "none"))
      .catch(() => setCompat("none"));

    devicesApi.updateStatus(dev.id)
      .then(({ data }) => { if (data) setUpdateStatus(data); })
      .catch(() => { });

    return () => { clearKeepalive(); };
  }, [dev.id]);

  async function handleRequestUpdate() {
    try {
      await devicesApi.requestUpdate(dev.id);
      const status: UpdateStatus = { status: "requested", updated_at: new Date().toISOString() };
      setUpdateStatus(status);
      onUpdateStatusChange(dev.device_id, status);
      toast.success("Update requested");
    } catch {
      toast.error("Failed to request update");
    }
  }

  async function handleStartSampling() {
    try {
      await devicesApi.startSampling(dev.id, samplingForm);
      toast.success("Sampling started");
    } catch {
      toast.error("Failed to start sampling");
    }
  }

  async function stopStream() {
    clearKeepalive();
    setActiveStream(null);
    try { await devicesApi.streamStop(dev.id); } catch { /* best-effort */ }
  }

  async function handleSnapshotStart() {
    if (activeStream) await stopStream();
    try {
      await devicesApi.streamSnapshotStart(dev.id);
      setActiveStream("snapshot");
      keepaliveRef.current = setInterval(() => devicesApi.streamKeepalive(dev.id), 10_000);
    } catch {
      toast.error("Failed to start snapshot stream");
    }
  }

  async function handleInferenceStart() {
    if (activeStream) await stopStream();
    try {
      const thr = fomoThreshold !== "" ? parseFloat(fomoThreshold) : undefined;
      await devicesApi.streamInferenceStart(dev.id, thr);
      setActiveStream("inference");
      keepaliveRef.current = setInterval(() => devicesApi.streamKeepalive(dev.id), 10_000);
    } catch {
      toast.error("Failed to start inference stream");
    }
  }

  // ── Deployment row ──
  let deploymentContent: ReactNode;
  if (compat === "loading") {
    deploymentContent = <span className="text-gray-500 flex items-center gap-1"><Loader2 size={12} className="animate-spin" /> Checking…</span>;
  } else if (compat === "none") {
    deploymentContent = <span className="text-gray-500">No compatible deployment</span>;
  } else if (!compat) {
    deploymentContent = <span className="text-gray-500">No compatible deployment</span>;
  } else {
    const upToDate = compat.deployment_id === dev.installed_deployment_id;
    deploymentContent = (
      <div className="flex items-center gap-3 flex-wrap">
        {upToDate ? (
          <span className="text-green-400 text-xs">Up to date</span>
        ) : (
          <span className="text-yellow-400 text-xs">
            Update available{compat.deployment_target ? `: ${compat.deployment_target}` : ""}
            {compat.device_profile ? ` · ${compat.device_profile}` : ""}
          </span>
        )}
        {updateStatus && (
          <span className={`text-xs font-mono px-1.5 py-0.5 rounded bg-gray-800 ${UPDATE_STATUS_COLORS[updateStatus.status] ?? "text-gray-400"}`}>
            {updateStatus.status}
          </span>
        )}
        {!upToDate && !["requested", "downloading", "flashing"].includes(updateStatus?.status ?? "") && (
          <button onClick={handleRequestUpdate} className="btn-secondary" style={{ padding: "2px 8px", fontSize: "11px" }}>
            Request update
          </button>
        )}
      </div>
    );
  }

  return (
    <div className="mt-4 space-y-4 border-t border-gray-800 pt-4">
      {/* Deployment */}
      <div>
        <p className="text-xs text-gray-500 mb-1 font-medium uppercase tracking-wide">Deployment</p>
        {deploymentContent}
      </div>

      {/* Sampling */}
      {dev.is_online && (
        <div>
          <p className="text-xs text-gray-500 mb-2 font-medium uppercase tracking-wide">Sampling</p>
          <div className="flex flex-wrap gap-2 items-end">
            <div>
              <label className="label">Label</label>
              <input
                className="input"
                style={{ width: 120 }}
                placeholder="e.g. walking"
                value={samplingForm.label}
                onChange={e => setSamplingForm(f => ({ ...f, label: e.target.value }))}
              />
            </div>
            <div>
              <label className="label">Length (ms)</label>
              <input
                type="number"
                className="input"
                style={{ width: 90 }}
                value={samplingForm.length_ms}
                onChange={e => setSamplingForm(f => ({ ...f, length_ms: Number(e.target.value) }))}
              />
            </div>
            <div>
              <label className="label">Frequency (Hz)</label>
              <input
                type="number"
                className="input"
                style={{ width: 90 }}
                value={samplingForm.frequency}
                onChange={e => setSamplingForm(f => ({ ...f, frequency: Number(e.target.value) }))}
              />
            </div>
            <button onClick={handleStartSampling} className="btn-primary" style={{ alignSelf: "flex-end" }}>
              Start sampling
            </button>
            {dev.mode && dev.mode !== "idle" && (
              <span className="text-xs px-2 py-1 rounded bg-blue-900/40 text-blue-300 capitalize">{dev.mode}</span>
            )}
          </div>
        </div>
      )}

      {/* Debug streams */}
      {dev.is_online && (
        <div>
          <p className="text-xs text-gray-500 mb-2 font-medium uppercase tracking-wide">Debug streams</p>
          <div className="space-y-3">
            {/* Snapshot stream */}
            {dev.supports_snapshot_streaming && (
              <div>
                <div className="flex items-center gap-2 mb-1">
                  <span className="text-xs text-gray-400">Snapshot</span>
                  {activeStream !== "snapshot" ? (
                    <button onClick={handleSnapshotStart} className="btn-secondary" style={{ padding: "2px 8px", fontSize: "11px" }}>Start</button>
                  ) : (
                    <button onClick={stopStream} className="btn-secondary" style={{ padding: "2px 8px", fontSize: "11px" }}>Stop</button>
                  )}
                </div>
                {snapFrame && (
                  <div className="bg-gray-950 border border-gray-800 rounded-lg p-2">
                    {snapFrame.image ? (
                      <img src={`data:image/jpeg;base64,${snapFrame.image}`} alt="snapshot" className="max-w-full rounded" />
                    ) : (
                      <pre className="text-xs text-gray-400 overflow-x-auto">{JSON.stringify(snapFrame, null, 2)}</pre>
                    )}
                  </div>
                )}
              </div>
            )}

            {/* Inference stream */}
            <div>
              <div className="flex items-center gap-2 mb-1 flex-wrap">
                <span className="text-xs text-gray-400">Inference</span>
                {activeStream !== "inference" ? (
                  <>
                    <input
                      type="number"
                      min="0.01" max="0.99" step="0.01"
                      placeholder="threshold (model default)"
                      value={fomoThreshold}
                      onChange={e => setFomoThreshold(e.target.value)}
                      className="input"
                      style={{ width: 180, fontSize: "11px", padding: "2px 6px" }}
                    />
                    <button onClick={handleInferenceStart} className="btn-secondary" style={{ padding: "2px 8px", fontSize: "11px" }}>Start</button>
                  </>
                ) : (
                  <button onClick={stopStream} className="btn-secondary" style={{ padding: "2px 8px", fontSize: "11px" }}>Stop</button>
                )}
              </div>
              {infResults.length > 0 && (
                <div className="space-y-1">
                  {infResults.slice(0, 5).map((r, i) => (
                    <div key={i} className="bg-gray-950 border border-gray-800 rounded px-2 py-1 text-xs font-mono text-gray-400">
                      {r && (r.is_fomo || r.model_type === "detection_heatmap")
                        ? r.count === 0
                          ? "No detections"
                          : `${r.count} detection${r.count !== 1 ? "s" : ""}${Array.isArray(r.detections) && r.detections.length > 0
                            ? ": " + r.detections.slice(0, 3).map((d: any) =>
                              `${d.label ?? d.class ?? "obj"}${d.confidence != null ? ` (${(d.confidence * 100).toFixed(0)}%)` : ""}`
                            ).join(", ")
                            : ""
                          }`
                        : r && r.label != null
                          ? `${r.label}${r.confidence != null ? ` (${(r.confidence * 100).toFixed(0)}%)` : ""}`
                          : JSON.stringify(r)}
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
