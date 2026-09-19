"use client";
import { useEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { CheckCircle2, RadioTower, RefreshCw, Usb, X, XCircle } from "lucide-react";
import toast from "react-hot-toast";
import { SUPPORTED_BOARDS } from "./boards";
import { createMockMotionConnectionService, type MotionConnectionService } from "./connectionService";
import DeviceInfoCard from "./DeviceInfoCard";
import EmptyState from "./EmptyState";
import SupportedDeviceList from "./SupportedDeviceList";
import type { ConnectedDeviceInfo, ConnectionPhase, DiscoveredDevice } from "./types";

const BRAND = "var(--app-brand, #6366f1)";

const CONNECT_STEPS = ["Requesting access", "Reading device identity", "Establishing connection"];
const ENTER_MS = 190;

// Mock fallback only — every real mount passes its own `service` prop (a
// per-project WebSerialMotionConnectionService, constructed by whoever opens
// this dialog). This exists so the dialog still renders standalone.
const defaultService = createMockMotionConnectionService();

/** Skeleton row matching a real device card's shape (icon + two text lines),
 *  so the loading state reads as "device cards resolving" rather than
 *  generic placeholder bars. */
function ScanningSkeletonRow() {
  return (
    <div
      className="flex items-center gap-3 rounded-xl px-4 py-3.5"
      style={{ background: "var(--app-surface)", border: "1px solid var(--app-border)" }}
    >
      <span className="dc-shimmer flex-shrink-0 w-10 h-10 rounded-lg" />
      <div className="min-w-0 flex-1 space-y-2">
        <span className="dc-shimmer block h-3 rounded" style={{ width: "55%" }} />
        <span className="dc-shimmer block h-2.5 rounded" style={{ width: "80%" }} />
      </div>
    </div>
  );
}

export default function USBConnectionDialog({
  open,
  onClose,
  onConnected,
  service = defaultService,
}: {
  open: boolean;
  onClose: () => void;
  onConnected: (device: ConnectedDeviceInfo) => void;
  service?: MotionConnectionService;
}) {
  const [mounted, setMounted] = useState(open);
  const [entered, setEntered] = useState(false);
  const [phase, setPhase] = useState<ConnectionPhase>("scanning");
  const [devices, setDevices] = useState<DiscoveredDevice[]>([]);
  const [rescanning, setRescanning] = useState(false);
  const [requestingDevice, setRequestingDevice] = useState(false);
  const [connectingScanId, setConnectingScanId] = useState<string | null>(null);
  const [stepIndex, setStepIndex] = useState(0);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [connectedDevice, setConnectedDevice] = useState<ConnectedDeviceInfo | null>(null);
  const pendingDeviceRef = useRef<DiscoveredDevice | null>(null);
  const aliveRef = useRef(true);

  useEffect(() => {
    aliveRef.current = true;
    return () => { aliveRef.current = false; };
  }, []);

  // Mount/animate-in immediately on open; delay actual unmount on close so
  // the closing transition can play out.
  useEffect(() => {
    if (open) {
      setMounted(true);
      const raf = requestAnimationFrame(() => setEntered(true));
      return () => cancelAnimationFrame(raf);
    }
    setEntered(false);
    const t = setTimeout(() => setMounted(false), ENTER_MS);
    return () => clearTimeout(t);
  }, [open]);

  useEffect(() => {
    if (!open) return;
    setPhase("scanning");
    setDevices([]);
    setConnectingScanId(null);
    setErrorMessage(null);
    setConnectedDevice(null);
    runScan();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open]);

  async function runScan(isRescan = false) {
    if (isRescan) setRescanning(true);
    else setPhase("scanning");
    try {
      const found = await service.scanDevices();
      if (!aliveRef.current) return;
      setDevices(found);
      setPhase("device-list");
    } catch {
      if (!aliveRef.current) return;
      setErrorMessage("Couldn't scan for USB devices. Check browser permissions and try again.");
      setPhase("error");
    } finally {
      if (aliveRef.current) setRescanning(false);
    }
  }

  async function handleConnect(device: DiscoveredDevice) {
    pendingDeviceRef.current = device;
    setConnectingScanId(device.scanId);
    setStepIndex(0);
    setPhase("connecting");

    const t1 = setTimeout(() => aliveRef.current && setStepIndex(1), 450);
    const t2 = setTimeout(() => aliveRef.current && setStepIndex(2), 950);

    try {
      const info = await service.connect(device);
      clearTimeout(t1); clearTimeout(t2);
      if (!aliveRef.current) return;
      setConnectedDevice(info);
      setPhase("connected");
      toast.success(`${info.deviceName} connected`);
      onConnected(info);
    } catch {
      clearTimeout(t1); clearTimeout(t2);
      if (!aliveRef.current) return;
      setErrorMessage(`Couldn't connect to ${device.suggestedName}. Make sure it's still plugged in.`);
      setPhase("error");
    } finally {
      if (aliveRef.current) setConnectingScanId(null);
    }
  }

  function handleRetry() {
    const device = pendingDeviceRef.current;
    if (device) handleConnect(device);
    else runScan();
  }

  // Authorizing a board never granted access before is a browser-native
  // picker, not a rescan of what's already authorized — Web Serial has no
  // way to discover unauthorized hardware silently. Resolving to a device
  // proceeds straight into the same connect flow a list click would.
  async function handleRequestDevice() {
    setRequestingDevice(true);
    try {
      const device = await service.requestDevice();
      if (!aliveRef.current) return;
      if (!device) {
        toast("No new device selected.");
        return;
      }
      setDevices((prev) => (prev.some((d) => d.scanId === device.scanId) ? prev : [...prev, device]));
      setPhase("device-list");
      await handleConnect(device);
    } catch {
      if (!aliveRef.current) return;
      toast.error("Couldn't open the device picker. Check browser permissions and try again.");
    } finally {
      if (aliveRef.current) setRequestingDevice(false);
    }
  }

  const connecting = phase === "connecting";

  if (!mounted) return null;

  return createPortal(
    <div
      className="overlay-modal fixed inset-0 z-[100] flex items-center justify-center px-4"
      style={{ opacity: entered ? 1 : 0, transition: `opacity ${ENTER_MS}ms ease` }}
      onClick={() => !connecting && onClose()}
    >
      <style>{`
        @keyframes dc-shimmer {
          0% { background-position: 100% 50%; }
          100% { background-position: 0% 50%; }
        }
        .dc-shimmer {
          background: linear-gradient(90deg,
            var(--app-surface-2) 25%,
            color-mix(in srgb, var(--app-brand, #6366f1) 14%, var(--app-surface-2)) 42%,
            var(--app-surface-2) 60%);
          background-size: 250% 100%;
          animation: dc-shimmer 1.5s ease-in-out infinite;
        }
        @keyframes dc-radar {
          0%   { transform: scale(0.7); opacity: 0.55; }
          100% { transform: scale(1.9); opacity: 0; }
        }
        .dc-radar-ring {
          animation: dc-radar 1.8s cubic-bezier(0.2, 0.6, 0.4, 1) infinite;
        }
        @media (prefers-reduced-motion: reduce) {
          .dc-shimmer, .dc-radar-ring { animation: none; }
        }
      `}</style>
      <div
        className="surface-raised rounded-2xl w-full flex flex-col"
        style={{
          maxWidth: 480,
          maxHeight: "88vh",
          opacity: entered ? 1 : 0,
          transform: entered ? "scale(1) translateY(0)" : "scale(0.96) translateY(10px)",
          transition: `opacity ${ENTER_MS}ms ease, transform ${ENTER_MS}ms cubic-bezier(0.2, 0.7, 0.3, 1)`,
        }}
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div className="flex items-start justify-between gap-4 px-5 sm:px-6 pt-5 sm:pt-6 pb-4 flex-shrink-0">
          <div className="flex items-center gap-3 min-w-0">
            <span
              className="flex-shrink-0 grid place-items-center w-10 h-10 rounded-lg"
              style={{ background: "color-mix(in srgb, var(--app-brand, #6366f1) 12%, var(--app-surface-2))", border: "1px solid var(--app-border)" }}
            >
              <Usb size={18} style={{ color: BRAND }} />
            </span>
            <div className="min-w-0">
              <h2 className="text-base font-semibold" style={{ color: "var(--app-text)" }}>Connect a device</h2>
              <p className="text-xs mt-0.5 truncate" style={{ color: "var(--app-text-muted)" }}>
                {SUPPORTED_BOARDS.arduino_uno_q.label} &amp; {SUPPORTED_BOARDS.raspberry_pi.label} over USB
              </p>
            </div>
          </div>
          <button
            type="button"
            onClick={onClose}
            disabled={connecting}
            aria-label="Close"
            className="flex-shrink-0 p-1.5 rounded-lg transition-colors disabled:opacity-40 disabled:cursor-not-allowed"
            style={{ color: "var(--app-text-soft)" }}
          >
            <X size={16} />
          </button>
        </div>

        {/* Body */}
        <div className="px-5 sm:px-6 pb-6 overflow-y-auto" style={{ minHeight: 220 }}>
          {phase === "scanning" && (
            <div className="space-y-2.5">
              <div className="flex flex-col items-center text-center gap-3 py-2 mb-1">
                <span className="relative grid place-items-center w-12 h-12 rounded-full" style={{ background: "color-mix(in srgb, var(--app-brand, #6366f1) 12%, var(--app-surface-2))" }}>
                  <span className="dc-radar-ring absolute inset-0 rounded-full" style={{ background: "color-mix(in srgb, var(--app-brand, #6366f1) 35%, transparent)" }} />
                  <span className="dc-radar-ring absolute inset-0 rounded-full" style={{ background: "color-mix(in srgb, var(--app-brand, #6366f1) 35%, transparent)", animationDelay: "0.6s" }} />
                  <RadioTower size={19} style={{ color: BRAND }} />
                </span>
                <p className="text-xs font-medium" style={{ color: "var(--app-text-muted)" }}>
                  Searching for supported boards over USB…
                </p>
              </div>
              <ScanningSkeletonRow />
              <ScanningSkeletonRow />
            </div>
          )}

          {phase === "device-list" && devices.length > 0 && (
            <>
              <p className="text-xs font-medium mb-3" style={{ color: "var(--app-text-muted)" }}>
                {devices.length} device{devices.length !== 1 ? "s" : ""} found — select one to connect
              </p>
              <SupportedDeviceList devices={devices} connectingId={connectingScanId} onConnect={handleConnect} />
            </>
          )}

          {phase === "device-list" && devices.length === 0 && (
            <EmptyState
              onRescan={() => runScan(true)}
              rescanning={rescanning}
              onRequestDevice={handleRequestDevice}
              requesting={requestingDevice}
            />
          )}

          {phase === "connecting" && pendingDeviceRef.current && (
            <div className="flex flex-col items-center text-center gap-4 py-4">
              <span className="relative grid place-items-center w-14 h-14 rounded-full" style={{ background: "color-mix(in srgb, var(--app-brand, #6366f1) 12%, var(--app-surface-2))" }}>
                <span className="absolute inset-0 rounded-full animate-ping" style={{ background: "color-mix(in srgb, var(--app-brand, #6366f1) 25%, transparent)" }} />
                <Usb size={20} className="relative" style={{ color: BRAND }} />
              </span>
              <div>
                <p className="text-sm font-semibold" style={{ color: "var(--app-text)" }}>
                  Connecting to {pendingDeviceRef.current.suggestedName}
                </p>
                <p className="text-xs mt-1" style={{ color: "var(--app-text-muted)" }}>Reading device info automatically…</p>
              </div>
              <div className="w-full space-y-2 mt-1 text-left">
                {CONNECT_STEPS.map((step, i) => {
                  const done = i < stepIndex;
                  const active = i === stepIndex;
                  return (
                    <div key={step} className="flex items-center gap-2.5 text-xs">
                      {done ? (
                        <CheckCircle2 size={14} className="text-green-400 flex-shrink-0" />
                      ) : active ? (
                        <span className="flex-shrink-0 w-3.5 h-3.5 rounded-full animate-spin" style={{ border: "1.5px solid var(--app-border-strong)", borderTopColor: BRAND }} />
                      ) : (
                        <span className="flex-shrink-0 w-3.5 h-3.5 rounded-full" style={{ border: "1.5px solid var(--app-border-strong)" }} />
                      )}
                      <span style={{ color: done || active ? "var(--app-text)" : "var(--app-text-soft)" }}>{step}</span>
                    </div>
                  );
                })}
              </div>
            </div>
          )}

          {phase === "connected" && connectedDevice && (
            <div className="space-y-4">
              <div className="flex items-center gap-2">
                <CheckCircle2 size={16} className="text-green-400" />
                <p className="text-sm font-semibold" style={{ color: "var(--app-text)" }}>Device connected</p>
              </div>
              <DeviceInfoCard device={connectedDevice} compact />
            </div>
          )}

          {phase === "error" && (
            <div className="flex flex-col items-center text-center gap-3 py-4">
              <span className="grid place-items-center w-12 h-12 rounded-full" style={{ background: "color-mix(in srgb, #ef4444 12%, transparent)" }}>
                <XCircle size={20} className="text-red-400" />
              </span>
              <div>
                <p className="text-sm font-semibold" style={{ color: "var(--app-text)" }}>Connection failed</p>
                <p className="text-xs mt-1 max-w-xs" style={{ color: "var(--app-text-muted)" }}>{errorMessage}</p>
              </div>
              <div className="flex flex-wrap items-center justify-center gap-2 mt-1">
                <button type="button" onClick={handleRetry} className="btn-primary" style={{ fontSize: "12px" }}>
                  <RefreshCw size={13} /> Retry
                </button>
                <button type="button" onClick={() => runScan()} className="btn-secondary" style={{ fontSize: "12px" }}>
                  Rescan
                </button>
              </div>
            </div>
          )}
        </div>

        {/* Footer */}
        <div
          className="flex items-center justify-between gap-3 px-5 sm:px-6 py-4 rounded-b-2xl flex-wrap flex-shrink-0"
          style={{ borderTop: "1px solid var(--app-border)", background: "var(--app-surface-2)" }}
        >
          <p className="text-[11px]" style={{ color: "var(--app-text-soft)" }}>
            {phase === "connected" ? "You can close this and connect again anytime." : "The app reads your device automatically."}
          </p>
          <div className="flex items-center gap-2">
            {(phase === "device-list" && devices.length > 0) && (
              <>
                <button
                  type="button"
                  onClick={handleRequestDevice}
                  disabled={requestingDevice}
                  className="btn-secondary"
                  style={{ fontSize: "12px" }}
                >
                  <Usb size={13} /> {requestingDevice ? "Waiting…" : "Select a different device…"}
                </button>
                <button type="button" onClick={() => runScan(true)} disabled={rescanning} className="btn-secondary" style={{ fontSize: "12px" }}>
                  <RefreshCw size={13} className={rescanning ? "animate-spin" : ""} /> Rescan
                </button>
              </>
            )}
            <button
              type="button"
              onClick={onClose}
              disabled={connecting}
              className={phase === "connected" ? "btn-primary" : "btn-secondary"}
              style={{ fontSize: "12px" }}
            >
              {phase === "connected" ? "Done" : "Cancel"}
            </button>
          </div>
        </div>
      </div>
    </div>,
    document.body
  );
}
