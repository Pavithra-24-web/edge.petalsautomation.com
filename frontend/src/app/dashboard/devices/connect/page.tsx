"use client";
import { useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import Link from "next/link";
import { useAppStore } from "@/store/appStore";
import { devicesApi, deviceClientApi } from "@/utils/api";
import { useStudioWS } from "@/hooks/useStudioWS";
import { applyStudioEvent } from "@/utils/devicesReducer";
import type { Device, StudioEvent, DeviceClientManifest } from "@/types/devices";
import {
  ArrowLeft, Copy, CheckCircle, Loader2, Wifi, WifiOff, Package, Terminal, ShieldCheck,
} from "lucide-react";
import toast from "react-hot-toast";

// The one command targets the backend that serves /install.sh at its root
// (contract §14.3) — the backend host, not the /api/v1 base. In production that's
// api.iotai.show (see next.config.js NEXT_PUBLIC_API_URL). Devices must reach it
// directly (never localhost, per §1); override with NEXT_PUBLIC_INSTALL_HOST for
// LAN/dev (e.g. http://<backend-lan-ip>:8010).
const INSTALL_HOST = process.env.NEXT_PUBLIC_INSTALL_HOST || "https://api.iotai.show";
const INSTALL_COMMAND = `curl -fsSL ${INSTALL_HOST}/install.sh | bash`;

const BRAND = "var(--app-brand, #6366f1)";

type Step = { title: string; detail?: string; code?: string };

type ConnectType = {
  id: string;
  label: string;
  icon: string;
  tagline: string;
  steps: Step[];
  note?: string;
};

// What the interactive provisioning wizard asks after the command installs the
// client. Identical for every Linux target (Pi / UNOQ / ESP32 host), so it's
// rendered once, below the steps.
const WIZARD_PROMPTS: { q: string; a: string }[] = [
  { q: "Backend host", a: INSTALL_HOST },
  { q: "Email & password", a: "your Petal login — the password stays hidden and is never saved" },
  { q: "Project", a: "pick this project from the list" },
  { q: "Device key", a: 'choose "Create a new device key" (option 1)' },
];

// Installation targets for the guide. Distinct from the register device_type
// list — these describe how each hardware class is brought online.
const CONNECT_TYPES: ConnectType[] = [
  {
    id: "raspberry_pi",
    label: "Raspberry Pi",
    icon: "🍓",
    tagline: "Pi 3 / 4 / 5 · Zero 2 W",
    steps: [
      {
        title: "Flash Raspberry Pi OS (64-bit) to an SD card",
        detail: "Use Raspberry Pi Imager. In its ⚙ settings, enable SSH and enter your Wi-Fi name and password so the Pi joins your network on first boot.",
      },
      {
        title: "Put the Pi on the same network as this backend",
        detail: `Wi-Fi or Ethernet. The Pi must be able to reach ${INSTALL_HOST} — use the LAN IP, never localhost.`,
      },
      {
        title: "Open a terminal on the Pi",
        detail: "Attach a keyboard and screen, or SSH in from your computer:",
        code: "ssh pi@<your-pi-ip>",
      },
      {
        title: "Run the install command",
        detail: "Copy the command below, paste it into the Pi's terminal, and press Enter. It needs sudo, so enter the Pi's password if asked.",
      },
      {
        title: "Complete the setup wizard",
        detail: "The installer then asks a few questions (see below) and writes the device's config.",
      },
      {
        title: "Watch it come online",
        detail: "Within a few seconds the Pi appears under Live status and on the Devices page — no refresh needed.",
      },
    ],
  },
  {
    id: "unoq",
    label: "UNOQ",
    icon: "🧠",
    tagline: "Petal UnoQ board",
    note:
      "The UNOQ is a Linux board, so it runs the client directly — same flow as the Pi. " +
      "(It can also run the on-device inference runtime separately; that's a different service from the one this installs.)",
    steps: [
      {
        title: "Power on the UNOQ and connect it to your network",
        detail: `Wi-Fi or Ethernet on the same network as ${INSTALL_HOST}.`,
      },
      {
        title: "Open a terminal on the UNOQ",
        detail: "SSH in from your computer, or attach a keyboard and display.",
      },
      {
        title: "Run the install command",
        detail: "Paste the command below and press Enter. It needs sudo, so enter the board's password if asked.",
      },
      {
        title: "Complete the setup wizard",
        detail: "The installer then asks a few questions (see below) and writes the device's config.",
      },
      {
        title: "Watch it come online",
        detail: "Within a few seconds the UNOQ appears under Live status and on the Devices page — no refresh needed.",
      },
    ],
  },
  {
    id: "esp32",
    label: "ESP32",
    icon: "📟",
    tagline: "ESP32 / Arduino MCU",
    note:
      "An ESP32 is a microcontroller — it can't run the Linux client itself. Run the command on a " +
      "companion Linux host (a Pi or PC); the ESP32 plugs into that host over USB and streams samples " +
      "through the forwarder.",
    steps: [
      {
        title: "Connect the ESP32 to a Linux host over USB",
        detail: "Flash your ESP32 firmware, then plug it into a Pi or Linux PC that is on the same network as this backend.",
      },
      {
        title: "Open a terminal on the host",
        detail: "Not on the ESP32 — on the Pi/PC it's plugged into.",
      },
      {
        title: "Run the install command on the host",
        detail: "Paste the command below and press Enter. It needs sudo.",
      },
      {
        title: "Complete the setup wizard",
        detail: "The installer then asks a few questions (see below) and writes the host's config.",
      },
      {
        title: "Stream ESP32 samples with the forwarder",
        detail: "From the install directory, run the serial forwarder pointed at the ESP32's port:",
        code: "python forwarder.py --port /dev/ttyUSB0 --axes accX,accY,accZ --label wave",
      },
      {
        title: "Watch the host come online",
        detail: "The host device appears under Live status and on the Devices page — no refresh needed.",
      },
    ],
  },
];

// ─── Package version badge ────────────────────────────────────────────────────
function PackageVersion() {
  const [manifest, setManifest] = useState<DeviceClientManifest | null>(null);
  const [state, setState] = useState<"loading" | "ready" | "none">("loading");

  useEffect(() => {
    let alive = true;
    deviceClientApi
      .latest()
      .then(({ data }) => {
        if (!alive) return;
        setManifest(data);
        setState("ready");
      })
      .catch(() => {
        // 404 = nothing published yet; any error → show a neutral fallback.
        if (alive) setState("none");
      });
    return () => { alive = false; };
  }, []);

  const ready = state === "ready";
  return (
    <span
      className="inline-flex items-center gap-1.5 text-xs px-3 py-1.5 rounded-full font-medium"
      style={{
        color: ready ? "var(--app-text)" : "var(--app-text-muted)",
        background: ready
          ? "color-mix(in srgb, var(--app-brand, #6366f1) 12%, transparent)"
          : "var(--app-surface-2)",
        border: `1px solid ${ready ? "color-mix(in srgb, var(--app-brand, #6366f1) 45%, transparent)" : "var(--app-border)"}`,
      }}
      title="The device-client version this command installs"
    >
      <Package size={13} style={{ color: ready ? BRAND : "var(--app-text-soft)" }} />
      {state === "loading" && <Loader2 size={11} className="animate-spin" />}
      {state === "loading" && "Checking latest…"}
      {state === "ready" && <>Installs <span className="font-mono">v{manifest?.version?.replace(/^v/, "")}</span></>}
      {state === "none" && "No package published yet"}
    </span>
  );
}

// ─── Live status: watch the new device flip online ───────────────────────────
function LiveStatus({ projectId }: { projectId: string }) {
  const [devices, setDevices] = useState<Device[]>([]);
  const [live, setLive] = useState<Record<string, Partial<Device>>>({});
  const loadedRef = useRef(false);

  async function loadDevices() {
    try {
      const { data } = await devicesApi.list(projectId);
      setDevices(data);
      loadedRef.current = true;
    } catch { /* transient — the next poll retries */ }
  }

  function handleEvent(e: StudioEvent) {
    // A device connecting/disconnecting changes REST-visible presence — refetch
    // so last_seen / is_online stay accurate, then apply the live override.
    if (e.type === "device.connected" || e.type === "device.disconnected") {
      loadDevices();
    }
    setLive(prev => applyStudioEvent(prev, e));
  }

  const { connected: wsConnected } = useStudioWS(projectId, handleEvent);

  useEffect(() => {
    loadDevices();
    // Heartbeat presence is REST-only (no WS event), so poll while this page is
    // open to catch heartbeat-only devices quickly — this is the "no manual
    // refresh" payoff working even without a live WS session.
    const interval = setInterval(loadDevices, 6_000);
    return () => clearInterval(interval);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [projectId]);

  const merged = useMemo(
    () => devices.map(d => ({ ...d, ...live[d.device_id] })),
    [devices, live],
  );
  const online = merged.filter(d => d.is_online);
  const offline = merged.filter(d => !d.is_online);

  return (
    <div className="card">
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-2">
          <span className="relative flex h-2 w-2">
            {online.length > 0 && (
              <span className="absolute inline-flex h-full w-full rounded-full bg-green-400 opacity-75 animate-ping" />
            )}
            <span
              className="relative inline-flex h-2 w-2 rounded-full"
              style={{ background: online.length > 0 ? "#4ade80" : "var(--app-text-soft)" }}
            />
          </span>
          <h3 className="text-sm font-semibold" style={{ color: "var(--app-text)" }}>Live status</h3>
        </div>
        <span
          className="text-[11px] px-2 py-0.5 rounded-full"
          style={{ color: "var(--app-text-soft)", background: "var(--app-surface-2)" }}
        >
          {wsConnected ? "live" : "reconnecting…"}
        </span>
      </div>

      {online.length === 0 ? (
        <div
          className="flex items-center gap-3 rounded-xl px-4 py-4"
          style={{ background: "var(--app-bg)", border: "1px dashed var(--app-border-strong)" }}
        >
          <Loader2 size={18} className="animate-spin" style={{ color: BRAND }} />
          <div>
            <p className="text-sm font-medium" style={{ color: "var(--app-text)" }}>Waiting for your device…</p>
            <p className="text-xs mt-0.5" style={{ color: "var(--app-text-soft)" }}>
              Run the command — it appears here automatically.
            </p>
          </div>
        </div>
      ) : (
        <div className="space-y-2">
          <p className="text-xs font-medium text-green-400 mb-2">
            🎉 {online.length} device{online.length !== 1 ? "s" : ""} online
          </p>
          {online.map(d => (
            <div
              key={d.id}
              className="flex items-center gap-2 rounded-lg px-3 py-2"
              style={{ background: "color-mix(in srgb, #22c55e 10%, transparent)", border: "1px solid color-mix(in srgb, #22c55e 25%, transparent)" }}
            >
              <Wifi size={14} className="text-green-400 flex-shrink-0" />
              <span className="text-sm truncate flex-1" style={{ color: "var(--app-text)" }}>{d.name || d.device_id}</span>
              <span className="text-[11px] font-mono truncate" style={{ color: "var(--app-text-soft)" }}>{d.device_id}</span>
            </div>
          ))}
          <Link href="/dashboard/devices" className="btn-secondary inline-flex mt-1" style={{ fontSize: "12px" }}>
            View all devices
          </Link>
        </div>
      )}

      {/* Offline devices, shown quietly for context. */}
      {offline.length > 0 && (
        <div className="mt-4 pt-3 space-y-1.5" style={{ borderTop: "1px solid var(--app-border)" }}>
          <p className="text-[11px] uppercase tracking-wide mb-1" style={{ color: "var(--app-text-soft)" }}>Offline</p>
          {offline.map(d => (
            <div key={d.id} className="flex items-center gap-2 text-xs" style={{ color: "var(--app-text-soft)" }}>
              <WifiOff size={12} /> <span className="truncate">{d.name || d.device_id}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

// ─── Small building blocks ────────────────────────────────────────────────────
function SectionTitle({ children }: { children: ReactNode }) {
  return (
    <h3 className="text-sm font-semibold mb-4" style={{ color: "var(--app-text)" }}>{children}</h3>
  );
}

// ─── Page ─────────────────────────────────────────────────────────────────────
export default function ConnectDevicePage() {
  const { activeProject } = useAppStore();
  const [selected, setSelected] = useState<string>("raspberry_pi");
  const [copied, setCopied] = useState(false);

  const type = CONNECT_TYPES.find(t => t.id === selected)!;

  function copyCommand() {
    navigator.clipboard.writeText(INSTALL_COMMAND);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
    toast.success("Command copied");
  }

  return (
    <div className="max-w-5xl mx-auto pb-10">
      {/* Header */}
      <div className="mb-6">
        <Link
          href="/dashboard/devices"
          className="inline-flex items-center gap-1.5 text-xs mb-4 transition-colors"
          style={{ color: "var(--app-text-soft)" }}
        >
          <ArrowLeft size={13} /> Back to devices
        </Link>
        <div className="flex items-start justify-between gap-4 flex-wrap">
          <div>
            <h1 className="text-2xl font-semibold tracking-tight" style={{ color: "var(--app-text)" }}>
              Connect a device
            </h1>
            <p className="text-sm mt-1.5" style={{ color: "var(--app-text-muted)" }}>
              One command installs the client, provisions it, and brings your device online.
            </p>
          </div>
          <PackageVersion />
        </div>
        {activeProject?.project_type === "motion" && (
          <p
            className="text-xs mt-3 rounded-lg px-3 py-2 inline-flex items-center gap-1.5"
            style={{
              color: "color-mix(in srgb, var(--app-brand, #6366f1) 85%, var(--app-text))",
              background: "color-mix(in srgb, var(--app-brand, #6366f1) 8%, transparent)",
              border: "1px solid color-mix(in srgb, var(--app-brand, #6366f1) 25%, transparent)",
            }}
          >
            Connecting a Motion board over USB? Use the{" "}
            <Link href="/dashboard/data/dataset" className="underline font-medium">
              Data Labeling
            </Link>{" "}
            page — the guide below is for Raspberry Pi, UNOQ and ESP32 install commands.
          </p>
        )}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-[minmax(0,1fr)_320px] gap-6 items-start">
        {/* ── Main column ───────────────────────────────────────────────── */}
        <div className="space-y-6 min-w-0">
          {/* Device-type selector */}
          <div>
            <p className="label mb-2.5">Choose your device</p>
            <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
              {CONNECT_TYPES.map(t => {
                const active = t.id === selected;
                return (
                  <button
                    key={t.id}
                    onClick={() => setSelected(t.id)}
                    className="group relative rounded-xl p-4 text-left transition-all duration-150"
                    style={{
                      background: active
                        ? "color-mix(in srgb, var(--app-brand, #6366f1) 8%, var(--app-surface))"
                        : "var(--app-surface)",
                      border: `1px solid ${active ? BRAND : "var(--app-border)"}`,
                      boxShadow: active
                        ? "0 0 0 3px color-mix(in srgb, var(--app-brand, #6366f1) 20%, transparent)"
                        : "var(--app-shadow)",
                    }}
                    aria-pressed={active}
                  >
                    <div className="flex items-start gap-3">
                      <span
                        className="flex-shrink-0 grid place-items-center w-10 h-10 rounded-lg text-xl"
                        style={{ background: "var(--app-surface-2)", border: "1px solid var(--app-border)" }}
                      >
                        {t.icon}
                      </span>
                      <div className="min-w-0 flex-1">
                        <p className="text-sm font-semibold" style={{ color: "var(--app-text)" }}>{t.label}</p>
                        <p className="text-[11px] truncate mt-0.5" style={{ color: "var(--app-text-soft)" }}>{t.tagline}</p>
                      </div>
                      {active && <CheckCircle size={16} style={{ color: BRAND }} className="flex-shrink-0" />}
                    </div>
                  </button>
                );
              })}
            </div>
          </div>

          {/* The one command — terminal chrome hero. */}
          <div>
              <div className="flex items-center gap-2 mb-2.5">
                <Terminal size={15} style={{ color: BRAND }} />
                <p className="label mb-0">Run this on your device</p>
              </div>
              <div
                className="rounded-xl overflow-hidden"
                style={{ border: "1px solid var(--app-border-strong)", boxShadow: "var(--app-shadow-lg)" }}
              >
                {/* window bar */}
                <div
                  className="flex items-center gap-2 px-4 py-2.5"
                  style={{ background: "var(--app-surface-2)", borderBottom: "1px solid var(--app-border)" }}
                >
                  <span className="flex gap-1.5">
                    <span className="w-3 h-3 rounded-full" style={{ background: "#ff5f56" }} />
                    <span className="w-3 h-3 rounded-full" style={{ background: "#ffbd2e" }} />
                    <span className="w-3 h-3 rounded-full" style={{ background: "#27c93f" }} />
                  </span>
                  <span className="text-[11px] font-mono ml-1" style={{ color: "var(--app-text-soft)" }}>
                    bash — your device
                  </span>
                  <button
                    onClick={copyCommand}
                    className="ml-auto inline-flex items-center gap-1.5 text-[11px] font-medium px-2.5 py-1 rounded-md transition-colors"
                    style={{
                      color: copied ? "#4ade80" : "var(--app-text-muted)",
                      background: "var(--app-bg)",
                      border: "1px solid var(--app-border)",
                    }}
                    aria-label="Copy install command"
                  >
                    {copied ? <CheckCircle size={13} /> : <Copy size={13} />}
                    {copied ? "Copied" : "Copy"}
                  </button>
                </div>
                {/* command body */}
                <div className="px-4 py-3.5 overflow-x-auto" style={{ background: "var(--app-bg)" }}>
                  <pre className="text-[13px] font-mono whitespace-pre" style={{ color: "var(--app-text)" }}>
                    <span style={{ color: "var(--app-text-soft)" }}>$ </span>{INSTALL_COMMAND}
                  </pre>
                </div>
              </div>
              <p className="flex items-center gap-1.5 text-[11px] mt-2" style={{ color: "var(--app-text-soft)" }}>
                <ShieldCheck size={12} className="text-green-400" />
                Verifies the download checksum before installing. Requires a Debian / Raspberry Pi OS host with sudo.
              </p>
          </div>

          {/* Before you begin */}
          <div className="card">
              <SectionTitle>Before you begin</SectionTitle>
              <ul className="space-y-2.5">
                {[
                  <>Your device is on the <span style={{ color: "var(--app-text)" }}>same network</span> as this backend and can reach <span className="font-mono" style={{ color: "var(--app-text)" }}>{INSTALL_HOST}</span> — use the LAN IP, never <span className="font-mono">localhost</span>.</>,
                  <>The device runs <span style={{ color: "var(--app-text)" }}>Debian / Raspberry Pi OS</span> and can use <span className="font-mono">sudo</span>.</>,
                  <>A device-client package is published — the badge above shows a version, not <span className="italic">“No package published yet.”</span></>,
                ].map((item, i) => (
                  <li key={i} className="flex gap-2.5 text-xs" style={{ color: "var(--app-text-muted)" }}>
                    <CheckCircle size={14} className="text-green-400 flex-shrink-0 mt-0.5" />
                    <span>{item}</span>
                  </li>
                ))}
              </ul>
          </div>

          {/* Per-type installation guide — connected stepper */}
          <div className="card">
            <div className="flex items-center gap-2 mb-1">
              <span className="text-lg">{type.icon}</span>
              <h3 className="text-sm font-semibold" style={{ color: "var(--app-text)" }}>
                {type.label} — installation guide
              </h3>
            </div>
            {type.note && (
              <p
                className="text-xs rounded-lg px-3 py-2.5 my-3"
                style={{
                  color: "color-mix(in srgb, #f59e0b 85%, var(--app-text))",
                  background: "color-mix(in srgb, #f59e0b 10%, transparent)",
                  border: "1px solid color-mix(in srgb, #f59e0b 30%, transparent)",
                }}
              >
                {type.note}
              </p>
            )}
            <ol className="mt-4">
              {type.steps.map((step, i) => {
                const last = i === type.steps.length - 1;
                return (
                  <li key={i} className={`relative flex gap-4 ${last ? "" : "pb-6"}`}>
                    {!last && (
                      <span
                        className="absolute top-8 bottom-0 w-px"
                        style={{ left: "15px", background: "var(--app-border)" }}
                        aria-hidden
                      />
                    )}
                    <span
                      className="relative z-10 flex-shrink-0 grid place-items-center w-8 h-8 rounded-full text-xs font-semibold"
                      style={{
                        background: "var(--app-surface-2)",
                        border: "1px solid var(--app-border-strong)",
                        color: "var(--app-text)",
                      }}
                    >
                      {i + 1}
                    </span>
                    <div className="min-w-0 flex-1 pt-1">
                      <p className="text-sm font-medium" style={{ color: "var(--app-text)" }}>{step.title}</p>
                      {step.detail && (
                        <p className="text-xs mt-1 leading-relaxed" style={{ color: "var(--app-text-muted)" }}>{step.detail}</p>
                      )}
                      {step.code && (
                        <pre
                          className="mt-2 rounded-md px-3 py-2 text-[11px] font-mono overflow-x-auto whitespace-pre"
                          style={{ background: "var(--app-bg)", border: "1px solid var(--app-border)", color: "var(--app-text)" }}
                        >
                          {step.code}
                        </pre>
                      )}
                    </div>
                  </li>
                );
              })}
            </ol>

            {/* Exactly what the interactive wizard asks — the step most people get
                stuck on. Spelled out so there are no surprises at the prompt. */}
            <div
                className="mt-5 rounded-xl p-4"
                style={{ background: "var(--app-bg)", border: "1px solid var(--app-border)" }}
              >
                <div className="flex items-center gap-2 mb-3">
                  <Terminal size={14} style={{ color: BRAND }} />
                  <p className="text-xs font-semibold" style={{ color: "var(--app-text)" }}>The setup wizard will ask you for</p>
                </div>
                <dl className="space-y-2">
                  {WIZARD_PROMPTS.map(p => (
                    <div key={p.q} className="flex gap-3 text-xs items-baseline">
                      <dt className="font-medium flex-shrink-0 w-28" style={{ color: "var(--app-text)" }}>{p.q}</dt>
                      <dd className="min-w-0" style={{ color: "var(--app-text-muted)" }}>
                        <span style={{ color: "var(--app-text-soft)" }}>→ </span>
                        <span className="break-words">{p.a}</span>
                      </dd>
                    </div>
                  ))}
                </dl>
                <p className="text-[11px] mt-3 pt-3" style={{ color: "var(--app-text-soft)", borderTop: "1px solid var(--app-border)" }}>
                  It writes <span className="font-mono">config.json</span> (locked to owner-only) and starts the
                  service. Re-running the command later is safe — it keeps your config.
                </p>
            </div>
          </div>
        </div>

        {/* ── Sticky rail: live status payoff ───────────────────────────── */}
        <aside className="lg:sticky lg:top-6 self-start space-y-4">
          {activeProject ? (
            <LiveStatus projectId={activeProject.id} />
          ) : (
            <div className="card">
              <SectionTitle>Live status</SectionTitle>
              <p className="text-sm" style={{ color: "var(--app-text-muted)" }}>
                Select a project to watch your device come online.
              </p>
            </div>
          )}
        </aside>
      </div>
    </div>
  );
}
