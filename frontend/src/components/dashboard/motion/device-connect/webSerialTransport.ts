// The real MotionConnectionService transport — Batch 2 of motion_phase1.md.
//
// Owns every `navigator.serial` call and every device-WebSocket message this
// application sends, per Architecture Principle 2 in motion_phase1.md: no
// other module may talk to either directly. `USBConnectionDialog` and its
// children stay exactly as Batch 1 built them — they render whatever this
// service returns and have no idea a real port is behind it now.
//
// Known, documented platform limits (Web Serial itself, not this
// implementation):
//   - No OS device path (COM5, /dev/ttyACM0) is exposed by the spec —
//     `SerialPort.getInfo()` returns only a USB vendor/product id pair. The
//     `port` field is therefore a fixed, honest label, not a fabricated path.
//   - No hardware serial number is exposed either, so a device's identity
//     across a page reload is *reconstructed*, not read — see
//     `resolveStableDeviceId` below. Two identical boards can, in the worst
//     case, swap identities across a reload; a single board of a given model
//     is stable. This is the real-hardware equivalent of Batch 1's identity
//     banner note: an honest degraded state, not a silent wrong answer.

import { devicesApi } from "@/utils/api";
import { SUPPORTED_BOARDS, SUPPORTED_BOARD_ORDER } from "./boards";
import type { MotionConnectionService } from "./connectionService";
import type {
  ConnectedDeviceInfo,
  DeviceSensor,
  DiscoveredDevice,
  SupportedBoardId,
} from "./types";

const DEFAULT_BAUD_RATE = 115200;
const IDENTITY_WINDOW_MS = 2500;
const HELLO_TIMEOUT_MS = 8000;
const HEARTBEAT_INTERVAL_MS = 20000;
const STORAGE_KEY = "pe_motion_serial_device_ids";

export function isWebSerialSupported(): boolean {
  return typeof navigator !== "undefined" && "serial" in navigator && !!navigator.serial;
}

function randomId(): string {
  if (typeof crypto !== "undefined" && "randomUUID" in crypto) return crypto.randomUUID();
  return `id-${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

// ─── Board matching ────────────────────────────────────────────────────────

const VENDOR_TO_BOARD: Record<number, SupportedBoardId> = Object.fromEntries(
  SUPPORTED_BOARD_ORDER.map((id) => [parseInt(SUPPORTED_BOARDS[id].vendorId, 16), id]),
) as Record<number, SupportedBoardId>;

function matchBoard(vendorId?: number): SupportedBoardId | null {
  if (vendorId == null) return null;
  return VENDOR_TO_BOARD[vendorId] ?? null;
}

function hex(n: number | undefined | null, width = 4): string {
  if (n == null) return "?".repeat(width);
  return n.toString(16).toUpperCase().padStart(width, "0");
}

export function requestPortFilters(): { usbVendorId: number }[] {
  return SUPPORTED_BOARD_ORDER.map((id) => ({ usbVendorId: parseInt(SUPPORTED_BOARDS[id].vendorId, 16) }));
}

// ─── Stable device identity across reloads ────────────────────────────────
//
// Web Serial exposes no hardware serial number, so there is no ground truth
// to read. `getPorts()` does return the same SerialPort *object* for the
// same authorized port within one page session, so an in-memory WeakMap is
// enough for that. Surviving a reload needs something persisted — this
// reclaims a previously-issued id for the same (vendorId, productId) pair
// when one is unclaimed this session, and only mints a new id when none is
// available. See the module doc comment above for the honest limitation.

interface StoredIdEntry {
  vendorId: number;
  productId: number | null;
  deviceId: string;
}

function loadStoredIds(): StoredIdEntry[] {
  if (typeof window === "undefined") return [];
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY);
    return raw ? (JSON.parse(raw) as StoredIdEntry[]) : [];
  } catch {
    return [];
  }
}

function saveStoredIds(entries: StoredIdEntry[]): void {
  if (typeof window === "undefined") return;
  try {
    window.localStorage.setItem(STORAGE_KEY, JSON.stringify(entries));
  } catch {
    // Storage unavailable (private mode, quota) — the id just won't survive
    // a reload; the connection itself still works.
  }
}

const stableIdByPort = new WeakMap<SerialPort, string>();
const claimedIdsThisSession = new Set<string>();

function resolveStableDeviceId(port: SerialPort, info: SerialPortInfo): string {
  const cached = stableIdByPort.get(port);
  if (cached) return cached;

  const vendorId = info.usbVendorId ?? 0;
  const productId = info.usbProductId ?? null;
  const entries = loadStoredIds();
  const reusable = entries.find(
    (e) => e.vendorId === vendorId && e.productId === productId && !claimedIdsThisSession.has(e.deviceId),
  );
  const deviceId = reusable?.deviceId ?? `web-serial-${hex(vendorId)}-${hex(productId ?? undefined)}-${randomId()}`;

  stableIdByPort.set(port, deviceId);
  claimedIdsThisSession.add(deviceId);
  if (!reusable) saveStoredIds([...entries, { vendorId, productId, deviceId }]);
  return deviceId;
}

// ─── Identity-banner probe ─────────────────────────────────────────────────
//
// Same wire convention Batch 1's retired Devices-page draft used: write
// `IDENTIFY\n`, read JSON lines for a short window, accept the first one
// with a `name` field. No firmware in this repository answers it yet, so
// this always falls back to the board's catalog label — an honest degraded
// state, not a blocking requirement.

interface IdentityBanner {
  name?: string;
  firmwareVersion?: string;
  sensors?: DeviceSensor[];
}

async function readIdentityBanner(port: SerialPort): Promise<IdentityBanner | null> {
  if (!port.writable || !port.readable) return null;

  const writer = port.writable.getWriter();
  try {
    await writer.write(new TextEncoder().encode("IDENTIFY\n"));
  } catch {
    return null;
  } finally {
    writer.releaseLock();
  }

  // TextDecoderStream's DOM-lib type declares `writable: WritableStream<BufferSource>`,
  // wider than the `Uint8Array` pipeThrough expects — a real mismatch only in
  // the type declarations (BufferSource legally includes Uint8Array), not at
  // runtime. Same cast the retired Devices-page draft needed for this exact pair.
  const textStream = port.readable.pipeThrough(
    new TextDecoderStream() as unknown as ReadableWritablePair<string, Uint8Array>,
  );
  const reader = textStream.getReader();
  const deadline = Date.now() + IDENTITY_WINDOW_MS;
  let buffer = "";
  try {
    while (Date.now() < deadline) {
      const remaining = deadline - Date.now();
      const timedOut = Symbol("timeout");
      const result = await Promise.race([
        reader.read(),
        new Promise<typeof timedOut>((resolve) => setTimeout(() => resolve(timedOut), remaining)),
      ]);
      if (result === timedOut) break;
      const { done, value } = result as ReadableStreamReadResult<string>;
      if (done) break;
      buffer += value ?? "";
      const newlineIdx = buffer.indexOf("\n");
      if (newlineIdx !== -1) {
        const line = buffer.slice(0, newlineIdx).trim();
        buffer = buffer.slice(newlineIdx + 1);
        if (!line) continue;
        try {
          const parsed = JSON.parse(line);
          if (parsed && typeof parsed === "object" && typeof parsed.name === "string") {
            return parsed as IdentityBanner;
          }
        } catch {
          // Not a JSON identity line — keep listening within the window.
        }
      }
    }
  } finally {
    try {
      await reader.cancel();
    } catch {
      // Port may already be gone (unplugged mid-probe) — nothing to clean up.
    }
    reader.releaseLock();
  }
  return null;
}

async function safeClosePort(port: SerialPort): Promise<void> {
  try {
    await port.close();
  } catch {
    // Already closed, or the device vanished — nothing more to do.
  }
}

// ─── Device WebSocket: hello / hello-ack ───────────────────────────────────

interface HelloAckResult {
  devicePk: string;
}

function openHelloSession(params: {
  deviceId: string;
  boardId: SupportedBoardId;
  apiKey: string;
  firmwareVersion: string | null;
  sensors: DeviceSensor[];
}): Promise<{ ws: WebSocket; ack: HelloAckResult }> {
  return new Promise((resolve, reject) => {
    let settled = false;
    const ws = new WebSocket(devicesApi.deviceWsUrl());

    const timeout = window.setTimeout(() => {
      if (settled) return;
      settled = true;
      ws.close();
      reject(new Error("Timed out waiting for the server to acknowledge the connection."));
    }, HELLO_TIMEOUT_MS);

    ws.addEventListener("open", () => {
      ws.send(
        JSON.stringify({
          type: "hello",
          version: "1",
          apiKey: params.apiKey,
          deviceId: params.deviceId,
          deviceType: params.boardId,
          connection: "serial",
          sensors: params.sensors,
          supportsSnapshotStreaming: false,
          firmwareVersion: params.firmwareVersion ?? undefined,
          protocolVersion: "1",
        }),
      );
    });

    ws.addEventListener("message", (event) => {
      if (settled) return;
      let msg: any;
      try {
        msg = JSON.parse(typeof event.data === "string" ? event.data : "");
      } catch {
        return;
      }
      if (msg?.type !== "hello-ack") return;
      settled = true;
      window.clearTimeout(timeout);
      if (msg.success) {
        resolve({ ws, ack: { devicePk: String(msg.id) } });
      } else {
        ws.close();
        reject(new Error(msg.error || "The server rejected the connection."));
      }
    });

    ws.addEventListener("error", () => {
      if (settled) return;
      settled = true;
      window.clearTimeout(timeout);
      reject(new Error("Could not reach the device connection endpoint."));
    });

    ws.addEventListener("close", () => {
      if (settled) return;
      settled = true;
      window.clearTimeout(timeout);
      reject(new Error("The connection closed before the server acknowledged it."));
    });
  });
}

// ─── Service factory ───────────────────────────────────────────────────────

interface Session {
  port: SerialPort;
  ws: WebSocket;
  heartbeatTimer: number;
  apiKey: string;
  devicePk: string;
  info: ConnectedDeviceInfo;
}

export function createWebSerialMotionConnectionService(projectId: string): MotionConnectionService {
  const portsByScanId = new Map<string, SerialPort>();
  const scanIdByPort = new WeakMap<SerialPort, string>();
  const sessions = new Map<string, Session>();
  let scanCounter = 0;

  function registerPort(port: SerialPort): DiscoveredDevice | null {
    const info = port.getInfo();
    const boardId = matchBoard(info.usbVendorId);
    if (!boardId) return null;

    let scanId = scanIdByPort.get(port);
    if (!scanId) {
      scanId = `serial-${++scanCounter}-${randomId()}`;
      scanIdByPort.set(port, scanId);
    }
    portsByScanId.set(scanId, port);

    return {
      scanId,
      boardId,
      suggestedName: SUPPORTED_BOARDS[boardId].label,
      // Web Serial does not expose an OS device path — see module doc comment.
      port: "USB serial port",
      vendorId: hex(info.usbVendorId),
      productId: hex(info.usbProductId),
    };
  }

  async function scanDevices(): Promise<DiscoveredDevice[]> {
    if (!isWebSerialSupported()) {
      throw new Error("Web Serial isn't supported in this browser.");
    }
    const ports = await navigator.serial!.getPorts();
    const discovered: DiscoveredDevice[] = [];
    for (const port of ports) {
      const device = registerPort(port);
      if (device) discovered.push(device);
    }
    return discovered;
  }

  async function requestDevice(): Promise<DiscoveredDevice | null> {
    if (!isWebSerialSupported()) {
      throw new Error("Web Serial isn't supported in this browser.");
    }
    let port: SerialPort;
    try {
      port = await navigator.serial!.requestPort({ filters: requestPortFilters() });
    } catch (err) {
      // The user dismissed the native port chooser — not a failure.
      if (err instanceof Error && err.name === "NotFoundError") return null;
      throw err;
    }
    return registerPort(port);
  }

  async function connect(device: DiscoveredDevice): Promise<ConnectedDeviceInfo> {
    const port = portsByScanId.get(device.scanId);
    if (!port) {
      throw new Error("This device is no longer available — rescan and try again.");
    }

    await port.open({ baudRate: DEFAULT_BAUD_RATE });

    const info = port.getInfo();
    const deviceId = resolveStableDeviceId(port, info);
    const banner = await readIdentityBanner(port).catch(() => null);
    const board = SUPPORTED_BOARDS[device.boardId];
    const deviceName = banner?.name?.trim() || board.label;
    const firmwareVersion = banner?.firmwareVersion ?? null;
    const sensors = banner?.sensors ?? [];

    let apiKey: string;
    try {
      const { data } = await devicesApi.createKey(projectId, `Web Serial – ${deviceId}`);
      apiKey = data.api_key;
    } catch {
      await safeClosePort(port);
      throw new Error("Could not provision a device key for this connection.");
    }

    let helloResult: { ws: WebSocket; ack: HelloAckResult };
    try {
      helloResult = await openHelloSession({
        deviceId,
        boardId: device.boardId,
        apiKey,
        firmwareVersion,
        sensors,
      });
    } catch (err) {
      await safeClosePort(port);
      throw err instanceof Error ? err : new Error("Could not establish the device connection.");
    }

    const connectedInfo: ConnectedDeviceInfo = {
      deviceId,
      deviceName,
      boardId: device.boardId,
      port: device.port,
      vendorId: device.vendorId,
      productId: device.productId,
      firmwareVersion,
      sensors,
      connectedAt: new Date().toISOString(),
    };

    const heartbeatTimer = window.setInterval(() => {
      devicesApi
        .heartbeat(helloResult.ack.devicePk, apiKey, { firmware_version: firmwareVersion ?? undefined })
        .catch(() => {
          // Best-effort — a missed beat self-heals on the next tick, and the
          // open WebSocket session already keeps the device marked online.
        });
    }, HEARTBEAT_INTERVAL_MS);

    const session: Session = {
      port,
      ws: helloResult.ws,
      heartbeatTimer,
      apiKey,
      devicePk: helloResult.ack.devicePk,
      info: connectedInfo,
    };
    sessions.set(deviceId, session);

    helloResult.ws.addEventListener("close", () => {
      window.clearInterval(session.heartbeatTimer);
    });

    return connectedInfo;
  }

  async function disconnect(deviceId: string): Promise<void> {
    const session = sessions.get(deviceId);
    if (!session) return;
    sessions.delete(deviceId);
    window.clearInterval(session.heartbeatTimer);
    try {
      session.ws.close();
    } catch {
      // Already closed.
    }
    await safeClosePort(session.port);
  }

  async function getDeviceInfo(deviceId: string): Promise<ConnectedDeviceInfo | null> {
    return sessions.get(deviceId)?.info ?? null;
  }

  async function getSensors(deviceId: string): Promise<DeviceSensor[]> {
    return sessions.get(deviceId)?.info.sensors ?? [];
  }

  return { scanDevices, requestDevice, connect, disconnect, getDeviceInfo, getSensors };
}
