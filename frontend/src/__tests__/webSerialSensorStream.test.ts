// Batch 2b — device-side streaming protocol + browser relay wiring.
//
// Exercises `createWebSerialMotionConnectionService`'s new post-hello
// command listener and serial relay loop (webSerialTransport.ts): a mocked
// SerialPort stands in for the firmware side of the new "START-STREAM
// <freqHz>" / "STOP-STREAM" / "SAMPLE ax,ay,az" wire protocol
// (firmware/uno_q_identity_responder.ino), and a mocked WebSocket stands in
// for the device WS the backend already sends start-sensor-stream /
// stop-sensor-stream over (`stream_control.py`). No real hardware or
// browser Web Serial implementation is involved.

import { createWebSerialMotionConnectionService } from "@/components/dashboard/motion/device-connect/webSerialTransport";

jest.mock("@/utils/api", () => ({
  devicesApi: {
    createKey: jest.fn().mockResolvedValue({ data: { api_key: "test-api-key" } }),
    heartbeat: jest.fn().mockResolvedValue({}),
    deviceWsUrl: jest.fn().mockReturnValue("ws://test.invalid/ws/device"),
  },
}));

// ─── Fake WebSocket ─────────────────────────────────────────────────────────

class FakeWebSocket {
  static OPEN = 1;
  static CONNECTING = 0;
  static CLOSED = 3;
  static instances: FakeWebSocket[] = [];

  readyState = FakeWebSocket.CONNECTING;
  sent: string[] = [];
  private listeners: Record<string, Array<(ev: any) => void>> = {};

  constructor(public url: string) {
    FakeWebSocket.instances.push(this);
    queueMicrotask(() => {
      this.readyState = FakeWebSocket.OPEN;
      this.emit("open", {});
    });
  }

  addEventListener(type: string, cb: (ev: any) => void): void {
    (this.listeners[type] ??= []).push(cb);
  }

  removeEventListener(type: string, cb: (ev: any) => void): void {
    this.listeners[type] = (this.listeners[type] ?? []).filter((l) => l !== cb);
  }

  send(data: string): void {
    this.sent.push(data);
  }

  close(): void {
    this.readyState = FakeWebSocket.CLOSED;
    this.emit("close", {});
  }

  private emit(type: string, ev: any): void {
    (this.listeners[type] ?? []).forEach((cb) => cb(ev));
  }

  // Test helper — simulates a server -> device WS message.
  serverSend(payload: unknown): void {
    this.emit("message", { data: JSON.stringify(payload) });
  }

  sentMessages(): any[] {
    return this.sent.map((s) => JSON.parse(s));
  }
}

// ─── Fake SerialPort ────────────────────────────────────────────────────────
//
// `readable` is a getter, matching real Web Serial: cancelling a reader lets
// a later `port.readable` access hand back a fresh stream. The first access
// (the identity-banner probe in `connect()`) immediately enqueues a banner
// line so the probe resolves without waiting out its real 2.5s window.

class FakeSerialPort {
  writtenChunks: string[] = [];
  private controller: ReadableStreamDefaultController<Uint8Array> | null = null;
  private currentReadable: ReadableStream<Uint8Array> | null = null;
  private identityBannerServed = false;

  constructor(
    private readonly usbVendorId: number,
    private readonly usbProductId: number,
    private readonly identityBanner: string,
  ) {}

  getInfo() {
    return { usbVendorId: this.usbVendorId, usbProductId: this.usbProductId };
  }

  async open(): Promise<void> {}
  async close(): Promise<void> {}

  get writable(): WritableStream<Uint8Array> {
    const self = this;
    return new WritableStream<Uint8Array>({
      write(chunk) {
        self.writtenChunks.push(new TextDecoder().decode(chunk));
      },
    });
  }

  // Matches real Web Serial: `readable` is a stable stream until its reader
  // is cancelled/released, at which point the *next* access hands back a
  // fresh one. `readIdentityBanner` reads this property twice (a truthy
  // check, then the actual pipe) before cancelling — a new stream per
  // access would silently enqueue the banner onto the discarded first one.
  get readable(): ReadableStream<Uint8Array> {
    if (!this.currentReadable) {
      const self = this;
      const isFirstEver = !self.identityBannerServed;
      this.currentReadable = new ReadableStream<Uint8Array>({
        start(controller) {
          self.controller = controller;
          if (isFirstEver) {
            self.identityBannerServed = true;
            controller.enqueue(new TextEncoder().encode(self.identityBanner + "\n"));
          }
        },
        cancel() {
          self.controller = null;
          self.currentReadable = null;
        },
      });
    }
    return this.currentReadable;
  }

  // Test helpers — simulate firmware serial output / a dropped port.
  emitLine(text: string): void {
    this.controller?.enqueue(new TextEncoder().encode(text));
  }

  errorCurrentReader(): void {
    this.controller?.error(new Error("simulated serial disconnect"));
  }
}

const IDENTITY_BANNER = JSON.stringify({
  name: "Arduino UNO Q",
  firmwareVersion: "1.0.0-identity-responder",
  sensors: [{ name: "IMU", type: "accelerometer+gyroscope", freqHz: 104, axes: ["x", "y", "z"] }],
});

function flushAsync(iterations = 8): Promise<void> {
  return (async () => {
    for (let i = 0; i < iterations; i++) {
      await Promise.resolve();
      await new Promise<void>((resolve) => setImmediate(resolve));
    }
  })();
}

// Every session created via setupConnectedSession() is torn down in
// afterEach — otherwise each session's heartbeat `setInterval` (and, for a
// session left mid-reconnect, its backoff timer) keeps the Node process
// alive past the test run.
let openSessions: Array<{ service: ReturnType<typeof createWebSerialMotionConnectionService>; deviceId: string }> = [];

async function setupConnectedSession(): Promise<{
  service: ReturnType<typeof createWebSerialMotionConnectionService>;
  deviceId: string;
  port: FakeSerialPort;
  ws: FakeWebSocket;
}> {
  const port = new FakeSerialPort(0x2341, 0x0078, IDENTITY_BANNER);
  (global as any).navigator = {
    serial: {
      requestPort: jest.fn().mockResolvedValue(port),
      getPorts: jest.fn().mockResolvedValue([]),
    },
  };

  const service = createWebSerialMotionConnectionService("project-1");
  const discovered = await service.requestDevice();
  if (!discovered) throw new Error("expected requestDevice() to resolve a discovered device");

  const connectPromise = service.connect(discovered);
  await flushAsync();
  const ws = FakeWebSocket.instances[FakeWebSocket.instances.length - 1];
  ws.serverSend({ type: "hello-ack", success: true, id: "device-pk-1" });
  const info = await connectPromise;

  openSessions.push({ service, deviceId: info.deviceId });
  return { service, deviceId: info.deviceId, port, ws };
}

describe("webSerialTransport — Batch 2b sensor-stream relay", () => {
  beforeAll(() => {
    (global as any).window = globalThis;
    (global as any).window.localStorage = {
      getItem: () => null,
      setItem: () => {},
    };
  });

  beforeEach(() => {
    FakeWebSocket.instances = [];
    openSessions = [];
    (global as any).WebSocket = FakeWebSocket;
  });

  afterEach(async () => {
    await Promise.all(
      openSessions.map(({ service, deviceId }) => service.disconnect(deviceId).catch(() => {})),
    );
    jest.restoreAllMocks();
  });

  it("start-sensor-stream sends START-STREAM with the requested frequency and forwards parsed samples", async () => {
    const { port, ws } = await setupConnectedSession();

    ws.serverSend({ type: "start-sensor-stream", payload: { sensor: "IMU", frequency: 50 } });
    await flushAsync();

    expect(port.writtenChunks).toContain("START-STREAM 50\n");

    port.emitLine("SAMPLE 1.5,-2.25,9.8\n");
    await flushAsync();

    const frames = ws.sentMessages().filter((m) => m.type === "sensor-frame");
    expect(frames).toHaveLength(1);
    expect(frames[0].payload).toEqual({ sensor: "IMU", values: [1.5, -2.25, 9.8] });
  });

  it("falls back to a default frequency when the server omits one", async () => {
    const { port, ws } = await setupConnectedSession();

    ws.serverSend({ type: "start-sensor-stream", payload: {} });
    await flushAsync();

    const startCommand = port.writtenChunks.find((c) => c.startsWith("START-STREAM"));
    expect(startCommand).toBeDefined();
    // Falls back to the declared sensor's own freqHz (104) from the identity banner.
    expect(startCommand).toBe("START-STREAM 104\n");
  });

  it("skips a malformed sample line but keeps forwarding well-formed ones after it", async () => {
    const { port, ws } = await setupConnectedSession();

    ws.serverSend({ type: "start-sensor-stream", payload: { frequency: 50 } });
    await flushAsync();

    port.emitLine("SAMPLE not,a,number\n");
    port.emitLine("garbage line with no prefix\n");
    port.emitLine("SAMPLE 1,2,3\n");
    await flushAsync();

    const frames = ws.sentMessages().filter((m) => m.type === "sensor-frame");
    expect(frames).toHaveLength(1);
    expect(frames[0].payload.values).toEqual([1, 2, 3]);
  });

  it("stop-sensor-stream sends STOP-STREAM and further sample lines are not forwarded", async () => {
    const { port, ws } = await setupConnectedSession();

    ws.serverSend({ type: "start-sensor-stream", payload: { frequency: 50 } });
    await flushAsync();
    port.emitLine("SAMPLE 1,1,1\n");
    await flushAsync();
    expect(ws.sentMessages().filter((m) => m.type === "sensor-frame")).toHaveLength(1);

    ws.serverSend({ type: "stop-sensor-stream", payload: {} });
    await flushAsync();
    expect(port.writtenChunks).toContain("STOP-STREAM\n");

    port.emitLine("SAMPLE 9,9,9\n");
    await flushAsync();
    // Still just the one frame from before stopping.
    expect(ws.sentMessages().filter((m) => m.type === "sensor-frame")).toHaveLength(1);
  });

  it("stop-sensor-stream with nothing active is a no-op, not an error", async () => {
    const { port, ws } = await setupConnectedSession();

    ws.serverSend({ type: "stop-sensor-stream", payload: {} });
    await flushAsync();

    expect(port.writtenChunks.some((c) => c === "STOP-STREAM\n")).toBe(false);
    expect(ws.sentMessages().filter((m) => m.type === "sensor-frame")).toHaveLength(0);
  });

  it("a duplicate start-sensor-stream cancels the previous reader before starting a new one", async () => {
    const { port, ws } = await setupConnectedSession();

    ws.serverSend({ type: "start-sensor-stream", payload: { frequency: 50 } });
    await flushAsync();
    ws.serverSend({ type: "start-sensor-stream", payload: { frequency: 100 } });
    await flushAsync();

    // Cancel-then-restart: a STOP-STREAM was sent before the second START-STREAM.
    const stopIdx = port.writtenChunks.indexOf("STOP-STREAM\n");
    const secondStartIdx = port.writtenChunks.indexOf("START-STREAM 100\n");
    expect(stopIdx).toBeGreaterThanOrEqual(0);
    expect(secondStartIdx).toBeGreaterThan(stopIdx);

    // Only the newest reader is active — one frame per emitted line, no duplicates.
    port.emitLine("SAMPLE 5,5,5\n");
    await flushAsync();
    const frames = ws.sentMessages().filter((m) => m.type === "sensor-frame");
    expect(frames).toHaveLength(1);
    expect(frames[0].payload.values).toEqual([5, 5, 5]);
  });

  it("a serial read error stops the stream without crashing, and a fresh start still works", async () => {
    const { service, deviceId, port, ws } = await setupConnectedSession();

    ws.serverSend({ type: "start-sensor-stream", payload: { frequency: 50 } });
    await flushAsync();

    expect(() => port.errorCurrentReader()).not.toThrow();
    await flushAsync();

    // The transport is still usable: disconnect cleanly, and a fresh
    // start-sensor-stream on a new session still relays samples.
    await expect(service.disconnect(deviceId)).resolves.toBeUndefined();

    const { port: port2, ws: ws2 } = await setupConnectedSession();
    ws2.serverSend({ type: "start-sensor-stream", payload: { frequency: 50 } });
    await flushAsync();
    port2.emitLine("SAMPLE 2,2,2\n");
    await flushAsync();
    const frames = ws2.sentMessages().filter((m) => m.type === "sensor-frame");
    expect(frames).toHaveLength(1);
    expect(frames[0].payload.values).toEqual([2, 2, 2]);
  });
});
