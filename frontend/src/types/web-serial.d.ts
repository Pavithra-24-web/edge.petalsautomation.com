// Minimal ambient types for the Web Serial API — not yet part of TypeScript's
// bundled DOM lib. Covers only what `motion/device-connect/webSerialTransport.ts`
// uses. Spec: https://wicg.github.io/serial/
//
// Deliberately NOT a full port of @types/w3c-web-serial (not installed) — this
// repo has no other Web Serial consumer, so a minimal, hand-maintained surface
// is easier to keep honest than a third-party package pulling in globals we
// don't use.

interface SerialPortInfo {
  usbVendorId?: number;
  usbProductId?: number;
}

interface SerialOptions {
  baudRate: number;
  dataBits?: 7 | 8;
  stopBits?: 1 | 2;
  parity?: "none" | "even" | "odd";
  bufferSize?: number;
  flowControl?: "none" | "hardware";
}

interface SerialPort extends EventTarget {
  readonly readable: ReadableStream<Uint8Array> | null;
  readonly writable: WritableStream<Uint8Array> | null;
  open(options: SerialOptions): Promise<void>;
  close(): Promise<void>;
  forget?(): Promise<void>;
  getInfo(): SerialPortInfo;
}

interface SerialPortFilter {
  usbVendorId?: number;
  usbProductId?: number;
}

interface SerialPortRequestOptions {
  filters?: SerialPortFilter[];
}

interface Serial extends EventTarget {
  getPorts(): Promise<SerialPort[]>;
  requestPort(options?: SerialPortRequestOptions): Promise<SerialPort>;
}

interface Navigator {
  readonly serial?: Serial;
}
