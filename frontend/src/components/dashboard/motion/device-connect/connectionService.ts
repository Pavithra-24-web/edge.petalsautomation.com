// Connection service — the seam between the "Connect Device" UI and USB.
//
// `MotionConnectionService` is the sole owner of `navigator.serial` and the
// device WebSocket, per motion_phase1.md's Architecture Principle 2 — no
// other module may talk to either directly. Two implementations satisfy it:
//   - `createWebSerialMotionConnectionService` (webSerialTransport.ts,
//     Batch 2) — the real transport, used everywhere the app runs.
//   - `createMockMotionConnectionService` (below) — a fixed, non-cycling
//     mock kept for isolated UI work (Storybook-style, or a browser with no
//     Web Serial support) so the dialog's empty / single / multiple device
//     states stay exercisable without hardware.

import { SUPPORTED_BOARDS } from "./boards";
import type { ConnectedDeviceInfo, DeviceSensor, DiscoveredDevice, SupportedBoardId } from "./types";

export interface MotionConnectionService {
  /** Enumerate USB-attached devices that match a supported board profile
   *  and have already been authorized in this browser (no permission
   *  prompt). */
  scanDevices(): Promise<DiscoveredDevice[]>;
  /** Prompt the browser's native device picker to authorize a device never
   *  granted access before. Resolves `null` if the user dismisses it —
   *  that is not an error. */
  requestDevice(): Promise<DiscoveredDevice | null>;
  /** Open a connection to a previously discovered device and read its identity. */
  connect(device: DiscoveredDevice): Promise<ConnectedDeviceInfo>;
  /** Close the active connection. */
  disconnect(deviceId: string): Promise<void>;
  /** Re-read a connected device's identity (name, board, firmware, port). */
  getDeviceInfo(deviceId: string): Promise<ConnectedDeviceInfo | null>;
  /** Re-read a connected device's reported sensors. */
  getSensors(deviceId: string): Promise<DeviceSensor[]>;
}

const MOCK_PROFILES: Record<
  SupportedBoardId,
  { port: string; productId: string; firmwareVersion: string; sensors: DeviceSensor[] }
> = {
  arduino_uno_q: {
    port: "COM5",
    productId: "0069",
    firmwareVersion: "1.4.2",
    sensors: [
      { name: "IMU", type: "accelerometer+gyroscope", freqHz: 104, axes: ["x", "y", "z"] },
      { name: "Magnetometer", type: "magnetometer", freqHz: 20, axes: ["x", "y", "z"] },
    ],
  },
  raspberry_pi: {
    port: "/dev/ttyACM0",
    productId: "000A",
    firmwareVersion: "0.9.1",
    sensors: [
      { name: "IMU", type: "accelerometer+gyroscope", freqHz: 100, axes: ["x", "y", "z"] },
    ],
  },
};

const delay = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));

const connected = new Map<string, ConnectedDeviceInfo>();

/**
 * Mock implementation. `connectedBoardIds` is the fixed set of boards this
 * instance pretends are physically plugged in — every `scanDevices()` call
 * returns exactly that set (stable across rescans, matching how a real
 * OS-level device enumeration behaves when nothing was plugged/unplugged),
 * not a rotating fixture. Defaults to both supported boards being present;
 * pass `[]` to exercise the empty state, or one id to exercise the
 * single-device state.
 */
export function createMockMotionConnectionService(
  connectedBoardIds: SupportedBoardId[] = ["arduino_uno_q", "raspberry_pi"],
): MotionConnectionService {
  return {
    async requestDevice() {
      // The mock's "physically connected" set is fixed — everything it can
      // ever return already comes back from scanDevices(). A native picker
      // has nothing further to authorize, so this mirrors "user found
      // nothing new" rather than fabricating a device scanDevices() didn't
      // already know about.
      return null;
    },

    async scanDevices() {
      await delay(900);
      return connectedBoardIds.map((boardId) => {
        const board = SUPPORTED_BOARDS[boardId];
        const profile = MOCK_PROFILES[boardId];
        return {
          scanId: boardId,
          boardId,
          suggestedName: board.label,
          port: profile.port,
          vendorId: board.vendorId,
          productId: profile.productId,
        } satisfies DiscoveredDevice;
      });
    },

    async connect(device) {
      await delay(1400);
      const profile = MOCK_PROFILES[device.boardId];
      const info: ConnectedDeviceInfo = {
        deviceId: `${device.boardId}-${device.port}`,
        deviceName: device.suggestedName,
        boardId: device.boardId,
        port: device.port,
        vendorId: device.vendorId,
        productId: device.productId,
        firmwareVersion: profile.firmwareVersion,
        sensors: profile.sensors,
        connectedAt: new Date().toISOString(),
      };
      connected.set(info.deviceId, info);
      return info;
    },

    async disconnect(deviceId) {
      await delay(300);
      connected.delete(deviceId);
    },

    async getDeviceInfo(deviceId) {
      return connected.get(deviceId) ?? null;
    },

    async getSensors(deviceId) {
      return connected.get(deviceId)?.sensors ?? [];
    },
  };
}
