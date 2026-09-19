// Types for the Motion "Connect Device" (USB) flow — Dataset page only.
//
// Everything here is auto-discovered: there is no typed field anywhere in
// this flow (Device Name, ID, Board Type, Port, VID/PID, firmware, sensors
// all come from the connection attempt itself). Adding a third board is one
// entry in `boards.ts` plus one branch in the mock service — no UI change.

import type { LucideIcon } from "lucide-react";

/** Boards this flow currently recognizes. Extend here as support grows. */
export type SupportedBoardId = "arduino_uno_q" | "raspberry_pi";

export interface SupportedBoardMeta {
  id: SupportedBoardId;
  label: string;
  tagline: string;
  vendorId: string;
  icon: LucideIcon;
}

/** One USB-attached candidate found by a scan, already matched against a
 *  supported board profile — unrecognized USB devices never reach the UI. */
export interface DiscoveredDevice {
  scanId: string;
  boardId: SupportedBoardId;
  suggestedName: string;
  port: string;
  vendorId: string;
  productId: string;
}

export interface DeviceSensor {
  name: string;
  type: string;
  freqHz?: number;
  axes?: string[];
}

/** Everything the app knows about a device once a connection succeeds — the
 *  full auto-populated identity shown in DeviceInfoCard. */
export interface ConnectedDeviceInfo {
  deviceId: string;
  deviceName: string;
  boardId: SupportedBoardId;
  port: string;
  vendorId: string;
  productId: string;
  firmwareVersion: string | null;
  sensors: DeviceSensor[];
  connectedAt: string;
}

export type ConnectionPhase =
  | "scanning"
  | "device-list"
  | "connecting"
  | "connected"
  | "error";
