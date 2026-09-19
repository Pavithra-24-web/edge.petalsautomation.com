import type { Device, StudioEvent } from "@/types/devices";

/**
 * Pure event reducer for device state.
 * Kept in a dedicated utility so it can be imported by both the
 * devices page and unit tests without triggering Next.js's restriction
 * on extra named exports from page files.
 */
export function applyStudioEvent(
  state: Record<string, Partial<Device>>,
  e: StudioEvent,
): Record<string, Partial<Device>> {
  if (!e.device_id) {
    if (e.type === "snapshot") {
      const next = { ...state };
      const snapshotDevices = (e.payload?.devices ?? (e as any).devices ?? []) as Array<{
        device_id: string;
        mode?: Device["mode"];
        connected?: boolean;
      }>;
      for (const d of snapshotDevices) {
        next[d.device_id] = { ...next[d.device_id], mode: d.mode, is_online: d.connected };
      }
      return next;
    }
    return state;
  }
  const id = e.device_id;
  switch (e.type) {
    case "device.connected":
      return { ...state, [id]: { ...state[id], mode: "idle", is_online: true } };
    case "device.disconnected":
      return { ...state, [id]: { ...state[id], is_online: false, mode: undefined } };
    case "device.mode_changed":
      return { ...state, [id]: { ...state[id], mode: e.payload?.mode } };
    default:
      return state;
  }
}
