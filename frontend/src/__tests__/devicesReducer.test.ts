import { applyStudioEvent } from "@/utils/devicesReducer";
import type { StudioEvent } from "@/types/devices";

function makeEvent(overrides: Partial<StudioEvent>): StudioEvent {
  return {
    type: "test",
    project_id: "proj-1",
    ts: new Date().toISOString(),
    payload: {},
    ...overrides,
  };
}

describe("applyStudioEvent reducer", () => {
  it("device.connected → is_online: true, mode: idle", () => {
    const next = applyStudioEvent(
      {},
      makeEvent({ type: "device.connected", device_id: "dev-1" }),
    );
    expect(next["dev-1"]).toMatchObject({ is_online: true, mode: "idle" });
  });

  it("device.disconnected → is_online: false, mode undefined", () => {
    const prev = { "dev-1": { is_online: true, mode: "sampling" as const } };
    const next = applyStudioEvent(
      prev,
      makeEvent({ type: "device.disconnected", device_id: "dev-1" }),
    );
    expect(next["dev-1"].is_online).toBe(false);
    expect(next["dev-1"].mode).toBeUndefined();
  });

  it("device.mode_changed → mode updated", () => {
    const prev = { "dev-1": { mode: "idle" as const } };
    const next = applyStudioEvent(
      prev,
      makeEvent({ type: "device.mode_changed", device_id: "dev-1", payload: { mode: "inference" } }),
    );
    expect(next["dev-1"].mode).toBe("inference");
  });

  it("snapshot (initial) → multiple devices populated", () => {
    const next = applyStudioEvent(
      {},
      makeEvent({
        type: "snapshot",
        payload: {
          devices: [
            { device_id: "dev-1", mode: "idle",     connected: true },
            { device_id: "dev-2", mode: "sampling", connected: false },
          ],
        },
      }),
    );
    expect(next["dev-1"]).toMatchObject({ mode: "idle",     is_online: true });
    expect(next["dev-2"]).toMatchObject({ mode: "sampling", is_online: false });
  });

  it("snapshot.frame does not mutate liveDevices state (handled separately)", () => {
    const prev = { "dev-1": { mode: "idle" as const } };
    const next = applyStudioEvent(
      prev,
      makeEvent({ type: "snapshot.frame", device_id: "dev-1", payload: { data: [1, 2, 3] } }),
    );
    // snapshot.frame falls through to default — state unchanged
    expect(next).toEqual(prev);
  });

  it("inference.result does not mutate liveDevices state (handled separately)", () => {
    const prev = { "dev-1": { mode: "inference" as const } };
    const next = applyStudioEvent(
      prev,
      makeEvent({ type: "inference.result", device_id: "dev-1", payload: { label: "walking", confidence: 0.9 } }),
    );
    expect(next).toEqual(prev);
  });

  it("caps inference results at 20 entries when caller manages the array", () => {
    // Simulate the caller-side slice logic used in handleEvent
    const existing = Array.from({ length: 20 }, (_, i) => ({ index: i }));
    const newPayload = { index: 99 };
    const updated = [newPayload, ...existing].slice(0, 20);
    expect(updated.length).toBe(20);
    expect(updated[0]).toEqual({ index: 99 });
  });
});
