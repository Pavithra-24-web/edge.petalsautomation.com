import {
  buildSignalChartData,
  toggleAxisVisibility,
  type SignalChartPoint,
} from "@/components/dashboard/labeling/motionSignalChart";

// Motion Phase 2 §5 AC1/AC2 (MotionSignalViewer, Batch 4):
//   AC1 — "Waveforms can be viewed per recording. Opening a motion sample
//          renders every axis."
//   AC2 — "Per-axis toggles show and hide individual channels."
//
// GET /samples/{id}/signal (Batch 3) already normalizes both a device
// recording's JSON envelope and a CSV upload to the same decoded
// {axes, values, frequency_hz, ...} shape before the frontend ever sees it,
// so these tests exercise that one normalized shape rather than duplicating
// the two source formats.

describe("buildSignalChartData (AC1 — every axis renders)", () => {
  test("every point carries a value for every axis, at every sample", () => {
    const signal = {
      values: [
        [1, 10, 100],
        [2, 20, 200],
        [3, 30, 300],
      ],
      frequency_hz: 100,
    };
    const points = buildSignalChartData(signal, 1000);
    expect(points).toHaveLength(3);
    for (const p of points) {
      expect(p).toHaveProperty("axis0");
      expect(p).toHaveProperty("axis1");
      expect(p).toHaveProperty("axis2");
    }
    expect(points[1].axis0).toBe(2);
    expect(points[1].axis1).toBe(20);
    expect(points[1].axis2).toBe(200);
  });

  test("derives elapsed time in ms from a positive frequency_hz", () => {
    const signal = { values: [[0], [1], [2], [3]], frequency_hz: 100 };
    const points = buildSignalChartData(signal, 1000);
    // 100 Hz -> 10ms per sample.
    expect(points.map((p: SignalChartPoint) => p.t)).toEqual([0, 10, 20, 30]);
  });

  test("falls back to the raw sample index when frequency_hz is not positive", () => {
    const signal = { values: [[0], [1], [2]], frequency_hz: 0 };
    const points = buildSignalChartData(signal, 1000);
    expect(points.map((p: SignalChartPoint) => p.t)).toEqual([0, 1, 2]);
  });

  test("decimates to at most maxPoints on a long recording, keeping every axis", () => {
    const rowCount = 10_000;
    const values = Array.from({ length: rowCount }, (_, i) => [i, -i]);
    const points = buildSignalChartData({ values, frequency_hz: 100 }, 500);
    expect(points.length).toBeLessThanOrEqual(500);
    expect(points.length).toBeGreaterThan(0);
    for (const p of points) {
      expect(p).toHaveProperty("axis0");
      expect(p).toHaveProperty("axis1");
    }
    // First sample is always retained so the waveform's start is never cropped.
    expect(points[0].t).toBe(0);
  });

  test("returns an empty array for a recording with no rows", () => {
    expect(buildSignalChartData({ values: [], frequency_hz: 100 }, 500)).toEqual([]);
  });

  test("single-axis recordings (mono channel) still produce a chartable series", () => {
    const points = buildSignalChartData({ values: [[5], [6], [7]], frequency_hz: 50 }, 1000);
    expect(points).toHaveLength(3);
    expect(points.every(p => typeof p.axis0 === "number")).toBe(true);
    expect(points.some(p => "axis1" in p)).toBe(false);
  });
});

describe("toggleAxisVisibility (AC2 — per-axis show/hide)", () => {
  test("hiding an axis adds its index to the hidden set", () => {
    const hidden = toggleAxisVisibility(new Set(), 1);
    expect(hidden.has(1)).toBe(true);
    expect(hidden.has(0)).toBe(false);
  });

  test("toggling an already-hidden axis shows it again", () => {
    const afterHide = toggleAxisVisibility(new Set(), 2);
    const afterShow = toggleAxisVisibility(afterHide, 2);
    expect(afterShow.has(2)).toBe(false);
  });

  test("other axes' visibility is unaffected by toggling one axis", () => {
    const hidden = toggleAxisVisibility(new Set([0, 2]), 1);
    expect(Array.from(hidden).sort()).toEqual([0, 1, 2]);
    const shown = toggleAxisVisibility(hidden, 0);
    expect(Array.from(shown).sort()).toEqual([1, 2]);
  });

  test("does not mutate the input set (pure function)", () => {
    const original = new Set([0]);
    const result = toggleAxisVisibility(original, 3);
    expect(original.has(3)).toBe(false);
    expect(result).not.toBe(original);
  });
});
