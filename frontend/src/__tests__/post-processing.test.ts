/**
 * Tests for post-processing helpers and API client methods.
 */

// ── Helper function tests ─────────────────────────────────────────────────────

import {
  parsePreviewInput,
  mergeWithDefaults,
  buildPreviewPayload,
  extractApiError,
  DEFAULT_SETTINGS,
  PREVIEW_STAGES,
} from "@/lib/post-processing-helpers";

describe("parsePreviewInput", () => {
  it("returns error for empty input", () => {
    expect(parsePreviewInput("")).toEqual({ ok: false, error: "Enter detection JSON first" });
    expect(parsePreviewInput("   ")).toEqual({ ok: false, error: "Enter detection JSON first" });
  });

  it("returns error for invalid JSON", () => {
    const result = parsePreviewInput("{bad json}");
    expect(result.ok).toBe(false);
    expect((result as any).error).toBe("Invalid JSON");
  });

  it("accepts a top-level JSON array in the backend detection format", () => {
    const input = '[{"class_name":"person","confidence":0.9,"x1":0.1,"y1":0.2,"x2":0.4,"y2":0.6}]';
    const result = parsePreviewInput(input);
    expect(result.ok).toBe(true);
    expect((result as any).detections).toHaveLength(1);
    expect((result as any).detections[0].class_name).toBe("person");
  });

  it("accepts an object with a detections array in the backend detection format", () => {
    const input = JSON.stringify({
      detections: [{ class_name: "car", confidence: 0.7, x1: 0.0, y1: 0.0, x2: 0.3, y2: 0.3 }],
    });
    const result = parsePreviewInput(input);
    expect(result.ok).toBe(true);
    expect((result as any).detections[0].class_name).toBe("car");
  });

  it("returns error when JSON is valid but not an array or detections wrapper", () => {
    const result = parsePreviewInput('{"label":"person"}');
    expect(result.ok).toBe(false);
  });

  it("accepts an empty detections array", () => {
    const result = parsePreviewInput("[]");
    expect(result.ok).toBe(true);
    expect((result as any).detections).toHaveLength(0);
  });
});

describe("mergeWithDefaults", () => {
  it("fills all missing fields with defaults", () => {
    const merged = mergeWithDefaults({});
    expect(merged).toEqual(DEFAULT_SETTINGS);
  });

  it("overrides defaults with provided values", () => {
    const merged = mergeWithDefaults({ enabled: false, threshold: 0.8 });
    expect(merged.enabled).toBe(false);
    expect(merged.threshold).toBe(0.8);
    expect(merged.keep_grace).toBe(DEFAULT_SETTINGS.keep_grace);
  });

  it("preserves class_filter array from data", () => {
    const merged = mergeWithDefaults({ class_filter: ["person", "car"] });
    expect(merged.class_filter).toEqual(["person", "car"]);
  });

  it("defaults class_filter to empty array when absent", () => {
    const merged = mergeWithDefaults({ enabled: true });
    expect(merged.class_filter).toEqual([]);
  });
});

describe("DEFAULT_SETTINGS matches backend contract", () => {
  it("enabled defaults to false", () => {
    expect(DEFAULT_SETTINGS.enabled).toBe(false);
  });

  it("max_observations defaults to 5", () => {
    expect(DEFAULT_SETTINGS.max_observations).toBe(5);
  });
});

describe("PREVIEW_STAGES keys match backend response shape", () => {
  const keys = PREVIEW_STAGES.map(s => s.key);

  it("includes input_detections", () => expect(keys).toContain("input_detections"));
  it("includes filtered_detections", () => expect(keys).toContain("filtered_detections"));
  it("includes post_nms_detections", () => expect(keys).toContain("post_nms_detections"));
  it("includes final_detections", () => expect(keys).toContain("final_detections"));

  it("has four stages", () => expect(PREVIEW_STAGES).toHaveLength(4));

  it("stage labels are human-readable (no _detections suffix)", () => {
    PREVIEW_STAGES.forEach(({ label }) => {
      expect(label).not.toMatch(/_detections$/);
    });
  });
});

describe("buildPreviewPayload", () => {
  const detections = [{ class_name: "person", confidence: 0.9, x1: 0.1, y1: 0.2, x2: 0.4, y2: 0.6 }];
  const tracker = { tracks: [{ id: 1 }] };

  it("includes only detections when useTrackerState is false", () => {
    const payload = buildPreviewPayload(detections, tracker, false);
    expect(payload).toEqual({ detections });
    expect("tracker_state" in payload).toBe(false);
  });

  it("includes tracker_state when useTrackerState is true and trackerState is non-null", () => {
    const payload = buildPreviewPayload(detections, tracker, true);
    expect(payload.tracker_state).toBe(tracker);
  });

  it("omits tracker_state when trackerState is null even if useTrackerState is true", () => {
    const payload = buildPreviewPayload(detections, null, true);
    expect("tracker_state" in payload).toBe(false);
  });

  it("passes detections through unmodified", () => {
    const payload = buildPreviewPayload(detections, null, false);
    expect(payload.detections).toBe(detections);
  });
});

describe("extractApiError", () => {
  it("prefers response.data.detail", () => {
    const e = { response: { data: { detail: "Not found" } } };
    expect(extractApiError(e)).toBe("Not found");
  });

  it("falls back to e.message when no detail", () => {
    const e = { message: "Network Error" };
    expect(extractApiError(e)).toBe("Network Error");
  });

  it("returns fallback string for unknown error shapes", () => {
    expect(extractApiError({})).toBe("An unexpected error occurred");
    expect(extractApiError(null)).toBe("An unexpected error occurred");
  });
});

// ── API client method tests ───────────────────────────────────────────────────

import axios from "axios";

jest.mock("axios", () => {
  const mockGet  = jest.fn().mockResolvedValue({ data: {} });
  const mockPut  = jest.fn().mockResolvedValue({ data: {} });
  const mockPost = jest.fn().mockResolvedValue({ data: {} });
  const instance = {
    get: mockGet,
    put: mockPut,
    post: mockPost,
    interceptors: {
      request:  { use: jest.fn() },
      response: { use: jest.fn() },
    },
  };
  const mockCreate = jest.fn().mockReturnValue(instance);
  return { default: { create: mockCreate }, create: mockCreate };
});

function getInstance() {
  const axiosMod = jest.requireMock("axios");
  return axiosMod.default.create.mock.results[0]?.value;
}

import { postProcessingApi } from "@/utils/api";

beforeEach(() => {
  const inst = getInstance();
  inst?.get.mockClear();
  inst?.put.mockClear();
  inst?.post.mockClear();
});

describe("postProcessingApi.getSettings", () => {
  it("calls GET /projects/{id}/post-processing-settings", async () => {
    await postProcessingApi.getSettings("proj-1");
    const [url] = getInstance().get.mock.calls[0];
    expect(url).toBe("/projects/proj-1/post-processing-settings");
  });
});

describe("postProcessingApi.updateSettings", () => {
  it("calls PUT /projects/{id}/post-processing-settings with the data payload", async () => {
    const data = { enabled: true, threshold: 0.6 };
    await postProcessingApi.updateSettings("proj-2", data);
    const [url, body] = getInstance().put.mock.calls[0];
    expect(url).toBe("/projects/proj-2/post-processing-settings");
    expect(body).toEqual(data);
  });
});

describe("postProcessingApi.runPreview", () => {
  it("calls POST /projects/{id}/post-processing-preview with the payload", async () => {
    const payload = { detections: [{ class_name: "car", confidence: 0.8, x1: 0.0, y1: 0.0, x2: 0.3, y2: 0.3 }] };
    await postProcessingApi.runPreview("proj-3", payload);
    const [url, body] = getInstance().post.mock.calls[0];
    expect(url).toBe("/projects/proj-3/post-processing-preview");
    expect(body).toEqual(payload);
  });

  it("includes tracker_state in the POST body when supplied", async () => {
    const payload = {
      detections: [],
      tracker_state: { tracks: [] },
    };
    await postProcessingApi.runPreview("proj-3", payload);
    const [, body] = getInstance().post.mock.calls[0];
    expect(body.tracker_state).toEqual({ tracks: [] });
  });
});

describe("legacy postProcessingApi methods are preserved", () => {
  it("getConfig calls GET /post-processing/model/{id}", async () => {
    await postProcessingApi.getConfig("model-1");
    const [url] = getInstance().get.mock.calls[0];
    expect(url).toBe("/post-processing/model/model-1");
  });
});
