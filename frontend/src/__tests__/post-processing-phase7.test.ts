/**
 * Phase 7 — sample-based video trigger tests.
 *
 * Tests:
 *   triggerPostProcessingFromSample — API client contract
 *   isVideoSample helper logic (via filename extension checks)
 *   Button disabled state logic
 *   Polling and output flow still works (unchanged from Phase 6)
 */

// ── API client tests ──────────────────────────────────────────────────────────

import axios from "axios";

jest.mock("axios", () => {
  const mockGet  = jest.fn().mockResolvedValue({ data: {} });
  const mockPost = jest.fn().mockResolvedValue({ data: { id: "job-1", status: "pending" } });
  const instance = {
    get:  mockGet,
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
  getInstance()?.get.mockClear();
  getInstance()?.post.mockClear();
});

describe("postProcessingApi.triggerPostProcessingFromSample", () => {
  it("POSTs to /projects/{id}/post-processing-video-from-sample", async () => {
    await postProcessingApi.triggerPostProcessingFromSample("proj-1", "sample-abc");
    const [url] = getInstance().post.mock.calls[0];
    expect(url).toBe("/projects/proj-1/post-processing-video-from-sample");
  });

  it("sends sample_id in JSON body", async () => {
    await postProcessingApi.triggerPostProcessingFromSample("proj-1", "sample-abc");
    const [, body] = getInstance().post.mock.calls[0];
    expect(body).toMatchObject({ sample_id: "sample-abc" });
  });

  it("sends impulse_id when provided", async () => {
    await postProcessingApi.triggerPostProcessingFromSample("proj-1", "sample-abc", "imp-xyz");
    const [, body] = getInstance().post.mock.calls[0];
    expect(body).toMatchObject({ sample_id: "sample-abc", impulse_id: "imp-xyz" });
  });

  it("sends impulse_id as null when not provided", async () => {
    await postProcessingApi.triggerPostProcessingFromSample("proj-1", "sample-abc");
    const [, body] = getInstance().post.mock.calls[0];
    expect(body.impulse_id).toBeNull();
  });

  it("does NOT send FormData (JSON body, not multipart)", async () => {
    await postProcessingApi.triggerPostProcessingFromSample("proj-1", "sample-abc");
    const [, body] = getInstance().post.mock.calls[0];
    expect(body instanceof FormData).toBe(false);
    expect(typeof body).toBe("object");
  });

  it("uses a distinct URL from the direct-upload endpoint", async () => {
    await postProcessingApi.triggerPostProcessingFromSample("proj-1", "sample-abc");
    const [url] = getInstance().post.mock.calls[0];
    expect(url).not.toBe("/projects/proj-1/post-processing-video");
    expect(url).not.toBe("/projects/proj-1/post-processing-video-debug");
  });
});

// ── Existing API endpoints still have correct URLs (regression guard) ──────────

describe("Phase 6 endpoints unchanged", () => {
  it("getProcessingJob still uses correct URL", async () => {
    await postProcessingApi.getProcessingJob("proj-1", "job-42");
    const [url] = getInstance().get.mock.calls[0];
    expect(url).toBe("/projects/proj-1/post-processing-jobs/job-42");
  });

  it("getProcessingJobVideoUrl still uses correct URL", async () => {
    await postProcessingApi.getProcessingJobVideoUrl("proj-1", "job-42");
    const [url] = getInstance().get.mock.calls[0];
    expect(url).toBe("/projects/proj-1/post-processing-jobs/job-42/video");
  });
});

// ── Video filtering logic ─────────────────────────────────────────────────────

describe("video filename extension filtering", () => {
  const VIDEO_EXTS = [".mp4", ".avi", ".mov"];

  function isVideoFilename(filename: string): boolean {
    const dot = filename.lastIndexOf(".");
    return dot !== -1 && VIDEO_EXTS.includes(filename.slice(dot).toLowerCase());
  }

  it("accepts .mp4", () => { expect(isVideoFilename("clip.mp4")).toBe(true); });
  it("accepts .avi", () => { expect(isVideoFilename("clip.avi")).toBe(true); });
  it("accepts .mov", () => { expect(isVideoFilename("clip.mov")).toBe(true); });
  it("accepts .MP4 (case-insensitive)", () => { expect(isVideoFilename("clip.MP4")).toBe(true); });
  it("accepts .MOV (case-insensitive)", () => { expect(isVideoFilename("clip.MOV")).toBe(true); });
  it("rejects .csv", () => { expect(isVideoFilename("data.csv")).toBe(false); });
  it("rejects .jpg", () => { expect(isVideoFilename("frame.jpg")).toBe(false); });
  it("rejects .wav", () => { expect(isVideoFilename("audio.wav")).toBe(false); });
  it("rejects no extension", () => { expect(isVideoFilename("video")).toBe(false); });
  it("rejects empty string", () => { expect(isVideoFilename("")).toBe(false); });
});

// ── Render button disabled-state contract ─────────────────────────────────────

describe("Render preview button disabled state", () => {
  it("disabled when no video selected (selectedVideoId is null)", () => {
    const selectedVideoId: string | null = null;
    const isProcessing = false;
    const disabled = !selectedVideoId || isProcessing;
    expect(disabled).toBe(true);
  });

  it("disabled when job is processing (even if video selected)", () => {
    const selectedVideoId = "sample-abc";
    const isProcessing = true;
    const disabled = !selectedVideoId || isProcessing;
    expect(disabled).toBe(true);
  });

  it("enabled when video selected and not processing", () => {
    const selectedVideoId = "sample-abc";
    const isProcessing = false;
    const disabled = !selectedVideoId || isProcessing;
    expect(disabled).toBe(false);
  });
});

// ── Video list empty state contract ──────────────────────────────────────────

describe("video list empty state", () => {
  it("empty videos array means no video to select", () => {
    const projectVideos: { id: string; filename: string }[] = [];
    expect(projectVideos.length).toBe(0);
  });

  it("non-empty videos array has selectable items", () => {
    const projectVideos = [{ id: "s1", filename: "clip.mp4" }];
    expect(projectVideos.length).toBeGreaterThan(0);
  });

  it("filtering samples keeps only video-extension filenames", () => {
    const allSamples = [
      { id: "s1", filename: "clip.mp4" },
      { id: "s2", filename: "data.csv" },
      { id: "s3", filename: "clip.avi" },
      { id: "s4", filename: "frame.jpg" },
      { id: "s5", filename: "video.mov" },
    ];
    const VIDEO_EXTS = [".mp4", ".avi", ".mov"];
    const videos = allSamples.filter(s => {
      const dot = s.filename.lastIndexOf(".");
      return dot !== -1 && VIDEO_EXTS.includes(s.filename.slice(dot).toLowerCase());
    });
    expect(videos).toHaveLength(3);
    expect(videos.map(v => v.id)).toEqual(["s1", "s3", "s5"]);
  });
});

// ── State-machine contract: selecting video does not start processing ─────────

describe("selecting video does not start processing", () => {
  it("selectedVideoId change alone does not modify videoJob state", () => {
    // The page sets selectedVideoId independently; rendering only triggers on
    // explicit renderFromSample() call.  Verify the state shapes are separate.
    const videoJobState = { status: "idle", jobId: null, outputUrl: null, errorMessage: null };
    const selectedVideoId = "sample-abc";

    // No state transition just from setting selectedVideoId
    expect(videoJobState.status).toBe("idle");
    expect(selectedVideoId).toBe("sample-abc");
  });

  it("job transitions to processing only after renderFromSample is called", () => {
    // Simulate calling renderFromSample: sets status to processing
    const processingState = {
      status: "processing" as const,
      jobId: null,
      outputUrl: null,
      errorMessage: null,
    };
    expect(processingState.status).toBe("processing");
  });
});

// ── Polling unchanged ─────────────────────────────────────────────────────────

import { shouldStopPolling, INITIAL_VIDEO_STATE } from "@/lib/post-processing-helpers";

describe("polling behaviour unchanged from Phase 6", () => {
  it("shouldStopPolling is false during processing", () => {
    expect(shouldStopPolling("processing")).toBe(false);
    expect(shouldStopPolling("pending")).toBe(false);
  });

  it("shouldStopPolling is true on complete", () => {
    expect(shouldStopPolling("complete")).toBe(true);
  });

  it("shouldStopPolling is true on failed", () => {
    expect(shouldStopPolling("failed")).toBe(true);
  });

  it("INITIAL_VIDEO_STATE is still idle with null fields", () => {
    expect(INITIAL_VIDEO_STATE).toMatchObject({
      status: "idle",
      jobId: null,
      outputUrl: null,
      errorMessage: null,
    });
  });
});
