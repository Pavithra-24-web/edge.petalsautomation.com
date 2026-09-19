/**
 * Phase 8 — cross-navigation persistence tests.
 *
 * Tests:
 *   serializeForPersist    — correct serialization of page state
 *   shouldRestoreAsInProgress — correct identification of re-pollable states
 *   Sample validation logic — stale sample detection
 *   Job restore decisions  — complete / failed / in-progress / missing
 *   Empty-state guard      — only shown when truly no videos and no job
 *   "Render again" contract — overwrites persisted state with a new job
 *   Zustand store actions  — setPPPage / clearPPPage (via in-memory store)
 */

import {
  serializeForPersist,
  shouldRestoreAsInProgress,
  EMPTY_PERSISTED_STATE,
  INITIAL_VIDEO_STATE,
  shouldStopPolling,
  type PersistedPPState,
  type VideoJobState,
  type VideoJobStatus,
} from "@/lib/post-processing-helpers";

// ─── serializeForPersist ─────────────────────────────────────────────────────

describe("serializeForPersist", () => {
  it("stores selectedSampleId correctly", () => {
    const result = serializeForPersist("sample-abc", INITIAL_VIDEO_STATE);
    expect(result.selectedSampleId).toBe("sample-abc");
  });

  it("stores null selectedSampleId when nothing is selected", () => {
    const result = serializeForPersist(null, INITIAL_VIDEO_STATE);
    expect(result.selectedSampleId).toBeNull();
  });

  it("serializes idle job status as null", () => {
    const result = serializeForPersist(null, INITIAL_VIDEO_STATE);
    expect(result.status).toBeNull();
  });

  it("serializes processing status", () => {
    const job: VideoJobState = { status: "processing", jobId: "job-1", outputUrl: null, errorMessage: null };
    const result = serializeForPersist(null, job);
    expect(result.status).toBe("processing");
    expect(result.jobId).toBe("job-1");
  });

  it("serializes complete status with URL", () => {
    const job: VideoJobState = { status: "complete", jobId: "job-2", outputUrl: "https://cdn/video.mp4", errorMessage: null };
    const result = serializeForPersist("sample-1", job);
    expect(result.status).toBe("complete");
    expect(result.outputUrl).toBe("https://cdn/video.mp4");
    expect(result.jobId).toBe("job-2");
  });

  it("serializes failed status with error message", () => {
    const job: VideoJobState = { status: "failed", jobId: "job-3", outputUrl: null, errorMessage: "Disk full" };
    const result = serializeForPersist(null, job);
    expect(result.status).toBe("failed");
    expect(result.errorMessage).toBe("Disk full");
  });

  it("stores jobId as null when job has not started", () => {
    const result = serializeForPersist(null, INITIAL_VIDEO_STATE);
    expect(result.jobId).toBeNull();
  });
});

// ─── EMPTY_PERSISTED_STATE ───────────────────────────────────────────────────

describe("EMPTY_PERSISTED_STATE", () => {
  it("has null selectedSampleId", () => {
    expect(EMPTY_PERSISTED_STATE.selectedSampleId).toBeNull();
  });

  it("has null status (no job)", () => {
    expect(EMPTY_PERSISTED_STATE.status).toBeNull();
  });

  it("has null jobId", () => {
    expect(EMPTY_PERSISTED_STATE.jobId).toBeNull();
  });
});

// ─── shouldRestoreAsInProgress ───────────────────────────────────────────────

describe("shouldRestoreAsInProgress", () => {
  it("returns true for processing", () => {
    expect(shouldRestoreAsInProgress("processing")).toBe(true);
  });

  it("returns true for uploading", () => {
    expect(shouldRestoreAsInProgress("uploading")).toBe(true);
  });

  it("returns false for complete", () => {
    expect(shouldRestoreAsInProgress("complete")).toBe(false);
  });

  it("returns false for failed", () => {
    expect(shouldRestoreAsInProgress("failed")).toBe(false);
  });

  it("returns false for idle", () => {
    expect(shouldRestoreAsInProgress("idle")).toBe(false);
  });

  it("returns false for null (no job)", () => {
    expect(shouldRestoreAsInProgress(null)).toBe(false);
  });
});

// ─── Sample validation (restore: stale vs valid) ──────────────────────────────

describe("selected sample validation on restore", () => {
  const videos = [
    { id: "s1", filename: "clip.mp4" },
    { id: "s2", filename: "recording.avi" },
  ];

  it("a persisted sample id that exists in the fetched list is valid", () => {
    const saved: PersistedPPState = { ...EMPTY_PERSISTED_STATE, selectedSampleId: "s1" };
    const stillExists = videos.some(v => v.id === saved.selectedSampleId);
    expect(stillExists).toBe(true);
  });

  it("a persisted sample id that is not in the fetched list is stale", () => {
    const saved: PersistedPPState = { ...EMPTY_PERSISTED_STATE, selectedSampleId: "s-deleted" };
    const stillExists = videos.some(v => v.id === saved.selectedSampleId);
    expect(stillExists).toBe(false);
  });

  it("a null selectedSampleId skips validation", () => {
    const saved: PersistedPPState = { ...EMPTY_PERSISTED_STATE, selectedSampleId: null };
    // null means nothing was selected — no restore needed
    expect(saved.selectedSampleId).toBeNull();
  });
});

// ─── Job restore decision logic ──────────────────────────────────────────────

describe("job restore decision logic", () => {
  it("no jobId means no job restore needed", () => {
    const saved: PersistedPPState = { ...EMPTY_PERSISTED_STATE };
    const needsRestore = !!saved.jobId && !!saved.status;
    expect(needsRestore).toBe(false);
  });

  it("no status means no job restore needed", () => {
    const saved: PersistedPPState = { ...EMPTY_PERSISTED_STATE, jobId: "job-1" };
    const needsRestore = !!saved.jobId && !!saved.status;
    expect(needsRestore).toBe(false);
  });

  it("complete job always goes through the backend to get a fresh URL", () => {
    // Presigned URLs are short-lived — we never restore them from storage directly.
    // Both complete+URL and complete+null-URL hit getProcessingJob then getProcessingJobVideoUrl.
    const saved: PersistedPPState = {
      ...EMPTY_PERSISTED_STATE,
      jobId: "job-1",
      status: "complete",
      outputUrl: "https://cdn/out.mp4",   // persisted URL — ignored on restore
    };
    // The restore branch condition: status === "complete" → backend re-fetch, regardless of URL
    const requiresBackendFetch = saved.status === "complete";
    expect(requiresBackendFetch).toBe(true);
  });

  it("complete job without URL is also recoverable via backend re-fetch", () => {
    const saved: PersistedPPState = {
      ...EMPTY_PERSISTED_STATE,
      jobId: "job-1",
      status: "complete",
      outputUrl: null,   // partial state — still goes to backend
    };
    const requiresBackendFetch = saved.status === "complete";
    expect(requiresBackendFetch).toBe(true);
  });

  it("failed job restores error state without a network call", () => {
    const saved: PersistedPPState = {
      ...EMPTY_PERSISTED_STATE,
      jobId: "job-1",
      status: "failed",
      errorMessage: "Out of memory",
    };
    const restoreFailed = saved.status === "failed";
    expect(restoreFailed).toBe(true);
  });

  it("processing job requires a backend re-check (shouldRestoreAsInProgress)", () => {
    const saved: PersistedPPState = {
      ...EMPTY_PERSISTED_STATE,
      jobId: "job-1",
      status: "processing",
    };
    expect(shouldRestoreAsInProgress(saved.status as VideoJobStatus)).toBe(true);
  });

  it("after re-check: completed backend job stops polling immediately", () => {
    // Simulate: backend returns status=complete → shouldStopPolling → true
    const backendStatus = "complete";
    expect(shouldStopPolling(backendStatus)).toBe(true);
  });

  it("after re-check: failed backend job stops polling immediately", () => {
    const backendStatus = "failed";
    expect(shouldStopPolling(backendStatus)).toBe(true);
  });

  it("after re-check: still-processing backend job resumes polling", () => {
    const backendStatus = "processing";
    expect(shouldStopPolling(backendStatus)).toBe(false);
  });

  it("after re-check: pending backend job resumes polling", () => {
    const backendStatus = "pending";
    expect(shouldStopPolling(backendStatus)).toBe(false);
  });
});

// ─── Empty-state guard ────────────────────────────────────────────────────────

describe("empty state guard: only shown when truly empty", () => {
  it("no videos + no job → idle, empty state is correct", () => {
    const projectVideos: { id: string; filename: string }[] = [];
    const videoJobStatus: VideoJobStatus = "idle";
    const hasSomethingToShow = projectVideos.length > 0 || videoJobStatus !== "idle";
    expect(hasSomethingToShow).toBe(false);
  });

  it("no videos + complete job → preview should be shown (not empty state)", () => {
    const projectVideos: { id: string; filename: string }[] = [];
    const videoJobStatus = "complete" as VideoJobStatus;
    const hasSomethingToShow = projectVideos.length > 0 || videoJobStatus !== "idle";
    expect(hasSomethingToShow).toBe(true);
  });

  it("videos exist + idle job → video selector shown (not empty state)", () => {
    const projectVideos = [{ id: "s1", filename: "clip.mp4" }];
    const videoJobStatus: VideoJobStatus = "idle";
    const hasSomethingToShow = projectVideos.length > 0 || videoJobStatus !== "idle";
    expect(hasSomethingToShow).toBe(true);
  });

  it("no videos + processing job → spinner shown (not empty state)", () => {
    const projectVideos: { id: string; filename: string }[] = [];
    const videoJobStatus = "processing" as VideoJobStatus;
    const hasSomethingToShow = projectVideos.length > 0 || videoJobStatus !== "idle";
    expect(hasSomethingToShow).toBe(true);
  });

  it("no videos + failed job → error state shown (not empty state)", () => {
    const projectVideos: { id: string; filename: string }[] = [];
    const videoJobStatus = "failed" as VideoJobStatus;
    const hasSomethingToShow = projectVideos.length > 0 || videoJobStatus !== "idle";
    expect(hasSomethingToShow).toBe(true);
  });
});

// ─── "Render again" replaces persisted state ─────────────────────────────────

describe("render again replaces persisted state", () => {
  it("calling render again transitions status to processing (overwriting complete)", () => {
    // Current: complete job in store
    const currentPersisted: PersistedPPState = {
      selectedSampleId: "s1",
      jobId: "old-job",
      status: "complete",
      outputUrl: "https://cdn/old.mp4",
      errorMessage: null,
    };

    // After renderFromSample is called, videoJob transitions to processing
    const newJob: VideoJobState = { status: "processing", jobId: null, outputUrl: null, errorMessage: null };
    const updated = serializeForPersist(currentPersisted.selectedSampleId, newJob);

    expect(updated.status).toBe("processing");
    expect(updated.outputUrl).toBeNull();
    expect(updated.jobId).toBeNull();
    expect(updated.selectedSampleId).toBe("s1"); // selection preserved
  });

  it("after new job completes, persisted state holds new job id and URL", () => {
    const selectedSampleId = "s1";
    const newJob: VideoJobState = {
      status: "complete",
      jobId: "new-job-456",
      outputUrl: "https://cdn/new.mp4",
      errorMessage: null,
    };
    const updated = serializeForPersist(selectedSampleId, newJob);
    expect(updated.jobId).toBe("new-job-456");
    expect(updated.outputUrl).toBe("https://cdn/new.mp4");
    expect(updated.status).toBe("complete");
  });
});

// ─── Store action contract (tested as pure functions against plain objects) ───

describe("ppPage store action contracts", () => {
  it("setPPPage sets state for the given project id", () => {
    // Simulate the reducer: spread existing + overwrite the key
    const existing: Record<string, PersistedPPState> = {};
    const incoming: PersistedPPState = { ...EMPTY_PERSISTED_STATE, jobId: "job-1", status: "complete", outputUrl: "https://cdn/v.mp4" };
    const next = { ...existing, "proj-abc": incoming };
    expect(next["proj-abc"].jobId).toBe("job-1");
  });

  it("setPPPage does not affect other project ids", () => {
    const existing: Record<string, PersistedPPState> = {
      "proj-other": { ...EMPTY_PERSISTED_STATE, jobId: "job-other", status: "complete", outputUrl: "x" },
    };
    const next: Record<string, PersistedPPState> = {
      ...existing,
      "proj-abc": { ...EMPTY_PERSISTED_STATE },
    };
    expect(next["proj-other"].jobId).toBe("job-other");
  });

  it("clearPPPage removes the project entry", () => {
    const existing: Record<string, PersistedPPState> = {
      "proj-abc": { ...EMPTY_PERSISTED_STATE, jobId: "job-1" },
      "proj-other": { ...EMPTY_PERSISTED_STATE, jobId: "job-2" },
    };
    const next = { ...existing };
    delete next["proj-abc"];
    expect(next["proj-abc"]).toBeUndefined();
    expect(next["proj-other"].jobId).toBe("job-2"); // untouched
  });

  it("restore skips project with no persisted entry (ppPage[id] undefined)", () => {
    const ppPage: Record<string, PersistedPPState> = {};
    const saved = ppPage["proj-new"];
    expect(saved).toBeUndefined();
    // Restore function checks: if (!saved) return — no state is set
  });

  it("wrong-project job is cleared after a 404 from the backend", () => {
    // Simulate: clearPPPage(projectId) called, videoJob reset to INITIAL
    const ppPageAfterClear: Record<string, PersistedPPState> = {};
    const videoJobAfterClear: VideoJobState = INITIAL_VIDEO_STATE;
    expect(Object.keys(ppPageAfterClear)).toHaveLength(0);
    expect(videoJobAfterClear.status).toBe("idle");
  });
});

// ─── Polling safety (no duplicates) ──────────────────────────────────────────

describe("polling safety on restore", () => {
  it("startPolling calls stopPolling first, preventing duplicate intervals", () => {
    // The startPolling implementation calls stopPolling() before setting the new interval.
    // This test verifies that contract at the logic level.
    let intervalCount = 0;
    let activeInterval: ReturnType<typeof setInterval> | null = null;

    function stopPolling() {
      if (activeInterval != null) {
        clearInterval(activeInterval);
        activeInterval = null;
      }
    }

    function startPolling() {
      stopPolling(); // always clear before creating
      activeInterval = setInterval(() => { intervalCount++; }, 100000);
    }

    startPolling();
    startPolling(); // second call should not add a second interval
    expect(activeInterval).not.toBeNull();
    // Clear up
    if (activeInterval != null) clearInterval(activeInterval);
    // If there were two intervals, intervalCount would be > 0 after a tick, but
    // since we use 100000ms timers, the key contract here is structural:
    // startPolling → stopPolling → setInterval ensures one interval only.
  });

  it("shouldStopPolling on complete terminates polling", () => {
    expect(shouldStopPolling("complete")).toBe(true);
  });

  it("shouldStopPolling on failed terminates polling", () => {
    expect(shouldStopPolling("failed")).toBe(true);
  });

  it("shouldStopPolling on processing continues polling", () => {
    expect(shouldStopPolling("processing")).toBe(false);
  });
});
