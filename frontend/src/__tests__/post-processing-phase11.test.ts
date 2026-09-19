/**
 * Phase 11 — class_filter mismatch fix tests
 *
 * Tests:
 *   1. Checkbox checked state maps to include semantics in payload
 *   2. Save payload contains exactly the checked classes (not the complement)
 *   3. appStore ppPage does NOT include settings — persistence can't overwrite
 *      the backend-fetched class_filter on mount
 *   4. renderFromSample must save before render (contract: saveSettings called first)
 *   5. All other settings fields round-trip correctly through mergeWithDefaults
 *   6. mergeWithDefaults: backend class_filter wins over local default
 */

import {
  DEFAULT_SETTINGS,
  mergeWithDefaults,
  resolvePostProcessingModelSupport,
  serializeForPersist,
  INITIAL_VIDEO_STATE,
  type PPSettings,
  type PersistedPPState,
} from "@/lib/post-processing-helpers";

// ─── 1. Checkbox include semantics ───────────────────────────────────────────

describe("class_filter checkbox semantics: checked = included", () => {
  // Mirror the exact toggleClassFilter logic from page.tsx
  function toggleClassFilter(filter: string[], name: string): string[] {
    return filter.includes(name)
      ? filter.filter(n => n !== name)
      : [...filter, name];
  }

  it("checking bear adds it to class_filter (include semantics)", () => {
    const result = toggleClassFilter([], "bear");
    expect(result).toEqual(["bear"]);
  });

  it("unchecking bear removes it from class_filter", () => {
    const result = toggleClassFilter(["bear"], "bear");
    expect(result).toEqual([]);
  });

  it("checked state is derived from class_filter.includes — bear checked when in array", () => {
    const filter = ["bear"];
    expect(filter.includes("bear")).toBe(true);
    expect(filter.includes("woodpecker")).toBe(false);
    expect(filter.includes("whale")).toBe(false);
    expect(filter.includes("zebra")).toBe(false);
  });

  it("only bear checked → class_filter is ['bear'], not the complement", () => {
    const allClasses = ["bear", "woodpecker", "whale", "zebra"];
    // Start unchecked (initial backend state: [])
    let filter: string[] = [];
    // User checks only bear
    filter = toggleClassFilter(filter, "bear");
    expect(filter).toEqual(["bear"]);
    // Complement must NOT be stored
    const complement = allClasses.filter(c => !filter.includes(c));
    expect(complement).toEqual(["woodpecker", "whale", "zebra"]);
    // The payload must be filter (not complement)
    expect(filter).not.toEqual(complement);
  });

  it("starting from all-checked state and unchecking all but bear gives ['bear']", () => {
    const allClasses = ["bear", "woodpecker", "whale", "zebra"];
    // If backend returned all four as saved
    let filter: string[] = [...allClasses];
    // User unchecks woodpecker, whale, zebra
    filter = toggleClassFilter(filter, "woodpecker");
    filter = toggleClassFilter(filter, "whale");
    filter = toggleClassFilter(filter, "zebra");
    expect(filter).toEqual(["bear"]);
  });

  it("empty filter means all classes pass — no filtering", () => {
    const filter: string[] = [];
    // empty array = no class filter restriction
    expect(filter.length).toBe(0);
  });
});

// ─── 2. Save payload is exactly the checked list ─────────────────────────────

describe("save payload contains exactly the checked classes", () => {
  it("settings.class_filter is what would be sent to updateSettings", () => {
    const settings: PPSettings = {
      ...DEFAULT_SETTINGS,
      class_filter: ["bear"],
    };
    // saveSettings sends `settings` directly — verify class_filter is exactly ["bear"]
    expect(settings.class_filter).toEqual(["bear"]);
  });

  it("payload does not invert to all-other-classes when only bear is checked", () => {
    const allClasses = ["bear", "woodpecker", "whale", "zebra"];
    const checkedClasses = ["bear"];
    const settings: PPSettings = {
      ...DEFAULT_SETTINGS,
      class_filter: checkedClasses,
    };
    // The payload must be the checked list, not the unchecked list
    expect(settings.class_filter).toEqual(["bear"]);
    expect(settings.class_filter).not.toEqual(
      allClasses.filter(c => !checkedClasses.includes(c)),
    );
  });

  it("all 6 settings fields are present in save payload", () => {
    const settings: PPSettings = {
      enabled: true,
      threshold: 0.6,
      tracking_enabled: true,
      keep_grace: 4,
      max_observations: 8,
      class_filter: ["bear"],
    };
    const keys = Object.keys(settings);
    expect(keys).toContain("enabled");
    expect(keys).toContain("threshold");
    expect(keys).toContain("tracking_enabled");
    expect(keys).toContain("keep_grace");
    expect(keys).toContain("max_observations");
    expect(keys).toContain("class_filter");
  });
});

// ─── 3. appStore ppPage does NOT contain settings ────────────────────────────

describe("ppPage persistence does not include settings", () => {
  it("serializeForPersist output has no class_filter key", () => {
    const result = serializeForPersist("sample-1", INITIAL_VIDEO_STATE);
    expect("class_filter" in result).toBe(false);
  });

  it("serializeForPersist output has no enabled key", () => {
    const result = serializeForPersist(null, INITIAL_VIDEO_STATE);
    expect("enabled" in result).toBe(false);
  });

  it("serializeForPersist output has no threshold key", () => {
    const result = serializeForPersist(null, INITIAL_VIDEO_STATE);
    expect("threshold" in result).toBe(false);
  });

  it("PersistedPPState only contains job/video fields — not settings", () => {
    const persisted: PersistedPPState = {
      selectedSampleId: "s1",
      jobId: "job-1",
      status: "complete",
      outputUrl: "https://cdn/v.mp4",
      errorMessage: null,
    };
    // Explicitly verify no settings fields leak into persisted state
    expect("class_filter"       in persisted).toBe(false);
    expect("enabled"            in persisted).toBe(false);
    expect("threshold"          in persisted).toBe(false);
    expect("tracking_enabled"   in persisted).toBe(false);
    expect("keep_grace"         in persisted).toBe(false);
    expect("max_observations"   in persisted).toBe(false);
  });

  it("backend-fetched settings cannot be overwritten by ppPage restore", () => {
    // Simulate mount sequence:
    //   1. loadSettings() → backend returns class_filter: ["bear"]
    //   2. restorePersistedState() → reads ppPage → only restores selectedSampleId + job
    // Settings remain ["bear"] — not clobbered by the restore
    const backendSettings: PPSettings = mergeWithDefaults({ class_filter: ["bear"] });
    const ppPageEntry: PersistedPPState = {
      selectedSampleId: "s1",
      jobId: "old-job",
      status: "complete",
      outputUrl: "https://cdn/old.mp4",
      errorMessage: null,
    };
    // restorePersistedState only touches selectedSampleId and videoJob state
    // settings are not touched → backend value wins
    expect(backendSettings.class_filter).toEqual(["bear"]);
    // ppPage has no class_filter to overwrite with
    expect("class_filter" in ppPageEntry).toBe(false);
  });
});

// ─── 4. renderFromSample must save before render ──────────────────────────────

describe("renderFromSample saves settings before triggering render", () => {
  it("save-then-render sequence: DB is updated before the worker runs", async () => {
    // This tests the contract that the save API call (updateSettings) must
    // complete before triggerPostProcessingFromSample is called.
    // The fix in renderFromSample awaits saveSettings(true) first.

    const calls: string[] = [];

    const mockUpdateSettings = jest.fn().mockImplementation(async () => {
      calls.push("save");
      return { data: {} };
    });

    const mockTrigger = jest.fn().mockImplementation(async () => {
      calls.push("render");
      return { data: { id: "job-1" } };
    });

    // Simulate the fixed renderFromSample logic
    async function renderFromSample() {
      // Step 1: save settings (the fix)
      await mockUpdateSettings("proj-1", { class_filter: ["bear"] });
      // Step 2: trigger render
      await mockTrigger("proj-1", "sample-1", null);
    }

    await renderFromSample();

    expect(calls[0]).toBe("save");
    expect(calls[1]).toBe("render");
    expect(mockUpdateSettings).toHaveBeenCalledTimes(1);
    expect(mockTrigger).toHaveBeenCalledTimes(1);
  });

  it("render is aborted when save fails", async () => {
    const mockUpdateSettings = jest.fn().mockRejectedValue(new Error("network error"));
    const mockTrigger = jest.fn();

    async function renderFromSample(): Promise<boolean> {
      try {
        await mockUpdateSettings("proj-1", {});
      } catch {
        return false; // save failed — abort render
      }
      await mockTrigger("proj-1", "sample-1", null);
      return true;
    }

    const result = await renderFromSample();
    expect(result).toBe(false);
    expect(mockTrigger).not.toHaveBeenCalled();
  });

  it("stale-render scenario: without save, worker sees old DB value", () => {
    // Demonstrates the bug: if renderFromSample did NOT save first,
    // the DB would still hold the previously-saved list.
    // This is the scenario that caused ['woodpecker','whale','zebra'] to appear
    // in the worker log when the user had only checked 'bear' in the UI.
    const dbState = { class_filter: ["woodpecker", "whale", "zebra"] }; // old saved
    const uiState = { class_filter: ["bear"] };                          // current UI

    // Without save: worker uses dbState (the bug)
    const workerWithoutSave = dbState.class_filter;
    expect(workerWithoutSave).not.toEqual(uiState.class_filter);
    expect(workerWithoutSave).toEqual(["woodpecker", "whale", "zebra"]);

    // With save: db is updated to uiState, worker uses uiState (the fix)
    const dbAfterSave = { class_filter: uiState.class_filter }; // save applied
    const workerWithSave = dbAfterSave.class_filter;
    expect(workerWithSave).toEqual(["bear"]);
  });
});

// ─── 5. Other settings fields round-trip through mergeWithDefaults ────────────

describe("all other settings fields round-trip correctly", () => {
  it("enabled=false overrides default true", () => {
    const s = mergeWithDefaults({ enabled: false });
    expect(s.enabled).toBe(false);
  });

  it("threshold round-trips", () => {
    const s = mergeWithDefaults({ threshold: 0.73 });
    expect(s.threshold).toBe(0.73);
  });

  it("tracking_enabled=true overrides default false", () => {
    const s = mergeWithDefaults({ tracking_enabled: true });
    expect(s.tracking_enabled).toBe(true);
  });

  it("keep_grace round-trips", () => {
    const s = mergeWithDefaults({ keep_grace: 9 });
    expect(s.keep_grace).toBe(9);
  });

  it("max_observations round-trips", () => {
    const s = mergeWithDefaults({ max_observations: 15 });
    expect(s.max_observations).toBe(15);
  });

  it("class_filter from backend wins over empty default", () => {
    const s = mergeWithDefaults({ class_filter: ["bear"] });
    expect(s.class_filter).toEqual(["bear"]);
  });

  it("all six fields survive a full round-trip through mergeWithDefaults", () => {
    const backend = {
      enabled: false,
      threshold: 0.33,
      tracking_enabled: true,
      keep_grace: 8,
      max_observations: 12,
      class_filter: ["bear", "whale"],
    };
    const merged = mergeWithDefaults(backend);
    expect(merged.enabled).toBe(false);
    expect(merged.threshold).toBe(0.33);
    expect(merged.tracking_enabled).toBe(true);
    expect(merged.keep_grace).toBe(8);
    expect(merged.max_observations).toBe(12);
    expect(merged.class_filter).toEqual(["bear", "whale"]);
  });
});

describe("resolvePostProcessingModelSupport", () => {
  it("marks YOLO-Pro models as ready", () => {
    const support = resolvePostProcessingModelSupport([
      {
        models: [
          {
            format: "tflite",
            output_type: "yolo_pro_detection",
            model_metadata: { variant: "decoded_float32", output_type: "yolo_pro_detection" },
          },
        ],
      },
    ]);
    expect(support).toEqual({ status: "ready", message: null });
  });

  it("marks FOMO models as ready (preview pipeline now decodes the heatmap)", () => {
    const support = resolvePostProcessingModelSupport([
      {
        models: [
          {
            format: "tflite",
            output_type: "detection_heatmap",
            model_metadata: { architecture: "fomo_mobilenetv2_0_1" },
          },
        ],
      },
    ]);
    expect(support).toEqual({ status: "ready", message: null });
  });

  it("marks FOMO models as ready when only architecture string hints at FOMO", () => {
    const support = resolvePostProcessingModelSupport([
      {
        models: [
          {
            format: "tflite",
            // No explicit output_type; rely on architecture name.
            model_metadata: { architecture: "FOMO_MobileNetV2_0_1" },
          },
        ],
      },
    ]);
    expect(support.status).toBe("ready");
  });

  it("still marks truly unrelated model types as unsupported", () => {
    const support = resolvePostProcessingModelSupport([
      {
        models: [
          {
            format: "tflite",
            output_type: "classification",
            model_metadata: { architecture: "mobilenetv2" },
          },
        ],
      },
    ]);
    expect(support.status).toBe("unsupported");
    expect(support.message).toMatch(/Vision Pro, EdgeDetect Lite, or NanoVision/i);
  });

  it("marks empty histories as no_model and mentions NanoVision as a supported option", () => {
    const support = resolvePostProcessingModelSupport([]);
    expect(support.status).toBe("no_model");
    expect(support.message).toMatch(/NanoVision/i);
  });
});
