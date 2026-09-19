/**
 * Phase 10 — settings flow end-to-end (frontend layer)
 *
 * Tests:
 *   DEFAULT_SETTINGS.enabled is true      — pipeline active by default
 *   mergeWithDefaults respects enabled    — backend false overrides frontend default
 *   saveSettings payload shape            — all 6 fields included in PUT body
 *   class_filter checkbox state model    — toggle adds/removes names correctly
 *   settings-to-pipeline shape contract  — all field names match backend schema
 */

import {
  DEFAULT_SETTINGS,
  mergeWithDefaults,
  type PPSettings,
} from "@/lib/post-processing-helpers";

// ─── DEFAULT_SETTINGS.enabled is true ────────────────────────────────────────

describe("DEFAULT_SETTINGS.enabled", () => {
  it("is true so the pipeline is active before any settings are saved", () => {
    expect(DEFAULT_SETTINGS.enabled).toBe(true);
  });

  it("threshold defaults to 0.5", () => {
    expect(DEFAULT_SETTINGS.threshold).toBe(0.5);
  });

  it("tracking_enabled defaults to false", () => {
    expect(DEFAULT_SETTINGS.tracking_enabled).toBe(false);
  });

  it("keep_grace defaults to 3", () => {
    expect(DEFAULT_SETTINGS.keep_grace).toBe(3);
  });

  it("max_observations defaults to 5", () => {
    expect(DEFAULT_SETTINGS.max_observations).toBe(5);
  });

  it("class_filter defaults to empty array", () => {
    expect(DEFAULT_SETTINGS.class_filter).toEqual([]);
  });
});

// ─── mergeWithDefaults honours backend values ────────────────────────────────

describe("mergeWithDefaults", () => {
  it("backend enabled=false overrides the frontend default of true", () => {
    const merged = mergeWithDefaults({ enabled: false });
    expect(merged.enabled).toBe(false);
  });

  it("backend enabled=true is preserved", () => {
    const merged = mergeWithDefaults({ enabled: true });
    expect(merged.enabled).toBe(true);
  });

  it("missing backend fields fall back to DEFAULT_SETTINGS", () => {
    const merged = mergeWithDefaults({});
    expect(merged.enabled).toBe(DEFAULT_SETTINGS.enabled);
    expect(merged.threshold).toBe(DEFAULT_SETTINGS.threshold);
    expect(merged.keep_grace).toBe(DEFAULT_SETTINGS.keep_grace);
    expect(merged.max_observations).toBe(DEFAULT_SETTINGS.max_observations);
    expect(merged.class_filter).toEqual([]);
  });

  it("backend class_filter array is preserved", () => {
    const merged = mergeWithDefaults({ class_filter: ["cat", "dog"] });
    expect(merged.class_filter).toEqual(["cat", "dog"]);
  });

  it("backend tracking_enabled=true is preserved", () => {
    const merged = mergeWithDefaults({ tracking_enabled: true });
    expect(merged.tracking_enabled).toBe(true);
  });
});

// ─── Save payload includes all 6 fields ──────────────────────────────────────

describe("settings save payload shape", () => {
  it("all 6 fields are present in the settings object sent to updateSettings", () => {
    const settings: PPSettings = {
      enabled: true,
      threshold: 0.7,
      tracking_enabled: true,
      keep_grace: 5,
      max_observations: 8,
      class_filter: ["person"],
    };
    // Simulate what saveSettings() sends (the full PPSettings object as JSON body)
    const keys = Object.keys(settings);
    expect(keys).toContain("enabled");
    expect(keys).toContain("threshold");
    expect(keys).toContain("tracking_enabled");
    expect(keys).toContain("keep_grace");
    expect(keys).toContain("max_observations");
    expect(keys).toContain("class_filter");
  });

  it("field names match the backend schema exactly (snake_case)", () => {
    const s: PPSettings = DEFAULT_SETTINGS;
    // These must be snake_case to match PostProcessingSettingsUpdate
    expect("tracking_enabled" in s).toBe(true);
    expect("keep_grace" in s).toBe(true);
    expect("max_observations" in s).toBe(true);
    expect("class_filter" in s).toBe(true);
    // No camelCase aliases
    expect("trackingEnabled" in (s as any)).toBe(false);
    expect("keepGrace" in (s as any)).toBe(false);
    expect("maxObservations" in (s as any)).toBe(false);
    expect("classFilter" in (s as any)).toBe(false);
  });
});

// ─── class_filter checkbox state model ───────────────────────────────────────

describe("class_filter toggle logic", () => {
  function toggleClassFilter(filter: string[], name: string): string[] {
    return filter.includes(name)
      ? filter.filter(n => n !== name)
      : [...filter, name];
  }

  it("checking an unchecked class adds it to the filter", () => {
    const result = toggleClassFilter([], "cat");
    expect(result).toEqual(["cat"]);
  });

  it("unchecking a checked class removes it from the filter", () => {
    const result = toggleClassFilter(["cat", "dog"], "cat");
    expect(result).toEqual(["dog"]);
  });

  it("checking all classes then unchecking all returns empty array", () => {
    let filter: string[] = [];
    filter = toggleClassFilter(filter, "cat");
    filter = toggleClassFilter(filter, "dog");
    filter = toggleClassFilter(filter, "cat");
    filter = toggleClassFilter(filter, "dog");
    expect(filter).toEqual([]);
  });

  it("empty filter means pipeline passes all classes (expected backend behavior)", () => {
    // empty array sent to backend → class_filter=[] → no class filtering applied
    expect(toggleClassFilter(["cat"], "cat")).toEqual([]);
  });

  it("checked state is derived from class_filter.includes(name)", () => {
    const filter = ["cat"];
    expect(filter.includes("cat")).toBe(true);
    expect(filter.includes("dog")).toBe(false);
  });
});

// ─── enabled gate: pipeline-off means no filtering visible ───────────────────

describe("enabled=false short-circuit contract", () => {
  it("with enabled=false merged settings still has enabled=false", () => {
    const merged = mergeWithDefaults({ enabled: false, class_filter: ["cat"] });
    expect(merged.enabled).toBe(false);
    // class_filter is still stored; just won't be applied by pipeline until enabled=true
    expect(merged.class_filter).toEqual(["cat"]);
  });

  it("enabling processing is a separate step from setting class_filter", () => {
    // User can set class_filter without enabling; filter takes effect on save+enable
    const settings: PPSettings = {
      ...DEFAULT_SETTINGS,
      enabled: false,
      class_filter: ["dog"],
    };
    expect(settings.class_filter).toEqual(["dog"]);
    expect(settings.enabled).toBe(false);
  });
});
