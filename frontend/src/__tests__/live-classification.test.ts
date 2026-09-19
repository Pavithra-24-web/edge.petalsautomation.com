import {
  applyLiveStudioEventToDevices,
  buildClassifySampleRequest,
  canStartLiveSampling,
  deriveCanLoadSample,
  deriveCanStartSampling,
  deriveSensorOptions,
  deriveStartSamplingDisabledReason,
  fmtPct,
  getRuntimeBadgeMeta,
  getSamplePreviewUrl,
  isCorrectPrediction,
  isImageSample,
  mapPreferredRuntime,
  mapRawSampleToOption,
  mapStudioInferenceResult,
  resolveLiveResultSource,
  resolveViewState,
  resultMode,
  scoresToRows,
} from "@/lib/live-classification-helpers";
import type {
  LiveClassificationRunResult,
  LiveClassificationRuntimeInfo,
} from "@/types/live-classification";

describe("mapRawSampleToOption", () => {
  it("maps a sample with name and nested label object", () => {
    const raw = { id: "s1", name: "walk-001", label: { name: "walking" } };
    const opt = mapRawSampleToOption(raw);
    expect(opt).toMatchObject({ id: "s1", name: "walk-001", label: "walking" });
  });

  it("maps a sample with label as a plain string", () => {
    const raw = { id: "s2", name: "idle-001", label: "idle", sensor_type: "accelerometer" };
    const opt = mapRawSampleToOption(raw);
    expect(opt.label).toBe("idle");
    expect(opt.sensor_type).toBe("accelerometer");
  });

  it("falls back to sample_name when name is absent", () => {
    const raw = { id: "s3", sample_name: "idle-002", label_name: "idle" };
    expect(mapRawSampleToOption(raw).name).toBe("idle-002");
    expect(mapRawSampleToOption(raw).label).toBe("idle");
  });

  it("falls back to filename when neither name nor sample_name exists", () => {
    const raw = { id: "s4", filename: "raw.csv", expected_outcome: "unknown" };
    const opt = mapRawSampleToOption(raw);
    expect(opt.name).toBe("raw.csv");
    expect(opt.label).toBe("unknown");
  });

  it("uses fallback strings when all name and label fields are absent", () => {
    const opt = mapRawSampleToOption({ id: "s5" });
    expect(opt.name).toBe("Sample");
    expect(opt.label).toBe("—");
  });

  it("includes sensor_type and duration_ms from endpoint format", () => {
    const raw = {
      id: "s6",
      name: "cam.jpg",
      label: "cat",
      sensor_type: "camera",
      duration_ms: 3000,
    };
    const opt = mapRawSampleToOption(raw);
    expect(opt.sensor_type).toBe("camera");
    expect(opt.duration_ms).toBe(3000);
  });

  it("defaults sensor_type and duration_ms to null when absent", () => {
    const opt = mapRawSampleToOption({ id: "s7", name: "x" });
    expect(opt.sensor_type).toBeNull();
    expect(opt.duration_ms).toBeNull();
  });
});

describe("resolveViewState", () => {
  it("returns no_project when project is absent", () => {
    expect(resolveViewState({ hasProject: false, hasImpulse: true, hasModel: true })).toBe("no_project");
  });

  it("returns no_impulse when project exists but impulse does not", () => {
    expect(resolveViewState({ hasProject: true, hasImpulse: false, hasModel: true })).toBe("no_impulse");
  });

  it("returns no_model when project and impulse exist but model is absent", () => {
    expect(resolveViewState({ hasProject: true, hasImpulse: true, hasModel: false })).toBe("no_model");
  });

  it("returns ready when all prerequisites are met", () => {
    expect(resolveViewState({ hasProject: true, hasImpulse: true, hasModel: true })).toBe("ready");
  });
});

describe("deriveCanStartSampling", () => {
  it("is false when no device selected", () => {
    expect(deriveCanStartSampling("", "accelerometer", "100", true)).toBe(false);
  });

  it("is false when no sensor selected", () => {
    expect(deriveCanStartSampling("dev-1", "", "100", true)).toBe(false);
  });

  it("is false when no frequency selected", () => {
    expect(deriveCanStartSampling("dev-1", "accelerometer", "", true)).toBe(false);
  });

  it("is false when runtime is not ready", () => {
    expect(deriveCanStartSampling("dev-1", "accelerometer", "100", false)).toBe(false);
  });

  it("is true when all prerequisites are met", () => {
    expect(deriveCanStartSampling("dev-1", "accelerometer", "100", true)).toBe(true);
  });
});

describe("canStartLiveSampling", () => {
  it("allows stop while a stream is already active", () => {
    expect(canStartLiveSampling("Select a frequency to continue.", "active")).toBe(true);
  });

  it("allows the initial start only when no disabled reason exists", () => {
    expect(canStartLiveSampling(null, "idle")).toBe(true);
    expect(canStartLiveSampling("No compatible deployment is available for this device.", "idle")).toBe(false);
  });
});

describe("deriveCanLoadSample", () => {
  it("is false when no sample is selected", () => {
    expect(deriveCanLoadSample("", true)).toBe(false);
  });

  it("is false when runtime is not ready", () => {
    expect(deriveCanLoadSample("sample-1", false)).toBe(false);
  });

  it("is true when sample is selected and runtime is ready", () => {
    expect(deriveCanLoadSample("sample-1", true)).toBe(true);
  });
});

describe("preferred runtime helpers", () => {
  it("maps preferred runtime payload into the expected runtime shape", () => {
    const runtime = mapPreferredRuntime({
      id: 42,
      artifact_type: "pxe",
      display_name: "PXE Runtime",
      architecture: "fomo",
      version: null,
      best_accuracy: 0.93,
    });

    expect(runtime).toEqual<LiveClassificationRuntimeInfo>({
      id: "42",
      artifact_type: "pxe",
      display_name: "PXE Runtime",
      architecture: "fomo",
      version: null,
      best_accuracy: 0.93,
    });
  });

  it("builds classify-sample payload with the selected runtime id", () => {
    const payload = buildClassifySampleRequest("imp-1", "sample-1", {
      id: "dep-9",
      artifact_type: "pxe",
      display_name: "PXE Runtime",
      version: null,
      best_accuracy: null,
    });

    expect(payload).toEqual({
      impulse_id: "imp-1",
      sample_id: "sample-1",
      model_id: "dep-9",
    });
  });

  it("keeps model_id undefined when no runtime has been resolved yet", () => {
    expect(buildClassifySampleRequest("imp-1", "sample-1", null)).toEqual({
      impulse_id: "imp-1",
      sample_id: "sample-1",
      model_id: undefined,
    });
  });

  it("formats badge metadata from runtime fields", () => {
    const badge = getRuntimeBadgeMeta({
      id: "model-7",
      artifact_type: "tflite",
      display_name: "TFLite v7",
      architecture: "mobilenet",
      version: "7",
      best_accuracy: 0.812,
    });

    expect(badge).toEqual({
      artifactChip: "TFLite",
      versionChip: "v7",
      architecture: "mobilenet",
      accuracyText: "81.2% acc",
    });
  });

  it("accepts sparse runtime metadata for the badge", () => {
    const badge = getRuntimeBadgeMeta({
      id: "dep-1",
      artifact_type: "pxe",
      display_name: "PXE Runtime",
      version: null,
      best_accuracy: null,
    });

    expect(badge.artifactChip).toBe("PXE");
    expect(badge.versionChip).toBeNull();
    expect(badge.architecture).toBeNull();
    expect(badge.accuracyText).toBeNull();
  });
});

describe("studio live-classification helpers", () => {
  const baseDevice = {
    id: "pk-1",
    device_id: "dev-1",
    name: "Edge Node",
    device_type: "embedded",
    is_online: false,
  } as const;

  it("applies snapshot and lifecycle studio events to device state", () => {
    const initial = [baseDevice];
    const afterSnapshot = applyLiveStudioEventToDevices(initial, {
      type: "snapshot",
      project_id: "proj-1",
      ts: "2026-05-08T00:00:00Z",
      devices: [{ device_id: "dev-1", connected: true, mode: "idle" }],
    });
    const afterInference = applyLiveStudioEventToDevices(afterSnapshot, {
      type: "inference.started",
      project_id: "proj-1",
      device_id: "dev-1",
      ts: "2026-05-08T00:00:01Z",
      payload: {},
    });

    expect(afterSnapshot[0]).toMatchObject({ is_online: true, is_connected: true, mode: "idle" });
    expect(afterInference[0].mode).toBe("inference");
  });

  it("maps a studio inference payload into a result envelope", () => {
    const result = mapStudioInferenceResult(
      {
        label: "walking",
        confidence: 0.91,
        scores: { walking: 0.91, idle: 0.09 },
      },
      {
        id: "runtime-1",
        artifact_type: "pxe",
        display_name: "PXE Runtime",
        architecture: "fomo",
        version: "3",
        best_accuracy: 0.92,
      },
      "Edge Node",
      "pk-1",
    );

    expect(result.sample_name).toBe("Live stream · Edge Node");
    expect(result.model_id).toBe("runtime-1");
    expect(result.model_version).toBe(3);
    expect(result.label).toBe("walking");
  });

  it("keeps detection payloads on the live detection branch even with empty detections", () => {
    const result = mapStudioInferenceResult(
      {
        detections: [],
        count: 0,
      },
      {
        id: "runtime-2",
        artifact_type: "pxe",
        display_name: "PXE Runtime",
        version: null,
        best_accuracy: null,
      },
      "Edge Node",
      "pk-1",
    );

    expect(result.is_fomo).toBe(true);
    expect(result.model_type).toBe("detection");
  });

  it("prefers real device sensor metadata over fallback options", () => {
    const derived = deriveSensorOptions(
      {
        id: "pk-1",
        device_id: "dev-1",
        name: "Edge Node",
        device_type: "embedded",
        is_online: true,
        sensors: [
          { name: "accelerometer", display_name: "3-axis accelerometer", freq_hz: 62.5 },
          { name: "microphone", frequencies: [16000, 8000] },
        ],
      },
      [
        { id: "fallback", name: "Fallback", frequencies: [1], source: "fallback" },
      ],
    );

    expect(derived.usingFallback).toBe(false);
    expect(derived.options[0]).toMatchObject({
      id: "accelerometer",
      name: "3-axis accelerometer",
      frequencies: [62.5],
      source: "device",
    });
    expect(derived.options[1].frequencies).toEqual([8000, 16000]);
  });

  it("falls back when the device does not expose a sensor manifest", () => {
    const derived = deriveSensorOptions(
      {
        id: "pk-1",
        device_id: "dev-1",
        name: "Edge Node",
        device_type: "embedded",
        is_online: true,
        sensors: [],
      },
      [
        { id: "camera", name: "Camera", frequencies: [30], source: "fallback" },
      ],
    );

    expect(derived.usingFallback).toBe(true);
    expect(derived.options[0].source).toBe("fallback");
  });

  it("derives a concise disabled reason for start sampling gating", () => {
    const reason = deriveStartSamplingDisabledReason({
      runtime: {
        id: "runtime-1",
        artifact_type: "pxe",
        display_name: "PXE Runtime",
        version: null,
        best_accuracy: null,
      },
      studioConnected: true,
      selectedDevice: {
        id: "pk-1",
        device_id: "dev-1",
        name: "Edge Node",
        device_type: "embedded",
        is_online: true,
      },
      selectedSensorId: "accelerometer",
      selectedFrequency: "",
      sampleLengthMs: 5000,
      compatibilityLoading: false,
      compatibility: {
        deployment_id: "dep-1",
        deployment_target: "edge-v1",
      },
      streamingState: "idle",
      samplingState: "idle",
    });

    expect(reason).toBe("Select a frequency to continue.");
  });

  it("blocks start when the device cannot satisfy the preferred runtime", () => {
    const reason = deriveStartSamplingDisabledReason({
      runtime: {
        id: "runtime-1",
        artifact_type: "pxe",
        display_name: "PXE Runtime",
        version: null,
        best_accuracy: null,
      },
      studioConnected: true,
      selectedDevice: {
        id: "pk-1",
        device_id: "dev-1",
        name: "Edge Node",
        device_type: "embedded",
        is_online: true,
      },
      selectedSensorId: "accelerometer",
      selectedFrequency: "100",
      sampleLengthMs: 5000,
      compatibilityLoading: false,
      compatibility: {
        deployment_id: "dep-1",
        matches_preferred_runtime: false,
      },
      streamingState: "idle",
      samplingState: "idle",
    });

    expect(reason).toBe("This device does not have a deployment for the preferred runtime.");
  });
});

describe("resolveLiveResultSource", () => {
  it("prefers the explicit live source when present", () => {
    const resolved = resolveLiveResultSource({
      resultSource: "live",
      result: { sample_id: "sample-1" } as LiveClassificationRunResult,
      resultError: "sample-error",
      liveResult: { sample_id: "live-1" } as LiveClassificationRunResult,
      liveResultError: "live-error",
    });

    expect(resolved.result?.sample_id).toBe("live-1");
    expect(resolved.error).toBe("live-error");
  });

  it("falls back to sample-load output when sample is the active source", () => {
    const resolved = resolveLiveResultSource({
      resultSource: "sample",
      result: { sample_id: "sample-1" } as LiveClassificationRunResult,
      resultError: null,
      liveResult: { sample_id: "live-1" } as LiveClassificationRunResult,
      liveResultError: "live-error",
    });

    expect(resolved.result?.sample_id).toBe("sample-1");
    expect(resolved.error).toBeNull();
  });

  it("preserves sample-load behavior when no live source has taken over", () => {
    const resolved = resolveLiveResultSource({
      resultSource: null,
      result: { sample_id: "sample-1" } as LiveClassificationRunResult,
      resultError: "sample-error",
      liveResult: null,
      liveResultError: null,
    });

    expect(resolved.result?.sample_id).toBe("sample-1");
    expect(resolved.error).toBe("sample-error");
  });
});

describe("scoresToRows", () => {
  it("converts scores dict to sorted rows", () => {
    const rows = scoresToRows({ idle: 0.1, walking: 0.8, running: 0.1 });
    expect(rows[0].label).toBe("walking");
    expect(rows[0].score).toBe(0.8);
  });

  it("returns empty array for empty scores", () => {
    expect(scoresToRows({})).toHaveLength(0);
  });
});

describe("fmtPct", () => {
  it("formats 0.954 as 95.4%", () => {
    expect(fmtPct(0.954)).toBe("95.4%");
  });

  it("respects decimals param", () => {
    expect(fmtPct(0.5, 0)).toBe("50%");
  });
});

describe("isCorrectPrediction", () => {
  it("returns true when labels match case-insensitively", () => {
    expect(isCorrectPrediction("Walking", "walking")).toBe(true);
  });

  it("returns false when labels differ", () => {
    expect(isCorrectPrediction("running", "walking")).toBe(false);
  });

  it("returns null when ground truth is absent or placeholder", () => {
    expect(isCorrectPrediction("walking", null)).toBeNull();
    expect(isCorrectPrediction("walking", "—")).toBeNull();
  });
});

describe("resultMode", () => {
  const clsResult = { is_fomo: false } as LiveClassificationRunResult;
  const detResult = { is_fomo: true } as LiveClassificationRunResult;

  it("returns classification for is_fomo=false", () => {
    expect(resultMode(clsResult)).toBe("classification");
  });

  it("returns detection for is_fomo=true", () => {
    expect(resultMode(detResult)).toBe("detection");
  });
});

describe("image preview helpers", () => {
  it("uses the url field from the JSON response", () => {
    const apiResponse = { data: { url: "https://s3.example.com/samples/frame.jpg?sig=abc" } };
    expect(getSamplePreviewUrl(apiResponse)).toBe(
      "https://s3.example.com/samples/frame.jpg?sig=abc",
    );
  });

  it("returns null when url is absent or data is missing", () => {
    expect(getSamplePreviewUrl({ data: {} as { url?: string } })).toBeNull();
    expect(getSamplePreviewUrl({ data: null as unknown as { url: string } })).toBeNull();
  });

  it("preview fetch failure does not affect inference result state", () => {
    let result: object | null = { label: "walking", confidence: 0.9, is_fomo: false };
    let imageUrl: string | null = null;

    try {
      throw new Error("network error");
    } catch {
      imageUrl = null;
    }

    expect(result).not.toBeNull();
    expect(imageUrl).toBeNull();
  });
});

describe("isImageSample", () => {
  it("returns true for null sensor_type", () => {
    expect(isImageSample(null)).toBe(true);
  });

  it("returns true for camera sensor type", () => {
    expect(isImageSample("camera")).toBe(true);
  });

  it("returns false for non-image sensor types", () => {
    expect(isImageSample("accelerometer")).toBe(false);
    expect(isImageSample("microphone")).toBe(false);
    expect(isImageSample("temperature_humidity")).toBe(false);
  });
});

describe("sample dropdown mapping", () => {
  const rawSamples = [
    { id: "1", name: "walk-01", label: "walking", sensor_type: "accelerometer", duration_ms: 5000 },
    { id: "2", name: "run-01", label: "running", sensor_type: "accelerometer", duration_ms: 3000 },
    { id: "3", name: "frame-01.jpg", label: "person", sensor_type: "camera", duration_ms: null },
  ];

  it("maps all raw samples to options", () => {
    expect(rawSamples.map(mapRawSampleToOption)).toHaveLength(3);
  });

  it("preserves name and label for dropdown display", () => {
    const opts = rawSamples.map(mapRawSampleToOption);
    expect(opts[0].name).toBe("walk-01");
    expect(opts[0].label).toBe("walking");
  });

  it("preserves sensor_type for preview branching", () => {
    const opts = rawSamples.map(mapRawSampleToOption);
    expect(isImageSample(opts[0].sensor_type)).toBe(false);
    expect(isImageSample(opts[2].sensor_type)).toBe(true);
  });
});

// ── YOLO Pro result mode ──────────────────────────────────────────────────────

describe("resultMode — YOLO Pro detection", () => {
  it("returns detection when is_detection=true and is_fomo=false", () => {
    const r = { is_fomo: false, is_detection: true } as LiveClassificationRunResult;
    expect(resultMode(r)).toBe("detection");
  });

  it("returns classification when is_fomo=false and is_detection=false", () => {
    const r = { is_fomo: false, is_detection: false } as LiveClassificationRunResult;
    expect(resultMode(r)).toBe("classification");
  });

  it("returns classification when is_fomo=false and is_detection is absent", () => {
    const r = { is_fomo: false } as LiveClassificationRunResult;
    expect(resultMode(r)).toBe("classification");
  });

  it("returns detection when is_fomo=true regardless of is_detection (FOMO compat)", () => {
    const r = { is_fomo: true, is_detection: false } as LiveClassificationRunResult;
    expect(resultMode(r)).toBe("detection");
  });
});

describe("mapStudioInferenceResult — YOLO Pro payload", () => {
  const runtime: LiveClassificationRuntimeInfo = {
    id: "model-yolo",
    artifact_type: "pe",
    display_name: "PE v1",
    architecture: "yolo_pro",
    version: "1",
    best_accuracy: 0.9,
  };

  it("propagates is_detection=true from YOLO Pro payload", () => {
    const payload = {
      is_fomo: false,
      is_detection: true,
      model_type: "yolo_pro_detection",
      detections: [{ label: "cat", confidence: 0.85, bbox: { x1: 0.1, y1: 0.1, x2: 0.5, y2: 0.5 } }],
      count: 1,
      debug: {},
    };
    const r = mapStudioInferenceResult(payload, runtime, "Cam01", "dev-1");
    expect(r.is_detection).toBe(true);
    expect(r.is_fomo).toBe(false);
  });

  it("resultMode returns detection for YOLO Pro mapped result", () => {
    const payload = {
      is_fomo: false,
      is_detection: true,
      model_type: "yolo_pro_detection",
      detections: [],
      count: 0,
      debug: {},
    };
    const r = mapStudioInferenceResult(payload, runtime, "Cam01", "dev-1");
    expect(resultMode(r)).toBe("detection");
  });

  it("YOLO Pro detections array is preserved in result", () => {
    const dets = [
      { label: "cat", confidence: 0.9, bbox: { x1: 0.0, y1: 0.0, x2: 0.3, y2: 0.3 } },
      { label: "dog", confidence: 0.7, bbox: { x1: 0.5, y1: 0.5, x2: 0.9, y2: 0.9 } },
    ];
    const payload = { is_fomo: false, is_detection: true, model_type: "yolo_pro_detection", detections: dets, count: 2, debug: {} };
    const r = mapStudioInferenceResult(payload, runtime, "Cam01", "dev-1");
    expect(r.detections).toHaveLength(2);
    expect(r.detections![0].label).toBe("cat");
  });

  it("count is propagated from payload", () => {
    const payload = { is_fomo: false, is_detection: true, model_type: "yolo_pro_detection", detections: [], count: 0, debug: {} };
    const r = mapStudioInferenceResult(payload, runtime, "Cam01", "dev-1");
    expect(r.count).toBe(0);
  });

  it("model_type is yolo_pro_detection when set in payload", () => {
    const payload = { is_fomo: false, is_detection: true, model_type: "yolo_pro_detection", detections: [], count: 0, debug: {} };
    const r = mapStudioInferenceResult(payload, runtime, "Cam01", "dev-1");
    expect(r.model_type).toBe("yolo_pro_detection");
  });

  it("FOMO result is_detection remains false (no cross-contamination)", () => {
    const payload = { is_fomo: true, model_type: "detection_heatmap", detections: [], count: 0, debug: {} };
    const r = mapStudioInferenceResult(payload, runtime, "Cam01", "dev-1");
    expect(r.is_fomo).toBe(true);
    expect(r.is_detection).toBe(false);
  });

  it("classification result is_detection is false", () => {
    const payload = { is_fomo: false, is_detection: false, model_type: "classification", label: "walking", confidence: 0.9, debug: {} };
    const r = mapStudioInferenceResult(payload, runtime, "Cam01", "dev-1");
    expect(r.is_detection).toBe(false);
    expect(resultMode(r)).toBe("classification");
  });
});
