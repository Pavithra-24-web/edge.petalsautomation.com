/**
 * Pure helper functions for the Live Classification feature.
 * Kept separate from the page module so they can be unit-tested
 * without violating Next.js App Router's page export constraints.
 */
import type {
  LiveClassificationCompatibleDeployment,
  LiveClassificationDetection,
  LiveClassificationDeviceOption,
  LiveClassificationRunResult,
  LiveClassificationRuntimeInfo,
  LiveClassificationSensorOption,
  LiveClassificationSampleOption,
  LiveClassificationSamplingState,
  LiveClassificationStreamingState,
  LiveClassificationSummaryRow,
  LiveClassificationViewState,
} from "@/types/live-classification";
import type { StudioEvent } from "@/types/devices";

// ── Bootstrap helpers ─────────────────────────────────────────────────────────

export function mapRawSampleToOption(raw: any): LiveClassificationSampleOption {
  // raw.label may be a plain string (new /inference/sample-options endpoint)
  // or an object { name: "..." } (legacy samplesApi format).
  const labelStr =
    typeof raw.label === "string"
      ? raw.label
      : (raw.label?.name ?? "");

  return {
    id: raw.id,
    name: raw.name || raw.sample_name || raw.filename || "Sample",
    label: labelStr || raw.label_name || raw.expected_outcome || "—",
    sensor_type: raw.sensor_type ?? null,
    duration_ms: raw.duration_ms ?? null,
  };
}

export function resolveViewState(opts: {
  hasProject: boolean;
  hasImpulse: boolean;
  hasModel: boolean;
}): LiveClassificationViewState {
  if (!opts.hasProject) return "no_project";
  if (!opts.hasImpulse) return "no_impulse";
  if (!opts.hasModel)   return "no_model";
  return "ready";
}

export function deriveCanStartSampling(
  deviceId: string,
  sensorId: string,
  frequency: string,
  modelReady: boolean,
): boolean {
  return modelReady && !!deviceId && !!sensorId && !!frequency;
}

export function deriveCanLoadSample(sampleId: string, modelReady: boolean): boolean {
  return modelReady && !!sampleId;
}

export function canStartLiveSampling(
  disabledReason: string | null,
  streamingState: LiveClassificationStreamingState,
): boolean {
  return streamingState === "active" || streamingState === "starting" || disabledReason == null;
}

export function mapPreferredRuntime(raw: any): LiveClassificationRuntimeInfo {
  return {
    id: String(raw.id),
    artifact_type: raw.artifact_type,
    display_name: raw.display_name,
    architecture: raw.architecture ?? undefined,
    version: raw.version ?? null,
    best_accuracy: raw.best_accuracy ?? null,
  };
}

export function buildClassifySampleRequest(
  impulseId: string,
  sampleId: string,
  runtime: LiveClassificationRuntimeInfo | null,
) {
  return {
    impulse_id: impulseId,
    sample_id: sampleId,
    model_id: runtime?.id,
  };
}

export function getSamplePreviewUrl(resp: { data?: { url?: string } | null }): string | null {
  return resp.data?.url ?? null;
}

export function getRuntimeBadgeMeta(runtime: LiveClassificationRuntimeInfo) {
  const artifactLabels: Record<LiveClassificationRuntimeInfo["artifact_type"], string> = {
    pxe: "PXE",
    pe: "PE",
    tflite: "TFLite",
  };

  return {
    artifactChip: artifactLabels[runtime.artifact_type] ?? runtime.artifact_type.toUpperCase(),
    versionChip: runtime.version ? `v${runtime.version}` : null,
    architecture: runtime.architecture ?? null,
    accuracyText:
      runtime.best_accuracy != null
        ? `${(runtime.best_accuracy * 100).toFixed(1)}% acc`
        : null,
  };
}

function asFiniteNumber(value: unknown): number | null {
  const num = typeof value === "number" ? value : Number(value);
  return Number.isFinite(num) && num > 0 ? num : null;
}

function normalizeSensorName(raw: Record<string, unknown>): string {
  const value = raw.name ?? raw.type ?? raw.sensor ?? raw.id;
  return typeof value === "string" && value.trim() ? value.trim() : "sensor";
}

function normalizeSensorLabel(raw: Record<string, unknown>, id: string): string {
  const value = raw.label ?? raw.display_name ?? raw.name ?? raw.type ?? raw.sensor;
  return typeof value === "string" && value.trim() ? value.trim() : id;
}

function extractSensorFrequencies(raw: Record<string, unknown>): number[] {
  const frequencies = [
    raw.freq_hz,
    raw.frequency_hz,
    raw.frequency,
    raw.sample_rate_hz,
    raw.sampleRateHz,
  ]
    .map(asFiniteNumber)
    .filter((value): value is number => value != null);

  const ranges = [
    raw.frequencies,
    raw.supported_frequencies,
    raw.supportedFrequencies,
    raw.sample_rates_hz,
    raw.sampleRatesHz,
  ];
  for (const candidate of ranges) {
    if (!Array.isArray(candidate)) continue;
    for (const item of candidate) {
      const num = asFiniteNumber(item);
      if (num != null) frequencies.push(num);
    }
  }

  return Array.from(new Set(frequencies)).sort((a, b) => a - b);
}

export function deriveSensorOptions(
  device: LiveClassificationDeviceOption | null,
  fallback: LiveClassificationSensorOption[],
): { options: LiveClassificationSensorOption[]; usingFallback: boolean } {
  if (!device) {
    return { options: fallback.map((entry) => ({ ...entry, source: "fallback" })), usingFallback: true };
  }

  const rawSensors = Array.isArray(device.sensors) ? device.sensors : [];
  const options = rawSensors
    .filter((sensor): sensor is Record<string, unknown> => !!sensor && typeof sensor === "object")
    .map((sensor) => {
      const id = normalizeSensorName(sensor);
      return {
        id,
        name: normalizeSensorLabel(sensor, id),
        frequencies: extractSensorFrequencies(sensor),
        source: "device" as const,
      };
    })
    .filter((sensor, index, all) => all.findIndex((entry) => entry.id === sensor.id) === index);

  if (options.length > 0) {
    return { options, usingFallback: false };
  }

  return { options: fallback.map((entry) => ({ ...entry, source: "fallback" })), usingFallback: true };
}

export function applyLiveStudioEventToDevices(
  devices: LiveClassificationDeviceOption[],
  event: StudioEvent,
): LiveClassificationDeviceOption[] {
  if (event.type === "snapshot") {
    const snapshotDevices = (event.payload?.devices ?? event.devices ?? []) as Array<{
      device_id: string;
      mode?: LiveClassificationDeviceOption["mode"];
      connected?: boolean;
    }>;

    return devices.map((device) => {
      const snapshot = snapshotDevices.find((entry) => entry.device_id === device.device_id);
      if (!snapshot) return device;
      return {
        ...device,
        is_online: snapshot.connected ?? device.is_online,
        is_connected: snapshot.connected ?? device.is_connected,
        mode: snapshot.mode ?? device.mode,
      };
    });
  }

  if (!event.device_id) return devices;

  return devices.map((device) => {
    if (device.device_id !== event.device_id) return device;

    if (event.type === "device.connected") {
      return { ...device, is_online: true, is_connected: true, mode: "idle" };
    }
    if (event.type === "device.disconnected") {
      return { ...device, is_online: false, is_connected: false, mode: undefined };
    }
    if (event.type === "device.mode_changed") {
      return { ...device, mode: event.payload?.mode as LiveClassificationDeviceOption["mode"] };
    }
    if (event.type === "sample.started") {
      return { ...device, mode: "sampling" };
    }
    if (event.type === "sample.stopped" || event.type === "sample.failed") {
      return { ...device, mode: "idle" };
    }
    if (event.type === "inference.started") {
      return { ...device, mode: "inference" };
    }
    if (event.type === "inference.stopped" || event.type === "stream.failed") {
      return { ...device, mode: "idle" };
    }
    return device;
  });
}

export function mapStudioInferenceResult(
  payload: Record<string, any>,
  runtime: LiveClassificationRuntimeInfo | null,
  deviceName: string,
  deviceId: string,
): LiveClassificationRunResult {
  const parsedVersion =
    runtime?.version != null && String(runtime.version).trim() !== ""
      ? Number(runtime.version)
      : null;
  const modelVersion = parsedVersion != null && !Number.isNaN(parsedVersion)
    ? parsedVersion
    : null;
  const isFomo = payload.is_fomo != null
    ? Boolean(payload.is_fomo)
    : Array.isArray(payload.detections);
  const isDetection = Boolean(payload.is_detection);

  return {
    sample_id: payload.sample_id ?? `live:${deviceId}`,
    sample_name: payload.sample_name ?? `Live stream · ${deviceName}`,
    ground_truth_label: payload.ground_truth_label ?? null,
    ground_truth_boxes: Array.isArray(payload.ground_truth_boxes)
      ? payload.ground_truth_boxes
      : undefined,
    model_id: payload.model_id ?? runtime?.id ?? `live:${deviceId}`,
    model_version: payload.model_version ?? modelVersion,
    model_architecture: payload.model_architecture ?? runtime?.architecture ?? "device",
    is_fomo: isFomo,
    is_detection: isDetection,
    model_type: payload.model_type ?? ((isFomo || isDetection) ? "detection" : "classification"),
    debug: payload.debug ?? {},
    label: payload.label,
    confidence: payload.confidence,
    scores: payload.scores,
    detections: payload.detections,
    count: payload.count ?? (Array.isArray(payload.detections) ? payload.detections.length : undefined),
    fomo_version: payload.fomo_version,
  };
}

export function resolveLiveResultSource(opts: {
  resultSource: "sample" | "live" | null;
  result: LiveClassificationRunResult | null;
  resultError: string | null;
  liveResult: LiveClassificationRunResult | null;
  liveResultError: string | null;
}) {
  const result = opts.resultSource === "live"
    ? opts.liveResult
    : opts.resultSource === "sample"
      ? opts.result
      : opts.liveResult ?? opts.result;
  const error = opts.resultSource === "live"
    ? opts.liveResultError
    : opts.resultSource === "sample"
      ? opts.resultError
      : opts.liveResultError ?? opts.resultError;

  return { result, error };
}

export function deriveStartSamplingDisabledReason(opts: {
  runtime: LiveClassificationRuntimeInfo | null;
  studioConnected: boolean;
  selectedDevice: LiveClassificationDeviceOption | null;
  selectedSensorId: string;
  selectedFrequency: string;
  sampleLengthMs: number;
  compatibilityLoading: boolean;
  compatibility: LiveClassificationCompatibleDeployment | null;
  streamingState: LiveClassificationStreamingState;
  samplingState: LiveClassificationSamplingState;
}): string | null {
  if (opts.streamingState === "starting" || opts.samplingState === "starting") {
    return "Starting live classification…";
  }
  if (opts.streamingState === "stopping" || opts.samplingState === "stopping") {
    return "Stopping live classification…";
  }
  if (opts.runtime == null) {
    return "No runtime available for this impulse.";
  }
  if (opts.selectedDevice == null) {
    return "Select a device to continue.";
  }
  if (!opts.selectedDevice.is_online) {
    return "Selected device is offline.";
  }
  if (!opts.studioConnected) {
    return "Studio connection is unavailable.";
  }
  if (opts.compatibilityLoading) {
    return "Checking device compatibility…";
  }
  if (opts.compatibility == null) {
    return "No compatible deployment is available for this device.";
  }
  if (opts.compatibility.matches_preferred_runtime === false) {
    return "This device does not have a deployment for the preferred runtime.";
  }
  if (
    opts.selectedDevice.installed_deployment_id &&
    opts.selectedDevice.installed_deployment_id !== opts.compatibility.deployment_id
  ) {
    return "A newer compatible deployment is available but not installed on the device.";
  }
  if (!opts.selectedSensorId) {
    return "Select a sensor to continue.";
  }
  if (!opts.selectedFrequency) {
    return "Select a frequency to continue.";
  }
  if (!Number.isFinite(opts.sampleLengthMs) || opts.sampleLengthMs < 100) {
    return "Sample length must be at least 100 ms.";
  }
  return null;
}

// ── Result formatting helpers (Phase 2) ───────────────────────────────────────

/** Convert the scores dict from the backend to a sorted array for display. */
export function scoresToRows(scores: Record<string, number>): LiveClassificationSummaryRow[] {
  return Object.entries(scores)
    .map(([label, score]) => ({ label, score }))
    .sort((a, b) => b.score - a.score);
}

/** Format a 0-1 confidence as a percentage string ("95.4%"). */
export function fmtPct(value: number, decimals = 1): string {
  return `${(value * 100).toFixed(decimals)}%`;
}

/** Return true when the predicted label matches the ground truth (case-insensitive). */
export function isCorrectPrediction(
  predicted: string | undefined,
  groundTruth: string | null,
): boolean | null {
  if (!predicted || !groundTruth || groundTruth === "—") return null;
  return predicted.trim().toLowerCase() === groundTruth.trim().toLowerCase();
}

/** Infer the result mode from the backend envelope. */
export function resultMode(result: LiveClassificationRunResult): "classification" | "detection" {
  return (result.is_fomo || result.is_detection) ? "detection" : "classification";
}

/**
 * Return true when the sample is likely an image file that can be previewed.
 * Heuristic: null/undefined sensor_type is treated as image, explicitly named
 * sensor types like "microphone" / "accelerometer" are treated as non-image.
 */
export function isImageSample(sensorType: string | null | undefined): boolean {
  if (!sensorType) return true; // unknown → optimistic
  const t = sensorType.toLowerCase();
  const NON_IMAGE = ["microphone", "accelerometer", "gyroscope", "magnetometer",
                     "temperature", "humidity", "pressure", "ultrasonic", "ir"];
  return !NON_IMAGE.some(k => t.includes(k));
}

/**
 * Deterministic, per-label color used for bbox strokes + label chips.
 * Mirrors the palette used by the Dataset/Data-labeling pages so labels
 * read with the same color across the app even though the project-wide
 * label→color map isn't yet centralized.
 */
const LABEL_PALETTE = [
  "#ef4444", "#f97316", "#eab308", "#22c55e", "#06b6d4",
  "#3b82f6", "#8b5cf6", "#ec4899", "#14b8a6", "#a855f7",
];
export function colorForLabel(label: string | null | undefined): string {
  const key = (label ?? "").toLowerCase().trim();
  if (!key) return "#94a3b8";
  let h = 0;
  for (let i = 0; i < key.length; i++) h = (h * 31 + key.charCodeAt(i)) | 0;
  return LABEL_PALETTE[Math.abs(h) % LABEL_PALETTE.length];
}

/**
 * Render a hex-comma preview from arbitrary feature values, truncated for
 * single-line display. Accepts the array/string form that the inference
 * pipeline may stash in `result.debug.raw_features` (or similar).
 */
export function formatRawFeaturesPreview(
  features: unknown,
  maxCount = 16,
): { text: string; truncated: boolean } | null {
  let arr: number[] | null = null;
  if (Array.isArray(features)) {
    arr = features
      .map(v => (typeof v === "number" ? v : typeof v === "string" ? Number(v) : NaN))
      .filter(v => Number.isFinite(v));
  } else if (typeof features === "string") {
    arr = features
      .split(/[,\s]+/)
      .map(s => s.trim())
      .filter(Boolean)
      .map(s => (s.startsWith("0x") || s.startsWith("0X") ? parseInt(s, 16) : Number(s)))
      .filter(v => Number.isFinite(v));
  }
  if (!arr || arr.length === 0) return null;
  const truncated = arr.length > maxCount;
  const head = arr.slice(0, maxCount).map(v => "0x" + Math.abs(Math.trunc(v)).toString(16));
  return { text: head.join(", "), truncated };
}
