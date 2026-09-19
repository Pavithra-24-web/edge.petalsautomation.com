export interface PPSettings {
  enabled: boolean;
  threshold: number;
  tracking_enabled: boolean;
  keep_grace: number;
  max_observations: number;
  class_filter: string[];
}

export const DEFAULT_SETTINGS: PPSettings = {
  enabled: true,
  threshold: 0.5,
  tracking_enabled: false,
  keep_grace: 3,
  max_observations: 5,
  class_filter: [],
};

export function mergeWithDefaults(data: Partial<PPSettings>): PPSettings {
  return { ...DEFAULT_SETTINGS, ...data };
}

export type ParseResult =
  | { ok: true; detections: any[] }
  | { ok: false; error: string };

export function parsePreviewInput(text: string): ParseResult {
  const trimmed = text.trim();
  if (!trimmed) return { ok: false, error: "Enter detection JSON first" };
  try {
    const parsed = JSON.parse(trimmed);
    const detections = Array.isArray(parsed) ? parsed : parsed?.detections;
    if (!Array.isArray(detections)) {
      return { ok: false, error: "Expected an array or { detections: [...] }" };
    }
    return { ok: true, detections };
  } catch {
    return { ok: false, error: "Invalid JSON" };
  }
}

export function buildPreviewPayload(
  detections: any[],
  trackerState: any,
  useTrackerState: boolean,
): Record<string, any> {
  const payload: Record<string, any> = { detections };
  if (useTrackerState && trackerState != null) {
    payload.tracker_state = trackerState;
  }
  return payload;
}

// Backend response keys for the preview pipeline stages.
export const PREVIEW_STAGES = [
  { key: "input_detections",    label: "input" },
  { key: "filtered_detections", label: "filtered" },
  { key: "post_nms_detections", label: "post-nms" },
  { key: "final_detections",    label: "final" },
] as const;

export function extractApiError(e: any): string {
  return e?.response?.data?.detail || e?.message || "An unexpected error occurred";
}

export type PostProcessingModelSupport =
  | { status: "ready"; message: null }
  | { status: "no_model"; message: string }
  | { status: "unsupported"; message: string };

function _isDetectionModel(outputType: string, architecture: string): boolean {
  const normalizedOutputType = outputType.toLowerCase();
  const normalizedArchitecture = architecture.toLowerCase();
  return (
    normalizedOutputType === "yolo_pro_detection"
    || normalizedOutputType === "ssd_detection"
    || normalizedOutputType === "object_detection"
    || normalizedOutputType === "detection"
    // FOMO is now a first-class supported preview model: the worker decodes
    // its heatmap into per-cell boxes via `decode_fomo_heatmap` and feeds
    // them into the same post-processing pipeline as YOLO-Pro / SSD.
    || normalizedOutputType === "detection_heatmap"
    || normalizedArchitecture.includes("yolo_pro")
    || normalizedArchitecture.includes("ssd")
    || normalizedArchitecture.includes("fomo")
  );
}

function _pickPreferredModel(models: any[]): any | null {
  if (!Array.isArray(models) || models.length === 0) return null;
  return (
    models.find((m: any) => m?.format === "tflite" && m?.model_metadata?.variant === "decoded_float32")
    || models.find((m: any) => m?.format === "tflite")
    || models.find((m: any) => m?.format === "keras")
    || models[0]
  );
}

export function resolvePostProcessingModelSupport(entries: any[]): PostProcessingModelSupport {
  if (!Array.isArray(entries) || entries.length === 0) {
    return {
      status: "no_model",
      message: "Train a Vision Pro, EdgeDetect Lite, or NanoVision detection model before using video post-processing preview.",
    };
  }

  const latestEntry = entries.find((entry: any) => Array.isArray(entry?.models) && entry.models.length > 0);
  if (!latestEntry) {
    return {
      status: "no_model",
      message: "Train a Vision Pro, EdgeDetect Lite, or NanoVision detection model before using video post-processing preview.",
    };
  }

  const model = _pickPreferredModel(latestEntry.models);
  const meta = model?.model_metadata ?? {};
  const outputType = String(model?.output_type ?? meta.output_type ?? "");
  const architecture = String(model?.architecture ?? meta.architecture ?? "");

  // YOLO-Pro / SSD / FOMO (detection_heatmap) all flow through the same
  // post-processing pipeline now; the worker chooses the right decode path.
  if (_isDetectionModel(outputType, architecture)) {
    return { status: "ready", message: null };
  }

  return {
    status: "unsupported",
    message: "This impulse's latest trained model is not compatible with video post-processing preview. Use a Vision Pro, EdgeDetect Lite, or NanoVision detection model.",
  };
}

// ─── Video job helpers ────────────────────────────────────────────────────────

export type VideoJobStatus = "idle" | "uploading" | "processing" | "complete" | "failed" | "cancelled";

// ─── Cross-navigation persistence ────────────────────────────────────────────

/** Shape written to and read from the Zustand persistent store, keyed by projectId. */
export interface PersistedPPState {
  selectedSampleId: string | null;
  jobId: string | null;
  /** null means no job has been started for this project yet. */
  status: VideoJobStatus | null;
  outputUrl: string | null;
  errorMessage: string | null;
}

export const EMPTY_PERSISTED_STATE: PersistedPPState = {
  selectedSampleId: null,
  jobId: null,
  status: null,
  outputUrl: null,
  errorMessage: null,
};

/**
 * Collapse current page state into a PersistedPPState for storage.
 * Idle jobs are stored as status:null so restore skips them cleanly.
 */
export function serializeForPersist(
  selectedSampleId: string | null,
  job: VideoJobState,
): PersistedPPState {
  return {
    selectedSampleId,
    jobId: job.jobId,
    status: job.status === "idle" ? null : job.status,
    outputUrl: job.outputUrl,
    errorMessage: job.errorMessage,
  };
}

/** Returns true for statuses that require a backend re-check on restore. */
export function shouldRestoreAsInProgress(status: VideoJobStatus | null): boolean {
  return status === "processing" || status === "uploading";
}

export interface VideoJobState {
  status: VideoJobStatus;
  jobId: string | null;
  outputUrl: string | null;
  errorMessage: string | null;
}

export const INITIAL_VIDEO_STATE: VideoJobState = {
  status: "idle",
  jobId: null,
  outputUrl: null,
  errorMessage: null,
};

/** Return true for terminal states where polling should stop. */
export function shouldStopPolling(status: string): boolean {
  return status === "complete" || status === "failed" || status === "cancelled";
}

/** Map a backend job status string to the frontend VideoJobStatus. */
export function mapApiJobStatus(apiStatus: string): VideoJobStatus {
  switch (apiStatus) {
    case "pending":
    case "processing":
      return "processing";
    case "complete":
      return "complete";
    case "failed":
      return "failed";
    case "cancelled":
      return "cancelled";
    default:
      return "processing";
  }
}

/** Allowed video MIME types for the file input accept attribute. */
export const ALLOWED_VIDEO_TYPES = "video/mp4,video/avi,video/quicktime,.mp4,.avi,.mov";

// ─── Debug: per-frame JSON validation (used by debug upload endpoint tests) ──

export type ParseVideoResult =
  | { ok: true; detectionsPerFrame: unknown[][] }
  | { ok: false; error: string };

/**
 * Validate that `text` is a JSON array-of-arrays (one inner array per frame).
 * Used by the debug upload path only; the main upload endpoint no longer
 * requires detections JSON from the user.
 */
export function parseVideoDetectionsInput(text: string): ParseVideoResult {
  const trimmed = text.trim();
  if (!trimmed) return { ok: false, error: "Enter detections JSON first" };
  let parsed: unknown;
  try {
    parsed = JSON.parse(trimmed);
  } catch {
    return { ok: false, error: "Invalid JSON" };
  }
  if (!Array.isArray(parsed)) {
    return { ok: false, error: "Expected an array of frame arrays (array-of-arrays)" };
  }
  for (let i = 0; i < parsed.length; i++) {
    if (!Array.isArray(parsed[i])) {
      return {
        ok: false,
        error: `Frame ${i} is not an array — expected [[...frame0], [...frame1], ...]`,
      };
    }
  }
  return { ok: true, detectionsPerFrame: parsed as unknown[][] };
}
