/**
 * TypeScript interfaces for the Live Classification feature.
 * Phase 1: shell types. Phase 2: inference result types added.
 */

// ── View state ────────────────────────────────────────────────────────────────

export type LiveClassificationViewState =
  | "loading"     // initial async bootstrap
  | "no_project"  // no active project in store
  | "no_impulse"  // project exists but has no impulse
  | "no_model"    // impulse exists but no trained model; shell still renders
  | "ready";      // impulse + trained model available

// ── Option shapes for dropdowns ───────────────────────────────────────────────

export interface LiveClassificationDeviceOption {
  id: string;
  device_id: string;
  name: string;
  device_type: string;
  is_online: boolean;
  is_connected?: boolean;
  supports_snapshot_streaming?: boolean;
  deployment_target?: string | null;
  device_profile?: string | null;
  installed_deployment_id?: string | null;
  installed_model_version?: string | null;
  sensors?: Array<Record<string, unknown>>;
  device_metadata?: Record<string, unknown>;
  mode?: "idle" | "sampling" | "inference";
}

export interface LiveClassificationCompatibleDeployment {
  deployment_id: string;
  model_id?: string;
  runtime_id?: string;
  artifact_type?: "tflite" | "pe" | "pxe" | null;
  matches_preferred_runtime?: boolean;
  deployment_target?: string;
  device_profile?: string;
  download_url?: string;
}

export type LiveClassificationStudioConnectionState = "disconnected" | "connected";

export type LiveClassificationSamplingState =
  | "idle"
  | "starting"
  | "active"
  | "stopping"
  | "error";

export type LiveClassificationStreamingState =
  | "idle"
  | "starting"
  | "active"
  | "stopping"
  | "error";

export interface LiveClassificationSensorOption {
  id: string;
  name: string;
  /** Available sample rates for this sensor in Hz */
  frequencies: number[];
  source?: "device" | "fallback";
}

export interface LiveClassificationSampleOption {
  id: string;
  /** Display name shown in the dropdown */
  name: string;
  /** Label / class shown in parentheses next to the name */
  label: string;
  /** Raw sensor type string from the backend (null for uploaded files) */
  sensor_type: string | null;
  /** Sample duration in milliseconds */
  duration_ms: number | null;
}

// ── Runtime artifact info (from GET /inference/preferred-runtime) ─────────────

export interface LiveClassificationRuntimeInfo {
  /** Artifact identity: TrainedModel.id for tflite/pe, Deployment.id for pxe */
  id: string;
  artifact_type: "tflite" | "pe" | "pxe";
  display_name: string;
  architecture?: string;
  version?: string | null;
  best_accuracy?: number | null;
}

// ── Model info (legacy — kept for backward compatibility with model-testing UI) ─

export interface LiveClassificationModelInfo {
  id: string;
  architecture: string;
  label_names: string[] | null;
  version: number;
  best_accuracy: number | null;
  input_shape: number[] | null;
}

// ── Inference result types (Phase 2) ─────────────────────────────────────────

export interface LiveClassificationDetection {
  label: string;
  confidence: number;
  bbox: { x1: number; y1: number; x2: number; y2: number };
}

/**
 * Sample-level ground-truth annotation captured during Data Labeling.
 * Coordinates are in PIXELS of the original image (x, y = top-left;
 * w, h = box size). The frontend normalizes against the rendered image's
 * natural width/height when drawing so it matches the prediction overlay.
 */
export interface LiveClassificationGroundTruthBox {
  label: string;
  x: number;
  y: number;
  w: number;
  h: number;
}

export interface LiveClassificationSummaryRow {
  label: string;
  score: number;
}

/** Normalized result envelope returned by POST /inference/classify-sample */
export interface LiveClassificationRunResult {
  // ── Metadata (always present) ─────────────────────────────────────────────
  sample_id: string;
  sample_name: string;
  ground_truth_label: string | null;
  /** GT bounding boxes authored in Data Labeling, in pixel coordinates. */
  ground_truth_boxes?: LiveClassificationGroundTruthBox[];
  model_id: string;
  model_version: number | null;  // null for .pxe artifacts (version lives inside the bundle)
  model_architecture: string;
  is_fomo: boolean;
  /** True for any detection model (YOLO Pro, future non-FOMO detectors). */
  is_detection?: boolean;
  model_type: string;
  debug: Record<string, unknown>;

  // ── Classification fields (is_fomo === false) ─────────────────────────────
  label?: string;
  confidence?: number;
  scores?: Record<string, number>;

  // ── Detection / FOMO fields (is_fomo === true) ────────────────────────────
  detections?: LiveClassificationDetection[];
  count?: number;
  fomo_version?: number;
}

export type LiveClassificationResultMode = "classification" | "detection";

// ── Visualization state ───────────────────────────────────────────────────────

export interface LiveClassificationVisualizationPayload {
  /** Presigned URL for image preview (null when unavailable or non-image sample) */
  sample_image_url: string | null;
  result: LiveClassificationRunResult | null;
  loading: boolean;
  error: string | null;
}

// ── Full page state snapshot (useful for Phase 3 context propagation) ─────────

export interface LiveClassificationPageState {
  viewState: LiveClassificationViewState;
  impulseId: string | null;
  impulseName: string | null;
  runtime: LiveClassificationRuntimeInfo | null;
  modelLoading: boolean;
  modelError: string | null;
  devices: LiveClassificationDeviceOption[];
  devicesLoading: boolean;
  selectedDeviceId: string;
  selectedSensorId: string;
  sampleLengthMs: number;
  selectedFrequency: string;
  compatibility: LiveClassificationCompatibleDeployment | null;
  compatibilityLoading: boolean;
  compatibilityReason: string | null;
  studioConnectionState: LiveClassificationStudioConnectionState;
  samplingState: LiveClassificationSamplingState;
  streamingState: LiveClassificationStreamingState;
  testSamples: LiveClassificationSampleOption[];
  testSamplesLoading: boolean;
  selectedSampleId: string;
  // Phase 2 additions
  loadingResult: boolean;
  result: LiveClassificationRunResult | null;
  resultError: string | null;
  sampleImageUrl: string | null;
  liveResult: LiveClassificationRunResult | null;
  liveResultError: string | null;
}
