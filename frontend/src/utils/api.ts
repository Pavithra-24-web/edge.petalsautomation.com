/**
 * API client — centralised axios instance with auth token injection
 */
import axios from "axios";

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8010";

const api = axios.create({
  baseURL: `${API_URL}/api/v1`,
  headers: { "Content-Type": "application/json" },
});

// Inject JWT on every request
api.interceptors.request.use((config) => {
  if (typeof window !== "undefined") {
    const token = localStorage.getItem("access_token");
    if (token) config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
});

// Handle 401 globally — redirect to login (but not for the auth endpoints themselves,
// otherwise a failed login reloads the page and swallows the error)
api.interceptors.response.use(
  (res) => res,
  (err) => {
    if (err.response?.status === 401 && typeof window !== "undefined") {
      const url: string = err.config?.url || "";
      const isAuthRequest =
        url.includes("/auth/login") || url.includes("/auth/register") || url.includes("/auth/google");
      if (!isAuthRequest) {
        localStorage.removeItem("access_token");
        window.location.href = "/login";
      }
    }
    return Promise.reject(err);
  }
);

export default api;

// ─── Auth ─────────────────────────────────────────────────────────────────────
export const authApi = {
  register: (data: any) => api.post("/auth/register", data),
  login: (data: any) => api.post("/auth/login", data),
  google: (data: { credential: string }) => api.post("/auth/google", data),
  verifyEmail: (token: string) => api.get("/auth/verify-email", { params: { token } }),
  resendVerification: (email: string) => api.post("/auth/resend-verification", { email }),
  me: () => api.get("/auth/me"),
  apiKeys: () => api.get("/auth/api-keys"),
  createKey: (name: string) => api.post("/auth/api-keys", { name }),
  revokeKey: (id: string) => api.delete(`/auth/api-keys/${id}`),
  deleteAccount: () => api.delete("/auth/me"),
  changePassword: (data: { current_password: string; new_password: string }) =>
    api.post("/auth/password", data),
  forgotPassword: (email: string) => api.post("/auth/forgot-password", { email }),
  resetPassword: (data: { token: string; password: string; confirm_password: string }) =>
    api.post("/auth/reset-password", data),
};

// ─── Demo requests (public, no auth) ───────────────────────────────────────────
export interface DemoRequestPayload {
  name: string;
  mobile: string;
  email: string;
  description: string;
  source: string;
  company?: string;
}
export const demoApi = {
  create: (data: DemoRequestPayload) => api.post("/demo-requests", data),
};

// ─── Contact requests (public, no auth) ────────────────────────────────────────
export type ContactCategory =
  | "Sales inquiries"
  | "Technical support"
  | "Product demo"
  | "Product feedback";
export interface ContactRequestPayload {
  category: ContactCategory;
  name: string;
  mobile: string;
  email: string;
  company: string;
  description: string;
  job_title?: string;
}
export const contactApi = {
  create: (data: ContactRequestPayload) => api.post("/contact-requests", data),
};

// ─── Projects ─────────────────────────────────────────────────────────────────
export const projectsApi = {
  list: () => api.get("/projects/"),
  create: (data: {
    name: string;
    description?: string;
    /** Omitted ⇒ server defaults to "object_detection". Required by the
     *  create dialog, which disables Save until one is chosen. */
    project_type?: "object_detection" | "motion";
  }) => api.post("/projects/", data),
  get: (id: string) => api.get(`/projects/${id}`),
  update: (id: string, data: {
    name?: string;
    description?: string;
    /** null clears the selection; omit to leave it alone. */
    target_device_slug?: string | null;
    /** The four overridable configuration values (Target Device Phase 3).
     *  Same rule on each: null resets that value to the board's own
     *  specification figure, omit to leave it alone. */
    target_device_custom_name?: string | null;
    target_device_ram_kb?: number | null;
    target_device_rom_kb?: number | null;
    target_device_latency_ms?: number | null;
  }) => api.patch(`/projects/${id}`, data),
  delete: (id: string) => api.delete(`/projects/${id}`),
  /** POST to the streaming-zip export endpoint. Returns the axios response
   *  with response.data as a Blob so the caller can trigger a browser download. */
  export: (id: string) =>
    api.post(`/projects/${id}/export`, undefined, { responseType: "blob" }),
  datasetHealth: (id: string) => api.get(`/projects/${id}/dataset-health`),
};

// ─── Jobs (union view across training/deployment/test/labeling) ──────────────
export const jobsApi = {
  list: (projectId: string, page: number = 1, pageSize: number = 20) =>
    api.get(`/projects/${projectId}/jobs`, { params: { page, page_size: pageSize } }),
  /** Read-only persisted logs for a single job. jobType ∈ training | retraining | dsp_feature. */
  logs: (projectId: string, jobType: string, jobId: string) =>
    api.get(`/projects/${projectId}/jobs/${jobType}/${jobId}/logs`),
};

// ─── Labels ───────────────────────────────────────────────────────────────────
export const labelsApi = {
  list: (projectId: string) => api.get(`/labels/project/${projectId}`),
  create: (data: any) => api.post("/labels/", data),
  delete: (id: string) => api.delete(`/labels/${id}`),
  pruneOrphans: (projectId: string) => api.post(`/labels/project/${projectId}/prune-orphans`),
};

// ─── Samples ──────────────────────────────────────────────────────────────────
export const samplesApi = {
  upload: (formData: FormData) => api.post("/samples/upload", formData, {
    headers: { "Content-Type": "multipart/form-data" },
  }),
  uploadBatch: (formData: FormData, onProgress?: (pct: number) => void) =>
    api.post("/samples/upload-batch", formData, {
      headers: { "Content-Type": "multipart/form-data" },
      onUploadProgress: onProgress
        ? (e) => onProgress(Math.round((e.loaded * 100) / (e.total ?? e.loaded + 1)))
        : undefined,
    }),
  list: (projectId: string, params?: any) => api.get(`/samples/project/${projectId}`, { params }),
  /** Distinct *assigned* labels for a project. Returns `{count, names}` —
   *  scans server-side instead of forcing the client to download every sample
   *  row to compute distinct classes. Used by the Output features cards on
   *  the impulse builder. */
  labelsSummary: (projectId: string) => api.get(`/samples/project/${projectId}/labels-summary`),
  /** Dataset feature/axis names parsed from uploaded CSVs (or device-declared
   *  sensor names) — the source of truth a Motion impulse's first processing
   *  block reads its input features from. */
  featureAxes: (projectId: string) => api.get(`/samples/project/${projectId}/feature-axes`),
  /** Canonical labeling-status counts for a project (Labeling Queue plan,
   *  Phase 1 §1.F / Phase 3 §3.F) — powers the Dataset page's outstanding-work
   *  readouts without a full sample-metadata fetch. */
  labelingStatusSummary: (projectId: string) =>
    api.get(`/samples/project/${projectId}/labeling-status-summary`),
  /** Ordered id list of every unlabeled, annotatable sample in the project,
   *  across all splits (Labeling Queue plan, Phase 3 §3.A). Powers the
   *  Labeling Queue page; omits extra_metadata by design — a sample's boxes
   *  are fetched via `get()` only once it becomes the current queue item. */
  unlabeledQueue: (projectId: string) =>
    api.get(`/samples/project/${projectId}/unlabeled-queue`),
  get: (id: string) => api.get(`/samples/${id}`),
  download: (id: string) => api.get(`/samples/${id}/download`),
  signal: (id: string) => api.get(`/samples/${id}/signal`),
  assignLabel: (id: string, labelId: string) => api.patch(`/samples/${id}/label`, { label_id: labelId }),
  updateSplit: (id: string, sampleType: string) => api.patch(`/samples/${id}/split`, { sample_type: sampleType }),
  update: (id: string, data: any) => api.patch(`/samples/${id}`, data),
  delete: (id: string) => api.delete(`/samples/${id}`),
  bulkUpdate: (sampleIds: string[], action: string, value?: any) =>
    api.post("/samples/bulk-update", { sample_ids: sampleIds, action, value }),
  bulkDelete: (sampleIds: string[]) =>
    api.post("/samples/bulk-delete", { sample_ids: sampleIds }),
};

// ─── Impulses ─────────────────────────────────────────────────────────────────
export const impulsesApi = {
  list: (projectId: string) => api.get(`/impulses/project/${projectId}`),
  /** Peek the next default impulse number for a project — pure read, does not
   *  advance the backend counter. Used by the builder to show the canonical
   *  "Impulse N" header the moment the user clicks New impulse. */
  nextNumber: (projectId: string) =>
    api.get<{ next_number: number; next_name: string }>(
      `/impulses/project/${projectId}/next-number`,
    ),
  get: (id: string) => api.get(`/impulses/${id}`),
  create: (data: any) => api.post("/impulses/", data),
  update: (id: string, data: any, options?: { saveParameters?: boolean }) =>
    api.put(`/impulses/${id}`, data, {
      params: options?.saveParameters ? { save_parameters: true } : undefined,
    }),
  patch: (id: string, data: any) => api.patch(`/impulses/${id}`, data),
  delete: (id: string) => api.delete(`/impulses/${id}`),
  bulkDelete: (ids: string[]) => api.post("/impulses/bulk-delete", { impulse_ids: ids }),

  /** Add or remove a single DSP/ML block without touching other fields */
  patchBlocks: (impulseId: string, data: {
    action: "add" | "remove",
    block_kind: "dsp" | "ml",
    block?: any,
    block_index?: number,
    // Optional root fields to sync
    image_width?: number,
    image_height?: number,
    window_size_ms?: number,
    window_increase_ms?: number,
    frequency_hz?: number,
    name?: string,
  }) => api.patch(`/impulses/${impulseId}/blocks`, data),

  /** Validate impulse readiness before training */
  validate: (impulseId: string) => api.post(`/impulses/${impulseId}/validate`),
};

// ─── DSP ──────────────────────────────────────────────────────────────────────
// Shape of GET /dsp/dataset-summary (backend/app/api/v1/endpoints/dsp.py).
// window_count/skipped_too_short aren't returned by that endpoint today —
// GenerateFeaturesShell falls back to "Not available" when they're absent —
// kept optional here so a future motion-specific summary can add them.
export interface DatasetSummaryResponse {
  total_samples: number;
  test_samples: number;
  num_classes: number;
  class_names: string[];
  background_samples: number;
  window_count?: number;
  skipped_too_short?: number;
}

export const dspApi = {
  /** Legacy — full unfiltered block list */
  blocks: () => api.get("/dsp/blocks"),

  /** EI-style: project-type-aware processing block catalog */
  processingBlocks: (impulseId?: string, showAll = false, projectType?: string) =>
    api.get("/dsp/processing-blocks", {
      params: {
        ...(impulseId ? { impulse_id: impulseId } : {}),
        ...(showAll ? { show_all: true } : {}),
        ...(projectType ? { project_type: projectType } : {}),
      },
    }),

  /** EI-style: project-type-aware learning block catalog */
  learningBlocks: (impulseId?: string, showAll = false, projectType?: string) =>
    api.get("/dsp/learning-blocks", {
      params: {
        ...(impulseId ? { impulse_id: impulseId } : {}),
        ...(showAll ? { show_all: true } : {}),
        ...(projectType ? { project_type: projectType } : {}),
      },
    }),

  extract: (data: any) => api.post("/dsp/extract", data),
  preview: (data: any) => api.post("/dsp/preview", data),
  datasetSummary: (impulseId: string) => api.get(`/dsp/dataset-summary?impulse_id=${impulseId}`),
  generateFeatures: (data: any) => api.post("/dsp/generate-features", data),
  cancel: (jobId: string) => api.post(`/dsp/${jobId}/cancel`),
  jobStatus: (jobId: string) => api.get(`/dsp/job-status/${jobId}`),
  getFeatures: (impulseId: string) => api.get(`/dsp/features/${impulseId}`),
  /** Fast HEAD-only readiness check (no PCA). Use for gating UI, not for plots. */
  featuresReady: (impulseId: string) => api.get(`/dsp/features-ready/${impulseId}`),
  /** Raw feature matrix (X) as a `.npy` file, mirroring projectsApi.export's blob pattern. */
  downloadFeatures: (impulseId: string) =>
    api.get(`/dsp/features/${impulseId}/download`, { responseType: "blob" }),
  /** Returns the real DSP feature count for an impulse by running a representative sample. */
  inputSize: (impulseId: string) => api.get(`/dsp/input-size?impulse_id=${impulseId}`),
};

// ─── Training ─────────────────────────────────────────────────────────────────
export const trainingApi = {
  start: (data: any) => api.post("/training/start", data),
  retrain: (data: any) => api.post("/training/retrain", data),

  get: (id: string) => api.get(`/training/${id}`),
  getJob: (id: string) => api.get(`/training/${id}`),          // alias
  listForImpulse: (id: string) => api.get(`/training/impulse/${id}`),
  listJobs: (id: string) => api.get(`/training/impulse/${id}`),  // alias
  impulseStatus: (impulseId: string) => api.get(`/training/impulse/${impulseId}/status`),

  cancel: (id: string) => api.post(`/training/${id}/cancel`),
  cancelJob: (id: string) => api.post(`/training/${id}/cancel`),       // alias
};

// ─── Evaluation ───────────────────────────────────────────────────────────────
export const evaluationApi = {
  confusionMatrix: (jobId: string) => api.get(`/evaluation/job/${jobId}/confusion-matrix`),
  metrics: (jobId: string) => api.get(`/evaluation/job/${jobId}/metrics`),
  forImpulse: (impulseId: string) => api.get(`/evaluation/impulse/${impulseId}`),
  classifyTestSet: (data: any) => api.post("/evaluation/classify-test-set", data),
};

// ─── Deployment ───────────────────────────────────────────────────────────────
export const deploymentApi = {
  // With no params: the static format catalog, unchanged. With `model_id`
  // (optionally + `device_profile`): the Target Device Phase 6 package-policy
  // answer for that model/device — one offered package, everything else
  // withheld with a reason.
  targets: (params?: { model_id?: string; device_profile?: string }) =>
    api.get("/deployment/targets", { params }),
  // The §1 package policy for all six deploy targets, with no ids — what
  // PetalEdge offers per target in the abstract, not per model/device. Reads
  // the same `compatibility.py` table `targets` above does.
  packagePolicy: () => api.get("/deployment/package-policy"),
  build: (data: any) => api.post("/deployment/build", data),
  // Read-only twin of the gate in POST /deployment/build — same service, same
  // answer, nothing enqueued. Used to disable an impossible combination before
  // the user starts a build (Target Device Phase 4).
  compatibility: (params: { model_id: string; format: string; device_profile?: string }) =>
    api.get("/deployment/compatibility", { params }),
  // Flash / RAM / latency estimate for a model × format × device combination
  // (Target Device Phase 5). Same read-only shape as `compatibility` above —
  // nothing enqueued, safe to call on every selection change.
  estimate: (params: { model_id: string; format: string; device_profile?: string }) =>
    api.get("/deployment/estimate", { params }),
  get: (id: string) => api.get(`/deployment/${id}`),
  cancel: (id: string) => api.post(`/deployment/${id}/cancel`),
  forModel: (modelId: string) => api.get(`/deployment/model/${modelId}`),
  listForModel: (modelId: string) => api.get(`/deployment/model/${modelId}`), // alias
};

// ─── Devices ──────────────────────────────────────────────────────────────────
export const devicesApi = {
  list: (projectId: string) => api.get(`/devices/project/${projectId}`),
  register: (data: any) => api.post("/devices/register", data),
  get: (pk: string) => api.get(`/devices/${pk}`),
  listProject: (projectId: string) => api.get(`/devices/project/${projectId}`),
  socketToken: (projectId: string) => api.post(`/devices/project/${projectId}/socket-token`, {}),
  listKeys: (projectId: string) => api.get(`/devices/project/${projectId}/keys`),
  createKey: (projectId: string, name: string) => api.post(`/devices/project/${projectId}/keys`, { name }),
  revokeKey: (projectId: string, keyId: string) => api.delete(`/devices/project/${projectId}/keys/${keyId}`),
  compatibleDeployment: (pk: string, impulseId?: string) => api.get(`/devices/${pk}/compatible-deployment`, {
    params: impulseId ? { impulse_id: impulseId } : undefined,
  }),
  requestUpdate: (pk: string, force = false) => api.post(`/devices/${pk}/request-update`, { force }),
  updateStatus: (pk: string) => api.get(`/devices/${pk}/update-status`),
  streamSnapshotStart: (pk: string) => api.post(`/devices/${pk}/streams/snapshot/start`, {}),
  streamInferenceStart: (pk: string, fomoThreshold?: number, opts?: { sensor?: string; frequency?: number; sample_length_ms?: number }) =>
    api.post(`/devices/${pk}/streams/inference/start`, {
      ...(fomoThreshold != null ? { fomo_threshold: fomoThreshold } : {}),
      ...(opts?.sensor != null ? { sensor: opts.sensor } : {}),
      ...(opts?.frequency != null ? { frequency: opts.frequency } : {}),
      ...(opts?.sample_length_ms != null ? { sample_length_ms: opts.sample_length_ms } : {}),
    }),
  streamKeepalive: (pk: string) => api.post(`/devices/${pk}/streams/keepalive`, {}),
  streamStop: (pk: string) => api.post(`/devices/${pk}/streams/stop`, {}),
  startSampling: (
    pk: string,
    body: {
      label: string;
      length_ms: number;
      frequency: number;
      /** Sensor name as reported by the device (e.g. "accelerometer"). */
      sensor?: string;
      /** Sample category → ingestion split: training | testing | post-processing | anomaly. */
      category?: string;
    },
  ) => api.post(`/devices/${pk}/start-sampling`, body),
  /** Device-authenticated (x-api-key, not the user's JWT) — the "any device
   *  type" ONLINE contract at `POST /devices/{pk}/heartbeat`. A browser-side
   *  Web Serial connection is a device client too, so it uses the same route
   *  real firmware does rather than a browser-only endpoint.
   *
   *  Deliberately bypasses the shared `api` instance: its response
   *  interceptor treats *any* 401 as "the user's session expired" and force
   *  -logs them out. A revoked or not-yet-committed device key returning 401
   *  here is a device-connection concern, not a user-session one — routing
   *  it through `api` would log the whole browser tab out over a device key
   *  problem unrelated to the signed-in user. */
  heartbeat: (
    pk: string,
    apiKey: string,
    body: { firmware_version?: string | null; protocol_version?: string | null } = {},
  ) =>
    axios.post(`${API_URL}/api/v1/devices/${pk}/heartbeat`, body, {
      headers: { "x-api-key": apiKey, "Content-Type": "application/json" },
    }),
  studioWsUrl: (): string => {
    const base = (process.env.NEXT_PUBLIC_API_URL || "http://localhost:8010").replace(/^http/, "ws");
    return `${base}/ws/studio`;
  },
  /** The one device-facing WebSocket — project is resolved server-side from
   *  the hello message's apiKey, so this URL carries no path parameter. */
  deviceWsUrl: (): string => {
    const base = (process.env.NEXT_PUBLIC_API_URL || "http://localhost:8010").replace(/^http/, "ws");
    return `${base}/ws/device`;
  },
};

// ─── Device-client packages (contract §14) ────────────────────────────────────
export const deviceClientApi = {
  /** Manifest JSON for the newest published version. 404 if none published. */
  latest: () => api.get("/device-client/latest"),
  /** Manifest JSON for a specific version. */
  version: (version: string) => api.get(`/device-client/${version}`),
};

// ─── Device catalog (Target Device Phase 1 / 1b / 2) ───────────────────────────
export const deviceCatalogApi = {
  /** Flat list of known hardware — identity fields only, unless `include:
   *  "specification"` embeds each entry's full Phase 2 specification.
   *  Optionally filtered by family, a case-insensitive search term over
   *  board name / family, and/or device class — all three compose. */
  list: (params?: { family?: string; q?: string; device_class?: string; include?: "specification" }) =>
    api.get("/device-catalog/", {
      params: params && Object.values(params).some(Boolean) ? params : undefined,
    }),
  /** Known hardware grouped by device family, identity fields only unless
   *  `include: "specification"` embeds each entry's full Phase 2 spec — the
   *  target-device configuration dialog needs that to populate a board's
   *  values the moment it's picked, with no second round trip. */
  families: (params?: { include?: "specification" }) =>
    api.get("/device-catalog/families", { params }),
  /** Single catalog entry by slug, with its full specification inline. 404 if unknown. */
  get: (slug: string) => api.get(`/device-catalog/${slug}`),
};

// ─── Inference ────────────────────────────────────────────────────────────────
export const inferenceApi = {
  predict: (data: any) => api.post("/inference/predict", data),
  wsUrl: (modelId: string) => {
    const baseUrl = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8010";
    const wsBase = baseUrl.replace(/^http/, "ws");
    return `${wsBase}/api/v1/inference/ws/${modelId}`;
  },
};

// ─── Trained Models ───────────────────────────────────────────────────────────
export const trainedModelsApi = {
  get: (id: string) => api.get(`/trained-models/${id}`),
  listForJob: (jobId: string) => api.get(`/trained-models/job/${jobId}`),
  listAllForImpulse: (impulseId: string) => api.get(`/trained-models/impulse/${impulseId}/all`),
  latestForImpulse: (impulseId: string, format = "tflite", variant?: string) =>
    api.get(`/trained-models/impulse/${impulseId}/latest`, {
      params: { format, ...(variant ? { variant } : {}) },
    }),
  panel: (impulseId: string, variant = "int8", engine = "tflite", threshold?: number) =>
    api.get(`/trained-models/impulse/${impulseId}/panel`, {
      params: { variant, engine, ...(threshold !== undefined ? { threshold } : {}) },
    }),
  delete: (id: string) => api.delete(`/trained-models/${id}`),
  /** Raw model file blob, mirroring projectsApi.export's blob pattern. */
  download: (id: string) => api.get(`/trained-models/${id}/download`, { responseType: "blob" }),
};

// ─── Versioning (C2) — project-level ────────────────────────────────────────
// See docs/Action/parityfix.md for the full design. Phase 1: store + list +
// get. Phase 2 (this file also now includes): restore. Publish/compare land
// in a later phase (parityfix.md §8). Every method takes a project_id or a
// version_id — none takes an impulse_id.
export const projectVersionsApi = {
  create: (data: { project_id: string; name?: string; description?: string }) =>
    api.post("/project-versions", data),
  list: (projectId: string) =>
    api.get("/project-versions", { params: { project_id: projectId } }),
  get: (versionId: string, page = 1, pageSize = 50) =>
    api.get(`/project-versions/${versionId}`, { params: { page, page_size: pageSize } }),
  restore: (versionId: string, data: { name: string; description?: string }) =>
    api.post(`/project-versions/${versionId}/restore`, data),
  compare: (versionId: string, otherVersionId: string) =>
    api.get(`/project-versions/${versionId}/compare/${otherVersionId}`),
  publish: (versionId: string) =>
    api.post(`/project-versions/${versionId}/publish`),
};

// ─── Post-Processing ──────────────────────────────────────────────────────────
export const postProcessingApi = {
  // Legacy model-based API (kept for backward compatibility)
  latestModel: (impulseId: string) => api.get(`/post-processing/impulse/${impulseId}/latest-model`),
  getConfig: (modelId: string) => api.get(`/post-processing/model/${modelId}`),
  updateConfig: (modelId: string, data: any) => api.put(`/post-processing/model/${modelId}`, data),
  apply: (data: any) => api.post("/post-processing/apply", data),

  // Project-scoped settings API (Phase 4+)
  getSettings: (projectId: string, impulseId?: string) =>
    api.get(`/projects/${projectId}/post-processing-settings`, {
      params: impulseId ? { impulse_id: impulseId } : undefined,
    }),
  updateSettings: (projectId: string, data: any, impulseId?: string) =>
    api.put(`/projects/${projectId}/post-processing-settings`, data, {
      params: impulseId ? { impulse_id: impulseId } : undefined,
    }),
  runPreview: (projectId: string, payload: any, impulseId?: string) =>
    api.post(`/projects/${projectId}/post-processing-preview`, payload, {
      params: impulseId ? { impulse_id: impulseId } : undefined,
    }),

  // Video processing API (Phase 6)
  uploadPostProcessingVideo: (projectId: string, file: File, impulseId?: string) => {
    const form = new FormData();
    form.append("file", file);
    if (impulseId) form.append("impulse_id", impulseId);
    return api.post(`/projects/${projectId}/post-processing-video`, form, {
      headers: { "Content-Type": "multipart/form-data" },
    });
  },
  uploadPostProcessingVideoDebug: (projectId: string, file: File, detectionsJson: string) => {
    const form = new FormData();
    form.append("file", file);
    form.append("detections_json", detectionsJson);
    return api.post(`/projects/${projectId}/post-processing-video-debug`, form, {
      headers: { "Content-Type": "multipart/form-data" },
    });
  },
  getProcessingJob: (projectId: string, jobId: string) =>
    api.get(`/projects/${projectId}/post-processing-jobs/${jobId}`),
  getProcessingJobVideoUrl: (projectId: string, jobId: string) =>
    api.get<{ url: string }>(`/projects/${projectId}/post-processing-jobs/${jobId}/video`),
  cancelProcessingJob: (projectId: string, jobId: string) =>
    api.post(`/projects/${projectId}/post-processing-jobs/${jobId}/cancel`),

  // Phase 7 — trigger from existing project sample (no re-upload)
  triggerPostProcessingFromSample: (projectId: string, sampleId: string, impulseId?: string) =>
    api.post(`/projects/${projectId}/post-processing-video-from-sample`, {
      sample_id: sampleId,
      impulse_id: impulseId ?? null,
    }),
};

// ─── Model Testing ────────────────────────────────────────────────────────────
export const modelTestingApi = {
  /**
   * Full page bootstrap — project, impulse, active version, accuracy, metrics, sample summary.
   * At least one of impulseId / projectId should be supplied; the backend auto-resolves.
   * Never returns a 404 — always a valid envelope.
   */
  pageData: (impulseId?: string | null, projectId?: string | null) =>
    api.get("/model-testing/page-data", {
      params: {
        ...(impulseId ? { impulse_id: impulseId } : {}),
        ...(projectId ? { project_id: projectId } : {}),
      },
    }),

  /** Paginated test data rows for the sample table */
  testData: (impulseId: string, page = 1, pageSize = 20) =>
    api.get("/model-testing/test-data", {
      params: { impulse_id: impulseId, page, page_size: pageSize },
    }),

  /** All model versions for the version dropdown */
  modelVersions: (impulseId: string) =>
    api.get("/model-testing/model-versions", { params: { impulse_id: impulseId } }),

  /** Trigger 'Classify all' — simulates classification & returns refreshed data */
  classifyAll: (impulseId: string, modelVersionId?: string) =>
    api.post("/model-testing/classify-all", {
      impulse_id: impulseId,
      model_version_id: modelVersionId ?? null,
    }),

  /** Poll async classify-all status until the worker completes */
  classifyAllStatus: (runId: string) =>
    api.get(`/model-testing/classify-all/${runId}/status`),

  /** Update expected_outcome or result_status for a single test sample */
  updateSample: (sampleId: string, data: {
    expected_outcome?: string;
    result_status?: "pending" | "pass" | "fail";
  }) => api.patch(`/model-testing/test-data/${sampleId}`, data),

  /** Latest test run metrics (Precision / Recall / F1 non-background) */
  metrics: (impulseId: string) =>
    api.get("/model-testing/metrics", { params: { impulse_id: impulseId } }),
};

// ─── Live Classification ──────────────────────────────────────────────────────
export const liveClassificationApi = {
  /**
   * Best available runtime artifact for the impulse (PXE > PE > TFLite).
   * Single bootstrap call — replaces the old hardcoded tflite→pe chain.
   * Returns 404 when no artifact exists yet.
   */
  preferredRuntime: (impulseId: string) =>
    api.get("/inference/preferred-runtime", { params: { impulse_id: impulseId } }),

  /**
   * Sample options scoped to the impulse via FeatureSet linkage (test split only).
   * Returns richer metadata than samplesApi.list — includes sensor_type & duration_ms.
   */
  sampleOptions: (impulseId: string) =>
    api.get("/inference/sample-options", { params: { impulse_id: impulseId } }),

  /** Run inference on an existing test sample via the real runtime artifact. */
  classifySample: (body: {
    impulse_id: string;
    sample_id: string;
    model_id?: string;
    fomo_threshold?: number;
  }) => api.post("/inference/classify-sample", body),

  /**
   * Fetch the presigned download URL for a sample (for image preview).
   * Backend returns { url: string } JSON — use response.data.url as img src directly.
   * No Blob / createObjectURL needed; the presigned URL is usable as-is.
   */
  sampleImageUrl: (sampleId: string) =>
    api.get<{ url: string }>(`/samples/${sampleId}/download`),
};

// ─── AI Labeling ──────────────────────────────────────────────────────────────
export const aiLabelingApi = {
  // Actions CRUD
  createAction: (data: any) => api.post("/ai-labeling/actions", data),
  listActions: (projectId: string) => api.get(`/ai-labeling/actions/project/${projectId}`),
  getAction: (id: string) => api.get(`/ai-labeling/actions/${id}`),
  updateAction: (id: string, data: any) => api.patch(`/ai-labeling/actions/${id}`, data),
  deleteAction: (id: string) => api.delete(`/ai-labeling/actions/${id}`),

  // Job execution
  run: (data: any) => api.post("/ai-labeling/run", data),
  getJob: (id: string) => api.get(`/ai-labeling/jobs/${id}`),
  listJobs: (projectId: string) => api.get(`/ai-labeling/jobs/project/${projectId}`),
  getPredictions: (jobId: string, params?: any) =>
    api.get(`/ai-labeling/jobs/${jobId}/predictions`, { params }),

  // Edit a single prediction's bounding boxes / label
  updatePrediction: (predictionId: string, data: any) =>
    api.patch(`/ai-labeling/predictions/${predictionId}`, data),

  // Apply / Reject
  apply: (data: any) => api.post("/ai-labeling/apply", data),
  reject: (data: any) => api.post("/ai-labeling/reject", data),
};

export const syntheticDataApi = {
  config: () => api.get("/synthetic-data/config"),
  generate: (data: any) => api.post("/synthetic-data/generate", data),
  getJob: (id: string) => api.get(`/synthetic-data/jobs/${id}`),
  listJobs: (projectId: string) => api.get(`/synthetic-data/jobs/project/${projectId}`),
};
