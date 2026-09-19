/**
 * Shared TypeScript interfaces for the Model Testing feature.
 * These match the JSON response shapes from /api/v1/model-testing/*.
 */

// ── Enums ─────────────────────────────────────────────────────────────────────

export type TestResultStatus = "pending" | "pass" | "fail" | "uncertain";
export type QuantizationType  = "int8" | "float32" | "eon_compiled";

// ── Model Version ─────────────────────────────────────────────────────────────

export interface ModelVersion {
  id:                string;
  version_number:    string;
  name:              string;
  quantization_type: QuantizationType;
  is_active:         boolean;
  created_at:        string | null;
}

// ── Test Sample ───────────────────────────────────────────────────────────────

export interface PredBox {
  x:     number;
  y:     number;
  x2:    number;
  y2:    number;
  score: number;
  label: string;
}

export interface GtBox {
  x:          number;
  y:          number;
  w:          number;
  h:          number;
  label_id?:  string;
  label?:     string;
  [key: string]: unknown;
}

export interface DetStats {
  tp: number | null;
  fp: number | null;
  fn: number | null;
}

export interface TestSampleRow {
  id:               string;
  sample_id:        string | null;
  sample_name:      string;
  expected_outcome: string;          // "-" if not set
  predicted_class:  string | null;
  precision_score:  number | null;   // backend field name (was f1_score)
  f1_score:         number | null;   // kept for backward compat — same value
  predicted_boxes:  PredBox[] | null;
  gt_boxes:         GtBox[]  | null;
  det_stats:        DetStats | null; // TP/FP/FN — null for non-detection or legacy data
  result_status:    TestResultStatus;
  updated_at:       string | null;
}

// ── Metric ────────────────────────────────────────────────────────────────────

export interface MetricRow {
  metric_name:         string;
  metric_display_name: string;
  metric_value:        number | null;
}

// ── Test Run ──────────────────────────────────────────────────────────────────

export interface TestRun {
  id:             string;
  accuracy:       number | null;
  total_samples:  number;
  passed_samples: number;
  failed_samples: number;
  created_at:     string | null;
}

// ── Page Data ─────────────────────────────────────────────────────────────────

export interface ModelTestingPageData {
  project: {
    id:   string | null;
    name: string | null;
  };
  /** null when no impulse could be resolved for the project */
  impulse: {
    id:         string;
    name:       string;
    project_id: string;
  } | null;
  available_impulses: Array<{ id: string; name: string; project_id: string }>;
  selected_model_version: ModelVersion | null;
  latest_run:             TestRun | null;
  accuracy:               number | null;
  metrics:                MetricRow[];
  summary: {
    total:     number;
    passed:    number;
    failed:    number;
    uncertain: number;
    pending:   number;
  };
  target_device: string;
}

// ── Paginated Test Data ───────────────────────────────────────────────────────

export interface PaginatedTestData {
  items:       TestSampleRow[];
  total:       number;
  page:        number;
  page_size:   number;
  total_pages: number;
}

// ── Classify All Response ─────────────────────────────────────────────────────

export interface ClassifyAllResponse {
  run:     TestRun;
  metrics: MetricRow[];
  samples: TestSampleRow[];
}

// ── Metrics Response ──────────────────────────────────────────────────────────

export interface MetricsResponse {
  run:     TestRun | null;
  metrics: MetricRow[];
  message?: string;
}
