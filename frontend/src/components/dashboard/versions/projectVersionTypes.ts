// Mirrors backend/app/schemas/project_versions.py (Phase 1: store + list +
// detail; Phase 2: restore; Phase 3: compare + publish). Project-level
// Versioning (C2) — see docs/Action/parityfix.md. Every shape here is keyed
// by project_id / version_id, never impulse_id.

export interface ProjectVersionSummary {
  id: string;
  project_id: string;
  version_number: number;
  name: string | null;
  description: string | null;
  sample_count: number;
  train_sample_count: number;
  test_sample_count: number;
  class_names: string[];
  impulse_count: number;
  best_accuracy: number | null;
  status: "draft" | "published";
  published_at: string | null;
  created_by: string | null;
  created_by_name: string | null;
  created_at: string;
}

export interface VersionImpulse {
  id: string;
  impulse_id: string | null;
  impulse_name: string;
  dsp_blocks_snapshot: Array<{ type?: string; name?: string; params?: Record<string, any> }>;
  ml_blocks_snapshot: Array<{ type?: string; name?: string; params?: Record<string, any> }>;
  output_config_snapshot: Record<string, any>;
  input_config_snapshot: Record<string, any>;
  trained_model_id: string | null;
  training_job_id: string | null;
  accuracy: number | null;
  final_loss: number | null;
  confusion_matrix: any;
  training_history: any;
  // Self-contained training configuration + artifact/feature-cache pointers
  // (migration 0038) — not rendered anywhere yet, kept here for type parity
  // with the backend's VersionImpulseOut.
  training_config_snapshot: Record<string, any> | null;
  trained_model_formats_snapshot: Array<Record<string, any>> | null;
  features_storage_key: string | null;
}

export interface VersionSample {
  id: string;
  sample_id: string | null;
  sample_name: string;
  label_name: string | null;
  sample_type: string;
}

export interface VersionImpulseDiff {
  impulse_name: string;
  impulse_id: string | null;
  status: "unchanged" | "changed" | "deleted" | "added_since";
  changed_fields: string[];
  accuracy_delta: number | null;
  live_accuracy: number | null;
}

export interface VersionDiffSummary {
  project_config_changed: boolean;
  changed_project_fields: string[];
  deployment_changed: boolean;
  sample_count_delta: number;
  class_names_added: string[];
  class_names_removed: string[];
  impulses: VersionImpulseDiff[];
  best_accuracy_delta: number | null;
  live_best_accuracy: number | null;
}

/** Restore clones the entire source project graph — dataset, impulses,
 * training history, trained models, deployment and post-processing settings
 * — and pins it to this version's configuration. The source project is
 * never modified, so there is nothing to reconcile or report back beyond
 * these diagnostics: `cloned_row_counts`/`cloned_asset_count` show what was
 * duplicated, and `skipped_sample_names` names any samples whose media was
 * deleted from storage after the version was stored — the one case restore
 * cannot make whole again, stated rather than silently absorbed. */
export interface VersionRestoreResponse {
  new_project_id: string;
  new_project_name: string;
  cloned_row_counts: Record<string, number>;
  cloned_asset_count: number;
  skipped_sample_names: string[];
}

/** Compact per-category summary of two project versions (parityfix.md §4.5,
 * §7.3) — no per-impulse array, just the counts the compare dialog renders. */
export interface VersionCompareResponse {
  version_a: ProjectVersionSummary;
  version_b: ProjectVersionSummary;
  project_config_same: boolean;
  deployment_same: boolean;
  sample_count_a: number;
  sample_count_b: number;
  class_count_a: number;
  class_count_b: number;
  impulse_total_a: number;
  impulse_total_b: number;
  impulses_added: number;
  impulses_removed: number;
  impulses_modified: number;
}

export interface ProjectVersionDetail extends ProjectVersionSummary {
  project_config_snapshot: Record<string, any>;
  deployment_snapshot: Record<string, any>;
  post_processing_snapshot: any;
  impulses: VersionImpulse[];
  samples: VersionSample[];
  samples_total: number;
  samples_page: number;
  samples_page_size: number;
  diff_summary: VersionDiffSummary;
}
