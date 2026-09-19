"use client";

import { useEffect, useState } from "react";
import { RefreshCw, ChevronLeft, ChevronRight, Boxes, Sliders, ImageIcon, Cpu, RotateCcw } from "lucide-react";
import { projectVersionsApi } from "@/utils/api";
import VersionRestoreDialog from "./VersionRestoreDialog";
import type { ProjectVersionDetail, VersionImpulse } from "./projectVersionTypes";

interface ProjectVersionDetailPanelProps {
  versionId: string;
}

const PAGE_SIZE = 25;

function blockLabel(block: { type?: string; name?: string }) {
  return block.name || block.type || "Block";
}

function formatDateTime(iso: string) {
  const d = new Date(iso);
  return d.toLocaleDateString(undefined, { month: "short", day: "numeric", year: "numeric" }) +
    " · " + d.toLocaleTimeString(undefined, { hour: "numeric", minute: "2-digit" });
}

function ImpulseSection({ impulse }: { impulse: VersionImpulse }) {
  return (
    <div className="pe-pver-impulse-card">
      <div className="pe-pver-impulse-head">
        <span className="pe-pver-impulse-name">{impulse.impulse_name}</span>
        {impulse.trained_model_id ? (
          <span className="pe-version-diff-chip">
            {impulse.accuracy !== null ? `${(impulse.accuracy * 100).toFixed(1)}% accuracy` : "Trained"}
          </span>
        ) : (
          <span className="pe-pver-impulse-untrained">No trained model at snapshot time</span>
        )}
      </div>

      <div className="pe-version-detail-grid">
        <section className="pe-version-detail-section">
          <h4 className="pe-version-detail-label">
            <Sliders size={12} /> Processing blocks
          </h4>
          {impulse.dsp_blocks_snapshot.length === 0 ? (
            <p className="pe-version-empty-note">None configured</p>
          ) : (
            <div className="pe-version-chip-row">
              {impulse.dsp_blocks_snapshot.map((b, i) => (
                <span key={i} className="pe-version-block-chip">{blockLabel(b)}</span>
              ))}
            </div>
          )}

          <h4 className="pe-version-detail-label pe-version-detail-label--spaced">
            <Boxes size={12} /> Learning blocks
          </h4>
          {impulse.ml_blocks_snapshot.length === 0 ? (
            <p className="pe-version-empty-note">None configured</p>
          ) : (
            <div className="pe-version-chip-row">
              {impulse.ml_blocks_snapshot.map((b, i) => (
                <span key={i} className="pe-version-block-chip pe-version-block-chip--ml">{blockLabel(b)}</span>
              ))}
            </div>
          )}
        </section>

        <section className="pe-version-detail-section">
          <h4 className="pe-version-detail-label">
            <ImageIcon size={12} /> Input configuration
          </h4>
          <dl className="pe-version-kv">
            {impulse.input_config_snapshot.input_type === "image" ? (
              <>
                <dt>Image size</dt>
                <dd>{impulse.input_config_snapshot.image_width}×{impulse.input_config_snapshot.image_height}</dd>
              </>
            ) : (
              <>
                <dt>Window size</dt>
                <dd>{impulse.input_config_snapshot.window_size_ms} ms</dd>
              </>
            )}
            <dt>Sensor</dt>
            <dd>{impulse.input_config_snapshot.sensor_type || "—"}</dd>
            <dt>Training data used</dt>
            <dd>{impulse.input_config_snapshot.train_subset_percent}%</dd>
          </dl>
        </section>
      </div>
    </div>
  );
}

/** The detail view for one project version, reached from the history
 * table's row name (parityfix.md §7.3). Fetches project config, every
 * impulse's block/training snapshot, the dataset manifest and a diff
 * against the project's current live state. */
export default function ProjectVersionDetailPanel({ versionId }: ProjectVersionDetailPanelProps) {
  const [detail, setDetail] = useState<ProjectVersionDetail | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(false);
  const [page, setPage] = useState(1);
  const [restoreOpen, setRestoreOpen] = useState(false);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError(false);
    projectVersionsApi
      .get(versionId, page, PAGE_SIZE)
      .then(({ data }) => {
        if (!cancelled) setDetail(data);
      })
      .catch(() => {
        if (!cancelled) setError(true);
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [versionId, page]);

  if (loading && !detail) {
    return (
      <div className="pe-version-loading">
        <RefreshCw size={16} className="pe-version-spin" /> Loading snapshot…
      </div>
    );
  }

  if (error || !detail) {
    return <div className="pe-version-loading pe-version-loading--error">Could not load this version.</div>;
  }

  const { diff_summary: diff } = detail;
  const totalPages = Math.max(1, Math.ceil(detail.samples_total / detail.samples_page_size));
  const deployment = detail.deployment_snapshot || {};

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="pe-version-card">
        <div className="pe-version-toolbar">
          <div>
            <span className="pe-version-count">
              v{detail.version_number} · {formatDateTime(detail.created_at)}
              {detail.created_by_name ? ` · ${detail.created_by_name}` : ""}
            </span>
          </div>
          {detail.status === "published" && <span className="pe-version-published-badge">Published</span>}
        </div>
        <div className="pe-version-detail" style={{ paddingTop: "1rem" }}>
          <h2 className="pe-versions-title" style={{ fontSize: "1.3rem" }}>
            {detail.name || "Untitled snapshot"}
          </h2>
          {detail.description && <p className="pe-versions-sub">{detail.description}</p>}

          <div className="pe-version-detail-grid" style={{ marginTop: "1rem" }}>
            <section className="pe-version-detail-section">
              <h4 className="pe-version-detail-label">Project configuration</h4>
              <dl className="pe-version-kv">
                <dt>Name</dt>
                <dd>{detail.project_config_snapshot.name}</dd>
                <dt>Type</dt>
                <dd>{detail.project_config_snapshot.project_type}</dd>
                {detail.project_config_snapshot.description && (
                  <>
                    <dt>Description</dt>
                    <dd>{detail.project_config_snapshot.description}</dd>
                  </>
                )}
              </dl>
            </section>
            <section className="pe-version-detail-section">
              <h4 className="pe-version-detail-label">
                <Cpu size={12} /> Target device
              </h4>
              <dl className="pe-version-kv">
                <dt>Board</dt>
                <dd>{deployment.target_device_slug || "Not chosen"}</dd>
                {deployment.target_device_custom_name && (
                  <>
                    <dt>Custom name</dt>
                    <dd>{deployment.target_device_custom_name}</dd>
                  </>
                )}
              </dl>
            </section>
          </div>
        </div>
      </div>

      {/* Since this snapshot */}
      <section className="pe-version-card">
        <div className="pe-version-detail" style={{ padding: "1.25rem 1.5rem" }}>
          <div className="pe-version-diff-head">
            <h4 className="pe-version-detail-label">Since this snapshot</h4>
            <button type="button" onClick={() => setRestoreOpen(true)} className="pe-version-btn-restore-sm">
              <RotateCcw size={13} /> Restore this version
            </button>
          </div>
          <div className="pe-version-diff-chips" style={{ marginTop: "0.6rem" }}>
            <span className={`pe-version-diff-chip ${diff.project_config_changed ? "is-changed" : ""}`}>
              {diff.project_config_changed
                ? `Project config changed (${diff.changed_project_fields.join(", ")})`
                : "Project config unchanged"}
            </span>
            <span className={`pe-version-diff-chip ${diff.deployment_changed ? "is-changed" : ""}`}>
              {diff.deployment_changed ? "Deployment settings changed" : "Deployment settings unchanged"}
            </span>
            <span className={`pe-version-diff-chip ${diff.sample_count_delta !== 0 ? "is-changed" : ""}`}>
              {diff.sample_count_delta === 0
                ? "Sample count unchanged"
                : diff.sample_count_delta > 0
                  ? `+${diff.sample_count_delta} samples`
                  : `${diff.sample_count_delta} samples`}
            </span>
            {(diff.class_names_added.length > 0 || diff.class_names_removed.length > 0) && (
              <span className="pe-version-diff-chip is-changed">
                {diff.class_names_added.length > 0 && `+${diff.class_names_added.join(", ")} `}
                {diff.class_names_removed.length > 0 && `−${diff.class_names_removed.join(", ")}`}
              </span>
            )}
            {diff.best_accuracy_delta !== null && (
              <span className={`pe-version-diff-chip ${Math.abs(diff.best_accuracy_delta) > 0.001 ? "is-changed" : ""}`}>
                Best accuracy {diff.best_accuracy_delta >= 0 ? "+" : ""}{(diff.best_accuracy_delta * 100).toFixed(1)}pp
              </span>
            )}
          </div>
          {diff.impulses.length > 0 && (
            <div className="pe-version-chip-row" style={{ marginTop: "0.75rem" }}>
              {diff.impulses.map((imp) => (
                <span
                  key={imp.impulse_id || imp.impulse_name}
                  className={`pe-version-diff-chip ${imp.status !== "unchanged" ? "is-changed" : ""}`}
                >
                  {imp.impulse_name}: {imp.status === "unchanged" ? "unchanged"
                    : imp.status === "changed" ? `changed (${imp.changed_fields.join(", ")})`
                    : imp.status === "deleted" ? "deleted since"
                    : "added since"}
                </span>
              ))}
            </div>
          )}
        </div>
      </section>

      {/* Per-impulse detail */}
      <section>
        <h4 className="pe-version-detail-label" style={{ marginBottom: "0.75rem" }}>
          Impulses <span className="pe-version-detail-label-count">{detail.impulses.length}</span>
        </h4>
        {detail.impulses.length === 0 ? (
          <p className="pe-version-empty-note">This project had no impulses at snapshot time.</p>
        ) : (
          <div className="pe-pver-impulse-list">
            {detail.impulses.map((imp) => (
              <ImpulseSection key={imp.id} impulse={imp} />
            ))}
          </div>
        )}
      </section>

      {/* Dataset manifest */}
      <section className="pe-version-card">
        <div className="pe-version-samples" style={{ padding: "1.25rem 1.5rem" }}>
          <h4 className="pe-version-detail-label">
            Dataset manifest <span className="pe-version-detail-label-count">{detail.samples_total}</span>
          </h4>
          <div className="pe-version-samples-table-wrap">
            <table className="pe-version-samples-table">
              <thead>
                <tr>
                  <th>Sample</th>
                  <th>Label</th>
                  <th>Split</th>
                </tr>
              </thead>
              <tbody>
                {detail.samples.map((s) => (
                  <tr key={s.id} className={s.sample_id ? "" : "is-orphaned"}>
                    <td>{s.sample_name}{!s.sample_id && <span className="pe-version-sample-gone"> (deleted)</span>}</td>
                    <td>{s.label_name || "—"}</td>
                    <td>{s.sample_type}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {totalPages > 1 && (
            <div className="pe-version-pager">
              <button
                type="button"
                onClick={() => setPage((p) => Math.max(1, p - 1))}
                disabled={page <= 1}
                className="pe-version-pager-btn"
                aria-label="Previous page"
              >
                <ChevronLeft size={14} />
              </button>
              <span className="pe-version-pager-label">Page {detail.samples_page} of {totalPages}</span>
              <button
                type="button"
                onClick={() => setPage((p) => Math.min(totalPages, p + 1))}
                disabled={page >= totalPages}
                className="pe-version-pager-btn"
                aria-label="Next page"
              >
                <ChevronRight size={14} />
              </button>
            </div>
          )}
        </div>
      </section>

      {restoreOpen && (
        <VersionRestoreDialog
          versionId={versionId}
          versionLabel={detail.name || `v${detail.version_number}`}
          onClose={() => setRestoreOpen(false)}
        />
      )}
    </div>
  );
}
