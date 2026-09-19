"use client";

import { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import { createPortal } from "react-dom";
import Link from "next/link";
import { History, Plus, RefreshCw, MoreVertical, Eye, RotateCcw, GitCompare, BadgeCheck, X } from "lucide-react";
import toast from "react-hot-toast";
import { projectVersionsApi } from "@/utils/api";
import StoreVersionDialog from "./StoreVersionDialog";
import VersionRestoreDialog from "./VersionRestoreDialog";
import VersionCompareDialog from "./VersionCompareDialog";
import type { ProjectVersionSummary } from "./projectVersionTypes";

interface ProjectVersionTableProps {
  projectId: string;
  /** Read-only capture-summary numbers for the Store dialog — computed by
   * the page from data it already has (impulses, dataset), not fetched here. */
  captureSummary: { sampleCount: number; classCount: number; impulseCount: number };
}

function formatDateTime(iso: string) {
  const d = new Date(iso);
  return d.toLocaleDateString(undefined, { month: "short", day: "numeric", year: "numeric" }) +
    ", " + d.toLocaleTimeString(undefined, { hour: "numeric", minute: "2-digit" });
}

/**
 * The version-history table (parityfix.md §7.2) — a table, not a timeline of
 * expandable cards: "a version history is tabular data and reads fastest as
 * a table." The row's name links out to the detail view; the row itself
 * does not expand.
 */
export default function ProjectVersionTable({ projectId, captureSummary }: ProjectVersionTableProps) {
  const [versions, setVersions] = useState<ProjectVersionSummary[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(false);
  const [createOpen, setCreateOpen] = useState(false);
  const [publishedOnly, setPublishedOnly] = useState(false);
  const [openMenuId, setOpenMenuId] = useState<string | null>(null);
  const [restoreTarget, setRestoreTarget] = useState<ProjectVersionSummary | null>(null);
  const [compareSelection, setCompareSelection] = useState<string[]>([]);
  const [compareOpen, setCompareOpen] = useState(false);
  const [publishingId, setPublishingId] = useState<string | null>(null);

  // The row menu is portaled to <body> (see menuAnchor below) so the table
  // wrapper's `overflow-x: auto` — needed for horizontal scrolling on narrow
  // viewports — can never clip it, the same fix already used for the account
  // menu in TopBar.tsx. triggerElRef tracks whichever row's button was
  // clicked; only one menu is open at a time so a single ref suffices.
  const triggerElRef = useRef<HTMLButtonElement | null>(null);
  const menuRef = useRef<HTMLDivElement>(null);
  const [menuAnchor, setMenuAnchor] = useState<{ top: number; right: number } | null>(null);

  useEffect(() => {
    if (!openMenuId) return;
    function close(e: MouseEvent) {
      const target = e.target as HTMLElement;
      // Ignore clicks on any row's trigger button — including a different
      // row's — so switching which row's menu is open doesn't first close
      // and then reopen; the button's own onClick handles that toggle.
      if (target.closest?.(".pe-pver-actions-btn")) return;
      if (menuRef.current?.contains(target)) return;
      setOpenMenuId(null);
    }
    document.addEventListener("mousedown", close);
    return () => document.removeEventListener("mousedown", close);
  }, [openMenuId]);

  useLayoutEffect(() => {
    if (!openMenuId) {
      setMenuAnchor(null);
      return;
    }
    function update() {
      const el = triggerElRef.current;
      if (!el) return;
      const rect = el.getBoundingClientRect();
      setMenuAnchor({ top: rect.bottom + 4, right: window.innerWidth - rect.right });
    }
    update();
    window.addEventListener("resize", update);
    window.addEventListener("scroll", update, true);
    return () => {
      window.removeEventListener("resize", update);
      window.removeEventListener("scroll", update, true);
    };
  }, [openMenuId]);

  const load = useCallback(async () => {
    setLoading(true);
    setError(false);
    try {
      const { data } = await projectVersionsApi.list(projectId);
      setVersions(Array.isArray(data) ? data : []);
    } catch {
      setError(true);
    } finally {
      setLoading(false);
    }
  }, [projectId]);

  useEffect(() => {
    void load();
  }, [load]);

  function handleCreated(version: any) {
    setCreateOpen(false);
    setVersions((prev) => [
      {
        id: version.id,
        project_id: version.project_id,
        version_number: version.version_number,
        name: version.name,
        description: version.description,
        sample_count: version.sample_count,
        train_sample_count: version.train_sample_count,
        test_sample_count: version.test_sample_count,
        class_names: version.class_names,
        impulse_count: version.impulse_count,
        best_accuracy: version.best_accuracy,
        status: version.status,
        published_at: version.published_at,
        created_by: version.created_by,
        created_by_name: version.created_by_name,
        created_at: version.created_at,
      },
      ...prev,
    ]);
  }

  const visibleVersions = useMemo(
    () => (publishedOnly ? versions.filter((v) => v.status === "published") : versions),
    [versions, publishedOnly],
  );

  function toggleCompareSelection(versionId: string) {
    setCompareSelection((prev) => {
      if (prev.includes(versionId)) return prev.filter((id) => id !== versionId);
      if (prev.length >= 2) return prev; // cap at two — uncheck one first
      return [...prev, versionId];
    });
  }

  function startCompareWith(versionId: string) {
    setOpenMenuId(null);
    setCompareSelection([versionId]);
  }

  async function handlePublish(version: ProjectVersionSummary) {
    setOpenMenuId(null);
    setPublishingId(version.id);
    try {
      const { data } = await projectVersionsApi.publish(version.id);
      setVersions((prev) =>
        prev.map((v) => {
          if (v.id === data.id) return { ...v, status: data.status, published_at: data.published_at };
          if (v.status === "published") return { ...v, status: "draft", published_at: null };
          return v;
        }),
      );
      toast.success(`v${version.version_number} published`);
    } catch (err: any) {
      toast.error(err?.response?.data?.detail || "Could not publish this version");
    } finally {
      setPublishingId(null);
    }
  }

  const storeButton = (
    <button type="button" onClick={() => setCreateOpen(true)} className="pe-version-btn-primary">
      <Plus size={15} /> Store Project Version
    </button>
  );

  if (loading) {
    return (
      <div className="pe-version-loading">
        <RefreshCw size={16} className="pe-version-spin" /> Loading version history…
      </div>
    );
  }

  if (error) {
    return (
      <div className="pe-version-loading pe-version-loading--error">
        Couldn&apos;t load version history.
        <button type="button" onClick={() => void load()} className="pe-version-retry">Retry</button>
      </div>
    );
  }

  return (
    <div className="pe-version-card">
      <div className="pe-version-toolbar">
        {versions.length === 0 ? (
          <span className="pe-version-count">No versions yet</span>
        ) : (
          <label className="pe-pver-toolbar-filter">
            <input
              type="checkbox"
              checked={publishedOnly}
              onChange={(e) => setPublishedOnly(e.target.checked)}
            />
            Show published only
          </label>
        )}
        <div className="pe-pver-toolbar-right">
          {compareSelection.length > 0 && (
            <div className="pe-pver-compare-bar">
              <span className="pe-pver-compare-bar-label">
                {compareSelection.length === 1
                  ? "Select one more version to compare"
                  : "2 versions selected"}
              </span>
              <button
                type="button"
                className="pe-pver-compare-bar-btn"
                disabled={compareSelection.length !== 2}
                onClick={() => setCompareOpen(true)}
              >
                <GitCompare size={14} /> Compare selected
              </button>
              <button
                type="button"
                className="pe-pver-compare-bar-clear"
                aria-label="Cancel comparison selection"
                onClick={() => setCompareSelection([])}
              >
                <X size={14} />
              </button>
            </div>
          )}
          {versions.length > 0 && (
            <span className="pe-version-count">
              {visibleVersions.length} version{visibleVersions.length === 1 ? "" : "s"}
            </span>
          )}
          {storeButton}
        </div>
      </div>

      {versions.length === 0 ? (
        <div className="pe-version-empty">
          <span className="pe-version-empty-icon" aria-hidden="true">
            <History size={28} strokeWidth={1.8} />
          </span>
          <p className="pe-version-empty-title">No versions yet</p>
          <p className="pe-version-empty-desc">
            Store one to record the current state of this project — its dataset, every impulse and
            their trained models, and its settings, exactly as they are right now.
          </p>
        </div>
      ) : (
        <div className="pe-pver-table-wrap">
          <table className="pe-pver-table">
            <thead>
              <tr>
                <th className="pe-pver-col-select" aria-hidden="true" />
                <th>#</th>
                <th>Name</th>
                <th className="pe-pver-col-desc">Description</th>
                <th>Created</th>
                <th>By</th>
                <th>Dataset</th>
                <th>Impulses</th>
                <th>Best acc.</th>
                <th aria-hidden="true" />
              </tr>
            </thead>
            <tbody>
              {visibleVersions.map((v) => (
                <tr key={v.id} className={compareSelection.includes(v.id) ? "is-compare-selected" : undefined}>
                  <td className="pe-pver-col-select">
                    <input
                      type="checkbox"
                      aria-label={`Select v${v.version_number} to compare`}
                      checked={compareSelection.includes(v.id)}
                      disabled={compareSelection.length >= 2 && !compareSelection.includes(v.id)}
                      onChange={() => toggleCompareSelection(v.id)}
                    />
                  </td>
                  <td className="pe-pver-num">v{v.version_number}</td>
                  <td>
                    <Link href={`/dashboard/versions/detail?versionId=${v.id}`} className="pe-pver-name-link">
                      {v.name || <span className="pe-pver-name-untitled">Untitled snapshot</span>}
                    </Link>
                    {v.status === "published" && <span className="pe-version-published-badge">Published</span>}
                  </td>
                  <td className="pe-pver-col-desc" title={v.description || undefined}>
                    {v.description || "—"}
                  </td>
                  <td>{formatDateTime(v.created_at)}</td>
                  <td>{v.created_by_name || "—"}</td>
                  <td className="pe-pver-dataset">
                    {v.sample_count.toLocaleString()}
                    <span className="pe-pver-dataset-sub">
                      {v.class_names.length} class{v.class_names.length === 1 ? "" : "es"}
                    </span>
                  </td>
                  <td>{v.impulse_count}</td>
                  <td className="pe-pver-accuracy">
                    {v.best_accuracy !== null
                      ? `${(v.best_accuracy * 100).toFixed(1)}%`
                      : <span className="pe-pver-accuracy-empty">—</span>}
                  </td>
                  <td className="pe-pver-actions-cell">
                    <button
                      type="button"
                      className="pe-pver-actions-btn"
                      aria-label={`Actions for version ${v.version_number}`}
                      aria-haspopup="menu"
                      aria-expanded={openMenuId === v.id}
                      onClick={(e) => {
                        e.stopPropagation();
                        triggerElRef.current = e.currentTarget;
                        setOpenMenuId(openMenuId === v.id ? null : v.id);
                      }}
                    >
                      <MoreVertical size={15} aria-hidden="true" />
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {openMenuId && menuAnchor && typeof document !== "undefined" && createPortal(
        (() => {
          const v = versions.find((row) => row.id === openMenuId);
          if (!v) return null;
          return (
            <div
              ref={menuRef}
              role="menu"
              className="pe-pver-actions-menu dropdown-panel"
              style={{ position: "fixed", top: menuAnchor.top, right: menuAnchor.right }}
            >
              <Link
                role="menuitem"
                className="pe-pver-actions-item"
                href={`/dashboard/versions/detail?versionId=${v.id}`}
                onClick={() => setOpenMenuId(null)}
              >
                <Eye size={14} aria-hidden="true" style={{ marginRight: "0.5rem" }} />
                View
              </Link>
              <button
                role="menuitem"
                className="pe-pver-actions-item"
                onClick={() => startCompareWith(v.id)}
              >
                <GitCompare size={14} aria-hidden="true" style={{ marginRight: "0.5rem" }} />
                Compare with…
              </button>
              <button
                role="menuitem"
                className="pe-pver-actions-item"
                onClick={() => {
                  setOpenMenuId(null);
                  setRestoreTarget(v);
                }}
              >
                <RotateCcw size={14} aria-hidden="true" style={{ marginRight: "0.5rem" }} />
                Restore
              </button>
              {v.status !== "published" && (
                <button
                  role="menuitem"
                  className="pe-pver-actions-item"
                  disabled={publishingId === v.id}
                  onClick={() => handlePublish(v)}
                >
                  <BadgeCheck size={14} aria-hidden="true" style={{ marginRight: "0.5rem" }} />
                  {publishingId === v.id ? "Publishing…" : "Publish"}
                </button>
              )}
            </div>
          );
        })(),
        document.body,
      )}

      {createOpen && (
        <StoreVersionDialog
          projectId={projectId}
          summary={captureSummary}
          onClose={() => setCreateOpen(false)}
          onCreated={handleCreated}
        />
      )}

      {restoreTarget && (
        <VersionRestoreDialog
          versionId={restoreTarget.id}
          versionLabel={restoreTarget.name || `v${restoreTarget.version_number}`}
          onClose={() => setRestoreTarget(null)}
        />
      )}

      {compareOpen && compareSelection.length === 2 && (
        <VersionCompareDialog
          versionIds={[compareSelection[0], compareSelection[1]]}
          onClose={() => {
            setCompareOpen(false);
            setCompareSelection([]);
          }}
        />
      )}
    </div>
  );
}
