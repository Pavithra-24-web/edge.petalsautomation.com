"use client";
import { Suspense, useEffect, useMemo, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { useRouter, useSearchParams } from "next/navigation";
import { useAppStore } from "@/store/appStore";
import { projectsApi } from "@/utils/api";
import { avatarInitial } from "@/utils/avatarInitial";
import toast from "react-hot-toast";
import {
  Plus, MoreVertical, ChevronDown, Folder, Calendar, Trash2, Pencil,
} from "lucide-react";
import CreateProjectModal, {
  ProjectTypePill, isDuplicateName, DUPLICATE_NAME_MESSAGE, ProjectType,
} from "@/components/dashboard/CreateProjectModal";

type ProjectRow = {
  id: string;
  name: string;
  created_at?: string;
  project_type?: ProjectType;
};

type SortKey = "last_accessed" | "oldest" | "newest" | "az" | "za";

const SORT_OPTIONS: { key: SortKey; label: string }[] = [
  { key: "last_accessed", label: "Last accessed" },
  { key: "oldest", label: "Oldest first" },
  { key: "newest", label: "Newest first" },
  { key: "az", label: "A to Z" },
  { key: "za", label: "Z to A" },
];

/** Case-insensitive, trim-aware duplicate check for Rename — same helper
 *  Create uses, imported from CreateProjectModal rather than redefined. */
function isDuplicateNameExcluding(
  projects: ProjectRow[],
  name: string,
  excludeId?: string,
): boolean {
  return isDuplicateName(
    projects.filter((p) => p.id !== excludeId).map((p) => p.name),
    name,
  );
}

function formatDate(iso?: string): string {
  if (!iso) return "—";
  const d = new Date(iso);
  if (isNaN(d.getTime())) return "—";
  return d.toLocaleDateString(undefined, { month: "short", day: "numeric", year: "numeric" });
}



export default function ProjectsPage() {
  return (
    <Suspense fallback={null}>
      <ProjectsPageInner />
    </Suspense>
  );
}

function ProjectsPageInner() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const { user, profileExtras, setActiveProject, activeProject, token } = useAppStore();
  const [mounted, setMounted] = useState(false);
  const [projects, setProjects] = useState<ProjectRow[]>([]);
  const [loading, setLoading] = useState(true);
  const [sortKey, setSortKey] = useState<SortKey>("oldest");
  const [sortOpen, setSortOpen] = useState(false);
  const [createOpen, setCreateOpen] = useState(false);
  const [openMenuId, setOpenMenuId] = useState<string | null>(null);
  const [deleteTarget, setDeleteTarget] = useState<ProjectRow | null>(null);
  const [deleting, setDeleting] = useState(false);
  const [renameTarget, setRenameTarget] = useState<ProjectRow | null>(null);
  const [renameValue, setRenameValue] = useState("");
  const [renameError, setRenameError] = useState<string | null>(null);
  const [renaming, setRenaming] = useState(false);
  const sortRef = useRef<HTMLDivElement | null>(null);
  const rowMenuRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    setMounted(true);
  }, []);

  useEffect(() => {
    if (mounted && !token) router.push("/login");
  }, [mounted, token, router]);

  useEffect(() => {
    if (!mounted || !token) return;
    let cancelled = false;
    setLoading(true);
    projectsApi
      .list()
      .then(({ data }) => {
        if (cancelled) return;
        setProjects(Array.isArray(data) ? data : []);
      })
      .catch(() => {
        if (!cancelled) setProjects([]);
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [mounted, token]);

  useEffect(() => {
    if (!mounted) return;
    if (searchParams?.get("new") === "1") {
      setCreateOpen(true);
      router.replace("/dashboard/projects");
    }
  }, [mounted, searchParams, router]);

  useEffect(() => {
    if (!sortOpen) return;
    function onDocClick(e: MouseEvent) {
      if (!sortRef.current) return;
      if (!sortRef.current.contains(e.target as Node)) setSortOpen(false);
    }
    document.addEventListener("mousedown", onDocClick);
    return () => document.removeEventListener("mousedown", onDocClick);
  }, [sortOpen]);

  useEffect(() => {
    if (!openMenuId) return;
    function onDocClick(e: MouseEvent) {
      if (!rowMenuRef.current) return;
      if (!rowMenuRef.current.contains(e.target as Node)) setOpenMenuId(null);
    }
    function onKey(e: KeyboardEvent) {
      if (e.key === "Escape") setOpenMenuId(null);
    }
    document.addEventListener("mousedown", onDocClick);
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("mousedown", onDocClick);
      document.removeEventListener("keydown", onKey);
    };
  }, [openMenuId]);

  const sortedProjects = useMemo(() => {
    const list = [...projects];
    switch (sortKey) {
      case "oldest":
        return list.sort((a, b) => (a.created_at || "").localeCompare(b.created_at || ""));
      case "newest":
        return list.sort((a, b) => (b.created_at || "").localeCompare(a.created_at || ""));
      case "az":
        return list.sort((a, b) => a.name.localeCompare(b.name));
      case "za":
        return list.sort((a, b) => b.name.localeCompare(a.name));
      case "last_accessed":
      default:
        return list;
    }
  }, [projects, sortKey]);

  const savedDisplayName =
    (user?.id && profileExtras[user.id]?.name?.trim()) || "";
  const displayName =
    savedDisplayName ||
    user?.username ||
    (user?.email ? user.email.split("@")[0] : "User");
  const initial = avatarInitial(savedDisplayName || user?.username, user?.email);

  function openProject(p: ProjectRow) {
    setActiveProject(p as any);
    router.push("/dashboard");
  }

  function handleCreated(project: ProjectRow) {
    setProjects((prev) => [...prev, project]);
    setActiveProject(project as any);
  }

  // Mirrors the dashboard page's deleteProject() — same API call
  // (projectsApi.delete) + same activeProject reset rule when the deleted
  // project happens to be the currently selected one.
  async function handleDelete() {
    if (!deleteTarget || deleting) return;
    setDeleting(true);
    try {
      await projectsApi.delete(deleteTarget.id);
      setProjects((prev) => prev.filter((p) => p.id !== deleteTarget.id));
      if (activeProject?.id === deleteTarget.id) {
        setActiveProject(null);
      }
      toast.success("Project deleted");
      setDeleteTarget(null);
    } catch {
      toast.error("Failed to delete project");
    } finally {
      setDeleting(false);
    }
  }

  function openRename(p: ProjectRow) {
    setRenameTarget(p);
    setRenameValue(p.name);
    setRenameError(null);
  }

  async function handleRename() {
    if (!renameTarget || renaming) return;
    const trimmed = renameValue.trim();
    if (!trimmed) {
      setRenameError("Project name cannot be empty.");
      return;
    }
    if (trimmed === renameTarget.name) {
      // No-op rename — just close the modal.
      setRenameTarget(null);
      return;
    }
    // Pre-check against the list we already have to give a clean inline
    // error without a round-trip. The server still enforces uniqueness as
    // the authority (handles races from other tabs / API clients).
    if (isDuplicateNameExcluding(projects, trimmed, renameTarget.id)) {
      setRenameError(DUPLICATE_NAME_MESSAGE);
      return;
    }
    setRenaming(true);
    try {
      const { data } = await projectsApi.update(renameTarget.id, { name: trimmed });
      setProjects((prev) =>
        prev.map((p) => (p.id === renameTarget.id ? { ...p, ...data } : p)),
      );
      if (activeProject?.id === renameTarget.id) {
        setActiveProject({ ...(activeProject as any), ...data });
      }
      toast.success("Project renamed");
      setRenameTarget(null);
      setRenameValue("");
      setRenameError(null);
    } catch (e: any) {
      // Server-enforced duplicate (race with another tab) lands here too.
      if (e?.response?.status === 409) {
        setRenameError(DUPLICATE_NAME_MESSAGE);
      } else {
        toast.error(e?.response?.data?.detail || "Failed to rename project");
      }
    } finally {
      setRenaming(false);
    }
  }

  if (!mounted || !token) return null;

  return (
    <div className="pe-projects-page mx-auto">
      <div className="pe-projects-grid">
        <section className="pe-projects-main">
          <div className="pe-projects-header">
            <div className="pe-projects-title-wrap">
              <span className="pe-projects-title-icon" aria-hidden="true">
                <Folder size={16} />
              </span>
              <h1 className="pe-projects-title">Projects</h1>
            </div>
            <div className="pe-projects-actions">
              <div className="pe-sort-wrap" ref={sortRef}>
                <button
                  type="button"
                  className="pe-sort-btn"
                  onClick={() => setSortOpen((v) => !v)}
                  aria-haspopup="menu"
                  aria-expanded={sortOpen}
                >
                  <span>Sort</span>
                  <ChevronDown size={14} aria-hidden="true" />
                </button>
                {sortOpen && (
                  <div role="menu" className="pe-sort-menu dropdown-panel rounded-xl">
                    {SORT_OPTIONS.map((opt) => (
                      <button
                        key={opt.key}
                        role="menuitem"
                        className={`pe-sort-menu-item ${opt.key === sortKey ? "is-active" : ""}`}
                        onClick={() => {
                          setSortKey(opt.key);
                          setSortOpen(false);
                        }}
                      >
                        {opt.label}
                      </button>
                    ))}
                  </div>
                )}
              </div>
              <button
                type="button"
                className="pe-create-btn"
                onClick={() => setCreateOpen(true)}
              >
                <Plus size={15} aria-hidden="true" />
                <span>Create new project</span>
              </button>
            </div>
          </div>

          <div className="pe-projects-list">
            {loading ? (
              <>
                {[1, 2, 3, 4].map((i) => (
                  <div
                    key={i}
                    className="pe-projects-row pe-projects-row--skeleton"
                    aria-hidden="true"
                  />
                ))}
              </>
            ) : sortedProjects.length === 0 ? (
              <div className="pe-projects-empty">
                <p>No projects yet.</p>
                <button
                  type="button"
                  className="pe-create-btn"
                  onClick={() => setCreateOpen(true)}
                >
                  <Plus size={15} aria-hidden="true" />
                  <span>Create your first project</span>
                </button>
              </div>
            ) : (
              sortedProjects.map((p) => (
                <div
                  key={p.id}
                  className={`pe-projects-row ${activeProject?.id === p.id ? "is-active" : ""}`}
                >
                  <button
                    type="button"
                    className="pe-projects-row-main"
                    onClick={() => openProject(p)}
                  >
                    <span className="pe-projects-row-avatar" aria-hidden="true">
                      {initial}
                    </span>
                    <span className="pe-projects-row-text">
                      <span className="pe-projects-row-owner">
                        {displayName} /
                      </span>{" "}
                      <span className="pe-projects-row-name">{p.name}</span>
                    </span>
                  </button>
                  <div className="pe-projects-row-meta">
                    <ProjectTypePill type={p.project_type} />
                    <Calendar size={13} aria-hidden="true" />
                    <span>{formatDate(p.created_at)}</span>
                  </div>
                  <div
                    className="pe-projects-row-menu-wrap"
                    ref={openMenuId === p.id ? rowMenuRef : undefined}
                  >
                    <button
                      type="button"
                      className="pe-projects-row-more"
                      aria-label={`More options for ${p.name}`}
                      aria-haspopup="menu"
                      aria-expanded={openMenuId === p.id}
                      onClick={(e) => {
                        e.stopPropagation();
                        setOpenMenuId(openMenuId === p.id ? null : p.id);
                      }}
                    >
                      <MoreVertical size={16} aria-hidden="true" />
                    </button>
                    {openMenuId === p.id && (
                      <div role="menu" className="pe-row-menu dropdown-panel rounded-xl">
                        <button
                          role="menuitem"
                          className="pe-row-menu-item"
                          onClick={() => {
                            setOpenMenuId(null);
                            openRename(p);
                          }}
                        >
                          <Pencil size={14} aria-hidden="true" />
                          <span>Rename project</span>
                        </button>
                        <button
                          role="menuitem"
                          className="pe-row-menu-item pe-row-menu-item--danger"
                          onClick={() => {
                            setOpenMenuId(null);
                            setDeleteTarget(p);
                          }}
                        >
                          <Trash2 size={14} aria-hidden="true" />
                          <span>Delete project</span>
                        </button>
                      </div>
                    )}
                  </div>
                </div>
              ))
            )}
          </div>
        </section>
      </div>

      <CreateProjectModal
        open={createOpen}
        onClose={() => setCreateOpen(false)}
        onCreated={handleCreated}
        existingNames={projects.map((p) => p.name)}
      />

      {renameTarget && createPortal(
        <div
          className="overlay-modal fixed inset-0 z-[100] flex items-center justify-center"
          onClick={() => {
            if (renaming) return;
            setRenameTarget(null);
            setRenameError(null);
          }}
        >
          <div
            className="pe-create-modal surface-raised rounded-2xl"
            onClick={(e) => e.stopPropagation()}
          >
            <h2 className="pe-create-modal-title">Rename project</h2>
            <label className="pe-create-modal-label" htmlFor="pe-rename-project-name">
              Enter a new name for this project:
            </label>
            <input
              id="pe-rename-project-name"
              className="input pe-create-modal-input"
              placeholder="Project name"
              value={renameValue}
              onChange={(e) => {
                setRenameValue(e.target.value);
                if (renameError) setRenameError(null);
              }}
              onKeyDown={(e) => e.key === "Enter" && handleRename()}
              autoFocus
              disabled={renaming}
              aria-invalid={renameError ? "true" : "false"}
              aria-describedby={renameError ? "pe-rename-error" : undefined}
              autoComplete="off"
              spellCheck={false}
            />
            {renameError && (
              <p
                id="pe-rename-error"
                role="alert"
                className="pe-create-modal-error"
              >
                {renameError}
              </p>
            )}
            <div className="pe-create-modal-actions">
              <button
                type="button"
                className="btn-secondary px-5 py-2 text-sm font-medium"
                onClick={() => {
                  setRenameTarget(null);
                  setRenameValue("");
                  setRenameError(null);
                }}
                disabled={renaming}
              >
                Cancel
              </button>
              <button
                type="button"
                className="btn-primary px-5 py-2 text-sm font-medium"
                onClick={handleRename}
                disabled={renaming || !renameValue.trim()}
              >
                {renaming ? "Saving…" : "Save"}
              </button>
            </div>
          </div>
        </div>,
        document.body
      )}

      {deleteTarget && createPortal(
        <div
          className="overlay-modal fixed inset-0 z-[100] flex items-center justify-center"
          onClick={() => !deleting && setDeleteTarget(null)}
        >
          <div
            className="surface-raised rounded-2xl p-8 max-w-md w-full mx-4 text-center"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="flex items-center justify-center mb-5">
              <div className="w-16 h-16 rounded-full border-2 border-rose-400 flex items-center justify-center">
                <span className="text-rose-500 text-3xl font-light">?</span>
              </div>
            </div>
            <h2 className="text-xl font-bold mb-3" style={{ color: "var(--app-text)" }}>
              Delete this project
            </h2>
            <p className="text-sm mb-6" style={{ color: "var(--app-text-muted)" }}>
              Are you sure you want to delete{" "}
              <span className="text-rose-500 font-medium">{deleteTarget.name}</span>?{" "}
              <span className="text-rose-500">
                This will permanently remove the project and all associated data. This is irrevocable!
              </span>
            </p>
            <div className="flex gap-3 justify-center">
              <button
                onClick={() => setDeleteTarget(null)}
                disabled={deleting}
                className="btn-secondary px-5 py-2 text-sm font-medium"
              >
                Cancel
              </button>
              <button
                onClick={handleDelete}
                disabled={deleting}
                className="px-5 py-2 rounded-lg bg-rose-500 hover:bg-rose-600 text-white text-sm font-medium transition-colors disabled:opacity-50"
              >
                {deleting ? "Deleting…" : "Yes, delete this project"}
              </button>
            </div>
          </div>
        </div>,
        document.body
      )}
    </div>
  );
}
