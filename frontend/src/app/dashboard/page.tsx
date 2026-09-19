"use client";
import { useEffect, useState } from "react";
import { createPortal } from "react-dom";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { useAppStore } from "@/store/appStore";
import {
  projectsApi,
  samplesApi,
  devicesApi,
  impulsesApi,
  trainingApi,
  trainedModelsApi,
  dspApi,
} from "@/utils/api";
import { extractApiError } from "@/lib/post-processing-helpers";
import { buildBlockOutputRows, resolveBlockOutputFilename, BlockOutputRow, BlockOutputRowKey } from "@/lib/block-output-helpers";
import { Database, BarChart3, Plus, ArrowRight, Cpu, GraduationCap, Download } from "lucide-react";
import { useRouter } from "next/navigation";
import toast from "react-hot-toast";
import CreateProjectModal, { ProjectTypePill } from "@/components/dashboard/CreateProjectModal";
import { useOnboarding } from "@/hooks/useOnboarding";

// Button copy for the Getting Started Tutorial card — "Restart" only while
// the tour is actively running, "Launch tutorial" for absent/completed/skipped.
// Not exported: Next.js's app-router typegen restricts page.tsx to its known
// route exports (default, metadata, ...), so this stays local to the component.
function tutorialButtonLabel(isActive: boolean): string {
  return isActive ? "Restart tutorial" : "Launch tutorial";
}

// README (Project.description reuse) size cap — mirrors _README_MAX_CHARS in
// backend/app/api/v1/endpoints/projects.py so the client can pre-check before
// the round trip that ultimately enforces it.
const README_MAX_CHARS = 32_000;

function isReadmeEmpty(description?: string | null): boolean {
  return !description || !description.trim();
}

function exceedsReadmeMax(description: string): boolean {
  return description.length > README_MAX_CHARS;
}

// ─── Download block output (C11 Phase 3) ───────────────────────────────────
// Filename resolution (Content-Disposition parsing + row-specific fallbacks)
// lives in @/lib/block-output-helpers so it's covered by a real unit test
// rather than a mirrored copy — see the note there on why the fallback must
// be per-row rather than one shared name.

function triggerBlobDownload(res: any, filename: string) {
  const blob = res.data instanceof Blob ? res.data : new Blob([res.data]);
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

export default function DashboardPage() {
  const { user, activeProject, setActiveProject, token } = useAppStore();
  const { isActive: tutorialActive, restart: restartTutorial } = useOnboarding();
  const [projects, setProjects] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);
  const [showNewProject, setShowNewProject] = useState(false);
  const router = useRouter();
  const [mounted, setMounted] = useState(false);
  const [dangerModal, setDangerModal] = useState<"trainTest" | "deleteProject" | "deleteData" | null>(null);
  const [dangerLoading, setDangerLoading] = useState(false);
  const [detailLoading, setDetailLoading] = useState(false);
  const [devicesCount, setDevicesCount] = useState(0);
  const [sampleTotal, setSampleTotal] = useState(0);
  const [labelingMethod, setLabelingMethod] = useState("bounding_boxes");
  const [readmeEditing, setReadmeEditing] = useState(false);
  const [readmeDraft, setReadmeDraft] = useState("");
  const [readmeSaving, setReadmeSaving] = useState(false);
  const [readmeError, setReadmeError] = useState<string | null>(null);
  const [blockOutputImpulses, setBlockOutputImpulses] = useState<any[]>([]);
  const [blockOutputImpulseId, setBlockOutputImpulseId] = useState<string>("");
  const [blockOutputRows, setBlockOutputRows] = useState<BlockOutputRow[]>([]);
  const [blockOutputLoading, setBlockOutputLoading] = useState(false);
  const [blockOutputDownloading, setBlockOutputDownloading] = useState<BlockOutputRowKey | null>(null);

  useEffect(() => {
    setMounted(true);
  }, []);

  useEffect(() => {
    if (mounted && !token) {
      router.push("/login");
    }
  }, [token, router, mounted]);

  useEffect(() => {
    if (mounted && token) {
      loadData();
    }
  }, [mounted, token]);

  useEffect(() => {
    setReadmeEditing(false);
    setReadmeError(null);
  }, [activeProject?.id]);

  useEffect(() => {
    if (!activeProject) return;
    setDetailLoading(true);
    Promise.all([
      devicesApi.list(activeProject.id).catch(() => ({ data: [] })),
      samplesApi.list(activeProject.id, { limit: 0 }).catch(() => ({ data: { total: 0 } })),
    ]).then(([devRes, sampRes]) => {
      const devData: any[] = devRes.data ?? [];
      setDevicesCount(Array.isArray(devData) ? devData.length : 0);
      setSampleTotal(sampRes.data?.total ?? 0);
    }).finally(() => setDetailLoading(false));
  }, [activeProject?.id]);

  useEffect(() => {
    if (!activeProject) {
      setBlockOutputImpulses([]);
      setBlockOutputImpulseId("");
      return;
    }
    impulsesApi.list(activeProject.id).then(({ data }) => {
      const list: any[] = Array.isArray(data) ? data : [];
      setBlockOutputImpulses(list);
      setBlockOutputImpulseId((prev) =>
        list.some((i) => i.id === prev) ? prev : (list[0]?.id ?? "")
      );
    }).catch(() => {
      setBlockOutputImpulses([]);
      setBlockOutputImpulseId("");
    });
  }, [activeProject?.id]);

  useEffect(() => {
    if (!blockOutputImpulseId) {
      setBlockOutputRows([]);
      return;
    }
    let cancelled = false;
    setBlockOutputLoading(true);
    Promise.all([
      dspApi.featuresReady(blockOutputImpulseId).catch(() => ({ data: { ready: false } })),
      trainingApi.impulseStatus(blockOutputImpulseId).catch(() => ({ data: { active_model_run_id: null } })),
    ]).then(async ([featuresRes, statusRes]) => {
      const activeModelRunId = statusRes.data?.active_model_run_id ?? null;
      let models: any[] = [];
      if (activeModelRunId) {
        try {
          const { data } = await trainedModelsApi.listForJob(activeModelRunId);
          models = Array.isArray(data) ? data : [];
        } catch {
          models = [];
        }
      }
      if (cancelled) return;
      setBlockOutputRows(
        buildBlockOutputRows({
          featuresReady: !!featuresRes.data?.ready,
          activeModelRunId,
          models,
        })
      );
    }).finally(() => {
      if (!cancelled) setBlockOutputLoading(false);
    });
    return () => { cancelled = true; };
  }, [blockOutputImpulseId]);

  async function downloadBlockOutput(row: BlockOutputRow) {
    if (!row.enabled || blockOutputDownloading) return;
    setBlockOutputDownloading(row.key);
    try {
      const impulseName = blockOutputImpulses.find((i) => i.id === blockOutputImpulseId)?.name;
      if (row.key === "features") {
        const res = await dspApi.downloadFeatures(blockOutputImpulseId);
        const filename = resolveBlockOutputFilename(row.key, res.headers?.["content-disposition"], impulseName);
        triggerBlobDownload(res, filename);
      } else if (row.modelId) {
        const res = await trainedModelsApi.download(row.modelId);
        const filename = resolveBlockOutputFilename(row.key, res.headers?.["content-disposition"], impulseName);
        triggerBlobDownload(res, filename);
      }
    } catch (e: any) {
      toast.error(extractApiError(e));
    } finally {
      setBlockOutputDownloading(null);
    }
  }

  if (!mounted || !token) return null;

  async function loadData() {
    try {
      const { data } = await projectsApi.list();
      setProjects(data);
      if (data.length > 0 && !activeProject) {
        setActiveProject(data[0]);
      }
    } catch (e) {
      console.error(e);
    } finally {
      setLoading(false);
    }
  }

  function handleProjectCreated(project: any) {
    setProjects((p) => [...p, project]);
    setActiveProject(project);
    setShowNewProject(false);
  }

  function startReadmeEdit() {
    setReadmeDraft(activeProject?.description || "");
    setReadmeError(null);
    setReadmeEditing(true);
  }

  function cancelReadmeEdit() {
    setReadmeEditing(false);
    setReadmeError(null);
  }

  async function saveReadme() {
    if (!activeProject) return;
    if (exceedsReadmeMax(readmeDraft)) {
      setReadmeError(`README must be at most ${README_MAX_CHARS.toLocaleString()} characters`);
      return;
    }
    setReadmeSaving(true);
    setReadmeError(null);
    try {
      const { data } = await projectsApi.update(activeProject.id, { description: readmeDraft });
      setActiveProject(data);
      setReadmeEditing(false);
    } catch (e: any) {
      setReadmeError(extractApiError(e));
    } finally {
      setReadmeSaving(false);
    }
  }

  async function performTrainTestSplit() {
    if (!activeProject) return;
    setDangerLoading(true);
    try {
      const res = await samplesApi.list(activeProject.id, { limit: 9999 });
      const allSamples: any[] = res.data.items || res.data;
      const eligible = allSamples.filter((s: any) => s.sample_type !== "postprocessing");
      const shuffled = [...eligible].sort(() => Math.random() - 0.5);
      const splitIdx = Math.round(shuffled.length * 0.8);
      const trainingIds = shuffled.slice(0, splitIdx).map((s: any) => s.id);
      const testingIds = shuffled.slice(splitIdx).map((s: any) => s.id);
      if (trainingIds.length) await samplesApi.bulkUpdate(trainingIds, "split", "training");
      if (testingIds.length) await samplesApi.bulkUpdate(testingIds, "split", "testing");
      toast.success("Train / test split performed");
      setDangerModal(null);
    } catch {
      toast.error("Failed to perform train / test split");
    } finally {
      setDangerLoading(false);
    }
  }

  async function deleteProject() {
    if (!activeProject) return;
    setDangerLoading(true);
    try {
      await projectsApi.delete(activeProject.id);
      setActiveProject(null);
      toast.success("Project deleted");
      setDangerModal(null);
      await loadData();
    } catch {
      toast.error("Failed to delete project");
    } finally {
      setDangerLoading(false);
    }
  }

  async function deleteAllData() {
    if (!activeProject) return;
    setDangerLoading(true);
    try {
      const res = await samplesApi.list(activeProject.id, { limit: 9999 });
      const allSamples: any[] = res.data.items || res.data;
      const ids = allSamples.map((s: any) => s.id);
      if (ids.length) await samplesApi.bulkDelete(ids);
      toast.success("All data deleted");
      setDangerModal(null);
    } catch {
      toast.error("Failed to delete data");
    } finally {
      setDangerLoading(false);
    }
  }

  function handleProjectSelect(project: any) {
    setActiveProject(project);
    router.push("/dashboard/data");
  }

  return (
    <div className="pe-dashboard max-w-7xl mx-auto space-y-7">
      <h2 className="pe-dashboard-welcome">
        Welcome back, {user?.username}
      </h2>

      <section className="pe-dashboard-hero">
        <div className="relative z-[1] max-w-4xl">
          <div className="min-w-0">
            <h2>{activeProject?.name || "Select a project"}</h2>
            <p className="mt-3">
              This is your Petal Edge  project.
              From here you acquire new training data,
              Design impulses and train models.
            </p>
          </div>
        </div>
      </section>

      <div className={`grid gap-6 ${activeProject ? "grid-cols-1 xl:grid-cols-[1.4fr_1fr]" : "grid-cols-1"}`}>
        {/* Projects list */}
        <div className="pe-card">
          <div className="flex items-center justify-between">
            <h3 className="pe-card-title">Projects</h3>
          </div>
          <div className="pe-card-divider" />

          <CreateProjectModal
            open={showNewProject}
            onClose={() => setShowNewProject(false)}
            onCreated={handleProjectCreated}
            existingNames={projects.map((p) => p.name)}
          />

          {loading ? (
            <div className="space-y-2.5">
              {[1, 2, 3, 4].map((i) => (
                <div key={i} className="h-14 rounded-xl animate-pulse" style={{ background: "var(--app-surface-2)" }} />
              ))}
            </div>
          ) : projects.length === 0 ? (
            <div className="text-center py-10">
              <Database size={32} className="mx-auto mb-2" style={{ color: "var(--app-text-soft)" }} />
              <p className="text-sm" style={{ color: "var(--app-text-muted)" }}>No projects yet</p>
              <button onClick={() => setShowNewProject(true)} className="pe-chip-new mt-3" data-tour="create-project">
                <Plus size={12} /> Create your first project
              </button>
            </div>
          ) : (
            <div className="space-y-2.5">
              {projects.map((p) => {
                const date = new Date(p.created_at);
                const dateStr = !isNaN(date.getTime()) ? date.toLocaleDateString() : "-";

                return (
                  <button
                    key={p.id}
                    onClick={() => handleProjectSelect(p)}
                    className={`pe-project-row ${activeProject?.id === p.id ? "is-active" : ""}`}
                  >
                    <div className="min-w-0">
                      <p className="pe-project-name truncate">{p.name}</p>
                      <p className="pe-project-meta flex items-center gap-2">
                        <span>Created {dateStr}</span>
                        <ProjectTypePill type={p.project_type} />
                      </p>
                    </div>
                    <span className="pe-arrow" aria-hidden="true">
                      <ArrowRight size={14} />
                    </span>
                  </button>
                );
              })}
            </div>
          )}
        </div>

        {/* Project details panel */}
        {activeProject && (
          <div className="space-y-5">
            {/* Summary card */}
            <div className="pe-card">
              <h3 className="pe-card-title">Summary</h3>
              <div className="pe-card-divider" />
              {detailLoading ? (
                <div className="space-y-4">
                  <div className="h-14 rounded-xl animate-pulse" style={{ background: "var(--app-surface-2)" }} />
                  <div className="h-14 rounded-xl animate-pulse" style={{ background: "var(--app-surface-2)" }} />
                </div>
              ) : (
                <div className="space-y-3">
                  <div className="pe-stat-row">
                    <div className="pe-stat-icon pe-stat-icon--emerald">
                      <Cpu size={18} />
                    </div>
                    <div>
                      <p className="pe-stat-eyebrow">Devices connected</p>
                      <p className="pe-stat-value">{devicesCount}</p>
                    </div>
                  </div>
                  <div className="pe-stat-row">
                    <div className="pe-stat-icon pe-stat-icon--rose">
                      <BarChart3 size={18} />
                    </div>
                    <div>
                      <p className="pe-stat-eyebrow">Data collected</p>
                      <p className="pe-stat-value">{sampleTotal.toLocaleString()} <span className="text-sm font-medium" style={{ color: "var(--app-text-muted)" }}>items</span></p>
                    </div>
                  </div>
                </div>
              )}
            </div>

            {/* Project info card */}
            <div className="pe-card">
              <h3 className="pe-card-title">Project info</h3>
              <div className="pe-card-divider" />
              <div className="pe-info-row">
                <span className="pe-info-label">Project ID</span>
                <span className="pe-info-value">{activeProject.id.slice(0, 8)}</span>
              </div>
              <div className="pe-info-row">
                <span className="pe-info-label">Labeling method</span>
                <label className="pe-info-chip relative">
                  <span>
                    {labelingMethod === "bounding_boxes" && "Bounding boxes (object detection)"}
                    {labelingMethod === "single_label" && "Single label (classification)"}
                    {labelingMethod === "multiple_labels" && "Multiple labels"}
                  </span>
                  <select
                    value={labelingMethod}
                    onChange={(e) => setLabelingMethod(e.target.value)}
                    className="absolute inset-0 opacity-0 cursor-pointer"
                    aria-label="Labeling method"
                  >
                    <option value="bounding_boxes">Bounding boxes (object detection)</option>
                    <option value="single_label">Single label (classification)</option>
                    <option value="multiple_labels">Multiple labels</option>
                  </select>
                </label>
              </div>
            </div>

            {/* About this project (README) card */}
            <div className="pe-card">
              <div className="flex items-center justify-between">
                <h3 className="pe-card-title">About this project</h3>
                {!readmeEditing && (
                  <button onClick={startReadmeEdit} className="pe-chip-new">
                    Edit
                  </button>
                )}
              </div>
              <div className="pe-card-divider" />
              {readmeEditing ? (
                <div className="space-y-3">
                  <textarea
                    className="pe-version-textarea"
                    style={{ minHeight: 160, width: "100%" }}
                    value={readmeDraft}
                    onChange={(e) => {
                      setReadmeDraft(e.target.value);
                      setReadmeError(null);
                    }}
                    placeholder="Add a README to describe this project, its goals, and how to use it…"
                  />
                  {readmeError && (
                    <p className="text-sm text-rose-500">{readmeError}</p>
                  )}
                  <div className="flex gap-2 justify-end">
                    <button
                      onClick={cancelReadmeEdit}
                      disabled={readmeSaving}
                      className="btn-secondary px-4 py-1.5 text-sm font-medium"
                    >
                      Cancel
                    </button>
                    <button onClick={saveReadme} disabled={readmeSaving} className="pe-chip-new">
                      {readmeSaving ? "Saving…" : "Save"}
                    </button>
                  </div>
                </div>
              ) : isReadmeEmpty(activeProject.description) ? (
                <p className="text-sm" style={{ color: "var(--app-text-muted)" }}>
                  Add a README to describe this project, its goals, and how to use it.
                </p>
              ) : (
                <div className="pe-readme-content text-sm" style={{ color: "var(--app-text)" }}>
                  <ReactMarkdown remarkPlugins={[remarkGfm]}>{activeProject.description}</ReactMarkdown>
                </div>
              )}
            </div>

            {/* Download block output card (C11 Phase 3) */}
            <div className="pe-card">
              <h3 className="pe-card-title">Download block output</h3>
              <div className="pe-card-divider" />
              {blockOutputImpulses.length === 0 ? (
                <p className="text-sm" style={{ color: "var(--app-text-muted)" }}>
                  Create an impulse to download its features and trained models.
                </p>
              ) : (
                <div className="space-y-3">
                  <label className="pe-info-row">
                    <span className="pe-info-label">Impulse</span>
                    <select
                      value={blockOutputImpulseId}
                      onChange={(e) => setBlockOutputImpulseId(e.target.value)}
                      className="pe-info-chip"
                      aria-label="Select impulse"
                    >
                      {blockOutputImpulses.map((imp) => (
                        <option key={imp.id} value={imp.id}>{imp.name}</option>
                      ))}
                    </select>
                  </label>
                  {blockOutputLoading ? (
                    <div className="h-14 rounded-xl animate-pulse" style={{ background: "var(--app-surface-2)" }} />
                  ) : (
                    <div className="space-y-2">
                      {blockOutputRows.map((row) => (
                        <div key={row.key} className="pe-info-row items-center">
                          <div className="min-w-0">
                            <p className="pe-info-label">{row.label}</p>
                            {!row.enabled && row.reason && (
                              <p className="text-xs" style={{ color: "var(--app-text-soft)" }}>{row.reason}</p>
                            )}
                          </div>
                          <button
                            onClick={() => downloadBlockOutput(row)}
                            disabled={!row.enabled || blockOutputDownloading !== null}
                            className="pe-chip-new disabled:opacity-50 disabled:cursor-not-allowed"
                          >
                            <Download size={12} />
                            {blockOutputDownloading === row.key ? "Downloading…" : "Download"}
                          </button>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              )}
            </div>

            {/* Getting Started Tutorial card */}
            <div className="pe-card">
              <h3 className="pe-card-title">Getting Started Tutorial</h3>
              <div className="pe-card-divider" />
              <p className="text-sm mb-4" style={{ color: "var(--app-text-muted)" }}>
                {tutorialActive
                  ? "The guided tour is in progress. Restart it to go back to the first step."
                  : "Take a guided tour of the dashboard to get up to speed quickly."}
              </p>
              <button onClick={restartTutorial} className="pe-chip-new">
                <GraduationCap size={12} /> {tutorialButtonLabel(tutorialActive)}
              </button>
            </div>
          </div>
        )}
      </div>
      {activeProject && (
        <div className="card border border-rose-900/40">
          <h3 className="text-rose-400 font-semibold text-sm uppercase tracking-wider mb-4">Danger zone</h3>
          <div className="h-px bg-gray-800 mb-4" />
          <div className="space-y-3">
            <button
              onClick={() => setDangerModal("trainTest")}
              className="flex items-center gap-2 px-4 py-2 rounded-lg bg-rose-600 hover:bg-rose-500 text-white text-sm font-medium transition-colors"
            >
              Perform train / test split
            </button>
            <button
              onClick={() => setDangerModal("deleteProject")}
              className="flex items-center gap-2 px-4 py-2 rounded-lg bg-rose-600 hover:bg-rose-500 text-white text-sm font-medium transition-colors"
            >
              Delete this project
            </button>
            <button
              onClick={() => setDangerModal("deleteData")}
              className="flex items-center gap-2 px-4 py-2 rounded-lg bg-rose-600 hover:bg-rose-500 text-white text-sm font-medium transition-colors"
            >
              Delete all data in this project
            </button>
          </div>
        </div>
      )}

      {dangerModal && createPortal(
        <div className="overlay-modal fixed inset-0 z-[100] flex items-center justify-center">
          <div className="surface-raised rounded-2xl p-8 max-w-md w-full mx-4 text-center">
            <div className="flex items-center justify-center mb-5">
              <div className="w-16 h-16 rounded-full border-2 border-rose-400 flex items-center justify-center">
                <span className="text-rose-500 text-3xl font-light">?</span>
              </div>
            </div>

            {dangerModal === "trainTest" && (
              <>
                <h2 className="text-xl font-bold mb-3" style={{ color: "var(--app-text)" }}>Perform train / test split</h2>
                <p className="text-sm mb-6" style={{ color: "var(--app-text-muted)" }}>
                  Are you sure you want to rebalance your dataset?{" "}
                  <span className="text-rose-500">
                    This splits all your data automatically between the training and testing set, and resets the categories for all data. This is irrevocable!
                  </span>
                </p>
                <div className="flex gap-3 justify-center">
                  <button onClick={() => setDangerModal(null)} disabled={dangerLoading} className="btn-secondary px-5 py-2 text-sm font-medium">
                    Cancel
                  </button>
                  <button onClick={performTrainTestSplit} disabled={dangerLoading} className="px-5 py-2 rounded-lg bg-rose-500 hover:bg-rose-600 text-white text-sm font-medium transition-colors disabled:opacity-50">
                    {dangerLoading ? "Splitting…" : "Yes, perform the train / test split"}
                  </button>
                </div>
              </>
            )}

            {dangerModal === "deleteProject" && (
              <>
                <h2 className="text-xl font-bold mb-3" style={{ color: "var(--app-text)" }}>Delete this project</h2>
                <p className="text-sm mb-6" style={{ color: "var(--app-text-muted)" }}>
                  Are you sure you want to delete{" "}
                  <span className="text-rose-500 font-medium">{activeProject?.name}</span>?{" "}
                  <span className="text-rose-500">This will permanently remove the project and all associated data. This is irrevocable!</span>
                </p>
                <div className="flex gap-3 justify-center">
                  <button onClick={() => setDangerModal(null)} disabled={dangerLoading} className="btn-secondary px-5 py-2 text-sm font-medium">
                    Cancel
                  </button>
                  <button onClick={deleteProject} disabled={dangerLoading} className="px-5 py-2 rounded-lg bg-rose-500 hover:bg-rose-600 text-white text-sm font-medium transition-colors disabled:opacity-50">
                    {dangerLoading ? "Deleting…" : "Yes, delete this project"}
                  </button>
                </div>
              </>
            )}

            {dangerModal === "deleteData" && (
              <>
                <h2 className="text-xl font-bold mb-3" style={{ color: "var(--app-text)" }}>Delete all data in this project</h2>
                <p className="text-sm mb-6" style={{ color: "var(--app-text-muted)" }}>
                  Are you sure you want to delete all samples in{" "}
                  <span className="text-rose-500 font-medium">{activeProject?.name}</span>?{" "}
                  <span className="text-rose-500">This permanently removes all uploaded data. This is irrevocable!</span>
                </p>
                <div className="flex gap-3 justify-center">
                  <button onClick={() => setDangerModal(null)} disabled={dangerLoading} className="btn-secondary px-5 py-2 text-sm font-medium">
                    Cancel
                  </button>
                  <button onClick={deleteAllData} disabled={dangerLoading} className="px-5 py-2 rounded-lg bg-rose-500 hover:bg-rose-600 text-white text-sm font-medium transition-colors disabled:opacity-50">
                    {dangerLoading ? "Deleting…" : "Yes, delete all data"}
                  </button>
                </div>
              </>
            )}
          </div>
        </div>,
        document.body
      )}
    </div>
  );
}
