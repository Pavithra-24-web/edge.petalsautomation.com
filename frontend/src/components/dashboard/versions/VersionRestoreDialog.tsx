"use client";

import { useEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { useRouter } from "next/navigation";
import { RotateCcw, X, Loader2, CheckCircle2, AlertTriangle } from "lucide-react";
import toast from "react-hot-toast";
import { projectVersionsApi, projectsApi } from "@/utils/api";
import { useAppStore } from "@/store/appStore";
import type { VersionRestoreResponse } from "./projectVersionTypes";

interface VersionRestoreDialogProps {
  versionId: string;
  versionLabel: string;
  onClose: () => void;
}

type Stage = "form" | "submitting" | "result";

/**
 * Restore dialog — asks only for the new project's name and an optional
 * description, then creates a brand-new project that is a full clone of the
 * source project, pinned to this version's configuration (Phase 2 redesign:
 * restore duplicates a trained project rather than reconstructing one from a
 * snapshot). The source project is never touched, so there is nothing to
 * preview or reconcile beforehand — no diff list, no technical
 * reconciliation details, just the name of the new project.
 */
export default function VersionRestoreDialog({ versionId, versionLabel, onClose }: VersionRestoreDialogProps) {
  const [stage, setStage] = useState<Stage>("form");
  const [name, setName] = useState("");
  const [description, setDescription] = useState("");
  const [result, setResult] = useState<VersionRestoreResponse | null>(null);
  const modalRef = useRef<HTMLDivElement>(null);
  const router = useRouter();
  const { setActiveProject } = useAppStore();

  useEffect(() => {
    modalRef.current?.focus();
    function handleKey(e: KeyboardEvent) {
      if (e.key === "Escape" && stage !== "submitting") onClose();
    }
    document.addEventListener("keydown", handleKey);
    return () => document.removeEventListener("keydown", handleKey);
  }, [onClose, stage]);

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    const trimmedName = name.trim();
    if (!trimmedName) return;
    setStage("submitting");
    try {
      const { data } = await projectVersionsApi.restore(versionId, {
        name: trimmedName,
        description: description.trim() || undefined,
      });
      setResult(data);
      setStage("result");
      toast.success("Project restored successfully");
    } catch (err: any) {
      toast.error(err?.response?.data?.detail || "Could not create a project from this version");
      setStage("form");
    }
  }

  async function handleOpenProject() {
    if (!result) return;
    try {
      const { data: freshProject } = await projectsApi.get(result.new_project_id);
      setActiveProject(freshProject);
    } catch {
      // Fall through to navigation regardless — the projects page will load it.
    }
    router.push("/dashboard");
  }

  if (typeof document === "undefined") return null;

  return createPortal(
    <div
      className="pe-version-overlay"
      onClick={stage === "submitting" ? undefined : onClose}
      role="dialog"
      aria-modal="true"
      aria-labelledby="pe-pver-restore-title"
    >
      <div
        ref={modalRef}
        className="pe-version-modal"
        tabIndex={-1}
        onClick={(e) => e.stopPropagation()}
      >
        <div className="pe-version-modal-head">
          <h3 id="pe-pver-restore-title" className="pe-version-modal-title">
            <span className="pe-version-modal-title-icon pe-version-modal-title-icon--restore" aria-hidden="true">
              <RotateCcw size={16} strokeWidth={2.1} />
            </span>
            {stage === "result" ? "Project restored" : "Restore Project Version"}
          </h3>
          {stage !== "submitting" && (
            <button type="button" onClick={onClose} className="pe-version-modal-close" aria-label="Close">
              <X size={16} />
            </button>
          )}
        </div>

        {stage !== "result" ? (
          <form onSubmit={handleSubmit}>
            <div className="pe-version-modal-body">
              <p className="pe-version-modal-intro">
                This creates a new independent project from <strong>{versionLabel}</strong>. The
                original project will not be modified.
              </p>

              <label className="pe-version-field">
                <span className="pe-version-field-label">Project Name</span>
                <input
                  type="text"
                  value={name}
                  onChange={(e) => setName(e.target.value)}
                  placeholder="e.g. &quot;Wave detector (restored)&quot;"
                  className="pe-version-input"
                  maxLength={120}
                  autoFocus
                  required
                  disabled={stage === "submitting"}
                />
              </label>

              <label className="pe-version-field">
                <span className="pe-version-field-label">
                  Description <span className="pe-version-field-optional">optional</span>
                </span>
                <textarea
                  value={description}
                  onChange={(e) => setDescription(e.target.value)}
                  placeholder="What is this restored copy for?"
                  className="pe-version-textarea"
                  rows={3}
                  maxLength={2000}
                  disabled={stage === "submitting"}
                />
              </label>
            </div>

            <div className="pe-version-modal-foot">
              <button type="button" onClick={onClose} className="pe-version-btn-ghost" disabled={stage === "submitting"}>
                Cancel
              </button>
              <button type="submit" className="pe-version-btn-restore" disabled={stage === "submitting" || !name.trim()}>
                {stage === "submitting" ? <Loader2 size={15} className="pe-version-spin" /> : <RotateCcw size={15} />}
                {stage === "submitting" ? "Creating…" : "Create Restored Project"}
              </button>
            </div>
          </form>
        ) : (
          <>
            <div className="pe-version-modal-body">
              <div
                className="pe-version-restore-warning"
                style={{ color: "#059669", background: "rgba(16, 185, 129, 0.10)", borderColor: "rgba(16, 185, 129, 0.28)" }}
              >
                <CheckCircle2 size={15} />
                <span>
                  ✓ Project restored successfully — <strong>{result?.new_project_name}</strong> is
                  a full, independent copy of this version. Your original project was left
                  untouched.
                </span>
              </div>
              {!!result?.skipped_sample_names?.length && (
                <div
                  className="pe-version-restore-warning"
                  style={{ color: "#B45309", background: "rgba(245, 158, 11, 0.10)", borderColor: "rgba(245, 158, 11, 0.28)", marginTop: 8 }}
                >
                  <AlertTriangle size={15} />
                  <span>
                    {result.skipped_sample_names.length} sample
                    {result.skipped_sample_names.length === 1 ? "" : "s"} could not be restored
                    because {result.skipped_sample_names.length === 1 ? "its" : "their"} original
                    file{result.skipped_sample_names.length === 1 ? " was" : "s were"} deleted
                    after this version was stored: {result.skipped_sample_names.slice(0, 5).join(", ")}
                    {result.skipped_sample_names.length > 5 ? ", …" : ""}
                  </span>
                </div>
              )}
            </div>
            <div className="pe-version-modal-foot">
              <button type="button" onClick={onClose} className="pe-version-btn-ghost">
                Stay here
              </button>
              <button type="button" onClick={handleOpenProject} className="pe-version-btn-primary">
                Open new project
              </button>
            </div>
          </>
        )}
      </div>
    </div>,
    document.body,
  );
}
