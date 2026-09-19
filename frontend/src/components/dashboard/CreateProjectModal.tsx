"use client";
import { useState } from "react";
import { createPortal } from "react-dom";
import { Camera, Hand } from "lucide-react";
import toast from "react-hot-toast";
import { projectsApi } from "@/utils/api";

export type ProjectType = "object_detection" | "motion";

export const DUPLICATE_NAME_MESSAGE =
  "A project with this name already exists. Choose a different name.";

/** Case-insensitive, trim-aware duplicate check shared by Create and Rename.
 *  The server enforces the same constraint as the authority — this only
 *  short-circuits the round-trip for the common case. */
export function isDuplicateName(existingNames: string[], name: string): boolean {
  const needle = name.trim().toLowerCase();
  if (!needle) return false;
  return existingNames.some((n) => n.trim().toLowerCase() === needle);
}

/** Label, icon and accent hue for each project type — the single source
 *  both the picker cards below and any list-row badge render from, so a
 *  third type costs one entry here rather than a change in three files. */
export const PROJECT_TYPE_META: Record<
  ProjectType,
  { label: string; description: string; icon: typeof Camera; hue: "emerald" | "violet" }
> = {
  object_detection: {
    label: "Object Detection",
    description: "Detect and locate objects in images across different categories.",
    icon: Camera,
    hue: "emerald",
  },
  motion: {
    label: "Motion: Gesture Recognition",
    description: "Recognize and classify human gestures and movements from motion sensor data.",
    icon: Hand,
    hue: "violet",
  },
};

const PROJECT_TYPE_ORDER: ProjectType[] = ["object_detection", "motion"];

/** Small pill used as a project-row subtitle on the projects list and the
 *  dashboard's project list. Falls back to "object_detection" for any
 *  pre-Phase-0 persisted state that has no `project_type` yet. */
export function ProjectTypePill({ type }: { type?: ProjectType | null }) {
  const meta = PROJECT_TYPE_META[type ?? "object_detection"];
  return (
    <span className={`pe-ptype-pill pe-ptype-pill--${meta.hue}`}>{meta.label}</span>
  );
}

interface CreateProjectModalProps {
  open: boolean;
  onClose: () => void;
  onCreated: (project: any) => void;
  /** Names already in use, for the client-side pre-check. */
  existingNames: string[];
}

export default function CreateProjectModal({
  open, onClose, onCreated, existingNames,
}: CreateProjectModalProps) {
  const [name, setName] = useState("");
  const [projectType, setProjectType] = useState<ProjectType | null>(null);
  const [creating, setCreating] = useState(false);
  const [error, setError] = useState<string | null>(null);

  if (!open) return null;

  function close() {
    if (creating) return;
    onClose();
    setName("");
    setProjectType(null);
    setError(null);
  }

  async function handleCreate() {
    const trimmed = name.trim();
    if (!trimmed || !projectType || creating) return;
    if (isDuplicateName(existingNames, trimmed)) {
      setError(DUPLICATE_NAME_MESSAGE);
      return;
    }
    setCreating(true);
    try {
      const { data } = await projectsApi.create({ name: trimmed, project_type: projectType });
      toast.success("Project created");
      onCreated(data);
      onClose();
      setName("");
      setProjectType(null);
      setError(null);
    } catch (e: any) {
      // Server-side dedupe (race with another tab) lands here.
      if (e?.response?.status === 409) {
        setError(DUPLICATE_NAME_MESSAGE);
      } else {
        toast.error(e?.response?.data?.detail || "Failed to create project");
      }
    } finally {
      setCreating(false);
    }
  }

  function handleTypeKeyDown(e: React.KeyboardEvent, idx: number) {
    if (!["ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown"].includes(e.key)) return;
    e.preventDefault();
    const dir = e.key === "ArrowLeft" || e.key === "ArrowUp" ? -1 : 1;
    const next = (idx + dir + PROJECT_TYPE_ORDER.length) % PROJECT_TYPE_ORDER.length;
    const nextType = PROJECT_TYPE_ORDER[next];
    setProjectType(nextType);
    document.getElementById(`pe-ptype-card-${nextType}`)?.focus();
  }

  return createPortal(
    <div
      className="overlay-modal fixed inset-0 z-[100] flex items-center justify-center"
      onClick={close}
    >
      <div
        className="pe-create-modal pe-create-modal--wide surface-raised rounded-2xl"
        onClick={(e) => e.stopPropagation()}
      >
        <h2 className="pe-create-modal-title">Create a new project</h2>
        <label className="pe-create-modal-label" htmlFor="pe-new-project-name">
          Enter the name for your new project:
        </label>
        <input
          id="pe-new-project-name"
          className="input pe-create-modal-input"
          placeholder="Enter a name for this project"
          value={name}
          onChange={(e) => {
            setName(e.target.value);
            if (error) setError(null);
          }}
          onKeyDown={(e) => e.key === "Enter" && handleCreate()}
          autoFocus
          disabled={creating}
          aria-invalid={error ? "true" : "false"}
          aria-describedby={error ? "pe-create-error" : undefined}
          autoComplete="off"
          spellCheck={false}
        />

        <div className="pe-ptype-divider">
          <span>Choose a project type</span>
        </div>
        <div className="pe-ptype-grid" role="radiogroup" aria-label="Project type">
          {PROJECT_TYPE_ORDER.map((type, idx) => {
            const meta = PROJECT_TYPE_META[type];
            const Icon = meta.icon;
            const selected = projectType === type;
            return (
              <button
                key={type}
                id={`pe-ptype-card-${type}`}
                type="button"
                role="radio"
                aria-checked={selected}
                tabIndex={selected || (!projectType && idx === 0) ? 0 : -1}
                className={`pe-ptype-card ${selected ? "is-selected" : ""}`}
                data-hue={meta.hue}
                onClick={() => setProjectType(type)}
                onKeyDown={(e) => handleTypeKeyDown(e, idx)}
                disabled={creating}
              >
                <span className={`pe-ptype-tile pe-ptype-tile--${meta.hue}`} aria-hidden="true">
                  <Icon size={20} />
                </span>
                <span className="pe-ptype-title">{meta.label}</span>
                <span className="pe-ptype-desc">{meta.description}</span>
              </button>
            );
          })}
        </div>

        {error && (
          <p id="pe-create-error" role="alert" className="pe-create-modal-error">
            {error}
          </p>
        )}

        <div className="pe-create-modal-actions">
          <button
            type="button"
            className="btn-secondary px-5 py-2 text-sm font-medium"
            onClick={close}
            disabled={creating}
          >
            Cancel
          </button>
          <button
            type="button"
            className="btn-primary px-5 py-2 text-sm font-medium"
            onClick={handleCreate}
            disabled={creating || !name.trim() || !projectType}
          >
            {creating ? "Saving…" : "Save"}
          </button>
        </div>
      </div>
    </div>,
    document.body
  );
}
