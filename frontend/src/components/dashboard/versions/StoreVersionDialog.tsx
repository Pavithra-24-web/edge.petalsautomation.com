"use client";

import { useEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { History, X, Loader2 } from "lucide-react";
import toast from "react-hot-toast";
import { projectVersionsApi } from "@/utils/api";
import type { ProjectVersionDetail } from "./projectVersionTypes";

interface StoreVersionDialogProps {
  projectId: string;
  /** Read-only preview of what's about to be captured (parityfix.md §7.3) —
   * computed from data the page already has, no extra request. */
  summary: { sampleCount: number; classCount: number; impulseCount: number };
  onClose: () => void;
  onCreated: (version: ProjectVersionDetail) => void;
}

/**
 * The only way a project version is ever created (parityfix.md §1.0, §7.3).
 * No hidden defaults are captured here beyond what the user types — the
 * snapshot itself is taken server-side from the project's current state.
 */
export default function StoreVersionDialog({ projectId, summary, onClose, onCreated }: StoreVersionDialogProps) {
  const [name, setName] = useState("");
  const [description, setDescription] = useState("");
  const [saving, setSaving] = useState(false);
  const modalRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    modalRef.current?.focus();
    function handleKey(e: KeyboardEvent) {
      if (e.key === "Escape") onClose();
    }
    document.addEventListener("keydown", handleKey);
    return () => document.removeEventListener("keydown", handleKey);
  }, [onClose]);

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    setSaving(true);
    try {
      const { data } = await projectVersionsApi.create({
        project_id: projectId,
        name: name.trim() || undefined,
        description: description.trim() || undefined,
      });
      toast.success(`Version ${data.version_number} stored`);
      onCreated(data);
    } catch (err: any) {
      toast.error(err?.response?.data?.detail || "Could not store this version");
    } finally {
      setSaving(false);
    }
  }

  if (typeof document === "undefined") return null;

  return createPortal(
    <div
      className="pe-version-overlay"
      onClick={onClose}
      role="dialog"
      aria-modal="true"
      aria-labelledby="pe-pver-store-title"
    >
      <div
        ref={modalRef}
        className="pe-version-modal"
        tabIndex={-1}
        onClick={(e) => e.stopPropagation()}
      >
        <div className="pe-version-modal-head">
          <h3 id="pe-pver-store-title" className="pe-version-modal-title">
            <span className="pe-version-modal-title-icon" aria-hidden="true">
              <History size={16} strokeWidth={2.1} />
            </span>
            Store project version
          </h3>
          <button type="button" onClick={onClose} className="pe-version-modal-close" aria-label="Close">
            <X size={16} />
          </button>
        </div>

        <form onSubmit={handleSubmit}>
          <div className="pe-version-modal-body">
            <p className="pe-version-modal-intro">
              Records this project&apos;s current dataset, every impulse&apos;s block configuration
              and active trained model, plus deployment and post-processing settings — a snapshot
              you can return to later.
            </p>

            <dl className="pe-pver-capture-summary">
              <div className="pe-pver-capture-stat">
                <dt>Samples</dt>
                <dd>{summary.sampleCount.toLocaleString()}</dd>
              </div>
              <div className="pe-pver-capture-stat">
                <dt>Classes</dt>
                <dd>{summary.classCount}</dd>
              </div>
              <div className="pe-pver-capture-stat">
                <dt>Impulses</dt>
                <dd>{summary.impulseCount}</dd>
              </div>
            </dl>

            <label className="pe-version-field">
              <span className="pe-version-field-label">
                Name <span className="pe-version-field-optional">optional</span>
              </span>
              <input
                type="text"
                value={name}
                onChange={(e) => setName(e.target.value)}
                placeholder={`e.g. "Before adding motion class"`}
                className="pe-version-input"
                maxLength={120}
                autoFocus
              />
            </label>

            <label className="pe-version-field">
              <span className="pe-version-field-label">
                Description <span className="pe-version-field-optional">optional</span>
              </span>
              <textarea
                value={description}
                onChange={(e) => setDescription(e.target.value)}
                placeholder="What's different about this snapshot?"
                className="pe-version-textarea"
                rows={3}
                maxLength={2000}
              />
            </label>
          </div>

          <div className="pe-version-modal-foot">
            <button type="button" onClick={onClose} className="pe-version-btn-ghost" disabled={saving}>
              Cancel
            </button>
            <button type="submit" className="pe-version-btn-primary" disabled={saving}>
              {saving ? <Loader2 size={15} className="pe-version-spin" /> : <History size={15} />}
              {saving ? "Storing…" : "Store project version"}
            </button>
          </div>
        </form>
      </div>
    </div>,
    document.body,
  );
}
