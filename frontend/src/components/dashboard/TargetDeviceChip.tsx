"use client";
import { useEffect, useState } from "react";
import { Cpu } from "lucide-react";
import { useAppStore } from "@/store/appStore";
import { projectsApi } from "@/utils/api";
import TargetDeviceConfigDialog from "./TargetDeviceConfigDialog";

/**
 * Topbar control stating which board the active project targets. Opens the
 * Target Device Phase 3 configuration dialog (docs/target_device_phase3.md)
 * — the one surface for selecting a board and its application budget.
 *
 * A second, separate picker here would be a second way to change the same
 * value, and the two would eventually diverge — so the chip is a trigger
 * only; every selection interaction lives in the dialog.
 */
export default function TargetDeviceChip() {
  const { activeProject, setActiveProject } = useAppStore();
  const [open, setOpen] = useState(false);

  // `activeProject` is persisted to localStorage, so a session that predates
  // this feature (or predates the Phase 3 configuration fields) holds a
  // project object missing one or both. `undefined` means "this object is
  // stale", which is distinct from `null` meaning "no device chosen" / "no
  // configuration yet" — only the former triggers a refetch.
  const projectId = activeProject?.id;
  const needsHydrate =
    !!activeProject &&
    (activeProject.target_device_slug === undefined ||
      activeProject.target_device_config === undefined);
  useEffect(() => {
    if (!projectId || !needsHydrate) return;
    let cancelled = false;
    projectsApi
      .get(projectId)
      .then(({ data }) => {
        if (!cancelled && data?.id === projectId) setActiveProject(data);
      })
      .catch(() => {
        /* Leave the chip in its unset state; the dialog still works. */
      });
    return () => {
      cancelled = true;
    };
  }, [projectId, needsHydrate, setActiveProject]);

  if (!activeProject) return null;

  const selected = activeProject.target_device ?? null;
  const customName = activeProject.target_device_config?.custom_name;
  const displayName = customName || selected?.display_name;
  const label = selected ? `Target: ${displayName}` : "Set target device";

  return (
    <>
      <button
        type="button"
        onClick={() => setOpen(true)}
        className={`pe-target-chip${selected ? "" : " pe-target-chip--empty"}`}
        aria-haspopup="dialog"
        aria-expanded={open}
        aria-label={
          selected
            ? `Target device: ${displayName}. Configure it.`
            : "No target device set. Choose one."
        }
        title={
          selected?.accelerator_note
            ? `${displayName} — ${selected.accelerator_note}`
            : displayName || "No target device set"
        }
      >
        <Cpu size={15} aria-hidden="true" />
        <span className="pe-target-chip-label">{label}</span>
      </button>

      <TargetDeviceConfigDialog open={open} onClose={() => setOpen(false)} />
    </>
  );
}
