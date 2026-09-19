"use client";
import { Download, Loader2 } from "lucide-react";
import { useExportProject } from "@/hooks/useExportProject";

interface Props {
  projectId: string;
  projectName: string;
  className?: string;
}

export default function ExportDataButton({ projectId, projectName, className }: Props) {
  const { exportProject, busy } = useExportProject();

  return (
    <button
      type="button"
      onClick={() => exportProject(projectId, projectName)}
      disabled={busy}
      className={
        className ||
        "inline-flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-colors bg-indigo-600 hover:bg-indigo-500 text-white disabled:opacity-60 disabled:cursor-not-allowed"
      }
      aria-busy={busy}
    >
      {busy ? <Loader2 size={14} className="animate-spin" /> : <Download size={14} />}
      {busy ? "Preparing export…" : "Export data"}
    </button>
  );
}
