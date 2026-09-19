"use client";
import { useState } from "react";
import toast from "react-hot-toast";
import { projectsApi } from "@/utils/api";

function parseFilenameFromDisposition(header: string | undefined): string | null {
  if (!header) return null;
  const star = /filename\*\s*=\s*[^']*''([^;]+)/i.exec(header);
  if (star && star[1]) {
    try {
      return decodeURIComponent(star[1].trim().replace(/^"|"$/g, ""));
    } catch {
      return star[1].trim().replace(/^"|"$/g, "");
    }
  }
  const plain = /filename\s*=\s*"?([^";]+)"?/i.exec(header);
  return plain && plain[1] ? plain[1].trim() : null;
}

function slugFilename(name: string): string {
  return name
    .trim()
    .replace(/[^A-Za-z0-9._\- ]+/g, "")
    .replace(/\s+/g, "-") || "project";
}

export function useExportProject() {
  const [busy, setBusy] = useState(false);

  async function exportProject(projectId: string, projectName: string) {
    if (busy) return;
    setBusy(true);
    try {
      const res = await projectsApi.export(projectId);
      const blob = res.data instanceof Blob
        ? res.data
        : new Blob([res.data], { type: "application/zip" });

      const headerName = parseFilenameFromDisposition(
        res.headers?.["content-disposition"] as string | undefined,
      );
      const filename = headerName || `${slugFilename(projectName)}-export.zip`;

      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = filename;
      document.body.appendChild(a);
      a.click();
      a.remove();
      setTimeout(() => URL.revokeObjectURL(url), 1000);
    } catch (err: any) {
      const detail = err?.response?.data?.detail;
      toast.error(typeof detail === "string" ? detail : "Failed to export dataset");
    } finally {
      setBusy(false);
    }
  }

  return { exportProject, busy };
}
