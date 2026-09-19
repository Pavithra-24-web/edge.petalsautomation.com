"use client";
import { ArrowLeft, Database, Download } from "lucide-react";
import Link from "next/link";
import { useState } from "react";
import { useAppStore } from "@/store/appStore";

export default function ImportDataPage() {
  const { activeProject } = useAppStore();
  const [sourceProject, setSourceProject] = useState("");

  return (
    <div className="max-w-4xl mx-auto space-y-6">
      <div className="flex items-center gap-4">
        <Link href="/dashboard/data" className="btn-ghost">
          <ArrowLeft size={16} /> Back
        </Link>
        <h2 className="section-title">Import Data</h2>
      </div>

      <div className="card space-y-6">
        <div className="flex items-center gap-3 text-brand-400 mb-2">
          <Database size={24} />
          <h3 className="text-lg font-semibold text-white">Import from another project</h3>
        </div>
        <p className="text-gray-400 text-sm">
          Copy samples from another project you have access to. All associated labels, metadata, and data splits will be preserved during the import.
        </p>

        <div className="space-y-4 max-w-lg mt-6">
          <div>
            <label className="label">Destination project</label>
            <input 
              type="text" 
              className="input bg-gray-900 cursor-not-allowed text-gray-500 border-gray-800"
              disabled
              value={activeProject?.name || "Loading..."}
            />
          </div>

          <div>
            <label className="label">Source project ID or URL</label>
            <input 
              type="text" 
              className="input" 
              placeholder="e.g. 930510"
              value={sourceProject}
              onChange={(e) => setSourceProject(e.target.value)}
            />
          </div>

          <div className="pt-4 flex justify-end">
            <button 
              className="btn-primary flex items-center gap-2"
              disabled={!sourceProject.trim()}
            >
              <Download size={16} /> Import Samples
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
