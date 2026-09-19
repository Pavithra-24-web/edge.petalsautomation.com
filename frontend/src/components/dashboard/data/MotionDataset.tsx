"use client";
import { useCallback, useRef, useState } from "react";
import { useDropzone } from "react-dropzone";
import { useRouter } from "next/navigation";
import { useAppStore } from "@/store/appStore";
import { samplesApi } from "@/utils/api";
import {
  Upload, UploadCloud, FolderOpen, Files, ChevronDown,
  CheckCircle2, XCircle, Loader2,
  FileText, Database,
  Activity, Info,
} from "lucide-react";
import toast from "react-hot-toast";
import { CollectFromDevice } from "./CollectFromDevice";
import MotionShellNotice from "../motion/MotionShellNotice";

// ─── Types ────────────────────────────────────────────────────────────────────

type UploadMode = "training" | "testing" | "automatic" | "postprocessing";

interface QueuedFile {
  id: string;
  file: File;
  relativePath: string;
  status: "pending" | "uploading" | "done" | "error";
  progress: number;
}

// ─── Constants ────────────────────────────────────────────────────────────────
// Motion sensor readings ship as tabular / structured data — CSV first, plus
// the same structured formats the OD dropzone already accepts. No image,
// audio or video MIME types: those affordances are hidden here, not filtered
// server-side (the upload endpoint is modality-agnostic, same as today).

const UPLOAD_MODE_OPTIONS: { value: UploadMode; label: string; desc: string }[] = [
  { value: "training", label: "Training data", desc: "All files go to the training set" },
  { value: "testing", label: "Testing data", desc: "All files go to the test set" },
  {
    value: "automatic", label: "Automatically split between training and testing",
    desc: "Splits the data 80/20 between training and testing data. Uses the hash of the file to determine in which category it's placed, so the same files always end up in the same category."
  },
  { value: "postprocessing", label: "Post-processing", desc: "Uploaded to a post-processing dataset" },
];

const ACCEPTED_MIME: Record<string, string[]> = {
  "text/csv": [".csv"],
  "text/plain": [".txt"],
  "text/xml": [".xml"],
  "text/yaml": [".yaml", ".yml"],
  "application/json": [".json"],
  "application/xml": [".xml"],
  "application/x-yaml": [".yaml", ".yml"],
  "application/cbor": [".cbor"],
  "application/octet-stream": [".cbor", ".parquet"],
};

function humanSize(bytes: number) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

let idSeq = 0;
const nextId = () => String(++idSeq);

// ─── Motion Dataset ───────────────────────────────────────────────────────────

export default function MotionDataset() {
  const { activeProject } = useAppStore();
  const router = useRouter();

  const [uploadMode, setUploadMode] = useState<UploadMode>("automatic");
  const [modeOpen, setModeOpen] = useState(false);
  const modeRef = useRef<HTMLDivElement>(null);

  const [queue, setQueue] = useState<QueuedFile[]>([]);
  const [uploading, setUploading] = useState(false);
  const [batchProgress, setBatchProgress] = useState(0);

  const folderInputRef = useRef<HTMLInputElement>(null);
  const multiInputRef = useRef<HTMLInputElement>(null);

  function enqueueFiles(files: File[], paths?: string[]) {
    const items: QueuedFile[] = files.map((f, i) => ({
      id: nextId(), file: f,
      relativePath: paths?.[i] ?? f.name,
      status: "pending", progress: 0,
    }));
    setQueue(prev => [...prev, ...items]);
  }

  function removeFromQueue(id: string) { setQueue(prev => prev.filter(q => q.id !== id)); }
  function clearQueue() { setQueue([]); }

  const onDrop = useCallback((accepted: File[]) => { enqueueFiles(accepted); }, []);
  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop, accept: ACCEPTED_MIME, noClick: true,
  });

  function handleFolderChange(e: React.ChangeEvent<HTMLInputElement>) {
    const files = Array.from(e.target.files ?? []);
    const paths = files.map(f => (f as any).webkitRelativePath || f.name);
    enqueueFiles(files, paths);
    e.target.value = "";
  }

  function handleMultiChange(e: React.ChangeEvent<HTMLInputElement>) {
    enqueueFiles(Array.from(e.target.files ?? []));
    e.target.value = "";
  }

  async function startUpload() {
    if (!activeProject) return toast.error("Select a project first");
    const pending = queue.filter(q => q.status === "pending");
    if (!pending.length) return toast.error("No files queued");

    setUploading(true);
    setBatchProgress(0);

    const form = new FormData();
    form.append("project_id", activeProject.id);
    form.append("sample_type", uploadMode);

    setQueue(prev => prev.map(q =>
      q.status === "pending" ? { ...q, status: "uploading" as const } : q
    ));

    for (const item of pending) {
      const renamed = new File([item.file], item.relativePath, { type: item.file.type });
      form.append("files", renamed);
    }

    try {
      const { data } = await samplesApi.uploadBatch(form, (pct) => setBatchProgress(pct));

      setQueue(prev => prev.map(q =>
        q.status === "uploading" ? { ...q, status: "done" as const, progress: 100 } : q
      ));
      toast.success(`Uploaded ${data.uploaded} file${data.uploaded !== 1 ? "s" : ""}`);

      setTimeout(() => router.push("/dashboard/data/dataset"), 800);
    } catch (err: any) {
      const msg = err?.response?.data?.detail ?? "Upload failed";
      setQueue(prev => prev.map(q =>
        q.status === "uploading" ? { ...q, status: "error" as const } : q
      ));
      toast.error(msg);
    } finally {
      setUploading(false);
      setBatchProgress(0);
    }
  }

  if (!activeProject) {
    return (
      <div className="flex items-center justify-center h-64">
        <p className="text-gray-500">Select a project from the sidebar to get started.</p>
      </div>
    );
  }

  const selectedMode = UPLOAD_MODE_OPTIONS.find(o => o.value === uploadMode)!;
  const pendingCount = queue.filter(q => q.status === "pending").length;

  return (
    <div className="pe-acq data-upload-page max-w-5xl mx-auto space-y-6">
      <style>{`
        @keyframes pe-cloud-float {
          0%   { transform: translateY(0)    rotate(0deg);   }
          25%  { transform: translateY(-5px) rotate(-1.5deg); }
          50%  { transform: translateY(-8px) rotate(0deg);   }
          75%  { transform: translateY(-5px) rotate(1.5deg);  }
          100% { transform: translateY(0)    rotate(0deg);   }
        }
        @keyframes pe-cloud-glow {
          0%, 100% { filter: drop-shadow(0 2px 4px rgba(139, 92, 246, 0.15)); }
          50%      { filter: drop-shadow(0 8px 18px rgba(139, 92, 246, 0.45)); }
        }
        .pe-acq-cloud {
          animation: pe-cloud-float 3.4s cubic-bezier(0.45, 0, 0.55, 1) infinite,
                     pe-cloud-glow  3.4s ease-in-out infinite;
          will-change: transform, filter;
          transition: transform 220ms cubic-bezier(0.34, 1.56, 0.64, 1);
          transform-origin: center bottom;
        }
        .pe-acq-dropzone.is-drag .pe-acq-cloud {
          transform: scale(1.12) translateY(-4px);
          animation-duration: 1.4s, 1.4s;
        }
        @media (prefers-reduced-motion: reduce) {
          .pe-acq-cloud { animation: none; }
        }
      `}</style>

      {/* Upload Mode selector */}
      <div>
        <label className="pe-acq-field-label" htmlFor="upload-mode-btn">Upload category</label>
        <div className={`pe-acq-select ${modeOpen ? "is-open" : ""}`} ref={modeRef}>
          <button
            id="upload-mode-btn"
            type="button"
            className="pe-acq-select-btn"
            onClick={() => setModeOpen(o => !o)}
            aria-haspopup="listbox"
            aria-expanded={modeOpen}
          >
            <span className="truncate">{selectedMode.label}</span>
          </button>
          <ChevronDown size={16} className="chevron" aria-hidden="true" />
          {modeOpen && (
            <div className="pe-acq-select-menu" role="listbox">
              {UPLOAD_MODE_OPTIONS.map(opt => (
                <button
                  key={opt.value}
                  type="button"
                  role="option"
                  aria-selected={opt.value === uploadMode}
                  className={`pe-acq-select-item ${opt.value === uploadMode ? "is-selected" : ""}`}
                  onClick={() => { setUploadMode(opt.value); setModeOpen(false); }}
                >
                  <p className="opt-label">{opt.label}</p>
                  <p className="opt-desc">{opt.desc}</p>
                </button>
              ))}
            </div>
          )}
        </div>
      </div>

      {/* Drag-and-drop zone — motion sensor data only */}
      <div
        {...getRootProps()}
        className={`pe-acq-dropzone ${isDragActive ? "is-drag" : ""}`}
      >
        <input {...getInputProps()} />
        <div className="pe-acq-cloud" aria-hidden="true">
          <UploadCloud size={26} strokeWidth={1.75} />
        </div>
        <p className="pe-acq-drop-title">Drag &amp; drop sensor recordings here</p>
        <p className="pe-acq-drop-sub">
          You can upload CSV, JSON, XML, YAML, CBOR or Parquet motion recordings.
        </p>

        <div className="pe-acq-drop-actions">
          <button id="pick-files-btn" type="button" className="data-acq-btn-primary"
            onClick={() => multiInputRef.current?.click()}>
            <Files size={14} className="acc" /> Choose files
          </button>
          <button id="pick-folder-btn" type="button" className="data-acq-btn-primary"
            onClick={() => folderInputRef.current?.click()}>
            <FolderOpen size={14} /> Upload folder
          </button>
        </div>

        <input ref={multiInputRef} type="file" multiple className="hidden"
          onChange={handleMultiChange}
          accept=".csv,.txt,.xml,.yaml,.yml,.json,.cbor,.parquet" />
        <input ref={folderInputRef} type="file" className="hidden"
          onChange={handleFolderChange}
          // @ts-ignore
          webkitdirectory="" multiple />
      </div>

      {/* Queue */}
      {queue.length > 0 && (
        <div className="space-y-2 mt-5">
          <div className="flex items-center justify-between mb-1">
            <p className="text-[11px] font-semibold uppercase tracking-wider" style={{ color: "var(--app-text-soft)" }}>
              File queue ({queue.length})
            </p>
            <button onClick={clearQueue} className="text-xs transition-colors" style={{ color: "var(--app-text-muted)" }}>
              Clear all
            </button>
          </div>

          <div className="max-h-56 overflow-y-auto space-y-1.5 pr-1">
            {queue.map(item => (
              <div key={item.id}
                className="data-queue-item flex items-center gap-3 rounded-lg px-3 py-2 group"
                style={{ background: "var(--app-surface-2)", border: "1px solid var(--app-border)" }}
              >
                <span style={{ color: "var(--app-text-soft)" }}>
                  {["csv", "json", "cbor", "parquet"].includes(item.file.name.split(".").pop()?.toLowerCase() ?? "")
                    ? <Database size={13} /> : <FileText size={13} />}
                </span>
                <div className="flex-1 min-w-0">
                  <p className="text-xs truncate font-mono" style={{ color: "var(--app-text)" }}>{item.relativePath}</p>
                  <p className="text-[10px]" style={{ color: "var(--app-text-soft)" }}>{humanSize(item.file.size)}</p>
                </div>
                {item.status === "done" && <CheckCircle2 size={14} className="text-green-500 shrink-0" />}
                {item.status === "error" && <XCircle size={14} className="text-red-500 shrink-0" />}
                {item.status === "uploading" && <Loader2 size={14} className="shrink-0 animate-spin" style={{ color: "#8b5cf6" }} />}
                {item.status === "pending" && (
                  <button onClick={() => removeFromQueue(item.id)}
                    className="shrink-0 opacity-0 group-hover:opacity-100 transition-opacity"
                    style={{ color: "var(--app-text-soft)" }}>
                    <XCircle size={14} />
                  </button>
                )}
              </div>
            ))}
          </div>

          {uploading && (
            <div className="w-full rounded-full h-1.5 mt-2" style={{ background: "var(--app-surface-2)" }}>
              <div className="h-1.5 rounded-full transition-all duration-300"
                style={{ width: `${batchProgress}%`, background: "linear-gradient(90deg, #7c3aed, #a855f7)" }} />
            </div>
          )}

          <div className="flex justify-end pt-1">
            <button
              id="start-upload-btn"
              onClick={startUpload}
              disabled={uploading || pendingCount === 0}
              className="data-acq-btn-primary disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {uploading
                ? <><Loader2 size={14} className="animate-spin" /> Uploading…</>
                : <><Upload size={14} /> Upload {pendingCount} file{pendingCount !== 1 ? "s" : ""}</>
              }
            </button>
          </div>
        </div>
      )}

      {/* Collect from a connected device — live USB recording, shared with Object Detection */}
      <CollectFromDevice projectId={activeProject.id} />

      {/* Signal preview — arrives with motion feature generation */}
      <div className="card">
        <div className="flex items-center gap-2 mb-1">
          <Activity size={16} className="text-gray-400" />
          <h3 className="text-sm font-semibold text-gray-300">Signal preview</h3>
        </div>
        <MotionShellNotice
          icon={Activity}
          title="Signal preview arrives with motion processing"
          description="Once a recording is uploaded or collected, its per-axis waveform will render here for a quick sanity check before you label it."
        />
      </div>

      <div className="pe-acq-info-note mt-3 flex items-start gap-2 rounded-md px-3 py-2 text-xs">
        <Info size={14} className="pe-acq-info-icon mt-[2px] shrink-0" aria-hidden="true" />
        <p className="leading-relaxed">
          <span className="pe-acq-info-strong font-semibold">Motion recordings:</span>{" "}
          Upload sensor CSV exports directly, or record live from a connected device above.
          Each recording becomes a sample you can label on the Data labeling page.
        </p>
      </div>
    </div>
  );
}
