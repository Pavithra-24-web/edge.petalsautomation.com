"use client";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useRouter } from "next/navigation";
import Link from "next/link";
import toast from "react-hot-toast";
import {
  Home, ChevronRight, ArrowLeft, Wand2, HelpCircle, Loader2,
  CheckCircle, AlertTriangle, ImageIcon, Eye, EyeOff,
} from "lucide-react";
import { useAppStore } from "@/store/appStore";
import { syntheticDataApi } from "@/utils/api";
import { notifySamplesChanged } from "../labeling/labelingEvents";

// Synthetic Data plan, Phase 4 (docs/Action/datasynthetic_implementationplan.md
// §5.3). Consumes the Phase 3 backend as-is — no new backend logic here.

const SPLIT_OPTIONS: { value: string; label: string }[] = [
  { value: "training", label: "Training" },
  { value: "testing", label: "Test" },
  { value: "postprocessing", label: "Post-processing" },
  { value: "automatic", label: "80-20 Training-Test" },
];

// Polling ceiling (plan §2.6 / §5.3) — if a job is still `running` after this
// long the poller stops and tells the user to check the Dataset page rather
// than spinning forever.
const POLL_INTERVAL_MS = 2000;
const POLL_CEILING_MS = 10 * 60 * 1000;

type Phase = "form" | "submitting" | "running" | "completed" | "failed";

function formatSize(apiSize: string) {
  return apiSize.replace("x", " × ");
}

// Defense-in-depth (belt-and-suspenders with the backend's own guarantee
// that a job never reaches `completed` with zero generated samples): the UI
// never announces success on `status === "completed"` alone — it also
// requires a real generated count and at least one recorded sample id, so a
// status-only regression on either side can't resurrect a false "success".
export function isRealSuccess(job: any): boolean {
  return (
    job?.status === "completed" &&
    (job?.generated_count ?? 0) > 0 &&
    Array.isArray(job?.sample_ids) &&
    job.sample_ids.length > 0
  );
}

function Help({ text }: { text: string }) {
  return (
    <span title={text} className="inline-flex align-middle cursor-help">
      <HelpCircle size={13} className="opacity-50" />
    </span>
  );
}

function FormRow({
  label, required, help, children,
}: { label: string; required?: boolean; help?: string; children: React.ReactNode }) {
  return (
    <div className="grid grid-cols-1 md:grid-cols-[200px_1fr] gap-x-6 gap-y-1.5 items-start">
      <label className="flex items-center gap-1.5 text-sm pt-2" style={{ color: "var(--app-text-muted)" }}>
        <span>{label}</span>
        {required && <span className="text-red-500">*</span>}
        {help && <Help text={help} />}
      </label>
      <div>{children}</div>
    </div>
  );
}

export default function SyntheticData() {
  const { activeProject } = useAppStore();
  const projectId = activeProject?.id;
  const router = useRouter();

  const [config, setConfig] = useState<any>(null);
  const [configLoading, setConfigLoading] = useState(true);

  // User-provided OpenAI key (plan §0/§5.3) — lives only in this component's
  // state for the current browser session; never written to localStorage,
  // never logged, sent only as part of a generate() request body.
  const [apiKey, setApiKey] = useState("");
  const [showApiKey, setShowApiKey] = useState(false);

  const [prompt, setPrompt] = useState("");
  const [label, setLabel] = useState("");
  const [count, setCount] = useState(3);
  const [size, setSize] = useState("1024x1024");
  const [quality, setQuality] = useState("standard");
  const [background, setBackground] = useState("auto");
  const [sampleType, setSampleType] = useState("automatic");

  const [phase, setPhase] = useState<Phase>("form");
  const [job, setJob] = useState<any>(null);

  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const pollStartRef = useRef<number>(0);

  useEffect(() => {
    syntheticDataApi.config()
      .then(({ data }) => {
        setConfig(data);
        if (data.sizes?.length && !data.sizes.includes(size)) setSize(data.sizes[0]);
        if (data.qualities?.length && !data.qualities.includes(quality)) setQuality(data.qualities[0]);
        if (data.backgrounds?.length && !data.backgrounds.includes(background)) setBackground(data.backgrounds[0]);
      })
      .catch(() => setConfig({ sizes: [], qualities: [], backgrounds: [], price_per_image_usd: {} }))
      .finally(() => setConfigLoading(false));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const stopPolling = useCallback(() => {
    if (pollRef.current) { clearInterval(pollRef.current); pollRef.current = null; }
  }, []);

  useEffect(() => () => stopPolling(), [stopPolling]);

  const pollJob = useCallback((jobId: string) => {
    stopPolling();
    pollStartRef.current = Date.now();
    pollRef.current = setInterval(async () => {
      if (Date.now() - pollStartRef.current > POLL_CEILING_MS) {
        stopPolling();
        toast.error("Still generating — check the Dataset page shortly for results.");
        return;
      }
      try {
        const { data: updated } = await syntheticDataApi.getJob(jobId);
        setJob(updated);
        if (updated.status === "completed" && isRealSuccess(updated)) {
          stopPolling();
          setPhase("completed");
          notifySamplesChanged();
        } else if (updated.status === "failed" || updated.status === "completed") {
          // A `status === "completed"` that fails `isRealSuccess` should be
          // impossible (the backend never marks a job completed with zero
          // samples) — if it ever happens anyway, this still refuses to
          // show a success state for it.
          stopPolling();
          setPhase("failed");
        }
      } catch {
        stopPolling();
        setPhase("failed");
      }
    }, POLL_INTERVAL_MS);
  }, [stopPolling]);

  const priceKey = `${size}:${quality}`;
  const perImagePrice: number | undefined = config?.price_per_image_usd?.[priceKey];
  const estimatedCost = perImagePrice !== undefined ? perImagePrice * count : undefined;

  const maxImages = config?.max_images_per_job ?? 25;
  const canSubmit = apiKey.trim().length > 0 && prompt.trim().length > 0 && label.trim().length > 0
    && count >= 1 && count <= maxImages;

  const handleSubmit = async () => {
    if (!projectId || !canSubmit) return;
    setPhase("submitting");
    try {
      const { data } = await syntheticDataApi.generate({
        project_id: projectId,
        api_key: apiKey.trim(),
        prompt: prompt.trim(),
        label: label.trim(),
        count,
        sample_type: sampleType,
        size,
        quality,
        background,
      });
      setJob(data);
      setPhase("running");
      pollJob(data.id);
    } catch (err: any) {
      const message = err?.response?.data?.detail || "Failed to start synthetic data generation.";
      toast.error(message);
      setPhase("form");
    }
  };

  const handleGenerateMore = () => {
    setJob(null);
    setPhase("form");
    setShowApiKey(false);
  };

  const handleTryAgain = () => {
    setJob(null);
    setPhase("form");
    setShowApiKey(false);
  };

  const splitLabel = useMemo(
    () => SPLIT_OPTIONS.find(o => o.value === sampleType)?.label ?? sampleType,
    [sampleType],
  );

  return (
    <div className="dataset-page data-acq-page w-full space-y-4">
      {/* Header — breadcrumb / title */}
      <div className="min-w-0 flex-1">
        <div className="data-acq-header-top">
          <button
            type="button"
            onClick={() => router.push("/dashboard/data/dataset")}
            className="data-acq-back-btn"
            aria-label="Back to Data Labeling"
          >
            <ArrowLeft size={13} strokeWidth={2.25} />
            <span>Back</span>
          </button>
          <span className="header-sep" aria-hidden="true" />
          <nav className="data-acq-breadcrumb" aria-label="Breadcrumb">
            <Home size={13} aria-hidden="true" />
            <ChevronRight size={12} aria-hidden="true" />
            <Link href="/dashboard/data/dataset" className="crumb-link">Data Labeling</Link>
            <ChevronRight size={12} aria-hidden="true" />
            <span className="crumb-current">Synthetic Data</span>
          </nav>
        </div>
        <h1 className="data-acq-title">Synthetic Data</h1>
        <p className="data-acq-subtitle">
          Generate labelled images for your dataset with OpenAI. Your prompt is sent to
          OpenAI — no sample images or other project data leave this deployment.
        </p>
      </div>

      <div className="rounded-xl overflow-hidden" style={{ background: "var(--app-surface)", border: "1px solid var(--app-border)", boxShadow: "var(--app-shadow)" }}>
        <div className="px-6 py-4" style={{ borderBottom: "1px solid var(--app-border)" }}>
          <h2 className="text-base font-semibold" style={{ color: "var(--app-text)" }}>Create synthetic data</h2>
        </div>

        {configLoading ? (
          <div className="flex items-center justify-center gap-2 py-16" style={{ color: "var(--app-text-muted)" }}>
            <Loader2 size={18} className="animate-spin" /> Loading…
          </div>
        ) : phase === "running" ? (
          <div className="px-6 py-10 flex flex-col items-center text-center gap-3">
            <Loader2 size={28} className="animate-spin" style={{ color: "var(--app-text-muted)" }} />
            <p className="text-sm font-medium" style={{ color: "var(--app-text)" }}>
              Generating {job?.requested_count ?? count} image{(job?.requested_count ?? count) === 1 ? "" : "s"}…
            </p>
            <div className="w-full max-w-sm h-2 rounded-full overflow-hidden" style={{ background: "var(--app-surface-2)" }}>
              <div
                className="h-full rounded-full transition-all"
                style={{
                  width: `${job?.requested_count ? Math.min(100, ((job.generated_count + job.failed_count) / job.requested_count) * 100) : 0}%`,
                  background: "linear-gradient(180deg, #6366f1 0%, #4338ca 100%)",
                }}
              />
            </div>
            <p className="text-xs" style={{ color: "var(--app-text-muted)" }}>
              {job?.generated_count ?? 0} of {job?.requested_count ?? count} generated
              {job?.failed_count ? ` · ${job.failed_count} failed` : ""}
            </p>
            <p className="text-xs" style={{ color: "var(--app-text-soft)" }}>
              You can leave this page — generation continues on the server.
            </p>
          </div>
        ) : phase === "completed" ? (
          <div className="px-6 py-10 flex flex-col items-center text-center gap-3">
            <CheckCircle size={28} className="text-green-500" />
            <p className="text-sm font-medium" style={{ color: "var(--app-text)" }}>
              {job?.generated_count ?? 0} of {job?.requested_count ?? count} images generated and added to
              {" "}{splitLabel}{job?.failed_count ? ` — ${job.failed_count} failed` : ""}.
            </p>
            {job?.failed_count > 0 && job?.generated_count > 0 && (
              <p className="text-xs" style={{ color: "var(--app-text-muted)" }}>
                Some images were rejected or failed — the rest are already in your dataset.
              </p>
            )}
            <div className="flex items-center gap-3 mt-2">
              <button type="button" className="data-acq-btn-secondary" onClick={handleGenerateMore}>
                Generate more
              </button>
              <button
                type="button"
                className="data-acq-btn-primary"
                onClick={() => router.push("/dashboard/data/dataset")}
              >
                <ImageIcon size={14} /> View in dataset
              </button>
            </div>
          </div>
        ) : phase === "failed" ? (
          <div className="px-6 py-10 flex flex-col items-center text-center gap-3">
            <AlertTriangle size={28} className="text-red-500" />
            <p className="text-sm font-medium" style={{ color: "var(--app-text)" }}>Generation failed</p>
            <p className="text-xs max-w-md" style={{ color: "var(--app-text-muted)" }}>
              {job?.error_message || "Something went wrong while generating images."}
            </p>
            {job?.generated_count > 0 && (
              <p className="text-xs" style={{ color: "var(--app-text-muted)" }}>
                {job.generated_count} image{job.generated_count === 1 ? "" : "s"} generated before the failure
                {job.generated_count === 1 ? " is" : " are"} already in your dataset.
              </p>
            )}
            <button type="button" className="data-acq-btn-primary mt-2" onClick={handleTryAgain}>
              Try again
            </button>
          </div>
        ) : (
          <>
            <div className="p-6 space-y-6">
              <FormRow label="Select a synthetic data source">
                <select className="input" disabled value="openai">
                  <option value="openai">{config?.provider_label || "OpenAI Image Generation"}</option>
                </select>
              </FormRow>

              <FormRow label="Description">
                <p className="text-sm" style={{ color: "var(--app-text-muted)" }}>
                  Use OpenAI&apos;s image generation model to create synthetic images for your
                  project. Generated images are marked with their generation prompt and model so
                  they stay auditable later.
                </p>
              </FormRow>

              <div>
                <p className="text-sm font-medium mb-3" style={{ color: "var(--app-text)" }}>Parameters</p>
                <div className="space-y-4">
                  <FormRow
                    label="OpenAI API Key"
                    required
                    help="Your own OpenAI API key. It is sent with this request and never stored — billing lands on your OpenAI account, not this deployment's."
                  >
                    <div className="relative">
                      <input
                        type={showApiKey ? "text" : "password"}
                        className="input pr-10"
                        placeholder="sk-…"
                        autoComplete="off"
                        value={apiKey}
                        disabled={phase === "submitting"}
                        onChange={e => setApiKey(e.target.value)}
                      />
                      <button
                        type="button"
                        onClick={() => setShowApiKey(v => !v)}
                        aria-label={showApiKey ? "Hide API key" : "Show API key"}
                        className="absolute inset-y-0 right-0 flex items-center px-3"
                        style={{ color: "var(--app-text-muted)" }}
                      >
                        {showApiKey ? <EyeOff size={14} /> : <Eye size={14} />}
                      </button>
                    </div>
                  </FormRow>

                  <FormRow label="Prompt" required help="Describe the image to generate, as you would to an image model.">
                    <textarea
                      className="input"
                      rows={4}
                      maxLength={4000}
                      placeholder="A photo of a factory worker wearing a hard hat"
                      value={prompt}
                      disabled={phase === "submitting"}
                      onChange={e => setPrompt(e.target.value)}
                    />
                  </FormRow>

                  <FormRow label="Label" required help="The label applied to every generated image.">
                    <input
                      type="text"
                      className="input"
                      maxLength={128}
                      placeholder="hard_hat"
                      value={label}
                      disabled={phase === "submitting"}
                      onChange={e => setLabel(e.target.value)}
                    />
                  </FormRow>

                  <FormRow label="Number of images" help={`Between 1 and ${maxImages} per job.`}>
                    <input
                      type="number"
                      className="input"
                      min={1}
                      max={maxImages}
                      value={count}
                      disabled={phase === "submitting"}
                      onChange={e => setCount(Math.max(1, Math.min(maxImages, Number(e.target.value) || 1)))}
                    />
                  </FormRow>

                  <FormRow label="Image size">
                    <select
                      className="input"
                      value={size}
                      disabled={phase === "submitting"}
                      onChange={e => setSize(e.target.value)}
                    >
                      {(config?.sizes || ["1024x1024", "1024x1536", "1536x1024"]).map((s: string) => (
                        <option key={s} value={s}>{formatSize(s)}</option>
                      ))}
                    </select>
                  </FormRow>

                  <FormRow label="Quality">
                    <select
                      className="input"
                      value={quality}
                      disabled={phase === "submitting"}
                      onChange={e => setQuality(e.target.value)}
                    >
                      {(config?.qualities || ["standard", "high"]).map((q: string) => (
                        <option key={q} value={q}>{q === "standard" ? "Standard" : "High"}</option>
                      ))}
                    </select>
                  </FormRow>

                  <FormRow label="Background">
                    <select
                      className="input"
                      value={background}
                      disabled={phase === "submitting"}
                      onChange={e => setBackground(e.target.value)}
                    >
                      {(config?.backgrounds || ["transparent", "opaque", "auto"]).map((b: string) => (
                        <option key={b} value={b}>{b.charAt(0).toUpperCase() + b.slice(1)}</option>
                      ))}
                    </select>
                  </FormRow>

                  <FormRow label="Dataset split" help="Which split newly generated images are added to.">
                    <select
                      className="input"
                      value={sampleType}
                      disabled={phase === "submitting"}
                      onChange={e => setSampleType(e.target.value)}
                    >
                      {SPLIT_OPTIONS.map(o => (
                        <option key={o.value} value={o.value}>{o.label}</option>
                      ))}
                    </select>
                  </FormRow>
                </div>
              </div>

              <div className="flex justify-end">
                <p className="text-sm" style={{ color: "var(--app-text-muted)" }}>
                  {estimatedCost !== undefined
                    ? <>Estimated cost: ~${estimatedCost.toFixed(2)} ({count} image{count === 1 ? "" : "s"})</>
                    : <>Estimated cost unavailable for this combination</>}
                </p>
              </div>
            </div>

            <div className="px-6 py-4 flex justify-end" style={{ borderTop: "1px solid var(--app-border)" }}>
              <button
                type="button"
                className="data-acq-btn-primary disabled:opacity-50 disabled:cursor-not-allowed"
                disabled={!canSubmit || phase === "submitting"}
                onClick={handleSubmit}
              >
                {phase === "submitting" ? <Loader2 size={14} className="animate-spin" /> : <Wand2 size={14} />}
                Generate Synthetic Data
              </button>
            </div>
          </>
        )}
      </div>
    </div>
  );
}
