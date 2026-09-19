"use client";
import { useEffect, useState, useCallback, useRef, useLayoutEffect } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import { useAppStore } from "@/store/appStore";
import { trainingApi, impulsesApi, dspApi, labelsApi } from "@/utils/api";
import { RefreshCw, FlaskConical, X, HelpCircle, MoreVertical, Rocket, ChevronDown, ChevronUp, Terminal, CircleDashed, Activity, CheckCircle2, XCircle, AlertTriangle } from "lucide-react";
import { TrainingLogOutput } from "@/app/dashboard/impulse/_components/TrainingLogOutput";
import { TrainingGraphs } from "@/app/dashboard/impulse/_components/TrainingGraphs";
import ImpulseNotReady from "@/components/dashboard/ImpulseNotReady";
import { invalidateTrainingStatus } from "@/hooks/useTrainingValidity";
import toast from "react-hot-toast";

const POLL_MS = 2500;

type JobStatus = "pending" | "running" | "completed" | "failed" | "cancelled";

const STATUS_META: Record<
  JobStatus,
  { label: string; sub: string; icon: React.ComponentType<any> }
> = {
  pending: { label: "Queued", sub: "Waiting in the build queue…", icon: CircleDashed },
  running: { label: "Running", sub: "Training in progress", icon: Activity },
  completed: { label: "Completed", sub: "Model build finished", icon: CheckCircle2 },
  failed: { label: "Failed", sub: "Model build failed", icon: XCircle },
  cancelled: { label: "Cancelled", sub: "Build was cancelled", icon: AlertTriangle },
};

const MODELS: Array<{
  id: string;
  name: string;
  author: string;
  description: string;
  official: boolean;
  preview: boolean;
  unavailable?: boolean;
}> = [
    {
      id: "mobilenet_v2_ssd_fpn_lite",
      name: "EdgeDetect Lite (320x320 only)",
      author: "Petal Edge",
      description: "EdgeDetect Lite for object detection. Good for devices with > 512KB RAM or Linux.",
      official: true,
      preview: false,
    },
    {
      id: "fomo_mobilenetv2_0_1",
      name: "NanoVision MobileNetV2 0.35",
      author: "Petal Edge",
      description: "NanoVision is a novel object detection architecture that's specifically designed to run on constrained devices.",
      official: true,
      preview: false,
    },
    {
      id: "yolo_pro",
      name: "Vision Pro",
      author: "Petal Edge",
      description: "Vision Pro state of the art object detection. Good for Linux devices or constrained edge devices with NPUs.",
      official: false,
      preview: true,
    }
  ];

const YOLO_PRO_SIZE_OPTIONS = [
  { value: "nano", label: "Nano", description: "smallest, fastest" },
  { value: "tiny", label: "Tiny", description: "small" },
  { value: "small", label: "Small", description: "balanced" },
  { value: "medium", label: "Medium", description: "larger" },
  { value: "large", label: "Large", description: "largest, slowest" },
] as const;

function getDefaultLearningRateForYoloSize(size: string): number {
  if (size === "nano") return 0.0007;
  return 0.001;
}

// Initialization-only epoch suggestion for first-time configs. Derived purely
// from existing frontend metadata (no API calls). The per-architecture band
// sets the floor/ceiling; dataset size moves the value continuously within it,
// so the result scales — there is no fixed constant.
//   small  + classification → 50–75
//   medium + FOMO           → 75–100
//   large  + YOLO-Pro       → 100–150
function getRecommendedEpochs(args: {
  totalSamples: number | null;
  labelCount: number | null;
  architectureId: string;
  featureCount: number | null;
}): number {
  const { totalSamples, labelCount, architectureId, featureCount } = args;

  // Heavier detectors need more cycles to converge; FOMO sits in the middle;
  // a lightweight classification head (none selectable yet) would need fewest.
  let min: number;
  let max: number;
  if (architectureId === "yolo_pro") {
    min = 100;
    max = 150;
  } else if (architectureId === "mobilenet_v2_ssd_fpn_lite") {
    // Mid-heavy detector: above FOMO, below YOLO-Pro.
    min = 90;
    max = 130;
  } else if (architectureId === "fomo_mobilenetv2_0_1") {
    min = 60;
    max = 100;
  } else {
    // Fallback band reserved for a future classification head.
    min = 50;
    max = 75;
  }

  const samples = totalSamples ?? 0;
  const classes = Math.max(1, labelCount ?? 1);
  const features = featureCount ?? 0;

  // Effective dataset size: samples per class, lightly weighted by input
  // dimensionality (richer feature vectors warrant a few more cycles). The
  // feature term is log-damped so it nudges rather than dominates.
  const perClass = samples / classes;
  const sizeSignal = perClass * (1 + Math.log10(1 + features) / 4);

  // Map ~50 samples/class → band floor, ~1500+ → band ceiling, clamped.
  const lo = 50;
  const hi = 1500;
  const t = Math.min(1, Math.max(0, (sizeSignal - lo) / (hi - lo)));

  return Math.round(min + t * (max - min));
}

function getYoloProSizeOption(size: string) {
  return YOLO_PRO_SIZE_OPTIONS.find(option => option.value === size) || null;
}

function getYoloProSizeLabel(size: string): string {
  const option = getYoloProSizeOption(size);
  return option ? `${option.label} (${option.description})` : size;
}

function getArchitectureIdFromImpulse(impulse: any): string | null {
  const block = impulse?.ml_blocks?.[0];
  return block?.type || block?.architecture || null;
}

function getYoloProSizeFromImpulse(impulse: any): string | null {
  const block = impulse?.ml_blocks?.[0];
  const size = block?.params?.size || block?.size || null;
  return typeof size === "string" ? size : null;
}

function findModelById(id: string | null) {
  return MODELS.find(model => model.id === id) || null;
}

function HelpTooltip({ text }: { text: string }) {
  const triggerRef = useRef<HTMLButtonElement>(null);
  const [open, setOpen] = useState(false);
  const [position, setPosition] = useState({ top: 0, left: 0, arrowLeft: 0 });

  const updatePosition = useCallback(() => {
    const trigger = triggerRef.current;
    if (!trigger) return;

    const rect = trigger.getBoundingClientRect();
    const tooltipWidth = 320;
    const gutter = 16;
    const idealLeft = rect.left + rect.width / 2 - tooltipWidth / 2;
    const left = Math.min(
      Math.max(gutter, idealLeft),
      window.innerWidth - tooltipWidth - gutter
    );

    setPosition({
      top: rect.top - 12,
      left,
      arrowLeft: rect.left + rect.width / 2 - left,
    });
  }, []);

  useLayoutEffect(() => {
    if (!open) return;

    updatePosition();
    window.addEventListener("resize", updatePosition);
    window.addEventListener("scroll", updatePosition, true);
    return () => {
      window.removeEventListener("resize", updatePosition);
      window.removeEventListener("scroll", updatePosition, true);
    };
  }, [open, updatePosition]);

  return (
    <span className="relative inline-flex items-center">
      <button
        ref={triggerRef}
        type="button"
        tabIndex={0}
        aria-label={text}
        onMouseEnter={() => setOpen(true)}
        onMouseLeave={() => setOpen(false)}
        onFocus={() => setOpen(true)}
        onBlur={() => setOpen(false)}
        className="inline-flex cursor-help items-center justify-center text-gray-600 transition-colors hover:text-gray-400 focus:text-gray-400 focus:outline-none"
      >
        <HelpCircle size={12} />
      </button>
      {open && (
        <span
          className="pointer-events-none fixed z-50 w-80 -translate-y-full rounded-xl bg-[#373768] px-4 py-3 text-left text-sm leading-relaxed text-white shadow-2xl"
          style={{
            top: position.top,
            left: position.left,
          }}
        >
          {text}
          <span
            className="absolute top-full h-3 w-3 -translate-x-1/2 -translate-y-1/2 rotate-45 bg-[#373768]"
            style={{ left: position.arrowLeft }}
          />
        </span>
      )}
    </span>
  );
}

function ChooseModelModal({ onAdd, onClose }: { onAdd: (model: any) => void, onClose: () => void }) {
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm">
      <div className="train-modal rounded-2xl shadow-2xl w-[680px] max-h-[85vh] flex flex-col overflow-hidden">
        <div className="flex items-center justify-between px-6 py-4 border-b border-gray-800">
          <div className="flex items-center gap-2">
            <FlaskConical size={17} className="text-brand-400" />
            <h2 className="text-[15px] font-semibold text-gray-200">Choose a different model</h2>
          </div>
          <button onClick={onClose} className="text-gray-500 hover:text-gray-300">
            <X size={17} />
          </button>
        </div>

        <div className="px-6 py-2.5 text-sm text-gray-500 text-center border-b border-gray-800 bg-gray-900/30">
          <strong>Did you know?</strong> You can{" "}
          <a href="#" className="text-brand-400 hover:underline">customize this model further</a>
          {" "}if you are an expert, or <a href="#" className="text-brand-400 hover:underline">bring your own model</a>.
        </div>

        <div className="grid grid-cols-[1fr_130px_100px] px-6 py-2 bg-gray-900 border-b border-gray-800">
          <span className="text-[11px] font-semibold text-gray-500 uppercase tracking-wider">Description</span>
          <span className="text-[11px] font-semibold text-gray-500 uppercase tracking-wider">Author</span>
          <span className="text-[11px] font-semibold text-gray-500 uppercase tracking-wider text-right pr-2"></span>
        </div>

        <div className="flex-1 overflow-y-auto divide-y divide-gray-800">
          {MODELS.map(model => (
            <div key={model.id}
              className={`grid grid-cols-[1fr_130px_100px] items-center px-6 py-4 transition-colors ${model.unavailable ? "opacity-60" : "hover:bg-gray-800"}`}>
              <div className="pr-4">
                <div className="flex items-center flex-wrap gap-2 mb-1">
                  <span className="text-sm font-semibold text-gray-200">{model.name}</span>
                  {model.official && (
                    <span className="px-1.5 py-px rounded-full bg-brand-900/20 text-brand-400 text-[10px] font-semibold uppercase tracking-wide border border-brand-800/50">
                      OFFICIALLY SUPPORTED
                    </span>
                  )}
                  {model.preview && (
                    <span className="px-1.5 py-px rounded-full bg-purple-900/20 text-purple-400 text-[10px] font-semibold uppercase tracking-wide border border-purple-800/50">
                      DEVELOPER PREVIEW
                    </span>
                  )}
                  {model.unavailable && (
                    <span className="px-1.5 py-px rounded-full bg-gray-800/80 text-gray-400 text-[10px] font-semibold uppercase tracking-wide border border-gray-700">
                      UNAVAILABLE
                    </span>
                  )}
                </div>
                <p className="text-sm text-gray-400 leading-snug">{model.description}</p>
              </div>
              <span className="text-sm text-gray-500">{model.author}</span>
              <div className="flex items-center justify-end gap-2 pr-1">
                <button
                  disabled={model.unavailable}
                  onClick={() => { onAdd(model); onClose(); }}
                  className="px-4 py-1.5 text-sm font-medium text-brand-400 border border-brand-700/50 rounded-lg transition-colors hover:bg-brand-900/40 disabled:cursor-not-allowed disabled:border-gray-700 disabled:text-gray-500 disabled:hover:bg-transparent"
                >
                  {model.unavailable ? "Unavailable" : "Add"}


                </button>
              </div>
            </div>
          ))}
        </div>

        <div className="px-6 py-3 border-t border-gray-800 bg-gray-900">
          <div className="flex justify-end">
            <button onClick={onClose}
              className="rounded-lg bg-brand-600 px-3 py-1.5 text-white shadow-sm transition-colors hover:bg-brand-500 disabled:cursor-not-allowed disabled:opacity-40">
              Cancel
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

export default function ObjectDetectionTraining() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const { activeProject, activeImpulse: storeActiveImpulse, setActiveImpulse: setStoreActiveImpulse } = useAppStore();
  const requestedImpulseId = searchParams.get("impulseId");

  const [activeImpulse, setActiveImpulse] = useState<any>(null);
  const [activeJob, setActiveJob] = useState<any>(null);
  const [isTraining, setIsTraining] = useState(false);
  const [showModelModal, setShowModelModal] = useState(false);
  const [logCollapsed, setLogCollapsed] = useState(false);
  const [graphsCollapsed, setGraphsCollapsed] = useState(true);
  // null = checking, true = ready, false = not generated
  const [featuresReady, setFeaturesReady] = useState<boolean | null>(null);
  const [inputSizeInfo, setInputSizeInfo] = useState<{
    feature_count: number | null;
    message: string;
    loading: boolean;
  }>({ feature_count: null, message: "", loading: false });

  const [labelCount, setLabelCount] = useState<number | null>(null);
  const [totalSamples, setTotalSamples] = useState<number | null>(null);

  // Smart-epoch bookkeeping. The recommendation recalculates whenever its
  // inputs change, but must never override a restored config (priority 1,
  // tracked via activeJob) or a user edit (priority 2, tracked via
  // epochsTouched). statusLoaded gates the recommendation behind loadStatus
  // resolving so restore is known first.
  const epochsTouched = useRef(false);
  const [statusLoaded, setStatusLoaded] = useState(false);

  const [config, setConfig] = useState({
    epochs: 60,
    learning_rate: 0.001,
    batch_size: 32,
    data_augmentation: true,
  });

  const [selectedArchitecture, setSelectedArchitecture] = useState(MODELS[1]);
  // Sent to /training/start as `device_preference`. Defaults to "cpu" — the API
  // default is "gpu", but a gpu request is now rejected (503) rather than
  // downgraded when GPU training is unavailable, so cpu is the default that
  // always queues. Overriding the API default here is deliberate.
  const [devicePreference, setDevicePreference] = useState<"cpu" | "gpu">("cpu");
  const [yoloProSize, setYoloProSize] = useState("nano");
  const [lrIsAuto, setLrIsAuto] = useState(true);
  const logEndRef = useRef<HTMLDivElement>(null);
  const isFomoSelected =
    selectedArchitecture.id === "fomo_mobilenetv2_0_1" ||
    selectedArchitecture.id === "fomo_v2_0.35";
  const isYoloProSelected = selectedArchitecture.id === "yolo_pro";
  const isAutoAugmentationModel = isFomoSelected || isYoloProSelected;

  const dspRoute: string =
    activeImpulse?.input_type === "image" ||
      (activeImpulse?.dsp_blocks ?? []).some((b: any) => b?.type === "image")
      ? "image"
      : (activeImpulse?.dsp_blocks?.[0]?.type ?? "image");

  // Load the explicitly selected impulse when present; otherwise fall back to the
  // store selection and only then to the first impulse in the project.
  useEffect(() => {
    if (!activeProject) return;
    impulsesApi.list(activeProject.id).then(r => {
      if (r.data.length === 0) return;
      const matchedImpulse =
        r.data.find((imp: any) => imp.id === requestedImpulseId) ||
        r.data.find((imp: any) => imp.id === storeActiveImpulse?.id) ||
        r.data[0];
      setActiveImpulse(matchedImpulse);
      setStoreActiveImpulse(matchedImpulse);
      const savedModel = findModelById(getArchitectureIdFromImpulse(matchedImpulse));
      if (savedModel) {
        setSelectedArchitecture(savedModel);
      }
      const savedYoloSize = getYoloProSizeFromImpulse(matchedImpulse);
      if (savedYoloSize) {
        setYoloProSize(savedYoloSize);
      }
      loadStatus(matchedImpulse.id);
      fetchInputSize(matchedImpulse.id);
      fetchLabelCount(matchedImpulse.id);
      checkFeaturesReady(matchedImpulse.id);
    }).catch(() => { });
  }, [activeProject?.id, requestedImpulseId, storeActiveImpulse?.id]);

  useEffect(() => {
    if (!storeActiveImpulse) return;
    if (requestedImpulseId && storeActiveImpulse.id !== requestedImpulseId) return;
    setActiveImpulse(storeActiveImpulse);
  }, [storeActiveImpulse, requestedImpulseId]);

  useEffect(() => {
    if (!activeImpulse) return;
    setActiveJob(null);
    setIsTraining(false);
    // Fresh impulse — re-run the restore→recommendation sequence from scratch.
    setStatusLoaded(false);
    epochsTouched.current = false;
    loadStatus(activeImpulse.id);
    fetchInputSize(activeImpulse.id);
    fetchLabelCount(activeImpulse.id);
    checkFeaturesReady(activeImpulse.id);
  }, [activeImpulse?.id]);

  useEffect(() => {
    const savedModel = findModelById(getArchitectureIdFromImpulse(activeImpulse));
    if (savedModel) {
      setSelectedArchitecture(savedModel);
    }
    const savedYoloSize = getYoloProSizeFromImpulse(activeImpulse);
    if (savedYoloSize) {
      setYoloProSize(savedYoloSize);
    }
  }, [activeImpulse?.id, activeImpulse?.ml_blocks]);

  useEffect(() => {
    if (!isAutoAugmentationModel) return;
    setConfig(prev => (prev.data_augmentation ? prev : { ...prev, data_augmentation: true }));
  }, [isAutoAugmentationModel]);

  // Track the last impulse we ran a readiness check for, so React StrictMode
  // double-invokes / overlapping effects don't repeat the request.
  const readinessCheckedFor = useRef<string | null>(null);
  const checkFeaturesReady = async (impId: string) => {
    if (readinessCheckedFor.current === impId) return;
    readinessCheckedFor.current = impId;
    setFeaturesReady(null);
    // Hard cap: if the backend stalls, default to "not ready" rather than
    // leaving the UI on the spinner forever. The warning path lets the user
    // navigate to Generate Features and retry; a stuck spinner does not.
    const timeoutMs = 4000;
    const timeout = new Promise<{ data: { ready: false; _timeout: true } }>(resolve =>
      setTimeout(() => resolve({ data: { ready: false, _timeout: true } }), timeoutMs)
    );
    try {
      const { data } = await Promise.race([dspApi.featuresReady(impId), timeout]);
      setFeaturesReady(data?.ready === true);
    } catch {
      setFeaturesReady(false);
    }
  };

  const fetchLabelCount = async (impulseId: string) => {
    try {
      const { data } = await dspApi.datasetSummary(impulseId);
      setLabelCount(data.num_classes);
      setTotalSamples(data.total_samples ?? null);
    } catch (e) {
      console.error("Failed to fetch label count", e);
    }
  };

  const fetchInputSize = async (impId: string) => {
    setInputSizeInfo({ feature_count: null, message: "", loading: true });
    try {
      const { data } = await dspApi.inputSize(impId);
      setInputSizeInfo({
        feature_count: data.feature_count,
        message: data.message || "",
        loading: false,
      });
    } catch (e: any) {
      setInputSizeInfo({
        feature_count: null,
        message: "Input layer unavailable: could not reach server.",
        loading: false,
      });
    }
  };

  const loadStatus = async (impId: string) => {
    try {
      const { data: jobs } = await trainingApi.listJobs(impId);
      // Pick the most recent regular training run. Retrain jobs live on
      // /dashboard/impulse/retrain — this page must keep showing the prior
      // "start" run's logs even after a retrain has happened.
      const job = (jobs as any[]).find((j) => j?.launch_mode !== "retrain");
      if (job) {
        setActiveJob(job);
        if (job.status === "running" || job.status === "pending") {
          setIsTraining(true);
        }
        // Restore training config from the most recent job so that navigating
        // away and back never reverts form fields to hardcoded defaults.
        setConfig(prev => ({
          epochs: job.requested_epochs ?? job.epochs ?? prev.epochs,
          learning_rate: job.learning_rate ?? prev.learning_rate,
          batch_size: job.batch_size ?? prev.batch_size,
          data_augmentation: job.data_augmentation ?? prev.data_augmentation,
        }));
        // Restore architecture from the job (takes precedence over ml_blocks when present).
        if (job.architecture) {
          const restoredModel = findModelById(job.architecture);
          if (restoredModel) setSelectedArchitecture(restoredModel);
        }
        if (job.architecture === "yolo_pro" && job.model_size) {
          setYoloProSize(job.model_size);
        }
      }
    } catch { }
    finally {
      // Restore (priority 1) is now resolved — activeJob reflects whether a
      // config was restored. Only now may the recommendation effect run.
      setStatusLoaded(true);
    }
  };

  // Smart epoch recommendation. Runs after loadStatus resolves (statusLoaded)
  // so restore is decided first, then recalculates whenever its inputs change
  // (architecture, samples, classes, feature count). Never overrides a restored
  // config (activeJob present) or a user edit (epochsTouched). If it doesn't
  // apply, the hardcoded default (60) stands.
  useEffect(() => {
    if (!statusLoaded) return;                 // wait for restore (priority 1)
    if (activeJob !== null) return;            // restored config — keep its value
    if (epochsTouched.current) return;         // user edit — never overwrite
    if (totalSamples === null || labelCount === null) return; // metadata not ready
    if (inputSizeInfo.loading) return;         // wait for input size to resolve

    const recommended = getRecommendedEpochs({
      totalSamples,
      labelCount,
      architectureId: selectedArchitecture.id,
      featureCount: inputSizeInfo.feature_count,
    });
    setConfig(prev => ({ ...prev, epochs: recommended }));
  }, [statusLoaded, activeJob, totalSamples, labelCount, selectedArchitecture.id, inputSizeInfo.feature_count, inputSizeInfo.loading]);

  // Poll
  useEffect(() => {
    if (!isTraining || !activeJob) return;
    const interval = setInterval(async () => {
      try {
        const { data } = await trainingApi.getJob(activeJob.id);
        setActiveJob(data);
        if (data.status === "completed" || data.status === "failed" || data.status === "cancelled") {
          setIsTraining(false);
          clearInterval(interval);
          // Re-derive `hasValidTrainingOutput` everywhere — on completion it
          // flips to true, on fail/cancel it flips to false and result panels
          // collapse back to placeholders.
          invalidateTrainingStatus(activeImpulse?.id);
          if (data.status === "completed") {
            toast.success("Job completed");
          } else if (data.status === "cancelled") {
            toast.success("Job cancelled");
          } else {
            // Error messages can be multi-paragraph (e.g. the feature-resolution
            // mismatch); the toast shows the headline, the log panel the rest.
            toast.error(
              "Job failed: " +
              String(data.error_message || "Unknown error").split("\n")[0]
            );
          }
        }
      } catch (e) {
        console.error(e);
      }
    }, POLL_MS);
    return () => clearInterval(interval);
  }, [isTraining, activeJob?.id]);

  // Auto-scroll logs whenever new lines arrive or status changes
  useEffect(() => {
    if (logEndRef.current) logEndRef.current.scrollIntoView({ behavior: "smooth" });
  }, [activeJob?.training_history?.log_lines?.length, activeJob?.status]);

  async function startTraining() {
    if (!activeImpulse) return toast.error("No active impulse");
    setIsTraining(true);
    setLogCollapsed(false);
    try {
      const { data } = await trainingApi.start({
        impulse_id: activeImpulse.id,
        epochs: config.epochs,
        learning_rate: config.learning_rate,
        batch_size: config.batch_size,
        device_preference: devicePreference,
        data_augmentation: isAutoAugmentationModel ? true : config.data_augmentation,
        extra_params: {
          architecture: selectedArchitecture.id,
          ...(isYoloProSelected ? { size: yoloProSize } : {}),
          ...(isFomoSelected ? { fomo_version: 1 } : {}),
        }
      });
      setActiveJob(data);
      // The latest run is now "pending"/"running" — clear any stale "trained"
      // state on result pages so old artifacts don't show alongside a live run.
      invalidateTrainingStatus(activeImpulse?.id);
      toast.success("Training job scheduled");
    } catch (e: any) {
      // /training/start returns a structured detail ({ code, message }) when a
      // GPU run can't be served — gpu_disabled or gpu_worker_unavailable. Other
      // errors are still bare strings, so handle both. Nothing else changes:
      // activeJob is untouched (no job row is shown as started) and the form
      // stays submittable with the device selection intact.
      const detail = e?.response?.data?.detail;
      const message =
        (detail && typeof detail === "object" && detail.message) ||
        (typeof detail === "string" && detail) ||
        e?.message ||
        "Failed to start";
      toast.error(message);
      setIsTraining(false);
    }
  }

  async function cancelTraining() {
    if (!activeJob) return;
    try {
      await trainingApi.cancelJob(activeJob.id);
      setActiveJob((prev: any) => prev ? ({
        ...prev,
        status: "cancelled",
        error_message: "Cancelled by user",
        completed_at: new Date().toISOString(),
      }) : prev);
      setIsTraining(false);
      invalidateTrainingStatus(activeImpulse?.id);
      toast.success("Job cancelled");
    } catch (e: any) {
      toast.error("Failed to cancel job");
    }
  }

  const dspBlockName =
    (activeImpulse?.dsp_blocks?.[0]?.name as string | undefined) ||
    (dspRoute ? dspRoute.charAt(0).toUpperCase() + dspRoute.slice(1) : "Image");

  // Until both activeImpulse and featuresReady resolve, we can't tell whether
  // to render the main UI or the "Almost there!" warning. Render a loading
  // state first so the warning is the first non-loading frame for draft
  // impulses (where activeImpulse loads from the store after hydration).
  const isInitialLoad = !!activeProject && (!activeImpulse || featuresReady === null);
  if (isInitialLoad) {
    return (
      <div className="flex h-[calc(100vh-theme('spacing.16'))] items-center justify-center pt-2">
        <span className="inline-flex items-center gap-2 text-sm text-[var(--app-text-soft)]">
          <RefreshCw size={14} className="animate-spin" />
          Verifying feature readiness...
        </span>
      </div>
    );
  }

  if (activeImpulse && featuresReady === false) {
    return (
      <ImpulseNotReady
        description={
          <>
            DSP Block &quot;{dspBlockName}&quot; has no generated features.
            <br />
            Generate features before training your model.
          </>
        }
        tip="Features must be generated from your DSP block before the model can be trained."
        actions={
          <button
            type="button"
            onClick={() => router.push(`/dashboard/impulse/${dspRoute}/generate-features?impulseId=${activeImpulse.id}`)}
            className="pe-warn-cta"
          >
            <Rocket size={15} /> Go to Generate features
          </button>
        }
      />
    );
  }

  return (
    <div className="training-premium-bg flex h-[calc(100vh-theme('spacing.16'))] flex-col overflow-hidden pt-2">
      <div className="grid min-h-0 flex-1 grid-cols-1 gap-5 pb-6 xl:grid-cols-[minmax(460px,0.94fr)_minmax(520px,1.06fr)]">

        {/* Left Panel: Configuration */}
        <div className="min-h-0 overflow-hidden overflow-y-auto rounded-2xl border border-[var(--app-border)] bg-[var(--app-surface)] shadow-[var(--app-shadow)]">
          {/* Header */}
          <div className="panel-header-premium sticky top-0 z-10 flex items-center justify-between rounded-t-2xl border-b border-[var(--app-border)] px-5 py-4">
            <div>
              <p className="panel-header-eyebrow text-[11px] font-semibold uppercase tracking-[0.16em]">Configuration</p>
              <h2 className="panel-header-title mt-1 text-base font-semibold">Neural Network settings</h2>
            </div>
            <button className="rounded-lg p-1.5 text-[var(--app-text-soft)] transition-colors hover:bg-white/10 hover:text-[var(--app-text)]" aria-label="More neural network settings">
              <MoreVertical size={18} />
            </button>
          </div>

          <div className="space-y-8 p-5 sm:p-6">
            {/* Training settings */}
            <div className="space-y-4">
              <div>
                <h3 className="text-sm font-semibold text-[var(--app-text)]">Training settings</h3>
                <p className="mt-1 text-xs text-[var(--app-text-soft)]">Tune the training budget and runtime behavior for this impulse.</p>
              </div>

              <div className="space-y-1">
                <div className="grid gap-2 border-b border-[var(--app-border)] py-3 sm:grid-cols-[1fr_180px] sm:items-center">
                  <div className="flex items-center gap-1.5 text-sm text-[var(--app-text-muted)]">
                    Maximum training cycles <HelpTooltip text="Maximum number of training cycles. Training will stop at this number, or earlier if validation accuracy stops improving for 20 consecutive cycles." />
                  </div>
                  <input type="number"
                    value={config.epochs}
                    onChange={e => {
                      epochsTouched.current = true;
                      setConfig({ ...config, epochs: Number(e.target.value) });
                    }}
                    className="w-full rounded-lg border border-[var(--app-border)] bg-[var(--app-surface-2)] px-3 py-2 text-sm text-[var(--app-text)] shadow-sm focus:border-brand-500 focus:outline-none focus:ring-2 focus:ring-brand-500/20"
                  />
                </div>

                <div className="grid gap-2 border-b border-[var(--app-border)] py-3 sm:grid-cols-[1fr_180px] sm:items-center">
                  <div className="flex items-center gap-1.5 text-sm text-[var(--app-text-muted)]">
                    Learning rate <HelpTooltip text="How fast the neural network learns. If the network overfits quickly, lower the learning rate." />
                  </div>
                  <input type="number" step="0.0001"
                    value={config.learning_rate}
                    onChange={e => {
                      setConfig({ ...config, learning_rate: Number(e.target.value) });
                      setLrIsAuto(false);
                    }}
                    className="w-full rounded-lg border border-[var(--app-border)] bg-[var(--app-surface-2)] px-3 py-2 text-sm text-[var(--app-text)] shadow-sm focus:border-brand-500 focus:outline-none focus:ring-2 focus:ring-brand-500/20"
                  />
                </div>

                {(selectedArchitecture.id === "fomo_mobilenetv2_0_1"
                  || selectedArchitecture.id === "fomo_v2_0.35"
                  || selectedArchitecture.id === "yolo_pro") && (
                    <div className="grid gap-2 border-b border-[var(--app-border)] py-3 sm:grid-cols-[1fr_180px] sm:items-center">
                      <div className="flex items-center gap-1.5 text-sm text-[var(--app-text-muted)]">
                        Batch size{" "}
                        <HelpTooltip text="The batch size used during training. If not set, we'll use the default value. Training may fail if the batch size is too high." />
                      </div>
                      <input
                        type="number"
                        min={1}
                        value={config.batch_size}
                        onChange={e => setConfig({ ...config, batch_size: Math.max(1, Number(e.target.value)) })}
                        className="w-full rounded-lg border border-[var(--app-border)] bg-[var(--app-surface-2)] px-3 py-2 text-sm text-[var(--app-text)] shadow-sm focus:border-brand-500 focus:outline-none focus:ring-2 focus:ring-brand-500/20"
                      />
                    </div>
                  )}

                {selectedArchitecture.id === "yolo_pro" && (
                  <div className="grid gap-2 border-b border-[var(--app-border)] py-3 sm:grid-cols-[1fr_210px] sm:items-center">
                    <div className="flex items-center gap-1.5 text-sm text-[var(--app-text-muted)]">
                      Model size <HelpTooltip text="Vision Pro capacity order: Nano < Tiny < Small < Medium < Large. Smaller models train and run faster; larger models have more capacity." />
                    </div>
                    <div className="space-y-1">
                      <select
                        value={yoloProSize}
                        onChange={e => {
                          const newSize = e.target.value;
                          setYoloProSize(newSize);
                          if (lrIsAuto) {
                            setConfig(c => ({ ...c, learning_rate: getDefaultLearningRateForYoloSize(newSize) }));
                          }
                        }}
                        className="w-full rounded-lg border border-[var(--app-border)] bg-[var(--app-surface-2)] px-3 py-2 pr-10 text-sm text-[var(--app-text)] shadow-sm focus:border-brand-500 focus:outline-none focus:ring-2 focus:ring-brand-500/20"
                      >
                        {YOLO_PRO_SIZE_OPTIONS.map(option => (
                          <option key={option.value} value={option.value}>
                            {option.label} ({option.description})
                          </option>
                        ))}
                      </select>
                      <p className="text-[11px] text-[var(--app-text-soft)]">
                        Selected: {getYoloProSizeLabel(yoloProSize)}
                      </p>
                    </div>
                  </div>
                )}

                <div className="grid gap-2 border-b border-[var(--app-border)] py-3 sm:grid-cols-[1fr_210px] sm:items-center">
                  <div className="flex items-center gap-1.5 text-sm text-[var(--app-text-muted)]">
                    Training device <HelpTooltip text="Which worker runs this job. GPU trains faster where GPU training is available; where it isn't, the job is rejected with an error rather than moved to CPU." />
                  </div>
                  <div className="space-y-1">
                    <select
                      value={devicePreference}
                      onChange={e => setDevicePreference(e.target.value as "cpu" | "gpu")}
                      className="w-full rounded-lg border border-[var(--app-border)] bg-[var(--app-surface-2)] px-3 py-2 pr-10 text-sm text-[var(--app-text)] shadow-sm focus:border-brand-500 focus:outline-none focus:ring-2 focus:ring-brand-500/20"
                    >
                      <option value="gpu">GPU</option>
                      <option value="cpu">CPU</option>
                    </select>
                  </div>
                </div>

                <div className="grid gap-2 py-3 sm:grid-cols-[1fr_180px] sm:items-center">
                  <div className="flex items-center gap-1.5 text-sm text-[var(--app-text-muted)]">
                    Data augmentation <HelpTooltip text="Randomly transform data during training. Allows you to run more training cycles without overfitting, which can improve accuracy." />
                  </div>
                  <div>
                    <input type="checkbox"
                      checked={isAutoAugmentationModel ? true : config.data_augmentation}
                      disabled={isAutoAugmentationModel}
                      onChange={e => setConfig({ ...config, data_augmentation: e.target.checked })}
                      className="h-4 w-4 rounded border-[var(--app-border)] bg-[var(--app-surface-2)] text-brand-500 focus:ring-brand-500 disabled:cursor-not-allowed disabled:opacity-70"
                    />
                  </div>
                </div>
              </div>
            </div>

            {/* Neural network architecture */}
            <div className="space-y-4">
              <div>
                <h3 className="text-sm font-semibold text-[var(--app-text)]">Neural network architecture</h3>
                <p className="mt-1 text-xs text-[var(--app-text-soft)]">A quick map of the data flowing into the selected model.</p>
              </div>

              <div className="space-y-2">
                {/* Input Layer */}
                <div className={`rounded-xl border px-4 py-3 text-center text-sm font-semibold shadow-sm ${inputSizeInfo.feature_count != null ? "border-brand-500 bg-brand-600 text-white" : "border-[var(--app-border)] bg-[var(--app-surface-2)] text-[var(--app-text-muted)]"
                  }`}>
                  {inputSizeInfo.loading ? (
                    <span className="flex items-center justify-center gap-2">
                      <RefreshCw size={13} className="animate-spin" />
                      Computing input layer...
                    </span>
                  ) : inputSizeInfo.feature_count != null ? (
                    `Input layer (${inputSizeInfo.feature_count.toLocaleString()} features)`
                  ) : (
                    <span title={inputSizeInfo.message}>
                      {inputSizeInfo.message || "Input layer - configure DSP first"}
                    </span>
                  )}
                </div>

                {/* Model Box */}
                <div className="flex flex-col items-center justify-center gap-3 rounded-2xl border border-dashed border-brand-500/40 bg-brand-500/10 p-5 text-center">
                  <div className="flex h-11 w-11 items-center justify-center rounded-2xl bg-brand-500/10 text-brand-500 ring-1 ring-brand-500/20">
                    <FlaskConical size={24} />
                  </div>
                  <p className="max-w-[34rem] text-sm font-semibold text-[var(--app-text)]">{selectedArchitecture.name}</p>
                  {selectedArchitecture.id === "yolo_pro" && (
                    <p className="text-center text-xs text-[var(--app-text-soft)]">
                      Size: {getYoloProSizeLabel(yoloProSize)}
                    </p>
                  )}

                  <button onClick={() => setShowModelModal(true)} disabled={isTraining}
                    className="mt-2 w-full rounded-xl border border-dashed border-[var(--app-border)] bg-[var(--app-surface)] px-3 py-2 text-sm font-medium text-[var(--app-text-muted)] transition-colors hover:border-brand-500/50 hover:text-[var(--app-text)] disabled:opacity-50">
                    Choose a different model
                  </button>
                </div>

                {/* Output Layer */}
                <div className={`rounded-xl border px-4 py-3 text-center text-sm font-semibold shadow-sm ${labelCount !== null && labelCount > 0 ? "border-[var(--app-border)] bg-[var(--app-surface-2)] text-[var(--app-text)]" : "border-[var(--app-border)] bg-[var(--app-surface-2)] text-[var(--app-text-muted)]"
                  }`}>
                  {labelCount === null ? (
                    "Loading labels..."
                  ) : labelCount > 0 ? (
                    `Output layer (${labelCount} ${labelCount === 1 ? 'class' : 'classes'})`
                  ) : (
                    "Output layer unavailable: no labels created"
                  )}
                </div>
              </div>

            </div>
          </div>
        </div>

        {/* Right Panel: Training output */}
        <div
          className={`flex flex-col overflow-hidden rounded-2xl border border-[var(--app-border)] bg-[var(--app-surface)] shadow-[var(--app-shadow)] ${logCollapsed ? "self-start" : "min-h-[520px]"}`}
        >
          <div className={`panel-header-premium flex flex-col gap-4 rounded-t-2xl px-5 py-4 sm:flex-row sm:items-center sm:justify-between ${logCollapsed ? "" : "border-b border-[var(--app-border)]"}`}>
            <div>
              <p className="panel-header-eyebrow text-[11px] font-semibold uppercase tracking-[0.16em]">Run log</p>
              <h2 className="panel-header-title mt-1 text-base font-semibold">Training output</h2>
            </div>
            <div className="flex items-center gap-4 text-sm font-semibold">
              {activeJob?.status && STATUS_META[activeJob.status as JobStatus] && (() => {
                const meta = STATUS_META[activeJob.status as JobStatus];
                const isLive = activeJob.status === "running" || activeJob.status === "pending";
                const chipClass = `bo-chip bo-chip--${activeJob.status}${isLive ? "" : " bo-chip--static"}`;
                return (
                  <span className={chipClass}>
                    <span className="bo-chip__dot" />
                    {meta.label}
                  </span>
                );
              })()}
              {isTraining ? (
                <button onClick={cancelTraining} className="rounded-lg bg-brand-600 px-3 py-1.5 text-white shadow-sm transition-colors hover:bg-brand-500 disabled:cursor-not-allowed disabled:opacity-40">Cancel</button>
              ) : (
                <button
                  onClick={startTraining}
                  disabled={featuresReady !== true}
                  title={featuresReady === null ? "Checking feature readiness..." : undefined}
                  className="rounded-lg bg-brand-600 px-3 py-1.5 text-white shadow-sm transition-colors hover:bg-brand-500 disabled:cursor-not-allowed disabled:opacity-40"
                >
                  Start training
                </button>
              )}
              <button
                type="button"
                onClick={() => setLogCollapsed(v => !v)}
                aria-label={logCollapsed ? "Expand training output" : "Collapse training output"}
                aria-expanded={!logCollapsed}
                title={logCollapsed ? "Expand" : "Collapse"}
                className="rounded-lg p-1.5 text-[var(--app-text-soft)] transition-colors hover:bg-white/10 hover:text-[var(--app-text)]"
              >
                {logCollapsed ? <ChevronDown size={16} /> : <ChevronUp size={16} />}
              </button>
            </div>
          </div>

          {!logCollapsed && (
            <div className="build-output__log-wrap">
              <span className="build-output__log-label">
                <Terminal size={12} />
                Output log
              </span>
              <div className="bo-log-frame">
                <div className="bo-log-frame__bar">
                  <span className="bo-log-frame__dot bo-log-frame__dot--r" />
                  <span className="bo-log-frame__dot bo-log-frame__dot--y" />
                  <span className="bo-log-frame__dot bo-log-frame__dot--g" />
                  <span className="bo-log-frame__name">
                    {activeJob ? `job-${String(activeJob.id).substring(0, 8)}.log` : "training.log"}
                  </span>
                </div>
                <div className="bo-log-frame__body train-log-area">
                  <TrainingLogOutput
                    job={activeJob}
                    isTraining={isTraining}
                    logEndRef={logEndRef}
                    fallbackEpochs={config.epochs}
                    emptyMessage="No recent jobs matching this configuration. Click Start training to begin."
                  />
                </div>
              </div>

              <div className="build-output__graphs mt-5">
                <button
                  type="button"
                  onClick={() => setGraphsCollapsed((v) => !v)}
                  className="build-output__log-label inline-flex cursor-pointer items-center gap-1.5 bg-transparent p-0 hover:text-[var(--app-text)]"
                  aria-expanded={!graphsCollapsed}
                  aria-controls="training-graphs-body"
                >
                  <Activity size={12} />
                  Training graphs
                  {graphsCollapsed ? <ChevronDown size={12} /> : <ChevronUp size={12} />}
                </button>
                {!graphsCollapsed && (
                  <div id="training-graphs-body" className="mt-2">
                    <TrainingGraphs
                      epochMetrics={activeJob?.training_history?.epoch_metrics}
                      isRunning={activeJob?.status === "running" || activeJob?.status === "pending"}
                    />
                  </div>
                )}
              </div>
            </div>
          )}
        </div>

      </div>

      {showModelModal && (
        <ChooseModelModal
          onAdd={m => setSelectedArchitecture(m)}
          onClose={() => setShowModelModal(false)}
        />
      )}
    </div>
  );
}
