"use client";

import { type ReactNode, useCallback, useEffect, useMemo, useState } from "react";
import {
  AlertTriangle,
  CircuitBoard,
  Clock,
  Cpu,
  HardDrive,
  Lightbulb,
  RefreshCw,
  Rocket,
} from "lucide-react";
import toast from "react-hot-toast";

import { useAppStore } from "@/store/appStore";
import { deploymentApi, trainedModelsApi } from "@/utils/api";
import ImpulseNotReady from "@/components/dashboard/ImpulseNotReady";

import DeploymentHero from "@/app/dashboard/impulse/deployment/DeploymentHero";
import ModelOverview from "@/app/dashboard/impulse/deployment/ModelOverview";
import DeviceSelector, { type DeviceOption } from "@/app/dashboard/impulse/deployment/DeviceSelector";
import DevicePreview from "@/app/dashboard/impulse/deployment/DevicePreview";
import DeploymentActions from "@/app/dashboard/impulse/deployment/DeploymentActions";
import BuildTimeline from "@/app/dashboard/impulse/deployment/BuildTimeline";

const POLL_MS = 3000;

type DeviceIconVariant = "purple" | "amber" | "slate" | "rose" | "blue";

// The legacy Generic-TFLite / Raspberry Pi / UNO Q picker — only shown for a
// project with no target device selected (Target Device Phase 6 §5: that
// path is unchanged, including this list). A project *with* a target device
// never reads this; its deploy target comes from the catalog instead. Each
// entry carries its own `category` for the picker's grouping — no separate
// target→category map, since there is nothing left to derive.
const SUPPORTED_DEVICE_PROFILES: {
  value: string;
  label: string;
  sub: string;
  target: string;
  category: string;
  iconVariant: DeviceIconVariant;
  Icon: React.ComponentType<any>;
}[] = [
    { value: "generic_tflite", label: "Generic tf lite", sub: "", target: "tflite", category: "TensorFlow Lite", iconVariant: "purple", Icon: Cpu },
    { value: "raspberry_pi_4", label: "Raspberry Pi 4", sub: "", target: "raspberry_pi", category: "Raspberry Pi", iconVariant: "rose", Icon: Cpu },
    { value: "unoq", label: "Arduino UNO Q", sub: "", target: "unoq", category: "Arduino", iconVariant: "blue", Icon: CircuitBoard },
  ];

const CATEGORY_ORDER = [
  "TensorFlow Lite",
  "Raspberry Pi",
  "Arduino",
  "Linux",
  "Windows",
  "Custom Device",
];

const CATEGORY_DESCRIPTIONS: Record<string, string> = {
  "TensorFlow Lite": "Cross-platform TFLite inference target.",
  "Raspberry Pi": "Linux SBC, ARM-based deployment.",
  Arduino: "Arduino-compatible microcontroller.",
  Linux: "Linux desktop / server build.",
  Windows: "Windows desktop build.",
  "Custom Device": "Custom deployment target.",
};

function isDeploymentActive(status?: string | null) {
  return status === "pending" || status === "running";
}

// Shape returned by GET /deployment/compatibility (Target Device Phase 4).
// `message` is the user-facing reason; `reason_code` is the machine-readable
// twin the later phases (format filtering, remediation) will branch on.
type CompatibilityResult = {
  compatible: boolean;
  reason_code: string;
  message: string;
  model_type: string;
  build_format: string;
  device_slug: string | null;
};

// GET /deployment/targets?model_id=…&device_profile=… (Target Device Phase 6)
// — the single package PetalEdge offers for this model × device, or none.
// `offered` is null for a target with no shipped package (arduino/esp32/cpp)
// or a model the offered package can't run; `unavailable` lists every format
// that isn't the offered one, each with its own reason.
type FormatOffer = {
  device_profile: string | null;
  deploy_target: string | null;
  offered: {
    package: "pe" | "pxe";
    target: string;
    deployment_format: "pe" | "pxe";
    build_format: string;
  } | null;
  unavailable: {
    package: string;
    build_format: string;
    available: false;
    reason_code: string;
    message: string;
  }[];
  reason_code: string;
  message: string;
};

// GET /deployment/estimate (Target Device Phase 5) returns raw bytes/ms —
// formatting is the UI's job, not the backend's.
function formatEstimateBytes(n: number | null | undefined): string {
  if (n === null || n === undefined) return "N/A";
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
  return `${(n / (1024 * 1024)).toFixed(1)} MB`;
}

function formatEstimateMs(n: number | null | undefined): string {
  if (n === null || n === undefined) return "N/A";
  return `${n < 10 ? n.toFixed(2) : n.toFixed(1)} ms`;
}

// One flash/RAM/latency figure from a MetricEstimate dict ({value, unit,
// source, budget, budget_source, margin, fits, reason_code, note,
// confidence}). Never a bare number — the note (and, for an estimated
// figure, the confidence caveat) always renders alongside it. An estimate
// that blows its budget is a warning here, never a disable — Phase 4's
// compatibility gate (canBuild/disabledReason) is the only thing that blocks
// a build.
function EstimateMetric({
  icon,
  label,
  metric,
  format,
}: {
  icon: ReactNode;
  label: string;
  metric: any;
  format: (n: number | null | undefined) => string;
}) {
  if (!metric) return null;
  const budgetSuffix = metric.budget != null ? ` / ${format(metric.budget)} budget` : "";
  return (
    <div className="pe-dep2-estimate-tile flex items-start gap-2 rounded-lg border border-[var(--app-border)] bg-[var(--app-surface-2)] p-3">
      <div className="mt-0.5 flex-shrink-0" style={{ color: "var(--app-text-soft)" }} aria-hidden="true">
        {icon}
      </div>
      <div className="min-w-0">
        <p className="text-[10px] uppercase tracking-wider" style={{ color: "var(--app-text-soft)" }}>
          {label}
        </p>
        <p className="text-sm font-semibold" style={{ color: "var(--app-text)" }}>
          {format(metric.value)}
          {budgetSuffix}
          {metric.fits === false && (
            <span className="ml-2 inline-flex items-center gap-1 text-[10px] font-semibold text-amber-500">
              <AlertTriangle size={10} /> over budget
            </span>
          )}
        </p>
        <p className="mt-0.5 text-[11px] leading-snug" style={{ color: "var(--app-text-soft)" }}>
          {metric.note}
          {metric.confidence ? ` ${metric.confidence}` : ""}
        </p>
      </div>
    </div>
  );
}

// Phase 7 optimization advice (Recommendation dicts: {constraint, severity,
// code, title, detail, expected_gain}) shown under the estimate metrics
// above. Visibly secondary — advice, not a gate: the build button's
// canBuild/disabledReason never depend on this. `reason` is a real answer
// ("fits comfortably" / "cannot be assessed" / "incompatible") whenever
// `items` is empty, so the block never renders blank.
function Recommendations({ items, reason }: { items: any[] | undefined; reason: string | null | undefined }) {
  if (!items || items.length === 0) {
    if (!reason) return null;
    return (
      <p className="pe-dep2-recommendations-empty mb-4 text-[12px] leading-snug" style={{ color: "var(--app-text-soft)" }}>
        {reason}
      </p>
    );
  }
  return (
    <div className="pe-dep2-recommendations mb-4 space-y-2">
      {items.map((rec, i) => (
        <div
          key={`${rec.code}-${rec.constraint}-${i}`}
          className="flex items-start gap-2 rounded-lg border border-dashed border-[var(--app-border)] bg-[var(--app-surface-2)] p-3"
        >
          <Lightbulb size={14} className="mt-0.5 flex-shrink-0" style={{ color: "var(--app-text-soft)" }} aria-hidden="true" />
          <div className="min-w-0">
            <p className="text-sm font-semibold" style={{ color: "var(--app-text)" }}>
              {rec.title}
              <span
                className="ml-2 rounded-full px-1.5 py-0.5 text-[10px] font-semibold uppercase tracking-wide"
                style={{
                  color: rec.severity === "over_budget" ? "#f59e0b" : "var(--app-text-soft)",
                  background: rec.severity === "over_budget" ? "rgba(245, 158, 11, 0.12)" : "var(--app-surface)",
                }}
              >
                {rec.constraint}
              </span>
            </p>
            <p className="mt-0.5 text-[11px] leading-snug" style={{ color: "var(--app-text-soft)" }}>
              {rec.detail} Expected gain: {rec.expected_gain}.
            </p>
          </div>
        </div>
      ))}
    </div>
  );
}

/**
 * Deployment — shared content, used by both ObjectDetectionDeployment and
 * MotionDeployment. Nothing here is modality-specific: the model/build path
 * dispatches on the trained model + device profile, never on project_type,
 * and the three device profiles (generic_tflite, raspberry_pi_4, unoq) match
 * §6.6's motion target list exactly (ESP32 has no profile yet — Phase 8,
 * see motion_phase0.md §11 T5). A motion project's impulse has no trained
 * model until motion training lands (Phase 6+), so this naturally renders
 * the existing "not fully trained" state for it today — an honest shell,
 * not a fork.
 */
export default function DeploymentContent() {
  const { activeProject, activeImpulse: storeActiveImpulse } = useAppStore();

  const [model, setModel] = useState<any>(null);
  const [modelLoading, setModelLoading] = useState(false);
  const [modelError, setModelError] = useState<string | null>(null);

  const [modelReady, setModelReady] = useState<boolean | null>(null);

  const [selectedDevice, setSelectedDevice] = useState<string>("generic_tflite");
  const [deployments, setDeployments] = useState<any[]>([]);
  const [building, setBuilding] = useState(false);
  const [activeDeployment, setActiveDeployment] = useState<any>(null);
  const [cancelling, setCancelling] = useState(false);
  const [refreshing, setRefreshing] = useState(false);
  const [compat, setCompat] = useState<CompatibilityResult | null>(null);
  const [estimateData, setEstimateData] = useState<any>(null);

  // A project with a target device resolves its deploy target from the
  // catalog and never asks the user to pick one here (Target Device Phase 6
  // §4). Without one, the legacy three-profile picker below is unchanged
  // (§5) and `selectedDevice` is what drives it.
  const hasTargetDevice = Boolean(activeProject?.target_device_slug);
  const targetDeviceInfo = activeProject?.target_device ?? null;
  const deviceProfile = hasTargetDevice ? (activeProject!.target_device_slug as string) : selectedDevice;

  // The single package PetalEdge offers for `deviceProfile` (Target Device
  // Phase 6). `formatOfferFailed` is distinct from "still null" — a failed
  // fetch must fall back to the pre-Phase-6 unfiltered behaviour rather than
  // disabling the page, same discipline the compatibility/estimate fetches
  // below already follow.
  const [formatOffer, setFormatOffer] = useState<FormatOffer | null>(null);
  const [formatOfferFailed, setFormatOfferFailed] = useState(false);

  useEffect(() => {
    setFormatOffer(null);
    setFormatOfferFailed(false);

    if (!model?.id || !deviceProfile) return;

    let cancelled = false;
    (async () => {
      try {
        const { data } = await deploymentApi.targets({ model_id: model.id, device_profile: deviceProfile });
        if (!cancelled) setFormatOffer(data);
      } catch (e: any) {
        console.error("Format offer check failed:", e);
        if (!cancelled) setFormatOfferFailed(true);
      }
    })();

    return () => {
      cancelled = true;
    };
  }, [model?.id, deviceProfile]);

  const offered = formatOffer?.offered ?? null;
  // Loaded-and-confirmed-unavailable is a real block; not-yet-loaded (or a
  // failed fetch) is not — POST /build's own gate is the authority either way.
  const packagePending = !formatOffer && !formatOfferFailed;
  const packageBlocked = formatOffer !== null && !offered;
  const buildFormat = offered?.build_format;

  const refreshDeploymentsForModel = useCallback(async (modelId: string, preferredId?: string | null) => {
    const { data: deps } = await deploymentApi.listForModel(modelId);
    setDeployments(deps);

    const preferred = preferredId
      ? deps.find((d: any) => d.id === preferredId && isDeploymentActive(d.status))
      : null;
    const running = preferred || deps.find((d: any) => isDeploymentActive(d.status)) || null;

    setActiveDeployment(running);
    setBuilding(Boolean(running));
    return { deps, running };
  }, []);

  const loadModelForImpulse = useCallback(async (impulse: any) => {
    setModel(null);
    setModelError(null);
    setDeployments([]);
    setActiveDeployment(null);
    setBuilding(false);
    setModelLoading(true);
    setModelReady(null);

    try {
      const { data } = await trainedModelsApi.latestForImpulse(impulse.id, "tflite");
      setModelReady(true);
      setModel(data);
      await refreshDeploymentsForModel(data.id);
    } catch (e: any) {
      console.error("Failed to load model:", e);
      setModelReady(false);
    } finally {
      setModelLoading(false);
    }
  }, [refreshDeploymentsForModel]);

  useEffect(() => {
    if (!activeProject || !storeActiveImpulse) return;
    loadModelForImpulse(storeActiveImpulse);
  }, [activeProject?.id, storeActiveImpulse?.id, loadModelForImpulse]);

  // Ask the backend whether this model can be built for the offered package
  // before anything is started. Cleared on every selection change so a stale
  // rejection never blocks a combination it was not about. Runs against the
  // package the formats check actually offered — there is nothing else to
  // check once that call has resolved.
  useEffect(() => {
    setCompat(null);

    if (!model?.id || !deviceProfile || !buildFormat) return;

    let cancelled = false;
    (async () => {
      try {
        const { data } = await deploymentApi.compatibility({
          model_id: model.id,
          format: buildFormat,
          device_profile: deviceProfile,
        });
        if (!cancelled) setCompat(data);
      } catch (e: any) {
        // A failed check must not block a build the backend would accept —
        // POST /build runs the same gate and is the authority.
        console.error("Compatibility check failed:", e);
      }
    })();

    return () => {
      cancelled = true;
    };
  }, [model?.id, deviceProfile, buildFormat]);

  // Flash / RAM / latency estimate for the offered package (Target Device
  // Phase 5). Read-only, like the compatibility check above — never blocks
  // the build button, just informs it.
  useEffect(() => {
    setEstimateData(null);

    if (!model?.id || !deviceProfile || !buildFormat) return;

    let cancelled = false;
    (async () => {
      try {
        const { data } = await deploymentApi.estimate({
          model_id: model.id,
          format: buildFormat,
          device_profile: deviceProfile,
        });
        if (!cancelled) setEstimateData(data);
      } catch (e: any) {
        console.error("Estimate check failed:", e);
      }
    })();

    return () => {
      cancelled = true;
    };
  }, [model?.id, deviceProfile, buildFormat]);

  useEffect(() => {
    if (!activeDeployment?.id || !building || !model?.id) return;

    const intervalId = setInterval(async () => {
      try {
        const { data } = await deploymentApi.get(activeDeployment.id);
        setActiveDeployment(data);
        setDeployments((current) => current.map((d) => (d.id === data.id ? data : d)));

        if (data.status === "completed" || data.status === "failed" || data.status === "cancelled") {
          clearInterval(intervalId);
          setBuilding(false);
          setActiveDeployment(null);
          await refreshDeploymentsForModel(model.id);

          if (data.status === "completed") {
            toast.success("Build complete, ready to download.");
          } else if (data.status === "cancelled") {
            toast("Build cancelled.");
          } else {
            toast.error(`Build failed: ${data.error_message || "unknown"}`);
          }
        }
      } catch (e: any) {
        console.error("Polling error:", e);
        clearInterval(intervalId);
        try {
          await refreshDeploymentsForModel(model.id, activeDeployment.id);
        } catch (refreshError) {
          console.error("Deployment refresh error:", refreshError);
          setActiveDeployment(null);
          setBuilding(false);
        }
        toast.error("Lost deployment status updates. Refreshed build history.");
      }
    }, POLL_MS);

    return () => clearInterval(intervalId);
  }, [activeDeployment?.id, building, model?.id, refreshDeploymentsForModel]);

  async function buildDeployment() {
    if (!model || !deviceProfile) return;

    setBuilding(true);
    try {
      // The offered package's own {target, deployment_format} pair — read
      // from the formats check, never re-derived locally (Target Device
      // Phase 6). A failed formats check falls back to the pre-Phase-6 guess;
      // POST /build's own compatibility gate is the real authority either way.
      const target = offered?.target
        ?? (hasTargetDevice ? targetDeviceInfo?.deploy_target : undefined)
        ?? SUPPORTED_DEVICE_PROFILES.find((p) => p.value === deviceProfile)?.target
        ?? "tflite";
      const deployment_format = offered?.deployment_format ?? "pe";

      const { data } = await deploymentApi.build({
        model_id: model.id,
        target,
        deployment_target: target,
        device_profile: deviceProfile,
        deployment_format,
        options: { device_profile: deviceProfile },
      });

      setActiveDeployment(data);
      setDeployments((current) => [data, ...current]);
      toast.success("Build started");
    } catch (e: any) {
      setBuilding(false);
      const detail = e.response?.data?.detail || e.message || "Build failed to start";
      toast.error(detail);
    }
  }

  async function cancelDeployment() {
    if (!activeDeployment?.id) return;

    setCancelling(true);
    try {
      const { data } = await deploymentApi.cancel(activeDeployment.id);
      setActiveDeployment(null);
      setBuilding(false);
      setDeployments((current) => current.map((d) => (d.id === data.id ? data : d)));
      await refreshDeploymentsForModel(model.id);
      toast.success("Build cancelled");
    } catch (e: any) {
      const detail = e.response?.data?.detail || e.message || "Failed to cancel build";
      toast.error(detail);
      await refreshDeploymentsForModel(model.id, activeDeployment.id);
    } finally {
      setCancelling(false);
    }
  }

  const latestCompleted = deployments.find((d) => d.status === "completed");
  const selectedProfile = SUPPORTED_DEVICE_PROFILES.find((p) => p.value === selectedDevice);

  // ───────── derive presentation data from existing sources only ─────────
  const deviceOptions: DeviceOption[] = useMemo(
    () =>
      SUPPORTED_DEVICE_PROFILES.map((p) => ({
        value: p.value,
        label: p.label,
        category: p.category,
        description: CATEGORY_DESCRIPTIONS[p.category] ?? "Deployment target.",
        Icon: p.Icon,
      })),
    [],
  );

  const deviceByProfile = useMemo(() => {
    const m = new Map<string, DeviceOption>();
    for (const o of deviceOptions) m.set(o.value, o);
    return m;
  }, [deviceOptions]);

  // A synthetic DeviceOption for the resolved target device, so the existing
  // DevicePreview component can render it with no changes of its own.
  const resolvedDeviceOption: DeviceOption | undefined = targetDeviceInfo
    ? {
        value: targetDeviceInfo.slug,
        label: targetDeviceInfo.display_name,
        category: targetDeviceInfo.family || "Target Device",
        description: targetDeviceInfo.accelerator_note || "Resolved from this project's target device.",
        Icon: targetDeviceInfo.deploy_target === "tflite" ? Cpu : CircuitBoard,
      }
    : undefined;

  const packageLabel = offered ? (offered.package === "pxe" ? ".pxe package" : ".pe package") : null;
  const noPackageReason = formatOffer && !offered ? formatOffer.message : null;

  const deviceLabel = hasTargetDevice ? (targetDeviceInfo?.display_name ?? deviceProfile) : selectedProfile?.label;
  const devicePlatform = hasTargetDevice ? (targetDeviceInfo?.deploy_target ?? undefined) : selectedProfile?.target;
  const selectedDeviceOptionForPreview = hasTargetDevice ? resolvedDeviceOption : deviceByProfile.get(selectedDevice);

  // Read-only "supported deployment targets" checklist for the Target Device
  // panel, restored to the pre-Phase-6 device-list look. Only ever lists
  // boards with a shipped package (never arduino/esp32/cpp — no `.pe`
  // fallback for those), and only the family the resolved device belongs to:
  // Generic TFLite stands alone (`.pe`); Raspberry Pi and UNO Q are shown
  // together as the `.pxe` family, with the resolved one listed first and
  // checked.
  const targetDeviceChecklist = useMemo(() => {
    if (!hasTargetDevice) return [];
    const byValue = (v: string) => SUPPORTED_DEVICE_PROFILES.find((p) => p.value === v)!;
    if (devicePlatform === "tflite") return [byValue("generic_tflite")];
    if (devicePlatform === "raspberry_pi" || devicePlatform === "unoq") {
      const entries = [byValue("unoq"), byValue("raspberry_pi_4")];
      return entries.sort((a, b) =>
        (a.target === devicePlatform ? 0 : 1) - (b.target === devicePlatform ? 0 : 1),
      );
    }
    return [];
  }, [hasTargetDevice, devicePlatform]);

  if (!storeActiveImpulse) {
    return (
      <ImpulseNotReady
        description="No impulse to deploy yet. Create an impulse and train a model before you can build a deployment."
        tip="Head to Impulse Design to create one, then train your model to produce a deployable artifact."
      />
    );
  }

  // Until modelReady resolves, we can't tell whether to render the main UI or
  // the "Almost there!" warning. Render a loading state first so the warning is
  // the first non-loading frame for untrained impulses.
  const isInitialLoad = !!activeProject && (modelLoading || modelReady === null);
  if (isInitialLoad) {
    return (
      <div className="flex h-[calc(100vh-theme('spacing.16'))] items-center justify-center pt-2">
        <span className="inline-flex items-center gap-2 text-sm text-[var(--app-text-soft)]">
          <RefreshCw size={14} className="animate-spin" />
          Verifying model status...
        </span>
      </div>
    );
  }

  if (activeProject && modelReady === false) {
    return (
      <ImpulseNotReady
        description="Your impulse is not fully trained. Use the items in the navigation bar to configure and train your model before you can deploy your model."
        tip="Train your model and produce a valid artifact before building a deployment package."
      />
    );
  }

  const ready = modelReady === true;
  const incompatible = compat !== null && !compat.compatible;
  const canBuild =
    ready && Boolean(model) && Boolean(deviceProfile) && !packagePending && !packageBlocked && !incompatible;
  const disabledReason = !ready
    ? "Train a valid model first"
    : !model
      ? "Model is still loading"
      : !deviceProfile
        ? "Choose a target device"
        : packagePending
          ? "Checking available package…"
          : packageBlocked
            ? noPackageReason || "No deployment package is available for this device"
            : incompatible
              ? compat!.message
              : undefined;

  return (
    <div className="pe-dep2-page mx-auto max-w-[88rem] space-y-5">
      <DeploymentHero ready={ready} />

      {modelLoading && (
        <div className="pe-dep2-surface p-8 text-center">
          <p className="text-sm" style={{ color: "var(--app-text-muted)" }}>
            Loading model info…
          </p>
        </div>
      )}

      {modelError && (
        <div className="pe-dep2-surface p-12 text-center">
          <Rocket size={32} className="mx-auto mb-3" style={{ color: "var(--app-text-soft)" }} aria-hidden="true" />
          <p style={{ color: "var(--app-text-muted)" }}>{modelError}</p>
        </div>
      )}

      {model && (
        <>
          <ModelOverview model={model} ready={ready} />

          <div className="pe-dep2-workspace">
            <section className="pe-dep2-surface pe-dep2-panel" aria-labelledby="pe-dep2-selector-title">
              <div className="pe-dep2-panel-head">
                <div className="min-w-0">
                  <h2 id="pe-dep2-selector-title" className="pe-dep2-panel-title">
                    Target Device
                  </h2>
                  <p className="pe-dep2-panel-sub">
                    {hasTargetDevice
                      ? "Resolved from this project's target device settings."
                      : "Pick a deployment target. Categories group similar runtimes."}
                  </p>
                </div>
              </div>
              {hasTargetDevice ? (
                <div className="pe-dep2-preview">
                  {targetDeviceChecklist.length > 0 ? (
                    <ul className="space-y-2" role="list" aria-label="Supported deployment targets">
                      {targetDeviceChecklist.map((p) => {
                        const isSelected = p.target === devicePlatform;
                        const pkg = p.target === "tflite" ? ".pe" : ".pxe";
                        const Icon = p.Icon;
                        return (
                          <li
                            key={p.value}
                            className="flex items-center gap-2 rounded-lg border p-2.5"
                            style={{
                              borderColor: isSelected ? "#8b5cf6" : "var(--app-border)",
                              background: isSelected
                                ? "linear-gradient(135deg, rgba(139, 92, 246, 0.16) 0%, rgba(56, 189, 248, 0.1) 100%)"
                                : "transparent",
                            }}
                          >
                            <span
                              className="flex h-5 w-5 flex-shrink-0 items-center justify-center text-sm font-bold"
                              style={{ color: isSelected ? "#8b5cf6" : "var(--app-text-soft)" }}
                              aria-hidden="true"
                            >
                              {isSelected ? "✔" : <Icon size={14} />}
                            </span>
                            <span
                              className="text-sm"
                              style={{ color: isSelected ? "var(--app-text)" : "var(--app-text-soft)", fontWeight: isSelected ? 600 : 400 }}
                            >
                              {p.label} ({pkg})
                            </span>
                          </li>
                        );
                      })}
                    </ul>
                  ) : (
                    <p className="text-sm font-medium" style={{ color: "var(--app-text)" }}>
                      Deployment package not implemented yet
                    </p>
                  )}
                  <p className="mt-3 text-[13px]" style={{ color: "var(--app-text-soft)" }}>
                    Set in this project's target device settings.
                  </p>
                </div>
              ) : (
                <DeviceSelector
                  options={deviceOptions}
                  value={selectedDevice}
                  onChange={setSelectedDevice}
                  categoryOrder={CATEGORY_ORDER}
                />
              )}
            </section>

            <section className="pe-dep2-surface pe-dep2-panel" aria-labelledby="pe-dep2-preview-title">
              <div className="pe-dep2-panel-head">
                <div className="min-w-0">
                  <h2 id="pe-dep2-preview-title" className="pe-dep2-panel-title">
                    Device Preview
                  </h2>
                  <p className="pe-dep2-panel-sub">Build details for the selected target.</p>
                </div>
              </div>
              <DevicePreview
                device={selectedDeviceOptionForPreview}
                platform={devicePlatform}
                packageType={packageLabel ?? (packageBlocked ? "Not available" : undefined)}
                impulse={storeActiveImpulse?.name}
              />
            </section>
          </div>

          <section className="pe-dep2-surface pe-dep2-panel" aria-labelledby="pe-dep2-actions-title">
            <div className="pe-dep2-panel-head">
              <div className="min-w-0">
                <h2 id="pe-dep2-actions-title" className="pe-dep2-panel-title">
                  Deployment
                </h2>
                <p className="pe-dep2-panel-sub">
                  Generate and download a package for {deviceLabel ?? "your device"}.
                </p>
              </div>
            </div>

            {estimateData && (
              <div className="pe-dep2-estimate mb-4 grid grid-cols-1 gap-3 sm:grid-cols-3">
                <EstimateMetric icon={<HardDrive size={14} />} label="Flash" metric={estimateData.flash_usage} format={formatEstimateBytes} />
                <EstimateMetric icon={<Cpu size={14} />} label="Peak RAM" metric={estimateData.ram_usage} format={formatEstimateBytes} />
                <EstimateMetric icon={<Clock size={14} />} label="Latency" metric={estimateData.inferencing_time} format={formatEstimateMs} />
              </div>
            )}

            {estimateData && (
              <Recommendations items={estimateData.recommendations} reason={estimateData.recommendations_reason} />
            )}

            <DeploymentActions
              deviceLabel={deviceLabel}
              building={building}
              canBuild={canBuild}
              disabledReason={disabledReason}
              onBuild={buildDeployment}
              refreshing={refreshing}
              onRefresh={
                model?.id
                  ? async () => {
                    if (refreshing) return;
                    setRefreshing(true);
                    try {
                      const { running } = await refreshDeploymentsForModel(
                        model.id,
                        activeDeployment?.id || null,
                      );
                      toast.success(
                        running ? "Refreshed — build still in progress" : "Status refreshed",
                      );
                    } catch (err: any) {
                      const detail =
                        err?.response?.data?.detail || err?.message || "Failed to refresh status";
                      toast.error(detail);
                    } finally {
                      setRefreshing(false);
                    }
                  }
                  : undefined
              }
              cancelling={cancelling}
              canCancel={Boolean(building && activeDeployment?.id)}
              onCancel={cancelDeployment}
              downloadUrl={latestCompleted?.download_url ?? null}
              downloadFilename={latestCompleted?.download_filename ?? null}
            />

            {activeDeployment && building && (
              <div className="pe-dep2-progress mt-4" role="status" aria-live="polite">
                <RefreshCw size={16} className="animate-spin shrink-0" style={{ color: "#c4b5fd" }} aria-hidden="true" />
                <div className="flex-1 min-w-0">
                  <div className="text-sm font-medium" style={{ color: "var(--app-text)" }}>
                    Building{" "}
                    {SUPPORTED_DEVICE_PROFILES.find((p) => p.value === activeDeployment.device_profile)?.label ||
                      targetDeviceInfo?.display_name ||
                      activeDeployment.target}{" "}
                    package…
                  </div>
                  <div className="pe-dep2-progress-bar mt-2">
                    <div />
                  </div>
                </div>
              </div>
            )}
          </section>

          <section className="pe-dep2-surface pe-dep2-panel" aria-labelledby="pe-dep2-timeline-title">
            <div className="pe-dep2-panel-head">
              <div className="min-w-0">
                <h2 id="pe-dep2-timeline-title" className="pe-dep2-panel-title">
                  Build Timeline
                </h2>
                <p className="pe-dep2-panel-sub">Recent deployment builds for this trained model.</p>
              </div>
            </div>
            <BuildTimeline deployments={deployments} deviceByProfile={deviceByProfile} />
          </section>
        </>
      )}
    </div>
  );
}
