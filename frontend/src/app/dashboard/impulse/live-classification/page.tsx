"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Activity, Cpu, FlaskConical, Loader2, Smartphone } from "lucide-react";
import ImpulseNotReady from "@/components/dashboard/ImpulseNotReady";
import MotionPhasePending from "@/components/dashboard/MotionPhasePending";
import toast from "react-hot-toast";
import { useStudioWS } from "@/hooks/useStudioWS";
import { useAppStore } from "@/store/appStore";
import {
  devicesApi,
  impulsesApi,
  liveClassificationApi,
} from "@/utils/api";
import { useTrainingValidity } from "@/hooks/useTrainingValidity";
import {
  applyLiveStudioEventToDevices,
  buildClassifySampleRequest,
  canStartLiveSampling,
  deriveSensorOptions,
  deriveCanLoadSample,
  deriveStartSamplingDisabledReason,
  getSamplePreviewUrl,
  mapPreferredRuntime,
  mapRawSampleToOption,
  mapStudioInferenceResult,
  resolveLiveResultSource,
} from "@/lib/live-classification-helpers";
import LiveClassificationResultPanel from "@/components/dashboard/live-classification/LiveClassificationResultPanel";
import LiveClassificationSummaryCard from "@/components/dashboard/live-classification/LiveClassificationSummaryCard";
import type {
  LiveClassificationCompatibleDeployment,
  LiveClassificationDeviceOption,
  LiveClassificationRunResult,
  LiveClassificationRuntimeInfo,
  LiveClassificationSensorOption,
  LiveClassificationSampleOption,
  LiveClassificationSamplingState,
  LiveClassificationStreamingState,
  LiveClassificationViewState,
} from "@/types/live-classification";
import type { StudioEvent } from "@/types/devices";

const SENSOR_OPTIONS = [
  { id: "accelerometer", name: "Built-in accelerometer", frequencies: [12.5, 25, 50, 62.5, 100, 200, 400] },
  { id: "microphone", name: "Built-in microphone", frequencies: [8000, 16000, 22050, 44100] },
  { id: "camera", name: "Camera", frequencies: [1, 5, 10, 15, 30] },
  { id: "environmental", name: "Environmental (temp / humidity)", frequencies: [1, 2, 5, 10] },
] satisfies LiveClassificationSensorOption[];

type ResultSource = "sample" | "live" | null;

function mapDeviceOption(raw: any): LiveClassificationDeviceOption {
  return {
    id: raw.id,
    device_id: raw.device_id,
    name: raw.name,
    device_type: raw.device_type ?? "",
    is_online: !!raw.is_online,
    is_connected: raw.is_connected ?? false,
    supports_snapshot_streaming: raw.supports_snapshot_streaming ?? false,
    deployment_target: raw.deployment_target ?? null,
    device_profile: raw.device_profile ?? null,
    installed_deployment_id: raw.installed_deployment_id ?? null,
    installed_model_version: raw.installed_model_version ?? null,
    sensors: Array.isArray(raw.sensors) ? raw.sensors : [],
    device_metadata: raw.device_metadata ?? {},
    mode: raw.mode,
  };
}

export default function LiveClassificationPage() {
  const { activeProject, activeImpulse } = useAppStore();
  const {
    hasValidTrainingOutput,
    loading: trainingValidityLoading,
  } = useTrainingValidity(activeImpulse?.id ?? null);
  const keepaliveRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const activeStreamDeviceIdRef = useRef<string | null>(null);
  const activeRuntimeIdRef = useRef<string | null>(null);

  const [viewState, setViewState] = useState<LiveClassificationViewState>("loading");
  const [impulseId, setImpulseId] = useState<string | null>(null);
  const [modelLoading, setModelLoading] = useState(true);
  const [modelError, setModelError] = useState<string | null>(null);
  const [runtime, setRuntime] = useState<LiveClassificationRuntimeInfo | null>(null);

  const [devicesLoading, setDevicesLoading] = useState(false);
  const [devices, setDevices] = useState<LiveClassificationDeviceOption[]>([]);
  const [selectedDeviceId, setSelectedDeviceId] = useState("");
  const [selectedSensorId, setSelectedSensorId] = useState("");
  const [sampleLengthMs, setSampleLengthMs] = useState(5000);
  const [selectedFrequency, setSelectedFrequency] = useState("");
  const [compatibilityLoading, setCompatibilityLoading] = useState(false);
  const [compatibility, setCompatibility] = useState<LiveClassificationCompatibleDeployment | null>(null);

  const [testSamplesLoading, setTestSamplesLoading] = useState(false);
  const [testSamples, setTestSamples] = useState<LiveClassificationSampleOption[]>([]);
  const [selectedSampleId, setSelectedSampleId] = useState("");
  const [selectedSampleMeta, setSelectedSampleMeta] = useState<LiveClassificationSampleOption | null>(null);

  const [loadingResult, setLoadingResult] = useState(false);
  const [result, setResult] = useState<LiveClassificationRunResult | null>(null);
  const [resultError, setResultError] = useState<string | null>(null);
  const [sampleImageUrl, setSampleImageUrl] = useState<string | null>(null);

  const [streamingState, setStreamingState] = useState<LiveClassificationStreamingState>("idle");
  const [samplingState, setSamplingState] = useState<LiveClassificationSamplingState>("idle");
  const [liveResult, setLiveResult] = useState<LiveClassificationRunResult | null>(null);
  const [liveResultError, setLiveResultError] = useState<string | null>(null);
  const [resultSource, setResultSource] = useState<ResultSource>(null);

  const clearKeepalive = useCallback(() => {
    if (keepaliveRef.current) {
      clearInterval(keepaliveRef.current);
      keepaliveRef.current = null;
    }
  }, []);

  const handleStudioEvent = useCallback((event: StudioEvent) => {
    setDevices((prev) => applyLiveStudioEventToDevices(prev, event));

    if (!selectedDeviceId) return;
    const currentDevice = devices.find((device) => device.id === selectedDeviceId);
    const selectedHardwareId = currentDevice?.device_id;
    if (!selectedHardwareId || event.device_id !== selectedHardwareId) {
      return;
    }

    if (event.type === "sample.started") {
      setSamplingState("active");
    } else if (event.type === "sample.stopped") {
      setSamplingState("idle");
    } else if (event.type === "sample.failed") {
      setSamplingState("error");
      setLiveResultError(String(event.payload?.error ?? "Sampling failed."));
    } else if (event.type === "inference.started") {
      setStreamingState("active");
    } else if (event.type === "inference.stopped") {
      clearKeepalive();
      activeStreamDeviceIdRef.current = null;
      setStreamingState("idle");
      setSamplingState("idle");
    } else if (event.type === "stream.failed") {
      clearKeepalive();
      activeStreamDeviceIdRef.current = null;
      setStreamingState("error");
      setSamplingState("error");
      setLiveResultError(String(event.payload?.error ?? "Live classification stream failed."));
    } else if (event.type === "device.disconnected") {
      clearKeepalive();
      activeStreamDeviceIdRef.current = null;
      setStreamingState("idle");
      setSamplingState("idle");
    } else if (event.type === "device.mode_changed") {
      const mode = event.payload?.mode;
      if (mode === "inference") {
        setStreamingState("active");
      } else if (mode === "sampling") {
        setSamplingState("active");
      } else if (mode === "idle") {
        if (streamingState !== "active") setSamplingState("idle");
      }
    } else if (event.type === "inference.result") {
      const mapped = mapStudioInferenceResult(
        event.payload ?? {},
        runtime,
        currentDevice?.name ?? "Device",
        selectedDeviceId,
      );
      setLiveResult(mapped);
      setLiveResultError(null);
      setResultSource("live");
    }
  }, [clearKeepalive, devices, runtime, selectedDeviceId, streamingState]);

  const { connected: wsConnected } = useStudioWS(activeProject?.id ?? null, handleStudioEvent);

  useEffect(() => {
    if (!activeProject) {
      setImpulseId(null);
      setRuntime(null);
      setModelError(null);
      setDevices([]);
      setSelectedDeviceId("");
      setSelectedSensorId("");
      setSelectedFrequency("");
      setCompatibility(null);
      setTestSamples([]);
      setSelectedSampleId("");
      setSelectedSampleMeta(null);
      setViewState("no_project");
      setModelLoading(false);
      clearKeepalive();
      activeStreamDeviceIdRef.current = null;
      activeRuntimeIdRef.current = null;
      return;
    }

    setDevicesLoading(true);
    devicesApi
      .list(activeProject.id)
      .then((response) => {
        setDevices((response.data ?? []).map(mapDeviceOption));
      })
      .catch(() => {
        setDevices([]);
      })
      .finally(() => setDevicesLoading(false));

    const bootstrap = async () => {
      setImpulseId(null);
      setRuntime(null);
      setModelError(null);
      setViewState("loading");
      setModelLoading(true);
      setTestSamplesLoading(true);
      try {
        let impulse: any =
          activeImpulse?.project_id === activeProject.id ? activeImpulse : null;
        if (!impulse) {
          const { data } = await impulsesApi.list(activeProject.id);
          impulse = (data ?? [])[0] ?? null;
        }
        if (!impulse) {
          setViewState("no_impulse");
          return;
        }
        setImpulseId(impulse.id);

        // Gate on the single source of truth — never display an older completed
        // run's runtime artifact while the latest run is cancelled/failed/running.
        // Wait for validity to resolve before deciding.
        if (trainingValidityLoading) {
          setViewState("loading");
          return;
        }
        if (!hasValidTrainingOutput) {
          setRuntime(null);
          setTestSamples([]);
          setViewState("no_model");
          return;
        }

        const [runtimeResult, samplesResult] = await Promise.allSettled([
          liveClassificationApi.preferredRuntime(impulse.id),
          liveClassificationApi.sampleOptions(impulse.id),
        ]);

        if (runtimeResult.status === "fulfilled") {
          setRuntime(mapPreferredRuntime(runtimeResult.value.data));
          setViewState("ready");
        } else {
          setModelError("No trained model found. Train the model first.");
          setViewState("no_model");
        }

        if (samplesResult.status === "fulfilled") {
          setTestSamples((samplesResult.value.data ?? []).map(mapRawSampleToOption));
        } else {
          setTestSamples([]);
        }
      } catch {
        setViewState("no_impulse");
      } finally {
        setModelLoading(false);
        setTestSamplesLoading(false);
      }
    };

    bootstrap();

    return () => {
      clearKeepalive();
      activeStreamDeviceIdRef.current = null;
      activeRuntimeIdRef.current = null;
    };
  }, [activeProject?.id, activeImpulse?.id, hasValidTrainingOutput, trainingValidityLoading, clearKeepalive]);

  const selectedDevice = useMemo(
    () => devices.find((device) => device.id === selectedDeviceId) ?? null,
    [devices, selectedDeviceId],
  );
  const sensorConfig = useMemo(
    () => deriveSensorOptions(selectedDevice, SENSOR_OPTIONS),
    [selectedDevice],
  );
  const availableSensorOptions = sensorConfig.options;
  const selectedSensorOption = useMemo(
    () => availableSensorOptions.find((sensor) => sensor.id === selectedSensorId) ?? null,
    [availableSensorOptions, selectedSensorId],
  );
  const availableFrequencies = useMemo(
    () =>
      (selectedSensorOption?.frequencies?.length
        ? selectedSensorOption.frequencies
        : SENSOR_OPTIONS.find((sensor) => sensor.id === selectedSensorId)?.frequencies ?? [])
        .map((frequency) => String(frequency)),
    [selectedSensorId, selectedSensorOption],
  );

  useEffect(() => {
    if (!selectedDeviceId) {
      setCompatibility(null);
      setCompatibilityLoading(false);
      return;
    }

    setCompatibilityLoading(true);
    devicesApi
      .compatibleDeployment(selectedDeviceId, impulseId ?? undefined)
      .then((response) => {
        setCompatibility(response.data ?? null);
      })
      .catch(() => {
        setCompatibility(null);
      })
      .finally(() => setCompatibilityLoading(false));
  }, [impulseId, selectedDeviceId, runtime?.id]);

  useEffect(() => {
    if (!selectedDeviceId) return;

    devicesApi
      .get(selectedDeviceId)
      .then((response) => {
        const detail = mapDeviceOption(response.data);
        setDevices((prev) => prev.map((device) => (device.id === detail.id ? { ...device, ...detail } : device)));
      })
      .catch(() => { });
  }, [selectedDeviceId]);

  useEffect(() => {
    if (selectedSensorId && !availableSensorOptions.some((sensor) => sensor.id === selectedSensorId)) {
      setSelectedSensorId("");
      setSelectedFrequency("");
    }
  }, [availableSensorOptions, selectedSensorId]);

  useEffect(() => {
    if (selectedFrequency && !availableFrequencies.includes(selectedFrequency)) {
      setSelectedFrequency("");
    }
  }, [availableFrequencies, selectedFrequency]);

  const modelReady = viewState === "ready" && runtime != null;
  const canLoadSample = deriveCanLoadSample(selectedSampleId, modelReady);
  const samplingDisabledReason = deriveStartSamplingDisabledReason({
    runtime,
    studioConnected: wsConnected,
    selectedDevice,
    selectedSensorId,
    selectedFrequency,
    sampleLengthMs,
    compatibilityLoading,
    compatibility,
    streamingState,
    samplingState,
  });
  const liveActive = streamingState === "active" || streamingState === "starting";
  const canStartSampling = canStartLiveSampling(samplingDisabledReason, streamingState);
  const displayState = resolveLiveResultSource({
    resultSource,
    result,
    resultError,
    liveResult,
    liveResultError,
  });
  const displayResult = displayState.result;
  const displayResultError = displayState.error;
  const displaySampleMeta = resultSource === "sample" ? selectedSampleMeta : null;
  const displayImageUrl = resultSource === "sample" ? sampleImageUrl : null;

  const startKeepalive = useCallback((devicePk: string) => {
    clearKeepalive();
    activeStreamDeviceIdRef.current = devicePk;
    keepaliveRef.current = setInterval(() => {
      devicesApi.streamKeepalive(devicePk).catch(() => {
        clearKeepalive();
        activeStreamDeviceIdRef.current = null;
        activeRuntimeIdRef.current = null;
        setStreamingState("error");
        setSamplingState("error");
        setLiveResultError("Lost contact with the live classification stream.");
      });
    }, 10_000);
  }, [clearKeepalive]);

  const stopLiveClassification = useCallback(async (showToast = false) => {
    const devicePk = activeStreamDeviceIdRef.current ?? selectedDeviceId;
    if (!devicePk) return;

    setStreamingState("stopping");
    setSamplingState("stopping");
    clearKeepalive();

    try {
      await devicesApi.streamStop(devicePk);
      setStreamingState("idle");
      setSamplingState("idle");
      activeStreamDeviceIdRef.current = null;
      activeRuntimeIdRef.current = null;
      if (showToast) toast.success("Live classification stopped");
    } catch (error: any) {
      setStreamingState("error");
      setSamplingState("error");
      setLiveResultError(
        error?.response?.data?.detail || error?.message || "Failed to stop live classification",
      );
    }
  }, [clearKeepalive, selectedDeviceId]);

  const handleStartSampling = useCallback(async () => {
    if (liveActive) {
      await stopLiveClassification(true);
      return;
    }
    if (!selectedDevice || samplingDisabledReason) return;

    setStreamingState("starting");
    setSamplingState("starting");
    setLiveResultError(null);
    setLiveResult(null);
    setResultSource(null);
    setResultError(null);
    setSampleImageUrl(null);

    try {
      await devicesApi.streamInferenceStart(selectedDevice.id, undefined, {
        sensor: selectedSensorId || undefined,
        frequency: selectedFrequency ? parseFloat(selectedFrequency) : undefined,
        sample_length_ms: sampleLengthMs || undefined,
      });
      setStreamingState("active");
      setSamplingState("active");
      activeRuntimeIdRef.current = runtime?.id ?? null;
      startKeepalive(selectedDevice.id);
      toast.success("Live classification started");
    } catch (error: any) {
      setStreamingState("error");
      setSamplingState("error");
      setLiveResultError(
        error?.response?.data?.detail || error?.message || "Failed to start live classification",
      );
      toast.error(
        error?.response?.data?.detail || error?.message || "Failed to start live classification",
      );
    }
  }, [liveActive, runtime?.id, samplingDisabledReason, selectedDevice, startKeepalive, stopLiveClassification]);

  useEffect(() => {
    if (!activeStreamDeviceIdRef.current) return;
    if (selectedDeviceId && activeStreamDeviceIdRef.current !== selectedDeviceId) {
      stopLiveClassification();
    }
  }, [selectedDeviceId, stopLiveClassification]);

  useEffect(() => {
    if (!activeStreamDeviceIdRef.current) return;
    if (
      activeRuntimeIdRef.current != null &&
      runtime?.id != null &&
      activeRuntimeIdRef.current !== runtime.id
    ) {
      stopLiveClassification();
    }
  }, [runtime?.id, stopLiveClassification]);

  useEffect(() => {
    return () => {
      clearKeepalive();
    };
  }, [clearKeepalive]);

  function handleSampleChange(id: string) {
    setSelectedSampleId(id);
    setSelectedSampleMeta(testSamples.find((sample) => sample.id === id) ?? null);
    setResult(null);
    setResultError(null);
    setResultSource(null);
  }

  const handleLoadSample = useCallback(async () => {
    if (!canLoadSample || !impulseId) return;
    if (loadingResult) return;

    setLoadingResult(true);
    setResult(null);
    setResultError(null);
    setSampleImageUrl(null);

    try {
      const { data } = await liveClassificationApi.classifySample(
        buildClassifySampleRequest(impulseId, selectedSampleId, runtime),
      );
      setResult(data as LiveClassificationRunResult);
      setResultSource("sample");

      liveClassificationApi
        .sampleImageUrl(selectedSampleId)
        .then((imageResponse) => {
          setSampleImageUrl(getSamplePreviewUrl(imageResponse));
        })
        .catch(() => { });
    } catch (error: any) {
      const detail =
        error?.response?.data?.detail || error?.message || "Classification failed";
      setResultError(detail);
      toast.error(detail);
    } finally {
      setLoadingResult(false);
    }
  }, [canLoadSample, impulseId, loadingResult, runtime, selectedSampleId]);

  const startButtonLabel =
    streamingState === "starting"
      ? "Starting…"
      : streamingState === "stopping"
        ? "Stopping…"
        : liveActive
          ? "Stop sampling"
          : "Start sampling";

  if (activeProject?.project_type === "motion") {
    return (
      <MotionPhasePending
        feature="Live Classification"
        description="Once motion live classification is wired up, this page will classify a live USB sensor stream in real time with a rolling confidence graph."
        tip="Live classification for motion lands in a later phase — for now, design and train your impulse."
        icon={Smartphone}
      />
    );
  }

  if (viewState === "no_project") {
    return (
      <div className="pe-live-classification mx-auto max-w-7xl space-y-6">
        <div className="pe-lc-card flex flex-col items-center justify-center py-16 text-center">
          <Cpu size={32} className="text-gray-500 mb-4" />
          <h3 className="text-sm font-semibold text-[color:var(--app-text)]">No project selected</h3>
          <p className="mt-1 text-xs text-[color:var(--app-text-muted)] max-w-xs">
            Select a project from the sidebar to start live classification.
          </p>
        </div>
      </div>
    );
  }

  // Until the bootstrap effect commits a concrete viewState, we can't tell
  // whether to render the main UI or the "Almost there!" warning. Render a
  // loading state first so the warning is the first non-loading frame for
  // untrained impulses.
  if (viewState === "loading") {
    return (
      <div className="flex h-[calc(100vh-theme('spacing.16'))] items-center justify-center pt-2">
        <span className="inline-flex items-center gap-2 text-sm text-[var(--app-text-soft)]">
          <Loader2 size={14} className="animate-spin" />
          Verifying model status...
        </span>
      </div>
    );
  }

  if (viewState === "no_impulse" || viewState === "no_model") {
    return <AlmostThereWarning />;
  }

  return (
    <div className="pe-live-classification mx-auto max-w-7xl space-y-6">

      {modelLoading && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6" aria-label="loading-skeleton">
          <div className="pe-lc-card animate-pulse space-y-4">
            <div className="h-4 bg-gray-300/40 dark:bg-gray-800 rounded w-2/5" />
            {[0, 1, 2, 3].map((index) => (
              <div key={index} className="h-9 bg-gray-300/40 dark:bg-gray-800 rounded" />
            ))}
            <div className="h-9 bg-gray-300/40 dark:bg-gray-800 rounded" />
          </div>
          <div className="pe-lc-card animate-pulse space-y-4">
            <div className="h-4 bg-gray-300/40 dark:bg-gray-800 rounded w-2/5" />
            <div className="h-9 bg-gray-300/40 dark:bg-gray-800 rounded" />
            <div className="h-9 bg-gray-300/40 dark:bg-gray-800 rounded" />
          </div>
        </div>
      )}

      {!modelLoading && viewState === "ready" && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 items-start">
          <div className="pe-lc-card">
            <div className="pe-lc-card-head">
              <div className="pe-lc-card-icon" aria-hidden="true">
                <Activity size={20} strokeWidth={2.2} />
              </div>
              <div>
                <h3 className="pe-lc-card-title">Classify new data</h3>
                <p className="pe-lc-card-sub">
                  Collect a sample directly from a connected device
                </p>
              </div>
            </div>

            <div>
              <label className="label" htmlFor="lc-device">Device</label>
              {devicesLoading ? (
                <div className="input animate-pulse text-transparent select-none">…</div>
              ) : (
                <select
                  id="lc-device"
                  aria-label="device-select"
                  className="input"
                  value={selectedDeviceId}
                  onChange={(event) => {
                    setSelectedDeviceId(event.target.value);
                    setSelectedSensorId("");
                    setSelectedFrequency("");
                    setLiveResultError(null);
                  }}
                >
                  <option value="">
                    {devices.length === 0 ? "No devices connected" : "Select device…"}
                  </option>
                  {devices.map((device) => (
                    <option key={device.id} value={device.id}>
                      {device.name}
                      {device.is_online ? "" : " (offline)"}
                    </option>
                  ))}
                </select>
              )}
            </div>

            <div>
              <label className="label" htmlFor="lc-sensor">Sensor</label>
              <select
                id="lc-sensor"
                aria-label="sensor-select"
                className="input"
                value={selectedSensorId}
                disabled={!selectedDeviceId}
                onChange={(event) => {
                  setSelectedSensorId(event.target.value);
                  setSelectedFrequency("");
                }}
              >
                <option value="">
                  {!selectedDeviceId ? "Select a device first" : "Select sensor…"}
                </option>
                {selectedDeviceId && availableSensorOptions.map((sensor) => (
                  <option key={sensor.id} value={sensor.id}>{sensor.name}</option>
                ))}
              </select>
              {selectedDeviceId && sensorConfig.usingFallback && (
                <p className="mt-1 text-xs text-[color:var(--app-text-muted)]">
                  Using fallback sensor options because this device has not reported a sensor manifest yet.
                </p>
              )}
            </div>

            <div>
              <label className="label" htmlFor="lc-length">Sample length (ms)</label>
              <input
                id="lc-length"
                aria-label="sample-length"
                type="number"
                className="input"
                value={sampleLengthMs}
                min={100}
                step={100}
                onChange={(event) => setSampleLengthMs(Number(event.target.value))}
              />
            </div>

            <div>
              <label className="label" htmlFor="lc-freq">Frequency (Hz)</label>
              <select
                id="lc-freq"
                aria-label="frequency-select"
                className="input"
                value={selectedFrequency}
                disabled={!selectedSensorId}
                onChange={(event) => setSelectedFrequency(event.target.value)}
              >
                <option value="">
                  {!selectedSensorId ? "Select a sensor first" : "Select frequency…"}
                </option>
                {selectedSensorId && availableFrequencies.map((frequency) => (
                  <option key={frequency} value={frequency}>{frequency} Hz</option>
                ))}
              </select>
              {selectedSensorOption?.source === "fallback" && selectedSensorId && (
                <p className="mt-1 text-xs text-[color:var(--app-text-muted)]">
                  Frequency options are using fallback values for this sensor.
                </p>
              )}
            </div>

            <button
              aria-label="start-sampling-btn"
              className="btn-primary pe-lc-cta"
              disabled={!liveActive && !canStartSampling}
              onClick={handleStartSampling}
            >
              {streamingState === "starting" || streamingState === "stopping" ? (
                <>
                  <Loader2 size={16} className="animate-spin" />
                  {startButtonLabel}
                </>
              ) : (
                <>
                  <Smartphone size={16} />
                  {startButtonLabel}
                </>
              )}
            </button>

            {!liveActive && samplingDisabledReason && (
              <p className="pe-lc-helper">{samplingDisabledReason}</p>
            )}
            {liveActive && (
              <p className="pe-lc-helper text-emerald-600 dark:text-emerald-400">
                Live classification is running on {selectedDevice?.name ?? "the selected device"}.
              </p>
            )}
            {liveResultError && (
              <p className="pe-lc-helper text-red-500 dark:text-red-400">{liveResultError}</p>
            )}
          </div>

          <div className="pe-lc-card">
            <div className="pe-lc-card-head">
              <div className="pe-lc-card-icon" aria-hidden="true">
                <FlaskConical size={20} strokeWidth={2.2} />
              </div>
              <div>
                <h3 className="pe-lc-card-title">Classify existing test sample</h3>
                <p className="pe-lc-card-sub">
                  Run inference on a sample from your test set
                </p>
              </div>
            </div>

            <div>
              <label className="label" htmlFor="lc-sample">Test sample</label>
              {testSamplesLoading ? (
                <div className="input animate-pulse text-transparent select-none">…</div>
              ) : (
                <select
                  id="lc-sample"
                  aria-label="sample-select"
                  className="input"
                  value={selectedSampleId}
                  disabled={testSamples.length === 0}
                  onChange={(event) => handleSampleChange(event.target.value)}
                >
                  <option value="">
                    {testSamples.length === 0
                      ? "No test samples available"
                      : "Select a sample…"}
                  </option>
                  {testSamples.map((sample) => (
                    <option key={sample.id} value={sample.id}>
                      {sample.name} ({sample.label})
                    </option>
                  ))}
                </select>
              )}
            </div>

            <button
              aria-label="load-sample-btn"
              className="btn-primary pe-lc-cta"
              disabled={!canLoadSample || loadingResult}
              onClick={handleLoadSample}
            >
              {loadingResult ? (
                <>
                  <Loader2 size={16} className="animate-spin" />
                  Classifying…
                </>
              ) : (
                <>
                  <FlaskConical size={16} />
                  Load sample
                </>
              )}
            </button>

            {resultError && resultSource !== "live" && (
              <p className="pe-lc-helper text-red-500 dark:text-red-400">{resultError}</p>
            )}
          </div>
        </div>
      )}

      {loadingResult && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <div className="card animate-pulse space-y-4">
            <div className="h-4 bg-gray-800 rounded w-1/3" />
            <div className="h-8 bg-gray-800 rounded w-1/2" />
            {[0, 1, 2].map((index) => (
              <div key={index} className="h-4 bg-gray-800 rounded" />
            ))}
          </div>
          <div className="card animate-pulse">
            <div className="h-48 bg-gray-800 rounded" />
          </div>
        </div>
      )}

      {displayResultError && !displayResult && (
        <div className="card">
          <p className="text-sm text-red-400">{displayResultError}</p>
        </div>
      )}

      {displayResult && !loadingResult && (
        <div aria-label="result-section" className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <LiveClassificationSummaryCard result={displayResult} />
          <LiveClassificationResultPanel
            result={displayResult}
            sampleMeta={displaySampleMeta}
            imageUrl={displayImageUrl}
          />
        </div>
      )}
    </div>
  );
}

function AlmostThereWarning() {
  return (
    <ImpulseNotReady
      description="Your impulse is not fully trained. Use the items in the navigation bar to configure and train your model before you can classify new data."
      tip="Train your model first — live classification needs a trained model to score incoming samples."
    />
  );
}
