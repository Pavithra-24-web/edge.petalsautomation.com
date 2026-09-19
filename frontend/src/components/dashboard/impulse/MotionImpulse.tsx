"use client";
import { useEffect, useState, useCallback, useMemo } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import { useAppStore } from "@/store/appStore";
import { impulsesApi, samplesApi, dspApi } from "@/utils/api";
import {
  Plus, Trash2, Zap, RefreshCw, Settings,
  CheckCircle, Activity, Database, FlaskConical,
  AlertCircle, Rocket,
} from "lucide-react";
import toast from "react-hot-toast";
import PremiumSaveButton from "@/components/PremiumSaveButton";
import ImpulseNotReady from "@/components/dashboard/ImpulseNotReady";
import {
  Arrow, SadFolderIcon, ProcessingBlockModal, BlockPickerModal, ProcessingBlockBase,
  normalizeLearningAuthor,
  BLOCK_ACCENT, PROCESSING_BLOCK_BADGE, DEFAULT_PROCESSING_BADGE, LEARNING_BLOCK_BADGE,
} from "./ImpulseShared";
import { computeFeatureChain, resolveSelectedAxes, FLATTEN_STAT_LABELS, FLATTEN_DEFAULT_STATS } from "./featurePipeline";

/* ─────────────────────────────────────────────────────────────────────────────
   New-impulse default payload — mirrors the backend's own motion seed
   (`_seed_from_project_type`, impulses.py) so a freshly created draft already
   shows the values the server would assign. `datasetAxes` is whatever the
   project's uploaded CSVs actually declared (samplesApi.featureAxes) — never
   a fixed axis list, since the sensor layout varies per dataset.
───────────────────────────────────────────────────────────────────────────── */
function buildEmptyImpulsePayload(projectId: string, datasetAxes: string[]) {
  return {
    project_id: projectId,
    window_size_ms: 1000,
    window_increase_ms: 500,
    frequency_hz: 62.5,
    input_type: "time-series",
    input_axes: datasetAxes,
    sensor_type: "accelerometer",
    train_subset_percent: 100,
    dsp_blocks: [],
    ml_blocks: [],
  };
}

/* ─────────────────────────────────────────────────────────────────────────────
   "Add a learning block" modal — motion offers whatever the backend's block
   registry allows for a time-series/accelerometer impulse (Classification,
   Conv1D, LSTM, Anomaly Detection today — see motion_phase0.md §7.3). Unlike
   Object Detection this has no fixed single entry, so no client-side filter
   or rename is applied: the catalog is shown as the backend returns it.
───────────────────────────────────────────────────────────────────────────── */
function MotionLearningBlockModal({
  impulseId, projectType, onAdd, onClose,
}: {
  impulseId: string;
  projectType?: string;
  onAdd: (block: any) => void;
  onClose: () => void;
}) {
  const [blocks, setBlocks] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);

  const load = useCallback(async () => {
    setLoading(true);
    try {
      const { data } = await dspApi.learningBlocks(impulseId, false, projectType);
      const visible = (data.blocks || []).map((block: any) => ({
        ...block,
        author: normalizeLearningAuthor(block.author),
      }));
      setBlocks(visible);
    } catch { toast.error("Failed to load learning blocks"); }
    finally { setLoading(false); }
  }, [impulseId, projectType]);

  useEffect(() => { load(); }, [load]);

  return (
    <BlockPickerModal
      titleIcon={FlaskConical}
      title="Add a learning block"
      loading={loading}
      blocks={blocks}
      emptyMessage="No learning blocks are available for this impulse."
      showFooterCancel
      onAdd={onAdd}
      onClose={onClose}
    />
  );
}

/* ─────────────────────────────────────────────────────────────────────────────
   Flatten's block-specific parameter — which per-axis statistics it computes.
   This is the one param that changes Flatten's output feature *names*
   (featurePipeline.ts reads params.features the same way), so it gets a real
   editor; every other block's params still render as the existing readonly
   fields below.
───────────────────────────────────────────────────────────────────────────── */
function FlattenFeatureParams({ block, onChange }: { block: any; onChange: (features: string[]) => void }) {
  const selected: string[] = block.params?.features?.length ? block.params.features : FLATTEN_DEFAULT_STATS;
  return (
    <div>
      <p className="text-[9px] text-gray-500 uppercase tracking-wider mb-1">Features</p>
      <div className="flex flex-wrap gap-1.5">
        {Object.entries(FLATTEN_STAT_LABELS).map(([stat, label]) => {
          const active = selected.includes(stat);
          return (
            <button
              key={stat}
              type="button"
              onClick={() => {
                const next = active ? selected.filter((s) => s !== stat) : [...selected, stat];
                if (next.length) onChange(next);
              }}
              className={`px-2 py-0.5 rounded-full text-[10px] border transition-colors ${
                active
                  ? "border-yellow-400/60 bg-yellow-500/20 text-yellow-200"
                  : "border-gray-600 text-gray-400 hover:border-gray-400"
              }`}
            >
              {label}
            </button>
          );
        })}
      </div>
    </div>
  );
}

/* ─────────────────────────────────────────────────────────────────────────────
   Main page — Impulse pipeline builder for a motion project. Same canvas
   architecture as ObjectDetectionImpulse (Input → Processing → Learning →
   Output), with the Input block fixed to time-series (motion never has an
   image branch) and its own learning-block picker.
───────────────────────────────────────────────────────────────────────────── */
export default function MotionImpulse() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const { activeProject, activeImpulse, savedActiveImpulse, setActiveImpulse, setSavedActiveImpulse, hasHydrated } = useAppStore();

  const [saving, setSaving] = useState(false);

  const [showDSPModal, setShowDSPModal] = useState(false);
  const [showMLModal, setShowMLModal] = useState(false);

  const [validationErrors] = useState<string[]>([]);

  const [subsetDraft, setSubsetDraft] = useState<string | null>(null);

  // null = readiness check in flight, true = at least one sample exists,
  // false = project has no uploaded recordings yet → empty-data screen.
  const [hasData, setHasData] = useState<boolean | null>(null);

  const [outputLabels, setOutputLabels] = useState<{ count: number; names: string[] }>({ count: 0, names: [] });
  const [outputLabelsLoading, setOutputLabelsLoading] = useState(false);

  // The dataset's own feature/axis names — parsed server-side from uploaded
  // CSV headers (or device-declared sensor names). This, not any literal
  // axis list, is what the first processing block's input features come from.
  const [datasetFeatures, setDatasetFeatures] = useState<string[]>([]);

  useEffect(() => {
    if (!activeProject?.id) {
      setDatasetFeatures([]);
      return;
    }
    let cancelled = false;
    samplesApi
      .featureAxes(activeProject.id)
      .then(({ data }) => {
        if (cancelled) return;
        setDatasetFeatures(Array.isArray(data?.features) ? data.features : []);
      })
      .catch(() => {
        if (!cancelled) setDatasetFeatures([]);
      });
    return () => {
      cancelled = true;
    };
  }, [activeProject?.id]);

  // Keep the pipeline in sync with the dataset: prune any axis a block or
  // the root input selected that the dataset no longer has, defaulting back
  // to "everything available" wherever a selection goes empty. Cascades
  // block-by-block so a change to the root axes (or to an earlier block's
  // output) reconciles every block downstream in one pass.
  useEffect(() => {
    if (!activeImpulse || datasetFeatures.length === 0) return;

    const rootCurrent: string[] = activeImpulse.input_axes || [];
    const rootNext = resolveSelectedAxes(datasetFeatures, rootCurrent);

    let cascadeAvailable = rootNext;
    let blocksChanged = false;
    const nextBlocks = (activeImpulse.dsp_blocks || []).map((block: any) => {
      const currentSel: string[] = block.input_axes || [];
      const selected = resolveSelectedAxes(cascadeAvailable, currentSel);
      const changed =
        selected.length !== currentSel.length || selected.some((a, i) => a !== currentSel[i]);
      if (changed) blocksChanged = true;
      const nextBlock = changed ? { ...block, input_axes: selected } : block;
      cascadeAvailable = computeFeatureChain(cascadeAvailable, [nextBlock]).outputs[0] || [];
      return nextBlock;
    });

    const rootChanged =
      rootNext.length !== rootCurrent.length || rootNext.some((a, i) => a !== rootCurrent[i]);
    if (rootChanged || blocksChanged) {
      setActiveImpulse({ ...activeImpulse, input_axes: rootNext, dsp_blocks: nextBlocks });
    }
    // Only re-run when the dataset itself changes or a different impulse loads —
    // not on every keystroke/toggle, which already writes the correct value directly.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [datasetFeatures, activeImpulse?.id]);

  useEffect(() => {
    if (hasHydrated && activeProject) loadImpulses();
  }, [hasHydrated, activeProject]);

  useEffect(() => {
    if (!activeProject?.id) {
      setHasData(null);
      return;
    }
    let cancelled = false;
    setHasData(null);
    samplesApi
      .list(activeProject.id, { limit: 1 })
      .then(({ data }) => {
        if (cancelled) return;
        const count = Array.isArray(data)
          ? data.length
          : Array.isArray(data?.items)
            ? data.items.length
            : Number(data?.total ?? data?.count ?? 0);
        setHasData(count > 0);
      })
      .catch(() => {
        if (!cancelled) setHasData(false);
      });
    return () => {
      cancelled = true;
    };
  }, [activeProject?.id]);

  useEffect(() => {
    if (!activeProject?.id) {
      setOutputLabels({ count: 0, names: [] });
      setOutputLabelsLoading(false);
      return;
    }
    let cancelled = false;
    setOutputLabelsLoading(true);
    samplesApi
      .labelsSummary(activeProject.id)
      .then(({ data }) => {
        if (cancelled) return;
        const names: string[] = Array.isArray(data?.names) ? data.names : [];
        const count: number = typeof data?.count === "number" ? data.count : names.length;
        setOutputLabels({ count, names });
      })
      .catch(() => {
        if (!cancelled) setOutputLabels({ count: 0, names: [] });
      })
      .finally(() => {
        if (!cancelled) setOutputLabelsLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [activeProject?.id]);

  async function loadImpulses() {
    if (!activeProject) return;
    try {
      const { data } = await impulsesApi.list(activeProject.id);
      if (data.length === 0) return;

      const urlId = searchParams.get("impulseId");
      if (urlId) {
        const match = data.find((i: any) => i.id === urlId);
        if (match) {
          setActiveImpulse(match);
          setSavedActiveImpulse(match);
          return;
        }
        try {
          const { data: fetched } = await impulsesApi.get(urlId);
          setActiveImpulse(fetched);
          setSavedActiveImpulse(fetched);
          return;
        } catch { /* fall through */ }
      }
      if (activeImpulse && data.some((i: any) => i.id === activeImpulse.id)) return;
      setActiveImpulse(data[0]);
      setSavedActiveImpulse(data[0]);
    } catch (err) {
      console.error("Failed to load impulses:", err);
      toast.error("Failed to load impulses");
    }
  }

  async function createImpulse() {
    if (!activeProject) return;
    try {
      const { data } = await impulsesApi.create(buildEmptyImpulsePayload(activeProject.id, datasetFeatures));
      setActiveImpulse(data);
      setSavedActiveImpulse(data);
      window.dispatchEvent(new Event("impulses:changed"));
      router.replace(`/dashboard/impulse?impulseId=${data.id}`);
    } catch (e: any) {
      toast.error(e?.response?.data?.detail || "Failed to create impulse");
    }
  }

  async function saveImpulse() {
    if (!activeImpulse) return;
    setSaving(true);
    try {
      const { data } = await impulsesApi.update(activeImpulse.id, activeImpulse);
      setActiveImpulse(data);
      setSavedActiveImpulse(data);
    } catch (e: any) {
      toast.error(e.message || "Save failed");
      throw e;
    } finally { setSaving(false); }
  }

  function addDSPBlock(block: any) {
    if (!activeImpulse) return;
    const defaultParams = Object.fromEntries(
      Object.entries(block.params || {}).map(([key, schema]: [string, any]) => [key, schema?.default])
    );
    const prevBlocks = activeImpulse.dsp_blocks || [];
    const chain = computeFeatureChain(activeImpulse.input_axes || [], prevBlocks);
    // A new block starts with every feature available to it selected —
    // whatever the last block in the chain outputs, or the dataset's own
    // axes when this is the first block.
    const availableForNewBlock = chain.outputs.length
      ? chain.outputs[chain.outputs.length - 1]
      : (activeImpulse.input_axes || []);
    const newBlock = { type: block.type, name: block.name, params: defaultParams, input_axes: [...availableForNewBlock] };
    setActiveImpulse({ ...activeImpulse, dsp_blocks: [...prevBlocks, newBlock] });
  }

  function removeDSPBlock(idx: number) {
    if (!activeImpulse) return;
    const nextBlocks = (activeImpulse.dsp_blocks || []).filter((_: any, i: number) => i !== idx);
    setActiveImpulse({ ...activeImpulse, dsp_blocks: nextBlocks });
  }

  function updateDSPBlockField(idx: number, field: string, value: any) {
    if (!activeImpulse) return;
    const nextBlocks = (activeImpulse.dsp_blocks || []).map((b: any, i: number) =>
      i === idx ? { ...b, [field]: value } : b
    );
    setActiveImpulse({ ...activeImpulse, dsp_blocks: nextBlocks });
  }

  function addMLBlock(block: any) {
    if (!activeImpulse) return;
    const newBlock = { type: block.type, name: block.name, params: {} };
    setActiveImpulse({ ...activeImpulse, ml_blocks: [newBlock] });
  }

  function removeMLBlock(idx: number) {
    if (!activeImpulse) return;
    const nextBlocks = (activeImpulse.ml_blocks || []).filter((_: any, i: number) => i !== idx);
    setActiveImpulse({ ...activeImpulse, ml_blocks: nextBlocks });
  }

  function updateField(key: string, value: any) {
    if (!activeImpulse) return;
    setActiveImpulse({ ...activeImpulse, [key]: value });
  }

  const dspBlocks = activeImpulse?.dsp_blocks || [];
  const mlBlocks = activeImpulse?.ml_blocks || [];
  const outputReady = dspBlocks.length > 0 && mlBlocks.length > 0;

  // The single place feature propagation is computed: block i's available
  // input is block i-1's output (or the dataset's own axes for block 0).
  const featureChain = useMemo(
    () => computeFeatureChain(activeImpulse?.input_axes || [], dspBlocks),
    [activeImpulse?.input_axes, dspBlocks]
  );
  const outputLabelsSummary = outputLabels.count > 0
    ? `${outputLabels.count} (${outputLabels.names.join(", ")})`
    : null;

  // What actually reaches the learning block: the last processing block's
  // output features, or the dataset's own axes when there's no processing
  // block yet. Pure display — reads the same chain the canvas already
  // computes, never recomputes feature propagation itself. Rendered as chips
  // in the Learning card below.
  const learningInputFeatures = featureChain.outputs.length
    ? featureChain.outputs[featureChain.outputs.length - 1]
    : (activeImpulse?.input_axes || []);

  return (
    <div className="pe-impulse impulse-page min-h-screen">

      {/* Header */}
      <div className="mb-8">
        <div className="flex items-start justify-between gap-6 flex-wrap">
          <div className="flex items-start gap-4 min-w-0">
            <div className="pe-impulse-header-icon flex-shrink-0">
              <Activity size={22} strokeWidth={2.2} />
            </div>
            <div className="min-w-0">
              <h1 className="pe-impulse-title">{activeImpulse?.name || "Impulse Design"}</h1>
              <p className="pe-impulse-sub">
                An impulse takes raw motion data, uses signal processing to extract features,
                and then uses a learning block to classify new data.
              </p>
            </div>
          </div>
          <div className="flex items-center gap-3 flex-shrink-0">
            <button onClick={() => router.push("/dashboard/impulse/manage")}
              className="pe-impulse-toolbar-btn">
              <Settings size={14} /> Manage impulse
            </button>
            <button onClick={createImpulse} className="pe-impulse-toolbar-btn">
              <Plus size={14} /> New impulse
            </button>
            {activeImpulse && (
              <PremiumSaveButton
                onSave={saveImpulse}
                onSuccess={() => toast.success("Impulse saved")}
                label="Save Impulse"
                loadingLabel="Saving..."
                disabled={saving}
              />
            )}
          </div>
        </div>
      </div>

      {activeProject && hasData === null ? (
        <div className="flex flex-col items-center justify-center py-32">
          <span className="inline-flex items-center gap-2 text-sm text-[var(--app-text-soft)]">
            <RefreshCw size={14} className="animate-spin" />
            Checking project data…
          </span>
        </div>
      ) : hasData === false ? (
        <ImpulseNotReady
          title="No recordings collected yet"
          description="You'll need some motion recordings to design your first impulse."
          icon={<SadFolderIcon />}
          actions={
            <button
              type="button"
              onClick={() => router.push("/dashboard/data")}
              className="pe-warn-cta"
            >
              <Rocket size={15} /> Go to data acquisition
            </button>
          }
        />
      ) : !activeImpulse ? (
        <div className="flex flex-col items-center justify-center py-32 border-2 border-dashed border-gray-700 rounded-2xl">
          <Zap size={36} className="text-gray-600 mb-4" />
          <p className="text-gray-400 text-sm mb-4">No impulse yet — create one to build your pipeline</p>
          <button onClick={createImpulse}
            className="flex items-center gap-2 px-4 py-2 text-sm font-medium text-white bg-brand-600 hover:bg-brand-500 rounded-lg transition-colors">
            <Plus size={14} /> Create impulse
          </button>
        </div>
      ) : (
        <>
          <div className="impulse-canvas">
            <div className="overflow-x-auto pb-2">
              <div className="flex min-w-max items-start justify-center gap-0 xl:min-w-full">

                {/* Input block — motion is always time-series */}
                <div className="impulse-input-block w-[clamp(17rem,22vw,20rem)] min-h-[26rem] border border-gray-700 rounded-xl flex-shrink-0 overflow-visible relative">
                  <div className="impulse-input-block-header bg-gradient-to-br from-violet-500 to-purple-500 px-5 py-4 flex items-center justify-between rounded-t-xl">
                    <p className="text-base font-semibold text-white">Time-series data</p>
                    <div className="impulse-input-icon h-10 w-10 bg-black/20 rounded-full flex items-center justify-center">
                      <Database size={20} className="text-white" />
                    </div>
                  </div>
                  <div className="p-4 sm:p-5 space-y-3">
                    <div>
                      <p className="text-[10px] text-gray-500 mb-0.5">Input axes</p>
                      <input type="text" className="impulse-field w-full rounded px-2 py-1 text-xs focus:border-brand-500 focus:outline-none"
                        value={(activeImpulse.input_axes || []).join(", ")}
                        onChange={e => updateField("input_axes", e.target.value.split(',').map((s: string) => s.trim()).filter(Boolean))} />
                    </div>
                    <div className="grid grid-cols-2 gap-2 mt-2">
                      <div>
                        <p className="text-[10px] text-gray-500 mb-0.5">Window (ms)</p>
                        <input type="number" className="impulse-field w-full rounded px-2 py-1 text-xs focus:border-brand-500 focus:outline-none"
                          value={activeImpulse.window_size_ms} onChange={e => updateField("window_size_ms", Number(e.target.value))} />
                      </div>
                      <div>
                        <p className="text-[10px] text-gray-500 mb-0.5">Increase (ms)</p>
                        <input type="number" className="impulse-field w-full rounded px-2 py-1 text-xs focus:border-brand-500 focus:outline-none"
                          value={activeImpulse.window_increase_ms} onChange={e => updateField("window_increase_ms", Number(e.target.value))} />
                      </div>
                    </div>
                    <div>
                      <p className="text-[10px] text-gray-500 mb-0.5">Frequency (Hz)</p>
                      <input type="number" className="impulse-field w-full rounded px-2 py-1 text-xs focus:border-brand-500 focus:outline-none"
                        value={activeImpulse.frequency_hz} onChange={e => updateField("frequency_hz", Number(e.target.value))} />
                    </div>
                    <div>
                      <p className="text-[10px] text-gray-500 mb-0.5">Train on data subset</p>
                      <div className="flex items-center gap-1.5">
                        <input
                          type="number"
                          min={1}
                          max={100}
                          step={1}
                          inputMode="numeric"
                          className="impulse-field w-20 rounded px-2 py-1 text-xs focus:border-brand-500 focus:outline-none"
                          value={subsetDraft ?? (activeImpulse.train_subset_percent ?? 100)}
                          onKeyDown={e => {
                            if ([".", ",", "e", "E", "+", "-"].includes(e.key)) e.preventDefault();
                          }}
                          onChange={e => {
                            const raw = e.target.value.replace(/\D/g, "");
                            if (raw === "") { setSubsetDraft(""); return; }
                            setSubsetDraft(raw);
                            const n = Math.min(100, Math.max(1, parseInt(raw, 10)));
                            updateField("train_subset_percent", n);
                          }}
                          onBlur={() => {
                            if (subsetDraft === "" || subsetDraft === null) {
                              updateField("train_subset_percent", activeImpulse.train_subset_percent ?? 100);
                            }
                            setSubsetDraft(null);
                          }}
                        />
                        <span className="text-xs text-gray-500 select-none">%</span>
                      </div>
                    </div>
                  </div>
                </div>

                <Arrow />

                {/* Processing block stack */}
                <div className="flex flex-col items-stretch gap-5 w-[clamp(17rem,22vw,20rem)] flex-shrink-0">
                  {dspBlocks.map((block: any, idx: number) => {
                    const badge = PROCESSING_BLOCK_BADGE[block.type] || DEFAULT_PROCESSING_BADGE;
                    const available = featureChain.inputs[idx] || [];
                    const selected = resolveSelectedAxes(available, block.input_axes);
                    return (
                      <ProcessingBlockBase
                        key={idx}
                        block={block}
                        badge={badge}
                        accentClass={BLOCK_ACCENT[block.type] || "border-gray-600/50 bg-gray-800/40"}
                        availableFeatures={available}
                        selectedFeatures={selected}
                        onAxesChange={(axes) => updateDSPBlockField(idx, "input_axes", axes)}
                        onDelete={() => removeDSPBlock(idx)}
                      >
                        {block.type === "flatten" && (
                          <FlattenFeatureParams
                            block={block}
                            onChange={(features) => updateDSPBlockField(idx, "params", { ...block.params, features })}
                          />
                        )}
                      </ProcessingBlockBase>
                    );
                  })}

                  {dspBlocks.length === 0 && (
                    <button
                      onClick={() => setShowDSPModal(true)}
                      className="impulse-add-card w-full min-h-[26rem] border-2 border-dashed border-gray-600 rounded-xl flex flex-col items-center justify-center gap-2 px-4 py-5 hover:border-indigo-500/60 hover:bg-indigo-900/10 transition-all group"
                    >
                      <div className="impulse-add-card-icon w-11 h-11 rounded-full border-2 border-dashed border-gray-600 group-hover:border-indigo-400 flex items-center justify-center transition-colors">
                        <Zap size={15} className="text-gray-500 group-hover:text-indigo-400 transition-colors" />
                      </div>
                      <span className="text-sm text-gray-500 group-hover:text-gray-300 transition-colors text-center px-4 leading-relaxed">
                        Add a processing block
                      </span>
                    </button>
                  )}
                </div>

                <Arrow />

                {/* Learning lane */}
                <div className="flex flex-col items-stretch gap-5 w-[clamp(17rem,22vw,20rem)] flex-shrink-0">
                  {mlBlocks.map((block: any, idx: number) => {
                    const LearningBadgeIcon = LEARNING_BLOCK_BADGE.icon;
                    const CHIP_LIMIT = 8;
                    const visibleChips = learningInputFeatures.slice(0, CHIP_LIMIT);
                    const overflowCount = learningInputFeatures.length - visibleChips.length;
                    return (
                      <div key={idx} className="impulse-learning-card w-full min-h-[26rem] border border-indigo-500/50 bg-indigo-900/20 rounded-xl flex-shrink-0 overflow-visible relative">
                        <div className="px-5 py-4 flex items-center justify-between rounded-t-xl border-b border-indigo-400/10">
                          <div className="flex-1 min-w-0 pr-3">
                            <p className="text-[9px] text-indigo-400 uppercase tracking-wider mb-0.5">Learning</p>
                            <p className="text-base font-semibold text-gray-100">{block.name || (block?.type ?? "").replace(/_/g, " ")}</p>
                          </div>
                          <div className="flex items-center gap-2 flex-shrink-0">
                            <button
                              type="button"
                              aria-label="Delete learning block"
                              onClick={() => removeMLBlock(idx)}
                              className="p-1.5 rounded text-gray-500 hover:text-red-400 hover:bg-red-500/10 transition-colors"
                            >
                              <Trash2 size={13} />
                            </button>
                            <div className={`h-10 w-10 rounded-full flex items-center justify-center shadow-md ${LEARNING_BLOCK_BADGE.bg}`}>
                              <LearningBadgeIcon size={20} className={LEARNING_BLOCK_BADGE.fg} strokeWidth={2.5} />
                            </div>
                          </div>
                        </div>
                        <div className="p-4 sm:p-5 space-y-3">
                          <button onClick={() => setShowMLModal(true)}
                            className="impulse-learning-action w-full text-sm border rounded-lg py-2 transition-colors">
                            Change block
                          </button>
                          <div className="impulse-learning-section">
                            <p className="impulse-learning-section-label text-[10px] uppercase tracking-wider text-indigo-300/80 mb-1">
                              Input features ({learningInputFeatures.length})
                            </p>
                            {learningInputFeatures.length === 0 ? (
                              <p className="impulse-learning-section-value text-sm text-gray-100">No features yet</p>
                            ) : (
                              <div className="flex flex-wrap gap-1.5">
                                {visibleChips.map((name: string) => (
                                  <span key={name}
                                    className="px-2 py-0.5 rounded-full text-[10px] border border-indigo-400/40 bg-indigo-500/15 text-indigo-200">
                                    {name}
                                  </span>
                                ))}
                                {overflowCount > 0 && (
                                  <span className="px-2 py-0.5 rounded-full text-[10px] border border-indigo-400/30 bg-indigo-500/10 text-indigo-300/80">
                                    +{overflowCount} more
                                  </span>
                                )}
                              </div>
                            )}
                          </div>
                          <div className="impulse-learning-section">
                            <p className="impulse-learning-section-label text-[10px] uppercase tracking-wider text-indigo-300/80 mb-1">Output features</p>
                            {outputLabelsLoading && !outputLabelsSummary ? (
                              <span aria-hidden="true" className="inline-block h-4 w-32 rounded bg-gray-700/60 animate-pulse" />
                            ) : (
                              <p className="impulse-learning-section-value text-sm text-gray-100">
                                {outputLabelsSummary ?? "0 (no labels assigned)"}
                              </p>
                            )}
                          </div>
                        </div>
                      </div>
                    );
                  })}

                  {mlBlocks.length === 0 && (
                    <button
                      onClick={() => setShowMLModal(true)}
                      className="impulse-add-card w-full min-h-[26rem] border-2 border-dashed border-gray-600 rounded-xl flex flex-col items-center justify-center gap-2 px-4 py-5 hover:border-indigo-500/60 hover:bg-indigo-900/10 transition-all group"
                    >
                      <div className="impulse-add-card-icon w-11 h-11 rounded-full border-2 border-dashed border-gray-600 group-hover:border-indigo-400 flex items-center justify-center transition-colors">
                        <FlaskConical size={15} className="text-gray-500 group-hover:text-indigo-400 transition-colors" />
                      </div>
                      <span className="text-sm text-gray-500 group-hover:text-gray-300 transition-colors text-center px-4 leading-relaxed">
                        Add a learning block
                      </span>
                    </button>
                  )}
                </div>

                <Arrow />

                {/* Output features */}
                <div className={`impulse-output-card w-[clamp(17rem,22vw,20rem)] min-h-[21rem] border rounded-xl flex-shrink-0 flex flex-col items-center justify-center px-4 ${outputReady ? "border-emerald-600/50 bg-emerald-900/10" : "border-gray-700/50 bg-gray-800/20"}`}>
                  <div className={`w-12 h-12 rounded-full flex items-center justify-center mb-3 ${outputReady ? "bg-emerald-500" : "bg-gray-700"}`}>
                    {outputReady ? <CheckCircle size={20} className="text-white" /> : <Settings size={15} className="text-gray-400" />}
                  </div>
                  <p className={`text-sm font-semibold text-center ${outputReady ? "text-emerald-300" : "text-gray-500"}`}>
                    Output features
                  </p>
                  {outputReady && (
                    outputLabelsLoading && !outputLabelsSummary ? (
                      <span aria-hidden="true" className="mt-1 inline-block h-4 w-32 rounded bg-gray-700/60 animate-pulse" />
                    ) : (
                      <p className="impulse-output-card-summary text-[12px] text-center text-gray-100 mt-1 px-2">
                        {outputLabelsSummary ?? "0 (no labels assigned)"}
                      </p>
                    )
                  )}
                </div>

              </div>
            </div>
          </div>

          {validationErrors.length > 0 && (
            <div className="mt-4 p-3 bg-red-900/20 border border-red-800/50 rounded-xl space-y-1">
              {validationErrors.map((e, i) => (
                <div key={i} className="flex items-center gap-2 text-sm text-red-400">
                  <AlertCircle size={13} className="flex-shrink-0" /> {e}
                </div>
              ))}
            </div>
          )}
        </>
      )}

      {/* Modals */}
      {showDSPModal && activeImpulse && (
        <ProcessingBlockModal
          inputType={activeImpulse.input_type || ""}
          sensorType={activeImpulse.sensor_type || ""}
          projectType={activeProject?.project_type}
          onAdd={addDSPBlock}
          onClose={() => setShowDSPModal(false)}
        />
      )}
      {showMLModal && activeImpulse && (
        <MotionLearningBlockModal
          impulseId={activeImpulse.id}
          projectType={activeProject?.project_type}
          onAdd={addMLBlock}
          onClose={() => setShowMLModal(false)}
        />
      )}
    </div>
  );
}
