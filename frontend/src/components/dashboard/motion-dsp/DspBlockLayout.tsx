"use client";
import { ReactNode, useEffect, useMemo, useState } from "react";
import { useSearchParams, useRouter } from "next/navigation";
import toast from "react-hot-toast";
import { useAppStore } from "@/store/appStore";
import { impulsesApi, dspApi } from "@/utils/api";
import { DspTabs } from "./common";
import ParameterForm, { DegradedField, ParamSchema } from "./ParameterForm";
import SaveParametersButton from "./SaveParametersButton";
import RawDataGraph from "./RawDataGraph";
import { DspResultCard, LabelCard, PerformanceCards, RawFeaturesStrip, RawPreview } from "./resultCards";
import { useSampleSelection } from "./selectors";

export type BlockType = "raw" | "spectral_analysis" | "flatten";

/** Shared Parameters-tab shell (§10.2 `DspBlockLayout`) — the impulse/catalog
 *  loading, sample selection, debounced `/dsp/preview` call and save flow
 *  used identically by all three motion DSP blocks (§10.4's page skeleton).
 *  Each block's `page.tsx` supplies only what actually differs: the schema
 *  section grouping, which options/fields are known-unsupported, and any
 *  block-specific DSP-result graphs. */
export default function DspBlockLayout({
  blockType,
  title,
  icon,
  sectionOf,
  excludedOptions,
  degradedFields,
  minSelected,
  fieldLabels,
  fieldOrder,
  optionLabels,
  optionOrder,
  hiddenFields,
  sectionOrder,
  fieldDescriptions,
  checkboxFields,
  variant,
  resultExtras,
  parametersHeaderExtra,
}: {
  blockType: BlockType;
  title: string;
  icon: ReactNode;
  sectionOf?: (key: string) => string;
  excludedOptions?: Record<string, string[]>;
  degradedFields?: DegradedField[];
  minSelected?: Record<string, number>;
  fieldLabels?: Record<string, string>;
  fieldOrder?: string[];
  optionLabels?: Record<string, Record<string, string>>;
  optionOrder?: Record<string, string[]>;
  hiddenFields?: string[];
  sectionOrder?: string[];
  fieldDescriptions?: Record<string, string>;
  checkboxFields?: string[];
  variant?: "stacked" | "row";
  resultExtras?: ReactNode;
  /** Rendered at the right of the Parameters card header — e.g. Spectral
   *  Analysis's `Autotune parameters` button (§10.6). */
  parametersHeaderExtra?: ReactNode;
}) {
  const searchParams = useSearchParams();
  const router = useRouter();
  const impulseId = searchParams.get("impulseId");
  const needsSave = searchParams.get("needsSave") === "1";
  const { activeImpulse, savedActiveImpulse, setActiveImpulse, setSavedActiveImpulse } = useAppStore();

  useEffect(() => {
    if (needsSave) toast("Save your parameters before generating features.", { icon: "ℹ️" });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const currentImpulse = useMemo(
    () => (impulseId ? (activeImpulse?.id === impulseId ? activeImpulse : null) : activeImpulse),
    [activeImpulse, impulseId]
  );

  const parametersSaved = useMemo(() => {
    if (!currentImpulse || savedActiveImpulse?.id !== currentImpulse.id) return false;
    return !!savedActiveImpulse?.dsp_params_saved_at;
  }, [currentImpulse, savedActiveImpulse]);

  useEffect(() => {
    if (!impulseId || activeImpulse?.id === impulseId) return;
    let cancelled = false;
    impulsesApi.get(impulseId).then(({ data }) => {
      if (!cancelled) {
        setActiveImpulse(data);
        setSavedActiveImpulse(data);
      }
    }).catch(() => toast.error("Failed to load the selected impulse"));
    return () => { cancelled = true; };
  }, [impulseId, activeImpulse?.id, setActiveImpulse, setSavedActiveImpulse]);

  const blockIndex = useMemo(
    () => (currentImpulse?.dsp_blocks || []).findIndex((b: any) => b.type === blockType),
    [currentImpulse, blockType]
  );

  // Live catalog — schema-driven form source of truth (R6/R7). No field is
  // ever hard-coded; whatever this returns is what renders.
  const [schema, setSchema] = useState<ParamSchema | null>(null);
  useEffect(() => {
    if (!currentImpulse?.id) return;
    let cancelled = false;
    dspApi.processingBlocks(currentImpulse.id, false, "motion")
      .then(({ data }) => {
        if (cancelled) return;
        const entry = (data?.blocks || []).find((b: any) => b.type === blockType);
        setSchema(entry?.params || {});
      })
      .catch(() => { if (!cancelled) setSchema({}); });
    return () => { cancelled = true; };
  }, [currentImpulse?.id, blockType]);

  const [params, setParams] = useState<Record<string, any>>({});
  useEffect(() => {
    if (blockIndex < 0 || !currentImpulse) return;
    setParams(currentImpulse.dsp_blocks[blockIndex]?.params || {});
  }, [currentImpulse, blockIndex]);

  const selection = useSampleSelection(currentImpulse?.project_id);

  const [preview, setPreview] = useState<number[] | null>(null);
  const [previewLoading, setPreviewLoading] = useState(false);
  const [previewLabel, setPreviewLabel] = useState<string | null>(null);
  const [rawPreview, setRawPreview] = useState<RawPreview | null>(null);
  const [previewError, setPreviewError] = useState<string | null>(null);

  useEffect(() => {
    if (!currentImpulse?.id || !selection.selectedSample || blockIndex < 0) {
      setPreview(null);
      setRawPreview(null);
      setPreviewError(null);
      return;
    }
    const timeout = setTimeout(() => {
      setPreviewLoading(true);
      dspApi.preview({
        sample_id: selection.selectedSample,
        impulse_id: currentImpulse.id,
        blocks: [{ type: blockType, params }],
      })
        .then(({ data }) => {
          setPreview(Array.isArray(data?.features) ? data.features : null);
          setPreviewLabel(data?.label ?? null);
          setRawPreview(data?.raw_preview ?? null);
          setPreviewError(null);
        })
        .catch((e) => {
          setPreview(null);
          setRawPreview(null);
          setPreviewError(e?.response?.data?.detail || "Could not compute a preview for this recording");
        })
        .finally(() => setPreviewLoading(false));
    }, 400);
    return () => clearTimeout(timeout);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [currentImpulse?.id, selection.selectedSample, JSON.stringify(params), blockIndex]);

  const axisCount = useMemo(() => (currentImpulse?.input_axes || []).length, [currentImpulse]);

  const handleSave = async () => {
    if (!currentImpulse || blockIndex < 0) return;
    const updatedImpulse = { ...currentImpulse };
    updatedImpulse.dsp_blocks = updatedImpulse.dsp_blocks.map((blk: any, i: number) =>
      i === blockIndex ? { ...blk, params: { ...params } } : { ...blk }
    );
    try {
      const { data } = await impulsesApi.update(currentImpulse.id, updatedImpulse, { saveParameters: true });
      setActiveImpulse(data);
      setSavedActiveImpulse(data);
    } catch (e: any) {
      toast.error(e?.response?.data?.detail || e.message || "Failed to save parameters");
      throw e;
    }
  };

  if (!currentImpulse) {
    return <div className="p-6 text-gray-400">Loading impulse...</div>;
  }

  if (blockIndex < 0) {
    return (
      <div className="pe-dsp flex flex-col min-h-full">
        <DspTabs blockType={blockType} impulseId={impulseId} active="parameters" parametersSaved={parametersSaved} />
        <div className="p-6 text-sm" style={{ color: "var(--app-text-muted)" }}>
          This impulse has no {title} block configured.
        </div>
      </div>
    );
  }

  const qs = impulseId ? `?impulseId=${impulseId}` : "";

  return (
    <div className="pe-dsp motion-dsp-page flex flex-col min-h-full">
      <DspTabs blockType={blockType} impulseId={impulseId} active="parameters" parametersSaved={parametersSaved} />

      <div className="pe-dsp-body" style={{ display: "flex", flexDirection: "column", gap: 20, padding: "1.25rem" }}>
        <RawDataGraph
          selectedSample={selection.selectedSample}
          setSelectedSample={selection.setSelectedSample}
          selectedLabel={selection.selectedLabel}
          setSelectedLabel={selection.setSelectedLabel}
          samples={selection.samples}
          filteredSamples={selection.filteredSamples}
          availableLabels={selection.availableLabels}
          loadingSamples={selection.loading}
        />

        <div className="pe-dsp-grid">
          <div className="space-y-5">
            <RawFeaturesStrip rawPreview={rawPreview} loading={previewLoading} error={previewError} />
            <LabelCard label={previewLabel} />

            <div className="pe-dsp-card">
              <div className="pe-dsp-card-head" style={{ justifyContent: "space-between" }}>
                <div className="flex items-center gap-2">
                  {icon}
                  <h3 className="head-title">Parameters</h3>
                </div>
                {parametersHeaderExtra}
              </div>
              <div className="pe-dsp-card-body">
                {schema === null ? (
                  <p className="text-sm" style={{ color: "var(--app-text-muted)" }}>Loading parameter schema…</p>
                ) : (
                  <ParameterForm
                    schema={schema}
                    values={params}
                    onChange={(key, value) => setParams((p) => ({ ...p, [key]: value }))}
                    sectionOf={sectionOf}
                    excludedOptions={excludedOptions}
                    minSelected={minSelected}
                    degradedFields={degradedFields}
                    fieldLabels={fieldLabels}
                    fieldOrder={fieldOrder}
                    optionLabels={optionLabels}
                    optionOrder={optionOrder}
                    hiddenFields={hiddenFields}
                    sectionOrder={sectionOrder}
                    fieldDescriptions={fieldDescriptions}
                    checkboxFields={checkboxFields}
                    variant={variant}
                  />
                )}
                <SaveParametersButton
                  onSave={handleSave}
                  onSaved={() => toast.success("Parameters saved")}
                  onSavedAndGenerate={() => {
                    toast.success("Parameters saved");
                    router.push(`/dashboard/impulse/${blockType}/generate-features${qs}`);
                  }}
                  disabled={schema === null}
                />
              </div>
            </div>
          </div>

          <div className="space-y-5">
            <DspResultCard blockType={blockType} features={preview} loading={previewLoading} axisCount={axisCount}>
              {resultExtras}
            </DspResultCard>
            <PerformanceCards />
          </div>
        </div>
      </div>
    </div>
  );
}
