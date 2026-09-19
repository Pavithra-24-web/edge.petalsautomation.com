"use client";
import { useState, useEffect, useMemo } from "react";
import { useSearchParams, useRouter } from "next/navigation";
import { useAppStore } from "@/store/appStore";
import { impulsesApi, samplesApi, dspApi } from "@/utils/api";
import Link from "next/link";
import toast from "react-hot-toast";
import { Image as ImageIcon, Box } from "lucide-react";
import PremiumSaveButton from "@/components/PremiumSaveButton";

// Mirrors the dataset preview's label palette so a label keeps a consistent
// color across pages.
const LABEL_PALETTE = [
  "#8b5cf6", "#6366f1", "#3b82f6", "#06b6d4",
  "#10b981", "#f59e0b", "#ef4444", "#ec4899",
  "#a855f7", "#14b8a6",
];
function colorForLabel(label: string): string {
  const key = String(label || "").toLowerCase();
  let h = 0;
  for (let i = 0; i < key.length; i++) h = (h * 31 + key.charCodeAt(i)) | 0;
  return LABEL_PALETTE[Math.abs(h) % LABEL_PALETTE.length];
}

export default function ParametersPage() {
  const searchParams = useSearchParams();
  const router = useRouter();
  const { activeImpulse, savedActiveImpulse, setActiveImpulse, setSavedActiveImpulse } = useAppStore();
  const impulseId = searchParams.get("impulseId");
  const needsSave = searchParams.get("needsSave") === "1";

  useEffect(() => {
    if (needsSave) {
      toast("Save your parameters before generating features.", { icon: "ℹ️" });
    }
    // Only fire on initial mount when the flag is present.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);
  const currentImpulse = useMemo(
    () => (impulseId ? (activeImpulse?.id === impulseId ? activeImpulse : null) : activeImpulse),
    [activeImpulse, impulseId]
  );
  // Parameters are "saved" only when the backend has stamped
  // `dsp_params_saved_at` for this impulse. Field presence is NOT sufficient:
  // `image_width` / `image_height` are populated by defaults at impulse
  // creation, so checking them would let users skip the explicit save step.
  const parametersSaved = useMemo(() => {
    if (!currentImpulse || savedActiveImpulse?.id !== currentImpulse.id) return false;
    return !!savedActiveImpulse?.dsp_params_saved_at;
  }, [currentImpulse, savedActiveImpulse]);
  const [params, setParams] = useState({
    image_width: 96,
    image_height: 96,
    grayscale: false,
    resize_mode: "Fit shortest axis",
  });
  const [saving, setSaving] = useState(false);
  const [samples, setSamples] = useState<any[]>([]);
  const [selectedSample, setSelectedSample] = useState("");
  const [selectedLabel, setSelectedLabel] = useState<string>("all");
  const [loadingSamples, setLoadingSamples] = useState(true);
  const [preview, setPreview] = useState<any>(null);
  const [loadingPreview, setLoadingPreview] = useState(false);
  // Original (pre-DSP) sample image + natural dimensions, fetched the same way
  // the /dashboard/data/dataset sample preview does it.
  const [rawImageUrl, setRawImageUrl] = useState<string | null>(null);
  const [rawImageNatural, setRawImageNatural] = useState<{ w: number; h: number } | null>(null);
  const isFomo =
    currentImpulse?.ml_blocks?.some(
      (block: any) => (block?.type ?? block?.architecture) === "fomo_mobilenetv2_0_1"
    ) ?? false;

  useEffect(() => {
    const needsImpulseLoad = impulseId && activeImpulse?.id !== impulseId;
    if (!needsImpulseLoad) return;

    let cancelled = false;

    const loadImpulse = async () => {
      try {
        const { data } = await impulsesApi.get(impulseId);
        if (!cancelled) {
          setActiveImpulse(data);
          setSavedActiveImpulse(data);
        }
      } catch (e) {
        if (!cancelled) {
          toast.error("Failed to load the selected impulse");
        }
      }
    };

    loadImpulse();

    return () => {
      cancelled = true;
    };
  }, [impulseId, activeImpulse?.id, setActiveImpulse, setSavedActiveImpulse]);

  useEffect(() => {
    if (!currentImpulse) return;
    const blockParams =
      currentImpulse.dsp_blocks?.length > 0
        ? currentImpulse.dsp_blocks[0].params || {}
        : {};
    // Dimension priority: block params → impulse root fields → hard default
    setParams({
      image_width:  Math.max(blockParams.image_width  ?? currentImpulse.image_width  ?? 96, isFomo ? 96 : 1),
      image_height: Math.max(blockParams.image_height ?? currentImpulse.image_height ?? 96, isFomo ? 96 : 1),
      // grayscale default matches backend DSP catalog default (false = RGB)
      grayscale:    blockParams.grayscale    ?? false,
      // resize_mode lives on the impulse root; block params take precedence once saved
      resize_mode:  blockParams.resize_mode  ?? currentImpulse.resize_mode  ?? "Fit shortest axis",
    });
  }, [currentImpulse, isFomo]);

  const fetchSamples = async () => {
    if (!currentImpulse?.project_id) return;
    setLoadingSamples(true);
    try {
      const res = await samplesApi.list(currentImpulse.project_id, { labeled_only: true, sample_type: "training", limit: 10000 });
      const data = res.data;
      if (!data || !Array.isArray(data.items)) {
        console.error("Invalid samples response structure", data);
        return;
      }
      const hasBoundingBoxes = data.items.some((s: any) => s.extra_metadata?.boundingBoxes?.length > 0);
      let refinedItems = data.items;
      
      // If the project possesses object detection annotations natively, strictly filter out folder-inferred blanks
      if (hasBoundingBoxes) {
         refinedItems = data.items.filter((s: any) => s.extra_metadata?.boundingBoxes?.length > 0);
         console.log(`[Samples Dataset] Constrained ${data.items.length} items purely to ${refinedItems.length} explicitly annotated ones`);
      } else {
         console.log(`[Samples Dataset] Fetched ${data.items.length} labeled items.`);
      }

      if (refinedItems.length > 0) {
        setSamples(refinedItems);
        setSelectedSample(refinedItems[0].id);
        console.log(`[Samples Dataset] Auto-selected prime labeled sample:`, refinedItems[0].id);
      } else {
        setSamples([]);
        setSelectedSample("");
        console.log(`[Samples Dataset] No labeled dataset items found!`);
      }
    } catch (e) {
      console.error("Exception requesting samples:", e);
    } finally {
      setLoadingSamples(false);
    }
  };

  useEffect(() => {
    fetchSamples();
  }, [currentImpulse?.project_id]);

  useEffect(() => {
    if (currentImpulse && selectedSample) {
      const timeout = setTimeout(() => fetchPreview(), 400);
      return () => clearTimeout(timeout);
    }
  }, [selectedSample, params, currentImpulse]);

  const availableLabels = useMemo(
    () => Array.from(new Set(samples.map(s => s.label_name || "Unknown"))).sort(),
    [samples]
  );

  const filteredSamples = useMemo(
    () =>
      selectedLabel === "all"
        ? samples
        : samples.filter(s => (s.label_name || "Unknown") === selectedLabel),
    [samples, selectedLabel]
  );

  useEffect(() => {
    if (filteredSamples.length === 0) return;
    if (!filteredSamples.some(s => s.id === selectedSample)) {
      setSelectedSample(filteredSamples[0].id);
    }
  }, [filteredSamples, selectedSample]);

  // Load the original sample image for the Raw data card.
  useEffect(() => {
    if (!selectedSample) {
      setRawImageUrl(null);
      setRawImageNatural(null);
      return;
    }
    let cancelled = false;
    setRawImageNatural(null);
    samplesApi.download(selectedSample)
      .then(({ data }) => { if (!cancelled) setRawImageUrl(data.url); })
      .catch(() => { if (!cancelled) setRawImageUrl(null); });
    return () => { cancelled = true; };
  }, [selectedSample]);

  const fetchPreview = async () => {
    if (!currentImpulse || !selectedSample) return;
    setLoadingPreview(true);
    console.log(`[Preview] Triggering extraction preview for sample_id: ${selectedSample}`);
    try {
      const payload = {
        impulse_id: currentImpulse.id,
        sample_id: selectedSample,
        blocks: [{
          type: "image",
          params: {
            ...params,
            image_width: Number(params.image_width),
            image_height: Number(params.image_height)
          }
        }]
      };
      console.log("[Preview] Request Payload:", payload);
      const res = await dspApi.preview(payload);
      console.log(`[Preview] Successfully rendered DSP view with shape`, res.data.shape);
      setPreview(res.data);
    } catch (e: any) {
      if (e.response?.status === 422) {
        console.error("422 Validation Error Payload:", e.response.data);
        toast.error("API Validation Failed (422) - Check payload format");
      } else {
        console.error(`[Preview] Network exception:`, e);
      }
    } finally {
      setLoadingPreview(false);
    }
  };

  const handleSave = async () => {
    if (!currentImpulse) return;
    setSaving(true);
    try {
      const updatedImpulse = { ...currentImpulse };
      if (!updatedImpulse.dsp_blocks || updatedImpulse.dsp_blocks.length === 0) {
        updatedImpulse.dsp_blocks = [{ "type": "image", "params": {} }];
      }
      // Deep-copy to avoid mutating activeImpulse in store
      updatedImpulse.dsp_blocks = updatedImpulse.dsp_blocks.map((blk: any, i: number) =>
        i === 0 ? { ...blk, params: { ...params } } : { ...blk }
      );

      // Keep root-level image fields in sync with DSP block params so the
      // training worker, /dsp/input-size, and GET impulse all see the same values.
      updatedImpulse.image_width  = Number(params.image_width);
      updatedImpulse.image_height = Number(params.image_height);
      updatedImpulse.resize_mode  = params.resize_mode;

      // Signal an explicit Parameters save so the backend stamps
      // `dsp_params_saved_at`. Without this flag the PUT is treated as a
      // generic impulse update and the Generate features gate stays locked.
      const { data } = await impulsesApi.update(currentImpulse.id, updatedImpulse, {
        saveParameters: true,
      });
      setActiveImpulse(data);
      setSavedActiveImpulse(data);

      // Guarantee valid state sync
      await fetchSamples();
    } catch (e: any) {
      toast.error(e.message || "Failed to save parameters");
      throw e;
    } finally {
      setSaving(false);
    }
  };

  if (!currentImpulse) {
    return <div className="p-6 text-gray-400">Loading impulse...</div>;
  }


  return (
    <div className="pe-dsp image-dsp-page image-dsp-parameters-page flex flex-col min-h-full">
      {/* Tabs */}
      <div className="pe-dsp-tabs">
        <Link
          href={`/dashboard/impulse/image/parameters${impulseId ? `?impulseId=${impulseId}` : ""}`}
          className="pe-dsp-tab is-active"
        >
          Pre-processing
        </Link>
        {parametersSaved ? (
          <Link
            href={`/dashboard/impulse/image/generate-features${impulseId ? `?impulseId=${impulseId}` : ""}`}
            className="pe-dsp-tab"
          >
            Generate features
          </Link>
        ) : (
          <button
            type="button"
            disabled
            aria-disabled="true"
            title="Save parameters before generating features."
            className="pe-dsp-tab is-disabled"
            style={{ opacity: 0.5, cursor: "not-allowed" }}
            onClick={(e) => e.preventDefault()}
          >
            Generate features
          </button>
        )}
      </div>

      <div className="pe-dsp-body">
        {/* Left sidebar */}
        <aside className="pe-dsp-sidebar">
          <div className="pe-dsp-side-card">
            <h2 className="pe-dsp-side-title">Image DSP Parameters</h2>

            <div className="pe-dsp-field">
              <label className="pe-dsp-field-label" htmlFor="dsp-image-width">Image width</label>
              <input
                id="dsp-image-width"
                type="number"
                min={isFomo ? 96 : 1}
                value={params.image_width}
                onChange={e => setParams({ ...params, image_width: Math.max(parseInt(e.target.value) || 0, isFomo ? 96 : 1) })}
                className="pe-dsp-input"
              />
            </div>

            <div className="pe-dsp-field">
              <label className="pe-dsp-field-label" htmlFor="dsp-image-height">Image height</label>
              <input
                id="dsp-image-height"
                type="number"
                min={isFomo ? 96 : 1}
                value={params.image_height}
                onChange={e => setParams({ ...params, image_height: Math.max(parseInt(e.target.value) || 0, isFomo ? 96 : 1) })}
                className="pe-dsp-input"
              />
            </div>

            <div className="pe-dsp-field">
              <label className="pe-dsp-field-label" htmlFor="dsp-color-channels">Color channels</label>
              <select
                id="dsp-color-channels"
                value={params.grayscale ? "Grayscale" : "RGB"}
                onChange={e => setParams({ ...params, grayscale: e.target.value === "Grayscale" })}
                className="pe-dsp-select"
              >
                <option>RGB</option>
                <option>Grayscale</option>
              </select>
            </div>

            <div className="pe-dsp-field">
              <label className="pe-dsp-field-label" htmlFor="dsp-resize-mode">Resize mode</label>
              <select
                id="dsp-resize-mode"
                value={params.resize_mode}
                onChange={e => setParams({ ...params, resize_mode: e.target.value })}
                className="pe-dsp-select"
              >
                <option value="Squash">Squash</option>
                <option value="Fit shortest axis">Fit shortest axis</option>
                <option value="Fit longest axis">Fit longest axis</option>
              </select>
            </div>
          </div>

          <PremiumSaveButton
            onSave={handleSave}
            onSuccess={() => {
              toast.success("Parameters saved");
              const qs = impulseId ? `?impulseId=${impulseId}` : "";
              router.push(`/dashboard/impulse/image/generate-features${qs}`);
            }}
            label="Save parameters"
            loadingLabel="Saving..."
            disabled={saving}
            fullWidth
          />
        </aside>

        {/* Right workspace */}
        <section className="pe-dsp-workspace">
          {/* Preview sample selector */}
          <div className="pe-dsp-control-row">
            <div className="field" style={{ display: "flex", alignItems: "flex-end", gap: 12, flexWrap: "wrap" }}>
              <div style={{ display: "flex", flexDirection: "column" }}>
                <label className="pe-dsp-field-label" htmlFor="dsp-preview-label">Show</label>
                <select
                  id="dsp-preview-label"
                  value={selectedLabel}
                  onChange={e => setSelectedLabel(e.target.value)}
                  className="pe-dsp-preview-select"
                  disabled={loadingSamples || availableLabels.length === 0}
                >
                  <option value="all">All labels</option>
                  {availableLabels.map(label => (
                    <option key={label} value={label}>{label}</option>
                  ))}
                </select>
              </div>
              <div style={{ display: "flex", flexDirection: "column", flex: 1, minWidth: 200 }}>
                <label className="pe-dsp-field-label" htmlFor="dsp-preview-sample">Preview sample</label>
                <select
                  id="dsp-preview-sample"
                  value={selectedSample}
                  onChange={e => setSelectedSample(e.target.value)}
                  className="pe-dsp-preview-select"
                  disabled={loadingSamples || filteredSamples.length === 0}
                >
                  {loadingSamples && <option value="">Loading samples...</option>}
                  {!loadingSamples && filteredSamples.length === 0 && <option value="">No samples for this label</option>}
                  {filteredSamples.map(s => (
                    <option key={s.id} value={s.id}>{s.filename} ({s.label_name || "Unknown"})</option>
                  ))}
                </select>
              </div>
            </div>
          </div>

          {(() => {
            const currentSampleData = samples.find(s => s.id === selectedSample);
            const boxes = currentSampleData?.extra_metadata?.boundingBoxes || [];

            if (loadingPreview) {
              return <div className="text-sm font-medium" style={{ color: "var(--app-text-muted)" }}>Fetching preview...</div>;
            }
            if (!loadingSamples && samples.length === 0) {
              return (
                <div className="w-full rounded-2xl flex items-center justify-center"
                  style={{ minHeight: 240, border: "2px dashed var(--app-border)", background: "var(--app-surface)" }}>
                  <p className="text-sm font-medium" style={{ color: "var(--app-text-muted)" }}>
                    No samples available in your training set
                  </p>
                </div>
              );
            }
            if (!preview) {
              return (
                <div className="w-full rounded-2xl flex items-center justify-center"
                  style={{ minHeight: 240, border: "2px dashed var(--app-border)", background: "var(--app-surface)" }}>
                  <p className="text-sm font-medium" style={{ color: "var(--app-text-muted)" }}>
                    Select a representation to generate preview
                  </p>
                </div>
              );
            }

            return (
              <div className="pe-dsp-grid">
                {/* Left column */}
                <div className="space-y-5">
                  {/* Raw data — original labelled sample, mirrors the dataset preview */}
                  <div className="pe-dsp-card">
                    <div className="pe-dsp-card-head">
                      <span className="head-icon icon-blue"><ImageIcon size={14} /></span>
                      <h3 className="head-title">Raw data</h3>
                    </div>
                    <div className="pe-dsp-card-body">
                      <div className="pe-dsp-img-frame">
                        {rawImageUrl ? (
                          <div
                            className="relative inline-block"
                            style={{
                              maxWidth: "100%",
                              maxHeight: 280,
                              aspectRatio: rawImageNatural ? `${rawImageNatural.w} / ${rawImageNatural.h}` : undefined,
                            }}
                          >
                            <img
                              src={rawImageUrl}
                              alt="Raw sample"
                              style={{ width: "100%", height: "100%", maxHeight: "none", objectFit: "contain", display: "block" }}
                              onLoad={(e) => {
                                const img = e.currentTarget;
                                setRawImageNatural({ w: img.naturalWidth, h: img.naturalHeight });
                              }}
                            />
                            {rawImageNatural && boxes.map((box: any, i: number) => {
                              const labelText = box.label || currentSampleData?.label_name || "Unknown";
                              const color = colorForLabel(labelText);
                              const bw = box.w ?? box.width ?? 0;
                              const bh = box.h ?? box.height ?? 0;
                              return (
                                <div
                                  key={i}
                                  className="absolute pointer-events-none"
                                  style={{
                                    left: `${(box.x / rawImageNatural.w) * 100}%`,
                                    top: `${(box.y / rawImageNatural.h) * 100}%`,
                                    width: `${(bw / rawImageNatural.w) * 100}%`,
                                    height: `${(bh / rawImageNatural.h) * 100}%`,
                                    border: `2px solid ${color}`,
                                    background: `${color}1a`,
                                  }}
                                >
                                  <div
                                    className="absolute -top-4 left-[-2px] text-white text-[9px] font-bold px-1 rounded-t whitespace-nowrap truncate max-w-full"
                                    style={{ background: color }}
                                  >
                                    {labelText}
                                  </div>
                                </div>
                              );
                            })}
                          </div>
                        ) : (
                          <span className="text-xs" style={{ color: "var(--app-text-muted)" }}>
                            Loading sample...
                          </span>
                        )}
                      </div>
                    </div>
                  </div>

                  {/* Raw features */}
                  <div className="pe-dsp-card">
                    <div className="pe-dsp-card-head">
                      <h3 className="head-title">Raw features</h3>
                    </div>
                    <div className="pe-dsp-raw-block">
                      [{preview.raw_features ? preview.raw_features.slice(0, 30).map((n: number) => n.toFixed(4)).join(", ") : ""} ...]
                    </div>
                  </div>
                </div>

                {/* Right column */}
                <div className="space-y-5">
                  {/* DSP result */}
                  <div className="pe-dsp-card">
                    <div className="pe-dsp-card-head">
                      <span className="head-icon icon-purple"><Box size={14} /></span>
                      <h3 className="head-title">DSP result</h3>
                    </div>
                    <div className="pe-dsp-card-body">
                      <div className="pe-dsp-img-frame">
                        {preview.processed_image ? (
                          <img
                            src={preview.processed_image}
                            alt="DSP Result"
                            style={{ imageRendering: "pixelated" }}
                          />
                        ) : (
                          <span className="text-xs" style={{ color: "var(--app-text-muted)" }}>
                            No DSP representation
                          </span>
                        )}
                      </div>
                      <p className="pe-dsp-caption">
                        Processed features {preview.shape.join(" × ")}
                      </p>
                      <p className="pe-dsp-code">
                        {(() => {
                          const getFlatSubset = (data: any, limit = 15): number[] => {
                            if (!data) return [];
                            if (typeof data === 'number') return [data];
                            const result: number[] = [];
                            const stack = [data];
                            while (stack.length > 0 && result.length < limit) {
                              const item = stack.pop();
                              if (typeof item === 'number') result.push(item);
                              else if (Array.isArray(item)) {
                                for (let i = item.length - 1; i >= 0; i--) stack.push(item[i]);
                              }
                            }
                            return result;
                          };
                          const subset = getFlatSubset(preview.features);
                          return `[${subset.map(n => n.toFixed(4)).join(", ")}${subset.length >= 15 ? " ..." : ""}]`;
                        })()}
                      </p>
                    </div>
                  </div>

                </div>
              </div>
            );
          })()}
        </section>
      </div>
    </div>
  );
}
