"use client";
import { useEffect, useMemo, useState } from "react";
import { samplesApi } from "@/utils/api";

/** Fetches the project's labeled training recordings (R29) and derives the
 *  distinct label list + label-filtered subset a Parameters page's preview
 *  selector needs. Shared by all three motion-dsp blocks so the fetch/filter
 *  logic exists exactly once. */
export function useSampleSelection(projectId: string | undefined | null) {
  const [samples, setSamples] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);
  const [selectedSample, setSelectedSample] = useState("");
  const [selectedLabel, setSelectedLabel] = useState("all");

  const fetchSamples = async () => {
    if (!projectId) return;
    setLoading(true);
    try {
      const { data } = await samplesApi.list(projectId, {
        labeled_only: true,
        sample_type: "training",
        limit: 10000,
      });
      const items = Array.isArray(data?.items) ? data.items : [];
      setSamples(items);
      setSelectedSample((current) =>
        items.some((s: any) => s.id === current) ? current : items[0]?.id || ""
      );
    } catch {
      setSamples([]);
      setSelectedSample("");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchSamples();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [projectId]);

  const availableLabels = useMemo(
    () => Array.from(new Set(samples.map((s) => s.label_name || "Unknown"))).sort(),
    [samples]
  );

  const filteredSamples = useMemo(
    () =>
      selectedLabel === "all"
        ? samples
        : samples.filter((s) => (s.label_name || "Unknown") === selectedLabel),
    [samples, selectedLabel]
  );

  useEffect(() => {
    if (filteredSamples.length === 0) return;
    if (!filteredSamples.some((s) => s.id === selectedSample)) {
      setSelectedSample(filteredSamples[0].id);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [filteredSamples]);

  return {
    samples,
    filteredSamples,
    availableLabels,
    loading,
    selectedSample,
    setSelectedSample,
    selectedLabel,
    setSelectedLabel,
    refetch: fetchSamples,
  };
}

export function LabelSelector({
  value,
  onChange,
  labels,
  disabled,
}: {
  value: string;
  onChange: (v: string) => void;
  labels: string[];
  disabled?: boolean;
}) {
  return (
    <div style={{ display: "flex", flexDirection: "column" }}>
      <label className="pe-dsp-field-label" htmlFor="motion-dsp-label-select">Show</label>
      <select
        id="motion-dsp-label-select"
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className="pe-dsp-preview-select"
        disabled={disabled || labels.length === 0}
      >
        <option value="all">All labels</option>
        {labels.map((label) => (
          <option key={label} value={label}>{label}</option>
        ))}
      </select>
    </div>
  );
}

export function SampleSelector({
  value,
  onChange,
  samples,
  loading,
}: {
  value: string;
  onChange: (v: string) => void;
  samples: any[];
  loading: boolean;
}) {
  return (
    <div style={{ display: "flex", flexDirection: "column", flex: 1, minWidth: 200 }}>
      <label className="pe-dsp-field-label" htmlFor="motion-dsp-sample-select">Preview sample</label>
      <select
        id="motion-dsp-sample-select"
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className="pe-dsp-preview-select"
        disabled={loading || samples.length === 0}
      >
        {loading && <option value="">Loading samples...</option>}
        {!loading && samples.length === 0 && <option value="">No samples for this label</option>}
        {samples.map((s) => (
          <option key={s.id} value={s.id}>{s.filename} ({s.label_name || "Unknown"})</option>
        ))}
      </select>
    </div>
  );
}
