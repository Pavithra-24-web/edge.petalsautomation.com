"use client";
import { BarChart2 } from "lucide-react";
import DspBlockLayout from "@/components/dashboard/motion-dsp/DspBlockLayout";
import type { DegradedField } from "@/components/dashboard/motion-dsp/ParameterForm";
import {
  FilterResponseGraph,
  AfterFilterGraph,
  WaveletFunctionGraph,
  WaveletApproximationGraph,
  AutotuneButton,
} from "@/components/dashboard/motion-dsp/blockPlaceholders";

// input_decimation_ratio, analysis_type, and take_log are real,
// backend-validated catalog fields (processor.py's _spectral_analysis).
//
// improve_low_frequency_resolution stays degraded/disabled on purpose: a
// prototype zero-padding + power-law re-binning implementation existed
// backend-side but was removed after review, because its output was never
// verified against Edge Impulse's own low-frequency-resolution behavior, and
// no authoritative reference for that behavior is available. Rendering it as
// a working toggle would silently change trained models' features in a way
// nothing here can vouch for — so it's shown disabled instead, matching the
// honest-degradation rule (§12.6), until a verified algorithm exists.
//
// "Order" (filter_order, backend gap G15) is intentionally excluded here —
// the reference screenshot for this block does not show it, so it is
// omitted entirely rather than rendered disabled.
const DEGRADED: DegradedField[] = [
  {
    key: "improve_low_frequency_resolution",
    section: "Analysis",
    label: "Improve low frequency resolution?",
    note: "An enhanced low-frequency FFT approach was prototyped but its output could not be verified against Edge Impulse's reference behavior, so it remains disabled pending verification.",
    controlType: "checkbox",
    value: false,
  },
];

// Section mapping — matches the reference layout's "Filter" and "Analysis"
// cards. Keyed on the real catalog field names, not assumed ones.
// noise_floor_db and filter_cutoff still route to a section (harmless — see
// HIDDEN below) so the mapping stays complete if they're ever un-hidden.
function sectionOf(key: string): string {
  if (["filter_type", "filter_cutoff", "scale_axes", "input_decimation_ratio"].includes(key)) return "Filter";
  if (["fft_length", "overlap", "noise_floor_db", "analysis_type", "take_log"].includes(key)) return "Analysis";
  return "Parameters";
}

// Live catalog fields the reference screenshot for this block never shows.
// Hidden rather than removed from the backend catalog — the saved/default
// value is untouched, only the form control is omitted (backend gap: none,
// this is a UI-only parity fix, not a functional change).
const HIDDEN_FIELDS = ["noise_floor_db", "filter_cutoff"];

// Sections render Filter-then-Analysis (reference order) regardless of the
// backend catalog's own key order (fft_length precedes filter_type there,
// which would otherwise put "Analysis" first).
const SECTION_ORDER = ["Filter", "Analysis"];

// Exact Edge Impulse control labels for the fields the backend does expose,
// so live fields read the same as their degraded neighbours in the same
// section instead of a raw title-cased key name.
const FIELD_LABELS: Record<string, string> = {
  scale_axes: "Scale axes",
  input_decimation_ratio: "Input decimation ratio",
  filter_type: "Type",
  analysis_type: "Type",
  fft_length: "FFT length",
  take_log: "Take log of spectrum?",
  overlap: "Overlap FFT frames?",
};

// The backend catalog carries its own `description` for input_decimation_ratio,
// analysis_type, and take_log (processor.py's validated params), so no
// override is needed here for those three — only for the fields the catalog
// leaves undocumented.
const FIELD_DESCRIPTIONS: Record<string, string> = {
  scale_axes: "Multiplies the axes by this number. Useful to scale unevenly-sampled axes into a similar range.",
  filter_type: "Optional low-pass or high-pass filter applied to the signal before spectral analysis.",
  fft_length: "The number of FFT points. Must be at least twice the number of samples in the window, typically a power of 2.",
  overlap: "When enabled, consecutive FFT frames overlap by 50%. When disabled, frames do not overlap.",
};

// `overlap` is a 0.0–0.95 ratio on the backend, but the reference UI shows a
// plain yes/no toggle — checked writes the catalog default (0.5, i.e. 50%
// overlap), unchecked writes 0 (no overlap). Same catalog field, same valid
// range; only the control presented to the user changes.
const CHECKBOX_FIELDS = ["overlap"];

// Reading order within each section, reproducing the reference screenshot's
// "Filter" (Scale axes, Input decimation ratio, Type) and "Analysis" (Type,
// FFT length, Take log of spectrum, Overlap FFT frames, Improve
// low-frequency resolution) grouping.
const FIELD_ORDER = [
  "scale_axes",
  "input_decimation_ratio",
  "filter_type",
  "analysis_type",
  "fft_length",
  "take_log",
  "overlap",
  "improve_low_frequency_resolution",
];

// `bandpass` is excluded from the filter_type dropdown per §10.4
// (unsupported in the live backend catalog — backend gap G16).
const EXCLUDED_OPTIONS: Record<string, string[]> = {
  filter_type: ["bandpass"],
};

export default function SpectralAnalysisParametersPage() {
  return (
    <DspBlockLayout
      blockType="spectral_analysis"
      title="Spectral Analysis"
      icon={<span className="head-icon icon-blue"><BarChart2 size={14} /></span>}
      sectionOf={sectionOf}
      excludedOptions={EXCLUDED_OPTIONS}
      degradedFields={DEGRADED}
      hiddenFields={HIDDEN_FIELDS}
      sectionOrder={SECTION_ORDER}
      fieldLabels={FIELD_LABELS}
      fieldDescriptions={FIELD_DESCRIPTIONS}
      checkboxFields={CHECKBOX_FIELDS}
      variant="row"
      fieldOrder={FIELD_ORDER}
      parametersHeaderExtra={<AutotuneButton />}
      resultExtras={
        <>
          <FilterResponseGraph />
          <AfterFilterGraph />
          <WaveletFunctionGraph />
          <WaveletApproximationGraph />
        </>
      }
    />
  );
}
