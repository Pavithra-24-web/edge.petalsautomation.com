"use client";
import { Layers } from "lucide-react";
import DspBlockLayout from "@/components/dashboard/motion-dsp/DspBlockLayout";
import type { DegradedField } from "@/components/dashboard/motion-dsp/ParameterForm";
import { BlockStateCard } from "@/components/dashboard/motion-dsp/blockPlaceholders";

// `moving_average` is documented (§10.7) but absent from the live Flatten
// catalog (backend gap G22). Rendered disabled per honest-degradation (§12.6).
const DEGRADED: DegradedField[] = [
  {
    key: "moving_average",
    section: "Method",
    label: "Moving Average",
    note: "Moving-average smoothing is not available in the current backend catalog (backend gap G22).",
  },
];

// Section mapping — the live Flatten catalog declares `scale_axes` and
// `features` (a list of statistic keys), not `method`/`combination`.
function sectionOf(key: string): string {
  if (key === "features") return "Method";
  if (key === "scale_axes") return "Scaling";
  return "Parameters";
}

const FIELD_LABELS: Record<string, string> = {
  scale_axes: "Scale axes",
  features: "Method",
};

// Display names for the `features` list — Edge Impulse shows these, not the
// backend's own option codes (backend gap G23; cosmetic-only mapping).
const OPTION_LABELS: Record<string, Record<string, string>> = {
  features: {
    mean: "Average",
    std: "Standard deviation",
    rms: "Root-mean square",
    max: "Maximum",
    min: "Minimum",
    skewness: "Skewness",
    kurtosis: "Kurtosis",
  },
};

// Section reading order: Scaling, then Method (the `features` checkbox list
// followed by the degraded `Moving Average` field).
const FIELD_ORDER = ["scale_axes", "features", "moving_average"];

// Checkbox order within "Method" — Average, Minimum, Maximum, Root-mean
// square, Standard deviation, Skewness, Kurtosis — matching the reference
// screenshot exactly, not the catalog's own `mean/std/rms/max/min/…` order.
const OPTION_ORDER: Record<string, string[]> = {
  features: ["mean", "min", "max", "rms", "std", "skewness", "kurtosis"],
};

// Flatten `features` is a list; at least one statistic must remain selected.
const MIN_SELECTED: Record<string, number> = {
  features: 1,
};

export default function FlattenParametersPage() {
  return (
    <DspBlockLayout
      blockType="flatten"
      title="Flatten"
      icon={<span className="head-icon icon-blue"><Layers size={14} /></span>}
      sectionOf={sectionOf}
      degradedFields={DEGRADED}
      minSelected={MIN_SELECTED}
      fieldLabels={FIELD_LABELS}
      fieldOrder={FIELD_ORDER}
      optionLabels={OPTION_LABELS}
      optionOrder={OPTION_ORDER}
      resultExtras={<BlockStateCard />}
    />
  );
}
