"use client";
import { Activity } from "lucide-react";
import DspBlockLayout from "@/components/dashboard/motion-dsp/DspBlockLayout";

// The reference screenshot (`Rawdata_parameter.png`) shows exactly one
// control on this page — `Scale axes` — under a single "Scaling" section.
// `normalize`, `normalize_per_axis` and `flatten` are live in the backend
// catalog but are Petal-Edge-internal knobs the Edge-Impulse-style UI never
// exposes; they keep whatever value is already saved (or the backend's own
// default) but are hidden from this form rather than rendered as extra
// controls the reference doesn't show. No degraded/placeholder row is shown
// either, for the same reason — the reference has none.
function sectionOf(key: string): string {
  if (key === "scale_axes") return "Scaling";
  return "Parameters";
}

const FIELD_LABELS: Record<string, string> = {
  scale_axes: "Scale axes",
};

const HIDDEN_FIELDS = ["normalize", "normalize_per_axis", "flatten"];

export default function RawParametersPage() {
  return (
    <DspBlockLayout
      blockType="raw"
      title="Raw Data"
      icon={<span className="head-icon icon-blue"><Activity size={14} /></span>}
      sectionOf={sectionOf}
      fieldLabels={FIELD_LABELS}
      hiddenFields={HIDDEN_FIELDS}
    />
  );
}
