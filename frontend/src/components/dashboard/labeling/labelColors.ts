import type { CSSProperties } from "react";
import { PREMIUM_SOFT_PALETTE } from "./LabelingShared";

// Build a stable label → color map using evenly-spaced HSL hues.
// Labels are sorted alphabetically (excluding "Unlabeled") so the same label
// always gets the same index — and thus the same hue — regardless of count
// changes or render order. This guarantees unique colors for every label and
// consistent colors across pie slices, legend dots, pills, and bbox overlays.
export function buildLabelColorMap(labelCounts: Record<string, number>): Map<string, string> {
  const sorted = Object.keys(labelCounts)
    .filter(l => l !== "Unlabeled")
    .sort((a, b) => a.localeCompare(b));
  const m = new Map<string, string>();
  sorted.forEach((label, i) => {
    // Cycle through the palette — each label gets a distinct slot; for
    // >24 labels the palette repeats but hues are still well-spread.
    m.set(label.toLowerCase(), PREMIUM_SOFT_PALETTE[i % PREMIUM_SOFT_PALETTE.length]);
  });
  return m;
}

export function colorForLabelFromMap(labelColorMap: Map<string, string>, label: string): string {
  if (String(label || "").toLowerCase() === "unlabeled") return "rgba(148,163,184,0.55)";
  return labelColorMap.get(String(label || "").toLowerCase()) ?? PREMIUM_SOFT_PALETTE[0];
}

export function pillStyleFromMap(labelColorMap: Map<string, string>, label: string): CSSProperties {
  // Strip any alpha already on the palette hex (e.g. "#2c469cff") down to
  // the 6-digit form before appending our pill alpha — otherwise "${c}1f"
  // produces "#xxxxxxff1f", which is invalid CSS and silently drops the
  // background/border, making the pill render as plain text.
  const raw = colorForLabelFromMap(labelColorMap, label);
  const c = /^#[0-9a-fA-F]{8}$/.test(raw) ? raw.slice(0, 7) : raw;
  return { background: `${c}1f`, color: c, borderColor: `${c}55` };
}
