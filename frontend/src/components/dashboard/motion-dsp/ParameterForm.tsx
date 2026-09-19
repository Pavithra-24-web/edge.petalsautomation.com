"use client";
import type { ReactNode } from "react";
import { HelpTooltip } from "./common";

export type ParamDef = {
  type: "int" | "float" | "bool" | "enum" | "list";
  default?: any;
  min?: number;
  max?: number;
  options?: string[];
  description?: string;
};
export type ParamSchema = Record<string, ParamDef>;

export type DegradedField = {
  key: string;
  section: string;
  label: string;
  note: string;
  /** How the disabled control should look — defaults to "select" (the
   *  original generic "Not yet available" dropdown). "checkbox" renders a
   *  disabled checkbox instead, for fields that are conceptually boolean
   *  (e.g. "Take log of spectrum?"). */
  controlType?: "select" | "checkbox";
  /** The value the disabled control displays. For "select" this is the
   *  option text shown in place of "Not yet available" (e.g. the backend's
   *  fixed value); for "checkbox" this is the checked state. Omit to fall
   *  back to the original "Not yet available" placeholder. */
  value?: string | boolean;
};

function titleCase(key: string): string {
  return key
    .split("_")
    .map((w) => (w.length ? w[0].toUpperCase() + w.slice(1) : w))
    .join(" ");
}

/** Schema-driven parameter form (§10.2 `ParameterForm`). Renders exactly the
 *  fields the live block catalog (`GET /dsp/processing-blocks`) declares —
 *  nothing is hard-coded — grouped into caller-supplied sections purely for
 *  layout. Unknown/future catalog keys fall back to a trailing "Parameters"
 *  group rather than being dropped, so a backend addition renders instead of
 *  silently disappearing. `degradedFields` renders known-missing controls in
 *  a visibly disabled state per the honest-degradation rule (§12.6) — they
 *  are never wired to `onChange`. */
export default function ParameterForm({
  schema,
  values,
  onChange,
  sectionOf,
  excludedOptions,
  minSelected,
  degradedFields = [],
  fieldLabels,
  fieldOrder,
  optionLabels,
  optionOrder,
  hiddenFields,
  sectionOrder,
  fieldDescriptions,
  checkboxFields,
  variant = "stacked",
}: {
  schema: ParamSchema;
  values: Record<string, any>;
  onChange: (key: string, value: any) => void;
  sectionOf?: (key: string) => string;
  excludedOptions?: Record<string, string[]>;
  minSelected?: Record<string, number>;
  degradedFields?: DegradedField[];
  /** Catalog keys to omit entirely — e.g. a live field the reference
   *  screenshot never shows. The value is still whatever was already saved
   *  (or the backend's own default when absent); this only removes it from
   *  the rendered form, it does not clear or override it. */
  hiddenFields?: string[];
  /** Overrides the auto-generated (title-cased) label for specific catalog keys,
   *  so a live field can carry its exact Edge Impulse label (e.g. `filter_cutoff`
   *  → "Cut-off frequency") without inventing a value the backend doesn't send. */
  fieldLabels?: Record<string, string>;
  /** Explicit read order for keys (live or degraded) within a section, matching
   *  the reference layout. Keys not listed keep their catalog/degraded-array order,
   *  appended after every listed key. */
  fieldOrder?: string[];
  /** Per-field display-name overrides for `enum`/`list` option values (e.g.
   *  `features.rms` → "Root-mean square"), keyed by catalog field then option.
   *  Cosmetic only (G23) — the stored value is still the backend's own code. */
  optionLabels?: Record<string, Record<string, string>>;
  /** Explicit display order for an `enum`/`list` field's options, keyed by
   *  catalog field name. Options not listed keep the catalog's own order,
   *  appended after every listed option. */
  optionOrder?: Record<string, string[]>;
  /** Explicit section render order (e.g. `["Filter", "Analysis"]`), matching
   *  the reference layout. Sections not listed keep the order they were first
   *  encountered in, appended after every listed section. Without this, order
   *  falls out of the catalog's own key order, which need not match a
   *  hand-designed section layout. */
  sectionOrder?: string[];
  /** Tooltip text for live catalog keys the backend doesn't send a
   *  `description` for, so a live field's "?" affordance can carry the same
   *  explanatory copy as its Edge Impulse counterpart without inventing
   *  backend metadata. Only used when the catalog entry has no `description`. */
  fieldDescriptions?: Record<string, string>;
  /** Live catalog keys to render as a checkbox instead of their schema type's
   *  default control — for a field that is numeric on the backend (e.g. an
   *  0–0.95 overlap ratio) but is a plain yes/no toggle in the Edge Impulse
   *  reference UI. Checked writes the field's schema `default`; unchecked
   *  writes `0`. Checked reflects any current non-zero/non-empty value, so an
   *  existing intermediate value (e.g. 0.3) still shows as checked until the
   *  user actually toggles it. */
  checkboxFields?: string[];
  /** "stacked" (default) keeps the original label-above-control layout used
   *  by every other DSP block page. "row" lays out label-left/control-right
   *  with a bolder, normal-case section heading and a divider — the Edge
   *  Impulse reference layout — scoped to whichever page opts in. */
  variant?: "stacked" | "row";
}) {
  const orderOptions = (key: string, options: string[]): string[] => {
    const order = optionOrder?.[key];
    if (!order) return options;
    const rank = new Map(order.map((o, i) => [o, i]));
    return [...options].sort((a, b) => {
      const ra = rank.has(a) ? rank.get(a)! : Infinity;
      const rb = rank.has(b) ? rank.get(b)! : Infinity;
      return ra - rb;
    });
  };

  const keys = Object.keys(schema).filter((key) => !hiddenFields?.includes(key));
  const groups = new Map<string, { key: string; degraded: boolean }[]>();
  for (const key of keys) {
    const section = sectionOf ? sectionOf(key) : "Parameters";
    if (!groups.has(section)) groups.set(section, []);
    groups.get(section)!.push({ key, degraded: false });
  }
  for (const df of degradedFields) {
    if (!groups.has(df.section)) groups.set(df.section, []);
    groups.get(df.section)!.push({ key: df.key, degraded: true });
  }

  if (fieldOrder) {
    const rank = new Map(fieldOrder.map((k, i) => [k, i]));
    for (const entries of groups.values()) {
      entries.sort((a, b) => {
        const ra = rank.has(a.key) ? rank.get(a.key)! : Infinity;
        const rb = rank.has(b.key) ? rank.get(b.key)! : Infinity;
        return ra - rb;
      });
    }
  }

  const orderedSections = (() => {
    const all = Array.from(groups.keys());
    if (!sectionOrder) return all;
    const rank = new Map(sectionOrder.map((s, i) => [s, i]));
    return [...all].sort((a, b) => {
      const ra = rank.has(a) ? rank.get(a)! : Infinity;
      const rb = rank.has(b) ? rank.get(b)! : Infinity;
      return ra - rb;
    });
  })();

  /** Wraps a control with its label per `variant` — "row" lays out
   *  label-left/control-right (the Edge Impulse reference), "stacked" keeps
   *  the original label-above-control layout every other DSP page uses. */
  const fieldRow = (
    key: string,
    labelNode: ReactNode,
    control: ReactNode,
    opts?: { disabled?: boolean; htmlFor?: string }
  ) => {
    if (variant === "row") {
      return (
        <div
          key={key}
          style={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: "1rem", padding: "0.55rem 0" }}
        >
          <label
            htmlFor={opts?.htmlFor}
            style={{
              fontSize: "0.85rem",
              fontWeight: 500,
              color: opts?.disabled ? "var(--app-text-soft)" : "var(--app-text)",
              display: "flex",
              alignItems: "center",
              gap: "0.35rem",
            }}
          >
            {labelNode}
          </label>
          <div style={{ width: "260px", maxWidth: "50%", flexShrink: 0 }}>{control}</div>
        </div>
      );
    }
    return (
      <div className="pe-dsp-field" key={key}>
        <label className="pe-dsp-field-label" htmlFor={opts?.htmlFor} style={opts?.disabled ? { opacity: 0.6 } : undefined}>
          {labelNode}
        </label>
        {control}
      </div>
    );
  };

  const renderField = (key: string) => {
    const def = schema[key];
    const value = values[key] ?? def.default;
    const label = fieldLabels?.[key] ?? titleCase(key);
    const description = def.description ?? fieldDescriptions?.[key];
    const labelNode = (
      <>
        {label}
        {description && <HelpTooltip text={description} />}
      </>
    );
    const id = `dsp-${key}`;

    if (checkboxFields?.includes(key)) {
      const checked = !!value && Number(value) !== 0;
      return fieldRow(
        key,
        labelNode,
        <input
          id={id}
          type="checkbox"
          className="pe-dsp-checkbox"
          checked={checked}
          onChange={(e) => onChange(key, e.target.checked ? def.default ?? 1 : 0)}
        />,
        { htmlFor: id }
      );
    }

    if (def.type === "bool") {
      return fieldRow(
        key,
        labelNode,
        <input
          id={id}
          type="checkbox"
          className="pe-dsp-checkbox"
          checked={!!value}
          onChange={(e) => onChange(key, e.target.checked)}
        />,
        { htmlFor: id }
      );
    }

    if (def.type === "enum") {
      const options = orderOptions(key, (def.options || []).filter((o) => !(excludedOptions?.[key] || []).includes(o)));
      return fieldRow(
        key,
        labelNode,
        <select
          id={id}
          className="pe-dsp-select"
          value={value}
          onChange={(e) => onChange(key, e.target.value)}
        >
          {options.map((opt) => (
            <option key={opt} value={opt}>{optionLabels?.[key]?.[opt] ?? titleCase(opt)}</option>
          ))}
        </select>,
        { htmlFor: id }
      );
    }

    if (def.type === "list") {
      const selected: string[] = Array.isArray(value) ? value : def.default || [];
      const min = minSelected?.[key] ?? 0;
      // Each option renders as its own labelled checkbox row (matching the
      // reference "Method" section), not a chip cloud — Edge Impulse's own
      // per-statistic checkboxes, one per line.
      return (
        <div key={key}>
          {orderOptions(key, def.options || []).map((opt) => {
            const active = selected.includes(opt);
            return (
              <div
                key={opt}
                style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: "0.5rem" }}
              >
                <label className="pe-dsp-field-label" htmlFor={`dsp-${key}-${opt}`} style={{ marginBottom: 0 }}>
                  {optionLabels?.[key]?.[opt] ?? titleCase(opt)}
                </label>
                <input
                  id={`dsp-${key}-${opt}`}
                  type="checkbox"
                  checked={active}
                  onChange={() => {
                    const next = active ? selected.filter((s) => s !== opt) : [...selected, opt];
                    if (next.length >= min) onChange(key, next);
                  }}
                />
              </div>
            );
          })}
        </div>
      );
    }

    // int / float
    return fieldRow(
      key,
      labelNode,
      <input
        id={id}
        type="number"
        className="pe-dsp-input"
        value={value}
        min={def.min}
        max={def.max}
        step={def.type === "int" ? 1 : "any"}
        onChange={(e) => {
          const raw = e.target.value;
          if (raw === "") return;
          const num = def.type === "int" ? parseInt(raw, 10) : parseFloat(raw);
          if (Number.isNaN(num)) return;
          let clamped = num;
          if (def.min != null) clamped = Math.max(def.min, clamped);
          if (def.max != null) clamped = Math.min(def.max, clamped);
          onChange(key, clamped);
        }}
      />,
      { htmlFor: id }
    );
  };

  const renderDegraded = (df: DegradedField) => {
    const labelNode = (
      <>
        {df.label}
        <HelpTooltip text={df.note} />
      </>
    );

    if (df.controlType === "checkbox") {
      return fieldRow(
        `degraded-${df.key}`,
        labelNode,
        <input
          type="checkbox"
          className="pe-dsp-checkbox"
          checked={!!df.value}
          disabled
          readOnly
          style={{ cursor: "not-allowed" }}
        />,
        { disabled: true }
      );
    }

    const displayValue = df.value != null ? String(df.value) : "Not yet available";
    return fieldRow(
      `degraded-${df.key}`,
      labelNode,
      <select className="pe-dsp-select" value="" disabled style={{ opacity: 0.5, cursor: "not-allowed" }}>
        <option value="">{displayValue}</option>
      </select>,
      { disabled: true }
    );
  };

  const degradedByKey = new Map(degradedFields.map((df) => [df.key, df]));

  return (
    <>
      {orderedSections.map((section) => {
        const entries = groups.get(section)!;
        return (
          <div key={section} style={{ marginBottom: variant === "row" ? 18 : 10 }}>
            {variant === "row" ? (
              <div style={{ marginBottom: "0.75rem" }}>
                <p style={{ fontSize: "0.95rem", fontWeight: 700, color: "var(--app-text)", marginBottom: "0.5rem" }}>
                  {section}
                </p>
                <div style={{ borderBottom: "1px solid var(--app-border)" }} />
              </div>
            ) : (
              <p className="text-[10px] font-bold uppercase tracking-widest mb-1.5" style={{ color: "var(--app-text-soft)" }}>
                {section}
              </p>
            )}
            {entries.map(({ key, degraded }) =>
              degraded ? renderDegraded(degradedByKey.get(key)!) : renderField(key)
            )}
          </div>
        );
      })}
    </>
  );
}
