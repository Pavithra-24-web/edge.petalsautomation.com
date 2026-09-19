"use client";

import { Network, ScatterChart, Tag, CheckCircle2, AlertTriangle } from "lucide-react";

type Props = {
  model: any;
  ready: boolean;
};

function valueOr(value: unknown, fallback = "—"): string {
  if (value === null || value === undefined) return fallback;
  if (typeof value === "string" && value.trim() === "") return fallback;
  return String(value);
}

export default function ModelOverview({ model, ready }: Props) {
  const classes = Array.isArray(model?.label_names) && model.label_names.length
    ? model.label_names.join(", ")
    : "—";
  const version = model?.version != null ? `v${model.version}` : "—";

  return (
    <div className="pe-dep2-metric-row" role="list" aria-label="Model overview">
      <div className="pe-dep2-metric" data-tone="indigo" role="listitem">
        <div className="pe-dep2-metric-head">
          <span className="pe-dep2-metric-icon" aria-hidden="true">
            <Network size={16} />
          </span>
        </div>
        <div className="pe-dep2-metric-label">Architecture</div>
        <div className="pe-dep2-metric-value" title={valueOr(model?.architecture)}>
          {valueOr(model?.architecture)}
        </div>
      </div>

      <div className="pe-dep2-metric" data-tone="sky" role="listitem">
        <div className="pe-dep2-metric-head">
          <span className="pe-dep2-metric-icon" aria-hidden="true">
            <ScatterChart size={16} />
          </span>
        </div>
        <div className="pe-dep2-metric-label">Classes</div>
        <div className="pe-dep2-metric-value" title={classes}>
          {classes}
        </div>
      </div>

      <div className="pe-dep2-metric" data-tone="indigo" role="listitem">
        <div className="pe-dep2-metric-head">
          <span className="pe-dep2-metric-icon" aria-hidden="true">
            <Tag size={16} />
          </span>
        </div>
        <div className="pe-dep2-metric-label">Version</div>
        <div className="pe-dep2-metric-value">{version}</div>
      </div>

      <div
        className="pe-dep2-metric"
        data-tone={ready ? "emerald" : "amber"}
        role="listitem"
      >
        <div className="pe-dep2-metric-head">
          <span className="pe-dep2-metric-icon" aria-hidden="true">
            {ready ? <CheckCircle2 size={16} /> : <AlertTriangle size={16} />}
          </span>
        </div>
        <div className="pe-dep2-metric-label">Status</div>
        <div className="pe-dep2-metric-value">
          {ready ? "Ready" : "Training Required"}
        </div>
      </div>
    </div>
  );
}
