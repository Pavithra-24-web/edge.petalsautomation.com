"use client";

import { Cpu, Download } from "lucide-react";
import type { DeviceOption } from "./DeviceSelector";

type StatusKind = "success" | "building" | "failed" | "cancelled" | "neutral";

function classifyStatus(status: string): StatusKind {
  if (status === "completed") return "success";
  if (status === "failed") return "failed";
  if (status === "cancelled") return "cancelled";
  if (status === "running" || status === "pending") return "building";
  return "neutral";
}

function formatRelative(iso: string): string {
  const t = new Date(iso).getTime();
  if (Number.isNaN(t)) return "";
  const diff = Date.now() - t;
  const s = Math.round(diff / 1000);
  if (s < 60) return `${s}s ago`;
  const m = Math.round(s / 60);
  if (m < 60) return `${m}m ago`;
  const h = Math.round(m / 60);
  if (h < 24) return `${h}h ago`;
  const d = Math.round(h / 24);
  if (d < 30) return `${d}d ago`;
  const mo = Math.round(d / 30);
  if (mo < 12) return `${mo}mo ago`;
  return `${Math.round(mo / 12)}y ago`;
}

type Props = {
  deployments: any[];
  deviceByProfile: Map<string, DeviceOption>;
};

export default function BuildTimeline({ deployments, deviceByProfile }: Props) {
  if (deployments.length === 0) {
    return (
      <div className="pe-dep2-empty">
        No deployments yet. Build your first package above.
      </div>
    );
  }

  return (
    <ol className="pe-dep2-timeline" aria-label="Build history">
      {deployments.map((dep) => {
        const device = deviceByProfile.get(dep.device_profile);
        const IconCmp = device?.Icon ?? Cpu;
        const kind = classifyStatus(dep.status);
        const created = dep.created_at ? new Date(dep.created_at) : null;
        const absolute = created ? created.toLocaleString() : "";
        const relative = created ? formatRelative(dep.created_at) : "";

        return (
          <li key={dep.id} className="pe-dep2-tl-row">
            <span
              className="pe-dep2-tl-dot"
              data-status={kind === "neutral" ? "cancelled" : kind}
              aria-hidden="true"
            />
            <span className="pe-dep2-tl-icon" aria-hidden="true">
              <IconCmp size={16} />
            </span>
            <div className="pe-dep2-tl-body">
              <div className="pe-dep2-tl-title">
                <span>{device?.label ?? dep.target}</span>
                <span className="pe-dep2-tl-fmt">
                  {dep.deployment_format === "pxe" ? ".pxe" : ".pe"}
                </span>
                <span
                  className={`pe-dep2-pill ${
                    kind === "success"
                      ? "pe-dep2-pill--success"
                      : kind === "failed"
                        ? "pe-dep2-pill--warn"
                        : kind === "cancelled"
                          ? "pe-dep2-pill--muted"
                          : "pe-dep2-pill--info"
                  }`}
                  style={{ padding: "0.15rem 0.5rem", fontSize: "0.65rem" }}
                >
                  <span className="pe-dep2-pill-dot" aria-hidden="true" />
                  {dep.status}
                </span>
              </div>
              <div className="pe-dep2-tl-meta">
                <span title={absolute}>{relative || absolute}</span>
              </div>
              {dep.status === "failed" && dep.error_message && (
                <div className="pe-dep2-tl-error" title={dep.error_message}>
                  {dep.error_message}
                </div>
              )}
            </div>
            <div>
              {dep.status === "completed" && dep.download_url && (
                <a
                  href={dep.download_url}
                  target="_blank"
                  rel="noopener noreferrer"
                  download={dep.download_filename || undefined}
                  className="pe-dep2-btn pe-dep2-btn--secondary"
                  style={{ padding: "0.45rem 0.75rem", fontSize: "0.75rem" }}
                  aria-label={`Download ${device?.label ?? dep.target} build`}
                >
                  <Download size={12} aria-hidden="true" />
                  Download
                </a>
              )}
            </div>
          </li>
        );
      })}
    </ol>
  );
}
