"use client";

import { Download, RefreshCw, Rocket, XCircle } from "lucide-react";

type Props = {
  deviceLabel?: string;
  building: boolean;
  canBuild: boolean;
  disabledReason?: string;
  onBuild: () => void;

  refreshing: boolean;
  onRefresh?: () => void;

  cancelling: boolean;
  canCancel: boolean;
  onCancel: () => void;

  downloadUrl?: string | null;
  downloadFilename?: string | null;
};

export default function DeploymentActions({
  deviceLabel,
  building,
  canBuild,
  disabledReason,
  onBuild,
  refreshing,
  onRefresh,
  cancelling,
  canCancel,
  onCancel,
  downloadUrl,
  downloadFilename,
}: Props) {
  const primaryDisabled = building || !canBuild;
  const primaryLabel = building
    ? "Building…"
    : deviceLabel
      ? `Build for ${deviceLabel}`
      : "Build Deployment Package";

  return (
    <div className="pe-dep2-actions" role="group" aria-label="Deployment actions">
      <span
        className="pe-dep2-tip"
        data-reason={primaryDisabled && disabledReason ? disabledReason : undefined}
      >
        <button
          type="button"
          onClick={onBuild}
          className="pe-dep2-btn pe-dep2-btn--primary"
          disabled={primaryDisabled}
          aria-disabled={primaryDisabled}
          aria-describedby={primaryDisabled && disabledReason ? "pe-dep2-build-reason" : undefined}
        >
          {building ? (
            <RefreshCw size={15} className="animate-spin" aria-hidden="true" />
          ) : (
            <Rocket size={15} aria-hidden="true" />
          )}
          {primaryLabel}
        </button>
        {primaryDisabled && disabledReason && (
          <span id="pe-dep2-build-reason" className="sr-only">
            {disabledReason}
          </span>
        )}
      </span>

      {onRefresh && (
        <button
          type="button"
          onClick={onRefresh}
          className="pe-dep2-btn pe-dep2-btn--secondary"
          disabled={refreshing}
        >
          <RefreshCw
            size={14}
            className={refreshing ? "animate-spin" : ""}
            aria-hidden="true"
          />
          {refreshing ? "Refreshing…" : "Refresh Status"}
        </button>
      )}

      {canCancel && (
        <button
          type="button"
          onClick={onCancel}
          className="pe-dep2-btn pe-dep2-btn--danger"
          disabled={cancelling}
        >
          <XCircle size={14} aria-hidden="true" />
          {cancelling ? "Cancelling…" : "Cancel build"}
        </button>
      )}

      {downloadUrl && (
        <a
          href={downloadUrl}
          target="_blank"
          rel="noopener noreferrer"
          download={downloadFilename || undefined}
          className="pe-dep2-btn pe-dep2-btn--secondary"
        >
          <Download size={14} aria-hidden="true" />
          Download Latest Build
        </a>
      )}
    </div>
  );
}
