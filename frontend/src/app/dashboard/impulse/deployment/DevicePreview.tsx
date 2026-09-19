"use client";

import type { DeviceOption } from "./DeviceSelector";

type Props = {
  device?: DeviceOption;
  platform?: string | null;
  packageType?: string | null;
  impulse?: string | null;
};

function fieldValue(value: string | null | undefined): string {
  if (value === null || value === undefined) return "—";
  const s = String(value).trim();
  return s.length === 0 ? "—" : s;
}

export default function DevicePreview({
  device,
  platform,
  packageType,
  impulse,
}: Props) {
  if (!device) {
    return (
      <div className="pe-dep2-preview">
        <p className="pe-dep2-empty">Select a device to see its details.</p>
      </div>
    );
  }

  const DeviceIcon = device.Icon;

  return (
    <div className="pe-dep2-preview">
      <div className="pe-dep2-preview-head">
        <span className="pe-dep2-preview-icon" aria-hidden="true">
          <DeviceIcon size={28} />
        </span>
        <div className="min-w-0">
          <div className="pe-dep2-preview-name">{device.label}</div>
          <div className="pe-dep2-preview-desc">{device.description}</div>
        </div>
      </div>

      <div className="pe-dep2-preview-grid">
        <div className="pe-dep2-preview-field">
          <div className="pe-dep2-preview-field-label">Platform</div>
          <div className="pe-dep2-preview-field-value">{fieldValue(platform)}</div>
        </div>
        <div className="pe-dep2-preview-field">
          <div className="pe-dep2-preview-field-label">Package</div>
          <div className="pe-dep2-preview-field-value">{fieldValue(packageType)}</div>
        </div>
        <div className="pe-dep2-preview-field">
          <div className="pe-dep2-preview-field-label">Impulse</div>
          <div className="pe-dep2-preview-field-value">{fieldValue(impulse)}</div>
        </div>
      </div>
    </div>
  );
}
