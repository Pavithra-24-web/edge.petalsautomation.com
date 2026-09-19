"use client";
import { useState } from "react";
import Link from "next/link";
import { HelpCircle, Copy, Check } from "lucide-react";

/** The `?` affordance used beside a control to explain its behaviour —
 *  keyboard-focusable and dismissible via native title/focus semantics. */
export function HelpTooltip({ text }: { text: string }) {
  return (
    <span
      tabIndex={0}
      title={text}
      role="note"
      aria-label={text}
      className="tt-icon-btn"
      style={{ display: "inline-flex", cursor: "help", color: "var(--app-text-soft)" }}
    >
      <HelpCircle size={13} />
    </span>
  );
}

/** Copy-to-clipboard with a transient "Copied" acknowledgement. */
export function CopyButton({ getText }: { getText: () => string }) {
  const [copied, setCopied] = useState(false);
  return (
    <button
      type="button"
      className="tt-icon-btn"
      title={copied ? "Copied" : "Copy"}
      aria-label="Copy to clipboard"
      onClick={async () => {
        try {
          await navigator.clipboard.writeText(getText());
          setCopied(true);
          setTimeout(() => setCopied(false), 1500);
        } catch {
          // Clipboard API unavailable — no-op rather than a confusing error.
        }
      }}
    >
      {copied ? <Check size={13} /> : <Copy size={13} />}
    </button>
  );
}

/** The `Parameters` / `Generate features` pill pair. `parametersSaved` gates
 *  the second tab per R13 (`dsp_params_saved_at`); an unsaved click is
 *  prevented client-side rather than left to 404/redirect on the next page. */
export function DspTabs({
  blockType,
  impulseId,
  active,
  parametersSaved,
}: {
  blockType: string;
  impulseId: string | null;
  active: "parameters" | "generate-features";
  parametersSaved: boolean;
}) {
  const qs = impulseId ? `?impulseId=${impulseId}` : "";
  return (
    <div className="pe-dsp-tabs">
      <Link
        href={`/dashboard/impulse/${blockType}/parameters${qs}`}
        className={`pe-dsp-tab${active === "parameters" ? " is-active" : ""}`}
      >
        Parameters
      </Link>
      {parametersSaved ? (
        <Link
          href={`/dashboard/impulse/${blockType}/generate-features${qs}`}
          className={`pe-dsp-tab${active === "generate-features" ? " is-active" : ""}`}
        >
          Generate features
        </Link>
      ) : (
        <button
          type="button"
          disabled
          aria-disabled="true"
          title="Save parameters before generating features."
          className={`pe-dsp-tab is-disabled${active === "generate-features" ? " is-active" : ""}`}
          style={{ opacity: 0.5, cursor: "not-allowed" }}
          onClick={(e) => e.preventDefault()}
        >
          Generate features
        </button>
      )}
    </div>
  );
}

/** `3 × 129 = 387` — window-aware feature count derived from a real
 *  `/dsp/preview` result (R26), never computed client-side. */
export function FeatureCountBadge({
  blockType,
  featureCount,
  axisCount,
}: {
  blockType: string;
  featureCount: number | null;
  axisCount: number;
}) {
  if (featureCount == null) return null;

  if (blockType === "raw") {
    return <p className="pe-dsp-caption">Whole-recording preview: {featureCount} features</p>;
  }

  if (blockType === "spectral_analysis") {
    const perAxis = axisCount > 0 ? featureCount / axisCount : 0;
    const showBreakdown = axisCount > 0 && Number.isInteger(perAxis) && perAxis > 0;
    return (
      <p className="pe-dsp-caption">
        {showBreakdown
          ? `${axisCount} axes × ${perAxis} per axis = ${featureCount} total features`
          : `${featureCount} total features`}
      </p>
    );
  }

  return <p className="pe-dsp-caption">{featureCount} total features</p>;
}
