"use client";
import { Save } from "lucide-react";
import { useCallback, useRef, useState } from "react";

const MIN_LOADING_MS = 2000;

type Props = {
  onSave: () => void | Promise<void>;
  onSuccess?: () => void;
  label?: string;
  loadingLabel?: string;
  disabled?: boolean;
  showIcon?: boolean;
  fullWidth?: boolean;
  className?: string;
};

export default function PremiumSaveButton({
  onSave,
  onSuccess,
  label = "Save",
  loadingLabel = "Saving...",
  disabled,
  showIcon = true,
  fullWidth = false,
  className = "",
}: Props) {
  const [loading, setLoading] = useState(false);
  const inflight = useRef(false);

  const handleClick = useCallback(async () => {
    if (inflight.current || disabled) return;
    inflight.current = true;
    setLoading(true);
    const started = Date.now();
    let succeeded = false;
    try {
      await onSave();
      succeeded = true;
    } catch {
      // Errors are surfaced immediately by the caller (toast etc.);
      // release loading right away so the user can retry.
      setLoading(false);
      inflight.current = false;
      return;
    }
    const elapsed = Date.now() - started;
    const wait = Math.max(0, MIN_LOADING_MS - elapsed);
    setTimeout(() => {
      setLoading(false);
      inflight.current = false;
      if (succeeded) onSuccess?.();
    }, wait);
  }, [onSave, onSuccess, disabled]);

  return (
    <button
      type="button"
      onClick={handleClick}
      disabled={disabled || loading}
      className={[
        "pe-save-btn",
        loading ? "is-loading" : "",
        fullWidth ? "w-full" : "",
        className,
      ].filter(Boolean).join(" ")}
      aria-busy={loading}
    >
      {loading ? (
        <span className="pe-save-spinner" aria-hidden="true" />
      ) : (
        showIcon && <Save size={14} aria-hidden="true" />
      )}
      <span>{loading ? loadingLabel : label}</span>
    </button>
  );
}
