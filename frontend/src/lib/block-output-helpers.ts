/**
 * Pure selector for the dashboard's "Download block output" card (C11 Phase 3).
 * Maps a `GET /trained-models/job/{job_id}` payload + features-ready flag to
 * the four fixed download rows, keyed by format/variant — never by array
 * position, since worker export order is not a contract.
 */
import { slugify } from "./pxe-filename";

export type BlockOutputRowKey = "features" | "keras" | "litert_float32" | "litert_int8";

export interface TrainedModelArtifact {
  id: string;
  format: string;
  variant?: string | null;
}

export interface BlockOutputRow {
  key: BlockOutputRowKey;
  label: string;
  enabled: boolean;
  /** Why the row is disabled; null when enabled. */
  reason: string | null;
  /** The trained-model id to download, for the three model rows. */
  modelId?: string;
}

export interface BuildBlockOutputRowsParams {
  featuresReady: boolean;
  activeModelRunId: string | null | undefined;
  /** Artifacts for the active run, from `trainedModelsApi.listForJob`. */
  models: TrainedModelArtifact[] | null | undefined;
}

export function buildBlockOutputRows({
  featuresReady,
  activeModelRunId,
  models,
}: BuildBlockOutputRowsParams): BlockOutputRow[] {
  const hasActiveModel = !!activeModelRunId;
  const list = models || [];

  const keras = list.find((m) => m.format === "keras");
  const float32 = list.find(
    (m) => m.format === "tflite" && (m.variant ?? "float32") === "float32"
  );
  const int8 = list.find((m) => m.format === "tflite" && m.variant === "int8");

  const noActiveModelReason = "No active model for this impulse";

  return [
    {
      key: "features",
      label: "Features (.npy)",
      enabled: featuresReady,
      reason: featuresReady ? null : "No features generated yet",
    },
    {
      key: "keras",
      label: "Keras model (.keras)",
      enabled: hasActiveModel && !!keras,
      reason: !hasActiveModel
        ? noActiveModelReason
        : !keras
        ? "Keras model unavailable for this run"
        : null,
      modelId: keras?.id,
    },
    {
      key: "litert_float32",
      label: "LiteRT float32 (unquantized)",
      enabled: hasActiveModel && !!float32,
      reason: !hasActiveModel
        ? noActiveModelReason
        : !float32
        ? "Unquantized model unavailable for this run"
        : null,
      modelId: float32?.id,
    },
    {
      key: "litert_int8",
      label: "LiteRT int8 (quantized)",
      enabled: hasActiveModel && !!int8,
      reason: !hasActiveModel
        ? noActiveModelReason
        : !int8
        ? "Quantized model unavailable — int8 conversion may have failed"
        : null,
      modelId: int8?.id,
    },
  ];
}

// ─── Download filename resolution ──────────────────────────────────────────
//
// Content-Disposition is not on the browser's CORS-safelisted response
// header set, so unless the backend sends `Access-Control-Expose-Headers:
// Content-Disposition` (see backend/app/main.py), `res.headers["content-
// disposition"]` reads as undefined for a cross-origin request even though
// curl/TestClient — which aren't subject to that restriction — see it fine.
// The fallback below must therefore be correct on its own, per row: a single
// shared "model.bin" fallback would make every model row indistinguishable
// on disk the moment that header isn't readable.

/** Parses a filename out of a raw `Content-Disposition` header value. */
export function parseFilenameFromDisposition(header: string | undefined | null): string | null {
  if (!header) return null;
  const star = /filename\*\s*=\s*[^']*''([^;]+)/i.exec(header);
  if (star && star[1]) {
    try {
      return decodeURIComponent(star[1].trim().replace(/^"|"$/g, ""));
    } catch {
      return star[1].trim().replace(/^"|"$/g, "");
    }
  }
  const plain = /filename\s*=\s*"?([^";]+)"?/i.exec(header);
  return plain && plain[1] ? plain[1].trim() : null;
}

const MODEL_ROW_FALLBACK_FILENAMES: Record<Exclude<BlockOutputRowKey, "features">, string> = {
  keras: "model.keras",
  litert_float32: "model_float32.tflite",
  litert_int8: "model_int8.tflite",
};

/**
 * Resolves the filename to save a block-output download under: the real
 * backend filename when `Content-Disposition` is readable, otherwise a
 * row-specific fallback (never one generic name shared across rows).
 */
export function resolveBlockOutputFilename(
  rowKey: BlockOutputRowKey,
  dispositionHeader: string | undefined | null,
  impulseName?: string | null,
): string {
  const fromHeader = parseFilenameFromDisposition(dispositionHeader);
  if (fromHeader) return fromHeader;
  if (rowKey === "features") {
    return `${slugify(impulseName) || "impulse"}_features.npy`;
  }
  return MODEL_ROW_FALLBACK_FILENAMES[rowKey];
}
