/**
 * User-facing model names.
 *
 * This module is display-only. It never touches architecture keys, payload
 * values, enums or anything the backend parses — those keep their existing
 * identifiers (`fomo_mobilenetv2_0_1`, `yolo_pro`, `mobilenet_v2_ssd_fpn_lite`,
 * …). Use it wherever a model name is rendered, especially for strings that
 * arrive from the backend (training logs, job status, notifications), so the
 * mapping happens at render time only.
 *
 * Version suffix rule: the product name replaces the old one and any existing
 * version / backbone qualifier is preserved, with version markers normalised to
 * lowercase `v1` / `v2`. No version is invented where the source had none.
 */

/** Object detection — formerly "MobileNetV2 SSD FPN-Lite". */
export const EDGE_DETECT_LITE = "EdgeDetect Lite";

/** Centroid detection — formerly "FOMO". */
export const NANO_VISION = "NanoVision";

/** Box detection — formerly "YOLO-Pro". */
export const VISION_PRO = "Vision Pro";

/** Centroid detection, first generation. */
export const NANO_VISION_V1 = `${NANO_VISION} v1`;

/** Centroid detection on the 0.35-width MobileNetV2 backbone. */
export const NANO_VISION_MNV2_035 = `${NANO_VISION} MobileNetV2 0.35`;

/**
 * Display name for a raw architecture / model key stamped by the backend.
 * Returns `null` for unknown keys so callers keep their own fallback.
 */
export function modelDisplayName(key: string | null | undefined): string | null {
  if (!key) return null;
  const k = String(key).toLowerCase().trim();
  if (!k) return null;
  if (k === "fomo_v1" || k === "fomo") return NANO_VISION_V1;
  if (k.startsWith("fomo")) return NANO_VISION_MNV2_035;
  if (k.startsWith("yolo_pro") || k === "yolo-pro") return VISION_PRO;
  if (
    k === "mobilenetv2_ssd_fpnlite_320x320" ||
    k === "mobilenet_v2_ssd_fpn_lite" ||
    k === "ssd_detection" ||
    k === "object_detection"
  ) {
    return EDGE_DETECT_LITE;
  }
  return null;
}

/**
 * Rewrite old model names inside a free-text string that came from the backend
 * (a training log line, an error message, a job status line) so the user reads
 * the current names. The string itself is only rewritten for display — nothing
 * is sent back, and snake_case identifiers such as `fomo_mobilenetv2_0_1` or
 * `yolo_pro_detection` are deliberately left alone.
 */
export function mapModelNamesInText(text: string): string {
  if (!text) return text;
  // The leading `(^|[^\w-])` group (rather than a lookbehind, which older
  // Safari cannot parse) keeps the preceding character and stops the rewrite
  // from firing inside snake_case identifiers.
  return text
    // "MobileNetV2 SSD FPN-Lite" / "MobileNetV2 SSD FPN Lite" (a size suffix stays)
    .replace(/MobileNetV2[ _]SSD[ _]FPN[- _]?Lite/gi, EDGE_DETECT_LITE)
    // "FOMO v2" / "FOMO V1" — keep the version, normalised to lowercase
    .replace(/(^|[^\w-])FOMO[ -]?[vV](\d+)(?![\w-])/g, `$1${NANO_VISION} v$2`)
    // bare "FOMO", but never inside an identifier like fomo_mobilenetv2_0_1
    .replace(/(^|[^\w-])FOMO(?![\w-])/g, `$1${NANO_VISION}`)
    // "YOLO-Pro" / "YOLO Pro", but never inside yolo_pro_detection
    .replace(/(^|[^\w-])YOLO[- ]Pro(?![\w-])/gi, `$1${VISION_PRO}`);
}
