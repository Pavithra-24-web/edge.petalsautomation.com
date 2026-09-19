/**
 * Canonical "is this sample labeled?" definition for object-detection
 * projects — TS mirror of `backend/app/services/labeling_status.py`.
 *
 * Operates on the `_sample_to_dict` API response shape, where
 * `is_background` / `is_disabled` are already flattened to top-level
 * booleans (unlike the backend's ORM `Sample`, where they live nested
 * inside `extra_metadata`). `extra_metadata.boundingBoxes` is still read
 * from the nested metadata object on both sides.
 *
 * The two implementations must agree on every case in
 * `backend/tests/fixtures/labeling_status_fixtures.json` — see
 * `frontend/src/__tests__/labelingStatus.test.ts` and
 * `backend/tests/test_labeling_status.py`.
 */

// Mirrors app.workers.sample_utils.UNLABELED_NAMES.
const UNLABELED_NAMES = new Set(["unlabeled", "unlabelled", "unknown"]);

// Mirrors labeling_status.IMAGE_EXTS — the only sample kind this editor/
// queue can annotate.
const IMAGE_EXTS = new Set([".jpg", ".jpeg", ".png"]);

function ext(filename: string | null | undefined): string {
  const name = (filename || "").toLowerCase();
  const idx = name.lastIndexOf(".");
  return idx !== -1 ? name.slice(idx) : "";
}

/** Mirrors sample_utils.boxes_have_label: true iff at least one box carries
 * a non-empty, non-placeholder label (via either `label_id` or `label`). */
function boxesHaveLabel(boxes: unknown): boolean {
  if (!Array.isArray(boxes)) return false;
  for (const box of boxes) {
    if (!box || typeof box !== "object") continue;
    const key = String((box as any).label_id ?? (box as any).label ?? "").trim();
    if (key && !UNLABELED_NAMES.has(key.toLowerCase())) return true;
  }
  return false;
}

/** True if `sample` is an image the editor can show and is not disabled. */
export function isAnnotatable(sample: any): boolean {
  if (sample?.is_disabled) return false;
  return IMAGE_EXTS.has(ext(sample?.filename));
}

/**
 * The Phase 1 definition of "unlabeled" for object detection.
 *
 * True iff `sample` is annotatable, is not marked background (a background
 * image is a deliberate, complete "none of the classes" annotation, not
 * pending work), and carries no bounding box with a real, non-placeholder
 * label. A top-level `label_id` with no boxes does not count as labeled —
 * for detection, the boxes are the labels.
 */
export function isSampleUnlabeled(sample: any): boolean {
  if (!isAnnotatable(sample)) return false;
  if (sample?.is_background) return false;
  return !boxesHaveLabel(sample?.extra_metadata?.boundingBoxes);
}

/** Inverse of `isSampleUnlabeled`, scoped to annotatable samples. A
 * non-annotatable sample (video, disabled) is neither labeled nor
 * unlabeled — it sits outside the set these predicates account for. */
export function isSampleLabeled(sample: any): boolean {
  return isAnnotatable(sample) && !isSampleUnlabeled(sample);
}
