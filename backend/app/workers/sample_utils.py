"""
Shared helpers for determining whether a Sample is usable for training / DSP.

Imported by dsp_worker, training_worker, and DSP API endpoints so the
"is this sample usable?" definition stays in sync everywhere.

Rules
-----
A sample is **usable for training** if it satisfies at least one of:

1. ``sample.label_id`` is set AND that label is not a placeholder
   (i.e. its DB name is not in UNLABELED_NAMES), OR
2. ``sample.extra_metadata["boundingBoxes"]`` contains at least one box
   whose ``label_id`` or ``label`` key is non-empty and not a placeholder, OR
3. the sample is explicitly marked as a background-only image
   (``extra_metadata[IS_BACKGROUND_KEY]``) **and** the caller opted in with
   ``allow_background=True``.

Rule 3 is the correction to a rule this module previously stated as policy:
that a sample with neither a label nor a box "has no training signal".  That is
true for classification, where the target *is* the label, and false for object
detection, where an image containing none of the classes is a legitimate
negative that teaches the model what not to fire on.  The distinction is the
caller's to make, so the opt-in defaults to ``False`` and every classification
path keeps behaving exactly as it did.

An *unmarked* sample with no label and no boxes is still excluded — that is an
unannotated image nobody has looked at, not an asserted negative.  Only the
explicit marker crosses the line.

Whatever the answer, it must be the same answer everywhere: feature caching,
cache-staleness checks and dataset assembly all consult this module so a
sample cannot be cached but not trained on, or vice versa.
"""
from __future__ import annotations

import io
from typing import Optional, Set

# Canonical set of label names that mean "no real label".
# Compared case-insensitively and after stripping whitespace.
UNLABELED_NAMES: frozenset = frozenset({"unlabeled", "unlabelled", "unknown"})

# Key in ``Sample.extra_metadata`` marking a deliberate *negative* (background-
# only) image: an image the user asserts contains none of the project's classes.
#
# Stored in the existing free-form JSON blob rather than as a column, mirroring
# the ``is_disabled`` flag that already lives there.  Defined once and imported
# everywhere so a typo in a string literal cannot silently fail open (a
# misspelled key reads as "not background", quietly dropping the sample instead
# of raising).
#
# Absence of the key means exactly the pre-existing behaviour.  Do NOT infer
# background from ``"boundingBoxes" in meta`` — the dataset page writes
# ``boundingBoxes: []`` when a user *clears* a label, which means "annotation
# removed", not "this is a training negative".
IS_BACKGROUND_KEY: str = "is_background"


def is_background_sample(sample) -> bool:
    """Return True if *sample* is explicitly marked as a background-only image."""
    meta = getattr(sample, "extra_metadata", None)
    if not isinstance(meta, dict):
        return False
    return bool(meta.get(IS_BACKGROUND_KEY))


def label_name_is_placeholder(name: Optional[str]) -> bool:
    """Return True if *name* is a placeholder / unassigned label name."""
    return (name or "").strip().lower() in UNLABELED_NAMES


_PREFETCH_SENTINEL = object()


def iter_prefetched(items, fetch, max_workers: int = 8, lookahead: Optional[int] = None):
    """Apply *fetch* to *items* concurrently, yielding ``(item, data, error)``
    tuples **in input order**.

    Built for the recurring "download every sample's bytes from object storage,
    then process it" pattern.  S3 fetches are latency-bound, so overlapping them
    across a small thread pool collapses total wait time without changing the
    order in which results are consumed.

    On success ``error`` is ``None`` and ``data`` holds ``fetch(item)``.  On
    failure ``data`` is ``None`` and ``error`` holds the exception — callers keep
    their existing per-item "log and skip" behaviour instead of aborting the
    whole loop (mirrors the try/except each call site already had).

    At most ``lookahead`` fetches are kept in flight (defaults to
    ``max_workers * 2``), bounding peak memory to that many objects regardless of
    how many samples are processed.
    """
    from collections import deque
    from concurrent.futures import ThreadPoolExecutor

    items = list(items)
    if not items:
        return
    if lookahead is None:
        lookahead = max(1, max_workers * 2)

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        pending = deque()
        it = iter(items)
        for _ in range(min(lookahead, len(items))):
            item = next(it)
            pending.append((item, pool.submit(fetch, item)))
        while pending:
            item, fut = pending.popleft()
            try:
                data, error = fut.result(), None
            except Exception as exc:  # surfaced to caller, not swallowed
                data, error = None, exc
            yield item, data, error
            nxt = next(it, _PREFETCH_SENTINEL)
            if nxt is not _PREFETCH_SENTINEL:
                pending.append((nxt, pool.submit(fetch, nxt)))


def apply_train_subset(samples: list, impulse) -> list:
    """Randomly subsample the *training* split of ``samples`` per the impulse's
    ``train_subset_percent`` and return the new list (test split always kept).

    Used by both the DSP worker (when building the feature cache) and the
    training worker (when recomputing features inline because the cache is
    missing/stale) so the same impulse + same percent always picks the same
    set of training samples — the RNG is seeded by ``impulse_id|percent``.

    A ``train_subset_percent`` of 100 (or missing) is a no-op fast path.
    """
    pct = float(getattr(impulse, "train_subset_percent", 100.0) or 100.0)
    if pct >= 100.0:
        return samples

    train_pool = []
    test_pool = []
    for s in samples:
        st = getattr(s, "sample_type", None)
        st_val = str(getattr(st, "value", st)).lower()
        (test_pool if st_val == "testing" else train_pool).append(s)

    if not train_pool:
        return samples

    keep_n = max(1, int(round(len(train_pool) * (pct / 100.0))))
    if keep_n >= len(train_pool):
        return samples

    # numpy is a heavy import for callers that never trigger subset — local-import.
    import numpy as np
    seed_str = f"{getattr(impulse, 'id', '?')}|{pct:.4f}"
    rng = np.random.default_rng(abs(hash(seed_str)) % (2 ** 32))
    idx = rng.choice(len(train_pool), size=keep_n, replace=False)
    chosen_train = [train_pool[int(i)] for i in idx]
    return chosen_train + test_pool


def boxes_have_label(boxes) -> bool:
    """
    Return True if *boxes* (a ``boundingBoxes`` list) contains at least one
    box whose label key is non-empty and NOT a placeholder name.

    Accepts both ``label_id`` (UUID string) and ``label`` (name string)
    box formats, mirroring the dual-key convention used throughout. Shared
    by ``has_usable_box`` (per-sample check) and ``enforce_background_invariant``
    (per-write invariant check) so "does this box list count as real
    annotation" is answered identically everywhere.
    """
    if not isinstance(boxes, list):
        return False
    for box in boxes:
        if not isinstance(box, dict):
            continue
        key = (box.get("label_id") or box.get("label") or "").strip()
        if key and key.lower() not in UNLABELED_NAMES:
            return True
    return False


def has_usable_box(sample) -> bool:
    """
    Return True if *sample* has at least one bounding-box annotation whose
    label key is non-empty and NOT a placeholder name.
    """
    meta = sample.extra_metadata
    if not isinstance(meta, dict):
        return False
    return boxes_have_label(meta.get("boundingBoxes", []))


def enforce_background_invariant(meta: dict, label_id: Optional[str] = None) -> dict:
    """
    Strip the background marker from *meta* if the sample has real signal.

    The invariant every write path must maintain: ``is_background`` can
    never coexist with a real ``label_id`` or a labeled bounding box — a
    background sample asserts "none of the project's classes are here",
    which a positive label or box directly contradicts. The training
    loader ("A negative never contributes GT, whatever the cache holds",
    see training_worker.py) treats the marker as authoritative and silently
    wipes any boxes still sitting on a background-flagged sample, so a
    write path that sets real content without clearing the marker doesn't
    fail loudly — it just trains on less data than the user thinks it has.

    Only ever *removes* the marker; never sets it. Setting it is each call
    site's own decision (and comes with its own obligation to blank
    ``boundingBoxes``/``label_id`` — see the AI-labeling zero-detection
    branch for the reference implementation of that direction).

    Mutates and returns *meta* in place for caller convenience; safe to
    call on a dict that doesn't have the key at all.
    """
    if not isinstance(meta, dict) or not meta.get(IS_BACKGROUND_KEY):
        return meta
    if label_id or boxes_have_label(meta.get("boundingBoxes")):
        meta.pop(IS_BACKGROUND_KEY, None)
    return meta


def is_sample_usable(
    sample,
    usable_label_ids: Optional[Set[str]] = None,
    allow_background: bool = False,
) -> bool:
    """
    Return True if *sample* should be included in feature caching and training.

    Parameters
    ----------
    sample:
        ORM Sample instance (needs ``label_id`` and ``extra_metadata`` attrs).
    usable_label_ids:
        Pre-computed set of ``Label.id`` values whose names are **not**
        placeholders.  Pass ``None`` only when label names cannot be checked
        (legacy path — not recommended).  When provided, a sample whose
        ``label_id`` points to a placeholder label is treated as unlabeled.
    allow_background:
        Detection callers pass True to include images explicitly marked as
        background-only negatives.  Left False (the default) the function is
        byte-identical to its pre-negative-image behaviour, which is what keeps
        classification results reproducible.
    """
    if sample.label_id:
        if usable_label_ids is None or sample.label_id in usable_label_ids:
            return True
    if has_usable_box(sample):
        return True
    return allow_background and is_background_sample(sample)


def normalize_bounding_boxes(
    boxes: list,
    image_bytes: Optional[bytes] = None,
) -> list[dict]:
    """
    Return boxes in YOLO-style normalised coordinates.

    The app currently stores boxes from multiple sources:
    - manual annotation / some UI paths: already normalised [0,1]
    - AI labeling / imported annotations: raw pixel coordinates

    YOLO-Pro training/eval expects normalised coordinates, so convert any
    pixel-space boxes using the raw image dimensions when available.

    Also resolves the label field using a priority chain and writes a
    canonical ``label_id`` key on every output box.  Boxes arriving from:
      • the standard DB / annotation UI   use  ``label_id`` (UUID string)
      • the AI-labeling pipeline          use  ``label``   (name string)
      • some legacy import paths          use  ``labelId`` or ``class_name``
    The resolved value is stored as ``label_id`` so ``_create_fomo_heatmap``'s
    dual-key ``label_map`` lookup always sees a consistent field name.
    If no label key can be resolved the box is still emitted with
    ``label_id=None`` so callers can filter it explicitly rather than have it
    silently dropped deep inside the heatmap builder.
    """
    if not isinstance(boxes, list):
        return []

    width = height = None
    if image_bytes:
        try:
            from PIL import Image
            img = Image.open(io.BytesIO(image_bytes))
            width, height = img.size
        except Exception:
            width = height = None

    normalised: list[dict] = []
    for box in boxes:
        if not isinstance(box, dict):
            continue

        x = float(box.get("x", 0) or 0)
        y = float(box.get("y", 0) or 0)
        # Accept both w/h (canonical) and width/height (legacy info.labels import)
        w = float(box.get("w") or box.get("width") or 0)
        h = float(box.get("h") or box.get("height") or 0)

        needs_normalisation = max(abs(x), abs(y), abs(w), abs(h)) > 1.0
        if needs_normalisation and width and height:
            x /= float(width)
            y /= float(height)
            w /= float(width)
            h /= float(height)

        # ── Label-field normalization ──────────────────────────────────────────
        # Priority chain: label_id (UUID) → label (name-string, AI-labeling) →
        # labelId (camelCase legacy) → class_name (some import pipelines).
        # Collapse empty strings to None so downstream code sees a clean None
        # instead of "" when no label is present.
        resolved_label: Optional[str] = (
            box.get("label_id")       # UUID path (standard DB / annotation UI)
            or box.get("label")       # name-string path (AI-labeling)
            or box.get("labelId")     # camelCase legacy variant
            or box.get("class_name")  # some import pipelines
        ) or None  # "" → None

        normalised.append({
            **box,
            "x": float(x),
            "y": float(y),
            "w": float(w),
            "h": float(h),
            # Always present; None when no label could be resolved.
            "label_id": resolved_label,
        })

    return normalised
