"""
Canonical "is this sample labeled?" definition for object-detection projects.

Single source of truth, replacing three previously-divergent definitions that
lived in ``GET /samples/project/{id}?unlabeled_only``, ``POST /ai-labeling/run``
(``skip_labeled``) and the Dataset page's own client-side count — see
``docs/labellingqueue_implementation.md`` Phase 1 for the audit that found
them disagreeing with each other and with what the training pipeline
actually trusts (``app.workers.sample_utils``).

For object detection **the boxes are the labels** (see the equivalent
comment in ``ObjectDetectionLabeling.tsx``): a bare top-level ``label_id``
with no boxes still counts as unlabeled here. That is a deliberate, narrower
definition than ``sample_utils.is_sample_usable`` (which answers "should the
training pipeline use this sample?", not "has a human finished labeling
it?") and the two are not interchangeable.

Mirrored on the frontend as
``frontend/src/components/dashboard/labeling/labelingStatus.ts``, which
operates on the ``_sample_to_dict`` response shape instead of the ORM model.
The two must agree on every case in the shared fixture table
(``backend/tests/fixtures/labeling_status_fixtures.json``) — see
``backend/tests/test_labeling_status.py``.
"""
from __future__ import annotations

from typing import Iterable

from app.workers.sample_utils import IS_BACKGROUND_KEY, boxes_have_label

# Images are the only sample kind this editor/queue can annotate; videos and
# any other file type are never annotatable regardless of label state.
IMAGE_EXTS = frozenset({".jpg", ".jpeg", ".png"})


def _ext(filename: str) -> str:
    name = (filename or "").lower()
    idx = name.rfind(".")
    return name[idx:] if idx != -1 else ""


def _meta(sample) -> dict:
    meta = getattr(sample, "extra_metadata", None)
    return meta if isinstance(meta, dict) else {}


def is_annotatable(sample) -> bool:
    """True if *sample* is an image the editor can show and is not disabled."""
    if _meta(sample).get("is_disabled"):
        return False
    return _ext(getattr(sample, "filename", "")) in IMAGE_EXTS


def is_sample_unlabeled(sample) -> bool:
    """The Phase 1 definition of "unlabeled" for object detection.

    True iff *sample* is annotatable, is not marked background (a background
    image is a deliberate, complete "none of the classes" annotation, not
    pending work), and carries no bounding box with a real, non-placeholder
    label. A top-level ``label_id`` with no boxes does not count as labeled.
    """
    if not is_annotatable(sample):
        return False
    meta = _meta(sample)
    if meta.get(IS_BACKGROUND_KEY):
        return False
    return not boxes_have_label(meta.get("boundingBoxes"))


def is_sample_labeled(sample) -> bool:
    """Inverse of ``is_sample_unlabeled``, scoped to annotatable samples.

    A non-annotatable sample (video, disabled) is neither labeled nor
    unlabeled — it sits outside the set these predicates account for.
    """
    return is_annotatable(sample) and not is_sample_unlabeled(sample)


def summarize(samples: Iterable) -> dict:
    """Aggregate labeling-status counts for a project's samples.

    Buckets are mutually exclusive and sum to ``total``:
    ``labeled + unlabeled + background + disabled + non_annotatable == total``.
    """
    total = labeled = unlabeled = background = disabled = non_annotatable = 0
    per_split: dict[str, dict[str, int]] = {}

    for sample in samples:
        total += 1

        st = getattr(sample, "sample_type", None)
        split = str(getattr(st, "value", st) or "")
        bucket = per_split.setdefault(split, {"total": 0, "unlabeled": 0})
        bucket["total"] += 1

        meta = _meta(sample)
        if meta.get("is_disabled"):
            disabled += 1
        elif _ext(getattr(sample, "filename", "")) not in IMAGE_EXTS:
            non_annotatable += 1
        elif meta.get(IS_BACKGROUND_KEY):
            background += 1
        elif is_sample_unlabeled(sample):
            unlabeled += 1
            bucket["unlabeled"] += 1
        else:
            labeled += 1

    return {
        "total": total,
        "labeled": labeled,
        "unlabeled": unlabeled,
        "background": background,
        "disabled": disabled,
        "non_annotatable": non_annotatable,
        "per_split": per_split,
    }
