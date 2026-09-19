"""
Non-maximum suppression for object detection bounding boxes.

Backend-agnostic — no coupling to ML frameworks, databases, or HTTP.

Public API
----------
apply_nms(detections, iou_threshold) -> List[Detection]
"""
from __future__ import annotations

from typing import List

from .types import Detection


# ─── Geometry helpers ─────────────────────────────────────────────────────────

def _iou(a: Detection, b: Detection) -> float:
    ix1 = max(a.x1, b.x1)
    iy1 = max(a.y1, b.y1)
    ix2 = min(a.x2, b.x2)
    iy2 = min(a.y2, b.y2)

    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih

    area_a = (a.x2 - a.x1) * (a.y2 - a.y1)
    area_b = (b.x2 - b.x1) * (b.y2 - b.y1)
    union = area_a + area_b - inter

    return inter / union if union > 0.0 else 0.0


def _is_valid(d: Detection) -> bool:
    """Box has positive area in both axes."""
    return d.x2 > d.x1 and d.y2 > d.y1


# ─── NMS ──────────────────────────────────────────────────────────────────────

def apply_nms(
    detections: List[Detection],
    iou_threshold: float = 0.45,
) -> List[Detection]:
    """
    Per-class non-maximum suppression.

    Boxes with x2 <= x1 or y2 <= y1 are silently dropped before NMS.
    Within each class bucket, detections are sorted confidence-descending;
    any box with IoU >= iou_threshold against a kept box is suppressed.

    The output is ordered by class name (lexicographic) then confidence
    descending, ensuring deterministic results for identical inputs.
    """
    valid = [d for d in detections if _is_valid(d)]

    by_class: dict[str, list[Detection]] = {}
    for d in valid:
        by_class.setdefault(d.class_name, []).append(d)

    result: list[Detection] = []
    for cls in sorted(by_class):
        bucket = sorted(by_class[cls], key=lambda d: d.confidence, reverse=True)
        suppressed = [False] * len(bucket)
        for i in range(len(bucket)):
            if suppressed[i]:
                continue
            result.append(bucket[i])
            for j in range(i + 1, len(bucket)):
                if not suppressed[j] and _iou(bucket[i], bucket[j]) >= iou_threshold:
                    suppressed[j] = True

    return result
