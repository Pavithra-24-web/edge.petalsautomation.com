"""PXE result normalisation and postprocessing for the UnoQ runtime."""
from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger("unoq")


class _PostprocessEngineMixin:
    """Mixin providing result-normalisation for UnoQRuntime."""

    def _normalize_pxe_result(self, raw: dict, manifest: dict) -> Optional[dict]:
        """Convert EI stdio-JSONL result → same dict shape as .pe inference output."""
        if not raw.get("result"):
            logger.warning("[pxe] classify error: %s", raw.get("error"))
            return None

        manifest_model_type = (
            manifest.get("model_type")
            or manifest.get("output_type")
            or "classification"
        )

        if "detections" in raw or "is_detection" in raw:
            out = {k: v for k, v in raw.items() if not k.startswith("_")}
            out.setdefault("is_detection", bool(raw.get("detections") is not None))
            out.setdefault("is_fomo", False)
            out.setdefault("model_type", manifest_model_type)
            return out

        if "bounding_boxes" in raw:
            bboxes = raw["bounding_boxes"]
            input_shape = manifest.get("input_shape", [96, 96, 3])
            img_h = int(input_shape[0]) if len(input_shape) >= 1 else 96
            img_w = int(input_shape[1]) if len(input_shape) >= 2 else 96
            return {
                "is_fomo":    True,
                "is_detection": False,
                "model_type": "detection_heatmap",
                "detections": [
                    {
                        "label":      b["label"],
                        "confidence": float(b.get("value", b.get("confidence", 0.0))),
                        "bbox": {
                            "x1": float(b["x"]) / img_w,
                            "y1": float(b["y"]) / img_h,
                            "x2": (float(b["x"]) + float(b["width"])) / img_w,
                            "y2": (float(b["y"]) + float(b["height"])) / img_h,
                        },
                    }
                    for b in bboxes
                ],
                "count":  len(bboxes),
                "timing": raw.get("timing", {}),
            }

        classification = raw.get("classification", {})
        if not classification:
            return None

        if manifest_model_type == "yolo_pro_detection":
            return {
                "detections": [],
                "count": 0,
                "is_fomo": False,
                "is_detection": True,
                "model_type": "yolo_pro_detection",
                "debug": {"stale_pxe_artifact": True},
                "timing": raw.get("timing", {}),
            }

        top_label  = max(classification, key=lambda k: classification[k])
        confidence = classification[top_label]
        return {
            "label":       top_label,
            "confidence":  confidence,
            "predictions": [{"label": k, "confidence": v} for k, v in classification.items()],
            "timing":      raw.get("timing", {}),
            "is_fomo":     False,
            "is_detection": False,
            "model_type":  manifest_model_type,
        }

    def _apply_pxe_tracking(self, result: Optional[dict]) -> Optional[dict]:
        """Apply PxeTracker to YOLO detection results when tracking is enabled.

        Tracking is skipped for:
        - None results (inference error)
        - FOMO results (is_fomo=True) — heatmap centroids are not stable bboxes
        - Classification results (is_detection=False) — no bounding boxes

        tracker_iou_threshold and center_dist_threshold use code-level defaults
        (0.3 and 0.2) because these fields are not persisted in postprocess_config.json.
        """
        if result is None:
            return None
        if not result.get("is_detection") or result.get("is_fomo"):
            return result

        pp_config = getattr(self, "_pp_config", {})
        if not pp_config.get("tracking_enabled"):
            return result

        from .tracker import PxeTracker
        tracker = PxeTracker(
            keep_grace=int(pp_config.get("keep_grace", 3)),
            max_observations=int(pp_config.get("max_observations", 5)),
        )
        tracker_state = getattr(self, "_tracker_state", {})
        tracked, new_state = tracker.update(result.get("detections", []), tracker_state)
        if hasattr(self, "_tracker_state"):
            self._tracker_state = new_state
        logger.debug(
            "[pxe] tracking: %d detections (%d carried)",
            len(tracked), sum(1 for d in tracked if d.get("is_carried")),
        )
        return {**result, "detections": tracked, "count": len(tracked)}
