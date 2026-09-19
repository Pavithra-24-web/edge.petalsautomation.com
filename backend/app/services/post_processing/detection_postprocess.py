"""
Detection post-processing pipeline orchestrator.

Runs the per-frame pipeline in order:
  1. Confidence filter  (threshold)
  2. Class filter       (class_filter list)
  3. NMS                (iou-based duplicate removal)
  4. Tracking           (stable object IDs, optional)

Consumes settings from the Phase 1/2 contract fields:
  enabled, threshold, tracking_enabled, keep_grace,
  max_observations, class_filter

Public API
----------
run_pipeline(detections, config, tracker_state) -> PipelineResult
PipelineConfig.from_orm_or_dict(settings)       -> PipelineConfig
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, List, Optional

from .nms import apply_nms
from .tracker import SimpleTracker
from .types import Detection

logger = logging.getLogger(__name__)


# ─── Config ───────────────────────────────────────────────────────────────────

@dataclass
class PipelineConfig:
    enabled: bool = True
    threshold: float = 0.5
    tracking_enabled: bool = False
    keep_grace: int = 3
    max_observations: int = 5
    class_filter: List[str] = field(default_factory=list)
    nms_iou_threshold: float = 0.45     # duplicate-suppression threshold
    tracker_iou_threshold: float = 0.30  # association threshold

    @classmethod
    def from_orm_or_dict(cls, settings: Any) -> "PipelineConfig":
        """
        Build from an ORM PostProcessingSettings row or a plain dict
        (e.g. the defaults returned when no row exists yet).
        """
        if hasattr(settings, "__dict__"):
            d = {k: v for k, v in vars(settings).items() if not k.startswith("_")}
        else:
            d = dict(settings)
        # Use `or` fallback so that NULL values in the DB (key present, value None)
        # are treated the same as a missing key rather than crashing on int(None).
        return cls(
            enabled=d.get("enabled") if d.get("enabled") is not None else True,
            threshold=d.get("threshold") if d.get("threshold") is not None else 0.5,
            tracking_enabled=d.get("tracking_enabled") if d.get("tracking_enabled") is not None else False,
            keep_grace=int(d.get("keep_grace") or 3),
            max_observations=int(d.get("max_observations") or 5),
            class_filter=list(d.get("class_filter") or []),
        )


# ─── Result ───────────────────────────────────────────────────────────────────

@dataclass
class PipelineResult:
    input_detections: List[Detection]
    filtered_detections: List[Detection]   # after threshold + class filter
    post_nms_detections: List[Detection]   # after NMS
    final_detections: List[Detection]       # after tracking (or same as post_nms)
    tracker_state: dict                     # serializable; pass back in next frame


# ─── Pipeline ─────────────────────────────────────────────────────────────────

def run_pipeline(
    detections: List[Detection],
    config: PipelineConfig,
    tracker_state: Optional[dict] = None,
) -> PipelineResult:
    """
    Execute the full post-processing pipeline for a single frame.

    When config.enabled is False the input is returned unchanged through
    every stage; the tracker state is passed through unmodified.
    """
    if tracker_state is None:
        tracker_state = {}

    input_dets = list(detections)

    if not config.enabled:
        logger.debug(
            "pipeline disabled — passing %d raw detections through unchanged",
            len(input_dets),
        )
        return PipelineResult(
            input_detections=input_dets,
            filtered_detections=input_dets,
            post_nms_detections=input_dets,
            final_detections=input_dets,
            tracker_state=tracker_state,
        )

    logger.debug(
        "pipeline: %d input detections | threshold=%.2f tracking=%s "
        "keep_grace=%d max_obs=%d class_filter=%r",
        len(input_dets), config.threshold, config.tracking_enabled,
        config.keep_grace, config.max_observations, config.class_filter,
    )

    # 1. Confidence filter
    filtered: List[Detection] = [
        d for d in input_dets if d.confidence >= config.threshold
    ]
    logger.debug(
        "pipeline: after threshold filter: %d/%d detections kept (threshold=%.2f)",
        len(filtered), len(input_dets), config.threshold,
    )

    # 2. Class filter (skip when list is empty = allow all)
    if config.class_filter:
        allowed = set(config.class_filter)
        before_cf = len(filtered)
        filtered = [d for d in filtered if d.class_name in allowed]
        logger.debug(
            "pipeline: class filter applied — allowed=%r  %d→%d detections",
            sorted(allowed), before_cf, len(filtered),
        )
    else:
        logger.debug("pipeline: class filter empty — all classes pass")

    # 3. NMS
    post_nms = apply_nms(filtered, iou_threshold=config.nms_iou_threshold)
    logger.debug(
        "pipeline: after NMS: %d/%d detections kept (iou_threshold=%.2f)",
        len(post_nms), len(filtered), config.nms_iou_threshold,
    )

    # 4. Tracking
    if config.tracking_enabled:
        logger.debug(
            "pipeline: tracking enabled — keep_grace=%d max_observations=%d",
            config.keep_grace, config.max_observations,
        )
        tracker = SimpleTracker(
            iou_threshold=config.tracker_iou_threshold,
            keep_grace=config.keep_grace,
            max_observations=config.max_observations,
        )
        final, new_tracker_state = tracker.update(post_nms, tracker_state)
        logger.debug(
            "pipeline: after tracking: %d detections (%d carried)",
            len(final), sum(1 for d in final if getattr(d, "is_carried", False)),
        )
    else:
        logger.debug("pipeline: tracking disabled — passing post-NMS detections as final")
        final = post_nms
        new_tracker_state = tracker_state

    return PipelineResult(
        input_detections=input_dets,
        filtered_detections=filtered,
        post_nms_detections=post_nms,
        final_detections=final,
        tracker_state=new_tracker_state,
    )
