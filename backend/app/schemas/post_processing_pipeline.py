"""
Pydantic schemas for the post-processing pipeline (Phases 1-3).

Covers PostProcessingSettings (per-project config), ProcessingJob
(individual job run records), and the detection preview contract.
These schemas are intentionally separate from the legacy
post_processing.py endpoint schemas which remain model_metadata-based.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator, computed_field


# ─── PostProcessingSettings ───────────────────────────────────────────────────

class PostProcessingSettingsResponse(BaseModel):
    id: Optional[str] = None          # None when returning schema defaults (no DB row yet)
    project_id: str
    impulse_id: Optional[str] = None
    enabled: bool
    threshold: float
    tracking_enabled: bool
    keep_grace: int
    max_observations: int
    class_filter: List[str]
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    model_config = ConfigDict(from_attributes=True)


class PostProcessingSettingsUpdate(BaseModel):
    enabled: Optional[bool] = None
    threshold: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    tracking_enabled: Optional[bool] = None
    keep_grace: Optional[int] = Field(default=None, ge=0)
    max_observations: Optional[int] = Field(default=None, ge=1)
    class_filter: Optional[List[str]] = None


# ─── ProcessingJob ────────────────────────────────────────────────────────────

class ProcessingJobResponse(BaseModel):
    id: str
    project_id: str
    status: str
    input_video_path: str
    output_video_path: Optional[str]
    error_message: Optional[str]
    inference_time_ms: Optional[float]
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)

    @computed_field  # type: ignore[misc]
    @property
    def output_ready(self) -> bool:
        """True when the job is complete and an output video is available."""
        return self.status == "complete" and self.output_video_path is not None


# ─── Detection preview (Phase 3) ─────────────────────────────────────────────

class RawDetectionInput(BaseModel):
    """A single detection as provided by the caller (normalized coordinates)."""
    x1: float = Field(ge=0.0, le=1.0)
    y1: float = Field(ge=0.0, le=1.0)
    x2: float = Field(ge=0.0, le=1.0)
    y2: float = Field(ge=0.0, le=1.0)
    class_name: str
    confidence: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _box_has_positive_area(self) -> "RawDetectionInput":
        if self.x2 <= self.x1:
            raise ValueError("x2 must be greater than x1")
        if self.y2 <= self.y1:
            raise ValueError("y2 must be greater than y1")
        return self


class DetectionOut(BaseModel):
    """A single detection as returned after pipeline processing."""
    x1: float
    y1: float
    x2: float
    y2: float
    class_name: str
    confidence: float
    object_id: Optional[int] = None
    observations: Optional[int] = None


class TriggerFromSampleRequest(BaseModel):
    """Request body for the sample-based video trigger endpoint (Phase 7)."""
    sample_id: str
    impulse_id: Optional[str] = None


class PreviewRequest(BaseModel):
    """Request body for the post-processing preview endpoint."""
    detections: List[RawDetectionInput]
    tracker_state: Optional[Dict[str, Any]] = None   # prior-frame state, None = fresh


class PreviewResponse(BaseModel):
    """Full pipeline breakdown returned by the preview endpoint."""
    input_detections: List[DetectionOut]
    filtered_detections: List[DetectionOut]    # after confidence + class filter
    post_nms_detections: List[DetectionOut]    # after NMS
    final_detections: List[DetectionOut]        # after tracking (or same as post_nms)
    tracker_state: Dict[str, Any]               # pass back as tracker_state next frame
