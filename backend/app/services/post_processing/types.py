"""
Shared internal types for the post-processing pipeline.

These dataclasses are used throughout nms.py, tracker.py, and
detection_postprocess.py.  They are intentionally independent of
SQLAlchemy, Pydantic, and any ML framework.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class Detection:
    x1: float
    y1: float
    x2: float
    y2: float
    class_name: str
    confidence: float
    object_id: Optional[int] = None
    observations: Optional[int] = None
    is_carried: Optional[bool] = None  # True during keep_grace window (no live detection)
