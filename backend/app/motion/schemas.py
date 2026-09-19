"""
Motion domain schemas — Phase 1.1, extended by each later sub-phase.
"""
from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field


class SensorDescriptor(BaseModel):
    """
    Normalised shape of one entry in `Device.sensors` (a free-form JSON
    column — see app/models/user.py). Produced by
    `app.motion.services.sensor_inventory.normalise`; consumed by 1.1's
    device-inventory display and, later, 1.2's frame validation and 1.4's
    request validation.
    """
    model_config = ConfigDict(extra="ignore")

    name: str
    type: Optional[str] = None
    freq_hz: Optional[float] = None
    axes: List[str] = Field(default_factory=list)


class SensorStreamStartRequest(BaseModel):
    """
    Body for `POST /devices/{pk}/streams/sensor/start` — Batch 3.1.

    Every field is optional: a device with one declared sensor needs none of
    them, matching `_InferenceStartBody`'s existing optional-everything shape
    in `endpoints/devices.py`. Frame-level use of `sensor`/`frequency` against
    the device's declared axes is Batch 3.2 (`frame_codec.py`); this batch only
    carries the values through to the device-facing `start-sensor-stream`
    command and the `StreamState` bookkeeping.
    """
    model_config = ConfigDict(extra="ignore")

    sensor: Optional[str] = None
    frequency: Optional[float] = None
    sample_length_ms: Optional[int] = None
