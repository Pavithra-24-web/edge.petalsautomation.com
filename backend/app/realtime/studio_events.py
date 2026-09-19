"""
Studio event envelope and broadcast helpers — Phase 5

All events pushed to browser Studio clients use StudioEvent as their
wire format.  Call emit() from any server-side handler to fan-out to
all Studio tabs open on that project.
"""
from __future__ import annotations

from datetime import datetime
from typing import TypedDict

from app.realtime.studio_manager import studio_manager


# ─── Event envelope ───────────────────────────────────────────────────────────

class StudioEvent(TypedDict):
    type:       str           # one of the EVENT_* constants below
    project_id: str
    device_id:  str | None    # None for project-level events
    ts:         str           # datetime.utcnow().isoformat()
    payload:    dict


# ─── Event type constants ─────────────────────────────────────────────────────

# Device lifecycle
EVENT_DEVICE_CONNECTED    = "device.connected"
EVENT_DEVICE_DISCONNECTED = "device.disconnected"
EVENT_DEVICE_MODE_CHANGED = "device.mode_changed"

# Data sampling lifecycle
EVENT_SAMPLE_REQUESTED = "sample.requested"
EVENT_SAMPLE_ACKED     = "sample.acked"
EVENT_SAMPLE_REJECTED  = "sample.rejected"
EVENT_SAMPLE_STARTED   = "sample.started"
EVENT_SAMPLE_STOPPED   = "sample.stopped"
EVENT_SAMPLE_FAILED    = "sample.failed"

# OTA update lifecycle
EVENT_UPDATE_REQUESTED   = "update.requested"
EVENT_UPDATE_ACCEPTED    = "update.accepted"
EVENT_UPDATE_REJECTED    = "update.rejected"
EVENT_UPDATE_DOWNLOADING = "update.downloading"
EVENT_UPDATE_FLASHING    = "update.flashing"
EVENT_UPDATE_DONE        = "update.done"
EVENT_UPDATE_FAILED      = "update.failed"

# Debug stream lifecycle
EVENT_SNAPSHOT_STARTED  = "snapshot.started"
EVENT_SNAPSHOT_FRAME    = "snapshot.frame"
EVENT_SNAPSHOT_STOPPED  = "snapshot.stopped"
EVENT_INFERENCE_STARTED = "inference.started"
EVENT_INFERENCE_RESULT  = "inference.result"
EVENT_INFERENCE_STOPPED = "inference.stopped"
EVENT_STREAM_FAILED     = "stream.failed"
EVENT_SENSOR_STARTED    = "sensor.started"
EVENT_SENSOR_FRAME      = "sensor.frame"
EVENT_SENSOR_STOPPED    = "sensor.stopped"


# ─── Emit helper ──────────────────────────────────────────────────────────────

async def emit(
    project_id: str,
    type: str,
    payload: dict,
    device_id: str | None = None,
) -> None:
    """Build a StudioEvent envelope and broadcast it to all Studio tabs."""
    event: StudioEvent = {
        "type":       type,
        "project_id": project_id,
        "device_id":  device_id,
        "ts":         datetime.utcnow().isoformat(),
        "payload":    payload,
    }
    await studio_manager.broadcast(project_id, event)
