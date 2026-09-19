"""
Sensor stream lifecycle over the shared StreamStore — Batch 3.1.

Mirrors the snapshot/inference start/stop shape already established in
`endpoints/devices.py` (`stream_snapshot_start`, `stream_inference_start`),
but adds the one thing those two never needed: a conflict check. Today
`StreamStore.start()` silently overwrites any existing stream for a device
(motion_phase1.md §13 F4) — snapshot and inference never collided in
practice, so nothing enforced it. A sensor preview is the first stream type
introduced *after* that gap was documented, so this module closes it instead
of repeating it: starting a sensor stream while a different stream type is
already active raises 409 rather than silently stealing the device's one
stream slot.

No frame handling here — `start`/`stop` only. Frame validation, decoding and
broadcast are Batch 3.2 (`frame_codec.py`, `websocket/sensor_frames.py`).
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from fastapi import HTTPException

from app.realtime.connection_manager import manager
from app.realtime.studio_events import (
    emit,
    EVENT_SENSOR_STARTED,
    EVENT_SENSOR_STOPPED,
)
from app.services.stream_store import stream_store, StreamState, StreamType


def _build_start_sensor_stream(
    sensor: Optional[str],
    frequency: Optional[float],
    sample_length_ms: Optional[int],
) -> dict:
    payload: Dict[str, Any] = {}
    if sensor is not None:
        payload["sensor"] = sensor
    if frequency is not None:
        payload["frequency"] = frequency
    if sample_length_ms is not None:
        payload["sample_length_ms"] = sample_length_ms
    return {"type": "start-sensor-stream", "payload": payload}


def _build_stop_sensor_stream() -> dict:
    return {"type": "stop-sensor-stream", "payload": {}}


async def start_sensor_stream(
    project_id: str,
    device_id: str,
    *,
    sensor: Optional[str] = None,
    frequency: Optional[float] = None,
    sample_length_ms: Optional[int] = None,
) -> StreamState:
    """Start a sensor stream, or raise 409 if a conflicting stream is active."""
    existing = stream_store.get(device_id)
    if existing is not None and existing.stream_type != StreamType.sensor:
        raise HTTPException(
            status_code=409,
            detail=f"Device already has an active {existing.stream_type.value} stream",
        )

    state = stream_store.start(
        project_id, device_id, StreamType.sensor,
        sensor=sensor, frequency=frequency, sample_length_ms=sample_length_ms,
    )
    await manager.send_json(
        project_id, device_id,
        _build_start_sensor_stream(sensor, frequency, sample_length_ms),
    )
    await emit(
        project_id, EVENT_SENSOR_STARTED,
        {"stream_id": state.stream_id}, device_id=device_id,
    )
    return state


async def stop_sensor_stream(project_id: str, device_id: str) -> Optional[StreamState]:
    """Stop the active sensor stream. Returns None if none is active (not this device's)."""
    existing = stream_store.get(device_id)
    if existing is None or existing.stream_type != StreamType.sensor:
        return None

    state = stream_store.stop(device_id)
    await manager.send_json(project_id, device_id, _build_stop_sensor_stream())
    await emit(
        project_id, EVENT_SENSOR_STOPPED,
        {"stream_id": state.stream_id}, device_id=device_id,
    )
    return state
