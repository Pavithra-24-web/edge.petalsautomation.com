"""
Sensor-frame decoding and validation — Batch 3.2.

Kept out of `websocket/sensor_frames.py` so the WS handler stays a thin
orchestrator (motion_phase1.md §7: `receive -> decode_frame -> validate_frame
-> broadcast_frame`). Pure functions over plain dicts/lists only — no
WebSocket, no DB, no Studio broadcast — so both halves are unit-testable in
isolation.

Wire shape mirrors every other device->server message already on this
connection (`inference-result`, `status`): `{"type": "sensor-frame",
"payload": {...}}` (see test_motion_streams.py's
`test_ws_device_routes_sensor_frame_to_motion_handler`, the only place this
message shape was previously exercised). `payload.values` is the one
required field; `payload.sensor` is optional and, when present, is
cross-checked against the device's declared sensor inventory
(`sensor_inventory.normalise`) in `validate_frame`.
"""
from __future__ import annotations

from numbers import Real
from typing import List, Optional, TypedDict

from app.motion.schemas import SensorDescriptor

# Generous upper bound on axis count — guards against a malformed/hostile
# frame forcing an unbounded per-message allocation. No real IMU/sensor
# declares anywhere near this many axes.
MAX_VALUES = 64


class DecodedFrame(TypedDict):
    sensor: Optional[str]
    values: List[float]


def decode_frame(msg: dict) -> Optional[DecodedFrame]:
    """
    Pull `{sensor, values}` out of a `sensor-frame` message.

    Returns None for anything that doesn't match the expected shape —
    the caller drops silently rather than raising, matching
    `sensor_inventory.normalise`'s tolerance for a device sending data the
    server can't fully trust.
    """
    if msg.get("type") != "sensor-frame":
        return None

    payload = msg.get("payload")
    if not isinstance(payload, dict):
        return None

    raw_values = payload.get("values")
    if not isinstance(raw_values, list) or not (0 < len(raw_values) <= MAX_VALUES):
        return None

    values: List[float] = []
    for v in raw_values:
        # bool is a Real subclass in Python; exclude it explicitly so
        # `[True, False]` doesn't silently pass as `[1.0, 0.0]`.
        if isinstance(v, bool) or not isinstance(v, Real):
            return None
        fv = float(v)
        if fv != fv or fv in (float("inf"), float("-inf")):  # NaN / Inf guard
            return None
        values.append(fv)

    sensor = payload.get("sensor")
    if sensor is not None and not isinstance(sensor, str):
        return None

    return {"sensor": sensor, "values": values}


def validate_frame(frame: DecodedFrame, sensors: List[SensorDescriptor]) -> bool:
    """
    Cross-check a decoded frame against the device's declared sensor inventory.

    - Device has no declared sensors (empty inventory, or predates the typed
      `sensors` shape) -> accept on shape alone; nothing to check against.
    - Frame doesn't name a sensor -> accept; a single-sensor device has no
      need to repeat its name on every frame.
    - Frame names a sensor -> it must match a declared one by name, and if
      that descriptor declares axes, the value count must match exactly.
    """
    if not sensors or frame["sensor"] is None:
        return True

    match = next((s for s in sensors if s.name == frame["sensor"]), None)
    if match is None:
        return False
    if match.axes and len(match.axes) != len(frame["values"]):
        return False
    return True
