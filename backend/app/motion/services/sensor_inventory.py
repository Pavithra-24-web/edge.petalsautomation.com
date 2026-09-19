"""
Sensor inventory normalisation — Phase 1.1.

`Device.sensors` is free-form JSON: `[{name, type, freq_hz, axes}]` per the
model comment (app/models/user.py:434), but rows written by the CSV/serial
forwarder path carry bare strings instead — the frontend's `sensorNames()`
(CollectFromDevice.tsx) has tolerated both since Phase 0. `normalise()` is
the backend-side counterpart, giving motion UI (MotionDeviceDetails) and
later sub-phases (1.2's frame_codec, 1.4's validation) one typed shape to
read instead of re-deriving the tolerance per caller.
"""
from __future__ import annotations

from typing import Any, List, Optional

from app.motion.schemas import SensorDescriptor


def normalise(raw: Optional[List[Any]]) -> List[SensorDescriptor]:
    """
    Turn a `Device.sensors` list into `SensorDescriptor` rows.

    Tolerates the same two shapes the frontend already does:
      - a bare string               -> SensorDescriptor(name=<string>)
      - a dict with at least `name` -> parsed field-by-field, extras dropped

    An entry with no derivable name is skipped rather than raising: this
    reads data a device already self-reported, so a malformed entry should
    not break rendering the rest of the inventory.
    """
    if not raw:
        return []

    out: List[SensorDescriptor] = []
    for entry in raw:
        if isinstance(entry, str):
            if entry:
                out.append(SensorDescriptor(name=entry))
            continue
        if isinstance(entry, dict):
            name = entry.get("name")
            if not isinstance(name, str) or not name:
                continue
            axes = entry.get("axes")
            freq_hz = entry.get("freq_hz")
            out.append(SensorDescriptor(
                name=name,
                type=entry.get("type"),
                freq_hz=freq_hz if isinstance(freq_hz, (int, float)) else None,
                axes=[a for a in axes if isinstance(a, str)] if isinstance(axes, list) else [],
            ))
    return out
