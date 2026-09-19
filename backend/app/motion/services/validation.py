"""
Motion-shaped request validation — Batch 7.

Rules here are *not* device capabilities (that's H7's job in shared
`app/services/devices.py::assert_frequency_supported`) — they're shape and
consistency checks specific to a motion sensor-stream request: does the named
sensor actually exist in the device's own declared inventory, does the
device's declared inventory even make internal sense (no sensor name mapped
to two different axis counts), is a supplied `sample_length_ms` within a sane
recording window. Device-capability enforcement (frequency) stays in
`app/services/devices.py` per motion_phase1.md §3.3; this module never
branches on `project_type` either — every check here dispatches on the
device's own declared `sensors` or on the request body already in hand.

Kept tolerant in the same spirit as `sensor_inventory.normalise` and
`frame_codec.validate_frame`: a device that hasn't declared a rich sensor
inventory yet is never blocked by these checks, only one that declares
something and then contradicts or is contradicted by the request.
"""
from __future__ import annotations

from typing import List, Optional

from fastapi import HTTPException

from app.motion.schemas import SensorDescriptor

MIN_LENGTH_MS = 100
MAX_LENGTH_MS = 120_000


def resolve_sensor(
    sensors: List[SensorDescriptor],
    sensor_name: Optional[str],
) -> Optional[SensorDescriptor]:
    """
    Resolve `sensor_name` against a device's declared sensor inventory.

    - No name given, or device has no declared inventory -> None; nothing to
      resolve against, so the caller proceeds unchecked (matches
      `frame_codec.validate_frame`'s existing tolerance for a device that
      predates the typed `sensors` shape).
    - Name given and the device has a declared inventory -> the matching
      descriptor, or a 422 if no sensor by that name is declared.
    """
    if not sensor_name or not sensors:
        return None
    match = next((s for s in sensors if s.name == sensor_name), None)
    if match is None:
        raise HTTPException(
            status_code=422,
            detail=f"Device has no declared sensor named '{sensor_name}'",
        )
    return match


def assert_axis_count_consistent(sensors: List[SensorDescriptor]) -> None:
    """
    422 if a device's own declared inventory names the same sensor twice with
    two different axis counts.

    `frame_codec.validate_frame` resolves a frame's sensor by name via
    `next((s for s in sensors if s.name == frame["sensor"]), None)` — the
    *first* match wins. A device record with two differently-shaped entries
    under the same name would make frame validation silently
    non-deterministic (whichever entry sorts first decides every frame's
    accepted axis count) rather than loudly wrong, so this is caught here,
    at stream-start time, instead.
    """
    seen: dict[str, int] = {}
    for s in sensors:
        if not s.axes:
            continue
        axis_count = len(s.axes)
        prior = seen.get(s.name)
        if prior is not None and prior != axis_count:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"Device declares inconsistent axis counts for sensor "
                    f"'{s.name}' ({prior} vs {axis_count})"
                ),
            )
        seen[s.name] = axis_count


def assert_length_ms_bounds(length_ms: Optional[int]) -> None:
    """422 if a supplied sample_length_ms falls outside a sane recording window."""
    if length_ms is None:
        return
    if not (MIN_LENGTH_MS <= length_ms <= MAX_LENGTH_MS):
        raise HTTPException(
            status_code=422,
            detail=(
                f"sample_length_ms must be between {MIN_LENGTH_MS} and "
                f"{MAX_LENGTH_MS} — got {length_ms}"
            ),
        )
