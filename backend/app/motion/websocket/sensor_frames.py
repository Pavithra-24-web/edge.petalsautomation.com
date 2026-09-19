"""
`sensor-frame` WS delegation target — Batch 3.1 (infrastructure only).

`handle()` is the stable entry point H4 (`ws_device.py`) delegates to for
every `sensor-frame` message on the existing hello/heartbeat session. This
batch only proves that routing path exists end to end — a device's
`sensor-frame` message reaches this function without a second WebSocket, a
new route, or any change to `ws_device.py`'s message loop beyond the one
`elif` branch.

Frame validation against the device's declared axes (`frame_codec.py`),
decoding and studio broadcast (`EVENT_SENSOR_FRAME`) are Batch 3.2. Takes no
`db` argument on purpose — sensor preview is never persisted, matching the
`_emit_snapshot_frame` precedent this handler will follow once Batch 3.2
fills it in.
"""
from __future__ import annotations

import logging

from app.realtime.connection_manager import DeviceSession

logger = logging.getLogger(__name__)


async def handle(session: DeviceSession, msg: dict) -> None:
    logger.debug(
        "sensor-frame received device_id=%s (Batch 3.2 processing not yet wired)",
        session.device_id,
    )
