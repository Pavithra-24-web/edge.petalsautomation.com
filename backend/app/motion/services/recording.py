"""
Recording (stop-sampling) lifecycle — Batch 5.

Mirrors `stream_control.py`'s start/stop shape (device-session check, a
fire-and-forget WS command, a studio event) but for the *existing* Phase 4
sampling pipeline (`services/sampling.py` + `endpoints/devices.py`'s
`start-sampling`) rather than a new one. Recording state lives on the
device's `ConnectionManager` session (`DeviceMode.sampling`), exactly as
`start-sampling` already sets it — not on `StreamStore`, which stays
reserved for the debug/preview streams (snapshot, inference, sensor) and is
reconciled with recording in Batch 7, not here.

`stop-sample` is fire-and-forget (`manager.send_json`, no ack awaited) —
confirmed correct against the actual protocol, not just matched-for-style to
the other stop-* commands (final protocol review, motion_phase1.md §9):

  - `device_client/PROTOCOL.md` documents `start-sample` only; there is no
    `stop-sample` ack/response defined anywhere in the protocol.
  - `unoq/runtime/protocol.py` is the one existing device implementation with
    any `stop-sample` handling at all, and it is a deliberate no-op:
    `elif t == "stop-sample": pass  # _sample_real runs to completion; stop
    is advisory on short windows` (protocol.py:251-252). It sends nothing
    back — no ack, no early `sample-stopped`. `_sample_real` (protocol.py:405)
    sends its own `sample-stopped` only after its own `asyncio.sleep(length)`
    + upload complete, still carrying the *original* `start-sample`
    correlationId, never the stop command's. A `send_and_await_ack` on the
    stop command's own correlationId would therefore never resolve on this
    (the only real) implementation — it would time out every time.
  - `device_client/client.py` doesn't handle `stop-sample` at all (falls
    through to the unhandled-type debug log), and `esp32_client.cpp` has no
    `stop-sample` handling either.

So the caller's 409/202 decision is correctly driven by the server's own
tracked mode, not a device round-trip that the protocol doesn't support. The
in-flight upload the device is already mid-flight on (if any) still lands
through the unmodified sample-* / ingestion path once it finishes — verified
safe: `ingestion_auth.py` / `services/ingestion.py` authenticate purely via
`x-api-key` + per-sample HMAC token, never read `ConnectionManager` mode, so
flipping mode to idle here cannot cause that upload to be rejected. The
device's eventual real `sample-stopped` message is handled unconditionally by
`ws_device.py`'s pre-existing `_handle_sample_lifecycle` regardless of the
mode already being idle (`manager.resolve_ack` on an already-consumed/absent
correlationId is a documented no-op, not an error), and each `start-sampling`
call issues its own distinct, independently-valid `SampleHmacRecord` token
(`sample_hmac_store.issue`), so a still-in-flight stopped recording's upload
never collides with a new one.
"""
from __future__ import annotations

from fastapi import HTTPException

from app.realtime.commands import build_stop_sample
from app.realtime.connection_manager import DeviceMode, DeviceSession, manager
from app.realtime.studio_events import emit, EVENT_SAMPLE_STOPPED


async def stop_recording(project_id: str, device_id: str) -> DeviceSession:
    """
    Request that an in-flight recording stop. Raises 409 if the device isn't
    sampling.

    This *requests* the stop and flips the server's own tracked mode back to
    idle immediately — it does not confirm the device actually stopped, since
    the protocol defines no ack for `stop-sample` (see module docstring). The
    device's own `sample-stopped` message, whenever it arrives, is what
    confirms the recording actually ended and its upload landed.
    """
    session = manager.get(project_id, device_id)
    if session is None:
        raise HTTPException(status_code=409, detail="Device is not connected")
    if session.mode != DeviceMode.sampling:
        raise HTTPException(status_code=409, detail="Device is not sampling")

    await manager.send_json(project_id, device_id, build_stop_sample())
    manager.update_mode(project_id, device_id, DeviceMode.idle)
    await emit(project_id, EVENT_SAMPLE_STOPPED, {}, device_id=device_id)
    return session
