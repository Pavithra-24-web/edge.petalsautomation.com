"""
Sensor stream start/stop routes — Batch 3.1.

Follows the same route shape as `endpoints/devices.py`'s
`stream_snapshot_start`/`stream_inference_start`/`stream_stop` (device-owner
check, offline guard, `StreamStore`, a studio event) but lives in the motion
package per motion_phase1.md §3: shared infra (`app/services/devices.py`,
`app/core/authz.py`, `app/realtime/*`) is reused, not forked, and motion picks
up its own route file instead of growing `endpoints/devices.py`.

Mounted at prefix "/devices" by H1, so paths below read as
`/devices/{device_pk}/streams/sensor/...` on the wire — identical shape to
the existing snapshot/inference routes one file over.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.core.auth import get_current_user
from app.core.authz import assert_project_owner
from app.core.database import get_db
from app.models.user import User
from app.motion.schemas import SensorStreamStartRequest
from app.motion.services import stream_control
from app.realtime.connection_manager import manager
from app.services import devices as svc

router = APIRouter()


def _require_device_owner(db: Session, device_pk: str, user: User):
    """Fetch an active device and assert the caller owns its project.

    Composed from the two public helpers `endpoints/devices.py` itself wraps
    (`svc.get_active_device`, `authz.assert_project_owner`) rather than
    importing that module's underscore-prefixed `_require_device_owner` —
    reuse of the shared primitives, not a reach into another endpoint file's
    private internals.
    """
    try:
        dev = svc.get_active_device(db, device_pk)
    except KeyError:
        raise HTTPException(status_code=404, detail="Device not found")
    assert_project_owner(db, dev.project_id, user)
    return dev


@router.post("/{device_pk}/streams/sensor/start", status_code=202)
async def stream_sensor_start(
    device_pk: str,
    body: SensorStreamStartRequest = SensorStreamStartRequest(),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    dev = _require_device_owner(db, device_pk, current_user)
    if not manager.is_connected(dev.project_id, dev.device_id):
        raise HTTPException(status_code=409, detail="Device is offline")

    state = await stream_control.start_sensor_stream(
        dev.project_id, dev.device_id,
        sensor=body.sensor,
        frequency=body.frequency,
        sample_length_ms=body.sample_length_ms,
    )
    return {"stream_id": state.stream_id, "stream_type": "sensor"}


@router.post("/{device_pk}/streams/sensor/stop", status_code=202)
async def stream_sensor_stop(
    device_pk: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    dev = _require_device_owner(db, device_pk, current_user)
    state = await stream_control.stop_sensor_stream(dev.project_id, dev.device_id)
    if state is None:
        raise HTTPException(status_code=409, detail="No active sensor stream for this device")
    return {"status": "stopped"}
