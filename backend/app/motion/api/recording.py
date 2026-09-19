"""
Recording stop route — Batch 5.

Follows the same route shape as `motion/api/streams.py` (device-owner check
via the shared `svc.get_active_device` + `assert_project_owner` primitives,
not `endpoints/devices.py`'s private `_require_device_owner`) but stops the
*existing* Phase 4 `start-sampling` recording rather than a debug stream.

Mounted at prefix "/devices" by H1, so this reads as
`POST /devices/{device_pk}/stop-sampling` on the wire — beside the existing
`POST /devices/{device_pk}/start-sampling` in `endpoints/devices.py`.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.core.auth import get_current_user
from app.core.authz import assert_project_owner
from app.core.database import get_db
from app.models.user import User
from app.motion.services import recording
from app.services import devices as svc

router = APIRouter()


def _require_device_owner(db: Session, device_pk: str, user: User):
    try:
        dev = svc.get_active_device(db, device_pk)
    except KeyError:
        raise HTTPException(status_code=404, detail="Device not found")
    assert_project_owner(db, dev.project_id, user)
    return dev


@router.post("/{device_pk}/stop-sampling", status_code=202)
async def stop_sampling(
    device_pk: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Request that an in-flight recording (started via
    `POST /{device_pk}/start-sampling`) stop.

    202 means the stop was accepted and dispatched — not that the device has
    finished stopping. The protocol defines no ack for `stop-sample` (see
    `motion/services/recording.py`'s module docstring for the protocol
    evidence), so the response can only report "stop requested", not
    "recording stopped". Returns 409 if the device is not currently sampling
    (offline, or idle/inference). The device's own `sample-stopped` lifecycle
    event — sent once it finishes uploading whatever it recorded so far —
    still lands through the unmodified sample-* / ingestion path regardless
    of how the stop was triggered.
    """
    dev = _require_device_owner(db, device_pk, current_user)

    if not dev.device_id:
        raise HTTPException(
            status_code=409,
            detail="Device has no hardware ID — cannot dispatch command",
        )

    await recording.stop_recording(dev.project_id, dev.device_id)
    return {"status": "stop_requested", "device_id": dev.id}
