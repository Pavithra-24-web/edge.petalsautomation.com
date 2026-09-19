"""
Device management endpoints — Phase 1 + Phase 4.

Phase 4 additions (all additive):
  - _detail / _list_item enriched with is_connected + mode from ConnectionManager
  - _key_summary includes api_key_prefix for client-side identification
  - POST /{device_pk}/start-sampling  — dispatch sample command via Phase 2 WS manager
  - POST /project/{project_id}/socket-token  — issue short-lived Studio WS token
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timedelta
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Response
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.core.auth import get_current_user
from app.core.ingestion_auth import resolve_api_key
from app.core.socket_token_store import socket_token_store
from app.core.storage import storage
from app.models.devices import ProjectDeviceKey
from app.models.user import User, Project, Deployment, DeviceUpdateHistory
from app.realtime.commands import (
    send_and_await_ack,
    send_deployment_update,
    send_start_snapshot,
    send_stop_snapshot,
    send_start_inference_stream,
    send_stop_inference_stream,
)
from app.realtime.connection_manager import DeviceMode, DeviceSession, manager
from app.realtime.studio_events import (
    emit,
    EVENT_SNAPSHOT_STARTED,
    EVENT_SNAPSHOT_STOPPED,
    EVENT_INFERENCE_STARTED,
    EVENT_INFERENCE_STOPPED,
    EVENT_STREAM_FAILED,
)
from app.services.stream_store import stream_store, StreamType
from app.services.compatibility import resolve_latest_compatible_deployment
from app.services.sampling import build_start_sample_command, issue_sample_hmac
from app.schemas.devices import (
    DeviceCreateRequest,
    DeviceDetailResponse,
    DeviceHeartbeatRequest,
    DeviceKeyCreateRequest,
    DeviceKeyResponse,
    DeviceKeySummary,
    DeviceListItem,
    DeviceRenameRequest,
    DeviceUpdateRequest,
    InferenceLogResponse,
    SocketTokenResponse,
    StartSamplingRequest,
    StartSamplingResponse,
)
from app.services import devices as svc

router = APIRouter()

_SOCKET_TOKEN_TTL = 60   # seconds
HEARTBEAT_INTERVAL_MIN_SECS = 10
HEARTBEAT_INTERVAL_MAX_SECS = 30
ONLINE_FRESHNESS_WINDOW_SECS = 45


def _get_preferred_runtime_for_device(impulse_id: str, db: Session, current_user: User) -> dict:
    from app.api.v1.endpoints.inference import get_preferred_runtime

    return get_preferred_runtime(
        impulse_id=impulse_id,
        db=db,
        current_user=current_user,
    )


# ─── Ownership guards ─────────────────────────────────────────────────────────

def _require_project(db: Session, project_id: str, user: User) -> Project:
    """
    Return the Project if owned by `user`.
    Returns 404 in both the missing and the unauthorized case to prevent
    cross-account project enumeration.
    """
    proj = db.query(Project).filter(
        Project.id == project_id,
        Project.owner_id == user.id,
    ).first()
    if not proj:
        raise HTTPException(status_code=404, detail="Project not found")
    return proj


def _require_device_owner(db: Session, device_pk: str, user: User):
    """Fetch an active device and assert the caller owns its project."""
    try:
        dev = svc.get_active_device(db, device_pk)
    except KeyError:
        raise HTTPException(status_code=404, detail="Device not found")
    _require_project(db, dev.project_id, user)
    return dev


# ─── Serialisers ──────────────────────────────────────────────────────────────

def _runtime_fields(dev, session: Optional[DeviceSession]) -> dict:
    """
    Derive runtime presence from the minimum device contract:
      - ONLINE if a recent heartbeat exists within the freshness window
      - OR if a live WS session exists
      - is_connected remains WS-only
    """
    fresh_cutoff = datetime.utcnow() - timedelta(seconds=ONLINE_FRESHNESS_WINDOW_SECS)
    heartbeat_fresh = bool(dev.last_seen and dev.last_seen >= fresh_cutoff)
    return {
        "is_online": heartbeat_fresh or session is not None,
        "is_connected": session is not None,
        "mode": session.mode.value if session else None,
    }


def _session_for(dev) -> Optional[DeviceSession]:
    """Look up a live session for a device by its external hardware ID."""
    if not dev.device_id:
        return None
    return manager.get(dev.project_id, dev.device_id)


def _detail(dev, session: Optional[DeviceSession] = None) -> dict:
    rt = _runtime_fields(dev, session)
    return DeviceDetailResponse(
        id=dev.id,
        project_id=dev.project_id,
        device_id=dev.device_id,
        name=dev.name,
        device_type=dev.device_type,
        connection=dev.connection,
        firmware_version=dev.firmware_version,
        protocol_version=dev.protocol_version,
        ip_address=dev.ip_address,
        supports_snapshot_streaming=dev.supports_snapshot_streaming or False,
        remote_mgmt_host=dev.remote_mgmt_host,
        sensors=dev.sensors or [],
        device_metadata=dev.device_metadata or {},
        last_seen=dev.last_seen,
        created_at=dev.created_at,
        updated_at=dev.updated_at,
        deployment_target=dev.deployment_target,
        device_profile=dev.device_profile,
        installed_deployment_id=dev.installed_deployment_id,
        installed_model_version=dev.installed_model_version,
        is_online=rt["is_online"],
        is_connected=rt["is_connected"],
        mode=rt["mode"],
    ).model_dump(mode="json")


def _list_item(dev, session: Optional[DeviceSession] = None) -> dict:
    rt = _runtime_fields(dev, session)
    return DeviceListItem(
        id=dev.id,
        project_id=dev.project_id,
        device_id=dev.device_id,
        name=dev.name,
        device_type=dev.device_type,
        connection=dev.connection,
        firmware_version=dev.firmware_version,
        ip_address=dev.ip_address,
        last_seen=dev.last_seen,
        created_at=dev.created_at,
        sensors=dev.sensors or [],
        deployment_target=dev.deployment_target,
        device_profile=dev.device_profile,
        installed_deployment_id=dev.installed_deployment_id,
        installed_model_version=dev.installed_model_version,
        is_online=rt["is_online"],
        is_connected=rt["is_connected"],
        mode=rt["mode"],
    ).model_dump(mode="json")


def _key_response(k) -> dict:
    return DeviceKeyResponse(
        id=k.id,
        project_id=k.project_id,
        name=k.name,
        api_key=k.api_key,
        hmac_key=k.hmac_key,
        is_active=k.is_active,
        created_at=k.created_at,
    ).model_dump(mode="json")


def _key_summary(k) -> dict:
    # Expose only the first 8 hex chars so users can recognise keys without
    # retrieving the full secret value.
    prefix = k.api_key[:8] if k.api_key else None
    return DeviceKeySummary(
        id=k.id,
        project_id=k.project_id,
        name=k.name,
        api_key_prefix=prefix,
        is_active=k.is_active,
        created_at=k.created_at,
    ).model_dump(mode="json")


def _log(entry) -> dict:
    return InferenceLogResponse(
        id=entry.id,
        project_id=entry.project_id,
        device_id=entry.device_id,
        impulse_id=entry.impulse_id,
        result=entry.result or {},
        created_at=entry.created_at,
    ).model_dump(mode="json")


# ─── Device CRUD ──────────────────────────────────────────────────────────────

@router.post("/register", status_code=201)
def register_device(
    req: DeviceCreateRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Register a device within a project.

    Idempotent via device_id: if a device with the same hardware ID exists
    (even soft-deleted) it is returned / restored rather than duplicated.
    """
    _require_project(db, req.project_id, current_user)
    dev = svc.register_or_restore(db, req)
    return _detail(dev, _session_for(dev))


@router.get("/project/{project_id}")
def list_devices(
    project_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """List all active devices for a project with live connection state."""
    _require_project(db, project_id, current_user)
    return [
        _list_item(d, _session_for(d))
        for d in svc.list_active_devices(db, project_id)
    ]


@router.get("/{device_pk}")
def get_device(
    device_pk: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Get a single device with live connection state."""
    dev = _require_device_owner(db, device_pk, current_user)
    return _detail(dev, _session_for(dev))


@router.patch("/{device_pk}/rename")
def rename_device(
    device_pk: str,
    req: DeviceRenameRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    _require_device_owner(db, device_pk, current_user)
    try:
        dev = svc.update_device(db, device_pk, DeviceUpdateRequest(name=req.name))
        return _detail(dev, _session_for(dev))
    except KeyError:
        raise HTTPException(status_code=404, detail="Device not found")


@router.patch("/{device_pk}")
def update_device(
    device_pk: str,
    req: DeviceUpdateRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    _require_device_owner(db, device_pk, current_user)
    try:
        dev = svc.update_device(db, device_pk, req)
        return _detail(dev, _session_for(dev))
    except KeyError:
        raise HTTPException(status_code=404, detail="Device not found")


@router.delete("/{device_pk}", status_code=204)
def delete_device(
    device_pk: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Soft-delete a device.  Reconnecting the same hardware ID restores it."""
    _require_device_owner(db, device_pk, current_user)
    try:
        svc.soft_delete_device(db, device_pk)
    except KeyError:
        raise HTTPException(status_code=404, detail="Device not found")
    return Response(status_code=204)


@router.post("/{device_pk}/heartbeat")
def heartbeat(
    device_pk: str,
    body: DeviceHeartbeatRequest,
    db: Session = Depends(get_db),
    key_record: ProjectDeviceKey = Depends(resolve_api_key),
):
    """
    Minimum ONLINE contract for any device type.

    Devices should POST every 10–30s with their project device key in the
    ``x-api-key`` header (same credential used by the ingestion endpoints and
    the device WebSocket). The UI/API treats a device as ONLINE while last_seen
    stays within the 45s freshness window.
    """
    # The device key must belong to the same project as the target device;
    # 404 (not 403) so keys can't enumerate device PKs in other projects.
    try:
        target = svc.get_active_device(db, device_pk)
    except KeyError:
        raise HTTPException(status_code=404, detail="Device not found")
    if target.project_id != key_record.project_id:
        raise HTTPException(status_code=404, detail="Device not found")

    try:
        dev = svc.record_heartbeat(db, device_pk, body.model_dump(exclude_none=True))
        return {
            "status": "ok",
            "last_seen": dev.last_seen.isoformat(),
            "fresh_for_seconds": ONLINE_FRESHNESS_WINDOW_SECS,
        }
    except KeyError:
        raise HTTPException(status_code=404, detail="Device not found")


# ─── Phase 4: start-sampling REST command ─────────────────────────────────────

@router.post("/{device_pk}/start-sampling", status_code=202)
async def start_sampling(
    device_pk: str,
    req: StartSamplingRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Trigger a data-collection sample on a connected device.

    Dispatches a `start-sample` WebSocket command (Phase 2 command layer)
    and waits for the device ack before returning.  The device is expected
    to upload the collected sample via the Phase 3 ingestion endpoints.

    A per-sample HMAC key is generated and embedded in the command so the
    device can authenticate its upload with a short-lived key.  The same
    token + key are returned to the caller for reference.

    interval_ms overrides frequency when set (frequency = 1000 / interval_ms).

    Returns 409 if the device is offline or already sampling.
    Returns 408 if the device does not ack within `req.timeout` seconds.
    """
    dev = _require_device_owner(db, device_pk, current_user)

    if not dev.device_id:
        raise HTTPException(
            status_code=409,
            detail="Device has no hardware ID — cannot dispatch command",
        )

    if not manager.is_connected(dev.project_id, dev.device_id):
        raise HTTPException(status_code=409, detail="Device is not connected")

    # interval_ms overrides frequency
    frequency = (1000.0 / req.interval_ms) if req.interval_ms else req.frequency

    # Issue per-sample HMAC key (Phase 4.7)
    hmac_rec = issue_sample_hmac(dev.project_id, dev.device_id)

    cmd = build_start_sample_command(
        label=req.label,
        length_ms=req.length_ms,
        frequency=frequency,
        sensor=req.sensor,
        category=req.category,
        hmac_key=hmac_rec.key_hex,
        sample_token=hmac_rec.token,
    )
    cid: str = cmd.get("correlationId", "")

    try:
        ack = await send_and_await_ack(
            dev.project_id,
            dev.device_id,
            cmd,
            timeout=req.timeout,
        )
        # Transition device mode to sampling after successful ack (Phase 4.9)
        manager.update_mode(dev.project_id, dev.device_id, DeviceMode.sampling)

        return StartSamplingResponse(
            status="started",
            device_id=dev.id,
            correlation_id=cid,
            sample_token=hmac_rec.token,
            hmac_key=hmac_rec.key_hex,
            ack=ack,
        ).model_dump(mode="json")

    except asyncio.TimeoutError:
        raise HTTPException(
            status_code=408,
            detail=f"Device did not acknowledge within {req.timeout:.0f}s",
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc))


# ─── Project device key endpoints ─────────────────────────────────────────────

@router.post("/project/{project_id}/keys", status_code=201)
def create_device_key(
    project_id: str,
    req: DeviceKeyCreateRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Generate a project-scoped device API key. Secrets returned once only."""
    _require_project(db, project_id, current_user)
    key = svc.create_device_key(db, project_id, req.name)
    return _key_response(key)


@router.get("/project/{project_id}/keys")
def list_device_keys(
    project_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """List active keys — full secrets omitted; api_key_prefix shown for identification."""
    _require_project(db, project_id, current_user)
    return [_key_summary(k) for k in svc.list_device_keys(db, project_id)]


@router.delete("/project/{project_id}/keys/{key_id}", status_code=204)
def revoke_device_key(
    project_id: str,
    key_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Revoke (soft-disable) a device key."""
    _require_project(db, project_id, current_user)
    try:
        svc.revoke_device_key(db, key_id, project_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Key not found")
    return Response(status_code=204)


# ─── Phase 4: socket token issuance ──────────────────────────────────────────

@router.post("/project/{project_id}/socket-token", status_code=201)
def issue_socket_token(
    project_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Issue a short-lived single-use token for the browser Studio WebSocket.

    The token is valid for 60 seconds and is consumed on first use.
    Phase 5 Studio WS handler authenticates connections by calling
    socket_token_store.consume(token) — no DB round-trip required.

    Flow:
      1. Browser calls this endpoint → receives token + expires_at.
      2. Browser opens Studio WS connection with token in query/header.
      3. WS handler calls consume(token) to authenticate the connection.
    """
    _require_project(db, project_id, current_user)
    record = socket_token_store.issue(
        project_id=project_id,
        issued_by=current_user.id,
        ttl=_SOCKET_TOKEN_TTL,
    )
    return SocketTokenResponse(
        token=record.value,
        project_id=record.project_id,
        expires_at=record.expires_at,
        ttl_seconds=_SOCKET_TOKEN_TTL,
    ).model_dump(mode="json")


# ─── Inference log endpoints ──────────────────────────────────────────────────

@router.get("/project/{project_id}/inference-logs")
def list_inference_logs(
    project_id: str,
    device_pk: Optional[str] = Query(default=None),
    impulse_id: Optional[str] = Query(default=None),
    limit: int = Query(default=100, le=500),
    skip: int = Query(default=0, ge=0),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """List inference logs with optional device/impulse filter and cursor-style pagination."""
    _require_project(db, project_id, current_user)
    return [
        _log(e)
        for e in svc.list_inference_logs(
            db, project_id, device_pk, limit, skip=skip, impulse_id=impulse_id
        )
    ]


# ─── Phase 6: debug-stream endpoints ─────────────────────────────────────────

@router.post("/{device_pk}/streams/snapshot/start", status_code=202)
async def stream_snapshot_start(
    device_pk: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    dev = _require_device_owner(db, device_pk, current_user)
    if not dev.supports_snapshot_streaming:
        raise HTTPException(status_code=409, detail="Device does not support snapshot streaming")
    if not manager.is_connected(dev.project_id, dev.device_id):
        raise HTTPException(status_code=409, detail="Device is offline")
    state = stream_store.start(dev.project_id, dev.device_id, StreamType.snapshot)
    await send_start_snapshot(dev.project_id, dev.device_id)
    await emit(dev.project_id, EVENT_SNAPSHOT_STARTED, {"stream_id": state.stream_id}, device_id=dev.device_id)
    return {"stream_id": state.stream_id, "stream_type": "snapshot"}


class _InferenceStartBody(BaseModel):
    fomo_threshold:  Optional[float] = None
    sensor:          Optional[str]   = None
    frequency:       Optional[float] = None
    sample_length_ms: Optional[int]  = None


@router.post("/{device_pk}/streams/inference/start", status_code=202)
async def stream_inference_start(
    device_pk: str,
    body: _InferenceStartBody = _InferenceStartBody(),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    dev = _require_device_owner(db, device_pk, current_user)
    if not manager.is_connected(dev.project_id, dev.device_id):
        raise HTTPException(status_code=409, detail="Device is offline")
    state = stream_store.start(
        dev.project_id, dev.device_id, StreamType.inference,
        sensor=body.sensor,
        frequency=body.frequency,
        sample_length_ms=body.sample_length_ms,
    )
    await send_start_inference_stream(
        dev.project_id, dev.device_id,
        fomo_threshold=body.fomo_threshold,
        sensor=body.sensor,
        frequency=body.frequency,
        sample_length_ms=body.sample_length_ms,
    )
    await emit(dev.project_id, EVENT_INFERENCE_STARTED, {"stream_id": state.stream_id}, device_id=dev.device_id)
    return {"stream_id": state.stream_id, "stream_type": "inference"}


@router.post("/{device_pk}/streams/keepalive", status_code=202)
async def stream_keepalive(
    device_pk: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    dev = _require_device_owner(db, device_pk, current_user)
    if not stream_store.refresh(dev.device_id):
        raise HTTPException(status_code=409, detail="No active stream for this device")
    return {"status": "ok"}


@router.post("/{device_pk}/streams/stop", status_code=202)
async def stream_stop(
    device_pk: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    dev = _require_device_owner(db, device_pk, current_user)
    state = stream_store.stop(dev.device_id)
    if state is not None:
        if state.stream_type == StreamType.snapshot:
            await send_stop_snapshot(dev.project_id, dev.device_id)
            await emit(dev.project_id, EVENT_SNAPSHOT_STOPPED, {"stream_id": state.stream_id}, device_id=dev.device_id)
        else:
            await send_stop_inference_stream(dev.project_id, dev.device_id)
            await emit(dev.project_id, EVENT_INFERENCE_STOPPED, {"stream_id": state.stream_id}, device_id=dev.device_id)
    return {"status": "stopped"}


# ─── OTA / deployment targeting (Phase 4.5) ───────────────────────────────────

class _UpdateRequestBody(BaseModel):
    force: bool = False


@router.get("/{device_pk}/compatible-deployment")
def compatible_deployment(
    device_pk: str,
    impulse_id: Optional[str] = Query(default=None),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Return the latest deployment compatible with this device, if any."""
    dev = _require_device_owner(db, device_pk, current_user)
    preferred_runtime = None
    preferred_model_id: str | None = None
    preferred_deployment_id: str | None = None

    if impulse_id:
        preferred_runtime = _get_preferred_runtime_for_device(impulse_id, db, current_user)
        if preferred_runtime["artifact_type"] == "pxe":
            preferred_deployment_id = preferred_runtime["id"]
        else:
            preferred_model_id = preferred_runtime["id"]

    dep = resolve_latest_compatible_deployment(
        db,
        dev,
        force=True,
        preferred_model_id=preferred_model_id,
        preferred_deployment_id=preferred_deployment_id,
    )
    if dep is None:
        # Empty state, not an error: the device is valid but nothing matches yet.
        return None

    # TTL 1 h — presigned URL is for immediate OTA use only
    download_url = (
        storage.get_presigned_url(dep.storage_key, expires_in=3600)
        if dep.storage_key
        else None
    )
    return {
        "deployment_id": dep.id,
        "model_id": dep.model_id,
        "runtime_id": preferred_runtime["id"] if preferred_runtime else dep.id,
        "artifact_type": preferred_runtime["artifact_type"] if preferred_runtime else None,
        "matches_preferred_runtime": preferred_runtime is not None,
        "deployment_target": dep.deployment_target,
        "device_profile": dep.device_profile,
        "download_url": download_url,
    }


@router.post("/{device_pk}/request-update", status_code=201)
async def request_update(
    device_pk: str,
    req: _UpdateRequestBody = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Push an OTA update command to a connected device."""
    if req is None:
        req = _UpdateRequestBody()

    dev = _require_device_owner(db, device_pk, current_user)

    if not manager.is_connected(dev.project_id, dev.device_id):
        raise HTTPException(status_code=409, detail="Device is not connected")

    # Determine if any compatible deployment exists at all (ignore installed state).
    any_dep = resolve_latest_compatible_deployment(db, dev, force=True)
    if any_dep is None:
        raise HTTPException(status_code=404, detail="No compatible deployment found")

    dep = resolve_latest_compatible_deployment(db, dev, force=req.force)
    if dep is None:
        # force=False and device already has the latest deployment
        return {"status": "up_to_date"}

    # TTL 1 h — presigned URL valid for the duration of the OTA download
    download_url = (
        storage.get_presigned_url(dep.storage_key, expires_in=3600)
        if dep.storage_key
        else ""
    )

    await send_deployment_update(
        dev.project_id,
        dev.device_id,
        download_url,
        dep.id,  # version fallback
        deployment_id=dep.id,
        deployment_target=dep.deployment_target or "",
        device_profile=dep.device_profile or "",
    )

    db.add(DeviceUpdateHistory(
        project_id=dev.project_id,
        device_id=dev.id,
        deployment_id=dep.id,
        status="requested",
    ))
    db.commit()

    return {"status": "requested", "deployment_id": dep.id}


@router.get("/{device_pk}/update-status")
def update_status(
    device_pk: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Return the most recent OTA update record for this device."""
    dev = _require_device_owner(db, device_pk, current_user)
    history = (
        db.query(DeviceUpdateHistory)
        .filter(DeviceUpdateHistory.device_id == dev.id)
        .order_by(DeviceUpdateHistory.created_at.desc())
        .first()
    )
    if history is None:
        # Empty state, not an error: no OTA request has been made yet.
        return None
    return {
        "status": history.status,
        "deployment_id": history.deployment_id,
        "message": history.message,
        "updated_at": history.updated_at.isoformat() if history.updated_at else None,
    }
