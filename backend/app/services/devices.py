"""
Device service — Phase 1 + Phase 2 business logic.

All DB mutations for the devices domain live here so endpoint functions stay
thin and this logic can be tested without FastAPI's request/response cycle.

Public surface:
    register_or_restore(db, req)        → Device
    upsert_device_from_hello(db, …)     → Device  (Phase 2 — WS hello path)
    get_active_device(db, device_pk)    → Device  (raises KeyError if missing)
    list_active_devices(db, project_id) → list[Device]
    update_device(db, device_pk, data)  → Device
    soft_delete_device(db, device_pk)   → None
    record_heartbeat(db, device_pk, payload) → Device
    normalize_sensors(raw)              → list  (placeholder hook)
"""
from __future__ import annotations

import secrets
from datetime import datetime
from typing import Any, Dict, List, Optional

from sqlalchemy.orm import Session, attributes as sa_attrs

from app.models.user import Device
from app.models.devices import ProjectDeviceKey, DeviceInferenceLog
from app.schemas.devices import DeviceCreateRequest, DeviceUpdateRequest


# ─── Device CRUD ──────────────────────────────────────────────────────────────

def get_active_device(db: Session, device_pk: str) -> Device:
    """
    Return a non-deleted Device by internal PK.
    Raises KeyError so callers (endpoints) can convert to 404 consistently.
    """
    dev = db.query(Device).filter(
        Device.id == device_pk,
        Device.deleted_at.is_(None),
    ).first()
    if not dev:
        raise KeyError(f"Device {device_pk!r} not found")
    return dev


def list_active_devices(db: Session, project_id: str) -> List[Device]:
    """Return all non-deleted devices for a project, ordered by name."""
    return (
        db.query(Device)
        .filter(Device.project_id == project_id, Device.deleted_at.is_(None))
        .order_by(Device.name)
        .all()
    )


def register_or_restore(db: Session, req: DeviceCreateRequest) -> Device:
    """
    Register a device, with three outcomes:

    1. No device_id supplied → always create a new record.
    2. device_id supplied, active record exists → return it unchanged (idempotent).
    3. device_id supplied, soft-deleted record exists → restore it and update fields.

    This lets hardware that was removed and re-added reclaim its history.
    """
    if req.device_id:
        existing = db.query(Device).filter(
            Device.project_id == req.project_id,
            Device.device_id == req.device_id,
        ).first()

        if existing:
            if existing.deleted_at is not None:
                # Restore soft-deleted device and refresh its fields.
                existing.deleted_at = None
                existing.name = req.name
                existing.device_type = req.device_type
                existing.connection = req.connection
                existing.firmware_version = req.firmware_version
                existing.protocol_version = req.protocol_version
                existing.supports_snapshot_streaming = req.supports_snapshot_streaming
                existing.remote_mgmt_host = req.remote_mgmt_host
                existing.sensors = normalize_sensors(req.sensors)
                existing.device_metadata = req.device_metadata
                db.commit()
                db.refresh(existing)
            return existing

    dev = Device(
        project_id=req.project_id,
        name=req.name,
        device_id=req.device_id,
        device_type=req.device_type,
        connection=req.connection,
        firmware_version=req.firmware_version,
        protocol_version=req.protocol_version,
        supports_snapshot_streaming=req.supports_snapshot_streaming,
        remote_mgmt_host=req.remote_mgmt_host,
        sensors=normalize_sensors(req.sensors),
        device_metadata=req.device_metadata,
    )
    db.add(dev)
    db.commit()
    db.refresh(dev)
    return dev


def upsert_device_from_hello(
    db: Session,
    project_id: str,
    device_id: str,
    *,
    device_type: Optional[str] = None,
    connection: Optional[str] = None,
    firmware_version: Optional[str] = None,
    protocol_version: Optional[str] = None,
    supports_snapshot_streaming: bool = False,
    sensors: Optional[List[Any]] = None,
) -> Device:
    """
    Find-or-create/restore a Device from a successful WebSocket hello.

    Outcomes:
      1. Active record exists           → update durable fields, return it.
      2. Soft-deleted record exists     → restore, update fields, return it.
      3. No record                      → create with deviceId as name placeholder.

    Does NOT overwrite a user-set `name`.  Only fields the device self-reports
    are updated; UI-set fields (name, remote_mgmt_host) are left unchanged.

    Callers pass individual fields (not a schema object) so the service layer
    stays free of imports from the realtime package.
    """
    existing = db.query(Device).filter(
        Device.project_id == project_id,
        Device.device_id == device_id,
    ).first()

    if existing is not None:
        if existing.deleted_at is not None:
            existing.deleted_at = None  # restore

        # Update only fields the device self-reports
        if device_type:
            existing.device_type = device_type
        if connection:
            existing.connection = connection
        if firmware_version:
            existing.firmware_version = firmware_version
        if protocol_version:
            existing.protocol_version = protocol_version
        existing.supports_snapshot_streaming = supports_snapshot_streaming
        if sensors:
            existing.sensors = normalize_sensors(sensors)
            sa_attrs.flag_modified(existing, "sensors")
        existing.last_seen = datetime.utcnow()
        db.commit()
        db.refresh(existing)
        return existing

    # First-ever connection for this hardware ID — create a placeholder record.
    dev = Device(
        project_id=project_id,
        device_id=device_id,
        name=device_id,              # placeholder; user renames via REST
        device_type=device_type,
        connection=connection,
        firmware_version=firmware_version,
        protocol_version=protocol_version,
        supports_snapshot_streaming=supports_snapshot_streaming,
        sensors=normalize_sensors(sensors or []),
    )
    dev.last_seen = datetime.utcnow()
    db.add(dev)
    db.commit()
    db.refresh(dev)
    return dev


def update_device(db: Session, device_pk: str, req: DeviceUpdateRequest) -> Device:
    dev = get_active_device(db, device_pk)
    update_data = req.model_dump(exclude_unset=True)
    if "sensors" in update_data:
        update_data["sensors"] = normalize_sensors(update_data["sensors"])
    if "device_metadata" in update_data:
        existing_meta = dev.device_metadata or {}
        update_data["device_metadata"] = {**existing_meta, **update_data["device_metadata"]}
    for field, value in update_data.items():
        setattr(dev, field, value)
    db.commit()
    db.refresh(dev)
    return dev


def soft_delete_device(db: Session, device_pk: str) -> None:
    dev = get_active_device(db, device_pk)
    dev.deleted_at = datetime.utcnow()
    db.commit()


def record_heartbeat(
    db: Session,
    device_pk: str,
    payload: Dict[str, Any],
) -> Device:
    """
    Update durable fields reported by the device on each heartbeat.

    Only persists fields that are stable hardware/firmware facts:
      - last_seen (always)
      - firmware_version, protocol_version, remote_mgmt_host, ip_address
        (if supplied)

    Ephemeral network state (ip_address, online/offline) is intentionally
    excluded here; Phase 2 will derive online state from last_seen TTL.

    Does not require user authentication — callers must secure at the
    network layer (e.g. project device key in a later phase).
    Raises KeyError if device is missing or soft-deleted.
    """
    dev = get_active_device(db, device_pk)
    dev.last_seen = datetime.utcnow()
    if payload.get("firmware_version"):
        dev.firmware_version = payload["firmware_version"]
    if payload.get("protocol_version"):
        dev.protocol_version = payload["protocol_version"]
    if payload.get("ip_address"):
        dev.ip_address = payload["ip_address"]
    if payload.get("remote_mgmt_host"):
        dev.remote_mgmt_host = payload["remote_mgmt_host"]

    # Merge a runtime health snapshot into device_metadata.diagnostics without
    # dropping other keys (e.g. current_model_id).  device_metadata is a JSON
    # column, so flag_modified is required for SQLAlchemy to persist the change.
    diagnostics = payload.get("diagnostics")
    if isinstance(diagnostics, dict) and diagnostics:
        meta = dict(dev.device_metadata or {})
        meta["diagnostics"] = diagnostics
        dev.device_metadata = meta
        sa_attrs.flag_modified(dev, "device_metadata")

    db.commit()
    db.refresh(dev)
    return dev


# ─── Sensor normalisation (placeholder hook) ──────────────────────────────────

def normalize_sensors(raw: Optional[List[Any]]) -> List[Any]:
    """
    Validate and normalise the sensors list supplied at registration time.

    Phase 1: pass-through with basic type safety.
    Phase 2+: enforce schema `{name, type, freq_hz, axes}`, coerce types,
              reject unknown sensor types, etc.
    """
    if not raw:
        return []
    return [s for s in raw if isinstance(s, dict)]


# ─── ProjectDeviceKey helpers ─────────────────────────────────────────────────

def create_device_key(
    db: Session,
    project_id: str,
    name: Optional[str] = None,
) -> ProjectDeviceKey:
    """
    Generate a new project-scoped device key.

    Both api_key and hmac_key are generated server-side with secrets.token_hex
    so they have sufficient entropy.  They are returned once in plain text;
    the caller is responsible for delivering them securely to the device.
    """
    key = ProjectDeviceKey(
        project_id=project_id,
        name=name,
        api_key=secrets.token_hex(32),   # 64-char hex → 256-bit
        hmac_key=secrets.token_hex(32),
    )
    db.add(key)
    db.commit()
    db.refresh(key)
    return key


def list_device_keys(db: Session, project_id: str) -> List[ProjectDeviceKey]:
    return (
        db.query(ProjectDeviceKey)
        .filter(
            ProjectDeviceKey.project_id == project_id,
            ProjectDeviceKey.is_active.is_(True),
        )
        .order_by(ProjectDeviceKey.created_at.desc())
        .all()
    )


def revoke_device_key(db: Session, key_id: str, project_id: str) -> None:
    key = db.query(ProjectDeviceKey).filter(
        ProjectDeviceKey.id == key_id,
        ProjectDeviceKey.project_id == project_id,
    ).first()
    if not key:
        raise KeyError(f"Device key {key_id!r} not found")
    key.is_active = False
    db.commit()


# ─── DeviceInferenceLog helpers ───────────────────────────────────────────────

def log_inference(
    db: Session,
    project_id: str,
    device_pk: Optional[str],
    impulse_id: Optional[str],
    result: Dict[str, Any],
) -> DeviceInferenceLog:
    entry = DeviceInferenceLog(
        project_id=project_id,
        device_id=device_pk,
        impulse_id=impulse_id,
        result=result,
    )
    db.add(entry)
    db.commit()
    db.refresh(entry)
    return entry


def list_inference_logs(
    db: Session,
    project_id: str,
    device_pk: Optional[str] = None,
    limit: int = 100,
    *,
    skip: int = 0,
    impulse_id: Optional[str] = None,
) -> List[DeviceInferenceLog]:
    q = db.query(DeviceInferenceLog).filter(
        DeviceInferenceLog.project_id == project_id
    )
    if device_pk:
        q = q.filter(DeviceInferenceLog.device_id == device_pk)
    if impulse_id:
        q = q.filter(DeviceInferenceLog.impulse_id == impulse_id)
    return (
        q.order_by(DeviceInferenceLog.created_at.desc())
        .offset(skip)
        .limit(limit)
        .all()
    )
