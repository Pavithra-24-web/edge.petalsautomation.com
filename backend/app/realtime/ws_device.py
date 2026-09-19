"""
Device Remote Management WebSocket — Phase 2

URL: ws://<host>/ws/device
     (project_id is resolved from the apiKey during hello — no path parameter)

Protocol sequence
─────────────────
1.  Device opens the WebSocket.
2.  Device sends a 'hello' JSON frame within HELLO_TIMEOUT_SECS.
3.  Server resolves project_id from apiKey via project_device_keys table.
4.  Server upserts/restores the device record (register_or_restore semantics).
5.  Server registers the session in ConnectionManager and sends hello-ack.
     • On any failure: sends error ack then closes (1008 Policy Violation).
6.  Authenticated message loop — routes by type:
     pong            → update last_seen in DB
     set-mode        → validate + update session mode, send set-mode-ack
     sample-ack      → resolve pending ack future (used by send_and_await_ack)
     sample-started  → switch mode to 'sampling', update last_seen
     sample-stopped  → switch mode to 'idle',     update last_seen
     sample-failed   → switch mode to 'idle',     log error
     snapshot        → placeholder log (Phase 3 binary upgrade)
     status          → log device status, update last_seen
     ack (generic)   → resolve pending ack or log
     ping            → reply with pong-ack (rare device-initiated ping)
     (unknown)       → debug log only
7.  Cleanup: session removed from manager on any disconnect or error.

Phase 3 additions (not here):
  Binary snapshot frames, debug-stream, inference-stream, model-update acks.
"""
from __future__ import annotations

import asyncio
import base64
import json
import logging
import struct
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.core.database import SessionLocal
from app.models.devices import ProjectDeviceKey
from app.realtime.connection_manager import (
    DeviceMode,
    DeviceSession,
    WsEncoding,
    manager,
)
from app.realtime.schemas import (
    ErrorMessage,
    HelloAck,
    HelloPayload,
    SetModeAck,
)
from app.realtime.studio_events import (
    emit,
    EVENT_DEVICE_CONNECTED,
    EVENT_DEVICE_DISCONNECTED,
    EVENT_DEVICE_MODE_CHANGED,
    EVENT_SAMPLE_STARTED,
    EVENT_SAMPLE_STOPPED,
    EVENT_SAMPLE_FAILED,
    EVENT_UPDATE_ACCEPTED,
    EVENT_UPDATE_REJECTED,
    EVENT_UPDATE_DOWNLOADING,
    EVENT_UPDATE_FLASHING,
    EVENT_UPDATE_DONE,
    EVENT_UPDATE_FAILED,
    EVENT_SNAPSHOT_FRAME,
    EVENT_INFERENCE_RESULT,
)
from app.services import devices as svc
from app.services.stream_store import stream_store, StreamType as _StreamType
from app.motion.websocket import sensor_frames as _motion_sensor_frames

logger = logging.getLogger(__name__)

router = APIRouter()

HELLO_TIMEOUT_SECS: float = 10.0


# ─── Auth helpers ─────────────────────────────────────────────────────────────

def _resolve_project_from_key(db, api_key: str) -> Optional[str]:
    """
    Look up the project that owns this api_key.

    Returns project_id on success, None if the key is missing or revoked.
    Resolving project from key (not from the URL) means devices only need
    to know one server URL and one secret — no project UUID in firmware.
    """
    row = db.query(ProjectDeviceKey).filter(
        ProjectDeviceKey.api_key == api_key,
        ProjectDeviceKey.is_active.is_(True),
    ).first()
    return row.project_id if row else None


# ─── Message handlers ─────────────────────────────────────────────────────────

async def _handle_pong(session: DeviceSession, msg: dict, db) -> None:
    """Keep-alive reply from device — refresh last_seen."""
    dev = db.query(__import__("app.models.user", fromlist=["Device"]).Device).filter_by(
        id=session.device_pk
    ).first()
    if dev:
        dev.last_seen = datetime.utcnow()
        db.commit()


async def _handle_set_mode(
    session: DeviceSession,
    msg: dict,
    ws: WebSocket,
) -> None:
    """Validate and apply a mode transition requested by the device."""
    mode_str = msg.get("mode", "")
    try:
        new_mode = DeviceMode(mode_str)
    except ValueError:
        await ws.send_json(
            SetModeAck(success=False, error=f"Unknown mode '{mode_str}'").model_dump(
                exclude_none=True
            )
        )
        return
    manager.update_mode(session.project_id, session.device_id, new_mode)
    await emit(session.project_id, EVENT_DEVICE_MODE_CHANGED, {"mode": new_mode.value}, device_id=session.device_id)
    await ws.send_json(
        SetModeAck(success=True, mode=new_mode.value).model_dump(exclude_none=True)
    )


async def _handle_sample_ack(
    session: DeviceSession,
    msg: dict,
    db,
) -> None:
    """
    Device acknowledged a start-sample command.

    Resolves the matching pending Future from send_and_await_ack so the
    REST caller waiting for the ack is unblocked.
    """
    cid = msg.get("correlationId")
    success = msg.get("success", True)
    if cid:
        if success:
            manager.resolve_ack(session.project_id, session.device_id, cid, msg)
        else:
            manager.reject_ack(
                session.project_id, session.device_id, cid,
                msg.get("error", "Device rejected sample request"),
            )
    else:
        logger.debug(
            "sample-ack without correlationId from device_id=%s", session.device_id
        )


async def _handle_sample_lifecycle(
    session: DeviceSession,
    msg: dict,
    db,
) -> None:
    """
    Handle sample-started / sample-stopped / sample-failed lifecycle events.

    Mode transitions:
      sample-started → DeviceMode.sampling
      sample-stopped → DeviceMode.idle
      sample-failed  → DeviceMode.idle  (logs error)
    """
    msg_type = msg.get("type", "")
    cid = msg.get("correlationId")

    if msg_type == "sample-started":
        manager.update_mode(session.project_id, session.device_id, DeviceMode.sampling)
        _touch_last_seen(session, db)
        await emit(session.project_id, EVENT_SAMPLE_STARTED, {}, device_id=session.device_id)
        if cid:
            manager.resolve_ack(session.project_id, session.device_id, cid, msg)

    elif msg_type == "sample-stopped":
        manager.update_mode(session.project_id, session.device_id, DeviceMode.idle)
        _touch_last_seen(session, db)
        await emit(session.project_id, EVENT_SAMPLE_STOPPED, {}, device_id=session.device_id)
        if cid:
            manager.resolve_ack(session.project_id, session.device_id, cid, msg)

    elif msg_type == "sample-failed":
        manager.update_mode(session.project_id, session.device_id, DeviceMode.idle)
        logger.warning(
            "sample-failed device_id=%s error=%r",
            session.device_id, msg.get("error"),
        )
        await emit(session.project_id, EVENT_SAMPLE_FAILED, {"error": msg.get("error")}, device_id=session.device_id)
        if cid:
            manager.reject_ack(
                session.project_id, session.device_id, cid,
                msg.get("error", "Sample failed"),
            )


async def _handle_inference_result(session: DeviceSession, msg: dict) -> None:
    payload = dict(msg.get("payload") or msg)
    state = stream_store.get(session.device_id)
    if state is not None and state.stream_type == _StreamType.inference:
        if state.sensor is not None:
            payload.setdefault("stream_sensor", state.sensor)
        if state.frequency is not None:
            payload.setdefault("stream_frequency", state.frequency)
        if state.sample_length_ms is not None:
            payload.setdefault("stream_sample_length_ms", state.sample_length_ms)
    await emit(
        session.project_id,
        EVENT_INFERENCE_RESULT,
        payload,
        device_id=session.device_id,
    )


async def _emit_snapshot_frame(
    session: DeviceSession,
    header: dict,
    data: bytes,
) -> None:
    """
    Pair a stashed snapshot header with its binary frame and emit it to Studio.

    The device sends the JPEG as a separate binary WS message prefixed with a
    4-byte little-endian length (device_client _snapshot_stream_task). Strip the
    prefix, base64-encode the JPEG, and emit under the "image" key so the Studio
    UI can render `data:image/jpeg;base64,<image>`.
    """
    if len(data) < 4:
        logger.debug(
            "Snapshot binary frame too short (%d bytes) device_id=%s — dropping",
            len(data), session.device_id,
        )
        return

    (length,) = struct.unpack("<I", data[:4])
    jpeg = data[4:4 + length] if length else data[4:]
    image_b64 = base64.b64encode(jpeg).decode("ascii")

    await emit(
        session.project_id,
        EVENT_SNAPSHOT_FRAME,
        {
            "width":  header.get("width"),
            "height": header.get("height"),
            "format": header.get("format"),
            "image":  image_b64,
        },
        device_id=session.device_id,
    )


async def _handle_status(
    session: DeviceSession,
    msg: dict,
    db,
) -> None:
    """Generic device status message — log data and refresh last_seen."""
    logger.info(
        "device status device_id=%s project=%s data=%s",
        session.device_id, session.project_id, msg.get("data", {}),
    )
    _touch_last_seen(session, db)


_UPDATE_STATUS_MAP: dict[str, str] = {
    "update-accepted":          "downloading",
    "update-rejected":          "failed",
    "update-download-started":  "downloading",
    "update-install-started":   "flashing",
    "update-install-succeeded": "done",
    "update-install-failed":    "failed",
}

_UPDATE_EVENT_MAP: dict[str, str] = {
    "update-accepted":          EVENT_UPDATE_ACCEPTED,
    "update-rejected":          EVENT_UPDATE_REJECTED,
    "update-download-started":  EVENT_UPDATE_DOWNLOADING,
    "update-install-started":   EVENT_UPDATE_FLASHING,
    "update-install-succeeded": EVENT_UPDATE_DONE,
    "update-install-failed":    EVENT_UPDATE_FAILED,
}


async def _handle_update_lifecycle(
    session: DeviceSession,
    msg: dict,
    db,
) -> None:
    """
    Persist OTA update state transitions reported by the device.

    msg_type                  → status written to DeviceUpdateHistory
    update-accepted           → downloading
    update-rejected           → failed
    update-download-started   → downloading
    update-install-started    → flashing
    update-install-succeeded  → done  (also sets device.installed_deployment_id/version)
    update-install-failed     → failed
    """
    from app.models.user import Device, DeviceUpdateHistory

    msg_type = msg.get("type", "")
    new_status = _UPDATE_STATUS_MAP.get(msg_type)
    if not new_status:
        return

    history = (
        db.query(DeviceUpdateHistory)
        .filter(
            DeviceUpdateHistory.project_id == session.project_id,
            DeviceUpdateHistory.device_id == session.device_pk,
            DeviceUpdateHistory.status.notin_(["done", "failed"]),
        )
        .order_by(DeviceUpdateHistory.created_at.desc())
        .first()
    )
    if history is None:
        logger.warning(
            "update lifecycle msg with no active history device_id=%s type=%r",
            session.device_id, msg_type,
        )
        return

    history.status = new_status
    history.message = msg.get("message")

    if msg_type == "update-install-succeeded":
        dev = db.query(Device).filter(Device.id == session.device_pk).first()
        if dev:
            dev.installed_deployment_id = history.deployment_id
            dev.installed_model_version = msg.get("version") or history.deployment_id

    db.commit()

    event_type = _UPDATE_EVENT_MAP.get(msg_type)
    if event_type:
        await emit(session.project_id, event_type, {"message": msg.get("message")}, device_id=session.device_id)


async def _handle_generic_ack(
    session: DeviceSession,
    msg: dict,
) -> None:
    """
    Generic ack — resolve or reject any matching pending Future.

    Also used as the default handler for any message that carries a
    correlationId but has no dedicated handler above.
    """
    cid = msg.get("correlationId")
    if not cid:
        logger.debug(
            "generic ack without correlationId device_id=%s type=%r",
            session.device_id, msg.get("type"),
        )
        return
    success = msg.get("success", True)
    if success:
        manager.resolve_ack(session.project_id, session.device_id, cid, msg)
    else:
        manager.reject_ack(
            session.project_id, session.device_id, cid,
            msg.get("error", "Device returned failure ack"),
        )


def _touch_last_seen(session: DeviceSession, db) -> None:
    """Update last_seen on the Device row without loading it fully."""
    from app.models.user import Device
    db.query(Device).filter(Device.id == session.device_pk).update(
        {"last_seen": datetime.utcnow()}, synchronize_session=False
    )
    db.commit()


# ─── Reject helper ────────────────────────────────────────────────────────────

async def _reject(ws: WebSocket, reason: str, code: int = 1008) -> None:
    try:
        await ws.send_json(
            HelloAck(success=False, error=reason).model_dump(exclude_none=True)
        )
        await ws.close(code=code)
    except Exception:
        pass


# ─── WebSocket endpoint ───────────────────────────────────────────────────────

@router.websocket("/device")
async def device_ws(websocket: WebSocket) -> None:
    """
    Single WebSocket URL for all device connections.
    Project membership is derived from the apiKey in the hello message.
    """
    await websocket.accept()

    remote_host: Optional[str] = (
        websocket.client.host if websocket.client else None
    )
    # Set after successful auth so the finally block can clean up.
    project_id: Optional[str] = None
    registered_device_id: Optional[str] = None
    registered_device_pk: Optional[str] = None   # internal DB PK for is_online clear

    db = SessionLocal()

    try:
        # ── 1. Wait for hello ─────────────────────────────────────────────
        try:
            raw = await asyncio.wait_for(
                websocket.receive_text(), timeout=HELLO_TIMEOUT_SECS
            )
        except asyncio.TimeoutError:
            await _reject(websocket, "Hello not received within timeout")
            return
        except WebSocketDisconnect:
            return

        try:
            payload_dict = json.loads(raw)
        except json.JSONDecodeError:
            await _reject(websocket, "Hello message is not valid JSON")
            return

        if payload_dict.get("type") != "hello":
            await _reject(
                websocket,
                f"Expected type='hello', got {payload_dict.get('type')!r}",
            )
            return

        try:
            hello = HelloPayload.model_validate(payload_dict)
        except Exception as exc:
            await _reject(websocket, f"Invalid hello payload: {exc}")
            return

        # ── 2. Resolve project from apiKey (Step 2.4) ─────────────────────
        project_id = _resolve_project_from_key(db, hello.apiKey)
        if project_id is None:
            logger.warning(
                "WS auth failure: unknown/revoked apiKey remote=%s deviceId=%s",
                remote_host, hello.deviceId,
            )
            await _reject(websocket, "Invalid or revoked API key")
            return

        # ── 3. Upsert/restore device from hello (Steps 2.4, 2.5) ─────────
        device = svc.upsert_device_from_hello(
            db,
            project_id=project_id,
            device_id=hello.deviceId,
            device_type=hello.deviceType,
            connection=hello.connection,
            firmware_version=hello.firmwareVersion,
            protocol_version=hello.protocolVersion,
            supports_snapshot_streaming=hello.supportsSnapshotStreaming,
            sensors=hello.sensors or [],
        )

        # ── 4. Register session ───────────────────────────────────────────
        registered_device_id = hello.deviceId
        registered_device_pk = device.id
        session = DeviceSession(
            websocket=websocket,
            project_id=project_id,
            device_id=hello.deviceId,
            device_pk=device.id,
            mode=DeviceMode.idle,
            encoding=WsEncoding.json,
            remote_host=remote_host,
        )
        manager.register(session)
        await emit(project_id, EVENT_DEVICE_CONNECTED, {"mode": DeviceMode.idle.value}, device_id=hello.deviceId)

        logger.info(
            "Device connected: project=%s device_id=%s pk=%s host=%s",
            project_id, hello.deviceId, device.id, remote_host,
        )

        # ── 5. Send hello-ack ─────────────────────────────────────────────
        await websocket.send_json(
            HelloAck(success=True, id=device.id, project_id=project_id).model_dump(exclude_none=True)
        )

        # ── 6. Authenticated message loop (Step 2.7) ─────────────────────
        # Most messages are JSON text frames. Snapshot IMAGE frames are the
        # exception: the device sends a JSON header (type=="snapshot") followed
        # by a separate binary WS message (4-byte LE length prefix + JPEG). We
        # stash the header and pair it with the next binary frame.
        pending_snapshot: Optional[dict] = None
        while True:
            try:
                raw_msg = await websocket.receive()
            except WebSocketDisconnect:
                break

            if raw_msg["type"] == "websocket.disconnect":
                break

            # ── Binary frame → pair with a stashed snapshot header ──
            if raw_msg.get("bytes") is not None:
                if pending_snapshot is None:
                    logger.debug(
                        "Binary frame with no pending snapshot header device_id=%s "
                        "(%d bytes dropped)",
                        hello.deviceId, len(raw_msg["bytes"]),
                    )
                    continue
                await _emit_snapshot_frame(session, pending_snapshot, raw_msg["bytes"])
                pending_snapshot = None
                continue

            # ── Text frame → existing JSON routing (unchanged) ──
            raw_text = raw_msg.get("text")
            if raw_text is None:
                continue

            try:
                msg = json.loads(raw_text)
            except json.JSONDecodeError:
                await websocket.send_json(
                    ErrorMessage(error="Message is not valid JSON").model_dump()
                )
                continue

            msg_type = msg.get("type", "")

            # Route by message type — keep each branch small (Step 2.7)
            if msg_type == "pong":
                await _handle_pong(session, msg, db)

            elif msg_type == "set-mode":
                await _handle_set_mode(session, msg, websocket)

            elif msg_type == "sample-ack":
                await _handle_sample_ack(session, msg, db)

            elif msg_type in ("sample-started", "sample-stopped", "sample-failed"):
                await _handle_sample_lifecycle(session, msg, db)

            elif msg_type == "snapshot":
                # Header only — stash and wait for the binary frame that follows.
                pending_snapshot = msg

            elif msg_type == "inference-result":
                await _handle_inference_result(session, msg)

            elif msg_type == "sensor-frame":
                await _motion_sensor_frames.handle(session, msg)

            elif msg_type == "status":
                await _handle_status(session, msg, db)

            elif msg_type == "ack":
                await _handle_generic_ack(session, msg)

            elif msg_type == "ping":
                # Device-initiated ping — reply so the device knows we're alive
                await websocket.send_json({"type": "pong-ack"})

            elif msg_type in (
                "update-accepted", "update-rejected",
                "update-download-started",
                "update-install-started",
                "update-install-succeeded",
                "update-install-failed",
            ):
                await _handle_update_lifecycle(session, msg, db)

            else:
                logger.debug(
                    "Unhandled message type=%r device_id=%s",
                    msg_type, hello.deviceId,
                )

    except WebSocketDisconnect:
        pass

    except Exception:
        logger.exception(
            "Unexpected error in device WS project=%s device_id=%s",
            project_id or "<unauthenticated>",
            registered_device_id or "<unknown>",
        )

    finally:
        if project_id and registered_device_id:
            manager.remove(project_id, registered_device_id)  # also cancels pending_acks
            await emit(project_id, EVENT_DEVICE_DISCONNECTED, {}, device_id=registered_device_id)
            logger.info(
                "Device disconnected: project=%s device_id=%s",
                project_id, registered_device_id,
            )

        # Transitional compat: clear the legacy is_online flag on disconnect.
        # This field is superseded by last_seen TTL checks in Phase 2+ but some
        # older REST consumers still read it; we keep it accurate here.
        if registered_device_pk:
            try:
                from app.models.user import Device
                db.query(Device).filter(Device.id == registered_device_pk).update(
                    {"is_online": False}, synchronize_session=False
                )
                db.commit()
            except Exception:
                pass  # best-effort; don't mask the original disconnect

        db.close()
