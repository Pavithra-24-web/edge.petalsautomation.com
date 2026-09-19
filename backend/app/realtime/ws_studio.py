"""
Browser Studio WebSocket — Phase 5

URL: ws://<host>/ws/studio?token=<value>

Protocol sequence
─────────────────
1.  Browser opens the WebSocket with a short-lived token in the query string.
2.  Server calls socket_token_store.consume(token):
      • None → close 1008 Policy Violation (invalid / expired / already used).
      • SocketTokenRecord → extract project_id.
3.  Register connection with studio_manager.
4.  Send initial device snapshot for the project.
5.  Keepalive loop:
      • Receive text frames; route by "type".
      • {"type": "ping"} → reply {"type": "pong"}.
      • Any other type is silently ignored (browsers may send telemetry).
      • WebSocketDisconnect → exit loop.
6.  finally: studio_manager.remove (idempotent).

Token notes
───────────
Tokens are single-use (consumed on first use).  The browser must request a
fresh token via POST /api/v1/devices/project/{project_id}/socket-token before
each WebSocket connection attempt.
"""
from __future__ import annotations

import json
import logging

from fastapi import APIRouter, Query, WebSocket, WebSocketDisconnect

from app.core.socket_token_store import socket_token_store
from app.realtime.connection_manager import manager
from app.realtime.studio_manager import studio_manager

logger = logging.getLogger(__name__)

router = APIRouter()


def _initial_snapshot(project_id: str) -> dict:
    """
    Build a device list snapshot to send immediately after auth.

    Pulls live session state from the device ConnectionManager so the
    browser has an accurate picture without waiting for the next event.
    """
    sessions = manager.get_project_sessions(project_id)
    return {
        "type":       "snapshot",
        "project_id": project_id,
        "devices": [
            {
                "device_id": s.device_id,
                "mode":      s.mode,
                "connected": True,
            }
            for s in sessions
        ],
    }


@router.websocket("/studio")
async def studio_ws(
    websocket: WebSocket,
    token: str = Query(..., description="Single-use socket token"),
) -> None:
    """
    Browser Studio WebSocket.  Authenticated via a single-use token
    issued by POST /api/v1/devices/project/{project_id}/socket-token.
    """
    await websocket.accept()

    record = socket_token_store.consume(token)
    if record is None:
        logger.warning("Studio WS auth failure: invalid/expired/reused token")
        await websocket.close(code=1008)
        return

    project_id = record.project_id
    connection_id: str | None = None

    try:
        connection_id = studio_manager.register(project_id, websocket)

        logger.info(
            "Studio connected: project=%s connection_id=%s issued_by=%s",
            project_id, connection_id, record.issued_by,
        )

        # Send initial device snapshot so the browser is immediately aware
        # of all currently-connected devices in this project.
        await websocket.send_json(_initial_snapshot(project_id))

        # ── Keepalive loop ────────────────────────────────────────────────
        while True:
            try:
                raw = await websocket.receive_text()
            except WebSocketDisconnect:
                break

            try:
                msg = json.loads(raw)
            except (json.JSONDecodeError, ValueError):
                continue  # ignore malformed frames silently

            if msg.get("type") == "ping":
                await websocket.send_json({"type": "pong"})

            # All other message types are silently ignored — browsers may
            # send telemetry or future client-initiated messages.

    except WebSocketDisconnect:
        pass

    except Exception:
        logger.exception(
            "Unexpected error in Studio WS project=%s connection_id=%s",
            project_id, connection_id or "<unregistered>",
        )

    finally:
        if connection_id is not None:
            studio_manager.remove(project_id, connection_id)
        logger.info(
            "Studio disconnected: project=%s connection_id=%s",
            project_id, connection_id or "<unregistered>",
        )
