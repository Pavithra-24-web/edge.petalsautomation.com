"""
Device Connection Manager — Phase 2

Maintains an in-memory registry of active device WebSocket sessions keyed by
(project_id, device_id).  WebSocket objects cannot be serialised, so the
connection objects are always local to the process.

Phase 3 note: for multi-instance deployments add a Redis pub/sub layer on top
of this module.  Publish command payloads to the device's channel; the process
that holds the socket forwards it.  The API surface below accommodates that
without callers changing.
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from fastapi import WebSocket

logger = logging.getLogger(__name__)


# ─── Enums ────────────────────────────────────────────────────────────────────

class DeviceMode(str, Enum):
    """
    Ephemeral runtime mode of a connected device.

    `idle`      — connected, not actively sampling or running inference
    `sampling`  — collecting a data sample for the dataset
    `inference` — executing a deployed model on the device
    """
    idle      = "idle"
    sampling  = "sampling"
    inference = "inference"


class WsEncoding(str, Enum):
    json   = "json"
    binary = "binary"   # Phase 3: snapshot / debug-stream frames


# ─── Session ──────────────────────────────────────────────────────────────────

@dataclass
class DeviceSession:
    """All per-connection state for one authenticated device WebSocket."""
    websocket:    WebSocket
    project_id:   str
    device_id:    str               # stable external hardware identifier (Device.device_id)
    device_pk:    str               # internal DB primary key (Device.id)
    mode:         DeviceMode        = DeviceMode.idle
    encoding:     WsEncoding        = WsEncoding.json
    connected_at: datetime          = field(default_factory=datetime.utcnow)
    remote_host:  Optional[str]     = None

    # Pending ack futures keyed by correlationId.
    # Populated by send_and_await_ack; resolved by the message router.
    pending_acks: Dict[str, asyncio.Future] = field(default_factory=dict)


# ─── Manager ──────────────────────────────────────────────────────────────────

class ConnectionManager:
    """
    Registry of active device WebSocket sessions.

    Keyed by (project_id, device_id).  All methods are synchronous except the
    send* helpers, which are async to match the WebSocket API.

    Thread-safety: asyncio is single-threaded; dict mutations are safe between
    awaits in the same event loop without an explicit lock.
    """

    def __init__(self) -> None:
        self._sessions: Dict[tuple[str, str], DeviceSession] = {}

    # ── Registration ──────────────────────────────────────────────────────────

    def register(self, session: DeviceSession) -> None:
        """Add (or replace) a session.  Replacing drops any pending acks on the old entry."""
        key = (session.project_id, session.device_id)
        old = self._sessions.get(key)
        if old is not None and old is not session:
            logger.warning(
                "Replacing stale session project=%s device_id=%s; cancelling %d pending acks",
                session.project_id, session.device_id, len(old.pending_acks),
            )
            self._cancel_all_acks(old)
        self._sessions[key] = session

    def remove(self, project_id: str, device_id: str) -> Optional[DeviceSession]:
        """Remove and return the session, cancelling any pending acks first."""
        session = self._sessions.pop((project_id, device_id), None)
        if session:
            self._cancel_all_acks(session)
        return session

    # ── Lookup ────────────────────────────────────────────────────────────────

    def get(self, project_id: str, device_id: str) -> Optional[DeviceSession]:
        return self._sessions.get((project_id, device_id))

    def is_connected(self, project_id: str, device_id: str) -> bool:
        return (project_id, device_id) in self._sessions

    def get_project_sessions(self, project_id: str) -> List[DeviceSession]:
        return [s for (pid, _), s in self._sessions.items() if pid == project_id]

    def all_sessions(self) -> List[DeviceSession]:
        return list(self._sessions.values())

    def count(self) -> int:
        return len(self._sessions)

    # ── State mutation ────────────────────────────────────────────────────────

    def update_mode(self, project_id: str, device_id: str, mode: DeviceMode) -> bool:
        """Change the mode of a connected device.  Returns False if not found."""
        session = self.get(project_id, device_id)
        if session is None:
            return False
        session.mode = mode
        return True

    # ── Pending-ack primitives (Step 2.10) ────────────────────────────────────

    def set_pending_ack(
        self,
        project_id: str,
        device_id: str,
        correlation_id: str,
        future: asyncio.Future,
    ) -> bool:
        """Register a future that will be resolved when the device acks this correlationId."""
        session = self.get(project_id, device_id)
        if session is None:
            return False
        session.pending_acks[correlation_id] = future
        return True

    def resolve_ack(
        self,
        project_id: str,
        device_id: str,
        correlation_id: str,
        result: Any,
    ) -> bool:
        """Resolve a pending ack future with a success result."""
        session = self.get(project_id, device_id)
        if session is None:
            return False
        future = session.pending_acks.pop(correlation_id, None)
        if future is None or future.done():
            return False
        future.set_result(result)
        return True

    def reject_ack(
        self,
        project_id: str,
        device_id: str,
        correlation_id: str,
        error: str,
    ) -> bool:
        """Resolve a pending ack future with an exception (device reported failure)."""
        session = self.get(project_id, device_id)
        if session is None:
            return False
        future = session.pending_acks.pop(correlation_id, None)
        if future is None or future.done():
            return False
        future.set_exception(RuntimeError(error))
        return True

    def cancel_pending_ack(
        self,
        project_id: str,
        device_id: str,
        correlation_id: str,
    ) -> None:
        """Cancel a specific pending ack (e.g. on caller-side timeout)."""
        session = self.get(project_id, device_id)
        if session is None:
            return
        future = session.pending_acks.pop(correlation_id, None)
        if future and not future.done():
            future.cancel()

    # ── Messaging ─────────────────────────────────────────────────────────────

    async def send_json(
        self, project_id: str, device_id: str, data: dict
    ) -> bool:
        """
        Send a JSON frame to the device.
        On error the session is removed and False is returned.
        """
        session = self.get(project_id, device_id)
        if session is None:
            return False
        try:
            await session.websocket.send_json(data)
            return True
        except Exception as exc:
            logger.warning(
                "send_json failed project=%s device_id=%s: %s",
                project_id, device_id, exc,
            )
            self.remove(project_id, device_id)
            return False

    async def send_binary(
        self, project_id: str, device_id: str, data: bytes
    ) -> bool:
        """
        Send a binary frame to the device.
        Placeholder for Phase 3 snapshot and debug-stream delivery.
        """
        session = self.get(project_id, device_id)
        if session is None:
            return False
        try:
            await session.websocket.send_bytes(data)
            return True
        except Exception as exc:
            logger.warning(
                "send_binary failed project=%s device_id=%s: %s",
                project_id, device_id, exc,
            )
            self.remove(project_id, device_id)
            return False

    async def broadcast_to_project(self, project_id: str, data: dict) -> int:
        """Send a JSON message to all connected devices in a project."""
        sessions = self.get_project_sessions(project_id)
        results = await asyncio.gather(
            *(self.send_json(project_id, s.device_id, data) for s in sessions),
            return_exceptions=True,
        )
        return sum(1 for r in results if r is True)

    # ── Internal ──────────────────────────────────────────────────────────────

    @staticmethod
    def _cancel_all_acks(session: DeviceSession) -> None:
        for cid, fut in list(session.pending_acks.items()):
            if not fut.done():
                fut.cancel()
        session.pending_acks.clear()


# ─── Singleton ────────────────────────────────────────────────────────────────

manager = ConnectionManager()
