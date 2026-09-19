"""
Studio Connection Manager — Phase 5

Maintains an in-memory registry of active browser (Studio) WebSocket
connections keyed by (project_id, connection_id).  One project may have
many concurrent browser tabs open simultaneously.

Thread-safety: asyncio is single-threaded; dict mutations between awaits
are safe without an explicit lock.
"""
from __future__ import annotations

import asyncio
import logging
import uuid
from typing import Dict, List, Tuple

from fastapi import WebSocket

logger = logging.getLogger(__name__)

# Internal connection record — keeps ws + opaque id together.
_Conn = Tuple[str, WebSocket]   # (connection_id, websocket)


class StudioManager:
    """
    Registry of active Studio (browser) WebSocket connections.

    Keyed by (project_id, connection_id) where connection_id is a UUID
    assigned at registration time.
    """

    def __init__(self) -> None:
        # { (project_id, connection_id): WebSocket }
        self._connections: Dict[Tuple[str, str], WebSocket] = {}

    # ── Registration ──────────────────────────────────────────────────────────

    def register(self, project_id: str, ws: WebSocket) -> str:
        """Add a new browser connection.  Returns the assigned connection_id."""
        connection_id = str(uuid.uuid4())
        self._connections[(project_id, connection_id)] = ws
        logger.debug(
            "Studio connected: project=%s connection_id=%s total=%d",
            project_id, connection_id, len(self._connections),
        )
        return connection_id

    def remove(self, project_id: str, connection_id: str) -> None:
        """Remove a browser connection (idempotent)."""
        self._connections.pop((project_id, connection_id), None)
        logger.debug(
            "Studio disconnected: project=%s connection_id=%s",
            project_id, connection_id,
        )

    # ── Lookup ────────────────────────────────────────────────────────────────

    def get_connections(self, project_id: str) -> List[WebSocket]:
        """Return all active WebSocket objects for a project."""
        return [
            ws
            for (pid, _), ws in self._connections.items()
            if pid == project_id
        ]

    def count(self, project_id: str | None = None) -> int:
        if project_id is None:
            return len(self._connections)
        return sum(1 for (pid, _) in self._connections if pid == project_id)

    # ── Broadcast ─────────────────────────────────────────────────────────────

    async def broadcast(self, project_id: str, data: dict) -> int:
        """
        Send a JSON message to all browser connections in a project.

        Per-socket send errors are swallowed — a stale socket must not
        prevent other tabs from receiving the event.  Returns the count of
        successful sends.
        """
        sockets = self.get_connections(project_id)
        sent = 0
        for ws in sockets:
            try:
                await ws.send_json(data)
                sent += 1
            except Exception as exc:
                logger.warning(
                    "Studio broadcast send error project=%s: %s", project_id, exc
                )
        return sent


# ─── Singleton ────────────────────────────────────────────────────────────────

studio_manager = StudioManager()
