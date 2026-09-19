"""
Ephemeral in-memory socket-token store — Phase 4.

Issues short-lived single-use tokens that authorise a browser to open the
Studio WebSocket connection.  Tokens never touch the database; they live only
in process memory with a configurable TTL.

Public surface:
    socket_token_store          module-level singleton
    SocketTokenRecord           dataclass returned by issue() and peek()
    SocketTokenStore.issue()    generate and store a new token
    SocketTokenStore.peek()     inspect without consuming (for latency checks)
    SocketTokenStore.consume()  validate + mark used in one atomic step

Phase 5 Studio WS auth:
    The WS handshake handler calls consume(token) and gets back the
    SocketTokenRecord (or None on expiry/reuse).  project_id and issued_by
    are used to scope the connection without any DB round-trip.
"""
from __future__ import annotations

import secrets
import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, Optional

_DEFAULT_TTL_SECONDS: int = 60
_TOKEN_BYTES: int = 24          # 48-char hex string


@dataclass
class SocketTokenRecord:
    value:      str
    project_id: str
    issued_by:  str             # user_id who requested the token
    expires_at: datetime
    used:       bool = False


class SocketTokenStore:
    """
    Thread-safe in-memory store for short-lived studio socket tokens.

    Tokens are single-use: once consumed they are marked `used=True` and any
    subsequent consume() call returns None.  Expired tokens are evicted lazily
    on each mutating call so the store never grows unbounded.
    """

    def __init__(self) -> None:
        self._store: Dict[str, SocketTokenRecord] = {}
        self._lock = threading.Lock()

    # ── Public API ────────────────────────────────────────────────────────────

    def issue(
        self,
        project_id: str,
        issued_by: str,
        ttl: int = _DEFAULT_TTL_SECONDS,
    ) -> SocketTokenRecord:
        """
        Generate and store a new token.

        Returns the SocketTokenRecord (value, project_id, issued_by,
        expires_at).  The caller should return `value` and `expires_at` to
        the client; the other fields are server-side only.
        """
        value = secrets.token_hex(_TOKEN_BYTES)
        record = SocketTokenRecord(
            value=value,
            project_id=project_id,
            issued_by=issued_by,
            expires_at=datetime.utcnow() + timedelta(seconds=ttl),
        )
        with self._lock:
            self._evict_expired()
            self._store[value] = record
        return record

    def peek(self, value: str) -> Optional[SocketTokenRecord]:
        """
        Return the token record without consuming it, or None if expired/used.
        Used for lightweight checks; prefer consume() for actual auth.
        """
        with self._lock:
            return self._get_valid(value)

    def consume(self, value: str) -> Optional[SocketTokenRecord]:
        """
        Validate and atomically mark the token as used (single-use enforcement).

        Returns the record on first valid call, None on any subsequent call or
        if the token is expired / unknown.
        """
        with self._lock:
            record = self._get_valid(value)
            if record is None:
                return None
            record.used = True
            return record

    def __len__(self) -> int:
        with self._lock:
            return len(self._store)

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _get_valid(self, value: str) -> Optional[SocketTokenRecord]:
        """Must be called under self._lock."""
        record = self._store.get(value)
        if record is None:
            return None
        if record.used:
            return None
        if datetime.utcnow() >= record.expires_at:
            del self._store[value]
            return None
        return record

    def _evict_expired(self) -> None:
        """Remove all expired entries.  Must be called under self._lock."""
        now = datetime.utcnow()
        stale = [k for k, v in self._store.items() if v.expires_at <= now]
        for k in stale:
            del self._store[k]


# Module-level singleton — shared across all request handlers in the process.
socket_token_store = SocketTokenStore()
