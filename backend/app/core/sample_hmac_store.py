"""
Ephemeral in-memory per-sample HMAC key store — Phase 4.

When start-sampling is called, a per-sample HMAC key is generated and stored
here with a short TTL.  The sample_token is embedded in the WS command sent to
the device.  When the device uploads the sample via the ingestion HTTP endpoint
it supplies the sample_token in the x-sample-token header; Phase 3's
get_sample_hmac_key() checks this store first so the uploaded sample is verified
with the per-sample key rather than the static project key.

Public surface:
    sample_hmac_store           module-level singleton
    SampleHmacRecord            dataclass returned by issue() / get()
    SampleHmacStore.issue()     generate and store a new per-sample key
    SampleHmacStore.get()       inspect without consuming
    SampleHmacStore.consume()   validate + mark consumed (single-use)
"""
from __future__ import annotations

import secrets
import threading
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, Optional

_DEFAULT_TTL_SECONDS: int = 300   # 5 min — enough for sample collection + upload
_KEY_BYTES: int = 32              # 256-bit key → 64-char hex
_TOKEN_BYTES: int = 24            # 192-bit token → 48-char hex


@dataclass
class SampleHmacRecord:
    token:      str        # 48-char hex sample_token sent to device
    project_id: str
    device_id:  str        # hardware device_id (Device.device_id)
    key_hex:    str        # 64-char hex HMAC key
    expires_at: datetime
    consumed:   bool = False


class SampleHmacStore:
    """
    Thread-safe in-memory store for per-sample HMAC keys.

    Tokens are single-use: once consumed they are marked consumed=True.
    Expired entries are evicted lazily on each mutating call.
    """

    def __init__(self) -> None:
        self._store: Dict[str, SampleHmacRecord] = {}
        self._lock = threading.Lock()

    # ── Public API ────────────────────────────────────────────────────────────

    def issue(
        self,
        project_id: str,
        device_id: str,
        ttl: int = _DEFAULT_TTL_SECONDS,
    ) -> SampleHmacRecord:
        """Generate and store a new per-sample HMAC key."""
        token   = secrets.token_hex(_TOKEN_BYTES)
        key_hex = secrets.token_hex(_KEY_BYTES)
        record = SampleHmacRecord(
            token=token,
            project_id=project_id,
            device_id=device_id,
            key_hex=key_hex,
            expires_at=datetime.utcnow() + timedelta(seconds=ttl),
        )
        with self._lock:
            self._evict_expired()
            self._store[token] = record
        return record

    def get(self, token: str) -> Optional[SampleHmacRecord]:
        """Return the record without consuming it, or None if expired/used/unknown."""
        with self._lock:
            return self._get_valid(token)

    def consume(self, token: str) -> Optional[SampleHmacRecord]:
        """
        Validate and atomically mark the token as consumed.

        Returns the record on first valid call, None on any subsequent call or
        if the token is expired / unknown.
        """
        with self._lock:
            record = self._get_valid(token)
            if record is None:
                return None
            record.consumed = True
            return record

    def __len__(self) -> int:
        with self._lock:
            return len(self._store)

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _get_valid(self, token: str) -> Optional[SampleHmacRecord]:
        """Must be called under self._lock."""
        record = self._store.get(token)
        if record is None:
            return None
        if record.consumed:
            return None
        if datetime.utcnow() >= record.expires_at:
            del self._store[token]
            return None
        return record

    def _evict_expired(self) -> None:
        """Remove all expired entries.  Must be called under self._lock."""
        now = datetime.utcnow()
        stale = [k for k, v in self._store.items() if v.expires_at <= now]
        for k in stale:
            del self._store[k]


# Module-level singleton — shared across all request handlers in the process.
sample_hmac_store = SampleHmacStore()
