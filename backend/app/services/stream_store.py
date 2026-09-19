"""
Ephemeral in-memory store for active debug streams (snapshot / inference).

No DB, no persistence — streams are lost on restart, which is intentional
(the device will need to restart its own stream too).
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, Optional


class StreamType(str, Enum):
    snapshot  = "snapshot"
    inference = "inference"
    sensor    = "sensor"


@dataclass
class StreamState:
    project_id:       str
    device_id:        str
    stream_type:      StreamType
    stream_id:        str
    last_keepalive:   datetime      = field(default_factory=datetime.utcnow)
    sensor:           Optional[str]   = None
    frequency:        Optional[float] = None
    sample_length_ms: Optional[int]   = None


class StreamStore:
    _TIMEOUT_SECS = 30

    def __init__(self) -> None:
        self._store: Dict[str, StreamState] = {}

    def start(
        self,
        project_id: str,
        device_id: str,
        stream_type: StreamType,
        *,
        sensor: Optional[str] = None,
        frequency: Optional[float] = None,
        sample_length_ms: Optional[int] = None,
    ) -> StreamState:
        state = StreamState(
            project_id=project_id,
            device_id=device_id,
            stream_type=stream_type,
            stream_id=str(uuid.uuid4()),
            sensor=sensor,
            frequency=frequency,
            sample_length_ms=sample_length_ms,
        )
        self._store[device_id] = state
        return state

    def get(self, device_id: str) -> Optional[StreamState]:
        return self._store.get(device_id)

    def refresh(self, device_id: str) -> bool:
        state = self._store.get(device_id)
        if state is None:
            return False
        state.last_keepalive = datetime.utcnow()
        return True

    def stop(self, device_id: str) -> Optional[StreamState]:
        return self._store.pop(device_id, None)

    def expired(self, device_id: str) -> bool:
        state = self._store.get(device_id)
        if state is None:
            return False
        return (datetime.utcnow() - state.last_keepalive) > timedelta(seconds=self._TIMEOUT_SECS)


stream_store = StreamStore()
