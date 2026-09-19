"""
Realtime message schemas — Phase 2

All messages are JSON text frames.  The `type` field is the discriminator.

Device → Server
───────────────
  hello            first message after connect; carries auth + device info
  pong             reply to a server ping
  set-mode         device requests a mode transition
  sample-ack       device acked a start-sample command
  sample-started   device began collecting a sample
  sample-stopped   device finished collecting a sample
  sample-failed    device failed to collect a sample
  snapshot         snapshot frame placeholder (Phase 3 binary upgrade)
  status           generic device status update
  ack              generic command acknowledgement

Server → Device
───────────────
  hello-ack        response to hello (success or failure + close)
  ping             keepalive sent by server; device must reply pong
  set-mode-ack     acknowledgement of a mode transition
  command          generic command frame (start-sample, snapshot-*, …)
  error            non-fatal error notification

Binary frames (snapshot payloads, debug streams) are carried out-of-band in
Phase 3 and do not use these schemas.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


# ─── Device → Server ──────────────────────────────────────────────────────────

class HelloPayload(BaseModel):
    """
    First message from the device after the WebSocket opens.

    `apiKey`   is looked up in project_device_keys to resolve the project.
    `deviceId` identifies the hardware; the server upserts the device record.
    Extra firmware fields are accepted and ignored via extra="allow".
    """
    model_config = ConfigDict(extra="allow")

    type:                     str             = "hello"
    version:                  str
    apiKey:                   str
    deviceId:                 str
    deviceType:               Optional[str]   = None
    connection:               Optional[str]   = None
    sensors:                  List[Any]       = Field(default_factory=list)
    supportsSnapshotStreaming: bool            = False
    firmwareVersion:          Optional[str]   = None
    protocolVersion:          Optional[str]   = None


class PongMessage(BaseModel):
    """Keepalive reply from device."""
    model_config = ConfigDict(extra="allow")
    type: str = "pong"


class SetModeRequest(BaseModel):
    """Device requests a mode transition.  Valid modes: idle, sampling, inference."""
    model_config = ConfigDict(extra="allow")
    type: str = "set-mode"
    mode: str


class SampleAckMessage(BaseModel):
    """
    Device acknowledges receipt of a start-sample command.

    `correlationId` matches the one sent in the command so the server can
    resolve the pending asyncio.Future from send_and_await_ack.
    """
    model_config = ConfigDict(extra="allow")
    type:          str            = "sample-ack"
    success:       bool
    correlationId: Optional[str]  = None
    error:         Optional[str]  = None
    payload:       Dict[str, Any] = Field(default_factory=dict)


class SampleLifecycleMessage(BaseModel):
    """
    Sample collection lifecycle notification.

    type values: "sample-started" | "sample-stopped" | "sample-failed"
    """
    model_config = ConfigDict(extra="allow")
    type:     str
    label:    Optional[str]       = None
    length:   Optional[int]       = None    # milliseconds
    error:    Optional[str]       = None
    payload:  Dict[str, Any]      = Field(default_factory=dict)


class SnapshotMessage(BaseModel):
    """
    Snapshot frame placeholder.

    Phase 3 upgrade: raw image bytes arrive as a binary frame immediately
    after this JSON header, prefixed by a 4-byte little-endian length.
    For Phase 2 only the JSON header is logged.
    """
    model_config = ConfigDict(extra="allow")
    type:   str = "snapshot"
    width:  Optional[int] = None
    height: Optional[int] = None
    format: Optional[str] = None   # "jpeg" | "rgb888" | …


class StatusMessage(BaseModel):
    """Generic device status update (memory, temperature, battery, …)."""
    model_config = ConfigDict(extra="allow")
    type:   str             = "status"
    data:   Dict[str, Any]  = Field(default_factory=dict)


class GenericAckMessage(BaseModel):
    """Generic device acknowledgement.  May carry a correlationId."""
    model_config = ConfigDict(extra="allow")
    type:          str
    success:       bool
    correlationId: Optional[str]  = None
    error:         Optional[str]  = None
    payload:       Dict[str, Any] = Field(default_factory=dict)


# ─── Server → Device ──────────────────────────────────────────────────────────

class HelloAck(BaseModel):
    """
    Server response to a hello message.

    Success  → {"type": "hello-ack", "success": true,  "id": "<device_pk>", "project_id": "<project_uuid>"}
    Failure  → {"type": "hello-ack", "success": false, "error": "<reason>"}

    On failure the server closes the connection immediately after sending.
    """
    type:    str            = "hello-ack"
    success: bool
    id:      Optional[str] = None    # internal Device PK; present on success
    project_id: Optional[str] = None
    error:   Optional[str] = None    # human-readable reason; present on failure


class PingMessage(BaseModel):
    """Server-initiated keepalive.  Device must reply with a pong."""
    type: str = "ping"


class SetModeAck(BaseModel):
    """Server acknowledgement of a mode transition."""
    type:    str            = "set-mode-ack"
    success: bool
    mode:    Optional[str] = None
    error:   Optional[str] = None


class CommandMessage(BaseModel):
    """
    Generic server-to-device command frame.

    Used for: start-sample, stop-sample, start-snapshot, stop-snapshot,
              start-inference-stream, stop-inference-stream, model-update.
    """
    type:          str
    correlationId: Optional[str]  = None
    payload:       Dict[str, Any] = Field(default_factory=dict)


class ErrorMessage(BaseModel):
    """Server error notification (non-fatal — connection stays open)."""
    type:  str            = "error"
    error: str
    code:  Optional[str] = None
