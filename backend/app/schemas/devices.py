"""
Pydantic schemas for the Devices domain — Phase 1 + Phase 4.

Phase 4 additions:
  - DeviceListItem / DeviceDetailResponse gain runtime-derived fields
    (is_connected, mode) that are never persisted.
  - DeviceKeySummary gains api_key_prefix for client-side identification.
  - StartSamplingRequest / StartSamplingResponse for the REST command flow.
  - SocketTokenResponse for the browser Studio WS token endpoint.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


# ─── Device schemas ───────────────────────────────────────────────────────────

class DeviceCreateRequest(BaseModel):
    """Register a new device within a project."""
    project_id: str
    name: str
    device_id: Optional[str] = Field(
        default=None,
        description="Stable external hardware identifier (MAC address, serial number, …). "
                    "When supplied, re-registering the same device_id returns the existing record.",
    )
    device_type: Optional[str] = None          # esp32, arduino, rpi, custom, …
    connection: Optional[str] = None           # wifi, ethernet, ble, serial, …
    firmware_version: Optional[str] = None
    protocol_version: Optional[str] = None
    supports_snapshot_streaming: bool = False
    remote_mgmt_host: Optional[str] = None
    sensors: List[Any] = Field(default_factory=list)
    device_metadata: Dict[str, Any] = Field(default_factory=dict)


class DeviceRenameRequest(BaseModel):
    """Rename a device (minimal update used by the UI rename flow)."""
    name: str


class DeviceUpdateRequest(BaseModel):
    """Partial update — all fields optional; only supplied fields are written."""
    name: Optional[str] = None
    device_type: Optional[str] = None
    connection: Optional[str] = None
    firmware_version: Optional[str] = None
    protocol_version: Optional[str] = None
    supports_snapshot_streaming: Optional[bool] = None
    remote_mgmt_host: Optional[str] = None
    sensors: Optional[List[Any]] = None
    device_metadata: Optional[Dict[str, Any]] = None
    deployment_target: Optional[str] = None
    device_profile: Optional[str] = None


class DeviceHeartbeatRequest(BaseModel):
    """Minimum ONLINE contract for any device client."""
    firmware_version: Optional[str] = None
    protocol_version: Optional[str] = None
    ip_address: Optional[str] = None
    remote_mgmt_host: Optional[str] = None
    # Free-form runtime health snapshot (cpu/mem/disk/temp/uptime + sensor presence).
    # All fields optional; merged into device_metadata.diagnostics on the backend.
    diagnostics: Optional[Dict[str, Any]] = None


class DeviceListItem(BaseModel):
    """
    Compact representation used in list responses.

    Phase 4: `is_connected` and `mode` are derived from the runtime
    ConnectionManager; they are never persisted in the database.
    """
    id: str
    project_id: str
    device_id: Optional[str]
    name: str
    device_type: Optional[str]
    connection: Optional[str]
    firmware_version: Optional[str]
    ip_address: Optional[str]
    last_seen: Optional[datetime]
    created_at: datetime

    # Reported sensors — surfaced in the list so callers (e.g. the
    # Data-acquisition "Collect from device" panel) can drive a sensor
    # picker straight from devicesApi.list without a per-device detail fetch.
    sensors: List[Any] = Field(default_factory=list)

    # Deployment targeting (4.5)
    deployment_target: Optional[str] = None
    device_profile: Optional[str] = None
    installed_deployment_id: Optional[str] = None
    installed_model_version: Optional[str] = None

    # Runtime-derived (not from DB)
    is_online: bool = False
    is_connected: bool = False
    mode: Optional[str] = None

    model_config = ConfigDict(from_attributes=True)


class DeviceDetailResponse(BaseModel):
    """
    Full device representation returned by create, get, and update.

    Phase 4: `is_connected` and `mode` are derived from runtime state.
    The `mode` value matches DeviceMode enum strings: idle / sampling / inference.
    """
    id: str
    project_id: str
    device_id: Optional[str]
    name: str
    device_type: Optional[str]
    connection: Optional[str]
    firmware_version: Optional[str]
    protocol_version: Optional[str]
    ip_address: Optional[str]
    supports_snapshot_streaming: bool
    remote_mgmt_host: Optional[str]
    sensors: List[Any]
    device_metadata: Dict[str, Any]
    last_seen: Optional[datetime]
    created_at: datetime
    updated_at: Optional[datetime]

    # Deployment targeting (4.5)
    deployment_target: Optional[str] = None
    device_profile: Optional[str] = None
    installed_deployment_id: Optional[str] = None
    installed_model_version: Optional[str] = None

    # Runtime-derived (not from DB)
    is_online: bool = False
    is_connected: bool = False
    mode: Optional[str] = None

    model_config = ConfigDict(from_attributes=True)


# ─── ProjectDeviceKey schemas ─────────────────────────────────────────────────

class DeviceKeyCreateRequest(BaseModel):
    """Generate a new project-scoped device key."""
    name: Optional[str] = Field(
        default=None,
        description="Human-readable label for this key (e.g. 'Production fleet').",
    )


class DeviceKeyResponse(BaseModel):
    """Returned once on creation — api_key and hmac_key are not retrievable afterwards."""
    id: str
    project_id: str
    name: Optional[str]
    api_key: str
    hmac_key: str
    is_active: bool
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class DeviceKeySummary(BaseModel):
    """
    Safe summary for list endpoints — omits full secret material.

    Phase 4: api_key_prefix exposes the first 8 hex chars of the key so
    users can identify which credential they are looking at without
    retrieving the full secret.
    """
    id: str
    project_id: str
    name: Optional[str]
    api_key_prefix: Optional[str] = None   # first 8 chars of api_key
    is_active: bool
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


# ─── DeviceInferenceLog schemas ───────────────────────────────────────────────

class InferenceLogResponse(BaseModel):
    id: str
    project_id: str
    device_id: Optional[str]    # internal Device PK
    impulse_id: Optional[str]
    result: Dict[str, Any]
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


# ─── Phase 4: start-sampling REST command ─────────────────────────────────────

class StartSamplingRequest(BaseModel):
    """
    Trigger a data-collection sample on a connected device via the REST API.
    The server dispatches a `start-sample` WebSocket command and awaits the
    device ack before returning.
    """
    label: str = Field(
        default="",
        description="Label to attach to the collected sample.",
    )
    length_ms: int = Field(
        default=5000,
        ge=100,
        description="Sample duration in milliseconds.",
    )
    frequency: float = Field(
        default=100.0,
        gt=0,
        description="Sensor sampling frequency in Hz.",
    )
    interval_ms: Optional[int] = Field(
        default=None,
        ge=1,
        description="Sampling interval in ms.  When set, overrides frequency (frequency = 1000 / interval_ms).",
    )
    sensor: str = Field(
        default="sensor",
        description="Sensor name as reported by the device (e.g. 'accelerometer').",
    )
    category: str = Field(
        default="training",
        description="Sample category: training | testing | post-processing | anomaly.",
    )
    timeout: float = Field(
        default=30.0,
        ge=1.0,
        le=120.0,
        description="How long to wait for the device ack (seconds).",
    )


class StartSamplingResponse(BaseModel):
    """Response from the start-sampling endpoint."""
    status: str                           # "started"
    device_id: str                        # internal Device PK
    correlation_id: Optional[str] = None
    sample_token: Optional[str] = None    # per-sample auth token for ingestion upload
    hmac_key: Optional[str] = None        # per-sample HMAC key (hex) for ingestion upload
    ack: Optional[Dict[str, Any]] = None


# ─── Phase 4: socket token ────────────────────────────────────────────────────

class SocketTokenResponse(BaseModel):
    """
    Short-lived single-use token for the browser Studio WebSocket.

    The token is valid for `ttl_seconds` and consumed on first use.
    Phase 5 Studio WS handler calls socket_token_store.consume(token)
    to authenticate the connection.
    """
    token: str
    project_id: str
    expires_at: datetime
    ttl_seconds: int
