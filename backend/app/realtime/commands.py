"""
Device command helpers — Phase 2

Centralised payload builders and send helpers for every server-to-device
command.  No REST surface here — callers are WebSocket handlers and future
Phase 4 REST endpoints that trigger device actions.

Step 2.9: payload builders + fire-and-forget senders
Step 2.10: send_and_await_ack — request/ack primitive with timeout

All `build_*` functions return a plain dict (no Pydantic) so they can be
composed freely before passing to send_and_await_ack or send_json.
"""
from __future__ import annotations

import asyncio
import uuid
import logging
from typing import Any, Dict, Optional

from app.realtime.connection_manager import manager

logger = logging.getLogger(__name__)

_DEFAULT_ACK_TIMEOUT: float = 30.0  # seconds


# ─── Internal builder helper ──────────────────────────────────────────────────

def _cmd(
    type_: str,
    payload: Optional[Dict[str, Any]] = None,
    correlation_id: Optional[str] = None,
) -> dict:
    msg: Dict[str, Any] = {"type": type_, "payload": payload or {}}
    if correlation_id:
        msg["correlationId"] = correlation_id
    return msg


def _new_cid() -> str:
    return str(uuid.uuid4())


# ─── Payload builders (Step 2.9) ──────────────────────────────────────────────

def build_sample_request(
    label: str,
    length_ms: int,
    frequency: float,
    sensor: str,
    correlation_id: Optional[str] = None,
    *,
    hmac_key: Optional[str] = None,
    sample_token: Optional[str] = None,
    path: Optional[str] = None,
) -> dict:
    """
    Start a data-collection sample on the device.

    Phase 4: hmac_key, sample_token, and path are optionally embedded so the
    device can authenticate its subsequent ingestion upload with the per-sample
    key rather than the static project key.
    """
    payload: Dict[str, Any] = {
        "label": label,
        "length": length_ms,
        "frequency": frequency,
        "sensor": sensor,
    }
    if hmac_key is not None:
        payload["hmacKey"] = hmac_key
    if sample_token is not None:
        payload["sampleToken"] = sample_token
    if path is not None:
        payload["path"] = path
    return _cmd("start-sample", payload, correlation_id or _new_cid())


def build_start_snapshot(correlation_id: Optional[str] = None) -> dict:
    """Ask the device to begin streaming snapshot frames (Phase 3)."""
    return _cmd("start-snapshot", {}, correlation_id or _new_cid())


def build_stop_snapshot(correlation_id: Optional[str] = None) -> dict:
    return _cmd("stop-snapshot", {}, correlation_id or _new_cid())


def build_start_inference_stream(
    correlation_id: Optional[str] = None,
    fomo_threshold: Optional[float] = None,
    sensor: Optional[str] = None,
    frequency: Optional[float] = None,
    sample_length_ms: Optional[int] = None,
) -> dict:
    """Ask the device to stream live inference results (Phase 3)."""
    payload: dict = {}
    if fomo_threshold is not None:
        payload["fomo_threshold"] = fomo_threshold
    if sensor is not None:
        payload["sensor"] = sensor
    if frequency is not None:
        payload["frequency"] = frequency
    if sample_length_ms is not None:
        payload["sample_length_ms"] = sample_length_ms
    return _cmd("start-inference-stream", payload, correlation_id or _new_cid())


def build_stop_inference_stream(correlation_id: Optional[str] = None) -> dict:
    return _cmd("stop-inference-stream", {}, correlation_id or _new_cid())


def build_model_update_request(
    model_url: str,
    version: str,
    correlation_id: Optional[str] = None,
    *,
    deployment_id: str = "",
    deployment_target: str = "",
    device_profile: str = "",
) -> dict:
    """
    Tell the device to download and apply a new model.

    Expected device behavior:
      1. Ack or reject via 'update-accepted' / 'update-rejected'.
      2. Download from url; send 'update-download-started'.
      3. Flash; send 'update-install-started'.
      4. Send 'update-install-succeeded' (includes version) or
         'update-install-failed' (includes message).
    """
    return _cmd(
        "model-update",
        {
            "url": model_url,
            "version": version,
            "deployment_id": deployment_id,
            "deployment_target": deployment_target,
            "device_profile": device_profile or "",
        },
        correlation_id or _new_cid(),
    )


# ─── Fire-and-forget senders (Step 2.9) ──────────────────────────────────────
# Use these when no ack tracking is needed (e.g. broadcast, best-effort push).

async def send_sample_request(
    project_id: str,
    device_id: str,
    label: str,
    length_ms: int,
    frequency: float,
    sensor: str,
) -> bool:
    return await manager.send_json(
        project_id, device_id,
        build_sample_request(label, length_ms, frequency, sensor),
    )


async def send_start_snapshot(project_id: str, device_id: str) -> bool:
    return await manager.send_json(project_id, device_id, build_start_snapshot())


async def send_stop_snapshot(project_id: str, device_id: str) -> bool:
    return await manager.send_json(project_id, device_id, build_stop_snapshot())


async def send_start_inference_stream(
    project_id: str,
    device_id: str,
    fomo_threshold: Optional[float] = None,
    sensor: Optional[str] = None,
    frequency: Optional[float] = None,
    sample_length_ms: Optional[int] = None,
) -> bool:
    return await manager.send_json(
        project_id, device_id,
        build_start_inference_stream(
            fomo_threshold=fomo_threshold,
            sensor=sensor,
            frequency=frequency,
            sample_length_ms=sample_length_ms,
        ),
    )


async def send_stop_inference_stream(project_id: str, device_id: str) -> bool:
    return await manager.send_json(project_id, device_id, build_stop_inference_stream())


async def send_model_update(
    project_id: str,
    device_id: str,
    model_url: str,
    version: str,
) -> bool:
    return await manager.send_json(
        project_id, device_id,
        build_model_update_request(model_url, version),
    )


async def send_deployment_update(
    project_id: str,
    device_id: str,
    model_url: str,
    version: str,
    *,
    deployment_id: str = "",
    deployment_target: str = "",
    device_profile: str = "",
) -> bool:
    """Fire-and-forget OTA push with full targeting metadata."""
    return await manager.send_json(
        project_id, device_id,
        build_model_update_request(
            model_url, version,
            deployment_id=deployment_id,
            deployment_target=deployment_target,
            device_profile=device_profile,
        ),
    )


# ─── Request/ack primitive (Step 2.10) ────────────────────────────────────────

async def send_and_await_ack(
    project_id: str,
    device_id: str,
    command: dict,
    timeout: float = _DEFAULT_ACK_TIMEOUT,
) -> Dict[str, Any]:
    """
    Send a command to the device and wait for a matching ack.

    The command dict must already contain a "correlationId" key (all build_*
    helpers add one automatically).  The message router in ws_device.py is
    responsible for calling manager.resolve_ack / manager.reject_ack when the
    matching ack arrives.

    Returns the ack payload dict on success.
    Raises asyncio.TimeoutError if the device does not reply within `timeout`.
    Raises RuntimeError if:
      - the device is not connected
      - the send fails
      - the device sends a failure ack

    Usage:
        result = await send_and_await_ack(
            project_id, device_id,
            build_sample_request("wave", 5000, 100.0, "accelerometer"),
            timeout=20.0,
        )
    """
    correlation_id: str = command.get("correlationId") or _new_cid()
    command = {**command, "correlationId": correlation_id}

    loop = asyncio.get_running_loop()
    future: asyncio.Future = loop.create_future()

    if not manager.set_pending_ack(project_id, device_id, correlation_id, future):
        raise RuntimeError(
            f"Device '{device_id}' is not connected to project '{project_id}'"
        )

    try:
        sent = await manager.send_json(project_id, device_id, command)
        if not sent:
            future.cancel()
            raise RuntimeError(
                f"Failed to deliver command to device '{device_id}'"
            )

        logger.debug(
            "Awaiting ack cid=%s device=%s project=%s timeout=%.1fs",
            correlation_id, device_id, project_id, timeout,
        )
        return await asyncio.wait_for(asyncio.shield(future), timeout=timeout)

    except asyncio.TimeoutError:
        manager.cancel_pending_ack(project_id, device_id, correlation_id)
        logger.warning(
            "Ack timeout cid=%s device=%s project=%s",
            correlation_id, device_id, project_id,
        )
        raise

    except Exception:
        manager.cancel_pending_ack(project_id, device_id, correlation_id)
        raise
