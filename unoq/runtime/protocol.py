"""WebSocket protocol, OTA flow, sampling, and snapshot mixin for UnoQRuntime."""
from __future__ import annotations

import asyncio
import hashlib
import hmac as _hmac_mod
import json
import logging
import pathlib
import struct
from typing import Any, Dict, List, Optional

import aiohttp

from .config import (
    PETAL_HOST, PETAL_API_KEY, PETAL_DEVICE_ID, PETAL_FIRMWARE,
    PETAL_USER_EMAIL, PETAL_USER_PASS, UNOQ_SNAPSHOT, UNOQ_SAMPLE_FREQ_HZ,
    HEARTBEAT_INTERVAL, WS_RECONNECT_DELAY, HELLO_TIMEOUT, INFER_TICK_INTERVAL,
)
from .dsp_engine import _DEFAULT_AXES, _build_sample_json, _capture_frame, _make_synthetic_jpeg
from .package_loader import (
    _detect_package_format, _unpack_pxe, _unpack_pe,
    _install_package, _activate_package, _load_inference_module,
)
from .validator import validate_pxe_sections

logger = logging.getLogger("unoq")


def _local_ip() -> str:
    import socket
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
    except Exception:
        return "127.0.0.1"


def _hmac_sign(key_hex: str, body: bytes) -> str:
    return _hmac_mod.new(bytes.fromhex(key_hex), body, hashlib.sha256).hexdigest()


class _ProtocolMixin:
    """Mixin providing all WS, OTA, sampling, and snapshot methods for UnoQRuntime."""

    host:             str
    api_key:          str
    device_id:        str
    firmware_version: str
    _device_pk:       Optional[str]
    _project_id:      Optional[str]
    _jwt:             Optional[str]
    _session:         Optional[aiohttp.ClientSession]
    _running:         bool
    _inference_active: bool
    _inference_task:  Optional[asyncio.Task]
    _stream_sensor:           Optional[str]
    _stream_frequency_hz:     Optional[float]
    _stream_sample_length_ms: Optional[int]
    _pkg_dir:         Optional[pathlib.Path]
    _pkg_format:      str
    _pxe_proc:        Any
    _infer_mod:       Any
    _interpreter:     Any
    _infer_tick:      int
    _snapshot_active: bool
    _snapshot_task:   Optional[asyncio.Task]

    # ── Auth ──────────────────────────────────────────────────────────────────

    async def _login(self) -> None:
        if not PETAL_USER_EMAIL or not PETAL_USER_PASS:
            logger.warning("[auth] credentials not set — profile PATCH will be skipped")
            return
        url = f"{self.host}/api/v1/auth/login"
        try:
            async with self._session.post(
                url, json={"email": PETAL_USER_EMAIL, "password": PETAL_USER_PASS},
                timeout=aiohttp.ClientTimeout(total=10),
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    self._jwt = data.get("access_token")
                    logger.info("[auth] logged in as %s", PETAL_USER_EMAIL)
                else:
                    logger.warning("[auth] login failed HTTP %s", resp.status)
        except Exception as exc:
            logger.warning("[auth] login error: %s", exc)

    # ── Heartbeat ─────────────────────────────────────────────────────────────

    async def _heartbeat_loop(self) -> None:
        while self._running:
            if self._device_pk:
                await self._send_heartbeat()
            await asyncio.sleep(HEARTBEAT_INTERVAL)

    async def _send_heartbeat(self) -> None:
        url  = f"{self.host}/api/v1/devices/{self._device_pk}/heartbeat"
        body = {"firmware_version": self.firmware_version, "ip_address": _local_ip()}
        try:
            async with self._session.post(
                url, json=body, timeout=aiohttp.ClientTimeout(total=10)
            ) as resp:
                if resp.status == 200:
                    logger.debug("[heartbeat] ok")
                else:
                    logger.warning("[heartbeat] HTTP %s", resp.status)
        except Exception as exc:
            logger.warning("[heartbeat] error: %s", exc)

    # ── WebSocket ─────────────────────────────────────────────────────────────

    async def _ws_loop(self) -> None:
        ws_url = self.host.replace("http://", "ws://").replace("https://", "wss://")
        ws_url = f"{ws_url}/ws/device"
        while self._running:
            try:
                logger.info("[ws] connecting %s", ws_url)
                async with self._session.ws_connect(ws_url, heartbeat=30) as ws:
                    if await self._handshake(ws):
                        await self._post_connect(ws)
                        await self._message_loop(ws)
            except Exception as exc:
                logger.warning("[ws] error, retry in %ds: %s", WS_RECONNECT_DELAY, exc)
            finally:
                self._inference_active = False
                self._snapshot_active  = False
            await asyncio.sleep(WS_RECONNECT_DELAY)

    async def _handshake(self, ws) -> bool:
        hello = {
            "type":                     "hello",
            "version":                  "1",
            "apiKey":                   self.api_key,
            "deviceId":                 self.device_id,
            "deviceType":               "unoq",
            "connection":               "wifi",
            "firmwareVersion":          self.firmware_version,
            "protocolVersion":          "2",
            "supportsSnapshotStreaming": UNOQ_SNAPSHOT,
            "sensors":                  [],
        }
        await ws.send_json(hello)
        try:
            msg = await asyncio.wait_for(ws.receive_json(), timeout=HELLO_TIMEOUT)
        except asyncio.TimeoutError:
            logger.error("[ws] hello-ack timeout")
            return False
        if not msg.get("success"):
            logger.error("[ws] hello rejected: %s", msg)
            return False
        self._device_pk  = msg.get("id")
        self._project_id = msg.get("project_id")
        logger.info("[ws] connected device_pk=%s project_id=%s", self._device_pk, self._project_id)
        return True

    async def _post_connect(self, ws) -> None:
        await self._patch_device_profile()
        self._load_installed_package()

    async def _message_loop(self, ws) -> None:
        async for raw in ws:
            if raw.type == aiohttp.WSMsgType.TEXT:
                try:
                    msg = json.loads(raw.data)
                except json.JSONDecodeError:
                    continue
                await self._dispatch(ws, msg)
            elif raw.type in (aiohttp.WSMsgType.CLOSE, aiohttp.WSMsgType.ERROR):
                logger.info("[ws] closed")
                break

    # ── Dispatch ──────────────────────────────────────────────────────────────

    async def _dispatch(self, ws, msg: dict) -> None:
        t   = msg.get("type", "")
        cid = msg.get("correlationId")
        pl  = msg.get("payload") or {}

        if t == "ping":
            await ws.send_json({"type": "pong"})

        elif t == "model-update":
            asyncio.create_task(self._ota_flow(ws, pl, cid))

        elif t == "start-inference-stream":
            if self._inference_task and not self._inference_task.done():
                self._inference_active = False
                self._inference_task.cancel()
            _raw_sensor = pl.get("sensor")
            _raw_freq   = pl.get("frequency")
            _raw_slen   = pl.get("sample_length_ms")
            self._stream_sensor = (
                str(_raw_sensor).strip() if isinstance(_raw_sensor, str) and _raw_sensor.strip()
                else None
            )
            try:
                self._stream_frequency_hz = float(_raw_freq) if _raw_freq is not None else None
                if self._stream_frequency_hz is not None and self._stream_frequency_hz <= 0:
                    self._stream_frequency_hz = None
            except (TypeError, ValueError):
                self._stream_frequency_hz = None
            try:
                self._stream_sample_length_ms = int(_raw_slen) if _raw_slen is not None else None
                if self._stream_sample_length_ms is not None and self._stream_sample_length_ms <= 0:
                    self._stream_sample_length_ms = None
            except (TypeError, ValueError):
                self._stream_sample_length_ms = None
            self._inference_active = True
            self._inference_task = asyncio.create_task(
                self._inference_stream_task(ws, cid)
            )
            logger.info(
                "[inference] stream started cid=%s  sensor=%s  freq=%s  window=%s",
                cid,
                self._stream_sensor           or "pkg-default",
                f"{self._stream_frequency_hz:.1f}Hz" if self._stream_frequency_hz is not None
                    else "pkg-default",
                f"{self._stream_sample_length_ms}ms" if self._stream_sample_length_ms is not None
                    else "pkg-default",
            )

        elif t == "stop-inference-stream":
            self._inference_active = False
            if self._inference_task and not self._inference_task.done():
                self._inference_task.cancel()
            self._inference_task = None
            logger.info("[inference] stream stopped")

        elif t == "start-snapshot":
            self._snapshot_active = True
            if self._snapshot_task and not self._snapshot_task.done():
                self._snapshot_task.cancel()
            self._snapshot_task = asyncio.create_task(
                self._snapshot_stream_task(ws, cid)
            )
            logger.info("[snapshot] started cid=%s", cid)

        elif t == "stop-snapshot":
            self._snapshot_active = False
            if self._snapshot_task and not self._snapshot_task.done():
                self._snapshot_task.cancel()
            self._snapshot_task = None
            logger.info("[snapshot] stopped")

        elif t == "start-sample":
            asyncio.create_task(self._sample_real(ws, pl, cid))

        elif t == "stop-sample":
            pass  # _sample_real runs to completion; stop is advisory on short windows

        else:
            logger.debug("[ws] unhandled type=%r", t)

    # ── Device profile PATCH ──────────────────────────────────────────────────

    async def _patch_device_profile(self) -> None:
        if not self._device_pk or not self._jwt:
            logger.warning("[profile] skipping PATCH — no device_pk or JWT")
            return
        url  = f"{self.host}/api/v1/devices/{self._device_pk}"
        body = {
            "device_profile":    "unoq",
            "deployment_target": "unoq",
            "device_type":       "unoq",
        }
        try:
            async with self._session.patch(
                url, json=body,
                headers={"Authorization": f"Bearer {self._jwt}"},
                timeout=aiohttp.ClientTimeout(total=10),
            ) as resp:
                if resp.status == 200:
                    logger.info("[profile] device_profile=unoq deployment_target=unoq applied")
                else:
                    logger.warning("[profile] PATCH failed HTTP %s", resp.status)
        except Exception as exc:
            logger.warning("[profile] PATCH error: %s", exc)

    # ── OTA ───────────────────────────────────────────────────────────────────

    async def _ota_flow(self, ws, payload: dict, cid: Optional[str]) -> None:
        url           = payload.get("url", "")
        version       = payload.get("version", "unknown")
        deployment_id = payload.get("deployment_id", "")

        logger.info("[ota] update received version=%r deployment_id=%r", version, deployment_id)

        if not url:
            await ws.send_json({"type": "update-rejected", "error": "missing url", "correlationId": cid})
            logger.warning("[ota] rejected — no url in payload")
            return

        await ws.send_json({"type": "update-accepted", "correlationId": cid})

        try:
            await ws.send_json({"type": "update-download-started", "correlationId": cid})
            logger.info("[ota] downloading from %s", url)

            async with self._session.get(url, timeout=aiohttp.ClientTimeout(total=120)) as resp:
                resp.raise_for_status()
                pkg_bytes = await resp.read()

            logger.info("[ota] downloaded %d bytes", len(pkg_bytes))

            await ws.send_json({"type": "update-install-started", "correlationId": cid})

            self._stop_pxe_runner()   # free file handles before overwriting package dir

            pkg_fmt  = _detect_package_format(pkg_bytes)
            sections = _unpack_pxe(pkg_bytes) if pkg_fmt == "pxe" else _unpack_pe(pkg_bytes)

            # Validate cross-section invariants for PXE packages before writing to disk.
            if pkg_fmt == "pxe":
                validate_pxe_sections(sections, strict=True)

            pkg_dir  = _install_package(deployment_id, sections)
            _activate_package(deployment_id, version, pkg_fmt)

            self._load_installed_package(pkg_dir, pkg_fmt)
            self.firmware_version = version

            await ws.send_json({
                "type":          "update-install-succeeded",
                "version":       version,
                "correlationId": cid,
            })
            logger.info("[ota] done version=%r", version)

        except Exception as exc:
            logger.error("[ota] failed: %s", exc)
            await ws.send_json({
                "type":          "update-install-failed",
                "message":       str(exc),
                "correlationId": cid,
            })

    # ── Inference stream ──────────────────────────────────────────────────────

    async def _inference_stream_task(self, ws, cid: Optional[str]) -> None:
        logger.info("[inference] mode=local  package=%s", self._pkg_dir)
        self._infer_tick = 0
        while self._inference_active:
            result = await asyncio.get_event_loop().run_in_executor(
                None, self._run_inference_tick
            )
            if result is not None:
                payload = {k: v for k, v in result.items() if k != "debug"}
                _scfg: dict = {}
                if self._stream_sensor is not None:
                    _scfg["sensor"] = self._stream_sensor
                if self._stream_frequency_hz is not None:
                    _scfg["frequency_hz"] = self._stream_frequency_hz
                if self._stream_sample_length_ms is not None:
                    _scfg["sample_length_ms"] = self._stream_sample_length_ms
                if _scfg:
                    payload["stream_config"] = _scfg
                await ws.send_json({
                    "type":          "inference-result",
                    "correlationId": cid,
                    "payload":       payload,
                })
                top = ""
                if result.get("is_fomo"):
                    dets = result.get("detections", [])
                    top  = f"count={result.get('count', 0)}  " + ", ".join(
                        f"{d['label']} ({d['confidence']*100:.0f}%)" for d in dets[:3]
                    )
                else:
                    top = f"{result.get('label')} ({result.get('confidence', 0)*100:.0f}%)"
                logger.info("[infer] tick=%d  %s", self._infer_tick, top)
            self._infer_tick += 1
            await asyncio.sleep(INFER_TICK_INTERVAL)

    # ── Snapshot stream ───────────────────────────────────────────────────────

    async def _snapshot_stream_task(self, ws, cid: Optional[str]) -> None:
        """
        Stream JPEG frames to Studio at ~10 fps.
        Source priority: UNOQ_TEST_IMAGE → OpenCV camera → synthetic test pattern.
        Protocol: JSON metadata frame followed by 4-byte LE length + JPEG bytes.
        """
        logger.info("[snapshot] stream task running")
        try:
            while self._snapshot_active and not ws.closed:
                frame = _capture_frame(self._pkg_dir) or _make_synthetic_jpeg()
                await ws.send_json({
                    "type":          "snapshot",
                    "width":         96,
                    "height":        96,
                    "format":        "jpeg",
                    "correlationId": cid,
                })
                await ws.send_bytes(struct.pack("<I", len(frame)) + frame)
                await asyncio.sleep(0.1)   # ~10 fps
        except Exception as exc:
            logger.error("[snapshot] stream error: %s", exc)
            self._snapshot_active = False
        logger.info("[snapshot] stream task ended")

    # ── Sampling ──────────────────────────────────────────────────────────────

    async def _sample_real(self, ws, payload: dict, cid: Optional[str]) -> None:
        """
        Capture sensor data and upload to the ingestion endpoint.

        Data source: synthetic sine-wave accelerometer samples keyed by label.
        To use real hardware: replace _sine_accel_values() with driver calls
        (e.g. smbus2 for I2C IMU, pyserial for UART sensor module).
        """
        label      = payload.get("label", "unknown")
        length_ms  = int(payload.get("length", 2000))
        freq_hz    = float(payload.get("frequency") or UNOQ_SAMPLE_FREQ_HZ)
        hmac_key   = payload.get("hmacKey")
        sample_tok = payload.get("sampleToken")
        path       = payload.get("path")

        axes: List[Dict[str, str]] = list(_DEFAULT_AXES)
        if self._pkg_dir:
            try:
                dsp_cfg  = json.loads((self._pkg_dir / "dsp_config.json").read_text())
                cfg_axes = dsp_cfg.get("axes", [])
                if cfg_axes:
                    axes = cfg_axes
                if not payload.get("frequency") and dsp_cfg.get("interval_ms"):
                    freq_hz = 1000.0 / dsp_cfg["interval_ms"]
            except Exception:
                pass

        logger.info("[sample] start label=%r length=%dms freq=%.1fHz axes=%d",
                    label, length_ms, freq_hz, len(axes))

        await ws.send_json({"type": "sample-ack", "success": True, "correlationId": cid})
        await ws.send_json({
            "type":          "sample-started",
            "label":         label,
            "length":        length_ms,
            "correlationId": cid,
        })

        try:
            body = _build_sample_json(label, length_ms, freq_hz, axes)
            await asyncio.sleep(min(length_ms / 1000.0, 5.0))
            await self._upload_sample(body, label, hmac_key, sample_tok, path)
            logger.info("[sample] done label=%r size=%d bytes", label, len(body))
            await ws.send_json({"type": "sample-stopped", "label": label, "correlationId": cid})
        except Exception as exc:
            logger.error("[sample] failed: %s", exc)
            await ws.send_json({"type": "sample-failed", "error": str(exc), "correlationId": cid})

    async def _upload_sample(
        self,
        body: bytes,
        label: str,
        hmac_key: Optional[str],
        sample_token: Optional[str],
        path: Optional[str] = None,
    ) -> None:
        # `path` is sent by the backend on the start-sample command (Phase 4;
        # see device_client/PROTOCOL.md) and is absolute from host root. Older
        # backends may omit it — fall back to the legacy hardcoded training
        # path for those, but log it since it silently miscategorises testing/
        # anomaly samples into training (P0-A5).
        if path:
            url = f"{self.host}{path}"
        else:
            url = f"{self.host}/api/v1/ingestion/training/data"
            logger.warning(
                "[sample] start-sample command had no 'path' field; "
                "falling back to the training ingestion path (backend may be outdated)"
            )
        headers: Dict[str, str] = {
            "Content-Type": "application/json",
            "x-api-key":    self.api_key,
        }
        if sample_token and hmac_key:
            headers["x-sample-token"] = sample_token
            headers["x-signature"]    = _hmac_sign(hmac_key, body)
        if label:
            headers["x-label"] = label

        async with self._session.post(
            url, data=body, headers=headers,
            timeout=aiohttp.ClientTimeout(total=30),
        ) as resp:
            if resp.status not in (200, 201):
                raise RuntimeError(f"upload HTTP {resp.status}: {(await resp.text())[:120]}")
            data = await resp.json()
            tag  = "dedup" if data.get("duplicate") else "new"
            logger.info(
                "[sample] uploaded(%s)  id=%s  hash=%s",
                tag,
                data.get("id", "?"),
                (data.get("payload_hash") or "")[:12],
            )
