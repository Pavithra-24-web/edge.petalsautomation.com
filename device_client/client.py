"""
Petal Edge — Python device client (simulation + validation reference)

This is a REFERENCE / validation client, not shipped device firmware. Use it to
exercise and validate the backend contract (WS hello, heartbeat, sampling,
snapshot/inference streams) from a workstation; real devices implement the same
contract in their own firmware (see docs/device-connection-contract.md).

Two-step setup (recommended — the edge-impulse-daemon equivalent):
    1. Provision once (interactive: login → pick project → device key):
           python provision.py
       This writes device_client/config.json (chmod 600) with host + api_key +
       device_id. Your password is typed at a hidden prompt and never stored —
       only the device api_key is written to the file.
    2. Run the device — config.json is picked up automatically, no env vars:
           python client.py

Config precedence for every setting: environment variable > config.json >
built-in default. So provisioning replaces the manual env-var workflow below,
but an explicit env var still overrides the file when you need to.

NOTE: Without a config.json (and without env overrides), PETAL_HOST and
PETAL_API_KEY fall back to placeholder defaults (``http://192.168.1.22:8010``
and ``ef_changeme``) that will NOT connect. Either run provision.py (recommended)
or override both — point PETAL_HOST at your backend and set PETAL_API_KEY to a
real project device key (create one in the Devices page or via
POST /api/v1/devices/project/{project_id}/keys).

Protocol source of truth:
  backend/app/realtime/schemas.py              — WS message types
  backend/app/realtime/commands.py             — server→device command shapes
  backend/app/services/ingestion.py            — sample JSON format + HMAC
  backend/app/schemas/devices.py               — DeviceUpdateRequest fields
  backend/app/api/v1/endpoints/auth.py         — POST /api/v1/auth/login
  backend/app/services/compatibility.py        — deployment_target / device_profile matching

Quick start (override BOTH placeholder defaults with your own values):
    pip install aiohttp
    PETAL_HOST=http://<your-backend-host>:8010 \\
    PETAL_API_KEY=ef_<your-project-device-key> \\
    PETAL_DEVICE_ID=python-dev-001 \\
    python client.py

UNO Q validation mode (also set user creds so profile PATCH is applied):
    PETAL_PROFILE=unoq \\
    PETAL_TARGET=unoq \\
    PETAL_USER_EMAIL=you@example.com \\
    PETAL_USER_PASS=yourpassword \\
    python client.py

Snapshot mode:
    PETAL_SNAPSHOT=1 python client.py

Inference stream — real backend DSP + TFLite (meaningful predictions):
    PETAL_USER_EMAIL=you@example.com \\
    PETAL_USER_PASS=yourpassword \\
    PETAL_MODEL_ID=<trained-model-uuid> \\
    python client.py

    Requires: numpy + scipy installed (same deps as backend DSP worker).
    The client imports DSPProcessor directly from backend/app/ml/dsp/processor.py
    so that feature extraction exactly matches the training pipeline.
    Inference results are driven by real sensor waveforms → real model scores.

    Inference starts when the Studio UI sends a start-inference-stream command.

DSP parity assumptions (real inference mode):
  • model_metadata["dsp_blocks"]  — must be present (stored by training worker)
  • model_metadata["frequency_hz"] — sampling frequency used during training
  • model_metadata["window_size_ms"] — sensor window length used during training
  These fields are written by training_worker._common_meta.  If absent (models
  trained before this was added), the client logs a WARNING, uses client defaults,
  and may produce a feature-count mismatch that causes it to refuse the inference
  call with a clear ERROR rather than silently sending wrong data.
"""
from __future__ import annotations

import asyncio
import hashlib
import hmac as _hmac_mod
import importlib.util
import io
import json
import logging
import math
import os
import pathlib
import socket
import struct
import sys
import time
import types
from typing import Any, Dict, List, Optional

import aiohttp

logger = logging.getLogger("petal.device")

# ── Backend DSP import (real inference mode) ──────────────────────────────────
# Add backend/ to sys.path so DSPProcessor can be imported directly.
# This gives exact feature parity with the training pipeline at zero duplication.
_BACKEND_SRC = pathlib.Path(__file__).parent.parent / "backend"
if str(_BACKEND_SRC) not in sys.path:
    sys.path.insert(0, str(_BACKEND_SRC))
try:
    from app.ml.dsp.processor import DSPProcessor as _DSPProcessor
    import numpy as _np
    _DSP_AVAILABLE = True
    _DSP_IMPORT_ERROR: Optional[str] = None
except ImportError as _dsp_import_err:
    _DSPProcessor = None  # type: ignore[assignment,misc]
    _np = None            # type: ignore[assignment]
    _DSP_AVAILABLE = False
    _DSP_IMPORT_ERROR: Optional[str] = str(_dsp_import_err)  # type: ignore[no-redef]

# ── Provisioned config file (device_client/config.json) ───────────────────────
# Written by provision.py (the edge-impulse-daemon equivalent). Precedence for
# every setting it carries is:
#     environment variable   >   config.json   >   built-in default
# so an explicit env var always wins, the provisioned file fills the gaps, and
# the placeholder defaults apply only when neither is present. When no config
# file exists this is a complete no-op: behavior is identical to reading the
# environment directly, preserving the existing env-var workflow.
_CONFIG_PATH = pathlib.Path(__file__).parent / "config.json"

# Short human keys written by provision.py → the PETAL_* names used below.
_CONFIG_KEY_ALIASES = {
    "host":        "PETAL_HOST",
    "api_key":     "PETAL_API_KEY",
    "device_id":   "PETAL_DEVICE_ID",
    "device_type": "PETAL_DEVICE_TYPE",
}


def _load_config_file() -> Dict[str, str]:
    """Load device_client/config.json into a {PETAL_*: value} dict.

    Returns {} when the file is absent, unreadable, or malformed — a bad file
    must never stop the client from starting (it just falls back to env/defaults).
    Both the short keys emitted by provision.py and raw PETAL_* keys are accepted.
    """
    if not _CONFIG_PATH.exists():
        return {}
    try:
        with open(_CONFIG_PATH, "r", encoding="utf-8") as fh:
            raw = json.load(fh)
    except Exception as exc:
        logger.warning("[config] ignoring %s (could not parse: %s)", _CONFIG_PATH, exc)
        return {}
    if not isinstance(raw, dict):
        logger.warning("[config] ignoring %s (expected a JSON object)", _CONFIG_PATH)
        return {}
    out: Dict[str, str] = {}
    for key, value in raw.items():
        if value is None:
            continue
        env_key = _CONFIG_KEY_ALIASES.get(key) or (key if key.startswith("PETAL_") else None)
        if env_key:
            out[env_key] = str(value)
    return out


_FILE_CONFIG: Dict[str, str] = _load_config_file()


def _cfg(key: str, default: str) -> str:
    """Resolve a setting as: environment > config.json > built-in default."""
    if key in os.environ:
        return os.environ[key]
    if key in _FILE_CONFIG:
        return _FILE_CONFIG[key]
    return default


# ── Config (env > config.json > built-in default) ─────────────────────────────
PETAL_HOST        = _cfg("PETAL_HOST",       "http://192.168.1.22:8010")  # PLACEHOLDER — override
PETAL_API_KEY     = _cfg("PETAL_API_KEY",    "ef_changeme")               # PLACEHOLDER — override with a real project device key
PETAL_DEVICE_ID   = _cfg("PETAL_DEVICE_ID",  "python-dev-001")
PETAL_DEVICE_TYPE = _cfg("PETAL_DEVICE_TYPE","python")
PETAL_FIRMWARE    = os.environ.get("PETAL_FIRMWARE",   "1.0.0")
PETAL_PROFILE     = os.environ.get("PETAL_PROFILE",    "")     # e.g. "unoq"
PETAL_TARGET      = os.environ.get("PETAL_TARGET",     "")     # e.g. "unoq"
PETAL_SNAPSHOT    = os.environ.get("PETAL_SNAPSHOT",   "0") in ("1","true","yes")
PETAL_LABEL_MODE  = os.environ.get("PETAL_LABEL_MODE", "cycle")  # cycle | random | fixed
PETAL_FIXED_LABEL = os.environ.get("PETAL_FIXED_LABEL","idle")
PETAL_USER_EMAIL  = os.environ.get("PETAL_USER_EMAIL", "")
PETAL_USER_PASS   = os.environ.get("PETAL_USER_PASS",  "")
PETAL_UPDATE_SIM         = os.environ.get("PETAL_UPDATE_SIM",         "1") in ("1","true","yes")
PETAL_FORCE_LOCAL_PACKAGE= os.environ.get("PETAL_FORCE_LOCAL_PACKAGE","0") in ("1","true","yes")
PETAL_MARKER_DIR  = os.environ.get("PETAL_MARKER_DIR", str(pathlib.Path(__file__).parent / ".state"))
PETAL_PE_DIR      = os.environ.get("PETAL_PE_DIR",     str(pathlib.Path(__file__).parent / ".model"))
PETAL_IMAGE_PATH  = os.environ.get("PETAL_IMAGE_PATH", "").strip()
# Inference: set to a TrainedModel UUID to use real backend inference.
# Requires PETAL_USER_EMAIL + PETAL_USER_PASS so the client can get a JWT.
# Without a model ID the client will try to infer the latest project model
# after WebSocket connect; if that fails it falls back to local simulation.
PETAL_MODEL_ID    = os.environ.get("PETAL_MODEL_ID", "")

HEARTBEAT_INTERVAL    = int(os.environ.get("PETAL_HB_INTERVAL",  "20"))
WS_RECONNECT_DELAY    = int(os.environ.get("PETAL_WS_RETRY",     "5"))
HEARTBEAT_RETRY_DELAYS = (5, 10, 30)
HELLO_TIMEOUT         = 10

# Sensor simulation parameters
_SAMPLE_LABELS    = ["idle", "running", "walking", "anomaly"]
_INFER_CLASSES    = ["idle", "running", "walking", "anomaly"]
_SAMPLE_FREQ_HZ   = 62.5
_SAMPLE_AXES      = [
    {"name": "accX", "units": "m/s2"},
    {"name": "accY", "units": "m/s2"},
    {"name": "accZ", "units": "m/s2"},
]


# ── Sensor simulation helpers ─────────────────────────────────────────────────

def _read_imu_values(n_samples: int, freq_hz: float) -> Optional[List[float]]:
    """Read real accelerometer via smbus2 (MPU-6050 / LSM6DS / similar).
    Returns flat [ax,ay,az,...] list or None if hardware is unavailable."""
    try:
        import smbus2  # type: ignore
        bus = smbus2.SMBus(1)
        # MPU-6050: wake up, then read 6 bytes from ACCEL_XOUT_H (0x3B)
        bus.write_byte_data(0x68, 0x6B, 0)  # PWR_MGMT_1 = 0 (wake)
        out: List[float] = []
        dt = 1.0 / freq_hz
        import time as _time
        for _ in range(n_samples):
            raw = bus.read_i2c_block_data(0x68, 0x3B, 6)
            def _s16(hi, lo): v = (hi << 8) | lo; return v - 65536 if v > 32767 else v
            ax = round(_s16(raw[0], raw[1]) / 16384.0 * 9.81, 4)
            ay = round(_s16(raw[2], raw[3]) / 16384.0 * 9.81, 4)
            az = round(_s16(raw[4], raw[5]) / 16384.0 * 9.81, 4)
            out.extend([ax, ay, az])
            _time.sleep(dt)
        bus.close()
        return out
    except Exception:
        return None


def _sine_accel_values(label: str, n_samples: int, freq_hz: float) -> List[float]:
    """Simulation fallback — deterministic sine-wave accelerometer samples."""
    profile = {
        "idle":    (0.05, 0.05, 9.81, 0.2),
        "running": (3.5,  1.0,  9.81, 4.0),
        "walking": (1.2,  0.3,  9.81, 1.8),
        "anomaly": (6.0,  2.0,  9.81, 9.0),
    }.get(label, (0.5, 0.2, 9.81, 1.0))

    amp_xy, amp_z, z_bias, motion_freq = profile
    dt = 1.0 / freq_hz
    out: List[float] = []
    for i in range(n_samples):
        t = i * dt
        phase = 2 * math.pi * motion_freq * t
        ax = round(amp_xy * math.sin(phase), 4)
        ay = round(amp_xy * math.cos(phase + 0.5), 4)
        az = round(z_bias + amp_z * math.sin(phase * 2 + 1.2), 4)
        out.extend([ax, ay, az])
    return out


def _accel_values(label: str, n_samples: int, freq_hz: float) -> List[float]:
    """Try real hardware first; fall back to simulation."""
    real = _read_imu_values(n_samples, freq_hz)
    return real if real is not None else _sine_accel_values(label, n_samples, freq_hz)


def _build_sample_json(label: str, length_ms: int, freq_hz: float) -> bytes:
    """Return Edge-Impulse–compatible JSON payload bytes."""
    n_samples = max(1, int(length_ms / 1000 * freq_hz))
    values = _accel_values(label, n_samples, freq_hz)
    doc = {
        "payload": {
            "device_type": PETAL_DEVICE_TYPE,
            "interval_ms": round(1000.0 / freq_hz, 4),
            "sensors": _SAMPLE_AXES,
            "values": values,
        }
    }
    return json.dumps(doc).encode()


def _hmac_sign(key_hex: str, body: bytes) -> str:
    return _hmac_mod.new(bytes.fromhex(key_hex), body, hashlib.sha256).hexdigest()


def _split_labels(raw: str) -> List[str]:
    labels = [part.strip() for part in raw.split(",") if part.strip()]
    return labels or [raw.strip() or "idle"]


def _infer_result_local(label_hint: str, tick: int, labels: Optional[List[str]] = None) -> dict:
    """
    Local fallback when no PETAL_MODEL_ID is configured.
    Returns deterministic scores — not real model output.
    """
    labels = list(labels or [])
    if not labels:
        labels = _INFER_CLASSES if label_hint in _INFER_CLASSES else [label_hint]
    if label_hint not in labels:
        labels.insert(0, label_hint)
    primary_idx = labels.index(label_hint) if label_hint in labels else (tick // 10 % len(labels))
    primary_conf = 0.70 + 0.25 * abs(math.sin(tick * 0.3))
    remaining = 1.0 - primary_conf
    scores: dict = {}
    for i, lbl in enumerate(labels):
        if i == primary_idx:
            scores[lbl] = round(primary_conf, 4)
        else:
            denom = max(1, len(labels) - 1)
            scores[lbl] = round(remaining / denom, 4)
    return {"label": labels[primary_idx], "confidence": round(primary_conf, 4), "scores": scores}



_PE_MAGIC = b"PEM1"
_PE_HEADER = struct.Struct("<4sIIIIII")  # magic fmt model dsp labels manifest inference


def _unpack_pe(pe_bytes: bytes, dest_dir: str) -> None:
    """Extract a PEM1 container into dest_dir (model.tflite, manifest.json, etc.)."""
    hdr_sz = _PE_HEADER.size
    magic, _fmt, model_sz, dsp_sz, labels_sz, manifest_sz, inference_sz = _PE_HEADER.unpack_from(pe_bytes, 0)
    if magic != _PE_MAGIC:
        raise ValueError(f"bad PE magic {magic!r}")
    off = hdr_sz
    parts = [
        ("model.tflite",    model_sz),
        ("dsp_config.json", dsp_sz),
        ("labels.txt",      labels_sz),
        ("manifest.json",   manifest_sz),
        ("inference.py",    inference_sz),
    ]
    dest = pathlib.Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)
    for name, sz in parts:
        (dest / name).write_bytes(pe_bytes[off:off + sz])
        off += sz


# Local package inference is handled by PetalDeviceClient._load_pe_package()
# and delegated to the unpacked package's inference.py via run_inference(features).
# This avoids reimplementing model-type-specific decoding (classification, FOMO,
# SSD, YOLO-Pro) and quantization handling that inference.py already contains.


def _pick_label(tick: int) -> str:
    if PETAL_LABEL_MODE == "fixed":
        return PETAL_FIXED_LABEL
    if PETAL_LABEL_MODE == "random":
        return _SAMPLE_LABELS[tick % len(_SAMPLE_LABELS)]
    # cycle
    return _SAMPLE_LABELS[tick % len(_SAMPLE_LABELS)]


def _capture_picamera2_jpeg(width: int = 96, height: int = 96) -> Optional[bytes]:
    """Capture a JPEG from the Pi camera via picamera2. Returns None if unavailable."""
    try:
        from picamera2 import Picamera2  # type: ignore
        cam = Picamera2()
        cfg = cam.create_still_configuration(main={"size": (width, height), "format": "RGB888"})
        cam.configure(cfg)
        cam.start()
        import time as _time; _time.sleep(0.1)
        frame = cam.capture_array()
        cam.stop()
        cam.close()
        from PIL import Image
        buf = io.BytesIO()
        Image.fromarray(frame).save(buf, format="JPEG", quality=80)
        return buf.getvalue()
    except Exception:
        return None


def _collect_diagnostics() -> Dict[str, Any]:
    """Best-effort runtime health snapshot for the heartbeat.

    Returns a dict with any of:
      cpu_percent, mem_percent, mem_total_mb, disk_percent, temp_c, uptime_s,
      sensors: {imu: bool, camera: bool}
    Every field is optional — anything that can't be read is simply omitted.
    Prefers psutil when importable; otherwise falls back to /proc and shutil.
    Never raises.
    """
    diag: Dict[str, Any] = {}

    try:
        import psutil  # type: ignore
        try:
            diag["cpu_percent"] = round(float(psutil.cpu_percent(interval=0.1)), 1)
        except Exception:
            pass
        try:
            vm = psutil.virtual_memory()
            diag["mem_percent"]  = round(float(vm.percent), 1)
            diag["mem_total_mb"] = round(vm.total / (1024 * 1024), 1)
        except Exception:
            pass
        try:
            diag["disk_percent"] = round(float(psutil.disk_usage("/").percent), 1)
        except Exception:
            pass
        try:
            bt = psutil.boot_time()
            if bt:
                diag["uptime_s"] = int(time.time() - bt)
        except Exception:
            pass
    except Exception:
        # ── psutil unavailable — Linux /proc + shutil fallbacks ───────────────
        # CPU: two /proc/stat samples over a short interval.
        try:
            def _cpu_totals() -> tuple[int, int]:
                with open("/proc/stat", "r") as fh:
                    parts = fh.readline().split()
                vals = [int(x) for x in parts[1:]]
                idle = vals[3] + (vals[4] if len(vals) > 4 else 0)  # idle + iowait
                return sum(vals), idle
            t0, i0 = _cpu_totals()
            time.sleep(0.1)
            t1, i1 = _cpu_totals()
            dt, di = t1 - t0, i1 - i0
            if dt > 0:
                diag["cpu_percent"] = round(100.0 * (dt - di) / dt, 1)
        except Exception:
            pass
        # Memory: /proc/meminfo (kB values).
        try:
            info: Dict[str, int] = {}
            with open("/proc/meminfo", "r") as fh:
                for line in fh:
                    k, _, rest = line.partition(":")
                    info[k.strip()] = int(rest.strip().split()[0])
            total = info.get("MemTotal")
            avail = info.get("MemAvailable", info.get("MemFree"))
            if total:
                diag["mem_total_mb"] = round(total / 1024.0, 1)
                if avail is not None:
                    diag["mem_percent"] = round(100.0 * (total - avail) / total, 1)
        except Exception:
            pass
        # Disk: shutil.disk_usage on root.
        try:
            import shutil
            du = shutil.disk_usage("/")
            if du.total > 0:
                diag["disk_percent"] = round(100.0 * du.used / du.total, 1)
        except Exception:
            pass
        # Uptime: /proc/uptime (seconds since boot).
        try:
            with open("/proc/uptime", "r") as fh:
                diag["uptime_s"] = int(float(fh.readline().split()[0]))
        except Exception:
            pass

    # Temperature: Pi thermal zone, then vcgencmd fallback.
    try:
        with open("/sys/class/thermal/thermal_zone0/temp", "r") as fh:
            diag["temp_c"] = round(int(fh.read().strip()) / 1000.0, 1)
    except Exception:
        try:
            import subprocess, re
            out = subprocess.check_output(["vcgencmd", "measure_temp"], timeout=2).decode()
            m = re.search(r"([\d.]+)", out)
            if m:
                diag["temp_c"] = round(float(m.group(1)), 1)
        except Exception:
            pass

    # Sensor presence — reuse the real hardware probes; never let them raise.
    sensors: Dict[str, bool] = {}
    try:
        sensors["imu"] = _read_imu_values(1, 100.0) is not None
    except Exception:
        pass
    try:
        sensors["camera"] = _capture_picamera2_jpeg() is not None
    except Exception:
        pass
    if sensors:
        diag["sensors"] = sensors

    return diag


def _make_snapshot_jpeg() -> bytes:
    """Return a JPEG frame — real Pi camera if available, else synthetic test pattern."""
    real = _capture_picamera2_jpeg()
    if real is not None:
        return real
    try:
        from PIL import Image, ImageDraw
        img = Image.new("RGB", (96, 96), color=(30, 30, 30))
        draw = ImageDraw.Draw(img)
        for i in range(0, 96, 8):
            draw.line([(i, 0), (96, 96 - i)], fill=(80 + i * 2, 120, 200 - i), width=1)
        ts = int(time.time()) % 1000
        draw.rectangle([(2, 2), (2 + (ts % 90), 6)], fill=(255, 100, 0))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=60)
        return buf.getvalue()
    except ImportError:
        return (
            b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00"
            b"\xff\xdb\x00C\x00\x08\x06\x06\x07\x06\x05\x08\x07\x07\x07\t\t"
            b"\x08\n\x0c\x14\r\x0c\x0b\x0b\x0c\x19\x12\x13\x0f\x14\x1d\x1a"
            b"\x1f\x1e\x1d\x1a\x1c\x1c $.' \",#\x1c\x1c(7),01444\x1f'9=82<.342\x1c\x1c"
            b"\xff\xc0\x00\x0b\x08\x00\x01\x00\x01\x01\x01\x11\x00\xff\xc4\x00"
            b"\x1f\x00\x00\x01\x05\x01\x01\x01\x01\x01\x01\x00\x00\x00\x00\x00"
            b"\x00\x00\x00\x01\x02\x03\x04\x05\x06\x07\x08\t\n\x0b\xff\xc4\x00"
            b"\xb5\x10\x00\x02\x01\x03\x03\x02\x04\x03\x05\x05\x04\x04\x00\x00"
            b"\x01}\x01\x02\x03\x00\x04\x11\x05\x12!1A\x06\x13Qa\x07\"q\x142\x81"
            b"\x91\xa1\x08#B\xb1\xc1\x15R\xd1\xf0$3br\x82\t\n\x16\x17\x18\x19"
            b"\x1a%&'()*456789:CDEFGHIJSTUVWXYZ"
            b"\xff\xda\x00\x08\x01\x01\x00\x00?\x00\xfb\xd3\xff\xd9"
        )


def _load_snapshot_frame() -> tuple[bytes, int, int, str]:
    """
    Return JPEG bytes + dimensions for snapshot/image inference.

    When PETAL_IMAGE_PATH is set, that image is used for both snapshot
    streaming and image-model inference. Otherwise a synthetic test pattern
    is generated.
    """
    if PETAL_IMAGE_PATH:
        path = pathlib.Path(PETAL_IMAGE_PATH).expanduser()
        if path.is_file():
            try:
                from PIL import Image
                with Image.open(path) as img:
                    img = img.convert("RGB")
                    # Resize to model input size before encoding so the JPEG
                    # passed to DSPProcessor matches training dimensions exactly
                    # and avoids sending a large screenshot over the wire.
                    img = img.resize((96, 96), Image.BILINEAR)
                    width, height = img.size   # always (96, 96) after resize
                    buf = io.BytesIO()
                    img.save(buf, format="JPEG", quality=85)
                    return buf.getvalue(), width, height, str(path)
            except Exception as exc:
                logger.warning("[image] failed to load PETAL_IMAGE_PATH=%s: %s", path, exc)
        else:
            logger.warning("[image] PETAL_IMAGE_PATH does not exist: %s", path)
    return _make_snapshot_jpeg(), 96, 96, "synthetic-test-pattern"


# ── Main client ───────────────────────────────────────────────────────────────

class PetalDeviceClient:
    """
    Realistic software-device client for Petal Edge backend validation.

    Runs heartbeat + WebSocket concurrently.  All stub methods are replaced
    with deterministic simulation so the full platform pipeline can be
    validated without real hardware.
    """

    def __init__(
        self,
        host: str = PETAL_HOST,
        api_key: str = PETAL_API_KEY,
        device_id: str = PETAL_DEVICE_ID,
        firmware_version: str = PETAL_FIRMWARE,
        device_type: str = PETAL_DEVICE_TYPE,
        device_profile: str = PETAL_PROFILE,
        deployment_target: str = PETAL_TARGET,
        supports_snapshot_streaming: bool = PETAL_SNAPSHOT,
        user_email: str = PETAL_USER_EMAIL,
        user_pass: str = PETAL_USER_PASS,
        model_id: str = PETAL_MODEL_ID,
    ):
        self.host              = host.rstrip("/")
        self.api_key           = api_key
        self.device_id         = device_id
        self.firmware_version  = firmware_version
        self.device_type       = device_type
        self.device_profile    = device_profile
        self.deployment_target = deployment_target
        self.supports_snapshot = supports_snapshot_streaming
        self.user_email        = user_email
        self.user_pass         = user_pass
        self.model_id          = model_id

        self._device_pk: Optional[str]   = None
        self._project_id: Optional[str]  = None
        self._jwt: Optional[str]         = None
        self._session: Optional[aiohttp.ClientSession] = None
        self._running = False
        self._sample_tick = 0
        self._infer_tick  = 0
        self._latest_label = PETAL_FIXED_LABEL if PETAL_LABEL_MODE == "fixed" else _SAMPLE_LABELS[0]
        self._latest_infer_labels = _split_labels(self._latest_label)
        # Set after _fetch_model_meta(); drives real backend inference.
        self._model_feature_count: int = 0
        self._model_label_names: List[str] = []
        # DSP config read from model_metadata — required for real feature generation.
        self._model_dsp_blocks: List[dict] = []
        self._model_frequency_hz: float = _SAMPLE_FREQ_HZ
        self._model_window_size_ms: int = 2000
        # Throttle repeated DSP extraction errors so they don't flood the log.
        self._dsp_error_count: int = 0

        self._snapshot_active  = False
        self._inference_active = False
        self._inference_task: Optional[asyncio.Task] = None
        self._fomo_threshold: Optional[float] = None  # None = use model metadata threshold
        # Live stream config sent by the server — None means use model/package defaults.
        self._stream_sensor:           Optional[str]   = None
        self._stream_frequency_hz:     Optional[float] = None
        self._stream_sample_length_ms: Optional[int]   = None

        # Local package cache — invalidated after each OTA unpack.
        self._pe_pkg: Optional[types.ModuleType] = None
        self._pe_pkg_mtime: float = 0.0
        self._pe_interp: Any = None  # TFLite interpreter; lazy-loaded after module is ready

        pathlib.Path(PETAL_MARKER_DIR).mkdir(parents=True, exist_ok=True)
        pathlib.Path(PETAL_PE_DIR).mkdir(parents=True, exist_ok=True)

    # ── entry point ───────────────────────────────────────────────────────────

    async def run(self) -> None:
        self._running = True
        async with aiohttp.ClientSession() as session:
            self._session = session
            if PETAL_FORCE_LOCAL_PACKAGE:
                # Load DSP metadata from installed package immediately so inference
                # is ready as soon as the first start-inference-stream command arrives.
                # Login is not required; credentials are still used for profile PATCH
                # if they happen to be set.
                self._load_package_meta()
            if self.user_email and self.user_pass:
                await self._login()
            await asyncio.gather(
                self._heartbeat_loop(),
                self._ws_loop(),
            )

    def stop(self) -> None:
        self._running = False

    # ── auth (optional, needed for device profile PATCH) ─────────────────────

    async def _login(self) -> None:
        url = f"{self.host}/api/v1/auth/login"
        try:
            async with self._session.post(url, json={"email": self.user_email, "password": self.user_pass}) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    self._jwt = data.get("access_token")
                    logger.info("[auth] logged in as %s", self.user_email)
                else:
                    logger.warning("[auth] login failed HTTP %s", resp.status)
        except Exception as exc:
            logger.warning("[auth] login error: %s", exc)

    async def _patch_device_profile(self) -> None:
        """
        Set deployment_target + device_profile on the backend device record.
        Requires a user JWT (set PETAL_USER_EMAIL + PETAL_USER_PASS).
        The backend compatibility resolver uses these to match OTA deployments.
        """
        if not self._jwt or not self._device_pk:
            return
        if not self.deployment_target and not self.device_profile:
            return

        url  = f"{self.host}/api/v1/devices/{self._device_pk}"
        body = {}
        if self.device_profile:
            body["device_profile"] = self.device_profile
        if self.deployment_target:
            body["deployment_target"] = self.deployment_target
        if self.device_type:
            body["device_type"] = self.device_type

        headers = {"Authorization": f"Bearer {self._jwt}"}
        try:
            async with self._session.patch(url, json=body, headers=headers) as resp:
                if resp.status == 200:
                    logger.info(
                        "[profile] device_profile=%r deployment_target=%r applied",
                        self.device_profile, self.deployment_target,
                    )
                else:
                    text = await resp.text()
                    logger.warning("[profile] PATCH failed HTTP %s: %s", resp.status, text[:120])
        except Exception as exc:
            logger.warning("[profile] PATCH error: %s", exc)

    # ── heartbeat ─────────────────────────────────────────────────────────────

    async def _heartbeat_loop(self) -> None:
        while self._running:
            if self._device_pk:
                await self._send_heartbeat()
            await asyncio.sleep(HEARTBEAT_INTERVAL)

    async def _send_heartbeat(self) -> bool:
        url  = f"{self.host}/api/v1/devices/{self._device_pk}/heartbeat"
        body: dict = {
            "firmware_version": self.firmware_version,
            "ip_address": _local_ip(),
        }
        if self.supports_snapshot:
            body["supports_snapshot_streaming"] = True

        diagnostics = _collect_diagnostics()
        if diagnostics:
            body["diagnostics"] = diagnostics

        for attempt, delay in enumerate((*HEARTBEAT_RETRY_DELAYS, None)):
            try:
                async with self._session.post(
                    url, json=body,
                    timeout=aiohttp.ClientTimeout(total=10),
                ) as resp:
                    if resp.status == 200:
                        logger.debug("[heartbeat] ok last_seen=%s", (await resp.json()).get("last_seen",""))
                        return True
                    logger.warning("[heartbeat] HTTP %s", resp.status)
            except Exception as exc:
                logger.warning("[heartbeat] attempt=%d error: %s", attempt + 1, exc)
            if delay is None:
                break
            await asyncio.sleep(delay)
        logger.error("[heartbeat] all retries failed")
        return False

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
                self._snapshot_active  = False
                self._inference_active = False
            await asyncio.sleep(WS_RECONNECT_DELAY)

    async def _handshake(self, ws) -> bool:
        hello = {
            "type":                     "hello",
            "version":                  "1",
            "apiKey":                   self.api_key,
            "deviceId":                 self.device_id,
            "deviceType":               self.device_type,
            "connection":               "wifi",
            "firmwareVersion":          self.firmware_version,
            "protocolVersion":          "2",
            "supportsSnapshotStreaming": self.supports_snapshot,
            "sensors":                  _SAMPLE_AXES,
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

        self._device_pk = msg.get("id")
        self._project_id = msg.get("project_id")
        logger.info("[ws] connected device_pk=%s", self._device_pk)
        return True

    async def _fetch_model_meta(self) -> None:
        """
        GET /api/v1/trained-models/{model_id} to read input_shape, label_names,
        dsp_blocks, frequency_hz, and window_size_ms from model_metadata.
        All five fields are required to run real backend inference.
        """
        if not self._jwt:
            logger.warning(
                "[infer] no JWT available — set PETAL_USER_EMAIL + PETAL_USER_PASS so "
                "the client can log in. Model metadata will not be fetched; falling "
                "back to local simulation."
            )
            return

        requested_model_id = self.model_id
        url: Optional[str] = None
        fallback_reason: Optional[str] = None
        if requested_model_id:
            url = f"{self.host}/api/v1/trained-models/{requested_model_id}"
        elif self._project_id:
            url = f"{self.host}/api/v1/trained-models/project/{self._project_id}/latest?format=tflite"
            fallback_reason = "PETAL_MODEL_ID not set"
        else:
            return

        headers = {"Authorization": f"Bearer {self._jwt}"}
        try:
            async with self._session.get(url, headers=headers, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                if (
                    resp.status == 404
                    and requested_model_id
                    and self._project_id
                ):
                    fallback_reason = f"requested model {requested_model_id} was not found"
                    fallback_url = (
                        f"{self.host}/api/v1/trained-models/project/{self._project_id}/latest?format=tflite"
                    )
                    logger.warning(
                        "[infer] model %s was not found; resolving latest project model for project=%s",
                        requested_model_id,
                        self._project_id,
                    )
                    async with self._session.get(
                        fallback_url,
                        headers=headers,
                        timeout=aiohttp.ClientTimeout(total=10),
                    ) as fallback_resp:
                        if fallback_resp.status != 200:
                            logger.warning(
                                "[infer] project latest-model fetch HTTP %s — backend inference disabled",
                                fallback_resp.status,
                            )
                            return
                        data = await fallback_resp.json()
                elif resp.status != 200:
                    logger.warning("[infer] model fetch HTTP %s — backend inference disabled", resp.status)
                    return
                else:
                    data = await resp.json()

                resolved_model_id = data.get("id") or requested_model_id or self.model_id
                if resolved_model_id:
                    self.model_id = resolved_model_id

                shape = data.get("input_shape") or []
                n = 1
                for dim in shape:
                    if isinstance(dim, int) and dim > 0:
                        n *= dim
                self._model_feature_count = n
                self._model_label_names   = data.get("label_names") or []

                # DSP config — lives inside model_metadata (written by training worker)
                meta = data.get("model_metadata") or {}

                dsp_blocks = meta.get("dsp_blocks") or []
                if not dsp_blocks:
                    logger.warning(
                        "[infer] model_metadata.dsp_blocks is absent — real DSP inference not possible. "
                        "Retrain the model so dsp_blocks is embedded in metadata. "
                        "Will fall back to local simulation."
                    )
                self._model_dsp_blocks = dsp_blocks

                # frequency_hz and window_size_ms are stored by training_worker._common_meta.
                # Warn loudly when absent so stale-metadata issues surface immediately.
                raw_freq = meta.get("frequency_hz")
                raw_win  = meta.get("window_size_ms")
                if raw_freq is None:
                    logger.warning(
                        "[infer] frequency_hz missing from model_metadata — assuming %.1f Hz. "
                        "Retrain model to embed this. Feature count may mismatch.",
                        _SAMPLE_FREQ_HZ,
                    )
                if raw_win is None:
                    logger.warning(
                        "[infer] window_size_ms missing from model_metadata — assuming 2000 ms. "
                        "Retrain model to embed this. Feature count may mismatch.",
                    )
                self._model_frequency_hz   = float(raw_freq) if raw_freq is not None else _SAMPLE_FREQ_HZ
                self._model_window_size_ms = int(raw_win)    if raw_win  is not None else 2000

                logger.info(
                    "[infer] model loaded id=%s feature_count=%d labels=%s "
                    "dsp_blocks=%s freq=%.1fHz window=%dms",
                    self.model_id, n, self._model_label_names,
                    [b.get("type") for b in dsp_blocks],
                    self._model_frequency_hz, self._model_window_size_ms,
                )
                if fallback_reason:
                    logger.info("[infer] using resolved project model id=%s (%s)", self.model_id, fallback_reason)

                # Push resolved model ID into device_metadata so the Devices UI can display it.
                if self._device_pk and self._jwt and self.model_id:
                    asyncio.create_task(self._patch_current_model(self.model_id))
        except Exception as exc:
            logger.warning("[infer] model meta fetch error: %s — backend inference disabled", exc)

    async def _patch_current_model(self, model_id: str) -> None:
        if not self._device_pk or not self._jwt:
            return
        url = f"{self.host}/api/v1/devices/{self._device_pk}"
        headers = {"Authorization": f"Bearer {self._jwt}"}
        try:
            async with self._session.patch(
                url,
                json={"device_metadata": {"current_model_id": model_id}},
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=5),
            ) as resp:
                if resp.status != 200:
                    logger.warning("[infer] failed to update current_model_id HTTP %s", resp.status)
        except Exception as exc:
            logger.warning("[infer] patch current_model_id error: %s", exc)

    def _load_package_meta(self) -> bool:
        """Populate DSP / model fields from the installed .pe package files.

        Reads PETAL_PE_DIR/dsp_config.json and PETAL_PE_DIR/manifest.json and
        mirrors the same instance fields that _fetch_model_meta() populates from
        the backend API, so _build_dsp_features() and _run_pe_inference() work
        without any network calls or JWT.

        frequency_hz / window_size_ms are not stored in the package files; override
        with PETAL_FREQ_HZ / PETAL_WINDOW_MS env vars, otherwise defaults are used.

        Returns True when at least dsp_blocks were loaded; False on failure / absent
        package (caller should log a warning and fall back to simulation).
        """
        pe_dir        = pathlib.Path(PETAL_PE_DIR)
        dsp_path      = pe_dir / "dsp_config.json"
        manifest_path = pe_dir / "manifest.json"

        if not dsp_path.exists() and not manifest_path.exists():
            logger.warning(
                "[local_pkg] no package found at %s — "
                "PETAL_FORCE_LOCAL_PACKAGE=1 requires an installed .pe package; "
                "falling back to simulation until a package is delivered via OTA",
                PETAL_PE_DIR,
            )
            return False

        try:
            if dsp_path.exists():
                dsp = json.loads(dsp_path.read_text())
                self._model_dsp_blocks = dsp.get("dsp_blocks") or []

            if manifest_path.exists():
                manifest = json.loads(manifest_path.read_text())
                self._model_label_names = manifest.get("label_names") or []
                shape = manifest.get("input_shape") or []
                n = 1
                for dim in shape:
                    if isinstance(dim, int) and dim > 0:
                        n *= dim
                if n > 1:
                    self._model_feature_count = n

            # frequency_hz / window_size_ms are not written into package files;
            # accept env-var overrides, then keep whatever the current default is.
            raw_freq = os.environ.get("PETAL_FREQ_HZ")
            raw_win  = os.environ.get("PETAL_WINDOW_MS")
            if raw_freq:
                self._model_frequency_hz = float(raw_freq)
            if raw_win:
                self._model_window_size_ms = int(raw_win)

            logger.info(
                "[local_pkg] metadata loaded from %s — "
                "dsp_blocks=%s  labels=%s  feature_count=%d  freq=%.1fHz  window=%dms",
                PETAL_PE_DIR,
                [b.get("type") for b in self._model_dsp_blocks],
                self._model_label_names,
                self._model_feature_count,
                self._model_frequency_hz,
                self._model_window_size_ms,
            )
            return bool(self._model_dsp_blocks)

        except Exception as exc:
            logger.warning("[local_pkg] failed to load package metadata: %s — sim fallback", exc)
            return False

    async def _post_connect(self, ws) -> None:
        """Run after successful hello-ack."""
        await self._patch_device_profile()
        if PETAL_FORCE_LOCAL_PACKAGE:
            self._load_package_meta()
        else:
            await self._fetch_model_meta()

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

    # ── dispatch ──────────────────────────────────────────────────────────────

    async def _dispatch(self, ws, msg: dict) -> None:
        t   = msg.get("type", "")
        cid = msg.get("correlationId")
        pl  = msg.get("payload", {})

        if t == "ping":
            await ws.send_json({"type": "pong"})

        elif t == "start-sample":
            asyncio.create_task(self._sample_flow(ws, pl, cid))

        elif t == "start-snapshot":
            self._snapshot_active = True
            asyncio.create_task(self._snapshot_stream_task(ws, cid))
            logger.info("[snapshot] started cid=%s", cid)

        elif t == "stop-snapshot":
            self._snapshot_active = False
            logger.info("[snapshot] stopped by server")

        elif t == "start-inference-stream":
            # Cancel any existing task before spawning a new one.
            if self._inference_task and not self._inference_task.done():
                self._inference_active = False   # signal old task to exit
                self._inference_task.cancel()
            self._fomo_threshold = pl.get("fomo_threshold")  # None = use model metadata
            # Parse optional stream config — absent or invalid fields keep None (model default).
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
            self._inference_task = asyncio.create_task(self._inference_stream_task(ws, cid))
            logger.info(
                "[inference] stream started cid=%s  threshold=%s  sensor=%s  freq=%s  window=%s",
                cid,
                self._fomo_threshold if self._fomo_threshold is not None else "model-default",
                self._stream_sensor           or "model-default",
                f"{self._stream_frequency_hz:.1f}Hz" if self._stream_frequency_hz is not None
                    else "model-default",
                f"{self._stream_sample_length_ms}ms" if self._stream_sample_length_ms is not None
                    else "model-default",
            )

        elif t == "stop-inference-stream":
            self._inference_active = False
            self._inference_task = None
            logger.info("[inference] stream stopped by server")

        elif t == "model-update":
            asyncio.create_task(self._ota_flow(ws, pl, cid))

        else:
            logger.debug("[ws] unhandled type=%r", t)

    # ── sampling ──────────────────────────────────────────────────────────────

    async def _sample_flow(self, ws, payload: dict, cid: Optional[str]) -> None:
        label      = payload.get("label") or _pick_label(self._sample_tick)
        length_ms  = int(payload.get("length") or 5000)
        freq_hz    = float(payload.get("frequency") or _SAMPLE_FREQ_HZ)
        hmac_key   = payload.get("hmacKey")
        sample_tok = payload.get("sampleToken")
        path       = payload.get("path")
        self._sample_tick += 1
        self._latest_infer_labels = _split_labels(label)
        self._latest_label = self._latest_infer_labels[0]

        logger.info("[sample] start label=%r length=%dms freq=%.1fHz", label, length_ms, freq_hz)

        await ws.send_json({"type": "sample-ack",     "success": True,  "correlationId": cid, "payload": {}})
        await ws.send_json({"type": "sample-started", "label": label,   "length": length_ms,  "correlationId": cid})

        try:
            body = _build_sample_json(label, length_ms, freq_hz)
            # simulate acquisition time (capped at 5 s for quick tests)
            await asyncio.sleep(min(length_ms / 1000, 5))
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
        headers: Dict[str, str] = {"Content-Type": "application/json"}
        # The current backend always resolves project context from x-api-key.
        # Per-sample auth headers add HMAC verification, but do not replace it.
        headers["x-api-key"] = self.api_key

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
            tag = "dedup" if data.get("duplicate") else "new"
            logger.info(
                "[sample] uploaded(%s)  id=%s  filename=%s  hash=%s",
                tag,
                data.get("id", "?"),
                data.get("filename", "?"),
                (data.get("payload_hash") or "")[:12],
            )

    # ── snapshot ──────────────────────────────────────────────────────────────

    async def _snapshot_stream_task(self, ws, cid: Optional[str]) -> None:
        logger.info("[snapshot] stream task running")
        try:
            while self._snapshot_active and not ws.closed:
                frame, width, height, _source = _load_snapshot_frame()
                await ws.send_json({
                    "type":          "snapshot",
                    "width":         width,
                    "height":        height,
                    "format":        "jpeg",
                    "correlationId": cid,
                })
                # Phase 3: 4-byte LE length + frame bytes
                await ws.send_bytes(struct.pack("<I", len(frame)) + frame)
                await asyncio.sleep(0.1)   # ~10 fps
        except Exception as exc:
            logger.error("[snapshot] stream error: %s", exc)
            self._snapshot_active = False
        logger.info("[snapshot] stream task ended")

    # ── inference mode resolution ─────────────────────────────────────────────

    def _resolve_inference_mode(self) -> tuple:
        """
        Evaluate every guard for backend inference mode in priority order.
        Returns (use_backend: bool, reason: str).
        reason is "" when use_backend is True; a machine-readable slug + human
        note when False, e.g. "no_model_id — set PETAL_MODEL_ID=<uuid>".
        """
        if PETAL_FORCE_LOCAL_PACKAGE:
            return False, "force_local_package — PETAL_FORCE_LOCAL_PACKAGE=1"
        if not self.model_id:
            if self._project_id and self._jwt:
                return False, "no_resolved_model_yet — waiting for project-scoped model lookup"
            return False, "no_model_id — set PETAL_MODEL_ID=<trained-model-uuid>"
        if not self._jwt:
            return False, (
                "no_jwt — login failed or PETAL_USER_EMAIL/PETAL_USER_PASS not set; "
                "check [auth] log lines above"
            )
        if not _DSP_AVAILABLE:
            detail = f" ({_DSP_IMPORT_ERROR})" if _DSP_IMPORT_ERROR else ""
            return False, f"dsp_import_failed — install numpy and scipy{detail}"
        if not self._model_dsp_blocks:
            return False, (
                "no_dsp_blocks — model_metadata.dsp_blocks is absent; "
                "retrain the model so DSP config is embedded in its metadata"
            )
        if not self._model_feature_count:
            return False, (
                "no_feature_count — model metadata fetch failed or input_shape is missing; "
                "check [infer] log lines above"
            )
        return True, ""

    # ── inference stream ──────────────────────────────────────────────────────

    async def _inference_stream_task(self, ws, cid: Optional[str]) -> None:
        use_backend, fallback_reason = self._resolve_inference_mode()

        if use_backend:
            _btypes     = [b.get("type") for b in self._model_dsp_blocks]
            _input_mode = "image" if any(t == "image" for t in _btypes) else "sensor"
            logger.info(
                "[inference] mode=backend  model_id=%s  features=%d  "
                "dsp=%s  input=%s  freq=%.1fHz  window=%dms",
                self.model_id, self._model_feature_count,
                _btypes, _input_mode,
                self._model_frequency_hz, self._model_window_size_ms,
            )
        else:
            _initial_pkg = (pathlib.Path(PETAL_PE_DIR) / "inference.py").exists()
            logger.info(
                "[inference] mode=%s  reason=%s",
                "local-package" if _initial_pkg else "local-sim",
                fallback_reason or "installed .pe package found" if _initial_pkg else fallback_reason,
            )

        try:
            while self._inference_active and not ws.closed:
                if use_backend:
                    result = await self._call_predict()
                else:
                    label = self._latest_label or _pick_label(self._infer_tick)
                    result = await asyncio.get_event_loop().run_in_executor(
                        None, self._run_local_package_tick
                    )
                    if result is None:
                        result = _infer_result_local(label, self._infer_tick, labels=self._latest_infer_labels)

                if result is not None:
                    # Log detection summary every tick so flow is observable.
                    dbg = result.get("debug") or {}
                    if result.get("is_fomo"):
                        dets = result.get("detections") or []
                        top  = ", ".join(
                            f"{d['label']} ({d['confidence']*100:.0f}%)"
                            for d in dets[:2]
                        ) if dets else "—"
                        thr = dbg.get("threshold")
                        if thr is None:
                            thr = self._fomo_threshold
                        if thr is None:
                            thr = 0.5
                        obj_max = dbg.get("object_max")
                        if obj_max is None:
                            obj_max = max(
                                (d.get("confidence", 0.0) for d in dets), default=0.0
                            )
                        logger.info(
                            "[infer] tick=%d  count=%d  top=[%s]  obj_max=%.3f  thr=%.2f",
                            self._infer_tick, result.get("count", 0), top,
                            obj_max,
                            thr,
                        )
                    else:
                        logger.info(
                            "[infer] tick=%d  label=%s  conf=%.0f%%  raw_max=%.3f",
                            self._infer_tick,
                            result.get("label", "?"),
                            result.get("confidence", 0) * 100,
                            dbg.get("raw_max", float("nan")),
                        )
                    _payload = {k: v for k, v in result.items() if k != "debug"}
                    _scfg: dict = {}
                    if self._stream_sensor is not None:
                        _scfg["sensor"] = self._stream_sensor
                    if self._stream_frequency_hz is not None:
                        _scfg["frequency_hz"] = self._stream_frequency_hz
                    if self._stream_sample_length_ms is not None:
                        _scfg["sample_length_ms"] = self._stream_sample_length_ms
                    if _scfg:
                        _payload["stream_config"] = _scfg
                    await ws.send_json({
                        "type":          "inference-result",
                        "correlationId": cid,
                        "payload":       _payload,
                    })
                self._infer_tick += 1
                await asyncio.sleep(0.2)   # ~5 fps
        except Exception as exc:
            logger.error("[inference] stream error: %s", exc)
            self._inference_active = False
        logger.info("[inference] stream task ended")

    def _run_local_package_tick(self) -> Optional[dict]:
        """One local-package inference tick (called via run_in_executor).

        Mirrors unoq/runtime.py _run_inference_tick():
          1. load (or return cached) inference module via _load_pe_package
          2. select input path: image tensor or DSP feature vector
          3. delegate to package run_inference via _run_pe_inference
          4. return result dict, or None so the caller falls back to simulation
        """
        pkg = self._load_pe_package()
        if pkg is None or not self._model_dsp_blocks:
            return None
        try:
            fv = self._build_image_tensor() if self._is_image_model() else self._build_dsp_features()
            if fv is None:
                return None
            return self._run_pe_inference(fv)
        except Exception as exc:
            logger.warning("[inference] package inference error: %s — sim fallback", exc)
            return None

    def _build_dsp_features(self) -> Optional[List[float]]:
        """
        Generate a real DSP feature vector using the same pipeline as training.

        Input routing (Edge Impulse style):
          image blocks  → raw JPEG bytes from _make_snapshot_jpeg()
          all others    → sensor JSON inner payload (values=[ax,ay,az,...])

        Each DSP block receives the input type it was trained on.  Feeding
        accelerometer JSON into an image block (or vice versa) would silently
        produce wrong features; this method prevents that by routing per block.

        Returns None on any error; the caller skips that tick.
        """
        if not _DSP_AVAILABLE:
            self._log_dsp_error(
                "DSPProcessor not available — install numpy and scipy"
            )
            return None

        if not self._model_dsp_blocks:
            self._log_dsp_error(
                "model_metadata.dsp_blocks is empty — retrain model or check metadata"
            )
            return None

        # Stream command overrides model defaults; model defaults are the fallback.
        freq_hz   = self._stream_frequency_hz   if self._stream_frequency_hz   is not None else self._model_frequency_hz
        window_ms = self._stream_sample_length_ms if self._stream_sample_length_ms is not None else self._model_window_size_ms

        # ── Lazy-generate raw inputs — only what this model's blocks actually need ──
        # image blocks expect raw JPEG/PNG bytes; sensor blocks expect flat JSON values.
        sensor_json: Optional[bytes] = None
        image_bytes: Optional[bytes] = None

        all_features: List[List[float]] = []
        for block_cfg in self._model_dsp_blocks:
            block_type = block_cfg.get("type", "raw")
            params     = dict(block_cfg.get("params") or {})

            if block_type == "image":
                # DSPProcessor._image() calls PIL.Image.open(io.BytesIO(data))
                # so we must supply valid image bytes, not sensor JSON.
                if image_bytes is None:
                    image_bytes, _w, _h, _source = _load_snapshot_frame()
                raw_input: bytes = image_bytes
            else:
                # DSPProcessor.extract() JSON-decodes bytes and reads ["values"].
                # Build the inner payload format that ingestion stores (no wrapper).
                if sensor_json is None:
                    n_samples  = max(1, int(window_ms / 1000.0 * freq_hz))
                    label      = self._latest_label or _pick_label(self._infer_tick)
                    raw_values = _accel_values(label, n_samples, freq_hz)
                    sensor_json = json.dumps({
                        "device_type": PETAL_DEVICE_TYPE,
                        "interval_ms": round(1000.0 / freq_hz, 4),
                        "sensors":     _SAMPLE_AXES,
                        "values":      raw_values,
                    }).encode()
                raw_input = sensor_json

            try:
                proc = _DSPProcessor(
                    block_type=block_type, params=params, frequency_hz=freq_hz
                )
                feat = proc.extract(raw_input)
                all_features.append(feat.flatten().tolist())
                # Clear error counter on first success so transient errors
                # don't permanently suppress future error messages.
                self._dsp_error_count = 0
            except Exception as exc:
                self._log_dsp_error(
                    f"block {block_type!r} extract failed: {exc}"
                )
                return None

        if not all_features:
            self._log_dsp_error("no features produced from dsp_blocks")
            return None

        features: List[float] = _np.concatenate(
            [_np.array(f, dtype=_np.float32).flatten() for f in all_features]
        ).tolist()

        # Strict count check: mismatch means DSP config and model shape diverged.
        if self._model_feature_count and len(features) != self._model_feature_count:
            self._log_dsp_error(
                f"feature count mismatch: DSP produced {len(features)}, "
                f"model expects {self._model_feature_count} "
                f"(window_ms={window_ms} freq_hz={freq_hz:.1f} "
                f"dsp_blocks={[b.get('type') for b in self._model_dsp_blocks]}). "
                "Check that frequency_hz/window_size_ms in model_metadata match training."
            )
            return None

        return features

    # Log DSP errors at full verbosity for the first few occurrences, then
    # throttle to once every 50 ticks so a broken model doesn't flood the log.
    _DSP_ERROR_LOG_LIMIT  = 3
    _DSP_ERROR_LOG_PERIOD = 50

    def _log_dsp_error(self, msg: str) -> None:
        self._dsp_error_count += 1
        n = self._dsp_error_count
        if n <= self._DSP_ERROR_LOG_LIMIT or n % self._DSP_ERROR_LOG_PERIOD == 0:
            suffix = f"  (error #{n}; suppressing until #{n - n % self._DSP_ERROR_LOG_PERIOD + self._DSP_ERROR_LOG_PERIOD})" if n > self._DSP_ERROR_LOG_LIMIT else ""
            logger.error("[dsp] %s%s", msg, suffix)
        elif n == self._DSP_ERROR_LOG_LIMIT + 1:
            logger.warning(
                "[dsp] repeated DSP errors — suppressing until every %dth occurrence "
                "(total so far: %d)",
                self._DSP_ERROR_LOG_PERIOD, n,
            )

    def _is_image_model(self) -> bool:
        """True when the loaded package expects an image tensor (H×W×C) as input.

        Detection order matches unoq/runtime.py:
          1. dsp_blocks contains an 'image' block (fastest, already in memory).
          2. manifest.json input_shape has 3 dimensions (H, W, C).
        """
        if any(b.get("type") == "image" for b in self._model_dsp_blocks):
            return True
        manifest_path = pathlib.Path(PETAL_PE_DIR) / "manifest.json"
        if manifest_path.exists():
            try:
                shape = json.loads(manifest_path.read_text()).get("input_shape") or []
                return len(shape) >= 3
            except Exception:
                pass
        return False

    def _build_image_tensor(self):
        """Preprocess image input for one local-package inference tick.

        Mirrors unoq/runtime.py _features_from_image():
          • reads input dimensions from manifest.json (H, W, C)
          • captures a frame via _load_snapshot_frame() / PETAL_IMAGE_PATH
          • decodes + resizes with PIL (float32 array of shape (H, W, C))
          • falls back to raw-buffer reshape when PIL is unavailable

        Called via run_in_executor — must not touch the event loop.
        Returns None on any error; caller falls back to simulation.
        """
        if _np is None:
            logger.warning("[infer] numpy not available — image inference skipped")
            return None

        manifest_path = pathlib.Path(PETAL_PE_DIR) / "manifest.json"
        try:
            manifest = json.loads(manifest_path.read_text())
        except Exception as exc:
            logger.error("[infer] manifest read failed: %s", exc)
            return None

        input_shape = manifest.get("input_shape") or []
        if len(input_shape) < 3:
            logger.error(
                "[infer] input_shape %s is not an image shape (need H×W×C)", input_shape
            )
            return None

        h, w, c = int(input_shape[0]), int(input_shape[1]), int(input_shape[2])

        jpeg_bytes, _fw, _fh, _src = _load_snapshot_frame()

        try:
            from PIL import Image as _PILImage
            img = _PILImage.open(io.BytesIO(jpeg_bytes))
            img = img.resize((w, h), _PILImage.BILINEAR)
            img = img.convert("L" if c == 1 else "RGB")
            arr = _np.array(img, dtype=_np.float32)
            if arr.ndim == 2:
                arr = arr[:, :, _np.newaxis]
            return arr
        except ImportError:
            # PIL unavailable — reshape raw bytes as a float32 buffer (may be garbage
            # data for JPEG, but preserves the correct tensor shape for the model).
            try:
                arr = _np.frombuffer(jpeg_bytes, dtype=_np.uint8).astype(_np.float32)
                return arr[: h * w * c].reshape(h, w, c)
            except Exception:
                return None
        except Exception as exc:
            logger.error("[infer] image preprocessing failed: %s", exc)
            return None

    async def _call_predict(self) -> Optional[dict]:
        """
        POST /api/v1/inference/predict using real DSP-processed features.

        Features are built by _build_dsp_features() which runs the backend
        DSPProcessor on simulated accelerometer data with the exact same
        configuration used during training.  Returns None (tick skipped) if
        DSP parity cannot be ensured.
        """
        features = self._build_dsp_features()
        if features is None:
            return None   # error already logged in _build_dsp_features

        url     = f"{self.host}/api/v1/inference/predict"
        headers = {"Authorization": f"Bearer {self._jwt}"}
        body: dict = {"model_id": self.model_id, "features": features}
        if self._fomo_threshold is not None:
            body["fomo_threshold"] = self._fomo_threshold
        try:
            async with self._session.post(
                url, json=body, headers=headers,
                timeout=aiohttp.ClientTimeout(total=5),
            ) as resp:
                data = await resp.json()
                if resp.status != 200:
                    logger.warning("[infer] predict HTTP %s: %s", resp.status, str(data)[:120])
                    return None
                # Keep debug key for per-tick raw_max logging; strip before
                # forwarding to the Studio so the WS payload stays compact.
                return data
        except asyncio.TimeoutError:
            logger.warning("[infer] predict timeout (tick=%d)", self._infer_tick)
            return None
        except Exception as exc:
            logger.warning("[infer] predict error: %s", exc)
            return None

    # ── OTA update ────────────────────────────────────────────────────────────

    async def _ota_flow(self, ws, payload: dict, cid: Optional[str]) -> None:
        url           = payload.get("url", "")
        version       = payload.get("version", "unknown")
        deployment_id = payload.get("deployment_id", "")

        logger.info("[ota] update received version=%r deployment_id=%r", version, deployment_id)

        if not url:
            await ws.send_json({"type": "update-rejected", "error": "missing url", "correlationId": cid})
            logger.warning("[ota] rejected — no url")
            return

        await ws.send_json({"type": "update-accepted", "correlationId": cid})
        logger.info("[ota] accepted")

        try:
            await ws.send_json({"type": "update-download-started", "correlationId": cid})
            logger.info("[ota] downloading from %s", url)

            if PETAL_UPDATE_SIM:
                # simulated download — just HEAD the URL to confirm reachability
                async with self._session.head(url, timeout=aiohttp.ClientTimeout(total=15)) as resp:
                    pkg_size = int(resp.headers.get("Content-Length", 0))
                logger.info("[ota] package reachable size=%d bytes (sim, not written to disk)", pkg_size)
            else:
                # real download
                async with self._session.get(url, timeout=aiohttp.ClientTimeout(total=120)) as resp:
                    resp.raise_for_status()
                    pkg = await resp.read()
                    pkg_size = len(pkg)
                logger.info("[ota] downloaded %d bytes", pkg_size)

            await ws.send_json({"type": "update-install-started", "correlationId": cid})
            logger.info("[ota] installing version=%r", version)

            if not PETAL_UPDATE_SIM and pkg_size > 0:
                # unpack PEM1 container so on-device TFLite can use it immediately
                try:
                    _unpack_pe(pkg, PETAL_PE_DIR)
                    logger.info("[ota] unpacked .pe package to %s", PETAL_PE_DIR)
                    # Invalidate cached module + interpreter so the next tick uses the new package.
                    self._pe_pkg = None
                    self._pe_pkg_mtime = 0.0
                    self._pe_interp = None
                    logger.info("[ota] package cache cleared — next inference tick reloads inference.py")
                    # In force-local mode the DSP fields were not populated by the backend;
                    # refresh them from the newly installed package files now.
                    if PETAL_FORCE_LOCAL_PACKAGE:
                        self._load_package_meta()
                except Exception as unpack_exc:
                    logger.error("[ota] unpack failed: %s", unpack_exc)
                    raise
            else:
                await asyncio.sleep(1.5)  # sim delay only

            # persist marker so the installed state survives restarts
            self._write_installed_marker(version, deployment_id)
            self.firmware_version = version

            await ws.send_json({
                "type":          "update-install-succeeded",
                "version":       version,
                "correlationId": cid,
            })
            logger.info("[ota] done version=%r deployment_id=%r", version, deployment_id)

        except Exception as exc:
            logger.error("[ota] failed: %s", exc)
            await ws.send_json({
                "type":          "update-install-failed",
                "message":       str(exc),
                "correlationId": cid,
            })

    def _write_installed_marker(self, version: str, deployment_id: str) -> None:
        marker = pathlib.Path(PETAL_MARKER_DIR) / f"{self.device_id}_installed.json"
        try:
            marker.write_text(json.dumps({
                "device_id":     self.device_id,
                "version":       version,
                "deployment_id": deployment_id,
                "installed_at":  time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }, indent=2))
            logger.info("[ota] marker written to %s", marker)
        except Exception as exc:
            logger.warning("[ota] could not write marker: %s", exc)

    def _load_pe_package(self) -> Optional[types.ModuleType]:
        """Load (or return cached) the inference.py from the installed .pe package.

        Uses mtime to detect when OTA has written a new package; the cache is also
        explicitly cleared by _ota_flow so the reload happens on the very next tick.

        The loaded module must expose both:
            load_interpreter(model_path: str) -> interpreter
            run_inference(interpreter, features) -> dict
        This matches the API generated by deployment_worker._gen_raspberry_pi() and
        mirrors the unoq/runtime.py _load_inference_module() pattern.

        Called via run_in_executor — must not touch the event loop.
        """
        script = pathlib.Path(PETAL_PE_DIR) / "inference.py"
        if not script.exists():
            return None
        try:
            mtime = script.stat().st_mtime
        except OSError:
            return None

        if self._pe_pkg is not None and mtime == self._pe_pkg_mtime:
            return self._pe_pkg

        # Cache miss — import (or re-import) the module.
        try:
            spec = importlib.util.spec_from_file_location("pe_inference", str(script))
            if spec is None or spec.loader is None:
                logger.warning("[pe_pkg] importlib could not create spec for %s", script)
                return None
            mod = importlib.util.module_from_spec(spec)
            # Add pe_dir to sys.path so inference.py can open its sibling files
            # (manifest.json, model.tflite) with relative paths, matching unoq pattern.
            pe_dir_str = str(pathlib.Path(PETAL_PE_DIR))
            sys.path.insert(0, pe_dir_str)
            try:
                spec.loader.exec_module(mod)  # type: ignore[union-attr]
            finally:
                sys.path.remove(pe_dir_str)
            missing = [fn for fn in ("load_interpreter", "run_inference") if not hasattr(mod, fn)]
            if missing:
                logger.warning(
                    "[pe_pkg] inference.py at %s is missing %s — "
                    "cannot use package inference; falling back to simulation",
                    PETAL_PE_DIR, missing,
                )
                return None
            self._pe_pkg = mod
            self._pe_pkg_mtime = mtime
            logger.info("[pe_pkg] loaded inference.py  pe_dir=%s  mtime=%.0f", PETAL_PE_DIR, mtime)
            return mod
        except Exception as exc:
            logger.warning("[pe_pkg] failed to import inference.py: %s", exc)
            self._pe_pkg = None
            self._pe_pkg_mtime = 0.0
            return None

    def _run_pe_inference(self, fv: Any) -> Optional[dict]:
        """Lazy-load the TFLite interpreter then call package run_inference(interp, fv).

        Mirrors unoq/runtime.py _ensure_interpreter() + _run_inference_tick().
        Called via run_in_executor — must not touch the event loop.
        """
        if self._pe_pkg is None:
            return None

        # Lazy-load interpreter on first call (or after OTA reset).
        if self._pe_interp is None:
            model_path = pathlib.Path(PETAL_PE_DIR) / "model.tflite"
            if not model_path.exists():
                logger.error("[pe_pkg] model.tflite not found in %s", PETAL_PE_DIR)
                return None
            try:
                self._pe_interp = self._pe_pkg.load_interpreter(str(model_path))
                logger.info("[pe_pkg] interpreter loaded from %s", model_path)
            except Exception as exc:
                logger.error("[pe_pkg] load_interpreter failed: %s", exc)
                return None

        try:
            np = _np
            if np is None:
                import numpy as np  # type: ignore[no-redef]
            features_np = np.array(fv, dtype=np.float32)
            return self._pe_pkg.run_inference(self._pe_interp, features_np)
        except Exception as exc:
            logger.error("[pe_pkg] run_inference failed: %s", exc)
            return None


# ── helpers ───────────────────────────────────────────────────────────────────

def _local_ip() -> str:
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except Exception:
        return "0.0.0.0"


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG if os.environ.get("PETAL_DEBUG") else logging.INFO,
        format="%(asctime)s  %(levelname)-7s  %(message)s",
        datefmt="%H:%M:%S",
    )

    profile_info = f"  profile={PETAL_PROFILE!r} target={PETAL_TARGET!r}" if PETAL_PROFILE else ""
    logger.info(
        "Starting Petal device client  host=%s  device_id=%s  fw=%s%s",
        PETAL_HOST, PETAL_DEVICE_ID, PETAL_FIRMWARE, profile_info,
    )

    # ── Startup config summary ────────────────────────────────────────────────
    # Log every inference-mode guard upfront so the user can see exactly why
    # backend mode will or won't activate before the event loop starts.
    if _DSP_AVAILABLE:
        logger.info("[startup] dsp=ok  (DSPProcessor imported from backend)")
    else:
        logger.warning(
            "[startup] dsp=FAILED  DSPProcessor import failed — "
            "real backend inference will be disabled.  "
            "Fix: pip install numpy scipy  |  error: %s",
            _DSP_IMPORT_ERROR or "unknown ImportError",
        )

    if PETAL_MODEL_ID:
        logger.info("[startup] model_id=%s", PETAL_MODEL_ID)
        if PETAL_USER_EMAIL and PETAL_USER_PASS:
            logger.info("[startup] credentials=set  (will login as %s)", PETAL_USER_EMAIL)
        else:
            logger.warning(
                "[startup] credentials=MISSING  PETAL_MODEL_ID is set but "
                "PETAL_USER_EMAIL / PETAL_USER_PASS are not — login will be skipped, "
                "no JWT, backend inference disabled."
            )
    if PETAL_IMAGE_PATH:
        logger.info("[startup] image_path=%s  (used for snapshot stream and image-model inference)", PETAL_IMAGE_PATH)
    elif PETAL_MODEL_ID:
        logger.info(
            "[startup] image_path=not set - image models use a synthetic test pattern. "
            "Set PETAL_IMAGE_PATH=<jpg/png> to test cat/dog-style detection models."
        )

    client = PetalDeviceClient()
    try:
        asyncio.run(client.run())
    except KeyboardInterrupt:
        logger.info("stopped")
