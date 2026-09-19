"""Shared configuration — env vars and runtime constants."""
from __future__ import annotations

import os
import pathlib

PETAL_HOST       = os.environ.get("PETAL_HOST",       "http://192.168.1.22:8010").rstrip("/")
PETAL_API_KEY    = os.environ.get("PETAL_API_KEY",    "ef_changeme")
PETAL_DEVICE_ID  = os.environ.get("PETAL_DEVICE_ID",  "unoq-001")
PETAL_FIRMWARE   = os.environ.get("PETAL_FIRMWARE",   "1.0.0")
PETAL_USER_EMAIL = os.environ.get("PETAL_USER_EMAIL", "")
PETAL_USER_PASS  = os.environ.get("PETAL_USER_PASS",  "")
PETAL_PACKAGE_DIR = pathlib.Path(
    os.environ.get("PETAL_PACKAGE_DIR", pathlib.Path.home() / ".petal" / "packages")
)
UNOQ_TEST_IMAGE     = os.environ.get("UNOQ_TEST_IMAGE", "")
UNOQ_CAMERA_INDEX   = int(os.environ.get("UNOQ_CAMERA_INDEX", "0"))
UNOQ_SAMPLE_FREQ_HZ = float(os.environ.get("UNOQ_SAMPLE_FREQ_HZ", "62.5"))
UNOQ_SNAPSHOT       = os.environ.get("UNOQ_SNAPSHOT", "0") == "1"

HEARTBEAT_INTERVAL  = 20   # seconds
WS_RECONNECT_DELAY  = 5    # seconds
HELLO_TIMEOUT       = 10   # seconds
INFER_TICK_INTERVAL = 0.5  # seconds between inference ticks

# Active package pointer file — survives restarts
_ACTIVE_PKG_FILE = PETAL_PACKAGE_DIR / "active.json"
