"""
Petal Edge — UNO Q Runtime Daemon entry point.

Quick start:
    pip install -r requirements.txt
    PETAL_HOST=http://192.168.1.22:8010 \\
    PETAL_API_KEY=<project_device_key> \\
    PETAL_USER_EMAIL=you@example.com \\
    PETAL_USER_PASS=yourpassword \\
    python -m unoq.runtime.runtime

Env vars:
    PETAL_HOST          Backend URL (default: http://192.168.1.22:8010)
    PETAL_API_KEY       Project device API key (required)
    PETAL_DEVICE_ID     Hardware identifier (default: unoq-001)
    PETAL_FIRMWARE      Firmware version string (default: 1.0.0)
    PETAL_USER_EMAIL    User email for JWT login (needed for profile PATCH)
    PETAL_USER_PASS     User password
    PETAL_PACKAGE_DIR   Where packages are installed (default: ~/.petal/packages)
    UNOQ_TEST_IMAGE     Path to a test image file for image-model inference
    UNOQ_CAMERA_INDEX   OpenCV camera index (default: 0)
    PETAL_DEBUG         Set to 1 for verbose logging
"""
from __future__ import annotations

import asyncio
import logging
import os

from .config import PETAL_HOST, PETAL_DEVICE_ID, PETAL_FIRMWARE
from .package_loader import _load_active_package
from .runtime_backend import UnoQRuntime

_level = logging.DEBUG if os.environ.get("PETAL_DEBUG") == "1" else logging.INFO
logging.basicConfig(
    level=_level,
    format="%(asctime)s  %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("unoq")


def _startup_checks() -> None:
    logger.info("Starting UNO Q runtime  host=%s  device_id=%s  fw=%s",
                PETAL_HOST, PETAL_DEVICE_ID, PETAL_FIRMWARE)
    try:
        import numpy  # noqa: F401
    except ImportError:
        logger.warning("[startup] numpy missing — install: pip install numpy")
    try:
        import tflite_runtime  # noqa: F401
    except ImportError:
        try:
            import tensorflow  # noqa: F401
        except ImportError:
            logger.warning("[startup] tflite_runtime missing — install: pip install tflite-runtime")
    try:
        import cv2  # noqa: F401
    except ImportError:
        logger.info("[startup] opencv-python not installed — camera capture unavailable; "
                    "set UNOQ_TEST_IMAGE for image inference")
    try:
        from PIL import Image  # noqa: F401
    except ImportError:
        logger.info("[startup] Pillow not installed — image resize uses raw buffer fallback")
    active = _load_active_package()
    if active:
        logger.info("[startup] active package: %s", active)
    else:
        logger.info("[startup] no installed package yet — run OTA from the Studio UI")


if __name__ == "__main__":
    _startup_checks()
    runtime = UnoQRuntime()
    try:
        asyncio.run(runtime.run())
    except KeyboardInterrupt:
        logger.info("stopped")
