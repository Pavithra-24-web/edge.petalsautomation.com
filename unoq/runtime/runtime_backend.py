"""UnoQRuntime — the top-level device daemon class."""
from __future__ import annotations

import asyncio
import json
import logging
import pathlib
import subprocess
import types
from typing import Any, Optional

import aiohttp

from .config import (
    PETAL_HOST, PETAL_API_KEY, PETAL_DEVICE_ID, PETAL_FIRMWARE,
    PETAL_PACKAGE_DIR, _ACTIVE_PKG_FILE,
)
from .dsp_engine import _DspEngineMixin
from .inference_engine import _InferenceEngineMixin
from .package_loader import _load_inference_module
from .postprocess_engine import _PostprocessEngineMixin
from .protocol import _ProtocolMixin
from .validator import validate_pxe_dir

logger = logging.getLogger("unoq")


class UnoQRuntime(
    _ProtocolMixin,
    _InferenceEngineMixin,
    _DspEngineMixin,
    _PostprocessEngineMixin,
):
    """
    Petal Edge — UNO Q Runtime Daemon

    Connects a UNO Q Linux device to the Petal Edge backend.
    Handles: WS hello, heartbeat, OTA .pe/.pxe package download+unpack+activation,
             and local TFLite inference streaming.
    """

    def __init__(self) -> None:
        self.host             = PETAL_HOST
        self.api_key          = PETAL_API_KEY
        self.device_id        = PETAL_DEVICE_ID
        self.firmware_version = PETAL_FIRMWARE

        self._device_pk:       Optional[str]         = None
        self._project_id:      Optional[str]         = None
        self._jwt:             Optional[str]         = None
        self._session:         Optional[aiohttp.ClientSession] = None
        self._running:         bool                  = True
        self._inference_active: bool                 = False
        self._inference_task:  Optional[asyncio.Task] = None
        self._stream_sensor:           Optional[str]   = None
        self._stream_frequency_hz:     Optional[float] = None
        self._stream_sample_length_ms: Optional[int]   = None

        self._pkg_dir:       Optional[pathlib.Path]  = None
        self._infer_mod:     Optional[types.ModuleType] = None
        self._interpreter:   Any                     = None
        self._infer_tick:    int                     = 0

        self._cv2_warned = False

        self._pkg_format: str                         = "pe"
        self._pxe_proc:   Optional[subprocess.Popen] = None

        self._snapshot_active: bool                  = False
        self._snapshot_task:   Optional[asyncio.Task] = None

        self._pp_config:      dict                    = {}
        self._tracker_state:  dict                    = {}

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    async def run(self) -> None:
        async with aiohttp.ClientSession() as session:
            self._session = session
            await self._login()
            await asyncio.gather(
                self._ws_loop(),
                self._heartbeat_loop(),
            )

    # ── Package loading ───────────────────────────────────────────────────────

    def _load_installed_package(
        self,
        pkg_dir: Optional[pathlib.Path] = None,
        pkg_format: Optional[str] = None,
    ) -> None:
        """Load (or reload) the active installed package."""
        if pkg_dir is None:
            if not _ACTIVE_PKG_FILE.exists():
                logger.info("[pkg] no installed package — inference unavailable until OTA")
                return
            try:
                info       = json.loads(_ACTIVE_PKG_FILE.read_text())
                pkg_dir    = PETAL_PACKAGE_DIR / info["deployment_id"]
                pkg_format = info.get("format", "pe")
            except Exception as exc:
                logger.warning("[pkg] could not read active.json: %s", exc)
                return

        if not pkg_dir.exists():
            logger.warning("[pkg] package dir missing: %s", pkg_dir)
            return

        self._pkg_dir    = pkg_dir
        self._pkg_format = pkg_format or "pe"

        # Reset postprocess config and tracker state on every package (re)load.
        self._pp_config     = {}
        self._tracker_state = {}
        pp_path = pkg_dir / "postprocess_config.json"
        if pp_path.exists():
            try:
                self._pp_config = json.loads(pp_path.read_text())
                logger.info("[pkg] postprocess_config loaded: tracking_enabled=%s",
                            self._pp_config.get("tracking_enabled", False))
            except Exception as exc:
                logger.warning("[pkg] could not load postprocess_config.json: %s", exc)

        if self._pkg_format == "pxe":
            # Validate on startup (non-strict — warn only, so old packages still load).
            validate_pxe_dir(pkg_dir, strict=False)
            self._infer_mod   = None
            self._interpreter = None
            self._launch_pxe_runner()
        else:
            self._infer_mod   = _load_inference_module(pkg_dir)
            self._interpreter = None
            if self._infer_mod is not None:
                logger.info("[pkg] .pe package loaded from %s", pkg_dir)
            else:
                logger.warning("[pkg] inference module failed to load from %s", pkg_dir)
