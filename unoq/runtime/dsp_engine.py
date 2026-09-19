"""DSP / sensor data preparation helpers and UnoQRuntime mixin."""
from __future__ import annotations

import io
import json
import logging
import math
import pathlib
import time
from typing import Any, Dict, List, Optional

from .config import UNOQ_TEST_IMAGE, UNOQ_CAMERA_INDEX, UNOQ_SAMPLE_FREQ_HZ

logger = logging.getLogger("unoq")

# Default sensor axes — 3-axis accelerometer, matches Edge Impulse convention
_DEFAULT_AXES: List[Dict[str, str]] = [
    {"name": "accX", "units": "m/s2"},
    {"name": "accY", "units": "m/s2"},
    {"name": "accZ", "units": "m/s2"},
]

try:
    import numpy as np
    _NP_AVAILABLE = True
except ImportError:
    np = None  # type: ignore[assignment]
    _NP_AVAILABLE = False


def _sine_accel_values(label: str, n_samples: int, freq_hz: float) -> List[float]:
    """
    Generate deterministic 3-axis accelerometer samples per label.
    Used as a software sensor fallback when no hardware driver is wired in.
    Wire a real I2C/SPI sensor driver here to produce real data.
    """
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
        out.extend([
            round(amp_xy * math.sin(phase), 4),
            round(amp_xy * math.cos(phase + 0.5), 4),
            round(z_bias + amp_z * math.sin(phase * 2 + 1.2), 4),
        ])
    return out


def _build_sample_json(
    label: str,
    length_ms: int,
    freq_hz: float,
    axes: List[Dict[str, str]],
) -> bytes:
    """Build an Edge Impulse–compatible signed JSON payload."""
    n_samples = max(1, int(length_ms / 1000 * freq_hz))
    values    = _sine_accel_values(label, n_samples, freq_hz)
    doc = {
        "payload": {
            "device_type": "unoq",
            "interval_ms": round(1000.0 / freq_hz, 4),
            "sensors":     axes,
            "values":      values,
        }
    }
    return json.dumps(doc).encode()


def _make_synthetic_jpeg() -> bytes:
    """Return a synthetic JPEG frame. Uses PIL if available, else a static 1×1 stub."""
    try:
        from PIL import Image, ImageDraw  # type: ignore[import]
        img  = Image.new("RGB", (96, 96), color=(30, 30, 30))
        draw = ImageDraw.Draw(img)
        for i in range(0, 96, 8):
            draw.line([(i, 0), (96, 96 - i)], fill=(80 + i * 2, 120, 200 - i), width=1)
        ts = int(time.time()) % 1000
        draw.rectangle([(2, 2), (2 + (ts % 90), 6)], fill=(255, 100, 0))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=60)
        return buf.getvalue()
    except ImportError:
        # minimal valid 1×1 white JPEG (107 bytes)
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


def _capture_frame(pkg_dir: Optional[pathlib.Path]) -> Optional[bytes]:
    """
    Capture a single camera frame as JPEG bytes.
    Priority: UNOQ_TEST_IMAGE env var → OpenCV camera → None (inference skipped).
    """
    if UNOQ_TEST_IMAGE:
        p = pathlib.Path(UNOQ_TEST_IMAGE)
        if p.exists():
            return p.read_bytes()
        logger.warning("[capture] UNOQ_TEST_IMAGE=%r not found", UNOQ_TEST_IMAGE)

    try:
        import cv2  # type: ignore[import]
        cap = cv2.VideoCapture(UNOQ_CAMERA_INDEX)
        ok, frame = cap.read()
        cap.release()
        if ok:
            _, buf = cv2.imencode(".jpg", frame)
            return buf.tobytes()
        logger.warning("[capture] camera index=%d: frame not captured", UNOQ_CAMERA_INDEX)
    except ImportError:
        pass
    except Exception as exc:
        logger.warning("[capture] camera error: %s", exc)

    return None


class _DspEngineMixin:
    """Mixin providing image and sensor feature-generation methods for UnoQRuntime."""

    _cv2_warned: bool
    _pkg_dir: Optional[pathlib.Path]
    _infer_tick: int
    _stream_frequency_hz: Optional[float]

    def _features_from_image(self, manifest: dict, input_shape: list) -> Optional[Any]:
        """Load and preprocess an image into the model input tensor."""
        if not _NP_AVAILABLE:
            return None
        frame_bytes = _capture_frame(self._pkg_dir)
        if frame_bytes is None:
            if not self._cv2_warned:
                logger.warning(
                    "[infer] no image source — set UNOQ_TEST_IMAGE=/path/to/image.jpg "
                    "or install opencv-python for live camera capture"
                )
                self._cv2_warned = True
            return None

        try:
            from PIL import Image  # type: ignore[import]
            img = Image.open(io.BytesIO(frame_bytes))
            h, w = int(input_shape[0]), int(input_shape[1])
            channels = int(input_shape[2]) if len(input_shape) > 2 else 3
            img = img.resize((w, h), Image.BILINEAR)
            img = img.convert("L" if channels == 1 else "RGB")
            arr = np.array(img, dtype=np.float32)
            if arr.ndim == 2:
                arr = arr[:, :, np.newaxis]
            return arr
        except ImportError:
            try:
                arr = np.frombuffer(frame_bytes, dtype=np.uint8).astype(np.float32)
                h, w = int(input_shape[0]), int(input_shape[1])
                c    = int(input_shape[2]) if len(input_shape) > 2 else 3
                return arr[: h * w * c].reshape(h, w, c)
            except Exception:
                return None
        except Exception as exc:
            logger.error("[infer] image preprocessing failed: %s", exc)
            return None

    def _raw_frame_pixels_for_pxe(self, manifest: dict) -> Optional[Any]:
        """Capture a camera frame and return raw pixel values for the .pxe runner.

        Unlike _features_from_image(), this method deliberately skips any resize
        or normalization.  The .pxe runner's embedded DSP pipeline (_dsp_image)
        applies the manifest's resize mode, channel conversion, and input_mean/
        input_std normalization.  Pre-applying those transforms here would
        double-process the image and destroy FOMO grid spatial semantics.
        """
        if not _NP_AVAILABLE:
            return None
        frame_bytes = _capture_frame(self._pkg_dir)
        if frame_bytes is None:
            if not self._cv2_warned:
                logger.warning(
                    "[infer] no image source — set UNOQ_TEST_IMAGE=/path/to/image.jpg "
                    "or install opencv-python for live camera capture"
                )
                self._cv2_warned = True
            return None
        try:
            from PIL import Image  # type: ignore[import]
            img = Image.open(io.BytesIO(frame_bytes))
            channel_order = str(manifest.get("channel_order", "rgb")).lower()
            img = img.convert("L" if channel_order == "grayscale" else "RGB")
            arr = np.array(img, dtype=np.float32)
            if arr.ndim == 2:
                arr = arr[:, :, np.newaxis]
            return arr
        except ImportError:
            try:
                return np.frombuffer(frame_bytes, dtype=np.uint8).astype(np.float32)
            except Exception:
                return None
        except Exception as exc:
            logger.error("[infer] raw frame decode failed: %s", exc)
            return None

    def _features_from_sensor(self, input_shape: list) -> Optional[Any]:
        """
        Build sensor feature vector for one inference tick.

        Uses sine-wave accelerometer data keyed by a cycling label so each tick
        produces distinct, non-zero values — giving the model a meaningful input
        instead of an all-zeros tensor that produces arbitrary class scores.

        To use real hardware: replace _sine_accel_values() in this method with
        driver calls (e.g. smbus2 for I2C IMU, pyserial for UART sensor module)
        and remove the sine-wave fallback.
        """
        if not _NP_AVAILABLE:
            return None

        n = 1
        for d in input_shape:
            n *= d

        # Resolve axes count + frequency: env-default → dsp_config → stream command (highest priority).
        n_axes  = 3
        freq_hz = UNOQ_SAMPLE_FREQ_HZ
        if self._pkg_dir:
            try:
                dsp_cfg = json.loads((self._pkg_dir / "dsp_config.json").read_text())
                cfg_axes = dsp_cfg.get("axes", [])
                if cfg_axes:
                    n_axes = len(cfg_axes)
                if dsp_cfg.get("interval_ms"):
                    freq_hz = 1000.0 / dsp_cfg["interval_ms"]
            except Exception:
                pass
        if self._stream_frequency_hz is not None:
            freq_hz = self._stream_frequency_hz

        _LABELS = ["idle", "walking", "running", "anomaly"]
        label   = _LABELS[(self._infer_tick // 20) % len(_LABELS)]

        n_samples = max(1, n // n_axes)
        raw       = _sine_accel_values(label, n_samples, freq_hz)
        arr       = np.array(raw[:n], dtype=np.float32)
        if len(arr) < n:
            arr = np.pad(arr, (0, n - len(arr)))
        logger.debug("[infer] sensor input label=%r n=%d freq=%.1f", label, n, freq_hz)
        return arr.reshape(input_shape)
