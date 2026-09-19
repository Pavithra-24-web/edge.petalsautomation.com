"""
Tests for the live .pxe inference path in UnoQRuntime.

Covers:
  1. _normalize_pxe_result() FOMO bbox shape — detections use
     { label, confidence, bbox: {x1,y1,x2,y2} } with [0,1]-normalized coords.
  2. _normalize_pxe_result() classification shape — unchanged.
  3. _normalize_pxe_result() bad result — returns None.
  4. _normalize_pxe_result() FOMO empty bbox list — count=0, detections=[].
  5. _normalize_pxe_result() bbox coords are normalized by manifest input_shape.
  6. _run_pxe_inference_tick() uses _raw_frame_pixels_for_pxe for image models,
     not _features_from_image (no double-preprocessing).
  7. _raw_frame_pixels_for_pxe() returns raw pixel array without resize.
  8. _raw_frame_pixels_for_pxe() returns None when no frame source available.
  9. Regression: FOMO live payload is frontend-compatible
     ({ label, confidence, bbox: {x1,y1,x2,y2} } shape matches backend contract).
 10. _run_pxe_inference_tick() still uses _features_from_sensor for sensor models.
"""
from __future__ import annotations

import io
import pathlib
import sys
import types
from unittest.mock import MagicMock, patch, call

import pytest

# ── helpers ──────────────────────────────────────────────────────────────────

def _import_runtime_module():
    rt_dir = str(pathlib.Path(__file__).parent.parent)
    if rt_dir not in sys.path:
        sys.path.insert(0, rt_dir)
    # Clear the package and all submodules to avoid stale caches across tests.
    for key in list(sys.modules.keys()):
        if key == "runtime" or key.startswith("runtime."):
            del sys.modules[key]
    import runtime as _mod
    return _mod


def _make_runtime() -> object:
    mod = _import_runtime_module()
    cls = mod.UnoQRuntime
    obj = cls.__new__(cls)
    obj._inference_active        = False
    obj._inference_task          = None
    obj._snapshot_active         = False
    obj._snapshot_task           = None
    obj._stream_sensor           = None
    obj._stream_frequency_hz     = None
    obj._stream_sample_length_ms = None
    obj._pkg_dir                 = pathlib.Path("/fake/pkg")
    obj._infer_mod               = None
    obj._interpreter             = None
    obj._infer_tick              = 0
    obj._pxe_proc                = None
    obj._pkg_format              = "pxe"
    obj._cv2_warned              = False
    return obj


def _make_pil_jpeg(width: int = 48, height: int = 32, channels: int = 3) -> bytes:
    """Return a minimal JPEG of given dimensions."""
    try:
        from PIL import Image
        import numpy as np
        if channels == 1:
            arr = (
                (lambda h, w: (h[:, None] * 5 + w[None, :] * 3) % 256)(
                    __import__("numpy").arange(height),
                    __import__("numpy").arange(width),
                )
            ).astype("uint8")
            img = Image.fromarray(arr, mode="L")
        else:
            import numpy as np
            arr = ((
                __import__("numpy").indices((height, width))[0] * 5
                + __import__("numpy").indices((height, width))[1] * 3
            ) % 256).astype("uint8")
            rgb = __import__("numpy").stack([arr, arr // 2, arr // 3], axis=-1)
            img = Image.fromarray(rgb, mode="RGB")
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        return buf.getvalue()
    except ImportError:
        pytest.skip("PIL not available")


# ── 1. FOMO bbox shape ────────────────────────────────────────────────────────

def test_normalize_pxe_result_fomo_detections_have_bbox_key():
    rt = _make_runtime()
    raw = {
        "result": 1,
        "bounding_boxes": [
            {"label": "cat", "value": 0.85, "x": 24, "y": 16, "width": 16, "height": 16},
        ],
        "timing": {"dsp": 1, "classification": 2, "anomaly": 0},
    }
    manifest = {"input_shape": [96, 96, 3]}
    result = rt._normalize_pxe_result(raw, manifest)
    assert result is not None
    assert result["is_fomo"] is True
    det = result["detections"][0]
    assert "bbox" in det, f"expected 'bbox' key in detection, got: {list(det.keys())}"
    assert "x1" in det["bbox"]
    assert "y1" in det["bbox"]
    assert "x2" in det["bbox"]
    assert "y2" in det["bbox"]


def test_normalize_pxe_result_fomo_no_legacy_xywh():
    rt = _make_runtime()
    raw = {
        "result": 1,
        "bounding_boxes": [
            {"label": "dog", "value": 0.9, "x": 10, "y": 20, "width": 30, "height": 30},
        ],
    }
    manifest = {"input_shape": [96, 96, 3]}
    result = rt._normalize_pxe_result(raw, manifest)
    det = result["detections"][0]
    assert "x" not in det, "legacy 'x' key must not appear in detection"
    assert "y" not in det
    assert "w" not in det
    assert "h" not in det


# ── 2. Classification shape unchanged ─────────────────────────────────────────

def test_normalize_pxe_result_classification_shape():
    rt = _make_runtime()
    raw = {
        "result": 1,
        "classification": {"cat": 0.1, "dog": 0.9},
        "timing": {},
    }
    result = rt._normalize_pxe_result(raw, {})
    assert result is not None
    assert result["is_fomo"] is False
    assert result["label"] == "dog"
    assert abs(result["confidence"] - 0.9) < 1e-6
    assert "predictions" in result


def test_normalize_pxe_result_yolo_passthrough_shape():
    rt = _make_runtime()
    raw = {
        "result": 1,
        "detections": [{"label": "cat", "confidence": 0.9, "bbox": {"x1": 0.1, "y1": 0.1, "x2": 0.4, "y2": 0.4}}],
        "count": 1,
        "timing": {},
    }
    result = rt._normalize_pxe_result(raw, {"model_type": "yolo_pro_detection"})
    assert result is not None
    assert result["is_detection"] is True
    assert result["is_fomo"] is False
    assert result["model_type"] == "yolo_pro_detection"
    assert result["count"] == 1


def test_normalize_pxe_result_yolo_stale_runner_returns_empty_detection_set():
    rt = _make_runtime()
    raw = {
        "result": 1,
        "classification": {"cat": 0.2, "dog": 0.8},
        "timing": {},
    }
    result = rt._normalize_pxe_result(raw, {"model_type": "yolo_pro_detection"})
    assert result is not None
    assert result["is_detection"] is True
    assert result["model_type"] == "yolo_pro_detection"
    assert result["detections"] == []
    assert result["debug"]["stale_pxe_artifact"] is True


# ── 3. Bad result returns None ─────────────────────────────────────────────────

def test_normalize_pxe_result_error_returns_none():
    rt = _make_runtime()
    raw = {"result": 0, "error": "model crashed"}
    result = rt._normalize_pxe_result(raw, {})
    assert result is None


# ── 4. Empty FOMO bbox list ───────────────────────────────────────────────────

def test_normalize_pxe_result_fomo_empty_detections():
    rt = _make_runtime()
    raw = {"result": 1, "bounding_boxes": []}
    result = rt._normalize_pxe_result(raw, {"input_shape": [96, 96, 3]})
    assert result is not None
    assert result["is_fomo"] is True
    assert result["count"] == 0
    assert result["detections"] == []


# ── 5. Bbox coords are normalized by manifest input_shape ─────────────────────

def test_normalize_pxe_result_fomo_coords_normalized():
    rt = _make_runtime()
    raw = {
        "result": 1,
        "bounding_boxes": [
            {"label": "cat", "value": 0.75, "x": 48, "y": 32, "width": 16, "height": 16},
        ],
    }
    manifest = {"input_shape": [64, 96, 3]}   # H=64, W=96
    result = rt._normalize_pxe_result(raw, manifest)
    bbox = result["detections"][0]["bbox"]
    # x=48/W=96=0.5, y=32/H=64=0.5, x2=(48+16)/96=0.667, y2=(32+16)/64=0.75
    assert abs(bbox["x1"] - 48 / 96) < 1e-6
    assert abs(bbox["y1"] - 32 / 64) < 1e-6
    assert abs(bbox["x2"] - (48 + 16) / 96) < 1e-6
    assert abs(bbox["y2"] - (32 + 16) / 64) < 1e-6


def test_normalize_pxe_result_fomo_coords_in_unit_range():
    rt = _make_runtime()
    raw = {
        "result": 1,
        "bounding_boxes": [
            {"label": "cat", "value": 0.9, "x": 0, "y": 0, "width": 96, "height": 96},
        ],
    }
    result = rt._normalize_pxe_result(raw, {"input_shape": [96, 96, 3]})
    bbox = result["detections"][0]["bbox"]
    assert bbox["x1"] == 0.0
    assert bbox["y1"] == 0.0
    assert abs(bbox["x2"] - 1.0) < 1e-6
    assert abs(bbox["y2"] - 1.0) < 1e-6


# ── 6. _run_pxe_inference_tick uses _raw_frame_pixels_for_pxe for images ─────

def test_run_pxe_inference_tick_image_model_calls_raw_frame_not_features_from_image():
    """For 3D input_shape, the tick must call _raw_frame_pixels_for_pxe,
    not _features_from_image, so the runner applies its own DSP pipeline."""
    import numpy as np
    import json

    rt = _make_runtime()

    manifest = {"input_shape": [96, 96, 3]}
    fake_proc = MagicMock()
    fake_proc.poll.return_value = None
    fake_proc.stdout.readline.return_value = (
        json.dumps({"result": 1, "classification": {"cat": 0.9, "dog": 0.1}}) + "\n"
    ).encode()
    rt._pxe_proc = fake_proc

    fake_arr = np.zeros((48, 32, 3), dtype="float32")

    manifest_json = json.dumps(manifest).encode()

    with (
        patch.object(rt, "_raw_frame_pixels_for_pxe", return_value=fake_arr) as mock_raw,
        patch.object(rt, "_features_from_image") as mock_img,
        patch.object(pathlib.Path, "read_text", return_value=json.dumps(manifest)),
    ):
        result = rt._run_pxe_inference_tick()

    mock_raw.assert_called_once_with(manifest)
    mock_img.assert_not_called()
    payload = json.loads(fake_proc.stdin.write.call_args[0][0].decode())
    assert len(payload["classify"]) == 48
    assert len(payload["classify"][0]) == 32
    assert len(payload["classify"][0][0]) == 3


# ── 7. _raw_frame_pixels_for_pxe returns array without resize ─────────────────

def test_raw_frame_pixels_for_pxe_returns_original_dimensions():
    """The returned array must have the original image dimensions, not resized
    to model input shape, so the runner's DSP handles resize correctly."""
    import numpy as np

    rt = _make_runtime()
    jpeg_bytes = _make_pil_jpeg(width=48, height=32, channels=3)
    manifest   = {"input_shape": [96, 96, 3], "channel_order": "rgb"}

    with patch("runtime.dsp_engine._capture_frame",return_value=jpeg_bytes):
        arr = rt._raw_frame_pixels_for_pxe(manifest)

    assert arr is not None
    assert arr.dtype == np.float32
    # Must NOT be resized to model input (96x96); original is 48x32
    assert arr.shape[0] == 32, f"height should be 32 (original), got {arr.shape[0]}"
    assert arr.shape[1] == 48, f"width should be 48 (original), got {arr.shape[1]}"


def test_raw_frame_pixels_for_pxe_values_in_uint8_range():
    """Pixel values must be in 0–255 (raw, not pre-normalized)."""
    rt = _make_runtime()
    jpeg_bytes = _make_pil_jpeg(width=16, height=16, channels=3)
    manifest   = {"channel_order": "rgb"}

    with patch("runtime.dsp_engine._capture_frame",return_value=jpeg_bytes):
        arr = rt._raw_frame_pixels_for_pxe(manifest)

    assert arr is not None
    assert arr.min() >= 0.0
    assert arr.max() <= 255.0


def test_raw_frame_pixels_for_pxe_grayscale():
    """channel_order='grayscale' → single-channel output (H, W, 1)."""
    rt = _make_runtime()
    jpeg_bytes = _make_pil_jpeg(width=16, height=16, channels=1)
    manifest   = {"channel_order": "grayscale"}

    with patch("runtime.dsp_engine._capture_frame",return_value=jpeg_bytes):
        arr = rt._raw_frame_pixels_for_pxe(manifest)

    assert arr is not None
    assert arr.shape[-1] == 1, f"expected 1 channel, got {arr.shape[-1]}"


# ── 8. _raw_frame_pixels_for_pxe returns None when no frame ──────────────────

def test_raw_frame_pixels_for_pxe_no_frame_returns_none():
    rt = _make_runtime()
    with patch("runtime.dsp_engine._capture_frame",return_value=None):
        result = rt._raw_frame_pixels_for_pxe({})
    assert result is None


def test_raw_frame_pixels_for_pxe_warns_once_on_missing_frame():
    rt = _make_runtime()
    rt._cv2_warned = False
    with (
        patch("runtime.dsp_engine._capture_frame",return_value=None),
        patch("runtime.dsp_engine.logger") as mock_log,
    ):
        rt._raw_frame_pixels_for_pxe({})
        rt._raw_frame_pixels_for_pxe({})  # second call should not warn again

    warning_calls = [c for c in mock_log.warning.call_args_list if "no image source" in str(c)]
    assert len(warning_calls) == 1, "warning should fire exactly once"


# ── 9. Regression: FOMO payload is frontend-compatible ───────────────────────

def test_fomo_live_payload_matches_backend_contract():
    """The normalized FOMO result shape from the live .pxe path must match
    the shape produced by the backend _normalize_pxe_classify_result path.

    Backend contract (from inference.py _normalize_pxe_classify_result):
      {
        "detections": [{"label": str, "confidence": float,
                         "bbox": {"x1": float, "y1": float,
                                  "x2": float, "y2": float}}],
        "count": int,
        "is_fomo": True,
      }
    """
    rt = _make_runtime()
    raw = {
        "result": 1,
        "bounding_boxes": [
            {"label": "person", "value": 0.92, "x": 12, "y": 8, "width": 24, "height": 24},
            {"label": "cat",    "value": 0.71, "x": 60, "y": 40, "width": 16, "height": 16},
        ],
        "timing": {"dsp": 5, "classification": 10, "anomaly": 0},
    }
    manifest = {"input_shape": [96, 96, 3]}
    result = rt._normalize_pxe_result(raw, manifest)

    # Top-level keys
    assert "detections" in result
    assert "count" in result
    assert result["is_fomo"] is True
    assert result["count"] == 2

    for det in result["detections"]:
        assert set(det.keys()) >= {"label", "confidence", "bbox"}, (
            f"detection missing required keys: {set(det.keys())}"
        )
        assert isinstance(det["label"], str)
        assert isinstance(det["confidence"], float)
        bbox = det["bbox"]
        assert set(bbox.keys()) == {"x1", "y1", "x2", "y2"}, (
            f"bbox keys wrong: {set(bbox.keys())}"
        )
        for v in bbox.values():
            assert 0.0 <= v <= 1.0, f"bbox coord {v} out of [0,1] for {det}"
        assert bbox["x2"] > bbox["x1"]
        assert bbox["y2"] > bbox["y1"]


# ── 10. Sensor models still use _features_from_sensor ─────────────────────────

def test_run_pxe_inference_tick_sensor_model_calls_features_from_sensor():
    """1D input_shape → must use _features_from_sensor, not _raw_frame_pixels_for_pxe."""
    import numpy as np
    import json

    rt = _make_runtime()

    manifest = {"input_shape": [128]}
    fake_proc = MagicMock()
    fake_proc.poll.return_value = None
    fake_proc.stdout.readline.return_value = (
        json.dumps({"result": 1, "classification": {"idle": 0.8, "walk": 0.2}}) + "\n"
    ).encode()
    rt._pxe_proc = fake_proc

    fake_arr = np.zeros((128,), dtype="float32")

    with (
        patch.object(rt, "_features_from_sensor", return_value=fake_arr) as mock_sensor,
        patch.object(rt, "_raw_frame_pixels_for_pxe") as mock_raw,
        patch.object(pathlib.Path, "read_text", return_value=json.dumps(manifest)),
    ):
        rt._run_pxe_inference_tick()

    mock_sensor.assert_called_once_with([128])
    mock_raw.assert_not_called()
    payload = json.loads(fake_proc.stdin.write.call_args[0][0].decode())
    assert len(payload["classify"]) == 128
