"""
Phase 3 PXE Tracking Integration tests.

Covers:
  A.  Packaging
      1.  Tracking fields present in v2 postprocess_config.json (defaults)
      2.  Tracking fields present when tracking_enabled=True is packaged
      3.  Old v1 packages load without postprocess_config.json (backward compat)
      4.  v2 packages with tracking_enabled=True validate at strict=True

  B.  Runtime — _apply_pxe_tracking()
      5.  tracking_enabled=False → result returned unchanged (no object_id)
      6.  tracking_enabled=True  → YOLO detection result gets object_id assigned
      7.  tracking_enabled=True  → second frame reuses same object_id (stable ID)
      8.  tracking_enabled=True  → carry-forward on missed frame (is_carried=True)
      9.  tracking_enabled=True  → FOMO result returned unchanged (not tracked)
      10. tracking_enabled=True  → classification result returned unchanged
      11. None result passed through unchanged
      12. No _pp_config attr (bare object) → result unchanged (backward compat)
      13. Tracker state resets on package reload

  C.  PxeTracker unit tests (unoq/runtime/tracker.py)
      14. New track gets object_id=1 on first frame
      15. Matching detection across frames keeps same object_id
      16. Unmatched track within keep_grace emits is_carried=True
      17. Unmatched track beyond keep_grace is expired
      18. max_observations caps box_history
      19. Center-distance fallback matches fast-moving objects
      20. Empty detection list with existing tracks → carry-forwards only
"""
from __future__ import annotations

import json
import pathlib
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ── path setup ────────────────────────────────────────────────────────────────

REPO_ROOT  = Path(__file__).parents[2]
UNOQ_ROOT  = REPO_ROOT / "unoq"
BACKEND_ROOT = REPO_ROOT / "backend"

for p in (str(REPO_ROOT), str(UNOQ_ROOT), str(BACKEND_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)


# ── helpers ──────────────────────────────────────────────────────────────────

def _import_runtime_module():
    rt_dir = str(UNOQ_ROOT)
    if rt_dir not in sys.path:
        sys.path.insert(0, rt_dir)
    for key in list(sys.modules.keys()):
        if key == "runtime" or key.startswith("runtime."):
            del sys.modules[key]
    import runtime as _mod
    return _mod


def _make_runtime(pp_config: dict | None = None) -> object:
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
    obj._pp_config               = pp_config if pp_config is not None else {}
    obj._tracker_state           = {}
    return obj


def _yolo_result(detections: list) -> dict:
    return {
        "is_detection": True,
        "is_fomo": False,
        "model_type": "yolo_pro_detection",
        "detections": detections,
        "count": len(detections),
        "timing": {},
    }


def _det(label: str, x1: float, y1: float, x2: float, y2: float,
         confidence: float = 0.9) -> dict:
    return {
        "label": label,
        "confidence": confidence,
        "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
    }


# ── packaging helpers ─────────────────────────────────────────────────────────

def _make_synthetic_tflite() -> bytes:
    from app.workers.deployment_worker import _TFLITE_FILE_IDENTIFIER
    return b"\x18\x00\x00\x00" + _TFLITE_FILE_IDENTIFIER + (b"\x00" * 600)


def _base_meta(extra: dict | None = None) -> dict:
    data = {
        "label_names":       ["cat", "dog"],
        "input_shape":       [96, 96, 3],
        "normalize_input":   True,
        "channel_order":     "rgb",
        "model_type":        "yolo_pro_detection",
        "output_type":       "yolo_pro_detection",
        "architecture":      "yolo_pro",
        "version":           1,
        "threshold":         0.5,
        "has_anomaly":       False,
        "image_input_frames": 1,
        "dsp_blocks":        [{"type": "image", "params": {"resize_mode": "Fit shortest axis"}}],
        "frequency_hz":      100.0,
        "n_axes":            1,
        "project_id":        1,
        "project_name":      "test",
    }
    if extra:
        data.update(extra)
    return data


# ═══════════════════════════════════════════════════════════════════════════════
# A — PACKAGING
# ═══════════════════════════════════════════════════════════════════════════════

def test_v2_package_contains_tracking_fields_with_defaults():
    """Default (no pp_settings) v2 package has tracking_enabled=False etc."""
    from app.workers.deployment_worker import _gen_pxe
    from runtime.package_loader import _unpack_pxe

    sections = _unpack_pxe(_gen_pxe(_make_synthetic_tflite(), _base_meta(), {}))
    assert "postprocess_config.json" in sections
    cfg = json.loads(sections["postprocess_config.json"])
    assert isinstance(cfg["tracking_enabled"], bool)
    assert isinstance(cfg["keep_grace"], int)
    assert isinstance(cfg["max_observations"], int)
    assert cfg["tracking_enabled"] is False


def test_v2_package_contains_tracking_enabled_true_when_set():
    """Packages built with tracking_enabled=True in settings persist the flag."""
    import types as _types
    from app.workers.deployment_worker import _gen_pxe, _build_postprocess_config
    from runtime.package_loader import _unpack_pxe

    settings = _types.SimpleNamespace(
        enabled=True, threshold=0.5, class_filter=[],
        tracking_enabled=True, keep_grace=5, max_observations=8,
    )
    pp_bytes = _build_postprocess_config(settings)
    sections = _unpack_pxe(_gen_pxe(_make_synthetic_tflite(), _base_meta(), {},
                                    postprocess_bytes=pp_bytes))
    cfg = json.loads(sections["postprocess_config.json"])
    assert cfg["tracking_enabled"] is True
    assert cfg["keep_grace"] == 5
    assert cfg["max_observations"] == 8


def test_old_v1_package_loads_without_postprocess_section():
    """v1 packages (no postprocess_config.json) still load unchanged."""
    from app.workers.deployment_worker import _gen_pxe, _PXE_HEADER_STRUCT, _PXE_MAGIC
    from app.ml.pxe_runner import _unpack_pxe as _backend_unpack
    from runtime.package_loader import _unpack_pxe

    # Build a v2 package then repack as v1 (strip postprocess section)
    pxe_v2 = _gen_pxe(_make_synthetic_tflite(), _base_meta(), {})
    with tempfile.TemporaryDirectory(prefix="unoq_v1_") as d:
        pkg_dir = Path(d)
        _backend_unpack(pxe_v2, pkg_dir)
        runner_bytes   = (pkg_dir / "runner.py").read_bytes()
        model_bytes    = (pkg_dir / "model.tflite").read_bytes()
        manifest_bytes = (pkg_dir / "manifest.json").read_bytes()
        labels_bytes   = (pkg_dir / "labels.txt").read_bytes()
        dsp_bytes      = (pkg_dir / "dsp_config.json").read_bytes()

    header = _PXE_HEADER_STRUCT.pack(
        _PXE_MAGIC, 1,
        len(runner_bytes), len(model_bytes), len(manifest_bytes),
        len(labels_bytes), len(dsp_bytes), 0,
    )
    v1_bytes = header + runner_bytes + model_bytes + manifest_bytes + labels_bytes + dsp_bytes
    sections = _unpack_pxe(v1_bytes)
    assert "postprocess_config.json" not in sections
    assert sections["runner.py"]


def test_v2_package_with_tracking_enabled_true_validates_strictly():
    """validate_pxe_sections raises no error for tracking_enabled=True package."""
    import types as _types
    from app.workers.deployment_worker import _gen_pxe, _build_postprocess_config
    from runtime.package_loader import _unpack_pxe
    from runtime.validator import validate_pxe_sections

    settings = _types.SimpleNamespace(
        enabled=True, threshold=0.4, class_filter=["cat"],
        tracking_enabled=True, keep_grace=2, max_observations=4,
    )
    pp_bytes = _build_postprocess_config(settings)
    sections = _unpack_pxe(_gen_pxe(_make_synthetic_tflite(), _base_meta(), {},
                                    postprocess_bytes=pp_bytes))
    validate_pxe_sections(sections, strict=True)  # must not raise


# ═══════════════════════════════════════════════════════════════════════════════
# B — RUNTIME _apply_pxe_tracking()
# ═══════════════════════════════════════════════════════════════════════════════

def test_tracking_disabled_yolo_result_unchanged():
    """tracking_enabled=False → detections pass through without object_id."""
    rt = _make_runtime(pp_config={"tracking_enabled": False, "keep_grace": 3, "max_observations": 5})
    result = _yolo_result([_det("cat", 0.1, 0.1, 0.4, 0.4)])
    out = rt._apply_pxe_tracking(result)
    assert out is result  # same object, untouched
    assert "object_id" not in out["detections"][0]


def test_tracking_enabled_assigns_object_id():
    """tracking_enabled=True → first-frame detection gets object_id assigned."""
    rt = _make_runtime(pp_config={"tracking_enabled": True, "keep_grace": 3, "max_observations": 5})
    result = _yolo_result([_det("cat", 0.1, 0.1, 0.4, 0.4)])
    out = rt._apply_pxe_tracking(result)
    assert len(out["detections"]) == 1
    assert "object_id" in out["detections"][0]
    assert isinstance(out["detections"][0]["object_id"], int)
    assert out["detections"][0]["is_carried"] is False


def test_tracking_stable_id_across_frames():
    """Same detection position in frame 2 gets the same object_id as frame 1."""
    rt = _make_runtime(pp_config={"tracking_enabled": True, "keep_grace": 3, "max_observations": 5})
    det = _det("cat", 0.1, 0.1, 0.4, 0.4)

    out1 = rt._apply_pxe_tracking(_yolo_result([det]))
    id1 = out1["detections"][0]["object_id"]

    out2 = rt._apply_pxe_tracking(_yolo_result([det]))
    id2 = out2["detections"][0]["object_id"]

    assert id1 == id2, f"Expected stable ID {id1}, got {id2} on frame 2"


def test_tracking_carry_forward_on_missed_frame():
    """After one missed frame within keep_grace, carry-forward detection is emitted."""
    rt = _make_runtime(pp_config={"tracking_enabled": True, "keep_grace": 3, "max_observations": 5})
    det = _det("cat", 0.1, 0.1, 0.4, 0.4)

    # Frame 1: detection present → track created
    out1 = rt._apply_pxe_tracking(_yolo_result([det]))
    obj_id = out1["detections"][0]["object_id"]

    # Frame 2: empty detection list → track should carry forward
    out2 = rt._apply_pxe_tracking(_yolo_result([]))
    assert len(out2["detections"]) == 1
    carried = out2["detections"][0]
    assert carried["is_carried"] is True
    assert carried["object_id"] == obj_id


def test_tracking_fomo_result_unchanged():
    """FOMO results (is_fomo=True) are never tracked — returned as-is."""
    rt = _make_runtime(pp_config={"tracking_enabled": True, "keep_grace": 3, "max_observations": 5})
    fomo_result = {
        "is_fomo": True,
        "is_detection": False,
        "model_type": "detection_heatmap",
        "detections": [_det("cat", 0.1, 0.1, 0.3, 0.3)],
        "count": 1,
        "timing": {},
    }
    out = rt._apply_pxe_tracking(fomo_result)
    assert out is fomo_result
    assert "object_id" not in out["detections"][0]


def test_tracking_classification_result_unchanged():
    """Classification results (is_detection=False) are never tracked."""
    rt = _make_runtime(pp_config={"tracking_enabled": True, "keep_grace": 3, "max_observations": 5})
    clf_result = {
        "is_fomo": False,
        "is_detection": False,
        "label": "cat",
        "confidence": 0.9,
        "predictions": [{"label": "cat", "confidence": 0.9}],
        "timing": {},
    }
    out = rt._apply_pxe_tracking(clf_result)
    assert out is clf_result


def test_tracking_none_result_passthrough():
    """None result (inference error) is returned as None unchanged."""
    rt = _make_runtime(pp_config={"tracking_enabled": True, "keep_grace": 3, "max_observations": 5})
    assert rt._apply_pxe_tracking(None) is None


def test_tracking_no_pp_config_attr_backward_compat():
    """Runtime object without _pp_config attr behaves as if tracking disabled."""
    rt = _make_runtime(pp_config={})
    delattr(rt, "_pp_config")  # simulate old runtime instance
    result = _yolo_result([_det("cat", 0.1, 0.1, 0.4, 0.4)])
    out = rt._apply_pxe_tracking(result)
    assert out is result  # unchanged


def test_tracker_state_resets_on_package_reload(tmp_path):
    """_tracker_state clears when _load_installed_package() is called."""
    from app.workers.deployment_worker import _gen_pxe

    pxe_bytes = _gen_pxe(_make_synthetic_tflite(), _base_meta(), {})

    # Unpack into a temp directory to simulate an installed package dir
    from app.ml.pxe_runner import _unpack_pxe as _backend_unpack
    pkg_dir = tmp_path / "dep1"
    pkg_dir.mkdir()
    _backend_unpack(pxe_bytes, pkg_dir)

    rt = _make_runtime()
    rt._tracker_state = {"tracks": {"1": {"object_id": 1, "x1": 0, "y1": 0, "x2": 0, "y2": 0,
                                           "label": "cat", "confidence": 0.9,
                                           "missed_frames": 0, "observations": 1,
                                           "box_history": [[0, 0, 0.3, 0.3]]}},
                         "next_id": 2}

    # Reload the package — state must clear
    rt._load_installed_package(pkg_dir=pkg_dir, pkg_format="pxe")

    assert rt._tracker_state == {}, "Tracker state must reset on package reload"


# ═══════════════════════════════════════════════════════════════════════════════
# C — PxeTracker unit tests
# ═══════════════════════════════════════════════════════════════════════════════

def _tracker(keep_grace: int = 3, max_observations: int = 5):
    for key in list(sys.modules.keys()):
        if key == "runtime.tracker" or key == "runtime":
            pass  # let it cache
    from runtime.tracker import PxeTracker
    return PxeTracker(keep_grace=keep_grace, max_observations=max_observations)


def test_new_track_gets_object_id_1():
    t = _tracker()
    dets = [_det("cat", 0.1, 0.1, 0.4, 0.4)]
    out, state = t.update(dets, {})
    assert len(out) == 1
    assert out[0]["object_id"] == 1
    assert out[0]["is_carried"] is False
    assert state["next_id"] == 2


def test_matching_detection_keeps_same_object_id():
    t = _tracker()
    det = _det("cat", 0.1, 0.1, 0.4, 0.4)
    out1, state1 = t.update([det], {})
    assert out1[0]["object_id"] == 1

    out2, state2 = t.update([det], state1)
    assert out2[0]["object_id"] == 1, "ID must be stable on matching detection"


def test_unmatched_track_within_keep_grace_emits_carry_forward():
    t = _tracker(keep_grace=2)
    det = _det("cat", 0.1, 0.1, 0.4, 0.4)
    _, state = t.update([det], {})

    out, _ = t.update([], state)
    assert len(out) == 1
    assert out[0]["is_carried"] is True
    assert out[0]["object_id"] == 1


def test_unmatched_track_beyond_keep_grace_expires():
    t = _tracker(keep_grace=1)
    det = _det("cat", 0.1, 0.1, 0.4, 0.4)
    _, state = t.update([det], {})

    _, state = t.update([], state)  # missed_frames=1 (within grace)
    out, state = t.update([], state)  # missed_frames=2 (beyond grace=1)
    assert len(out) == 0, "Track should have expired"
    assert state["tracks"] == {}


def test_max_observations_caps_box_history():
    t = _tracker(max_observations=3)
    det = _det("cat", 0.1, 0.1, 0.4, 0.4)
    state: dict = {}
    for _ in range(10):
        _, state = t.update([det], state)
    track_data = list(state["tracks"].values())[0]
    assert len(track_data["box_history"]) <= 3


def test_center_distance_fallback_matches_fast_moving_object():
    """Object that moves too fast for IoU match still gets reassigned via center-dist."""
    from runtime.tracker import PxeTracker
    t = PxeTracker(iou_threshold=0.5, center_dist_threshold=0.4)

    # Frame 1: object at top-left
    det1 = _det("cat", 0.0, 0.0, 0.2, 0.2)
    _, state = t.update([det1], {})
    obj_id = list(state["tracks"].values())[0]["object_id"]

    # Frame 2: object has moved so IoU ≈ 0 but center distance is small
    det2 = _det("cat", 0.21, 0.21, 0.41, 0.41)
    out, _ = t.update([det2], state)
    assert len(out) == 1
    assert out[0]["object_id"] == obj_id, \
        "Center-distance fallback must reuse the same track ID"


def test_empty_detections_all_tracks_carry_forward():
    t = _tracker(keep_grace=3)
    det_a = _det("cat", 0.0, 0.0, 0.2, 0.2)
    det_b = _det("dog", 0.5, 0.5, 0.8, 0.8)
    _, state = t.update([det_a, det_b], {})

    out, _ = t.update([], state)
    assert len(out) == 2
    assert all(d["is_carried"] for d in out)
