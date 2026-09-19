from __future__ import annotations

import sys
import tempfile
from pathlib import Path


REPO_ROOT = Path(__file__).parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "backend") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "backend"))
if str(REPO_ROOT / "unoq") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "unoq"))

from app.ml.pxe_runner import _unpack_pxe as _backend_unpack_pxe
from app.workers.deployment_worker import (
    _PXE_HEADER_STRUCT,
    _PXE_MAGIC,
    _gen_pxe,
)
from runtime.package_loader import _unpack_pxe
from runtime.validator import validate_pxe_sections


def _make_synthetic_tflite() -> bytes:
    from app.workers.deployment_worker import _TFLITE_FILE_IDENTIFIER

    return b"\x18\x00\x00\x00" + _TFLITE_FILE_IDENTIFIER + (b"\x00" * 600)


def _base_meta(extra: dict | None = None) -> dict:
    data = {
        "label_names": ["cat", "dog"],
        "input_shape": [96, 96, 3],
        "normalize_input": True,
        "channel_order": "rgb",
        "model_type": "classification",
        "output_type": "classification",
        "architecture": "mobilenet_v2",
        "version": 1,
        "threshold": 0.5,
        "has_anomaly": False,
        "image_input_frames": 1,
        "dsp_blocks": [{"type": "image", "params": {"resize_mode": "Fit shortest axis"}}],
        "frequency_hz": 100.0,
        "n_axes": 1,
        "project_id": 1,
        "project_name": "test",
    }
    if extra:
        data.update(extra)
    return data


def _repack_as_v1_pxe(pxe_v2_bytes: bytes) -> bytes:
    with tempfile.TemporaryDirectory(prefix="unoq_pxe_v1_") as d:
        pkg_dir = Path(d)
        _backend_unpack_pxe(pxe_v2_bytes, pkg_dir)
        runner_bytes = (pkg_dir / "runner.py").read_bytes()
        model_bytes = (pkg_dir / "model.tflite").read_bytes()
        manifest_bytes = (pkg_dir / "manifest.json").read_bytes()
        labels_bytes = (pkg_dir / "labels.txt").read_bytes()
        dsp_bytes = (pkg_dir / "dsp_config.json").read_bytes()
    header = _PXE_HEADER_STRUCT.pack(
        _PXE_MAGIC,
        1,
        len(runner_bytes),
        len(model_bytes),
        len(manifest_bytes),
        len(labels_bytes),
        len(dsp_bytes),
        0,
    )
    return b"".join((
        header,
        runner_bytes,
        model_bytes,
        manifest_bytes,
        labels_bytes,
        dsp_bytes,
    ))


import struct as _struct
import pytest


# ── Loader rejection tests (Phase 4) ─────────────────────────────────────────

def _corrupt_byte(data: bytes, offset: int, value: int) -> bytes:
    ba = bytearray(data)
    ba[offset] = value
    return bytes(ba)


def test_unoq_loader_rejects_truncated_header():
    """Package shorter than minimum v1 header raises ValueError."""
    with pytest.raises(ValueError, match="too small"):
        _unpack_pxe(b"PXE1\x00" * 2)   # only 10 bytes — well under 32-byte header


def test_unoq_loader_rejects_wrong_magic():
    """Package with bad magic bytes raises ValueError."""
    pkg = _gen_pxe(_make_synthetic_tflite(), _base_meta(), {})
    bad = _corrupt_byte(pkg, 0, ord("X"))   # overwrite 'P' with 'X' → "XXE1"
    with pytest.raises(ValueError, match="[Mm]agic"):
        _unpack_pxe(bad)


def test_unoq_loader_rejects_unsupported_version():
    """Package with unknown version field raises ValueError."""
    pkg = _gen_pxe(_make_synthetic_tflite(), _base_meta(), {})
    # Version is bytes 4-7 (uint32 LE). Set version = 99.
    ba = bytearray(pkg)
    _struct.pack_into("<I", ba, 4, 99)
    with pytest.raises(ValueError, match="[Vv]ersion"):
        _unpack_pxe(bytes(ba))


def test_unoq_loader_rejects_declared_size_larger_than_actual():
    """Truncated payload (declared > actual total) raises ValueError."""
    pkg = _gen_pxe(_make_synthetic_tflite(), _base_meta(), {})
    with pytest.raises(ValueError, match="[Mm]alformed|declares"):
        _unpack_pxe(pkg[:-10])   # remove last 10 bytes


def test_unoq_loader_rejects_declared_size_smaller_than_actual():
    """Extra trailing bytes (declared < actual total) raises ValueError."""
    pkg = _gen_pxe(_make_synthetic_tflite(), _base_meta(), {})
    with pytest.raises(ValueError, match="[Mm]alformed|declares"):
        _unpack_pxe(pkg + b"\x00" * 20)   # append 20 spurious bytes


def test_unoq_loader_accepts_old_v1_pxe_without_postprocess_section():
    pkg_v2 = _gen_pxe(_make_synthetic_tflite(), _base_meta(), {})
    pkg_v1 = _repack_as_v1_pxe(pkg_v2)
    sections = _unpack_pxe(pkg_v1)
    assert "postprocess_config.json" not in sections
    assert sections["runner.py"]
    assert sections["model.tflite"]


def test_unoq_loader_accepts_new_v2_pxe_with_postprocess_section():
    sections = _unpack_pxe(_gen_pxe(_make_synthetic_tflite(), _base_meta(), {}))
    assert "postprocess_config.json" in sections
    validate_pxe_sections(sections, strict=True)
