"""PXE / PE package unpacking, installation, and activation."""
from __future__ import annotations

import importlib.util
import json
import logging
import pathlib
import struct
import sys
import time
import types
from typing import Dict, Optional

from .config import PETAL_PACKAGE_DIR, _ACTIVE_PKG_FILE

logger = logging.getLogger("unoq")

# .pe binary format constants (must match deployment_worker.py)
_PE_MAGIC          = b"PEM1"
_PE_HEADER_V2_FMT  = "<4sIIIIII"   # magic + ver + model + dsp + labels + manifest + inference
_PE_HEADER_V2_SIZE = struct.calcsize(_PE_HEADER_V2_FMT)

# .pxe binary format constants (must match deployment_worker.py and pxe_runner.py)
_PXE_MAGIC         = b"PXE1"
_PXE_HEADER_V1_FMT = "<4sIIIIIII"    # v1: magic+ver+runner+model+manifest+labels+dsp+flags (32 bytes)
_PXE_HEADER_V2_FMT = "<4sIIIIIIII"   # v2: adds postprocess_sz before flags (36 bytes)
_PXE_HEADER_FMT    = _PXE_HEADER_V1_FMT  # kept for any external references; prefer versioned names
_PXE_HEADER_SIZE   = struct.calcsize(_PXE_HEADER_V1_FMT)


def _unpack_pe(data: bytes) -> Dict[str, bytes]:
    """Parse a PEM1 v2 binary container → dict of section name → bytes."""
    if len(data) < _PE_HEADER_V2_SIZE:
        raise ValueError(f".pe package too small ({len(data)} bytes)")
    magic, ver, model_sz, dsp_sz, labels_sz, manifest_sz, inference_sz = struct.unpack_from(
        _PE_HEADER_V2_FMT, data, 0
    )
    if magic != _PE_MAGIC:
        raise ValueError(f"Bad .pe magic {magic!r}; expected {_PE_MAGIC!r}")
    if ver != 2:
        raise ValueError(f"Unsupported .pe format version {ver}; only v2 is supported")

    offset = _PE_HEADER_V2_SIZE
    sections: Dict[str, bytes] = {}
    for name, size in (
        ("model.tflite",    model_sz),
        ("dsp_config.json", dsp_sz),
        ("labels.txt",      labels_sz),
        ("manifest.json",   manifest_sz),
        ("inference.py",    inference_sz),
    ):
        sections[name] = data[offset : offset + size]
        offset += size
    return sections


def _detect_package_format(data: bytes) -> str:
    """Return 'pxe' or 'pe' based on magic bytes. Raises on unknown format."""
    if data[:4] == _PXE_MAGIC:
        return "pxe"
    if data[:4] == _PE_MAGIC:
        return "pe"
    raise ValueError(f"Unknown package magic {data[:4]!r}; expected PXE1 or PEM1")


def _unpack_pxe(data: bytes) -> Dict[str, bytes]:
    """Parse a PXE1 v1 or v2 binary container → dict of section name → bytes.

    v1 (format_version=1): 5 sections, 32-byte header, no postprocess_config.json
    v2 (format_version=2): 6 sections, 36-byte header, includes postprocess_config.json

    Old v1 packages load unchanged; new v2 packages expose the extra section.
    """
    v1_hdr_size = struct.calcsize(_PXE_HEADER_V1_FMT)
    if len(data) < v1_hdr_size:
        raise ValueError(f".pxe package too small ({len(data)} bytes)")

    # Peek at version field (bytes 4-7, uint32 LE)
    ver = struct.unpack_from("<I", data, 4)[0]

    if ver == 1:
        magic, _ver, runner_sz, model_sz, manifest_sz, labels_sz, dsp_sz, _flags = \
            struct.unpack_from(_PXE_HEADER_V1_FMT, data, 0)
        postprocess_sz = 0
        hdr_size = v1_hdr_size
    elif ver == 2:
        v2_hdr_size = struct.calcsize(_PXE_HEADER_V2_FMT)
        if len(data) < v2_hdr_size:
            raise ValueError(f".pxe v2 package too small ({len(data)} bytes)")
        magic, _ver, runner_sz, model_sz, manifest_sz, labels_sz, dsp_sz, postprocess_sz, _flags = \
            struct.unpack_from(_PXE_HEADER_V2_FMT, data, 0)
        hdr_size = v2_hdr_size
    else:
        raise ValueError(f"Unsupported .pxe format version {ver}; only v1 and v2 are supported")

    if magic != _PXE_MAGIC:
        raise ValueError(f"Bad .pxe magic {magic!r}; expected {_PXE_MAGIC!r}")

    expected_size = hdr_size + runner_sz + model_sz + manifest_sz + labels_sz + dsp_sz + postprocess_sz
    if len(data) != expected_size:
        raise ValueError(
            f"Malformed .pxe package: header declares {expected_size} bytes, got {len(data)}"
        )

    offset = hdr_size
    sections: Dict[str, bytes] = {}
    for name, size in (
        ("runner.py",       runner_sz),
        ("model.tflite",    model_sz),
        ("manifest.json",   manifest_sz),
        ("labels.txt",      labels_sz),
        ("dsp_config.json", dsp_sz),
    ):
        sections[name] = data[offset : offset + size]
        offset += size

    if postprocess_sz:
        sections["postprocess_config.json"] = data[offset : offset + postprocess_sz]

    return sections


def _install_package(deployment_id: str, sections: Dict[str, bytes]) -> pathlib.Path:
    """Write unpacked sections to PETAL_PACKAGE_DIR/<deployment_id>/."""
    pkg_dir = PETAL_PACKAGE_DIR / deployment_id
    pkg_dir.mkdir(parents=True, exist_ok=True)
    for filename, content in sections.items():
        (pkg_dir / filename).write_bytes(content)
    logger.info("[ota] installed package to %s", pkg_dir)
    return pkg_dir


def _activate_package(deployment_id: str, version: str, pkg_format: str = "pe") -> None:
    """Write active.json pointer so the package survives a restart."""
    PETAL_PACKAGE_DIR.mkdir(parents=True, exist_ok=True)
    _ACTIVE_PKG_FILE.write_text(json.dumps({
        "deployment_id": deployment_id,
        "version":        version,
        "format":         pkg_format,
        "activated_at":   time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }, indent=2))
    logger.info("[ota] activated package deployment_id=%s version=%s format=%s",
                deployment_id, version, pkg_format)


def _load_active_package() -> Optional[pathlib.Path]:
    """Return the path of the currently installed package, or None."""
    if not _ACTIVE_PKG_FILE.exists():
        return None
    try:
        info = json.loads(_ACTIVE_PKG_FILE.read_text())
        pkg_dir = PETAL_PACKAGE_DIR / info["deployment_id"]
        if pkg_dir.exists():
            return pkg_dir
    except Exception as exc:
        logger.warning("[pkg] could not read active.json: %s", exc)
    return None


def _load_inference_module(pkg_dir: pathlib.Path) -> Optional[types.ModuleType]:
    """Import inference.py from the installed package as a module."""
    inference_path = pkg_dir / "inference.py"
    if not inference_path.exists():
        logger.warning("[infer] inference.py not found in package %s", pkg_dir)
        return None
    try:
        spec = importlib.util.spec_from_file_location("unoq_inference", inference_path)
        mod  = importlib.util.module_from_spec(spec)          # type: ignore[arg-type]
        sys.path.insert(0, str(pkg_dir))
        spec.loader.exec_module(mod)                          # type: ignore[union-attr]
        sys.path.pop(0)
        return mod
    except Exception as exc:
        logger.error("[infer] failed to load inference.py: %s", exc)
        return None
