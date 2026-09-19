"""
Phase 4 — PXE Validation audit tests (UnoQ device-side).

Covers validate_pxe_sections() and validate_pxe_dir():
  1.  Valid v2 sections pass strict validation
  2.  Valid v1 sections (no postprocess) pass strict validation
  3.  Missing manifest.json → specific error, strict raises / warn-only logs
  4.  Malformed manifest.json (invalid JSON) → specific error
  5.  Missing dsp_config.json → specific error
  6.  Malformed dsp_config.json → specific error
  7.  Missing labels.txt → specific error
  8.  Malformed postprocess_config.json → strict raises
  9.  manifest.version != dsp_config.version → mismatch detected
  10. labels.txt lines != manifest.label_names → mismatch detected
  11. strict=False on version mismatch → only logs (no exception)
  12. strict=False on labels mismatch → only logs (no exception)
  13. validate_pxe_dir() skips postprocess_config.json when file is absent
  14. validate_pxe_dir() validates postprocess_config.json when file is present
  15. validate_pxe_dir() reports missing mandatory files clearly
"""
from __future__ import annotations

import json
import logging
import pathlib
import sys
from pathlib import Path

import pytest

# ── path setup ────────────────────────────────────────────────────────────────

REPO_ROOT = Path(__file__).parents[2]
UNOQ_ROOT = REPO_ROOT / "unoq"
for p in (str(REPO_ROOT), str(UNOQ_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from runtime.validator import validate_pxe_sections, validate_pxe_dir


# ── helpers ──────────────────────────────────────────────────────────────────

def _valid_manifest(version: int = 1, label_names: list | None = None) -> dict:
    return {
        "version":      version,
        "label_names":  label_names if label_names is not None else ["cat", "dog"],
        "input_shape":  [96, 96, 3],
        "model_type":   "classification",
    }


def _valid_dsp(version: int = 1) -> dict:
    return {
        "version":   version,
        "dsp_blocks": [{"type": "image", "params": {}}],
        "interval_ms": 10.0,
    }


def _valid_pp() -> dict:
    return {
        "enabled":          True,
        "threshold":        0.5,
        "class_filter":     [],
        "tracking_enabled": False,
        "keep_grace":       3,
        "max_observations": 5,
    }


def _sections(
    *,
    manifest: dict | None = None,
    dsp: dict | None = None,
    labels: list | None = None,
    pp: dict | None = None,
    include_pp: bool = True,
    manifest_raw: bytes | None = None,
    dsp_raw: bytes | None = None,
    labels_raw: bytes | None = None,
    pp_raw: bytes | None = None,
    omit: set | None = None,
) -> dict:
    if omit is None:
        omit = set()

    result = {}
    if "manifest.json" not in omit:
        result["manifest.json"] = manifest_raw or json.dumps(
            manifest if manifest is not None else _valid_manifest()
        ).encode()
    if "dsp_config.json" not in omit:
        result["dsp_config.json"] = dsp_raw or json.dumps(
            dsp if dsp is not None else _valid_dsp()
        ).encode()
    if "labels.txt" not in omit:
        lbls = labels if labels is not None else ["cat", "dog"]
        result["labels.txt"] = labels_raw or "\n".join(lbls).encode()
    if include_pp and "postprocess_config.json" not in omit:
        result["postprocess_config.json"] = pp_raw or json.dumps(
            pp if pp is not None else _valid_pp()
        ).encode()
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Valid packages pass
# ═══════════════════════════════════════════════════════════════════════════════

def test_valid_v2_sections_pass_strict():
    validate_pxe_sections(_sections(), strict=True)  # must not raise


def test_valid_v1_sections_pass_strict():
    """v1 sections (no postprocess_config.json) still pass validation."""
    validate_pxe_sections(_sections(include_pp=False), strict=True)


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Missing required sections
# ═══════════════════════════════════════════════════════════════════════════════

def test_missing_manifest_raises_strict():
    with pytest.raises(ValueError, match="manifest.json"):
        validate_pxe_sections(_sections(omit={"manifest.json"}), strict=True)


def test_missing_dsp_raises_strict():
    with pytest.raises(ValueError, match="dsp_config.json"):
        validate_pxe_sections(_sections(omit={"dsp_config.json"}), strict=True)


def test_missing_labels_raises_strict():
    with pytest.raises(ValueError, match="labels.txt"):
        validate_pxe_sections(_sections(omit={"labels.txt"}), strict=True)


def test_missing_manifest_warns_not_strict(caplog):
    with caplog.at_level(logging.WARNING, logger="unoq"):
        validate_pxe_sections(_sections(omit={"manifest.json"}), strict=False)
    assert any("manifest.json" in r.message for r in caplog.records)


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Malformed JSON sections
# ═══════════════════════════════════════════════════════════════════════════════

def test_malformed_manifest_json_raises_strict():
    with pytest.raises(ValueError, match="manifest.json"):
        validate_pxe_sections(_sections(manifest_raw=b"not json {{{"), strict=True)


def test_malformed_dsp_json_raises_strict():
    with pytest.raises(ValueError, match="dsp_config.json"):
        validate_pxe_sections(_sections(dsp_raw=b"not json {{{"), strict=True)


def test_malformed_postprocess_json_raises_strict():
    with pytest.raises(ValueError, match="[Pp]ost[Pp]rocess|postprocess"):
        validate_pxe_sections(_sections(pp_raw=b"not json {{{"), strict=True)


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Cross-section invariant mismatches
# ═══════════════════════════════════════════════════════════════════════════════

def test_version_mismatch_raises_strict():
    secs = _sections(
        manifest=_valid_manifest(version=1),
        dsp=_valid_dsp(version=2),
    )
    with pytest.raises(ValueError, match="[Vv]ersion"):
        validate_pxe_sections(secs, strict=True)


def test_labels_mismatch_raises_strict():
    secs = _sections(
        manifest=_valid_manifest(label_names=["cat", "dog"]),
        labels=["cat", "fish"],
    )
    with pytest.raises(ValueError, match="[Ll]abel"):
        validate_pxe_sections(secs, strict=True)


def test_version_mismatch_warns_not_strict(caplog):
    secs = _sections(
        manifest=_valid_manifest(version=1),
        dsp=_valid_dsp(version=99),
    )
    with caplog.at_level(logging.WARNING, logger="unoq"):
        validate_pxe_sections(secs, strict=False)  # must not raise
    assert any("version" in r.message.lower() for r in caplog.records)


def test_labels_mismatch_warns_not_strict(caplog):
    secs = _sections(
        manifest=_valid_manifest(label_names=["cat", "dog"]),
        labels=["apple", "banana"],
    )
    with caplog.at_level(logging.WARNING, logger="unoq"):
        validate_pxe_sections(secs, strict=False)  # must not raise
    assert any("label" in r.message.lower() for r in caplog.records)


# ═══════════════════════════════════════════════════════════════════════════════
# 5. postprocess_config field validation
# ═══════════════════════════════════════════════════════════════════════════════

def test_bad_postprocess_tracking_enabled_type_raises():
    bad_pp = {**_valid_pp(), "tracking_enabled": "yes"}  # should be bool
    with pytest.raises(ValueError, match="tracking_enabled"):
        validate_pxe_sections(_sections(pp=bad_pp), strict=True)


def test_bad_postprocess_threshold_range_raises():
    bad_pp = {**_valid_pp(), "threshold": 1.5}
    with pytest.raises(ValueError, match="threshold"):
        validate_pxe_sections(_sections(pp=bad_pp), strict=True)


def test_bad_postprocess_keep_grace_negative_raises():
    bad_pp = {**_valid_pp(), "keep_grace": -1}
    with pytest.raises(ValueError, match="keep_grace"):
        validate_pxe_sections(_sections(pp=bad_pp), strict=True)


def test_bad_postprocess_max_observations_zero_raises():
    bad_pp = {**_valid_pp(), "max_observations": 0}
    with pytest.raises(ValueError, match="max_observations"):
        validate_pxe_sections(_sections(pp=bad_pp), strict=True)


# ═══════════════════════════════════════════════════════════════════════════════
# 6. validate_pxe_dir() — directory-based validation
# ═══════════════════════════════════════════════════════════════════════════════

def _write_pkg(tmp_path: pathlib.Path, secs: dict) -> pathlib.Path:
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    for name, content in secs.items():
        (pkg / name).write_bytes(content)
    return pkg


def test_validate_pxe_dir_valid_package_passes(tmp_path):
    pkg = _write_pkg(tmp_path, _sections(include_pp=True))
    validate_pxe_dir(pkg, strict=True)  # must not raise


def test_validate_pxe_dir_v1_package_no_pp_passes(tmp_path):
    """Package without postprocess_config.json is silently skipped for pp checks."""
    pkg = _write_pkg(tmp_path, _sections(include_pp=False))
    validate_pxe_dir(pkg, strict=True)  # must not raise


def test_validate_pxe_dir_with_bad_postprocess_raises(tmp_path):
    bad_pp = {**_valid_pp(), "keep_grace": -5}
    pkg = _write_pkg(tmp_path, _sections(pp=bad_pp))
    with pytest.raises(ValueError, match="keep_grace"):
        validate_pxe_dir(pkg, strict=True)


def test_validate_pxe_dir_missing_manifest_reports(tmp_path, caplog):
    """Missing mandatory file is reported (warn-only when strict=False)."""
    pkg = tmp_path / "empty_pkg"
    pkg.mkdir()
    # Only write dsp and labels, no manifest
    (pkg / "dsp_config.json").write_bytes(json.dumps(_valid_dsp()).encode())
    (pkg / "labels.txt").write_bytes(b"cat\ndog")
    with caplog.at_level(logging.WARNING, logger="unoq"):
        validate_pxe_dir(pkg, strict=False)  # must not raise
    assert any("manifest" in r.message.lower() or "package" in r.message.lower()
               for r in caplog.records)
