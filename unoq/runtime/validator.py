"""Cross-section integrity checks for installed PXE packages."""
from __future__ import annotations

import json
import logging
import pathlib
from typing import Dict

logger = logging.getLogger("unoq")


def _validate_postprocess_config_dict(cfg: dict) -> None:
    """Validate a deserialized postprocess_config.json dict.

    Checks only real supported fields from the PostProcessingSettings DB schema.
    Raises ValueError on constraint violations; unknown extra keys are ignored
    for forward-compatibility.
    """
    if not isinstance(cfg.get("enabled"), bool):
        raise ValueError("postprocess_config.enabled must be a bool")
    th = cfg.get("threshold")
    if not isinstance(th, (int, float)) or not (0.0 <= float(th) <= 1.0):
        raise ValueError(f"postprocess_config.threshold must be float in [0,1], got {th!r}")
    cf = cfg.get("class_filter")
    if not isinstance(cf, list) or not all(isinstance(x, str) for x in cf):
        raise ValueError("postprocess_config.class_filter must be a list of strings")
    te = cfg.get("tracking_enabled")
    if not isinstance(te, bool):
        raise ValueError("postprocess_config.tracking_enabled must be a bool")
    kg = cfg.get("keep_grace")
    if not isinstance(kg, int) or kg < 0:
        raise ValueError(f"postprocess_config.keep_grace must be int >= 0, got {kg!r}")
    mo = cfg.get("max_observations")
    if not isinstance(mo, int) or mo < 1:
        raise ValueError(f"postprocess_config.max_observations must be int >= 1, got {mo!r}")


def validate_pxe_sections(sections: Dict[str, bytes], *, strict: bool = True) -> None:
    """Validate cross-section invariants for a freshly-unpacked PXE package.

    Checks:
    - manifest.version == dsp_config.version
    - labels.txt lines == manifest.label_names
    - postprocess_config.json (if present) passes schema validation

    With strict=True (OTA install path) raises ValueError on any mismatch.
    With strict=False (startup load path) logs a warning instead.
    Old valid v1 packages always satisfy both base invariants.
    Old packages without postprocess_config.json are silently skipped.
    """
    try:
        manifest = json.loads(sections["manifest.json"])
    except KeyError:
        _report("manifest.json section is missing from the PXE package", strict)
        return
    except Exception as exc:
        _report(f"manifest.json is not valid JSON: {exc}", strict)
        return

    try:
        dsp_config = json.loads(sections["dsp_config.json"])
    except KeyError:
        _report("dsp_config.json section is missing from the PXE package", strict)
        return
    except Exception as exc:
        _report(f"dsp_config.json is not valid JSON: {exc}", strict)
        return

    try:
        labels_txt = [l for l in sections["labels.txt"].decode().splitlines() if l]
    except KeyError:
        _report("labels.txt section is missing from the PXE package", strict)
        return
    except Exception as exc:
        _report(f"labels.txt could not be decoded: {exc}", strict)
        return

    m_ver = manifest.get("version")
    d_ver = dsp_config.get("version")
    if m_ver != d_ver:
        _report(
            f"manifest.version={m_ver!r} does not match dsp_config.version={d_ver!r}",
            strict,
        )

    m_labels = manifest.get("label_names", [])
    if labels_txt != m_labels:
        _report(
            f"labels.txt {labels_txt!r} does not match manifest.label_names {m_labels!r}",
            strict,
        )

    pp_bytes = sections.get("postprocess_config.json")
    if pp_bytes:
        try:
            pp_cfg = json.loads(pp_bytes)
        except Exception as exc:
            _report(f"postprocess_config.json is not valid JSON: {exc}", strict)
            return
        try:
            _validate_postprocess_config_dict(pp_cfg)
        except ValueError as exc:
            _report(str(exc), strict)


def validate_pxe_dir(pkg_dir: pathlib.Path, *, strict: bool = False) -> None:
    """Load sections from an installed package directory and validate them."""
    try:
        sections: Dict[str, bytes] = {
            "manifest.json":   (pkg_dir / "manifest.json").read_bytes(),
            "dsp_config.json": (pkg_dir / "dsp_config.json").read_bytes(),
            "labels.txt":      (pkg_dir / "labels.txt").read_bytes(),
        }
    except Exception as exc:
        _report(f"Could not read package files from {pkg_dir}: {exc}", strict)
        return

    # postprocess_config.json is optional (absent in v1 packages)
    pp_path = pkg_dir / "postprocess_config.json"
    if pp_path.exists():
        sections["postprocess_config.json"] = pp_path.read_bytes()

    validate_pxe_sections(sections, strict=strict)


def _report(msg: str, strict: bool) -> None:
    if strict:
        raise ValueError(f"[validator] {msg}")
    logger.warning("[validator] %s", msg)
