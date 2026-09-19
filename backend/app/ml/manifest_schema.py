"""Canonical .pxe manifest schema validator.

The .pxe manifest follows the Edge Impulse .eim wire format: enumerations are
the EI canonical values, project_id is an integer, every field a device
consumer needs to parse outputs without runtime heuristics is declared
explicitly. The packager invokes :func:`validate_pxe_manifest` before sealing
the .pxe container so malformed manifests fail at build time, not on device.
"""
from __future__ import annotations

import re
from typing import Any

PXE_MANIFEST_VERSION = 1

# EI-canonical model_type wire vocabulary. Internal training names
# (yolo_pro_detection, detection_heatmap) are translated by the packager.
ALLOWED_MODEL_TYPES = frozenset({
    "classification",
    "object_detection",
    "constrained_object_detection",
    "anomaly_gmm",
    "regression",
})

# Image resize mode wire vocabulary. fit-longest is EI-aligned and used by
# petaledge's "Fit longest axis" DSP block.
ALLOWED_IMAGE_RESIZE_MODES = frozenset({
    "fit-shortest",
    "fit-longest",
    "squash",
    "crop",
})

ALLOWED_COORDINATE_SPACES = frozenset({"normalized", "pixel"})

ALLOWED_OUTPUT_TENSOR_FORMATS = frozenset({
    "decoded",
    "raw_multi_head",
    "fomo_heatmap",
    "softmax",
})

ALLOWED_CHANNEL_ORDERS = frozenset({"rgb", "bgr", "grayscale"})

_MODEL_HASH_RE = re.compile(r"^sha256:[0-9a-f]{64}$")

# Fields whose types must match exactly. (name, expected_type) — bool is
# rejected for int fields because bool is a subclass of int in Python.
_REQUIRED_FIELDS: tuple[tuple[str, type | tuple[type, ...]], ...] = (
    ("version",                int),
    ("project_id",             int),
    ("project_name",           str),
    ("model_type",             str),
    ("input_shape",            list),
    ("image_input_frames",     int),
    ("image_resize_mode",      str),
    ("coordinate_space",       str),
    ("output_tensor_format",   str),
    ("label_names",            list),
    ("num_classes",            int),
    ("threshold",              (int, float)),
    ("thresholds",             list),
    ("iou_threshold",          (int, float)),
    ("has_anomaly",            bool),
    ("normalize_input",        bool),
    ("input_mean",             (int, float)),
    ("input_std",              (int, float)),
    ("channel_order",          str),
    ("model_hash",             str),
)


def _check_type(field: str, value: Any, expected: type | tuple[type, ...]) -> None:
    types = expected if isinstance(expected, tuple) else (expected,)
    # bool subclasses int — reject when we wanted only int/float numerics
    if int in types and float not in types and isinstance(value, bool):
        raise ValueError(
            f".pxe manifest field {field!r}: must be int, got bool {value!r}"
        )
    if bool not in types and isinstance(value, bool) and not (int in types or float in types):
        # not numeric, not bool — bool not allowed
        pass
    if not isinstance(value, types):
        type_names = "/".join(t.__name__ for t in types)
        raise ValueError(
            f".pxe manifest field {field!r}: must be {type_names}, "
            f"got {type(value).__name__} ({value!r})"
        )


def validate_pxe_manifest(manifest: dict) -> None:
    """Strict validation of a .pxe manifest dict. Raises ValueError on the
    first problem found so the build error is unambiguous.
    """
    if not isinstance(manifest, dict):
        raise ValueError(f".pxe manifest must be a dict, got {type(manifest).__name__}")

    # ── 1. Required fields present with correct types ──────────────────────
    for field, expected in _REQUIRED_FIELDS:
        if field not in manifest:
            raise ValueError(f".pxe manifest is missing required field {field!r}")
        _check_type(field, manifest[field], expected)

    # ── 2. version ─────────────────────────────────────────────────────────
    if manifest["version"] != PXE_MANIFEST_VERSION:
        raise ValueError(
            f".pxe manifest.version: expected {PXE_MANIFEST_VERSION}, "
            f"got {manifest['version']!r}"
        )

    # ── 3. project_id is a positive int (never a UUID string) ──────────────
    if manifest["project_id"] <= 0:
        raise ValueError(
            f".pxe manifest.project_id: must be a positive int, "
            f"got {manifest['project_id']!r}"
        )

    # ── 4. Enumerations ────────────────────────────────────────────────────
    if manifest["model_type"] not in ALLOWED_MODEL_TYPES:
        raise ValueError(
            f".pxe manifest.model_type={manifest['model_type']!r} not in "
            f"{sorted(ALLOWED_MODEL_TYPES)}"
        )
    if manifest["image_resize_mode"] not in ALLOWED_IMAGE_RESIZE_MODES:
        raise ValueError(
            f".pxe manifest.image_resize_mode={manifest['image_resize_mode']!r} "
            f"not in {sorted(ALLOWED_IMAGE_RESIZE_MODES)}"
        )
    if manifest["coordinate_space"] not in ALLOWED_COORDINATE_SPACES:
        raise ValueError(
            f".pxe manifest.coordinate_space={manifest['coordinate_space']!r} "
            f"not in {sorted(ALLOWED_COORDINATE_SPACES)}"
        )
    if manifest["output_tensor_format"] not in ALLOWED_OUTPUT_TENSOR_FORMATS:
        raise ValueError(
            f".pxe manifest.output_tensor_format={manifest['output_tensor_format']!r} "
            f"not in {sorted(ALLOWED_OUTPUT_TENSOR_FORMATS)}"
        )
    if manifest["channel_order"] not in ALLOWED_CHANNEL_ORDERS:
        raise ValueError(
            f".pxe manifest.channel_order={manifest['channel_order']!r} "
            f"not in {sorted(ALLOWED_CHANNEL_ORDERS)}"
        )

    # ── 5. input_shape: list of positive ints ──────────────────────────────
    input_shape = manifest["input_shape"]
    if len(input_shape) == 0:
        raise ValueError(".pxe manifest.input_shape: must be non-empty")
    bad = [
        i for i, d in enumerate(input_shape)
        if not isinstance(d, int) or isinstance(d, bool) or d <= 0
    ]
    if bad:
        raise ValueError(
            f".pxe manifest.input_shape: invalid dimension(s) at {bad}: {input_shape!r}"
        )

    # ── 6. labels ──────────────────────────────────────────────────────────
    labels = manifest["label_names"]
    if len(labels) == 0:
        raise ValueError(".pxe manifest.label_names: must be non-empty")
    bad = [
        i for i, l in enumerate(labels)
        if not isinstance(l, str) or not l.strip()
    ]
    if bad:
        raise ValueError(
            f".pxe manifest.label_names: blank/non-string entries at {bad}"
        )
    if manifest["num_classes"] != len(labels):
        raise ValueError(
            f".pxe manifest.num_classes={manifest['num_classes']} != "
            f"len(label_names)={len(labels)}"
        )

    # ── 7. thresholds: one entry per label, each {label, value} ────────────
    thresholds = manifest["thresholds"]
    if len(thresholds) != len(labels):
        raise ValueError(
            f".pxe manifest.thresholds: length {len(thresholds)} != "
            f"label_names length {len(labels)}"
        )
    label_set = list(labels)
    for i, entry in enumerate(thresholds):
        if not isinstance(entry, dict):
            raise ValueError(
                f".pxe manifest.thresholds[{i}]: must be a dict, got {entry!r}"
            )
        if entry.get("label") != label_set[i]:
            raise ValueError(
                f".pxe manifest.thresholds[{i}].label={entry.get('label')!r} "
                f"must equal label_names[{i}]={label_set[i]!r} (order must match)"
            )
        v = entry.get("value")
        if not isinstance(v, (int, float)) or isinstance(v, bool):
            raise ValueError(
                f".pxe manifest.thresholds[{i}].value: must be number, got {v!r}"
            )

    # ── 8. input_std must be non-zero (manifest preprocessing divides by it) ─
    if float(manifest["input_std"]) == 0.0:
        raise ValueError(
            ".pxe manifest.input_std: must be non-zero "
            "(runner divides input by input_std during preprocessing)"
        )

    # ── 9. model_hash format ───────────────────────────────────────────────
    if not _MODEL_HASH_RE.match(manifest["model_hash"]):
        raise ValueError(
            f".pxe manifest.model_hash={manifest['model_hash']!r}: "
            "must match 'sha256:<64 hex chars>'"
        )

    # ── 10. reg_max (YOLO-only): when present, must be a positive int ──────
    if "reg_max" in manifest:
        rm = manifest["reg_max"]
        if not isinstance(rm, int) or isinstance(rm, bool) or rm <= 0:
            raise ValueError(
                f".pxe manifest.reg_max: must be positive int when present, got {rm!r}"
            )
