"""
Deployment Worker — generates edge deployment packages for multiple targets.

Targets:
  tflite       — model.tflite + Python inference wrapper
  arduino      — .zip Arduino library with C++ inference
  esp32        — ESP-IDF project archive
  raspberry_pi — Python wheel + inference script
  cpp          — Standalone C++17 project
"""
import os
import io
import json
import time
import pathlib
import zipfile
import struct
import tempfile
import logging
import hashlib
import zlib
from datetime import datetime
from string import Template

import numpy as np

from app.workers.celery_app import celery_app
from app.core.database import SessionLocal
from app.core.storage import storage
from app.ml.manifest_schema import (
    validate_pxe_manifest,
    PXE_MANIFEST_VERSION,
    ALLOWED_MODEL_TYPES,
    ALLOWED_IMAGE_RESIZE_MODES,
    ALLOWED_COORDINATE_SPACES,
    ALLOWED_OUTPUT_TENSOR_FORMATS,
)
from app.ml.pxe_filename import build_pxe_download_filename
from app.models.user import Deployment, TrainedModel, TrainingJob, JobStatus, Sample, Project, PostProcessingSettings
from app.core.logging_config import get_logger as _get_channel_logger, log_event as _log_event
from app.services.compatibility import (
    check_compatibility,
    classify_model_type,
    MODEL_TYPE_FOMO,
    MODEL_TYPE_SSD,
    MODEL_TYPE_YOLO_PRO,
)

logger = logging.getLogger(__name__)

# Structured deployment build metrics → logs/deployment.log
_deployment_log = _get_channel_logger("deployment")

_PE_MAGIC = b"PEM1"
_PE_FORMAT_VERSION = 2


def _safe_protocol_project_id(value):
    """Preserve packaged project ids; fall back only when the field is empty."""
    if value is None:
        return 1
    if isinstance(value, str):
        return value if value.strip() else 1
    return value


# ─── TFLite validation constants ─────────────────────────────────────────────
# Smallest plausible flatbuffer that contains at least one op and its tensors.
_TFLITE_MIN_BYTES = 512
# FlatBuffers layout: bytes 0-3 = root-table offset, bytes 4-7 = file identifier.
_TFLITE_FILE_IDENTIFIER_OFFSET = 4
_TFLITE_FILE_IDENTIFIER = b"TFL3"

# ─── PE container struct (matches inference.py) ────────────────────────────
# magic(4) | format_version(4) | model_sz(4) | dsp_sz(4) | labels_sz(4) | manifest_sz(4)
_PE_HEADER_STRUCT = struct.Struct("<4sIIIIII")

# ── .pxe binary container (fully separate from .pe / PEM1) ──────────────────
# Header layout (v1, little-endian, 32 bytes):
#   magic(4s) | format_version(I) | runner_sz(I) | model_sz(I) |
#   manifest_sz(I) | labels_sz(I) | dsp_sz(I) | flags(I)
#
# Header layout (v2, little-endian, 36 bytes):
#   magic(4s) | format_version(I) | runner_sz(I) | model_sz(I) |
#   manifest_sz(I) | labels_sz(I) | dsp_sz(I) | postprocess_sz(I) | flags(I)
#   Appends postprocess_config.json section (postprocess_sz bytes) after dsp_config.json.
#   postprocess_sz == 0 means no postprocess section (reserved, not currently emitted).
#   Old v1 loaders reject v2 packages gracefully; new loaders handle both.
_PXE_MAGIC             = b"PXE1"
_PXE_FORMAT_VERSION    = 1   # kept for v1 compatibility references
_PXE_FORMAT_VERSION_V2 = 2
_PXE_HEADER_STRUCT     = struct.Struct("<4sIIIIIII")   # v1: 32 bytes
_PXE_HEADER_STRUCT_V2  = struct.Struct("<4sIIIIIIII")  # v2: 36 bytes

# ─── Metadata validation constants ────────────────────────────────────────────
# (field_name, expected_python_type)  — order determines error message sequence.
_REQUIRED_META_FIELD_TYPES: tuple = (
    ("label_names",     list),
    ("input_shape",     list),
    ("normalize_input", bool),
    ("channel_order",   str),
    ("model_type",      str),
    ("version",         int),
)
_VALID_CHANNEL_ORDERS = frozenset({"rgb", "bgr", "grayscale"})


def _make_tflite_interpreter(*, model_content: bytes | None = None, model_path: str | None = None):
    """Create a TFLite interpreter with a Windows-safe fallback."""
    if model_content is None and model_path is None:
        raise ValueError("Either model_content or model_path must be provided.")

    kwargs = {}
    if model_content is not None:
        kwargs["model_content"] = model_content
    if model_path is not None:
        kwargs["model_path"] = model_path

    if os.name == "nt":
        try:
            import tensorflow as _tf
            return _tf.lite.Interpreter(
                **kwargs,
                experimental_op_resolver_type=(
                    _tf.lite.experimental.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES
                ),
            )
        except Exception as exc:
            logger.warning(
                "Windows-safe TFLite interpreter setup failed; falling back to default delegates: %s",
                exc,
            )

    try:
        from tflite_runtime.interpreter import Interpreter as _Interp
    except ImportError:
        import tensorflow as _tf
        _Interp = _tf.lite.Interpreter
    return _Interp(**kwargs)


@celery_app.task(bind=True, name="app.workers.deployment_worker.run_deployment_job")
def run_deployment_job(self, deployment_id: str):
    db = SessionLocal()
    dep = None
    _build_t0 = time.perf_counter()
    _resolved_target = None
    try:
        dep = db.query(Deployment).filter(Deployment.id == deployment_id).first()
        if not dep:
            raise ValueError(f"Deployment {deployment_id} not found")

        dep.status = JobStatus.running
        db.commit()
        _log_event(
            _deployment_log, "deployment.start",
            deployment_id=deployment_id, target=dep.target,
        )

        model = db.query(TrainedModel).filter(TrainedModel.id == dep.model_id).first()
        meta  = model.model_metadata or {}

        generators = {
            "tflite":       _gen_tflite,
            "arduino":      _gen_arduino,
            "esp32":        _gen_esp32,
            "raspberry_pi": _gen_raspberry_pi,
            "unoq":         _gen_unoq,
            "cpp":          _gen_cpp,
            "pxe":          _gen_pxe,
        }
        deployment_format = (dep.options or {}).get("deployment_format", "pe")
        if deployment_format == "pxe":
            resolved_target = "pxe"
            gen_fn          = generators["pxe"]
        else:
            resolved_target = _resolve_deployment_target(dep)
            gen_fn = generators.get(resolved_target)
            if not gen_fn:
                raise ValueError(f"Unknown deployment target: {resolved_target}")
        _resolved_target = resolved_target  # for the failure-path log below

        # Inject resolved device_profile into meta so _build_dsp_config can surface it.
        device_profile = getattr(dep, "device_profile", None) or (dep.options or {}).get("device_profile")
        if device_profile:
            meta = {**meta, "device_profile": device_profile}

        # Download TFLite model bytes, then validate before packaging.
        # PXE + YOLO Pro: use the decoded float32 TFLite so the embedded runner
        # receives a single (1, N, 6) tensor instead of raw 6-head outputs.
        # PXE + SSD: same decoded contract, but via the SSD-specific decoded
        # variant (separate path — never routes through the YOLO-Pro decoder).
        is_yolo_pro = meta.get("model_type") == "yolo_pro_detection"
        # Scope strictly to the SSD trainer's model_type. The legacy
        # "object_detection" label (also in _INTERNAL_TO_WIRE_MODEL_TYPE) keeps
        # its prior dispatch behaviour and is not rerouted here.
        is_ssd      = meta.get("model_type") == "ssd_detection"
        if deployment_format == "pxe" and is_yolo_pro:
            tflite_bytes = _get_decoded_tflite_bytes(db, model)
        elif deployment_format == "pxe" and is_ssd:
            tflite_bytes = _get_ssd_decoded_tflite_bytes(db, model)
            # Fix C: the SSD decoded graph keeps a uint8 input + in-graph
            # NormalizationLayer (x/127.5-1). The runner must pass raw uint8
            # through with NO extra [0,1] scaling, so force normalize_input
            # off for the SSD PXE manifest. Conditioned on SSD only — YOLO-Pro
            # and FOMO input contracts are untouched.
            meta = {**meta, "normalize_input": False}
        else:
            tflite_bytes = _get_tflite_bytes(db, model)
        _validate_tflite_bytes(tflite_bytes)          # size + TFL3 sig + interp load

        # Patch any derivable / legacy fields before strict validation runs.
        meta = _ensure_required_meta_defaults(meta)
        meta = _align_image_input_contract_with_tflite(meta, tflite_bytes)
        meta = _enrich_package_metadata(db, dep, model, meta, tflite_bytes)

        # Strict metadata validation before any packaging begins.
        _validate_manifest_meta(meta, "model_metadata")

        # For PXE builds: load saved post-processing settings and embed as
        # postprocess_config.json.  Falls back to defaults when no row exists.
        if resolved_target == "pxe":
            _project_id = meta.get("project_id") or getattr(dep, "project_id", None)
            _pp_settings = None
            if _project_id:
                _pp_settings = (
                    db.query(PostProcessingSettings)
                    .filter(PostProcessingSettings.project_id == str(_project_id))
                    .first()
                )
            postprocess_bytes = _build_postprocess_config(_pp_settings)
            package_bytes = _gen_pxe(tflite_bytes, meta, dep.options or {},
                                     postprocess_bytes=postprocess_bytes)
        else:
            package_bytes = gen_fn(tflite_bytes, meta, dep.options or {})

        # For PE-container targets: verify header-declared sizes round-trip exactly,
        # then run 1 test input through the model and assert detections is a list.
        if resolved_target in ("tflite", "raspberry_pi", "unoq"):
            _verify_pe_container(package_bytes)
            _post_build_pe_validation(package_bytes, meta)

        if resolved_target == "pxe":
            _verify_pxe_container(package_bytes)

        # Inference smoke test: run one forward pass on the packaged model bytes.
        _smoke_test_inference(tflite_bytes, meta)

        # Golden test: validate IoU > 0.5 against stored expected detections (if present).
        golden_result = _golden_test_validation(tflite_bytes, meta)

        # Health report: derive quality metrics from stored training classification_report.
        # Populate gt_cells from actual DB annotations so the cell-excess warning fires.
        _gt_cells    = _count_gt_boxes(db, meta.get("project_id", ""), meta)
        _training_cr: dict = {}
        if model.training_job_id:
            _tj = db.query(TrainingJob).filter(TrainingJob.id == model.training_job_id).first()
            if _tj and _tj.classification_report:
                _training_cr = _tj.classification_report
        health_report = _compute_health_report(
            {}, {**meta, "gt_cells": _gt_cells}, training_cr=_training_cr
        )

        # Upload package
        if resolved_target == "pxe":
            _pkg_ext = ".pxe"
        elif resolved_target in ("tflite", "raspberry_pi", "unoq"):
            _pkg_ext = ".pe"
        else:
            _pkg_ext = ".zip"
        pkg_key = f"projects/{meta.get('project_id','unknown')}/deployments/{deployment_id}/{dep.target}_package{_pkg_ext}"
        content_type = (
            "application/octet-stream"
            if resolved_target in ("tflite", "raspberry_pi", "unoq", "pxe")
            else "application/zip"
        )
        storage.upload_bytes(package_bytes, pkg_key, content_type)

        # For .pxe: serve the package with the EI-style filename
        # ({project}-{impulse}-#{n}.pxe) via Content-Disposition on the
        # presigned URL. Persist the canonical name so the GET endpoint can
        # regenerate the same URL later without re-deriving it.
        download_filename = None
        if resolved_target == "pxe":
            download_filename = build_pxe_download_filename(
                meta.get("project_name") or "",
                str(meta.get("impulse_name") or "impulse"),
                int(meta.get("deploy_version", 1)),
            )
            merged_options = dict(dep.options or {})
            merged_options["download_filename"] = download_filename
            dep.options = merged_options
        download_url = storage.get_presigned_url(
            pkg_key, expires_in=86400, download_filename=download_filename
        )

        dep.status       = JobStatus.completed
        dep.storage_key  = pkg_key
        dep.download_url = download_url
        dep.completed_at = datetime.utcnow()
        db.commit()

        _log_event(
            _deployment_log, "deployment.finished",
            deployment_id=deployment_id, target=dep.target,
            resolved_target=resolved_target, status="completed",
            build_ms=round((time.perf_counter() - _build_t0) * 1000.0, 2),
        )
        return {
            "deployment_id": deployment_id,
            "download_url":  download_url,
            "health_report": health_report,
            **({"golden_test": golden_result} if golden_result else {}),
        }

    except Exception as e:
        logger.exception(f"Deployment {deployment_id} failed: {e}")
        _log_event(
            _deployment_log, "deployment.finished",
            level=logging.ERROR,
            deployment_id=deployment_id,
            target=getattr(dep, "target", None),
            resolved_target=_resolved_target, status="failed", reason=str(e),
            build_ms=round((time.perf_counter() - _build_t0) * 1000.0, 2),
        )
        if dep:
            dep.status        = JobStatus.failed
            dep.error_message = str(e)
            db.commit()
        raise
    finally:
        db.close()


_device_profile_cache: dict | None = None


def _supported_device_profiles() -> dict:
    """Device-profile → {target, display_name}, sourced from the device catalog.

    Target Device Phase 1 (docs/target_device_phase1.md) promotes the catalog
    to the authority on device identity; this is the map that used to be the
    literal `_SUPPORTED_DEVICE_PROFILES` dict. Loaded once per worker process
    and cached — the catalog is read-only in this phase, so there's nothing to
    invalidate the cache for, and resolving a deployment target should not
    cost a DB round trip every time.
    """
    global _device_profile_cache
    if _device_profile_cache is None:
        from app.models.devices import DeviceCatalogEntry

        db = SessionLocal()
        try:
            _device_profile_cache = {
                entry.slug: {
                    "target": entry.deploy_target.value,
                    "display_name": entry.display_name,
                }
                for entry in db.query(DeviceCatalogEntry).all()
            }
        finally:
            db.close()
    return _device_profile_cache


def _resolve_deployment_target(dep) -> str:
    device_profile = getattr(dep, "device_profile", None) or (dep.options or {}).get("device_profile")
    if device_profile:
        profiles = _supported_device_profiles()
        profile = profiles.get(device_profile)
        if not profile:
            raise ValueError(
                f"Unsupported deployment device_profile: '{device_profile}'. "
                f"Supported values: {', '.join(sorted(profiles))}"
            )
        return profile["target"]
    return dep.target


def _get_tflite_bytes(db, model: TrainedModel) -> bytes:
    """Return the best available TFLite bytes for this training job.

    Priority:
      1. int8-quantized TFLite (model_metadata->quantized == true)
      2. float32 TFLite
      3. Whatever storage_key the model record itself points to
    """
    from app.models.user import TrainedModel as TM

    candidates = (
        db.query(TM)
        .filter(TM.training_job_id == model.training_job_id, TM.format == "tflite")
        .all()
    )

    int8_record = None
    f32_record = None
    for c in candidates:
        meta = c.model_metadata or {}
        if meta.get("quantized") is True:
            int8_record = c
        elif f32_record is None:
            f32_record = c

    chosen = int8_record or f32_record
    if chosen:
        return storage.download_bytes(chosen.storage_key)

    # Last resort: the record passed in (e.g. a Keras SavedModel)
    return storage.download_bytes(model.storage_key)


def _get_decoded_tflite_bytes(db, model: TrainedModel) -> bytes:
    """Return the decoded float32 TFLite for YOLO Pro (variant == 'decoded_float32').

    Falls back to the raw float32 TFLite if the decoded variant is not found
    (e.g. artifact was generated before decoded export was added).
    """
    from app.models.user import TrainedModel as TM

    candidates = (
        db.query(TM)
        .filter(TM.training_job_id == model.training_job_id, TM.format == "tflite")
        .all()
    )

    decoded_record = None
    f32_record = None
    for c in candidates:
        meta = c.model_metadata or {}
        if meta.get("variant") == "decoded_float32":
            decoded_record = c
        elif not meta.get("quantized") and f32_record is None:
            f32_record = c

    chosen = decoded_record or f32_record
    if chosen:
        return storage.download_bytes(chosen.storage_key)

    return storage.download_bytes(model.storage_key)


def _get_ssd_decoded_tflite_bytes(db, model: TrainedModel) -> bytes:
    """Return the decoded float32 TFLite for MobileNetV2 SSD (variant ==
    'decoded_float32').

    The SSD trainer (mobilenetv2_ssd_worker) builds this variant at train time
    by wrapping the raw model with MobileNetV2 SSD/decode_export.SSDDecodeLayer,
    producing a single (1, N, 6) [x1,y1,x2,y2,score,class_id] tensor in
    normalized coordinates — the exact contract the runner's decoded handler
    expects. This is a separate path from YOLO-Pro's _get_decoded_tflite_bytes;
    the two only converge at the PXE manifest contract.

    Unlike the YOLO-Pro helper, this does NOT fall back to the raw float32
    TFLite: the raw SSD model emits two undecoded tensors that the decoded
    runner cannot interpret, so a missing decoded variant is a hard error
    (re-train to regenerate it) rather than a silently-broken package.
    """
    from app.models.user import TrainedModel as TM

    candidates = (
        db.query(TM)
        .filter(TM.training_job_id == model.training_job_id, TM.format == "tflite")
        .all()
    )
    for c in candidates:
        if (c.model_metadata or {}).get("variant") == "decoded_float32":
            return storage.download_bytes(c.storage_key)

    raise ValueError(
        "MobileNetV2 SSD PXE deployment requires the 'decoded_float32' TFLite "
        "variant, which was not found for this training job. Re-train the model "
        "to regenerate it (decoded export was added with PXE support)."
    )


def _validate_tflite_bytes(model_bytes: bytes) -> None:
    """Validate TFLite model bytes before packaging.  Raises ValueError on any
    of the following conditions so that the deployment job fails early with a
    clear error rather than silently producing a broken package.

    Checks (in order):
      1. model_bytes is non-empty and at least _TFLITE_MIN_BYTES in size.
      2. The FlatBuffers file identifier at bytes[4:8] equals b"TFL3".
      3. tflite_runtime.Interpreter (or tf.lite.Interpreter) can load and
         allocate tensors for the bytes without raising an exception.
    """
    # ── 1. Size / truncation check ───────────────────────────────────────────
    if not model_bytes:
        raise ValueError(
            "TFLite model_bytes is empty; the model was not exported or the "
            "storage download returned no data."
        )
    if len(model_bytes) < _TFLITE_MIN_BYTES:
        raise ValueError(
            f"TFLite model_bytes is only {len(model_bytes)} B, below the minimum "
            f"plausible size of {_TFLITE_MIN_BYTES} B.  The model slice appears "
            f"truncated or corrupt."
        )

    # ── 2. FlatBuffers / TFLite file-identifier check ────────────────────────
    id_start = _TFLITE_FILE_IDENTIFIER_OFFSET
    id_end   = id_start + len(_TFLITE_FILE_IDENTIFIER)
    if len(model_bytes) < id_end:
        raise ValueError(
            f"TFLite model_bytes ({len(model_bytes)} B) is too short to contain "
            f"the FlatBuffers file identifier at bytes[{id_start}:{id_end}]."
        )
    file_id = model_bytes[id_start:id_end]
    if file_id != _TFLITE_FILE_IDENTIFIER:
        raise ValueError(
            f"TFLite flatbuffer file-identifier mismatch: expected "
            f"{_TFLITE_FILE_IDENTIFIER!r} at bytes[{id_start}:{id_end}], "
            f"got {file_id!r}.  The bytes are not a valid TFLite flatbuffer."
        )

    # ── 3. Interpreter load test ─────────────────────────────────────────────
    try:
        _interp = _make_tflite_interpreter(model_content=model_bytes)
        _interp.allocate_tensors()
    except Exception as exc:
        raise ValueError(
            f"TFLite Interpreter could not load model_bytes "
            f"({len(model_bytes)} B): {exc}"
        ) from exc


def _align_image_input_contract_with_tflite(meta: dict, model_bytes: bytes) -> dict:
    """Patch legacy image-model metadata to match the actual TFLite input contract."""
    input_shape = meta.get("input_shape") or []
    if len(input_shape) < 3:
        return meta

    try:
        interp = _make_tflite_interpreter(model_content=model_bytes)
        interp.allocate_tensors()
        inp_detail = interp.get_input_details()[0]
    except Exception:
        return meta

    if inp_detail.get("dtype") != np.int8:
        return meta

    scale, zero_point = inp_detail.get("quantization", (0.0, 0))
    if scale in (0, 0.0):
        return meta

    patched = dict(meta)

    # FOMO exports that strip preprocessing quantize around [-1, 1].
    if abs(float(scale) - (1.0 / 127.5)) < 1e-4 and int(zero_point) in (-1, 0):
        patched["normalize_input"] = True
        patched["input_mean"] = 127.5
        patched["input_std"] = 127.5

    return patched


def _validate_manifest_meta(meta: dict, source: str) -> None:
    """Strict type and content validation of the metadata dict that feeds
    manifest.json and dsp_config.json.  Raises ValueError on the first
    problem found so the error message is unambiguous.

    Enforces:
      - All required fields are present.
      - Each field has the correct Python type (version must be int, not str/float/bool).
      - label_names: non-empty list of non-blank strings.
      - input_shape: non-empty list of positive ints.
      - channel_order: one of "rgb" / "bgr" (case-insensitive).
      - model_type: non-blank string.
    """
    for field, expected_type in _REQUIRED_META_FIELD_TYPES:
        if field not in meta:
            raise ValueError(
                f"Metadata from {source!r} is missing required field {field!r}. "
                "Rebuild the .pe package."
            )
        val = meta[field]
        # bool is a subclass of int in Python — reject it for integer fields
        # so accidental JSON booleans don't silently pass the int check.
        if expected_type is int and isinstance(val, bool):
            raise ValueError(
                f"Metadata from {source!r}: field {field!r} must be int, "
                f"got bool {val!r}.  Rebuild the .pe package."
            )
        if not isinstance(val, expected_type):
            raise ValueError(
                f"Metadata from {source!r}: field {field!r} must be "
                f"{expected_type.__name__}, got {type(val).__name__} ({val!r}).  "
                "Rebuild the .pe package."
            )

    # label_names: non-empty list of non-blank strings
    label_names = meta["label_names"]
    if len(label_names) == 0:
        raise ValueError(
            f"Metadata from {source!r}: label_names is an empty list; "
            "at least one class label is required."
        )
    bad_labels = [
        i for i, ln in enumerate(label_names)
        if not isinstance(ln, str) or not ln.strip()
    ]
    if bad_labels:
        raise ValueError(
            f"Metadata from {source!r}: label_names contains non-string or blank "
            f"entries at indices {bad_labels}: {[label_names[i] for i in bad_labels]!r}."
        )

    # input_shape: non-empty list of positive ints (bools rejected)
    input_shape = meta["input_shape"]
    if len(input_shape) == 0:
        raise ValueError(
            f"Metadata from {source!r}: input_shape is an empty list."
        )
    bad_dims = [
        i for i, d in enumerate(input_shape)
        if not isinstance(d, int) or isinstance(d, bool) or d <= 0
    ]
    if bad_dims:
        raise ValueError(
            f"Metadata from {source!r}: input_shape has invalid dimension(s) at "
            f"indices {bad_dims}: {input_shape!r}.  All dimensions must be positive ints."
        )

    # channel_order: must be "rgb" or "bgr" (stored lowered)
    channel_order = meta["channel_order"].lower()
    if channel_order not in _VALID_CHANNEL_ORDERS:
        raise ValueError(
            f"Metadata from {source!r}: channel_order={meta['channel_order']!r} is invalid; "
            f"must be one of {sorted(_VALID_CHANNEL_ORDERS)}."
        )

    # model_type: non-blank string
    if not meta["model_type"].strip():
        raise ValueError(
            f"Metadata from {source!r}: model_type is blank; a non-empty string is required."
        )


def _verify_pe_container(pe_bytes: bytes) -> None:
    """Parse the PEM1 header and verify every declared section size is
    consistent with the actual serialized payload.  Raises ValueError on:

      - Truncated or wrong-magic header.
      - Any section declared as 0 bytes.
      - Total declared length != len(pe_bytes).
      - Any reconstructed section slice shorter than its declared size.
    """
    hdr_size = _PE_HEADER_STRUCT.size
    if len(pe_bytes) < hdr_size:
        raise ValueError(
            f".pe package is too short to hold a valid header: "
            f"{len(pe_bytes)} B < {hdr_size} B."
        )
    magic, fmt_ver, model_sz, dsp_sz, labels_sz, manifest_sz, inference_sz = (
        _PE_HEADER_STRUCT.unpack_from(pe_bytes, 0)
    )
    if magic != _PE_MAGIC:
        raise ValueError(
            f".pe package header magic mismatch: expected {_PE_MAGIC!r}, got {magic!r}."
        )
    if fmt_ver != _PE_FORMAT_VERSION:
        raise ValueError(
            f".pe package format version mismatch: expected {_PE_FORMAT_VERSION}, "
            f"got {fmt_ver}."
        )

    section_sizes = (
        ("model",      model_sz),
        ("dsp_config", dsp_sz),
        ("labels",     labels_sz),
        ("manifest",   manifest_sz),
        # inference section is optional: size==0 is valid for non-scripted targets
    )
    empty_sections = [name for name, sz in section_sizes if sz == 0]
    if empty_sections:
        raise ValueError(
            f".pe package header declares empty section(s): {empty_sections}. "
            "Rebuild the package."
        )

    expected_total = hdr_size + model_sz + dsp_sz + labels_sz + manifest_sz + inference_sz
    if len(pe_bytes) != expected_total:
        raise ValueError(
            f".pe package size mismatch: header declares {expected_total} B total "
            f"(header={hdr_size} + model={model_sz} + dsp={dsp_sz} + "
            f"labels={labels_sz} + manifest={manifest_sz} + inference={inference_sz}), "
            f"actual={len(pe_bytes)} B."
        )

    # Walk each section and verify the slice is exactly the declared size.
    offset = hdr_size
    for name, sz in section_sizes:
        actual_len = len(pe_bytes[offset: offset + sz])
        if actual_len != sz:
            raise ValueError(
                f".pe package section {name!r}: declared {sz} B but only "
                f"{actual_len} B available at offset {offset}."
            )
        offset += sz


def _verify_pxe_container(data: bytes) -> None:
    """Verify a v2 PXE binary: header integrity, size consistency, and cross-section invariants.

    Only called on freshly-built packages (always v2).  Old v1 packages are never
    passed here; they are validated at runtime by the loader/validator instead.
    """
    hdr = _PXE_HEADER_STRUCT_V2.size
    if len(data) < hdr:
        raise ValueError(f".pxe too small: {len(data)} bytes")
    magic, ver, runner_sz, model_sz, manifest_sz, labels_sz, dsp_sz, postprocess_sz, flags = \
        _PXE_HEADER_STRUCT_V2.unpack_from(data)
    if magic != _PXE_MAGIC:
        raise ValueError(f"Bad .pxe magic {magic!r}")
    if ver != _PXE_FORMAT_VERSION_V2:
        raise ValueError(f"Unexpected .pxe version {ver} in freshly-built package (expected {_PXE_FORMAT_VERSION_V2})")

    # All required sections must declare non-zero bytes.  postprocess_sz == 0 is
    # explicitly allowed (reserved for v1-compat; currently always > 0 in built packages).
    # Check before total-size so the error is specific ("empty section") not generic ("size mismatch").
    empty_required = [
        name for name, sz in (
            ("runner",     runner_sz),
            ("model",      model_sz),
            ("manifest",   manifest_sz),
            ("labels",     labels_sz),
            ("dsp_config", dsp_sz),
        ) if sz == 0
    ]
    if empty_required:
        raise ValueError(
            f".pxe header declares empty required section(s): {empty_required}. "
            "Rebuild the package."
        )

    expected = hdr + runner_sz + model_sz + manifest_sz + labels_sz + dsp_sz + postprocess_sz
    if len(data) != expected:
        raise ValueError(
            f".pxe size mismatch: header declares {expected} bytes, got {len(data)}"
        )

    # Cross-section consistency: manifest.version == dsp_config.version,
    # labels.txt lines == manifest.label_names.
    offset = hdr + runner_sz + model_sz
    manifest_bytes = data[offset : offset + manifest_sz]
    offset += manifest_sz
    labels_bytes   = data[offset : offset + labels_sz]
    offset += labels_sz
    dsp_bytes      = data[offset : offset + dsp_sz]
    offset += dsp_sz
    postprocess_bytes = data[offset : offset + postprocess_sz] if postprocess_sz else None

    try:
        manifest   = json.loads(manifest_bytes)
        dsp_config = json.loads(dsp_bytes)
        labels_txt = [l for l in labels_bytes.decode().splitlines() if l]
    except Exception as exc:
        raise ValueError(f".pxe section parse error: {exc}") from exc

    m_ver = manifest.get("version")
    d_ver = dsp_config.get("version")
    if m_ver != d_ver:
        raise ValueError(
            f".pxe version mismatch: manifest.version={m_ver!r} vs dsp_config.version={d_ver!r}"
        )

    m_labels = manifest.get("label_names", [])
    if labels_txt != m_labels:
        raise ValueError(
            f".pxe labels mismatch: labels.txt {labels_txt!r} != manifest.label_names {m_labels!r}"
        )

    if postprocess_bytes:
        try:
            pp_cfg = json.loads(postprocess_bytes)
        except Exception as exc:
            raise ValueError(f".pxe postprocess_config.json parse error: {exc}") from exc
        try:
            _validate_postprocess_config_dict(pp_cfg)
        except ValueError as exc:
            raise ValueError(f".pxe postprocess_config validation failed: {exc}") from exc


def _smoke_test_inference(model_bytes: bytes, meta: dict) -> None:
    """Run one forward pass with a zero-filled dummy input to confirm the
    model executes without error inside the deployment environment.

    Raises ValueError if:
      - Interpreter cannot load or allocate tensors.
      - No input or output tensor details are returned.
      - The forward pass (invoke) raises any exception.
      - Any output tensor has zero elements (degenerate output shape).
    """
    try:
        interp = _make_tflite_interpreter(model_content=model_bytes)
        interp.allocate_tensors()

        input_details  = interp.get_input_details()
        output_details = interp.get_output_details()

        if not input_details:
            raise ValueError(
                "Interpreter returned no input tensors after allocate_tensors()."
            )
        if not output_details:
            raise ValueError(
                "Interpreter returned no output tensors after allocate_tensors()."
            )

        # Build a correctly shaped zero tensor, respecting quantised dtypes.
        inp_detail = input_details[0]
        inp_dtype  = inp_detail["dtype"]
        inp_shape  = tuple(inp_detail["shape"])

        dummy = np.zeros(inp_shape, dtype=np.float32)
        if inp_dtype == np.int8:
            scale, zero_point = inp_detail.get("quantization", (1.0, 0))
            if scale == 0.0:
                scale = 1.0
            dummy = np.clip(
                np.round(dummy / scale + zero_point), -128, 127
            ).astype(np.int8)
        else:
            dummy = dummy.astype(inp_dtype)

        interp.set_tensor(inp_detail["index"], dummy)
        interp.invoke()

        for od in output_details:
            out = interp.get_tensor(od["index"])
            if out.size == 0:
                name = od.get("name", str(od["index"]))
                raise ValueError(
                    f"Output tensor {name!r} has zero elements after inference; "
                    "model output shape is degenerate."
                )

    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(
            f"Inference smoke test failed for model_bytes ({len(model_bytes)} B): {exc}"
        ) from exc


def _run_model_unified(tflite_bytes: bytes, meta: dict, input_np: np.ndarray | None = None) -> dict:
    """Single run_model() contract for all targets: FOMO, YOLO-Pro, SSD, classification.

    All preprocessing driven by meta keys: normalize_input, channel_order, threshold,
    input_mean, input_std, reg_max, num_classes (all read from manifest).

    Returns {"detections": list, "count": int, "debug": {"raw_min", "raw_max", "raw_mean"}}.
    Classification wraps the top prediction as a single pseudo-detection with full-image bbox.
    """
    interp = _make_tflite_interpreter(model_content=tflite_bytes)
    interp.allocate_tensors()
    inp_detail = interp.get_input_details()[0]
    out_details = interp.get_output_details()

    if input_np is None:
        input_np = np.zeros(tuple(map(int, inp_detail["shape"])), dtype=np.float32)

    inp = input_np.flatten().reshape(inp_detail["shape"]).astype(np.float32)

    channel_order = meta.get("channel_order", "rgb").lower()
    normalize     = meta.get("normalize_input", False)
    threshold     = float(meta.get("threshold", 0.5))
    input_mean    = float(meta.get("input_mean", 0.0))
    input_std     = float(meta.get("input_std", 255.0))
    label_names   = meta.get("label_names", [])
    output_type   = meta.get("output_type") or meta.get("model_type", "classification")
    num_classes   = int(meta.get("num_classes", len(label_names))) or 1
    reg_max       = int(meta.get("reg_max", 16))
    input_shape   = meta.get("input_shape", [224, 224, 3])

    if channel_order == "bgr" and inp.ndim >= 3 and inp.shape[-1] == 3:
        inp = inp[..., ::-1].copy()

    if normalize:
        denom = input_std if input_std != 0.0 else 255.0
        inp = (inp - input_mean) / denom

    scale, zp = inp_detail.get("quantization", (0.0, 0))
    if inp_detail["dtype"] == np.int8 and scale not in (0, 0.0):
        inp = np.clip(np.round(inp / scale + zp), -128, 127).astype(np.int8)

    interp.set_tensor(inp_detail["index"], inp)
    interp.invoke()

    raw_outputs = []
    for od in out_details:
        raw = interp.get_tensor(od["index"]).astype(np.float32)
        s, z = od.get("quantization", (0.0, 0))
        if s not in (0, 0.0):
            raw = (raw - z) * s
        raw_outputs.append(raw)

    all_raw = np.concatenate([r.flatten() for r in raw_outputs]) if raw_outputs else np.array([0.0])
    debug = {
        "raw_min":  float(all_raw.min()),
        "raw_max":  float(all_raw.max()),
        "raw_mean": float(all_raw.mean()),
    }

    # ── FOMO heatmap ──────────────────────────────────────────────────────────
    if output_type == "detection_heatmap" or "fomo" in meta.get("architecture", "").lower():
        raw4d = raw_outputs[0]
        if raw4d.ndim == 4:
            _, gh, gw, n_all = raw4d.shape
        else:
            n_all = num_classes + 1
            total = raw4d.size
            side  = max(1, int(round((total / n_all) ** 0.5)))
            gh = gw = side
        cell_scores   = raw4d.flatten().reshape(gh, gw, n_all)
        object_scores = cell_scores[:, :, 1:]
        detections = []
        for row in range(gh):
            for col in range(gw):
                cls_idx = int(np.argmax(object_scores[row, col]))
                conf    = float(object_scores[row, col, cls_idx])
                if conf < threshold:
                    continue
                cx, cy = (col + 0.5) / gw, (row + 0.5) / gh
                hw, hh = 0.5 / gw, 0.5 / gh
                detections.append({
                    "label":      label_names[cls_idx] if cls_idx < len(label_names) else str(cls_idx),
                    "confidence": conf,
                    "bbox":       {"x1": max(0.0, cx - hw), "y1": max(0.0, cy - hh),
                                   "x2": min(1.0, cx + hw), "y2": min(1.0, cy + hh)},
                })
        return {"detections": detections, "count": len(detections), "debug": debug}

    # ── YOLO-Pro multi-scale ──────────────────────────────────────────────────
    if output_type == "yolo_pro_detection":
        named      = {od.get("name", f"out_{i}"): raw_outputs[i] for i, od in enumerate(out_details)}
        input_h    = int(input_shape[0])
        input_w    = int(input_shape[1])
        strides    = [8, 16, 32]
        detections = []
        def _pick_head(want_size, want_dims):
            """Resolve one head by element count, with SHAPE breaking ties.

            Size alone cross-wires heads: with strides (8, 16, 32) the p3 grid
            is 4x the p5 grid, so
                size(cls_p3) = 16 * gh5 * gw5 * num_classes
                size(reg_p5) =      gh5 * gw5 * 4 * reg_max
            are equal whenever ``num_classes == reg_max / 4`` — every 4-class
            model at the default reg_max=16.  The reshape below then silently
            reinterprets the p5 DFL tensor as p3 class scores (p3 alone carries
            ~76% of all anchors), so the decode returns confident nonsense.
            """
            cands = [k for k, v in named.items() if v.size == want_size]
            if not cands:
                return None
            for k in cands:
                if tuple(named[k].shape)[-3:] == want_dims:
                    return k
            return cands[0]

        for level, stride in enumerate(strides, start=3):
            gh, gw  = input_h // stride, input_w // stride
            exp_cls = gh * gw * num_classes
            exp_reg = gh * gw * 4 * reg_max
            cls_key = next((k for k in named if f"p{level}" in k.lower() and "cls" in k.lower()), None)
            reg_key = next((k for k in named if f"p{level}" in k.lower() and "reg" in k.lower()), None)
            if cls_key is None:
                cls_key = _pick_head(exp_cls, (gh, gw, num_classes))
            if reg_key is None:
                reg_key = _pick_head(exp_reg, (gh, gw, 4 * reg_max))
            # cls_key == reg_key means the two heads are indistinguishable by
            # name, size AND shape (num_classes == 4 * reg_max, e.g. 64 classes
            # at reg_max=16).  Skipping the level beats popping the same tensor
            # twice (KeyError) or decoding DFL logits as class scores.
            if cls_key is None or reg_key is None or cls_key == reg_key:
                continue
            # Retire both tensors so a later level cannot reuse them.
            cls_out  = named.pop(cls_key).flatten().reshape(gh, gw, num_classes)
            reg_out  = named.pop(reg_key).flatten().reshape(gh, gw, 4 * reg_max)
            # NO sigmoid: the cls head already ends in one inside the model
            # (yolo_pro/head.py — Activation("sigmoid")), so every export emits
            # probabilities.  Applying it twice maps [0,1] onto [0.5, 0.73],
            # putting every anchor above a 0.25/0.35 runtime threshold.
            cls_prob = cls_out
            for y in range(gh):
                for x in range(gw):
                    cls_idx = int(np.argmax(cls_prob[y, x]))
                    score   = float(cls_prob[y, x, cls_idx])
                    if score < threshold:
                        continue
                    reg  = reg_out[y, x].reshape(4, reg_max)
                    prob = np.exp(reg - reg.max(axis=1, keepdims=True))
                    prob /= prob.sum(axis=1, keepdims=True)
                    bins = np.arange(reg_max, dtype=np.float32)
                    # DFL distances come out in GRID units — they must be scaled by stride to reach pixels, exactly as yolo_pro/decode.py::decode_raw_outputs_np
                    # does (ltrb_px = ltrb_grid * stride).  Without the scale the
                    # boxes come out `stride` times too small — collapsed to
                    # near-points that NMS cannot merge, so one object became
                    # dozens of overlapping slivers.
                    l, t, r, b = (prob * bins[None, :]).sum(axis=1) * float(stride)
                    cx, cy = (x + 0.5) * stride, (y + 0.5) * stride
                    detections.append({
                        "label":      label_names[cls_idx] if cls_idx < len(label_names) else str(cls_idx),
                        "confidence": score,
                        "bbox":       {"x1": float(max(0.0, (cx - l) / input_w)),
                                       "y1": float(max(0.0, (cy - t) / input_h)),
                                       "x2": float(min(1.0, (cx + r) / input_w)),
                                       "y2": float(min(1.0, (cy + b) / input_h))},
                    })
        if detections:
            _boxes  = np.array([[d["bbox"]["x1"], d["bbox"]["y1"], d["bbox"]["x2"], d["bbox"]["y2"]]
                                 for d in detections], dtype=np.float32)
            _sc     = np.array([d["confidence"] for d in detections], dtype=np.float32)
            order   = np.argsort(_sc)[::-1]; keep = []
            while len(order):
                i = int(order[0]); keep.append(i)
                if len(order) == 1: break
                ix1 = np.maximum(_boxes[i, 0], _boxes[order[1:], 0])
                iy1 = np.maximum(_boxes[i, 1], _boxes[order[1:], 1])
                ix2 = np.minimum(_boxes[i, 2], _boxes[order[1:], 2])
                iy2 = np.minimum(_boxes[i, 3], _boxes[order[1:], 3])
                inter = np.maximum(0.0, ix2 - ix1) * np.maximum(0.0, iy2 - iy1)
                ai    = (_boxes[i, 2] - _boxes[i, 0]) * (_boxes[i, 3] - _boxes[i, 1])
                aj    = (_boxes[order[1:], 2] - _boxes[order[1:], 0]) * (_boxes[order[1:], 3] - _boxes[order[1:], 1])
                order = order[1:][inter / (ai + aj - inter + 1e-6) < 0.45]
            detections = [detections[i] for i in keep]
        return {"detections": detections, "count": len(detections), "debug": debug}

    # ── SSD ───────────────────────────────────────────────────────────────────
    if output_type in ("ssd_detection", "object_detection"):
        named = {od.get("name", f"out_{i}"): raw_outputs[i] for i, od in enumerate(out_details)}
        boxes = classes_arr = scores_arr = count = None
        for name, out in named.items():
            lname = name.lower()
            flat  = out.flatten()
            if boxes is None and ("box" in lname or (out.ndim >= 2 and out.shape[-1] == 4)):
                boxes = out.reshape(-1, 4)
            elif classes_arr is None and "class" in lname:
                classes_arr = flat.astype(np.int32)
            elif scores_arr is None and "score" in lname:
                scores_arr = flat.astype(np.float32)
            elif count is None and ("count" in lname or flat.size == 1):
                count = int(flat[0])
        if boxes is None or scores_arr is None:
            for out in raw_outputs:
                flat = out.flatten()
                if out.ndim >= 2 and out.shape[-1] == 4 and boxes is None:
                    boxes = out.reshape(-1, 4)
                elif classes_arr is None and np.all(np.equal(np.mod(flat[:min(len(flat), 16)], 1), 0)):
                    classes_arr = flat.astype(np.int32)
                elif count is None and flat.size == 1:
                    count = int(flat[0])
                elif scores_arr is None:
                    scores_arr = flat.astype(np.float32)
        if boxes is None or scores_arr is None:
            return {"detections": [], "count": 0, "debug": debug}
        if classes_arr is None:
            classes_arr = np.zeros(boxes.shape[0], dtype=np.int32)
        if count is None:
            count = min(len(boxes), len(scores_arr), len(classes_arr))
        detections = []
        for i in range(min(count, len(boxes), len(scores_arr), len(classes_arr))):
            score = float(scores_arr[i])
            if score < threshold:
                continue
            cls = int(classes_arr[i])
            y1, x1, y2, x2 = boxes[i].tolist()
            detections.append({
                "label":      label_names[cls] if 0 <= cls < len(label_names) else str(cls),
                "confidence": score,
                "bbox":       {"x1": float(max(0.0, min(1.0, x1))), "y1": float(max(0.0, min(1.0, y1))),
                               "x2": float(max(0.0, min(1.0, x2))), "y2": float(max(0.0, min(1.0, y2)))},
            })
        return {"detections": detections, "count": len(detections), "debug": debug}

    # ── Classification ────────────────────────────────────────────────────────
    flat = raw_outputs[0].flatten()
    e    = np.exp(flat - flat.max())
    prob = e / e.sum()
    best = int(np.argmax(prob))
    return {
        "detections": [{
            "label":      label_names[best] if best < len(label_names) else str(best),
            "confidence": float(prob[best]),
            "bbox":       {"x1": 0.0, "y1": 0.0, "x2": 1.0, "y2": 1.0},
        }],
        "count": 1,
        "debug": debug,
    }


def _post_build_pe_validation(pe_bytes: bytes, meta: dict) -> None:
    """Extract the TFLite model section from the .pe container and run 1 forward pass.

    Validates that the bytes actually packaged (not the pre-package source bytes)
    produce a well-formed 'detections' list.  Fails deployment on malformed output.
    """
    hdr_size = _PE_HEADER_STRUCT.size
    _, _, model_sz, *_ = _PE_HEADER_STRUCT.unpack_from(pe_bytes, 0)
    packaged_tflite = pe_bytes[hdr_size : hdr_size + model_sz]
    try:
        result = _run_model_unified(packaged_tflite, meta)
    except Exception as exc:
        raise ValueError(f"Post-build .pe validation inference failed: {exc}") from exc

    if not isinstance(result.get("detections"), list):
        raise ValueError(
            f"Post-build .pe validation: expected 'detections' to be a list, "
            f"got {type(result.get('detections')).__name__!r}. "
            "Model output format is invalid."
        )
    logger.info(
        "Post-build .pe validation passed: detections=%d debug=%s",
        len(result["detections"]), result.get("debug"),
    )


def _compute_detection_ious(predicted: list, expected: list) -> list:
    """Return max-IoU of each expected detection against all predicted boxes."""
    if not expected or not predicted:
        return []

    def _box_iou(a: dict, b: dict) -> float:
        ix1 = max(a["x1"], b["x1"]); iy1 = max(a["y1"], b["y1"])
        ix2 = min(a["x2"], b["x2"]); iy2 = min(a["y2"], b["y2"])
        inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        ua    = (a["x2"] - a["x1"]) * (a["y2"] - a["y1"]) + (b["x2"] - b["x1"]) * (b["y2"] - b["y1"]) - inter
        return inter / (ua + 1e-6)

    pred_boxes = [d.get("bbox", {}) for d in predicted]
    ious = []
    for exp_det in expected:
        exp_box = exp_det.get("bbox", {})
        if not exp_box:
            continue
        ious.append(max((_box_iou(exp_box, pb) for pb in pred_boxes if pb), default=0.0))
    return ious


def _golden_test_validation(tflite_bytes: bytes, meta: dict) -> dict | None:
    """Run golden test if meta contains golden_test.{input_flat, expected_detections}.

    Validates iou(pred, expected) > 0.5 (mean across expected detections).
    Returns {"iou_mean", "passed", "predicted", "expected"} or None if no golden data.
    """
    golden = meta.get("golden_test")
    if not golden:
        return None
    input_flat = golden.get("input_flat")
    expected   = golden.get("expected_detections", [])
    if input_flat is None:
        return None

    input_np = np.array(input_flat, dtype=np.float32)
    try:
        result = _run_model_unified(tflite_bytes, meta, input_np=input_np)
    except Exception as exc:
        raise ValueError(f"Golden test inference failed: {exc}") from exc

    predicted  = result.get("detections", [])
    iou_scores = _compute_detection_ious(predicted, expected)
    iou_mean   = float(np.mean(iou_scores)) if iou_scores else 0.0
    passed     = iou_mean > 0.5 if expected else True

    if not passed:
        raise ValueError(
            f"Golden test failed: mean_iou={iou_mean:.3f} < 0.5 on {len(expected)} expected detections. "
            f"Predicted {len(predicted)}. Model output has regressed."
        )
    logger.info(
        "Golden test passed: mean_iou=%.3f predicted=%d expected=%d",
        iou_mean, len(predicted), len(expected),
    )
    return {"iou_mean": iou_mean, "passed": passed, "predicted": len(predicted), "expected": len(expected)}


def _count_gt_boxes(db, project_id: str, meta: dict) -> int:
    """Return the total number of GT bounding-box annotations for the project.

    For detection models (FOMO, YOLO-Pro, SSD): sums len(boundingBoxes) across
    every sample in the project.  For classification: counts labeled samples.
    Returns 0 if project_id is empty or no annotated data is found.
    """
    if not project_id:
        return 0

    output_type = meta.get("output_type") or meta.get("model_type", "classification")
    is_detection = (
        output_type in ("detection_heatmap", "yolo_pro_detection", "ssd_detection", "object_detection")
        or "fomo" in meta.get("architecture", "").lower()
    )

    samples = db.query(Sample).filter(Sample.project_id == project_id).all()

    if is_detection:
        total = 0
        for s in samples:
            boxes = (s.extra_metadata or {}).get("boundingBoxes") or []
            total += len(boxes) if isinstance(boxes, list) else 0
        return total

    # Classification: count samples that have a label assigned
    return sum(1 for s in samples if s.label_id is not None)


def _compute_health_report(
    result: dict, meta: dict, training_cr: dict | None = None
) -> dict:
    """Compute deployment health report from stored training metrics.

    `result` may be empty ({}); when non-empty its detections override the
    training-derived predicted_cells so a live-inference caller still works.

    Warns if macro_f1 < f1_threshold or predicted_cells >> gt_cells.
    Returns {"model_quality", "avg_confidence", "predicted_cells", "gt_cells", "warning"}.
    """
    gt_cells      = int(meta.get("gt_cells", 0))
    f1_threshold  = float(meta.get("f1_threshold", 0.3))
    model_quality = "unknown"
    macro_f1: float | None = None
    predicted_cells = 0
    avg_confidence  = 0.0
    is_detection_eval = False

    if training_cr:
        macro_avg = training_cr.get("macro avg") or {}
        _f1 = macro_avg.get("f1-score")
        if _f1 is not None:
            try:
                macro_f1 = float(_f1)
                if macro_f1 >= 0.8:
                    model_quality = "good"
                elif macro_f1 >= 0.5:
                    model_quality = "fair"
                else:
                    model_quality = "poor"
            except (TypeError, ValueError):
                pass

        map50 = training_cr.get("map50")
        if map50 is not None:
            is_detection_eval = True
            try:
                map50 = float(map50)
                if map50 >= 0.8:
                    model_quality = "good"
                elif map50 >= 0.5:
                    model_quality = "fair"
                else:
                    model_quality = "poor"
            except (TypeError, ValueError):
                pass

        stored_predicted = training_cr.get("total_predicted_cells")
        if stored_predicted is None:
            stored_predicted = (training_cr.get("detailed_metrics") or {}).get("total_predicted_cells")
        if stored_predicted is not None:
            try:
                predicted_cells = int(stored_predicted)
            except (TypeError, ValueError):
                pass

        stored_avg_conf = training_cr.get("avg_confidence")
        if stored_avg_conf is None:
            stored_avg_conf = (training_cr.get("detailed_metrics") or {}).get("avg_confidence")
        if stored_avg_conf is not None:
            try:
                avg_confidence = float(stored_avg_conf)
            except (TypeError, ValueError):
                pass

        # FOMO: source predicted_cells from the training sweep entry at the stored threshold.
        if training_cr.get("fomo_eval_status") == "success":
            sweep          = training_cr.get("threshold_sweep") or []
            stored_thresh  = training_cr.get("threshold")
            if sweep and stored_thresh is not None:
                match = next(
                    (s for s in sweep if abs(s.get("threshold", -1) - stored_thresh) < 0.001),
                    None,
                )
                if match:
                    predicted_cells = int(match.get("total_predicted_cells", 0))

    # Live-inference detections (e.g. from golden test caller) take precedence.
    detections = result.get("detections", [])
    if detections:
        predicted_cells = len(detections)
        avg_confidence  = float(np.mean([d["confidence"] for d in detections]))

    warnings = []
    if macro_f1 is not None and macro_f1 < f1_threshold:
        warnings.append(f"macro_f1={macro_f1:.3f} < f1_threshold={f1_threshold:.3f}")
    if is_detection_eval and training_cr:
        try:
            _map50 = float(training_cr.get("map50"))
            if _map50 < f1_threshold:
                warnings.append(f"map50={_map50:.3f} < quality_threshold={f1_threshold:.3f}")
        except (TypeError, ValueError):
            pass
    if gt_cells > 0 and predicted_cells > gt_cells * 3:
        warnings.append(f"predicted_cells={predicted_cells} >> gt_cells={gt_cells} (3× excess)")

    warning = "; ".join(warnings) if warnings else None
    if warning:
        logger.warning("Deployment health check: %s", warning)

    return {
        "model_quality":   model_quality,
        "avg_confidence":  avg_confidence,
        "predicted_cells": predicted_cells,
        "gt_cells":        gt_cells,
        "warning":         warning,
    }


def _derive_normalize_input(meta: dict) -> bool:
    """Derive a normalize_input value from DSP block config when absent from model_metadata.

    Logic (in order):
      1. If the first DSP block's params contain normalize=True, return True.
      2. If the DSP block type is "image" and params contain a rescaling/scale
         key that implies [0,1] normalisation, return True.
      3. Otherwise return False — safe default meaning no input rescaling step.

    This is only called when normalize_input is genuinely missing; the result is
    consistent with what the actual preprocessing pipeline would have done.
    """
    dsp_blocks = meta.get("dsp_blocks") or []
    if not dsp_blocks:
        return False
    first_block = dsp_blocks[0] or {}
    params = first_block.get("params") or {}
    # Explicit normalize flag set by the training pipeline
    if params.get("normalize", False):
        return True
    # Rescaling layer present (e.g. rescaling_scale set to 1/255)
    rescaling_scale = params.get("rescaling_scale") or params.get("scale")
    if rescaling_scale is not None:
        try:
            return float(rescaling_scale) < 1.0
        except (TypeError, ValueError):
            pass
    return False


def _derive_channel_order(meta: dict) -> str:
    """Infer channel_order from input_shape when the field is absent.

    Logic:
      - Read the last dimension of input_shape (the channels axis for image models).
      - 1 channel  → "grayscale"
      - 3 channels → "rgb"  (the DSP processor always emits RGB; BGR is never used
                             by the current training pipeline)
      - anything else → "rgb" as a safe fallback (matches the hardcoded value in
                         all training workers)
    """
    input_shape = meta.get("input_shape") or []
    channels = int(input_shape[-1]) if input_shape else 3
    if channels == 1:
        return "grayscale"
    return "rgb"


def _ensure_required_meta_defaults(meta: dict) -> dict:
    """Return a copy of *meta* with any derivable missing required fields filled in.

    Patches:
      - ``normalize_input``: FOMO models historically omit this field; derived
        from DSP block config via :func:`_derive_normalize_input`.
      - ``channel_order``: older SSD / pre-channel-order model records omit this
        field; derived from the last dimension of ``input_shape`` via
        :func:`_derive_channel_order` (1 ch → "grayscale", ≥2 ch → "rgb").
      - ``model_type``: older YOLO-Pro / pre-model_type records store
        ``output_type`` but not ``model_type``; derived directly from
        ``output_type`` (they are identical in all training workers).

    All other required fields (label_names, input_shape, version) must come
    from training; their absence is a real error and is left for
    _validate_manifest_meta to report.

    Returns the original dict unchanged if no fields need patching (the common
    path), so there is zero overhead for well-formed metadata.
    """
    needs_patch = (
        "normalize_input" not in meta
        or "channel_order" not in meta
        or "model_type" not in meta
    )
    if not needs_patch:
        return meta

    patched = dict(meta)

    if "normalize_input" not in patched:
        patched["normalize_input"] = _derive_normalize_input(patched)
        logger.debug(
            "normalize_input was absent from model_metadata; derived value=%s "
            "from DSP block config and injected before validation.",
            patched["normalize_input"],
        )

    if "channel_order" not in patched:
        patched["channel_order"] = _derive_channel_order(patched)
        logger.debug(
            "channel_order was absent from model_metadata; derived value=%r "
            "from input_shape=%s and injected before validation.",
            patched["channel_order"],
            patched.get("input_shape"),
        )

    if "model_type" not in patched:
        patched["model_type"] = patched.get("output_type") or "classification"
        logger.debug(
            "model_type was absent from model_metadata; derived value=%r "
            "from output_type=%r and injected before validation.",
            patched["model_type"],
            patched.get("output_type"),
        )

    return patched


def _enrich_package_metadata(
    db,
    dep: Deployment,
    model: TrainedModel,
    meta: dict,
    tflite_bytes: bytes,
) -> dict:
    """Fill deployment/package provenance fields from the DB when available."""
    patched = dict(meta)

    impulse = getattr(getattr(model, "training_job", None), "impulse", None)
    project_id = (
        patched.get("project_id")
        or getattr(dep, "project_id", None)
        or getattr(impulse, "project_id", None)
    )
    if project_id:
        patched["project_id"] = str(project_id)

    if not patched.get("project_name") and project_id:
        project = db.query(Project).filter(Project.id == str(project_id)).first()
        if project and project.name:
            patched["project_name"] = project.name

    # Impulse name powers the download filename ({project}-{impulse}-#{n}.pxe).
    # Petaledge has no separate "impulse name" concept distinct from impulses.name,
    # so we read it directly off the impulse row here. If the relationship is
    # absent (orphan model), fall back to the literal "impulse".
    # TODO: revisit if a richer impulse-display-name field is introduced.
    if not patched.get("impulse_name"):
        impulse_name = getattr(impulse, "name", None)
        patched["impulse_name"] = impulse_name if impulse_name else "impulse"

    # deploy_version: prefer an explicit value already in meta; otherwise count
    # prior deployments for this model and use that ordinal (1-based).
    if "deploy_version" not in patched:
        try:
            prior = (
                db.query(Deployment)
                .filter(Deployment.model_id == dep.model_id)
                .filter(Deployment.created_at <= dep.created_at)
                .filter(Deployment.id != dep.id)
                .count()
            )
        except Exception:
            prior = 0
        patched["deploy_version"] = int(prior) + 1

    patched["model_hash"] = f"sha256:{hashlib.sha256(tflite_bytes).hexdigest()}"
    return patched


def _package_dsp_blocks(meta: dict) -> list:
    """Return dsp_blocks with image params filled from packaged metadata."""
    blocks = []
    input_shape = meta.get("input_shape") or []
    default_width = int(meta.get("image_width") or (input_shape[1] if len(input_shape) >= 2 else 96))
    default_height = int(meta.get("image_height") or (input_shape[0] if len(input_shape) >= 1 else 96))
    default_grayscale = bool(
        meta.get("grayscale", False)
        or meta.get("channel_order") == "grayscale"
        or (len(input_shape) >= 3 and int(input_shape[-1]) == 1)
    )
    default_resize_mode = meta.get("resize_mode", "Fit shortest axis")

    for block in meta.get("dsp_blocks", []):
        block_copy = dict(block)
        params = dict(block_copy.get("params", {}))
        if block_copy.get("type") == "image":
            params.setdefault("image_width", default_width)
            params.setdefault("image_height", default_height)
            params.setdefault("grayscale", default_grayscale)
            params.setdefault("resize_mode", default_resize_mode)
            block_copy["params"] = params
        blocks.append(block_copy)
    return blocks


# ── .pxe manifest (EI-canonical wire format) ────────────────────────────────

# Internal training model_type → EI .eim wire model_type.
_INTERNAL_TO_WIRE_MODEL_TYPE: dict[str, str] = {
    "classification":     "classification",
    "yolo_pro_detection": "object_detection",
    "ssd_detection":      "object_detection",   # MobileNetV2 SSD FPN-Lite (decoded PXE)
    "object_detection":   "object_detection",
    "detection_heatmap":  "constrained_object_detection",
    "fomo":               "constrained_object_detection",
    "anomaly_gmm":        "anomaly_gmm",
    "anomaly":            "anomaly_gmm",
    "regression":         "regression",
}

# DSP block resize_mode (internal vocab) → EI wire image_resize_mode.
_RESIZE_MODE_TO_WIRE: dict[str, str] = {
    "Fit shortest axis": "fit-shortest",
    "Fit longest axis":  "fit-longest",
    "Squash":            "squash",
    "Crop":              "crop",
}


def _to_wire_model_type(internal: str) -> str:
    """Translate the internal training model_type to the EI wire value.

    Fails the build (ValueError) on an unrecognized value so an unknown
    architecture never silently emits a wrong manifest. Out-of-spec values
    are a packaging bug, not a runtime fallback case.
    """
    if not isinstance(internal, str) or not internal.strip():
        raise ValueError(
            f"meta.model_type must be a non-empty string, got {internal!r}"
        )
    wire = _INTERNAL_TO_WIRE_MODEL_TYPE.get(internal.strip())
    if wire is None:
        raise ValueError(
            f"meta.model_type={internal!r} has no EI wire translation; "
            f"add it to _INTERNAL_TO_WIRE_MODEL_TYPE or fix the trainer"
        )
    return wire


def _derive_image_resize_mode(meta: dict) -> str:
    """Pick the EI wire image_resize_mode from the first image DSP block."""
    for block in meta.get("dsp_blocks", []) or []:
        if (block or {}).get("type") != "image":
            continue
        params = (block or {}).get("params", {}) or {}
        raw = params.get("resize_mode") or meta.get("resize_mode") or "Fit shortest axis"
        wire = _RESIZE_MODE_TO_WIRE.get(raw)
        if wire is None:
            raise ValueError(
                f"unrecognized dsp_blocks[].params.resize_mode={raw!r}; "
                f"known modes: {sorted(_RESIZE_MODE_TO_WIRE)}"
            )
        return wire
    # Non-image models still need a value — emit a benign default.
    raw = meta.get("resize_mode", "Fit shortest axis")
    return _RESIZE_MODE_TO_WIRE.get(raw, "fit-shortest")


def _derive_coordinate_space(wire_model_type: str) -> str:
    """Bbox coordinate space for the wire model_type.

    YOLO Pro's decoded export produces normalized coordinates; FOMO's heatmap
    is emitted in pixel space by the runner's _postprocess_fomo. Classification
    and regression have no bboxes — declare normalized so the field is always
    a valid enum value.
    """
    if wire_model_type == "constrained_object_detection":
        return "pixel"
    return "normalized"


def _derive_output_tensor_format(wire_model_type: str, meta: dict) -> str:
    """How the runner should parse output tensors.

    YOLO Pro: 'decoded' (single (1,N,6)) once the decoded TFLite variant is in
    use; runtime still detects raw-multi-head on older models, but the manifest
    declares 'decoded' because the build pipeline always uses the decoded
    variant for PXE (see run_deployment_job).
    FOMO: 'fomo_heatmap'.
    Classification: 'softmax'.
    Anomaly / regression: 'softmax' is the closest neutral pass-through value.
    """
    if wire_model_type == "object_detection":
        return "decoded"
    if wire_model_type == "constrained_object_detection":
        return "fomo_heatmap"
    return "softmax"


def _project_id_to_int(value) -> int:
    """Coerce a Petaledge project_id (UUID string or int) into a positive int.

    Petaledge has no numeric project ID column; the canonical ID is a UUID.
    The .eim consumer expects a numeric ID, so the packager derives one as
    the low 31 bits of CRC32 of the UUID. This is stable per project and
    fits comfortably inside JSON-safe int range.
    """
    if isinstance(value, bool):
        raise ValueError(f"meta.project_id must not be a bool, got {value!r}")
    if isinstance(value, int):
        if value <= 0:
            raise ValueError(f"meta.project_id must be positive, got {value!r}")
        return value
    if isinstance(value, str) and value.strip():
        # Stable derivation: CRC32 → mask to 31 bits, force >= 1.
        derived = (zlib.crc32(value.strip().encode("utf-8")) & 0x7FFFFFFF) or 1
        return derived
    raise ValueError(
        f"meta.project_id missing or invalid: {value!r}; expected int or non-empty string"
    )


def _build_thresholds(labels: list[str], default: float) -> list[dict]:
    return [{"label": l, "value": float(default)} for l in labels]


def _build_pxe_manifest_dict(meta: dict, model_hash: str) -> dict:
    """Return the canonical EI-aligned .pxe manifest as a dict.

    No internal-only fields (architecture, output_type, target, device_profile,
    fomo_version, grid_size, format, yolo_size) appear in the manifest — those
    were dead weight to the EI consumer. If build-time diagnostics need them,
    they belong in a separate build_metadata section.
    """
    labels = list(meta.get("label_names", []))
    wire_model_type = _to_wire_model_type(meta.get("model_type", ""))
    threshold = float(meta.get("threshold", 0.5))

    manifest: dict = {
        "version":              PXE_MANIFEST_VERSION,
        "project_id":           _project_id_to_int(meta.get("project_id")),
        "project_name":         str(meta.get("project_name") or ""),
        "model_type":           wire_model_type,
        "input_shape":          [int(d) for d in meta.get("input_shape", [])],
        "image_input_frames":   int(meta.get("image_input_frames", 1)),
        "image_resize_mode":    _derive_image_resize_mode(meta),
        "coordinate_space":     _derive_coordinate_space(wire_model_type),
        "output_tensor_format": _derive_output_tensor_format(wire_model_type, meta),
        "label_names":          labels,
        "num_classes":          len(labels),
        "threshold":            threshold,
        "thresholds":           _build_thresholds(labels, threshold),
        "iou_threshold":        float(meta.get("iou_threshold", 0.45)),
        "has_anomaly":          bool(meta.get("has_anomaly", False)),
        "normalize_input":      bool(meta.get("normalize_input", False)),
        "input_mean":           float(meta.get("input_mean", 0.0)),
        "input_std":            float(meta.get("input_std", 255.0) or 255.0),
        "channel_order":        str(meta.get("channel_order", "rgb")).lower(),
        "model_hash":           model_hash,
    }

    # YOLO-only: emit reg_max so the runner's raw-multi-head path can decode DFL.
    if wire_model_type == "object_detection":
        manifest["reg_max"] = int(meta.get("reg_max", 16))

    # Optional manifest overrides for the runner's `project` block.
    if "project_owner" in meta:
        manifest["project_owner"] = str(meta["project_owner"])
    if "deploy_version" in meta:
        manifest["deploy_version"] = int(meta["deploy_version"])

    # Canonical EI-style download filename, baked into the package so any
    # consumer (backend, frontend, third-party CLI) can read the intended
    # name straight from manifest.json without re-querying.
    manifest["download_filename"] = build_pxe_download_filename(
        manifest["project_name"],
        str(meta.get("impulse_name") or "impulse"),
        int(meta.get("deploy_version", 1)),
    )

    return manifest


def _build_manifest_dict(meta: dict, target: str, device_profile: str | None = None) -> dict:
    """Return the manifest payload as a dict."""
    label_names = meta.get("label_names", [])
    return {
        "version":            meta.get("version", 1),
        "model_type":         meta.get("model_type", meta.get("output_type", "classification")),
        "output_type":        meta.get("output_type", "classification"),
        "architecture":       meta.get("architecture", ""),
        "input_shape":        meta.get("input_shape", []),
        "num_classes":        meta.get("num_classes", len(label_names)),
        "label_names":        label_names,
        "normalize_input":    meta.get("normalize_input", False),
        "channel_order":      meta.get("channel_order", "rgb"),
        "reg_max":            meta.get("reg_max", 16),
        "threshold":          meta.get("threshold", 0.5),
        "input_mean":         meta.get("input_mean", 0.0),
        "input_std":          meta.get("input_std", 255.0),
        "yolo_size":          meta.get("yolo_size", ""),
        "target":             target,
        "device_profile":     device_profile or "",
        "project_id":         meta.get("project_id", ""),
        "project_name":       meta.get("project_name", ""),
        "model_hash":         meta.get("model_hash", ""),
        "fomo_version":       meta.get("fomo_version", 1),
        "grid_size":          meta.get("grid_size", 0),
        "has_anomaly":        bool(meta.get("has_anomaly", False)),
        "image_input_frames": int(meta.get("image_input_frames", 1)),
        "iou_threshold":      float(meta.get("iou_threshold", 0.45)),
    }


def _build_dsp_config(meta: dict, device_profile: str | None = None) -> bytes:
    """Return a minimal dsp_config.json as UTF-8 bytes."""
    config = {
        "dsp_blocks":      _package_dsp_blocks(meta),
        "input_shape":     meta.get("input_shape", []),
        "label_names":     meta.get("label_names", []),
        "architecture":    meta.get("architecture", ""),
        "output_type":     meta.get("output_type", ""),
        "normalize_input": meta.get("normalize_input", False),
        "channel_order":   meta.get("channel_order", "rgb"),
        "model_type":      meta.get("model_type", meta.get("output_type", "classification")),
        "version":         meta.get("version", 1),
        "device_profile":  device_profile or "",
        "frequency_hz":    float(meta.get("frequency_hz", 100.0)),
        "n_axes":          int(len(meta.get("axes", [])) or meta.get("n_axes", 1)),
        "axes":            meta.get("axes", []),
        "interval_ms":     round(1000.0 / float(meta.get("frequency_hz", 100.0)), 4),
    }
    return json.dumps(config, indent=2).encode()


# ── Default postprocess settings (mirrors post_processing_pipeline.py _DEFAULTS) ──
_PP_DEFAULTS: dict = {
    "enabled":          True,
    "threshold":        0.5,
    "tracking_enabled": False,
    "keep_grace":       3,
    "max_observations": 5,
    "class_filter":     [],
}


def _build_postprocess_config(pp_settings: "PostProcessingSettings | None") -> bytes:
    """Serialize PostProcessingSettings → postprocess_config.json UTF-8 bytes.

    Only fields from the real PostProcessingSettings DB model are included:
      enabled, threshold, class_filter, tracking_enabled, keep_grace, max_observations

    Fields NOT included (deferred):
      - nms_iou_threshold / tracker_iou_threshold: code-level PipelineConfig
        defaults only; not persisted in PostProcessingSettings DB model.
        NMS IoU is already controlled by manifest.iou_threshold in the runner.
      - smoothing / per_class_thresholds / uncertainty_label / anomaly: legacy
        model_metadata["post_processing"] fields; not in PostProcessingSettings schema.

    Tracking fields (tracking_enabled, keep_grace, max_observations) are packaged
    now but runtime application is deferred to Phase 3.
    """
    if pp_settings is not None:
        d = {k: v for k, v in vars(pp_settings).items() if not k.startswith("_")}
    else:
        d = {}

    config = {
        "enabled":          bool(d.get("enabled") if d.get("enabled") is not None
                                 else _PP_DEFAULTS["enabled"]),
        "threshold":        float(d.get("threshold") if d.get("threshold") is not None
                                  else _PP_DEFAULTS["threshold"]),
        "class_filter":     list(d.get("class_filter") or _PP_DEFAULTS["class_filter"]),
        "tracking_enabled": bool(d.get("tracking_enabled") if d.get("tracking_enabled") is not None
                                 else _PP_DEFAULTS["tracking_enabled"]),
        "keep_grace":       int(d.get("keep_grace") if d.get("keep_grace") is not None
                                 else _PP_DEFAULTS["keep_grace"]),
        "max_observations": int(d.get("max_observations") if d.get("max_observations") is not None
                                 else _PP_DEFAULTS["max_observations"]),
    }
    _validate_postprocess_config_dict(config)
    return json.dumps(config, indent=2).encode()


def _validate_postprocess_config_dict(cfg: dict) -> None:
    """Validate a deserialized postprocess_config.json dict. Raises ValueError on bad data."""
    if not isinstance(cfg.get("enabled"), bool):
        raise ValueError("postprocess_config.enabled must be a bool")
    th = cfg.get("threshold")
    if not isinstance(th, (int, float)) or not (0.0 <= float(th) <= 1.0):
        raise ValueError(f"postprocess_config.threshold must be float in [0,1], got {th!r}")
    cf = cfg.get("class_filter")
    if not isinstance(cf, list) or not all(isinstance(x, str) for x in cf):
        raise ValueError("postprocess_config.class_filter must be a list of strings")
    if not isinstance(cfg.get("tracking_enabled"), bool):
        raise ValueError("postprocess_config.tracking_enabled must be a bool")
    kg = cfg.get("keep_grace")
    if not isinstance(kg, int) or kg < 0:
        raise ValueError(f"postprocess_config.keep_grace must be int >= 0, got {kg!r}")
    mo = cfg.get("max_observations")
    if not isinstance(mo, int) or mo < 1:
        raise ValueError(f"postprocess_config.max_observations must be int >= 1, got {mo!r}")


def _build_manifest(meta: dict, target: str, device_profile: str | None = None) -> bytes:
    """Return a manifest.json as UTF-8 bytes.

    Mirrors the fields Edge Impulse ships inside every .eim package so that
    .pe consumers can inspect the package without loading the model.
    """
    return json.dumps(_build_manifest_dict(meta, target, device_profile), indent=2).encode()


# ─── TFLite package ───────────────────────────────────────────────────────────

def _gen_tflite(tflite_bytes, meta, options) -> bytes:
    label_names = meta.get("label_names", [])
    output_type = meta.get("output_type") or ("yolo_pro_detection" if "yolo_pro" in meta.get("architecture", "").lower() else "classification")

    if output_type == "yolo_pro_detection":
        inference_py = _YOLO_INFERENCE_PY_TEMPLATE
    elif output_type in ("ssd_detection", "object_detection"):
        inference_py = _SSD_INFERENCE_PY_TEMPLATE
    else:
        inference_py = _INFERENCE_PY_TEMPLATE
    # inference_py is now a plain string — labels and shapes are loaded from
    # manifest.json at runtime, not substituted at build time.

    return _build_pe_container(
        model_bytes=tflite_bytes,
        dsp_config_bytes=_build_dsp_config(meta, meta.get("device_profile")),
        labels_bytes="\n".join(label_names).encode(),
        manifest_bytes=_build_manifest(meta, "tflite", meta.get("device_profile")),
        inference_bytes=inference_py.encode(),
    )


# ─── Arduino library ──────────────────────────────────────────────────────────

def _is_fomo_model(meta: dict) -> bool:
    return classify_model_type(meta) == MODEL_TYPE_FOMO


# The two helpers below are the last line of defence, not the primary gate.
# `POST /deployment/build` runs the same `check_compatibility()` before it
# creates the Deployment row, so an incompatible build never reaches here in
# the normal flow.  They stay because a build enqueued before Phase 4 shipped —
# or by any caller that skips the API — still has to fail loudly rather than
# emit a package whose inference loop is silently wrong.  Their messages come
# straight from the service, so there is one wording, not two.

def _raise_if_fomo_embedded(meta: dict, target: str) -> None:
    """FOMO models have a 4-D output tensor (1, H, W, N+1) that requires a
    spatial post-processing loop.  The generated C++/Arduino/ESP32 templates
    assume a flat 1-D classification output and would silently produce wrong
    results.  Fail clearly instead."""
    if classify_model_type(meta) == MODEL_TYPE_FOMO:
        _raise_if_incompatible(meta, target)


def _raise_if_detection_embedded(meta: dict, target: str) -> None:
    """Reject multi-output detection models (YOLO-Pro, SSD) for embedded C
    targets that only support a single flat classification output tensor."""
    if classify_model_type(meta) in (MODEL_TYPE_SSD, MODEL_TYPE_YOLO_PRO):
        _raise_if_incompatible(meta, target)


def _raise_if_incompatible(meta: dict, target: str) -> None:
    result = check_compatibility(meta, target)
    if not result.compatible:
        raise ValueError(result.message)


def _estimate_tensor_arena_size(tflite_bytes: bytes) -> int:
    """Estimate a safe tensor arena size for TFLite Micro.

    Rule of thumb: 3× the TFLite file size, rounded up to the nearest 1 KB,
    with a minimum floor of 32 KB and a maximum cap of 512 KB.
    """
    estimated = len(tflite_bytes) * 3
    rounded   = ((estimated + 1023) // 1024) * 1024
    return max(32 * 1024, min(rounded, 512 * 1024))


def _gen_arduino(tflite_bytes, meta, options) -> bytes:
    _raise_if_fomo_embedded(meta, "arduino")
    _raise_if_detection_embedded(meta, "arduino")
    label_names  = meta.get("label_names", [])
    input_shape  = meta.get("input_shape", [64])
    input_length = 1
    for d in input_shape:
        input_length *= d
    # raw_input_length: size of the sensor buffer before DSP processing.
    # Falls back to input_length for raw-passthrough DSP blocks.
    raw_input_length = meta.get("raw_input_length", input_length)

    # Determine the first DSP block type to generate the correct classify_raw body.
    dsp_blocks = meta.get("dsp_blocks", []) or [{"type": "raw", "params": {}}]
    dsp_block_type = (dsp_blocks[0] or {}).get("type", "raw")
    dsp_params = (dsp_blocks[0] or {}).get("params", {}) or {}
    dsp_normalize = bool(dsp_params.get("normalize", False))

    if len(input_shape) >= 3:
        image_height   = int(input_shape[0])
        image_width    = int(input_shape[1])
        image_channels = int(input_shape[2])
    elif len(input_shape) == 2:
        image_height   = int(input_shape[0])
        image_width    = int(input_shape[1])
        image_channels = 1
    else:
        image_height   = 1
        image_width    = int(input_shape[0]) if input_shape else 1
        image_channels = 1

    dsp_norm_mode = "zero_one" if dsp_normalize else "none"

    if dsp_block_type == "raw":
        classify_raw_impl = (
            "// classify_raw: raw DSP block — features equal raw samples; pass through directly.\n"
            "int PetalEdge::classify_raw(float* raw, int raw_len) {\n"
            "  return classify(raw, raw_len);\n"
            "}"
        )
    elif dsp_block_type == "image":
        classify_raw_impl = (
            "// classify_raw: image DSP block — normalise flat pixel input [0,255]->[0,1] via run_classifier.\n"
            "// raw must be a flat W*H*C array of pixel values in [0,255].\n"
            "int PetalEdge::classify_raw(float* raw, int raw_len) {\n"
            "  Signal s{raw, raw_len};\n"
            "  auto p = run_classifier(&s);\n"
            "  int best = 0;\n"
            "  float best_s = p.scores.empty() ? p.confidence : p.scores[0];\n"
            "  for (int i = 1; i < (int)p.scores.size(); i++) {\n"
            "    if (p.scores[i] > best_s) { best_s = p.scores[i]; best = i; }\n"
            "  }\n"
            "  return best;\n"
            "}"
        )
    else:
        classify_raw_impl = (
            f'// classify_raw: DSP block "{dsp_block_type}" requires offline feature extraction.\n'
            f"// See dsp_config.json for parameters.\n"
            f'#error "classify_raw does not support DSP block type \\"{dsp_block_type}\\"; extract features offline."\n'
            f"int PetalEdge::classify_raw(float* raw, int raw_len) {{ (void)raw; (void)raw_len; return -1; }}"
        )

    # Convert TFLite bytes to C array
    c_array = _bytes_to_c_array(tflite_bytes, "model_tflite")

    arena_size  = _estimate_tensor_arena_size(tflite_bytes)

    arduino_cpp = _ARDUINO_INFERENCE_CPP.substitute(
        NUM_LABELS=len(label_names),
        LABELS_ARRAY=", ".join(f'"{l}"' for l in label_names),
        INPUT_LENGTH=input_length,
        RAW_INPUT_LENGTH=raw_input_length,
        TENSOR_ARENA_SIZE=arena_size,
        CLASSIFY_RAW_IMPL=classify_raw_impl,
        DSP_BLOCK_TYPE=dsp_block_type,
        DSP_NORMALIZE=("true" if dsp_normalize else "false"),
        IMAGE_WIDTH=image_width,
        IMAGE_HEIGHT=image_height,
        IMAGE_CHANNELS=image_channels,
        DSP_NORM_MODE=dsp_norm_mode,
        DSP_INPUT_LENGTH=input_length,
    )

    return _zip_files({
        "PetalEdgeInference/PetalEdgeInference.h":    _ARDUINO_H.encode(),
        "PetalEdgeInference/PetalEdgeInference.cpp":  arduino_cpp.encode(),
        "PetalEdgeInference/model.h":                 c_array.encode(),
        "PetalEdgeInference/library.properties":      _ARDUINO_PROPS.encode(),
        "PetalEdgeInference/examples/inference_example/inference_example.ino":
            _ARDUINO_EXAMPLE.substitute(INPUT_LENGTH=input_length, RAW_INPUT_LENGTH=raw_input_length).encode(),
        "labels.txt":      "\n".join(label_names).encode(),
        "dsp_config.json": _build_dsp_config(meta, meta.get("device_profile")),
        "manifest.json":   _build_manifest(meta, "arduino", meta.get("device_profile")),
        "README.md": _ARDUINO_README,
    })


# ─── ESP32 project ────────────────────────────────────────────────────────────

def _gen_esp32(tflite_bytes, meta, options) -> bytes:
    _raise_if_fomo_embedded(meta, "esp32")
    _raise_if_detection_embedded(meta, "esp32")
    label_names  = meta.get("label_names", [])
    input_shape  = meta.get("input_shape", [64])
    input_length = 1
    for d in input_shape:
        input_length *= d

    c_array = _bytes_to_c_array(tflite_bytes, "model_tflite")

    arena_size = _estimate_tensor_arena_size(tflite_bytes)

    main_cpp = _ESP32_MAIN_CPP.substitute(
        NUM_LABELS=len(label_names),
        LABELS_ARRAY=", ".join(f'"{l}"' for l in label_names),
        INPUT_LENGTH=input_length,
        TENSOR_ARENA_SIZE=arena_size,
    )

    return _zip_files({
        "main/main.cpp":         main_cpp.encode(),
        "main/model.h":          c_array.encode(),
        "main/CMakeLists.txt":   _ESP32_CMAIN.encode(),
        "CMakeLists.txt":        _ESP32_CROOT.encode(),
        "sdkconfig.defaults":    _ESP32_SDK.encode(),
        "labels.txt":            "\n".join(label_names).encode(),
        "dsp_config.json":       _build_dsp_config(meta, meta.get("device_profile")),
        "manifest.json":         _build_manifest(meta, "esp32", meta.get("device_profile")),
        "README.md":             _ESP32_README,
    })


# ─── Raspberry Pi package ─────────────────────────────────────────────────────

def _gen_raspberry_pi(tflite_bytes, meta, options) -> bytes:
    label_names = meta.get("label_names", [])
    output_type = meta.get("output_type", "classification")
    if output_type == "yolo_pro_detection":
        inference_py = _YOLO_INFERENCE_PY_TEMPLATE
    elif output_type in ("ssd_detection", "object_detection"):
        inference_py = _SSD_INFERENCE_PY_TEMPLATE
    else:
        inference_py = _INFERENCE_PY_TEMPLATE
    return _build_pe_container(
        model_bytes=tflite_bytes,
        dsp_config_bytes=_build_dsp_config(meta, meta.get("device_profile")),
        labels_bytes="\n".join(label_names).encode(),
        manifest_bytes=_build_manifest(meta, "raspberry_pi", meta.get("device_profile")),
        inference_bytes=inference_py.encode(),
    )


# ─── UNO Q package ────────────────────────────────────────────────────────────

def _gen_unoq(tflite_bytes, meta, options) -> bytes:
    """Linux runtime package for UNO Q — same PEM1 container as raspberry_pi."""
    label_names = meta.get("label_names", [])
    output_type = meta.get("output_type", "classification")
    if output_type == "yolo_pro_detection":
        inference_py = _YOLO_INFERENCE_PY_TEMPLATE
    elif output_type in ("ssd_detection", "object_detection"):
        inference_py = _SSD_INFERENCE_PY_TEMPLATE
    else:
        inference_py = _INFERENCE_PY_TEMPLATE
    return _build_pe_container(
        model_bytes=tflite_bytes,
        dsp_config_bytes=_build_dsp_config(meta, meta.get("device_profile")),
        labels_bytes="\n".join(label_names).encode(),
        manifest_bytes=_build_manifest(meta, "unoq", meta.get("device_profile")),
        inference_bytes=inference_py.encode(),
    )


# ─── PXE package ──────────────────────────────────────────────────────────────

def _gen_pxe(
    tflite_bytes: bytes,
    meta: dict,
    options: dict,
    postprocess_bytes: bytes | None = None,
) -> bytes:
    """Build a standalone .pxe v2 binary package.

    Format: PXE1 v2 opaque binary container (36-byte header).
    Runtime: host unpacks to temp dir, launches runner.py as OS process,
             communicates via Edge Impulse .eim stdio-JSONL protocol.

    postprocess_bytes: serialized postprocess_config.json.  When None (e.g. in
    tests that call _gen_pxe directly without a DB), defaults are used so the
    section is always present in v2 packages.
    """
    # Hash is computed from the final tflite bytes; the manifest builder always
    # emits it (overriding any meta-supplied value to avoid a stale hash).
    model_hash = f"sha256:{hashlib.sha256(tflite_bytes).hexdigest()}"
    manifest_payload = _build_pxe_manifest_dict(meta, model_hash)
    validate_pxe_manifest(manifest_payload)
    manifest = json.dumps(manifest_payload, indent=2).encode()

    if postprocess_bytes is None:
        postprocess_bytes = _build_postprocess_config(None)

    return _build_pxe_container(
        runner_bytes=_PXE_RUNNER.encode(),
        model_bytes=tflite_bytes,
        manifest_bytes=manifest,
        labels_bytes="\n".join(meta.get("label_names", [])).encode(),
        dsp_config_bytes=_build_dsp_config(meta, meta.get("device_profile")),
        postprocess_config_bytes=postprocess_bytes,
    )


# ─── C++ library ──────────────────────────────────────────────────────────────

def _gen_cpp(tflite_bytes, meta, options) -> bytes:
    _raise_if_fomo_embedded(meta, "cpp")
    _raise_if_detection_embedded(meta, "cpp")
    label_names  = meta.get("label_names", [])
    input_shape  = meta.get("input_shape", [64])
    input_length = 1
    for d in input_shape:
        input_length *= d

    dsp_blocks     = meta.get("dsp_blocks", []) or [{"type": "raw", "params": {}}]
    dsp_block_type = (dsp_blocks[0] or {}).get("type", "raw")
    dsp_params     = (dsp_blocks[0] or {}).get("params", {}) or {}
    dsp_normalize  = bool(dsp_params.get("normalize", False))

    if len(input_shape) >= 3:
        image_height   = int(input_shape[0])
        image_width    = int(input_shape[1])
        image_channels = int(input_shape[2])
    elif len(input_shape) == 2:
        image_height   = int(input_shape[0])
        image_width    = int(input_shape[1])
        image_channels = 1
    else:
        image_height   = 1
        image_width    = int(input_shape[0]) if input_shape else 1
        image_channels = 1

    dsp_norm_mode = "zero_one" if dsp_normalize else "none"

    c_array    = _bytes_to_c_array(tflite_bytes, "model_tflite")
    arena_size = _estimate_tensor_arena_size(tflite_bytes)

    return _zip_files({
        "src/inference.h":    _CPP_INFERENCE_H.encode(),
        "src/inference.cpp":  _CPP_INFERENCE_CPP.substitute(
            NUM_LABELS=len(label_names),
            LABELS_ARRAY=", ".join(f'"{l}"' for l in label_names),
            INPUT_LENGTH=input_length,
            TENSOR_ARENA_SIZE=arena_size,
            DSP_BLOCK_TYPE=dsp_block_type,
            DSP_NORMALIZE=("true" if dsp_normalize else "false"),
            IMAGE_WIDTH=image_width,
            IMAGE_HEIGHT=image_height,
            IMAGE_CHANNELS=image_channels,
            DSP_NORM_MODE=dsp_norm_mode,
            DSP_INPUT_LENGTH=input_length,
        ).encode(),
        "src/model.h":        c_array.encode(),
        "src/main_example.cpp": _CPP_MAIN_EXAMPLE.encode(),
        "CMakeLists.txt":     _CPP_CMAKE.encode(),
        "labels.txt":         "\n".join(label_names).encode(),
        "dsp_config.json":    _build_dsp_config(meta, meta.get("device_profile")),
        "manifest.json":      _build_manifest(meta, "cpp", meta.get("device_profile")),
        "README.md":          _CPP_README,
    })


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _zip_files(files: dict) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, data in files.items():
            if isinstance(data, str):
                data = data.encode()
            zf.writestr(name, data)
    return buf.getvalue()


def _build_pe_container(
    *,
    model_bytes: bytes,
    dsp_config_bytes: bytes,
    labels_bytes: bytes,
    manifest_bytes: bytes,
    inference_bytes: bytes = b"",
) -> bytes:
    """Serialize a .pe package into the fixed binary container format.

    Header layout (v2):
      magic(4s) | format_version(I) | model_sz(I) | dsp_sz(I) |
      labels_sz(I) | manifest_sz(I) | inference_sz(I)

    The inference section carries the model-type-specific inference.py script.
    It may be 0 bytes for targets that embed inference code separately (e.g. Arduino).
    """
    header = struct.pack(
        "<4sIIIIII",
        _PE_MAGIC,
        _PE_FORMAT_VERSION,
        len(model_bytes),
        len(dsp_config_bytes),
        len(labels_bytes),
        len(manifest_bytes),
        len(inference_bytes),
    )
    return b"".join((
        header,
        model_bytes,
        dsp_config_bytes,
        labels_bytes,
        manifest_bytes,
        inference_bytes,
    ))


def _build_pxe_container(
    *,
    runner_bytes: bytes,
    model_bytes: bytes,
    manifest_bytes: bytes,
    labels_bytes: bytes,
    dsp_config_bytes: bytes,
    postprocess_config_bytes: bytes,
) -> bytes:
    """Serialize a .pxe package into the PXE1 v2 binary container format.

    Header (36 bytes, little-endian):
      magic(4s) | format_version(I) | runner_sz(I) | model_sz(I) |
      manifest_sz(I) | labels_sz(I) | dsp_sz(I) | postprocess_sz(I) | flags(I)

    Sections (in order):
      runner.py | model.tflite | manifest.json | labels.txt |
      dsp_config.json | postprocess_config.json

    Backward compatibility:
      Old v1 loaders reject this package at "Unsupported .pxe version 2"
      (clean error, no silent corruption).  New loaders support both v1 and v2.
    """
    header = _PXE_HEADER_STRUCT_V2.pack(
        _PXE_MAGIC,
        _PXE_FORMAT_VERSION_V2,
        len(runner_bytes),
        len(model_bytes),
        len(manifest_bytes),
        len(labels_bytes),
        len(dsp_config_bytes),
        len(postprocess_config_bytes),
        0,   # flags — reserved, always zero
    )
    return b"".join((
        header,
        runner_bytes,
        model_bytes,
        manifest_bytes,
        labels_bytes,
        dsp_config_bytes,
        postprocess_config_bytes,
    ))


def _bytes_to_c_array(data: bytes, var_name: str) -> str:
    hex_values = ", ".join(f"0x{b:02x}" for b in data)
    return (
        f"// Auto-generated by PetalEdge\n"
        f"#pragma once\n"
        f"#include <stdint.h>\n\n"
        f"const unsigned int {var_name}_len = {len(data)};\n"
        f"alignas(8) const uint8_t {var_name}[] = {{\n  {hex_values}\n}};\n"
    )


# == PXE runner (EI stdio-JSONL protocol, loaded from unoq/runtime/runner.py) ==
#
# The runner.py source lives in the unoq package so it can be linted, tested,
# and edited as a normal Python file. The packager reads it at import time
# and embeds the bytes verbatim into every .pxe container.

_PXE_RUNNER_PATH = pathlib.Path(__file__).resolve().parents[3] / "unoq" / "runtime" / "runner.py"
try:
    _PXE_RUNNER = _PXE_RUNNER_PATH.read_text(encoding="utf-8")
except FileNotFoundError as _runner_load_exc:
    raise RuntimeError(
        f".pxe runner template missing at {_PXE_RUNNER_PATH!s}; "
        f"this is a repo layout error, not a runtime condition"
    ) from _runner_load_exc

# ─── Code templates ───────────────────────────────────────────────────────────

_INFERENCE_PY_TEMPLATE = """\
\"\"\"
PetalEdge-compatible Inference — all runtime parameters loaded from manifest.json.
Do not hardcode labels or shapes here; edit manifest.json and rebuild if needed.
\"\"\"
import json
import os as _os
import numpy as np
try:
    from tflite_runtime.interpreter import Interpreter
except ImportError:
    import tensorflow as tf
    Interpreter = tf.lite.Interpreter

# ─── Manifest (source of truth — same role as .eim embedded manifest) ────────
_MANIFEST_PATH = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "manifest.json")
if not _os.path.exists(_MANIFEST_PATH):
    raise RuntimeError(
        f"manifest.json not found at {_MANIFEST_PATH!r}; ensure the .pe package is intact"
    )
with open(_MANIFEST_PATH) as _f:
    _MANIFEST = json.load(_f)

_REQUIRED = ("label_names", "input_shape", "normalize_input", "channel_order", "model_type", "version")
_missing = [k for k in _REQUIRED if k not in _MANIFEST]
if _missing:
    raise RuntimeError(f"manifest.json is missing required fields: {_missing}; rebuild the .pe package")

LABELS        = _MANIFEST["label_names"]
INPUT_SHAPE   = _MANIFEST["input_shape"]
NORMALIZE     = _MANIFEST["normalize_input"]
CHANNEL_ORDER = _MANIFEST["channel_order"].lower()
MODEL_TYPE    = _MANIFEST["model_type"]
THRESHOLD     = float(_MANIFEST.get("threshold", 0.5))
INPUT_MEAN    = float(_MANIFEST.get("input_mean", 0.0))
INPUT_STD     = float(_MANIFEST.get("input_std", 255.0))

if CHANNEL_ORDER not in ("rgb", "bgr", "grayscale"):
    raise RuntimeError(
        f"manifest.json channel_order={CHANNEL_ORDER!r} is invalid; "
        "only 'rgb', 'bgr', or 'grayscale' are supported"
    )


def load_interpreter(model_path: str = "model.tflite"):
    interp = Interpreter(model_path=model_path)
    interp.allocate_tensors()
    return interp


def run_inference(interpreter, features: np.ndarray) -> dict:
    input_details  = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    inp = features.flatten().reshape(input_details[0]["shape"]).astype(np.float32)

    # Channel order from manifest — no heuristics
    if CHANNEL_ORDER == "bgr" and inp.ndim >= 3 and inp.shape[-1] == 3:
        inp = inp[..., ::-1].copy()

    # Normalization from manifest — preserve the declared numeric range
    if NORMALIZE:
        _denom = INPUT_STD if INPUT_STD != 0.0 else 255.0
        inp = (inp - INPUT_MEAN) / _denom

    is_fomo = (MODEL_TYPE == "detection_heatmap")

    # int8 quantization: round → clip → cast (no truncation)
    quant = input_details[0].get("quantization")
    if quant and len(quant) == 2:
        scale, zero_point = quant
        if input_details[0]["dtype"] == np.int8:
            if scale == 0:
                raise RuntimeError("int8 input tensor has scale=0; model is corrupt")
            inp = np.clip(np.round(inp / scale + zero_point), -128, 127).astype(np.int8)

    interpreter.set_tensor(input_details[0]["index"], inp)
    interpreter.invoke()

    output = interpreter.get_tensor(output_details[0]["index"])
    scores = output.flatten().astype(np.float32)

    # Dequantize output if int8
    out_scale, out_zp = output_details[0].get("quantization", (0, 0))
    if out_scale != 0:
        scores = (scores - out_zp) * out_scale

    debug = {"raw_min": float(scores.min()), "raw_max": float(scores.max()), "raw_mean": float(scores.mean())}

    if is_fomo:
        # Output shape: (1, grid_h, grid_w, num_classes+1); class 0 is background.
        # label_names contains object classes only (0-indexed → class index 1..N).
        _, gh, gw, n_all = output.shape
        cell_scores   = scores.reshape(gh, gw, n_all)
        object_scores = cell_scores[:, :, 1:]       # drop background channel
        threshold     = THRESHOLD
        detections    = []
        for row in range(gh):
            for col in range(gw):
                cls_idx = int(np.argmax(object_scores[row, col]))
                conf    = float(object_scores[row, col, cls_idx])
                if conf < threshold:
                    continue
                cx     = (col + 0.5) / gw
                cy     = (row + 0.5) / gh
                half_w = 0.5 / gw
                half_h = 0.5 / gh
                detections.append({
                    "label":      LABELS[cls_idx] if cls_idx < len(LABELS) else str(cls_idx),
                    "confidence": conf,
                    "bbox": {
                        "x1": max(0.0, cx - half_w),
                        "y1": max(0.0, cy - half_h),
                        "x2": min(1.0, cx + half_w),
                        "y2": min(1.0, cy + half_h),
                    },
                })
        return {"detections": detections, "count": len(detections), "is_fomo": True, "debug": debug}

    e = np.exp(scores - scores.max())
    scores = e / e.sum()
    best = int(np.argmax(scores))
    return {
        "label":      LABELS[best] if best < len(LABELS) else str(best),
        "confidence": float(scores[best]),
        "scores":     {LABELS[i] if i < len(LABELS) else str(i): float(scores[i]) for i in range(len(scores))},
        "is_fomo":    False,
        "debug":      debug,
    }


if __name__ == "__main__":
    interp = load_interpreter()
    dummy  = np.zeros(INPUT_SHAPE, dtype=np.float32)
    result = run_inference(interp, dummy)
    print("Prediction:", result)
"""

_SSD_INFERENCE_PY_TEMPLATE = """\
\"\"\"
PetalEdge-compatible SSD Inference — all runtime parameters loaded from manifest.json.
Do not hardcode labels or shapes here; edit manifest.json and rebuild if needed.
\"\"\"
import json
import os as _os
import numpy as np
try:
    from tflite_runtime.interpreter import Interpreter
except ImportError:
    import tensorflow as tf
    Interpreter = tf.lite.Interpreter

# ─── Manifest (source of truth — same role as .eim embedded manifest) ────────
_MANIFEST_PATH = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "manifest.json")
if not _os.path.exists(_MANIFEST_PATH):
    raise RuntimeError(
        f"manifest.json not found at {_MANIFEST_PATH!r}; ensure the .pe package is intact"
    )
with open(_MANIFEST_PATH) as _f:
    _MANIFEST = json.load(_f)

_REQUIRED = ("label_names", "input_shape", "normalize_input", "channel_order", "model_type", "version")
_missing = [k for k in _REQUIRED if k not in _MANIFEST]
if _missing:
    raise RuntimeError(f"manifest.json is missing required fields: {_missing}; rebuild the .pe package")

LABELS        = _MANIFEST["label_names"]
INPUT_SHAPE   = _MANIFEST["input_shape"]
NORMALIZE     = _MANIFEST["normalize_input"]
CHANNEL_ORDER = _MANIFEST["channel_order"].lower()
THRESHOLD     = float(_MANIFEST.get("threshold", 0.25))
INPUT_MEAN    = float(_MANIFEST.get("input_mean", 0.0))
INPUT_STD     = float(_MANIFEST.get("input_std", 255.0))

if CHANNEL_ORDER not in ("rgb", "bgr", "grayscale"):
    raise RuntimeError(
        f"manifest.json channel_order={CHANNEL_ORDER!r} is invalid; "
        "only 'rgb', 'bgr', or 'grayscale' are supported"
    )


def load_interpreter(model_path: str = "model.tflite"):
    interpreter = Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter


def _dequantize(tensor: np.ndarray, detail: dict) -> np.ndarray:
    arr = tensor.astype(np.float32)
    scale, zero = detail.get("quantization", (0.0, 0))
    if scale not in (0, 0.0):
        arr = (arr - zero) * scale
    return arr


def run_inference(interpreter, image_np: np.ndarray, score_threshold: float = THRESHOLD) -> dict:
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    inp = image_np.flatten().reshape(input_details[0]["shape"]).astype(np.float32)

    # Channel order from manifest — no heuristics
    if CHANNEL_ORDER == "bgr" and inp.ndim >= 3 and inp.shape[-1] == 3:
        inp = inp[..., ::-1].copy()

    # Normalization from manifest — no range sniffing
    if NORMALIZE:
        _denom = INPUT_STD if INPUT_STD != 0.0 else 255.0
        inp = np.clip((inp - INPUT_MEAN) / _denom, 0.0, 1.0)

    # int8 quantization: round → clip → cast (no truncation)
    in_scale, in_zero = input_details[0].get("quantization", (0.0, 0))
    if in_scale not in (0, 0.0):
        inp = np.clip(np.round(inp / in_scale + in_zero), -128, 127).astype(np.int8)

    interpreter.set_tensor(input_details[0]["index"], inp)
    interpreter.invoke()

    outputs = [_dequantize(interpreter.get_tensor(od["index"]), od) for od in output_details]
    named = {od.get("name", f"out_{i}"): outputs[i] for i, od in enumerate(output_details)}
    _all_raw = np.concatenate([o.flatten() for o in outputs]) if outputs else np.array([0.0])
    debug = {"raw_min": float(_all_raw.min()), "raw_max": float(_all_raw.max()), "raw_mean": float(_all_raw.mean())}

    boxes = classes = scores = count = None

    for name, out in named.items():
        lname = name.lower()
        flat = out.flatten()
        if boxes is None and ("box" in lname or (out.ndim >= 2 and out.shape[-1] == 4)):
            boxes = out.reshape(-1, 4)
        elif classes is None and "class" in lname:
            classes = flat.astype(np.int32)
        elif scores is None and "score" in lname:
            scores = flat.astype(np.float32)
        elif count is None and ("count" in lname or flat.size == 1):
            count = int(flat[0])

    if boxes is None or scores is None:
        for out in outputs:
            flat = out.flatten()
            if out.ndim >= 2 and out.shape[-1] == 4 and boxes is None:
                boxes = out.reshape(-1, 4)
            elif classes is None and np.all(np.equal(np.mod(flat[: min(len(flat), 16)], 1), 0)):
                classes = flat.astype(np.int32)
            elif count is None and flat.size == 1:
                count = int(flat[0])
            elif scores is None:
                scores = flat.astype(np.float32)

    if boxes is None or scores is None:
        return {"detections": [], "count": 0, "error": "ssd_outputs_not_recognized", "debug": debug}

    if classes is None:
        classes = np.zeros((boxes.shape[0],), dtype=np.int32)
    if count is None:
        count = min(len(boxes), len(scores), len(classes))

    detections = []
    for i in range(min(count, len(boxes), len(scores), len(classes))):
        score = float(scores[i])
        if score < score_threshold:
            continue

        cls = int(classes[i])

        # assuming boxes are [y1, x1, y2, x2] (standard TFLite SSD)
        y1, x1, y2, x2 = boxes[i].tolist()

        detections.append({
            "label": LABELS[cls] if 0 <= cls < len(LABELS) else str(cls),
            "confidence": score,
            "bbox": {
                "x1": float(max(0.0, min(1.0, x1))),
                "y1": float(max(0.0, min(1.0, y1))),
                "x2": float(max(0.0, min(1.0, x2))),
                "y2": float(max(0.0, min(1.0, y2))),
            },
        })

    return {"detections": detections, "count": len(detections), "debug": debug}
"""

_YOLO_INFERENCE_PY_TEMPLATE = """\
\"\"\"
PetalEdge-compatible YOLO-Pro Inference — all runtime parameters loaded from manifest.json.
Do not hardcode labels or shapes here; edit manifest.json and rebuild if needed.
\"\"\"
import json
import os as _os
import numpy as np
try:
    from tflite_runtime.interpreter import Interpreter
except ImportError:
    import tensorflow as tf
    Interpreter = tf.lite.Interpreter

# ─── Manifest (source of truth — same role as .eim embedded manifest) ────────
_MANIFEST_PATH = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "manifest.json")
if not _os.path.exists(_MANIFEST_PATH):
    raise RuntimeError(
        f"manifest.json not found at {_MANIFEST_PATH!r}; ensure the .pe package is intact"
    )
with open(_MANIFEST_PATH) as _f:
    _MANIFEST = json.load(_f)

_REQUIRED = ("label_names", "input_shape", "normalize_input", "channel_order", "model_type", "version")
_missing = [k for k in _REQUIRED if k not in _MANIFEST]
if _missing:
    raise RuntimeError(f"manifest.json is missing required fields: {_missing}; rebuild the .pe package")

LABELS         = _MANIFEST["label_names"]
INPUT_SHAPE    = _MANIFEST["input_shape"]
NORMALIZE      = _MANIFEST["normalize_input"]
CHANNEL_ORDER  = _MANIFEST["channel_order"].lower()
REG_MAX        = int(_MANIFEST.get("reg_max", 16))
THRESHOLD      = float(_MANIFEST.get("threshold", 0.25))
IOU_THRESHOLD  = float(_MANIFEST.get("iou_threshold", 0.45))
INPUT_MEAN     = float(_MANIFEST.get("input_mean", 0.0))
INPUT_STD      = float(_MANIFEST.get("input_std", 255.0))

if CHANNEL_ORDER not in ("rgb", "bgr", "grayscale"):
    raise RuntimeError(
        f"manifest.json channel_order={CHANNEL_ORDER!r} is invalid; "
        "only 'rgb', 'bgr', or 'grayscale' are supported"
    )


def load_interpreter(model_path: str = "model.tflite"):
    interpreter = Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter


def _dequantize(tensor: np.ndarray, detail: dict) -> np.ndarray:
    arr = tensor.astype(np.float32)
    scale, zero = detail.get("quantization", (0.0, 0))
    if scale not in (0, 0.0):
        arr = (arr - zero) * scale
    return arr


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


def _dfl_decode(reg: np.ndarray, reg_max: int = REG_MAX) -> np.ndarray:
    reg = reg.reshape(4, reg_max)
    prob = _softmax(reg, axis=1)
    bins = np.arange(reg_max, dtype=np.float32)
    return (prob * bins[None, :]).sum(axis=1)


def _iou_xyxy(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])

    inter = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    area1 = np.maximum(0.0, box[2] - box[0]) * np.maximum(0.0, box[3] - box[1])
    area2 = np.maximum(0.0, boxes[:, 2] - boxes[:, 0]) * np.maximum(0.0, boxes[:, 3] - boxes[:, 1])
    return inter / (area1 + area2 - inter + 1e-6)


def _nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float = 0.45) -> list[int]:
    if len(boxes) == 0:
        return []
    order = np.argsort(scores)[::-1]
    keep = []
    while len(order) > 0:
        i = int(order[0])
        keep.append(i)
        if len(order) == 1:
            break
        ious = _iou_xyxy(boxes[i], boxes[order[1:]])
        order = order[1:][ious < iou_threshold]
    return keep


def run_inference(
    interpreter,
    image_np: np.ndarray,
    score_threshold: float = THRESHOLD,
    iou_threshold: float = IOU_THRESHOLD,
) -> dict:
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    inp = image_np.flatten().reshape(input_details[0]["shape"]).astype(np.float32)

    # Channel order from manifest — no heuristics
    if CHANNEL_ORDER == "bgr" and inp.ndim >= 3 and inp.shape[-1] == 3:
        inp = inp[..., ::-1].copy()

    # Normalization from manifest — no range sniffing
    if NORMALIZE:
        _denom = INPUT_STD if INPUT_STD != 0.0 else 255.0
        inp = np.clip((inp - INPUT_MEAN) / _denom, 0.0, 1.0)

    # int8 quantization: round → clip → cast (no truncation)
    in_scale, in_zero = input_details[0].get("quantization", (0.0, 0))
    if in_scale not in (0, 0.0):
        inp = np.clip(np.round(inp / in_scale + in_zero), -128, 127).astype(np.int8)

    interpreter.set_tensor(input_details[0]["index"], inp)
    interpreter.invoke()

    outputs = {
        od.get("name", f"out_{i}"): _dequantize(interpreter.get_tensor(od["index"]), od)
        for i, od in enumerate(output_details)
    }
    _all_raw = np.concatenate([v.flatten() for v in outputs.values()]) if outputs else np.array([0.0])
    debug = {"raw_min": float(_all_raw.min()), "raw_max": float(_all_raw.max()), "raw_mean": float(_all_raw.mean())}

    input_h, input_w = INPUT_SHAPE[0], INPUT_SHAPE[1]
    strides = [8, 16, 32]
    detections = []

    def _pick_head(want_size, want_dims):
        # Element count alone cross-wires heads: size(cls_p3) == size(reg_p5)
        # whenever len(LABELS) == REG_MAX / 4 (any 4-class model at REG_MAX=16),
        # and the reshape below would then read the p5 DFL tensor as p3 class
        # scores.  Shape breaks the tie; consumed tensors are popped.
        cands = [k for k, v in outputs.items() if v.size == want_size]
        if not cands:
            return None
        for k in cands:
            if tuple(outputs[k].shape)[-3:] == want_dims:
                return k
        return cands[0]

    for level, stride in enumerate(strides, start=3):
        cls_key = next((k for k in outputs if f"p{level}" in k.lower() and "cls" in k.lower()), None)
        reg_key = next((k for k in outputs if f"p{level}" in k.lower() and "reg" in k.lower()), None)

        gh = input_h // stride
        gw = input_w // stride
        expected_cls = gh * gw * len(LABELS)
        expected_reg = gh * gw * 4 * REG_MAX

        if cls_key is None:
            cls_key = _pick_head(expected_cls, (gh, gw, len(LABELS)))
        if reg_key is None:
            reg_key = _pick_head(expected_reg, (gh, gw, 4 * REG_MAX))

        # Same key for both heads = indistinguishable by name, size and shape
        # (num_classes == 4 * REG_MAX); skip rather than pop the same tensor twice.
        if cls_key is None or reg_key is None or cls_key == reg_key:
            continue

        cls_out = outputs.pop(cls_key).reshape(gh, gw, len(LABELS))
        reg_out = outputs.pop(reg_key).reshape(gh, gw, 4 * REG_MAX)

        # The cls head ends in a sigmoid inside the model, so these are already
        # probabilities — a second sigmoid maps [0,1] onto [0.5, 0.73] and every
        # anchor clears the threshold.
        cls_prob = cls_out

        for y in range(gh):
            for x in range(gw):
                cls_idx = int(np.argmax(cls_prob[y, x]))
                score = float(cls_prob[y, x, cls_idx])
                if score < score_threshold:
                    continue

                # DFL distances are in grid units; scale to pixels by stride
                # (matches the canonical decode in yolo_pro/decode.py).
                l, t, r, b = _dfl_decode(reg_out[y, x]) * float(stride)
                cx = (x + 0.5) * stride
                cy = (y + 0.5) * stride

                x1 = max(0.0, (cx - l) / input_w)
                y1 = max(0.0, (cy - t) / input_h)
                x2 = min(1.0, (cx + r) / input_w)
                y2 = min(1.0, (cy + b) / input_h)

                detections.append({
                    "label": LABELS[cls_idx] if 0 <= cls_idx < len(LABELS) else str(cls_idx),
                    "confidence": score,
                    "bbox": {
                        "x1": float(x1),
                        "y1": float(y1),
                        "x2": float(x2),
                        "y2": float(y2),
                    },
                })

    if not detections:
        return {"detections": [], "count": 0, "debug": debug}

    boxes = np.array([
        [d["bbox"]["x1"], d["bbox"]["y1"], d["bbox"]["x2"], d["bbox"]["y2"]]
        for d in detections
    ], dtype=np.float32)
    scores = np.array([d["confidence"] for d in detections], dtype=np.float32)

    keep = _nms(boxes, scores, iou_threshold=iou_threshold)
    detections = [detections[i] for i in keep]

    return {"detections": detections, "count": len(detections), "debug": debug}
"""


_ARDUINO_H = """\
// Auto-generated by PetalEdge — Arduino inference header
#pragma once
#include <Arduino.h>
#include <vector>

namespace PetalEdge {
  struct Signal {
    const float* data;
    int length;
  };
  struct Prediction {
    const char* label;
    float       confidence;
    std::vector<float> scores;
  };

  bool begin();
  // Pass pre-computed DSP features directly to the model.
  int  classify(float* features, int feature_len);
  Prediction run_classifier(const Signal* signal);
  // Pass raw sensor samples; DSP metadata is in dsp_config.json bundled with this package.
  int  classify_raw(float* raw, int raw_len);
  const char* getLabel(int index);
  float getConfidence(int index);
  float getScore(int index);
}
"""

_ARDUINO_INFERENCE_CPP = Template("""\
// Auto-generated by PetalEdge
#include "PetalEdgeInference.h"
#include "model.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"

static const int TENSOR_ARENA_SIZE = $TENSOR_ARENA_SIZE;
static uint8_t tensor_arena[TENSOR_ARENA_SIZE];
static const char* labels[] = { $LABELS_ARRAY };
static const int NUM_LABELS   = $NUM_LABELS;
static const int INPUT_LEN    = $INPUT_LENGTH;
// Raw sensor samples expected before DSP processing (see dsp_config.json).
static const int RAW_INPUT_LEN = $RAW_INPUT_LENGTH;
static const char* DSP_BLOCK_TYPE   = "$DSP_BLOCK_TYPE";
static const int   DSP_INPUT_LENGTH = $DSP_INPUT_LENGTH;
static const bool  DSP_NORMALIZE    = $DSP_NORMALIZE;
static const int   IMAGE_WIDTH      = $IMAGE_WIDTH;
static const int   IMAGE_HEIGHT     = $IMAGE_HEIGHT;
static const int   IMAGE_CHANNELS   = $IMAGE_CHANNELS;
static const char* DSP_NORM_MODE    = "$DSP_NORM_MODE";

static tflite::AllOpsResolver resolver;
static const tflite::Model* tfl_model = nullptr;
static tflite::MicroInterpreter* interpreter = nullptr;
static TfLiteTensor* input_tensor = nullptr;
static float confidences[NUM_LABELS];

bool PetalEdge::begin() {
  tfl_model = tflite::GetModel(model_tflite);
  if (tfl_model->version() != TFLITE_SCHEMA_VERSION) return false;
  static tflite::MicroInterpreter static_interp(tfl_model, resolver, tensor_arena, TENSOR_ARENA_SIZE);
  interpreter = &static_interp;
  if (interpreter->AllocateTensors() != kTfLiteOk) return false;
  input_tensor = interpreter->input(0);
  return true;
}

int PetalEdge::classify(float* features, int feature_len) {
  for (int i = 0; i < feature_len && i < INPUT_LEN; i++) {
    if (input_tensor->type == kTfLiteInt8) {
      float scale = input_tensor->params.scale;
      int32_t zp  = input_tensor->params.zero_point;
      input_tensor->data.int8[i] = (int8_t)(features[i] / scale + zp);
    } else {
      input_tensor->data.f[i] = features[i];
    }
  }
  if (interpreter->Invoke() != kTfLiteOk) return -1;
  TfLiteTensor* out = interpreter->output(0);
  int best = 0;
  float best_score = -1e9;
  for (int i = 0; i < NUM_LABELS; i++) {
    float s = (out->type == kTfLiteInt8)
      ? (out->data.int8[i] - out->params.zero_point) * out->params.scale
      : out->data.f[i];
    confidences[i] = s;
    if (s > best_score) { best_score = s; best = i; }
  }
  return best;
}

const char* PetalEdge::getLabel(int i)      { return (i >= 0 && i < NUM_LABELS) ? labels[i] : "?"; }
float       PetalEdge::getConfidence(int i) { return (i >= 0 && i < NUM_LABELS) ? confidences[i] : 0; }
float       PetalEdge::getScore(int i)      { return (i >= 0 && i < NUM_LABELS) ? confidences[i] : 0; }

static void resize_nearest(
    const float* src, int src_w, int src_h, int src_c,
    float* dst, int dst_w, int dst_h, int dst_c
) {
  int channels = (src_c < dst_c) ? src_c : dst_c;
  for (int y = 0; y < dst_h; y++) {
    int src_y = (y * src_h) / dst_h;
    for (int x = 0; x < dst_w; x++) {
      int src_x = (x * src_w) / dst_w;
      for (int c = 0; c < channels; c++) {
        int src_idx = (src_y * src_w + src_x) * src_c + c;
        int dst_idx = (y * dst_w + x) * dst_c + c;
        dst[dst_idx] = src[src_idx];
      }
      for (int c = channels; c < dst_c; c++) {
        int dst_idx = (y * dst_w + x) * dst_c + c;
        dst[dst_idx] = 0.0f;
      }
    }
  }
}

static bool run_dsp(const PetalEdge::Signal* signal, float* out_features, int out_len) {
  if (!signal || !signal->data || signal->length <= 0 || !out_features || out_len <= 0) {
    return false;
  }

  if (std::string(DSP_BLOCK_TYPE) == "raw") {
    int copy_len = signal->length < out_len ? signal->length : out_len;
    for (int i = 0; i < copy_len; i++) {
      out_features[i] = signal->data[i];
    }
    for (int i = copy_len; i < out_len; i++) {
      out_features[i] = 0.0f;
    }
    return true;
  }

  if (std::string(DSP_BLOCK_TYPE) == "image") {
    const int expected_len = IMAGE_WIDTH * IMAGE_HEIGHT * IMAGE_CHANNELS;
    if (expected_len <= 0 || out_len != expected_len) {
      return false;
    }

    int src_len = signal->length;
    if (src_len <= 0) {
      return false;
    }

    if (src_len == expected_len) {
      for (int i = 0; i < expected_len; i++) {
        float v = signal->data[i];
        if (std::string(DSP_NORM_MODE) == "zero_one") {
          v = v / 255.0f;
        } else if (std::string(DSP_NORM_MODE) == "minus_one_one") {
          v = (v / 127.5f) - 1.0f;
        }
        out_features[i] = v;
      }
      return true;
    }

    if (IMAGE_CHANNELS <= 0) {
      return false;
    }

    if ((src_len % IMAGE_CHANNELS) != 0) {
      return false;
    }

    int src_pixels = src_len / IMAGE_CHANNELS;
    if (src_pixels <= 0) {
      return false;
    }

    int src_side = 1;
    while (src_side * src_side < src_pixels) {
      src_side++;
    }
    if (src_side * src_side != src_pixels) {
      return false;
    }

    float resized[IMAGE_WIDTH * IMAGE_HEIGHT * IMAGE_CHANNELS];
    resize_nearest(
      signal->data, src_side, src_side, IMAGE_CHANNELS,
      resized, IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS
    );

    for (int i = 0; i < expected_len; i++) {
      float v = resized[i];
      if (std::string(DSP_NORM_MODE) == "zero_one") {
        v = v / 255.0f;
      } else if (std::string(DSP_NORM_MODE) == "minus_one_one") {
        v = (v / 127.5f) - 1.0f;
      }
      out_features[i] = v;
    }
    return true;
  }

  return false;
}

Prediction PetalEdge::run_classifier(const Signal* signal) {
  Prediction p{};
  if (!signal || !signal->data || signal->length <= 0) {
    p.label = "";
    p.confidence = 0.0f;
    return p;
  }

  float features[DSP_INPUT_LENGTH];
  if (!run_dsp(signal, features, DSP_INPUT_LENGTH)) {
    p.label = "unsupported_dsp";
    p.confidence = 0.0f;
    return p;
  }

  int best = PetalEdge::classify(features, DSP_INPUT_LENGTH);
  p.label = PetalEdge::getLabel(best);
  p.confidence = PetalEdge::getConfidence(best);
  for (int i = 0; i < $NUM_LABELS; i++) {
    p.scores.push_back(PetalEdge::getScore(i));
  }
  return p;
}

$CLASSIFY_RAW_IMPL
""")

_ARDUINO_PROPS = """\
name=PetalEdgeInference
version=1.0.0
author=PetalEdge
sentence=Auto-generated edge inference library
paragraph=Run your PetalEdge model on Arduino devices
category=Signal Input/Output
url=https://github.com/yourorg/petaledge
architectures=*
"""

_ARDUINO_EXAMPLE = Template("""\
#include <PetalEdgeInference.h>

void setup() {
  Serial.begin(115200);
  if (!PetalEdge::begin()) {
    Serial.println("Failed to initialize model!");
    while(1);
  }
  Serial.println("PetalEdge inference ready.");
}

void loop() {
  float features[$INPUT_LENGTH] = {0};  // Provide raw signal/image data here; image DSP supports exact-size input or square nearest-neighbor resize.


  PetalEdge::Signal signal{features, $INPUT_LENGTH};
  auto result = PetalEdge::run_classifier(&signal);

  Serial.print("Prediction: ");
  Serial.print(result.label.c_str());
  Serial.print(" (");
  Serial.print(result.confidence * 100);
  Serial.println("%)");

  delay(500);
}
""")

_ESP32_MAIN_CPP = Template("""\
// Auto-generated by PetalEdge — ESP32 inference
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_log.h"
#include "model.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"

static const char* TAG = "PetalEdge";
static const int ARENA_SIZE  = $TENSOR_ARENA_SIZE;
static uint8_t tensor_arena[ARENA_SIZE];
static const char* labels[] = { $LABELS_ARRAY };
static const int NUM_LABELS  = $NUM_LABELS;
static const int INPUT_LEN   = $INPUT_LENGTH;

extern "C" void app_main(void) {
    const tflite::Model* model = tflite::GetModel(model_tflite);
    tflite::AllOpsResolver resolver;
    tflite::MicroInterpreter interpreter(model, resolver, tensor_arena, ARENA_SIZE);
    interpreter.AllocateTensors();

    TfLiteTensor* input = interpreter.input(0);

    // Fill with dummy features — replace with actual sensor + DSP pipeline
    for (int i = 0; i < INPUT_LEN; i++) {
        if (input->type == kTfLiteInt8) input->data.int8[i] = 0;
        else input->data.f[i] = 0.0f;
    }

    interpreter.Invoke();

    TfLiteTensor* output = interpreter.output(0);
    int best = 0;
    float best_score = -1e9f;
    for (int i = 0; i < NUM_LABELS; i++) {
        float s = (output->type == kTfLiteInt8)
            ? (output->data.int8[i] - output->params.zero_point) * output->params.scale
            : output->data.f[i];
        if (s > best_score) { best_score = s; best = i; }
    }
    ESP_LOGI(TAG, "Prediction: %s (%.1f%%)", labels[best], best_score * 100.0f);
}
""")

_ESP32_CMAIN = """\
idf_component_register(
    SRCS "main.cpp"
    INCLUDE_DIRS "."
    REQUIRES tensorflow-lite-micro
)
"""

_ESP32_CROOT = """\
cmake_minimum_required(VERSION 3.16)
include($ENV{IDF_PATH}/tools/cmake/project.cmake)
project(petaledge_inference)
"""

_ESP32_SDK = """\
CONFIG_ESP_MAIN_TASK_STACK_SIZE=32768
CONFIG_FREERTOS_HZ=1000
"""

_CPP_INFERENCE_H = """\
// Auto-generated by PetalEdge — C++ inference header
#pragma once
#include <cstdint>
#include <string>
#include <vector>

namespace PetalEdge {

struct Prediction {
  std::string label;
  float       confidence;
  std::vector<float> scores;
};
struct Signal {
  const float* data;
  int length;
};

struct Result {
  std::string label;
  float confidence;
  std::vector<float> scores;
};
class Classifier {
public:
  bool init(const char* model_path = nullptr);
  Prediction predict(const float* features, int feature_len);
  int num_classes() const;
  const char* label(int idx) const;

private:
  void* interpreter_ = nullptr;
};

Prediction run_classifier(const Signal* signal);

}  // namespace PetalEdge
"""

_CPP_INFERENCE_CPP = Template("""\
// Auto-generated by PetalEdge — C++ inference implementation (TFLite Micro)
#include "inference.h"
#include "model.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include <cstring>
#include <algorithm>

static const char* LABELS[]     = { $LABELS_ARRAY };
static const int   NUM_LABELS   = $NUM_LABELS;
static const int   INPUT_LENGTH = $INPUT_LENGTH;
static const char* DSP_BLOCK_TYPE   = "$DSP_BLOCK_TYPE";
static const int   DSP_INPUT_LENGTH = $DSP_INPUT_LENGTH;
static const bool  DSP_NORMALIZE    = $DSP_NORMALIZE;
static const int   IMAGE_WIDTH      = $IMAGE_WIDTH;
static const int   IMAGE_HEIGHT     = $IMAGE_HEIGHT;
static const int   IMAGE_CHANNELS   = $IMAGE_CHANNELS;
static const char* DSP_NORM_MODE    = "$DSP_NORM_MODE";

static const int TENSOR_ARENA_SIZE = $TENSOR_ARENA_SIZE;
static uint8_t   tensor_arena[TENSOR_ARENA_SIZE];

static tflite::AllOpsResolver      resolver;
static const tflite::Model*        tfl_model   = nullptr;
static tflite::MicroInterpreter*   interpreter = nullptr;
static TfLiteTensor*               input_tensor = nullptr;

namespace PetalEdge {

bool Classifier::init(const char* /*model_path*/) {
    tfl_model = tflite::GetModel(model_tflite);
    if (tfl_model->version() != TFLITE_SCHEMA_VERSION) return false;
    static tflite::MicroInterpreter static_interp(
        tfl_model, resolver, tensor_arena, TENSOR_ARENA_SIZE);
    interpreter  = &static_interp;
    if (interpreter->AllocateTensors() != kTfLiteOk) return false;
    input_tensor = interpreter->input(0);
    return true;
}

Prediction Classifier::predict(const float* features, int feature_len) {
    // Match Arduino quantization behavior for float32 and int8 TFLite tensors.
    int input_len = input_tensor->bytes / static_cast<int>(sizeof(float));
    if (input_tensor->type == kTfLiteInt8) {
        input_len = input_tensor->bytes;
    }

    int copy_len = feature_len < input_len ? feature_len : input_len;
    for (int i = 0; i < copy_len; i++) {
        if (input_tensor->type == kTfLiteFloat32) {
            input_tensor->data.f[i] = features[i];
        } else if (input_tensor->type == kTfLiteInt8) {
            float scale = input_tensor->params.scale;
            int zero_point = input_tensor->params.zero_point;
            float q = features[i] / scale + zero_point;
            if (q < -128.0f) q = -128.0f;
            if (q > 127.0f) q = 127.0f;
            input_tensor->data.int8[i] = static_cast<int8_t>(q);
        }
    }

    interpreter->Invoke();

    TfLiteTensor* out = interpreter->output(0);

    int num_scores = out->bytes /
        (out->type == kTfLiteInt8 ? 1 : static_cast<int>(sizeof(float)));

    Prediction p;
    for (int i = 0; i < num_scores; i++) {
        float score = 0.0f;
        if (out->type == kTfLiteInt8) {
            score = (out->data.int8[i] - out->params.zero_point)
                    * out->params.scale;
        } else if (out->type == kTfLiteFloat32) {
            score = out->data.f[i];
        }
        p.scores.push_back(score);
    }

    int best = 0;
    for (int i = 1; i < (int)p.scores.size(); ++i)
        if (p.scores[i] > p.scores[best]) best = i;

    p.label      = LABELS[best];
    p.confidence = p.scores[best];
    return p;
}

int         Classifier::num_classes() const { return NUM_LABELS; }
const char* Classifier::label(int i)  const { return (i>=0&&i<NUM_LABELS)?LABELS[i]:"?"; }

static void resize_nearest(
    const float* src, int src_w, int src_h, int src_c,
    float* dst, int dst_w, int dst_h, int dst_c
) {
  int channels = (src_c < dst_c) ? src_c : dst_c;
  for (int y = 0; y < dst_h; y++) {
    int src_y = (y * src_h) / dst_h;
    for (int x = 0; x < dst_w; x++) {
      int src_x = (x * src_w) / dst_w;
      for (int c = 0; c < channels; c++) {
        int src_idx = (src_y * src_w + src_x) * src_c + c;
        int dst_idx = (y * dst_w + x) * dst_c + c;
        dst[dst_idx] = src[src_idx];
      }
      for (int c = channels; c < dst_c; c++) {
        int dst_idx = (y * dst_w + x) * dst_c + c;
        dst[dst_idx] = 0.0f;
      }
    }
  }
}

static bool run_dsp(const Signal* signal, float* out_features, int out_len) {
  if (!signal || !signal->data || signal->length <= 0 || !out_features || out_len <= 0) {
    return false;
  }

  if (std::string(DSP_BLOCK_TYPE) == "raw") {
    int copy_len = signal->length < out_len ? signal->length : out_len;
    for (int i = 0; i < copy_len; i++) {
      out_features[i] = signal->data[i];
    }
    for (int i = copy_len; i < out_len; i++) {
      out_features[i] = 0.0f;
    }
    return true;
  }

  if (std::string(DSP_BLOCK_TYPE) == "image") {
    const int expected_len = IMAGE_WIDTH * IMAGE_HEIGHT * IMAGE_CHANNELS;
    if (expected_len <= 0 || out_len != expected_len) {
      return false;
    }

    int src_len = signal->length;
    if (src_len <= 0) {
      return false;
    }

    if (src_len == expected_len) {
      for (int i = 0; i < expected_len; i++) {
        float v = signal->data[i];
        if (std::string(DSP_NORM_MODE) == "zero_one") {
          v = v / 255.0f;
        } else if (std::string(DSP_NORM_MODE) == "minus_one_one") {
          v = (v / 127.5f) - 1.0f;
        }
        out_features[i] = v;
      }
      return true;
    }

    if (IMAGE_CHANNELS <= 0) {
      return false;
    }

    if ((src_len % IMAGE_CHANNELS) != 0) {
      return false;
    }

    int src_pixels = src_len / IMAGE_CHANNELS;
    if (src_pixels <= 0) {
      return false;
    }

    int src_side = 1;
    while (src_side * src_side < src_pixels) {
      src_side++;
    }
    if (src_side * src_side != src_pixels) {
      return false;
    }

    float resized[IMAGE_WIDTH * IMAGE_HEIGHT * IMAGE_CHANNELS];
    resize_nearest(
      signal->data, src_side, src_side, IMAGE_CHANNELS,
      resized, IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS
    );

    for (int i = 0; i < expected_len; i++) {
      float v = resized[i];
      if (std::string(DSP_NORM_MODE) == "zero_one") {
        v = v / 255.0f;
      } else if (std::string(DSP_NORM_MODE) == "minus_one_one") {
        v = (v / 127.5f) - 1.0f;
      }
      out_features[i] = v;
    }
    return true;
  }

  return false;
}

Prediction run_classifier(const Signal* signal) {
    static Classifier clf;
    static bool initialized = false;
    if (!initialized) { clf.init(); initialized = true; }
    if (!signal || !signal->data || signal->length <= 0) return Prediction{};

    float features[DSP_INPUT_LENGTH];
    if (!run_dsp(signal, features, DSP_INPUT_LENGTH)) {
        Prediction p{};
        p.label      = "unsupported_dsp";
        p.confidence = 0.0f;
        return p;
    }

    return clf.predict(features, DSP_INPUT_LENGTH);
}

}  // namespace PetalEdge
""")

_CPP_MAIN_EXAMPLE = """\
#include "inference.h"
#include <iostream>
#include <vector>

int main() {
    // Provide raw signal/image data here; image DSP supports exact-size input or square nearest-neighbor resize.
    std::vector<float> features(64, 0.0f);

    PetalEdge::Signal signal{features.data(), (int)features.size()};
    auto result = PetalEdge::run_classifier(&signal);

    std::cout << "Label: "        << result.label
              << "  Confidence: " << result.confidence * 100.0f << "%\\n";

    // Parity line — parsed by validate_parity.py to compare with Python/TFLite output.
    std::cout << "PARITY_RESULT " << result.label << " " << result.confidence;
    for (float s : result.scores) { std::cout << " " << s; }
    std::cout << std::endl;

    return 0;
}
"""

_CPP_CMAKE = """\
cmake_minimum_required(VERSION 3.14)
project(petaledge_inference CXX)
set(CMAKE_CXX_STANDARD 17)
find_package(tensorflow-lite-micro REQUIRED)
add_executable(inference src/main_example.cpp src/inference.cpp)
target_include_directories(inference PRIVATE src)
target_link_libraries(inference PRIVATE tensorflow-lite-micro)
"""

_TFLITE_README = b"# PetalEdge TFLite Package\nRun `pip install -r requirements.txt` then `python inference.py`"
_ARDUINO_README = b"# PetalEdge Arduino Library\nExtract into your Arduino libraries folder."
_ESP32_README   = b"# PetalEdge ESP32 Package\nOpen with ESP-IDF: `idf.py build flash monitor`"
_RPI_README     = b"# PetalEdge Raspberry Pi Package\nRun `pip install -r requirements.txt` then `python inference.py`"
_CPP_README     = b"# PetalEdge C++ Package\nSee CMakeLists.txt. Requires TensorFlow Lite C++ library."
