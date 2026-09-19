"""
Real-time inference endpoint â€" REST + WebSocket

Runtime parity: .pe packages are self-contained (like .eim).  All runtime
behaviour (labels, normalization, channel order) is driven exclusively by
metadata packaged inside the file.  model.model_metadata is used for raw
TFLite models that were never wrapped into a .pe archive.
"""
from fastapi import APIRouter, Depends, HTTPException, Query, WebSocket, WebSocketDisconnect
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import Optional
import asyncio
import hashlib
import io
import zipfile
import struct
import numpy as np
import json
import logging
import os

from app.core.database import get_db
from app.core.auth import get_current_user
from app.core.authz import (
    assert_impulse_owner,
    assert_trained_model_owner,
    assert_deployment_owner,
)
from app.core.storage import storage
from app.models.user import User, TrainedModel

router = APIRouter()
logger = logging.getLogger(__name__)

# Fields that every package must declare.  Inference will not start without them.
_REQUIRED_META = (
    "label_names", "input_shape", "normalize_input",
    "channel_order", "model_type", "version",
)

_PE_MAGIC      = b"PEM1"
_PE_HEADER_V1  = struct.Struct("<4sIIIII")   # v1: magic|ver|model|dsp|labels|manifest
_PE_HEADER_V2  = struct.Struct("<4sIIIIII")  # v2: + inference section
_PE_HEADER_STRUCT = _PE_HEADER_V1  # kept for legacy callers

_PXE_MAGIC_BYTES = b"PXE1"

def _is_pxe(raw_bytes: bytes) -> bool:
    return raw_bytes[:4] == _PXE_MAGIC_BYTES


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
        import tflite_runtime.interpreter as tflite
        Interpreter = tflite.Interpreter
    except ImportError:
        import tensorflow as _tf
        Interpreter = _tf.lite.Interpreter
    return Interpreter(**kwargs)


class InferenceRequest(BaseModel):
    model_config = {"protected_namespaces": ()}
    model_id: str
    features: list            # flat float list
    fomo_threshold: Optional[float] = None  # runtime-only override; None = use env/metadata


# â"€â"€â"€ package helpers â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€

def _require_meta_fields(meta: dict, source: str) -> None:
    """Raise if any required metadata field is absent (fail fast)."""
    missing = [k for k in _REQUIRED_META if k not in meta]
    if missing:
        raise ValueError(
            f"Package metadata from {source!r} is missing required fields: {missing}. "
            "Rebuild the package so that manifest.json is complete."
        )


def _verify_model_hash(tflite_bytes: bytes, meta: dict) -> None:
    """If manifest declares model_hash, verify the loaded model bytes match it.

    Expected format: ``"sha256:<lowercase-hex-digest>"``.
    Raises ValueError on algorithm mismatch, format error, or digest mismatch.
    Silently skips when the field is absent (hash is optional).
    """
    declared = meta.get("model_hash")
    if not declared:
        return
    if not isinstance(declared, str) or not declared.startswith("sha256:"):
        raise ValueError(
            f"Unsupported model_hash format in manifest: {declared!r}. "
            "Expected 'sha256:<hex>'. Rebuild the package."
        )
    expected = declared[len("sha256:"):]
    actual   = hashlib.sha256(tflite_bytes).hexdigest()
    if actual != expected:
        raise ValueError(
            "Package integrity error: model.tflite hash mismatch — "
            f"expected sha256:{expected!r}, computed sha256:{actual!r}. "
            "The package may have been tampered with. Rebuild the package."
        )


def _load_pe_package(pe_bytes: bytes) -> tuple:
    """Extract (tflite_bytes, meta) from a .pe package (binary PEM1 or legacy ZIP)."""
    # ZIP fallback: if binary magic is absent, try the legacy zip-based format
    # so old packages remain loadable without caller changes.
    if pe_bytes[:4] != _PE_MAGIC:
        if pe_bytes[:2] == b"PK":
            logger.warning(
                "Loading legacy ZIP-format .pe package; rebuild to binary format."
            )
            return _load_pe_zip(pe_bytes)
        raise ValueError(
            f"Unrecognised .pe format: first 4 bytes are {pe_bytes[:4]!r}. "
            "Rebuild the package."
        )

    # ── Binary (PEM1) path — version-safe ────────────────────────────────────
    if len(pe_bytes) < 8:
        raise ValueError("Malformed .pe package: header is truncated. Rebuild the package.")

    _, peek_ver = struct.unpack_from("<4sI", pe_bytes, 0)

    if peek_ver == 1:
        hdr = _PE_HEADER_V1
        if len(pe_bytes) < hdr.size:
            raise ValueError("Malformed .pe package: v1 header truncated. Rebuild the package.")
        magic, format_version, model_size, dsp_config_size, labels_size, manifest_size = (
            hdr.unpack_from(pe_bytes, 0)
        )
        inference_size = 0
    elif peek_ver == 2:
        hdr = _PE_HEADER_V2
        if len(pe_bytes) < hdr.size:
            raise ValueError("Malformed .pe package: v2 header truncated. Rebuild the package.")
        magic, format_version, model_size, dsp_config_size, labels_size, manifest_size, inference_size = (
            hdr.unpack_from(pe_bytes, 0)
        )
    else:
        raise ValueError(
            f"Unsupported .pe format_version={peek_ver}. "
            "Only v1 and v2 are supported. Rebuild the package."
        )

    if magic != _PE_MAGIC:
        raise ValueError(
            f"Malformed .pe package: unexpected magic {magic!r}. Rebuild the package."
        )

    # Fail fast: every section must be present (non-zero length).
    _section_sizes = {
        "model":      model_size,
        "dsp_config": dsp_config_size,
        "labels":     labels_size,
        "manifest":   manifest_size,
    }
    missing_sections = [name for name, sz in _section_sizes.items() if sz == 0]
    if missing_sections:
        raise ValueError(
            f"Malformed .pe package: required section(s) are empty: {missing_sections}. "
            "Rebuild the package."
        )

    expected_size = (
        hdr.size
        + model_size
        + dsp_config_size
        + labels_size
        + manifest_size
        + inference_size
    )
    if len(pe_bytes) != expected_size:
        raise ValueError(
            f"Malformed .pe package: expected {expected_size} bytes, got {len(pe_bytes)}. "
            "Rebuild the package."
        )

    offset = hdr.size
    tflite_bytes = pe_bytes[offset:offset + model_size]
    offset += model_size
    dsp_cfg_bytes = pe_bytes[offset:offset + dsp_config_size]
    offset += dsp_config_size
    labels_bytes = pe_bytes[offset:offset + labels_size]
    offset += labels_size
    manifest_bytes = pe_bytes[offset:offset + manifest_size]
    # inference section (v2) is informational only at runtime; skip past it.

    meta = json.loads(manifest_bytes)
    dsp_cfg = json.loads(dsp_cfg_bytes)
    labels_txt = [ln for ln in labels_bytes.decode().splitlines() if ln.strip()]

    # Validate required manifest fields first so cross-checks can use direct
    # key access and produce unambiguous errors (not "field mismatch" when the
    # real problem is a missing field).
    _require_meta_fields(meta, "manifest.json")

    # Cross-validation 1: labels.txt must exactly match manifest["label_names"].
    if labels_txt != meta["label_names"]:
        raise ValueError(
            f"Package integrity error: labels.txt {labels_txt!r} does not match "
            f"manifest.json label_names {meta['label_names']!r}. Rebuild the package."
        )

    # Cross-validation 2: manifest["version"] must match dsp_config["version"].
    dsp_ver = dsp_cfg.get("version")
    if meta["version"] != dsp_ver:
        raise ValueError(
            f"Package integrity error: manifest.json version={meta['version']!r} does not "
            f"match dsp_config.json version={dsp_ver!r}. Rebuild the package."
        )

    # Step 7: optional model hash integrity check — fail on tamper/mismatch.
    _verify_model_hash(tflite_bytes, meta)

    # Attach dsp_config so callers have full DSP block visibility without
    # re-parsing the package.
    meta["dsp_config"] = dsp_cfg

    return tflite_bytes, meta


def _load_pe_zip(pe_bytes: bytes) -> tuple:
    """Load a legacy ZIP-format .pe package (fallback for pre-binary packages).

    Requires the same four files as the binary format:
    model.tflite, dsp_config.json, labels.txt, manifest.json.
    Applies identical validation so runtime behaviour is unchanged.
    """
    try:
        zf = zipfile.ZipFile(io.BytesIO(pe_bytes))
    except zipfile.BadZipFile as exc:
        raise ValueError(
            f"Malformed .pe package: not a valid ZIP archive — {exc}. "
            "Rebuild the package."
        ) from exc

    names         = set(zf.namelist())
    required_files = {"model.tflite", "dsp_config.json", "labels.txt", "manifest.json"}
    missing_files  = required_files - names
    if missing_files:
        raise ValueError(
            f"Legacy .pe ZIP is missing required file(s): {sorted(missing_files)}. "
            "Rebuild the package."
        )

    tflite_bytes   = zf.read("model.tflite")
    dsp_cfg_bytes  = zf.read("dsp_config.json")
    labels_bytes   = zf.read("labels.txt")
    manifest_bytes = zf.read("manifest.json")

    meta       = json.loads(manifest_bytes)
    dsp_cfg    = json.loads(dsp_cfg_bytes)
    labels_txt = [ln for ln in labels_bytes.decode().splitlines() if ln.strip()]

    _require_meta_fields(meta, "manifest.json")

    if labels_txt != meta["label_names"]:
        raise ValueError(
            f"Package integrity error: labels.txt {labels_txt!r} does not match "
            f"manifest.json label_names {meta['label_names']!r}. Rebuild the package."
        )

    dsp_ver = dsp_cfg.get("version")
    if meta["version"] != dsp_ver:
        raise ValueError(
            f"Package integrity error: manifest.json version={meta['version']!r} does not "
            f"match dsp_config.json version={dsp_ver!r}. Rebuild the package."
        )

    _verify_model_hash(tflite_bytes, meta)

    meta["dsp_config"] = dsp_cfg
    return tflite_bytes, meta


def _resolve_model(model: TrainedModel, raw_bytes=None) -> tuple:
    """Return (tflite_bytes, meta) from either a raw TFLite or a .pe package.

    ``raw_bytes`` lets a caller hand in already-downloaded model bytes (e.g. when
    the bytes were fetched to sniff the format) so we don't download twice.
    """
    if raw_bytes is None:
        raw_bytes = storage.download_bytes(model.storage_key)

    if model.format == "pe" or (model.storage_key or "").endswith(".pe"):
        tflite_bytes, meta = _load_pe_package(raw_bytes)
        logger.debug("Loaded metadata from manifest.json inside .pe package")
        return tflite_bytes, meta

    # Raw TFLite path: metadata lives in the DB record.
    meta = model.model_metadata or {}
    _require_meta_fields(meta, "model.model_metadata")
    return raw_bytes, meta


# â"€â"€â"€ endpoints â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€

@router.post("/predict")
async def predict(
    req: InferenceRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Run inference using a stored TFLite model, .pe package, or .pxe binary."""
    from app.models.user import Deployment

    model = db.query(TrainedModel).filter(
        TrainedModel.id == req.model_id,
        TrainedModel.format.in_(["tflite", "pe"]),
    ).first()

    # .pxe packages are stored on the Deployment record, not the TrainedModel.
    # Accept either a TrainedModel.id or a Deployment.id in req.model_id.
    # Ownership is enforced on whichever artifact resolves (404 on non-owned).
    storage_key = None
    if model is not None:
        assert_trained_model_owner(db, model.id, current_user)
        storage_key = model.storage_key
    if not storage_key:
        dep = db.query(Deployment).filter(Deployment.id == req.model_id).first()
        if dep:
            assert_deployment_owner(db, dep.id, current_user)
            storage_key = dep.storage_key

    if not storage_key:
        raise HTTPException(404, "Model not found (tflite, pe, or pxe format required)")

    loop = asyncio.get_event_loop()
    # Blocking boto3 download — run it in the threadpool so the event loop is not
    # stalled for the full fetch + inference.
    raw_bytes = await loop.run_in_executor(None, storage.download_bytes, storage_key)

    # .pxe — out-of-process path, fully separate from .pe
    if _is_pxe(raw_bytes):
        from app.ml.pxe_runner import run_pxe_inference
        raw_result = await loop.run_in_executor(
            None, run_pxe_inference, raw_bytes, req.features
        )
        return _normalize_pxe_classify_result(raw_result)

    # .pe / raw tflite — existing in-process path
    if not model:
        raise HTTPException(404, "Model not found (tflite or pe format required)")

    # Reuse the bytes we already fetched when they are the model's own bytes,
    # avoiding _resolve_model's second download. CPU-bound TFLite inference is
    # also offloaded so it never blocks the event loop.
    _preloaded = raw_bytes if storage_key == model.storage_key else None

    def _resolve_and_run():
        tflite_bytes, meta = _resolve_model(model, raw_bytes=_preloaded)
        return _run_tflite(tflite_bytes, req.features, meta, fomo_threshold=req.fomo_threshold)

    return await loop.run_in_executor(None, _resolve_and_run)


@router.websocket("/ws/{model_id}")
async def inference_websocket(websocket: WebSocket, model_id: str,
                               token: Optional[str] = Query(None),
                               db: Session = Depends(get_db)):
    """
    WebSocket endpoint for streaming real-time inference.

    Auth: a JWT must be supplied as the ``token`` query parameter (browsers
    cannot set Authorization headers on a WS handshake). Invalid/missing tokens
    and non-owned artifacts close the socket with 1008 (policy violation).

    Client sends: {"features": [1.0, 2.0, ...]}
    Server replies: {"label": "walking", "confidence": 0.95, "scores": {...}}
    """
    from app.models.user import Deployment
    from app.core.auth import get_user_from_token

    await websocket.accept()

    current_user = get_user_from_token(db, token)
    if current_user is None:
        await websocket.send_json({"error": "Authentication required"})
        await websocket.close(code=1008)
        return

    logger.info(f"WebSocket inference opened for model {model_id}")

    # Resolve the artifact and enforce ownership (TrainedModel or Deployment).
    model = db.query(TrainedModel).filter(
        TrainedModel.id == model_id,
        TrainedModel.format.in_(["tflite", "pe"]),
    ).first()

    storage_key = None
    try:
        if model is not None:
            assert_trained_model_owner(db, model.id, current_user)
            storage_key = model.storage_key
        if not storage_key:
            dep = db.query(Deployment).filter(Deployment.id == model_id).first()
            if dep:
                assert_deployment_owner(db, dep.id, current_user)
                storage_key = dep.storage_key
    except HTTPException:
        storage_key = None

    if not storage_key:
        await websocket.send_json({"error": "Model not found"})
        await websocket.close(code=1008)
        return

    raw_bytes = storage.download_bytes(storage_key)

    # .pxe — launch once, keep alive for the full WebSocket session
    if _is_pxe(raw_bytes):
        from app.ml.pxe_runner import PxeProcess
        loop = asyncio.get_event_loop()
        try:
            pxe = await loop.run_in_executor(None, PxeProcess, raw_bytes)
        except Exception as exc:
            await websocket.send_json({"error": str(exc)})
            await websocket.close()
            return
        try:
            await websocket.send_json(pxe.hello)
            while True:
                try:
                    frame = await websocket.receive_json()
                except Exception:
                    break
                features = frame.get("features", [])
                raw_result = await loop.run_in_executor(None, pxe.classify, features)
                raw_result["_pxe_manifest"] = pxe.manifest
                await websocket.send_json(_normalize_pxe_classify_result(raw_result))
        finally:
            await loop.run_in_executor(None, pxe._cleanup)
        return

    # .pe / raw tflite — existing streaming loop
    if not model:
        await websocket.send_json({"error": "Model not found"})
        await websocket.close()
        return

    try:
        tflite_bytes, meta = _resolve_model(model)
    except Exception as e:
        await websocket.send_json({"error": f"Package load failed: {e}"})
        await websocket.close()
        return

    try:
        while True:
            raw = await websocket.receive_text()
            data = json.loads(raw)
            features = data.get("features", [])
            if not features:
                await websocket.send_json({"error": "No features provided"})
                continue
            fomo_thr = data.get("fomo_threshold")
            if fomo_thr is not None:
                fomo_thr = float(fomo_thr)
            result = _run_tflite(tflite_bytes, features, meta, fomo_threshold=fomo_thr)
            await websocket.send_json(result)
    except WebSocketDisconnect:
        logger.info(f"WebSocket inference closed for model {model_id}")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        await websocket.send_json({"error": str(e)})


# ─── core inference ───────────────────────────────────────────────────────────

def _normalize_pxe_classify_result(result: dict) -> dict:
    """Convert raw PXE runner output (EI stdio-JSONL format) to the
    frontend-expected LiveClassificationRunResult shape.

    The PXE runner speaks EI's own protocol:
      - classification → {"classification": {"label": score, ...}}
      - FOMO           → {"bounding_boxes": [{"label","value","x","y","width","height"}, ...]}
      - YOLO Pro (new runner) → {"detections": [...], "is_detection": True, ...}

    Pops ``_pxe_manifest`` from *result* (inserted by run_pxe_inference).
    """
    manifest = result.pop("_pxe_manifest", {})
    manifest_model_type: str = (
        manifest.get("model_type")
        or manifest.get("output_type")
        or "classification"
    )

    # Already in frontend format (new runner with _postprocess_yolo_pro)
    if "detections" in result or "is_detection" in result:
        out = {k: v for k, v in result.items() if not k.startswith("_")}
        out.setdefault("is_detection", bool(result.get("detections") is not None))
        out.setdefault("is_fomo", False)
        out.setdefault("model_type", manifest_model_type)
        return out

    # EI bounding_boxes format (FOMO PXE runner)
    if "bounding_boxes" in result:
        input_shape = manifest.get("input_shape", [96, 96, 3])
        img_h = int(input_shape[0]) if len(input_shape) >= 1 else 96
        img_w = int(input_shape[1]) if len(input_shape) >= 2 else 96
        detections = []
        for bb in result.get("bounding_boxes", []):
            px = float(bb.get("x", 0))
            py = float(bb.get("y", 0))
            pw = float(bb.get("width", 0))
            ph = float(bb.get("height", 0))
            detections.append({
                "label": bb.get("label", ""),
                "confidence": float(bb.get("value", bb.get("confidence", 0.0))),
                "bbox": {
                    "x1": px / img_w,
                    "y1": py / img_h,
                    "x2": (px + pw) / img_w,
                    "y2": (py + ph) / img_h,
                },
            })
        return {
            "detections": detections,
            "count": len(detections),
            "is_fomo": True,
            "is_detection": False,
            "model_type": "detection_heatmap",
            "debug": result.get("debug", {}),
        }

    # EI classification format
    if "classification" in result:
        cls: dict = result["classification"] or {}
        if cls:
            top_label = max(cls, key=lambda k: cls[k])
            top_conf = float(cls[top_label])
        else:
            top_label = None
            top_conf = 0.0

        # Manifest says YOLO Pro but old runner returned classification output —
        # stale PXE artifact. Surface as an empty detection set so the frontend
        # renders detection mode (with 0 boxes) rather than a meaningless
        # classification table.
        if manifest_model_type == "yolo_pro_detection":
            return {
                "detections": [],
                "count": 0,
                "is_fomo": False,
                "is_detection": True,
                "model_type": "yolo_pro_detection",
                "debug": {"stale_pxe_artifact": True},
            }

        return {
            "label": top_label,
            "confidence": top_conf,
            "scores": cls,
            "is_fomo": False,
            "is_detection": False,
            "model_type": manifest_model_type,
        }

    # Fallback: strip private keys, inject missing flags
    out = {k: v for k, v in result.items() if not k.startswith("_")}
    out.setdefault("is_fomo", False)
    out.setdefault("is_detection", False)
    out.setdefault("model_type", manifest_model_type)
    return out


def _is_fomo_model(meta: dict) -> bool:
    """Return True when metadata identifies this as a FOMO/detection_heatmap model."""
    return (
        meta.get("model_type") == "detection_heatmap"
        or meta.get("output_type") == "detection_heatmap"
        or "fomo" in meta.get("architecture", "").lower()
    )


def _is_yolo_pro_model(meta: dict) -> bool:
    """Return True when metadata identifies this as a YOLO Pro detection model."""
    return (
        meta.get("model_type") == "yolo_pro_detection"
        or meta.get("output_type") == "yolo_pro_detection"
        or "yolo_pro" in meta.get("architecture", "").lower()
    )


def _ground_truth_label(sample) -> Optional[str]:
    """Return the ground-truth label for a sample.

    For detection/FOMO samples the canonical label lives in
    extra_metadata.boundingBoxes[0].label, not on sample.label (which may be
    empty or hold only the top-level folder label).  Classification samples
    fall through to sample.label as before.
    """
    meta = sample.extra_metadata or {}
    boxes = meta.get("boundingBoxes") if isinstance(meta, dict) else None
    if isinstance(boxes, list):
        for box in boxes:
            if isinstance(box, dict):
                name = (box.get("label") or "").strip()
                if name:
                    return name
    return sample.label.name if sample.label else None


def _ground_truth_boxes(sample) -> list:
    """Return the sample's ground-truth bounding boxes in pixel coordinates.

    Annotations stored during Data Labeling live on
    extra_metadata.boundingBoxes as {x, y, w, h, label, label_id}, with x/y
    in pixel-space relative to the original image. The Live Classification
    panel renders these alongside model predictions so the "Raw data" frame
    can show the annotation the user authored and the "Classification
    result" frame can show the model's prediction.

    Returns [] when the sample carries no usable annotations.
    """
    meta = sample.extra_metadata or {}
    boxes = meta.get("boundingBoxes") if isinstance(meta, dict) else None
    if not isinstance(boxes, list):
        return []
    out = []
    for box in boxes:
        if not isinstance(box, dict):
            continue
        # Accept both canonical {w,h} and EI legacy {width,height}; the UI
        # editor writes {w,h} but imported samples may still carry the legacy
        # keys before the next normalize-on-write.
        x = box.get("x")
        y = box.get("y")
        w = box.get("w", box.get("width"))
        h = box.get("h", box.get("height"))
        label = (box.get("label") or "").strip()
        if None in (x, y, w, h) or not label:
            continue
        try:
            out.append({
                "label": label,
                "x": float(x),
                "y": float(y),
                "w": float(w),
                "h": float(h),
            })
        except (TypeError, ValueError):
            continue
    return out


def _raw_image_pixels(image_bytes: bytes, channel_order: str = "rgb") -> list:
    """Decode image bytes to a flat float32 list of raw pixel values (0–255).

    Used for .pxe image-model inference so the runner's embedded _dsp_image
    pipeline can apply the manifest's resize mode, normalization, and channel
    conversion.  Sending already-DSP'd features would double-apply those
    transforms and corrupt FOMO grid spatial semantics.
    """
    try:
        from PIL import Image as _PILImage
        img = _PILImage.open(io.BytesIO(image_bytes))
        img = img.convert("L" if str(channel_order).lower() == "grayscale" else "RGB")
        return np.array(img, dtype=np.float32).flatten().tolist()
    except Exception:
        return np.frombuffer(image_bytes, dtype=np.uint8).astype(np.float32).tolist()


def _raw_image_tensor(image_bytes: bytes, channel_order: str = "rgb") -> list:
    """Decode image bytes to a shape-preserving JSON-serializable pixel tensor."""
    try:
        from PIL import Image as _PILImage
        img = _PILImage.open(io.BytesIO(image_bytes))
        img = img.convert("L" if str(channel_order).lower() == "grayscale" else "RGB")
        arr = np.array(img, dtype=np.float32)
        if arr.ndim == 2:
            arr = arr[:, :, np.newaxis]
        return arr.tolist()
    except Exception:
        return _raw_image_pixels(image_bytes, channel_order)


def _nms_yolo(
    detections: list,
    iou_threshold: float = 0.45,
    max_dets: int = 100,
    class_agnostic: bool = False,
) -> list:
    """Greedy NMS for YOLO-Pro detections.

    Each detection is {'class_idx': int, 'score': float, 'box': [x1,y1,x2,y2]}.
    Kept local to avoid importing TF (mirrors yolo_pro_worker._nms).

    class_agnostic=False (default): suppress within each class bucket only.
    class_agnostic=True: suppress across all classes — a single physical object
        predicted as both 'bus' and 'car' collapses to the highest-scoring label.
        Use this for live classification display.
    """
    if not detections:
        return []

    def _iou(a: list, b: list) -> float:
        ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
        ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
        inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
        area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
        union = area_a + area_b - inter
        return inter / union if union > 0 else 0.0

    def _greedy(pool: list) -> list:
        result: list = []
        while pool:
            best = pool[0]
            result.append(best)
            pool = [d for d in pool[1:] if _iou(best["box"], d["box"]) < iou_threshold]
        return result

    if class_agnostic:
        pool = sorted(detections, key=lambda d: d["score"], reverse=True)
        kept = _greedy(pool)
    else:
        by_class: dict = {}
        for d in detections:
            by_class.setdefault(d["class_idx"], []).append(d)
        kept: list = []
        for cls_dets in by_class.values():
            pool = sorted(cls_dets, key=lambda d: d["score"], reverse=True)
            kept.extend(_greedy(pool))

    kept.sort(key=lambda d: d["score"], reverse=True)
    return kept[:max_dets]


def _apply_manifest_input_contract(
    x: np.ndarray,
    meta: dict,
    *,
    skip_if_pre_normalized: bool = False,
) -> np.ndarray:
    """Apply manifest-driven channel order and normalization to an input tensor."""
    channel_order = meta["channel_order"].lower()
    normalize = bool(meta["normalize_input"])
    input_mean = float(meta.get("input_mean", 0.0))
    input_std = float(meta.get("input_std", 255.0))

    if channel_order not in ("rgb", "bgr"):
        raise ValueError(
            f"channel_order in package metadata is '{channel_order}'; "
            "only 'rgb' and 'bgr' are supported."
        )
    if channel_order == "bgr" and x.ndim >= 3 and x.shape[-1] == 3:
        x = x[..., ::-1].copy()

    if normalize:
        already_normalized = (
            skip_if_pre_normalized
            and x.size > 0
            and float(np.max(x)) <= 1.0
        )
        if not already_normalized:
            denom = input_std if input_std != 0.0 else 255.0
            x = (x - input_mean) / denom

    return x


def _postprocess_output(raw_output: np.ndarray, meta: dict, debug: dict,
                        threshold_override: Optional[float] = None) -> dict:
    """Pure post-processing: route to FOMO or classification branch.

    Separated from TFLite execution so it can be unit-tested without a real
    interpreter.  Callers must pass a dequantized float32 array.

    threshold_override: per-request FOMO threshold from the UI slider.
    Priority: threshold_override > model metadata threshold.
    Model metadata threshold is set by training (F1-optimal sweep) — same as Edge Impulse.
    Never affects training thresholds or saved metrics.
    """
    label_names = meta["label_names"]

    # ── FOMO / detection_heatmap branch ──────────────────────────────────────
    # Output tensor is (1, grid_H, grid_W, num_classes+1).
    # Channel 0 is background; channels 1..N map to label_names[0..N-1].
    # We find the argmax over object channels per cell and keep cells that
    # exceed the confidence threshold — no softmax over the flattened vector.
    # fomo_version is read from metadata for logging/validation only.
    # v1 path (unchanged): grid derived from tensor shape (stride-16 → 6×6 at 96×96).
    # v2 path: same grid derivation from tensor shape (stride-8 → 16×16 at 128×128).
    # Guard: all v2-specific logic is behind the fomo_version == 2 check below.
    if _is_fomo_model(meta):
        from app.ml.fomo_evaluator import decode_fomo_heatmap

        # Model metadata threshold is the source of truth (written by training sweep).
        # Per-request override (UI slider) is the only way to change it at runtime.
        threshold = float(meta.get("threshold", 0.5))
        if threshold_override is not None:
            threshold = float(threshold_override)

        # Support both (1, H, W, C+1) and already-squeezed (H, W, C+1) tensors.
        fomo_output = np.asarray(raw_output, dtype=np.float32)
        if fomo_output.ndim == 4:
            fomo_output = fomo_output[0]
        if fomo_output.ndim != 3:
            raise ValueError(
                f"Unexpected FOMO output shape: {tuple(np.shape(raw_output))}; "
                "expected (H, W, C+1) or (1, H, W, C+1)."
            )

        gh, gw, _n_all = fomo_output.shape
        pred_cells, probs = decode_fomo_heatmap(
            fomo_output,
            threshold,
            min_peak_gap=0.0,
            fomo_version=int(meta.get("fomo_version", 1)),
            grid_size=int(meta.get("grid_size", 0) or 0),
        )
        detections = []
        for (row, col), cls_idx in pred_cells.items():
            conf = float(probs[row, col, cls_idx + 1])  # +1 skips background channel
            cx, cy = (col + 0.5) / gw, (row + 0.5) / gh
            hw, hh = 0.5 / gw, 0.5 / gh
            detections.append({
                "label":      label_names[cls_idx] if cls_idx < len(label_names) else str(cls_idx),
                "confidence": conf,
                "bbox": {
                    "x1": max(0.0, cx - hw),
                    "y1": max(0.0, cy - hh),
                    "x2": min(1.0, cx + hw),
                    "y2": min(1.0, cy + hh),
                },
            })
        _fomo_ver = int(meta.get("fomo_version", 1))
        _grid_size_stored = meta.get("grid_size", 0)
        return {
            "detections":  detections,
            "count":       len(detections),
            "model_type":  "detection_heatmap",
            "is_fomo":     True,
            "fomo_version": _fomo_ver,
            "debug":       {
                **debug,
                # object_max: highest score across all object channels (background
                # excluded).  Compare against threshold to diagnose count=0 cases:
                # if object_max > threshold → suppressed by local-max logic;
                # if object_max < threshold → model not activating or threshold too high.
                "object_max":   float(probs[:, :, 1:].max()),
                "threshold":    threshold,
                "fomo_version": _fomo_ver,
                "grid_h":       gh,
                "grid_w":       gw,
                "grid_size_stored": _grid_size_stored,
            },
        }

    # ── YOLO Pro detection branch ─────────────────────────────────────────────
    # Exported YOLO Pro TFLite output is already decoded: (1, N_total, 6) where
    # each row is [x1, y1, x2, y2, score, class_id] with coords normalised 0-1.
    # Apply confidence threshold + per-class NMS to produce the detection list.
    if _is_yolo_pro_model(meta):
        threshold = float(meta.get("threshold", 0.25))
        if threshold_override is not None:
            threshold = float(threshold_override)
        iou_threshold = float(meta.get("iou_threshold", 0.45))

        # Support decoded YOLO tensors with extra singleton dims:
        #   (1, N, 6), (N, 6), (1, 1, N, 6), ...
        dets_np = np.asarray(raw_output, dtype=np.float32)
        dets_np = np.squeeze(dets_np)
        if dets_np.ndim != 2 or dets_np.shape[-1] != 6:
            raise ValueError(
                f"Unexpected YOLO Pro decoded output shape: {tuple(np.shape(raw_output))}; "
                "expected (N, 6) after squeezing singleton dimensions."
            )

        raw_dets = []
        for row in dets_np:
            x1, y1, x2, y2 = float(row[0]), float(row[1]), float(row[2]), float(row[3])
            score    = float(row[4])
            class_id = int(row[5])
            if score < threshold:
                continue
            if x2 <= x1 or y2 <= y1:
                continue
            raw_dets.append({"class_idx": class_id, "score": score, "box": [x1, y1, x2, y2]})

        kept = _nms_yolo(raw_dets, iou_threshold=iou_threshold, class_agnostic=True)
        detections = [
            {
                "label":      label_names[d["class_idx"]] if d["class_idx"] < len(label_names) else str(d["class_idx"]),
                "confidence": d["score"],
                "bbox": {
                    "x1": d["box"][0], "y1": d["box"][1],
                    "x2": d["box"][2], "y2": d["box"][3],
                },
            }
            for d in kept
        ]
        return {
            "detections":   detections,
            "count":        len(detections),
            "model_type":   "yolo_pro_detection",
            "is_fomo":      False,
            "is_detection": True,
            "debug":        {
                **debug,
                "threshold":     threshold,
                "iou_threshold": iou_threshold,
                "total_anchors": len(dets_np),
            },
        }

    # ── Classification branch ─────────────────────────────────────────────────
    # Flatten, apply softmax, return top label + full per-class score dict.
    output = raw_output.flatten()
    e = np.exp(output - output.max())
    scores = e / e.sum()
    best = int(np.argmax(scores))
    return {
        "label":        label_names[best] if best < len(label_names) else str(best),
        "confidence":   float(scores[best]),
        "scores":       {
            label_names[i] if i < len(label_names) else str(i): float(scores[i])
            for i in range(len(scores))
        },
        "model_type":   meta.get("model_type", "classification"),
        "is_fomo":      False,
        "is_detection": False,
        "debug":        debug,
    }


# ─── classify-sample endpoints ───────────────────────────────────────────────

class ClassifySampleRequest(BaseModel):
    model_config = {"protected_namespaces": ()}
    impulse_id: str
    sample_id: str
    model_id: Optional[str] = None
    fomo_threshold: Optional[float] = None


def _load_cached_feature_sample_ids(impulse) -> set[str]:
    """
    Fall back to the DSP feature cache when per-sample FeatureSet rows do not exist.

    The current DSP worker writes features.npz but does not populate feature_sets,
    so live classification must understand both representations.
    """
    if not impulse:
        return set()

    try:
        cache_key = storage.model_key(impulse.project_id, impulse.id, "features.npz")
        data_bytes = storage.download_bytes(cache_key)
        with np.load(io.BytesIO(data_bytes), allow_pickle=True) as data:
            ids = data.get("ids", [])
        return {
            str(sample_id)
            for sample_id in ids
            if sample_id is not None and str(sample_id).strip()
        }
    except Exception:
        return set()


@router.get("/sample-options")
def get_sample_options(
    impulse_id: str = Query(..., description="Impulse ID to scope samples to"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Return project test-split samples for the impulse.

    Live Classification preprocesses raw sample bytes on demand during
    ``classify-sample``, so newly uploaded testing samples should be selectable
    immediately without requiring a prior "Generate Features" run or FeatureSet
    cache entry.
    """
    from app.models.user import Impulse, Sample, SampleType

    impulse = assert_impulse_owner(db, impulse_id, current_user)

    samples = (
        db.query(Sample)
        .filter(
            Sample.project_id == impulse.project_id,
            Sample.sample_type == SampleType.testing,
        )
        .order_by(Sample.created_at.desc())
        .limit(200)
        .all()
    )

    return [
        {
            "id": s.id,
            "name": s.filename,
            "label": _ground_truth_label(s) or "—",
            "sensor_type": s.sensor_type,
            "duration_ms": s.duration_ms,
        }
        for s in samples
    ]


def _sample_in_impulse_scope(sample_id: str, impulse_id: str, db: Session) -> bool:
    """Return True when the sample belongs to the same project as the impulse.

    ``classify-sample`` always preprocesses the raw sample bytes on demand, so
    project ownership is the relevant safety boundary. Requiring FeatureSet or
    cache membership prevents newly uploaded testing samples from being used.
    """
    from app.models.user import Impulse, Sample

    impulse = db.query(Impulse).filter(Impulse.id == impulse_id).first()
    if not impulse:
        return False

    sample = db.query(Sample).filter(Sample.id == sample_id).first()
    if not sample:
        return False

    return sample.project_id == impulse.project_id


@router.get("/preferred-runtime")
def get_preferred_runtime(
    impulse_id: str = Query(..., description="Impulse ID to find the best artifact for"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Return the best available live-classification runtime artifact for the impulse.

    Priority: PXE > PE > TFLite
    - PXE: a completed Deployment with target=pxe (most optimised, self-contained)
    - PE:  a TrainedModel with format=pe from the latest completed TrainingJob
    - TFLite: a TrainedModel with format=tflite from the latest completed TrainingJob

    Frontend uses this as the single bootstrap call instead of the hardcoded
    tflite→pe chain, making PXE reachable from the normal page flow.

    Returns 404 when no artifact is available for the impulse.
    """
    from app.models.user import Impulse, TrainingJob, JobStatus, Deployment

    impulse = assert_impulse_owner(db, impulse_id, current_user)

    # ── Priority 1: PXE Deployment ────────────────────────────────────────────
    pxe_candidates = (
        db.query(Deployment)
        .join(TrainedModel, Deployment.model_id == TrainedModel.id)
        .join(TrainingJob, TrainedModel.training_job_id == TrainingJob.id)
        .filter(
            TrainingJob.impulse_id == impulse_id,
            Deployment.status == JobStatus.completed,
            Deployment.storage_key.isnot(None),
        )
        .order_by(Deployment.created_at.desc())
        .all()
    )
    pxe_dep = next(
        (
            dep for dep in pxe_candidates
            if (dep.options or {}).get("deployment_format") == "pxe"
        ),
        None,
    )
    if pxe_dep:
        tm = db.query(TrainedModel).filter(TrainedModel.id == pxe_dep.model_id).first()
        meta = (tm.model_metadata or {}) if tm else {}
        job  = (
            db.query(TrainingJob).filter(TrainingJob.id == tm.training_job_id).first()
            if tm else None
        )
        return {
            "id":            pxe_dep.id,
            "artifact_type": "pxe",
            "display_name":  "PXE Runtime",
            "architecture":  meta.get("architecture", "pxe"),
            "version":       tm.version if tm else None,
            "best_accuracy": job.best_accuracy if job else None,
        }

    # ── Priority 2/3: PE then TFLite from latest completed training job ───────
    job = (
        db.query(TrainingJob)
        .filter(
            TrainingJob.impulse_id == impulse_id,
            TrainingJob.status == JobStatus.completed,
        )
        .order_by(TrainingJob.created_at.desc())
        .first()
    )
    if job:
        pe_model = (
            db.query(TrainedModel)
            .filter(
                TrainedModel.training_job_id == job.id,
                TrainedModel.format == "pe",
            )
            .order_by(TrainedModel.version.desc())
            .first()
        )
        if pe_model:
            meta = pe_model.model_metadata or {}
            return {
                "id":            pe_model.id,
                "artifact_type": "pe",
                "display_name":  f"PE v{pe_model.version}",
                "architecture":  meta.get("architecture", "dense"),
                "version":       pe_model.version,
                "best_accuracy": job.best_accuracy,
            }

        tflite_candidates = (
            db.query(TrainedModel)
            .filter(
                TrainedModel.training_job_id == job.id,
                TrainedModel.format == "tflite",
            )
            .all()
        )
        if tflite_candidates:
            def _tflite_preference(tm: TrainedModel) -> tuple[int, int]:
                meta = tm.model_metadata or {}
                variant = str(meta.get("variant", ""))
                architecture = str(meta.get("architecture", "")).lower()
                is_yolo_pro = (
                    meta.get("model_type") == "yolo_pro_detection"
                    or meta.get("output_type") == "yolo_pro_detection"
                    or "yolo_pro" in architecture
                )
                if is_yolo_pro:
                    if variant == "decoded_float32":
                        return (0, -int(tm.version or 0))
                    if variant == "float32":
                        return (1, -int(tm.version or 0))
                    if variant == "int8":
                        return (2, -int(tm.version or 0))
                    return (3, -int(tm.version or 0))
                if variant == "float32":
                    return (0, -int(tm.version or 0))
                if variant == "decoded_float32":
                    return (1, -int(tm.version or 0))
                if variant == "int8":
                    return (2, -int(tm.version or 0))
                return (3, -int(tm.version or 0))

            tm = sorted(tflite_candidates, key=_tflite_preference)[0]
            meta = tm.model_metadata or {}
            variant = meta.get("variant")
            variant_suffix = f" ({variant})" if variant else ""
            return {
                "id":            tm.id,
                "artifact_type": "tflite",
                "display_name":  f"TFLITE v{tm.version}{variant_suffix}",
                "architecture":  meta.get("architecture", "dense"),
                "version":       tm.version,
                "best_accuracy": job.best_accuracy,
            }

    raise HTTPException(404, "No trained model artifact found for this impulse")


@router.post("/classify-sample")
async def classify_sample(
    req: ClassifySampleRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Run inference on an existing test sample using the real runtime artifact.

    Supports three artifact types — resolution priority is:
      1. TrainedModel (tflite / pe)  — identified by TrainedModel.id in req.model_id
      2. Deployment (.pxe)           — identified by Deployment.id in req.model_id
      3. Auto-resolve (no model_id)  — latest tflite/pe TrainedModel, then latest
                                       completed Deployment (covers .pxe builds)

    DSP preprocessing is applied to the raw sample bytes before inference —
    cached evaluation rows are never used here (rule: always infer, never recall).
    """
    from app.models.user import Impulse, Sample, Deployment

    # ── 1. Validate impulse (ownership-checked) ───────────────────────────────
    impulse = assert_impulse_owner(db, req.impulse_id, current_user)

    # ── 2. Validate sample (must belong to the impulse's project) ─────────────
    sample = db.query(Sample).filter(
        Sample.id == req.sample_id,
        Sample.project_id == impulse.project_id,
    ).first()
    if not sample:
        raise HTTPException(404, "Sample not found or does not belong to this project")

    # ── 2b. Verify sample is in the active impulse's DSP scope ────────────────
    # Mirrors get_sample_options — only samples with FeatureSet entries for this
    # impulse are accepted, keeping the dropdown and classify path consistent.
    if not _sample_in_impulse_scope(req.sample_id, req.impulse_id, db):
        raise HTTPException(
            422,
            "Sample is not in scope for this impulse. "
            "Run 'Generate Features' to include it, then retry.",
        )

    # ── 2c. SSD object-detection short-circuit ────────────────────────────────
    # MobileNetV2 SSD FPN-Lite must be decoded + scored exactly like the Model
    # Testing / evaluation path: the decoded (1, N, 6) variant, conf=0.01 (never
    # the 0.25 classification default that filters out every SSD detection), per-
    # class NMS, and the adaptive per-image gate. The generic _run_tflite path
    # has no SSD branch and would fall through to the softmax classification
    # branch → is_detection=False, zero boxes (the bug). Likewise the PXE artifact
    # (preferred below) does not score SSD with training-parity confidence.
    # Routing SSD here keeps Live Classification aligned with evaluation scoring.
    # Returns None for every non-SSD model (after only a cheap metadata lookup,
    # before any DSP preprocessing), so all other models use the path below.
    from app.services.model_testing_service import detect_ssd_for_live

    _loop_ssd = asyncio.get_event_loop()
    _ssd_result = await _loop_ssd.run_in_executor(
        None, detect_ssd_for_live, sample, impulse, db
    )
    if _ssd_result is not None:
        return {
            **_ssd_result,
            "sample_id": sample.id,
            "sample_name": sample.filename,
            "ground_truth_label": _ground_truth_label(sample),
            "ground_truth_boxes": _ground_truth_boxes(sample),
        }

    # ── 3. Resolve runtime artifact ───────────────────────────────────────────
    # model is only set when the artifact is a TrainedModel (required for _resolve_model).
    # storage_key is the canonical pointer regardless of artifact type.
    model = None
    storage_key = None

    if req.model_id:
        # Try TrainedModel first (tflite / pe formats)
        model = db.query(TrainedModel).filter(
            TrainedModel.id == req.model_id,
            TrainedModel.format.in_(["tflite", "pe"]),
        ).first()
        if model:
            assert_trained_model_owner(db, model.id, current_user)
            storage_key = model.storage_key
        else:
            # model_id may refer to a Deployment (e.g. a .pxe build)
            dep = db.query(Deployment).filter(Deployment.id == req.model_id).first()
            if dep:
                assert_deployment_owner(db, dep.id, current_user)
                storage_key = dep.storage_key
        if not storage_key:
            raise HTTPException(404, "Model not found (tflite, pe, or pxe format required)")
    else:
        # Auto-resolve via the shared preferred-runtime policy so page bootstrap
        # and classify-sample use the same artifact priority: PXE > PE > TFLite.
        try:
            preferred = get_preferred_runtime(
                impulse_id=req.impulse_id,
                db=db,
                current_user=current_user,
            )
        except HTTPException as exc:
            if exc.status_code == 404:
                raise HTTPException(
                    422,
                    "No trained model artifact found. Train a model first.",
                )
            raise

        artifact_id = preferred["id"]
        artifact_type = preferred["artifact_type"]

        if artifact_type == "pxe":
            dep = db.query(Deployment).filter(Deployment.id == artifact_id).first()
            if dep:
                storage_key = dep.storage_key
        else:
            model = db.query(TrainedModel).filter(TrainedModel.id == artifact_id).first()
            if model:
                storage_key = model.storage_key

        if not storage_key:
            raise HTTPException(
                404,
                "Preferred runtime artifact could not be resolved",
            )

    # ── 4. DSP preprocessing (run in thread pool — downloads S3 + DSP) ───────
    from app.services.model_testing_service import _preprocess_sample

    loop = asyncio.get_event_loop()
    features_arr = await loop.run_in_executor(None, _preprocess_sample, sample, impulse)
    if features_arr is None:
        raise HTTPException(
            422,
            "Failed to preprocess sample — check sample storage and DSP configuration",
        )
    features_flat = features_arr.flatten().tolist()

    # ── 5. Download artifact bytes and route by format ────────────────────────
    # Blocking boto3 download — offload so the event loop is not stalled.
    raw_bytes = await loop.run_in_executor(None, storage.download_bytes, storage_key)

    if _is_pxe(raw_bytes):
        # .pxe path — out-of-process runner.
        # Use PxeProcess directly (instead of run_pxe_inference) so we can
        # inspect the manifest before choosing which features to send.
        # For image models the embedded runner's _dsp_image applies its own
        # resize/normalization; sending already-DSP'd features would double-apply
        # those transforms and break FOMO spatial semantics.
        from app.ml.pxe_runner import PxeProcess

        _sample_storage_key = sample.storage_key  # capture for closure

        def _pxe_classify_sample():
            with PxeProcess(raw_bytes) as proc:
                pxe_input_shape = proc.manifest.get("input_shape", [])
                if len(pxe_input_shape) >= 3:
                    # Image model: download raw bytes and pass pixel values.
                    raw_sample_bytes = storage.download_bytes(_sample_storage_key)
                    pxe_features = _raw_image_tensor(
                        raw_sample_bytes,
                        proc.manifest.get("channel_order", "rgb"),
                    )
                else:
                    # Sensor model: DSP-processed features are correct.
                    pxe_features = features_flat
                result = proc.classify(pxe_features)
                result["_pxe_manifest"] = proc.manifest
                return result

        try:
            raw_result = await loop.run_in_executor(None, _pxe_classify_sample)
        except Exception as exc:
            logger.error("PXE inference failed for sample %s: %s", req.sample_id, exc)
            raise HTTPException(422, f"PXE inference failed: {exc}")
        # Normalize EI stdio-JSONL output → frontend LiveClassificationRunResult shape.
        # _pxe_manifest is consumed (popped) inside the normalizer.
        result = _normalize_pxe_classify_result(raw_result)
        return {
            **result,
            "sample_id": sample.id,
            "sample_name": sample.filename,
            "ground_truth_label": _ground_truth_label(sample),
            "ground_truth_boxes": _ground_truth_boxes(sample),
            "model_id": req.model_id or storage_key,
            "model_version": None,
            "model_architecture": result.get("model_type", "pxe"),
        }

    # .tflite / .pe path — existing in-process interpreter
    if not model:
        raise HTTPException(
            404,
            "Resolved artifact is not a valid tflite/pe model and is not a .pxe binary",
        )
    # Reuse the bytes already fetched in step 5 when they are the model's own
    # bytes, avoiding _resolve_model's second download.
    _preloaded = raw_bytes if storage_key == model.storage_key else None
    try:
        tflite_bytes, meta = _resolve_model(model, raw_bytes=_preloaded)
    except ValueError as exc:
        raise HTTPException(422, f"Model artifact error: {exc}")

    # Match Model Testing for FOMO/detection_heatmap when the UI does not
    # explicitly override the threshold: prefer the training-time threshold
    # from classification_report over package/default metadata.
    effective_fomo_threshold = req.fomo_threshold
    if effective_fomo_threshold is None and _is_fomo_model(meta) and model.training_job_id:
        try:
            from app.models.user import TrainingJob
            from app.ml.fomo_evaluator import resolve_fomo_threshold, ThresholdSource

            tj = db.query(TrainingJob).filter(TrainingJob.id == model.training_job_id).first()
            if tj and isinstance(tj.classification_report, dict):
                effective_fomo_threshold, _threshold_source = resolve_fomo_threshold(
                    mode=ThresholdSource.TRAINING_PARITY,
                    training_classification_report=tj.classification_report,
                )
                logger.info(
                    "[inference] classify_sample using FOMO threshold %.4f from %s for model %s",
                    effective_fomo_threshold,
                    _threshold_source,
                    model.id,
                )
        except Exception as exc:
            logger.warning(
                "[inference] Failed to resolve training-parity FOMO threshold for model %s; "
                "falling back to request/meta threshold: %s",
                model.id,
                exc,
            )

    try:
        result = await loop.run_in_executor(
            None, _run_tflite, tflite_bytes, features_flat, meta, effective_fomo_threshold
        )
    except Exception as exc:
        logger.error("Inference failed for sample %s: %s", req.sample_id, exc)
        raise HTTPException(422, f"Inference failed: {exc}")

    meta_info = model.model_metadata or {}
    return {
        **result,
        "sample_id": sample.id,
        "sample_name": sample.filename,
        "ground_truth_label": _ground_truth_label(sample),
        "ground_truth_boxes": _ground_truth_boxes(sample),
        "model_id": model.id,
        "model_version": model.version,
        "model_architecture": meta_info.get("architecture", "dense"),
    }


# ─── TFLite inference core ────────────────────────────────────────────────────

def _run_tflite(tflite_bytes: bytes, features: list, meta: dict, fomo_threshold: Optional[float] = None) -> dict:
    """Load TFLite flatbuffer into interpreter and run one inference.

    All runtime decisions (normalization, channel order, label mapping) are
    driven by *meta*, which must have been validated by _require_meta_fields
    before this function is called.  No defaults are applied here.

    Two post-processing branches:
      - FOMO / detection_heatmap: reshape output to (H, W, C+1) grid, decode
        per-cell detections above threshold.  No softmax over the flat vector.
      - Classification: flatten output, apply softmax, return top label +
        per-class score dict.
    """
    interp = _make_tflite_interpreter(model_content=tflite_bytes)
    interp.allocate_tensors()

    inp_details = interp.get_input_details()[0]
    out_details = interp.get_output_details()[0]

    # label_names, normalize_input, channel_order are guaranteed present by
    # _require_meta_fields â€" access directly so a missing key is a loud crash,
    # not a silent wrong-answer default.
    label_names = meta["label_names"]

    x = np.array(features, dtype=np.float32).reshape(inp_details["shape"])
    x = _apply_manifest_input_contract(x, meta, skip_if_pre_normalized=True)

    # Quantize if int8 model
    quant_params = inp_details.get("quantization")
    if not quant_params or len(quant_params) != 2:
        raise ValueError(f"Unexpected quantization params: {quant_params}")
    scale, zero_point = quant_params
    if inp_details["dtype"] == np.int8:
        if scale == 0:
            raise ValueError("int8 input tensor has scale=0; model metadata is corrupt")
        x = np.clip(np.round(x / scale + zero_point), -128, 127).astype(np.int8)

    interp.set_tensor(inp_details["index"], x)
    interp.invoke()

    # Keep the raw tensor (preserves shape for FOMO 4-D outputs).
    raw_output = interp.get_tensor(out_details["index"])

    # Dequantize output if int8.
    out_scale, out_zp = out_details.get("quantization", (0, 0))
    if out_scale != 0:
        raw_output = (raw_output.astype(np.float32) - out_zp) * out_scale
    raw_output = raw_output.astype(np.float32)

    flat = raw_output.flatten()
    debug = {
        "raw_min":  float(flat.min()),
        "raw_max":  float(flat.max()),
        "raw_mean": float(flat.mean()),
    }

    return _postprocess_output(raw_output, meta, debug, threshold_override=fomo_threshold)
