"""
Model Testing Service
=====================
All business logic for the /api/v1/model-testing endpoints lives here.

Key design rules
----------------
- classify_all loads the real trained model from storage and runs actual
  inference on every test sample.  No simulation is used.

- Architecture is detected from TrainedModel.model_metadata["output_type"]
  (not from training_history JSON, which is unreliable).

- classify_all is synchronous for now (Celery async is Fix #2).
  It processes samples in batches of INFERENCE_BATCH_SIZE so the DB
  session is not held open across the entire run.

- Scoring is real:
    Classification → argmax vs expected_label, score = confidence [0-1]
    Detection      → IoU matching at 0.50, score = max IoU matched box
  (Full mAP accumulation is Fix #3.)

- Never raise 404/422 for "no test runs yet" — return an empty-but-valid
  envelope so the frontend always has something to render.

- All list fields are always arrays (never null).
"""
from __future__ import annotations

import io
import os
import logging
import math
import tempfile
from datetime import datetime
from typing import Optional, List, Tuple, Dict

import numpy as np
from fastapi import HTTPException
from sqlalchemy.orm import Session

from app.models.user import (
    Impulse, Project, Sample, Label, TrainedModel,
    SampleType, TrainingJob, JobStatus,
)
from app.models.model_testing import (
    ModelVersion, ModelTestRun, MetricResult, ModelTestSample,
    TestResultStatus, QuantizationType, UnsupportedModelTestingError,
)
from app.core.storage import storage
from app.ml.dsp.processor import DSPProcessor, merge_image_params

logger = logging.getLogger(__name__)

# Sample types treated as the "test split".
# "automatic" uploads are resolved to training/testing at ingest time, so model
# testing should only read rows whose final stored split is "testing".
_TEST_SPLIT = (SampleType.testing,)

# Number of samples processed per inference call
INFERENCE_BATCH_SIZE = 16

# Confidence threshold: predicted class score must exceed this to count as a
# detection (classification).  Kept low so precision/recall curves are
# computed over the full score range; callers may tighten for display.
CONF_THRESHOLD = 0.0

# IoU threshold for a detection box to count as a true positive
IOU_THRESHOLD_MATCH = 0.50

# Uncertain threshold: correct prediction below this confidence is marked
# "uncertain" instead of "pass" (Fix: uncertain bucket support)
UNCERTAIN_THRESHOLD = 0.6

# Output types with no detection-scoring path in Model Testing yet.
# (Empty: SSD detection scoring is now wired in — see is_detection routing
# and _scale_to_uint8_pixel_range. The gate mechanism is retained so a future
# unsupported architecture can fail fast with a clear message instead of
# crashing inside a mismatched scorer.)
MODEL_TESTING_UNSUPPORTED_OUTPUT_TYPES: tuple = ()

_UNSUPPORTED_TESTING_MSG = (
    "Model Testing does not yet support '{ot}' models. SSD detection scoring "
    "is not wired into the Model Testing pipeline; use the training evaluation "
    "report for detection metrics instead."
)


# ─── Serialisers ──────────────────────────────────────────────────────────────

def _version_out(v: ModelVersion) -> dict:
    return {
        "id":                v.id,
        "version_number":    v.version_number,
        "name":              v.name,
        "quantization_type": v.quantization_type,
        "is_active":         v.is_active,
        "created_at":        v.created_at.isoformat() if v.created_at else None,
    }


def _sample_out(s: ModelTestSample, gt_boxes: Optional[list] = None) -> dict:
    raw_pb = getattr(s, "predicted_boxes", None)
    if isinstance(raw_pb, dict) and "boxes" in raw_pb:
        # v2 format: {"v": 2, "boxes": [...], "tp": int, "fp": int, "fn": int}
        boxes     = raw_pb.get("boxes") or []
        det_stats: Optional[dict] = {
            "tp": raw_pb.get("tp"),
            "fp": raw_pb.get("fp"),
            "fn": raw_pb.get("fn"),
        }
    else:
        # Legacy format: list or None (data classified before v2)
        boxes     = raw_pb
        det_stats = None

    return {
        "id":               s.id,
        "sample_id":        getattr(s, "sample_id", None),
        "sample_name":      s.sample_name,
        "expected_outcome": s.expected_outcome or "-",
        "precision_score":  s.f1_score,
        "predicted_class":  getattr(s, "predicted_class", None),
        "iou_score":        getattr(s, "iou_score", None),
        "predicted_boxes":  boxes,
        "det_stats":        det_stats,
        "gt_boxes":         gt_boxes or [],
        "result_status":    s.result_status,
        "updated_at":       s.updated_at.isoformat() if s.updated_at else None,
    }


def _metric_out(m: MetricResult) -> dict:
    return {
        "metric_name":         m.metric_name,
        "metric_display_name": m.metric_display_name,
        "metric_value":        m.metric_value,
    }


def _run_out(r: ModelTestRun) -> dict:
    return {
        "id":               r.id,
        "status":           r.status,
        "accuracy":         r.accuracy,
        "total_samples":    r.total_samples,
        "passed_samples":   r.passed_samples,
        "failed_samples":   r.failed_samples,
        "model_version_id": r.model_version_id,
        "error":            r.error,
        "created_at":       r.created_at.isoformat()   if r.created_at   else None,
        "started_at":       r.started_at.isoformat()   if r.started_at   else None,
        "completed_at":     r.completed_at.isoformat() if r.completed_at else None,
    }


def _impulse_out(i: Impulse) -> dict:
    return {
        "id":         i.id,
        "name":       i.name,
        "project_id": i.project_id,
    }


# ─── Auto-provisioning ────────────────────────────────────────────────────────

def _derive_expected_outcome(sample, label_map: dict) -> str:
    """Return expected label for a sample, preferring the first bbox label for detection samples."""
    bboxes = (sample.extra_metadata or {}).get("boundingBoxes") or []
    if bboxes:
        first = bboxes[0]
        bid   = first.get("label_id")
        bname = first.get("label")
        if bid and bid in label_map:
            return label_map[bid]
        if bname:
            return bname
    return label_map.get(sample.label_id, "-") if sample.label_id else "-"


def _ensure_test_samples_provisioned(impulse: Impulse, db: Session) -> List[ModelTestSample]:
    """
    Idempotent bridge between the real `samples` table and `model_test_samples`.

    Uses per-row SAVEPOINT rollback to be safe against concurrent duplicate
    inserts.  Flushes only when new rows were inserted.  Never commits —
    the caller decides transaction boundaries.

    Returns the full list of ModelTestSample rows for the impulse (after any
    new insertions).
    """
    now = datetime.utcnow()

    real_samples: List[Sample] = (
        db.query(Sample)
        .filter(
            Sample.project_id == impulse.project_id,
            Sample.sample_type.in_(_TEST_SPLIT),
        )
        .order_by(Sample.created_at.asc())
        .all()
    )

    desired_sample_ids = {s.id for s in real_samples}

    # Remove stale bridge rows that point at samples no longer in the testing
    # split. This keeps old "automatic"-backed rows from being counted forever.
    stale_q = db.query(ModelTestSample).filter(
        ModelTestSample.impulse_id == impulse.id,
        ModelTestSample.sample_id.isnot(None),
    )
    if desired_sample_ids:
        stale_q = stale_q.filter(~ModelTestSample.sample_id.in_(desired_sample_ids))
    stale_removed = stale_q.delete(synchronize_session=False)
    if stale_removed:
        logger.debug(
            "Provisioning: removed %s stale test sample row(s) for impulse %s",
            stale_removed, impulse.id,
        )

    if not real_samples:
        return (
            db.query(ModelTestSample)
            .filter(ModelTestSample.impulse_id == impulse.id)
            .order_by(ModelTestSample.created_at.asc())
            .all()
        )

    # Build label map for expected_outcome (top-level + bbox label_ids)
    label_ids = {s.label_id for s in real_samples if s.label_id}
    for s in real_samples:
        for box in (s.extra_metadata or {}).get("boundingBoxes") or []:
            bid = box.get("label_id")
            if bid:
                label_ids.add(bid)
    label_map: dict = {}
    if label_ids:
        for lbl in db.query(Label).filter(Label.id.in_(label_ids)).all():
            label_map[lbl.id] = lbl.name

    existing_ids: set = {
        row.sample_id
        for row in db.query(ModelTestSample.sample_id)
        .filter(
            ModelTestSample.impulse_id == impulse.id,
            ModelTestSample.sample_id.isnot(None),
        )
        .all()
    }

    # Remove legacy duplicate bridge rows for the same real sample_id.
    # Keep the most recently updated row so user edits survive when possible.
    existing_rows = (
        db.query(ModelTestSample)
        .filter(
            ModelTestSample.impulse_id == impulse.id,
            ModelTestSample.sample_id.isnot(None),
        )
        .order_by(ModelTestSample.updated_at.desc(), ModelTestSample.created_at.desc())
        .all()
    )
    seen_sample_ids: set = set()
    duplicate_ids: List[str] = []
    for row in existing_rows:
        if row.sample_id in seen_sample_ids:
            duplicate_ids.append(row.id)
        else:
            seen_sample_ids.add(row.sample_id)
    if duplicate_ids:
        db.query(ModelTestSample).filter(ModelTestSample.id.in_(duplicate_ids)).delete(
            synchronize_session=False
        )
        logger.debug(
            "Provisioning: removed %s duplicate test sample row(s) for impulse %s",
            len(duplicate_ids), impulse.id,
        )
        existing_ids = seen_sample_ids

    # Map kept row per sample_id for stale expected_outcome refresh
    existing_row_by_sample_id: dict = {}
    for row in existing_rows:
        if row.sample_id not in existing_row_by_sample_id:
            existing_row_by_sample_id[row.sample_id] = row

    inserted = 0
    refreshed = 0
    for sample in real_samples:
        expected = _derive_expected_outcome(sample, label_map)

        if sample.id in existing_ids:
            row = existing_row_by_sample_id.get(sample.id)
            if row and (row.expected_outcome or "-") != expected:
                row.expected_outcome = expected
                row.updated_at = now
                refreshed += 1
            continue

        display_name = os.path.basename(sample.filename)
        new_row = ModelTestSample(
            project_id       = impulse.project_id,
            impulse_id       = impulse.id,
            sample_id        = sample.id,
            sample_name      = display_name,
            expected_outcome = expected,
            f1_score         = None,
            result_status    = TestResultStatus.pending,
            created_at       = now,
            updated_at       = now,
        )

        try:
            db.begin_nested()   # SAVEPOINT
            db.add(new_row)
            db.flush()
            inserted += 1
        except Exception:
            db.rollback()
            logger.debug(
                f"Provisioning: sample {sample.id} already exists "
                f"(concurrent insert) — skipping."
            )

    if inserted > 0:
        logger.debug(f"Provisioned {inserted} new test sample row(s) for impulse {impulse.id}")
    if refreshed > 0:
        logger.debug(f"Refreshed expected_outcome for {refreshed} stale row(s) for impulse {impulse.id}")

    return (
        db.query(ModelTestSample)
        .filter(ModelTestSample.impulse_id == impulse.id)
        .order_by(ModelTestSample.created_at.asc())
        .all()
    )


# ─── Reference resolution helpers ─────────────────────────────────────────────

def _resolve_impulse(
    impulse_id: Optional[str],
    project_id: Optional[str],
    db: Session,
) -> Optional[Impulse]:
    if impulse_id:
        obj = db.query(Impulse).filter(Impulse.id == impulse_id).first()
        if not obj:
            raise HTTPException(404, f"Impulse '{impulse_id}' not found")
        return obj
    if project_id:
        return (
            db.query(Impulse)
            .filter(Impulse.project_id == project_id)
            .order_by(Impulse.created_at.asc())
            .first()
        )
    return None


def _get_active_version(
    impulse_id: str,
    model_version_id: Optional[str],
    db: Session,
) -> Optional[ModelVersion]:
    if model_version_id:
        v = db.query(ModelVersion).filter(
            ModelVersion.id == model_version_id,
            ModelVersion.impulse_id == impulse_id,
        ).first()
        if not v:
            raise HTTPException(404, "Model version not found")
        return v
    return (
        db.query(ModelVersion)
        .filter(ModelVersion.impulse_id == impulse_id, ModelVersion.is_active == True)
        .order_by(ModelVersion.created_at.desc())
        .first()
    )


def _latest_test_run(impulse_id: str, db: Session) -> Optional[ModelTestRun]:
    return (
        db.query(ModelTestRun)
        .filter(ModelTestRun.impulse_id == impulse_id)
        .order_by(ModelTestRun.created_at.desc())
        .first()
    )


# ─── Model loading ────────────────────────────────────────────────────────────

def _load_trained_model(impulse_id: str, model_version_id: Optional[str], db: Session):
    """
    Load the best available TrainedModel for this impulse.

    Priority:
      1. TrainedModel linked from the active ModelVersion (model_version_id)
      2. Most-recent completed TrainingJob → prefer float32 TFLite, else keras

    Returns (trained_model_row, output_type_str) or raises HTTP 422.
    """
    # Path 1: via ModelVersion.trained_model_id
    if model_version_id:
        mv = db.query(ModelVersion).filter(ModelVersion.id == model_version_id).first()
        if mv and mv.trained_model_id:
            tm = db.query(TrainedModel).filter(TrainedModel.id == mv.trained_model_id).first()
            if tm:
                out_type = (tm.model_metadata or {}).get("output_type", "classification")
                return tm, out_type

    # Path 2: latest completed training job for this impulse
    latest_job = (
        db.query(TrainingJob)
        .filter(
            TrainingJob.impulse_id == impulse_id,
            TrainingJob.status == JobStatus.completed,
        )
        .order_by(TrainingJob.created_at.desc())
        .first()
    )
    if not latest_job:
        raise HTTPException(
            422,
            "No completed training job found for this impulse. "
            "Train the model first before running model testing.",
        )

    candidates = (
        db.query(TrainedModel)
        .filter(TrainedModel.training_job_id == latest_job.id)
        .all()
    )
    if not candidates:
        raise HTTPException(
            422,
            "Training completed but no model artifacts were saved. "
            "Re-run training to generate model files.",
        )

    def _preference(tm: TrainedModel) -> int:
        meta = tm.model_metadata or {}
        variant = meta.get("variant", "")
        fmt = tm.format or ""
        if fmt == "tflite" and variant == "decoded_float32":
            return 0
        if fmt == "tflite" and variant == "float32":
            return 1
        if fmt == "keras":
            return 2
        return 3

    best = sorted(candidates, key=_preference)[0]
    out_type = (best.model_metadata or {}).get("output_type", "classification")
    return best, out_type


def _load_model_into_memory(trained_model: TrainedModel):
    """
    Download the model artifact from S3 and return a callable inference object.

    Returns either a TFLite interpreter (for .tflite) or a Keras model (for .keras).
    """
    import tensorflow as tf

    model_bytes = storage.download_bytes(trained_model.storage_key)

    if trained_model.format == "tflite":
        # On Windows, the default XNNPACK delegate has been observed to
        # terminate the model-testing worker after successful inference.
        # Use the built-in resolver first for a more stable classify-all path.
        try:
            interpreter = tf.lite.Interpreter(
                model_content=model_bytes,
                experimental_op_resolver_type=(
                    tf.lite.experimental.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES
                ),
            )
        except Exception:
            interpreter = tf.lite.Interpreter(model_content=model_bytes)
        interpreter.allocate_tensors()
        return ("tflite", interpreter)

    if trained_model.format == "keras":
        with tempfile.NamedTemporaryFile(suffix=".keras", delete=False) as f:
            f.write(model_bytes)
            tmp_path = f.name
        try:
            model = tf.keras.models.load_model(tmp_path)
        finally:
            os.unlink(tmp_path)
        return ("keras", model)

    raise HTTPException(422, f"Unsupported model format: {trained_model.format}")


def _run_inference(model_handle, image_batch: np.ndarray) -> np.ndarray:
    """
    Run inference on a (B, H, W, C) float32 image batch.
    """
    import tensorflow as tf

    kind, model = model_handle
    if kind == "tflite":
        interp = model
        input_details  = interp.get_input_details()
        output_details = interp.get_output_details()

        input_shape  = input_details[0]["shape"]
        tflite_batch = int(input_shape[0])

        # Dynamic-batch models export batch dim as -1; treat ≤1 as single-sample.
        # Always iterate one sample at a time to avoid native TFLite shape aborts.
        in_dtype  = input_details[0].get("dtype", np.float32)
        in_quant  = input_details[0].get("quantization", (0.0, 0))
        in_scale, in_zp = (float(in_quant[0]), int(in_quant[1])) if len(in_quant) == 2 else (0.0, 0)

        def _quantize_input(x: np.ndarray) -> np.ndarray:
            """Quantize float32 → int8/uint8 if the model requires it."""
            if in_dtype == np.int8 and in_scale != 0.0:
                return np.clip(np.round(x / in_scale + in_zp), -128, 127).astype(np.int8)
            if in_dtype == np.uint8 and in_scale != 0.0:
                return np.clip(np.round(x / in_scale + in_zp), 0, 255).astype(np.uint8)
            return x.astype(in_dtype)

        def _dequantize_output(raw: np.ndarray, od: dict) -> np.ndarray:
            """Dequantize int8/uint8 output → float32 if needed."""
            q = od.get("quantization", (0.0, 0))
            scale, zp = (float(q[0]), int(q[1])) if len(q) == 2 else (0.0, 0)
            if scale != 0.0:
                return (raw.astype(np.float32) - zp) * scale
            return raw.astype(np.float32)

        if tflite_batch != len(image_batch) or tflite_batch <= 0:
            outputs = []
            for i in range(len(image_batch)):
                interp.set_tensor(input_details[0]["index"], _quantize_input(image_batch[i:i+1]))
                interp.invoke()
                outs = [_dequantize_output(interp.get_tensor(od["index"]), od) for od in output_details]
                outputs.append(outs)
            if len(output_details) == 1:
                return np.concatenate([o[0] for o in outputs], axis=0)
            return [np.concatenate([o[i] for o in outputs], axis=0)
                    for i in range(len(output_details))]
        else:
            interp.set_tensor(input_details[0]["index"], _quantize_input(image_batch))
            interp.invoke()
            if len(output_details) == 1:
                return _dequantize_output(interp.get_tensor(output_details[0]["index"]), output_details[0])
            return [_dequantize_output(interp.get_tensor(od["index"]), od) for od in output_details]

    result = model(image_batch, training=False)
    if isinstance(result, dict):
        return {k: v.numpy() if hasattr(v, "numpy") else np.array(v)
                for k, v in result.items()}
    arr = result.numpy() if hasattr(result, "numpy") else np.array(result)
    return arr


def _scale_to_uint8_pixel_range(model_handle, image_batch: np.ndarray) -> np.ndarray:
    """Rescale a normalized [0,1] float batch to raw [0,255] for uint8-input models.

    The SSD decoded TFLite keeps a uint8 input + an in-graph NormalizationLayer
    (x/127.5 - 1), so it expects raw [0,255] pixels. The DSP pipeline delivers
    normalized [0,1] floats (processor.py: ``data = data / 255.0``); without
    this rescale, ``_run_inference._quantize_input`` would cast [0,1] straight
    to uint8 → all-zero pixels → garbage detections.

    No-op for float-input models (e.g. YOLO-Pro's decoded export) and for data
    that is already in [0,255], so it is safe to call generically.
    """
    kind, model = model_handle
    if kind != "tflite":
        return image_batch
    try:
        in_dtype = model.get_input_details()[0].get("dtype", np.float32)
    except Exception:
        return image_batch
    if in_dtype == np.uint8 and image_batch.size and float(np.nanmax(image_batch)) <= 1.5:
        return np.clip(image_batch * 255.0, 0.0, 255.0).astype(np.float32)
    return image_batch


def _resolve_model_input_shape(
    model_handle,
    trained_model: TrainedModel,
    fallback_shape: Tuple[int, int, int],
) -> Tuple[int, int, int]:
    """
    Resolve the model's effective single-sample input shape as (H, W, C).
    Prefer stored metadata, then inspect the loaded model/interpreter.
    """
    meta = trained_model.model_metadata or {}
    raw_shape = meta.get("input_shape")

    def _normalize_shape(shape) -> Optional[Tuple[int, int, int]]:
        if not isinstance(shape, (list, tuple)):
            return None
        dims = [int(d) for d in shape if isinstance(d, (int, float)) and int(d) > 0]
        if len(shape) >= 4 and len(dims) >= 3:
            return tuple(dims[-3:])
        if len(shape) == 3 and len(dims) == 3:
            return tuple(dims)
        return None

    normalized = _normalize_shape(raw_shape)
    if normalized:
        return normalized

    kind, model = model_handle
    try:
        if kind == "tflite":
            input_details = model.get_input_details()
            details = input_details[0] if input_details else {}
            normalized = _normalize_shape(details.get("shape_signature"))
            if normalized:
                return normalized
            normalized = _normalize_shape(details.get("shape"))
            if normalized:
                return normalized
        else:
            input_shape = getattr(model, "input_shape", None)
            if isinstance(input_shape, list) and input_shape:
                input_shape = input_shape[0]
            normalized = _normalize_shape(input_shape)
            if normalized:
                return normalized
    except Exception:
        logger.debug("[model_testing] Could not inspect model input shape", exc_info=True)

    return fallback_shape


def _coerce_image_to_shape(
    image: np.ndarray,
    source_shape: Tuple[int, int, int],
    target_shape: Tuple[int, int, int],
) -> np.ndarray:
    """
    Coerce an image tensor or flat feature vector to the model's input shape.
    """
    import tensorflow as tf

    arr = np.asarray(image, dtype=np.float32)
    source_elems = int(np.prod(source_shape))
    target_elems = int(np.prod(target_shape))

    if arr.ndim == 1:
        if arr.size == source_elems:
            arr = arr.reshape(source_shape)
        elif arr.size == target_elems:
            arr = arr.reshape(target_shape)
        else:
            raise ValueError(
                f"Flat feature size {arr.size} does not match source {source_shape} "
                f"or target {target_shape}"
            )
    elif arr.ndim == 2:
        arr = arr[..., np.newaxis]
    elif arr.ndim != 3:
        raise ValueError(f"Unsupported sample tensor rank {arr.ndim} for shape {arr.shape}")

    if arr.ndim == 3 and arr.shape != target_shape and arr.size == target_elems:
        arr = arr.reshape(target_shape)

    if arr.ndim != 3:
        raise ValueError(f"Could not coerce sample into rank-3 tensor: {arr.shape}")

    src_h, src_w, _src_c = arr.shape
    tgt_h, tgt_w, tgt_c = target_shape

    if arr.shape[2] != tgt_c:
        if arr.shape[2] == 1 and tgt_c == 3:
            arr = np.repeat(arr, 3, axis=2)
        elif arr.shape[2] == 3 and tgt_c == 1:
            arr = arr.mean(axis=2, keepdims=True)
        else:
            raise ValueError(
                f"Channel mismatch for sample tensor: got {arr.shape[2]}, expected {tgt_c}"
            )

    if (src_h, src_w) != (tgt_h, tgt_w):
        arr = tf.image.resize(arr, [tgt_h, tgt_w]).numpy()

    return np.asarray(arr, dtype=np.float32)


def _iter_scalar_metric_dicts(metric_dicts: List[dict]) -> List[dict]:
    """
    Persist only scalar metric values. Structured values belong in JSON fields,
    not MetricResult.metric_value.
    """
    scalar_metrics: List[dict] = []
    skipped_metrics: List[str] = []

    for md in metric_dicts:
        value = md.get("metric_value")
        if value is None or np.isscalar(value):
            scalar_metrics.append(md)
        else:
            skipped_metrics.append(str(md.get("metric_name") or "unknown"))

    if skipped_metrics:
        logger.info(
            "[model_testing] Skipping non-scalar metric_result rows: %s",
            ", ".join(skipped_metrics),
        )

    return scalar_metrics


# ─── DSP preprocessing ────────────────────────────────────────────────────────

def _preprocess_sample(sample: Sample, impulse: Impulse) -> Optional[np.ndarray]:
    """
    Download a sample from S3 and run the impulse DSP pipeline on it.
    Returns a float32 numpy array ready for model input, or None on error.
    """
    try:
        raw_bytes = storage.download_bytes(sample.storage_key)
    except Exception as e:
        logger.warning(f"Could not download sample {sample.id}: {e}")
        return None

    try:
        dsp_blocks = impulse.dsp_blocks or [{"type": "image", "params": {}}]
        features: List[np.ndarray] = []
        for block_cfg in dsp_blocks:
            params = merge_image_params(impulse, dict(block_cfg.get("params", {})))
            processor = DSPProcessor(block_cfg.get("type", "image"), params)
            feat = processor.extract(raw_bytes)
            features.append(np.asarray(feat, dtype=np.float32))
        if len(features) == 1:
            return features[0]
        return np.concatenate([f.flatten() for f in features])
    except Exception as e:
        logger.warning(f"DSP processing failed for sample {sample.id}: {e}")
        return None


# ─── IoU helper ───────────────────────────────────────────────────────────────

def _iou(box_a: List[float], box_b: List[float]) -> float:
    """Compute IoU between two [x1, y1, x2, y2] boxes (normalised)."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1 = max(ax1, bx1); iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2); iy2 = min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


# ─── Per-sample scoring ───────────────────────────────────────────────────────

def _score_classification(
    raw_output: np.ndarray,
    expected_label: str,
    label_names: List[str],
    uncertain_threshold: float = UNCERTAIN_THRESHOLD,
) -> Tuple[float, TestResultStatus, str, None]:
    """
    Score one classification output.

    raw_output          : (num_classes,) float32 — softmax probabilities
    uncertain_threshold : confidence below this marks a correct prediction as
                          "uncertain" instead of "pass" (default 0.6)

    Returns (score_0_to_100, status, predicted_class_name, None)

    score = confidence of the predicted class if it matches expected_label,
            else 0.0.  Expressed on 0–100 scale for DB storage.
    """
    if raw_output.ndim > 1:
        raw_output = raw_output.squeeze()

    probs = raw_output.astype(np.float32)
    predicted_idx = int(np.argmax(probs))
    confidence = float(probs[predicted_idx])

    if predicted_idx < len(label_names):
        predicted_label = label_names[predicted_idx]
    else:
        predicted_label = str(predicted_idx)

    if expected_label and expected_label not in ("-", ""):
        match = predicted_label.strip().lower() == expected_label.strip().lower()
        score = round(confidence * 100.0, 2) if match else 0.0
        if match:
            status = (
                TestResultStatus.pass_
                if confidence >= uncertain_threshold
                else TestResultStatus.uncertain
            )
        else:
            status = TestResultStatus.fail
    else:
        score = round(confidence * 100.0, 2)
        status = TestResultStatus.pass_

    return score, status, predicted_label, None


def _score_detection(
    raw_output: np.ndarray,
    expected_label: str,
    label_names: List[str],
    gt_boxes: Optional[List[dict]] = None,
    iou_threshold: float = IOU_THRESHOLD_MATCH,
) -> Tuple[float, TestResultStatus, Optional[str], Optional[float]]:
    """
    Score one detection output (decoded tensor: shape (N, 6) [x1,y1,x2,y2,score,cls_id]).
    Returns (best_iou * 100, pass_/fail, predicted_class_name, best_iou).
    """
    if raw_output.ndim == 3:
        raw_output = raw_output[0]

    if raw_output.shape[0] == 0:
        return 0.0, TestResultStatus.fail, None, None

    scores = raw_output[:, 4]
    mask = scores >= CONF_THRESHOLD
    dets = raw_output[mask]

    if len(dets) == 0:
        return 0.0, TestResultStatus.fail, None, None

    if not gt_boxes:
        best_score = float(np.max(dets[:, 4]))
        top_cls = int(dets[np.argmax(dets[:, 4]), 5])
        pred_cls = label_names[top_cls] if top_cls < len(label_names) else str(top_cls)
        status = TestResultStatus.pass_ if best_score >= 0.25 else TestResultStatus.fail
        return round(best_score * 100.0, 2), status, pred_cls, None

    best_iou = 0.0
    matched = False
    best_cls_name: Optional[str] = None

    if len(dets) > 0:
        top_det = dets[np.argmax(dets[:, 4])]
        top_cls = int(top_det[5])
        best_cls_name = label_names[top_cls] if top_cls < len(label_names) else str(top_cls)

    for gt in gt_boxes:
        bx = float(gt.get("x", 0)); by = float(gt.get("y", 0))
        bw = float(gt.get("w", 0)); bh = float(gt.get("h", 0))
        gt_box = [bx, by, bx + bw, by + bh]

        for det in dets:
            pred_box = [float(det[0]), float(det[1]), float(det[2]), float(det[3])]
            iou_val = _iou(pred_box, gt_box)
            if iou_val > best_iou:
                best_iou = iou_val
            if iou_val >= iou_threshold:
                matched = True

    score = round(best_iou * 100.0, 2)
    iou_stored = round(best_iou, 4)
    status = TestResultStatus.pass_ if matched else TestResultStatus.fail
    return score, status, best_cls_name, iou_stored


# ─── NMS-aware detection scorer ───────────────────────────────────────────────

def _score_detection_from_dets(
    nms_dets: list,
    label_names: List[str],
    gt_boxes: Optional[List[dict]] = None,
    iou_threshold: float = IOU_THRESHOLD_MATCH,
    label_to_idx: Optional[Dict[str, int]] = None,
) -> Tuple[float, TestResultStatus, Optional[str], Optional[float], dict]:
    """
    Score post-NMS detections against GT boxes using greedy bipartite matching.

    Predictions are sorted by confidence descending; each GT can be claimed
    by at most one prediction (and vice-versa).

    Returns (score_0_100, status, predicted_class_name, best_iou_0_1, det_stats)
    where det_stats = {"tp": int, "fp": int, "fn": int}

    Status semantics:
      pass_     – all GT matched, zero unmatched predictions above threshold
      uncertain – all GT matched, but unmatched FP predictions remain
      fail      – at least one GT was missed (FN > 0) or no predictions at all
    """
    empty_stats: dict = {"tp": 0, "fp": 0, "fn": 0}

    if not nms_dets:
        fn = len(gt_boxes) if gt_boxes else 0
        return 0.0, TestResultStatus.fail, None, None, {"tp": 0, "fp": 0, "fn": fn}

    top         = max(nms_dets, key=lambda d: d["score"])
    top_cls     = top["class_idx"]
    best_cls_name = label_names[top_cls] if top_cls < len(label_names) else str(top_cls)

    # No GT → score by confidence alone; every detection is technically a FP.
    if not gt_boxes:
        conf   = top["score"]
        status = TestResultStatus.pass_ if conf >= 0.25 else TestResultStatus.fail
        return round(conf * 100.0, 2), status, best_cls_name, None, empty_stats

    # Use the caller-supplied dual-key mapping (label_id UUID + name) when available
    # so GT boxes with label_id fields resolve correctly. Fall back to name-only.
    _lbl_map: Dict[str, int] = label_to_idx if label_to_idx is not None else {name: i for i, name in enumerate(label_names)}

    # Build GT entries with [x1,y1,x2,y2] boxes.
    gt_entries: List[dict] = []
    for box in gt_boxes:
        gt_lbl  = (box.get("label_id") or box.get("label") or "").strip()
        cls_idx = _lbl_map.get(gt_lbl)  # None only if truly unknown label → strict class enforcement
        bx = float(box.get("x", 0)); by = float(box.get("y", 0))
        bw = float(box.get("w", 0)); bh = float(box.get("h", 0))
        gt_entries.append({"box": [bx, by, bx + bw, by + bh], "cls": cls_idx})

    # Greedy matching: high-confidence predictions claim GT first.
    sorted_dets = sorted(nms_dets, key=lambda d: d["score"], reverse=True)
    matched_gt  = [False] * len(gt_entries)
    best_iou    = 0.0
    tp = fp = 0

    for det in sorted_dets:
        best_match_iou = 0.0
        best_match_idx = -1
        for i, gt in enumerate(gt_entries):
            if matched_gt[i]:
                continue
            cls_ok = (gt["cls"] is None) or (det["class_idx"] == gt["cls"])
            if not cls_ok:
                continue
            iou_val = _iou(det["box"], gt["box"])
            if iou_val > best_match_iou:
                best_match_iou = iou_val
                best_match_idx = i

        if best_match_iou >= iou_threshold and best_match_idx >= 0:
            matched_gt[best_match_idx] = True
            tp += 1
            if best_match_iou > best_iou:
                best_iou = best_match_iou
        else:
            fp += 1

    fn         = matched_gt.count(False)
    det_stats  = {"tp": tp, "fp": fp, "fn": fn}

    if fn > 0:
        status = TestResultStatus.fail
    elif fp > 0:
        status = TestResultStatus.uncertain   # All GT matched but stray FPs remain
    else:
        status = TestResultStatus.pass_

    return (
        round(best_iou * 100.0, 2),
        status,
        best_cls_name,
        round(best_iou, 4) if best_iou > 0 else None,
        det_stats,
    )


# ─── Aggregate metrics ────────────────────────────────────────────────────────

def _compute_aggregate_metrics(
    results: List[Tuple],
    label_names: List[str],
    output_type: str = "classification",
    det_image_results: Optional[list] = None,
) -> List[dict]:
    """
    Delegate to model_testing_scoring for production-grade metrics.

    Classification → confusion matrix + per-class precision/recall/F1/accuracy
    Detection      → mAP50, mAP75, mAP@COCO, per-class AP50/P/R
    """
    from app.services.model_testing_scoring import (
        compute_classification_metrics,
        compute_detection_metrics,
        compute_fomo_metrics,
        ClassificationResult,
    )

    is_box_detection = output_type in ("yolo_pro_detection", "detection", "ssd_detection")
    is_fomo          = output_type == "detection_heatmap"

    if is_box_detection and det_image_results is not None:
        return compute_detection_metrics(det_image_results, label_names)

    if is_fomo and det_image_results is not None:
        return compute_fomo_metrics(det_image_results, label_names)

    # Classification path: build (predicted, expected, confidence) triples
    cl_results: List[ClassificationResult] = []
    for sample, score, status, pred_cls, iou_val, _boxes in results:
        conf = score / 100.0
        expected = (sample.expected_outcome or "").strip()
        if not expected or expected == "-":
            continue
        if status == TestResultStatus.pass_:
            predicted = expected
        else:
            predicted = pred_cls if pred_cls else f"__wrong__{sample.id}"

        cl_results.append((predicted, expected, conf))

    if not cl_results:
        return []

    return compute_classification_metrics(cl_results, label_names)


# ─── Label name resolution ────────────────────────────────────────────────────

def _resolve_label_names(impulse: Impulse, trained_model: TrainedModel, db: Session) -> List[str]:
    """
    Return ordered label names for the impulse.

    Priority:
      1. model_metadata["label_names"] — written at export time
      2. Labels from the DB ordered by name (fallback)
    """
    meta = trained_model.model_metadata or {}
    if meta.get("label_names"):
        return list(meta["label_names"])

    labels = (
        db.query(Label)
        .filter(Label.project_id == impulse.project_id)
        .order_by(Label.name.asc())
        .all()
    )
    return [lbl.name for lbl in labels]


def _resolve_label_to_idx(impulse: Impulse, label_names: List[str], db: Session) -> Dict[str, int]:
    """Build a dual-key label lookup: both UUID label_id and name string."""
    label_to_idx: Dict[str, int] = {name: i for i, name in enumerate(label_names)}
    if not label_names:
        return label_to_idx

    labels = db.query(Label).filter(Label.project_id == impulse.project_id).all()
    name_to_idx = {name: i for i, name in enumerate(label_names)}
    for lbl in labels:
        idx = name_to_idx.get(lbl.name)
        if idx is not None:
            label_to_idx[lbl.id] = idx
            label_to_idx[lbl.name] = idx
    return label_to_idx


from app.ml.fomo_evaluator import (
    decode_fomo_heatmap as _fomo_decode_heatmap,
    gt_cells_from_boxes as _fomo_gt_cells_from_boxes_raw,
    score_sample as _fomo_score_sample,
    resolve_fomo_threshold, ThresholdSource,
    log_fomo_run_diagnostics as _fomo_log_diagnostics,
    FOMO_SWEEP_THRESHOLDS as _FOMO_SWEEP_THRESHOLDS,
)

def _fomo_gt_cells_from_boxes(norm_gt, fH, fW, label_to_idx, sample_id=None):
    cells, _ = _fomo_gt_cells_from_boxes_raw(norm_gt, fH, fW, label_to_idx, sample_id=sample_id)
    return cells

def _fomo_per_sample_score(pred_cells, gt_cells, n_classes, label_names):
    r = _fomo_score_sample(pred_cells, gt_cells, n_classes, label_names)
    return r["score_0_100"], r["status"], r["pred_cls"], r["f1"], r["tp"], r["fp"], r["fn"]

def _pick_fomo_model_testing_threshold(items, label_names, preferred_threshold):
    thr, _ = resolve_fomo_threshold(mode=ThresholdSource.DEPLOYED_MODEL, sweep_items=items, label_names=label_names, preferred_threshold=preferred_threshold)
    return thr


# MobileNetV2-SSD scores detections at this confidence during its post-training
# mAP evaluation (mobilenetv2_ssd_worker._ssd_evaluate_map, conf_threshold=0.01).
# SSD logits are tiny — top scores cluster around 0.01-0.02 — so the
# classification default (0.25) wipes out every detection and Model Testing
# reports TP=0/FN>0 even though training eval shows a healthy mAP. Score SSD at
# the same confidence training used so the two agree (mirrors FOMO's
# TRAINING_PARITY resolution). This is SSD-scoped; YOLO-Pro keeps its own
# stamped threshold and the 0.25 classification default is untouched elsewhere.
SSD_TRAINING_EVAL_CONF: float = 0.01

# Per-image detection presentation for the SSD per-sample view (Sample
# Inspector + the TP/FP pass-fail count).
#
# SSD cls heads are initialised with a focal-loss prior bias (~ -4.6 ≈ 1%) and a
# weakly-trained model barely moves off it: every anchor's sigmoid score sits in
# a thin band (~0.008-0.013) with the true object only marginally above the
# background floor. The ranking is still correct (that is why mAP@50 is healthy),
# but no fixed absolute threshold can separate signal from noise the way
# YOLO-Pro's 0.25 does — at the conf=0.01 mAP threshold ~60 background anchors
# survive NMS and flood the view (TP=1/FP=60).
#
# So the per-sample view uses an *adaptive* gate, calibrated to each image's own
# score distribution (mirrors FOMO's deployed-model threshold calibration):
# keep only detections within SSD_REL_CONF_RATIO of the image's top score, then
# cap to SSD_MODEL_TESTING_MAX_DETS (the SSD deployment top-K, primitives.py:68).
# This surfaces the model's most-confident detection(s) cleanly (~1 box/image)
# instead of the 1% anchor grid. The aggregate-mAP detection set is left untouched
# (conf=0.01, max_dets=100) so the headline mAP still matches the training eval.
SSD_REL_CONF_RATIO: float = 0.95
SSD_MODEL_TESTING_MAX_DETS: int = 10


def _ssd_per_sample_dets(nms_dets: List[dict]) -> List[dict]:
    """Adaptive per-image gate for the SSD per-sample view.

    Keeps detections whose score is within SSD_REL_CONF_RATIO of the image's top
    score (floored at SSD_TRAINING_EVAL_CONF), then caps to the SSD deployment
    top-K. Input is assumed already NMS'd. Returns a score-sorted slice.
    """
    if not nms_dets:
        return []
    top_score = max(d["score"] for d in nms_dets)
    gate = max(SSD_TRAINING_EVAL_CONF, top_score * SSD_REL_CONF_RATIO)
    kept = [d for d in nms_dets if d["score"] >= gate]
    kept.sort(key=lambda d: d["score"], reverse=True)
    return kept[:SSD_MODEL_TESTING_MAX_DETS]


def _ssd_box_display_score(box: List[float], class_idx: int, gt_by_class: dict) -> float:
    """Score shown on the SSD inspector overlay: IoU match quality against the
    best same-class ground-truth box, not the model's raw confidence.

    A weakly-trained SSD reports a near-uniform ~1% confidence for every anchor,
    so the raw score is useless on the overlay ("tiger 1%"). The IoU against GT
    tells the user how well the box actually localises the object — the same
    quantity the table's MATCH SCORE column reports. Returns 0.0 when there is no
    same-class GT (i.e. the detection's class is wrong / unmatched).
    """
    best = 0.0
    for g in gt_by_class.get(class_idx, []):
        best = max(best, _iou(box, g))
    return best


def _resolve_ssd_threshold(
    trained_model: TrainedModel,
    training_cr: Optional[dict],
) -> Tuple[float, str]:
    """Pick the confidence Model Testing should score an SSD model at.

    Priority (parity with the training mAP evaluator):
      1. threshold stamped into model_metadata (if a future export records one)
      2. threshold recorded in the training classification_report
      3. SSD_TRAINING_EVAL_CONF (0.01) — the evaluator's fixed default

    Never falls back to the caller's classification default (0.25), which would
    collapse every SSD detection to zero.
    """
    meta = trained_model.model_metadata or {}
    meta_thr = meta.get("threshold")
    if meta_thr is not None:
        return float(meta_thr), "model_metadata.threshold"
    if isinstance(training_cr, dict):
        cr_thr = training_cr.get("threshold", training_cr.get("conf_threshold"))
        if cr_thr is not None:
            return float(cr_thr), "training classification_report"
    return SSD_TRAINING_EVAL_CONF, "SSD training-parity default (0.01)"


def _resolve_yolo_pro_threshold(
    trained_model: TrainedModel,
    training_cr: Optional[dict],
    caller_conf: float,
) -> Tuple[float, str]:
    """Pick the confidence Model Testing should score a YOLO-Pro model at.

    Priority:
      1. classification_report["conf_threshold"] — the eval-parity value
         (``_eval_conf``, ~0.005) that produced the training-time mAP50
         reported as ``restored_full_safety_map50``.
      2. model_metadata["threshold"] — legacy fallback for jobs trained
         before the eval threshold was persisted.
      3. the caller-supplied value.

    model_metadata["threshold"] is deliberately NOT preferred here: the
    YOLO-Pro worker stamps the RUNTIME/deployment threshold into that field
    (``_compute_runtime_conf_threshold`` — 0.25, or 0.35 for tiny-dataset /
    low-step runs), which is an order of magnitude above the eval threshold.
    Scoring the PR curve at 0.25 truncates recall and collapses mAP@50 far
    below the training log (e.g. 6.3% vs 93.81%). Deployment/inference paths
    still read model_metadata["threshold"] — only eval scoring swaps in the
    eval-parity value.
    """
    if isinstance(training_cr, dict):
        cr_thr = training_cr.get("conf_threshold")
        if cr_thr is not None:
            return float(cr_thr), "training classification_report.conf_threshold"
    meta_thr = (trained_model.model_metadata or {}).get("threshold")
    if meta_thr is not None:
        return float(meta_thr), "model_metadata.threshold (runtime default — no eval conf on job)"
    return float(caller_conf), "caller default"


def _decoded_rows_to_dets(sample_out, conf_threshold: float) -> List[dict]:
    """Filter a decoded detection tensor (N, 6) = [x1,y1,x2,y2,score,cls] into
    the {class_idx, score, box} dicts the NMS/scoring path consumes.

    Shared by the batch (_run_classify_all) and single (reclassify_single_sample)
    detection paths so both apply the same confidence gate identically.
    """
    raw_dets: List[dict] = []
    if isinstance(sample_out, np.ndarray) and sample_out.ndim == 2 and sample_out.shape[1] == 6:
        for row in sample_out:
            s = float(row[4])
            if s >= conf_threshold:
                raw_dets.append({
                    "class_idx": int(row[5]),
                    "score":     s,
                    "box":       [float(row[0]), float(row[1]),
                                  float(row[2]), float(row[3])],
                })
    return raw_dets


def detect_ssd_for_live(
    sample,
    impulse: Impulse,
    db: Session,
    model_version_id: Optional[str] = None,
) -> Optional[dict]:
    """SSD detection for Live Classification, using the SAME decode + scoring as
    Model Testing / evaluation (single source of truth).

    Returns the live-classification detection envelope when the impulse's active
    model is MobileNetV2 SSD FPN-Lite; returns ``None`` for every other model
    type so the caller falls through to its normal (classification/FOMO/YOLO)
    path. Resolution and the cheap output-type check happen BEFORE any DSP
    preprocessing, so non-SSD calls pay only a couple of metadata queries.

    Parity with the evaluation path:
      - decoded_float32 (1, N, 6) variant via _load_trained_model preference,
      - conf threshold from _resolve_ssd_threshold (0.01, never the 0.25
        classification default that zeroes out every SSD detection),
      - class-agnostic NMS + the adaptive per-image gate (_ssd_per_sample_dets),
      - normalized [0,1] xyxy box coordinates (same as YOLO-Pro decoded).
    """
    try:
        trained_model, output_type = _load_trained_model(impulse.id, model_version_id, db)
    except HTTPException:
        return None
    if output_type != "ssd_detection":
        return None  # not SSD — caller handles it on the normal path

    img = _preprocess_sample(sample, impulse)
    if img is None:
        return None

    label_names = _resolve_label_names(impulse, trained_model, db)

    # Confidence parity with the training mAP evaluator (0.01), resolved from
    # model_metadata / training classification_report with a 0.01 fallback.
    _ssd_cr = None
    if trained_model.training_job_id:
        _tj = db.query(TrainingJob).filter(
            TrainingJob.id == trained_model.training_job_id
        ).first()
        if _tj and isinstance(_tj.classification_report, dict):
            _ssd_cr = _tj.classification_report
    conf_threshold, conf_source = _resolve_ssd_threshold(trained_model, _ssd_cr)

    model_handle = _load_model_into_memory(trained_model)
    image_batch = np.stack([img], axis=0).astype(np.float32)
    # SSD decoded TFLite keeps a uint8 input + in-graph norm — feed raw [0,255].
    image_batch = _scale_to_uint8_pixel_range(model_handle, image_batch)
    raw_outputs = _run_inference(model_handle, image_batch)

    sample_out = raw_outputs[0] if isinstance(raw_outputs, list) else raw_outputs[0]
    raw_dets = _decoded_rows_to_dets(sample_out, conf_threshold)

    from app.ml.yolo_pro_worker import _nms
    nms_dets = _nms(raw_dets, iou_threshold=0.45, max_dets=100, class_agnostic=True)
    scored_dets = _ssd_per_sample_dets(nms_dets)

    detections = [
        {
            "label": (
                label_names[d["class_idx"]]
                if 0 <= d["class_idx"] < len(label_names)
                else str(d["class_idx"])
            ),
            "confidence": float(d["score"]),
            "bbox": {
                "x1": float(d["box"][0]), "y1": float(d["box"][1]),
                "x2": float(d["box"][2]), "y2": float(d["box"][3]),
            },
        }
        for d in scored_dets
    ]

    meta = trained_model.model_metadata or {}
    return {
        # Top label/confidence mirror the detection list for summary widgets.
        "label":        detections[0]["label"] if detections else None,
        "confidence":   detections[0]["confidence"] if detections else None,
        "detections":   detections,
        "count":        len(detections),
        "model_type":   "ssd_detection",
        "is_fomo":      False,
        "is_detection": True,
        "model_id":     trained_model.id,
        "model_version": trained_model.version,
        "model_architecture": meta.get("architecture", "mobilenet_v2_ssd_fpn_lite"),
        "debug": {
            "threshold":        conf_threshold,
            "threshold_source": conf_source,
            "dets_pre_gate":    len(nms_dets),
        },
    }


def _run_classify_all(
    impulse: Impulse,
    test_samples: List[ModelTestSample],
    trained_model: TrainedModel,
    output_type: str,
    db: Session,
    conf_threshold: float = 0.25,
) -> Tuple[List[Tuple], List[str], list]:
    """
    Load the model once, run inference on all test samples in batches.

    Returns:
        results          : [(ModelTestSample, score_0_100, status, pred_cls, iou_val, top_boxes), ...]
        label_names      : ordered class names used by the model
        det_image_results: per-image detection data for mAP scoring (detection only)
    """
    from app.workers.sample_utils import normalize_bounding_boxes
    from app.workers.dsp_worker import make_features_storage_key, compute_dsp_config_hash

    # Gate unsupported detection types (e.g. SSD) before any model load /
    # inference, so the run fails fast with a clear message rather than
    # crashing inside _score_classification on a (N, 6) detection tensor.
    if output_type in MODEL_TESTING_UNSUPPORTED_OUTPUT_TYPES:
        raise UnsupportedModelTestingError(_UNSUPPORTED_TESTING_MSG.format(ot=output_type))

    model_handle = _load_model_into_memory(trained_model)
    label_names = _resolve_label_names(impulse, trained_model, db)
    label_to_idx = _resolve_label_to_idx(impulse, label_names, db)

    # SSD: score at the training-eval confidence (0.01), never the 0.25
    # classification default — SSD logits are tiny and 0.25 zeroes out every
    # detection (see _resolve_ssd_threshold). Resolved from metadata / training
    # report with a 0.01 fallback.
    _caller_conf = conf_threshold
    if output_type == "ssd_detection":
        _ssd_cr = None
        if trained_model.training_job_id:
            _tj_ssd = db.query(TrainingJob).filter(
                TrainingJob.id == trained_model.training_job_id
            ).first()
            if _tj_ssd and isinstance(_tj_ssd.classification_report, dict):
                _ssd_cr = _tj_ssd.classification_report
        conf_threshold, _ssd_src = _resolve_ssd_threshold(trained_model, _ssd_cr)
        logger.info(
            "[model_testing] SSD conf_threshold=%.4f (source: %s; caller passed %.4f).",
            conf_threshold, _ssd_src, _caller_conf,
        )
    # YOLO-Pro: score at the training-eval confidence recorded on the training
    # job, NOT model_metadata["threshold"] — that field holds the runtime /
    # deployment threshold (see _resolve_yolo_pro_threshold).
    elif output_type == "yolo_pro_detection":
        _yp_cr = None
        if trained_model.training_job_id:
            _tj_yp = db.query(TrainingJob).filter(
                TrainingJob.id == trained_model.training_job_id
            ).first()
            if _tj_yp and isinstance(_tj_yp.classification_report, dict):
                _yp_cr = _tj_yp.classification_report
        conf_threshold, _yp_src = _resolve_yolo_pro_threshold(
            trained_model, _yp_cr, conf_threshold
        )
        logger.info(
            "[model_testing] YOLO-Pro conf_threshold=%.4f (source: %s; caller passed %.4f).",
            conf_threshold, _yp_src, _caller_conf,
        )
    # Other detection types keep the legacy model_metadata threshold.
    else:
        _meta_threshold = (trained_model.model_metadata or {}).get("threshold")
        if _meta_threshold is not None and output_type == "detection":
            _effective_conf = float(_meta_threshold)
            logger.debug(
                f"[model_testing] Using threshold={_effective_conf} from model_metadata "
                f"(caller passed conf_threshold={conf_threshold})."
            )
            conf_threshold = _effective_conf

    sample_ids = [ts.sample_id for ts in test_samples if ts.sample_id]
    real_samples_map: dict = {}
    if sample_ids:
        for s in db.query(Sample).filter(Sample.id.in_(sample_ids)).all():
            real_samples_map[s.id] = s

    # ── DSP feature cache lookup (avoids S3 re-download + DSP reprocessing) ────
    _dsp_feature_cache: dict = {}   # sample_id -> np.ndarray
    try:
        import io as _io
        import json as _json
        import numpy as _np_cache
        _cache_key = make_features_storage_key(impulse.project_id, impulse.id)
        _cache_bytes = storage.download_bytes(_cache_key)
        with _np_cache.load(_io.BytesIO(_cache_bytes), allow_pickle=True) as _npz:
            if "meta_json" in _npz and "ids" in _npz and "X" in _npz:
                _meta = _json.loads(str(_npz["meta_json"][0]))
                _expected_hash = compute_dsp_config_hash(impulse.dsp_blocks or [{"type": "raw", "params": {}}])
                if _meta.get("impulse_id") == impulse.id and _meta.get("dsp_config_hash") == _expected_hash:
                    for _sid, _feat in zip(_npz["ids"].tolist(), _npz["X"]):
                        _dsp_feature_cache[str(_sid)] = _np_cache.asarray(_feat, dtype=np.float32)
                    logger.info(
                        f"[model_testing] DSP cache hit for impulse {impulse.id} "
                        f"— {len(_dsp_feature_cache)} features loaded, skipping S3/DSP"
                    )
    except Exception as _e:
        logger.debug(f"[model_testing] DSP cache not available, will preprocess from S3: {_e}")

    # Target shape for one sample — cache stores flat rows, model needs (H, W, C)
    dsp_blocks = impulse.dsp_blocks or [{"type": "image", "params": {}}]
    image_block = dsp_blocks[0]
    merged_img_params = merge_image_params(impulse, dict(image_block.get("params", {})))
    dsp_img_w = int(merged_img_params.get("image_width") or getattr(impulse, "image_width", None) or 96)
    dsp_img_h = int(merged_img_params.get("image_height") or getattr(impulse, "image_height", None) or 96)
    dsp_grayscale = bool(merged_img_params.get("grayscale", False))
    dsp_img_c = 1 if dsp_grayscale else 3
    dsp_img_shape = (dsp_img_h, dsp_img_w, dsp_img_c)
    model_img_shape = _resolve_model_input_shape(model_handle, trained_model, dsp_img_shape)
    cache_img_shape = (
        int(_meta.get("image_height") or dsp_img_h) if "_meta" in locals() and isinstance(_meta, dict) else dsp_img_h,
        int(_meta.get("image_width") or dsp_img_w) if "_meta" in locals() and isinstance(_meta, dict) else dsp_img_w,
        dsp_img_c,
    )

    is_detection = output_type in ("yolo_pro_detection", "detection", "ssd_detection")
    is_ssd       = output_type == "ssd_detection"
    is_yolo_pro  = output_type == "yolo_pro_detection"
    is_fomo      = output_type == "detection_heatmap"

    # Aggregate mAP scores YOLO-Pro at the permissive eval conf (~0.005) so the
    # PR curve matches training. The PER-SAMPLE view (inspector overlay + the
    # TP/FP status in the table) must NOT: at 0.005 every image returns the full
    # 100-detection NMS set and the overlay becomes unreadable. Gate that view
    # at the RUNTIME threshold from model_metadata — what the deployed model
    # would actually emit, and what this view showed before the eval-parity fix.
    _yp_display_conf = conf_threshold
    if is_yolo_pro:
        _meta_disp_thr = (trained_model.model_metadata or {}).get("threshold")
        _yp_display_conf = max(
            float(_meta_disp_thr) if _meta_disp_thr is not None else float(_caller_conf),
            conf_threshold,
        )
        logger.info(
            "[model_testing] YOLO-Pro per-sample display gate=%.4f "
            "(mAP scored at %.4f).", _yp_display_conf, conf_threshold,
        )

    if is_fomo:
        from app.ml.fomo_evaluator import assert_class_ordering
        meta_names = list((trained_model.model_metadata or {}).get("label_names", []))
        if meta_names:
            assert_class_ordering(meta_names, label_names, context="model_testing")
        else:
            logger.error(
                "[FOMO model_testing] model_metadata missing label_names — "
                "falling back to DB alphabetical order; metrics may be wrong. "
                "model_id=%s", trained_model.id,
            )

    # Fix 1: read FOMO threshold from training classification_report, not hardcoded caller value.
    # classification_report["threshold"] is set by _evaluate_fomo_detection via threshold sweep.
    _fomo_conf_threshold: float = conf_threshold   # initial hint; resolved below
    _fomo_iou_threshold:  float = 0.5
    _fomo_threshold_source: str = "caller default (non-FOMO path)"
    _fomo_training_cr: Optional[dict] = None

    if is_fomo and trained_model.training_job_id:
        _tj = db.query(TrainingJob).filter(
            TrainingJob.id == trained_model.training_job_id
        ).first()
        if _tj and isinstance(_tj.classification_report, dict):
            _fomo_training_cr = _tj.classification_report
            _fomo_iou_threshold = float(_fomo_training_cr.get("iou_threshold", 0.5))
        else:
            logger.warning(
                "[FOMO eval] training_job=%s has no classification_report; "
                "threshold will be calibrated via deployed-model sweep.",
                trained_model.training_job_id,
            )

    # NOTE: For the batch path we default to TRAINING_PARITY so metrics match
    # the training evaluation.  If training_cr is missing we fall through to
    # DEPLOYED_MODEL calibration at the end of the fomo_pending loop.
    if is_fomo and _fomo_training_cr is not None:
        try:
            _fomo_conf_threshold, _fomo_threshold_source = resolve_fomo_threshold(
                mode=ThresholdSource.TRAINING_PARITY,
                training_classification_report=_fomo_training_cr,
            )
        except ValueError as _te:
            logger.error(
                "[FOMO eval] resolve_fomo_threshold(TRAINING_PARITY) failed: %s. "
                "Will recalibrate via DEPLOYED_MODEL sweep.", _te,
            )
            _fomo_training_cr = None   # trigger sweep fallback below

    results: List[Tuple] = []
    det_image_results: list = []
    fomo_pending: List[Tuple[ModelTestSample, np.ndarray, Dict[Tuple[int, int], int], int, int, int]] = []

    for batch_start in range(0, len(test_samples), INFERENCE_BATCH_SIZE):
        batch = test_samples[batch_start: batch_start + INFERENCE_BATCH_SIZE]

        images: List[np.ndarray] = []
        valid_indices: List[int] = []

        for local_i, ts in enumerate(batch):
            real_sample = real_samples_map.get(ts.sample_id) if ts.sample_id else None
            if real_sample is None:
                logger.warning(f"No real sample row for test_sample {ts.id} — skipping")
                results.append((ts, 0.0, TestResultStatus.fail, None, None, None))
                continue

            cached_feat = _dsp_feature_cache.get(str(ts.sample_id)) if ts.sample_id else None
            if cached_feat is not None:
                try:
                    img = _coerce_image_to_shape(cached_feat, cache_img_shape, model_img_shape)
                except Exception:
                    img = _preprocess_sample(real_sample, impulse)
            else:
                img = _preprocess_sample(real_sample, impulse)
            if img is None:
                results.append((ts, 0.0, TestResultStatus.fail, None, None, None))
                continue

            try:
                img = _coerce_image_to_shape(img, dsp_img_shape, model_img_shape)
            except Exception as e:
                logger.warning(
                    f"[model_testing] Skipping sample {ts.sample_id or ts.id}: "
                    f"could not coerce input tensor to {model_img_shape}: {e}"
                )
                results.append((ts, 0.0, TestResultStatus.fail, None, None, None))
                continue

            images.append(img)
            valid_indices.append(local_i)

        if not images:
            continue

        try:
            image_batch = np.stack(images, axis=0).astype(np.float32)
            if is_fomo and model_handle[0] == "tflite":
                # TFLite export strips the "normalize" layer; apply manually.
                # Keras artifacts still contain the layer, so skip this.
                image_batch = image_batch * 2.0 - 1.0
            elif is_ssd:
                # SSD decoded TFLite has a uint8 input + in-graph normalization;
                # feed raw [0,255] pixels (DSP delivers normalized [0,1]).
                image_batch = _scale_to_uint8_pixel_range(model_handle, image_batch)
            raw_outputs = _run_inference(model_handle, image_batch)
        except Exception as e:
            logger.error(f"Inference failed on batch starting at {batch_start}: {e}", exc_info=True)
            for local_i in valid_indices:
                results.append((batch[local_i], 0.0, TestResultStatus.fail, None, None, None))
            continue

        for out_i, local_i in enumerate(valid_indices):
            ts = batch[local_i]
            real_sample = real_samples_map.get(ts.sample_id)
            top_boxes = None

            if is_detection:
                if isinstance(raw_outputs, np.ndarray) and raw_outputs.ndim == 3:
                    sample_out = raw_outputs[out_i]
                elif isinstance(raw_outputs, list):
                    sample_out = raw_outputs[0][out_i]
                else:
                    sample_out = raw_outputs

                raw_gt_boxes = []
                if real_sample:
                    raw_gt_boxes = (real_sample.extra_metadata or {}).get("boundingBoxes", [])

                raw_bytes = None
                if real_sample and real_sample.storage_key:
                    try:
                        raw_bytes = storage.download_bytes(real_sample.storage_key)
                    except Exception:
                        raw_bytes = None
                norm_gt = normalize_bounding_boxes(raw_gt_boxes, image_bytes=raw_bytes)

                gt_by_class: dict = {}
                for box in norm_gt:
                    key = box.get("label_id") or box.get("label") or ""
                    cls_idx = label_to_idx.get(key.strip())
                    if cls_idx is None:
                        continue
                    x1 = float(box.get("x", 0))
                    y1 = float(box.get("y", 0))
                    x2 = x1 + float(box.get("w", 0))
                    y2 = y1 + float(box.get("h", 0))
                    if x2 > x1 and y2 > y1:
                        gt_by_class.setdefault(cls_idx, []).append([x1, y1, x2, y2])

                raw_dets = _decoded_rows_to_dets(sample_out, conf_threshold)

                from app.ml.yolo_pro_worker import _nms
                nms_dets = _nms(raw_dets, iou_threshold=0.45, max_dets=100, class_agnostic=True)

                # The full NMS set (up to 100) feeds aggregate mAP so the
                # headline matches the training eval (also max_dets=100).
                pred_boxes_list: List[List[float]] = []
                pred_scores_list: List[float] = []
                pred_cls_list: List[int] = []
                for d in nms_dets:
                    pred_boxes_list.append(d["box"])
                    pred_scores_list.append(d["score"])
                    pred_cls_list.append(d["class_idx"])

                det_image_results.append((
                    pred_boxes_list,
                    pred_scores_list,
                    pred_cls_list,
                    gt_by_class,
                ))

                # Per-sample view (TP/FP + inspector): for SSD apply the adaptive
                # per-image gate so only the model's most-confident detection(s)
                # show, instead of the conf=0.01 anchor grid. YOLO-Pro re-gates
                # at the runtime threshold for the same reason (aggregate mAP
                # above already consumed the full low-conf set). Plain detection
                # keeps the full set — it gates on a higher conf and doesn't flood.
                if is_ssd:
                    scored_dets = _ssd_per_sample_dets(nms_dets)
                elif is_yolo_pro:
                    scored_dets = [d for d in nms_dets if d["score"] >= _yp_display_conf]
                else:
                    scored_dets = nms_dets

                score, status, pred_cls, iou_val, det_stats = _score_detection_from_dets(
                    scored_dets,
                    label_names,
                    gt_boxes=norm_gt if norm_gt else None,
                    iou_threshold=IOU_THRESHOLD_MATCH,
                    label_to_idx=label_to_idx,
                )
                top_boxes = {
                    "v": 2,
                    "boxes": [
                        {"x": d["box"][0], "y": d["box"][1],
                         "x2": d["box"][2], "y2": d["box"][3],
                         # SSD overlay shows IoU match quality (raw conf is a
                         # meaningless ~1%); other detectors show real confidence.
                         "score": round(
                             _ssd_box_display_score(d["box"], d["class_idx"], gt_by_class)
                             if is_ssd else d["score"],
                             4,
                         ),
                         "label": label_names[d["class_idx"]]
                                   if d["class_idx"] < len(label_names)
                                   else str(d["class_idx"])}
                        for d in scored_dets[:20]
                    ],
                    **det_stats,
                }
            else:
                if isinstance(raw_outputs, np.ndarray):
                    sample_out = raw_outputs[out_i]
                else:
                    sample_out = raw_outputs[out_i]

                if is_fomo:
                    fomo_out = np.array(sample_out, dtype=np.float32)
                    if fomo_out.ndim == 4 and fomo_out.shape[0] == 1:
                        fomo_out = fomo_out[0]
                    if fomo_out.ndim != 3:
                        score, status, pred_cls, iou_val, top_boxes = (
                            0.0, TestResultStatus.fail, None, None, None
                        )
                    else:
                        try:
                            fH, fW, _ = fomo_out.shape
                            # _fomo_decode_heatmap applies softmax + full-channel argmax
                            pred_cells, probs = _fomo_decode_heatmap(fomo_out, _fomo_conf_threshold)

                            # GT cells: center-point of each normalised bounding box
                            fomo_gt_raw: list = []
                            fomo_raw_bytes = None
                            if real_sample:
                                fomo_gt_raw = (real_sample.extra_metadata or {}).get("boundingBoxes", [])
                                if real_sample.storage_key:
                                    try:
                                        fomo_raw_bytes = storage.download_bytes(real_sample.storage_key)
                                    except Exception as _dl_err:
                                        logger.warning(
                                            "[FOMO eval] sample %s: image download failed; "
                                            "GT normalization will use raw coordinates (may be wrong): %s",
                                            ts.sample_id, _dl_err,
                                        )
                            fomo_norm_gt = normalize_bounding_boxes(fomo_gt_raw, image_bytes=fomo_raw_bytes)
                            if fomo_raw_bytes is None and any(
                                max(abs(float(b.get("x", 0))), abs(float(b.get("y", 0))),
                                    abs(float(b.get("w", 0))), abs(float(b.get("h", 0)))) > 1.0
                                for b in fomo_norm_gt if isinstance(b, dict)
                            ):
                                logger.warning(
                                    "[FOMO eval] sample %s: image bytes unavailable and GT boxes "
                                    "appear to be pixel-space coordinates — normalization was skipped; "
                                    "GT cells will be clamped and are likely wrong.",
                                    ts.sample_id,
                                )
                            gt_cells, _n_unres_i = _fomo_gt_cells_from_boxes_raw(fomo_norm_gt, fH, fW, label_to_idx)
                            fomo_pending.append((ts, probs, gt_cells, fH, fW, _n_unres_i))
                            continue
                        except Exception as _build_err:
                            logger.warning(
                                "[FOMO eval] sample %s: pending-item build failed, scoring as failed: %s",
                                ts.sample_id, _build_err,
                            )
                            score, status, pred_cls, iou_val, top_boxes = (
                                0.0, TestResultStatus.fail, None, None, None
                            )
                else:
                    score, status, pred_cls, iou_val = _score_classification(
                        sample_out,
                        ts.expected_outcome or "-",
                        label_names,
                    )

            results.append((ts, score, status, pred_cls, iou_val, top_boxes))

    if is_fomo and fomo_pending:
        # Threshold mode: TRAINING_PARITY if we loaded a valid training CR;
        # DEPLOYED_MODEL sweep otherwise.
        if _fomo_training_cr is not None:
            # Already resolved above — reuse to stay consistent.
            selected_threshold = _fomo_conf_threshold
            selected_source    = _fomo_threshold_source
        else:
            # No training CR available: calibrate over this exported model's outputs.
            selected_threshold, selected_source = resolve_fomo_threshold(
                mode=ThresholdSource.DEPLOYED_MODEL,
                sweep_items=[(probs, gt_cells) for _ts, probs, gt_cells, _fH, _fW, _nu in fomo_pending],
                label_names=label_names,
                preferred_threshold=_fomo_conf_threshold,
            )

        logger.info(
            "[FOMO eval] final threshold=%.4f  source=%s",
            selected_threshold, selected_source,
        )

        n_classes = len(label_names)
        # Accumulate for aggregate diagnostics
        _diag_tp   = np.zeros(n_classes, dtype=np.int64)
        _diag_fp   = np.zeros(n_classes, dtype=np.int64)
        _diag_fn   = np.zeros(n_classes, dtype=np.int64)
        _total_gt  = 0
        _n_unres   = sum(t[5] for t in fomo_pending)

        for ts, probs, gt_cells, fH, fW, _nu in fomo_pending:
            pred_cells, _ = _fomo_decode_heatmap(probs, selected_threshold)
            score, status, pred_cls, iou_val, tp_arr, fp_arr, fn_arr = _fomo_per_sample_score(
                pred_cells, gt_cells, n_classes, label_names,
            )
            det_image_results.append((tp_arr, fp_arr, fn_arr))
            _diag_tp += tp_arr; _diag_fp += fp_arr; _diag_fn += fn_arr
            _total_gt += len(gt_cells)
            top_boxes = [
                {
                    "x": c / fW, "y": r / fH,
                    "x2": (c + 1) / fW, "y2": (r + 1) / fH,
                    "score": round(float(probs[r, c, cls + 1]), 4),
                    "label": label_names[cls] if cls < n_classes else str(cls),
                }
                for (r, c), cls in list(pred_cells.items())[:20]
            ]
            results.append((ts, score, status, pred_cls, iou_val, top_boxes))

        _fomo_log_diagnostics(
            threshold=selected_threshold,
            threshold_source=selected_source,
            total_gt_cells=_total_gt,
            total_pred_cells=int(_diag_tp.sum() + _diag_fp.sum()),
            tp=_diag_tp, fp=_diag_fp, fn=_diag_fn,
            label_names=label_names,
            n_unresolved_gt=_n_unres,
            context="classify_all",
        )

    return results, label_names, det_image_results

# ─── Public service API ───────────────────────────────────────────────────────

def get_page_data(
    impulse_id: Optional[str],
    project_id: Optional[str],
    db: Session,
) -> dict:
    impulse = _resolve_impulse(impulse_id, project_id, db)

    available_impulses: List[dict] = []
    project = None

    if impulse:
        project = db.query(Project).filter(Project.id == impulse.project_id).first()
        available_impulses = [
            _impulse_out(i)
            for i in db.query(Impulse)
            .filter(Impulse.project_id == impulse.project_id)
            .order_by(Impulse.created_at.asc())
            .all()
        ]
    elif project_id:
        project = db.query(Project).filter(Project.id == project_id).first()
        available_impulses = [
            _impulse_out(i)
            for i in db.query(Impulse)
            .filter(Impulse.project_id == project_id)
            .order_by(Impulse.created_at.asc())
            .all()
        ]

    if not impulse:
        return {
            "project":               {"id": project.id if project else None,
                                       "name": project.name if project else None},
            "impulse":               None,
            "available_impulses":    available_impulses,
            "selected_model_version": None,
            "latest_run":            None,
            "accuracy":              None,
            "metrics":               [],
            "summary":               {"total": 0, "passed": 0, "failed": 0, "uncertain": 0, "pending": 0},
            "target_device":         "Arduino UNO Q (Cortex-M0+)",
        }

    all_samples = _ensure_test_samples_provisioned(impulse, db)
    db.commit()

    all_samples = (
        db.query(ModelTestSample)
        .filter(ModelTestSample.impulse_id == impulse.id)
        .all()
    )

    active_version = _get_active_version(impulse.id, None, db)
    latest_run     = _latest_test_run(impulse.id, db)

    passed    = sum(1 for s in all_samples if s.result_status == TestResultStatus.pass_)
    failed    = sum(1 for s in all_samples if s.result_status == TestResultStatus.fail)
    uncertain = sum(1 for s in all_samples if s.result_status == TestResultStatus.uncertain)
    pending   = sum(1 for s in all_samples if s.result_status == TestResultStatus.pending)
    metrics = [_metric_out(m) for m in latest_run.metrics] if latest_run else []

    return {
        "project": {
            "id":   project.id   if project else None,
            "name": project.name if project else None,
        },
        "impulse":               _impulse_out(impulse),
        "available_impulses":    available_impulses,
        "selected_model_version": _version_out(active_version) if active_version else None,
        "latest_run":             _run_out(latest_run)          if latest_run     else None,
        "accuracy":               latest_run.accuracy           if latest_run     else None,
        "metrics":                metrics,
        "summary": {
            "total":     len(all_samples),
            "passed":    passed,
            "failed":    failed,
            "uncertain": uncertain,
            "pending":   pending,
        },
        "target_device": "Arduino UNO Q (Cortex-M0+)",
    }


def get_test_data(
    impulse_id: str,
    page: int,
    page_size: int,
    db: Session,
) -> dict:
    impulse = db.query(Impulse).filter(Impulse.id == impulse_id).first()
    if not impulse:
        raise HTTPException(404, f"Impulse '{impulse_id}' not found")

    _ensure_test_samples_provisioned(impulse, db)
    db.commit()

    q = (
        db.query(ModelTestSample)
        .filter(ModelTestSample.impulse_id == impulse_id)
        .order_by(ModelTestSample.created_at.asc())
    )
    total   = q.count()
    samples = q.offset((page - 1) * page_size).limit(page_size).all()

    # Batch-load underlying Sample rows to get stored GT bounding boxes.
    sid_list = [s.sample_id for s in samples if s.sample_id]
    raw_sample_map: dict = {}
    if sid_list:
        raw_samples = db.query(Sample).filter(Sample.id.in_(sid_list)).all()
        raw_sample_map = {rs.id: rs for rs in raw_samples}

    items = []
    for s in samples:
        raw = raw_sample_map.get(s.sample_id) if s.sample_id else None
        gt  = (raw.extra_metadata or {}).get("boundingBoxes", []) if raw else []
        items.append(_sample_out(s, gt_boxes=gt))

    return {
        "items":       items,
        "total":       total,
        "page":        page,
        "page_size":   page_size,
        "total_pages": math.ceil(total / page_size) if page_size > 0 else 1,
    }


def get_model_versions(impulse_id: str, db: Session) -> List[dict]:
    versions = (
        db.query(ModelVersion)
        .filter(ModelVersion.impulse_id == impulse_id)
        .order_by(ModelVersion.is_active.desc(), ModelVersion.created_at.asc())
        .all()
    )
    return [_version_out(v) for v in versions]


def classify_all(
    impulse_id: str,
    model_version_id: Optional[str],
    db: Session,
    conf_threshold: float = 0.25,
) -> dict:
    """
    Classify all test samples using the real trained model.

    Called directly by the sync endpoint (legacy) or internally by the
    Celery worker via _run_classify_all.
    """
    impulse = db.query(Impulse).filter(Impulse.id == impulse_id).first()
    if not impulse:
        raise HTTPException(404, f"Impulse '{impulse_id}' not found")

    samples = _ensure_test_samples_provisioned(impulse, db)
    db.commit()

    if not samples:
        any_samples = (
            db.query(Sample)
            .filter(Sample.project_id == impulse.project_id)
            .first()
        )
        if any_samples:
            detail = (
                "No test-split samples found for this project. "
                "Go to Data acquisition and set at least one sample's split to 'Testing'."
            )
        else:
            detail = (
                "No samples found for this project. "
                "Upload samples via Data acquisition first."
            )
        raise HTTPException(422, detail)

    trained_model, output_type = _load_trained_model(impulse_id, model_version_id, db)

    logger.info(
        f"[classify_all] impulse={impulse_id} model={trained_model.id} "
        f"output_type={output_type} samples={len(samples)}"
    )

    n_reset = _reset_stale_scores(impulse_id, trained_model.id, db)
    if n_reset > 0:
        logger.info(f"[classify_all] Reset {n_reset} sample(s) to pending.")
        db.commit()
        from app.models.model_testing import ModelTestSample as _MTS
        samples = (
            db.query(_MTS)
            .filter(_MTS.impulse_id == impulse_id)
            .order_by(_MTS.created_at.asc())
            .all()
        )

    try:
        sim_results, label_names, det_image_results = _run_classify_all(
            impulse=impulse,
            test_samples=samples,
            trained_model=trained_model,
            output_type=output_type,
            db=db,
            conf_threshold=conf_threshold,
        )
    except Exception as e:
        logger.error(f"[classify_all] inference loop failed: {e}", exc_info=True)
        raise HTTPException(500, f"Model inference failed: {e}")

    metric_dicts = _compute_aggregate_metrics(
        results=sim_results,
        label_names=label_names,
        output_type=output_type,
        det_image_results=det_image_results if det_image_results else None,
    )

    now = datetime.utcnow()
    passed    = 0
    uncertain = 0
    fail_cnt  = 0
    for sample, f1, status, pred_cls, iou_val, pred_boxes in sim_results:
        sample.f1_score                   = f1
        sample.result_status              = status
        sample.predicted_class            = pred_cls
        sample.iou_score                  = iou_val
        sample.predicted_boxes            = pred_boxes
        sample.scored_by_trained_model_id = trained_model.id
        sample.updated_at                 = now
        if status == TestResultStatus.pass_:
            passed += 1
        elif status == TestResultStatus.uncertain:
            uncertain += 1
        else:
            fail_cnt += 1

    total    = len(sim_results)
    # failed_samples stores fail+uncertain (both are "not pass") for DB backward-compat.
    failed   = fail_cnt + uncertain
    accuracy = round((passed / total) * 100, 2) if total > 0 else 0.0

    version = _get_active_version(impulse_id, model_version_id, db)

    run = ModelTestRun(
        project_id       = impulse.project_id,
        impulse_id       = impulse_id,
        model_version_id = version.id if version else None,
        accuracy         = accuracy,
        total_samples    = total,
        passed_samples   = passed,
        failed_samples   = failed,
        created_at       = now,
    )
    db.add(run)
    db.flush()

    for md in _iter_scalar_metric_dicts(metric_dicts):
        db.add(MetricResult(
            test_run_id         = run.id,
            metric_name         = md["metric_name"],
            metric_display_name = md["metric_display_name"],
            metric_value        = md["metric_value"],
            created_at          = now,
        ))

    for sample, *_ in sim_results:
        sample.test_run_id = run.id

    db.commit()
    db.refresh(run)

    db_metrics = db.query(MetricResult).filter(MetricResult.test_run_id == run.id).all()

    project = db.query(Project).filter(Project.id == impulse.project_id).first()
    active_version = _get_active_version(impulse_id, version.id if version else None, db)

    return {
        "run":     _run_out(run),
        "metrics": [_metric_out(m) for m in db_metrics],
        "samples": [_sample_out(s) for s, *_ in sim_results],
        "page_data": {
            "project": {
                "id":   project.id   if project else None,
                "name": project.name if project else None,
            },
            "impulse":               _impulse_out(impulse),
            "available_impulses":    [],
            "selected_model_version": _version_out(active_version) if active_version else None,
            "latest_run":             _run_out(run),
            "accuracy":               accuracy,
            "metrics":                [_metric_out(m) for m in db_metrics],
            "summary": {
                "total":     total,
                "passed":    passed,
                "failed":    fail_cnt,
                "uncertain": uncertain,
                "pending":   0,
            },
            "target_device": "Arduino UNO Q (Cortex-M0+)",
        },
    }


def update_test_sample(
    sample_id: str,
    expected_outcome: Optional[str],
    result_status: Optional[str],
    db: Session,
) -> dict:
    sample = db.query(ModelTestSample).filter(ModelTestSample.id == sample_id).first()
    if not sample:
        raise HTTPException(404, f"Test sample '{sample_id}' not found")
    if expected_outcome is not None:
        sample.expected_outcome = expected_outcome
    if result_status is not None:
        try:
            sample.result_status = TestResultStatus(result_status)
        except ValueError:
            raise HTTPException(
                422,
                f"Invalid result_status '{result_status}'. Allowed: pending, pass, fail, uncertain",
            )
    sample.updated_at = datetime.utcnow()
    db.commit()
    db.refresh(sample)
    return _sample_out(sample)


def reclassify_single_sample(
    sample_id: str,
    model_version_id: Optional[str],
    conf_threshold: float,
    db: Session,
) -> dict:
    """
    Run inference on one ModelTestSample and persist the result.
    Reuses DSP cache when available; falls back to S3 download + DSP.
    """
    sample = db.query(ModelTestSample).filter(ModelTestSample.id == sample_id).first()
    if not sample:
        raise HTTPException(404, f"Test sample '{sample_id}' not found")

    impulse = db.query(Impulse).filter(Impulse.id == sample.impulse_id).first()
    if not impulse:
        raise HTTPException(404, "Impulse not found")

    trained_model, output_type = _load_trained_model(sample.impulse_id, model_version_id, db)

    # Gate unsupported detection types (e.g. SSD) — same chokepoint as the
    # batch path; avoids the (N, 6) tensor crashing _score_classification.
    if output_type in MODEL_TESTING_UNSUPPORTED_OUTPUT_TYPES:
        raise HTTPException(422, _UNSUPPORTED_TESTING_MSG.format(ot=output_type))

    real_sample = None
    if sample.sample_id:
        real_sample = db.query(Sample).filter(Sample.id == sample.sample_id).first()
    if real_sample is None:
        raise HTTPException(422, "No underlying raw sample found for this test sample")

    # Try DSP cache first
    from app.workers.dsp_worker import make_features_storage_key, compute_dsp_config_hash
    img: Optional[np.ndarray] = None
    try:
        import io as _io
        _cache_key = make_features_storage_key(impulse.project_id, impulse.id)
        _cache_bytes = storage.download_bytes(_cache_key)
        with np.load(_io.BytesIO(_cache_bytes), allow_pickle=True) as _npz:
            if "meta_json" in _npz and "ids" in _npz and "X" in _npz:
                import json as _json
                _meta = _json.loads(str(_npz["meta_json"][0]))
                _expected_hash = compute_dsp_config_hash(
                    impulse.dsp_blocks or [{"type": "raw", "params": {}}]
                )
                if (_meta.get("impulse_id") == impulse.id
                        and _meta.get("dsp_config_hash") == _expected_hash):
                    _id_list = [str(s) for s in _npz["ids"].tolist()]
                    if str(sample.sample_id) in _id_list:
                        idx      = _id_list.index(str(sample.sample_id))
                        _flat    = np.asarray(_npz["X"][idx], dtype=np.float32)
                        _img_w   = int(getattr(impulse, "image_width",  None) or 96)
                        _img_h   = int(getattr(impulse, "image_height", None) or 96)
                        _grey    = bool((impulse.dsp_blocks or [{}])[0].get("params", {}).get("grayscale", False))
                        _img_c   = 1 if _grey else 3
                        try:
                            img = _flat.reshape((_img_h, _img_w, _img_c))
                        except ValueError:
                            img = None  # shape mismatch → fall back to _preprocess_sample
    except Exception:
        pass

    if img is None:
        img = _preprocess_sample(real_sample, impulse)
    if img is None:
        raise HTTPException(422, "DSP preprocessing failed for this sample")

    label_names = _resolve_label_names(impulse, trained_model, db)
    is_detection = output_type in ("yolo_pro_detection", "detection", "ssd_detection")
    is_ssd_single = output_type == "ssd_detection"
    is_fomo_single = output_type == "detection_heatmap"

    # SSD: score at the training-eval confidence (0.01), never 0.25 — see
    # _resolve_ssd_threshold (mirrors the batch path).
    if is_ssd_single:
        _ssd_cr = None
        if trained_model.training_job_id:
            _tj_ssd = db.query(TrainingJob).filter(
                TrainingJob.id == trained_model.training_job_id
            ).first()
            if _tj_ssd and isinstance(_tj_ssd.classification_report, dict):
                _ssd_cr = _tj_ssd.classification_report
        conf_threshold, _ssd_src = _resolve_ssd_threshold(trained_model, _ssd_cr)
        logger.info(
            "[model_testing single] SSD conf_threshold=%.4f (source: %s).",
            conf_threshold, _ssd_src,
        )
    # YOLO-Pro (and plain detection): keep the RUNTIME threshold from
    # model_metadata here. This path re-scores a single sample for the
    # inspector — it computes no aggregate mAP — so the permissive eval-parity
    # conf the batch path uses (see _resolve_yolo_pro_threshold) would only
    # flood the overlay with sub-1% boxes. The runtime gate matches what the
    # batch path shows per sample (_yp_display_conf) and what deployment emits.
    elif is_detection:
        _meta_thr = (trained_model.model_metadata or {}).get("threshold")
        if _meta_thr is not None:
            conf_threshold = float(_meta_thr)

    # Load FOMO threshold from training_job.classification_report (mirrors batch path).
    _fomo_conf_threshold: float  = conf_threshold
    _fomo_threshold_source: str  = "caller default"
    _fomo_single_training_cr: Optional[dict] = None

    if is_fomo_single and trained_model.training_job_id:
        _tj = db.query(TrainingJob).filter(
            TrainingJob.id == trained_model.training_job_id
        ).first()
        if _tj and isinstance(_tj.classification_report, dict):
            _fomo_single_training_cr = _tj.classification_report
        else:
            logger.warning(
                "[FOMO eval single] training_job=%s has no classification_report; "
                "will use caller-supplied hint=%.4f.",
                trained_model.training_job_id, _fomo_conf_threshold,
            )

    if is_fomo_single and _fomo_single_training_cr is not None:
        try:
            _fomo_conf_threshold, _fomo_threshold_source = resolve_fomo_threshold(
                mode=ThresholdSource.TRAINING_PARITY,
                training_classification_report=_fomo_single_training_cr,
            )
        except ValueError as _te:
            logger.error(
                "[FOMO eval single] resolve_fomo_threshold failed: %s. "
                "Proceeding with caller-supplied hint=%.4f.",
                _te, _fomo_conf_threshold,
            )

    model_handle = _load_model_into_memory(trained_model)
    image_batch = np.stack([img], axis=0).astype(np.float32)
    if is_ssd_single:
        # SSD decoded TFLite: uint8 input + in-graph norm — feed raw [0,255].
        image_batch = _scale_to_uint8_pixel_range(model_handle, image_batch)
    raw_outputs = _run_inference(model_handle, image_batch)

    if is_detection:
        from app.workers.sample_utils import normalize_bounding_boxes
        from app.ml.yolo_pro_worker import _nms
        label_to_idx = _resolve_label_to_idx(impulse, label_names, db)

        sample_out = raw_outputs[0] if isinstance(raw_outputs, list) else raw_outputs[0]
        raw_dets = _decoded_rows_to_dets(sample_out, conf_threshold)
        nms_dets = _nms(raw_dets, iou_threshold=0.45, max_dets=100, class_agnostic=True)
        # SSD: adaptive per-image gate (mirrors the batch path); YOLO-Pro/plain
        # detection keep the full NMS set.
        scored_dets = _ssd_per_sample_dets(nms_dets) if is_ssd_single else nms_dets
        raw_gt_boxes = (real_sample.extra_metadata or {}).get("boundingBoxes", [])
        raw_bytes = None
        if real_sample.storage_key:
            try:
                raw_bytes = storage.download_bytes(real_sample.storage_key)
            except Exception:
                raw_bytes = None
        norm_gt = normalize_bounding_boxes(raw_gt_boxes, image_bytes=raw_bytes)
        score, status, pred_cls, iou_val, det_stats = _score_detection_from_dets(
            scored_dets, label_names, gt_boxes=norm_gt or None,
            iou_threshold=IOU_THRESHOLD_MATCH,
            label_to_idx=label_to_idx,
        )
        # GT grouped by class for the SSD IoU-display score (mirrors batch path).
        gt_by_class: dict = {}
        for box in norm_gt:
            key = box.get("label_id") or box.get("label") or ""
            cls_idx = label_to_idx.get(key.strip())
            if cls_idx is None:
                continue
            x1 = float(box.get("x", 0)); y1 = float(box.get("y", 0))
            x2 = x1 + float(box.get("w", 0)); y2 = y1 + float(box.get("h", 0))
            if x2 > x1 and y2 > y1:
                gt_by_class.setdefault(cls_idx, []).append([x1, y1, x2, y2])
        top_boxes = {
            "v": 2,
            "boxes": [
                {"x": d["box"][0], "y": d["box"][1],
                 "x2": d["box"][2], "y2": d["box"][3],
                 # SSD overlay shows IoU match quality; others show real confidence.
                 "score": round(
                     _ssd_box_display_score(d["box"], d["class_idx"], gt_by_class)
                     if is_ssd_single else d["score"],
                     4,
                 ),
                 "label": label_names[d["class_idx"]] if d["class_idx"] < len(label_names) else str(d["class_idx"])}
                for d in scored_dets[:20]
            ],
            **det_stats,
        }
    elif is_fomo_single:
        from app.workers.sample_utils import normalize_bounding_boxes
        label_to_idx = _resolve_label_to_idx(impulse, label_names, db)
        sample_out = raw_outputs[0] if isinstance(raw_outputs, (list, np.ndarray)) else raw_outputs
        fomo_out = np.array(sample_out, dtype=np.float32)
        raw_gt_boxes = (real_sample.extra_metadata or {}).get("boundingBoxes", [])
        raw_bytes = None
        if real_sample.storage_key:
            try:
                raw_bytes = storage.download_bytes(real_sample.storage_key)
            except Exception:
                raw_bytes = None
        norm_gt = normalize_bounding_boxes(raw_gt_boxes, image_bytes=raw_bytes)
        if fomo_out.ndim != 3:
            score, status, pred_cls, iou_val, top_boxes = (
                0.0, TestResultStatus.fail, None, None, None
            )
        else:
            fH, fW, _ = fomo_out.shape
            n_classes = len(label_names)
            pred_cells, probs = _fomo_decode_heatmap(fomo_out, _fomo_conf_threshold)
            gt_cells, _n_unres_single = _fomo_gt_cells_from_boxes_raw(
                fomo_norm_gt, fH, fW, label_to_idx,
                sample_id=str(getattr(real_sample, "id", "unknown")),
            )
            score, status, pred_cls, iou_val, _tp, _fp, _fn = _fomo_per_sample_score(
                pred_cells, gt_cells, n_classes, label_names,
            )
            _fomo_log_diagnostics(
                threshold=_fomo_conf_threshold,
                threshold_source=_fomo_threshold_source,
                total_gt_cells=len(gt_cells),
                total_pred_cells=len(pred_cells),
                tp=_tp, fp=_fp, fn=_fn,
                label_names=label_names,
                n_unresolved_gt=_n_unres_single,
                context="single_sample",
    )
            if len(pred_cells) == 0 and len(gt_cells) > 0:
                logger.warning(
                    "[FOMO eval] single-sample path produced 0 detections at threshold=%s",
                    _fomo_conf_threshold,
                )
            top_boxes = [
                {
                    "x": c / fW, "y": r / fH,
                    "x2": (c + 1) / fW, "y2": (r + 1) / fH,
                    "score": round(float(probs[r, c, cls + 1]), 4),
                    "label": label_names[cls] if cls < n_classes else str(cls),
                }
                for (r, c), cls in list(pred_cells.items())[:20]
            ]
    else:
        sample_out = raw_outputs[0] if hasattr(raw_outputs, '__len__') else raw_outputs
        score, status, pred_cls, iou_val = _score_classification(
            sample_out, sample.expected_outcome or "-", label_names,
        )
        top_boxes = None

    now = datetime.utcnow()
    sample.f1_score                   = score
    sample.result_status              = status
    sample.predicted_class            = pred_cls
    sample.iou_score                  = iou_val
    sample.predicted_boxes            = top_boxes
    sample.scored_by_trained_model_id = trained_model.id
    sample.updated_at                 = now
    db.commit()
    db.refresh(sample)
    gt_raw = (real_sample.extra_metadata or {}).get("boundingBoxes", []) if real_sample else []
    return _sample_out(sample, gt_boxes=gt_raw)


def get_metrics(impulse_id: str, db: Session) -> dict:
    run = _latest_test_run(impulse_id, db)
    if not run:
        return {
            "run":     None,
            "metrics": [],
            "message": "No test run found. Run 'Classify all' first.",
        }
    return {
        "run":     _run_out(run),
        "metrics": [_metric_out(m) for m in run.metrics],
    }


def list_runs(impulse_id: str, limit: int, db: Session) -> dict:
    """
    Return up to `limit` completed runs for the impulse, newest first.
    Each run includes its metrics so callers can compare across model versions.
    """
    runs = (
        db.query(ModelTestRun)
        .filter(ModelTestRun.impulse_id == impulse_id)
        .order_by(ModelTestRun.created_at.desc())
        .limit(limit)
        .all()
    )
    return {
        "runs": [
            {**_run_out(r), "metrics": [_metric_out(m) for m in r.metrics]}
            for r in runs
        ]
    }


# ─── Stale-score reset ────────────────────────────────────────────────────────

def _reset_stale_scores(
    impulse_id: str,
    active_trained_model_id: str,
    db: Session,
) -> int:
    """
    Reset every ModelTestSample for this impulse to pending if it was scored
    by a different TrainedModel than the one currently being used.
    Returns the number of samples that were reset.
    """
    stale = (
        db.query(ModelTestSample)
        .filter(
            ModelTestSample.impulse_id == impulse_id,
            ModelTestSample.result_status != TestResultStatus.pending,
            ModelTestSample.scored_by_trained_model_id.isnot(None),
            ModelTestSample.scored_by_trained_model_id != active_trained_model_id,
        )
        .all()
    )

    now = datetime.utcnow()
    for s in stale:
        s.result_status              = TestResultStatus.pending
        s.f1_score                   = None
        s.predicted_class            = None
        s.iou_score                  = None
        s.predicted_boxes            = None
        s.test_run_id                = None
        s.scored_by_trained_model_id = None
        s.updated_at                 = now

    if stale:
        db.flush()

    return len(stale)
