"""
Post-processing video worker — Phase 6.

Celery task that:
  1. Marks the ProcessingJob as 'processing'.
  2. Loads the project's post-processing settings (or falls back to defaults).
  3. Loads the project's best available trained object-detection model
     (YOLO-Pro, SSD, or FOMO/detection_heatmap).
  4. Builds a per-frame inference callable from the loaded model — uses the
     box-detection path for YOLO-Pro / SSD and the FOMO heatmap-decode path
     for detection_heatmap models. FOMO decoding reuses
     `app.ml.fomo_evaluator.decode_fomo_heatmap` so the runtime contract is
     identical to the rest of the app (model_testing_service, inference
     endpoint, training-time evaluator).
  5. Runs VideoProcessor.process() with the inference callable.
  6. Marks the job 'complete' with the output storage key and timing.
  7. On any exception marks it 'failed' with a truncated error message.

  Fallback path: if no detection model is found but pre-computed per-frame
  detections were stored at the companion S3 key (debug endpoint), those are
  used instead.  If neither exists the job is marked failed with a clear
  message telling the user to train a detection model first.
"""
from __future__ import annotations

import json
import logging
import time
from typing import Callable, List, Optional, Tuple

from app.workers.celery_app import celery_app
from app.core.database import SessionLocal
from app.core.storage import storage, StorageService
from app.models.user import PostProcessingSettings, ProcessingJob
from app.services import post_processing_pipeline as pp_pipeline_svc
from app.services.post_processing import PipelineConfig
from app.services.post_processing.types import Detection
from app.services.post_processing.video_processor import ProcessingCancelledError, VideoProcessor

logger = logging.getLogger(__name__)

_MAX_ERROR_LEN = 1000   # truncate very long tracebacks stored in DB

# Cancellation is polled at most once per this interval — a fresh DB session
# on every single video frame would be wasteful. See _make_cancel_check.
_CANCEL_CHECK_INTERVAL_SECONDS = 1.0


def _make_cancel_check(job_id: str) -> Callable[[], bool]:
    """Build a throttled, cooperative cancel-check for VideoProcessor.process().

    The post_processing queue runs on --pool=threads on Windows (see cmd.txt),
    where Celery's revoke(terminate=True) cannot reliably kill an in-flight
    task — the worker process IS the thread. So the frame loop polls this DB
    flag itself and stops voluntarily, same pattern as
    app.workers.cancel_utils for training jobs (kept separate here since it
    targets TrainingJob, not ProcessingJob).

    Once cancelled=True is observed the check short-circuits without hitting
    the DB again — cancellation is terminal for a given job run.
    """
    state = {"last_check": 0.0, "cancelled": False}

    def check() -> bool:
        if state["cancelled"]:
            return True
        now = time.monotonic()
        if now - state["last_check"] < _CANCEL_CHECK_INTERVAL_SECONDS:
            return False
        state["last_check"] = now
        fresh_db = SessionLocal()
        try:
            job = fresh_db.query(ProcessingJob).filter(ProcessingJob.id == job_id).first()
            if job is not None and job.status == "cancelled":
                state["cancelled"] = True
        finally:
            fresh_db.close()
        return state["cancelled"]

    return check


# ─── Detection deserialisation (debug/fallback path) ─────────────────────────

def _parse_detections_per_frame(raw: list) -> List[List[Detection]]:
    """
    Convert a JSON-decoded list-of-lists into typed Detection objects.
    Each inner list is one frame.  Unknown keys are silently ignored so the
    worker stays forward-compatible with richer payloads.
    """
    result: List[List[Detection]] = []
    for frame_raw in raw:
        frame: List[Detection] = []
        for d in frame_raw:
            try:
                frame.append(Detection(
                    x1=float(d["x1"]),
                    y1=float(d["y1"]),
                    x2=float(d["x2"]),
                    y2=float(d["y2"]),
                    class_name=str(d["class_name"]),
                    confidence=float(d["confidence"]),
                ))
            except (KeyError, TypeError, ValueError) as exc:
                logger.warning("Skipping malformed detection dict %r: %s", d, exc)
        result.append(frame)
    return result


# ─── Model loading (Phase 6) ──────────────────────────────────────────────────

# Output types this worker can render. Includes FOMO (`detection_heatmap`) —
# decoded into per-cell boxes by `_make_fomo_inference_fn` before being fed
# into the same post-processing pipeline as YOLO-Pro / SSD.
_SUPPORTED_DETECTION_OUTPUT_TYPES: Tuple[str, ...] = (
    "yolo_pro_detection",
    "ssd_detection",
    "object_detection",
    "detection",
    "detection_heatmap",
)


def _metadata_indicates_fomo(meta: dict) -> bool:
    """Return True when this trained-model metadata describes a FOMO model.

    Mirrors `app.api.v1.endpoints.inference._is_fomo_model` so the worker uses
    the same identification rule as the inference endpoint and model-testing
    service. Single source of intent for FOMO detection.
    """
    if not isinstance(meta, dict):
        return False
    if meta.get("model_type") == "detection_heatmap":
        return True
    if meta.get("output_type") == "detection_heatmap":
        return True
    arch = str(meta.get("architecture") or "").lower()
    return "fomo" in arch


def _load_best_detection_model_for_project(
    project_id: str,
    db,
    impulse_id: Optional[str] = None,
) -> Tuple:
    """
    Find the best available trained object-detection model for the project.

    Accepts YOLO-Pro, SSD, and FOMO (detection_heatmap) trained models.

    If impulse_id is provided the search is restricted to that single impulse
    (the one the user is currently viewing on the post-processing page).
    Otherwise every impulse in the project is searched, which is only correct
    for single-impulse projects.

    Priority within candidates: decoded_float32 tflite (0) > float32 tflite (1)
    > keras (2). FOMO models are never exported with `decoded_float32` (that
    variant is YOLO-specific) so they naturally fall into priorities 1–2.

    Returns (trained_model, label_names, input_shape_hwc).
    Raises ValueError with a user-facing message if no detection model exists.
    """
    from app.models.user import Impulse, JobStatus, Label, TrainedModel, TrainingJob

    if impulse_id:
        # Ownership check: impulse must belong to the project
        impulse = (
            db.query(Impulse)
            .filter(Impulse.id == impulse_id, Impulse.project_id == project_id)
            .first()
        )
        if not impulse:
            raise ValueError(
                f"Impulse {impulse_id!r} not found in project {project_id!r}."
            )
        impulses = [impulse]
    else:
        impulses = db.query(Impulse).filter(Impulse.project_id == project_id).all()
        if not impulses:
            raise ValueError(
                "No impulse found for this project. "
                "Create and train an object-detection impulse first."
            )

    best: Optional[TrainedModel] = None
    best_priority = 99

    for impulse in impulses:
        latest_job = (
            db.query(TrainingJob)
            .filter(
                TrainingJob.impulse_id == impulse.id,
                TrainingJob.status == JobStatus.completed,
            )
            .order_by(TrainingJob.created_at.desc())
            .first()
        )
        if not latest_job:
            continue

        candidates = (
            db.query(TrainedModel)
            .filter(TrainedModel.training_job_id == latest_job.id)
            .all()
        )
        for tm in candidates:
            meta = tm.model_metadata or {}
            output_type = meta.get("output_type")
            # Accept by explicit output_type tag OR by architecture string —
            # some early FOMO records were saved without `output_type` so we
            # fall back to architecture-name sniffing (same rule as the
            # inference endpoint).
            if output_type not in _SUPPORTED_DETECTION_OUTPUT_TYPES and not _metadata_indicates_fomo(meta):
                continue
            variant = meta.get("variant", "")
            fmt = tm.format or ""
            if fmt == "tflite" and variant == "decoded_float32":
                priority = 0
            elif fmt == "tflite" and variant == "float32":
                priority = 1
            elif fmt == "keras":
                priority = 2
            else:
                priority = 3

            if best is None or priority < best_priority:
                best = tm
                best_priority = priority

    if best is None:
        raise ValueError(_build_missing_detection_model_message(project_id, db, impulse_id=impulse_id))

    # Resolve ordered label names
    meta = best.model_metadata or {}
    if meta.get("label_names"):
        label_names: List[str] = list(meta["label_names"])
    else:
        from app.models.user import Label
        labels = (
            db.query(Label)
            .filter(Label.project_id == project_id)
            .order_by(Label.name.asc())
            .all()
        )
        label_names = [lbl.name for lbl in labels]

    # Resolve model input shape (H, W, C)
    raw_shape = meta.get("input_shape")
    input_shape_hwc: Tuple[int, int, int] = (96, 96, 3)
    if raw_shape:
        dims = [int(d) for d in raw_shape if isinstance(d, (int, float)) and int(d) > 0]
        if len(dims) >= 3:
            input_shape_hwc = (dims[-3], dims[-2], dims[-1])
        elif len(dims) == 2:
            input_shape_hwc = (dims[0], dims[1], 3)

    return best, label_names, input_shape_hwc


def _latest_trained_model_hint(
    project_id: str,
    db,
    impulse_id: Optional[str] = None,
) -> Optional[str]:
    """
    Return a short hint about the latest completed model for this scope.

    Used only to improve user-facing failure messages when no compatible
    box-detection model is available for video post-processing.
    """
    from app.models.user import Impulse, JobStatus, TrainedModel, TrainingJob

    if impulse_id:
        impulses = (
            db.query(Impulse)
            .filter(Impulse.id == impulse_id, Impulse.project_id == project_id)
            .all()
        )
    else:
        impulses = db.query(Impulse).filter(Impulse.project_id == project_id).all()

    latest_job = None
    for impulse in impulses:
        job = (
            db.query(TrainingJob)
            .filter(
                TrainingJob.impulse_id == impulse.id,
                TrainingJob.status == JobStatus.completed,
            )
            .order_by(TrainingJob.created_at.desc())
            .first()
        )
        if job is not None and (latest_job is None or job.created_at > latest_job.created_at):
            latest_job = job

    if latest_job is None:
        return None

    models = db.query(TrainedModel).filter(TrainedModel.training_job_id == latest_job.id).all()
    if not models:
        return None

    preferred = (
        next(
            (
                m for m in models
                if m.format == "tflite" and (m.model_metadata or {}).get("variant") == "decoded_float32"
            ),
            None,
        )
        or next((m for m in models if m.format == "tflite"), None)
        or next((m for m in models if m.format == "keras"), None)
        or models[0]
    )
    meta = preferred.model_metadata or {}
    output_type = str(meta.get("output_type", "")).lower()
    architecture = str(meta.get("architecture", "")).lower()

    if output_type == "detection_heatmap" or "fomo" in architecture:
        return "fomo"
    return None


def _build_missing_detection_model_message(
    project_id: str,
    db,
    impulse_id: Optional[str] = None,
) -> str:
    """Build the user-facing 'no detection model' message.

    FOMO is now rendered through `_make_fomo_inference_fn`, so reaching this
    message means no YOLO-Pro / SSD / FOMO model is available at all — the
    user must train one of them first. The function still consults
    `_latest_trained_model_hint` to keep the call surface intact for callers
    that inspect intermediate state (e.g. tests).
    """
    _ = _latest_trained_model_hint(project_id, db, impulse_id=impulse_id)
    return (
        "No trained object-detection model found for this project. "
        "Train a YOLO-Pro, SSD, or FOMO detection model first, then retry."
    )


# ─── Per-frame inference callable (Phase 6) ──────────────────────────────────

# Pre-pipeline confidence floor. Both the YOLO/SSD and FOMO branches pre-filter
# detections at this very-low cutoff so the pipeline's user-controlled
# threshold slider (PipelineConfig.threshold) remains authoritative. Anything
# below this is dropped purely as a performance optimisation.
_PIPELINE_CONF_FLOOR: float = 0.05


def _preprocess_frame_for_model(
    frame_bgr,
    input_shape_hwc: Tuple[int, int, int],
    model_metadata: Optional[dict] = None,
):
    """Build the float32 input batch the model expects, driven by metadata.

    Mirrors the contract that
    `app.api.v1.endpoints.inference._apply_manifest_input_contract` enforces
    for the live inference endpoint, so the worker feeds models exactly what
    they were trained on:

      * RGB → BGR swap when `channel_order == "bgr"`.
      * `(x - input_mean) / input_std` normalization when
        `normalize_input` is true.
      * Grayscale (`img_c == 1`) takes the luminance branch and skips the
        channel-order swap — there's no meaningful "RGB vs BGR" for one
        channel.

    Defaults (used when a field is missing from metadata) match the
    inference-endpoint defaults for legacy YOLO/SSD models:
      channel_order = "rgb", normalize_input = True,
      input_mean = 0.0, input_std = 255.0  →  [0, 1] float input.

    For FOMO models the training worker writes
      normalize_input=True, input_mean=127.5, input_std=127.5
    so this helper produces the [-1, 1] input the FOMO TFLite export expects.
    Feeding /255 (the previous hardcoded behaviour) silently produced empty
    or wildly incorrect detections.

    Returns the (1, H, W, C) float32 batch ready to hand to `_run_inference`.
    """
    import cv2
    import numpy as np

    meta = model_metadata or {}
    img_h, img_w, img_c = input_shape_hwc

    # 1. Resize → raw 0..255 RGB (or grayscale) pixel tensor.
    if img_c == 1:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (img_w, img_h))
        x = np.expand_dims(resized, axis=-1).astype(np.float32)
    else:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (img_w, img_h))
        x = resized.astype(np.float32)

    # 2. Apply the manifest input contract — channel order + normalization.
    channel_order = str(meta.get("channel_order", "rgb")).lower()
    normalize = bool(meta.get("normalize_input", True))
    input_mean = float(meta.get("input_mean", 0.0))
    input_std = float(meta.get("input_std", 255.0))

    # Channel swap only when the model genuinely wants BGR and we have 3 ch.
    if channel_order == "bgr" and x.ndim == 3 and x.shape[-1] == 3:
        x = x[..., ::-1].copy()

    if normalize:
        denom = input_std if input_std != 0.0 else 255.0
        x = (x - input_mean) / denom

    return x[np.newaxis]  # (1, H, W, C)


def _make_inference_fn(
    model_handle,
    label_names: List[str],
    input_shape_hwc: Tuple[int, int, int],
    model_metadata: Optional[dict] = None,
    ssd_conf_threshold: Optional[float] = None,
) -> Callable:
    """Build a per-frame inference callable for VideoProcessor.

    Dispatches on the trained-model metadata:
      * FOMO (detection_heatmap) → `decode_fomo_heatmap` heatmap path.
      * MobileNetV2 SSD (`ssd_detection`) → the shared evaluation/live SSD
        scoring path (conf 0.01 + adaptive per-image gate); see
        `_make_ssd_inference_fn`. `ssd_conf_threshold` is resolved at the call
        site (where the TrainedModel + DB are available) via
        `_resolve_ssd_threshold`.
      * Everything else (YOLO-Pro / generic box) → the generic decoded-tensor
        box path, unchanged.

    The resulting callable has the same contract for all branches: takes a BGR
    numpy frame and returns `List[Detection]` with coords normalised to [0, 1].

    When `model_metadata` is None we default to the box-detection path —
    preserves backward compatibility for callers / tests that still pass three
    positional args.
    """
    if model_metadata and _metadata_indicates_fomo(model_metadata):
        return _make_fomo_inference_fn(
            model_handle, label_names, input_shape_hwc, model_metadata,
        )
    if model_metadata and model_metadata.get("output_type") == "ssd_detection":
        return _make_ssd_inference_fn(
            model_handle, label_names, input_shape_hwc, model_metadata,
            ssd_conf_threshold=ssd_conf_threshold,
        )
    return _make_box_inference_fn(
        model_handle, label_names, input_shape_hwc, model_metadata=model_metadata,
    )


def _make_box_inference_fn(
    model_handle,
    label_names: List[str],
    input_shape_hwc: Tuple[int, int, int],
    model_metadata: Optional[dict] = None,
) -> Callable:
    """
    Per-frame inference for box-detection models (YOLO-Pro / SSD).

    The returned function takes a BGR numpy frame and returns a list of
    Detection objects (un-thresholded — the post-processing pipeline applies
    the configured threshold downstream).

    Pre-NMS confidence pre-filter is set very low (_PIPELINE_CONF_FLOOR) so the
    pipeline threshold slider controls what the user sees, not a hardcoded cutoff.

    Input normalization is metadata-driven via `_preprocess_frame_for_model`
    so models trained with non-default mean/std (e.g. [-1, 1]) get the right
    input contract, not a hardcoded /255.
    """
    import numpy as np
    from app.ml.yolo_pro_worker import _nms
    from app.services.model_testing_service import _run_inference

    meta = model_metadata or {}

    def infer(frame_bgr: np.ndarray) -> List[Detection]:
        import numpy as _np

        batch = _preprocess_frame_for_model(frame_bgr, input_shape_hwc, meta)

        try:
            raw = _run_inference(model_handle, batch)
        except Exception as exc:
            logger.warning("Per-frame inference failed: %s", exc)
            return []

        # Extract (N, 6) detection tensor: [x1, y1, x2, y2, score, class_id]
        out: Optional[np.ndarray] = None
        if isinstance(raw, _np.ndarray):
            out = raw[0] if raw.ndim == 3 else raw
        elif isinstance(raw, list) and raw:
            first = raw[0]
            if isinstance(first, _np.ndarray):
                out = first[0] if first.ndim == 3 else first

        if out is None or not (isinstance(out, _np.ndarray) and out.ndim == 2 and out.shape[1] == 6):
            return []

        # Broad pre-filter — intentionally low; the pipeline threshold filters further
        raw_dets = [
            {
                "class_idx": int(row[5]),
                "score":     float(row[4]),
                "box":       [float(row[0]), float(row[1]), float(row[2]), float(row[3])],
            }
            for row in out
            if float(row[4]) >= _PIPELINE_CONF_FLOOR
        ]
        nms_dets = _nms(raw_dets, iou_threshold=0.45, max_dets=100, class_agnostic=True)

        return [
            Detection(
                x1=d["box"][0], y1=d["box"][1],
                x2=d["box"][2], y2=d["box"][3],
                class_name=(
                    label_names[d["class_idx"]]
                    if d["class_idx"] < len(label_names)
                    else str(d["class_idx"])
                ),
                confidence=d["score"],
            )
            for d in nms_dets
        ]

    return infer


def _make_ssd_inference_fn(
    model_handle,
    label_names: List[str],
    input_shape_hwc: Tuple[int, int, int],
    model_metadata: Optional[dict] = None,
    ssd_conf_threshold: Optional[float] = None,
) -> Callable:
    """
    Per-frame inference for MobileNetV2 SSD FPN-Lite.

    SSD logits are tiny — top scores cluster ~0.008-0.02 — so the generic box
    path's fixed 0.05 floor (and the pipeline's 0.5 absolute threshold) drop
    every detection. This branch reuses the *exact* evaluation/live-classification
    SSD scoring contract instead of forking it:

      * confidence pre-filter at `_resolve_ssd_threshold` (0.01 training parity,
        resolved at the call site) via `_decoded_rows_to_dets`,
      * class-agnostic NMS (`_nms`, iou=0.45),
      * adaptive per-image gate `_ssd_per_sample_dets` (keep within 0.95× the
        frame's top score, capped to the SSD top-K).

    Decoder, coordinate space, resize, and normalization are untouched — the
    decoded (1, N, 6) [x1,y1,x2,y2,score,class_id] tensor and the metadata-driven
    `_preprocess_frame_for_model` contract are shared with the generic box path.

    Display confidence: the post-processing pipeline applies a downstream
    *absolute* threshold (`PipelineConfig.threshold`, default 0.5) which SSD's
    raw ~0.01 scores can never clear. The evaluation overlay sidesteps this by
    showing an IoU-vs-GT score, but video frames have no ground truth — so we
    present a per-frame *normalized* score (score / frame-top), putting the
    gated cluster in [~0.95, 1.0]. This keeps the "Detection threshold" slider
    meaningful without silently re-dropping the boxes the gate already chose.
    """
    import numpy as np
    from app.ml.yolo_pro_worker import _nms
    from app.services.model_testing_service import (
        _run_inference,
        _decoded_rows_to_dets,
        _ssd_per_sample_dets,
        SSD_TRAINING_EVAL_CONF,
    )

    meta = model_metadata or {}
    conf = float(ssd_conf_threshold) if ssd_conf_threshold is not None else SSD_TRAINING_EVAL_CONF

    def infer(frame_bgr: np.ndarray) -> List[Detection]:
        import numpy as _np

        batch = _preprocess_frame_for_model(frame_bgr, input_shape_hwc, meta)

        try:
            raw = _run_inference(model_handle, batch)
        except Exception as exc:
            logger.warning("SSD per-frame inference failed: %s", exc)
            return []

        # Extract the decoded (N, 6) tensor: [x1, y1, x2, y2, score, class_id].
        out: Optional[np.ndarray] = None
        if isinstance(raw, _np.ndarray):
            out = raw[0] if raw.ndim == 3 else raw
        elif isinstance(raw, list) and raw:
            first = raw[0]
            if isinstance(first, _np.ndarray):
                out = first[0] if first.ndim == 3 else first

        if out is None or not (isinstance(out, _np.ndarray) and out.ndim == 2 and out.shape[1] == 6):
            return []

        # Shared SSD scoring contract: conf 0.01 → class-agnostic NMS → gate.
        raw_dets = _decoded_rows_to_dets(out, conf)
        nms_dets = _nms(raw_dets, iou_threshold=0.45, max_dets=100, class_agnostic=True)
        gated = _ssd_per_sample_dets(nms_dets)
        if not gated:
            return []

        top_score = max(d["score"] for d in gated) or 1.0
        return [
            Detection(
                x1=d["box"][0], y1=d["box"][1],
                x2=d["box"][2], y2=d["box"][3],
                class_name=(
                    label_names[d["class_idx"]]
                    if d["class_idx"] < len(label_names)
                    else str(d["class_idx"])
                ),
                confidence=float(min(1.0, d["score"] / top_score)) if top_score > 0 else float(d["score"]),
            )
            for d in gated
        ]

    return infer


def _make_fomo_inference_fn(
    model_handle,
    label_names: List[str],
    input_shape_hwc: Tuple[int, int, int],
    model_metadata: dict,
) -> Callable:
    """
    Per-frame inference for FOMO (detection_heatmap) models.

    Pipeline:
      1. Resize the BGR frame to the model's training input shape.
      2. Apply the metadata-driven input contract — channel order +
         (x - input_mean) / input_std — via
         `_preprocess_frame_for_model`. For FOMO the training worker writes
         normalize_input=True, input_mean=127.5, input_std=127.5, so the
         model receives the [-1, 1] tensor it was exported to consume.
         Hardcoding `/255` here used to silently produce empty / wrong
         detections because the FOMO TFLite has no built-in rescale layer.
      3. Run `_run_inference` to produce a raw heatmap tensor.
       The output is squeezed to (gH, gW, C+1) where channel 0 is background
       and channels 1..N map to label_names[0..N-1].
      4. Decode cells via `decode_fomo_heatmap` from `app.ml.fomo_evaluator`
       — the same function the inference endpoint and the model-testing
       service use. The decode threshold is intentionally set to
       `_PIPELINE_CONF_FLOOR`, NOT the model's stored threshold, so the
       pipeline's user-controlled threshold slider remains authoritative
       downstream (same convention as the YOLO box path).
      5. Convert each surviving cell to a normalised bbox centred on the
       cell (full-cell extent) and wrap in `Detection`.

    Cell-to-box geometry mirrors `inference.py::_postprocess_output`:
        cx, cy = (col + 0.5) / gw, (row + 0.5) / gh
        hw, hh = 0.5 / gw, 0.5 / gh
    Coordinates are normalised [0, 1]; VideoProcessor scales them up to pixel
    coords during rendering.
    """
    import numpy as np
    from app.ml.fomo_evaluator import decode_fomo_heatmap
    from app.services.model_testing_service import _run_inference

    # FOMO version + grid_size are passed through to decode_fomo_heatmap for
    # logging/validation only — the decode algorithm is the same for v1 and
    # v2. We default to v1 when metadata is silent (older saved models).
    fomo_version = int(model_metadata.get("fomo_version", 1) or 1)
    grid_size    = int(model_metadata.get("grid_size", 0) or 0)

    def infer(frame_bgr: np.ndarray) -> List[Detection]:
        import numpy as _np

        # 1–2. Preprocess: resize + metadata-driven channel/normalize.
        batch = _preprocess_frame_for_model(
            frame_bgr, input_shape_hwc, model_metadata,
        )

        # 3. Inference.
        try:
            raw = _run_inference(model_handle, batch)
        except Exception as exc:
            logger.warning("Per-frame FOMO inference failed: %s", exc)
            return []

        # Normalise the raw output to (gH, gW, C+1). _run_inference may return
        # an ndarray with leading batch dim, a list (multi-output), or already-
        # squeezed (gH, gW, C+1). Be tolerant of all three shapes.
        out: Optional[_np.ndarray] = None
        if isinstance(raw, _np.ndarray):
            out = raw
        elif isinstance(raw, list) and raw:
            first = raw[0]
            if isinstance(first, _np.ndarray):
                out = first
        if out is None:
            return []
        out = _np.asarray(out, dtype=_np.float32)
        if out.ndim == 4:
            out = out[0]
        if out.ndim != 3 or out.shape[-1] < 2:
            # Need (gH, gW, C+1) with C ≥ 1 (at least one object channel).
            logger.warning(
                "Unexpected FOMO output shape: %s; expected (gH, gW, C+1).",
                tuple(out.shape),
            )
            return []

        gh, gw, _ = out.shape

        # 4. Decode cells. Decode threshold = pipeline floor so the pipeline
        #    threshold filter (applied later) controls visible detections.
        try:
            pred_cells, probs = decode_fomo_heatmap(
                out,
                _PIPELINE_CONF_FLOOR,
                min_peak_gap=0.0,
                fomo_version=fomo_version,
                grid_size=grid_size,
            )
        except Exception as exc:
            logger.warning("decode_fomo_heatmap failed: %s", exc)
            return []

        # 5. Convert cells → normalised boxes → Detection.
        detections: List[Detection] = []
        for (row, col), cls_idx in pred_cells.items():
            # +1 skips background channel
            conf = float(probs[row, col, cls_idx + 1])
            cx, cy = (col + 0.5) / gw, (row + 0.5) / gh
            hw, hh = 0.5 / gw, 0.5 / gh
            x1 = max(0.0, cx - hw)
            y1 = max(0.0, cy - hh)
            x2 = min(1.0, cx + hw)
            y2 = min(1.0, cy + hh)
            name = (
                label_names[cls_idx]
                if 0 <= cls_idx < len(label_names)
                else str(cls_idx)
            )
            detections.append(
                Detection(
                    x1=x1, y1=y1, x2=x2, y2=y2,
                    class_name=name,
                    confidence=conf,
                )
            )
        return detections

    return infer


# ─── Task ─────────────────────────────────────────────────────────────────────

@celery_app.task(
    bind=True,
    name="app.workers.post_processing_worker.run_video_job",
    max_retries=0,          # no automatic retry — bad video should fail, not loop
)
def run_video_job(self, job_id: str, impulse_id: Optional[str] = None) -> None:
    db = SessionLocal()
    job: Optional[ProcessingJob] = None

    try:
        # ── Load job ──────────────────────────────────────────────────────────
        job = db.query(ProcessingJob).filter(ProcessingJob.id == job_id).first()
        if not job:
            raise ValueError(f"ProcessingJob {job_id!r} not found")

        if job.status == "cancelled":
            # Cancelled before the worker picked up the task (e.g. queued
            # behind other jobs). Leave the status as-is.
            logger.info("Job %s was cancelled before processing started", job_id)
            return

        job.status = "processing"
        db.commit()

        cancel_check = _make_cancel_check(job_id)

        # ── Load settings (fall back to pipeline defaults if no row yet) ──────
        settings_row = pp_pipeline_svc.get_settings(
            db,
            job.project_id,
            impulse_id=impulse_id,
        )
        config = (
            PipelineConfig.from_orm_or_dict(settings_row)
            if settings_row is not None
            else PipelineConfig()
        )
        logger.info(
            "Job %s: settings loaded (impulse=%s) — enabled=%s threshold=%.2f tracking=%s "
            "keep_grace=%d max_observations=%d class_filter=%r",
            job_id, impulse_id, config.enabled, config.threshold, config.tracking_enabled,
            config.keep_grace, config.max_observations, config.class_filter,
        )

        # ── Try to load the project's best detection model ────────────────────
        # impulse_id scopes model selection to the impulse the user is viewing;
        # without it the worker falls back to searching all project impulses.
        inference_fn: Optional[Callable] = None
        try:
            trained_model, label_names, input_shape = _load_best_detection_model_for_project(
                job.project_id, db, impulse_id=impulse_id
            )
            from app.services.model_testing_service import _load_model_into_memory
            model_handle = _load_model_into_memory(trained_model)
            tm_meta = trained_model.model_metadata or {}

            # SSD: resolve the training-parity confidence (0.01) here, where the
            # TrainedModel + DB are available, using the SAME helper evaluation
            # and live-classification use. Non-SSD models skip this entirely.
            ssd_conf_threshold: Optional[float] = None
            if tm_meta.get("output_type") == "ssd_detection":
                from app.services.model_testing_service import _resolve_ssd_threshold
                from app.models.user import TrainingJob as _TrainingJob
                _ssd_cr = None
                if trained_model.training_job_id:
                    _tj_ssd = db.query(_TrainingJob).filter(
                        _TrainingJob.id == trained_model.training_job_id
                    ).first()
                    if _tj_ssd and isinstance(_tj_ssd.classification_report, dict):
                        _ssd_cr = _tj_ssd.classification_report
                ssd_conf_threshold, _ssd_src = _resolve_ssd_threshold(trained_model, _ssd_cr)
                logger.info(
                    "Job %s: SSD conf_threshold=%.4f (source: %s)",
                    job_id, ssd_conf_threshold, _ssd_src,
                )

            inference_fn = _make_inference_fn(
                model_handle,
                label_names,
                input_shape,
                model_metadata=tm_meta,
                ssd_conf_threshold=ssd_conf_threshold,
            )
            _model_kind = "fomo" if _metadata_indicates_fomo(tm_meta) else "box"
            logger.info(
                "Job %s: inference via model %s (%s / %s / kind=%s)",
                job_id, trained_model.id, trained_model.format,
                tm_meta.get("variant", ""), _model_kind,
            )
        except ValueError as model_err:
            logger.warning(
                "Job %s: no detection model available (%s); trying S3 detections fallback",
                job_id, model_err,
            )

        output_key = StorageService.video_output_key(job.project_id, job.id)

        if inference_fn is not None:
            # ── Primary path: per-frame model inference ────────────────────────
            logger.info(
                "Job %s: render config — enabled=%s threshold=%.2f class_filter=%r "
                "tracking=%s keep_grace=%d max_observations=%d",
                job_id, config.enabled, config.threshold, config.class_filter,
                config.tracking_enabled, config.keep_grace, config.max_observations,
            )
            summary = VideoProcessor.process(
                input_key=job.input_video_path,
                output_key=output_key,
                config=config,
                inference_fn=inference_fn,
                cancel_check=cancel_check,
            )
        else:
            # ── Fallback path: pre-computed detections from S3 (debug upload) ─
            detections_per_frame: List[List[Detection]] = []
            det_key = StorageService.video_detections_key(job.project_id, job.id)
            try:
                det_bytes = storage.download_bytes(det_key)
                raw = json.loads(det_bytes)
                detections_per_frame = _parse_detections_per_frame(raw)
                logger.info(
                    "Job %s: using %d frames of pre-computed detections from S3",
                    job_id, len(detections_per_frame),
                )
            except Exception:
                model_message = _build_missing_detection_model_message(
                    job.project_id, db, impulse_id=impulse_id
                )
                raise ValueError(
                    f"{model_message.rstrip('.')} and no pre-computed detections were uploaded."
                )

            logger.info(
                "Job %s: render config (fallback) — enabled=%s threshold=%.2f class_filter=%r "
                "tracking=%s keep_grace=%d max_observations=%d",
                job_id, config.enabled, config.threshold, config.class_filter,
                config.tracking_enabled, config.keep_grace, config.max_observations,
            )
            summary = VideoProcessor.process(
                input_key=job.input_video_path,
                output_key=output_key,
                config=config,
                detections_per_frame=detections_per_frame,
                cancel_check=cancel_check,
            )

        # ── Mark complete ─────────────────────────────────────────────────────
        job.status = "complete"
        job.output_video_path = output_key
        job.inference_time_ms = summary.processing_ms
        db.commit()

        logger.info(
            "Job %s complete — %d frames, %.0f ms",
            job_id, summary.frames_processed, summary.processing_ms,
        )

    except ProcessingCancelledError:
        # Status is already 'cancelled' (written by the cancel endpoint before
        # this was observed) — nothing further to persist.
        logger.info("Video processing job %s cancelled by user", job_id)
    except Exception as exc:
        logger.exception("Video processing job %s failed", job_id)
        if job is not None:
            job.status = "failed"
            job.error_message = str(exc)[:_MAX_ERROR_LEN]
            try:
                db.commit()
            except Exception:
                db.rollback()
    finally:
        db.close()
