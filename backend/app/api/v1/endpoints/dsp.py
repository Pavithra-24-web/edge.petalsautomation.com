"""
DSP endpoints — configure and run feature extraction blocks.

New routes added:
  GET /dsp/processing-blocks          — Edge-Impulse-style block catalog (project-type-aware)
  GET /dsp/learning-blocks            — ML block catalog (project-type-aware)
  GET /dsp/blocks                     — Legacy alias kept for backward-compat
  POST /dsp/extract
  POST /dsp/preview
"""
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import Optional, List, Any
import json
import numpy as np

from app.core.database import get_db
from app.core.auth import get_current_user
from app.core.authz import (
    assert_project_owner,
    assert_impulse_owner,
    assert_sample_owner,
)
from app.models.user import User, Impulse, Sample, Label
from app.ml.dsp.processor import DSPProcessor
from app.workers.sample_utils import (
    label_name_is_placeholder,
    is_background_sample,
    is_sample_usable,
    UNLABELED_NAMES,
)

router = APIRouter()


def _normalize_box_label(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    lowered = text.lower()
    if lowered in {"unlabeled", "unlabelled", "unknown"}:
        return ""
    return text


def _resolve_sample_display_label(sample: Sample, db: Session) -> str:
    meta = sample.extra_metadata or {}
    boxes = meta.get("boundingBoxes") if isinstance(meta, dict) else None
    if isinstance(boxes, list):
        for box in boxes:
            if not isinstance(box, dict):
                continue

            box_label_name = _normalize_box_label(box.get("label"))
            if box_label_name:
                return box_label_name

            box_label_id = box.get("label_id")
            if box_label_id:
                lbl = (
                    db.query(Label)
                    .filter(Label.project_id == sample.project_id, Label.id == box_label_id)
                    .first()
                )
                if lbl:
                    return lbl.name

    return sample.label.name if sample.label else "Unknown"


def _resolve_cached_box_label(boxes_payload: Any, id_to_name: dict[str, str]) -> str:
    if boxes_payload is None:
        return ""

    try:
        boxes = json.loads(boxes_payload) if isinstance(boxes_payload, str) else boxes_payload
    except Exception:
        return ""

    if not isinstance(boxes, list):
        return ""

    for box in boxes:
        if not isinstance(box, dict):
            continue

        box_label_name = _normalize_box_label(box.get("label"))
        if box_label_name:
            return box_label_name

        box_label_id = box.get("label_id")
        if box_label_id and box_label_id in id_to_name:
            return id_to_name[box_label_id]

    return ""


# ─── Schemas ──────────────────────────────────────────────────────────────────

class DSPBlockConfig(BaseModel):
    type: str
    params: dict = {}


class DSPRequest(BaseModel):
    sample_id: str
    block: DSPBlockConfig


class DSPPreviewRequest(BaseModel):
    sample_id: str
    impulse_id: str
    blocks: List[DSPBlockConfig]


# ─── Block catalog data ───────────────────────────────────────────────────────

_PROCESSING_BLOCKS_ALL = [
    {
        "type": "spectral_analysis",
        "name": "Spectral Analysis",
        "official": True,
        "author": "Edge Impulse",
        "recommended": True,
        "description": "Extracts frequency-domain features using FFT. Works well for vibration, motion, and audio sensor data.",
        "sensor_types": ["accelerometer", "gyroscope", "microphone", "vibration", "time-series", "custom"],
        "input_types": ["time-series"],
        "params": {
            "fft_length":    {"type": "int",   "default": 256,   "min": 32,  "max": 4096},
            "overlap":       {"type": "float", "default": 0.5,   "min": 0.0, "max": 0.95},
            "noise_floor_db":{"type": "float", "default": -52.0},
            "filter_type":   {"type": "enum",  "options": ["none","low","high","bandpass"], "default": "none"},
            "filter_cutoff": {"type": "float", "default": 100.0},
            "scale_axes":    {"type": "float", "default": 1.0},
        },
    },
    {
        "type": "mfcc",
        "name": "MFCC",
        "official": True,
        "author": "Edge Impulse",
        "recommended": True,
        "description": "Computes Mel-frequency cepstral coefficients. The industry standard for speech and audio keyword spotting.",
        "sensor_types": ["microphone", "audio", "time-series"],
        "input_types": ["time-series"],
        "params": {
            "num_coefficients": {"type": "int",   "default": 13,     "min": 1,    "max": 40},
            "frame_length":     {"type": "int",   "default": 256},
            "frame_stride":     {"type": "int",   "default": 128},
            "num_filters":      {"type": "int",   "default": 40},
            "fft_length":       {"type": "int",   "default": 256},
            "low_frequency":    {"type": "float", "default": 300.0},
            "high_frequency":   {"type": "float", "default": 8000.0},
            "noise_floor_db":   {"type": "float", "default": -52.0},
        },
    },
    {
        "type": "spectrogram",
        "name": "Spectrogram",
        "official": True,
        "author": "Edge Impulse",
        "recommended": False,
        "description": "Produces a log-mel spectrogram as a 2-D image representation of audio signal power over time.",
        "sensor_types": ["microphone", "audio", "time-series"],
        "input_types": ["time-series"],
        "params": {
            "frame_length":    {"type": "int", "default": 256},
            "frame_stride":    {"type": "int", "default": 128},
            "fft_length":      {"type": "int", "default": 256},
            "num_mel_filters": {"type": "int", "default": 32},
            "noise_floor_db":  {"type": "float", "default": -52.0},
        },
    },
    {
        "type": "image",
        "name": "Image",
        "official": True,
        "author": "Edge Impulse",
        "recommended": True,
        "description": "Preprocesses and normalizes image data, and optionally reduces the color depth.",
        "sensor_types": ["camera", "image"],
        "input_types": ["image"],
        "params": {
            "image_width":  {"type": "int",  "default": 96},
            "image_height": {"type": "int",  "default": 96},
            "grayscale":    {"type": "bool", "default": False},
        },
    },
    {
        "type": "raw",
        "name": "Raw Data",
        "official": True,
        "author": "Edge Impulse",
        "recommended": False,
        "description": "Passes raw sensor samples directly with optional normalization. Use when you want full control over feature engineering.",
        "sensor_types": ["accelerometer", "gyroscope", "custom", "time-series"],
        "input_types": ["time-series"],
        "params": {
            "scale_axes": {"type": "float", "default": 1.0},
            "normalize":  {"type": "bool",  "default": True},
            "flatten":    {"type": "bool",  "default": True},
        },
    },
    {
        "type": "flatten",
        "name": "Flatten",
        "official": True,
        "author": "Edge Impulse",
        "recommended": False,
        "description": "Computes statistical features (mean, std, RMS, max, min, skewness, kurtosis) over each time-series axis.",
        "sensor_types": ["accelerometer", "gyroscope", "custom", "time-series"],
        "input_types": ["time-series"],
        "params": {
            "scale_axes": {"type": "float", "default": 1.0},
            "features": {
                "type": "list",
                "options": ["mean", "std", "rms", "max", "min", "skewness", "kurtosis"],
                "default": ["mean", "std", "rms"],
            },
        },
    },
    {
        "type": "eeg",
        "name": "EEG",
        "official": True,
        "author": "Edge Impulse",
        "recommended": False,
        "description": "Filters noise and extracts spectral power features from EEG signals.",
        "sensor_types": ["eeg", "biosignal", "custom"],
        "input_types": ["time-series"],
        "params": {
            "num_frequency_bands": {"type": "int",   "default": 5},
            "noise_floor_db":      {"type": "float", "default": -52.0},
        },
    },
]

_LEARNING_BLOCKS_ALL = [
    {
        "type": "dense",
        "name": "Classification",
        "official": True,
        "author": "Edge Impulse",
        "recommended": True,
        "description": "A fully-connected neural network for classification tasks. Works well with any flat feature vector from DSP blocks.",
        "input_types": ["time-series", "custom"],
        "sensor_types": ["accelerometer", "gyroscope", "microphone", "custom", "time-series"],
        "params": {
            "units":      {"type": "list_int", "default": [128, 64]},
            "dropout":    {"type": "float",    "default": 0.3},
            "activation": {"type": "enum",     "options": ["relu", "elu", "selu"], "default": "relu"},
        },
    },
    {
        "type": "conv1d",
        "name": "1D Convolutional (CNN)",
        "official": True,
        "author": "Edge Impulse",
        "recommended": True,
        "description": "A 1-D convolutional network for temporal patterns. Best for sequential sensor data and audio.",
        "input_types": ["time-series"],
        "sensor_types": ["accelerometer", "gyroscope", "microphone", "vibration", "time-series"],
        "params": {
            "filters":     {"type": "list_int", "default": [32, 64, 128]},
            "kernel_size": {"type": "int",       "default": 3},
            "dropout":     {"type": "float",     "default": 0.3},
        },
    },
    {
        "type": "conv2d",
        "name": "2D Convolutional (CNN)",
        "official": True,
        "author": "Edge Impulse",
        "recommended": False,
        "description": "A 2-D convolutional network for spectrogram and image inputs.",
        "input_types": ["time-series", "image"],
        "sensor_types": ["microphone", "camera", "image", "time-series"],
        "params": {
            "filters":     {"type": "list_int", "default": [16, 32, 64]},
            "kernel_size": {"type": "int",       "default": 3},
            "dropout":     {"type": "float",     "default": 0.3},
        },
    },
    {
        "type": "lstm",
        "name": "LSTM",
        "official": True,
        "author": "Edge Impulse",
        "recommended": False,
        "description": "A long short-term memory (LSTM) recurrent network for long-range sequential dependencies.",
        "input_types": ["time-series"],
        "sensor_types": ["accelerometer", "gyroscope", "custom", "time-series"],
        "params": {
            "lstm_units": {"type": "list_int", "default": [64, 32]},
            "dropout":    {"type": "float",    "default": 0.3},
        },
    },
    {
        "type": "mobilenet",
        "name": "MobileNetV2 (Images)",
        "official": True,
        "author": "Edge Impulse",
        "recommended": True,
        "description": "Transfer learning with MobileNetV2. Efficient image classification optimized for edge devices.",
        "input_types": ["image"],
        "sensor_types": ["camera", "image"],
        "params": {
            "alpha":      {"type": "float", "default": 0.35},
            "dropout":    {"type": "float", "default": 0.2},
            "image_size": {"type": "int",   "default": 96},
        },
    },
    {
        "type": "object_detection",
        "name": "Object Detection (Images)",
        "official": True,
        "author": "Edge Impulse",
        "recommended": False,
        "description": "Fine tune a pre-trained object detection model on your data. Good performance even with relatively small image datasets.",
        "input_types": ["image"],
        "sensor_types": ["camera", "image"],
        "params": {
            "model":      {"type": "enum", "options": ["fomo", "yolov5"], "default": "fomo"},
            "image_size": {"type": "int",  "default": 96},
        },
    },
    {
        "type": "fomo_mobilenetv2_0_1",
        "name": "FOMO (Faster Objects, More Objects) MobileNetV2 0.35",
        "official": True,
        "author": "Edge Impulse",
        "recommended": True,
        "description": (
            "FOMO is a novel machine learning algorithm that brings real-time object detection, "
            "counting and location to microcontrollers for the first time. "
            "Uses MobileNetV2 with alpha=0.35 backbone — efficient for edge deployment. "
            "Outputs a class heatmap rather than bounding boxes, suitable for constrained MCUs. "
            "Choose FOMO version: v1 (Legacy 96×96, stride-16, stable) or "
            "v2 (Adaptive Resolution, stride-8, variable input size)."
        ),
        "input_types": ["image"],
        "sensor_types": ["camera", "image"],
        "params": {
            "alpha":        {"type": "float", "default": 0.35,
                             "description": "MobileNetV2 width multiplier. 0.35 = smallest supported with pretrained weights."},
            "image_width":  {"type": "int",   "default": 96,
                             "description": "Input image width (must be multiple of 8)."},
            "image_height": {"type": "int",   "default": 96,
                             "description": "Input image height (must be multiple of 8)."},
            "epochs":       {"type": "int",   "default": 60},
            "learning_rate":{"type": "float", "default": 0.001},
            "batch_size":   {"type": "int",   "default": 32},
            "fomo_version": {"type": "int",   "default": 1,
                             "description": (
                                 "FOMO version. "
                                 "1 = FOMO (Legacy 96×96): stable, stride-16, enforces 96×96 minimum. "
                                 "2 = FOMO (Adaptive Resolution): stride-8, variable input size, "
                                 "4× more detection cells, no 96×96 floor."
                             )},
        },
    },
    {
        "type": "anomaly_gmm",
        "name": "Anomaly Detection",
        "official": True,
        "author": "Edge Impulse",
        "recommended": False,
        "description": "Find outliers in new data. New data that is unlikely according to this model can be considered anomalous. This block supports custom ML blocks for anomaly detection.",
        "input_types": ["time-series", "image", "custom"],
        "sensor_types": ["accelerometer", "gyroscope", "microphone", "custom", "time-series", "camera"],
        "params": {
            "n_components": {"type": "int",   "default": 32},
            "threshold":    {"type": "float", "default": 0.3},
        },
    },
]


def _filter_blocks(blocks: list, input_type: str, sensor_type: str, show_all: bool) -> list:
    if show_all:
        return blocks
    result = []
    for b in blocks:
        type_match   = not input_type  or input_type  in b.get("input_types",  [])
        sensor_match = not sensor_type or sensor_type in b.get("sensor_types", [])
        if type_match or sensor_match:
            result.append(b)
    return result if result else blocks


# Per-project-type allowlist for the two catalogs above (motion_phase0.md
# §7.3) — an allowlist, not a complement: a block absent from both rows
# ("mfcc", "spectrogram", "eeg") is offered to neither type and stays
# reachable only through show_all=true. Reproduces exactly what each type is
# served today (§7.4); neither registry literal is edited by this map.
_BLOCKS_BY_PROJECT_TYPE = {
    "object_detection": {
        "processing": {"image"},
        "learning":   {"object_detection", "fomo_mobilenetv2_0_1", "mobilenet", "conv2d", "anomaly_gmm"},
    },
    "motion": {
        "processing": {"raw", "flatten", "spectral_analysis"},
        "learning":   {"dense", "conv1d", "lstm", "anomaly_gmm"},
    },
}


def _filter_by_project_type(blocks: list, project_type: Optional[str], catalog: str) -> list:
    """Narrow an already-filtered block list to a project type's allowlist.
    An omitted or unrecognized project_type is a no-op — today's payload."""
    allowed = _BLOCKS_BY_PROJECT_TYPE.get(project_type or "", {}).get(catalog)
    if not allowed:
        return blocks
    return [b for b in blocks if b.get("type") in allowed]


# ─── Routes ───────────────────────────────────────────────────────────────────

@router.get("/processing-blocks")
def list_processing_blocks(
    impulse_id: Optional[str] = Query(None),
    show_all:   bool           = Query(False),
    project_type: Optional[str] = Query(None),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Edge-Impulse-style processing block catalog.
    When impulse_id is provided, filters to blocks compatible with the impulse's
    input type and sensor type. Pass show_all=true to bypass filtering.
    Pass project_type to additionally narrow to that type's allowlist
    (bypassed by show_all=true, same as the input/sensor filter above).

    Motion skips the input/sensor filter entirely: motion's allowlist in
    _BLOCKS_BY_PROJECT_TYPE is already the complete, correct definition of
    what belongs to a motion impulse, and layering the input/sensor filter on
    top makes the catalog depend on an impulse row's live input_type/
    sensor_type — fields that can go stale (e.g. an impulse whose input_type
    was set to "image" by an earlier DSP block that has since been removed;
    _sync_root_input_fields's preserve_explicit_image only resets that when
    the client stops re-sending "image", which a motion impulse — whose UI
    never lets the user pick that value — has no way to do). Object Detection
    is unaffected: it still runs through _filter_blocks exactly as before.
    """
    input_type = ""
    sensor_type = ""

    if impulse_id:
        imp = assert_impulse_owner(db, impulse_id, current_user)
        input_type  = imp.input_type  or "time-series"
        sensor_type = imp.sensor_type or ""

    if project_type == "motion" and not show_all:
        visible = _PROCESSING_BLOCKS_ALL
    else:
        visible = _filter_blocks(_PROCESSING_BLOCKS_ALL, input_type, sensor_type, show_all)
    if not show_all:
        visible = _filter_by_project_type(visible, project_type, "processing")
    hidden  = len(_PROCESSING_BLOCKS_ALL) - len(visible)

    return {
        "blocks":       visible,
        "hidden_count": hidden,
        "input_type":   input_type,
        "sensor_type":  sensor_type,
    }


@router.get("/learning-blocks")
def list_learning_blocks(
    impulse_id: Optional[str] = Query(None),
    show_all:   bool           = Query(False),
    project_type: Optional[str] = Query(None),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Edge-Impulse-style learning block catalog.
    Filters by impulse input type when impulse_id is provided.
    Pass project_type to additionally narrow to that type's allowlist
    (bypassed by show_all=true, same as the input/sensor filter above).

    Motion skips the input/sensor filter entirely — see the matching note on
    list_processing_blocks above; the same stale input_type/sensor_type risk
    applies here; motion's own learning allowlist is already authoritative.
    """
    input_type  = ""
    sensor_type = ""

    if impulse_id:
        imp = assert_impulse_owner(db, impulse_id, current_user)
        input_type  = imp.input_type  or "time-series"
        sensor_type = imp.sensor_type or ""

    if project_type == "motion" and not show_all:
        visible = _LEARNING_BLOCKS_ALL
    else:
        visible = _filter_blocks(_LEARNING_BLOCKS_ALL, input_type, sensor_type, show_all)
    if not show_all:
        visible = _filter_by_project_type(visible, project_type, "learning")
    hidden  = len(_LEARNING_BLOCKS_ALL) - len(visible)

    return {
        "blocks":       visible,
        "hidden_count": hidden,
        "input_type":   input_type,
        "sensor_type":  sensor_type,
    }


@router.get("/blocks")
def list_dsp_blocks_legacy(current_user: User = Depends(get_current_user)):
    """Legacy alias — returns processing blocks in the old format for backward compat."""
    return {"blocks": _PROCESSING_BLOCKS_ALL}


@router.post("/extract")
def extract_features(
    req: DSPRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Extract features from a single sample using a DSP block config."""
    from app.core.storage import storage
    import json

    sample = assert_sample_owner(db, req.sample_id, current_user)

    from app.core.storage import download_bytes_cached
    raw_bytes = download_bytes_cached(sample.storage_key)

    processor = DSPProcessor(
        block_type=req.block.type,
        params=req.block.params,
        frequency_hz=sample.frequency_hz or 100.0,
    )
    try:
        features = processor.extract(raw_bytes)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not parse sample data: {e}")

    return {
        "sample_id":    req.sample_id,
        "block_type":   req.block.type,
        "features":     features.tolist(),
        "feature_shape":list(features.shape),
        "num_features": features.size,
    }





from celery.result import AsyncResult
from app.workers.celery_app import celery_app
from app.workers.dsp_worker import extract_features_for_impulse
from app.models.user import Label, DspFeatureJob, JobStatus

class GenerateRequest(BaseModel):
    impulse_id: str

class ParamsRequest(BaseModel):
    impulse_id: str
    params: dict
    block_index: int = 0  # which DSP block to update; defaults to 0



@router.post("/generate-features")
def generate_all_features(
    req: GenerateRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    # Validate the impulse so a bad id doesn't silently spawn a Celery task.
    impulse = assert_impulse_owner(db, req.impulse_id, current_user)

    # One active feature-generation run per impulse. Mirrors the equivalent
    # guard on training (training.py: "A training job is already running")
    # so the jobs page can't show two in-flight rows for the same impulse.
    in_flight = (
        db.query(DspFeatureJob)
        .filter(
            DspFeatureJob.impulse_id == req.impulse_id,
            DspFeatureJob.status.in_([JobStatus.pending, JobStatus.running]),
        )
        .first()
    )
    if in_flight:
        raise HTTPException(
            status_code=409,
            detail="A feature generation job is already running for this impulse",
        )

    # Persist the row BEFORE dispatching Celery so the row exists even if
    # the broker rejects the message; the worker can then update by job id.
    # Commit here (not after dispatch) so the row is durable even when the
    # broker publish below fails.
    job = DspFeatureJob(
        impulse_id=req.impulse_id,
        project_id=impulse.project_id,
        input_type=impulse.input_type,
        status=JobStatus.pending,
    )
    db.add(job)
    db.commit()
    db.refresh(job)

    # Dispatch to the DSP worker. If the broker is unreachable, .delay() raises;
    # letting that escape produces a raw 500 that skips the CORS middleware and
    # shows up in the browser as a misleading "CORS policy" error. Mark the job
    # failed and return a clean 503 instead so the failure is visible and the
    # in-flight guard above doesn't wedge future runs on a "pending" row.
    try:
        task = extract_features_for_impulse.delay(req.impulse_id, job.id)
    except Exception as e:
        job.status = JobStatus.failed
        job.error_message = f"Could not queue feature generation: {e}"
        db.commit()
        raise HTTPException(
            status_code=503,
            detail="Feature generation queue is unavailable. Please try again shortly.",
        )

    job.celery_task_id = task.id
    db.commit()
    db.refresh(job)

    return {"job_id": job.id, "status": "pending"}

@router.get("/job-status/{job_id}")
def get_job_status(
    job_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Read job status.

    Prefers the DspFeatureJob DB row (durable — Celery results expire from
    the broker after a short window, but the DB row sticks around so the
    /jobs page can read history). Falls back to the live Celery state for
    progress information and for legacy ids that pre-date the DB row.
    """
    # 1) DB row — durable status that survives Celery result expiry.
    db_job = db.query(DspFeatureJob).filter(DspFeatureJob.id == job_id).first()
    # Enforce ownership when a real job row exists. The legacy fallback path
    # (no DB row — job_id is a bare Celery task id) only ever returns a status
    # string with no project data, so it stays accessible to any authed user.
    if db_job is not None:
        assert_project_owner(db, db_job.project_id, current_user)

    # 2) Live Celery state — uses celery_task_id from the DB row when
    #    present so progress is still surfaced. For legacy ids (no row),
    #    treat job_id itself as the celery task id like the old behaviour.
    celery_id = db_job.celery_task_id if (db_job and db_job.celery_task_id) else job_id

    info: dict = {}
    progress = 0
    celery_state = None
    try:
        task = AsyncResult(celery_id, app=celery_app)
        celery_state = (task.state or "").lower()
        info = task.info if isinstance(task.info, dict) else {}
        if celery_state in ("started", "running"):
            progress = info.get("progress", 0)
        elif celery_state == "success":
            progress = 100
    except Exception:
        # Celery is best-effort here; DB row is authoritative for status.
        pass

    if db_job is not None:
        status_str = db_job.status.value if hasattr(db_job.status, "value") else str(db_job.status)
        if status_str == "completed":
            progress = 100
        return {
            "status":  status_str,
            "progress": progress,
            "result":  None,
            "error":   db_job.error_message,
            "meta":    info,
        }

    # Legacy path: no DB row exists for this id. Map Celery state directly,
    # preserving the pre-existing response shape.
    status_str = (
        "completed" if celery_state == "success"
        else ("failed" if celery_state == "failure" else (celery_state or "pending"))
    )
    return {
        "status":  status_str,
        "progress": progress,
        "result":  None,
        "error":   None,
        "meta":    info,
    }


@router.get("/input-size")
def get_input_size(
    impulse_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Compute the real DSP feature vector length for an impulse by running
    a representative sample through the configured DSP pipeline.

    Returns:
        feature_count: int   — number of features (flattened)
        feature_shape: list  — raw shape from DSP output
        source: str          — how it was computed
        message: str         — human-readable description
    """
    from app.core.storage import storage

    impulse = assert_impulse_owner(db, impulse_id, current_user)

    dsp_blocks = impulse.dsp_blocks or []
    if not dsp_blocks:
        return {
            "feature_count": None,
            "feature_shape": None,
            "source": "no_dsp",
            "message": "No DSP blocks configured. Add a processing block first.",
        }

    # Pick a representative sample (prefer training split, labeled)
    sample = (
        db.query(Sample)
        .filter(
            Sample.project_id == impulse.project_id,
            Sample.label_id.isnot(None),
            Sample.sample_type == "training",
        )
        .first()
    )
    if sample is None:
        # Relax constraint: any labeled sample
        sample = (
            db.query(Sample)
            .filter(
                Sample.project_id == impulse.project_id,
                Sample.label_id.isnot(None),
            )
            .first()
        )
    if sample is None:
        # Last resort: any sample in the project
        sample = (
            db.query(Sample)
            .filter(Sample.project_id == impulse.project_id)
            .first()
        )

    if sample is None:
        return {
            "feature_count": None,
            "feature_shape": None,
            "source": "no_sample",
            "message": "Input layer unavailable: no samples found. Upload data first.",
        }

    try:
        from app.core.storage import download_bytes_cached
        raw_bytes = download_bytes_cached(sample.storage_key)
    except Exception as e:
        return {
            "feature_count": None,
            "feature_shape": None,
            "source": "storage_error",
            "message": f"Input layer unavailable: could not load sample — {e}",
        }

    try:
        from app.ml.dsp.processor import merge_image_params
        freq = sample.frequency_hz or impulse.frequency_hz or 100.0
        all_features = []
        for block_cfg in dsp_blocks:
            params = merge_image_params(impulse, block_cfg)
            proc = DSPProcessor(
                block_type=block_cfg.get("type", "raw"),
                params=params,
                frequency_hz=freq,
            )
            all_features.append(proc.extract(raw_bytes).flatten())

        features = np.concatenate(all_features) if len(all_features) > 1 else all_features[0]
        feature_count = int(features.size)
        feature_shape = list(features.shape)
    except Exception as e:
        return {
            "feature_count": None,
            "feature_shape": None,
            "source": "dsp_error",
            "message": f"Input layer unavailable: DSP extraction failed — {e}",
        }

    block_types = ",".join(b.get("type", "?") for b in dsp_blocks)
    return {
        "feature_count": feature_count,
        "feature_shape": feature_shape,
        "source": "dsp_computed",
        "message": f"Input layer ({feature_count:,} features)",
        "sample_id": str(sample.id),
        "block_type": block_types,
    }


@router.get("/dataset-summary")
def get_dataset_summary(
    impulse_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    impulse = assert_impulse_owner(db, impulse_id, current_user)

    labels = db.query(Label).filter(Label.project_id == impulse.project_id).all()

    all_samples = (
        db.query(Sample)
        .filter(Sample.project_id == impulse.project_id)
        .all()
    )

    # Only labels with non-placeholder names are real training classes.
    usable_label_ids = {l.id for l in labels if not label_name_is_placeholder(l.name)}

    # Count train and test separately — match Edge Impulse which shows each split independently.
    train_samples = [s for s in all_samples if str(getattr(s.sample_type, "value", s.sample_type)).lower() != "testing"]
    test_samples  = [s for s in all_samples if str(getattr(s.sample_type, "value", s.sample_type)).lower() == "testing"]

    # This endpoint has no architecture context of its own, so resolve whether
    # the impulse trains a detection model the same way the training worker
    # does.  Only detection consumes background ("negative") images; counting
    # them for a classification impulse would over-report the training set by
    # samples that path will never load.
    from app.api.v1.endpoints.training import _resolve_training_architecture
    _architecture = _resolve_training_architecture(impulse)
    _is_detection = any(
        token in str(_architecture).lower()
        for token in ("fomo", "yolo", "ssd", "object_detection", "detection")
    )

    train_count = sum(
        1 for s in train_samples
        if is_sample_usable(s, usable_label_ids, allow_background=_is_detection)
    )
    test_count  = sum(
        1 for s in test_samples
        if is_sample_usable(s, usable_label_ids, allow_background=_is_detection)
    )
    # Reported as its own field rather than folded into total_samples: a user
    # reading "120 samples" needs to know how many of them contain nothing.
    background_count = sum(
        1 for s in all_samples if is_background_sample(s)
    ) if _is_detection else 0

    # Build the set of label IDs actually referenced by usable samples,
    # excluding placeholder labels and placeholder-keyed box annotations.
    # Background samples are skipped so a negative can never mint a class.
    used_label_ids: set = set()
    for s in all_samples:
        if is_background_sample(s):
            continue
        if s.label_id and s.label_id in usable_label_ids:
            used_label_ids.add(s.label_id)
        if isinstance(s.extra_metadata, dict):
            for box in s.extra_metadata.get("boundingBoxes", []):
                lid = box.get("label_id") or box.get("label")
                if lid and str(lid).lower() not in UNLABELED_NAMES:
                    used_label_ids.add(lid)

    active_labels = [
        l for l in labels
        if (l.id in used_label_ids or l.name in used_label_ids)
        and not label_name_is_placeholder(l.name)
    ]

    return {
        "total_samples": train_count,       # training split only — matches Edge Impulse "Data in training set"
        "test_samples": test_count,
        "num_classes": len(active_labels),
        "class_names": [l.name for l in active_labels],
        # Background ("negative") images across both splits. 0 for
        # classification impulses, which never train on them.
        "background_samples": background_count,
    }

@router.post("/{job_id}/cancel")
def cancel_dsp_job(
    job_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Cancel a pending or running feature generation job."""
    from datetime import datetime

    job = db.query(DspFeatureJob).filter(DspFeatureJob.id == job_id).first()
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")

    assert_project_owner(db, job.project_id, current_user)

    if job.status not in (JobStatus.pending, JobStatus.running):
        raise HTTPException(400, f"Job is already {job.status} — cannot cancel")

    job.status = JobStatus.cancelled
    job.error_message = "Cancelled by user"
    job.completed_at = datetime.utcnow()
    db.commit()

    if job.celery_task_id:
        try:
            celery_app.control.revoke(job.celery_task_id)
        except Exception:
            pass

    return {"message": "Job cancelled", "job_id": job_id}


@router.get("/features-ready/{impulse_id}")
def get_features_ready(
    impulse_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Fast readiness check — HEAD-only, no download, no PCA.

    The full /features/{impulse_id} endpoint runs PCA on the entire feature
    matrix and can take many seconds; the Training page only needs to know
    whether features.npz exists, so it polls this lightweight endpoint instead.
    """
    from app.core.storage import storage
    from botocore.exceptions import ClientError

    impulse = assert_impulse_owner(db, impulse_id, current_user)
    key = storage.model_key(impulse.project_id, impulse_id, "features.npz")
    try:
        storage.client.head_object(Bucket=storage.bucket, Key=key)
        return {"ready": True}
    except ClientError:
        return {"ready": False}


@router.get("/features/{impulse_id}")
def get_features_data(
    impulse_id: str, 
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Fetch the pre-computed features and project them to 2D using PCA.
    Returns data points for the Feature Explorer scatter plot.
    """
    from app.core.storage import storage
    from sklearn.decomposition import PCA
    from botocore.exceptions import ClientError
    import numpy as np
    import io
    import logging

    logger = logging.getLogger(__name__)

    impulse = assert_impulse_owner(db, impulse_id, current_user)

    storage_key = storage.model_key(impulse.project_id, impulse_id, "features.npz")
    # `ready` signals presence of the features.npz blob — training only needs the
    # cached features to exist. It is decoupled from PCA success so downstream
    # gates (e.g. the Training page's "features generated" check) don't misread a
    # PCA-visualization failure as "features missing".
    ready = False
    try:
        try:
            data_bytes = storage.download_bytes(storage_key)
            ready = True
        except ClientError as e:
            code = e.response.get("Error", {}).get("Code", "")
            if code in ("NoSuchKey", "404", "NoSuchBucket"):
                return {"points": [], "ready": False}
            raise
        with np.load(io.BytesIO(data_bytes), allow_pickle=True) as data:
            X = data["X"]
            y = data["y"]
            ids = data.get("ids", []) # handle legacy files without ids
            boxes_json = data.get("boxes_json", [])
            sample_types = data.get("sample_types", np.array([]))

        if X.shape[0] < 2:
            return {"points": [], "ready": ready}

        # PCA expects a 2-D matrix of shape (samples, features). Image pipelines
        # now store tensors like (N, H, W, C), so flatten each sample only for
        # visualization/projection without changing the stored training data.
        if getattr(X, "dtype", None) == object:
            try:
                X = np.stack([np.asarray(sample, dtype=np.float32).flatten() for sample in X], axis=0)
            except Exception as e:
                return {
                    "points": [],
                    "ready": ready,
                    "message": f"Features could not be projected consistently: {e}",
                }
        elif X.ndim > 2:
            X = X.reshape((X.shape[0], -1)).astype(np.float32)

        # Project to 2D — fit PCA on TRAIN samples only (Edge Impulse behavior),
        # then transform all samples so test points are visible but don't influence axes.
        n_components = 2
        pca = PCA(n_components=n_components)
        if len(sample_types) == len(X):
            train_mask = np.array([
                str(st).lower() not in ("testing", "test", "sampletype.testing")
                for st in sample_types
            ])
            X_train_for_fit = X[train_mask] if train_mask.sum() >= 2 else X
        else:
            X_train_for_fit = X  # legacy cache without sample_types
        pca.fit(X_train_for_fit)
        X_2d = pca.transform(X)

        # Map labels to names
        labels = db.query(Label).filter(Label.project_id == impulse.project_id).all()
        id_to_name = {l.id: l.name for l in labels}
        
        points = []
        for i in range(X_2d.shape[0]):
            label_id = y[i]
            box_label = (
                _resolve_cached_box_label(boxes_json[i], id_to_name)
                if len(boxes_json) > i
                else ""
            )
            points.append({
                "x": float(X_2d[i, 0]),
                "y": float(X_2d[i, 1]),
                "label": box_label or id_to_name.get(label_id, "Unknown"),
                "sample_id": str(ids[i]) if len(ids) > i else None
            })

        return {
            "points": points,
            "ready": ready,
            "variance_ratio": pca.explained_variance_ratio_.tolist()
        }
    except Exception:
        logger.exception("Failed to load features for impulse %s", impulse_id)
        return {"points": [], "ready": ready, "message": "We couldn't load the feature data. Please try regenerating features."}


def _feature_filename_slug(text: str) -> str:
    """Filename-safe slug for the downloaded features file."""
    import re
    cleaned = re.sub(r"[^A-Za-z0-9._\- ]+", "", text or "").strip()
    cleaned = re.sub(r"\s+", "-", cleaned)
    return cleaned or "impulse"


@router.get("/features/{impulse_id}/download")
def download_features(
    impulse_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Download the raw feature matrix (X) for an impulse as a `.npy` file.

    Loads the same `features.npz` cache `get_features_data` reads and applies
    the same object-dtype stack/flatten path before writing `X` out — this is
    the usable numeric feature matrix, not the raw per-sample tensor container.
    """
    from app.core.storage import storage
    from botocore.exceptions import ClientError
    from fastapi.responses import StreamingResponse
    import io

    impulse = assert_impulse_owner(db, impulse_id, current_user)

    storage_key = storage.model_key(impulse.project_id, impulse_id, "features.npz")
    try:
        data_bytes = storage.download_bytes(storage_key)
    except ClientError as e:
        code = e.response.get("Error", {}).get("Code", "")
        if code in ("NoSuchKey", "404", "NoSuchBucket"):
            raise HTTPException(status_code=404, detail="Features have not been generated for this impulse")
        raise

    with np.load(io.BytesIO(data_bytes), allow_pickle=True) as data:
        X = data["X"]

    # Same stack/flatten path get_features_data applies before PCA — image
    # pipelines store per-sample tensors as an object array; flatten each
    # sample into a single numeric matrix rather than exporting the raw
    # object-dtype container.
    if getattr(X, "dtype", None) == object:
        X = np.stack([np.asarray(sample, dtype=np.float32).flatten() for sample in X], axis=0)
    elif X.ndim > 2:
        X = X.reshape((X.shape[0], -1)).astype(np.float32)

    buf = io.BytesIO()
    np.save(buf, X)
    buf.seek(0)

    filename = f"{_feature_filename_slug(impulse.name)}_features.npy"
    return StreamingResponse(
        buf,
        media_type="application/octet-stream",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.post("/parameters")
def save_dsp_parameters(
    req: ParamsRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    from app.core.storage import storage
    from sqlalchemy.orm.attributes import flag_modified

    impulse = assert_impulse_owner(db, req.impulse_id, current_user)

    blocks = list(impulse.dsp_blocks or [])
    if not blocks:
        raise HTTPException(
            status_code=400,
            detail="Impulse has no DSP blocks. Add a processing block first.",
        )
    if req.block_index < 0 or req.block_index >= len(blocks):
        raise HTTPException(
            status_code=400,
            detail=(
                f"block_index {req.block_index} out of range "
                f"(impulse has {len(blocks)} DSP block(s))."
            ),
        )

    # Replace only the targeted block's params — leave other blocks untouched.
    blocks[req.block_index] = {**blocks[req.block_index], "params": req.params}
    impulse.dsp_blocks = blocks

    # Sync root-level image fields when an image block is being saved.
    block_type = blocks[req.block_index].get("type", "")
    if block_type == "image":
        if req.params.get("image_width"):
            impulse.image_width = int(req.params["image_width"])
        if req.params.get("image_height"):
            impulse.image_height = int(req.params["image_height"])
        if req.params.get("resize_mode"):
            impulse.resize_mode = req.params["resize_mode"]

    # Invalidate the pre-computed feature cache so the explorer shows fresh data.
    try:
        cache_key = storage.model_key(impulse.project_id, impulse.id, "features.npz")
        storage.delete_file(cache_key)
    except Exception:
        pass  # cache may not exist yet — that's fine

    flag_modified(impulse, "dsp_blocks")
    db.commit()
    return {"status": "success", "dsp_blocks": blocks}

@router.post("/preview")
def create_dsp_preview(
    req: DSPPreviewRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    from app.core.storage import storage
    from app.ml.dsp.processor import merge_image_params
    import base64
    from PIL import Image
    import io
    import numpy as np

    sample = assert_sample_owner(db, req.sample_id, current_user)

    impulse = assert_impulse_owner(db, req.impulse_id, current_user)

    from app.core.storage import download_bytes_cached
    raw_bytes = download_bytes_cached(sample.storage_key)
    label_name = _resolve_sample_display_label(sample, db)

    # Capture raw image dimensions for the UI's bounding-box overlay.
    raw_img_width, raw_img_height = 0, 0
    try:
        img_temp = Image.open(io.BytesIO(raw_bytes))
        raw_img_width, raw_img_height = img_temp.width, img_temp.height
    except Exception:
        pass

    raw_data_uri = f"data:image/jpeg;base64,{base64.b64encode(raw_bytes).decode('utf-8')}"

    # Run every configured DSP block; merge impulse root image params as fallbacks.
    blocks_to_run = req.blocks if req.blocks else [DSPBlockConfig(type="image", params={})]
    freq = sample.frequency_hz or 100.0

    all_flat_features = []
    proc_data_uri = None   # reconstructed processed image (first image block only)
    first_image_shape = []

    for block_cfg in blocks_to_run:
        merged_params = merge_image_params(impulse, {"params": dict(block_cfg.params)})
        proc = DSPProcessor(
            block_type=block_cfg.type,
            params=merged_params,
            frequency_hz=freq,
        )
        feat = proc.extract(raw_bytes)
        all_flat_features.append(feat.flatten())

        # Build the visual "processed image" preview from the first image-type block.
        if proc_data_uri is None and block_cfg.type == "image":
            first_image_shape = list(feat.shape)
            width   = int(merged_params.get("image_width",  96))
            height  = int(merged_params.get("image_height", 96))
            is_gray = bool(merged_params.get("grayscale", False))
            try:
                if is_gray:
                    arr  = (feat * 255).astype(np.uint8).reshape((height, width))
                    mode = "L"
                else:
                    arr  = (feat * 255).astype(np.uint8).reshape((height, width, 3))
                    mode = "RGB"
                p_img = Image.fromarray(arr, mode=mode)
                buf   = io.BytesIO()
                p_img.save(buf, format="PNG")
                proc_data_uri = (
                    "data:image/png;base64,"
                    + base64.b64encode(buf.getvalue()).decode("utf-8")
                )
            except Exception:
                proc_data_uri = None

    # Concatenate features from all blocks into one vector.
    features = (
        np.concatenate(all_flat_features) if len(all_flat_features) > 1
        else all_flat_features[0]
    )
    shape = first_image_shape if first_image_shape else list(features.shape)

    try:
        raw_img      = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
        raw_features = np.array(raw_img).flatten()[:100].tolist()
    except Exception:
        raw_features = list(raw_bytes[:100])

    return {
        "features": features.tolist(),
        "processed_features": features.tolist(),
        "raw_features": raw_features,
        "shape": shape,
        "raw_image": raw_data_uri,
        "raw_image_width": raw_img_width,
        "raw_image_height": raw_img_height,
        "processed_image": proc_data_uri,
        "label": label_name
    }