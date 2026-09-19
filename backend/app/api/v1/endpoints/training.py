"""
Training endpoints — start, monitor, cancel, and retrain training jobs.

Supports:
  POST /training/start                   — Launch a new training job
  POST /training/retrain                 — Retrain using last job's hyperparams
  GET  /training/{job_id}               — Poll job status + live metrics
  GET  /training/impulse/{id}           — List all jobs for an impulse
  POST /training/{job_id}/cancel        — Cancel a running job
  GET  /training/impulse/{id}/status    — Quick summary: trained / not trained
"""
import logging

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import Literal, Optional
from datetime import datetime

from app.core.config import settings
from app.core.database import get_db
from app.core.auth import get_current_user
from app.core.authz import assert_impulse_owner, assert_training_job_owner
from app.core.gpu_availability import gpu_queue_has_live_worker
from app.models.user import User, TrainingJob, Impulse, TrainedModel, JobStatus
from app.workers.training_worker import run_training_job
from app.ml.training_ui_log_filter import filter_training_ui_log_lines

logger = logging.getLogger(__name__)

router = APIRouter()


_UNSUPPORTED_TRAINING_ARCHITECTURES = {
    # Object-detection architectures not yet implemented end-to-end
    "fomo_v2_0.35",
    # mobilenet_v2_ssd_fpn_lite is fully implemented via
    # app.ml.MobileNetV2 SSD + mobilenetv2_ssd_worker.
    # yolo_pro removed — now fully implemented via app.ml.yolo_pro + yolo_pro_worker
    # Generic catalog entry — callers must use an explicit architecture name
    # (e.g. fomo_mobilenetv2_0_1). The "object_detection" type has no builder.
    "object_detection",
    # Anomaly detection requires a GMM-specific pipeline, not Keras training.
    "anomaly_gmm",
}


# ─── Schemas ──────────────────────────────────────────────────────────────────

class TrainingRequest(BaseModel):
    impulse_id: str
    epochs: int = 100
    learning_rate: float = 0.001
    batch_size: int = 32
    validation_split: float = 0.2
    device_preference: Literal["cpu", "gpu"] = "gpu"
    # Training toggles surfaced on /dashboard/impulse/training. Persisted into
    # extra_params at job creation so workers (e.g. the SSD trainer) can honor
    # them. Default-True keeps existing classification/FOMO/YOLO behavior.
    data_augmentation: bool = True
    early_stopping: bool = True
    extra_params: dict = {}


class RetrainRequest(BaseModel):
    impulse_id: str
    epochs: Optional[int] = None
    learning_rate: Optional[float] = None
    batch_size: Optional[int] = None


# ─── Serializer ───────────────────────────────────────────────────────────────

def _job_to_dict(j: TrainingJob) -> dict:
    th = j.training_history or {}
    ep = j.extra_params or {}
    # The DB stores the unfiltered diagnostic stream so the audit trail is
    # complete; the UI sees a simplified projection.  Filter here rather
    # than at write time so backend logger diagnostics, the raw stored
    # log_lines, and saved-model metadata are all unaffected.
    if isinstance(th, dict) and "log_lines" in th:
        th = {**th, "log_lines": filter_training_ui_log_lines(th.get("log_lines"))}
    return {
        "id":                    j.id,
        "impulse_id":            j.impulse_id,
        "status":                j.status,
        "epochs":                j.epochs,
        "learning_rate":         j.learning_rate,
        "batch_size":            j.batch_size,
        "validation_split":      j.validation_split,
        "best_accuracy":         j.best_accuracy,
        "best_loss":             j.best_loss,
        "final_accuracy":        j.final_accuracy,
        "training_history":      th,
        # Architecture name — stored in extra_params at job creation time.
        # The frontend uses this as a fallback FOMO signal for older jobs
        # that predate the training_history.is_fomo flag.
        "architecture":          ep.get("architecture"),
        "model_size":            ep.get("size"),
        "fomo_version":          ep.get("fomo_version"),
        "launch_mode":           ep.get("launch_mode", "start"),
        # Authoritative kind of the run — independent of any `extra_params` key.
        # Persisted as a real column, defaults to "fresh" for historical rows.
        "run_kind":              j.run_kind or "fresh",
        # Early-stop convenience fields promoted to top level
        "requested_epochs":      th.get("requested_epochs", j.epochs),
        "actual_epochs":         th.get("actual_epochs"),
        "stopped_early":         th.get("stopped_early", False),
        "stop_reason":           th.get("stop_reason"),
        "device_type":           j.device_type or "cpu",
        "confusion_matrix":      j.confusion_matrix,
        "classification_report": j.classification_report,
        "error_message":         j.error_message,
        "started_at":            j.started_at.isoformat() if j.started_at else None,
        "completed_at":          j.completed_at.isoformat() if j.completed_at else None,
        "created_at":            j.created_at.isoformat(),
    }


def _resolve_training_architecture(impulse: Impulse, extra_params: Optional[dict] = None) -> str:
    ml_config = impulse.ml_blocks[0] if impulse.ml_blocks else {}
    params = extra_params or {}
    return (
        params.get("architecture")
        or ml_config.get("type")
        or ml_config.get("architecture")
        or "dense"
    )


def _ensure_supported_training_architecture(impulse: Impulse, extra_params: Optional[dict] = None) -> str:
    architecture = _resolve_training_architecture(impulse, extra_params)
    if architecture in _UNSUPPORTED_TRAINING_ARCHITECTURES:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Architecture '{architecture}' is recognized by the catalog but its "
                "training pipeline is not yet implemented. "
                "Supported architectures: dense, conv1d, conv2d, lstm, mobilenet, "
                "transfer, fomo_mobilenetv2_0_1, yolo_pro, mobilenet_v2_ssd_fpn_lite."
            ),
        )
    return architecture


def _assert_gpu_training_available() -> None:
    """Refuse a GPU training request that this deployment cannot actually serve.

    Only called for device_preference == "gpu". Raises 503 with a structured
    detail ({"code", "message"}) so the UI can tell the causes apart:

      gpu_disabled          — GPU routing is not permitted here at all.
      gpu_worker_unavailable — permitted, but nothing is consuming training_gpu.

    Never mutates device_preference. Silently downgrading to CPU gives the user
    a run they did not ask for, on hardware they did not choose.
    """
    if not settings.GPU_ENABLED:
        raise HTTPException(
            status_code=503,
            detail={
                "code": "gpu_disabled",
                "message": (
                    "GPU training is not enabled on this deployment. "
                    "Select CPU to train now, or ask an administrator to enable GPU."
                ),
            },
        )

    if not settings.GPU_REQUIRE_LIVE_WORKER:
        # Queue-as-buffer mode: enqueue against a possibly-absent GPU host and
        # let the lifecycle manager boot one (docs/gpu_lifecycle_architecture.md).
        return

    has_worker = gpu_queue_has_live_worker()

    if has_worker is False:
        raise HTTPException(
            status_code=503,
            detail={
                "code": "gpu_worker_unavailable",
                "message": (
                    "No GPU training worker is currently available. "
                    "Select CPU to train now, or try again once a GPU worker is online."
                ),
            },
        )

    # has_worker is None → the probe could not reach a verdict. Fail open: an
    # unreachable broker is not evidence that no GPU worker exists, and blocking
    # training on an inconclusive health check would turn a monitoring blip into
    # an outage. If the broker really is down, apply_async below fails and
    # returns its own 503.
    if has_worker is None:
        logger.warning(
            "[GPU] Liveness probe inconclusive — allowing GPU request through"
        )


# ─── Routes ───────────────────────────────────────────────────────────────────

@router.post("/start", status_code=201)
def start_training(
    req: TrainingRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Launch a Celery training job for the given impulse."""
    impulse = assert_impulse_owner(db, req.impulse_id, current_user)

    architecture = _ensure_supported_training_architecture(impulse, req.extra_params)
    extra_params = {
        **(req.extra_params or {}),
        "launch_mode": "start",
        # Training toggles — persisted so workers read them from job.extra_params.
        # Carried across retrains via the inherited extra_params (see retrain).
        "data_augmentation": bool(req.data_augmentation),
        "early_stopping": bool(req.early_stopping),
    }

    # A GPU request is honored or refused — never quietly turned into a CPU run.
    # CPU requests short-circuit here: no probe, no broker round-trip, no new
    # failure mode on the path that already worked.
    device_preference = req.device_preference
    if device_preference == "gpu":
        _assert_gpu_training_available()

    running = (
        db.query(TrainingJob)
        .filter(
            TrainingJob.impulse_id == req.impulse_id,
            TrainingJob.status == JobStatus.running,
        )
        .first()
    )
    if running:
        raise HTTPException(409, "A training job is already running for this impulse")

    job = TrainingJob(
        impulse_id=req.impulse_id,
        epochs=req.epochs,
        learning_rate=req.learning_rate,
        batch_size=req.batch_size,
        validation_split=req.validation_split,
        optimizer="adam",
        device_type=device_preference,
        extra_params=extra_params,
        run_kind="fresh",
    )
    db.add(job)
    db.commit()
    db.refresh(job)

    queue = "training_gpu" if device_preference == "gpu" else "training_cpu"
    try:
        task = run_training_job.apply_async(args=[job.id], queue=queue)
        job.celery_task_id = task.id
        db.commit()
    except Exception as e:
        db.delete(job)
        db.commit()
        raise HTTPException(
            status_code=503,
            detail=f"Failed to queue training job. Is the Celery worker and Redis running? Error: {str(e)}",
        )

    return _job_to_dict(job)


@router.post("/retrain", status_code=201)
def retrain(
    req: RetrainRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Retrain impulse using last job's hyperparams. Matches Edge Impulse
    'Retrain model' page — clones the last job config, overriding only
    what the user explicitly changes.
    """
    impulse = assert_impulse_owner(db, req.impulse_id, current_user)

    last_job = (
        db.query(TrainingJob)
        .filter(TrainingJob.impulse_id == req.impulse_id)
        .order_by(TrainingJob.created_at.desc())
        .first()
    )

    extra_params  = {**((last_job.extra_params or {}) if last_job else {}), "launch_mode": "retrain"}
    _ensure_supported_training_architecture(impulse, extra_params)

    running = (
        db.query(TrainingJob)
        .filter(
            TrainingJob.impulse_id == req.impulse_id,
            TrainingJob.status == JobStatus.running,
        )
        .first()
    )
    if running:
        raise HTTPException(409, "A training job is already running for this impulse")

    epochs        = req.epochs        or (last_job.epochs        if last_job else 100)
    learning_rate = req.learning_rate or (last_job.learning_rate if last_job else 0.001)
    batch_size    = req.batch_size    or (last_job.batch_size    if last_job else 32)
    device_type   = (last_job.device_type or "cpu") if last_job else "cpu"

    if device_type == "gpu" and not settings.GPU_ENABLED:
        device_type = "cpu"

    optimizer = "adam"
    val_split = last_job.validation_split if last_job else 0.2

    job = TrainingJob(
        impulse_id=req.impulse_id,
        epochs=epochs,
        learning_rate=learning_rate,
        batch_size=batch_size,
        validation_split=val_split,
        optimizer=optimizer,
        device_type=device_type,
        extra_params=extra_params,
        run_kind="retrain",
    )
    db.add(job)
    db.commit()
    db.refresh(job)

    queue = "training_gpu" if device_type == "gpu" else "training_cpu"
    try:
        task = run_training_job.apply_async(args=[job.id], queue=queue)
        job.celery_task_id = task.id
        db.commit()
    except Exception as e:
        db.delete(job)
        db.commit()
        raise HTTPException(
            status_code=503, 
            detail=f"Failed to queue training job. Is the Celery worker and Redis running? Error: {str(e)}"
        )

    return _job_to_dict(job)


@router.get("/impulse/{impulse_id}/status")
def get_impulse_training_status(
    impulse_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Quick status check for the Impulse Design page: has this impulse been
    trained? Returns the latest job + model summary to gate downstream pages.
    """
    impulse = assert_impulse_owner(db, impulse_id, current_user)

    latest_job = (
        db.query(TrainingJob)
        .filter(TrainingJob.impulse_id == impulse_id)
        .order_by(TrainingJob.created_at.desc())
        .first()
    )

    # The Active Model is whatever `impulse.active_model_run_id` points at.
    # It is set ONLY on successful completion and is NEVER cleared — so a
    # cancelled/failed/in-flight latest run does not erase the previous
    # pointer in the database. The UI decides whether to surface those
    # artifacts based on current_job.run_kind + status, not on this field.
    active_run_id = impulse.active_model_run_id
    active_job = (
        db.query(TrainingJob).filter(TrainingJob.id == active_run_id).first()
        if active_run_id
        else None
    )

    latest_model = None
    if active_job and active_job.status == JobStatus.completed:
        tflite = (
            db.query(TrainedModel)
            .filter(
                TrainedModel.training_job_id == active_job.id,
                TrainedModel.format == "tflite",
            )
            .first()
        )
        if tflite:
            meta = tflite.model_metadata or {}
            latest_model = {
                "id":           tflite.id,
                "format":       tflite.format,
                "version":      tflite.version,
                "label_names":  meta.get("label_names", []),
                "input_shape":  meta.get("input_shape", []),
                "architecture": meta.get("architecture", "dense"),
            }

    return {
        "impulse_id":          impulse_id,
        # has_trained_model and latest_job preserved for callers that already
        # rely on them — both reflect the same Active Model pointer now.
        "has_trained_model":   active_job is not None and active_job.status == JobStatus.completed,
        "latest_job":          _job_to_dict(latest_job) if latest_job else None,
        # Canonical names per the spec — current_job is "the latest run
        # regardless of status", active_model_run_id is the pointer.
        "current_job":         _job_to_dict(latest_job) if latest_job else None,
        "active_model_run_id": active_run_id,
        "latest_model":        latest_model,
        "best_accuracy":       active_job.best_accuracy if active_job else None,
    }


@router.get("/{job_id}")
def get_training_job(
    job_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Poll a training job. Also returns trained model IDs on completion."""
    job = assert_training_job_owner(db, job_id, current_user)

    result = _job_to_dict(job)
    if job.status == JobStatus.completed:
        models = (
            db.query(TrainedModel)
            .filter(TrainedModel.training_job_id == job_id)
            .all()
        )
        result["trained_models"] = [
            {"id": m.id, "format": m.format, "version": m.version}
            for m in models
        ]
    return result


@router.get("/impulse/{impulse_id}")
def list_jobs_for_impulse(
    impulse_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """List all training jobs for an impulse, newest first."""
    assert_impulse_owner(db, impulse_id, current_user)
    jobs = (
        db.query(TrainingJob)
        .filter(TrainingJob.impulse_id == impulse_id)
        .order_by(TrainingJob.created_at.desc())
        .all()
    )
    return [_job_to_dict(j) for j in jobs]


@router.post("/{job_id}/cancel")
def cancel_job(
    job_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Cancel a running or pending training job.

    For queued (pending) jobs, revoke() prevents the task from starting.
    For running jobs, revoke(terminate=True) is best-effort; the worker
    also polls the DB status and exits cooperatively when it sees 'cancelled'.
    """
    job = assert_training_job_owner(db, job_id, current_user)

    if job.status not in (JobStatus.pending, JobStatus.running):
        raise HTTPException(400, f"Job is already {job.status} — cannot cancel")

    # Write the cancellation signal into the DB first so the worker's
    # cooperative poll sees it even if revoke() has no effect.
    job.status = JobStatus.cancelled
    job.error_message = "Cancelled by user"
    job.completed_at = datetime.utcnow()
    db.commit()

    # Best-effort Celery revoke — works reliably for queued tasks;
    # for running tasks the cooperative DB check in the worker is the
    # primary stop mechanism.
    if job.celery_task_id:
        from app.workers.celery_app import celery_app
        try:
            celery_app.control.revoke(job.celery_task_id, terminate=True, signal="SIGTERM")
        except Exception:
            pass  # revoke is best-effort; worker DB poll handles the rest

    return {"message": "Job cancelled", "job_id": job_id}
