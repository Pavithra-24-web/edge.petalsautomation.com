"""
Post-processing pipeline endpoints — Phases 2, 3, 5 & 6.

New table-based settings and job-lifecycle endpoints, scoped per project.
These are additive; the legacy model_metadata-based endpoints in
post_processing.py are not touched.

Routes registered under /api/v1/projects/{project_id}/:
  GET  /post-processing-settings
  PUT  /post-processing-settings
  GET  /post-processing-jobs/{job_id}
  POST /post-processing-preview              (Phase 3)
  POST /post-processing-video                (Phase 6 — inference-backed upload)
  POST /post-processing-video-debug          (Phase 6 — debug: caller-supplied detections)
  GET  /post-processing-jobs/{job_id}/video  (Phase 5 — download URL)
  POST /post-processing-jobs/{job_id}/cancel (cancel a pending/processing job)
"""
import json
import logging
import os
from typing import Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.core.auth import get_current_user
from app.core.authz import require_admin
from app.core.storage import storage, StorageService
from app.models.user import ProcessingJob, Sample, User
from pydantic import ValidationError

from app.schemas.post_processing_pipeline import (
    DetectionOut,
    PostProcessingSettingsResponse,
    PostProcessingSettingsUpdate,
    PreviewRequest,
    PreviewResponse,
    ProcessingJobResponse,
    RawDetectionInput,
    TriggerFromSampleRequest,
)
from app.services import post_processing_pipeline as svc
from app.services.post_processing import run_pipeline
from app.services.post_processing.detection_postprocess import PipelineConfig
from app.services.post_processing.types import Detection

logger = logging.getLogger(__name__)

_ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov"}

router = APIRouter()

# ─── Defaults returned when no settings row exists yet ───────────────────────

_DEFAULTS = {
    "enabled": True,
    "threshold": 0.5,
    "tracking_enabled": False,
    "keep_grace": 3,
    "max_observations": 5,
    "class_filter": [],
}


# ─── Settings ─────────────────────────────────────────────────────────────────

@router.get("/{project_id}/post-processing-settings", response_model=PostProcessingSettingsResponse)
def get_post_processing_settings(
    project_id: str,
    impulse_id: Optional[str] = Query(None),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Return saved post-processing settings, or schema defaults if none saved yet."""
    try:
        svc.get_project_or_404(db, project_id, current_user.id)
    except KeyError:
        raise HTTPException(404, "Project not found")
    if impulse_id:
        try:
            svc.get_impulse_for_project_or_404(db, project_id, impulse_id)
        except KeyError:
            raise HTTPException(404, "Impulse not found")

    row = svc.get_settings(db, project_id, impulse_id=impulse_id)
    if row is not None:
        return PostProcessingSettingsResponse.model_validate(row)

    return PostProcessingSettingsResponse(
        project_id=project_id,
        impulse_id=impulse_id,
        **_DEFAULTS,
    )


@router.put("/{project_id}/post-processing-settings", response_model=PostProcessingSettingsResponse)
def update_post_processing_settings(
    project_id: str,
    body: PostProcessingSettingsUpdate,
    impulse_id: Optional[str] = Query(None),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Create or update post-processing settings for a project."""
    try:
        svc.get_project_or_404(db, project_id, current_user.id)
    except KeyError:
        raise HTTPException(404, "Project not found")
    if impulse_id:
        try:
            svc.get_impulse_for_project_or_404(db, project_id, impulse_id)
        except KeyError:
            raise HTTPException(404, "Impulse not found")

    logger.info(
        "PUT post-processing-settings: project=%s impulse=%s payload=%s",
        project_id, impulse_id, body.model_dump(exclude_none=True),
    )
    row = svc.upsert_settings(db, project_id, body, impulse_id=impulse_id)
    return PostProcessingSettingsResponse.model_validate(row)


# ─── Jobs ─────────────────────────────────────────────────────────────────────

@router.get(
    "/{project_id}/post-processing-jobs/{job_id}",
    response_model=ProcessingJobResponse,
)
def get_processing_job(
    project_id: str,
    job_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Return a single processing job, enforcing project ownership."""
    try:
        svc.get_project_or_404(db, project_id, current_user.id)
    except KeyError:
        raise HTTPException(404, "Project not found")

    try:
        job = svc.get_job(db, project_id, job_id)
    except KeyError:
        raise HTTPException(404, "Job not found")

    return ProcessingJobResponse.model_validate(job)


# ─── Preview (Phase 3) ────────────────────────────────────────────────────────

@router.post("/{project_id}/post-processing-preview", response_model=PreviewResponse)
def preview_post_processing(
    project_id: str,
    body: PreviewRequest,
    impulse_id: Optional[str] = Query(None),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Run the full detection post-processing pipeline on a single frame of raw
    detections and return a stage-by-stage breakdown for debugging.

    Settings are loaded from the project's saved row (or schema defaults if
    none exists yet).  No trained model is required — the caller supplies the
    raw detections directly.

    Pass back tracker_state from the response as tracker_state in the next
    request to maintain object ID continuity across frames.
    """
    try:
        svc.get_project_or_404(db, project_id, current_user.id)
    except KeyError:
        raise HTTPException(404, "Project not found")
    if impulse_id:
        try:
            svc.get_impulse_for_project_or_404(db, project_id, impulse_id)
        except KeyError:
            raise HTTPException(404, "Impulse not found")

    # Load settings or fall back to defaults
    row = svc.get_settings(db, project_id, impulse_id=impulse_id)
    if row is not None:
        config = PipelineConfig.from_orm_or_dict(row)
    else:
        config = PipelineConfig.from_orm_or_dict(_DEFAULTS)

    # Convert request detections to internal type
    internal_dets = [
        Detection(
            x1=d.x1, y1=d.y1, x2=d.x2, y2=d.y2,
            class_name=d.class_name, confidence=d.confidence,
        )
        for d in body.detections
    ]

    result = run_pipeline(internal_dets, config, tracker_state=body.tracker_state)

    def _to_out(dets):
        return [
            DetectionOut(
                x1=d.x1, y1=d.y1, x2=d.x2, y2=d.y2,
                class_name=d.class_name, confidence=d.confidence,
                object_id=d.object_id,
                observations=d.observations,
            )
            for d in dets
        ]

    return PreviewResponse(
        input_detections=_to_out(result.input_detections),
        filtered_detections=_to_out(result.filtered_detections),
        post_nms_detections=_to_out(result.post_nms_detections),
        final_detections=_to_out(result.final_detections),
        tracker_state=result.tracker_state,
    )


# ─── Shared upload helper ─────────────────────────────────────────────────────

def _create_job_and_enqueue(
    project_id: str,
    file: UploadFile,
    content: bytes,
    db: Session,
    impulse_id: Optional[str] = None,
) -> ProcessingJob:
    """
    Create a ProcessingJob row, upload the input video to S3, enqueue the
    Celery task.  Returns the flushed + committed job row.

    impulse_id, when supplied, is forwarded to the worker as a kwarg so the
    worker scopes model selection to that specific impulse rather than
    searching across all impulses in the project.

    Raises HTTPException(503) if Celery is unavailable, cleaning up S3 first.
    """
    job = ProcessingJob(
        project_id=project_id,
        status="pending",
        input_video_path="pending",
    )
    db.add(job)
    db.flush()

    input_key = StorageService.video_input_key(project_id, job.id, file.filename or "input.mp4")
    storage.upload_bytes(content, input_key, content_type=file.content_type or "video/mp4")
    job.input_video_path = input_key

    db.commit()
    db.refresh(job)

    from app.workers.post_processing_worker import run_video_job  # local to avoid circular
    try:
        result = run_video_job.apply_async(
            args=[job.id],
            kwargs={"impulse_id": impulse_id},
            queue="post_processing",
        )
        job.celery_task_id = result.id
        db.commit()
    except Exception as exc:
        try:
            storage.delete_file(input_key)
        except Exception:
            pass
        db.delete(job)
        db.commit()
        raise HTTPException(
            status_code=503,
            detail=f"Failed to queue processing job. Is the Celery worker running? ({exc})",
        )

    return job


# ─── Video upload — inference-backed (Phase 6) ───────────────────────────────

@router.post("/{project_id}/post-processing-video", status_code=201, response_model=ProcessingJobResponse)
async def upload_post_processing_video(
    project_id: str,
    file: UploadFile = File(...),
    impulse_id: Optional[str] = Form(None),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Upload a video for post-processing preview.

    The Celery worker loads the best available trained object-detection model
    and runs per-frame inference.  If impulse_id is supplied the worker scopes
    model selection to that impulse only; otherwise it searches across all
    impulses in the project (useful when a project has exactly one impulse).

    If no trained detection model is found the job is marked failed with a
    clear error_message.

    Returns the created ProcessingJob row (status: pending).
    """
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in _ALLOWED_VIDEO_EXTENSIONS:
        raise HTTPException(
            status_code=422,
            detail=f"Unsupported video format '{ext}'. Allowed: mp4, avi, mov.",
        )

    try:
        svc.get_project_or_404(db, project_id, current_user.id)
    except KeyError:
        raise HTTPException(404, "Project not found")

    content = await file.read()
    job = _create_job_and_enqueue(project_id, file, content, db, impulse_id=impulse_id)
    return ProcessingJobResponse.model_validate(job)


# ─── Video upload — debug (caller-supplied detections) ───────────────────────

@router.post("/{project_id}/post-processing-video-debug", status_code=201, response_model=ProcessingJobResponse)
async def upload_post_processing_video_debug(
    project_id: str,
    file: UploadFile = File(...),
    detections_json: str = Form(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(require_admin),
):
    """
    Debug endpoint: upload a video together with pre-computed per-frame
    detections.  Use this when you want to test the post-processing pipeline
    with a known fixed detection set, bypassing model inference.

    detections_json must be a JSON array-of-arrays (one inner array per frame),
    each element matching {class_name, confidence, x1, y1, x2, y2}.

    The worker will use these detections instead of running inference.
    Use the main /post-processing-video endpoint for normal production flow.
    """
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in _ALLOWED_VIDEO_EXTENSIONS:
        raise HTTPException(
            status_code=422,
            detail=f"Unsupported video format '{ext}'. Allowed: mp4, avi, mov.",
        )

    # Validate detections_json
    try:
        parsed = json.loads(detections_json)
        if not isinstance(parsed, list):
            raise ValueError("top-level value must be an array of frame arrays")
        for frame_idx, frame in enumerate(parsed):
            if not isinstance(frame, list):
                raise ValueError(
                    f"frame {frame_idx} is not an array "
                    f"(expected [[...frame0], [...frame1], ...])"
                )
            for det_idx, det in enumerate(frame):
                if not isinstance(det, dict):
                    raise ValueError(f"frame {frame_idx}[{det_idx}] is not an object")
                try:
                    RawDetectionInput.model_validate(det)
                except ValidationError as ve:
                    first = ve.errors()[0]
                    field = ".".join(str(p) for p in first["loc"]) if first["loc"] else "value"
                    raise ValueError(
                        f"frame {frame_idx}[{det_idx}].{field}: {first['msg']}"
                    ) from ve
    except (json.JSONDecodeError, ValueError) as exc:
        raise HTTPException(
            status_code=422,
            detail=f"detections_json validation failed: {exc}",
        )

    try:
        svc.get_project_or_404(db, project_id, current_user.id)
    except KeyError:
        raise HTTPException(404, "Project not found")

    content = await file.read()

    job = ProcessingJob(
        project_id=project_id,
        status="pending",
        input_video_path="pending",
    )
    db.add(job)
    db.flush()

    input_key = StorageService.video_input_key(project_id, job.id, file.filename or "input.mp4")
    storage.upload_bytes(content, input_key, content_type=file.content_type or "video/mp4")
    job.input_video_path = input_key

    det_key = StorageService.video_detections_key(project_id, job.id)
    storage.upload_bytes(detections_json.encode(), det_key, content_type="application/json")

    db.commit()
    db.refresh(job)

    from app.workers.post_processing_worker import run_video_job
    try:
        result = run_video_job.apply_async(args=[job.id], queue="post_processing")
        job.celery_task_id = result.id
        db.commit()
    except Exception as exc:
        for key in (input_key, det_key):
            try:
                storage.delete_file(key)
            except Exception:
                pass
        db.delete(job)
        db.commit()
        raise HTTPException(
            status_code=503,
            detail=f"Failed to queue processing job. Is the Celery worker running? ({exc})",
        )

    return ProcessingJobResponse.model_validate(job)


# ─── Trigger from existing sample (Phase 7) ──────────────────────────────────

@router.post(
    "/{project_id}/post-processing-video-from-sample",
    status_code=201,
    response_model=ProcessingJobResponse,
)
def trigger_post_processing_from_sample(
    project_id: str,
    body: TriggerFromSampleRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Trigger post-processing on a video that already exists in the project's
    Data Acquisition store.  No re-upload: the worker reads the file directly
    from the sample's storage key.

    Returns the created ProcessingJob row (status: pending).
    """
    try:
        svc.get_project_or_404(db, project_id, current_user.id)
    except KeyError:
        raise HTTPException(404, "Project not found")

    sample = (
        db.query(Sample)
        .filter(Sample.id == body.sample_id, Sample.project_id == project_id)
        .first()
    )
    if sample is None:
        raise HTTPException(404, "Sample not found in this project")

    ext = os.path.splitext(sample.filename or "")[1].lower()
    if ext not in _ALLOWED_VIDEO_EXTENSIONS:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Sample '{sample.filename}' is not a supported video format. "
                "Allowed: mp4, avi, mov."
            ),
        )

    job = ProcessingJob(
        project_id=project_id,
        status="pending",
        input_video_path=sample.storage_key,
    )
    db.add(job)
    db.commit()
    db.refresh(job)

    from app.workers.post_processing_worker import run_video_job  # local to avoid circular
    try:
        result = run_video_job.apply_async(
            args=[job.id],
            kwargs={"impulse_id": body.impulse_id},
            queue="post_processing",
        )
        job.celery_task_id = result.id
        db.commit()
    except Exception as exc:
        db.delete(job)
        db.commit()
        raise HTTPException(
            status_code=503,
            detail=f"Failed to queue processing job. Is the Celery worker running? ({exc})",
        )

    return ProcessingJobResponse.model_validate(job)


# ─── Cancel ───────────────────────────────────────────────────────────────────

@router.post(
    "/{project_id}/post-processing-jobs/{job_id}/cancel",
    response_model=ProcessingJobResponse,
)
def cancel_processing_job(
    project_id: str,
    job_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Cancel a pending or in-progress video post-processing job.

    Writes the cancellation into the DB first so the worker's cooperative
    poll (see post_processing_worker._make_cancel_check) sees it even on
    platforms where Celery's terminate signal can't reliably kill the running
    task (e.g. the --pool=threads worker used on Windows dev). The best-effort
    revoke() below still helps for jobs that haven't started yet.
    """
    try:
        svc.get_project_or_404(db, project_id, current_user.id)
    except KeyError:
        raise HTTPException(404, "Project not found")

    try:
        job = svc.get_job(db, project_id, job_id)
    except KeyError:
        raise HTTPException(404, "Job not found")

    if job.status not in ("pending", "processing"):
        raise HTTPException(400, f"Job is already {job.status} — cannot cancel")

    job.status = "cancelled"
    job.error_message = "Cancelled by user"
    db.commit()
    db.refresh(job)

    if job.celery_task_id:
        from app.workers.celery_app import celery_app
        try:
            # No terminate=True: the threads pool used on Windows dev doesn't
            # implement kill_job, so terminate would only log a worker-side
            # NotImplementedError with no benefit — the cooperative DB poll
            # in post_processing_worker._make_cancel_check already stops the
            # running task. Plain revoke() still drops the task if it's
            # merely queued and hasn't started yet.
            celery_app.control.revoke(job.celery_task_id)
        except Exception:
            pass  # revoke is best-effort; worker DB poll handles the rest

    return ProcessingJobResponse.model_validate(job)


# ─── Video download URL ───────────────────────────────────────────────────────

@router.get("/{project_id}/post-processing-jobs/{job_id}/video")
def get_job_video_url(
    project_id: str,
    job_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Return a short-lived presigned URL for the processed output video.

    Returns 404 when:
    - the project does not exist or is not owned by the caller
    - the job does not belong to the project
    - the job is not yet complete or has no output
    """
    try:
        svc.get_project_or_404(db, project_id, current_user.id)
    except KeyError:
        raise HTTPException(404, "Project not found")

    try:
        job = svc.get_job(db, project_id, job_id)
    except KeyError:
        raise HTTPException(404, "Job not found")

    if job.status != "complete" or not job.output_video_path:
        raise HTTPException(404, "Output video not available — job is not complete")

    url = storage.get_presigned_url(job.output_video_path, expires_in=3600)
    return {"url": url}
