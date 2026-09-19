"""
Synthetic Data endpoints — generate labelled samples with OpenAI Image
Generation (Phase 3 of docs/Action/datasynthetic_implementationplan.md).

Every image produced here goes through `create_sample_from_bytes`
(app/services/sample_ingest.py) — the same helper `POST /samples/upload`
calls — so a generated sample is indistinguishable, to every downstream
consumer, from an uploaded one (see the plan's §1.0). Provenance lives
entirely in `Sample.extra_metadata["synthetic"]`; no new column, no second
write path.

The worker thread mirrors `_run_ai_labeling_job` in ai_labeling.py: a daemon
`threading.Thread` with its own SQLAlchemy engine/session, because this is
I/O-bound network work with no ML dependency and no reason to route through
the Celery/GPU training queue (plan §2.5).
"""
from __future__ import annotations

import logging
import re
import threading
from datetime import datetime, timedelta
from typing import Optional

from botocore.exceptions import BotoCoreError, ClientError
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import func
from sqlalchemy.orm import Session
from sqlalchemy.orm.attributes import flag_modified

from app.core.auth import get_current_user
from app.core.authz import assert_project_owner, assert_synthetic_job_owner
from app.core.config import settings
from app.core.database import get_db
from app.models.user import Project, SyntheticDataJob, SyntheticJobStatus, User
from app.schemas.synthetic_data import (
    SyntheticDataConfig,
    SyntheticDataGenerateRequest,
    SyntheticDataJobOut,
)
from app.services import openai_images
from app.services.openai_images import OpenAIImageError
from app.services.sample_ingest import _get_or_create_label, create_sample_from_bytes

logger = logging.getLogger(__name__)
router = APIRouter()


# ─── Fixed enums (plan §2.4 / §5.3 field inventory) ───────────────────────────

_ALLOWED_SAMPLE_TYPES = {"training", "testing", "postprocessing", "automatic"}
_ALLOWED_SIZES = set(openai_images.UI_SIZE_TO_API.keys())

# Cheap syntactic sanity check for the user-supplied API key (plan §2.4) —
# rejects obvious garbage (embedded control/newline characters) before any
# network call. Not a format/prefix check: OpenAI key shapes are not a
# stable contract to validate against client-side.
_CONTROL_CHAR_RE = re.compile(r"[\x00-\x1f\x7f]")

# Serial, one image per request, one job per project (plan §4.5). Kept as a
# job-level parameter rather than a constant so it is one-line changeable
# without a schema migration.
_DEFAULT_N_PER_REQUEST = 1


def _job_to_dict(job: SyntheticDataJob) -> dict:
    return SyntheticDataJobOut.model_validate(
        {
            "id": job.id,
            "project_id": job.project_id,
            "status": job.status.value if job.status else "pending",
            "provider": job.provider,
            "model": job.model,
            "prompt": job.prompt,
            "label_name": job.label_name,
            "label_id": job.label_id,
            "requested_count": job.requested_count,
            "generated_count": job.generated_count or 0,
            "failed_count": job.failed_count or 0,
            "sample_type": job.sample_type,
            "parameters": job.parameters or {},
            "sample_ids": job.sample_ids or [],
            "estimated_cost_usd": job.estimated_cost_usd,
            "error_message": job.error_message,
            "started_at": job.started_at,
            "completed_at": job.completed_at,
            "created_at": job.created_at,
        }
    ).model_dump(mode="json")


def _slugify(name: str) -> str:
    """Turn a label into the folder segment used by the filename convention
    (plan §2.3): lowercase, non-alnum runs collapsed to one underscore, no
    leading/trailing underscore. Never empty — falls back to "label"."""
    slug = re.sub(r"[^a-z0-9]+", "_", name.strip().lower()).strip("_")
    return slug or "label"


# ─── GET /synthetic-data/config ───────────────────────────────────────────────

@router.get("/config", response_model=SyntheticDataConfig)
def get_config(current_user: User = Depends(get_current_user)):
    """What the form needs to render — sizes, qualities, backgrounds, model,
    cap, price table (plan §2.4). None of this depends on any particular
    user's key, so the response carries no authentication-state field at
    all — the frontend gates the Generate button on the form's own API Key
    field, never on this endpoint."""
    price_table = {
        f"{size}:{ui_quality}": openai_images.PRICE_TABLE_USD[(size, api_quality)]
        for size in _ALLOWED_SIZES
        for ui_quality, api_quality in openai_images.UI_QUALITY_TO_API.items()
        if (size, api_quality) in openai_images.PRICE_TABLE_USD
    }
    return SyntheticDataConfig(
        model=settings.OPENAI_IMAGE_MODEL,
        max_images_per_job=settings.SYNTHETIC_MAX_IMAGES_PER_JOB,
        sizes=sorted(_ALLOWED_SIZES),
        qualities=["standard", "high"],
        backgrounds=["transparent", "opaque", "auto"],
        price_per_image_usd=price_table,
    )


# ─── POST /synthetic-data/generate ────────────────────────────────────────────

@router.post("/generate", status_code=202, response_model=SyntheticDataJobOut)
def generate(
    body: SyntheticDataGenerateRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    project = assert_project_owner(db, body.project_id, current_user)

    # Project-type guard — generated images are useless to a motion project
    # and its Dataset page never shows the entry point (plan §2.4, §5.1).
    if project.project_type == "motion":
        raise HTTPException(404, "Project not found")

    # User-provided OpenAI key (plan §0/§2.4) — required, trimmed, non-empty,
    # and a cheap syntactic sanity check (no control characters) that rejects
    # obvious garbage before any network call. This is not proof the key is
    # valid; the pre-flight call below is.
    api_key = (body.api_key or "").strip()
    if not api_key or _CONTROL_CHAR_RE.search(api_key):
        raise HTTPException(400, "An OpenAI API key is required to generate images.")

    prompt = (body.prompt or "").strip()
    if not prompt or len(prompt) > 4000:
        raise HTTPException(400, "prompt is required and must be 1-4000 characters")

    label = (body.label or "").strip()
    if not label or len(label) > 128:
        raise HTTPException(400, "label is required and must be 1-128 characters")
    if label.startswith(("{", "[")):
        raise HTTPException(400, "label must not look like a serialized object")

    if not isinstance(body.count, int) or not (1 <= body.count <= settings.SYNTHETIC_MAX_IMAGES_PER_JOB):
        raise HTTPException(
            400, f"count must be an integer between 1 and {settings.SYNTHETIC_MAX_IMAGES_PER_JOB}"
        )

    if body.sample_type not in _ALLOWED_SAMPLE_TYPES:
        raise HTTPException(400, f"sample_type must be one of {sorted(_ALLOWED_SAMPLE_TYPES)}")

    try:
        api_size, api_quality, api_background = openai_images.map_ui_params(
            body.size, body.quality, body.background
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc))

    # ── Guards ─────────────────────────────────────────────────────────────
    # Pre-flight key validation (plan §0/§2.4/§4.1) — fails fast, before any
    # job row or thread exists, so a bad key is a synchronous 401/403 on this
    # request rather than a job that starts and then silently fails.
    try:
        openai_images.validate_api_key(api_key=api_key)
    except OpenAIImageError as err:
        if err.status == 401:
            raise HTTPException(401, err.user_message)
        if err.status == 403:
            raise HTTPException(403, err.user_message)
        raise HTTPException(502, err.user_message or "Could not verify the OpenAI API key.")

    already_running = (
        db.query(SyntheticDataJob)
        .filter(
            SyntheticDataJob.project_id == project.id,
            SyntheticDataJob.status == SyntheticJobStatus.running,
        )
        .first()
    )
    if already_running:
        raise HTTPException(409, "A synthetic data generation job is already running for this project")

    day_ago = datetime.utcnow() - timedelta(hours=24)
    daily_count, oldest_created_at = (
        db.query(func.count(SyntheticDataJob.id), func.min(SyntheticDataJob.created_at))
        .join(Project, SyntheticDataJob.project_id == Project.id)
        .filter(Project.owner_id == current_user.id, SyntheticDataJob.created_at >= day_ago)
        .first()
    )
    if (daily_count or 0) >= settings.SYNTHETIC_MAX_JOBS_PER_USER_PER_DAY:
        reset_at = (oldest_created_at or datetime.utcnow()) + timedelta(hours=24)
        raise HTTPException(
            429,
            f"Daily synthetic data job limit reached "
            f"({settings.SYNTHETIC_MAX_JOBS_PER_USER_PER_DAY}/day). "
            f"Try again after {reset_at.isoformat()}.",
        )

    label_id = _get_or_create_label(label, project.id, db)

    estimated_cost = openai_images.estimate_cost_usd(api_size, api_quality, body.count)

    job = SyntheticDataJob(
        project_id=project.id,
        status=SyntheticJobStatus.running,
        provider="openai",
        model=settings.OPENAI_IMAGE_MODEL,
        prompt=prompt,
        label_name=label,
        label_id=label_id,
        requested_count=body.count,
        generated_count=0,
        failed_count=0,
        sample_type=body.sample_type,
        parameters={
            "size": api_size,
            "quality": api_quality,
            "background": api_background,
            "output_format": "png",
            "n_per_request": _DEFAULT_N_PER_REQUEST,
        },
        sample_ids=[],
        estimated_cost_usd=estimated_cost,
        started_at=datetime.utcnow(),
    )
    db.add(job)
    db.commit()
    db.refresh(job)

    # `api_key` is passed as a plain function argument only — never written
    # to `job.parameters` or any other persisted field (plan §0/§2.5). It
    # lives in this handler's locals and the thread's argument tuple only.
    thread = threading.Thread(
        target=_run_synthetic_job,
        args=(job.id, api_key, settings.DATABASE_URL),
        daemon=True,
    )
    thread.start()

    return _job_to_dict(job)


# ─── GET /synthetic-data/jobs/{job_id} ────────────────────────────────────────

@router.get("/jobs/{job_id}", response_model=SyntheticDataJobOut)
def get_job(
    job_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    job = assert_synthetic_job_owner(db, job_id, current_user)

    # Stale-job recovery (plan §2.6): a job left `running` by an API restart
    # would spin the frontend's poller forever. No reaper process — the read
    # path is where this gets noticed and fixed.
    if job.status == SyntheticJobStatus.running and job.started_at is not None:
        age = (datetime.utcnow() - job.started_at).total_seconds()
        if age > settings.SYNTHETIC_JOB_STALE_SECONDS:
            job.status = SyntheticJobStatus.failed
            job.error_message = "Generation was interrupted (server restarted)"
            job.completed_at = datetime.utcnow()
            db.commit()
            db.refresh(job)

    return _job_to_dict(job)


# ─── GET /synthetic-data/jobs/project/{project_id} ────────────────────────────

@router.get("/jobs/project/{project_id}", response_model=list[SyntheticDataJobOut])
def list_jobs(
    project_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    assert_project_owner(db, project_id, current_user)
    jobs = (
        db.query(SyntheticDataJob)
        .filter(SyntheticDataJob.project_id == project_id)
        .order_by(SyntheticDataJob.created_at.desc())
        .all()
    )
    return [_job_to_dict(j) for j in jobs]


# ─── Worker ───────────────────────────────────────────────────────────────────

def _should_fail_whole_job(err: OpenAIImageError) -> bool:
    """True when this error will fail identically for every remaining image
    in the job (auth, org verification, a bad parameter) — retrying it is
    both pointless and, if the request was partially billed, wasteful of
    real money (plan §4.1). Moderation rejections and exhausted-retry
    429/5xx/timeouts are per-image and must not discard images that already
    succeeded."""
    if err.is_moderation:
        return False
    if err.retryable:
        # generate_images() already retried this request up to its internal
        # attempt cap before raising — by the time it reaches here, this
        # specific request has exhausted its retries, but a *later* request
        # (e.g. after the rate limit window rolls over) might still succeed.
        return False
    return True


def _fail_job(session: Session, job: SyntheticDataJob, message: str) -> None:
    job.status = SyntheticJobStatus.failed
    job.error_message = message
    job.completed_at = datetime.utcnow()
    session.commit()


def _run_synthetic_job(job_id: str, api_key: str, db_url: str) -> None:
    """Background worker entry point. `api_key` is a plain positional
    argument passed once at `threading.Thread(...)` construction (plan
    §0/§2.5) — never a lookup this function performs against settings or the
    database. Opens its own DB session since this runs in a background
    thread (mirrors `_run_ai_labeling_job` in ai_labeling.py), then delegates
    to `_run_synthetic_job_with_session` — split out so tests can drive the
    actual generation loop against an already-open test session instead of a
    second, real DB connection."""
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    eng = create_engine(db_url, pool_pre_ping=True)
    Session = sessionmaker(bind=eng)
    session = Session()
    try:
        _run_synthetic_job_with_session(session, job_id, api_key)
    finally:
        session.close()


def _run_synthetic_job_with_session(session: Session, job_id: str, api_key: str) -> None:
    """`api_key` lives only in this function's local scope (plan §0/§2.5):
    used for every `openai_images.generate_images(api_key=...)` call this
    job makes, never written to `job.parameters` or any other attribute of
    the `job` ORM object, and discarded when this function returns."""
    try:
        job = session.query(SyntheticDataJob).filter(SyntheticDataJob.id == job_id).first()
        if not job:
            return

        project = session.query(Project).filter(Project.id == job.project_id).first()
        if not project:
            _fail_job(session, job, "Project no longer exists")
            return

        params = job.parameters or {}
        api_size = params.get("size", "1024x1024")
        api_quality = params.get("quality", "medium")
        api_background = params.get("background", "auto")
        n_per_request = max(1, int(params.get("n_per_request", _DEFAULT_N_PER_REQUEST)))

        label_slug = _slugify(job.label_name)
        job_short = job.id.replace("-", "")[:6]

        image_index = 0
        remaining = job.requested_count
        last_error_message: Optional[str] = None

        while remaining > 0:
            batch_n = min(n_per_request, remaining)
            try:
                images = openai_images.generate_images(
                    api_key=api_key,
                    prompt=job.prompt,
                    count=batch_n,
                    size=api_size,
                    quality=api_quality,
                    background=api_background,
                    model=job.model,
                )
            except OpenAIImageError as err:
                if _should_fail_whole_job(err):
                    _fail_job(session, job, err.user_message)
                    return
                last_error_message = err.user_message
                job.failed_count = (job.failed_count or 0) + batch_n
                session.commit()
                remaining -= batch_n
                continue

            for img_bytes in images:
                image_index += 1
                try:
                    sample = create_sample_from_bytes(
                        session,
                        project_id=job.project_id,
                        filename=f"synthetic/{label_slug}/{job_short}_{image_index:03d}.png",
                        content=img_bytes,
                        content_type="image/png",
                        sample_type=job.sample_type,
                        label_id=job.label_id,
                        extra_metadata={
                            "synthetic": {
                                "provider": job.provider,
                                "model": job.model,
                                "prompt": job.prompt,
                                "label": job.label_name,
                                "generated_at": datetime.utcnow().isoformat() + "Z",
                                "job_id": job.id,
                                "image_index": image_index,
                                "parameters": {
                                    "size": api_size,
                                    "quality": api_quality,
                                    "background": api_background,
                                    "output_format": "png",
                                    "requested_count": job.requested_count,
                                },
                                "revised_prompt": None,
                            }
                        },
                        uploaded_by=project.owner_id,
                    )
                except (ClientError, BotoCoreError) as exc:
                    logger.warning("synthetic job %s: storage upload failed: %s", job_id, exc)
                    job.failed_count = (job.failed_count or 0) + 1
                    session.commit()
                    continue

                job.sample_ids = (job.sample_ids or []) + [sample.id]
                flag_modified(job, "sample_ids")
                job.generated_count = (job.generated_count or 0) + 1
                session.commit()

            remaining -= batch_n

        # A job that never produced a single sample is a failure, not a
        # success, even when every individual request failed for a
        # per-image reason (moderation) or an exhausted-retry transient one
        # (429/5xx/timeout) rather than a whole-job one (auth/permissions) —
        # `_should_fail_whole_job` deliberately lets those keep trying later
        # batches, but if none of them ever produced a sample there is
        # nothing to call "completed".
        if (job.generated_count or 0) == 0:
            _fail_job(session, job, last_error_message or "No images could be generated.")
            return

        job.status = SyntheticJobStatus.completed
        job.completed_at = datetime.utcnow()
        session.commit()

    except Exception as exc:  # noqa: BLE001 - mirrors _run_ai_labeling_job's envelope
        logger.error("Synthetic data job %s failed: %s", job_id, exc)
        try:
            job = session.query(SyntheticDataJob).filter(SyntheticDataJob.id == job_id).first()
            if job:
                job.status = SyntheticJobStatus.failed
                job.error_message = str(exc)
                job.completed_at = datetime.utcnow()
                session.commit()
        except Exception:
            pass
    finally:
        # `api_key` goes out of scope when this function returns — nothing
        # else in the process holds a reference to it (plan §0/§2.5). This
        # `del` is a documented no-op-in-effect reminder of that invariant,
        # not a load-bearing security control by itself.
        del api_key
