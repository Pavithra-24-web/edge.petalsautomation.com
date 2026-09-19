"""
Post-processing pipeline service — Phase 2.

All DB mutations for the new post_processing_settings / processing_jobs
tables live here.  Endpoint functions stay thin; this logic can be
tested without FastAPI's request/response cycle.

Public surface:
    get_project_or_404(db, project_id, user_id)   → Project  (raises KeyError)
    get_settings(db, project_id, impulse_id)      → PostProcessingSettings | None
    upsert_settings(db, project_id, update, impulse_id) → PostProcessingSettings
    get_job(db, project_id, job_id)               → ProcessingJob  (raises KeyError)
"""
from __future__ import annotations

import logging

from sqlalchemy.orm import Session
from sqlalchemy.orm.attributes import flag_modified

from app.models.user import Impulse, Project, PostProcessingSettings, ProcessingJob
from app.schemas.post_processing_pipeline import PostProcessingSettingsUpdate

logger = logging.getLogger(__name__)


# ─── Project guard ────────────────────────────────────────────────────────────

def get_project_or_404(db: Session, project_id: str, user_id: str) -> Project:
    """
    Return the project if it exists and is owned by user_id.
    Raises KeyError so endpoints can convert to a consistent 404.
    """
    project = (
        db.query(Project)
        .filter(Project.id == project_id, Project.owner_id == user_id)
        .first()
    )
    if not project:
        raise KeyError(f"Project {project_id!r} not found")
    return project


def get_impulse_for_project_or_404(
    db: Session,
    project_id: str,
    impulse_id: str,
) -> Impulse:
    """Return the impulse if it belongs to the given project."""
    impulse = (
        db.query(Impulse)
        .filter(Impulse.id == impulse_id, Impulse.project_id == project_id)
        .first()
    )
    if not impulse:
        raise KeyError(
            f"Impulse {impulse_id!r} not found in project {project_id!r}"
        )
    return impulse


# ─── Settings CRUD ────────────────────────────────────────────────────────────

def get_settings(
    db: Session,
    project_id: str,
    impulse_id: str | None = None,
) -> PostProcessingSettings | None:
    """Return the saved settings row for a project/impulse scope.

    When impulse_id is provided, prefer an impulse-specific row and fall back
    to the legacy project-wide row (impulse_id IS NULL) if one exists.
    """
    if impulse_id:
        scoped = (
            db.query(PostProcessingSettings)
            .filter(
                PostProcessingSettings.project_id == project_id,
                PostProcessingSettings.impulse_id == impulse_id,
            )
            .first()
        )
        if scoped is not None:
            return scoped

    return (
        db.query(PostProcessingSettings)
        .filter(
            PostProcessingSettings.project_id == project_id,
            PostProcessingSettings.impulse_id.is_(None),
        )
        .first()
    )


def upsert_settings(
    db: Session,
    project_id: str,
    update: PostProcessingSettingsUpdate,
    impulse_id: str | None = None,
) -> PostProcessingSettings:
    """
    Create the settings row if it does not exist, then apply the supplied
    fields (only non-None values are written).  Returns the persisted row.
    """
    row = None
    if impulse_id:
        row = (
            db.query(PostProcessingSettings)
            .filter(
                PostProcessingSettings.project_id == project_id,
                PostProcessingSettings.impulse_id == impulse_id,
            )
            .first()
        )
        if row is None:
            legacy_row = (
                db.query(PostProcessingSettings)
                .filter(
                    PostProcessingSettings.project_id == project_id,
                    PostProcessingSettings.impulse_id.is_(None),
                )
                .first()
            )
            if legacy_row is not None:
                row = PostProcessingSettings(
                    project_id=project_id,
                    impulse_id=impulse_id,
                    enabled=legacy_row.enabled,
                    threshold=legacy_row.threshold,
                    tracking_enabled=legacy_row.tracking_enabled,
                    keep_grace=legacy_row.keep_grace,
                    max_observations=legacy_row.max_observations,
                    class_filter=list(legacy_row.class_filter or []),
                )
                db.add(row)
    else:
        row = (
            db.query(PostProcessingSettings)
            .filter(
                PostProcessingSettings.project_id == project_id,
                PostProcessingSettings.impulse_id.is_(None),
            )
            .first()
        )

    if row is None:
        row = PostProcessingSettings(project_id=project_id, impulse_id=impulse_id)
        db.add(row)

    fields = update.model_dump(exclude_none=True)
    for field, value in fields.items():
        setattr(row, field, value)

    # JSON columns need explicit dirty-flagging so SQLAlchemy always emits the UPDATE
    # when the list value is replaced with an equal-by-value new object.
    # Only flag when class_filter was part of this update (otherwise the attribute
    # may not be loaded into the ORM state yet and flag_modified would raise).
    if "class_filter" in fields:
        flag_modified(row, "class_filter")

    db.commit()
    db.refresh(row)
    logger.info(
        "upsert_settings: project=%s impulse=%s class_filter=%r enabled=%s threshold=%.2f "
        "tracking=%s keep_grace=%d max_observations=%d",
        project_id, impulse_id, row.class_filter, row.enabled, row.threshold,
        row.tracking_enabled, row.keep_grace, row.max_observations,
    )
    return row


# ─── Job lookup ───────────────────────────────────────────────────────────────

def get_job(db: Session, project_id: str, job_id: str) -> ProcessingJob:
    """
    Return the job only if it belongs to the given project.
    Raises KeyError for missing job OR wrong-project access.
    """
    job = (
        db.query(ProcessingJob)
        .filter(ProcessingJob.id == job_id, ProcessingJob.project_id == project_id)
        .first()
    )
    if not job:
        raise KeyError(f"Job {job_id!r} not found in project {project_id!r}")
    return job
