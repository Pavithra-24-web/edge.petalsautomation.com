"""
Centralized authorization helpers.

Authentication (who you are) lives in ``app.core.auth`` (``get_current_user``).
This module owns *authorization* (what you may touch): role gating and
resource-ownership enforcement.

Design rules:
  * Ownership always resolves back to ``Project.owner_id``. Every protected
    resource is reachable from a project; helpers below walk the FK chain
    (model → training_job → impulse → project) where there is no direct
    ``project_id`` column.
  * A resource the caller does not own raises **404**, not 403, so the API
    never reveals that someone else's resource exists (enumeration guard).
  * Helpers are plain functions taking ``(db, id, user)`` so they work whether
    the id arrives via path, query, or request body. ``require_admin`` and
    ``owned_project`` are FastAPI dependencies for the common path-param case.
"""
from __future__ import annotations

from fastapi import Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.core.auth import get_current_user
from app.core.database import get_db
from app.models.user import (
    AILabelingAction,
    AILabelingJob,
    AIPrediction,
    Deployment,
    Impulse,
    Label,
    Project,
    ProjectVersion,
    Sample,
    SyntheticDataJob,
    TrainedModel,
    TrainingJob,
    User,
    UserRole,
)

__all__ = [
    "require_admin",
    "owned_project",
    "assert_project_owner",
    "assert_impulse_owner",
    "assert_sample_owner",
    "assert_label_owner",
    "assert_training_job_owner",
    "assert_trained_model_owner",
    "assert_deployment_owner",
    "assert_ai_action_owner",
    "assert_ai_job_owner",
    "assert_ai_prediction_owner",
    "assert_project_version_owner",
    "assert_synthetic_job_owner",
]


def _not_found(what: str, rid: str) -> HTTPException:
    # 404 (not 403) so non-owners cannot distinguish "forbidden" from "absent".
    return HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"{what} {rid!r} not found",
    )


# ─── Role gating ────────────────────────────────────────────────────────────

def require_admin(current_user: User = Depends(get_current_user)) -> User:
    """Dependency: allow only admin users; 403 otherwise."""
    if current_user.role != UserRole.admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin privileges required",
        )
    return current_user


# ─── Ownership assertions ───────────────────────────────────────────────────

def assert_project_owner(db: Session, project_id: str, user: User) -> Project:
    project = (
        db.query(Project)
        .filter(Project.id == project_id, Project.owner_id == user.id)
        .first()
    )
    if project is None:
        raise _not_found("Project", project_id)
    return project


def assert_impulse_owner(db: Session, impulse_id: str, user: User) -> Impulse:
    impulse = (
        db.query(Impulse)
        .join(Project, Impulse.project_id == Project.id)
        .filter(Impulse.id == impulse_id, Project.owner_id == user.id)
        .first()
    )
    if impulse is None:
        raise _not_found("Impulse", impulse_id)
    return impulse


def assert_sample_owner(db: Session, sample_id: str, user: User) -> Sample:
    sample = (
        db.query(Sample)
        .join(Project, Sample.project_id == Project.id)
        .filter(Sample.id == sample_id, Project.owner_id == user.id)
        .first()
    )
    if sample is None:
        raise _not_found("Sample", sample_id)
    return sample


def assert_label_owner(db: Session, label_id: str, user: User) -> Label:
    label = (
        db.query(Label)
        .join(Project, Label.project_id == Project.id)
        .filter(Label.id == label_id, Project.owner_id == user.id)
        .first()
    )
    if label is None:
        raise _not_found("Label", label_id)
    return label


def assert_training_job_owner(db: Session, job_id: str, user: User) -> TrainingJob:
    # TrainingJob has no project_id: TrainingJob → Impulse → Project.
    job = (
        db.query(TrainingJob)
        .join(Impulse, TrainingJob.impulse_id == Impulse.id)
        .join(Project, Impulse.project_id == Project.id)
        .filter(TrainingJob.id == job_id, Project.owner_id == user.id)
        .first()
    )
    if job is None:
        raise _not_found("Training job", job_id)
    return job


def assert_trained_model_owner(db: Session, model_id: str, user: User) -> TrainedModel:
    # TrainedModel → TrainingJob → Impulse → Project.
    model = (
        db.query(TrainedModel)
        .join(TrainingJob, TrainedModel.training_job_id == TrainingJob.id)
        .join(Impulse, TrainingJob.impulse_id == Impulse.id)
        .join(Project, Impulse.project_id == Project.id)
        .filter(TrainedModel.id == model_id, Project.owner_id == user.id)
        .first()
    )
    if model is None:
        raise _not_found("Trained model", model_id)
    return model


def assert_project_version_owner(db: Session, version_id: str, user: User) -> ProjectVersion:
    # ProjectVersion carries project_id directly, but scope through Project
    # for the ownership check like every other helper here.
    version = (
        db.query(ProjectVersion)
        .join(Project, ProjectVersion.project_id == Project.id)
        .filter(ProjectVersion.id == version_id, Project.owner_id == user.id)
        .first()
    )
    if version is None:
        raise _not_found("Version", version_id)
    return version


def assert_deployment_owner(db: Session, deployment_id: str, user: User) -> Deployment:
    # Deployment → TrainedModel → TrainingJob → Impulse → Project.
    deployment = (
        db.query(Deployment)
        .join(TrainedModel, Deployment.model_id == TrainedModel.id)
        .join(TrainingJob, TrainedModel.training_job_id == TrainingJob.id)
        .join(Impulse, TrainingJob.impulse_id == Impulse.id)
        .join(Project, Impulse.project_id == Project.id)
        .filter(Deployment.id == deployment_id, Project.owner_id == user.id)
        .first()
    )
    if deployment is None:
        raise _not_found("Deployment", deployment_id)
    return deployment


# ─── AI labeling resources ──────────────────────────────────────────────────

def assert_ai_action_owner(db: Session, action_id: str, user: User) -> AILabelingAction:
    action = (
        db.query(AILabelingAction)
        .join(Project, AILabelingAction.project_id == Project.id)
        .filter(AILabelingAction.id == action_id, Project.owner_id == user.id)
        .first()
    )
    if action is None:
        raise _not_found("Action", action_id)
    return action


def assert_ai_job_owner(db: Session, job_id: str, user: User) -> AILabelingJob:
    job = (
        db.query(AILabelingJob)
        .join(Project, AILabelingJob.project_id == Project.id)
        .filter(AILabelingJob.id == job_id, Project.owner_id == user.id)
        .first()
    )
    if job is None:
        raise _not_found("Job", job_id)
    return job


def assert_ai_prediction_owner(db: Session, prediction_id: str, user: User) -> AIPrediction:
    # AIPrediction → AILabelingJob → Project.
    pred = (
        db.query(AIPrediction)
        .join(AILabelingJob, AIPrediction.job_id == AILabelingJob.id)
        .join(Project, AILabelingJob.project_id == Project.id)
        .filter(AIPrediction.id == prediction_id, Project.owner_id == user.id)
        .first()
    )
    if pred is None:
        raise _not_found("Prediction", prediction_id)
    return pred


# ─── Synthetic data resources ───────────────────────────────────────────────

def assert_synthetic_job_owner(db: Session, job_id: str, user: User) -> SyntheticDataJob:
    job = (
        db.query(SyntheticDataJob)
        .join(Project, SyntheticDataJob.project_id == Project.id)
        .filter(SyntheticDataJob.id == job_id, Project.owner_id == user.id)
        .first()
    )
    if job is None:
        raise _not_found("Synthetic data job", job_id)
    return job


# ─── FastAPI dependency for the `/{project_id}` path-param case ──────────────

def owned_project(
    project_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> Project:
    """Path-param dependency: resolves ``project_id`` to a project owned by the
    caller, or 404. Use as ``project: Project = Depends(owned_project)``."""
    return assert_project_owner(db, project_id, current_user)
