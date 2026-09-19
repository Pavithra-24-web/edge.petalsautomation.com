"""
Projects endpoints
"""
import io
import json
import logging
import re
import zipfile
from typing import Optional

from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Query, Response
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session, joinedload
from pydantic import BaseModel

from app.core.database import get_db
from app.core.auth import get_current_user
from app.core.storage import storage
from app.models.user import (
    User, Project, Impulse, TrainingJob, TrainedModel, Deployment,
    DeviceUpdateHistory, Sample, SampleType,
    AILabelingJob, AILabelingAction,
    DspFeatureJob,
    ProjectVersion, ProjectVersionImpulse, ProjectVersionSample,
    SyntheticDataJob,
)
from app.models.devices import (
    ProjectDeviceKey, DeviceInferenceLog, DeviceCatalogEntry, DeviceSpecification,
)
from app.models.model_testing import (
    ModelVersion, ModelTestRun, ModelTestSample, MetricResult,
)
from app.ml.training_ui_log_filter import filter_training_ui_log_lines
from app.workers.sample_utils import IS_BACKGROUND_KEY

router = APIRouter()

logger = logging.getLogger(__name__)

# Which content the shared dashboard pages render for a project. See
# docs/Motion recognition/motion_phase0.md §2 — presentation-only, never a
# pipeline discriminator.
PROJECT_TYPES = {"object_detection", "motion"}

# `Project.description` doubles as the README body (Dashboard_projectinfo_implmentation.md
# Phase 2) — it is otherwise an unbounded `Text` column, so cap it here rather
# than at the DB layer.
_README_MAX_CHARS = 32_000

class ProjectCreate(BaseModel):
    name: str
    description: Optional[str] = None
    # Optional server-side (omitted ⇒ defaults to "object_detection", the
    # same value the migration backfills) so the 32 existing call sites that
    # post {"name": ...} with no type keep working. Required in the UI —
    # CreateProjectModal disables Save until a card is chosen.
    project_type: Optional[str] = None

class ProjectUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    # None is a meaningful value here — it clears the selection — so this field
    # is read via `model_fields_set` rather than an `is not None` check like the
    # two above.
    target_device_slug: Optional[str] = None
    # The four overridable configuration values (Target Device Phase 3,
    # docs/target_device_phase3.md). Same `model_fields_set` discipline as
    # `target_device_slug`: an explicit null resets one value to the board's
    # own specification figure, an omitted field leaves it alone. "Reset to
    # default settings" in the UI is simply all four sent as explicit nulls.
    target_device_custom_name: Optional[str] = None
    target_device_ram_kb: Optional[int] = None
    target_device_rom_kb: Optional[int] = None
    target_device_latency_ms: Optional[int] = None


# The four columns above, alongside the field on `ProjectUpdate` that sets
# each. Shared by the "needs a device" guard and the "board changed, wipe
# them" reset in `update_project`.
_OVERRIDE_FIELDS = (
    "target_device_custom_name",
    "target_device_ram_kb",
    "target_device_rom_kb",
    "target_device_latency_ms",
)


def _catalog_entry_dict(entry: DeviceCatalogEntry) -> dict:
    return {
        "slug": entry.slug,
        "display_name": entry.display_name,
        "family": entry.family,
        "deploy_target": entry.deploy_target.value,
        "accelerator_note": entry.accelerator_note,
    }


def _fetch_entry(db: Session, slug: Optional[str]) -> Optional[DeviceCatalogEntry]:
    """The catalog entry for a slug, with its specification eager-loaded —
    the one query both `_target_device_dict` and `_target_device_config` read
    from, so resolving a project's full response never issues two."""
    if not slug:
        return None
    return (
        db.query(DeviceCatalogEntry)
        .options(joinedload(DeviceCatalogEntry.specification))
        .filter(DeviceCatalogEntry.slug == slug)
        .first()
    )


def _target_device_dict(slug: Optional[str], entry: Optional[DeviceCatalogEntry]) -> Optional[dict]:
    """Resolve a project's target-device slug to its catalog entry.

    A slug with no matching row is reported as an unresolved selection rather
    than as no selection: the catalog is migration-owned, so an entry can be
    retired under a project that had chosen it, and silently showing "no target
    device" would misreport what the project is actually set to.
    """
    if not slug:
        return None
    if entry is None:
        return {"slug": slug, "display_name": slug, "family": None,
                "deploy_target": None, "accelerator_note": None, "unresolved": True}
    return _catalog_entry_dict(entry)


def _target_device(db: Session, slug: Optional[str]) -> Optional[dict]:
    """Convenience wrapper for callers that don't already have the entry."""
    return _target_device_dict(slug, _fetch_entry(db, slug))


def _override_field(override, spec_value):
    """One RAM/ROM/latency entry in the resolved configuration: the override
    wins if set, otherwise the board's own specification value — either way
    the board's figure is reported alongside so the UI can render "customised,
    was 2 GB" without a second call. `spec_value=None` (unknown hardware fact,
    or no specification at all) stays None rather than being substituted."""
    return {
        "value": override if override is not None else spec_value,
        "board_default": spec_value,
        "overridden": override is not None,
    }


def _target_device_config(p: Project, entry: Optional[DeviceCatalogEntry]) -> Optional[dict]:
    """The resolved application-budget configuration a project's target-device
    dialog reads and Phase 4/5/6 will later read. None when no device is
    selected — there is nothing to configure yet, and that's a valid state
    everywhere this is read.

    Tolerates an unresolved slug (`entry is None`, a retired catalog row) the
    same way `_target_device_dict` does: every specification-derived value is
    None, and any override is still reported (there's simply no board default
    to compare it against).

    Processor family, processor, CPU architecture and clock rate have no
    project-level storage at all — they come straight off the specification
    and are read-only for exactly that reason.
    """
    if not p.target_device_slug:
        return None
    spec: Optional[DeviceSpecification] = entry.specification if entry is not None else None
    return {
        "custom_name": p.target_device_custom_name,
        "processor_family": spec.processor_family if spec else None,
        "processor": spec.processor if spec else None,
        "cpu_architecture": spec.cpu_architecture if spec else None,
        "clock_rate_mhz": spec.clock_rate_mhz if spec else None,
        "ram_kb": _override_field(p.target_device_ram_kb, spec.ram_kb if spec else None),
        "rom_kb": _override_field(p.target_device_rom_kb, spec.rom_kb if spec else None),
        "latency_ms": _override_field(p.target_device_latency_ms, spec.latency_budget_ms if spec else None),
    }


def _proj(p, target_device: Optional[dict] = None, target_device_config: Optional[dict] = None):
    return {"id": p.id, "name": p.name, "description": p.description,
            "created_at": p.created_at.isoformat(), "owner_id": p.owner_id,
            "project_type": p.project_type,
            "target_device_slug": p.target_device_slug,
            "target_device": target_device,
            "target_device_config": target_device_config}

@router.post("/", status_code=201)
def create_project(req: ProjectCreate, db: Session = Depends(get_db),
                   current_user: User = Depends(get_current_user)):
    name = (req.name or "").strip()
    if not name:
        raise HTTPException(400, "Project name cannot be empty")
    if req.description is not None and len(req.description) > _README_MAX_CHARS:
        raise HTTPException(413, f"description must be at most {_README_MAX_CHARS} characters")
    project_type = req.project_type or "object_detection"
    if project_type not in PROJECT_TYPES:
        raise HTTPException(422, f"Invalid project_type: {req.project_type!r}")
    # Per-owner name uniqueness. The frontend pre-checks but the server is
    # the authority — handles concurrent creates from a second tab racing
    # past the client guard.
    clash = db.query(Project).filter(
        Project.owner_id == current_user.id,
        Project.name == name,
    ).first()
    if clash:
        raise HTTPException(409, "A project with this name already exists")
    proj = Project(owner_id=current_user.id, name=name, description=req.description,
                    project_type=project_type)
    db.add(proj); db.commit(); db.refresh(proj)
    return _proj(proj)

@router.get("/")
def list_projects(db: Session = Depends(get_db),
                  current_user: User = Depends(get_current_user)):
    projects = db.query(Project).filter(Project.owner_id == current_user.id).all()
    # One catalog query for the whole page rather than one per project — the
    # topbar calls this on every menu open. Specification eager-loaded so the
    # resolved configuration (§2) costs nothing extra here either.
    slugs = {p.target_device_slug for p in projects if p.target_device_slug}
    entries = (
        db.query(DeviceCatalogEntry)
        .options(joinedload(DeviceCatalogEntry.specification))
        .filter(DeviceCatalogEntry.slug.in_(slugs))
        .all()
        if slugs else []
    )
    by_slug = {e.slug: e for e in entries}

    return [
        _proj(
            p,
            _target_device_dict(p.target_device_slug, by_slug.get(p.target_device_slug)),
            _target_device_config(p, by_slug.get(p.target_device_slug)),
        )
        for p in projects
    ]

@router.get("/{project_id}")
def get_project(project_id: str, db: Session = Depends(get_db),
                current_user: User = Depends(get_current_user)):
    p = db.query(Project).filter(Project.id == project_id,
                                 Project.owner_id == current_user.id).first()
    if not p: raise HTTPException(404, "Project not found")
    entry = _fetch_entry(db, p.target_device_slug)
    return _proj(p, _target_device_dict(p.target_device_slug, entry), _target_device_config(p, entry))

@router.patch("/{project_id}")
def update_project(project_id: str, req: ProjectUpdate,
                   db: Session = Depends(get_db),
                   current_user: User = Depends(get_current_user)):
    p = db.query(Project).filter(Project.id == project_id,
                                  Project.owner_id == current_user.id).first()
    if not p:
        raise HTTPException(404, "Project not found")
    if req.name is not None:
        new_name = req.name.strip()
        if not new_name:
            raise HTTPException(400, "Project name cannot be empty")
        # Enforce per-owner name uniqueness so two projects in the same
        # account can't collide. The frontend pre-checks this for a clean
        # inline error, but a concurrent rename from another tab can still
        # race past it — the server is the authority.
        clash = db.query(Project).filter(
            Project.owner_id == current_user.id,
            Project.name == new_name,
            Project.id != project_id,
        ).first()
        if clash:
            raise HTTPException(409, "A project with this name already exists")
        p.name = new_name
    if req.description is not None:
        if len(req.description) > _README_MAX_CHARS:
            raise HTTPException(413, f"description must be at most {_README_MAX_CHARS} characters")
        p.description = req.description
    # Presence in `model_fields_set`, not `is not None` — an explicit null is
    # how the client clears the selection, and "not chosen" is a valid state.
    if "target_device_slug" in req.model_fields_set:
        slug = (req.target_device_slug or "").strip() or None
        if slug is not None:
            known = db.query(DeviceCatalogEntry).filter(
                DeviceCatalogEntry.slug == slug
            ).first()
            if known is None:
                raise HTTPException(404, f"Device catalog entry '{slug}' not found")
        if slug != p.target_device_slug:
            # A different board (or none) means every override carried from
            # the old one is a wrong number, not a customisation — clear all
            # four in this same transaction. Re-selecting the SAME slug is
            # not a change and falls through with overrides intact.
            p.target_device_custom_name = None
            p.target_device_ram_kb = None
            p.target_device_rom_kb = None
            p.target_device_latency_ms = None
        p.target_device_slug = slug

    touched_overrides = [f for f in _OVERRIDE_FIELDS if f in req.model_fields_set]
    if touched_overrides and not p.target_device_slug:
        raise HTTPException(400, "Set a target device before overriding its configuration")

    for field, label in (
        ("target_device_ram_kb", "RAM"),
        ("target_device_rom_kb", "ROM"),
        ("target_device_latency_ms", "latency"),
    ):
        if field in req.model_fields_set:
            value = getattr(req, field)
            if value is not None and value <= 0:
                raise HTTPException(400, f"{label} override must be a positive number")

    if "target_device_custom_name" in req.model_fields_set:
        p.target_device_custom_name = (req.target_device_custom_name or "").strip() or None
    if "target_device_ram_kb" in req.model_fields_set:
        p.target_device_ram_kb = req.target_device_ram_kb
    if "target_device_rom_kb" in req.model_fields_set:
        p.target_device_rom_kb = req.target_device_rom_kb
    if "target_device_latency_ms" in req.model_fields_set:
        p.target_device_latency_ms = req.target_device_latency_ms

    db.commit(); db.refresh(p)
    entry = _fetch_entry(db, p.target_device_slug)
    return _proj(p, _target_device_dict(p.target_device_slug, entry), _target_device_config(p, entry))

def _purge_project_children(db: Session, project_id: str) -> None:
    """Delete every row that references ``projects.id`` but is NOT covered by
    an ORM cascade on the Project model.

    Several child tables lack a cascade relationship and have no DB-level
    ON DELETE CASCADE, so removing the project row without clearing them first
    raises ForeignKeyViolation (first one observed in the wild:
    project_device_keys_project_id_fkey). Clear them here in an order that
    respects intra-group FK dependencies; the caller then removes the project
    row itself and lets ORM cascade handle the relationships that ARE
    configured (labels, samples, impulses, devices, ai_labeling_actions,
    post_processing_settings, processing_jobs).

    Shared by ``delete_project`` and account deletion so both close the exact
    same FK gaps. Does NOT delete the project row or purge object storage —
    those are the caller's responsibility.

    Bulk-delete via ``.delete(synchronize_session=False)`` issues a raw SQL
    DELETE — it does NOT trigger ORM cascades, so any grandchildren
    (e.g. metric_results below model_test_runs) must be cleared first.
    """
    # 1) model_testing tree — leaf rows first
    db.query(MetricResult).filter(
        MetricResult.test_run_id.in_(
            db.query(ModelTestRun.id).filter(ModelTestRun.project_id == project_id)
        )
    ).delete(synchronize_session=False)
    db.query(ModelTestSample).filter(
        ModelTestSample.project_id == project_id
    ).delete(synchronize_session=False)
    db.query(ModelTestRun).filter(
        ModelTestRun.project_id == project_id
    ).delete(synchronize_session=False)
    db.query(ModelVersion).filter(
        ModelVersion.project_id == project_id
    ).delete(synchronize_session=False)

    # 2) Device feature tables
    db.query(DeviceUpdateHistory).filter(
        DeviceUpdateHistory.project_id == project_id
    ).delete(synchronize_session=False)
    db.query(DeviceInferenceLog).filter(
        DeviceInferenceLog.project_id == project_id
    ).delete(synchronize_session=False)
    db.query(ProjectDeviceKey).filter(
        ProjectDeviceKey.project_id == project_id
    ).delete(synchronize_session=False)

    # 3) Deployments. The Project→Impulse→TrainingJob→TrainedModel→Deployment
    #    ORM chain normally cascades these, but Deployment.project_id is an
    #    independent FK that can still fail if a deployment exists outside
    #    that chain (e.g. legacy data, deployments created with no model).
    #    Belt-and-braces bulk delete.
    db.query(Deployment).filter(
        Deployment.project_id == project_id
    ).delete(synchronize_session=False)

    # 3b) DspFeatureJob has the same shape as deployment here: ORM cascade
    #     via Impulse handles the normal case, but the direct project_id
    #     FK can fail for any orphan row whose impulse was bulk-deleted.
    db.query(DspFeatureJob).filter(
        DspFeatureJob.project_id == project_id
    ).delete(synchronize_session=False)

    # 4) project_versions tree (docs/Action/parityfix.md) — leaf rows first,
    #    same shape as the model_testing tree in (1). Project has no ORM
    #    relationship to ProjectVersion, and project_versions.project_id has
    #    no DB-level ON DELETE CASCADE, so it FK-violates the same way
    #    project_device_keys did before this function existed.
    db.query(ProjectVersionSample).filter(
        ProjectVersionSample.version_id.in_(
            db.query(ProjectVersion.id).filter(ProjectVersion.project_id == project_id)
        )
    ).delete(synchronize_session=False)
    db.query(ProjectVersionImpulse).filter(
        ProjectVersionImpulse.version_id.in_(
            db.query(ProjectVersion.id).filter(ProjectVersion.project_id == project_id)
        )
    ).delete(synchronize_session=False)
    db.query(ProjectVersion).filter(
        ProjectVersion.project_id == project_id
    ).delete(synchronize_session=False)

    # 5) synthetic_data_jobs — same belt-and-braces shape as (3)/(3b): an ORM
    #    cascade is declared on Project.synthetic_jobs, but this function
    #    exists precisely because that alone has not been trustworthy enough
    #    for every direct-FK child table in this codebase.
    db.query(SyntheticDataJob).filter(
        SyntheticDataJob.project_id == project_id
    ).delete(synchronize_session=False)


@router.delete("/{project_id}", status_code=204)
def delete_project(project_id: str, db: Session = Depends(get_db),
                   current_user: User = Depends(get_current_user)):
    p = db.query(Project).filter(Project.id == project_id,
                                  Project.owner_id == current_user.id).first()
    if not p: raise HTTPException(404, "Project not found")

    _purge_project_children(db, p.id)

    # Finally remove the project itself. ORM cascade handles the remaining
    # children declared on Project.
    db.delete(p)
    db.commit()
    return Response(status_code=204)


# ─── Jobs (read-only union view) ──────────────────────────────────────────────
#
# There is no unified jobs table in this app. Background work is recorded
# across four independent tables, each with its own status vocabulary:
#
#   training_jobs    — keyed via impulse_id (JOIN through impulses for project)
#   deployments      — project_id directly
#   model_test_runs  — project_id directly
#   ai_labeling_jobs — project_id directly
#
# This endpoint fetches each source, normalises rows into a single shape,
# merges them ordered by created_at DESC, and pages the result. The merge
# happens in Python rather than via SQL UNION ALL because the per-table
# columns and enums differ; for a single project the row count is bounded
# and the cost is negligible.
#
# Dataset export, DSP feature generation, and feature-explorer/profiling/
# impulse-cloning are KNOWINGLY EXCLUDED — none of them persist a job row.

# Map each source-table's status vocabulary into the unified set used by
# the response. Lookup is lowercased. Unmapped values pass through
# verbatim and the frontend renders them as "unknown".
_STATUS_MAP = {
    "queued": "pending",
    "pending": "pending",
    "draft": "pending",      # AILabelingStatus.draft
    "running": "running",
    "in_progress": "running",
    "completed": "completed",
    "success": "completed",
    "succeeded": "completed",
    "failed": "failed",
    "error": "failed",
    "cancelled": "cancelled",
    "canceled": "cancelled",
}


def _norm_status(raw) -> str:
    if raw is None:
        return "unknown"
    # SAEnum values come through as Enum instances; .value gives the string.
    value = getattr(raw, "value", raw)
    return _STATUS_MAP.get(str(value).lower(), str(value).lower())


def _duration_seconds(created_at, completed_at, status: str) -> float:
    if completed_at is not None and created_at is not None:
        return max(0.0, (completed_at - created_at).total_seconds())
    if status == "running" and created_at is not None:
        return max(0.0, (datetime.utcnow() - created_at).total_seconds())
    return 0.0


def _iso(dt):
    return dt.isoformat() if dt else None


# The single source of truth for what the /jobs endpoint is allowed to
# return. Defence in depth: a row whose `type` is outside this set is
# filtered out at the end of row assembly and logged — see
# list_project_jobs below.
#
# Knowingly excluded:
#   • Feature explorer — `/dsp/features/{impulse_id}` is a SYNCHRONOUS
#     read of `features.npz` plus an in-process PCA. It is not a background
#     job at all; it is computed on each request and returned inline.
#   • Dataset export   — streaming ZIP endpoint, no DB row.
#
# If either gains a real persistence layer in the future, add a new
# branch to list_project_jobs AND add the new type string here. Do not
# attempt to synthesise rows from logs, blob timestamps, or Celery task
# history — the result would look broken next to the real jobs because
# status/duration/started/finished would all be hardcoded.
_ALLOWED_JOB_TYPES = {
    "training",
    "retraining",
    "deployment",
    "model_test",
    "ai_labeling",
    "feature_generation",
    "synthetic_data",
}


# ── Type-label formatter ──────────────────────────────────────────────────────
#
# Generic labels like "Training" or "Model testing" are ambiguous when a
# project has more than one impulse. The formatter below resolves each job
# to a human-readable string with impulse context, derived entirely from
# data already on the joined rows. The same helper is used for every source
# so the format never drifts between branches.

# Architecture → category. Pulled from TrainingJob.extra_params["architecture"].
# Anything outside these sets falls back to no category segment — never
# "Training" alone.
_OBJECT_DETECTION_ARCHS = {
    "fomo_mobilenetv2_0_1",
    "fomo_v2_0.35",
    "yolo_pro",
    "mobilenet_v2_ssd_fpn_lite",
}
_CLASSIFICATION_ARCHS = {
    "dense", "conv1d", "conv2d", "lstm", "mobilenet", "transfer",
}


def _model_category(architecture) -> str | None:
    if not architecture:
        return None
    arch = str(architecture).lower()
    if arch in _OBJECT_DETECTION_ARCHS:
        return "Object detection"
    if arch in _CLASSIFICATION_ARCHS:
        return "Classification"
    return None


# Deployment target → human label. Covers the DeployTarget enum
# (tflite/arduino/esp32/raspberry_pi/unoq/cpp/pxe). Unknown values fall
# through to a title-cased rendering of the raw string.
_DEPLOY_TARGET_LABELS = {
    "tflite":       "TensorFlow Lite",
    "arduino":      "Arduino",
    "esp32":        "ESP32",
    "raspberry_pi": "Raspberry Pi",
    "unoq":         "Arduino UNO Q",
    "cpp":          "C++",
    "pxe":          "PXE package",
}


def _deploy_target_human(raw) -> str | None:
    if not raw:
        return None
    key = str(raw).lower()
    if key in _DEPLOY_TARGET_LABELS:
        return _DEPLOY_TARGET_LABELS[key]
    # Final fallback: humanise the raw value rather than emit empty parens.
    return str(raw).replace("_", " ").strip().title()


# Impulse names default to "Impulse N" via _next_default_impulse_name in
# impulses.py. We parse that pattern so jobs can render "Impulse #N"
# consistently; renamed impulses (e.g. "Cow Detector") fall back to the
# verbatim name.
_IMPULSE_DEFAULT_NAME_RE = re.compile(r"^\s*Impulse\s+(\d+)\s*$", re.IGNORECASE)


def _impulse_label(impulse_name: str | None) -> str:
    if not impulse_name:
        return "unknown impulse"
    m = _IMPULSE_DEFAULT_NAME_RE.match(impulse_name)
    if m:
        return f"Impulse #{m.group(1)}"
    return impulse_name.strip()


_INPUT_TYPE_LABELS = {
    "image":       "Image",
    "audio":       "Audio",
    "time-series": "Time series",
    "time_series": "Time series",
}


def _input_type_human(raw) -> str | None:
    if not raw:
        return None
    key = str(raw).lower()
    if key in _INPUT_TYPE_LABELS:
        return _INPUT_TYPE_LABELS[key]
    return str(raw).replace("_", " ").strip().title()


def _format_type_label(
    *,
    kind: str,
    impulse_name: str | None = None,
    architecture: str | None = None,
    deploy_target: str | None = None,
    action_name: str | None = None,
    input_type: str | None = None,
) -> str:
    """Build the user-facing Type cell for a single job row.

    Format reference (kind → output):
      training           → "Training model ({category}, {imp})"   ← category omitted if unknown
      retraining         → "Retraining model ({category}, {imp})"
      model_test         → "Model testing ({imp})"
      deployment         → "Building deployment ({target}, {imp})" ← target/imp omitted if missing
      ai_labeling        → "AI labeling ({action_name})"           ← action_name fallback "AI labeling"
      feature_generation → "Generating features ({input_type}, {imp})"
      synthetic_data     → "Synthetic data generation"

    Feature explorer is NOT handled here — see _ALLOWED_JOB_TYPES above for
    why it cannot appear on the Jobs page (synchronous read, not a job).
    If a real persistence layer is added later, add a new branch here AND
    a new entry to _ALLOWED_JOB_TYPES.

    Invariants: never emits empty parens, never falls back to a bare
    generic label when any qualifying metadata exists.
    """
    imp = _impulse_label(impulse_name) if impulse_name is not None else None

    if kind in ("training", "retraining"):
        head = "Retraining model" if kind == "retraining" else "Training model"
        category = _model_category(architecture)
        parts = [p for p in (category, imp) if p]
        return f"{head} ({', '.join(parts)})" if parts else head

    if kind == "model_test":
        return f"Model testing ({imp})" if imp else "Model testing"

    if kind == "deployment":
        target_human = _deploy_target_human(deploy_target)
        parts = [p for p in (target_human, imp) if p]
        return f"Building deployment ({', '.join(parts)})" if parts else "Building deployment"

    if kind == "ai_labeling":
        return f"AI labeling ({action_name})" if action_name else "AI labeling"

    if kind == "feature_generation":
        input_human = _input_type_human(input_type)
        parts = [p for p in (input_human, imp) if p]
        return f"Generating features ({', '.join(parts)})" if parts else "Generating features"

    if kind == "synthetic_data":
        return "Synthetic data generation"

    return kind.replace("_", " ").title()


@router.get("/{project_id}/jobs")
def list_project_jobs(
    project_id: str,
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Read-only union of jobs that ran in this project.

    Sources merged:
      • Training / Retraining  (training_jobs ↔ impulses)
      • Deployment builds      (deployments)
      • Model testing          (model_test_runs)
      • AI labeling            (ai_labeling_jobs)
      • Feature generation     (dsp_feature_jobs)
      • Synthetic data         (synthetic_data_jobs)

    Status from each source is normalised into
    {pending, running, completed, failed, cancelled} via _STATUS_MAP.

    `started_by` is the project owner's identifier — none of the source
    tables carry a per-job user_id, so attribution is at the project level.

    Response shape:
        { items: [...], total: int, page: int, page_size: int }
    Ordered by created_at DESC across the union; ties broken by id DESC
    so pagination stays stable for rows with identical timestamps.
    """
    project = (
        db.query(Project)
        .filter(Project.id == project_id, Project.owner_id == current_user.id)
        .first()
    )
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    owner = project.owner
    started_by = (owner.email or owner.username) if owner else None

    rows: list[dict] = []

    # ── 1) Training & retraining ──────────────────────────────────────
    training_rows = (
        db.query(
            TrainingJob,
            Impulse.id.label("impulse_id"),
            Impulse.name.label("impulse_name"),
        )
        .join(Impulse, Impulse.id == TrainingJob.impulse_id)
        .filter(Impulse.project_id == project_id)
        .all()
    )
    for job, impulse_id, impulse_name in training_rows:
        extra = job.extra_params or {}
        launch_mode = extra.get("launch_mode", "start")
        is_retrain = launch_mode == "retrain"
        status = _norm_status(job.status)
        kind = "retraining" if is_retrain else "training"
        rows.append({
            "id":                job.id,
            "type":              kind,
            "type_label":        _format_type_label(
                kind=kind,
                impulse_name=impulse_name,
                architecture=extra.get("architecture"),
            ),
            "status":            status,
            "started_by":        started_by,
            "created_at":        _iso(job.created_at),
            "completed_at":      _iso(job.completed_at),
            "duration_seconds":  _duration_seconds(job.created_at, job.completed_at, status),
            "source_table":      "training_jobs",
            "source_id":         job.id,
            "impulse_id":        impulse_id,
            "job_id":            job.id,
            "launch_mode":       launch_mode,
            # Internal sort fields, stripped before response:
            "_sort_dt":          job.created_at,
        })

    # ── 2) Deployment builds ──────────────────────────────────────────
    # Walk Deployment → TrainedModel → TrainingJob → Impulse so the label
    # can carry impulse context. The chain is LEFT OUTER because a legacy
    # row may have a missing model FK; the formatter handles a null name.
    deployment_rows = (
        db.query(Deployment, Impulse.name.label("impulse_name"))
        .outerjoin(TrainedModel, TrainedModel.id == Deployment.model_id)
        .outerjoin(TrainingJob, TrainingJob.id == TrainedModel.training_job_id)
        .outerjoin(Impulse, Impulse.id == TrainingJob.impulse_id)
        .filter(Deployment.project_id == project_id)
        .all()
    )
    for d, impulse_name in deployment_rows:
        status = _norm_status(d.status)
        deploy_target = d.deployment_target or (d.target.value if d.target else None)
        rows.append({
            "id":                d.id,
            "type":              "deployment",
            "type_label":        _format_type_label(
                kind="deployment",
                impulse_name=impulse_name,
                deploy_target=deploy_target,
            ),
            "status":            status,
            "started_by":        started_by,
            "created_at":        _iso(d.created_at),
            "completed_at":      _iso(d.completed_at),
            "duration_seconds":  _duration_seconds(d.created_at, d.completed_at, status),
            "source_table":      "deployments",
            "source_id":         d.id,
            "download_url":      d.download_url,
            "_sort_dt":          d.created_at,
        })

    # ── 3) Model test runs ────────────────────────────────────────────
    # JOIN Impulse so the label can render "Model testing (Impulse #N)".
    test_rows = (
        db.query(ModelTestRun, Impulse.name.label("impulse_name"))
        .outerjoin(Impulse, Impulse.id == ModelTestRun.impulse_id)
        .filter(ModelTestRun.project_id == project_id)
        .all()
    )
    for r, impulse_name in test_rows:
        status = _norm_status(r.status)
        rows.append({
            "id":                r.id,
            "type":              "model_test",
            "type_label":        _format_type_label(
                kind="model_test",
                impulse_name=impulse_name,
            ),
            "status":            status,
            "started_by":        started_by,
            "created_at":        _iso(r.created_at),
            "completed_at":      _iso(r.completed_at),
            "duration_seconds":  _duration_seconds(r.created_at, r.completed_at, status),
            "source_table":      "model_test_runs",
            "source_id":         r.id,
            "_sort_dt":          r.created_at,
        })

    # ── 4) AI labeling jobs ───────────────────────────────────────────
    # JOIN AILabelingAction so the label surfaces the action's name —
    # AILabelingJob is action-scoped (not impulse-scoped), so the action
    # is the only meaningful identifier for the run.
    labeling_rows = (
        db.query(AILabelingJob, AILabelingAction.name.label("action_name"))
        .outerjoin(AILabelingAction, AILabelingAction.id == AILabelingJob.action_id)
        .filter(AILabelingJob.project_id == project_id)
        .all()
    )
    for j, action_name in labeling_rows:
        status = _norm_status(j.status)
        rows.append({
            "id":                j.id,
            "type":              "ai_labeling",
            "type_label":        _format_type_label(
                kind="ai_labeling",
                action_name=action_name,
            ),
            "status":            status,
            "started_by":        started_by,
            "created_at":        _iso(j.created_at),
            "completed_at":      _iso(j.completed_at),
            "duration_seconds":  _duration_seconds(j.created_at, j.completed_at, status),
            "source_table":      "ai_labeling_jobs",
            "source_id":         j.id,
            "_sort_dt":          j.created_at,
        })

    # ── 5) DSP feature-generation jobs ────────────────────────────────
    # project_id is denormalised on dsp_feature_jobs (see model) so we
    # don't need to JOIN through impulses here; the Impulse JOIN below
    # supplies the human-readable name for the label.
    feature_rows = (
        db.query(DspFeatureJob, Impulse.name.label("impulse_name"))
        .outerjoin(Impulse, Impulse.id == DspFeatureJob.impulse_id)
        .filter(DspFeatureJob.project_id == project_id)
        .all()
    )
    for f, impulse_name in feature_rows:
        status = _norm_status(f.status)
        rows.append({
            "id":                f.id,
            "type":              "feature_generation",
            "type_label":        _format_type_label(
                kind="feature_generation",
                impulse_name=impulse_name,
                input_type=f.input_type,
            ),
            "status":            status,
            "started_by":        started_by,
            "created_at":        _iso(f.created_at),
            "completed_at":      _iso(f.completed_at),
            "duration_seconds":  _duration_seconds(f.created_at, f.completed_at, status),
            "source_table":      "dsp_feature_jobs",
            "source_id":         f.id,
            "_sort_dt":          f.created_at,
        })

    # ── 6) Synthetic data generation jobs ─────────────────────────────
    synthetic_rows = (
        db.query(SyntheticDataJob)
        .filter(SyntheticDataJob.project_id == project_id)
        .all()
    )
    for sd in synthetic_rows:
        status = _norm_status(sd.status)
        rows.append({
            "id":                sd.id,
            "type":              "synthetic_data",
            "type_label":        _format_type_label(kind="synthetic_data"),
            "status":            status,
            "started_by":        started_by,
            "created_at":        _iso(sd.created_at),
            "completed_at":      _iso(sd.completed_at),
            "duration_seconds":  _duration_seconds(sd.created_at, sd.completed_at, status),
            "source_table":      "synthetic_data_jobs",
            "source_id":         sd.id,
            "_sort_dt":          sd.created_at,
        })

    # ── Allow-list guard ──────────────────────────────────────────────
    # Runtime filter (NOT assert — production runs under python -O may
    # strip asserts, and we'd rather degrade to a missing row than 500
    # the whole page). Anything outside _ALLOWED_JOB_TYPES is dropped
    # and counted; if anything was dropped we log a warning so the
    # regression surfaces in ops without crashing the user.
    pre_filter_total = len(rows)
    rows = [r for r in rows if r["type"] in _ALLOWED_JOB_TYPES]
    dropped = pre_filter_total - len(rows)
    if dropped:
        logger.warning(
            "jobs endpoint dropped %d out-of-scope row(s) for project %s",
            dropped, project_id,
        )

    # ── Sort + paginate ───────────────────────────────────────────────
    # Order by created_at DESC; tie-break by id DESC so the page boundary
    # stays stable across calls when timestamps collide.
    rows.sort(
        key=lambda r: (
            r["_sort_dt"] or datetime.min,
            r["id"],
        ),
        reverse=True,
    )

    total = len(rows)
    start = (page - 1) * page_size
    end = start + page_size
    page_rows = rows[start:end]

    # Strip internal sort key before returning.
    for r in page_rows:
        r.pop("_sort_dt", None)

    return {
        "items":     page_rows,
        "total":     total,
        "page":      page,
        "page_size": page_size,
    }


# ─── Job logs (read-only viewer) ────────────────────────────────────────────────
#
# Edge-Impulse-style log viewer: a read-only projection over logs we already
# persist. No new storage, no reconstruction, no synthesis.
#
# Only job types that actually persist a log stream are addressable here:
#   • training / retraining → TrainingJob.training_history["log_lines"]
#   • dsp_feature           → DspFeatureJob has no log column today, so the
#                             response carries an empty list. We surface the
#                             type rather than 400 so the UI can still open a
#                             modal and show its empty state — but we never
#                             fabricate lines from artifacts or DB history.
#
# Anything else (deployment, model_test, ai_labeling, unknown strings) is
# rejected with 400 — those rows render no View Logs action on the frontend.
_LOGGABLE_JOB_TYPES = {"training", "retraining", "dsp_feature"}


@router.get("/{project_id}/jobs/{job_type}/{job_id}/logs")
def get_job_logs(
    project_id: str,
    job_type: str,
    job_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Return persisted log lines for a single historical job.

    Response:
        { job_id, job_type, status, log_lines }

    `log_lines` preserves stored order verbatim — no trimming, deduping, or
    re-formatting. Training/retraining lines pass through the same UI filter
    the training page applies so the viewer is identical to the live view.
    """
    if job_type not in _LOGGABLE_JOB_TYPES:
        raise HTTPException(status_code=400, detail=f"Unsupported job type: {job_type}")

    # Same project-membership check as every other project endpoint. 404 (not
    # 403) keeps project existence opaque to non-members.
    project = (
        db.query(Project)
        .filter(Project.id == project_id, Project.owner_id == current_user.id)
        .first()
    )
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    if job_type in ("training", "retraining"):
        # Scope the lookup to the project via the impulse JOIN so a job that
        # belongs to another project reads as a 404 rather than leaking.
        row = (
            db.query(TrainingJob)
            .join(Impulse, Impulse.id == TrainingJob.impulse_id)
            .filter(TrainingJob.id == job_id, Impulse.project_id == project_id)
            .first()
        )
        if not row:
            raise HTTPException(status_code=404, detail="Job not found")
        th = row.training_history or {}
        raw_lines = th.get("log_lines") if isinstance(th, dict) else None
        log_lines = filter_training_ui_log_lines(raw_lines)
        status = _norm_status(row.status)
    else:  # dsp_feature
        row = (
            db.query(DspFeatureJob)
            .filter(DspFeatureJob.id == job_id, DspFeatureJob.project_id == project_id)
            .first()
        )
        if not row:
            raise HTTPException(status_code=404, detail="Job not found")
        # DspFeatureJob persists no log stream today — return empty, never
        # synthesise from features.npz or run history.
        log_lines = []
        status = _norm_status(row.status)

    return {
        "job_id":    job_id,
        "job_type":  job_type,
        "status":    status,
        "log_lines": log_lines,
    }


# ─── Dataset health ───────────────────────────────────────────────────────────

# Healthy when each per-label split sits inside this window AND the overall
# train share is in here too. 80% is the suggested target; the band gives
# enough room for small datasets to wobble without flagging false positives.
_HEALTHY_TRAIN_PCT_MIN = 70
_HEALTHY_TRAIN_PCT_MAX = 90
# Below this total a per-label split is too noisy to score meaningfully —
# we surface the warning so users add more samples rather than chasing the
# ratio.
_MIN_SAMPLES_FOR_RATIO = 5


def _classify_split_health(training: int, testing: int) -> tuple[bool, Optional[str]]:
    """Return (healthy, reason). Reasons are picked in priority order:

    1. no_training_samples — nothing in the training split
    2. no_testing_samples  — nothing in the testing split
    3. too_few_samples     — fewer than _MIN_SAMPLES_FOR_RATIO combined
    4. imbalanced_split    — train share outside the healthy band
    """
    total = training + testing
    if total == 0:
        # No data either side; surface as "no training" so the UI nudges
        # the user to upload, not to rebalance.
        return False, "no_training_samples"
    if training == 0:
        return False, "no_training_samples"
    if testing == 0:
        return False, "no_testing_samples"
    if total < _MIN_SAMPLES_FOR_RATIO:
        return False, "too_few_samples"
    train_pct = (training / total) * 100
    if train_pct < _HEALTHY_TRAIN_PCT_MIN or train_pct > _HEALTHY_TRAIN_PCT_MAX:
        return False, "imbalanced_split"
    return True, None


@router.get("/{project_id}/dataset-health")
def get_dataset_health(
    project_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Per-label and overall train/test split health for the dataset.

    Returns:
        {
          "overall": {"training": int, "testing": int, "healthy": bool, "reason": str|None},
          "labels": [
            {"name": str, "training": int, "testing": int, "healthy": bool, "reason": str|None},
            ...
          ]
        }

    This dataset is multi-label: a sample carries Sample.label.name (when set)
    plus one label per bounding box, and it is counted once for EVERY distinct
    label it carries. Per-label counts therefore do NOT sum to the total sample
    count. Post-processing samples are excluded — they aren't part of the
    train/test pipeline.
    """
    project = (
        db.query(Project)
        .filter(Project.id == project_id, Project.owner_id == current_user.id)
        .first()
    )
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    from sqlalchemy.orm import joinedload

    samples = (
        db.query(Sample)
        .options(joinedload(Sample.label))
        .filter(Sample.project_id == project_id)
        .all()
    )

    # Collect the FULL set of labels on each sample (matching the frontend):
    # classification label + every non-empty bounding-box label, deduplicated
    # case-insensitively. This dataset is multi-label, so a sample contributes
    # to every one of its label buckets and per-label counts do NOT sum to the
    # sample count. Samples with no label at all are EXCLUDED from health
    # entirely: they can't be train/test-balanced because they have no class,
    # and including them produced a 0% / 100% noise row that flipped the inline
    # warning on for projects whose real classes were all healthy. The
    # data-acquisition table still shows unlabeled samples — health is the only
    # surface that ignores them.
    #
    # Background ("negative") images are the one exception to the
    # exclude-the-unlabeled rule.  They have no class by design, so they cannot
    # appear in any per-label bucket, but they ARE trained on and they DO need
    # a train/test balance — a project whose negatives all sit in training
    # learns to suppress backgrounds it is never scored on.  They get their own
    # row rather than vanishing from the surface entirely.
    per_label: dict[str, dict[str, int]] = {}
    overall = {"training": 0, "testing": 0}
    background_counts = {"training": 0, "testing": 0}

    def _clean_label(raw) -> str:
        if raw is None:
            return ""
        text = str(raw).strip()
        if not text or text.lower() in ("unlabeled", "unlabelled", "unknown"):
            return ""
        return text

    for s in samples:
        if s.sample_type == SampleType.postprocessing:
            continue

        _meta_bg = s.extra_metadata or {}
        if isinstance(_meta_bg, dict) and _meta_bg.get(IS_BACKGROUND_KEY):
            _split = "training" if s.sample_type != SampleType.testing else "testing"
            background_counts[_split] += 1
            overall[_split] += 1
            continue

        # {lowercased name: first-seen display casing}
        keys: dict[str, str] = {}
        text = _clean_label(s.label.name if s.label else None)
        if text:
            keys[text.lower()] = text
        meta = s.extra_metadata or {}
        boxes = meta.get("boundingBoxes") if isinstance(meta, dict) else None
        if isinstance(boxes, list):
            for b in boxes:
                text = _clean_label(b.get("label") if isinstance(b, dict) else None)
                if text:
                    keys.setdefault(text.lower(), text)

        if not keys:
            # Skip from both per-label and overall — see comment above.
            continue

        split = "training" if s.sample_type != SampleType.testing else "testing"
        for key in keys.values():
            bucket = per_label.setdefault(key, {"training": 0, "testing": 0})
            bucket[split] += 1
            overall[split] += 1

    background_payload: Optional[dict] = None
    if background_counts["training"] or background_counts["testing"]:
        _bg_healthy, _bg_reason = _classify_split_health(
            background_counts["training"], background_counts["testing"]
        )
        background_payload = {
            "name": "Background",
            "training": background_counts["training"],
            "testing": background_counts["testing"],
            "healthy": _bg_healthy,
            "reason": _bg_reason,
        }

    labels_payload: list[dict] = []
    for name in sorted(per_label.keys(), key=lambda n: n.lower()):
        counts = per_label[name]
        healthy, reason = _classify_split_health(counts["training"], counts["testing"])
        labels_payload.append({
            "name": name,
            "training": counts["training"],
            "testing": counts["testing"],
            "healthy": healthy,
            "reason": reason,
        })

    # If there are no labeled samples at all, treat overall as healthy —
    # there's nothing to balance, so the inline warning would only
    # confuse a fresh-project state.
    if not labels_payload:
        overall_healthy, overall_reason = True, None
    else:
        overall_healthy, overall_reason = _classify_split_health(
            overall["training"], overall["testing"]
        )
        # The overall card is the headline indicator — if any single label is
        # unhealthy, the overall ratio should warn too even if the totals
        # happen to land inside the band.
        if overall_healthy and any(not l["healthy"] for l in labels_payload):
            overall_healthy = False
            overall_reason = "label_imbalance"

    return {
        "overall": {
            "training": overall["training"],
            "testing": overall["testing"],
            "healthy": overall_healthy,
            "reason": overall_reason,
        },
        "labels": labels_payload,
        # Separate from `labels` so the frontend never mistakes it for a class:
        # None when the project has no background images at all.
        "background": background_payload,
    }


# ─── Dataset export ───────────────────────────────────────────────────────────

class _ZipStreamBuffer(io.RawIOBase):
    """A write-only file-like that lets a generator drain ZipFile output.

    zipfile.ZipFile insists on a writable target. We give it this buffer,
    then between write() calls the streaming generator calls drain() to
    emit any new bytes to the HTTP response. No full-file buffering, so
    the dataset size is bounded by per-file memory (S3 download_bytes
    holds one sample at a time), not by the sum of all samples.
    """

    def __init__(self):
        self._buf = bytearray()

    def writable(self) -> bool:
        return True

    def write(self, data) -> int:
        self._buf.extend(data)
        return len(data)

    def drain(self) -> bytes:
        data = bytes(self._buf)
        self._buf.clear()
        return data


def _slug(text: str) -> str:
    """Filename-safe slug; preserves case but strips path/odd chars."""
    cleaned = re.sub(r"[^A-Za-z0-9._\- ]+", "", text or "").strip()
    cleaned = re.sub(r"\s+", "-", cleaned)
    return cleaned or "project"


def _bare_filename(path: str) -> str:
    """Strip any directory prefix from a stored sample filename."""
    name = (path or "").replace("\\", "/")
    return name.split("/")[-1] if name else ""


def _unique_filename(taken: set[str], desired: str) -> str:
    """Disambiguate filename collisions inside a split by suffixing -1, -2, …"""
    if desired and desired not in taken:
        taken.add(desired)
        return desired
    stem, dot, ext = (desired or "sample").rpartition(".")
    if not dot:
        stem, ext = desired or "sample", ""
        dot = ""
    i = 1
    while True:
        candidate = f"{stem}-{i}{dot}{ext}" if dot else f"{stem}-{i}"
        if candidate not in taken:
            taken.add(candidate)
            return candidate
        i += 1


def _stem_no_ext(name: str) -> str:
    if "." in name:
        return name[: name.rfind(".")]
    return name


def _build_export_box(box: dict) -> Optional[dict]:
    """Convert canonical {label,x,y,w,h} box → EI export {label,x,y,width,height} ints.

    Returns None if geometry is missing/zero. Labels are passed through
    as-is (empty string when absent — caller may still emit the box).
    """
    if not isinstance(box, dict):
        return None
    try:
        x = float(box.get("x", 0))
        y = float(box.get("y", 0))
        w = float(box.get("w", box.get("width", 0)))
        h = float(box.get("h", box.get("height", 0)))
    except (TypeError, ValueError):
        return None
    if w <= 0 or h <= 0:
        return None
    raw_label = box.get("label")
    label = str(raw_label).strip() if raw_label is not None else ""
    return {
        "label": label,
        "x": int(round(x)),
        "y": int(round(y)),
        "width": int(round(w)),
        "height": int(round(h)),
    }


def _build_sample_export_record(
    arc_filename: str,
    category: str,
    label_name: Optional[str],
    raw_boxes: object,
    is_background: bool = False,
) -> tuple[dict, list[dict]]:
    """Build the info.labels entry + box list for one sample.

    The classification label prefers the assigned Sample.label; falls back
    to the first non-empty bounding-box label; final fallback is the
    sample's basename (without extension), matching how the CLI uploader
    treats files with no sidecar.

    ``is_background`` samples emit an explicit ``boundingBoxes: []``.  That is
    EI's own way of saying "this image was annotated and contains nothing", as
    opposed to omitting the key, which says "this image was never annotated".
    Without it the export loses every training negative and the round-trip is
    lossy in one direction.
    """
    boxes: list[dict] = []
    if isinstance(raw_boxes, list):
        for raw in raw_boxes:
            built = _build_export_box(raw)
            if built is not None:
                boxes.append(built)

    if label_name:
        classification_label = label_name
    else:
        bbox_label = next((b["label"] for b in boxes if b["label"]), "")
        classification_label = bbox_label or _stem_no_ext(arc_filename)

    record = {
        "path": f"{category}/{arc_filename}",
        "name": _stem_no_ext(arc_filename),
        "category": category,
        "label": {"type": "label", "label": classification_label},
    }
    if boxes:
        record["boundingBoxes"] = [dict(b) for b in boxes]
    elif is_background:
        record["boundingBoxes"] = []
    return record, boxes


@router.post("/{project_id}/export")
def export_project_dataset(
    project_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Stream the project's dataset as a ZIP in Edge Impulse uploader format.

    Layout:
        README.txt
        info.labels                     ← master index covering all samples
        training/
            <image files>
            info.labels                 ← subset, paths relative to this folder
            bounding_boxes.labels       ← only if any sample has boxes
        testing/
            <image files>
            info.labels
            bounding_boxes.labels
        postprocessing/
            <image/video files>
            info.labels
            bounding_boxes.labels       ← only if any sample has boxes

    Streamed via zipfile + a draining buffer so dataset size is bounded by
    per-file memory, not by the total dataset size.
    """
    project = (
        db.query(Project)
        .filter(Project.id == project_id, Project.owner_id == current_user.id)
        .first()
    )
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    owner = (
        db.query(User).filter(User.id == project.owner_id).first()
        if project.owner_id else None
    )
    owner_name = (owner.username if owner else "unknown")

    # Group samples by split. Treat anything that isn't explicitly "testing"
    # as training — covers "training", "automatic" (already resolved at
    # ingest time), and any future enum values.
    #
    # Materialize every field the streaming generator needs *now*. The DB
    # session is closed by FastAPI's dependency teardown the moment this
    # handler returns, so anything the StreamingResponse generator touches
    # later via a lazy relationship would raise DetachedInstanceError.
    from sqlalchemy.orm import joinedload

    samples = (
        db.query(Sample)
        .options(joinedload(Sample.label))
        .filter(Sample.project_id == project_id)
        .order_by(Sample.created_at.asc(), Sample.id.asc())
        .all()
    )

    grouped: dict[str, list[dict]] = {"training": [], "testing": [], "postprocessing": []}
    seen_per_split: dict[str, set[str]] = {
        "training": set(), "testing": set(), "postprocessing": set(),
    }
    has_any_boxes = False
    for s in samples:
        if s.sample_type == SampleType.postprocessing:
            category = "postprocessing"
        elif s.sample_type == SampleType.testing:
            category = "testing"
        else:
            category = "training"
        bare = _bare_filename(s.filename) or s.id
        arc = _unique_filename(seen_per_split[category], bare)
        meta = s.extra_metadata or {}
        raw_boxes = meta.get("boundingBoxes") if isinstance(meta, dict) else None
        _is_bg = bool(meta.get(IS_BACKGROUND_KEY)) if isinstance(meta, dict) else False
        if isinstance(raw_boxes, list) and any(_build_export_box(b) for b in raw_boxes):
            has_any_boxes = True
        elif _is_bg:
            # A background sample only exists in a detection project, so the
            # bounding_boxes.labels sidecar belongs in the archive even when the
            # negatives are the only annotated samples present.
            has_any_boxes = True
        grouped[category].append({
            "storage_key": s.storage_key,
            "arc_filename": arc,
            "label_name": s.label.name if s.label else None,
            "raw_boxes": raw_boxes,
            "is_background": _is_bg,
            "sample_id": s.id,
        })

    readme_text = (
        f"Exported dataset for {owner_name} / {project.name}.\n\n"
        "To re-import, run the CLI uploader pointing at info.labels, "
        "or upload the files via Data acquisition > Upload data.\n"
    )

    def iter_zip():
        buf = _ZipStreamBuffer()
        with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
            # README first so partial downloads still identify themselves.
            zf.writestr("README.txt", readme_text)
            yield buf.drain()

            root_entries: list[dict] = []
            per_split_entries: dict[str, list[dict]] = {
                "training": [], "testing": [], "postprocessing": [],
            }
            per_split_boxes: dict[str, dict[str, list[dict]]] = {
                "training": {}, "testing": {}, "postprocessing": {},
            }

            for category in ("training", "testing", "postprocessing"):
                for entry in grouped[category]:
                    arc_name = entry["arc_filename"]
                    try:
                        data = storage.download_bytes(entry["storage_key"])
                    except Exception:
                        logger.warning(
                            "export: skipping sample %s — storage fetch failed",
                            entry["sample_id"],
                        )
                        continue

                    zf.writestr(f"{category}/{arc_name}", data)
                    yield buf.drain()

                    record, boxes = _build_sample_export_record(
                        arc_name, category, entry["label_name"], entry["raw_boxes"],
                        is_background=entry["is_background"],
                    )
                    root_entries.append(record)

                    split_record = dict(record)
                    split_record["path"] = arc_name  # no folder prefix inside split
                    per_split_entries[category].append(split_record)

                    if boxes or entry["is_background"]:
                        per_split_boxes[category][arc_name] = boxes

            # Master index covering ALL samples.
            zf.writestr(
                "info.labels",
                json.dumps({"version": 1, "files": root_entries}, indent=2),
            )
            yield buf.drain()

            # Per-split info.labels and (optional) bounding_boxes.labels.
            for category in ("training", "testing", "postprocessing"):
                zf.writestr(
                    f"{category}/info.labels",
                    json.dumps(
                        {"version": 1, "files": per_split_entries[category]},
                        indent=2,
                    ),
                )
                yield buf.drain()

                if has_any_boxes:
                    zf.writestr(
                        f"{category}/bounding_boxes.labels",
                        json.dumps(
                            {
                                "version": 1,
                                "type": "bounding-boxes",
                                "boundingBoxes": per_split_boxes[category],
                            },
                            indent=2,
                        ),
                    )
                    yield buf.drain()

        # ZipFile finalises the central directory on close — flush the tail.
        yield buf.drain()

    download_name = f"{_slug(project.name)}-export.zip"
    headers = {
        "Content-Disposition": f'attachment; filename="{download_name}"',
        # Expose to browser JS so the UI can pick up the filename from the response.
        "Access-Control-Expose-Headers": "Content-Disposition",
    }
    return StreamingResponse(
        iter_zip(),
        media_type="application/zip",
        headers=headers,
    )
