"""
Versioning (C2) endpoints — manual, user-initiated snapshots of a whole
project.

See docs/Action/parityfix.md for the full design — parityfix.md §1.1: "a
version is a snapshot of the whole project, not of a single impulse." Phase 1
covers store + list + detail (the read path). Phase 2 adds restore
(parityfix.md §4.3, §5). Phase 3 (this file also now includes) adds compare
and publish (parityfix.md §8 Phase 3).

A version is created exclusively by POST /project-versions — no worker,
hook, or training-completion path ever calls it (parityfix.md §1.0, §4).
Snapshot creation is synchronous (a handful of INSERTs over already-in-memory
JSON columns plus a couple of SELECTs), not a Celery job — see parityfix.md
§4.1.
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import func
from sqlalchemy.orm import Session, joinedload

from app.core.database import get_db
from app.core.auth import get_current_user
from app.core.authz import assert_project_owner, assert_project_version_owner
from app.core.storage import storage
from app.models.user import (
    User,
    Project,
    Impulse,
    ProjectVersion,
    ProjectVersionImpulse,
    ProjectVersionSample,
    Sample,
    TrainingJob,
    TrainedModel,
    Deployment,
    PostProcessingSettings,
    JobStatus,
)
from app.services.project_clone import (
    CloneError,
    CloneResult,
    CloneScope,
    audit_clone,
    clone_project_graph,
    discard_clone,
)
from app.schemas.project_versions import (
    ProjectVersionCreate,
    ProjectVersionSummary,
    ProjectVersionDetail,
    VersionImpulseOut,
    VersionSampleOut,
    VersionDiffSummary,
    VersionImpulseDiff,
    VersionRestoreRequest,
    VersionRestoreResponse,
    VersionCompareResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter()


# ─── Snapshot helpers ───────────────────────────────────────────────────────

# Mirrors the "Input config" grouping in parityfix.md §2.2.
_INPUT_CONFIG_FIELDS = [
    "window_size_ms", "window_increase_ms", "frequency_hz", "zero_pad_allowed",
    "input_type", "input_axes", "sensor_type",
    "image_width", "image_height", "resize_mode", "train_subset_percent",
]

_PROJECT_CONFIG_FIELDS = ["name", "description", "project_type"]

_DEPLOYMENT_PROJECT_FIELDS = [
    "target_device_slug", "target_device_custom_name",
    "target_device_ram_kb", "target_device_rom_kb", "target_device_latency_ms",
]


def _input_config_snapshot(impulse: Impulse) -> dict:
    return {field: getattr(impulse, field) for field in _INPUT_CONFIG_FIELDS}


def _project_config_snapshot(project: Project) -> dict:
    return {field: getattr(project, field) for field in _PROJECT_CONFIG_FIELDS}


def _resolve_active_trained_model(db: Session, impulse: Impulse):
    """Mirrors training.py's active-model resolution: `active_model_run_id`
    points at a TrainingJob, not directly at a TrainedModel. Prefer the
    tflite artifact — the format every other panel treats as canonical —
    falling back to whatever format exists for that job."""
    if not impulse.active_model_run_id:
        return None, None
    job = db.query(TrainingJob).filter(TrainingJob.id == impulse.active_model_run_id).first()
    if job is None or job.status != JobStatus.completed:
        return None, None
    model = (
        db.query(TrainedModel)
        .filter(TrainedModel.training_job_id == job.id, TrainedModel.format == "tflite")
        .first()
    )
    if model is None:
        model = db.query(TrainedModel).filter(TrainedModel.training_job_id == job.id).first()
    return job, model


_TRAINING_CONFIG_FIELDS = [
    "epochs", "learning_rate", "batch_size", "validation_split",
    "optimizer", "device_type", "extra_params", "run_kind", "status",
]


def _training_config_snapshot(job: TrainingJob) -> dict:
    """The hyperparameters that produced `job`'s results — captured
    alongside the metrics (parityfix.md §2.2, migration 0038) so restore
    never has to reach into the live TrainingJob row to reconstruct a
    working, retrainable impulse."""
    snap = {f: getattr(job, f) for f in _TRAINING_CONFIG_FIELDS}
    snap["status"] = str(getattr(snap["status"], "value", snap["status"]))
    snap["started_at"] = job.started_at.isoformat() if job.started_at else None
    snap["completed_at"] = job.completed_at.isoformat() if job.completed_at else None
    return snap


def _trained_model_formats_snapshot(db: Session, job: TrainingJob) -> List[dict]:
    """Every TrainedModel row `job` produced (keras/tflite/onnx/...), not
    just the single tflite-preferred row `trained_model_id` points at — so a
    restored impulse has every format a live one would (migration 0038)."""
    models = db.query(TrainedModel).filter(TrainedModel.training_job_id == job.id).all()
    return [
        {
            "format": m.format,
            "version": m.version,
            "storage_key": m.storage_key,
            "file_size_bytes": m.file_size_bytes,
            "model_metadata": m.model_metadata,
        }
        for m in models
    ]


def _features_storage_key_if_exists(project_id: str, impulse_id: str) -> Optional[str]:
    """The impulse's `features.npz` DSP feature-cache S3 key, if one exists
    right now — a cheap existence check (HEAD), not a download. Recorded on
    the snapshot (migration 0038) so restore can clone + remap it instead of
    leaving the restored impulse's Feature Explorer empty."""
    key = storage.model_key(project_id, impulse_id, "features.npz")
    try:
        storage.client.head_object(Bucket=storage.bucket, Key=key)
        return key
    except Exception:
        return None


def _latest_deployment_for_impulse(db: Session, impulse_id: str) -> Optional[Deployment]:
    return (
        db.query(Deployment)
        .join(TrainedModel, Deployment.model_id == TrainedModel.id)
        .join(TrainingJob, TrainedModel.training_job_id == TrainingJob.id)
        .filter(TrainingJob.impulse_id == impulse_id)
        .order_by(Deployment.created_at.desc())
        .first()
    )


def _deployment_snapshot(db: Session, project: Project, impulses: List[Impulse]) -> dict:
    """parityfix.md §2.1 — target-device fields plus the last-used deployment
    target/profile/options per impulse."""
    per_impulse: Dict[str, Any] = {}
    for impulse in impulses:
        dep = _latest_deployment_for_impulse(db, impulse.id)
        if dep is not None:
            per_impulse[impulse.id] = {
                "deployment_target": dep.deployment_target,
                "device_profile": dep.device_profile,
                "options": dep.options,
            }
    snapshot = {field: getattr(project, field) for field in _DEPLOYMENT_PROJECT_FIELDS}
    snapshot["impulses"] = per_impulse
    return snapshot


def _post_processing_snapshot(db: Session, project_id: str) -> Optional[List[dict]]:
    """Every PostProcessingSettings row for the project — the project-scoped
    row (impulse_id NULL) and any impulse-scoped rows (parityfix.md §2.1)."""
    rows = (
        db.query(PostProcessingSettings)
        .filter(PostProcessingSettings.project_id == project_id)
        .all()
    )
    if not rows:
        return None
    return [
        {
            "impulse_id": pp.impulse_id,
            "enabled": pp.enabled,
            "threshold": pp.threshold,
            "tracking_enabled": pp.tracking_enabled,
            "keep_grace": pp.keep_grace,
            "max_observations": pp.max_observations,
            "class_filter": pp.class_filter,
        }
        for pp in rows
    ]


def _live_dataset_state(db: Session, project_id: str):
    """Current sample_count/train/test/class_names for a project — the same
    shape captured in a snapshot, computed fresh. Used both for creating a
    new version and for diffing an existing one against live state."""
    samples = (
        db.query(Sample)
        .options(joinedload(Sample.label))
        .filter(Sample.project_id == project_id)
        .all()
    )
    test_count = sum(1 for s in samples if str(getattr(s.sample_type, "value", s.sample_type)) == "testing")
    class_names = sorted({s.label.name for s in samples if s.label is not None})
    return {
        "samples": samples,
        "sample_count": len(samples),
        "train_sample_count": len(samples) - test_count,
        "test_sample_count": test_count,
        "class_names": class_names,
    }


def _next_version_number(db: Session, project_id: str) -> int:
    current_max = (
        db.query(func.max(ProjectVersion.version_number))
        .filter(ProjectVersion.project_id == project_id)
        .scalar()
    )
    return (current_max or 0) + 1


def _job_accuracy(job: Optional[TrainingJob]) -> Optional[float]:
    if job is None:
        return None
    return job.final_accuracy if job.final_accuracy is not None else job.best_accuracy


# ─── Diff computation (parityfix.md §4.5) ────────────────────────────────────

def _build_diff_summary(db: Session, version: ProjectVersion, project: Project) -> VersionDiffSummary:
    live_config = _project_config_snapshot(project)
    changed_project_fields = [
        f for f in _PROJECT_CONFIG_FIELDS
        if (version.project_config_snapshot or {}).get(f) != live_config.get(f)
    ]

    live_impulses = db.query(Impulse).filter(Impulse.project_id == project.id).all()
    live_deployment = _deployment_snapshot(db, project, live_impulses)
    deployment_changed = (version.deployment_snapshot or {}) != live_deployment

    live = _live_dataset_state(db, project.id)
    live_class_set = set(live["class_names"])
    version_class_set = set(version.class_names or [])

    live_by_id = {imp.id: imp for imp in live_impulses}
    version_impulses = (
        db.query(ProjectVersionImpulse)
        .filter(ProjectVersionImpulse.version_id == version.id)
        .all()
    )
    matched_live_ids = set()
    impulse_diffs: List[VersionImpulseDiff] = []
    live_accuracies: List[float] = []

    for vi in version_impulses:
        live_imp = live_by_id.get(vi.impulse_id) if vi.impulse_id else None
        if live_imp is None:
            impulse_diffs.append(VersionImpulseDiff(
                impulse_name=vi.impulse_name, impulse_id=vi.impulse_id, status="deleted",
            ))
            continue
        matched_live_ids.add(live_imp.id)
        changed_fields = []
        if vi.dsp_blocks_snapshot != (live_imp.dsp_blocks or []):
            changed_fields.append("dsp_blocks")
        if vi.ml_blocks_snapshot != (live_imp.ml_blocks or []):
            changed_fields.append("ml_blocks")
        if vi.output_config_snapshot != (live_imp.output_config or {}):
            changed_fields.append("output_config")
        if vi.input_config_snapshot != _input_config_snapshot(live_imp):
            changed_fields.append("input_config")

        live_job, _ = _resolve_active_trained_model(db, live_imp)
        live_accuracy = _job_accuracy(live_job)
        if live_accuracy is not None:
            live_accuracies.append(live_accuracy)
        accuracy_delta = (
            live_accuracy - vi.accuracy
            if live_accuracy is not None and vi.accuracy is not None
            else None
        )
        impulse_diffs.append(VersionImpulseDiff(
            impulse_name=vi.impulse_name,
            impulse_id=vi.impulse_id,
            status="changed" if changed_fields else "unchanged",
            changed_fields=changed_fields,
            accuracy_delta=accuracy_delta,
            live_accuracy=live_accuracy,
        ))

    for imp in live_impulses:
        if imp.id in matched_live_ids:
            continue
        live_job, _ = _resolve_active_trained_model(db, imp)
        live_accuracy = _job_accuracy(live_job)
        if live_accuracy is not None:
            live_accuracies.append(live_accuracy)
        impulse_diffs.append(VersionImpulseDiff(
            impulse_name=imp.name, impulse_id=imp.id, status="added_since",
            live_accuracy=live_accuracy,
        ))

    live_best_accuracy = max(live_accuracies) if live_accuracies else None
    best_accuracy_delta = (
        live_best_accuracy - version.best_accuracy
        if live_best_accuracy is not None and version.best_accuracy is not None
        else None
    )

    return VersionDiffSummary(
        project_config_changed=len(changed_project_fields) > 0,
        changed_project_fields=changed_project_fields,
        deployment_changed=deployment_changed,
        sample_count_delta=live["sample_count"] - version.sample_count,
        class_names_added=sorted(live_class_set - version_class_set),
        class_names_removed=sorted(version_class_set - live_class_set),
        impulses=impulse_diffs,
        best_accuracy_delta=best_accuracy_delta,
        live_best_accuracy=live_best_accuracy,
    )


def _build_detail(
    db: Session, version: ProjectVersion, project: Project, page: int, page_size: int,
) -> ProjectVersionDetail:
    samples_total = (
        db.query(func.count(ProjectVersionSample.id))
        .filter(ProjectVersionSample.version_id == version.id)
        .scalar()
    ) or 0
    sample_rows = (
        db.query(ProjectVersionSample)
        .filter(ProjectVersionSample.version_id == version.id)
        .order_by(ProjectVersionSample.sample_name)
        .offset((page - 1) * page_size)
        .limit(page_size)
        .all()
    )
    impulse_rows = (
        db.query(ProjectVersionImpulse)
        .filter(ProjectVersionImpulse.version_id == version.id)
        .order_by(ProjectVersionImpulse.impulse_name)
        .all()
    )

    return ProjectVersionDetail(
        **_summary_kwargs(version),
        project_config_snapshot=version.project_config_snapshot,
        deployment_snapshot=version.deployment_snapshot,
        post_processing_snapshot=version.post_processing_snapshot,
        impulses=[VersionImpulseOut.model_validate(i) for i in impulse_rows],
        samples=[VersionSampleOut.model_validate(r) for r in sample_rows],
        samples_total=samples_total,
        samples_page=page,
        samples_page_size=page_size,
        diff_summary=_build_diff_summary(db, version, project),
    )


def _summary_kwargs(version: ProjectVersion) -> dict:
    data = ProjectVersionSummary.model_validate(version).model_dump()
    data["created_by_name"] = version.creator.username if version.creator else None
    return data

# ─── Restore (parityfix.md §4.3, §5 — Phase 2 redesign) ──────────────────────
#
# Restore duplicates a trained project; it does not rebuild one. The whole
# graph is cloned from the *live* source project by
# `app.services.project_clone` — every table, every storage asset, every
# foreign key remapped — and the version's snapshot is then laid over the
# result to pin the clone to the state the version recorded. See that
# module's docstring for the five steps and for the tables deliberately left
# out of the clone.
#
# The snapshot's role changed with this redesign. It no longer *is* the
# restored project (a snapshot only ever held block config, metrics and a
# manifest, which is why the old restore produced something that merely
# resembled a trained project — no dataset annotations of its own, one
# synthesised training run, no test history, assets borrowed from the source
# project's storage). It now does two narrower jobs: it defines the clone's
# scope — which impulses and samples existed when the version was taken — and
# it supplies the configuration overlay for those impulses, so restoring v1
# of a project that has since been re-tuned gives back v1's blocks rather
# than today's.


class RestoreUnavailable(HTTPException):
    """The version cannot be restored because part of the source project it
    describes no longer exists. A 409, not a 404: the version row is fine,
    the world it points at has moved."""

    def __init__(self, detail: str):
        super().__init__(status_code=409, detail=detail)


def _restore_scope(db: Session, version: ProjectVersion) -> tuple[CloneScope, List[str]]:
    """Resolve the version into a clone scope over the live source project.

    Impulses and samples are treated differently on purpose, because their
    failure modes are different:

      * A deleted **impulse** aborts the restore. Its training jobs, trained
        models, feature rows and test runs were cascade-deleted with it, so
        the only thing left to build from is the snapshot's block config and
        metrics — a reconstruction, which is precisely what this redesign
        removed. Failing loudly beats handing back a project that looks
        trained and cannot retrain.
      * A deleted **sample** is skipped and counted. ``DELETE /samples/{id}``
        removes the underlying object from storage too, so the media is gone
        in the strong sense: there is nothing any implementation could clone.
        The restore reports how many were dropped rather than pretending the
        dataset came back whole.
    """
    source = db.query(Project).filter(Project.id == version.project_id).first()
    if source is None:
        raise RestoreUnavailable(
            "The project this version belongs to no longer exists, so there is "
            "nothing to restore from."
        )

    impulse_rows = (
        db.query(ProjectVersionImpulse)
        .filter(ProjectVersionImpulse.version_id == version.id)
        .all()
    )
    live_impulse_ids = {
        row[0] for row in db.query(Impulse.id).filter(Impulse.project_id == source.id).all()
    }
    missing_impulses = sorted(
        vi.impulse_name for vi in impulse_rows
        if not vi.impulse_id or vi.impulse_id not in live_impulse_ids
    )
    if missing_impulses:
        raise RestoreUnavailable(
            "This version cannot be restored: "
            + ", ".join(f"impulse {name!r}" for name in missing_impulses)
            + " no longer exists in the source project, and its training runs "
              "were deleted with it."
        )

    sample_rows = (
        db.query(ProjectVersionSample)
        .filter(ProjectVersionSample.version_id == version.id)
        .all()
    )
    manifest_sample_ids = {r.sample_id for r in sample_rows if r.sample_id}
    live_sample_ids = {
        row[0] for row in db.query(Sample.id).filter(
            Sample.project_id == source.id,
            Sample.id.in_(manifest_sample_ids),
        ).all()
    } if manifest_sample_ids else set()
    missing_samples = [
        r.sample_name for r in sample_rows
        if not r.sample_id or r.sample_id not in live_sample_ids
    ]

    scope = CloneScope(
        project=source,
        impulse_ids={vi.impulse_id for vi in impulse_rows if vi.impulse_id},
        sample_ids=live_sample_ids,
    )
    return scope, missing_samples


def _apply_version_overlay(
    db: Session, version: ProjectVersion, result: CloneResult,
) -> None:
    """Pin the freshly cloned graph to the state the version recorded.

    The clone reproduces the source project as it stands *today*. Everything
    the snapshot captured is written over that, so restoring an older version
    gives back that version's configuration rather than the current one. Only
    fields the snapshot actually holds are overlaid — anything it never
    captured keeps its cloned (live) value, which is strictly better than the
    NULL the old reconstruct-from-snapshot path left behind.
    """
    id_map = result.id_map
    project = result.project

    config = version.project_config_snapshot or {}
    if config.get("project_type"):
        project.project_type = config["project_type"]

    deployment = version.deployment_snapshot or {}
    for column in _DEPLOYMENT_PROJECT_FIELDS:
        if column in deployment:
            setattr(project, column, deployment[column])

    version_impulses = (
        db.query(ProjectVersionImpulse)
        .filter(ProjectVersionImpulse.version_id == version.id)
        .all()
    )
    for vi in version_impulses:
        new_impulse_id = id_map.get(vi.impulse_id or "")
        if not new_impulse_id:
            continue
        impulse = db.query(Impulse).filter(Impulse.id == new_impulse_id).first()
        if impulse is None:
            continue

        impulse.name = vi.impulse_name
        impulse.dsp_blocks = vi.dsp_blocks_snapshot or []
        impulse.ml_blocks = vi.ml_blocks_snapshot or []
        impulse.output_config = vi.output_config_snapshot or {}
        for field_name, value in (vi.input_config_snapshot or {}).items():
            if field_name in _INPUT_CONFIG_FIELDS:
                setattr(impulse, field_name, value)

        # The active-model pointer names the run the version was taken
        # against, not whatever has been trained since. `training_job_id` is
        # a source id; its clone is in the map because the job belongs to an
        # in-scope impulse.
        if vi.training_job_id:
            cloned_job_id = id_map.get(vi.training_job_id)
            if cloned_job_id:
                impulse.active_model_run_id = cloned_job_id

    # Post-processing settings as the snapshot recorded them, re-keyed onto
    # the cloned impulses. Rows the clone produced for impulses the snapshot
    # has no entry for keep their cloned values.
    for snap in (version.post_processing_snapshot or []):
        snap_impulse_id = snap.get("impulse_id")
        if snap_impulse_id is None:
            new_impulse_id = None
        else:
            new_impulse_id = id_map.get(snap_impulse_id)
            if new_impulse_id is None:
                continue
        row = (
            db.query(PostProcessingSettings)
            .filter(
                PostProcessingSettings.project_id == project.id,
                PostProcessingSettings.impulse_id == new_impulse_id,
            )
            .first()
        )
        if row is None:
            row = PostProcessingSettings(project_id=project.id, impulse_id=new_impulse_id)
            db.add(row)
        for column in ("enabled", "threshold", "tracking_enabled", "keep_grace", "max_observations"):
            if column in snap:
                setattr(row, column, snap[column])
        # `class_filter` is a list of label ids as they existed when the
        # snapshot was taken — a soft FK, not a plain value, so it has to go
        # through the same id map as everything else. An id no longer in the
        # map names a label that no longer exists (deleted since the
        # snapshot) and is dropped rather than left pointing at the source
        # project's label table.
        if "class_filter" in snap:
            row.class_filter = [
                id_map[old_id] for old_id in (snap["class_filter"] or []) if old_id in id_map
            ]

    db.flush()


def _assert_name_available(db: Session, owner_id: str, name: str) -> str:
    """Same per-owner name-uniqueness rule ``POST /projects`` enforces — a
    restored project is created through the same door as any other."""
    name = (name or "").strip()
    if not name:
        raise HTTPException(400, "Project name cannot be empty")
    clash = (
        db.query(Project)
        .filter(Project.owner_id == owner_id, Project.name == name)
        .first()
    )
    if clash:
        raise HTTPException(409, "A project with this name already exists")
    return name


# ─── Compare (parityfix.md §4.5, §7.3, Phase 3) ─────────────────────────────

def _build_compare(
    db: Session, version_a: ProjectVersion, version_b: ProjectVersion,
) -> VersionCompareResponse:
    """Compact per-category summary of two stored project versions
    (parityfix.md §4.5). Impulses are still matched by `impulse_name`
    server-side — two versions can carry different impulse ids for what is
    logically the same impulse — but only the resulting counts are returned,
    not a per-impulse array; the compare dialog shows a summary, not a
    listing."""
    config_a = version_a.project_config_snapshot or {}
    config_b = version_b.project_config_snapshot or {}
    project_config_same = all(config_a.get(f) == config_b.get(f) for f in _PROJECT_CONFIG_FIELDS)
    deployment_same = (version_a.deployment_snapshot or {}) == (version_b.deployment_snapshot or {})

    impulses_a = {
        vi.impulse_name: vi
        for vi in db.query(ProjectVersionImpulse).filter(ProjectVersionImpulse.version_id == version_a.id).all()
    }
    impulses_b = {
        vi.impulse_name: vi
        for vi in db.query(ProjectVersionImpulse).filter(ProjectVersionImpulse.version_id == version_b.id).all()
    }

    added = removed = modified = 0
    for name in set(impulses_a) | set(impulses_b):
        vi_a = impulses_a.get(name)
        vi_b = impulses_b.get(name)
        if vi_a is None:
            added += 1
        elif vi_b is None:
            removed += 1
        elif (
            vi_a.dsp_blocks_snapshot != vi_b.dsp_blocks_snapshot
            or vi_a.ml_blocks_snapshot != vi_b.ml_blocks_snapshot
            or vi_a.output_config_snapshot != vi_b.output_config_snapshot
            or vi_a.input_config_snapshot != vi_b.input_config_snapshot
        ):
            modified += 1

    return VersionCompareResponse(
        version_a=ProjectVersionSummary(**_summary_kwargs(version_a)),
        version_b=ProjectVersionSummary(**_summary_kwargs(version_b)),
        project_config_same=project_config_same,
        deployment_same=deployment_same,
        sample_count_a=version_a.sample_count,
        sample_count_b=version_b.sample_count,
        class_count_a=len(version_a.class_names or []),
        class_count_b=len(version_b.class_names or []),
        impulse_total_a=len(impulses_a),
        impulse_total_b=len(impulses_b),
        impulses_added=added,
        impulses_removed=removed,
        impulses_modified=modified,
    )


# ─── Routes ───────────────────────────────────────────────────────────────────

@router.post("", response_model=ProjectVersionDetail)
def create_version(
    req: ProjectVersionCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Store a snapshot of the whole project's current state: project config,
    every impulse's block config + active trained model + metrics,
    deployment settings, post-processing settings, and the dataset manifest.
    The sole caller is the Store Project Version button — this endpoint is
    never invoked automatically (parityfix.md §1.0)."""
    project = assert_project_owner(db, req.project_id, current_user)

    impulses = db.query(Impulse).filter(Impulse.project_id == project.id).all()
    dataset = _live_dataset_state(db, project.id)

    accuracies: List[float] = []
    impulse_rows: List[ProjectVersionImpulse] = []
    for impulse in impulses:
        job, model = _resolve_active_trained_model(db, impulse)
        accuracy = _job_accuracy(job)
        if accuracy is not None:
            accuracies.append(accuracy)
        impulse_rows.append(ProjectVersionImpulse(
            impulse_id=impulse.id,
            impulse_name=impulse.name,
            dsp_blocks_snapshot=impulse.dsp_blocks or [],
            ml_blocks_snapshot=impulse.ml_blocks or [],
            output_config_snapshot=impulse.output_config or {},
            input_config_snapshot=_input_config_snapshot(impulse),
            trained_model_id=model.id if model else None,
            training_job_id=job.id if job else None,
            accuracy=accuracy,
            final_loss=job.best_loss if job else None,
            confusion_matrix=job.confusion_matrix if job else None,
            training_history=job.training_history if job else None,
            training_config_snapshot=_training_config_snapshot(job) if job else None,
            trained_model_formats_snapshot=_trained_model_formats_snapshot(db, job) if job else None,
            features_storage_key=_features_storage_key_if_exists(project.id, impulse.id),
        ))

    version = ProjectVersion(
        project_id=project.id,
        version_number=_next_version_number(db, project.id),
        name=req.name,
        description=req.description,
        project_config_snapshot=_project_config_snapshot(project),
        deployment_snapshot=_deployment_snapshot(db, project, impulses),
        post_processing_snapshot=_post_processing_snapshot(db, project.id),
        sample_count=dataset["sample_count"],
        train_sample_count=dataset["train_sample_count"],
        test_sample_count=dataset["test_sample_count"],
        class_names=dataset["class_names"],
        impulse_count=len(impulses),
        best_accuracy=max(accuracies) if accuracies else None,
        status="draft",
        created_by=current_user.id,
    )
    db.add(version)
    db.flush()  # assign version.id before inserting child rows

    for row in impulse_rows:
        row.version_id = version.id
        db.add(row)

    for sample in dataset["samples"]:
        db.add(ProjectVersionSample(
            version_id=version.id,
            sample_id=sample.id,
            sample_name=sample.filename,
            label_name=sample.label.name if sample.label else None,
            sample_type=str(getattr(sample.sample_type, "value", sample.sample_type)),
        ))

    db.commit()
    db.refresh(version)
    return _build_detail(db, version, project, page=1, page_size=50)


@router.get("", response_model=List[ProjectVersionSummary])
def list_versions(
    project_id: str = Query(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """List versions for a project, newest first. Header fields only — the
    same list regardless of which impulse is selected in the sidebar
    (parityfix.md §7.1), since this endpoint takes no impulse parameter."""
    assert_project_owner(db, project_id, current_user)
    versions = (
        db.query(ProjectVersion)
        .filter(ProjectVersion.project_id == project_id)
        .order_by(ProjectVersion.version_number.desc())
        .all()
    )
    return [ProjectVersionSummary(**_summary_kwargs(v)) for v in versions]


@router.get("/{version_id}", response_model=ProjectVersionDetail)
def get_version(
    version_id: str,
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=500),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Full detail: header fields + per-impulse rows + paginated sample
    manifest + a diff summary against the project's current live state."""
    version = assert_project_version_owner(db, version_id, current_user)
    project = db.query(Project).filter(Project.id == version.project_id).first()
    if project is None:
        raise HTTPException(status_code=404, detail="Project for this version no longer exists")
    return _build_detail(db, version, project, page=page, page_size=page_size)


@router.post("/{version_id}/restore", response_model=VersionRestoreResponse)
def restore_version(
    version_id: str,
    req: VersionRestoreRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Create a new project that is a full clone of the project this version
    was taken from, pinned to the version's configuration. The source project
    is never modified.

    One transaction, five steps (``app.services.project_clone``):

    1. A new Project row, created the same way ``POST /projects`` creates one
       — same name rule, same owner, fresh id.
    2. The entire project graph cloned: labels, samples and their
       annotations, impulses, every training job and trained model, DSP
       feature rows and feature-generation jobs, deployments, post-processing
       settings, video processing jobs, AI-labeling actions/jobs/predictions,
       and the whole Model Testing / Model Performance set.
    3. Every storage asset copied to a key under the new project's own
       prefix, including the ``features.npz`` DSP cache — whose embedded
       sample/label/impulse/project ids are rewritten, since it is a table in
       a binary wrapper, not opaque bytes.
    4. Every foreign key, every id embedded in a JSON column, and every id
       embedded in a storage key remapped through one old→new id map.
    5. An integrity audit over the written graph. If anything fails to
       resolve, still names the source project, or shows an impulse that lost
       its training or feature state, the transaction is rolled back and the
       copied objects are deleted — the caller gets an error, never a
       half-restored project.

    The version's snapshot is then laid over the clone, so the restored
    project carries the block configuration, active-model pointer and
    post-processing settings the version recorded rather than whatever the
    source project looks like now.
    """
    version = assert_project_version_owner(db, version_id, current_user)
    name = _assert_name_available(db, current_user.id, req.name)
    scope, missing_samples = _restore_scope(db, version)

    result: Optional[CloneResult] = None
    try:
        result = clone_project_graph(
            db, scope,
            owner_id=current_user.id,
            name=name,
            description=req.description,
        )
        _apply_version_overlay(db, version, result)

        failures = audit_clone(db, scope, result)
        if failures:
            raise CloneError("; ".join(failures[:10]))

        db.commit()
    except HTTPException:
        discard_clone(db, result.copied_keys if result else [])
        raise
    except CloneError as exc:
        discard_clone(db, result.copied_keys if result else [])
        logger.error("Restore of version %s failed its integrity audit: %s", version_id, exc)
        raise HTTPException(
            500,
            "The restored project failed its integrity check and was discarded. "
            "The original project was not modified.",
        )
    except Exception as exc:  # noqa: BLE001 - every failure must leave nothing behind
        discard_clone(db, result.copied_keys if result else [])
        logger.exception("Restore of version %s failed", version_id)
        raise HTTPException(500, f"Could not restore this version: {exc}")

    db.refresh(result.project)
    return VersionRestoreResponse(
        new_project_id=result.project.id,
        new_project_name=result.project.name,
        cloned_row_counts=result.counts,
        cloned_asset_count=len(result.copied_keys),
        skipped_sample_names=missing_samples,
    )


@router.get("/{version_id}/compare/{other_version_id}", response_model=VersionCompareResponse)
def compare_versions(
    version_id: str,
    other_version_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Diff two project versions: project/deployment config field diff,
    per-impulse block-config diff matched by impulse name, impulses added or
    removed between the two, class/sample-count deltas, accuracy deltas
    (parityfix.md §4, Phase 3). Pure read — nothing here is persisted."""
    version_a = assert_project_version_owner(db, version_id, current_user)
    version_b = assert_project_version_owner(db, other_version_id, current_user)
    return _build_compare(db, version_a, version_b)


@router.post("/{version_id}/publish", response_model=ProjectVersionSummary)
def publish_version(
    version_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Mark a project version as the project's official stable version
    (parityfix.md §6, §8 Phase 3). Publish does not restore the project,
    deploy a model, or retrain — it only flags the row. Only one version can
    be published per project at a time; publishing a new one unpublishes
    whichever version held that status before."""
    version = assert_project_version_owner(db, version_id, current_user)
    db.query(ProjectVersion).filter(
        ProjectVersion.project_id == version.project_id,
        ProjectVersion.status == "published",
        ProjectVersion.id != version.id,
    ).update({"status": "draft", "published_at": None})
    version.status = "published"
    version.published_at = datetime.utcnow()
    db.commit()
    db.refresh(version)
    return ProjectVersionSummary(**_summary_kwargs(version))
