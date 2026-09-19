"""
Deployment endpoints — generate and download edge deployment packages
"""
from datetime import datetime, timedelta

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import Optional

from app.core.database import get_db
from app.core.auth import get_current_user
from app.core.authz import assert_trained_model_owner, assert_deployment_owner
from app.core.storage import storage
from app.models.user import User, Deployment, TrainedModel, JobStatus, DeployTarget
from app.services.compatibility import (
    check_compatibility,
    resolve_build_format,
    deployment_format_offer,
    all_package_policies,
)
from app.services.estimation import estimate
from app.workers.celery_app import celery_app
from app.workers.deployment_worker import run_deployment_job
from celery.result import AsyncResult
import redis.exceptions

router = APIRouter()

_STALE_DEPLOYMENT_TIMEOUT = timedelta(minutes=10)


class DeploymentRequest(BaseModel):
    model_config = {"protected_namespaces": ()}
    model_id: str
    target: str  # tflite, arduino, esp32, raspberry_pi, cpp
    device_profile: str | None = None  # e.g. "arduino_nano_33_ble"; routed by worker
    deployment_target: str | None = None
    deployment_format: str = "pe"       # "pe" or "pxe"
    options: dict = {}


class DeploymentResponse(BaseModel):
    model_config = {"protected_namespaces": ()}
    id: str
    model_id: str
    target: str
    status: str
    download_url: Optional[str]
    error_message: Optional[str]
    deployment_target: Optional[str]
    device_profile: Optional[str]
    created_at: str
    completed_at: Optional[str]


def _dep_to_dict(d: Deployment) -> dict:
    return {
        "id": d.id,
        "model_id": d.model_id,
        "target": d.target,
        "status": d.status,
        "download_url": d.download_url,
        "error_message": d.error_message,
        "deployment_target": d.deployment_target,
        "device_profile": d.device_profile,
        "created_at": d.created_at.isoformat(),
        "completed_at": d.completed_at.isoformat() if d.completed_at else None,
        "deployment_format": (d.options or {}).get("deployment_format", "pe"),
        "download_filename": (d.options or {}).get("download_filename"),
    }


def _reconcile_deployment_status(db: Session, dep: Deployment) -> Deployment:
    """Best-effort reconciliation for async deployment rows.

    Deployment builds run in Celery, so if Redis or the worker dies mid-build the
    DB row can remain stuck in pending/running forever. For obviously stale rows,
    surface that as a failed build instead of leaving the UI polling forever.
    """
    if dep.status not in (JobStatus.pending, JobStatus.running):
        return dep

    # Cheap age check first. A row that has already exceeded the stale timeout
    # and produced no artifact can only resolve to "failed", so mark it without a
    # Celery/Redis round-trip. This avoids a Redis call (and its timeout) per row
    # when many orphaned builds pile up after a worker/Redis outage — the list
    # endpoint reconciles every pending/running row. Rows that still carry a
    # storage_key fall through to the Redis check below so a genuinely-finished
    # build can still be recovered as "completed".
    age = datetime.utcnow() - dep.created_at
    if age >= _STALE_DEPLOYMENT_TIMEOUT and not dep.storage_key:
        dep.status = JobStatus.failed
        if not dep.error_message:
            dep.error_message = (
                "Deployment job became stale before completion. "
                "Redis and/or the Celery deployment worker may be offline. "
                "Start Redis and the Celery worker, then rebuild."
            )
        dep.completed_at = dep.completed_at or datetime.utcnow()
        db.commit()
        db.refresh(dep)
        return dep

    task_id = dep.celery_task_id
    task_state = None
    if task_id:
        try:
            task_state = AsyncResult(task_id, app=celery_app).state
        except redis.exceptions.RedisError:
            task_state = None
        except Exception:
            task_state = None

    if task_state == "SUCCESS" and dep.storage_key:
        dep.status = JobStatus.completed
        dep.completed_at = dep.completed_at or datetime.utcnow()
        db.commit()
        db.refresh(dep)
        return dep

    if task_state in {"FAILURE", "REVOKED"}:
        dep.status = JobStatus.failed
        if not dep.error_message:
            dep.error_message = f"Deployment task ended in Celery state: {task_state.lower()}"
        dep.completed_at = dep.completed_at or datetime.utcnow()
        db.commit()
        db.refresh(dep)
        return dep

    age = datetime.utcnow() - dep.created_at
    if age >= _STALE_DEPLOYMENT_TIMEOUT:
        dep.status = JobStatus.failed
        if not dep.error_message:
            dep.error_message = (
                "Deployment job became stale before completion. "
                "Redis and/or the Celery deployment worker may be offline. "
                "Start Redis and the Celery worker, then rebuild."
            )
        dep.completed_at = dep.completed_at or datetime.utcnow()
        db.commit()
        db.refresh(dep)

    return dep


@router.post("/build", status_code=201)
def build_deployment(
    req: DeploymentRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Start building a deployment package for the given target."""
    model = assert_trained_model_owner(db, req.model_id, current_user)

    if req.target not in [t.value for t in DeployTarget]:
        raise HTTPException(status_code=400, detail=f"Invalid target: {req.target}")

    if req.target == "pxe":
        raise HTTPException(
            status_code=400,
            detail="Select a Linux device target like UNO Q or Raspberry Pi and then choose the .pxe package format.",
        )

    # Compatibility gate (Target Device Phase 4). Runs before the Deployment row
    # exists: an incompatible build used to be persisted, dispatched, and only
    # fail minutes later as a `failed` deployment. Same service the worker's
    # `_raise_if_*_embedded` helpers now call, so the answer here and the answer
    # mid-build cannot diverge.
    build_format = resolve_build_format(
        db,
        target=req.target,
        device_profile=req.device_profile,
        deployment_format=req.deployment_format,
    )
    compat = check_compatibility(
        model, build_format, device_slug=req.device_profile, db=db,
    )
    if not compat.compatible:
        raise HTTPException(status_code=400, detail=compat.message)

    # Derive project_id: training worker stores it in model_metadata; fall back
    # to the ORM relationship (training_job → impulse) if metadata is absent.
    project_id = (model.model_metadata or {}).get("project_id")
    if not project_id and model.training_job and model.training_job.impulse:
        project_id = model.training_job.impulse.project_id

    merged_options = {**req.options}
    if req.device_profile:
        merged_options["device_profile"] = req.device_profile
    merged_options["deployment_format"] = req.deployment_format
    dep = Deployment(
        model_id=req.model_id,
        project_id=project_id,
        target=req.target,
        options=merged_options,
        deployment_target=req.deployment_target,
        device_profile=req.device_profile,
    )
    db.add(dep)
    db.commit()
    db.refresh(dep)

    task = run_deployment_job.delay(dep.id)
    dep.celery_task_id = task.id
    db.commit()

    return _dep_to_dict(dep)


@router.get("/compatibility")
def check_deployment_compatibility(
    model_id: str,
    format: str,
    device_profile: str | None = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Read-only twin of the gate in `POST /build` — same function, same answer.

    Lets the deployment UI disable an impossible combination at the moment it is
    selected instead of after a failed build. Nothing is created or enqueued.

    `format` is the requested target, resolved exactly as `POST /build` resolves
    it — `device_profile`'s catalog `deploy_target` wins, and `"pxe"` short-
    circuits. Passing it straight through instead would let the two endpoints
    answer differently for one selection: `format=tflite` with a Raspberry Pi
    profile is a device mismatch unresolved, and a plain `raspberry_pi` build
    resolved, which is what actually runs.

    Declared above `GET /{deployment_id}`: FastAPI matches routes in
    registration order, so the literal path has to win over the parameterised
    one or "compatibility" is read as a deployment id.
    """
    # `model_id` is an ownership surface even for a read: resolve it through
    # authz first so a cross-tenant caller gets 404 rather than learning which
    # model ids exist from an "incompatible" answer.
    model = assert_trained_model_owner(db, model_id, current_user)
    build_format = resolve_build_format(
        db,
        target=format,
        device_profile=device_profile,
        deployment_format="pxe" if format == "pxe" else "pe",
    )
    result = check_compatibility(model, build_format, device_slug=device_profile, db=db)
    return result.to_dict()


@router.get("/estimate")
def estimate_deployment_performance(
    model_id: str,
    format: str,
    device_profile: str | None = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Flash / RAM / latency estimates for this model, built as `format`, on
    `device_profile` (Target Device Phase 5).

    Same shape of read as `GET /deployment/compatibility`: resolves the build
    format exactly as `POST /build` would, runs the Phase 4 compatibility gate
    first, and never mutates or enqueues anything. Declared above
    `GET /{deployment_id}` for the same routing reason `/compatibility` is.
    """
    model = assert_trained_model_owner(db, model_id, current_user)
    build_format = resolve_build_format(
        db,
        target=format,
        device_profile=device_profile,
        deployment_format="pxe" if format == "pxe" else "pe",
    )
    result = estimate(model, build_format, device_slug=device_profile, db=db)
    return result.to_dict()


_STATIC_TARGETS = {
    "targets": [
        {"id": "tflite", "name": "TensorFlow Lite", "description": "Generic .tflite model + inference library"},
        {"id": "arduino", "name": "Arduino Library", "description": "Arduino .zip library with inference code"},
        {"id": "esp32", "name": "ESP32", "description": "ESP-IDF project with optimized inference"},
        {"id": "raspberry_pi", "name": "Raspberry Pi", "description": "Python package for RPi inference"},
        {"id": "unoq",         "name": "UNO Q",        "description": "Linux runtime .pe package for UNO Q"},
        {"id": "cpp",          "name": "C++ Library",  "description": "Portable C++17 inference library"},
        {"id": "pxe",          "name": "PXE Runtime",  "description": "Opaque PXE1 binary — EI stdio-JSONL protocol runner"},
    ]
}


@router.get("/targets")
def get_supported_targets(
    model_id: str | None = None,
    device_profile: str | None = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """The static format catalog, or — with `model_id` — the Target Device
    Phase 6 package-policy answer for one model (optionally narrowed to one
    device via `device_profile`).

    No params: today's static list, byte-identical — a public catalog of
    formats that something may yet call directly, unfiltered.

    `model_id` only: the legacy Generic TFLite (`.pe`) answer, model-only
    filtering with no device rule applied — a project with no target device
    deploys exactly as before.

    `model_id` + `device_profile`: the deploy target resolved from the
    catalog, the §1 package policy applied, and Phase 4's compatibility gate
    run on what survives. `assert_trained_model_owner` runs before any of that
    — same ownership surface as `GET /deployment/compatibility` — so a
    cross-tenant `model_id` gets 404, never a data-bearing "unavailable" body.

    Declared above `GET /{deployment_id}`: FastAPI matches routes in
    registration order, so this literal path has to win over the
    parameterised one or "targets" is read as a deployment id.
    """
    if not model_id:
        return _STATIC_TARGETS

    model = assert_trained_model_owner(db, model_id, current_user)
    return deployment_format_offer(db, model, device_profile=device_profile)


@router.get("/package-policy")
def get_package_policy(current_user: User = Depends(get_current_user)):
    """The full §1 package policy — what PetalEdge offers per deploy target,
    for all six deploy targets, with no ids.

    Takes no model or device: it answers "what does PetalEdge offer for
    target X" in the abstract, not "can this model build for this device"
    (that's `GET /deployment/targets?model_id=...&device_profile=...`). No
    ids means no ownership surface to check.

    Calls `all_package_policies`, the same `compatibility.py` table
    `deployment_format_offer` (and so `/targets`) reads — this endpoint and
    that one can never disagree because there is only one policy function.
    """
    return all_package_policies()


@router.get("/{deployment_id}")
def get_deployment(
    deployment_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    dep = assert_deployment_owner(db, deployment_id, current_user)
    dep = _reconcile_deployment_status(db, dep)
    if dep.storage_key and dep.status == JobStatus.completed:
        dep.download_url = storage.get_presigned_url(
            dep.storage_key,
            expires_in=86400,
            download_filename=(dep.options or {}).get("download_filename"),
        )
    return _dep_to_dict(dep)


@router.get("/model/{model_id}")
def list_deployments(
    model_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    assert_trained_model_owner(db, model_id, current_user)
    deps = (
        db.query(Deployment)
        .filter(Deployment.model_id == model_id)
        .order_by(Deployment.created_at.desc())
        .all()
    )
    return [_dep_to_dict(_reconcile_deployment_status(db, d)) for d in deps]


@router.post("/{deployment_id}/cancel")
def cancel_deployment(
    deployment_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Cancel a running or pending deployment build."""
    dep = assert_deployment_owner(db, deployment_id, current_user)

    dep = _reconcile_deployment_status(db, dep)
    if dep.status not in (JobStatus.pending, JobStatus.running):
        raise HTTPException(status_code=400, detail=f"Deployment is already {dep.status} and cannot be cancelled")

    if dep.celery_task_id:
        celery_app.control.revoke(dep.celery_task_id, terminate=True)

    dep.status = JobStatus.cancelled
    dep.error_message = "Cancelled by user"
    dep.completed_at = datetime.utcnow()
    db.commit()
    db.refresh(dep)

    return _dep_to_dict(dep)
