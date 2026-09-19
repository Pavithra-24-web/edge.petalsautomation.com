"""
Model Testing endpoints
=======================
Powers the /dashboard/impulse/model-testing frontend page.

POST /classify-all
  - Returns HTTP 202 Accepted immediately with {run_id, status="pending"}
  - Enqueues run_model_testing_job Celery task instead of running inline

GET /classify-all/{run_id}/status  (new)
  - Polls the ModelTestRun row: pending | running | completed | failed
  - Returns progress counts and full result envelope when completed

All other routes are unchanged.
"""
from typing import Optional
from datetime import datetime

from fastapi import APIRouter, Depends, Query, HTTPException
from pydantic import BaseModel, Field, field_validator
from sqlalchemy.orm import Session

from app.core.auth import get_current_user
from app.core.authz import assert_project_owner, assert_impulse_owner
from app.core.database import get_db
from app.models.user import User, Impulse, Sample
from app.models.model_testing import ModelTestRun, TestRunStatus, ModelTestSample, MetricResult, TestResultStatus, ModelVersion
from app.models.user import gen_uuid
import app.services.model_testing_service as svc

router = APIRouter()


# ─── Request / response schemas ───────────────────────────────────────────────

class ClassifyAllRequest(BaseModel):
    model_config = {"protected_namespaces": ()}
    impulse_id:       str           = Field(..., description="Impulse to classify")
    model_version_id: Optional[str] = Field(None)
    conf_threshold:   float         = Field(0.25, ge=0.0, le=1.0,
                                            description="Min confidence to count as detection")


class UpdateTestSampleRequest(BaseModel):
    expected_outcome: Optional[str] = Field(None, max_length=255)
    result_status:    Optional[str] = Field(None)

    @field_validator("result_status")
    @classmethod
    def _validate_status(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and v not in ("pending", "pass", "fail", "uncertain"):
            raise ValueError("result_status must be one of: pending, pass, fail, uncertain")
        return v


# ─── Routes ───────────────────────────────────────────────────────────────────

@router.get("/page-data", summary="Full page bootstrap data")
def get_page_data(
    impulse_id: Optional[str] = Query(None),
    project_id: Optional[str] = Query(None),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    # Enforce ownership on whichever scope is supplied; when neither is given
    # the service resolves to no impulse (empty payload), so there is nothing
    # to leak.
    if impulse_id:
        assert_impulse_owner(db, impulse_id, current_user)
    if project_id:
        assert_project_owner(db, project_id, current_user)
    return svc.get_page_data(impulse_id, project_id, db)


@router.get("/test-data", summary="Paginated test data rows")
def get_test_data(
    impulse_id: str = Query(...),
    page:       int = Query(1,  ge=1),
    page_size:  int = Query(20, ge=1, le=200),
    db:           Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    assert_impulse_owner(db, impulse_id, current_user)
    return svc.get_test_data(impulse_id, page, page_size, db)


@router.get("/model-versions", summary="Available model versions for dropdown")
def get_model_versions(
    impulse_id: str = Query(...),
    db:           Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    assert_impulse_owner(db, impulse_id, current_user)
    return svc.get_model_versions(impulse_id, db)


@router.post("/classify-all", summary="Enqueue classify-all job", status_code=202)
def classify_all(
    req: ClassifyAllRequest,
    db:  Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Enqueue an async model testing run.

    Returns HTTP 202 immediately with {run_id, status="pending"}.
    Poll GET /classify-all/{run_id}/status to track progress.
    """
    impulse = assert_impulse_owner(db, req.impulse_id, current_user)

    any_test_sample = (
        db.query(Sample)
        .filter(
            Sample.project_id == impulse.project_id,
            Sample.sample_type == "testing",
        )
        .first()
    )
    if not any_test_sample:
        raise HTTPException(
            422,
            "No test-split samples found for this project. "
            "Go to Data acquisition and set at least one sample's split to 'Testing'.",
        )

    now = datetime.utcnow()
    run = ModelTestRun(
        id               = gen_uuid(),
        project_id       = impulse.project_id,
        impulse_id       = req.impulse_id,
        model_version_id = req.model_version_id,   # worker overwrites with resolved id on completion
        status           = TestRunStatus.pending,
        created_at       = now,
    )
    db.add(run)
    db.commit()
    db.refresh(run)

    from app.workers.model_testing_worker import run_model_testing_job
    run_model_testing_job.apply_async(
        kwargs={
            "run_id":           run.id,
            "impulse_id":       req.impulse_id,
            "model_version_id": req.model_version_id,
            "conf_threshold":   req.conf_threshold,
        },
        task_id=run.id,
    )

    return {
        "run_id":  run.id,
        "status":  "pending",
        "message": "Classification job enqueued. Poll /classify-all/{run_id}/status for progress.",
    }


@router.get("/classify-all/{run_id}/status", summary="Poll classify-all job status")
def get_classify_all_status(
    run_id: str,
    db:     Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Poll the status of an async classify-all run.

    Returns {run_id, status, progress, result, error}.
    status: pending | running | completed | failed
    result is populated (full page_data envelope) only when status == "completed".
    """
    run = db.query(ModelTestRun).filter(ModelTestRun.id == run_id).first()
    if not run:
        raise HTTPException(404, f"Test run '{run_id}' not found")
    assert_project_owner(db, run.project_id, current_user)

    all_samples = (
        db.query(ModelTestSample)
        .filter(ModelTestSample.test_run_id == run_id)
        .all()
    )

    total_count       = run.total_samples or len(all_samples)
    passed_so_far     = sum(1 for s in all_samples if s.result_status == TestResultStatus.pass_)
    failed_so_far     = sum(1 for s in all_samples if s.result_status == TestResultStatus.fail)
    uncertain_so_far  = sum(1 for s in all_samples if s.result_status == TestResultStatus.uncertain)
    done_so_far       = passed_so_far + failed_so_far + uncertain_so_far

    progress = {
        "passed":    passed_so_far,
        "failed":    failed_so_far,
        "uncertain": uncertain_so_far,
        "done":      done_so_far,
        "total":     total_count,
        "remaining": max(0, total_count - done_so_far),
    }

    response: dict = {
        "run_id":   run_id,
        "status":   run.status,
        "progress": progress,
        "error":    run.error,
        "result":   None,
    }

    if run.status == TestRunStatus.completed:
        db_metrics = db.query(MetricResult).filter(MetricResult.test_run_id == run_id).all()
        from app.models.user import Project, Impulse as ImpulseModel
        impulse = db.query(ImpulseModel).filter(ImpulseModel.id == run.impulse_id).first()
        project = db.query(Project).filter(Project.id == run.project_id).first() if impulse else None

        formatted_metrics = [
            {
                "metric_name":         m.metric_name,
                "metric_display_name": m.metric_display_name,
                "metric_value":        m.metric_value,
            }
            for m in db_metrics
        ]

        # Resolve the model version used so the page_data is self-consistent.
        resolved_version = None
        if run.model_version_id:
            resolved_version = db.query(ModelVersion).filter(
                ModelVersion.id == run.model_version_id
            ).first()

        run_dict = {
            "id":             run.id,
            "status":         run.status,
            "accuracy":       run.accuracy,
            "total_samples":  run.total_samples,
            "passed_samples": run.passed_samples,
            "failed_samples": run.failed_samples,
            "model_version_id": run.model_version_id,
            "error":          run.error,
            "created_at":     run.created_at.isoformat()   if run.created_at   else None,
            "started_at":     run.started_at.isoformat()   if run.started_at   else None,
            "completed_at":   run.completed_at.isoformat() if run.completed_at else None,
        }

        version_dict = None
        if resolved_version:
            version_dict = {
                "id":                resolved_version.id,
                "version_number":    resolved_version.version_number,
                "name":              resolved_version.name,
                "quantization_type": resolved_version.quantization_type,
                "is_active":         resolved_version.is_active,
                "created_at":        resolved_version.created_at.isoformat()
                                     if resolved_version.created_at else None,
            }

        impulse_dict = None
        if impulse:
            impulse_dict = {
                "id":         impulse.id,
                "name":       impulse.name,
                "project_id": impulse.project_id,
            }

        response["result"] = {
            "run":     run_dict,
            "metrics": formatted_metrics,
            "page_data": {
                "project": {
                    "id":   project.id   if project else None,
                    "name": project.name if project else None,
                },
                "impulse":                impulse_dict,
                "available_impulses":     [impulse_dict] if impulse_dict else [],
                "selected_model_version": version_dict,
                "latest_run":             run_dict,
                "accuracy":               run.accuracy,
                "metrics":                formatted_metrics,
                "summary": {
                    "total":     run.total_samples,
                    "passed":    run.passed_samples,
                    # run.failed_samples stores fail+uncertain; split out using sample counts.
                    "failed":    run.failed_samples - uncertain_so_far,
                    "uncertain": uncertain_so_far,
                    "pending":   0,
                },
                "target_device": "Arduino UNO Q (Cortex-M0+)",
            },
        }

    return response


@router.delete("/classify-all/{run_id}", summary="Cancel a classify-all job", status_code=200)
def cancel_classify_all(
    run_id: str,
    db:     Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Cancel a pending or running classify-all job.

    - Revokes the Celery task (best-effort; does not kill already-executing batches instantly).
    - Sets run status to 'cancelled' so polling reflects the change immediately.
    - Safe to call on already-finished jobs — returns current status without error.
    """
    run = db.query(ModelTestRun).filter(ModelTestRun.id == run_id).first()
    if not run:
        raise HTTPException(404, f"Test run '{run_id}' not found")
    assert_project_owner(db, run.project_id, current_user)

    # Already terminal — nothing to do
    if run.status in (TestRunStatus.completed, TestRunStatus.failed, TestRunStatus.cancelled):
        return {"run_id": run_id, "status": run.status, "cancelled": False,
                "message": f"Run already in terminal state: {run.status}"}

    # Best-effort Celery revoke — task_id == run.id (set at enqueue time)
    try:
        from app.workers.celery_app import celery_app as _capp
        _capp.control.revoke(run_id, terminate=True, signal="SIGTERM")
    except Exception as _e:
        # Revoke failure is non-fatal; we still mark the DB row cancelled
        import logging as _log
        _log.getLogger(__name__).warning(
            f"[model_testing] revoke({run_id}) failed (non-fatal): {_e}"
        )

    run.status       = TestRunStatus.cancelled
    run.error        = "Cancelled by user"
    run.completed_at = datetime.utcnow()
    db.commit()

    return {"run_id": run_id, "status": "cancelled", "cancelled": True}


@router.patch("/test-data/{sample_id}", summary="Update test sample")
def update_test_sample(
    sample_id: str,
    req:       UpdateTestSampleRequest,
    db:        Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    # Ownership: a non-owned test sample is treated as not-found (the service
    # raises 404 when the row is missing entirely).
    mts = db.query(ModelTestSample).filter(ModelTestSample.id == sample_id).first()
    if mts is not None:
        assert_project_owner(db, mts.project_id, current_user)
    return svc.update_test_sample(
        sample_id, req.expected_outcome, req.result_status, db,
    )


@router.get("/runs", summary="Run history with metrics for version comparison")
def list_runs(
    impulse_id: str = Query(...),
    limit:      int = Query(10, ge=1, le=100),
    db:           Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Returns up to `limit` past runs newest-first, each with its metrics.
    Use model_version_id in each run to compare accuracy across versions.
    """
    assert_impulse_owner(db, impulse_id, current_user)
    return svc.list_runs(impulse_id, limit, db)


@router.get("/metrics", summary="Latest test run metrics")
def get_metrics(
    impulse_id: str = Query(...),
    db:           Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    assert_impulse_owner(db, impulse_id, current_user)
    return svc.get_metrics(impulse_id, db)


class ReclassifyRequest(BaseModel):
    model_version_id: Optional[str] = Field(None, description="Override model version; omit for latest")
    conf_threshold:   float          = Field(0.25, ge=0.0, le=1.0)


@router.post(
    "/test-data/{sample_id}/reclassify",
    summary="Run inference on a single test sample",
    status_code=200,
)
def reclassify_sample(
    sample_id: str,
    req:       ReclassifyRequest = ReclassifyRequest(),
    db:        Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Re-run model inference on one test sample and persist the updated score.
    Returns the updated sample record.
    """
    mts = db.query(ModelTestSample).filter(ModelTestSample.id == sample_id).first()
    if mts is not None:
        assert_project_owner(db, mts.project_id, current_user)
    return svc.reclassify_single_sample(
        sample_id        = sample_id,
        model_version_id = req.model_version_id,
        conf_threshold   = req.conf_threshold,
        db               = db,
    )
