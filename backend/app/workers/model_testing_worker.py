"""
app/workers/model_testing_worker.py
────────────────────────────────────
Celery task for async model testing ("Classify all").

Flow
----
1.  POST /classify-all  →  enqueues this task, returns {job_id, status="pending"}
2.  Celery worker picks it up, runs inference in batches
3.  After each batch, per-sample results are committed immediately so the
    frontend can see progress by polling GET /classify-all/{job_id}/status
4.  On completion, aggregate metrics (ModelTestRun + MetricResult) are
    committed and the job status is set to "completed"

ModelTestRun row lifecycle
--------------------------
  created with status="running"  on task start
  updated with status="completed" / "failed"  on finish

The run row id is the job_id returned to the caller.
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional

from app.workers.celery_app import celery_app
from app.core.database import SessionLocal
from app.models.user import Impulse, Sample, JobStatus
from app.models.model_testing import (
    ModelTestRun, MetricResult, ModelTestSample, TestResultStatus,
    UnsupportedModelTestingError,
)

logger = logging.getLogger(__name__)


@celery_app.task(
    bind=True,
    name="app.workers.model_testing_worker.run_model_testing_job",
    # No queue= here on purpose. A decorator queue silently outranks task_routes
    # (celery merges them as lpmerge(route, options)), and this task used to
    # carry queue="training" — which pinned it to the legacy queue while the
    # routing table claimed training_cpu. Routing now lives in one place:
    # task_routes in app/workers/celery_app.py. Model testing is CPU work.
    time_limit=1800,            # 30 min hard limit
    soft_time_limit=1740,
    acks_late=True,
)
def run_model_testing_job(
    self,
    run_id: str,
    impulse_id: str,
    model_version_id: Optional[str],
    conf_threshold: float = 0.25,
) -> dict:
    """
    Execute a full classify-all run asynchronously.

    Args:
        run_id           : id of the ModelTestRun row pre-created by the endpoint
        impulse_id       : impulse being tested
        model_version_id : optional explicit model version override
    """
    db = SessionLocal()
    try:
        # ── Fetch the pre-created run row ─────────────────────────────────────
        run = db.query(ModelTestRun).filter(ModelTestRun.id == run_id).first()
        if not run:
            logger.error(f"[model_testing] run_id={run_id} not found — aborting")
            return {"status": "failed", "error": "Run record not found"}

        impulse = db.query(Impulse).filter(Impulse.id == impulse_id).first()
        if not impulse:
            _fail_run(run, db, "Impulse not found")
            return {"status": "failed"}

        run.status     = "running"
        run.started_at = datetime.utcnow()
        db.commit()

        # ── Import service helpers (avoids circular import at module level) ───
        from app.services.model_testing_service import (
            _ensure_test_samples_provisioned,
            _load_trained_model,
            _get_active_version,
            _run_classify_all,
            _compute_aggregate_metrics,
            _iter_scalar_metric_dicts,
        )

        # ── Provision test samples ────────────────────────────────────────────
        samples = _ensure_test_samples_provisioned(impulse, db)
        db.commit()

        if not samples:
            _fail_run(run, db, "No test-split samples found")
            return {"status": "failed"}

        # ── Load model ────────────────────────────────────────────────────────
        trained_model, output_type = _load_trained_model(impulse_id, model_version_id, db)

        # Fix #4: reset stale scores before running with current model
        from app.services.model_testing_service import _reset_stale_scores
        n_reset = _reset_stale_scores(impulse_id, trained_model.id, db)
        if n_reset > 0:
            logger.info(
                f"[model_testing] run={run_id} reset {n_reset} stale sample(s) "
                f"scored by previous model version"
            )
            db.commit()
            from app.models.model_testing import ModelTestSample as _MTS
            samples = (
                db.query(_MTS)
                .filter(_MTS.impulse_id == impulse_id)
                .order_by(_MTS.created_at.asc())
                .all()
            )
        logger.info(
            f"[model_testing] run={run_id} model={trained_model.id} "
            f"output_type={output_type} samples={len(samples)}"
        )

        # ── Inference loop ────────────────────────────────────────────────────
        results, label_names, det_image_results = _run_classify_all(
            impulse=impulse,
            test_samples=samples,
            trained_model=trained_model,
            output_type=output_type,
            db=db,
            conf_threshold=conf_threshold,
        )

        # ── Persist per-sample results ────────────────────────────────────────
        now       = datetime.utcnow()
        passed    = 0
        uncertain = 0
        fail_cnt  = 0
        for sample, score, status, pred_cls, iou_val, pred_boxes in results:
            sample.f1_score                   = score
            sample.result_status              = status
            sample.predicted_class            = pred_cls
            sample.iou_score                  = iou_val
            sample.predicted_boxes            = pred_boxes
            sample.test_run_id                = run_id
            sample.scored_by_trained_model_id = trained_model.id
            sample.updated_at                 = now
            if status == TestResultStatus.pass_:
                passed += 1
            elif status == TestResultStatus.uncertain:
                uncertain += 1
            else:
                fail_cnt += 1

        total    = len(results)
        # failed_samples stores fail+uncertain (both non-pass) for DB backward-compat.
        failed   = fail_cnt + uncertain
        accuracy = round((passed / total) * 100, 2) if total > 0 else 0.0

        # ── Aggregate metrics ─────────────────────────────────────────────────
        metric_dicts = _compute_aggregate_metrics(
            results=results,
            label_names=label_names,
            output_type=output_type,
            det_image_results=det_image_results if det_image_results else None,
        )
        for md in _iter_scalar_metric_dicts(metric_dicts):
            db.add(MetricResult(
                test_run_id         = run_id,
                metric_name         = md["metric_name"],
                metric_display_name = md["metric_display_name"],
                metric_value        = md["metric_value"],
                created_at          = now,
            ))

        # ── Finalise run row ──────────────────────────────────────────────────
        # Resolve and persist the actual ModelVersion used (handles None → active version).
        resolved_version = _get_active_version(impulse_id, model_version_id, db)
        run.model_version_id = resolved_version.id if resolved_version else None
        run.accuracy         = accuracy
        run.total_samples    = total
        run.passed_samples   = passed
        run.failed_samples   = failed
        run.status           = "completed"
        run.completed_at     = now

        db.commit()
        logger.info(
            f"[model_testing] run={run_id} complete — "
            f"accuracy={accuracy:.2f}% passed={passed}/{total}"
        )
        return {
            "status":   "completed",
            "run_id":   run_id,
            "accuracy": accuracy,
            "passed":   passed,
            "failed":   failed,
            "total":    total,
        }

    except UnsupportedModelTestingError as e:
        # Expected, known terminal condition (e.g. SSD has no detection-scoring
        # path yet). Mark the run failed with the clear message, but log at
        # WARNING without a traceback and do NOT re-raise — this is a gated
        # config, not an unexpected crash, so Celery should not flag it FAILED.
        logger.warning(f"[model_testing] run={run_id} gated: {e}")
        try:
            run = db.query(ModelTestRun).filter(ModelTestRun.id == run_id).first()
            if run:
                _fail_run(run, db, str(e))
        except Exception:
            pass
        return
    except Exception as e:
        logger.exception(f"[model_testing] run={run_id} failed: {e}")
        try:
            run = db.query(ModelTestRun).filter(ModelTestRun.id == run_id).first()
            if run:
                _fail_run(run, db, str(e))
        except Exception:
            pass
        raise  # re-raise so Celery marks the task as FAILED
    finally:
        db.close()


def _fail_run(run: ModelTestRun, db, error_msg: str) -> None:
    """Mark a ModelTestRun as failed and commit."""
    try:
        run.status       = "failed"
        run.completed_at = datetime.utcnow()
        run.error        = error_msg[:500]
        db.commit()
    except Exception:
        db.rollback()
