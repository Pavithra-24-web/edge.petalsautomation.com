"""
Trained Models endpoints — retrieve, list, and download model artifacts
produced by completed training jobs.

These are consumed by: Retrain page, Live Classification, Model Testing,
Post-Processing, and Deployment pages.
"""
from fastapi import APIRouter, Depends, HTTPException, Response
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
from typing import Optional
import io

from app.core.database import get_db
from app.core.auth import get_current_user
from app.core.authz import (
    assert_project_owner,
    assert_impulse_owner,
    assert_training_job_owner,
    assert_trained_model_owner,
)
from app.core.storage import storage
from app.models.user import User, TrainedModel, TrainingJob, Impulse, JobStatus
from app.services import estimation
from app.services.compatibility import resolve_build_format

router = APIRouter()


# ── Engine registry (module-level for importability in tests) ────────────────

_ENGINE_REGISTRY = [
    {
        "id": "eon_ram_optimized",
        "label": "EON\u2122 Compiler (RAM optimized)",
        "supported": False,
        "status": "coming_soon",
        "unsupported_reason": (
            "EON Compiler with RAM optimization is not yet available. "
            "Use TensorFlow Lite for now."
        ),
    },
    {
        "id": "eon",
        "label": "EON\u2122 Compiler",
        "supported": False,
        "status": "coming_soon",
        "unsupported_reason": (
            "EON Compiler is not yet available. "
            "Use TensorFlow Lite for now."
        ),
    },
    {
        "id": "tflite",
        "label": "TensorFlow Lite",
        "supported": True,
        "status": "available",
        "unsupported_reason": None,
    },
]


def _model_dict(m: TrainedModel) -> dict:
    meta = m.model_metadata or {}
    return {
        "id":              m.id,
        "training_job_id": m.training_job_id,
        "version":         m.version,
        "format":          m.format,
        "variant":         meta.get("variant", "float32"),
        "file_size_bytes": m.file_size_bytes,
        "label_names":     meta.get("label_names", []),
        "input_shape":     meta.get("input_shape", []),
        "num_classes":     meta.get("num_classes", 0),
        "architecture":    meta.get("architecture", "dense"),
        "quantized":       meta.get("quantized", False),
        "output_type":     meta.get("output_type"),
        "model_metadata":  meta,
        "created_at":      m.created_at.isoformat(),
    }


@router.get("/job/{job_id}")
def list_models_for_job(
    job_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """List all model artifacts produced by a training job."""
    job = assert_training_job_owner(db, job_id, current_user)
    models = (
        db.query(TrainedModel)
        .filter(TrainedModel.training_job_id == job_id)
        .all()
    )
    return [_model_dict(m) for m in models]


@router.get("/impulse/{impulse_id}/latest")
def get_latest_model(
    impulse_id: str,
    format: Optional[str] = "tflite",
    variant: Optional[str] = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Get the latest trained model for an impulse.

    Query params:
      format  – "tflite" (default) or "keras"
      variant – "int8", "float32", or None (returns first match)

    Used by Live Classification, Model Testing, and Post-Processing pages.
    """
    impulse = assert_impulse_owner(db, impulse_id, current_user)

    job = (
        db.query(TrainingJob)
        .filter(
            TrainingJob.impulse_id == impulse_id,
            TrainingJob.status == JobStatus.completed,
        )
        .order_by(TrainingJob.created_at.desc())
        .first()
    )
    if not job:
        raise HTTPException(404, "No completed training job found for this impulse")

    # Fetch all models for the job, then pick best match
    candidates = (
        db.query(TrainedModel)
        .filter(TrainedModel.training_job_id == job.id)
        .all()
    )
    if not candidates:
        raise HTTPException(404, "No trained model artifacts found")

    # Priority: exact format+variant > exact format > any
    model = None
    variant_was_fallback = False
    if variant:
        model = next(
            (m for m in candidates
             if m.format == format
             and (m.model_metadata or {}).get("variant") == variant),
            None,
        )
    if not model:
        model = next((m for m in candidates if m.format == format), None)
        if variant:
            variant_was_fallback = True
    if not model:
        model = candidates[0]
        if variant:
            variant_was_fallback = True

    result = _model_dict(model)

    # List all available variants for this job so the frontend can build a picker
    available_variants = []
    for m in candidates:
        v_meta = m.model_metadata or {}
        available_variants.append({
            "id": m.id,
            "format": m.format,
            "variant": v_meta.get("variant", "float32"),
            "quantized": v_meta.get("quantized", False),
            "file_size_bytes": m.file_size_bytes,
        })

    result["available_variants"] = available_variants
    result["requested_variant"] = variant
    result["is_fallback"] = variant_was_fallback
    if variant_was_fallback:
        actual = (model.model_metadata or {}).get("variant", "float32")
        result["fallback_reason"] = (
            f"Requested variant '{variant}' is not available. "
            f"Fell back to '{actual}'."
        )
    result["training_job"] = {
        "id":             job.id,
        "status":         job.status,
        "best_accuracy":  job.best_accuracy,
        "final_accuracy": job.final_accuracy,
        "best_loss":      job.best_loss,
        "epochs":         job.epochs,
        "completed_at":   job.completed_at.isoformat() if job.completed_at else None,
    }
    return result


@router.get("/project/{project_id}/latest")
def get_latest_model_for_project(
    project_id: str,
    format: Optional[str] = "tflite",
    variant: Optional[str] = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Get the latest trained model for a project.

    Selection is based on the newest completed training job whose artifacts
    carry model_metadata.project_id == project_id.
    """
    assert_project_owner(db, project_id, current_user)
    model_rows = (
        db.query(TrainedModel, TrainingJob)
        .join(TrainingJob, TrainedModel.training_job_id == TrainingJob.id)
        .filter(
            TrainingJob.status == JobStatus.completed,
        )
        .order_by(TrainingJob.created_at.desc(), TrainedModel.created_at.desc())
        .all()
    )
    candidates = [
        model for model, _job in model_rows
        if (model.model_metadata or {}).get("project_id") == project_id
    ]
    if not candidates:
        raise HTTPException(404, "No completed trained models found for this project")

    selected_model = None
    variant_was_fallback = False
    if variant:
        selected_model = next(
            (
                m for m in candidates
                if m.format == format and (m.model_metadata or {}).get("variant") == variant
            ),
            None,
        )
    if not selected_model:
        selected_model = next((m for m in candidates if m.format == format), None)
        if variant:
            variant_was_fallback = True
    if not selected_model:
        selected_model = candidates[0]
        if variant:
            variant_was_fallback = True

    result = _model_dict(selected_model)
    result["requested_variant"] = variant
    result["is_fallback"] = variant_was_fallback
    if variant_was_fallback:
        actual = (selected_model.model_metadata or {}).get("variant", "float32")
        result["fallback_reason"] = (
            f"Requested variant '{variant}' is not available. "
            f"Fell back to '{actual}'."
        )
    return result


@router.get("/impulse/{impulse_id}/all")
def list_all_models_for_impulse(
    impulse_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    List all trained models across all training jobs for an impulse.
    Used by the Retrain page to show training history.
    """
    impulse = assert_impulse_owner(db, impulse_id, current_user)

    jobs = (
        db.query(TrainingJob)
        .filter(TrainingJob.impulse_id == impulse_id)
        .order_by(TrainingJob.created_at.desc())
        .all()
    )

    result = []
    for job in jobs:
        models = (
            db.query(TrainedModel)
            .filter(TrainedModel.training_job_id == job.id)
            .all()
        )
        result.append({
            "job": {
                "id":             job.id,
                "status":         job.status,
                "best_accuracy":  job.best_accuracy,
                "final_accuracy": job.final_accuracy,
                "best_loss":      job.best_loss,
                "epochs":         job.epochs,
                "learning_rate":  job.learning_rate,
                "batch_size":     job.batch_size,
                "training_history": job.training_history,
                "confusion_matrix": job.confusion_matrix,
                "classification_report": job.classification_report,
                "error_message":  job.error_message,
                "started_at":     job.started_at.isoformat() if job.started_at else None,
                "completed_at":   job.completed_at.isoformat() if job.completed_at else None,
                "created_at":     job.created_at.isoformat(),
            },
            "models": [_model_dict(m) for m in models],
        })
    return result


@router.get("/impulse/{impulse_id}/panel")
def get_model_panel(
    impulse_id: str,
    variant: Optional[str] = "int8",
    engine: Optional[str] = "tflite",
    threshold: Optional[float] = None,   # FOMO detection threshold override (0.0–1.0)
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Unified response for the post-training Model panel.

    Returns everything the frontend needs to render:
      - model version dropdown  (Quantized int8, Unoptimized float32)
      - engine dropdown         (TensorFlow Lite, EON Compiler, EON RAM-optimized)
      - training performance    (F1, accuracy, or loss-based for FOMO)
      - confusion matrix
      - per-class metrics
      - on-device performance   (estimated or unavailable)

    Query params:
      variant – "int8" (default) or "float32"
      engine  – "tflite" (default), "eon", or "eon_ram_optimized"
    """
    impulse = assert_impulse_owner(db, impulse_id, current_user)

    job = (
        db.query(TrainingJob)
        .filter(
            TrainingJob.impulse_id == impulse_id,
            TrainingJob.status == JobStatus.completed,
        )
        .order_by(TrainingJob.created_at.desc())
        .first()
    )
    if not job:
        raise HTTPException(404, "No completed training job found for this impulse")

    # ── All model artifacts for this job ──────────────────────────────────
    all_models = (
        db.query(TrainedModel)
        .filter(TrainedModel.training_job_id == job.id)
        .all()
    )
    if not all_models:
        raise HTTPException(404, "No trained model artifacts found")

    # ── Build available_model_versions ────────────────────────────────────
    tflite_models = [m for m in all_models if m.format == "tflite"]
    available_model_versions = []
    for m in tflite_models:
        meta = m.model_metadata or {}
        v = meta.get("variant", "float32")
        # Metrics this variant scored on its own — written post-export by
        # app.ml.variant_eval.evaluate_exported_variants.  Absent for jobs
        # trained before per-variant eval existed; the dropdown then shows an
        # explicit "not scored" state rather than another variant's numbers.
        vm = meta.get("variant_metrics") or {}
        metrics_available = vm.get("status") == "success"
        available_model_versions.append({
            "id": m.id,
            "variant": v,
            "label": f"Quantized ({v})" if meta.get("quantized") else f"Unoptimized ({v})",
            "quantized": meta.get("quantized", False),
            "file_size_bytes": m.file_size_bytes,
            "available": True,
            "unavailable_reason": None,
            "metrics": vm or None,
            "metrics_available": metrics_available,
            "metrics_unavailable_reason": (
                None if metrics_available
                else (
                    vm.get("error")
                    or (
                        "This variant has not been scored. It was exported "
                        "before per-variant evaluation was introduced — "
                        "retrain to get its own metrics."
                        if not vm else
                        f"Evaluation did not complete for this variant "
                        f"(status: {vm.get('status')})."
                    )
                )
            ),
        })

    # If only one variant exists, mark the other as unavailable with reason.
    # Prefer the real error from training_history if recorded by the worker.
    existing_variants = {v["variant"] for v in available_model_versions}
    history = job.training_history or {}
    _FALLBACK_REASONS = {
        "int8": (
            "Int8 quantization failed during training. "
            "The model may use ops not supported by the int8 converter. "
            "Use the float32 variant instead."
        ),
        "float32": (
            "Float32 TFLite conversion failed during training. "
            "This is unexpected — please retrain the model."
        ),
    }
    for expected in ["int8", "float32"]:
        if expected not in existing_variants:
            # Use recorded error from training history if available
            if expected == "int8" and history.get("int8_error"):
                reason = history["int8_error"]
            else:
                reason = _FALLBACK_REASONS[expected]
            available_model_versions.append({
                "id": None,
                "variant": expected,
                "label": f"Quantized ({expected})" if expected == "int8" else f"Unoptimized ({expected})",
                "quantized": expected == "int8",
                "file_size_bytes": None,
                "available": False,
                "unavailable_reason": reason,
                # No artifact ⇒ nothing was scored.  Carry the export failure
                # reason through so the UI explains the gap instead of showing
                # some other variant's numbers.
                "metrics": None,
                "metrics_available": False,
                "metrics_unavailable_reason": reason,
            })

    # Sort: int8 first (default selection)
    available_model_versions.sort(key=lambda v: (v["variant"] != "int8", v["variant"]))

    # ── Select the active model variant ──────────────────────────────────
    requested_variant = variant
    selected_model = next(
        (m for m in tflite_models
         if (m.model_metadata or {}).get("variant") == requested_variant),
        None,
    )
    variant_was_fallback = False
    fallback_reason = None
    if not selected_model and tflite_models:
        selected_model = tflite_models[0]
        variant_was_fallback = True
        fallback_reason = (
            f"Requested variant '{requested_variant}' is not available. "
            f"Fell back to '{(selected_model.model_metadata or {}).get('variant', 'float32')}'."
        )

    selected_meta = (selected_model.model_metadata or {}) if selected_model else {}
    selected_variant = selected_meta.get("variant", "float32")

    # ── Available engines ────────────────────────────────────────────────
    available_engines = _ENGINE_REGISTRY

    _VALID_ENGINE_IDS = {e["id"] for e in _ENGINE_REGISTRY}
    requested_engine = engine if engine in _VALID_ENGINE_IDS else "tflite"
    engine_entry = next(e for e in _ENGINE_REGISTRY if e["id"] == requested_engine)
    # If user picks an unsupported engine, record that and fall back to tflite
    engine_was_fallback = False
    engine_fallback_reason = None
    if not engine_entry["supported"]:
        engine_was_fallback = True
        engine_fallback_reason = engine_entry["unsupported_reason"]
        # Fall back to tflite for device_performance — but keep
        # selected_engine as the user's request so UI can show the selection.
    selected_engine = requested_engine

    # ── Architecture / FOMO / YOLO-Pro detection ─────────────────────────────
    architecture = selected_meta.get("architecture", "")
    output_type  = selected_meta.get("output_type")
    is_yolo_pro  = output_type in ("yolo_pro_detection", "ssd_detection")
    is_fomo      = (
        not is_yolo_pro
        and (
            output_type == "detection_heatmap"
            or "fomo" in architecture.lower()
        )
    )
    label_names = selected_meta.get("label_names", [])

    # ── Training performance ─────────────────────────────────────────────
    history = job.training_history or {}
    cr = job.classification_report or {}

    # ── Metrics for the SELECTED variant ──────────────────────────────────
    # Each exported variant is scored on its own after export; prefer that over
    # the job-wide report, which reflects the in-memory Keras model and is the
    # same for every variant.  Falling back to `cr` when a variant has no stored
    # metrics is only for pre-existing jobs — the dropdown separately reports
    # metrics_available=False so the UI never passes float32 numbers off as int8's.
    selected_variant_metrics = selected_meta.get("variant_metrics") or {}
    has_variant_metrics = selected_variant_metrics.get("status") == "success"
    # Single source of truth for the selected variant's detection numbers —
    # used by BOTH the headline metric and the aggregate block below, so the
    # two can never read from different reports again.
    eval_src = selected_variant_metrics if has_variant_metrics else cr

    if is_yolo_pro:
        # ── YOLO-Pro detection metrics ─────────────────────────────────────
        _src = eval_src
        if has_variant_metrics:
            yp_status = "success"
            yp_error  = None
        else:
            yp_status = cr.get("yolo_pro_eval_status")  # success|no_data|inference_failed|not_run
            yp_error  = cr.get("yolo_pro_eval_error")
        is_real_yp_eval = (yp_status == "success")

        yp_map50 = _src.get("map50")
        yp_map   = _src.get("map")
        yp_map75 = _src.get("map75")
        yp_prec  = _src.get("precision")
        yp_rec   = _src.get("recall")

        if is_real_yp_eval and yp_map50 is not None:
            headline    = {"name": "mAP@50", "value": round(yp_map50 * 100, 1), "unit": "%"}
            metric_type = "detection"
        else:
            headline    = {"name": "Best val_loss", "value": job.best_loss, "unit": None}
            metric_type = "loss"

        training_performance = {
            "metric_type":              metric_type,
            "headline_metric":          headline,
            "accuracy":                 None,
            "f1_score":                 None,
            "best_loss":                job.best_loss,
            "is_fomo":                  False,
            "is_yolo_pro":              True,
            "is_real_detection_eval":   is_real_yp_eval,
            "yolo_pro_eval_status":     yp_status,
            "yolo_pro_eval_error":      yp_error,
            "map":                      yp_map,
            "map50":                    yp_map50,
            "map75":                    yp_map75,
            "precision":                yp_prec,
            "recall":                   yp_rec,
            "per_class":                _src.get("per_class", {}),
            # True when these numbers came from scoring THIS variant, not the
            # job-wide Keras eval.
            "is_variant_scoped":        has_variant_metrics,
            "scored_variant":           selected_variant if has_variant_metrics else None,
            "metric_note":              history.get("metric_note",
                "YOLO-Pro detection: accuracy not tracked. "
                "Use mAP/precision/recall for model quality."),
        }

        # YOLO-Pro has no confusion matrix — return empty
        threshold_sweep        = []
        confidence_diagnostics = None
        active_threshold       = None
        active_threshold_entry = None
        fomo_eval_status       = None
        is_real_fomo_eval      = False

    elif is_fomo:
        # "Real" evaluation means the worker ran _evaluate_fomo_detection()
        # successfully — signalled by fomo_eval_status == "success".
        fomo_eval_status  = cr.get("fomo_eval_status")   # "success"|"no_data"|"inference_failed"|None
        fomo_eval_error   = cr.get("fomo_eval_error")
        is_real_fomo_eval = (fomo_eval_status == "success")

        # ── Threshold resolution ──────────────────────────────────────────
        # The stored cr has metrics at the training-time threshold (default 0.5).
        # If the caller passes a different threshold AND we have a pre-computed
        # threshold_sweep, we read that entry instead of the stored cr values.
        threshold_sweep         = cr.get("threshold_sweep", [])
        confidence_diagnostics  = cr.get("confidence_diagnostics")
        stored_threshold        = cr.get("threshold", 0.5)

        # Resolve the "active" threshold and its sweep entry (if any)
        active_threshold_entry  = None
        active_threshold        = stored_threshold
        if threshold is not None and is_real_fomo_eval and threshold_sweep:
            _entry = next(
                (s for s in threshold_sweep
                 if abs(s.get("threshold", -1) - threshold) < 0.001),
                None,
            )
            if _entry is not None:
                active_threshold_entry = _entry
                active_threshold       = threshold

        # Macro F1 — from sweep entry when available, otherwise from stored cr
        if active_threshold_entry is not None:
            fomo_macro_f1 = active_threshold_entry.get("macro_f1")
        else:
            fomo_macro_f1 = cr.get("macro avg", {}).get("f1-score") if is_real_fomo_eval else None

        if is_real_fomo_eval and fomo_macro_f1 is not None:
            headline    = {"name": "F1 (detection)", "value": round(fomo_macro_f1 * 100, 1), "unit": "%"}
            metric_type = "detection"
        else:
            headline    = {"name": "Best val_loss", "value": job.best_loss, "unit": None}
            metric_type = "loss"

        training_performance = {
            "metric_type":            metric_type,
            "headline_metric":        headline,
            "accuracy":               None,
            "f1_score":               fomo_macro_f1,
            "best_loss":              job.best_loss,
            "is_fomo":                True,
            "is_real_detection_eval": is_real_fomo_eval,
            "fomo_eval_status":       fomo_eval_status,
            "fomo_eval_error":        fomo_eval_error,
            "detection_threshold":    active_threshold if is_real_fomo_eval else None,
            "metric_note":            history.get("metric_note",
                "FOMO heatmap training: accuracy not tracked. Use val_loss for convergence."),
        }
    else:
        # Classification: accuracy + F1
        threshold_sweep        = []
        confidence_diagnostics = None
        active_threshold       = None
        active_threshold_entry = None
        macro    = cr.get("macro avg", {})
        weighted = cr.get("weighted avg", {})
        f1       = weighted.get("f1-score", macro.get("f1-score"))
        training_performance = {
            "metric_type": "accuracy",
            "headline_metric": {
                "name": "F1 SCORE",
                "value": round(f1 * 100, 1) if f1 is not None else None,
                "unit": "%",
            },
            "accuracy":   job.best_accuracy,
            "f1_score":   f1,
            "best_loss":  job.best_loss,
            "is_fomo":    False,
            "metric_note": None,
        }

    # ── Per-epoch training metrics (architecture-agnostic) ───────────────
    # Surface the array the workers persist incrementally into
    # training_history.epoch_metrics. Always a list — empty when the run
    # predates this change or hasn't completed an epoch — so the frontend
    # empty-states without branching on worker or model type.
    _epoch_metrics = history.get("epoch_metrics")
    training_performance["epoch_metrics"] = (
        _epoch_metrics if isinstance(_epoch_metrics, list) else []
    )

    # ── Confusion matrix ─────────────────────────────────────────────────
    cm_raw = job.confusion_matrix or []
    if is_yolo_pro:
        cm_labels = label_names
    elif is_fomo:
        # FOMO has label_names + "background" as columns/rows
        cm_labels = ["background"] + label_names
    else:
        cm_labels = label_names

    confusion_matrix = {
        "labels": cm_labels,
        "matrix": cm_raw,
        "is_fomo": is_fomo,
    }

    # ── Per-class metrics ────────────────────────────────────────────────
    # is_real_fomo_eval / active_threshold_entry were resolved in the block
    # above (they're in scope via the training_performance block).
    # For clarity, re-bind the eval-status from cr (single source of truth).
    _fomo_eval_status = cr.get("fomo_eval_status")
    is_real_fomo_eval = (_fomo_eval_status == "success") if not is_yolo_pro else False

    metrics_rows = []
    if is_yolo_pro:
        per_class_yp = cr.get("per_class", {})
        for lbl in label_names:
            pc = per_class_yp.get(lbl, {})
            metrics_rows.append({
                "label":     lbl,
                "ap":        pc.get("ap"),
                "precision": pc.get("precision"),
                "recall":    pc.get("recall"),
                "f1_score":  None,   # not stored; can be derived client-side
                "support":   None,
            })
    else:
        # FOMO or classification
        for lbl in label_names:
            if is_fomo and is_real_fomo_eval:
                if active_threshold_entry is not None:
                    # Use per-class data from the pre-computed threshold sweep
                    pc = active_threshold_entry.get("per_class", {}).get(lbl, {})
                    row: dict = {
                        "label":     lbl,
                        "precision": pc.get("precision"),
                        "recall":    pc.get("recall"),
                        "f1_score":  pc.get("f1"),
                        "support":   None,    # sweep does not re-compute support
                        "tp":        pc.get("tp"),
                        "fp":        pc.get("fp"),
                        "fn":        pc.get("fn"),
                    }
                else:
                    # Use the stored cr values (computed at training-time threshold)
                    lbl_report = cr.get(lbl, {})
                    row = {
                        "label":     lbl,
                        "precision": lbl_report.get("precision"),
                        "recall":    lbl_report.get("recall"),
                        "f1_score":  lbl_report.get("f1-score"),
                        "support":   lbl_report.get("support"),
                        "tp":        lbl_report.get("tp"),
                        "fp":        lbl_report.get("fp"),
                        "fn":        lbl_report.get("fn"),
                    }
            else:
                lbl_report = cr.get(lbl, {}) if not is_fomo else {}
                row = {
                    "label":     lbl,
                    "precision": lbl_report.get("precision"),
                    "recall":    lbl_report.get("recall"),
                    "f1_score":  lbl_report.get("f1-score"),
                    "support":   lbl_report.get("support"),
                }
            metrics_rows.append(row)

    # ── Aggregate metrics ────────────────────────────────────────────────
    if is_yolo_pro:
        # Same eval_src as the headline metric above — variant-scoped when the
        # variant has its own stored eval, job-wide `cr` only as a fallback.
        _dm = eval_src.get("detailed_metrics", {}) or {}
        _agg = {
            "precision":        eval_src.get("precision"),
            "recall":           eval_src.get("recall"),
            "f1_score":         None,
            "map":              eval_src.get("map"),
            "map50":            eval_src.get("map50"),
            "map75":            eval_src.get("map75"),
            # Extended COCO-style metrics (None for old jobs without detailed_metrics)
            "map_small":        _dm.get("map_small"),
            "map_medium":       _dm.get("map_medium"),
            "map_large":        _dm.get("map_large"),
            "recall_max1":      _dm.get("recall_max1"),
            "recall_max10":     _dm.get("recall_max10"),
            "recall_max100":    _dm.get("recall_max100"),
            "recall_small":     _dm.get("recall_small"),
            "recall_medium":    _dm.get("recall_medium"),
            "recall_large":     _dm.get("recall_large"),
            "precision_legacy": _dm.get("precision_legacy", eval_src.get("precision")),
            # Background ("negative") image reporting — None when the eval split
            # had no background images. mAP cannot see these, so without the
            # explicit metric a user who added negatives has no feedback.
            "background_images":  eval_src.get("background_images"),
            "background_fp_rate": eval_src.get("background_fp_rate"),
            "background_images_with_fp": eval_src.get("background_images_with_fp"),
        }
    elif is_fomo and is_real_fomo_eval:
        if active_threshold_entry is not None:
            _agg = {
                "precision": None,   # sweep stores macro_f1 but not macro prec/rec
                "recall":    None,
                "f1_score":  active_threshold_entry.get("macro_f1"),
            }
        else:
            _agg_src = cr.get("macro avg", {})
            _agg = {
                "precision": _agg_src.get("precision"),
                "recall":    _agg_src.get("recall"),
                "f1_score":  _agg_src.get("f1-score"),
            }
        # Background FP reporting is threshold-independent of the sweep entry —
        # it is computed once at the resolved threshold, so it is attached in
        # both branches above.
        _agg["background_images"]  = cr.get("background_images")
        _agg["background_fp_rate"] = cr.get("background_fp_rate")
        _agg["background_images_with_fp"] = cr.get("background_images_with_fp")
    elif not is_fomo:
        _agg_src = cr.get("weighted avg", {})
        _agg = {
            "precision": _agg_src.get("precision"),
            "recall":    _agg_src.get("recall"),
            "f1_score":  _agg_src.get("f1-score"),
        }
    else:
        _agg = None

    metrics = {
        "label":                  "Metrics (validation set)",
        "rows":                   metrics_rows,
        "is_yolo_pro":            is_yolo_pro,
        "yolo_pro_eval_status":   cr.get("yolo_pro_eval_status") if is_yolo_pro else None,
        "yolo_pro_eval_error":    cr.get("yolo_pro_eval_error")  if is_yolo_pro else None,
        "fomo_eval_status":       _fomo_eval_status if is_fomo else None,
        "fomo_eval_error":        cr.get("fomo_eval_error") if is_fomo else None,
        "is_real_detection_eval": (
            (cr.get("yolo_pro_eval_status") == "success") if is_yolo_pro
            else (is_real_fomo_eval if is_fomo else None)
        ),
        "detection_threshold":    active_threshold if (is_fomo and is_real_fomo_eval) else None,
        "aggregate":              _agg,
        # Diagnostic data for threshold analysis (FOMO only; empty for others)
        "threshold_sweep":        threshold_sweep if is_fomo else [],
        "confidence_diagnostics": confidence_diagnostics if is_fomo else None,
    }

    # ── On-device performance ────────────────────────────────────────────
    device_performance = _build_device_performance(
        selected_model, selected_engine, engine_entry,
        device_slug=impulse.project.target_device_slug,
        db=db,
    )

    return {
        "impulse_id":   impulse_id,
        "job_id":       job.id,
        "architecture": architecture,
        "output_type":  output_type,
        "is_fomo":      is_fomo,
        "is_yolo_pro":  is_yolo_pro,

        "available_model_versions": available_model_versions,
        "selected_model_version": {
            "id":                selected_model.id if selected_model else None,
            "variant":           selected_variant,
            "quantized":         selected_meta.get("quantized", False),
            "requested_variant": requested_variant,
            "is_fallback":       variant_was_fallback,
            "fallback_reason":   fallback_reason,
            # Per-variant evaluation result for the selected variant.  The UI
            # renders an explicit unavailable state (with this reason) rather
            # than showing another variant's metrics in its place.
            "metrics":                     selected_variant_metrics or None,
            "metrics_available":           has_variant_metrics,
            "metrics_unavailable_reason":  (
                None if has_variant_metrics
                else (
                    selected_variant_metrics.get("error")
                    or (
                        "This variant has not been scored. It was exported "
                        "before per-variant evaluation was introduced — "
                        "retrain to get its own metrics."
                        if not selected_variant_metrics else
                        f"Evaluation did not complete for this variant "
                        f"(status: {selected_variant_metrics.get('status')})."
                    )
                )
            ),
        },

        "available_engines": available_engines,
        "selected_engine": {
            "id":            selected_engine,
            "supported":     engine_entry["supported"],
            "is_fallback":   engine_was_fallback,
            "fallback_reason": engine_fallback_reason,
        },

        "training_performance": training_performance,
        "confusion_matrix":     confusion_matrix,
        "metrics":              metrics,
        "device_performance":   device_performance,
    }


def _build_device_performance(
    selected_model, engine_id: str, engine_entry: dict,
    *, device_slug: Optional[str] = None, db: Optional[Session] = None,
) -> dict:
    """
    Build the device_performance block for the Model panel.

    Delegates the actual flash/RAM/latency figures to
    `app.services.estimation.estimate` (Target Device Phase 5) — the same
    function `GET /deployment/estimate` calls — so the panel and that
    endpoint can never disagree about the same model/device.
    """
    # If the engine is unsupported, return a clear unavailable block
    if not engine_entry["supported"]:
        return {
            "available": False,
            "engine": engine_id,
            "engine_supported": False,
            "unavailable_reason": engine_entry.get("unsupported_reason",
                f"Engine '{engine_id}' is not yet supported."),
            "flash_usage": None,
            "ram_usage": None,
            "inferencing_time": None,
            "recommendations": [],
            "recommendations_reason": None,
        }

    # No model artifact at all
    if not selected_model:
        return {
            "available": False,
            "engine": engine_id,
            "engine_supported": True,
            "unavailable_reason": "No model artifact available for this variant.",
            "flash_usage": None,
            "ram_usage": None,
            "inferencing_time": None,
            "recommendations": [],
            "recommendations_reason": None,
        }

    # ── Supported engine with a real model artifact ──────────────────────
    # "tflite" is a representative build format for the panel's own on-device
    # estimate (this variant is always a .tflite artifact); `device_slug`'s
    # catalog deploy_target still wins if the project's target device
    # resolves to something else, exactly as `resolve_build_format` does for
    # a real build request.
    build_format = resolve_build_format(db, target="tflite", device_profile=device_slug, deployment_format="pe")
    result = estimation.estimate(selected_model, build_format, device_slug=device_slug, db=db)

    return {
        "available": result.compatible,
        "engine": engine_id,
        "engine_supported": True,
        "unavailable_reason": None if result.compatible else result.compatibility.message,
        "flash_usage": result.flash.to_dict(),
        "ram_usage": result.ram.to_dict(),
        "inferencing_time": result.latency.to_dict(),
        "recommendations": [r.to_dict() for r in result.recommendations],
        "recommendations_reason": result.recommendations_reason,
    }


@router.get("/{model_id}")
def get_model(
    model_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Get a specific trained model by ID."""
    model = assert_trained_model_owner(db, model_id, current_user)
    return _model_dict(model)


@router.get("/{model_id}/download")
def download_model(
    model_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Download the raw model file (TFLite or Keras).
    Returns the file bytes as a streaming response.
    """
    model = assert_trained_model_owner(db, model_id, current_user)

    try:
        model_bytes = storage.download_bytes(model.storage_key)
    except Exception as e:
        raise HTTPException(500, f"Failed to download model file: {e}")

    # tflite is the one format with two sibling artifacts per version
    # (float32/int8) — include the variant so downloading both doesn't
    # overwrite the same filename on disk.
    variant = (model.model_metadata or {}).get("variant")
    tflite_name = (
        f"model_v{model.version}_{variant}.tflite" if variant
        else f"model_v{model.version}.tflite"
    )
    filename_map = {
        "tflite": tflite_name,
        "keras":  f"model_v{model.version}.keras",
        "onnx":   f"model_v{model.version}.onnx",
    }
    filename = filename_map.get(model.format, f"model_v{model.version}.bin")
    media_type_map = {
        "tflite": "application/octet-stream",
        "keras":  "application/octet-stream",
        "onnx":   "application/octet-stream",
    }
    media_type = media_type_map.get(model.format, "application/octet-stream")

    return StreamingResponse(
        io.BytesIO(model_bytes),
        media_type=media_type,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.delete("/{model_id}", status_code=204)
def delete_model(
    model_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Delete a trained model artifact from storage and DB."""
    model = assert_trained_model_owner(db, model_id, current_user)

    try:
        storage.delete_file(model.storage_key)
    except Exception:
        pass  # Best-effort storage cleanup

    db.delete(model)
    db.commit()
    return Response(status_code=204)
