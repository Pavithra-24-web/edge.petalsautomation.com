"""
Evaluation (Model Testing) endpoints — run test inference across a dataset,
return per-sample predictions, confusion matrix, and classification report.

Mirrors Edge Impulse's 'Model testing' page which classifies all test-split
samples and shows accuracy, per-class precision/recall, and a confusion matrix.
"""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import Optional, List
import numpy as np
import logging

from app.core.database import get_db
from app.core.auth import get_current_user
from app.core.authz import (
    assert_impulse_owner,
    assert_training_job_owner,
    assert_trained_model_owner,
)
from app.core.storage import storage
from app.models.user import (
    User, TrainingJob, TrainedModel, Impulse, Sample, Label, SampleType, JobStatus
)
from app.ml.dsp.processor import DSPProcessor

router = APIRouter()
logger = logging.getLogger(__name__)


def _combine_dsp_features(feature_list):
    if not feature_list:
        return np.empty((0,), dtype=np.float32)
    if len(feature_list) == 1:
        return np.asarray(feature_list[0], dtype=np.float32)
    return np.concatenate([np.asarray(feat, dtype=np.float32).flatten() for feat in feature_list])


# ─── Schemas ──────────────────────────────────────────────────────────────────

class ClassifyTestSetRequest(BaseModel):
    model_config = {"protected_namespaces": ()}
    model_id: str
    impulse_id: str
    sample_type: str = "testing"       # testing | training | all


# ─── Serializers ──────────────────────────────────────────────────────────────

def _job_dict(j: TrainingJob) -> dict:
    ep = j.extra_params or {}
    return {
        "id":                    j.id,
        "impulse_id":            j.impulse_id,
        "status":                j.status,
        "epochs":                j.epochs,
        "best_accuracy":         j.best_accuracy,
        "best_loss":             j.best_loss,
        "final_accuracy":        j.final_accuracy,
        "training_history":      j.training_history,
        "architecture":          ep.get("architecture"),
        "confusion_matrix":      j.confusion_matrix,
        "classification_report": j.classification_report,
        "started_at":            j.started_at.isoformat() if j.started_at else None,
        "completed_at":          j.completed_at.isoformat() if j.completed_at else None,
        "created_at":            j.created_at.isoformat(),
    }


# ─── Routes ───────────────────────────────────────────────────────────────────

@router.get("/job/{job_id}/confusion-matrix")
def get_confusion_matrix(
    job_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Return the confusion matrix and classification report from a training job."""
    job = assert_training_job_owner(db, job_id, current_user)

    cr = job.classification_report or {}
    is_yolo_pro = cr.get("yolo_pro_eval_status") is not None

    response = {
        "confusion_matrix":      job.confusion_matrix,
        "classification_report": cr,
        "best_accuracy":         job.best_accuracy,
        "training_history":      job.training_history,
    }

    if is_yolo_pro:
        response["yolo_pro_detection_metrics"] = {
            "eval_status": cr.get("yolo_pro_eval_status"),
            "eval_error":  cr.get("yolo_pro_eval_error"),
            "map":         cr.get("map"),
            "map50":       cr.get("map50"),
            "map75":       cr.get("map75"),
            "precision":   cr.get("precision"),
            "recall":      cr.get("recall"),
            "per_class":   cr.get("per_class", {}),
        }

    return response


@router.get("/job/{job_id}/metrics")
def get_metrics(
    job_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Return epoch-level training metrics for the learning curves chart."""
    job = assert_training_job_owner(db, job_id, current_user)
    history = job.training_history or {}
    cr      = job.classification_report or {}

    is_yolo_pro = cr.get("yolo_pro_eval_status") is not None

    response = {
        "epochs_completed": len(history.get("loss", history.get("accuracy", []))),
        "best_accuracy":    job.best_accuracy,   # None for FOMO / YOLO-Pro
        "best_loss":        job.best_loss,
        "final_accuracy":   job.final_accuracy,  # None for FOMO / YOLO-Pro
        "is_fomo":          history.get("is_fomo", False),
        "is_yolo_pro":      history.get("is_yolo_pro", False),
        "metric_note":      history.get("metric_note"),
        "training_history": history,
    }

    if is_yolo_pro:
        response["yolo_pro_detection_metrics"] = {
            "eval_status": cr.get("yolo_pro_eval_status"),
            "eval_error":  cr.get("yolo_pro_eval_error"),
            "map":         cr.get("map"),
            "map50":       cr.get("map50"),
            "map75":       cr.get("map75"),
            "precision":   cr.get("precision"),
            "recall":      cr.get("recall"),
            "per_class":   cr.get("per_class", {}),
        }

    return response


@router.get("/impulse/{impulse_id}")
def get_impulse_evaluation(
    impulse_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Return the evaluation results for the latest completed training job of
    an impulse. This is the primary entry point for the Model Testing page.
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

    tflite = (
        db.query(TrainedModel)
        .filter(
            TrainedModel.training_job_id == job.id,
            TrainedModel.format == "tflite",
        )
        .first()
    )
    meta = tflite.model_metadata if tflite else {}
    label_names = (meta or {}).get("label_names", [])
    architecture = (meta or {}).get("architecture", "")
    _out_type = (meta or {}).get("output_type", "")
    is_detection = (
        _out_type == "detection_heatmap"
        or _out_type == "yolo_pro_detection"
        or "fomo" in architecture.lower()
    )

    cr = job.classification_report or {}
    is_yolo_pro_eval = cr.get("yolo_pro_eval_status") is not None

    response = {
        "job":                   _job_dict(job),
        "model_id":              tflite.id if tflite else None,
        "label_names":           label_names,
        "architecture":          architecture,
        "is_detection":          is_detection,
        "accuracy":              job.best_accuracy,
        "confusion_matrix":      job.confusion_matrix,
        "classification_report": cr,
        "training_history":      job.training_history,
    }

    if is_yolo_pro_eval:
        response["yolo_pro_detection_metrics"] = {
            "eval_status": cr.get("yolo_pro_eval_status"),
            "eval_error":  cr.get("yolo_pro_eval_error"),
            "map":         cr.get("map"),
            "map50":       cr.get("map50"),
            "map75":       cr.get("map75"),
            "precision":   cr.get("precision"),
            "recall":      cr.get("recall"),
            "per_class":   cr.get("per_class", {}),
        }

    return response


@router.post("/classify-test-set")
def classify_test_set(
    req: ClassifyTestSetRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Run model inference on all test-split samples and return per-sample
    predictions. Matches Edge Impulse's 'Classify all' button in Model Testing.

    This runs synchronously — for large datasets consider a background task.
    """
    import json

    model = assert_trained_model_owner(db, req.model_id, current_user)
    if model.format != "tflite":
        raise HTTPException(400, "Only TFLite models are supported for test classification")

    impulse = assert_impulse_owner(db, req.impulse_id, current_user)

    meta = model.model_metadata or {}
    label_names = meta.get("label_names", [])

    # Detect model type early so we can branch the sample query.
    _model_out_type = meta.get("output_type", "")
    is_fomo_model = (
        _model_out_type == "detection_heatmap"
        or _model_out_type == "yolo_pro_detection"
        or "fomo" in meta.get("architecture", "").lower()
    )

    is_ssd_model = _model_out_type == "ssd_detection"   

    # YOLO-Pro multi-output models are not supported for sample-by-sample
    # classification via this endpoint — return an explanatory error early.
    if _model_out_type == "yolo_pro_detection":
        return {
            "total_samples": 0,
            "total_classification": 0,
            "correct": 0,
            "accuracy": None,
            "label_names": meta.get("label_names", []),
            "results": [],
            "yolo_pro_note": (
                "YOLO-Pro is a multi-output anchor-free detector. "
                "Per-sample classification is not supported via this endpoint. "
                "Use the Model panel to view training loss and export metrics."
            ),
        }

    # MobileNetV2 SSD is an anchor-based detector — like YOLO-Pro it produces
    # bounding boxes, not a single per-sample class. Per-sample "classification"
    # accuracy is not meaningful, and the detection metrics (mAP / precision /
    # recall) are surfaced from the training-time evaluator instead. Gate it out
    # here (mirrors the YOLO-Pro gate above) rather than running ad-hoc decode.
    if is_ssd_model:
        return {
            "total_samples": 0,
            "total_classification": 0,
            "correct": 0,
            "accuracy": None,
            "label_names": meta.get("label_names", []),
            "results": [],
            "ssd_note": (
                "MobileNetV2 SSD FPN-Lite is an object detector. "
                "Per-sample classification is not supported via this endpoint. "
                "Use the Model panel to view detection metrics (mAP / precision / recall)."
            ),
        }

    # Load TFLite bytes once
    try:
        tflite_bytes = storage.download_bytes(model.storage_key)
    except Exception as e:
        raise HTTPException(500, f"Failed to load model: {e}")

    # Build interpreter
    interp, inp_detail, out_detail = _build_interpreter(tflite_bytes)

    # Determine which samples to classify
    sample_filter = {
        "testing":  [SampleType.testing],
        "training": [SampleType.training, SampleType.automatic],
        "all":      [SampleType.testing, SampleType.training, SampleType.automatic],
    }.get(req.sample_type, [SampleType.testing])

    labels = db.query(Label).filter(Label.project_id == impulse.project_id).all()
    label_id_to_name = {l.id: l.name for l in labels}

    # FOMO: include samples without a top-level label_id — their ground-truth
    # comes from per-box labels in extra_metadata["boundingBoxes"].
    # Classification: only samples with an assigned label are meaningful.
    if is_fomo_model:
        samples = (
            db.query(Sample)
            .filter(
                Sample.project_id == impulse.project_id,
                Sample.sample_type.in_(sample_filter),
            )
            .all()
        )
    else:
        samples = (
            db.query(Sample)
            .filter(
                Sample.project_id == impulse.project_id,
                Sample.sample_type.in_(sample_filter),
                Sample.label_id.isnot(None),
            )
            .all()
        )

    dsp_blocks = impulse.dsp_blocks or [{"type": "raw", "params": {}}]

    # Reuse the precomputed DSP feature cache (features.npz) when it is fresh for
    # this impulse, so we skip the per-sample S3 download + DSP recompute that the
    # DSP worker already did.  Restricted to non-image impulses: off the image
    # path merge_image_params is a no-op, so the cached features are byte-identical
    # to what this endpoint would compute live, and results are unchanged.  Any
    # problem (missing/stale cache, hash mismatch) silently falls back to the live
    # download + DSP path below — identical to the previous behaviour.
    cached_features: dict = {}
    if impulse.input_type != "image":
        try:
            import io as _io
            import json as _json
            from app.workers.dsp_worker import (
                compute_dsp_config_hash,
                make_features_storage_key,
            )

            _cache_bytes = storage.download_bytes(
                make_features_storage_key(impulse.project_id, impulse.id)
            )
            with np.load(_io.BytesIO(_cache_bytes), allow_pickle=True) as _npz:
                if "meta_json" in _npz:
                    _cmeta = _json.loads(str(_npz["meta_json"][0]))
                    if _cmeta.get("dsp_config_hash") == compute_dsp_config_hash(dsp_blocks):
                        _ids = _npz["ids"]
                        _X = _npz["X"]
                        cached_features = {
                            str(_ids[i]): _X[i] for i in range(len(_ids))
                        }
        except Exception:
            cached_features = {}

    results = []
    correct = 0

    for sample in samples:
        try:
            true_label = label_id_to_name.get(sample.label_id, "unknown")

            cached = cached_features.get(sample.id)
            if cached is not None:
                # Cache hit — no download, no DSP recompute.
                features = np.asarray(cached, dtype=np.float32)
            else:
                raw = storage.download_bytes(sample.storage_key)
                values = raw

                # DSP. Preserve image tensors for single-block image pipelines;
                # otherwise flatten and concatenate as a feature vector.
                all_features = []
                for block_cfg in dsp_blocks:
                    proc = DSPProcessor(
                        block_type=block_cfg.get("type", "raw"),
                        params=block_cfg.get("params", {}),
                        frequency_hz=sample.frequency_hz or impulse.frequency_hz,
                    )
                    all_features.append(proc.extract(values))

                features = _combine_dsp_features(all_features)

            scores = _run_inference(interp, inp_detail, out_detail, features)

            # FOMO Handling: output is (GridH, GridW, Classes+1)
            # Standard classification handling for everything else
            is_fomo = len(out_detail["shape"]) == 4
            
            if is_fomo:
                # FOMO output is a spatial heatmap (1, grid_h, grid_w, num_classes+1).
                # Detection accuracy is NOT comparable to classification accuracy —
                # do not use predicted-label == true-label string comparison.
                # Instead return the best-cell detection result with confidence,
                # and mark the sample as a detection result (correct=None).
                fomo_map = scores.reshape(out_detail["shape"][1:])
                object_map = fomo_map[:, :, 1:]  # slice off background channel
                best_cell_idx = np.unravel_index(np.argmax(object_map), object_map.shape)
                best_class_idx = int(best_cell_idx[2])
                confidence = float(object_map[best_cell_idx])

                predicted = label_names[best_class_idx] if best_class_idx < len(label_names) else str(best_class_idx)
                true_label = label_id_to_name.get(sample.label_id, "unknown")

                # Do NOT count toward accuracy — detection evaluation requires
                # IoU-based metrics, not per-sample label matching.
                results.append({
                    "sample_id":       sample.id,
                    "filename":        sample.filename,
                    "true_label":      true_label,
                    "predicted_label": predicted,
                    "confidence":      confidence,
                    "correct":         None,   # not computed — use detection metrics
                    "is_fomo":         True,
                    "map_shape":       list(out_detail["shape"]),
                    "detection_note":  "FOMO heatmap output; accuracy not computed via label matching",
                })
            else:
                best_idx = int(np.argmax(scores))
                predicted = label_names[best_idx] if best_idx < len(label_names) else str(best_idx)
                true_label = label_id_to_name.get(sample.label_id, "unknown")
                is_correct = predicted == true_label
                if is_correct:
                    correct += 1

                results.append({
                    "sample_id":      sample.id,
                    "filename":       sample.filename,
                    "true_label":     true_label,
                    "predicted_label": predicted,
                    "confidence":     float(scores[best_idx]),
                    "correct":        is_correct,
                    "scores": {
                        label_names[i] if i < len(label_names) else str(i): float(scores[i])
                        for i in range(len(scores))
                    },
                })
        except Exception as e:
            logger.warning(f"Failed to classify sample {sample.id}: {e}")
            results.append({
                "sample_id":      sample.id,
                "filename":       sample.filename,
                "true_label":     label_id_to_name.get(sample.label_id, "unknown"),
                "predicted_label": None,
                "confidence":     None,
                "correct":        False,
                "error":          str(e),
            })

    # Only count classification samples (correct != None) in accuracy.
    # FOMO detection samples set correct=None and are excluded.
    classification_results = [r for r in results if r.get("correct") is not None]
    fomo_results = [r for r in results if r.get("is_fomo")]
    total_classification = len(classification_results)
    accuracy = correct / total_classification if total_classification > 0 else 0.0

    response: dict = {
        "total_samples":            len(results),
        "total_classification":     total_classification,
        "correct":                  correct,
        "accuracy":                 accuracy,
        "label_names":              label_names,
        "results":                  results,
    }
    if fomo_results:
        response["fomo_detection_count"] = len(fomo_results)
        response["fomo_note"] = (
            "FOMO (object detection) results are included but excluded from accuracy. "
            "Post-training detection metrics use grid-cell center-point matching — "
            "see the Model panel for precision / recall / F1 results."
        )
    return response


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _build_interpreter(tflite_bytes: bytes):
    try:
        import tflite_runtime.interpreter as tflite
        Interpreter = tflite.Interpreter
    except ImportError:
        import tensorflow as tf
        Interpreter = tf.lite.Interpreter

    interp = Interpreter(model_content=tflite_bytes)
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    out = interp.get_output_details()[0]
    return interp, inp, out


def _run_inference(interp, inp_detail, out_detail, features: np.ndarray) -> np.ndarray:
    x = features.reshape(inp_detail["shape"]).astype(np.float32)

    scale, zp = inp_detail.get("quantization", (0, 0))
    if scale != 0:
        x = (x / scale + zp).astype(np.int8)

    interp.set_tensor(inp_detail["index"], x)
    interp.invoke()

    output = interp.get_tensor(out_detail["index"]).astype(np.float32)
    out_scale, out_zp = out_detail.get("quantization", (0, 0))
    if out_scale != 0:
        output = (output - out_zp) * out_scale

    # FOMO Detection: Shape (1, H, W, N)
    if output.ndim == 4:
        # Sigmoid output (from FOMO head), already normalized [0,1], no softmax
        return output.flatten()

    # Classification Softmax
    flat = output.flatten()
    e = np.exp(flat - flat.max())
    return e / e.sum()