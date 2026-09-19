"""
Post-Processing endpoints — threshold tuning, anomaly detection calibration,
sliding-window smoothing, and per-class confidence configuration.

This mirrors Edge Impulse's Post-processing page which lets users configure
how raw model output scores are converted into final labels before deployment.
"""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import numpy as np

from app.core.database import get_db
from app.core.auth import get_current_user
from app.core.authz import (
    assert_impulse_owner,
    assert_training_job_owner,
    assert_trained_model_owner,
)
from app.models.user import User, TrainingJob, TrainedModel, Impulse
from app.core.storage import storage

router = APIRouter()


# ─── Schemas ──────────────────────────────────────────────────────────────────

class ThresholdConfig(BaseModel):
    label: str
    threshold: float = 0.8
    enabled: bool = True


class SmoothingConfig(BaseModel):
    enabled: bool = False
    window_size: int = 5                  # number of inference frames to smooth over
    method: str = "mean"                  # mean | max | majority_vote


class AnomalyConfig(BaseModel):
    enabled: bool = False
    threshold: float = 0.3               # anomaly score above this = anomalous
    mode: str = "gmm"                    # gmm | zscore


class PostProcessingConfig(BaseModel):
    model_config = {"protected_namespaces": ()}
    model_id: str
    min_confidence: float = 0.6          # global minimum — reject below this
    uncertainty_label: str = "uncertain" # label when no class clears threshold
    per_class_thresholds: List[ThresholdConfig] = []
    smoothing: SmoothingConfig = SmoothingConfig()
    anomaly: AnomalyConfig = AnomalyConfig()


class PostProcessingUpdateRequest(BaseModel):
    model_config = {"protected_namespaces": ()}
    model_id: str
    config: PostProcessingConfig


class LivePostProcessRequest(BaseModel):
    """Run post-processing on a raw scores vector (for preview/testing)."""
    model_config = {"protected_namespaces": ()}
    model_id: str
    raw_scores: List[float]                # softmax output from model
    label_names: Optional[List[str]] = None
    config: Optional[PostProcessingConfig] = None


# ─── Routes ───────────────────────────────────────────────────────────────────

@router.get("/model/{model_id}")
def get_post_processing_config(
    model_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Return the current post-processing config for a trained model.
    If none has been saved yet, returns sensible defaults derived from
    the model's label names.
    """
    model = assert_trained_model_owner(db, model_id, current_user)

    meta = model.model_metadata or {}
    label_names = meta.get("label_names", [])

    # Load stored config or build defaults
    stored = (meta.get("post_processing") or {})
    per_class = stored.get("per_class_thresholds") or [
        {"label": lbl, "threshold": 0.8, "enabled": True}
        for lbl in label_names
    ]

    return {
        "model_id": model_id,
        "label_names": label_names,
        "config": {
            "model_id": model_id,
            "min_confidence": stored.get("min_confidence", 0.6),
            "uncertainty_label": stored.get("uncertainty_label", "uncertain"),
            "per_class_thresholds": per_class,
            "smoothing": stored.get("smoothing", {
                "enabled": False,
                "window_size": 5,
                "method": "mean",
            }),
            "anomaly": stored.get("anomaly", {
                "enabled": False,
                "threshold": 0.3,
                "mode": "gmm",
            }),
        },
    }


@router.put("/model/{model_id}")
def update_post_processing_config(
    model_id: str,
    req: PostProcessingUpdateRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Persist the post-processing config inside the model's metadata JSON."""
    model = assert_trained_model_owner(db, model_id, current_user)

    meta = dict(model.model_metadata or {})
    meta["post_processing"] = req.config.dict()
    model.model_metadata = meta
    db.commit()
    db.refresh(model)

    return {"message": "Post-processing config saved", "model_id": model_id}


@router.post("/apply")
def apply_post_processing(
    req: LivePostProcessRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Apply post-processing rules to a raw softmax scores vector and return
    the final label decision. Used for live preview in the UI without
    running a full inference cycle.
    """
    model = assert_trained_model_owner(db, req.model_id, current_user)

    meta = model.model_metadata or {}
    label_names = req.label_names or meta.get("label_names", [])

    config = req.config
    if config is None:
        stored = meta.get("post_processing", {})
        config = PostProcessingConfig(
            model_id=req.model_id,
            min_confidence=stored.get("min_confidence", 0.6),
            uncertainty_label=stored.get("uncertainty_label", "uncertain"),
            per_class_thresholds=[
                ThresholdConfig(**t) for t in stored.get("per_class_thresholds", [])
            ],
        )

    scores = np.array(req.raw_scores, dtype=np.float32)

    # Softmax normalize if needed
    if scores.sum() < 0.99 or scores.sum() > 1.01:
        e = np.exp(scores - scores.max())
        scores = e / e.sum()

    result = _apply_rules(scores, label_names, config)
    return result


@router.get("/impulse/{impulse_id}/latest-model")
def get_latest_model_for_impulse(
    impulse_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Return the latest trained model for an impulse — used by the
    post-processing page to auto-select the correct model.
    """
    impulse = assert_impulse_owner(db, impulse_id, current_user)

    # Find the most recent completed training job
    job = (
        db.query(TrainingJob)
        .filter(
            TrainingJob.impulse_id == impulse_id,
            TrainingJob.status == "completed",
        )
        .order_by(TrainingJob.created_at.desc())
        .first()
    )
    if not job:
        raise HTTPException(404, "No completed training job found for this impulse")

    models = (
        db.query(TrainedModel)
        .filter(TrainedModel.training_job_id == job.id)
        .all()
    )
    if not models:
        raise HTTPException(404, "No trained models found")

    tflite = next((m for m in models if m.format == "tflite"), None)
    keras  = next((m for m in models if m.format == "keras"),  None)
    best   = tflite or keras or models[0]

    meta = best.model_metadata or {}
    return {
        "model_id":       best.id,
        "format":         best.format,
        "version":        best.version,
        "label_names":    meta.get("label_names", []),
        "input_shape":    meta.get("input_shape", []),
        "architecture":   meta.get("architecture", "dense"),
        "training_job_id": job.id,
        "best_accuracy":  job.best_accuracy,
        "created_at":     best.created_at.isoformat(),
    }


@router.get("/trained-models/job/{job_id}")
def get_trained_models_for_job(
    job_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """List all trained model artifacts (keras, tflite) for a training job."""
    job = assert_training_job_owner(db, job_id, current_user)

    models = (
        db.query(TrainedModel)
        .filter(TrainedModel.training_job_id == job_id)
        .all()
    )
    return [_model_to_dict(m) for m in models]


# ─── Internal helpers ─────────────────────────────────────────────────────────

def _apply_rules(
    scores: np.ndarray,
    label_names: list,
    config: PostProcessingConfig,
) -> dict:
    """
    Apply threshold, per-class, and smoothing rules to a scores vector.
    Returns the final label decision with confidence.
    """
    best_idx   = int(np.argmax(scores))
    best_score = float(scores[best_idx])
    best_label = label_names[best_idx] if best_idx < len(label_names) else str(best_idx)

    # Build per-class threshold lookup
    threshold_map: Dict[str, float] = {
        t.label: t.threshold
        for t in config.per_class_thresholds
        if t.enabled
    }

    # Check global minimum first
    if best_score < config.min_confidence:
        return {
            "label":      config.uncertainty_label,
            "confidence": best_score,
            "rejected":   True,
            "reason":     f"Below global min_confidence ({config.min_confidence:.2f})",
            "scores":     _scores_dict(scores, label_names),
        }

    # Check per-class threshold
    class_threshold = threshold_map.get(best_label, config.min_confidence)
    if best_score < class_threshold:
        return {
            "label":      config.uncertainty_label,
            "confidence": best_score,
            "rejected":   True,
            "reason":     f"Below per-class threshold for '{best_label}' ({class_threshold:.2f})",
            "scores":     _scores_dict(scores, label_names),
        }

    return {
        "label":      best_label,
        "confidence": best_score,
        "rejected":   False,
        "reason":     "Accepted",
        "scores":     _scores_dict(scores, label_names),
    }


def _scores_dict(scores: np.ndarray, label_names: list) -> dict:
    return {
        label_names[i] if i < len(label_names) else str(i): float(scores[i])
        for i in range(len(scores))
    }


def _model_to_dict(m: TrainedModel) -> dict:
    return {
        "id":              m.id,
        "training_job_id": m.training_job_id,
        "version":         m.version,
        "format":          m.format,
        "file_size_bytes": m.file_size_bytes,
        "model_metadata":  m.model_metadata,
        "created_at":      m.created_at.isoformat(),
    }
