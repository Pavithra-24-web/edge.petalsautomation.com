"""
Training Worker — Celery task that runs the full ML training pipeline.

Flow:
  1. Load impulse config from DB
  2. Load all labeled samples for the project
  3. Run DSP feature extraction on each sample
  4. Build the model architecture
  5. Train with callbacks (early stopping, checkpointing)
  6. Evaluate on test split
  7. Save Keras + TFLite models to S3
  8. Write metrics back to DB
"""
import os
import io
import json
import math
import time
import hashlib
import logging
import tempfile
import numpy as np
import tensorflow as tf
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from app.ml.gpu_utils import assert_job_device, get_strategy, get_worker_device

from app.workers.celery_app import celery_app
from app.core.database import SessionLocal
from app.core.storage import storage
from app.models.user import (
    TrainingJob, Impulse, Sample, Label, TrainedModel, JobStatus, SampleType
)
from app.ml.dsp.processor import DSPProcessor, merge_image_params, _uses_fomo_min_resolution
from app.ml.training.model_builder import build_model
from app.ml.yolo_pro_worker import (
    is_yolo_pro,
    run_yolo_pro_training,
    _sanitize_json,
    _snap32,
    YOLO_PRO_MIN_INPUT_SIZE,
)
from app.ml.mobilenetv2_ssd_worker import (
    is_mobilenetv2_ssd,
    run_mobilenetv2_ssd_training,
    MV2_SSD_INPUT_H,
    MV2_SSD_INPUT_W,
)
# Shared cache helpers (also used by dsp_worker)
from app.workers.dsp_worker import (
    FEATURE_CACHE_VERSION,
    compute_dsp_config_hash,
    make_features_storage_key,
)
from app.workers.sample_utils import (
    IS_BACKGROUND_KEY,
    UNLABELED_NAMES,
    apply_train_subset,
    is_background_sample,
    label_name_is_placeholder,
    is_sample_usable,
    iter_prefetched,
    normalize_bounding_boxes,
)
from app.ml.fomo_evaluator import (
    decode_fomo_heatmap,
    gt_cells_from_heatmap,
    match_cells,
    aggregate_metrics,
    assert_grid_shape_match,
    log_fomo_run_diagnostics,
    log_fomo_v2_diagnostics,
    resolve_fomo_threshold,
    ThresholdSource,
    FOMO_SWEEP_THRESHOLDS,
    _macro_f1_from_arrays,
)

logger = logging.getLogger(__name__)

# Structured per-phase training metrics → logs/training.log; feature-cache
# hit/miss is a DSP concern and is routed to logs/dsp.log.
from app.core.logging_config import get_logger as _get_channel_logger, log_event as _log_event
_training_log = _get_channel_logger("training")
_dsp_log = _get_channel_logger("dsp")


def _fomo_training_target_config(fomo_version: int = 1) -> Dict[str, object]:
    """
    Return the training-target policy for the selected FOMO version.

    v1 is preserved exactly: strict single-cell targets with no spread.
    v2 keeps the exact center as the dominant target and only leaves a tiny
    non-competing neighbor hint during training. Exact-center ranking is
    enforced primarily through the loss, not through softened targets.
    """
    if fomo_version == 2:
        return {
            "spread": False,
            "mode": "strict_center",
            "center_value": 1.0,
            "neighbor_value": 0.0,
            "diagonal_value": 0.0,
            "reason": (
                "v2 training uses strict single-cell center targets to match "
                "the evaluation contract exactly; the 0.02 neighbor hints were "
                "removed because they encouraged non-zero class probability at "
                "ortho neighbors, which (combined with the NMS fix that now uses "
                "strict >) could still produce tie candidates and suppress the "
                "true GT center."
            ),
        }
    return {
        "spread": False,
        "mode": "strict_center",
        "center_value": 1.0,
        "neighbor_value": 0.0,
        "diagonal_value": 0.0,
        "reason": (
            "legacy v1 strict single-cell targets"
        ),
    }


def _fomo_loss_tuning_config(fomo_version: int = 1) -> Dict[str, float]:
    """
    Return version-specific loss tuning knobs.

    v1 remains on the legacy hard-center behavior.
    v2 adds extra center-cell dominance plus an explicit neighbor-ranking
    penalty so the GT center outranks nearby cells more decisively.
    """
    if fomo_version == 2:
        return {
            "soft_obj_threshold": 0.50,
            "soft_neighbor_weight": 1.0,
            # Raised again for v2: ranking loss already improves ordering, but
            # exact-hit recall still depends on the GT center reaching a usable
            # absolute probability. Higher dominance pushes the
            # GT center probability up so a moderate threshold (0.4–0.6) fires,
            # instead of the model needing threshold≥0.90 to suppress FPs.
            "center_dominance": 5.25,
            # Reduced from 4.50/0.25: the old sum-over-8 formulation could
            # additively explode to ~1.125 per cell, overpowering detection loss
            # and spiking the threshold.  Mean-normalization (below) bounds each
            # cell's ranking contribution to ≤ penalty × margin = 0.10.
            "center_rank_penalty": 1.20,
            "center_rank_margin": 0.12,
            # Direct local-argmax penalty: the strongest nearby rival matters
            # most for exact-cell evaluation, so v2 explicitly trains the GT
            # center to beat the max nearby cell by a healthy margin.
            "center_peak_penalty": 2.10,
            "center_peak_margin": 0.22,
            # Orthogonal neighbours are the dominant off-by-one failure mode on
            # stride-8 grids, so they receive the strongest peak-placement
            # penalty. Diagonals still matter, but less.
            "center_peak_orthogonal_scale": 1.35,
            "center_peak_diagonal_scale": 0.65,
            # all-pairs: each of the 8 surrounding cells is penalised
            # independently rather than only the single argmax neighbour.
            # This gives direct gradient to every offending cell simultaneously
            # — the root-cause fix for high outranked_rate.
            "center_rank_all_pairs": True,
        }
    return {
        "soft_obj_threshold": 0.50,
        "soft_neighbor_weight": 1.0,
        "center_dominance": 1.0,
        "center_rank_penalty": 0.0,
        "center_rank_margin": 0.0,
        "center_peak_penalty": 0.0,
        "center_peak_margin": 0.0,
        "center_peak_orthogonal_scale": 1.0,
        "center_peak_diagonal_scale": 1.0,
        "center_rank_all_pairs": False,
    }


def _fomo_bg_weight_bounds(fomo_version: int = 1) -> Tuple[float, float, float]:
    """
    Return (scale, floor, ceiling) for adaptive bg_weight selection.

    v1 is unchanged.
    v2 keeps its own tuning path only when fomo_version == 2 and now permits
    materially stronger background suppression than the previous 0.015 cap.
    """
    if fomo_version == 2:
        return 1.2, 0.008, 0.03
    return 0.8, 0.015, 0.05


def _fomo_finetune_config(fomo_version: int = 1) -> Dict[str, object]:
    """
    Return the two-phase training schedule for the selected FOMO version.

    v1 is preserved exactly.
    v2 uses a longer warmup, a slightly gentler fine-tune LR, and live
    BatchNorm during full-backbone fine-tuning so phase 2 can adapt from the
    best warmup checkpoint instead of getting stuck with stale running stats.
    """
    if fomo_version == 2:
        return {
            "warmup_epochs": 16,
            "fine_tune_lr": 8.5e-5,
            "unfreeze_strategy": "full_backbone_live_bn",
            "unfreeze_last_layers": None,
            "freeze_batchnorm": False,
            "fine_tune_focal_bg_gamma": 3.5,
        }
    return {
        "warmup_epochs": 15,
        "fine_tune_lr": 1e-4,
        "unfreeze_strategy": "full_backbone",
        "unfreeze_last_layers": None,
        "freeze_batchnorm": False,
        "fine_tune_focal_bg_gamma": 3.5,
    }


def _resolve_fomo_phase2_patience(total_epochs: int, warmup_epochs: int) -> int:
    """
    Derive FOMO Phase 2 patience from the configured training budget.

    Warmup stays fixed and unchanged. Early stopping only applies during
    fine-tuning, so patience tracks that available Phase 2 budget: we allow
    roughly half of the configured fine-tune window to stagnate before
    stopping, rounded up so short runs still get a fair chance to improve.
    """
    phase2_epochs = max(1, int(total_epochs) - int(warmup_epochs))
    return max(1, math.ceil(phase2_epochs / 2))


def _fomo_sparse_positive_config(object_ratio: float, fomo_version: int = 1) -> Dict[str, float]:
    """
    Return sparse-positive loss tuning knobs.

    v1 remains unchanged.
    v2 gets extra object-cell pressure on very sparse stride-8 grids without
    reintroducing binary spread to the training targets.
    """
    if fomo_version != 2:
        return {
            "obj_weight": 1.0,
            "warmup_focal_bg_gamma": 2.0,
            "fine_tune_focal_bg_gamma": 3.5,
        }

    safe_ratio = max(float(object_ratio), 1e-6)
    sparse_scale = float(np.clip(0.012 / safe_ratio, 1.0, 2.2))
    # Ceiling raised 2.4 → 3.2 to allow stronger object-cell pressure when
    # the stride-8 grid is very sparse (cat/dog ≈ 1–2 cells per 256-cell image).
    obj_weight = float(np.clip(1.20 * sparse_scale, 1.35, 3.2))
    return {
        "obj_weight": obj_weight,
        # warmup 2.2 → 2.5: slightly harder BG mining during head-only phase
        # keeps background cells from firing at high confidence early on.
        "warmup_focal_bg_gamma": 2.5,
        # fine-tune 3.2 → 3.8: stronger focal BG suppression after backbone
        # unfreezing.  Forces the backbone to sharpen class discrimination
        # rather than firing broadly at high confidence.
        "fine_tune_focal_bg_gamma": 3.8,
    }


def _fomo_bg_obj_coscale(current_bg_weight: float, obj_scale: float, fomo_version: int, clamp_lo: float, clamp_hi: float) -> float:
    """
    Co-scale bg_weight when obj_weight changes.

    v1 preserves proportional co-scaling.
    v2 only applies a partial co-scale so center-confidence boosts do not make
    the model overly cautious during fine-tuning.
    """
    if fomo_version == 2:
        effective_scale = 1.0 + max(0.0, obj_scale - 1.0) * 0.40
    else:
        effective_scale = obj_scale
    return float(np.clip(current_bg_weight * effective_scale, clamp_lo, clamp_hi))


def _apply_fomo_finetune_trainable_layers(model, fomo_version: int = 1) -> Dict[str, object]:
    """
    Apply the fine-tune trainable-layer policy and return a small summary.

    v1 keeps the legacy full-backbone unfreeze.
    v2 now fine-tunes with live BatchNorm so the fused stride-8/stride-16
    features can adapt to the current dataset during phase 2.
    """
    cfg = _fomo_finetune_config(fomo_version)
    strategy = str(cfg["unfreeze_strategy"])
    head_names = frozenset({"fomo_head_conv", "fomo_head"})
    freeze_batchnorm = bool(cfg.get("freeze_batchnorm", False))

    if strategy in {"full_backbone", "full_backbone_frozen_bn", "full_backbone_live_bn"}:
        for layer in model.layers:
            is_bn = "batchnormalization" in layer.__class__.__name__.lower()
            layer.trainable = not (freeze_batchnorm and is_bn)
        return {
            "strategy": strategy,
            "trainable_layers": sum(1 for layer in model.layers if layer.trainable),
            "tail_layers": len(model.layers),
        }

    tail_count = int(cfg.get("unfreeze_last_layers") or 0)
    candidate_layers = [
        layer for layer in model.layers
        if layer.name not in head_names and "input" not in layer.__class__.__name__.lower()
    ]
    tail_names = {layer.name for layer in candidate_layers[-tail_count:]} if tail_count > 0 else set()

    trainable_count = 0
    for layer in model.layers:
        is_bn = "batchnormalization" in layer.__class__.__name__.lower()
        should_train = layer.name in head_names or layer.name in tail_names
        if freeze_batchnorm and is_bn:
            should_train = False
        layer.trainable = should_train
        if should_train:
            trainable_count += 1

    return {
        "strategy": strategy,
        "trainable_layers": trainable_count,
        "tail_layers": len(tail_names),
    }


def _compute_fomo_class_weights(
    cell_counts: np.ndarray,
    label_names: List[str],
    fomo_version: int = 1,
    warmup_per_class_metrics: Optional[Dict[str, Dict[str, float]]] = None,
    warmup_per_class_history: Optional[Dict[str, List[Dict[str, float]]]] = None,
) -> Optional[np.ndarray]:
    """
    Compute FOMO per-class object weights.

    v1 keeps the legacy inverse-frequency weighting exactly.
    v2 can additionally boost underlearned classes based on warmup validation
    metrics so hard classes receive extra pressure even when raw counts are
    nearly balanced.
    """
    counts = np.asarray(cell_counts, dtype=np.float32)
    if counts.size == 0:
        return None

    nonzero_mask = counts > 0
    if not np.any(nonzero_mask):
        return None

    safe_counts = np.where(nonzero_mask, counts, np.max(counts[nonzero_mask]))
    raw_w = 1.0 / safe_counts
    weights = (raw_w / np.min(raw_w)).astype(np.float32)

    if fomo_version == 2 and (warmup_per_class_metrics or warmup_per_class_history):
        difficulty_factors = np.ones_like(weights, dtype=np.float32)
        avg_recall_by_class = np.zeros_like(weights, dtype=np.float32)
        recent_avg_recall_by_class = np.zeros_like(weights, dtype=np.float32)
        recent_zero_tp_by_class = np.zeros_like(weights, dtype=np.float32)
        recent_lagging_epochs_by_class = np.zeros_like(weights, dtype=np.float32)
        for i, name in enumerate(label_names):
            stats = warmup_per_class_metrics.get(name, {}) or {}
            hist = (warmup_per_class_history or {}).get(name, [])
            recall = float(stats.get("recall", 0.0) or 0.0)
            f1 = float(stats.get("f1", stats.get("f1-score", 0.0)) or 0.0)
            tp = int(stats.get("tp", 0) or 0)
            fn = int(stats.get("fn", 0) or 0)
            support = tp + fn
            avg_recall = (
                float(sum(float(h.get("recall", 0.0) or 0.0) for h in hist) / len(hist))
                if hist else recall
            )
            avg_recall_by_class[i] = avg_recall
            zero_tp_epochs = sum(1 for h in hist if int(h.get("tp", 0) or 0) == 0 and int(h.get("fn", 0) or 0) > 0)
            lagging_epochs = sum(
                1 for h in hist
                if float(h.get("recall", 0.0) or 0.0) < 0.05
            )
            recent_hist = hist[-4:] if hist else []
            recent_avg_recall = (
                float(sum(float(h.get("recall", 0.0) or 0.0) for h in recent_hist) / len(recent_hist))
                if recent_hist else avg_recall
            )
            recent_avg_recall_by_class[i] = recent_avg_recall
            recent_zero_tp = sum(
                1 for h in recent_hist
                if int(h.get("tp", 0) or 0) == 0 and int(h.get("fn", 0) or 0) > 0
            )
            recent_lagging_epochs = sum(
                1 for h in recent_hist
                if float(h.get("recall", 0.0) or 0.0) < 0.05
                and int(h.get("fn", 0) or 0) > 0
            )
            recent_zero_tp_by_class[i] = float(recent_zero_tp)
            recent_lagging_epochs_by_class[i] = float(recent_lagging_epochs)

            factor = 1.0
            factor += max(0.0, 0.80 - recall) * 2.8
            factor += max(0.0, 0.65 - f1) * 1.8
            factor += max(0.0, 0.70 - avg_recall) * 1.8
            factor += max(0.0, 0.60 - recent_avg_recall) * 2.8
            if support > 0 and recall < 0.08:
                factor += 1.5
            if support > 0 and recent_avg_recall < 0.08:
                factor += 1.2
            if support > 0 and tp == 0:
                factor = max(factor, 3.6)
            if hist:
                factor += min(1.35, 0.24 * zero_tp_epochs)
                factor += min(0.9, 0.14 * lagging_epochs)
            if recent_hist:
                factor += min(1.2, 0.30 * recent_zero_tp)
            if support > 0 and recent_avg_recall < 0.10:
                factor += 0.9
            if support > 0 and recent_avg_recall < 0.05:
                factor += 0.9
            difficulty_factors[i] = min(factor, 6.5)

        best_avg_recall = float(np.max(avg_recall_by_class[nonzero_mask])) if np.any(nonzero_mask) else 0.0
        if best_avg_recall > 0.0:
            recall_gap = np.maximum(0.0, best_avg_recall - avg_recall_by_class)
            difficulty_factors += np.clip(recall_gap * 10.0, 0.0, 2.5)

        # Make the easiest active class the baseline again so we selectively
        # boost the harder class instead of inflating all classes together.
        active_factors = difficulty_factors[nonzero_mask]
        if active_factors.size > 0:
            difficulty_factors = difficulty_factors / max(1.0, float(np.min(active_factors)))

        weights = weights * difficulty_factors
        if best_avg_recall > 0.0:
            persistent_lag_gap = np.maximum(0.0, best_avg_recall - recent_avg_recall_by_class)
            persistent_lag_multiplier = np.ones_like(weights, dtype=np.float32)
            lag_mask = np.logical_and(
                nonzero_mask,
                np.logical_and(
                    recent_avg_recall_by_class < 0.18,
                    persistent_lag_gap > 0.02,
                ),
            )
            if np.any(lag_mask):
                lag_indices = np.where(lag_mask)[0]
                gap_boost = np.clip(persistent_lag_gap[lag_indices] * 16.0, 0.0, 1.6)
                low_recall_boost = np.clip(
                    (0.18 - recent_avg_recall_by_class[lag_indices]) * 6.0,
                    0.0,
                    0.95,
                )
                zero_tp_boost = np.clip(
                    recent_zero_tp_by_class[lag_indices] * 0.22,
                    0.0,
                    0.66,
                )
                recent_lag_boost = np.clip(
                    recent_lagging_epochs_by_class[lag_indices] * 0.16,
                    0.0,
                    0.48,
                )
                lag_multiplier = 1.0 + (
                    gap_boost
                    + low_recall_boost
                    + zero_tp_boost
                    + recent_lag_boost
                )
                severe_mask = recent_avg_recall_by_class[lag_indices] < 0.08
                if np.any(severe_mask):
                    lag_multiplier[severe_mask] = np.maximum(
                        lag_multiplier[severe_mask],
                        2.4,
                    )
                persistent_lag_multiplier[lag_indices] = np.minimum(lag_multiplier, 3.2)
            weights = weights * persistent_lag_multiplier

    if fomo_version == 2:
        weights = np.clip(weights, 1.0, 8.0).astype(np.float32)
    else:
        cap = np.where(weights > 4.0, 5.0, 3.0).astype(np.float32)
        weights = np.minimum(weights, cap).astype(np.float32)

    return weights


def _fomo_v2_needs_class_weight_refresh(
    per_class_history: Dict[str, List[Dict[str, float]]],
    n_recent: int = 4,
    lag_threshold: int = 3,
) -> bool:
    """
    Return True if any class has recall < 0.05 in at least *lag_threshold*
    of the last *n_recent* epochs AND had actual GT objects (fn > 0).

    Used to trigger a mid-fine-tune class weight re-computation for v2 when
    a class has been persistently failing to produce true positives despite
    having ground-truth coverage.  v1 never calls this.
    """
    for hist in per_class_history.values():
        if len(hist) < n_recent:
            continue
        recent = hist[-n_recent:]
        n_lagging = sum(
            1 for h in recent
            if float(h.get("recall", 0.0) or 0.0) < 0.05
            and int(h.get("fn", 0) or 0) > 0
        )
        if n_lagging >= lag_threshold:
            return True
    return False


def _fomo_v2_can_refresh_class_weights(
    *,
    fomo_version: int,
    cell_counts: Optional[np.ndarray],
    n_classes: int,
    y_train_ndim: int,
    per_class_history: Optional[Dict[str, List[Dict[str, float]]]],
    refresh_count: int,
    max_refreshes: int,
) -> bool:
    """
    Return True when the v2 lagging-class refresh path has the inputs it needs.

    The key guard is the concrete ``cell_counts is not None`` check; using
    ``"_cell_counts" in dir()`` is fragile and can silently disable refreshes.
    """
    return bool(
        fomo_version == 2
        and cell_counts is not None
        and n_classes >= 2
        and y_train_ndim == 4
        and refresh_count < max_refreshes
        and per_class_history
        and _fomo_v2_needs_class_weight_refresh(per_class_history)
    )


def _restore_fomo_v2_best_warmup_weights(
    model,
    fomo_f1_cb,
    warmup_epochs: int,
) -> bool:
    """
    Restore the best warmup checkpoint before starting v2 phase 2.

    Warmup still runs for its full scheduled duration, but fine-tuning should
    start from the strongest warmup weights rather than the final warmup epoch.
    """
    if model is None or fomo_f1_cb is None or getattr(fomo_f1_cb, "best_weights", None) is None:
        return False
    best_epoch = int(getattr(fomo_f1_cb, "best_epoch", -1))
    if best_epoch < 0 or best_epoch >= int(warmup_epochs):
        return False
    model.set_weights([w.copy() for w in fomo_f1_cb.best_weights])
    logger.info(
        "[FOMO v2] Restored best warmup checkpoint before phase 2: epoch %d (val_f1=%.4f).",
        best_epoch + 1,
        float(getattr(fomo_f1_cb, "best_f1", 0.0) or 0.0),
    )
    return True


from app.workers.cancel_utils import (
    CancelledError as _CancelledError,
    raise_if_cancelled as _raise_if_job_cancelled,
    raise_if_cancelled_fast as _raise_if_job_cancelled_fast,
    assert_not_cancelled_before_completing as _assert_not_cancelled_before_completing,
    clear_cancel_cache as _clear_cancel_cache,
    promote_run_to_active as _promote_run_to_active,
    pop_live_buffer as _pop_live_buffer,
    clear_live_buffer as _clear_live_buffer,
)


# ─── Negative (background-image) ratio cap ────────────────────────────────────

# Fraction of the assembled training set that may be background images.
# Stored in existing JSON (impulse.ml_blocks[0]["params"] or the job's
# extra_params) — never a new column — following the same read pattern as
# `fomo_version` and `architecture`.
NEGATIVE_RATIO_CAP_DEFAULT: float = 0.2
NEGATIVE_RATIO_CAP_MIN: float = 0.0
NEGATIVE_RATIO_CAP_MAX: float = 0.5


def _resolve_negative_ratio_cap(impulse, extra_params: Optional[dict] = None) -> float:
    """Resolve the maximum share of the training set that may be negatives.

    Priority mirrors `_resolve_training_architecture` / `_fomo_version`: the
    job's extra_params win (the UI can override without re-saving the impulse),
    then the impulse's ML-block params, then the default.  Clamped to
    [NEGATIVE_RATIO_CAP_MIN, NEGATIVE_RATIO_CAP_MAX]; an unparseable value falls
    back to the default rather than failing the job.
    """
    ml_params = {}
    try:
        ml_blocks = impulse.ml_blocks or []
        if ml_blocks:
            ml_params = ml_blocks[0].get("params") or {}
    except Exception:
        ml_params = {}

    raw = (extra_params or {}).get("negative_ratio_cap")
    if raw is None:
        raw = ml_params.get("negative_ratio_cap")
    if raw is None:
        return NEGATIVE_RATIO_CAP_DEFAULT
    try:
        value = float(raw)
    except (TypeError, ValueError):
        logger.warning(
            "[Training] negative_ratio_cap=%r is not a number — using default %.2f.",
            raw, NEGATIVE_RATIO_CAP_DEFAULT,
        )
        return NEGATIVE_RATIO_CAP_DEFAULT
    if not math.isfinite(value):
        logger.warning(
            "[Training] negative_ratio_cap=%r is not finite — using default %.2f.",
            raw, NEGATIVE_RATIO_CAP_DEFAULT,
        )
        return NEGATIVE_RATIO_CAP_DEFAULT
    clamped = float(min(max(value, NEGATIVE_RATIO_CAP_MIN), NEGATIVE_RATIO_CAP_MAX))
    if clamped != value:
        # Clamping (rather than raising) keeps a long-running job alive on a
        # bad UI value, but a silent clamp hides the fact that the run did not
        # use the ratio the user asked for — which is exactly the number they
        # will look at when reasoning about background false positives.
        logger.warning(
            "[Training] negative_ratio_cap=%.4f is outside the supported range "
            "[%.2f, %.2f] — clamped to %.2f for this run.",
            value, NEGATIVE_RATIO_CAP_MIN, NEGATIVE_RATIO_CAP_MAX, clamped,
        )
    return clamped


def _negative_keep_indices(
    negative_flags: List[bool],
    cap: float,
    seed: str,
) -> List[int]:
    """Return the indices of the entries to KEEP so that

        n_negatives / (n_negatives + n_positives)  <=  cap

    Positives are never dropped; only surplus negatives are.  Selection is
    seeded so the same impulse + same dataset always trims the same rows,
    which keeps re-runs comparable.  The seed goes through crc32, not the
    builtin ``hash()`` — string hashing is salted per interpreter process, so a
    hash-derived seed would pick a different subset on every worker restart.

    MUST be applied at the *end* of dataset assembly — class balancing
    oversamples minority positives and augmentation multiplies every kept
    sample, so both change the positive count the ratio is measured against.
    Order is preserved so callers can index parallel arrays with the result.
    """
    n = len(negative_flags)
    neg_idx = [i for i in range(n) if negative_flags[i]]
    pos_idx = [i for i in range(n) if not negative_flags[i]]
    if not neg_idx:
        return list(range(n))

    if cap <= 0.0:
        allowed = 0
    else:
        # neg / (neg + pos) <= cap  ⟺  neg <= cap * pos / (1 - cap)
        allowed = int(len(pos_idx) * cap / (1.0 - cap)) if cap < 1.0 else len(neg_idx)
    if len(neg_idx) <= allowed:
        return list(range(n))

    import zlib
    rng = np.random.default_rng(
        zlib.crc32(f"{seed}|{cap:.4f}|{n}".encode("utf-8"))
    )
    keep_neg = set(
        int(i) for i in rng.choice(neg_idx, size=allowed, replace=False)
    ) if allowed > 0 else set()
    return [i for i in range(n) if (not negative_flags[i]) or i in keep_neg]


def _apply_fomo_negative_cap(
    X_tr: list, y_tr: list, y_tr_eval: list,
    neg_flags: List[bool], cap: float, seed: str, prefix: str,
) -> Tuple[list, list, list]:
    """Trim surplus negatives from the assembled FOMO training arrays.

    Called after class balancing AND augmentation, because both change the
    positive count the ratio is measured against.
    """
    keep = _negative_keep_indices(neg_flags, cap, seed)
    if len(keep) != len(neg_flags):
        X_tr      = [X_tr[i] for i in keep]
        y_tr      = [y_tr[i] for i in keep]
        y_tr_eval = [y_tr_eval[i] for i in keep]
        neg_flags = [neg_flags[i] for i in keep]
    _log_negative_ratio(prefix, sum(1 for f in neg_flags if f), len(neg_flags), cap)
    return X_tr, y_tr, y_tr_eval


def _log_negative_ratio(prefix: str, n_neg: int, n_total: int, cap: float) -> None:
    ratio = (n_neg / n_total) if n_total else 0.0
    logger.info(
        "%s negatives: %d / %d training samples (%.1f%%), cap=%.0f%%",
        prefix, n_neg, n_total, ratio * 100.0, cap * 100.0,
    )


def _combine_dsp_features(feature_list):
    """
    Preserve image-like tensors when the pipeline contains a single image block.
    Mixed or multi-block pipelines fall back to flat concatenation.
    """
    if not feature_list:
        return np.empty((0,), dtype=np.float32)

    if len(feature_list) == 1:
        return np.asarray(feature_list[0], dtype=np.float32)

    return np.concatenate([np.asarray(feat, dtype=np.float32).flatten() for feat in feature_list])


# ─── Feature cache helpers ────────────────────────────────────────────────────

def _try_load_feature_cache(
    impulse: Impulse,
    current_sample_ids: List[str],
    is_fomo: bool,
    current_label_names: Optional[List[str]] = None,
    is_yolo_pro_model: bool = False,
) -> Optional[Dict]:
    """
    Attempt to load and validate the pre-generated feature cache for an impulse.

    Validation checks (in order):
      1. Cache file exists in S3
      2. meta_json present (v2 format — legacy v1 has no metadata)
      3. impulse_id matches
      4. dsp_config_hash matches (any block type or param change → stale)
      5. image_width / image_height match (image impulses only)
      5b. FOMO minimum grid size (96×96 → 12×12 grid) — FOMO path only
      5c. YOLO-Pro minimum input resolution (YOLO_PRO_MIN_INPUT_SIZE) —
          YOLO-Pro path only.  Low-res cached features cannot be silently
          upsampled; they must be recomputed at the correct resolution.
      6. Label set matches (label added/deleted since last Generate Features → stale)
      7. No new samples in the DB that are absent from the cache

    Returns a dict {X, y, ids, sample_types, boxes_json, meta} on success,
    or None if the cache is missing, a legacy format, or stale.
    """
    dsp_blocks    = impulse.dsp_blocks or [{"type": "raw", "params": {}}]
    expected_hash = compute_dsp_config_hash(dsp_blocks)
    cache_key     = make_features_storage_key(impulse.project_id, impulse.id)

    # ── Fetch ─────────────────────────────────────────────────────────────────
    try:
        data_bytes = storage.download_bytes(cache_key)
    except Exception:
        logger.info(
            "[Training] Feature cache: no cached file found — "
            "will recompute features from raw samples."
        )
        return None

    # ── Parse ─────────────────────────────────────────────────────────────────
    try:
        with np.load(io.BytesIO(data_bytes), allow_pickle=True) as npz:

            # Presence of meta_json distinguishes v2 cache from the old format.
            if "meta_json" not in npz:
                logger.info(
                    "[Training] Feature cache: legacy format (no metadata) — "
                    "will recompute. Run Generate Features to produce a v2 cache."
                )
                return None

            meta = json.loads(str(npz["meta_json"][0]))

            # ── Check 0: cache format version ────────────────────────────────
            # A cache written by an older DSP worker is not merely missing a
            # field — v3 and below excluded background ("negative") images
            # entirely, so training a detection model on one would silently drop
            # every negative the user added. Reject rather than degrade.
            _cache_version = int(meta.get("version", 1) or 1)
            if _cache_version < FEATURE_CACHE_VERSION:
                logger.info(
                    "[Training] Feature cache stale: format v%d predates v%d "
                    "(background images were not cached before v4) — recomputing. "
                    "Run Generate Features to rebuild.",
                    _cache_version, FEATURE_CACHE_VERSION,
                )
                return None

            # ── Check 1: impulse identity ─────────────────────────────────────
            if meta.get("impulse_id") != impulse.id:
                logger.info("[Training] Feature cache stale: impulse_id mismatch — recomputing.")
                return None

            # ── Check 2: DSP config ───────────────────────────────────────────
            if meta.get("dsp_config_hash") != expected_hash:
                logger.info(
                    f"[Training] Feature cache stale: DSP config changed "
                    f"(cached={meta.get('dsp_config_hash','?')[:8]}… "
                    f"current={expected_hash[:8]}…) — recomputing."
                )
                return None

            # ── Check 3: image dimensions (image impulses only) ───────────────
            # Compare against effective dimensions — merge_image_params applies
            # the FOMO 96×96 floor so the "expected" size is always what the DSP
            # worker would produce today, not what the raw impulse DB field says.
            _img_block: dict = {"params": {}}
            for blk in dsp_blocks:
                if blk.get("type") == "image":
                    _img_block = blk
                    break
            merged_img_params = merge_image_params(impulse, _img_block)
            if merged_img_params:
                w = int(merged_img_params.get("image_width")  or impulse.image_width  or 96)
                h = int(merged_img_params.get("image_height") or impulse.image_height or 96)
                if meta.get("image_width") != w or meta.get("image_height") != h:
                    logger.info(
                        f"[Training] Feature cache stale: image dimensions changed "
                        f"(cached={meta.get('image_width')}×{meta.get('image_height')} "
                        f"current={w}×{h}) — recomputing."
                    )
                    return None

            if not meta.get("boxes_normalized", False):
                logger.info(
                    "[Training] Feature cache stale: cached boxes are not normalized "
                    "for YOLO/FOMO training — recomputing from raw samples."
                )
                return None

            # ── Check 3b: FOMO minimum grid size ─────────────────────────────
            # In Edge Impulse, FOMO requires a minimum 12×12 grid (96×96 input).
            # If the cached features produce a smaller grid the cache is treated
            # as stale so Generate Features re-runs at the correct resolution.
            # Uses _uses_fomo_min_resolution to match both "fomo_mobilenetv2_0_1"
            # and "object_detection" + params.model="fomo" block types.
            if is_fomo or _uses_fomo_min_resolution(impulse):
                cached_w = meta.get("image_width", 0) or 0
                cached_h = meta.get("image_height", 0) or 0
                cached_grid_w = cached_w // 16
                cached_grid_h = cached_h // 16
                if cached_grid_w < 6 or cached_grid_h < 6:
                    logger.info(
                        f"[Training] Feature cache stale: FOMO cache at {cached_w}×{cached_h} "
                        f"produces a {cached_grid_w}×{cached_grid_h} grid "
                        f"(minimum required: 6×6 = 96×96 input at stride-16). "
                        f"Re-running Generate Features at the recommended resolution."
                    )
                    return None

            # ── Check 3c: YOLO-Pro minimum input resolution ───────────────────
            # YOLO-Pro requires features at or above YOLO_PRO_MIN_INPUT_SIZE.
            # Cached features below this size are incompatible: upsampling a
            # low-resolution cached tensor does NOT produce valid high-resolution
            # training data and must NOT be used silently.  Force recomputation.
            if is_yolo_pro_model:
                cached_w = meta.get("image_width", 0) or 0
                cached_h = meta.get("image_height", 0) or 0
                if cached_w < YOLO_PRO_MIN_INPUT_SIZE or cached_h < YOLO_PRO_MIN_INPUT_SIZE:
                    logger.warning(
                        f"[Training] [ACTION:CACHE_REJECTED_LOW_RES] "
                        f"YOLO-Pro cache at {cached_w}×{cached_h} is below the "
                        f"minimum input resolution "
                        f"({YOLO_PRO_MIN_INPUT_SIZE}×{YOLO_PRO_MIN_INPUT_SIZE}). "
                        "Low-resolution cached features are incompatible — upsampling "
                        "them would NOT produce valid high-resolution training data. "
                        "Recomputing features from raw samples. "
                        "To avoid this: set DSP image block to "
                        f">= {YOLO_PRO_MIN_INPUT_SIZE}×{YOLO_PRO_MIN_INPUT_SIZE} "
                        "and re-run Generate Features."
                    )
                    return None

            # ── Check 4: label set ───────────────────────────────────────────
            # If a label was added or deleted since the last Generate Features
            # run the number/identity of output classes has changed → the cached
            # X features are still valid but the cache metadata is stale, so
            # training must recompute to pick up the correct class ordering.
            if current_label_names is not None:
                cached_lbl = sorted(meta.get("label_names", []))
                current_lbl = sorted(current_label_names)
                if cached_lbl != current_lbl:
                    logger.info(
                        f"[Training] Feature cache stale: label set changed "
                        f"(cached={cached_lbl}, current={current_lbl}) — recomputing. "
                        "Run Generate Features again after adding/deleting labels."
                    )
                    return None

            # ── Check 4b: train_subset_percent ───────────────────────────────
            # If the impulse's subset percent has changed since the cache was
            # built, the cache contains the *wrong* sample mix. Force a recompute.
            cached_subset = float(meta.get("train_subset_percent", 100.0) or 100.0)
            current_subset = float(getattr(impulse, "train_subset_percent", 100.0) or 100.0)
            if abs(cached_subset - current_subset) > 1e-6:
                logger.info(
                    f"[Training] Feature cache stale: train_subset_percent changed "
                    f"(cached={cached_subset:.1f}, current={current_subset:.1f}) — recomputing."
                )
                return None

            # ── Check 5: sample coverage ──────────────────────────────────────
            # When the user has opted in to a < 100% subset, the cache is
            # intentionally smaller than the current sample set, so skip the
            # "missing samples" check entirely — the DSP worker's RNG with the
            # same impulse_id+percent seed guarantees the chosen samples are
            # stable across runs.
            cached_ids  = set(str(s) for s in npz["ids"].tolist())
            current_set = set(current_sample_ids)
            if current_subset >= 100.0:
                missing = current_set - cached_ids
                if missing:
                    logger.info(
                        f"[Training] Feature cache stale: {len(missing)} new sample(s) uploaded "
                        "since last Generate Features — recomputing to include all data."
                    )
                    return None

                extra = cached_ids - current_set
                if extra:
                    logger.warning(
                        f"[Training] Feature cache: {len(extra)} cached sample(s) no longer in DB "
                        "(deleted since last Generate Features run). Continuing with cache."
                    )

            # ── All checks passed ─────────────────────────────────────────────
            cached = {
                "X":            npz["X"],
                "y":            npz["y"],       # label_ids (strings)
                "ids":          npz["ids"],
                "sample_types": npz.get("sample_types", np.array([], dtype=object)),
                "boxes_json":   npz.get("boxes_json",   np.array([], dtype=object)),
                # Per-sample negative flag (v4+). Never inferred from an empty
                # label + empty box list — see the dsp_worker comment.
                "background":   npz.get("background",   np.zeros(0, dtype=np.int8)),
                "meta":         meta,
            }
            logger.info(
                f"[Training] Using cached generated features — "
                f"{len(cached['X'])} samples "
                f"({int(meta.get('background_count', 0))} background), "
                f"dsp_hash={expected_hash[:8]}…, "
                f"cached_at={meta.get('created_at', '?')}"
            )
            return cached

    except Exception as exc:
        logger.warning(
            f"[Training] Feature cache: error reading cache ({exc}) — "
            "will recompute from raw samples."
        )
        return None


def _build_arrays_from_cache(
    cached: Dict,
    impulse: Impulse,
    label_map: dict,
    label_names: List[str],
    is_fomo: bool,
    img_w: int,
    img_h: int,
    balance_classes: bool = True,
    fomo_version: int = 1,
    negative_ratio_cap: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """
    Reconstruct train/test split arrays from a validated feature cache.

    Classification path
    -------------------
    • X comes directly from cache (pre-computed DSP tensors).
    • y (class indices) is rebuilt by mapping cached label_ids through label_map.

    FOMO/detection path
    -------------------
    • X comes directly from cache.
    • y (heatmaps) is rebuilt from cached boxes_json so label-map changes and
      new annotations are always reflected without re-running DSP.
    """
    X_all          = cached["X"]
    y_raw          = cached["y"]            # label_ids (strings) for classification
    sample_types_c = cached["sample_types"]
    boxes_json_arr = cached["boxes_json"]
    background_arr = cached.get("background", np.zeros(0, dtype=np.int8))
    n              = len(X_all)

    def _is_background_i(i: int) -> bool:
        return i < len(background_arr) and bool(background_arr[i])

    def _is_test(st: str) -> bool:
        return st.lower() in ("testing", "test", "sampletype.testing")

    if is_fomo:
        # ── Phase 1: collect (features, boxes) pairs from the cache ──────────
        fomo_train: list = []
        fomo_test:  list = []
        _n_background = 0

        for i in range(n):
            boxes_str = str(boxes_json_arr[i]) if i < len(boxes_json_arr) else "[]"
            boxes     = json.loads(boxes_str)
            features  = np.asarray(X_all[i], dtype=np.float32)

            # Resolve label fields: boxes in the cache may have been stored with
            # label_id=null (JSON null → Python None) or with only a "label"
            # name-string field.  Apply the same fallback chain that
            # normalize_bounding_boxes uses so the dual-key label_map lookup
            # in _create_fomo_heatmap succeeds for both formats.
            resolved_boxes = []
            for b in boxes:
                if not isinstance(b, dict):
                    continue
                resolved_label = (
                    b.get("label_id")
                    or b.get("label")
                    or b.get("labelId")
                    or b.get("class_name")
                ) or None
                if resolved_label is None:
                    continue  # no resolvable label — skip to avoid None-key spam
                resolved_boxes.append({**b, "label_id": resolved_label})
            boxes = resolved_boxes

            # NOTE: no synthetic-box fallback here.  A labeled but box-less
            # sample used to get a fabricated centre-50% box invented for it,
            # while the raw-sample path (_load_dataset) simply skipped it.  That
            # made a cache hit and a cache miss produce *different* datasets from
            # the same DB rows — the model learned a phantom centre object on
            # every classification-labeled image that had no annotation.
            # Both paths now agree: no boxes → no training signal → skip …
            # …unless the sample is an explicitly marked background image, which
            # trains as a negative: all-background GT, no object cells.
            _bg = _is_background_i(i)
            if not boxes and not _bg:
                continue  # no annotation → skip (same rule as raw-sample path)
            if _bg:
                boxes = []          # a negative never contributes GT, ever
                _n_background += 1

            stype = str(sample_types_c[i]) if i < len(sample_types_c) else "training"
            entry = (features, boxes)
            (fomo_test if _is_test(stype) else fomo_train).append(entry)

        all_fomo = fomo_train + fomo_test
        if not all_fomo:
            return np.empty((0,)), np.array([]), np.array([]), np.empty((0,)), np.array([]), label_names

        # ── Phase 2: derive grid from actual feature shape ────────────────────
        feat_shape = np.asarray(all_fomo[0][0], dtype=np.float32).shape

        def _snap8(x: int) -> int:
            return x if x % 8 == 0 else x + (8 - x % 8)

        # v1: stride-16 (block_13); v2: stride-8 (block_6).
        # Guard: v2 logic is only active when fomo_version == 2.
        _grid_divisor = 8 if fomo_version == 2 else 16

        if len(feat_shape) >= 2:
            grid_h = _snap8(feat_shape[0]) // _grid_divisor
            grid_w = _snap8(feat_shape[1]) // _grid_divisor
        else:
            grid_h = _snap8(img_h) // _grid_divisor
            grid_w = _snap8(img_w) // _grid_divisor

        logger.info(
            f"[Training] (cache) FOMO v{fomo_version} grid: {grid_h}×{grid_w} "
            f"(stride-{_grid_divisor}, feature shape {feat_shape}, img={img_w}×{img_h})"
        )
        if grid_w < 6 or grid_h < 6:
            _hint = (
                f"Set image size to at least 48×48 and re-run Generate Features. "
                f"A 48×48 input gives a 6×6 grid at stride-8."
                if fomo_version == 2 else
                f"Set image width and height to 96×96 (or larger) in your DSP settings "
                f"and re-run Generate Features. A 96×96 input gives a 6×6 grid (36 cells) at stride-16."
            )
            logger.warning(
                f"[FOMO v{fomo_version}] Grid {grid_h}×{grid_w} is below the recommended minimum of 6×6. "
                f"Current image size {img_w}×{img_h} gives only {grid_w * grid_h} detection cells. "
                f"Action: {_hint}"
            )

        # ── Phase 3: class balance (training split only) ─────────────────────
        if balance_classes:
            fomo_train = _fomo_balance_classes(fomo_train, label_map, label_names)

        # ── Phase 4: build heatmap targets ────────────────────────────────────
        # v1 keeps strict center-cell targets.
        # v2 keeps strict evaluation targets but now trains with a conservative
        # soft center falloff to improve exact-cell ranking on sparse grids.
        _target_cfg = _fomo_training_target_config(fomo_version)
        _use_spread = bool(_target_cfg["spread"])
        logger.info(
            "[FOMO v%d] (cache) target mode=%s  spread=%s  grid=%d×%d  img=%d×%d  reason=%s",
            fomo_version, _target_cfg["mode"], _use_spread, grid_h, grid_w, img_w, img_h,
            _target_cfg["reason"],
        )
        X_tr, y_tr, y_tr_eval, X_te, y_te = [], [], [], [], []
        _neg_flags: List[bool] = []
        _aug_stats_total: dict = {"flip": 0, "brightness": 0, "contrast": 0, "rotation": 0}
        for feats, boxes in fomo_train:
            _is_neg = not boxes
            tgt = _create_fomo_heatmap(
                boxes, grid_w, grid_h, img_w, img_h, len(label_names), label_map,
                spread=_use_spread,
                target_mode=str(_target_cfg["mode"]),
                center_value=float(_target_cfg.get("center_value", 1.0)),
                neighbor_value=float(_target_cfg.get("neighbor_value", 0.0)),
                diagonal_value=float(_target_cfg.get("diagonal_value", 0.0)),
            )
            eval_tgt = _create_fomo_heatmap(
                boxes, grid_w, grid_h, img_w, img_h, len(label_names), label_map,
                spread=False,
            )
            X_tr.append(feats);  y_tr.append(tgt);  y_tr_eval.append(eval_tgt)
            _neg_flags.append(_is_neg)
            aug_list, aug_stats = _augment_fomo_sample(feats, boxes, img_w, img_h, fomo_version=fomo_version)
            for k in _aug_stats_total:
                _aug_stats_total[k] += aug_stats.get(k, 0)
            for aug_feat, aug_boxes in aug_list:
                aug_tgt = _create_fomo_heatmap(
                    aug_boxes, grid_w, grid_h, img_w, img_h, len(label_names), label_map,
                    spread=_use_spread,
                    target_mode=str(_target_cfg["mode"]),
                    center_value=float(_target_cfg.get("center_value", 1.0)),
                    neighbor_value=float(_target_cfg.get("neighbor_value", 0.0)),
                    diagonal_value=float(_target_cfg.get("diagonal_value", 0.0)),
                )
                aug_eval_tgt = _create_fomo_heatmap(
                    aug_boxes, grid_w, grid_h, img_w, img_h, len(label_names), label_map,
                    spread=False,
                )
                X_tr.append(aug_feat);  y_tr.append(aug_tgt);  y_tr_eval.append(aug_eval_tgt)
                _neg_flags.append(_is_neg)

        # Cap the negative share LAST — balancing and augmentation above both
        # move the positive count this ratio is measured against.
        _cap = (
            negative_ratio_cap if negative_ratio_cap is not None
            else _resolve_negative_ratio_cap(impulse)
        )
        X_tr, y_tr, y_tr_eval = _apply_fomo_negative_cap(
            X_tr, y_tr, y_tr_eval, _neg_flags, _cap,
            seed=f"{getattr(impulse, 'id', '?')}|cache", prefix="[FOMO] (cache)",
        )

        for feats, boxes in fomo_test:
            tgt = _create_fomo_heatmap(
                boxes, grid_w, grid_h, img_w, img_h, len(label_names), label_map,
                spread=False,
            )
            X_te.append(feats);  y_te.append(tgt)

    else:
        # ── Classification: label_id → class index ────────────────────────────
        X_tr, y_tr, y_tr_eval, X_te, y_te = [], [], [], [], []
        for i in range(n):
            lid = str(y_raw[i]) if i < len(y_raw) else ""
            if lid not in label_map:
                continue  # unknown / empty label → skip
            cls  = label_map[lid]
            feat = np.asarray(X_all[i], dtype=np.float32)
            stype = str(sample_types_c[i]) if i < len(sample_types_c) else "training"
            if _is_test(stype):
                X_te.append(feat);  y_te.append(cls)
            else:
                X_tr.append(feat);  y_tr.append(cls);  y_tr_eval.append(cls)

    X_train = np.array(X_tr, dtype=np.float32) if X_tr else np.empty((0,))
    y_train = np.array(y_tr)
    y_train_eval = np.array(y_tr_eval)
    X_test  = np.array(X_te, dtype=np.float32) if X_te else np.empty((0,))
    y_test  = np.array(y_te)

    if is_fomo:
        _aug_total = len(X_tr) - len(fomo_train)   # augmented copies kept
        logger.info(
            f"[Training] (cache) FOMO dataset: {len(fomo_train)} train samples "
            f"({_n_background} background/negative) "
            f"→ {len(X_tr)} after augmentation and negative cap "
            f"({_aug_total} copies kept, 0 discarded — "
            f"flip={_aug_stats_total['flip']}, brightness={_aug_stats_total['brightness']}, "
            f"contrast={_aug_stats_total['contrast']}, rotation={_aug_stats_total['rotation']}), "
            f"{len(fomo_test)} test samples."
        )
    else:
        logger.info(
            f"[Training] (cache) split: "
            f"train={len(X_train)}, test={len(X_test)}, "
            f"labels={label_names}"
        )
    return X_train, y_train, y_train_eval, X_test, y_test, label_names


# ─── Terminal-state persistence helpers ──────────────────────────────────────
#
# When a run ends in failed/cancelled, the log viewer must show every line the
# training loop produced PLUS a synthesized terminal line.  The historical bug
# was that the terminal handlers called db.rollback() first, discarding any
# log_lines / epoch_metrics that had been appended to the in-memory buffers
# since the last incremental commit.  These two helpers implement the required
# flow without dropping the buffered lines:
#
#     flush in-memory buffers → append terminal line(s) → set status → commit
#
# _flush_training_buffers persists the buffers (re-fetching the job after the
# failed transaction is cleared).  _append_terminal_log_and_status then appends
# the terminal line(s) and writes the terminal status.  Both are resilient to a
# dirty/aborted session and never silently drop the buffered lines on the floor.


def _flush_training_buffers(db, job_id: str, log_lines, epoch_metrics) -> None:
    """
    Merge the in-memory log/epoch buffers into training_history and commit.

    Safe to call from a finally/except block: it clears any failed transaction,
    re-fetches the job row, merges the buffers into the existing
    training_history (preserving every other field), and commits.  Empty
    buffers are left untouched so this is a no-op for architectures that own no
    buffer of that kind.  Never raises.
    """
    try:
        try:
            db.rollback()
        except Exception:
            pass
        job = db.query(TrainingJob).filter(TrainingJob.id == job_id).first()
        if job is None:
            return
        th = dict(job.training_history or {})
        if log_lines:
            th["log_lines"] = list(log_lines)
        if epoch_metrics:
            th["epoch_metrics"] = list(epoch_metrics)
        job.training_history = _sanitize_json(th)
        db.commit()
    except Exception:
        try:
            db.rollback()
        except Exception:
            pass
        logger.warning(f"[{job_id}] Failed to flush training buffers", exc_info=True)


def _resolve_terminal_buffers(job_id, fomo_log_lines, fomo_epoch_metrics, clf_epoch_metrics):
    """
    Return the (log_lines, epoch_metrics) buffers to flush for a terminal run.

    The inline FOMO/classifier loop keeps its buffers in run_training_job's own
    scope.  The YOLO-Pro / SSD loops register theirs in the live-buffer registry
    (keyed by job_id).  Prefer the registered buffers when present (the run took
    a detection branch and returned early, so the inline buffers are empty);
    otherwise fall back to the inline buffers.
    """
    reg = _pop_live_buffer(job_id)
    if reg is not None:
        return (reg.get("log_lines") or []), (reg.get("epoch_metrics") or [])
    # Inline FOMO uses fomo_epoch_metrics; the classifier path uses
    # clf_epoch_metrics (and owns no log_lines buffer).  Exactly one is populated.
    return fomo_log_lines, (fomo_epoch_metrics or clf_epoch_metrics)


def _terminal_reason_line(message: str) -> str:
    """Collapse an exception message to a single, user-presentable line."""
    parts = (message or "").strip().splitlines()
    first = parts[0].strip() if parts else ""
    if not first:
        first = "Unknown error"
    return first[:500]


def _append_terminal_log_and_status(
    db,
    job_id: str,
    *,
    status,
    lines,
    error_message: Optional[str] = None,
    skip_if_status=None,
) -> None:
    """
    Append the terminal outcome line(s) to training_history.log_lines and write
    the terminal status — after the per-architecture buffers have already been
    flushed.  Re-fetches the (possibly stale) job row so the buffered lines are
    preserved rather than overwritten.

    skip_if_status guards the failure→cancelled race: if the row is already in
    that status the write is skipped, so a cancellation is never relabelled
    "failed" (and vice-versa).
    """
    try:
        db.rollback()
    except Exception:
        pass
    job = db.query(TrainingJob).filter(TrainingJob.id == job_id).first()
    if job is None:
        return
    if skip_if_status is not None and job.status == skip_if_status:
        return
    th = dict(job.training_history or {})
    log_lines = list(th.get("log_lines") or [])
    log_lines.extend(lines)
    th["log_lines"] = log_lines
    th["log_lines_complete"] = True
    job.training_history = _sanitize_json(th)
    job.status = status
    if error_message is not None:
        job.error_message = error_message
    job.completed_at = datetime.utcnow()
    db.commit()


@celery_app.task(bind=True, name="app.workers.training_worker.run_training_job")
def run_training_job(self, job_id: str):
    """Main Celery task — trains a model for the given training job ID."""
    db = SessionLocal()
    # Drop any cached cancel state left over from a prior run with the
    # same job_id (shouldn't happen in practice, but defensive).
    _clear_cancel_cache(job_id)
    _clear_live_buffer(job_id)
    job = None
    # In-memory buffers owned by the inline FOMO/classifier training loop.
    # Pre-declared here so the terminal except handlers can flush them even when
    # an exception fires before the loop assigns them, or when the run took the
    # YOLO-Pro / SSD branch (which owns its own buffers and returns early).
    _fomo_log_lines: list = []
    _fomo_epoch_metrics: list = []
    _clf_epoch_metrics: list = []
    try:
        job = db.query(TrainingJob).filter(TrainingJob.id == job_id).first()
        if not job:
            raise ValueError(f"Training job {job_id} not found")
        if job.status == JobStatus.cancelled:
            logger.info(f"[{job_id}] Training job was cancelled before start")
            return {"job_id": job_id, "status": "cancelled"}

        device_pref = job.device_type or "cpu"
        assert_job_device(device_pref, job_id)
        strategy = get_strategy()
        logger.info(
            # visible_gpus must come from get_visible_devices, not
            # list_physical_devices: the latter reports host hardware and would
            # log GPUs on a CPU worker that has already hidden them.
            "[%s] requested=%s worker_device=%s visible_gpus=%s host_gpus=%d queue=training_%s",
            job_id, device_pref, get_worker_device(),
            [g.name for g in tf.config.get_visible_devices("GPU")],
            len(tf.config.list_physical_devices("GPU")),
            device_pref,
        )

        job.status = JobStatus.running
        job.started_at = datetime.utcnow()
        db.commit()

        impulse = db.query(Impulse).filter(Impulse.id == job.impulse_id).first()

        # ── 1. Load and preprocess dataset ──────────────────────────────────
        logger.info(f"[{job_id}] Loading dataset for project {impulse.project_id}...")
        
        job_params = job.extra_params or {}
        ml_config  = impulse.ml_blocks[0] if impulse.ml_blocks else {}
        selected_architecture = (
            job_params.get("architecture") or
            ml_config.get("type") or
            ml_config.get("architecture", "dense")
        )
        # FOMO version: prefer extra_params (set by the UI at job-start time) so
        # the frontend can override without re-saving the impulse.  Falls back to
        # impulse ml_block params, then to 1 (v1 default — unchanged behaviour).
        _fomo_version = int(
            job_params.get("fomo_version")
            or (ml_config.get("params") or {}).get("fomo_version", 1)
            or 1
        )

        # ── Structured training metrics (training.log) ──────────────────────
        # Phase durations accumulate here and are emitted on each terminal path.
        _phase_ms: dict = {}
        _log_event(
            _training_log, "training.start",
            job_id=job_id, architecture=selected_architecture, device=device_pref,
        )

        # ── YOLO-Pro: dedicated detection pipeline ──────────────────────────
        if is_yolo_pro(selected_architecture):
            _raise_if_job_cancelled(job_id)
            _ds_start = time.perf_counter()
            (
                X_train, boxes_train, X_test, boxes_test,
                label_names, label_map, input_shape,
            ) = _load_yolo_dataset(db, impulse, extra_params=job_params)
            _phase_ms["dataset_load_ms"] = round((time.perf_counter() - _ds_start) * 1000.0, 2)

            _raise_if_job_cancelled(job_id)
            _train_start = time.perf_counter()
            _yolo_result = run_yolo_pro_training(
                job=job,
                impulse=impulse,
                X_train=X_train,
                boxes_train=boxes_train,
                X_test=X_test,
                boxes_test=boxes_test,
                label_names=label_names,
                label_map=label_map,
                input_shape=input_shape,
                storage=storage,
                db=db,
                job_id=job_id,
            )
            _log_event(
                _training_log, "training.finished",
                job_id=job_id, architecture=selected_architecture,
                status=(_yolo_result or {}).get("status", "completed"),
                train_export_ms=round((time.perf_counter() - _train_start) * 1000.0, 2),
                **_phase_ms,
            )
            return _yolo_result
        # ── End YOLO-Pro branch ──────────────────────────────────────────────

        # ── MobileNetV2 SSD: dedicated detection pipeline ─────────────────────
        if is_mobilenetv2_ssd(selected_architecture):
            _raise_if_job_cancelled(job_id)
            _ds_start = time.perf_counter()
            (
                X_train, boxes_train, X_test, boxes_test,
                label_names, label_map, input_shape,
            ) = _load_ssd_dataset(db, impulse, extra_params=job_params)
            _phase_ms["dataset_load_ms"] = round((time.perf_counter() - _ds_start) * 1000.0, 2)

            _raise_if_job_cancelled(job_id)
            _train_start = time.perf_counter()
            _ssd_result = run_mobilenetv2_ssd_training(
                job=job, impulse=impulse,
                X_train=X_train, boxes_train=boxes_train,
                X_test=X_test,   boxes_test=boxes_test,
                label_names=label_names, label_map=label_map,
                input_shape=input_shape, storage=storage,
                db=db, job_id=job_id,
            )
            _log_event(
                _training_log, "training.finished",
                job_id=job_id, architecture=selected_architecture,
                status=(_ssd_result or {}).get("status", "completed"),
                train_export_ms=round((time.perf_counter() - _train_start) * 1000.0, 2),
                **_phase_ms,
            )
            return _ssd_result
        # ── End MobileNetV2 SSD branch ─────────────────────────────────────────

        # Check before expensive dataset load
        _raise_if_job_cancelled(job_id)

        _ds_start = time.perf_counter()
        X_train, y_train, y_train_eval, X_test, y_test, label_names = _load_dataset(
            db, impulse, selected_architecture, fomo_version=_fomo_version,
            extra_params=job_params,
        )
        _phase_ms["dataset_load_ms"] = round((time.perf_counter() - _ds_start) * 1000.0, 2)

        # Check after dataset load, before model build
        _raise_if_job_cancelled(job_id)

        if len(X_train) == 0:
            raise ValueError("No training samples found. Please label and add samples first.")

        n_classes = len(label_names)
        input_shape = X_train[0].shape
        logger.info(f"[{job_id}] Dataset: {len(X_train)} train, {len(X_test)} test, {n_classes} classes, input={input_shape}")

        # Training spans model build + fit; export/evaluate is timed separately below.
        _train_start = time.perf_counter()

        # ── 1b. Small-dataset batch-size guard (Edge Impulse style) ─────────
        # On small datasets job.batch_size can equal or exceed train_samples,
        # collapsing steps_per_epoch to 1 and severely under-training the model.
        # Fix: clamp effective_batch_size so at least a few gradient updates
        # happen per epoch.  User epochs are NEVER changed.
        import math as _math
        _n_train = len(X_train)
        effective_batch_size = job.batch_size
        if effective_batch_size >= _n_train:
            # Prefer 8, floor to 1 when dataset is that tiny
            effective_batch_size = max(1, min(8, _n_train))
            logger.warning(
                f"[{job_id}] batch_size={job.batch_size} ≥ train_samples={_n_train}. "
                f"Clamped to effective_batch_size={effective_batch_size} so multiple "
                f"gradient updates occur per epoch. User epochs ({job.epochs}) unchanged."
            )
        _steps = _math.ceil(_n_train / effective_batch_size)
        if _steps == 1 and _n_train > 1:
            effective_batch_size = max(1, _n_train // 2)
            _steps = _math.ceil(_n_train / effective_batch_size)
            logger.warning(
                f"[{job_id}] steps_per_epoch was 1; reduced batch to "
                f"{effective_batch_size} → {_steps} steps/epoch."
            )
        _total_updates = job.epochs * _steps
        if _total_updates < 200:
            logger.warning(
                f"[{job_id}] Total gradient updates = {_total_updates} "
                f"({job.epochs} epochs × {_steps} steps). "
                f"Consider adding more samples or increasing epochs for better convergence."
            )

        # ── 2. Build model ───────────────────────────────────────────────────
        logger.info(
            f"[{job_id}] FOMO batch size: requested={job.batch_size} "
            f"effective={effective_batch_size} steps_per_epoch={_steps}"
        )

        _model_params = dict(ml_config.get("params") or {})
        # Inject fomo_version so model_builder dispatches to v1 or v2 builder.
        # Old impulses without the field default to 1 — v1 path is unchanged.
        if "fomo" in selected_architecture.lower():
            _model_params.setdefault("fomo_version", _fomo_version)

        # Check before model build (can be slow for transfer-learning models)
        _raise_if_job_cancelled(job_id)

        model = build_model(
            architecture=selected_architecture,
            input_shape=input_shape,
            num_classes=n_classes,
            params=_model_params,
        )
        model.summary(print_fn=logger.info)

        # Check after model build, before compile
        _raise_if_job_cancelled(job_id)

        # ── 3. Compile ───────────────────────────────────────────────────────
        is_fomo = "fomo" in selected_architecture.lower()
        
        # Safety check: SSD/Standard Object Detection still require non-standard training loops

        optimizer = tf.keras.optimizers.Adam(learning_rate=job.learning_rate)

        if is_fomo:
            _fomo_phase_cfg = _fomo_finetune_config(_fomo_version)
            # Two-phase transfer-learning schedule.
            #
            # Phase 1 (head warmup): freeze backbone, train only the two head
            # Conv2D layers at the full learning rate.  This protects the
            # pretrained ImageNet backbone features while the randomly-
            # initialised head learns basic object-vs-background discrimination.
            #
            # Phase 2 (fine-tune): backbone is unfrozen and the model is
            # recompiled at lr/10 between the two model.fit() calls below.
            FOMO_WARMUP_EPOCHS = int(_fomo_phase_cfg["warmup_epochs"])

            # bg_weight controls how much the loss penalises background cells
            # relative to object cells.  Lower → model predicts objects more
            # aggressively (higher recall, lower precision).
            #
            # Empirical validation on this dataset:
            #   bg_weight=0.01  → bg_mean=0.016, spatial ratio=47.5, FPs=55  ✓
            #   bg_weight=0.0024 → bg_mean=0.082, spatial ratio=7.4,  FPs=204 ✗
            #
            # The "gradient imbalance" argument (background gradient > object
            # gradient at 0.01) is real arithmetically, but empirically
            # bg_weight=0.01 achieves excellent spatial discrimination (ratio=47)
            # because the model still sees proportionally MORE object gradient at
            # TRUE object cells across the epoch.  Reducing bg_weight to 0.0024
            # collapses background suppression and causes FP explosion.
            #
            # bg_weight=0.01 (100:1 object:bg ratio).
            # 0.02 caused tiger recall to collapse (tiger weight=1.0 couldn't
            # overcome stronger bg penalty while monkey at 2.0x cap was fine —
            # asymmetric class weights make bg_weight changes class-specific).
            # Override via extra_params['fomo_bg_weight'] for experiments.
            _bg_weight_override = (job.extra_params or {}).get("fomo_bg_weight")
            if _bg_weight_override is not None:
                _bg_weight: float = float(_bg_weight_override)
                _bg_weight_adaptive = False
                logger.info(
                    "[FOMO] bg_weight=%.4f (manual override — adaptive tuning disabled)",
                    _bg_weight,
                )
            else:
                _bg_weight = _compute_initial_fomo_bg_weight(y_train, fomo_version=_fomo_version)
                _bg_weight_adaptive = True
                logger.info(
                    "[FOMO v%d] bg_weight=%.4f (dataset-adaptive, object:bg ratio ≈ %d:1%s)",
                    _fomo_version, _bg_weight, round(1.0 / _bg_weight),
                    " with strengthened v2 suppression window" if _fomo_version == 2 else "",
                )

            _grid_h = y_train.shape[1] if y_train.ndim == 4 else 0
            _grid_w = y_train.shape[2] if y_train.ndim == 4 else 0
            _target_cfg = _fomo_training_target_config(_fomo_version)
            _object_ratio = (
                float(np.sum(y_train[:, :, :, 1:] > 0.5)) / float(y_train.shape[0] * y_train.shape[1] * y_train.shape[2])
                if y_train.ndim == 4 and y_train.shape[0] > 0 else 0.0
            )
            _fomo_sparse_cfg = _fomo_sparse_positive_config(_object_ratio, _fomo_version)
            logger.info(
                "[FOMO] Training metadata: version=%d  grid=%d×%d  target_mode=%s  spread=%s  bg_weight=%.4f",
                _fomo_version, _grid_h, _grid_w, _target_cfg["mode"], _target_cfg["spread"], _bg_weight,
            )
            logger.info(
                "[FOMO v%d] Fine-tune plan: warmup_epochs=%d  fine_tune_lr=%.2e  strategy=%s",
                _fomo_version,
                FOMO_WARMUP_EPOCHS,
                float(_fomo_phase_cfg["fine_tune_lr"]),
                _fomo_phase_cfg["unfreeze_strategy"],
            )
            logger.info(
                "[FOMO v%d] Sparse-positive tuning: object_ratio=%.4f  obj_weight=%.2f  warmup_focal_bg_gamma=%.2f  fine_tune_focal_bg_gamma=%.2f",
                _fomo_version,
                _object_ratio,
                _fomo_sparse_cfg["obj_weight"],
                _fomo_sparse_cfg["warmup_focal_bg_gamma"],
                _fomo_sparse_cfg["fine_tune_focal_bg_gamma"],
            )

            # Per-class inverse-frequency loss weights.
            #
            # When one object class has more GT heatmap cells than another,
            # it contributes proportionally more gradient, causing the model to
            # learn the majority class well while underlearning the minority class.
            # Inverse-frequency weights correct this at the loss level.
            #
            # Normalisation: divide by min(raw_w), i.e. the class with the MOST
            # cells gets weight=1.0 (unchanged); classes with fewer cells get
            # weight > 1.0 (upweighted).  This ensures no class LOSES gradient
            # relative to the uniform baseline — only the minority is boosted.
            #
            # Example (min-norm): cat=510 cells, dog=423 cells →
            #   raw_w      = [1/423, 1/510] = [0.00237, 0.00196]
            #   min(raw_w) = 0.00196  (cat — majority class)
            #   final      = [1.206, 1.000]
            # Dog cells get +20% gradient; cat cells unchanged at 1.0.
            # (Previous mean-norm gave dog=1.093, cat=0.907 — cat lost 9% gradient,
            # contributing to cat recall regression from 62% → 38%.)
            _fomo_class_weights: Optional[np.ndarray] = None
            _cell_counts: Optional[np.ndarray] = None
            if n_classes >= 2 and y_train.ndim == 4:
                _cell_counts = np.array(
                    [float(np.sum(y_train[:, :, :, ci + 1] > 0.5)) for ci in range(n_classes)],
                    dtype=np.float32,
                )
                logger.info(
                    "[FOMO] Per-class GT cell counts in training heatmaps: "
                    + ", ".join(f"{label_names[i]}={int(_cell_counts[i])}" for i in range(n_classes))
                )
                _nonzero_mask = _cell_counts > 0
                if np.any(_nonzero_mask):
                    if not np.all(_nonzero_mask):
                        logger.warning(
                            "[FOMO] Classes with zero GT cells — weight clamped to majority; "
                            "check label_map UUID↔name alignment for: "
                            + ", ".join(label_names[i] for i in range(n_classes) if not _nonzero_mask[i])
                        )
                    # Substitute zero-count classes with max observed count so 1/count doesn't
                    # blow up; those classes get weight=1.0 (majority baseline) rather than inf.
                    _safe_counts = np.where(_nonzero_mask, _cell_counts,
                                            np.max(_cell_counts[_nonzero_mask]))
                    _fomo_class_weights = _compute_fomo_class_weights(
                        _safe_counts,
                        label_names,
                        fomo_version=_fomo_version,
                    )
                    _weight_label = (
                        " with v2 difficulty path"
                        if _fomo_version == 2 else
                        " (inverse-frequency, majority=1.0)"
                    )
                    logger.info(
                        "[FOMO] Per-class loss weights%s: %s",
                        _weight_label,
                        ", ".join(
                            f"{label_names[i]}={_fomo_class_weights[i]:.3f}"
                            for i in range(n_classes)
                        ),
                    )
                else:
                    logger.info(
                        "[FOMO] All classes have zero GT cells — "
                        "skipping per-class loss weighting (check label_map)."
                    )
            _fomo_loss_fn = _fomo_weighted_softmax_loss(
                bg_weight=_bg_weight,
                class_weights=_fomo_class_weights,
                focal_bg_gamma=float(_fomo_sparse_cfg["warmup_focal_bg_gamma"]),
                obj_weight=float(_fomo_sparse_cfg["obj_weight"]),
                **_fomo_loss_tuning_config(_fomo_version),
            )

            _fomo_head_names = frozenset({"fomo_head_conv", "fomo_head"})
            for _layer in model.layers:
                _layer.trainable = _layer.name in _fomo_head_names

            model.compile(
                optimizer=optimizer,  # full LR for head-only warmup
                loss=_fomo_loss_fn,
            )
            logger.info(
                "[FOMO] Batch size: requested=%s  effective=%s  steps_per_epoch=%s",
                job.batch_size,
                effective_batch_size,
                _steps,
            )
            logger.info(
                f"[FOMO] Phase 1 — backbone frozen, "
                f"training head only for {FOMO_WARMUP_EPOCHS} warmup epochs "
                f"at lr={job.learning_rate:.2e}"
            )
        else:
            model.compile(
                optimizer=optimizer,
                loss="sparse_categorical_crossentropy",
                metrics=["accuracy"],
            )

        # ── 4. Train ─────────────────────────────────────────────────────────
        PATIENCE = 20
        fomo_patience = None
        if is_fomo:
            fomo_patience = _resolve_fomo_phase2_patience(job.epochs, FOMO_WARMUP_EPOCHS)
            # Align FOMO checkpoint selection with the final evaluation target.
            # When an explicit test split exists, monitor val_f1 on that held-
            # out split instead of the Keras validation tail from X_train/y_train.
            # Keep checkpoint selection aligned with strict evaluation targets.
            # The fallback train-tail validation path still scores against the
            # strict y_train_eval tensors for metric parity.
            if len(X_test) > 0 and len(y_test) > 0:
                _X_val = X_test
                _y_val = y_test
            else:
                # No explicit test split available — fall back to the same tail
                # Keras uses for validation_split on the training tensors, but
                # score against hard (spread=False) targets for metric parity.
                _n_val   = max(1, int(len(X_train) * job.validation_split))
                _X_val   = X_train[-_n_val:]
                _y_val   = y_train_eval[-_n_val:]

            fomo_f1_cb = _FomoValF1Callback(_X_val, _y_val, label_names, fomo_version=_fomo_version, job_id=job_id)
            bg_adapt_cb = _FomoBgAdaptCallback(
                X_val=_X_val,
                y_val=_y_val,
                label_names=label_names,
                bg_weight=_bg_weight,
                class_weights=_fomo_class_weights,
                warmup_epochs=FOMO_WARMUP_EPOCHS,
                fine_tune_lr=float(_fomo_phase_cfg["fine_tune_lr"]),
                fomo_version=_fomo_version,
                base_obj_weight=float(_fomo_sparse_cfg["obj_weight"]),
                adapt_every=2,
                adapt_max_steps=10,
            ) if _bg_weight_adaptive else None

        else:
            fomo_f1_cb = None
            early_stop_cb = tf.keras.callbacks.EarlyStopping(
                monitor="val_accuracy", patience=PATIENCE,
                restore_best_weights=True,
            )

        progress_metric = "val_f1" if is_fomo else "val_accuracy"
        # FOMO Phase 2 early stopping is handled by _FomoStopPolicy in the outer
        # loop (see below).  Keras EarlyStopping is intentionally excluded: its
        # on_train_begin resets the wait counter on every model.fit() call, so it
        # cannot enforce patience across chunk boundaries.
        # ── Build and emit live setup log lines before training starts ──────────
        if is_fomo:
            _fomo_log_lines: list = []
            # Per-epoch training metrics, persisted incrementally alongside
            # log_lines so partial runs survive a crash. FOMO is a detection
            # architecture — only loss is tracked; accuracy fields are omitted
            # (val_f1 is detection F1, not classification accuracy).
            _fomo_epoch_metrics: list = []
            try:
                _total_p   = model.count_params()
                _train_p   = int(sum(int(np.prod(v.shape)) for v in model.trainable_variables))
                _nontr_p   = _total_p - _train_p
                _fomo_log_lines += [
                    f"Total params: {_total_p:,}",
                    f"Trainable params: {_train_p:,}",
                    f"Non-trainable params: {_nontr_p:,}",
                ]
            except Exception:
                pass
            _bg_adaptive_tag = (
                "dataset-adaptive, object:bg ratio ≈ %d:1%s" % (
                    round(1.0 / _bg_weight),
                    " with strengthened v2 suppression window" if _fomo_version == 2 else "",
                ) if _bg_weight_adaptive else "manual override"
            )
            _fomo_log_lines.append(
                f"[FOMO v{_fomo_version}] adaptive bg_weight={_bg_weight:.4f} ({_bg_adaptive_tag})"
            )
            _fomo_log_lines.append(
                f"[FOMO] Training metadata: version={_fomo_version}  grid={_grid_h}x{_grid_w}"
                f"  target_mode={_target_cfg['mode']}  spread={_target_cfg['spread']}"
                f"  bg_weight={_bg_weight:.4f}"
            )
            _fomo_log_lines.append(
                f"[FOMO v{_fomo_version}] Fine-tune plan: warmup_epochs={FOMO_WARMUP_EPOCHS}"
                f"  fine_tune_lr={float(_fomo_phase_cfg['fine_tune_lr']):.2e}"
                f"  strategy={_fomo_phase_cfg['unfreeze_strategy']}"
            )
            _fomo_log_lines.append(
                f"[FOMO v{_fomo_version}] Sparse-positive tuning: object_ratio={_object_ratio:.4f}"
                f"  obj_weight={_fomo_sparse_cfg['obj_weight']:.2f}"
                f"  warmup_focal_bg_gamma={_fomo_sparse_cfg['warmup_focal_bg_gamma']:.2f}"
                f"  fine_tune_focal_bg_gamma={_fomo_sparse_cfg['fine_tune_focal_bg_gamma']:.2f}"
            )
            if _cell_counts is not None and n_classes >= 2:
                _fomo_log_lines.append(
                    "[FOMO] Per-class GT cell counts: "
                    + ", ".join(f"{label_names[i]}={int(_cell_counts[i])}" for i in range(n_classes))
                )
                _weight_label = (
                    " with v2 difficulty path" if _fomo_version == 2
                    else " (inverse-frequency, majority=1.0)"
                )
                if _fomo_class_weights is not None:
                    _fomo_log_lines.append(
                        f"[FOMO] Per-class loss weights{_weight_label}: "
                        + ", ".join(
                            f"{label_names[i]}={_fomo_class_weights[i]:.3f}"
                            for i in range(n_classes)
                        )
                    )
            _fomo_log_lines.append(
                f"[FOMO] Batch size: requested={job.batch_size}"
                f"  effective={effective_batch_size}  steps_per_epoch={_steps}"
            )
            _fomo_log_lines.append(
                f"[FOMO] Phase 1 — backbone frozen, training head only"
                f" for {FOMO_WARMUP_EPOCHS} warmup epochs at lr={job.learning_rate:.2e}"
            )
            # Write setup lines to DB so the frontend can display them before epochs start.
            job.training_history = _sanitize_json({"log_lines": list(_fomo_log_lines), "epoch_metrics": list(_fomo_epoch_metrics), "is_fomo": True})
            db.commit()

        if fomo_f1_cb is not None:
            _live_log_cb = _FomoLiveLogCallback(job, db, _fomo_log_lines, _fomo_epoch_metrics, job.epochs)
            callbacks = [cb for cb in [
                fomo_f1_cb,  # computes + writes val_f1 into logs each epoch
                bg_adapt_cb, # periodic bg_weight re-tuning + recompile flag
                _live_log_cb,
                _ProgressCallback(self, job_id, job.epochs, progress_metric=progress_metric),
            ] if cb is not None]
        else:
            # Per-epoch metrics for the Training Graphs, persisted incrementally
            # (same cadence as FOMO's live-log callback) so partial runs survive.
            _clf_epoch_metrics: list = []
            callbacks = [
                early_stop_cb,
                _ClassifierLiveMetricsCallback(job, db, _clf_epoch_metrics),
                _ProgressCallback(self, job_id, job.epochs, progress_metric=progress_metric),
            ]

        if is_fomo:
            # ── Two-phase training (avoids mid-fit recompile) ─────────────────
            #
            # WHY two model.fit() calls instead of a callback:
            #   Calling model.compile() inside on_epoch_begin sets
            #   model.train_function = None.  Keras builds train_function once
            #   before the epoch loop; the training steps then call
            #   None(iterator) → TypeError: 'NoneType' object is not callable.
            #   Two sequential fit() calls compile the model at a clean
            #   boundary and avoid this internal state issue entirely.
            #
            # Check before training begins
            _raise_if_job_cancelled(job_id)

            # Phase 1 — head-only warmup (backbone frozen, full LR)
            # No early stopping during warmup: we always run exactly
            # FOMO_WARMUP_EPOCHS so the head can converge on the frozen features.
            h_warmup = model.fit(
                X_train, y_train,
                epochs=FOMO_WARMUP_EPOCHS,
                batch_size=effective_batch_size,
                validation_split=job.validation_split,
                callbacks=[fomo_f1_cb, _live_log_cb],
                verbose=0,
            )

            # Rebuild loss before fine-tune recompile (adaptive updates via _FomoBgAdaptCallback).
            _fomo_loss_fn = _fomo_weighted_softmax_loss(
                bg_weight=_bg_weight,
                class_weights=_fomo_class_weights,
                focal_bg_gamma=float(_fomo_sparse_cfg["fine_tune_focal_bg_gamma"]),
                obj_weight=float(_fomo_sparse_cfg["obj_weight"]),
                **_fomo_loss_tuning_config(_fomo_version),
            )

            if (
                _fomo_version == 2
                and n_classes >= 2
                and y_train.ndim == 4
                and _cell_counts is not None
                and getattr(fomo_f1_cb, "best_per_class", None)
            ):
                _fomo_class_weights = _compute_fomo_class_weights(
                    _cell_counts,
                    label_names,
                    fomo_version=_fomo_version,
                    warmup_per_class_metrics=fomo_f1_cb.best_per_class,
                    warmup_per_class_history=fomo_f1_cb.per_class_history,
                )
                logger.info(
                    "[FOMO v2] Warmup-informed class weights for fine-tuning: %s",
                    ", ".join(
                        f"{label_names[i]}={_fomo_class_weights[i]:.3f}"
                        for i in range(n_classes)
                    ),
                )
                _fomo_loss_fn = _fomo_weighted_softmax_loss(
                    bg_weight=_bg_weight,
                    class_weights=_fomo_class_weights,
                    focal_bg_gamma=float(_fomo_sparse_cfg["fine_tune_focal_bg_gamma"]),
                    obj_weight=float(_fomo_sparse_cfg["obj_weight"]),
                    **_fomo_loss_tuning_config(_fomo_version),
                )

            _restored_warmup_best = False
            if _fomo_version == 2:
                _restored_warmup_best = _restore_fomo_v2_best_warmup_weights(
                    model,
                    fomo_f1_cb,
                    FOMO_WARMUP_EPOCHS,
                )

            # Check between Phase 1 and Phase 2
            _raise_if_job_cancelled(job_id)

            _ft_layer_summary = _apply_fomo_finetune_trainable_layers(model, _fomo_version)
            # Emit Phase 2 start line to live log now that _ft_layer_summary is known.
            _fomo_log_lines.append(
                f"[FOMO v{_fomo_version}] Phase 2 — fine-tuning"
                f" at lr={float(_fomo_phase_cfg['fine_tune_lr']):.2e}"
                f"  strategy={_ft_layer_summary.get('strategy', '?')}"
                f"  trainable_layers={_ft_layer_summary.get('trainable_layers', '?')}"
            )
            job.training_history = _sanitize_json({"log_lines": list(_fomo_log_lines), "epoch_metrics": list(_fomo_epoch_metrics), "is_fomo": True})
            db.commit()
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=float(_fomo_phase_cfg["fine_tune_lr"])),
                loss=_fomo_loss_fn,
            )
            logger.info(
                "[FOMO] Phase 2 — fine-tuning at lr=%.2e  strategy=%s  trainable_layers=%d%s",
                float(_fomo_phase_cfg["fine_tune_lr"]),
                _ft_layer_summary["strategy"],
                _ft_layer_summary["trainable_layers"],
                "  start=best_warmup" if _restored_warmup_best else "",
            )

            # Phase 2 — chunked fine-tune so _FomoBgAdaptCallback can recompile
            # safely between fit() calls instead of mid-epoch (avoids train_function=None).
            _finetune_epochs = max(1, job.epochs - FOMO_WARMUP_EPOCHS)
            _adapt_chunk     = bg_adapt_cb.adapt_every if bg_adapt_cb is not None else _finetune_epochs
            _remaining       = _finetune_epochs
            _ft_histories    = []
            _phase2_epoch_cursor = FOMO_WARMUP_EPOCHS
            # v2: cap the number of mid-finetune class-weight refreshes so we
            # don't recompile every chunk when a class is persistently lagging.
            _v2_class_refresh_count = 0
            _MAX_V2_CLASS_REFRESHES = 3
            # Persistent cross-chunk early-stop policy for Phase 2.
            # Replaces Keras EarlyStopping (which resets wait on every fit()).
            _fomo_stop_policy = _FomoStopPolicy(patience=fomo_patience)
            _fomo_stop_policy.start(fomo_f1_cb, FOMO_WARMUP_EPOCHS)
            while _remaining > 0:
                _chunk = min(_adapt_chunk, _remaining)
                _phase2_next_epoch = _phase2_epoch_cursor + _chunk
                _h = model.fit(
                    X_train, y_train,
                    initial_epoch=_phase2_epoch_cursor,
                    epochs=_phase2_next_epoch,
                    batch_size=effective_batch_size,
                    validation_split=job.validation_split,
                    callbacks=callbacks,
                    verbose=0,
                )
                # Use actual epochs run (< _chunk if a within-chunk mechanism
                # fired model.stop_training, e.g. v2 center-drift detection).
                _actual_chunk = len(_h.history.get("loss", []))
                _ft_histories.append(_h)
                _remaining -= _actual_chunk
                _phase2_epoch_cursor += _actual_chunk
                # Check cross-chunk stop policy and cancellation before recompiling.
                _raise_if_job_cancelled(job_id)
                if _fomo_stop_policy.step(_actual_chunk, fomo_f1_cb) or _actual_chunk < _chunk:
                    break

                # v2 dynamic lagging-class boost: re-compute class weights when
                # a class has been stuck at recall < 0.05 for most of the last
                # n_recent epochs.  Activates only after warmup (history is
                # populated by _FomoValF1Callback from the warmup phase onward).
                _v2_class_refresh = _fomo_v2_can_refresh_class_weights(
                    fomo_version=_fomo_version,
                    cell_counts=_cell_counts,
                    n_classes=n_classes,
                    y_train_ndim=y_train.ndim,
                    per_class_history=(fomo_f1_cb.per_class_history if fomo_f1_cb is not None else None),
                    refresh_count=_v2_class_refresh_count,
                    max_refreshes=_MAX_V2_CLASS_REFRESHES,
                )
                if _v2_class_refresh:
                    _new_cw = _compute_fomo_class_weights(
                        _cell_counts,
                        label_names,
                        fomo_version=_fomo_version,
                        warmup_per_class_metrics=fomo_f1_cb.best_per_class,
                        warmup_per_class_history=fomo_f1_cb.per_class_history,
                    )
                    if _new_cw is not None:
                        _fomo_class_weights = _new_cw
                        _v2_class_refresh_count += 1
                        logger.info(
                            "[FOMO v2] lagging-class weight boost reactivated (%d/%d): %s",
                            _v2_class_refresh_count,
                            _MAX_V2_CLASS_REFRESHES,
                            ", ".join(
                                f"{label_names[i]}={_fomo_class_weights[i]:.3f}"
                                for i in range(n_classes)
                            ),
                        )

                _bg_adapt_needed = bg_adapt_cb is not None and bg_adapt_cb._recompile_needed
                if _bg_adapt_needed or _v2_class_refresh:
                    _recompile_bg  = bg_adapt_cb.bg_weight if _bg_adapt_needed else _bg_weight
                    _recompile_obj = bg_adapt_cb.obj_weight if _bg_adapt_needed else float(_fomo_sparse_cfg["obj_weight"])
                    _adapted_loss = _fomo_weighted_softmax_loss(
                        bg_weight=_recompile_bg,
                        class_weights=_fomo_class_weights,
                        focal_bg_gamma=float(_fomo_phase_cfg["fine_tune_focal_bg_gamma"]),
                        obj_weight=_recompile_obj,
                        **_fomo_loss_tuning_config(_fomo_version),
                    )
                    _apply_fomo_finetune_trainable_layers(model, _fomo_version)
                    model.compile(
                        optimizer=tf.keras.optimizers.Adam(learning_rate=float(_fomo_phase_cfg["fine_tune_lr"])),
                        loss=_adapted_loss,
                    )
                    if _bg_adapt_needed:
                        bg_adapt_cb._recompile_needed = False
                        logger.info(
                            "[FOMO] recompiled between chunks — bg_weight=%.4f (%d epochs remaining)",
                            bg_adapt_cb.bg_weight, _remaining,
                        )

            # Merge chunked histories into a single object for downstream use.
            h_finetune = _ft_histories[-1]
            if len(_ft_histories) > 1:
                for _k in h_finetune.history:
                    h_finetune.history[_k] = sum(
                        (_fh.history.get(_k, []) for _fh in _ft_histories), []
                    )

            if bg_adapt_cb is not None:
                _bg_weight = bg_adapt_cb.bg_weight

            # Merge both histories so downstream epoch-count / metric storage
            # sees one continuous training run.
            _all_keys = set(
                list(h_warmup.history.keys()) + list(h_finetune.history.keys())
            )
            _merged_dict = {
                k: list(h_warmup.history.get(k, []))
                   + list(h_finetune.history.get(k, []))
                for k in _all_keys
            }
            history = type("_History", (), {"history": _merged_dict})()

        else:
            # Check before training begins
            _raise_if_job_cancelled(job_id)
            history = model.fit(
                X_train, y_train,
                epochs=job.epochs,
                batch_size=effective_batch_size,
                validation_split=job.validation_split,
                callbacks=callbacks,
                verbose=0,
            )

        # For FOMO: restore the epoch with the highest val_f1 (best detection
        # quality), not the lowest val_loss.  This mirrors EI checkpoint selection.
        if is_fomo and fomo_f1_cb is not None:
            _fomo_actual_epochs = len(history.history.get("loss", []))
            _fomo_stopped_early = _fomo_actual_epochs < job.epochs
            if (
                _fomo_version == 2
                and fomo_f1_cb.best_epoch >= 0
                and fomo_f1_cb.best_epoch < FOMO_WARMUP_EPOCHS
            ):
                logger.warning(
                    "[FOMO v2] Fine-tuning underperformed warmup; preferring warmup checkpoint from epoch %d (val_f1=%.4f).",
                    fomo_f1_cb.best_epoch + 1,
                    fomo_f1_cb.best_f1,
                )
            if _fomo_stopped_early:
                _fomo_log_lines.append(
                    "[FOMO] Phase 2 early stop: val_f1 did not improve for "
                    f"{fomo_patience} consecutive epochs "
                    f"(best_f1={fomo_f1_cb.best_f1:.4f} at epoch {fomo_f1_cb.best_epoch + 1}, "
                    f"patience={fomo_patience})."
                )
                job.training_history = _sanitize_json({"log_lines": list(_fomo_log_lines), "epoch_metrics": list(_fomo_epoch_metrics), "is_fomo": True})
                db.commit()
            if fomo_f1_cb.restore_best_weights():
                _fomo_log_lines.append(
                    f"[FOMO] Restored best weights (val_f1={fomo_f1_cb.best_f1:.4f})."
                )
                job.training_history = _sanitize_json({"log_lines": list(_fomo_log_lines), "epoch_metrics": list(_fomo_epoch_metrics), "is_fomo": True})
                db.commit()

        # Check after training, before evaluate/export
        _raise_if_job_cancelled(job_id)

        _phase_ms["train_ms"] = round((time.perf_counter() - _train_start) * 1000.0, 2)
        _export_start = time.perf_counter()

        # ── 5. Evaluate ──────────────────────────────────────────────────────
        if is_fomo:
            # Real detection evaluation using grid-cell center-point matching.
            # Uses the test split when available, falls back to training split
            # (same heuristic used for the classification path below).
            _eval_X = X_test if len(X_test) > 0 else X_train
            _eval_y = y_test if len(y_test) > 0 else y_train_eval
            # How many images in the evaluated split are deliberate negatives.
            # An all-background GT set means two very different things depending
            # on this number: with zero negatives it means the label_map lost
            # its keys and every heatmap came out empty by accident (a failure);
            # with negatives present it is a legitimate false-positive-rate
            # split (a result).  Counted from the DB because the heatmaps
            # themselves cannot tell the two apart.
            _eval_split = SampleType.testing if len(X_test) > 0 else SampleType.training
            _n_bg_eval = sum(
                1
                for _s in db.query(Sample).filter(
                    Sample.project_id == impulse.project_id,
                    Sample.sample_type == _eval_split,
                ).all()
                if is_background_sample(_s)
            )
            cr = _evaluate_fomo_detection(
                model,
                _eval_X,
                _eval_y,
                label_names,
                bg_weight=_bg_weight,
                fomo_version=_fomo_version,
                job_id=job_id,
                n_background=_n_bg_eval,
            )
            # Use the detection-style CM built inside _evaluate_fomo_detection
            # (non-empty only when fomo_eval_status == "success").
            cm = cr.get("confusion_matrix", [])
            _threshold = cr.get("threshold")
            _macro = cr.get("macro avg", {}) if isinstance(cr, dict) else {}
            _macro_p = _macro.get("precision")
            _macro_r = _macro.get("recall")
            _macro_f1 = _macro.get("f1-score")
            _eval_line = "[FOMO] Final evaluation"
            if _threshold is not None:
                _eval_line += f" @ threshold={float(_threshold):.2f}"
            if _macro_f1 is not None:
                _eval_line += f" - macro_F1: {float(_macro_f1):.4f}"
            if _macro_p is not None:
                _eval_line += f" - macro_P: {float(_macro_p):.4f}"
            if _macro_r is not None:
                _eval_line += f" - macro_R: {float(_macro_r):.4f}"
            _fomo_log_lines.append(_eval_line)
            job.training_history = _sanitize_json({"log_lines": list(_fomo_log_lines), "epoch_metrics": list(_fomo_epoch_metrics), "is_fomo": True})
            db.commit()
        else:
            from sklearn.metrics import confusion_matrix, classification_report
            y_pred = np.argmax(model.predict(X_test if len(X_test) > 0 else X_train), axis=1)
            y_true = y_test if len(y_test) > 0 else y_train

            labels_indices = list(range(len(label_names)))
            cm = confusion_matrix(y_true, y_pred, labels=labels_indices).tolist()
            cr = classification_report(
                y_true, y_pred, labels=labels_indices, target_names=label_names, 
                output_dict=True, zero_division=0
            )

        # Epoch count — use "loss" which is always present for both FOMO and
        # classification.  "accuracy" is absent for FOMO so it cannot be used here.
        actual_epochs = len(history.history.get("loss", []))
        stopped_early = actual_epochs < job.epochs

        if is_fomo:
            best_val_acc   = None   # not applicable — heatmap cells are mostly background
            final_acc      = None
            _best_f1_val   = fomo_f1_cb.best_f1 if fomo_f1_cb is not None else 0.0
            stop_reason    = (
                f"Stopped early at epoch {actual_epochs}/{job.epochs} — "
                f"val_f1 did not improve for {fomo_patience} consecutive epochs."
                if stopped_early else
                f"Completed all {actual_epochs} requested training cycle(s)."
            )
        else:
            best_val_acc   = max(history.history.get("val_accuracy", [0]))
            final_acc      = history.history.get("val_accuracy", [0])[-1]
            stop_reason    = (
                f"Stopped early at epoch {actual_epochs}/{job.epochs} — "
                f"val_accuracy did not improve for {PATIENCE} consecutive epochs."
                if stopped_early else
                f"Completed all {actual_epochs} requested training cycle(s)."
            )

        # ── 6. Save Keras model ──────────────────────────────────────────────
        _raise_if_job_cancelled(job_id)
        version = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        with tempfile.TemporaryDirectory() as tmpdir:
            keras_path = os.path.join(tmpdir, "model.keras")
            model.save(keras_path)

            with open(keras_path, "rb") as f:
                keras_key = storage.model_key(impulse.project_id, version, "model.keras")
                storage.upload_file(f, keras_key, "application/octet-stream")

            # FOMO metadata: v1 is unchanged; v2 records its stride-8 grid info.
            # grid_size = img_h // stride (number of grid rows/cols for the
            # training resolution). Consumers use it to decode without knowing stride.
            if is_fomo:
                _grid_divisor_meta = 8 if _fomo_version == 2 else 16
                _grid_size_meta = input_shape[0] // _grid_divisor_meta if len(input_shape) >= 2 else 0
                _fomo_extra = {
                    "output_type":   "detection_heatmap",
                    "fomo_version":  _fomo_version,
                    "grid_size":     _grid_size_meta,
                }
            else:
                _fomo_extra = {}

            # Derive normalization from the image DSP block when present so the
            # exported metadata matches the preprocessing that produced training
            # features.
            _dsp_blocks = impulse.dsp_blocks or [{"type": "raw", "params": {}}]
            _norm_block = next(
                (blk for blk in _dsp_blocks if (blk or {}).get("type") == "image"),
                _dsp_blocks[0],
            )
            _norm_params = (_norm_block or {}).get("params", {}) or {}
            # Deployment consumers receive raw image pixels, not the DSP output
            # tensors used during training. FOMO exports strip preprocessing for
            # TFLite conversion, so the deployed model expects [-1, 1] input.
            _normalize_input = True if is_fomo else not bool(_norm_params.get("normalize", True))
            _input_mean = 127.5 if is_fomo else 0.0
            _input_std = 127.5 if is_fomo else 255.0

            # channel_order: the DSP processor always outputs RGB; mark it explicitly
            # so inference.py and offline loaders never fall back to heuristics.
            _channel_order = "rgb"

            # model_type mirrors output_type for the consumer side.
            _model_type = "detection_heatmap" if is_fomo else "classification"

            _common_meta = {
                "project_id":      impulse.project_id,
                "input_shape":     list(input_shape),
                "num_classes":     n_classes,
                "label_names":     label_names,
                "architecture":    selected_architecture,
                "dsp_blocks":      _dsp_blocks,
                # Sampling parameters — consumed by device_client for DSP parity
                "frequency_hz":    float(impulse.frequency_hz or 100.0),
                "window_size_ms":  int(impulse.window_size_ms or 1000),
                "version":         1,
                "normalize_input": _normalize_input,
                "input_mean":      _input_mean,
                "input_std":       _input_std,
                "channel_order":   _channel_order,
                "model_type":      _model_type,
                **_fomo_extra,
            }

            keras_record = TrainedModel(
                training_job_id=job_id,
                version=version,
                format="keras",
                storage_key=keras_key,
                file_size_bytes=os.path.getsize(keras_path),
                model_metadata={
                    **_common_meta,
                    "variant": "float32",
                    "quantized": False,
                },
            )
            db.add(keras_record)

            # ── 7. Convert to TFLite (float32 + int8) ────────────────────────
            rep_data = X_train[:200]

            # 7a. Float32 (unoptimized)
            tflite_f32_bytes = _convert_to_tflite(model, rep_data, quantize=False)
            if tflite_f32_bytes:
                tflite_f32_key = storage.model_key(
                    impulse.project_id, version, "model_float32.tflite",
                )
                storage.upload_bytes(
                    tflite_f32_bytes, tflite_f32_key, "application/octet-stream",
                )
                db.add(TrainedModel(
                    training_job_id=job_id,
                    version=version,
                    format="tflite",
                    storage_key=tflite_f32_key,
                    file_size_bytes=len(tflite_f32_bytes),
                    model_metadata={
                        **_common_meta,
                        "variant": "float32",
                        "quantized": False,
                    },
                ))
            else:
                logger.warning(f"[{job_id}] Float32 TFLite conversion failed")

            # 7b. Int8 (quantized)
            int8_status = "available"
            int8_error = None
            tflite_int8_bytes = _convert_to_tflite(model, rep_data, quantize=True)
            if tflite_int8_bytes:
                tflite_int8_key = storage.model_key(
                    impulse.project_id, version, "model_int8.tflite",
                )
                storage.upload_bytes(
                    tflite_int8_bytes, tflite_int8_key, "application/octet-stream",
                )
                db.add(TrainedModel(
                    training_job_id=job_id,
                    version=version,
                    format="tflite",
                    storage_key=tflite_int8_key,
                    file_size_bytes=len(tflite_int8_bytes),
                    model_metadata={
                        **_common_meta,
                        "variant": "int8",
                        "quantized": True,
                    },
                ))
            else:
                int8_status = "failed"
                int8_error = (
                    "Int8 quantization failed. The model may use ops not "
                    "supported by the TFLite int8 converter."
                )
                logger.warning(f"[{job_id}] {int8_error}")

            # ── Per-variant export evaluation ─────────────────────────────────
            # Score each exported TFLite variant independently on the same
            # validation split, reusing this architecture's own eval so the
            # Model Version dropdown reports each variant's real numbers
            # instead of float32's.  Post-export only; per-variant failures are
            # recorded and never fail the training job.
            try:
                from app.ml.variant_eval import evaluate_exported_variants

                _ve_X = X_test if len(X_test) > 0 else X_train

                if is_fomo:
                    _ve_y = y_test if len(y_test) > 0 else y_train_eval

                    def _score_variant(_runner) -> dict:
                        _rep = _evaluate_fomo_detection(
                            _runner, _ve_X, _ve_y, label_names,
                            bg_weight=_bg_weight,
                            fomo_version=_fomo_version,
                            job_id=job_id,
                            n_background=_n_bg_eval,
                        )
                        _macro = _rep.get("macro avg", {}) or {}
                        return {
                            "eval_status":      _rep.get("fomo_eval_status"),
                            "eval_error":       _rep.get("fomo_eval_error"),
                            "precision":        _macro.get("precision"),
                            "recall":           _macro.get("recall"),
                            "f1_score":         _macro.get("f1-score"),
                            "threshold":        _rep.get("threshold"),
                            "background_fp_rate": _rep.get("background_fp_rate"),
                            "evaluated_images": int(len(_ve_X)),
                        }
                else:
                    from sklearn.metrics import classification_report as _clf_report

                    _ve_y_true = y_test if len(y_test) > 0 else y_train
                    _ve_labels = list(range(len(label_names)))

                    def _score_variant(_runner) -> dict:
                        _pred = np.argmax(_runner.predict(_ve_X), axis=1)
                        _rep = _clf_report(
                            _ve_y_true, _pred, labels=_ve_labels,
                            target_names=label_names,
                            output_dict=True, zero_division=0,
                        )
                        _macro = _rep.get("macro avg", {}) or {}
                        return {
                            "eval_status":      "success",
                            "accuracy":         _rep.get("accuracy"),
                            "precision":        _macro.get("precision"),
                            "recall":           _macro.get("recall"),
                            "f1_score":         _macro.get("f1-score"),
                            "per_class":        _rep,
                            "evaluated_images": int(len(_ve_X)),
                        }

                evaluate_exported_variants(
                    db=db,
                    job_id=job_id,
                    scorer=_score_variant,
                    variant_bytes={
                        "float32": tflite_f32_bytes,
                        "int8":    tflite_int8_bytes,
                    },
                    export_errors={"int8": int8_error},
                    log_prefix=f"[{job_id}] ",
                )
            except Exception as _ve_exc:
                logger.warning(
                    f"[{job_id}] per-variant export eval skipped: {_ve_exc}",
                    exc_info=True,
                )

        # ── 8. Write results to DB ───────────────────────────────────────────
        # Final guard: if cancellation arrived during model save / TFLite
        # conversion (the long-running steps above), do NOT overwrite the
        # cancelled status with 'completed'.  Re-fetches the DB row so we
        # see what the cancel endpoint wrote.
        _assert_not_cancelled_before_completing(job_id)
        job.status       = JobStatus.completed
        job.completed_at = datetime.utcnow()
        # Pointer flip — every successful run (fresh OR retrain) becomes the
        # new Active Model. Never cleared anywhere else, so previous pointers
        # survive cancels/failures of subsequent fresh runs.
        _promote_run_to_active(job.impulse_id, job.id)
        job.best_loss    = float(min(history.history.get("val_loss", [0])))

        if is_fomo:
            # Accuracy fields are left None — heatmap detection cannot be
            # honestly summarised as a single accuracy percentage.
            # Callers should check training_history["is_fomo"] and render
            # val_loss curves instead of accuracy curves.
            job.best_accuracy  = None
            job.final_accuracy = None

            # _fomo_log_lines already accumulated setup + per-epoch lines via
            # _FomoLiveLogCallback and the pre-training setup block above.

            _best_epoch = (
                int(fomo_f1_cb.best_epoch + 1)
                if fomo_f1_cb is not None and fomo_f1_cb.best_epoch >= 0
                else None
            )
            _final_eval_f1 = None
            if isinstance(cr, dict):
                _final_eval_f1 = cr.get("macro avg", {}).get("f1-score")
            _fomo_log_lines.append(
                "[FOMO] Training complete"
                + (f" - best_val_f1: {_best_f1_val:.4f}" if _best_f1_val is not None else "")
                + (f" - best_epoch: {_best_epoch}" if _best_epoch is not None else "")
                + (f" - final_macro_F1: {float(_final_eval_f1):.4f}" if _final_eval_f1 is not None else "")
            )

            job.training_history = _sanitize_json({
                # Only loss curves are stored; accuracy keys are intentionally absent.
                "loss":             [float(v) for v in history.history.get("loss",     [])],
                "val_loss":         [float(v) for v in history.history.get("val_loss", [])],
                "val_f1":           [float(v) for v in history.history.get("val_f1",   [])],
                "best_val_f1":      float(_best_f1_val),
                "best_val_f1_epoch": _best_epoch,
                "requested_epochs": job.epochs,
                "actual_epochs":    actual_epochs,
                "stopped_early":    stopped_early,
                "stop_reason":      stop_reason,
                "is_fomo":          True,
                "warmup_epochs":    FOMO_WARMUP_EPOCHS,
                "log_lines":        _fomo_log_lines,
                "log_lines_complete": True,
                "epoch_metrics":    list(_fomo_epoch_metrics),
                "best_threshold": (
                    float(fomo_f1_cb.best_threshold)
                    if fomo_f1_cb is not None and getattr(fomo_f1_cb, "best_threshold", None) is not None
                    else None
                ),
                "best_predicted_class_distribution": (
                    dict(fomo_f1_cb.best_predicted_class_distribution)
                    if fomo_f1_cb is not None else {}
                ),
                "metric_note": (
                    "FOMO heatmap training: checkpoint and early-stopping driven by val_f1 "
                    "(macro detection F1 across thresholds). "
                    "val_loss is also tracked for convergence diagnostics."
                ),
                "int8_status":      int8_status,
                "int8_error":       int8_error,
            })
            logger.info(
                f"[{job_id}] FOMO training complete — "
                f"best_val_f1={_best_f1_val:.4f}, best_val_loss={job.best_loss:.4f}"
            )
            result = {
                "job_id": job_id,
                "best_val_f1": float(_best_f1_val),
                "best_val_loss": float(job.best_loss),
            }
        else:
            job.best_accuracy  = float(best_val_acc)
            job.final_accuracy = float(final_acc)
            job.training_history = _sanitize_json({
                "accuracy":        [float(v) for v in history.history.get("accuracy",     [])],
                "val_accuracy":    [float(v) for v in history.history.get("val_accuracy", [])],
                "loss":            [float(v) for v in history.history.get("loss",         [])],
                "val_loss":        [float(v) for v in history.history.get("val_loss",     [])],
                "requested_epochs": job.epochs,
                "actual_epochs":    actual_epochs,
                "stopped_early":    stopped_early,
                "stop_reason":      stop_reason,
                "int8_status":      int8_status,
                "int8_error":       int8_error,
                "epoch_metrics":    list(_clf_epoch_metrics),
            })
            logger.info(f"[{job_id}] Training complete — val_accuracy={final_acc:.4f}")
            result = {"job_id": job_id, "accuracy": float(final_acc)}

        job.confusion_matrix      = cm
        job.classification_report = _sanitize_json(cr)
        db.commit()

        _phase_ms["export_ms"] = round((time.perf_counter() - _export_start) * 1000.0, 2)
        _log_event(
            _training_log, "training.finished",
            job_id=job_id, architecture=selected_architecture,
            status="completed", **_phase_ms,
        )
        return result

    except _CancelledError:
        # Cooperative cancellation.  Flow: flush the in-memory buffers (so every
        # pre-cancel line survives), append the single terminal line, then ensure
        # the cancelled status + completed_at — never rollback() between flush and
        # commit.  The inline FOMO/classifier loop owns its buffers in this scope;
        # the YOLO-Pro / SSD loops register theirs (resolved below) since their
        # source is introspected by tests and must not be restructured.
        logger.info(f"[{job_id}] Training cancelled cooperatively by worker")
        _log_lines, _epoch_metrics = _resolve_terminal_buffers(
            job_id, _fomo_log_lines, _fomo_epoch_metrics, _clf_epoch_metrics
        )
        _flush_training_buffers(db, job_id, _log_lines, _epoch_metrics)
        _append_terminal_log_and_status(
            db, job_id,
            status=JobStatus.cancelled,
            lines=["Training cancelled by user."],
        )
        # Do not re-raise — a cancelled task is not a Celery failure.
        _log_event(
            _training_log, "training.finished",
            job_id=job_id, status="cancelled",
            **(locals().get("_phase_ms") or {}),
        )
        return {"job_id": job_id, "status": "cancelled"}
    except Exception as e:
        logger.exception(f"[{job_id}] Training failed: {e}")
        _log_event(
            _training_log, "training.finished",
            level=logging.ERROR,
            job_id=job_id, status="failed", reason=str(e),
            **(locals().get("_phase_ms") or {}),
        )
        # Same flow as cancellation: flush buffers, append the terminal line(s),
        # then set status=failed.  The skip_if_status guard keeps a cancellation
        # that raced with a real error from being relabelled "failed".
        _log_lines, _epoch_metrics = _resolve_terminal_buffers(
            job_id, _fomo_log_lines, _fomo_epoch_metrics, _clf_epoch_metrics
        )
        _flush_training_buffers(db, job_id, _log_lines, _epoch_metrics)
        _append_terminal_log_and_status(
            db, job_id,
            status=JobStatus.failed,
            lines=["Training failed.", f"Reason: {_terminal_reason_line(str(e))}"],
            error_message=str(e),
            skip_if_status=JobStatus.cancelled,
        )
        raise
    finally:
        # Drop any registered live buffer for this job (covers the success path,
        # where neither except handler ran and so never popped it).
        _clear_live_buffer(job_id)
        db.close()


# ─── FOMO Loss ────────────────────────────────────────────────────────────────


def _compute_initial_fomo_bg_weight(y_train: np.ndarray, fomo_version: int = 1) -> float:
    """
    Estimate bg_weight from dataset object density.
    y_train shape: (N, H, W, C+1) — channel 0 is background.

    v2 uses its own adaptive window, but is no longer capped at the previous
    ultra-low ceiling that weakened background suppression and encouraged FP.

    Denominator = cells of *positive* images only
    --------------------------------------------
    `object_ratio` used to divide by every cell in the set, negatives included.
    That inverts the incentive the knob exists to express: adding background
    images raises the cell count without raising `gt_cells`, so `object_ratio`
    falls, so `bg_weight` falls — the model suppresses background *less* the
    more background evidence it is given.  Measuring density over the images
    that actually contain objects keeps the weight a property of how dense the
    objects are, independent of how many negatives sit alongside them.

    On a dataset with no negatives every image is positive, so this is
    numerically identical to the previous formula.

    Deliberately NOT done here: a separate weight for negative-image cells.
    Empty-scene cells and near-object-context cells really are different
    distributions, but a second weight would have to be plumbed through
    `_fomo_weighted_softmax_loss` as a per-sample tensor, and the resulting
    two-knob interaction with `_adapt_fomo_bg_weight`'s runtime feedback loop
    is not something to introduce blind.  The cap on negatives bounds the
    damage the single weight can do in the meantime.
    """
    if y_train.ndim != 4 or y_train.shape[-1] < 2:
        _, floor, _ = _fomo_bg_weight_bounds(fomo_version)
        return 0.01 if fomo_version == 1 else floor

    cells_per_image = y_train.shape[1] * y_train.shape[2]
    per_image_gt    = np.sum(y_train[:, :, :, 1:] > 0.5, axis=(1, 2, 3))
    n_pos_images    = int(np.count_nonzero(per_image_gt))
    gt_cells        = int(per_image_gt.sum())
    total_cells     = n_pos_images * cells_per_image
    if total_cells == 0 or gt_cells == 0:
        _, floor, _ = _fomo_bg_weight_bounds(fomo_version)
        return 0.01 if fomo_version == 1 else floor

    object_ratio = gt_cells / total_cells
    scale, floor, ceiling = _fomo_bg_weight_bounds(fomo_version)
    weight = float(np.clip(object_ratio * scale, floor, ceiling))
    logger.info(
        "[FOMO v%d] adaptive bg_weight: gt_cells=%d total_cells=%d "
        "(over %d/%d positive images) object_ratio=%.4f scale=%.2f "
        "bounds=[%.4f, %.4f] → initial_bg_weight=%.4f",
        fomo_version, gt_cells, total_cells, n_pos_images, int(y_train.shape[0]),
        object_ratio, scale, floor, ceiling, weight,
    )
    return weight


def _adapt_fomo_bg_weight(
    current: float,
    gt_mean: Optional[float],
    bg_mean: Optional[float],
    per_sample_max_nonbg_mean: Optional[float],
    fomo_version: int = 1,
) -> tuple:
    """
    Multi-tier bg_weight adjustment. Returns (new_weight, rule_name).
    Caller must enforce clamp, delta guard, and step limit.
    """
    if None in (gt_mean, bg_mean, per_sample_max_nonbg_mean):
        return current, "skip_missing"
    if fomo_version == 2:
        if bg_mean > 0.30 or per_sample_max_nonbg_mean > 0.85:
            return current * 1.20, "v2_strong_increase"
        if bg_mean > 0.08 or per_sample_max_nonbg_mean > 0.80:
            return current * 1.15, "v2_increase"
        if bg_mean > 0.05:
            return current * 1.08, "v2_moderate_increase"
        if gt_mean < 0.50 and bg_mean < 0.05:
            # BG is already quiet; gently reduce suppression so GT cells can
            # activate more freely.  Self-corrects if FPs appear next cycle.
            return current * 0.95, "v2_relax"
        return current, "stable"
    if bg_mean > 0.30 or per_sample_max_nonbg_mean > 0.85:
        return current * 2.0, "strong_increase"
    if bg_mean > 0.08 or per_sample_max_nonbg_mean > 0.80:
        return current * 1.8, "increase"           # was 1.4
    if bg_mean > 0.05:
        return current * 1.4, "moderate_increase"  # was 1.2
    if gt_mean < 0.50 and bg_mean < 0.05:
        return current * 1.1, "relax"
    return current, "stable"


def _fomo_weighted_softmax_loss(
    bg_weight: float = 0.01,
    class_weights=None,
    focal_bg_gamma: float = 0.0,
    obj_weight: float = 1.0,
    soft_obj_threshold: float = 0.5,
    soft_neighbor_weight: float = 1.0,
    center_dominance: float = 1.0,
    center_rank_penalty: float = 0.0,
    center_rank_margin: float = 0.0,
    center_peak_penalty: float = 0.0,
    center_peak_margin: float = 0.0,
    center_peak_orthogonal_scale: float = 1.0,
    center_peak_diagonal_scale: float = 1.0,
    center_rank_all_pairs: bool = False,
):
    """
    FOMO training loss: cell-level weighted softmax cross-entropy over logits.

    Alignment with Edge Impulse
    ---------------------------
    The original EI FOMO Keras script trains with a categorical cross-entropy
    loss applied per grid cell, with class weights that down-weight background
    cells to avoid gradient collapse when ~99 % of cells are background.

    Implementation
    --------------
    y_true  shape: (B, H, W, n_classes+1)  — one-hot heatmap; ch 0 = background.
    y_pred  shape: (B, H, W, n_classes+1)  — raw logits (no activation on head).

    CELL-LEVEL weighting (correct):
        scalar_ce = -sum_c [ y_true_c * log(softmax(y_pred)_c) ]  # per cell
        L_cell = cell_weight * scalar_ce
        where cell_weight = class_weight[cls] if the cell contains an object,
              bg_weight otherwise.

    class_weights — per-class inverse-frequency multipliers (shape: n_classes,)
        Upweights gradient for underrepresented object classes (e.g. dog) and
        downweights overrepresented ones (e.g. cat).  Computed from actual GT
        heatmap cell counts so the weighting reflects the real training-set
        imbalance.  When None, all object cells are weighted equally at 1.0
        (original behaviour).

    Why channel-level weighting is wrong:
        With channel-level weights the gradient for an object logit z_k at a
        background cell is +softmax_k * bg_weight (pushing it DOWN).  At a
        typical 96×96 image that gives ~2957 bg cells × 0.1 × 0.333 ≈ 98.5
        "down" units vs only ~40 obj cells × 0.667 ≈ 26.7 "up" units → net
        +71.8, which drives the model to uniform predictions (val_f1 = 0).

    With cell-level bg_weight=0.01 (default, ~100:1 object:bg ratio):
        NET = 40×0.667 − 2957×0.01×0.333 ≈ 26.7 − 9.8 = +16.9  → UP  ✓
        (Previous default 0.02 gave NET≈+7.0 — still positive but weaker signal)

    Parameters
    ----------
    bg_weight : float
        Loss weight applied to the scalar per-cell CE for background cells.
        Default 0.01 matches the ~100:1 object:bg ratio used in the Edge
        Impulse FOMO reference.  Previous default was 0.02 (≈50:1).
        Must satisfy bg_weight < n_obj_cells / (n_bg_cells * softmax_obj) to
        guarantee a net upward gradient on object logits.  0.01 is safe for
        grids up to ~200:1 bg/obj ratio.
        Configurable at training time via job.extra_params["fomo_bg_weight"].
    """
    import tensorflow as tf

    # Pre-convert class_weights to a TF constant once so the closure avoids
    # per-call numpy→tensor conversion overhead during training.
    _cw = (
        tf.constant(np.asarray(class_weights, dtype=np.float32), dtype=tf.float32)
        if class_weights is not None else None
    )

    # Scale the object-loss clip ceiling with both center_dominance AND the
    # maximum class weight so that boosted lagging classes (e.g. class_weight=8)
    # are not silently truncated back to an effective weight of 1.15.
    # Without _cw_max: weight=8, dominance=5.25 → loss=105 → clipped to 15.09
    #   → effective weight collapses to 1.15 for every class_weight > 1.15.
    # With _cw_max:    ceiling = 2.5 * 5.25 * 8.0 * 1.15 = 120.75 → no clipping.
    # v1 (dominance=1.0, no class_weights) stays at the legacy ceiling of 6.5.
    _cw_max = (
        float(np.max(class_weights))
        if class_weights is not None and len(class_weights) > 0
        else 1.0
    )
    _obj_clip_hi = max(6.5, 2.5 * float(center_dominance) * float(_cw_max) * 1.15)

    def loss_fn(y_true, y_pred):
        # Scalar per-cell CE: sum over channel dim first, then weight spatially.
        log_probs    = tf.nn.log_softmax(y_pred, axis=-1)          # (B, H, W, C+1)
        per_cell_ce  = -tf.reduce_sum(y_true * log_probs, axis=-1) # (B, H, W)
        probs_pred   = tf.nn.softmax(y_pred, axis=-1)
        # Cell-level spatial weight: hard-center object cells remain dominant.
        # v2 may also pass soft object support cells (<0.5 targets) so nearby
        # cells get some localization gradient without being treated as full
        # binary positives.
        obj_channels = y_true[:, :, :, 1:]                          # (B, H, W, C)
        obj_strength = tf.reduce_max(obj_channels, axis=-1)         # (B, H, W)
        has_obj      = obj_strength > tf.cast(soft_obj_threshold, tf.float32)
        hard_center  = obj_strength >= 0.99
        soft_neighbor = tf.logical_and(has_obj, tf.logical_not(hard_center))

        # Focal BG penalty: concentrate gradient on cells producing high object confidence.
        # focal_bg_gamma=0 → uniform bg_weight (original behaviour).
        # focal_bg_gamma=2 → BG cell at p_obj=0.9 gets weight ≈2.6× bg_weight;
        #                    BG cell at p_obj=0.05 gets weight ≈1.0× bg_weight.
        if focal_bg_gamma > 0.0:
            _max_obj_prob  = tf.reduce_max(probs_pred[:, :, :, 1:], axis=-1)  # (B,H,W)
            _focal_factor  = 1.0 + focal_bg_gamma * tf.square(_max_obj_prob)
            _bg_cell_w     = tf.cast(bg_weight, tf.float32) * _focal_factor
        else:
            _bg_cell_w = tf.fill(tf.shape(per_cell_ce), tf.cast(bg_weight, tf.float32))

        if _cw is not None:
            # Per-class object cell weight (inverse-frequency weighting)
            cell_obj_weight = tf.reduce_sum(obj_channels * _cw, axis=-1)  # (B, H, W)

            # ↑ Increase GT confidence (calibration fix)
            object_loss = cell_obj_weight * 2.5
            object_loss = tf.where(
                hard_center,
                object_loss * tf.cast(center_dominance, tf.float32),
                object_loss,
            )
            object_loss = tf.where(
                soft_neighbor,
                object_loss * tf.cast(soft_neighbor_weight, tf.float32),
                object_loss,
            )

            # Safety clip: ceiling scales with center_dominance to avoid
            # truncating the stronger v2 dominance multiplier.
            object_loss = tf.clip_by_value(object_loss, 0.1, _obj_clip_hi)

            cell_weight = tf.where(
                has_obj,
                object_loss * tf.cast(obj_weight, tf.float32),
                _bg_cell_w
            )
        else:
            # Uniform object weighting (no class weights)
            object_loss = tf.ones_like(per_cell_ce) * 3.5
            object_loss = tf.where(
                hard_center,
                object_loss * tf.cast(center_dominance, tf.float32),
                object_loss,
            )
            object_loss = tf.where(
                soft_neighbor,
                object_loss * tf.cast(soft_neighbor_weight, tf.float32),
                object_loss,
            )
            object_loss = tf.clip_by_value(object_loss, 0.1, _obj_clip_hi)

            cell_weight = tf.where(
                has_obj,
                object_loss * tf.cast(obj_weight, tf.float32),
                _bg_cell_w
            )

        total_loss = per_cell_ce * cell_weight

        if (
            center_rank_penalty > 0.0
            and center_rank_margin > 0.0
        ) or (
            center_peak_penalty > 0.0
            and center_peak_margin > 0.0
        ):
            center_class_mask = tf.cast(hard_center[..., tf.newaxis], tf.float32) * obj_channels
            if center_class_mask.shape.rank == 4:
                obj_probs = probs_pred[:, :, :, 1:]

                def _shift_with_neg_inf(tensor, dy, dx):
                    paddings = [
                        [0, 0],
                        [max(dy, 0), max(-dy, 0)],
                        [max(dx, 0), max(-dx, 0)],
                        [0, 0],
                    ]
                    padded = tf.pad(tensor, paddings, constant_values=-1e9)
                    h = tf.shape(tensor)[1]
                    w = tf.shape(tensor)[2]
                    start_y = max(-dy, 0)
                    start_x = max(-dx, 0)
                    return padded[:, start_y:start_y + h, start_x:start_x + w, :]

                orth_neighbor_logits = tf.stack([
                    _shift_with_neg_inf(obj_probs, -1, 0),
                    _shift_with_neg_inf(obj_probs,  1, 0),
                    _shift_with_neg_inf(obj_probs,  0, -1),
                    _shift_with_neg_inf(obj_probs,  0,  1),
                ], axis=0)
                diag_neighbor_logits = tf.stack([
                    _shift_with_neg_inf(obj_probs, -1, -1),
                    _shift_with_neg_inf(obj_probs, -1,  1),
                    _shift_with_neg_inf(obj_probs,  1, -1),
                    _shift_with_neg_inf(obj_probs,  1,  1),
                ], axis=0)
                neighbor_logits = tf.concat([orth_neighbor_logits, diag_neighbor_logits], axis=0)
                neighbor_max = tf.reduce_max(neighbor_logits, axis=0)
                orth_neighbor_max = tf.reduce_max(orth_neighbor_logits, axis=0)
                diag_neighbor_max = tf.reduce_max(diag_neighbor_logits, axis=0)
                center_probs = obj_probs * center_class_mask
                center_neighbor_max = neighbor_max * center_class_mask
                center_orth_neighbor_max = orth_neighbor_max * center_class_mask
                center_diag_neighbor_max = diag_neighbor_max * center_class_mask
                if _cw is not None:
                    center_class_weight = tf.reduce_sum(center_class_mask * _cw, axis=-1)
                else:
                    center_class_weight = tf.reduce_sum(center_class_mask, axis=-1)
                center_class_weight = tf.where(
                    center_class_weight > 0.0,
                    center_class_weight,
                    tf.ones_like(center_class_weight),
                )
                center_penalty_scale = tf.cast(obj_weight, tf.float32)

                if center_rank_penalty > 0.0 and center_rank_margin > 0.0:
                    _margin  = tf.cast(center_rank_margin, tf.float32)
                    _penalty = tf.cast(center_rank_penalty, tf.float32)

                    if center_rank_all_pairs:
                        # All-pairs pairwise hinge: each of the 8 surrounding cells
                        # is penalised independently.  Gradient flows to EVERY
                        # offending neighbour, not only the single argmax.
                        # Mean (not sum) over the 8 neighbours so that multi-
                        # neighbour cases can't additively explode the penalty.
                        pair_sum = tf.zeros_like(obj_probs)
                        # Count of neighbours that actually violate the margin
                        # (unweighted; used only for normalisation).  This avoids
                        # the original /8.0 which diluted the penalty to ~3–4% of
                        # its intended magnitude when only 1 of 8 neighbours fires.
                        violating_n = tf.zeros_like(obj_probs)
                        for _dy, _dx, _nb_scale in [
                            (-1, 0, 1.35), (1, 0, 1.35), (0, -1, 1.35), (0, 1, 1.35),
                            (-1, -1, 0.65), (-1, 1, 0.65), (1, -1, 0.65), (1, 1, 0.65),
                        ]:
                            _nb = _shift_with_neg_inf(obj_probs, _dy, _dx)
                            _pv = (
                                tf.nn.relu(_nb - obj_probs + _margin)
                                * center_class_mask
                                * tf.cast(_nb_scale, tf.float32)
                            )
                            pair_sum = pair_sum + _pv
                            violating_n = violating_n + tf.cast(_pv > 0, tf.float32)
                        # Normalise by actual violating-neighbour count (≥1) so
                        # the per-violation gradient magnitude is constant whether
                        # 1 or 8 neighbours exceed the margin.  Multi-neighbour
                        # explosion is prevented because we still average across
                        # violating neighbours rather than summing.
                        rank_penalty_map = tf.reduce_sum(
                            pair_sum / tf.maximum(violating_n, 1.0), axis=-1
                        )
                        total_loss = total_loss + (
                            rank_penalty_map
                            * _penalty
                            * center_penalty_scale
                            * center_class_weight
                        )
                    else:
                        # Legacy max-based approach (v1 path; penalty=0 for v1 anyway).
                        rank_violation = tf.nn.relu(
                            center_neighbor_max - center_probs + _margin
                        )
                        rank_penalty_map = tf.reduce_sum(
                            rank_violation * center_class_mask, axis=-1
                        )
                        total_loss = total_loss + (
                            rank_penalty_map
                            * _penalty
                            * center_penalty_scale
                            * tf.cast(center_dominance, tf.float32)
                            * center_class_weight
                        )

                if center_peak_penalty > 0.0 and center_peak_margin > 0.0:
                    # Direct local-argmax penalty: exact-cell evaluation only
                    # cares whether the GT center beats the strongest nearby
                    # rival, with orthogonal neighbours treated as the primary
                    # off-by-one competitors.
                    #
                    # Keep a linear hinge component so small local misorderings
                    # still get a meaningful correction. The earlier pure-square
                    # form became too gentle precisely in the "nearby cell beats
                    # the center by a bit" regime that dominates v2 exact-hit
                    # failures, especially for orthogonal neighbours.
                    _peak_margin = tf.cast(center_peak_margin, tf.float32)
                    _peak_penalty = tf.cast(center_peak_penalty, tf.float32)
                    orth_peak_violation = tf.nn.relu(
                        center_orth_neighbor_max - center_probs + _peak_margin
                    )
                    diag_peak_violation = tf.nn.relu(
                        center_diag_neighbor_max - center_probs + (_peak_margin * 0.85)
                    )
                    orth_peak_cost = orth_peak_violation + tf.square(orth_peak_violation)
                    diag_peak_cost = diag_peak_violation + tf.square(diag_peak_violation)
                    peak_penalty_map = tf.reduce_sum(
                        (
                            orth_peak_cost * tf.cast(center_peak_orthogonal_scale, tf.float32)
                            + diag_peak_cost * tf.cast(center_peak_diagonal_scale, tf.float32)
                        ) * center_class_mask,
                        axis=-1,
                    )
                    total_loss = total_loss + (
                        peak_penalty_map
                        * _peak_penalty
                        * center_penalty_scale
                        * center_class_weight
                    )
        # Sum over spatial dims (H, W), mean over batch (B).
        # This matches Keras class_weight behaviour: each image contributes one
        # loss value, then batch-mean is taken.  reduce_mean over B×H×W would
        # divide gradients by H×W (=81 for a 9×9 grid) — making them 81× too
        # small for the model to converge in a reasonable number of epochs.
        return tf.reduce_mean(tf.reduce_sum(total_loss, axis=[1, 2]))

    return loss_fn


# Keep the old name as an alias so any external callers don't break.
def _fomo_detection_loss(gamma: float = 2.0, bg_weight: float = 0.02):
    """Deprecated alias for _fomo_weighted_softmax_loss.  gamma is ignored."""
    return _fomo_weighted_softmax_loss(bg_weight=bg_weight)


def _fomo_probs_for_logging(raw_preds: np.ndarray) -> np.ndarray:
    """Normalize raw FOMO logits/probabilities into probabilities for logging."""
    arr = np.asarray(raw_preds, dtype=np.float32)
    if arr.size == 0:
        return arr
    sums = arr.sum(axis=-1, keepdims=True)
    if np.all(arr >= 0.0) and np.all(arr <= 1.0) and np.all(np.isfinite(sums)) and np.all(np.abs(sums - 1.0) <= 5e-3):
        return arr
    return tf.nn.softmax(arr, axis=-1).numpy()


def _fomo_center_alignment_diagnostics(
    preds: np.ndarray,
    y_heatmap: Optional[np.ndarray],
    n_classes: int,
) -> Dict[str, float]:
    """
    Return local center-vs-neighbor diagnostics using probabilities.

    Mirrors the evaluation rule closely enough for training-time guards:
    exact center placement is good only when the GT center outranks its nearby
    cells for the matching class.
    """
    result = {
        "mean_gap": 0.0,
        "outranked_rate": 0.0,
        "exact_hit_recall": 0.0,
        "mean_center_prob": 0.0,
        "mean_nearby_prob": 0.0,
        "n_gt": 0,
    }
    if y_heatmap is None or preds is None:
        return result
    probs = _fomo_probs_for_logging(preds)
    if probs.ndim != 4 or y_heatmap.shape != probs.shape or n_classes <= 0:
        return result

    center_neighbor_gaps: List[float] = []
    gt_center_probs: List[float] = []
    mean_near_probs: List[float] = []
    outranked_count = 0
    exact_hit_count = 0
    grid_h, grid_w = probs.shape[1], probs.shape[2]

    for bi in range(y_heatmap.shape[0]):
        gt_positions = np.argwhere(np.max(y_heatmap[bi, :, :, 1:], axis=-1) > 0.5)
        for gy, gx in gt_positions:
            cls_idx = int(np.argmax(y_heatmap[bi, gy, gx, 1:]))
            center_prob = float(probs[bi, gy, gx, cls_idx + 1])
            near_probs = []
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dy == 0 and dx == 0:
                        continue
                    ny, nx = gy + dy, gx + dx
                    if 0 <= ny < grid_h and 0 <= nx < grid_w:
                        near_probs.append(float(probs[bi, ny, nx, cls_idx + 1]))
            if not near_probs:
                continue
            max_near = max(near_probs)
            gap = center_prob - max_near
            center_neighbor_gaps.append(gap)
            gt_center_probs.append(center_prob)
            mean_near_probs.append(float(np.mean(near_probs)))
            if gap < 0.0:
                outranked_count += 1
            else:
                exact_hit_count += 1

    if center_neighbor_gaps:
        n_gt = len(center_neighbor_gaps)
        result.update({
            "mean_gap": float(np.mean(center_neighbor_gaps)),
            "outranked_rate": float(outranked_count / n_gt),
            "exact_hit_recall": float(exact_hit_count / n_gt),
            "mean_center_prob": float(np.mean(gt_center_probs)),
            "mean_nearby_prob": float(np.mean(mean_near_probs)),
            "n_gt": int(n_gt),
        })
    return result


def _fomo_v2_should_hold_bg_weight(
    *,
    bg_mean: Optional[float],
    center_diag: Optional[Dict[str, float]],
) -> bool:
    """
    Hold v2 bg_weight steady when background is already quiet but center
    placement is still poor.
    """
    if bg_mean is None or not center_diag:
        return False
    mean_gap = float(center_diag.get("mean_gap", 0.0) or 0.0)
    outranked_rate = float(center_diag.get("outranked_rate", 0.0) or 0.0)
    exact_hit_recall = float(center_diag.get("exact_hit_recall", 0.0) or 0.0)
    n_gt = int(center_diag.get("n_gt", 0) or 0)
    return bool(
        n_gt > 0
        and bg_mean < 0.08
        and (
            mean_gap < -0.02
            or outranked_rate > 0.35
            or exact_hit_recall < 0.35
        )
    )


def _fomo_v2_center_score(center_diag: Optional[Dict[str, float]]) -> float:
    """
    Collapse v2 center-placement diagnostics into one comparable score.

    Higher is better. This is intentionally only a secondary/tie-break signal;
    macro-F1 remains the primary checkpoint metric.
    """
    if not center_diag:
        return -1e9
    mean_gap = float(center_diag.get("mean_gap", 0.0) or 0.0)
    outranked_rate = float(center_diag.get("outranked_rate", 1.0) or 1.0)
    exact_hit_recall = float(center_diag.get("exact_hit_recall", 0.0) or 0.0)
    mean_center_prob = float(center_diag.get("mean_center_prob", 0.0) or 0.0)
    return (
        mean_gap * 3.0
        + exact_hit_recall * 2.5
        - outranked_rate * 2.0
        + mean_center_prob * 0.35
    )


def _fomo_v2_center_tie_break_better(
    *,
    candidate_gap: float,
    candidate_outranked: float,
    candidate_exact_hit: float,
    candidate_center_score: float,
    best_gap: float,
    best_outranked: float,
    best_exact_hit: float,
    best_center_score: float,
) -> bool:
    """
    Return True when v2 center placement is meaningfully better for close-F1 ties.
    """
    metric_wins = 0
    if candidate_exact_hit > best_exact_hit + 0.02:
        metric_wins += 1
    if candidate_gap > best_gap + 0.02:
        metric_wins += 1
    if candidate_outranked < best_outranked - 0.04:
        metric_wins += 1

    if candidate_center_score > best_center_score + 0.015 and metric_wins >= 1:
        return True
    return metric_wins >= 2


def _fomo_v2_center_dominant_candidate(
    *,
    candidate_gap: float,
    candidate_outranked: float,
    candidate_exact_hit: float,
    candidate_center_score: float,
    best_gap: float,
    best_outranked: float,
    best_exact_hit: float,
    best_center_score: float,
    candidate_f1: float,
    best_f1: float,
) -> bool:
    """
    Return True when center placement improved enough to justify a tiny F1 tradeoff.
    """
    if candidate_f1 < best_f1 - 0.004:
        return False
    if candidate_center_score <= best_center_score + 0.015:
        return False
    return bool(
        candidate_exact_hit > best_exact_hit + 0.05
        or candidate_gap > best_gap + 0.04
        or candidate_outranked < best_outranked - 0.08
        or _fomo_v2_center_tie_break_better(
            candidate_gap=candidate_gap,
            candidate_outranked=candidate_outranked,
            candidate_exact_hit=candidate_exact_hit,
            candidate_center_score=candidate_center_score,
            best_gap=best_gap,
            best_outranked=best_outranked,
            best_exact_hit=best_exact_hit,
            best_center_score=best_center_score,
        )
    )


def _fomo_v2_center_clearly_regressed(
    *,
    candidate_gap: float,
    candidate_outranked: float,
    candidate_exact_hit: float,
    candidate_center_score: float,
    best_gap: float,
    best_outranked: float,
    best_exact_hit: float,
    best_center_score: float,
) -> bool:
    """
    Return True when later fine-tuning has clearly drifted away from the best
    exact-center placement regime.
    """
    metric_losses = 0
    if candidate_exact_hit < best_exact_hit - 0.03:
        metric_losses += 1
    if candidate_gap < best_gap - 0.03:
        metric_losses += 1
    if candidate_outranked > best_outranked + 0.05:
        metric_losses += 1
    return bool(
        candidate_center_score < best_center_score - 0.035
        or metric_losses >= 2
    )


# ─── FOMO Helpers ─────────────────────────────────────────────────────────────

# Thresholds used for the per-threshold F1 sweep stored at training time.
# The panel endpoint surfaces these so the UI can show a sweep table without
# re-running inference on demand.
EVAL_THRESHOLDS: List[float] = [
    0.01, 0.02, 0.03, 0.05, 0.10, 0.15, 0.20, 0.25,
    0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70,
    0.75, 0.80, 0.85, 0.90,
]

class _FomoStopPolicy:
    """
    Persistent Phase 2 early-stop controller for FOMO fine-tuning.

    Keras ``EarlyStopping`` resets its ``wait`` counter in ``on_train_begin``
    on every ``model.fit()`` call, so it cannot enforce patience across the
    chunked Phase 2 loop.  This class owns the stop state for the entire
    fine-tune run and is queried from the outer loop after each chunk.

    Improvement is measured against ``_FomoValF1Callback.best_f1`` /
    ``best_epoch``, which already track the global optimum across chunks.
    The controller counts phase-2 epochs elapsed since the last time
    ``best_epoch`` advanced past the warmup baseline, then fires when that
    count reaches ``patience``.

    Parameters
    ----------
    patience  : int    Epochs without improvement before stop triggers.
                       For FOMO this is derived from the configured fine-tune
                       epoch budget so the chunk-persistent policy scales with
                       the requested training run.
    min_delta : float  Minimum absolute improvement in ``val_f1`` that counts
                       as progress.  Matches the primary threshold used by
                       ``_FomoValF1Callback._should_promote`` for v1 (1e-6).
    """

    def __init__(self, patience: int, min_delta: float = 1e-6) -> None:
        self.patience  = patience
        self.min_delta = min_delta
        # Snapshot of warmup's best state (set by start()).
        self._phase2_start_epoch:  int   = -1
        self._baseline_best_epoch: int   = -1
        # Best phase-2 state seen so far (updated by step()).
        self._tracked_best_f1:    float = -1.0
        self._tracked_best_epoch: int   = -1
        # Cumulative phase-2 epochs actually executed.
        self._total_phase2_epochs: int  = 0
        self.stop_triggered: bool = False

    def start(self, fomo_f1_cb: "object", phase2_start_epoch: int) -> None:
        """Snapshot warmup state.  Call once, just before the Phase 2 loop."""
        self._phase2_start_epoch   = phase2_start_epoch
        self._baseline_best_epoch  = int(getattr(fomo_f1_cb, "best_epoch", -1) or -1)
        self._tracked_best_f1      = float(getattr(fomo_f1_cb, "best_f1",   -1.0) or -1.0)
        self._tracked_best_epoch   = self._baseline_best_epoch

    def step(self, actual_chunk_epochs: int, fomo_f1_cb: "object") -> bool:
        """
        Account for the epochs just completed and return True if training should stop.

        Parameters
        ----------
        actual_chunk_epochs : int
            Epochs that *actually* ran (use ``len(h.history["loss"])`` rather
            than the planned chunk size so a mid-chunk stop is counted correctly).
        fomo_f1_cb : _FomoValF1Callback
            Live callback whose ``best_f1`` / ``best_epoch`` reflect the global
            optimum across all chunks seen so far.
        """
        self._total_phase2_epochs += actual_chunk_epochs

        cb_best_f1    = float(getattr(fomo_f1_cb, "best_f1",    -1.0) or -1.0)
        cb_best_epoch = int(  getattr(fomo_f1_cb, "best_epoch",  -1)   or -1)

        # Advance tracked baseline when the callback recorded a new Phase 2 best
        # with meaningful improvement (best_epoch must have moved forward into
        # Phase 2 territory, i.e. past the warmup snapshot).
        if (
            cb_best_epoch > self._tracked_best_epoch
            and cb_best_f1 > self._tracked_best_f1 + self.min_delta
        ):
            self._tracked_best_f1    = cb_best_f1
            self._tracked_best_epoch = cb_best_epoch

        # Phase-2 epochs elapsed since the last tracked improvement.
        if self._tracked_best_epoch >= self._phase2_start_epoch:
            phase2_improvement_at = self._tracked_best_epoch - self._phase2_start_epoch
            epochs_without = self._total_phase2_epochs - phase2_improvement_at - 1
        else:
            # best_epoch is still in warmup — no Phase 2 improvement yet.
            epochs_without = self._total_phase2_epochs

        if epochs_without >= self.patience:
            self.stop_triggered = True
            logger.info(
                "[FOMO] Phase 2 early stop: val_f1 did not improve for %d consecutive "
                "epochs (best_f1=%.4f at epoch %d, patience=%d).",
                epochs_without,
                self._tracked_best_f1,
                self._tracked_best_epoch + 1,
                self.patience,
            )

        return self.stop_triggered


class _FomoValF1Callback(tf.keras.callbacks.Callback):
    """
    Computes a detection-quality val_f1 after every epoch for FOMO models.

    This callback re-uses the existing _evaluate_fomo_detection() logic
    (including the multi-threshold sweep) to derive a macro-F1 from the
    validation split.  The best macro-F1 over all thresholds in EVAL_THRESHOLDS
    is logged as the ``val_f1`` metric so that EarlyStopping and
    ModelCheckpoint can monitor it instead of val_loss.

    The callback also applies softmax to the raw logits produced by the new
    head before calling the evaluator, matching the probability-based
    thresholding used in _evaluate_fomo_detection.

    Parameters
    ----------
    X_val        : np.ndarray  Validation feature tensors.
    y_val        : np.ndarray  Validation heatmap targets (one-hot, (N,H,W,C+1)).
    label_names  : list[str]   Class names (excluding background).
    patience_ckpt: str | None  Path for manual best-weight checkpoint (optional;
                               used internally by the training loop).
    """

    def __init__(
        self,
        X_val: np.ndarray,
        y_val: np.ndarray,
        label_names: List[str],
        fomo_version: int = 1,
        job_id: str = "",
    ):
        super().__init__()
        self.X_val       = X_val
        self.y_val       = y_val
        self.label_names = label_names
        self.fomo_version = fomo_version
        self.job_id      = job_id
        self.best_f1     = -1.0
        self.best_weights: Optional[list] = None
        self.best_epoch: int = -1
        self.best_per_class: Dict[str, Dict[str, float]] = {}
        self.best_predicted_class_distribution: Dict[str, int] = {}
        self.per_class_history: Dict[str, List[Dict[str, float]]] = {
            name: [] for name in label_names
        }
        self.best_threshold: float = 0.5
        self.best_min_recall: float = -1.0
        self.best_total_fp: int = 10**9
        self.best_center_gap: float = -1e9
        self.best_outranked_rate: float = 1e9
        self.best_exact_hit_recall: float = -1e9
        self.best_center_score: float = -1e9
        self.threshold_history: List[Dict[str, float]] = []
        self._v2_center_drift_epochs: int = 0

    # ------------------------------------------------------------------
    # Public helpers (called by the training loop after fit() completes)
    # ------------------------------------------------------------------

    def restore_best_weights(self) -> bool:
        """Restore the weights that achieved the highest val_f1.  Returns True on success."""
        if self.best_weights is not None and self.model is not None:
            self.model.set_weights(self.best_weights)
            logger.info(
                f"[FOMO] Restored best weights (val_f1={self.best_f1:.4f})."
            )
            return True
        return False

    # ------------------------------------------------------------------
    # Keras callback hook
    # ------------------------------------------------------------------

    def on_epoch_end(self, epoch: int, logs=None):
        # Diagnostic source: Steps 2, 4 — _FomoValF1Callback (per-epoch)
        _raise_if_job_cancelled(self.job_id)
        logs = logs or {}

        if len(self.X_val) == 0:
            logs["val_f1"] = 0.0
            return

        _raise_if_job_cancelled(self.job_id)
        try:
            raw_preds = self.model.predict(self.X_val, verbose=0)
        except _CancelledError:
            raise
        except Exception as exc:
            logger.warning(f"[FomoValF1Callback] predict failed at epoch {epoch}: {exc}")
            logs["val_f1"] = 0.0
            return

        _raise_if_job_cancelled(self.job_id)
        # Apply softmax to convert raw logits → probabilities.
        # _evaluate_fomo_detection expects probability scores for thresholding.
        probs = tf.nn.softmax(raw_preds, axis=-1).numpy()

        # Use the threshold sweep to pick the best macro-F1 across all thresholds.
        # This mirrors EI's validation metric which implicitly picks the
        # "operating point" that maximises detection quality.
        best_macro_f1 = 0.0
        _best_tp = _best_fp = _best_fn = 0
        _best_thresh = 0.5
        _best_per_class: Dict[str, Dict[str, float]] = {}
        _best_predicted_dist: Dict[str, int] = {}
        _LOG_THRESH_SET = {0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90}
        _raise_if_job_cancelled(self.job_id)
        try:
            sweep = _fomo_threshold_sweep(probs, self.y_val, self.label_names, job_id=self.job_id)
            if sweep:
                best_macro_f1 = max(entry["macro_f1"] for entry in sweep)
                _cands = [e for e in sweep if round(e["threshold"], 2) in _LOG_THRESH_SET]
                _best_entry = (
                    max(_cands, key=lambda e: e["macro_f1"]) if _cands
                    else max(sweep, key=lambda e: e["macro_f1"])
                )
                _best_thresh = _best_entry["threshold"]
                _best_tp = _best_entry["total_tp"]
                _best_fp = _best_entry["total_fp"]
                _best_fn = _best_entry["total_fn"]
                _best_per_class = _best_entry.get("per_class", {})
                _best_predicted_dist = _best_entry.get("predicted_class_distribution", {})
        except _CancelledError:
            raise
        except Exception as exc:
            logger.warning(f"[FomoValF1Callback] threshold sweep failed at epoch {epoch}: {exc}")

        logs["val_f1"] = float(best_macro_f1)
        _raise_if_job_cancelled(self.job_id)
        _center_diag = (
            _fomo_center_alignment_diagnostics(probs, self.y_val, len(self.label_names))
            if self.fomo_version == 2 else {}
        )
        _center_score = (
            _fomo_v2_center_score(_center_diag)
            if self.fomo_version == 2 else -1e9
        )
        self.threshold_history.append({
            "epoch": float(epoch + 1),
            "threshold": float(_best_thresh),
            "val_f1": float(best_macro_f1),
        })

        # Track best weights manually so we can restore them even when
        # EarlyStopping does not call restore_best_weights itself.
        _candidate_min_recall = min(
            (float(v.get("recall", 0.0) or 0.0) for v in _best_per_class.values()),
            default=0.0,
        )
        _candidate_gap = float(_center_diag.get("mean_gap", -1e9) or -1e9)
        _candidate_outranked = float(_center_diag.get("outranked_rate", 1e9) or 1e9)
        _candidate_exact_hit = float(_center_diag.get("exact_hit_recall", -1e9) or -1e9)
        _v2_tie_break_better = (
            self.fomo_version == 2
            and _fomo_v2_center_tie_break_better(
                candidate_gap=_candidate_gap,
                candidate_outranked=_candidate_outranked,
                candidate_exact_hit=_candidate_exact_hit,
                candidate_center_score=_center_score,
                best_gap=self.best_center_gap,
                best_outranked=self.best_outranked_rate,
                best_exact_hit=self.best_exact_hit_recall,
                best_center_score=self.best_center_score,
            )
        )
        # For v2: materially better exact-center placement can justify a tiny
        # F1 tradeoff, but only when the direct center diagnostics confirm the
        # model has moved into a genuinely better local-placement regime.
        _v2_center_dominant = (
            self.fomo_version == 2
            and _fomo_v2_center_dominant_candidate(
                candidate_gap=_candidate_gap,
                candidate_outranked=_candidate_outranked,
                candidate_exact_hit=_candidate_exact_hit,
                candidate_center_score=_center_score,
                best_gap=self.best_center_gap,
                best_outranked=self.best_outranked_rate,
                best_exact_hit=self.best_exact_hit_recall,
                best_center_score=self.best_center_score,
                candidate_f1=best_macro_f1,
                best_f1=self.best_f1,
            )
        )
        _v2_center_clearly_worse = (
            self.fomo_version == 2
            and _fomo_v2_center_clearly_regressed(
                candidate_gap=_candidate_gap,
                candidate_outranked=_candidate_outranked,
                candidate_exact_hit=_candidate_exact_hit,
                candidate_center_score=_center_score,
                best_gap=self.best_center_gap,
                best_outranked=self.best_outranked_rate,
                best_exact_hit=self.best_exact_hit_recall,
                best_center_score=self.best_center_score,
            )
        )
        if self.fomo_version == 2:
            _f1_delta = best_macro_f1 - self.best_f1
            _should_promote = (
                _f1_delta > 0.002
                or _v2_center_dominant
                or (
                    _f1_delta > 1e-6 and not _v2_center_clearly_worse
                )
                or (
                    best_macro_f1 >= self.best_f1 - 0.004 and (
                        _v2_tie_break_better or
                        _candidate_min_recall > self.best_min_recall + 1e-6 or
                        (
                            abs(_candidate_min_recall - self.best_min_recall) <= 1e-6 and
                            _best_fp < self.best_total_fp
                        )
                    )
                )
            )
        else:
            _should_promote = (
                best_macro_f1 > self.best_f1 + 1e-6 or
                (
                    abs(best_macro_f1 - self.best_f1) <= 0.002 and (
                        _candidate_min_recall > self.best_min_recall + 1e-6 or
                        (
                            abs(_candidate_min_recall - self.best_min_recall) <= 1e-6 and
                            _best_fp < self.best_total_fp
                        )
                    )
                )
            )
        if _should_promote:
            self.best_f1     = best_macro_f1
            self.best_weights = [w.copy() for w in self.model.get_weights()]
            self.best_epoch = epoch
            self.best_per_class = dict(_best_per_class)
            self.best_predicted_class_distribution = dict(_best_predicted_dist)
            self.best_threshold = float(_best_thresh)
            self.best_min_recall = float(_candidate_min_recall)
            self.best_total_fp = int(_best_fp)
            self.best_center_gap = _candidate_gap
            self.best_outranked_rate = _candidate_outranked
            self.best_exact_hit_recall = _candidate_exact_hit
            self.best_center_score = _center_score
            self._v2_center_drift_epochs = 0
        elif self.fomo_version == 2:
            _clear_f1_regression = best_macro_f1 < self.best_f1 - 0.008
            _clear_center_regression = _fomo_v2_center_clearly_regressed(
                candidate_gap=_candidate_gap,
                candidate_outranked=_candidate_outranked,
                candidate_exact_hit=_candidate_exact_hit,
                candidate_center_score=_center_score,
                best_gap=self.best_center_gap,
                best_outranked=self.best_outranked_rate,
                best_exact_hit=self.best_exact_hit_recall,
                best_center_score=self.best_center_score,
            )
            if _clear_f1_regression and _clear_center_regression:
                self._v2_center_drift_epochs += 1
            else:
                self._v2_center_drift_epochs = 0
            if self._v2_center_drift_epochs >= 3:
                logger.warning(
                    "[FOMO v2] Center-placement drift detected for %d consecutive epochs; "
                    "stopping fine-tuning and keeping the earlier best checkpoint.",
                    self._v2_center_drift_epochs,
                )
                self.model.stop_training = True

        logger.info(
            f"Epoch {epoch + 1} | val_F1={best_macro_f1:.4f} | "
            f"TP={_best_tp} FP={_best_fp} FN={_best_fn} | best_thresh={_best_thresh:.2f}"
        )
        for _cls_name, _cls_counts in _best_per_class.items():
            self.per_class_history.setdefault(_cls_name, []).append(dict(_cls_counts))
            logger.info(
                f"  class={_cls_name} thresh={_best_thresh:.2f} "
                f"TP={_cls_counts['tp']} FP={_cls_counts['fp']} FN={_cls_counts['fn']}"
            )
        if _best_predicted_dist:
            logger.info(
                "[FOMO] predicted class distribution @%.2f: %s",
                _best_thresh,
                ", ".join(f"{k}={v}" for k, v in _best_predicted_dist.items()),
            )
        _trend_payload = {}
        for _cls_name, _hist in self.per_class_history.items():
            if not _hist:
                continue
            _recent = _hist[-3:]
            _trend_payload[_cls_name] = {
                "recall_history": [round(float(h.get("recall", 0.0) or 0.0), 4) for h in _recent],
                "latest_recall": round(float(_recent[-1].get("recall", 0.0) or 0.0), 4),
                "avg_recall": round(sum(float(h.get("recall", 0.0) or 0.0) for h in _recent) / len(_recent), 4),
                "lagging_epochs": sum(1 for h in _hist if float(h.get("recall", 0.0) or 0.0) < 0.05),
            }
        _threshold_trend = {}
        if self.threshold_history:
            _recent_thr = self.threshold_history[-4:]
            _threshold_trend = {
                "threshold_history": [round(float(h["threshold"]), 4) for h in _recent_thr],
                "val_f1_history": [round(float(h["val_f1"]), 4) for h in _recent_thr],
                "stability_range": round(
                    max(float(h["threshold"]) for h in _recent_thr) - min(float(h["threshold"]) for h in _recent_thr),
                    4,
                ),
            }

        if self.fomo_version == 2:
            log_fomo_v2_diagnostics(
                fomo_version=self.fomo_version,
                feature_tap="epoch_val",
                grid_h=self.y_val.shape[1],
                grid_w=self.y_val.shape[2],
                spread_mode=False,
                bg_weight=0.0,
                preds_probs=probs,
                y_heatmap=self.y_val,
                n_classes=len(self.label_names),
                per_class_metrics=_best_per_class,
                predicted_class_distribution=_best_predicted_dist,
                per_class_trends=_trend_payload,
                threshold_trend=_threshold_trend,
                context=f"epoch_{epoch + 1}",
            )


class _FomoBgAdaptCallback(tf.keras.callbacks.Callback):
    """
    Periodically re-tunes bg_weight during fine-tune phase and recompiles
    the model in-place. Runs at most _adapt_max_steps times, every
    _adapt_every epochs, and only after warmup_epochs.
    """
    _CLAMP_LO = 0.002
    _CLAMP_HI = 0.20
    _MIN_DELTA_FRAC = 0.05

    def __init__(
        self,
        X_val: np.ndarray,
        y_val: np.ndarray,
        label_names: list,
        bg_weight: float,
        class_weights,           # np.ndarray or None — passed to loss rebuild
        warmup_epochs: int,
        fine_tune_lr: float,
        fomo_version: int = 1,
        base_obj_weight: float = 1.0,
        adapt_every: int = 3,
        adapt_max_steps: int = 10,
    ):
        super().__init__()
        self.X_val           = X_val
        self.y_val           = y_val
        self.label_names     = label_names
        self.bg_weight       = bg_weight
        self.class_weights   = class_weights
        self.warmup_epochs   = warmup_epochs
        self.fine_tune_lr    = fine_tune_lr
        self.fomo_version    = fomo_version
        self.base_obj_weight = base_obj_weight
        self.adapt_every     = adapt_every
        self.adapt_max_steps = adapt_max_steps
        self._adapt_steps    = 0
        self._recompile_needed = False
        self.obj_weight      = base_obj_weight
        self._bg_mean_history: list = []   # rolling window for smoothed bg_mean
        self._CLAMP_HI_LOCAL = 0.025 if fomo_version == 2 else self._CLAMP_HI

    def on_epoch_end(self, epoch: int, logs=None):
        post_warmup_epoch = epoch - self.warmup_epochs
        if post_warmup_epoch < 0:
            return
        if self._adapt_steps >= self.adapt_max_steps:
            return
        if post_warmup_epoch % self.adapt_every != 0:
            return
        try:
            raw  = self.model.predict(self.X_val, verbose=0)
            diag = _fomo_confidence_diagnostics(
                raw, len(self.label_names), y_heatmap=self.y_val,
            )
        except Exception as exc:
            logger.warning(
                "[FOMO] bg_adapt diagnostics failed at epoch %d: %s", epoch + 1, exc,
            )
            return

        gt_diag = diag.get("gt_cell_confidence", {})
        gt_mean = gt_diag.get("gt_mean_obj_conf")
        bg_mean = gt_diag.get("bg_mean_obj_conf")
        nonbg   = diag.get("per_sample_max_nonbg_confidence", {}).get("mean")
        center_diag = _fomo_center_alignment_diagnostics(
            raw,
            self.y_val,
            len(self.label_names),
        ) if self.fomo_version == 2 else {}

        # Smooth bg_mean over a 2-epoch window to avoid reacting to single-epoch
        # augmentation-induced noise spikes.
        if bg_mean is not None:
            self._bg_mean_history.append(bg_mean)
            if len(self._bg_mean_history) > 2:
                self._bg_mean_history.pop(0)
            bg_mean = float(sum(self._bg_mean_history) / len(self._bg_mean_history))

        _hold_bg_for_center_ranking = (
            self.fomo_version == 2
            and _fomo_v2_should_hold_bg_weight(
                bg_mean=bg_mean,
                center_diag=center_diag,
            )
        )
        if _hold_bg_for_center_ranking:
            logger.info(
                "[FOMO v2] Holding bg_weight at %.4f while center ranking is still weak "
                "(mean_gap=%.4f, outranked_rate=%.4f, exact_hit_recall=%.4f).",
                self.bg_weight,
                float(center_diag.get("mean_gap", 0.0) or 0.0),
                float(center_diag.get("outranked_rate", 0.0) or 0.0),
                float(center_diag.get("exact_hit_recall", 0.0) or 0.0),
            )

        # obj_weight boost: amplify GT cell gradient when GT confidence is weak.
        # e.g. gt_mean=0.44 → obj_weight=2.84; gt_mean=0.70 → obj_weight=1.80; gt_mean=0.90 → obj_weight=1.0
        _new_obj_w = (
            min(3.0, self.base_obj_weight + max(0.0, 0.90 - gt_mean) * 3.5)
            if gt_mean is not None else self.base_obj_weight
        )
        if abs(_new_obj_w - self.obj_weight) > 0.1:
            logger.info(
                "[FOMO] obj_weight %.3f → %.3f (gt_mean=%.4f, epoch=%d)",
                self.obj_weight, _new_obj_w, gt_mean or 0.0, epoch + 1,
            )
            # v2 only partially co-scales bg_weight so center-confidence boosts
            # are not cancelled out by an equally aggressive rise in background
            # suppression during fine-tuning.
            _bg_before_obj_scale = self.bg_weight
            _obj_scale = _new_obj_w / max(self.obj_weight, 1e-6)
            _scaled_bg = _fomo_bg_obj_coscale(
                current_bg_weight=self.bg_weight,
                obj_scale=_obj_scale,
                fomo_version=self.fomo_version,
                clamp_lo=self._CLAMP_LO,
                clamp_hi=self._CLAMP_HI_LOCAL,
            )
            if abs(_scaled_bg - self.bg_weight) / (self.bg_weight + 1e-9) >= self._MIN_DELTA_FRAC:
                logger.info("[FOMO] bg_weight co-scaled %.4f → %.4f (obj_scale=%.3f)",
                            self.bg_weight, _scaled_bg, _obj_scale)
                self.bg_weight = _scaled_bg
            if _hold_bg_for_center_ranking and self.bg_weight > _bg_before_obj_scale:
                logger.info(
                    "[FOMO v2] Reverting bg_weight co-scale while center ranking remains the main bottleneck."
                )
                self.bg_weight = _bg_before_obj_scale
            self.obj_weight = _new_obj_w
            self._recompile_needed = True

        if (bg_mean is not None and nonbg is not None and gt_mean is not None 
            and bg_mean < 0.06 and nonbg < 0.70 and gt_mean > 0.60):
            logger.info(
                "[FOMO] bg_adapt stop — stable regime: bg_mean=%.4f nonbg=%.4f",
                bg_mean, nonbg,
            )
            self._adapt_steps = self.adapt_max_steps
            return

        new_weight, rule = _adapt_fomo_bg_weight(
            current=self.bg_weight,
            gt_mean=gt_mean,
            bg_mean=bg_mean,
            per_sample_max_nonbg_mean=nonbg,
            fomo_version=self.fomo_version,
        )
        if _hold_bg_for_center_ranking and new_weight > self.bg_weight:
            logger.info(
                "[FOMO v2] Skipping bg_weight increase (rule=%s) because background "
                "is already controlled and the remaining bottleneck is center placement.",
                rule,
            )
            # Avoid a deadlock where v2 keeps holding increases but then also
            # does nothing because the remaining relax path is too small to
            # clear the generic delta guard. When BG is already quiet, gently
            # unwind suppression so center-ranking fixes can take effect.
            if bg_mean is not None and bg_mean < 0.05:
                new_weight = self.bg_weight * 0.92
                rule = "v2_center_rank_relax"
            else:
                return
        if rule in ("skip_missing", "stable"):
            logger.info(
                "[FOMO] bg_weight unchanged at epoch %d (rule=%s)", epoch + 1, rule,
            )
            return

        new_weight = float(np.clip(new_weight, self._CLAMP_LO, self._CLAMP_HI_LOCAL))
        _delta_frac = abs(new_weight - self.bg_weight) / (self.bg_weight + 1e-9)
        if (
            rule != "v2_center_rank_relax"
            and _delta_frac < self._MIN_DELTA_FRAC
        ):
            logger.info(
                "[FOMO] bg_weight delta <5%% — skipping (%.4f → %.4f)",
                self.bg_weight, new_weight,
            )
            return

        logger.info(
            "[FOMO] bg_weight adjusted %.4f → %.4f "
            "(bg_mean=%.4f, nonbg=%.4f, rule=%s, step=%d/%d)",
            self.bg_weight, new_weight,
            bg_mean or 0.0, nonbg or 0.0,
            rule, self._adapt_steps + 1, self.adapt_max_steps,
        )
        self.bg_weight   = new_weight
        self._adapt_steps += 1
        self._recompile_needed = True
        logger.info(
            "[FOMO] bg_weight updated to %.4f — recompile pending between fit() chunks",
            self.bg_weight,
        )


class _FomoLiveLogCallback(tf.keras.callbacks.Callback):
    """Appends one log line per epoch to training_history and commits to DB."""

    def __init__(self, job, db, log_lines: list, epoch_metrics: list, req_epochs: int):
        super().__init__()
        self._job = job
        self._db = db
        self._log_lines = log_lines
        self._epoch_metrics = epoch_metrics
        self._req_epochs = req_epochs

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        loss = logs.get("loss")
        val_loss = logs.get("val_loss")
        val_f1 = logs.get("val_f1")
        line = f"Epoch {epoch + 1}/{self._req_epochs}"
        if loss is not None:
            line += f" - loss: {loss:.4f}"
        if val_loss is not None:
            line += f" - val_loss: {val_loss:.4f}"
        if val_f1 is not None:
            line += f" - val_f1: {val_f1:.4f}"
        self._log_lines.append(line)
        # Capture per-epoch loss for the training graphs. FOMO is a detection
        # architecture, so accuracy fields are omitted (val_f1 is detection F1,
        # not classification accuracy — never surface it as val_accuracy).
        if loss is not None and val_loss is not None:
            self._epoch_metrics.append({
                "epoch":      epoch + 1,
                "train_loss": float(loss),
                "val_loss":   float(val_loss),
            })
        self._job.training_history = _sanitize_json({
            "log_lines": list(self._log_lines),
            "epoch_metrics": list(self._epoch_metrics),
            "is_fomo": True,
        })
        try:
            self._db.commit()
        except Exception:
            self._db.rollback()


class _ClassifierLiveMetricsCallback(tf.keras.callbacks.Callback):
    """
    Persists per-epoch loss/accuracy to training_history.epoch_metrics each
    epoch so the Training Graphs render live and partial runs survive a crash.

    Classification models compute accuracy, so train/val accuracy are included
    when present. Any field absent from the Keras logs is omitted rather than
    nulled, per the epoch_metrics contract.
    """

    def __init__(self, job, db, epoch_metrics: list):
        super().__init__()
        self._job = job
        self._db = db
        self._epoch_metrics = epoch_metrics

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        train_loss = logs.get("loss")
        val_loss = logs.get("val_loss")
        if train_loss is None or val_loss is None:
            return
        entry = {
            "epoch":      epoch + 1,
            "train_loss": float(train_loss),
            "val_loss":   float(val_loss),
        }
        train_acc = logs.get("accuracy")
        val_acc = logs.get("val_accuracy")
        if train_acc is not None:
            entry["train_accuracy"] = float(train_acc)
        if val_acc is not None:
            entry["val_accuracy"] = float(val_acc)
        self._epoch_metrics.append(entry)
        self._job.training_history = _sanitize_json({
            "epoch_metrics": list(self._epoch_metrics),
        })
        try:
            self._db.commit()
        except Exception:
            self._db.rollback()


class _FomoUnfreezeCallback(tf.keras.callbacks.Callback):
    """
    Implements two-phase FOMO transfer learning in a single model.fit() call.

    Phase 1 (epochs 0 … warmup_epochs-1)
    -------------------------------------
    Only fomo_head_conv and fomo_head are trainable.  The pretrained backbone
    is frozen so the randomly-initialised head can learn object-vs-background
    discrimination without immediately overwriting the ImageNet features that
    the backbone starts from.

    Phase 2 (epoch warmup_epochs onward)
    -------------------------------------
    All layers are unfrozen.  The model is recompiled at fine_tune_lr (1/10 of
    the warmup learning rate) so that the backbone adapts slowly — large LR
    here would cause catastrophic forgetting of the pretrained features.

    Why this matters on small datasets
    -----------------------------------
    With ~372 training samples and a random backbone, the head can't learn
    meaningful spatial features no matter how many epochs are run (there is no
    spatial signal in random noise).  With pretrained backbone + head warmup:
    • Epoch 1 already produces object-relevant features at stride-8 cells.
    • The head quickly learns that those features fire at object positions.
    • Fine-tuning then adjusts the backbone for the specific domain (cats/dogs).
    """

    _HEAD_NAMES = frozenset({"fomo_head_conv", "fomo_head"})

    def __init__(self, warmup_epochs: int, loss_fn, fine_tune_lr: float):
        super().__init__()
        self.warmup_epochs = warmup_epochs
        self.loss_fn       = loss_fn
        self.fine_tune_lr  = fine_tune_lr

    def on_epoch_begin(self, epoch: int, logs=None):
        if epoch == self.warmup_epochs:
            for layer in self.model.layers:
                layer.trainable = True
            self.model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=self.fine_tune_lr),
                loss=self.loss_fn,
            )
            logger.info(
                f"[FOMO] Phase 2 starts — backbone unfrozen at epoch {epoch + 1}, "
                f"fine-tuning lr={self.fine_tune_lr:.2e}"
            )


def _fomo_confidence_diagnostics(
    preds: np.ndarray,
    n_classes: int,
    y_heatmap: Optional[np.ndarray] = None,
) -> Dict:
    """
    Summarise the raw prediction confidence distribution of a FOMO model.

    This diagnostic is the first thing to check when threshold=0.5 yields zero
    predictions:

    • max_confidence_per_class — if all values are <<0.5 the object channels
      never fired, almost certainly due to background-dominance during training.
    • predicted_cells_per_threshold — shows at what threshold the model would
      begin making predictions (argmax non-background AND confidence > t).
    • per_sample_max_nonbg_confidence — per-sample max over all non-background
      channels; p50 ~< 0.1 means the model is background-dominated.
    • gt_cell_confidence — when y_heatmap is provided, reports mean/max
      object-channel confidence specifically at GT-occupied cells vs background
      cells.  If gt_mean >> bg_mean the model has spatial signal; if equal,
      the model ignores input content entirely.

    Returns a JSON-safe dict stored in classification_report so the panel
    endpoint can surface it without re-running inference.
    """
    N = preds.shape[0]
    grid_cells = int(preds.shape[1] * preds.shape[2])

    # Apply softmax if the input looks like raw logits (may have negative values).
    # The model head outputs logits; we need probabilities for threshold comparisons.
    def _np_softmax(arr: np.ndarray) -> np.ndarray:
        shifted = arr - arr.max(axis=-1, keepdims=True)
        e = np.exp(shifted)
        return e / (e.sum(axis=-1, keepdims=True) + 1e-12)

    if np.any(preds < -1e-3) or not np.allclose(preds.sum(axis=-1), 1.0, atol=0.05):
        probs = _np_softmax(preds)
    else:
        probs = preds

    # Flatten spatial dims: (N*H*W, n_classes+1)
    flat = probs.reshape(-1, n_classes + 1)

    # Max confidence per object class (ch 0 = background → skip)
    max_per_class: Dict[int, float] = {
        c: float(np.max(flat[:, c + 1])) for c in range(n_classes)
    }

    # Count cells where an object class wins argmax AND its softmax probability
    # exceeds the threshold.  flat is now always probabilities.
    max_obj   = np.max(flat[:, 1:], axis=1)   # (N*H*W,) — best object-class prob
    argmax_ch = np.argmax(flat, axis=1)        # (N*H*W,) — winning channel index

    predicted_cells_per_threshold: Dict[str, int] = {}
    for t in EVAL_THRESHOLDS:
        cnt = int(np.sum((argmax_ch > 0) & (max_obj > t)))
        predicted_cells_per_threshold[str(t)] = cnt

    # Per-sample max non-background confidence (use probs, not raw logits)
    per_sample_max = np.max(probs[:, :, :, 1:], axis=(1, 2, 3))   # (N,)

    result: Dict = {
        "max_confidence_per_class": max_per_class,
        "predicted_cells_per_threshold": predicted_cells_per_threshold,
        "per_sample_max_nonbg_confidence": {
            "min":  float(np.min(per_sample_max)),
            "max":  float(np.max(per_sample_max)),
            "mean": float(np.mean(per_sample_max)),
            "p25":  float(np.percentile(per_sample_max, 25)),
            "p50":  float(np.percentile(per_sample_max, 50)),
            "p75":  float(np.percentile(per_sample_max, 75)),
        },
        "total_eval_samples": N,
        "grid_cells_per_sample": grid_cells,
    }

    # GT-aligned confidence: does the model output higher object confidence
    # at cells that contain a GT object than at background cells?
    # This directly answers whether the model has learned any spatial signal.
    #   gt_mean >> bg_mean  → model has spatial signal, threshold may be fixable
    #   gt_mean ≈ bg_mean   → model ignores input content, data/training issue
    if y_heatmap is not None and y_heatmap.shape == preds.shape:
        try:
            # GT mask: any object class active at this cell (ch 1+)
            gt_obj_mask = np.any(y_heatmap[:, :, :, 1:] > 0.5, axis=-1).reshape(-1)  # (N*H*W,)
            pred_obj_conf = max_obj  # best object-channel confidence per cell

            gt_cells   = pred_obj_conf[gt_obj_mask]
            bg_cells   = pred_obj_conf[~gt_obj_mask]

            result["gt_cell_confidence"] = {
                "gt_occupied_cells":  int(gt_obj_mask.sum()),
                "bg_cells":           int((~gt_obj_mask).sum()),
                "gt_mean_obj_conf":   float(np.mean(gt_cells))   if len(gt_cells) > 0 else None,
                "gt_max_obj_conf":    float(np.max(gt_cells))    if len(gt_cells) > 0 else None,
                "bg_mean_obj_conf":   float(np.mean(bg_cells))   if len(bg_cells) > 0 else None,
                "bg_max_obj_conf":    float(np.max(bg_cells))    if len(bg_cells) > 0 else None,
                # ratio > 1.0 means model scores GT cells higher than background cells
                "gt_vs_bg_mean_ratio": (
                    float(np.mean(gt_cells) / np.mean(bg_cells))
                    if len(gt_cells) > 0 and len(bg_cells) > 0 and np.mean(bg_cells) > 0
                    else None
                ),
            }
        except Exception as _e:
            result["gt_cell_confidence"] = {"error": str(_e)}

    return result


def _fomo_threshold_sweep(
    preds: np.ndarray,
    y_heatmap: np.ndarray,
    label_names: List[str],
    thresholds: Optional[List[float]] = None,
    job_id: str = "",
) -> List[Dict]:
    """
    Compute per-threshold detection metrics for a set of probability thresholds.

    Delegates cell decoding and matching to fomo_evaluator so the sweep
    uses the identical contract as both the main eval loop and model testing.
    """
    if thresholds is None:
        thresholds = EVAL_THRESHOLDS   # unchanged — module-level constant

    n_classes = len(label_names)

    # Pre-build GT cells once using shared helper
    gt_list: List[Dict] = [
        gt_cells_from_heatmap(y_heatmap[i])
        for i in range(len(y_heatmap))
    ]

    sweep: List[Dict] = []
    for t in thresholds:
        _raise_if_job_cancelled(job_id)
        tp = np.zeros(n_classes, dtype=np.int64)
        fp = np.zeros(n_classes, dtype=np.int64)
        fn = np.zeros(n_classes, dtype=np.int64)
        pred_class_counts = np.zeros(n_classes, dtype=np.int64)

        for i in range(len(preds)):
            if i % 20 == 0:
                _raise_if_job_cancelled(job_id)
            pred_cells, _ = decode_fomo_heatmap(preds[i], t, min_peak_gap=0.0)
            for pred_cls in pred_cells.values():
                if 0 <= pred_cls < n_classes:
                    pred_class_counts[pred_cls] += 1
            _tp, _fp, _fn = match_cells(pred_cells, gt_list[i], n_classes)
            tp += _tp; fp += _fp; fn += _fn

        f1s: List[float] = []
        per_class: Dict = {}
        for ci in range(n_classes):
            prec = float(tp[ci]) / (tp[ci] + fp[ci]) if (tp[ci] + fp[ci]) > 0 else 0.0
            rec  = float(tp[ci]) / (tp[ci] + fn[ci]) if (tp[ci] + fn[ci]) > 0 else 0.0
            f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
            f1s.append(round(f1, 4))
            per_class[label_names[ci]] = {
                "tp": int(tp[ci]), "fp": int(fp[ci]), "fn": int(fn[ci]),
                "precision": round(prec, 4),
                "recall":    round(rec,  4),
                "f1":        round(f1,   4),
            }

        macro_f1 = round(sum(f1s) / max(n_classes, 1), 4)
        sweep.append({
            "threshold":             t,
            "macro_f1":              macro_f1,
            "total_tp":              int(sum(tp)),
            "total_fp":              int(sum(fp)),
            "total_fn":              int(sum(fn)),
            "total_predicted_cells": int(sum(tp) + sum(fp)),
            "per_class":             per_class,
            "predicted_class_distribution": {
                label_names[ci]: int(pred_class_counts[ci]) for ci in range(n_classes)
            },
        })

    max_f1    = max(r["macro_f1"] for r in sweep)
    candidates = [r for r in sweep if r["macro_f1"] >= max_f1 - 0.001]
    best = min(candidates, key=lambda x: (x["total_fn"], x["total_fp"], x["threshold"]))
    reported_thresh = max(best["threshold"], 0.3)
    logger.info(
        "FOMO threshold sweep complete — "
        "best macro_F1=%s at threshold=%s "
        "(TP=%s FP=%s FN=%s)",
        best["macro_f1"], reported_thresh,
        best["total_tp"], best["total_fp"], best["total_fn"],
    )
    return sweep


# Diagnostic source: Steps 1, 3, 5 — _evaluate_fomo_detection (end-of-training)
def _evaluate_fomo_detection(
    model,
    X: np.ndarray,
    y_heatmap: np.ndarray,
    label_names: List[str],
    threshold: float = 0.5,   # hint only — overridden by sweep
    bg_weight: Optional[float] = None,
    fomo_version: int = 1,
    job_id: str = "",
    n_background: int = 0,
) -> Dict:
    """
    Evaluate a trained FOMO model using the shared fomo_evaluator contract.

    Algorithm (per sample):
      Ground truth:  gt_cells_from_heatmap() — argmax>0 cells, class = ch-1.
      Prediction:    decode_fomo_heatmap()   — conditional softmax + argmax.
      Matching:      match_cells()           — exact (row, col, class) triple.

    Threshold: resolved via resolve_fomo_threshold(TRAINING_PARITY) using the
    threshold sweep best value stored in the sweep result.  The `threshold`
    parameter serves as the initial candidate fed into the sweep.

    Returns a dict in sklearn classification_report(output_dict=True) format.
    Extra keys: fomo_detection_eval, fomo_eval_status, fomo_eval_error,
                threshold, confidence_diagnostics, threshold_sweep.
    """
    # ── Degraded-state shell ────────────────────────────────────────────────
    def _degraded(status: str, error: str) -> Dict:
        shell: Dict = {
            name: {"precision": None, "recall": None, "f1-score": None,
                   "support": None, "tp": None, "fp": None, "fn": None}
            for name in label_names
        }
        shell["macro avg"]           = {"precision": None, "recall": None, "f1-score": None, "support": 0}
        shell["weighted avg"]        = {"precision": None, "recall": None, "f1-score": None, "support": 0}
        shell["fomo_detection_eval"] = False
        shell["fomo_eval_status"]    = status
        shell["fomo_eval_error"]     = error
        shell["threshold"]           = threshold
        return shell

    # ── Guard: no data ──────────────────────────────────────────────────────
    if len(X) == 0 or len(y_heatmap) == 0:
        logger.warning("FOMO evaluation skipped: no evaluation samples available.")
        return _degraded("no_data", "No evaluation samples available.")

    # ── GT pre-check ────────────────────────────────────────────────────────
    gt_list = [gt_cells_from_heatmap(y_heatmap[i]) for i in range(len(y_heatmap))]
    total_gt_cells      = sum(len(g) for g in gt_list)
    samples_with_objects = sum(1 for g in gt_list if g)
    logger.info(
        "FOMO evaluation pre-check: %d eval samples, "
        "%d samples with ≥1 GT object cell, "
        "%d total GT occupied cells, initial_threshold=%.4f",
        len(X), samples_with_objects, total_gt_cells, threshold,
    )
    # All-background GT used to be treated as "no data".  That is right when the
    # split has no negatives either (it means the label_map lost its keys and
    # every heatmap came out empty by accident) and wrong when the user has
    # deliberately built an all-negative split to measure the false-positive
    # rate — there is genuinely nothing to detect, and whether the model fires
    # anyway is exactly the question being asked.
    #
    # `n_background` counts the images the caller marked as negatives; with a
    # non-zero count the run continues and reports background_fp_rate, with
    # precision/recall left at their vacuous values.
    if total_gt_cells == 0 and not n_background:
        logger.warning(
            "FOMO evaluation: ALL ground-truth heatmaps are background-only. "
            "Likely cause: label_map missing UUID or name keys. Retrain after fix."
        )
        return _degraded(
            "no_data",
            f"All {len(X)} evaluation heatmaps are background-only — no GT objects found.",
        )
    if total_gt_cells == 0:
        logger.info(
            "FOMO evaluation: all %d evaluation images are background — "
            "precision/recall are undefined, reporting background_fp_rate only.",
            len(X),
        )

    # ── Inference ───────────────────────────────────────────────────────────
    try:
        raw_preds = model.predict(X, verbose=0)
        # decode_fomo_heatmap applies softmax conditionally — do NOT softmax here.
        # Convert to numpy float32 array so decode can use it per-sample.
        preds = np.asarray(raw_preds, dtype=np.float32)
    except _CancelledError:
        raise
    except Exception as exc:
        logger.warning("FOMO evaluation: model.predict failed — %s.", exc)
        return _degraded("inference_failed", f"Model inference failed: {exc}")

    _raise_if_job_cancelled(job_id)

    # ── Grid-shape guard ────────────────────────────────────────────────────
    try:
        assert_grid_shape_match(preds.shape, y_heatmap.shape, context="training_worker")
    except ValueError as exc:
        logger.warning(str(exc))
        return _degraded("inference_failed", str(exc))

    # ── Confidence diagnostics (training-only, non-fatal) ──────────────────
    confidence_diagnostics: Optional[Dict] = None
    try:
        confidence_diagnostics = _fomo_confidence_diagnostics(preds, len(label_names), y_heatmap=y_heatmap)
        diag = confidence_diagnostics
        _gt_diag = diag.get("gt_cell_confidence", {})
        _gt_ratio = _gt_diag.get("gt_vs_bg_mean_ratio")
        _gt_mean  = _gt_diag.get("gt_mean_obj_conf") or 0.0
        _bg_mean  = _gt_diag.get("bg_mean_obj_conf") or 0.0
        logger.info(
            "FOMO confidence diagnostics — "
            "max_per_class=%s, predicted_cells=%s, "
            "per_sample_max_nonbg: min=%.4f mean=%.4f max=%.4f p50=%.4f | "
            "GT-cell obj_conf: gt_mean=%.4f bg_mean=%.4f ratio=%s",
            diag["max_confidence_per_class"],
            diag["predicted_cells_per_threshold"],
            diag["per_sample_max_nonbg_confidence"]["min"],
            diag["per_sample_max_nonbg_confidence"]["mean"],
            diag["per_sample_max_nonbg_confidence"]["max"],
            diag["per_sample_max_nonbg_confidence"]["p50"],
            _gt_mean,
            _bg_mean,
            f"{_gt_ratio:.3f}" if _gt_ratio is not None else "n/a",
        )
        max_any_class = max(diag["max_confidence_per_class"].values()) if diag["max_confidence_per_class"] else 0.0
        if max_any_class < threshold:
            logger.warning(
                "FOMO diagnostics: max object-class confidence across ALL eval cells "
                "is %.4f, which is below threshold=%.4f. "
                "Zero predictions will be made at this threshold. "
                "Root causes: insufficient training data, insufficient epochs, "
                "or label/bounding-box noise. See threshold sweep for guidance.",
                max_any_class, threshold,
            )
        # Overprediction-collapse check: GT and background confidence are indistinguishable.
        if _gt_ratio is not None and _gt_ratio < 1.1:
            _total_cells = preds.shape[0] * preds.shape[1] * preds.shape[2] if preds.ndim == 4 else 1
            _pred_high   = diag["predicted_cells_per_threshold"].get("0.1", _total_cells)
            logger.warning(
                "FOMO collapse detected: gt_vs_bg_mean_ratio=%.3f (expected >>1.0). "
                "Model fires object logits broadly (predicted_cells@0.1=%d). "
                "Check bg_weight (current=%s), spread labels, or augmentation.",
                _gt_ratio,
                _pred_high,
                f"{bg_weight:.4f}" if bg_weight is not None else "unknown",
            )
    except _CancelledError:
        raise
    except Exception as _diag_exc:
        logger.warning("FOMO confidence diagnostics failed (non-fatal): %s", _diag_exc)

    # ── Threshold sweep ─────────────────────────────────────────────────────
    _raise_if_job_cancelled(job_id)
    threshold_sweep: List[Dict] = []
    _selected_sweep_entry: Optional[Dict] = None
    try:
        threshold_sweep = _fomo_threshold_sweep(preds, y_heatmap, label_names, job_id=job_id)
    except _CancelledError:
        raise
    except Exception as _sweep_exc:
        logger.warning("FOMO threshold sweep failed (non-fatal): %s", _sweep_exc)

    # Resolve canonical threshold via shared resolver.
    # Build a minimal classification_report stub with the sweep best value.
    if threshold_sweep:
        _max_f1 = max(r["macro_f1"] for r in threshold_sweep)
        _cands = [r for r in threshold_sweep if r["macro_f1"] >= _max_f1 - 0.001]
        _best = min(_cands, key=lambda x: (x["total_fn"], x["total_fp"], x["threshold"]))
        _selected_sweep_entry = _best
        _sweep_best = max(_best["threshold"], 0.3)
        # Wrap as a training_classification_report so resolve_fomo_threshold
        # can enforce its contract (finite, in range).
        _stub_cr = {"threshold": _sweep_best}
        threshold, threshold_source = resolve_fomo_threshold(
            mode=ThresholdSource.TRAINING_PARITY,
            training_classification_report=_stub_cr,
        )
    else:
        # No sweep — use the caller-supplied hint but validate it.
        from app.ml.fomo_evaluator import _validate_threshold
        try:
            _validate_threshold(threshold)
        except ValueError as _te:
            return _degraded("inference_failed", str(_te))
        threshold_source = "caller-supplied hint (no sweep available)"

    # ── Main evaluation loop ────────────────────────────────────────────────
    _raise_if_job_cancelled(job_id)
    n_classes = len(label_names)
    tp_total = np.zeros(n_classes, dtype=np.int64)
    fp_total = np.zeros(n_classes, dtype=np.int64)
    fn_total = np.zeros(n_classes, dtype=np.int64)

    sample_gt_cells_dbg: List = []
    sample_pred_cells_dbg: List = []

    for i in range(len(X)):
        pred_cells, _ = decode_fomo_heatmap(preds[i], threshold, min_peak_gap=0.0)
        gt_cells      = gt_list[i]
        _tp, _fp, _fn = match_cells(pred_cells, gt_cells, n_classes)
        tp_total += _tp; fp_total += _fp; fn_total += _fn
        if i < 3:
            sample_gt_cells_dbg.append(gt_cells)
            sample_pred_cells_dbg.append(pred_cells)

    # ── Build result dict (classification_report-compatible) ───────────────
    agg = aggregate_metrics(tp_total, fp_total, fn_total, label_names)
    result: Dict = {}
    for name in label_names:
        result[name] = agg["per_class"][name]
    result["macro avg"]    = agg["macro_avg"]
    result["weighted avg"] = agg["weighted_avg"]

    total_pred_cells = int(tp_total.sum() + fp_total.sum())
    _n_eval_samples  = len(X)
    avg_pred_cells   = total_pred_cells / _n_eval_samples if _n_eval_samples > 0 else 0.0
    avg_gt_objects   = total_gt_cells / _n_eval_samples if _n_eval_samples > 0 else 0.0

    # ── Shared diagnostics log ──────────────────────────────────────────────
    log_fomo_run_diagnostics(
        threshold=threshold,
        threshold_source=threshold_source,
        total_gt_cells=total_gt_cells,
        total_pred_cells=total_pred_cells,
        tp=tp_total,
        fp=fp_total,
        fn=fn_total,
        label_names=label_names,
        n_unresolved_gt=0,   # training heatmaps never have unresolved labels
        sample_gt_cells=sample_gt_cells_dbg,
        sample_pred_cells=sample_pred_cells_dbg,
        context="training_worker",
    )
    logger.info(
        "[FOMO v%d] eval: avg_pred_cells_per_image=%.2f  avg_gt_objects_per_image=%.2f  total_pred=%d  total_gt=%d  n_samples=%d",
        fomo_version, avg_pred_cells, avg_gt_objects, total_pred_cells, total_gt_cells, _n_eval_samples,
    )
    if avg_gt_objects > 0.0 and avg_pred_cells >= max(avg_gt_objects * 3.0, avg_gt_objects + 3.0):
        logger.warning(
            "[FOMO v%d] validation overprediction: avg_pred_cells_per_image=%.2f vs avg_gt_objects_per_image=%.2f. "
            "False positives are likely dominating validation.",
            fomo_version, avg_pred_cells, avg_gt_objects,
        )
    if fomo_version == 2:
        _diag_probs = _fomo_probs_for_logging(preds)
        log_fomo_v2_diagnostics(
            fomo_version=fomo_version,
            feature_tap="validation",
            grid_h=y_heatmap.shape[1],
            grid_w=y_heatmap.shape[2],
            spread_mode=bool(_fomo_training_target_config(fomo_version)["spread"]),
            bg_weight=float(bg_weight) if bg_weight is not None else 0.0,
            avg_predicted_cells_per_image=avg_pred_cells,
            avg_gt_objects_per_image=avg_gt_objects,
            preds_probs=_diag_probs,
            y_heatmap=y_heatmap,
            n_classes=n_classes,
            per_class_metrics=(_selected_sweep_entry or {}).get("per_class", {}),
            predicted_class_distribution=(_selected_sweep_entry or {}).get("predicted_class_distribution", {}),
            context="validation",
        )

    # Threshold-sweep zero-prediction warning
    if total_pred_cells == 0 and total_gt_cells > 0 and threshold_sweep:
        _w_max_f1  = max(s["macro_f1"] for s in threshold_sweep)
        _w_cands   = [s for s in threshold_sweep if s["macro_f1"] >= _w_max_f1 - 0.001]
        best_entry = min(_w_cands, key=lambda x: x["threshold"])
        if best_entry["total_predicted_cells"] > 0:
            logger.warning(
                "FOMO evaluation: %d GT cells found but ZERO predictions above "
                "threshold=%.4f. Sweep found best macro_F1=%.4f at threshold=%s.",
                total_gt_cells, threshold,
                best_entry["macro_f1"], best_entry["threshold"],
            )
        else:
            logger.warning(
                "FOMO evaluation: %d GT cells found but ZERO predictions at ANY "
                "threshold %s. Model is background-dominated — more data/epochs needed.",
                total_gt_cells, [s["threshold"] for s in threshold_sweep],
            )

    # ── Detection-style confusion matrix ───────────────────────────────────
    grid_h, grid_w = y_heatmap.shape[1], y_heatmap.shape[2]
    total_cells   = len(X) * grid_h * grid_w
    total_bg_gt   = total_cells - total_gt_cells
    total_fp_all  = int(fp_total.sum())
    _n = n_classes + 1
    _cm: list = [[0] * _n for _ in range(_n)]
    _cm[0][0] = max(0, total_bg_gt - total_fp_all)
    for _j in range(n_classes):
        _cm[0][_j + 1]       = int(fp_total[_j])
        _cm[_j + 1][0]       = int(fn_total[_j])
        _cm[_j + 1][_j + 1]  = int(tp_total[_j])

    result["confusion_matrix"]            = _cm
    result["fomo_detection_eval"]         = True
    result["fomo_eval_status"]            = "success"
    result["fomo_eval_error"]             = None
    result["threshold"]                   = threshold
    result["confidence_diagnostics"]      = confidence_diagnostics
    result["threshold_sweep"]             = threshold_sweep
    result["fomo_version"]                = fomo_version
    result["avg_pred_cells_per_image"]    = round(avg_pred_cells, 2)
    result["avg_gt_objects_per_image"]    = round(avg_gt_objects, 2)

    # ── Background false-positive rate ──────────────────────────────────────
    # Per-class precision/recall are near-blind to negatives: an image with no
    # GT cells contributes only to the shared FP tally, where it is
    # indistinguishable from a mislocalised detection on a positive image.
    # Counted over images whose GT heatmap is empty, which is exactly the set
    # of background images once the loaders stop inventing GT for them.
    _bg_idx = [i for i in range(len(X)) if not gt_list[i]]
    if _bg_idx:
        _bg_fp_cells = 0
        _bg_imgs_with_fp = 0
        for i in _bg_idx:
            _pc, _ = decode_fomo_heatmap(preds[i], threshold, min_peak_gap=0.0)
            _n_fp = sum(1 for c in _pc.values() if 0 <= c < n_classes)
            _bg_fp_cells += _n_fp
            if _n_fp:
                _bg_imgs_with_fp += 1
        result["background_images"]         = len(_bg_idx)
        result["background_fp_rate"]        = round(_bg_fp_cells / len(_bg_idx), 4)
        result["background_images_with_fp"] = _bg_imgs_with_fp
        result["background_fp_threshold"]   = round(float(threshold), 4)
        logger.info(
            "[FOMO v%d] background FP rate: %d predicted cells over %d background "
            "images = %.3f per image (%d image(s) fired at all) at threshold=%.4f",
            fomo_version, _bg_fp_cells, len(_bg_idx),
            result["background_fp_rate"], _bg_imgs_with_fp, threshold,
        )
    else:
        result["background_images"]         = 0
        result["background_fp_rate"]        = None
        result["background_images_with_fp"] = 0
        result["background_fp_threshold"]   = round(float(threshold), 4)
    return result


def _create_fomo_heatmap(
    boxes, grid_w, grid_h, img_w, img_h, num_classes, label_map,
    spread: bool = False,
    target_mode: str = "strict_center",
    center_value: float = 1.0,
    neighbor_value: float = 0.0,
    diagonal_value: float = 0.0,
):
    """
    Convert bounding boxes into a FOMO-style heatmap.
    Grid size is typically image_size / 8.
    Channel 0 = Background, 1..N = Classes.

    Coordinate convention — boxes may arrive in two formats:
      • Normalized [0, 1]:  produced by normalize_bounding_boxes / DSP cache.
        Grid cell = int(cx * grid_w).  Matches gt_cells_from_boxes in testing.
      • Pixel-space [0, img_w]:  fallback centre-boxes and some raw-DB sources.
        Grid cell = int((cx / img_w) * grid_w).
    The format is detected per-box via max(|x|, |y|, |w|, |h|) > 1.0, mirroring
    normalize_bounding_boxes.  Passing pixel-space boxes with img_w=1 would
    silently collapse all cells to (0, 0) — this guard catches that regression.

    label_map must include BOTH UUID keys and name-string keys so that boxes
    saved by the AI-labeling pipeline (label: "name") and boxes saved by other
    paths (label_id: "<uuid>") are handled uniformly.

    target_mode="soft_center_falloff" is a conservative v2-only option that
    keeps the GT center as the strongest target and adds a small soft ring to
    nearby cells. Evaluation still uses strict one-cell targets.

    spread=True is a legacy/experimental option that expands a GT cell to the
    4 orthogonal neighbours using soft labels. It is no longer the default for
    v2 training because strict center-cell targets match evaluation better and
    reduce false-positive pressure.

    Why 0.6 instead of 0.5:
      The loss function treats cells as "has_obj" only when any non-background
      channel > 0.5.  With soft=0.5 the condition is False, so spread cells
      fall through to bg_weight treatment (0.01) and contribute almost zero
      object gradient — the footprint widening is wasted.  With soft=0.6
      those 4 cells ARE counted as object cells (0.6 > 0.5) and receive full
      per-class weight.  This increases the effective object training signal
      per GT instance from 1 cell to ~5 cells without changing the bg_weight
      or altering background suppression.  In gradient terms:
        Before: 1 × class_weight  per GT object
        After:  1 × 1.0 (center, hard) + 4 × 0.6 (neighbours, soft)
              ≈ 3.4× more object gradient per instance
      The evaluation path always uses spread=False (hard one-cell GT).
    """
    # Initialize with background (channel 0) as 1.0, others 0.0
    heatmap = np.zeros((grid_h, grid_w, num_classes + 1), dtype=np.float32)
    heatmap[:, :, 0] = 1.0

    skipped_keys: list = []
    placed = 0

    def _box_center_to_grid(cx, cy, box, img_w, img_h, grid_w, grid_h):
        """Map a box center to grid indices using a single, consistent normalization
        check applied once per box. Uses int() (floor) which correctly matches the
        grid's own cell partitioning: cell k owns [k/grid_w, (k+1)/grid_w)."""
        is_pixel = max(
            abs(float(box.get("x", 0))), abs(float(box.get("y", 0))),
            abs(float(box.get("w", 0))), abs(float(box.get("h", 0))),
        ) > 1.0
        if is_pixel:
            gx = int((cx / img_w) * grid_w)
            gy = int((cy / img_h) * grid_h)
        else:
            gx = int(cx * grid_w)
            gy = int(cy * grid_h)
        gx = max(0, min(gx, grid_w - 1))
        gy = max(0, min(gy, grid_h - 1))
        return gx, gy

    for box in boxes:
        # Accept either field name; the dual-key label_map resolves both.
        label_key = box.get("label_id") or box.get("label")
        if label_key not in label_map:
            skipped_keys.append(label_key)
            continue

        # Determine center point
        cx = box["x"] + (box["w"] / 2)
        cy = box["y"] + (box["h"] / 2)

        gx, gy = _box_center_to_grid(cx, cy, box, img_w, img_h, grid_w, grid_h)

        # Set class channel (0 = background, 1..N = classes)
        class_idx = label_map[label_key] + 1
        heatmap[gy, gx, class_idx] = float(center_value)
        heatmap[gy, gx, 0] = max(0.0, 1.0 - float(center_value))
        placed += 1

        if target_mode in {"soft_center_falloff", "near_strict_center"}:
            for dy, dx, soft_val in [
                (-1, 0, neighbor_value), (1, 0, neighbor_value),
                (0, -1, neighbor_value), (0, 1, neighbor_value),
                (-1, -1, diagonal_value), (-1, 1, diagonal_value),
                (1, -1, diagonal_value), (1, 1, diagonal_value),
            ]:
                if soft_val <= 0.0:
                    continue
                ny, nx = gy + dy, gx + dx
                if 0 <= ny < grid_h and 0 <= nx < grid_w and heatmap[ny, nx, 0] == 1.0:
                    heatmap[ny, nx, class_idx] = float(soft_val)
                    heatmap[ny, nx, 0] = max(0.0, 1.0 - float(soft_val))

        # Soft label spread to orthogonal neighbours (training only).
        # Only updates cells that are still pure background — does not
        # overwrite hard labels placed by other boxes.
        # soft=0.6 so the loss function's has_obj check (> 0.5) treats
        # these cells as object cells, giving them full per-class gradient
        # weight instead of the background-suppression weight (0.01).
        if spread:
            for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ny, nx = gy + dy, gx + dx
                if 0 <= ny < grid_h and 0 <= nx < grid_w:
                    if heatmap[ny, nx, 0] == 1.0:   # still background
                        heatmap[ny, nx, class_idx] = 0.6  # soft object label — > 0.5 so has_obj gate fires → full class weight
                        heatmap[ny, nx, 0]         = 0.4  # partial background

    if skipped_keys:
        # Separate None keys (box has no label at all) from genuinely unknown
        # label strings (label present but not in the map).
        none_count    = skipped_keys.count(None)
        unknown_keys  = [k for k in skipped_keys if k is not None]
        unknown_sample = list(dict.fromkeys(unknown_keys))[:5]

        if none_count:
            # Boxes with no label field should have been filtered before reaching
            # here.  Log once at WARNING level (not per-box) so the root cause is
            # visible without flooding the log every epoch.
            logger.warning(
                "_create_fomo_heatmap: %d/%d box(es) had label_id=None and were "
                "skipped. This means the box normalization step did not resolve a "
                "label for these boxes. Check that extra_metadata[boundingBoxes] "
                "entries carry a 'label_id' (UUID) or 'label' (name-string) field.",
                none_count, len(boxes),
            )
        if unknown_keys:
            logger.warning(
                "_create_fomo_heatmap: skipped %d/%d boxes because their label key "
                "was not found in label_map. "
                "Sample missing keys: %s. "
                "label_map keys (first 5): %s. "
                "Ensure label_map includes both UUID and name-string keys.",
                len(unknown_keys), len(boxes),
                unknown_sample,
                list(label_map.keys())[:5],
            )

    obj_support = int(np.sum(heatmap[:, :, 1:] > 0.5))
    logger.debug(
        f"_create_fomo_heatmap: placed={placed}, skipped={len(skipped_keys)}, "
        f"grid={grid_h}×{grid_w}, img={img_w}×{img_h}, spread={spread}, "
        f"obj_support_cells={obj_support}"
    )
    return heatmap


# ─── Augmentation primitives ──────────────────────────────────────────────────

try:
    from scipy.ndimage import rotate as _scipy_rotate, zoom as _scipy_zoom
    _SCIPY_ROTATE_AVAILABLE = True
except ImportError:  # pragma: no cover
    _SCIPY_ROTATE_AVAILABLE = False
    _scipy_zoom = None
    logger.warning(
        "[FOMO] scipy.ndimage not available — spatial augmentation disabled. "
        "Install scipy to enable: pip install scipy"
    )


def _boxes_are_pixel_space(boxes: list) -> bool:
    """Return True if any box coordinate exceeds 1.0 (pixel space)."""
    for b in boxes:
        if max(abs(b.get("x", 0)), abs(b.get("y", 0)),
               abs(b.get("w", 0)), abs(b.get("h", 0))) > 1.0:
            return True
    return False


def _clamp_boxes(boxes: list, img_w: float, img_h: float) -> list:
    """
    Clamp each box so its centre stays within the image bounds.
    Handles both pixel-space and normalized (0-1) boxes.
    """
    if not boxes:
        return []
    pixel = _boxes_are_pixel_space(boxes)
    max_x = float(img_w) if pixel else 1.0
    max_y = float(img_h) if pixel else 1.0
    clamped = []
    for b in boxes:
        fb = dict(b)
        raw_cx = b["x"] + b["w"] / 2
        raw_cy = b["y"] + b["h"] / 2
        cx = max(0.0, min(max_x, raw_cx))
        cy = max(0.0, min(max_y, raw_cy))
        fb["x"] = cx - b["w"] / 2
        fb["y"] = cy - b["h"] / 2
        clamped.append(fb)
    return clamped


def _aug_hflip(
    features: np.ndarray, boxes: list, img_w: float
) -> tuple:
    """Horizontal flip — mirrors image; reflects box x-coords.
    Handles both pixel-space and normalized (0-1) boxes.
    """
    if features.ndim == 3:
        aug = features[:, ::-1, :].copy()
    elif features.ndim == 2:
        aug = features[:, ::-1].copy()
    else:
        aug = features  # flat array: no spatial flip
    pixel = _boxes_are_pixel_space(boxes)
    scale = float(img_w) if pixel else 1.0
    aug_boxes = []
    for b in boxes:
        fb = dict(b)
        fb["x"] = max(0.0, scale - b["x"] - b["w"])
        aug_boxes.append(fb)
    return aug, aug_boxes


def _aug_vflip(
    features: np.ndarray,
    boxes: list,
    img_h: float,
) -> tuple:
    """Vertical flip — mirrors image; reflects box y-coords."""
    if features.ndim == 3:
        aug = features[::-1, :, :].copy()
    elif features.ndim == 2:
        aug = features[::-1, :].copy()
    else:
        aug = features
    pixel = _boxes_are_pixel_space(boxes)
    scale = float(img_h) if pixel else 1.0
    aug_boxes = []
    for b in boxes:
        fb = dict(b)
        fb["y"] = max(0.0, scale - b["y"] - b["h"])
        aug_boxes.append(fb)
    return aug, aug_boxes


def _aug_translate(
    features: np.ndarray,
    boxes: list,
    img_w: float,
    img_h: float,
    rng: np.random.Generator,
    max_shift: float = 0.10,
) -> tuple:
    """Random translation ±max_shift × image size.

    Shifts image and box coords; out-of-bounds boxes are clamped.
    Combats spatial memorisation — same object appears at different
    grid positions across augmented copies.
    """
    H, W = features.shape[0], features.shape[1]
    pixel = _boxes_are_pixel_space(boxes)
    ref_w = float(img_w) if pixel else 1.0
    ref_h = float(img_h) if pixel else 1.0

    dx = float(rng.uniform(-max_shift, max_shift)) * ref_w
    dy = float(rng.uniform(-max_shift, max_shift)) * ref_h
    px = int(round(dx / ref_w * W))
    py = int(round(dy / ref_h * H))

    canvas = np.zeros_like(features)
    src_x0, src_x1 = max(0, -px), min(W, W - px)
    dst_x0, dst_x1 = max(0,  px), min(W, W + px)
    src_y0, src_y1 = max(0, -py), min(H, H - py)
    dst_y0, dst_y1 = max(0,  py), min(H, H + py)
    if features.ndim == 3:
        canvas[dst_y0:dst_y1, dst_x0:dst_x1] = features[src_y0:src_y1, src_x0:src_x1]
    else:
        canvas[dst_y0:dst_y1, dst_x0:dst_x1] = features[src_y0:src_y1, src_x0:src_x1]

    aug_boxes = []
    for b in boxes:
        fb = dict(b)
        fb["x"] = b["x"] + dx
        fb["y"] = b["y"] + dy
        aug_boxes.append(fb)
    return canvas.astype(features.dtype), _clamp_boxes(aug_boxes, img_w, img_h)


def _aug_rotate_clamped(
    features: np.ndarray,
    boxes: list,
    img_w: float,
    img_h: float,
    angle_deg: float,
) -> tuple:
    """
    Rotate image by angle_deg (positive = CCW in screen coordinates) and
    transform each box centre through the same rotation, then **clamp** any
    centre that drifted outside the image back to the nearest valid position.

    Clamping (not discarding) ensures every augmented copy carries valid
    bounding-box targets, matching the tolerance already applied by
    _create_fomo_heatmap.

    Rotation is capped at ±5° — small enough that it cannot push a centred
    object more than ~4 % of the image width outside the boundary, yet still
    provides useful viewpoint variation for training.

    Falls back to identity transform if scipy is unavailable or features are
    not at least 2-D.
    """
    if not _SCIPY_ROTATE_AVAILABLE or features.ndim < 2:
        return features, list(boxes)

    rotated = _scipy_rotate(
        features, angle_deg, axes=(0, 1),
        reshape=False, order=1, cval=0.0, mode="constant",
    )
    rotated = np.clip(rotated, 0.0, 1.0).astype(features.dtype)

    # CCW-in-screen rotation of each box centre around the image midpoint.
    # Use 0.5 as midpoint for normalized boxes, img/2 for pixel-space boxes.
    cos_a = np.cos(np.deg2rad(angle_deg))
    sin_a = np.sin(np.deg2rad(angle_deg))
    pixel = _boxes_are_pixel_space(boxes)
    hw = img_w / 2.0 if pixel else 0.5
    hh = img_h / 2.0 if pixel else 0.5
    rotated_boxes = []
    for b in boxes:
        cx_c = (b["x"] + b["w"] / 2) - hw
        cy_c = (b["y"] + b["h"] / 2) - hh
        new_cx = cx_c * cos_a - cy_c * sin_a + hw
        new_cy = cx_c * sin_a + cy_c * cos_a + hh
        fb = dict(b)
        fb["x"] = new_cx - b["w"] / 2
        fb["y"] = new_cy - b["h"] / 2
        rotated_boxes.append(fb)

    # Clamp centres back into the image (matches heatmap builder convention).
    aug_boxes = _clamp_boxes(rotated_boxes, img_w, img_h)
    return rotated, aug_boxes


def _aug_color_jitter(
    features: np.ndarray,
    rng: np.random.Generator,
    brightness: tuple = (0.85, 1.15),
    contrast: tuple = (0.90, 1.10),
    color_strength: float = 0.05,
) -> np.ndarray:
    """
    Pixel-only colour jitter — boxes are **never** modified, so these
    transforms are unconditionally valid and never require discarding.

    Applies (in order):
      1. Brightness: multiplicative uniform scale.
      2. Contrast: scale around per-channel spatial mean.
      3. Per-channel colour scale (RGB only, mild).

    Ranges are conservative by default (±15 % brightness, ±10 % contrast,
    ±5 % per-channel colour) to avoid introducing unrealistic colours at the
    small 48 × 48 resolution typical of FOMO impulses.
    """
    aug = features.astype(np.float32)

    # 1. Brightness
    aug = aug * float(rng.uniform(*brightness))

    # 2. Contrast (scale around per-channel / global mean)
    c_scale = float(rng.uniform(*contrast))
    if c_scale != 1.0:
        mean = aug.mean(axis=(0, 1), keepdims=True) if aug.ndim == 3 else aug.mean()
        aug = mean + (aug - mean) * c_scale

    # 3. Per-channel colour jitter (RGB images only)
    if color_strength > 0.0 and aug.ndim == 3 and aug.shape[2] == 3:
        ch = rng.uniform(1.0 - color_strength, 1.0 + color_strength, size=3)
        aug = aug * ch[np.newaxis, np.newaxis, :]

    return np.clip(aug, 0.0, 1.0).astype(features.dtype)


def _aug_scale_jitter(
    features: np.ndarray,
    boxes: list,
    img_w: float,
    img_h: float,
    scale: float,
) -> tuple:
    """Center-aligned scale jitter (0.9–1.1). Labels transform identically; no crop-shift."""
    if _scipy_zoom is None or features.ndim < 2:
        return features, list(boxes)

    H, W = features.shape[0], features.shape[1]
    zoom_f = (scale, scale, 1.0) if features.ndim == 3 else (scale, scale)
    try:
        scaled = _scipy_zoom(features, zoom_f, order=1, mode="constant", cval=0.0)
    except Exception:
        return features, list(boxes)

    sH, sW = scaled.shape[0], scaled.shape[1]
    if scale > 1.0:
        r0, c0 = (sH - H) // 2, (sW - W) // 2
        aug = scaled[r0:r0 + H, c0:c0 + W, :] if features.ndim == 3 else scaled[r0:r0 + H, c0:c0 + W]
    else:
        canvas = np.zeros_like(features)
        r0, c0 = (H - sH) // 2, (W - sW) // 2
        if features.ndim == 3:
            canvas[r0:r0 + sH, c0:c0 + sW, :] = scaled
        else:
            canvas[r0:r0 + sH, c0:c0 + sW] = scaled
        aug = canvas

    aug = np.clip(aug, 0.0, 1.0).astype(features.dtype)

    pixel = _boxes_are_pixel_space(boxes)
    cx_ref = img_w / 2.0 if pixel else 0.5
    cy_ref = img_h / 2.0 if pixel else 0.5
    aug_boxes = []
    for b in boxes:
        fb = dict(b)
        cx = scale * (b["x"] + b["w"] / 2.0 - cx_ref) + cx_ref
        cy = scale * (b["y"] + b["h"] / 2.0 - cy_ref) + cy_ref
        fb["x"] = cx - b["w"] / 2.0
        fb["y"] = cy - b["h"] / 2.0
        aug_boxes.append(fb)
    return aug, _clamp_boxes(aug_boxes, img_w, img_h)


def _augment_fomo_sample(
    features: np.ndarray,
    boxes: list,
    img_w: float,
    img_h: float,
    fomo_version: int = 1,
) -> tuple:
    """
    Generate augmented copies of a FOMO training sample.

    v1 (stride-16): 10 copies, stronger spatial/colour transforms.
      • Rotation ±15°, translation ±6%, scale 0.88–1.12
      • Brightness/contrast ±25%/±20%
    v2 (stride-8): 5 copies, milder transforms to stabilise early learning.
      • Rotation ±8°, translation ±4%, scale 0.92–1.08
      • Brightness/contrast ±15%/±10%
    Both versions: horizontal flip p=0.5, colour jitter ±5% (pixel-only).

    v2 rationale: the FPN fusion head needs a few epochs to learn spatial
    discrimination before aggressive augmentation stops being noise.
    Reducing copies from 10 to 5 also shrinks the training set, which keeps
    epoch time short while base learning is unstable.

    Returns
    -------
    (augmented, stats) : tuple
        augmented — list of (features, boxes) tuples, NOT including original
        stats     — dict with per-type counts: "flip", "brightness", "contrast", "rotation"
    """
    rng = np.random.default_rng()
    augmented: list = []
    stats: dict = {"flip": 0, "translate": 0, "brightness": 0, "contrast": 0, "rotation": 0}

    # Version-specific augmentation budgets.
    _n_copies       = 5    if fomo_version == 2 else 10

    # Background ("negative") samples get their own, much smaller budget.
    # Every training sample is otherwise multiplied 5× (v2) / 10× (v1); applying
    # that to an empty scene produces near-duplicate background cells that carry
    # no localisation signal, floods the set with the easiest examples the model
    # will ever see, and silently pushes the achieved neg:pos ratio far past
    # whatever cap was requested (the cap is applied after augmentation, so the
    # flood would just get trimmed away again — wasted DSP work either way).
    # A couple of copies still buys colour/geometry diversity.
    if not boxes:
        _n_copies   = 2    if fomo_version == 2 else 3
    _rot_range      = 8.0  if fomo_version == 2 else 15.0
    _translate_max  = 0.04 if fomo_version == 2 else 0.06
    _scale_range    = (0.92, 1.08) if fomo_version == 2 else (0.88, 1.12)
    _bright_range   = (0.85, 1.15) if fomo_version == 2 else (0.75, 1.25)
    _contrast_range = (0.90, 1.10) if fomo_version == 2 else (0.80, 1.20)

    for _ in range(_n_copies):
        aug = features.copy()
        aug_boxes = list(boxes)

        # Spatial transforms — mutually exclusive between heavy (rotate) and
        # light (translate+scale) to prevent label-position errors from stacking.
        if rng.random() < 0.4:
            angle = float(rng.uniform(-_rot_range, _rot_range))
            try:
                aug, aug_boxes = _aug_rotate_clamped(aug, aug_boxes, img_w, img_h, angle)
                stats["rotation"] += 1
            except Exception:
                pass
        else:
            aug, aug_boxes = _aug_translate(aug, aug_boxes, img_w, img_h, rng, max_shift=_translate_max)
            stats["translate"] += 1
            scale = float(rng.uniform(*_scale_range))
            aug, aug_boxes = _aug_scale_jitter(aug, aug_boxes, img_w, img_h, scale)

        # Horizontal flip p=0.5 (realistic — left/right symmetry valid for animals)
        if rng.random() < 0.5:
            aug, aug_boxes = _aug_hflip(aug, aug_boxes, img_w)
            stats["flip"] += 1

        # Colour jitter — version-specific ranges
        aug = _aug_color_jitter(aug, rng, brightness=_bright_range,
                                contrast=_contrast_range, color_strength=0.05)
        stats["brightness"] += 1

        augmented.append((aug, aug_boxes))

    return augmented, stats


# ─── FOMO class balancer ──────────────────────────────────────────────────────

def _fomo_balance_classes(
    fomo_train: list,
    label_map: dict,
    label_names: List[str],
    max_oversample: int = 5,
) -> list:
    """
    Oversample underrepresented classes in the FOMO training split.

    Alignment with Edge Impulse
    ---------------------------
    EI applies conservative oversampling to detection datasets so that
    minority classes receive roughly as many gradient updates as the majority
    class.  The default cap is raised to 5× (from 3×) so that severely
    underrepresented classes (e.g. dog when cat has ≥4× more samples) are
    more fully rebalanced at the data level.  Each oversampled copy still
    receives four unique augmented variants, so the effective diversity grows
    faster than the raw repeat count.

    Algorithm
    ---------
    1. Count how many training samples contain ≥1 box of each class.
    2. For each class below the majority count, compute:
           extra_copies = min(floor(majority / minority) - 1, max_oversample - 1)
    3. Append that many extra copies of each minority-class sample.
    4. Shuffle so duplicates are interleaved rather than block-appended.

    Only the training split is modified; test/eval data is never touched.

    Parameters
    ----------
    fomo_train    : list of (features, boxes) tuples
    label_map     : dict mapping label_id or label_name → class index (0-based)
    label_names   : ordered list of object class names (background excluded)
    max_oversample: maximum total copies of any sample (including the original)
    """
    n_classes = len(label_names)
    if n_classes < 2 or not fomo_train:
        return fomo_train

    # ── Step 0: hold negatives aside ──────────────────────────────────────────
    # A background sample's `present` set is empty, so it is counted for no
    # class and oversampled for none.  Left in the pool it would still inflate
    # `len(fomo_train)` while every minority *positive* class around it gets
    # duplicated up to 5×, so the neg:pos ratio the caller asked for would drift
    # by a different amount on every run depending on which class was the
    # minority.  Balancing operates on positives only; the negatives are then
    # scaled by the same factor the positive pool grew by, which preserves the
    # incoming ratio exactly.  No negatives → identical behaviour to before.
    positives = [(f, b) for f, b in fomo_train if b]
    negatives = [(f, b) for f, b in fomo_train if not b]
    if not positives:
        return fomo_train
    fomo_train = positives

    # ── Step 1: count how many training samples contain each class ────────────
    sample_classes: List[set] = []          # set of class indices per sample
    class_sample_count = [0] * n_classes   # samples that contain class i

    for features, boxes in fomo_train:
        present: set = set()
        for box in boxes:
            key = box.get("label_id") or box.get("label")
            if key is not None and key in label_map:
                idx = label_map[key]
                if 0 <= idx < n_classes:
                    present.add(idx)
        sample_classes.append(present)
        for idx in present:
            class_sample_count[idx] += 1

    raw_report = ", ".join(
        f"{label_names[i]}={class_sample_count[i]}" for i in range(n_classes)
    )
    logger.info(f"[FOMO] Class balance — raw train samples per class: {raw_report}")

    def _restore_negatives(balanced_positives: list) -> list:
        """Re-attach the negatives, scaled so the neg:pos ratio is unchanged."""
        if not negatives:
            return balanced_positives
        growth = len(balanced_positives) / max(len(positives), 1)
        target_neg = max(len(negatives), int(round(len(negatives) * growth)))
        out = list(balanced_positives)
        while len(out) - len(balanced_positives) < target_neg:
            out.extend(negatives[: target_neg - (len(out) - len(balanced_positives))])
        logger.info(
            "[FOMO] Class balance — negatives held at the incoming ratio: "
            "%d → %d copies (positives grew %.2f×)",
            len(negatives), target_neg, growth,
        )
        np.random.shuffle(out)
        return out

    majority_count = max(class_sample_count) if class_sample_count else 0
    if majority_count == 0:
        return _restore_negatives(fomo_train)

    # ── Step 2: skip if already balanced (all counts within 20 % of majority) ─
    if all(c == 0 or c >= 0.8 * majority_count for c in class_sample_count):
        logger.info("[FOMO] Class balance — dataset already balanced, no oversampling.")
        return _restore_negatives(fomo_train)

    # ── Step 3: oversample each underrepresented class ────────────────────────
    balanced = list(fomo_train)

    for cls_idx in range(n_classes):
        count = class_sample_count[cls_idx]
        if count == 0 or count >= majority_count:
            continue

        # extra_copies: how many additional full rounds of cls_samples to append
        ratio = majority_count / count                        # e.g. 50 / 10 = 5.0
        extra_copies = min(int(ratio) - 1, max_oversample - 1)  # e.g. min(4, 2) = 2
        if extra_copies <= 0:
            continue

        cls_samples = [
            fomo_train[i]
            for i, sc in enumerate(sample_classes)
            if cls_idx in sc
        ]
        for _ in range(extra_copies):
            balanced.extend(cls_samples)

        estimated = count + extra_copies * len(cls_samples)
        logger.info(
            f"[FOMO] Class balance — oversampling '{label_names[cls_idx]}': "
            f"{count} → ~{estimated} samples "
            f"(+{extra_copies}× copies, cap={max_oversample}×)"
        )

    # ── Step 4: shuffle so duplicates are distributed through the epoch ───────
    np.random.shuffle(balanced)
    logger.info(
        f"[FOMO] Class balance — balanced train set: "
        f"{len(balanced)} positive samples (was {len(fomo_train)})"
    )
    return _restore_negatives(balanced)


# ─── Dataset loader ───────────────────────────────────────────────────────────

def _load_dataset(
    db, impulse: Impulse, architecture: str = "dense", balance_classes: bool = True,
    fomo_version: int = 1, extra_params: Optional[dict] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """Load all labeled samples, run DSP, return train/test arrays."""
    is_fomo = "fomo" in architecture.lower()
    # Detection only. Every allow_background=True below sits inside an
    # `if is_fomo:` branch, so a classification impulse loads exactly the
    # samples it always did.
    _neg_cap = _resolve_negative_ratio_cap(impulse, extra_params)

    all_labels = db.query(Label).filter(Label.project_id == impulse.project_id).all()

    # IDs of labels whose names are NOT placeholders ("Unlabeled", etc.).
    # Used by is_sample_usable() to exclude samples whose label_id points to
    # an accidental placeholder label row.
    usable_label_ids: set = {
        l.id for l in all_labels if not label_name_is_placeholder(l.name)
    }

    if not is_fomo:
        # Classification: only include labels that have ≥1 sample assigned
        # AND whose name is not a placeholder.  This prevents phantom output
        # neurons for deleted/placeholder labels.
        active_label_ids: set = {
            row[0]
            for row in db.query(Sample.label_id)
            .filter(
                Sample.project_id == impulse.project_id,
                Sample.label_id.in_(usable_label_ids),
            )
            .distinct()
            .all()
        }
        labels = [l for l in all_labels if l.id in active_label_ids]
        if len(labels) != len(all_labels):
            dropped = [l.name for l in all_labels if l.id not in active_label_ids]
            logger.info(
                f"[Training] Skipping {len(dropped)} label(s) with zero samples "
                f"or placeholder names: {dropped}. "
                f"Using {len(labels)} active label(s) for training."
            )
    else:
        # FOMO / detection: keep only labels actually referenced by a real
        # annotation.  Exclude placeholder-named labels and boxes whose label
        # key is a placeholder string ("Unlabeled", etc.).
        _all_samples_fomo = (
            db.query(Sample)
            .filter(Sample.project_id == impulse.project_id)
            .all()
        )
        _active_fomo: set = set()
        for _s in _all_samples_fomo:
            # A background sample asserts none of the classes are present, so it
            # must not contribute to the active label set — otherwise a stale
            # annotation left on a sample the user later marked as a negative
            # would mint a phantom output class.
            if is_background_sample(_s):
                continue
            if isinstance(_s.extra_metadata, dict):
                for _box in _s.extra_metadata.get("boundingBoxes", []):
                    _key = _box.get("label_id") or _box.get("label")
                    if _key and str(_key).lower() not in UNLABELED_NAMES:
                        _active_fomo.add(_key)
        labels = [
            l for l in all_labels
            if (l.id in _active_fomo or l.name in _active_fomo)
            and not label_name_is_placeholder(l.name)
        ]
        if len(labels) != len(all_labels):
            _dropped = [
                l.name for l in all_labels
                if (l.id not in _active_fomo and l.name not in _active_fomo)
                or label_name_is_placeholder(l.name)
            ]
            logger.info(
                f"[Training] FOMO: skipping {len(_dropped)} label(s) with no "
                f"real annotations or placeholder names: {_dropped}. "
                f"Using {len(labels)} active label(s)."
            )

    # Build a dual-key label_map so bounding boxes can be resolved by either
    # UUID (label_id field, standard DB reference) OR name string (label field,
    # the format written by AI-labeling and the annotation UI).
    #
    # Root cause of TP=0/FP=0/FN=0: AI-labeling stores boxes as
    #   {"label": "monkey", ...}  — NOT {"label_id": "<uuid>", ...}
    # so a UUID-only label_map silently skips every box → all-background GT
    # heatmaps → model trains/evaluates on empty targets → all zeros.
    label_map: dict = {}
    for idx, l in enumerate(labels):
        label_map[l.id]   = idx   # boxes with  label_id: "<uuid>"
        label_map[l.name] = idx   # boxes with  label:    "name"   (AI-labeling)
    label_names = [l.name for l in labels]

    dsp_blocks = impulse.dsp_blocks or [{"type": "raw", "params": {}}]

    # Derive the canonical image dimensions from the DSP image-block params.
    # This is the same source the DSPProcessor uses, so DSP output and the
    # heatmap normalisation always agree on width/height.
    # Priority: dsp_block.params > impulse.image_width/height > 96 (last resort).
    _img_block: dict = {"params": {}}
    for _blk in dsp_blocks:
        if _blk.get("type") == "image":
            _img_block = _blk
            break
    _dsp_img_params = merge_image_params(impulse, _img_block)
    img_w: int = (
        _dsp_img_params.get("image_width")
        or impulse.image_width
        or 96
    )
    img_h: int = (
        _dsp_img_params.get("image_height")
        or impulse.image_height
        or 96
    )
    if img_w == 96 and img_h == 96 and not (impulse.image_width or _dsp_img_params.get("image_width")):
        logger.warning(
            "FOMO: no image_width/height found in DSP params or impulse config — "
            "defaulting to 96×96. Configure impulse.image_width/height or "
            "dsp_block.params.image_width/height to avoid this fallback."
        )
    img_w = int(img_w)
    img_h = int(img_h)

    # ── Try pre-generated feature cache ──────────────────────────────────────
    # Collect only the IDs of *usable* samples (same definition as DSP worker)
    # so that newly-uploaded unlabeled / placeholder samples do not falsely
    # invalidate a valid cache.
    if is_fomo:
        _all_project_samples = (
            db.query(Sample)
            .filter(Sample.project_id == impulse.project_id)
            .all()
        )
        _usable_samples = [
            s for s in _all_project_samples
            if is_sample_usable(s, usable_label_ids, allow_background=True)
        ]
        _current_sample_ids = [s.id for s in _usable_samples]
        logger.info(
            f"[Training] Cache check: {len(_all_project_samples)} total project samples, "
            f"{len(_current_sample_ids)} usable (labeled/annotated), "
            f"{len(_all_project_samples) - len(_current_sample_ids)} skipped "
            f"(unlabeled or placeholder label)"
        )
    else:
        # Classification: samples whose label_id is in usable_label_ids only.
        _current_sample_ids = [
            row[0]
            for row in db.query(Sample.id)
            .filter(
                Sample.project_id == impulse.project_id,
                Sample.label_id.in_(usable_label_ids),
            )
            .all()
        ]
        logger.info(
            f"[Training] Cache check: {len(_current_sample_ids)} usable "
            f"(non-placeholder label) samples for classification."
        )

    _cache_t0 = time.perf_counter()
    _cached = _try_load_feature_cache(impulse, _current_sample_ids, is_fomo, label_names)
    _log_event(
        _dsp_log, "dsp.feature_cache",
        impulse_id=impulse.id, hit=bool(_cached),
        lookup_ms=round((time.perf_counter() - _cache_t0) * 1000.0, 2),
    )
    if _cached is not None:
        return _build_arrays_from_cache(
            _cached, impulse, label_map, label_names, is_fomo, img_w, img_h,
            balance_classes=balance_classes,
            fomo_version=fomo_version,
            negative_ratio_cap=_neg_cap,
        )

    logger.info(
        "[Training] Cached features missing or stale — "
        "recomputing features from raw samples."
    )

    X_train, y_train, y_train_eval, X_test, y_test = [], [], [], [], []

    # FOMO: include only usable project samples — detection targets come from
    # per-box labels in extra_metadata["boundingBoxes"], not top-level label_id.
    # Classification: only include samples with a non-placeholder label.
    if is_fomo:
        # Reuse the sample list already loaded for FOMO label discovery above
        # rather than re-querying every project sample a second time.
        # Pre-filter to usable samples so the loop skips unlabeled ones
        # without downloading their raw bytes from object storage.
        samples = [
            s for s in _all_samples_fomo
            if is_sample_usable(s, usable_label_ids, allow_background=True)
        ]
        # Honor train_subset_percent in this inline-recompute path too — when
        # the user lowered the subset percent and clicked "Train" without
        # re-running Generate Features, the cache was invalidated above and we
        # end up here. apply_train_subset is a no-op at 100%.
        samples = apply_train_subset(samples, impulse)

        # ── FOMO two-phase loading ────────────────────────────────────────────
        # Phase 1: run DSP and collect (features, boxes) for every sample.
        # Grid computation is deferred until we know the actual feature shape
        # so it matches model_builder's mult-of-8 snapping exactly — even when
        # impulse.image_width is not a multiple of 8 (e.g. 54 → snaps to 56).
        fomo_train: list = []   # [(features, boxes), ...]
        fomo_test:  list = []

        for sample, raw, fetch_error in iter_prefetched(
            samples, lambda s: storage.download_bytes(s.storage_key)
        ):
            try:
                if fetch_error is not None:
                    raise fetch_error
                all_features = []
                for block_cfg in dsp_blocks:
                    params = merge_image_params(impulse, block_cfg)
                    proc = DSPProcessor(
                        block_type=block_cfg.get("type", "raw"),
                        params=params,
                        frequency_hz=sample.frequency_hz or impulse.frequency_hz or 100.0,
                    )
                    all_features.append(proc.extract(raw))
                features = _combine_dsp_features(all_features)

                # Strip placeholder-labeled boxes so they don't produce
                # all-background GT heatmaps (label_map has no entry for them).
                raw_boxes = (sample.extra_metadata or {}).get("boundingBoxes", [])
                raw_boxes = [
                    b for b in raw_boxes
                    if isinstance(b, dict)
                    and (b.get("label_id") or b.get("label") or "").strip().lower()
                    not in UNLABELED_NAMES
                ]
                # Normalize to [0,1] using real image dims so GT cell mapping
                # matches the model-testing contract (normalize_bounding_boxes →
                # gt_cells_from_boxes).  Without this, pixel-space boxes divided
                # by img_w=96 (DSP size) map to different cells than testing.
                # normalize_bounding_boxes also resolves the label field via a
                # priority chain (label_id → label → labelId → class_name).
                boxes = normalize_bounding_boxes(raw_boxes, image_bytes=raw)
                # Drop boxes where no label could be resolved after the full
                # fallback chain.  These would be silently skipped inside
                # _create_fomo_heatmap — filtering here keeps the warning quiet
                # and makes the data-quality issue visible at sample level.
                boxes = [b for b in boxes if b.get("label_id") is not None]
                # A sample with no resolvable boxes is dropped — EXCEPT an
                # explicitly marked background image, which is a legitimate
                # negative and trains with an all-background GT heatmap.
                if not boxes and not is_background_sample(sample):
                    continue
                if is_background_sample(sample):
                    boxes = []      # a negative never contributes GT, ever

                bucket = fomo_test if sample.sample_type == SampleType.testing else fomo_train
                bucket.append((features, boxes))
            except Exception as e:
                logger.warning(f"Skipping sample {sample.id}: {e}")

        all_fomo = fomo_train + fomo_test
        if not all_fomo:
            return np.empty((0,)), np.array([]), np.empty((0,)), np.array([]), label_names

        # Phase 2: derive grid from the actual DSP feature shape.
        # Apply the same mult-of-8 ceiling that model_builder uses so that
        # heatmap target shape == model output shape unconditionally.
        # v1: stride-16 (block_13_expand_relu); v2: stride-8 (block_6_expand_relu).
        # Guard: all v2 logic is behind the fomo_version == 2 branch.
        feat_shape = np.asarray(all_fomo[0][0], dtype=np.float32).shape
        def _snap8(n: int) -> int:
            return n if n % 8 == 0 else n + (8 - n % 8)
        _grid_div = 8 if fomo_version == 2 else 16
        _fomo_version_local = fomo_version
        if len(feat_shape) >= 2:
            grid_h = _snap8(feat_shape[0]) // _grid_div
            grid_w = _snap8(feat_shape[1]) // _grid_div
        else:
            grid_h = _snap8(img_h) // _grid_div
            grid_w = _snap8(img_w) // _grid_div
        logger.info(
            f"FOMO v{_fomo_version_local} grid: {grid_h}×{grid_w} "
            f"(feature shape {feat_shape}, snapped mult-of-8, stride-{_grid_div})"
        )
        if grid_w < 6 or grid_h < 6:
            _hint = (
                f"Set image size to ≥48×48 and re-run Generate Features. "
                f"48×48 → 6×6 grid at stride-8."
                if _fomo_version_local == 2 else
                f"Set image width and height to 96×96 (or larger) in your DSP settings "
                f"and re-run Generate Features. A 96×96 input gives a 6×6 grid (36 cells) at stride-16."
            )
            logger.warning(
                f"[FOMO v{_fomo_version_local}] Grid {grid_h}×{grid_w} is below the recommended minimum of 6×6. "
                f"Current image size {img_w}×{img_h} gives only {grid_w * grid_h} detection cells. "
                f"Action: {_hint}"
            )

        # Phase 3: class balance (training split only, test data unchanged).
        if balance_classes:
            fomo_train = _fomo_balance_classes(fomo_train, label_map, label_names)

        # v1 stays on strict one-cell targets.
        # v2 keeps strict eval targets but adds a conservative soft falloff
        # around the center during training to improve exact-cell ranking.
        _target_cfg = _fomo_training_target_config(fomo_version)
        _use_spread = bool(_target_cfg["spread"])
        logger.info(
            "[FOMO v%d] target mode=%s  spread=%s  grid=%d×%d  img=%d×%d  reason=%s",
            fomo_version, _target_cfg["mode"], _use_spread, grid_h, grid_w, img_w, img_h,
            _target_cfg["reason"],
        )
        _aug_stats_total: dict = {"flip": 0, "brightness": 0, "contrast": 0, "rotation": 0}
        _neg_flags: List[bool] = []
        for features, boxes in fomo_train:
            _is_neg = not boxes
            target = _create_fomo_heatmap(
                boxes, grid_w, grid_h, img_w, img_h, len(label_names), label_map,
                spread=_use_spread,
                target_mode=str(_target_cfg["mode"]),
                center_value=float(_target_cfg.get("center_value", 1.0)),
                neighbor_value=float(_target_cfg.get("neighbor_value", 0.0)),
                diagonal_value=float(_target_cfg.get("diagonal_value", 0.0)),
            )
            eval_target = _create_fomo_heatmap(
                boxes, grid_w, grid_h, img_w, img_h, len(label_names), label_map,
                spread=False,
            )
            X_train.append(features)
            y_train.append(target)
            y_train_eval.append(eval_target)
            _neg_flags.append(_is_neg)
            # Augmented copies (v1: 10 per sample, v2: 5 per sample)
            aug_list, aug_stats = _augment_fomo_sample(features, boxes, img_w, img_h, fomo_version=fomo_version)
            for k in _aug_stats_total:
                _aug_stats_total[k] += aug_stats.get(k, 0)
            for aug_feat, aug_boxes in aug_list:
                aug_target = _create_fomo_heatmap(
                    aug_boxes, grid_w, grid_h, img_w, img_h, len(label_names), label_map,
                    spread=_use_spread,
                    target_mode=str(_target_cfg["mode"]),
                    center_value=float(_target_cfg.get("center_value", 1.0)),
                    neighbor_value=float(_target_cfg.get("neighbor_value", 0.0)),
                    diagonal_value=float(_target_cfg.get("diagonal_value", 0.0)),
                )
                aug_eval_target = _create_fomo_heatmap(
                    aug_boxes, grid_w, grid_h, img_w, img_h, len(label_names), label_map,
                    spread=False,
                )
                X_train.append(aug_feat)
                y_train.append(aug_target)
                y_train_eval.append(aug_eval_target)
                _neg_flags.append(_is_neg)

        # Cap the negative share LAST — balancing and augmentation above both
        # move the positive count this ratio is measured against.
        X_train, y_train, y_train_eval = _apply_fomo_negative_cap(
            X_train, y_train, y_train_eval, _neg_flags, _neg_cap,
            seed=f"{getattr(impulse, 'id', '?')}|raw", prefix="[FOMO] (raw)",
        )

        for features, boxes in fomo_test:
            target = _create_fomo_heatmap(
                boxes, grid_w, grid_h, img_w, img_h, len(label_names), label_map,
                spread=False,
            )
            X_test.append(features)
            y_test.append(target)
        _aug_kept = len(X_train) - len(fomo_train)
        logger.info(
            f"FOMO dataset: {len(fomo_train)} train samples (after balancing) "
            f"→ {len(X_train)} after augmentation "
            f"({_aug_kept} copies kept, 0 discarded — "
            f"flip={_aug_stats_total['flip']}, brightness={_aug_stats_total['brightness']}, "
            f"contrast={_aug_stats_total['contrast']}, rotation={_aug_stats_total['rotation']}), "
            f"{len(fomo_test)} test samples."
        )

    else:
        # ── Classification ────────────────────────────────────────────────────
        # Only fetch samples with a non-placeholder label so that samples
        # pointing to "Unlabeled" label rows are excluded from dataset loading.
        samples = (
            db.query(Sample)
            .filter(
                Sample.project_id == impulse.project_id,
                Sample.label_id.in_(usable_label_ids),
            )
            .all()
        )
        # Same subset-percent application as the FOMO path above; no-op at 100%.
        samples = apply_train_subset(samples, impulse)

        for sample, raw, fetch_error in iter_prefetched(
            samples, lambda s: storage.download_bytes(s.storage_key)
        ):
            try:
                if fetch_error is not None:
                    raise fetch_error
                values = raw

                all_features = []
                for block_cfg in dsp_blocks:
                    params = merge_image_params(impulse, block_cfg)
                    proc = DSPProcessor(
                        block_type=block_cfg.get("type", "raw"),
                        params=params,
                        frequency_hz=sample.frequency_hz or impulse.frequency_hz or 100.0,
                    )
                    all_features.append(proc.extract(values))

                features = _combine_dsp_features(all_features)

                if sample.label_id not in label_map:
                    continue
                target = label_map[sample.label_id]

                if sample.sample_type == SampleType.testing:
                    X_test.append(features)
                    y_test.append(target)
                else:
                    X_train.append(features)
                    y_train.append(target)
                    y_train_eval.append(target)

            except Exception as e:
                logger.warning(f"Skipping sample {sample.id}: {e}")

    X_train = np.array(X_train, dtype=np.float32) if X_train else np.empty((0,))
    y_train = np.array(y_train)
    y_train_eval = np.array(y_train_eval)
    X_test  = np.array(X_test,  dtype=np.float32) if X_test  else np.empty((0,))
    y_test  = np.array(y_test)

    return X_train, y_train, y_train_eval, X_test, y_test, label_names


def _build_fomo_export_model_for_tflite(model):
    """
    Build a preprocessing-free TFLite export model from a trained FOMO model.

    Why this exists
    ---------------
    This is a deployment-contract transform, NOT a converter workaround.
    Devices feed raw camera pixels, not DSP output tensors, so the shipped
    model must not carry the training graph's Rescaling / Resizing layers,
    and it must emit probabilities rather than raw logits.

    Re-wire the backbone and head layer OBJECTS (same Python instances → same
    trained weights) to a fresh keras.Input in [-1, 1] range.  The result is
    a single clean functional model with:
      • no preprocessing layers (no Resizing, no Rescaling)
      • a Softmax appended to the head

    The deployment caller is responsible for applying the normalization:
        normalized = dsp_output * 2.0 - 1.0

    The metadata written alongside the export encodes that contract as
    normalize_input=True, input_mean=127.5, input_std=127.5.

    Historical note: this rewiring was previously believed to also cure the
    "LLVM ERROR: Failed to infer result type(s)" abort by removing the
    backbone's dual variable registration.  It does not — the abort was
    caused by exporting a Keras 3 model through tf.saved_model.save(), and
    it reproduces on a freshly loaded model with no nested sub-models.  See
    _convert_to_tflite for the actual cause and fix.  Keep this function:
    the deployment contract above still depends on it.

    Returns the export model, or the original model unchanged if rewiring
    fails for any reason (so callers always get a usable model object).
    """
    from tensorflow import keras

    # Locate the normalization gate — this is the boundary between
    # preprocessing and the backbone.
    try:
        norm_layer = model.get_layer("normalize")
    except ValueError:
        return model  # no preprocessing to strip

    backbone_shape = norm_layer.output.shape[1:]  # (H, W, 3)

    # Fresh input in the backbone's expected [-1, 1] range.
    export_inp = keras.Input(shape=backbone_shape, name="input")

    # Layers to skip when rebuilding the export graph.
    skip = {"input", "normalize"}
    try:
        model.get_layer("resize")
        skip.add("resize")
    except ValueError:
        pass

    # Tensor remapping: training-graph KerasTensor id → export-graph tensor.
    # Seed: normalize.output (training) maps to export_inp.
    tensor_map: dict = {id(norm_layer.output): export_inp}

    # model.layers is topologically sorted — iterate in order.
    for layer in model.layers:
        if layer.name in skip or isinstance(layer, keras.layers.InputLayer):
            continue
            
        # Keras 3 (TensorFlow 2.16+) uses _inbound_nodes (private) or node-based access.
        # Fall back to _inbound_nodes if inbound_nodes is missing.
        nodes = getattr(layer, "inbound_nodes", getattr(layer, "_inbound_nodes", None))
        if not nodes:
            continue

        # Read the original training-graph connectivity (index 0).
        # Calling layer(...) below will append a new node; index 0 stays.
        node = nodes[0]
        
        # Check for input_tensors (standard for functional models)
        if not hasattr(node, "input_tensors"):
            continue
        raw_in = node.input_tensors
        if not isinstance(raw_in, (list, tuple)):
            raw_in = [raw_in]

        export_in = [tensor_map.get(id(t)) for t in raw_in]
        if any(v is None for v in export_in):
            # Input not yet mapped — layer is upstream of normalize; skip.
            continue

        # Call the SAME layer object on the export tensors.
        # This creates a new inbound node while preserving trained weights.
        call_arg = export_in[0] if len(export_in) == 1 else export_in
        try:
            new_out = layer(call_arg)
        except Exception as exc:
            logger.warning(
                f"FOMO export rewiring: layer {layer.name!r} failed ({exc}); "
                "falling back to training model for TFLite conversion."
            )
            return model

        # Check for output_tensors (standard for functional models)
        if hasattr(node, "output_tensors"):
            raw_out = node.output_tensors
        else:
            # Fallback for some layers that might use 'outputs'
            raw_out = getattr(node, "outputs", [])
            
        if not isinstance(raw_out, (list, tuple)):
            raw_out = [raw_out]
        new_outs = new_out if isinstance(new_out, (list, tuple)) else [new_out]
        for ro, no in zip(raw_out, new_outs):
            tensor_map[id(ro)] = no

    export_out = tensor_map.get(id(model.output))
    if export_out is None:
        logger.warning(
            "FOMO export rewiring: model output not found in tensor map; "
            "falling back to training model for TFLite conversion."
        )
        return model

    # The training head outputs raw logits (no activation).  Append a Softmax
    # so the TFLite model delivers probabilities — matching EI's deployment
    # contract and keeping the post-processing threshold logic unchanged.
    try:
        export_out = keras.layers.Softmax(axis=-1, name="fomo_softmax")(export_out)
    except Exception as _sm_exc:
        logger.warning(
            f"FOMO export: could not append Softmax ({_sm_exc}); "
            "logits exported as-is (downstream thresholding must account for this)."
        )

    try:
        return keras.Model(export_inp, export_out, name=f"{model.name}_tflite")
    except Exception as exc:
        logger.warning(
            f"FOMO export model build failed ({exc}); "
            "falling back to training model for TFLite conversion."
        )
        return model


def _prepare_fomo_export_cal_data(
    representative_data: np.ndarray,
    export_input_shape: tuple,
) -> np.ndarray:
    """
    Transform FOMO representative calibration data to match the export model's
    input shape (H_snap, W_snap, 3) after preprocessing layers have been stripped.

    The training model's input may be:
      • 2-D grayscale: (N, H, W)   e.g. (N, 63, 63)
      • 3-D RGB:       (N, H, W, 3)

    In both cases the export model's first real layer (Conv2D) expects rank-4
    input: (batch, H_snap, W_snap, 3).  This function applies four steps:

      1. Add channel dim if missing    (N, H, W) -> (N, H, W, 1)
      2. Expand to 3 channels          (N, H, W, 1) -> (N, H, W, 3)
      3. Resize to export spatial dims (N, H, W, 3) -> (N, H_snap, W_snap, 3)
      4. Normalize [0, 1] -> [-1, 1]   (DSP output range -> backbone expected range)

    Returns a float32 ndarray shaped (N, H_snap, W_snap, 3).
    """
    import tensorflow as tf

    tgt_h, tgt_w, tgt_c = int(export_input_shape[0]), int(export_input_shape[1]), int(export_input_shape[2])
    data = representative_data.astype(np.float32)

    # Step 1: ensure 4-D (add channel dim for 2-D grayscale inputs)
    if data.ndim == 3:
        data = data[:, :, :, np.newaxis]  # (N, H, W, 1)

    # Step 2: expand single channel to tgt_c channels
    if data.shape[3] == 1 and tgt_c > 1:
        data = np.repeat(data, tgt_c, axis=3)  # (N, H, W, 3)

    # Step 3: resize spatial dims if they don't already match
    cur_h, cur_w = data.shape[1], data.shape[2]
    if cur_h != tgt_h or cur_w != tgt_w:
        data = tf.image.resize(data, [tgt_h, tgt_w]).numpy()

    # Step 4: normalize [0, 1] -> [-1, 1] (DSP output to backbone contract)
    data = data * 2.0 - 1.0

    return data.astype(np.float32)


def _convert_to_tflite(
    model, representative_data: np.ndarray, quantize: bool = False,
) -> bytes:
    """
    Convert Keras model to TFLite in-process.

    For FOMO models, first calls _build_fomo_export_model_for_tflite to strip
    preprocessing layers (Rescaling / Resizing) and append the deployment
    Softmax.  That rewiring is a deployment-contract requirement, not a
    converter workaround — see that function's docstring.

    If preprocessing was stripped, the representative dataset is reshaped and
    normalized to match the export model's input contract (H_snap, W_snap, 3)
    in [-1, 1] range, so int8 calibration receives rank-4 tensors of the
    correct spatial size regardless of the original training input shape.

    Why model.export() and not tf.saved_model.save() / from_keras_model()
    ────────────────────────────────────────────────────────────────────
    This project pins tensorflow==2.16.1, which ships Keras 3.  Under Keras 3
    the only export API the TFLite converter can consume is the Keras
    ExportArchive written by keras.Model.export().

    Passing a Keras 3 model to tf.saved_model.save() produces a SavedModel
    whose resource variables the MLIR freezing pass cannot lift into global
    tensors.  Every ReadVariableOp is then left without a 'value' attribute,
    shape inference fails, and the converter aborts the *process*:

        loc(fused["ReadVariableOp:", "<model>/Conv1_1/convolution/ReadVariableOp"]):
            error: missing attribute 'value'
        LLVM ERROR: Failed to infer result type(s).

    That is a C++ abort() (SIGABRT), not a Python exception, so `except`
    cannot catch it and the Celery worker dies mid-task.  from_keras_model()
    fails the same way because it re-traces the Keras 3 model through the
    same path.  Both were verified to abort on CPU with CUDA disabled, so
    this is unrelated to GPU/CUDA/cuDNN/XLA.

    keras.Model.export() writes a fully-traced tf.function with its variables
    already resolved, which from_saved_model() converts cleanly.  The same
    fix is already in use in the MobileNetV2-SSD exporter.

    Conversion strategies (applied to the export model, in order):
      1. model.export() + from_saved_model  — the supported Keras 3 path
      2. frozen-const  + from_saved_model   — converts every ReadVariableOp to
                                              a Const before saving, so the
                                              graph has no resource reads left
    Both are MLIR paths that fail with catchable Python exceptions.  The
    legacy TOCO strategies were removed: TF 2.16 ships no toco_from_protos
    binary, and from_keras_model() aborts the process (see above).
    """
    import tensorflow as tf

    # Build preprocessing-free export model (no-op for non-FOMO models).
    export_model = _build_fomo_export_model_for_tflite(model)
    preprocessing_stripped = export_model is not model

    # Transform representative data when preprocessing was stripped.
    # For int8 calibration the converter feeds samples directly to the export
    # model's first real layer (Conv2D), which expects rank-4 input shaped
    # (1, H_snap, W_snap, 3) in [-1, 1] range.  Training data may be 2-D
    # grayscale (N, H, W) in [0, 1], so we must add the channel dim, tile to
    # 3 channels, resize to the snapped export size, and then normalise.
    if preprocessing_stripped and quantize and len(representative_data) > 0:
        cal_data = _prepare_fomo_export_cal_data(
            representative_data, export_model.input_shape[1:]
        )
    else:
        cal_data = representative_data

    def _apply_quant(converter):
        if not quantize or len(cal_data) == 0:
            return
        rep = cal_data[:200].astype(np.float32)

        def representative_dataset():
            for i in range(len(rep)):
                yield [rep[i : i + 1]]

        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative_dataset
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8

    def _strategy_keras_export() -> bytes:
        # Keras 3 ExportArchive → SavedModel → MLIR converter.  This is the
        # only export API TF 2.16 / Keras 3 officially supports for TFLite.
        with tempfile.TemporaryDirectory() as tmpdir:
            saved_model_path = os.path.join(tmpdir, "saved_model")
            export_model.export(saved_model_path)
            converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_path)
            _apply_quant(converter)
            return converter.convert()

    def _strategy_frozen_const() -> bytes:
        # Fallback with a genuinely different failure mode: trace the model
        # once, then rewrite every ReadVariableOp into a Const holding the
        # variable's current value.  The saved object is a plain tf.Module
        # whose graph contains no resource reads at all, so the MLIR freezing
        # pass that breaks on Keras 3 models has nothing left to freeze.
        #
        # tf.saved_model.save() is safe *here* precisely because the graph is
        # already variable-free — do not "simplify" this by passing the Keras
        # model directly, which is the crash described in the docstring.
        from tensorflow.python.framework.convert_to_constants import (
            convert_variables_to_constants_v2,
        )

        input_spec = [
            tf.TensorSpec(
                shape=[1] + list(export_model.input_shape[1:]),
                dtype=tf.float32,
                name="input",
            )
        ]

        @tf.function(input_signature=input_spec)
        def _serve(x):
            return export_model(x, training=False)

        frozen_fn = convert_variables_to_constants_v2(_serve.get_concrete_function())

        with tempfile.TemporaryDirectory() as tmpdir:
            module = tf.Module()
            module.serve = frozen_fn
            tf.saved_model.save(
                module, tmpdir, signatures={"serving_default": frozen_fn}
            )
            converter = tf.lite.TFLiteConverter.from_saved_model(tmpdir)
            _apply_quant(converter)
            return converter.convert()

    if preprocessing_stripped:
        logger.info(
            "TFLite export: FOMO preprocessing stripped; export model expects "
            "[-1, 1] input and emits softmax probabilities."
        )

    strategies = [
        ("keras_export", _strategy_keras_export),
        ("frozen_const", _strategy_frozen_const),
    ]

    for idx, (label, strategy) in enumerate(strategies, start=1):
        try:
            return strategy()
        except Exception as e:
            log_fn = logger.error if idx == len(strategies) else logger.warning
            log_fn(
                f"TFLite strategy {idx} ({label}) failed (quantize={quantize}): {e}"
            )

    return b""


# ─── Progress callback ────────────────────────────────────────────────────────

import tensorflow as tf

class _ProgressCallback(tf.keras.callbacks.Callback):
    """
    Reports epoch-level training progress back to the Celery task state.

    progress_metric controls which Keras log key is surfaced as the headline
    metric in the task state:
      - classification models: "val_accuracy"  (default)
      - FOMO detection models: "val_f1"        (best detection macro-F1 over thresholds)
    """

    def __init__(self, task, job_id: str, total_epochs: int,
                 progress_metric: str = "val_accuracy"):
        super().__init__()
        self.task            = task
        self.job_id          = job_id
        self.total           = total_epochs
        self.progress_metric = progress_metric

    def on_epoch_end(self, epoch, logs=None):
        _raise_if_job_cancelled(self.job_id)
        logs = logs or {}
        progress = int((epoch + 1) / self.total * 100)
        # Use the designated metric; fall back gracefully to val_loss / loss.
        metric_val = logs.get(
            self.progress_metric,
            logs.get("val_loss", logs.get("loss", 0)),
        )
        self.task.update_state(
            state="PROGRESS",
            meta={
                "job_id":        self.job_id,
                "epoch":         epoch + 1,
                "total":         self.total,
                "progress":      progress,
                "metric":        self.progress_metric,
                "metric_value":  float(metric_val),
                "loss":          float(logs.get("val_loss", logs.get("loss", 0))),
            },
        )

    def on_train_batch_begin(self, batch, logs=None):
        # Poll every batch via the time-throttled cache so we stop within
        # ~0.5s of cancel without flooding the DB with one query per batch.
        _raise_if_job_cancelled_fast(self.job_id)


# ─── YOLO-Pro dataset loader ──────────────────────────────────────────────────


class FeatureResolutionMismatchError(ValueError):
    """
    Raised when the resolution the features were generated at cannot be used to
    train the selected model.

    This is a hard, user-visible failure on purpose: the alternative recoveries
    (recomputing every feature from raw data, resizing, or upsampling) all hide
    a misconfiguration behind minutes of wasted compute and produce a model
    trained on interpolated pixels.  Propagating out of the dataset loader lets
    the training task's terminal handler write job.status = failed and surface
    the message on the Training page via job.error_message.
    """


def _feature_resolution_mismatch_message(
    gen_w: int, gen_h: int, target_w: int, target_h: int
) -> str:
    """
    Build the user-facing feature-resolution-mismatch message.

    States the RULE (multiples of 32) rather than one hardcoded resolution, so
    the copy stays correct for every project: 136×136 rounds to 160×160,
    200×200 rounds to 224×224, and so on.
    """
    return (
        "Feature resolution mismatch\n"
        "\n"
        f"Your current features were generated at {gen_w}×{gen_h}, but the selected YOLO-Pro model will "
        f"train using {target_w}×{target_h} (because YOLO-Pro requires input "
        "dimensions that are multiples of 32).\n"
        "\n"
        f"Please regenerate the features using {target_w}×{target_h} before "
        "starting training."
    )


def _cap_detection_negatives(
    X: list, boxes: list, cap: float, seed: str, prefix: str,
) -> Tuple[list, list]:
    """Trim surplus negatives from an assembled (images, per-image boxes) pair.

    Shared by the YOLO-Pro and SSD loaders, whose training set is exactly these
    two parallel lists — a negative is simply an entry with an empty box list.
    """
    neg_flags = [not b for b in boxes]
    keep = _negative_keep_indices(neg_flags, cap, seed)
    if len(keep) != len(neg_flags):
        X     = [X[i] for i in keep]
        boxes = [boxes[i] for i in keep]
    _log_negative_ratio(prefix, sum(1 for b in boxes if not b), len(boxes), cap)
    return X, boxes


def _load_yolo_dataset(db, impulse, extra_params: Optional[dict] = None):
    """
    Load image samples + bounding-box annotations for YOLO-Pro training.

    Returns:
        X_train       : (N_train, H, W, C) float32
        boxes_train   : list of per-image box dicts  {'label','x','y','w','h'} 0-1 normalised
        X_test        : (N_test,  H, W, C) float32  (may be empty)
        boxes_test    : list of per-image box dicts
        label_names   : list[str]
        label_map     : {label_id_or_name -> class_index}
        input_shape   : (H, W, C)  — already snapped to multiples of 32,
                        and enforced to be at least YOLO_PRO_MIN_INPUT_SIZE
    """
    # _snap32 and YOLO_PRO_MIN_INPUT_SIZE are imported at module level.

    all_labels = db.query(Label).filter(Label.project_id == impulse.project_id).all()
    usable_label_ids = {
        l.id for l in all_labels if not label_name_is_placeholder(l.name)
    }

    samples = (
        db.query(Sample)
        .filter(Sample.project_id == impulse.project_id)
        .all()
    )
    samples = [
        s for s in samples
        if is_sample_usable(s, usable_label_ids, allow_background=True)
    ]
    _neg_cap = _resolve_negative_ratio_cap(impulse, extra_params)

    # Collect active labels from bounding-box annotations (same dual-key
    # approach as the FOMO path to support both label_id and label-name boxes).
    # Background samples are skipped entirely: a negative asserts that none of
    # the project's classes are present, so letting its (possibly stale)
    # label_id into `active` would mint an output class out of an image that
    # contains no instance of it.
    active: set = set()
    for s in samples:
        if is_background_sample(s):
            continue
        if isinstance(s.extra_metadata, dict):
            for box in s.extra_metadata.get("boundingBoxes", []):
                key = box.get("label_id") or box.get("label")
                if key and str(key).lower() not in UNLABELED_NAMES:
                    active.add(key)
        if s.label_id and s.label_id in usable_label_ids:
            active.add(s.label_id)

    labels = [
        l for l in all_labels
        if (l.id in active or l.name in active)
        and not label_name_is_placeholder(l.name)
    ]
    label_map: dict = {}
    for idx, l in enumerate(labels):
        label_map[l.id]   = idx
        label_map[l.name] = idx
    label_names = [l.name for l in labels]

    # Image dimensions — resolved via merged DSP image params (block-level
    # values take precedence over impulse root-level fields) then snapped to
    # multiples of 32 for YOLO-Pro's stride-32 path.
    dsp_blocks = impulse.dsp_blocks or [{"type": "image", "params": {}}]

    # Find the image DSP block (if any) and merge with impulse-level settings.
    _img_blk: dict = {"params": {}}
    for blk in dsp_blocks:
        if blk.get("type") == "image":
            _img_blk = blk
            break
    merged_img = merge_image_params(impulse, _img_blk)

    # Use merged params > impulse root > shared YOLO-Pro minimum/default.
    img_w = int(merged_img.get("image_width")  or impulse.image_width  or YOLO_PRO_MIN_INPUT_SIZE)
    img_h = int(merged_img.get("image_height") or impulse.image_height or YOLO_PRO_MIN_INPUT_SIZE)

    # Enforce YOLO-Pro minimum input resolution before snapping.
    if img_w < YOLO_PRO_MIN_INPUT_SIZE or img_h < YOLO_PRO_MIN_INPUT_SIZE:
        clamped_w = max(img_w, YOLO_PRO_MIN_INPUT_SIZE)
        clamped_h = max(img_h, YOLO_PRO_MIN_INPUT_SIZE)
        logger.warning(
            f"[YOLO-Pro] [ACTION:RESOLUTION_CLAMPED] "
            f"Resolved image size {img_w}×{img_h} is below the YOLO-Pro "
            f"minimum ({YOLO_PRO_MIN_INPUT_SIZE}×{YOLO_PRO_MIN_INPUT_SIZE}). "
            f"Clamping to {clamped_w}×{clamped_h} for this run. "
            "Tradeoff: training images were resized up from the cached resolution; "
            "the upscaled pixels carry no additional spatial information. "
            "Fix: set DSP image-block width/height >= "
            f"{YOLO_PRO_MIN_INPUT_SIZE} and re-run Generate Features."
        )
        img_w = clamped_w
        img_h = clamped_h

    # Resolution the DSP image block actually produces (post-clamp, pre-snap).
    # The raw-sample path below runs DSPProcessor with these same merged params,
    # so this is the size fresh features come out at — kept for the fail-fast
    # comparison against the snapped training resolution.
    dsp_gen_w, dsp_gen_h = img_w, img_h

    img_w = _snap32(img_w)
    img_h = _snap32(img_h)

    # Detect channel count from DSP image-block params
    n_ch = 3
    ch_mode = (merged_img or {}).get("channels", "RGB") or "RGB"
    n_ch = 1 if str(ch_mode).upper() in ("GRAY", "GRAYSCALE", "1") else 3

    input_shape = (img_h, img_w, n_ch)

    logger.info(
        f"[YOLO-Pro] Dataset load — target shape={input_shape}  "
        f"labels={label_names}  samples={len(samples)}"
    )

    # ── Try feature cache first ───────────────────────────────────────────────
    current_sample_ids = [str(s.id) for s in samples]
    _cache_t0 = time.perf_counter()
    _cached = _try_load_feature_cache(
        impulse,
        current_sample_ids,
        is_fomo=False,
        current_label_names=label_names,
        is_yolo_pro_model=True,
    )
    _log_event(
        _dsp_log, "dsp.feature_cache",
        impulse_id=impulse.id, hit=bool(_cached), model="yolo_pro",
        lookup_ms=round((time.perf_counter() - _cache_t0) * 1000.0, 2),
    )

    # ── YOLO-Pro resolved-resolution guard (blocking failure) ────────────────
    # _try_load_feature_cache verified that the cached dimensions are ≥
    # YOLO_PRO_MIN_INPUT_SIZE, but img_w/img_h above have since been snapped to
    # the nearest multiple of 32 (and possibly clamped to the minimum).  If the
    # post-snap target differs from what the cache stores, the features on disk
    # simply cannot train this model.
    #
    # Policy: fail the job here.  Silently rejecting the cache and recomputing
    # every feature from raw data burned minutes of compute and still ended in
    # upsampled pixels, with nothing shown to the user.  Raising instead lets
    # the training task's terminal handler mark the job failed and render this
    # message on the Training page.
    #
    # Example scenario that triggers this:
    #   Cache built at 136×136  →  _try_load_feature_cache passes (≥96).
    #   img_w = _snap32(136) = 160  →  160 ≠ 136  →  job fails here.
    if _cached is not None:
        _cmeta = _cached["meta"]
        _c_w   = int(_cmeta.get("image_width",  0) or 0)
        _c_h   = int(_cmeta.get("image_height", 0) or 0)
        if _c_w != img_w or _c_h != img_h:
            logger.error(
                f"[YOLO-Pro] [ACTION:FAIL_FEATURE_RESOLUTION_MISMATCH] "
                f"Cached feature resolution {_c_w}×{_c_h} does not match the "
                f"resolved training resolution {img_w}×{img_h} "
                f"(after snap-to-32 and minimum-resolution enforcement). "
                "Failing the job — features are not recomputed, resized or "
                "upsampled."
            )
            raise FeatureResolutionMismatchError(
                _feature_resolution_mismatch_message(_c_w, _c_h, img_w, img_h)
            )

    if _cached is not None:
        X_all          = _cached["X"]
        sample_types_c = _cached["sample_types"]
        boxes_json_arr = _cached["boxes_json"]
        background_arr = _cached.get("background", np.zeros(0, dtype=np.int8))
        n              = len(X_all)

        logger.info(
            f"[YOLO-Pro] Using cached generated features — "
            f"{n} samples, input_shape={input_shape}"
        )

        def _is_test_c(st: str) -> bool:
            return st.lower() in ("testing", "test", "sampletype.testing")

        X_tr_c, b_tr_c, X_te_c, b_te_c = [], [], [], []
        n_shape_rejected = 0
        n_bg_flag_dropped_boxes = 0   # had real box annotations, wiped by the background flag
        n_no_valid_boxes = 0          # had no annotations, or only unlabeled/placeholder ones
        for i in range(n):
            arr = np.asarray(X_all[i], dtype=np.float32)
            # Only reshape flat vectors that have the exact right number of
            # elements.  Do NOT upsample or resize cached tensors that are at
            # a different spatial resolution — that would silently treat low-
            # resolution cached data as valid high-resolution training data.
            # Such samples are skipped; the caller should recompute from raw.
            if arr.shape != input_shape:
                if arr.size == img_h * img_w * n_ch:
                    # Flat vector with the right element count — safe reshape.
                    arr = arr.reshape(input_shape)
                else:
                    # Spatial resolution mismatch that slipped past the whole-cache
                    # guard above.  This path should NOT be reachable for a normal
                    # stale-cache scenario (those fail the job at the
                    # FAIL_FEATURE_RESOLUTION_MISMATCH guard before the loop).
                    # If reached, it indicates cache corruption or a mixed-resolution
                    # cache.  Skip only this sample as a last-resort safeguard.
                    logger.warning(
                        f"[YOLO-Pro] [ACTION:CACHE_SAMPLE_SHAPE_MISMATCH] "
                        f"Cache sample {i}: shape {arr.shape} cannot be reshaped "
                        f"to target {input_shape} — skipping this sample. "
                        "(This is a per-sample corruption guard; stale-resolution "
                        "caches fail the job before the loop. "
                        "Re-run Generate Features if this recurs.)"
                    )
                    n_shape_rejected += 1
                    continue

            boxes_str = str(boxes_json_arr[i]) if i < len(boxes_json_arr) else "[]"
            raw_boxes = json.loads(boxes_str)
            is_bg = bool(i < len(background_arr) and background_arr[i])
            # A negative never contributes GT, whatever the cache holds.
            if is_bg:
                raw_boxes = []
            boxes = []
            for box in raw_boxes:
                key = box.get("label_id") or box.get("label")
                if not key or str(key).lower() in UNLABELED_NAMES:
                    continue
                boxes.append({
                    "label":    key,
                    "label_id": key,
                    "x":        float(box.get("x", 0)),
                    "y":        float(box.get("y", 0)),
                    "w":        float(box.get("w", 0)),
                    "h":        float(box.get("h", 0)),
                })

            if not boxes:
                # Distinguish *why* this sample carries no ground truth, so a
                # cache dominated by one specific cause can say so below
                # instead of listing every possibility.
                orig_str = str(boxes_json_arr[i]) if i < len(boxes_json_arr) else "[]"
                had_real_box = any(
                    (b.get("label_id") or b.get("label"))
                    and str(b.get("label_id") or b.get("label")).lower() not in UNLABELED_NAMES
                    for b in json.loads(orig_str)
                    if isinstance(b, dict)
                )
                if is_bg and had_real_box:
                    n_bg_flag_dropped_boxes += 1
                else:
                    n_no_valid_boxes += 1

            stype = str(sample_types_c[i]) if i < len(sample_types_c) else "training"
            if _is_test_c(stype):
                X_te_c.append(arr)
                b_te_c.append(boxes)
            else:
                X_tr_c.append(arr)
                b_tr_c.append(boxes)

        n_train_pre_cap = len(X_tr_c)
        n_train_positive_pre_cap = sum(1 for b in b_tr_c if b)

        X_tr_c, b_tr_c = _cap_detection_negatives(
            X_tr_c, b_tr_c, _neg_cap,
            seed=f"{getattr(impulse, 'id', '?')}|yolo-cache",
            prefix="[YOLO-Pro] (cache)",
        )

        if not X_tr_c:
            n_test = len(X_te_c)
            if n_train_pre_cap == 0:
                cause = (
                    f"all {n} cached samples were assigned to the test split "
                    f"({n_test} test) or rejected for shape corruption "
                    f"({n_shape_rejected})."
                )
            elif n_train_positive_pre_cap == 0:
                cause = (
                    f"{n_train_pre_cap} training samples were found in the cache, "
                    f"but 0 have a usable bounding-box annotation — "
                    f"{n_bg_flag_dropped_boxes} have real box annotations that were "
                    f"discarded because the sample is marked as a background/negative "
                    f"image (clear the background marker on the Data page if that's "
                    f"wrong), and {n_no_valid_boxes} have no annotation or only "
                    f"unlabeled/placeholder boxes. With zero positives, the "
                    f"negative-sample cap ({_neg_cap:.0%}) removed all "
                    f"{n_train_pre_cap} of them rather than training on negatives "
                    f"alone."
                )
            else:
                cause = (
                    f"{n_train_pre_cap} training samples were found with "
                    f"{n_train_positive_pre_cap} positives, but the negative-sample "
                    f"cap ({_neg_cap:.0%}) still removed every one of them — this "
                    f"should not happen; please report it."
                )
            raise ValueError(
                f"No usable YOLO-Pro training samples found in feature cache after "
                f"filtering: {cause} Re-run Generate Features to rebuild the cache "
                f"if the underlying sample data has since changed."
            )

        logger.info(
            f"[YOLO-Pro] Cache dataset ready — "
            f"{len(X_tr_c)} train / {len(X_te_c)} test  input_shape={input_shape}"
        )
        return (
            np.stack(X_tr_c),
            b_tr_c,
            np.stack(X_te_c) if X_te_c else np.empty((0, *input_shape), dtype=np.float32),
            b_te_c,
            label_names,
            label_map,
            input_shape,
        )
    # ── Cache miss: fall back to raw-sample download + DSP ────────────────────

    # Same fail-fast as the cache branch above.  img_w/img_h were snapped to a
    # multiple of 32 regardless of whether a cache was hit, but the raw path
    # feeds DSPProcessor the *unsnapped* merged image params — so it generates
    # at dsp_gen_w×dsp_gen_h and the per-sample shape correction below would
    # tf.image.resize every sample up to the snapped target.  That is the same
    # silent upsampling the cache branch now refuses, so refuse it here too,
    # before downloading a single sample.
    if dsp_gen_w != img_w or dsp_gen_h != img_h:
        logger.error(
            f"[YOLO-Pro] [ACTION:FAIL_FEATURE_RESOLUTION_MISMATCH] "
            f"DSP image block generates features at {dsp_gen_w}×{dsp_gen_h}, "
            f"which does not match the resolved training resolution "
            f"{img_w}×{img_h} (after snap-to-32). "
            "Failing the job — features are not resized or upsampled."
        )
        raise FeatureResolutionMismatchError(
            _feature_resolution_mismatch_message(dsp_gen_w, dsp_gen_h, img_w, img_h)
        )

    X_tr, b_tr, X_te, b_te = [], [], [], []

    for sample, raw, fetch_error in iter_prefetched(
        samples, lambda s: storage.download_bytes(s.storage_key)
    ):
        if fetch_error is not None:
            logger.warning(f"[YOLO-Pro] Skipping sample {sample.id}: download failed — {fetch_error}")
            continue
        try:
            all_features = []
            for block_cfg in dsp_blocks:
                params = merge_image_params(impulse, block_cfg)
                proc = DSPProcessor(
                    block_type=block_cfg.get("type", "raw"),
                    params=params,
                    frequency_hz=sample.frequency_hz or impulse.frequency_hz or 100.0,
                )
                all_features.append(proc.extract(raw))
        except Exception as e:
            logger.warning(f"[YOLO-Pro] Skipping sample {sample.id}: DSP failed — {e}")
            continue

        arr = np.asarray(_combine_dsp_features(all_features), dtype=np.float32)

        # Reshape to (H, W, C) if the DSP returned a flat vector
        if arr.ndim == 1:
            expected_flat = img_h * img_w * n_ch
            if arr.size != expected_flat:
                logger.warning(
                    f"[YOLO-Pro] Skipping sample {sample.id}: "
                    f"flat DSP output size {arr.size} != expected {expected_flat}"
                )
                continue
            arr = arr.reshape(input_shape)
        elif arr.shape != input_shape:
            # Try TF resize as a best-effort correction (e.g. 321×321 → 320×320)
            try:
                arr = tf.image.resize(arr, [img_h, img_w]).numpy()
                if arr.shape != input_shape:
                    continue
            except Exception:
                logger.warning(
                    f"[YOLO-Pro] Skipping sample {sample.id}: "
                    f"shape {arr.shape} != target {input_shape}"
                )
                continue

        # Extract bounding boxes. A negative never contributes GT, whatever
        # boxes may still be sitting in its metadata.
        boxes = []
        if isinstance(sample.extra_metadata, dict) and not is_background_sample(sample):
            raw_boxes = normalize_bounding_boxes(
                sample.extra_metadata.get("boundingBoxes", []),
                image_bytes=raw,
            )
            for box in raw_boxes:
                key = box.get("label_id") or box.get("label")
                if not key or str(key).lower() in UNLABELED_NAMES:
                    continue
                boxes.append({
                    "label":    key,
                    "label_id": key,
                    "x":        float(box.get("x", 0)),
                    "y":        float(box.get("y", 0)),
                    "w":        float(box.get("w", 0)),
                    "h":        float(box.get("h", 0)),
                })

        _is_test_sample = (
            getattr(sample, "sample_type", None) == SampleType.testing
            or getattr(sample, "category",    None) == "testing"
        )
        if _is_test_sample:
            X_te.append(arr)
            b_te.append(boxes)
        else:
            X_tr.append(arr)
            b_tr.append(boxes)

    X_tr, b_tr = _cap_detection_negatives(
        X_tr, b_tr, _neg_cap,
        seed=f"{getattr(impulse, 'id', '?')}|yolo-raw",
        prefix="[YOLO-Pro] (raw)",
    )

    if not X_tr:
        raise ValueError(
            "No usable YOLO-Pro training samples found. "
            "Ensure samples have bounding-box annotations and the impulse "
            f"image block is configured (current: {img_w}\u00d7{img_h}\u00d7{n_ch})."
        )

    logger.info(
        f"[YOLO-Pro] Dataset ready — "
        f"{len(X_tr)} train / {len(X_te)} test samples  "
        f"input_shape={input_shape}"
    )

    return (
        np.stack(X_tr),
        b_tr,
        np.stack(X_te) if X_te else np.empty((0, *input_shape), dtype=np.float32),
        b_te,
        label_names,
        label_map,
        input_shape,
    )


# ─── MobileNetV2 SSD dataset loader ───────────────────────────────────────────

def _load_ssd_dataset(db, impulse, extra_params: Optional[dict] = None):
    """
    Load image samples + bounding-box annotations for MobileNetV2 SSD training.

    Resolution is ALWAYS fixed at MV2_SSD_INPUT_H × MV2_SSD_INPUT_W (320×320×3)
    regardless of the impulse DSP config.  Raw images are downloaded and resized
    directly — the YOLO-Pro feature cache is NOT used.

    Returns same tuple as _load_yolo_dataset:
        X_train, boxes_train, X_test, boxes_test, label_names, label_map, input_shape
    """
    H, W, C = MV2_SSD_INPUT_H, MV2_SSD_INPUT_W, 3
    input_shape = (H, W, C)

    all_labels = db.query(Label).filter(Label.project_id == impulse.project_id).all()
    usable_label_ids = {l.id for l in all_labels if not label_name_is_placeholder(l.name)}

    samples = (
        db.query(Sample)
        .filter(Sample.project_id == impulse.project_id)
        .all()
    )
    samples = [
        s for s in samples
        if is_sample_usable(s, usable_label_ids, allow_background=True)
    ]
    _neg_cap = _resolve_negative_ratio_cap(impulse, extra_params)

    # Background samples are skipped here for the same reason as in the YOLO
    # loader: a negative must not mint an output class out of a stale label_id.
    active: set = set()
    for s in samples:
        if is_background_sample(s):
            continue
        if isinstance(s.extra_metadata, dict):
            for box in s.extra_metadata.get("boundingBoxes", []):
                key = box.get("label_id") or box.get("label")
                if key and str(key).lower() not in UNLABELED_NAMES:
                    active.add(key)
        if s.label_id and s.label_id in usable_label_ids:
            active.add(s.label_id)

    labels = [
        l for l in all_labels
        if (l.id in active or l.name in active)
        and not label_name_is_placeholder(l.name)
    ]
    label_map: dict = {}
    for idx, l in enumerate(labels):
        label_map[l.id]   = idx
        label_map[l.name] = idx
    label_names = [l.name for l in labels]

    logger.info(
        f"[SSD] Dataset load — fixed input_shape={input_shape}  "
        f"labels={label_names}  samples={len(samples)}"
    )

    X_tr, b_tr, X_te, b_te = [], [], [], []

    for sample, raw, fetch_error in iter_prefetched(
        samples, lambda s: storage.download_bytes(s.storage_key)
    ):
        if fetch_error is not None:
            logger.warning(f"[SSD] Skipping sample {sample.id}: download failed — {fetch_error}")
            continue

        try:
            import io as _io
            from PIL import Image as _Image
            img = _Image.open(_io.BytesIO(raw)).convert("RGB").resize((W, H))
            arr = np.asarray(img, dtype=np.float32)          # (320, 320, 3)
        except Exception as e:
            logger.warning(f"[SSD] Skipping sample {sample.id}: decode/resize failed — {e}")
            continue

        # A negative never contributes GT, whatever boxes may still be sitting
        # in its metadata.
        boxes = []
        if isinstance(sample.extra_metadata, dict) and not is_background_sample(sample):
            raw_boxes = normalize_bounding_boxes(
                sample.extra_metadata.get("boundingBoxes", []),
                image_bytes=raw,
            )
            for box in raw_boxes:
                key = box.get("label_id") or box.get("label")
                if not key or str(key).lower() in UNLABELED_NAMES:
                    continue
                boxes.append({
                    "label":    key,
                    "label_id": key,
                    "x":        float(box.get("x", 0)),
                    "y":        float(box.get("y", 0)),
                    "w":        float(box.get("w", 0)),
                    "h":        float(box.get("h", 0)),
                })

        _is_test_sample = (
            getattr(sample, "sample_type", None) == SampleType.testing
            or getattr(sample, "category", None) == "testing"
        )
        if _is_test_sample:
            X_te.append(arr)
            b_te.append(boxes)
        else:
            X_tr.append(arr)
            b_tr.append(boxes)

    X_tr, b_tr = _cap_detection_negatives(
        X_tr, b_tr, _neg_cap,
        seed=f"{getattr(impulse, 'id', '?')}|ssd",
        prefix="[SSD]",
    )

    if not X_tr:
        raise ValueError(
            "No usable SSD training samples found. "
            "Ensure samples have bounding-box annotations."
        )

    logger.info(
        f"[SSD] Dataset ready — "
        f"{len(X_tr)} train / {len(X_te)} test  input_shape={input_shape}"
    )
    return (
        np.stack(X_tr),
        b_tr,
        np.stack(X_te) if X_te else np.empty((0, *input_shape), dtype=np.float32),
        b_te,
        label_names,
        label_map,
        input_shape,
    )
