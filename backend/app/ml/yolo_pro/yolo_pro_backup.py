"""
app/ml/yolo_pro_worker.py
─────────────────────────
YOLO-Pro training pipeline — invoked by training_worker.py when
``selected_architecture == "yolo_pro"``.

Key improvements over the original:
  1. Task-Aligned Assigner (TAL) with IoU×cls score weighting instead of
     naive point-in-box with static 1.0 labels.
  2. Varifocal targets use actual IoU quality scores (not 1.0).
  3. Stride decoding in _train_step uses per-scale anchor arrays correctly.
  4. Gradient clipping added to prevent training instability.
  5. EMA decay adapts to training length to avoid shadow collapse.
  6. Per-class NMS in evaluation.
  7. Confidence threshold raised to 0.01 at eval for cleaner AP curves.
  8. Standalone _decode_reg_pred fixed (undefined ax/ay bug removed).
"""

from __future__ import annotations

import io
import os
import queue
import logging
import tempfile
import threading
import time
import numpy as np
import tensorflow as tf
from datetime import datetime
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

if TYPE_CHECKING:
    from sqlalchemy.orm import Session
    from app.models.user import TrainingJob, Impulse

import math

from app.workers.cancel_utils import (
    CancelledError as _CancelledError,
    raise_if_cancelled as _raise_if_cancelled,
    raise_if_cancelled_fast as _raise_if_cancelled_fast,
    assert_not_cancelled_before_completing as _assert_not_cancelled_before_completing,
    promote_run_to_active as _promote_run_to_active,
    register_live_buffer as _register_live_buffer,
)

logger = logging.getLogger(__name__)

YOLO_PRO_ARCHITECTURE = "yolo_pro"


# ─────────────────────────────────────────────────────────────────────────────
# JSON safety helpers
# ─────────────────────────────────────────────────────────────────────────────

def _sanitize_json(obj):
    """Recursively replace non-finite floats (nan, inf, -inf) with None.

    PostgreSQL's JSON/JSONB columns reject IEEE 754 non-finite values.
    Call this on any dict/list before assigning to a SQLAlchemy JSON column.
    """
    if isinstance(obj, float):
        return None if not math.isfinite(obj) else obj
    if isinstance(obj, dict):
        return {k: _sanitize_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_json(v) for v in obj]
    return obj


# ─────────────────────────────────────────────────────────────────────────────
# IoU utilities
# ─────────────────────────────────────────────────────────────────────────────

def _box_iou_matrix(pred_boxes: np.ndarray, gt_boxes: np.ndarray) -> np.ndarray:
    """
    Compute IoU between every predicted box and every GT box.

    Args:
        pred_boxes : (P, 4) float32  x1,y1,x2,y2 normalised [0,1]
        gt_boxes   : (G, 4) float32  x1,y1,x2,y2 normalised [0,1]
    Returns:
        iou_matrix : (P, G) float32
    """
    if len(pred_boxes) == 0 or len(gt_boxes) == 0:
        return np.zeros((len(pred_boxes), len(gt_boxes)), dtype=np.float32)

    px1, py1, px2, py2 = (pred_boxes[:, i] for i in range(4))
    gx1, gy1, gx2, gy2 = (gt_boxes[:, i] for i in range(4))

    inter_x1 = np.maximum(px1[:, None], gx1[None, :])
    inter_y1 = np.maximum(py1[:, None], gy1[None, :])
    inter_x2 = np.minimum(px2[:, None], gx2[None, :])
    inter_y2 = np.minimum(py2[:, None], gy2[None, :])

    inter_w = np.maximum(0.0, inter_x2 - inter_x1)
    inter_h = np.maximum(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    pred_area = np.maximum(px2 - px1, 0.0) * np.maximum(py2 - py1, 0.0)
    gt_area   = np.maximum(gx2 - gx1, 0.0) * np.maximum(gy2 - gy1, 0.0)
    union_area = pred_area[:, None] + gt_area[None, :] - inter_area

    return np.where(union_area > 0, inter_area / union_area, 0.0).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Decode and NMS helpers
# ─────────────────────────────────────────────────────────────────────────────

def _decode_yolo_pro_predictions(
    raw_preds: dict,
    anchor_grids: List[Tuple[np.ndarray, int]],
    H: int,
    W: int,
    reg_max: int,
    conf_threshold: float = 0.01,
) -> List[dict]:
    """
    Decode raw YOLO-Pro model outputs for a single image into detection dicts.

    Thin wrapper around decode_raw_outputs_np() — the canonical batch decode
    implementation in yolo_pro/decode.py.  All DFL and anchor math now lives
    there so export, evaluation, and runtime use an identical code path.

    Args:
        raw_preds     : dict {cls_p3, reg_p3, cls_p4, reg_p4, cls_p5, reg_p5}
        anchor_grids  : kept for call-site compatibility; no longer used
                        (decode_raw_outputs_np derives anchors from H/W/strides)
        H, W          : input image pixel dims
        reg_max       : DFL bins
        conf_threshold: minimum class score to emit a detection

    Returns:
        list of {'class_idx': int, 'score': float, 'box': [x1,y1,x2,y2]}
    """
    from app.ml.yolo_pro.decode import decode_raw_outputs_np
    from app.ml.variant_eval import DECODED_ROWS_KEY

    if DECODED_ROWS_KEY in raw_preds:
        # Already-decoded (B, N_total, 6) rows, handed over by the
        # decoded_float32 TFLite variant.  DecodeDetectionsLayer (baked into
        # that export) mirrors decode_raw_outputs_np exactly — same normalized
        # xyxy, same reduce_max score — so its rows drop straight in here
        # rather than being decoded a second time.
        all_dets = np.asarray(raw_preds[DECODED_ROWS_KEY], dtype=np.float32)
    else:
        # Batch decode all anchors — shape (1, N_total, 6)
        all_dets = decode_raw_outputs_np(raw_preds, H=H, W=W, reg_max=reg_max)
    dets_np  = all_dets[0]   # (N_total, 6) — first (and only) image in batch

    detections = []
    for row in dets_np:
        x1, y1, x2, y2, score, class_id = (
            float(row[0]), float(row[1]), float(row[2]), float(row[3]),
            float(row[4]), int(row[5]),
        )
        if score <= conf_threshold:
            continue
        if x2 <= x1 or y2 <= y1:
            continue
        detections.append({
            "class_idx": class_id,
            "score":     score,
            "box":       [x1, y1, x2, y2],
        })
    return detections


# ─────────────────────────────────────────────────────────────────────────────
# Numpy DFL decode — used for dynamic TAL assignment (no TF grad needed)
# ─────────────────────────────────────────────────────────────────────────────

def _decode_reg_np(
    reg_flat: np.ndarray,
    anchor_pts: np.ndarray,
    stride: float,
    H_img: int,
    W_img: int,
    reg_max: int,
) -> np.ndarray:
    """
    Decode a batch of DFL logits → normalised (x1,y1,x2,y2) pred boxes.

    Pure numpy — runs outside tf.GradientTape for dynamic TAL assignment.
    Mirrors the TF logic in _decode_reg_scale exactly.

    Args:
        reg_flat   : (B, N, 4*reg_max)  raw DFL logits from model output
        anchor_pts : (N, 2)             anchor centres in pixel coords
        stride     : scale stride (8, 16, or 32)
        H_img, W_img : input image dims in pixels
        reg_max    : DFL bins
    Returns:
        pred_boxes : (B, N, 4)  float32  normalised [0, 1]  x1y1x2y2
    """
    B, N, _ = reg_flat.shape
    logits   = reg_flat.reshape(B, N, 4, reg_max)
    logits   = logits - logits.max(axis=-1, keepdims=True)   # numerical stability
    probs    = np.exp(logits)
    probs   /= probs.sum(axis=-1, keepdims=True) + 1e-9
    bins     = np.arange(reg_max, dtype=np.float32)
    ltrb_grid = (probs * bins).sum(axis=-1)                  # (B, N, 4) grid units
    ltrb_px   = ltrb_grid * float(stride)                   # → pixels

    ax = anchor_pts[:, 0]                                    # (N,) broadcast with (B,N)
    ay = anchor_pts[:, 1]
    x1 = np.clip((ax - ltrb_px[:, :, 0]) / float(W_img), 0., 1.)
    y1 = np.clip((ay - ltrb_px[:, :, 1]) / float(H_img), 0., 1.)
    x2 = np.clip((ax + ltrb_px[:, :, 2]) / float(W_img), 0., 1.)
    y2 = np.clip((ay + ltrb_px[:, :, 3]) / float(H_img), 0., 1.)
    return np.stack([x1, y1, x2, y2], axis=-1).astype(np.float32)   # (B, N, 4)


def _nms(detections, iou_threshold=0.45, max_dets=100, class_agnostic=False):
    """Greedy NMS.

    class_agnostic=False (default): suppress duplicates only within each class
        bucket — COCO/YOLO standard, preserves per-class AP curves during
        training evaluation.
    class_agnostic=True: treat all detections as one pool; a high-IoU box is
        suppressed regardless of its predicted class.  Use this for inference
        display and model testing so a single object that fires as both 'bus'
        and 'car' collapses to the single highest-confidence prediction.
    """
    if not detections:
        return []

    def _greedy(candidates: list) -> list:
        result: list = []
        while candidates:
            best = candidates[0]
            result.append(best)
            candidates = candidates[1:]
            if not candidates:
                break
            best_box   = np.array([best["box"]], dtype=np.float32)
            rest_boxes = np.array([d["box"] for d in candidates], dtype=np.float32)
            ious       = _box_iou_matrix(best_box, rest_boxes)[0]
            candidates = [d for d, iou in zip(candidates, ious) if iou < iou_threshold]
        return result

    if class_agnostic:
        candidates = sorted(detections, key=lambda d: d["score"], reverse=True)
        kept = _greedy(candidates)
        kept.sort(key=lambda d: d["score"], reverse=True)
        return kept[:max_dets]

    by_class: dict[int, list] = {}
    for d in detections:
        by_class.setdefault(d["class_idx"], []).append(d)

    kept: list[dict] = []
    for cls_dets in by_class.values():
        candidates = sorted(cls_dets, key=lambda d: d["score"], reverse=True)
        kept.extend(_greedy(candidates))

    kept.sort(key=lambda d: d["score"], reverse=True)
    return kept[:max_dets]

def _compute_ap_at_iou(
    pred_scores: np.ndarray,
    pred_match:  np.ndarray,
    n_gt: int,
) -> float:
    """
    Compute Average Precision (AP) from a sorted match vector.

    Uses 101-point interpolation (VOC2010/COCO style).
    """
    if n_gt == 0:
        return 0.0
    tp = np.cumsum(pred_match.astype(np.float32))
    fp = np.cumsum((~pred_match).astype(np.float32))
    precision = tp / (tp + fp + 1e-9)
    recall    = tp / (n_gt + 1e-9)

    ap = 0.0
    for t in np.linspace(0.0, 1.0, 101):
        mask = recall >= t
        ap += precision[mask].max() if mask.any() else 0.0
    return float(ap / 101.0)


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_yolo_pro_detection(
    model: "tf.keras.Model",
    X_test: np.ndarray,
    boxes_test: List[Optional[List[dict]]],
    label_names: List[str],
    label_map: dict,
    anchor_grids: List[Tuple[np.ndarray, int]],
    H: int,
    W: int,
    reg_max: int,
    conf_threshold: float = 0.25,
    iou_threshold_match: float = 0.50,
    job_id: str = "",
    runtime_conf_threshold: "float | None" = None,
) -> dict:
    """
    Run post-training detection evaluation for a YOLO-Pro model.

    Computes mAP50, mAP (COCO), mAP75, precision, recall, per_class AP.

    Args:
        conf_threshold : minimum sigmoid class score for a detection to enter the
                         AP computation.  0.005 is the evaluation default:
                         • mAP is the area under the full precision-recall curve.
                           A high threshold (e.g. 0.25) silently truncates recall
                           by discarding true positives below that score, collapsing
                           AP to 0.0 on under-trained or small models.
                         • 0.005 passes predictions just above the background prior
                           (~0.01 after sigmoid) while class-agnostic NMS
                           (iou_threshold=0.45, max_dets=100) eliminates duplicate
                           cross-class boxes before AP matching.
                         • This matches standard COCO-API evaluation practice.
                         • This parameter affects evaluation ONLY.  The exported
                           TFLite model emits all N_total anchors; runtime or
                           firmware applies its own deployment threshold downstream.

        runtime_conf_threshold :
            Optional deployment/runtime confidence threshold.  When supplied
            the report carries a SECOND prediction-count metric —
            ``runtime_total_predicted_cells`` and ``runtime_conf_threshold``
            — derived by filtering the post-NMS detection set by score >
            runtime_conf_threshold.  This is what the prediction-flood
            safety gate consumes, decoupling it from the permissive AP eval
            threshold (which can legitimately produce many low-confidence
            detections on a healthy ranked model).  Does NOT affect mAP,
            precision, recall, or any ranking metric — those continue to be
            computed at ``conf_threshold``.
    """
    n_classes = len(label_names)
    logger.info(
        f"[{job_id}] YOLO-Pro eval — {len(X_test)} test images, "
        f"{n_classes} classes, IoU@{iou_threshold_match}, conf={conf_threshold}"
    )

    if len(X_test) == 0 or not any(b for b in boxes_test if b):
        logger.warning(f"[{job_id}] YOLO-Pro eval: no test data — skipping.")
        return {
            "yolo_pro_eval_status": "no_data",
            "yolo_pro_eval_error":  "Test set is empty or has no annotated boxes.",
            "map": None, "map50": None, "map75": None,
            "precision": None, "recall": None, "per_class": {},
            "evaluated_images": int(len(X_test)),
        }

    all_preds = [[] for _ in range(n_classes)]
    gt_count  = np.zeros(n_classes, dtype=np.int32)

    IOU_THRESHOLDS_COCO = np.linspace(0.50, 0.95, 10)
    all_preds_multi = [[[] for _ in range(n_classes)] for _ in IOU_THRESHOLDS_COCO]
    per_image_data: List[dict] = []

    try:
        for img_idx, (img, boxes) in enumerate(zip(X_test, boxes_test)):
            if img_idx % 20 == 0:
                _raise_if_cancelled(job_id)
            img_tensor = np.expand_dims(img, 0).astype(np.float32)

            try:
                raw_preds = model(img_tensor, training=False)
                raw_preds_np = {
                    k: v.numpy() if hasattr(v, "numpy") else np.array(v)
                    for k, v in raw_preds.items()
                }
            except Exception as e:
                logger.warning(f"[{job_id}] eval img {img_idx}: inference failed: {e}")
                continue

            dets = _decode_yolo_pro_predictions(
                raw_preds_np, anchor_grids, H, W, reg_max, conf_threshold
            )
            # NMS: class-agnostic suppression aligned with runtime/model testing.
            # iou_threshold=0.45 is the standard YOLO/COCO NMS value.
            # max_dets=100 matches COCO primary mAP evaluation (AR@maxDets=100).
            dets = _nms(
                dets,
                iou_threshold=0.45,
                max_dets=100,
                class_agnostic=True,
            )

            if img_idx < 3:
                n_gt_boxes = sum(1 for b in (boxes or []) if isinstance(b, dict))
                logger.info(
                    f"[{job_id}] eval img {img_idx}: "
                    f"{len(dets)} dets after NMS, gt_boxes={n_gt_boxes}"
                )

            # Build GT structure
            gt_boxes_by_class: dict[int, List[np.ndarray]] = {}
            gt_boxes_all: List[np.ndarray] = []
            for box in (boxes or []):
                _lid = box.get("label_id")
                raw_label = _lid if _lid is not None else box.get("label")
                cls_idx = label_map.get(raw_label)
                if cls_idx is None:
                    continue
                bx = float(box.get("x", 0)); by = float(box.get("y", 0))
                bw = float(box.get("w", 0)); bh = float(box.get("h", 0))
                x1, y1, x2, y2 = bx, by, bx + bw, by + bh
                if x2 <= x1 or y2 <= y1:
                    continue
                gt_count[cls_idx] += 1
                gt_arr = np.array([x1, y1, x2, y2], dtype=np.float32)
                gt_boxes_by_class.setdefault(cls_idx, []).append(gt_arr)
                gt_boxes_all.append(gt_arr)

            if img_idx < 3 and dets:
                best_iou = 0.0
                if gt_boxes_all:
                    for det in dets:
                        pb = np.array([det["box"]], dtype=np.float32)
                        ga = np.array(gt_boxes_all, dtype=np.float32)
                        best_iou = max(best_iou, float(_box_iou_matrix(pb, ga)[0].max()))
                logger.info(
                    f"[{job_id}] eval img {img_idx}: best_any_cls_iou={best_iou:.4f}"
                )

            # Collect per-image data for extended COCO metrics
            _img_gt_areas = []
            for _c, _gts in gt_boxes_by_class.items():
                for _g in _gts:
                    _img_gt_areas.append({
                        "class_idx": _c,
                        "area": float((_g[2] - _g[0]) * (_g[3] - _g[1])),
                    })
            per_image_data.append({
                "dets": sorted(dets, key=lambda d: d["score"], reverse=True),
                "gt_by_class": {
                    _c: [_g.tolist() for _g in _gts]
                    for _c, _gts in gt_boxes_by_class.items()
                },
                "gt_areas": _img_gt_areas,
            })

            # Match predictions at each IoU threshold
            for iou_idx, iou_thr in enumerate(IOU_THRESHOLDS_COCO):
                is_primary = abs(iou_thr - iou_threshold_match) < 0.001
                gt_used: dict[int, set] = {c: set() for c in gt_boxes_by_class}

                for det in dets:
                    c     = det["class_idx"]
                    score = det["score"]
                    if c >= n_classes:
                        continue

                    pred_box = np.array([det["box"]], dtype=np.float32)
                    gts_c    = gt_boxes_by_class.get(c, [])
                    matched  = False

                    if gts_c:
                        gt_arr  = np.array(gts_c, dtype=np.float32)
                        ious    = _box_iou_matrix(pred_box, gt_arr)[0]
                        best_gt = int(np.argmax(ious))
                        if ious[best_gt] >= iou_thr and best_gt not in gt_used.get(c, set()):
                            matched = True
                            gt_used.setdefault(c, set()).add(best_gt)

                    all_preds_multi[iou_idx][c].append((score, matched))
                    if is_primary:
                        all_preds[c].append((score, matched))

    except _CancelledError:
        raise
    except Exception as e:
        logger.error(f"[{job_id}] YOLO-Pro eval: unexpected error: {e}", exc_info=True)
        return {
            "yolo_pro_eval_status": "inference_failed",
            "yolo_pro_eval_error":  str(e),
            "map": None, "map50": None, "map75": None,
            "precision": None, "recall": None, "per_class": {},
            "evaluated_images": int(len(X_test)),
        }

    total_preds = sum(len(p) for p in all_preds)
    all_primary_scores = [
        float(score)
        for preds_c in all_preds
        for score, _matched in preds_c
    ]
    avg_confidence = (
        float(np.mean(all_primary_scores))
        if all_primary_scores else None
    )

    # TP/FP score-gap diagnostic — key signal for score-separation health.
    # After VFL quality-floor removal, TPs (IoU-quality trained) should score
    # clearly above FPs (near-negative, score ≈ prior 0.01).
    # Log the mean TP score, mean FP score, and gap so every eval run captures
    # whether the fix is holding or regressing.
    # Cross-class duplicate suppression happens before this diagnostic, so the
    # TP/FP gap now reflects residual ranking quality instead of NMS artifacts.
    _tp_scores = [s for preds_c in all_preds for s, m in preds_c if m]
    _fp_scores = [s for preds_c in all_preds for s, m in preds_c if not m]
    _mean_tp   = float(np.mean(_tp_scores)) if _tp_scores else None
    _mean_fp   = float(np.mean(_fp_scores)) if _fp_scores else None
    _score_gap = (
        round(_mean_tp - _mean_fp, 6) if (_mean_tp is not None and _mean_fp is not None) else None
    )

    total_gt    = int(gt_count.sum())
    logger.info(
        f"[{job_id}] eval done — total_gt={total_gt}  total_preds_after_nms={total_preds}  "
        f"tp_score_mean={_mean_tp:.4f}  fp_score_mean={_mean_fp:.4f}  "
        f"tp_fp_gap={_score_gap:.4f}"
        if (_mean_tp is not None and _mean_fp is not None)
        else f"[{job_id}] eval done — total_gt={total_gt}  total_preds_after_nms={total_preds}  "
             f"(no TP/FP scores to report)"
    )
    if total_preds == 0:
        logger.warning(
            f"[{job_id}] Zero predictions survived decode+NMS. "
            f"Check conf_threshold ({conf_threshold}), model convergence, ltrb scaling."
        )

    # ── Per-class AP at primary IoU threshold ─────────────────────────────────
    per_class_ap   = {}
    per_class_prec = {}
    per_class_rec  = {}

    for c, cls_name in enumerate(label_names):
        preds_c = sorted(all_preds[c], key=lambda x: x[0], reverse=True)
        if not preds_c:
            per_class_ap[cls_name]   = 0.0
            per_class_prec[cls_name] = 0.0
            per_class_rec[cls_name]  = 0.0
            continue
        scores  = np.array([p[0] for p in preds_c])
        matches = np.array([p[1] for p in preds_c])
        ap = _compute_ap_at_iou(scores, matches, int(gt_count[c]))
        per_class_ap[cls_name] = round(float(ap), 6)

        tp_cum  = np.cumsum(matches.astype(np.float32))
        fp_cum  = np.cumsum((~matches).astype(np.float32))
        rec_arr = tp_cum / (int(gt_count[c]) + 1e-9)
        pre_arr = tp_cum / (tp_cum + fp_cum + 1e-9)
        f1_arr  = 2 * pre_arr * rec_arr / (pre_arr + rec_arr + 1e-9)
        best_i  = int(np.argmax(f1_arr)) if len(f1_arr) > 0 else 0
        per_class_prec[cls_name] = round(float(pre_arr[best_i]), 6) if len(pre_arr) else 0.0
        per_class_rec[cls_name]  = round(float(rec_arr[best_i]),  6) if len(rec_arr) else 0.0

    map50 = float(np.mean(list(per_class_ap.values()))) if per_class_ap else None
    mean_prec = float(np.mean(list(per_class_prec.values()))) if per_class_prec else None
    mean_rec  = float(np.mean(list(per_class_rec.values()))) if per_class_rec else None

    # ── mAP@0.50:0.95 and mAP@75 ─────────────────────────────────────────────
    ap_per_iou_per_class = []
    ap75_per_class = []

    for iou_idx, iou_thr in enumerate(IOU_THRESHOLDS_COCO):
        aps_at_thr = []
        for c, cls_name in enumerate(label_names):
            preds_c = sorted(all_preds_multi[iou_idx][c], key=lambda x: x[0], reverse=True)
            if not preds_c:
                aps_at_thr.append(0.0)
                continue
            scores  = np.array([p[0] for p in preds_c])
            matches = np.array([p[1] for p in preds_c])
            aps_at_thr.append(_compute_ap_at_iou(scores, matches, int(gt_count[c])))
        ap_per_iou_per_class.append(aps_at_thr)
        if abs(iou_thr - 0.75) < 0.001:
            ap75_per_class = aps_at_thr

    map_coco = float(np.mean(ap_per_iou_per_class)) if ap_per_iou_per_class else None
    map75    = float(np.mean(ap75_per_class))        if ap75_per_class else None

    per_class_out = {
        cls_name: {
            "ap":        per_class_ap.get(cls_name),
            "precision": per_class_prec.get(cls_name),
            "recall":    per_class_rec.get(cls_name),
        }
        for cls_name in label_names
    }

    logger.info(
        f"[{job_id}] YOLO-Pro eval complete — "
        f"mAP50={map50:.4f}  mAP={map_coco:.4f}  mAP75={map75:.4f}  "
        f"P={mean_prec:.4f}  R={mean_rec:.4f}"
    )

    _extra = _compute_extra_coco_metrics(
        per_image_data, n_classes, H, W, IOU_THRESHOLDS_COCO
    )

    # ── Runtime-threshold prediction count (flood-safety metric) ──────────────
    # Separate from the AP-threshold count above: this filters the post-NMS
    # detection set by score > runtime_conf_threshold so the flood gate is
    # measured at the same threshold the deployed model actually applies.
    # Filtering the AP NMS output is mathematically equivalent to re-running
    # NMS on `score > runtime_conf_threshold` detections, because NMS is
    # greedy-by-score and any detection that NMS@conf=0.005 kept would also
    # have been kept by NMS@runtime_conf (suppression-by-higher-scoring is
    # monotone in the conf filter).
    _runtime_total_preds: Optional[int] = None
    if runtime_conf_threshold is not None and math.isfinite(float(runtime_conf_threshold)):
        _runtime_total_preds = int(sum(
            1
            for _img in per_image_data
            for _d in _img["dets"]
            if float(_d["score"]) > float(runtime_conf_threshold)
        ))
        _rt_ppi_log = (
            _runtime_total_preds / max(1, len(per_image_data))
        )
        logger.info(
            f"[{job_id}] runtime preds-per-image flood metric: "
            f"runtime_conf_threshold={float(runtime_conf_threshold):.4f}  "
            f"runtime_total_predicted_cells={_runtime_total_preds}  "
            f"runtime_preds_per_image={_rt_ppi_log:.2f}  "
            f"(AP eval threshold was {conf_threshold:.4f} with "
            f"total_predicted_cells={int(total_preds)})"
        )

    return {
        "yolo_pro_eval_status": "success",
        "yolo_pro_eval_error":  None,
        "map":       round(float(map_coco), 6) if map_coco is not None else None,
        "map50":     round(float(map50),    6) if map50    is not None else None,
        "map75":     round(float(map75),    6) if map75    is not None else None,
        "precision": round(float(mean_prec), 6) if mean_prec is not None else None,
        "recall":    round(float(mean_rec),  6) if mean_rec  is not None else None,
        "avg_confidence": round(float(avg_confidence), 6) if avg_confidence is not None else None,
        "tp_score_mean":  round(_mean_tp,   6) if _mean_tp   is not None else None,
        "fp_score_mean":  round(_mean_fp,   6) if _mean_fp   is not None else None,
        "tp_fp_score_gap": _score_gap,
        "total_predicted_cells": int(total_preds),
        # Runtime-threshold prediction count — consumed by the flood safety
        # gate.  None when no runtime threshold was supplied (callers that
        # only want AP/mAP can omit the parameter and retain the legacy
        # AP-only fields).
        "runtime_conf_threshold":      (
            float(runtime_conf_threshold)
            if runtime_conf_threshold is not None
            and math.isfinite(float(runtime_conf_threshold))
            else None
        ),
        "runtime_total_predicted_cells": _runtime_total_preds,
        "gt_cells": total_gt,
        # Denominator for the preds-per-image safety gate
        # (_checkpoint_quality_gate_failures).  Set to the number of test
        # images we attempted to evaluate so the gate handles the safety
        # eval (32-image subset) and the final eval identically.
        "evaluated_images": int(len(X_test)),
        "per_class": per_class_out,
        "detailed_metrics": {
            "map":              round(float(map_coco), 6) if map_coco is not None else None,
            "map50":            round(float(map50),    6) if map50    is not None else None,
            "map75":            round(float(map75),    6) if map75    is not None else None,
            "map_small":        _extra["map_small"],
            "map_medium":       _extra["map_medium"],
            "map_large":        _extra["map_large"],
            "recall_max1":      _extra["recall_max1"],
            "recall_max10":     _extra["recall_max10"],
            "recall_max100":    _extra["recall_max100"],
            "recall_small":     _extra["recall_small"],
            "recall_medium":    _extra["recall_medium"],
            "recall_large":     _extra["recall_large"],
            "precision_legacy": round(float(mean_prec), 6) if mean_prec is not None else None,
            "avg_confidence":   round(float(avg_confidence), 6) if avg_confidence is not None else None,
            "total_predicted_cells": int(total_preds),
            "gt_cells":         total_gt,
        },
    }


def _compute_extra_coco_metrics(
    per_image_data: List[dict],
    n_classes: int,
    H: int,
    W: int,
    IOU_THRESHOLDS: np.ndarray,
) -> dict:
    """
    Compute COCO-style extended metrics not in the main eval loop:

      AR@maxDets=1/10/100   (average recall over IoU 0.50:0.95)
      mAP@area=small/medium/large  (AP averaged over IoU thresholds, per area bucket)
      AR@area=small/medium/large   (recall averaged over IoU thresholds, maxDets=100)

    Area bucket thresholds follow standard COCO pixel-area definitions:
      small  : pixel_area < 32² = 1 024
      medium : 32² ≤ pixel_area < 96² = 9 216
      large  : pixel_area ≥ 96² = 9 216

    Boxes are stored in normalised [0,1] coords, so
      pixel_area = norm_area × H × W.

    Returns -1.0 for any bucket that has no GT objects (mirrors Edge Impulse).
    """
    _MISSING = {
        "recall_max1":   -1.0,
        "recall_max10":  -1.0,
        "recall_max100": -1.0,
        "map_small":     -1.0,
        "map_medium":    -1.0,
        "map_large":     -1.0,
        "recall_small":  -1.0,
        "recall_medium": -1.0,
        "recall_large":  -1.0,
    }
    if not per_image_data:
        return _MISSING.copy()

    total_pixels = float(H * W)
    # Area thresholds in normalised coords (area = (x2-x1)*(y2-y1))
    area_ranges = {
        "small":  (0.0,                      1024.0 / total_pixels),
        "medium": (1024.0 / total_pixels,    9216.0 / total_pixels),
        "large":  (9216.0 / total_pixels,    2.0),   # 2.0 > 1.0 = entire frame
    }

    # Pre-check: does any GT exist in each area bucket?
    _gt_in_area: dict = {}
    for aname, (alo, ahi) in area_ranges.items():
        _gt_in_area[aname] = any(
            alo <= gta["area"] < ahi
            for img in per_image_data
            for gta in img["gt_areas"]
        )

    # ── AR @ maxDets ──────────────────────────────────────────────────────────
    def _ar_maxdets(max_det: int) -> float:
        recalls: List[float] = []
        for iou_thr in IOU_THRESHOLDS:
            for img in per_image_data:
                gbc = img["gt_by_class"]
                total_gt = sum(len(v) for v in gbc.values())
                if total_gt == 0:
                    continue
                dets = img["dets"][:max_det]
                gt_used: dict = {c: set() for c in gbc}
                matched = 0
                for det in dets:
                    c = det["class_idx"]
                    gts = gbc.get(c, [])
                    if not gts:
                        continue
                    pred_box = np.array([det["box"]], dtype=np.float32)
                    gt_arr   = np.array(gts, dtype=np.float32)
                    ious     = _box_iou_matrix(pred_box, gt_arr)[0]
                    best_gi  = int(np.argmax(ious))
                    if ious[best_gi] >= iou_thr and best_gi not in gt_used[c]:
                        gt_used[c].add(best_gi)
                        matched += 1
                recalls.append(matched / total_gt)
        return round(float(np.mean(recalls)), 6) if recalls else -1.0

    # ── mAP @ area ────────────────────────────────────────────────────────────
    def _map_area(area_name: str) -> float:
        if not _gt_in_area[area_name]:
            return -1.0
        alo, ahi = area_ranges[area_name]
        ap_per_iou: List[float] = []
        for iou_thr in IOU_THRESHOLDS:
            class_preds = [[] for _ in range(n_classes)]
            gt_count_area = np.zeros(n_classes, dtype=np.int32)
            for img in per_image_data:
                gbc = img["gt_by_class"]
                # Build area-filtered GT index per class for this image
                gt_in_area: dict = {}
                for c, gts in gbc.items():
                    for gi, g in enumerate(gts):
                        a = (g[2] - g[0]) * (g[3] - g[1])
                        if alo <= a < ahi:
                            gt_in_area.setdefault(c, []).append(gi)
                            gt_count_area[c] += 1
                gt_used: dict = {c: set() for c in gt_in_area}
                for det in img["dets"][:100]:
                    c = det["class_idx"]
                    if c >= n_classes:
                        continue
                    gts = gbc.get(c, [])
                    gt_area_c = gt_in_area.get(c, [])
                    matched = False
                    if gts and gt_area_c:
                        pred_box = np.array([det["box"]], dtype=np.float32)
                        gt_arr   = np.array(gts, dtype=np.float32)
                        ious     = _box_iou_matrix(pred_box, gt_arr)[0]
                        for gi in gt_area_c:
                            if ious[gi] >= iou_thr and gi not in gt_used.get(c, set()):
                                matched = True
                                gt_used.setdefault(c, set()).add(gi)
                                break
                    class_preds[c].append((det["score"], matched))
            aps: List[float] = []
            for c in range(n_classes):
                if gt_count_area[c] == 0:
                    continue
                preds_c = sorted(class_preds[c], key=lambda x: x[0], reverse=True)
                if preds_c:
                    sc = np.array([p[0] for p in preds_c])
                    mt = np.array([p[1] for p in preds_c])
                    aps.append(_compute_ap_at_iou(sc, mt, int(gt_count_area[c])))
                else:
                    aps.append(0.0)
            if aps:
                ap_per_iou.append(float(np.mean(aps)))
        return round(float(np.mean(ap_per_iou)), 6) if ap_per_iou else -1.0

    # ── AR @ area @ maxDets=100 ────────────────────────────────────────────────
    def _ar_area(area_name: str) -> float:
        if not _gt_in_area[area_name]:
            return -1.0
        alo, ahi = area_ranges[area_name]
        recalls: List[float] = []
        for iou_thr in IOU_THRESHOLDS:
            for img in per_image_data:
                gbc = img["gt_by_class"]
                gt_in_area: dict = {}
                total_gt_in_area = 0
                for c, gts in gbc.items():
                    for gi, g in enumerate(gts):
                        a = (g[2] - g[0]) * (g[3] - g[1])
                        if alo <= a < ahi:
                            gt_in_area.setdefault(c, []).append(gi)
                            total_gt_in_area += 1
                if total_gt_in_area == 0:
                    continue
                gt_used: dict = {c: set() for c in gt_in_area}
                matched = 0
                for det in img["dets"][:100]:
                    c = det["class_idx"]
                    gts = gbc.get(c, [])
                    gt_area_c = gt_in_area.get(c, [])
                    if not gts or not gt_area_c:
                        continue
                    pred_box = np.array([det["box"]], dtype=np.float32)
                    gt_arr   = np.array(gts, dtype=np.float32)
                    ious     = _box_iou_matrix(pred_box, gt_arr)[0]
                    for gi in gt_area_c:
                        if ious[gi] >= iou_thr and gi not in gt_used.get(c, set()):
                            gt_used.setdefault(c, set()).add(gi)
                            matched += 1
                            break
                recalls.append(matched / total_gt_in_area)
        return round(float(np.mean(recalls)), 6) if recalls else -1.0

    return {
        "recall_max1":   _ar_maxdets(1),
        "recall_max10":  _ar_maxdets(10),
        "recall_max100": _ar_maxdets(100),
        "map_small":     _map_area("small"),
        "map_medium":    _map_area("medium"),
        "map_large":     _map_area("large"),
        "recall_small":  _ar_area("small"),
        "recall_medium": _ar_area("medium"),
        "recall_large":  _ar_area("large"),
    }


def is_yolo_pro(architecture: str) -> bool:
    """True when the selected architecture is the YOLO-Pro detector."""
    return architecture.lower().strip() == YOLO_PRO_ARCHITECTURE


# ─────────────────────────────────────────────────────────────────────────────
# Dataset helpers
# ─────────────────────────────────────────────────────────────────────────────

def _snap32(v: int) -> int:
    """Snap v up to the nearest multiple of 32 (YOLO-Pro stride-32 constraint)."""
    return v if v % 32 == 0 else v + (32 - v % 32)


# Minimum input resolution enforced for YOLO-Pro training.
# With stride-32 at the P5 scale this gives a 3×3 detection grid — the bare
# minimum for any spatial detection.  Cached features produced below this size
# are treated as stale and must be recomputed from raw data.
YOLO_PRO_MIN_INPUT_SIZE: int = 96

# Minimum gradient steps for a meaningful run; expands epochs for tiny datasets.
MIN_TOTAL_STEPS: int = 1000

# Absolute minimum batch size the stabilizer may use.
_MIN_BATCH_SIZE: int = 1


def _compute_dataset_complexity_stats(
    boxes_train: List[Optional[List[dict]]],
    label_names: List[str],
    label_map: dict,
) -> dict:
    """
    Summarize stable dataset signals used by automatic training presets.

    Signals are intentionally coarse and deterministic so policy selection stays
    explainable across projects.
    """
    num_images = len(boxes_train)
    num_classes = len(label_names)
    class_counts = np.zeros(max(num_classes, 1), dtype=np.int32)
    total_boxes = 0
    labeled_images = 0

    for boxes in boxes_train:
        valid_here = 0
        for box in (boxes or []):
            raw_label = box.get("label_id")
            if raw_label is None:
                raw_label = box.get("label")
            cls_idx = label_map.get(raw_label)
            if cls_idx is None or cls_idx < 0 or cls_idx >= len(class_counts):
                continue
            class_counts[cls_idx] += 1
            total_boxes += 1
            valid_here += 1
        if valid_here > 0:
            labeled_images += 1

    avg_boxes_per_image = (float(total_boxes) / num_images) if num_images > 0 else 0.0
    active_class_count = int(np.count_nonzero(class_counts[:num_classes])) if num_classes > 0 else 0
    class_fraction_max = (
        float(class_counts[:num_classes].max()) / float(total_boxes)
        if total_boxes > 0 and num_classes > 0 else 0.0
    )
    labeled_image_fraction = (
        float(labeled_images) / float(num_images) if num_images > 0 else 0.0
    )

    return {
        "num_images": num_images,
        "num_classes": num_classes,
        "active_class_count": active_class_count,
        "total_boxes": total_boxes,
        "avg_boxes_per_image": avg_boxes_per_image,
        "class_fraction_max": class_fraction_max,
        "labeled_image_fraction": labeled_image_fraction,
    }


def _resolve_training_balance_policy(
    dataset_stats: dict,
    stabilization_policy: dict,
) -> dict:
    """
    Resolve bounded automatic presets for cls/object ranking behaviour.

    This is intentionally conservative: presets shift only a few knobs and stay
    within narrow bands so projects remain comparable and stable.
    """
    from app.ml.yolo_pro.losses import WEIGHT_BOX
    num_classes = int(dataset_stats.get("num_classes") or 0)
    active_class_count = int(dataset_stats.get("active_class_count") or 0)
    avg_boxes = float(dataset_stats.get("avg_boxes_per_image") or 0.0)
    class_fraction_max = float(dataset_stats.get("class_fraction_max") or 0.0)
    low_step_budget = bool(stabilization_policy.get("low_step_budget"))
    tiny_dataset_mode = bool(stabilization_policy.get("tiny_dataset_mode"))

    policy = {
        "w_cls": 1.0,
        "w_box": WEIGHT_BOX,
        "w_dfl": 1.5,
        "w_o2o": 0.25,
        "prior_prob": 0.01,
        "preset": "default",
        "reason": "baseline preset",
        "guarded_adapt_enabled": False,
        "guarded_adapt_cls_cap": 1.5,
        "guarded_adapt_o2o_cap": 0.35,
    }

    if avg_boxes >= 2.0:
        policy["prior_prob"] = 0.02
        policy["preset"] = "dense"
        policy["reason"] = "denser object frequency raises cls prior"

    if avg_boxes >= 4.0:
        policy["prior_prob"] = 0.03
        policy["preset"] = "very_dense"
        policy["reason"] = "very dense scenes benefit from a higher cls prior"

    ranking_risk = (
        num_classes >= 6
        or active_class_count >= 6
        or low_step_budget
        or tiny_dataset_mode
        or class_fraction_max >= 0.65
    )

    if ranking_risk:
        policy.update({
            "w_cls": 1.25,
            "w_o2o": 0.30,
            "guarded_adapt_enabled": True,
            "preset": "ranking_guard",
            "reason": (
                "multi-class/low-budget or imbalanced dataset; add mild cls and "
                "o2o support for score separation"
            ),
        })
        policy["prior_prob"] = max(policy["prior_prob"], 0.02 if avg_boxes >= 1.0 else 0.01)

    if num_classes >= 10 and low_step_budget:
        policy.update({
            "w_cls": 1.35,
            "w_o2o": 0.32,
            "guarded_adapt_enabled": True,
            "preset": "high_class_low_budget",
            "reason": "many classes with limited steps; strengthen cls ranking conservatively",
        })

    return policy


def _maybe_adapt_training_balance(
    balance_policy: dict,
    epoch_index: int,
    warmup_epochs: int,
    mean_val_box_dfl: "float | None",
    mean_val_cls: "float | None",
    best_val_box_dfl: float,
    best_val_cls: float,
    adapted_already: bool,
) -> tuple[dict, bool, Optional[str]]:
    """
    One-shot guarded adaptation when box learning advances but cls stalls.

    The adaptation is intentionally small and monotonic.
    """
    if adapted_already or not balance_policy.get("guarded_adapt_enabled"):
        return balance_policy, adapted_already, None
    if epoch_index < max(1, warmup_epochs):
        return balance_policy, adapted_already, None
    if mean_val_box_dfl is None or mean_val_cls is None:
        return balance_policy, adapted_already, None
    if not math.isfinite(mean_val_box_dfl) or not math.isfinite(mean_val_cls):
        return balance_policy, adapted_already, None
    if not math.isfinite(best_val_box_dfl) or not math.isfinite(best_val_cls):
        return balance_policy, adapted_already, None

    cls_stalled = mean_val_cls >= (best_val_cls - 1e-3)
    box_healthy = mean_val_box_dfl <= (best_val_box_dfl + 5e-3)

    if not (cls_stalled and box_healthy):
        return balance_policy, adapted_already, None

    new_policy = dict(balance_policy)
    old_cls = float(new_policy["w_cls"])
    old_o2o = float(new_policy["w_o2o"])
    new_policy["w_cls"] = min(float(new_policy["guarded_adapt_cls_cap"]), old_cls + 0.15)
    new_policy["w_o2o"] = min(float(new_policy["guarded_adapt_o2o_cap"]), old_o2o + 0.05)
    new_policy["preset"] = str(new_policy.get("preset", "default")) + "+guarded"
    reason = (
        f"guarded balance adaptation at epoch {epoch_index + 1}: "
        f"val_cls stalled ({mean_val_cls:.4f} vs best {best_val_cls:.4f}) "
        f"while val_box_dfl stayed healthy ({mean_val_box_dfl:.4f} vs best {best_val_box_dfl:.4f}); "
        f"w_cls {old_cls:.2f}→{new_policy['w_cls']:.2f}, "
        f"w_o2o {old_o2o:.2f}→{new_policy['w_o2o']:.2f}"
    )
    return new_policy, True, reason


def _maybe_adapt_cls_calibration(
    balance_policy: dict,
    safety_report: "dict | None",
    epoch: int,
    warmup_epochs: int,
    prior_prob: float,
    w_cls_cap: "float | None" = None,
    w_cls_step: "float | None" = None,
    val_cls_pos: "float | None" = None,
    val_cls_neg: "float | None" = None,
    best_val_cls_pos: "float | None" = None,
    best_val_cls_neg: "float | None" = None,
) -> Tuple[dict, bool, Optional[str]]:
    """
    Classifier-calibration adaptation driven by the val-loss VFL branch
    split — NOT the real-inference safety-eval report.

    Phase C of the Edge-Impulse-style migration: the per-epoch loop is
    becoming loss-only, so this adapter can no longer lean on the
    (expensive, probe-derived) ``safety_report`` fields tp_score_mean /
    fp_score_mean / precision / map50 / preds_per_image for its firing
    decision.  ``safety_report`` is still accepted as a parameter — the
    probe still runs this phase and callers still pass it — but it is no
    longer read here; it is retained only so the call site does not need
    restructuring before the probe itself is removed in the next phase.

    Distinct from ``_maybe_adapt_training_balance``:
      • Still reads the VFL positive/negative branch split rather than
        the combined ``mean_val_cls``, so it can tell *which* branch is
        failing (background not suppressed vs. foreground not learned)
        instead of just "cls is stalled".
      • Multi-shot — may re-fire across epochs until ``w_cls`` reaches
        the cap.  The legacy one-shot bump (+0.15 to a cap of 1.5) was
        too small for the failure mode where the cls head sits near the
        prior across the entire training horizon.
      • Step / cap are calibrated so the cls head can reach up to ~36 %
        of the box+dfl+cls weighted sum even after both adapters fire.

    Trigger — all three conditions required (mirrors the stalled/healthy
    pattern already used by ``_maybe_adapt_training_balance``, applied to
    the VFL branch split instead of the combined cls loss):
      • neg_stalled:     val_cls_neg  >= best_val_cls_neg - _CLS_CAL_NEG_STALL_EPS
            (the negative/background branch has not improved on its own
             best-seen value — background is not being suppressed).
      • pos_separating:  val_cls_pos  <= best_val_cls_pos + _CLS_CAL_POS_HEALTHY_EPS
            (the positive/foreground branch is still healthy — this is
             not just generic mid-training noise on both branches).
      • val_cls_neg > val_cls_pos
            (loss-side confirmation that the negative branch is the one
             that is actually worse, not merely "not yet improved").

    Every input silently no-ops when missing or non-finite so the adapter
    cannot fire on absence of signal — same contract as before.

    Trigger metrics are NOT used as checkpoint ranking metrics — only as
    an adaptation signal.  The configured monitor (val_box_dfl /
    val_loss) continues to drive checkpoint selection unchanged.

    Returns
    -------
    (new_policy, adapted, log_reason)
        new_policy : copy of ``balance_policy`` with possibly higher
                     ``w_cls``; same object reference returned on no-op
                     for callers that do identity-compare for diagnostics.
        adapted    : True only if w_cls strictly increased this call.
        log_reason : non-None message containing the exact phrase
                     ``"classifier calibration adaptation"`` and the full
                     metric snapshot when ``adapted`` is True; None
                     otherwise.
    """
    # Resolve module-level defaults at call time so the constants do not
    # have to be defined above this function's `def`.
    if w_cls_cap is None:
        w_cls_cap = _CLS_CAL_W_CLS_CAP
    if w_cls_step is None:
        w_cls_step = _CLS_CAL_W_CLS_STEP

    # Warmup guard — the val-loss split is noisy while TAL bootstraps.
    if epoch < max(0, warmup_epochs):
        return balance_policy, False, None

    current_w_cls = float(balance_policy.get("w_cls", 1.0))
    # Already at cap → adapter cannot meaningfully fire again.
    if current_w_cls >= float(w_cls_cap) - 1e-9:
        return balance_policy, False, None

    def _is_finite_number(v) -> bool:
        return isinstance(v, (int, float)) and math.isfinite(float(v))

    if not (
        _is_finite_number(val_cls_pos)
        and _is_finite_number(val_cls_neg)
        and _is_finite_number(best_val_cls_pos)
        and _is_finite_number(best_val_cls_neg)
    ):
        return balance_policy, False, None

    val_cls_pos_f = float(val_cls_pos)
    val_cls_neg_f = float(val_cls_neg)
    best_val_cls_pos_f = float(best_val_cls_pos)
    best_val_cls_neg_f = float(best_val_cls_neg)

    neg_stalled = val_cls_neg_f >= (best_val_cls_neg_f - _CLS_CAL_NEG_STALL_EPS)
    pos_separating = val_cls_pos_f <= (best_val_cls_pos_f + _CLS_CAL_POS_HEALTHY_EPS)
    neg_worse_than_pos = val_cls_neg_f > val_cls_pos_f

    if not (neg_stalled and pos_separating and neg_worse_than_pos):
        return balance_policy, False, None

    new_w_cls = min(float(w_cls_cap), current_w_cls + float(w_cls_step))
    if new_w_cls <= current_w_cls + 1e-9:
        # At cap already; the step does not move w_cls — treat as no-op.
        return balance_policy, False, None

    new_policy = dict(balance_policy)
    new_policy["w_cls"] = new_w_cls
    new_policy["preset"] = (
        str(new_policy.get("preset", "default")) + "+cls_cal"
    )

    log_reason = (
        f"classifier calibration adaptation at epoch {epoch + 1}: "
        f"w_cls {current_w_cls:.2f}→{new_w_cls:.2f} (cap={float(w_cls_cap):.2f}); "
        f"val_cls_pos={val_cls_pos_f:.4f} (best={best_val_cls_pos_f:.4f}), "
        f"val_cls_neg={val_cls_neg_f:.4f} (best={best_val_cls_neg_f:.4f}); "
        f"trigger_reasons=["
        f"neg_stalled: val_cls_neg={val_cls_neg_f:.4f} >= "
        f"best_val_cls_neg-{_CLS_CAL_NEG_STALL_EPS:g}="
        f"{best_val_cls_neg_f - _CLS_CAL_NEG_STALL_EPS:.4f}; "
        f"pos_separating: val_cls_pos={val_cls_pos_f:.4f} <= "
        f"best_val_cls_pos+{_CLS_CAL_POS_HEALTHY_EPS:g}="
        f"{best_val_cls_pos_f + _CLS_CAL_POS_HEALTHY_EPS:.4f}; "
        f"val_cls_neg={val_cls_neg_f:.4f} > val_cls_pos={val_cls_pos_f:.4f}]"
    )

    return new_policy, True, log_reason


# Early-stopping patience margin applied on top of the epoch-budget formulas
# below (+80%).  Premature stopping is the dominant accuracy regression on
# detector runs: val_loss plateaus several epochs before mAP does — especially
# across the close-mosaic transition, where the clean-image distribution shift
# flattens the monitor while localization is still tightening.  A patience
# sized purely off the monitor therefore exits before the refinement window
# pays off.  The cost is bounded — the effective epoch budget still caps the
# run — so the trade favours accuracy.  Raised from +50% to +80% (a 20% larger
# resolved patience) to give the final refinement window even more room.
_PATIENCE_MARGIN: float = 1.8


def _scaled_patience(base_patience: int) -> int:
    """
    Apply the ``_PATIENCE_MARGIN`` (+80%) safety margin to a formula-derived
    early-stopping patience, rounding up so small budgets gain at least one
    epoch (e.g. 20 → 36, 5 → 9).
    """
    base = int(base_patience)
    return max(base, int(math.ceil(base * _PATIENCE_MARGIN)))


def _compute_stabilization_policy(
    n_train: int,
    requested_epochs: int,
    user_batch: "int | None" = None,
) -> dict:
    """
    Resolve effective batch_size, epochs, and early-stopping patience from
    n_train, requested_epochs, and an optional explicit user_batch.

    Rules (deterministic, no fixed magic-number defaults):
      n_train == 0  → no-training-data sentinel
      n_train == 1  → batch_size = 1
      user_batch    → batch_size = min(user_batch, n_train); warn if spe < 4
      else          → largest batch giving steps_per_epoch >= min_target_spe

    Step budget:
      total_steps = spe * eff_epochs
      low_step_budget = total_steps < MIN_TOTAL_STEPS * 2

    Patience (based on eff_epochs, never so small it cancels the budget),
    then scaled by `_PATIENCE_MARGIN` (+80%) via `_scaled_patience`:
      spe <= 2 → max(5,  int(0.15 * eff_epochs))
      else     → max(8,  int(0.15 * eff_epochs))
      n <= 50  → max(8,  int(0.20 * eff_epochs))

    Returns keys:
      steps_per_epoch, total_steps, eff_batch_size, eff_epochs,
      requested_epochs, close_mosaic_n, warmup_epochs, static_tal_transition,
      tiny_dataset_mode, low_step_budget, batch_size_reduced, epochs_expanded, patience,
      no_training_data (bool), batch_size_warning (str | None)
    """
    import math as _m

    MIN_TARGET_SPE: int = 16

    req_ep = max(1, requested_epochs)

    # ── Edge case: no data ────────────────────────────────────────────────────
    if n_train <= 0:
        return {
            "no_training_data":     True,
            "steps_per_epoch":      0,
            "total_steps":          0,
            "eff_batch_size":       1,
            "epochs":               req_ep,   # user-requested, never mutated
            "effective_epochs":     req_ep,   # internal training budget
            "eff_epochs":           req_ep,   # legacy alias
            "requested_epochs":     req_ep,   # legacy alias
            "close_mosaic_n":       0,
            "warmup_epochs":        0,
            "static_tal_transition": 0,
            "tiny_dataset_mode":    True,
            "batch_size_reduced":   False,
            "epochs_expanded":      False,
            "patience":             _scaled_patience(max(5, int(0.15 * req_ep))),
            "batch_size_warning":   None,
        }

    n = n_train
    _warn: "str | None" = None

    # ── Edge case: single sample ──────────────────────────────────────────────
    if n == 1:
        eff_batch = 1
    elif n <= 50:
        # Tiny datasets still benefit from a small batch, but avoid batch=1
        # because its gradient noise is too high for stable detector training.
        eff_batch = min(4, n)
    elif user_batch is not None:
        # Honor explicit user choice when it fits inside the dataset.
        # When user_batch >= n, do NOT collapse to full-dataset batching —
        # use the same small-dataset floor as the auto path so spe stays >= 1.
        if user_batch < n:
            eff_batch = user_batch
        else:
            eff_batch = max(1, min(8, n))
        spe_check = max(1, _m.ceil(n / eff_batch))
        if n >= MIN_TARGET_SPE and spe_check < MIN_TARGET_SPE:
            _warn = (
                f"user_batch={user_batch} gives steps_per_epoch={spe_check} "
                f"(< min_target_spe={MIN_TARGET_SPE}); "
                f"consider a smaller batch to improve gradient coverage."
            )
    else:
        # Choose largest batch that still gives spe >= MIN_TARGET_SPE
        # batch = floor(n / MIN_TARGET_SPE), clipped to [1, n]
        eff_batch = max(1, min(n, n // MIN_TARGET_SPE))

    spe        = max(1, _m.ceil(n / eff_batch))
    batch_was  = eff_batch
    eff_epochs = req_ep
    total      = eff_epochs * spe
    low_step_budget = False

    # ── Tiny-dataset guardrail ────────────────────────────────────────────────
    # Reduce batch until spe >= MIN_TARGET_SPE or batch == 1.
    # Runs regardless of whether user_batch was supplied — the clamp above
    # already ensured user_batch >= n is handled, but n <= 8 edge cases still
    # need the loop to reach MIN_TARGET_SPE when possible.
    if n > 50 and n >= MIN_TARGET_SPE and spe < MIN_TARGET_SPE:
        while eff_batch > 1 and spe < MIN_TARGET_SPE:
            eff_batch -= 1
            spe = max(1, _m.ceil(n / eff_batch))
        total = eff_epochs * spe

    low_step_budget = (total < MIN_TOTAL_STEPS * 2)

    # ── Step-budget policy ───────────────────────────────────────────────────
    # User-requested epochs are authoritative; do not silently expand the
    # internal loop budget beyond the requested epoch count.

    # close-mosaic policy
    if n <= 50:
        close_mosaic_frac = 0.50  # tiny datasets: spend half the run on clean localization
    elif low_step_budget:
        # Borderline standard runs often achieve recall before they achieve
        # tight boxes. Give them the same longer clean-image refinement window
        # as tiny runs without forcing the whole training job onto tiny-mode.
        close_mosaic_frac = 0.50
    elif total >= 800:
        close_mosaic_frac = 0.38   # ← now disabled for final ~38%
    elif total >= 400:
        close_mosaic_frac = 0.38
    else:
        close_mosaic_frac = 0.35

    close_mosaic_n = (
        min(eff_epochs - 1, max(3, _m.ceil(close_mosaic_frac * eff_epochs)))
        if eff_epochs > 1 else 0
    )
    warmup_epochs         = max(1, min(3, eff_epochs))
    static_tal_transition = (
        min(3, max(1, close_mosaic_n // 3)) if close_mosaic_n > 0 else 0
    )

    # ── Patience ─────────────────────────────────────────────────────────────
    if n <= 50:
        # Tiny datasets: use same patience formula as non-tiny — premature stopping
        # (old min=5/max=3) fires before dynamic TAL IoU quality has time to rise.
        patience = max(8, int(0.20 * eff_epochs))
    elif spe <= 2:
        patience = max(5, int(0.15 * eff_epochs))
    else:
        patience = max(8, int(0.15 * eff_epochs))

    # +80% margin so early stopping cannot pre-empt the post-close-mosaic
    # refinement window (see _PATIENCE_MARGIN).
    patience = _scaled_patience(patience)

    # Keep the tiny safeguards for truly small datasets, and for low-budget
    # runs only while the dataset itself is still modest. This avoids pulling
    # medium/large datasets into the tiny path just because they land slightly
    # below the step-budget target after batch-size stabilization.
    tiny_dataset_mode = (n <= 50) or (low_step_budget and n <= 128)

    return {
        "no_training_data":     False,
        "steps_per_epoch":      spe,
        "total_steps":          total,
        "eff_batch_size":       eff_batch,
        "epochs":               req_ep,      # user-requested, unchanged
        "effective_epochs":     eff_epochs,  # internal training budget
        "eff_epochs":           eff_epochs,  # legacy alias
        "requested_epochs":     req_ep,      # legacy alias
        "close_mosaic_n":       close_mosaic_n,
        "warmup_epochs":        warmup_epochs,
        "static_tal_transition": static_tal_transition,
        "tiny_dataset_mode":    tiny_dataset_mode,
        "batch_size_reduced":   eff_batch < batch_was,
        "epochs_expanded":      eff_epochs > req_ep,
        "patience":             patience,
        "batch_size_warning":   _warn,
        "low_step_budget":      low_step_budget,
    }


def _build_yolo_dataset(
    X: np.ndarray,
    boxes_json: List[Optional[List[dict]]],
    label_map: dict,
    n_classes: int,
    input_shape: Tuple[int, int, int],
    reg_max: int = 16,
) -> Tuple[np.ndarray, List[List[dict]]]:
    """Validate array shapes and return (X, boxes_list)."""
    H, W, C = input_shape
    assert X.ndim == 4, f"Expected 4-D image array (N,H,W,C); got shape {X.shape}"
    assert X.shape[1] == H and X.shape[2] == W, (
        f"Image array spatial dims {X.shape[1:3]} don't match input_shape ({H},{W}). "
        "Ensure the DSP image block outputs the same resolution as the YOLO model input."
    )
    return X, boxes_json


def _compute_eval_conf_threshold(policy: dict) -> float:
    """
    Choose a post-training evaluation confidence threshold for mAP computation.

    mAP is defined as the area under the full precision-recall curve.  A high
    threshold (e.g. 0.25) silently truncates recall by discarding true positives
    below that score, collapsing AP to 0.0 on under-trained or small models
    whose cls scores cluster near the 0.01 prior-init baseline.

    0.005 is chosen because:
      • It admits predictions just above the background prior (~0.01 sigmoid)
        while class-agnostic NMS (iou_threshold=0.45, max_dets=100) removes
        duplicate cross-class boxes before matching.
      • This matches standard COCO-API evaluation practice and the OG spec.
      • Tiny-dataset and low-step-budget runs are the cases *most* likely to
        produce weak cls scores; using a strict threshold there is exactly
        backwards and was the primary cause of zero-metrics results.
    """
    return 0.005


def _compute_runtime_conf_threshold(policy: dict, size: str) -> float:
    """
    Choose a deployment/runtime confidence threshold for model metadata.

    This is distinct from the mAP eval threshold (0.005).  Runtime inference
    and firmware apply this value; it must suppress low-confidence FP floods
    without truncating legitimate detections on a well-trained model.

      • Standard runs:              0.25  (deployment default)
      • tiny_dataset_mode or
        low_step_budget runs:       0.35  (models are likely under-converged;
                                          a stricter gate reduces FP noise)
    """
    if policy.get("tiny_dataset_mode") or policy.get("low_step_budget"):
        return 0.35
    return 0.25


def _select_checkpoint_metric_name(
    policy: dict,
    mean_val_loss: "float | None",
    mean_val_box_dfl: "float | None",
) -> str:
    """
    Resolve the primary checkpoint / early-stop monitor for the current epoch.

    val_box_dfl correlates well with localization quality on healthy
    runs, but it is normalized over TAL positives — when the cls head
    partially collapses (e.g. post-mosaic on tiny / low-budget runs),
    fewer positives are assigned and the normalized box+DFL loss DROPS
    even though recall has collapsed.  Selecting on val_box_dfl in that
    regime silently exports a ranking-broken EMA snapshot
    (tp_score_mean < fp_score_mean → mAP ≈ 0).

    Policy rule:
      • tiny_dataset_mode OR low_step_budget → prefer val_loss.
        val_loss includes the un-normalized VFL classification term,
        which stays high when cls collapses and prevents the bad epoch
        from being saved as the best checkpoint.
      • Otherwise → prefer val_box_dfl, which on healthy runs gives a
        cleaner localization-quality signal.

    Fallback chain: chosen-primary → val_loss → val_box_dfl → train_loss.
    """
    prefer_val_loss = bool(
        policy.get("tiny_dataset_mode") or policy.get("low_step_budget")
    )
    if prefer_val_loss:
        if mean_val_loss is not None:
            return "val_loss"
        if mean_val_box_dfl is not None:
            return "val_box_dfl"
        return "train_loss"
    if mean_val_box_dfl is not None:
        return "val_box_dfl"
    if mean_val_loss is not None:
        return "val_loss"
    return "train_loss"


# Maximum allowed regression of mean_val_cls above its all-time best before the
# checkpoint guard rejects an epoch that improves val_box_dfl.  Calibrated so:
#   • Normal training oscillation (~5–15 % above best) passes.
#   • Genuine ranking collapse, where the classification head undertrains and
#     val_cls spikes 2×–10× above best while val_box_dfl keeps improving, is
#     rejected — preventing export of models with tp_score_mean < fp_score_mean.
# Independent of dataset, model size, or training horizon — it is a ratio over
# the model's own best classification loss, not a hardcoded threshold.
_CKPT_RANKING_GUARD_TOLERANCE: float = 0.5

# Abort training if this many non-finite (NaN/Inf) steps occur within one
# epoch.  Why: the per-step gate skips the optimizer + EMA update on a bad
# step, but BN moving stats may still be partially poisoned from the
# training=True forward pass that produced the NaN.  Bounding the count
# prevents the model degrading further while still tolerating rare
# transient instability.
_MAX_NAN_STEPS_PER_EPOCH: int = 3


# Subset size (number of val images) for the per-epoch calibration probe.
# This stays cheap enough to run every epoch.  Checkpoints that are about to
# be saved are re-checked on the full validation set before entering the EMA
# export buffer, so an ordered or lucky subset cannot green-light a snapshot
# that the final export eval will reject.
_CHECKPOINT_SAFETY_EVAL_SUBSET: int = 32


# ── Detector-quality safety thresholds ───────────────────────────────────────
# Hard FP-flood / detector-quality gates applied alongside the
# ``tp_score_mean > fp_score_mean`` ranking invariant.  These are *safety*
# gates — they do not participate in ranking among accepted candidates.
#
# Calibration anchor (production regression):
#   • tp_score_mean ≈ fp_score_mean (0.1058 vs 0.1056) passed the legacy
#     invariant by a +0.0003 gap;
#   • post-NMS preds_per_image ≈ 53.7  (real models typically ≤ 10–15);
#   • precision ≈ 0.0022   (real models clear 0.05+ within a handful of
#                            epochs once classification starts to separate);
#   • mAP50    ≈ 0.0011    (an unusable detector — random-baseline range).
#
# Bounds are intentionally lax: they only fire on flood / collapsed runs
# and must not block borderline-but-learning checkpoints.  Each gate
# silently no-ops when the corresponding metric is missing or non-finite.
_CKPT_SAFETY_MAX_PREDS_PER_IMAGE: float = 20.0
_CKPT_SAFETY_MIN_PRECISION:      float = 0.03
_CKPT_SAFETY_MIN_MAP50:          float = 0.01


# ── Safety-quality rescue checkpoint thresholds ──────────────────────────────
# Rescue path complements the primary monitor (val_box_dfl / val_loss): if a
# safety-passing epoch shows a meaningful improvement in real-inference quality
# but the primary monitor has stalled or regressed, save the EMA snapshot
# anyway so the better detector is not lost.  Real-world failure that motivated
# this: training stopped at EP86 and restored EP54 because EP54 had the best
# val_box_dfl, even though EP69-74 had safety mAP50 around 0.10-0.14
# (vs EP54 mAP50=0.0167) but never beat EP54's val_box_dfl.  These
# improvements were silently discarded.
#
# Rescue is gated on:
#   • composite safety gate passes (same as the monitor path);
#   • current safety mAP50 strictly improves over the best safety mAP50
#     of any previously SAVED checkpoint (monitor or rescue) by at least
#     ``_CKPT_RESCUE_MIN_MAP50_DELTA`` — not vs the immediately previous
#     epoch, so noisy local bumps cannot spam the buffer;
#   • tp_fp_gap, precision, runtime_preds_per_image all healthy.
#
# Rescue is a save-only mechanism — it never participates in ranking among
# already-accepted checkpoints and never replaces the primary monitor metric.
_CKPT_RESCUE_MIN_MAP50_DELTA:    float = 0.02
_CKPT_RESCUE_MIN_TP_FP_GAP:      float = 0.05
_CKPT_RESCUE_MIN_PRECISION:      float = _CKPT_SAFETY_MIN_PRECISION
_CKPT_RESCUE_MAX_RUNTIME_PREDS_PER_IMAGE: float = _CKPT_SAFETY_MAX_PREDS_PER_IMAGE


# ── Safety-stability rescue checkpoint thresholds ────────────────────────────
# Complements the strict safety-quality rescue path above.  The strict rescue
# only fires when current safety mAP50 IMPROVES over the best saved mAP50 by
# at least ``_CKPT_RESCUE_MIN_MAP50_DELTA`` (0.02).  A real-world failure was
# observed where the strict rule discarded clearly-healthy near-best epochs:
#
#   EP81 mAP50=0.1498 precision=0.7407 tp_fp_gap=0.2053 ppi=4.19   ← saved
#   EP83 mAP50=0.1453 precision=0.4766 tp_fp_gap=0.1742 ppi=4.09   ← dropped
#
# EP83 was only 0.0045 below EP81 yet would not save, so when EP81's full
# export validation failed the walkback had to fall back to a much older,
# weaker checkpoint instead of the near-best EP83.
#
# Stability rescue saves an EMA snapshot when ALL are true:
#   • composite safety gate passes;
#   • current safety mAP50 is within
#     ``_CKPT_STABILITY_NEAR_BEST_DELTA`` (0.01) of the best saved mAP50;
#   • tp_score_mean > fp_score_mean AND tp_fp_gap >= 0.10;
#   • precision >= 0.20;
#   • runtime_preds_per_image <= 20 (computed at runtime_conf_threshold).
#
# Independent of val_box_dfl improvement and never used as a ranking metric.
_CKPT_STABILITY_NEAR_BEST_DELTA: float = 0.01
_CKPT_STABILITY_MIN_TP_FP_GAP:   float = 0.10
_CKPT_STABILITY_MIN_PRECISION:   float = 0.20
_CKPT_STABILITY_MAX_RUNTIME_PREDS_PER_IMAGE: float = 20.0


# ── Classifier calibration adaptation (Layer 2) ──────────────────────────────
# Per-epoch adaptation that increases the classification loss weight
# (``w_cls``) when the val-loss VFL branch split shows the negative
# (background) branch stalling while the positive (foreground) branch
# stays healthy.  May re-fire across epochs until w_cls reaches the cap.
#
# Phase C of the Edge-Impulse-style migration: this used to read the
# real-inference safety-eval report (tp_score_mean / fp_score_mean /
# precision / map50 / preds_per_image — see git history for the retired
# multi-trigger design).  It now reads only signals already computed for
# free in the per-epoch val-loss pass, so the adapter no longer needs the
# expensive probe to decide whether to fire.  The probe itself is still
# run this phase (feeding the checkpoint safety gate and the calibration
# trace) but is no longer consulted here.
#
# Trigger — mirrors the stalled/healthy pattern already used by
# ``_maybe_adapt_training_balance`` (mean_val_cls vs. best_val_cls),
# applied to the VFL positive/negative branch split instead of the
# combined cls loss:
#   • neg_stalled:    val_cls_neg  has not improved on its own best-seen
#                     value (within ``_CLS_CAL_NEG_STALL_EPS``).
#   • pos_separating: val_cls_pos  is still at-or-near its own best-seen
#                     value (within ``_CLS_CAL_POS_HEALTHY_EPS``) — rules
#                     out generic mid-training noise on both branches.
#   • val_cls_neg > val_cls_pos — loss-side confirmation that the
#                     negative branch is the one that is actually worse.
#
# Epsilons reuse the values already established by
# ``_maybe_adapt_training_balance`` for the same VFL cls / box loss scale
# (1e-3 for the "stalled" comparison, 5e-3 for the "healthy" comparison).
#
# A bump step of 0.5 is intentionally larger than the legacy
# ``_maybe_adapt_training_balance`` step of 0.15: by the time this
# adapter's condition is met the legacy bump has either fired and failed
# or never fired at all, so the new mechanism needs to move meaningfully.
# Cap of 2.0 keeps box/dfl as the dominant gradient share (box=2.0 +
# dfl=1.5 = 3.5 vs cls=2.0 → cls ≤ 36 % of the weighted sum).
_CLS_CAL_W_CLS_STEP:    float = 0.5
_CLS_CAL_W_CLS_CAP:     float = 2.0
_CLS_CAL_NEG_STALL_EPS:   float = 1e-3
_CLS_CAL_POS_HEALTHY_EPS: float = 5e-3


def _compute_val_ranking_proxy_batch(
    cls_preds_per_scale: "list",
    fg_masks_per_scale: "list",
) -> tuple:
    """
    DEPRECATED as the checkpoint ranking gate — kept as a diagnostic only.

    The proxy is the mean max-class sigmoid score on TAL-positive anchor
    cells (tp_proxy) vs TAL-negative cells (fp_proxy), evaluated during
    the per-epoch val pass.  It is the training-time analogue of the
    final-eval tp_score_mean / fp_score_mean gap — same scoring rule
    (max-class sigmoid), partitioned by TAL assignment instead of by
    post-NMS GT matching.

    Why this is the right signal:
      • Score rule matches what the runtime/test path actually sees.
      • TAL positives are the same cells the head is trained to push to
        IoU-quality on (VFL target).  Their max-class sigmoid is the
        score that will become a TP at deployment.
      • TAL negatives are background — what the head must push toward
        the prior (~0.01).  Their max-class sigmoid is the score that
        will become an FP at deployment if it rises.
      • The gap therefore directly predicts the final eval
        tp_fp_score_gap, with no GT matching or NMS required.

    Args:
        cls_preds_per_scale : list of (B, N, n_classes) sigmoid scores
                              (one entry per FPN scale).
        fg_masks_per_scale  : list of (B, N) bool TAL fg masks,
                              same scale order.

    Returns:
        (tp_sum, tp_n, fp_sum, fp_n) — running sums for the batch.
    """
    tp_sum = 0.0
    tp_n = 0
    fp_sum = 0.0
    fp_n = 0
    for cls, fg in zip(cls_preds_per_scale, fg_masks_per_scale):
        max_cls = cls.max(axis=-1)
        if fg.any():
            tp_vals = max_cls[fg]
            tp_sum += float(tp_vals.sum())
            tp_n += int(tp_vals.size)
        neg = ~fg
        if neg.any():
            fp_vals = max_cls[neg]
            fp_sum += float(fp_vals.sum())
            fp_n += int(fp_vals.size)
    return tp_sum, tp_n, fp_sum, fp_n


def _ranking_proxy_passes(
    tp_proxy: "float | None",
    fp_proxy: "float | None",
    epoch: int,
    warmup_epochs: int,
) -> bool:
    """
    DEPRECATED as the checkpoint ranking gate — superseded by
    `_checkpoint_safety_gate_passes`, which evaluates EMA weights with the
    real inference pipeline (decode + class-agnostic NMS + GT IoU match).
    Retained as a no-cost diagnostic and to anchor the regression test
    that shows this TAL-anchor mean proxy can pass while post-NMS false
    positives still outrank true positives.

    This complements `_checkpoint_passes_ranking_guard` (which uses
    val_cls regression as a proxy and only fires under val_box_dfl).
    The ranking proxy here is monitor-agnostic and architecture-level:
    it requires `tp_proxy > fp_proxy` — the same invariant the final
    eval surfaces as `tp_fp_score_gap > 0`.

    Hardening rationale:
      • Standard runs checkpoint on val_box_dfl, whose normalization by
        TAL-positive count can DROP when cls partially collapses
        (fewer positives → smaller denominator).  val_cls regression
        alone is a delayed signal; the direct proxy catches the
        collapse on the epoch it appears.
      • Applied to val_loss runs too: val_loss can drift downward via
        VFL on negatives while the head still ranks background above
        foreground; the direct proxy catches that case as well.

    Behaviour:
      • epoch < warmup_epochs   → always allow.  The proxy is noisy
        during warmup; gating here would suppress all early saves.
        After warmup the gate applies even to the very first save —
        a collapsed-ranking snapshot must never be allowed to land
        as the initial best checkpoint (it can persist undisplaced
        when later val_box_dfl values never beat the early artifact).
      • Either proxy missing / non-finite → allow (no signal to act on).
      • Otherwise → require tp_proxy > fp_proxy.

    Returns True (allow save) or False (reject save).
    """
    if epoch < max(0, warmup_epochs):
        return True
    if tp_proxy is None or fp_proxy is None:
        return True
    if not (math.isfinite(tp_proxy) and math.isfinite(fp_proxy)):
        return True
    return tp_proxy > fp_proxy


def _checkpoint_quality_gate_failures(safety_report: dict) -> List[str]:
    """
    Detector-quality safety gates applied on top of the ranking invariant.

    The ranking invariant ``tp_score_mean > fp_score_mean`` confirms only
    that the score *direction* is correct.  It does not confirm that the
    detector is usable — a checkpoint can satisfy it by a +0.0003 margin
    while still producing dozens of false-positive boxes per image with
    precision ≈ 0 and mAP ≈ 0 (the production regression).  This helper
    returns the list of hard-safety failure reasons; an empty list means
    every applicable gate passes.

    Each individual gate silently no-ops when its metric is absent or
    non-finite so the gate cannot block on absence of signal.  This
    matches the no-op behaviour of the ranking invariant when tp/fp are
    missing.

    Enforced gates (all on already-computed fields of
    `evaluate_yolo_pro_detection`'s return value):
      1. Ranking invariant — ``tp_score_mean > fp_score_mean``.
      2. preds_per_image    ≤ _CKPT_SAFETY_MAX_PREDS_PER_IMAGE   (FP flood).
      3. precision          ≥ _CKPT_SAFETY_MIN_PRECISION         (FP flood).
      4. map50              ≥ _CKPT_SAFETY_MIN_MAP50             (unusable
                                                                  detector).

    Used by both the per-candidate checkpoint gate and the final-eval
    deployment classifier so checkpoint acceptance and `ranking_status`
    stay semantically consistent.
    """
    failures: List[str] = []

    # 1. Ranking invariant.
    tp = safety_report.get("tp_score_mean")
    fp = safety_report.get("fp_score_mean")
    if (
        tp is not None and fp is not None
        and math.isfinite(tp) and math.isfinite(fp)
        and not (tp > fp)
    ):
        failures.append(
            f"tp_score_mean={tp:.4f} <= fp_score_mean={fp:.4f}"
        )

    # 2. Post-NMS predictions-per-image ceiling — catches FP flood directly.
    # Prefer the runtime/deployment-threshold prediction count when the
    # report carries it: the AP eval threshold (~0.005) legitimately
    # admits many low-confidence detections on a healthy ranked model,
    # so a flood gate measured at AP confidence would reject good
    # checkpoints solely because the AP eval ran at conf=0.005.  The
    # runtime threshold matches what the deployed model actually applies,
    # so it is the correct flood-safety metric.
    #
    # Backward compatibility: when no runtime-threshold count is present
    # in the report, fall back to the legacy total_predicted_cells path
    # so older callers / tests retain the same gate semantics and failure
    # string ("preds_per_image=...").
    n_images       = safety_report.get("evaluated_images")
    runtime_total  = safety_report.get("runtime_total_predicted_cells")
    runtime_thr    = safety_report.get("runtime_conf_threshold")
    _runtime_used  = False
    if (
        isinstance(runtime_total, (int, float))
        and math.isfinite(float(runtime_total))
        and isinstance(n_images, (int, float))
        and n_images is not None
        and float(n_images) > 0
    ):
        _runtime_used = True
        runtime_preds_per_image = float(runtime_total) / float(n_images)
        if runtime_preds_per_image > _CKPT_SAFETY_MAX_PREDS_PER_IMAGE:
            _thr_str = (
                f" at runtime_conf_threshold={float(runtime_thr):.4f}"
                if isinstance(runtime_thr, (int, float))
                and math.isfinite(float(runtime_thr))
                else ""
            )
            failures.append(
                f"runtime_preds_per_image={runtime_preds_per_image:.2f} > "
                f"{_CKPT_SAFETY_MAX_PREDS_PER_IMAGE:.2f}{_thr_str}"
            )
    if not _runtime_used:
        # Legacy path: no runtime-threshold count supplied — fall back to
        # the AP-threshold count so the gate stays applicable on reports
        # produced before this field existed.
        total_preds = safety_report.get("total_predicted_cells")
        if (
            isinstance(total_preds, (int, float))
            and isinstance(n_images, (int, float))
            and math.isfinite(float(total_preds))
            and n_images is not None
            and float(n_images) > 0
        ):
            preds_per_image = float(total_preds) / float(n_images)
            if preds_per_image > _CKPT_SAFETY_MAX_PREDS_PER_IMAGE:
                failures.append(
                    f"preds_per_image={preds_per_image:.2f} > "
                    f"{_CKPT_SAFETY_MAX_PREDS_PER_IMAGE:.2f}"
                )

    # 3. Precision floor.
    prec = safety_report.get("precision")
    if prec is not None and math.isfinite(prec) and prec < _CKPT_SAFETY_MIN_PRECISION:
        failures.append(
            f"precision={prec:.4f} < {_CKPT_SAFETY_MIN_PRECISION:.4f}"
        )

    # 4. mAP50 floor (only if finite — skipped when eval reported None).
    map50 = safety_report.get("map50")
    if map50 is not None and math.isfinite(map50) and map50 < _CKPT_SAFETY_MIN_MAP50:
        failures.append(
            f"map50={map50:.4f} < {_CKPT_SAFETY_MIN_MAP50:.4f}"
        )

    return failures


def _classify_ranking_health(eval_report: dict) -> str:
    """
    Final-eval ranking-health classifier — turns the measured
    tp_score_mean / fp_score_mean separation **and the detector-quality
    safety gates** from `evaluate_yolo_pro_detection` into a single
    deployment-gating status that is stamped into every TrainedModel's
    metadata so model_testing and inference layers can refuse to deploy
    a broken model silently.

    Returns one of:
      • "ok"        — ``tp_score_mean > fp_score_mean`` AND every applicable
                      detector-quality safety gate passes (precision floor,
                      mAP50 floor, preds-per-image ceiling).
      • "collapsed" — ranking invariant violated, OR any detector-quality
                      safety gate failed (FP flood, precision < floor,
                      mAP50 < floor).  This is the deployment-blocking state.
      • "unknown"   — eval produced no TP or no FP scores (cannot compute
                      the ranking invariant; e.g. zero predictions
                      survived NMS, or every prediction matched a GT).

    Consistent with the per-candidate `_checkpoint_safety_gate_passes` —
    both delegate to `_checkpoint_quality_gate_failures` so an accepted
    checkpoint's ranking_status cannot disagree with the gate decision
    that accepted it.
    """
    tp = eval_report.get("tp_score_mean")
    fp = eval_report.get("fp_score_mean")
    if tp is None or fp is None:
        return "unknown"
    if not (math.isfinite(tp) and math.isfinite(fp)):
        return "unknown"
    failures = _checkpoint_quality_gate_failures(eval_report)
    return "ok" if not failures else "collapsed"


# Sentinel-classifier prefix shared between log lines and the structured
# RuntimeError raised when no safety-passing checkpoint can be exported.
# Tests grep for this exact phrase, and downstream tooling can use it to
# distinguish a deployment-blocking abort from a generic training crash.
_NO_SAFE_CKPT_ERROR_TAG = "no safety-passing checkpoint"


def _summarize_rejected_candidates(
    rejected: List[dict],
    best_safety_metrics: "dict | None" = None,
) -> dict:
    """
    Build a structured summary of every monitor-improving checkpoint
    candidate the safety gate rejected during training.

    Used to populate both the operator log and the structured RuntimeError
    when no safety-passing checkpoint exists at export time.  The summary
    intentionally focuses on *why* candidates were rejected — never on the
    monitor metric itself — so the safety gates stay observable as gates,
    not as a competing ranking signal.

    Returns
    -------
    dict with keys:
      candidates_evaluated     — int, number of rejection records.
      candidates_rejected      — int, identical to candidates_evaluated
                                 (every record represents a rejection).
      most_common_reasons      — list[ (reason_prefix, count) ] ordered
                                 by count desc; reasons are normalised to
                                 their gate name (e.g. "precision",
                                 "preds_per_image") so they aggregate even
                                 when numeric thresholds vary across runs.
      best_observed_safety     — dict of best observed precision / map50 /
                                 max tp-fp gap / min preds_per_image
                                 across all rejected candidates; values
                                 are None when no candidate reported them.
    """
    summary: dict = {
        "candidates_evaluated": len(rejected),
        "candidates_rejected":  len(rejected),
        "most_common_reasons":  [],
        "best_observed_safety": {
            "precision":      None,
            "map50":          None,
            "max_tp_fp_gap":  None,
            "min_preds_per_image":         None,
            # Runtime/deployment-threshold preds count — the metric the
            # flood gate actually consumes.  None when no rejection record
            # carried it (e.g. eval ran before runtime preds was wired).
            "min_runtime_preds_per_image": None,
        },
    }
    if not rejected:
        if best_safety_metrics:
            summary["best_observed_safety"] = dict(best_safety_metrics)
        return summary

    # Reason histogram.  Normalise to the gate prefix so "precision=0.001 <
    # 0.03" and "precision=0.0007 < 0.03" both aggregate as "precision".
    reason_counts: dict = {}
    for cand in rejected:
        for raw in cand.get("failed_gates") or []:
            if not isinstance(raw, str) or not raw:
                continue
            # Split on first ':' (cls_guard label) or first '=' / '<' / '>'
            prefix = raw
            for sep in (":", "="):
                if sep in prefix:
                    prefix = prefix.split(sep, 1)[0].strip()
                    break
            if not prefix:
                continue
            reason_counts[prefix] = reason_counts.get(prefix, 0) + 1
    summary["most_common_reasons"] = sorted(
        reason_counts.items(), key=lambda kv: (-kv[1], kv[0])
    )

    # Best observed safety metrics across all rejected candidates.
    _best_prec: "float | None" = None
    _best_map: "float | None" = None
    _max_gap: "float | None"  = None
    _min_ppi: "float | None"  = None
    _min_rt_ppi: "float | None" = None
    for cand in rejected:
        sm = cand.get("safety_metrics") or {}
        prec = sm.get("precision")
        map50 = sm.get("map50")
        tp = sm.get("tp_score_mean")
        fp = sm.get("fp_score_mean")
        ppi = sm.get("preds_per_image")
        rt_ppi = sm.get("runtime_preds_per_image")
        if isinstance(prec, (int, float)) and math.isfinite(prec):
            _best_prec = prec if _best_prec is None else max(_best_prec, prec)
        if isinstance(map50, (int, float)) and math.isfinite(map50):
            _best_map = map50 if _best_map is None else max(_best_map, map50)
        if (
            isinstance(tp, (int, float)) and isinstance(fp, (int, float))
            and math.isfinite(tp) and math.isfinite(fp)
        ):
            gap = float(tp) - float(fp)
            _max_gap = gap if _max_gap is None else max(_max_gap, gap)
        if isinstance(ppi, (int, float)) and math.isfinite(ppi):
            _min_ppi = ppi if _min_ppi is None else min(_min_ppi, ppi)
        if isinstance(rt_ppi, (int, float)) and math.isfinite(rt_ppi):
            _min_rt_ppi = (
                rt_ppi if _min_rt_ppi is None else min(_min_rt_ppi, rt_ppi)
            )
    summary["best_observed_safety"] = {
        "precision":                   _best_prec,
        "map50":                       _best_map,
        "max_tp_fp_gap":               _max_gap,
        "min_preds_per_image":         _min_ppi,
        "min_runtime_preds_per_image": _min_rt_ppi,
    }
    # Forward per-walk-back-candidate divergence records (admission-time
    # canonical safety metrics vs final eval metrics).  Only present
    # when the rejected entry came from walk-back — per-epoch monitor
    # rejections never carry these fields.
    _saved_vs_final: List[dict] = []
    for cand in rejected:
        if any(
            k in cand for k in (
                "saved_full_safety_map50",
                "saved_full_safety_precision",
                "final_eval_map50",
                "final_eval_precision",
            )
        ):
            _saved_vs_final.append({
                "epoch":                       cand.get("epoch"),
                "saved_full_safety_map50":     cand.get("saved_full_safety_map50"),
                "saved_full_safety_precision": cand.get("saved_full_safety_precision"),
                "final_eval_map50":            cand.get("final_eval_map50"),
                "final_eval_precision":        cand.get("final_eval_precision"),
                "eval_mode_match":             cand.get("eval_mode_match"),
            })
    if _saved_vs_final:
        summary["saved_vs_final"] = _saved_vs_final
    return summary


def _format_no_safe_ckpt_message(
    summary: dict,
    cause: str,
) -> str:
    """
    Format the structured RuntimeError / log message raised when no
    safety-passing checkpoint can be exported.

    ``cause`` is a short tag — typically ``"empty_buffer"`` (no monitor-
    improving candidate ever passed the composite safety gate) or
    ``"walkback_exhausted"`` (every buffered candidate's final-eval status
    came back collapsed).  The tag and ``_NO_SAFE_CKPT_ERROR_TAG`` are both
    embedded in the message so test grep stays stable.
    """
    reasons = summary.get("most_common_reasons") or []
    reason_str = (
        ", ".join(f"{name}×{count}" for name, count in reasons)
        if reasons else "no per-gate reasons recorded"
    )
    bs = summary.get("best_observed_safety") or {}
    parts = [
        f"YOLO-Pro export aborted: {_NO_SAFE_CKPT_ERROR_TAG} ({cause}).",
        f"candidates_evaluated={summary.get('candidates_evaluated', 0)}",
        f"candidates_rejected={summary.get('candidates_rejected', 0)}",
        f"most_common_reasons=[{reason_str}]",
        (
            "best_observed_safety="
            f"precision={bs.get('precision')} "
            f"map50={bs.get('map50')} "
            f"max_tp_fp_gap={bs.get('max_tp_fp_gap')} "
            f"min_preds_per_image={bs.get('min_preds_per_image')} "
            f"min_runtime_preds_per_image={bs.get('min_runtime_preds_per_image')}"
        ),
    ]
    # Saved-vs-final divergence — only surfaced on walk-back failures,
    # where each rejected candidate carries both its admission-time
    # canonical safety metrics and the matching final-eval metrics.
    # The presence of large deltas here tells the operator that the
    # checkpoint was admitted on a healthy full eval but a different
    # model state was scored at export, so the root cause is not a
    # genuinely-collapsed checkpoint.
    divergences = summary.get("saved_vs_final") or []
    if divergences:
        div_strs: List[str] = []
        for d in divergences:
            div_strs.append(
                f"EP{d.get('epoch')}: "
                f"saved_full_safety_map50={d.get('saved_full_safety_map50')} "
                f"final_eval_map50={d.get('final_eval_map50')} "
                f"saved_full_safety_precision={d.get('saved_full_safety_precision')} "
                f"final_eval_precision={d.get('final_eval_precision')} "
                f"eval_mode_match={d.get('eval_mode_match')}"
            )
        parts.append(f"saved_vs_final=[{' | '.join(div_strs)}]")
    return "  ".join(parts)


def _run_checkpoint_safety_eval(
    model: "tf.keras.Model",
    X_test: np.ndarray,
    boxes_test: List[Optional[List[dict]]],
    label_names: List[str],
    label_map: dict,
    anchor_grids: List[Tuple[np.ndarray, int]],
    H: int,
    W: int,
    reg_max: int,
    conf_threshold: float,
    subset_size: "int | None" = _CHECKPOINT_SAFETY_EVAL_SUBSET,
    job_id: str = "",
    runtime_conf_threshold: "float | None" = None,
) -> dict:
    """
    Lightweight real-inference checkpoint safety probe.

    Runs the same decode + class-agnostic NMS + GT IoU matching pipeline
    as the final post-training evaluation.  By default it uses a small,
    evenly-spaced validation subset for per-epoch calibration.  Pass
    ``subset_size=None`` to evaluate the full validation set; checkpoint
    candidates use that full pass before they are saved into the export
    buffer.

    Why a separate real-inference probe instead of the TAL-anchor mean
    proxy: the proxy aggregates max-class sigmoid over TAL-positive vs
    TAL-negative anchor cells, but the checkpoint we ship is judged on
    what survives decode + class-agnostic NMS + GT IoU matching.  Those
    survivors can include high-scoring cells that lie outside every GT
    (post-NMS false positives) even when the TAL-cell-level proxy looks
    healthy.  Running the same inference path used at deployment is the
    only signal that actually predicts the export's ranking quality.
    """
    if subset_size is None:
        n_subset = len(X_test)
    else:
        n_subset = min(int(subset_size), len(X_test))
    if n_subset <= 0:
        return {
            "tp_score_mean": None,
            "fp_score_mean": None,
            "tp_fp_score_gap": None,
            "evaluated_images": 0,
            "yolo_pro_eval_status": "no_data",
        }
    if n_subset >= len(X_test):
        X_sub = X_test
        boxes_sub = list(boxes_test)
    else:
        # Use coverage across the validation ordering instead of the first N
        # images.  Many datasets are class/file ordered, and a prefix-only
        # probe can look healthy while later classes or scenes are collapsing.
        subset_idx = np.linspace(0, len(X_test) - 1, n_subset, dtype=np.int64)
        X_sub = X_test[subset_idx]
        boxes_sub = [boxes_test[int(i)] for i in subset_idx]
    return evaluate_yolo_pro_detection(
        model=model,
        X_test=X_sub,
        boxes_test=boxes_sub,
        label_names=label_names,
        label_map=label_map,
        anchor_grids=anchor_grids,
        H=H, W=W, reg_max=reg_max,
        conf_threshold=conf_threshold,
        iou_threshold_match=0.50,
        job_id=f"{job_id}/safety" if job_id else "safety",
        runtime_conf_threshold=runtime_conf_threshold,
    )


# ── Canonical export-safety eval ─────────────────────────────────────────────
# The probe (subset_size=_CHECKPOINT_SAFETY_EVAL_SUBSET) is a cheap signal for
# calibration / candidate triage.  The CANONICAL export-safety eval below uses
# the same dataset (full X_test / boxes_test), same conf_threshold, same NMS
# parameters, same iou_threshold_match, and same runtime_conf_threshold as the
# final post-training detection eval.  It is the single source of truth for:
#   • checkpoint admission to the export buffer;
#   • rescue / stability rescue qualification;
#   • export-restore ranking;
#   • walk-back candidate ordering.
# Probe metrics never participate in those decisions.
def _run_export_safety_eval(
    model: "tf.keras.Model",
    X_test: np.ndarray,
    boxes_test: List[Optional[List[dict]]],
    label_names: List[str],
    label_map: dict,
    anchor_grids: List[Tuple[np.ndarray, int]],
    H: int,
    W: int,
    reg_max: int,
    conf_threshold: float,
    runtime_conf_threshold: "float | None",
    job_id: str = "",
) -> dict:
    """
    Canonical full-validation safety eval used for export admission.

    Thin wrapper around ``_run_checkpoint_safety_eval`` with
    ``subset_size=None`` pinned so the admission eval and the final
    post-training eval operate on identical inputs.  Centralising the
    call site prevents the probe / full / final paths from drifting
    apart on a future refactor.
    """
    return _run_checkpoint_safety_eval(
        model=model,
        X_test=X_test,
        boxes_test=boxes_test,
        label_names=label_names,
        label_map=label_map,
        anchor_grids=anchor_grids,
        H=H, W=W, reg_max=reg_max,
        conf_threshold=conf_threshold,
        subset_size=None,                          # full eval — never a subset
        job_id=f"{job_id}/export-safety" if job_id else "export-safety",
        runtime_conf_threshold=runtime_conf_threshold,
    )


def _snapshot_non_trainable_state(model: "tf.keras.Model") -> list:
    """
    Capture every non-trainable variable (BatchNorm moving_mean /
    moving_variance and the like) so a saved checkpoint can be restored
    bit-for-bit at final / walk-back time.

    Why this matters: ``ModelEMA`` shadows only trainable_weights.  When
    we admit a checkpoint at epoch N, the admission's full-validation
    eval runs against (EMA-shadowed trainable weights, epoch-N BN moving
    stats).  Training continues for many more epochs, updating BN moving
    stats in place.  If at final restore we only re-apply the EMA shadows,
    we evaluate (epoch-N trainable weights, epoch-LAST BN stats) — a
    different model.  Snapshotting non-trainable state here and restoring
    it at final / walk-back makes the admission's full eval the same
    model as the final eval.
    """
    return [tf.identity(v) for v in model.non_trainable_variables]


def _restore_non_trainable_state(model: "tf.keras.Model", snapshot: list) -> None:
    """
    Restore a snapshot produced by ``_snapshot_non_trainable_state``.
    No-op when the snapshot is None or shape-mismatched (defensive — a
    legacy buffer entry without a snapshot must still be restorable so
    older runs keep working).
    """
    if not snapshot:
        return
    live_vars = list(model.non_trainable_variables)
    if len(snapshot) != len(live_vars):
        return
    for live, saved in zip(live_vars, snapshot):
        try:
            live.assign(saved)
        except Exception:
            # A shape mismatch on a single var should not break the
            # whole restore — log via the caller's normal channels.
            continue


def _save_ema_checkpoint_keras(
    model: "tf.keras.Model",
    ema: "ModelEMA",
    path: str,
) -> None:
    """
    Write an EMA-applied ``.keras`` checkpoint to ``path``.

    Mirrors the final-export save (``backup_weights`` → ``apply`` →
    ``model.save`` → ``restore``, see the export block in
    ``run_yolo_pro_training``) so the file on disk is bit-for-bit the
    admission model: the EMA-shadowed trainable weights over the model's
    current non-trainable (BatchNorm moving) stats.

    The live training weights are ALWAYS restored in the ``finally`` block,
    so this is a pure side-observation of training state — it never
    perturbs the optimiser trajectory, the in-RAM buffer, or any metric.
    Restoring a checkpoint loaded from ``path`` is therefore equivalent to
    restoring that epoch's in-RAM ``ema_shadows`` + ``non_trainable_snapshot``
    (the Phase A fidelity invariant).
    """
    backup = ema.backup_weights(model)
    ema.apply(model)
    try:
        model.save(path)
    finally:
        ema.restore(model, backup)


# Phase B: the end-of-training finalist eval scores each candidate by
# LOADING its admission ``.keras`` from the disk manifest instead of
# restoring the in-RAM EMA shadows.  The two are bit-for-bit equivalent
# (Phase A fidelity invariant), so ranking / selection / export stay
# identical — this just moves the source of truth from RAM to disk.
#
# The flag keeps the legacy RAM-restore path reachable so the parity test
# can assert both paths select the same epoch on a fixed seed.  Disk is the
# production default.
_YOLO_PRO_FINALIST_EVAL_FROM_DISK: bool = True


def _resolve_finalist_eval_model(
    entry: dict,
    manifest_by_epoch: "dict",
    model: "tf.keras.Model",
    ema: "ModelEMA",
    *,
    use_disk: bool,
) -> "tuple":
    """
    Return ``(model_to_eval, source)`` for one retained finalist ``entry``.

    Phase E: the disk manifest is the SOLE source of truth — the in-RAM
    ``best_ema_buffer`` (with its ``ema_shadows`` / ``non_trainable_snapshot``)
    has been removed.  Load the entry's admission ``.keras`` with
    ``safe_mode=False`` (the model contains a Lambda layer) and return it; the
    loaded model is bit-for-bit the admission model (Phase A fidelity).

    The path is resolved from ``manifest_by_epoch`` (keyed by epoch) and, as a
    belt-and-suspenders fallback, the entry's own ``path``.  When no ``.keras``
    exists (a rare best-effort write failure at save time), return
    ``(None, "missing")`` so the caller records an ``inference_failed`` report
    for this finalist instead of crashing — a missing checkpoint simply cannot
    win selection.

    ``model``, ``ema`` and ``use_disk`` are retained in the signature for
    call-site / test compatibility but are no longer consulted: disk is always
    authoritative now.

    ``source`` is ``"disk"`` or ``"missing"`` (for logging and the parity test).
    """
    _m = manifest_by_epoch.get(entry.get("epoch"))
    _path = (_m.get("path") if _m else None) or entry.get("path")
    if _path and os.path.exists(_path):
        loaded = tf.keras.models.load_model(_path, safe_mode=False)
        return loaded, "disk"
    return None, "missing"


# Divergence thresholds — when the probe and full eval disagree by more
# than these deltas, log a "probe/full safety divergence" line so the
# operator can audit biased probe samples.  Pure diagnostic — never used
# to gate a save or change ranking.
_PROBE_FULL_DIVERGENCE_MAP50_DELTA:     float = 0.03
_PROBE_FULL_DIVERGENCE_PRECISION_DELTA: float = 0.10


def _probe_full_divergence(
    probe_report: "dict | None",
    full_report: "dict | None",
    map50_delta:     float = _PROBE_FULL_DIVERGENCE_MAP50_DELTA,
    precision_delta: float = _PROBE_FULL_DIVERGENCE_PRECISION_DELTA,
) -> "dict | None":
    """
    Compare probe vs canonical full-eval metrics and return a divergence
    record when the gap exceeds the spec thresholds.  Returns None when
    neither metric diverges, or when either side is missing the metric.

    Spec thresholds (helps detect biased probe samples):
      • probe_map50 - full_map50 > 0.03    → probe optimistic on mAP50
      • probe_precision - full_precision > 0.10 → probe optimistic on precision
    """
    if not probe_report or not full_report:
        return None

    def _f(report: dict, key: str) -> "float | None":
        v = report.get(key)
        if isinstance(v, (int, float)) and math.isfinite(float(v)):
            return float(v)
        return None

    pm, fm = _f(probe_report, "map50"), _f(full_report, "map50")
    pp, fp = _f(probe_report, "precision"), _f(full_report, "precision")

    map50_div = (
        pm is not None and fm is not None and (pm - fm) > float(map50_delta)
    )
    prec_div = (
        pp is not None and fp is not None and (pp - fp) > float(precision_delta)
    )
    if not (map50_div or prec_div):
        return None
    return {
        "probe_map50":      pm,
        "full_map50":       fm,
        "probe_precision":  pp,
        "full_precision":   fp,
        "map50_delta":      None if (pm is None or fm is None) else (pm - fm),
        "precision_delta":  None if (pp is None or fp is None) else (pp - fp),
        "map50_diverges":   map50_div,
        "precision_diverges": prec_div,
    }


def _checkpoint_safety_gate_passes(
    safety_report: "dict | None",
    epoch: int,
    warmup_epochs: int,
) -> bool:
    """
    Real-inference checkpoint safety gate.

    Replaces the older TAL-anchor mean ranking proxy as the checkpoint
    ranking gate.  The proxy could pass while post-NMS false positives
    still outranked true positives, because it scored TAL-assigned anchor
    cells rather than what actually survives decode + class-agnostic NMS +
    GT IoU matching.  This gate operates on the real-inference report and
    enforces the ranking invariant ``tp_score_mean > fp_score_mean``
    *and* a set of hard detector-quality safety gates (preds-per-image
    ceiling, precision floor, mAP50 floor — see
    `_checkpoint_quality_gate_failures`).

    Treated as a hard *safety* gate only — checkpoint selection among
    candidates that pass continues to be driven by the configured monitor
    metric (val_box_dfl / val_loss).  The quality gates do not rank
    candidates; they only reject ones with FP-flood / unusable-detector
    pathologies that the ranking invariant alone cannot detect.

    Behaviour:
      • epoch < warmup_epochs   → allow (no useful signal during warmup).
      • safety_report missing   → allow (nothing to gate on).
      • tp/fp missing or non-finite → allow (no signal to act on).
      • otherwise → require ranking invariant AND every applicable
        detector-quality safety gate to pass.
    """
    if epoch < max(0, warmup_epochs):
        return True
    if safety_report is None:
        return True
    tp = safety_report.get("tp_score_mean")
    fp = safety_report.get("fp_score_mean")
    if tp is None or fp is None:
        return True
    if not (math.isfinite(tp) and math.isfinite(fp)):
        return True
    return len(_checkpoint_quality_gate_failures(safety_report)) == 0


def _extract_runtime_preds_per_image(safety_report: "dict | None") -> "float | None":
    """
    Compute runtime_preds_per_image from a safety report's runtime-threshold
    prediction count.  Returns None when the runtime count or evaluated-image
    count is missing or non-finite — the rescue gate then treats the metric
    as absent and falls back to the composite safety gate's ceiling check.
    """
    if not safety_report:
        return None
    rt_total = safety_report.get("runtime_total_predicted_cells")
    n_imgs   = safety_report.get("evaluated_images")
    if (
        isinstance(rt_total, (int, float))
        and math.isfinite(float(rt_total))
        and isinstance(n_imgs, (int, float))
        and n_imgs is not None
        and float(n_imgs) > 0
    ):
        return float(rt_total) / float(n_imgs)
    return None


def _checkpoint_rescue_qualifies(
    safety_report: "dict | None",
    best_safety_map50: "float | None",
    epoch: int,
    warmup_epochs: int,
    min_map50_delta: float = _CKPT_RESCUE_MIN_MAP50_DELTA,
    min_tp_fp_gap:   float = _CKPT_RESCUE_MIN_TP_FP_GAP,
    min_precision:   float = _CKPT_RESCUE_MIN_PRECISION,
    max_runtime_preds_per_image: float = _CKPT_RESCUE_MAX_RUNTIME_PREDS_PER_IMAGE,
) -> bool:
    """
    Safety-quality rescue checkpoint gate.

    A separate save path that complements the primary monitor (val_box_dfl /
    val_loss).  Lets a safety-passing epoch save its EMA snapshot when real
    inference quality has meaningfully improved, even if the primary monitor
    has stalled or regressed.  Never replaces the primary monitor metric and
    never participates in ranking among already-accepted checkpoints.

    Required for rescue:
      • epoch >= warmup_epochs (the safety report is noisy during warmup);
      • composite safety gate passes
        (``_checkpoint_safety_gate_passes`` — ranking invariant plus the
        detector-quality safety gates: preds-per-image ceiling, precision
        floor, mAP50 floor);
      • safety map50 is finite;
      • safety map50 >= best_safety_map50 + min_map50_delta
        (compared against the best SAVED rescue/safety-quality map50, not
        the immediately previous epoch — noisy local bumps must not spam
        the buffer);
      • safety tp_fp_gap >= min_tp_fp_gap;
      • safety precision >= min_precision;
      • runtime_preds_per_image <= max_runtime_preds_per_image
        (when present; absent means the composite gate already enforced
        the equivalent ceiling, so the rescue gate does not double-fail
        on missing signal).

    Returns True iff every applicable condition is met; False otherwise.
    """
    if epoch < max(0, warmup_epochs):
        return False
    if safety_report is None:
        return False
    if not _checkpoint_safety_gate_passes(
        safety_report, epoch=epoch, warmup_epochs=warmup_epochs
    ):
        return False

    map50 = safety_report.get("map50")
    if not (isinstance(map50, (int, float)) and math.isfinite(float(map50))):
        return False
    baseline = (
        float(best_safety_map50)
        if best_safety_map50 is not None
        and isinstance(best_safety_map50, (int, float))
        and math.isfinite(float(best_safety_map50))
        else 0.0
    )
    if float(map50) < baseline + float(min_map50_delta):
        return False

    tp = safety_report.get("tp_score_mean")
    fp = safety_report.get("fp_score_mean")
    if not (
        isinstance(tp, (int, float)) and isinstance(fp, (int, float))
        and math.isfinite(float(tp)) and math.isfinite(float(fp))
    ):
        return False
    if (float(tp) - float(fp)) < float(min_tp_fp_gap):
        return False

    prec = safety_report.get("precision")
    if not (isinstance(prec, (int, float)) and math.isfinite(float(prec))):
        return False
    if float(prec) < float(min_precision):
        return False

    rt_ppi = _extract_runtime_preds_per_image(safety_report)
    if rt_ppi is not None and rt_ppi > float(max_runtime_preds_per_image):
        return False

    return True


def _checkpoint_stability_rescue_qualifies(
    safety_report: "dict | None",
    best_safety_map50: "float | None",
    epoch: int,
    warmup_epochs: int,
    near_best_delta:  float = _CKPT_STABILITY_NEAR_BEST_DELTA,
    min_tp_fp_gap:    float = _CKPT_STABILITY_MIN_TP_FP_GAP,
    min_precision:    float = _CKPT_STABILITY_MIN_PRECISION,
    max_runtime_preds_per_image: float = _CKPT_STABILITY_MAX_RUNTIME_PREDS_PER_IMAGE,
) -> "Tuple[bool, List[str]]":
    """
    Safety-stability rescue gate.

    Companion to ``_checkpoint_rescue_qualifies``: that strict path saves
    only on a meaningful mAP50 IMPROVEMENT (best + 0.02).  This path saves
    NEAR-BEST safety-quality checkpoints (best - 0.01) so the export /
    walk-back candidate pool does not fall back to much older / weaker
    snapshots when the single best safety checkpoint later fails full
    export validation.

    Returns (qualifies, skip_reasons).  ``skip_reasons`` lists every gate
    that blocked the save (empty when qualifies is True) so the per-epoch
    skip log can name exactly which rule rejected the save.  Independent
    of val_box_dfl improvement.

    Required (all must hold):
      • epoch >= warmup_epochs;
      • safety_report present;
      • composite safety gate passes
        (ranking invariant + preds-per-image ceiling + precision floor +
         mAP50 floor — same gate the monitor save path uses);
      • current safety mAP50 finite and
        >= (best_safety_map50 - near_best_delta) when best is known;
      • tp_score_mean > fp_score_mean AND
        (tp_score_mean - fp_score_mean) >= min_tp_fp_gap;
      • precision >= min_precision;
      • runtime_preds_per_image, when measurable, <= max_runtime_preds_per_image.
        The AP-threshold prediction count is intentionally NOT consulted —
        that metric admits low-confidence detections that the deployed
        model never sees.  Runtime is computed at runtime_conf_threshold.
    """
    skip_reasons: List[str] = []

    if epoch < max(0, warmup_epochs):
        skip_reasons.append("epoch_within_warmup")
        return False, skip_reasons
    if safety_report is None:
        skip_reasons.append("no_safety_report")
        return False, skip_reasons
    if not _checkpoint_safety_gate_passes(
        safety_report, epoch=epoch, warmup_epochs=warmup_epochs
    ):
        skip_reasons.append("composite_safety_gate_failed")
        return False, skip_reasons

    # mAP50 near-best delta.
    map50 = safety_report.get("map50")
    if not (isinstance(map50, (int, float)) and math.isfinite(float(map50))):
        skip_reasons.append("map50_not_finite")
        return False, skip_reasons
    if (
        best_safety_map50 is not None
        and isinstance(best_safety_map50, (int, float))
        and math.isfinite(float(best_safety_map50))
    ):
        floor = float(best_safety_map50) - float(near_best_delta)
        if float(map50) < floor:
            skip_reasons.append(
                f"map50 not within near-best delta: "
                f"current_safety_map50={float(map50):.4f} < "
                f"best_saved_safety_map50={float(best_safety_map50):.4f} - "
                f"{float(near_best_delta):.4f} = {floor:.4f}"
            )

    # Real-detector quality block.
    tp = safety_report.get("tp_score_mean")
    fp = safety_report.get("fp_score_mean")
    tp_fp_gap_val: "float | None" = None
    if not (
        isinstance(tp, (int, float)) and isinstance(fp, (int, float))
        and math.isfinite(float(tp)) and math.isfinite(float(fp))
    ):
        skip_reasons.append("tp_or_fp_score_mean_not_finite")
    else:
        tp_fp_gap_val = float(tp) - float(fp)
        if not (float(tp) > float(fp)):
            skip_reasons.append(
                f"tp_fp_gap below threshold: "
                f"tp_score_mean={float(tp):.4f} <= fp_score_mean={float(fp):.4f}"
            )
        elif tp_fp_gap_val < float(min_tp_fp_gap):
            skip_reasons.append(
                f"tp_fp_gap below threshold: "
                f"tp_fp_gap={tp_fp_gap_val:.4f} < {float(min_tp_fp_gap):.4f}"
            )

    prec = safety_report.get("precision")
    if not (isinstance(prec, (int, float)) and math.isfinite(float(prec))):
        skip_reasons.append("precision_not_finite")
    elif float(prec) < float(min_precision):
        skip_reasons.append(
            f"precision below threshold: "
            f"precision={float(prec):.4f} < {float(min_precision):.4f}"
        )

    rt_ppi = _extract_runtime_preds_per_image(safety_report)
    if rt_ppi is not None and rt_ppi > float(max_runtime_preds_per_image):
        skip_reasons.append(
            f"runtime_preds_per_image above threshold: "
            f"runtime_preds_per_image={rt_ppi:.2f} > "
            f"{float(max_runtime_preds_per_image):.2f}"
        )

    return (len(skip_reasons) == 0), skip_reasons


def _export_candidate_rank_key(entry: dict, index: int) -> "tuple":
    """
    Build the ranking key used for export restore and walk-back ordering.

    Higher tuple → preferred candidate.  Order, per spec:
      1. higher full-validation safety_map50
      2. higher precision
      3. higher tp_fp_gap
      4. lower runtime_preds_per_image
      5. newer epoch (buffer index breaks the last tie)

    Missing or non-finite values are mapped to the worst possible value
    for that field so they never displace a finite-value candidate:
      • map50 / precision / tp_fp_gap missing → -inf
      • runtime_preds_per_image missing       → +inf (negated below)

    The returned tuple ranks finalists for export restore
    (``_select_export_restore_index``).
    """
    def _f(v: object, missing: float) -> float:
        if isinstance(v, (int, float)) and math.isfinite(float(v)):
            return float(v)
        return missing

    # Prefer the explicit ``full_*`` fields written by the canonical
    # export-safety eval.  Fall back to the legacy ``safety_*`` names
    # for buffer entries persisted by older worker runs (or test
    # fixtures) so backward compatibility is preserved.
    def _pick(entry: dict, full_key: str, legacy_key: str, missing: float) -> float:
        v = entry.get(full_key)
        if isinstance(v, (int, float)) and math.isfinite(float(v)):
            return float(v)
        return _f(entry.get(legacy_key), missing)

    map50      = _pick(entry, "full_safety_map50",            "safety_map50",            float("-inf"))
    precision  = _pick(entry, "full_safety_precision",        "safety_precision",        float("-inf"))
    tp_fp_gap  = _pick(entry, "full_safety_tp_fp_gap",        "safety_tp_fp_gap",        float("-inf"))
    rt_ppi     = _pick(entry, "full_runtime_preds_per_image", "runtime_preds_per_image", float("inf"))
    # Negate runtime preds so larger tuple value = lower (better) preds.
    return (map50, precision, tp_fp_gap, -rt_ppi, index)


def _select_export_restore_index(buffer: List[dict]) -> Tuple[int, str]:
    """
    Pick the buffer entry to restore for export.

    Preference order (delegated to ``_export_candidate_rank_key``):
      1. highest ``safety_map50``;
      2. highest ``safety_precision``;
      3. highest ``safety_tp_fp_gap``;
      4. lowest ``runtime_preds_per_image``;
      5. newer epoch (higher index in the ring buffer).

    If no entry carries a finite ``safety_map50``, fall back to the
    newest entry — preserves legacy behavior on older runs whose safety
    probe did not record the field.

    Returns
    -------
    (index, preference_tag)
        index           : position in ``buffer`` of the chosen entry.
        preference_tag  : "highest_safety_map50" or
                          "newest_entry_no_safety_map50".

    Raises ValueError when ``buffer`` is empty so the caller cannot
    silently restore an undefined entry.
    """
    if not buffer:
        raise ValueError("_select_export_restore_index requires a non-empty buffer")

    def _safety_map50(entry: dict) -> float:
        # Prefer the canonical full export-safety value; fall back to the
        # legacy generic field for buffer entries from older runs.
        v = entry.get("full_safety_map50")
        if not (isinstance(v, (int, float)) and math.isfinite(float(v))):
            v = entry.get("safety_map50")
        if isinstance(v, (int, float)) and math.isfinite(float(v)):
            return float(v)
        return float("-inf")

    if all(_safety_map50(e) == float("-inf") for e in buffer):
        return len(buffer) - 1, "newest_entry_no_safety_map50"

    idx = max(
        range(len(buffer)),
        key=lambda i: _export_candidate_rank_key(buffer[i], i),
    )
    return idx, "highest_safety_map50"


def _select_checkpoint_eviction_index(manifest: List[dict]) -> int:
    """
    Choose which retained checkpoint to evict when the disk manifest is
    over the top-K budget (Phase E).

    Retention is **top-K by val_loss**: evict the entry with the WORST
    (highest) ``val_loss``.  When an entry's ``val_loss`` is None (the
    monitor was val_box_dfl / train_loss), fall back to its
    ``monitor_value`` — matching the filename-stub fallback used at save
    time.  Entries with neither finite value sort as worst so they are
    dropped first.  Ties break toward the OLDER epoch (lower epoch number)
    so, all else equal, the newer checkpoint is retained.

    Raises ValueError on an empty manifest so the caller cannot silently
    evict from nothing.
    """
    if not manifest:
        raise ValueError("_select_checkpoint_eviction_index requires a non-empty manifest")

    def _worst_key(i: int) -> tuple:
        entry = manifest[i]
        v = entry.get("val_loss")
        if not (isinstance(v, (int, float)) and math.isfinite(float(v))):
            v = entry.get("monitor_value")
        # Higher loss = worse; missing/non-finite → +inf (evict first).
        loss = float(v) if isinstance(v, (int, float)) and math.isfinite(float(v)) else float("inf")
        ep = entry.get("epoch")
        ep = int(ep) if isinstance(ep, (int, float)) else 0
        # max() picks the largest tuple: worst loss first, then older epoch
        # (negate epoch so the smaller/older epoch wins the tie).
        return (loss, -ep)

    return max(range(len(manifest)), key=_worst_key)


def _prune_checkpoints_to_topk(
    manifest: List[dict],
    k: int,
    *,
    job_id: str = "",
) -> List[dict]:
    """
    Prune ``manifest`` in place to the top-``k`` checkpoints by val_loss
    (Phase E retention), deleting each evicted entry's ``.keras`` from disk.

    Eviction target is chosen by ``_select_checkpoint_eviction_index`` (worst
    val_loss, tie → older epoch).  File removal is best-effort: a failure is
    logged at debug and swallowed so retention bookkeeping never raises.
    No-op when the manifest is already at or under ``k``.

    Returns the list of evicted entries (oldest-evicted first) so the caller
    can log them with per-epoch context.
    """
    evicted: List[dict] = []
    while len(manifest) > k:
        idx = _select_checkpoint_eviction_index(manifest)
        entry = manifest.pop(idx)
        evicted.append(entry)
        _path = entry.get("path")
        if _path and os.path.exists(_path):
            try:
                os.remove(_path)
            except Exception as _rm_exc:
                logger.debug(
                    f"[{job_id}] evicted checkpoint removal failed ({_path}): {_rm_exc}"
                )
    return evicted


def _final_eval_priority_key(entry: dict) -> tuple:
    """
    Pre-eval ranking key for one manifest entry: best (lowest) ``val_loss``
    first, falling back to ``monitor_value`` when the monitor was not val_loss
    (mirroring the filename-stub and eviction fallbacks).  Missing/non-finite
    scores sort last.  Ties break toward the LATER epoch — all else equal the
    more-refined weights are the better finalist.
    """
    v = entry.get("val_loss")
    if not (isinstance(v, (int, float)) and math.isfinite(float(v))):
        v = entry.get("monitor_value")
    loss = (
        float(v) if isinstance(v, (int, float)) and math.isfinite(float(v))
        else float("inf")
    )
    ep = entry.get("epoch")
    ep = int(ep) if isinstance(ep, (int, float)) else 0
    return (loss, -ep)


# Number of most-recent training epochs whose weights are ALWAYS scored by the
# post-training finalist eval, regardless of whether they improved the monitor
# or survived top-K retention.  The final clean-image epochs (post-close-mosaic,
# lowest LR) frequently deliver the best real detection accuracy even when
# val_loss ticks up, so they must get a fair shot at being the exported model
# rather than being filtered out by the monitor-improvement + top-K path.  They
# are evaluated ALONGSIDE the retained checkpoints and the best-by-mAP finalist
# is shipped — maximising final accuracy.
_ALWAYS_EVAL_FINAL_EPOCHS: int = 2


def _select_final_eval_order(manifest: List[dict]) -> Tuple[List[int], str]:
    """
    Order the retained finalists for the post-training full evaluation so the
    caller evaluates the fewest checkpoints needed to make a sound export
    decision.

    Always-evaluate final epochs: entries flagged ``always_eval`` (the final
    ``_ALWAYS_EVAL_FINAL_EPOCHS`` trained epochs) are placed FIRST and are
    always in scope.  The caller's early-stop-at-first-``ok`` shortcut is held
    until every always_eval entry has been scored, so the most-refined weights
    are compared head-to-head (by full-validation mAP) with the monitor
    checkpoints instead of being skipped.

    Scope rule for the remaining (monitor) checkpoints: only those saved after
    mosaic augmentation was disabled (``mosaic_active`` falsy — the close-mosaic
    phase) are primary eval candidates.  Mosaic-on checkpoints are trained on a
    distribution the deployed model never sees, so paying a full validation
    eval for each is wasted time.  Within the mosaic-off set, the
    best-by-``val_loss`` entry comes first (see ``_final_eval_priority_key``).

    The remaining entries stay in the returned order (always_eval, then
    mosaic-off, then mosaic-on) purely as an escalation fallback: mosaic-on
    entries are only reached if every better-ranked candidate collapses, which
    preserves the ``_NO_SAFE_CKPT_ERROR`` safety net.

    Returns ``(ordered_indices, scope)`` where ``scope`` describes the primary
    candidate set for logging: ``"mosaic_off"`` when a close-mosaic checkpoint
    (or always_eval final epoch) exists, ``"no_mosaic_off_fallback"`` otherwise.
    """
    always_eval = [
        i for i, e in enumerate(manifest) if bool(e.get("always_eval"))
    ]
    mosaic_off = [
        i for i, e in enumerate(manifest)
        if not bool(e.get("mosaic_active")) and not bool(e.get("always_eval"))
    ]
    mosaic_on = [
        i for i, e in enumerate(manifest)
        if bool(e.get("mosaic_active")) and not bool(e.get("always_eval"))
    ]
    scope = "mosaic_off" if (mosaic_off or always_eval) else "no_mosaic_off_fallback"
    always_eval.sort(key=lambda i: _final_eval_priority_key(manifest[i]))
    mosaic_off.sort(key=lambda i: _final_eval_priority_key(manifest[i]))
    mosaic_on.sort(key=lambda i: _final_eval_priority_key(manifest[i]))
    return always_eval + mosaic_off + mosaic_on, scope


def _checkpoint_passes_ranking_guard(
    ckpt_metric_name: str,
    mean_val_cls: "float | None",
    baseline_val_cls: "float | None",
    tolerance: float = _CKPT_RANKING_GUARD_TOLERANCE,
) -> bool:
    """
    Localization-only checkpoint guard against ranking collapse.

    When the primary checkpoint monitor is ``val_box_dfl`` (localization loss),
    val_loss / val_cls do not gate checkpoint selection at all.  A model whose
    classification head has regressed catastrophically (TP scores no longer
    separate from FP prior → tp_score_mean < fp_score_mean → collapsed mAP)
    can still improve val_box_dfl — and the saved EMA snapshot then exports a
    detector that fires copious low-confidence boxes indistinguishable from
    background.  This was observed on nano runs after the close-mosaic
    transition.

    The baseline must be the val_cls of the currently-saved checkpoint, NOT
    the global low-water-mark of val_cls.  Using the global low-water-mark
    over-rejects: an unsaved epoch (one that did not improve val_box_dfl) can
    ratchet the cls floor downward and starve later, genuinely-better
    checkpoints.  Comparing against the *saved* checkpoint instead ensures
    the guard only blocks epochs that are worse on classification than the
    model currently on disk.

    Args:
        ckpt_metric_name : which monitor is driving checkpoint selection.
        mean_val_cls     : this epoch's classification VFL loss (None if
                           val data was unavailable this epoch).
        baseline_val_cls : the val_cls recorded at the currently-saved
                           checkpoint epoch (None if no checkpoint saved
                           yet — guard is then no-op so the first save can
                           always land).
        tolerance        : fractional regression budget vs baseline.

    Returns:
        True  → epoch is allowed to advance the best checkpoint.
        False → improvement should be ignored (don't save EMA snapshot).
    """
    if ckpt_metric_name != "val_box_dfl":
        return True
    if mean_val_cls is None or baseline_val_cls is None:
        return True
    if not math.isfinite(mean_val_cls) or not math.isfinite(baseline_val_cls):
        return True
    # No prior saved checkpoint → no baseline to gate against; always allow.
    if baseline_val_cls <= 0.0:
        return True
    return mean_val_cls <= baseline_val_cls * (1.0 + tolerance)


def _early_stop_allowed(policy: dict, epoch_index: int, mosaic_epochs: int) -> bool:
    """
    Borderline standard runs should reach the close-mosaic phase before early
    stopping is allowed, otherwise we can exit before localization refinement
    has even started.
    """
    if policy.get("low_step_budget") and not policy.get("tiny_dataset_mode"):
        return epoch_index >= mosaic_epochs
    if policy.get("tiny_dataset_mode"):
        # tiny runs must not early-stop before clean-image refinement begins.
        return epoch_index >= mosaic_epochs
    return True


def _close_mosaic_grace_epochs(policy: dict) -> int:
    """
    Give borderline standard runs a short grace window after mosaic shuts off.
    The first clean-image epochs often spike while TAL and classification
    recalibrate to the new distribution, so early stopping should ignore that
    transient.
    """
    if policy.get("low_step_budget") and not policy.get("tiny_dataset_mode"):
        return 3
    if policy.get("tiny_dataset_mode"):
        # tiny runs need a longer grace window: the distribution shift at close-mosaic
        # is more disruptive on small datasets, and patience should reset fully here.
        return 5
    return 0


def _close_mosaic_min_epochs(effective_epochs: int, close_mosaic_epochs: int) -> int:
    """
    Minimum refinement window (in epochs) after mosaic turns OFF during which
    early stopping must be suppressed.  Patience accrued under mosaic is not
    representative of the clean-image distribution, and mAP frequently
    improves once mosaic is disabled — so we guarantee at least this many
    clean-image epochs before allowing an early stop.

    Scales with total epoch budget: roughly 10% of the effective training
    length, never less than 3, never more than 15, and never larger than the
    actual close-mosaic phase length.  Returns 0 when there is no close-mosaic
    phase (close_mosaic_epochs <= 0), which leaves stock early-stopping
    behavior intact for runs where mosaic was never enabled.

        close_mosaic_min_epochs = max(
            3,
            min(15, round(effective_epochs * 0.10), close_mosaic_epochs),
        )
    """
    n = int(close_mosaic_epochs) if close_mosaic_epochs is not None else 0
    if n <= 0:
        return 0
    eff = int(effective_epochs) if effective_epochs is not None else 0
    return max(3, min(15, round(eff * 0.10), n))


def _refinement_aug_probs(policy: dict, train_cfg) -> tuple[float, float]:
    """
    Return (hflip_prob, color_jitter_prob) for the close-mosaic refinement
    phase. Borderline standard runs benefit from a cleaner tail so localization
    can tighten on near-inference images instead of continuing to absorb strong
    appearance perturbations.
    """
    if policy.get("low_step_budget") and not policy.get("tiny_dataset_mode"):
        return 0.25, 0.0
    return train_cfg.hflip_prob, train_cfg.color_jitter_prob


# ─────────────────────────────────────────────────────────────────────────────
# Size-aware knob helpers  (nano-only tuning; larger sizes return current defaults)
# ─────────────────────────────────────────────────────────────────────────────

def _resolve_size_aware_learning_rate(size: str, job_lr: "float | None") -> float:
    """
    Return effective base_lr for the training schedule.

    nano                       → 7e-4  when no explicit LR is set.
                                 Reasoning: nano has channels as low as 16 at
                                 the stem — fewer parameters, tighter optima.
                                 70 % of the standard default reduces overshoot
                                 risk without meaningfully slowing convergence.
    tiny (and all other sizes) → 1e-3  (current default, unchanged).

    User-supplied job_lr always wins, regardless of size.
    """
    if job_lr is not None:
        return job_lr          # explicit user LR always respected
    return 7e-4 if size == "nano" else 1e-3


def _resolve_size_aware_patience(size: str, patience_base: int) -> int:
    """
    Return the early-stopping patience for this model size.

    nano → ceil(1.25 × base).  The shallower nano model can oscillate longer
            before settling; extra patience avoids stopping before a good
            local minimum is found.
    tiny (and larger sizes) → patience_base unchanged.
    """
    if size != "nano":
        return patience_base
    import math as _m
    return max(patience_base, int(_m.ceil(1.25 * patience_base)))


def _resolve_size_aware_close_mosaic_n(
    size: str, close_mosaic_n: int, eff_epochs: int
) -> int:
    """
    Return the number of close-mosaic (clean-image refinement) epochs.

    nano → add ceil(5 % × eff_epochs) more clean epochs (minimum 1 extra).
            nano has less capacity to generalise from 4-image composites;
            a longer unaugmented tail improves localization stability.
            Result is capped at eff_epochs - 1 to preserve at least one
            mosaic epoch.
    tiny (and larger sizes) → close_mosaic_n unchanged.
    """
    if size != "nano":
        return close_mosaic_n
    import math as _m
    extra = max(1, int(_m.ceil(0.05 * eff_epochs)))
    return min(eff_epochs - 1, close_mosaic_n + extra)


def _resolve_size_aware_refinement_aug(
    size: str, hflip: float, color_jitter: float
) -> "tuple[float, float]":
    """
    Return (hflip_prob, color_jitter_prob) for the close-mosaic refinement phase.

    nano → (hflip, color_jitter × 0.5).  Halving HSV jitter during the
            clean-image tail lets the nano model stabilise on unperturbed
            colour distributions before export; hflip is kept unchanged
            because mirroring does not shift the colour distribution.
    tiny (and larger sizes) → (hflip, color_jitter) unchanged.
    """
    if size == "nano":
        return hflip, color_jitter * 0.5
    return hflip, color_jitter


def _resolve_size_aware_eval_conf(size: str, conf_base: float) -> float:
    """
    Return the post-training evaluation confidence threshold.

    All size variants use conf_base unchanged.  The mAP metric requires the
    full precision-recall curve, so no size variant raises the threshold above
    conf_base — a raised threshold would truncate recall and collapse AP on
    under-converged models regardless of variant size.
    """
    return conf_base


def _log_size_benchmark_entry(
    job_id: str,
    size: str,
    eval_report: dict,
    ckpt_metric: str,
    actual_epochs: int,
    effective_epochs: int,
    base_lr: float,
) -> None:
    """
    Emit a structured benchmark log line for cross-size comparison.

    Tracks: mAP50, mAP75, precision, recall, post-NMS prediction count,
    checkpoint criterion, epochs completed, effective LR.

    No external framework required — grep 'BENCHMARK' across training logs
    from nano/tiny/small runs to collect a comparable table.
    """
    n_preds = int(eval_report.get("total_predicted_cells") or -1)
    logger.info(
        f"[{job_id}] BENCHMARK  size={size}  "
        f"mAP50={eval_report.get('map50') or -1.0:.4f}  "
        f"mAP75={eval_report.get('map75') or -1.0:.4f}  "
        f"precision={eval_report.get('precision') or -1.0:.4f}  "
        f"recall={eval_report.get('recall') or -1.0:.4f}  "
        f"post_nms_preds={n_preds}  "
        f"ckpt_criterion={ckpt_metric}  "
        f"epochs={actual_epochs}/{effective_epochs}  "
        f"base_lr={base_lr:.2e}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Anchor-point generation (fixed per scale, no learned anchors)
# ─────────────────────────────────────────────────────────────────────────────

def _make_anchor_points(
    H: int, W: int, strides: List[int]
) -> List[Tuple[np.ndarray, int]]:
    """
    Generate anchor-point grids for P3/P4/P5.

    Returns list of (anchor_points, stride) where anchor_points has shape
    (Hg*Wg, 2) with (cx, cy) in input-image pixel coordinates.
    """
    result = []
    for s in strides:
        Hg, Wg = H // s, W // s
        gy, gx = np.meshgrid(np.arange(Hg), np.arange(Wg), indexing="ij")
        cx = (gx.flatten() + 0.5) * s
        cy = (gy.flatten() + 0.5) * s
        anchor_pts = np.stack([cx, cy], axis=-1).astype(np.float32)
        result.append((anchor_pts, s))
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Task-Aligned Assigner (TAL) — replaces naive point-in-box
# ─────────────────────────────────────────────────────────────────────────────

def _tal_score_boxes(
    boxes: List[dict],
    label_map: dict,
    anchor_pts: np.ndarray,
    stride: int,
    H_img: int,
    W_img: int,
    n_classes: int,
    reg_max: int,
    alpha: float = 0.5,
    beta: float = 6.0,
    pred_cls_np: Optional[np.ndarray] = None,
    pred_boxes_np: Optional[np.ndarray] = None,
) -> Tuple[List[dict], int]:
    """
    Shared per-GT TAL scoring for one scale — the top-k-INDEPENDENT half of the
    assignment (Steps 1-2 of _assign_boxes_to_anchors: candidate ``inside_idx``,
    ``cls_score``, ``iou_vals``, ``tal_score``, ``vfl_quality``).  Performs NO
    top-k cut and NO target writes, so a single scoring pass can be resolved at
    any top-k (o2m=10 and o2o=1) without recomputing the expensive IoU work.

    Returns ``(scored, N)``:
        N      : number of anchors (``len(anchor_pts)``)
        scored : list of per-GT records in box iteration order.  GTs the original
                 code skipped via ``continue`` (unknown class, degenerate box)
                 are skipped here too, so they contribute to neither path.  Each
                 record holds:
                   cls_idx     : int
                   inside_idx  : (K,) int  candidate anchor indices
                   tal_score   : (K,) f32  TAL alignment score
                   vfl_quality : (K,) f32  varifocal quality target
                   x1,y1,x2,y2 : float     GT box in pixel coords
    """
    N = len(anchor_pts)
    scored: List[dict] = []
    if not boxes:
        return scored, N

    ax, ay = anchor_pts[:, 0], anchor_pts[:, 1]   # (N,) pixel coords

    for box in boxes:
        _lid = box.get("label_id")
        raw_label = _lid if _lid is not None else box.get("label")
        cls_idx = label_map.get(raw_label)
        if cls_idx is None:
            continue

        bx = float(box.get("x", 0)); by = float(box.get("y", 0))
        bw = float(box.get("w", 0)); bh = float(box.get("h", 0))
        x1 = bx * W_img;  y1 = by * H_img
        x2 = (bx + bw) * W_img;  y2 = (by + bh) * H_img

        if x2 <= x1 or y2 <= y1:
            continue

        # Step 1: candidate anchors — those whose centre lies inside the GT box.
        inside = (ax >= x1) & (ax <= x2) & (ay >= y1) & (ay <= y2)
        inside_idx = np.where(inside)[0]

        if len(inside_idx) == 0:
            # Fallback: assign the single closest anchor by centre distance.
            dist = np.hypot(ax - (x1 + x2) / 2, ay - (y1 + y2) / 2)
            inside_idx = np.array([np.argmin(dist)])

        # Step 2: TAL alignment score + VFL quality target (see the dynamic /
        # static rationale documented on _assign_boxes_to_anchors).
        if pred_cls_np is not None and pred_boxes_np is not None:
            # ── Dynamic TAL ──────────────────────────────────────────────────
            cls_score = np.clip(pred_cls_np[inside_idx, cls_idx], 1e-4, 1.0)   # (K,)
            pb   = pred_boxes_np[inside_idx]                                    # (K, 4) normalised
            gt_n = np.array([[x1 / W_img, y1 / H_img, x2 / W_img, y2 / H_img]])
            iou_vals = _box_iou_matrix(pb, gt_n)[:, 0]                         # (K,)
            iou_vals = np.clip(iou_vals, 0., 1.)
            tal_score   = np.maximum((cls_score ** alpha) * (iou_vals ** beta), 1e-8)
            vfl_quality = iou_vals
        else:
            # ── Static geometric fallback ─────────────────────────────────────
            hs = stride / 2.0
            ax_c = ax[inside_idx]; ay_c = ay[inside_idx]
            ix1 = np.maximum(ax_c - hs, x1); iy1 = np.maximum(ay_c - hs, y1)
            ix2 = np.minimum(ax_c + hs, x2); iy2 = np.minimum(ay_c + hs, y2)
            inter_area = np.maximum(ix2 - ix1, 0.) * np.maximum(iy2 - iy1, 0.)
            gt_area    = max((x2 - x1) * (y2 - y1), 1e-9)
            union_area = stride * stride + gt_area - inter_area + 1e-9
            iou_approx = np.clip(inter_area / union_area, 0., 1.)
            tal_score   = np.maximum(iou_approx ** beta, 1e-8)
            vfl_quality = iou_approx

        scored.append({
            "cls_idx":     cls_idx,
            "inside_idx":  inside_idx,
            "tal_score":   tal_score,
            "vfl_quality": vfl_quality,
            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
        })

    return scored, N


def _resolve_targets(
    scored: List[dict],
    N: int,
    anchor_pts: np.ndarray,
    stride: int,
    H_img: int,
    W_img: int,
    n_classes: int,
    reg_max: int,
    topk: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Top-k cut + conflict resolution + target writes — the top-k-DEPENDENT half of
    the assignment (Steps 3-4 of _assign_boxes_to_anchors).  Consumes the output
    of :func:`_tal_score_boxes` and writes the final target arrays.

    Allocates its OWN ``best_score`` and target buffers and treats ``scored`` as
    read-only (the top-k slice creates fresh arrays via fancy indexing, never
    mutating the records).  Therefore calling it twice on the SAME ``scored`` —
    ``topk=10`` for o2m and ``topk=1`` for o2o — produces two fully independent
    assignments, byte-identical to two separate _assign_boxes_to_anchors calls.
    """
    cls_targets  = np.zeros((N, n_classes), dtype=np.float32)
    box_targets  = np.zeros((N, 4),         dtype=np.float32)
    ltrb_targets = np.zeros((N, 4),         dtype=np.float32)
    fg_mask      = np.zeros((N,),           dtype=bool)

    if not scored:
        return cls_targets, box_targets, ltrb_targets, fg_mask

    ax, ay = anchor_pts[:, 0], anchor_pts[:, 1]
    # Per-anchor best assignment score (for conflict resolution) — private to
    # THIS resolve pass, so o2m and o2o never contend for the same anchor.
    best_score = np.zeros(N, dtype=np.float32)

    for rec in scored:
        cls_idx     = rec["cls_idx"]
        inside_idx  = rec["inside_idx"]
        tal_score   = rec["tal_score"]
        vfl_quality = rec["vfl_quality"]
        x1 = rec["x1"]; y1 = rec["y1"]; x2 = rec["x2"]; y2 = rec["y2"]

        # Step 3: keep only top-k by TAL score.  Fancy indexing yields new arrays
        # — the shared ``rec`` is never mutated, so the other path sees the full
        # candidate set.
        if len(inside_idx) > topk:
            topk_local  = np.argsort(tal_score)[-topk:]
            inside_idx  = inside_idx[topk_local]
            vfl_quality = vfl_quality[topk_local]
            tal_score   = tal_score[topk_local]

        # Step 4: conflict resolution — anchor already claimed by higher GT?
        for local_j, anchor_j in enumerate(inside_idx):
            score_j = tal_score[local_j]
            if score_j <= best_score[anchor_j]:
                continue   # this GT loses the contest for this anchor

            best_score[anchor_j] = score_j
            iou_q = float(vfl_quality[local_j])

            cls_targets[anchor_j, :]       = 0.0           # clear previous
            cls_targets[anchor_j, cls_idx] = iou_q         # IoU-weighted VFL label

            box_norm = np.array(
                [x1 / W_img, y1 / H_img, x2 / W_img, y2 / H_img],
                dtype=np.float32,
            )
            box_targets[anchor_j] = box_norm

            ltrb = np.array([
                (ax[anchor_j] - x1) / stride,
                (ay[anchor_j] - y1) / stride,
                (x2 - ax[anchor_j]) / stride,
                (y2 - ay[anchor_j]) / stride,
            ], dtype=np.float32)
            ltrb_targets[anchor_j] = np.clip(ltrb, 0.0, reg_max - 1e-3)
            fg_mask[anchor_j]      = True

    return cls_targets, box_targets, ltrb_targets, fg_mask


def _assign_boxes_to_anchors(
    boxes: List[dict],
    label_map: dict,
    anchor_pts: np.ndarray,
    stride: int,
    H_img: int,
    W_img: int,
    n_classes: int,
    reg_max: int,
    topk: int = 10,
    alpha: float = 0.5,
    beta: float = 6.0,
    pred_cls_np: Optional[np.ndarray] = None,
    pred_boxes_np: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Task-Aligned Assignment for one scale.

    When pred_cls_np and pred_boxes_np are supplied (dynamic TAL):
      - TAL score = cls_pred[gt_class]^alpha * iou(pred_box, gt_box)^beta
      - VFL quality target = iou(pred_box, gt_box)   ← real quality signal
      This creates a curriculum: poor early predictions → low but non-zero
      targets; improving predictions → rising targets → stronger gradient.

    When they are None (static / geometric fallback):
      - TAL score = cell_iou^beta  (anchor cell vs GT box overlap)
      - VFL quality target = cell_iou  ← geometric approximation

    Args:
        boxes         : list of {'label':..., 'x','y','w','h'} normalised [0,1]
        anchor_pts    : (N, 2)  anchor centres (cx, cy) in pixel coords
        stride        : scale stride (8, 16, or 32)
        H_img, W_img  : input image dimensions in pixels
        n_classes     : number of object classes
        reg_max       : DFL distribution bins
        topk          : max anchors per GT box (default 10; was 13).
                        Fewer anchors removes marginal positives at the GT
                        boundary, giving the regression head cleaner targets.
        alpha         : cls-score exponent in TAL metric (default 0.5)
        beta          : IoU exponent in TAL metric (default 6.0; was 4.0).
                        Higher beta makes assignment strongly prefer
                        well-aligned anchors — ratio IoU=0.7 vs IoU=0.3
                        is ~1000:1 at beta=6 vs ~40:1 at beta=4 — which
                        directly improves box tightness and mAP75.
        pred_cls_np   : (N, n_classes)  float32  live sigmoid cls scores, or None
        pred_boxes_np : (N, 4)          float32  live decoded pred boxes (normalised), or None

    Returns:
        cls_targets  : (N, n_classes)  float32  IoU-weighted varifocal labels
        box_targets  : (N, 4)          float32  GT boxes (x1y1x2y2) normalised
        ltrb_targets : (N, 4)          float32  ltrb distances in grid units
        fg_mask      : (N,)            bool

    Implemented as ``_tal_score_boxes`` (shared per-GT scoring — Steps 1-2)
    followed by ``_resolve_targets`` (top-k + conflict resolution — Steps 3-4).
    This is a pure refactor: for a single ``topk`` the two-stage path is
    byte-identical to the previous inline loop.  The split exists so o2m and o2o
    can share ONE scoring pass (see ``_assign_boxes_o2m_o2o``) instead of
    recomputing the IoU-heavy scoring twice per sample per scale.
    """
    scored, N = _tal_score_boxes(
        boxes, label_map, anchor_pts, stride, H_img, W_img, n_classes, reg_max,
        alpha=alpha, beta=beta, pred_cls_np=pred_cls_np, pred_boxes_np=pred_boxes_np,
    )
    return _resolve_targets(
        scored, N, anchor_pts, stride, H_img, W_img, n_classes, reg_max, topk,
    )


# ─────────────────────────────────────────────────────────────────────────────
# One-to-one (o2o) assignment — thin wrapper around the TAL assigner
# ─────────────────────────────────────────────────────────────────────────────

def _assign_boxes_to_anchors_o2o(
    boxes: List[dict],
    label_map: dict,
    anchor_pts: np.ndarray,
    stride: int,
    H_img: int,
    W_img: int,
    n_classes: int,
    reg_max: int,
    alpha: float = 0.5,
    beta: float = 6.0,
    pred_cls_np: Optional[np.ndarray] = None,
    pred_boxes_np: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    One-to-one (o2o) assignment for one scale.

    Identical to _assign_boxes_to_anchors but with topk=1, so at most ONE
    anchor is selected per GT box.  Conflict resolution ensures each anchor is
    claimed by at most one GT (highest TAL score wins), exactly mirroring the
    OG Edge Impulse o2o path visible in training logs as o2o_box_ciou /
    o2o_box_dfl / o2o_class.

    The same head outputs (cls_pred, reg_pred) are used for both o2m and o2o
    loss calls; only the assignment targets differ.  This avoids any model
    architecture change — the dual path is purely a training-time signal.

    Returns:
        Same 4-tuple as _assign_boxes_to_anchors:
        cls_targets, box_targets, ltrb_targets, fg_mask
    """
    return _assign_boxes_to_anchors(
        boxes=boxes,
        label_map=label_map,
        anchor_pts=anchor_pts,
        stride=stride,
        H_img=H_img,
        W_img=W_img,
        n_classes=n_classes,
        reg_max=reg_max,
        topk=1,
        alpha=alpha,
        beta=beta,
        pred_cls_np=pred_cls_np,
        pred_boxes_np=pred_boxes_np,
    )


def _assign_boxes_o2m_o2o(
    boxes: List[dict],
    label_map: dict,
    anchor_pts: np.ndarray,
    stride: int,
    H_img: int,
    W_img: int,
    n_classes: int,
    reg_max: int,
    topk_o2m: int = 10,
    alpha: float = 0.5,
    beta: float = 6.0,
    pred_cls_np: Optional[np.ndarray] = None,
    pred_boxes_np: Optional[np.ndarray] = None,
) -> Tuple[
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
]:
    """
    Score once, resolve twice: return ``(o2m_targets, o2o_targets)`` for one
    (sample, scale).

    The IoU-heavy per-GT TAL scoring (``inside_idx``, predicted-box IoU,
    ``tal_score``, ``vfl_quality``) is computed a SINGLE time via
    :func:`_tal_score_boxes` and shared between the two paths; only the cheap
    top-k cut + conflict resolution runs per path.  Output is byte-identical to
    calling ``_assign_boxes_to_anchors(topk=10)`` and
    ``_assign_boxes_to_anchors_o2o`` (topk=1) separately: the two
    :func:`_resolve_targets` passes are fully independent (each allocates its own
    ``best_score``) and o2o stays top-k=1 of the SAME metric — the passes are
    never merged.
    """
    scored, N = _tal_score_boxes(
        boxes, label_map, anchor_pts, stride, H_img, W_img, n_classes, reg_max,
        alpha=alpha, beta=beta, pred_cls_np=pred_cls_np, pred_boxes_np=pred_boxes_np,
    )
    o2m = _resolve_targets(
        scored, N, anchor_pts, stride, H_img, W_img, n_classes, reg_max, topk_o2m,
    )
    o2o = _resolve_targets(
        scored, N, anchor_pts, stride, H_img, W_img, n_classes, reg_max, 1,
    )
    return o2m, o2o


# ─────────────────────────────────────────────────────────────────────────────
# Decode reg predictions for CIoU in the training step
# ─────────────────────────────────────────────────────────────────────────────

def _decode_reg_scale(
    reg_pred:   tf.Tensor,
    anchor_pts: tf.Tensor,
    stride:     float,
    H_img:      int,
    W_img:      int,
    reg_max:    int,
) -> tf.Tensor:
    """
    Decode DFL logits → (x1,y1,x2,y2) normalised boxes for one scale.

    Args:
        reg_pred   : (B, N, 4*reg_max) raw DFL logits
        anchor_pts : (N, 2) anchor centres in pixel coords (as tf.Tensor)
        stride     : stride for this scale (scalar float, NOT derived from anchors)
        H_img, W_img : image dimensions
        reg_max    : DFL bins

    Returns:
        pred_boxes : (B, N, 4) float32 normalised [0,1]
    """
    B = tf.shape(reg_pred)[0]
    N = tf.shape(reg_pred)[1]

    # Sanitize raw DFL logits: a single non-finite value would propagate
    # through softmax → NaN ltrb → NaN pred_box → NaN CIoU → NaN gradient
    # for every cell on this scale.  Replace with 0 (uniform softmax for
    # that cell); downstream masking handles the actual loss contribution.
    reg_pred = tf.where(
        tf.math.is_finite(reg_pred), reg_pred, tf.zeros_like(reg_pred)
    )

    logits = tf.reshape(reg_pred, (B, N, 4, reg_max))
    probs  = tf.nn.softmax(logits, axis=-1)           # (B, N, 4, reg_max)
    bins   = tf.cast(tf.range(reg_max), tf.float32)   # (reg_max,)
    ltrb_grid = tf.reduce_sum(probs * bins, axis=-1)  # (B, N, 4) in grid units
    ltrb_px   = ltrb_grid * tf.cast(stride, tf.float32)  # → pixels

    ax = anchor_pts[:, 0]   # (N,)
    ay = anchor_pts[:, 1]

    x1 = tf.clip_by_value((ax - ltrb_px[:, :, 0]) / float(W_img), 0., 1.)
    y1 = tf.clip_by_value((ay - ltrb_px[:, :, 1]) / float(H_img), 0., 1.)
    x2 = tf.clip_by_value((ax + ltrb_px[:, :, 2]) / float(W_img), 0., 1.)
    y2 = tf.clip_by_value((ay + ltrb_px[:, :, 3]) / float(H_img), 0., 1.)
    pred_boxes = tf.stack([x1, y1, x2, y2], axis=-1)   # (B, N, 4)
    # Final sanitization safety net.
    pred_boxes = tf.where(
        tf.math.is_finite(pred_boxes), pred_boxes, tf.zeros_like(pred_boxes)
    )
    return pred_boxes


# ─────────────────────────────────────────────────────────────────────────────
# Training step
# ─────────────────────────────────────────────────────────────────────────────

# NOTE: intentionally UNdecorated. XLA (`jit_compile`) is a CPU pessimization
# here (a huge one-time compile with no runtime win) and is only worth it on
# GPU. The training function wraps this body in a `tf.function` at setup time
# with `jit_compile` gated on the GPU probe — see `_train_step = tf.function(...)`
# near the model-build site. Keep this body pure/traceable.
def _train_step_impl(
    model:     tf.keras.Model,
    optimizer: tf.keras.optimizers.Optimizer,
    images:    tf.Tensor,
    w_cls:     tf.Tensor,
    w_box:     tf.Tensor,
    w_dfl:     tf.Tensor,
    w_o2o:     tf.Tensor,
    # ── o2m targets (one-to-many, topk=10) ──────────────────────────────────
    cls_t_p3: tf.Tensor, box_t_p3: tf.Tensor, ltrb_t_p3: tf.Tensor, fg_p3: tf.Tensor,
    cls_t_p4: tf.Tensor, box_t_p4: tf.Tensor, ltrb_t_p4: tf.Tensor, fg_p4: tf.Tensor,
    cls_t_p5: tf.Tensor, box_t_p5: tf.Tensor, ltrb_t_p5: tf.Tensor, fg_p5: tf.Tensor,
    # ── o2o targets (one-to-one, topk=1) ────────────────────────────────────
    o2o_cls_t_p3: tf.Tensor, o2o_box_t_p3: tf.Tensor, o2o_ltrb_t_p3: tf.Tensor, o2o_fg_p3: tf.Tensor,
    o2o_cls_t_p4: tf.Tensor, o2o_box_t_p4: tf.Tensor, o2o_ltrb_t_p4: tf.Tensor, o2o_fg_p4: tf.Tensor,
    o2o_cls_t_p5: tf.Tensor, o2o_box_t_p5: tf.Tensor, o2o_ltrb_t_p5: tf.Tensor, o2o_fg_p5: tf.Tensor,
    # ── anchors / strides / dims ─────────────────────────────────────────────
    anchor_p3: tf.Tensor, anchor_p4: tf.Tensor, anchor_p5: tf.Tensor,
    stride_p3: tf.Tensor, stride_p4: tf.Tensor, stride_p5: tf.Tensor,
    H_img: int, W_img: int, reg_max: int,
) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """
    One gradient step — dual-path (o2m + o2o) loss.

    Both paths share the same backbone/neck/head forward pass; only the
    assignment targets differ.  The total gradient is:
        total = o2m_total + WEIGHT_O2O_PATH * o2o_total

    Returns:
        (total_loss, o2m_total, o2o_total)  — all scalar tf.float32 tensors
    """
    from app.ml.yolo_pro.losses import yolo_pro_loss

    with tf.GradientTape() as tape:
        preds = model(images, training=True)
        o2m_total = tf.constant(0.0, dtype=tf.float32)
        o2o_total = tf.constant(0.0, dtype=tf.float32)

        for (scale_name,
             cls_t,     box_t,     ltrb_t,     fg,
             o2o_cls_t, o2o_box_t, o2o_ltrb_t, o2o_fg,
             anch, stride_t) in [
            ("p3", cls_t_p3, box_t_p3, ltrb_t_p3, fg_p3,
                   o2o_cls_t_p3, o2o_box_t_p3, o2o_ltrb_t_p3, o2o_fg_p3,
                   anchor_p3, stride_p3),
            ("p4", cls_t_p4, box_t_p4, ltrb_t_p4, fg_p4,
                   o2o_cls_t_p4, o2o_box_t_p4, o2o_ltrb_t_p4, o2o_fg_p4,
                   anchor_p4, stride_p4),
            ("p5", cls_t_p5, box_t_p5, ltrb_t_p5, fg_p5,
                   o2o_cls_t_p5, o2o_box_t_p5, o2o_ltrb_t_p5, o2o_fg_p5,
                   anchor_p5, stride_p5),
        ]:
            cls_pred = tf.reshape(
                preds[f"cls_{scale_name}"],
                (tf.shape(images)[0], -1, tf.shape(cls_t)[-1])
            )
            reg_pred = tf.reshape(
                preds[f"reg_{scale_name}"],
                (tf.shape(images)[0], -1, 4 * reg_max)
            )
            # Under the mixed_float16 policy the head outputs are float16; run
            # the entire decode + loss computation in float32 so the loss (and
            # therefore accuracy) is numerically identical to the fp32 path.
            cls_pred = tf.cast(cls_pred, tf.float32)
            reg_pred = tf.cast(reg_pred, tf.float32)

            # Decode once — shared by both loss paths (same head output)
            pred_boxes = _decode_reg_scale(
                reg_pred, anch,
                stride=tf.cast(stride_t, tf.float32),
                H_img=H_img, W_img=W_img, reg_max=reg_max,
            )

            # ── o2m loss (one-to-many, topk=10) ──────────────────────────────
            o2m_losses = yolo_pro_loss(
                cls_pred=cls_pred,   reg_pred=reg_pred,
                cls_targets=cls_t,   box_targets=box_t,
                ltrb_targets=ltrb_t, fg_mask=fg,
                pred_boxes=pred_boxes, reg_max=reg_max,
                w_cls=w_cls, w_box=w_box, w_dfl=w_dfl,
            )
            o2m_total = o2m_total + o2m_losses["loss_total"]

            # ── o2o loss (one-to-one, topk=1) ────────────────────────────────
            o2o_losses = yolo_pro_loss(
                cls_pred=cls_pred,       reg_pred=reg_pred,
                cls_targets=o2o_cls_t,   box_targets=o2o_box_t,
                ltrb_targets=o2o_ltrb_t, fg_mask=o2o_fg,
                pred_boxes=pred_boxes,   reg_max=reg_max,
                w_cls=w_cls, w_box=w_box, w_dfl=w_dfl,
            )
            o2o_total = o2o_total + o2o_losses["loss_total"]

        total_loss = o2m_total + tf.cast(w_o2o, tf.float32) * o2o_total
        # Mixed-precision loss scaling (Keras 3 API): under a LossScaleOptimizer
        # (GPU/MP path) scale the float32 loss up inside the tape so small
        # gradients survive the float16 backward pass; apply_gradients below then
        # unscales, clips, skips non-finite steps, and adjusts the dynamic scale.
        # On the fp32/CPU path the optimizer is a plain AdamW (no LSO wrap), so
        # there is nothing to scale — differentiate the raw loss.  Both paths
        # yield identical fp32 loss values (scaling is a power-of-two no-op), and
        # the isinstance check resolves at trace time (the optimizer identity is
        # fixed for the life of the compiled step).
        if isinstance(optimizer, tf.keras.mixed_precision.LossScaleOptimizer):
            loss_for_grad = optimizer.scale_loss(total_loss)
        else:
            loss_for_grad = total_loss

    grads = tape.gradient(loss_for_grad, model.trainable_variables)

    # Finiteness flag drives EMA + accumulator suppression on the host side (a
    # bad step must not update the EMA or pollute the epoch mean).  Scaling by a
    # finite, non-zero factor preserves finiteness, so reading it off `grads`
    # (scaled or not) is identical — and matches the skip decision the optimizer
    # makes internally on the unscaled gradients.
    _finite_flags = [
        tf.reduce_all(tf.math.is_finite(g))
        for g in grads if g is not None
    ]
    if _finite_flags:
        grads_finite = tf.reduce_all(tf.stack(_finite_flags))
    else:
        grads_finite = tf.constant(False)
    step_finite = tf.logical_and(grads_finite, tf.math.is_finite(total_loss))

    # apply_gradients: under the LSO this unscales → clips (inner
    # global_clipnorm=10.0) → applies, or skips + lowers the loss scale on
    # non-finite gradients; on the plain AdamW it clips → applies directly.
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return total_loss, o2m_total, o2o_total, step_finite


@tf.function
def _val_step(
    model:     tf.keras.Model,
    images:    tf.Tensor,
    w_cls:     tf.Tensor,
    w_box:     tf.Tensor,
    w_dfl:     tf.Tensor,
    w_o2o:     tf.Tensor,
    # ── o2m targets ──────────────────────────────────────────────────────────
    cls_t_p3: tf.Tensor, box_t_p3: tf.Tensor, ltrb_t_p3: tf.Tensor, fg_p3: tf.Tensor,
    cls_t_p4: tf.Tensor, box_t_p4: tf.Tensor, ltrb_t_p4: tf.Tensor, fg_p4: tf.Tensor,
    cls_t_p5: tf.Tensor, box_t_p5: tf.Tensor, ltrb_t_p5: tf.Tensor, fg_p5: tf.Tensor,
    # ── o2o targets ──────────────────────────────────────────────────────────
    o2o_cls_t_p3: tf.Tensor, o2o_box_t_p3: tf.Tensor, o2o_ltrb_t_p3: tf.Tensor, o2o_fg_p3: tf.Tensor,
    o2o_cls_t_p4: tf.Tensor, o2o_box_t_p4: tf.Tensor, o2o_ltrb_t_p4: tf.Tensor, o2o_fg_p4: tf.Tensor,
    o2o_cls_t_p5: tf.Tensor, o2o_box_t_p5: tf.Tensor, o2o_ltrb_t_p5: tf.Tensor, o2o_fg_p5: tf.Tensor,
    # ── anchors / strides / dims ─────────────────────────────────────────────
    anchor_p3: tf.Tensor, anchor_p4: tf.Tensor, anchor_p5: tf.Tensor,
    stride_p3: tf.Tensor, stride_p4: tf.Tensor, stride_p5: tf.Tensor,
    H_img: int, W_img: int, reg_max: int,
) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    """
    Forward pass only (no gradient updates) — mirrors _train_step dual-path loss.

    Returns:
        total_loss   : o2m + WEIGHT_O2O_PATH * o2o  (used as the primary
                       checkpoint / early-stopping signal via val_loss).
        box_dfl_loss : sum of CIoU + DFL over all three scales (o2m path only).
                       Direct localization quality proxy; stored as a diagnostic
                       metric alongside val_loss.
        cls_loss     : sum of VFL classification loss over all three scales
                       (o2m path only).  Tracked separately so operators can
                       see whether the cls head is still converging independently
                       of the box-dominated total val_loss.  A falling val_cls
                       while val_loss plateaus signals that cls is still learning
                       after box regression has settled — useful for diagnosing
                       tp_score_mean < fp_score_mean regressions.
        cls_pos_loss : VFL positive-branch contribution summed across all three
                       scales (o2m path only).  Diagnostic only — does not gate
                       checkpoints.  Lets operators see whether TP score is
                       rising independently of background drift.  Sums with
                       ``cls_neg_loss`` to ``cls_loss``.
        cls_neg_loss : VFL negative-branch contribution summed across all three
                       scales (o2m path only).  Diagnostic only.  Rising
                       ``cls_neg_loss`` while ``cls_pos_loss`` is flat is the
                       FP-confidence-drift signature ``VFL_NEG_WEIGHT_FLOOR``
                       was added to suppress.
    """
    preds = model(images, training=False)
    return _val_loss_body(
        preds,
        w_cls, w_box, w_dfl, w_o2o,
        cls_t_p3, box_t_p3, ltrb_t_p3, fg_p3,
        cls_t_p4, box_t_p4, ltrb_t_p4, fg_p4,
        cls_t_p5, box_t_p5, ltrb_t_p5, fg_p5,
        o2o_cls_t_p3, o2o_box_t_p3, o2o_ltrb_t_p3, o2o_fg_p3,
        o2o_cls_t_p4, o2o_box_t_p4, o2o_ltrb_t_p4, o2o_fg_p4,
        o2o_cls_t_p5, o2o_box_t_p5, o2o_ltrb_t_p5, o2o_fg_p5,
        anchor_p3, anchor_p4, anchor_p5,
        stride_p3, stride_p4, stride_p5,
        H_img, W_img, reg_max,
    )


@tf.function
def _val_step_from_preds(
    cls_pred_p3: tf.Tensor, reg_pred_p3: tf.Tensor,
    cls_pred_p4: tf.Tensor, reg_pred_p4: tf.Tensor,
    cls_pred_p5: tf.Tensor, reg_pred_p5: tf.Tensor,
    w_cls:     tf.Tensor,
    w_box:     tf.Tensor,
    w_dfl:     tf.Tensor,
    w_o2o:     tf.Tensor,
    cls_t_p3: tf.Tensor, box_t_p3: tf.Tensor, ltrb_t_p3: tf.Tensor, fg_p3: tf.Tensor,
    cls_t_p4: tf.Tensor, box_t_p4: tf.Tensor, ltrb_t_p4: tf.Tensor, fg_p4: tf.Tensor,
    cls_t_p5: tf.Tensor, box_t_p5: tf.Tensor, ltrb_t_p5: tf.Tensor, fg_p5: tf.Tensor,
    o2o_cls_t_p3: tf.Tensor, o2o_box_t_p3: tf.Tensor, o2o_ltrb_t_p3: tf.Tensor, o2o_fg_p3: tf.Tensor,
    o2o_cls_t_p4: tf.Tensor, o2o_box_t_p4: tf.Tensor, o2o_ltrb_t_p4: tf.Tensor, o2o_fg_p4: tf.Tensor,
    o2o_cls_t_p5: tf.Tensor, o2o_box_t_p5: tf.Tensor, o2o_ltrb_t_p5: tf.Tensor, o2o_fg_p5: tf.Tensor,
    anchor_p3: tf.Tensor, anchor_p4: tf.Tensor, anchor_p5: tf.Tensor,
    stride_p3: tf.Tensor, stride_p4: tf.Tensor, stride_p5: tf.Tensor,
    H_img: int, W_img: int, reg_max: int,
) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    """
    Loss-only twin of :func:`_val_step` that consumes ALREADY-COMPUTED head
    predictions instead of running a second forward pass.

    The per-epoch val-loss pass already forwards ``model(v_imgs, training=False)``
    to obtain the predictions it decodes for NumPy target assignment; feeding
    those same tensors here removes the redundant second forward inside
    ``_val_step`` (2× → 1× forward over X_test/epoch).  The weights are identical
    between the two calls (EMA already applied, no update between), so the loss is
    numerically identical to calling ``_val_step(model, v_imgs, …)``.
    """
    preds = {
        "cls_p3": cls_pred_p3, "reg_p3": reg_pred_p3,
        "cls_p4": cls_pred_p4, "reg_p4": reg_pred_p4,
        "cls_p5": cls_pred_p5, "reg_p5": reg_pred_p5,
    }
    return _val_loss_body(
        preds,
        w_cls, w_box, w_dfl, w_o2o,
        cls_t_p3, box_t_p3, ltrb_t_p3, fg_p3,
        cls_t_p4, box_t_p4, ltrb_t_p4, fg_p4,
        cls_t_p5, box_t_p5, ltrb_t_p5, fg_p5,
        o2o_cls_t_p3, o2o_box_t_p3, o2o_ltrb_t_p3, o2o_fg_p3,
        o2o_cls_t_p4, o2o_box_t_p4, o2o_ltrb_t_p4, o2o_fg_p4,
        o2o_cls_t_p5, o2o_box_t_p5, o2o_ltrb_t_p5, o2o_fg_p5,
        anchor_p3, anchor_p4, anchor_p5,
        stride_p3, stride_p4, stride_p5,
        H_img, W_img, reg_max,
    )


def _val_loss_body(
    preds:     dict,
    w_cls:     tf.Tensor,
    w_box:     tf.Tensor,
    w_dfl:     tf.Tensor,
    w_o2o:     tf.Tensor,
    cls_t_p3: tf.Tensor, box_t_p3: tf.Tensor, ltrb_t_p3: tf.Tensor, fg_p3: tf.Tensor,
    cls_t_p4: tf.Tensor, box_t_p4: tf.Tensor, ltrb_t_p4: tf.Tensor, fg_p4: tf.Tensor,
    cls_t_p5: tf.Tensor, box_t_p5: tf.Tensor, ltrb_t_p5: tf.Tensor, fg_p5: tf.Tensor,
    o2o_cls_t_p3: tf.Tensor, o2o_box_t_p3: tf.Tensor, o2o_ltrb_t_p3: tf.Tensor, o2o_fg_p3: tf.Tensor,
    o2o_cls_t_p4: tf.Tensor, o2o_box_t_p4: tf.Tensor, o2o_ltrb_t_p4: tf.Tensor, o2o_fg_p4: tf.Tensor,
    o2o_cls_t_p5: tf.Tensor, o2o_box_t_p5: tf.Tensor, o2o_ltrb_t_p5: tf.Tensor, o2o_fg_p5: tf.Tensor,
    anchor_p3: tf.Tensor, anchor_p4: tf.Tensor, anchor_p5: tf.Tensor,
    stride_p3: tf.Tensor, stride_p4: tf.Tensor, stride_p5: tf.Tensor,
    H_img: int, W_img: int, reg_max: int,
) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    """
    Shared dual-path val-loss math for a set of head predictions ``preds`` (dict
    with keys ``cls_p{3,4,5}`` / ``reg_p{3,4,5}``).  Extracted verbatim from the
    old ``_val_step`` loop body so both ``_val_step`` (forward + loss) and
    ``_val_step_from_preds`` (loss only) compute identical numbers — there is a
    single source of truth for the loss.  Pure tensor ops, so it traces into
    either ``@tf.function`` caller.
    """
    from app.ml.yolo_pro.losses import yolo_pro_loss, varifocal_loss_components

    o2m_total    = tf.constant(0.0, dtype=tf.float32)
    o2o_total    = tf.constant(0.0, dtype=tf.float32)
    box_dfl_total = tf.constant(0.0, dtype=tf.float32)
    cls_total     = tf.constant(0.0, dtype=tf.float32)
    cls_pos_total = tf.constant(0.0, dtype=tf.float32)
    cls_neg_total = tf.constant(0.0, dtype=tf.float32)
    _B_imgs = tf.shape(preds["cls_p3"])[0]

    for (scale_name,
         cls_t,     box_t,     ltrb_t,     fg,
         o2o_cls_t, o2o_box_t, o2o_ltrb_t, o2o_fg,
         anch, stride_t) in [
        ("p3", cls_t_p3, box_t_p3, ltrb_t_p3, fg_p3,
               o2o_cls_t_p3, o2o_box_t_p3, o2o_ltrb_t_p3, o2o_fg_p3,
               anchor_p3, stride_p3),
        ("p4", cls_t_p4, box_t_p4, ltrb_t_p4, fg_p4,
               o2o_cls_t_p4, o2o_box_t_p4, o2o_ltrb_t_p4, o2o_fg_p4,
               anchor_p4, stride_p4),
        ("p5", cls_t_p5, box_t_p5, ltrb_t_p5, fg_p5,
               o2o_cls_t_p5, o2o_box_t_p5, o2o_ltrb_t_p5, o2o_fg_p5,
               anchor_p5, stride_p5),
    ]:
        cls_pred = tf.reshape(
            preds[f"cls_{scale_name}"],
            (_B_imgs, -1, tf.shape(cls_t)[-1])
        )
        reg_pred = tf.reshape(
            preds[f"reg_{scale_name}"],
            (_B_imgs, -1, 4 * reg_max)
        )
        # Match _train_step: compute the val loss in float32 regardless of the
        # (mixed_float16) head output dtype so the checkpoint/early-stop signal
        # is unchanged from the fp32 baseline.
        cls_pred = tf.cast(cls_pred, tf.float32)
        reg_pred = tf.cast(reg_pred, tf.float32)
        pred_boxes = _decode_reg_scale(
            reg_pred, anch,
            stride=tf.cast(stride_t, tf.float32),
            H_img=H_img, W_img=W_img, reg_max=reg_max,
        )
        o2m_losses = yolo_pro_loss(
            cls_pred=cls_pred,   reg_pred=reg_pred,
            cls_targets=cls_t,   box_targets=box_t,
            ltrb_targets=ltrb_t, fg_mask=fg,
            pred_boxes=pred_boxes, reg_max=reg_max,
            w_cls=w_cls, w_box=w_box, w_dfl=w_dfl,
        )
        o2m_total    = o2m_total    + o2m_losses["loss_total"]
        # Accumulate box+DFL (localization quality) and VFL cls across scales.
        # Both are diagnostic — neither drives checkpointing directly.
        box_dfl_total = box_dfl_total + o2m_losses["loss_box"] + o2m_losses["loss_dfl"]
        cls_total     = cls_total     + o2m_losses["loss_cls"]
        # VFL positive / negative components — diagnostic only.  Sum across
        # scales identically to ``cls_total`` so ``pos + neg == cls_total``.
        _l_cls_pos, _l_cls_neg = varifocal_loss_components(
            cls_pred, cls_t, fg,
        )
        cls_pos_total = cls_pos_total + _l_cls_pos
        cls_neg_total = cls_neg_total + _l_cls_neg

        o2o_losses = yolo_pro_loss(
            cls_pred=cls_pred,       reg_pred=reg_pred,
            cls_targets=o2o_cls_t,   box_targets=o2o_box_t,
            ltrb_targets=o2o_ltrb_t, fg_mask=o2o_fg,
            pred_boxes=pred_boxes,   reg_max=reg_max,
            w_cls=w_cls, w_box=w_box, w_dfl=w_dfl,
        )
        o2o_total = o2o_total + o2o_losses["loss_total"]

    return (
        o2m_total + tf.cast(w_o2o, tf.float32) * o2o_total,
        box_dfl_total,
        cls_total,
        cls_pos_total,
        cls_neg_total,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Pre-compute assignment targets
# ─────────────────────────────────────────────────────────────────────────────

def _flip_boxes_horizontal(
    boxes_list: List[Optional[List[dict]]],
) -> List[Optional[List[dict]]]:
    """
    Mirror every bounding box horizontally: new_x = 1 - (x + w).

    Boxes are assumed to be in normalised [0, 1] coordinates with the
    standard {x, y, w, h} key layout used throughout the pipeline.
    Boxes where the flipped x-coordinate would be out of [0, 1] are
    clipped — this only occurs for improperly-annotated boxes that
    already lie outside the image boundary.
    """
    flipped_list: List[Optional[List[dict]]] = []
    for boxes in boxes_list:
        if boxes is None:
            flipped_list.append(None)
            continue
        new_boxes = []
        for b in boxes:
            bw = float(b.get("w", 0))
            bx = float(b.get("x", 0))
            new_x = max(0.0, min(1.0, 1.0 - bx - bw))
            new_boxes.append({**b, "x": new_x})
        flipped_list.append(new_boxes)
    return flipped_list


def _precompute_targets(
    boxes_list: List[Optional[List[dict]]],
    label_map: dict,
    anchor_grids: List[Tuple[np.ndarray, int]],
    H: int,
    W: int,
    n_classes: int,
    reg_max: int,
) -> List[tuple]:
    """
    Build per-sample assignment targets for all three scales — both o2m and o2o.

    Returns list of 24-tuples:
        [0-3]   o2m_cls_p3, o2m_box_p3, o2m_ltrb_p3, o2m_fg_p3
        [4-7]   o2m_cls_p4, o2m_box_p4, o2m_ltrb_p4, o2m_fg_p4
        [8-11]  o2m_cls_p5, o2m_box_p5, o2m_ltrb_p5, o2m_fg_p5
        [12-15] o2o_cls_p3, o2o_box_p3, o2o_ltrb_p3, o2o_fg_p3
        [16-19] o2o_cls_p4, o2o_box_p4, o2o_ltrb_p4, o2o_fg_p4
        [20-23] o2o_cls_p5, o2o_box_p5, o2o_ltrb_p5, o2o_fg_p5

    The first 12 entries are identical to the previous 12-tuple format so that
    code reading indices [0-11] (e.g. diagnostic logging) is unchanged.
    Uses static geometric fallback (no live predictions) for both paths.
    """
    results = []
    for boxes in boxes_list:
        sample_targets = []
        # ── o2m path: topk=10, one-to-many ───────────────────────────────────
        for anchor_pts, stride in anchor_grids:
            ct, bt, lt, fg = _assign_boxes_to_anchors(
                boxes=boxes or [],
                label_map=label_map,
                anchor_pts=anchor_pts,
                stride=stride,
                H_img=H,
                W_img=W,
                n_classes=n_classes,
                reg_max=reg_max,
            )
            sample_targets.extend([ct, bt, lt, fg])
        # ── o2o path: topk=1, one-to-one ─────────────────────────────────────
        for anchor_pts, stride in anchor_grids:
            ct, bt, lt, fg = _assign_boxes_to_anchors_o2o(
                boxes=boxes or [],
                label_map=label_map,
                anchor_pts=anchor_pts,
                stride=stride,
                H_img=H,
                W_img=W,
                n_classes=n_classes,
                reg_max=reg_max,
            )
            sample_targets.extend([ct, bt, lt, fg])
        results.append(tuple(sample_targets))
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Mosaic augmentation helpers
# ─────────────────────────────────────────────────────────────────────────────

def _resize_img(img: np.ndarray, h: int, w: int) -> np.ndarray:
    """Bilinear resize; preserves float32 without uint8 round-trip."""
    return tf.image.resize(img, [h, w], method="bilinear").numpy()


def _hsv_jitter(
    img: np.ndarray,
    h_gain: float = 0.015,
    s_gain: float = 0.7,
    v_gain: float = 0.4,
) -> np.ndarray:
    """
    YOLOv8-style HSV colour jitter for 3-channel float32 RGB images.

    Independently perturbs hue, saturation, and value with random
    scale factors drawn from uniform distributions centred at 1.0.
    Hue wraps; saturation and value are clipped to [0, 1].
    Falls back to simple brightness scaling for non-RGB images.
    """
    if img.shape[2] != 3:
        factor = np.float32(0.7 + 0.6 * np.random.random())
        return np.clip(img * factor, 0.0, 1.0)
    r = np.random.uniform(-1.0, 1.0, 3) * [h_gain, s_gain, v_gain] + 1.0
    hsv = tf.image.rgb_to_hsv(img.astype(np.float32)).numpy()
    hsv[..., 0] = np.mod(hsv[..., 0] * r[0], 1.0)
    hsv[..., 1] = np.clip(hsv[..., 1] * r[1], 0.0, 1.0)
    hsv[..., 2] = np.clip(hsv[..., 2] * r[2], 0.0, 1.0)
    return tf.image.hsv_to_rgb(hsv).numpy()


def _make_mosaic(
    imgs: List[np.ndarray],
    boxes_list: List[Optional[List[dict]]],
    H: int,
    W: int,
) -> Tuple[np.ndarray, List[dict]]:
    """
    Compose a 2×2 mosaic from four training images.

    Each source image is bilinearly resized to fill its quadrant.
    Box coordinates are transformed into the mosaic's normalised [0,1]
    space; boxes that shrink below 2 px in either dimension after
    quadrant-boundary clipping are discarded.

    The split point is sampled uniformly in [H//4, 3H//4] × [W//4, 3W//4]
    so every quadrant is large enough to contain meaningful context.

    Args:
        imgs       : four (h, w, C) float32 images in [0, 1]
        boxes_list : four Optional[List[dict]] of normalised xywh boxes
        H, W       : output mosaic height/width (same as training image size)

    Returns:
        mosaic   : (H, W, C) float32 composite image
        combined : normalised xywh box list for the full mosaic
    """
    C = imgs[0].shape[2]
    mosaic = np.zeros((H, W, C), dtype=np.float32)

    # Split point constrained to [35 %, 65 %] of each dimension.
    # Narrower than the previous [25 %, 75 %] window: ensures every quadrant
    # is at least 35 % of the full image side, reducing extreme scale
    # compression of objects and shortening the DFL recalibration needed in
    # the close-mosaic phase.
    cy = int(np.random.uniform(int(0.35 * H), int(0.65 * H)))
    cx = int(np.random.uniform(int(0.35 * W), int(0.65 * W)))

    # (row_start, row_end, col_start, col_end) for each quadrant
    quads: List[Tuple[int, int, int, int]] = [
        (0,  cy, 0,  cx),   # top-left
        (0,  cy, cx, W),    # top-right
        (cy, H,  0,  cx),   # bottom-left
        (cy, H,  cx, W),    # bottom-right
    ]

    combined: List[dict] = []

    for idx, (r1, r2, c1, c2) in enumerate(quads):
        dst_h = r2 - r1
        dst_w = c2 - c1
        if dst_h <= 0 or dst_w <= 0:
            continue

        src   = imgs[idx]
        boxes = boxes_list[idx] or []

        mosaic[r1:r2, c1:c2] = _resize_img(src, dst_h, dst_w)

        # Quadrant offset and scale in normalised mosaic coords
        ox = c1 / W;  oy = r1 / H
        sx = dst_w / W;  sy = dst_h / H

        for b in boxes:
            bx = float(b.get("x", 0));  by = float(b.get("y", 0))
            bw = float(b.get("w", 0));  bh = float(b.get("h", 0))

            mx = ox + bx * sx
            my = oy + by * sy
            mw = bw * sx
            mh = bh * sy

            # Clip to mosaic boundary and measure surviving box
            mx2 = min(1.0, mx + mw);  my2 = min(1.0, my + mh)
            mx  = max(0.0, mx);       my  = max(0.0, my)
            mw  = mx2 - mx;           mh  = my2 - my

            # Drop boxes too small to provide useful supervision
            if mw < 2.0 / W or mh < 2.0 / H:
                continue

            combined.append({**b, "x": mx, "y": my, "w": mw, "h": mh})

    return mosaic, combined


# ─────────────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────────────

def run_yolo_pro_training(
    job,
    impulse,
    X_train: np.ndarray,
    boxes_train: List[Optional[List[dict]]],
    X_test: np.ndarray,
    boxes_test: List[Optional[List[dict]]],
    label_names: List[str],
    label_map: dict,
    input_shape: Tuple[int, int, int],
    storage,
    db,
    job_id: str,
) -> dict:
    """
    Full YOLO-Pro training run.

    Called from training_worker.py when is_yolo_pro(selected_architecture).
    """
    from app.ml.yolo_pro import build_yolo_pro
    from app.ml.yolo_pro.train_config import TrainConfig
    from app.ml.yolo_pro.export import export_tflite_int8, export_tflite_float32
    from app.models.user import TrainedModel, JobStatus

    H, W, C = input_shape
    n_classes = len(label_names)
    reg_max   = int((job.extra_params or {}).get("reg_max", 16))
    _size_raw = (job.extra_params or {}).get("size")
    if not _size_raw:
        raise ValueError(
            "YOLO-Pro training requires a model size in extra_params['size']. "
            "Expected one of: nano, tiny, small, medium, large."
        )
    from app.ml.yolo_pro.config import CONFIGS as _YOLO_PRO_CONFIGS
    size = str(_size_raw).lower().strip()
    if size not in _YOLO_PRO_CONFIGS:
        valid = ", ".join(_YOLO_PRO_CONFIGS)
        raise ValueError(f"Unknown YOLO-Pro size '{size}'. Valid options: {valid}")
    strides   = [8, 16, 32]
    _requested_epochs = job.epochs
    policy = _compute_stabilization_policy(
        n_train=len(X_train),
        requested_epochs=_requested_epochs,
        user_batch=job.batch_size,
    )
    policy["patience"]       = _resolve_size_aware_patience(size, policy["patience"])
    policy["close_mosaic_n"] = _resolve_size_aware_close_mosaic_n(
        size, policy["close_mosaic_n"], policy["effective_epochs"]
    )
    dataset_stats = _compute_dataset_complexity_stats(boxes_train, label_names, label_map)
    balance_policy = _resolve_training_balance_policy(dataset_stats, policy)

    logger.info(
        f"[{job_id}] YOLO-Pro training — size={size}  "
        f"input={input_shape}  n_classes={n_classes}  reg_max={reg_max}"
    )

    # Check before model build (can be slow for large YOLO-Pro configs)
    _raise_if_cancelled(job_id)

    # ── Mixed precision (float16 compute, float32 master weights) ────────────
    # Enable the mixed_float16 policy ONLY when a GPU is present — float16 is a
    # slowdown on CPU and offers no benefit.  The policy is set *before* the
    # model is built so every layer captures float16 compute / float32
    # variables, then restored to the previous (fp32) global policy immediately
    # after: already-constructed layers keep their per-layer policy, while any
    # model rebuilt later (e.g. the TFLite export/decode graph) stays fp32.
    _prev_mp_policy      = tf.keras.mixed_precision.global_policy()
    _use_mixed_precision = bool(tf.config.list_physical_devices("GPU"))
    if _use_mixed_precision:
        tf.keras.mixed_precision.set_global_policy("mixed_float16")
        logger.info(
            f"[{job_id}] Mixed precision ENABLED — policy=mixed_float16, "
            f"loss-scaled AdamW (fp32 master weights), XLA jit_compile on train step."
        )

    model = build_yolo_pro(
        input_shape=input_shape,
        num_classes=n_classes,
        size=size,
        reg_max=reg_max,
        prior_prob=float(balance_policy["prior_prob"]),
    )
    if _use_mixed_precision:
        # Restore fp32 as the *global* default; the built model's layers retain
        # their mixed_float16 policy captured at construction above.
        tf.keras.mixed_precision.set_global_policy(_prev_mp_policy)
    model.summary(print_fn=logger.info)
    _dbg_cfg = _YOLO_PRO_CONFIGS[size]
    logger.info(
        f"[{job_id}] Model build — size={size}  "
        f"depths={list(_dbg_cfg.depths)}  channels={list(_dbg_cfg.channels)}  "
        f"total_params={model.count_params():,}  prior_prob={float(balance_policy['prior_prob']):.4f}"
    )

    # ── Graph-compile the train step + dynamic-TAL forward (bottlenecks #1/#2) ─
    # Two hot paths were running eagerly / paying for CPU-XLA:
    #   1. The dynamic-TAL inference forward in `_build_batch` ran the whole
    #      network eagerly every step. Wrap it once in a `tf.function` so it
    #      executes as a graph. No XLA (jit) needed — graph mode alone kills the
    #      per-op Python/eager overhead; `reduce_retracing` tolerates the
    #      trailing partial batch's smaller leading dim without a full retrace.
    #   2. `_train_step` was statically `@tf.function(jit_compile=True)`, so this
    #      CPU-only box paid a large one-time XLA compile for zero runtime gain
    #      (and float32, since MP is GPU-gated). Gate XLA on the SAME GPU probe
    #      used for mixed precision so XLA and MP flip together: XLA on GPU only.
    _use_xla = _use_mixed_precision  # reuse the :3914 GPU probe — MP & XLA together
    _train_step = tf.function(
        _train_step_impl, jit_compile=_use_xla, reduce_retracing=True,
    )
    _tal_infer = tf.function(
        lambda x: model(x, training=False), reduce_retracing=True,
    )
    logger.info(
        f"[{job_id}] Train-step compile: graph=ON  "
        f"jit_compile(XLA)={'ON' if _use_xla else 'OFF (CPU — no GPU)'}  "
        f"dynamic-TAL forward=graph-compiled (no XLA)."
    )

    # ── Stabilization policy ────────────────────────────────────────────────
    # Compute the effective training plan before any schedule or EMA is built.
    # This replaces epoch-count special-casing with a single dataset-aware
    # policy that derives all schedule parameters from total gradient steps.
    import math as _math
    # If the user did not explicitly set a batch size, pick a default that
    _requested_epochs = job.epochs

    policy = _compute_stabilization_policy(
        n_train=len(X_train),
        requested_epochs=_requested_epochs,
        user_batch=job.batch_size,   # None → auto-select; explicit → honored with warning
    )
    # Apply model-size-specific overrides (tiny only; nano: no change).
    policy["patience"]       = _resolve_size_aware_patience(size, policy["patience"])
    policy["close_mosaic_n"] = _resolve_size_aware_close_mosaic_n(
        size, policy["close_mosaic_n"], policy["effective_epochs"]
    )

    if policy["no_training_data"]:
        raise ValueError("No training data — cannot start YOLO-Pro training.")

    batch_size       = policy["eff_batch_size"]
    # User-requested epochs is preserved exactly (never mutated).
    # effective_epochs is the internal training-loop budget (may be higher
    # for tiny datasets to reach MIN_TOTAL_STEPS); it is NEVER surfaced as
    # `epochs` in UI-facing result fields.
    effective_epochs = policy["effective_epochs"]
    steps_per_epoch  = policy["steps_per_epoch"]
    total_steps      = policy["total_steps"]
    _close_mosaic_n  = policy["close_mosaic_n"]
    PATIENCE         = policy["patience"]

    # Log policy decisions so every compensation is visible in the training log.
    if policy["batch_size_warning"]:
        logger.warning(f"[{job_id}] Stabilizer: {policy['batch_size_warning']}")
    if policy["batch_size_reduced"]:
        logger.warning(
            f"[{job_id}] Stabilizer: batch_size reduced to {batch_size} "
            f"to reach steps_per_epoch >= 16."
        )
    if policy["epochs_expanded"]:
        logger.warning(
            f"[{job_id}] Stabilizer: internal effective_epochs "
            f"{_requested_epochs} → {effective_epochs} "
            f"(total_steps below {MIN_TOTAL_STEPS}; user-requested epochs "
            f"{_requested_epochs} preserved in outputs)."
        )
    if policy["tiny_dataset_mode"]:
        logger.info(
            f"[{job_id}] Tiny-dataset mode ACTIVE "
            f"(n_train={len(X_train)}, total_steps={total_steps}): "
            f"mosaic_epochs={effective_epochs - _close_mosaic_n}  "
            f"close_mosaic_n={_close_mosaic_n}/{effective_epochs} epochs, "
            f"stabilized TAL transition, fast EMA."
        )

    logger.info(
        f"[{job_id}] Stabilization policy — "
        f"requested: epochs={_requested_epochs} batch={job.batch_size!r} | "
        f"internal: effective_epochs={effective_epochs} batch={batch_size} "
        f"steps_per_epoch={steps_per_epoch} total_steps={total_steps} "
        f"tiny_dataset_mode={policy['tiny_dataset_mode']}"
    )
    logger.info(
        f"[{job_id}] Dataset complexity â€” "
        f"images={dataset_stats['num_images']}  boxes={dataset_stats['total_boxes']}  "
        f"avg_boxes_per_image={dataset_stats['avg_boxes_per_image']:.3f}  "
        f"classes={dataset_stats['num_classes']} active_classes={dataset_stats['active_class_count']}  "
        f"max_class_fraction={dataset_stats['class_fraction_max']:.3f}"
    )
    logger.info(
        f"[{job_id}] Balance preset â€” preset={balance_policy['preset']}  "
        f"w_cls={balance_policy['w_cls']:.2f}  w_box={balance_policy['w_box']:.2f}  "
        f"w_dfl={balance_policy['w_dfl']:.2f}  w_o2o={balance_policy['w_o2o']:.2f}  "
        f"prior_prob={balance_policy['prior_prob']:.4f}  reason={balance_policy['reason']}"
    )
    logger.info(f"[{job_id}] Final batch size: {batch_size}")

    train_cfg = TrainConfig(
        size=size,
        num_classes=n_classes,
        input_size=(H, W),
        epochs=effective_epochs,               # internal training budget
        warmup_epochs=policy["warmup_epochs"],
        mosaic_epochs=effective_epochs - _close_mosaic_n,
        base_lr=_resolve_size_aware_learning_rate(size, job.learning_rate),
        batch_size=batch_size,                 # effective batch size
    )

    # LR schedule: calibrated to effective total_steps so cosine reaches
    # min_lr exactly at the last gradient step (not early/late due to
    # under/over-counting steps_per_epoch).
    lr_schedule_fn = train_cfg.build_lr_schedule(steps_per_epoch)
    optimizer      = train_cfg.build_optimizer(lr_schedule_fn(0))
    # Wrap in a LossScaleOptimizer ONLY under mixed precision (GPU).  There, the
    # dynamic loss scale is what keeps small float16 gradients from flushing to
    # zero, so `_train_step` scales the loss, and apply_gradients unscales, clips,
    # and skips non-finite steps.  On the fp32/CPU path the scale would be a
    # power-of-two no-op (numerically exact round-trip) that buys nothing but a
    # per-step unscale, so we keep the plain AdamW — its global_clipnorm=10.0
    # still clips and `_train_step` branches to apply the unscaled gradients
    # directly.  Either optimizer exposes `.learning_rate` for the per-step
    # assignment below (proxied through the wrapper when present).
    if _use_mixed_precision:
        optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)

    # EMA decay: derived from effective total_steps.
    # Formula: half-life = total_steps / 10  →  decay = 1 - ln2 / half-life
    # Tiered floor guarantees a meaningful shadow even for very short runs:
    #   total_steps < 100  → floor 0.90  (0.90^30 ≈ 4 %, 96 % learned)
    #   total_steps < 500  → floor 0.95  (0.95^100 ≈ 0.6 %, 99 % learned)
    #   total_steps ≥ 500  → floor 0.99  (standard; 0.99^500 ≈ 0.7 %)
    ema_decay = min(0.9999, 1.0 - _math.log(2) / max(total_steps / 10, 1))
    if total_steps < 100:
        ema_decay = max(ema_decay, 0.90)
    elif total_steps < 500:
        ema_decay = max(ema_decay, 0.95)
    else:
        ema_decay = max(ema_decay, 0.99)

    logger.info(
        f"[{job_id}] Training schedule: "
        f"effective_epochs={effective_epochs}  epochs={_requested_epochs}  "
        f"patience={PATIENCE}  base_lr={train_cfg.base_lr:.2e}  "
        f"cosine_min_lr={train_cfg.min_lr:.2e}  "
        f"ema_decay={ema_decay:.6f}  total_steps={total_steps}"
    )

    from app.ml.yolo_pro.train_config import ModelEMA
    ema = ModelEMA(model, decay=ema_decay)

    optimizer.build(model.trainable_variables)

    # Pre-compute anchor grids and corresponding stride tensors
    anchor_grids = _make_anchor_points(H, W, strides)
    anchor_tensors = [tf.constant(pts, dtype=tf.float32) for pts, _ in anchor_grids]
    stride_tensors = [tf.constant(s, dtype=tf.float32) for _, s in anchor_grids]

    current_balance = dict(balance_policy)
    _balance_adapted = False
    logger.info(
        f"[{job_id}] Dynamic TAL enabled — assignment uses live model predictions "
        f"per batch (no static pre-computation for train targets)."
    )
    # Log all localization-relevant hyperparameters in one place so that
    # per-run comparisons can be read directly from the training log.
    logger.info(
        f"[{job_id}] Localization config: "
        f"WEIGHT_DFL={current_balance['w_dfl']:.2f}  "
        f"WEIGHT_BOX={current_balance['w_box']:.2f}  "
        f"WEIGHT_CLS={current_balance['w_cls']:.2f}  "
        f"WEIGHT_O2O={current_balance['w_o2o']:.2f}  "
        f"TAL_topk=10  TAL_beta=6.0  TAL_alpha=0.5  "
        f"reg_max={reg_max}"
    )

    # ── Build live setup log lines and write to DB before training starts ─────
    _yolo_log_lines: list = []
    # Register the buffer so the outer training_worker terminal handler can flush
    # every accumulated line into training_history when the run ends in
    # failed/cancelled — instead of losing the lines queued since the last
    # incremental commit.  Holds a reference to this same list, so later appends
    # are visible at flush time.  Cleared by the outer handler on every exit.
    _register_live_buffer(job_id, log_lines=_yolo_log_lines)
    # Informational (not a warning): when the adaptive policy lands on a batch
    # size different from the one the user asked for, say so in plain language
    # as the first line of the run log.  Only fires for an explicit request —
    # `batch_size=None` means "auto", so there is nothing to reconcile.
    if job.batch_size is not None and int(job.batch_size) != int(batch_size):
        # Padded with blank lines so the notice reads as its own block instead of
        # blending into the setup lines around it.  The blanks are single spaces,
        # not "" — the log panels render each line as a <div> under
        # `whitespace-pre-wrap`, and an empty <div> collapses to zero height.
        _yolo_log_lines += [
            " ",
            "ℹ Batch size adjusted.",
            f"Requested batch size: {int(job.batch_size)}.",
            f"Effective batch size: {int(batch_size)}"
            f" (automatically adjusted for stable and efficient training).",
            " ",
        ]
        logger.info(
            f"[{job_id}] Batch size adjusted: requested {int(job.batch_size)} "
            f"→ effective {int(batch_size)}."
        )
    try:
        _total_p = model.count_params()
        _train_p = int(sum(int(np.prod(v.shape)) for v in model.trainable_variables))
        _nontr_p = _total_p - _train_p
        _yolo_log_lines += [
            f"Total params: {_total_p:,}",
            f"Trainable params: {_train_p:,}",
            f"Non-trainable params: {_nontr_p:,}",
        ]
    except Exception:
        _total_p = 0
    _yolo_log_lines.append(
        f"Model build — size={size}"
        f"  depths={list(_dbg_cfg.depths)}  channels={list(_dbg_cfg.channels)}"
        f"  total_params={_total_p:,}  prior_prob={float(balance_policy['prior_prob']):.4f}"
    )
    _yolo_log_lines.append(
        f"Stabilization policy — requested: epochs={_requested_epochs} batch={job.batch_size!r}"
        f" | internal: effective_epochs={effective_epochs} batch={batch_size}"
        f" steps_per_epoch={steps_per_epoch} total_steps={total_steps}"
        f" tiny_dataset_mode={policy['tiny_dataset_mode']}"
    )
    _yolo_log_lines.append(
        f"Dataset complexity — images={dataset_stats['num_images']}"
        f"  boxes={dataset_stats['total_boxes']}"
        f"  avg_boxes_per_image={dataset_stats['avg_boxes_per_image']:.3f}"
        f"  classes={dataset_stats['num_classes']}"
        f"  active_classes={dataset_stats['active_class_count']}"
        f"  max_class_fraction={dataset_stats['class_fraction_max']:.3f}"
    )
    _yolo_log_lines.append(
        f"Balance preset — preset={balance_policy['preset']}"
        f"  w_cls={balance_policy['w_cls']:.2f}  w_box={balance_policy['w_box']:.2f}"
        f"  w_dfl={balance_policy['w_dfl']:.2f}  w_o2o={balance_policy['w_o2o']:.2f}"
        f"  prior_prob={balance_policy['prior_prob']:.4f}  reason={balance_policy['reason']}"
    )
    # _yolo_log_lines.append(f"Final batch size: {batch_size}")
    _yolo_log_lines.append(
        f"Training schedule: effective_epochs={effective_epochs}  epochs={_requested_epochs}"
        f"  patience={PATIENCE}  base_lr={train_cfg.base_lr:.2e}"
        f"  cosine_min_lr={train_cfg.min_lr:.2e}  ema_decay={ema_decay:.6f}"
        f"  total_steps={total_steps}"
    )
    _yolo_log_lines.append(
        "Dynamic TAL enabled — assignment uses live model predictions per batch"
        " (no static pre-computation for train targets)."
    )
    _yolo_log_lines.append(
        f"Localization config: WEIGHT_DFL={current_balance['w_dfl']:.2f}"
        f"  WEIGHT_BOX={current_balance['w_box']:.2f}"
        f"  WEIGHT_CLS={current_balance['w_cls']:.2f}"
        f"  WEIGHT_O2O={current_balance['w_o2o']:.2f}"
        f"  TAL_topk=10  TAL_beta=6.0  TAL_alpha=0.5  reg_max={reg_max}"
    )
    # _mosaic_n/_nomosaic_n computed below; append mosaic/aug lines after that.

    # ── Validation data gate ──────────────────────────────────────────────────
    # val_loss is computed each epoch using dynamic TAL on EMA weights.
    # No pre-computation needed — targets are built inline per epoch batch.
    has_val_data = len(X_test) > 0 and any(b for b in boxes_test if b is not None)
    if not has_val_data:
        logger.info(f"[{job_id}] No validation data — val_loss will not be tracked.")

    # ── Training loop ─────────────────────────────────────────────────────────
    loss_history:         List[float] = []
    val_loss_history:     List[float] = []
    val_box_dfl_history:  List[float] = []
    val_cls_history:      List[float] = []
    best_loss         = float("inf")
    # Most recent finite epoch mean_loss — surfaced in NaN-abort diagnostics
    # so operators can see the last healthy loss before the run went non-finite.
    last_finite_mean_loss: float = float("nan")
    best_val_loss     = float("inf")
    best_val_cls      = float("inf")
    # Best-seen VFL positive / negative branch losses — running low-water
    # marks feeding the loss-only cls-calibration adapter (Phase C:
    # _maybe_adapt_cls_calibration compares the current epoch's split
    # against these instead of the retired safety-report triggers).
    best_val_cls_pos  = float("inf")
    best_val_cls_neg  = float("inf")
    best_seen_val_box_dfl = float("inf")
    # Box+DFL localization loss tracked separately for checkpoint selection.
    # mAP@75 correlates with CIoU+DFL tightness more directly than total
    # val_loss (which is dominated by VFL classification on negative cells).
    best_val_box_dfl  = float("inf")
    best_ckpt_metric_name = "train_loss"
    best_ckpt_metric_value = float("inf")
    # mean_val_cls at the epoch whose EMA snapshot is currently saved as the
    # "best" checkpoint.  This is the baseline the ranking guard compares
    # against — NOT best_val_cls (which is a global low-water mark that can be
    # lowered by epochs we never actually save, causing the guard to over-reject
    # later genuine checkpoints).  None until the first checkpoint is saved.
    saved_ckpt_val_cls: Optional[float] = None
    patience_count   = 0
    # One-shot flag so the close-mosaic warmup suppression notice is logged
    # at most once per run (the condition can persist for many epochs).
    _close_mosaic_warmup_suppress_logged: bool = False
    global_step    = 0
    # ── On-device epoch accumulators (Change #1: no per-step host sync) ───────
    # The training step returns loss / finiteness as device tensors.  Reading
    # them back with `.numpy()` / `float(...)` every step forces a device→host
    # sync that serialises the GPU.  Instead we accumulate them into these
    # device-resident tf.Variables each step (pure device ops, no sync) and read
    # them back only every _LOSS_READBACK_EVERY steps and once per epoch.
    _acc_loss          = tf.Variable(0.0, trainable=False, dtype=tf.float32)
    _acc_o2m           = tf.Variable(0.0, trainable=False, dtype=tf.float32)
    _acc_o2o           = tf.Variable(0.0, trainable=False, dtype=tf.float32)
    _acc_finite_count  = tf.Variable(0.0, trainable=False, dtype=tf.float32)
    _acc_skip_count    = tf.Variable(0.0, trainable=False, dtype=tf.float32)
    _acc_loss_nan_count = tf.Variable(0.0, trainable=False, dtype=tf.float32)
    _LOSS_READBACK_EVERY = 20
    # ── Top-K disk-backed checkpoint retention (Phase E) ──────────────────
    # The trainer keeps the K best epochs by val_loss as EMA-applied
    # ``.keras`` files on disk, registered in ``checkpoint_manifest``.  There
    # is NO in-RAM candidate buffer anymore — the disk manifest is the SOLE
    # source of truth for retention, eviction, and the post-training finalist
    # eval (which re-scores exactly these K checkpoints and selects the shipped
    # one).  On eviction the worst-val_loss ``.keras`` is deleted from disk.
    # Admission itself is the loss-only decision (Phase D); the authoritative
    # full mAP / deployment-safety eval that actually picks the export runs
    # once per finalist at end of training.  Retaining K (default 8) finalists
    # keeps the val-loss-best epoch — frequently NOT the mAP-best epoch — from
    # crowding a slightly-worse-loss but safer checkpoint out of the pool.
    # Tunable via ``TrainConfig.topk_checkpoints``.
    _topk_checkpoints = int(train_cfg.topk_checkpoints)
    checkpoint_dir = tempfile.mkdtemp(prefix="yolo_pro_ckpt_")
    checkpoint_manifest: List[dict] = []
    # Rolling on-disk buffer of the most-recent epochs' EMA weights, kept so the
    # final ``_ALWAYS_EVAL_FINAL_EPOCHS`` trained epochs are ALWAYS available to
    # the post-training finalist eval — even if they never improved the monitor
    # or would have been evicted by top-K retention.  Kept separate from
    # ``checkpoint_manifest`` (which top-K pruning mutates in-loop) and holding
    # its own dedicated ``.keras`` files, so eviction can never delete a
    # final-epoch model out from under the finalist eval.  Early stopping can
    # end the run on any epoch, so which epochs are "final" is unknown ahead of
    # time; we therefore roll the last N here and merge them in after the loop.
    _final_tail_ckpts: List[dict] = []
    # Disk-budget guard state: the real per-checkpoint size × K is surfaced in
    # the log once, right after the first successful ``.keras`` write (see the
    # save site below).  Soft budget is generous — nano/96² checkpoints are a
    # few MB each; it only trips for large backbones / high input resolution
    # where ``topk_checkpoints`` should be reconsidered against disk.
    _ckpt_disk_budget_checked: bool = False
    _CKPT_DISK_WARN_MB = 4096.0
    _ckpt_dir_line = (
        f"[{job_id}] disk checkpoint dir (retain top-{_topk_checkpoints} "
        f"by val_loss): {checkpoint_dir}"
    )
    logger.info(_ckpt_dir_line)

    # Phase D: the per-epoch real-inference safety probe, safety gate, and
    # both rescue paths were removed — the loop is now loss-only, so the
    # ``best_safety_map50`` ratchet and its ``best_safety_epoch`` /
    # ``best_safety_reason`` companions no longer exist.  All mAP / rescue /
    # deployment-safety evaluation now lives in the single post-training
    # finalist eval (see "Model Testing stage" below).

    # Audit trail of every monitor-improving candidate the cls guard rejected.
    # Used by the end-of-training summary so an operator can reconstruct
    # *why* no safe checkpoint was exportable without scraping per-epoch logs.
    # Each entry: {epoch, monitor_metric, monitor_value, failed_gates, val_cls,
    # saved_ckpt_val_cls, safety_metrics: {tp_score_mean, fp_score_mean,
    # precision, map50, preds_per_image}}
    _rejected_ckpt_candidates: List[dict] = []

    # Per-epoch classifier-calibration trace.  Each entry captures the
    # diagnostic snapshot needed to identify which calibration signal is
    # degrading without scraping raw log lines: tp/fp/gap/precision/map50/
    # preds_per_image from the safety probe, val_cls split into positive
    # and negative VFL contributions, the live w_cls weight, and whether
    # the cls-calibration adapter fired this epoch.  Persisted to
    # ``training_history["calibration_trace"]`` so the operator UI / log
    # collectors can replay the trajectory.  Diagnostic only — never
    # consulted for checkpoint selection or export gating.
    _calibration_trace: List[dict] = []

    # ── Phase 0 instrumentation (measurement only; no behavior change) ──
    # Per-epoch wall-clock timings for each expensive stage of the
    # validation/checkpoint block, plus the flags needed to correlate the
    # cost with control state (candidate epoch, mosaic phase, monitor
    # improvement).  Persisted into ``training_history["phase0_timings"]``
    # so the baseline pathology (full export-safety eval firing on nearly
    # every post-mosaic epoch) is quantified per job for Phases 1-3.
    # Diagnostic only — never consulted by any gate, threshold, or buffer.
    _phase0_timings: List[dict] = []
    # End-of-training snapshot: the selected/exported epoch, its full
    # export-safety mAP50, and the full ring-buffer contents (epoch + full
    # metrics).  Populated once, at export-restore time.  Diagnostic only.
    _phase0_final_summary: dict = {}

    # Per-epoch training metrics for the Training Graphs. YOLO-Pro is a
    # detection architecture, so only loss is tracked — accuracy fields are
    # omitted (never surface mAP as val_accuracy). Persisted incrementally via
    # _calibration_history_keys() at every training_history write site so a
    # partial run survives a crash.
    _yolo_epoch_metrics: List[dict] = []
    # Register alongside log_lines so a failed/cancelled run keeps every
    # completed-epoch metric accumulated before the exception.
    _register_live_buffer(job_id, epoch_metrics=_yolo_epoch_metrics)

    # Aggregate state for the classifier-calibration adapter.  Counts
    # firings so the end-of-training summary can answer "did the adapter
    # fire often enough to move w_cls?" without re-parsing per-epoch logs.
    _cls_cal_firings: int = 0
    _cls_cal_first_epoch: Optional[int] = None
    _cls_cal_last_epoch:  Optional[int] = None

    def _cls_calibration_summary() -> dict:
        """
        Build the aggregate adapter-firing summary persisted into
        ``training_history["cls_calibration_summary"]``.  Diagnostic only.
        Returns a fresh dict each call so callers can safely mutate it.
        """
        final_w_cls = float(current_balance.get("w_cls", 1.0))
        return {
            "firings":            int(_cls_cal_firings),
            "first_firing_epoch": _cls_cal_first_epoch,
            "last_firing_epoch":  _cls_cal_last_epoch,
            "final_w_cls":        final_w_cls,
            "w_cls_cap":          float(_CLS_CAL_W_CLS_CAP),
            "cap_reached":        final_w_cls >= float(_CLS_CAL_W_CLS_CAP) - 1e-9,
        }

    def _calibration_history_keys() -> dict:
        """
        Two-key dict — ``calibration_trace`` and ``cls_calibration_summary``
        — to merge into every ``training_history`` write site so the
        diagnostic snapshot is never silently clobbered by a later log-only
        write.  Diagnostic only.
        """
        return {
            "calibration_trace":       list(_calibration_trace),
            "cls_calibration_summary": _cls_calibration_summary(),
            "epoch_metrics":           list(_yolo_epoch_metrics),
            # Phase 0 instrumentation (measurement only) — see _phase0_timings.
            "phase0_timings":          list(_phase0_timings),
            "phase0_final_summary":    dict(_phase0_final_summary),
        }

    # Confidence threshold used for the per-candidate checkpoint safety eval.
    # Matches the final post-training eval threshold so the safety probe
    # samples the same low end of the score distribution (the regime where
    # ranking collapse manifests as background outranking foreground).
    _safety_eval_conf = _resolve_size_aware_eval_conf(
        size, _compute_eval_conf_threshold(policy)
    )

    # Runtime/deployment confidence threshold used for the prediction-flood
    # safety gate.  This is the same value stamped into exported TFLite
    # metadata, so the flood gate measures what the deployed model actually
    # produces — not what the permissive AP eval threshold admits.  Hoisted
    # to the training-setup phase so the per-candidate safety eval, the
    # final eval, and the walk-back eval all sample the same runtime regime.
    _runtime_safety_conf = _compute_runtime_conf_threshold(policy, size)
    logger.info(
        f"[{job_id}] Safety thresholds resolved: "
        f"AP eval conf_threshold={_safety_eval_conf:.4f} "
        f"(used for mAP/precision/recall, tp_score_mean/fp_score_mean, "
        f"and the AP-threshold prediction count), "
        f"runtime/flood conf_threshold={_runtime_safety_conf:.4f} "
        f"(used for the preds_per_image safety ceiling — gate fails when "
        f"runtime_preds_per_image > {_CKPT_SAFETY_MAX_PREDS_PER_IMAGE:.2f})."
    )

    _mosaic_n   = train_cfg.mosaic_epochs
    _nomosaic_n = effective_epochs - _mosaic_n
    # Log the actual close-mosaic share after all tiny-dataset adjustments.
    _close_frac_pct = (
        f"{int(round(100.0 * _nomosaic_n / max(effective_epochs, 1)))} % "
        f"({'tiny-dataset' if policy['tiny_dataset_mode'] else 'standard'})"
    )
    # Scheduler bookkeeping — worker log only (debug), not the UI stream.
    logger.debug(
        f"[{job_id}] Mosaic schedule: ON for first {_mosaic_n} epoch(s), "
        f"OFF for final {_nomosaic_n} epoch(s) (close-mosaic refinement).  "
        f"[close_n={_nomosaic_n}  fraction={_close_frac_pct}]"
    )

    # Append augmentation setup line now that _mosaic_n is known (the mosaic
    # schedule itself is scheduler bookkeeping — logged above at debug level,
    # not user-facing).
    _yolo_log_lines.append(
        f"EP1 augmentation: mosaic={'ON' if _mosaic_n > 0 else 'OFF'}"
        f"  hflip=True  scale=True  brightness=True  contrast=True"
    )
    # Write setup lines to DB so the frontend sees them before the first epoch.
    job.training_history = _sanitize_json({"log_lines": list(_yolo_log_lines), "is_yolo_pro": True})
    try:
        db.commit()
    except Exception:
        db.rollback()

    # Check before the training loop starts
    _raise_if_cancelled(job_id)

    # ── Batch prep (runs on a background producer thread) ────────────────────
    # `_build_batch` performs ALL CPU-side prep for one training step:
    #   1. mosaic / h-flip / HSV augmentation (Phase 1),
    #   2. the dynamic-TAL forward pass (Phase 2, stop-gradient), and
    #   3. NumPy o2m + o2o target assignment (Phase 3).
    # It is invoked one step AHEAD of the GPU (see the per-epoch producer in the
    # loop below) so this work overlaps the GPU train step instead of
    # serialising in front of it.  The np.random call order is byte-identical to
    # the previous inline version, so augmentation is unchanged; the only
    # behavioural difference is that the dynamic-TAL forward reads weights that
    # are ~1-2 optimizer steps stale — negligible for TAL's soft assignment /
    # VFL quality guidance and within run-to-run noise.  With tal_refresh_stride
    # k>1 (Phase 4, opt-in) that staleness widens to ~k steps on reuse batches.
    _tal_fail_state = [0]   # bounded log counter for dynamic-TAL forward failures

    # ── Phase 4 (opt-in): dynamic-TAL refresh stride ────────────────────────
    # Run the stop-gradient dynamic-TAL forward only on every k-th batch and
    # reuse the cached predictions on the intervening batches, trading a small
    # amount of assignment staleness for fewer forward passes when the train
    # step is still forward-bound after Phases 1-3.  k=1 (default) preserves the
    # every-batch behavior byte-for-byte (the reuse branch is never taken and
    # the cache is never written).  The reuse window is epoch-local: the refresh
    # gate keys off the in-epoch batch index (batch_start // batch_size), so
    # batch 0 of every epoch always refreshes — the cache is therefore seeded
    # from a full-size batch before any reuse, and reuse never over-indexes.
    # Predictions are grid-structured and the weights move slowly, so reusing a
    # neighbouring batch's predictions only approximates the alignment metric —
    # hence this is gated and carries a small, real accuracy risk.
    _tal_refresh_stride = max(
        1, int((job.extra_params or {}).get("tal_refresh_stride", 1))
    )
    # Cache of the most recent successful forward.  ``None`` means "no valid
    # cache yet" (nothing refreshed, or the last refresh failed) → the reuse
    # path falls back to a fresh forward (or static assignment if that fails).
    _tal_cache: Dict[str, Optional[Dict[str, np.ndarray]]] = {
        "cls": None, "boxes": None,
    }
    if _tal_refresh_stride > 1:
        logger.info(
            f"[{job_id}] Dynamic-TAL refresh stride = {_tal_refresh_stride} "
            f"(opt-in): forward runs on every {_tal_refresh_stride}-th batch; "
            f"intervening batches reuse cached predictions (small accuracy "
            f"risk — sweep k and verify mAP50 before shipping k>1)."
        )

    # ── Per-epoch val-loss forward de-dup (default ON) ───────────────────────
    # The val-loss pass already forwards each val batch once (to obtain the
    # predictions it decodes for NumPy target assignment); the loss step then
    # re-forwarded the identical batch a second time.  With this on, the loss is
    # computed from the SAME predictions (`_val_step_from_preds`) — halving the
    # val forward count with numerically identical results (same EMA weights,
    # `training=False`, no update between the two forwards).  Set
    # extra_params["dedup_val_forward"]=false to fall back to the old two-forward
    # path (byte-identical behavior, just slower).
    _dedup_val_forward = bool(
        (job.extra_params or {}).get("dedup_val_forward", True)
    )

    # ── Parallel TAL target assignment (opt-in, default 0 = serial) ──────────
    # Phase 3 of `_build_batch` scores + resolves TAL targets independently per
    # sample and uses NO RNG and NO TF ops — only NumPy over the anchor grid
    # (P3 alone = 6400 anchors × per-GT scoring).  It is therefore safe to fan
    # the per-sample work across a thread pool: `executor.map` preserves input
    # order, so the stacked target arrays are BYTE-IDENTICAL to the serial loop,
    # and the heavy NumPy releases the GIL enough to overlap across cores.
    # Augmentation (Phase 1) is intentionally NOT parallelized here — it draws
    # from the global `np.random` stream and calls tf.image ops, so threading it
    # would be non-deterministic and unsafe.  `_assign_pool` is created per epoch
    # in the producer setup and read by `_build_batch` through this closure var.
    _assign_workers = max(
        0, int((job.extra_params or {}).get("assign_workers", 0))
    )
    _assign_pool = None
    if _assign_workers > 0:
        logger.info(
            f"[{job_id}] Parallel TAL assignment ENABLED — "
            f"assign_workers={_assign_workers} (Phase 3 only; targets are "
            f"byte-identical to the serial path — order preserved, no RNG)."
        )

    def _build_batch(batch_start, batch_indices,
                     mosaic_epoch_gate, epoch_hflip_prob, epoch_color_jitter_prob):
        # Phase 1 — augmentation (identical np.random sequence to the old loop).
        aug_imgs_list: List[np.ndarray] = []
        aug_boxes_list: List[List[dict]] = []
        for i in batch_indices:
            if mosaic_epoch_gate and np.random.random() < train_cfg.mosaic_prob:
                extra = np.random.choice(len(X_train), 3, replace=True)
                four_imgs  = [X_train[i]] + [X_train[j] for j in extra]
                four_boxes = [boxes_train[i] or []] + [boxes_train[j] or [] for j in extra]
                img, m_boxes = _make_mosaic(four_imgs, four_boxes, H, W)
                if np.random.random() < epoch_hflip_prob:
                    img     = np.flip(img, axis=1).copy()
                    m_boxes = _flip_boxes_horizontal([m_boxes])[0] or []
                if np.random.random() < epoch_color_jitter_prob:
                    img = _hsv_jitter(img)
                aug_imgs_list.append(img)
                aug_boxes_list.append(m_boxes)
            else:
                img   = X_train[i]
                boxes = list(boxes_train[i] or [])
                if np.random.random() < epoch_hflip_prob:
                    img   = np.flip(img, axis=1).copy()
                    boxes = _flip_boxes_horizontal([boxes])[0] or []
                if np.random.random() < epoch_color_jitter_prob:
                    img = _hsv_jitter(img)
                aug_imgs_list.append(img)
                aug_boxes_list.append(boxes)

        imgs_np = np.stack(aug_imgs_list).astype(np.float32)
        _B = len(aug_imgs_list)

        # Phase 2 — dynamic-TAL forward (stop-gradient; informs assignment only).
        # Runs here on the producer thread so its GPU forward + host copy overlap
        # the main thread's train step instead of doubling serial latency.  With
        # tal_refresh_stride k>1 (Phase 4) the forward runs only on every k-th
        # batch; the intervening batches reuse the cached predictions from the
        # most recent successful forward.
        _dyn_cls:   Dict[str, np.ndarray] = {}
        _dyn_boxes: Dict[str, np.ndarray] = {}
        _use_dynamic_tal = False
        _batch_idx    = batch_start // batch_size
        _cached_cls   = _tal_cache["cls"]
        _cached_boxes = _tal_cache["boxes"]
        # Reuse only when the stride says "skip this batch", a valid cache
        # exists, and it holds at least this batch's sample count.  Batch 0
        # always refreshes (0 % k == 0), so within an epoch the cache is seeded
        # full-size before any reuse and the shape guard is belt-and-suspenders.
        _reuse_tal = (
            _tal_refresh_stride > 1
            and _batch_idx % _tal_refresh_stride != 0
            and _cached_cls is not None
            and _cached_boxes is not None
            and next(iter(_cached_cls.values())).shape[0] >= _B
        )
        if _reuse_tal:
            _dyn_cls   = _cached_cls
            _dyn_boxes = _cached_boxes
            _use_dynamic_tal = True
        else:
            try:
                # Graph-compiled forward (see `_tal_infer` at the model-build site) —
                # replaces the eager `model(imgs_np, training=False)` call. imgs_np is
                # already float32 (stacked above), so this constant wrap is a no-op cast.
                _raw = _tal_infer(tf.constant(imgs_np, dtype=tf.float32))
                for _sname, (_apt, _s) in zip(["p3", "p4", "p5"], anchor_grids):
                    _cr = _raw[f"cls_{_sname}"]
                    _rr = _raw[f"reg_{_sname}"]
                    _cn = (
                        _cr.numpy() if hasattr(_cr, "numpy") else np.array(_cr)
                    ).reshape(_B, -1, n_classes)
                    _rn = (
                        _rr.numpy() if hasattr(_rr, "numpy") else np.array(_rr)
                    ).reshape(_B, -1, 4 * reg_max)
                    _dyn_cls[_sname]   = _cn
                    _dyn_boxes[_sname] = _decode_reg_np(
                        _rn, _apt, float(_s), H, W, reg_max
                    )
                _use_dynamic_tal = True
                # Seed the reuse cache with this fresh forward (Phase 4).  Skipped
                # on the k=1 default so that path retains no extra references.
                if _tal_refresh_stride > 1:
                    _tal_cache["cls"]   = _dyn_cls
                    _tal_cache["boxes"] = _dyn_boxes
            except Exception as _tal_exc:
                if _tal_fail_state[0] < 3:
                    _tal_fail_state[0] += 1
                    logger.warning(
                        f"[{job_id}] Dynamic TAL forward pass failed "
                        f"(batch_start {batch_start}): {_tal_exc} — using static fallback."
                    )

        # Phase 3 — TAL target assignment per sample (o2m topk=10, o2o topk=1).
        # Score once per (sample, scale) and resolve both paths from that single
        # scoring pass (bottlenecks #3/#4) — see _assign_boxes_o2m_o2o.  o2m parts
        # and o2o parts are collected separately then concatenated so the 24-tuple
        # layout is unchanged: o2m ×3 scales at idx 0-11, o2o ×3 at 12-23.
        def _assign_one_sample(_si: int) -> tuple:
            # Per-sample assignment across all three scales → the 24-slot tuple
            # (o2m ×3 at idx 0-11, o2o ×3 at 12-23).  Reads only per-sample
            # slices (`_dyn_*[_sname][_si]`, `aug_boxes_list[_si]`) and shared
            # read-only arrays (anchors), so calls for distinct `_si` never
            # touch shared mutable state — safe to run concurrently.
            _o2m_parts: List = []
            _o2o_parts: List = []
            for _sname, (_apt, _s) in zip(["p3", "p4", "p5"], anchor_grids):
                _pkw: dict = {}
                if _use_dynamic_tal:
                    _pkw["pred_cls_np"]   = _dyn_cls[_sname][_si]
                    _pkw["pred_boxes_np"] = _dyn_boxes[_sname][_si]
                (ct, bxt, lt, fg), (ct2, bxt2, lt2, fg2) = _assign_boxes_o2m_o2o(
                    boxes=aug_boxes_list[_si],
                    label_map=label_map,
                    anchor_pts=_apt,
                    stride=_s,
                    H_img=H, W_img=W,
                    n_classes=n_classes,
                    reg_max=reg_max,
                    **_pkw,
                )
                _o2m_parts.extend([ct, bxt, lt, fg])
                _o2o_parts.extend([ct2, bxt2, lt2, fg2])
            return tuple(_o2m_parts + _o2o_parts)

        # `executor.map` yields results in input order, so `bt` — and therefore
        # every stacked target array below — is byte-identical to the serial
        # comprehension regardless of `_assign_workers`.
        if _assign_pool is not None:
            bt: List[tuple] = list(_assign_pool.map(_assign_one_sample, range(_B)))
        else:
            bt = [_assign_one_sample(_si) for _si in range(_B)]

        # Batch the per-sample target tuples into 24 stacked arrays on THIS
        # (producer) thread so the 24× np.stack is off the consumer's critical
        # path (bottleneck #5).  The 24-slot layout is unchanged — o2m ×3 scales
        # at tuple idx 0-11, o2o ×3 at 12-23 — and the dict keys mirror the
        # tf.constant names the consumer builds one-to-one.
        _target_keys = (
            "cls_p3", "box_p3", "ltrb_p3", "fg_p3",
            "cls_p4", "box_p4", "ltrb_p4", "fg_p4",
            "cls_p5", "box_p5", "ltrb_p5", "fg_p5",
            "o2o_cls_p3", "o2o_box_p3", "o2o_ltrb_p3", "o2o_fg_p3",
            "o2o_cls_p4", "o2o_box_p4", "o2o_ltrb_p4", "o2o_fg_p4",
            "o2o_cls_p5", "o2o_box_p5", "o2o_ltrb_p5", "o2o_fg_p5",
        )
        targets: Dict[str, np.ndarray] = {
            _k: np.stack([t[_i] for t in bt])
            for _i, _k in enumerate(_target_keys)
        }

        return batch_start, imgs_np, targets, _use_dynamic_tal

    def _consume_batch(batch_start, imgs_np, targets, _use_dynamic_tal):
        # Runs on the MAIN thread, one batch at a time, so all train-step /
        # EMA / accumulator ordering is identical to the synchronous loop.
        nonlocal global_step, _epoch_step, _skip_logged, _last_step_finite

        # ── Assignment diagnostics (first batch of each epoch) ────────────────
        # Reads the pre-stacked target arrays (bottleneck #5): summing the whole
        # batched fg array equals the old per-sample sum, and masking the batched
        # cls array by the batched fg mask yields the same set of per-positive
        # max scores the old per-sample loop collected (order differs, mean does
        # not).
        if batch_start == 0:
            _pos_p3 = int(targets["fg_p3"].sum())
            _pos_p4 = int(targets["fg_p4"].sum())
            _pos_p5 = int(targets["fg_p5"].sum())
            _o2o_p3 = int(targets["o2o_fg_p3"].sum())
            _o2o_p4 = int(targets["o2o_fg_p4"].sum())
            _o2o_p5 = int(targets["o2o_fg_p5"].sum())
            _all_vfl: List[float] = []
            for _ck, _fk in (("cls_p3", "fg_p3"),
                             ("cls_p4", "fg_p4"),
                             ("cls_p5", "fg_p5")):
                _cls_s = targets[_ck]
                _fg_s  = targets[_fk]
                if _fg_s.any():
                    _all_vfl.extend(_cls_s[_fg_s].max(axis=-1).tolist())
            _avg_vfl = float(np.mean(_all_vfl)) if _all_vfl else 0.0
            _avg_vfl_tag = (
                " [no positives — model likely NaN-poisoned upstream]"
                if not _all_vfl and not _last_step_finite
                else ""
            )
            logger.info(
                f"[{job_id}] EP{epoch+1} TAL | "
                f"o2m — p3={_pos_p3} p4={_pos_p4} p5={_pos_p5} | "
                f"o2o — p3={_o2o_p3} p4={_o2o_p4} p5={_o2o_p5} | "
                f"avg_vfl={_avg_vfl:.4f}{_avg_vfl_tag} | "
                f"{'dynamic' if _use_dynamic_tal else 'STATIC FALLBACK'} | "
                f"topk=10  beta=6.0"
            )

        imgs = tf.constant(imgs_np, dtype=tf.float32)
        # Targets are already batched on the producer thread (bottleneck #5); the
        # consumer only wraps each ready array in a tf.constant with the right
        # dtype.  Keys mirror the tuple layout: o2m ×3 at 0-11, o2o ×3 at 12-23.
        cls_p3  = tf.constant(targets["cls_p3"],  dtype=tf.float32)
        box_p3  = tf.constant(targets["box_p3"],  dtype=tf.float32)
        ltrb_p3 = tf.constant(targets["ltrb_p3"], dtype=tf.float32)
        fg_p3   = tf.constant(targets["fg_p3"],   dtype=tf.bool)
        cls_p4  = tf.constant(targets["cls_p4"],  dtype=tf.float32)
        box_p4  = tf.constant(targets["box_p4"],  dtype=tf.float32)
        ltrb_p4 = tf.constant(targets["ltrb_p4"], dtype=tf.float32)
        fg_p4   = tf.constant(targets["fg_p4"],   dtype=tf.bool)
        cls_p5  = tf.constant(targets["cls_p5"],  dtype=tf.float32)
        box_p5  = tf.constant(targets["box_p5"],  dtype=tf.float32)
        ltrb_p5 = tf.constant(targets["ltrb_p5"], dtype=tf.float32)
        fg_p5   = tf.constant(targets["fg_p5"],   dtype=tf.bool)
        # o2o targets (indices 12-23)
        o2o_cls_p3  = tf.constant(targets["o2o_cls_p3"],  dtype=tf.float32)
        o2o_box_p3  = tf.constant(targets["o2o_box_p3"],  dtype=tf.float32)
        o2o_ltrb_p3 = tf.constant(targets["o2o_ltrb_p3"], dtype=tf.float32)
        o2o_fg_p3   = tf.constant(targets["o2o_fg_p3"],   dtype=tf.bool)
        o2o_cls_p4  = tf.constant(targets["o2o_cls_p4"],  dtype=tf.float32)
        o2o_box_p4  = tf.constant(targets["o2o_box_p4"],  dtype=tf.float32)
        o2o_ltrb_p4 = tf.constant(targets["o2o_ltrb_p4"], dtype=tf.float32)
        o2o_fg_p4   = tf.constant(targets["o2o_fg_p4"],   dtype=tf.bool)
        o2o_cls_p5  = tf.constant(targets["o2o_cls_p5"],  dtype=tf.float32)
        o2o_box_p5  = tf.constant(targets["o2o_box_p5"],  dtype=tf.float32)
        o2o_ltrb_p5 = tf.constant(targets["o2o_ltrb_p5"], dtype=tf.float32)
        o2o_fg_p5   = tf.constant(targets["o2o_fg_p5"],   dtype=tf.bool)

        optimizer.learning_rate.assign(lr_schedule_fn(global_step))

        _step_total, _step_o2m, _step_o2o, _step_finite_t = _train_step(
            model, optimizer, imgs,
            tf.constant(current_balance["w_cls"], dtype=tf.float32),
            tf.constant(current_balance["w_box"], dtype=tf.float32),
            tf.constant(current_balance["w_dfl"], dtype=tf.float32),
            tf.constant(current_balance["w_o2o"], dtype=tf.float32),
            cls_p3, box_p3, ltrb_p3, fg_p3,
            cls_p4, box_p4, ltrb_p4, fg_p4,
            cls_p5, box_p5, ltrb_p5, fg_p5,
            o2o_cls_p3, o2o_box_p3, o2o_ltrb_p3, o2o_fg_p3,
            o2o_cls_p4, o2o_box_p4, o2o_ltrb_p4, o2o_fg_p4,
            o2o_cls_p5, o2o_box_p5, o2o_ltrb_p5, o2o_fg_p5,
            anchor_tensors[0], anchor_tensors[1], anchor_tensors[2],
            stride_tensors[0], stride_tensors[1], stride_tensors[2],
            H, W, reg_max,
        )
        # ── On-device accumulation (Change #1) — no host sync per step ────────
        _finite_f   = tf.cast(_step_finite_t, tf.float32)
        _loss_nan_f = 1.0 - tf.cast(tf.math.is_finite(_step_total), tf.float32)
        _acc_loss.assign_add(_step_total * _finite_f)
        _acc_o2m.assign_add(_step_o2m * _finite_f)
        _acc_o2o.assign_add(_step_o2o * _finite_f)
        _acc_finite_count.assign_add(_finite_f)
        _acc_skip_count.assign_add(1.0 - _finite_f)
        _acc_loss_nan_count.assign_add(_loss_nan_f)
        # EMA update: fused graph op, suppressed on-device when non-finite.
        ema.update(model, _step_finite_t)

        global_step += 1
        _epoch_step += 1

        # Read the device counters back only every N=20 steps for logging /
        # the non-finite abort check — not every iteration.
        if _epoch_step % _LOSS_READBACK_EVERY == 0:
            _skipped  = int(_acc_skip_count.numpy())
            _loss_nan = int(_acc_loss_nan_count.numpy())
            _last_step_finite = (_skipped == _skip_logged)
            if _skipped > _skip_logged:
                _skip_logged = _skipped
                _nan_msg = (
                    f"EP{epoch+1} step {_epoch_step}: {_skipped} non-finite "
                    f"step(s) so far this epoch — optimizer step(s) skipped, "
                    f"EMA update suppressed "
                    f"(genuine non-finite-loss steps={_loss_nan})"
                )
                logger.warning(f"[{job_id}] {_nan_msg}")
                _yolo_log_lines.append(_nan_msg)
                job.training_history = _sanitize_json({
                    "log_lines": list(_yolo_log_lines), "is_yolo_pro": True,
                    **_calibration_history_keys(),
                })
                try:
                    db.commit()
                except Exception:
                    db.rollback()
            if _loss_nan >= _MAX_NAN_STEPS_PER_EPOCH:
                raise RuntimeError(
                    f"[{job_id}] Aborting: {_loss_nan} "
                    f"non-finite-loss training steps in EP{epoch+1} "
                    f"(threshold={_MAX_NAN_STEPS_PER_EPOCH}). "
                    f"Last finite mean_loss before NaN run: "
                    f"{last_finite_mean_loss}. Numerical instability in "
                    f"loss/gradient path — not a checkpoint-selection "
                    f"issue. Investigate regression head and CIoU "
                    f"gradient path before retrying. Note: BN moving "
                    f"stats may be partially poisoned from the non-finite "
                    f"forward passes; this is a known bounded limitation."
                )

    for epoch in range(effective_epochs):
        # Check at the start of each epoch for prompt cancellation response
        _raise_if_cancelled(job_id)

        # ── Phase 0 timers (measurement only) ────────────────────────────
        # Default to 0.0 so the per-epoch record always carries a value even
        # for stages that are skipped this epoch (probe / full eval / snapshot
        # are conditional).  Overwritten in place by each stage below.
        _ph0_train_s:     float = 0.0
        _ph0_val_loss_s:  float = 0.0
        _ph0_probe_s:     float = 0.0
        _ph0_full_eval_s: float = 0.0
        _ph0_snapshot_s:  float = 0.0

        # BN-moment finiteness probe: if a prior step poisoned BN moving
        # stats despite the per-step gate (BN updates happen inside the
        # training=True forward, before gradients are visible), abort here
        # instead of running an entire epoch on a NaN-poisoned model.
        # Perf: reduce every BN moving-stat variable to a scalar finiteness flag
        # ON-DEVICE, stack them, and read back with a SINGLE `.numpy()` host sync
        # per epoch — the old form paid one D2H stall per BN variable every epoch.
        # The abort condition and message (offending variable name) are unchanged;
        # the per-variable scan only runs on the cold NaN-poisoned path.
        _bn_vars = [
            _v for _v in model.variables
            if "moving_mean" in _v.name or "moving_variance" in _v.name
        ]
        if _bn_vars:
            _bn_finite = tf.stack([
                tf.reduce_all(tf.math.is_finite(_v)) for _v in _bn_vars
            ])
            if not bool(tf.reduce_all(_bn_finite).numpy()):
                _bad_name = next(
                    _v.name for _v, _ok in zip(_bn_vars, _bn_finite.numpy())
                    if not bool(_ok)
                )
                raise RuntimeError(
                    f"[{job_id}] EP{epoch+1} start: BN variable "
                    f"'{_bad_name}' has non-finite moving stats — model is "
                    f"NaN-poisoned. Aborting."
                )

        # Reset the on-device epoch accumulators (Change #1).
        _acc_loss.assign(0.0)
        _acc_o2m.assign(0.0)
        _acc_o2o.assign(0.0)
        _acc_finite_count.assign(0.0)
        _acc_skip_count.assign(0.0)
        _acc_loss_nan_count.assign(0.0)
        # Host-side bookkeeping for the periodic readback / logging.
        _epoch_step: int = 0            # step index within this epoch
        _skip_logged: int = 0           # skipped-step count already logged
        nan_step_count_this_epoch: int = 0  # genuine non-finite-loss steps (abort signal)
        # Tracks the most recent readback's finiteness so the avg_vfl logger can
        # disambiguate a genuine empty-positive batch from a NaN-poisoned one.
        _last_step_finite: bool = True
        indices = np.random.permutation(len(X_train))
        # Epoch gate: mosaic is allowed only during the first mosaic_epochs epochs.
        # Close-mosaic phase (final epochs) always uses clean single images so the
        # model can stabilise on unaugmented distributions before export.
        mosaic_epoch_gate = epoch < train_cfg.mosaic_epochs
        if mosaic_epoch_gate:
            epoch_hflip_prob = train_cfg.hflip_prob
            epoch_color_jitter_prob = train_cfg.color_jitter_prob
        else:
            epoch_hflip_prob, epoch_color_jitter_prob = _refinement_aug_probs(
                policy, train_cfg
            )
            # tiny-only: halve color_jitter during refinement tail (nano: no change)
            epoch_hflip_prob, epoch_color_jitter_prob = _resolve_size_aware_refinement_aug(
                size, epoch_hflip_prob, epoch_color_jitter_prob
            )

        # Log augmentation state at phase boundaries so it is always clear
        # which augmentations are active without inspecting the full loop.
        if epoch == 0:
            logger.info(
                f"[{job_id}] EP1  Augmentation: "
                f"mosaic_epoch_gate=ON  mosaic_p={train_cfg.mosaic_prob}  "
                f"hflip_p={epoch_hflip_prob}  "
                f"color_jitter_p={epoch_color_jitter_prob} (HSV)  "
                f"[validation/eval: no augmentation]"
            )
        elif epoch == train_cfg.mosaic_epochs:
            # Mosaic ON → OFF transition.  Always reset early-stop patience:
            # patience accumulated under mosaic is not representative of the
            # clean-image distribution, and mAP often improves once mosaic is
            # disabled.  Without this reset, a run whose patience was already
            # exhausted under mosaic would exit at the first close-mosaic
            # epoch and never receive any refinement window.
            _cm_n_transition = max(0, effective_epochs - train_cfg.mosaic_epochs)
            _cm_min_transition = _close_mosaic_min_epochs(
                effective_epochs, _cm_n_transition
            )
            patience_count = 0
            _cm_reset_line = (
                f"[{job_id}] close-mosaic patience reset "
                f"(min_refinement_epochs={_cm_min_transition}, "
                f"grace_epochs={_close_mosaic_grace_epochs(policy)})."
            )
            # Scheduler internals — worker log only (debug), not the UI stream.
            logger.debug(_cm_reset_line)
            logger.info(
                f"[{job_id}] EP{epoch+1}  Augmentation: "
                f"mosaic_epoch_gate=OFF (close-mosaic phase)  "
                f"hflip_p={epoch_hflip_prob}  "
                f"color_jitter_p={epoch_color_jitter_prob} (HSV)"
            )

        # ── Double-buffered batch producer ───────────────────────────────────
        # Build the next batch (augmentation + dynamic-TAL forward + NumPy target
        # assignment, all in `_build_batch`) on a background thread while the GPU
        # trains the current batch, so CPU-side prep no longer starves the
        # device.  maxsize=2 → ~1-2 batches of look-ahead: bounded memory and a
        # bounded (~1-2 optimizer step) staleness of the dynamic-TAL weights.
        # `_consume_batch` (train step + EMA + accumulators) still runs serially
        # on THIS thread, so all optimizer/EMA ordering is byte-for-byte the same.
        _ph0_train_t0 = time.perf_counter()
        _prep_q: "queue.Queue" = queue.Queue(maxsize=2)
        _stop_prep = threading.Event()
        _epoch_indices = indices

        # Per-epoch assignment thread pool (opt-in).  Created here and torn down
        # in the finally below so no worker threads leak across epochs / jobs.
        # `_build_batch` (running on the producer thread) reads this via closure.
        if _assign_workers > 0:
            import concurrent.futures as _cf
            _assign_pool = _cf.ThreadPoolExecutor(
                max_workers=_assign_workers,
                thread_name_prefix=f"yolo-assign-ep{epoch+1}",
            )

        def _prep_producer():
            try:
                for _bs in range(0, len(X_train), batch_size):
                    if _stop_prep.is_set():
                        return
                    _prepared = _build_batch(
                        _bs, _epoch_indices[_bs:_bs + batch_size],
                        mosaic_epoch_gate, epoch_hflip_prob, epoch_color_jitter_prob,
                    )
                    # Bounded put so a bailed-out consumer can never deadlock us.
                    while not _stop_prep.is_set():
                        try:
                            _prep_q.put(_prepared, timeout=0.5)
                            break
                        except queue.Full:
                            continue
            except BaseException as _prep_exc:          # surface to the consumer
                _prep_q.put(("__prep_error__", _prep_exc))
            finally:
                _prep_q.put(("__epoch_done__", None))

        _prep_thread = threading.Thread(
            target=_prep_producer, name=f"yolo-prep-ep{epoch+1}", daemon=True,
        )
        _prep_thread.start()
        try:
            while True:
                _prepared = _prep_q.get()
                # Control markers are 2-tuples; a real prepared batch is a 4-tuple.
                if isinstance(_prepared, tuple) and len(_prepared) == 2:
                    if _prepared[0] == "__epoch_done__":
                        break
                    if _prepared[0] == "__prep_error__":
                        raise _prepared[1]
                _raise_if_cancelled(job_id)
                _batch_start, _imgs_np, _targets, _use_dyn = _prepared
                _consume_batch(_batch_start, _imgs_np, _targets, _use_dyn)
        finally:
            # Stop the producer and drain the queue so a producer parked on a
            # full put() unblocks, then wait for the thread to exit.
            _stop_prep.set()
            try:
                while True:
                    _prep_q.get_nowait()
            except queue.Empty:
                pass
            _prep_thread.join(timeout=30.0)
            # Tear down the per-epoch assignment pool AFTER the producer (its only
            # user) has stopped, so there is no submit-after-shutdown race.
            if _assign_pool is not None:
                _assign_pool.shutdown(wait=True)
                _assign_pool = None
        _ph0_train_s = time.perf_counter() - _ph0_train_t0

        # ── Epoch-end readback (Change #1): one host sync for the epoch means ──
        # sum / finite_count reproduces np.mean over the finite steps exactly.
        _finite_count   = int(_acc_finite_count.numpy())
        _skipped_final  = int(_acc_skip_count.numpy())
        nan_step_count_this_epoch = int(_acc_loss_nan_count.numpy())
        if _skipped_final > _skip_logged:
            _skip_logged = _skipped_final
            _nan_msg = (
                f"EP{epoch+1}: {_skipped_final} non-finite step(s) this epoch "
                f"— optimizer step(s) skipped, EMA suppressed "
                f"(genuine non-finite-loss steps={nan_step_count_this_epoch})"
            )
            logger.warning(f"[{job_id}] {_nan_msg}")
            _yolo_log_lines.append(_nan_msg)
        if _finite_count == 0:
            # Defense-in-depth: the non-finite-loss threshold should fire first.
            raise RuntimeError(
                f"[{job_id}] EP{epoch+1}: every step skipped as non-finite "
                f"— no valid loss to report."
            )
        mean_loss = float(_acc_loss.numpy()) / _finite_count
        mean_o2m  = float(_acc_o2m.numpy()) / _finite_count
        mean_o2o  = float(_acc_o2o.numpy()) / _finite_count
        if math.isfinite(mean_loss):
            last_finite_mean_loss = mean_loss
        loss_history.append(mean_loss)

        # ── Per-epoch validation pass (on EMA weights) ───────────────────────
        # IMPORTANT: apply EMA weights before running the val pass so that the
        # metric we monitor — and checkpoint on — measures the same model that
        # will be exported.  Without this, val_loss is computed on live weights
        # while the exported model uses EMA weights; the "best" epoch identified
        # from live-weight val_loss may be a different (worse) EMA checkpoint.
        mean_val_loss:     Optional[float] = None
        mean_val_box_dfl:  Optional[float] = None
        mean_val_cls:      Optional[float] = None
        mean_val_cls_pos:  Optional[float] = None
        mean_val_cls_neg:  Optional[float] = None
        _ph0_val_t0 = time.perf_counter()
        if has_val_data:
            try:
                _live_backup = ema.backup_weights(model)
                ema.apply(model)
                val_batch_losses   = []
                val_batch_box_dfl  = []
                val_batch_cls      = []
                # VFL positive / negative branch diagnostics.  Surfaced per
                # epoch as `mean_val_cls_pos` / `mean_val_cls_neg` so the
                # FP-drift band can be distinguished from positive-branch
                # stagnation without scraping per-batch logs.
                val_batch_cls_pos  = []
                val_batch_cls_neg  = []
                try:
                    for vb_start in range(0, len(X_test), batch_size):
                        _raise_if_cancelled(job_id)
                        vb_idx = list(range(vb_start, min(vb_start + batch_size, len(X_test))))
                        v_imgs = tf.constant(X_test[vb_idx], dtype=tf.float32)
                        _vraw = model(v_imgs, training=False)
                        _vB = len(vb_idx)
                        _vdyn_cls: Dict[str, np.ndarray] = {}
                        _vdyn_boxes: Dict[str, np.ndarray] = {}
                        for _vsname, (_vapt, _vs) in zip(["p3", "p4", "p5"], anchor_grids):
                            _vcr = _vraw[f"cls_{_vsname}"]
                            _vrr = _vraw[f"reg_{_vsname}"]
                            _vcn = (_vcr.numpy() if hasattr(_vcr, "numpy") else np.array(_vcr)).reshape(_vB, -1, n_classes)
                            _vrn = (_vrr.numpy() if hasattr(_vrr, "numpy") else np.array(_vrr)).reshape(_vB, -1, 4 * reg_max)
                            _vdyn_cls[_vsname] = _vcn
                            _vdyn_boxes[_vsname] = _decode_reg_np(_vrn, _vapt, float(_vs), H, W, reg_max)
                        vbt = []
                        for _vsi, _vidx in enumerate(vb_idx):
                            # Score once per (sample, scale); resolve o2m + o2o
                            # from the shared scoring (same dedup as the train
                            # loop).  Layout preserved: o2m idx 0-11, o2o 12-23.
                            _vo2m_parts = []
                            _vo2o_parts = []
                            for _vsname, (_vapt, _vs) in zip(["p3", "p4", "p5"], anchor_grids):
                                (ct, bt, lt, fg), (ct2, bt2, lt2, fg2) = _assign_boxes_o2m_o2o(
                                    boxes=boxes_test[_vidx] or [], label_map=label_map, anchor_pts=_vapt, stride=_vs,
                                    H_img=H, W_img=W, n_classes=n_classes, reg_max=reg_max,
                                    pred_cls_np=_vdyn_cls[_vsname][_vsi], pred_boxes_np=_vdyn_boxes[_vsname][_vsi],
                                )
                                _vo2m_parts.extend([ct, bt, lt, fg])
                                _vo2o_parts.extend([ct2, bt2, lt2, fg2])
                            vbt.append(tuple(_vo2m_parts + _vo2o_parts))
                        # o2m targets (indices 0-11)
                        v_cls_p3  = tf.constant(np.stack([t[0]  for t in vbt]), dtype=tf.float32)
                        v_box_p3  = tf.constant(np.stack([t[1]  for t in vbt]), dtype=tf.float32)
                        v_ltrb_p3 = tf.constant(np.stack([t[2]  for t in vbt]), dtype=tf.float32)
                        v_fg_p3   = tf.constant(np.stack([t[3]  for t in vbt]), dtype=tf.bool)
                        v_cls_p4  = tf.constant(np.stack([t[4]  for t in vbt]), dtype=tf.float32)
                        v_box_p4  = tf.constant(np.stack([t[5]  for t in vbt]), dtype=tf.float32)
                        v_ltrb_p4 = tf.constant(np.stack([t[6]  for t in vbt]), dtype=tf.float32)
                        v_fg_p4   = tf.constant(np.stack([t[7]  for t in vbt]), dtype=tf.bool)
                        v_cls_p5  = tf.constant(np.stack([t[8]  for t in vbt]), dtype=tf.float32)
                        v_box_p5  = tf.constant(np.stack([t[9]  for t in vbt]), dtype=tf.float32)
                        v_ltrb_p5 = tf.constant(np.stack([t[10] for t in vbt]), dtype=tf.float32)
                        v_fg_p5   = tf.constant(np.stack([t[11] for t in vbt]), dtype=tf.bool)
                        # o2o targets (indices 12-23)
                        v_o2o_cls_p3  = tf.constant(np.stack([t[12] for t in vbt]), dtype=tf.float32)
                        v_o2o_box_p3  = tf.constant(np.stack([t[13] for t in vbt]), dtype=tf.float32)
                        v_o2o_ltrb_p3 = tf.constant(np.stack([t[14] for t in vbt]), dtype=tf.float32)
                        v_o2o_fg_p3   = tf.constant(np.stack([t[15] for t in vbt]), dtype=tf.bool)
                        v_o2o_cls_p4  = tf.constant(np.stack([t[16] for t in vbt]), dtype=tf.float32)
                        v_o2o_box_p4  = tf.constant(np.stack([t[17] for t in vbt]), dtype=tf.float32)
                        v_o2o_ltrb_p4 = tf.constant(np.stack([t[18] for t in vbt]), dtype=tf.float32)
                        v_o2o_fg_p4   = tf.constant(np.stack([t[19] for t in vbt]), dtype=tf.bool)
                        v_o2o_cls_p5  = tf.constant(np.stack([t[20] for t in vbt]), dtype=tf.float32)
                        v_o2o_box_p5  = tf.constant(np.stack([t[21] for t in vbt]), dtype=tf.float32)
                        v_o2o_ltrb_p5 = tf.constant(np.stack([t[22] for t in vbt]), dtype=tf.float32)
                        v_o2o_fg_p5   = tf.constant(np.stack([t[23] for t in vbt]), dtype=tf.bool)
                        _v_wcls = tf.constant(current_balance["w_cls"], dtype=tf.float32)
                        _v_wbox = tf.constant(current_balance["w_box"], dtype=tf.float32)
                        _v_wdfl = tf.constant(current_balance["w_dfl"], dtype=tf.float32)
                        _v_wo2o = tf.constant(current_balance["w_o2o"], dtype=tf.float32)
                        if _dedup_val_forward:
                            # Reuse the `_vraw` forward already taken above for the
                            # NumPy assignment — no second forward over this batch.
                            v_loss, v_box_dfl, v_cls, v_cls_pos, v_cls_neg = _val_step_from_preds(
                                _vraw["cls_p3"], _vraw["reg_p3"],
                                _vraw["cls_p4"], _vraw["reg_p4"],
                                _vraw["cls_p5"], _vraw["reg_p5"],
                                _v_wcls, _v_wbox, _v_wdfl, _v_wo2o,
                                v_cls_p3, v_box_p3, v_ltrb_p3, v_fg_p3,
                                v_cls_p4, v_box_p4, v_ltrb_p4, v_fg_p4,
                                v_cls_p5, v_box_p5, v_ltrb_p5, v_fg_p5,
                                v_o2o_cls_p3, v_o2o_box_p3, v_o2o_ltrb_p3, v_o2o_fg_p3,
                                v_o2o_cls_p4, v_o2o_box_p4, v_o2o_ltrb_p4, v_o2o_fg_p4,
                                v_o2o_cls_p5, v_o2o_box_p5, v_o2o_ltrb_p5, v_o2o_fg_p5,
                                anchor_tensors[0], anchor_tensors[1], anchor_tensors[2],
                                stride_tensors[0], stride_tensors[1], stride_tensors[2],
                                H, W, reg_max,
                            )
                        else:
                            v_loss, v_box_dfl, v_cls, v_cls_pos, v_cls_neg = _val_step(
                                model, v_imgs,
                                _v_wcls, _v_wbox, _v_wdfl, _v_wo2o,
                                v_cls_p3, v_box_p3, v_ltrb_p3, v_fg_p3,
                                v_cls_p4, v_box_p4, v_ltrb_p4, v_fg_p4,
                                v_cls_p5, v_box_p5, v_ltrb_p5, v_fg_p5,
                                v_o2o_cls_p3, v_o2o_box_p3, v_o2o_ltrb_p3, v_o2o_fg_p3,
                                v_o2o_cls_p4, v_o2o_box_p4, v_o2o_ltrb_p4, v_o2o_fg_p4,
                                v_o2o_cls_p5, v_o2o_box_p5, v_o2o_ltrb_p5, v_o2o_fg_p5,
                                anchor_tensors[0], anchor_tensors[1], anchor_tensors[2],
                                stride_tensors[0], stride_tensors[1], stride_tensors[2],
                                H, W, reg_max,
                            )
                        v_loss_f     = float(v_loss)
                        v_box_dfl_f  = float(v_box_dfl)
                        v_cls_f      = float(v_cls)
                        v_cls_pos_f  = float(v_cls_pos)
                        v_cls_neg_f  = float(v_cls_neg)
                        if math.isfinite(v_loss_f):
                            val_batch_losses.append(v_loss_f)
                        if math.isfinite(v_box_dfl_f):
                            val_batch_box_dfl.append(v_box_dfl_f)
                        if math.isfinite(v_cls_f):
                            val_batch_cls.append(v_cls_f)
                        if math.isfinite(v_cls_pos_f):
                            val_batch_cls_pos.append(v_cls_pos_f)
                        if math.isfinite(v_cls_neg_f):
                            val_batch_cls_neg.append(v_cls_neg_f)
                finally:
                    # Always restore live weights so training continues normally.
                    ema.restore(model, _live_backup)
                mean_val_loss    = float(np.mean(val_batch_losses))   if val_batch_losses   else None
                mean_val_box_dfl = float(np.mean(val_batch_box_dfl))  if val_batch_box_dfl  else None
                mean_val_cls     = float(np.mean(val_batch_cls))       if val_batch_cls      else None
                mean_val_cls_pos = float(np.mean(val_batch_cls_pos))   if val_batch_cls_pos  else None
                mean_val_cls_neg = float(np.mean(val_batch_cls_neg))   if val_batch_cls_neg  else None
                if mean_val_loss is not None:
                    val_loss_history.append(mean_val_loss)
                if mean_val_box_dfl is not None:
                    val_box_dfl_history.append(mean_val_box_dfl)
                    if mean_val_box_dfl < best_seen_val_box_dfl:
                        best_seen_val_box_dfl = mean_val_box_dfl
                if mean_val_cls is not None:
                    val_cls_history.append(mean_val_cls)
                    if mean_val_cls < best_val_cls:
                        best_val_cls = mean_val_cls
                if mean_val_cls_pos is not None and mean_val_cls_pos < best_val_cls_pos:
                    best_val_cls_pos = mean_val_cls_pos
                if mean_val_cls_neg is not None and mean_val_cls_neg < best_val_cls_neg:
                    best_val_cls_neg = mean_val_cls_neg
            except _CancelledError:
                raise
            except Exception as _vl_exc:
                logger.warning(
                    f"[{job_id}] val_loss computation failed at epoch {epoch+1}: {_vl_exc}"
                )
                mean_val_loss    = None
                mean_val_box_dfl = None
                mean_val_cls     = None
                mean_val_cls_pos = None
                mean_val_cls_neg = None
        _ph0_val_loss_s = time.perf_counter() - _ph0_val_t0

        current_balance, _balance_adapted, _balance_adapt_reason = _maybe_adapt_training_balance(
            balance_policy=current_balance,
            epoch_index=epoch,
            warmup_epochs=policy["warmup_epochs"],
            mean_val_box_dfl=mean_val_box_dfl,
            mean_val_cls=mean_val_cls,
            best_val_box_dfl=best_seen_val_box_dfl,
            best_val_cls=best_val_cls,
            adapted_already=_balance_adapted,
        )
        if _balance_adapt_reason:
            logger.info(f"[{job_id}] Balance adaptation â€” {_balance_adapt_reason}")

        _o2o_weighted = current_balance["w_o2o"] * mean_o2o
        _loss_fmt = (
            f"o2m={mean_o2m:.4f}  "
            f"o2o_raw={mean_o2o:.4f}  o2o_w={_o2o_weighted:.4f}  "
            f"loss={mean_loss:.4f}"
        )

        # ── Early stopping / checkpointing ─────────────────────────────────────
        # Pure val_box_dfl is an unreliable checkpoint signal on tiny datasets:
        # when the model collapses to predicting fewer classes, TAL assigns
        # fewer positives for the abandoned class → those cells fall out of
        # fg_mask → the normalized box_dfl is LOWER even though recall
        # collapsed.  Checkpointing on pure box_dfl therefore rewards
        # class-collapse silently (observed: dog class dropped to 0 % recall).
        #
        # val_loss (total) includes the VFL classification term which sums over
        # ALL anchor cells before normalising.  When a class is abandoned the
        # VFL for its GT cells stays high → val_loss rises → checkpoint NOT
        # saved → we keep the epoch where all classes were still active.
        #
        # Fallback chain: val_loss → train_loss.
        # NOTE: val_box_dfl is excluded as the primary monitor for tiny_dataset_mode
        # because it is normalized over TAL positives — when the model partially
        # collapses, fewer positives are assigned → box_dfl drops even as recall
        # collapses (confirmed in production). val_loss (total VFL) stays high for
        # abandoned GT cells, so it is a reliable checkpoint signal even on tiny sets.
        ckpt_metric_name = _select_checkpoint_metric_name(
            policy, mean_val_loss, mean_val_box_dfl
        )

        if ckpt_metric_name == "val_box_dfl":
            monitor_loss = (
                mean_val_box_dfl if mean_val_box_dfl is not None else mean_loss
            )
            _best_ref = (
                best_val_box_dfl if mean_val_box_dfl is not None else best_loss
            )
        elif ckpt_metric_name == "val_loss":
            monitor_loss = (
                mean_val_loss if mean_val_loss is not None else mean_loss
            )
            _best_ref = (
                best_val_loss if mean_val_loss is not None else best_loss
            )
        else:
            monitor_loss = mean_loss
            _best_ref = best_loss

        _ckpt_monitor_improved = monitor_loss < _best_ref - 1e-4
        # Guard baseline = val_cls of the currently-saved checkpoint, NOT the
        # global best_val_cls low-water mark.  This prevents an unsaved epoch
        # (one whose monitor metric did not improve) from ratcheting the cls
        # floor downward and over-rejecting future genuine improvements.
        _ckpt_cls_guard_passed = _checkpoint_passes_ranking_guard(
            ckpt_metric_name, mean_val_cls, saved_ckpt_val_cls
        )

        # ── Classifier-calibration adaptation (loss-only) ─────────────────
        # Phase D: the per-epoch real-inference safety probe has been
        # removed — the loop is now loss-only.  Classifier calibration is
        # driven entirely by the val-loss VFL branch split
        # (mean_val_cls_pos / mean_val_cls_neg vs. their running bests),
        # which the val-loss pass already computes for free.  Runs on every
        # post-warmup val epoch, independent of checkpoint-candidate status,
        # so a drifting cls head is corrected even when the monitor plateaus
        # or regresses or the cls guard would reject the candidate.  The
        # adapter's internal warmup / missing-signal guards make it a no-op
        # when any of the four loss-side values is unavailable or the
        # branches show healthy separation.  All mAP / NMS / IoU / runtime /
        # deployment-safety evaluation is deferred to the single
        # post-training finalist eval (see "Model Testing stage" below).
        _past_warmup = epoch >= max(0, policy["warmup_epochs"])
        if _past_warmup and has_val_data:
            current_balance, _cls_cal_adapted, _cls_cal_reason = (
                _maybe_adapt_cls_calibration(
                    balance_policy=current_balance,
                    safety_report=None,
                    epoch=epoch,
                    warmup_epochs=policy["warmup_epochs"],
                    prior_prob=float(
                        current_balance.get("prior_prob") or 0.01
                    ),
                    val_cls_pos=mean_val_cls_pos,
                    val_cls_neg=mean_val_cls_neg,
                    best_val_cls_pos=best_val_cls_pos,
                    best_val_cls_neg=best_val_cls_neg,
                )
            )
            if _cls_cal_reason:
                # WARNING-level so the event surfaces in operator dashboards
                # the same way checkpoint rejections do.  The exact phrase
                # "classifier calibration adaptation" must appear in the
                # log line so it can be grep'd by deployment tooling.
                logger.warning(f"[{job_id}] {_cls_cal_reason}")
                _yolo_log_lines.append(_cls_cal_reason)
                # Aggregate counter for the end-of-training calibration
                # summary — separate from per-epoch trace so it answers
                # "how many times did the adapter actually move w_cls?"
                # at a glance.
                _cls_cal_firings += 1
                if _cls_cal_first_epoch is None:
                    _cls_cal_first_epoch = epoch + 1
                _cls_cal_last_epoch = epoch + 1

        # ── Checkpoint candidate decision (loss-only) ─────────────────────
        # Phase D: save on monitor improvement, gated only by the loss-side
        # cls ranking guard (``_ckpt_cls_guard_passed``, computed above from
        # val_cls vs. the saved checkpoint's val_cls).  The real-inference
        # safety gate and both rescue paths were removed; all deployment-
        # safety evaluation is deferred to the single post-training finalist
        # eval, which re-scores every buffered checkpoint on the full
        # validation set and refuses to export a collapsed one.
        _save_ckpt = _ckpt_monitor_improved and _ckpt_cls_guard_passed
        _ckpt_guard_rejected = _ckpt_monitor_improved and not _ckpt_cls_guard_passed
        _checkpoint_reason: Optional[str] = "monitor" if _save_ckpt else None

        # ── Admission snapshot: no longer needed (Phase E) ────────────────
        # The retained candidate is now the on-disk ``.keras`` written below
        # by ``_save_ema_checkpoint_keras``, which applies EMA over the current
        # (epoch-N) BN stats before saving — capturing exactly the (EMA-N
        # trainables, epoch-N BN) admission model.  The separate in-RAM BN
        # snapshot that the removed in-RAM candidate buffer relied on is gone,
        # so ``_ph0_snapshot_s`` stays 0 (the per-epoch snapshot cost is gone).

        # ── Calibration trace (diagnostic only) ─────────────────────────
        # Emit a per-epoch record so the calibration trajectory can be
        # replayed without scraping log lines.  Every field is either a
        # finite number or None; the record is written for every epoch
        # (including warmup) so the operator can see the bootstrap phase.
        # This trace is not consulted by any gate or ranking metric.
        #
        # Phase D: the real-inference probe is gone, so the detector-side
        # tp/fp/gap/precision/map50/preds_per_image signals are no longer
        # available in-loop.  The record keeps its field names (populated as
        # None) so downstream readers / tests keyed on its shape don't break;
        # the loss-side val_cls split is now the live calibration signal.
        _tr_tp = None
        _tr_fp = None
        _tr_gap = None
        _calibration_trace.append({
            "epoch":           epoch + 1,
            "tp_score_mean":   None,
            "fp_score_mean":   None,
            "tp_fp_gap":       None,
            "precision":       None,
            "map50":           None,
            "preds_per_image": None,
            "val_cls":         mean_val_cls,
            "val_cls_pos":     mean_val_cls_pos,
            "val_cls_neg":     mean_val_cls_neg,
            "w_cls":           float(current_balance.get("w_cls", 1.0)),
            "adapter_fired":   bool(_cls_cal_reason) if _past_warmup and has_val_data else False,
        })

        if _ckpt_guard_rejected:
            # Monitor improved but the loss-side cls ranking guard refused the
            # save: val_cls regressed vs. the saved checkpoint's val_cls by
            # more than the tolerance (a post-mosaic / cls-recovery
            # regression).  Patience is held below so we don't early-stop
            # before classification recovers.  Deployment-safety collapse is
            # now caught by the post-training finalist eval, not in-loop.
            logger.warning(
                f"[{job_id}] EP{epoch+1} checkpoint rejected — "
                f"{ckpt_metric_name} improved to {monitor_loss:.4f} but val_cls "
                f"({mean_val_cls:.4f}) exceeds saved-checkpoint baseline "
                f"({saved_ckpt_val_cls:.4f}) by "
                f">{int(_CKPT_RANKING_GUARD_TOLERANCE * 100)}% — likely "
                f"ranking collapse.  Patience held to allow recovery."
            )
            # Audit the rejection so the end-of-training summary can name the
            # cls-guard rejection without scraping per-epoch logs.  Loss-only:
            # no real-inference safety_metrics are available in-loop anymore.
            _rejected_ckpt_candidates.append({
                "epoch":            epoch + 1,
                "monitor_metric":   ckpt_metric_name,
                "monitor_value":    float(monitor_loss) if math.isfinite(monitor_loss) else None,
                "failed_gates":     [
                    f"cls_guard: val_cls={mean_val_cls} > "
                    f"saved_ckpt_val_cls={saved_ckpt_val_cls} "
                    f"by >{int(_CKPT_RANKING_GUARD_TOLERANCE * 100)}%"
                ],
                "val_cls":            mean_val_cls,
                "saved_ckpt_val_cls": saved_ckpt_val_cls,
                "safety_metrics":     None,
            })
        # Phase D: admission is the loss-only decision above — save iff the
        # monitor improved and the cls ranking guard passed.  Phase E: the
        # admitted EMA snapshot is written straight to disk and registered in
        # ``checkpoint_manifest`` (the SOLE candidate store — no in-RAM buffer).
        # The authoritative full export-safety eval runs ONCE at end of
        # training, re-scores every retained checkpoint, and refuses to export
        # a collapsed one; retained entries carry only loss-side metrics, and
        # their ``full_*`` fields stay null (backfilled by that eval).
        if _save_ckpt:
            # Primary monitor bookkeeping — advance the monitor's low-water
            # mark for the (only remaining) monitor save path.
            if ckpt_metric_name == "val_box_dfl" and mean_val_box_dfl is not None:
                best_val_box_dfl = monitor_loss
            elif ckpt_metric_name == "val_loss" and mean_val_loss is not None:
                best_val_loss = monitor_loss
            else:
                best_loss = monitor_loss
            best_ckpt_metric_name = ckpt_metric_name
            best_ckpt_metric_value = monitor_loss

            # ── Persist the admitted EMA snapshot to disk (Phase E) ──────
            # Write the EMA-applied model to a ``.keras`` file and register it
            # with every field the post-training finalist eval / ranking /
            # Phase-0 summary read.  Filename encodes the epoch and val_loss
            # (or the monitor metric's value when val_loss is None), matching
            # the Edge-Impulse ``e{epoch}-val_loss_{value}.keras`` scheme.
            # Best-effort: a write failure is logged and swallowed — that epoch
            # is simply not retained; the save decision, patience, and metrics
            # are untouched.
            try:
                if mean_val_loss is not None and math.isfinite(float(mean_val_loss)):
                    _ckpt_stub = f"e{epoch+1:04d}-val_loss_{float(mean_val_loss):.2f}"
                elif math.isfinite(monitor_loss):
                    _ckpt_stub = (
                        f"e{epoch+1:04d}-{ckpt_metric_name}_{float(monitor_loss):.2f}"
                    )
                else:
                    _ckpt_stub = f"e{epoch+1:04d}-{ckpt_metric_name}"
                _ckpt_path = os.path.join(checkpoint_dir, f"{_ckpt_stub}.keras")
                _save_ema_checkpoint_keras(model, ema, _ckpt_path)
                checkpoint_manifest.append({
                    "epoch":            epoch + 1,
                    "path":             _ckpt_path,
                    "val_loss":         mean_val_loss,
                    "val_box_dfl":      mean_val_box_dfl,
                    "val_cls":          mean_val_cls,
                    "ckpt_metric_name": ckpt_metric_name,
                    "monitor_metric":   ckpt_metric_name,
                    "monitor_value":    (
                        float(monitor_loss) if math.isfinite(monitor_loss) else None
                    ),
                    "checkpoint_reason": _checkpoint_reason or "monitor",
                    # Mosaic-phase tag retained for the Phase-0 finalist summary.
                    "mosaic_active":    bool(epoch < train_cfg.mosaic_epochs),
                    # Canonical full export-safety metrics — DEFERRED to the
                    # single post-training finalist eval, which backfills them
                    # and is the ranking source of truth.  Present-but-null so
                    # downstream readers / tests keep the field names.
                    "full_safety_map50":            None,
                    "full_safety_precision":        None,
                    "full_safety_tp_fp_gap":        None,
                    "full_runtime_preds_per_image": None,
                    "full_evaluated_images":        None,
                    "ranking_status":               None,
                })
                # ── Disk-budget guard: measured once, after the first write ──
                if not _ckpt_disk_budget_checked:
                    _ckpt_disk_budget_checked = True
                    try:
                        _ckpt_bytes = int(os.path.getsize(_ckpt_path))
                    except Exception:
                        _ckpt_bytes = int(_total_p) * 4
                    _ckpt_entry_mb = _ckpt_bytes / (1024 * 1024)
                    _ckpt_peak_mb  = _ckpt_entry_mb * _topk_checkpoints
                    # The per-checkpoint disk-budget line is backend-only noise,
                    # so nothing is emitted here — only the soft-budget warning
                    # below is worth the user's attention.
                    if _ckpt_peak_mb > _CKPT_DISK_WARN_MB:
                        _ckpt_budget_warn = (
                            f"[{job_id}] WARNING: retained checkpoints ~{_ckpt_peak_mb:.0f} MB "
                            f"exceed {_CKPT_DISK_WARN_MB:.0f} MB soft disk budget "
                            f"(top-{_topk_checkpoints}, ~{_ckpt_entry_mb:.1f} MB each).  "
                            f"Consider lowering TrainConfig.topk_checkpoints for this model size."
                        )
                        logger.warning(_ckpt_budget_warn)
                        _yolo_log_lines.append(_ckpt_budget_warn)
            except Exception as _ckpt_disk_exc:
                _ckpt_disk_line = (
                    f"[{job_id}] EP{epoch+1} disk checkpoint write failed: "
                    f"{_ckpt_disk_exc} — epoch not retained."
                )
                logger.warning(_ckpt_disk_line)
                _yolo_log_lines.append(_ckpt_disk_line)
            # ── Top-K-by-val_loss retention: evict the worst over budget ──
            # Runs outside the write try/except so a bookkeeping hiccup is
            # never misreported as a write failure.  No-op when at/under K.
            for _evicted in _prune_checkpoints_to_topk(
                checkpoint_manifest, _topk_checkpoints, job_id=job_id
            ):
                logger.info(
                    f"[{job_id}] EP{epoch+1} retention: evicted EP{_evicted.get('epoch')} "
                    f"(val_loss={_evicted.get('val_loss')}) — keeping top-{_topk_checkpoints}"
                )
            # Save-time log line.  Phase D: the loop is loss-only, so the
            # admission record carries only the monitor value — all detector
            # (map50 / precision / gap / runtime-ppi) metrics are produced by
            # the single post-training finalist eval, not here.
            _save_monitor_str = (
                f"{float(monitor_loss):.4f}" if math.isfinite(monitor_loss) else "nan"
            )
            _save_val_loss_str = (
                "None" if mean_val_loss is None else f"{float(mean_val_loss):.4f}"
            )
            _save_summary_line = (
                f"[{job_id}] checkpoint admitted EP{epoch+1} "
                f"reason={_checkpoint_reason or 'monitor'}  "
                f"monitor={ckpt_metric_name}={_save_monitor_str}  "
                f"val_loss={_save_val_loss_str}  "
                f"full(deferred to end-of-training finalist eval)"
            )
            logger.info(_save_summary_line)
            _yolo_log_lines.append(_save_summary_line)
            # Record the val_cls associated with this saved checkpoint so the
            # ranking guard compares future epochs against what's on disk.
            saved_ckpt_val_cls = mean_val_cls
            patience_count = 0
        elif _ckpt_guard_rejected:
            # Monitor improved, but the guard refused the save.  This is the
            # post-mosaic / cls-recovery window — do NOT burn patience, or we
            # may early-stop before classification recovers.  Holding patience
            # is bounded: genuinely-non-improving epochs still increment it,
            # and the overall epoch budget always applies.
            pass
        else:
            patience_count += 1
        # Keep best_loss in sync with train loss for DB/export regardless
        if mean_loss < best_loss:
            best_loss = mean_loss

        # ── Always-evaluate final-epoch tail (rolling) ────────────────────────
        # Persist this epoch's EMA weights to a dedicated rolling ``.keras`` so
        # the final ``_ALWAYS_EVAL_FINAL_EPOCHS`` epochs are ALWAYS scored by the
        # end-of-training finalist eval — independent of the monitor-improvement
        # + top-K path.  We cannot know which epoch is last (early stop can fire
        # on any epoch), so we roll the last N on disk and let the finalist stage
        # merge them in.  Dedicated files (never shared with checkpoint_manifest)
        # keep top-K eviction from deleting a tail model out from under the eval.
        # Requires a validation pass — the finalist eval scores on val data — and
        # is best-effort: a write failure is logged and skipped, leaving that
        # epoch simply not force-evaluated.
        if has_val_data and _ALWAYS_EVAL_FINAL_EPOCHS > 0:
            try:
                _tail_path = os.path.join(
                    checkpoint_dir, f"final-e{epoch+1:04d}.keras"
                )
                _save_ema_checkpoint_keras(model, ema, _tail_path)
                _final_tail_ckpts.append({
                    "epoch":            epoch + 1,
                    "path":             _tail_path,
                    "val_loss":         mean_val_loss,
                    "val_box_dfl":      mean_val_box_dfl,
                    "val_cls":          mean_val_cls,
                    "ckpt_metric_name": ckpt_metric_name,
                    "monitor_metric":   ckpt_metric_name,
                    "monitor_value":    (
                        float(monitor_loss) if math.isfinite(monitor_loss) else None
                    ),
                    "checkpoint_reason": "final_tail",
                    "always_eval":       True,
                    # Final epochs are past the close-mosaic transition; carry the
                    # tag so the finalist scope logic treats them as mosaic-off.
                    "mosaic_active":     bool(epoch < train_cfg.mosaic_epochs),
                    "full_safety_map50":            None,
                    "full_safety_precision":        None,
                    "full_safety_tp_fp_gap":        None,
                    "full_runtime_preds_per_image": None,
                    "full_evaluated_images":        None,
                    "ranking_status":               None,
                })
                # Retain only the most-recent N; delete older tail files.
                while len(_final_tail_ckpts) > _ALWAYS_EVAL_FINAL_EPOCHS:
                    _stale = _final_tail_ckpts.pop(0)
                    _stale_path = _stale.get("path")
                    if _stale_path and os.path.exists(_stale_path):
                        try:
                            os.remove(_stale_path)
                        except Exception as _rm_exc:
                            logger.debug(
                                f"[{job_id}] stale final-tail removal failed "
                                f"({_stale_path}): {_rm_exc}"
                            )
            except Exception as _tail_exc:
                logger.warning(
                    f"[{job_id}] EP{epoch+1} final-tail checkpoint write failed: "
                    f"{_tail_exc} — epoch not force-evaluated."
                )

        # ── Per-epoch summary log ─────────────────────────────────────────────
        # Logged AFTER the checkpoint decision so _ckpt_star is accurate.
        # Phase D: the loop is loss-only, so there is a single save path —
        # '★ckpt' marks a monitor-improving, cls-guard-passing save.
        if _save_ckpt:
            _ckpt_star = " ★ckpt"
        else:
            _ckpt_star = ""
        _ep_hdr = (
            f"ep {epoch+1}/{effective_epochs} (user_epochs={_requested_epochs})"
            if effective_epochs != _requested_epochs
            else f"ep {epoch+1}/{effective_epochs}"
        )
        # Optional VFL-component / safety-trajectory tags appended to the
        # per-epoch INFO line so operators can identify which calibration
        # signal is degrading without opening a separate trace dashboard.
        _val_cls_pos_str = (
            f"  val_cls_pos={mean_val_cls_pos:.4f}"
            if mean_val_cls_pos is not None else ""
        )
        _val_cls_neg_str = (
            f"  val_cls_neg={mean_val_cls_neg:.4f}"
            if mean_val_cls_neg is not None else ""
        )
        _gap_str = (
            f"  tp/fp/gap={_tr_tp:.4f}/{_tr_fp:.4f}/{_tr_gap:.4f}"
            if (_tr_tp is not None and _tr_fp is not None and _tr_gap is not None
                and isinstance(_tr_tp, (int, float)) and isinstance(_tr_fp, (int, float))
                and math.isfinite(float(_tr_tp)) and math.isfinite(float(_tr_fp)))
            else ""
        )
        if mean_val_loss is not None:
            _box_dfl_str = (
                f"  val_box_dfl={mean_val_box_dfl:.4f}"
                if mean_val_box_dfl is not None else ""
            )
            _val_cls_str = (
                f"  val_cls={mean_val_cls:.4f}"
                if mean_val_cls is not None else ""
            )
            logger.info(
                f"[{job_id}] YOLO-Pro {_ep_hdr}  spe={steps_per_epoch}  "
                f"{_loss_fmt}  val_loss={mean_val_loss:.4f}{_box_dfl_str}{_val_cls_str}"
                f"{_val_cls_pos_str}{_val_cls_neg_str}{_gap_str}"
                f"  ckpt_criterion={ckpt_metric_name}{_ckpt_star}"
            )
        else:
            logger.info(
                f"[{job_id}] YOLO-Pro {_ep_hdr}  spe={steps_per_epoch}  "
                f"{_loss_fmt}  val_loss=N/A{_gap_str}  "
                f"ckpt_criterion={ckpt_metric_name}{_ckpt_star}"
            )

        # ── Phase 0 per-epoch timing record (measurement only) ───────────
        # Phase D: the per-epoch probe and full export-safety eval are gone,
        # so ``probe_s`` and ``full_eval_s`` are now always 0 — this is the
        # timing win the migration was after.  ``is_candidate`` is now simply
        # "this epoch was saved" (monitor improved + cls guard passed); it is
        # exactly the condition under which the admission BN-snapshot ran
        # (i.e. snapshot_s > 0 iff this is True, barring an instant raise).
        _ph0_is_candidate = bool(_save_ckpt and has_val_data)
        _ph0_mosaic_active = bool(epoch < train_cfg.mosaic_epochs)
        _ph0_monitor_improved = bool(_ckpt_monitor_improved)
        _ph0_timing_line = (
            f"[{job_id}] PHASE0-TIMING EP{epoch+1} "
            f"train_s={_ph0_train_s:.3f} val_loss_s={_ph0_val_loss_s:.3f} "
            f"probe_s={_ph0_probe_s:.3f} full_eval_s={_ph0_full_eval_s:.3f} "
            f"snapshot_s={_ph0_snapshot_s:.3f} "
            f"is_candidate={_ph0_is_candidate} "
            f"mosaic_active={_ph0_mosaic_active} "
            f"monitor_improved={_ph0_monitor_improved}"
        )
        # Backend diagnostic only — not surfaced in the UI terminal
        # (kept in logs + persisted to training_history["phase0_timings"]).
        logger.info(_ph0_timing_line)
        _phase0_timings.append({
            "epoch":            epoch + 1,
            "train_s":          round(float(_ph0_train_s), 4),
            "val_loss_s":       round(float(_ph0_val_loss_s), 4),
            "probe_s":          round(float(_ph0_probe_s), 4),
            "full_eval_s":      round(float(_ph0_full_eval_s), 4),
            "snapshot_s":       round(float(_ph0_snapshot_s), 4),
            "is_candidate":     _ph0_is_candidate,
            "mosaic_active":    _ph0_mosaic_active,
            "monitor_improved": _ph0_monitor_improved,
        })

        # Append live epoch line and commit to DB so the frontend can display it.
        _ep_live_line = f"Epoch {epoch + 1}/{effective_epochs} - loss: {mean_loss:.4f}"
        if mean_val_loss is not None:
            _ep_live_line += f" - val_loss: {mean_val_loss:.4f}"
        _yolo_log_lines.append(_ep_live_line)
        # Capture per-epoch loss for the Training Graphs (detection: loss only).
        # val_loss is required by the epoch_metrics contract, so skip epochs that
        # ran without a validation pass rather than emitting a null val_loss.
        if mean_loss is not None and mean_val_loss is not None:
            _yolo_epoch_metrics.append({
                "epoch":      epoch + 1,
                "train_loss": float(mean_loss),
                "val_loss":   float(mean_val_loss),
            })
        job.training_history = _sanitize_json({
            "log_lines": list(_yolo_log_lines),
            "is_yolo_pro": True,
            # Diagnostic-only: per-epoch calibration trace.  Consumers
            # must not treat any field as a ranking metric.
            **_calibration_history_keys(),
        })
        try:
            db.commit()
        except Exception:
            db.rollback()

        _close_grace = _close_mosaic_grace_epochs(policy)
        _past_close_mosaic_grace = (
            epoch >= (train_cfg.mosaic_epochs + _close_grace)
        )
        # Close-mosaic warmup window: when mosaic was enabled, suppress early
        # stop for the first ``close_mosaic_min_epochs`` clean-image epochs so
        # the model can refine on the un-augmented distribution.  Monitor
        # checkpoint saves continue normally — only the early-stop exit is
        # gated.
        _close_mosaic_n_es = max(0, effective_epochs - train_cfg.mosaic_epochs)
        _close_mosaic_min_es = _close_mosaic_min_epochs(
            effective_epochs, _close_mosaic_n_es
        )
        _in_close_mosaic_warmup = (
            train_cfg.mosaic_epochs > 0
            and epoch < (train_cfg.mosaic_epochs + _close_mosaic_min_es)
        )
        if (
            patience_count >= PATIENCE
            and _in_close_mosaic_warmup
            and not _close_mosaic_warmup_suppress_logged
        ):
            _cm_suppress_line = (
                f"[{job_id}] early stop suppressed during close-mosaic warmup "
                f"(epoch {epoch+1}, warmup ends after epoch "
                f"{train_cfg.mosaic_epochs + _close_mosaic_min_es})."
            )
            logger.info(_cm_suppress_line)
            _yolo_log_lines.append(_cm_suppress_line)
            _close_mosaic_warmup_suppress_logged = True
        if (
            patience_count >= PATIENCE
            and _early_stop_allowed(policy, epoch, train_cfg.mosaic_epochs)
            and _past_close_mosaic_grace
            and not _in_close_mosaic_warmup
        ):
            logger.info(
                f"[{job_id}] Early stop at epoch {epoch+1} — "
                f"no improvement for {PATIENCE} epochs."
            )
            _yolo_log_lines.append(
                f"Early stop at epoch {epoch+1} - no improvement for {PATIENCE} epochs."
            )
            job.training_history = _sanitize_json({
                "log_lines": list(_yolo_log_lines),
                "is_yolo_pro": True,
                **_calibration_history_keys(),
            })
            try:
                db.commit()
            except Exception:
                db.rollback()
            break
    actual_epochs = len(loss_history)
    stopped_early = actual_epochs < effective_epochs
    stop_reason   = (
        f"Stopped early at epoch {actual_epochs}/{effective_epochs} — "
        f"no improvement for {PATIENCE} consecutive epochs."
        if stopped_early
        else f"Completed all {actual_epochs} requested training epoch(s)."
    )

    # ── Model Testing stage — end-of-training authoritative full eval ────────
    # Phase 2: the per-epoch full export-safety eval was removed in Phase 1,
    # so buffer finalists carry only cheap-probe metrics and null ``full_*``
    # fields.  Here we run the canonical full-validation safety eval ONCE per
    # finalist, backfill each entry's ``full_*`` metrics + ``ranking_status``,
    # and select the shipped checkpoint by full-validation mAP50 gated on
    # ranking health.  This promotes the old "restore-best-then-walk-back-on-
    # collapse" path into a first-class stage that tests EVERY finalist.
    # Bounded cost: ONE full eval in the healthy case (the best post-close-mosaic
    # checkpoint), at most len(checkpoint_manifest) if every candidate collapses
    # — never per-epoch.
    _raise_if_cancelled(job_id)
    # Reuse the hoisted safety-eval threshold so the finalist eval and the
    # per-candidate admission probe sample the same conf regime.
    _eval_conf = _safety_eval_conf
    _eval_mode = (
        "tiny-dataset"
        if policy["tiny_dataset_mode"]
        else ("low-step-budget" if policy.get("low_step_budget") else "standard")
    )

    if not checkpoint_manifest:
        # No monitor-improving candidate was ever retained on disk (or every
        # ``.keras`` write failed).  Exporting the live EMA weights here would
        # ship a model whose classifier health was never validated by real
        # inference — exactly the silent failure the safety gates were added to
        # prevent.  Fail loudly with a structured summary instead.
        _summary = _summarize_rejected_candidates(_rejected_ckpt_candidates)
        _err_msg = _format_no_safe_ckpt_message(_summary, cause="empty_buffer")
        logger.error(f"[{job_id}] {_err_msg}")
        _yolo_log_lines.append(_err_msg)
        job.training_history = _sanitize_json({
            "log_lines": list(_yolo_log_lines),
            "is_yolo_pro": True,
            "no_safe_checkpoint_summary": _summary,
            **_calibration_history_keys(),
        })
        try:
            db.commit()
        except Exception:
            db.rollback()
        raise RuntimeError(_err_msg)

    # ── Merge always-evaluate final-epoch tail into the finalist set ──────────
    # The final ``_ALWAYS_EVAL_FINAL_EPOCHS`` epochs are scored ALONGSIDE the
    # retained monitor checkpoints so the most-refined weights can win export on
    # full-validation accuracy even if they never improved the monitor.  An
    # epoch already retained by the top-K path is flagged ``always_eval`` in
    # place (no duplicate eval) and its now-redundant tail file removed; a tail
    # epoch not in the manifest is appended so the finalist loop loads and scores
    # it.  Only reached when the manifest is non-empty (the empty-manifest
    # collapse above keeps its ``_NO_SAFE_CKPT`` semantics unchanged).
    _manifest_epochs = {
        _e.get("epoch") for _e in checkpoint_manifest if _e.get("epoch") is not None
    }
    for _tail in _final_tail_ckpts:
        _tep = _tail.get("epoch")
        if _tep in _manifest_epochs:
            for _e in checkpoint_manifest:
                if _e.get("epoch") == _tep:
                    _e["always_eval"] = True
                    break
            _tpath = _tail.get("path")
            if _tpath and os.path.exists(_tpath):
                try:
                    os.remove(_tpath)
                except Exception:
                    pass
        else:
            checkpoint_manifest.append(_tail)
            _manifest_epochs.add(_tep)
            _tail_line = (
                f"[{job_id}] Model Testing: final-epoch EP{_tep} added to finalist "
                f"eval (always-evaluate final {_ALWAYS_EVAL_FINAL_EPOCHS} epochs)."
            )
            # Checkpoint-retention bookkeeping — worker log only (debug).
            logger.debug(_tail_line)

    logger.info(
        f"[{job_id}] Model Testing: running YOLO-Pro export-safety eval over "
        f"the in-scope subset of {len(checkpoint_manifest)} retained finalist(s) … "
        f"conf_threshold={_eval_conf} (AP) / "
        f"runtime_conf_threshold={_runtime_safety_conf} (flood gate) "
        f"(eval_mode={_eval_mode}, tiny_dataset_mode={policy['tiny_dataset_mode']})"
    )

    def _finite(v: object) -> "float | None":
        if isinstance(v, (int, float)) and math.isfinite(float(v)):
            return float(v)
        return None

    def _report_preds_per_image(report: dict, total_key: str) -> "float | None":
        _total = report.get(total_key)
        _imgs  = report.get("evaluated_images")
        if (
            isinstance(_total, (int, float)) and math.isfinite(float(_total))
            and isinstance(_imgs, (int, float)) and _imgs is not None
            and float(_imgs) > 0
        ):
            return float(_total) / float(_imgs)
        return None

    # Evaluate the finalists on the full validation set, best-first, and
    # backfill the canonical ``full_*`` metrics + ranking_status onto each
    # evaluated manifest entry.  ``_finalist_reports`` stays index-aligned with
    # ``checkpoint_manifest`` (pre-sized, filled by manifest index) so the
    # selection / rejection logic below can recover each entry's eval report.
    # Phase E: the disk manifest is the SOLE candidate store — score each
    # finalist by LOADING its admission ``.keras`` (keyed by epoch).  The saved
    # model is bit-for-bit the admission model (Phase A fidelity), so the
    # backfilled ``full_*`` metrics — and therefore the ranking / selection
    # below — reflect the exact weights that were admitted.
    #
    # Eval scope: ``_select_final_eval_order`` restricts the primary candidate
    # set to close-mosaic (mosaic-OFF) checkpoints and ranks them best-by-loss
    # first; the loop stops at the first ranking-ok finalist.  In the healthy
    # case that is exactly ONE full eval instead of one per retained
    # checkpoint.  Lower-ranked entries are only reached when every better
    # candidate collapses, which keeps the ``_NO_SAFE_CKPT_ERROR`` safety net.
    _manifest_by_epoch = {
        _m["epoch"]: _m
        for _m in checkpoint_manifest
        if _m.get("epoch") is not None
    }
    _use_disk_finalist_eval = _YOLO_PRO_FINALIST_EVAL_FROM_DISK
    _eval_order, _eval_scope = _select_final_eval_order(checkpoint_manifest)
    _n_mosaic_off = sum(
        1 for _e in checkpoint_manifest if not bool(_e.get("mosaic_active"))
    )
    # Indices of the always-evaluate final epochs: the early-stop-at-first-``ok``
    # shortcut is held until every one of these has been scored, so the final
    # epochs are always compared head-to-head with the monitor checkpoints.
    _always_eval_indices = {
        i for i, _e in enumerate(checkpoint_manifest) if bool(_e.get("always_eval"))
    }
    _always_eval_seen: set = set()
    _found_ok_finalist: bool = False
    _scope_line = (
        f"[{job_id}] Model Testing: eval scope={_eval_scope} — "
        f"{_n_mosaic_off}/{len(checkpoint_manifest)} retained checkpoint(s) are "
        f"post-close-mosaic; always-evaluating {len(_always_eval_indices)} "
        f"final epoch(s), then stopping at the first ranking-ok finalist."
    )
    # Eval-scope reasoning — worker log only (debug), not the UI stream.
    logger.debug(_scope_line)
    _finalist_reports: List["dict | None"] = [None] * len(checkpoint_manifest)
    for _rank, _idx in enumerate(_eval_order):
        _entry = checkpoint_manifest[_idx]
        # Mark always-evaluate (final-epoch) entries as processed the moment we
        # reach them, so the early-stop-at-first-``ok`` break below is held until
        # every final epoch has been scored (they are ordered first).
        if _idx in _always_eval_indices:
            _always_eval_seen.add(_idx)
        _raise_if_cancelled(job_id)
        # Load the retained ``.keras`` for this finalist — the exact admission
        # model, no BN drift between admission and this end-of-training eval.
        _eval_model, _eval_src = _resolve_finalist_eval_model(
            _entry, _manifest_by_epoch, model, ema,
            use_disk=_use_disk_finalist_eval,
        )
        if _eval_model is None:
            # The retained ``.keras`` is missing (a rare best-effort write
            # failure).  Record an inference_failed report so this finalist is
            # ranking-rejected instead of crashing the eval loop — it simply
            # cannot win selection.
            logger.error(
                f"[{job_id}] Model Testing: finalist EP{_entry.get('epoch')} "
                f"checkpoint file missing ({_entry.get('path')}) — marking "
                f"inference_failed."
            )
            _rep = {
                "yolo_pro_eval_status": "inference_failed",
                "yolo_pro_eval_error":  "checkpoint_file_missing",
                "map": None, "map50": None, "map75": None,
                "precision": None, "recall": None, "per_class": {},
                "evaluated_images": int(len(X_test)),
            }
            _rep["conf_threshold"] = _eval_conf
            _rep["nms_iou_threshold"] = 0.45
            _status = _classify_ranking_health(_rep)
            _rep["ranking_status"] = _status
            _entry["full_safety_map50"]            = None
            _entry["full_safety_precision"]        = None
            _entry["full_safety_tp_fp_gap"]        = None
            _entry["full_runtime_preds_per_image"] = None
            _entry["full_evaluated_images"]        = _rep.get("evaluated_images")
            _entry["ranking_status"]               = _status
            _finalist_reports[_idx] = _rep
            _mt_line = (
                f"[{job_id}] Model Testing: finalist {_rank + 1}/{len(_eval_order)} "
                f"EP{_entry.get('epoch')} (src=missing) → status={_status} "
                f"full(map50=None prec=None gap=None)"
            )
            logger.info(_mt_line)
            _yolo_log_lines.append(_mt_line)
            continue
        try:
            # Canonical export-safety eval — identical inputs and parameters to
            # the per-candidate admission probe, so any divergence here is
            # model-state drift, not eval-config drift.
            _rep = _run_export_safety_eval(
                model=_eval_model,
                X_test=X_test,
                boxes_test=boxes_test,
                label_names=label_names,
                label_map=label_map,
                anchor_grids=anchor_grids,
                H=H, W=W, reg_max=reg_max,
                # 0.005 COCO-style threshold for full PR-curve mAP — eval only.
                conf_threshold=_eval_conf,
                runtime_conf_threshold=_runtime_safety_conf,
                job_id=job_id,
            )
        except Exception as _eval_exc:
            logger.error(
                f"[{job_id}] Model Testing: finalist EP{_entry.get('epoch')} "
                f"eval raised an unexpected exception: {_eval_exc}",
                exc_info=True,
            )
            _rep = {
                "yolo_pro_eval_status": "inference_failed",
                "yolo_pro_eval_error":  str(_eval_exc),
                "map": None, "map50": None, "map75": None,
                "precision": None, "recall": None, "per_class": {},
                "evaluated_images": int(len(X_test)),
            }
        # Record the eval thresholds (0.005 AP-style) for audit parity with the
        # legacy single-eval path.
        _rep["conf_threshold"] = _eval_conf
        _rep["nms_iou_threshold"] = 0.45
        _status = _classify_ranking_health(_rep)
        _rep["ranking_status"] = _status

        # Backfill canonical full-validation metrics onto the buffer entry so
        # export ranking (``_select_export_restore_index`` /
        # ``_export_candidate_rank_key``) reads authoritative numbers, and the
        # Phase-0 final summary archives them.
        _rep_tp = _rep.get("tp_score_mean")
        _rep_fp = _rep.get("fp_score_mean")
        _rep_gap = (
            float(_rep_tp) - float(_rep_fp)
            if isinstance(_rep_tp, (int, float)) and isinstance(_rep_fp, (int, float))
            and math.isfinite(float(_rep_tp)) and math.isfinite(float(_rep_fp))
            else None
        )
        _entry["full_safety_map50"]            = _finite(_rep.get("map50"))
        _entry["full_safety_precision"]        = _finite(_rep.get("precision"))
        _entry["full_safety_tp_fp_gap"]        = _rep_gap
        _entry["full_runtime_preds_per_image"] = _extract_runtime_preds_per_image(_rep)
        _entry["full_evaluated_images"]        = _rep.get("evaluated_images")
        _entry["ranking_status"]               = _status
        _finalist_reports[_idx] = _rep

        def _fmt(v: object, spec: str = ".4f") -> str:
            if isinstance(v, (int, float)) and math.isfinite(float(v)):
                return format(float(v), spec)
            return "None"
        _mt_line = (
            f"[{job_id}] Model Testing: finalist {_rank + 1}/{len(_eval_order)} "
            f"EP{_entry.get('epoch')} (reason={_entry.get('checkpoint_reason') or 'monitor'}, "
            f"src={_eval_src}, mosaic_active={bool(_entry.get('mosaic_active'))}) "
            f"→ status={_status} "
            f"full(map50={_fmt(_rep.get('map50'))} prec={_fmt(_rep.get('precision'))} "
            f"gap={_fmt(_rep_gap)})"
        )
        logger.info(_mt_line)
        _yolo_log_lines.append(_mt_line)
        if _status == "ok":
            _found_ok_finalist = True
        if _found_ok_finalist and _always_eval_indices.issubset(_always_eval_seen):
            # A ranking-ok finalist has been found AND every always-evaluate
            # final epoch has been scored.  Remaining monitor entries are worse
            # on the pre-eval monitor (or mosaic-on), so evaluating them cannot
            # change the export decision — the export ranking already reads the
            # backfilled full-validation metrics for the final epochs and picks
            # the best-by-mAP among all ranking-ok finalists.  Stop here; that
            # is the eval-time saving.
            break

    # Mark every finalist the scoped eval never reached.  They keep null
    # ``full_*`` metrics and a non-"ok" ranking_status so they are excluded
    # from export selection, while the index-aligned report list stays intact
    # for the rejection-summary path below.
    _n_evaluated = sum(1 for _r in _finalist_reports if _r is not None)
    for _i, _r in enumerate(_finalist_reports):
        if _r is not None:
            continue
        _skipped_rep = {
            "yolo_pro_eval_status": "skipped",
            "yolo_pro_eval_error":  "not_in_eval_scope",
            "map": None, "map50": None, "map75": None,
            "precision": None, "recall": None, "per_class": {},
            "evaluated_images": 0,
            "conf_threshold": _eval_conf,
            "nms_iou_threshold": 0.45,
            "ranking_status": "skipped",
        }
        _finalist_reports[_i] = _skipped_rep
        checkpoint_manifest[_i]["ranking_status"] = "skipped"
    if _n_evaluated < len(checkpoint_manifest):
        _skip_line = (
            f"[{job_id}] Model Testing: evaluated {_n_evaluated}/"
            f"{len(checkpoint_manifest)} retained checkpoint(s) — "
            f"{len(checkpoint_manifest) - _n_evaluated} skipped by eval scope "
            f"({_eval_scope})."
        )
        # Checkpoint-retention bookkeeping — worker log only (debug).
        logger.debug(_skip_line)

    # ── Select the shipped checkpoint ────────────────────────────────────────
    # Ship the highest full_safety_map50 among finalists whose full-validation
    # ranking health came back "ok" — reusing the same ranking key
    # (_select_export_restore_index / _export_candidate_rank_key) as the in-loop
    # restore preference, now reading the backfilled ``full_*`` fields.
    _ok_indices = [
        i for i, _e in enumerate(checkpoint_manifest)
        if _e.get("ranking_status") == "ok"
    ]
    if not _ok_indices:
        # No finalist survived the full-validation ranking-health gate.  Every
        # accepted checkpoint passed the cheap probe at training time but
        # collapsed on the full validation set — the same silent failure the
        # safety gates exist to prevent.  Fail with the combined rejection
        # summary (per-epoch monitor rejections + per-finalist final-eval
        # rejections), preserving the exact cause tags of the pre-Phase-2
        # walk-back / final-eval-collapsed paths.
        _testing_rejections: List[dict] = []
        for i, _e in enumerate(checkpoint_manifest):
            _rep = _finalist_reports[i]
            _fb_failures = _checkpoint_quality_gate_failures(_rep)
            # Admission-time probe metrics ("saved") vs authoritative full eval
            # ("final") — a large delta here means the probe over-admitted.
            _testing_rejections.append({
                "epoch":          _e.get("epoch"),
                "monitor_metric": _e.get("ckpt_metric_name"),
                "monitor_value":  _e.get("monitor_value"),
                "failed_gates":   _fb_failures,
                "val_cls":        _e.get("val_cls"),
                "saved_ckpt_val_cls": None,
                "saved_full_safety_map50":     _e.get("probe_map50", _e.get("safety_map50")),
                "saved_full_safety_precision": _e.get("probe_precision", _e.get("safety_precision")),
                "final_eval_map50":            _rep.get("map50"),
                "final_eval_precision":        _rep.get("precision"),
                "eval_mode_match":             True,   # canonical fn used on both sides
                "safety_metrics": {
                    "tp_score_mean": _rep.get("tp_score_mean"),
                    "fp_score_mean": _rep.get("fp_score_mean"),
                    "precision":     _rep.get("precision"),
                    "map50":         _rep.get("map50"),
                    "preds_per_image":         _report_preds_per_image(_rep, "total_predicted_cells"),
                    "runtime_preds_per_image": _extract_runtime_preds_per_image(_rep),
                    "runtime_conf_threshold":  _rep.get("runtime_conf_threshold"),
                },
            })
        _combined = list(_rejected_ckpt_candidates) + _testing_rejections
        _summary = _summarize_rejected_candidates(_combined)
        # Preserve the pre-Phase-2 cause tags: a single collapsed finalist maps
        # to "final_eval_collapsed"; multiple exhausted finalists to
        # "walkback_exhausted".  (empty_buffer is handled above.)
        _cause = (
            "final_eval_collapsed" if len(checkpoint_manifest) <= 1
            else "walkback_exhausted"
        )
        _err_msg = _format_no_safe_ckpt_message(_summary, cause=_cause)
        logger.error(f"[{job_id}] {_err_msg}")
        _yolo_log_lines.append(_err_msg)
        job.training_history = _sanitize_json({
            "log_lines": list(_yolo_log_lines),
            "is_yolo_pro": True,
            "no_safe_checkpoint_summary": _summary,
            **_calibration_history_keys(),
        })
        try:
            db.commit()
        except Exception:
            db.rollback()
        raise RuntimeError(_err_msg)

    _ok_entries = [checkpoint_manifest[i] for i in _ok_indices]
    _sel_local_idx, _restore_preference = _select_export_restore_index(_ok_entries)
    _restore_idx    = _ok_indices[_sel_local_idx]
    _restored_entry = checkpoint_manifest[_restore_idx]
    eval_report     = _finalist_reports[_restore_idx]
    _ranking_status = _restored_entry.get("ranking_status") or "ok"

    _restored_safety_map50  = _restored_entry.get("full_safety_map50")
    _restored_monitor_value = _restored_entry.get("monitor_value")
    _restored_reason        = _restored_entry.get("checkpoint_reason") or "monitor"
    if best_ckpt_metric_name == "val_box_dfl" and best_val_box_dfl < float("inf"):
        restore_tag = f"criterion=val_box_dfl  best_val_box_dfl={best_val_box_dfl:.4f}"
    elif best_ckpt_metric_name == "val_loss" and best_val_loss < float("inf"):
        # best_val_box_dfl is the box+DFL loss recorded AT the checkpoint epoch
        # (not a global minimum across all epochs).  This correctly reflects the
        # localization quality of the weights being exported.
        _box_tag = (
            f"  [val_box_dfl@ckpt={best_val_box_dfl:.4f}]"
            if val_box_dfl_history and best_val_box_dfl < float("inf") else ""
        )
        restore_tag = (
            f"criterion=val_loss  best_val_loss={best_val_loss:.4f}{_box_tag}"
        )
    else:
        restore_tag = f"criterion=train_loss  best_train_loss={best_loss:.4f}"
    _safety_map50_str = (
        f"{float(_restored_safety_map50):.4f}"
        if isinstance(_restored_safety_map50, (int, float))
        and math.isfinite(float(_restored_safety_map50))
        else "None"
    )
    _monitor_value_str = (
        f"{float(_restored_monitor_value):.4f}"
        if isinstance(_restored_monitor_value, (int, float))
        and math.isfinite(float(_restored_monitor_value))
        else "None"
    )
    logger.info(
        f"[{job_id}] Model Testing selected EMA checkpoint EP{_restored_entry['epoch']} "
        f"(restore_preference={_restore_preference}  "
        f"restored_reason={_restored_reason}  "
        f"restored_full_safety_map50={_safety_map50_str}  "
        f"ranking_status={_ranking_status}  "
        f"restored_monitor_value={_monitor_value_str}  "
        f"{restore_tag}) before export/eval.  "
        f"Selected from {len(_ok_indices)}/{len(checkpoint_manifest)} ranking-ok finalist(s)."
    )
    _yolo_log_lines.append(
        f"Model Testing selected EMA checkpoint EP{_restored_entry['epoch']} "
        f"(restore_preference={_restore_preference}  "
        f"restored_reason={_restored_reason}  "
        f"restored_full_safety_map50={_safety_map50_str}  "
        f"ranking_status={_ranking_status}  "
        f"{restore_tag}) before export/eval."
    )

    # ── Phase 0 end-of-training summary (now backfilled with full metrics) ────
    # Record the selected/exported epoch, its full export-safety mAP50, and the
    # full retained-manifest contents (epoch + full metrics + ranking_status)
    # so the baseline for Phases 1-3 is archived per job in training_history.
    _ph0_buffer_contents = [
        {
            "epoch":                        _e.get("epoch"),
            "checkpoint_reason":            _e.get("checkpoint_reason"),
            "monitor_metric":               _e.get("monitor_metric"),
            "monitor_value":                _e.get("monitor_value"),
            "full_safety_map50":            _e.get("full_safety_map50"),
            "full_safety_precision":        _e.get("full_safety_precision"),
            "full_safety_tp_fp_gap":        _e.get("full_safety_tp_fp_gap"),
            "full_runtime_preds_per_image": _e.get("full_runtime_preds_per_image"),
            "full_evaluated_images":        _e.get("full_evaluated_images"),
            "ranking_status":               _e.get("ranking_status"),
            "mosaic_active":                _e.get("mosaic_active"),
        }
        for _e in checkpoint_manifest
    ]
    _phase0_final_summary.clear()
    _phase0_final_summary.update({
        "selected_epoch":             _restored_entry.get("epoch"),
        "selected_full_safety_map50": _restored_entry.get("full_safety_map50"),
        "restore_preference":         _restore_preference,
        "buffer_contents":            _ph0_buffer_contents,
    })
    _ph0_buf_str = "; ".join(
        f"EP{_c['epoch']}(reason={_c['checkpoint_reason']},"
        f"full_map50={_c['full_safety_map50']},status={_c['ranking_status']})"
        for _c in _ph0_buffer_contents
    )
    _ph0_final_line = (
        f"[{job_id}] PHASE0-FINAL exported_epoch={_restored_entry.get('epoch')} "
        f"exported_full_safety_map50={_restored_entry.get('full_safety_map50')} "
        f"buffer=[{_ph0_buf_str}]"
    )
    # Backend diagnostic only — not surfaced in the UI terminal
    # (kept in logs + persisted to training_history["phase0_final_summary"]).
    logger.info(_ph0_final_line)

    job.training_history = _sanitize_json({
        "log_lines": list(_yolo_log_lines),
        "is_yolo_pro": True,
        **_calibration_history_keys(),
    })
    try:
        db.commit()
    except Exception:
        db.rollback()

    # Phase E: resolve the selected finalist as a model to EXPORT by loading
    # its admission ``.keras`` from disk — the exact model that produced
    # ``eval_report`` (deterministic load — Phase A fidelity), so the exported
    # artifacts match the reported metrics.  The selected finalist is ranking-ok
    # (its eval loaded successfully), and retention eviction only runs during
    # training, so its file is still present here.
    _export_model, _export_src = _resolve_finalist_eval_model(
        _restored_entry, _manifest_by_epoch, model, ema,
        use_disk=_use_disk_finalist_eval,
    )
    if _export_model is None:
        # Defensive: the selected checkpoint file vanished between eval and
        # export.  Fail loudly rather than export unvalidated live weights.
        _err_msg = _format_no_safe_ckpt_message(
            _summarize_rejected_candidates(_rejected_ckpt_candidates),
            cause="final_eval_collapsed",
        )
        logger.error(
            f"[{job_id}] selected checkpoint EP{_restored_entry.get('epoch')} "
            f"file missing at export time ({_restored_entry.get('path')})."
        )
        raise RuntimeError(_err_msg)
    logger.info(
        f"[{job_id}] Export model source for EP{_restored_entry.get('epoch')}: "
        f"{_export_src}"
    )

    # Check after training loop, before export — cancelled jobs must not
    # produce exported models or be written as completed.
    _raise_if_cancelled(job_id)

    # ── Export ────────────────────────────────────────────────────────────────
    version = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

    # Derive normalize_input from the first DSP block's image_scaling config.
    # YOLO-Pro always uses an image DSP block; "0..1" scaling means the inference
    # template must divide pixel values by 255.0 before running the model.
    _dsp_blocks = list(impulse.dsp_blocks or [])
    _dsp0_config = (_dsp_blocks[0] or {}).get("params", {}) if _dsp_blocks else {}
    _image_scaling = _dsp0_config.get("image_scaling", "0..1")
    _normalize_input = (_image_scaling == "0..1")

    _common_meta = {
        "project_id":      impulse.project_id,
        "input_shape":     list(input_shape),
        "num_classes":     n_classes,
        "label_names":     label_names,
        "architecture":    YOLO_PRO_ARCHITECTURE,
        "output_type":     "yolo_pro_detection",
        "model_type":      "yolo_pro_detection",
        "normalize_input": _normalize_input,
        "channel_order":   "rgb",
        "version":         1,
        "yolo_size":       size,
        "reg_max":         reg_max,
        "dsp_blocks":      _dsp_blocks,
    }

    # ── Float32 twin for TFLite/PTQ export ──────────────────────────────────
    # ROOT-CAUSE FIX for INT8 ERROR_NEEDS_FLEX_OPS.
    #
    # ``_export_model`` was built (line ~4060) while the global policy was
    # ``mixed_float16`` (GPU path), so every layer captured compute_dtype=
    # float16 / variable_dtype=float32. Restoring the global policy to fp32
    # afterwards (line ~4070) does NOT change those already-constructed layers,
    # and the disk-restored ``.keras`` twin preserves the same per-layer policy
    # in its config — so the frozen inference graph traced in export.py is
    # float16 end to end (tensor<…xf16>). INT8 PTQ cannot lower a float16
    # Conv2D/DepthwiseConv2d/etc. to a builtin int8 kernel, and because
    # ``target_spec.supported_ops`` is int8-only the converter falls back to
    # demanding Flex/TF-Select (ERROR_NEEDS_FLEX_OPS). The FP32 and decoded
    # exports "succeed" only because they leave supported_ops unconstrained, so
    # the MLIR path keeps the ops as builtin float16 kernels.
    #
    # The variables ARE float32 masters, so rebuilding the architecture under a
    # float32 policy and copying the weights is lossless. Use this fp32 twin for
    # ALL TFLite exports; keep the trained model for the .keras artifact. On any
    # failure, fall back to ``_export_model`` so the fp32/decoded exports still
    # run exactly as before (INT8 will still fail-isolated, as it did).
    _tflite_model = _export_model
    _prev_export_policy = tf.keras.mixed_precision.global_policy()
    try:
        tf.keras.mixed_precision.set_global_policy("float32")
        _fp32_twin = build_yolo_pro(
            input_shape=input_shape,
            num_classes=n_classes,
            size=size,
            reg_max=reg_max,
            prior_prob=float(balance_policy["prior_prob"]),
        )
        _fp32_twin.set_weights(_export_model.get_weights())
        _tflite_model = _fp32_twin
        logger.info(
            f"[{job_id}] Built float32 export twin for TFLite/PTQ "
            f"(compute_dtype={_fp32_twin.dtype_policy.compute_dtype})."
        )
    except Exception as _twin_exc:
        logger.warning(
            f"[{job_id}] float32 export-twin build failed "
            f"({_twin_exc}); falling back to trained model for TFLite export "
            f"(INT8 PTQ may require a float32 graph)."
        )
    finally:
        tf.keras.mixed_precision.set_global_policy(_prev_export_policy)

    # Export artifacts, kept in scope past the temp dir so the post-export
    # per-variant eval below can score each one.  Pre-initialised to None so a
    # failed export leaves a falsy marker rather than an unbound name.
    tflite_f32_bytes: "bytes | None" = None
    tflite_int8_bytes: "bytes | None" = None
    tflite_decoded_bytes: "bytes | None" = None
    _f32_export_error: "str | None" = None
    _decoded_export_error: "str | None" = None

    with tempfile.TemporaryDirectory() as tmpdir:
        keras_path = os.path.join(tmpdir, "model.keras")
        _export_model.save(keras_path)
        with open(keras_path, "rb") as f:
            keras_key = storage.model_key(impulse.project_id, version, "model.keras")
            storage.upload_file(f, keras_key, "application/octet-stream")
        db.add(TrainedModel(
            training_job_id=job_id, version=version, format="keras",
            storage_key=keras_key, file_size_bytes=os.path.getsize(keras_path),
            model_metadata={**_common_meta, "variant": "float32", "quantized": False},
        ))

        _raise_if_cancelled(job_id)
        tflite_f32_path = os.path.join(tmpdir, "model_float32.tflite")
        try:
            tflite_f32_bytes = export_tflite_float32(
                model=_tflite_model, output_path=tflite_f32_path, input_shape=input_shape,
            )
            f32_key = storage.model_key(impulse.project_id, version, "model_float32.tflite")
            storage.upload_bytes(tflite_f32_bytes, f32_key, "application/octet-stream")
            db.add(TrainedModel(
                training_job_id=job_id, version=version, format="tflite",
                storage_key=f32_key, file_size_bytes=len(tflite_f32_bytes),
                model_metadata={**_common_meta, "variant": "float32", "quantized": False},
            ))
            logger.info(f"[{job_id}] float32 TFLite: {len(tflite_f32_bytes)//1024} KB")
        except Exception as e:
            _f32_export_error = str(e)
            logger.warning(f"[{job_id}] float32 TFLite export failed: {e}")

        _raise_if_cancelled(job_id)
        tflite_int8_path = os.path.join(tmpdir, "model_int8.tflite")
        int8_status = "available"; int8_error = None
        try:
            rep_data = X_train[:200] if len(X_train) >= 200 else X_train
            tflite_int8_bytes = export_tflite_int8(
                model=_tflite_model, output_path=tflite_int8_path, input_shape=input_shape,
                representative_data=rep_data,
                num_calibration_steps=min(100, len(rep_data)),
            )
            int8_key = storage.model_key(impulse.project_id, version, "model_int8.tflite")
            storage.upload_bytes(tflite_int8_bytes, int8_key, "application/octet-stream")
            db.add(TrainedModel(
                training_job_id=job_id, version=version, format="tflite",
                storage_key=int8_key, file_size_bytes=len(tflite_int8_bytes),
                model_metadata={**_common_meta, "variant": "int8", "quantized": True},
            ))
            logger.info(f"[{job_id}] int8 TFLite: {len(tflite_int8_bytes)//1024} KB")
        except Exception as e:
            int8_status = "failed"; int8_error = str(e)
            logger.warning(f"[{job_id}] int8 TFLite export failed: {e}")

        _raise_if_cancelled(job_id)
        # ── Decoded float32 TFLite — OG-style single decoded output ──────────
        # Produces (1, N_total, 6) = [x1, y1, x2, y2, score, class_id], matching
        # the OG Edge Impulse export format: TensorSpec(shape=(None,189,6), f32).
        tflite_decoded_path = os.path.join(tmpdir, "model_decoded.tflite")
        try:
            from app.ml.yolo_pro.export import export_tflite_float32_decoded
            tflite_decoded_bytes = export_tflite_float32_decoded(
                model=_tflite_model, output_path=tflite_decoded_path,
                input_shape=input_shape, reg_max=reg_max,
            )
            decoded_key = storage.model_key(
                impulse.project_id, version, "model_decoded.tflite"
            )
            storage.upload_bytes(tflite_decoded_bytes, decoded_key, "application/octet-stream")
            db.add(TrainedModel(
                training_job_id=job_id, version=version, format="tflite",
                storage_key=decoded_key, file_size_bytes=len(tflite_decoded_bytes),
                model_metadata={
                    **_common_meta,
                    "variant":  "decoded_float32",
                    "quantized": False,
                    "output_format": "decoded_xyxy_score_cls",
                },
            ))
            logger.info(f"[{job_id}] decoded float32 TFLite: {len(tflite_decoded_bytes)//1024} KB")
        except Exception as e:
            _decoded_export_error = str(e)
            logger.warning(f"[{job_id}] decoded float32 TFLite export failed: {e}")

    # ── Per-variant export evaluation ────────────────────────────────────────
    # Every exported variant is scored INDEPENDENTLY on the same validation set
    # via the same canonical eval used for finalist selection, so the Model
    # Version dropdown shows each variant's real numbers instead of float32's.
    # int8 quantization loss is exactly what this surfaces.  Post-export only:
    # nothing here retrains, re-selects a checkpoint, or changes what is
    # exported.  Failures are recorded per variant and never raise.
    try:
        from app.ml.variant_eval import evaluate_exported_variants

        def _score_variant(_runner) -> dict:
            return _run_export_safety_eval(
                model=_runner,
                X_test=X_test,
                boxes_test=boxes_test,
                label_names=label_names,
                label_map=label_map,
                anchor_grids=anchor_grids,
                H=H, W=W, reg_max=reg_max,
                conf_threshold=_eval_conf,
                runtime_conf_threshold=_runtime_safety_conf,
                job_id=job_id,
            )

        evaluate_exported_variants(
            db=db,
            job_id=job_id,
            scorer=_score_variant,
            variant_bytes={
                "float32":         tflite_f32_bytes,
                "int8":            tflite_int8_bytes,
                "decoded_float32": tflite_decoded_bytes,
            },
            runner_kwargs={
                "output_type": "yolo_pro_detection",
                "num_classes": n_classes,
                "reg_max":     reg_max,
                "input_h":     H,
                "input_w":     W,
            },
            export_errors={
                "float32":         _f32_export_error,
                "int8":            int8_error,
                "decoded_float32": _decoded_export_error,
            },
            log_prefix=f"[{job_id}] ",
        )
    except Exception as _ve_exc:
        # Belt-and-braces: per-variant metrics are additive telemetry, never a
        # reason to fail a training run that has already produced artifacts.
        logger.warning(
            f"[{job_id}] per-variant export eval skipped: {_ve_exc}", exc_info=True
        )

    # Runtime/deployment threshold — NOT the permissive mAP eval value.
    _runtime_conf = _compute_runtime_conf_threshold(policy, size)

    # Backfill the runtime threshold AND ranking_status into every
    # TrainedModel row so the deployment manifest, model testing, and
    # inference path can read a single source of truth.
    try:
        trained_models_this_job = (
            db.query(TrainedModel)
            .filter(TrainedModel.training_job_id == job_id)
            .all()
        )
        for _tm in trained_models_this_job:
            _meta = dict(_tm.model_metadata or {})
            _meta["threshold"] = _runtime_conf
            _meta["ranking_status"] = _ranking_status
            _tm.model_metadata = _meta
        logger.info(
            f"[{job_id}] Stamped runtime threshold={_runtime_conf} "
            f"(eval threshold was {_eval_conf}) and ranking_status="
            f"{_ranking_status} into "
            f"{len(trained_models_this_job)} TrainedModel metadata row(s)."
        )
    except Exception as _stamp_exc:
        logger.warning(
            f"[{job_id}] Could not stamp threshold into TrainedModel metadata: {_stamp_exc}"
        )

    _log_size_benchmark_entry(
        job_id=job_id,
        size=size,
        eval_report=eval_report,
        ckpt_metric=best_ckpt_metric_name,
        actual_epochs=actual_epochs,
        effective_epochs=effective_epochs,
        base_lr=train_cfg.base_lr,
    )

    # ── Write DB results ───────────────────────────────────────────────────────
    # job.best_loss holds the metric that was actually used for early stopping
    # and checkpointing: best_val_loss when val data was available, else best
    # training loss.  The UI surfaces this as "BEST_VAL_LOSS" for YOLO-Pro jobs.
    if best_ckpt_metric_name == "val_box_dfl" and math.isfinite(best_val_box_dfl):
        reported_best = float(best_val_box_dfl)
    elif best_ckpt_metric_name == "val_loss" and math.isfinite(best_val_loss):
        reported_best = float(best_val_loss)
    else:
        reported_best = float(best_loss)
    # Final guard before marking completed: if the user cancelled during
    # eval / TFLite export / artifact upload (sections that don't have
    # per-iteration cancel polls), preserve the cancelled status instead
    # of overwriting it with 'completed'.
    _assert_not_cancelled_before_completing(job_id)
    job.status        = JobStatus.completed
    job.completed_at  = datetime.utcnow()
    # Pointer flip — successful completion of any run kind becomes the
    # new Active Model. Never cleared elsewhere.
    _promote_run_to_active(job.impulse_id, job.id)
    job.best_loss     = reported_best
    job.best_accuracy  = None
    job.final_accuracy = None

    # _yolo_log_lines already accumulated setup + per-epoch lines incrementally
    # during training.  No rebuild needed here.

    job.training_history = {
        "loss":                [float(v) for v in loss_history],
        "best_loss":           float(best_loss) if math.isfinite(best_loss) else None,
        "best_checkpoint_metric": best_ckpt_metric_name,
        "best_checkpoint_value": (
            float(best_ckpt_metric_value)
            if math.isfinite(best_ckpt_metric_value) else None
        ),
        "val_loss":            [float(v) for v in val_loss_history],
        "val_loss_history":    [float(v) for v in val_loss_history],
        # best_val_loss is the TRUE best validation loss (not a composite).
        # Checkpointing criterion is val_box_dfl (primary) → val_loss → train_loss.
        "best_val_loss":       (
            float(best_val_loss)
            if val_loss_history and math.isfinite(best_val_loss)
            else None
        ),
        "checkpoint_criterion": best_ckpt_metric_name,
        "val_box_dfl_history": [float(v) for v in val_box_dfl_history],
        # best_val_box_dfl is the val_box_dfl AT the checkpoint epoch — not the
        # global minimum across all epochs.  It reflects the localization
        # quality of the exported model, not an optimistic earlier value.
        "val_box_dfl_at_ckpt": (
            float(best_val_box_dfl)
            if val_box_dfl_history and math.isfinite(best_val_box_dfl)
            else None
        ),
        # Legacy key kept for backwards compatibility with older API consumers.
        "best_val_box_dfl":    (
            float(best_val_box_dfl)
            if val_box_dfl_history and math.isfinite(best_val_box_dfl)
            else None
        ),
        # Epoch / step bookkeeping — distinguish what the user asked for vs what
        # actually ran so the UI can report "ran N effective epochs (M requested)".
        "epochs":              _requested_epochs,   # user-requested, unchanged (UI-facing)
        "requested_epochs":    _requested_epochs,   # legacy alias
        "effective_epochs":    effective_epochs,    # internal training budget
        "actual_epochs":       actual_epochs,
        "stopped_early":       stopped_early,
        "stop_reason":         stop_reason,
        "patience":            PATIENCE,
        # Batch-size and step-budget fields from the stabilization policy.
        "requested_batch_size":    job.batch_size,  # None if user did not set it
        "effective_batch_size":    policy["eff_batch_size"],
        "steps_per_epoch":         policy["steps_per_epoch"],
        "total_steps":             policy["total_steps"],
        "tiny_dataset_mode":       policy["tiny_dataset_mode"],
        "batch_size_reduced":      policy["batch_size_reduced"],
        "epochs_expanded":         policy["epochs_expanded"],
        "dataset_complexity": {
            k: (float(v) if isinstance(v, float) else v)
            for k, v in dataset_stats.items()
        },
        "training_balance_policy": {
            k: (float(v) if isinstance(v, float) else v)
            for k, v in current_balance.items()
        },
        "is_yolo_pro":      True,
        "yolo_size":        size,
        "reg_max":          reg_max,
        "ema_decay":        float(ema_decay),
        "metric_note": (
            "YOLO-Pro detection: no scalar accuracy. "
            "Evaluate mAP50/mAP75/mAP50-95 on the exported model using "
            "COCO-style evaluation. mAP75 is reported but NOT used as the "
            "checkpoint trigger — see checkpoint_policy_note."
        ),
        # Why mAP75 and val_box_dfl are NOT the primary checkpoint triggers:
        #
        # On tiny validation sets (e.g. 5 images) mAP75 is extremely noisy:
        # a single missed detection flips it from 1.0 to 0.0.  Using it as the
        # primary trigger causes erratic checkpointing that may export a model
        # that was just lucky on the tiny val set rather than genuinely better.
        #
        # val_box_dfl alone is also unsafe: when the model collapses to fewer
        # classes, TAL assigns fewer positives → normalised box_dfl drops even
        # though recall collapsed (observed in production).
        #
        # val_loss (total) is stable even on small sets because VFL sums over
        # ALL anchor cells — a class-collapse keeps VFL high for abandoned GT
        # cells, so val_loss rises and the checkpoint is NOT saved.
        #
        # val_box_dfl_at_ckpt is stored as a diagnostic: it reflects the
        # localization quality of the exported epoch, not a separate trigger.
        "checkpoint_policy_note": (
            "Primary criterion: val_loss (total). "
            "Fallback when no val data: train_loss. "
            "mAP75 excluded: too noisy on tiny validation sets. "
            "val_box_dfl excluded as primary: rewards class-collapse silently. "
            "val_box_dfl_at_ckpt = diagnostic snapshot at the checkpoint epoch."
        ),
        "int8_status":       int8_status,
        "int8_error":        int8_error,
        "log_lines":         _yolo_log_lines,
        "log_lines_complete": True,
        # Diagnostic-only calibration snapshot — full per-epoch trace and
        # the adapter-firing aggregate.  Consumers must not treat any of
        # these fields as a ranking metric; the checkpoint criterion is
        # ``checkpoint_criterion`` above.
        **_calibration_history_keys(),
    }
    job.training_history      = _sanitize_json(job.training_history)
    job.confusion_matrix      = []
    job.classification_report = _sanitize_json(eval_report)
    try:
        db.commit()
    except Exception:
        db.rollback()
        raise

    _tiny_tag  = "  tiny_dataset_mode=ON" if policy["tiny_dataset_mode"] else ""
    _exp_tag   = (
        f"  epochs_expanded={_requested_epochs}→{effective_epochs}"
        if policy["epochs_expanded"] else ""
    )
    _bs_tag    = (
        f"  batch_reduced={job.batch_size!r}→{policy['eff_batch_size']}"
        if policy["batch_size_reduced"] else ""
    )
    _ckpt_tag  = (
        f"best_val_box_dfl={best_val_box_dfl:.4f}"
        if best_ckpt_metric_name == "val_box_dfl" and best_val_box_dfl < float("inf")
        else (
            f"best_val_loss={best_val_loss:.4f}"
            if best_ckpt_metric_name == "val_loss" and best_val_loss < float("inf")
            else f"best_train_loss={best_loss:.4f}"
        )
    )
    logger.info(
        f"[{job_id}] YOLO-Pro training complete — "
        f"{_ckpt_tag}  ckpt_criterion={best_ckpt_metric_name}  "
        f"actual_epochs={actual_epochs}/{effective_epochs}  "
        f"total_steps={total_steps}  spe={steps_per_epoch}"
        f"{_tiny_tag}{_exp_tag}{_bs_tag}"
    )
    _eval_map50 = eval_report.get("map50")
    _eval_map_coco = eval_report.get("map")
    _eval_prec = eval_report.get("precision")
    _eval_rec = eval_report.get("recall")
    _eval_summary = "Final metric summary"
    if _eval_map50 is not None:
        _eval_summary += f" - mAP50: {float(_eval_map50):.4f}"
    if _eval_map_coco is not None:
        _eval_summary += f" - mAP50-95: {float(_eval_map_coco):.4f}"
    if _eval_prec is not None:
        _eval_summary += f" - precision: {float(_eval_prec):.4f}"
    if _eval_rec is not None:
        _eval_summary += f" - recall: {float(_eval_rec):.4f}"
    logger.info(f"[{job_id}] {_eval_summary}")
    _yolo_log_lines.append(_eval_summary)
    _yolo_log_lines.append(
        "YOLO-Pro training complete"
        f" - {_ckpt_tag}"
        f" - actual_epochs={actual_epochs}/{effective_epochs}"
        f" - total_steps={total_steps}"
        f" - steps_per_epoch={steps_per_epoch}"
    )
    job.training_history["log_lines"] = _yolo_log_lines
    job.training_history = _sanitize_json(job.training_history)
    try:
        db.commit()
    except Exception:
        db.rollback()
        raise
    return {
        "job_id":               job_id,
        "best_loss":            reported_best,
        "is_yolo_pro":          True,
        "checkpoint_criterion": best_ckpt_metric_name,
        "tiny_dataset_mode":    policy["tiny_dataset_mode"],
        "effective_epochs":     effective_epochs,
        "requested_epochs":     _requested_epochs,
        "total_steps":          total_steps,
        "steps_per_epoch":      steps_per_epoch,
        "effective_batch_size": policy["eff_batch_size"],
    }
