"""
app/services/model_testing_scoring.py
──────────────────────────────────────
Production-grade scoring for the model testing pipeline.

Classification
──────────────
  • Per-class TP, FP, FN counted from (predicted_class, expected_class) pairs.
  • Confusion matrix:  N_classes × N_classes integer grid.
  • Per-class precision = TP / (TP + FP)
              recall    = TP / (TP + FN)
              F1        = 2 * P * R / (P + R)
  • Macro-averaged P / R / F1 across all classes that appear in the test set.

Detection
─────────
  • Per-image IoU matching at a configurable threshold (default 0.50).
  • Accumulates (score, matched_bool) pairs per class across all images.
  • 101-point interpolated AP (VOC2010 / COCO style) per class.
  • mAP50   = mean AP @ IoU 0.50
  • mAP75   = mean AP @ IoU 0.75
  • mAP     = mean AP @ IoU 0.50:0.05:0.95  (10 thresholds, COCO primary)
  • Macro-averaged precision / recall at the best F1 operating point per class.

FOMO (detection_heatmap)
────────────────────────
  • Cell-based exact (row, col, class) matching — no box IoU, no COCO mAP.
  • Accumulates per-class TP / FP / FN arrays across all images.
  • Per-class precision / recall / F1.
  • Macro-averaged precision / recall / F1.
  • Mirrors training_worker._evaluate_fomo_detection semantics exactly.

All three modules expose a single function:
  compute_classification_metrics(results, label_names) → List[MetricDict]
  compute_detection_metrics(det_results, label_names)  → List[MetricDict]
  compute_fomo_metrics(fomo_results, label_names)      → List[MetricDict]

where MetricDict = {"metric_name", "metric_display_name", "metric_value"}.
"""
from __future__ import annotations

from typing import List, Tuple, Dict, Optional
import numpy as np
from app.ml.fomo_evaluator import aggregate_metrics as _fomo_aggregate_metrics


# ─── Type aliases ─────────────────────────────────────────────────────────────

# (predicted_class_name, expected_class_name, confidence_score_0_1)
ClassificationResult = Tuple[str, str, float]

# (pred_boxes, pred_scores, pred_class_ids, gt_boxes_by_class)
# pred_boxes        : List[[x1,y1,x2,y2]] normalised
# pred_scores       : List[float]
# pred_class_ids    : List[int]
# gt_boxes_by_class : {class_idx: [[x1,y1,x2,y2], ...]}
DetectionImageResult = Tuple[
    List[List[float]],               # pred_boxes
    List[float],                     # pred_scores
    List[int],                       # pred_class_ids
    Dict[int, List[List[float]]],    # gt_boxes_by_class
]

# (tp_arr, fp_arr, fn_arr) — each np.ndarray of shape (n_classes,) int64
# Produced per image by the FOMO inference path in model_testing_service.py
FomoImageResult = Tuple[np.ndarray, np.ndarray, np.ndarray]

MetricDict = Dict[str, object]


# ─── IoU helper ───────────────────────────────────────────────────────────────

def _iou_matrix(pred_boxes: np.ndarray, gt_boxes: np.ndarray) -> np.ndarray:
    """
    Vectorised IoU between P predicted boxes and G GT boxes.

    Args:
        pred_boxes : (P, 4) float32  [x1, y1, x2, y2] normalised
        gt_boxes   : (G, 4) float32  [x1, y1, x2, y2] normalised
    Returns:
        (P, G) float32 IoU matrix
    """
    if len(pred_boxes) == 0 or len(gt_boxes) == 0:
        return np.zeros((len(pred_boxes), len(gt_boxes)), dtype=np.float32)

    px1, py1, px2, py2 = pred_boxes[:, 0], pred_boxes[:, 1], pred_boxes[:, 2], pred_boxes[:, 3]
    gx1, gy1, gx2, gy2 = gt_boxes[:, 0],  gt_boxes[:, 1],  gt_boxes[:, 2],  gt_boxes[:, 3]

    ix1 = np.maximum(px1[:, None], gx1[None, :])
    iy1 = np.maximum(py1[:, None], gy1[None, :])
    ix2 = np.minimum(px2[:, None], gx2[None, :])
    iy2 = np.minimum(py2[:, None], gy2[None, :])

    inter = np.maximum(0.0, ix2 - ix1) * np.maximum(0.0, iy2 - iy1)
    a_area = np.maximum(0.0, px2 - px1) * np.maximum(0.0, py2 - py1)
    g_area = np.maximum(0.0, gx2 - gx1) * np.maximum(0.0, gy2 - gy1)
    union  = a_area[:, None] + g_area[None, :] - inter
    return np.where(union > 0, inter / union, 0.0).astype(np.float32)


# ─── AP computation ───────────────────────────────────────────────────────────

def _compute_ap(scores: np.ndarray, matches: np.ndarray, n_gt: int) -> float:
    """
    101-point interpolated Average Precision (VOC2010 / COCO style).

    Args:
        scores  : (N,) float32  confidence scores, sorted descending by caller
        matches : (N,) bool     whether each prediction is a true positive
        n_gt    : int           total number of GT objects for this class
    Returns:
        AP in [0, 1]
    """
    if n_gt == 0 or len(scores) == 0:
        return 0.0

    tp_cum = np.cumsum(matches.astype(np.float32))
    fp_cum = np.cumsum((~matches).astype(np.float32))

    precision = tp_cum / (tp_cum + fp_cum + 1e-9)
    recall    = tp_cum / (n_gt + 1e-9)

    # Append sentinel points so the curve starts at recall=0
    precision = np.concatenate([[1.0], precision, [0.0]])
    recall    = np.concatenate([[0.0], recall,    [recall[-1]]])

    # Make precision monotonically decreasing from right
    for i in range(len(precision) - 2, -1, -1):
        precision[i] = max(precision[i], precision[i + 1])

    ap = 0.0
    for t in np.linspace(0.0, 1.0, 101):
        mask = recall >= t
        ap  += precision[mask].max() if mask.any() else 0.0
    return float(ap / 101.0)


# ═══════════════════════════════════════════════════════════════════════════════
# Classification metrics
# ═══════════════════════════════════════════════════════════════════════════════

def compute_classification_metrics(
    results: List[ClassificationResult],
    label_names: List[str],
) -> List[MetricDict]:
    """
    Compute per-class and macro-averaged classification metrics from
    (predicted_label, expected_label, confidence) triples.

    Also computes the full N×N confusion matrix stored as a flat list of dicts
    for easy JSON serialisation.

    Returns a list of MetricDict entries covering:
      accuracy, macro_precision, macro_recall, macro_f1,
      per_class/{label}/precision, per_class/{label}/recall, per_class/{label}/f1,
      confusion_matrix (serialised as a JSON-safe structure)
    """
    if not results or not label_names:
        return []

    n = len(label_names)
    label_to_idx = {name: i for i, name in enumerate(label_names)}

    # N×N confusion matrix  (row = expected / true, col = predicted)
    cm = np.zeros((n, n), dtype=np.int32)

    for pred_label, exp_label, _ in results:
        pred_idx = label_to_idx.get(pred_label.strip())
        exp_idx  = label_to_idx.get(exp_label.strip())
        if pred_idx is not None and exp_idx is not None:
            cm[exp_idx, pred_idx] += 1

    # Per-class TP / FP / FN
    tp = np.diag(cm).astype(np.float32)
    fp = cm.sum(axis=0).astype(np.float32) - tp   # col sum − diagonal
    fn = cm.sum(axis=1).astype(np.float32) - tp   # row sum − diagonal

    per_class_precision = np.divide(
        tp,
        tp + fp,
        out=np.zeros_like(tp, dtype=np.float32),
        where=(tp + fp) > 0,
    )
    per_class_recall = np.divide(
        tp,
        tp + fn,
        out=np.zeros_like(tp, dtype=np.float32),
        where=(tp + fn) > 0,
    )
    denom = per_class_precision + per_class_recall
    per_class_f1 = np.divide(
        2 * per_class_precision * per_class_recall,
        denom,
        out=np.zeros_like(denom, dtype=np.float32),
        where=denom > 0,
    )
    row_totals = cm.sum(axis=1).astype(np.float32)
    per_class_accuracy = np.divide(
        tp,
        row_totals,
        out=np.zeros_like(tp, dtype=np.float32),
        where=row_totals > 0,
    )

    # Classes that actually appear in the test set (has GT or predictions)
    active = (cm.sum(axis=1) + cm.sum(axis=0)) > 0
    n_active = int(active.sum())

    macro_precision = float(per_class_precision[active].mean()) if n_active else 0.0
    macro_recall    = float(per_class_recall[active].mean())    if n_active else 0.0
    denom_macro     = macro_precision + macro_recall
    macro_f1        = (2 * macro_precision * macro_recall / denom_macro
                       if denom_macro > 0 else 0.0)

    total_correct = int(tp.sum())
    total_samples = int(cm.sum())
    accuracy      = total_correct / total_samples if total_samples > 0 else 0.0

    metrics: List[MetricDict] = [
        _m("accuracy",         "Accuracy",                     round(accuracy,        4)),
        _m("macro_precision",  "Precision (macro avg)",        round(macro_precision, 4)),
        _m("macro_recall",     "Recall (macro avg)",           round(macro_recall,    4)),
        _m("macro_f1",         "F1 Score (macro avg)",         round(macro_f1,        4)),
        # Keep legacy names so existing API consumers don't break
        _m("precision_non_background", "Precision (non-background)", round(macro_precision, 4)),
        _m("recall_non_background",    "Recall (non-background)",    round(macro_recall,    4)),
        _m("f1_score_non_background",  "F1 Score (non-background)",  round(macro_f1,        4)),
    ]

    # Per-class breakdown
    for i, name in enumerate(label_names):
        if not active[i]:
            continue
        safe = name.replace(" ", "_").replace("/", "_")
        metrics += [
            _m(f"per_class/{safe}/precision", f"{name} — precision",
               round(float(per_class_precision[i]), 4)),
            _m(f"per_class/{safe}/recall",    f"{name} — recall",
               round(float(per_class_recall[i]),    4)),
            _m(f"per_class/{safe}/f1",        f"{name} — F1",
               round(float(per_class_f1[i]),        4)),
            _m(f"per_class/{safe}/accuracy", f"{name} — accuracy",
               round(float(per_class_accuracy[i]), 4)),
        ]

    # Confusion matrix as JSON-safe flat list of {true, predicted, count}
    cm_entries = []
    for r in range(n):
        for c in range(n):
            if cm[r, c] > 0:
                cm_entries.append({
                    "true":      label_names[r],
                    "predicted": label_names[c],
                    "count":     int(cm[r, c]),
                })
    metrics.append(_m("confusion_matrix", "Confusion matrix", cm_entries))

    return metrics


# ═══════════════════════════════════════════════════════════════════════════════
# Detection metrics
# ═══════════════════════════════════════════════════════════════════════════════

_COCO_IOU_THRESHOLDS = np.linspace(0.50, 0.95, 10)   # [0.50, 0.55, …, 0.95]


def compute_detection_metrics(
    image_results: List[DetectionImageResult],
    label_names: List[str],
    iou_threshold_primary: float = 0.50,
) -> List[MetricDict]:
    """
    Compute COCO-style detection metrics from per-image detection results.

    Args:
        image_results : list of (pred_boxes, pred_scores, pred_class_ids, gt_by_class)
                        One entry per test image.
        label_names   : ordered class names (index must match pred_class_ids)
        iou_threshold_primary : IoU threshold for mAP50 and per-class AP (default 0.50)

    Returns list of MetricDict covering:
      mAP50, mAP75, mAP (COCO 0.50:0.95),
      per-class AP50, precision, recall,
      macro precision, macro recall
    """
    n_classes = len(label_names)
    if n_classes == 0 or not image_results:
        return []

    # Accumulators: one list per (iou_threshold, class)
    # all_preds_multi[iou_idx][class_idx] = [(score, matched), ...]
    all_preds_multi = [
        [[] for _ in range(n_classes)]
        for _ in _COCO_IOU_THRESHOLDS
    ]
    gt_count = np.zeros(n_classes, dtype=np.int32)

    for pred_boxes, pred_scores, pred_class_ids, gt_by_class in image_results:
        # Count GT
        for cls_idx, gt_list in gt_by_class.items():
            if 0 <= cls_idx < n_classes:
                gt_count[cls_idx] += len(gt_list)

        if not pred_boxes:
            continue

        pred_arr  = np.array(pred_boxes,   dtype=np.float32)   # (P, 4)
        score_arr = np.array(pred_scores,  dtype=np.float32)   # (P,)
        cls_arr   = np.array(pred_class_ids, dtype=np.int32)    # (P,)

        # Sort by descending score (required for AP)
        order = np.argsort(-score_arr)
        pred_arr  = pred_arr[order]
        score_arr = score_arr[order]
        cls_arr   = cls_arr[order]

        for iou_idx, iou_thr in enumerate(_COCO_IOU_THRESHOLDS):
            # Per-GT matched flag (reset per image per IoU threshold)
            gt_used: Dict[int, set] = {c: set() for c in gt_by_class}

            for det_i in range(len(pred_arr)):
                c     = int(cls_arr[det_i])
                score = float(score_arr[det_i])
                if c < 0 or c >= n_classes:
                    continue

                gt_list = gt_by_class.get(c, [])
                matched = False

                if gt_list:
                    gt_arr  = np.array(gt_list, dtype=np.float32)   # (G, 4)
                    pb      = pred_arr[det_i:det_i+1]                # (1, 4)
                    ious    = _iou_matrix(pb, gt_arr)[0]             # (G,)
                    best_gi = int(np.argmax(ious))
                    if ious[best_gi] >= iou_thr and best_gi not in gt_used.get(c, set()):
                        matched = True
                        gt_used.setdefault(c, set()).add(best_gi)

                all_preds_multi[iou_idx][c].append((score, matched))

    # ── Per-class AP at each IoU threshold ────────────────────────────────────
    ap_per_iou = np.zeros((len(_COCO_IOU_THRESHOLDS), n_classes), dtype=np.float32)

    for iou_idx in range(len(_COCO_IOU_THRESHOLDS)):
        for c in range(n_classes):
            preds_c = sorted(all_preds_multi[iou_idx][c], key=lambda x: x[0], reverse=True)
            if not preds_c or gt_count[c] == 0:
                continue
            sc = np.array([p[0] for p in preds_c], dtype=np.float32)
            mt = np.array([p[1] for p in preds_c], dtype=bool)
            ap_per_iou[iou_idx, c] = _compute_ap(sc, mt, int(gt_count[c]))

    # Classes that have at least one GT
    has_gt = gt_count > 0

    def _mean_ap(iou_indices) -> Optional[float]:
        aps = ap_per_iou[np.ix_(list(iou_indices), np.where(has_gt)[0])]
        return float(aps.mean()) if aps.size > 0 else None

    # Primary IoU threshold index
    primary_idx = int(np.argmin(np.abs(_COCO_IOU_THRESHOLDS - iou_threshold_primary)))
    idx75       = int(np.argmin(np.abs(_COCO_IOU_THRESHOLDS - 0.75)))

    map50  = _mean_ap([primary_idx])
    map75  = _mean_ap([idx75])
    map_coco = _mean_ap(range(len(_COCO_IOU_THRESHOLDS)))

    # Per-class metrics at primary IoU
    per_class_ap   = {}
    per_class_prec = {}
    per_class_rec  = {}

    for c, name in enumerate(label_names):
        if not has_gt[c]:
            continue
        preds_c = sorted(all_preds_multi[primary_idx][c], key=lambda x: x[0], reverse=True)
        if not preds_c:
            per_class_ap[name] = 0.0
            per_class_prec[name] = 0.0
            per_class_rec[name]  = 0.0
            continue

        sc = np.array([p[0] for p in preds_c], dtype=np.float32)
        mt = np.array([p[1] for p in preds_c], dtype=bool)

        per_class_ap[name] = round(_compute_ap(sc, mt, int(gt_count[c])), 4)

        tp_cum  = np.cumsum(mt.astype(np.float32))
        fp_cum  = np.cumsum((~mt).astype(np.float32))
        rec_arr = tp_cum / (int(gt_count[c]) + 1e-9)
        pre_arr = tp_cum / (tp_cum + fp_cum + 1e-9)
        f1_arr  = 2 * pre_arr * rec_arr / (pre_arr + rec_arr + 1e-9)
        best_i  = int(np.argmax(f1_arr)) if len(f1_arr) > 0 else 0

        per_class_prec[name] = round(float(pre_arr[best_i]), 4) if len(pre_arr) else 0.0
        per_class_rec[name]  = round(float(rec_arr[best_i]),  4) if len(rec_arr) else 0.0

    macro_prec = float(np.mean(list(per_class_prec.values()))) if per_class_prec else 0.0
    macro_rec  = float(np.mean(list(per_class_rec.values())))  if per_class_rec  else 0.0

    metrics: List[MetricDict] = [
        _m("map50",            "mAP @ IoU 0.50",              round(map50,      4) if map50      is not None else None),
        _m("map75",            "mAP @ IoU 0.75",              round(map75,      4) if map75      is not None else None),
        _m("map",              "mAP @ IoU 0.50:0.95 (COCO)",  round(map_coco,   4) if map_coco   is not None else None),
        _m("macro_precision",  "Precision (macro avg)",        round(macro_prec, 4)),
        _m("macro_recall",     "Recall (macro avg)",           round(macro_rec,  4)),
        # Legacy names
        _m("precision_non_background", "Precision (non-background)", round(macro_prec, 4)),
        _m("recall_non_background",    "Recall (non-background)",    round(macro_rec,  4)),
    ]

    # ── Background false-positive rate ────────────────────────────────────────
    # mAP is near-blind to negatives by construction: an image with no GT
    # contributes no AP term at all, so detections fired on it barely move the
    # number.  Counted over images whose gt_by_class holds no boxes.
    _bg = [
        (pb, ps) for pb, ps, _pc, gt_by_class in image_results
        if not any(gt_by_class.get(c) for c in gt_by_class)
    ]
    if _bg:
        _bg_fp_total = int(sum(len(pb) for pb, _ps in _bg))
        _bg_with_fp  = int(sum(1 for pb, _ps in _bg if pb))
        metrics += [
            _m("background_images",  "Background images", len(_bg)),
            _m("background_fp_rate", "Background FP rate (per image)",
               round(_bg_fp_total / len(_bg), 4)),
            _m("background_images_with_fp", "Background images with a false positive",
               _bg_with_fp),
        ]

    for name in label_names:
        if name not in per_class_ap:
            continue
        safe = name.replace(" ", "_").replace("/", "_")
        metrics += [
            _m(f"per_class/{safe}/ap50",      f"{name} — AP50",      per_class_ap[name]),
            _m(f"per_class/{safe}/precision", f"{name} — precision", per_class_prec.get(name, 0.0)),
            _m(f"per_class/{safe}/recall",    f"{name} — recall",    per_class_rec.get(name,  0.0)),
        ]

    return metrics


# ═══════════════════════════════════════════════════════════════════════════════
# FOMO (detection_heatmap) metrics  — cell-based, NOT box-IoU / COCO mAP
# ═══════════════════════════════════════════════════════════════════════════════

def compute_fomo_metrics(
    image_results: List[FomoImageResult],
    label_names: List[str],
) -> List[MetricDict]:
    """
    Aggregate cell-based FOMO metrics from per-image (tp, fp, fn) arrays.

    Mirrors training_worker._evaluate_fomo_detection semantics:
      - exact (row, col, class) cell matching — no IoU, no box overlap
      - per-class precision / recall / F1
      - macro-averaged precision / recall / F1

    Args:
        image_results : list of (tp_arr, fp_arr, fn_arr), one per test image.
                        Each array has shape (n_classes,) and dtype int64.
                        This is exactly the 3-tuple the FOMO inference path
                        appends to det_image_results in model_testing_service.py.
        label_names   : ordered class names (index matches array positions).

    Returns list of MetricDict entries covering:
      macro_precision, macro_recall, macro_f1,
      per_class/{label}/precision, /recall, /f1,
      plus legacy aliases precision_non_background / recall_non_background /
      f1_score_non_background so existing API consumers are not broken.
    """
    n_classes = len(label_names)
    if n_classes == 0 or not image_results:
        return []

    # Accumulate TP / FP / FN across all images
    tp_total = np.zeros(n_classes, dtype=np.int64)
    fp_total = np.zeros(n_classes, dtype=np.int64)
    fn_total = np.zeros(n_classes, dtype=np.int64)

    for tp_arr, fp_arr, fn_arr in image_results:
        tp_arr = np.asarray(tp_arr, dtype=np.int64)
        fp_arr = np.asarray(fp_arr, dtype=np.int64)
        fn_arr = np.asarray(fn_arr, dtype=np.int64)
        if tp_arr.shape == (n_classes,):
            tp_total += tp_arr
            fp_total += fp_arr
            fn_total += fn_arr

    # Delegate to shared contract — identical arithmetic as training_worker
    agg = _fomo_aggregate_metrics(tp_total, fp_total, fn_total, label_names)

    macro_precision = agg["macro_avg"]["precision"]
    macro_recall    = agg["macro_avg"]["recall"]
    macro_f1        = agg["macro_avg"]["f1-score"]

    metrics: List[MetricDict] = [
        _m("macro_precision",          "Precision (macro avg)",      macro_precision),
        _m("macro_recall",             "Recall (macro avg)",         macro_recall),
        _m("macro_f1",                 "F1 Score (macro avg)",       macro_f1),
        # Legacy aliases — keep existing API consumers working
        _m("precision_non_background", "Precision (non-background)", macro_precision),
        _m("recall_non_background",    "Recall (non-background)",    macro_recall),
        _m("f1_score_non_background",  "F1 Score (non-background)",  macro_f1),
    ]

    # ── Background false-positive rate ────────────────────────────────────────
    # Macro precision/recall are near-blind to negatives: a background image
    # contributes only to the shared FP tally, where it is indistinguishable
    # from a mislocalised detection on a positive image. Without this metric a
    # user who added background images gets no feedback on whether it helped.
    #
    # A background image is one with no ground truth at all, which shows up as
    # tp + fn == 0 across every class (match_cells routes unmatched GT to fn, so
    # any GT present makes this sum non-zero). No extra plumbing needed.
    metrics += _background_fp_metrics([
        (np.asarray(tp, dtype=np.int64), np.asarray(fp, dtype=np.int64), np.asarray(fn, dtype=np.int64))
        for tp, fp, fn in image_results
    ], n_classes)

    for name, pc in agg["per_class"].items():
        # Only emit per-class metrics for active classes
        if pc["tp"] == 0 and pc["fp"] == 0 and pc["fn"] == 0:
            continue
        safe = name.replace(" ", "_").replace("/", "_")
        metrics += [
            _m(f"per_class/{safe}/precision", f"{name} — precision", pc["precision"]),
            _m(f"per_class/{safe}/recall",    f"{name} — recall",    pc["recall"]),
            _m(f"per_class/{safe}/f1",        f"{name} — F1",        pc["f1-score"]),
        ]

    return metrics


# ─── Internal helpers ─────────────────────────────────────────────────────────

def _background_fp_metrics(
    image_results: List[FomoImageResult],
    n_classes: int,
) -> List[MetricDict]:
    """Emit background-image FP metrics, or nothing if there are no negatives."""
    bg = [
        (tp, fp, fn) for tp, fp, fn in image_results
        if tp.shape == (n_classes,) and int(tp.sum() + fn.sum()) == 0
    ]
    if not bg:
        return []
    total_fp = int(sum(int(fp.sum()) for _tp, fp, _fn in bg))
    with_fp  = int(sum(1 for _tp, fp, _fn in bg if int(fp.sum()) > 0))
    return [
        _m("background_images",  "Background images",
           len(bg)),
        _m("background_fp_rate", "Background FP rate (per image)",
           round(total_fp / len(bg), 4)),
        _m("background_images_with_fp", "Background images with a false positive",
           with_fp),
    ]


def _m(name: str, display: str, value) -> MetricDict:
    return {
        "metric_name":         name,
        "metric_display_name": display,
        "metric_value":        value,
    }