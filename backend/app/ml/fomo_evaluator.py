"""
fomo_evaluator.py
=================
Single shared contract for FOMO (detection_heatmap) evaluation.

Both training_worker and model_testing_service must import from here.
No FOMO scoring logic lives anywhere else.

Contract
--------
* GT:          argmax over channels of the GT heatmap cell; channel > 0 is an
               occupied cell; class index = channel - 1 (zero-indexed).
* Prediction:  softmax(raw_output) per cell; cell fires when argmax > 0
               AND prob[argmax] > threshold; class index = argmax - 1.
* Match:       exact (row, col, class) triple — no IoU, no COCO mAP.
* Softmax:     applied exactly once.  If the tensor already looks like
               probabilities (all values in [0,1], rows sum ≈ 1) softmax
               is skipped.
* Threshold:   resolved by resolve_fomo_threshold(); no hard-coded 0.5 / 0.01
               inside the scoring path.
"""
from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ── Types ──────────────────────────────────────────────────────────────────────
# {(row, col): zero_indexed_class}
CellMap = Dict[Tuple[int, int], int]

FOMO_SWEEP_THRESHOLDS: List[float] = [
    0.01, 0.02, 0.03, 0.05, 0.10, 0.15, 0.20, 0.25,
    0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70,
    0.75, 0.80, 0.85, 0.90, 0.95, 0.99,
]


# ── 1. Softmax / decode ────────────────────────────────────────────────────────

def _softmax(arr: np.ndarray) -> np.ndarray:
    """Numerically-stable softmax over the last axis."""
    shifted = arr - arr.max(axis=-1, keepdims=True)
    e = np.exp(shifted)
    return e / (e.sum(axis=-1, keepdims=True) + 1e-12)


def _looks_like_probabilities(arr: np.ndarray, tol: float = 1e-3) -> bool:
    """Return True when the (H, W, C) tensor is already a probability map."""
    if arr.ndim != 3:
        return False
    if np.any(arr < -tol) or np.any(arr > 1.0 + tol):
        return False
    sums = arr.sum(axis=-1)
    return bool(
        np.all(np.isfinite(sums)) and np.all(np.abs(sums - 1.0) <= 5 * tol)
    )


def decode_fomo_heatmap(
    raw_output: np.ndarray,
    threshold: float,
    min_peak_gap: float = 0.0,
    fomo_version: int = 1,
    grid_size: int = 0,
) -> Tuple[CellMap, np.ndarray]:
    """Decode one FOMO output tensor (H, W, C+1) into predicted cells.

    Steps:
      1. Conditionally softmax (skipped when input is already probabilities).
      2. Per-cell argmax over all C+1 channels (background ch=0 included).
      3. Cell fires when pred_ch > 0 AND prob[pred_ch] > threshold.
      4. Ignore background channel — never returned in CellMap.
      5. Bounds are implicit: loops stay within array dimensions.

    fomo_version / grid_size are informational — the decode algorithm is
    identical for v1 and v2 because both produce (H, W, C+1) tensors.
    v1 path is unchanged at runtime; v2 path adds no new logic here.

    Args:
        raw_output:   (H, W, C+1) float array — raw logits or probabilities.
        threshold:    float in (0, 1); must be finite (caller must validate).
        fomo_version: 1 (Legacy 96×96) or 2 (Adaptive Resolution). Passed
                      through from model metadata for logging/validation.
        grid_size:    stored grid dimension from metadata (informational only).
        min_peak_gap: cell survives only if all 8-neighbours are ≥ min_peak_gap below it.
                      Gap=0.0 matches original strictly-greater-than behaviour.

    Returns:
        pred_cells: {(row, col): zero_indexed_class}
        probs:      (H, W, C+1) probability array after conditional softmax.
    """
    raw_output = np.asarray(raw_output, dtype=np.float32)
    probs: np.ndarray = (
        raw_output
        if _looks_like_probabilities(raw_output)
        else _softmax(raw_output)
    )

    fH, fW = probs.shape[:2]
    # Collect all candidate cells before suppression.
    candidates: dict = {}  # (r,c) -> (zero_indexed_class, confidence)
    for r in range(fH):
        for c in range(fW):
            cell = probs[r, c]
            pred_ch = int(np.argmax(cell))
            conf = float(cell[pred_ch])
            if pred_ch > 0 and conf > threshold:
                candidates[(r, c)] = (pred_ch - 1, conf)

    # Local-max NMS: suppress any cell that has an 8-connected neighbour with
    # strictly higher object-class confidence.  8-connectivity catches diagonal
    # BG peaks that survive 4-connected NMS when their confidence ties or
    # slightly trails an orthogonal neighbour but beats a diagonal one.
    pred_cells: CellMap = {}
    for (r, c), (cls, conf) in candidates.items():
        is_local_max = True
        for dr, dc in (
            (-1, 0), (1, 0), (0, -1), (0, 1),
            (-1, -1), (-1, 1), (1, -1), (1, 1),
        ):
            nb = (r + dr, c + dc)
            if nb in candidates and candidates[nb][1] > conf - min_peak_gap:
                is_local_max = False
                break
        if is_local_max:
            pred_cells[(r, c)] = cls
    return pred_cells, probs


def gt_cells_from_heatmap(heatmap: np.ndarray) -> CellMap:
    """Extract occupied GT cells from a pre-built (H, W, C+1) heatmap.

    Used in training_worker where GT is already in heatmap form.
    Channel 0 = background; class index = channel - 1.
    """
    grid_h, grid_w = heatmap.shape[:2]
    cells: CellMap = {}
    for r in range(grid_h):
        for c in range(grid_w):
            cls_ch = int(np.argmax(heatmap[r, c]))
            if cls_ch > 0:
                cells[(r, c)] = cls_ch - 1
    return cells


def gt_cells_from_boxes(
    norm_gt: list,
    fH: int,
    fW: int,
    label_to_idx: Dict[str, int],
    sample_id: Optional[str] = None,
) -> Tuple[CellMap, int]:
    """Map normalised GT bounding boxes → occupied grid cells.

    Uses center-point of each box, exactly like training_worker.

    Args:
        norm_gt:       list of normalised box dicts with keys
                       x, y, w, h, label_id (UUID), label (name).
        fH, fW:        grid dimensions.
        label_to_idx:  dual-key dict keyed by both UUID and name string
                       (built by resolve_label_to_idx).
        sample_id:     used in warning messages only.

    Returns:
        (gt_cells, n_unresolved)
    """
    gt_cells: CellMap = {}
    n_unresolved = 0
    for b in norm_gt:
        # Try label_id (UUID) first, then fall back to label name string.
        lid = b.get("label_id")
        raw_key = lid if lid is not None else (b.get("label") or "")
        cls = label_to_idx.get(str(raw_key).strip())

        # Last-resort: numeric index passed as string / int
        if cls is None:
            try:
                int_key = int(raw_key)
                if 0 <= int_key < len(label_to_idx):
                    cls = int_key
            except (ValueError, TypeError):
                pass

        if cls is None:
            n_unresolved += 1
            logger.warning(
                "[FOMO gt_cells] unresolved GT label — "
                "sample_id=%s label_id=%r label=%r raw_payload=%r",
                sample_id,
                b.get("label_id"),
                b.get("label"),
                b,
            )
            continue

        cx = float(b.get("x", 0)) + float(b.get("w", 0)) / 2.0
        cy = float(b.get("y", 0)) + float(b.get("h", 0)) / 2.0
        r = min(int(cy * fH), fH - 1)
        c = min(int(cx * fW), fW - 1)
        gt_cells[(r, c)] = cls

    return gt_cells, n_unresolved


# ── 2. Threshold resolver ──────────────────────────────────────────────────────

class ThresholdSource:
    TRAINING_PARITY  = "training_parity"   # classification_report["threshold"]
    DEPLOYED_MODEL   = "deployed_model"    # sweep over actual exported model


def resolve_fomo_threshold(
    *,
    mode: str,
    training_classification_report: Optional[dict] = None,
    sweep_items: Optional[List[Tuple[np.ndarray, CellMap]]] = None,
    label_names: Optional[List[str]] = None,
    preferred_threshold: Optional[float] = None,
) -> Tuple[float, str]:
    """Single threshold resolver for all FOMO evaluation paths.

    Modes
    -----
    ThresholdSource.TRAINING_PARITY
        Read `training_classification_report["threshold"]` — the value written
        by _evaluate_fomo_detection via threshold sweep at train time.
        Call this from:  model_testing batch path (classify-all)
                         model_testing single-sample path
        `training_classification_report` must be provided and contain "threshold".
        Raises ValueError if missing (fail loudly, never fall back silently).

    ThresholdSource.DEPLOYED_MODEL
        Run a threshold sweep over the actual exported model outputs
        (`sweep_items` must be provided) and pick the best macro-F1 threshold.
        Tie-break: prefer the lowest threshold.
        Call this from:  model_testing batch path when the operator explicitly
                         wants per-exported-model calibration.

    Returns
        (threshold, source_description_string)
    """
    if mode == ThresholdSource.TRAINING_PARITY:
        if training_classification_report is None:
            raise ValueError(
                "resolve_fomo_threshold(mode=TRAINING_PARITY) requires "
                "training_classification_report to be provided."
            )
        if "threshold" not in training_classification_report:
            raise ValueError(
                "resolve_fomo_threshold(mode=TRAINING_PARITY): "
                "training_classification_report has no 'threshold' key. "
                "Ensure the training job has completed with _evaluate_fomo_detection."
            )
        thr = float(training_classification_report["threshold"])
        _validate_threshold(thr)
        source = f"{ThresholdSource.TRAINING_PARITY} (training_job classification_report)"
        logger.info("[FOMO threshold] mode=%s  threshold=%.4f  source=%s", mode, thr, source)
        return thr, source

    if mode == ThresholdSource.DEPLOYED_MODEL:
        if not sweep_items or not label_names:
            raise ValueError(
                "resolve_fomo_threshold(mode=DEPLOYED_MODEL) requires "
                "sweep_items and label_names."
            )
        thr = _sweep_best_threshold(
            sweep_items,
            label_names,
            preferred_threshold=preferred_threshold,
        )
        _validate_threshold(thr)
        source = f"{ThresholdSource.DEPLOYED_MODEL} (sweep over exported model outputs)"
        logger.info("[FOMO threshold] mode=%s  threshold=%.4f  source=%s", mode, thr, source)
        return thr, source

    raise ValueError(f"resolve_fomo_threshold: unknown mode {mode!r}. "
                     f"Use ThresholdSource.TRAINING_PARITY or .DEPLOYED_MODEL.")


def _validate_threshold(thr: float) -> None:
    if not math.isfinite(thr) or not (0.0 < thr < 1.0):
        raise ValueError(
            f"FOMO threshold must be a finite float in (0, 1), got {thr!r}."
        )


def _sweep_best_threshold(
    items: List[Tuple[np.ndarray, CellMap]],
    label_names: List[str],
    preferred_threshold: Optional[float] = None,
) -> float:
    """Internal: find the threshold with the best macro-F1 over sweep_items."""
    n_classes = len(label_names)
    candidates = sorted(
        set(
            ([float(preferred_threshold)] if preferred_threshold else [])
            + FOMO_SWEEP_THRESHOLDS
        )
    )
    best_threshold = candidates[0]
    best_f1 = -1.0

    for thr in candidates:
        tp = np.zeros(n_classes, dtype=np.int64)
        fp = np.zeros(n_classes, dtype=np.int64)
        fn = np.zeros(n_classes, dtype=np.int64)
        for probs, gt_cells in items:
            pc, _ = decode_fomo_heatmap(probs, thr, min_peak_gap=0.0)
            _tp, _fp, _fn = match_cells(pc, gt_cells, n_classes)
            tp += _tp; fp += _fp; fn += _fn

        macro_f1 = _macro_f1_from_arrays(tp, fp, fn, n_classes)
        if macro_f1 > best_f1 + 1e-9 or (
            abs(macro_f1 - best_f1) <= 1e-3 and thr < best_threshold
        ):
            best_f1 = macro_f1
            best_threshold = thr

    return best_threshold


# ── 3. Cell matching ───────────────────────────────────────────────────────────

def match_cells(
    pred_cells: CellMap,
    gt_cells: CellMap,
    n_classes: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Exact (row, col, class) matching → per-class TP / FP / FN arrays.

    This is the canonical matching function.  Both training_worker and
    model_testing_service call it (or call score_sample which wraps it).
    """
    tp = np.zeros(n_classes, dtype=np.int64)
    fp = np.zeros(n_classes, dtype=np.int64)
    fn = np.zeros(n_classes, dtype=np.int64)

    for (r, c), gt_cls in gt_cells.items():
        if gt_cls < 0 or gt_cls >= n_classes:
            logger.warning("[FOMO match] GT class index %d out of range [0, %d)", gt_cls, n_classes)
            continue
        if pred_cells.get((r, c)) == gt_cls:
            tp[gt_cls] += 1
        else:
            fn[gt_cls] += 1

    for (r, c), pred_cls in pred_cells.items():
        if pred_cls < 0 or pred_cls >= n_classes:
            logger.warning("[FOMO match] pred class index %d out of range [0, %d)", pred_cls, n_classes)
            continue
        if gt_cells.get((r, c)) != pred_cls:
            fp[pred_cls] += 1

    return tp, fp, fn


def score_sample(
    pred_cells: CellMap,
    gt_cells: CellMap,
    n_classes: int,
    label_names: List[str],
) -> dict:
    """Per-sample score dict for model-testing paths.

    Returns:
        score_0_100, status_pass_fail, pred_cls_name, f1_0_1, tp, fp, fn
    All keys available as dict entries; TP/FP/FN are np.int64 arrays.
    """
    from app.models.model_testing import TestResultStatus  # avoid circular at module level

    tp, fp, fn = match_cells(pred_cells, gt_cells, n_classes)

    total_tp = int(tp.sum())
    total_fp = int(fp.sum())
    total_fn = int(fn.sum())
    prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    rec  = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

    status = (
        (TestResultStatus.pass_ if total_tp > 0 else TestResultStatus.fail)
        if gt_cells
        else (TestResultStatus.pass_ if total_fp == 0 else TestResultStatus.fail)
    )

    pred_cls_name: Optional[str] = None
    if pred_cells:
        vals = list(pred_cells.values())
        most = max(set(vals), key=vals.count)
        pred_cls_name = label_names[most] if most < n_classes else str(most)

    return {
        "score_0_100": round(f1 * 100.0, 2),
        "status":      status,
        "pred_cls":    pred_cls_name,
        "f1":          round(f1, 4),
        "tp":          tp,
        "fp":          fp,
        "fn":          fn,
    }


# ── 4. Aggregate metrics ───────────────────────────────────────────────────────

def aggregate_metrics(
    tp_total: np.ndarray,
    fp_total: np.ndarray,
    fn_total: np.ndarray,
    label_names: List[str],
) -> dict:
    """Compute per-class + macro precision/recall/F1 from accumulated arrays.

    This is the canonical aggregation used by both _evaluate_fomo_detection
    (training_worker) and compute_fomo_metrics (model_testing_scoring).

    Returns a dict with keys:
        per_class: {name: {precision, recall, f1, support, tp, fp, fn}}
        macro_avg: {precision, recall, f1, support}
        weighted_avg: {precision, recall, f1, support}
        macro_f1: float   (convenience shortcut)
    """
    n_classes = len(label_names)
    result: dict = {"per_class": {}}
    total_support = 0
    macro_p = macro_r = macro_f1_sum = 0.0
    weighted_p = weighted_r = weighted_f1 = 0.0

    for i, name in enumerate(label_names):
        support = int(tp_total[i] + fn_total[i])
        prec = float(tp_total[i]) / (tp_total[i] + fp_total[i]) if (tp_total[i] + fp_total[i]) > 0 else 0.0
        rec  = float(tp_total[i]) / (tp_total[i] + fn_total[i]) if (tp_total[i] + fn_total[i]) > 0 else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        result["per_class"][name] = {
            "precision": round(prec, 4),
            "recall":    round(rec,  4),
            "f1-score":  round(f1,   4),
            "support":   support,
            "tp": int(tp_total[i]),
            "fp": int(fp_total[i]),
            "fn": int(fn_total[i]),
        }
        total_support    += support
        macro_p          += prec
        macro_r          += rec
        macro_f1_sum     += f1
        weighted_p       += prec * support
        weighted_r       += rec  * support
        weighted_f1      += f1   * support

    n = max(n_classes, 1)
    result["macro_avg"] = {
        "precision": round(macro_p      / n, 4),
        "recall":    round(macro_r      / n, 4),
        "f1-score":  round(macro_f1_sum / n, 4),
        "support":   total_support,
    }
    result["weighted_avg"] = {
        "precision": round(weighted_p  / total_support, 4) if total_support else 0.0,
        "recall":    round(weighted_r  / total_support, 4) if total_support else 0.0,
        "f1-score":  round(weighted_f1 / total_support, 4) if total_support else 0.0,
        "support":   total_support,
    }
    result["macro_f1"] = result["macro_avg"]["f1-score"]
    return result


def _macro_f1_from_arrays(
    tp: np.ndarray, fp: np.ndarray, fn: np.ndarray, n_classes: int
) -> float:
    """Quick macro-F1 scalar — used inside threshold sweep."""
    f1_sum = 0.0
    for i in range(n_classes):
        prec = float(tp[i]) / (tp[i] + fp[i]) if (tp[i] + fp[i]) > 0 else 0.0
        rec  = float(tp[i]) / (tp[i] + fn[i]) if (tp[i] + fn[i]) > 0 else 0.0
        f1_sum += 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return f1_sum / max(n_classes, 1)


# ── 5. Guards ──────────────────────────────────────────────────────────────────

def assert_grid_shape_match(
    pred_shape: Tuple[int, ...],
    gt_shape: Tuple[int, ...],
    context: str = "",
) -> None:
    """Raise ValueError when prediction and GT grid dimensions differ."""
    if pred_shape[1:3] != gt_shape[1:3]:
        raise ValueError(
            f"[FOMO {context}] Grid shape mismatch: "
            f"prediction {pred_shape} vs ground-truth {gt_shape}. "
            "Training and testing must use the same FOMO grid resolution."
        )


def assert_class_ordering(
    label_names_from_model: List[str],
    label_names_from_gt: List[str],
    context: str = "",
) -> None:
    """Warn loudly when class orderings differ — would produce wrong TP/FP/FN."""
    if label_names_from_model != label_names_from_gt:
        logger.error(
            "[FOMO %s] Class index ordering mismatch! "
            "model label_names=%s  gt label_names=%s  "
            "Metrics will be incorrect until both use the same ordering.",
            context,
            label_names_from_model,
            label_names_from_gt,
        )


# ── 6. Diagnostics ────────────────────────────────────────────────────────────

def log_fomo_run_diagnostics(
    *,
    threshold: float,
    threshold_source: str,
    total_gt_cells: int,
    total_pred_cells: int,
    tp: np.ndarray,
    fp: np.ndarray,
    fn: np.ndarray,
    label_names: List[str],
    n_unresolved_gt: int = 0,
    sample_gt_cells: Optional[List[CellMap]] = None,
    sample_pred_cells: Optional[List[CellMap]] = None,
    context: str = "",
) -> None:
    """Emit a structured INFO log covering all required FOMO diagnostics."""
    n_classes = len(label_names)

    def _cell_count(cells) -> int:
        if cells is None:
            return 0
        if isinstance(cells, dict):
            # Canonical FOMO cell maps are {(row, col): class_idx}, so the
            # number of occupied cells is the dict size. Keep a fallback for
            # any older grouped-dict shapes whose values are sequences.
            if not cells:
                return 0
            first_value = next(iter(cells.values()))
            if isinstance(first_value, (list, tuple, set, dict)):
                return sum(len(v) for v in cells.values())
            return len(cells)
        return len(cells)

    per_class_lines = []
    macro_p = macro_r = macro_f1_sum = 0.0
    for i, name in enumerate(label_names):
        prec = float(tp[i]) / (tp[i] + fp[i]) if (tp[i] + fp[i]) > 0 else 0.0
        rec  = float(tp[i]) / (tp[i] + fn[i]) if (tp[i] + fn[i]) > 0 else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        macro_p += prec; macro_r += rec; macro_f1_sum += f1
        per_class_lines.append(
            f"  {name}: P={prec:.4f} R={rec:.4f} F1={f1:.4f} "
            f"(TP={tp[i]} FP={fp[i]} FN={fn[i]})"
        )

    n = max(n_classes, 1)
    macro_f1 = round(macro_f1_sum / n, 4)

    logger.info(
        "[FOMO eval%s]\n"
        "  threshold=%.4f  source=%s\n"
        "  total_GT_cells=%d  total_pred_cells=%d\n"
        "  TP=%d  FP=%d  FN=%d\n"
        "  macro_P=%.4f  macro_R=%.4f  macro_F1=%.4f\n"
        "  unresolved_GT_labels=%d\n"
        "%s",
        f" [{context}]" if context else "",
        threshold,
        threshold_source,
        total_gt_cells,
        total_pred_cells,
        int(tp.sum()), int(fp.sum()), int(fn.sum()),
        macro_p / n, macro_r / n, macro_f1,
        n_unresolved_gt,
        "\n".join(per_class_lines),
    )

    # First-few sample snapshots — compact counts at INFO, full maps at DEBUG.
    if sample_gt_cells or sample_pred_cells:
        _peek = []
        for _i, (_sg, _sp) in enumerate(zip(
            (sample_gt_cells or [])[:3],
            (sample_pred_cells or [])[:3],
        )):
            _gt_n  = _cell_count(_sg)
            _pr_n  = _cell_count(_sp)
            _peek.append(f"s{_i}: GT={_gt_n} pred={_pr_n}")
        logger.info(
            "[FOMO eval%s] sample peek (GT/pred counts): %s",
            f" [{context}]" if context else "",
            "  ".join(_peek),
        )
        logger.debug(
            "[FOMO eval%s] first GT cell maps: %s  first pred cell maps: %s",
            f" [{context}]" if context else "",
            (sample_gt_cells or [])[:3],
            (sample_pred_cells or [])[:3],
        )

    # Actionable warnings for any version
    if total_gt_cells > 0 and total_pred_cells == 0:
        logger.warning(
            "[FOMO eval%s] GT cells exist (%d) but predicted cells == 0 at threshold=%.4f. "
            "Model may be under-confident — lower the threshold or retrain.",
            f" [{context}]" if context else "",
            total_gt_cells,
            threshold,
        )
    if total_pred_cells > 0 and int(tp.sum()) == 0:
        logger.warning(
            "[FOMO eval%s] Predictions exist (%d cells) but ZERO exact matches. "
            "Check label ordering, grid-shape alignment, and GT label resolution.",
            f" [{context}]" if context else "",
            total_pred_cells,
        )
    if n_unresolved_gt > 0:
        logger.warning(
            "[FOMO eval%s] %d GT boxes could not be resolved to a class index. "
            "Ensure label_to_idx contains both UUID and name keys.",
            f" [{context}]" if context else "",
            n_unresolved_gt,
        )


# ── 7. v2-specific diagnostics ────────────────────────────────────────────────

def log_fomo_v2_diagnostics(
    *,
    fomo_version: int,
    feature_tap: str,
    grid_h: int,
    grid_w: int,
    spread_mode: bool,
    bg_weight: float,
    avg_predicted_cells_per_image: Optional[float] = None,
    avg_gt_objects_per_image: Optional[float] = None,
    preds_probs: Optional[np.ndarray] = None,
    y_heatmap: Optional[np.ndarray] = None,
    n_classes: int = 0,
    per_class_metrics: Optional[Dict[str, Dict[str, float]]] = None,
    predicted_class_distribution: Optional[Dict[str, int]] = None,
    per_class_trends: Optional[Dict[str, Dict[str, object]]] = None,
    threshold_trend: Optional[Dict[str, object]] = None,
    context: str = "",
) -> None:
    """Log structured diagnostics specific to FOMO v2 (stride-8 grid).

    Covers:
      • Training config metadata (version, tap, grid, spread, bg_weight)
      • Average predicted cells per image (FP proxy)
      • GT-cell vs BG-cell object confidence (spatial discrimination signal)
      • Per-class confidence at GT cells (class-level learning signal)
      • Per-class TP/FP/FN and predicted class distribution
      • Per-class recall trends across recent epochs
      • Threshold stability across nearby epochs

    Safe to call with partial arguments; missing optional arrays are skipped.

    Parameters
    ----------
    fomo_version : int    1 or 2 (logged for context).
    feature_tap  : str    Layer name or "fpn_fusion" for the FPN path.
    grid_h/w     : int    Grid dimensions (image_h//8 for v2).
    spread_mode  : bool   Whether training targets used spread=True.
    bg_weight    : float  Background cell loss weight used during training.
    avg_predicted_cells_per_image : float  Mean (TP+FP) per image at evaluation.
    avg_gt_objects_per_image : float  Mean GT occupied cells per image.
    preds_probs  : (N, H, W, C+1)  Probability arrays (post-softmax).
    y_heatmap    : (N, H, W, C+1)  GT heatmap targets (one-hot).
    n_classes    : int    Number of object classes (background excluded).
    per_class_metrics : dict  Optional per-class TP/FP/FN/precision/recall/F1.
    predicted_class_distribution : dict  Optional predicted object-cell counts by class.
    per_class_trends : dict  Optional per-class trend summary across epochs.
    threshold_trend : dict  Optional recent threshold/F1 history.
    context      : str    Label for the log lines (e.g. "training", "test").
    """
    tag = f" [{context}]" if context else ""

    logger.info(
        "[FOMO v%d%s] config: feature_tap=%s  grid=%d×%d  spread=%s  bg_weight=%.4f",
        fomo_version, tag, feature_tap, grid_h, grid_w, spread_mode, bg_weight,
    )

    if avg_predicted_cells_per_image is not None:
        grid_total = grid_h * grid_w
        pred_ratio = avg_predicted_cells_per_image / grid_total if grid_total > 0 else 0.0
        gt_suffix = (
            f"  avg_gt_objects/image={avg_gt_objects_per_image:.2f}"
            if avg_gt_objects_per_image is not None else
            ""
        )
        logger.info(
            "[FOMO v%d%s] avg_pred_cells/image=%.2f%s  grid_cells=%d  pred_ratio=%.4f",
            fomo_version, tag,
            avg_predicted_cells_per_image, gt_suffix, grid_total, pred_ratio,
        )
        if pred_ratio > 0.05:
            logger.warning(
                "[FOMO v%d%s] pred_ratio=%.4f (>5%% of grid cells predicted as objects). "
                "High false-positive rate likely — check bg_weight and threshold.",
                fomo_version, tag, pred_ratio,
            )
        if (
            avg_gt_objects_per_image is not None
            and avg_gt_objects_per_image > 0.0
            and avg_predicted_cells_per_image >= max(avg_gt_objects_per_image * 3.0, avg_gt_objects_per_image + 3.0)
        ):
            logger.warning(
                "[FOMO v%d%s] avg_pred_cells/image=%.2f is far above avg_gt_objects/image=%.2f. "
                "Validation is overpredicting object cells — check bg_weight, thresholds, and target policy.",
                fomo_version, tag, avg_predicted_cells_per_image, avg_gt_objects_per_image,
            )

    if predicted_class_distribution:
        logger.info(
            "[FOMO v%d%s] predicted class distribution: %s",
            fomo_version,
            tag,
            ", ".join(f"{name}={count}" for name, count in predicted_class_distribution.items()),
        )
        active_pred = {name: count for name, count in predicted_class_distribution.items() if count > 0}
        missing_pred = [name for name, count in predicted_class_distribution.items() if count == 0]
        if missing_pred and active_pred:
            logger.warning(
                "[FOMO v%d%s] class prediction imbalance: no predicted cells for %s while %s are active.",
                fomo_version,
                tag,
                ", ".join(missing_pred),
                ", ".join(active_pred.keys()),
            )

    if per_class_metrics:
        for cls_name, stats in per_class_metrics.items():
            logger.info(
                "[FOMO v%d%s] class=%s TP=%s FP=%s FN=%s P=%.4f R=%.4f F1=%.4f",
                fomo_version,
                tag,
                cls_name,
                stats.get("tp", 0),
                stats.get("fp", 0),
                stats.get("fn", 0),
                float(stats.get("precision", 0.0) or 0.0),
                float(stats.get("recall", stats.get("r", 0.0)) or 0.0),
                float(stats.get("f1", stats.get("f1-score", 0.0)) or 0.0),
            )

        collapsed = [
            cls_name for cls_name, stats in per_class_metrics.items()
            if int(stats.get("tp", 0) or 0) == 0 and int(stats.get("fn", 0) or 0) > 0
        ]
        learned = [
            cls_name for cls_name, stats in per_class_metrics.items()
            if int(stats.get("tp", 0) or 0) > 0
        ]
        if collapsed and learned:
            logger.warning(
                "[FOMO v%d%s] class collapse detected: %s not learning while %s has true positives.",
                fomo_version,
                tag,
                ", ".join(collapsed),
                ", ".join(learned),
            )

    if per_class_trends:
        for cls_name, trend in per_class_trends.items():
            logger.info(
                "[FOMO v%d%s] class=%s recall trend: recent=%s latest=%.4f avg=%.4f lagging_epochs=%s",
                fomo_version,
                tag,
                cls_name,
                trend.get("recall_history", []),
                float(trend.get("latest_recall", 0.0) or 0.0),
                float(trend.get("avg_recall", 0.0) or 0.0),
                trend.get("lagging_epochs", 0),
            )
            _lagging_eps = int(trend.get("lagging_epochs", 0) or 0)
            if _lagging_eps >= 4:
                logger.warning(
                    "[FOMO v%d%s] class=%s has been lagging (recall<0.05) for %d epochs. "
                    "Consider boosting its loss weight or checking GT annotation coverage.",
                    fomo_version, tag, cls_name, _lagging_eps,
                )
        if len(per_class_trends) >= 2:
            _avg_recalls = {
                cls_name: float(trend.get("avg_recall", 0.0) or 0.0)
                for cls_name, trend in per_class_trends.items()
            }
            _best_cls = max(_avg_recalls, key=_avg_recalls.get)
            _worst_cls = min(_avg_recalls, key=_avg_recalls.get)
            if _avg_recalls[_best_cls] >= _avg_recalls[_worst_cls] + 0.08:
                logger.warning(
                    "[FOMO v%d%s] class recall trend imbalance: %s avg_recall=%.4f is consistently below %s avg_recall=%.4f.",
                    fomo_version,
                    tag,
                    _worst_cls,
                    _avg_recalls[_worst_cls],
                    _best_cls,
                    _avg_recalls[_best_cls],
                )

    if threshold_trend:
        logger.info(
            "[FOMO v%d%s] threshold trend: thresholds=%s val_f1=%s stability_range=%.4f",
            fomo_version,
            tag,
            threshold_trend.get("threshold_history", []),
            threshold_trend.get("val_f1_history", []),
            float(threshold_trend.get("stability_range", 0.0) or 0.0),
        )
        if float(threshold_trend.get("stability_range", 0.0) or 0.0) > 0.15:
            logger.warning(
                "[FOMO v%d%s] threshold stability is poor across nearby epochs (range=%.4f). "
                "Score ranking between GT and background cells is still unstable.",
                fomo_version,
                tag,
                float(threshold_trend.get("stability_range", 0.0) or 0.0),
            )

    if preds_probs is not None and y_heatmap is not None:
        try:
            # GT-cell vs BG-cell confidence comparison
            gt_mask      = np.any(y_heatmap[:, :, :, 1:] > 0.5, axis=-1).reshape(-1)
            max_obj_conf = np.max(preds_probs[:, :, :, 1:], axis=-1).reshape(-1)
            gt_conf = max_obj_conf[gt_mask]
            bg_conf = max_obj_conf[~gt_mask]

            gt_mean = float(np.mean(gt_conf)) if len(gt_conf) > 0 else 0.0
            gt_max  = float(np.max(gt_conf))  if len(gt_conf) > 0 else 0.0
            bg_mean = float(np.mean(bg_conf)) if len(bg_conf) > 0 else 0.0
            bg_max  = float(np.max(bg_conf))  if len(bg_conf) > 0 else 0.0
            ratio   = gt_mean / bg_mean if bg_mean > 0 else float("inf")

            logger.info(
                "[FOMO v%d%s] GT-cell conf: mean=%.4f max=%.4f  |  "
                "BG-cell conf: mean=%.4f max=%.4f  |  gt/bg_ratio=%.2f",
                fomo_version, tag, gt_mean, gt_max, bg_mean, bg_max, ratio,
            )
            if ratio < 1.5:
                logger.warning(
                    "[FOMO v%d%s] gt/bg_ratio=%.2f (<1.5). "
                    "Model not discriminating GT vs background cells — "
                    "check bg_weight, spread, and training epochs.",
                    fomo_version, tag, ratio,
                )

            # Per-class confidence at GT cells (requires channel info)
            if n_classes > 0 and preds_probs.shape[-1] == n_classes + 1:
                for ci in range(n_classes):
                    cls_gt_mask = (y_heatmap[:, :, :, ci + 1] > 0.5).reshape(-1)
                    if cls_gt_mask.sum() == 0:
                        continue
                    cls_conf = preds_probs[:, :, :, ci + 1].reshape(-1)[cls_gt_mask]
                    logger.info(
                        "[FOMO v%d%s] class %d GT-cell conf: mean=%.4f max=%.4f  "
                        "(n_gt_cells=%d)",
                        fomo_version, tag, ci,
                        float(np.mean(cls_conf)), float(np.max(cls_conf)),
                        int(cls_gt_mask.sum()),
                    )

            if n_classes > 0 and preds_probs.shape[-1] == n_classes + 1:
                center_neighbor_gaps: List[float] = []
                gt_center_probs: List[float] = []
                mean_near_probs: List[float] = []
                outranked_count = 0
                exact_hit_count = 0
                per_class_gaps: Dict[int, List[float]] = {}
                per_class_center_probs: Dict[int, List[float]] = {}
                per_class_exact_hits: Dict[int, int] = {}
                per_class_outranked: Dict[int, int] = {}
                per_class_total_gt: Dict[int, int] = {}
                for bi in range(y_heatmap.shape[0]):
                    gt_positions = np.argwhere(np.max(y_heatmap[bi, :, :, 1:], axis=-1) > 0.5)
                    for gy, gx in gt_positions:
                        cls_idx = int(np.argmax(y_heatmap[bi, gy, gx, 1:]))
                        center_prob = float(preds_probs[bi, gy, gx, cls_idx + 1])
                        near_probs = []
                        for dy in (-1, 0, 1):
                            for dx in (-1, 0, 1):
                                if dy == 0 and dx == 0:
                                    continue
                                ny, nx = gy + dy, gx + dx
                                if 0 <= ny < grid_h and 0 <= nx < grid_w:
                                    near_probs.append(float(preds_probs[bi, ny, nx, cls_idx + 1]))
                        if not near_probs:
                            continue
                        max_near = max(near_probs)
                        gap = center_prob - max_near
                        center_neighbor_gaps.append(gap)
                        gt_center_probs.append(center_prob)
                        mean_near_probs.append(float(np.mean(near_probs)))
                        per_class_gaps.setdefault(cls_idx, []).append(gap)
                        per_class_center_probs.setdefault(cls_idx, []).append(center_prob)
                        per_class_total_gt[cls_idx] = per_class_total_gt.get(cls_idx, 0) + 1
                        if gap < 0.0:
                            outranked_count += 1
                            per_class_outranked[cls_idx] = per_class_outranked.get(cls_idx, 0) + 1
                        else:
                            exact_hit_count += 1
                            per_class_exact_hits[cls_idx] = per_class_exact_hits.get(cls_idx, 0) + 1

                if center_neighbor_gaps:
                    n_gt = len(center_neighbor_gaps)
                    mean_gap = float(np.mean(center_neighbor_gaps))
                    outrank_rate = float(outranked_count / n_gt)
                    exact_hit_recall = float(exact_hit_count / n_gt)
                    mean_center = float(np.mean(gt_center_probs))
                    mean_nearby = float(np.mean(mean_near_probs))
                    logger.info(
                        "[FOMO v%d%s] GT center prob: mean=%.4f  |  "
                        "nearby-cell prob: mean=%.4f  |  "
                        "center-vs-nearby gap: mean=%.4f  |  "
                        "outranked_rate=%.4f  exact_hit_recall=%.4f  n_gt=%d",
                        fomo_version, tag,
                        mean_center, mean_nearby, mean_gap, outrank_rate, exact_hit_recall, n_gt,
                    )
                    for cls_idx, gaps in sorted(per_class_gaps.items()):
                        _cls_center_mean = float(np.mean(per_class_center_probs.get(cls_idx, [0.0])))
                        _cls_total = per_class_total_gt.get(cls_idx, 0)
                        _cls_exact_recall = (
                            per_class_exact_hits.get(cls_idx, 0) / _cls_total
                            if _cls_total > 0 else 0.0
                        )
                        logger.info(
                            "[FOMO v%d%s] class %d center-vs-nearby prob gap: mean=%.4f  "
                            "center_prob=%.4f  exact_hit_recall=%.4f  n_gt=%d",
                            fomo_version, tag, cls_idx,
                            float(np.mean(gaps)), _cls_center_mean, _cls_exact_recall, len(gaps),
                        )

                    # Warn when one class's center confidence lags another's.
                    if len(per_class_center_probs) >= 2:
                        _cls_cp_means = {
                            ci: float(np.mean(ps))
                            for ci, ps in per_class_center_probs.items()
                            if ps
                        }
                        _best_cp  = max(_cls_cp_means, key=_cls_cp_means.get)
                        _worst_cp = min(_cls_cp_means, key=_cls_cp_means.get)
                        if _cls_cp_means[_best_cp] >= _cls_cp_means[_worst_cp] + 0.10:
                            logger.warning(
                                "[FOMO v%d%s] per-class center-confidence imbalance: "
                                "class %d center_prob=%.4f vs class %d center_prob=%.4f "
                                "— lagging class may need a higher loss weight.",
                                fomo_version, tag,
                                _worst_cp, _cls_cp_means[_worst_cp],
                                _best_cp,  _cls_cp_means[_best_cp],
                            )
                        _cls_er = {
                            ci: (per_class_exact_hits.get(ci, 0) / per_class_total_gt[ci])
                            for ci in per_class_total_gt
                            if per_class_total_gt[ci] > 0
                        }
                        if len(_cls_er) >= 2:
                            _best_er  = max(_cls_er, key=_cls_er.get)
                            _worst_er = min(_cls_er, key=_cls_er.get)
                            if _cls_er[_best_er] >= _cls_er[_worst_er] + 0.15:
                                logger.warning(
                                    "[FOMO v%d%s] per-class exact-hit recall imbalance: "
                                    "class %d exact_hit_recall=%.4f vs class %d exact_hit_recall=%.4f "
                                    "— lower class needs stronger center-confidence training signal.",
                                    fomo_version, tag,
                                    _worst_er, _cls_er[_worst_er],
                                    _best_er,  _cls_er[_best_er],
                                )

                    if outrank_rate > 0.25 or mean_gap < 0.02:
                        logger.warning(
                            "[FOMO v%d%s] near-miss cells are frequently rivaling or outranking the GT center "
                            "(mean_gap=%.4f, outranked_rate=%.4f, exact_hit_recall=%.4f). "
                            "Exact-cell ranking is still weak.",
                            fomo_version, tag, mean_gap, outrank_rate, exact_hit_recall,
                        )
        except Exception as _e:
            logger.warning("[FOMO v%d%s] v2 diagnostics failed: %s", fomo_version, tag, _e)
