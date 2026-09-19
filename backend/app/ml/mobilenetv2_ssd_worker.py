"""
app/ml/mobilenetv2_ssd_worker.py
─────────────────────────────────
MobileNetV2 SSD FPN-Lite training pipeline — invoked by training_worker.py
when ``selected_architecture == "mobilenet_v2_ssd_fpn_lite"``.

Mirrors the structure of yolo_pro_worker.py exactly so the dispatch pattern
in training_worker.py is symmetric and easy to extend.

Architecture lives in:  backend/app/ml/MobileNetV2 SSD/
This worker imports it via the package shim at  backend/app/ml/MobileNetV2 SSD/__init__.py,
which injects the folder onto sys.path so bare-name intra-package imports
(``from primitives import CFG``, etc.) resolve without any source changes.
"""

from __future__ import annotations

import io
import logging
import math
import os
import sys
import tempfile
import numpy as np
import tensorflow as tf
from datetime import datetime
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

if TYPE_CHECKING:
    from sqlalchemy.orm import Session
    from app.models.user import TrainingJob, Impulse

logger = logging.getLogger(__name__)

MV2_SSD_ARCHITECTURE = "mobilenet_v2_ssd_fpn_lite"

# Fixed input resolution required by the SSD model (must be 320×320×3).
MV2_SSD_INPUT_H: int = 320
MV2_SSD_INPUT_W: int = 320
MV2_SSD_INPUT_C: int = 3
MV2_SSD_INPUT_SHAPE: Tuple[int, int, int] = (MV2_SSD_INPUT_H, MV2_SSD_INPUT_W, MV2_SSD_INPUT_C)

# ─── Hard-negative mining budget ──────────────────────────────────────────────

# Classic SSD ratio: this many negative anchors mined per positive anchor.
SSD_NEG_POS_RATIO: int = 3

# Floor on the per-image hard-negative budget, applied ONLY to images with zero
# positive anchors — i.e. background ("negative") images.
#
# Why it is needed: the budget is NEG_POS_RATIO × num_pos_i.  A background image
# has num_pos_i == 0, so the old max(num_pos, 1) × 3 handed it **3** negative
# anchors out of the ~8 028 this model generates.  That image costs a full
# forward + backward pass and delivers almost no gradient, which makes adding
# negatives look ineffective rather than merely slow.
#
# Why 32: a moderately-populated positive image (8-10 objects) already receives
# 24-30 negatives at ratio 3, so 32 puts a background image on comparable
# footing while still touching only ~0.4 % of the anchor grid — far too few to
# swamp the positive term in cls_loss.  The number is chosen against *this*
# model's anchor count (see generate_anchors()); do not copy it into a detector
# with a different anchor budget.
SSD_MIN_NEGATIVES_PER_IMAGE: int = 32


def ssd_negative_budget(pos_count_per_img):
    """Per-image hard-negative budget, given each image's positive-anchor count.

    Images WITH positives keep the classic NEG_POS_RATIO × num_pos budget
    untouched, so this is a numerical no-op on any dataset that contains no
    background images.  Images with zero positives get
    SSD_MIN_NEGATIVES_PER_IMAGE instead.

    Accepts any float tensor shape; returns the same shape.
    """
    import tensorflow as _tf
    pos = _tf.cast(pos_count_per_img, _tf.float32)
    has_pos = _tf.cast(pos > 0.0, _tf.float32)
    return (
        has_pos * pos * _tf.cast(SSD_NEG_POS_RATIO, _tf.float32)
        + (1.0 - has_pos) * _tf.cast(SSD_MIN_NEGATIVES_PER_IMAGE, _tf.float32)
    )


from app.workers.cancel_utils import (
    CancelledError as _CancelledError,
    raise_if_cancelled as _raise_if_job_cancelled,
    raise_if_cancelled_fast as _raise_if_job_cancelled_fast,
    assert_not_cancelled_before_completing as _assert_not_cancelled_before_completing,
    promote_run_to_active as _promote_run_to_active,
    register_live_buffer as _register_live_buffer,
)


# ─────────────────────────────────────────────────────────────────────────────
# Public predicate — called by training_worker.py dispatch
# ─────────────────────────────────────────────────────────────────────────────

def is_mobilenetv2_ssd(architecture: str) -> bool:
    """True when the selected architecture is MobileNetV2 SSD FPN-Lite."""
    return architecture.lower().strip() == MV2_SSD_ARCHITECTURE


# ─────────────────────────────────────────────────────────────────────────────
# JSON safety helper (mirrors yolo_pro_worker._sanitize_json)
# ─────────────────────────────────────────────────────────────────────────────

def _sanitize_json(obj):
    """Recursively replace non-finite floats with None for JSON/JSONB safety."""
    if isinstance(obj, float):
        return None if not math.isfinite(obj) else obj
    if isinstance(obj, dict):
        return {k: _sanitize_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_json(v) for v in obj]
    return obj


# ─────────────────────────────────────────────────────────────────────────────
# Internal: ensure SSD package is importable
# ─────────────────────────────────────────────────────────────────────────────

def _ensure_ssd_package() -> str:
    """
    Return the absolute path to the MobileNetV2 SSD/ folder and guarantee
    it is on sys.path.  The __init__.py shim already does this on import, but
    this helper makes the path available for file-level operations (e.g. export
    output directory resolution).

    Folder layout (mirrors yolo_pro/):
        backend/app/ml/mobilenetv2_ssd_worker.py   ← this file
        backend/app/ml/MobileNetV2 SSD/__init__.py ← one level up from here
    """
    # This worker lives at:  .../backend/app/ml/mobilenetv2_ssd_worker.py
    # The SSD folder lives at: .../backend/app/ml/MobileNetV2 SSD/
    # So: os.path.dirname(this_file) == .../backend/app/ml/
    #
    # REMOTE WORKER / DOCKER IMAGE NOTE:
    # backend/app/ml/MobileNetV2 SSD/ must be present on every machine that
    # runs this worker (including AWS GPU instances).  The Docker build context
    # must include the full backend/app/ml/ tree — do NOT package this worker
    # file in isolation.
    ml_dir  = os.path.dirname(os.path.abspath(__file__))
    ssd_dir = os.path.join(ml_dir, "MobileNetV2 SSD")
    if not os.path.isdir(ssd_dir):
        raise ImportError(
            f"MobileNetV2 SSD folder not found at expected path: {ssd_dir}"
            "Ensure the folder is placed at backend/app/ml/MobileNetV2 SSD/ "
            "(alongside yolo_pro/)."
        )
    if ssd_dir not in sys.path:
        sys.path.insert(0, ssd_dir)
    return ssd_dir


# ─────────────────────────────────────────────────────────────────────────────
# Box utility: convert {label, x, y, w, h} dicts → [y1, x1, y2, x2] arrays
# ─────────────────────────────────────────────────────────────────────────────

def _boxes_to_array(
    box_list: Optional[List[dict]],
    label_map: dict,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert a per-image list of bounding-box dicts to numpy arrays.

    Args:
        box_list  : list of dicts with keys label/label_id, x, y, w, h (0-1 normalised).
                    None or [] → returns empty arrays.
        label_map : {label_id_or_name → class_index}

    Returns:
        boxes   : (N, 4) float32  [y1, x1, y2, x2] normalised
        classes : (N,)   int32    class indices
    """
    if not box_list:
        return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.int32)

    box_arr, cls_arr = [], []
    for b in box_list:
        key = b.get("label_id") or b.get("label")
        if key not in label_map:
            continue
        x, y, w, h = float(b["x"]), float(b["y"]), float(b["w"]), float(b["h"])
        x1, y1 = max(0.0, x), max(0.0, y)
        x2, y2 = min(1.0, x + w), min(1.0, y + h)
        if x2 <= x1 or y2 <= y1:
            continue
        box_arr.append([y1, x1, y2, x2])
        cls_arr.append(label_map[key])

    if not box_arr:
        return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.int32)
    return (
        np.array(box_arr, dtype=np.float32),
        np.array(cls_arr, dtype=np.int32),
    )


# ─────────────────────────────────────────────────────────────────────────────
# SSD target encoding
# ─────────────────────────────────────────────────────────────────────────────

def _encode_targets(
    boxes_np: np.ndarray,
    classes_np: np.ndarray,
    anchors_np: np.ndarray,
    num_classes: int,
    match_iou_threshold: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Encode GT boxes into SSD regression + classification targets for one image.

    Args:
        boxes_np   : (M, 4) float32 GT boxes [y1, x1, y2, x2] normalised
        classes_np : (M,)   int32   GT class indices (0-based)
        anchors_np : (A, 4) float32 anchors [cy, cx, h, w] normalised
        num_classes: number of foreground classes

    Returns:
        loc_target : (A, 4)          float32  encoded box deltas
        cls_target : (A, num_classes+1) float32  one-hot; col-0 = background
    """
    A = len(anchors_np)
    loc_target = np.zeros((A, 4), dtype=np.float32)
    cls_target = np.zeros((A, num_classes + 1), dtype=np.float32)
    cls_target[:, 0] = 1.0  # all background by default

    if len(boxes_np) == 0:
        return loc_target, cls_target

    # Convert anchors from [cy, cx, h, w] → [y1, x1, y2, x2] for IoU
    acy, acx, ah, aw = (anchors_np[:, i] for i in range(4))
    ay1, ax1 = acy - ah / 2, acx - aw / 2
    ay2, ax2 = acy + ah / 2, acx + aw / 2

    gy1, gx1, gy2, gx2 = (boxes_np[:, i] for i in range(4))

    # IoU matrix (A, M)
    inter_y1 = np.maximum(ay1[:, None], gy1[None, :])
    inter_x1 = np.maximum(ax1[:, None], gx1[None, :])
    inter_y2 = np.minimum(ay2[:, None], gy2[None, :])
    inter_x2 = np.minimum(ax2[:, None], gx2[None, :])
    inter = np.maximum(0.0, inter_y2 - inter_y1) * np.maximum(0.0, inter_x2 - inter_x1)
    a_area = (ay2 - ay1) * (ax2 - ax1)
    g_area = (gy2 - gy1) * (gx2 - gx1)
    union = a_area[:, None] + g_area[None, :] - inter
    iou = np.where(union > 0, inter / union, 0.0)  # (A, M)

    best_gt_idx = iou.argmax(axis=1)   # (A,) — best GT for each anchor
    best_iou    = iou[np.arange(A), best_gt_idx]

    _NEG_IOU_THRESHOLD = 0.4   # [_NEG_IOU, match_iou_threshold) → neutral band
    positive_mask = best_iou >= match_iou_threshold
    neutral_mask  = (best_iou >= _NEG_IOU_THRESHOLD) & ~positive_mask  # [0.4, 0.5)

    # Force-match: every GT must have at least one anchor
    best_anchor_per_gt = iou.argmax(axis=0)  # (M,)
    positive_mask[best_anchor_per_gt] = True
    neutral_mask[best_anchor_per_gt]  = False  # force-positive overrides neutral
    best_gt_idx[best_anchor_per_gt] = np.arange(len(boxes_np))

    # Mark neutral anchors with sentinel col-0 = -1 so the loss ignores them
    # Convention: col-0  1.0 → negative,  0.0 → positive,  -1.0 → neutral
    cls_target[neutral_mask, 0] = -1.0

    pos_indices = np.where(positive_mask)[0]
    if len(pos_indices) == 0:
        return loc_target, cls_target

    matched_boxes = boxes_np[best_gt_idx[pos_indices]]   # (P, 4) [y1,x1,y2,x2]
    matched_cls   = classes_np[best_gt_idx[pos_indices]] # (P,)

    # Encode box deltas (SSD style with variances [0.1, 0.1, 0.2, 0.2])
    VARIANCES = np.array([0.1, 0.1, 0.2, 0.2], dtype=np.float32)
    a_cy = anchors_np[pos_indices, 0]
    a_cx = anchors_np[pos_indices, 1]
    a_h  = anchors_np[pos_indices, 2]
    a_w  = anchors_np[pos_indices, 3]

    g_cy = (matched_boxes[:, 0] + matched_boxes[:, 2]) / 2
    g_cx = (matched_boxes[:, 1] + matched_boxes[:, 3]) / 2
    g_h  = matched_boxes[:, 2] - matched_boxes[:, 0]
    g_w  = matched_boxes[:, 3] - matched_boxes[:, 1]

    g_h  = np.maximum(g_h,  1e-7)
    g_w  = np.maximum(g_w,  1e-7)
    a_h  = np.maximum(a_h,  1e-7)
    a_w  = np.maximum(a_w,  1e-7)

    dy = ((g_cy - a_cy) / a_h) / VARIANCES[0]
    dx = ((g_cx - a_cx) / a_w) / VARIANCES[1]
    dh = np.log(g_h / a_h) / VARIANCES[2]
    dw = np.log(g_w / a_w) / VARIANCES[3]

    loc_target[pos_indices] = np.stack([dy, dx, dh, dw], axis=1)

    # One-hot class targets (foreground classes start at col 1)
    cls_target[pos_indices, 0] = 0.0
    cls_target[pos_indices, matched_cls + 1] = 1.0

    return loc_target, cls_target


# ─────────────────────────────────────────────────────────────────────────────
# Dataset builder for SSD
# ─────────────────────────────────────────────────────────────────────────────

def _build_horizontal_mirror_index(anchors_np: np.ndarray) -> np.ndarray:
    """
    Build the anchor permutation for a horizontal (left↔right) image flip.

    Targets reach the data pipeline already encoded as per-anchor deltas, so a
    pixel flip alone would desynchronise image and boxes. Under a horizontal
    flip the object matched to anchor i (centre cx) moves to the mirror anchor
    j whose centre is 1 - cx (same cy, h, w). The SSD anchor grid is symmetric
    (cell centres at (k+0.5)/grid), so every anchor has an exact mirror in the
    set. Returns `mirror` with mirror[i] = j; gathering encoded targets by this
    index (plus negating the dx delta) yields flip-consistent box targets.

    Falls back to identity for any anchor whose mirror is not found (keeps the
    sample valid rather than corrupting targets).
    """
    A = anchors_np.shape[0]

    def _key(cy, cx, h, w):
        return (round(float(cy), 6), round(float(cx), 6),
                round(float(h), 6), round(float(w), 6))

    key_to_idx = {
        _key(anchors_np[i, 0], anchors_np[i, 1], anchors_np[i, 2], anchors_np[i, 3]): i
        for i in range(A)
    }
    mirror = np.arange(A, dtype=np.int32)
    for i in range(A):
        cy, cx, h, w = anchors_np[i]
        j = key_to_idx.get(_key(cy, 1.0 - cx, h, w))
        if j is not None:
            mirror[i] = j
    return mirror


def _build_ssd_tf_dataset(
    images: np.ndarray,
    loc_targets: np.ndarray,
    cls_targets: np.ndarray,
    batch_size: int,
    shuffle: bool = True,
    augment: bool = False,
    mirror_idx: Optional[np.ndarray] = None,
) -> tf.data.Dataset:
    """
    Wrap numpy arrays as a batched, optionally shuffled tf.data.Dataset.

    When `augment` is True (train split only — never pass it for val), each
    sample is independently transformed every epoch:
      - mild photometric jitter (brightness/contrast), which leaves boxes
        unchanged, and
      - a 50% horizontal flip that flips the image AND remaps the encoded
        targets via `mirror_idx` (negating the dx delta) so boxes stay aligned.
    Images stay uint8 end-to-end so the model's preprocessing contract is
    identical whether augmentation is on or off.
    """
    ds = tf.data.Dataset.from_tensor_slices((
        images,
        {"loc": loc_targets, "cls": cls_targets},
    ))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(images), reshuffle_each_iteration=True)

    if augment:
        # dx is column 1 of the encoded [dy, dx, dh, dw] delta — it negates
        # under a horizontal flip; dy/dh/dw are flip-invariant.
        _flip_sign = tf.constant([1.0, -1.0, 1.0, 1.0], dtype=tf.float32)
        _mirror = tf.constant(
            mirror_idx if mirror_idx is not None
            else np.arange(loc_targets.shape[1], dtype=np.int32),
            dtype=tf.int32,
        )

        def _augment_sample(image, targets):
            img = tf.cast(image, tf.float32)
            # Photometric jitter — no geometric effect on boxes.
            img = tf.image.random_brightness(img, max_delta=0.10 * 255.0)
            img = tf.image.random_contrast(img, lower=0.85, upper=1.15)
            img = tf.clip_by_value(img, 0.0, 255.0)

            loc = targets["loc"]
            cls = targets["cls"]

            def _flipped():
                return (
                    tf.image.flip_left_right(img),
                    tf.gather(loc, _mirror, axis=0) * _flip_sign,
                    tf.gather(cls, _mirror, axis=0),
                )

            img, loc, cls = tf.cond(
                tf.random.uniform([]) < 0.5,
                _flipped,
                lambda: (img, loc, cls),
            )
            return tf.cast(img, tf.uint8), {"loc": loc, "cls": cls}

        ds = ds.map(_augment_sample, num_parallel_calls=tf.data.AUTOTUNE)

    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


# ─────────────────────────────────────────────────────────────────────────────
# Public entry point — called by training_worker.py
# ─────────────────────────────────────────────────────────────────────────────

def run_mobilenetv2_ssd_training(
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
    Full MobileNetV2 SSD FPN-Lite training run.

    Called from training_worker.py when is_mobilenetv2_ssd(selected_architecture).

    Signature is intentionally identical to run_yolo_pro_training so the
    dispatch block in training_worker.py is symmetric.

    Input contract:
        X_train / X_test : (N, H, W, C) uint8 — the model's NormalizationLayer
                           handles [0,255] → [-1,1] internally.
        boxes_*          : list of per-image box dicts
                           {'label'/'label_id', 'x', 'y', 'w', 'h'} 0-1 normalised.
        input_shape      : ignored for SSD — the model is fixed at (320, 320, 3).
                           Images are resized to 320×320 before encoding.

    Returns:
        dict with keys: architecture, num_classes, input_shape, epochs,
                        final_train_loss, final_val_loss, tflite_size_bytes,
                        label_names — stored as job.result by the caller.
    """
    # ── Edge-Impulse-style early input-size validation ────────────────────
    # SSD FPN-Lite is fixed at 320×320. If the impulse's input block is set
    # to anything else, fail fast with a clear user-facing message instead
    # of silently resizing or crashing deep in the model build.
    # Read the impulse's configured input-block image size (the SSD dataset
    # loader force-resizes to 320×320, so `input_shape` / X_train.shape here
    # are always 320 — we must inspect the impulse config directly).
    try:
        _in_w = int(getattr(impulse, "image_width", None) or 0)
        _in_h = int(getattr(impulse, "image_height", None) or 0)
    except Exception:
        _in_w, _in_h = 0, 0
    if _in_w == 0 or _in_h == 0:
        try:
            _in_h = int(input_shape[0]); _in_w = int(input_shape[1])
        except Exception:
            pass
    if _in_w != MV2_SSD_INPUT_W or _in_h != MV2_SSD_INPUT_H:
        raise ValueError(
            f"Your image size is currently set to {_in_w}x{_in_h}. "
            f"MobileNetV2 SSD FPN-Lite 320x320 currently only supports a 320x320 input.\n\n"
            f"To use this model, change the image size in your input block to a "
            f"supported value; or switch your base model via \"Choose a different model\"."
        )

    _ensure_ssd_package()
    from model import build_model as ssd_build_model         # noqa: E402 (after path injection)
    from anchor_generator import generate_anchors             # noqa: E402
    from losses import smooth_l1_loss as _smooth_l1_loss, sigmoid_bce_loss as _sigmoid_bce_loss                # noqa: E402
    from box_coder import decode_boxes as _ssd_decode_boxes  # noqa: E402
    from app.models.user import TrainedModel, JobStatus

    # ── constants ──────────────────────────────────────────────────────────
    H, W, C   = MV2_SSD_INPUT_H, MV2_SSD_INPUT_W, MV2_SSD_INPUT_C
    n_classes = len(label_names)

    job_params      = job.extra_params or {}
    requested_batch = int(job.batch_size or 16)
    lr              = float(job.learning_rate or 1e-3)
    epochs          = int(job.epochs or 100)
    neck_ch         = int(job_params.get("neck_channels", 256))

    # Training toggles from the UI (persisted into extra_params by the training
    # endpoint). Default True preserves prior behavior if a job predates the
    # toggles. Augmentation is applied to the train split only; early stopping
    # monitors val_loss and only engages after the warmup phase.
    use_augmentation = bool(job_params.get("data_augmentation", True))
    use_early_stop   = bool(job_params.get("early_stopping", True))

    # ── Adaptive batch sizing (Edge-Impulse-style) ─────────────────────────
    # Small datasets can produce < 1 step/epoch with the user-requested batch,
    # causing the training loop to collapse.  Clamp to a sensible effective
    # size while honouring the requested value as an upper bound.
    _n_train = len(X_train)   # raw count before resize; same length as X_train_r
    if _n_train < 16:
        _adaptive = min(8, _n_train)
        _reason   = f"train_samples={_n_train} < 16 → capped to {_adaptive}"
    elif _n_train < 32:
        _adaptive = 8
        _reason   = f"train_samples={_n_train} < 32 → set to 8"
    elif _n_train < 100:
        _adaptive = 16
        _reason   = f"train_samples={_n_train} < 100 → set to 16"
    else:
        _adaptive = min(requested_batch, 32)
        _reason   = f"train_samples={_n_train} → min(requested={requested_batch}, 32)={_adaptive}"

    # User-requested value is always an additional upper bound; never exceed
    # the training-set size (would yield 0 steps per epoch).
    batch_size = max(1, min(_adaptive, requested_batch, _n_train))
    _steps_per_epoch = max(1, _n_train // batch_size)

    logger.info(
        f"[{job_id}] MobileNetV2 SSD training — "
        f"input=({H},{W},{C})  n_classes={n_classes}  "
        f"epochs={epochs}  requested_batch={requested_batch}  "
        f"effective_batch={batch_size}  steps/epoch≈{_steps_per_epoch}  "
        f"lr={lr}  [{_reason}]"
    )

    # ── 1. Build model ─────────────────────────────────────────────────────
    model = ssd_build_model(
        input_shape=(H, W, C),
        num_classes=n_classes,
        neck_channels=neck_ch,
    )
    model.summary(print_fn=logger.info)

    # ── 2. Generate anchors ────────────────────────────────────────────────
    anchors_np = generate_anchors().numpy()   # (8028, 4) [cy, cx, h, w]
    A = len(anchors_np)

    # Precompute the horizontal-flip anchor permutation once. Only needed when
    # augmentation is enabled (used to keep encoded box targets aligned with a
    # flipped image — see _build_ssd_tf_dataset).
    mirror_idx = _build_horizontal_mirror_index(anchors_np) if use_augmentation else None
    if use_augmentation:
        _matched = int(np.sum(mirror_idx != np.arange(A)))
        logger.info(
            f"[{job_id}] Data augmentation: ENABLED (train only) — "
            f"hflip+brightness/contrast; {_matched}/{A} anchors have a mirror"
        )
    else:
        logger.info(f"[{job_id}] Data augmentation: DISABLED")

    # ── 3. Resize images to 320×320 if needed ─────────────────────────────
    def _resize_images(imgs: np.ndarray) -> np.ndarray:
        if imgs.shape[1] == H and imgs.shape[2] == W:
            return imgs
        out = tf.image.resize(imgs.astype(np.float32), (H, W)).numpy().astype(np.uint8)
        return out

    logger.info(f"[{job_id}] Resizing train images to {H}×{W} …")
    X_train_r = _resize_images(X_train)
    X_test_r  = _resize_images(X_test) if len(X_test) > 0 else X_test

    # ── 4. Encode targets ──────────────────────────────────────────────────
    logger.info(f"[{job_id}] Encoding SSD targets for {len(X_train_r)} train images …")

    def _encode_batch(images, box_lists):
        N = len(images)
        loc_t = np.zeros((N, A, 4),             dtype=np.float32)
        cls_t = np.zeros((N, A, n_classes + 1), dtype=np.float32)
        for i, bl in enumerate(box_lists):
            boxes_i, classes_i = _boxes_to_array(bl, label_map)
            loc_t[i], cls_t[i] = _encode_targets(
                boxes_i, classes_i, anchors_np, n_classes
            )
        return loc_t, cls_t

    loc_train, cls_train = _encode_batch(X_train_r, boxes_train)
    loc_test,  cls_test  = _encode_batch(X_test_r,  boxes_test) \
        if len(X_test_r) > 0 else (
            np.zeros((0, A, 4), np.float32),
            np.zeros((0, A, n_classes + 1), np.float32),
        )

    # ── 5. Build tf.data pipelines ─────────────────────────────────────────
    train_ds = _build_ssd_tf_dataset(
        X_train_r, loc_train, cls_train, batch_size, shuffle=True,
        augment=use_augmentation, mirror_idx=mirror_idx,
    )
    # val_ds is never augmented — evaluation must see the real distribution.
    val_ds   = _build_ssd_tf_dataset(X_test_r,  loc_test,  cls_test,  batch_size, shuffle=False) \
        if len(X_test_r) > 0 else None

    # ── 6. Compile ─────────────────────────────────────────────────────────
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

    # Custom training loop — SSD loss operates on pre-encoded targets
    # (loc_targets, cls_targets) produced by _encode_targets() above.
    # losses.py:ssd_loss expects raw GT boxes/labels + anchors and does its
    # own matching — incompatible with this pre-encoded pipeline.
    # _ssd_loss_encoded below matches the pre-encoded target format exactly.
    NEG_POS_RATIO = SSD_NEG_POS_RATIO

    def _ssd_loss_encoded(box_pred, cls_pred, loc_targets, cls_targets):
        """
        SSD loss over a batch using pre-encoded targets.

        loc_targets : (B, A, 4)           — encoded box deltas (zeros for bg)
        cls_targets : (B, A, n_classes+1) — one-hot; col-0 = background

        Returns scalar total loss (box_loss + cls_loss).
        """
        # Positive mask: anchors assigned to a foreground class
        # col-0 sentinel: 0.0=positive, 1.0=negative, -1.0=neutral
        pos_mask  = tf.equal(cls_targets[:, :, 0],  0.0)  # (B, A) bool
        neg_mask  = tf.equal(cls_targets[:, :, 0],  1.0)  # (B, A) bool — excludes neutrals
        pos_float = tf.cast(pos_mask, tf.float32)          # (B, A)
        num_pos   = tf.maximum(tf.reduce_sum(pos_float), 1.0)

        # ── Box loss: Smooth L1 on positives only ─────────────────────────
        diff = box_pred - loc_targets                     # (B, A, 4)
        abs_diff = tf.abs(diff)
        smooth_l1 = tf.where(abs_diff < 1.0,
                             0.5 * tf.square(diff),
                             abs_diff - 0.5)              # (B, A, 4)
        box_loss_per = tf.reduce_sum(smooth_l1, axis=-1)  # (B, A)
        box_loss = tf.reduce_sum(pos_float * box_loss_per) / num_pos

        # ── Class loss: sigmoid BCE ────────────────────────────────────────
        # cls_pred  : (B, A, n_classes)   — model outputs (no background col)
        # cls_targets col 0 = background → foreground targets = cols 1..end
        fg_targets = cls_targets[:, :, 1:]                # (B, A, n_classes)
        bce_per = tf.reduce_sum(
            tf.nn.sigmoid_cross_entropy_with_logits(
                labels=fg_targets, logits=cls_pred),
            axis=-1,
        )                                                 # (B, A)

        # ── Per-image hard negative mining ────────────────────────────────
        # Mask non-negatives with -inf so they rank last in argsort.
        # Neutral anchors (col-0 == -1) are already excluded by neg_mask.
        neg_loss_ranked = tf.where(
            neg_mask, bce_per, tf.fill(tf.shape(bce_per), -1e9)
        )                                                 # (B, A)

        # Double-argsort gives per-row rank; rank 0 = highest neg loss.
        sorted_rank = tf.argsort(
            tf.argsort(-neg_loss_ranked, axis=1), axis=1
        )                                                 # (B, A) int32

        # Per-image budget, shape (B, 1) for broadcast — see ssd_negative_budget.
        pos_count_per_img = tf.reduce_sum(pos_float, axis=1, keepdims=True)  # (B, 1)
        neg_budget = ssd_negative_budget(pos_count_per_img)                  # (B, 1)

        hard_neg_mask = tf.cast(
            neg_mask & (tf.cast(sorted_rank, tf.float32) < neg_budget),
            tf.float32,
        )                                                 # (B, A)

        keep = pos_float + hard_neg_mask                  # (B, A)
        cls_loss = tf.reduce_sum(keep * bce_per) / num_pos

        return box_loss + cls_loss

    # Two-phase training — mirrors FOMO's warmup → fine-tune schedule:
    # Phase 1 (epochs 1..SSD_WARMUP_EPOCHS): backbone frozen, only FPN-Lite
    #   neck + SSD head trained at full lr.  Pretrained ImageNet features are
    #   preserved while the randomly-initialised head learns to detect objects.
    # Phase 2 (epochs SSD_WARMUP_EPOCHS+1..end): backbone unfrozen, all layers
    #   fine-tuned at lr/10 so backbone features adapt slowly without being
    #   destroyed (catastrophic forgetting).
    #
    # @tf.function is intentionally omitted: the decorator traces model.trainable_variables
    # once at first call.  After backbone unfreeze the variable set changes, so
    # eager execution is required for phase 2 gradients to include backbone vars.
    SSD_WARMUP_EPOCHS = min(10, epochs)

    def _train_step(images, loc_targets, cls_targets):
        with tf.GradientTape() as tape:
            outputs = model(images, training=True)
            box_pred = outputs["box_predictions"]  # (B, A, 4)
            cls_pred = outputs["cls_predictions"]  # (B, A, n_classes)
            loss = _ssd_loss_encoded(box_pred, cls_pred, loc_targets, cls_targets)

        grads = tape.gradient(loss, model.trainable_variables)
        grads, _ = tf.clip_by_global_norm(grads, 10.0)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return loss

    def _val_step(images, loc_targets, cls_targets):
        outputs  = model(images, training=False)
        box_pred = outputs["box_predictions"]
        cls_pred = outputs["cls_predictions"]
        return _ssd_loss_encoded(box_pred, cls_pred, loc_targets, cls_targets)

    # ── 7. Training loop ───────────────────────────────────────────────────
    history: Dict[str, List[float]] = {"train_loss": [], "val_loss": []}
    # Per-epoch metrics for the Training Graphs. SSD is a detection
    # architecture, so only loss is tracked — accuracy fields are omitted
    # (never surface mAP as val_accuracy). Persisted incrementally on the same
    # cadence as the live-loss snapshot so partial runs survive a crash.
    epoch_metrics: List[dict] = []
    # Human-readable per-epoch lines for the live log panel, mirroring
    # YOLO-Pro (yolo_pro_worker.py:5667) and FOMO (training_worker.py). The
    # TrainingLogOutput panel renders training_history.log_lines, so SSD must
    # accumulate and stream these for the live log to populate.
    log_lines: List[str] = []
    # Register both buffers so the outer training_worker terminal handler can
    # flush every completed-epoch metric and log line into training_history when
    # the run ends in failed/cancelled (instead of losing the entries queued
    # since the last incremental commit). Both hold references to these same
    # lists, so later appends are visible at flush time.
    _register_live_buffer(job_id, epoch_metrics=epoch_metrics, log_lines=log_lines)

    # ── Best-val checkpointing + early stopping ────────────────────────────
    # The keep/exported model is the best-val-loss checkpoint, not the last
    # epoch — otherwise an overfit final epoch (train↓ while val↑) gets shipped.
    # Early stopping (val_loss, "lower is better") only engages AFTER the warmup
    # phase so the backbone fine-tune (phase 2) always gets a chance to run; it
    # also requires a validation set. Patience scales with the phase-2 budget.
    best_val_loss   = float("inf")
    best_weights    = None
    best_epoch      = 0
    epochs_no_improve = 0
    _MIN_DELTA      = 1e-4
    _phase2_epochs  = max(1, epochs - SSD_WARMUP_EPOCHS)
    es_patience     = max(5, math.ceil(_phase2_epochs / 3))
    # Early stopping is only meaningful with a val set; reflect the effective
    # state (this is what the frontend log string reads back).
    es_active       = bool(use_early_stop and val_ds is not None)
    if es_active:
        logger.info(
            f"[{job_id}] Early stopping: ENABLED — monitor=val_loss, "
            f"patience={es_patience} epochs (phase 2 only)"
        )
    else:
        _why = "no validation set" if (use_early_stop and val_ds is None) else "toggle off"
        logger.info(f"[{job_id}] Early stopping: DISABLED ({_why})")

    for epoch in range(1, epochs + 1):
        _raise_if_job_cancelled(job_id)

        # ── Phase switch: unfreeze backbone at warmup boundary ────────────
        if epoch == SSD_WARMUP_EPOCHS + 1:
            for _layer in model.layers:
                _layer.trainable = True
            optimizer = tf.keras.optimizers.Adam(learning_rate=lr * 0.1)
            logger.info(
                f"[{job_id}] SSD Phase 2 — backbone unfrozen, "
                f"fine-tuning all layers at lr={lr * 0.1:.2e}"
            )
            log_lines.append(
                f"Phase 2 - backbone unfrozen, fine-tuning all layers "
                f"at lr={lr * 0.1:.2e}"
            )

        train_losses = []
        for batch_imgs, batch_targets in train_ds:
            _raise_if_job_cancelled_fast(job_id)
            loss_val = _train_step(
                batch_imgs,
                batch_targets["loc"],
                batch_targets["cls"],
            )
            train_losses.append(float(loss_val))

        epoch_train_loss = float(np.mean(train_losses))
        history["train_loss"].append(epoch_train_loss)

        epoch_val_loss = None
        if val_ds is not None:
            val_losses = []
            for batch_imgs, batch_targets in val_ds:
                _raise_if_job_cancelled_fast(job_id)
                vl = _val_step(batch_imgs, batch_targets["loc"], batch_targets["cls"])
                val_losses.append(float(vl))
            epoch_val_loss = float(np.mean(val_losses))
            history["val_loss"].append(epoch_val_loss)

            # Best-val checkpoint: snapshot weights whenever val_loss improves
            # (tracked every epoch, including warmup, so we always keep the best
            # seen). Early-stop patience only accrues in phase 2.
            if epoch_val_loss < best_val_loss - _MIN_DELTA:
                best_val_loss     = epoch_val_loss
                best_weights      = model.get_weights()
                best_epoch        = epoch
                epochs_no_improve = 0
            elif epoch > SSD_WARMUP_EPOCHS:
                epochs_no_improve += 1

        log_msg = (
            f"[{job_id}] Epoch {epoch}/{epochs}  "
            f"train_loss={epoch_train_loss:.4f}"
        )
        if epoch_val_loss is not None:
            log_msg += f"  val_loss={epoch_val_loss:.4f}"
        logger.info(log_msg)

        # Append the per-epoch live log line for the TrainingLogOutput panel,
        # mirroring YOLO-Pro's format (yolo_pro_worker.py:5667).
        _ep_live_line = f"Epoch {epoch}/{epochs} - train_loss: {epoch_train_loss:.4f}"
        if epoch_val_loss is not None:
            _ep_live_line += f" - val_loss: {epoch_val_loss:.4f}"
        log_lines.append(_ep_live_line)

        # Capture per-epoch loss for the Training Graphs. val_loss is required by
        # the epoch_metrics contract, so skip epochs that ran without a
        # validation pass rather than emitting a null val_loss.
        if epoch_val_loss is not None:
            epoch_metrics.append({
                "epoch":      epoch,
                "train_loss": epoch_train_loss,
                "val_loss":   epoch_val_loss,
            })

        # Persist live metrics every epoch (matching YOLO-Pro/FOMO cadence) so
        # the log panel and Training Graphs stream live. Top-level loss/val_loss
        # arrays feed TrainingLogOutput's fallback; log_lines feeds the live
        # panel; epoch_metrics feeds the graph (shape unchanged).
        try:
            job.training_history = _sanitize_json({
                "epoch": epoch,
                "train_loss": epoch_train_loss,
                # Top-level loss/val_loss are ARRAYS (epoch-indexed), matching
                # the YOLO-Pro/FOMO contract and feeding TrainingLogOutput's
                # fallback (TrainingLogOutput.tsx:58,108-109).
                "loss": list(history["train_loss"]),
                "val_loss": list(history["val_loss"]),
                "history": history,
                "log_lines": list(log_lines),
                "epoch_metrics": list(epoch_metrics),
                "is_ssd": True,
                "output_type": "object_detection",
                # Drives the frontend's "Early stopping" log line (honest, not
                # a hardcoded literal) — TrainingLogOutput.tsx.
                "early_stopping": es_active,
                "patience": es_patience,
            })
            db.commit()
        except Exception as _e:
            logger.warning(f"[{job_id}] Failed to persist metrics: {_e}")

        # ── Early stopping check (phase 2 only, val_loss monitor) ──────────
        if es_active and epochs_no_improve >= es_patience:
            logger.info(
                f"[{job_id}] Early stopping at epoch {epoch}/{epochs} — "
                f"no val_loss improvement for {epochs_no_improve} epochs "
                f"(best={best_val_loss:.4f} @ epoch {best_epoch})"
            )
            log_lines.append(
                f"Early stopping at epoch {epoch} - best val_loss "
                f"{best_val_loss:.4f} @ epoch {best_epoch} "
                f"(patience {es_patience})"
            )
            break

    # ── Restore best-val weights so eval (mAP) and all exported TFLite ─────
    # variants reflect the best checkpoint, not the final (possibly overfit)
    # epoch. Both the mAP evaluation below and the TFLite export downstream
    # use this in-memory `model`, so restoring here covers both.
    if best_weights is not None:
        model.set_weights(best_weights)
        logger.info(
            f"[{job_id}] Restored best-val weights from epoch {best_epoch} "
            f"(val_loss={best_val_loss:.4f}) for eval + export"
        )

    # ── 7b. Post-training detection evaluation (mAP) ──────────────────────
    # Mirrors YOLO-Pro: produces a `classification_report`-shaped dict the
    # evaluation endpoints already know how to surface as
    # `yolo_pro_detection_metrics`, so the existing object-detection UI
    # (mAP@50 headline, metrics table, per-class rows) renders without any
    # further frontend/API changes.
    def _ssd_evaluate_map(
        conf_threshold: float = 0.01,
        nms_iou: float = 0.45,
        max_dets: int = 100,
        model_fn=None,
        eval_batch_size: Optional[int] = None,
    ) -> dict:
        """Score a detector on the validation set.

        ``model_fn`` / ``eval_batch_size`` exist so the post-export per-variant
        eval can drive this SAME metric code with a TFLite-backed stand-in
        instead of the in-memory Keras model — identical decode, NMS, IoU
        matching and AP math, so float32/int8 numbers are directly comparable.
        Defaults preserve the original in-training behaviour exactly.
        """
        _model_fn = model_fn if model_fn is not None else model
        # Reuse YOLO-Pro's COCO helpers so per-class AP, mAP@[.50:.95], mAP@75,
        # per-area AP, and AR@maxDets are computed identically to YOLO-Pro.
        from app.ml.yolo_pro_worker import (
            _box_iou_matrix,
            _compute_ap_at_iou,
            _compute_extra_coco_metrics,
        )

        n_te = int(X_test_r.shape[0]) if X_test_r is not None and len(X_test_r) > 0 else 0
        has_gt = any(
            (bl is not None and any((b.get("label_id") or b.get("label")) in label_map for b in bl))
            for bl in (boxes_test or [])
        )
        if n_te == 0 or not has_gt:
            return {
                "yolo_pro_eval_status": "no_data",
                "yolo_pro_eval_error":  "Test set is empty or has no annotated boxes.",
                "map": None, "map50": None, "map75": None,
                "precision": None, "recall": None, "per_class": {},
                "detailed_metrics": {},
            }
        try:
            anchors_tf = tf.convert_to_tensor(anchors_np, dtype=tf.float32)

            IOU_THRESHOLDS_COCO = np.linspace(0.50, 0.95, 10)
            all_preds       = [[] for _ in range(n_classes)]                      # primary IoU 0.50
            all_preds_multi = [[[] for _ in range(n_classes)] for _ in IOU_THRESHOLDS_COCO]
            gt_count        = np.zeros(n_classes, dtype=np.int32)
            per_image_data: List[dict] = []

            # TFLite variants have a fixed batch-1 input signature, so the
            # per-variant eval passes eval_batch_size=1.
            bs = max(1, int(eval_batch_size or batch_size))
            for start in range(0, n_te, bs):
                # Stop promptly if user cancels mid-evaluation.  Evaluation
                # on large test sets can take tens of seconds and previously
                # ran to completion regardless of cancel.
                _raise_if_job_cancelled_fast(job_id)
                xb = X_test_r[start:start + bs]
                outs = _model_fn(xb, training=False)
                box_pred = outs["box_predictions"]                                # (B, A, 4)
                cls_pred = outs["cls_predictions"]                                # (B, A, C)
                decoded  = _ssd_decode_boxes(box_pred, anchors_tf).numpy()        # y1,x1,y2,x2
                scores_b = tf.sigmoid(cls_pred).numpy()                           # (B, A, C)

                for bi in range(xb.shape[0]):
                    gi = start + bi
                    gt_list = boxes_test[gi] if gi < len(boxes_test) else None
                    gt_yxyx, gt_classes = _boxes_to_array(gt_list or [], label_map)

                    # Convert both GT and predictions to YOLO-Pro canonical order [x1,y1,x2,y2]
                    gt_xyxy = gt_yxyx[:, [1, 0, 3, 2]] if len(gt_yxyx) else gt_yxyx

                    img_boxes_yxyx  = decoded[bi]                                 # (A, 4)
                    img_boxes_xyxy  = img_boxes_yxyx[:, [1, 0, 3, 2]]
                    img_scores      = scores_b[bi]                                # (A, C)

                    # ── Per-class NMS to build the image's detection list ──
                    dets: list = []
                    for c in range(n_classes):
                        sc = img_scores[:, c]
                        keep = sc >= conf_threshold
                        if not np.any(keep):
                            continue
                        cb = img_boxes_xyxy[keep]
                        cs = sc[keep]
                        # tf.image.non_max_suppression expects [y1,x1,y2,x2]
                        cb_yx = cb[:, [1, 0, 3, 2]]
                        nms_idx = tf.image.non_max_suppression(
                            boxes=tf.constant(cb_yx, dtype=tf.float32),
                            scores=tf.constant(cs,   dtype=tf.float32),
                            max_output_size=max_dets,
                            iou_threshold=nms_iou,
                        ).numpy()
                        for k in nms_idx:
                            dets.append({
                                "class_idx": int(c),
                                "score":     float(cs[k]),
                                "box":       cb[k].tolist(),   # [x1,y1,x2,y2]
                            })
                    # Global top-max_dets sort (COCO primary eval spec)
                    dets.sort(key=lambda d: d["score"], reverse=True)
                    dets = dets[:max_dets]

                    # ── GT structure ──
                    gt_boxes_by_class: dict = {}
                    gt_boxes_all: list = []
                    for _gi in range(len(gt_xyxy)):
                        ci = int(gt_classes[_gi])
                        g  = gt_xyxy[_gi].astype(np.float32)
                        gt_count[ci] += 1
                        gt_boxes_by_class.setdefault(ci, []).append(g)
                        gt_boxes_all.append(g)

                    _img_gt_areas = []
                    for _c, _gts in gt_boxes_by_class.items():
                        for _g in _gts:
                            _img_gt_areas.append({
                                "class_idx": _c,
                                "area":      float((_g[2] - _g[0]) * (_g[3] - _g[1])),
                            })
                    per_image_data.append({
                        "dets":        dets,
                        "gt_by_class": {_c: [_g.tolist() for _g in _gts]
                                        for _c, _gts in gt_boxes_by_class.items()},
                        "gt_areas":    _img_gt_areas,
                    })

                    # ── Match at each IoU threshold ──
                    for iou_idx, iou_thr in enumerate(IOU_THRESHOLDS_COCO):
                        is_primary = abs(iou_thr - 0.50) < 0.001
                        gt_used: dict = {c: set() for c in gt_boxes_by_class}
                        for det in dets:
                            c     = det["class_idx"]
                            score = det["score"]
                            gts_c = gt_boxes_by_class.get(c, [])
                            matched = False
                            if gts_c:
                                pb = np.array([det["box"]], dtype=np.float32)
                                ga = np.array(gts_c,       dtype=np.float32)
                                ious = _box_iou_matrix(pb, ga)[0]
                                best = int(np.argmax(ious))
                                if ious[best] >= iou_thr and best not in gt_used.get(c, set()):
                                    matched = True
                                    gt_used.setdefault(c, set()).add(best)
                            all_preds_multi[iou_idx][c].append((score, matched))
                            if is_primary:
                                all_preds[c].append((score, matched))

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
            total_gt = int(gt_count.sum())

            # ── Per-class AP / Precision / Recall at primary IoU=0.50 ──
            per_class_ap, per_class_prec, per_class_rec = {}, {}, {}
            for c, cname in enumerate(label_names):
                preds_c = sorted(all_preds[c], key=lambda x: x[0], reverse=True)
                if not preds_c:
                    per_class_ap[cname]   = 0.0
                    per_class_prec[cname] = 0.0
                    per_class_rec[cname]  = 0.0
                    continue
                s = np.array([p[0] for p in preds_c])
                m = np.array([p[1] for p in preds_c])
                ap = _compute_ap_at_iou(s, m, int(gt_count[c]))
                per_class_ap[cname] = round(float(ap), 6)
                tp_cum = np.cumsum(m.astype(np.float32))
                fp_cum = np.cumsum((~m).astype(np.float32))
                rec_arr = tp_cum / (int(gt_count[c]) + 1e-9)
                pre_arr = tp_cum / (tp_cum + fp_cum + 1e-9)
                f1_arr  = 2 * pre_arr * rec_arr / (pre_arr + rec_arr + 1e-9)
                bi = int(np.argmax(f1_arr)) if len(f1_arr) else 0
                per_class_prec[cname] = round(float(pre_arr[bi]), 6) if len(pre_arr) else 0.0
                per_class_rec[cname]  = round(float(rec_arr[bi]), 6) if len(rec_arr) else 0.0

            map50     = float(np.mean(list(per_class_ap.values())))   if per_class_ap   else None
            mean_prec = float(np.mean(list(per_class_prec.values()))) if per_class_prec else None
            mean_rec  = float(np.mean(list(per_class_rec.values())))  if per_class_rec  else None

            # ── mAP@[0.50:0.95] + mAP@75 via the same multi-IoU accumulator ──
            ap_per_iou_per_class: list = []
            ap75_per_class: list       = []
            for iou_idx, iou_thr in enumerate(IOU_THRESHOLDS_COCO):
                aps_at_thr = []
                for c in range(n_classes):
                    preds_c = sorted(all_preds_multi[iou_idx][c], key=lambda x: x[0], reverse=True)
                    if not preds_c:
                        aps_at_thr.append(0.0)
                        continue
                    s = np.array([p[0] for p in preds_c])
                    m = np.array([p[1] for p in preds_c])
                    aps_at_thr.append(_compute_ap_at_iou(s, m, int(gt_count[c])))
                ap_per_iou_per_class.append(aps_at_thr)
                if abs(iou_thr - 0.75) < 0.001:
                    ap75_per_class = aps_at_thr

            map_coco = float(np.mean(ap_per_iou_per_class)) if ap_per_iou_per_class else None
            map75    = float(np.mean(ap75_per_class))        if ap75_per_class        else None

            # ── Extended COCO metrics (per-area / max-detections) ──
            _extra = _compute_extra_coco_metrics(
                per_image_data, n_classes, H, W, IOU_THRESHOLDS_COCO
            )

            per_class_out = {
                cname: {
                    "ap":        per_class_ap.get(cname),
                    "precision": per_class_prec.get(cname),
                    "recall":    per_class_rec.get(cname),
                }
                for cname in label_names
            }

            logger.info(
                f"[{job_id}] SSD eval — "
                f"mAP50={(map50 or 0):.4f}  mAP={(map_coco or 0):.4f}  mAP75={(map75 or 0):.4f}  "
                f"P={(mean_prec or 0):.4f}  R={(mean_rec or 0):.4f}"
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
                "total_predicted_cells": int(total_preds),
                "gt_cells": total_gt,
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
        except _CancelledError:
            # Cancellation must NOT be swallowed by the generic eval-failure
            # handler — let it propagate up so the outer training_worker
            # exception path preserves status='cancelled'.
            raise
        except Exception as _ex:
            logger.error(f"[{job_id}] SSD eval failed: {_ex}", exc_info=True)
            return {
                "yolo_pro_eval_status": "inference_failed",
                "yolo_pro_eval_error":  str(_ex),
                "map": None, "map50": None, "map75": None,
                "precision": None, "recall": None, "per_class": {},
                "detailed_metrics": {},
            }

    logger.info(f"[{job_id}] Running SSD post-training detection evaluation …")
    ssd_eval_report = _ssd_evaluate_map()
    try:
        job.classification_report = _sanitize_json(ssd_eval_report)
        db.commit()
    except Exception as _e:
        logger.warning(f"[{job_id}] Failed to persist SSD eval report: {_e}")

    # Stop before the long export step if the user has cancelled during eval.
    _raise_if_job_cancelled(job_id)

    # ── 8. Export to TFLite (float32 + int8) ───────────────────────────────
    logger.info(f"[{job_id}] Exporting MobileNetV2 SSD to TFLite (float32 + int8) …")

    tflite_f32_bytes: Optional[bytes] = None
    tflite_int8_bytes: Optional[bytes] = None
    int8_status: str = "available"
    int8_error: Optional[str] = None
    f32_error: Optional[str] = None

    with tempfile.TemporaryDirectory() as tmp_dir:
        saved_model_dir = os.path.join(tmp_dir, "saved_model")
        model.export(saved_model_dir)

        # ---- float32 ------------------------------------------------------
        try:
            conv_f32 = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
            conv_f32.optimizations               = []
            conv_f32.target_spec.supported_types = [tf.float32]
            conv_f32.target_spec.supported_ops   = [tf.lite.OpsSet.TFLITE_BUILTINS]
            conv_f32.allow_custom_ops            = False
            conv_f32.experimental_new_converter  = True
            tflite_f32_bytes = conv_f32.convert()
            logger.info(f"[{job_id}] float32 TFLite: {len(tflite_f32_bytes)//1024} KB")
        except Exception as _e:
            f32_error = str(_e)
            logger.warning(f"[{job_id}] float32 TFLite export failed: {_e}")

        # ---- int8 (full-integer quant, uint8 input preserved) -------------
        try:
            rep_src = X_train_r if len(X_train_r) > 0 else X_test_r
            n_calib = int(min(100, len(rep_src)))
            if n_calib <= 0:
                raise RuntimeError("no calibration samples available for int8 quant")

            def _rep_gen():
                for i in range(n_calib):
                    yield [rep_src[i:i + 1].astype(np.uint8)]

            conv_i8 = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
            conv_i8.optimizations              = [tf.lite.Optimize.DEFAULT]
            conv_i8.representative_dataset     = _rep_gen
            conv_i8.target_spec.supported_ops  = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            conv_i8.inference_input_type       = tf.int8
            conv_i8.inference_output_type      = tf.int8
            conv_i8.experimental_new_converter = True
            tflite_int8_bytes = conv_i8.convert()
            logger.info(f"[{job_id}] int8 TFLite: {len(tflite_int8_bytes)//1024} KB")
        except Exception as _e:
            int8_status = "failed"
            int8_error = str(_e)
            logger.warning(f"[{job_id}] int8 TFLite export failed: {_e}")

    # ── 8b. Decoded TFLite variant for PXE (single (1, N, 6) output) ───────
    # Wrap the trained model with SSDDecodeLayer (anchors baked in as graph
    # constants) so the production .pxe runner receives a decoded
    # [x1, y1, x2, y2, score, class_id] tensor in normalized coordinates —
    # the identical contract YOLO-Pro's decoded export produces. NMS and
    # confidence filtering are applied by the runner, not in-graph. The raw
    # training graph (model.py), anchors and box coder are NOT modified; this
    # is a pure export-time wrapper (see MobileNetV2 SSD/decode_export.py).
    tflite_decoded_bytes: Optional[bytes] = None
    try:
        from decode_export import export_ssd_decoded_tflite as _export_ssd_decoded
        tflite_decoded_bytes = _export_ssd_decoded(model, input_shape=MV2_SSD_INPUT_SHAPE)
        logger.info(f"[{job_id}] decoded float32 TFLite: {len(tflite_decoded_bytes)//1024} KB")
    except Exception as _e:
        logger.warning(f"[{job_id}] decoded TFLite export failed: {_e}")

    # ── 9. Persist artefacts + TrainedModel rows ───────────────────────────
    final_train_loss = history["train_loss"][-1] if history["train_loss"] else None
    # Report the BEST val_loss (the checkpoint we actually keep/export), not the
    # last epoch — the last epoch may be overfit. Falls back to the last value
    # if no improvement was ever recorded (e.g. single-epoch run).
    final_val_loss   = (
        best_val_loss if math.isfinite(best_val_loss)
        else (history["val_loss"][-1] if history["val_loss"] else None)
    )
    # Surface best val_loss as the job's scalar metric so the Evaluation page's
    # "Best val_loss" tile shows a real number (job.best_loss was previously
    # never set for SSD → "N/A"). Read by evaluation.py / trained_models.py.
    if final_val_loss is not None:
        job.best_loss = float(final_val_loss)

    # channel_order: MV2 SSD always consumes 3-channel RGB images (MV2_SSD_INPUT_C == 3).
    # normalize_input: the SSD preprocessing pipeline scales uint8 pixels to [0, 1]
    #   before the MobileNetV2 backbone's built-in [-1, 1] rescaling layer, so the
    #   runtime loader must NOT apply a second normalisation pass.
    _common_meta = {
        "architecture":      MV2_SSD_ARCHITECTURE,
        "output_type":       "ssd_detection",
        "model_type":        "ssd_detection",
        "input_shape":       list(MV2_SSD_INPUT_SHAPE),
        "num_classes":       n_classes,
        "label_names":       label_names,
        "normalize_input":   True,
        "channel_order":     "rgb",
        "version":           1,
        "final_train_loss":  final_train_loss,
        "final_val_loss":    final_val_loss,
    }

    def _persist_variant(tflite_bytes: Optional[bytes], variant: str, quantized: bool, filename: str,
                         extra_meta: Optional[dict] = None) -> Optional[int]:
        if tflite_bytes is None:
            return None
        key = f"models/{impulse.project_id}/{job_id}/{filename}"
        try:
            storage.upload_bytes(tflite_bytes, key, content_type="application/octet-stream")
        except Exception as _e:
            logger.warning(f"[{job_id}] Storage upload failed for {variant}: {_e}")
            key = ""
        try:
            tm = TrainedModel(
                training_job_id=job.id,
                version="1",
                format="tflite",
                storage_key=key,
                file_size_bytes=len(tflite_bytes),
                model_metadata=_sanitize_json({
                    **_common_meta,
                    "variant":   variant,
                    "quantized": quantized,
                    **(extra_meta or {}),
                }),
                created_at=datetime.utcnow(),
            )
            db.add(tm)
            db.commit()
            db.refresh(tm)
            logger.info(f"[{job_id}] TrainedModel ({variant}) created: id={tm.id}")
            return len(tflite_bytes)
        except Exception as _e:
            logger.warning(f"[{job_id}] Failed to create TrainedModel ({variant}) record: {_e}")
            return len(tflite_bytes)

    # Last chance to abort before persisting artifacts.  If we get past this
    # point and then write status='completed', any cancel that arrived during
    # export would be overwritten — so re-check the DB authoritatively.
    _assert_not_cancelled_before_completing(job_id)

    f32_size  = _persist_variant(tflite_f32_bytes,  "float32", False, "mobilenetv2_ssd_float32.tflite")
    int8_size = _persist_variant(tflite_int8_bytes, "int8",    True,  "mobilenetv2_ssd_int8.tflite")
    # Decoded variant consumed by the PXE deployment path (deployment_worker.
    # _get_ssd_decoded_tflite_bytes). It self-normalizes uint8 input in-graph,
    # so normalize_input=False; output is a single decoded (1, N, 6) tensor.
    _persist_variant(
        tflite_decoded_bytes, "decoded_float32", False,
        "mobilenetv2_ssd_decoded_float32.tflite",
        extra_meta={
            "output_format":   "decoded_xyxy_score_cls",
            "normalize_input": False,
        },
    )

    tflite_size = f32_size or int8_size or 0

    # ── Record int8 export outcome ───────────────────────────────────────────
    # training_history is written per-epoch inside the training loop, so the
    # export outcome has to be merged in afterwards.  Without this the Model
    # panel has no source for "why is int8 missing" and falls back to a generic
    # message (trained_models.get_model_panel reads history["int8_error"]).
    # training_worker and yolo_pro_worker already record these keys; SSD did not.
    try:
        _hist = dict(job.training_history or {})
        _hist["int8_status"] = int8_status
        _hist["int8_error"]  = int8_error
        job.training_history = _sanitize_json(_hist)
        db.commit()
    except Exception as _e:
        logger.warning(f"[{job_id}] Failed to record int8 export status: {_e}")

    # ── Per-variant export evaluation ────────────────────────────────────────
    # Score each exported TFLite variant independently on the same validation
    # set, through the SAME _ssd_evaluate_map metric code used in training, so
    # the Model Version dropdown reports each variant's real numbers rather
    # than reusing float32's.  Post-export only; failures never fail the job.
    try:
        from app.ml.variant_eval import evaluate_exported_variants

        def _score_ssd_variant(_runner) -> dict:
            return _ssd_evaluate_map(model_fn=_runner, eval_batch_size=1)

        evaluate_exported_variants(
            db=db,
            job_id=job.id,
            scorer=_score_ssd_variant,
            variant_bytes={
                "float32":         tflite_f32_bytes,
                "int8":            tflite_int8_bytes,
                # Exported for the PXE deployment path.  Its graph already
                # applies decode + sigmoid + NMS, while _ssd_evaluate_map takes
                # the RAW box/cls heads and decodes them itself — scoring it
                # here would double-decode and report meaningless numbers.
                "decoded_float32": tflite_decoded_bytes,
            },
            skip_reasons={
                "decoded_float32": (
                    "The decoded SSD export applies box decoding and NMS inside "
                    "the graph, so it cannot be scored by the raw-head detection "
                    "eval. Its weights are identical to the float32 variant — "
                    "see that variant's metrics."
                ),
            },
            runner_kwargs={
                "output_type": "ssd_detection",
                "num_classes": n_classes,
                "input_h":     MV2_SSD_INPUT_SHAPE[0],
                "input_w":     MV2_SSD_INPUT_SHAPE[1],
            },
            export_errors={"float32": f32_error, "int8": int8_error},
            log_prefix=f"[{job_id}] ",
        )
    except Exception as _ve_exc:
        logger.warning(
            f"[{job_id}] per-variant export eval skipped: {_ve_exc}", exc_info=True
        )

    # Final guard immediately before marking completed.
    _assert_not_cancelled_before_completing(job_id)
    job.status = JobStatus.completed
    job.completed_at = datetime.utcnow()
    # Pointer flip — successful completion of any run kind becomes the
    # new Active Model. Never cleared elsewhere.
    _promote_run_to_active(job.impulse_id, job.id)
    db.commit()

    return _sanitize_json({
        "architecture":      MV2_SSD_ARCHITECTURE,
        "output_type":       "ssd_detection",
        "num_classes":       n_classes,
        "input_shape":       list(MV2_SSD_INPUT_SHAPE),
        "epochs":            epochs,
        "final_train_loss":  final_train_loss,
        "final_val_loss":    final_val_loss,
        "tflite_size_bytes": tflite_size,
        "label_names":       label_names,
    })
