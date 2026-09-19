# losses.py
# Step 10 — Anchor Matching and Training Losses
# MobileNetV2 SSD FPN-Lite 320x320

from __future__ import annotations

import tensorflow as tf

from primitives import CFG
from box_coder import encode_boxes, corners_to_centroids, VARIANCES


# ============================================================================
# Constants
# ============================================================================

_POS_IOU_THRESHOLD  = 0.5    # anchor is positive  if IoU >= this
_NEG_IOU_THRESHOLD  = 0.4    # anchor is negative  if IoU <  this
                             # Gap [0.4, 0.5) creates a neutral band: anchors
                             # whose best-GT IoU falls in this range are neither
                             # positive nor negative — they are ignored (flag=-1)
                             # in both the box loss and hard-negative mining.
                             # This prevents ambiguous anchors that partially
                             # overlap an object from corrupting either the
                             # foreground signal (too low IoU to be a reliable
                             # positive) or the background signal (too high IoU
                             # to be a clean negative).  The original value of
                             # 0.5 made neg_threshold == pos_threshold, collapsing
                             # the neutral band to zero and forcing every anchor
                             # to be either positive or negative with no buffer.
_NEG_POS_RATIO      = 3      # hard negative mining: up to 3× as many negatives as positives
_HUBER_DELTA        = 1.0    # Smooth L1 inflection point


# ============================================================================
# IoU computation
# ============================================================================

def compute_iou(
    boxes_a: tf.Tensor,
    boxes_b: tf.Tensor,
) -> tf.Tensor:
    """
    Compute pairwise IoU between two sets of boxes.

    Both inputs must be in [y_min, x_min, y_max, x_max] format,
    normalised to [0, 1].

    Args:
        boxes_a: (N, 4) float32.
        boxes_b: (M, 4) float32.

    Returns:
        (N, M) float32 IoU matrix.

    Shape flow:
        boxes_a : (N, 4)
        boxes_b : (M, 4)
        → iou   : (N, M)
    """
    # Expand for broadcasting: (N,1,4) vs (1,M,4)
    a = tf.expand_dims(boxes_a, 1)   # (N, 1, 4)
    b = tf.expand_dims(boxes_b, 0)   # (1, M, 4)

    # Intersection
    inter_y_min = tf.maximum(a[..., 0], b[..., 0])
    inter_x_min = tf.maximum(a[..., 1], b[..., 1])
    inter_y_max = tf.minimum(a[..., 2], b[..., 2])
    inter_x_max = tf.minimum(a[..., 3], b[..., 3])

    inter_h = tf.maximum(inter_y_max - inter_y_min, 0.0)
    inter_w = tf.maximum(inter_x_max - inter_x_min, 0.0)
    inter   = inter_h * inter_w                           # (N, M)

    # Areas
    area_a = (boxes_a[:, 2] - boxes_a[:, 0]) * (boxes_a[:, 3] - boxes_a[:, 1])  # (N,)
    area_b = (boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1])  # (M,)

    union = (tf.expand_dims(area_a, 1) +
             tf.expand_dims(area_b, 0) - inter)           # (N, M)

    return tf.math.divide_no_nan(inter, union)            # (N, M)


# ============================================================================
# Anchor matching
# ============================================================================

def match_anchors(
    anchors_xyxy:  tf.Tensor,
    gt_boxes_xyxy: tf.Tensor,
    gt_labels:     tf.Tensor,
    pos_threshold: float = _POS_IOU_THRESHOLD,
    neg_threshold: float = _NEG_IOU_THRESHOLD,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """
    Match anchors to ground-truth boxes using IoU.

    Assignment rules:
        IoU >= pos_threshold → positive  (label = gt_class,  box target = encoded GT)
        IoU <  neg_threshold → negative  (label = 0,          box target = zeros)
        otherwise            → neutral   (label = -1,         ignored in loss)

    Additionally: for each GT box, the single anchor with the highest IoU
    is forced positive regardless of threshold (ensures every GT is covered).

    Args:
        anchors_xyxy:  (A, 4) float32 anchors in [y_min, x_min, y_max, x_max].
                       Convert from [cy,cx,h,w] using centroids_to_corners().
        gt_boxes_xyxy: (G, 4) float32 GT boxes in [y_min, x_min, y_max, x_max].
        gt_labels:     (G,)   int32   GT class labels (1-indexed, 0 = background).
        pos_threshold: IoU threshold for positive assignment (default 0.5).
        neg_threshold: IoU threshold for negative assignment (default 0.4).
                       Anchors with IoU in [neg_threshold, pos_threshold) fall
                       into the neutral band and are ignored in the loss.

    Returns:
        matched_labels:   (A,) int32  — class label per anchor (-1 = neutral, 0 = bg)
        matched_boxes:    (A, 4) float32 — encoded box offsets per anchor (zeros for bg)
        anchor_flags:     (A,) int32  — +1 positive, 0 negative, -1 neutral

    Shape flow:
        anchors_xyxy  : (A, 4)
        gt_boxes_xyxy : (G, 4)
        → iou_matrix  : (A, G)
        → matched_*   : (A, *)
    """
    A = tf.shape(anchors_xyxy)[0]
    G = tf.shape(gt_boxes_xyxy)[0]

    # ---- IoU matrix --------------------------------------------------------
    # Guard: if there are no GT boxes every anchor is background (flag=0).
    # Use tf.cond instead of a Python if-branch so the function is fully
    # traceable inside tf.function / tf.map_fn without graph-mode errors.
    def _no_gt():
        return (tf.zeros([A], dtype=tf.int32),
                tf.zeros([A, 4], dtype=tf.float32),
                tf.zeros([A], dtype=tf.int32))

    def _with_gt():
        return _match_with_gt(
            anchors_xyxy, gt_boxes_xyxy, gt_labels,
            pos_threshold, neg_threshold, A, G,
        )

    return tf.cond(tf.equal(G, 0), _no_gt, _with_gt)


def _match_with_gt(
    anchors_xyxy:  tf.Tensor,
    gt_boxes_xyxy: tf.Tensor,
    gt_labels:     tf.Tensor,
    pos_threshold: float,
    neg_threshold: float,
    A:             tf.Tensor,
    G:             tf.Tensor,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """Inner body of match_anchors for the G > 0 case."""
    iou = compute_iou(anchors_xyxy, gt_boxes_xyxy)   # (A, G)

    # Best GT for each anchor
    best_gt_iou  = tf.reduce_max(iou, axis=1)         # (A,)
    best_gt_idx  = tf.argmax(iou, axis=1, output_type=tf.int32)   # (A,)

    # Best anchor for each GT (force-positive) — only index needed
    best_anc_idx = tf.argmax(iou, axis=0, output_type=tf.int32)   # (G,)

    # ---- Assignment flags --------------------------------------------------
    # Start all as neutral (-1)
    flags = tf.fill([A], -1)

    # Negative: IoU < neg_threshold
    neg_mask = best_gt_iou < neg_threshold
    flags = tf.where(neg_mask, tf.zeros([A], dtype=tf.int32), flags)

    # Positive: IoU >= pos_threshold
    pos_mask = best_gt_iou >= pos_threshold
    flags = tf.where(pos_mask, tf.ones([A], dtype=tf.int32), flags)

    # Force-positive: best anchor per GT box (even if below threshold)
    flags = tf.tensor_scatter_nd_update(
        flags,
        tf.expand_dims(best_anc_idx, 1),
        tf.ones([G], dtype=tf.int32),
    )

    # ---- Labels ------------------------------------------------------------
    # Gather GT label for each anchor's best matching GT
    matched_gt_labels = tf.gather(gt_labels, best_gt_idx)    # (A,)

    # Negatives get label 0 (background), neutrals will be masked in loss
    anchor_labels = tf.where(flags == 1, matched_gt_labels,
                             tf.zeros([A], dtype=tf.int32))

    # ---- Box targets (encoded offsets) -------------------------------------
    # Convert anchors and GT boxes to centroid format for encoding
    anchors_cwh   = corners_to_centroids(anchors_xyxy)         # (A, 4) [cy,cx,h,w]
    matched_gt_boxes_xyxy = tf.gather(gt_boxes_xyxy, best_gt_idx)  # (A, 4)
    matched_gt_cwh = corners_to_centroids(matched_gt_boxes_xyxy)   # (A, 4) [cy,cx,h,w]

    encoded_boxes = encode_boxes(matched_gt_cwh, anchors_cwh, VARIANCES)  # (A, 4)

    # Zero out encoded boxes for non-positive anchors
    pos_float = tf.cast(tf.equal(flags, 1), tf.float32)
    matched_boxes = encoded_boxes * tf.expand_dims(pos_float, 1)   # (A, 4)

    return anchor_labels, matched_boxes, flags


# ============================================================================
# Loss functions
# ============================================================================

def smooth_l1_loss(
    predictions: tf.Tensor,
    targets:     tf.Tensor,
    delta:       float = _HUBER_DELTA,
) -> tf.Tensor:
    """
    Smooth L1 (Huber) loss, element-wise.

    Formula:
        |x| < delta  →  0.5 * x²
        |x| ≥ delta  →  delta * (|x| - 0.5 * delta)

    Args:
        predictions: (..., 4) float32 predicted box offsets.
        targets:     (..., 4) float32 encoded box targets.
        delta:       Inflection point (default 1.0).

    Returns:
        (..., 4) float32 element-wise loss values.
    """
    diff    = predictions - targets
    abs_diff = tf.abs(diff)
    loss     = tf.where(
        abs_diff < delta,
        0.5 * tf.square(diff),
        delta * (abs_diff - 0.5 * delta),
    )
    return loss


def sigmoid_bce_loss(
    logits:  tf.Tensor,
    labels:  tf.Tensor,
) -> tf.Tensor:
    """
    Sigmoid binary cross-entropy loss (per-class), element-wise.

    Applies sigmoid activation implicitly via the numerically stable
    tf.nn.sigmoid_cross_entropy_with_logits kernel, which computes:

        loss = max(logit, 0) - logit * label + log(1 + exp(-|logit|))

    This is standard binary CE, treating each of the `num_classes` output
    channels as an independent binary classifier (multi-label formulation).
    Used here with one-hot targets so only positives contribute a foreground
    signal; hard-negative mining then selects which background anchors
    contribute to the loss via their all-zeros target row.

    TODO: Replace with true sigmoid focal loss (Lin et al., RetinaNet 2017)
          to down-weight the easy negatives that dominate training and
          focus gradient on hard examples:
              p_t   = sigmoid(logit) if label == 1 else 1 - sigmoid(logit)
              FL    = -(1 - p_t) ** gamma * log(p_t)
          Typical hyper-parameters: alpha=0.25, gamma=2.0.
          True focal loss would replace or supplement hard-negative mining.

    Args:
        logits: (..., num_classes) float32 raw class predictions.
        labels: (..., num_classes) float32 one-hot targets in {0, 1}.

    Returns:
        (..., num_classes) float32 element-wise binary CE loss.
    """
    return tf.nn.sigmoid_cross_entropy_with_logits(
        labels=labels,
        logits=logits,
    )


# ============================================================================
# Hard Negative Mining
# ============================================================================

def hard_negative_mining(
    cls_losses:  tf.Tensor,
    flags:       tf.Tensor,
    neg_pos_ratio: int = _NEG_POS_RATIO,
) -> tf.Tensor:
    """
    Select hard negatives: the neg_pos_ratio × num_positives highest-loss
    negative anchors per image.

    Positives are always included.  Neutral anchors (flags == -1) are
    excluded entirely.  Among the remaining negatives, only those with the
    largest classification loss are kept.

    Args:
        cls_losses: (A,) float32 — per-anchor classification loss (summed over classes).
        flags:      (A,) int32   — +1 positive, 0 negative, -1 neutral.
        neg_pos_ratio: Maximum ratio of negatives to positives.

    Returns:
        (A,) bool — True for anchors that contribute to the loss.

    Shape flow:
        cls_losses : (A,)
        flags      : (A,)
        → mask     : (A,) bool
    """
    pos_mask = tf.equal(flags, 1)                             # (A,) bool
    neg_mask = tf.equal(flags, 0)                             # (A,) bool

    num_pos  = tf.reduce_sum(tf.cast(pos_mask, tf.int32))     # scalar
    num_neg  = tf.minimum(
        tf.reduce_sum(tf.cast(neg_mask, tf.int32)),
        num_pos * neg_pos_ratio,
    )
    num_neg  = tf.maximum(num_neg, 1)                         # always keep ≥ 1 neg

    # Rank negatives by their classification loss (highest → hardest).
    # Mask non-negatives with -inf so they rank last and are never selected.
    neg_losses = tf.where(neg_mask, cls_losses, -1e9 * tf.ones_like(cls_losses))

    # Build a dense hard-negative mask using a double-argsort rank.
    # sorted_neg_rank[i] == 0 means anchor i has the highest neg loss.
    sorted_neg_rank = tf.argsort(tf.argsort(-neg_losses))    # (A,) rank per anchor
    hard_neg_mask   = neg_mask & (sorted_neg_rank < num_neg) # (A,) bool

    return pos_mask | hard_neg_mask                           # (A,) bool


# ============================================================================
# Combined SSD Loss
# ============================================================================

def ssd_loss(
    box_preds:       tf.Tensor,
    cls_preds:       tf.Tensor,
    anchors_xyxy:    tf.Tensor,
    gt_boxes_batch:  list[tf.Tensor],
    gt_labels_batch: list[tf.Tensor],
    num_classes:     int   = CFG.NUM_CLASSES,
    neg_pos_ratio:   int   = _NEG_POS_RATIO,
    max_gt:          int   = 100,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """
    Full SSD training loss for a batch.

    Processes each image via tf.map_fn (graph-safe, tf.function-compatible).
    GT boxes/labels are padded to a fixed `max_gt` rows with a boolean
    validity mask so every per-image tensor has the same static shape.

    Args:
        box_preds:       (B, A, 4) float32 — raw box regression outputs.
        cls_preds:       (B, A, num_classes) float32 — raw class logits.
        anchors_xyxy:    (A, 4) float32 — anchors in corner format.
        gt_boxes_batch:  List[Tensor] of length B, each (G_i, 4) xyxy float32.
        gt_labels_batch: List[Tensor] of length B, each (G_i,) int32.
        num_classes:     Number of classes (default 90).
        neg_pos_ratio:   Hard negative mining ratio (default 3).
        max_gt:          Maximum GT boxes per image after padding (default 100).

    Returns:
        total_loss  : scalar float32 — box_loss + cls_loss.
        box_loss    : scalar float32 — mean Smooth L1 over positives.
        cls_loss    : scalar float32 — mean sigmoid CE over pos+hard-neg.

    Shape flow:
        box_preds  : (B, A, 4)
        cls_preds  : (B, A, num_classes)
        → losses   : scalars
    """
    # ------------------------------------------------------------------
    # 1. Pad GT boxes and labels to fixed max_gt rows.
    #    Carry a boolean validity mask so the per-image fn can slice only
    #    the real GT entries and pass them to match_anchors.
    # ------------------------------------------------------------------
    def _pad_gt(gt_boxes: tf.Tensor, gt_labels: tf.Tensor):
        """Pad (G, 4) and (G,) to (max_gt, 4) / (max_gt,) + bool mask."""
        G = tf.shape(gt_boxes)[0]
        pad_b = tf.maximum(max_gt - G, 0)
        boxes_pad  = tf.pad(gt_boxes,  [[0, pad_b], [0, 0]])[:max_gt]   # (max_gt, 4)
        labels_pad = tf.pad(gt_labels, [[0, pad_b]])[:max_gt]            # (max_gt,)
        valid_mask = tf.sequence_mask(tf.minimum(G, max_gt),
                                      maxlen=max_gt)                     # (max_gt,) bool
        return boxes_pad, labels_pad, valid_mask

    padded_boxes_list  = []
    padded_labels_list = []
    valid_masks_list   = []
    for gt_b, gt_l in zip(gt_boxes_batch, gt_labels_batch):
        pb, pl, vm = _pad_gt(gt_b, gt_l)
        padded_boxes_list.append(pb)
        padded_labels_list.append(pl)
        valid_masks_list.append(vm)

    # Stack into batched tensors — now all shapes are static along last dims.
    gt_boxes_pad  = tf.stack(padded_boxes_list,  axis=0)   # (B, max_gt, 4)
    gt_labels_pad = tf.stack(padded_labels_list, axis=0)   # (B, max_gt)
    gt_valid_mask = tf.stack(valid_masks_list,   axis=0)   # (B, max_gt) bool

    # ------------------------------------------------------------------
    # 2. Define the per-image loss function to be mapped.
    #    Inputs arrive as a tuple slice along axis 0.
    # ------------------------------------------------------------------
    def _loss_single(
        elems: tuple,
    ) -> tuple[tf.Tensor, tf.Tensor]:
        box_pred_i, cls_pred_i, boxes_pad_i, labels_pad_i, valid_i = elems
        # box_pred_i  : (A, 4)
        # cls_pred_i  : (A, num_classes)
        # boxes_pad_i : (max_gt, 4)
        # labels_pad_i: (max_gt,)
        # valid_i     : (max_gt,) bool

        # Recover only the valid GT rows (boolean_mask gives dynamic shape,
        # but match_anchors handles that via tf.shape internally).
        gt_boxes_i  = tf.boolean_mask(boxes_pad_i,  valid_i)   # (G_i, 4)
        gt_labels_i = tf.boolean_mask(labels_pad_i, valid_i)   # (G_i,)

        # ---- Match anchors to GT ----------------------------------------
        anchor_labels, target_boxes, flags = match_anchors(
            anchors_xyxy, gt_boxes_i, gt_labels_i,
        )
        # anchor_labels : (A,) int32
        # target_boxes  : (A, 4) float32 encoded offsets
        # flags         : (A,) int32  +1/0/-1

        pos_mask = tf.equal(flags, 1)   # (A,) bool
        num_pos  = tf.maximum(
            tf.reduce_sum(tf.cast(pos_mask, tf.float32)), 1.0
        )

        # ---- Box loss (Smooth L1, positives only) -----------------------
        box_loss_all = tf.reduce_sum(
            smooth_l1_loss(box_pred_i, target_boxes), axis=-1
        )                                                       # (A,)
        box_loss_pos = tf.reduce_sum(
            tf.where(pos_mask, box_loss_all, tf.zeros_like(box_loss_all))
        )
        box_loss_i = box_loss_pos / num_pos                     # scalar

        # ---- Class loss (sigmoid CE + hard negative mining) -------------
        one_hot = tf.one_hot(
            anchor_labels, depth=num_classes, dtype=tf.float32
        )                                                       # (A, C)

        # Zero out the one-hot for every non-positive anchor.
        # anchor_labels is set to 0 (background) for negatives/neutrals by
        # match_anchors, so tf.one_hot would place a spurious 1 at index 0
        # for those anchors.  Classes are 1-indexed (1–num_classes), so
        # index 0 is unused — but the loss still sees a non-zero target.
        # Masking with pos_float collapses those rows to all-zeros, giving
        # negatives the correct "no foreground" target before HNM selects
        # among them, and prevents neutral anchors from contributing at all.
        pos_float = tf.cast(pos_mask, tf.float32)               # (A,)
        one_hot   = one_hot * tf.expand_dims(pos_float, axis=1) # (A, C)

        cls_loss_all = tf.reduce_sum(
            sigmoid_bce_loss(cls_pred_i, one_hot), axis=-1
        )                                                       # (A,)

        keep_mask = hard_negative_mining(
            cls_loss_all, flags, neg_pos_ratio
        )                                                       # (A,) bool
        cls_loss_kept = tf.reduce_sum(
            tf.where(keep_mask, cls_loss_all, tf.zeros_like(cls_loss_all))
        )
        cls_loss_i = cls_loss_kept / num_pos                    # scalar

        return box_loss_i, cls_loss_i

    # ------------------------------------------------------------------
    # 3. Map over the batch dimension.
    #    tf.map_fn requires fixed output dtypes/shapes declared up-front.
    # ------------------------------------------------------------------
    box_losses, cls_losses = tf.map_fn(
        _loss_single,
        elems=(
            box_preds,       # (B, A, 4)
            cls_preds,       # (B, A, num_classes)
            gt_boxes_pad,    # (B, max_gt, 4)
            gt_labels_pad,   # (B, max_gt)
            gt_valid_mask,   # (B, max_gt)
        ),
        fn_output_signature=(tf.float32, tf.float32),
    )
    # box_losses, cls_losses : (B,) each

    box_loss   = tf.reduce_mean(box_losses)
    cls_loss   = tf.reduce_mean(cls_losses)
    total_loss = box_loss + cls_loss

    return total_loss, box_loss, cls_loss


# ============================================================================
# Smoke test  (python losses.py)
# ============================================================================

if __name__ == "__main__":
    import numpy as np
    from anchor_generator import generate_anchors
    from box_coder import centroids_to_corners

    print("=== Losses — smoke tests ===\n")

    # Generate anchors in corner format
    anchors_cwh  = generate_anchors()                        # (8028, 4) [cy,cx,h,w]
    anchors_xyxy = centroids_to_corners(anchors_cwh)         # (8028, 4) [ymin,xmin,ymax,xmax]
    A = CFG.TOTAL_ANCHORS

    # --- 1. IoU ---
    print("1. IoU matrix")
    boxes_a = tf.constant([[0.1, 0.1, 0.5, 0.5],
                            [0.4, 0.4, 0.9, 0.9]], dtype=tf.float32)
    boxes_b = tf.constant([[0.1, 0.1, 0.5, 0.5],
                            [0.0, 0.0, 1.0, 1.0]], dtype=tf.float32)
    iou = compute_iou(boxes_a, boxes_b)
    print(f"   iou[0,0] = {iou[0,0]:.4f}  (expected 1.0)")
    print(f"   iou[0,1] = {iou[0,1]:.4f}  (expected 0.16)")
    assert abs(float(iou[0, 0]) - 1.0) < 1e-5, "IoU self-match failed"
    print("   ✓ IoU OK\n")

    # --- 2. Matching ---
    print("2. Anchor matching")
    gt_boxes  = tf.constant([[0.1, 0.1, 0.5, 0.5]], dtype=tf.float32)  # (1,4)
    gt_labels = tf.constant([3], dtype=tf.int32)
    lbl, tgt, flags = match_anchors(anchors_xyxy, gt_boxes, gt_labels)
    num_pos = int(tf.reduce_sum(tf.cast(tf.equal(flags, 1), tf.int32)))
    num_neg = int(tf.reduce_sum(tf.cast(tf.equal(flags, 0), tf.int32)))
    num_neu = int(tf.reduce_sum(tf.cast(tf.equal(flags,-1), tf.int32)))
    print(f"   positives : {num_pos:>6,}")
    print(f"   negatives : {num_neg:>6,}")
    print(f"   neutrals  : {num_neu:>6,}")
    assert num_pos >= 1, "No positives found"
    print("   ✓ Matching OK\n")

    # --- 3. Hard negative mining ---
    print("3. Hard negative mining (3:1 ratio)")
    fake_cls_loss = tf.random.uniform([A])
    keep = hard_negative_mining(fake_cls_loss, flags, neg_pos_ratio=3)
    n_keep_pos = int(tf.reduce_sum(tf.cast(keep & tf.equal(flags,  1), tf.int32)))
    n_keep_neg = int(tf.reduce_sum(tf.cast(keep & tf.equal(flags,  0), tf.int32)))
    print(f"   kept positives : {n_keep_pos:>6,}")
    print(f"   kept negatives : {n_keep_neg:>6,}")
    assert n_keep_neg <= n_keep_pos * 3 + 1, "Hard neg ratio violated"
    print("   ✓ Hard negative mining OK\n")

    # --- 4. Full batch loss ---
    print("4. Full SSD loss  (B=2)")
    B = 2
    box_preds = tf.random.normal((B, A, 4))
    cls_preds = tf.random.normal((B, A, CFG.NUM_CLASSES))

    gt_boxes_batch  = [
        tf.constant([[0.1, 0.1, 0.5, 0.5], [0.6, 0.6, 0.9, 0.9]], dtype=tf.float32),
        tf.constant([[0.2, 0.3, 0.7, 0.8]], dtype=tf.float32),
    ]
    gt_labels_batch = [
        tf.constant([1, 2], dtype=tf.int32),
        tf.constant([5],    dtype=tf.int32),
    ]

    total, box_l, cls_l = ssd_loss(
        box_preds, cls_preds, anchors_xyxy,
        gt_boxes_batch, gt_labels_batch,
    )
    print(f"   total_loss : {float(total):.4f}")
    print(f"   box_loss   : {float(box_l):.4f}")
    print(f"   cls_loss   : {float(cls_l):.4f}")
    assert float(total) > 0.0, "Loss should be positive"
    assert not np.isnan(float(total)), "NaN in loss"
    print("   ✓ SSD loss OK\n")

    print("✓ All loss tests passed")
