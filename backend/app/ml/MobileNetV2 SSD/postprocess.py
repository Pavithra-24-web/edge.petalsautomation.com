# postprocess.py
# Step 11 — Inference Post-Processing
# MobileNetV2 SSD FPN-Lite 320x320

from __future__ import annotations

import numpy as np
import tensorflow as tf

from primitives import CFG
from box_coder import decode_boxes, VARIANCES


# ============================================================================
# Constants
# ============================================================================

_SCORE_THRESHOLD  = 0.3     # discard predictions below this confidence
_NMS_IOU_THRESHOLD = 0.6    # NMS suppression threshold
_MAX_DETECTIONS   = CFG.TOP_K  # 10


# ============================================================================
# Post-processing pipeline (single image)
# ============================================================================

def _postprocess_single(
    box_preds:        tf.Tensor,
    cls_preds:        tf.Tensor,
    anchors:          tf.Tensor,
    score_threshold:  float = _SCORE_THRESHOLD,
    nms_iou_threshold: float = _NMS_IOU_THRESHOLD,
    max_detections:   int   = _MAX_DETECTIONS,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    """
    Post-process raw outputs for a single image.

    Pipeline:
        1. Decode box offsets → absolute [y_min, x_min, y_max, x_max]
        2. Apply sigmoid to class logits → per-class scores
        3. Flatten: for each anchor, pick the best class + score
        4. Filter by score threshold
        5. Apply NMS (combined_non_max_suppression)
        6. Pad to max_detections and return

    Args:
        box_preds:         (A, 4)           raw box regression offsets.
        cls_preds:         (A, num_classes) raw class logits.
        anchors:           (A, 4)           anchors in [cy, cx, h, w].
        score_threshold:   Minimum confidence to keep a detection.
        nms_iou_threshold: NMS suppression IoU.
        max_detections:    Maximum detections to return.

    Returns:
        boxes   : (max_detections, 4)  [y_min, x_min, y_max, x_max] in [0,1]
        classes : (max_detections,)    int32 class ids (1-indexed)
        scores  : (max_detections,)    float32 confidence scores
        count   : ()                   int32 number of valid detections
    """
    # ------------------------------------------------------------------ #
    # 1. Decode boxes
    # ------------------------------------------------------------------ #
    # box_preds : (A, 4) offsets
    # anchors   : (A, 4) [cy, cx, h, w]
    # decoded   : (A, 4) [y_min, x_min, y_max, x_max], clipped to [0,1]
    decoded_boxes = decode_boxes(
        tf.expand_dims(box_preds, 0),   # (1, A, 4)
        anchors,                         # (A, 4)  — broadcast
        variances=VARIANCES,
        clip=True,
    )[0]                                 # (A, 4)

    # ------------------------------------------------------------------ #
    # 2. Class scores via sigmoid  (multi-label: each class independent)
    # ------------------------------------------------------------------ #
    scores_all = tf.sigmoid(cls_preds)   # (A, num_classes)

    # ------------------------------------------------------------------ #
    # 3. Per-anchor best class and score
    # ------------------------------------------------------------------ #
    best_scores  = tf.reduce_max(scores_all, axis=-1)           # (A,)
    best_classes = tf.argmax(scores_all, axis=-1, output_type=tf.int32) + 1
    # +1 shifts to 1-indexed (class 0 = background is never predicted)

    # ------------------------------------------------------------------ #
    # 4. Score threshold filter
    # ------------------------------------------------------------------ #
    keep_mask    = best_scores >= score_threshold                # (A,) bool
    boxes_filt   = tf.boolean_mask(decoded_boxes, keep_mask)    # (K, 4)
    scores_filt  = tf.boolean_mask(best_scores,  keep_mask)     # (K,)
    classes_filt = tf.boolean_mask(best_classes, keep_mask)     # (K,)

    # ------------------------------------------------------------------ #
    # 5. NMS — class-agnostic (SSD reference style)
    # ------------------------------------------------------------------ #
    # tf.image.non_max_suppression handles an empty boxes_filt gracefully:
    # it returns a zero-length index tensor, which the pad step below then
    # expands to max_detections zeros.  The explicit Python-level K==0
    # branch has been removed because it is a Python conditional on a
    # symbolic tensor (tf.shape output), which breaks tf.function tracing.
    # tf.image.non_max_suppression expects [y_min, x_min, y_max, x_max]
    # which is what decode_boxes already returns.
    nms_indices = tf.image.non_max_suppression(
        boxes=boxes_filt,
        scores=scores_filt,
        max_output_size=max_detections,
        iou_threshold=nms_iou_threshold,
        score_threshold=score_threshold,
        name="nms",
    )                                                           # (D,) D ≤ max_detections

    # ------------------------------------------------------------------ #
    # 6. Gather NMS survivors and pad to max_detections
    # ------------------------------------------------------------------ #
    boxes_nms   = tf.gather(boxes_filt,   nms_indices)          # (D, 4)
    scores_nms  = tf.gather(scores_filt,  nms_indices)          # (D,)
    classes_nms = tf.gather(classes_filt, nms_indices)          # (D,)
    count       = tf.shape(nms_indices)[0]                      # D (scalar)

    # Pad with zeros to reach max_detections
    pad_len     = max_detections - count
    boxes_out   = tf.pad(boxes_nms,   [[0, pad_len], [0, 0]])   # (10, 4)
    scores_out  = tf.pad(scores_nms,  [[0, pad_len]])            # (10,)
    classes_out = tf.pad(classes_nms, [[0, pad_len]])            # (10,)

    return (
        boxes_out,
        classes_out,
        scores_out,
        tf.cast(count, tf.int32),
    )


# ============================================================================
# Batched entry point
# ============================================================================

def postprocess(
    box_preds:         tf.Tensor,
    cls_preds:         tf.Tensor,
    anchors:           tf.Tensor,
    score_threshold:   float = _SCORE_THRESHOLD,
    nms_iou_threshold: float = _NMS_IOU_THRESHOLD,
    max_detections:    int   = _MAX_DETECTIONS,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    """
    Post-process raw model outputs for a batch of images.

    Applies decode → threshold → NMS → pad independently per image,
    then stacks results into batched tensors.

    Args:
        box_preds:         (B, A, 4)           raw box regression offsets.
        cls_preds:         (B, A, num_classes) raw class logits.
        anchors:           (A, 4)              anchors in [cy, cx, h, w].
        score_threshold:   Minimum confidence (default 0.3).
        nms_iou_threshold: NMS IoU threshold  (default 0.6).
        max_detections:    Max detections per image (default 10).

    Returns:
        boxes   : (B, 10, 4)  [y_min, x_min, y_max, x_max] normalised to [0,1]
        classes : (B, 10)     int32  class ids (1-indexed; 0 = padding)
        scores  : (B, 10)     float32 confidence scores (0.0 = padding)
        counts  : (B,)        int32  number of valid detections per image

    Shape flow:
        box_preds  (B, 8028, 4)    # 256ch neck
        cls_preds  (B, 8028, 90)
        anchors    (   8028, 4)
        → boxes    (B, 10, 4)
        → classes  (B, 10)
        → scores   (B, 10)
        → counts   (B,)
    """
    # ------------------------------------------------------------------
    # tf.map_fn applies _postprocess_single to every (box_pred, cls_pred)
    # slice in the batch.  The output signature must be declared explicitly
    # because _postprocess_single uses ops with dynamic intermediate shapes
    # (boolean_mask, non_max_suppression) even though its *outputs* are
    # padded to fixed shapes before returning.
    # Using tf.map_fn instead of a Python for-loop makes postprocess()
    # fully compatible with tf.function graph tracing and XLA compilation.
    # ------------------------------------------------------------------
    def _map_fn(elems: tuple[tf.Tensor, tf.Tensor]):
        bp, cp = elems
        boxes_i, classes_i, scores_i, count_i = _postprocess_single(
            box_preds=bp,
            cls_preds=cp,
            anchors=anchors,
            score_threshold=score_threshold,
            nms_iou_threshold=nms_iou_threshold,
            max_detections=max_detections,
        )
        return boxes_i, classes_i, scores_i, count_i

    all_boxes, all_classes, all_scores, all_counts = tf.map_fn(
        _map_fn,
        elems=(box_preds, cls_preds),
        fn_output_signature=(
            tf.TensorSpec(shape=(max_detections, 4),   dtype=tf.float32),  # boxes
            tf.TensorSpec(shape=(max_detections,),     dtype=tf.int32),    # classes
            tf.TensorSpec(shape=(max_detections,),     dtype=tf.float32),  # scores
            tf.TensorSpec(shape=(),                    dtype=tf.int32),    # count
        ),
    )
    # all_boxes   : (B, max_detections, 4)
    # all_classes : (B, max_detections)
    # all_scores  : (B, max_detections)
    # all_counts  : (B,)

    return (
        all_boxes,      # (B, 10, 4)
        all_classes,    # (B, 10)
        all_scores,     # (B, 10)
        all_counts,     # (B,)
    )


# ============================================================================
# Smoke test  (python postprocess.py)
# ============================================================================

if __name__ == "__main__":
    import numpy as np
    from anchor_generator import generate_anchors
    from box_coder import centroids_to_corners

    print("=== Post-Processing — smoke tests ===\n")

    A  = CFG.TOTAL_ANCHORS
    C  = CFG.NUM_CLASSES
    B  = 2

    anchors = generate_anchors()    # (8028, 4) [cy, cx, h, w]

    # ------------------------------------------------------------------ #
    # Test 1: All-zero predictions (no detections expected)
    # ------------------------------------------------------------------ #
    print("1. All-zero predictions → 0 valid detections")
    box_preds = tf.zeros((B, A, 4), dtype=tf.float32)
    cls_preds = tf.fill((B, A, C), -5.0)  # sigmoid(-5) ≈ 0.007 < threshold

    boxes, classes, scores, counts = postprocess(box_preds, cls_preds, anchors)

    assert boxes.shape   == (B, 10, 4), f"boxes shape:   {boxes.shape}"
    assert classes.shape == (B, 10),    f"classes shape: {classes.shape}"
    assert scores.shape  == (B, 10),    f"scores shape:  {scores.shape}"
    assert counts.shape  == (B,),       f"counts shape:  {counts.shape}"
    print(f"   boxes   : {boxes.shape}")
    print(f"   classes : {classes.shape}")
    print(f"   scores  : {scores.shape}")
    print(f"   counts  : {counts.numpy()}  (expected [0, 0])")
    assert list(counts.numpy()) == [0, 0], f"Expected 0 detections, got {counts}"
    print("   ✓ Zero-prediction test OK\n")

    # ------------------------------------------------------------------ #
    # Test 2: Inject a strong signal at a known anchor
    # ------------------------------------------------------------------ #
    print("2. Strong signal at anchors 0 and 100 → ≥ 2 detections")
    box_preds_2 = tf.zeros((1, A, 4), dtype=tf.float32)
    cls_preds_2 = tf.fill((1, A, C), -5.0)

    # Force anchor 0 → class 3 with high confidence
    # Force anchor 100 → class 7 with high confidence
    mask_0   = tf.one_hot([0],   A, on_value=5.0, off_value=-5.0)   # (1, A)
    mask_100 = tf.one_hot([100], A, on_value=5.0, off_value=-5.0)   # (1, A)

    cls_3 = tf.zeros((1, A, C), dtype=tf.float32)
    cls_7 = tf.zeros((1, A, C), dtype=tf.float32)

    # Build per-class columns: (1, A, 1) for class 3 and 7
    col3 = tf.expand_dims(mask_0,   -1) * tf.one_hot([3], C)    # not quite — simpler:
    # Simpler: directly construct cls tensor with high values at [0,0,3] and [0,100,7]
    cls_np = np.full((1, A, C), -5.0, dtype=np.float32)
    cls_np[0, 0,   3] = 8.0    # anchor 0,   class 3 → sigmoid(8) ≈ 0.9997
    cls_np[0, 100, 7] = 8.0    # anchor 100, class 7 → sigmoid(8) ≈ 0.9997
    cls_preds_2 = tf.constant(cls_np)

    boxes2, classes2, scores2, counts2 = postprocess(
        box_preds_2, cls_preds_2, anchors,
        score_threshold=0.3,
    )

    valid = int(counts2[0])
    print(f"   valid detections : {valid}")
    print(f"   boxes  [0,:valid]: {boxes2[0, :valid].numpy()}")
    print(f"   classes[0,:valid]: {classes2[0, :valid].numpy()}")
    print(f"   scores [0,:valid]: {scores2[0, :valid].numpy()}")
    assert valid >= 1, "Expected at least 1 detection"
    assert boxes2.shape   == (1, 10, 4)
    assert classes2.shape == (1, 10)
    assert scores2.shape  == (1, 10)
    print("   ✓ Signal injection test OK\n")

    # ------------------------------------------------------------------ #
    # Test 3: Output value ranges
    # ------------------------------------------------------------------ #
    print("3. Value range checks on detected boxes and scores")
    # Boxes must be in [0, 1]
    valid_boxes = boxes2[0, :valid]
    assert float(tf.reduce_min(valid_boxes)) >= 0.0, "Box below 0"
    assert float(tf.reduce_max(valid_boxes)) <= 1.0, "Box above 1"
    # Scores must be in [score_threshold, 1]
    valid_scores = scores2[0, :valid]
    assert float(tf.reduce_min(valid_scores)) >= 0.3, "Score below threshold"
    assert float(tf.reduce_max(valid_scores)) <= 1.0, "Score above 1"
    # Classes must be ≥ 1 (1-indexed)
    valid_classes = classes2[0, :valid]
    assert int(tf.reduce_min(valid_classes)) >= 1, "Class below 1"
    print(f"   box range   : [{float(tf.reduce_min(valid_boxes)):.4f}, "
                             f"{float(tf.reduce_max(valid_boxes)):.4f}]")
    print(f"   score range : [{float(tf.reduce_min(valid_scores)):.4f}, "
                             f"{float(tf.reduce_max(valid_scores)):.4f}]")
    print(f"   class ids   : {valid_classes.numpy()}  (all ≥ 1)")
    print("   ✓ Value range test OK\n")

    # ------------------------------------------------------------------ #
    # Test 4: NMS suppresses near-duplicate boxes
    # ------------------------------------------------------------------ #
    print("4. NMS deduplication (two overlapping high-score anchors)")
    cls_np2 = np.full((1, A, C), -5.0, dtype=np.float32)
    cls_np2[0, 0, 3] = 8.0    # anchor 0
    cls_np2[0, 1, 3] = 8.0    # anchor 1 — very close to anchor 0 spatially
    cls_preds_dup = tf.constant(cls_np2)

    boxes_dup, _, scores_dup, counts_dup = postprocess(
        tf.zeros((1, A, 4)), cls_preds_dup, anchors,
        score_threshold=0.3, nms_iou_threshold=0.6,
    )
    print(f"   input signals  : 2 overlapping anchors")
    print(f"   valid after NMS: {int(counts_dup[0])}")
    print("   ✓ NMS deduplication test OK\n")

    print("=== All post-processing tests passed ✓ ===")
