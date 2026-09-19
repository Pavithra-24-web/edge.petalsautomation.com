# box_coder.py
# Step 9 — SSD Box Encoding and Decoding
# MobileNetV2 SSD FPN-Lite 320x320

from __future__ import annotations

import numpy as np
import tensorflow as tf

from primitives import CFG


# ============================================================================
# Constants
# ============================================================================

# SSD variance scaling factors — applied to offset targets during encoding
# and divided out during decoding.  Scaling by 10/10/5/5 amplifies the
# regression targets so that gradients are better conditioned during training.
#
# Convention (index maps to):
#   VARIANCES[0] → cy  (centre-y offset)
#   VARIANCES[1] → cx  (centre-x offset)
#   VARIANCES[2] → h   (log height offset)
#   VARIANCES[3] → w   (log width offset)
VARIANCES = tf.constant([10.0, 10.0, 5.0, 5.0], dtype=tf.float32)

# Minimum ground-truth box size to prevent log(0) in height/width encoding
_MIN_SIZE = 1e-6


# ============================================================================
# Format helpers
# ============================================================================

def corners_to_centroids(boxes: tf.Tensor) -> tf.Tensor:
    """
    Convert boxes from [y_min, x_min, y_max, x_max] → [cy, cx, h, w].

    Args:
        boxes: (..., 4) float32 in corner format.

    Returns:
        (..., 4) float32 in centroid format.
    """
    y_min, x_min, y_max, x_max = tf.unstack(boxes, axis=-1)
    cy = (y_min + y_max) / 2.0
    cx = (x_min + x_max) / 2.0
    h  = y_max - y_min
    w  = x_max - x_min
    return tf.stack([cy, cx, h, w], axis=-1)


def centroids_to_corners(boxes: tf.Tensor) -> tf.Tensor:
    """
    Convert boxes from [cy, cx, h, w] → [y_min, x_min, y_max, x_max].

    Args:
        boxes: (..., 4) float32 in centroid format.

    Returns:
        (..., 4) float32 in corner format.
    """
    cy, cx, h, w = tf.unstack(boxes, axis=-1)
    y_min = cy - h / 2.0
    x_min = cx - w / 2.0
    y_max = cy + h / 2.0
    x_max = cx + w / 2.0
    return tf.stack([y_min, x_min, y_max, x_max], axis=-1)


# ============================================================================
# Encoder
# ============================================================================

def encode_boxes(
    gt_boxes:  tf.Tensor,
    anchors:   tf.Tensor,
    variances: tf.Tensor = VARIANCES,
) -> tf.Tensor:
    """
    Encode ground-truth boxes as offsets relative to anchors.

    SSD encoding formula (all values in [cy, cx, h, w] space):

        t_cy = (gt_cy - anc_cy) / anc_h  * variance[0]
        t_cx = (gt_cx - anc_cx) / anc_w  * variance[1]
        t_h  = log(gt_h  / anc_h)        * variance[2]
        t_w  = log(gt_w  / anc_w)        * variance[3]

    The offset targets (t_cy, t_cx, t_h, t_w) are what the box regression
    head learns to predict.  Multiplying by variance amplifies the targets
    so that the loss has a more numerically stable gradient magnitude.

    Args:
        gt_boxes:  (N, 4) or (B, N, 4) float32 in [cy, cx, h, w] format.
                   Ground-truth boxes already in centroid format.
        anchors:   (N, 4) float32 in [cy, cx, h, w] format.
                   Broadcast-compatible with gt_boxes.
        variances: (4,) scaling factors — default [10, 10, 5, 5].

    Returns:
        (N, 4) or (B, N, 4) float32 encoded offset targets.

    Shape flow:
        gt_boxes  : (N, 4)
        anchors   : (N, 4)
        → encoded : (N, 4)
    """
    gt_cy,  gt_cx,  gt_h,  gt_w  = tf.unstack(gt_boxes, axis=-1)
    anc_cy, anc_cx, anc_h, anc_w = tf.unstack(anchors,  axis=-1)

    # Guard against degenerate anchors or GT boxes
    anc_h = tf.maximum(anc_h, _MIN_SIZE)
    anc_w = tf.maximum(anc_w, _MIN_SIZE)
    gt_h  = tf.maximum(gt_h,  _MIN_SIZE)
    gt_w  = tf.maximum(gt_w,  _MIN_SIZE)

    # Centre offsets (normalised by anchor size)
    t_cy = (gt_cy - anc_cy) / anc_h
    t_cx = (gt_cx - anc_cx) / anc_w

    # Size offsets (log-space)
    t_h = tf.math.log(gt_h / anc_h)
    t_w = tf.math.log(gt_w / anc_w)

    # Apply variance scaling
    v_cy, v_cx, v_h, v_w = tf.unstack(variances, axis=-1)
    t_cy = t_cy * v_cy
    t_cx = t_cx * v_cx
    t_h  = t_h  * v_h
    t_w  = t_w  * v_w

    return tf.stack([t_cy, t_cx, t_h, t_w], axis=-1)


# ============================================================================
# Decoder
# ============================================================================

def decode_boxes(
    box_preds: tf.Tensor,
    anchors:   tf.Tensor,
    variances: tf.Tensor = VARIANCES,
    clip:      bool      = True,
) -> tf.Tensor:
    """
    Decode raw box regression predictions back to absolute coordinates.

    Inverse of encode_boxes.  Given predicted offsets (t_cy, t_cx, t_h, t_w)
    and the matching anchors, recovers the predicted box in
    [y_min, x_min, y_max, x_max] format.

    Decoding formula:

        pred_cy = t_cy / variance[0] * anc_h + anc_cy
        pred_cx = t_cx / variance[1] * anc_w + anc_cx
        pred_h  = exp(t_h / variance[2]) * anc_h
        pred_w  = exp(t_w / variance[3]) * anc_w

    Then convert centroid → corners:

        y_min = pred_cy - pred_h / 2
        x_min = pred_cx - pred_w / 2
        y_max = pred_cy + pred_h / 2
        x_max = pred_cx + pred_w / 2

    Numerical stability:
        - exp() is clamped via tf.clip_by_value on the exponent before
          exponentiation to prevent overflow on large regression outputs.
        - Optionally clip decoded boxes to [0, 1].

    Args:
        box_preds: (N, 4) or (B, N, 4) float32 predicted offsets.
                   Output of the SSD box regression head.
        anchors:   (N, 4) float32 in [cy, cx, h, w] format.
                   Same anchors used at encode time.
        variances: (4,) scaling factors — must match encode_boxes variances.
        clip:      If True, clips decoded corners to [0, 1].  Default True.

    Returns:
        (N, 4) or (B, N, 4) float32 decoded boxes in
        [y_min, x_min, y_max, x_max] format.

    Shape flow:
        box_preds : (B, 8028, 4)
        anchors   : (   8028, 4)  → broadcast to (B, 8028, 4)
        → decoded : (B, 8028, 4) in corner format
    """
    t_cy, t_cx, t_h, t_w        = tf.unstack(box_preds, axis=-1)
    anc_cy, anc_cx, anc_h, anc_w = tf.unstack(anchors,  axis=-1)

    v_cy, v_cx, v_h, v_w = tf.unstack(variances, axis=-1)

    # Centre coordinates
    pred_cy = (t_cy / v_cy) * anc_h + anc_cy
    pred_cx = (t_cx / v_cx) * anc_w + anc_cx

    # Size — clamp exponent to prevent overflow (exp(>88) → inf on float32)
    _MAX_EXP = 10.0   # exp(10) ≈ 22,026 × anchor_size → already huge
    t_h_safe = tf.clip_by_value(t_h / v_h, -_MAX_EXP, _MAX_EXP)
    t_w_safe = tf.clip_by_value(t_w / v_w, -_MAX_EXP, _MAX_EXP)
    pred_h = tf.exp(t_h_safe) * anc_h
    pred_w = tf.exp(t_w_safe) * anc_w

    # Convert centroid → corners
    y_min = pred_cy - pred_h / 2.0
    x_min = pred_cx - pred_w / 2.0
    y_max = pred_cy + pred_h / 2.0
    x_max = pred_cx + pred_w / 2.0

    decoded = tf.stack([y_min, x_min, y_max, x_max], axis=-1)

    if clip:
        decoded = tf.clip_by_value(decoded, 0.0, 1.0)

    return decoded


# ============================================================================
# Round-trip helper (encode → decode, for verification)
# ============================================================================

def encode_decode_roundtrip(
    gt_boxes:  tf.Tensor,
    anchors:   tf.Tensor,
    variances: tf.Tensor = VARIANCES,
    atol:      float     = 1e-5,
) -> bool:
    """
    Verify that decode(encode(gt)) ≈ gt within numerical tolerance.

    Converts gt_boxes to corner format after decode so both sides are
    in the same [y_min, x_min, y_max, x_max] space for comparison.

    Args:
        gt_boxes:  (N, 4) float32 in [cy, cx, h, w] format.
        anchors:   (N, 4) float32 in [cy, cx, h, w] format.
        variances: Variance factors (must match encode/decode).
        atol:      Absolute tolerance for the allclose check.

    Returns:
        True if round-trip error is within atol everywhere.
    """
    encoded   = encode_boxes(gt_boxes, anchors, variances)
    decoded   = decode_boxes(encoded,  anchors, variances, clip=False)
    recovered = centroids_to_corners(decoded)
    original  = centroids_to_corners(gt_boxes)

    max_err = tf.reduce_max(tf.abs(recovered - original)).numpy()
    ok      = bool(max_err < atol)
    return ok, float(max_err)


# ============================================================================
# Smoke test  (python box_coder.py)
# ============================================================================

if __name__ == "__main__":
    from anchor_generator import generate_anchors

    print("=== Box Coder — smoke tests ===\n")

    # --- 1. Format conversion round-trip ---
    print("1. Format conversion (centroids ↔ corners)")
    boxes_centroids = tf.constant([[0.5, 0.5, 0.4, 0.6],    # centred large box
                                   [0.2, 0.3, 0.1, 0.15]],  # small box
                                  dtype=tf.float32)
    corners  = centroids_to_corners(boxes_centroids)
    back     = corners_to_centroids(corners)
    max_err  = tf.reduce_max(tf.abs(back - boxes_centroids)).numpy()
    print(f"   centroids → corners → centroids  max_err={max_err:.2e}")
    assert max_err < 1e-6, f"Format conversion error too large: {max_err}"
    print("   ✓ format round-trip OK\n")

    # --- 2. Encode / decode round-trip with real anchors ---
    print("2. Encode → decode round-trip (real anchors)")
    anchors = generate_anchors()                            # (8028, 4)

    # Build synthetic GT boxes matched to the first 8 anchors
    # (small perturbation around each anchor centre)
    n = 8
    anc_sample = anchors[:n]                               # (8, 4)
    cy, cx, h, w = [anc_sample[:, i] for i in range(4)]
    gt_sample = tf.stack([
        cy + 0.02,           # shift cy slightly
        cx - 0.01,           # shift cx slightly
        h  * 1.2,            # scale h up
        w  * 0.8,            # scale w down
    ], axis=-1)              # (8, 4)

    ok, max_err = encode_decode_roundtrip(gt_sample, anc_sample)
    print(f"   max round-trip error: {max_err:.2e}")
    assert ok, f"Round-trip error too large: {max_err:.2e}"
    print("   ✓ encode → decode round-trip OK\n")

    # --- 3. Batch decode (head output shape) ---
    print("3. Batch decode  (B, 8028, 4) → (B, 8028, 4)")
    B = 2
    fake_preds = tf.random.normal((B, CFG.TOTAL_ANCHORS, 4), stddev=0.1)
    decoded    = decode_boxes(fake_preds, anchors, clip=True)
    assert decoded.shape == (B, CFG.TOTAL_ANCHORS, 4), \
        f"Shape mismatch: {decoded.shape}"
    assert float(tf.reduce_min(decoded)) >= 0.0, "Clipped min below 0"
    assert float(tf.reduce_max(decoded)) <= 1.0, "Clipped max above 1"
    print(f"   output shape : {decoded.shape}")
    print(f"   value range  : [{float(tf.reduce_min(decoded)):.4f}, "
          f"{float(tf.reduce_max(decoded)):.4f}]")
    print("   ✓ batch decode OK\n")

    # --- 4. Numerical stability — extreme predictions ---
    print("4. Numerical stability (extreme offsets)")
    extreme = tf.constant([[100.0, -100.0, 50.0, -50.0]], dtype=tf.float32)
    extreme_anchors = tf.constant([[0.5, 0.5, 0.3, 0.3]], dtype=tf.float32)
    decoded_extreme = decode_boxes(extreme, extreme_anchors, clip=True)
    has_nan = tf.reduce_any(tf.math.is_nan(decoded_extreme))
    has_inf = tf.reduce_any(tf.math.is_inf(decoded_extreme))
    assert not bool(has_nan), "NaN in extreme decode output"
    assert not bool(has_inf), "Inf in extreme decode output"
    print(f"   decoded extreme: {decoded_extreme.numpy()}")
    print("   ✓ no NaN / Inf on extreme inputs\n")

    print("✓ All box coder tests passed")
