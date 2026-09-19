# model.py
# Step 12 — Full Model Assembly
# MobileNetV2 SSD FPN-Lite 320x320

from __future__ import annotations

import numpy as np
import tensorflow as tf
from tensorflow import keras

from primitives import CFG
from preprocessing import NormalizationLayer, build_input_layer
from fpn_lite import build_fpn_lite
from ssd_head import build_ssd_head


# ============================================================================
# Model Assembly
# ============================================================================

def build_model(
    input_shape: tuple = CFG.INPUT_SHAPE,
    num_classes: int   = CFG.NUM_CLASSES,
    neck_channels: int = CFG.NECK_CHANNELS,
) -> keras.Model:
    """
    Assemble the full MobileNetV2 SSD FPN-Lite 320×320 detection model.

    Pipeline:
        uint8 input (B, 320, 320, 3)
          → NormalizationLayer        → float32 [-1, 1]
          → MobileNetV2 Backbone      → C3, C4, C5
          → FPN-Lite Neck             → [F3, F4, F5, F6, F7, F8]
          → SSD Shared Head           → box_preds, cls_preds

    NMS and decoding are intentionally excluded from the graph.
    Call postprocess.postprocess() on the outputs at inference time.

    Shape flow:
        Input         : (B, 320, 320,   3)  uint8
        Normalized    : (B, 320, 320,   3)  float32 [-1, 1]
        C3            : (B,  40,  40,  32)
        C4            : (B,  20,  20,  96)
        C5            : (B,  10,  10, 1280)  after last_conv
        F3            : (B,  40,  40, 256)
        F4            : (B,  20,  20, 256)
        F5            : (B,  10,  10, 256)

        F6            : (B,   5,   5, 256)
        F7            : (B,   3,   3, 256)
        F8            : (B,   2,   2, 256)
        box_preds     : (B, 8028,   4)     raw regression offsets
        cls_preds     : (B, 8028,  90)     raw class logits

    Args:
        input_shape:   (H, W, C) — fixed at (320, 320, 3).
        num_classes:   Number of detection classes (default 90 COCO).
        neck_channels: FPN-Lite uniform channel width (default 256).

    Returns:
        keras.Model with:
            inputs  : single uint8 tensor (B, 320, 320, 3)
            outputs : dict {"box_predictions": ..., "cls_predictions": ...}
    """

    # ------------------------------------------------------------------ #
    # 1. Input + Normalization
    # ------------------------------------------------------------------ #
    inputs, normalized = build_input_layer(input_shape)
    # inputs     : keras.Input  (B, 320, 320, 3)  uint8
    # normalized : tf.Tensor    (B, 320, 320, 3)  float32 [-1, 1]

    # ------------------------------------------------------------------ #
    # 2. MobileNetV2 Backbone — pretrained ImageNet weights
    # ------------------------------------------------------------------ #
    # Use tf.keras.applications.MobileNetV2 with ImageNet weights to match
    # the OG Edge Impulse SSDLite MobileNetV2 320x320 pretrained backbone.
    # input_tensor=normalized connects our uint8→[-1,1] normalization layer
    # to the pretrained backbone which also expects float32 [-1, 1].
    _backbone = tf.keras.applications.MobileNetV2(
        input_tensor=normalized,
        alpha=1.0,
        include_top=False,
        weights='imagenet',
    )
    # Freeze the pretrained backbone: only the FPN-Lite neck and SSD head
    # (randomly initialised) are trained.  Fine-tuning MobileNetV2 at the
    # same lr as the head (1e-3) immediately corrupts ImageNet features and
    # produces worse metrics than a frozen backbone.  This matches OG EI
    # SSDLite behaviour for small detection datasets.
    _backbone.trainable = False
    C3 = _backbone.get_layer('block_5_add').output   # (B, 40, 40,   32) stride-8
    C4 = _backbone.get_layer('block_12_add').output  # (B, 20, 20,   96) stride-16
    C5 = _backbone.get_layer('out_relu').output       # (B, 10, 10, 1280) stride-32

    # ------------------------------------------------------------------ #
    # 3. FPN-Lite Neck
    # ------------------------------------------------------------------ #
    feature_maps = build_fpn_lite(C3, C4, C5, neck_channels=neck_channels)
    # [F3, F4, F5, F6, F7, F8]
    # F3 : (B, 40, 40, 256)
    # F4 : (B, 20, 20, 256)
    # F5 : (B, 10, 10, 256)
    # F6 : (B,  5,  5, 256)
    # F7 : (B,  3,  3, 256)
    # F8 : (B,  2,  2, 256)

    # ------------------------------------------------------------------ #
    # 4. SSD Shared Prediction Head
    # ------------------------------------------------------------------ #
    box_preds, cls_preds = build_ssd_head(
        feature_maps,
        num_classes=num_classes,
        anchors_per_loc=CFG.ANCHORS_PER_LOC,
    )
    # box_preds : (B, 8028,  4)
    # cls_preds : (B, 8028, 90)

    # ------------------------------------------------------------------ #
    # 5. Build and return the Keras Model
    # ------------------------------------------------------------------ #
    model = keras.Model(
        inputs=inputs,
        outputs={
            "box_predictions": box_preds,
            "cls_predictions": cls_preds,
        },
        name="mobilenetv2_ssd_fpnlite_320x320",
    )

    return model


# ============================================================================
# Convenience: model summary by sub-module
# ============================================================================

def print_layer_shapes(model: keras.Model, batch_size: int = 1) -> None:
    """
    Print the output shape of every named sub-module in the assembled model.

    Useful for confirming every tap and feature map has the expected shape
    without running a full forward pass.
    """
    important_keywords = [
        "normalization", "stem",
        "stage_1", "stage_2", "stage_3", "stage_4", "stage_5",
        "stage_6", "stage_7",
        "lateral_C3", "lateral_C4", "lateral_C5",
        "refine_p3", "refine_p4", "refine_p5",
        "extra_F6", "extra_F7", "extra_F8",
        "box_predictions", "cls_predictions",
    ]
    print(f"\n{'Layer':<45} {'Output shape'}")
    print("-" * 75)
    for layer in model.layers:
        for kw in important_keywords:
            if kw in layer.name:
                shape = (batch_size, *layer.output_shape[1:])
                print(f"  {layer.name:<43} {str(shape)}")
                break


# ============================================================================
# Smoke test  (python model.py)
# ============================================================================

if __name__ == "__main__":

    print("=== Full Model Assembly — smoke test ===\n")

    # Build the model
    model = build_model()

    # Summary
    model.summary(line_length=88, expand_nested=False)
    print_layer_shapes(model)

    # Forward pass with random uint8 input
    B     = 2
    dummy = np.random.randint(0, 256, (B, 320, 320, 3), dtype=np.uint8)
    outs  = model(dummy, training=False)

    box_preds = outs["box_predictions"]
    cls_preds = outs["cls_predictions"]

    print("\n--- Output shapes ---")
    print(f"  box_predictions : {box_preds.shape}   "
          f"(expected ({B}, {CFG.TOTAL_ANCHORS}, 4))")
    print(f"  cls_predictions : {cls_preds.shape}   "
          f"(expected ({B}, {CFG.TOTAL_ANCHORS}, {CFG.NUM_CLASSES}))")

    assert box_preds.shape == (B, CFG.TOTAL_ANCHORS, 4), \
        f"box_predictions shape mismatch: {box_preds.shape}"
    assert cls_preds.shape == (B, CFG.TOTAL_ANCHORS, CFG.NUM_CLASSES), \
        f"cls_predictions shape mismatch: {cls_preds.shape}"

    # dtype checks
    assert box_preds.dtype == tf.float32, f"box dtype: {box_preds.dtype}"
    assert cls_preds.dtype == tf.float32, f"cls dtype: {cls_preds.dtype}"

    # No NaN / Inf in outputs
    assert not bool(tf.reduce_any(tf.math.is_nan(box_preds))), "NaN in box_preds"
    assert not bool(tf.reduce_any(tf.math.is_nan(cls_preds))), "NaN in cls_preds"
    assert not bool(tf.reduce_any(tf.math.is_inf(box_preds))), "Inf in box_preds"
    assert not bool(tf.reduce_any(tf.math.is_inf(cls_preds))), "Inf in cls_preds"

    total_params = model.count_params()
    print(f"\n  Total parameters : {total_params:,}")
    print(f"  Input dtype      : uint8")
    print(f"  Output dtype     : float32")
    print(f"  NaN / Inf check  : ✓ clean")

    # Quick end-to-end with postprocessor
    print("\n--- End-to-end: model → postprocess ---")
    from anchor_generator import generate_anchors
    from postprocess import postprocess

    anchors = generate_anchors()   # (8028, 4) [cy, cx, h, w]
    boxes, classes, scores, counts = postprocess(box_preds, cls_preds, anchors)

    print(f"  boxes   : {boxes.shape}    (expected ({B}, 10, 4))")
    print(f"  classes : {classes.shape}       (expected ({B}, 10))")
    print(f"  scores  : {scores.shape}       (expected ({B}, 10))")
    print(f"  counts  : {counts.numpy()}")

    assert boxes.shape   == (B, 10, 4)
    assert classes.shape == (B, 10)
    assert scores.shape  == (B, 10)
    assert counts.shape  == (B,)

    print("\n✓ Full model assembly and end-to-end pipeline verified")
