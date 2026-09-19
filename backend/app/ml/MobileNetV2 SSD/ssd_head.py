# ssd_head.py
# Step 7 — SSD Shared Prediction Head
# MobileNetV2 SSD FPN-Lite 320x320

from __future__ import annotations

import math
import numpy as np
import tensorflow as tf
from tensorflow import keras

from primitives import CFG, depthwise_conv_bn_relu6

# Prior probability for foreground at init: sigmoid(bias) ≈ 0.01.
# Initialising the classification head to predict near-zero foreground
# probability avoids the large initial loss from 8 k+ background anchors
# all predicting 0.5 confidence, which would dominate gradients for the
# first several epochs and harm box-regression convergence.
_CLS_PRIOR_PROB = 0.01
_CLS_PRIOR_BIAS = math.log(_CLS_PRIOR_PROB / (1.0 - _CLS_PRIOR_PROB))  # ≈ -4.595


# ============================================================================
# Shared Predictor Layers
# ============================================================================

class SharedPredictor(keras.layers.Layer):
    """
    Shared-weight SSD predictor applied identically across every FPN level.

    Each feature level F_i of shape (B, H_i, W_i, 256) passes through
    the same depthwise-separable conv tower, then splits into:
      - box branch  : predicts (dy, dx, dh, dw) offsets per anchor
      - class branch: predicts per-class logits per anchor

    Weights are shared — the same layer instance is called on every
    feature map, so F3…F8 all use identical predictor weights.

    Architecture per branch (matches TF OD API SSDLite):
        DWConv 3×3 → BN → ReLU6          (feature mixing)
        Conv  1×1  → (output)             (no activation on final layer)

    The final Conv2D has no BN and no activation:
        - Box branch  : raw regression offsets (unbounded)
        - Class branch: raw logits (sigmoid/softmax applied at loss time)

    Args:
        num_anchors_per_loc: Number of anchors at each spatial location
                             for THIS feature level.
        num_classes:         Number of detection classes (default 90 COCO).
        name:                Layer name prefix.
    """

    def __init__(
        self,
        num_anchors_per_loc: int,
        num_classes: int = CFG.NUM_CLASSES,
        name: str = "shared_predictor",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        self.num_anchors_per_loc = num_anchors_per_loc
        self.num_classes = num_classes

        # ---- Box branch ------------------------------------------------
        # DWConv tower (shared spatial mixing before prediction)
        self.box_dw = keras.layers.DepthwiseConv2D(
            kernel_size=3,
            strides=1,
            padding="same",
            use_bias=False,
            name=f"{name}_box_dw_conv",
        )
        self.box_dw_bn = keras.layers.BatchNormalization(
            momentum=0.99, epsilon=1e-3, name=f"{name}_box_dw_bn"
        )
        self.box_dw_relu = keras.layers.ReLU(max_value=6.0, name=f"{name}_box_dw_relu6")

        # Final 1×1 conv: outputs 4 box offsets per anchor, no activation
        self.box_pw = keras.layers.Conv2D(
            filters=num_anchors_per_loc * 4,
            kernel_size=1,
            strides=1,
            padding="same",
            use_bias=True,              # bias ON — no BN on final layer
            name=f"{name}_box_pw_conv",
        )

        # ---- Class branch ----------------------------------------------
        self.cls_dw = keras.layers.DepthwiseConv2D(
            kernel_size=3,
            strides=1,
            padding="same",
            use_bias=False,
            name=f"{name}_cls_dw_conv",
        )
        self.cls_dw_bn = keras.layers.BatchNormalization(
            momentum=0.99, epsilon=1e-3, name=f"{name}_cls_dw_bn"
        )
        self.cls_dw_relu = keras.layers.ReLU(max_value=6.0, name=f"{name}_cls_dw_relu6")

        # Final 1×1 conv: outputs num_classes logits per anchor, no activation.
        # Bias is initialised so sigmoid(bias) ≈ 0.01, matching the expected
        # low foreground prior and avoiding a large initial class loss from
        # the ~8 k background anchors that would otherwise all output 0.5.
        self.cls_pw = keras.layers.Conv2D(
            filters=num_anchors_per_loc * num_classes,
            kernel_size=1,
            strides=1,
            padding="same",
            use_bias=True,
            bias_initializer=keras.initializers.Constant(_CLS_PRIOR_BIAS),
            name=f"{name}_cls_pw_conv",
        )

    def call(
        self, x: tf.Tensor, training: bool = False
    ) -> tuple[tf.Tensor, tf.Tensor]:
        """
        Forward pass for one feature level.

        Args:
            x:        Feature map (B, H, W, 256).
            training: BN training flag.

        Returns:
            box_out : (B, H, W, num_anchors_per_loc * 4)
            cls_out : (B, H, W, num_anchors_per_loc * num_classes)
        """
        # Box branch
        b = self.box_dw(x)
        b = self.box_dw_bn(b, training=training)
        b = self.box_dw_relu(b)
        box_out = self.box_pw(b)           # (B, H, W, A*4)

        # Class branch
        c = self.cls_dw(x)
        c = self.cls_dw_bn(c, training=training)
        c = self.cls_dw_relu(c)
        cls_out = self.cls_pw(c)           # (B, H, W, A*num_classes)

        return box_out, cls_out

    def get_config(self) -> dict:
        cfg = super().get_config()
        cfg.update(
            num_anchors_per_loc=self.num_anchors_per_loc,
            num_classes=self.num_classes,
        )
        return cfg


# ============================================================================
# SSD Head
# ============================================================================

# Anchors per spatial location per feature level — must match anchor generator.
# [F3, F4, F5, F6, F7, F8] → per-level spatial anchor counts:
#   F3: 40×40×3  =  4,800
#   F4: 20×20×6  =  2,400
#   F5: 10×10×6  =    600
#   F6:  5× 5×6  =    150
#   F7:  3× 3×6  =     54
#   F8:  2× 2×6  =     24
#                 Σ = 8,028   ← exact total for this architecture
_ANCHORS_PER_LOC = CFG.ANCHORS_PER_LOC   # (3, 6, 6, 6, 6, 6)


def build_ssd_head(
    feature_maps: list[tf.Tensor],
    num_classes: int = CFG.NUM_CLASSES,
    anchors_per_loc: tuple[int, ...] = _ANCHORS_PER_LOC,
) -> tuple[tf.Tensor, tf.Tensor]:
    """
    SSD prediction head matching the TF OD API SSDLite MobileNetV2 reference.

    Each feature level gets its OWN independent SharedPredictor instance —
    weights are NOT shared across levels.  This matches the TF OD API
    SSDLite reference implementation and the TF Hub SSDLite MobileNetV2
    320×320 SavedModel (6 predictors total: one per FPN level).

    Weight sharing was previously applied across levels with the same
    anchors_per_loc value (F4-F8 shared one predictor).  That reduced the
    parameter count by ~300K and diverged from the reference architecture.

    Predictions from each level are:
        1. Reshaped from (B, H, W, A*K) → (B, H*W*A, K) using fully static
           target shapes derived from CFG.FEATURE_MAP_SIZES[i] at build time.
           Static shapes (rather than -1) let the TFLite converter emit a
           RESHAPE op with a compile-time constant shape tensor, which is
           required for hardware-delegate compatibility and avoids a
           dynamic shape computation on every inference call.
        2. Fused into one contiguous tensor via tf.concat (axis=1) rather
           than keras.layers.Concatenate, reducing graph node count and
           making the output tensor layout explicit to downstream tooling.

    Args:
        feature_maps:    [F3, F4, F5, F6, F7, F8] from the FPN-Lite neck.
        num_classes:     Number of object classes (default 90).
        anchors_per_loc: Per-level anchor counts tuple (default CFG value).

    Returns:
        box_preds  : (B, total_anchors, 4)
        cls_preds  : (B, total_anchors, num_classes)

    total_anchors is determined at build time from the feature map spatial
    dims and anchors_per_loc.  For the reference 320×320 design this
    resolves to 8,028 (base grid for 6 FPN levels with ANCHORS_PER_LOC=(3,6,6,6,6,6)).
    """
    assert len(feature_maps) == len(anchors_per_loc), (
        f"Got {len(feature_maps)} feature maps but "
        f"{len(anchors_per_loc)} anchor specs."
    )

    all_box_preds: list[tf.Tensor] = []
    all_cls_preds: list[tf.Tensor] = []

    for i, (fmap, a) in enumerate(zip(feature_maps, anchors_per_loc)):
        level_name = CFG.FEATURE_MAP_NAMES[i]       # "F3" … "F8"

        # Each level gets its own independent predictor — no weight sharing.
        # This matches the TF OD API SSDLite reference (6 separate predictors).
        predictor = SharedPredictor(
            num_anchors_per_loc=a,
            num_classes=num_classes,
            name=f"predictor_{level_name}",
        )

        # Apply predictor to this feature level
        # box_out: (B, H, W, a*4)
        # cls_out: (B, H, W, a*num_classes)
        box_out, cls_out = predictor(fmap)

        # Compute the fully-static anchor count for this level at build time.
        # CFG.FEATURE_MAP_SIZES[i] is the side length of the square grid
        # (40, 20, 10, 5, 3, 2 for F3…F8), known as a Python int.
        # Using a static integer here — rather than tf.shape or -1 — lets
        # the TFLite converter emit a RESHAPE with a constant shape tensor,
        # which is required for delegate compatibility.
        H = W  = CFG.FEATURE_MAP_SIZES[i]   # square feature maps
        n_anc  = H * W * a                  # total anchors for this level

        # (B, H, W, a*4) → (B, n_anc, 4)   — static target shape
        box_flat = keras.layers.Reshape(
            (n_anc, 4),
            name=f"box_reshape_{level_name}",
        )(box_out)

        # (B, H, W, a*num_classes) → (B, n_anc, num_classes)   — static
        cls_flat = keras.layers.Reshape(
            (n_anc, num_classes),
            name=f"cls_reshape_{level_name}",
        )(cls_out)

        all_box_preds.append(box_flat)
        all_cls_preds.append(cls_flat)

    # Fuse all levels into one contiguous tensor along the anchor axis.
    # keras.layers.Concatenate is required here: tf.concat raises
    # "KerasTensor cannot be used with raw TensorFlow ops" under Keras 3.
    # The output names ("box_predictions", "cls_predictions") are set by
    # the dict keys in keras.Model(outputs={...}) in model.py — no tf.identity needed.
    box_preds = keras.layers.Concatenate(axis=1, name="box_predictions")(all_box_preds)
    cls_preds = keras.layers.Concatenate(axis=1, name="cls_predictions")(all_cls_preds)

    return box_preds, cls_preds   # (B, total_anchors, 4), (B, total_anchors, num_classes)


# ============================================================================
# Smoke test  (python ssd_head.py)
# ============================================================================

if __name__ == "__main__":

    # Synthetic FPN-Lite outputs — must match expected FPN output shapes
    FMAP_SHAPES = [
        (40, 40, 256),   # F3
        (20, 20, 256),   # F4
        (10, 10, 256),   # F5
        ( 5,  5, 256),   # F6
        ( 3,  3, 256),   # F7
        ( 2,  2, 256),   # F8
    ]
    B = 2

    fmap_inputs = [keras.Input(shape=s, name=n)
                   for s, n in zip(FMAP_SHAPES, CFG.FEATURE_MAP_NAMES)]
    box_preds, cls_preds = build_ssd_head(fmap_inputs)

    model = keras.Model(
        inputs=fmap_inputs,
        outputs={"box_predictions": box_preds, "cls_predictions": cls_preds},
        name="ssd_head_smoke",
    )
    model.summary(line_length=88)

    # Forward pass with random feature maps
    dummy_fmaps = [
        np.random.randn(B, *s).astype("float32")
        for s in FMAP_SHAPES
    ]
    outs = model(dummy_fmaps, training=False)

    box_out = outs["box_predictions"].numpy()
    cls_out = outs["cls_predictions"].numpy()

    # Anchor count breakdown
    anchors_per_level = [
        s[0] * s[1] * a
        for s, a in zip(FMAP_SHAPES, CFG.ANCHORS_PER_LOC)
    ]
    total = sum(anchors_per_level)

    print("\n--- Per-level anchor counts ---")
    for name, n in zip(CFG.FEATURE_MAP_NAMES, anchors_per_level):
        print(f"  {name}: {n:>5,}")
    print(f"  {'Total':>2}: {total:>5,}  (reference target: {CFG.TOTAL_ANCHORS:,})")

    print("\n--- Output shapes ---")
    print(f"  box_predictions : {box_out.shape}   (expected (B, {total}, 4))")
    print(f"  cls_predictions : {cls_out.shape}   (expected (B, {total}, {CFG.NUM_CLASSES}))")

    assert box_out.shape == (B, total, 4), \
        f"Box shape mismatch: {box_out.shape}"
    assert cls_out.shape == (B, total, CFG.NUM_CLASSES), \
        f"Class shape mismatch: {cls_out.shape}"

    print(f"\n  Head params: {model.count_params():,}")
    print("\n✓ All SSD head shape assertions passed")
