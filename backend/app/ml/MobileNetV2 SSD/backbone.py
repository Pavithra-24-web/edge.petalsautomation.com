# backbone.py
# Step 5 — Full MobileNetV2 Backbone
# MobileNetV2 SSD FPN-Lite 320x320

from __future__ import annotations

import numpy as np
import tensorflow as tf
from tensorflow import keras

from primitives import CFG, conv_bn_relu6
from stem import build_stem
from stage_builder import make_stage


# ============================================================================
# MobileNetV2 Backbone
# ============================================================================

def build_backbone(inputs: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """
    Full MobileNetV2 backbone for 320×320 input.

    Implements the exact stage sequence from MobileNetV2 Table 2, then
    returns the three feature maps used by the FPN-Lite neck:

        C3 — stride-8  output  →  (B,  40,  40,  32)
        C4 — stride-16 output  →  (B,  20,  20,  96)
        C5 — stride-32 output  →  (B,  10,  10, 1280)  after last_conv

    Stage sequence:
        Stem      : Conv 3×3 s=2          (B, 320, 320,    3) → (B, 160, 160,   32)
        Stage 1   : t=1 c=16  n=1 s=1     (B, 160, 160,   32) → (B, 160, 160,   16)
        Stage 2   : t=6 c=24  n=2 s=2     (B, 160, 160,   16) → (B,  80,  80,   24)
        Stage 3   : t=6 c=32  n=3 s=2     (B,  80,  80,   24) → (B,  40,  40,   32)  ← C3
        Stage 4   : t=6 c=64  n=4 s=2     (B,  40,  40,   32) → (B,  20,  20,   64)
        Stage 5   : t=6 c=96  n=3 s=1     (B,  20,  20,   64) → (B,  20,  20,   96)  ← C4
        Stage 6   : t=6 c=160 n=3 s=2     (B,  20,  20,   96) → (B,  10,  10,  160)
        Stage 7   : t=6 c=320 n=1 s=1     (B,  10,  10,  160) → (B,  10,  10,  320)
        last_conv : Conv 1×1  s=1  BN ReLU6 (B, 10, 10,  320) → (B,  10,  10, 1280)  ← C5

    The "last_conv" layer (Conv 1×1 320→1280, BN, ReLU6) is the standard
    MobileNetV2 final expansion layer retained in the SSDLite reference
    design.  It is applied after stage_7 and before the FPN-Lite neck.
    This layer contributes ~410K parameters and is required to match the
    reference TF Hub SSDLite MobileNetV2 320×320 weight layout (~3.4M
    params total, ~3.7 MB as int8 / ~14 MB as float32).

    Args:
        inputs: Float32 tensor of shape (B, 320, 320, 3), values in [-1, 1].
                Typically the output of NormalizationLayer from preprocessing.py.

    Returns:
        Tuple (C3, C4, C5):
            C3: (B,  40,  40,   32)   stride-8  feature map
            C4: (B,  20,  20,   96)   stride-16 feature map
            C5: (B,  10,  10, 1280)   stride-32 feature map (after last_conv)
    """

    # --- Stem ---------------------------------------------------------------
    # (B, 320, 320, 3) → (B, 160, 160, 32)
    x = build_stem(inputs, name="stem")

    # --- Stage 1 : t=1 c=16 n=1 s=1 ----------------------------------------
    # (B, 160, 160, 32) → (B, 160, 160, 16)
    # Expansion=1 so the expand conv is skipped inside each IRB.
    # No residual: C_in(32) ≠ C_out(16).
    x = make_stage(x, t=1, c=16, n=1, s=1, name="stage_1")

    # --- Stage 2 : t=6 c=24 n=2 s=2 ----------------------------------------
    # (B, 160, 160, 16) → (B, 80, 80, 24)
    # block_0: stride=2  16→24  no residual
    # block_1: stride=1  24→24  residual ✓
    x = make_stage(x, t=6, c=24, n=2, s=2, name="stage_2")

    # --- Stage 3 : t=6 c=32 n=3 s=2  →  C3 tap -----------------------------
    # (B, 80, 80, 24) → (B, 40, 40, 32)
    # block_0: stride=2  24→32  no residual
    # block_1: stride=1  32→32  residual ✓
    # block_2: stride=1  32→32  residual ✓
    x = make_stage(x, t=6, c=32, n=3, s=2, name="stage_3")
    C3 = x   # (B, 40, 40, 32)  — stride-8 feature tap

    # --- Stage 4 : t=6 c=64 n=4 s=2 ----------------------------------------
    # (B, 40, 40, 32) → (B, 20, 20, 64)
    # block_0: stride=2  32→64  no residual
    # block_1: stride=1  64→64  residual ✓
    # block_2: stride=1  64→64  residual ✓
    # block_3: stride=1  64→64  residual ✓
    x = make_stage(x, t=6, c=64, n=4, s=2, name="stage_4")

    # --- Stage 5 : t=6 c=96 n=3 s=1  →  C4 tap -----------------------------
    # (B, 20, 20, 64) → (B, 20, 20, 96)
    # block_0: stride=1  64→96  no residual (C_in≠C_out)
    # block_1: stride=1  96→96  residual ✓
    # block_2: stride=1  96→96  residual ✓
    x = make_stage(x, t=6, c=96, n=3, s=1, name="stage_5")
    C4 = x   # (B, 20, 20, 96)  — stride-16 feature tap

    # --- Stage 6 : t=6 c=160 n=3 s=2 ---------------------------------------
    # (B, 20, 20, 96) → (B, 10, 10, 160)
    # block_0: stride=2   96→160  no residual
    # block_1: stride=1  160→160  residual ✓
    # block_2: stride=1  160→160  residual ✓
    x = make_stage(x, t=6, c=160, n=3, s=2, name="stage_6")

    # --- Stage 7 : t=6 c=320 n=1 s=1 ----------------------------------------
    # (B, 10, 10, 160) → (B, 10, 10, 320)
    # block_0: stride=1  160→320  no residual (C_in≠C_out)
    x = make_stage(x, t=6, c=320, n=1, s=1, name="stage_7")

    # --- Last conv : 1×1 Conv → BN → ReLU6  →  C5 tap ----------------------
    # (B, 10, 10, 320) → (B, 10, 10, 1280)
    # This is the standard MobileNetV2 "last conv" expansion layer retained
    # in the SSDLite reference design.  Expanding to 1280 channels before
    # the FPN-Lite neck gives the neck richer features to project from and
    # matches the TF Hub SSDLite MobileNetV2 320×320 parameter layout.
    x = conv_bn_relu6(x, filters=1280, kernel_size=1, strides=1, name="last_conv")
    C5 = x   # (B, 10, 10, 1280)  — stride-32 feature tap (after last_conv)

    return C3, C4, C5


# ============================================================================
# Model factory (optional convenience wrapper)
# ============================================================================

def build_backbone_model(
    input_shape: tuple = CFG.INPUT_SHAPE,
) -> keras.Model:
    """
    Wrap the backbone as a standalone Keras Model for inspection/export.

    Args:
        input_shape: (H, W, C) tuple; default (320, 320, 3).

    Returns:
        keras.Model with inputs=(B,320,320,3) float32
                     and outputs=[C3, C4, C5]
                     where C5 has 1280 channels after last_conv.
    """
    inputs = keras.Input(shape=input_shape, dtype=tf.float32, name="backbone_input")
    C3, C4, C5 = build_backbone(inputs)
    return keras.Model(
        inputs=inputs,
        outputs={"C3": C3, "C4": C4, "C5": C5},
        name="mobilenetv2_backbone",
    )


# ============================================================================
# Smoke test  (python backbone.py)
# ============================================================================

if __name__ == "__main__":

    print("=== MobileNetV2 Backbone — shape test ===\n")

    model = build_backbone_model()
    model.summary(line_length=80)

    dummy = np.random.uniform(-1.0, 1.0, (2, 320, 320, 3)).astype("float32")
    outs  = model(dummy, training=False)

    C3 = outs["C3"].numpy()
    C4 = outs["C4"].numpy()
    C5 = outs["C5"].numpy()

    print("\n--- Feature tap shapes ---")
    print(f"  C3 : {C3.shape}   (expected (2,  40,  40,   32))")
    print(f"  C4 : {C4.shape}   (expected (2,  20,  20,   96))")
    print(f"  C5 : {C5.shape}   (expected (2,  10,  10, 1280))")

    assert C3.shape == (2,  40,  40,   32), f"C3 shape mismatch: {C3.shape}"
    assert C4.shape == (2,  20,  20,   96), f"C4 shape mismatch: {C4.shape}"
    assert C5.shape == (2,  10,  10, 1280), f"C5 shape mismatch: {C5.shape}"

    total_params = model.count_params()
    print(f"\n  Total trainable params : {total_params:,}")
    print(f"  (Reference MobileNetV2 backbone ≈ 2.2 M params)")

    print("\n✓ All backbone shape assertions passed")
