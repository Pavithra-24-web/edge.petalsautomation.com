# stem.py
# Step 2 — MobileNetV2 Stem
# MobileNetV2 SSD FPN-Lite 320x320

from __future__ import annotations

import numpy as np
import tensorflow as tf
from tensorflow import keras

from primitives import CFG, conv_bn_relu6


# ============================================================================
# Stem
# ============================================================================

def build_stem(inputs: tf.Tensor, name: str = "stem") -> tf.Tensor:
    """
    MobileNetV2 stem: Conv2D 3×3, stride 2 → BatchNorm → ReLU6.

    This is the very first stage of the backbone. It aggressively halves
    spatial resolution on the first pass so that all subsequent inverted-
    residual blocks operate on cheaper 160×160 feature maps.

    Shape flow:
        Input  : (B, 320, 320,  3)   uint8 or float32 [-1, 1]
        Output : (B, 160, 160, 32)   float32

    Args:
        inputs: Input tensor of shape (B, 320, 320, 3).
                Expected to be the normalised float32 output from
                NormalizationLayer (values in [-1, 1]).
        name:   Name prefix for all child layers (default "stem").

    Returns:
        tf.Tensor of shape (B, 160, 160, 32).
    """
    x = conv_bn_relu6(
        inputs,
        filters=32,          # reference MobileNetV2 stem width
        kernel_size=3,
        strides=2,           # 320 → 160
        padding="same",
        use_bias=False,
        name=name,
    )
    return x                 # (B, 160, 160, 32)


# ============================================================================
# Smoke test  (python stem.py)
# ============================================================================

if __name__ == "__main__":
    # --- build model ---
    inputs = keras.Input(shape=CFG.INPUT_SHAPE, dtype=tf.float32, name="input_image")
    stem_out = build_stem(inputs)

    model = keras.Model(inputs, stem_out, name="stem_smoke_test")
    model.summary(line_length=72)

    # --- shape assertions ---
    dummy = np.random.uniform(-1.0, 1.0, (2, 320, 320, 3)).astype("float32")
    out = model(dummy, training=False).numpy()

    assert out.shape == (2, 160, 160, 32), f"Shape mismatch: {out.shape}"
    assert out.dtype == np.float32,        f"dtype mismatch: {out.dtype}"

    print(f"\n✓ output shape : {out.shape}  (expected (2, 160, 160, 32))")
    print(f"✓ output dtype : {out.dtype}")
    print(f"✓ trainable params: {model.count_params():,}")
    print("✓ Stem OK")
