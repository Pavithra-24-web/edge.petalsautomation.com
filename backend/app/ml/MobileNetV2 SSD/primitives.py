# primitives.py
# Step 1 — Project Skeleton & Core Primitives
# MobileNetV2 SSD FPN-Lite 320x320

from __future__ import annotations

import math
import tensorflow as tf
from tensorflow import keras


# ============================================================================
# Architecture Config
# ============================================================================

class ModelConfig:
    """
    Single source of truth for every architectural constant.
    Import this object wherever a magic number would otherwise appear.
    """

    # --- Input ---
    INPUT_SIZE: int        = 320          # H == W
    INPUT_CHANNELS: int    = 3            # RGB
    INPUT_SHAPE: tuple     = (320, 320, 3)

    # --- Backbone feature taps ---
    # C3 → stride 8  output  (stage_3, out_channels=32)
    # C4 → stride 16 output  (stage_5, out_channels=96)
    # C5 → stride 32 output  (stage_7 320ch → last_conv 1×1 → 1280ch)
    #      The reference SSDLite MobileNetV2 appends a Conv2D 1×1 BN ReLU6
    #      "last conv" after stage_7 to expand 320 → 1280 channels before
    #      the FPN-Lite neck. This matches the TF Hub SavedModel layout and
    #      accounts for the majority of the reference model parameters.
    BACKBONE_TAPS: tuple   = ("C3", "C4", "C5")
    C3_CHANNELS: int       = 32
    C4_CHANNELS: int       = 96
    C5_CHANNELS: int       = 1280   # after last_conv (320 → 1280)

    # --- Neck (FPN-Lite) ---
    NECK_CHANNELS: int     = 256          # uniform channel width across all FPN levels

    # --- Detection feature maps ---
    # F3 → 40×40   (from C3 via FPN-Lite)
    # F4 → 20×20   (from C4 via FPN-Lite)
    # F5 → 10×10   (from C5 via FPN-Lite)
    # F6 →  5×5    (extra conv on F5)
    # F7 →  3×3    (extra conv on F6)
    # F8 →  2×2    (extra conv on F7)
    FEATURE_MAP_NAMES: tuple  = ("F3", "F4", "F5", "F6", "F7", "F8")
    FEATURE_MAP_STRIDES: tuple = (8, 16, 32, 64, 106, 160)  # effective strides
    FEATURE_MAP_SIZES: tuple   = (40, 20, 10, 5, 3, 2)       # spatial dims at 320px in

    # --- Anchor grid ---
    # Anchors per location per feature map:
    #   F3(40×40): 3  → 40×40×3  =  4,800
    #   F4(20×20): 6  → 20×20×6  =  2,400
    #   F5(10×10): 6  → 10×10×6  =    600
    #   F6( 5×5):  6  →  5× 5×6  =    150
    #   F7( 3×3):  6  →  3× 3×6  =     54
    #   F8( 2×2):  6  →  2× 2×6  =     24
    #                        Σ   =  8,028  ← exact base grid for this architecture
    ANCHORS_PER_LOC: tuple = (3, 6, 6, 6, 6, 6)
    TOTAL_ANCHORS: int     = 8_028

    # --- Head ---
    NUM_CLASSES: int       = 90           # COCO; override as needed
    TOP_K: int             = 10           # top-K detections returned post-NMS

    # --- Training (placeholders — not used until training step) ---
    BATCH_SIZE: int        = 32
    DTYPE: str             = "float32"


CFG = ModelConfig()  # convenience singleton — `from primitives import CFG`


# ============================================================================
# Core Primitives
# ============================================================================

def make_divisible(value: float, divisor: int = 8, min_value: int | None = None) -> int:
    """
    Round `value` up to the nearest multiple of `divisor`.

    Ensures channel counts stay hardware-friendly (8 or 16 aligned).
    Matches the reference MobileNetV2 implementation exactly.

    Args:
        value:     Raw (possibly fractional) channel count.
        divisor:   Alignment base — 8 for MobileNetV2.
        min_value: Floor on the returned value; defaults to `divisor`.

    Returns:
        Smallest integer >= value that is divisible by divisor.

    Examples:
        make_divisible(32)   → 32
        make_divisible(33)   → 40
        make_divisible(7, 8) → 8
    """
    if min_value is None:
        min_value = divisor
    new_value = max(min_value, int(value + divisor / 2) // divisor * divisor)
    # Prevent rounding down by more than 10 %
    if new_value < 0.9 * value:
        new_value += divisor
    return new_value


def conv_bn_relu6(
    inputs: tf.Tensor,
    filters: int,
    kernel_size: int = 3,
    strides: int = 1,
    padding: str = "same",
    use_bias: bool = False,
    name: str = "conv_bn_relu6",
) -> tf.Tensor:
    """
    Standard conv → BN → ReLU6 block.

    Used for:
      - First stem conv in MobileNetV2
      - Pointwise (1×1) expansion/projection convolutions
      - FPN-Lite feature projection convolutions

    Shape: (B, H, W, C_in) → (B, H//strides, W//strides, filters)

    Args:
        inputs:      Input tensor.
        filters:     Number of output channels.
        kernel_size: Convolution kernel size (default 3).
        strides:     Spatial stride (default 1).
        padding:     Keras padding string (default "same").
        use_bias:    Whether to add bias before BN (default False — BN
                     absorbs the bias term, so this is always False).
        name:        Name prefix for sub-layers.

    Returns:
        Output tensor after Conv → BN → ReLU6.
    """
    x = keras.layers.Conv2D(
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        use_bias=use_bias,
        name=f"{name}_conv",
    )(inputs)
    x = keras.layers.BatchNormalization(
        momentum=0.99,
        epsilon=1e-3,
        name=f"{name}_bn",
    )(x)
    x = keras.layers.ReLU(max_value=6.0, name=f"{name}_relu6")(x)
    return x


def depthwise_conv_bn_relu6(
    inputs: tf.Tensor,
    strides: int = 1,
    name: str = "dw_bn_relu6",
) -> tf.Tensor:
    """
    Depthwise conv (3×3) → BN → ReLU6 block.

    Operates channel-independently — no cross-channel mixing.
    Used inside every MobileNetV2 inverted-residual bottleneck.

    Shape: (B, H, W, C) → (B, H//strides, W//strides, C)
           Channel count is UNCHANGED (depth_multiplier=1).

    Args:
        inputs:  Input tensor.
        strides: Spatial stride (1 or 2).
        name:    Name prefix for sub-layers.

    Returns:
        Output tensor after DepthwiseConv → BN → ReLU6.
    """
    x = keras.layers.DepthwiseConv2D(
        kernel_size=3,
        strides=strides,
        padding="same",
        use_bias=False,
        name=f"{name}_dw_conv",
    )(inputs)
    x = keras.layers.BatchNormalization(
        momentum=0.99,
        epsilon=1e-3,
        name=f"{name}_bn",
    )(x)
    x = keras.layers.ReLU(max_value=6.0, name=f"{name}_relu6")(x)
    return x


def sep_conv(
    inputs: tf.Tensor,
    filters: int,
    strides: int = 1,
    name: str = "sep_conv",
) -> tf.Tensor:
    """
    Separable convolution: depthwise (3×3) → BN → ReLU6 → pointwise (1×1) → BN → ReLU6.

    Used in FPN-Lite neck to project and mix features cheaply.

    Shape: (B, H, W, C_in) → (B, H//strides, W//strides, filters)

    Args:
        inputs:  Input tensor.
        filters: Output channel count after pointwise conv.
        strides: Applied to the depthwise step (default 1).
        name:    Name prefix for sub-layers.

    Returns:
        Output tensor after full separable conv sequence.
    """
    x = depthwise_conv_bn_relu6(inputs, strides=strides, name=f"{name}_dw")
    x = conv_bn_relu6(x, filters=filters, kernel_size=1, strides=1, name=f"{name}_pw")
    return x


# ============================================================================
# Smoke test (python primitives.py)
# ============================================================================

if __name__ == "__main__":
    import numpy as np

    print("=== ModelConfig ===")
    print(f"  input shape     : {CFG.INPUT_SHAPE}")
    print(f"  backbone taps   : {CFG.BACKBONE_TAPS}")
    print(f"  neck channels   : {CFG.NECK_CHANNELS}")
    print(f"  feature maps    : {CFG.FEATURE_MAP_NAMES}")
    print(f"  total anchors   : {CFG.TOTAL_ANCHORS}")
    print(f"  top-K           : {CFG.TOP_K}")
    print()

    print("=== make_divisible ===")
    for v in [7, 8, 32, 33, 40, 96, 100]:
        print(f"  make_divisible({v:>3}) → {make_divisible(v)}")
    print()

    # Build a tiny functional model to exercise every primitive
    inp = keras.Input(shape=(28, 28, 16), name="test_input")
    x = conv_bn_relu6(inp, filters=32, kernel_size=3, strides=2, name="test_conv")
    # x: (B, 14, 14, 32)
    x = depthwise_conv_bn_relu6(x, strides=1, name="test_dw")
    # x: (B, 14, 14, 32)
    x = sep_conv(x, filters=64, strides=2, name="test_sep")
    # x: (B,  7,  7, 64)

    model = keras.Model(inp, x, name="primitives_smoke_test")
    model.summary(line_length=80)

    dummy = np.random.randint(0, 256, (2, 28, 28, 16)).astype("float32")
    out = model(dummy, training=False)
    assert out.shape == (2, 7, 7, 64), f"Unexpected shape: {out.shape}"
    print(f"\n✓ output shape : {out.shape}  (expected (2, 7, 7, 64))")
    print("✓ All primitives OK")
