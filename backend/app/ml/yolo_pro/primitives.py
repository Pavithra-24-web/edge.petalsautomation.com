"""
yolo_pro/primitives.py
──────────────────────
Low-level, stateless building blocks shared by every module
(backbone, neck, head).

Public API
----------
    silu(x)               — activation function
    conv_bn(x, ...)       — Conv2D → BN → SiLU
    dw_conv(x, ...)       — DepthwiseConv2D → BN → SiLU

Design rules
------------
  * Every function is purely functional: takes a tensor, returns a tensor.
  * No Module subclassing here — composability is via plain function calls.
  * BN momentum and epsilon are module-level constants; change once,
    propagates everywhere.
  * Bias is always False when BN follows (BN subsumes the bias term).
"""

import tensorflow as tf
from tensorflow.keras import layers

# ── Shared BN hyper-parameters ────────────────────────────────────────────────
BN_MOMENTUM: float = 0.9    # spec-mandated; inherited from FOMO baseline
BN_EPSILON:  float = 1e-3


# ─────────────────────────────────────────────────────────────────────────────
# 1.  SiLU  (Sigmoid Linear Unit / Swish)
# ─────────────────────────────────────────────────────────────────────────────
def silu(x: tf.Tensor) -> tf.Tensor:
    """
    SiLU activation: f(x) = x · σ(x)

    Identical to tf.keras.activations.swish.
    Defined explicitly so model.summary() shows 'silu' layer names.

    Shape: unchanged — (B, H, W, C) → (B, H, W, C)
    """
    return x * tf.math.sigmoid(x)


# ─────────────────────────────────────────────────────────────────────────────
# 2.  conv_bn  —  Conv2D → BatchNorm → SiLU
# ─────────────────────────────────────────────────────────────────────────────
def conv_bn(
    x: tf.Tensor,
    filters: int,
    kernel_size: int,
    strides: int = 1,
    padding: str = "same",
    name: str = "",
) -> tf.Tensor:
    """
    Standard convolution block: Conv2D → BN → SiLU.

    Used for:
      - Stem 3×3 conv
      - Pointwise (1×1) projections
      - Neck downsampling convolutions
      - Head conv layers

    Args:
        x           : input tensor  (B, H, W, C_in)
        filters     : output channel count
        kernel_size : spatial kernel size (int)
        strides     : spatial stride (default 1)
        padding     : 'same' keeps H×W when stride=1 (default)
        name        : layer-name prefix (empty → auto-named)

    Returns:
        Tensor  (B, H', W', filters)

    Shape examples:
        stride=1  →  H' = H,   W' = W
        stride=2  →  H' = H/2, W' = W/2  (with padding='same')
    """
    p = f"{name}_" if name else ""

    x = layers.Conv2D(
        filters            = filters,
        kernel_size        = kernel_size,
        strides            = strides,
        padding            = padding,
        use_bias           = False,
        kernel_initializer = "he_normal",
        name               = f"{p}conv",
    )(x)
    x = layers.BatchNormalization(
        momentum = BN_MOMENTUM,
        epsilon  = BN_EPSILON,
        name     = f"{p}bn",
    )(x)
    x = layers.Lambda(silu, name=f"{p}silu")(x)
    return x


# ─────────────────────────────────────────────────────────────────────────────
# 3.  dw_conv  —  DepthwiseConv2D → BatchNorm → SiLU
# ─────────────────────────────────────────────────────────────────────────────
def dw_conv(
    x: tf.Tensor,
    kernel_size: int = 3,
    strides: int = 1,
    name: str = "",
) -> tf.Tensor:
    """
    Depthwise convolution block: DWConv2D → BN → SiLU.

    Channels are NOT changed — each input channel is convolved by its
    own kernel (depth_multiplier=1).  A pointwise conv_bn(1×1) must
    follow if you need to change the channel count.

    Used inside InvertedResidual blocks (next step) as the spatial
    mixing stage.

    Args:
        x           : input tensor  (B, H, W, C)
        kernel_size : spatial kernel size (default 3)
        strides     : spatial stride (default 1)
        name        : layer-name prefix

    Returns:
        Tensor  (B, H', W', C)   ← same channel count as input

    Shape examples:
        stride=1  →  H' = H,   W' = W,   C_out = C_in
        stride=2  →  H' = H/2, W' = W/2, C_out = C_in
    """
    p = f"{name}_" if name else ""

    x = layers.DepthwiseConv2D(
        kernel_size            = kernel_size,
        strides                = strides,
        padding                = "same",
        use_bias               = False,
        depthwise_initializer  = "he_normal",
        name                   = f"{p}dw",
    )(x)
    x = layers.BatchNormalization(
        momentum = BN_MOMENTUM,
        epsilon  = BN_EPSILON,
        name     = f"{p}bn",
    )(x)
    x = layers.Lambda(silu, name=f"{p}silu")(x)
    return x
