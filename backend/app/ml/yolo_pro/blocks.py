"""
yolo_pro/blocks.py
──────────────────
Reusable primitives for the YOLO-Pro architecture.

All convolutions follow the same pattern:
    Conv2D → BatchNorm (momentum=0.9) → SiLU

These blocks are intentionally kept stateless and functional so they
compose cleanly into backbone, neck, and head modules.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# ── Constants ────────────────────────────────────────────────────────────────
BN_MOMENTUM = 0.9          # inherited from FOMO; do NOT change for small models
BN_EPSILON   = 1e-3


# ─────────────────────────────────────────────────────────────────────────────
# 1.  SiLU activation
#     TF 2.x ships keras.activations.swish which is identical to SiLU.
#     We expose it as a named layer so it shows up clearly in model.summary().
# ─────────────────────────────────────────────────────────────────────────────
def silu(x: tf.Tensor) -> tf.Tensor:
    """SiLU / Swish activation: x * sigmoid(x)."""
    return x * tf.math.sigmoid(x)


# ─────────────────────────────────────────────────────────────────────────────
# 2.  ConvBnSilu  — the atomic conv unit used everywhere
# ─────────────────────────────────────────────────────────────────────────────
def conv_bn_silu(
    x: tf.Tensor,
    filters: int,
    kernel_size: int,
    strides: int = 1,
    padding: str = "same",
    groups: int = 1,          # groups=1 → standard conv; groups=C → depthwise
    use_bias: bool = False,
    name: str = "",
) -> tf.Tensor:
    """
    Conv2D → BatchNorm → SiLU.

    Args:
        x           : input tensor  (B, H, W, C_in)
        filters     : number of output channels
        kernel_size : spatial kernel (int or tuple)
        strides     : spatial stride
        padding     : 'same' or 'valid'
        groups      : channel groups (use filters == C_in for depthwise)
        use_bias    : False (BN subsumes bias)
        name        : prefix for layer names (helps debugging)

    Returns:
        Tensor  (B, H', W', filters)
    """
    prefix = f"{name}_" if name else ""

    x = layers.Conv2D(
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        groups=groups,
        use_bias=use_bias,
        kernel_initializer="he_normal",
        name=f"{prefix}conv",
    )(x)

    x = layers.BatchNormalization(
        momentum=BN_MOMENTUM,
        epsilon=BN_EPSILON,
        name=f"{prefix}bn",
    )(x)

    x = layers.Lambda(silu, name=f"{prefix}silu")(x)

    return x


# ─────────────────────────────────────────────────────────────────────────────
# 3.  DepthwiseSeparable  — DW conv + PW conv, each with BN+SiLU
# ─────────────────────────────────────────────────────────────────────────────
def depthwise_separable(
    x: tf.Tensor,
    filters_out: int,
    kernel_size: int = 3,
    strides: int = 1,
    name: str = "",
) -> tf.Tensor:
    """
    Depthwise-Separable convolution:
        DW (C_in → C_in, BN+SiLU)  →  PW 1×1 (C_in → filters_out, BN+SiLU)

    Tensor flow:
        (B, H, W, C_in)
            → depthwise  →  (B, H', W', C_in)
            → pointwise  →  (B, H', W', filters_out)
    """
    prefix = f"{name}_" if name else ""
    c_in = x.shape[-1]

    # Depthwise: each channel convolved independently
    x = layers.DepthwiseConv2D(
        kernel_size=kernel_size,
        strides=strides,
        padding="same",
        use_bias=False,
        depthwise_initializer="he_normal",
        name=f"{prefix}dw_conv",
    )(x)
    x = layers.BatchNormalization(
        momentum=BN_MOMENTUM, epsilon=BN_EPSILON, name=f"{prefix}dw_bn"
    )(x)
    x = layers.Lambda(silu, name=f"{prefix}dw_silu")(x)

    # Pointwise: mix channels
    x = layers.Conv2D(
        filters=filters_out,
        kernel_size=1,
        strides=1,
        padding="same",
        use_bias=False,
        kernel_initializer="he_normal",
        name=f"{prefix}pw_conv",
    )(x)
    x = layers.BatchNormalization(
        momentum=BN_MOMENTUM, epsilon=BN_EPSILON, name=f"{prefix}pw_bn"
    )(x)
    x = layers.Lambda(silu, name=f"{prefix}pw_silu")(x)

    return x


# ─────────────────────────────────────────────────────────────────────────────
# 4.  CSP split-fuse wrapper
#     Wraps *any* callable block in a Cross-Stage-Partial structure:
#
#         input (B, H, W, C)
#            ├─ identity_half  (B, H, W, C//2)  ──────────────────────────┐
#            └─ block_half     (B, H, W, C//2) → block(·) → (B,H,W,C//2) ─┤
#                                                                          ↓
#                                                              Concat → (B,H,W,C)
#                                                              Fuse 1×1 → (B,H,W,C_out)
# ─────────────────────────────────────────────────────────────────────────────
def csp_wrap(
    x: tf.Tensor,
    block_fn,           # callable: (tensor, name_prefix) -> tensor
    filters_out: int,
    name: str = "",
) -> tf.Tensor:
    """
    Generic CSP (Cross-Stage-Partial) wrapper.

    Splits input channels in half:
        - One half goes through `block_fn` (processed route).
        - Other half is a skip (identity route).
    Both halves are concatenated and fused with a 1×1 conv.

    Args:
        x           : input tensor  (B, H, W, C)
        block_fn    : function(tensor, name) → tensor  — operates on C//2 channels
        filters_out : output channel count after fusion
        name        : layer name prefix

    Returns:
        Tensor  (B, H', W', filters_out)

    Shape note:
        C_in must be even.  Each half = C_in // 2 channels.
    """
    prefix = f"{name}_" if name else ""
    c_in   = x.shape[-1]
    half   = c_in // 2

    # Split along the channel axis
    identity = layers.Lambda(
        lambda t: t[..., :half], name=f"{prefix}csp_identity"
    )(x)
    processed = layers.Lambda(
        lambda t: t[..., half:], name=f"{prefix}csp_processed_in"
    )(x)

    # Run the block on the processed half
    processed = block_fn(processed, name=f"{prefix}csp_block")

    # Re-join
    x = layers.Concatenate(axis=-1, name=f"{prefix}csp_concat")(
        [identity, processed]
    )

    # 1×1 fusion projection
    x = conv_bn_silu(
        x,
        filters=filters_out,
        kernel_size=1,
        name=f"{prefix}csp_fuse",
    )

    return x


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Quick smoke-test (run this file directly to verify shapes)
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import numpy as np

    # --- conv_bn_silu ---
    inp = keras.Input(shape=(160, 160, 3))
    out = conv_bn_silu(inp, filters=24, kernel_size=3, strides=2, name="stem")
    m   = keras.Model(inp, out)
    print("conv_bn_silu  :", inp.shape, "→", out.shape)
    # Expected: (None,160,160,3) → (None,80,80,24)

    # --- depthwise_separable ---
    inp2 = keras.Input(shape=(80, 80, 24))
    out2 = depthwise_separable(inp2, filters_out=40, strides=2, name="dws_test")
    print("depthwise_sep :", inp2.shape, "→", out2.shape)
    # Expected: (None,80,80,24) → (None,40,40,40)

    # --- csp_wrap ---
    inp3 = keras.Input(shape=(40, 40, 40))

    def _simple_block(t, name=""):
        return conv_bn_silu(t, filters=t.shape[-1], kernel_size=3, name=name)

    out3 = csp_wrap(inp3, block_fn=_simple_block, filters_out=40, name="csp_test")
    print("csp_wrap      :", inp3.shape, "→", out3.shape)
    # Expected: (None,40,40,40) → (None,40,40,40)

    print("\n✅  All blocks verified.")
