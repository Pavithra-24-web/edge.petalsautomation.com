# inverted_residual.py
# Step 3 — MobileNetV2 Inverted Residual Block
# MobileNetV2 SSD FPN-Lite 320x320

from __future__ import annotations

import numpy as np
import tensorflow as tf
from tensorflow import keras

from primitives import CFG, conv_bn_relu6, depthwise_conv_bn_relu6


# ============================================================================
# Inverted Residual Block
# ============================================================================

def inverted_residual_block(
    inputs: tf.Tensor,
    expansion: int,
    out_channels: int,
    strides: int = 1,
    name: str = "irb",
) -> tf.Tensor:
    """
    MobileNetV2 Inverted Residual Block (IRB).

    Internal structure:
        1. Expand   — 1×1 Conv → BN → ReLU6   (in_ch → in_ch * expansion)
        2. Depthwise— 3×3 DWConv → BN → ReLU6 (spatial mixing, same channels)
        3. Project  — 1×1 Conv → BN            (in_ch * expansion → out_channels)
                      *** NO activation after projection (linear bottleneck) ***

    Residual connection is added IFF:
        stride == 1  AND  in_channels == out_channels

    When expansion == 1 the expand step is skipped entirely (first IRB
    in the reference design has t=1 and no pointwise expand conv).

    Shape flow (stride=1, in_ch=C_in, out_ch=C_out):
        Input  : (B, H,   W,   C_in)
        Expand : (B, H,   W,   C_in * expansion)   [skipped if expansion==1]
        DW     : (B, H/s, W/s, C_in * expansion)
        Project: (B, H/s, W/s, C_out)
        Output : (B, H/s, W/s, C_out)              [+ input if residual applies]

    Args:
        inputs:      Input tensor.
        expansion:   Channel expansion factor t (typically 1 or 6).
        out_channels:Number of output channels after projection.
        strides:     Stride applied to the depthwise conv (1 or 2).
        name:        Name prefix for all child layers.

    Returns:
        Output tensor of shape (B, H//strides, W//strides, out_channels).
    """
    in_channels = inputs.shape[-1]           # C_in (static, known at build time)
    mid_channels = in_channels * expansion   # expanded width

    # ------------------------------------------------------------------ #
    # 1. Expand  —  1×1 pointwise conv → BN → ReLU6
    #    Skipped when expansion == 1 (no channel change needed)
    # ------------------------------------------------------------------ #
    if expansion == 1:
        x = inputs                           # (B, H, W, C_in)
    else:
        x = conv_bn_relu6(
            inputs,
            filters=mid_channels,
            kernel_size=1,
            strides=1,
            name=f"{name}_expand",
        )                                    # (B, H, W, C_in*t)

    # ------------------------------------------------------------------ #
    # 2. Depthwise  —  3×3 depthwise conv → BN → ReLU6
    #    Stride applied here — this is where spatial downsampling happens.
    # ------------------------------------------------------------------ #
    x = depthwise_conv_bn_relu6(
        x,
        strides=strides,
        name=f"{name}_dw",
    )                                        # (B, H/s, W/s, C_in*t)

    # ------------------------------------------------------------------ #
    # 3. Project  —  1×1 pointwise conv → BN   (NO ReLU6!)
    #    The linear bottleneck: ReLU after projection destroys information
    #    in low-dimensional spaces, so it is deliberately omitted.
    # ------------------------------------------------------------------ #
    x = keras.layers.Conv2D(
        filters=out_channels,
        kernel_size=1,
        strides=1,
        padding="same",
        use_bias=False,
        name=f"{name}_project_conv",
    )(x)                                     # (B, H/s, W/s, C_out)
    x = keras.layers.BatchNormalization(
        momentum=0.99,
        epsilon=1e-3,
        name=f"{name}_project_bn",
    )(x)                                     # (B, H/s, W/s, C_out)  — linear

    # ------------------------------------------------------------------ #
    # 4. Residual connection
    #    Condition: stride==1 AND in_channels==out_channels
    #    No projection shortcut — input and output dims must match exactly.
    # ------------------------------------------------------------------ #
    use_residual = (strides == 1) and (in_channels == out_channels)
    if use_residual:
        x = keras.layers.Add(name=f"{name}_add")([inputs, x])
                                             # (B, H, W, C_out) element-wise sum

    return x


# ============================================================================
# Smoke test  (python inverted_residual.py)
# ============================================================================

if __name__ == "__main__":

    def _make_and_check(
        in_ch: int,
        out_ch: int,
        expansion: int,
        strides: int,
        spatial: int = 80,
        tag: str = "",
    ) -> None:
        """Build a single-block model, run a forward pass, assert shape."""
        inp = keras.Input(shape=(spatial, spatial, in_ch))
        out = inverted_residual_block(
            inp,
            expansion=expansion,
            out_channels=out_ch,
            strides=strides,
            name="irb_test",
        )
        model = keras.Model(inp, out)
        dummy = np.random.uniform(-1, 1, (2, spatial, spatial, in_ch)).astype("float32")
        result = model(dummy, training=False).numpy()

        exp_spatial = spatial // strides
        exp_shape   = (2, exp_spatial, exp_spatial, out_ch)
        residual    = (strides == 1) and (in_ch == out_ch)

        assert result.shape == exp_shape, \
            f"[{tag}] Shape mismatch: got {result.shape}, expected {exp_shape}"

        status = "residual ✓" if residual else "no residual"
        print(
            f"  [{tag:30s}]  "
            f"({spatial}×{spatial}×{in_ch:3d}) t={expansion} s={strides} "
            f"→ {result.shape}   {status}"
        )

    print("=== Inverted Residual Block — shape tests ===\n")

    # Case 1: expansion=1, stride=1, no residual (first IRB, in≠out)
    _make_and_check(32, 16, expansion=1, strides=1, tag="t=1 s=1 32→16 no-res")

    # Case 2: expansion=6, stride=1, residual applies (in==out)
    _make_and_check(16, 16, expansion=6, strides=1, tag="t=6 s=1 16→16 residual")

    # Case 3: expansion=6, stride=2, no residual (spatial halved)
    _make_and_check(16, 24, expansion=6, strides=2, spatial=160, tag="t=6 s=2 16→24 no-res")

    # Case 4: expansion=6, stride=1, no residual (in≠out)
    _make_and_check(24, 24, expansion=6, strides=1, tag="t=6 s=1 24→24 residual")

    # Case 5: deeper channels — mimics C4 tap region
    _make_and_check(64, 96, expansion=6, strides=1, spatial=20, tag="t=6 s=1 64→96 no-res")

    print("\n✓ All inverted residual block tests passed")
