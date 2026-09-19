"""
yolo_pro/csp_irb.py
────────────────────
CSP-InvertedResidual Block (CSP-IRB) — the atomic backbone unit.

Tensor flow (one block):
─────────────────────────────────────────────────────────────────────────────

                     Input  (B, H, W, C_in)
                        │
              ┌─────────┴──────────┐
              │  channel split     │
              │   at C_in // 2     │
              │                    │
     bypass   │               branch
  (B,H,W,C_in//2)         (B,H,W,C_in//2)
              │                    │
              │          1×1 conv_bn  → expand to C_in//2 * t
              │                    │  (B, H,   W,   C_in//2 * t)
              │                    │
              │          3×3 dw_conv  → stride applies here
              │                    │  (B, H/s, W/s, C_in//2 * t)
              │                    │
              │          1×1 conv_bn  → project to C_out//2
              │                    │  (B, H/s, W/s, C_out//2)
              │                    │
              │      [residual add?]   only when:
              │                    │    stride == 1
              │                    │    C_in//2 == C_out//2
              │                    │  shortcut added to branch output
              │                    │
              └─────────┬──────────┘
                   Concatenate (axis=-1)
                   (B, H/s, W/s, C_in//2 + C_out//2)
                        │
                   1×1 conv_bn  → fuse to C_out
                        │
                   Output (B, H/s, W/s, C_out)

Key points:
  - The bypass half carries the identity gradient path (CSP property).
  - The stride-2 downsampling lives inside the 3×3 depthwise conv.
  - The residual add is on the *branch half only* (not the full tensor).
  - Channel counts:
      bypass   : C_in  // 2  (unchanged throughout)
      expanded : (C_in // 2) * t
      projected: C_out // 2
      fused out: C_out
─────────────────────────────────────────────────────────────────────────────
"""

import tensorflow as tf
from tensorflow.keras import layers

from .primitives import conv_bn, dw_conv


def csp_irb(
    x: tf.Tensor,
    filters_out: int,
    expand_ratio: int = 6,
    strides: int = 1,
    name: str = "",
) -> tf.Tensor:
    """
    CSP-InvertedResidual Block.

    Args:
        x            : input tensor   (B, H, W, C_in)
        filters_out  : total output channels C_out  (must be even)
        expand_ratio : expansion multiplier t for the MV2 inner bottleneck
                       (t=1 for Stage 0 / no expansion; t=6 elsewhere)
        strides      : spatial stride applied in the 3×3 dw conv
                       1 → same H×W,  2 → halve H×W
        name         : layer-name prefix

    Returns:
        Tensor  (B, H/strides, W/strides, filters_out)

    Residual add on the branch half is applied automatically when:
        strides == 1  AND  C_in // 2 == filters_out // 2
    """
    assert filters_out % 2 == 0, (
        f"filters_out must be even for CSP half-split; got {filters_out}"
    )

    p       = f"{name}_" if name else ""
    c_in    = x.shape[-1]
    half_in = c_in        // 2      # bypass channel width
    half_out= filters_out // 2      # branch output channel width
    c_exp   = half_in * expand_ratio  # expanded width inside bottleneck

    use_residual = (strides == 1) and (half_in == half_out)

    # ── 1. Channel split ─────────────────────────────────────────────────────
    # Slice is deterministic and index-stable; preferred over tf.split here
    # because it produces a single node in the graph with a clear name.
    bypass = layers.Lambda(
        lambda t: t[..., :half_in],
        name=f"{p}bypass",
    )(x)

    branch = layers.Lambda(
        lambda t: t[..., half_in:],
        name=f"{p}branch_in",
    )(x)

    # ── 2. Branch: 1×1 expansion ─────────────────────────────────────────────
    # Projects half_in channels up to c_exp for richer spatial mixing.
    # Skipped when expand_ratio == 1 (Stage 0 config, t=1).
    if expand_ratio != 1:
        branch = conv_bn(
            branch,
            filters     = c_exp,
            kernel_size = 1,
            strides     = 1,
            name        = f"{p}expand",
        )
    # shape: (B, H, W, c_exp)

    # ── 3. Branch: 3×3 depthwise conv (stride lives here) ────────────────────
    branch = dw_conv(
        branch,
        kernel_size = 3,
        strides     = strides,
        name        = f"{p}dw",
    )
    # shape: (B, H/s, W/s, c_exp)

    # ── 4. Branch: 1×1 projection → half_out channels ────────────────────────
    branch = conv_bn(
        branch,
        filters     = half_out,
        kernel_size = 1,
        strides     = 1,
        name        = f"{p}project",
    )
    # shape: (B, H/s, W/s, half_out)

    # ── 5. Optional residual add on the branch half only ─────────────────────
    # Mirrors MobileNetV2 shortcut semantics, but scoped to the branch half.
    # The bypass already carries an unmodified gradient; the add here gives
    # the branch its own skip connection for deeper configs.
    if use_residual:
        shortcut = layers.Lambda(
            lambda t: t[..., half_in:],
            name=f"{p}shortcut",
        )(x)
        branch = layers.Add(name=f"{p}residual_add")([branch, shortcut])
    # shape unchanged: (B, H/s, W/s, half_out)

    # ── 6. Concatenate bypass + branch ───────────────────────────────────────
    # bypass : (B, H/s, W/s, half_in)   ← note: if stride=2, bypass is still
    # Note: when strides=2 the bypass still has H×W (not H/s×W/s).
    # We must spatially align it before concat via an avg-pool when striding.
    if strides == 2:
        bypass = layers.AveragePooling2D(
            pool_size=2, strides=2, padding="same",
            name=f"{p}bypass_pool",
        )(bypass)
    # bypass shape now: (B, H/s, W/s, half_in)

    merged = layers.Concatenate(axis=-1, name=f"{p}concat")(
        [bypass, branch]
    )
    # shape: (B, H/s, W/s, half_in + half_out)

    # ── 7. 1×1 fusion conv → filters_out ─────────────────────────────────────
    out = conv_bn(
        merged,
        filters     = filters_out,
        kernel_size = 1,
        name        = f"{p}fuse",
    )
    # shape: (B, H/s, W/s, filters_out)

    return out
