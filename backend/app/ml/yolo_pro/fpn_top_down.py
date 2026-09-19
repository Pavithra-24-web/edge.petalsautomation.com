"""
yolo_pro/fpn_top_down.py
────────────────────────
FPN Top-Down Path — semantic enrichment from deep → shallow.

Spec reference: Section 3.1 (Top-Down Path)

The top-down path flows semantic richness downward from the deepest,
most abstract feature (P5) toward the shallowest, most spatial feature
(P3). Each step:
    1. Upsample the deeper feature map 2× (nearest-neighbour, no params).
    2. Concatenate with the matching backbone tap at the same resolution.
    3. Fuse with a Conv-BN-SiLU to blend the two feature sets.

Full shape flow (nano, 320×320 input):
─────────────────────────────────────────────────────────────────────────
  Step     Operation          Input(s)                   Output
  ──────────────────────────────────────────────────────────────────────
  TD-1     Upsample2×         P5  (B,10,10,320)    →    (B,20,20,320)
  TD-2     Concat             [TD-1 ; P4]          →    (B,20,20,320+192)
                                                        = (B,20,20,512)
  TD-3     Conv-BN-SiLU 1×1   (B,20,20,512)        →  N4(B,20,20,192)

  TD-4     Upsample2×         N4  (B,20,20,192)    →    (B,40,40,192)
  TD-5     Concat             [TD-4; P3]            →    (B,40,40,192+80)
                                                        = (B,40,40,272)
  TD-6     Conv-BN-SiLU 1×1   (B,40,40,272)        →  N3(B,40,40, 80)
─────────────────────────────────────────────────────────────────────────

`out_ch` for each fuse step is taken from cfg:
    N4  →  cfg.channels[4]  (same as P4 backbone width)
    N3  →  cfg.channels[2]  (same as P3 backbone width)

This keeps the neck channels aligned with the backbone taps so the
subsequent bottom-up path can re-concatenate them cleanly.
"""

import tensorflow as tf
from tensorflow.keras import layers

from .config     import YoloProConfig
from .primitives import conv_bn


def fpn_fuse(
    feat_high: tf.Tensor,
    feat_low:  tf.Tensor,
    out_ch:    int,
    name:      str = "",
) -> tf.Tensor:
    """
    One FPN top-down fusion step.

    Args:
        feat_high : deeper feature map  (B, H,   W,   C_high)
                    — higher-level semantics, lower spatial resolution
        feat_low  : shallower feature   (B, 2H,  2W,  C_low)
                    — more spatial detail, higher resolution
        out_ch    : output channel count after fusion
        name      : layer-name prefix

    Returns:
        Tensor  (B, 2H, 2W, out_ch)

    Shape flow:
        feat_high (B,  H,  W, C_high)
            → Upsample2×       → (B, 2H, 2W, C_high)
            → Concat(feat_low) → (B, 2H, 2W, C_high + C_low)
            → Conv-BN-SiLU 1×1 → (B, 2H, 2W, out_ch)
    """
    p = f"{name}_" if name else ""

    # Step 1 — nearest-neighbour 2× upsample (no learnable params)
    upsampled = layers.UpSampling2D(
        size            = (2, 2),
        interpolation   = "nearest",
        name            = f"{p}upsample",
    )(feat_high)
    # shape: (B, 2H, 2W, C_high)

    # Step 2 — concatenate along channel axis
    merged = layers.Concatenate(axis=-1, name=f"{p}concat")(
        [upsampled, feat_low]
    )
    # shape: (B, 2H, 2W, C_high + C_low)

    # Step 3 — 1×1 conv-bn-silu to fuse and project to out_ch
    out = conv_bn(
        merged,
        filters     = out_ch,
        kernel_size = 1,
        name        = f"{p}fuse",
    )
    # shape: (B, 2H, 2W, out_ch)

    return out


def build_top_down(
    p3: tf.Tensor,
    p4: tf.Tensor,
    p5: tf.Tensor,
    cfg: YoloProConfig,
) -> tuple[tf.Tensor, tf.Tensor]:
    """
    Build the complete FPN top-down path.

    Uses the two fpn_fuse calls specified in Section 3.1:
        N4 = fpn_fuse(P5, P4, out_ch=cfg.channels[4])
        N3 = fpn_fuse(N4, P3, out_ch=cfg.channels[2])

    Args:
        p3  : Stage 2 backbone output  (B, H/8,  W/8,  cfg.channels[2])
        p4  : Stage 4 backbone output  (B, H/16, W/16, cfg.channels[4])
        p5  : Stage 5 backbone output  (B, H/32, W/32, cfg.channels[5])
        cfg : YoloProConfig — supplies out_ch values

    Returns:
        (N3, N4)
        N4  (B, H/16, W/16, cfg.channels[4])  — medium-object features
        N3  (B, H/8,  W/8,  cfg.channels[2])  — small-object features
    """
    # TD step 1-3: P5 (H/32) → upsample → fuse with P4 (H/16) → N4
    n4 = fpn_fuse(
        feat_high = p5,
        feat_low  = p4,
        out_ch    = cfg.channels[4],
        name      = "td_n4",
    )
    # N4: (B, H/16, W/16, cfg.channels[4])

    # TD step 4-6: N4 (H/16) → upsample → fuse with P3 (H/8) → N3
    n3 = fpn_fuse(
        feat_high = n4,
        feat_low  = p3,
        out_ch    = cfg.channels[2],
        name      = "td_n3",
    )
    # N3: (B, H/8, W/8, cfg.channels[2])

    return n3, n4
