"""
yolo_pro/pan_bottom_up.py
─────────────────────────
PAN Bottom-Up Path — spatial refinement from shallow → deep.

Spec reference: Section 3.2 (Bottom-Up Path)

The bottom-up path is a second pass through the feature pyramid.
After the top-down path enriched P3/P4/P5 with semantics, the
bottom-up path runs in the opposite direction: it re-introduces
fine-grained spatial structure back into the medium- and large-object
scales, using learned stride-2 convolutions instead of pooling.

Each step:
    1. Stride-2 Conv-BN-SiLU on the shallower enriched feature
       (halves H and W, matching the resolution of the skip).
    2. Concatenate with the matching top-down / backbone feature
       at the same resolution.
    3. Fuse with a Conv-BN-SiLU 1×1 to blend and project to out_ch.

Full shape flow (nano, 320×320 input):
─────────────────────────────────────────────────────────────────────────
  Step     Operation            Input(s)                    Output
  ──────────────────────────────────────────────────────────────────────
  BU-1     Conv3×3 s2           N3  (B,40,40, 80)  →  (B,20,20, 80)
  BU-2     Concat               [BU-1 ; N4]         →  (B,20,20, 80+192)
                                                       = (B,20,20,272)
  BU-3     Conv1×1-BN-SiLU      (B,20,20,272)       →  N4′(B,20,20,192)

  BU-4     Conv3×3 s2           N4′ (B,20,20,192)  →  (B,10,10,192)
  BU-5     Concat               [BU-4; P5]          →  (B,10,10,192+320)
                                                       = (B,10,10,512)
  BU-6     Conv1×1-BN-SiLU      (B,10,10,512)       →  N5′(B,10,10,320)
─────────────────────────────────────────────────────────────────────────

Final neck outputs feeding the three detection heads:
    N3   (B, H/8,  W/8,  cfg.channels[2])  → small  objects  (P3 head)
    N4′  (B, H/16, W/16, cfg.channels[4])  → medium objects  (P4 head)
    N5′  (B, H/32, W/32, cfg.channels[5])  → large  objects  (P5 head)

N3 passes straight through from the top-down path — no bottom-up
modification needed at the shallowest scale.
"""

import tensorflow as tf
from tensorflow.keras import layers

from .config     import YoloProConfig
from .primitives import conv_bn


def pan_fuse(
    feat_prev: tf.Tensor,
    feat_skip: tf.Tensor,
    out_ch:    int,
    name:      str = "",
) -> tf.Tensor:
    """
    One PAN bottom-up fusion step.

    Args:
        feat_prev : shallower enriched feature  (B, 2H, 2W, C_prev)
                    — comes from the previous (higher-resolution) neck output
        feat_skip : deeper skip connection      (B, H,  W,  C_skip)
                    — the N4 top-down output or the original P5 backbone tap
        out_ch    : output channel count after fusion
        name      : layer-name prefix

    Returns:
        Tensor  (B, H, W, out_ch)

    Shape flow:
        feat_prev (B, 2H, 2W, C_prev)
            → Conv3×3 s2          → (B, H,  W,  C_prev)   ← stride-2 learns
            → Concat(feat_skip)   → (B, H,  W,  C_prev + C_skip)
            → Conv1×1-BN-SiLU     → (B, H,  W,  out_ch)
    """
    p = f"{name}_" if name else ""

    # Step 1 — stride-2 conv to halve spatial resolution
    # Using a learned 3×3 conv rather than pooling: captures richer
    # downsampling features and is consistent with the spec's Conv2D s2.
    downsampled = conv_bn(
        feat_prev,
        filters     = feat_prev.shape[-1],  # preserve channel width before fuse
        kernel_size = 3,
        strides     = 2,
        name        = f"{p}down",
    )
    # shape: (B, H, W, C_prev)

    # Step 2 — concatenate with skip connection at same resolution
    merged = layers.Concatenate(axis=-1, name=f"{p}concat")(
        [downsampled, feat_skip]
    )
    # shape: (B, H, W, C_prev + C_skip)

    # Step 3 — 1×1 conv-bn-silu to fuse and project
    out = conv_bn(
        merged,
        filters     = out_ch,
        kernel_size = 1,
        name        = f"{p}fuse",
    )
    # shape: (B, H, W, out_ch)

    return out


def build_bottom_up(
    n3: tf.Tensor,
    n4: tf.Tensor,
    p5: tf.Tensor,
    cfg: YoloProConfig,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """
    Build the complete PAN bottom-up path.

    Uses the two pan_fuse calls specified in Section 3.2:
        N4′ = pan_fuse(N3,  N4, out_ch=cfg.channels[4])
        N5′ = pan_fuse(N4′, P5, out_ch=cfg.channels[5])

    N3 is passed through unchanged as the small-object neck output.

    Args:
        n3  : top-down N3 output  (B, H/8,  W/8,  cfg.channels[2])
        n4  : top-down N4 output  (B, H/16, W/16, cfg.channels[4])
        p5  : backbone P5 tap     (B, H/32, W/32, cfg.channels[5])
        cfg : YoloProConfig — supplies out_ch values

    Returns:
        (N3, N4_prime, N5_prime)
        N3       (B, H/8,  W/8,  cfg.channels[2])  → small-object head
        N4_prime (B, H/16, W/16, cfg.channels[4])  → medium-object head
        N5_prime (B, H/32, W/32, cfg.channels[5])  → large-object head
    """
    # BU step 1-3: N3 (H/8) downsample → fuse with N4 (H/16) → N4′
    n4_prime = pan_fuse(
        feat_prev = n3,
        feat_skip = n4,
        out_ch    = cfg.channels[4],
        name      = "bu_n4p",
    )
    # N4′: (B, H/16, W/16, cfg.channels[4])

    # BU step 4-6: N4′ (H/16) downsample → fuse with P5 (H/32) → N5′
    n5_prime = pan_fuse(
        feat_prev = n4_prime,
        feat_skip = p5,
        out_ch    = cfg.channels[5],
        name      = "bu_n5p",
    )
    # N5′: (B, H/32, W/32, cfg.channels[5])

    return n3, n4_prime, n5_prime
