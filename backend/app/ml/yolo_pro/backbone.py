"""
yolo_pro/backbone.py
────────────────────
YOLO-Pro CSP-MobileNetV2 Backbone.

Wires the stem and six make_stage calls into a single callable that
returns the three FPN feature taps P3, P4, P5.

Full tensor flow (nano, 320×320×3 input):
──────────────────────────────────────────────────────────────────────────
  Layer           Op                          Output shape     Stride (cum)
  ──────────────────────────────────────────────────────────────────────
  Input                                       (B,320,320,  3)   ×1
  Stem            Conv3×3 s2 + BN + SiLU      (B,160,160, 24)   ×2
  Stage 0  ×1     CSP-IRB  s1  t=1            (B,160,160, 24)   ×2
  Stage 1  ×2     CSP-IRB  s2  t=6            (B, 80, 80, 40)   ×4
  Stage 2  ×3     CSP-IRB  s2  t=6            (B, 40, 40, 80)   ×8   → P3
  Stage 3  ×3     CSP-IRB  s1  t=6            (B, 40, 40,112)   ×8
  Stage 4  ×4     CSP-IRB  s2  t=6            (B, 20, 20,192)   ×16  → P4
  Stage 5  ×2     CSP-IRB  s2  t=6            (B, 10, 10,320)   ×32  → P5
──────────────────────────────────────────────────────────────────────────

FPN taps:
    P3  =  Stage 2 output  (H/8,  small objects,  most spatial detail)
    P4  =  Stage 4 output  (H/16, medium objects)
    P5  =  Stage 5 output  (H/32, large objects,  richest semantics)

Stage 3 is a refinement stage (stride=1) — it deepens Stage 2's P3
features before they flow into Stage 4. Its output is NOT tapped; only
Stage 2's output is P3.
"""

import tensorflow as tf
from tensorflow import keras

from .config import YoloProConfig, get_config
from .stem   import build_stem
from .stage  import make_stage


# ── Per-stage specification (index-stable, matches spec Section 2.2) ─────────
#   (stride, expand_ratio, is_fpn_tap)
_STAGE_SPEC = [
    (1, 1, False),   # Stage 0 — t=1, no spatial change
    (2, 6, False),   # Stage 1
    (2, 6, True),    # Stage 2 → P3  (H/8)
    (1, 6, False),   # Stage 3 — refinement, stride=1
    (2, 6, True),    # Stage 4 → P4  (H/16)
    (2, 6, True),    # Stage 5 → P5  (H/32)
]


def build_backbone(
    inputs: tf.Tensor,
    cfg: YoloProConfig,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """
    Build the YOLO-Pro CSP-MobileNetV2 backbone.

    Args:
        inputs : Keras input tensor  (B, H, W, C)
                 H and W must be multiples of 32.
        cfg    : YoloProConfig for the desired size variant.

    Returns:
        (P3, P4, P5) — three feature tensors for the FPN neck:
            P3  (B, H/8,  W/8,  cfg.channels[2])   small objects
            P4  (B, H/16, W/16, cfg.channels[4])   medium objects
            P5  (B, H/32, W/32, cfg.channels[5])   large objects
    """
    fpn_taps: list[tf.Tensor] = []

    # ── Stem ─────────────────────────────────────────────────────────────────
    x = build_stem(inputs, cfg, name="stem")
    # (B, H/2, W/2, cfg.channels[0])

    # ── Six backbone stages ───────────────────────────────────────────────────
    for idx, (stride, t, is_tap) in enumerate(_STAGE_SPEC):
        x = make_stage(
            x,
            out_ch       = cfg.channels[idx],
            n_blocks     = cfg.depths[idx],
            stride       = stride,
            expand_ratio = t,
            name         = f"s{idx}",
        )
        if is_tap:
            fpn_taps.append(x)

    assert len(fpn_taps) == 3, "Backbone must produce exactly 3 FPN taps"
    p3, p4, p5 = fpn_taps
    return p3, p4, p5


def build_backbone_model(
    input_shape: tuple[int, int, int],
    cfg: YoloProConfig,
) -> keras.Model:
    """
    Wrap build_backbone in a standalone Keras Model.

    Useful for weight inspection, unit-testing, or transfer-learning from
    a pretrained backbone checkpoint.

    Args:
        input_shape : (H, W, C), e.g. (320, 320, 3)
        cfg         : YoloProConfig

    Returns:
        keras.Model with outputs [P3, P4, P5]
    """
    inputs      = keras.Input(shape=input_shape, name="image")
    p3, p4, p5  = build_backbone(inputs, cfg)
    return keras.Model(
        inputs  = inputs,
        outputs = [p3, p4, p5],
        name    = f"yolo_pro_{cfg.name}_backbone",
    )
