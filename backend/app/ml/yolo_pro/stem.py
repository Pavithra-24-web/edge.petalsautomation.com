"""
yolo_pro/stem.py
────────────────
YOLO-Pro Stem — first layer of the backbone.

Architecture (Section 2.1 of spec):
    Conv2D 3×3, stride 2 → BatchNorm(momentum=0.9) → SiLU

The stem immediately halves spatial resolution and projects the input
channels into the first backbone feature width defined by the config.

Shape contract:
    Input  : (B, H,   W,   C_in)          e.g. (B, 320, 320, 3)
    Output : (B, H/2, W/2, channels[0])   e.g. (B, 160, 160, 24)  ← nano

H and W must both be multiples of 32 (spec requirement — P5 path has
stride 32 total).  The stem itself only enforces divisibility by 2, but
the constraint is documented here as the canonical validation point.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from .config     import YoloProConfig, get_config
from .primitives import conv_bn


# ─────────────────────────────────────────────────────────────────────────────
def build_stem(
    x: tf.Tensor,
    cfg: YoloProConfig,
    name: str = "stem",
) -> tf.Tensor:
    """
    YOLO-Pro Stem block.

    Applies a single Conv2D 3×3 (stride 2) + BN + SiLU to the input
    tensor, projecting it to cfg.channels[0] feature maps at half
    spatial resolution.

    Args:
        x    : input tensor  (B, H, W, C_in)
               H and W must be multiples of 32.
        cfg  : YoloProConfig — supplies channels[0] as the filter count.
        name : layer-name prefix (default 'stem').

    Returns:
        Tensor  (B, H/2, W/2, cfg.channels[0])

    Shape examples by variant:
        nano  (320×320×3)  →  (160×160×16)
        tiny  (320×320×3)  →  (160×160×24)
        small (640×640×3)  →  (320×320×48)
    """
    _, H, W, _ = x.shape
    assert H is None or H % 32 == 0, f"Input height {H} must be a multiple of 32"
    assert W is None or W % 32 == 0, f"Input width  {W} must be a multiple of 32"

    return conv_bn(
        x,
        filters     = cfg.channels[0],
        kernel_size = 3,
        strides     = 2,
        name        = name,
    )
