"""
yolo_pro/head.py
────────────────
Decoupled Anchor-Free Detection Head.

One head is applied independently to each of the three neck outputs
(N3, N4′, N5′).  The classification and regression branches share the
same input feature but are processed through completely separate
convolutional stacks — "decoupled" means they never share weights.

Architecture per scale:
──────────────────────────────────────────────────────────────────────
                     Input feat  (B, Hg, Wg, C)
                          │
            ┌─────────────┴─────────────┐
            │                           │
      ── cls branch ──          ── reg branch ──
      conv3×3-BN-SiLU           conv3×3-BN-SiLU
            │                           │
      conv3×3-BN-SiLU           conv3×3-BN-SiLU
            │                           │
      conv1×1 (C → num_cls)     conv1×1 (C → 4*reg_max)
      Sigmoid                    (raw logits for DFL)
──────────────────────────────────────────────────────────────────────

Bias initialisation for cls head
──────────────────────────────────
  The final cls bias is initialised to  log(p / (1 - p))  where p is
  the expected object prior (default 0.01).  This makes the sigmoid
  output ≈ 0.01 at init, matching the typical foreground frequency and
  preventing the Varifocal loss from being dominated by trivially-large
  negative gradients in early training.  Without this, all sigmoid
  outputs start near 0.5 and the network spends many epochs just
  driving negatives to zero.
"""

import math
import tensorflow as tf
from tensorflow.keras import layers

from .primitives import conv_bn


def detection_head(
    feat:           tf.Tensor,
    num_classes:    int,
    reg_max:        int   = 16,
    name:           str   = "head",
    prior_prob:     float = 0.01,
) -> tuple[tf.Tensor, tf.Tensor]:
    """
    Single-scale decoupled anchor-free detection head.

    Args:
        feat        : neck feature tensor  (B, Hg, Wg, C)
        num_classes : number of object classes (no background)
        reg_max     : DFL distribution bins (default 16)
        name        : layer-name prefix — use unique names per scale
        prior_prob  : foreground prior for cls bias init (default 0.01)
                      Larger datasets with more objects per image can use 0.02–0.05.

    Returns:
        (cls_out, reg_out)

        cls_out  (B, Hg, Wg, num_classes)   Sigmoid class probabilities.
        reg_out  (B, Hg, Wg, 4 * reg_max)   Raw DFL logits, no activation.
    """
    p  = f"{name}_"
    ch = feat.shape[-1]

    # Bias init: sigmoid(bias) ≈ prior_prob → bias = log(p/(1-p))
    # Clipped to avoid inf for p→0 or p→1.
    prior_prob = float(max(prior_prob, 1e-6))
    cls_bias_init_val = math.log(prior_prob / (1.0 - prior_prob))
    cls_bias_init = tf.keras.initializers.Constant(cls_bias_init_val)

    # ── Classification branch ────────────────────────────────────────────────
    cls = conv_bn(feat, filters=ch, kernel_size=3, name=f"{p}cls_conv1")
    cls = conv_bn(cls,  filters=ch, kernel_size=3, name=f"{p}cls_conv2")
    cls_out = layers.Conv2D(
        filters            = num_classes,
        kernel_size        = 1,
        padding            = "same",
        use_bias           = True,
        kernel_initializer = "he_normal",
        bias_initializer   = cls_bias_init,   # prior-matched init
        name               = f"{p}cls_pred",
    )(cls)
    cls_out = layers.Activation("sigmoid", name=f"{p}cls_sigmoid")(cls_out)

    # ── Regression branch ────────────────────────────────────────────────────
    reg = conv_bn(feat, filters=ch, kernel_size=3, name=f"{p}reg_conv1")
    reg = conv_bn(reg,  filters=ch, kernel_size=3, name=f"{p}reg_conv2")
    reg_out = layers.Conv2D(
        filters            = 4 * reg_max,
        kernel_size        = 1,
        padding            = "same",
        use_bias           = True,
        kernel_initializer = "he_normal",
        bias_initializer   = "zeros",
        name               = f"{p}reg_pred",
    )(reg)
    # No activation — softmax is applied at decode/loss time

    return cls_out, reg_out