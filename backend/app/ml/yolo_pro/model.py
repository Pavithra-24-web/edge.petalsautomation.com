"""
yolo_pro/model.py
─────────────────
Top-level YOLO-Pro model assembly.

Wires every module built in previous steps into a single Keras Model:

    Image input
        └─ Backbone  (stem + 6 CSP-MobileNetV2 stages)
               └─ P3, P4, P5
                     └─ FPN Top-Down  (build_top_down)
                              └─ N3, N4
                                    └─ PAN Bottom-Up  (build_bottom_up)
                                             └─ N3, N4′, N5′
                                                   └─ Detection Heads ×3
                                                          └─ 6 output tensors

Output dictionary:
    {
        "cls_p3": (B, H/8,  W/8,  num_classes)   small  objects — sigmoid probs
        "reg_p3": (B, H/8,  W/8,  4 * reg_max)   small  objects — DFL logits
        "cls_p4": (B, H/16, W/16, num_classes)   medium objects — sigmoid probs
        "reg_p4": (B, H/16, W/16, 4 * reg_max)   medium objects — DFL logits
        "cls_p5": (B, H/32, W/32, num_classes)   large  objects — sigmoid probs
        "reg_p5": (B, H/32, W/32, 4 * reg_max)   large  objects — DFL logits
    }

Post-processing (NMS, DFL decode, anchor-point generation) is intentionally
kept outside this graph so the TFLite export stays clean.
"""

import tensorflow as tf
from tensorflow import keras

from .config        import YoloProConfig, get_config
from .backbone      import build_backbone
from .fpn_top_down  import build_top_down
from .pan_bottom_up import build_bottom_up
from .head          import detection_head


def build_yolo_pro(
    input_shape: tuple[int, int, int],
    num_classes: int,
    size:        str = "nano",
    reg_max:     int = 16,
    prior_prob:  float = 0.01,
    weights:     str | None = None,
) -> keras.Model:
    """
    Assemble the full YOLO-Pro detection model.

    Args:
        input_shape : (H, W, C) — H and W must be multiples of 32.
        num_classes : number of object classes (no background class).
        size        : model size variant — 'nano' | 'tiny' | 'small' | 'medium' | 'large'.
        reg_max     : DFL distribution bins (default 16, fixed in spec).
        prior_prob  : foreground prior for cls-head bias init.
        weights     : optional path to a .weights.h5 / .keras checkpoint
                      to load after graph construction.

    Returns:
        keras.Model with named outputs:
            cls_p3, reg_p3  — small-object scale   (stride  8, H/8)
            cls_p4, reg_p4  — medium-object scale  (stride 16, H/16)
            cls_p5, reg_p5  — large-object scale   (stride 32, H/32)

    Example:
        model = build_yolo_pro(
            input_shape=(320, 320, 3),
            num_classes=4,
            size='nano',
        )
        model.summary()

    Output shapes for input_shape=(320,320,3), num_classes=4, reg_max=16:
        cls_p3  (B, 40, 40,  4)      reg_p3  (B, 40, 40, 64)
        cls_p4  (B, 20, 20,  4)      reg_p4  (B, 20, 20, 64)
        cls_p5  (B, 10, 10,  4)      reg_p5  (B, 10, 10, 64)
    """
    cfg = get_config(size)

    # ── Input ─────────────────────────────────────────────────────────────────
    inputs = keras.Input(shape=input_shape, name="image")
    # (B, H, W, C)

    # ── Backbone: stem + 6 CSP-MobileNetV2 stages ─────────────────────────────
    p3, p4, p5 = build_backbone(inputs, cfg)
    # P3  (B, H/8,  W/8,  cfg.channels[2])
    # P4  (B, H/16, W/16, cfg.channels[4])
    # P5  (B, H/32, W/32, cfg.channels[5])

    # ── FPN Top-Down: semantic enrichment ─────────────────────────────────────
    n3, n4 = build_top_down(p3, p4, p5, cfg)
    # N3  (B, H/8,  W/8,  cfg.channels[2])
    # N4  (B, H/16, W/16, cfg.channels[4])

    # ── PAN Bottom-Up: spatial refinement ─────────────────────────────────────
    n3_out, n4_prime, n5_prime = build_bottom_up(n3, n4, p5, cfg)
    # N3      (B, H/8,  W/8,  cfg.channels[2])   → small
    # N4′     (B, H/16, W/16, cfg.channels[4])   → medium
    # N5′     (B, H/32, W/32, cfg.channels[5])   → large

    # ── Detection Heads: one per scale, weights not shared ────────────────────
    cls_p3, reg_p3 = detection_head(
        n3_out, num_classes, reg_max, name="head_p3", prior_prob=prior_prob
    )
    cls_p4, reg_p4 = detection_head(
        n4_prime, num_classes, reg_max, name="head_p4", prior_prob=prior_prob
    )
    cls_p5, reg_p5 = detection_head(
        n5_prime, num_classes, reg_max, name="head_p5", prior_prob=prior_prob
    )

    # Name the output tensors explicitly for clean SavedModel / TFLite export.
    # Use Activation("linear") instead of tf.identity — tf.identity is a TF op
    # and cannot be called on KerasTensors during functional model construction.
    cls_p3 = keras.layers.Activation("linear", name="cls_p3")(cls_p3)
    reg_p3 = keras.layers.Activation("linear", name="reg_p3")(reg_p3)
    cls_p4 = keras.layers.Activation("linear", name="cls_p4")(cls_p4)
    reg_p4 = keras.layers.Activation("linear", name="reg_p4")(reg_p4)
    cls_p5 = keras.layers.Activation("linear", name="cls_p5")(cls_p5)
    reg_p5 = keras.layers.Activation("linear", name="reg_p5")(reg_p5)

    # ── Assemble Keras Model ───────────────────────────────────────────────────
    model = keras.Model(
        inputs  = inputs,
        outputs = {
            "cls_p3": cls_p3,
            "reg_p3": reg_p3,
            "cls_p4": cls_p4,
            "reg_p4": reg_p4,
            "cls_p5": cls_p5,
            "reg_p5": reg_p5,
        },
        name = f"yolo_pro_{size}",
    )

    # ── Optional weight loading ────────────────────────────────────────────────
    if weights is not None:
        model.load_weights(weights)

    return model
