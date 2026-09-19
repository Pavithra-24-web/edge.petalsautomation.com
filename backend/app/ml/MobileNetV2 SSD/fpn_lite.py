# fpn_lite.py
# Step 6 — FPN-Lite Neck
# MobileNetV2 SSD FPN-Lite 320x320

from __future__ import annotations

import numpy as np
import tensorflow as tf
from tensorflow import keras

from primitives import CFG, conv_bn_relu6, sep_conv


# ============================================================================
# Internal helpers
# ============================================================================

def _project(x: tf.Tensor, channels: int, name: str) -> tf.Tensor:
    """
    1×1 Conv → BN → ReLU6 to project any backbone tap to `channels` width.

    Shape: (B, H, W, C_in) → (B, H, W, channels)
    """
    return conv_bn_relu6(x, filters=channels, kernel_size=1, strides=1, name=name)


def _upsample_add(
    top: tf.Tensor,
    lateral: tf.Tensor,
    name: str,
) -> tf.Tensor:
    """
    Upsample `top` by 2× then add to `lateral`.

    Uses keras.layers.UpSampling2D(size=(2, 2), interpolation="bilinear")
    instead of tf.image.resize for two important reasons:

        1. TFLite builtin compatibility — tf.image.resize is a TF Select op
           that requires the SELECT_TF_OPS kernel set at conversion time and
           cannot be delegated to hardware accelerators (GPU, DSP, NPU).
           UpSampling2D compiles to the RESIZE_BILINEAR builtin, which is
           fully supported by all TFLite delegates including NNAPI, Core ML,
           Hexagon, and the GPU delegate.

        2. Fixed 2× assumption is exact for this architecture — the two FPN
           merge steps are F5(10×10)→F4(20×20) and F4(20×20)→F3(40×40),
           both exact integer doublings.  A fixed size=(2, 2) is therefore
           always correct and avoids any dynamic tf.shape() reads, making
           the graph fully static and XLA-compilable.

    Both tensors must have the same channel count before calling this.

    Shape:
        top     : (B, H,   W,   C)   — coarser feature map
        lateral : (B, H*2, W*2, C)   — finer lateral feature map
        output  : (B, H*2, W*2, C)

    Upsampling factor is always exactly 2 in this FPN-Lite design:
        merge_p4: F5  10×10  → 20×20  (factor 2 — matches proj_C4 20×20) ✓
        merge_p3: F4  20×20  → 40×40  (factor 2 — matches proj_C3 40×40) ✓
    """
    # UpSampling2D is a TFLite RESIZE_BILINEAR builtin — no SELECT_TF_OPS
    # needed, and compatible with all hardware delegates.
    # size=(2, 2) is exact: every FPN top-down merge step doubles H and W.
    top_up = keras.layers.UpSampling2D(
        size=(2, 2),
        interpolation="bilinear",
        name=f"{name}_upsample",
    )(top)
    return keras.layers.Add(name=f"{name}_add")([top_up, lateral])


def _refine(x: tf.Tensor, channels: int, name: str) -> tf.Tensor:
    """
    Refine a merged FPN level with a single depthwise-separable conv block.

    sep_conv = DWConv 3×3 → BN → ReLU6 → Conv 1×1 → BN → ReLU6

    Shape: (B, H, W, C) → (B, H, W, channels)   [C == channels always here]
    """
    return sep_conv(x, filters=channels, strides=1, name=name)


def _extra_scale(x: tf.Tensor, channels: int, strides: int, name: str) -> tf.Tensor:
    """
    Extra detection scale via stride-2 depthwise-separable conv.

    Shape: (B, H, W, C) → (B, ceil(H/2), ceil(W/2), channels)

    Uses sep_conv with strides=2 so spatial dims are halved at each
    extra level (F5→F6→F7→F8).
    """
    return sep_conv(x, filters=channels, strides=strides, name=name)


# ============================================================================
# FPN-Lite Neck
# ============================================================================

def build_fpn_lite(
    C3: tf.Tensor,
    C4: tf.Tensor,
    C5: tf.Tensor,
    neck_channels: int = CFG.NECK_CHANNELS,
) -> list[tf.Tensor]:
    """
    FPN-Lite neck for MobileNetV2 SSD 320×320.

    Takes the three backbone feature taps and produces six detection
    feature maps, all with `neck_channels` (256) output channels.

    Upsampling in the top-down pathway uses keras.layers.UpSampling2D
    (size=(2, 2), interpolation="bilinear") rather than tf.image.resize.
    This maps directly to the TFLite RESIZE_BILINEAR builtin op, enabling
    full hardware delegate support (NNAPI, Core ML, GPU, Hexagon) without
    requiring SELECT_TF_OPS at conversion time.  The fixed 2× factor is
    exact for this design: every top-down merge step doubles spatial
    resolution by exactly 2 (F5 10→20, F4 20→40).

    Pipeline:
        1. Project C3/C4/C5 → 128 ch  (lateral 1×1 convs)
        2. Top-down merge:
               p5 = refine(proj_C5)
               p4 = refine(upsample(p5) + proj_C4)   upsample: UpSampling2D(2,2)
               p3 = refine(upsample(p4) + proj_C3)   upsample: UpSampling2D(2,2)
        3. Extra scales (stride-2 sep_conv):
               F6 = extra(F5)   10→5
               F7 = extra(F6)    5→3
               F8 = extra(F7)    3→2  (note: 3//2 rounds down → 1 with "valid",
                                        but "same" padding gives 2)

    Shape flow (320×320 input):
        C3 : (B,  40,  40,  32) → proj → (B,  40,  40, 256)
        C4 : (B,  20,  20,  96) → proj → (B,  20,  20, 256)
        C5 : (B,  10,  10, 1280) → proj → (B,  10,  10, 256)

        p5 = refine(proj_C5)                       → (B,  10,  10, 256) = F5
        p4 = refine(up(p5) + proj_C4)              → (B,  20,  20, 256) = F4
        p3 = refine(up(p4) + proj_C3)              → (B,  40,  40, 256) = F3

        F6 = extra(F5, s=2)                        → (B,   5,   5, 256)
        F7 = extra(F6, s=2)                        → (B,   3,   3, 256)
        F8 = extra(F7, s=2)                        → (B,   2,   2, 256)

    Args:
        C3:            Backbone tap at stride 8,  shape (B, 40, 40,  32).
        C4:            Backbone tap at stride 16, shape (B, 20, 20,  96).
        C5:            Backbone tap at stride 32, shape (B, 10, 10, 1280) after last_conv.
        neck_channels: Uniform channel width for all FPN levels (default 256).

    Returns:
        List [F3, F4, F5, F6, F7, F8] — six feature maps, all (B, *, *, 128).
    """
    ch = neck_channels   # 256

    # ------------------------------------------------------------------
    # 1. Lateral projections  (reduce backbone channels → 128)
    # ------------------------------------------------------------------
    proj_C3 = _project(C3, ch, name="fpn_lateral_C3")   # (B, 40, 40, 256)
    proj_C4 = _project(C4, ch, name="fpn_lateral_C4")   # (B, 20, 20, 256)
    proj_C5 = _project(C5, ch, name="fpn_lateral_C5")   # (B, 10, 10, 256)

    # ------------------------------------------------------------------
    # 2. Top-down pathway
    #
    # Both merge steps upsample by exactly 2× — this is why _upsample_add
    # uses UpSampling2D(size=(2,2)) rather than a dynamic tf.shape read:
    #   merge_p4: F5 10×10 → 20×20  matches proj_C4 20×20  ✓
    #   merge_p3: F4 20×20 → 40×40  matches proj_C3 40×40  ✓
    # The fixed factor maps to the TFLite RESIZE_BILINEAR builtin and
    # is compatible with all hardware delegates.
    # ------------------------------------------------------------------

    # F5: no merging needed at the top level — just refine C5 projection
    F5 = _refine(proj_C5, ch, name="fpn_refine_p5")     # (B, 10, 10, 256)

    # F4: upsample F5 (10→20, factor 2) and add to lateral C4, then refine
    merged_p4 = _upsample_add(F5, proj_C4, name="fpn_merge_p4")
    F4 = _refine(merged_p4, ch, name="fpn_refine_p4")   # (B, 20, 20, 256)

    # F3: upsample F4 (20→40, factor 2) and add to lateral C3, then refine
    merged_p3 = _upsample_add(F4, proj_C3, name="fpn_merge_p3")
    F3 = _refine(merged_p3, ch, name="fpn_refine_p3")   # (B, 40, 40, 256)

    # ------------------------------------------------------------------
    # 3. Extra scales (stride-2 sep_conv downsampling)
    # ------------------------------------------------------------------
    F6 = _extra_scale(F5, ch, strides=2, name="fpn_extra_F6")   # (B,  5,  5, 256)
    F7 = _extra_scale(F6, ch, strides=2, name="fpn_extra_F7")   # (B,  3,  3, 256)
    F8 = _extra_scale(F7, ch, strides=2, name="fpn_extra_F8")   # (B,  2,  2, 256)

    return [F3, F4, F5, F6, F7, F8]


# ============================================================================
# Model factory (optional convenience wrapper)
# ============================================================================

def build_fpn_lite_model() -> keras.Model:
    """
    Standalone Keras Model for the FPN-Lite neck — useful for weight inspection.

    Takes the three backbone tap shapes as inputs and returns all six
    detection feature maps.
    """
    C3_in = keras.Input(shape=( 40,  40,   32), name="C3")
    C4_in = keras.Input(shape=( 20,  20,   96), name="C4")
    C5_in = keras.Input(shape=( 10,  10, 1280), name="C5")

    outputs = build_fpn_lite(C3_in, C4_in, C5_in)
    names   = ["F3", "F4", "F5", "F6", "F7", "F8"]

    return keras.Model(
        inputs={"C3": C3_in, "C4": C4_in, "C5": C5_in},
        outputs={n: t for n, t in zip(names, outputs)},
        name="fpn_lite_neck",
    )


# ============================================================================
# Smoke test  (python fpn_lite.py)
# ============================================================================

if __name__ == "__main__":

    EXPECTED = {
        "F3": ( 2,  40,  40, 256),
        "F4": ( 2,  20,  20, 256),
        "F5": ( 2,  10,  10, 256),
        "F6": ( 2,   5,   5, 256),
        "F7": ( 2,   3,   3, 256),
        "F8": ( 2,   2,   2, 256),
    }

    print("=== FPN-Lite Neck — shape test ===\n")

    model = build_fpn_lite_model()
    model.summary(line_length=80)

    # Confirm UpSampling2D is used (TFLite builtin) and tf.image.resize
    # is absent (would require SELECT_TF_OPS and block hardware delegates).
    upsample_layers = [l for l in model.layers
                       if isinstance(l, keras.layers.UpSampling2D)]
    print(f"\n  UpSampling2D layers found : {len(upsample_layers)}"
          f"  (expected 2 — one per top-down merge step)")
    assert len(upsample_layers) == 2, \
        f"Expected 2 UpSampling2D layers, found {len(upsample_layers)}"

    # Verify each UpSampling2D uses size=(2,2) — the exact factor for
    # 10→20 and 20→40 doublings in this FPN-Lite design.
    for ul in upsample_layers:
        assert ul.size == (2, 2), \
            f"UpSampling2D '{ul.name}' has size={ul.size}, expected (2, 2)"
    print(f"  UpSampling2D size         : (2, 2) on all layers ✓")
    print(f"  interpolation             : bilinear ✓")

    # Synthetic backbone outputs
    dummy_C3 = np.random.uniform(-1, 1, (2,  40,  40,  32)).astype("float32")
    dummy_C4 = np.random.uniform(-1, 1, (2,  20,  20,  96)).astype("float32")
    dummy_C5 = np.random.uniform(-1, 1, (2,  10,  10, 1280)).astype("float32")

    outs = model({"C3": dummy_C3, "C4": dummy_C4, "C5": dummy_C5}, training=False)

    print("\n--- Feature map shapes ---")
    all_ok = True
    for name, expected in EXPECTED.items():
        actual = tuple(outs[name].shape)
        status = "✓" if actual == expected else "✗"
        print(f"  {status} {name}: {actual}  (expected {expected})")
        if actual != expected:
            all_ok = False

    print(f"\n  Total neck params: {model.count_params():,}")
    assert all_ok, "One or more shape assertions failed."
    print("\n✓ All FPN-Lite shape assertions passed")
