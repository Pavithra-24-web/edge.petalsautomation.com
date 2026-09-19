"""
Model Builder — constructs Keras model architectures for edge deployment.

Architectures:
  dense                — Fully-connected neural network (good for flat feature vectors)
  conv1d               — 1D CNN (time-series, audio)
  conv2d               — 2D CNN (spectrograms, images)
  lstm                 — LSTM (sequential sensor data)
  mobilenet            — MobileNetV2 (image classification)
  transfer             — Transfer learning from MobileNetV2
  fomo_mobilenetv2_0_1 — FOMO v1 (Legacy 96×96, stride-16, block_13) or
                         FOMO v2 (Adaptive Resolution, stride-8, block_6)
                         dispatched via params["fomo_version"] (default 1).
  yolo_pro             — YOLO-Pro anchor-free multi-scale object detection
"""
import logging
import tensorflow as tf
from tensorflow import keras
from typing import Tuple, Dict, Any

logger = logging.getLogger(__name__)

# Architectures that are recognized but require specialised training pipelines.
# build_model() raises NotImplementedError for these so callers get a clear,
# structured signal instead of silently falling back to a wrong architecture.
_RECOGNISED_NOT_IMPLEMENTED: set = {
    # "fomo_v2_0.35",
    # "mobilenet_v2_ssd_fpn_lite",
    # yolo_pro removed from this set — now fully implemented via app.ml.yolo_pro
}


def build_model(
    architecture: str,
    input_shape: Tuple,
    num_classes: int,
    params: Dict[str, Any] = None,
) -> keras.Model:
    params = params or {}

    builders = {
        "dense":                _build_dense,
        "conv1d":               _build_conv1d,
        "conv2d":               _build_conv2d,
        "lstm":                 _build_lstm,
        "mobilenet":            _build_mobilenet,
        "transfer":             _build_transfer,
        "fomo_mobilenetv2_0_1": _build_fomo_mobilenetv2_0_1,
        "yolo_pro":             _build_yolo_pro_wrapper,
    }

    if architecture in _RECOGNISED_NOT_IMPLEMENTED:
        raise NotImplementedError(
            f"Architecture '{architecture}' is recognized by the backend but its "
            "full training pipeline (loss functions, dataset format, etc.) is not yet implemented. "
            "Please select a standard classification model for now."
        )

    fn = builders.get(architecture)
    if fn is None:
        raise ValueError(
            f"Unknown architecture '{architecture}'. "
            f"Valid options: {sorted(builders.keys())}"
        )
    return fn(input_shape, num_classes, params)


# ─── YOLO-Pro ─────────────────────────────────────────────────────────────────

def _build_yolo_pro_wrapper(input_shape, num_classes, params):
    """
    Thin adapter that calls app.ml.yolo_pro.build_yolo_pro with the
    backend's standard (input_shape, num_classes, params) signature.

    Expected params keys (all optional — sensible defaults apply):
        size    : 'nano' | 'tiny' | 'small' | 'medium' | 'large'  (default 'nano')
        reg_max : DFL distribution bins       (default 16)
        weights : path to a .keras checkpoint (default None — train from scratch)

    Input shape contract:
        input_shape must be a 3-tuple (H, W, C) with H and W both
        divisible by 32, minimum 96×96.  The DSP image block should be
        configured to match (e.g. 320×320×3 or 416×416×3).
    """
    from app.ml.yolo_pro import build_yolo_pro

    size    = params.get("size",    "nano")
    reg_max = params.get("reg_max", 16)
    weights = params.get("weights", None)

    if len(input_shape) != 3:
        raise ValueError(
            f"yolo_pro requires a 3-D input shape (H, W, C); got {input_shape}. "
            "Configure the DSP image block so the output is (H, W, C)."
        )
    H, W, C = input_shape
    if H % 32 != 0 or W % 32 != 0:
        raise ValueError(
            f"yolo_pro requires H and W to be multiples of 32; got ({H}, {W}). "
            "Use image dimensions like 320×320 or 416×416."
        )
    if H < 96 or W < 96:
        raise ValueError(
            f"yolo_pro minimum input size is 96×96; got ({H}×{W})."
        )

    logger.info(
        f"[model_builder] Building YOLO-Pro ({size}) — "
        f"input={input_shape}  num_classes={num_classes}  reg_max={reg_max}"
    )
    return build_yolo_pro(
        input_shape=input_shape,
        num_classes=num_classes,
        size=size,
        reg_max=reg_max,
        weights=weights,
    )


# ─── Dense ────────────────────────────────────────────────────────────────────

def _build_dense(input_shape, num_classes, params):
    units       = params.get("units", [128, 64])
    dropout     = params.get("dropout", 0.3)
    activation  = params.get("activation", "relu")

    inp = keras.Input(shape=input_shape, name="input")
    x   = keras.layers.Flatten()(inp)
    for u in units:
        x = keras.layers.Dense(u, activation=activation)(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Dropout(dropout)(x)
    out = keras.layers.Dense(num_classes, activation="softmax", name="output")(x)
    return keras.Model(inp, out, name="dense_classifier")


# ─── Conv1D ───────────────────────────────────────────────────────────────────

def _build_conv1d(input_shape, num_classes, params):
    filters    = params.get("filters", [32, 64, 128])
    kernel     = params.get("kernel_size", 3)
    pool       = params.get("pool_size", 2)
    dense_units= params.get("dense_units", [128])
    dropout    = params.get("dropout", 0.3)

    # Ensure 3D input (samples, timesteps, channels)
    if len(input_shape) == 1:
        reshape_shape = (input_shape[0], 1)
    else:
        reshape_shape = input_shape

    inp = keras.Input(shape=input_shape, name="input")
    x = keras.layers.Reshape(reshape_shape)(inp) if len(input_shape) == 1 else inp

    for f in filters:
        x = keras.layers.Conv1D(f, kernel, activation="relu", padding="same")(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.MaxPooling1D(pool)(x)

    x = keras.layers.GlobalAveragePooling1D()(x)
    for u in dense_units:
        x = keras.layers.Dense(u, activation="relu")(x)
        x = keras.layers.Dropout(dropout)(x)

    out = keras.layers.Dense(num_classes, activation="softmax", name="output")(x)
    return keras.Model(inp, out, name="conv1d_classifier")


# ─── Conv2D ───────────────────────────────────────────────────────────────────

def _build_conv2d(input_shape, num_classes, params):
    filters   = params.get("filters", [16, 32, 64])
    kernel    = params.get("kernel_size", (3, 3))
    dense_units = params.get("dense_units", [128])
    dropout   = params.get("dropout", 0.3)

    # If input is flat, try to reshape to 2D
    if len(input_shape) == 1:
        import math
        side = int(math.sqrt(input_shape[0]))
        if side * side == input_shape[0]:
            reshape_to = (side, side, 1)
        else:
            reshape_to = (input_shape[0], 1, 1)
    elif len(input_shape) == 2:
        reshape_to = (*input_shape, 1)
    else:
        reshape_to = input_shape

    inp = keras.Input(shape=input_shape, name="input")
    x   = keras.layers.Reshape(reshape_to)(inp)

    for f in filters:
        x = keras.layers.Conv2D(f, kernel, activation="relu", padding="same")(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.MaxPooling2D((2, 2))(x)

    x = keras.layers.GlobalAveragePooling2D()(x)
    for u in dense_units:
        x = keras.layers.Dense(u, activation="relu")(x)
        x = keras.layers.Dropout(dropout)(x)

    out = keras.layers.Dense(num_classes, activation="softmax", name="output")(x)
    return keras.Model(inp, out, name="conv2d_classifier")


# ─── LSTM ─────────────────────────────────────────────────────────────────────

def _build_lstm(input_shape, num_classes, params):
    units    = params.get("lstm_units", [64, 32])
    dropout  = params.get("dropout", 0.3)
    rec_drop = params.get("recurrent_dropout", 0.1)
    dense_u  = params.get("dense_units", 64)

    if len(input_shape) == 1:
        reshape_to = (input_shape[0], 1)
    else:
        reshape_to = input_shape

    inp = keras.Input(shape=input_shape, name="input")
    x   = keras.layers.Reshape(reshape_to)(inp) if len(input_shape) == 1 else inp

    for i, u in enumerate(units):
        return_seq = (i < len(units) - 1)
        x = keras.layers.LSTM(u, return_sequences=return_seq,
                               dropout=dropout, recurrent_dropout=rec_drop)(x)

    x   = keras.layers.Dense(dense_u, activation="relu")(x)
    x   = keras.layers.Dropout(dropout)(x)
    out = keras.layers.Dense(num_classes, activation="softmax", name="output")(x)
    return keras.Model(inp, out, name="lstm_classifier")


# ─── MobileNet ────────────────────────────────────────────────────────────────

def _build_mobilenet(input_shape, num_classes, params):
    """Lightweight MobileNetV2 — for image inputs (96x96x3 or similar)."""
    alpha      = params.get("alpha", 0.35)
    dropout    = params.get("dropout", 0.2)
    img_size   = params.get("image_size", 96)

    inp = keras.Input(shape=input_shape, name="input")

    # Reshape / resize to (img_size, img_size, 3)
    if len(input_shape) == 1:
        x = keras.layers.Reshape((img_size, img_size, 1))(inp)
        x = keras.layers.Conv2D(3, 1, padding="same")(x)
    elif len(input_shape) == 2:
        x = keras.layers.Reshape((*input_shape, 1))(inp)
        x = keras.layers.Conv2D(3, 1, padding="same")(x)
    else:
        x = inp

    x = keras.layers.Resizing(img_size, img_size)(x)
    x = keras.layers.Rescaling(1.0 / 127.5, offset=-1)(x)

    base = keras.applications.MobileNetV2(
        input_shape=(img_size, img_size, 3),
        alpha=alpha,
        include_top=False,
        weights=None,
    )
    x = base(x)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dropout(dropout)(x)
    out = keras.layers.Dense(num_classes, activation="softmax", name="output")(x)
    return keras.Model(inp, out, name="mobilenet_classifier")


# ─── Transfer Learning ────────────────────────────────────────────────────────

def _build_transfer(input_shape, num_classes, params):
    """MobileNetV2 with ImageNet weights, fine-tune top layers."""
    img_size  = params.get("image_size", 96)
    freeze_pct= params.get("freeze_pct", 0.7)
    dropout   = params.get("dropout", 0.2)
    dense_u   = params.get("dense_units", 128)

    inp = keras.Input(shape=input_shape, name="input")
    x   = keras.layers.Resizing(img_size, img_size)(inp)
    x   = keras.layers.Rescaling(1.0 / 127.5, offset=-1)(x)

    base = keras.applications.MobileNetV2(
        input_shape=(img_size, img_size, 3),
        include_top=False,
        weights="imagenet",
    )
    freeze_up_to = int(len(base.layers) * freeze_pct)
    for layer in base.layers[:freeze_up_to]:
        layer.trainable = False

    x   = base(x, training=False)
    x   = keras.layers.GlobalAveragePooling2D()(x)
    x   = keras.layers.Dense(dense_u, activation="relu")(x)
    x   = keras.layers.Dropout(dropout)(x)
    out = keras.layers.Dense(num_classes, activation="softmax", name="output")(x)
    return keras.Model(inp, out, name="transfer_classifier")


# ─── FOMO MobileNetV2 0.35 ────────────────────────────────────────────────────

def _build_fomo_mobilenetv2_0_1(input_shape, num_classes, params):
    """
    FOMO (Faster Objects, More Objects) — MobileNetV2 alpha=0.35 backbone.

    Architecture overview
    ---------------------
    FOMO replaces the classification head of MobileNetV2 with a per-pixel
    (heatmap) head so the network can localise objects without bounding-box
    regression.  The backbone is frozen MobileNetV2 at width-multiplier 0.35,
    the smallest alpha that ships with pretrained ImageNet weights in Keras.

    Output shape:  (batch, grid_h, grid_w, num_classes + 1)
        - last axis:  class 0 = background, classes 1…N = objects
        - grid size = image_size / 16  (MobileNetV2 stride-16 feature map, block_13_expand_relu)

    Input preprocessing contract
    ----------------------------
    The DSP image block (DSPProcessor._image) outputs pixel values in **[0, 1]**
    (it divides raw uint8 pixels by 255).  This model's "normalize" Rescaling
    layer maps that [0, 1] range to the [-1, 1] range expected by MobileNetV2:

        output = input * 2.0 - 1.0   (Rescaling(2.0, offset=-1))

    Do NOT pass [0, 255] raw pixels directly — they would saturate at +1.0.

    Training notes
    ---------------
    • Loss:    weighted softmax cross-entropy over raw logits per cell
               (_fomo_weighted_softmax_loss, bg_weight=0.1).
    • Labels:  dense heatmap targets of shape (grid_h, grid_w, num_classes+1),
               not integer class indices.
    • Head:    two-stage — Conv2D(32, relu) → Conv2D(num_classes+1, linear).
    • training_worker.py detects FOMO via ``"fomo" in architecture.lower()``,
      compiles with _fomo_weighted_softmax_loss, and builds per-cell heatmap
      targets via ``_create_fomo_heatmap()`` — the standard
      sparse_categorical_crossentropy path is bypassed entirely.

    Parameters (from params dict)
    -----------------------------
    alpha        float  0.35   MobileNetV2 width multiplier.
    image_width  int    96     Input width  (multiple of 8).
    image_height int    96     Input height (multiple of 8).
    """
    # ── FOMO version dispatch ─────────────────────────────────────────────────
    # Guard: all v2 logic lives in _build_fomo_mobilenetv2_v2.
    # This function's code below is v1-only and is never executed for v2.
    if int(params.get("fomo_version", 1)) == 2:
        return _build_fomo_mobilenetv2_v2(input_shape, num_classes, params)

    alpha = params.get("alpha", 0.35)

    # Derive the processing image size from the actual DSP output shape.
    #
    # For 2-D or 3-D image inputs (H×W) or (H×W×C): use input_shape as the
    # single source of truth and snap up to the nearest multiple of 8.
    # MobileNetV2's stride schedule requires multiples of 8, but the DSP may
    # emit non-aligned sizes (e.g. 71×71 or 54×54).  A small Resizing layer
    # compensates; grid_h/grid_w in the training worker use the same formula
    # so heatmap targets always match the model output shape.
    #
    # For flat (1-D) inputs only: fall back to params and validate strictly —
    # the caller must explicitly configure a valid FOMO image size.
    if len(input_shape) >= 2:
        raw_h, raw_w = input_shape[0], input_shape[1]
        image_height = raw_h if raw_h % 8 == 0 else raw_h + (8 - raw_h % 8)
        image_width  = raw_w if raw_w % 8 == 0 else raw_w + (8 - raw_w % 8)
    else:
        image_width  = params.get("image_width",  96)
        image_height = params.get("image_height", 96)
        if image_width % 8 != 0 or image_height % 8 != 0:
            raise ValueError(
                f"FOMO requires image dimensions that are multiples of 8. "
                f"Got ({image_width}, {image_height})."
            )

    # ── Input ────────────────────────────────────────────────────────────────
    inp = keras.Input(shape=input_shape, name="input")

    # Normalise to [-1, 1] as expected by MobileNetV2 pre-processing
    if len(input_shape) == 1:
        # Flat vector → reshape into (H, W, 1) then expand to 3 channels
        x = keras.layers.Reshape((image_height, image_width, 1))(inp)
        x = keras.layers.Conv2D(3, 1, padding="same", use_bias=False,
                                name="channel_expand")(x)
    elif len(input_shape) == 2:
        x = keras.layers.Reshape((*input_shape, 1))(inp)
        x = keras.layers.Conv2D(3, 1, padding="same", use_bias=False,
                                name="channel_expand")(x)
    else:
        x = inp

    # Only add Resizing when the spatial dimensions don't already match.
    # A no-op Resizing layer is harmless at runtime but can confuse MLIR
    # lowering in some TFLite versions, so we skip it when possible.
    if x.shape[1] != image_height or x.shape[2] != image_width:
        x = keras.layers.Resizing(image_height, image_width, name="resize")(x)
    # DSP image block outputs [0, 1]; map to MobileNetV2's expected [-1, 1].
    # Formula: output = input * 2.0 - 1.0
    x = keras.layers.Rescaling(2.0, offset=-1, name="normalize")(x)

    # ── MobileNetV2 backbone — pretrained ImageNet initialization ────────────
    #
    # Use input_tensor=x so that the backbone layers are wired directly into
    # the outer functional graph — this avoids nested keras.Model calls which
    # generate PartitionedCall ops that TFLite's MLIR lowering cannot handle.
    #
    # Pretrained weights are the single highest-impact change for small datasets
    # (< 500 images): the backbone immediately produces meaningful edge/texture/
    # shape features, giving the head a signal to learn from on epoch 1 instead
    # of pure random noise.  Without this, even 100 epochs is not enough for
    # a 10 K-param network to learn visual features from scratch on ~372 samples.
    #
    # Alpha=0.35 is the smallest value that ships with pretrained ImageNet weights
    # in Keras.  The exported TFLite model is ~200 KB — suitable for most edge
    # targets that have a Python runtime.
    _backbone_alpha = alpha
    backbone = keras.applications.MobileNetV2(
        input_tensor=x,
        alpha=_backbone_alpha,
        include_top=False,
        weights="imagenet",
    )

    # Speed up BatchNorm running-stat convergence for small datasets.
    # Keras default momentum=0.99 takes ~230 batches to converge; with a
    # small dataset (~5 batches/epoch) that is >46 epochs of unstable BN stats.
    # momentum=0.9 converges in ~22 batches (~5 epochs), matching the Edge
    # Impulse FOMO reference build and ensuring stable inference from epoch 5.
    for _layer in backbone.layers:
        if isinstance(_layer, keras.layers.BatchNormalization):
            _layer.momentum = 0.9

    # Stride-16 feature map — use the output of block_13_expand_relu.
    # For alpha=0.35 this layer has 192 channels and a spatial resolution of
    # image_size / 16 (6×6 for 96×96 input).  Prior tap at block_6_expand_relu
    # (stride-8, 96ch) produced ratio=1.223 because the shallow features lacked
    # abstract object-level discrimination.  block_13 sits at the transition to
    # stride-32 and contains richer semantic content at the cost of a smaller grid.
    # For very small alpha the exact layer name may vary; we fall back to the
    # full backbone output (stride-32) if block_13 is absent..

    try:
        features = backbone.get_layer("block_13_expand_relu").output
    except ValueError:
        # Fallback: use the full backbone output (stride-32)
        features = backbone.output

    # ── FOMO head — two-stage 1×1 conv heatmap ───────────────────────────────
    # Mirrors the Edge Impulse FOMO head design:
    #
    #   Stage 1 — intermediate projection:
    #     Conv2D(32, 1×1, activation="relu", name="fomo_head_conv")
    #     Projects 192-channel block_13 features into a 32-channel space.
    #     A deeper head (128→96→3) created a bottleneck on 96-channel features
    #     that suppressed max_conf from 0.72 → 0.54; reverting to 2-stage keeps
    #     the head simple and lets the richer stride-16 features do the work.
    #
    #   Stage 2 — output logits (no activation):
    #     Conv2D(num_classes+1, 1×1, activation=None, name="fomo_head")
    #     Raw logits: channel 0 = background, channels 1..N = object classes.
    #     No activation — the training loss (_fomo_weighted_softmax_loss) applies
    #     log_softmax internally and must receive logits, not probabilities.
    #
    # At inference: apply tf.nn.softmax(preds, axis=-1) before thresholding.
    # The TFLite export model bakes softmax in automatically.
    # See _evaluate_fomo_detection and _FomoValF1Callback in training_worker.py.
    head_classes = num_classes + 1
    x = keras.layers.Conv2D(
        32, kernel_size=1, padding="same",
        activation="relu", name="fomo_head_conv",
    )(features)
    x = keras.layers.Conv2D(
        head_classes, kernel_size=1, padding="same",
        activation=None, name="fomo_head",
    )(x)

    return keras.Model(inp, x, name="fomo_mobilenetv2_0_1")


# ─── FOMO v2 — Adaptive Resolution (stride-8) ─────────────────────────────────

def build_fomo_v2(input_shape, num_classes: int, params: dict = None):
    """Public entry point for FOMO v2 (stride-8, adaptive resolution).

    Identical backbone and head to v1 but taps block_6_expand_relu (stride-8)
    instead of block_13_expand_relu (stride-16), producing a grid that is 4×
    larger in area:  img_h//8 × img_w//8  versus  img_h//16 × img_w//16.

    Example grids (square inputs):
        128×128 → 16×16  (256 cells)
        160×160 → 20×20  (400 cells)
        192×192 → 24×24  (576 cells)

    Parameters
    ----------
    input_shape : tuple  e.g. (128, 128, 3)
    num_classes : int    number of object classes (background added internally)
    params      : dict   optional; keys forwarded to v1 builder (alpha, …)
    """
    params = dict(params or {})
    params["fomo_version"] = 2
    return _build_fomo_mobilenetv2_0_1(input_shape, num_classes, params)


def _build_fomo_mobilenetv2_v2(input_shape, num_classes, params):
    """FOMO v2 internal builder — FPN-style fusion of stride-8 + stride-16 features.

    Guard: only called when params["fomo_version"] == 2.
    All v1 code paths are untouched at runtime.

    Architecture
    ------------
    v1 uses block_13_expand_relu (stride-16, deep semantic features) directly.
    v2 uses a Feature Pyramid Network (FPN)-style fusion to combine:
      • block_6_expand_relu  — stride-8, 96ch  — spatial detail, shallow
      • block_13_expand_relu — stride-16, 192ch — semantic richness, deep
    The stride-16 features are projected to 96ch then upsampled 2× to stride-8
    spatial resolution and added element-wise.  This gives the head both the
    positional precision of stride-8 and the semantic quality of stride-16,
    which alone is too shallow to localise objects reliably.

    Output grid = image_h/8 × image_w/8:
        128×128 → 16×16  (256 cells)
        160×160 → 20×20  (400 cells)
        192×192 → 24×24  (576 cells)

    FPN layers are named for testability:
        fomo_v2_sem_proj   Conv2D(96, 1×1) — projects stride-16 to 96ch
        fomo_v2_upsample   Resizing(H/8, W/8) — bilinear upsample to stride-8
        fomo_v2_fusion     Add([stride8, semantic]) — element-wise fusion

    Head uses 64 filters (vs 32 in v1) to handle the richer fused features.
    """
    alpha = params.get("alpha", 0.35)

    if len(input_shape) >= 2:
        raw_h, raw_w = input_shape[0], input_shape[1]
        image_height = raw_h if raw_h % 8 == 0 else raw_h + (8 - raw_h % 8)
        image_width  = raw_w if raw_w % 8 == 0 else raw_w + (8 - raw_w % 8)
    else:
        image_width  = params.get("image_width",  128)
        image_height = params.get("image_height", 128)
        if image_width % 8 != 0 or image_height % 8 != 0:
            raise ValueError(
                f"FOMO v2 requires image dimensions divisible by 8. "
                f"Got ({image_width}, {image_height})."
            )

    inp = keras.Input(shape=input_shape, name="input")

    if len(input_shape) == 1:
        x = keras.layers.Reshape((image_height, image_width, 1))(inp)
        x = keras.layers.Conv2D(3, 1, padding="same", use_bias=False,
                                name="channel_expand")(x)
    elif len(input_shape) == 2:
        x = keras.layers.Reshape((*input_shape, 1))(inp)
        x = keras.layers.Conv2D(3, 1, padding="same", use_bias=False,
                                name="channel_expand")(x)
    else:
        x = inp

    if x.shape[1] != image_height or x.shape[2] != image_width:
        x = keras.layers.Resizing(image_height, image_width, name="resize")(x)
    x = keras.layers.Rescaling(2.0, offset=-1, name="normalize")(x)

    backbone = keras.applications.MobileNetV2(
        input_tensor=x,
        alpha=alpha,
        include_top=False,
        weights="imagenet",
    )
    for _layer in backbone.layers:
        if isinstance(_layer, keras.layers.BatchNormalization):
            _layer.momentum = 0.9

    # ── Stride-8 features (spatial detail) ────────────────────────────────────
    _stride8_candidates = [
        "block_6_expand_relu",
        "block_6_depthwise_relu",
        "block_5_expand_relu",
        "block_4_expand_relu",
    ]
    features_s8 = None
    _s8_name = None
    for _n in _stride8_candidates:
        try:
            features_s8 = backbone.get_layer(_n).output
            _s8_name = _n
            break
        except ValueError:
            continue

    # ── Stride-16 features (semantic richness) ────────────────────────────────
    _stride16_candidates = [
        "block_13_expand_relu",
        "block_12_expand_relu",
        "block_11_expand_relu",
    ]
    features_s16 = None
    _s16_name = None
    for _n in _stride16_candidates:
        try:
            features_s16 = backbone.get_layer(_n).output
            _s16_name = _n
            break
        except ValueError:
            continue

    # ── FPN-style fusion ──────────────────────────────────────────────────────
    # Project stride-16 features to 96ch to match stride-8 channel count, then
    # upsample to exact stride-8 spatial dimensions and add element-wise.
    # Resizing (bilinear) guarantees exact target shape for any input resolution,
    # avoiding the off-by-1 that UpSampling2D produces for non-multiples of 16.
    _s8_h = image_height // 8
    _s8_w = image_width  // 8

    if features_s8 is not None and features_s16 is not None:
        _sem = keras.layers.Conv2D(
            96, kernel_size=1, padding="same", activation="relu",
            name="fomo_v2_sem_proj",
        )(features_s16)
        _sem_up = keras.layers.Resizing(
            _s8_h, _s8_w, interpolation="bilinear", name="fomo_v2_upsample",
        )(_sem)
        features = keras.layers.Add(name="fomo_v2_fusion")([features_s8, _sem_up])
        logger.info(
            "[FOMO v2] FPN fusion: %s (stride-8) + %s (stride-16 → stride-8, upsampled 2×)"
            " → output grid %d×%d",
            _s8_name, _s16_name, _s8_h, _s8_w,
        )
    elif features_s8 is not None:
        features = features_s8
        logger.info(
            "[FOMO v2] stride-8 only (no stride-16 layer available): tap=%s  grid=%d×%d",
            _s8_name, _s8_h, _s8_w,
        )
    else:
        features = backbone.output
        logger.warning("[FOMO v2] no stride-8 layer found; falling back to backbone output")

    # ── FOMO head — two-stage 1×1 conv heatmap ────────────────────────────────
    # 64 filters (vs 32 in v1) to handle the richer fused feature representation.
    #
    # Refinement uses a 1×1 Conv2D (channel mixing only, no spatial blending).
    # The previous 3×3 SeparableConv2D caused outranked_rate ≈ 0.9: operating at
    # grid-cell resolution its 3×3 kernel explicitly mixed each cell's features
    # with all 8 neighbours', making adjacent cells indistinguishable from the GT
    # center for the 1×1 head. A 1×1 conv refines channel weights without
    # touching spatial layout, preserving the backbone's peak at the GT center.
    #
    # Use the refined pointwise output directly rather than re-adding the raw
    # fused tensor right before the head. This keeps the final head input more
    # cell-specific and avoids letting the pre-refine fused features make nearby
    # cells look unnecessarily interchangeable.
    _refine_channels = int(features.shape[-1]) if features.shape[-1] is not None else 96
    features = keras.layers.Conv2D(
        _refine_channels, kernel_size=1, padding="same", activation="relu",
        name="fomo_v2_local_refine",
    )(features)
    features = keras.layers.Activation("linear", name="fomo_v2_refined_fusion")(features)
    head_classes = num_classes + 1
    x = keras.layers.Conv2D(
        64, kernel_size=1, padding="same",
        activation="relu", name="fomo_head_conv",
    )(features)
    x = keras.layers.Conv2D(
        head_classes, kernel_size=1, padding="same",
        activation=None, name="fomo_head",
    )(x)

    return keras.Model(inp, x, name="fomo_mobilenetv2_v2")
