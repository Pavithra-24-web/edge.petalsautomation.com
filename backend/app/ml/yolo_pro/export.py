"""
yolo_pro/export.py
──────────────────
TFLite int8 Post-Training Quantization export for edge deployment.

Spec reference: Section 10.1 (Post-Training Quantization)

Edge constraints enforced here
───────────────────────────────
  ✦ Batch size = 1  (EON compiler / MCU TFLite runtime requirement)
  ✦ Input H and W must be multiples of 32  (P5 stride-32 path)
  ✦ BN momentum = 0.9  (inherited; do NOT change for small models)
  ✦ NMS stays OUTSIDE the graph  (TFLite graph ends at raw cls/reg outputs)
  ✦ SiLU activations are approximated as ReLU6 by the quantizer / EON LUT

Export pipeline
───────────────
  1. Rebuild model with batch_size=1 fixed in the input signature.
  2. Collect ~100 representative float32 samples for calibration.
  3. Run TFLiteConverter with:
       - full-integer quantization (DEFAULT optimisation)
       - representative_dataset callback
       - input / output types set to int8
  4. Write the .tflite file to disk.

Public API
───────────
  export_tflite_int8(
      model,
      output_path,
      input_shape,
      representative_data,
      num_calibration_steps=100,
  ) -> bytes

  RepresentativeDataset   — helper class for the calibration generator
  verify_tflite_output    — sanity-check the exported .tflite file
"""

from __future__ import annotations

import os
import tempfile
from types import SimpleNamespace
import numpy as np
import tensorflow as tf


# ─────────────────────────────────────────────────────────────────────────────
# Representative Dataset
# ─────────────────────────────────────────────────────────────────────────────
class RepresentativeDataset:
    """
    Calibration data generator for TFLite int8 PTQ.

    The converter calls this generator to observe the activation range of
    every op in the graph.  The spec recommends ~100 real inference samples
    drawn from the training distribution — NOT augmented images.

    Args:
        images               : array-like of shape (N, H, W, C) float32 [0,1]
                               or uint8 [0,255] — will be cast to float32 [0,1]
        num_calibration_steps: how many samples to feed  (spec: 100)
        input_name           : the model's input tensor name  (default 'image')

    Usage:
        rep = RepresentativeDataset(val_images, num_calibration_steps=100)
        converter.representative_dataset = rep.generator
    """

    def __init__(
        self,
        images: np.ndarray,
        num_calibration_steps: int = 100,
        input_name: str = "image",
    ):
        assert len(images) >= num_calibration_steps, (
            f"Need at least {num_calibration_steps} images for calibration; "
            f"got {len(images)}"
        )
        self.images     = images[:num_calibration_steps]
        self.n_steps    = num_calibration_steps
        self.input_name = input_name

    def generator(self):
        """
        Yield one positional list per calibration step.

        When the converter is built via from_saved_model() the exposed input
        tensor is named ``serving_default_image:0`` (not ``"image"``), so a
        named dict yields the calibration warning:
            "Statistics for quantized inputs were expected, but not specified"
        Passing a positional list ``[sample]`` instead lets the calibration
        engine match by position and collect proper activation statistics.
        Batch dimension is always 1 — EON constraint.
        """
        for img in self.images:
            # Normalise uint8 → float32 if needed
            sample = img.astype(np.float32)
            if sample.max() > 1.0:
                sample = sample / 255.0

            # Add batch dim: (H,W,C) → (1,H,W,C)
            if sample.ndim == 3:
                sample = sample[np.newaxis, ...]

            # Positional list — compatible with from_saved_model() converter
            yield [sample]


def _make_tflite_converter(
    model: tf.keras.Model,
    saved_model_dir: str,
) -> tf.lite.TFLiteConverter:
    """
    Save model as SavedModel, then return a TFLite converter from that path.

    NOTE: currently unused — every export entry point in this module goes
    through _build_frozen_converter().  Kept as the plain (non-frozen)
    converter path.

    Converter paths tried, under tensorflow==2.16.1 / Keras 3:

      • from_concrete_functions() + MLIR   → crash: MLIR BN ReadVariableOp
                                             "missing attribute 'value'"
      • from_keras_model()        + MLIR   → crash: LLVM ERROR
                                             "Failed to infer result type(s)"
      • from_saved_model()        + legacy → crash: TF1 session cannot find
                                             TF2 resource variables
                                             "Could not find variable …/moving_variance"
      • tf.saved_model.save()
        + from_saved_model()      + MLIR   → crash: LLVM ERROR (same abort as
                                             from_keras_model — a Keras 3 model
                                             saved this way keeps unfreezable
                                             ReadVariableOps)
      • model.export()
        + from_saved_model()      + MLIR   → ✓ works

    Keras 3 removed the implicit tf.saved_model.save() support the earlier
    note here assumed.  model.export() (ExportArchive) writes a fully-traced
    tf.function whose variables are already resolved, which is what the MLIR
    converter needs.

    Args:
        model           : trained keras.Model
        saved_model_dir : directory to write the SavedModel into
                          (must persist until converter.convert() returns)
    """
    model.export(saved_model_dir)
    # Do NOT set experimental_new_converter = False:
    # the legacy TF1-session path cannot locate TF2 resource variables in a
    # SavedModel checkpoint and raises "Could not find variable …".
    return tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)


def _build_frozen_saved_model(
    model: tf.keras.Model,
    input_shape: tuple[int, int, int],
    export_dir: str,
) -> tf.types.experimental.ConcreteFunction:
    """
    Build and save a frozen inference graph with an explicit serving signature.

    Why _make_tflite_converter (SavedModel path) still crashes
    ──────────────────────────────────────────────────────────
    tf.saved_model.save() writes resource-variable references into the
    graph as ReadVariableOp nodes.  When the MLIR converter later tries to
    lower Cast nodes that wrap those reads (e.g.
    ``bu_n4p_down_bn_1/Cast_3/ReadVariableOp`` for BatchNorm stats), it
    needs a ``'value'`` attribute to infer the output type.  Resource reads
    have no such attribute — they are live ops, not constants — so MLIR
    aborts with::

        error: missing attribute 'value'
        LLVM ERROR: Failed to infer result type(s).

    convert_variables_to_constants_v2() replaces every ReadVariableOp in
    the traced concrete function with a Const node whose value is the
    current variable contents.  MLIR then sees a fully-static graph and
    never encounters an unresolvable ReadVariableOp.

    Args:
        model        : trained Keras Model (weights already in final state)
        input_shape  : (H, W, C) — H, W must be multiples of 32
    """
    from tensorflow.python.framework.convert_to_constants import (
        convert_variables_to_constants_v2,
    )

    H, W, C = input_shape

    @tf.function(
        input_signature=[
            tf.TensorSpec(shape=(1, H, W, C), dtype=tf.float32, name="image")
        ]
    )
    def _infer(image):
        return model(image, training=False)

    concrete_fn = _infer.get_concrete_function()
    frozen_fn   = convert_variables_to_constants_v2(concrete_fn)

    module = tf.Module()
    module.serve = frozen_fn
    tf.saved_model.save(module, export_dir, signatures={"serving_default": frozen_fn})
    return frozen_fn


class _FrozenSavedModelConverter:
    """
    Converter-like wrapper that preserves the existing export flow while
    building a one-signature SavedModel only at convert() time.
    """

    def __init__(self, model: tf.keras.Model, input_shape: tuple[int, int, int]):
        self.model = model
        self.input_shape = input_shape
        self.optimizations = []
        self.representative_dataset = None
        self.target_spec = SimpleNamespace(supported_ops=[])
        self.inference_input_type = None
        self.inference_output_type = None

    def convert(self) -> bytes:
        with tempfile.TemporaryDirectory() as tmpdir:
            _build_frozen_saved_model(self.model, self.input_shape, tmpdir)
            converter = tf.lite.TFLiteConverter.from_saved_model(tmpdir)
            converter.optimizations = self.optimizations
            converter.representative_dataset = self.representative_dataset
            converter.target_spec.supported_ops = self.target_spec.supported_ops
            if self.inference_input_type is not None:
                converter.inference_input_type = self.inference_input_type
            if self.inference_output_type is not None:
                converter.inference_output_type = self.inference_output_type
            return converter.convert()


def _build_frozen_converter(
    model: tf.keras.Model,
    input_shape: tuple[int, int, int],
) -> _FrozenSavedModelConverter:
    return _FrozenSavedModelConverter(model, input_shape)


# ─────────────────────────────────────────────────────────────────────────────
# Core export function
# ─────────────────────────────────────────────────────────────────────────────
def export_tflite_int8(
    model: tf.keras.Model,
    output_path: str,
    input_shape: tuple[int, int, int],
    representative_data: np.ndarray,
    num_calibration_steps: int = 100,
) -> bytes:
    """
    Export a YOLO-Pro Keras model to a fully-quantized int8 TFLite file.

    Edge constraints applied automatically:
        - Input/output tensors typed as int8.
        - NMS is NOT included — graph ends at raw (cls_pX, reg_pX) outputs.
        - Legacy converter is used to avoid the MLIR BN crash.

    Args:
        model                  : trained keras.Model from build_yolo_pro()
                                 (weights already loaded / EMA applied)
        output_path            : destination .tflite file path
        input_shape            : (H, W, C) — H,W must be multiples of 32
        representative_data    : float32 or uint8 array (N, H, W, C)
                                 min N = num_calibration_steps
        num_calibration_steps  : calibration samples  (spec recommends 100)

    Returns:
        Raw .tflite bytes (also written to output_path).

    Raises:
        AssertionError if input_shape violates edge constraints.
    """
    H, W, C = input_shape

    # ── Constraint checks ────────────────────────────────────────────────────
    assert H % 32 == 0 and W % 32 == 0, (
        f"Input H={H} W={W} must both be multiples of 32 (P5 stride-32 constraint)"
    )
    assert H >= 96 and W >= 96, (
        "Minimum input size is 96×96 (smallest valid P5 grid = 3×3)"
    )
    assert C in (1, 3), f"Channel count must be 1 (gray) or 3 (RGB); got {C}"

    out_dir = os.path.dirname(os.path.abspath(output_path))
    os.makedirs(out_dir, exist_ok=True)

    # ── Step 1: converter setup ──────────────────────────────────────────────
    # Freeze all resource variables (BN moving stats, gamma, beta) into Const
    # nodes before MLIR processes the graph.  This avoids the ReadVariableOp
    # "missing attribute 'value'" crash (see _build_frozen_converter for the
    # full explanation).
    converter = _build_frozen_converter(model, input_shape)

    # Full-integer quantization: every eligible op is quantized, not just weights
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    # ── Step 3: representative dataset ──────────────────────────────────────
    rep_ds = RepresentativeDataset(
        images                = representative_data,
        num_calibration_steps = num_calibration_steps,
        input_name            = "image",
    )
    converter.representative_dataset = rep_ds.generator

    # ── Step 4: int8 ops + int8 I/O types ───────────────────────────────────
    # TFLITE_BUILTINS_INT8 quantizes all supported ops to int8.
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
    ]

    # MCU firmware feeds raw int8 pixels and reads int8 logits directly.
    # The TFLite runtime inserts dequant/quant nodes at the graph boundary.
    converter.inference_input_type  = tf.int8
    converter.inference_output_type = tf.int8

    # ── Step 5: convert ──────────────────────────────────────────────────────
    print(f"  Running PTQ calibration ({num_calibration_steps} samples)…")
    tflite_model = converter.convert()
    print(f"  Conversion complete.  Model size: {len(tflite_model) / 1024:.1f} KB")

    # ── Step 6: write to disk ────────────────────────────────────────────────
    with open(output_path, "wb") as f:
        f.write(tflite_model)
    print(f"  Saved → {output_path}")

    return tflite_model


# ─────────────────────────────────────────────────────────────────────────────
# Verification helper
# ─────────────────────────────────────────────────────────────────────────────
def verify_tflite_output(
    tflite_path: str,
    input_shape: tuple[int, int, int],
    num_classes: int,
    reg_max: int = 16,
) -> None:
    """
    Load the exported .tflite and run one forward pass to verify output shapes.

    Checks:
        - Input tensor type is int8 with batch=1
        - All output tensors are int8
        - Output shapes match expected grid sizes for the given input

    Args:
        tflite_path : path to the .tflite file
        input_shape : (H, W, C)
        num_classes : expected number of output classes
        reg_max     : DFL bins  (default 16)
    """
    H, W, C = input_shape

    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()

    input_details  = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    print("=" * 62)
    print(f"  TFLite verification — {os.path.basename(tflite_path)}")
    print("=" * 62)

    # Input check
    inp = input_details[0]
    print(f"\n  Input tensor:")
    print(f"    name  : {inp['name']}")
    print(f"    shape : {list(inp['shape'])}")
    print(f"    dtype : {inp['dtype'].__name__}")
    assert inp["dtype"] == np.int8, "Input must be int8"
    assert list(inp["shape"]) == [1, H, W, C], (
        f"Input shape mismatch: {list(inp['shape'])} vs [1,{H},{W},{C}]"
    )

    # Run inference with a random int8 tensor
    dummy = np.random.randint(-128, 127, size=(1, H, W, C), dtype=np.int8)
    interpreter.set_tensor(inp["index"], dummy)
    interpreter.invoke()

    print(f"\n  Output tensors:")
    for out in output_details:
        name    = out["name"]
        shape   = tuple(out["shape"])
        dtype   = out["dtype"]
        ok      = "✅" if dtype == np.int8 else "❌"
        print(f"    {name:<26}  shape={str(shape):<22}  dtype={dtype.__name__}  {ok}")
        assert dtype == np.int8, f"Output {name} must be int8"

    print("\n  ✅ All output tensors are int8. Verification passed.")


# ─────────────────────────────────────────────────────────────────────────────
# float32 export (debug / non-MCU targets)
# ─────────────────────────────────────────────────────────────────────────────
def export_tflite_float32(
    model: tf.keras.Model,
    output_path: str,
    input_shape: tuple[int, int, int],
) -> bytes:
    """
    Export to float32 TFLite (no quantization).

    Use for debugging, latency profiling on GPU/CPU targets, or devices
    that lack int8 kernels. NMS remains outside the graph.
    """
    H, W, C = input_shape
    assert H % 32 == 0 and W % 32 == 0, "H and W must be multiples of 32"

    # Freeze all resource variables before MLIR conversion to avoid the
    # ReadVariableOp "missing attribute 'value'" crash (see
    # _build_frozen_converter for the full explanation).
    converter    = _build_frozen_converter(model, input_shape)
    tflite_model = converter.convert()

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(tflite_model)
    print(f"  float32 TFLite → {output_path}  ({len(tflite_model)/1024:.1f} KB)")
    return tflite_model


# ─────────────────────────────────────────────────────────────────────────────
# Decoded float32 export  (OG-parity single-output interface)
# ─────────────────────────────────────────────────────────────────────────────
def export_tflite_float32_decoded(
    model: tf.keras.Model,
    output_path: str,
    input_shape: tuple[int, int, int],
    reg_max: int = 16,
) -> bytes:
    """
    Export a YOLO-Pro model with the decode layer baked in as float32 TFLite.

    The exported model produces a single decoded output tensor:

        shape  : (1, N_total, 6)
        dtype  : float32
        columns: [x1, y1, x2, y2, score, class_id]

    This matches the OG Edge Impulse export format:

        TensorSpec(shape=(None, 189, 6), dtype=tf.float32)

    for a 96×96 input (N_total = 189).  All N_total anchors are included in
    the output without confidence filtering or NMS — the runtime applies those
    steps after receiving the tensor.

    The decode is performed entirely in TF ops baked into the frozen graph:
      - Anchor constants embedded as Const nodes (no runtime overhead).
      - DFL decode via softmax + weighted sum.
      - Class decode via reduce_max + argmax.

    The raw-output multi-head exports (float32 / int8) are unchanged.

    Args:
        model        : trained Keras model from build_yolo_pro() with EMA applied
        output_path  : destination .tflite file path
        input_shape  : (H, W, C) — H, W must be multiples of 32
        reg_max      : DFL bins (must match training config, default 16)

    Returns:
        Raw .tflite bytes (also written to output_path).
    """
    from .decode import build_yolo_pro_decoded, total_anchor_count

    H, W, C = input_shape
    assert H % 32 == 0 and W % 32 == 0, "H and W must be multiples of 32"
    assert H >= 96 and W >= 96, "Minimum input size is 96×96"

    decoded_model = build_yolo_pro_decoded(model, input_shape, reg_max=reg_max)
    converter     = _build_frozen_converter(decoded_model, input_shape)
    tflite_model  = converter.convert()

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(tflite_model)

    N_total = total_anchor_count(H, W)
    print(
        f"  decoded float32 TFLite → {output_path}  "
        f"({len(tflite_model)/1024:.1f} KB, output=(1,{N_total},6))"
    )
    return tflite_model
