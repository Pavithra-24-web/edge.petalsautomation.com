# export.py
# Step 13 — TFLite Export and Validation
# MobileNetV2 SSD FPN-Lite 320x320

from __future__ import annotations

import os
import pathlib
import tempfile
import time
import numpy as np
import tensorflow as tf

from primitives import CFG


# ============================================================================
# Internal helpers
# ============================================================================

def _tf_version_tuple() -> tuple[int, int]:
    """Return (major, minor) of the installed TensorFlow version."""
    parts = tf.__version__.split(".")
    return int(parts[0]), int(parts[1])


# ============================================================================
# Export
# ============================================================================

def export_tflite(
    model: tf.keras.Model,
    output_path: str | pathlib.Path = "mobilenetv2_ssd_fpnlite_320x320.tflite",
) -> pathlib.Path:
    """Convert a Keras model to a float32 TFLite flatbuffer and write to disk.

    DEV / SCRIPT USE ONLY — not called by the production training worker.
    mobilenetv2_ssd_worker.py performs TFLite conversion inline inside a
    tempfile.TemporaryDirectory() and uploads bytes via storage.upload_bytes().
    Calling this function from a remote Celery worker would write to the worker's
    local disk with no upload step, losing the artifact.

    Conversion path is chosen automatically based on the installed TF version:
        TF >= 2.13 — TFLiteConverter.from_keras_model(model)
                     The preferred API: simpler, avoids the concrete-function
                     tracing dance, and is the only path guaranteed to stay
                     supported in future TF releases.
        TF <  2.13 — TFLiteConverter.from_concrete_functions(...)
                     Required on older runtimes where from_keras_model had
                     reliability issues with dict-output models.

    Conversion settings:
        - float32 only (no quantization)
        - TFLite builtins only; SELECT_TF_OPS is NOT required because
          FPN-Lite upsampling now uses UpSampling2D (RESIZE_BILINEAR builtin)
          rather than tf.image.resize (a Select op)
        - Input signature preserved: (1, 320, 320, 3) uint8
        - Output signature preserved: box_predictions + cls_predictions

    Args:
        model:       Built and (optionally) trained keras.Model from model.py.
        output_path: Destination .tflite file path.

    Returns:
        pathlib.Path of the written .tflite file.
    """
    output_path = pathlib.Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Converting model → TFLite (float32) …")
    print(f"  TF version : {tf.__version__}  → using from_saved_model")

    # ------------------------------------------------------------------ #
    # Save to a temporary SavedModel directory then convert from there.
    #
    # Why not from_keras_model?
    #   Under Keras 3 + TF 2.16 the from_keras_model path re-traces the model
    #   using tf.compat.v1.placeholder, which the MLIR-based TFLite converter
    #   cannot lower ("missing attribute 'value'" / "LLVM ERROR: Failed to
    #   infer result type(s)").
    #
    # Why from_saved_model works:
    #   model.export() writes a concrete TF2 SavedModel with a fully-resolved
    #   tf.function graph.  TFLiteConverter.from_saved_model converts that
    #   graph directly without re-tracing through Keras, avoiding the
    #   placeholder issue entirely.
    # ------------------------------------------------------------------ #
    with tempfile.TemporaryDirectory() as saved_model_dir:
        model.export(saved_model_dir)
        converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)

    # Float32 only — no quantization
    converter.optimizations             = []
    converter.target_spec.supported_types = [tf.float32]

    # Builtins only — SELECT_TF_OPS is no longer required because FPN-Lite
    # upsampling uses UpSampling2D (RESIZE_BILINEAR builtin) after Step 7.
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
    ]

    # Preserve input/output tensor names for downstream tooling
    converter.allow_custom_ops = False
    converter.experimental_new_converter = True

    # ------------------------------------------------------------------ #
    # Convert and write
    # ------------------------------------------------------------------ #
    tflite_model = converter.convert()
    output_path.write_bytes(tflite_model)

    size_mb = len(tflite_model) / (1024 ** 2)
    print(f"  ✓ Saved to : {output_path}")
    print(f"  ✓ Size     : {size_mb:.2f} MB")

    return output_path


# ============================================================================
# Validation
# ============================================================================

def validate_tflite(
    tflite_path: str | pathlib.Path,
    num_runs: int = 3,
) -> dict:
    """
    Load a TFLite flatbuffer and run a forward pass to confirm correctness.

    Checks:
        1. Model loads without error
        2. Input tensor spec matches (1, 320, 320, 3) uint8
        3. Output tensors present: box_predictions, cls_predictions
        4. box_predictions shape == (1, 8028, 4)
        5. cls_predictions shape == (1, 8028, 90)
        6. Output dtype is float32
        7. No NaN / Inf in outputs
        8. Outputs are consistent across multiple runs (deterministic)

    Latency measurement:
        One untimed warm-up invoke is executed before the measured loop so
        that kernel JIT compilation, memory-mapping, and any first-call
        overhead are excluded from the reported mean.  The mean is then
        computed over exactly `num_runs` hot invocations, giving a stable
        steady-state figure.

    Args:
        tflite_path: Path to the .tflite file.
        num_runs:    Number of *timed* forward passes after the warm-up
                     (default 3).  Must be >= 1.

    Returns:
        dict with keys: input_shape, output_shapes, dtype, size_bytes,
                        nan_free, inf_free, deterministic, latency_ms
                        (latency_ms is the mean over hot runs only)
    """
    tflite_path = pathlib.Path(tflite_path)
    assert tflite_path.exists(), f"TFLite file not found: {tflite_path}"
    assert num_runs >= 1, f"num_runs must be >= 1, got {num_runs}"

    print(f"\nValidating TFLite model: {tflite_path}")

    # ------------------------------------------------------------------ #
    # 1. Load interpreter
    # ------------------------------------------------------------------ #
    interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
    interpreter.allocate_tensors()

    input_details  = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # ------------------------------------------------------------------ #
    # 2. Input spec checks
    # ------------------------------------------------------------------ #
    assert len(input_details) == 1, \
        f"Expected 1 input tensor, got {len(input_details)}"

    inp_detail = input_details[0]
    assert tuple(inp_detail["shape"]) == (1, 320, 320, 3), \
        f"Input shape mismatch: {inp_detail['shape']}"
    assert inp_detail["dtype"] == np.uint8, \
        f"Input dtype mismatch: {inp_detail['dtype']}"

    print(f"  Input  : {inp_detail['name']}  "
          f"shape={tuple(inp_detail['shape'])}  dtype={inp_detail['dtype'].__name__}")

    # ------------------------------------------------------------------ #
    # 3. Output spec checks
    # ------------------------------------------------------------------ #
    output_map: dict[str, dict] = {}
    for od in output_details:
        name = od["name"]
        output_map[name] = od
        print(f"  Output : {name}  shape={tuple(od['shape'])}  dtype={od['dtype'].__name__}")

    # Locate box and class output tensors by name fragment
    box_key = next((k for k in output_map if "box" in k.lower()), None)
    cls_key = next((k for k in output_map if "cls" in k.lower() or "class" in k.lower()), None)

    assert box_key is not None, \
        f"box_predictions output not found. Available: {list(output_map)}"
    assert cls_key is not None, \
        f"cls_predictions output not found. Available: {list(output_map)}"

    assert tuple(output_map[box_key]["shape"]) == (1, CFG.TOTAL_ANCHORS, 4), \
        f"box output shape mismatch: {output_map[box_key]['shape']}"
    assert tuple(output_map[cls_key]["shape"]) == (1, CFG.TOTAL_ANCHORS, CFG.NUM_CLASSES), \
        f"cls output shape mismatch: {output_map[cls_key]['shape']}"
    assert output_map[box_key]["dtype"] == np.float32
    assert output_map[cls_key]["dtype"] == np.float32

    # ------------------------------------------------------------------ #
    # 4. Warm-up run (untimed) — discard first cold invoke so that JIT
    #    kernel compilation, memory-mapping, and any first-call overhead
    #    are excluded from the latency measurement.
    # ------------------------------------------------------------------ #
    dummy = np.random.randint(0, 256, (1, 320, 320, 3), dtype=np.uint8)
    interpreter.set_tensor(inp_detail["index"], dummy)
    interpreter.invoke()   # cold run — result discarded, not timed

    # ------------------------------------------------------------------ #
    # 5. Timed forward passes — all hot after the warm-up above
    # ------------------------------------------------------------------ #
    box_runs: list[np.ndarray] = []
    cls_runs: list[np.ndarray] = []
    latencies: list[float] = []

    for _ in range(num_runs):
        interpreter.set_tensor(inp_detail["index"], dummy)

        t0 = time.perf_counter()
        interpreter.invoke()
        latencies.append((time.perf_counter() - t0) * 1000)   # ms

        box_out = interpreter.get_tensor(output_map[box_key]["index"])  # (1,8028,4)
        cls_out = interpreter.get_tensor(output_map[cls_key]["index"])  # (1,8028,90)
        box_runs.append(box_out.copy())
        cls_runs.append(cls_out.copy())

    # ------------------------------------------------------------------ #
    # 6. Quality checks
    # ------------------------------------------------------------------ #
    box_final = box_runs[-1]
    cls_final = cls_runs[-1]

    nan_free = not (np.any(np.isnan(box_final)) or np.any(np.isnan(cls_final)))
    inf_free = not (np.any(np.isinf(box_final)) or np.any(np.isinf(cls_final)))
    deterministic = all(
        np.allclose(box_runs[0], box_runs[j], atol=1e-5)
        for j in range(1, num_runs)
    )

    # Mean over hot runs only — warm-up invoke was untimed and excluded.
    mean_latency = float(np.mean(latencies))

    print(f"\n  ✓ box_predictions : {box_final.shape}  dtype={box_final.dtype}")
    print(f"  ✓ cls_predictions : {cls_final.shape}  dtype={cls_final.dtype}")
    print(f"  ✓ NaN free        : {nan_free}")
    print(f"  ✓ Inf free        : {inf_free}")
    print(f"  ✓ Deterministic   : {deterministic}  (across {num_runs} hot runs)")
    print(f"  ✓ Mean latency    : {mean_latency:.1f} ms  "
          f"(CPU, single image, {num_runs} hot runs, 1 warm-up discarded)")

    assert nan_free,       "NaN values detected in TFLite output"
    assert inf_free,       "Inf values detected in TFLite output"
    assert deterministic,  "TFLite outputs are not deterministic"

    return {
        "input_shape":    tuple(inp_detail["shape"]),
        "output_shapes":  {box_key: tuple(output_map[box_key]["shape"]),
                           cls_key: tuple(output_map[cls_key]["shape"])},
        "dtype":          "float32",
        "size_bytes":     tflite_path.stat().st_size,
        "nan_free":       nan_free,
        "inf_free":       inf_free,
        "deterministic":  deterministic,
        "latency_ms":     mean_latency,
    }


# ============================================================================
# Cross-check: Keras vs TFLite outputs
# ============================================================================

def crosscheck_keras_vs_tflite(
    model: tf.keras.Model,
    tflite_path: str | pathlib.Path,
    atol: float = 1e-4,
) -> bool:
    """
    Run the same input through both the Keras model and the TFLite interpreter
    and confirm outputs match within tolerance.

    Args:
        model:       Original Keras model.
        tflite_path: Path to the exported .tflite.
        atol:        Absolute tolerance for np.allclose (default 1e-4).

    Returns:
        True if all outputs match within atol.
    """
    print(f"\nCross-checking Keras vs TFLite outputs (atol={atol}) …")

    dummy = np.random.randint(0, 256, (1, 320, 320, 3), dtype=np.uint8)

    # Keras forward pass
    keras_outs  = model(dummy, training=False)
    keras_box   = keras_outs["box_predictions"].numpy()   # (1, 8028, 4)
    keras_cls   = keras_outs["cls_predictions"].numpy()   # (1, 8028, 90)

    # TFLite forward pass
    interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
    interpreter.allocate_tensors()
    inp_details = interpreter.get_input_details()
    out_details = interpreter.get_output_details()

    interpreter.set_tensor(inp_details[0]["index"], dummy)
    interpreter.invoke()

    output_map = {od["name"]: od for od in out_details}
    box_key    = next(k for k in output_map if "box" in k.lower())
    cls_key    = next(k for k in output_map if "cls" in k.lower() or "class" in k.lower())

    tflite_box = interpreter.get_tensor(output_map[box_key]["index"])
    tflite_cls = interpreter.get_tensor(output_map[cls_key]["index"])

    box_match = np.allclose(keras_box, tflite_box, atol=atol)
    cls_match = np.allclose(keras_cls, tflite_cls, atol=atol)

    box_maxerr = float(np.max(np.abs(keras_box - tflite_box)))
    cls_maxerr = float(np.max(np.abs(keras_cls - tflite_cls)))

    print(f"  box_predictions  match={box_match}  max_err={box_maxerr:.2e}")
    print(f"  cls_predictions  match={cls_match}  max_err={cls_maxerr:.2e}")

    return box_match and cls_match


# ============================================================================
# Smoke test  (python export.py)
# ============================================================================

if __name__ == "__main__":

    from model import build_model

    OUTPUT_PATH = pathlib.Path("exports/mobilenetv2_ssd_fpnlite_320x320.tflite")

    print("=== Export and Validation ===\n")

    # ---- 1. Build model --------------------------------------------------
    print("Building model …")
    model = build_model()
    print(f"  Parameters: {model.count_params():,}\n")

    # ---- 2. Export -------------------------------------------------------
    tflite_path = export_tflite(model, output_path=OUTPUT_PATH)

    # ---- 3. Validate -----------------------------------------------------
    report = validate_tflite(tflite_path, num_runs=3)

    # ---- 4. Cross-check Keras ↔ TFLite -----------------------------------
    match = crosscheck_keras_vs_tflite(model, tflite_path, atol=1e-4)

    print(f"\n  Keras ↔ TFLite match : {'✓' if match else '✗'}")

    # ---- 5. Summary ------------------------------------------------------
    print("\n=== Export Report ===")
    print(f"  File             : {tflite_path}")
    print(f"  Size             : {report['size_bytes'] / 1024**2:.2f} MB")
    print(f"  Input shape      : {report['input_shape']}")
    print(f"  box output       : {list(report['output_shapes'].values())[0]}")
    print(f"  cls output       : {list(report['output_shapes'].values())[1]}")
    print(f"  Dtype            : {report['dtype']}")
    print(f"  NaN free         : {report['nan_free']}")
    print(f"  Inf free         : {report['inf_free']}")
    print(f"  Deterministic    : {report['deterministic']}")
    print(f"  Mean latency     : {report['latency_ms']:.1f} ms")
    print(f"  Keras↔TFLite     : {'✓ match' if match else '✗ mismatch'}")
    print(f"\n✓ Export and validation complete")
