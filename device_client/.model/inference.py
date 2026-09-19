"""
PetalEdge-compatible Inference — all runtime parameters loaded from manifest.json.
Do not hardcode labels or shapes here; edit manifest.json and rebuild if needed.
"""
import json
import os as _os
import numpy as np
try:
    from tflite_runtime.interpreter import Interpreter
except ImportError:
    import tensorflow as tf
    Interpreter = tf.lite.Interpreter

# ─── Manifest (source of truth — same role as .eim embedded manifest) ────────
_MANIFEST_PATH = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "manifest.json")
if not _os.path.exists(_MANIFEST_PATH):
    raise RuntimeError(
        f"manifest.json not found at {_MANIFEST_PATH!r}; ensure the .pe package is intact"
    )
with open(_MANIFEST_PATH) as _f:
    _MANIFEST = json.load(_f)

_REQUIRED = ("label_names", "input_shape", "normalize_input", "channel_order", "model_type", "version")
_missing = [k for k in _REQUIRED if k not in _MANIFEST]
if _missing:
    raise RuntimeError(f"manifest.json is missing required fields: {_missing}; rebuild the .pe package")

LABELS        = _MANIFEST["label_names"]
INPUT_SHAPE   = _MANIFEST["input_shape"]
NORMALIZE     = _MANIFEST["normalize_input"]
CHANNEL_ORDER = _MANIFEST["channel_order"].lower()
MODEL_TYPE    = _MANIFEST["model_type"]
THRESHOLD     = float(_MANIFEST.get("threshold", 0.5))
INPUT_MEAN    = float(_MANIFEST.get("input_mean", 0.0))
INPUT_STD     = float(_MANIFEST.get("input_std", 255.0))

if CHANNEL_ORDER not in ("rgb", "bgr", "grayscale"):
    raise RuntimeError(
        f"manifest.json channel_order={CHANNEL_ORDER!r} is invalid; "
        "only 'rgb', 'bgr', or 'grayscale' are supported"
    )


def load_interpreter(model_path: str = "model.tflite"):
    interp = Interpreter(model_path=model_path)
    interp.allocate_tensors()
    return interp


def run_inference(interpreter, features: np.ndarray) -> dict:
    input_details  = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    inp = features.flatten().reshape(input_details[0]["shape"]).astype(np.float32)

    # Channel order from manifest — no heuristics
    if CHANNEL_ORDER == "bgr" and inp.ndim >= 3 and inp.shape[-1] == 3:
        inp = inp[..., ::-1].copy()

    # Normalization from manifest — preserve the declared numeric range
    if NORMALIZE:
        _denom = INPUT_STD if INPUT_STD != 0.0 else 255.0
        inp = (inp - INPUT_MEAN) / _denom

    is_fomo = (MODEL_TYPE == "detection_heatmap")

    # int8 quantization: round → clip → cast (no truncation)
    quant = input_details[0].get("quantization")
    if quant and len(quant) == 2:
        scale, zero_point = quant
        if input_details[0]["dtype"] == np.int8:
            if scale == 0:
                raise RuntimeError("int8 input tensor has scale=0; model is corrupt")
            inp = np.clip(np.round(inp / scale + zero_point), -128, 127).astype(np.int8)

    interpreter.set_tensor(input_details[0]["index"], inp)
    interpreter.invoke()

    output = interpreter.get_tensor(output_details[0]["index"])
    scores = output.flatten().astype(np.float32)

    # Dequantize output if int8
    out_scale, out_zp = output_details[0].get("quantization", (0, 0))
    if out_scale != 0:
        scores = (scores - out_zp) * out_scale

    debug = {"raw_min": float(scores.min()), "raw_max": float(scores.max()), "raw_mean": float(scores.mean())}

    if is_fomo:
        # Output shape: (1, grid_h, grid_w, num_classes+1); class 0 is background.
        # label_names contains object classes only (0-indexed → class index 1..N).
        _, gh, gw, n_all = output.shape
        cell_scores   = scores.reshape(gh, gw, n_all)
        object_scores = cell_scores[:, :, 1:]       # drop background channel
        threshold     = THRESHOLD
        detections    = []
        for row in range(gh):
            for col in range(gw):
                cls_idx = int(np.argmax(object_scores[row, col]))
                conf    = float(object_scores[row, col, cls_idx])
                if conf < threshold:
                    continue
                cx     = (col + 0.5) / gw
                cy     = (row + 0.5) / gh
                half_w = 0.5 / gw
                half_h = 0.5 / gh
                detections.append({
                    "label":      LABELS[cls_idx] if cls_idx < len(LABELS) else str(cls_idx),
                    "confidence": conf,
                    "bbox": {
                        "x1": max(0.0, cx - half_w),
                        "y1": max(0.0, cy - half_h),
                        "x2": min(1.0, cx + half_w),
                        "y2": min(1.0, cy + half_h),
                    },
                })
        return {"detections": detections, "count": len(detections), "is_fomo": True, "debug": debug}

    e = np.exp(scores - scores.max())
    scores = e / e.sum()
    best = int(np.argmax(scores))
    return {
        "label":      LABELS[best] if best < len(LABELS) else str(best),
        "confidence": float(scores[best]),
        "scores":     {LABELS[i] if i < len(LABELS) else str(i): float(scores[i]) for i in range(len(scores))},
        "is_fomo":    False,
        "debug":      debug,
    }


if __name__ == "__main__":
    interp = load_interpreter()
    dummy  = np.zeros(INPUT_SHAPE, dtype=np.float32)
    result = run_inference(interp, dummy)
    print("Prediction:", result)
