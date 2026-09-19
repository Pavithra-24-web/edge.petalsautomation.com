#!/usr/bin/env python3
"""
.pxe runner — Edge Impulse .eim stdio-JSONL protocol.
Applies embedded DSP pipeline before TFLite inference.
Launched as an OS process; communicate via stdin/stdout JSON lines.
"""
import json, sys, os, time, tempfile, hashlib
import numpy as np

# ── Runtime-overridable parameters ───────────────────────────────────────────
# Updated by the `set_threshold` and `set_parameter` protocol messages. When
# None, code paths fall back to the manifest-baked default (no override).
# Changes take effect on the very next classify call — no restart needed.
_runtime_threshold = None

def _load(name):
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), name), "rb") as f:
        return f.read()

manifest   = json.loads(_load("manifest.json"))
dsp_config = json.loads(_load("dsp_config.json"))
_manifest_labels = manifest.get("label_names")
if _manifest_labels:
    labels = list(_manifest_labels)
else:
    labels = _load("labels.txt").decode().strip().splitlines()
_model_bytes = _load("model.tflite")
_pp_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "postprocess_config.json")
if os.path.exists(_pp_path):
    with open(_pp_path, "r", encoding="utf-8") as _f:
        postprocess_config = json.load(_f)
else:
    postprocess_config = {}

_dsp_blocks = dsp_config.get("dsp_blocks", [])
_input_shape = manifest.get("input_shape", [])
# Image models report freq=0.0 (no DSP rate); audio/time-series report the
# real packager-emitted frequency_hz. A missing frequency_hz on a non-image
# model is a packager bug.
_is_image_model = len(_input_shape) >= 3
_freq_hz = 0.0 if _is_image_model else float(dsp_config.get("frequency_hz", 0.0))
_dsp_ms     = [0]

try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    import tensorflow.lite as tflite

def _validate_postprocess_config(cfg):
    if not isinstance(cfg, dict):
        raise ValueError("postprocess_config.json must contain a JSON object")
    if not isinstance(cfg.get("enabled"), bool):
        raise ValueError("postprocess_config.enabled must be a bool")
    th = cfg.get("threshold")
    if not isinstance(th, (int, float)) or not (0.0 <= float(th) <= 1.0):
        raise ValueError(f"postprocess_config.threshold must be float in [0,1], got {th!r}")
    cf = cfg.get("class_filter")
    if not isinstance(cf, list) or not all(isinstance(x, str) for x in cf):
        raise ValueError("postprocess_config.class_filter must be a list of strings")
    if not isinstance(cfg.get("tracking_enabled"), bool):
        raise ValueError("postprocess_config.tracking_enabled must be a bool")
    kg = cfg.get("keep_grace")
    if not isinstance(kg, int) or kg < 0:
        raise ValueError(f"postprocess_config.keep_grace must be int >= 0, got {kg!r}")
    mo = cfg.get("max_observations")
    if not isinstance(mo, int) or mo < 1:
        raise ValueError(f"postprocess_config.max_observations must be int >= 1, got {mo!r}")

if postprocess_config:
    _validate_postprocess_config(postprocess_config)

def _postprocess_enabled():
    return bool(postprocess_config.get("enabled", True)) if postprocess_config else False

def _packaged_threshold(default):
    if not _postprocess_enabled():
        return float(default)
    return float(postprocess_config.get("threshold", default))

def _packaged_class_filter():
    if not _postprocess_enabled():
        return []
    return list(postprocess_config.get("class_filter") or [])

def _verify_model_hash():
    declared = manifest.get("model_hash")
    if not declared:
        return
    if not isinstance(declared, str) or not declared.startswith("sha256:"):
        raise RuntimeError(f"unsupported model_hash format: {declared!r}")
    actual = hashlib.sha256(_model_bytes).hexdigest()
    expected = declared.split(":", 1)[1]
    if actual != expected:
        raise RuntimeError(
            f"model_hash mismatch: expected sha256:{expected}, got sha256:{actual}"
        )

_verify_model_hash()

_tmp = tempfile.NamedTemporaryFile(suffix=".tflite", delete=False)
_tmp.write(_model_bytes)
_tmp.flush(); _tmp.close()
import atexit; atexit.register(os.unlink, _tmp.name)

interpreter = tflite.Interpreter(model_path=_tmp.name)
interpreter.allocate_tensors()
_inp = interpreter.get_input_details()[0]
_out = interpreter.get_output_details()[0]


# ── Manifest-driven dispatch (no runtime heuristics) ─────────────────────────
#
# coordinate_space describes the model's *raw* box output: "pixel" means the
# raw values are in absolute pixel space and must be divided by (input_w,
# input_h) before emitting normalized bboxes on the wire; "normalized" means
# the values are already in [0, 1] and pass through unchanged. The wire
# convention for `detections[].bbox` is always normalized.
#
# output_tensor_format describes how the runner must parse outputs:
#   - "decoded":        single (1, N, 6) tensor [x1,y1,x2,y2,score,class_id]
#   - "raw_multi_head": six tensors (cls_p3/reg_p3/cls_p4/reg_p4/cls_p5/reg_p5)
#   - "fomo_heatmap":   FOMO grid heatmap → bounding_boxes in pixel space
#   - "softmax":        classification probabilities
#
# Both fields are mandatory: a missing or unrecognized value is a packaging
# bug. Fail loudly at startup, not at first classify.

_ALLOWED_COORDINATE_SPACES   = ("normalized", "pixel")
_ALLOWED_OUTPUT_TENSOR_FORMATS = ("decoded", "raw_multi_head", "fomo_heatmap", "softmax")

if "coordinate_space" not in manifest:
    raise RuntimeError(
        ".pxe manifest is missing required field 'coordinate_space'; "
        "rebuild the package with a packager that emits the EI-canonical schema"
    )
_COORDINATE_SPACE = manifest["coordinate_space"]
if _COORDINATE_SPACE not in _ALLOWED_COORDINATE_SPACES:
    raise RuntimeError(
        f".pxe manifest.coordinate_space={_COORDINATE_SPACE!r} is not one of "
        f"{_ALLOWED_COORDINATE_SPACES}; this is a packaging bug"
    )

if "output_tensor_format" not in manifest:
    raise RuntimeError(
        ".pxe manifest is missing required field 'output_tensor_format'; "
        "rebuild the package with a packager that emits the EI-canonical schema"
    )
_OUTPUT_TENSOR_FORMAT = manifest["output_tensor_format"]
if _OUTPUT_TENSOR_FORMAT not in _ALLOWED_OUTPUT_TENSOR_FORMATS:
    raise RuntimeError(
        f".pxe manifest.output_tensor_format={_OUTPUT_TENSOR_FORMAT!r} is not one of "
        f"{_ALLOWED_OUTPUT_TENSOR_FORMATS}; this is a packaging bug"
    )

# Verify declared format matches the actual tensor count/shape. A mismatch
# means the packager declared the wrong format — we refuse to guess.
_output_details_at_boot = interpreter.get_output_details()
_n_outputs = len(_output_details_at_boot)
if _OUTPUT_TENSOR_FORMAT == "decoded":
    if _n_outputs != 1:
        raise RuntimeError(
            f".pxe declares output_tensor_format='decoded' but model has "
            f"{_n_outputs} output tensors (expected 1); manifest does not match the model"
        )
    _decoded_shape = list(_output_details_at_boot[0]["shape"])
    if len(_decoded_shape) != 3:
        raise RuntimeError(
            f".pxe declares output_tensor_format='decoded' but the sole output "
            f"tensor has shape {_decoded_shape!r} (expected 3-D [1, N, 6]); "
            f"manifest does not match the model"
        )
elif _OUTPUT_TENSOR_FORMAT == "raw_multi_head":
    if _n_outputs != 6:
        raise RuntimeError(
            f".pxe declares output_tensor_format='raw_multi_head' but model has "
            f"{_n_outputs} output tensors (expected 6 — cls/reg × p3/p4/p5); "
            f"manifest does not match the model"
        )


# ── DSP pipeline (numpy-only, matches DSPProcessor in the backend) ──────────

def _frame(data, frame_len, hop):
    if len(data) < frame_len:
        data = np.pad(data, (0, frame_len - len(data)))
    n = 1 + (len(data) - frame_len) // hop
    idx = np.arange(frame_len)[None, :] + hop * np.arange(n)[:, None]
    return data[idx]

def _mel_filterbank(n_filters, fft_len, sr, low_hz, high_hz):
    n_fft = fft_len // 2 + 1
    mel  = lambda hz: 2595.0 * np.log10(1.0 + hz / 700.0)
    imel = lambda m:   700.0 * (10.0 ** (m / 2595.0) - 1.0)
    pts  = np.array([imel(m) for m in np.linspace(mel(low_hz), mel(high_hz), n_filters + 2)])
    bins = np.floor((fft_len + 1) * pts / sr).astype(int)
    fb   = np.zeros((n_filters, n_fft))
    for m in range(1, n_filters + 1):
        f0, f1, f2 = bins[m-1], bins[m], bins[m+1]
        for k in range(f0, f1):
            if f1 > f0: fb[m-1, k] = (k - f0) / (f1 - f0)
        for k in range(f1, f2):
            if f2 > f1: fb[m-1, k] = (f2 - k) / (f2 - f1)
    return fb

def _dct2(x):
    N = x.shape[-1]
    v = np.concatenate([x[:, ::2], x[:, 1::2][:, ::-1]], axis=-1)
    V = np.fft.fft(v, axis=-1)
    return np.real(V * 2 * np.exp(-1j * np.pi * np.arange(N) / (2 * N)))

def _apply_manifest_preprocessing(arr):
    channel_order = str(manifest.get("channel_order", "rgb")).lower()
    normalize = bool(manifest.get("normalize_input", False))
    input_mean = float(manifest.get("input_mean", 0.0))
    input_std = float(manifest.get("input_std", 255.0))

    if channel_order not in ("rgb", "bgr", "grayscale"):
        raise ValueError(f"unsupported channel_order={channel_order!r}")
    if channel_order == "bgr" and arr.ndim >= 3 and arr.shape[-1] == 3:
        arr = arr[..., ::-1].copy()

    if normalize:
        denom = input_std if input_std != 0.0 else 255.0
        arr = (arr - input_mean) / denom

    return arr

def _reshape_flat_image(arr, width, height, channels):
    arr = np.array(arr, dtype=np.float32)
    # ── Packed-pixel safety net ──────────────────────────────────────────
    # The EI JS image-classifier historically encodes each pixel as a single
    # 24-bit packed integer (0xRRGGBB), producing W*H values. This runner
    # expects flat channel values [R,G,B,...] producing W*H*C values.
    # When the incoming classify array length equals W*H and the model
    # expects 3 channels, treat each value as a packed 24-bit int and
    # unpack into three bytes: bits 16-23 → R, 8-15 → G, 0-7 → B.
    # This handles old JS senders and third-party clients gracefully.
    packed_len = width * height
    if arr.size == packed_len and channels == 3 and packed_len != packed_len * channels:
        print("WARNING: received packed-pixel format, unpacking to flat channels",
              file=sys.stderr, flush=True)
        ints = arr.astype(np.int32)
        r = ((ints >> 16) & 0xFF).astype(np.float32)
        g = ((ints >> 8) & 0xFF).astype(np.float32)
        b = (ints & 0xFF).astype(np.float32)
        arr = np.stack([r, g, b], axis=-1).flatten()
    # ── End packed-pixel safety net ──────────────────────────────────────
    expected = width * height * channels
    if arr.size == expected:
        if channels == 1:
            return arr.reshape((height, width))
        return arr.reshape((height, width, channels))

    pixels = arr.size // max(channels, 1) if channels > 0 else arr.size
    if channels > 0 and arr.size % channels == 0:
        side = int(round(pixels ** 0.5))
        if side * side == pixels:
            if channels == 1:
                return arr.reshape((side, side))
            return arr.reshape((side, side, channels))

        aspect = width / max(height, 1)
        src_h = max(1, int(round((pixels / max(aspect, 1e-8)) ** 0.5)))
        src_w = max(1, int(round(src_h * aspect)))
        if src_w * src_h == pixels:
            if channels == 1:
                return arr.reshape((src_h, src_w))
            return arr.reshape((src_h, src_w, channels))

    padded = np.zeros(expected, dtype=np.float32)
    copy_len = min(arr.size, expected)
    padded[:copy_len] = arr[:copy_len]
    if channels == 1:
        return padded.reshape((height, width))
    return padded.reshape((height, width, channels))

def _resize_pil_image(img, image_width, image_height, resize_mode):
    from PIL import Image

    orig_w, orig_h = img.size

    if resize_mode == "Fit shortest axis":
        scale = max(image_width / orig_w, image_height / orig_h)
        new_w = max(image_width, round(orig_w * scale))
        new_h = max(image_height, round(orig_h * scale))
        img = img.resize((new_w, new_h), Image.LANCZOS)
        left = (new_w - image_width) // 2
        top = (new_h - image_height) // 2
        return img.crop((left, top, left + image_width, top + image_height))

    if resize_mode == "Fit longest axis":
        scale = min(image_width / orig_w, image_height / orig_h)
        new_w = max(1, round(orig_w * scale))
        new_h = max(1, round(orig_h * scale))
        img = img.resize((new_w, new_h), Image.LANCZOS)
        canvas = Image.new(img.mode, (image_width, image_height), 0)
        canvas.paste(img, ((image_width - new_w) // 2, (image_height - new_h) // 2))
        return canvas

    return img.resize((image_width, image_height), Image.LANCZOS)

def _dsp_image(arr, params):
    from PIL import Image

    image_width = int(params.get("image_width") or 96)
    image_height = int(params.get("image_height") or 96)
    grayscale = bool(params.get("grayscale", False))
    resize_mode = params.get("resize_mode", "Fit shortest axis")
    channels = 1 if grayscale else 3

    arr = np.array(arr, dtype=np.float32)
    if arr.ndim == 1:
        arr = _reshape_flat_image(arr, image_width, image_height, channels)
    elif arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[..., 0]

    if arr.ndim == 2:
        img = Image.fromarray(np.clip(arr, 0.0, 255.0).astype(np.uint8), mode="L")
    elif arr.ndim == 3:
        if arr.shape[-1] >= 3:
            img = Image.fromarray(np.clip(arr[..., :3], 0.0, 255.0).astype(np.uint8), mode="RGB")
        elif arr.shape[-1] == 1:
            img = Image.fromarray(np.clip(arr[..., 0], 0.0, 255.0).astype(np.uint8), mode="L")
        else:
            raise ValueError(f"unsupported image array shape: {arr.shape!r}")
    else:
        raise ValueError(f"unsupported image array shape: {arr.shape!r}")

    img = img.convert("L" if grayscale else "RGB")
    data = np.array(_resize_pil_image(img, image_width, image_height, resize_mode), dtype=np.float32)
    return _apply_manifest_preprocessing(data)

def _dsp_raw(arr, params):
    arr = arr * float(params.get("scale_axes", 1.0))
    if params.get("normalize", True):
        rng = arr.max() - arr.min()
        if rng > 1e-8:
            arr = (arr - arr.min()) / rng
    return arr

def _dsp_spectral(arr, params, freq_hz):
    fft_len = int(params.get("fft_length", 256))
    overlap = float(params.get("overlap", 0.5))
    floor   = float(params.get("noise_floor_db", -52.0))
    arr     = arr * float(params.get("scale_axes", 1.0))
    n_axes  = int(dsp_config.get("n_axes", 1))
    hop     = max(1, int(fft_len * (1 - overlap)))

    if n_axes > 1 and len(arr) % n_axes == 0:
        channels = arr.reshape(-1, n_axes).T
    else:
        channels = arr[np.newaxis, :]

    out = []
    for ch in channels:
        frames = _frame(ch, fft_len, hop)
        win    = np.hanning(fft_len)
        spec   = np.abs(np.fft.rfft(frames * win, n=fft_len))
        db     = np.maximum(20 * np.log10(spec + 1e-10), floor)
        out.append(np.mean(db, axis=0))
    return np.concatenate(out).astype(np.float32)

def _dsp_mfcc(arr, params, freq_hz):
    n_c   = int(params.get("num_coefficients", 13))
    fl    = int(params.get("frame_length", 256))
    fs    = int(params.get("frame_stride", 128))
    nf    = int(params.get("num_filters", 40))
    flen  = int(params.get("fft_length", 256))
    floor = float(params.get("noise_floor_db", -52.0))
    lf    = float(params.get("low_frequency", 300.0))
    hf    = min(float(params.get("high_frequency", 8000.0)), freq_hz / 2.0)
    n_axes = int(dsp_config.get("n_axes", 1))

    if n_axes > 1 and len(arr) % n_axes == 0:
        channels = arr.reshape(-1, n_axes).T
    else:
        channels = arr[np.newaxis, :]

    out = []
    for ch in channels:
        frames = _frame(ch, fl, fs)
        win    = np.hanning(fl)
        pw     = np.abs(np.fft.rfft(frames * win, n=flen)) ** 2
        fb     = _mel_filterbank(nf, flen, freq_hz, lf, hf)
        log_m  = np.maximum(np.log(np.dot(pw, fb.T) + 1e-10), floor)
        out.append(_dct2(log_m)[:, :n_c].flatten())
    return np.concatenate(out).astype(np.float32)

def _dsp_spectrogram(arr, params, freq_hz):
    fl    = int(params.get("frame_length", 256))
    fs    = int(params.get("frame_stride", 128))
    flen  = int(params.get("fft_length", 256))
    nm    = int(params.get("num_mel_filters", 32))
    floor = float(params.get("noise_floor_db", -52.0))
    n_axes = int(dsp_config.get("n_axes", 1))

    if n_axes > 1 and len(arr) % n_axes == 0:
        channels = arr.reshape(-1, n_axes).T
    else:
        channels = arr[np.newaxis, :]

    out = []
    for ch in channels:
        frames = _frame(ch, fl, fs)
        win    = np.hanning(fl)
        pw     = np.abs(np.fft.rfft(frames * win, n=flen)) ** 2
        fb     = _mel_filterbank(nm, flen, freq_hz, 0, freq_hz / 2.0)
        log_m  = np.maximum(10 * np.log10(np.dot(pw, fb.T) + 1e-10), floor)
        out.append(log_m.flatten())
    return np.concatenate(out).astype(np.float32)

def _dsp_flatten(arr, params):
    arr    = arr * float(params.get("scale_axes", 1.0))
    feats  = params.get("features", ["mean", "std", "rms"])
    n_axes = int(dsp_config.get("n_axes", 1))

    if n_axes > 1 and len(arr) % n_axes == 0:
        chs = arr.reshape(-1, n_axes).T.tolist()
    else:
        chs = [arr.flatten().tolist()]

    out = []
    for ch in chs:
        ch = np.array(ch, dtype=np.float32)
        for f in feats:
            if f == "mean":  out.append(float(np.mean(ch)))
            elif f == "std": out.append(float(np.std(ch)))
            elif f == "rms": out.append(float(np.sqrt(np.mean(ch**2))))
            elif f == "max": out.append(float(np.max(ch)))
            elif f == "min": out.append(float(np.min(ch)))
    return np.array(out, dtype=np.float32)

_DSP_FN = {
    "image":             _dsp_image,
    "raw":               _dsp_raw,
    "spectral_analysis": _dsp_spectral,
    "mfcc":              _dsp_mfcc,
    "spectrogram":       _dsp_spectrogram,
    "flatten":           _dsp_flatten,
}

def _apply_dsp(raw):
    arr = np.array(raw, dtype=np.float32)
    for block in _dsp_blocks:
        fn = _DSP_FN.get(block.get("type", "raw"))
        if fn is None:
            continue
        params = block.get("params", {})
        needs_freq = block.get("type") in ("spectral_analysis", "mfcc", "spectrogram")
        arr = fn(arr, params, _freq_hz) if needs_freq else fn(arr, params)
    return arr.flatten()


# ── Inference ────────────────────────────────────────────────────────────────

def _classify(raw_features):
    if _dsp_blocks:
        t_dsp = time.monotonic_ns()
        arr = _apply_dsp(raw_features)
        _dsp_ms[0] = (time.monotonic_ns() - t_dsp) // 1_000_000
    else:
        arr = np.array(raw_features, dtype=np.float32)
        arr = _apply_manifest_preprocessing(arr)
        _dsp_ms[0] = 0

    scale, zp = _inp["quantization"]
    if _inp["dtype"] == np.int8:
        if scale == 0:
            raise ValueError("int8 input tensor has scale=0; model metadata is corrupt")
        q = np.clip(np.round(arr / scale + zp), -128, 127).astype(np.int8)
    else:
        q = arr.astype(_inp["dtype"])
    interpreter.set_tensor(_inp["index"], q.reshape(_inp["shape"]))
    t0 = time.monotonic_ns()
    interpreter.invoke()
    ms = (time.monotonic_ns() - t0) // 1_000_000

    raw = interpreter.get_tensor(_out["index"])[0]
    o_scale, o_zp = _out["quantization"]
    if o_scale != 0:
        raw = (raw.astype(np.float32) - o_zp) * o_scale

    # Dispatch directly on the manifest-declared output_tensor_format. No
    # tensor-count or tensor-shape sniffing; the packager already verified the
    # match at build time and the runner verified it at startup.
    if _OUTPUT_TENSOR_FORMAT == "decoded":
        return _postprocess_yolo_decoded(ms)
    if _OUTPUT_TENSOR_FORMAT == "raw_multi_head":
        return _postprocess_yolo_raw_multi_head(ms)
    if _OUTPUT_TENSOR_FORMAT == "fomo_heatmap":
        return _postprocess_fomo(raw, ms)
    if _OUTPUT_TENSOR_FORMAT == "softmax":
        return _postprocess_classification(raw, ms)
    # Unreachable — startup validation rejected anything else.
    raise RuntimeError(
        f"unhandled output_tensor_format={_OUTPUT_TENSOR_FORMAT!r}"
    )


def _postprocess_classification(raw, ms):
    flat = raw.flatten().astype(np.float32)
    if abs(float(flat.sum()) - 1.0) < 0.01:
        scores = flat
    else:
        e = np.exp(flat - flat.max())
        scores = e / e.sum()
    n = min(len(scores), len(labels))
    return (
        {"classification": {labels[i]: float(scores[i]) for i in range(n)}},
        ms,
    )


def _yolo_inputs():
    """Common setup for the YOLO postprocess branches.

    YOLO: manifest threshold is the model-calibrated value from training
    (default 0.25). postprocess_config.threshold is a FOMO/classification
    post-filter and must NOT override YOLO's confidence gate. Read directly
    from manifest.

    Runtime override: `_runtime_threshold`, when not None, takes precedence
    over the manifest default. It is mutated by the `set_threshold` /
    `set_parameter` protocol messages and picked up on the next classify.
    """
    if _runtime_threshold is not None:
        threshold = float(_runtime_threshold)
    else:
        threshold = float(manifest.get("threshold", 0.25))
    iou_threshold = float(manifest.get("iou_threshold", 0.45))
    class_filter = set(_packaged_class_filter())
    input_shape = manifest.get("input_shape", [96, 96, 3])
    input_h = int(input_shape[0]) if len(input_shape) >= 1 else 96
    input_w = int(input_shape[1]) if len(input_shape) >= 2 else 96
    return threshold, iou_threshold, class_filter, input_h, input_w


def _normalize_bbox_for_wire(x1, y1, x2, y2, input_w, input_h):
    """Convert a raw bbox (in the manifest-declared coordinate_space) to the
    wire convention, which is always normalized [0, 1].

    coordinate_space == "pixel"      → raw values are absolute pixels; divide
                                       by input dims to produce normalized.
    coordinate_space == "normalized" → raw values are already in [0, 1]; pass
                                       through verbatim.
    """
    if _COORDINATE_SPACE == "pixel":
        return (x1 / input_w, y1 / input_h, x2 / input_w, y2 / input_h)
    return (x1, y1, x2, y2)


def _postprocess_yolo_decoded(ms):
    """Decoded format: single (1, N, 6) tensor [x1,y1,x2,y2,score,class_id]."""
    threshold, iou_threshold, class_filter, input_h, input_w = _yolo_inputs()
    output_details = interpreter.get_output_details()
    tensor = interpreter.get_tensor(output_details[0]["index"]).astype(np.float32)
    rows = tensor[0]  # (N_total, 6)

    debug = {
        "format": "decoded",
        "n_anchors": int(rows.shape[0]),
        "raw_max_score": float(rows[:, 4].max()) if len(rows) > 0 else 0.0,
    }
    detections = []
    for row in rows:
        score = float(row[4])
        if score < threshold:
            continue
        raw_x1, raw_y1 = float(row[0]), float(row[1])
        raw_x2, raw_y2 = float(row[2]), float(row[3])
        # Tensor values are in the manifest-declared coordinate_space; convert
        # to the wire convention (always normalized).
        x1, y1, x2, y2 = _normalize_bbox_for_wire(
            raw_x1, raw_y1, raw_x2, raw_y2, input_w, input_h,
        )
        if x2 <= x1 or y2 <= y1:
            continue
        cls_idx = int(round(float(row[5])))
        label = labels[cls_idx] if 0 <= cls_idx < len(labels) else str(cls_idx)
        detections.append({
            "label": label,
            "confidence": score,
            "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
            "_box": [x1, y1, x2, y2],
            "_score": score,
        })

    return _finalize_yolo(detections, debug, iou_threshold, class_filter, ms)


def _postprocess_yolo_raw_multi_head(ms):
    """Raw multi-head format: six outputs (cls_p3/reg_p3/cls_p4/reg_p4/cls_p5/reg_p5).

    cls outputs are already sigmoid probabilities from the detection head —
    do NOT apply sigmoid again. The decoded bbox is in pixel-space anchor
    geometry; final emission goes through _emit_yolo_bbox so the wire bbox
    matches manifest.coordinate_space.
    """
    threshold, iou_threshold, class_filter, input_h, input_w = _yolo_inputs()
    output_details = interpreter.get_output_details()

    outputs = {}
    for i, od in enumerate(output_details):
        tensor = interpreter.get_tensor(od["index"]).astype(np.float32)
        o_scale, o_zp = od.get("quantization", (0.0, 0))
        if o_scale not in (0, 0.0):
            tensor = (tensor - o_zp) * o_scale
        outputs[od.get("name", f"out_{i}")] = tensor
    _all_raw = np.concatenate([v.flatten() for v in outputs.values()]) if outputs else np.array([0.0], dtype=np.float32)
    debug = {
        "format": "raw_multi_head",
        "raw_min": float(_all_raw.min()),
        "raw_max": float(_all_raw.max()),
        "raw_mean": float(_all_raw.mean()),
    }

    reg_max = int(manifest.get("reg_max", 16))
    strides = [8, 16, 32]
    detections = []
    def _pick_head(want_size, want_dims):
        """Resolve one head by element count, with SHAPE breaking ties.

        Element count alone cross-wires heads: with strides (8, 16, 32),
            size(cls_p3) = 16 * gh5 * gw5 * len(labels)
            size(reg_p5) =      gh5 * gw5 * 4 * reg_max
        are equal whenever ``len(labels) == reg_max / 4`` — every 4-class model
        at the default reg_max=16 — and the reshape below then reads the p5 DFL
        tensor as p3 class scores (p3 carries ~76% of all anchors).
        """
        cands = [k for k, v in outputs.items() if v.size == want_size]
        if not cands:
            return None
        for k in cands:
            if tuple(outputs[k].shape)[-3:] == want_dims:
                return k
        return cands[0]

    for level, stride in enumerate(strides, start=3):
        cls_key = next((k for k in outputs if f"p{level}" in k.lower() and "cls" in k.lower()), None)
        reg_key = next((k for k in outputs if f"p{level}" in k.lower() and "reg" in k.lower()), None)
        gh = input_h // stride
        gw = input_w // stride
        expected_cls = gh * gw * len(labels)
        expected_reg = gh * gw * 4 * reg_max
        if cls_key is None:
            cls_key = _pick_head(expected_cls, (gh, gw, len(labels)))
        if reg_key is None:
            reg_key = _pick_head(expected_reg, (gh, gw, 4 * reg_max))
        # Same key for both heads = indistinguishable by name, size and shape
        # (len(labels) == 4 * reg_max); skip rather than pop the same tensor twice.
        if cls_key is None or reg_key is None or cls_key == reg_key:
            continue
        # pop: retire both tensors so a later level cannot reuse them.
        cls_out = outputs.pop(cls_key).reshape(gh, gw, len(labels))
        reg_out = outputs.pop(reg_key).reshape(gh, gw, 4 * reg_max)
        for y in range(gh):
            for x in range(gw):
                cls_idx = int(np.argmax(cls_out[y, x]))
                score = float(cls_out[y, x, cls_idx])
                if score < threshold:
                    continue
                reg = reg_out[y, x].reshape(4, reg_max)
                reg_shifted = reg - reg.max(axis=1, keepdims=True)
                e = np.exp(reg_shifted)
                prob = e / e.sum(axis=1, keepdims=True)
                bins = np.arange(reg_max, dtype=np.float32)
                # DFL distances are in grid units; scale to pixels by stride
                # (matches the canonical decode in yolo_pro/decode.py).
                l, t, r, b = (prob * bins[None, :]).sum(axis=1) * float(stride)
                cx = (x + 0.5) * stride
                cy = (y + 0.5) * stride
                # Raw multi-head reconstruction is always in pixel anchor space
                # by construction (stride * grid coords ± DFL regression bins).
                # The wire convention is always normalized, so divide by input
                # dims regardless of manifest.coordinate_space — that field
                # describes the model's *tensor* output and the multi-head path
                # doesn't read bbox values from a tensor.
                x1 = max(0.0, (cx - l) / input_w)
                y1 = max(0.0, (cy - t) / input_h)
                x2 = min(1.0, (cx + r) / input_w)
                y2 = min(1.0, (cy + b) / input_h)
                if x2 <= x1 or y2 <= y1:
                    continue
                detections.append({
                    "label": labels[cls_idx] if 0 <= cls_idx < len(labels) else str(cls_idx),
                    "confidence": score,
                    "bbox": {"x1": float(x1), "y1": float(y1), "x2": float(x2), "y2": float(y2)},
                    "_box": [x1, y1, x2, y2],
                    "_score": score,
                })

    return _finalize_yolo(detections, debug, iou_threshold, class_filter, ms)


def _finalize_yolo(detections, debug, iou_threshold, class_filter, ms):
    """Shared NMS + result-shape tail for both YOLO postprocess branches."""
    if class_filter:
        detections = [d for d in detections if d["label"] in class_filter]

    if detections:
        boxes_arr = np.array([d["_box"] for d in detections], dtype=np.float32)
        scores_arr = np.array([d["_score"] for d in detections], dtype=np.float32)
        order = np.argsort(scores_arr)[::-1]
        keep_indices = []
        while len(order) > 0:
            i = int(order[0])
            keep_indices.append(i)
            if len(order) == 1:
                break
            b = boxes_arr[i]
            rest = boxes_arr[order[1:]]
            x1_i = np.maximum(b[0], rest[:, 0])
            y1_i = np.maximum(b[1], rest[:, 1])
            x2_i = np.minimum(b[2], rest[:, 2])
            y2_i = np.minimum(b[3], rest[:, 3])
            inter = np.maximum(0.0, x2_i - x1_i) * np.maximum(0.0, y2_i - y1_i)
            area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
            area_r = np.maximum(0.0, rest[:, 2] - rest[:, 0]) * np.maximum(0.0, rest[:, 3] - rest[:, 1])
            iou = inter / (area_b + area_r - inter + 1e-6)
            order = order[1:][iou < iou_threshold]
        detections = [detections[i] for i in keep_indices]

    shape = manifest.get("input_shape", [96, 96, 3])
    input_h = int(shape[0]) if len(shape) >= 1 else 96
    input_w = int(shape[1]) if len(shape) >= 2 else 96
    result = []
    for d in detections:
        bb = d["bbox"]
        x1_px = int(round(bb["x1"] * input_w))
        y1_px = int(round(bb["y1"] * input_h))
        x2_px = int(round(bb["x2"] * input_w))
        y2_px = int(round(bb["y2"] * input_h))
        result.append({
            "label":  d["label"],
            "value":  d["confidence"],
            "x":      x1_px,
            "y":      y1_px,
            "width":  x2_px - x1_px,
            "height": y2_px - y1_px,
        })
    return (
        {
            "bounding_boxes": result,
            "debug": debug,
        },
        ms,
    )


def _postprocess_fomo(raw, ms):
    threshold = _packaged_threshold(manifest.get("threshold", 0.5))
    shape = manifest.get("input_shape", [96, 96, 3])
    img_h = int(shape[0]) if len(shape) >= 1 else 96
    img_w = int(shape[1]) if len(shape) >= 2 else 96
    n_cls_plus1 = len(labels) + 1
    total = raw.size
    aspect = img_w / max(img_h, 1)
    gh     = max(1, int(round((total / (n_cls_plus1 * aspect)) ** 0.5)))
    gw     = max(1, int(round(gh * aspect)))
    if gh * gw * n_cls_plus1 != total:
        side = max(1, int(round((total / n_cls_plus1) ** 0.5)))
        gh = gw = side
    cell  = raw.flatten().reshape(gh, gw, n_cls_plus1)
    obj   = cell[:, :, 1:]

    stride_x = img_w / gw
    stride_y = img_h / gh
    bboxes = []
    for row in range(gh):
        for col in range(gw):
            cls_idx = int(np.argmax(obj[row, col]))
            conf    = float(obj[row, col, cls_idx])
            if conf < threshold:
                continue
            peak = True
            # 8-connected local-max NMS matches decode_fomo_heatmap in fomo_evaluator.py.
            for dr, dc in ((-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)):
                nr, nc = row+dr, col+dc
                if 0 <= nr < gh and 0 <= nc < gw:
                    if float(obj[nr, nc, cls_idx]) > conf:
                        peak = False; break
            if not peak:
                continue
            bboxes.append({
                "label":  labels[cls_idx] if cls_idx < len(labels) else str(cls_idx),
                "value":  conf,
                "x":      int(col * stride_x),
                "y":      int(row * stride_y),
                "width":  int(stride_x),
                "height": int(stride_y),
            })
    return ({"bounding_boxes": bboxes}, ms)


# ── EI .eim stdio-JSONL protocol ─────────────────────────────────────────────

_shape = _input_shape
_block_types = {b.get("type") for b in _dsp_blocks}
if len(_shape) >= 3:
    _sensor_type = 3
elif _block_types & {"mfcc", "spectrogram"}:
    _sensor_type = 1
else:
    _sensor_type = 2


_NO_ID = object()

def _hello_payload(req_id=_NO_ID):
    """Build the EI hello / re-handshake frame.

    Manifest is canonical: project_id is int, model_type is EI wire vocab,
    image_resize_mode / thresholds are passed through verbatim. No fabricated
    values or fallback magic numbers — a missing required manifest field is
    a packager bug and should fail loudly.
    """
    shape = _shape
    image_h = int(shape[0]) if len(shape) >= 1 else 0
    image_w = int(shape[1]) if len(shape) >= 2 else 0
    image_c = int(shape[2]) if len(shape) >= 3 else 0
    input_features_count = image_h * image_w * image_c if _is_image_model else int(
        np.prod(shape) if shape else 0
    )
    payload = {
        "hello":   1,
        "version": "1",
        "model_parameters": {
            "runner_type":          "pxe",
            "sensor":               _sensor_type,
            "image_input_width":    image_w,
            "image_input_height":   image_h,
            "image_channel_count":  image_c,
            "image_input_frames":   int(manifest.get("image_input_frames", 1)),
            "image_resize_mode":    manifest["image_resize_mode"],
            "input_features_count": int(input_features_count),
            "freq":                 _freq_hz,
            "has_anomaly":          1 if manifest.get("has_anomaly") else 0,
            "label_count":          len(labels),
            "labels":               labels,
            "model_type":           manifest["model_type"],
            "thresholds":           manifest["thresholds"],
        },
        "project": {
            "id":             int(manifest["project_id"]),
            "name":           manifest.get("project_name", "pxe_model"),
            "owner":          manifest.get("project_owner", "petaledge"),
            "deploy_version": int(manifest.get("deploy_version", 1)),
        },
    }
    if req_id is not _NO_ID:
        payload["id"] = req_id
    return payload


print(json.dumps(_hello_payload()), flush=True)

for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    try:
        msg = json.loads(line)
    except json.JSONDecodeError:
        print(json.dumps({"result": 0, "error": "invalid json"}), flush=True)
        continue
    if "stop" in msg:
        break
    if "hello" in msg:
        print(json.dumps(_hello_payload(msg.get("id"))), flush=True)
        continue
    if "set_threshold" in msg:
        try:
            _runtime_threshold = float(msg["set_threshold"])
            print(json.dumps({"success": True, "id": msg.get("id")}), flush=True)
        except (TypeError, ValueError) as exc:
            print(json.dumps({
                "success": False,
                "id": msg.get("id"),
                "error": f"invalid set_threshold value: {exc}",
            }), flush=True)
        continue
    if "set_parameter" in msg:
        payload = msg["set_parameter"]
        if isinstance(payload, dict) and "threshold" in payload:
            try:
                _runtime_threshold = float(payload["threshold"])
            except (TypeError, ValueError) as exc:
                print(json.dumps({
                    "success": False,
                    "id": msg.get("id"),
                    "error": f"invalid set_parameter.threshold value: {exc}",
                }), flush=True)
                continue
        print(json.dumps({"success": True, "id": msg.get("id")}), flush=True)
        continue
    if "classify" in msg:
        try:
            payload, ms = _classify(msg["classify"])
            print(json.dumps({
                "result":  1,
                "id":      msg.get("id"),
                **payload,
                "timing":  {"dsp": _dsp_ms[0], "classification": ms, "anomaly": 0},
            }), flush=True)
        except Exception as exc:
            print(json.dumps({"result": 0, "id": msg.get("id"), "error": str(exc)}), flush=True)
