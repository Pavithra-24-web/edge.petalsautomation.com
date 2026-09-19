# decode_export.py
# Decoded-output export for MobileNetV2 SSD FPN-Lite 320x320 — PXE parity.
#
# The training/raw SSD graph (model.py) emits TWO raw tensors:
#     box_predictions : (B, 8028, 4)   anchor offsets (variances [10,10,5,5])
#     cls_predictions : (B, 8028, 90)  per-class logits (no background column)
#
# The production .pxe runtime contract (unoq/runtime/runner.py
# `_postprocess_yolo_decoded`) consumes a SINGLE decoded tensor:
#     (1, N, 6)  →  [x1, y1, x2, y2, score, class_id]   normalized [0, 1]
#
# This module wraps a trained SSD model with a decode layer that bakes the
# anchors in as graph constants, decodes box offsets, applies sigmoid, and
# emits that (1, N, 6) tensor — exactly mirroring yolo_pro/decode.py.
#
# NMS placement (decision, mirrors YOLO-Pro):
#   ALL anchors are emitted with no confidence filter and no NMS.  The runner's
#   decoded handler applies the threshold (per row) and NMS (`_finalize_yolo`).
#   This matches yolo_pro.export.export_tflite_float32_decoded /
#   DecodeDetectionsLayer, which also emit every anchor and defer NMS to the
#   runtime.  See yolo_pro/decode.py docstring ("WITHOUT confidence filtering
#   or NMS").
#
# The training graph (model.py), anchors (anchor_generator.py) and box coder
# (box_coder.py) are NOT modified — this is a pure export-time wrapper.

from __future__ import annotations

import os
import tempfile
from typing import Tuple

import numpy as np
import tensorflow as tf
from tensorflow import keras

# Bare-name imports resolve once the MobileNetV2 SSD/ folder is on sys.path
# (done by the package __init__.py shim and mobilenetv2_ssd_worker._ensure_ssd_package).
from primitives import CFG
from anchor_generator import generate_anchors
from box_coder import decode_boxes, VARIANCES


# ─────────────────────────────────────────────────────────────────────────────
# Numpy reference decode  (test / evaluation parity)
# ─────────────────────────────────────────────────────────────────────────────

def decode_ssd_outputs_np(
    box_preds: np.ndarray,
    cls_preds: np.ndarray,
    anchors: np.ndarray,
) -> np.ndarray:
    """
    Batch numpy decode of raw SSD head outputs → (B, N, 6).

    Canonical reference implementation; ``SSDDecodeLayer`` mirrors this math
    exactly so export and evaluation are numerically identical.  Decode math is
    the inverse of mobilenetv2_ssd_worker._encode_targets and matches
    box_coder.decode_boxes (variances [10, 10, 5, 5], exponent clamp ±10).

    Args:
        box_preds : (B, A, 4)            raw offsets [dy, dx, dh, dw].
        cls_preds : (B, A, num_classes)  raw class logits.
        anchors   : (A, 4)               anchors [cy, cx, h, w], normalized.

    Returns:
        (B, A, 6) float32 — columns [x1, y1, x2, y2, score, class_id].
        All anchors included (no threshold, no NMS).  class_id is the 0-indexed
        argmax over num_classes, stored as float32.
    """
    box_preds = np.asarray(box_preds, dtype=np.float32)
    cls_preds = np.asarray(cls_preds, dtype=np.float32)
    anchors   = np.asarray(anchors,   dtype=np.float32)

    v = np.asarray(VARIANCES, dtype=np.float32)  # [10, 10, 5, 5]
    anc_cy, anc_cx, anc_h, anc_w = (anchors[:, i] for i in range(4))

    t_cy, t_cx, t_h, t_w = (box_preds[..., i] for i in range(4))

    pred_cy = (t_cy / v[0]) * anc_h + anc_cy
    pred_cx = (t_cx / v[1]) * anc_w + anc_cx
    pred_h  = np.exp(np.clip(t_h / v[2], -10.0, 10.0)) * anc_h
    pred_w  = np.exp(np.clip(t_w / v[3], -10.0, 10.0)) * anc_w

    y1 = np.clip(pred_cy - pred_h / 2.0, 0.0, 1.0)
    x1 = np.clip(pred_cx - pred_w / 2.0, 0.0, 1.0)
    y2 = np.clip(pred_cy + pred_h / 2.0, 0.0, 1.0)
    x2 = np.clip(pred_cx + pred_w / 2.0, 0.0, 1.0)

    scores_all = 1.0 / (1.0 + np.exp(-cls_preds))            # sigmoid (B, A, C)
    score      = scores_all.max(axis=-1)                     # (B, A)
    class_id   = scores_all.argmax(axis=-1).astype(np.float32)

    # Wire order is [x1, y1, x2, y2] (runner _postprocess_yolo_decoded).
    boxes = np.stack([x1, y1, x2, y2], axis=-1)             # (B, A, 4)
    return np.concatenate(
        [boxes, score[..., np.newaxis], class_id[..., np.newaxis]], axis=-1,
    ).astype(np.float32)                                     # (B, A, 6)


# ─────────────────────────────────────────────────────────────────────────────
# TF-native decode layer  (export / runtime)
# ─────────────────────────────────────────────────────────────────────────────

class SSDDecodeLayer(keras.layers.Layer):
    """
    Decode raw SSD heads into a single (B, N, 6) detection tensor.

    Input  : dict {"box_predictions": (B, A, 4), "cls_predictions": (B, A, C)}
    Output : (B, A, 6) float32 — [x1, y1, x2, y2, score, class_id], normalized.

    Anchors are stored as a numpy array in __init__ and converted to
    tf.constant inside call(), so they become Const nodes in the exported
    graph (no runtime anchor computation), mirroring yolo_pro
    DecodeDetectionsLayer.  All anchors are emitted — confidence filtering and
    NMS happen in the runner.
    """

    def __init__(self, anchors_np: np.ndarray | None = None, **kwargs):
        super().__init__(**kwargs)
        if anchors_np is None:
            anchors_np = generate_anchors().numpy()
        self._anchors_np = np.asarray(anchors_np, dtype=np.float32)  # (A, 4)

    def call(self, inputs: dict) -> tf.Tensor:
        box_preds = inputs["box_predictions"]   # (B, A, 4)
        cls_preds = inputs["cls_predictions"]   # (B, A, C)

        anchors = tf.constant(self._anchors_np, dtype=tf.float32)    # (A, 4)

        # decode_boxes returns [y_min, x_min, y_max, x_max] clipped to [0, 1].
        decoded_yxyx = decode_boxes(box_preds, anchors, variances=VARIANCES, clip=True)
        y1, x1, y2, x2 = tf.unstack(decoded_yxyx, axis=-1)
        boxes_xyxy = tf.stack([x1, y1, x2, y2], axis=-1)            # (B, A, 4)

        scores_all = tf.sigmoid(cls_preds)                          # (B, A, C)
        score = tf.reduce_max(scores_all, axis=-1, keepdims=True)   # (B, A, 1)
        class_id = tf.cast(
            tf.argmax(scores_all, axis=-1, output_type=tf.int32), tf.float32,
        )
        class_id = tf.expand_dims(class_id, axis=-1)                # (B, A, 1)

        return tf.concat([boxes_xyxy, score, class_id], axis=-1)    # (B, A, 6)

    def get_config(self) -> dict:
        cfg = super().get_config()
        cfg.update({"anchors_np": self._anchors_np.tolist()})
        return cfg

    @classmethod
    def from_config(cls, config: dict) -> "SSDDecodeLayer":
        anchors = config.pop("anchors_np", None)
        if anchors is not None:
            anchors = np.asarray(anchors, dtype=np.float32)
        return cls(anchors_np=anchors, **config)


# ─────────────────────────────────────────────────────────────────────────────
# Decoded model builder
# ─────────────────────────────────────────────────────────────────────────────

def build_ssd_decoded_model(
    raw_model: keras.Model,
    input_shape: Tuple[int, int, int] = CFG.INPUT_SHAPE,
    anchors_np: np.ndarray | None = None,
) -> keras.Model:
    """
    Wrap a trained SSD model with SSDDecodeLayer.

    The resulting model keeps the raw model's uint8 input + in-graph
    NormalizationLayer and exposes a single decoded output:

        (batch, N, 6)  =  [x1, y1, x2, y2, score, class_id]   normalized

    The raw model's weights are unchanged — this wrapper is export-only and
    mirrors yolo_pro.decode.build_yolo_pro_decoded.

    Args:
        raw_model   : trained model from MobileNetV2 SSD/model.py:build_model
                      (outputs dict {box_predictions, cls_predictions}).
        input_shape : (H, W, C) — must match the raw model (320, 320, 3).
        anchors_np  : optional anchors override (defaults to generate_anchors()).

    Returns:
        keras.Model with a single (batch, N, 6) float32 output.
    """
    inputs = keras.Input(shape=input_shape, dtype=tf.uint8, name="input_image")
    raw_outputs = raw_model(inputs)
    decoded = SSDDecodeLayer(anchors_np=anchors_np, name="ssd_decode")(raw_outputs)
    return keras.Model(inputs=inputs, outputs=decoded,
                       name=raw_model.name + "_decoded")


# ─────────────────────────────────────────────────────────────────────────────
# TFLite export  (float32, decoded single output)
# ─────────────────────────────────────────────────────────────────────────────

def export_ssd_decoded_tflite(
    raw_model: keras.Model,
    output_path: str | None = None,
    input_shape: Tuple[int, int, int] = CFG.INPUT_SHAPE,
    anchors_np: np.ndarray | None = None,
) -> bytes:
    """
    Build the decoded SSD model and convert it to a float32 TFLite flatbuffer.

    Mirrors the float32 conversion path used by mobilenetv2_ssd_worker
    (model.export() → TFLiteConverter.from_saved_model, builtins only).  The
    decoded model self-normalizes uint8 input via the in-graph NormalizationLayer,
    so the runtime must feed raw uint8 pixels (manifest normalize_input=False).

    Returns the raw .tflite bytes (also written to ``output_path`` if given).
    """
    decoded_model = build_ssd_decoded_model(raw_model, input_shape, anchors_np)

    with tempfile.TemporaryDirectory() as saved_model_dir:
        decoded_model.export(saved_model_dir)
        conv = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
        conv.optimizations               = []
        conv.target_spec.supported_types = [tf.float32]
        conv.target_spec.supported_ops   = [tf.lite.OpsSet.TFLITE_BUILTINS]
        conv.allow_custom_ops            = False
        conv.experimental_new_converter  = True
        tflite_bytes = conv.convert()

    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, "wb") as f:
            f.write(tflite_bytes)

    return tflite_bytes
