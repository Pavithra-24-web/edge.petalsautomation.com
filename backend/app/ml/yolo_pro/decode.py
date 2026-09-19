"""
yolo_pro/decode.py
──────────────────
Unified decoded detection output for YOLO-Pro export, evaluation, and runtime.

OG parity context
─────────────────
The OG Edge Impulse YOLO-Pro model exports a single decoded tensor:

    TensorSpec(shape=(None, 189, 6), dtype=tf.float32)

where 189 = P3(144) + P4(36) + P5(9) for a 96×96 input (strides 8/16/32), and:

    column 0-3 : x1, y1, x2, y2  — normalised [0, 1]
    column 4   : score            — max sigmoid class probability
    column 5   : class_id        — argmax class index (stored as float32)

All N_total anchor positions are emitted WITHOUT confidence filtering or NMS.
The runtime / firmware applies those steps after receiving the tensor.

This module provides the following components, which all implement the same
decode math so that export and evaluation are numerically identical:

  decode_raw_outputs_np(raw_preds, H, W, reg_max, strides)
      Batch numpy decode → (B, N_total, 6) ndarray.
      Used in evaluation (replaces the old per-anchor Python loop).

  DecodeDetectionsLayer(H, W, reg_max, strides)
      Keras layer: dict of raw outputs → (B, N_total, 6) TF tensor.
      Anchor constants are embedded as Const nodes in the exported graph.

  build_yolo_pro_decoded(raw_model, input_shape, reg_max, strides)
      Wraps any raw YOLO-Pro model with the decode layer.
      The resulting model is the direct equivalent of the OG exported model.

  total_anchor_count(H, W, strides)
      Returns the expected N_total for a given input size.

Residual gap from OG
────────────────────
The OG spec shows `dtype=tf.float32` output, which our float32 decoded export
matches exactly.  The existing int8 export is kept as-is (raw head outputs)
since decoded [x1,y1,x2,y2,score,class_id] coordinates cannot be meaningfully
quantized to int8 without loss of localization precision.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import tensorflow as tf
from tensorflow import keras


# ─────────────────────────────────────────────────────────────────────────────
# Anchor helpers
# ─────────────────────────────────────────────────────────────────────────────

def _anchor_np(H: int, W: int, stride: int) -> np.ndarray:
    """
    Compute anchor centre points for one scale.

    Returns (Hg*Wg, 2) float32 array in pixel coordinates (cx, cy).
    Uses the same formula as _make_anchor_points() in yolo_pro_worker.py so
    decode_raw_outputs_np and _decode_yolo_pro_predictions are numerically
    identical.
    """
    Hg, Wg = H // stride, W // stride
    gy, gx = np.meshgrid(np.arange(Hg), np.arange(Wg), indexing="ij")
    cx = (gx.flatten() + 0.5) * stride
    cy = (gy.flatten() + 0.5) * stride
    return np.stack([cx, cy], axis=-1).astype(np.float32)   # (N, 2)


def total_anchor_count(
    H: int,
    W: int,
    strides: Tuple[int, ...] = (8, 16, 32),
) -> int:
    """
    Return total anchor count across all scales.

    For 96×96 input  → 144 + 36 + 9  = 189  (matches OG TensorSpec)
    For 320×320 input → 1600 + 400 + 100 = 2100
    """
    return sum((H // s) * (W // s) for s in strides)


# ─────────────────────────────────────────────────────────────────────────────
# Numpy batch decode  (evaluation / reference implementation)
# ─────────────────────────────────────────────────────────────────────────────

def decode_raw_outputs_np(
    raw_preds: dict,
    H: int,
    W: int,
    reg_max: int,
    strides: Tuple[int, ...] = (8, 16, 32),
) -> np.ndarray:
    """
    Batch numpy decode of raw YOLO-Pro head outputs.

    Canonical decode implementation — the TF-native DecodeDetectionsLayer
    mirrors this math exactly, ensuring evaluation and export are identical.

    Args:
        raw_preds : dict with keys cls_p3/reg_p3/cls_p4/reg_p4/cls_p5/reg_p5.
                    Values may be numpy arrays or TF tensors (auto-converted).
        H, W      : input image height/width in pixels.
        reg_max   : DFL distribution bins (must match training config, default 16).
        strides   : per-scale strides for P3/P4/P5 (default (8, 16, 32)).

    Returns:
        (B, N_total, 6) float32 ndarray.
        Columns: [x1, y1, x2, y2, score, class_id]
        All N_total anchors are included — no threshold, no NMS.
        Callers apply confidence filtering and NMS downstream.
    """
    bins = np.arange(reg_max, dtype=np.float32)
    scale_names = ["p3", "p4", "p5"][: len(strides)]
    scale_results: List[np.ndarray] = []

    for scale_name, stride in zip(scale_names, strides):
        cls_raw = raw_preds.get(f"cls_{scale_name}")
        reg_raw = raw_preds.get(f"reg_{scale_name}")
        if cls_raw is None or reg_raw is None:
            continue

        # Accept TF tensors transparently
        if hasattr(cls_raw, "numpy"):
            cls_raw = cls_raw.numpy()
        if hasattr(reg_raw, "numpy"):
            reg_raw = reg_raw.numpy()

        cls_raw = np.asarray(cls_raw, dtype=np.float32)
        reg_raw = np.asarray(reg_raw, dtype=np.float32)

        B, Hg, Wg, n_cls = cls_raw.shape
        N = Hg * Wg

        cls_flat = cls_raw.reshape(B, N, n_cls)            # (B, N, n_cls)
        reg_flat = reg_raw.reshape(B, N, 4 * reg_max)      # (B, N, 4*reg_max)

        # DFL decode: softmax → expected bin → grid units → pixels
        logits    = reg_flat.reshape(B, N, 4, reg_max)
        logits    = logits - logits.max(axis=-1, keepdims=True)  # numerical stability
        probs     = np.exp(logits)
        probs    /= probs.sum(axis=-1, keepdims=True) + 1e-9
        ltrb_grid = (probs * bins).sum(axis=-1)             # (B, N, 4) grid units
        ltrb_px   = ltrb_grid * float(stride)               # → pixels

        # Anchor centres in pixel coords (same formula as _make_anchor_points)
        anch = _anchor_np(H, W, stride)   # (N, 2)
        ax, ay = anch[:, 0], anch[:, 1]   # (N,)

        x1 = np.clip((ax - ltrb_px[:, :, 0]) / float(W), 0., 1.)
        y1 = np.clip((ay - ltrb_px[:, :, 1]) / float(H), 0., 1.)
        x2 = np.clip((ax + ltrb_px[:, :, 2]) / float(W), 0., 1.)
        y2 = np.clip((ay + ltrb_px[:, :, 3]) / float(H), 0., 1.)

        # Class: max probability and argmax index
        scores    = cls_flat.max(axis=-1)                        # (B, N)
        class_ids = cls_flat.argmax(axis=-1).astype(np.float32)  # (B, N)

        boxes = np.stack([x1, y1, x2, y2], axis=-1)             # (B, N, 4)
        dets  = np.concatenate(
            [boxes, scores[:, :, np.newaxis], class_ids[:, :, np.newaxis]],
            axis=-1,
        )                                                         # (B, N, 6)
        scale_results.append(dets)

    return np.concatenate(scale_results, axis=1)                 # (B, N_total, 6)


# ─────────────────────────────────────────────────────────────────────────────
# TF-native decode layer  (export / runtime)
# ─────────────────────────────────────────────────────────────────────────────

class DecodeDetectionsLayer(keras.layers.Layer):
    """
    Post-processing decode layer for YOLO-Pro.

    Converts the raw {cls_pX, reg_pX} head output dict into a single decoded
    detection tensor that matches the OG Edge Impulse export format:

        (batch, N_total, 6)  =  [x1, y1, x2, y2, score, class_id]

    All N_total anchor positions are emitted without confidence filtering or
    NMS — consistent with the OG convention where the runtime handles those
    steps downstream.

    Implementation notes
    ────────────────────
    - Anchor constants are stored as numpy arrays in __init__ and converted to
      tf.constant tensors inside call().  This embeds them as Const nodes in
      the frozen graph (convert_variables_to_constants_v2 path used by the
      exporter), avoiding any runtime anchor computation.
    - DFL decode uses tf.nn.softmax over reg_max bins, identical in math to
      the numpy decode_raw_outputs_np() reference.
    - class_id is stored as tf.float32 for TFLite output compatibility.
    """

    def __init__(
        self,
        H: int,
        W: int,
        reg_max: int,
        strides: Tuple[int, ...] = (8, 16, 32),
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.H       = H
        self.W       = W
        self.reg_max = reg_max
        self.strides = list(strides)
        # Pre-compute anchor arrays as numpy — embedded as Const at export
        self._anchors_np: List[np.ndarray] = [
            _anchor_np(H, W, s) for s in strides
        ]

    def call(self, inputs: dict) -> tf.Tensor:
        """
        Args:
            inputs : dict {cls_p3, reg_p3, cls_p4, reg_p4, cls_p5, reg_p5}
        Returns:
            (B, N_total, 6) float32 tensor
        """
        bins  = tf.cast(tf.range(self.reg_max), tf.float32)  # (reg_max,)
        parts: List[tf.Tensor] = []

        scale_names = ["p3", "p4", "p5"][: len(self.strides)]

        for scale_name, anch_np, stride in zip(
            scale_names, self._anchors_np, self.strides
        ):
            cls_raw = inputs[f"cls_{scale_name}"]   # (B, Hg, Wg, n_cls)
            reg_raw = inputs[f"reg_{scale_name}"]   # (B, Hg, Wg, 4*reg_max)

            B    = tf.shape(cls_raw)[0]
            Hg   = tf.shape(cls_raw)[1]
            Wg   = tf.shape(cls_raw)[2]
            n_cls = tf.shape(cls_raw)[3]
            N    = Hg * Wg

            cls_flat = tf.reshape(cls_raw, [B, N, n_cls])              # (B, N, C)
            reg_flat = tf.reshape(reg_raw, [B, N, 4 * self.reg_max])   # (B, N, 4R)

            # DFL decode → ltrb in grid units → pixels
            logits    = tf.reshape(reg_flat, [B, N, 4, self.reg_max])
            probs     = tf.nn.softmax(logits, axis=-1)                  # (B, N, 4, R)
            ltrb_grid = tf.reduce_sum(probs * bins, axis=-1)            # (B, N, 4)
            ltrb_px   = ltrb_grid * float(stride)

            # Anchor constants — become Const nodes in the frozen export graph
            anch = tf.constant(anch_np, dtype=tf.float32)               # (N, 2)
            ax, ay = anch[:, 0], anch[:, 1]                             # (N,)

            x1 = tf.clip_by_value((ax - ltrb_px[:, :, 0]) / float(self.W), 0., 1.)
            y1 = tf.clip_by_value((ay - ltrb_px[:, :, 1]) / float(self.H), 0., 1.)
            x2 = tf.clip_by_value((ax + ltrb_px[:, :, 2]) / float(self.W), 0., 1.)
            y2 = tf.clip_by_value((ay + ltrb_px[:, :, 3]) / float(self.H), 0., 1.)
            boxes = tf.stack([x1, y1, x2, y2], axis=-1)                # (B, N, 4)

            scores = tf.reduce_max(cls_flat, axis=-1, keepdims=True)   # (B, N, 1)
            class_ids = tf.cast(
                tf.argmax(cls_flat, axis=-1, output_type=tf.int32),
                tf.float32,
            )
            class_ids = tf.expand_dims(class_ids, axis=-1)             # (B, N, 1)

            parts.append(tf.concat([boxes, scores, class_ids], axis=-1))  # (B, N, 6)

        return tf.concat(parts, axis=1)                                 # (B, N_total, 6)

    def get_config(self) -> dict:
        cfg = super().get_config()
        cfg.update({
            "H":       self.H,
            "W":       self.W,
            "reg_max": self.reg_max,
            "strides": self.strides,
        })
        return cfg


# ─────────────────────────────────────────────────────────────────────────────
# Decoded model builder
# ─────────────────────────────────────────────────────────────────────────────

def build_yolo_pro_decoded(
    raw_model: keras.Model,
    input_shape: Tuple[int, int, int],
    reg_max: int = 16,
    strides: Tuple[int, ...] = (8, 16, 32),
) -> keras.Model:
    """
    Wrap a raw YOLO-Pro model with the DecodeDetectionsLayer.

    The resulting model exposes a single decoded output:

        (batch, N_total, 6)  =  [x1, y1, x2, y2, score, class_id]

    This matches the OG Edge Impulse export format:

        TensorSpec(shape=(None, 189, 6), dtype=tf.float32)

    for a 96×96 input (N_total = 144+36+9 = 189).

    The raw model's weights are unchanged — training continues on raw_model.
    This wrapper is used exclusively for export and runtime inference.

    Args:
        raw_model    : trained YOLO-Pro model returned by build_yolo_pro()
        input_shape  : (H, W, C) — must match the raw model's input shape
        reg_max      : DFL bins — must match the raw model's training config
        strides      : per-scale strides — must match the raw model

    Returns:
        keras.Model with a single (batch, N_total, 6) float32 output tensor
    """
    H, W, C = input_shape
    inputs      = keras.Input(shape=input_shape, name="image")
    raw_outputs = raw_model(inputs)
    decoded     = DecodeDetectionsLayer(
        H=H, W=W, reg_max=reg_max, strides=strides, name="decode_detections",
    )(raw_outputs)
    return keras.Model(
        inputs  = inputs,
        outputs = decoded,
        name    = raw_model.name + "_decoded",
    )
