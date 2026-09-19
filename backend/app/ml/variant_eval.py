"""
Per-variant post-export evaluation — Edge Impulse style "Model Version" metrics.

Background
──────────
Each training run exports several TFLite variants of the *same* weights
(``float32``, ``int8``, and for the detectors ``decoded_float32``).  Each one
gets its own ``TrainedModel`` row carrying ``model_metadata["variant"]``.

Historically only ONE evaluation was ever run — against the in-memory Keras
model — and its numbers were stored job-wide on ``TrainingJob
.classification_report``.  Every consumer then showed those same numbers no
matter which variant the user picked, so the accuracy cost of int8
quantization was invisible: the UI defaults to ``int8`` while every scoring
path silently fell back to ``float32``/``decoded_float32``.

This module closes that gap.  After export completes, each successfully
exported variant is loaded back through the TFLite interpreter and scored
independently on the very same validation set, and the resulting metrics are
persisted onto that variant's own ``TrainedModel`` row under
``model_metadata["variant_metrics"]``.

Why ``model_metadata`` and not a new table
──────────────────────────────────────────
``TrainedModel`` is already exactly one row per variant, and
``model_metadata`` is already a JSON column that already carries the
per-variant discriminators (``variant``, ``quantized``, ``threshold``,
``ranking_status``).  Metrics are 1:1 with that row, are read in the same
query that loads the row, and are written by the same post-export backfill
pass that already stamps ``threshold``/``ranking_status``.  A separate table
would add a join and a migration to model a relationship that is already
1:1, with no query the JSON column cannot serve.  So: no new table, no
migration.

Reuse, not reimplementation
───────────────────────────
No new metric math lives here.  Each worker passes in the *same* evaluation
callable it already uses for its architecture; this module only supplies the
TFLite-backed stand-in for the Keras model, so the eval function cannot tell
the difference:

    metrics = scorer(runner)     # runner quacks like model(x, training=False)

Quantization handling is the round → clip → cast / (raw - zero) * scale
convention already used by ``deployment_worker._run_model_unified`` and
``model_testing_service._run_inference`` — not new scaling math.

Failure isolation
─────────────────
A variant whose eval raises is recorded with ``status="failed"`` and the
error string; it never propagates.  One bad variant must not fail the
training job or block the other variants' metrics.
"""

from __future__ import annotations

import logging
import math
from typing import Callable, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# ``model_metadata`` key holding this variant's own evaluation result.
VARIANT_METRICS_KEY = "variant_metrics"

# Sentinel dict key used to hand an ALREADY-decoded (B, N, 6) detection tensor
# through to ``_decode_yolo_pro_predictions``.  The decoded_float32 export bakes
# ``DecodeDetectionsLayer`` into the graph, and that layer's math is a mirror of
# ``decode_raw_outputs_np`` (same normalized xyxy, same reduce_max score) — so
# its output can be substituted for the numpy decode's output directly rather
# than being decoded a second time.
DECODED_ROWS_KEY = "__decoded_rows__"

# Metric fields lifted out of an eval report onto the stored variant record.
# Everything the detection eval already computes is kept; ``detailed_metrics``
# carries the extended COCO breakdown when the eval produced one.
_METRIC_FIELDS = (
    # Detection (YOLO-Pro, SSD)
    "map", "map50", "map75", "precision", "recall",
    "per_class", "detailed_metrics",
    "conf_threshold", "nms_iou_threshold",
    # Classification / FOMO
    "accuracy", "f1_score", "threshold",
    # Common
    "evaluated_images",
)


def _sanitize(obj):
    """Replace non-finite floats with None so the JSON column accepts them.

    Mirrors ``yolo_pro_worker._sanitize_json``; duplicated here to keep this
    module importable from every worker without a circular import.
    """
    if isinstance(obj, float):
        return None if not math.isfinite(obj) else obj
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return _sanitize(obj.item())
    if isinstance(obj, np.ndarray):
        return _sanitize(obj.tolist())
    return obj


# ─────────────────────────────────────────────────────────────────────────────
# TFLite stand-in for a Keras model
# ─────────────────────────────────────────────────────────────────────────────

class TFLiteVariantRunner:
    """Callable wrapper that makes a .tflite variant quack like a Keras model.

    ``evaluate_yolo_pro_detection`` (and the SSD/FOMO evals) invoke their model
    as ``model(x, training=False)`` and read a dict of named outputs.  This
    class presents that same interface over a TFLite interpreter, handling the
    int8 input quantization and output dequantization so the caller's eval code
    is byte-identical across variants.

    Output naming is the problem TFLite conversion creates: converted graphs
    rename tensors to things like ``StatefulPartitionedCall:0``.  For YOLO-Pro
    we resolve names the same way ``deployment_worker._run_model_unified``
    does — match ``p{level}`` + ``cls``/``reg`` in the tensor name first, then
    fall back to matching the tensor's element count against the expected
    ``gh*gw*num_classes`` / ``gh*gw*4*reg_max``.
    """

    def __init__(
        self,
        tflite_bytes: bytes,
        *,
        output_type: str = "",
        num_classes: int = 0,
        reg_max: int = 16,
        input_h: int = 0,
        input_w: int = 0,
        strides: tuple = (8, 16, 32),
    ):
        import tensorflow as tf

        # Match model_testing_service._load_model_into_memory: the default
        # XNNPACK delegate has been observed to abort the worker on Windows,
        # so prefer the plain builtin resolver and only fall back if it is
        # unavailable in this TF build.
        try:
            self._interp = tf.lite.Interpreter(
                model_content=tflite_bytes,
                experimental_op_resolver_type=(
                    tf.lite.experimental.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES
                ),
            )
        except Exception:
            self._interp = tf.lite.Interpreter(model_content=tflite_bytes)
        self._interp.allocate_tensors()

        self._in_detail   = self._interp.get_input_details()[0]
        self._out_details = self._interp.get_output_details()
        self.output_type  = output_type
        self.num_classes  = int(num_classes)
        self.reg_max      = int(reg_max)
        self.input_h      = int(input_h)
        self.input_w      = int(input_w)
        self.strides      = tuple(strides)

    # -- input / output plumbing ------------------------------------------
    def _prepare_input(self, x: np.ndarray) -> np.ndarray:
        """Reshape to the interpreter's fixed input shape and quantize if int8.

        round → clip → cast, the same convention as
        ``deployment_worker._run_model_unified`` (a bare cast truncates and
        biases every activation low).
        """
        arr = np.asarray(x, dtype=np.float32)
        arr = arr.reshape(tuple(int(d) for d in self._in_detail["shape"]))

        dtype = self._in_detail["dtype"]
        scale, zero = self._in_detail.get("quantization", (0.0, 0))
        if scale not in (0, 0.0):
            lo, hi = (-128, 127) if dtype == np.int8 else (0, 255)
            return np.clip(np.round(arr / scale + zero), lo, hi).astype(dtype)
        return arr.astype(dtype)

    def _read_outputs(self) -> List[np.ndarray]:
        """Fetch every output tensor, dequantizing int8/uint8 back to float."""
        outs = []
        for od in self._out_details:
            raw = self._interp.get_tensor(od["index"]).astype(np.float32)
            scale, zero = od.get("quantization", (0.0, 0))
            if scale not in (0, 0.0):
                raw = (raw - zero) * scale
            outs.append(raw)
        return outs

    def _describe_outputs(self, arrays: List[np.ndarray]) -> str:
        """``name(shape)`` for every graph output — for unresolved-head errors."""
        return ", ".join(
            f"{od.get('name', f'out_{i}')}{tuple(arrays[i].shape)}"
            for i, od in enumerate(self._out_details)
        ) or "<none>"

    def _name_outputs(self, arrays: List[np.ndarray]) -> Dict[str, np.ndarray]:
        """Map interpreter outputs back onto the eval's expected key names."""
        named = {
            od.get("name", f"out_{i}"): arrays[i]
            for i, od in enumerate(self._out_details)
        }

        # Single decoded (B, N, 6) tensor — the decoded_float32 export.  Hand it
        # through under the sentinel key; its rows are already in exactly the
        # form decode_raw_outputs_np would have produced.
        if len(arrays) == 1 and arrays[0].ndim == 3 and arrays[0].shape[-1] == 6:
            return {DECODED_ROWS_KEY: arrays[0]}

        if self.output_type == "ssd_detection":
            # MobileNetV2 SSD's eval reads two named heads.  Resolve by name
            # first, then by trailing dimension: the box head is (B, A, 4) and
            # the class head is (B, A, num_classes).
            resolved_ssd: Dict[str, np.ndarray] = {}
            for key, token, last_dim in (
                ("box_predictions", "box", 4),
                ("cls_predictions", "cls", self.num_classes),
            ):
                hit = next((k for k in named if token in k.lower()), None)
                if hit is None:
                    hit = next(
                        (k for k, v in named.items()
                         if v.ndim >= 2 and v.shape[-1] == last_dim), None,
                    )
                if hit is not None:
                    resolved_ssd[key] = named.pop(hit)
            if len(resolved_ssd) == 2:
                return resolved_ssd
            # Neither the name tokens nor the trailing-dimension fallback found
            # both heads.  Returning the unresolved dict here would hand the
            # eval a dict without ``box_predictions``/``cls_predictions`` and
            # surface as a KeyError (or worse, a wrong-shape decode) far from
            # the cause.  Fail loudly with what the graph actually exposes.
            raise RuntimeError(
                "Model inference failed: could not resolve the SSD output heads "
                f"(box_predictions/cls_predictions) from this graph. Resolved "
                f"{sorted(resolved_ssd)}; available outputs: "
                f"{self._describe_outputs(arrays)} "
                f"(expected a (B, A, 4) box head and a (B, A, {self.num_classes}) "
                "class head)."
            )

        if self.output_type != "yolo_pro_detection":
            # Non-YOLO architectures consume positional outputs; keep the
            # interpreter's own names plus stable positional aliases.
            #
            # A single-output graph gets NO alias.  Aliasing it would produce a
            # 2-key dict ({tensor_name, "out_0"}) pointing at the SAME array,
            # and ``predict`` only collapses a dict to an ndarray when it holds
            # one output — so FOMO's ``np.asarray(model.predict(X), np.float32)``
            # received a dict and died with "float() argument must be a string
            # or a real number, not 'dict'".
            if len(arrays) > 1:
                named.update({f"out_{i}": a for i, a in enumerate(arrays)})
            return named

        # YOLO-Pro raw heads: name match first, SHAPE match second.
        #
        # Shape, not element count.  Matching on ``v.size`` alone silently
        # crosses the wires whenever two heads have the same number of
        # elements, and the reshape below then reinterprets the bytes instead
        # of raising: with strides (8, 16, 32) the p3 grid is 4x the p5 grid,
        # so
        #     size(cls_p3) = 16 * gh5 * gw5 * num_classes
        #     size(reg_p5) =      gh5 * gw5 * 4 * reg_max
        # collide exactly when ``num_classes == reg_max / 4`` — i.e. every
        # 4-class model at the default reg_max=16.  On such a model cls_p3
        # (76% of all anchors) was fed the p5 DFL tensor, and the variant's
        # mAP collapsed (~6%) while the decoded_float32 export, which has a
        # single output and skips this path entirely, scored correctly.
        #
        # Consumed tensors are also popped, so two heads can never resolve to
        # the same graph output.
        resolved: Dict[str, np.ndarray] = {}
        for level, stride in zip((3, 4, 5), self.strides):
            gh = self.input_h // stride
            gw = self.input_w // stride
            exp_cls = gh * gw * self.num_classes
            exp_reg = gh * gw * 4 * self.reg_max
            shape_cls = (gh, gw, self.num_classes)
            shape_reg = (gh, gw, 4 * self.reg_max)

            def _by_shape(want: tuple) -> Optional[str]:
                """First unconsumed output whose trailing dims are exactly `want`.

                Accepts the (B, gh, gw, C) layout TFLite emits as well as an
                already-flattened (B, gh*gw, C) view of the same head.
                """
                flat = (want[0] * want[1], want[2])
                for k, v in named.items():
                    dims = tuple(int(d) for d in v.shape[1:])
                    if dims == want or dims == flat:
                        return k
                return None

            cls_key = next(
                (k for k in named
                 if f"p{level}" in k.lower() and "cls" in k.lower()), None,
            )
            reg_key = next(
                (k for k in named
                 if f"p{level}" in k.lower() and "reg" in k.lower()), None,
            )
            if cls_key is None:
                cls_key = _by_shape(shape_cls)
            if reg_key is None:
                reg_key = _by_shape(shape_reg)
            if cls_key is not None and cls_key == reg_key:
                # Both heads resolved to one tensor: indistinguishable by name,
                # size and shape (num_classes == 4 * reg_max).  Report it rather
                # than popping the same key twice.
                raise RuntimeError(
                    "Model inference failed: the YOLO-Pro cls and reg heads for "
                    f"level p{level} both resolved to '{cls_key}' — they are "
                    f"indistinguishable at num_classes={self.num_classes} / "
                    f"reg_max={self.reg_max}. Available outputs: "
                    f"{self._describe_outputs(arrays)}."
                )
            if cls_key is None or reg_key is None:
                # A missing level silently produced a partial ``resolved`` dict,
                # and decode_raw_outputs_np then read whichever levels happened
                # to survive — wrong boxes rather than an error.  Report the
                # unresolved graph instead.
                missing = [
                    n for n, k in (("cls", cls_key), ("reg", reg_key)) if k is None
                ]
                raise RuntimeError(
                    "Model inference failed: could not resolve YOLO-Pro head(s) "
                    f"{missing} for level p{level} (stride {stride}, expected "
                    f"cls size {exp_cls} / reg size {exp_reg}). Available "
                    f"outputs: {self._describe_outputs(arrays)}."
                )

            # decode_raw_outputs_np indexes these as (B, Hg, Wg, C); TFLite may
            # hand back a flattened or differently-batched view of the same
            # data.  pop() retires each tensor so a later level cannot reuse it.
            cls_arr = named.pop(cls_key)
            reg_arr = named.pop(reg_key)
            if cls_arr.size != exp_cls or reg_arr.size != exp_reg:
                raise RuntimeError(
                    "Model inference failed: YOLO-Pro head resolution for level "
                    f"p{level} (stride {stride}) picked tensors of the wrong "
                    f"size (cls {cls_arr.size} vs expected {exp_cls}, reg "
                    f"{reg_arr.size} vs expected {exp_reg}). Available outputs: "
                    f"{self._describe_outputs(arrays)}."
                )
            resolved[f"cls_p{level}"] = cls_arr.reshape(1, gh, gw, self.num_classes)
            resolved[f"reg_p{level}"] = reg_arr.reshape(1, gh, gw, 4 * self.reg_max)
        return resolved

    # -- keras-compatible call ---------------------------------------------
    def __call__(self, x, training: bool = False):
        """Run one forward pass.  ``training`` is accepted and ignored."""
        self._interp.set_tensor(self._in_detail["index"], self._prepare_input(x))
        self._interp.invoke()
        return self._name_outputs(self._read_outputs())

    def predict(self, x, *args, **kwargs):
        """Batch predict, matching ``keras.Model.predict``'s array-in/array-out.

        The classification and FOMO evals call ``model.predict(X)`` over the
        whole validation array; a TFLite graph has a fixed batch-1 signature,
        so iterate and stack.  Extra keras args (batch_size, verbose, …) are
        accepted and ignored.
        """
        arr = np.asarray(x, dtype=np.float32)
        rows = [self(arr[i:i + 1]) for i in range(len(arr))]
        if not rows:
            return np.empty((0,), dtype=np.float32)

        # Single-output graphs (classification logits, FOMO heatmap) stack into
        # one array; multi-output graphs keep their dict shape per key.
        #
        # "Single output" is a property of the GRAPH, not of the key count:
        # several keys can alias the same tensor.  Keying off ``len(first) == 1``
        # alone let a one-output FOMO graph come back as a dict, which the
        # caller's ``np.asarray(..., np.float32)`` cannot convert.
        first = rows[0]
        if isinstance(first, dict):
            keys = self._distinct_output_keys(first)
            if len(keys) == 1:
                key = keys[0]
                return np.concatenate([r[key] for r in rows], axis=0)
            return {k: np.concatenate([r[k] for r in rows], axis=0) for k in first}
        return np.concatenate(rows, axis=0)

    @staticmethod
    def _distinct_output_keys(named: Dict[str, np.ndarray]) -> List[str]:
        """One key per distinct output tensor, alias keys dropped.

        Deduped by object identity: ``_name_outputs`` maps aliases onto the very
        same ndarray, while two genuinely different heads (YOLO-Pro's
        cls_p*/reg_p*, SSD's box/cls) are always separate objects even when they
        happen to share a shape — so this never collapses a real multi-output
        graph.
        """
        seen: set = set()
        keys: List[str] = []
        for key, value in named.items():
            marker = id(value)
            if marker in seen:
                continue
            seen.add(marker)
            keys.append(key)
        return keys

    def get_weights(self):  # pragma: no cover - defensive parity shim
        raise NotImplementedError("TFLiteVariantRunner wraps a frozen graph")


# ─────────────────────────────────────────────────────────────────────────────
# Orchestration
# ─────────────────────────────────────────────────────────────────────────────

def _extract_metrics(report: dict) -> dict:
    """Lift the metric fields out of an eval report into a stored record."""
    out = {k: report.get(k) for k in _METRIC_FIELDS if k in report}
    # The detection evals report their own status under an arch-specific key;
    # normalise it so consumers do not need to know which worker produced it.
    eval_status = (
        report.get("yolo_pro_eval_status")
        or report.get("fomo_eval_status")
        or report.get("eval_status")
    )
    eval_error = (
        report.get("yolo_pro_eval_error")
        or report.get("fomo_eval_error")
        or report.get("eval_error")
    )
    _has_numbers = any(
        out.get(k) is not None for k in ("map50", "accuracy", "f1_score", "precision")
    )
    if eval_status == "success" or (eval_status is None and _has_numbers):
        out["status"] = "success"
    else:
        out["status"] = eval_status or "not_run"
    out["error"] = eval_error
    return out


def evaluate_exported_variants(
    *,
    db,
    job_id,
    scorer: Callable[[TFLiteVariantRunner], dict],
    variant_bytes: Dict[str, Optional[bytes]],
    runner_kwargs: Optional[dict] = None,
    export_errors: Optional[Dict[str, str]] = None,
    skip_reasons: Optional[Dict[str, str]] = None,
    log_prefix: str = "",
) -> Dict[str, dict]:
    """Score every exported TFLite variant and persist per-variant metrics.

    Args:
        db            : active SQLAlchemy session (rows are committed here).
        job_id        : TrainingJob id whose TrainedModel rows are updated.
        scorer        : the worker's OWN eval function, pre-bound to its
                        validation set — called as ``scorer(runner)``.  Must
                        return the eval report dict.
        variant_bytes : {variant_name: tflite_bytes | None}.  ``None`` means the
                        export failed and the variant is recorded unavailable.
        runner_kwargs : forwarded to TFLiteVariantRunner (output_type,
                        num_classes, reg_max, input_h, input_w).
        export_errors : {variant_name: reason} for variants that failed export,
                        so the UI can say *why* a variant has no metrics.
        skip_reasons  : {variant_name: reason} for variants that exported fine
                        but this architecture's eval cannot score (e.g. the SSD
                        ``decoded_float32`` export, whose graph has already
                        applied decode + sigmoid + NMS, so feeding it to an eval
                        that decodes raw heads would double-decode).  Recorded
                        as ``status="not_scored"`` with the reason, so the panel
                        states the real cause instead of guessing "trained
                        before per-variant eval existed".
        log_prefix    : caller's log tag, typically f"[{job_id}] ".

    Returns:
        {variant_name: stored metrics record}.  Always returns; never raises.
    """
    from app.models.user import TrainedModel

    runner_kwargs = runner_kwargs or {}
    export_errors = export_errors or {}
    skip_reasons = skip_reasons or {}
    results: Dict[str, dict] = {}

    for variant, blob in variant_bytes.items():
        if blob and variant in skip_reasons:
            # Exported successfully, but not scoreable by this eval.  Recorded
            # explicitly so the variant never silently inherits another
            # variant's numbers and never claims a wrong reason.
            results[variant] = {
                "status": "not_scored",
                "error": skip_reasons[variant],
            }
            continue

        if not blob:
            results[variant] = {
                "status": "export_failed",
                "error": export_errors.get(variant) or "Export did not produce an artifact.",
            }
            continue

        try:
            runner = TFLiteVariantRunner(blob, **runner_kwargs)
            report = scorer(runner)
            record = _extract_metrics(report if isinstance(report, dict) else {})
            logger.info(
                f"{log_prefix}variant eval {variant}: status={record.get('status')} "
                f"map50={record.get('map50')} precision={record.get('precision')}"
            )
        except Exception as exc:
            # Per-variant isolation: a variant that cannot be scored is recorded
            # as failed and the remaining variants continue.
            record = {"status": "failed", "error": str(exc)}
            logger.warning(
                f"{log_prefix}variant eval {variant} failed: {exc}", exc_info=True
            )
        results[variant] = record

    # Stamp each record onto its own TrainedModel row.  Same backfill shape the
    # workers already use for threshold/ranking_status.
    try:
        # The workers ``db.add()`` each variant's row during export and leave it
        # pending — and ``SessionLocal`` is built with ``autoflush=False``, so a
        # query alone would NOT see those rows and would silently match nothing.
        # Flush first so the pending INSERTs are visible to this query;
        # otherwise every metric computed above is dropped on the floor and the
        # UI shows "not scored" for a run whose eval actually succeeded.
        db.flush()
        rows = (
            db.query(TrainedModel)
            .filter(TrainedModel.training_job_id == job_id)
            .all()
        )
        if not rows:
            logger.warning(
                f"{log_prefix}per-variant metrics computed but no TrainedModel "
                f"rows found for job {job_id} — nothing persisted."
            )
        for row in rows:
            meta = dict(row.model_metadata or {})
            record = results.get(meta.get("variant"))
            if record is None or row.format != "tflite":
                continue
            meta[VARIANT_METRICS_KEY] = _sanitize(record)
            row.model_metadata = meta
        db.commit()
    except Exception as exc:
        logger.warning(
            f"{log_prefix}failed to persist per-variant metrics: {exc}", exc_info=True
        )
        try:
            db.rollback()
        except Exception:
            pass

    return results
