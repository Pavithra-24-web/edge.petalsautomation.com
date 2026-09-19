"""
yolo_pro/_debug_probe.py
────────────────────────
DEBUG INSTRUMENTATION ONLY — changes no training behaviour.

A tensor-finiteness probe used to diff a working 224x224 run against a
failing 640x640 run stage by stage.  Every call site is a no-op unless
``YOLO_PRO_NAN_PROBE=1`` is set in the environment, so the production
training path is byte-identical when the flag is off.

Usage
─────
    from app.ml.yolo_pro._debug_probe import probe, set_ctx, ENABLED

    set_ctx(res=640, epoch=0, step=0, batch=0)
    probe("MODEL_OUT", "cls_p3", cls_pred)

Output — one line per tensor, greppable:

    PROBE res=640 ep=1 st=0 bi=0 | 02_MODEL_OUT | cls_p3 shape=(4,6400,1)
        dtype=float32 n=25600 nonfinite=0 nan=0 inf=0 min=... max=...
        mean=... std=... absmax=... | yolo_pro_worker.py:12300

The first tensor observed to be non-finite is latched and re-printed with a
*** FIRST NON-FINITE *** banner naming the producing stage and caller line.
"""
from __future__ import annotations

import os
import sys
import inspect

import numpy as np

try:                                  # tf is always present in the worker
    import tensorflow as tf
except Exception:                     # pragma: no cover — probe must never break import
    tf = None  # type: ignore

ENABLED: bool = os.environ.get("YOLO_PRO_NAN_PROBE", "0") == "1"
# Optional second sink; stderr always receives the line.
_SINK_PATH: str = os.environ.get("YOLO_PRO_NAN_PROBE_FILE", "")
_sink = None
if ENABLED and _SINK_PATH:
    _sink = open(_SINK_PATH, "a", buffering=1, encoding="utf-8")

_CTX = {"res": "?", "epoch": 0, "step": 0, "batch": 0, "tag": "-"}
_FIRST_BAD = {"hit": False}


def set_ctx(res=None, epoch=None, step=None, batch=None, tag=None) -> None:
    """Record the resolution / epoch / step / batch / tag stamped on every line."""
    if not ENABLED:
        return
    if tag is not None:
        _CTX["tag"] = tag
    if res is not None:
        _CTX["res"] = res
    if epoch is not None:
        _CTX["epoch"] = epoch
    if step is not None:
        _CTX["step"] = step
    if batch is not None:
        _CTX["batch"] = batch


def _hdr() -> str:
    return (
        f"PROBE res={_CTX['res']} ep={_CTX['epoch']} "
        f"st={_CTX['step']} bi={_CTX['batch']} tag={_CTX['tag']}"
    )


def _emit(line: str) -> None:
    print(line, file=sys.stderr, flush=True)
    if _sink is not None:
        _sink.write(line + "\n")


def _caller(depth: int = 2) -> str:
    try:
        fr = inspect.stack()[depth]
        return f"{os.path.basename(fr.filename)}:{fr.lineno}"
    except Exception:
        return "?:?"


def _to_np(x):
    """Best-effort conversion to numpy; None when the value is symbolic."""
    if x is None:
        return None
    if isinstance(x, np.ndarray):
        return x
    if isinstance(x, (int, float, bool, np.number)):
        return np.asarray(x)
    if tf is not None and isinstance(x, tf.Tensor):
        if not tf.executing_eagerly():
            return None            # symbolic — handled by the graph branch
        try:
            return x.numpy()
        except Exception:
            return None
    if hasattr(x, "numpy"):
        try:
            return x.numpy()
        except Exception:
            return None
    try:
        return np.asarray(x)
    except Exception:
        return None


def probe(stage: str, name: str, x, note: str = "") -> None:
    """
    Log shape / dtype / min / max / mean / std / finiteness for one tensor.

    Eager tensors and numpy arrays are summarised exactly.  Symbolic tensors
    (inside a ``tf.function``) fall back to ``tf.print`` so the same stages stay
    observable when the step is graph-compiled.
    """
    if not ENABLED:
        return
    src = _caller()

    a = _to_np(x)
    if a is None:
        # ── graph-mode branch ────────────────────────────────────────────────
        if tf is None or not isinstance(x, tf.Tensor):
            _emit(f"{_hdr()} | {stage:<16} | {name} <unmaterialisable {type(x)}> | {src}")
            return
        xf = tf.cast(x, tf.float32)
        fin = tf.math.is_finite(xf)
        safe = tf.where(fin, xf, tf.zeros_like(xf))
        tf.print(
            _hdr(), "|", stage, "|", name,
            "shape=", tf.shape(xf),
            "dtype=", x.dtype.name,
            "nonfinite=", tf.reduce_sum(tf.cast(tf.logical_not(fin), tf.int64)),
            "nan=", tf.reduce_sum(tf.cast(tf.math.is_nan(xf), tf.int64)),
            "min=", tf.reduce_min(safe),
            "max=", tf.reduce_max(safe),
            "mean=", tf.reduce_mean(safe),
            note, "|", src,
            output_stream=sys.stderr,
            summarize=8,
        )
        return

    # ── eager / numpy branch ────────────────────────────────────────────────
    if a.dtype == np.bool_:
        n = a.size
        frac = float(a.mean()) if n else 0.0
        _emit(
            f"{_hdr()} | {stage:<16} | {name} shape={tuple(a.shape)} dtype=bool "
            f"n={n} true={int(a.sum())} frac={frac:.6g} {note} | {src}"
        )
        return

    af = a.astype(np.float64, copy=False)
    fin = np.isfinite(af)
    n_nan = int(np.isnan(af).sum())
    n_inf = int(np.isinf(af).sum())
    n_bad = n_nan + n_inf
    good = af[fin]
    if good.size:
        mn, mx = float(good.min()), float(good.max())
        me, sd = float(good.mean()), float(good.std())
        amx = float(np.abs(good).max())
    else:
        mn = mx = me = sd = amx = float("nan")

    flag = "  <<< NON-FINITE" if n_bad else ""
    _emit(
        f"{_hdr()} | {stage:<16} | {name} shape={tuple(a.shape)} "
        f"dtype={a.dtype} n={a.size} nonfinite={n_bad} nan={n_nan} inf={n_inf} "
        f"min={mn:.6g} max={mx:.6g} mean={me:.6g} std={sd:.6g} absmax={amx:.6g} "
        f"{note} | {src}{flag}"
    )

    if n_bad and not _FIRST_BAD["hit"]:
        _FIRST_BAD["hit"] = True
        _emit(
            f"{_hdr()} | *** FIRST NON-FINITE *** stage={stage} tensor={name} "
            f"nan={n_nan} inf={n_inf} of {a.size} | produced at {src}"
        )


def _is_symbolic(x) -> bool:
    """
    True for a graph-mode (non-eager) tensor-like value.

    Covers tf.Tensor, tf.Variable and Keras 3 variables alike: inside a
    ``tf.function`` none of them can be read into numpy, and coercing them
    anyway produces a bogus NaN.
    """
    if tf is None or x is None:
        return False
    if tf.executing_eagerly():
        return False
    return isinstance(x, (tf.Tensor, tf.Variable)) or (
        hasattr(x, "dtype") and hasattr(x, "shape") and not isinstance(x, np.ndarray)
    )


def probe_scalar(stage: str, name: str, value, note: str = "") -> None:
    """
    Log a plain python/numpy scalar (counts, ratios, norms).

    Symbolic (graph-mode) values are routed to ``tf.print`` — coercing them with
    ``float()`` would raise and be reported as a bogus NaN.
    """
    if not ENABLED:
        return
    src = _caller()
    if _is_symbolic(value):
        tf.print(
            _hdr(), "|", stage, "|", f"{name}=", tf.cast(value, tf.float32),
            note, "| (graph)", src,
            output_stream=sys.stderr,
        )
        return
    try:
        v = float(value)
        ok = bool(np.isfinite(v))
    except Exception:
        _emit(f"{_hdr()} | {stage:<16} | {name}=<unreadable {type(value).__name__}> "
              f"{note} | {src}")
        return
    flag = "" if ok else "  <<< NON-FINITE"
    _emit(f"{_hdr()} | {stage:<16} | {name}={v:.10g} {note} | {src}{flag}")


def probe_grads(stage: str, grads, variables, note: str = "") -> None:
    """
    Summarise a gradient list: finite global norm, per-variable non-finite
    counts, and the FIRST variable (in ``trainable_variables`` order) whose
    gradient is non-finite.
    """
    if not ENABLED:
        return
    src = _caller()
    _live = [g for g in grads if g is not None]
    if _live and _is_symbolic(_live[0]):
        # Graph mode: per-variable numpy stats are unavailable, so emit the two
        # aggregates that still answer "are the gradients finite, and how big".
        _flat = tf.concat([tf.reshape(tf.cast(g, tf.float32), [-1]) for g in _live],
                          axis=0)
        _fin = tf.math.is_finite(_flat)
        tf.print(
            _hdr(), "|", stage, "| grads(graph) vars=", len(_live),
            "nonfinite=", tf.reduce_sum(tf.cast(tf.logical_not(_fin), tf.int64)),
            "nan=", tf.reduce_sum(tf.cast(tf.math.is_nan(_flat), tf.int64)),
            "global_norm=", tf.norm(tf.where(_fin, _flat, tf.zeros_like(_flat))),
            note, "|", src,
            output_stream=sys.stderr,
        )
        return
    total = 0
    bad_vars = []
    sq = 0.0
    first_bad = None
    biggest = (0.0, "-")
    for g, v in zip(grads, variables):
        if g is None:
            continue
        a = _to_np(g)
        if a is None:
            continue
        total += 1
        af = a.astype(np.float64, copy=False)
        fin = np.isfinite(af)
        nb = int((~fin).sum())
        good = af[fin]
        gn = float(np.sqrt((good ** 2).sum())) if good.size else 0.0
        if gn > biggest[0]:
            biggest = (gn, v.name)
        sq += gn ** 2
        if nb:
            bad_vars.append((v.name, nb, a.size))
            if first_bad is None:
                first_bad = (v.name, nb, a.size, tuple(a.shape))
    gnorm = float(np.sqrt(sq))
    _emit(
        f"{_hdr()} | {stage:<16} | grads vars={total} bad_vars={len(bad_vars)} "
        f"finite_global_norm={gnorm:.6g} max_var_norm={biggest[0]:.6g}({biggest[1]}) "
        f"{note} | {src}"
    )
    if first_bad:
        _emit(
            f"{_hdr()} | {stage:<16} | FIRST BAD GRAD var={first_bad[0]} "
            f"nonfinite={first_bad[1]}/{first_bad[2]} shape={first_bad[3]} | {src}"
        )
        for nm, nb, sz in bad_vars[:12]:
            _emit(f"{_hdr()} | {stage:<16} |   bad_grad {nm} {nb}/{sz} | {src}")
        if len(bad_vars) > 12:
            _emit(f"{_hdr()} | {stage:<16} |   ... {len(bad_vars) - 12} more | {src}")
    if bad_vars and not _FIRST_BAD["hit"]:
        _FIRST_BAD["hit"] = True
        _emit(
            f"{_hdr()} | *** FIRST NON-FINITE *** stage={stage} "
            f"tensor=grad[{bad_vars[0][0]}] | produced at {src}"
        )


def probe_weights(stage: str, variables, note: str = "") -> None:
    """Summarise every model weight (e.g. after an optimizer update)."""
    if not ENABLED:
        return
    src = _caller()
    if variables and _is_symbolic(variables[0]):
        _flat = tf.concat([tf.reshape(tf.cast(v, tf.float32), [-1]) for v in variables],
                          axis=0)
        _fin = tf.math.is_finite(_flat)
        tf.print(
            _hdr(), "|", stage, "| weights(graph) vars=", len(variables),
            "nonfinite=", tf.reduce_sum(tf.cast(tf.logical_not(_fin), tf.int64)),
            "l2=", tf.norm(tf.where(_fin, _flat, tf.zeros_like(_flat))),
            "absmax=", tf.reduce_max(tf.abs(tf.where(_fin, _flat, tf.zeros_like(_flat)))),
            note, "|", src,
            output_stream=sys.stderr,
        )
        return
    bad = []
    sq = 0.0
    amax = 0.0
    for v in variables:
        a = _to_np(v)
        if a is None:
            continue
        af = a.astype(np.float64, copy=False)
        fin = np.isfinite(af)
        nb = int((~fin).sum())
        good = af[fin]
        if good.size:
            sq += float((good ** 2).sum())
            amax = max(amax, float(np.abs(good).max()))
        if nb:
            bad.append((v.name, nb, a.size))
    _emit(
        f"{_hdr()} | {stage:<16} | weights vars={len(variables)} bad_vars={len(bad)} "
        f"l2={np.sqrt(sq):.6g} absmax={amax:.6g} {note} | {src}"
    )
    for nm, nb, sz in bad[:12]:
        _emit(f"{_hdr()} | {stage:<16} |   bad_weight {nm} {nb}/{sz} | {src}")
    if bad and not _FIRST_BAD["hit"]:
        _FIRST_BAD["hit"] = True
        _emit(
            f"{_hdr()} | *** FIRST NON-FINITE *** stage={stage} "
            f"tensor=weight[{bad[0][0]}] | produced at {src}"
        )


def reset_first() -> None:
    """Clear the first-non-finite latch (between runs)."""
    _FIRST_BAD["hit"] = False


def set_tag(tag: str) -> None:
    """Stamp subsequent probe lines with a path/scale tag (e.g. "o2m.p3")."""
    if not ENABLED:
        return
    _CTX["tag"] = tag
