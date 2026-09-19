"""
Training-output UI log filter.

The YOLO-Pro worker emits a rich stream of diagnostic lines into
``training_history.log_lines`` for backend audit purposes.  Many of those
lines reference internal fields (probe vs full safety eval, runtime
preds-per-image, tp/fp gap, ranking-guard rejections, etc.) that are
confusing in the user-facing Training output panel.

``format_training_ui_log_line`` is applied at serialization time, just
before ``log_lines`` is handed to the frontend.  It returns:

  * ``None``                 — hide this line from the UI entirely
  * a transformed string     — for checkpoint events, show a concise
                               user-facing replacement
  * the original line        — for normal user-relevant progress lines

The backend ``logger.*`` calls and the raw ``training_history.log_lines``
stored in the database are untouched by this filter — only the UI view
is simplified.
"""
from __future__ import annotations

import re
from typing import Optional, List


# ── Phrases that mean "internal diagnostic, never show to the user" ──────────
_HIDE_SUBSTRINGS: tuple[str, ...] = (
    "probe(",
    "full(",
    "probe/full safety divergence",
    "safety stability rescue skipped",
    "checkpoint blocked by full export-safety eval",
    "runtime preds-per-image flood metric",
    "eval img ",
    "best_any_cls_iou",
    # Internal field tokens — any line that mentions these is by
    # construction a diagnostic dump, not user-facing progress.
    "prev_safety_map50",
    "new_safety_map50",
    "current_safety_map50",
    "best_saved_safety_map50",
    "tp_fp_gap",
    "runtime_preds_per_image",
    "runtime_ppi",
    "trigger_reasons",
)


# ── Checkpoint admission lines ──────────────────────────────────────────────
# Examples (the leading "[JOB_ID] " prefix is optional):
#   [job123] checkpoint admitted EP4 reason=monitor  probe(...)  full(...)
#   [job123] ★ckpt-rescue EP17  prev_safety_map50=...  new_safety_map50=...
#   [job123] ★ckpt-stability safety stability rescue checkpoint at epoch 92 ...
_RE_CKPT_ADMITTED = re.compile(
    r"checkpoint admitted EP(?P<epoch>\d+)\s+reason=(?P<reason>\w+)",
    re.IGNORECASE,
)
_RE_CKPT_RESCUE = re.compile(
    r"★?ckpt-rescue\s+EP(?P<epoch>\d+)",
    re.IGNORECASE,
)
_RE_CKPT_STABILITY = re.compile(
    r"★?ckpt-stability.*?(?:epoch|EP)\s*(?P<epoch>\d+)",
    re.IGNORECASE,
)


def _format_checkpoint_line(raw: str) -> Optional[str]:
    """
    Recognize the three checkpoint-save markers emitted by the worker and
    rewrite them into a single concise user-facing line.  Returns ``None``
    if the line is not a checkpoint marker.
    """
    # Rescue / stability markers must be checked BEFORE the generic
    # "checkpoint admitted" pattern, because a future combined line might
    # match both.  Today they are written as separate lines, but the
    # ordering is the defensible choice either way.
    m = _RE_CKPT_RESCUE.search(raw)
    if m:
        return (
            f"Checkpoint saved at epoch {int(m.group('epoch'))} "
            f"— detector quality improved."
        )

    m = _RE_CKPT_STABILITY.search(raw)
    if m:
        return (
            f"Checkpoint saved at epoch {int(m.group('epoch'))} "
            f"— detector quality is stable."
        )

    m = _RE_CKPT_ADMITTED.search(raw)
    if m:
        # The "checkpoint admitted" line covers ALL save reasons (monitor,
        # safety_quality_rescue, safety_stability_rescue).  When a rescue
        # was the reason, the dedicated ★ckpt-rescue / ★ckpt-stability
        # line that follows will produce the richer user-facing wording,
        # so here we only emit the plain monitor-style message — the
        # later rescue line will (correctly) replace it visually as the
        # operator scans down the log.
        reason = m.group("reason").lower()
        epoch = int(m.group("epoch"))
        if reason == "safety_quality_rescue":
            return f"Checkpoint saved at epoch {epoch} — detector quality improved."
        if reason == "safety_stability_rescue":
            return f"Checkpoint saved at epoch {epoch} — detector quality is stable."
        return f"Checkpoint saved at epoch {epoch}."

    return None


def format_training_ui_log_line(raw_line: object) -> Optional[str]:
    """
    Transform a single ``_yolo_log_lines`` entry for the Training output
    UI.  See module docstring for the contract.
    """
    if raw_line is None:
        return None
    if not isinstance(raw_line, str):
        # Non-string entries are never legitimate UI content — hide them
        # rather than stringifying.  Stringifying is what turned a stray
        # empty container in a stored run into a literal "[]" line.
        return None
    if raw_line == "":
        return None

    # Checkpoint events: rewrite to a concise message.  Done first so the
    # blanket "internal-field" hide rules below (e.g. ``runtime_ppi`` in
    # the admitted line, ``tp_fp_gap`` in the rescue line) do not strip
    # the event entirely.
    rewritten = _format_checkpoint_line(raw_line)
    if rewritten is not None:
        return rewritten

    for needle in _HIDE_SUBSTRINGS:
        if needle in raw_line:
            return None

    return raw_line


def filter_training_ui_log_lines(raw_lines: object) -> List[str]:
    """
    Convenience wrapper for serializers: apply
    ``format_training_ui_log_line`` to an iterable, drop ``None``s, and
    return a fresh list of UI-ready strings.  Returns ``[]`` for any
    non-iterable or missing input.
    """
    if not raw_lines:
        return []
    try:
        iterator = iter(raw_lines)
    except TypeError:
        return []
    out: List[str] = []
    for raw in iterator:
        formatted = format_training_ui_log_line(raw)
        if formatted is not None:
            if out and out[-1] == formatted:
                continue
            out.append(formatted)
    return out
