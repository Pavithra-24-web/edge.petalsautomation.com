"""
Optimization recommendations — Target Device Phase 7
(docs/target_device_phase7.md, docs/target_device_phase7_prompt.md).

Answers what Phase 5's estimate leaves open: when a model does not fit its
target device, what should change? Pure and rule-based — reads an already
computed `estimation.EstimationResult` plus the model row (for variant
availability) and returns advice. It never recomputes a metric, touches the
database, or applies anything: training, export, and the catalog are
untouched by this module.

The rule table below (docs/target_device_phase7_prompt.md §"Rule table") is
the single source of truth; where code and table disagree, the table wins.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from app.services.estimation import EstimationResult, MetricEstimate

# ── Constraints ──────────────────────────────────────────────────────────────
CONSTRAINT_FLASH = "flash"
CONSTRAINT_RAM = "ram"
CONSTRAINT_LATENCY = "latency"
_CONSTRAINT_ORDER = {CONSTRAINT_FLASH: 0, CONSTRAINT_RAM: 1, CONSTRAINT_LATENCY: 2}

# ── Severity ─────────────────────────────────────────────────────────────────
SEVERITY_OVER_BUDGET = "over_budget"
SEVERITY_MARGINAL = "marginal"
_SEVERITY_ORDER = {SEVERITY_OVER_BUDGET: 0, SEVERITY_MARGINAL: 1}

# A fit is "marginal" when it fits but leaves less than this fraction of
# budget as margin. One constant, shared by flash/RAM/latency, per the table.
MARGINAL_MARGIN_RATIO = 0.20

# ── Recommendation codes ─────────────────────────────────────────────────────
# Stable and machine-readable — the UI groups advice on these, so the same
# underlying action (e.g. quantizing) keeps the same code across constraints.
CODE_SWITCH_TO_INT8_VARIANT = "switch_to_int8_variant"
CODE_QUANTIZE_INT8 = "quantize_int8"
CODE_REDUCE_INPUT_SIZE = "reduce_input_size"
CODE_LIGHTER_ARCHITECTURE = "lighter_architecture"

# ── "Nothing applies" reasons ────────────────────────────────────────────────
REASON_FITS_COMFORTABLY = "All estimated metrics fit comfortably within budget — no optimization needed."
REASON_CANNOT_ASSESS = "One or more metrics cannot be estimated yet, so no recommendation can be made from them."
REASON_INCOMPATIBLE = "This model is not compatible with the selected format/device — see the compatibility message above."


@dataclass(frozen=True)
class Recommendation:
    """One piece of advice. Mirrors `estimation.MetricEstimate`'s discipline:
    every field is always present, never a bare string the caller has to
    parse."""

    constraint: str  # flash | ram | latency
    severity: str  # over_budget | marginal
    code: str
    title: str
    detail: str
    expected_gain: str

    def to_dict(self) -> dict:
        return {
            "constraint": self.constraint,
            "severity": self.severity,
            "code": self.code,
            "title": self.title,
            "detail": self.detail,
            "expected_gain": self.expected_gain,
        }


def _model_metadata(model) -> dict:
    """Mirrors estimation._model_metadata / compatibility._model_metadata:
    accept a TrainedModel row or the model_metadata dict itself."""
    if model is None:
        return {}
    if isinstance(model, dict):
        return model
    return getattr(model, "model_metadata", None) or {}


def _has_int8_sibling(model) -> bool:
    """True when a quantized tflite variant of `model` already exists in the
    same training job — the fact that turns "retrain" into "switch" for flash
    advice. Reads the relationship already on the row (same pattern as
    `estimation._resolve_project`); no fresh query is issued."""
    if model is None or isinstance(model, dict):
        return False
    training_job = getattr(model, "training_job", None)
    siblings = getattr(training_job, "trained_models", None) if training_job else None
    if not siblings:
        return False
    return any(
        s.id != getattr(model, "id", None)
        and s.format == "tflite"
        and bool((s.model_metadata or {}).get("quantized", False))
        for s in siblings
    )


def _severity(metric: "MetricEstimate") -> Optional[str]:
    """over_budget | marginal | None (unknown, or comfortably fits).

    `fits is None` (unknown) and a comfortable fit both yield no severity —
    callers tell the two apart via `metric.fits` when composing the empty-list
    reason. When `fits` is not None, `budget`/`margin` are guaranteed set
    (estimation.py only produces `fits` alongside a resolved budget).
    """
    if metric.fits is None:
        return None
    if not metric.fits:
        return SEVERITY_OVER_BUDGET
    if metric.budget and metric.margin is not None and metric.margin < MARGINAL_MARGIN_RATIO * metric.budget:
        return SEVERITY_MARGINAL
    return None


def _format_bytes(n: float) -> str:
    n = abs(n)
    if n >= 1024 * 1024:
        return f"{n / (1024 * 1024):.2f} MB"
    if n >= 1024:
        return f"{n / 1024:.1f} KB"
    return f"{n:.0f} bytes"


def _format_ms(n: float) -> str:
    return f"{abs(n):.2f} ms"


def _gap_phrase(metric: "MetricEstimate", formatter) -> str:
    """'128.0 KB over budget' when over, 'only 40.0 KB of headroom' when
    marginal — the actual gap a recommendation's detail/gain quotes."""
    if metric.margin is None:
        return ""
    if metric.margin < 0:
        return f"{formatter(metric.margin)} over budget"
    return f"only {formatter(metric.margin)} of headroom"


def _flash_recommendations(flash: "MetricEstimate", quantized: bool, has_int8_sibling: bool) -> list:
    severity = _severity(flash)
    if severity is None:
        return []

    gap = _gap_phrase(flash, _format_bytes)

    if not quantized:
        if has_int8_sibling:
            return [Recommendation(
                constraint=CONSTRAINT_FLASH, severity=severity,
                code=CODE_SWITCH_TO_INT8_VARIANT,
                title="Switch to the int8 variant",
                detail=(
                    f"Flash usage is {gap}. This model ships as float32, but "
                    "an int8-quantized variant of the same model already "
                    "exists and needs no retraining."
                ),
                expected_gain="~4x smaller weights",
            )]
        return [Recommendation(
            constraint=CONSTRAINT_FLASH, severity=severity,
            code=CODE_QUANTIZE_INT8,
            title="Retrain with int8 quantization",
            detail=(
                f"Flash usage is {gap}. This model ships as float32; "
                "quantizing to int8 roughly quarters its weight size."
            ),
            expected_gain="~4x smaller weights",
        )]

    return [Recommendation(
        constraint=CONSTRAINT_FLASH, severity=severity,
        code=CODE_LIGHTER_ARCHITECTURE,
        title="Move to a lighter architecture class",
        detail=(
            f"Flash usage is {gap}. This model is already int8-quantized, so "
            "flash usage can only come down with fewer or smaller weights."
        ),
        expected_gain=gap,
    )]


def _ram_recommendations(ram: "MetricEstimate", quantized: bool, meta: dict) -> list:
    severity = _severity(ram)
    if severity is None:
        return []

    gap = _gap_phrase(ram, _format_bytes)
    input_shape = meta.get("input_shape")

    recs = [Recommendation(
        constraint=CONSTRAINT_RAM, severity=severity,
        code=CODE_REDUCE_INPUT_SIZE,
        title="Reduce the input size",
        detail=(
            f"Estimated peak RAM is {gap}. RAM scales with input area, so a "
            "smaller input directly reduces it."
        ),
        expected_gain=f"current input shape {input_shape}",
    )]

    if not quantized:
        recs.append(Recommendation(
            constraint=CONSTRAINT_RAM, severity=severity,
            code=CODE_QUANTIZE_INT8,
            title="Quantize to int8",
            detail=(
                f"Estimated peak RAM is {gap}. This model ships as float32; "
                "quantizing to int8 halves the per-element size the RAM "
                "estimate is built from."
            ),
            expected_gain="4 bytes/element -> 1",
        ))

    return recs


def _latency_recommendations(latency: "MetricEstimate") -> list:
    severity = _severity(latency)
    if severity is None:
        return []

    gap = _gap_phrase(latency, _format_ms)
    detail_base = f"Estimated latency is {gap}."

    return [
        Recommendation(
            constraint=CONSTRAINT_LATENCY, severity=severity,
            code=CODE_REDUCE_INPUT_SIZE,
            title="Reduce the input size",
            detail=f"{detail_base} Latency scales with MAC count; a smaller input reduces MACs directly.",
            expected_gain=gap,
        ),
        Recommendation(
            constraint=CONSTRAINT_LATENCY, severity=severity,
            code=CODE_LIGHTER_ARCHITECTURE,
            title="Move to a lighter architecture class",
            detail=f"{detail_base} Fewer or cheaper layers reduce MACs directly.",
            expected_gain=gap,
        ),
    ]


def recommend(result: "EstimationResult", model) -> "tuple[list, Optional[str]]":
    """The ordered advice list for `result`, plus the reason when that list is
    empty (never both — an empty list always carries a reason, per the
    Phase 7 spec's "no recommendation applies is a real answer").

    Order: severity (over_budget before marginal), then flash -> RAM ->
    latency, matching the rule table.
    """
    if not result.compatible:
        return [], REASON_INCOMPATIBLE

    meta = _model_metadata(model)
    quantized = bool(meta.get("quantized", False))
    has_int8_sibling = _has_int8_sibling(model)

    recs: list = []
    recs += _flash_recommendations(result.flash, quantized, has_int8_sibling)
    recs += _ram_recommendations(result.ram, quantized, meta)
    recs += _latency_recommendations(result.latency)

    if recs:
        recs.sort(key=lambda r: (_SEVERITY_ORDER[r.severity], _CONSTRAINT_ORDER[r.constraint]))
        return recs, None

    metrics = (result.flash, result.ram, result.latency)
    if any(m.fits is None for m in metrics):
        return [], REASON_CANNOT_ASSESS
    return [], REASON_FITS_COMFORTABLY
