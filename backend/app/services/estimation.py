"""
Performance estimation — Target Device Phase 5 (docs/target_device_phase5.md).

Answers "does this model fit, and how fast will it run?" for a (trained
model, build format, device) triple. Pure functions only: no writes, no HTTP.
Reads the database (via an optional `db` session) the same way
`app.services.compatibility` does, to resolve a device's catalog/specification
row and a project's budget overrides.

Every metric is a `MetricEstimate`, never a bare number — the same discipline
`compatibility.CompatibilityResult` uses: `value=None` always pairs with a
`reason_code` that explains why, and every value carries `source` so a caller
can never mistake an estimate for a measurement.

── Latency's missing input ──────────────────────────────────────────────────
Latency needs an operation count (MACs) and nothing in this codebase produces
one today: `model_metadata` records shapes and flags, never op counts (see
`training_worker._common_meta`). This phase does not add a TFLite op-count
walker — a hand-rolled MAC counter over an arbitrary graph (conv, depthwise,
dense, detection heads, each with their own shape arithmetic) is easy to get
subtly wrong, and a confidently-wrong "estimated" figure is worse than an
honest "unavailable" one for a number users will size hardware against.
Latency reads `model_metadata["mac_count"]` if a caller (or a future export
step) has put one there, and reports `REASON_NO_OP_COUNT` otherwise. If a
trustworthy MAC count is ever computed at the source (e.g. during export, off
the real graph), latency estimation activates automatically — nothing here
needs to change.

── The one invented input ───────────────────────────────────────────────────
Turning a MAC count into milliseconds needs a throughput figure, and no real
benchmark data exists for any of these boards yet. `_DEVICE_CLASS_MACS_PER_CYCLE`
is an order-of-magnitude guess at "MACs executed per CPU clock cycle" for a
quantized int8 CNN, by device class — not a vendor number. `CONFIDENCE_NOTE`
travels with every estimated metric so this is never mistaken for a
measurement.
"""

import math
from dataclasses import dataclass, replace
from typing import Optional

from sqlalchemy.orm import Session

from app.models.user import DeviceClass
from app.services.compatibility import CompatibilityResult, check_compatibility

# ── Units ─────────────────────────────────────────────────────────────────────
UNIT_BYTES = "bytes"
UNIT_MS = "ms"

# ── Source tags ───────────────────────────────────────────────────────────────
SOURCE_MEASURED = "measured"
SOURCE_ESTIMATED = "estimated"
SOURCE_UNAVAILABLE = "unavailable"

# ── Budget provenance ────────────────────────────────────────────────────────
BUDGET_SPECIFICATION = "specification"
BUDGET_OVERRIDE = "override"

# ── Reason codes ─────────────────────────────────────────────────────────────
# Machine-readable counterpart to `note`, mirroring compatibility.py's split.
REASON_OK = "ok"
REASON_NO_BUDGET = "no_budget"
REASON_NO_ARTIFACT = "no_artifact"
REASON_NO_INPUT_SHAPE = "no_input_shape"
REASON_NO_OP_COUNT = "no_op_count"
REASON_NO_DEVICE = "no_device"
REASON_NO_CLOCK_RATE = "no_clock_rate"
# Incompatible pairs reuse compatibility.py's own reason codes verbatim
# (REASON_MODEL_INCOMPATIBLE, REASON_DEVICE_INCOMPATIBLE, etc.) rather than a
# re-derived one here — see `estimate()`.

# ── Confidence ────────────────────────────────────────────────────────────────
# One constant, attached to every `estimated` metric, so no figure this phase
# produces is ever rendered as a measurement.
CONFIDENCE_NOTE = (
    "Estimated, not measured — expect roughly ±50% error. Latency in "
    "particular is order-of-magnitude only until the model has actually run "
    "on this hardware."
)

# Assumed MACs executed per CPU clock cycle for a quantized int8 CNN, by
# device class. An order-of-magnitude modeling assumption, not a vendor
# benchmark — see module docstring. Class baseline reflects typical
# CPU-side SIMD width (none/DSP-extension on a microcontroller, NEON-class on
# an SBC or phone, AVX-class on a desktop); `accelerator` covers boards whose
# device_class itself IS a dedicated inference chip.
_DEVICE_CLASS_MACS_PER_CYCLE = {
    DeviceClass.microcontroller: 0.5,
    DeviceClass.linux_sbc: 4.0,
    DeviceClass.mobile: 8.0,
    DeviceClass.desktop: 32.0,
    DeviceClass.accelerator: 64.0,
}

# Extra multiplier when a board also carries a *named* on-chip accelerator
# (`has_ai_accelerator=True`) beyond whatever its device_class baseline
# already assumes — e.g. a microcontroller with a CMSIS-NN block, or an SBC
# with an attached NPU. Also an invented order-of-magnitude assumption.
_AI_ACCELERATOR_MULTIPLIER = 8.0


@dataclass(frozen=True)
class MetricEstimate:
    """One flash/RAM/latency figure. `margin = budget - value` (negative
    means over budget); both are None when there is no budget to compare
    against. `value=None` always pairs with a `reason_code` explaining why."""

    value: Optional[float]
    unit: str
    source: str  # measured | estimated | unavailable
    budget: Optional[float]
    budget_source: Optional[str]  # specification | override | None
    margin: Optional[float]
    fits: Optional[bool]
    reason_code: str
    note: str
    confidence: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "value": self.value,
            "unit": self.unit,
            "source": self.source,
            "budget": self.budget,
            "budget_source": self.budget_source,
            "margin": self.margin,
            "fits": self.fits,
            "reason_code": self.reason_code,
            "note": self.note,
            "confidence": self.confidence,
        }


@dataclass(frozen=True)
class EstimationResult:
    compatible: bool
    compatibility: CompatibilityResult
    flash: MetricEstimate
    ram: MetricEstimate
    latency: MetricEstimate
    # Target Device Phase 7 (docs/target_device_phase7.md): optimization
    # advice derived from the metrics above. Populated by `estimate()` via
    # `app.services.recommendations`, never recomputed by a caller.
    recommendations: tuple = ()
    recommendations_reason: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "compatible": self.compatible,
            "compatibility": self.compatibility.to_dict(),
            "flash_usage": self.flash.to_dict(),
            "ram_usage": self.ram.to_dict(),
            "inferencing_time": self.latency.to_dict(),
            "recommendations": [r.to_dict() for r in self.recommendations],
            "recommendations_reason": self.recommendations_reason,
        }


def _model_metadata(model) -> dict:
    """Mirrors compatibility._model_metadata: accept a TrainedModel row or
    the model_metadata dict itself."""
    if model is None:
        return {}
    if isinstance(model, dict):
        return model
    return getattr(model, "model_metadata", None) or {}


def _resolve_project(model):
    """TrainedModel -> TrainingJob -> Impulse -> Project, tolerating a bare
    dict (worker call sites) or a row with any link missing."""
    if model is None or isinstance(model, dict):
        return None
    training_job = getattr(model, "training_job", None)
    impulse = getattr(training_job, "impulse", None) if training_job else None
    return getattr(impulse, "project", None) if impulse else None


def _device_spec(db: Optional[Session], device_slug: Optional[str]):
    """The catalog entry's specification row for `device_slug`, or None if
    there is no device, no db, or no specification seeded for it."""
    if not device_slug or db is None:
        return None
    from app.models.devices import DeviceCatalogEntry

    entry = (
        db.query(DeviceCatalogEntry)
        .filter(DeviceCatalogEntry.slug == device_slug)
        .first()
    )
    return entry.specification if entry is not None else None


def _resolve_budget(project, spec, device_slug, spec_attr: str, override_attr: str):
    """One RAM/ROM/latency budget: a project override wins, but ONLY when
    `device_slug` is the project's own configured target device — an
    override recorded against one board is an application-level number, not
    a fact about a different board a caller is previewing via `device_slug`.
    Otherwise falls back to the board specification. Both None means no
    budget to compare against at all (spec_value=None is "unknown hardware
    fact", never a substituted default)."""
    spec_value = getattr(spec, spec_attr, None) if spec is not None else None
    if project is not None and device_slug and getattr(project, "target_device_slug", None) == device_slug:
        override = getattr(project, override_attr, None)
        if override is not None:
            return override, BUDGET_OVERRIDE
    if spec_value is not None:
        return spec_value, BUDGET_SPECIFICATION
    return None, None


def _incompatible_metric(compat: CompatibilityResult, unit: str) -> MetricEstimate:
    return MetricEstimate(
        value=None,
        unit=unit,
        source=SOURCE_UNAVAILABLE,
        budget=None,
        budget_source=None,
        margin=None,
        fits=None,
        reason_code=compat.reason_code,
        note=compat.message,
    )


def _flash_metric(model, spec, project, device_slug) -> MetricEstimate:
    file_bytes = getattr(model, "file_size_bytes", None) if not isinstance(model, dict) else model.get("file_size_bytes")
    budget_kb, budget_source = _resolve_budget(project, spec, device_slug, "rom_kb", "target_device_rom_kb")
    budget_bytes = budget_kb * 1024 if budget_kb is not None else None

    if not file_bytes:
        return MetricEstimate(
            value=None, unit=UNIT_BYTES, source=SOURCE_UNAVAILABLE,
            budget=budget_bytes, budget_source=budget_source, margin=None, fits=None,
            reason_code=REASON_NO_ARTIFACT,
            note="No model artifact available — cannot measure flash usage.",
        )

    margin = (budget_bytes - file_bytes) if budget_bytes is not None else None
    return MetricEstimate(
        value=file_bytes, unit=UNIT_BYTES, source=SOURCE_MEASURED,
        budget=budget_bytes, budget_source=budget_source,
        margin=margin, fits=(margin >= 0) if margin is not None else None,
        reason_code=REASON_OK if budget_bytes is not None else REASON_NO_BUDGET,
        note=(
            "Measured from the exported model file. The deployed package "
            "(runtime + generated code) is larger than this bare model size."
        ),
    )


def _ram_metric(meta: dict, spec, project, device_slug) -> MetricEstimate:
    """Unchanged from the pre-Phase-5 Model panel heuristic
    (`_estimate_peak_ram` in trained_models.py): 2x the input tensor size,
    as a stand-in for double-buffered arena allocation. Not improved here —
    the number was already user-visible through `panel`, so changing the
    formula would silently change what users already see."""
    budget_kb, budget_source = _resolve_budget(project, spec, device_slug, "ram_kb", "target_device_ram_kb")
    budget_bytes = budget_kb * 1024 if budget_kb is not None else None

    input_shape = meta.get("input_shape") or []
    if not input_shape:
        return MetricEstimate(
            value=None, unit=UNIT_BYTES, source=SOURCE_UNAVAILABLE,
            budget=budget_bytes, budget_source=budget_source, margin=None, fits=None,
            reason_code=REASON_NO_INPUT_SHAPE,
            note="Input shape unknown — cannot estimate peak RAM usage.",
        )

    quantized = bool(meta.get("quantized", False))
    bytes_per_el = 1 if quantized else 4
    estimate_bytes = math.prod(input_shape) * bytes_per_el * 2

    margin = (budget_bytes - estimate_bytes) if budget_bytes is not None else None
    return MetricEstimate(
        value=estimate_bytes, unit=UNIT_BYTES, source=SOURCE_ESTIMATED,
        budget=budget_bytes, budget_source=budget_source,
        margin=margin, fits=(margin >= 0) if margin is not None else None,
        reason_code=REASON_OK if budget_bytes is not None else REASON_NO_BUDGET,
        note=(
            f"Estimated as 2x the input tensor ({input_shape}, "
            f"{'int8' if quantized else 'float32'}) for double-buffering. "
            "Actual peak RAM depends on operator scheduling and arena "
            "allocation, which this heuristic does not model."
        ),
        confidence=CONFIDENCE_NOTE,
    )


def _latency_metric(meta: dict, spec, project, device_slug) -> MetricEstimate:
    budget_ms, budget_source = _resolve_budget(
        project, spec, device_slug, "latency_budget_ms", "target_device_latency_ms",
    )

    mac_count = meta.get("mac_count")
    if not mac_count:
        return MetricEstimate(
            value=None, unit=UNIT_MS, source=SOURCE_UNAVAILABLE,
            budget=budget_ms, budget_source=budget_source, margin=None, fits=None,
            reason_code=REASON_NO_OP_COUNT,
            note=(
                "No operation count available for this model. Nothing in the "
                "training/export pipeline records a MAC count yet, so "
                "latency cannot be estimated from the artifact alone."
            ),
        )

    if spec is None:
        return MetricEstimate(
            value=None, unit=UNIT_MS, source=SOURCE_UNAVAILABLE,
            budget=budget_ms, budget_source=budget_source, margin=None, fits=None,
            reason_code=REASON_NO_DEVICE,
            note=(
                "No target device selected — latency depends on device "
                "throughput, so there is nothing to estimate against."
                if not device_slug else
                "This device has no hardware specification on file yet — "
                "latency cannot be estimated without it."
            ),
        )

    if not spec.clock_rate_mhz:
        return MetricEstimate(
            value=None, unit=UNIT_MS, source=SOURCE_UNAVAILABLE,
            budget=budget_ms, budget_source=budget_source, margin=None, fits=None,
            reason_code=REASON_NO_CLOCK_RATE,
            note="This device's clock rate is unknown — latency cannot be estimated.",
        )

    macs_per_cycle = _DEVICE_CLASS_MACS_PER_CYCLE.get(spec.device_class, _DEVICE_CLASS_MACS_PER_CYCLE[DeviceClass.microcontroller])
    accelerated = bool(spec.has_ai_accelerator)
    if accelerated:
        macs_per_cycle *= _AI_ACCELERATOR_MULTIPLIER

    hz = spec.clock_rate_mhz * 1_000_000
    seconds = mac_count / (macs_per_cycle * hz)
    latency_ms = seconds * 1000

    margin = (budget_ms - latency_ms) if budget_ms is not None else None
    return MetricEstimate(
        value=latency_ms, unit=UNIT_MS, source=SOURCE_ESTIMATED,
        budget=budget_ms, budget_source=budget_source,
        margin=margin, fits=(margin >= 0) if margin is not None else None,
        reason_code=REASON_OK if budget_ms is not None else REASON_NO_BUDGET,
        note=(
            f"Estimated from {int(mac_count):,} MACs at an assumed "
            f"{macs_per_cycle:g} MACs/cycle for a {spec.device_class.value} "
            f"device{' with its AI accelerator' if accelerated else ''} at "
            f"{spec.clock_rate_mhz} MHz. Throughput is a modeling assumption, "
            "not a benchmark."
        ),
        confidence=CONFIDENCE_NOTE,
    )


def estimate(
    model,
    build_format: str,
    *,
    device_slug: Optional[str] = None,
    db: Optional[Session] = None,
) -> EstimationResult:
    """Flash / RAM / latency estimates for `model`, built as `build_format`,
    against `device_slug`'s budget (or no budget, if no device is selected).

    Runs Phase 4's `check_compatibility()` first: an incompatible pair is
    reported as such — via its own reason_code and message, not re-derived
    here — and no metric is computed for it.

    Also attaches Phase 7's recommendations (docs/target_device_phase7.md),
    derived from the metrics above via `app.services.recommendations`. A
    raise from that rule pass degrades to no advice, never to an error — the
    estimate itself must always render.
    """
    compat = check_compatibility(model, build_format, device_slug=device_slug, db=db)
    if not compat.compatible:
        result = EstimationResult(
            compatible=False,
            compatibility=compat,
            flash=_incompatible_metric(compat, UNIT_BYTES),
            ram=_incompatible_metric(compat, UNIT_BYTES),
            latency=_incompatible_metric(compat, UNIT_MS),
        )
    else:
        meta = _model_metadata(model)
        spec = _device_spec(db, device_slug)
        project = _resolve_project(model)

        result = EstimationResult(
            compatible=True,
            compatibility=compat,
            flash=_flash_metric(model, spec, project, device_slug),
            ram=_ram_metric(meta, spec, project, device_slug),
            latency=_latency_metric(meta, spec, project, device_slug),
        )

    try:
        # Imported here, not at module load, so this module and
        # `recommendations` (which type-hints against `EstimationResult`)
        # never form a real import cycle.
        from app.services.recommendations import recommend
        recs, reason = recommend(result, model)
    except Exception:
        recs, reason = (), None

    return replace(result, recommendations=tuple(recs), recommendations_reason=reason)
