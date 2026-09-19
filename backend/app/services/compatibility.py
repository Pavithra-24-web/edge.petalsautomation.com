from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

from sqlalchemy.orm import Session
from sqlalchemy import or_

from app.models.user import Deployment, DeploymentStatus, Device


def resolve_latest_compatible_deployment(
    db: Session,
    device: Device,
    *,
    force: bool = False,
    preferred_model_id: str | None = None,
    preferred_deployment_id: str | None = None,
) -> Deployment | None:
    if not device.deployment_target:
        return None

    query = (
        db.query(Deployment)
        .filter(
            Deployment.project_id == device.project_id,
            Deployment.deployment_target == device.deployment_target,
            Deployment.status == DeploymentStatus.completed,
            or_(
                Deployment.device_profile == device.device_profile,
                Deployment.device_profile.is_(None),
            ),
        )
    )

    if preferred_deployment_id:
        query = query.filter(Deployment.id == preferred_deployment_id)
    elif preferred_model_id:
        query = query.filter(Deployment.model_id == preferred_model_id)

    dep = query.order_by(
        (Deployment.device_profile == device.device_profile).desc(),
        Deployment.created_at.desc(),
    ).first()

    if dep is None:
        return None
    if not force and dep.id == device.installed_deployment_id:
        return None  # already up to date
    return dep


# ─────────────────────────────────────────────────────────────────────────────
# Model / device / build-format compatibility  (Target Device Phase 4)
#
# The question answered below is a different one from
# `resolve_latest_compatible_deployment` above: that one picks which *existing*
# build to push to a device; this one decides whether a build is *possible* at
# all.  They deliberately stay separate functions in the same module.
#
# Until Phase 4 these rules lived inside the deployment worker's generators, so
# an incompatible build was accepted, persisted, dispatched, and only failed
# minutes later.  The rules are stated here once, as data, and evaluated before
# the Deployment row is created.  The worker helpers now delegate here, so there
# is exactly one place that knows the rules.
# ─────────────────────────────────────────────────────────────────────────────

# ── Model types ──────────────────────────────────────────────────────────────
# One value per output-tensor contract.  Everything downstream keys on this,
# never on the raw metadata, so the three signals that identify a model type
# (`output_type`, `is_ssd`, `architecture`) are read in exactly one place.

MODEL_TYPE_CLASSIFICATION = "classification"
MODEL_TYPE_FOMO           = "fomo"
MODEL_TYPE_YOLO_PRO       = "yolo_pro"
MODEL_TYPE_SSD            = "ssd"

MODEL_TYPES = frozenset({
    MODEL_TYPE_CLASSIFICATION,
    MODEL_TYPE_FOMO,
    MODEL_TYPE_YOLO_PRO,
    MODEL_TYPE_SSD,
})

# Detection output types whose tensors the embedded C templates cannot handle.
# YOLO-Pro emits multi-scale detection tensors; SSD emits separate boxes /
# scores / classes / count tensors.  Both are multi-output and cannot be
# consumed by the single flat-output loop in the generated C++/Arduino code.
_SSD_OUTPUT_TYPES = frozenset({
    "object_detection",    # MobileNetV2-SSD (legacy label)
    "ssd_detection",       # MobileNetV2-SSD (canonical label)
})
_YOLO_PRO_OUTPUT_TYPES = frozenset({"yolo_pro_detection"})

# Same set the worker's `_UNSUPPORTED_EMBEDDED_DETECTION_OUTPUT_TYPES` named,
# kept here as the single definition both sides read.
UNSUPPORTED_EMBEDDED_DETECTION_OUTPUT_TYPES = _SSD_OUTPUT_TYPES | _YOLO_PRO_OUTPUT_TYPES


def _model_metadata(model) -> dict:
    """Accept a TrainedModel row or the `model_metadata` dict itself.

    The API gate holds the ORM row; the worker helpers only ever see the dict
    they were already passed.  Both must resolve to the same model type.
    """
    if model is None:
        return {}
    if isinstance(model, dict):
        return model
    return getattr(model, "model_metadata", None) or {}


def classify_model_type(model) -> str:
    """Collapse the three metadata signals into a single model-type value.

    FOMO is tested first, matching the worker's call order
    (`_raise_if_fomo_embedded` before `_raise_if_detection_embedded`): a model
    that trips both signals is reported as FOMO, with FOMO's reason text.

    An unknown or missing `output_type` with no other signal is classification.
    That is the pre-Phase-4 worker behaviour — such a model builds today — and
    changing it here would reject builds that currently succeed.
    """
    meta = _model_metadata(model)
    output_type  = meta.get("output_type") or ""
    architecture = meta.get("architecture") or ""

    if output_type == "detection_heatmap" or "fomo" in architecture.lower():
        return MODEL_TYPE_FOMO
    if output_type in _YOLO_PRO_OUTPUT_TYPES:
        return MODEL_TYPE_YOLO_PRO
    if output_type in _SSD_OUTPUT_TYPES or meta.get("is_ssd", False):
        return MODEL_TYPE_SSD
    return MODEL_TYPE_CLASSIFICATION


# ── Build format × model type ────────────────────────────────────────────────
# Stated as what each format ACCEPTS, never as a list of exceptions: a format
# added to `DeployTarget` without an entry here is rejected as `unknown_format`
# at the API rather than silently defaulting to permissive and failing mid-build.

_CLASSIFICATION_ONLY = frozenset({MODEL_TYPE_CLASSIFICATION})
_EVERY_MODEL_TYPE    = frozenset(MODEL_TYPES)

FORMAT_MODEL_SUPPORT: Mapping[str, frozenset] = MappingProxyType({
    # Embedded C++ code generators — single flat classification output only.
    "arduino":      _CLASSIFICATION_ONLY,
    "esp32":        _CLASSIFICATION_ONLY,
    "cpp":          _CLASSIFICATION_ONLY,
    # Runtime packages — they ship their own inference code for every type.
    "tflite":       _EVERY_MODEL_TYPE,
    "raspberry_pi": _EVERY_MODEL_TYPE,
    "unoq":         _EVERY_MODEL_TYPE,
    "pxe":          _EVERY_MODEL_TYPE,
})

# Build formats that produce a runnable package rather than a C++ source tree.
# Mirrors the worker's own `.pe` grouping (`resolved_target in ("tflite",
# "raspberry_pi", "unoq")`): only those run the .pxe stdio-JSONL runner, so only
# a device resolving to one of them can be handed a .pxe package.
_PXE_CAPABLE_TARGETS = frozenset({"tflite", "raspberry_pi", "unoq"})

# Rejection copy, keyed by model type.  Carried over verbatim from the worker's
# `_raise_if_fomo_embedded` / `_raise_if_detection_embedded` — those helpers now
# raise these strings, and `test_device_packages.py` asserts on them.
_FOMO_REJECTION = (
    "FOMO (fomo_mobilenetv2_0_1) models are not supported for the "
    "'{target}' deployment target.  The FOMO output tensor is 4-dimensional "
    "(1, grid_h, grid_w, num_classes+1) and requires a custom heatmap "
    "post-processing loop that is not yet generated for embedded C++ targets.  "
    "Use the 'tflite' or 'raspberry_pi' target instead — both include "
    "correct FOMO inference code."
)

_DETECTION_REJECTION = (
    "Detection model (output_type='{output_type}') is not supported "
    "for the '{target}' deployment target.  YOLO-Pro and SSD models "
    "produce multiple output tensors (boxes, scores, classes, count) "
    "that require a custom post-processing loop not present in the "
    "generated embedded C++ template.  Use the 'tflite' or "
    "'raspberry_pi' target instead."
)

_MODEL_REJECTION_TEMPLATES: Mapping[str, str] = MappingProxyType({
    MODEL_TYPE_FOMO:     _FOMO_REJECTION,
    MODEL_TYPE_YOLO_PRO: _DETECTION_REJECTION,
    MODEL_TYPE_SSD:      _DETECTION_REJECTION,
})


# ── Reason codes ─────────────────────────────────────────────────────────────
# Machine-readable counterpart to `message`.  Callers branch on the code; only
# humans read the message.

REASON_COMPATIBLE          = "compatible"
REASON_UNKNOWN_FORMAT      = "unknown_format"
REASON_MODEL_INCOMPATIBLE  = "model_format_incompatible"
REASON_DEVICE_INCOMPATIBLE = "device_format_incompatible"
REASON_UNKNOWN_DEVICE      = "unknown_device"


@dataclass(frozen=True)
class CompatibilityResult:
    """The answer to "can this model, in this format, run on this device?".

    Never a bare bool: the API returns `reason_code`, the UI renders `message`.
    A caller handed only a bool re-derives the reason and drifts from it.
    """
    compatible: bool
    reason_code: str
    message: str
    model_type: str
    build_format: str
    device_slug: str | None = None

    def to_dict(self) -> dict:
        return {
            "compatible":   self.compatible,
            "reason_code":  self.reason_code,
            "message":      self.message,
            "model_type":   self.model_type,
            "build_format": self.build_format,
            "device_slug":  self.device_slug,
        }


def catalog_deploy_target(db: Session | None, device_slug: str | None) -> str | None:
    """Resolve a catalog slug to its `deploy_target`, or None if it does not resolve.

    None means one of two very different things, and callers must not conflate
    them: *no device to look up* (no slug, or no session to look it up with) is a
    valid state, while *a slug the catalog does not have* is a bad request — the
    worker's `_resolve_deployment_target` raises on exactly that, so the API has
    to predict the rejection rather than wave the build through. Only
    `check_compatibility` separates the two; it checks the slug first.

    Public (not `_`-prefixed): `deployment_format_offer` (Target Device Phase 6)
    needs the raw resolution too, ahead of applying the package policy.
    """
    if not device_slug or db is None:
        return None

    from app.models.devices import DeviceCatalogEntry

    entry = (
        db.query(DeviceCatalogEntry)
        .filter(DeviceCatalogEntry.slug == device_slug)
        .first()
    )
    if entry is None:
        return None
    target = entry.deploy_target
    return getattr(target, "value", target)


def resolve_build_format(
    db: Session | None,
    *,
    target: str,
    device_profile: str | None = None,
    deployment_format: str = "pe",
) -> str:
    """The format the worker will actually build for this request.

    Mirrors the worker's dispatch (`run_deployment_job` →
    `_resolve_deployment_target`): a `.pxe` deployment_format short-circuits to
    `"pxe"`; otherwise the device profile's catalog `deploy_target` wins over
    the requested `target`. Validating anything other than this format would
    check a build that is not the one about to run.

    An unknown profile falls back to `target` rather than raising: the worker
    raises there, and `check_compatibility` predicts that rejection as
    `unknown_device`. Raising here too would mean two rejection paths for one
    condition, and the caller could no longer report a reason.
    """
    if deployment_format == "pxe":
        return "pxe"
    return catalog_deploy_target(db, device_profile) or target


def _compatible(model_type: str, build_format: str, device_slug: str | None) -> "CompatibilityResult":
    """The pass result — reached both with a device and with no device at all."""
    return CompatibilityResult(
        compatible=True,
        reason_code=REASON_COMPATIBLE,
        message=(
            f"A {model_type} model is supported by the '{build_format}' "
            f"deployment format."
        ),
        model_type=model_type,
        build_format=build_format,
        device_slug=device_slug,
    )


def supported_formats_for_deploy_target(deploy_target: str) -> frozenset:
    """Build formats a device resolving to `deploy_target` can be handed."""
    if deploy_target in _PXE_CAPABLE_TARGETS:
        return frozenset({deploy_target, "pxe"})
    return frozenset({deploy_target})


def check_compatibility(
    model,
    build_format: str,
    *,
    device_slug: str | None = None,
    db: Session | None = None,
) -> CompatibilityResult:
    """Decide whether `model` can be built as `build_format` for `device_slug`.

    `model` is a TrainedModel row or its `model_metadata` dict.
    `build_format` is the format that will actually be produced — for a `.pxe`
    build that is `"pxe"`, otherwise the device's catalog `deploy_target`.
    `device_slug` is optional: a project with no target device is still valid,
    it just has no device rule to evaluate.
    """
    model_type = classify_model_type(model)
    meta = _model_metadata(model)

    accepted = FORMAT_MODEL_SUPPORT.get(build_format)
    if accepted is None:
        return CompatibilityResult(
            compatible=False,
            reason_code=REASON_UNKNOWN_FORMAT,
            message=(
                f"Unknown deployment format '{build_format}'.  Supported "
                f"formats: {', '.join(sorted(FORMAT_MODEL_SUPPORT))}."
            ),
            model_type=model_type,
            build_format=build_format,
            device_slug=device_slug,
        )

    # ── model × format ───────────────────────────────────────────────────────
    if model_type not in accepted:
        template = _MODEL_REJECTION_TEMPLATES[model_type]
        return CompatibilityResult(
            compatible=False,
            reason_code=REASON_MODEL_INCOMPATIBLE,
            message=template.format(
                target=build_format,
                output_type=meta.get("output_type", ""),
            ),
            model_type=model_type,
            build_format=build_format,
            device_slug=device_slug,
        )

    # ── device × format ──────────────────────────────────────────────────────
    # No device and an unknown device are different answers. A project with no
    # target device selected has nothing to check here; a slug the catalog does
    # not have is a build the worker would refuse with
    # "Unsupported deployment device_profile", and refusing it here is the whole
    # point of predicting rejections before the build starts.
    if not device_slug or db is None:
        return _compatible(model_type, build_format, device_slug)

    deploy_target = catalog_deploy_target(db, device_slug)
    if deploy_target is None:
        return CompatibilityResult(
            compatible=False,
            reason_code=REASON_UNKNOWN_DEVICE,
            message=(
                f"Target device '{device_slug}' is not in the device catalog, so "
                f"no package can be built for it.  Select a device from the "
                f"catalog, or clear the target device to build for "
                f"'{build_format}' directly."
            ),
            model_type=model_type,
            build_format=build_format,
            device_slug=device_slug,
        )

    if build_format not in supported_formats_for_deploy_target(deploy_target):
        alternative = (
            f"'{deploy_target}' or '.pxe'"
            if deploy_target in _PXE_CAPABLE_TARGETS
            else f"'{deploy_target}'"
        )
        return CompatibilityResult(
            compatible=False,
            reason_code=REASON_DEVICE_INCOMPATIBLE,
            message=(
                f"Target device '{device_slug}' cannot be built as "
                f"'{build_format}'.  It builds as {alternative}.  Select that "
                f"format, or choose a device that builds as '{build_format}'."
            ),
            model_type=model_type,
            build_format=build_format,
            device_slug=device_slug,
        )

    return _compatible(model_type, build_format, device_slug)


# ─────────────────────────────────────────────────────────────────────────────
# Package policy — what PetalEdge offers per deploy target  (Target Device
# Phase 6, docs/target_device_phase6_prompt.md)
#
# A different question from `supported_formats_for_deploy_target` above: that
# one asks what a build format CAN produce for a deploy target (a capability
# fact the worker's dispatch depends on, and stays unchanged). This asks what
# PetalEdge actually OFFERS the user for it — a product decision layered on
# top. `tflite` is `.pxe`-capable but policy offers only `.pe`: capability is
# not permission.
#
# Stated as what each target OFFERS, exhaustively over every deploy target —
# never as exceptions over a permissive default. A target added later with no
# entry here fails closed to "no package" via `.get()` rather than silently
# inheriting `.pe`.
# ─────────────────────────────────────────────────────────────────────────────

REASON_POLICY_NO_PACKAGE = "policy_no_package"

_PACKAGE_POLICY: Mapping[str, str | None] = MappingProxyType({
    "tflite":       "pe",
    "raspberry_pi": "pxe",
    "unoq":         "pxe",
    "arduino":      None,
    "esp32":        None,
    "cpp":          None,
})

# Every deploy target the policy has an opinion on — the six device-catalog
# `deploy_target` values. Not `DeployTarget`'s seven: `"pxe"` there is a build
# format, not something a device resolves to, so it has no policy entry.
ALL_DEPLOY_TARGETS: tuple = tuple(_PACKAGE_POLICY.keys())

_NO_PACKAGE_MESSAGE = (
    "No Petal Edge deployment package is available for the '{target}' "
    "target yet — its .pxe generator has not been implemented. This is a "
    "limitation of the target, not of the hardware."
)


@dataclass(frozen=True)
class PackagePolicyResult:
    """The answer to "what package, if any, does PetalEdge offer for this
    deploy target?" — exactly one package per target, never two, per the
    policy table in docs/target_device_phase6_prompt.md §1.
    """
    deploy_target: str
    package: str | None            # "pe", "pxe", or None
    build_format: str | None       # the value `resolve_build_format` would
                                    # produce for this package — `deploy_target`
                                    # itself for "pe", "pxe" for "pxe"
    available: bool
    reason_code: str
    message: str

    def to_dict(self) -> dict:
        return {
            "deploy_target": self.deploy_target,
            "package": self.package,
            "build_format": self.build_format,
            "available": self.available,
            "reason_code": self.reason_code,
            "message": self.message,
        }


def package_policy_for(deploy_target: str) -> PackagePolicyResult:
    """The single package PetalEdge offers for `deploy_target`, or none.

    Read by both `GET /deployment/targets` and (indirectly, through that
    endpoint) the deployment page — neither may re-derive this table.
    """
    package = _PACKAGE_POLICY.get(deploy_target)
    if package is None:
        return PackagePolicyResult(
            deploy_target=deploy_target,
            package=None,
            build_format=None,
            available=False,
            reason_code=REASON_POLICY_NO_PACKAGE,
            message=_NO_PACKAGE_MESSAGE.format(target=deploy_target),
        )
    build_format = "pxe" if package == "pxe" else deploy_target
    return PackagePolicyResult(
        deploy_target=deploy_target,
        package=package,
        build_format=build_format,
        available=True,
        reason_code=REASON_COMPATIBLE,
        message=f"'{deploy_target}' ships the '.{package}' deployment package.",
    )


def all_package_policies() -> dict:
    """`package_policy_for` applied to every deploy target — the full policy
    table, keyed by `deploy_target`.

    The source for `GET /deployment/package-policy`. Callers must not
    re-derive the target list or the policy themselves; this is the one place
    both are enumerated.
    """
    return {
        target: package_policy_for(target).to_dict()
        for target in ALL_DEPLOY_TARGETS
    }


def _offered_package_label(build_format: str) -> str:
    return "pxe" if build_format == "pxe" else "pe"


def deployment_format_offer(
    db: Session | None,
    model,
    *,
    device_profile: str | None = None,
) -> dict:
    """The full "what can this build as?" answer for one model × device.

    Composes, in order: catalog resolution of `device_profile` to a deploy
    target, the §1 package policy for that target, and — for a target the
    policy offers — Phase 4's `check_compatibility` to catch a model the
    target's package cannot run. Only one of these three ever supplies the
    final reason: an unresolved device_profile short-circuits to
    `REASON_UNKNOWN_DEVICE` (mirroring the worker's rejection), a target with
    no offered package short-circuits to `REASON_POLICY_NO_PACKAGE` without
    ever checking the model — "there is no package to be incompatible with" —
    and only a policy-available target reaches the model check.

    No `device_profile` resolves against `tflite` directly (the legacy
    Generic TFLite default) with no device rule applied, so a project with no
    target device gets exactly the same model-only answer it always has.
    """
    if device_profile is None:
        deploy_target = "tflite"
    else:
        deploy_target = catalog_deploy_target(db, device_profile)
        if deploy_target is None:
            rejection = check_compatibility(model, "tflite", device_slug=device_profile, db=db)
            return {
                "device_profile": device_profile,
                "deploy_target": None,
                "offered": None,
                "unavailable": [],
                "reason_code": rejection.reason_code,
                "message": rejection.message,
            }

    candidates = sorted(supported_formats_for_deploy_target(deploy_target))
    policy = package_policy_for(deploy_target)

    if not policy.available:
        unavailable = [
            {
                "package": _offered_package_label(bf),
                "build_format": bf,
                "available": False,
                "reason_code": policy.reason_code,
                "message": policy.message,
            }
            for bf in candidates
        ]
        return {
            "device_profile": device_profile,
            "deploy_target": deploy_target,
            "offered": None,
            "unavailable": unavailable,
            "reason_code": policy.reason_code,
            "message": policy.message,
        }

    compat = check_compatibility(model, policy.build_format, device_slug=device_profile, db=db)

    unavailable = [
        {
            "package": _offered_package_label(bf),
            "build_format": bf,
            "available": False,
            "reason_code": REASON_POLICY_NO_PACKAGE,
            "message": (
                f"PetalEdge offers only the '.{policy.package}' package for "
                f"'{deploy_target}'; '.{_offered_package_label(bf)}' is not offered."
            ),
        }
        for bf in candidates
        if bf != policy.build_format
    ]

    if not compat.compatible:
        unavailable.insert(0, {
            "package": policy.package,
            "build_format": policy.build_format,
            "available": False,
            "reason_code": compat.reason_code,
            "message": compat.message,
        })
        return {
            "device_profile": device_profile,
            "deploy_target": deploy_target,
            "offered": None,
            "unavailable": unavailable,
            "reason_code": compat.reason_code,
            "message": compat.message,
        }

    return {
        "device_profile": device_profile,
        "deploy_target": deploy_target,
        "offered": {
            "package": policy.package,
            "target": deploy_target,
            "deployment_format": policy.package,
            "build_format": policy.build_format,
        },
        "unavailable": unavailable,
        "reason_code": compat.reason_code,
        "message": compat.message,
    }
