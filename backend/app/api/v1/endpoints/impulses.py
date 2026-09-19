"""
Impulse (pipeline) endpoints — full Edge Impulse-compatible workflow.

Routes:
  POST   /impulses/                         create
  GET    /impulses/project/{project_id}     list for project
  GET    /impulses/{impulse_id}             get single
  PUT    /impulses/{impulse_id}             update (full replace)
  PATCH  /impulses/{impulse_id}/blocks      add / remove a single block
  POST   /impulses/{impulse_id}/validate    validate readiness for training
  DELETE /impulses/{impulse_id}             delete
"""
import re
from fastapi import APIRouter, Depends, HTTPException, Body, Response
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import Optional, List, Literal
from datetime import datetime

from app.core.database import get_db
from app.core.auth import get_current_user
from app.core.authz import assert_project_owner, assert_impulse_owner
from app.models.user import (
    User,
    Project,
    Impulse,
    TrainingJob,
    JobStatus,
    TrainedModel,
    Deployment,
    DeviceUpdateHistory,
    FeatureSet,
    PostProcessingSettings,
)
from app.models.model_testing import (
    MetricResult,
    ModelTestRun,
    ModelTestSample,
    ModelVersion,
)
from app.ml.dsp.processor import merge_image_params

router = APIRouter()


def _delete_impulse_dependencies(db: Session, impulse_id: str) -> None:
    """Remove all rows that still reference an impulse before deleting it."""
    db.query(PostProcessingSettings).filter(
        PostProcessingSettings.impulse_id == impulse_id
    ).delete(synchronize_session=False)

    training_job_ids = [
        row[0]
        for row in db.query(TrainingJob.id)
        .filter(TrainingJob.impulse_id == impulse_id)
        .all()
    ]
    trained_model_ids = []
    deployment_ids = []

    if training_job_ids:
        trained_model_ids = [
            row[0]
            for row in db.query(TrainedModel.id)
            .filter(TrainedModel.training_job_id.in_(training_job_ids))
            .all()
        ]

    if trained_model_ids:
        deployment_ids = [
            row[0]
            for row in db.query(Deployment.id)
            .filter(Deployment.model_id.in_(trained_model_ids))
            .all()
        ]

    run_ids = [
        row[0]
        for row in db.query(ModelTestRun.id)
        .filter(ModelTestRun.impulse_id == impulse_id)
        .all()
    ]

    db.query(FeatureSet).filter(
        FeatureSet.impulse_id == impulse_id
    ).delete(synchronize_session=False)

    if run_ids:
        db.query(MetricResult).filter(
            MetricResult.test_run_id.in_(run_ids)
        ).delete(synchronize_session=False)

    if deployment_ids:
        db.query(DeviceUpdateHistory).filter(
            DeviceUpdateHistory.deployment_id.in_(deployment_ids)
        ).delete(synchronize_session=False)
        db.query(Deployment).filter(
            Deployment.id.in_(deployment_ids)
        ).delete(synchronize_session=False)

    if trained_model_ids:
        db.query(TrainedModel).filter(
            TrainedModel.id.in_(trained_model_ids)
        ).delete(synchronize_session=False)

    if training_job_ids:
        db.query(TrainingJob).filter(
            TrainingJob.id.in_(training_job_ids)
        ).delete(synchronize_session=False)

    db.query(ModelTestSample).filter(
        ModelTestSample.impulse_id == impulse_id
    ).delete(synchronize_session=False)

    db.query(ModelTestRun).filter(
        ModelTestRun.impulse_id == impulse_id
    ).delete(synchronize_session=False)

    db.query(ModelVersion).filter(
        ModelVersion.impulse_id == impulse_id
    ).delete(synchronize_session=False)


# ─── Schemas ──────────────────────────────────────────────────────────────────

class ImpulseCreate(BaseModel):
    project_id: str
    # Optional: when omitted or blank, the backend assigns the next default
    # name from the project's monotonic impulse counter.
    name: Optional[str] = None
    description: Optional[str] = None

    # Time-series input
    window_size_ms:    int   = 1000
    window_increase_ms:int   = 500
    frequency_hz:      float = 100.0
    zero_pad_allowed:  bool  = True

    # Input type and axes
    input_type:  str        = "time-series"   # time-series | image | other
    input_axes:  List[str]  = []
    sensor_type: Optional[str] = None         # accelerometer | microphone | camera | custom

    # Image input
    image_width:  int = 96
    image_height: int = 96
    resize_mode:  str = "Fit shortest axis"

    # Percent of training samples to actually use during feature gen + training.
    # 100 = full dataset. Clamped to [1, 100] in the route handlers below.
    train_subset_percent: float = 100.0

    dsp_blocks: List[dict] = []
    ml_blocks:  List[dict] = []


class BlockPatch(BaseModel):
    action:     Literal["add", "remove"]
    block_kind: Literal["dsp", "ml"]       # which list to modify
    block:      Optional[dict] = None      # required for add
    block_index:Optional[int]  = None      # required for remove
    
    # Optional root fields to sync during patch
    image_width:  Optional[int] = None
    image_height: Optional[int] = None
    window_size_ms: Optional[int] = None
    window_increase_ms: Optional[int] = None
    frequency_hz: Optional[float] = None
    name: Optional[str] = None


class ImpulseValidateResponse(BaseModel):
    valid:    bool
    errors:   List[str]
    warnings: List[str]


# ─── Serializer ───────────────────────────────────────────────────────────────

def _imp(i: Impulse) -> dict:
    dsp_blocks = [dict(b) for b in (i.dsp_blocks or [])]
    ml_blocks = [dict(b) for b in (i.ml_blocks or [])]

    # A motion impulse can end up with input_type/input_axes stuck at
    # "image"/["image"] from a prior save that briefly held an image DSP
    # block (see _sync_root_input_fields's preserve_explicit_image — it
    # trusts whatever input_type the client last echoed back, and a client
    # that never resets its own local copy keeps re-sending "image" forever).
    # Motion never has an image branch, so for a motion project with no
    # actual image block this is always corrupted state, not a real choice —
    # correct it at read time rather than only on the next save, so a stuck
    # impulse displays correctly the moment it's opened.
    has_image_block = any(b.get("type") == "image" for b in dsp_blocks)
    is_motion_project = getattr(getattr(i, "project", None), "project_type", None) == "motion"
    stuck_on_image = (
        is_motion_project and not has_image_block
        and (getattr(i, "input_type", "time-series") or "time-series") == "image"
    )
    display_input_type = "time-series" if stuck_on_image else (getattr(i, "input_type", "time-series") or "time-series")
    display_input_axes = getattr(i, "input_axes", []) or []
    if stuck_on_image and display_input_axes == ["image"]:
        display_input_axes = []
    if stuck_on_image:
        dsp_blocks = [
            {**b, "input_axes": []} if b.get("input_axes") == ["image"] else b
            for b in dsp_blocks
        ]

    image_block = next((b for b in dsp_blocks if b.get("type") == "image"), {"params": {}})
    merged_img_params = merge_image_params(
        {
            "input_type": getattr(i, "input_type", "time-series"),
            "image_width": getattr(i, "image_width", 96),
            "image_height": getattr(i, "image_height", 96),
            "resize_mode": getattr(i, "resize_mode", "Fit shortest axis"),
            "ml_blocks": ml_blocks,
        },
        image_block,
    )
    normalized_w = int(merged_img_params.get("image_width", getattr(i, "image_width", 96)))
    normalized_h = int(merged_img_params.get("image_height", getattr(i, "image_height", 96)))
    normalized_resize = merged_img_params.get("resize_mode", getattr(i, "resize_mode", "Fit shortest axis"))

    normalized_dsp_blocks = []
    for block in dsp_blocks:
        if block.get("type") == "image":
            normalized_dsp_blocks.append({
                **block,
                "params": {
                    **(block.get("params") or {}),
                    "image_width": normalized_w,
                    "image_height": normalized_h,
                    "resize_mode": normalized_resize,
                },
            })
        else:
            normalized_dsp_blocks.append(block)

    return {
        "id":                 i.id,
        "project_id":         i.project_id,
        "name":               i.name,
        "description":        i.description,

        # Time-series
        "window_size_ms":     i.window_size_ms,
        "window_increase_ms": i.window_increase_ms,
        "frequency_hz":       i.frequency_hz,
        "zero_pad_allowed":   i.zero_pad_allowed,

        # Input type
        "input_type":         display_input_type,
        "input_axes":         display_input_axes,
        "sensor_type":        getattr(i, "sensor_type", None),

        # Image
        "image_width":        normalized_w,
        "image_height":       normalized_h,
        "resize_mode":        normalized_resize,

        # Training subset (% of training samples to actually use). Default 100.
        "train_subset_percent": float(getattr(i, "train_subset_percent", 100.0) or 100.0),

        # Blocks
        "dsp_blocks":         normalized_dsp_blocks,
        "ml_blocks":          ml_blocks,

        # Output metadata
        "output_config":      getattr(i, "output_config", {}) or {},

        # Backend-confirmed Parameters-save state. NULL means the user has not
        # successfully saved the Parameters step yet; downstream pages gate on
        # this value being truthy. Always serialize so the frontend never has
        # to infer save-state from field presence.
        "dsp_params_saved_at": (
            i.dsp_params_saved_at.isoformat()
            if getattr(i, "dsp_params_saved_at", None) else None
        ),

        "created_at":         i.created_at.isoformat(),
        "updated_at":         i.updated_at.isoformat(),
    }


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _apply_defaults(block: dict) -> dict:
    """Fill in default params for a DSP / ML block if params are missing."""
    from app.api.v1.endpoints.dsp import _PROCESSING_BLOCKS_ALL, _LEARNING_BLOCKS_ALL
    all_blocks = _PROCESSING_BLOCKS_ALL + _LEARNING_BLOCKS_ALL
    catalog    = {b["type"]: b for b in all_blocks}
    meta       = catalog.get(block.get("type", ""), {})
    params_schema = meta.get("params", {})
    filled_params = {}
    for k, schema in params_schema.items():
        filled_params[k] = block.get("params", {}).get(k, schema.get("default"))
    return {**block, "params": {**filled_params, **block.get("params", {})}}


def _validate_impulse(imp: Impulse):
    errors, warnings = [], []
    if not imp.dsp_blocks:
        errors.append("At least one processing block is required.")
    if not imp.ml_blocks:
        errors.append("At least one learning block is required.")
    if imp.input_type == "time-series" and imp.frequency_hz <= 0:
        errors.append("Frequency must be greater than 0 Hz.")
    if imp.input_type == "image":
        if imp.image_width <= 0 or imp.image_height <= 0:
            errors.append("Image width and height must be greater than 0.")
        dsp_types = [b.get("type") for b in (imp.dsp_blocks or [])]
        if "image" not in dsp_types:
            warnings.append("Image impulses typically use the Image processing block.")

        # FOMO minimum resolution check — mirrors Edge Impulse behaviour where
        # the UI warns the user when their input block size is too small for the
        # FOMO learning block to produce a usable detection grid.
        # v1: at least 6×6 grid (96×96 input at stride-16).
        # v2: at least 6×6 grid (48×48 input at stride-8), no 96×96 floor.
        from app.ml.dsp.processor import _uses_fomo_min_resolution, _get_fomo_version
        if _uses_fomo_min_resolution(imp):
            _fver = _get_fomo_version(imp)
            eff_w = imp.image_width  or (128 if _fver == 2 else 96)
            eff_h = imp.image_height or (128 if _fver == 2 else 96)
            if _fver == 2:
                # v2: stride-8; minimum usable grid is 6×6 = 48×48 input.
                grid_w = eff_w // 8
                grid_h = eff_h // 8
                if grid_w < 6 or grid_h < 6:
                    warnings.append(
                        f"FOMO v2 requires at least a 6×6 detection grid. "
                        f"Current image size {eff_w}×{eff_h} at stride-8 produces a {grid_w}×{grid_h} grid. "
                        f"Set image width and height to at least 48×48 and re-run Generate Features."
                    )
                if eff_w % 8 != 0 or eff_h % 8 != 0:
                    warnings.append(
                        f"FOMO v2 (Adaptive Resolution) requires image dimensions divisible by 8. "
                        f"Current size {eff_w}×{eff_h} will be auto-snapped to the nearest multiple of 8."
                    )
            else:
                # v1 (unchanged): stride-16; recommended minimum 96×96 → 6×6 grid.
                grid_w = eff_w // 8
                grid_h = eff_h // 8
                if grid_w < 12 or grid_h < 12:
                    warnings.append(
                        f"FOMO works best with a minimum 12×12 detection grid. "
                        f"Current image size {eff_w}×{eff_h} produces a {grid_w}×{grid_h} grid "
                        f"({grid_w * grid_h} cells total). "
                        f"Set image width and height to at least 96×96 and re-run "
                        f"Generate Features to get the recommended 12×12 grid (144 cells)."
                    )

    return errors, warnings


def _sync_root_input_fields(
    imp: Impulse, dsp_blocks: list, preserve_explicit_image: bool = False,
    project_type: Optional[str] = None,
):
    """
    Keep the root impulse input fields aligned with the DSP pipeline.

    The UI and training code use the root input_type as the source of truth,
    so adding an image DSP block must flip the impulse to image mode. When the
    image block is removed, fall back to the default time-series mode.

    `project_type` gates the image branches to object_detection: motion never
    has an image branch (its processing-block catalog excludes "image"
    entirely, so a real image DSP block can never legitimately land here for
    a motion impulse), and `preserve_explicit_image` only exists to stop a
    *client re-save* from flipping an intentionally-image impulse back to
    time-series before its first image block is added — trusting it for
    motion would instead make an accidental/stale input_type="image" stick
    forever, since the client just keeps echoing back whatever the server
    last told it. Passing project_type=None (the caller didn't specify) keeps
    prior behavior — needed so existing object_detection call sites are
    unaffected without every one of them threading the project through.
    """
    is_motion = project_type == "motion"

    image_block = None if is_motion else next(
        (b for b in (dsp_blocks or []) if b.get("type") == "image"), None
    )
    if image_block:
        imp.input_type = "image"
        imp.input_axes = ["image"]
        if not imp.sensor_type:
            imp.sensor_type = "camera"
        params = image_block.get("params") or {}
        if params.get("image_width"):
            imp.image_width = int(params["image_width"])
        if params.get("image_height"):
            imp.image_height = int(params["image_height"])
        return

    if not is_motion and preserve_explicit_image and imp.input_type == "image":
        imp.input_axes = ["image"]
        if not imp.sensor_type:
            imp.sensor_type = "camera"
        return

    imp.input_type = "time-series"
    if imp.input_axes == ["image"]:
        imp.input_axes = []


def _snap_to_mult8(v: int) -> int:
    """Round up to the nearest multiple of 8. e.g. 97→104, 96→96, 48→48."""
    return v if v % 8 == 0 else v + (8 - v % 8)


def _enforce_fomo_min_resolution(imp: Impulse):
    """
    Enforce image dimension constraints for FOMO impulses at save time.

    Two rules applied in order, matching Edge Impulse behaviour:
      1. Minimum 96×96 — FOMO needs at least a 12×12 detection grid.
      2. Snap to nearest multiple of 8 — MobileNetV2 stride schedule requires
         multiples of 8. A non-multiple (e.g. 97) causes a silent Resizing
         layer that stretches every image and inflates MACs by up to 5×.
         97 → 104, 100 → 104, 96 → 96 (unchanged).

    Both rules apply only when the impulse uses a FOMO learning block.
    Non-FOMO impulses (classification, time-series) are untouched.
    """
    merged_img_params = merge_image_params(
        {
            "input_type": imp.input_type,
            "image_width": imp.image_width,
            "image_height": imp.image_height,
            "resize_mode": imp.resize_mode,
            "ml_blocks": imp.ml_blocks or [],
        },
        next((b for b in (imp.dsp_blocks or []) if b.get("type") == "image"), {"params": {}}),
    )

    normalized_w = int(merged_img_params.get("image_width", imp.image_width or 96))
    normalized_h = int(merged_img_params.get("image_height", imp.image_height or 96))
    normalized_resize = merged_img_params.get("resize_mode", imp.resize_mode or "Fit shortest axis")

    # Snap to multiple of 8 for FOMO impulses so no Resizing layer is inserted.
    # v1: merge_image_params already applied the 96×96 floor; snap on top.
    # v2: no 96×96 floor — only snap to mult-of-8 (divisibility requirement).
    # Guard: v2 snap logic is identical to v1 snap; only the floor differs,
    # which is already handled by merge_image_params.
    from app.ml.dsp.processor import _uses_fomo_min_resolution
    if _uses_fomo_min_resolution(imp):
        normalized_w = _snap_to_mult8(normalized_w)
        normalized_h = _snap_to_mult8(normalized_h)

    imp.image_width = normalized_w
    imp.image_height = normalized_h
    imp.resize_mode = normalized_resize

    if imp.dsp_blocks:
        imp.dsp_blocks = [
            {
                **block,
                "params": {
                    **(block.get("params") or {}),
                    "image_width": normalized_w,
                    "image_height": normalized_h,
                    "resize_mode": normalized_resize,
                },
            }
            if block.get("type") == "image"
            else block
            for block in (imp.dsp_blocks or [])
        ]


# ─── Routes ───────────────────────────────────────────────────────────────────

_DEFAULT_IMPULSE_NAME_RE = re.compile(r"^\s*Impulse\s+(\d+)\s*$", re.IGNORECASE)


def _clamp_subset_percent(value) -> float:
    """Coerce ``train_subset_percent`` to [1, 100]. Missing/garbage → 100."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return 100.0
    if v != v:  # NaN
        return 100.0
    return max(1.0, min(100.0, v))


def _next_default_impulse_name(db: Session, project: Project) -> str:
    """Return ``Impulse N`` where N is the next value of the persistent
    per-project counter. The counter never decreases — deleting an impulse
    does not free its number — so gaps are never refilled.

    Also bumps past any matching ``Impulse N`` name currently on the project
    (covers a user manually naming one e.g. ``Impulse 99``), and writes the
    advanced value back to ``project.impulse_seq``.

    Concurrency: callers must hold a row lock on ``project`` (via
    ``SELECT ... FOR UPDATE``) so two simultaneous creates cannot read the
    same counter and produce duplicate names.
    """
    current = int(project.impulse_seq or 0)
    existing = db.query(Impulse.name).filter(Impulse.project_id == project.id).all()
    for (name,) in existing:
        m = _DEFAULT_IMPULSE_NAME_RE.match(name or "")
        if m:
            n = int(m.group(1))
            if n > current:
                current = n
    next_n = current + 1
    project.impulse_seq = next_n
    return f"Impulse {next_n}"


# Defaults an impulse-create request omits, keyed by the parent project's
# `project_type`. Presentation-layer seeding only — see
# docs/Motion recognition/motion_phase0.md §3 / §8 task 5. A project type
# with no entry here (e.g. a legacy project_type value) seeds nothing, and
# `ImpulseCreate`'s own defaults apply exactly as today.
_MOTION_IMPULSE_SEED = {
    "input_type": "time-series",
    "sensor_type": "accelerometer",
    "window_size_ms": 1000,
    "window_increase_ms": 500,
    "frequency_hz": 62.5,
    "input_axes": ["accX", "accY", "accZ"],
}
_OBJECT_DETECTION_IMPULSE_SEED = {
    "input_type": "image",
    "sensor_type": "camera",
}
_IMPULSE_SEED_BY_PROJECT_TYPE = {
    "motion": _MOTION_IMPULSE_SEED,
    "object_detection": _OBJECT_DETECTION_IMPULSE_SEED,
}


def _seed_from_project_type(req: "ImpulseCreate", project: Project) -> dict:
    """Fields `req` omitted, filled in from the parent project's declared
    type. Gated on `model_fields_set`: `ImpulseCreate` declares non-Optional
    defaults for every one of these fields, so an omitted field is otherwise
    indistinguishable from one explicitly sent with the same value as the
    default — reading `model_fields_set` is the only way to tell them apart.
    An explicit client value always wins over the seed.
    """
    seed = _IMPULSE_SEED_BY_PROJECT_TYPE.get(project.project_type, {})
    return {k: v for k, v in seed.items() if k not in req.model_fields_set}


@router.post("/", status_code=201)
def create_impulse(
    req: ImpulseCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    # Serialize concurrent creates on the same project: lock the project row
    # so the read-max-then-insert sequence below cannot race against another
    # request issuing the same default name. Custom-named creates take the
    # lock too — it's cheap and keeps the path uniform.
    project = (
        db.query(Project)
        .filter(Project.id == req.project_id, Project.owner_id == current_user.id)
        .with_for_update()
        .first()
    )
    if project is None:
        raise HTTPException(404, "Project not found")

    incoming_name = (req.name or "").strip()
    if not incoming_name or _DEFAULT_IMPULSE_NAME_RE.match(incoming_name):
        resolved_name = _next_default_impulse_name(db, project)
    else:
        resolved_name = incoming_name

    seeded = _seed_from_project_type(req, project)

    imp = Impulse(
        project_id=        req.project_id,
        name=              resolved_name,
        description=       req.description,
        window_size_ms=    seeded.get("window_size_ms", req.window_size_ms),
        window_increase_ms=seeded.get("window_increase_ms", req.window_increase_ms),
        frequency_hz=      seeded.get("frequency_hz", req.frequency_hz),
        zero_pad_allowed=  req.zero_pad_allowed,
        input_type=        seeded.get("input_type", req.input_type),
        input_axes=        seeded.get("input_axes", req.input_axes),
        sensor_type=       seeded.get("sensor_type", req.sensor_type),
        image_width=       req.image_width,
        image_height=      req.image_height,
        resize_mode=       req.resize_mode,
        train_subset_percent=_clamp_subset_percent(req.train_subset_percent),
        dsp_blocks=        [_apply_defaults(b) for b in req.dsp_blocks],
        ml_blocks=         [_apply_defaults(b) for b in req.ml_blocks],
    )
    _sync_root_input_fields(
        imp,
        imp.dsp_blocks or [],
        # Read from the seeded value, not `req.input_type` directly — a
        # vision project's first impulse would otherwise seed input_type to
        # "image" here but still fail this check (req.input_type still
        # reads the ImpulseCreate default "time-series"), and
        # _sync_root_input_fields would immediately normalize it back.
        preserve_explicit_image=imp.input_type == "image",
        project_type=project.project_type,
    )
    _enforce_fomo_min_resolution(imp)
    db.add(imp)
    db.commit()
    db.refresh(imp)
    return _imp(imp)


@router.get("/project/{project_id}")
def list_impulses(
    project_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    assert_project_owner(db, project_id, current_user)
    rows = db.query(Impulse).filter(Impulse.project_id == project_id).all()
    return [_imp(i) for i in rows]


@router.get("/project/{project_id}/next-number")
def get_next_impulse_number(
    project_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Peek the next default impulse number — same arithmetic as
    ``_next_default_impulse_name`` but WITHOUT advancing ``project.impulse_seq``.
    The builder calls this on "New impulse" so the header can show "Impulse N"
    immediately, matching what the backend will assign at save time. Pure read,
    no row lock needed: a concurrent create that bumps the seq between peek and
    save just means the eventual save returns N+1 and the header rehydrates
    from the response — the persisted name and the displayed name still match.
    """
    project = assert_project_owner(db, project_id, current_user)
    current = int(project.impulse_seq or 0)
    existing = db.query(Impulse.name).filter(Impulse.project_id == project_id).all()
    for (name,) in existing:
        m = _DEFAULT_IMPULSE_NAME_RE.match(name or "")
        if m:
            n = int(m.group(1))
            if n > current:
                current = n
    next_n = current + 1
    return {"next_number": next_n, "next_name": f"Impulse {next_n}"}


@router.get("/{impulse_id}")
def get_impulse(
    impulse_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    i = assert_impulse_owner(db, impulse_id, current_user)

    result = _imp(i)

    # Attach latest training job summary
    latest_job = (
        db.query(TrainingJob)
        .filter(TrainingJob.impulse_id == impulse_id)
        .order_by(TrainingJob.created_at.desc())
        .first()
    )
    if latest_job:
        result["latest_job"] = {
            "id":            latest_job.id,
            "status":        latest_job.status,
            "best_accuracy": latest_job.best_accuracy,
            "completed_at":  latest_job.completed_at.isoformat() if latest_job.completed_at else None,
        }

    return result


@router.put("/{impulse_id}")
def update_impulse(
    impulse_id: str,
    req: ImpulseCreate,
    save_parameters: bool = False,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Update an impulse.

    The optional `save_parameters` query flag is the explicit signal that this
    PUT represents a confirmed "Save parameters" action from the user. Only
    then do we stamp `dsp_params_saved_at`, which downstream pages
    (Generate features) use as the gate. We never infer save-state from the
    mere presence of `image_width`/`image_height` in the payload, because
    those fields are populated by defaults at impulse creation time.
    """
    i = assert_impulse_owner(db, impulse_id, current_user)

    fields = req.dict(exclude={"project_id"})
    # Clamp subset percent to [1, 100] before persisting — protects against
    # client-side input bugs (negatives, NaN, 5000%) reaching the DB.
    if "train_subset_percent" in fields:
        fields["train_subset_percent"] = _clamp_subset_percent(fields["train_subset_percent"])
    # Apply default params to blocks before saving
    if "dsp_blocks" in fields:
        fields["dsp_blocks"] = [_apply_defaults(b) for b in fields["dsp_blocks"]]
    if "ml_blocks" in fields:
        fields["ml_blocks"] = [_apply_defaults(b) for b in fields["ml_blocks"]]

    # Keep root image_width/height in sync with the image DSP block params.
    # This is the single most common source of the "96×96 at training" bug:
    # the Parameters page saves image_width in dsp_block.params but the root
    # fields are never updated, so training still reads the stale root value.
    dsp_blocks = fields.get("dsp_blocks") or []
    for blk in dsp_blocks:
        if blk.get("type") == "image":
            blk_params = blk.get("params") or {}
            if blk_params.get("image_width"):
                fields["image_width"] = int(blk_params["image_width"])
            if blk_params.get("image_height"):
                fields["image_height"] = int(blk_params["image_height"])
            break

    for k, v in fields.items():
        setattr(i, k, v)

    if "dsp_blocks" in fields:
        _sync_root_input_fields(
            i,
            fields["dsp_blocks"],
            preserve_explicit_image=fields.get("input_type") == "image",
            project_type=i.project.project_type,
        )

    _enforce_fomo_min_resolution(i)

    # JSON columns are not tracked automatically — flag them so SQLAlchemy
    # flushes the mutation to Postgres even if the list object identity
    # did not change.
    from sqlalchemy.orm.attributes import flag_modified
    flag_modified(i, "dsp_blocks")
    if "dsp_blocks" in fields:
        flag_modified(i, "dsp_blocks")
    if "ml_blocks" in fields:
        flag_modified(i, "ml_blocks")

    i.updated_at = datetime.utcnow()

    if save_parameters and "dsp_blocks" in fields:
        i.dsp_params_saved_at = datetime.utcnow()

    db.commit()
    db.refresh(i)

    # Invalidate cached DSP features whenever the DSP pipeline config changes so the
    # feature explorer doesn't display stale projections after a parameter save.
    if "dsp_blocks" in fields:
        try:
            from app.core.storage import storage
            cache_key = storage.model_key(i.project_id, impulse_id, "features.npz")
            storage.delete_file(cache_key)
        except Exception:
            pass

    return _imp(i)


@router.patch("/{impulse_id}")
def patch_impulse(
    impulse_id: str,
    req: dict = Body(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    i = assert_impulse_owner(db, impulse_id, current_user)

    from sqlalchemy.orm.attributes import flag_modified

    allowed_fields = [
        "name", "description", "window_size_ms", "window_increase_ms",
        "frequency_hz", "zero_pad_allowed", "input_type", "input_axes",
        "sensor_type", "image_width", "image_height", "resize_mode",
        "train_subset_percent",
    ]

    for k, v in req.items():
        if k in allowed_fields:
            if k == "train_subset_percent":
                v = _clamp_subset_percent(v)
            setattr(i, k, v)

    _enforce_fomo_min_resolution(i)
    flag_modified(i, "dsp_blocks")

    i.updated_at = datetime.utcnow()
    db.commit()
    db.refresh(i)
    return _imp(i)


@router.patch("/{impulse_id}/blocks")
def patch_blocks(
    impulse_id: str,
    req: BlockPatch,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Add or remove a single block from the impulse.
    Keeps all other fields intact — safe for the pipeline canvas.
    """
    i = assert_impulse_owner(db, impulse_id, current_user)

    if req.block_kind == "dsp":
        blocks = list(i.dsp_blocks or [])
    else:
        blocks = list(i.ml_blocks or [])

    if req.action == "add":
        if not req.block:
            raise HTTPException(400, "block is required for action=add")
        blocks.append(_apply_defaults(req.block))
    elif req.action == "remove":
        if req.block_index is None:
            raise HTTPException(400, "block_index is required for action=remove")
        if req.block_index < 0 or req.block_index >= len(blocks):
            raise HTTPException(400, f"block_index {req.block_index} out of range")
        blocks.pop(req.block_index)

    if req.block_kind == "dsp":
        i.dsp_blocks = blocks
        # _sync_root_input_fields reads block params and may set image_width/height
        # from defaults (96). We override below with explicit user values, so this
        # call only matters for input_type / input_axes / sensor_type sync.
        _sync_root_input_fields(i, blocks, project_type=i.project.project_type)
        # The DSP block set itself changed — any prior Parameters save is no
        # longer valid for the new pipeline shape. Clear the gate so the user
        # has to re-save before Generate features unlocks again.
        i.dsp_params_saved_at = None
    else:
        i.ml_blocks = blocks

    # Apply user-provided root fields AFTER _sync_root_input_fields so they
    # always take priority over any defaults filled by _apply_defaults.
    # Also mirror them into the image DSP block params so training code
    # (which reads block.params.image_width) stays consistent.
    if req.window_size_ms is not None: i.window_size_ms = req.window_size_ms
    if req.window_increase_ms is not None: i.window_increase_ms = req.window_increase_ms
    if req.frequency_hz is not None: i.frequency_hz = req.frequency_hz
    if req.name is not None: i.name = req.name

    if req.image_width is not None:
        i.image_width = req.image_width
        if req.block_kind == "dsp":
            i.dsp_blocks = [
                {**b, "params": {**b.get("params", {}), "image_width": req.image_width}}
                if b.get("type") == "image" else b
                for b in (i.dsp_blocks or [])
            ]

    if req.image_height is not None:
        i.image_height = req.image_height
        if req.block_kind == "dsp":
            i.dsp_blocks = [
                {**b, "params": {**b.get("params", {}), "image_height": req.image_height}}
                if b.get("type") == "image" else b
                for b in (i.dsp_blocks or [])
            ]

    _enforce_fomo_min_resolution(i)

    from sqlalchemy.orm.attributes import flag_modified
    flag_modified(i, "dsp_blocks")
    flag_modified(i, "ml_blocks")

    i.updated_at = datetime.utcnow()
    db.commit()
    db.refresh(i)
    return _imp(i)


@router.post("/{impulse_id}/validate")
def validate_impulse(
    impulse_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Validate that an impulse is ready to be trained.
    Returns errors (blocking) and warnings (non-blocking).
    """
    i = assert_impulse_owner(db, impulse_id, current_user)

    errors, warnings = _validate_impulse(i)
    return {
        "valid":    len(errors) == 0,
        "errors":   errors,
        "warnings": warnings,
    }


class BulkImpulseAction(BaseModel):
    impulse_ids: List[str]


@router.post("/bulk-delete")
def bulk_delete_impulses(
    req: BulkImpulseAction,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Delete multiple impulses and their associated data."""
    deleted = 0
    failed_ids = []
    
    for iid in req.impulse_ids:
        try:
            # Ownership-checked lookup: non-owned / missing ids raise 404 and
            # are recorded as failures rather than silently deleted.
            i = assert_impulse_owner(db, iid, current_user)
            _delete_impulse_dependencies(db, iid)
            db.delete(i)
            db.commit()
            deleted += 1
        except HTTPException:
            failed_ids.append(iid)
        except Exception:
            db.rollback()
            failed_ids.append(iid)
            
    return {"status": "success", "deleted": deleted, "failed": failed_ids}


@router.delete("/{impulse_id}", status_code=204)
def delete_impulse(
    impulse_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    i = assert_impulse_owner(db, impulse_id, current_user)
    _delete_impulse_dependencies(db, impulse_id)
    db.delete(i)
    db.commit()
    return Response(status_code=204)
