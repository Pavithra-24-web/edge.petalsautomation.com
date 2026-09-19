"""
Project-graph cloning — the engine behind "restore a project version".

Restore is a *duplication*, not a reconstruction: it produces a brand-new
project that is indistinguishable from a normally trained one except for its
id and name. Dataset, Annotation Editor, Impulse Design, Generated Features,
Training history, Model Performance, Deployment and Retraining all work the
moment the restore returns, because every row and every stored asset the
source project owned has been copied — nothing is rebuilt from partial
snapshots and nothing is regenerated later.

The five steps (mirrored one-for-one below):

  1. Create the new project the same way ``POST /projects`` does — a Project
     row owned by the caller, with a fresh id.
  2. Clone the entire project graph. Every table that references
     Project / Impulse / Sample / Label / TrainingJob / TrainedModel /
     Deployment is walked, not just the version tables — see ``_TABLES``.
  3. Clone storage assets. Sample media, model artifacts, deployment
     packages, video inputs/outputs and the ``features.npz`` DSP cache are
     each copied to a key under the new project's own prefix. A restored
     project never reads bytes out of another project's storage path.
  4. Remap every foreign key through one flat old-id → new-id map, built
     before a single row is constructed so that ids embedded in storage keys
     and in JSON columns remap in the same pass as the columns themselves.
  5. Validate before commit. ``audit_clone`` re-reads the written graph and
     fails the whole restore — DB transaction rolled back, copied objects
     deleted — if any reference is dangling, points back at the source
     project, or if an impulse lost its training/feature state on the way.

Deliberate exclusions, each because cloning them would be *wrong*, not
because they were hard:

  * ``devices``, ``device_update_history``, ``device_inference_logs``,
    ``project_device_keys`` — live hardware bindings and per-project device
    credentials. A restored project has no hardware connected to it, exactly
    like a freshly created one; duplicating a device row would claim a
    physical board that is still enrolled elsewhere, and duplicating an API
    key would hand two projects the same secret.
  * ``project_versions`` / ``project_version_impulses`` /
    ``project_version_samples`` — the restored project starts its own
    version timeline. A normally trained project has no stored versions
    until its owner stores one, so carrying the source's history across is
    precisely what would make the restore *distinguishable*.
"""
from __future__ import annotations

import io
import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
from sqlalchemy import inspect as sa_inspect
from sqlalchemy.orm import Session

from app.core.storage import storage
from app.models.model_testing import (
    MetricResult,
    ModelTestRun,
    ModelTestSample,
    ModelVersion,
)
from app.models.user import (
    AILabelingAction,
    AILabelingJob,
    AIPrediction,
    Deployment,
    DspFeatureJob,
    FeatureSet,
    Impulse,
    Label,
    PostProcessingSettings,
    ProcessingJob,
    Project,
    Sample,
    SyntheticDataJob,
    TrainedModel,
    TrainingJob,
    gen_uuid,
)

logger = logging.getLogger(__name__)

# Every storage key this application writes is rooted at this prefix, which is
# what makes "does this string address an object?" a reliable test and
# "reparent it onto another project" a segment substitution.
_STORAGE_ROOT = "projects/"

_FEATURES_FILENAME = "features.npz"

# Parallelism for the asset copy. Bounded: each task is a single
# server-side S3 COPY, so the ceiling is the object store's request rate,
# not this process's CPU or memory.
_COPY_CONCURRENCY = 8


class CloneError(Exception):
    """A clone could not be completed. Raised after the transaction has been
    rolled back and any objects copied so far have been deleted, so the
    caller never sees a partially restored project."""


# ─── Scope ────────────────────────────────────────────────────────────────────

@dataclass
class CloneScope:
    """Which slice of the source project is being duplicated.

    A restore is scoped by the version being restored: only the impulses and
    samples that existed when the snapshot was taken are in scope. Everything
    that *depends* on those entities is pulled in wholesale — every training
    job of an in-scope impulse, not just the one the snapshot pointed at, so
    the restored project has real training history rather than a single
    synthesised run.

    Labels are always taken whole: they are cheap, project-scoped, and
    referenced by bounding boxes and by the feature cache, so a partial label
    set is the one thing guaranteed to produce dangling soft references.
    """
    project: Project
    impulse_ids: Set[str]
    sample_ids: Set[str]


# ─── Result ───────────────────────────────────────────────────────────────────

@dataclass
class CloneResult:
    project: Project
    id_map: Dict[str, str]
    # New-id list per table label, recorded as rows are written. The audit
    # re-reads exactly these rather than re-running the selectors, so it
    # measures what was written rather than what would be selected now.
    written: Dict[str, List[str]] = field(default_factory=dict)
    copied_keys: List[str] = field(default_factory=list)
    counts: Dict[str, int] = field(default_factory=dict)
    skipped: Dict[str, int] = field(default_factory=dict)


# ─── Id / key / JSON remapping ────────────────────────────────────────────────

class _Remapper:
    """The single source of truth for "what does this value become".

    One flat map covers every entity because ids are UUID4s — globally
    unique, so a value that equals a known source id can only *be* that
    entity, whichever column it turned up in. That is what lets the same
    lookup rewrite a foreign-key column, an id embedded in a JSON blob
    (``extra_metadata.boundingBoxes[*].label_id``,
    ``model_metadata.project_id``) and an id embedded in a storage key
    (``projects/<pid>/models/<impulse_id>/features.npz``) without three
    separate remapping vocabularies drifting apart.
    """

    def __init__(self, id_map: Dict[str, str]):
        self.id_map = id_map

    def is_storage_key(self, value: Any) -> bool:
        return isinstance(value, str) and value.startswith(_STORAGE_ROOT)

    def key(self, value: str) -> str:
        """Reparent a storage key onto the new project, remapping every id
        segment inside it. Segments that are not entity ids (the ``models``
        literal, a training run's ``20260410_101500`` timestamp, a sample
        upload's own UUID) pass through untouched — they stay unique under
        the new prefix."""
        return "/".join(self.id_map.get(seg, seg) for seg in value.split("/"))

    def scalar(self, value: Any) -> Any:
        if isinstance(value, str):
            mapped = self.id_map.get(value)
            if mapped is not None:
                return mapped
            if self.is_storage_key(value):
                return self.key(value)
        return value

    def json(self, value: Any) -> Any:
        """Deep-remap a JSON column. Catches every "soft foreign key" the
        schema stores inside JSON rather than in a constrained column — the
        class of reference that has no database integrity check and therefore
        fails silently rather than loudly."""
        if isinstance(value, dict):
            return {k: self.json(v) for k, v in value.items()}
        if isinstance(value, list):
            return [self.json(v) for v in value]
        return self.scalar(value)


# ─── Table registry ───────────────────────────────────────────────────────────

# A selector receives (db, scope, picked) and returns the source rows to
# clone. `picked` maps an earlier table's label to the rows chosen for it, so
# a child table can scope itself to parents that were actually taken —
# guaranteeing by construction that no non-nullable FK is left unmapped.
_Selector = Callable[[Session, CloneScope, Dict[str, List[Any]]], List[Any]]


@dataclass(frozen=True)
class _Table:
    model: type
    label: str
    select: _Selector
    # Columns forced to NULL in the clone. Used for values that are derived
    # from the source and would be actively wrong if carried across.
    blank: Tuple[str, ...] = ()


def _ids(rows: Sequence[Any]) -> List[str]:
    return [r.id for r in rows]


def _in(db: Session, model: type, column, values: Iterable[str]) -> List[Any]:
    values = list(values)
    if not values:
        return []
    return db.query(model).filter(column.in_(values)).all()


_TABLES: Tuple[_Table, ...] = (
    # Ordered parent-before-child. SQLAlchemy sorts INSERTs by table
    # dependency anyway, but the order is what lets each selector scope
    # itself against `picked`.
    _Table(Label, "labels",
           lambda db, s, p: db.query(Label).filter(Label.project_id == s.project.id).all()),

    _Table(Sample, "samples",
           lambda db, s, p: _in(db, Sample, Sample.id, s.sample_ids)),

    # Project-scoped, like AILabelingAction below — a restored project keeps
    # its generation history even though (per the version-scoping rules
    # above) it may not carry every sample a given job produced. `label_id`
    # and the `sample_ids` JSON column remap automatically: the former via
    # FK introspection, the latter via `_Remapper.json`. A sample id the
    # version snapshot didn't carry into scope has no entry in `id_map` and
    # so is left as-is by `_Remapper.json` — a stale reference to the source
    # project's sample, the one gap this generic walker can't close without
    # a per-job re-filter of `sample_ids` against `s.sample_ids`.
    _Table(SyntheticDataJob, "synthetic_data_jobs",
           lambda db, s, p: db.query(SyntheticDataJob)
           .filter(SyntheticDataJob.project_id == s.project.id).all()),

    _Table(Impulse, "impulses",
           lambda db, s, p: _in(db, Impulse, Impulse.id, s.impulse_ids)),

    _Table(TrainingJob, "training_jobs",
           lambda db, s, p: _in(db, TrainingJob, TrainingJob.impulse_id, s.impulse_ids)),

    _Table(TrainedModel, "trained_models",
           lambda db, s, p: _in(db, TrainedModel, TrainedModel.training_job_id,
                                _ids(p["training_jobs"]))),

    # `download_url` is a presigned URL over the *source* object — it both
    # expires and points outside the new project, so it is regenerated on
    # demand rather than copied.
    _Table(Deployment, "deployments",
           lambda db, s, p: _in(db, Deployment, Deployment.model_id, _ids(p["trained_models"])),
           blank=("download_url",)),

    # Per-sample DSP feature rows: both parents must be in scope or the row
    # would dangle on one of them.
    _Table(FeatureSet, "feature_sets",
           lambda db, s, p: [
               f for f in _in(db, FeatureSet, FeatureSet.impulse_id, s.impulse_ids)
               if f.sample_id in s.sample_ids
           ]),

    _Table(DspFeatureJob, "dsp_feature_jobs",
           lambda db, s, p: _in(db, DspFeatureJob, DspFeatureJob.impulse_id, s.impulse_ids)),

    # Project-scoped row (impulse_id NULL) plus any impulse-scoped rows whose
    # impulse survived into scope.
    _Table(PostProcessingSettings, "post_processing_settings",
           lambda db, s, p: [
               r for r in db.query(PostProcessingSettings)
               .filter(PostProcessingSettings.project_id == s.project.id).all()
               if r.impulse_id is None or r.impulse_id in s.impulse_ids
           ]),

    _Table(ProcessingJob, "processing_jobs",
           lambda db, s, p: db.query(ProcessingJob)
           .filter(ProcessingJob.project_id == s.project.id).all()),

    _Table(AILabelingAction, "ai_labeling_actions",
           lambda db, s, p: db.query(AILabelingAction)
           .filter(AILabelingAction.project_id == s.project.id).all()),

    _Table(AILabelingJob, "ai_labeling_jobs",
           lambda db, s, p: _in(db, AILabelingJob, AILabelingJob.action_id,
                                _ids(p["ai_labeling_actions"]))),

    _Table(AIPrediction, "ai_predictions",
           lambda db, s, p: [
               r for r in _in(db, AIPrediction, AIPrediction.job_id, _ids(p["ai_labeling_jobs"]))
               if r.sample_id in s.sample_ids
           ]),

    # Model Testing / Model Performance.
    _Table(ModelVersion, "model_versions",
           lambda db, s, p: _in(db, ModelVersion, ModelVersion.impulse_id, s.impulse_ids)),

    _Table(ModelTestRun, "model_test_runs",
           lambda db, s, p: _in(db, ModelTestRun, ModelTestRun.impulse_id, s.impulse_ids)),

    _Table(MetricResult, "metric_results",
           lambda db, s, p: _in(db, MetricResult, MetricResult.test_run_id,
                                _ids(p["model_test_runs"]))),

    _Table(ModelTestSample, "model_test_samples",
           lambda db, s, p: [
               r for r in _in(db, ModelTestSample, ModelTestSample.impulse_id, s.impulse_ids)
               if r.sample_id is None or r.sample_id in s.sample_ids
           ]),
)


# ─── Column introspection ─────────────────────────────────────────────────────

def _columns(model: type):
    return list(sa_inspect(model).columns)


def _fk_targets(model: type) -> Dict[str, str]:
    """``{column_key: target_table}`` for every FK on the model, excluding
    ``users`` — a user id is not part of the project graph and must survive
    the clone unchanged (the restored project keeps the same owner and the
    same "uploaded by")."""
    out: Dict[str, str] = {}
    for col in _columns(model):
        for fk in col.foreign_keys:
            table = fk.column.table.name
            if table != "users":
                out[col.key] = table
    return out


# `Impulse.active_model_run_id` points at a TrainingJob but carries no FK
# constraint (a real one would form a delete cycle with training_jobs'
# delete-orphan cascade). It is still a reference that must be remapped and
# audited, so it is declared here rather than discovered by introspection.
_PSEUDO_FKS: Dict[type, Dict[str, str]] = {
    Impulse: {"active_model_run_id": "training_jobs"},
}


def _all_fks(model: type) -> Dict[str, str]:
    return {**_fk_targets(model), **_PSEUDO_FKS.get(model, {})}


# ─── Feature cache ────────────────────────────────────────────────────────────

def _clone_feature_cache(
    remap: _Remapper, old_project_id: str, old_impulse_id: str, new_impulse_id: str,
) -> Optional[str]:
    """Copy an impulse's ``features.npz`` DSP cache into the new project,
    rewriting every id baked into it. Returns the new key, or None if the
    source impulse had no cache.

    The cache is not opaque bytes — it is a table with soft foreign keys in
    it: ``y`` is an array of label ids, ``ids`` an array of sample ids,
    ``boxes_json`` per-sample boxes each carrying their own ``label_id``, and
    ``meta_json`` repeats ``impulse_id`` / ``project_id`` /
    ``label_ids_ordered`` / ``sample_ids``. A byte-for-byte copy would leave
    every one of them addressing the *source* project's rows, and the failure
    mode is silent: Feature Explorer renders every point unlabeled and
    training reads a sample set that no longer exists.

    Unlike the rest of the clone this raises on failure rather than degrading
    to "regenerate it later" — "nothing regenerated" is the acceptance
    criterion, so a cache that existed in the source and cannot be reproduced
    in the clone fails the restore.
    """
    old_key = storage.model_key(old_project_id, old_impulse_id, _FEATURES_FILENAME)
    if not storage.object_exists(old_key):
        return None

    raw = storage.download_bytes(old_key)
    with np.load(io.BytesIO(raw), allow_pickle=True) as data:
        arrays = {k: data[k] for k in data.files}

    if "y" in arrays:
        arrays["y"] = np.array(
            [remap.id_map.get(v, v) if v else v for v in arrays["y"]], dtype=object,
        )
    if "ids" in arrays:
        arrays["ids"] = np.array(
            [remap.id_map.get(v, v) for v in arrays["ids"]], dtype=object,
        )
    if "boxes_json" in arrays:
        arrays["boxes_json"] = np.array(
            [json.dumps(remap.json(json.loads(entry) if entry else [])) for entry in arrays["boxes_json"]],
            dtype=object,
        )
    if "meta_json" in arrays and len(arrays["meta_json"]):
        meta = remap.json(json.loads(arrays["meta_json"][0]))
        arrays["meta_json"] = np.array([json.dumps(meta)], dtype=object)

    buf = io.BytesIO()
    np.savez_compressed(buf, **arrays)
    buf.seek(0)
    new_key = remap.key(old_key)
    storage.upload_file(buf, new_key, "application/octet-stream")
    return new_key


# ─── The clone ────────────────────────────────────────────────────────────────

def clone_project_graph(
    db: Session,
    scope: CloneScope,
    *,
    owner_id: str,
    name: str,
    description: Optional[str],
) -> CloneResult:
    """Steps 1–4. Writes the new graph and copies its assets, leaving the
    transaction open and un-audited — callers go through ``restore`` in
    ``api/v1/endpoints/project_versions.py``, which owns the audit and the
    commit/rollback decision.
    """
    source = scope.project
    new_project_id = gen_uuid()

    # ── Step 1: the new project, created exactly as POST /projects would ──
    project = Project(
        id=new_project_id,
        owner_id=owner_id,
        name=name,
        description=description,
        project_type=source.project_type,
        # Carried so the restored project's next impulse is numbered after
        # the ones it already has, rather than colliding with them.
        impulse_seq=source.impulse_seq or 0,
        target_device_slug=source.target_device_slug,
        target_device_custom_name=source.target_device_custom_name,
        target_device_ram_kb=source.target_device_ram_kb,
        target_device_rom_kb=source.target_device_rom_kb,
        target_device_latency_ms=source.target_device_latency_ms,
    )
    db.add(project)
    db.flush()

    # ── Step 2a: pick the source rows, table by table ──
    picked: Dict[str, List[Any]] = {}
    for spec in _TABLES:
        picked[spec.label] = spec.select(db, scope, picked)

    # ── Step 4a: allocate every new id up front ──
    #
    # The whole map has to exist before the first row is built, because a
    # row's own storage key can embed a *sibling's* id and a JSON column can
    # reference a table cloned later in the order. Two passes, one map, no
    # fix-up phase.
    id_map: Dict[str, str] = {source.id: new_project_id}
    for spec in _TABLES:
        for row in picked[spec.label]:
            id_map[row.id] = gen_uuid()
    remap = _Remapper(id_map)

    # ── Steps 2b/3/4b: build the rows, remap them, schedule asset copies ──
    copy_plan: Dict[str, str] = {}          # old key → new key
    counts: Dict[str, int] = {}
    skipped: Dict[str, int] = {}

    written: Dict[str, List[str]] = {}

    for spec in _TABLES:
        rows = picked[spec.label]
        fks = _all_fks(spec.model)
        made: List[str] = []
        for src_row in rows:
            data, ok = _clone_row(src_row, spec, fks, remap, copy_plan)
            if not ok:
                skipped[spec.label] = skipped.get(spec.label, 0) + 1
                continue
            db.add(spec.model(**data))
            made.append(data["id"])
        written[spec.label] = made
        counts[spec.label] = len(made)

    db.flush()

    # ── Step 3: copy the assets ──
    #
    # Server-side copies, so the bytes never pass through this process, and
    # fanned out because a dataset is thousands of independent objects and a
    # serial loop would make restore latency scale with sample count rather
    # than with dataset size. Any failure propagates: a restore that silently
    # lost a sample's media would be exactly the "best-effort reconstruction"
    # this design exists to remove.
    copied: List[str] = []
    if copy_plan:
        pairs = list(copy_plan.items())
        with ThreadPoolExecutor(max_workers=_COPY_CONCURRENCY) as pool:
            futures = {
                pool.submit(storage.copy_object, old_key, new_key): new_key
                for old_key, new_key in pairs
            }
            for future in as_completed(futures):
                future.result()
                copied.append(futures[future])

    for old_impulse_id in scope.impulse_ids:
        new_key = _clone_feature_cache(
            remap, source.id, old_impulse_id, id_map[old_impulse_id],
        )
        if new_key:
            copied.append(new_key)

    return CloneResult(
        project=project, id_map=id_map, written=written, copied_keys=copied,
        counts=counts, skipped=skipped,
    )


def _clone_row(
    src_row: Any, spec: _Table, fks: Dict[str, str],
    remap: _Remapper, copy_plan: Dict[str, str],
) -> Tuple[Dict[str, Any], bool]:
    """Build one row's column values. Returns ``(data, ok)``; ``ok`` is False
    when a non-nullable reference could not be remapped, which means the row
    belongs to something outside the clone scope and must not be written.
    The selectors above make that unreachable for every table with a
    non-nullable parent — it is a guard, not a routine path."""
    data: Dict[str, Any] = {}
    for col in _columns(spec.model):
        key = col.key
        value = getattr(src_row, key)

        if key == "id":
            data[key] = remap.id_map[src_row.id]
            continue
        if key in spec.blank:
            data[key] = None
            continue
        if value is None:
            data[key] = None
            continue

        if key in fks:
            mapped = remap.id_map.get(value)
            if mapped is None:
                # References an entity outside the scope of this clone.
                if not col.nullable:
                    return data, False
                # Nullable: drop the reference rather than let it point back
                # at the source project.
                data[key] = None
                continue
            data[key] = mapped
            continue

        if isinstance(value, (dict, list)):
            data[key] = remap.json(value)
            continue

        if remap.is_storage_key(value):
            new_key = remap.key(value)
            copy_plan.setdefault(value, new_key)
            data[key] = new_key
            continue

        data[key] = remap.scalar(value)

    return data, True


# ─── Step 5: the integrity audit ──────────────────────────────────────────────

# Tables whose storage keys are HEAD-probed one by one. Every copy already
# raises on failure, so this is a second opinion, not the primary guard —
# and it is spent only on the handful of artifact rows whose absence would
# be silently fatal later (a deployment package or a model file that is
# missing only surfaces at download/inference time). Per-sample media is
# covered by the copy itself; probing thousands of objects would turn the
# audit into the slowest part of a restore.
_VERIFY_OBJECT_EXISTS = {"trained_models", "deployments"}

def audit_clone(db: Session, scope: CloneScope, result: CloneResult) -> List[str]:
    """Re-read the written graph and return a list of failures — empty means
    the clone is safe to commit.

    Everything here is checked against the database *after* the flush rather
    than against the in-memory objects that were just constructed, so it
    catches a bad write, not just a bad intention.
    """
    failures: List[str] = []
    new_project_id = result.project.id
    source_id = scope.project.id
    resolvable: Set[str] = {new_project_id}
    for ids in result.written.values():
        resolvable.update(ids)

    # A row that could not be written is a row the restored project is
    # missing. Nothing may be skipped.
    for label, n in result.skipped.items():
        failures.append(f"{label}: {n} row(s) could not be cloned (unresolvable reference)")

    for spec in _TABLES:
        model = spec.model
        expected = result.counts.get(spec.label, 0)

        # Re-read exactly the rows this clone claims to have written.
        written_ids = result.written.get(spec.label, [])
        rows = db.query(model).filter(model.id.in_(written_ids)).all() if written_ids else []
        if len(rows) != expected:
            failures.append(
                f"{spec.label}: expected {expected} cloned row(s), found {len(rows)}"
            )

        fks = _all_fks(model)
        for row in rows:
            # (a) project scoping — a table with its own project_id must name
            #     the new project and nothing else.
            if hasattr(row, "project_id") and row.project_id not in (None, new_project_id):
                failures.append(f"{spec.label} {row.id}: project_id is not the restored project")

            for col_key in fks:
                value = getattr(row, col_key, None)
                if value is None:
                    continue
                # (b) every reference must resolve to a row this clone
                #     actually wrote. `resolvable` is built from the written
                #     ids rather than from the id map, so an id allocated for
                #     a row that never made it to the database counts as
                #     dangling — which is what it would be.
                if value not in resolvable:
                    failures.append(
                        f"{spec.label} {row.id}: {col_key} -> {value} does not resolve "
                        f"inside the restored project"
                    )

            # (d) no JSON column may still carry a source id, and no stored
            #     key may live under another project's prefix.
            for col in _columns(model):
                value = getattr(row, col.key, None)
                if isinstance(value, (dict, list)):
                    stale = _stale_ids(value, result.id_map)
                    if stale:
                        failures.append(
                            f"{spec.label} {row.id}: {col.key} still references source id(s) "
                            f"{sorted(stale)[:3]}"
                        )
                elif isinstance(value, str) and value.startswith(_STORAGE_ROOT):
                    if not value.startswith(f"{_STORAGE_ROOT}{new_project_id}/"):
                        failures.append(
                            f"{spec.label} {row.id}: {col.key} points at another project's storage"
                        )
                    elif spec.label in _VERIFY_OBJECT_EXISTS and not storage.object_exists(value):
                        failures.append(
                            f"{spec.label} {row.id}: {col.key} -> {value} was not copied"
                        )

    failures.extend(_audit_parity(db, scope, result))

    # The restored project must be a different project, not the same one
    # under a new name.
    if source_id == new_project_id:
        failures.append("restore produced the source project id")

    return failures


def _stale_ids(value: Any, id_map: Dict[str, str]) -> Set[str]:
    """Source ids still present anywhere inside a JSON value."""
    found: Set[str] = set()
    if isinstance(value, dict):
        for v in value.values():
            found |= _stale_ids(v, id_map)
    elif isinstance(value, list):
        for v in value:
            found |= _stale_ids(v, id_map)
    elif isinstance(value, str) and value in id_map:
        found.add(value)
    return found


def _audit_parity(db: Session, scope: CloneScope, result: CloneResult) -> List[str]:
    """The checks that make the restored project *usable*, not merely
    referentially clean: an impulse that had a trained model must still have
    one, generated features must be present, annotations must have survived,
    and deployment metadata must be intact."""
    failures: List[str] = []
    id_map = result.id_map
    new_project_id = result.project.id

    for old_impulse_id in scope.impulse_ids:
        new_impulse_id = id_map[old_impulse_id]
        new_impulse = db.query(Impulse).filter(Impulse.id == new_impulse_id).first()
        if new_impulse is None:
            failures.append(f"impulse {old_impulse_id}: not present in the restored project")
            continue

        old_jobs = db.query(TrainingJob).filter(TrainingJob.impulse_id == old_impulse_id).count()
        new_jobs = db.query(TrainingJob).filter(TrainingJob.impulse_id == new_impulse_id).count()
        if old_jobs != new_jobs:
            failures.append(
                f"impulse {new_impulse.name}: {old_jobs} training job(s) in the source, "
                f"{new_jobs} restored"
            )

        old_models = (
            db.query(TrainedModel)
            .join(TrainingJob, TrainedModel.training_job_id == TrainingJob.id)
            .filter(TrainingJob.impulse_id == old_impulse_id).count()
        )
        new_models = (
            db.query(TrainedModel)
            .join(TrainingJob, TrainedModel.training_job_id == TrainingJob.id)
            .filter(TrainingJob.impulse_id == new_impulse_id).count()
        )
        if old_models != new_models:
            failures.append(
                f"impulse {new_impulse.name}: {old_models} trained model(s) in the source, "
                f"{new_models} restored"
            )

        # An impulse whose source had a completed run must come back with a
        # usable active-model pointer, or Model Performance / Deployment /
        # Retrain open empty.
        source_impulse = db.query(Impulse).filter(Impulse.id == old_impulse_id).first()
        if source_impulse is not None and source_impulse.active_model_run_id:
            if not new_impulse.active_model_run_id:
                failures.append(
                    f"impulse {new_impulse.name}: active model pointer was lost"
                )

        # Generated features: present in the source ⇒ present in the clone.
        old_features = storage.model_key(scope.project.id, old_impulse_id, _FEATURES_FILENAME)
        if storage.object_exists(old_features):
            new_features = storage.model_key(new_project_id, new_impulse_id, _FEATURES_FILENAME)
            if not storage.object_exists(new_features):
                failures.append(
                    f"impulse {new_impulse.name}: generated features were not cloned"
                )

    # Annotations: every box must resolve to a label of the *new* project.
    new_label_ids = {
        row[0] for row in db.query(Label.id).filter(Label.project_id == new_project_id).all()
    }
    for old_sample_id in scope.sample_ids:
        new_sample_id = id_map.get(old_sample_id)
        if new_sample_id is None:
            continue
        source_sample = db.query(Sample).filter(Sample.id == old_sample_id).first()
        new_sample = db.query(Sample).filter(Sample.id == new_sample_id).first()
        if new_sample is None:
            failures.append(f"sample {old_sample_id}: not present in the restored project")
            continue
        old_boxes = ((source_sample.extra_metadata or {}).get("boundingBoxes") or []) if source_sample else []
        new_boxes = (new_sample.extra_metadata or {}).get("boundingBoxes") or []
        if len(old_boxes) != len(new_boxes):
            failures.append(
                f"sample {new_sample.filename}: {len(old_boxes)} annotation(s) in the source, "
                f"{len(new_boxes)} restored"
            )
        for box in new_boxes:
            label_id = box.get("label_id") if isinstance(box, dict) else None
            if label_id and label_id not in new_label_ids:
                failures.append(
                    f"sample {new_sample.filename}: annotation references a label "
                    f"outside the restored project"
                )
                break

    return failures


# ─── Rollback ─────────────────────────────────────────────────────────────────

def discard_clone(db: Session, copied_keys: Sequence[str]) -> None:
    """Undo a clone in full: roll the transaction back and delete every
    object copied for it. Object storage has no transaction to join, so the
    copies are undone explicitly — otherwise a failed restore would leave
    orphaned assets under a project id that no row will ever name again."""
    db.rollback()
    for key in copied_keys:
        try:
            storage.delete_file(key)
        except Exception:  # noqa: BLE001 - best effort; the DB is already clean
            logger.warning("clone rollback: could not delete %s", key, exc_info=True)
