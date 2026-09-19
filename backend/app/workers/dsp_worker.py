"""
DSP Worker — Celery task for background feature extraction.

Generates a rich .npz feature cache (v2) that the training worker can validate
and reuse, eliminating redundant DSP computation at training time.

Cache format (v2)
-----------------
  X            float32 array  (N, *feature_shape)   — DSP output per sample
  y            object  array  (N,)                  — label UUIDs (for Feature Explorer)
  ids          object  array  (N,)                  — sample UUIDs
  sample_types object  array  (N,)                  — "training"/"testing"/"automatic"
  boxes_json   object  array  (N,)                  — JSON-serialised bounding boxes
                                                       (empty "[]" for non-detection samples)
  meta_json    object  array  (1,)                  — JSON validity-metadata blob

Backward-compatible: Feature Explorer only reads X, y, ids which are unchanged.
"""
import io
import json
import time
import hashlib
import logging
import numpy as np
from datetime import datetime

from app.core.logging_config import get_logger as _get_channel_logger, log_event as _log_event
from app.workers.celery_app import celery_app
from app.core.database import SessionLocal
from app.core.storage import storage
from app.models.user import Impulse, Sample, Label, DspFeatureJob, JobStatus
from app.ml.dsp.processor import DSPProcessor, merge_image_params
from app.workers.sample_utils import (
    UNLABELED_NAMES,
    apply_train_subset,
    is_background_sample,
    label_name_is_placeholder,
    is_sample_usable,
    iter_prefetched,
    normalize_bounding_boxes,
)
from app.workers.cancel_utils import CancelledError, raise_if_cancelled_fast, assert_not_cancelled_before_completing

logger = logging.getLogger(__name__)

# Structured DSP metrics → logs/dsp.log
_dsp_log = _get_channel_logger("dsp")


# ─── Shared helpers (also imported by training_worker) ────────────────────────

# Feature-cache format version.
#
#   3 → boxes normalised to [0,1]; background images excluded from the cache.
#   4 → background ("negative") images are included, with an explicit per-sample
#       `background` flag array and a `background_count` in the metadata.
#
# The training worker refuses any cache older than this, so a detection model
# can never be trained on a cache that silently has its negatives missing.
FEATURE_CACHE_VERSION: int = 4


def compute_dsp_config_hash(dsp_blocks: list) -> str:
    """
    Stable MD5 hash of the DSP block configuration.

    Used by both the DSP worker (writing cache) and the training worker
    (validating cache) so any change to block type or params invalidates it.
    """
    serialized = json.dumps(dsp_blocks or [], sort_keys=True, default=str)
    return hashlib.md5(serialized.encode()).hexdigest()


def make_features_storage_key(project_id: str, impulse_id: str) -> str:
    """Canonical S3 key for the feature cache of a given impulse."""
    return storage.model_key(project_id, impulse_id, "features.npz")


def _combine_dsp_features(feature_list):
    if not feature_list:
        return np.empty((0,), dtype=np.float32)
    if len(feature_list) == 1:
        return np.asarray(feature_list[0], dtype=np.float32)
    return np.concatenate([np.asarray(f, dtype=np.float32).flatten() for f in feature_list])


# ─── Celery task ──────────────────────────────────────────────────────────────

@celery_app.task(bind=True, name="app.workers.dsp_worker.extract_features_for_impulse")
def extract_features_for_impulse(self, impulse_id: str, job_id: str | None = None):
    """
    Pre-compute and persist DSP features for ALL samples in an impulse.

    job_id is the DspFeatureJob row created at /generate-features dispatch
    time; we flip it to running/completed/failed so the /jobs endpoint can
    show real status, started_at, and completed_at. It is optional purely
    so legacy callers that still pass only impulse_id keep working —
    when None, only the Celery state is updated (old behavior).

    Key improvements over the v1 stub:
    • Processes every project sample, not just top-level-labelled ones — so
      FOMO/detection images annotated via bounding boxes are included.
    • Stores per-sample bounding-box data (boxes_json) and split-type
      (sample_types) so the training worker can rebuild FOMO heatmaps
      without touching raw files.
    • Writes a meta_json blob the training worker uses to detect staleness
      (DSP config change, image-dimension change, new samples added).
    """
    logger.info(f"[DSP] Feature extraction started for impulse {impulse_id} (job {job_id})")
    _dsp_t0 = time.perf_counter()
    db = SessionLocal()
    try:
        impulse = db.query(Impulse).filter(Impulse.id == impulse_id).first()
        if not impulse:
            raise ValueError(f"Impulse {impulse_id} not found")

        # Flip the DB row to running. Wrapped in its own try so a missing/
        # invalid job_id (legacy dispatch) doesn't fail the task.
        if job_id:
            job = db.query(DspFeatureJob).filter(DspFeatureJob.id == job_id).first()
            if job is not None:
                job.status = JobStatus.running
                job.started_at = datetime.utcnow()
                db.commit()

        self.update_state(state="RUNNING", meta={"status": "running", "progress": 0})

        dsp_blocks = impulse.dsp_blocks or [{"type": "raw", "params": {}}]
        dsp_hash   = compute_dsp_config_hash(dsp_blocks)

        # Canonical image dimensions (same priority as training worker + processor)
        _img_block: dict = {"params": {}}
        for blk in dsp_blocks:
            if blk.get("type") == "image":
                _img_block = blk
                break
        merged_img_params = merge_image_params(impulse, _img_block)
        image_width  = int(merged_img_params.get("image_width")  or impulse.image_width  or 96)
        image_height = int(merged_img_params.get("image_height") or impulse.image_height or 96)

        # ── Fetch ALL samples ─────────────────────────────────────────────────
        # Detection impulses get labels from per-box annotations, not from the
        # top-level label_id field, so we must not filter by label_id here.
        samples = (
            db.query(Sample)
            .filter(Sample.project_id == impulse.project_id)
            .all()
        )

        # ── Train-on-data-subset ──────────────────────────────────────────────
        # Random subsample of the training split per impulse.train_subset_percent.
        # apply_train_subset is a no-op when percent >= 100, and is shared with
        # the training worker's inline-recompute path so both end up on the
        # same deterministic subset.
        subset_pct = float(getattr(impulse, "train_subset_percent", 100.0) or 100.0)
        if subset_pct < 100.0:
            before = len(samples)
            samples = apply_train_subset(samples, impulse)
            logger.info(
                f"[DSP] train_subset_percent={subset_pct:.1f} → "
                f"{len(samples)}/{before} samples after subsampling"
            )

        # ── Label registry — active, non-placeholder labels only ─────────────
        # Collect label IDs/names actually referenced by samples, then filter
        # to labels with real names.  Placeholder labels ("Unlabeled", etc.)
        # and labels only referenced by placeholder-keyed boxes are excluded so
        # they do not inflate the class count or appear in cache metadata.
        _active_ids: set = set()
        for _s in samples:
            if _s.label_id:
                _active_ids.add(_s.label_id)
            if isinstance(_s.extra_metadata, dict):
                for _box in _s.extra_metadata.get("boundingBoxes", []):
                    _key = _box.get("label_id") or _box.get("label")
                    # Only register box keys that are not placeholder names
                    if _key and str(_key).lower() not in UNLABELED_NAMES:
                        _active_ids.add(_key)

        all_labels = db.query(Label).filter(Label.project_id == impulse.project_id).all()
        labels = [
            l for l in all_labels
            if (l.id in _active_ids or l.name in _active_ids)
            and not label_name_is_placeholder(l.name)
        ]
        label_names     = [l.name for l in labels]
        label_ids_order = [l.id   for l in labels]
        # Dual-key map: UUID → UUID and name → UUID (for box label resolution)
        lbl_map: dict = {}
        for lbl in labels:
            lbl_map[lbl.id]   = lbl.id
            lbl_map[lbl.name] = lbl.id

        # Pre-compute the set of label IDs with non-placeholder names so the
        # sample-usability check below can exclude samples whose label_id points
        # to an "Unlabeled" / placeholder label row.
        usable_label_ids: set = {l.id for l in labels}

        total = len(samples)
        logger.info(f"[DSP] Processing {total} total samples for impulse {impulse_id}")

        # ── Feature extraction loop ───────────────────────────────────────────
        features_list:     list = []
        label_ids_list:    list = []
        sample_ids_list:   list = []
        sample_types_list: list = []
        boxes_json_list:   list = []
        background_list:   list = []

        # Skip samples with no usable training signal up front (no real label_id,
        # or label_id pointing to a placeholder, and no usable box) so we never
        # download their bytes. The remaining usable samples then have their raw
        # bytes fetched concurrently — S3 latency-bound, so overlapping fetches
        # collapses total wait — while results are still consumed in input order
        # so the persisted arrays are byte-for-byte identical to the serial path.
        # allow_background=True unconditionally: the feature cache is shared by
        # every architecture an impulse might train, so it has to be a superset.
        # A background sample lands in the cache with label_id "" and boxes "[]",
        # which the classification reader already skips (its label is not in
        # label_map) and the detection readers now consume as a negative.
        # Building a detection-only cache would make cache validity depend on
        # which model was trained last.
        usable_samples = [
            s for s in samples
            if is_sample_usable(s, usable_label_ids, allow_background=True)
        ]
        skipped_unlabeled = total - len(usable_samples)

        processed = 0
        for sample, raw_bytes, fetch_error in iter_prefetched(
            usable_samples, lambda s: storage.download_bytes(s.storage_key)
        ):
            if job_id:
                raise_if_cancelled_fast(job_id)
            try:
                if fetch_error is not None:
                    raise fetch_error

                boxes_pre = normalize_bounding_boxes(
                    (sample.extra_metadata or {}).get("boundingBoxes", []),
                    image_bytes=raw_bytes,
                )

                all_features = []
                for block_cfg in dsp_blocks:
                    params = merge_image_params(impulse, block_cfg)
                    proc = DSPProcessor(
                        block_type=block_cfg.get("type", "raw"),
                        params=params,
                        frequency_hz=sample.frequency_hz or impulse.frequency_hz or 100.0,
                    )
                    all_features.append(proc.extract(raw_bytes))

                features = _combine_dsp_features(all_features)

                # Bounding boxes (already fetched above for the skip-check)
                boxes = boxes_pre

                # For detection samples, keep the feature-point label aligned
                # with the box annotation shown in the UI by preferring the
                # first bounding-box label over the sample's top-level label.
                top_label_id = ""
                if boxes:
                    box_key = boxes[0].get("label_id") or boxes[0].get("label") or ""
                    top_label_id = lbl_map.get(box_key, "")
                if not top_label_id and not is_background_sample(sample):
                    # Background samples keep an empty label deliberately. A
                    # sample can carry a stale top-level label_id and still be
                    # marked background (e.g. an imported folder label, or the
                    # bulk action applied to an already-labeled sample); writing
                    # that label into the cache would make it a classification
                    # example of a class it asserts is absent, and would mint an
                    # output class from a negative.
                    top_label_id = sample.label_id or ""

                stype = (
                    sample.sample_type.value
                    if hasattr(sample.sample_type, "value")
                    else str(sample.sample_type)
                )

                features_list.append(features)
                label_ids_list.append(top_label_id)
                sample_ids_list.append(sample.id)
                sample_types_list.append(stype)
                boxes_json_list.append(json.dumps(boxes))
                background_list.append(1 if is_background_sample(sample) else 0)

                processed += 1
                if processed % 5 == 0:
                    pct = int((processed / max(len(usable_samples), 1)) * 100)
                    self.update_state(
                        state="RUNNING",
                        meta={"status": "processing", "progress": pct},
                    )

            except Exception as exc:
                logger.warning(f"[DSP] Skipping sample {sample.id}: {exc}")

        num = len(features_list)
        # Counted from what was actually cached, not from what was selected —
        # a background sample whose DSP raised is not in the cache and must not
        # be reported as if it were.
        n_background = int(sum(background_list))
        failed = total - skipped_unlabeled - num
        logger.info(
            f"[DSP] {total} total samples — "
            f"{num} extracted ({n_background} background/negative), "
            f"{skipped_unlabeled} skipped (unlabeled/placeholder), "
            f"{failed} failed (exception)"
        )

        # ── Build numpy arrays ────────────────────────────────────────────────
        # X may be non-uniform across samples (image tensor vs flat vector).
        # Try a uniform float32 array first; fall back to an object array.
        try:
            X = np.array(features_list, dtype=np.float32)
        except ValueError:
            X = np.empty(num, dtype=object)
            for i, f in enumerate(features_list):
                X[i] = np.asarray(f, dtype=np.float32)

        y            = np.array(label_ids_list,    dtype=object)
        ids          = np.array(sample_ids_list,   dtype=object)
        sample_types = np.array(sample_types_list, dtype=object)
        boxes_json   = np.array(boxes_json_list,   dtype=object)
        # Explicit per-sample negative flag. The training worker must not have
        # to re-derive "was this a background image?" from an empty label plus
        # an empty box list — that inference holds today only by an invariant
        # nothing enforces, and it would fail open (silently dropping the
        # sample) the first time the invariant moved.
        background   = np.array(background_list,   dtype=np.int8)

        # ── Validity metadata blob (v2) ───────────────────────────────────────
        meta = {
            # v4: background ("negative") images are now included in the cache.
            # Bumping the version means every cache written before this change
            # is treated as stale and regenerated, rather than silently training
            # a detection model on a dataset with its negatives missing.
            "version":           FEATURE_CACHE_VERSION,
            "background_count":  n_background,
            "impulse_id":        impulse_id,
            "project_id":        impulse.project_id,
            "dsp_config_hash":   dsp_hash,
            "image_width":       image_width,
            "image_height":      image_height,
            "boxes_normalized":  True,
            "label_names":       label_names,
            "label_ids_ordered": label_ids_order,
            "sample_count":      num,
            "sample_ids":        sample_ids_list,
            # Recorded so the training worker can detect when the user has
            # changed the subset percent without re-running feature gen and
            # invalidate the cache instead of silently training on the wrong
            # sample mix.
            "train_subset_percent": float(subset_pct),
            "created_at":        datetime.utcnow().isoformat(),
        }

        # ── Persist ───────────────────────────────────────────────────────────
        buf = io.BytesIO()
        np.savez_compressed(
            buf,
            X=X,
            y=y,
            ids=ids,
            sample_types=sample_types,
            boxes_json=boxes_json,
            background=background,
            meta_json=np.array([json.dumps(meta)], dtype=object),
        )
        buf.seek(0)

        out_key = make_features_storage_key(impulse.project_id, impulse_id)
        storage.upload_file(buf, out_key, "application/octet-stream")
        logger.info(
            f"[DSP] Feature cache (v{FEATURE_CACHE_VERSION}) saved → {out_key}  "
            f"({num} samples, {n_background} background, dsp_hash={dsp_hash[:8]}…)"
        )

        if job_id:
            assert_not_cancelled_before_completing(job_id)
            job = db.query(DspFeatureJob).filter(DspFeatureJob.id == job_id).first()
            if job is not None:
                job.status = JobStatus.completed
                job.completed_at = datetime.utcnow()
                db.commit()

        _log_event(
            _dsp_log, "dsp.feature_extraction",
            job_id=job_id, impulse_id=impulse_id, status="completed",
            samples_total=total, samples_processed=num,
            samples_skipped=skipped_unlabeled, samples_failed=failed,
            duration_ms=round((time.perf_counter() - _dsp_t0) * 1000.0, 2),
        )
        return {
            "status":          "completed",
            "impulse_id":      impulse_id,
            "num_samples":     num,
            "storage_key":     out_key,
            "dsp_config_hash": dsp_hash,
        }

    except CancelledError:
        logger.info(f"[DSP] Feature extraction cancelled for impulse {impulse_id} (job {job_id})")
        _log_event(
            _dsp_log, "dsp.feature_extraction",
            job_id=job_id, impulse_id=impulse_id, status="cancelled",
            duration_ms=round((time.perf_counter() - _dsp_t0) * 1000.0, 2),
        )
        if job_id:
            try:
                job = db.query(DspFeatureJob).filter(DspFeatureJob.id == job_id).first()
                if job is not None and job.status != JobStatus.cancelled:
                    job.status = JobStatus.cancelled
                    job.completed_at = datetime.utcnow()
                    db.commit()
            except Exception:
                db.rollback()
        return {"status": "cancelled", "impulse_id": impulse_id}
    except Exception as exc:
        logger.exception(f"[DSP] Worker failed for impulse {impulse_id}: {exc}")
        _log_event(
            _dsp_log, "dsp.feature_extraction",
            level=logging.ERROR,
            job_id=job_id, impulse_id=impulse_id, status="failed",
            reason=str(exc),
            duration_ms=round((time.perf_counter() - _dsp_t0) * 1000.0, 2),
        )
        # Best-effort failure record. A separate try-block keeps a DB error
        # here from masking the original exception in logs.
        if job_id:
            try:
                job = db.query(DspFeatureJob).filter(DspFeatureJob.id == job_id).first()
                if job is not None:
                    job.status = JobStatus.failed
                    job.error_message = str(exc)
                    job.completed_at = datetime.utcnow()
                    db.commit()
            except Exception:
                db.rollback()
                logger.exception("[DSP] Failed to mark DspFeatureJob as failed")
        return {"status": "failed", "impulse_id": impulse_id, "error": str(exc)}
    finally:
        db.close()
