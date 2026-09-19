"""
Shared cancellation primitives used by all training workers.

Importing from a single module avoids duplicating the cancelled-status check
across training_worker, yolo_pro_worker, and mobilenetv2_ssd_worker.

Cooperative cancellation is the primary stop mechanism because Celery is run
with --pool=solo (see cmd.txt). With the solo pool, revoke(terminate=True)
cannot actually kill the currently-executing task — the worker process IS the
task. So workers must poll the DB and exit voluntarily when they see
status='cancelled'.
"""
import time
import threading

from app.core.database import SessionLocal


class CancelledError(Exception):
    """Raised when a training job is cancelled by the user.

    Kept distinct from RuntimeError so each worker's exception handler can
    set status='cancelled' instead of 'failed' without inspecting the message.
    """


def is_job_cancelled(job_id: str) -> bool:
    """Return True if the DB row for job_id has status='cancelled'."""
    from app.models.user import TrainingJob, JobStatus
    db = SessionLocal()
    try:
        job = db.query(TrainingJob).filter(TrainingJob.id == job_id).first()
        return bool(job and job.status == JobStatus.cancelled)
    finally:
        db.close()


def raise_if_cancelled(job_id: str) -> None:
    """Raise CancelledError if the job has been cancelled."""
    if is_job_cancelled(job_id):
        raise CancelledError("Cancelled by user")


# ── Time-throttled fast check ────────────────────────────────────────────────
# Per-batch cancel checks (e.g. every training batch in the SSD/YOLO loops)
# fire hundreds of times per epoch.  A fresh DB session per call is wasteful
# and can lock-contend with the worker's own metric writes.  This helper
# coalesces calls within a small window so the tight-loop overhead stays
# negligible while user-visible cancel latency stays well under a second.

_LAST_CHECK_LOCK = threading.Lock()
_LAST_CHECK: dict = {}  # job_id -> (monotonic_ts, was_cancelled)
_FAST_TTL_SECONDS = 0.5


def raise_if_cancelled_fast(job_id: str, ttl: float = _FAST_TTL_SECONDS) -> None:
    """Same as raise_if_cancelled but cached for `ttl` seconds.

    Safe inside tight inner loops (per-batch).  Once the cached value flips
    to True, every subsequent call raises immediately without hitting the DB.
    """
    now = time.monotonic()
    with _LAST_CHECK_LOCK:
        entry = _LAST_CHECK.get(job_id)
    if entry is not None:
        ts, cancelled = entry
        if cancelled:
            raise CancelledError("Cancelled by user")
        if now - ts < ttl:
            return
    cancelled = is_job_cancelled(job_id)
    with _LAST_CHECK_LOCK:
        _LAST_CHECK[job_id] = (now, cancelled)
    if cancelled:
        raise CancelledError("Cancelled by user")


def clear_cancel_cache(job_id: str) -> None:
    """Drop any cached cancel state for a job (call at job start)."""
    with _LAST_CHECK_LOCK:
        _LAST_CHECK.pop(job_id, None)


def assert_not_cancelled_before_completing(job_id: str) -> None:
    """Final guard before writing JobStatus.completed.

    Re-fetches the DB row (bypasses any caching) and raises CancelledError
    if the cancel endpoint has set status='cancelled' while the worker was
    in a long-running section (eval, TFLite export, artifact upload) that
    didn't have its own cancel checks.

    Use this immediately before `job.status = JobStatus.completed`.
    """
    if is_job_cancelled(job_id):
        raise CancelledError("Cancelled by user")


# ── Live training-buffer registry ────────────────────────────────────────────
# The per-architecture training loops own their in-memory log_lines /
# epoch_metrics buffers.  When a run ends in failed/cancelled, the outer Celery
# handler (run_training_job) must persist every line/metric those buffers hold —
# including the ones appended since the last incremental commit — before it
# writes the terminal status.  The loops register their buffers here so the
# outer handler can reach them without threading them through return values or
# splitting the (heavily source-introspected) loop functions.
#
# Keyed by job_id and popped/cleared by the outer handler on every exit, so
# entries never leak across runs.  The registry stores references to the live
# list objects, so appends made after registration are visible at flush time.

_LIVE_BUFFER_LOCK = threading.Lock()
_LIVE_BUFFERS: dict = {}  # job_id -> {"log_lines": list|None, "epoch_metrics": list|None}


def register_live_buffer(job_id: str, *, log_lines=None, epoch_metrics=None) -> None:
    """Register (or update) the live log/epoch buffers for a running job.

    Either argument may be omitted; only the provided buffer(s) are recorded.
    The loops call this once per buffer, as soon as each buffer is created.
    """
    with _LIVE_BUFFER_LOCK:
        entry = _LIVE_BUFFERS.setdefault(job_id, {"log_lines": None, "epoch_metrics": None})
        if log_lines is not None:
            entry["log_lines"] = log_lines
        if epoch_metrics is not None:
            entry["epoch_metrics"] = epoch_metrics


def pop_live_buffer(job_id: str):
    """Remove and return the registered buffers for a job, or None if absent."""
    with _LIVE_BUFFER_LOCK:
        return _LIVE_BUFFERS.pop(job_id, None)


def clear_live_buffer(job_id: str) -> None:
    """Drop any registered buffers for a job (call at job start and on exit)."""
    with _LIVE_BUFFER_LOCK:
        _LIVE_BUFFERS.pop(job_id, None)


def promote_run_to_active(impulse_id: str, job_id: str) -> None:
    """Point the impulse's `active_model_run_id` at this successfully-completed run.

    Called on the same session that just flipped `job.status = completed`. The
    pointer is updated for BOTH fresh and retrain runs — completion is the only
    write site. We never clear it elsewhere (start/cancel/fail), so the previous
    pointer survives whenever a fresh run is started, cancelled, or fails, which
    is the recoverability invariant.

    Operates via a dedicated SessionLocal so callers don't have to thread their
    own Session in; the worker's existing session is unaffected.
    """
    from app.models.user import Impulse
    db = SessionLocal()
    try:
        impulse = db.query(Impulse).filter(Impulse.id == impulse_id).first()
        if impulse is None:
            return
        impulse.active_model_run_id = job_id
        db.commit()
    finally:
        db.close()
