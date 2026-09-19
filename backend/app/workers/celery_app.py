"""
Celery application — task queue for training, DSP, and deployment jobs
"""
import logging
import os
import time

from celery import Celery, Task
from celery.signals import (
    celeryd_init,
    worker_init,
    worker_process_init,
    task_prerun,
    task_postrun,
    task_failure,
)
from kombu import Queue
from app.core.config import settings
from app.core.logging_config import configure_logging, get_logger, log_event
from app.workers.queues import (
    ALL_QUEUES,
    DEFAULT_CPU_QUEUE,
    DEPLOYMENT_QUEUE,
    DSP_QUEUE,
    GPU_QUEUES,
    POST_PROCESSING_QUEUE,
    TRAINING_CPU_QUEUE,
    TRAINING_GPU_QUEUE,
    is_gpu_eligible,
)

# Ensure the per-channel JSON loggers exist in every worker process.
configure_logging()
_celery_log = get_logger("celery")

# task_id → perf_counter() start, set in prerun and consumed in postrun so each
# task's wall-clock duration can be reported. Keyed by task_id (unique per
# task) so this is safe to mutate from multiple concurrently-running tasks —
# whether that concurrency comes from prefork child processes or, on Windows
# where fork() is unavailable, multiple threads in a --pool=threads worker.
_task_starts: dict[str, float] = {}


class NonTrainingTaskOnGpuWorker(RuntimeError):
    """A task that is not GPU-eligible was dispatched to a GPU worker."""


def _worker_is_gpu() -> bool:
    """True when this process was started with WORKER_DEVICE=gpu."""
    return os.environ.get("WORKER_DEVICE", "cpu").lower().strip() == "gpu"


class DeviceGuardedTask(Task):
    """Task base that refuses to execute non-training work on a GPU worker.

    The check lives in __call__ rather than in a task_prerun signal handler
    because Celery *catches and logs* exceptions raised by signal receivers and
    then runs the task anyway — a prerun guard reports the violation without
    preventing it. Overriding __call__ puts the check inside the traced call
    path (celery.app.trace dispatches through __call__ whenever a task defines
    a custom one), so raising here fails the task properly.

    Routing already makes this unreachable; this is the backstop for when
    routing configuration is wrong. See app/workers/queues.py.
    """

    def __call__(self, *args, **kwargs):
        if _worker_is_gpu() and not is_gpu_eligible(self.name):
            log_event(
                _celery_log, "celery.gpu_admission_rejected",
                level=logging.ERROR, task=self.name,
                task_id=getattr(self.request, "id", None),
            )
            raise NonTrainingTaskOnGpuWorker(
                f"Task {self.name!r} is not GPU-eligible but was dispatched to "
                "a GPU worker. The GPU worker must consume only the "
                "training_gpu queue; everything else belongs on a CPU worker. "
                "Check this worker's -Q flags and the task_routes table in "
                "app/workers/celery_app.py."
            )
        return super().__call__(*args, **kwargs)


celery_app = Celery(
    "petaledge",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND,
    task_cls=DeviceGuardedTask,
    include=[
        "app.workers.training_worker",
        "app.workers.deployment_worker",
        "app.workers.dsp_worker",
        "app.workers.model_testing_worker",
        "app.workers.post_processing_worker",
    ],
)

@celeryd_init.connect
def _preflight_check_worker_queues(sender, conf, options=None, **kwargs):
    """Refuse to start a GPU worker that consumes anything but training_gpu.

    Runs in the controller before the pool is spawned, alongside the CUDA
    preflight below, and exits non-zero for the same reason: a GPU worker
    subscribed to `dsp` or `deployment` would quietly run export builds and
    feature generation on the training card, competing for VRAM with a live
    training run. Better to never come up than to come up wrong.

    CPU workers are unconstrained — they may consume any queue, including the
    legacy `training` queue, which is exactly what drains it.
    """
    import sys

    if not _worker_is_gpu():
        return

    requested = options.get("queues") if options else None
    if not requested:
        # No -Q given: the worker would consume every declared queue, GPU and
        # CPU alike. Never valid for a GPU worker.
        logger = logging.getLogger(__name__)
        logger.critical(
            "[GPU] FATAL: WORKER_DEVICE=gpu but no -Q was given, so this worker "
            "would consume every declared queue. Start it with "
            "-Q %s. Refusing to start.", TRAINING_GPU_QUEUE,
        )
        sys.exit(1)

    extra = {str(q) for q in requested} - set(GPU_QUEUES)
    if extra:
        logger = logging.getLogger(__name__)
        logger.critical(
            "[GPU] FATAL: WORKER_DEVICE=gpu but this worker also consumes %s. "
            "A GPU worker may consume only %s — everything else belongs on a "
            "CPU worker. Refusing to start.",
            sorted(extra), TRAINING_GPU_QUEUE,
        )
        sys.exit(1)


@celeryd_init.connect
def _preflight_check_worker_device(sender, conf, **kwargs):
    """GPU preflight in the main worker process, before the pool is spawned.

    Calls sys.exit(1) if WORKER_DEVICE=gpu but no GPU is visible, so the worker
    dies decisively rather than letting every prefork child fail individually.
    CPU workers return immediately without touching TF.
    """
    from app.ml.gpu_utils import validate_worker_device_preflight
    validate_worker_device_preflight()


@worker_process_init.connect
@worker_init.connect
def _configure_worker_device(sender=None, **kwargs):
    """Configure GPU/CPU once per worker process at startup.

    worker_process_init fires post-fork, per child (prefork pool) — the
    right place for GPU setup there, since configuring CUDA before a fork
    and inheriting it into children is unsafe. worker_init fires once in the
    parent, before the pool exists; for non-forking pools (solo, threads —
    threads is how this project gets CPU-queue concurrency on Windows, where
    prefork's os.fork() isn't available) that parent process IS the only
    process, so worker_init is the only signal that will ever fire for them.
    Skip prefork here and let worker_process_init handle it per child instead.
    """
    if "prefork" in str(getattr(sender, "pool_cls", "") or "").lower():
        return
    from app.ml.gpu_utils import get_worker_device, init_worker_device
    if get_worker_device() != "uninitialized":
        return
    # Re-bind the channel loggers in each forked child so file handles are owned
    # by the child process rather than inherited from the parent. No-op if
    # already configured (configure_logging is idempotent).
    configure_logging(force=True)
    init_worker_device()


# ─── Task lifecycle → celery.log ──────────────────────────────────────────────

@task_prerun.connect
def _log_task_prerun(task_id=None, task=None, **kwargs):
    _task_starts[task_id] = time.perf_counter()
    log_event(
        _celery_log, "celery.task_start",
        task=getattr(task, "name", None), task_id=task_id,
    )


@task_postrun.connect
def _log_task_postrun(task_id=None, task=None, state=None, **kwargs):
    start = _task_starts.pop(task_id, None)
    duration_ms = round((time.perf_counter() - start) * 1000.0, 2) if start else None
    log_event(
        _celery_log, "celery.task_finish",
        level=logging.ERROR if state == "FAILURE" else logging.INFO,
        task=getattr(task, "name", None), task_id=task_id,
        state=state, duration_ms=duration_ms,
    )


@task_failure.connect
def _log_task_failure(task_id=None, sender=None, exception=None, **kwargs):
    # Drop the start marker so postrun doesn't compute a duration twice; postrun
    # still fires with state=FAILURE for the timing line.
    log_event(
        _celery_log, "celery.task_failed",
        level=logging.ERROR,
        task=getattr(sender, "name", None), task_id=task_id,
        reason=str(exception) if exception else None,
        exc_type=type(exception).__name__ if exception else None,
    )


celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_acks_late=True,
    worker_prefetch_multiplier=1,
    # Every route below points at a CPU queue. `training_gpu` is deliberately
    # absent: it is never a routing destination, only an explicit
    # apply_async(queue=...) target from the training endpoints. See the module
    # docstring in app/workers/queues.py for why the routing floor must be CPU.
    task_routes={
        # Training defaults to CPU. The endpoints override this per job with an
        # explicit queue= (training_gpu or training_cpu), which outranks the
        # route — this entry is the floor for any other dispatch path, such as
        # a bare .delay() or an automatic .retry().
        "app.workers.training_worker.*":           {"queue": TRAINING_CPU_QUEUE},
        "app.workers.deployment_worker.*":         {"queue": DEPLOYMENT_QUEUE},
        "app.workers.dsp_worker.*":                {"queue": DSP_QUEUE},
        "app.workers.model_testing_worker.*":      {"queue": TRAINING_CPU_QUEUE},
        "app.workers.post_processing_worker.*":    {"queue": POST_PROCESSING_QUEUE},
    },
    # A task matching no route lands here rather than in an undeclared "celery"
    # queue that no worker consumes. CPU, so an unrouted task can never reach
    # the GPU.
    task_default_queue=DEFAULT_CPU_QUEUE,
    task_queues=tuple(Queue(name) for name in ALL_QUEUES),
    task_time_limit=3600,
    task_soft_time_limit=3540,
)
