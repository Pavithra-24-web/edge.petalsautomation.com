"""
GPU/CPU device configuration for training workers.

Design: device selection is PROCESS-LEVEL, not per-job.

Startup flow (enforced via Celery worker_init signal in celery_app.py):
  1. Worker reads WORKER_DEVICE env var ("cpu" or "gpu", default "cpu").
  2. init_worker_device() calls configure_gpu() exactly once per process.
  3. Per-job code calls assert_job_device() to validate routing is correct.

Why process-level only:
  tf.config.set_visible_devices() and set_memory_growth() are permanent for
  the lifetime of the process. Calling configure_gpu("cpu") hides GPUs
  permanently — a subsequent GPU job in the same process would fail even on
  a GPU host. Separate worker pools (training_cpu / training_gpu) enforce
  the split at the OS/container level instead.
"""

import logging
import os
import tensorflow as tf

logger = logging.getLogger(__name__)

# Module-level record of what this worker process was configured as.
# Set once by init_worker_device() at worker startup.
_worker_device: str = "uninitialized"


def init_worker_device() -> str:
    """
    Read WORKER_DEVICE env var and configure TF device visibility once.
    Call this from the Celery worker_init signal — never from a task.

    Returns the resolved device string ("CPU:0" or "GPU:0").
    """
    global _worker_device
    pref = os.environ.get("WORKER_DEVICE", "cpu").lower().strip()
    if pref not in ("cpu", "gpu"):
        logger.warning(
            "[GPU] Unknown WORKER_DEVICE=%r — defaulting to cpu", pref
        )
        pref = "cpu"
    device = configure_gpu(pref)
    _worker_device = device
    return device


def configure_gpu(device_preference: str) -> str:
    """
    Configure TensorFlow device visibility and memory growth.
    Must be called at most once per process (enforced by init_worker_device).

    Args:
        device_preference: "cpu" or "gpu"

    Returns:
        Resolved device string — "CPU:0" or "GPU:0"

    Raises:
        RuntimeError: if "gpu" is requested but no GPU is visible to TF.
    """
    gpus = tf.config.list_physical_devices("GPU")

    if device_preference == "gpu":
        if not gpus:
            raise RuntimeError(
                "WORKER_DEVICE=gpu but TensorFlow sees no GPU devices. "
                "Ensure CUDA drivers are installed and the worker is on a GPU host."
            )
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logger.info(
            "[GPU] worker configured as GPU:0  visible_gpus=%s",
            [g.name for g in gpus],
        )
        return "GPU:0"

    # cpu — hide all GPUs permanently for this process
    tf.config.set_visible_devices([], "GPU")
    logger.info("[GPU] worker configured as CPU:0 (all GPUs hidden)")
    return "CPU:0"


def get_worker_device() -> str:
    """Return the device this worker process was configured for."""
    return _worker_device


def worker_uses_gpu() -> bool:
    """
    True when work in THIS process will actually execute on a GPU.

    This is the only probe training code should use to switch on device —
    never ``tf.config.list_physical_devices("GPU")``, which enumerates host
    hardware and is unaffected by the ``set_visible_devices([], "GPU")`` call
    that configure_gpu("cpu") makes. On a GPU host running a CPU worker,
    list_physical_devices still reports the GPUs, so gating on it turns on
    GPU-only paths (mixed_float16, XLA) while TF is restricted to CPU.

    Falls back to the visibility-aware probe when the process was never
    initialized by init_worker_device() — tests, scripts, and anything that
    imports the training modules outside a Celery worker — so those keep the
    behaviour they have today.
    """
    if _worker_device == "GPU:0":
        return True
    if _worker_device == "CPU:0":
        return False
    return bool(tf.config.get_visible_devices("GPU"))


def assert_job_device(job_device_type: str, job_id: str) -> None:
    """
    Validate that the job's requested device matches this worker's device.
    Called once per job after init_worker_device() has run.

    Raises:
        RuntimeError: GPU job landed on a CPU worker (hard routing failure).
    Logs a warning: CPU job landed on a GPU worker (suboptimal but safe —
        GPUs are already hidden via configure_gpu("cpu") on CPU workers,
        so this case cannot arise in practice when queues are configured
        correctly; the warning exists as a safety net).
    """
    if _worker_device == "uninitialized":
        logger.warning(
            "[%s] worker device was never initialized — "
            "init_worker_device() was not called at startup. "
            "Defaulting to CPU:0.",
            job_id,
        )
        return

    requested = (job_device_type or "cpu").lower()

    if requested == "gpu" and _worker_device == "CPU:0":
        raise RuntimeError(
            f"Job {job_id} requested device=gpu but this worker is configured "
            "as CPU:0. Check Celery queue routing — GPU jobs must only be "
            "consumed by workers started with WORKER_DEVICE=gpu on a GPU host."
        )

    logger.info(
        "[%s] device check  requested=%s  worker=%s",
        job_id, requested, _worker_device,
    )


def validate_worker_device_preflight() -> None:
    """Fail-fast GPU check for the main Celery worker process.

    Call from a startup signal that fires in the controller (e.g. celeryd_init),
    before the prefork pool is spawned.  CPU workers return immediately.

    If WORKER_DEVICE=gpu but TF sees no GPUs the worker logs a fatal message and
    calls sys.exit(1) so the whole worker dies decisively instead of spawning a
    pool of children that all fail individually.
    """
    import sys

    pref = os.environ.get("WORKER_DEVICE", "cpu").lower().strip()
    if pref != "gpu":
        return  # CPU workers always pass — nothing to check

    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        logger.critical(
            "[GPU] FATAL: WORKER_DEVICE=gpu but TensorFlow sees no GPU devices. "
            "Ensure CUDA drivers are installed and this worker is running on a GPU host. "
            "Refusing to start — exiting with status 1."
        )
        sys.exit(1)

    logger.info(
        "[GPU] preflight passed: %d GPU(s) visible — %s",
        len(gpus), [g.name for g in gpus],
    )


def get_strategy() -> tf.distribute.Strategy:
    """
    Return the distribution strategy for this worker's configured device.
    Call after init_worker_device() has run.
    """
    if _worker_device != "GPU:0":
        return tf.distribute.get_strategy()

    gpus = tf.config.list_physical_devices("GPU")
    if len(gpus) > 1:
        strategy = tf.distribute.MirroredStrategy()
        logger.info("[GPU] using MirroredStrategy across %d GPUs", len(gpus))
        return strategy

    strategy = tf.distribute.OneDeviceStrategy("GPU:0")
    logger.info("[GPU] using OneDeviceStrategy on GPU:0")
    return strategy
