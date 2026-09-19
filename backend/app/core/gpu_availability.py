"""
GPU training availability — the checks that run in the API request path.

Deliberately free of TensorFlow: app.ml.gpu_utils imports `tensorflow` at module
scope, which is far too heavy to pull into a synchronous FastAPI handler. This
module talks only to the Celery control API.

The probe here answers one narrow question — "is anything consuming the
`training_gpu` queue right now?" — and answers it in three states, because
"nobody replied" is not the same as "nobody is there". See
`gpu_queue_has_live_worker`.
"""
import logging
from typing import Optional

from app.core.config import settings
# app.workers.queues is dependency-free (no celery, no tensorflow) precisely so
# this module can import it without dragging either into the request path.
from app.workers.queues import TRAINING_GPU_QUEUE as GPU_TRAINING_QUEUE

logger = logging.getLogger(__name__)


def gpu_queue_has_live_worker(timeout: Optional[float] = None) -> Optional[bool]:
    """Ask Celery whether any worker is consuming the GPU training queue.

    Args:
        timeout: seconds to wait for worker replies. Defaults to
            settings.GPU_WORKER_PROBE_TIMEOUT_SECONDS (1.0s). Kept short
            because this runs inline in a request.

    Returns:
        True  — at least one worker reports consuming `training_gpu`.
        False — workers replied and none of them consumes `training_gpu`.
                This is the only *definitive negative*, and the only result a
                caller may reject on.
        None  — inconclusive: the broker was unreachable, the call raised, or
                no worker answered within the timeout. Celery returns None both
                when the broker is down and when every worker is merely slow,
                so absence of a reply is NOT evidence of absence of a worker.
                Callers must fail open on this.

    Known race: a GPU worker that is alive but slower to reply than `timeout`,
    while a CPU worker replies in time, yields a non-empty mapping without the
    GPU queue in it — reported as a definitive False. Widening the timeout trades
    request latency for a narrower window; 1.0s is the chosen balance.
    """
    probe_timeout = (
        settings.GPU_WORKER_PROBE_TIMEOUT_SECONDS if timeout is None else timeout
    )

    try:
        # Imported lazily: app.workers.celery_app pulls in the worker modules,
        # and this keeps that cost off the import path of every API process that
        # never asks for GPU.
        from app.workers.celery_app import celery_app

        active = celery_app.control.inspect(timeout=probe_timeout).active_queues()
    except Exception as e:
        logger.warning(
            "[GPU] Worker liveness probe failed (%s: %s) — treating as inconclusive",
            type(e).__name__, e,
        )
        return None

    if not active:
        # None (nothing replied) or {} — indistinguishable from a broker outage.
        logger.warning(
            "[GPU] Worker liveness probe got no reply within %.2fs — "
            "treating as inconclusive",
            probe_timeout,
        )
        return None

    try:
        for worker_name, queues in active.items():
            for queue in (queues or []):
                if queue.get("name") == GPU_TRAINING_QUEUE:
                    logger.debug(
                        "[GPU] Worker %s consumes %s", worker_name, GPU_TRAINING_QUEUE
                    )
                    return True
    except AttributeError as e:
        # Unexpected payload shape — not proof of absence.
        logger.warning(
            "[GPU] Worker liveness probe returned an unreadable payload (%s) — "
            "treating as inconclusive", e,
        )
        return None

    logger.info(
        "[GPU] %d worker(s) replied, none consuming %s",
        len(active), GPU_TRAINING_QUEUE,
    )
    return False
