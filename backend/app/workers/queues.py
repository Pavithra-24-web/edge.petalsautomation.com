"""
Celery queue topology — the single source of truth for which work runs where.

Deliberately dependency-free: no celery, no tensorflow, no app.core imports.
Both the Celery app (app/workers/celery_app.py) and the TF-free API-side probe
(app/core/gpu_availability.py) import from here, and the latter exists precisely
to stay off TensorFlow's import path.

── The invariant ─────────────────────────────────────────────────────────────

The GPU worker consumes exactly one queue, `training_gpu`, and that queue
carries exactly one task, `run_training_job`. Everything else in the system —
API request handling, inference, DSP feature generation, model testing,
deployment/export builds, post-processing, and every database write — runs on a
CPU worker or in the API process, where GPUs are hidden outright by
configure_gpu("cpu").

Three independent mechanisms hold that invariant, because queue conventions
alone are only a convention:

  1. Routing (`task_routes` in celery_app.py) sends every task to a CPU queue by
     default. `training_gpu` is never a routing *destination* — the only way to
     reach it is an explicit `apply_async(queue="training_gpu")`, which only the
     training endpoints do, and only after `_assert_gpu_training_available()`.
  2. `task_default_queue` is a CPU queue, so a task that somehow matches no
     route still cannot land on the GPU.
  3. A `task_prerun` guard (celery_app.py) rejects any task not in
     GPU_ELIGIBLE_TASKS when WORKER_DEVICE=gpu, even if a misconfigured `-Q`
     put it there.

── Why routing defaults must point at CPU, never GPU ─────────────────────────

Celery merges routing options as `lpmerge(route, options)` — an explicit
`apply_async(queue=...)` beats `task_routes`, and a task decorator's `queue=`
*also* beats `task_routes` (verified against celery 5.4.0). So `task_routes` is
the weakest of the three and can only ever be a floor. Making that floor CPU
means every failure mode — a missing route, a stray `.delay()`, an automatic
`.retry()` — degrades to running on CPU, which is slow but correct. A GPU floor
would degrade to contending for 24 GB of VRAM against a live training run.

For the same reason no task in this project should carry `queue=` in its
`@celery_app.task` decorator: it silently outranks the routing table and makes
the topology unreadable from one place. `run_model_testing_job` used to, which
pinned it to the legacy `training` queue while the routing table claimed
`training_cpu`.
"""

# ── Queue names ───────────────────────────────────────────────────────────────

#: The one and only GPU queue. Reached exclusively by an explicit
#: apply_async(queue=...) from the training endpoints.
TRAINING_GPU_QUEUE = "training_gpu"

#: Training that the user asked to run on CPU, or that fell back to CPU because
#: GPU_ENABLED=false / no GPU worker is live.
TRAINING_CPU_QUEUE = "training_cpu"

#: Export / .eim / .pxe builds. TensorFlow-heavy but CPU-only by design.
DEPLOYMENT_QUEUE = "deployment"

#: Signal processing / feature generation.
DSP_QUEUE = "dsp"

#: Video post-processing pipeline.
POST_PROCESSING_QUEUE = "post_processing"

#: Catch-all for anything unrouted. Declared and consumed so that a task with no
#: matching route surfaces as a normal backlog instead of vanishing into an
#: undeclared "celery" queue nobody reads.
DEFAULT_CPU_QUEUE = "cpu_default"

#: Pre-split queue name. Kept declared and consumed by CPU workers only, so
#: messages enqueued before the training_cpu/training_gpu split still drain.
#: Safe to delete once the Redis list is empty on every environment.
LEGACY_TRAINING_QUEUE = "training"

#: Every queue a CPU worker should consume. Order is the -Q order.
CPU_QUEUES = (
    TRAINING_CPU_QUEUE,
    DSP_QUEUE,
    DEPLOYMENT_QUEUE,
    POST_PROCESSING_QUEUE,
    DEFAULT_CPU_QUEUE,
    LEGACY_TRAINING_QUEUE,
)

#: Every queue a GPU worker should consume. Exactly one, by design.
GPU_QUEUES = (TRAINING_GPU_QUEUE,)

ALL_QUEUES = CPU_QUEUES + GPU_QUEUES


# ── GPU admission control ─────────────────────────────────────────────────────

#: The only task names permitted to execute in a WORKER_DEVICE=gpu process.
#: Enforced at runtime by the task_prerun guard in celery_app.py. Adding a task
#: here is a deliberate decision to let it hold the GPU.
GPU_ELIGIBLE_TASKS = frozenset({
    "app.workers.training_worker.run_training_job",
})


def is_gpu_eligible(task_name: str | None) -> bool:
    """True if `task_name` may run on a GPU worker. Unknown/None names are not."""
    return task_name in GPU_ELIGIBLE_TASKS
