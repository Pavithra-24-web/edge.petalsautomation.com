# PetalEdge backend image — serves the FastAPI API and, via CMD override in
# k8s (see k8s/worker-cpu-deployment.yaml), the CPU Celery worker. Same image,
# different command: keeps dependency versions identical between API and worker.
FROM python:3.11-slim

# System libs needed at runtime (not just build time), so they stay in the
# final image rather than a discarded builder stage:
#   libsndfile1, ffmpeg  -> librosa (audio decode)
#   libgl1, libglib2.0-0 -> opencv-python-headless (imported even in headless mode)
#   curl                 -> container healthcheck / debugging
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY backend/requirements.txt .
# torch's default PyPI wheel bundles full CUDA (nvidia-cublas-cu12,
# nvidia-cusolver-cu12, etc. — ~2GB combined), even though this image only
# ever runs CPU inference/training (API is GPU-disabled in main.py, and the
# only worker deployed via k8s/worker-cpu-deployment.yaml is CPU-only — the
# real GPU worker runs on bare metal, outside this image, per the Makefile).
# Installing from PyTorch's own CPU index first avoids ever pulling those
# CUDA packages; the requirements.txt install below then finds torch/
# torchvision already satisfied at the pinned version and skips them.
RUN pip install --no-cache-dir torch==2.3.0 torchvision==0.18.0 \
        --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

COPY backend/alembic.ini .
COPY backend/alembic ./alembic
COPY backend/app ./app
# deployment_worker.py reads unoq/runtime/runner.py at import time — resolved
# as three parents above its own file (backend/app/workers/deployment_worker.py
# -> repo root in a normal checkout), i.e. /unoq here, a sibling of /app, not
# something copied into it. Embedded verbatim into every generated .pxe
# device package, so the whole tree — not just runner.py — needs to be
# present, matching the real repo layout.
COPY unoq /unoq

# Runs as non-root; /app is owned by this user so alembic/Celery can write
# local caches (e.g. HF/transformers cache) under it if needed.
RUN useradd --create-home --uid 1000 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
