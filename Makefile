# PetalEdge — Developer Makefile (Simplified)
# ─────────────────────────────────────────────────────────────
# Infrastructure runs in Docker. App code runs locally.
# ─────────────────────────────────────────────────────────────

.PHONY: help up down logs backend frontend minio \
        worker-cpu worker-gpu \
        migrate psql redis-cli test lint setup clean

help:
	@echo ""
	@echo "  PetalEdge — Commands"
	@echo "  ════════════════════════════════════════════════════"
	@echo ""
	@echo "  make up           Start Postgres + Redis (Docker)"
	@echo "  make down         Stop Docker services"
	@echo "  make logs         Follow Docker logs"
	@echo "  make backend      Run FastAPI locally (hot-reload)"
	@echo "  make frontend     Run Next.js locally"
	@echo "  make minio        Start MinIO (optional)"
	@echo ""
	@echo "  make worker-cpu   Celery CPU worker (all queues but training_gpu)"
	@echo "  make worker-gpu   Celery GPU worker (training_gpu only, GPU host)"
	@echo ""
	@echo "  make migrate      Run Alembic migrations"
	@echo "  make psql         Open psql shell"
	@echo "  make redis-cli    Open redis-cli"
	@echo "  make test         Run backend tests"
	@echo "  make lint         Run frontend linter"
	@echo "  make setup        Install Python + Node deps"
	@echo "  make clean        Stop all, wipe volumes + cache"
	@echo ""

# ── Docker (infrastructure only) ──────────────────────────────

up:
	docker compose --env-file .env.dev up -d
	@echo ""
	@echo "  ✅  Postgres → localhost:5439"
	@echo "  ✅  Redis    → localhost:6379"
	@echo ""

down:
	docker compose down

logs:
	docker compose logs -f

minio:
	docker compose --env-file .env.dev --profile minio up -d minio
	@echo "  ✅  MinIO API     → http://localhost:9007"
	@echo "  ✅  MinIO Console → http://localhost:9006"

# ── Local app servers ─────────────────────────────────────────

backend:
	cd backend && \
	  set -a && . ../.env.dev && set +a && \
	  uvicorn app.main:app --host 0.0.0.0 --port 8010 --reload

frontend:
	cd frontend && npm run dev

# ── Celery workers ────────────────────────────────────────────
# Two worker classes, and the split is load-bearing:
#
#   worker-cpu  consumes everything except training_gpu. DSP, deployment
#               builds, model testing, post-processing, CPU training.
#               WORKER_DEVICE=cpu hides all GPUs in the process.
#   worker-gpu  consumes training_gpu and nothing else. Runs ONLY on the GPU
#               host. Adding queues to its -Q is not a shortcut — the
#               task_prerun guard in celery_app.py fails any non-training task
#               that arrives there.
#
# --pool=solo on the GPU worker: one training job at a time owns the card.
# Concurrency there means two jobs racing for the same 24 GB of VRAM.

worker-cpu:
	cd backend && \
	  set -a && . ../.env.dev && set +a && \
	  WORKER_DEVICE=cpu celery -A app.workers.celery_app worker \
	    -Q training_cpu,dsp,deployment,post_processing,cpu_default,training \
	    --concurrency=2 --hostname=cpu@%h -l info

# XLA_FLAGS: XLA's cuDNN convolution *autotuner* is broken in the conda-forge
# TensorFlow 2.16.1 cuda120 build we run on the non-AVX GPU host — every
# YOLO-Pro train step dies with CUDNN_STATUS_EXECUTION_FAILED at
# cuda_dnn.cc(8206). Reproduced 2/2 on the real architecture with XLA on and
# 0/2 with it off; --xla_gpu_autotune_level=0 fixes it while keeping XLA
# enabled, at ~4% step time (37.5 vs 38.9 ms/step, nano @224, batch 13).
# TF_CUDNN_USE_FRONTEND=0, TF_CUDNN_USE_AUTOTUNE=0 and
# TF_USE_CUDNN_BATCHNORM_SPATIAL_PERSISTENT=0 were all tried and do NOT help.
# Remove this once the host runs the official pip TF wheel (needs AVX, i.e. a
# host-passthrough guest CPU) — the stock build's autotuner is expected to work.
worker-gpu:
	cd backend && \
	  set -a && . ../.env.dev && set +a && \
	  WORKER_DEVICE=gpu XLA_FLAGS=--xla_gpu_autotune_level=0 \
	  celery -A app.workers.celery_app worker \
	    -Q training_gpu \
	    --pool=solo --hostname=gpu@%h -l info

# ── Database ──────────────────────────────────────────────────

migrate:
	cd backend && \
	  set -a && . ../.env.dev && set +a && \
	  alembic upgrade head

migrate-new:
	cd backend && \
	  set -a && . ../.env.dev && set +a && \
	  alembic revision --autogenerate -m "$(MSG)"

psql:
	docker compose exec postgres psql -U PetalEdge -d PetalEdge

redis-cli:
	docker compose exec redis redis-cli

# ── Testing / Linting ────────────────────────────────────────

test:
	cd backend && pytest tests/ -v --tb=short

lint:
	cd frontend && npm run lint

# ── Setup / Clean ─────────────────────────────────────────────

setup:
	cd backend && pip install -r requirements.txt
	cd frontend && npm install

clean:
	docker compose --profile minio down -v --remove-orphans
	find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null; true
	find . -name "*.pyc" -delete 2>/dev/null; true
