"""
petaledge Backend — FastAPI Application
"""
# ── Keep the API process off the GPU ──────────────────────────────────────────
# Must run before ANY import that can pull in tensorflow or torch: both read
# CUDA_VISIBLE_DEVICES once, when their CUDA runtime initializes, and ignore
# later changes.
#
# The API serves inline inference (endpoints/inference.py, ai_labeling.py,
# evaluation.py) in the request path. TensorFlow allocates on every visible GPU
# at first use, so an API process on a GPU host would take the card out from
# under the training worker that is supposed to own it exclusively — Celery
# routing cannot prevent this, because none of it goes through Celery.
#
# Set API_ALLOW_GPU=true to opt a deployment back into GPU-backed API inference.
import os as _os

if _os.environ.get("API_ALLOW_GPU", "").lower().strip() not in ("1", "true", "yes"):
    # Assigned, not setdefault: an inherited CUDA_VISIBLE_DEVICES from the shell
    # must not silently hand the API a GPU. API_ALLOW_GPU is the one way out.
    _os.environ["CUDA_VISIBLE_DEVICES"] = ""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import Response
from starlette.requests import Request as _StarletteRequest
from contextlib import asynccontextmanager
from pathlib import Path
from sqlalchemy import text
from alembic.config import Config as _AlembicConfig
from alembic.script import ScriptDirectory as _AlembicScriptDirectory
import asyncio
import logging

# Raise multipart upload limits above Starlette's defaults (1000 files / 1000 fields)
# so dataset batch uploads can include up to 5000 files per request. FastAPI 0.111 calls
# `request.form()` with no args, so the only effective lever is the default on
# Request.form itself — patching MultiPartParser class attrs has no effect because
# `max_files`/`max_fields` are per-instance kwargs in Starlette 0.37+.
_UPLOAD_MAX_FILES = 5000
_UPLOAD_MAX_FIELDS = 5000
_original_form = _StarletteRequest.form


def _form_with_higher_limits(
    self,
    *,
    max_files: int | float = _UPLOAD_MAX_FILES,
    max_fields: int | float = _UPLOAD_MAX_FIELDS,
):
    return _original_form(self, max_files=max_files, max_fields=max_fields)


_StarletteRequest.form = _form_with_higher_limits

from app.core.config import settings
from app.core.database import engine
from app.core.logging_config import (
    configure_logging,
    get_logger,
    request_id_var,
    log_event,
)
from app.api.v1 import router as api_router
from app.realtime.ws_device import router as ws_router
from app.realtime.ws_studio import router as ws_studio_router
from app.realtime.commands import send_stop_snapshot, send_stop_inference_stream
from app.realtime.connection_manager import manager
from app.realtime.studio_events import emit, EVENT_STREAM_FAILED
from app.services.stream_store import stream_store, StreamType

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Structured per-channel JSON logging (logs/api.log, etc.). Configured at import
# time so the API process is ready to log before the first request arrives.
configure_logging()
api_log = get_logger("api")


async def _stream_expiry_loop() -> None:
    """Every 10 s, expire streams that missed their keepalive window."""
    while True:
        await asyncio.sleep(10)
        for device_id, state in list(stream_store._store.items()):
            if stream_store.expired(device_id):
                stream_store.stop(device_id)
                if manager.is_connected(state.project_id, state.device_id):
                    if state.stream_type == StreamType.snapshot:
                        await send_stop_snapshot(state.project_id, state.device_id)
                    else:
                        await send_stop_inference_stream(state.project_id, state.device_id)
                await emit(
                    state.project_id,
                    EVENT_STREAM_FAILED,
                    {"reason": "keepalive_timeout"},
                    device_id=state.device_id,
                )


_BACKEND_DIR = Path(__file__).resolve().parent.parent


def _current_alembic_head() -> str | None:
    """The migration chain's head revision, per alembic/versions/ on disk."""
    config = _AlembicConfig(str(_BACKEND_DIR / "alembic.ini"))
    # script_location in alembic.ini is relative to the process's cwd, which
    # is not reliable here (uvicorn may be launched from anywhere) — resolve
    # it explicitly against this file's own location instead.
    config.set_main_option("script_location", str(_BACKEND_DIR / "alembic"))
    return _AlembicScriptDirectory.from_config(config).get_current_head()


def _check_schema_migrated(engine) -> None:
    """Assert the live database is stamped at the migration chain's head.

    Base.metadata.create_all() no longer runs at startup (see 2026-09-06 fix:
    two sources of truth for the schema, with the faster one winning). Alembic
    is now the only schema authority, so a database that hasn't been migrated
    to head is a startup-time configuration error, not something the app
    should paper over.
    """
    expected_head = _current_alembic_head()
    try:
        with engine.connect() as conn:
            current = conn.execute(text("SELECT version_num FROM alembic_version")).scalar()
    except Exception as e:
        current = None
        logger.error(f"Could not read alembic_version — has this database been migrated? ({e})")

    if current != expected_head:
        logger.error(
            "Schema drift detected: database is at alembic revision "
            f"{current!r}, but the migration chain's head is {expected_head!r}. "
            "Run `alembic upgrade head` against this database."
        )
        if settings.FAIL_ON_SCHEMA_DRIFT:
            raise RuntimeError(
                f"Database schema is not at head (db={current!r}, head={expected_head!r}) "
                "and FAIL_ON_SCHEMA_DRIFT is set — refusing to start."
            )


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup — verify the database is reachable and on the expected schema.
    # Alembic migrations are the only schema authority; this process no
    # longer creates or alters tables itself.
    logger.info("Starting PetalEdge API...")
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        logger.info(f"Database connected: {settings.POSTGRES_HOST}:{settings.POSTGRES_PORT}/{settings.POSTGRES_DB}")
    except Exception as e:
        logger.error(f"Database connection failed: {e}")
        raise
    _check_schema_migrated(engine)
    expiry_task = asyncio.create_task(_stream_expiry_loop())
    yield
    # Shutdown
    expiry_task.cancel()
    logger.info("Shutting down PetalEdge API...")


app = FastAPI(
    title="PetalEdge API",
    description="Open-source Edge AI Platform API",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    # Response headers are NOT readable cross-origin by default beyond the
    # CORS-safelisted set (Content-Type, Content-Length, ...) — Content-Disposition
    # is not in that set, so without this, JS on a different origin/port
    # (e.g. the Next.js dev server) cannot read the filename that download
    # routes (trained-models/download, dsp/features/download, projects/export)
    # send back, even though curl/TestClient see it fine.
    expose_headers=["Content-Disposition"],
)
app.add_middleware(GZipMiddleware, minimum_size=1000)


@app.middleware("http")
async def _request_logging_middleware(request: _StarletteRequest, call_next):
    """Log every HTTP request to api.log and propagate a request id.

    Reuses an inbound ``X-Request-ID`` header when present (so a request id set
    by a gateway/frontend flows through), otherwise mints one. The id is echoed
    back on the response and stashed in a context var so DB/storage logs emitted
    while serving the request carry the same id. Behaviour-preserving: the
    downstream response is returned untouched aside from the added header.
    """
    import time as _time
    import uuid as _uuid

    request_id = request.headers.get("X-Request-ID") or _uuid.uuid4().hex
    token = request_id_var.set(request_id)
    start = _time.perf_counter()
    status_code = 500  # assume failure until a response is produced
    try:
        response = await call_next(request)
        status_code = response.status_code
        response.headers["X-Request-ID"] = request_id
        return response
    finally:
        # Single emission point so every request — success, 4xx, 5xx, or raised
        # exception (status stays 500) — is logged exactly once. The request is
        # returned/re-raised untouched, preserving all existing behaviour.
        elapsed_ms = (_time.perf_counter() - start) * 1000.0
        if status_code >= 500:
            level = logging.ERROR
        elif status_code >= 400:
            level = logging.WARNING
        else:
            level = logging.INFO
        route = request.scope.get("route")
        log_event(
            api_log,
            "api.request",
            level=level,
            request_id=request_id,
            method=request.method,
            route=getattr(route, "path", None) or request.url.path,
            path=request.url.path,
            status=status_code,
            response_time_ms=round(elapsed_ms, 2),
        )
        request_id_var.reset(token)

# Routes
app.include_router(api_router,        prefix="/api/v1")
app.include_router(ws_router,         prefix="/ws", tags=["realtime"])
app.include_router(ws_studio_router,  prefix="/ws", tags=["realtime"])


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "version": "1.0.0",
        "database": f"{settings.POSTGRES_HOST}/{settings.POSTGRES_DB}",
    }


@app.get("/")
async def root():
    return {"message": "petaledge API", "docs": "/docs"}


# ─── Device-client bootstrap installer (contract §14.3/§14.5) ────────────────
# Mounted at the ROOT, outside the /api/v1 router and its auth, so the one
# command `curl -fsSL https://<host>/install.sh | bash` works with no clone and
# no credentials. The backend host is interpolated from the incoming request so
# the script points every fetch back at the same host the device reached us on.
#
# The served script is the network-fetching installer maintained as a real,
# lintable shell file at device_client/bootstrap_install.sh — it implements the
# 7-step §14.5 flow (detect → fetch manifest → download → verify sha256 →
# extract → install/start service → provision). We only substitute the backend
# host (the @@…@@ placeholder, which appears exactly once, on the BACKEND_HOST
# line) at serve time.
from pathlib import Path as _Path

_BOOTSTRAP_INSTALLER = (
    _Path(__file__).resolve().parents[2] / "device_client" / "bootstrap_install.sh"
)
_INSTALL_HOST_TOKEN = "@@PETAL_BACKEND_HOST@@"


@app.get("/install.sh")
async def install_sh(request: _StarletteRequest):
    """Public bootstrap installer script (contract §14.3). `text/x-shellscript`."""
    try:
        script = _BOOTSTRAP_INSTALLER.read_text(encoding="utf-8")
    except OSError as exc:  # pragma: no cover - deployment/packaging misconfig
        logger.error("bootstrap installer not readable at %s: %s", _BOOTSTRAP_INSTALLER, exc)
        raise HTTPException(status_code=500, detail="installer unavailable")
    backend_host = str(request.base_url).rstrip("/")
    script = script.replace(_INSTALL_HOST_TOKEN, backend_host)
    return Response(content=script, media_type="text/x-shellscript")
