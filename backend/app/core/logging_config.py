"""
Structured logging configuration for the Petal Edge backend.

Provides seven isolated, per-channel JSON loggers, each writing to its own
daily-rotating file under ``<backend>/logs/``:

    api.log         training.log    dsp.log         deployment.log
    celery.log      database.log    storage.log

Design goals
------------
* **Concurrency-safe rotation** — uses ``concurrent-log-handler`` so the API
  process and every Celery worker can hold a handler on the same channel file;
  the midnight rollover is serialised through a lock file. (The stdlib
  ``TimedRotatingFileHandler`` raised ``PermissionError: [WinError 32]`` on
  Windows because rotation renames a file another process still has open.)
  No structlog / python-json-logger dependency — JSON is still hand-rolled.
* **One JSON object per line** so the files are trivially ``jq``-able and
  ingestible by Loki / ELK / CloudWatch without a parser.
* **Isolated channels** — each channel logger has ``propagate = False`` and its
  own handler, so writing to ``training.log`` never leaks into the root logger,
  the console, or any other channel file. Existing ``logging.basicConfig`` /
  ``getLogger(__name__)`` behaviour elsewhere in the app is left untouched.
* **request_id propagation** — a context variable is injected into every record
  via a logging filter, so a DB or storage log emitted while handling an HTTP
  request automatically carries the same ``request_id`` as the ``api.log`` line.

Usage
-----
    from app.core.logging_config import get_logger, log_event

    log = get_logger("training")
    log_event(log, "training.completed", job_id=job_id, status="completed",
              train_ms=1234.5)
"""
from __future__ import annotations

import contextvars
import json
import logging
import os
from datetime import datetime, timezone
from concurrent_log_handler import ConcurrentTimedRotatingFileHandler
from pathlib import Path
from typing import Any

# ─── Constants ────────────────────────────────────────────────────────────────

# logs/ lives at the backend root. This file is at backend/app/core/, so
# parents[2] == backend/ regardless of the process working directory.
LOG_DIR = Path(__file__).resolve().parents[2] / "logs"

_LOGGER_PREFIX = "petaledge"
BACKUP_COUNT = 7             # keep 7 days (one week) of rotated logs per channel
DB_SLOW_QUERY_MS = 500.0     # queries slower than this are flagged "slow"

# Channel → log filename.
_CHANNELS: dict[str, str] = {
    "api":        "api.log",
    "training":   "training.log",
    "dsp":        "dsp.log",
    "deployment": "deployment.log",
    "celery":     "celery.log",
    "database":   "database.log",
    "storage":    "storage.log",
}

# Whether per-query DEBUG rows are written to database.log. Slow queries
# (> DB_SLOW_QUERY_MS) are ALWAYS logged regardless of this flag.
DB_LOG_ALL = os.getenv("DB_LOG_ALL", "false").strip().lower() in ("1", "true", "yes", "on")

# Whether successful per-operation rows are written to storage.log. During bulk
# DSP each sample download would otherwise emit a line (one locked file write per
# op), so success logging is opt-in. Failures are ALWAYS logged regardless.
STORAGE_LOG_ALL = os.getenv("STORAGE_LOG_ALL", "false").strip().lower() in ("1", "true", "yes", "on")

# Standard LogRecord attributes. Anything *not* in this set found on a record is
# treated as caller-supplied structured context and merged into the JSON output.
_RESERVED_ATTRS = set(vars(logging.makeLogRecord({}))) | {
    "message", "asctime", "taskName",
}


# ─── request_id propagation ───────────────────────────────────────────────────

# Set by the API middleware at the start of each request; read by the filter
# below so every log line emitted during that request carries the id.
request_id_var: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "request_id", default=None
)


class RequestContextFilter(logging.Filter):
    """Inject the current request_id (if any) onto every record."""

    def filter(self, record: logging.LogRecord) -> bool:  # noqa: A003
        if not hasattr(record, "request_id"):
            rid = request_id_var.get()
            if rid is not None:
                record.request_id = rid
        return True


# ─── JSON formatter ───────────────────────────────────────────────────────────

class JsonFormatter(logging.Formatter):
    """Render a LogRecord as a single-line JSON object.

    The log *message* is used as the ``event`` name and any keyword fields passed
    via ``extra={...}`` are merged in as top-level keys.
    """

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "channel": record.name.split(".", 1)[-1],
            "event": record.getMessage(),
        }
        # Merge structured context supplied via `extra=` (and request_id from the
        # filter). Skip private/dunder keys and standard LogRecord attributes.
        for key, value in record.__dict__.items():
            if key in _RESERVED_ATTRS or key.startswith("_"):
                continue
            payload[key] = value
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        if record.stack_info:
            payload["stack_info"] = self.formatStack(record.stack_info)
        return json.dumps(payload, default=str, ensure_ascii=False)


# ─── Configuration ────────────────────────────────────────────────────────────

_configured = False


def configure_logging(force: bool = False) -> None:
    """Build the seven channel loggers and their file handlers (idempotent).

    Safe to call from the API process, every Celery worker process, and repeatedly
    — duplicate handlers are never attached. Uses ``delay=True`` so a channel's
    file is only created the first time something is actually logged to it.
    """
    global _configured
    if _configured and not force:
        return

    # A logging handler must never raise into application code (DSP jobs, storage
    # downloads, Celery tasks). The stdlib already routes emit-time failures to
    # Handler.handleError rather than re-raising; this additionally silences the
    # noisy stderr traceback that handleError prints by default, so a transient
    # logging hiccup can't masquerade as an application error in the logs.
    logging.raiseExceptions = False

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    formatter = JsonFormatter()
    ctx_filter = RequestContextFilter()

    for channel, filename in _CHANNELS.items():
        logger = logging.getLogger(f"{_LOGGER_PREFIX}.{channel}")
        logger.setLevel(logging.DEBUG)
        # Isolated: never bubble up to the root logger / console handlers.
        logger.propagate = False

        already = any(
            getattr(h, "_petaledge_channel", None) == channel
            for h in logger.handlers
        )
        if already:
            continue

        # ConcurrentTimedRotatingFileHandler is a drop-in replacement for the
        # stdlib TimedRotatingFileHandler that is safe when the API process and
        # every Celery worker (training / dsp / deployment / …) open a handler on
        # the *same* channel file. It serialises writes and the midnight rollover
        # through a sibling lock file, so the rename no longer collides with
        # another process's open handle — which is what produced
        # ``PermissionError: [WinError 32]`` on Windows (POSIX never locks open
        # files, hence the bug was Windows-only).
        handler = ConcurrentTimedRotatingFileHandler(
            LOG_DIR / filename,
            when="midnight",
            backupCount=BACKUP_COUNT,
            utc=True,
            encoding="utf-8",
            delay=True,
            use_gzip=True,   # compress each file as it rotates -> .log.<date>.gz
        )
        handler.setFormatter(formatter)
        handler.addFilter(ctx_filter)
        handler._petaledge_channel = channel  # type: ignore[attr-defined]
        logger.addHandler(handler)

    _configured = True


def get_logger(channel: str) -> logging.Logger:
    """Return the JSON logger for ``channel`` (one of the seven channels).

    Calls ``configure_logging()`` on first use so callers never have to remember
    to bootstrap logging themselves.
    """
    if channel not in _CHANNELS:
        raise ValueError(
            f"Unknown log channel {channel!r}; expected one of {sorted(_CHANNELS)}"
        )
    configure_logging()
    return logging.getLogger(f"{_LOGGER_PREFIX}.{channel}")


def log_event(
    logger: logging.Logger,
    event: str,
    *,
    level: int = logging.INFO,
    **fields: Any,
) -> None:
    """Emit a structured event: ``event`` becomes the message, ``fields`` the JSON body.

    ``None`` values are dropped so optional fields don't clutter the output.
    """
    clean = {k: v for k, v in fields.items() if v is not None}
    logger.log(level, event, extra=clean)
