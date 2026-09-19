"""
Database engine, session, and base model
"""
import logging
import time

from sqlalchemy import create_engine, event
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from app.core.config import settings
from app.core.logging_config import (
    get_logger,
    log_event,
    DB_LOG_ALL,
    DB_SLOW_QUERY_MS,
)

engine = create_engine(
    settings.DATABASE_URL,
    pool_pre_ping=True,
    pool_size=10,
    max_overflow=20,
)


# ─── Query timing → database.log ──────────────────────────────────────────────
# Slow queries (> DB_SLOW_QUERY_MS) are always logged at WARNING. Every query is
# logged at DEBUG only when DB_LOG_ALL is enabled, to avoid flooding the file on
# the hot path. Listeners are pure measurement — they never alter query results.
_db_log = get_logger("database")


def _truncate_sql(statement: str, limit: int = 500) -> str:
    sql = " ".join(statement.split())
    return sql if len(sql) <= limit else sql[:limit] + "…"


@event.listens_for(engine, "before_cursor_execute")
def _db_before_cursor_execute(conn, cursor, statement, parameters, context, executemany):
    conn.info.setdefault("_query_start_stack", []).append(time.perf_counter())


@event.listens_for(engine, "after_cursor_execute")
def _db_after_cursor_execute(conn, cursor, statement, parameters, context, executemany):
    stack = conn.info.get("_query_start_stack")
    if not stack:
        return
    elapsed_ms = (time.perf_counter() - stack.pop()) * 1000.0
    is_slow = elapsed_ms >= DB_SLOW_QUERY_MS
    if not is_slow and not DB_LOG_ALL:
        return
    log_event(
        _db_log,
        "db.slow_query" if is_slow else "db.query",
        level=logging.WARNING if is_slow else logging.DEBUG,
        duration_ms=round(elapsed_ms, 2),
        slow=is_slow or None,
        executemany=executemany or None,
        statement=_truncate_sql(statement),
    )

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


def get_db():
    """FastAPI dependency — yields a DB session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
