"""Engine + session management (IMPLEMENTATION.md P1-02).

SQLite by design for v2: one file, zero operational overhead. Everything goes through
SQLAlchemy, so moving to PostgreSQL later is a `DATABASE_URL` change, not a rewrite.

Two SQLite settings are not optional here:
  * **WAL** — the API thread reads job status while the pipeline worker thread writes
    results. Without WAL those block each other and status polling stalls.
  * **foreign_keys=ON** — SQLite ignores foreign keys unless asked, which would let
    orphan report items outlive their job.

The engine is created lazily and cached so tests can point `DATABASE_URL` at a temp
file and call `reset_engine()`.
"""
from __future__ import annotations

import logging
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path
from typing import Iterator

from sqlalchemy import Engine, create_engine, event
from sqlalchemy.orm import Session, sessionmaker

from ..config import get_settings
from .models import Base

logger = logging.getLogger(__name__)


def _apply_sqlite_pragmas(dbapi_connection, _record) -> None:
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.execute("PRAGMA synchronous=NORMAL")  # WAL + NORMAL is durable enough here
    cursor.execute("PRAGMA busy_timeout=30000")  # wait for short concurrent writes instead of spurious lock errors
    cursor.close()


@lru_cache
def get_engine() -> Engine:
    settings = get_settings()
    url = settings.database_url_resolved
    is_sqlite = url.startswith("sqlite")

    if is_sqlite:
        path = url.split("///", 1)[-1]
        if path and path != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)

    engine_kwargs = {
        "connect_args": {"check_same_thread": False, "timeout": 30} if is_sqlite else {},
        "echo": settings.db_echo,
        "future": True,
    }
    if not is_sqlite:
        # Serverless Postgres endpoints can suspend/resume underneath an idle client.
        # Validate pooled connections before reuse and recycle long-idle handles.
        engine_kwargs.update(pool_pre_ping=True, pool_recycle=300, pool_size=2, max_overflow=3)

    engine = create_engine(url, **engine_kwargs)
    if is_sqlite:
        event.listen(engine, "connect", _apply_sqlite_pragmas)
    logger.info("Database engine ready: %s", url)
    return engine


@lru_cache
def get_session_factory() -> sessionmaker[Session]:
    return sessionmaker(bind=get_engine(), expire_on_commit=False, future=True)


@contextmanager
def session_scope() -> Iterator[Session]:
    """Transactional scope: commit on success, roll back on failure, always close."""
    session = get_session_factory()()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def init_db() -> None:
    """Create any missing tables.

    Alembic owns schema *evolution* (`alembic upgrade head`); this is the safety net so
    a fresh checkout, a test run, or a first start never fails on a missing table.
    """
    Base.metadata.create_all(bind=get_engine())


def reset_engine() -> None:
    """Drop the cached engine/session factory — for tests that switch DATABASE_URL."""
    try:
        get_engine().dispose()
    except Exception:  # pragma: no cover - engine may not exist yet
        pass
    get_engine.cache_clear()
    get_session_factory.cache_clear()
