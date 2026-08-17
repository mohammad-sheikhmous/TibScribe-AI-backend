from __future__ import annotations

from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, inspect, text

from app.config import get_settings
from scripts import migrate_db


@pytest.mark.parametrize(
    "legacy_revision",
    [
        None,
        migrate_db.BASE,
        migrate_db.PATIENT_STATE,
        migrate_db.ASR_QUALITY,
        migrate_db.UNCERTAINTY,
        migrate_db.GATEWAY,
        migrate_db.HEAD,
    ],
)
def test_migration_bootstrap_upgrades_fresh_and_unversioned_legacy_schemas(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, legacy_revision: str | None
) -> None:
    """Every historically supported schema must reach the one canonical head.

    Older TibScribe builds used ``create_all`` and therefore had no
    ``alembic_version`` table.  We reproduce that state by creating each historical
    revision and deleting the version marker before invoking the production bootstrap.
    """
    db = tmp_path / f"legacy-{legacy_revision or 'fresh'}.db"
    url = f"sqlite:///{db.as_posix()}"
    monkeypatch.setenv("DATABASE_URL", url)
    get_settings.cache_clear()

    cfg = Config(str(Path(migrate_db.ROOT / "alembic.ini")))
    cfg.set_main_option(
        "script_location", str(migrate_db.ROOT / "app" / "db" / "migrations")
    )

    if legacy_revision is not None:
        command.upgrade(cfg, legacy_revision)
        engine = create_engine(url, future=True)
        try:
            with engine.begin() as conn:
                conn.execute(text("DROP TABLE alembic_version"))
        finally:
            engine.dispose()

    migrate_db.migrate()

    engine = create_engine(url, future=True)
    try:
        inspector = inspect(engine)
        with engine.connect() as conn:
            current = conn.execute(
                text("SELECT version_num FROM alembic_version LIMIT 1")
            ).scalar_one()
        assert current == migrate_db.HEAD
        assert "is_active" in {c["name"] for c in inspector.get_columns("suggestions")}
        assert {"external_source", "external_id"} <= {
            c["name"] for c in inspector.get_columns("patients")
        }
        assert {"external_session_id", "external_doctor_id"} <= {
            c["name"] for c in inspector.get_columns("jobs")
        }
    finally:
        engine.dispose()
        get_settings.cache_clear()


def test_migration_bootstrap_does_not_disable_existing_application_loggers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Alembic logging config must not disable loggers already owned by the app.

    ``logging.config.fileConfig`` defaults to ``disable_existing_loggers=True``.  If
    migration bootstrap runs in-process (tests, admin tooling, or a future embedded
    startup), that default can silently turn off clinical/report warnings.
    """
    import logging

    db = tmp_path / "logging-preservation.db"
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db.as_posix()}")
    get_settings.cache_clear()

    app_logger = logging.getLogger("app.core.nlp.sections")
    app_logger.disabled = False
    migrate_db.migrate()

    assert app_logger.disabled is False
    get_settings.cache_clear()
