from __future__ import annotations

from types import SimpleNamespace

from app.config import Settings
from app.jobs.runner import _commit_external_storage


def test_neon_postgresql_url_uses_psycopg3_dialect():
    s = Settings(database_url="postgresql://u:p@example.neon.tech/db?sslmode=require")
    assert s.database_url_resolved.startswith("postgresql+psycopg://")


def test_legacy_postgres_url_uses_psycopg3_dialect():
    s = Settings(database_url="postgres://u:p@example.neon.tech/db")
    assert s.database_url_resolved.startswith("postgresql+psycopg://")


def test_migration_url_can_be_direct_while_runtime_is_pooled():
    s = Settings(
        database_url="postgresql://u:p@host-pooler/db",
        migration_database_url="postgresql://u:p@host/db",
    )
    assert "host-pooler" in s.database_url_resolved
    assert "@host/db" in s.migration_database_url_resolved


def test_storage_commit_hook_is_optional_and_called_when_present():
    calls = []
    app = SimpleNamespace(state=SimpleNamespace(storage_commit_hook=lambda: calls.append("commit")))
    _commit_external_storage(app)
    assert calls == ["commit"]

    app_without_hook = SimpleNamespace(state=SimpleNamespace())
    _commit_external_storage(app_without_hook)
