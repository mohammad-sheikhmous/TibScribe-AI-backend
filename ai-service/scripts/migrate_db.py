"""Upgrade the AI database safely, including legacy unversioned databases.

Older TibScribe builds called ``Base.metadata.create_all`` at startup but did not run
Alembic.  A database created by one of those builds therefore has real tables but no
``alembic_version`` marker. Running ``alembic upgrade head`` blindly against it tries
recreating the initial tables and fails.

This bootstrap inspects only schema features introduced by our linear migration
history, stamps the highest *contiguous* revision already represented on disk, then
lets Alembic perform the remaining upgrades. Fresh databases go through Alembic from
revision zero. No application rows are rewritten here.
"""
from __future__ import annotations

import sys
from pathlib import Path

from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, inspect, text

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.config import get_settings  # noqa: E402

BASE = "e3e71370f97f"
PATIENT_STATE = "5510797204e4"
ASR_QUALITY = "5059d7197646"
UNCERTAINTY = "8f8ec9d933cd"
GATEWAY = "20260814gateway"
SUGGEST_ACTIVE = "20260814suggestactive"
HEAD = "20260819textraw"

CORE_TABLES = {"patients", "jobs", "reports", "report_items", "suggestions"}


def _columns(inspector, table: str) -> set[str]:
    return {str(col["name"]) for col in inspector.get_columns(table)}


def detect_unversioned_revision(inspector) -> str | None:
    """Return the newest contiguous migration already reflected by a legacy schema.

    ``None`` means there are no TibScribe application tables yet. A partial/unknown
    schema is rejected rather than stamped optimistically, because a wrong stamp can
    silently leave clinical data on an incompatible layout.
    """
    tables = set(inspector.get_table_names())
    if not (tables & CORE_TABLES):
        return None
    if not CORE_TABLES.issubset(tables):
        missing = sorted(CORE_TABLES - tables)
        raise RuntimeError(
            "Unversioned AI database has an incomplete core schema; missing: "
            + ", ".join(missing)
        )

    revision = BASE
    if "patient_states" not in tables:
        return revision
    revision = PATIENT_STATE

    item_cols = _columns(inspector, "report_items")
    asr_cols = {"combined_confidence", "speaker_confidence", "is_asr_suspect"}
    if not asr_cols.issubset(item_cols):
        return revision
    revision = ASR_QUALITY

    uncertainty_cols = {
        "labels", "also_in_sections", "entropy", "ood_score", "low_confidence_reasons"
    }
    if not uncertainty_cols.issubset(item_cols):
        return revision
    revision = UNCERTAINTY

    patient_cols = _columns(inspector, "patients")
    job_cols = _columns(inspector, "jobs")
    gateway_patient = {"external_source", "external_id"}
    gateway_job = {"external_session_id", "external_doctor_id"}
    if not (gateway_patient.issubset(patient_cols) and gateway_job.issubset(job_cols)):
        return revision
    revision = GATEWAY

    suggestion_cols = _columns(inspector, "suggestions")
    if "is_active" not in suggestion_cols:
        return revision
    revision = SUGGEST_ACTIVE

    # Canonicalization provenance: the original Whisper/ASR sentence is stored
    # beside the canonical text consumed by AraBERT.
    if "text_raw" in item_cols:
        revision = HEAD
    return revision


def _existing_alembic_revision(engine) -> str | None:
    inspector = inspect(engine)
    if "alembic_version" not in inspector.get_table_names():
        return None
    with engine.connect() as conn:
        row = conn.execute(text("SELECT version_num FROM alembic_version LIMIT 1")).first()
    return str(row[0]) if row else None


def migrate() -> None:
    settings = get_settings()
    url = settings.migration_database_url_resolved
    engine = create_engine(
        url,
        connect_args={"check_same_thread": False} if url.startswith("sqlite") else {},
        future=True,
    )
    try:
        current = _existing_alembic_revision(engine)
        cfg = Config(str(ROOT / "alembic.ini"))
        cfg.set_main_option("script_location", str(ROOT / "app" / "db" / "migrations"))

        if current is None:
            inspector = inspect(engine)
            revision = detect_unversioned_revision(inspector)
            if revision is not None:
                print(f"Legacy unversioned AI schema detected; stamping {revision}.")
                command.stamp(cfg, revision)

        command.upgrade(cfg, "head")
        print("AI database migrations are at head.")
    finally:
        engine.dispose()


if __name__ == "__main__":
    migrate()
