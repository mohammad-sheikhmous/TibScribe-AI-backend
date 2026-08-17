"""Central configuration for the medical-scribe service.

All values can be overridden via environment variables or a local .env file
(see .env.example). Safety-first defaults are used. The final service requires a complete, validated
checkpoint in model_output/ and refuses to serve an untrained classifier head.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# Repository root = two levels up from this file (app/config.py -> app -> root).
ROOT_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        protected_namespaces=(),  # allow field names like "model_dir" without warnings
    )

    # --- Service-to-service security ---
    # Laravel is the only public backend. Every business endpoint in this AI service
    # requires this shared service credential; /health and /ready stay public for
    # orchestration probes. Use a random value >= 32 chars in real deployments.
    service_auth_required: bool = True
    service_token: str = ""
    service_name: str = "laravel"
    gateway_identity_required: bool = True

    # --- Models ---
    arobert_model_name: str = "aubmindlab/bert-base-arabertv02"
    model_dir: str = "model_output"
    whisper_model_size: str = "medium"
    asr_language: str = "ar"

    # --- Classifier ---
    # Fallback only: model_config.json (written at train time) always wins.
    default_max_len: int = 128
    low_confidence_threshold: float = 0.5
    # Production guard (IMPLEMENTATION.md P0-09): turn checkpoint trust problems
    # (preprocessing skew, missing weights/mapping) into a startup failure instead
    # of a log line that nobody reads.
    # Final build: never serve predictions from a missing/mismatched/random head.
    strict_model_checks: bool = True

    # --- Uncertainty (P4) ---
    # MC-dropout runs only on sentences below the classifier's own confidence screen,
    # so this is cheap on a typical visit. Disable to fall back to plain batching.
    uncertainty_enabled: bool = True
    mc_passes: int = 8

    # --- Speaker diarization (P3) ---
    # Off by default: the pyannote model is gated behind an accepted licence + token.
    # When disabled the pipeline still runs, with speakers left unattributed.
    diarization_enabled: bool = False
    hf_token: str = ""
    # Rebuild the doctor's voiceprint every N archived recordings.
    voiceprint_refresh_every: int = 100
    # Shortest segment kept on its own; anything smaller is merged into a neighbour.
    min_segment_chars: int = 20

    # --- Segmentation ---
    # 140 chars keeps served segments inside the length distribution the classifier
    # was trained on (data max ~135 chars, p95 ~100). 220 fed it out-of-distribution
    # inputs 63% longer than anything it ever saw. See PLAN_V2.md P0.
    segment_max_chars: int = 140
    word_pause_gap_sec: float = 0.5

    # --- Storage ---
    upload_dir: str = "data/uploads"
    result_dir: str = "data/results"
    # Permanent, content-addressed audio store. NEVER pruned — see PLAN_V2.md §7.
    audio_dir: str = "data/audio"
    # Warn (never delete) once the audio store passes this size.
    audio_warn_gb: float = 50.0

    # --- Database (P1) ---
    # Empty = SQLite at data/scribe.db. Any SQLAlchemy URL works, so moving to
    # PostgreSQL is a config change rather than a code change.
    database_url: str = ""
    # Optional direct/admin URL for Alembic. If empty, DATABASE_URL is used.
    # Useful with Neon: pooled URL for runtime, direct URL for migrations.
    migration_database_url: str = ""
    db_echo: bool = False

    # --- Final EXPERTA_MED KBS ---
    # Number of prior completed reports supplied for explicit temporal trends.
    # Ordinary clinical rules still use only the current report.
    kbs_history_reports: int = 5

    # --- Jobs / concurrency ---
    executor_max_workers: int = 1
    max_concurrent_jobs: int = 4
    max_upload_mb: int = 100

    # --- Resolved absolute paths (computed helpers) ---
    @property
    def model_dir_path(self) -> Path:
        return self._abs(self.model_dir)

    @property
    def upload_dir_path(self) -> Path:
        return self._abs(self.upload_dir)

    @property
    def result_dir_path(self) -> Path:
        return self._abs(self.result_dir)

    @property
    def audio_dir_path(self) -> Path:
        return self._abs(self.audio_dir)

    @staticmethod
    def _sqlalchemy_url(value: str) -> str:
        """Normalize provider-style Postgres URLs to the psycopg3 SQLAlchemy dialect.

        Neon (and many PaaS providers) expose ``postgresql://...`` connection strings.
        SQLAlchemy otherwise tries the legacy psycopg2 driver. The cloud image ships
        psycopg3, so make the driver explicit while leaving SQLite/other URLs intact.
        """
        if value.startswith("postgresql://"):
            return "postgresql+psycopg://" + value[len("postgresql://"):]
        if value.startswith("postgres://"):
            return "postgresql+psycopg://" + value[len("postgres://"):]
        return value

    @property
    def database_url_resolved(self) -> str:
        """Explicit DATABASE_URL, else a SQLite file next to the other data dirs."""
        if self.database_url:
            return self._sqlalchemy_url(self.database_url)
        return f"sqlite:///{(ROOT_DIR / 'data' / 'scribe.db').as_posix()}"

    @property
    def migration_database_url_resolved(self) -> str:
        """Direct/admin migration URL when configured, otherwise the runtime URL."""
        return self._sqlalchemy_url(self.migration_database_url) if self.migration_database_url else self.database_url_resolved

    @staticmethod
    def _abs(value: str) -> Path:
        p = Path(value)
        return p if p.is_absolute() else (ROOT_DIR / p)


@lru_cache
def get_settings() -> Settings:
    return Settings()
