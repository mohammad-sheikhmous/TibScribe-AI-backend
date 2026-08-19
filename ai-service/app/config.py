"""Central configuration for the medical-scribe service.

All values can be overridden via environment variables or a local .env file
(see .env.example). Safety-first defaults are used.

The final service requires a complete, validated checkpoint in model_output/
and refuses to serve an untrained classifier head.
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
        protected_namespaces=(),
    )

    # -------------------------------------------------------------------------
    # Service-to-service security
    # -------------------------------------------------------------------------

    # Laravel is the only public backend. Every business endpoint in this AI
    # service requires this shared service credential; /health and /ready stay
    # public for orchestration probes.
    #
    # Use a random value >= 32 chars in real deployments.
    service_auth_required: bool = True
    service_token: str = ""
    service_name: str = "laravel"
    gateway_identity_required: bool = True

    # -------------------------------------------------------------------------
    # Models
    # -------------------------------------------------------------------------

    arobert_model_name: str = "aubmindlab/bert-base-arabertv02"
    model_dir: str = "model_output"

    whisper_model_size: str = "medium"
    asr_language: str = "ar"

    # -------------------------------------------------------------------------
    # Classifier
    # -------------------------------------------------------------------------

    # Fallback only: model_config.json (written at train time) always wins.
    default_max_len: int = 128

    low_confidence_threshold: float = 0.5

    # Production guard:
    # never serve predictions from a missing/mismatched/random classifier head.
    strict_model_checks: bool = True

    # -------------------------------------------------------------------------
    # Uncertainty (P4)
    # -------------------------------------------------------------------------

    # MC-dropout runs only on sentences below the classifier's own confidence
    # screen, so this is cheap on a typical visit.
    uncertainty_enabled: bool = True
    mc_passes: int = 8

    # -------------------------------------------------------------------------
    # Speaker diarization (P3)
    # -------------------------------------------------------------------------

    # Off by default: the pyannote model is gated behind an accepted licence
    # + token.
    diarization_enabled: bool = False
    hf_token: str = ""

    # Rebuild the doctor's voiceprint every N archived recordings.
    voiceprint_refresh_every: int = 100

    # Shortest segment kept on its own; anything smaller is merged into a
    # neighbour.
    min_segment_chars: int = 20

    # -------------------------------------------------------------------------
    # Segmentation
    # -------------------------------------------------------------------------

    # 140 chars keeps served segments inside the length distribution the
    # classifier was trained on.
    segment_max_chars: int = 140

    word_pause_gap_sec: float = 0.5

    # -------------------------------------------------------------------------
    # Medical canonicalization — PRE-AraBERT
    # -------------------------------------------------------------------------
    #
    # This stage sits between segmentation and AraBERT.
    #
    # Its purpose is to convert dialectal / noisy Whisper text into cleaner,
    # medically-safe Arabic closer to the distribution used for AraBERT
    # fine-tuning.
    #
    # Example:
    #
    #   "في عندها وجع راس خفيف"
    #
    # becomes:
    #
    #   "تعاني من صداع خفيف"
    #
    # while preserving the original text in Segment.text_raw.
    #
    # Disabled by default so existing deployments continue to behave exactly
    # as before until the LLM credentials are intentionally configured.

    canonicalization_enabled: bool = False

    # Provider/backend name.
    #
    # We keep this configurable so the NLP pipeline is not permanently tied to
    # one LLM vendor.
    #
    # The client implementation will interpret this value later.
    canonicalization_provider: str = "openai_compatible"

    # LLM model used ONLY for canonicalization.
    #
    # Keep empty by default. Production startup/runtime logic will require this
    # value when canonicalization_enabled=True.
    canonicalization_model: str = ""

    # Secret API credential.
    #
    # Never commit the real value to Git.
    # On Modal this should later be supplied through Modal Secrets.
    canonicalization_api_key: str = ""

    # Optional provider base URL.
    #
    # Keeping it configurable allows an OpenAI-compatible provider or a future
    # internal/local model endpoint to be used without changing pipeline code.
    canonicalization_base_url: str = ""

    # Maximum time allowed for one LLM request.
    # If it fails/times out, canonicalization.py safely falls back to the
    # original ASR text.
    canonicalization_timeout_sec: float = 20.0

    # Number of transcript segments sent to the LLM in one request.
    #
    # canonicalization.py currently defaults to 12; this setting lets us tune
    # it from the environment without changing code.
    canonicalization_batch_size: int = 12

    # Whisper segments already marked as suspicious should normally NOT be
    # rewritten by an LLM, because that could turn an ASR hallucination into a
    # fluent and convincing medical statement.
    canonicalization_skip_asr_suspect: bool = True

    # -------------------------------------------------------------------------
    # Storage
    # -------------------------------------------------------------------------

    upload_dir: str = "data/uploads"
    result_dir: str = "data/results"

    # Permanent, content-addressed audio store.
    # NEVER pruned — see PLAN_V2.md §7.
    audio_dir: str = "data/audio"

    # Warn (never delete) once the audio store passes this size.
    audio_warn_gb: float = 50.0

    # -------------------------------------------------------------------------
    # Database (P1)
    # -------------------------------------------------------------------------

    # Empty = SQLite at data/scribe.db.
    # Any SQLAlchemy URL works.
    database_url: str = ""

    # Optional direct/admin URL for Alembic.
    # If empty, DATABASE_URL is used.
    migration_database_url: str = ""

    db_echo: bool = False

    # -------------------------------------------------------------------------
    # Final EXPERTA_MED KBS
    # -------------------------------------------------------------------------

    # Number of prior completed reports supplied for explicit temporal trends.
    # Ordinary clinical rules still use only the current report.
    kbs_history_reports: int = 5

    # -------------------------------------------------------------------------
    # Jobs / concurrency
    # -------------------------------------------------------------------------

    executor_max_workers: int = 1
    max_concurrent_jobs: int = 4
    max_upload_mb: int = 100

    # -------------------------------------------------------------------------
    # Resolved absolute paths
    # -------------------------------------------------------------------------

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
        """Normalize provider-style Postgres URLs to psycopg3 SQLAlchemy.

        Neon and many PaaS providers expose ``postgresql://...`` URLs.
        SQLAlchemy otherwise tries the legacy psycopg2 driver.

        The cloud image ships psycopg3, so make the driver explicit while
        leaving SQLite/other URLs unchanged.
        """

        if value.startswith("postgresql://"):
            return (
                "postgresql+psycopg://"
                + value[len("postgresql://"):]
            )

        if value.startswith("postgres://"):
            return (
                "postgresql+psycopg://"
                + value[len("postgres://"):]
            )

        return value

    @property
    def database_url_resolved(self) -> str:
        """Explicit DATABASE_URL, else SQLite beside the other data dirs."""

        if self.database_url:
            return self._sqlalchemy_url(self.database_url)

        return (
            f"sqlite:///"
            f"{(ROOT_DIR / 'data' / 'scribe.db').as_posix()}"
        )

    @property
    def migration_database_url_resolved(self) -> str:
        """Direct/admin migration URL when configured, else runtime URL."""

        if self.migration_database_url:
            return self._sqlalchemy_url(
                self.migration_database_url
            )

        return self.database_url_resolved

    @staticmethod
    def _abs(value: str) -> Path:
        p = Path(value)

        return (
            p
            if p.is_absolute()
            else ROOT_DIR / p
        )


@lru_cache
def get_settings() -> Settings:
    return Settings()