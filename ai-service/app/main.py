"""FastAPI application entrypoint.

Startup order matters: the database and the audio store come up FIRST (they are
cheap and everything else depends on them), then the models are loaded once and
shared via `app.state`.

Run with:

    uvicorn app.main:app --reload
"""

from __future__ import annotations

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI

from .api.admin import router as admin_router
from .api.audio import router as audio_router
from .api.corrections import router as corrections_router
from .api.jobs import router as jobs_router
from .api.patients import router as patients_router
from .api.suggestions import router as suggestions_router
from .config import get_settings
from .core.pipeline import MedicalScribePipeline


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _load_voice_profile():
    """The active doctor voiceprint, if one has been built from the archive yet.

    Stored in `model_registry` like any other model, so it activates and rolls
    back the same way.

    Absent on day one — role assignment then falls back to linguistic evidence
    alone.
    """

    from .core.asr.voiceprint import VoiceProfile
    from .db import repo, session_scope

    try:
        with session_scope() as session:
            entry = repo.active_model(session, "voiceprint")

            if entry is None or not entry.metrics:
                return None

            profile = VoiceProfile.from_payload(entry.metrics)

            logger.info(
                "Loaded doctor voiceprint (%d prototypes, %d recordings)",
                len(profile.prototypes),
                profile.recordings_seen,
            )

            return profile

    except Exception:  # noqa: BLE001
        # A missing profile must never block startup.
        logger.warning(
            "Could not load the voiceprint; continuing without it",
            exc_info=True,
        )

        return None


def _build_canonicalization_stage(settings):
    """Build the pre-AraBERT medical canonicalization stage.

    Canonicalization is disabled by default.

    Disabled:
        NoOpCanonicalizationStage

    Enabled:
        OpenAICanonicalizationClient
            ↓
        LLMMedicalCanonicalizationStage

    The stage itself is provider-independent. Only this composition/bootstrap
    code knows which concrete LLM client is being used.
    """

    # Lazy imports keep the normal startup path independent of the LLM client
    # when canonicalization is disabled.
    from .core.nlp.canonicalization import (
        LLMMedicalCanonicalizationStage,
        NoOpCanonicalizationStage,
    )

    if not settings.canonicalization_enabled:
        logger.info(
            "Medical canonicalization disabled; "
            "AraBERT will receive segmented ASR text unchanged"
        )

        return NoOpCanonicalizationStage()

    provider = settings.canonicalization_provider.strip().lower()

    if provider != "openai_compatible":
        raise RuntimeError(
            "Unsupported CANONICALIZATION_PROVIDER="
            f"{settings.canonicalization_provider!r}. "
            "Currently supported: 'openai_compatible'."
        )

    if not settings.canonicalization_model.strip():
        raise RuntimeError(
            "CANONICALIZATION_MODEL must be configured when "
            "CANONICALIZATION_ENABLED=true."
        )

    if not settings.canonicalization_api_key.strip():
        raise RuntimeError(
            "CANONICALIZATION_API_KEY must be configured when "
            "CANONICALIZATION_ENABLED=true."
        )

    if settings.canonicalization_batch_size <= 0:
        raise RuntimeError(
            "CANONICALIZATION_BATCH_SIZE must be greater than zero."
        )

    if settings.canonicalization_timeout_sec <= 0:
        raise RuntimeError(
            "CANONICALIZATION_TIMEOUT_SEC must be greater than zero."
        )

    from .core.nlp.canonicalization_client import (
        OpenAICanonicalizationClient,
    )

    client = OpenAICanonicalizationClient(
        api_key=settings.canonicalization_api_key,
        model=settings.canonicalization_model,
        timeout_sec=settings.canonicalization_timeout_sec,
        base_url=settings.canonicalization_base_url or None,
    )

    stage = LLMMedicalCanonicalizationStage(
        client,
        batch_size=settings.canonicalization_batch_size,
        skip_asr_suspect=settings.canonicalization_skip_asr_suspect,
    )

    logger.info(
        "Medical canonicalization enabled "
        "(provider=%s, model=%s, batch_size=%d, "
        "skip_asr_suspect=%s)",
        provider,
        settings.canonicalization_model,
        settings.canonicalization_batch_size,
        settings.canonicalization_skip_asr_suspect,
    )

    return stage


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    app.state.settings = settings

    if (
        settings.service_auth_required
        and len(settings.service_token.strip()) < 32
    ):
        raise RuntimeError(
            "SERVICE_TOKEN must be configured with at least 32 characters when "
            "SERVICE_AUTH_REQUIRED=true."
        )

    # -------------------------------------------------------------------------
    # Durable layers first
    # -------------------------------------------------------------------------

    from .db import init_db
    from .jobs.store import JobStore
    from .storage.audio_store import LocalAudioStore

    init_db()

    # EXPERTA_MED is mandatory in the final build.
    # Fail startup rather than marking the service ready with a missing
    # rule engine/dependency.
    from EXPERTA_MED.engine import RULES_VERSION as KBS_RULES_VERSION
    from .core.kbs.service import analyze_report_with_history

    app.state.kbs_rules_version = KBS_RULES_VERSION
    app.state.kbs_analyzer = analyze_report_with_history

    logger.info(
        "EXPERTA_MED ready (rules %s)",
        KBS_RULES_VERSION,
    )

    app.state.audio_store = LocalAudioStore(
        settings.audio_dir_path
    )

    app.state.job_store = JobStore()

    logger.info(
        "Database + audio archive ready (%s)",
        settings.audio_dir_path,
    )

    # -------------------------------------------------------------------------
    # ML models
    # -------------------------------------------------------------------------

    # Lazy imports so `uvicorn app.main:app` fails loudly only if the ML deps are
    # genuinely missing, not at module import time.
    from .core.asr.whisper_service import WhisperTranscriber
    from .core.nlp.classifier import MedicalSentenceClassifier

    logger.info(
        "Loading models (this may take a while on first run)..."
    )

    transcriber = WhisperTranscriber(
        model_size=settings.whisper_model_size,
        language=settings.asr_language,
    )

    # -------------------------------------------------------------------------
    # Optional diarization
    # -------------------------------------------------------------------------

    # Without pyannote + HF_TOKEN this returns a NoOp diarizer and the pipeline
    # simply leaves speakers unattributed.
    from .core.asr.diarization import build_diarizer

    diarizer = build_diarizer(
        settings.diarization_enabled,
        settings.hf_token or None,
    )

    logger.info(
        "Diarization backend: %s",
        getattr(diarizer, "backend", "none"),
    )

    # -------------------------------------------------------------------------
    # AraBERT classifier
    # -------------------------------------------------------------------------

    classifier = MedicalSentenceClassifier(
        model_dir=str(settings.model_dir_path),
        model_name=settings.arobert_model_name,
        default_max_len=settings.default_max_len,
        low_confidence_threshold=settings.low_confidence_threshold,

        # Raises CheckpointMismatchError and aborts startup on an
        # untrustworthy checkpoint rather than serving mislabelled reports.
        strict=settings.strict_model_checks,
    )

    # -------------------------------------------------------------------------
    # NEW: Medical canonicalization stage
    # -------------------------------------------------------------------------
    #
    # This is created here, but its POSITION inside the pipeline will be
    # implemented in app/core/pipeline.py:
    #
    # Whisper
    #     ↓
    # Segmentation
    #     ↓
    # canonicalization_stage
    #     ↓
    # AraBERT
    #

    canonicalization_stage = _build_canonicalization_stage(
        settings
    )

    # -------------------------------------------------------------------------
    # Main medical pipeline
    # -------------------------------------------------------------------------

    app.state.pipeline = MedicalScribePipeline(
        transcriber,
        classifier,

        segment_max_chars=settings.segment_max_chars,
        word_pause_gap_sec=settings.word_pause_gap_sec,
        low_confidence_threshold=settings.low_confidence_threshold,

        diarizer=diarizer,

        # NEW
        canonicalization_stage=canonicalization_stage,

        # A single global voiceprint is unsafe in gateway/multi-doctor mode:
        # it could identify every speaker against one doctor's enrollment.
        # Until per-doctor profiles are keyed by external_doctor_id, use
        # linguistic attribution only.
        #
        # Standalone deployments may still use the active profile.
        voice_profile=(
            None
            if settings.gateway_identity_required
            else _load_voice_profile()
        ),

        uncertainty_enabled=settings.uncertainty_enabled,
        mc_passes=settings.mc_passes,
    )

    app.state.executor = ThreadPoolExecutor(
        max_workers=settings.executor_max_workers
    )

    app.state.ready = True

    # -------------------------------------------------------------------------
    # Durable recovery
    # -------------------------------------------------------------------------

    # Queued/running jobs survive an unclean restart in SQL.
    #
    # Resubmit them after every dependency has loaded so a gateway retry cannot
    # get stuck forever behind the external_session_id idempotency key.
    from .jobs.runner import run_job

    app.state.recovery_tasks = set()

    for orphan_job_id in app.state.job_store.active_job_ids():
        task = asyncio.create_task(
            run_job(app, orphan_job_id),
            name=f"recover-{orphan_job_id}",
        )

        app.state.recovery_tasks.add(task)
        task.add_done_callback(
            app.state.recovery_tasks.discard
        )

    if app.state.recovery_tasks:
        logger.warning(
            "Recovering %d durable AI job(s) after restart",
            len(app.state.recovery_tasks),
        )

    logger.info("Startup complete.")

    try:
        yield

    finally:
        app.state.ready = False

        for task in list(
            getattr(app.state, "recovery_tasks", set())
        ):
            task.cancel()

        app.state.executor.shutdown(
            wait=False,
            cancel_futures=True,
        )


app = FastAPI(
    title="TibScribe AI Service",
    version="3.0",
    lifespan=lifespan,
)


from .api.deps import require_service_token


_service_auth = [
    Depends(require_service_token)
]


app.include_router(
    jobs_router,
    dependencies=_service_auth,
)

app.include_router(
    audio_router,
    dependencies=_service_auth,
)

app.include_router(
    corrections_router,
    dependencies=_service_auth,
)

app.include_router(
    patients_router,
    dependencies=_service_auth,
)

app.include_router(
    suggestions_router,
    dependencies=_service_auth,
)

app.include_router(
    admin_router,
    dependencies=_service_auth,
)


@app.get(
    "/health",
    tags=["meta"],
)
async def health() -> dict[str, str]:
    """Liveness: the process is up. Says nothing about the models."""

    return {
        "status": "ok"
    }


@app.get(
    "/ready",
    tags=["meta"],
    dependencies=_service_auth,
)
async def ready() -> dict[str, object]:
    """Readiness: models loaded, database reachable, archive mounted."""

    from fastapi import HTTPException

    from .db import session_scope
    from .db.models import Job

    problems: list[str] = []

    if not getattr(
        app.state,
        "ready",
        False,
    ):
        problems.append(
            "models not loaded"
        )

    try:
        with session_scope() as session:
            session.query(Job).limit(1).all()

    except Exception as exc:  # pragma: no cover
        problems.append(
            f"database unreachable: {type(exc).__name__}"
        )

    if not getattr(
        app.state,
        "kbs_rules_version",
        None,
    ):
        problems.append(
            "EXPERTA_MED KBS not loaded"
        )

    store = getattr(
        app.state,
        "audio_store",
        None,
    )

    if (
        store is None
        or not store.root.exists()
    ):
        problems.append(
            "audio archive not mounted"
        )

    from .core.asr.whisper_service import ffmpeg_path

    if ffmpeg_path() is None:
        problems.append(
            "ffmpeg binary not on PATH — audio cannot be decoded"
        )

    if problems:
        raise HTTPException(
            503,
            {
                "status": "not-ready",
                "problems": problems,
            },
        )

    return {
        "status": "ready"
    }