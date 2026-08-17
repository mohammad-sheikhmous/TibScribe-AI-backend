"""Shared fixtures: every test runs against a throwaway database + audio archive.

No test may touch the real `data/` tree — least of all the audio archive, which the
application is never allowed to delete from.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.config import get_settings


@pytest.fixture
def temp_env(tmp_path, monkeypatch):
    """Point every storage setting at tmp_path and rebuild the engine around it."""
    from app.db import init_db, reset_engine

    monkeypatch.setenv("UPLOAD_DIR", str(tmp_path / "uploads"))
    monkeypatch.setenv("RESULT_DIR", str(tmp_path / "results"))
    monkeypatch.setenv("AUDIO_DIR", str(tmp_path / "audio"))
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{(tmp_path / 'test.db').as_posix()}")
    # Business tests use routers in isolation; gateway security has dedicated tests.
    monkeypatch.setenv("SERVICE_AUTH_REQUIRED", "false")

    get_settings.cache_clear()
    reset_engine()
    init_db()
    yield tmp_path
    reset_engine()
    get_settings.cache_clear()


@pytest.fixture
def audio_store(temp_env):
    from app.storage.audio_store import LocalAudioStore

    return LocalAudioStore(get_settings().audio_dir_path)


class StubPipeline:
    """Stands in for MedicalScribePipeline.process without loading any models."""

    def __init__(self, n_items: int = 1):
        self.n_items = n_items

    def process(self, audio_path, job_id, filename=None, patient_info=None):
        from app.core.report.builder import build_report
        from app.core.report.schema import AudioMeta, ClassifiedSegment, PipelineMeta

        segs = [
            ClassifiedSegment(
                order_index=i,
                text=f"صداع شديد {i}",
                start_sec=float(i),
                end_sec=float(i) + 1.0,
                source_segment_index=i,
                label="symptom" if i % 2 == 0 else "emergency",
                confidence=0.9 - (0.5 * (i % 2)),
            )
            for i in range(self.n_items)
        ]
        return build_report(
            job_id,
            segs,
            audio_meta=AudioMeta(filename=filename, duration_sec=float(self.n_items),
                                 detected_language="ar", whisper_model="stub"),
            pipeline_meta=PipelineMeta(classifier_max_len=128, arabert_model_name="stub"),
            low_confidence_threshold=0.5,
            patient_info=patient_info,
        )


def build_app(pipeline=None) -> FastAPI:
    """A minimal app wired exactly like main.py, minus the ML model loading."""
    from app.api.admin import router as admin_router
    from app.api.audio import router as audio_router
    from app.api.corrections import router as corrections_router
    from app.api.jobs import router as jobs_router
    from app.api.patients import router as patients_router
    from app.api.suggestions import router as suggestions_router
    from app.jobs.store import JobStore
    from app.storage.audio_store import LocalAudioStore
    from app.api.deps import require_service_token

    app = FastAPI()
    app.include_router(jobs_router)
    app.include_router(audio_router)
    app.include_router(corrections_router)
    app.include_router(patients_router)
    app.include_router(suggestions_router)
    app.include_router(admin_router)
    # Unit/contract tests exercise business behavior, not the shared-secret layer.
    app.dependency_overrides[require_service_token] = lambda: None
    app.state.job_store = JobStore()
    app.state.audio_store = LocalAudioStore(get_settings().audio_dir_path)
    app.state.pipeline = pipeline or StubPipeline()
    app.state.executor = ThreadPoolExecutor(max_workers=1)
    return app


@pytest.fixture
def client(temp_env):
    app = build_app()
    with TestClient(app) as test_client:
        yield test_client
    app.state.executor.shutdown(wait=False)


# A tiny but real WAV so playback/Range tests exercise actual bytes.
WAV_HEADER = (
    b"RIFF$\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00"
    b"\x44\xac\x00\x00\x88X\x01\x00\x02\x00\x10\x00data\x00\x00\x00\x00"
)


@pytest.fixture
def wav_bytes() -> bytes:
    return WAV_HEADER + bytes(range(256)) * 8  # 44 + 2048 bytes
