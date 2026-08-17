from __future__ import annotations

from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

from app.api.deps import require_service_token
from app.config import get_settings
from app.api.jobs import router as jobs_router
from tests.conftest import WAV_HEADER, build_app


def test_service_token_dependency(monkeypatch):
    monkeypatch.setenv("SERVICE_AUTH_REQUIRED", "true")
    monkeypatch.setenv("SERVICE_TOKEN", "x" * 40)
    get_settings.cache_clear()
    app = FastAPI()
    @app.get("/private", dependencies=[Depends(require_service_token)])
    def private():
        return {"ok": True}
    c = TestClient(app)
    assert c.get("/private").status_code == 401
    assert c.get("/private", headers={"Authorization": "Bearer wrong"}).status_code == 401
    assert c.get("/private", headers={"Authorization": "Bearer " + "x" * 40}).json() == {"ok": True}
    get_settings.cache_clear()


def test_external_identity_and_idempotent_session(temp_env):
    app = build_app()
    with TestClient(app) as c:
        headers = {
            "X-TibScribe-Doctor-ID": "doctor-94b19f5c",
            "X-TibScribe-Patient-ID": "patient-e995b7a1",
            "X-TibScribe-Session-ID": "session-72d3ad8b",
        }
        files = {"file": ("visit.wav", WAV_HEADER, "audio/wav")}
        first = c.post("/jobs", files=files, headers=headers)
        assert first.status_code == 202
        payload = first.json()
        assert payload["external_session_id"] == "session-72d3ad8b"
        assert payload["idempotent_replay"] is False
        # The retry must resolve the same job instead of creating another one.
        second = c.post("/jobs", files=files, headers=headers)
        assert second.status_code == 202
        p2 = second.json()
        assert p2["job_id"] == payload["job_id"]
        assert p2["idempotent_replay"] is True

        mapped = c.get("/patients/by-external/laravel/patient-e995b7a1")
        assert mapped.status_code == 200
        assert mapped.json()["patient"]["id"] == payload["patient_id"]


def test_failed_job_can_be_retried_without_creating_a_second_job(temp_env):
    from app.jobs.schema import JobStatus

    app = build_app()
    with TestClient(app) as c:
        headers = {
            "X-TibScribe-Doctor-ID": "doctor-retry",
            "X-TibScribe-Patient-ID": "patient-retry",
            "X-TibScribe-Session-ID": "session-retry",
        }
        created = c.post(
            "/jobs", files={"file": ("visit.wav", WAV_HEADER, "audio/wav")}, headers=headers
        )
        assert created.status_code == 202
        job_id = created.json()["job_id"]

        # Simulate a durable processing failure after the upload was accepted.
        app.state.job_store.update_status(job_id, JobStatus.FAILED, error="transient")
        retried = c.post(f"/jobs/{job_id}/retry")
        assert retried.status_code == 202
        assert retried.json()["job_id"] == job_id
        assert retried.json()["status"] == "queued"

        # TestClient runs BackgroundTasks before returning control; the same durable job
        # should now be complete, not duplicated under a new id.
        status = c.get(f"/jobs/{job_id}")
        assert status.status_code == 200
        assert status.json()["status"] == "complete"

        assert c.post(f"/jobs/{job_id}/retry").status_code == 409


def test_external_session_id_cannot_be_reused_for_another_patient_or_doctor(temp_env):
    app = build_app()
    with TestClient(app) as c:
        headers = {
            "X-TibScribe-Doctor-ID": "doctor-A",
            "X-TibScribe-Patient-ID": "patient-A",
            "X-TibScribe-Session-ID": "session-stable",
        }
        files = {"file": ("visit.wav", WAV_HEADER, "audio/wav")}
        first = c.post("/jobs", files=files, headers=headers)
        assert first.status_code == 202

        wrong_patient = dict(headers)
        wrong_patient["X-TibScribe-Patient-ID"] = "patient-B"
        assert c.post("/jobs", files=files, headers=wrong_patient).status_code == 409

        wrong_doctor = dict(headers)
        wrong_doctor["X-TibScribe-Doctor-ID"] = "doctor-B"
        assert c.post("/jobs", files=files, headers=wrong_doctor).status_code == 409


def test_external_session_replay_requires_same_audio_and_keeps_one_visit(temp_env):
    from sqlalchemy import func, select

    from app.db import session_scope
    from app.db.models import Visit
    from app.db import repo

    app = build_app()
    headers = {
        "X-TibScribe-Doctor-ID": "doctor-audio-bind",
        "X-TibScribe-Patient-ID": "patient-audio-bind",
        "X-TibScribe-Session-ID": "session-audio-bind",
    }
    with TestClient(app) as c:
        first = c.post(
            "/jobs",
            files={"file": ("visit.wav", WAV_HEADER, "audio/wav")},
            headers=headers,
        )
        assert first.status_code == 202

        replay = c.post(
            "/jobs",
            files={"file": ("visit.wav", WAV_HEADER + b"different-recording", "audio/wav")},
            headers=headers,
        )
        assert replay.status_code == 409

    with session_scope() as session:
        patient = repo.get_patient_by_external(
            session, source="laravel", external_id="patient-audio-bind"
        )
        assert patient is not None
        visit_count = session.scalar(
            select(func.count(Visit.id)).where(Visit.patient_id == patient.id)
        )
        assert visit_count == 1


def test_concurrent_replay_losing_insert_rolls_back_its_visit(temp_env, monkeypatch):
    """Simulate the SELECT/INSERT race around external_session_id.

    The second request intentionally misses the first idempotency lookup, then hits the
    DB unique constraint. Its Visit must roll back with the losing Job insert.
    """
    from sqlalchemy import func, select

    from app.db import repo, session_scope
    from app.db.models import Visit

    app = build_app()
    headers = {
        "X-TibScribe-Doctor-ID": "doctor-race",
        "X-TibScribe-Patient-ID": "patient-race",
        "X-TibScribe-Session-ID": "session-race",
    }
    with TestClient(app) as c:
        first = c.post(
            "/jobs",
            files={"file": ("visit.wav", WAV_HEADER, "audio/wav")},
            headers=headers,
        )
        assert first.status_code == 202

        original_lookup = repo.get_job_by_external_session
        calls = {"n": 0}

        def miss_once(session, external_session_id):
            calls["n"] += 1
            if calls["n"] == 1:
                return None
            return original_lookup(session, external_session_id)

        monkeypatch.setattr(repo, "get_job_by_external_session", miss_once)
        replay = c.post(
            "/jobs",
            files={"file": ("visit.wav", WAV_HEADER, "audio/wav")},
            headers=headers,
        )
        assert replay.status_code == 202
        assert replay.json()["idempotent_replay"] is True
        assert replay.json()["job_id"] == first.json()["job_id"]

    with session_scope() as session:
        patient = repo.get_patient_by_external(session, source="laravel", external_id="patient-race")
        assert patient is not None
        count = session.scalar(select(func.count(Visit.id)).where(Visit.patient_id == patient.id))
        assert count == 1
