"""Safety regressions for clinician corrections and gateway patient mapping."""
from __future__ import annotations

from types import SimpleNamespace

from fastapi.testclient import TestClient

from app.db import repo, session_scope
from app.db.models import Suggestion
from tests.conftest import build_app


def _upload(client: TestClient, wav_bytes: bytes) -> str:
    response = client.post(
        "/jobs",
        files={"file": ("visit.wav", wav_bytes, "audio/wav")},
    )
    assert response.status_code == 202
    job_id = response.json()["job_id"]
    status = client.get(f"/jobs/{job_id}")
    assert status.status_code == 200
    assert status.json()["status"] == "complete"
    return job_id


def test_external_patient_mapping_put_is_idempotent(temp_env):
    app = build_app()
    with TestClient(app) as client:
        first = client.put("/patients/by-external/laravel/42")
        second = client.put("/patients/by-external/laravel/42")
        assert first.status_code == 200
        assert second.status_code == 200
        assert first.json()["patient"]["id"] == second.json()["patient"]["id"]
    app.state.executor.shutdown(wait=False)


def test_text_correction_reextracts_entities_and_retires_stale_suggestions(
    temp_env, wav_bytes
):
    app = build_app()
    # Production installs a real KBS analyzer. This deterministic stub makes the safety
    # property testable without importing Experta in the unit-test environment.
    app.state.kbs_analyzer = lambda report, history, patient_context: SimpleNamespace(
        result={"suggestions": []}
    )

    with TestClient(app) as client:
        job_id = _upload(client, wav_bytes)
        report = client.get(f"/jobs/{job_id}/report").json()
        item_id = report["soap"]["subjective"]["items"][0]["item_id"]

        with session_scope() as session:
            repo.save_suggestions(
                session,
                job_id=job_id,
                patient_id=None,
                suggestions=[{
                    "rule_id": "TEST-R1",
                    "rule_version": "1",
                    "severity": "high",
                    "condition": "headache",
                    "title_ar": "قديم",
                    "detail_ar": "يجب ألا يبقى فعالاً بعد التصحيح",
                }],
            )

        corrected = client.patch(
            f"/jobs/{job_id}/items/{item_id}",
            json={"text": "لا يوجد صداع", "actor": "doctor:test"},
        )
        assert corrected.status_code == 200

        # The active clinical view must contain no stale advice.
        active = client.get(f"/jobs/{job_id}/suggestions")
        assert active.status_code == 200
        assert active.json()["total"] == 0

        with session_scope() as session:
            item = repo.get_item(session, job_id, item_id)
            headache = [x for x in (item.entity_links or []) if x.get("code") == "headache"]
            assert headache and headache[0]["assertion"] == "absent"

            # Audit history is retained rather than deleted.
            historical = session.query(Suggestion).filter_by(job_id=job_id).all()
            assert len(historical) == 1
            assert historical[0].is_active is False

    app.state.executor.shutdown(wait=False)
