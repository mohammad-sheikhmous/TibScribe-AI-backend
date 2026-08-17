"""Patient identity, timeline and context through the API (P2-01…P2-07).

The point of these tests is cohesion: three uploads with the same MRN must become one
patient history *automatically*, and the obstetric context the rules will later read
must come from the record — not from a substring in an old transcript.
"""
from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient

from tests.conftest import build_app


class ContextPipeline:
    """Stub pipeline whose transcript text is supplied per test."""

    def __init__(self, sentences: list[str]):
        self.sentences = sentences

    def process(self, audio_path, job_id, filename=None, patient_info=None):
        from app.core.report.builder import build_report
        from app.core.report.schema import AudioMeta, ClassifiedSegment

        segs = [
            ClassifiedSegment(
                order_index=i, text=text, start_sec=float(i * 5), end_sec=float(i * 5 + 4),
                source_segment_index=i, label="info", confidence=0.9,
            )
            for i, text in enumerate(self.sentences)
        ]
        return build_report(
            job_id, segs,
            audio_meta=AudioMeta(filename=filename, duration_sec=30.0),
            patient_info=patient_info,
        )


def _upload(client, wav_bytes, name="v.wav", **data):
    return client.post(
        "/jobs", files={"file": (name, wav_bytes, "audio/wav")}, data=data or None
    ).json()


# --- identity ------------------------------------------------------------------------

def test_create_and_find_a_patient(client):
    created = client.post("/patients", json={"mrn": "MRN-100", "display_name": "سارة"})
    assert created.status_code == 201
    patient_id = created.json()["id"]

    assert client.get("/patients", params={"q": "MRN-100"}).json()["total"] == 1
    assert client.get(f"/patients/{patient_id}").json()["patient"]["mrn"] == "MRN-100"
    assert client.get("/patients/nope").status_code == 404


def test_same_mrn_is_never_duplicated(client):
    first = client.post("/patients", json={"mrn": "MRN-1"}).json()["id"]
    second = client.post("/patients", json={"mrn": "MRN-1"}).json()["id"]
    assert first == second
    assert client.get("/patients").json()["total"] == 1


def test_three_uploads_with_one_mrn_become_one_history(client, wav_bytes):
    for i in range(3):
        _upload(client, wav_bytes + bytes([i]), name=f"v{i}.wav", mrn="MRN-77")

    patient_id = client.get("/patients", params={"q": "MRN-77"}).json()["patients"][0]["id"]
    timeline = client.get(f"/patients/{patient_id}/timeline").json()

    assert timeline["visits"] == 3
    assert len(timeline["timeline"]) == 3
    assert timeline["timeline"][0]["total_segments"] >= 1
    # newest first
    stamps = [entry["created_at"] for entry in timeline["timeline"]]
    assert stamps == sorted(stamps, reverse=True)


def test_report_carries_patient_identity(temp_env, wav_bytes):
    app = build_app(pipeline=ContextPipeline(["المريضة حامل بالأسبوع 20"]))
    with TestClient(app) as client:
        job = _upload(client, wav_bytes, mrn="MRN-55")
        report = client.get(f"/jobs/{job['job_id']}/report").json()
        info = report["patient_info"]
        assert info["patient_id"] == job["patient_id"]
        assert info["mrn"] == "MRN-55"
        assert info["visit_index"] == 1
    app.state.executor.shutdown(wait=False)


def test_upload_without_mrn_stays_unlinked(client, wav_bytes):
    job = _upload(client, wav_bytes)
    assert job["patient_id"] is None
    assert client.get(f"/jobs/{job['job_id']}/report").json()["patient_info"] == {}


# --- obstetric context flowing from transcript to record --------------------------------

def test_pregnancy_from_the_transcript_reaches_the_patient_record(temp_env, wav_bytes):
    app = build_app(pipeline=ContextPipeline([
        "المريضة حامل بالأسبوع 30",
        "تشتكي من تورم بالقدمين",
    ]))
    with TestClient(app) as client:
        job = _upload(client, wav_bytes, mrn="MRN-preg")
        state = client.get(f"/patients/{job['patient_id']}").json()["obstetric_state"]
        assert state["status"] == "pregnant" and state["is_pregnant"] is True
        assert state["edd"] is not None
    app.state.executor.shutdown(wait=False)


def test_denied_pregnancy_does_not_mark_the_patient_pregnant(temp_env, wav_bytes):
    """End-to-end version of ك-٤: 'ليست حامل' must not flip the record."""
    app = build_app(pipeline=ContextPipeline(["المريضة ليست حامل حالياً وتشتكي من صداع"]))
    with TestClient(app) as client:
        job = _upload(client, wav_bytes, mrn="MRN-not-preg")
        state = client.get(f"/patients/{job['patient_id']}").json()["obstetric_state"]
        assert state["is_pregnant"] is False
    app.state.executor.shutdown(wait=False)


def test_general_nutrition_visit_leaves_context_unknown(temp_env, wav_bytes):
    app = build_app(pipeline=ContextPipeline([
        "الشوكولاتة الداكنة بكميات قليلة مفيدة للقلب",
        "المشروبات الطبيعية بدون سكر أفضل من الصناعية",
    ]))
    with TestClient(app) as client:
        job = _upload(client, wav_bytes, mrn="MRN-food")
        body = client.get(f"/patients/{job['patient_id']}").json()
        assert body["obstetric_state"]["status"] == "unknown"
        assert body["obstetric_state"]["is_pregnant"] is False
    app.state.executor.shutdown(wait=False)


def test_state_history_records_each_observing_visit(temp_env, wav_bytes):
    app = build_app(pipeline=ContextPipeline(["المريضة حامل بالأسبوع 12"]))
    with TestClient(app) as client:
        job = _upload(client, wav_bytes, mrn="MRN-hist")
        history = client.get(f"/patients/{job['patient_id']}/timeline").json()["state_history"]
        assert len(history) == 1
        assert history[0]["status"] == "pregnant" and history[0]["ga_weeks"] == 12
        assert history[0]["job_id"] == job["job_id"]
    app.state.executor.shutdown(wait=False)


def test_clinician_override_wins_because_it_is_newer(temp_env, wav_bytes):
    app = build_app(pipeline=ContextPipeline(["المريضة حامل بالأسبوع 30"]))
    with TestClient(app) as client:
        job = _upload(client, wav_bytes, mrn="MRN-fix")
        patient_id = job["patient_id"]
        assert client.get(f"/patients/{patient_id}").json()["obstetric_state"]["is_pregnant"]

        # No effective_at: the override applies from now. It wins over the same-instant
        # inferred row because the state query breaks ties on insertion order.
        resp = client.post(f"/patients/{patient_id}/state", json={"pregnant": False})
        assert resp.status_code == 201

        state = client.get(f"/patients/{patient_id}").json()["obstetric_state"]
        assert state["is_pregnant"] is False and state["source"] == "explicit"
    app.state.executor.shutdown(wait=False)


def test_empty_state_override_is_rejected(client):
    patient_id = client.post("/patients", json={"mrn": "MRN-x"}).json()["id"]
    assert client.post(f"/patients/{patient_id}/state", json={}).status_code == 400


def test_context_is_read_as_of_a_past_moment(temp_env, wav_bytes):
    """`at=` answers 'was she pregnant then?', which is what a series needs."""
    app = build_app(pipeline=ContextPipeline(["المريضة حامل بالأسبوع 8"]))
    with TestClient(app) as client:
        job = _upload(client, wav_bytes, mrn="MRN-time")
        patient_id = job["patient_id"]
        long_after = (datetime.now(timezone.utc) + timedelta(days=400)).isoformat()
        state = client.get(f"/patients/{patient_id}",
                           params={"at": long_after}).json()["obstetric_state"]
        assert state["is_pregnant"] is False and state["expired"] is True
    app.state.executor.shutdown(wait=False)



def test_manual_gestational_age_alone_asserts_pregnancy(client):
    patient_id = client.post("/patients", json={"mrn": "MRN-ga-explicit"}).json()["id"]
    response = client.post(f"/patients/{patient_id}/state", json={"ga_weeks": 32})
    assert response.status_code == 201
    state = response.json()
    assert state["is_pregnant"] is True
    assert state["ga_weeks_at_observation"] == 32


def test_postpartum_cannot_be_combined_with_current_gestational_age(client):
    patient_id = client.post("/patients", json={"mrn": "MRN-state-conflict"}).json()["id"]
    response = client.post(
        f"/patients/{patient_id}/state", json={"postpartum": True, "ga_weeks": 32}
    )
    assert response.status_code == 422


def test_state_override_can_refresh_the_same_completed_jobs_kbs(temp_env, wav_bytes):
    from types import SimpleNamespace

    calls = []

    def analyzer(report, history, *, patient_context):
        calls.append(dict(patient_context))
        return SimpleNamespace(result={
            "engine": "test-kbs",
            "rules_version": "test",
            "suggestions": [],
            "patient_context": patient_context,
            "trends": [],
            "audit_trail": [],
        })

    app = build_app(pipeline=ContextPipeline(["المريضة حامل بالأسبوع 30"]))
    app.state.kbs_analyzer = analyzer
    with TestClient(app) as client:
        job = _upload(client, wav_bytes, mrn="MRN-state-refresh")
        assert len(calls) == 1

        response = client.post(
            f"/patients/{job['patient_id']}/state",
            json={
                "postpartum": True,
                "effective_at": client.get(f"/jobs/{job['job_id']}/report").json()["created_at"],
                "refresh_job_id": job["job_id"],
            },
        )
        assert response.status_code == 201
        assert len(calls) == 2
        assert calls[-1]["postpartum"] is True
        assert calls[-1]["pregnant"] is False

        report = client.get(f"/jobs/{job['job_id']}/report").json()
        assert report["patient_info"]["effective_obstetric_status"] == "postpartum"
    app.state.executor.shutdown(wait=False)
