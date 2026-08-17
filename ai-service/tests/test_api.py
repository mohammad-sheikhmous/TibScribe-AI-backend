"""API wiring with a STUB pipeline (no torch/whisper needed).

Covers upload -> background processing -> status -> report retrieval, the error paths,
and the v2 additions: content-addressed archiving, patient linking and job listing.
"""
from tests.conftest import build_app

from fastapi.testclient import TestClient


def _upload(client, wav_bytes, name="note.wav", **data):
    return client.post(
        "/jobs", files={"file": (name, wav_bytes, "audio/wav")}, data=data or None
    )


def test_full_flow(client, wav_bytes):
    resp = _upload(client, wav_bytes)
    assert resp.status_code == 202
    body = resp.json()
    job_id = body["job_id"]
    assert len(body["audio_sha256"]) == 64
    assert body["duplicate_audio"] is False

    # TestClient runs background tasks before returning, so it should be COMPLETE.
    status = client.get(f"/jobs/{job_id}")
    assert status.status_code == 200
    assert status.json()["status"] == "complete"

    report = client.get(f"/jobs/{job_id}/report")
    assert report.status_code == 200
    payload = report.json()
    assert payload["job_id"] == job_id
    assert payload["soap"]["subjective"]["items"][0]["label"] == "symptom"
    assert payload["summary"]["total_segments"] == 1

    md = client.get(f"/jobs/{job_id}/report.md")
    assert md.status_code == 200
    assert md.headers["content-type"].startswith("text/markdown")
    assert "SOAP" in md.text and "ملخّص تنفيذي" in md.text


def test_unknown_job_404(client):
    assert client.get("/jobs/does-not-exist").status_code == 404
    assert client.get("/jobs/does-not-exist/report").status_code == 404


def test_unsupported_extension_400(client):
    resp = client.post("/jobs", files={"file": ("note.txt", b"hello", "text/plain")})
    assert resp.status_code == 400


def test_oversized_upload_is_rejected_and_leaves_nothing_archived(client, monkeypatch):
    from app.config import get_settings

    get_settings.cache_clear()
    monkeypatch.setenv("MAX_UPLOAD_MB", "0")
    get_settings.cache_clear()
    try:
        resp = client.post("/jobs", files={"file": ("big.wav", b"x" * 4096, "audio/wav")})
        assert resp.status_code == 413
        assert client.get("/admin/storage").json()["objects"] == 0
    finally:
        monkeypatch.delenv("MAX_UPLOAD_MB", raising=False)
        get_settings.cache_clear()


# --- v2: content addressing, patients, listing --------------------------------------

def test_same_recording_uploaded_twice_is_archived_once(client, wav_bytes):
    first = _upload(client, wav_bytes).json()
    second = _upload(client, wav_bytes, name="copy.wav").json()

    assert first["job_id"] != second["job_id"]          # two jobs
    assert first["audio_sha256"] == second["audio_sha256"]
    assert second["duplicate_audio"] is True
    assert client.get("/admin/storage").json()["objects"] == 1  # one object on disk


def test_mrn_links_jobs_to_one_patient(client, wav_bytes):
    a = _upload(client, wav_bytes, mrn="MRN-77").json()
    b = _upload(client, wav_bytes + b"different", name="v2.wav", mrn="MRN-77").json()

    assert a["patient_id"] and a["patient_id"] == b["patient_id"]
    listing = client.get("/jobs", params={"patient_id": a["patient_id"]}).json()
    assert listing["total"] == 2


def test_job_listing_filters_by_status(client, wav_bytes):
    _upload(client, wav_bytes)
    _upload(client, wav_bytes + b"2", name="b.wav")

    all_jobs = client.get("/jobs").json()
    assert all_jobs["total"] == 2
    assert {"job_id", "status", "filename", "created_at"} <= set(all_jobs["jobs"][0])

    complete = client.get("/jobs", params={"status": "complete"}).json()
    assert complete["total"] == 2
    assert client.get("/jobs", params={"status": "failed"}).json()["total"] == 0


def test_failed_job_reports_409_and_a_sanitized_error(temp_env, wav_bytes):
    class ExplodingPipeline:
        def process(self, *_args, **_kwargs):
            raise RuntimeError("secret internal detail")

    app = build_app(pipeline=ExplodingPipeline())
    with TestClient(app) as client:
        job_id = _upload(client, wav_bytes).json()["job_id"]
        status = client.get(f"/jobs/{job_id}").json()
        assert status["status"] == "failed"
        assert "secret internal detail" not in (status["error"] or "")
        assert client.get(f"/jobs/{job_id}/report").status_code == 409
    app.state.executor.shutdown(wait=False)
