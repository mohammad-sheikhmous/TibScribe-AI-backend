"""Durability across a process restart (IMPLEMENTATION.md P1-08).

This is the regression test for gap ث-٦: in v1 the job store was a dict, so a restart
answered 404 for every past job while its report sat unread on disk. Here the app is
torn down and rebuilt from scratch — a genuinely new JobStore, a new audio store, a new
engine — and everything must still be there.
"""
from fastapi.testclient import TestClient

from tests.conftest import build_app


def _upload(client, wav_bytes, **data):
    return client.post(
        "/jobs", files={"file": ("note.wav", wav_bytes, "audio/wav")}, data=data or None
    )


def test_report_survives_a_restart(temp_env, wav_bytes):
    app = build_app()
    with TestClient(app) as client:
        job_id = _upload(client, wav_bytes, mrn="MRN-9").json()["job_id"]
        before = client.get(f"/jobs/{job_id}/report").json()
    app.state.executor.shutdown(wait=False)

    # --- simulate a full process restart: nothing in memory carries over ---
    restarted = build_app()
    with TestClient(restarted) as client:
        assert client.get(f"/jobs/{job_id}").json()["status"] == "complete"
        after = client.get(f"/jobs/{job_id}/report").json()
        assert after == before                       # byte-identical report
        assert client.get(f"/jobs/{job_id}/report.md").status_code == 200
        assert client.get("/jobs").json()["total"] == 1
    restarted.state.executor.shutdown(wait=False)


def test_audio_is_still_playable_after_a_restart(temp_env, wav_bytes):
    app = build_app()
    with TestClient(app) as client:
        job_id = _upload(client, wav_bytes).json()["job_id"]
    app.state.executor.shutdown(wait=False)

    restarted = build_app()
    with TestClient(restarted) as client:
        resp = client.get(f"/jobs/{job_id}/audio")
        assert resp.status_code == 200
        assert resp.content == wav_bytes
    restarted.state.executor.shutdown(wait=False)


def test_patient_history_accumulates_across_restarts(temp_env, wav_bytes):
    app = build_app()
    with TestClient(app) as client:
        first = _upload(client, wav_bytes, mrn="MRN-42").json()
    app.state.executor.shutdown(wait=False)

    restarted = build_app()
    with TestClient(restarted) as client:
        second = client.post(
            "/jobs",
            files={"file": ("visit2.wav", wav_bytes + b"more", "audio/wav")},
            data={"mrn": "MRN-42"},
        ).json()
        assert second["patient_id"] == first["patient_id"]  # same patient, new process
        assert client.get("/jobs", params={"patient_id": first["patient_id"]}).json()[
            "total"
        ] == 2
    restarted.state.executor.shutdown(wait=False)


def test_json_export_is_a_convenience_not_the_source_of_truth(temp_env, wav_bytes):
    """Deleting the exported JSON must not affect the API — the DB is authoritative."""
    from app.config import get_settings

    app = build_app()
    with TestClient(app) as client:
        job_id = _upload(client, wav_bytes).json()["job_id"]
        exported = get_settings().result_dir_path / f"{job_id}.json"
        assert exported.exists()
        exported.unlink()
        assert client.get(f"/jobs/{job_id}/report").status_code == 200
    app.state.executor.shutdown(wait=False)
