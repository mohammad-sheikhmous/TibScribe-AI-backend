"""Real post-deployment smoke test through the PUBLIC Laravel API only.

Usage:
  python scripts/clinical_e2e.py --email doctor@example.com --password 'Password1!' --audio visit.wav

This intentionally never contacts FastAPI directly. A PASS proves the public gateway
can complete login -> patient -> audio session -> AI job -> SOAP -> KBS suggestions.
"""
from __future__ import annotations

import argparse
import pathlib
import sys
import time
import uuid

import requests


def fail(message: str) -> "NoReturn":
    raise SystemExit(f"[FAIL] {message}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--email", required=True)
    parser.add_argument("--password", required=True)
    parser.add_argument("--audio", required=True)
    parser.add_argument("--timeout", type=int, default=600)
    args = parser.parse_args()

    audio = pathlib.Path(args.audio)
    if not audio.is_file():
        fail(f"audio file not found: {audio}")

    base = args.base_url.rstrip("/") + "/api/doctor"
    session = requests.Session()
    response = session.post(base + "/login", json={"email": args.email, "password": args.password}, timeout=30)
    if not response.ok:
        fail(f"login {response.status_code}: {response.text}")
    token = response.json().get("token")
    if not token:
        fail("login response did not contain token")
    session.headers["Authorization"] = f"Bearer {token}"
    session.headers["Accept"] = "application/json"

    ready = session.get(base + "/ai/ready", timeout=30)
    if not ready.ok:
        fail(f"AI not ready {ready.status_code}: {ready.text}")

    unique = uuid.uuid4().hex[:10]
    patient = session.post(
        base + "/patients",
        json={"mrn": f"E2E-{unique}", "first_name": "E2E", "last_name": "Patient"},
        timeout=30,
    )
    if not patient.ok:
        fail(f"patient create {patient.status_code}: {patient.text}")
    patient_id = patient.json()["patient"]["id"]

    with audio.open("rb") as handle:
        created = session.post(
            base + "/sessions",
            data={"patient_id": str(patient_id)},
            files={"audio": (audio.name, handle, "application/octet-stream")},
            headers={"Idempotency-Key": str(uuid.uuid4())},
            timeout=150,
        )
    if created.status_code != 202:
        fail(f"session create {created.status_code}: {created.text}")
    clinical_session_id = created.json()["session"]["id"]
    print(f"[OK] session {clinical_session_id} queued")

    deadline = time.monotonic() + args.timeout
    status = "queued"
    while time.monotonic() < deadline:
        poll = session.get(base + f"/sessions/{clinical_session_id}", timeout=30)
        if not poll.ok:
            fail(f"status poll {poll.status_code}: {poll.text}")
        payload = poll.json()["session"]
        status = payload.get("status")
        stage = payload.get("stage")
        print(f"[WAIT] status={status} stage={stage}")
        if status == "complete":
            break
        if status == "failed":
            fail(f"AI job failed: {payload.get('ai_error')}")
        time.sleep(2)
    else:
        fail(f"timeout waiting for AI completion; last status={status}")

    report = session.get(base + f"/sessions/{clinical_session_id}/report", timeout=60)
    suggestions = session.get(base + f"/sessions/{clinical_session_id}/suggestions", timeout=60)
    if not report.ok:
        fail(f"report {report.status_code}: {report.text}")
    if not suggestions.ok:
        fail(f"suggestions {suggestions.status_code}: {suggestions.text}")

    finalized = session.post(base + f"/sessions/{clinical_session_id}/finalize", timeout=60)
    if not finalized.ok:
        fail(f"finalize {finalized.status_code}: {finalized.text}")

    print("[PASS] Laravel -> FastAPI -> SOAP -> EXPERTA_MED -> Laravel end-to-end flow succeeded")
    print(f"       session={clinical_session_id}")


if __name__ == "__main__":
    main()
