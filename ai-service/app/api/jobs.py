"""Job endpoints: upload audio, poll status, list jobs, fetch the structured report."""
from __future__ import annotations

import hashlib
import hmac
import logging
import os
import uuid
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, File, Form, Header, HTTPException, Request, UploadFile
from fastapi.responses import PlainTextResponse
from sqlalchemy.exc import IntegrityError

from ..config import get_settings
from ..core.report.markdown import report_to_markdown
from ..core.report.schema import Report
from ..db import repo, session_scope
from ..jobs.runner import run_job
from ..jobs.schema import JobRecord, JobStatus
from ..storage.paths import upload_path
from .schemas import JobListResponse, JobStatusResponse, JobSummary, UploadResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/jobs", tags=["jobs"])

ALLOWED_EXTS = {
    ".wav", ".mp3", ".m4a", ".ogg", ".flac", ".webm", ".mp4", ".mpeg", ".mpga",
    # voice-note formats (Telegram sends .oga/.opus; WhatsApp uses .opus/.aac/.amr)
    ".oga", ".opus", ".aac", ".amr", ".wma",
}


def _idempotent_upload_snapshot(
    existing,
    *,
    external_session_id: str,
    external_doctor_id: Optional[str],
    external_patient_id: Optional[str],
) -> tuple[UploadResponse, str]:
    """Validate a gateway replay and snapshot the existing immutable audio identity.

    The session UUID is an idempotency key, not an authorization substitute.  We bind
    it to Doctor + Patient + audio SHA-256 so a restore/client bug cannot silently
    attach a different consultation recording to an existing medical job.
    """
    if external_doctor_id and existing.external_doctor_id != external_doctor_id:
        raise HTTPException(409, "Clinical-session idempotency key belongs to another doctor identity.")
    if external_patient_id:
        patient = existing.patient
        if (
            patient is None
            or patient.external_source != "laravel"
            or patient.external_id != external_patient_id
        ):
            raise HTTPException(409, "Clinical-session idempotency key belongs to another patient identity.")
    audio = existing.audio
    if audio is None or not audio.sha256:
        raise HTTPException(409, "Existing clinical session has no recoverable audio identity.")
    return (
        UploadResponse(
            job_id=existing.id, status=existing.status,
            audio_sha256=audio.sha256, duplicate_audio=True,
            patient_id=existing.patient_id, external_session_id=external_session_id,
            idempotent_replay=True,
        ),
        audio.sha256,
    )


@router.post("", response_model=UploadResponse, status_code=202)
async def create_job(
    request: Request,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    mrn: Optional[str] = Form(None, description="patient record number; links visits into a series"),
    visit_at: Optional[datetime] = Form(None),
    external_doctor_id: Optional[str] = Header(None, alias="X-TibScribe-Doctor-ID"),
    external_patient_id: Optional[str] = Header(None, alias="X-TibScribe-Patient-ID"),
    external_session_id: Optional[str] = Header(None, alias="X-TibScribe-Session-ID"),
) -> UploadResponse:
    settings = get_settings()
    store = request.app.state.job_store
    audio_store = request.app.state.audio_store

    if settings.service_auth_required and settings.gateway_identity_required:
        missing = [name for name, value in (
            ("X-TibScribe-Doctor-ID", external_doctor_id),
            ("X-TibScribe-Patient-ID", external_patient_id),
            ("X-TibScribe-Session-ID", external_session_id),
        ) if not value]
        if missing:
            raise HTTPException(422, f"Missing trusted gateway identity headers: {', '.join(missing)}")

    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in ALLOWED_EXTS:
        raise HTTPException(
            400, f"Unsupported audio type '{ext}'. Allowed: {sorted(ALLOWED_EXTS)}"
        )

    # Snapshot a possible replay before reading the body. We still consume/hash the
    # incoming audio below and require the same SHA-256 before returning that job.
    replay: tuple[UploadResponse, str] | None = None
    if external_session_id:
        with session_scope() as session:
            existing = repo.get_job_by_external_session(session, external_session_id)
            if existing is not None:
                replay = _idempotent_upload_snapshot(
                    existing, external_session_id=external_session_id,
                    external_doctor_id=external_doctor_id,
                    external_patient_id=external_patient_id,
                )

    job_id = uuid.uuid4().hex
    staging = upload_path(job_id, ext)
    max_bytes = settings.max_upload_mb * 1024 * 1024
    size = 0
    digest = hashlib.sha256()
    try:
        with open(staging, "wb") as out:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                size += len(chunk)
                if size > max_bytes:
                    raise HTTPException(413, f"File too large (> {settings.max_upload_mb} MB).")
                digest.update(chunk)
                out.write(chunk)
    except HTTPException:
        staging.unlink(missing_ok=True)
        raise
    incoming_sha256 = digest.hexdigest()

    if replay is not None:
        response, expected_sha256 = replay
        staging.unlink(missing_ok=True)
        if not hmac.compare_digest(incoming_sha256, expected_sha256):
            raise HTTPException(409, "Clinical-session idempotency key was replayed with different audio.")
        return response

    # Only truly new jobs count against the queue limit. Idempotent replays above are
    # always recoverable even when the service is currently at capacity.
    if store.active_count() >= settings.max_concurrent_jobs:
        staging.unlink(missing_ok=True)
        raise HTTPException(429, "Server busy: too many jobs in progress. Try again shortly.")

    # Archive permanently (content-addressed, de-duplicated, never deleted).
    stored = audio_store.put(staging, ext)
    if audio_store.exists(stored.storage_key):
        staging.unlink(missing_ok=True)

    # Patient + Visit + Job are one DB transaction. This is critical: if two HTTP
    # retries race on external_session_id, the losing unique insert rolls back its
    # Visit too instead of leaving an orphan clinical visit in longitudinal history.
    try:
        with session_scope() as session:
            audio = repo.get_or_create_audio(
                session,
                sha256=stored.sha256,
                storage_key=stored.storage_key,
                original_filename=file.filename,
                mime=file.content_type,
                size_bytes=stored.size_bytes,
            )
            patient = (
                repo.get_or_create_external_patient(
                    session, source="laravel", external_id=external_patient_id
                )
                if external_patient_id
                else repo.get_or_create_patient(session, mrn=mrn)
            )
            visit = (
                repo.create_visit(session, patient_id=patient.id, visit_at=visit_at)
                if patient
                else None
            )
            repo.create_job(
                session, job_id=job_id, audio_id=audio.id,
                patient_id=patient.id if patient else None,
                visit_id=visit.id if visit else None,
                external_session_id=external_session_id,
                external_doctor_id=external_doctor_id, status=JobStatus.QUEUED.value,
            )
            patient_id = patient.id if patient else None
    except IntegrityError:
        # A concurrent retry may win either the external-session or content-addressed
        # unique constraint. After rollback no Visit from this request survives.
        if not external_session_id:
            raise
        with session_scope() as session:
            existing = repo.get_job_by_external_session(session, external_session_id)
            if existing is None:
                raise
            response, expected_sha256 = _idempotent_upload_snapshot(
                existing, external_session_id=external_session_id,
                external_doctor_id=external_doctor_id,
                external_patient_id=external_patient_id,
            )
            if not hmac.compare_digest(incoming_sha256, expected_sha256):
                raise HTTPException(409, "Clinical-session idempotency key was replayed with different audio.")
            return response
    background_tasks.add_task(run_job, request.app, job_id)
    return UploadResponse(
        job_id=job_id,
        status=JobStatus.QUEUED.value,
        audio_sha256=stored.sha256,
        duplicate_audio=not stored.was_new,
        patient_id=patient_id,
        external_session_id=external_session_id,
    )


@router.get("", response_model=JobListResponse)
async def list_jobs(
    request: Request,
    patient_id: Optional[str] = None,
    status: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
) -> JobListResponse:
    rows, total = request.app.state.job_store.list(
        patient_id=patient_id, status=status, limit=min(limit, 200), offset=offset
    )
    return JobListResponse(
        total=total, limit=limit, offset=offset,
        jobs=[JobSummary(**row) for row in rows],
    )




@router.post("/{job_id}/retry", response_model=JobStatusResponse, status_code=202)
async def retry_job(job_id: str, request: Request, background_tasks: BackgroundTasks) -> JobStatusResponse:
    """Retry a durable failed job without creating a second clinical visit.

    Gateway retries after a lost upload response are already handled by
    ``external_session_id`` in ``POST /jobs``. This endpoint is for a job whose id is
    known but whose processing failed after creation (transient decoder/model/storage
    errors, for example). Completed jobs are immutable and active jobs are returned
    idempotently rather than submitted twice.
    """
    store = request.app.state.job_store
    rec = store.get(job_id)
    if rec is None:
        raise HTTPException(404, "Unknown job_id.")
    if rec.status is JobStatus.COMPLETE:
        raise HTTPException(409, "Completed jobs cannot be retried.")
    if rec.status in {JobStatus.QUEUED, JobStatus.RUNNING}:
        return JobStatusResponse(
            job_id=rec.job_id, status=rec.status.value, stage=rec.stage, error=rec.error,
            external_session_id=rec.external_session_id,
        )

    settings = get_settings()
    if store.active_count() >= settings.max_concurrent_jobs:
        raise HTTPException(429, "Server busy: too many jobs in progress. Try again shortly.")
    store.update_status(job_id, JobStatus.QUEUED, stage=None, error=None)
    background_tasks.add_task(run_job, request.app, job_id)
    return JobStatusResponse(
        job_id=rec.job_id, status=JobStatus.QUEUED.value, stage=None, error=None,
        external_session_id=rec.external_session_id,
    )

@router.get("/{job_id}", response_model=JobStatusResponse)
async def get_status(job_id: str, request: Request) -> JobStatusResponse:
    rec = request.app.state.job_store.get(job_id)
    if rec is None:
        raise HTTPException(404, "Unknown job_id.")
    return JobStatusResponse(
        job_id=rec.job_id, status=rec.status.value, stage=rec.stage, error=rec.error,
        external_session_id=rec.external_session_id,
    )


def _completed_report(request: Request, job_id: str) -> Report:
    """Shared 404/409 handling: the report is read from the database, so it survives
    a restart (the v1 dict-backed store answered 404 after every restart)."""
    rec = request.app.state.job_store.get(job_id)
    if rec is None:
        raise HTTPException(404, "Unknown job_id.")
    if rec.status != JobStatus.COMPLETE or rec.result is None:
        raise HTTPException(409, f"Report not ready (status={rec.status.value}).")
    return rec.result


@router.get("/{job_id}/report", response_model=Report)
async def get_report(job_id: str, request: Request) -> Report:
    return _completed_report(request, job_id)


@router.get("/{job_id}/transcript")
async def get_transcript(job_id: str, include_raw: bool = False) -> dict:
    """Whisper's own output plus its quality signals (P3-04).

    The raw JSON is stored whole and served on request: it is the evidence behind every
    timestamp, and the fields this version ignores are what a later version will need.
    """
    from sqlalchemy import select

    from ..db.models import Transcript

    with session_scope() as session:
        row = session.scalar(select(Transcript).where(Transcript.job_id == job_id))
        if row is None:
            raise HTTPException(404, "No transcript stored for this job.")
        payload = {
            "job_id": job_id,
            "language": row.language,
            "avg_logprob": row.avg_logprob,
            "no_speech_ratio": row.no_speech_ratio,
            "created_at": row.created_at,
        }
        if include_raw and row.raw_json:
            import json as _json

            payload["raw"] = _json.loads(row.raw_json)
        return payload


@router.get("/{job_id}/report.md", response_class=PlainTextResponse)
async def get_report_markdown(job_id: str, request: Request) -> PlainTextResponse:
    return PlainTextResponse(
        report_to_markdown(_completed_report(request, job_id)),
        media_type="text/markdown; charset=utf-8",
    )
