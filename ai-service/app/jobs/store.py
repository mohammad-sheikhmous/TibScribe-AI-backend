"""JobStore — the same tiny interface as v1, now backed by the database.

v1 kept jobs in a dict, so every job and report vanished on restart while its JSON sat
unread on disk (gap ث-٦). The interface was deliberately small so this swap would
touch one file: `create` / `get` / `update_status` / `set_result` / `active_count` are
unchanged, and callers (the API router and the job runner) did not change with it.

A short-lived session per operation is the right unit here: operations are tiny and
independent, and the pipeline runs in a worker thread while the event loop reads
status — no session is ever shared across threads.
"""
from __future__ import annotations

from typing import Optional

from ..core.report.schema import Report
from ..db import repo, session_scope
from .schema import JobRecord, JobStatus


class JobStore:
    def create(self, record: JobRecord) -> JobRecord:
        """Persist a queued job. The audio row is created by the upload endpoint."""
        with session_scope() as session:
            repo.create_job(
                session,
                job_id=record.job_id,
                audio_id=record.audio_id,
                patient_id=record.patient_id,
                visit_id=record.visit_id,
                external_session_id=record.external_session_id,
                external_doctor_id=record.external_doctor_id,
                status=record.status.value,
            )
        return record

    def get(self, job_id: str) -> Optional[JobRecord]:
        with session_scope() as session:
            job = repo.get_job(session, job_id)
            if job is None:
                return None
            status = JobStatus(job.status)
            # The report is only materialised for a completed job — polling status
            # stays a single cheap row read.
            result = repo.load_report(session, job_id) if status is JobStatus.COMPLETE else None
            audio = job.audio
            return JobRecord(
                job_id=job.id,
                audio_id=job.audio_id,
                audio_path=str(audio.storage_key) if audio else None,
                filename=audio.original_filename if audio else None,
                patient_id=job.patient_id,
                visit_id=job.visit_id,
                external_session_id=job.external_session_id,
                external_doctor_id=job.external_doctor_id,
                status=status,
                stage=job.stage,
                error=job.error,
                result=result,
                created_at=job.created_at,
                updated_at=job.updated_at,
            )

    def update_status(
        self,
        job_id: str,
        status: JobStatus,
        *,
        stage: Optional[str] = None,
        error: Optional[str] = None,
    ) -> None:
        with session_scope() as session:
            repo.update_job(
                session, job_id, status=status.value, stage=stage, error=error
            )

    def set_result(self, job_id: str, result: Report, duration_ms: Optional[int] = None) -> None:
        """Store the report and flip the job to complete in ONE transaction.

        Atomicity matters: a job marked complete whose report failed to save would
        answer 404 forever.
        """
        with session_scope() as session:
            repo.save_report(session, result)
            repo.update_job(
                session, job_id, status=JobStatus.COMPLETE.value,
                stage=None, duration_ms=duration_ms,
            )

    def active_count(self) -> int:
        with session_scope() as session:
            return repo.count_active_jobs(session)

    def active_job_ids(self) -> list[str]:
        with session_scope() as session:
            return repo.active_job_ids(session)

    def list(
        self,
        *,
        patient_id: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[list[dict], int]:
        """Job summaries for `GET /jobs` (P1-09) — no report bodies."""
        with session_scope() as session:
            jobs, total = repo.list_jobs(
                session, patient_id=patient_id, status=status, limit=limit, offset=offset
            )
            rows = [
                {
                    "job_id": job.id,
                    "status": job.status,
                    "stage": job.stage,
                    "error": job.error,
                    "patient_id": job.patient_id,
                    "external_session_id": job.external_session_id,
                    "external_doctor_id": job.external_doctor_id,
                    "filename": job.audio.original_filename if job.audio else None,
                    "created_at": job.created_at,
                    "updated_at": job.updated_at,
                }
                for job in jobs
            ]
            return rows, total
