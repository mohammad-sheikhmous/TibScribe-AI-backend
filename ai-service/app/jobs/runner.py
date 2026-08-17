"""Background job runner for the complete clinical pipeline.

The job is considered COMPLETE only after all durable stages succeed:

    audio -> ASR -> AraBERT -> SOAP -> longitudinal context -> EXPERTA_MED -> DB

The JSON export is a convenience artifact only; the database is the source of truth.
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from datetime import datetime
from functools import partial
from types import SimpleNamespace
from typing import Optional

from fastapi import FastAPI

from ..core.kbs.service import resolve_effective_context
from ..core.report.schema import Report
from ..config import get_settings
from ..db import repo, session_scope
from ..db.patient_state import UNKNOWN, current_state, record_observation
from ..storage.paths import result_path
from .schema import JobStatus

logger = logging.getLogger(__name__)


def _commit_external_storage(app: FastAPI) -> None:
    """Persist provider-mounted storage when the deployment exposes a commit hook.

    Local/Docker deployments do not install the hook and remain a no-op. Modal sets
    ``app.state.storage_commit_hook`` to ``Volume.commit`` so newly archived audio and
    result exports survive container scale-down. Failures are logged but never rewrite
    an already committed clinical database transaction.
    """
    hook = getattr(app.state, "storage_commit_hook", None)
    if not callable(hook):
        return
    try:
        hook()
    except Exception:  # noqa: BLE001 - storage durability warning, job DB state remains authoritative
        logger.exception("External storage commit hook failed")


def _patient_info(job_id: str, patient_id: Optional[str], visit_at: datetime) -> dict:
    """Identity + obstetric context known before this visit."""
    if not patient_id:
        return {}
    with session_scope() as session:
        patient = repo.get_patient(session, patient_id)
        job = repo.get_job(session, job_id)
        visit = repo.get_visit(session, job.visit_id) if job and job.visit_id else None
        effective_visit_at = visit.visit_at if visit else visit_at
        if effective_visit_at.tzinfo is None:
            from datetime import timezone
            effective_visit_at = effective_visit_at.replace(tzinfo=timezone.utc)
        state = current_state(session, patient_id, at=effective_visit_at)
        _, visit_count = repo.list_jobs(session, patient_id=patient_id, limit=1)
        return {
            "patient_id": patient_id,
            "mrn": patient.mrn if patient else None,
            "visit_index": visit_count,
            "visit_at": effective_visit_at.isoformat(),
            "obstetric_status_before_visit": state.status,
            "gestational_age_weeks_before_visit": state.ga_weeks_at(effective_visit_at),
        }


def _history_and_state(
    patient_id: Optional[str], current_job_id: str, report: Report, *, history_limit: int
):
    if not patient_id:
        return [], UNKNOWN
    with session_scope() as session:
        history = repo.recent_reports_for_patient(
            session,
            patient_id,
            exclude_job_id=current_job_id,
            before=report.created_at,
            limit=history_limit,
        )
        state = current_state(session, patient_id, at=report.created_at)
        return history, state


def _persist_complete_result(
    *,
    job_id: str,
    patient_id: Optional[str],
    report: Report,
    transcript: dict,
    transcript_quality: dict,
    kbs_result: dict,
    observation,
    duration_ms: int,
) -> None:
    """One DB transaction for transcript + report + entities + KBS + final job state."""
    raw_json = json.dumps(transcript, ensure_ascii=False, default=str)
    with session_scope() as session:
        repo.save_transcript(
            session,
            job_id=job_id,
            language=transcript.get("language"),
            raw_json=raw_json,
            avg_logprob=transcript_quality.get("avg_logprob"),
            no_speech_ratio=transcript_quality.get("no_speech_ratio"),
        )
        repo.save_report(session, report)
        repo.save_suggestions(
            session,
            job_id=job_id,
            patient_id=patient_id,
            suggestions=list(kbs_result.get("suggestions") or []),
            scope="visit",
        )
        if patient_id and observation.has_signal:
            record_observation(
                session,
                patient_id=patient_id,
                observation=observation,
                effective_at=report.created_at,
                job_id=job_id,
            )
        repo.update_job(
            session,
            job_id,
            status=JobStatus.COMPLETE.value,
            stage=None,
            duration_ms=duration_ms,
            rules_version=str(kbs_result.get("rules_version") or ""),
            asr_model=report.audio.whisper_model,
            classifier_version=report.pipeline_meta.arabert_model_name,
        )


async def run_job(app: FastAPI, job_id: str) -> None:
    store = app.state.job_store
    rec = store.get(job_id)
    if rec is None:
        logger.error("run_job: unknown job_id %s", job_id)
        return

    settings = getattr(app.state, "settings", None) or get_settings()
    audio_path = str(app.state.audio_store.path_for(rec.audio_path))
    patient_info = _patient_info(job_id, rec.patient_id, rec.created_at)

    store.update_status(job_id, JobStatus.RUNNING, stage="transcribing")
    started = time.monotonic()
    loop = asyncio.get_running_loop()

    try:
        # Heavy ML stages run in the worker pool, never on the event loop.
        # `process_with_artifacts` is the production contract. The legacy `.process`
        # fallback exists only for lightweight test doubles and older embedding code;
        # main.py always wires the full production pipeline.
        process_with_artifacts = getattr(app.state.pipeline, "process_with_artifacts", None)
        if callable(process_with_artifacts):
            artifacts = await loop.run_in_executor(
                app.state.executor,
                partial(
                    process_with_artifacts,
                    audio_path,
                    job_id,
                    filename=rec.filename,
                    patient_info=patient_info,
                ),
            )
        else:
            report_only = await loop.run_in_executor(
                app.state.executor,
                partial(
                    app.state.pipeline.process,
                    audio_path,
                    job_id,
                    filename=rec.filename,
                    patient_info=patient_info,
                ),
            )
            artifacts = SimpleNamespace(
                report=report_only, transcript={}, transcript_quality={}
            )
        report = artifacts.report
        # For uploaded historical visits, clinical chronology follows visit_at rather
        # than processing time. This is critical for PatientState and temporal trends.
        if patient_info.get("visit_at"):
            report.created_at = datetime.fromisoformat(str(patient_info["visit_at"]))

        # Resolve the current longitudinal context before the KBS sees the report.
        history, prior_state = _history_and_state(
            rec.patient_id,
            job_id,
            report,
            history_limit=settings.kbs_history_reports,
        )
        supplemental_texts: list[str] = []
        raw_transcript_text = (artifacts.transcript or {}).get("text")
        if isinstance(raw_transcript_text, str) and raw_transcript_text.strip():
            supplemental_texts.append(raw_transcript_text)

        observation, effective_context = resolve_effective_context(
            prior_state,
            report,
            supplemental_texts=supplemental_texts,
        )
        if rec.patient_id:
            # Assign a fresh dict so persistence sees one authoritative snapshot of
            # pre-visit identity plus the context resolved from this visit.
            report.patient_info = {**(report.patient_info or {}), **effective_context}

        store.update_status(job_id, JobStatus.RUNNING, stage="reasoning")
        analyzer = getattr(app.state, "kbs_analyzer", None)
        if callable(analyzer):
            analysis = await loop.run_in_executor(
                app.state.executor,
                partial(
                    analyzer,
                    report,
                    history,
                    patient_context=effective_context,
                ),
            )
        else:
            # Minimal FastAPI apps used by unit tests intentionally omit the KBS.
            # The production app cannot reach this branch: lifespan() validates the
            # dependency and always installs `app.state.kbs_analyzer`.
            logger.warning(
                "Job %s is running without a KBS analyzer (test/embedded app mode)", job_id
            )
            analysis = SimpleNamespace(
                result={
                    "engine": "disabled-test-mode",
                    "rules_version": "",
                    "patient_context": effective_context,
                    "trends": [],
                    "suggestions": [],
                    "audit_trail": [],
                }
            )

        store.update_status(job_id, JobStatus.RUNNING, stage="persisting")
        elapsed_ms = int((time.monotonic() - started) * 1000)
        _persist_complete_result(
            job_id=job_id,
            patient_id=rec.patient_id,
            report=report,
            transcript=artifacts.transcript,
            transcript_quality=artifacts.transcript_quality,
            kbs_result=analysis.result,
            observation=observation,
            duration_ms=elapsed_ms,
        )

        _export_json(job_id, report, analysis.result)
        logger.info(
            "Job %s complete in %d ms (%d KBS suggestions)",
            job_id,
            elapsed_ms,
            len(analysis.result.get("suggestions") or []),
        )
    except Exception:  # noqa: BLE001 — full trace in logs, sanitized API message
        logger.exception("Job %s failed", job_id)
        store.update_status(
            job_id,
            JobStatus.FAILED,
            error="Processing failed. See server logs for details.",
        )
    finally:
        # On Modal this commits the persistent data Volume. On local/Docker it is a no-op.
        await loop.run_in_executor(None, _commit_external_storage, app)


def _export_json(job_id: str, report: Report, kbs_result: Optional[dict] = None) -> None:
    """Best-effort export. Failure never invalidates a successfully committed job."""
    try:
        out_path = result_path(job_id)
        payload = report.model_dump(mode="json")
        if kbs_result is not None:
            payload["kbs"] = {
                "engine": kbs_result.get("engine"),
                "rules_version": kbs_result.get("rules_version"),
                "patient_context": kbs_result.get("patient_context"),
                "trends": kbs_result.get("trends") or [],
                "suggestions": kbs_result.get("suggestions") or [],
                "audit_trail": kbs_result.get("audit_trail") or [],
            }
        out_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
    except OSError:
        logger.warning(
            "Job %s: could not write JSON export (database copy is intact)",
            job_id,
            exc_info=True,
        )
