"""Recompute derived patient context and KBS advice for an existing completed job.

Used after clinician corrections and after an explicit obstetric-state override tied to
that same visit. Keeping this in one place prevents two update paths from drifting and
serving stale suggestions.
"""
from __future__ import annotations

import asyncio
from functools import partial

from fastapi import HTTPException, Request

from .service import resolve_effective_context
from ...db import repo, session_scope
from ...db.patient_state import UNKNOWN, current_state, record_context_revision


async def refresh_reasoning_for_job(
    request: Request,
    job_id: str,
    *,
    expected_patient_id: str | None = None,
    record_revision: bool = False,
) -> None:
    analyzer = getattr(request.app.state, "kbs_analyzer", None)
    if not callable(analyzer):
        # Lightweight unit-test/embedding apps intentionally omit EXPERTA_MED.
        return

    with session_scope() as session:
        job = repo.get_job(session, job_id)
        report = repo.load_report(session, job_id)
        if job is None or report is None or job.status != "complete":
            raise HTTPException(409, "Completed report is not available for reasoning refresh.")
        if expected_patient_id is not None and job.patient_id != expected_patient_id:
            raise HTTPException(409, "The requested report does not belong to this patient.")
        if job.patient_id:
            history = repo.recent_reports_for_patient(
                session, job.patient_id, exclude_job_id=job_id,
                before=report.created_at, limit=5,
            )
            prior_state = current_state(
                session, job.patient_id, at=report.created_at, exclude_job_id=job_id
            )
        else:
            history, prior_state = [], UNKNOWN
        patient_id = job.patient_id

    _observation, effective_context = resolve_effective_context(prior_state, report)
    if patient_id:
        report.patient_info.update(effective_context)

    # Never expose stale advice while recomputation is in progress/failing.
    with session_scope() as session:
        if patient_id:
            repo.update_report_patient_info(session, job_id, report.patient_info)
        repo.retire_suggestions_for_job(session, job_id)

    loop = asyncio.get_running_loop()
    executor = getattr(request.app.state, "executor", None)
    call = partial(analyzer, report, history, patient_context=effective_context)
    analysis = await loop.run_in_executor(executor, call) if executor else call()

    with session_scope() as session:
        repo.save_suggestions(
            session, job_id=job_id, patient_id=patient_id,
            suggestions=list(analysis.result.get("suggestions") or []), scope="visit",
        )
        if patient_id and record_revision:
            record_context_revision(
                session, patient_id=patient_id, job_id=job_id,
                effective_at=report.created_at, effective_context=effective_context,
            )
