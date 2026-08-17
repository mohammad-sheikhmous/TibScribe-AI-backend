"""Patient endpoints (IMPLEMENTATION.md P2-01 / P2-04 / P2-06).

Why this router is the "cohesion" work rather than a CRUD nicety: without a patient
identity, every upload is an island. The temporal rules in EXPERTA_MED (persistent
symptom, worsening blood pressure) need a *series*, and until now the only way to make
one was to hand-copy report files into a folder. An MRN on upload does it automatically.

`GET /patients/{id}` also answers the one question the clinical rules must not get
wrong: is this patient pregnant *right now*? That comes from the state machine
(`db/patient_state.py`), which expires, rather than from a substring search over an old
transcript.
"""
from __future__ import annotations

from datetime import date, datetime, timezone
from typing import Optional

from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel, Field

from ..core.nlp.context import ContextObservation
from ..core.kbs.refresh import refresh_reasoning_for_job
from ..db import repo, session_scope
from ..db.patient_state import current_state, record_observation, state_history

router = APIRouter(prefix="/patients", tags=["patients"])


class PatientCreate(BaseModel):
    mrn: Optional[str] = None
    display_name: Optional[str] = None
    birth_date: Optional[date] = None
    notes: Optional[str] = None


class PatientOut(BaseModel):
    id: str
    mrn: Optional[str] = None
    display_name: Optional[str] = None
    birth_date: Optional[date] = None
    created_at: Optional[datetime] = None


class PatientListOut(BaseModel):
    total: int
    limit: int
    offset: int
    patients: list[PatientOut] = Field(default_factory=list)


class StateOverride(BaseModel):
    """A clinician correcting the inferred state — recorded as `explicit`, and it wins
    over inference because it is newer, not because of a special case."""

    pregnant: Optional[bool] = None
    postpartum: bool = False
    ga_weeks: Optional[int] = Field(default=None, ge=1, le=44)
    effective_at: Optional[datetime] = None
    refresh_job_id: Optional[str] = Field(default=None, exclude=True)


def _patient_out(patient) -> PatientOut:
    return PatientOut(
        id=patient.id, mrn=patient.mrn, display_name=patient.display_name,
        birth_date=patient.birth_date, created_at=patient.created_at,
    )


@router.post("", response_model=PatientOut, status_code=201)
async def create_patient(payload: PatientCreate) -> PatientOut:
    with session_scope() as session:
        if payload.mrn:
            existing = repo.get_or_create_patient(session, mrn=payload.mrn,
                                                  display_name=payload.display_name)
            return _patient_out(existing)
        return _patient_out(
            repo.create_patient(
                session, display_name=payload.display_name,
                birth_date=payload.birth_date, notes=payload.notes,
            )
        )


@router.get("", response_model=PatientListOut)
async def list_patients(
    q: Optional[str] = Query(None, description="match against MRN or name"),
    limit: int = 50,
    offset: int = 0,
) -> PatientListOut:
    with session_scope() as session:
        rows, total = repo.list_patients(session, q=q, limit=min(limit, 200), offset=offset)
        return PatientListOut(
            total=total, limit=limit, offset=offset,
            patients=[_patient_out(row) for row in rows],
        )


@router.put("/by-external/{source}/{external_id}", status_code=200)
async def ensure_patient_by_external(source: str, external_id: str) -> dict:
    """Idempotently establish an opaque gateway patient mapping without PII."""
    if not source.strip() or not external_id.strip():
        raise HTTPException(400, "source and external_id are required.")
    with session_scope() as session:
        patient = repo.get_or_create_external_patient(
            session, source=source.strip(), external_id=external_id.strip()
        )
        return {"patient": _patient_out(patient).model_dump()}


@router.get("/by-external/{source}/{external_id}")
async def get_patient_by_external(source: str, external_id: str) -> dict:
    """Resolve a trusted gateway identity to the AI service's opaque patient id."""
    with session_scope() as session:
        patient = repo.get_patient_by_external(session, source=source, external_id=external_id)
        if patient is None:
            raise HTTPException(404, "Unknown external patient.")
        return {"patient": _patient_out(patient).model_dump()}


@router.get("/{patient_id}")
async def get_patient(patient_id: str, at: Optional[datetime] = None) -> dict:
    """Patient record + obstetric state (as of `at`, default now) + visit count."""
    moment = at or datetime.now(timezone.utc)
    with session_scope() as session:
        patient = repo.get_patient(session, patient_id)
        if patient is None:
            raise HTTPException(404, "Unknown patient.")
        _, job_count = repo.list_jobs(session, patient_id=patient_id, limit=1)
        state = current_state(session, patient_id, at=moment)
        return {
            "patient": _patient_out(patient).model_dump(),
            "jobs": job_count,
            "obstetric_state": state.to_dict(),
            # None unless currently pregnant — projected from when it was observed.
            "gestational_age_weeks_now": state.ga_weeks_at(moment),
        }


@router.get("/{patient_id}/timeline")
async def get_timeline(patient_id: str) -> dict:
    """Every visit for this patient, newest first — the input to series reasoning."""
    with session_scope() as session:
        if repo.get_patient(session, patient_id) is None:
            raise HTTPException(404, "Unknown patient.")
        entries = repo.patient_timeline(session, patient_id)
        history = [
            {
                "effective_at": row.effective_at,
                "status": row.status,
                "ga_weeks": row.ga_weeks,
                "source": row.source,
                "job_id": row.job_id,
            }
            for row in state_history(session, patient_id)
        ]
        state = current_state(session, patient_id)
        return {
            "patient_id": patient_id,
            "visits": len(entries),
            "obstetric_state": state.to_dict(),
            "state_history": history,
            "timeline": entries,
        }


@router.post("/{patient_id}/state", status_code=201)
async def set_state(patient_id: str, payload: StateOverride, request: Request) -> dict:
    """Record a clinician-stated context, overriding what the transcript implied."""
    with session_scope() as session:
        if repo.get_patient(session, patient_id) is None:
            raise HTTPException(404, "Unknown patient.")
        if payload.postpartum and (payload.pregnant is True or payload.ga_weeks is not None):
            raise HTTPException(422, "postpartum cannot be combined with pregnant=true or ga_weeks.")
        explicit_pregnant = payload.pregnant
        if payload.ga_weeks is not None and explicit_pregnant is None and not payload.postpartum:
            # Gestational age is itself an explicit assertion of current pregnancy.
            explicit_pregnant = True
        observation = ContextObservation(
            pregnant=explicit_pregnant,
            postpartum=payload.postpartum,
            ga_weeks=payload.ga_weeks,
            evidence=["أدخلها الطبيب يدوياً"],
        )
        if not observation.has_signal:
            raise HTTPException(400, "Nothing to record: set pregnant, postpartum or ga_weeks.")
        record_observation(
            session, patient_id=patient_id, observation=observation,
            effective_at=payload.effective_at, source="explicit",
        )
        state = current_state(session, patient_id).to_dict()

    if payload.refresh_job_id:
        await refresh_reasoning_for_job(
            request, payload.refresh_job_id, expected_patient_id=patient_id, record_revision=False
        )
    return state
