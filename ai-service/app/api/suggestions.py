"""EXPERTA_MED suggestion and clinician-feedback endpoints."""
from __future__ import annotations

from datetime import datetime
from typing import Any, Literal, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from ..db import repo, session_scope

router = APIRouter(tags=["suggestions"])


class SuggestionResponse(BaseModel):
    id: str
    job_id: Optional[str] = None
    patient_id: Optional[str] = None
    scope: str
    rule_id: str
    rule_version: str
    severity: str
    condition: Optional[str] = None
    title_ar: str
    detail_ar: str
    missing_tests: list[Any] = Field(default_factory=list)
    evidence: list[Any] = Field(default_factory=list)
    inference_chain: list[Any] = Field(default_factory=list)
    references: list[Any] = Field(default_factory=list)
    created_at: datetime


class SuggestionListResponse(BaseModel):
    total: int
    suggestions: list[SuggestionResponse] = Field(default_factory=list)


class FeedbackRequest(BaseModel):
    action: Literal["accepted", "rejected", "deferred", "acted"]
    actor: Optional[str] = None
    reason: Optional[str] = None


class FeedbackResponse(BaseModel):
    id: str
    suggestion_id: str
    action: str
    actor: Optional[str] = None
    reason: Optional[str] = None
    created_at: datetime


def _suggestion(row) -> SuggestionResponse:
    return SuggestionResponse(
        id=row.id,
        job_id=row.job_id,
        patient_id=row.patient_id,
        scope=row.scope,
        rule_id=row.rule_id,
        rule_version=row.rule_version,
        severity=row.severity,
        condition=row.condition,
        title_ar=row.title_ar,
        detail_ar=row.detail_ar,
        missing_tests=row.missing_tests or [],
        evidence=row.evidence or [],
        inference_chain=row.inference_chain or [],
        references=row.references or [],
        created_at=row.created_at,
    )


@router.get("/jobs/{job_id}/suggestions", response_model=SuggestionListResponse)
def get_job_suggestions(job_id: str) -> SuggestionListResponse:
    with session_scope() as session:
        if repo.get_job(session, job_id) is None:
            raise HTTPException(404, "Unknown job_id.")
        rows = repo.suggestions_for_job(session, job_id)
        return SuggestionListResponse(
            total=len(rows), suggestions=[_suggestion(row) for row in rows]
        )


@router.get("/patients/{patient_id}/suggestions", response_model=SuggestionListResponse)
def get_patient_suggestions(patient_id: str, limit: int = 200) -> SuggestionListResponse:
    with session_scope() as session:
        if repo.get_patient(session, patient_id) is None:
            raise HTTPException(404, "Unknown patient_id.")
        rows = repo.suggestions_for_patient(session, patient_id, limit=min(limit, 500))
        return SuggestionListResponse(
            total=len(rows), suggestions=[_suggestion(row) for row in rows]
        )


@router.post(
    "/suggestions/{suggestion_id}/feedback",
    response_model=FeedbackResponse,
    status_code=201,
)
def add_feedback(suggestion_id: str, body: FeedbackRequest) -> FeedbackResponse:
    with session_scope() as session:
        try:
            row = repo.add_suggestion_feedback(
                session,
                suggestion_id=suggestion_id,
                action=body.action,
                actor=body.actor,
                reason=body.reason,
            )
        except LookupError:
            raise HTTPException(404, "Unknown suggestion_id.") from None
        return FeedbackResponse(
            id=row.id,
            suggestion_id=row.suggestion_id,
            action=row.action,
            actor=row.actor,
            reason=row.reason,
            created_at=row.created_at,
        )


@router.get(
    "/suggestions/{suggestion_id}/feedback",
    response_model=list[FeedbackResponse],
)
def get_feedback(suggestion_id: str) -> list[FeedbackResponse]:
    with session_scope() as session:
        if repo.get_suggestion(session, suggestion_id) is None:
            raise HTTPException(404, "Unknown suggestion_id.")
        rows = repo.feedback_for_suggestion(session, suggestion_id)
        return [
            FeedbackResponse(
                id=row.id,
                suggestion_id=row.suggestion_id,
                action=row.action,
                actor=row.actor,
                reason=row.reason,
                created_at=row.created_at,
            )
            for row in rows
        ]
