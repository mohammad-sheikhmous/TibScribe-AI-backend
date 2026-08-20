"""Doctor corrections and the review queue (IMPLEMENTATION.md P7-01 / P7-04).

The most valuable data in this system is already being produced and thrown away: every
time a doctor reads a generated report, they silently correct it in their head. This
router captures that.

Two properties make it worth capturing rather than merely allowing:

* **A correction is an append.** The old value is kept alongside the new one. The pair
  (predicted -> corrected) is the training signal; storing only the correction would
  discard the more informative half, and a clinical record must show what changed.
* **The queue is ordered by uncertainty, not by document order.** Reading the three
  least-certain sentences teaches the model far more than reading the first three, so
  the doctor's minutes buy the most information available (P7-04, measured by P7-05).
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Literal, Optional

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field, model_validator

from ..core.kbs.refresh import refresh_reasoning_for_job
from ..core.nlp.sections import ACTIVE_LABELS, SOAP_ORDER
from ..db import repo, session_scope

logger = logging.getLogger(__name__)

router = APIRouter(tags=["corrections"])

CORRECTABLE_FIELDS = ("label", "text", "soap_section", "speaker")


class ItemCorrection(BaseModel):
    """Only the fields a clinician can meaningfully judge are correctable.

    Confidence, timestamps and entity links are derived — correcting them by hand would
    create a record that no longer describes anything the system did.
    """

    label: Optional[str] = None
    text: Optional[str] = None
    soap_section: Optional[Literal["subjective", "objective", "assessment", "plan"]] = None
    speaker: Optional[Literal["doctor", "patient", "unknown"]] = None
    actor: Optional[str] = Field(default=None, description="who made the correction")

    @model_validator(mode="after")
    def at_least_one_change(self):
        if not any(getattr(self, field) is not None for field in CORRECTABLE_FIELDS):
            raise ValueError(f"provide at least one of {CORRECTABLE_FIELDS}")
        if self.label is not None and self.label not in ACTIVE_LABELS:
            raise ValueError(f"unknown label {self.label!r}; expected one of {ACTIVE_LABELS}")
        return self


class CorrectionOut(BaseModel):
    id: str
    item_id: str
    field: str
    old_value: Optional[str] = None
    new_value: Optional[str] = None
    actor: Optional[str] = None
    created_at: Optional[datetime] = None


class ReviewItem(BaseModel):
    item_id: str
    job_id: str
    order_index: int
    text: str
    text_canonical: Optional[str] = None
    label: str
    confidence: float
    review_priority: Optional[float] = None
    reasons: list[str] = Field(default_factory=list)
    start_sec: float
    end_sec: float


@router.patch("/jobs/{job_id}/items/{item_id}", response_model=list[CorrectionOut])
async def correct_item(
    job_id: str, item_id: str, payload: ItemCorrection, request: Request
) -> list[CorrectionOut]:
    """Correct one report item and refresh every derived clinical artifact.

    A clinician edit is not merely cosmetic: text/label changes can alter entities,
    patient context, and the deterministic KBS output. Those derived layers are
    recomputed before this request succeeds, so a finalized report can never contain
    corrected SOAP text paired with stale clinical suggestions.
    """
    changed_fields: set[str] = set()
    written: list[CorrectionOut] = []
    with session_scope() as session:
        item = repo.get_item(session, job_id, item_id)
        if item is None:
            raise HTTPException(404, "Unknown report item.")

        for field in CORRECTABLE_FIELDS:
            new_value = getattr(payload, field)
            if new_value is None or str(getattr(item, field, None)) == str(new_value):
                continue
            correction = repo.apply_correction(
                session, item=item, field=field, new_value=str(new_value),
                actor=payload.actor,
            )
            changed_fields.add(field)
            written.append(CorrectionOut(
                id=correction.id, item_id=correction.item_id, field=correction.field,
                old_value=correction.old_value, new_value=correction.new_value,
                actor=correction.actor, created_at=correction.created_at,
            ))

        if not written:
            raise HTTPException(409, "Nothing to correct: the values are already those.")

        if changed_fields & {"text", "label"}:
            repo.refresh_item_entities(session, item)
        repo.refresh_report_summary(session, job_id)

    await refresh_reasoning_for_job(request, job_id, record_revision=True)
    logger.info("job %s item %s corrected: %s", job_id, item_id, sorted(changed_fields))
    return written


@router.get("/jobs/{job_id}/corrections", response_model=list[CorrectionOut])
async def list_corrections(job_id: str) -> list[CorrectionOut]:
    """Every correction made to this report — the audit trail of what a human changed."""
    with session_scope() as session:
        return [
            CorrectionOut(
                id=c.id, item_id=c.item_id, field=c.field, old_value=c.old_value,
                new_value=c.new_value, actor=c.actor, created_at=c.created_at,
            )
            for c in repo.corrections_for_job(session, job_id)
        ]


@router.get("/jobs/{job_id}/review-queue", response_model=list[ReviewItem])
async def job_review_queue(job_id: str, limit: int = 5) -> list[ReviewItem]:
    """The few sentences worth a doctor's attention first, in this report.

    Default of 5: the point is that a short, well-chosen list beats reading everything.
    """
    with session_scope() as session:
        return [_to_review_item(row) for row in
                repo.review_queue(session, job_id=job_id, limit=min(limit, 50))]


@router.get("/review-queue", response_model=list[ReviewItem])
async def global_review_queue(limit: int = 20) -> list[ReviewItem]:
    """Across all reports — the labelling worklist for active learning."""
    with session_scope() as session:
        return [_to_review_item(row) for row in
                repo.review_queue(session, limit=min(limit, 200))]


def _to_review_item(row) -> ReviewItem:
    return ReviewItem(
        item_id=row.item_id, job_id=row.job_id, order_index=row.order_index,
        text=row.text, text_canonical=getattr(row, "text_canonical", None),label=row.label, confidence=row.confidence,
        review_priority=row.review_priority,
        reasons=row.low_confidence_reasons or [],
        start_sec=row.start_sec, end_sec=row.end_sec,
    )
