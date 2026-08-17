"""Obstetric state machine (IMPLEMENTATION.md P2-06).

The second half of the fix for gap ك-٤. Text detection stops a *denial* being read as
a confirmation; this stops a confirmation lasting forever:

    visit 2023-01: "حامل بالأسبوع 30"
    visit 2026-07: "صداع وتشوش بالرؤية"
    -> the old code answered pregnant=True and fired a pre-eclampsia alert, three years
       after the delivery.

Pregnancy is bounded by biology, so state carries an expiry:

    pregnant   -> expires at (EDD + grace); silence past that is NOT pregnancy
    postpartum -> expires ~6 weeks after delivery, then unknown
    unknown    -> the honest default; no obstetric rule may fire on it

State is **append-only**: each visit adds an observation instead of overwriting one, so
the history stays auditable (and P6's temporal rules can read it). `current_state()`
picks the newest row that is still valid at a given moment.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Literal, Optional

from sqlalchemy import or_, select
from sqlalchemy.orm import Session

from ..core.nlp.context import ContextObservation, trimester_from_weeks
from .models import PatientState

Status = Literal["pregnant", "postpartum", "unknown"]

FULL_TERM_DAYS = 280            # 40 weeks from LMP
POST_TERM_GRACE_DAYS = 21       # delivery may run late / the note may be stale
POSTPARTUM_WINDOW_DAYS = 42     # the classical puerperium
# With no gestational age given, assume mid-pregnancy so the state still expires.
ASSUMED_GA_WEEKS_WHEN_UNKNOWN = 20


@dataclass(frozen=True)
class ResolvedState:
    status: Status
    effective_at: Optional[datetime] = None
    ga_weeks_at_observation: Optional[int] = None
    edd: Optional[datetime] = None
    expires_at: Optional[datetime] = None
    source: Optional[str] = None
    is_expired: bool = False
    note: Optional[str] = None

    @property
    def is_pregnant(self) -> bool:
        """The single question the clinical rules ask. Expired never means pregnant."""
        return self.status == "pregnant" and not self.is_expired

    def ga_weeks_at(self, moment: datetime) -> Optional[int]:
        """Project gestational age forward from when it was observed."""
        if not self.is_pregnant or self.ga_weeks_at_observation is None:
            return None
        if self.effective_at is None:
            return self.ga_weeks_at_observation
        elapsed_weeks = (_aware(moment) - _aware(self.effective_at)).days // 7
        projected = self.ga_weeks_at_observation + elapsed_weeks
        return projected if 0 < projected <= 45 else None

    def to_dict(self) -> dict:
        return {
            "status": self.status,
            "is_pregnant": self.is_pregnant,
            "effective_at": self.effective_at,
            "ga_weeks_at_observation": self.ga_weeks_at_observation,
            "edd": self.edd,
            "expires_at": self.expires_at,
            "source": self.source,
            "expired": self.is_expired,
            "note": self.note,
        }


UNKNOWN = ResolvedState(status="unknown")


def _aware(value: datetime) -> datetime:
    return value if value.tzinfo else value.replace(tzinfo=timezone.utc)


def compute_edd(effective_at: datetime, ga_weeks: Optional[int]) -> Optional[datetime]:
    """Estimated delivery date = observation + the weeks still to run."""
    if ga_weeks is None:
        return None
    return _aware(effective_at) + timedelta(days=FULL_TERM_DAYS - ga_weeks * 7)


def record_observation(
    session: Session,
    *,
    patient_id: str,
    observation: ContextObservation,
    effective_at: Optional[datetime] = None,
    job_id: Optional[str] = None,
    source: str = "inferred",
) -> Optional[PatientState]:
    """Append what a visit observed. Returns None when the visit said nothing."""
    if not observation.has_signal:
        return None  # silence is not evidence — the previous state simply stands

    moment = _aware(effective_at or datetime.now(timezone.utc))

    if observation.postpartum:
        status: Status = "postpartum"
        expires_at = moment + timedelta(days=POSTPARTUM_WINDOW_DAYS)
        edd = None
    elif observation.pregnant is True:
        status = "pregnant"
        ga = observation.ga_weeks or ASSUMED_GA_WEEKS_WHEN_UNKNOWN
        edd = compute_edd(moment, ga)
        expires_at = edd + timedelta(days=POST_TERM_GRACE_DAYS)
    elif observation.pregnant is False:
        status = "unknown"       # an explicit denial resets, it does not linger
        expires_at = None
        edd = None
    else:
        status = "unknown"
        expires_at = None
        edd = None

    state = PatientState(
        patient_id=patient_id,
        job_id=job_id,
        effective_at=moment,
        status=status,
        ga_weeks=observation.ga_weeks,
        trimester=observation.trimester
        or (trimester_from_weeks(observation.ga_weeks) if observation.ga_weeks else None),
        edd=edd,
        expires_at=expires_at,
        source=source,
        evidence="\n".join(observation.evidence[:5]) or None,
    )
    session.add(state)
    session.flush()
    return state


def current_state(
    session: Session, patient_id: str, *, at: Optional[datetime] = None,
    exclude_job_id: Optional[str] = None,
) -> ResolvedState:
    """The patient's obstetric state as of `at` (default: now)."""
    moment = _aware(at or datetime.now(timezone.utc))
    stmt = select(PatientState).where(
        PatientState.patient_id == patient_id,
        PatientState.effective_at <= moment.replace(tzinfo=None),
    )
    if exclude_job_id:
        stmt = stmt.where(
            or_(PatientState.job_id.is_(None), PatientState.job_id != exclude_job_id)
        )
    row = session.scalar(
        stmt.order_by(PatientState.effective_at.desc(), PatientState.created_at.desc()).limit(1)
    )
    if row is None:
        return UNKNOWN

    expires_at = _aware(row.expires_at) if row.expires_at else None
    expired = bool(expires_at and moment > expires_at)
    note = None
    if expired and row.status == "pregnant":
        note = (
            "آخر حمل مسجّل تجاوز تاريخه المتوقع — الحالة غير معروفة حتى تُوثَّق زيارة جديدة."
        )

    return ResolvedState(
        status="unknown" if expired else row.status,  # type: ignore[arg-type]
        effective_at=_aware(row.effective_at),
        ga_weeks_at_observation=row.ga_weeks,
        edd=_aware(row.edd) if row.edd else None,
        expires_at=expires_at,
        source=row.source,
        is_expired=expired,
        note=note,
    )


def record_context_revision(
    session: Session, *, patient_id: str, job_id: str, effective_at: datetime,
    effective_context: dict,
) -> PatientState:
    """Append the state implied by a clinician-corrected report.

    The original inferred row is deliberately retained for audit. A newer row at the
    same clinical time wins via ``created_at`` while future reasoning can still see
    exactly what the system believed before the correction.
    """
    status = str(effective_context.get("effective_obstetric_status") or "unknown")
    ga = effective_context.get("gestational_age_weeks")
    if status == "postpartum":
        observation = ContextObservation(postpartum=True, ga_weeks=ga, evidence=["clinician-corrected report context"])
    elif status == "pregnant":
        observation = ContextObservation(pregnant=True, ga_weeks=ga, evidence=["clinician-corrected report context"])
    else:
        observation = ContextObservation(pregnant=False, evidence=["clinician-corrected report context"])
    row = record_observation(
        session, patient_id=patient_id, observation=observation,
        effective_at=effective_at, job_id=job_id, source="corrected",
    )
    assert row is not None
    return row


def state_history(session: Session, patient_id: str) -> list[PatientState]:
    return list(
        session.scalars(
            select(PatientState)
            .where(PatientState.patient_id == patient_id)
            .order_by(PatientState.effective_at.asc())
        )
    )
