"""Job state models.

`JobRecord` is now a read-through view of the `jobs` row (plus its audio and, when
complete, its report) rather than the storage itself — the database is the source of
truth. Kept as a plain dataclass so the router and runner did not have to change when
persistence landed.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from ..core.report.schema import Report


class JobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETE = "complete"
    FAILED = "failed"


def _now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class JobRecord:
    job_id: str
    audio_id: Optional[str] = None
    audio_path: Optional[str] = None   # store-relative key, not an absolute path
    filename: Optional[str] = None
    patient_id: Optional[str] = None
    visit_id: Optional[str] = None
    external_session_id: Optional[str] = None
    external_doctor_id: Optional[str] = None
    status: JobStatus = JobStatus.QUEUED
    stage: Optional[str] = None
    error: Optional[str] = None
    result: Optional[Report] = None
    created_at: datetime = field(default_factory=_now)
    updated_at: datetime = field(default_factory=_now)
