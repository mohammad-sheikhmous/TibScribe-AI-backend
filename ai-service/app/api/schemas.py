"""Thin API-layer response models (kept separate from the domain Report model)."""
from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class UploadResponse(BaseModel):
    job_id: str
    status: str
    audio_sha256: Optional[str] = None
    # True when this exact recording was already archived: it is stored once and
    # referenced by both jobs (content-addressed store).
    duplicate_audio: bool = False
    patient_id: Optional[str] = None
    external_session_id: Optional[str] = None
    idempotent_replay: bool = False


class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    stage: Optional[str] = None
    error: Optional[str] = None
    external_session_id: Optional[str] = None


class JobSummary(BaseModel):
    job_id: str
    status: str
    stage: Optional[str] = None
    error: Optional[str] = None
    patient_id: Optional[str] = None
    external_session_id: Optional[str] = None
    filename: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


class JobListResponse(BaseModel):
    total: int
    limit: int
    offset: int
    jobs: list[JobSummary] = Field(default_factory=list)


class StorageStats(BaseModel):
    """Audio archive health. There is no delete action here on purpose — growth is
    reported and warned about, never trimmed (PLAN_V2 §7)."""

    objects: int
    total_bytes: int
    total_gb: float
    oldest_upload: Optional[datetime] = None
    never_verified: int = 0
    warn_threshold_gb: float
    over_threshold: bool
    note: str = "Audio is retained permanently by design; this endpoint never deletes."
