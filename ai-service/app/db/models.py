"""SQLAlchemy models — the durable side of the service (IMPLEMENTATION.md P1-01).

Normalization boundary (deliberate, see ADR-008):
  * **Rows** for anything queryable or referenceable — jobs, report items, entities,
    suggestions, corrections. A correction points at one item; an entity points at one
    item; a suggestion cites item ids. These must be real rows with real foreign keys.
  * **JSON columns** for self-contained metadata blobs that are read whole and never
    queried by field — `audio`, `pipeline_meta`, `summary`, `patient_info`. Storing
    them as JSON means P3/P4 can add fields (speaker, asr_confidence, calibration)
    without a migration for every one.

The pydantic `Report` is NOT stored as a blob: it is reconstructed from these rows
(`repo.load_report`), so there is exactly one source of truth and no drift between a
cached document and the rows that were edited.

Retention: nothing here is ever deleted by the application. `audio_files` in
particular is append-only by design (PLAN_V2 §7) — there is no delete path at all.
"""
from __future__ import annotations

import uuid
from datetime import date, datetime, timezone
from typing import Any, Optional

from sqlalchemy import (
    JSON,
    Boolean,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def new_id() -> str:
    return uuid.uuid4().hex


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    """Shared declarative base. `type_annotation_map` keeps JSON columns terse."""

    type_annotation_map = {dict[str, Any]: JSON, list[Any]: JSON}


# --- clinical identity --------------------------------------------------------------

class Patient(Base):
    __tablename__ = "patients"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    mrn: Mapped[Optional[str]] = mapped_column(String(64), unique=True, index=True)
    # Opaque identity owned by the public backend (Laravel). No patient PII is needed
    # here; the pair (source, id) is enough to keep longitudinal AI context.
    external_source: Mapped[Optional[str]] = mapped_column(String(32), index=True)
    external_id: Mapped[Optional[str]] = mapped_column(String(100), index=True)
    display_name: Mapped[Optional[str]] = mapped_column(String(200))
    birth_date: Mapped[Optional[date]] = mapped_column(Date)
    notes: Mapped[Optional[str]] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    visits: Mapped[list["Visit"]] = relationship(back_populates="patient")
    jobs: Mapped[list["Job"]] = relationship(back_populates="patient")

    __table_args__ = (
        UniqueConstraint("external_source", "external_id", name="uq_patient_external_identity"),
    )


class Visit(Base):
    __tablename__ = "visits"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    patient_id: Mapped[str] = mapped_column(ForeignKey("patients.id"), index=True)
    visit_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    kind: Mapped[Optional[str]] = mapped_column(String(32))  # anc|postpartum|gyn|emergency
    gestational_age_weeks: Mapped[Optional[int]] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    patient: Mapped[Patient] = relationship(back_populates="visits")


class PatientState(Base):
    """Append-only obstetric state observations (see db/patient_state.py).

    One row per visit that said something about pregnancy/postpartum. Never updated in
    place: the history is what lets a later reader see *when* a state was believed, and
    `expires_at` is what stops a 2023 pregnancy still being "true" in 2026 (gap ك-٤).
    """

    __tablename__ = "patient_states"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    patient_id: Mapped[str] = mapped_column(ForeignKey("patients.id"), index=True)
    job_id: Mapped[Optional[str]] = mapped_column(ForeignKey("jobs.id"))

    effective_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    status: Mapped[str] = mapped_column(String(16))  # pregnant|postpartum|unknown
    ga_weeks: Mapped[Optional[int]] = mapped_column(Integer)
    trimester: Mapped[Optional[int]] = mapped_column(Integer)
    edd: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    # After this moment the observation no longer implies the status (biology, not text).
    expires_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    source: Mapped[str] = mapped_column(String(16), default="inferred")  # inferred|explicit
    evidence: Mapped[Optional[str]] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    __table_args__ = (Index("ix_patient_states_patient_time", "patient_id", "effective_at"),)


# --- audio: append-only, never deleted ----------------------------------------------

class AudioFile(Base):
    """One physical recording. Content-addressed, so re-uploading the same file
    reuses this row instead of storing a second copy."""

    __tablename__ = "audio_files"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    sha256: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    storage_key: Mapped[str] = mapped_column(String(255))
    original_filename: Mapped[Optional[str]] = mapped_column(String(255))
    mime: Mapped[Optional[str]] = mapped_column(String(100))
    size_bytes: Mapped[int] = mapped_column(Integer, default=0)
    duration_sec: Mapped[Optional[float]] = mapped_column(Float)
    codec: Mapped[Optional[str]] = mapped_column(String(32))
    uploaded_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    uploaded_by: Mapped[Optional[str]] = mapped_column(String(100))
    checksum_verified_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    # "moved to cold storage", never "removed" — see PLAN_V2 §7.
    is_archived: Mapped[bool] = mapped_column(Boolean, default=False)


# --- jobs ---------------------------------------------------------------------------

class Job(Base):
    __tablename__ = "jobs"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    patient_id: Mapped[Optional[str]] = mapped_column(ForeignKey("patients.id"), index=True)
    visit_id: Mapped[Optional[str]] = mapped_column(ForeignKey("visits.id"))
    audio_id: Mapped[Optional[str]] = mapped_column(ForeignKey("audio_files.id"), index=True)

    # Correlation identifiers supplied by trusted Laravel. external_session_id is
    # unique so POST /jobs is idempotent across HTTP retries.
    external_session_id: Mapped[Optional[str]] = mapped_column(String(100), unique=True, index=True)
    external_doctor_id: Mapped[Optional[str]] = mapped_column(String(100), index=True)

    status: Mapped[str] = mapped_column(String(16), default="queued", index=True)
    stage: Mapped[Optional[str]] = mapped_column(String(32))
    error: Mapped[Optional[str]] = mapped_column(Text)

    # which model versions produced this job's output (reproducibility)
    asr_model: Mapped[Optional[str]] = mapped_column(String(64))
    classifier_version: Mapped[Optional[str]] = mapped_column(String(64))
    ner_version: Mapped[Optional[str]] = mapped_column(String(64))
    rules_version: Mapped[Optional[str]] = mapped_column(String(32))

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow,
                                                 index=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    duration_ms: Mapped[Optional[int]] = mapped_column(Integer)

    patient: Mapped[Optional[Patient]] = relationship(back_populates="jobs")
    audio: Mapped[Optional[AudioFile]] = relationship()
    report: Mapped[Optional["ReportRecord"]] = relationship(
        back_populates="job", uselist=False
    )
    items: Mapped[list["ReportItemRow"]] = relationship(
        back_populates="job", order_by="ReportItemRow.order_index"
    )


class Transcript(Base):
    """Whisper's raw output, kept whole — no information is discarded (principle P1)."""

    __tablename__ = "transcripts"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    job_id: Mapped[str] = mapped_column(ForeignKey("jobs.id"), unique=True, index=True)
    language: Mapped[Optional[str]] = mapped_column(String(16))
    raw_json: Mapped[Optional[str]] = mapped_column(Text)
    avg_logprob: Mapped[Optional[float]] = mapped_column(Float)
    no_speech_ratio: Mapped[Optional[float]] = mapped_column(Float)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


# --- the report ---------------------------------------------------------------------

class ReportRecord(Base):
    """Report-level metadata. The items live in `report_items`; together they
    reconstruct the pydantic `Report` exactly (see repo.load_report)."""

    __tablename__ = "reports"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    job_id: Mapped[str] = mapped_column(ForeignKey("jobs.id"), unique=True, index=True)
    schema_version: Mapped[str] = mapped_column(String(16), default="1.0")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    audio: Mapped[dict[str, Any]] = mapped_column(default=dict)
    pipeline_meta: Mapped[dict[str, Any]] = mapped_column(default=dict)
    patient_info: Mapped[dict[str, Any]] = mapped_column(default=dict)
    summary: Mapped[dict[str, Any]] = mapped_column(default=dict)

    job: Mapped[Job] = relationship(back_populates="report")


class ReportItemRow(Base):
    """One classified sentence. `item_id` is the natural key ({job_id}-{order:04d})
    and is what corrections, entities and suggestion evidence all point at."""

    __tablename__ = "report_items"

    item_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    job_id: Mapped[str] = mapped_column(ForeignKey("jobs.id"), index=True)
    order_index: Mapped[int] = mapped_column(Integer)

    # ``text`` remains the backwards-compatible effective text consumed by existing
    # APIs/KBS.  The immutable ASR baseline and accepted canonical candidate are kept
    # separately so every generated rewrite stays auditable.
    text: Mapped[str] = mapped_column(Text)
    text_raw: Mapped[Optional[str]] = mapped_column(Text)
    text_canonical: Mapped[Optional[str]] = mapped_column(Text)
    canonicalization_status: Mapped[str] = mapped_column(String(32), default="not_run")
    canonicalization_confidence: Mapped[Optional[float]] = mapped_column(Float)
    canonicalization_model: Mapped[Optional[str]] = mapped_column(String(255))
    canonicalization_reasons: Mapped[Optional[list[Any]]] = mapped_column(JSON)
    # Deprecated compatibility field.  Presentation formatting now lives at report level.
    text_rephrased: Mapped[Optional[str]] = mapped_column(Text)

    label: Mapped[str] = mapped_column(String(64), index=True)   # primary label
    labels: Mapped[Optional[list[Any]]] = mapped_column(JSON)    # multi-label (P4-09)
    label_ar: Mapped[Optional[str]] = mapped_column(String(128))
    soap_section: Mapped[str] = mapped_column(String(16))
    also_in_sections: Mapped[Optional[list[Any]]] = mapped_column(JSON)
    confidence: Mapped[float] = mapped_column(Float)          # classifier only
    combined_confidence: Mapped[Optional[float]] = mapped_column(Float)     # x ASR (P3-11)
    confidence_calibrated: Mapped[Optional[float]] = mapped_column(Float)   # P4-03

    speaker: Mapped[Optional[str]] = mapped_column(String(16))            # P3-08
    speaker_confidence: Mapped[Optional[float]] = mapped_column(Float)    # P3-08
    asr_confidence: Mapped[Optional[float]] = mapped_column(Float)        # P3-02
    # Flagged, never dropped: a suspected hallucination stays in the record.
    is_asr_suspect: Mapped[bool] = mapped_column(Boolean, default=False)  # P3-03

    start_sec: Mapped[float] = mapped_column(Float)
    end_sec: Mapped[float] = mapped_column(Float)
    timestamp_precision: Mapped[str] = mapped_column(String(16), default="segment")
    source_segment_index: Mapped[int] = mapped_column(Integer, default=0)

    # --- uncertainty (P4-06): the evidence behind is_low_confidence ---
    entropy: Mapped[Optional[float]] = mapped_column(Float)
    ood_score: Mapped[Optional[float]] = mapped_column(Float)
    low_confidence_reasons: Mapped[Optional[list[Any]]] = mapped_column(JSON)

    is_urgent: Mapped[bool] = mapped_column(Boolean, default=False)
    is_low_confidence: Mapped[bool] = mapped_column(Boolean, default=False)
    is_corrected: Mapped[bool] = mapped_column(Boolean, default=False)  # P7-01
    # Ranks what a human should review first — the active-learning selector (P7-04).
    review_priority: Mapped[Optional[float]] = mapped_column(Float)

    entity_links: Mapped[Optional[list[Any]]] = mapped_column(JSON)     # P5-14
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    job: Mapped[Job] = relationship(back_populates="items")

    __table_args__ = (Index("ix_report_items_job_order", "job_id", "order_index"),)


# --- downstream layers (tables created now, populated in P5/P6/P7) ------------------

class Entity(Base):
    __tablename__ = "entities"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    item_id: Mapped[str] = mapped_column(ForeignKey("report_items.item_id"), index=True)
    job_id: Mapped[str] = mapped_column(ForeignKey("jobs.id"), index=True)
    patient_id: Mapped[Optional[str]] = mapped_column(ForeignKey("patients.id"), index=True)

    kind: Mapped[str] = mapped_column(String(32))
    code: Mapped[str] = mapped_column(String(64), index=True)
    assertion: Mapped[str] = mapped_column(String(16), default="present")
    value: Mapped[Optional[float]] = mapped_column(Float)
    value2: Mapped[Optional[float]] = mapped_column(Float)
    unit: Mapped[Optional[str]] = mapped_column(String(16))
    status: Mapped[Optional[str]] = mapped_column(String(16))
    char_start: Mapped[Optional[int]] = mapped_column(Integer)
    char_end: Mapped[Optional[int]] = mapped_column(Integer)

    extractor: Mapped[str] = mapped_column(String(32), default="lexicon")
    extractor_version: Mapped[Optional[str]] = mapped_column(String(32))
    confidence: Mapped[float] = mapped_column(Float, default=1.0)


class Suggestion(Base):
    __tablename__ = "suggestions"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    job_id: Mapped[Optional[str]] = mapped_column(ForeignKey("jobs.id"), index=True)
    patient_id: Mapped[Optional[str]] = mapped_column(ForeignKey("patients.id"), index=True)
    scope: Mapped[str] = mapped_column(String(16), default="visit")  # visit|series

    rule_id: Mapped[str] = mapped_column(String(16), index=True)
    rule_version: Mapped[str] = mapped_column(String(16))
    severity: Mapped[str] = mapped_column(String(16), index=True)
    condition: Mapped[Optional[str]] = mapped_column(String(64))
    title_ar: Mapped[str] = mapped_column(Text)
    detail_ar: Mapped[str] = mapped_column(Text)

    missing_tests: Mapped[Optional[list[Any]]] = mapped_column(JSON)
    evidence: Mapped[Optional[list[Any]]] = mapped_column(JSON)
    inference_chain: Mapped[Optional[list[Any]]] = mapped_column(JSON)
    references: Mapped[Optional[list[Any]]] = mapped_column(JSON)
    # Recomputing the KBS after a clinician correction must not destroy the audit trail.
    # Suggestions that no longer apply are retained but hidden from active clinical views.
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class SuggestionFeedback(Base):
    """Turns rule quality from an opinion into a measured number (P7-02/03)."""

    __tablename__ = "suggestion_feedback"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    suggestion_id: Mapped[str] = mapped_column(ForeignKey("suggestions.id"), index=True)
    actor: Mapped[Optional[str]] = mapped_column(String(100))
    action: Mapped[str] = mapped_column(String(16))  # accepted|rejected|deferred|acted
    reason: Mapped[Optional[str]] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class Correction(Base):
    """Doctor edits — the active-learning corpus. The original value is kept, so a
    correction is an append, never an overwrite (principle P1)."""

    __tablename__ = "corrections"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    item_id: Mapped[str] = mapped_column(ForeignKey("report_items.item_id"), index=True)
    actor: Mapped[Optional[str]] = mapped_column(String(100))
    field: Mapped[str] = mapped_column(String(32))  # label|text|soap_section|speaker
    old_value: Mapped[Optional[str]] = mapped_column(Text)
    new_value: Mapped[Optional[str]] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    exported_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))


class ModelRegistry(Base):
    """Which model is live, with what metrics — activation and rollback are a flag
    flip here, never a code change (P4-11 / P7-09)."""

    __tablename__ = "model_registry"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    kind: Mapped[str] = mapped_column(String(32), index=True)  # classifier|ner|calibration|voiceprint
    version: Mapped[str] = mapped_column(String(64))
    path: Mapped[Optional[str]] = mapped_column(String(255))
    preprocessing: Mapped[Optional[str]] = mapped_column(String(16))
    max_len: Mapped[Optional[int]] = mapped_column(Integer)
    metrics: Mapped[Optional[dict[str, Any]]] = mapped_column(JSON)
    trained_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    is_active: Mapped[bool] = mapped_column(Boolean, default=False)

    __table_args__ = (UniqueConstraint("kind", "version", name="uq_model_kind_version"),)


class AuditLog(Base):
    __tablename__ = "audit_log"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    actor: Mapped[Optional[str]] = mapped_column(String(100))
    action: Mapped[str] = mapped_column(String(64), index=True)
    resource: Mapped[Optional[str]] = mapped_column(String(255))
    at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)
    meta: Mapped[Optional[dict[str, Any]]] = mapped_column(JSON)
