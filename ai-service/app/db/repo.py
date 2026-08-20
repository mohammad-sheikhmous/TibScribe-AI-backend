"""Repositories: the only place that turns rows into domain objects and back.

Keeping the mapping here (rather than in endpoints) is what lets `JobStore` stay the
same small interface the rest of the app already uses, while its storage moves from a
dict to a database.

`save_report` / `load_report` are exact inverses — `tests/test_db_repo.py` pins the
round-trip, which is what makes it safe to have no cached JSON copy of the report.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional, Sequence

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from ..core.nlp.sections import SOAP_ORDER, SOAP_TITLES_AR, label_ar, soap_for_labels
from ..core.report.formatter import ClinicalSoapFormatter
from ..core.report.schema import (
    AudioMeta,
    PipelineMeta,
    Report,
    ReportItem,
    ReportSection,
    ReportSummary,
)
from .models import (
    AudioFile,
    Correction,
    Entity as EntityRow,
    Job,
    ModelRegistry,
    Patient,
    ReportItemRow,
    ReportRecord,
    Suggestion as SuggestionRow,
    SuggestionFeedback,
    Transcript,
    Visit,
)


def _aware(value: Optional[datetime]) -> Optional[datetime]:
    """SQLite drops tzinfo; every timestamp we store is UTC, so put it back."""
    if value is None:
        return None
    return value if value.tzinfo else value.replace(tzinfo=timezone.utc)

def _numeric_entity_value(value):
    if value is None:
        return None

    if isinstance(value, bool):
        return None

    return float(value)

# --- audio ---------------------------------------------------------------------------

def get_or_create_audio(
    session: Session,
    *,
    sha256: str,
    storage_key: str,
    original_filename: Optional[str],
    mime: Optional[str],
    size_bytes: int,
) -> AudioFile:
    """De-duplicated by content hash: the same recording is archived once."""
    existing = session.scalar(select(AudioFile).where(AudioFile.sha256 == sha256))
    if existing:
        return existing
    audio = AudioFile(
        sha256=sha256,
        storage_key=storage_key,
        original_filename=original_filename,
        mime=mime,
        size_bytes=size_bytes,
    )
    session.add(audio)
    session.flush()
    return audio


def get_audio(session: Session, audio_id: str) -> Optional[AudioFile]:
    return session.get(AudioFile, audio_id)


def audio_for_job(session: Session, job_id: str) -> Optional[AudioFile]:
    job = session.get(Job, job_id)
    return job.audio if job else None


def audio_stats(session: Session) -> dict:
    total, count, oldest = session.execute(
        select(func.sum(AudioFile.size_bytes), func.count(AudioFile.id),
               func.min(AudioFile.uploaded_at))
    ).one()
    unverified = session.scalar(
        select(func.count(AudioFile.id)).where(AudioFile.checksum_verified_at.is_(None))
    )
    return {
        "objects": int(count or 0),
        "total_bytes": int(total or 0),
        "oldest_upload": _aware(oldest),
        "never_verified": int(unverified or 0),
    }


# --- patients / visits ----------------------------------------------------------------

def get_or_create_patient(
    session: Session, *, mrn: Optional[str], display_name: Optional[str] = None
) -> Optional[Patient]:
    if not mrn:
        return None
    patient = session.scalar(select(Patient).where(Patient.mrn == mrn))
    if patient is None:
        patient = Patient(mrn=mrn, display_name=display_name)
        session.add(patient)
        session.flush()
    return patient


def get_or_create_external_patient(
    session: Session, *, source: str, external_id: str
) -> Patient:
    patient = session.scalar(
        select(Patient).where(
            Patient.external_source == source, Patient.external_id == external_id
        )
    )
    if patient is None:
        patient = Patient(external_source=source, external_id=external_id)
        session.add(patient)
        session.flush()
    return patient


def get_patient_by_external(
    session: Session, *, source: str, external_id: str
) -> Optional[Patient]:
    return session.scalar(
        select(Patient).where(
            Patient.external_source == source, Patient.external_id == external_id
        )
    )


def create_visit(
    session: Session, *, patient_id: str, visit_at: Optional[datetime] = None
) -> Visit:
    visit = Visit(patient_id=patient_id, visit_at=visit_at or datetime.now(timezone.utc))
    session.add(visit)
    session.flush()
    return visit


def get_patient(session: Session, patient_id: str) -> Optional[Patient]:
    return session.get(Patient, patient_id)


def get_visit(session: Session, visit_id: str) -> Optional[Visit]:
    return session.get(Visit, visit_id)


def create_patient(
    session: Session,
    *,
    mrn: Optional[str] = None,
    display_name: Optional[str] = None,
    birth_date=None,
    notes: Optional[str] = None,
) -> Patient:
    patient = Patient(mrn=mrn, display_name=display_name, birth_date=birth_date, notes=notes)
    session.add(patient)
    session.flush()
    return patient


def list_patients(
    session: Session, *, q: Optional[str] = None, limit: int = 50, offset: int = 0
) -> tuple[Sequence[Patient], int]:
    stmt = select(Patient)
    count_stmt = select(func.count(Patient.id))
    if q:
        pattern = f"%{q}%"
        condition = Patient.mrn.like(pattern) | Patient.display_name.like(pattern)
        stmt = stmt.where(condition)
        count_stmt = count_stmt.where(condition)
    total = int(session.scalar(count_stmt) or 0)
    rows = session.scalars(
        stmt.order_by(Patient.created_at.desc()).limit(limit).offset(offset)
    ).all()
    return rows, total


def patient_timeline(session: Session, patient_id: str) -> list[dict]:
    """One entry per job ordered by *clinical visit time*, newest first.

    Upload/processing time is kept separately. This prevents a late upload of an old
    consultation from appearing newer than a genuinely later clinical visit.
    """
    rows = session.execute(
        select(Job, Visit.visit_at)
        .outerjoin(Visit, Visit.id == Job.visit_id)
        .where(Job.patient_id == patient_id)
    ).all()

    entries: list[dict] = []
    for job, visit_at in rows:
        record = session.scalar(select(ReportRecord).where(ReportRecord.job_id == job.id))
        summary = (record.summary if record else None) or {}
        labels = summary.get("per_label_counts") or {}
        clinical_at = _aware(visit_at) or (_aware(record.created_at) if record else None) or _aware(job.created_at)
        entries.append(
            {
                "job_id": job.id,
                "external_session_id": job.external_session_id,
                "status": job.status,
                # Backward-compatible name: historically the UI read `created_at`.
                # It now correctly means clinical visit time, not upload time.
                "created_at": clinical_at,
                "visit_at": clinical_at,
                "processing_created_at": _aware(job.created_at),
                "visit_id": job.visit_id,
                "filename": job.audio.original_filename if job.audio else None,
                "duration_sec": (record.audio or {}).get("duration_sec") if record else None,
                "total_segments": summary.get("total_segments", 0),
                "avg_confidence": summary.get("avg_confidence"),
                "urgent_count": summary.get("urgent_count", 0),
                "low_confidence_count": summary.get("low_confidence_count", 0),
                "top_labels": sorted(labels.items(), key=lambda kv: -kv[1])[:5],
            }
        )
    entries.sort(key=lambda row: row["visit_at"] or datetime.min.replace(tzinfo=timezone.utc), reverse=True)
    return entries


# --- jobs -----------------------------------------------------------------------------

def create_job(
    session: Session,
    *,
    job_id: str,
    audio_id: Optional[str],
    patient_id: Optional[str] = None,
    visit_id: Optional[str] = None,
    external_session_id: Optional[str] = None,
    external_doctor_id: Optional[str] = None,
    status: str = "queued",
) -> Job:
    job = Job(
        id=job_id, audio_id=audio_id, patient_id=patient_id,
        visit_id=visit_id, external_session_id=external_session_id,
        external_doctor_id=external_doctor_id, status=status,
    )
    session.add(job)
    session.flush()
    return job


def get_job(session: Session, job_id: str) -> Optional[Job]:
    return session.get(Job, job_id)


def get_job_by_external_session(session: Session, external_session_id: str) -> Optional[Job]:
    return session.scalar(
        select(Job).where(Job.external_session_id == external_session_id)
    )


def update_job(
    session: Session,
    job_id: str,
    *,
    status: Optional[str] = None,
    stage: Optional[str] = None,
    error: Optional[str] = None,
    duration_ms: Optional[int] = None,
    rules_version: Optional[str] = None,
    asr_model: Optional[str] = None,
    classifier_version: Optional[str] = None,
) -> Optional[Job]:
    job = session.get(Job, job_id)
    if job is None:
        return None
    if status is not None:
        job.status = status
    if stage is not None or status in {"complete", "failed"}:
        job.stage = stage
    if error is not None:
        job.error = error
    if duration_ms is not None:
        job.duration_ms = duration_ms
    if rules_version is not None:
        job.rules_version = rules_version
    if asr_model is not None:
        job.asr_model = asr_model
    if classifier_version is not None:
        job.classifier_version = classifier_version
    job.updated_at = datetime.now(timezone.utc)
    return job


def list_jobs(
    session: Session,
    *,
    patient_id: Optional[str] = None,
    status: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
) -> tuple[Sequence[Job], int]:
    stmt = select(Job)
    count_stmt = select(func.count(Job.id))
    if patient_id:
        stmt = stmt.where(Job.patient_id == patient_id)
        count_stmt = count_stmt.where(Job.patient_id == patient_id)
    if status:
        stmt = stmt.where(Job.status == status)
        count_stmt = count_stmt.where(Job.status == status)
    total = int(session.scalar(count_stmt) or 0)
    jobs = session.scalars(
        stmt.order_by(Job.created_at.desc()).limit(limit).offset(offset)
    ).all()
    return jobs, total


def count_active_jobs(session: Session) -> int:
    return int(
        session.scalar(
            select(func.count(Job.id)).where(Job.status.in_(("queued", "running")))
        )
        or 0
    )


def active_job_ids(session: Session) -> list[str]:
    """Durable work left behind by an unclean service restart."""
    return list(
        session.scalars(
            select(Job.id)
            .where(Job.status.in_(("queued", "running")))
            .order_by(Job.created_at.asc())
        ).all()
    )


# --- transcripts -----------------------------------------------------------------------

def save_transcript(
    session: Session,
    *,
    job_id: str,
    language: Optional[str],
    raw_json: Optional[str],
    avg_logprob: Optional[float] = None,
    no_speech_ratio: Optional[float] = None,
) -> Transcript:
    """Persist Whisper's raw output and quality summary (idempotent)."""
    existing = session.scalar(select(Transcript).where(Transcript.job_id == job_id))
    if existing:
        existing.language = language
        existing.raw_json = raw_json
        existing.avg_logprob = avg_logprob
        existing.no_speech_ratio = no_speech_ratio
        return existing
    transcript = Transcript(
        job_id=job_id, language=language, raw_json=raw_json,
        avg_logprob=avg_logprob, no_speech_ratio=no_speech_ratio,
    )
    session.add(transcript)
    session.flush()
    return transcript


# --- reports ---------------------------------------------------------------------------

def save_report(session: Session, report: Report) -> ReportRecord:
    """Persist a Report + normalized entity rows (idempotent).

    SQLite enforces ``entities.item_id -> report_items.item_id`` immediately.
    SQLAlchemy is free to batch INSERTs for unrelated ORM objects unless an explicit
    relationship orders them, so adding ReportItemRow and EntityRow objects in one
    loop can let an entity INSERT race ahead of its parent.  Persist all parent items
    first, flush them, then persist entity children.
    """
    # Entity rows reference report_items, so children must be removed first on rerun.
    session.query(EntityRow).filter(EntityRow.job_id == report.job_id).delete()
    session.query(ReportItemRow).filter(ReportItemRow.job_id == report.job_id).delete()

    record = session.scalar(select(ReportRecord).where(ReportRecord.job_id == report.job_id))
    if record is None:
        record = ReportRecord(job_id=report.job_id)
        session.add(record)

    record.schema_version = report.schema_version
    record.created_at = report.created_at
    record.audio = report.audio.model_dump(mode="json")
    record.pipeline_meta = report.pipeline_meta.model_dump(mode="json")
    record.patient_info = report.patient_info or {}
    record.summary = report.summary.model_dump(mode="json")

    job = session.get(Job, report.job_id)
    patient_id = job.patient_id if job else None
    all_items = [item for section in report.soap.values() for item in section.items]

    # PASS 1: every FK parent exists physically before any EntityRow INSERT.
    for item in all_items:
        session.add(
            ReportItemRow(
                item_id=item.item_id,
                job_id=report.job_id,
                order_index=item.order_index,
                text=item.text,
                text_raw=item.text_raw or item.text,
                text_canonical=item.text_canonical,
                canonicalization_status=item.canonicalization_status,
                canonicalization_confidence=item.canonicalization_confidence,
                canonicalization_model=item.canonicalization_model,
                canonicalization_reasons=item.canonicalization_reasons,
                text_rephrased=item.text_rephrased,
                label=item.label,
                labels=item.labels,
                label_ar=item.label_ar,
                soap_section=item.soap_section,
                also_in_sections=item.also_in_sections,
                confidence=item.confidence,
                combined_confidence=item.combined_confidence,
                entropy=item.entropy,
                ood_score=item.ood_score,
                review_priority=item.review_priority,
                low_confidence_reasons=item.low_confidence_reasons,
                speaker=item.speaker,
                speaker_confidence=item.speaker_confidence,
                asr_confidence=item.asr_confidence,
                is_asr_suspect=item.is_asr_suspect,
                start_sec=item.start_sec,
                end_sec=item.end_sec,
                timestamp_precision=item.timestamp_precision,
                source_segment_index=item.source_segment_index,
                is_urgent=item.is_urgent,
                is_low_confidence=item.is_low_confidence,
                entity_links=item.entity_links,
            )
        )
    session.flush()

    # PASS 2: normalized children.
    for item in all_items:
        for link in item.entity_links or []:
            if not isinstance(link, dict) or not link.get("code"):
                continue
            session.add(
                EntityRow(
                    item_id=item.item_id,
                    job_id=report.job_id,
                    patient_id=patient_id,
                    kind=str(link.get("kind", "clinical")),
                    code=str(link.get("code")),
                    assertion=str(link.get("assertion", "present")),
                    value=_numeric_entity_value(link.get("value")),
                    value2=_numeric_entity_value(link.get("value2")),
                    unit=link.get("unit"),
                    status=link.get("status"),
                    char_start=link.get("char_start"),
                    char_end=link.get("char_end"),
                    extractor=str(link.get("extractor", "lexicon")),
                    extractor_version=link.get("extractor_version"),
                    confidence=float(link.get("confidence", 1.0)),
                )
            )
    session.flush()
    return record

def load_report(session: Session, job_id: str) -> Optional[Report]:
    """Rebuild the exact Report from its rows (inverse of save_report)."""
    record = session.scalar(select(ReportRecord).where(ReportRecord.job_id == job_id))
    if record is None:
        return None

    rows = session.scalars(
        select(ReportItemRow)
        .where(ReportItemRow.job_id == job_id)
        .order_by(ReportItemRow.order_index)
    ).all()
    items = [
        ReportItem(
            item_id=row.item_id,
            order_index=row.order_index,
            text=row.text,
            text_raw=row.text_raw or row.text,
            text_canonical=row.text_canonical,
            canonicalization_status=row.canonicalization_status or "not_run",
            canonicalization_confidence=row.canonicalization_confidence,
            canonicalization_model=row.canonicalization_model,
            canonicalization_reasons=row.canonicalization_reasons or [],
            text_rephrased=row.text_rephrased,
            label=row.label,
            labels=row.labels or [row.label],
            label_ar=row.label_ar or row.label,
            soap_section=row.soap_section,  # type: ignore[arg-type]
            also_in_sections=row.also_in_sections or [],
            confidence=row.confidence,
            combined_confidence=row.combined_confidence,
            entropy=row.entropy,
            ood_score=row.ood_score,
            review_priority=row.review_priority,
            low_confidence_reasons=row.low_confidence_reasons or [],
            speaker=row.speaker,
            speaker_confidence=row.speaker_confidence,
            asr_confidence=row.asr_confidence,
            is_asr_suspect=row.is_asr_suspect,
            is_low_confidence=row.is_low_confidence,
            is_urgent=row.is_urgent,
            start_sec=row.start_sec,
            end_sec=row.end_sec,
            timestamp_precision=row.timestamp_precision,  # type: ignore[arg-type]
            source_segment_index=row.source_segment_index,
            entity_links=row.entity_links,
        )
        for row in rows
    ]

    soap = {
        key: ReportSection(
            soap_key=key,  # type: ignore[arg-type]
            title_ar=SOAP_TITLES_AR[key],
            items=[it for it in items if it.soap_section == key],
        )
        for key in SOAP_ORDER
    }
    # Formatted SOAP is deliberately derived rather than persisted.  That keeps a
    # clinician text correction and the doctor-facing paragraph in one source of truth.
    soap_formatted = ClinicalSoapFormatter().format(soap, patient_info=record.patient_info or {})
    return Report(
        schema_version=record.schema_version,
        job_id=job_id,
        created_at=_aware(record.created_at),  # type: ignore[arg-type]
        audio=AudioMeta(**(record.audio or {})),
        pipeline_meta=PipelineMeta(**(record.pipeline_meta or {})),
        patient_info=record.patient_info or {},
        soap=soap,
        soap_formatted=soap_formatted,
        summary=ReportSummary(**(record.summary or {})),
    )


# --- KBS suggestions / feedback ---------------------------------------------------------

VALID_FEEDBACK_ACTIONS = {"accepted", "rejected", "deferred", "acted"}


def save_suggestions(
    session: Session,
    *,
    job_id: str,
    patient_id: Optional[str],
    suggestions: list[dict],
    scope: str = "visit",
) -> list[SuggestionRow]:
    """Synchronize active KBS suggestions without destroying audit/feedback history.

    Every recomputation first retires the previous active set. Rules that still match
    are reactivated in place (preserving their suggestion id and feedback); rules that
    no longer match remain stored with ``is_active=False`` for auditability.
    """
    existing = session.scalars(
        select(SuggestionRow).where(SuggestionRow.job_id == job_id)
    ).all()
    for previous in existing:
        previous.is_active = False
    by_key = {
        (row.rule_id, row.rule_version, row.scope, row.condition): row
        for row in existing
    }
    rows: list[SuggestionRow] = []
    for raw in suggestions:
        rule_id = str(raw.get("rule_id", ""))
        row_scope = "series" if rule_id in {"R11", "R12"} else scope
        key = (
            rule_id,
            str(raw.get("rule_version", "")),
            row_scope,
            raw.get("condition"),
        )
        row = by_key.get(key)
        if row is None:
            row = SuggestionRow(
                job_id=job_id, patient_id=patient_id, scope=row_scope,
                rule_id=key[0], rule_version=key[1],
                severity=str(raw.get("severity", "low")),
                condition=raw.get("condition"),
                title_ar=str(raw.get("title_ar", "")),
                detail_ar=str(raw.get("detail_ar", "")),
            )
            session.add(row)
        row.patient_id = patient_id
        row.scope = row_scope
        row.is_active = True
        row.severity = str(raw.get("severity", "low"))
        row.title_ar = str(raw.get("title_ar", ""))
        row.detail_ar = str(raw.get("detail_ar", ""))
        row.missing_tests = list(raw.get("missing_tests") or [])
        row.evidence = list(raw.get("evidence") or [])
        row.inference_chain = list(raw.get("inference_chain") or [])
        row.references = list(raw.get("references") or [])
        rows.append(row)
    session.flush()
    return rows


def suggestions_for_job(session: Session, job_id: str) -> list[SuggestionRow]:
    severity_rank = {"critical": 0, "high": 1, "medium": 2, "low": 3}
    rows = list(session.scalars(
        select(SuggestionRow).where(
            SuggestionRow.job_id == job_id, SuggestionRow.is_active.is_(True)
        )
    ).all())
    return sorted(rows, key=lambda r: (severity_rank.get(r.severity, 9), r.rule_id))


def suggestions_for_patient(
    session: Session, patient_id: str, *, limit: int = 200
) -> list[SuggestionRow]:
    return list(session.scalars(
        select(SuggestionRow)
        .where(
            SuggestionRow.patient_id == patient_id, SuggestionRow.is_active.is_(True)
        )
        .order_by(SuggestionRow.created_at.desc())
        .limit(limit)
    ).all())


def get_suggestion(session: Session, suggestion_id: str) -> Optional[SuggestionRow]:
    return session.get(SuggestionRow, suggestion_id)


def add_suggestion_feedback(
    session: Session,
    *,
    suggestion_id: str,
    action: str,
    actor: Optional[str] = None,
    reason: Optional[str] = None,
) -> SuggestionFeedback:
    if action not in VALID_FEEDBACK_ACTIONS:
        raise ValueError(f"Unsupported feedback action: {action}")
    suggestion = session.get(SuggestionRow, suggestion_id)
    if suggestion is None or not suggestion.is_active:
        raise LookupError(suggestion_id)
    row = SuggestionFeedback(
        suggestion_id=suggestion_id, actor=actor, action=action, reason=reason
    )
    session.add(row)
    session.flush()
    return row


def feedback_for_suggestion(session: Session, suggestion_id: str) -> list[SuggestionFeedback]:
    return list(session.scalars(
        select(SuggestionFeedback)
        .where(SuggestionFeedback.suggestion_id == suggestion_id)
        .order_by(SuggestionFeedback.created_at.asc())
    ).all())


def recent_reports_for_patient(
    session: Session,
    patient_id: str,
    *,
    exclude_job_id: Optional[str] = None,
    before: Optional[datetime] = None,
    limit: int = 5,
) -> list[Report]:
    """Most recent completed reports, returned in chronological order."""
    stmt = (
        select(Job)
        .join(ReportRecord, ReportRecord.job_id == Job.id)
        .where(Job.patient_id == patient_id, Job.status == "complete")
        .order_by(ReportRecord.created_at.desc())
        .limit(max(0, limit))
    )
    if exclude_job_id:
        stmt = stmt.where(Job.id != exclude_job_id)
    if before is not None:
        # ReportRecord.created_at is the clinical visit time in the final pipeline.
        # SQLite drops tzinfo; SQLAlchemy still compares the normalized UTC value safely.
        cutoff = before.astimezone(timezone.utc).replace(tzinfo=None) if before.tzinfo else before
        stmt = stmt.where(ReportRecord.created_at < cutoff)
    jobs = session.scalars(stmt).all()
    reports = [load_report(session, job.id) for job in reversed(jobs)]
    return [report for report in reports if report is not None]


# --- model registry ---------------------------------------------------------------------

def get_item(session: Session, job_id: str, item_id: str) -> Optional[ReportItemRow]:
    return session.scalar(
        select(ReportItemRow).where(
            ReportItemRow.item_id == item_id, ReportItemRow.job_id == job_id
        )
    )


def apply_correction(
    session: Session,
    *,
    item: ReportItemRow,
    field: str,
    new_value: str,
    actor: Optional[str] = None,
) -> Correction:
    """Record a doctor's correction — an APPEND, never an overwrite.

    The item row is updated so the report reads correctly from now on, but the previous
    value is preserved in `corrections`. That matters twice over: a clinical record must
    show what was changed and by whom, and the (old -> new) pair IS the training signal
    (P7-07). Storing only the new value would throw away the more informative half.
    """
    old_value = getattr(item, field, None)
    correction = Correction(
        item_id=item.item_id, actor=actor, field=field,
        old_value=None if old_value is None else str(old_value),
        new_value=new_value,
    )
    session.add(correction)

    setattr(item, field, new_value)
    if field == "text":
        # Never erase Whisper evidence.  A clinician correction becomes the trusted
        # canonical/effective text and supersedes any model-generated candidate.
        item.text_canonical = new_value
        item.canonicalization_status = "clinician_corrected"
        item.canonicalization_confidence = 1.0
        item.canonicalization_model = "clinician"
        item.canonicalization_reasons = []
    item.is_corrected = True
    # A corrected label is a human decision: it is certain, and it must stop being
    # flagged for the review it has just had.
    if field == "label":
        item.labels = [new_value]
        item.label_ar = label_ar(new_value)
        item.soap_section, also_in = soap_for_labels([new_value])
        item.also_in_sections = also_in
        item.is_low_confidence = False
        item.low_confidence_reasons = []
        item.review_priority = 0.0
        item.is_urgent = new_value == "emergency"
    session.flush()
    return correction


def refresh_item_entities(session: Session, item: ReportItemRow) -> list[dict]:
    """Re-extract structured entities after a clinician changes text or label.

    Keeping the old ``entity_links`` after a text correction would let the KBS reason
    over facts the clinician just removed. The normalized entity table and JSON link
    field are therefore refreshed together in the same transaction.
    """
    from ..core.nlp.extraction import extract_for_item

    links = extract_for_item({"text": item.text, "label": item.label})
    item.entity_links = links
    session.query(EntityRow).filter(EntityRow.item_id == item.item_id).delete()
    job = session.get(Job, item.job_id)
    patient_id = job.patient_id if job else None
    for link in links:
        if not isinstance(link, dict) or not link.get("code"):
            continue
        session.add(
            EntityRow(
                item_id=item.item_id, job_id=item.job_id, patient_id=patient_id,
                kind=str(link.get("kind", "clinical")), code=str(link.get("code")),
                assertion=str(link.get("assertion", "present")), value=_numeric_entity_value(link.get("value")),
                value2=_numeric_entity_value(link.get("value2")), status=link.get("status"),
                char_start=link.get("char_start"), char_end=link.get("char_end"),
                extractor=str(link.get("extractor", "lexicon")),
                extractor_version=link.get("extractor_version"),
                confidence=float(link.get("confidence", 1.0)),
            )
        )
    session.flush()
    return links


def refresh_report_summary(session: Session, job_id: str) -> Optional[dict]:
    """Keep report summary counts consistent with human-corrected labels."""
    record = session.scalar(select(ReportRecord).where(ReportRecord.job_id == job_id))
    if record is None:
        return None
    rows = list(session.scalars(
        select(ReportItemRow).where(ReportItemRow.job_id == job_id)
    ).all())
    per_label: dict[str, int] = {}
    for row in rows:
        per_label[row.label] = per_label.get(row.label, 0) + 1
    summary = {
        "total_segments": len(rows),
        "avg_confidence": round(sum(r.confidence for r in rows) / len(rows), 4) if rows else 0.0,
        "low_confidence_count": sum(1 for r in rows if r.is_low_confidence),
        "urgent_count": sum(1 for r in rows if r.is_urgent),
        "per_label_counts": per_label,
    }
    record.summary = summary
    session.flush()
    return summary


def update_report_patient_info(session: Session, job_id: str, patient_info: dict) -> None:
    record = session.scalar(select(ReportRecord).where(ReportRecord.job_id == job_id))
    if record is None:
        raise LookupError(job_id)
    record.patient_info = dict(patient_info)
    session.flush()


def retire_suggestions_for_job(session: Session, job_id: str) -> None:
    for row in session.scalars(
        select(SuggestionRow).where(SuggestionRow.job_id == job_id, SuggestionRow.is_active.is_(True))
    ):
        row.is_active = False
    session.flush()


def corrections_for_job(session: Session, job_id: str) -> list[Correction]:
    return list(
        session.scalars(
            select(Correction)
            .join(ReportItemRow, Correction.item_id == ReportItemRow.item_id)
            .where(ReportItemRow.job_id == job_id)
            .order_by(Correction.created_at)
        )
    )


def pending_corrections(session: Session, limit: int = 5000) -> list[Correction]:
    """Corrections not yet exported into a training set (P7-07)."""
    return list(
        session.scalars(
            select(Correction)
            .where(Correction.exported_at.is_(None))
            .order_by(Correction.created_at)
            .limit(limit)
        )
    )


def correction_stats(session: Session) -> dict:
    """How often each predicted label gets corrected — and to what.

    This is the field-truth version of a confusion matrix: it is measured on real
    visits rather than on a held-out split of synthetic data, so it shows which classes
    actually fail in use.
    """
    rows = session.execute(
        select(Correction.old_value, Correction.new_value, func.count(Correction.id))
        .where(Correction.field == "label")
        .group_by(Correction.old_value, Correction.new_value)
    ).all()

    total_items = int(session.scalar(select(func.count(ReportItemRow.item_id))) or 0)
    corrected_items = int(
        session.scalar(
            select(func.count(ReportItemRow.item_id)).where(
                ReportItemRow.is_corrected.is_(True)
            )
        )
        or 0
    )

    confusions = [
        {"predicted": old, "corrected_to": new, "count": int(count)}
        for old, new, count in sorted(rows, key=lambda r: -r[2])
    ]
    per_label: dict[str, int] = {}
    for entry in confusions:
        per_label[entry["predicted"]] = per_label.get(entry["predicted"], 0) + entry["count"]

    return {
        "items_total": total_items,
        "items_corrected": corrected_items,
        "correction_rate": round(corrected_items / total_items, 4) if total_items else 0.0,
        "most_corrected_labels": dict(sorted(per_label.items(), key=lambda kv: -kv[1])[:10]),
        "confusions": confusions[:25],
    }


def review_queue(
    session: Session, *, job_id: Optional[str] = None, limit: int = 20
) -> list[ReportItemRow]:
    """Items a reviewer should read first, most informative first (P7-04).

    Ordered by `review_priority` (uncertainty-driven), not by document order: reading
    the three least-certain sentences buys far more than reading the first three.
    Already-corrected items drop out — they have had their review.
    """
    stmt = select(ReportItemRow).where(ReportItemRow.is_corrected.is_(False))
    if job_id:
        stmt = stmt.where(ReportItemRow.job_id == job_id)
    return list(
        session.scalars(
            stmt.order_by(
                ReportItemRow.review_priority.desc().nulls_last(),
                ReportItemRow.confidence.asc(),
            ).limit(limit)
        )
    )


def active_model(session: Session, kind: str) -> Optional[ModelRegistry]:
    return session.scalar(
        select(ModelRegistry).where(
            ModelRegistry.kind == kind, ModelRegistry.is_active.is_(True)
        )
    )


def register_model(
    session: Session,
    *,
    kind: str,
    version: str,
    path: Optional[str] = None,
    metrics: Optional[dict] = None,
    preprocessing: Optional[str] = None,
    max_len: Optional[int] = None,
    activate: bool = False,
) -> ModelRegistry:
    entry = session.scalar(
        select(ModelRegistry).where(
            ModelRegistry.kind == kind, ModelRegistry.version == version
        )
    )
    if entry is None:
        entry = ModelRegistry(kind=kind, version=version)
        session.add(entry)
    entry.path = path
    entry.metrics = metrics
    entry.preprocessing = preprocessing
    entry.max_len = max_len
    if activate:
        for other in session.scalars(
            select(ModelRegistry).where(ModelRegistry.kind == kind)
        ):
            other.is_active = False
        entry.is_active = True
    session.flush()
    return entry
