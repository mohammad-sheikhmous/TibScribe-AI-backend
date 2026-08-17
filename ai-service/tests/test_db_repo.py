"""Persistence layer (IMPLEMENTATION.md P1-01…P1-04).

The critical test here is the Report round-trip: because no serialized copy of the
report is kept, `load_report(save_report(x)) == x` is what makes the database a safe
single source of truth.
"""
from datetime import datetime, timezone

import pytest
from sqlalchemy import text

from app.core.report.builder import build_report
from app.core.report.schema import AudioMeta, ClassifiedSegment, PipelineMeta
from app.db import repo, session_scope
from app.db.session import get_engine


def _report(job_id="job1", n=3):
    segs = [
        ClassifiedSegment(
            order_index=i, text=f"جملة رقم {i}", start_sec=float(i), end_sec=float(i) + 1,
            source_segment_index=i,
            label=["symptom", "vital", "emergency"][i % 3],
            confidence=[0.95, 0.42, 0.88][i % 3],
        )
        for i in range(n)
    ]
    return build_report(
        job_id, segs,
        audio_meta=AudioMeta(filename="note.mp3", duration_sec=12.5,
                             detected_language="ar", whisper_model="medium"),
        pipeline_meta=PipelineMeta(classifier_max_len=128, arabert_model_name="arabert"),
        created_at=datetime(2026, 7, 26, 10, 30, tzinfo=timezone.utc),
    )


# --- engine configuration ----------------------------------------------------------

def test_sqlite_runs_in_wal_with_foreign_keys_on(temp_env):
    """Both pragmas are load-bearing: WAL for concurrent read/write, FK for integrity."""
    with get_engine().connect() as conn:
        assert conn.execute(text("PRAGMA journal_mode")).scalar().lower() == "wal"
        assert conn.execute(text("PRAGMA foreign_keys")).scalar() == 1


# --- report round-trip -------------------------------------------------------------

def test_report_survives_a_full_save_load_cycle(temp_env):
    original = _report()
    with session_scope() as session:
        repo.create_job(session, job_id="job1", audio_id=None)
        repo.save_report(session, original)

    with session_scope() as session:
        loaded = repo.load_report(session, "job1")

    assert loaded is not None
    assert loaded.model_dump(mode="json") == original.model_dump(mode="json")


def test_round_trip_preserves_item_order_and_flags(temp_env):
    with session_scope() as session:
        repo.create_job(session, job_id="job1", audio_id=None)
        repo.save_report(session, _report(n=6))
    with session_scope() as session:
        loaded = repo.load_report(session, "job1")

    items = [it for section in loaded.soap.values() for it in section.items]
    assert sorted(it.order_index for it in items) == list(range(6))
    assert any(it.is_urgent for it in items)          # emergency label
    assert any(it.is_low_confidence for it in items)  # confidence 0.42 < 0.5


def test_saving_twice_replaces_items_instead_of_duplicating(temp_env):
    with session_scope() as session:
        repo.create_job(session, job_id="job1", audio_id=None)
        repo.save_report(session, _report(n=3))
        repo.save_report(session, _report(n=2))
    with session_scope() as session:
        loaded = repo.load_report(session, "job1")
    assert sum(len(s.items) for s in loaded.soap.values()) == 2


def test_load_report_returns_none_for_unknown_job(temp_env):
    with session_scope() as session:
        assert repo.load_report(session, "nope") is None


# --- audio de-duplication ----------------------------------------------------------

def test_audio_row_is_reused_for_identical_content(temp_env):
    with session_scope() as session:
        first = repo.get_or_create_audio(
            session, sha256="a" * 64, storage_key="aa/x.wav",
            original_filename="one.wav", mime="audio/wav", size_bytes=10,
        )
        second = repo.get_or_create_audio(
            session, sha256="a" * 64, storage_key="aa/x.wav",
            original_filename="two.wav", mime="audio/wav", size_bytes=10,
        )
        assert first.id == second.id


def test_audio_stats_counts_unverified_objects(temp_env):
    with session_scope() as session:
        repo.get_or_create_audio(session, sha256="b" * 64, storage_key="bb/y.wav",
                                 original_filename="y.wav", mime=None, size_bytes=2048)
        stats = repo.audio_stats(session)
    assert stats["objects"] == 1
    assert stats["total_bytes"] == 2048
    assert stats["never_verified"] == 1


# --- jobs ---------------------------------------------------------------------------

def test_active_count_tracks_queued_and_running(temp_env):
    with session_scope() as session:
        repo.create_job(session, job_id="a", audio_id=None, status="queued")
        repo.create_job(session, job_id="b", audio_id=None, status="running")
        repo.create_job(session, job_id="c", audio_id=None, status="complete")
        assert repo.count_active_jobs(session) == 2


def test_list_jobs_filters_and_paginates(temp_env):
    with session_scope() as session:
        for i in range(5):
            repo.create_job(session, job_id=f"j{i}", audio_id=None,
                            status="complete" if i % 2 else "queued")
    with session_scope() as session:
        done, total = repo.list_jobs(session, status="complete")
        assert total == 2 and len(done) == 2
        page, total_all = repo.list_jobs(session, limit=2, offset=0)
        assert total_all == 5 and len(page) == 2


# --- model registry ------------------------------------------------------------------

def test_activating_a_model_deactivates_the_previous_one(temp_env):
    with session_scope() as session:
        repo.register_model(session, kind="classifier", version="v1", activate=True)
        repo.register_model(session, kind="classifier", version="v2", activate=True)
    with session_scope() as session:
        active = repo.active_model(session, "classifier")
        assert active.version == "v2"


def test_patient_is_created_once_per_mrn(temp_env):
    with session_scope() as session:
        a = repo.get_or_create_patient(session, mrn="MRN-1")
        b = repo.get_or_create_patient(session, mrn="MRN-1")
        assert a.id == b.id
        assert repo.get_or_create_patient(session, mrn=None) is None


# --- clinical chronology ------------------------------------------------------------

def test_recent_history_excludes_future_clinical_visits(temp_env):
    """A historical visit uploaded late must never reason from a future visit."""
    from datetime import timedelta

    t1 = datetime(2026, 1, 10, 10, 0, tzinfo=timezone.utc)
    t2 = datetime(2026, 2, 10, 10, 0, tzinfo=timezone.utc)
    with session_scope() as session:
        patient = repo.create_patient(session, mrn="MRN-chronology")
        repo.create_job(session, job_id="old", audio_id=None, patient_id=patient.id, status="complete")
        old_report = _report("old", n=1)
        old_report.created_at = t1
        repo.save_report(session, old_report)
        repo.create_job(session, job_id="future", audio_id=None, patient_id=patient.id, status="complete")
        future_report = _report("future", n=1)
        future_report.created_at = t2
        repo.save_report(session, future_report)

    with session_scope() as session:
        history = repo.recent_reports_for_patient(
            session, patient.id, exclude_job_id="old", before=t1, limit=5
        )
        assert history == []


def test_patient_timeline_orders_by_visit_time_not_upload_time(temp_env):
    t_old = datetime(2026, 1, 1, 9, 0, tzinfo=timezone.utc)
    t_new = datetime(2026, 3, 1, 9, 0, tzinfo=timezone.utc)
    with session_scope() as session:
        patient = repo.create_patient(session, mrn="MRN-timeline")
        # Create the newer clinical visit first, then upload the old visit later.
        v_new = repo.create_visit(session, patient_id=patient.id, visit_at=t_new)
        repo.create_job(session, job_id="new-clinical", audio_id=None, patient_id=patient.id, visit_id=v_new.id, status="complete")
        v_old = repo.create_visit(session, patient_id=patient.id, visit_at=t_old)
        repo.create_job(session, job_id="old-clinical", audio_id=None, patient_id=patient.id, visit_id=v_old.id, status="complete")

    with session_scope() as session:
        rows = repo.patient_timeline(session, patient.id)
        assert [row["job_id"] for row in rows] == ["new-clinical", "old-clinical"]
        assert rows[0]["visit_at"] == t_new


def test_report_items_are_flushed_before_entity_fk_rows(temp_env):
    with session_scope() as session:
        patient = repo.create_patient(session, mrn="MRN-ENTITY-FK")
        repo.create_job(session, job_id="job-entity-fk", audio_id=None, patient_id=patient.id)
        report = _report("job-entity-fk", n=1)
        item = next(it for section in report.soap.values() for it in section.items)
        item.entity_links = [{
            "kind": "symptom", "code": "headache", "assertion": "present",
            "char_start": 0, "char_end": 4, "extractor": "lexicon",
            "extractor_version": "lexicon-1.0", "confidence": 1.0,
        }]
        repo.save_report(session, report)

    with session_scope() as session:
        count = session.execute(text(
            "SELECT COUNT(*) FROM entities WHERE job_id='job-entity-fk' AND code='headache'"
        )).scalar()
        assert count == 1
