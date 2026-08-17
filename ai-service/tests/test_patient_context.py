"""Obstetric context + state machine — the regression suite for gap ك-٤.

Every failing case documented in the analysis report is pinned here:
  * "ليست حامل" read as pregnant
  * negation appearing AFTER the term
  * a pregnancy from years ago still being "true" today
  * a general dietary sentence proving pregnancy
"""
from datetime import datetime, timedelta, timezone

import pytest

from app.core.nlp.assertion import find_negation_aware, is_negated
from app.core.nlp.context import ContextObservation, detect_context, extract_ga_weeks
from app.db import repo, session_scope
from app.db.patient_state import current_state, record_observation, state_history


# --- negation ---------------------------------------------------------------------

def test_negation_before_the_term():
    text = "لا يوجد نزيف مهبلي"
    assert is_negated(text, text.index("نزيف"))


def test_negation_after_the_term_is_caught():
    """The measured failure: a prefix-only window scores this as a present symptom."""
    text = "صداع لا يوجد عند المريضة"
    assert is_negated(text, text.index("صداع"), text.index("صداع") + 4)


def test_plain_statement_is_not_negated():
    text = "تشتكي من صداع شديد"
    assert not is_negated(text, text.index("صداع"))


def test_termination_marker_stops_the_negation_scope():
    """In 'لا يوجد نزيف لكن يوجد صداع' the صداع is NOT negated."""
    text = "لا يوجد نزيف لكن يوجد صداع"
    assert is_negated(text, text.index("نزيف"))
    assert not is_negated(text, text.index("صداع"))


def test_pseudo_negation_does_not_negate():
    text = "لا شك أن لديها صداع"
    assert not is_negated(text, text.index("صداع"))


def test_one_negated_mention_does_not_cancel_an_affirmed_one():
    found, asserted = find_negation_aware("ما عندها صداع، عندها صداع نصفي شديد", ("صداع",))
    assert found and asserted


# --- gestational age ----------------------------------------------------------------

@pytest.mark.parametrize(
    "text,weeks",
    [
        ("المريضة حامل بالأسبوع 34", 34),
        ("حامل بالاسبوع ٢٨", 28),          # Arabic-Indic digits
        ("المريضة حامل بالأسبوع الـ 28", 28),  # Arabic tatweel + spaced article
        ("المريضة بالشهر السابع", 30),
        ("حامل بالأسبوع 99", None),        # out of range -> not a gestational age
    ],
)
def test_gestational_age_extraction(text, weeks):
    assert extract_ga_weeks(text) == weeks


# --- context detection ---------------------------------------------------------------

def test_explicit_denial_is_not_read_as_pregnancy():
    """ك-٤, case 1: this used to produce pregnant=True."""
    observation = detect_context(["المريضة ليست حامل حالياً"])
    assert observation.pregnant is False


def test_pregnancy_is_detected_with_gestational_age():
    observation = detect_context(["المريضة حامل بالأسبوع 34 وتشتكي من صداع"])
    assert observation.pregnant is True
    assert observation.ga_weeks == 34 and observation.trimester == 3


def test_silence_leaves_context_unknown():
    """No mention must not be read as a denial — it leaves the previous state standing."""
    observation = detect_context(["تشتكي من صداع", "أعطيناها مسكن"])
    assert observation.pregnant is None and not observation.has_signal


def test_general_nutrition_advice_does_not_imply_pregnancy():
    """ك-٤, case 2: a dietary tip used to prove pregnancy via its classifier label."""
    observation = detect_context([
        "الشوكولاتة الداكنة بكميات قليلة مفيدة للقلب",
        "الكرز يساعد على تخفيف الالتهابات",
    ])
    assert observation.pregnant is None


def test_negation_in_one_sentence_does_not_leak_into_the_next():
    observation = detect_context(["لا يوجد نزيف", "المريضة حامل بالأسبوع 12"])
    assert observation.pregnant is True


def test_postpartum_is_detected():
    observation = detect_context(["المريضة بعد الولادة وتشتكي من حرارة"])
    assert observation.postpartum is True


# --- the state machine ----------------------------------------------------------------

def _patient(session, mrn="MRN-1"):
    return repo.get_or_create_patient(session, mrn=mrn).id


def test_pregnancy_recorded_then_read_back(temp_env):
    now = datetime.now(timezone.utc)
    with session_scope() as session:
        pid = _patient(session)
        record_observation(
            session, patient_id=pid,
            observation=ContextObservation(pregnant=True, ga_weeks=30),
            effective_at=now,
        )
    with session_scope() as session:
        state = current_state(session, pid, at=now)
        assert state.is_pregnant is True
        assert state.ga_weeks_at_observation == 30
        assert state.edd is not None


def test_pregnancy_expires_instead_of_lasting_forever(temp_env):
    """ك-٤, case 3: pregnant in 2023, a visit in 2026 must NOT be pregnant."""
    conceived = datetime(2023, 1, 1, tzinfo=timezone.utc)
    with session_scope() as session:
        pid = _patient(session)
        record_observation(
            session, patient_id=pid,
            observation=ContextObservation(pregnant=True, ga_weeks=30),
            effective_at=conceived,
        )
    with session_scope() as session:
        during = current_state(session, pid, at=conceived + timedelta(days=30))
        assert during.is_pregnant is True

        later = current_state(session, pid, at=datetime(2026, 7, 1, tzinfo=timezone.utc))
        assert later.is_pregnant is False
        assert later.status == "unknown" and later.is_expired
        assert "غير معروفة" in (later.note or "")


def test_gestational_age_is_projected_forward(temp_env):
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    with session_scope() as session:
        pid = _patient(session)
        record_observation(
            session, patient_id=pid,
            observation=ContextObservation(pregnant=True, ga_weeks=20),
            effective_at=start,
        )
    with session_scope() as session:
        state = current_state(session, pid, at=start + timedelta(weeks=6))
        assert state.ga_weeks_at(start + timedelta(weeks=6)) == 26


def test_postpartum_supersedes_pregnancy_then_expires(temp_env):
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    with session_scope() as session:
        pid = _patient(session)
        record_observation(session, patient_id=pid,
                           observation=ContextObservation(pregnant=True, ga_weeks=39),
                           effective_at=start)
        record_observation(session, patient_id=pid,
                           observation=ContextObservation(postpartum=True),
                           effective_at=start + timedelta(days=7))
    with session_scope() as session:
        soon = current_state(session, pid, at=start + timedelta(days=14))
        assert soon.status == "postpartum" and not soon.is_pregnant

        later = current_state(session, pid, at=start + timedelta(days=120))
        assert later.status == "unknown"


def test_denial_resets_the_state(temp_env):
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    with session_scope() as session:
        pid = _patient(session)
        record_observation(session, patient_id=pid,
                           observation=ContextObservation(pregnant=True, ga_weeks=10),
                           effective_at=start)
        record_observation(session, patient_id=pid,
                           observation=ContextObservation(pregnant=False),
                           effective_at=start + timedelta(days=30))
    with session_scope() as session:
        state = current_state(session, pid, at=start + timedelta(days=31))
        assert state.status == "unknown" and not state.is_pregnant


def test_silent_visit_records_nothing_and_keeps_the_previous_state(temp_env):
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    with session_scope() as session:
        pid = _patient(session)
        record_observation(session, patient_id=pid,
                           observation=ContextObservation(pregnant=True, ga_weeks=12),
                           effective_at=start)
        added = record_observation(session, patient_id=pid,
                                   observation=ContextObservation(),
                                   effective_at=start + timedelta(days=20))
        assert added is None
    with session_scope() as session:
        assert len(state_history(session, pid)) == 1
        assert current_state(session, pid, at=start + timedelta(days=21)).is_pregnant


def test_unknown_patient_has_no_state(temp_env):
    with session_scope() as session:
        pid = _patient(session)
        assert current_state(session, pid).status == "unknown"
        assert current_state(session, pid).is_pregnant is False


def test_state_history_is_append_only(temp_env):
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    with session_scope() as session:
        pid = _patient(session)
        for i, weeks in enumerate((10, 20, 30)):
            record_observation(session, patient_id=pid,
                               observation=ContextObservation(pregnant=True, ga_weeks=weeks),
                               effective_at=start + timedelta(weeks=10 * i))
    with session_scope() as session:
        history = state_history(session, pid)
        assert [row.ga_weeks for row in history] == [10, 20, 30]  # nothing overwritten


def _context_report(text: str, created_at: datetime):
    from app.core.report.builder import build_report
    from app.core.report.schema import AudioMeta, ClassifiedSegment

    report = build_report(
        "context-test-job",
        [ClassifiedSegment(
            order_index=0, text=text, start_sec=0.0, end_sec=1.0,
            source_segment_index=0, label="info", confidence=0.99,
        )],
        audio_meta=AudioMeta(filename="context.wav", duration_sec=1.0),
    )
    report.created_at = created_at
    return report


def test_same_visit_explicit_state_overrides_conflicting_transcript():
    from app.core.kbs.service import resolve_effective_context
    from app.db.patient_state import ResolvedState

    moment = datetime(2026, 8, 14, 10, 0, tzinfo=timezone.utc)
    report = _context_report("المريضة حامل بالأسبوع 30", moment)
    explicit = ResolvedState(
        status="postpartum", effective_at=moment, source="explicit",
    )

    _observation, context = resolve_effective_context(explicit, report)
    assert context["postpartum"] is True
    assert context["pregnant"] is False
    assert context["effective_obstetric_status"] == "postpartum"


def test_old_explicit_state_does_not_freeze_a_later_visit():
    from app.core.kbs.service import resolve_effective_context
    from app.db.patient_state import ResolvedState

    old = datetime(2026, 1, 1, 10, 0, tzinfo=timezone.utc)
    later = datetime(2026, 8, 14, 10, 0, tzinfo=timezone.utc)
    report = _context_report("المريضة بعد الولادة وتحتاج متابعة النفاس", later)
    explicit_old = ResolvedState(
        status="pregnant", effective_at=old, ga_weeks_at_observation=20,
        source="explicit",
    )

    _observation, context = resolve_effective_context(explicit_old, report)
    assert context["postpartum"] is True
    assert context["pregnant"] is False
    assert context["effective_obstetric_status"] == "postpartum"
