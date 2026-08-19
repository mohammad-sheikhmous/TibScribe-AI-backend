from app.core.nlp.canonicalization import (
    CanonicalizationCandidate,
    ClinicalSafetyGuard,
    canonicalize_texts,
)


class MappingCanonicalizer:
    applied = True
    model_name = "fake-dialect-model"

    def __init__(self, mapping):
        self.mapping = mapping

    def canonicalize_batch(self, texts):
        return [CanonicalizationCandidate(self.mapping[text], 0.91) for text in texts]


class BrokenCanonicalizer:
    applied = True
    model_name = "broken"

    def canonicalize_batch(self, texts):
        raise RuntimeError("boom")


def test_dialect_to_msa_candidate_is_accepted_without_a_static_phrase_rule():
    raw = "رجليها شوي متنفخين"
    candidate = "تعاني من تورم بسيط في القدمين"
    result = canonicalize_texts([raw], MappingCanonicalizer({raw: candidate}))[0]
    assert result.status == "accepted"
    assert result.effective_text == candidate
    assert result.raw_text == raw


def test_numbers_are_immutable_across_rewrite():
    raw = "ضغطها 128 على 82"
    candidate = "ضغط الدم 138 على 82"
    result = canonicalize_texts([raw], MappingCanonicalizer({raw: candidate}))[0]
    assert result.status == "rejected"
    assert result.effective_text == raw
    assert "numeric_facts_changed" in result.reasons


def test_negation_and_entity_assertion_must_survive_rewrite():
    guard = ClinicalSafetyGuard()
    safe = guard.validate(
        "ما عندها تشوش بالرؤية",
        "لا تعاني من تشوش في الرؤية",
    )
    unsafe = guard.validate(
        "ما عندها تشوش بالرؤية",
        "تعاني من تشوش في الرؤية",
    )
    assert safe.accepted is True
    assert unsafe.accepted is False
    assert any("negated_scope_changed" in reason or "assertion_changed" in reason
               for reason in unsafe.reasons)


def test_hypothetical_safety_netting_stays_hypothetical():
    guard = ClinicalSafetyGuard()
    safe = guard.validate(
        "اذا صار عندها نزيف تراجع فوراً",
        "في حال حدوث نزيف يجب أن تراجع فوراً",
    )
    unsafe = guard.validate(
        "اذا صار عندها نزيف تراجع فوراً",
        "تعاني من نزيف وتحتاج مراجعة فورية",
    )
    assert safe.accepted is True
    assert unsafe.accepted is False
    assert any("hypothetical_scope_changed" in reason or "assertion_changed" in reason
               for reason in unsafe.reasons)


def test_model_failure_degrades_to_raw_text_instead_of_breaking_job():
    raw = "عندها وجع راس"
    result = canonicalize_texts([raw], BrokenCanonicalizer())[0]
    assert result.status == "failed"
    assert result.effective_text == raw
    assert result.canonical_text is None


def test_spoken_dose_unit_cannot_silently_change():
    guard = ClinicalSafetyGuard()
    decision = guard.validate(
        "وصفت لها حديد 30 ملغ يومياً",
        "وُصف لها الحديد بجرعة 30 غرام يومياً",
    )
    assert decision.accepted is False
    assert any(reason.startswith("clinical_units_changed") for reason in decision.reasons)


def test_lab_status_cannot_flip_high_to_low():
    guard = ClinicalSafetyGuard()
    decision = guard.validate(
        "الهيموغلوبين 9.5 منخفض",
        "الهيموغلوبين 9.5 مرتفع",
    )
    assert decision.accepted is False
    assert "clinical_qualifier_changed" in decision.reasons


def test_unit_matching_does_not_treat_arabic_substrings_as_units():
    guard = ClinicalSafetyGuard()
    decision = guard.validate(
        "اسم المريضة لينا",
        "اسم المريضة هو لينا",
    )
    assert decision.accepted is True


def test_compound_unit_does_not_double_count_gram_suffix():
    guard = ClinicalSafetyGuard()
    decision = guard.validate(
        "الجرعة 10 ميليغرام يومياً",
        "الجرعة 10 ملغ يومياً",
    )
    assert decision.accepted is True


def test_compound_high_qualifier_is_not_misread_as_normal():
    guard = ClinicalSafetyGuard()
    decision = guard.validate(
        "القياس فوق الطبيعي",
        "القياس مرتفع",
    )
    assert decision.accepted is True
