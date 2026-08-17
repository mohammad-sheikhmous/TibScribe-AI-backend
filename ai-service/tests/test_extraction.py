"""Shared entity extraction (IMPLEMENTATION.md P5-11 … P5-16).

Pins the three defects the analysis measured, plus the lexicon-as-data contract.
"""
import pytest

from app.core.nlp.extraction import (
    drug_family,
    extract_entities,
    extract_for_item,
    load_lexicon,
)


def codes(entities, kind=None):
    return {e.code for e in entities if kind is None or e.kind == kind}


def by_code(entities, code):
    return next(e for e in entities if e.code == code)


# --- the lexicon is data -------------------------------------------------------------

def test_lexicon_loads_every_category():
    size = load_lexicon().size
    for category in ("symptoms", "conditions", "tests", "labs", "drugs",
                     "procedures", "nutrition"):
        assert size[category] > 0, f"{category} lexicon is empty"


def test_longest_surface_form_wins():
    """'ألم أسفل البطن' must not be shadowed by a bare 'ألم'."""
    entities = extract_entities("تشتكي من ألم أسفل البطن")
    assert "abdominal_pain" in codes(entities)
    assert by_code(entities, "abdominal_pain").matched_text == "ألم أسفل البطن"


# --- assertions reach the entities (gaps ك-٢ / ك-٣) --------------------------------

def test_plain_symptom_is_actionable():
    entity = by_code(extract_entities("تشتكي من صداع شديد"), "headache")
    assert entity.assertion == "present" and entity.is_actionable


def test_denied_symptom_is_marked_absent():
    entity = by_code(extract_entities("لا يوجد نزيف مهبلي"), "vaginal_bleeding")
    assert entity.assertion == "absent" and not entity.is_actionable


def test_safety_netting_advice_is_not_actionable():
    """The sentence that fired a false pre-eclampsia alert in the analysis."""
    entities = extract_entities("إذا صار عندك صداع أو تشوش بالرؤية راجعينا فوراً", "plan")
    assert {"headache", "blurred_vision"} <= codes(entities)
    assert not any(e.is_actionable for e in entities
                   if e.code in {"headache", "blurred_vision"})


def test_family_history_is_not_the_patients_finding():
    entity = by_code(extract_entities("أمها عندها سكري", "history"), "diabetes")
    assert entity.assertion == "family" and not entity.is_actionable


# --- lab values (gap ك-٨) -------------------------------------------------------------

def test_qualifier_does_not_cross_a_clause_boundary():
    """v1 read this as a HIGH haemoglobin because it scanned the whole sentence."""
    entity = by_code(extract_entities("الهيموغلوبين طبيعي لكن الضغط مرتفع", "lab"),
                     "hemoglobin")
    assert entity.status == "normal"


def test_numeric_value_is_graded_against_the_reference_range():
    entity = by_code(extract_entities("الهيموغلوبين 7", "lab"), "hemoglobin")
    assert entity.value == 7.0 and entity.status == "low"
    assert entity.unit == "g/dL"


def test_normal_value_is_graded_normal():
    assert by_code(extract_entities("الهيموغلوبين 12", "lab"), "hemoglobin").status == "normal"


def test_a_reported_lab_implies_its_test_was_done():
    assert "hemoglobin_test" in codes(extract_entities("الهيموغلوبين 11", "lab"), "test")


# --- vitals ---------------------------------------------------------------------------

def test_blood_pressure_is_parsed_with_both_numbers():
    entity = by_code(extract_entities("ضغط الدم 150 على 90", "vital"), "bp")
    assert entity.value == 150.0 and entity.value2 == 90.0 and entity.unit == "mmHg"


def test_levantine_short_form_blood_pressure_is_expanded():
    entity = by_code(extract_entities("الضغط 11/7", "vital"), "bp")
    assert (entity.value, entity.value2) == (110.0, 70.0)


def test_temperature_and_pulse_are_parsed():
    entities = extract_entities("الحرارة 38.5 والنبض 96", "vital")
    assert by_code(entities, "temp").value == 38.5
    assert by_code(entities, "pulse").value == 96.0


# --- drugs (gap ك-٨) ---------------------------------------------------------------------

def test_known_drug_is_recognised_with_its_family():
    entities = extract_entities("أعطيناها أموكسيسيلين", "medication")
    assert "amoxicillin" in codes(entities)
    assert drug_family("amoxicillin") == "penicillin"


def test_whisper_misspellings_still_match():
    """A term list of correct spellings only would miss the real transcript."""
    assert "ibuprofen" in codes(extract_entities("صرفنا بروفن", "medication"))
    assert "amlodipine" in codes(extract_entities("زدنا جرعة أمل وديبين", "medication"))


def test_english_filler_words_are_not_invented_as_drugs():
    """v1 produced a drug called 'medications' from this exact sentence."""
    entities = extract_entities("المريضة allergic to Sulfa medications", "allergy")
    assert "sulfa" in codes(entities)
    assert "unknown_drug" not in codes(entities)


def test_a_genuinely_unknown_drug_is_flagged_not_guessed():
    entities = extract_entities("المريضة تاخذ Rosuvastatin يوميا", "medication")
    unknown = by_code(entities, "unknown_drug")
    assert unknown.confidence < 0.5 and "needs review" in (unknown.note or "")


def test_allergy_and_medication_are_told_apart_by_the_label():
    assert extract_entities("حساسية من البنسلين", "allergy")[0].kind == "allergy"
    assert extract_entities("أعطيناها بنسلين", "medication")[0].kind == "medication"


# --- output shape ------------------------------------------------------------------------

def test_extract_for_item_returns_storable_dicts():
    links = extract_for_item({"text": "ضغط الدم 150 على 90 وتشتكي من صداع", "label": "vital"})
    assert isinstance(links, list) and links
    assert all(isinstance(link, dict) and "code" in link for link in links)
    assert all("assertion" in link for link in links)


def test_empty_text_yields_nothing():
    assert extract_entities("") == []
    assert extract_entities("   ") == []


def test_spoken_blurred_vision_word_order_is_recognised():
    entities = extract_entities("وكمان تقول ان الرؤية عندها مشوشة شوي")
    assert "blurred_vision" in codes(entities, "symptom")


def test_urine_protein_context_does_not_fire_nutrition_protein():
    entities = extract_entities("بين وجود بروتين او زلال بالبول", "diagnosis")
    assert "urine_protein" in codes(entities, "lab")
    assert "protein_rich_food" not in codes(entities, "nutrition")



def test_normal_lab_qualifier_overrides_remote_generic_negation():
    """`ما فيه مشاكل` must not negate a later explicitly normal haemoglobin result."""
    entities = extract_entities(
        "تحليل الدم الأخير ما فيه مشاكل مهمة والهيموغلوبين ضمن المقبول",
        "lab",
    )
    hb = next(e for e in entities if e.kind == "lab" and e.code == "hemoglobin")
    assert hb.assertion == "present"
    assert hb.status == "normal"


def test_scheduled_followup_is_planned_even_when_later_clause_is_conditional():
    entities = extract_entities(
        "الموعد الجاي للمتابعة بعد حوالي 4 أسابيع إذا ما صار أي عرض جديد",
        "follow_up",
    )
    visit = next(e for e in entities if e.kind == "procedure" and e.code == "follow_up_visit")
    assert visit.assertion == "planned"

# --- Scenario 3: colloquial GDM + anaemia hardening (2026-08-16 V5) -----------------

def test_decimal_lab_value_is_not_truncated_by_clause_punctuation():
    hb = by_code(extract_entities("كمان الهيموغلوبين طلع 9.5", "lab"), "hemoglobin")
    assert hb.value == 9.5
    assert hb.status == "low"


def test_explicit_colloquial_gdm_phrase_is_structured_condition():
    entity = by_code(
        extract_entities("هذا الشي بيعني انه هي معها سكر الحمل", "pregnancy_risk"),
        "gdm",
    )
    assert entity.kind == "condition"
    assert entity.assertion == "present"


def test_gestational_glucose_test_can_be_high_without_numeric_value():
    entities = extract_entities(
        "بالنسبة لتحليل اختبار السكر بالحمل طلع عندها يام ارتفاع",
        "lab",
    )
    assert "gestational_glucose_test" in codes(entities, "test")
    screen = by_code(entities, "gestational_glucose_screen")
    assert screen.kind == "lab"
    assert screen.assertion == "present"
    assert screen.status == "high"
    assert screen.value is None


def test_test_name_does_not_itself_diagnose_gdm():
    entities = extract_entities("تحليل اختبار السكر بالحمل طلع مرتفع", "lab")
    assert "gdm" not in codes(entities, "condition")


def test_dietary_sugars_do_not_create_diabetes_diagnosis():
    entities = extract_entities("خاصة انه هي لازم تخفف السكريات", "nutrition")
    assert "diabetes" not in codes(entities, "condition")
    assert "gdm" not in codes(entities, "condition")


def test_whisper_dialect_variant_dyspnea_is_negated_correctly():
    entities = extract_entities("ما عندها لا دوخة ولا ضيق نفاس", "cardiology")
    dyspnea = by_code(entities, "shortness_of_breath")
    dizziness = by_code(entities, "dizziness")
    assert dyspnea.assertion == "absent"
    assert dizziness.assertion == "absent"
