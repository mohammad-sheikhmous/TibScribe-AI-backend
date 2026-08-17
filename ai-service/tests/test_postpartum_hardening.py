"""Scenario 5 postpartum-infection hardening (V7)."""
from app.core.nlp.extraction import extract_entities, extract_for_item
from app.core.nlp.sections import soap_for_item, text_implies_plan
from app.core.report.builder import build_report
from app.core.report.schema import ClassifiedSegment
from EXPERTA_MED.extraction import extract_report


def by_code(entities, code):
    return next(e for e in entities if e.code == code)


def make_seg(i: int, text: str, label: str, confidence: float = 0.9):
    seg = ClassifiedSegment(
        order_index=i,
        text=text,
        start_sec=float(i),
        end_sec=float(i + 1),
        source_segment_index=i,
        label=label,
        confidence=confidence,
        asr_confidence=0.9,
    )
    seg.entity_links = extract_for_item({"text": text, "label": label})
    return seg


def test_postpartum_days_fever_and_lower_abdominal_pain_are_structured():
    entities = extract_entities(
        "المريضة ولدت من 5 أيام ومن يومين صارت بتحس بحرارة وتعب عام وألم بأسفل البطن",
        "symptom",
    )
    assert by_code(entities, "postpartum_days_since_birth").value == 5.0
    assert by_code(entities, "postpartum_hours_since_birth").value == 120.0
    assert by_code(entities, "fever").assertion == "present"
    assert by_code(entities, "lower_abdominal_pain").assertion == "present"


def test_foul_postpartum_discharge_is_not_reduced_to_generic_discharge():
    entities = extract_entities(
        "بتقول كمان أن الإفرازات بعد الولادة ريحتها صارت غير طبيعية",
        "postpartum",
    )
    foul = by_code(entities, "foul_vaginal_discharge")
    assert foul.assertion == "present"
    assert foul.status == "foul_smelling"


def test_qualitative_fast_pulse_is_structured_without_inventing_a_number():
    entity = by_code(
        extract_entities("الحرارة اليوم كانت 38.7 والنبض أسرع شوي من الطبيعي", "vital"),
        "maternal_tachycardia",
    )
    assert entity.value is None
    assert entity.status == "high"
    assert entity.assertion == "present"
    assert entity.status == "high"
    assert entity.unit is None


def test_suspected_postpartum_infection_preserves_uncertainty_and_whisper_variant():
    entity = by_code(
        extract_entities(
            "في ألم بأسفل البطن عند الفحص حسب الأعراض لازم نفكر بعدوة بعد الولادة",
            "diagnosis",
        ),
        "puerperal_infection",
    )
    assert entity.assertion == "present"
    assert entity.status == "suspected"


def test_drug_matching_is_word_aware_tahdid_is_not_iron():
    entities = extract_entities(
        "المريضة بحاجة لتقييم طبي سريع وفحوصات إضافية وتحديد العلاج المناسب حسب النتائج",
        "treatment",
    )
    assert not any(e.kind == "medication" and e.code == "iron" for e in entities)
    assert any(e.kind == "medication" and e.code == "iron" for e in extract_entities("تاخد الحديد", "treatment"))


def test_diagnostic_sentence_is_not_misrouted_to_plan_by_earlier_exam_word():
    text = "في ألم بأسفل البطن عند الفحص حسب الأعراض لازم نفكر بعدوة بعد الولادة"
    links = extract_for_item({"text": text, "label": "diagnosis"})
    assert text_implies_plan(text) is False
    primary, _ = soap_for_item(["diagnosis"], text, links)
    assert primary == "assessment"


def test_patient_reported_postpartum_discharge_routes_subjective():
    text = "بتقول كمان أن الإفرازات بعد الولادة ريحتها صارت غير طبيعية"
    links = extract_for_item({"text": text, "label": "postpartum"})
    primary, also = soap_for_item(["postpartum"], text, links)
    assert primary == "subjective"
    assert "assessment" in also


def test_explicit_fast_medical_evaluation_is_plan_and_urgent():
    seg = make_seg(
        0,
        "والمريضة بحاجة لتقييم طبي سريع وفحوصات إضافية وتحديد العلاج المناسب حسب النتائج",
        "treatment",
    )
    report = build_report("s5-urgent", [seg])
    item = report.soap["plan"].items[0]
    assert item.is_urgent is True
    assert report.summary.urgent_count == 1
    assert not any(link.get("code") == "iron" for link in (item.entity_links or []))


def test_kbs_adapter_gets_postpartum_infection_facts_without_false_iron_or_confirmed_diagnosis():
    raw = [
        (0, "المريضة ولدت من 5 أيام وراجعت اليوم لأنها من يومين صارت بتحس بحرارة وتعب عام وألم بأسفل البطن", "symptom", "subjective"),
        (1, "بتقول كمان أن الإفرازات بعد الولادة ريحتها صارت غير طبيعية", "postpartum", "subjective"),
        (2, "الحرارة اليوم كانت 38.7 والنبض أسرع شوي من الطبيعي", "vital", "objective"),
        (3, "في ألم بأسفل البطن عند الفحص حسب الأعراض لازم نفكر بعدوة بعد الولادة", "diagnosis", "assessment"),
        (4, "والمريضة بحاجة لتقييم طبي سريع وفحوصات إضافية وتحديد العلاج المناسب حسب النتائج", "treatment", "plan"),
    ]
    sections = {"subjective": {"items": []}, "objective": {"items": []}, "assessment": {"items": []}, "plan": {"items": []}}
    for idx, text, label, soap in raw:
        item = {
            "item_id": f"s5-{idx:04d}",
            "order_index": idx,
            "text": text,
            "label": label,
            "soap_section": soap,
            "confidence": 0.9,
            "combined_confidence": 0.85,
            "is_low_confidence": False,
            "is_urgent": idx == 4,
        }
        item["entity_links"] = extract_for_item(item)
        sections[soap]["items"].append(item)

    report = {
        "job_id": "s5-contract",
        "patient_info": {"effective_obstetric_status": "postpartum", "postpartum": True, "pregnant": False},
        "soap": sections,
    }
    ext = extract_report(report)
    assert ext.postpartum is True
    entities = ext.entities
    assert any(e.kind == "symptom" and e.code == "fever" and not e.negated for e in entities)
    assert any(e.kind == "symptom" and e.code == "lower_abdominal_pain" and not e.negated for e in entities)
    assert any(e.kind == "symptom" and e.code == "foul_vaginal_discharge" and not e.negated for e in entities)
    assert any(e.kind == "vital" and e.code == "temp" and e.value == 38.7 for e in entities)
    assert any(e.kind == "clinical" and e.code == "postpartum_hours_since_birth" and e.value == 120.0 for e in entities)
    assert any(e.kind == "clinical" and e.code == "maternal_tachycardia" and e.status == "high" and e.assertion == "present" for e in entities)
    assert any(e.kind == "clinical" and e.code == "suspected_puerperal_infection" for e in entities)
    assert not any(e.kind == "diagnosis" and e.code == "puerperal_infection" for e in entities)
    assert not any(e.kind == "medication" and e.code == "iron" for e in entities)
