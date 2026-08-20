"""Realistic Arabic OB/GYN hardening scenarios.

These tests intentionally exercise deterministic layers that can be validated offline:
entity extraction, assertion/context, cross-segment discourse and fact-first SOAP
formatting.  They do not pretend to test Whisper/Qwen/AraBERT weights locally.
"""
from __future__ import annotations

import pytest

from app.core.nlp.discourse import enrich_cross_segment_context
from app.core.nlp.extraction import extract_for_item
from app.core.report.builder import build_report
from app.core.report.schema import ClassifiedSegment


def seg(order: int, text: str, label: str, *, start: float | None = None, end: float | None = None):
    s = ClassifiedSegment(
        order_index=order,
        text=text,
        start_sec=float(order if start is None else start),
        end_sec=float(order + 1 if end is None else end),
        source_segment_index=order,
        label=label,
        confidence=0.95,
    )
    s.entity_links = extract_for_item({"text": text, "label": label})
    return s


def code_map(segment: ClassifiedSegment) -> dict[str, dict]:
    return {
        str(link["code"]): link
        for link in (segment.entity_links or [])
        if isinstance(link, dict) and link.get("code")
    }


@pytest.mark.parametrize(
    "text,expected",
    [
        # Levantine / Syrian
        ("لا عندها دوخة ولا ضيق نفس", {"dizziness": "absent", "shortness_of_breath": "absent"}),
        ("ما عندها لا دوخة ولا ضيق نفاس", {"dizziness": "absent", "shortness_of_breath": "absent"}),
        ("عندها سكر حمل", {"gdm": "present"}),
        ("قالت معها سكر الحمل", {"gdm": "present"}),
        ("الهيموغلوبين طلع 9.5", {"hemoglobin": "present"}),
        ("لازم تخفف السكريات", {"sugar_restriction": "present"}),
        ("بدنا نعيد تحليل الدم بعد فترة", {"repeat_blood_test": "planned"}),
        ("لازم نتابع نمو الجنين", {"fetal_growth_monitoring": "planned"}),
        ("تقيس السكر بالبيت", {"home_glucose_monitoring": "planned"}),
        # MSA / near-MSA
        ("لا تعاني من الدوخة أو ضيق في التنفس", {"dizziness": "absent", "shortness_of_breath": "absent"}),
        ("تم تشخيص سكري الحمل", {"gdm": "present"}),
        ("بلغ الهيموغلوبين 9.5", {"hemoglobin": "present"}),
        ("نُصحت بتقليل السكريات", {"sugar_restriction": "planned"}),
        ("سيعاد تحليل الدم لاحقاً", {"repeat_blood_test": "planned"}),
        ("تحتاج إلى متابعة نمو الجنين", {"fetal_growth_monitoring": "planned"}),
        ("قياس السكر بالبيت بشكل منتظم", {"home_glucose_monitoring": "planned"}),
        # Jordanian/Lebanese variants close to Syrian
        ("ما معها دوخة ولا ضيق نفس", {"dizziness": "absent", "shortness_of_breath": "absent"}),
        ("بدها تخفف سكريات", {"sugar_restriction": "present"}),
        ("رح نعيد فحص الدم", {"repeat_blood_test": "planned"}),
        ("بدنا نراقب نمو الجنين", {"fetal_growth_monitoring": "planned"}),
    ],
)
def test_shami_and_msa_entity_scenarios(text, expected):
    links = extract_for_item({"text": text, "label": "plan"})
    found = {str(link.get("code")): str(link.get("assertion", "present")) for link in links}
    for code, assertion in expected.items():
        assert code in found, (text, code, found)
        assert found[code] == assertion, (text, code, found[code])


def test_garbled_glucose_phrase_is_not_promoted_to_home_monitoring():
    text = "وتضلع متئيس السكر عندها بشكل منتظم"
    links = extract_for_item({"text": text, "label": "plan"})
    assert not any(link.get("code") == "home_glucose_monitoring" for link in links)


def test_second_patient_gdm_anemia_report_is_fact_first_and_not_raw():
    parts = [
        seg(0, "مريدة آلة، عمرها 28 سنة، حامل بالأسبوع 30، وجت لعنا اليوم شان نتائج التحليل", "info"),
        seg(1, "بالنسبة لتحليل اختبار السكر بالحمل طلع عندها يام ارتفاع، هذا الشيء بيعني انه هي معها سكر الحمل", "lab"),
        seg(2, "كمان الهيموغلوبين طلع 9.5، هذا الشيء بيعني انه هي معها في اردم واضح", "lab"),
        seg(3, "ما عندها لا دوخة ولا ضيق نفاس", "cardiology"),
        seg(4, "كمان حكينا عن تنظيم الاكل خاصة انه هي لازم تخفف السكريات و تضلع متئيس السكر عندها بشكل منتظم", "plan"),
        seg(5, "كمان حكينا انه هي لازم تابع و تضلع تاخد مكملات الحديد مشان فئر الدم", "plan"),
        seg(6, "و نرجع نعيد تحليل الدم بعد فترة و كمان لازم نتابع نمو الجنين، هذا الشيء بيكون اقرب من الحمل الطبيعي", "plan"),
    ]
    parts = enrich_cross_segment_context(parts)
    report = build_report(
        "patient-two",
        parts,
        patient_info={
            "display_name": "آية أحمد",
            "age_years": 28,
            "gestational_age_weeks": 30,
            "gestational_age_weeks_before_visit": 32,
            "effective_obstetric_status": "pregnant",
        },
    )

    subjective = report.soap_formatted["subjective"].text
    objective = report.soap_formatted["objective"].text
    assessment = report.soap_formatted["assessment"].text
    plan = report.soap_formatted["plan"].text

    assert "المريضة آية أحمد" in subjective
    assert "العمر 28 سنة" in subjective
    assert "حامل في الأسبوع 30" in subjective
    assert "مريدة آلة" not in subjective
    assert "مراجعة نتائج التحاليل" in subjective
    assert "تنفي المريضة وجود" in subjective
    assert "دوخة" in subjective and "ضيق النفس" in subjective

    assert "اختبار سكر الحمل مرتفع" in objective
    assert "الهيموغلوبين 9.5 غ/دل (منخفض)" in objective
    assert "يام ارتفاع" not in objective

    assert "سكري الحمل" in assessment
    assert "التقييم يتضمن" not in assessment  # concise doctor-facing wording

    assert "تنظيم الغذاء" in plan
    assert "تقليل السكريات" in plan
    assert "وُصف الحديد" in plan
    assert "سيُعاد تحليل الدم" in plan
    assert "متابعة نمو الجنين" in plan
    assert "تضلع متئيس" not in plan
    assert "فئر الدم" not in plan

    warnings = report.soap_formatted["subjective"].warnings
    assert any("30 مقابل 32" in warning for warning in warnings)


def test_explicit_condition_inside_lab_sentence_is_promoted_to_assessment_only_as_presentation():
    item = seg(
        0,
        "تحليل اختبار السكر بالحمل مرتفع وهذا يعني أن لديها سكري الحمل",
        "lab",
    )
    report = build_report("gdm-cross-section", [item])
    assert "اختبار سكر الحمل مرتفع" in report.soap_formatted["objective"].text
    assert "سكري الحمل" in report.soap_formatted["assessment"].text
    # Source routing/provenance stays Objective.
    assert report.soap["objective"].items[0].label == "lab"
    assert report.soap["assessment"].items == []


def test_low_normal_and_high_hb_statuses_render_explicitly():
    cases = [
        ("الهيموغلوبين 9.5", "(منخفض)"),
        ("الهيموغلوبين 11.5", "(طبيعي)"),
        ("الهيموغلوبين 16", "(مرتفع)"),
    ]
    for idx, (text, expected) in enumerate(cases):
        report = build_report(f"hb-{idx}", [seg(0, text, "lab")])
        assert expected in report.soap_formatted["objective"].text


def test_danger_sign_list_across_whisper_boundaries_stays_hypothetical():
    parts = [
        seg(0, "اذا حستت فيها مثل انه يصير فيها عندها وجع راس اوي يصير عندها", "symptom", start=0.0, end=4.0),
        seg(1, "نزيف تشوش بالرؤية او حست بحركة الجنين ببطنها انه هي خفت اذا", "symptom", start=4.0, end=8.0),
        seg(2, "صار في احد هاي الاعراض فهي لازم تراجعني حتى لو كان قبل اسبوعين", "diagnosis", start=8.0, end=11.0),
    ]
    enriched = enrich_cross_segment_context(parts)
    middle = code_map(enriched[1])
    assert middle["vaginal_bleeding"]["assertion"] == "hypothetical"
    assert middle["blurred_vision"]["assertion"] == "hypothetical"
    assert middle["reduced_fetal_movement"]["assertion"] == "hypothetical"
    assert code_map(enriched[2])["contingency_action"]["kind"] == "context"


def test_current_danger_sign_is_not_downgraded_by_previous_conditional():
    parts = [
        seg(0, "اذا صار صداع شديد لازم تراجع", "plan", start=0.0, end=2.0),
        seg(1, "حالياً عندها نزيف وتشوش بالرؤية", "symptom", start=2.0, end=4.0),
    ]
    enriched = enrich_cross_segment_context(parts)
    current = code_map(enriched[1])
    assert current["vaginal_bleeding"]["assertion"] == "present"
    assert current["blurred_vision"]["assertion"] == "present"


@pytest.mark.parametrize(
    "raw,candidate",
    [
        ("عمرها 28 سنة", "عمرها 28 عاماً"),
        ("ارتفاع الرحم 30 سنتيمتر", "ارتفاع الرحم 30 سنتيمتراً"),
    ],
)
def test_arabic_case_endings_do_not_create_false_unit_changes(raw, candidate):
    from app.core.nlp.canonicalization import ClinicalSafetyGuard

    decision = ClinicalSafetyGuard().validate(raw, candidate)
    assert not any(reason.startswith("clinical_units_changed") for reason in decision.reasons)
    assert not any(reason.startswith("introduced_clinical_unit") for reason in decision.reasons)


@pytest.mark.parametrize(
    "text,label,code,assertion,status,value",
    [
        # Symptoms / negation: Syrian, Lebanese/Jordanian, MSA
        ("ما في نزيف", "symptom", "vaginal_bleeding", "absent", None, None),
        ("مو موجود نزيف", "symptom", "vaginal_bleeding", "absent", None, None),
        ("ما عندها تشوش بالرؤية", "symptom", "blurred_vision", "absent", None, None),
        ("لا تعاني من صداع", "symptom", "headache", "absent", None, None),
        ("عندها وجع راس خفيف", "symptom", "headache", "present", None, None),
        ("رجليها منفخين شوي", "symptom", "edema", "present", None, None),
        ("حاسّة بدوخة من الصبح", "symptom", "dizziness", "present", None, None),
        # Safety-netting must not become current disease
        ("إذا صار نزيف راجعينا فوراً", "plan", "vaginal_bleeding", "hypothetical", None, None),
        ("لو صار تشوش بالرؤية تراجعي فوراً", "plan", "blurred_vision", "hypothetical", None, None),
        ("في حال خفت حركة الجنين راجعي المستشفى", "plan", "reduced_fetal_movement", "hypothetical", None, None),
        # Vitals / fetal assessment
        ("ضغطها 120 على 80", "vital", "bp", "present", None, 120.0),
        ("الضغط 12 على 8", "vital", "bp", "present", None, 120.0),
        ("نبضها 88 بالدقيقة", "vital", "pulse", "present", None, 88.0),
        ("حرارتها 37.2", "vital", "temp", "present", None, 37.2),
        ("وزنها 70 كيلو", "vital", "weight_kg", "present", None, 70.0),
        ("نبض الجنين 145 ومنتظم", "vital", "fetal_heart_rate_bpm", "present", None, 145.0),
        ("ارتفاع الرحم 29 سنتيمتر", "vital", "fundal_height_cm", "present", None, 29.0),
        ("حركة الجنين طبيعية ومنتظمة", "vital", "fetal_movement_normal", "present", "normal", None),
        # Labs
        ("الهيموغلوبين 10.2", "lab", "hemoglobin", "present", "low", 10.2),
        ("الهيموغلوبين 12.1 طبيعي", "lab", "hemoglobin", "present", "normal", 12.1),
        ("السكر 92", "lab", "blood_glucose", "present", None, 92.0),
        ("البروتين بالبول سلبي", "lab", "urine_protein", "present", "normal", None),
        ("البروتين بالبول إيجابي", "lab", "urine_protein", "present", "high", None),
        ("تحليل سكر الحمل طلع مرتفع", "lab", "gestational_glucose_screen", "present", "high", None),
        # Diagnosis/history
        ("معها سكر الحمل", "diagnosis", "gdm", "present", None, None),
        ("بالحمل السابق صار معها تسمم حمل", "history", "preeclampsia", "historical", None, None),
        ("أمها عندها سكري", "history", "diabetes", "family", None, None),
        # Clear plans
        ("موعدنا الجاي بعد أسبوعين", "plan", "follow_up_visit", "planned", None, None),
        ("نصحتها تقيس الضغط بالبيت مرتين باليوم", "plan", "home_bp_monitoring", "planned", None, None),
        ("طلبت منها تعمل سونار الأسبوع الجاي", "plan", "obstetric_ultrasound", "planned", None, None),
        ("نصحتها تقلل الملح", "plan", "salt_restriction", "planned", None, None),
        ("نصحتها بتنظيم الأكل", "plan", "dietary_plan", "planned", None, None),
        ("نصحتها تراقب السكر بالبيت", "plan", "home_glucose_monitoring", "planned", None, None),
        ("سنُعيد تحليل الدم لاحقاً", "plan", "repeat_blood_test", "planned", None, None),
        ("سنتابع نمو الجنين بشكل أقرب", "plan", "fetal_growth_monitoring", "planned", None, None),
    ],
)
def test_extended_shami_msa_clinical_scenarios(text, label, code, assertion, status, value):
    links = extract_for_item({"text": text, "label": label})
    hit = next((link for link in links if link.get("code") == code), None)
    assert hit is not None, (text, code, links)
    assert hit.get("assertion", "present") == assertion, (text, code, hit)
    if status is not None:
        assert hit.get("status") == status, (text, code, hit)
    if value is not None:
        assert float(hit.get("value")) == pytest.approx(value), (text, code, hit)


def test_patient_record_age_mismatch_is_warned_not_silently_overwritten():
    report = build_report(
        "age-warning",
        [seg(0, "مريضة عمرها 24 سنة حامل بالأسبوع 20", "info")],
        patient_info={
            "display_name": "سارة",
            "age_years": 28,
            "gestational_age_weeks": 20,
        },
    )
    assert "العمر 28 سنة" in report.soap_formatted["subjective"].text
    assert any(
        "العمر المذكور في التسجيل" in warning
        for warning in report.soap_formatted["subjective"].warnings
    )
