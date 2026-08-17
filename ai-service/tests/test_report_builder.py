import json

from app.core.nlp.sections import SOAP_ORDER
from app.core.report.builder import build_report
from app.core.report.schema import ClassifiedSegment


def _seg(order, text, label, conf, start=0.0, end=1.0):
    return ClassifiedSegment(
        order_index=order,
        text=text,
        start_sec=start,
        end_sec=end,
        source_segment_index=order,
        label=label,
        confidence=conf,
    )


def _sample():
    return [
        _seg(0, "صداع شديد", "symptom", 0.95),
        _seg(1, "ضغط الدم مرتفع", "vital", 0.80),
        _seg(2, "تشخيص تسمم حمل", "pregnancy_risk", 0.70),
        _seg(3, "نوصي بالراحة", "plan", 0.40),  # low confidence
        _seg(4, "حالة طارئة نزيف", "emergency", 0.99),  # urgent -> assessment
    ]


def test_soap_grouping_and_flags():
    report = build_report("job1", _sample(), low_confidence_threshold=0.5)

    # all four SOAP sections always present, fixed order
    assert list(report.soap.keys()) == SOAP_ORDER

    assert [i.label for i in report.soap["subjective"].items] == ["symptom"]
    assert [i.label for i in report.soap["objective"].items] == ["vital"]
    assert {i.label for i in report.soap["assessment"].items} == {"pregnancy_risk", "emergency"}
    assert [i.label for i in report.soap["plan"].items] == ["plan"]


def test_item_fields():
    report = build_report("job1", _sample(), low_confidence_threshold=0.5)
    emergency = report.soap["assessment"].items
    urgent = [i for i in emergency if i.label == "emergency"][0]
    assert urgent.is_urgent is True
    assert urgent.item_id == "job1-0004"
    assert urgent.label_ar == "الطوارئ"

    low = report.soap["plan"].items[0]
    assert low.is_low_confidence is True
    assert low.entity_links is None
    assert low.text_rephrased is None


def test_summary_counts():
    report = build_report("job1", _sample(), low_confidence_threshold=0.5)
    s = report.summary
    assert s.total_segments == 5
    assert s.low_confidence_count == 1
    assert s.urgent_count == 1
    assert s.per_label_counts["symptom"] == 1
    assert report.pipeline_meta.rephrase_applied is False


def test_report_json_serializable():
    report = build_report("job1", _sample())
    dumped = json.dumps(report.model_dump(mode="json"), ensure_ascii=False)
    assert "job1-0000" in dumped


def test_spoken_action_language_routes_to_plan_without_changing_classifier_label():
    seg = _seg(0, "لذلك المريضة بيحاجة لتقنيم عاجل بالمستشفى", "pregnancy_risk", 0.80)
    report = build_report("job-plan", [seg])
    item = report.soap["plan"].items[0]
    assert item.label == "pregnancy_risk"
    assert "assessment" in item.also_in_sections
    assert item.is_urgent is True


def test_monitoring_instruction_routes_procedure_to_plan():
    seg = _seg(0, "ومراقبة قريبة للام والجنين", "procedure", 0.82)
    report = build_report("job-monitor", [seg])
    item = report.soap["plan"].items[0]
    assert item.label == "procedure"
    assert "objective" in item.also_in_sections


def test_completed_procedure_stays_objective():
    seg = _seg(0, "تمت مراقبة الجنين لمدة عشرين دقيقة", "procedure", 0.90)
    report = build_report("job-done", [seg])
    assert report.soap["objective"].items[0].label == "procedure"


def test_severe_range_bp_is_cross_cutting_urgent_even_without_emergency_label():
    seg = ClassifiedSegment(
        order_index=0, text="الضغط 160 على 110", start_sec=0.0, end_sec=1.0,
        source_segment_index=0, label="vital", confidence=0.95,
        entity_links=[{
            "kind": "vital", "code": "bp", "assertion": "present",
            "value": 160.0, "value2": 110.0, "unit": "mmHg",
        }],
    )
    report = build_report("job-bp", [seg])
    assert report.soap["objective"].items[0].is_urgent is True
    assert report.summary.urgent_count == 1



def test_negated_emergency_topic_is_not_urgent_and_routes_to_subjective():
    seg = ClassifiedSegment(
        order_index=0,
        text="ما في نزيف ولا ألم شديد ولا تسرب سوائل",
        start_sec=0.0,
        end_sec=1.0,
        source_segment_index=0,
        label="emergency",
        confidence=0.97,
        entity_links=[{
            "kind": "symptom", "code": "vaginal_bleeding", "assertion": "absent"
        }],
    )
    report = build_report("job-neg-emergency", [seg])
    item = report.soap["subjective"].items[0]
    assert item.label == "emergency"  # model output retained for auditability
    assert item.is_urgent is False
    assert "assessment" in item.also_in_sections
    assert report.summary.urgent_count == 0


def test_patient_reported_normal_fetal_movement_routes_followup_to_subjective():
    seg = _seg(0, "حركة الجنين جيدة حسب كلامها", "follow_up", 0.80)
    report = build_report("job-fm", [seg])
    item = report.soap["subjective"].items[0]
    assert item.label == "follow_up"
    assert item.is_urgent is False
    assert "plan" in item.also_in_sections
