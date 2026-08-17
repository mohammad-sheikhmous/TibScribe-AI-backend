"""Tests for the EXPERTA_MED KBS: extraction, single-report rules, temporal rules."""
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from EXPERTA_MED.extraction import extract_report
from EXPERTA_MED.runner import analyze_reports


def _item(oid, text, label, conf=0.9, urgent=False, low=False, section="subjective"):
    return {
        "item_id": None, "order_index": oid, "text": text, "label": label,
        "confidence": conf, "is_urgent": urgent, "is_low_confidence": low,
        "soap_section": section, "start_sec": oid, "end_sec": oid + 1,
        "timestamp_precision": "segment", "source_segment_index": oid, "entity_links": None,
    }


def _report(job, items, when):
    soap = {"subjective": {"soap_key": "subjective", "title_ar": "", "items": []},
            "objective": {"soap_key": "objective", "title_ar": "", "items": []},
            "assessment": {"soap_key": "assessment", "title_ar": "", "items": []},
            "plan": {"soap_key": "plan", "title_ar": "", "items": []}}
    for it in items:
        it["item_id"] = f"{job}-{it['order_index']:04d}"  # job-prefixed, like the real builder
        soap[it["soap_section"]]["items"].append(it)
    return {"schema_version": "1.0", "job_id": job, "created_at": when.isoformat(),
            "audio": {}, "pipeline_meta": {}, "patient_info": {}, "soap": soap, "summary": {}}


# ---- extraction ----

def test_extraction_bp_shortform_and_context():
    rep = _report("t1", [
        _item(0, "المريضة حامل بالأسبوع 34 وتشتكي من صداع", "symptom"),
        _item(1, "فيتال: ضغط 11/7", "vital", section="objective"),
    ], datetime(2026, 1, 1, tzinfo=timezone.utc))
    ext = extract_report(rep, 0)
    assert ext.pregnant and ext.ga_weeks == 34
    codes = {(e.kind, e.code) for e in ext.entities}
    assert ("symptom", "headache") in codes
    bp = [e for e in ext.entities if e.kind == "vital" and e.code == "bp"][0]
    assert bp.value == 110 and bp.value2 == 70  # Levantine short form expanded


def test_negation_detected():
    rep = _report("t2", [_item(0, "لا يوجد نزيف مهبلي", "symptom")],
                  datetime(2026, 1, 1, tzinfo=timezone.utc))
    ext = extract_report(rep, 0)
    bleed = [e for e in ext.entities if e.code == "vaginal_bleeding"]
    assert bleed and bleed[0].negated is True


# ---- single-report rules ----

def test_preeclampsia_and_allergy_conflict():
    rep = _report("t3", [
        _item(0, "حامل بالأسبوع 33 وتشتكي من صداع شديد وتشوش بالرؤية", "symptom"),
        _item(1, "فيتال: ضغط 160 على 100", "vital", section="objective"),
        _item(2, "حساسية من البنسلين", "allergy"),
        _item(3, "أعطيناها أموكسيسيلين", "medication", section="plan"),
    ], datetime(2026, 1, 1, tzinfo=timezone.utc))
    res = analyze_reports([rep])
    ids = {s["rule_id"] for s in res["suggestions"]}
    assert {"R01", "R02", "R06"} <= ids
    r06 = [s for s in res["suggestions"] if s["rule_id"] == "R06"][0]
    assert r06["severity"] == "critical"
    # every suggestion is explainable + traceable
    for s in res["suggestions"]:
        assert s["inference_chain"] and s["rule_version"]


def test_missing_tests_uses_knowledge_graph():
    rep = _report("t4", [_item(0, "التشخيص تسمم حمل", "pregnancy_risk", section="assessment")],
                  datetime(2026, 1, 1, tzinfo=timezone.utc))
    res = analyze_reports([rep])
    r03 = [s for s in res["suggestions"] if s["rule_id"] == "R03"][0]
    assert "urine_protein_test" in r03["missing_tests"]


# ---- temporal rules across a series ----

def test_temporal_persistent_and_worsening():
    series = [
        _report("v1", [_item(0, "حامل بالأسبوع 30 وعندها صداع", "symptom"),
                       _item(1, "ضغط 130 على 85", "vital", section="objective")],
                 datetime(2026, 6, 1, tzinfo=timezone.utc)),
        _report("v2", [_item(0, "لسا عندها صداع", "symptom"),
                       _item(1, "ضغط 148 على 92", "vital", section="objective")],
                 datetime(2026, 6, 20, tzinfo=timezone.utc)),
        _report("v3", [_item(0, "الصداع مستمر", "symptom"),
                       _item(1, "ضغط 165 على 105", "vital", section="objective")],
                 datetime(2026, 7, 10, tzinfo=timezone.utc)),
    ]
    res = analyze_reports(series)
    ids = {s["rule_id"] for s in res["suggestions"]}
    assert "R11" in ids  # persistent headache
    assert "R12" in ids  # worsening BP
    r12 = [s for s in res["suggestions"] if s["rule_id"] == "R12"][0]
    assert len(r12["evidence"]) == 2  # first + last visit


def test_no_false_positives_on_empty():
    rep = _report("t5", [_item(0, "الحالة مستقرة والفحص طبيعي", "info")],
                  datetime(2026, 1, 1, tzinfo=timezone.utc))
    res = analyze_reports([rep])
    assert res["suggestions"] == []


# ---- P5-15: the KBS now consumes entities the pipeline already extracted ----

def _item_with_links(oid, text, label, section="subjective"):
    """A report item as the v2 pipeline produces it: entity_links already filled."""
    from app.core.nlp.extraction import extract_for_item

    it = _item(oid, text, label, section=section)
    it["item_id"] = f"x-{oid:04d}"
    it["entity_links"] = extract_for_item(it)
    return it


def test_safety_netting_advice_no_longer_fires_a_rule():
    """Gap ك-٣: this exact sentence used to raise a pre-eclampsia alert."""
    rep = _report("s1", [
        _item_with_links(0, "المريضة حامل بالأسبوع 30 والفحص طبيعي", "info"),
        _item_with_links(1, "إذا صار عندك صداع أو تشوش بالرؤية راجعينا فوراً", "plan",
                         section="plan"),
    ], datetime(2026, 1, 1, tzinfo=timezone.utc))
    assert analyze_reports([rep])["suggestions"] == []


def test_real_symptoms_still_fire_the_same_rule():
    """The other half of the fix: suppressing advice must not suppress findings."""
    rep = _report("s2", [
        _item_with_links(0, "المريضة حامل بالأسبوع 33 وتشتكي من صداع شديد", "symptom"),
        _item_with_links(1, "وعندها تشوش بالرؤية منذ يومين", "symptom"),
    ], datetime(2026, 1, 1, tzinfo=timezone.utc))
    assert "R01" in {s["rule_id"] for s in analyze_reports([rep])["suggestions"]}


def test_a_relatives_diagnosis_is_not_the_patients():
    rep = _report("s3", [_item_with_links(0, "أمها عندها سكري", "history")],
                  datetime(2026, 1, 1, tzinfo=timezone.utc))
    result = analyze_reports([rep])
    assert result["extracted_facts_count"] == 0
    assert result["suggestions"] == []


def test_legacy_reports_without_entity_links_still_work():
    """Backward compatibility: pre-P5 reports fall back to the in-module extractor."""
    rep = _report("s4", [_item(0, "المريضة حامل بالأسبوع 33 وتشتكي من صداع شديد وتشوش بالرؤية",
                               "symptom")],
                  datetime(2026, 1, 1, tzinfo=timezone.utc))
    assert analyze_reports([rep])["extracted_facts_count"] > 0
