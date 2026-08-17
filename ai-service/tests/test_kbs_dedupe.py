from EXPERTA_MED.suggestions import Suggestion, merge_suggestions


def test_duplicate_critical_rule_merges_evidence_instead_of_duplicate_cards():
    first = Suggestion(
        rule_id="R09", rule_version="0.8.0", severity="critical",
        title_ar="تصعيد", detail_ar="عاجل",
        evidence=[{"item_id": "i1", "text": "الضغط 160 على 110"}],
        chain=["EmergencyFlag", "R09"],
    )
    second = Suggestion(
        rule_id="R09", rule_version="0.8.0", severity="critical",
        title_ar="تصعيد", detail_ar="عاجل",
        evidence=[{"item_id": "i2", "text": "تقييم عاجل بالمستشفى"}],
        chain=["EmergencyFlag", "R09"],
    )
    merge_suggestions(first, second)
    assert {e["item_id"] for e in first.evidence} == {"i1", "i2"}
    assert first.chain == ["EmergencyFlag", "R09"]


def test_engine_r09_uses_one_logical_dedupe_key():
    from pathlib import Path
    source = Path("EXPERTA_MED/engine.py").read_text(encoding="utf-8")
    assert 'dedupe=("R09", "emergency_escalation")' in source
