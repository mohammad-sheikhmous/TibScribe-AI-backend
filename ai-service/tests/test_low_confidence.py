"""Composite low-confidence flagging + multi-label SOAP routing (P4-08 / P4-10).

The v1 rule was `confidence < 0.5` and nothing else — one arbitrary number, with no
explanation attached. These tests pin the replacement: four independent reasons, each
stated in the report so a reviewer knows whether to re-listen, re-label, or ignore.
"""
from app.core.nlp.sections import SOAP_PRIORITY, soap_for_labels
from app.core.report.builder import build_report
from app.core.report.schema import ClassifiedSegment


def item(**overrides):
    base = dict(
        order_index=0, text="جملة", start_sec=0.0, end_sec=1.0, source_segment_index=0,
        label="symptom", confidence=0.95,
    )
    base.update(overrides)
    return ClassifiedSegment(**base)


def only_item(*segments, threshold=0.5):
    report = build_report("job", list(segments), low_confidence_threshold=threshold)
    return next(it for section in report.soap.values() for it in section.items)


# --- the four reasons ------------------------------------------------------------------

def test_confident_clean_item_is_not_flagged():
    result = only_item(item(asr_confidence=0.95, entropy=0.1, ood_score=0.1))
    assert result.is_low_confidence is False
    assert result.low_confidence_reasons == []


def test_low_confidence_is_flagged_with_the_number():
    result = only_item(item(confidence=0.3))
    assert result.is_low_confidence
    assert any("confidence" in reason for reason in result.low_confidence_reasons)


def test_suspected_hallucination_flags_even_a_confident_label():
    """A sentence that may not have been said cannot be trusted however sure the model is."""
    result = only_item(item(confidence=0.99, asr_confidence=0.98, is_asr_suspect=True))
    assert result.is_low_confidence
    assert any("hallucination" in reason for reason in result.low_confidence_reasons)


def test_high_entropy_flags_an_ambiguous_sentence():
    result = only_item(item(confidence=0.95, entropy=0.9))
    assert result.is_low_confidence
    assert any("ambiguous" in reason for reason in result.low_confidence_reasons)


def test_out_of_distribution_input_is_flagged():
    """The signal softmax cannot express: confident, but about something foreign."""
    result = only_item(item(confidence=0.99, ood_score=0.99))
    assert result.is_low_confidence
    assert any("OOD" in reason for reason in result.low_confidence_reasons)




def test_ood_alone_does_not_discard_strong_structured_clinical_evidence():
    result = only_item(item(
        confidence=0.99,
        ood_score=1.0,
        entity_links=[{
            "kind": "vital", "code": "bp", "assertion": "present",
            "value": 160.0, "value2": 110.0,
        }],
    ))
    assert result.is_low_confidence is False
    assert result.ood_score == 1.0

def test_bad_audio_drags_down_a_confident_classification():
    result = only_item(item(confidence=0.99, asr_confidence=0.2))
    assert result.combined_confidence < 0.25
    assert result.is_low_confidence


def test_several_problems_are_all_reported():
    result = only_item(item(confidence=0.3, entropy=0.9, ood_score=0.9, is_asr_suspect=True))
    assert len(result.low_confidence_reasons) == 4


def test_uncertainty_scores_reach_the_report():
    result = only_item(item(entropy=0.42, ood_score=0.31, review_priority=0.55))
    assert result.entropy == 0.42
    assert result.ood_score == 0.31
    assert result.review_priority == 0.55


# --- multi-label SOAP routing -------------------------------------------------------------

def test_single_label_behaviour_is_unchanged():
    primary, also_in = soap_for_labels(["symptom"])
    assert primary == "subjective" and also_in == []


def test_multi_label_item_lands_in_the_most_actionable_section():
    """vital(Objective) + symptom(Subjective) + medication(Plan) -> Plan, and says so."""
    primary, also_in = soap_for_labels(["vital", "symptom", "medication"])
    assert primary == "plan"
    assert set(also_in) == {"subjective", "objective"}


def test_priority_order_is_explicit():
    assert SOAP_PRIORITY[0] == "plan"
    assert soap_for_labels(["symptom", "diagnosis"])[0] == "assessment"


def test_empty_label_list_is_handled():
    assert soap_for_labels([])[0] == "assessment"


def test_report_item_defaults_labels_to_the_single_label():
    """Consumers can read `labels` today, before the multi-label head lands."""
    result = only_item(item(label="vital"))
    assert result.labels == ["vital"]
    assert result.also_in_sections == []


def test_multi_label_segment_flows_through_the_builder():
    result = only_item(item(label="medication", labels=["vital", "medication"]))
    assert result.soap_section == "plan"
    assert result.also_in_sections == ["objective"]
