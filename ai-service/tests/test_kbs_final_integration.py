"""Final KBS integration contract tests that do not require model weights/Experta.

These tests pin the safety boundary between the probabilistic NLP pipeline and the
final EXPERTA_MED rule package.  Actual rule firing is covered by the KBS integration
suite when the `experta` dependency is installed.
"""
from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient

from EXPERTA_MED.extraction import extract_report
from tests.conftest import build_app


def _report(item: dict) -> dict:
    return {
        "job_id": "contract-job",
        "created_at": "2026-08-14T00:00:00+00:00",
        "soap": {
            "subjective": {"items": [item]},
            "objective": {"items": []},
            "assessment": {"items": []},
            "plan": {"items": []},
        },
    }


def _base_item() -> dict:
    return {
        "item_id": "contract-job-0000",
        "order_index": 0,
        "text": "المريضة تعاني من صداع شديد وتشوش بالرؤية",
        "label": "symptom",
        "soap_section": "subjective",
        "confidence": 0.96,
    }


def test_structured_present_parent_allows_kbs_specific_refinement():
    item = _base_item()
    item["entity_links"] = [
        {"kind": "symptom", "code": "headache", "assertion": "present"},
        {"kind": "symptom", "code": "blurred_vision", "assertion": "present"},
    ]
    entities = extract_report(_report(item)).entities
    facts = {(e.kind, e.code, e.negated) for e in entities}
    assert ("symptom", "headache", False) in facts
    assert ("symptom", "severe_headache", False) in facts
    assert ("symptom", "blurred_vision", False) in facts


def test_structured_absent_or_historical_entity_cannot_be_promoted_by_text_fallback():
    for assertion in ("absent", "historical", "hypothetical", "family", "planned"):
        item = _base_item()
        item["entity_links"] = [
            {"kind": "symptom", "code": "headache", "assertion": assertion},
            {"kind": "symptom", "code": "blurred_vision", "assertion": "present"},
        ]
        entities = extract_report(_report(item)).entities
        positives = {(e.kind, e.code) for e in entities if not e.negated}
        assert ("symptom", "severe_headache") not in positives, assertion


def test_asr_suspect_item_is_review_signal_not_actionable_fact():
    item = _base_item()
    item.update(
        is_asr_suspect=True,
        is_urgent=True,
        entity_links=[
            {"kind": "symptom", "code": "headache", "assertion": "present"},
            {"kind": "symptom", "code": "blurred_vision", "assertion": "present"},
        ],
    )
    entities = extract_report(_report(item)).entities
    assert entities
    assert all(e.kind == "low_conf" for e in entities)
    assert not any(e.code in {"headache", "severe_headache", "blurred_vision"} for e in entities)



def test_structured_item_keeps_cross_cutting_emergency_flag():
    item = _base_item()
    item.update(
        is_urgent=True,
        entity_links=[
            {"kind": "vital", "code": "bp", "assertion": "present",
             "value": 160.0, "value2": 110.0},
        ],
    )
    entities = extract_report(_report(item)).entities
    assert any(e.kind == "emergency" for e in entities)
    assert any(e.kind == "vital" and e.code == "bp" for e in entities)


def test_follow5_ids_are_stable_and_retired_ids_are_never_reused():
    source = Path("EXPERTA_MED/md_rules/Pre_pregnancy.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    meta_node = next(
        node.value
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "RULE_META" for target in node.targets)
    )
    meta = ast.literal_eval(meta_node)
    expected = {f"R{i}" for i in range(1, 28)} | {f"R{i}" for i in range(33, 49)}
    assert set(meta) == expected
    assert not ({f"R{i}" for i in range(28, 33)} & set(meta))

    emitted = {
        node.args[0].value
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "_follow5_emit"
        and node.args
        and isinstance(node.args[0], ast.Constant)
        and isinstance(node.args[0].value, str)
    }
    assert emitted == expected


def test_job_persists_kbs_suggestion_and_feedback_api(temp_env, wav_bytes):
    app = build_app()

    def fake_kbs(report, history, *, patient_context):
        return SimpleNamespace(
            result={
                "engine": "EXPERTA_MED",
                "rules_version": "0.8.0",
                "patient_context": patient_context,
                "trends": [],
                "audit_trail": [],
                "suggestions": [
                    {
                        "rule_id": "R10",
                        "rule_version": "0.8.0",
                        "severity": "low",
                        "condition": "data_quality",
                        "title_ar": "يحتاج تأكيد",
                        "detail_ar": "راجع المقطع قبل الاعتماد عليه.",
                        "missing_tests": [],
                        "evidence": [{"item_id": f"{report.job_id}-0000"}],
                        "inference_chain": ["low confidence", "R10"],
                        "references": [],
                    }
                ],
            }
        )

    app.state.kbs_analyzer = fake_kbs
    with TestClient(app) as client:
        response = client.post(
            "/jobs", files={"file": ("note.wav", wav_bytes, "audio/wav")}
        )
        assert response.status_code == 202
        job_id = response.json()["job_id"]
        assert client.get(f"/jobs/{job_id}").json()["status"] == "complete"

        suggestions = client.get(f"/jobs/{job_id}/suggestions")
        assert suggestions.status_code == 200
        payload = suggestions.json()
        assert payload["total"] == 1
        row = payload["suggestions"][0]
        assert row["rule_id"] == "R10"
        assert row["rule_version"] == "0.8.0"

        feedback = client.post(
            f"/suggestions/{row['id']}/feedback",
            json={"action": "accepted", "actor": "doctor", "reason": "confirmed"},
        )
        assert feedback.status_code == 201
        assert feedback.json()["action"] == "accepted"
        history = client.get(f"/suggestions/{row['id']}/feedback").json()
        assert len(history) == 1 and history[0]["actor"] == "doctor"

    app.state.executor.shutdown(wait=False)


def test_production_artifacts_path_persists_raw_transcript(temp_env, wav_bytes):
    from app.core.report.builder import build_report
    from app.core.report.schema import AudioMeta, ClassifiedSegment, PipelineMeta

    class ArtifactPipeline:
        def process_with_artifacts(self, audio_path, job_id, filename=None, patient_info=None):
            report = build_report(
                job_id,
                [
                    ClassifiedSegment(
                        order_index=0,
                        text="المريضة تشكو من صداع",
                        start_sec=0.0,
                        end_sec=1.0,
                        source_segment_index=0,
                        label="symptom",
                        confidence=0.91,
                    )
                ],
                audio_meta=AudioMeta(filename=filename, duration_sec=1.0, whisper_model="stub-whisper"),
                pipeline_meta=PipelineMeta(arabert_model_name="stub-arabert"),
                patient_info=patient_info,
            )
            transcript = {
                "language": "ar",
                "text": "المريضة تشكو من صداع",
                "segments": [{"text": "المريضة تشكو من صداع", "avg_logprob": -0.1}],
            }
            return SimpleNamespace(
                report=report,
                transcript=transcript,
                transcript_quality={"avg_logprob": -0.1, "no_speech_ratio": 0.0},
            )

    app = build_app(pipeline=ArtifactPipeline())
    with TestClient(app) as client:
        response = client.post(
            "/jobs", files={"file": ("note.wav", wav_bytes, "audio/wav")}
        )
        job_id = response.json()["job_id"]
        stored = client.get(f"/jobs/{job_id}/transcript", params={"include_raw": True})
        assert stored.status_code == 200
        payload = stored.json()
        assert payload["language"] == "ar"
        assert payload["avg_logprob"] == -0.1
        assert payload["raw"]["text"] == "المريضة تشكو من صداع"
    app.state.executor.shutdown(wait=False)


def test_benign_gestational_age_context_does_not_emit_low_conf_critical():
    item = {
        "item_id": "contract-job-0000",
        "order_index": 0,
        "text": "المريضة حامل تقريبا بالاسبوع 34",
        "label": "pregnancy_risk",
        "soap_section": "assessment",
        "confidence": 0.92,
        "combined_confidence": 0.75,
        "is_low_confidence": True,
        "is_urgent": False,
        "entity_links": [],
    }
    entities = extract_report(_report(item)).entities
    assert not any(e.kind == "low_conf" and e.code == "pregnancy_risk" for e in entities)


def test_low_conf_pregnancy_risk_with_real_clinical_evidence_still_emits_review_fact():
    item = {
        "item_id": "contract-job-0000",
        "order_index": 0,
        "text": "حامل بالاسبوع 34 وعندها صداع شديد",
        "label": "pregnancy_risk",
        "soap_section": "assessment",
        "confidence": 0.40,
        "combined_confidence": 0.35,
        "is_low_confidence": True,
        "is_urgent": False,
        "entity_links": [
            {"kind": "symptom", "code": "headache", "assertion": "present"},
        ],
    }
    entities = extract_report(_report(item)).entities
    assert any(e.kind == "low_conf" and e.code == "pregnancy_risk" for e in entities)



def test_emergency_topic_with_explicitly_false_urgent_flag_does_not_emit_emergency_fact():
    item = {
        "item_id": "contract-job-0000",
        "order_index": 0,
        "text": "ما في نزيف ولا ألم شديد ولا تسرب سوائل",
        "label": "emergency",
        "soap_section": "subjective",
        "confidence": 0.97,
        "combined_confidence": 0.85,
        "is_low_confidence": False,
        "is_urgent": False,
        "entity_links": [
            {"kind": "symptom", "code": "vaginal_bleeding", "assertion": "absent"},
        ],
    }
    entities = extract_report(_report(item)).entities
    assert not any(e.kind == "emergency" for e in entities)


def test_kbs_fallback_ga_parser_accepts_tatweel_article_week_form():
    item = {
        "item_id": "contract-job-0000",
        "order_index": 0,
        "text": "المريضة حامل بالأسبوع الـ 28",
        "label": "pregnancy_risk",
        "soap_section": "assessment",
        "confidence": 0.98,
        "combined_confidence": 0.90,
        "is_low_confidence": False,
        "is_urgent": False,
        "entity_links": [],
    }
    extracted = extract_report(_report(item))
    assert extracted.ga_weeks == 28
    assert extracted.trimester == 3


def test_raw_transcript_recovers_gestational_age_into_persisted_patient_info(temp_env, wav_bytes):
    """A clearly spoken GA must survive into PatientState/report even if SOAP loses it.

    This pins the production failure observed with ``الأسبوع الـ 28``: EXPERTA could
    recover 28 from text while the persisted report still exposed GA/trimester as null.
    The runner now uses raw ASR text as a conservative obstetric-context recovery path.
    """
    from app.core.report.builder import build_report
    from app.core.report.schema import AudioMeta, ClassifiedSegment, PipelineMeta

    class ArtifactPipeline:
        def process_with_artifacts(self, audio_path, job_id, filename=None, patient_info=None):
            # Deliberately omit the numeric GA from the SOAP item to exercise the
            # transcript-recovery path rather than the ordinary report parser.
            report = build_report(
                job_id,
                [
                    ClassifiedSegment(
                        order_index=0,
                        text="المريضة حامل وجاءت للمتابعة العادية",
                        start_sec=0.0,
                        end_sec=2.0,
                        source_segment_index=0,
                        label="pregnancy_risk",
                        confidence=0.98,
                    )
                ],
                audio_meta=AudioMeta(
                    filename=filename,
                    duration_sec=2.0,
                    detected_language="ar",
                    whisper_model="stub-whisper",
                ),
                pipeline_meta=PipelineMeta(arabert_model_name="stub-arabert"),
                patient_info=patient_info,
            )
            transcript = {
                "language": "ar",
                "text": "المريضة حامل بالأسبوع الـ 28 وجاءت للمتابعة العادية",
                "segments": [
                    {
                        "text": "المريضة حامل بالأسبوع الـ 28 وجاءت للمتابعة العادية",
                        "avg_logprob": -0.1,
                    }
                ],
            }
            return SimpleNamespace(
                report=report,
                transcript=transcript,
                transcript_quality={"avg_logprob": -0.1, "no_speech_ratio": 0.0},
            )

    app = build_app(pipeline=ArtifactPipeline())
    with TestClient(app) as client:
        response = client.post(
            "/jobs",
            files={"file": ("normal-pregnancy.wav", wav_bytes, "audio/wav")},
            data={"mrn": "TEST-GA-PERSIST-V4"},
        )
        assert response.status_code == 202
        job_id = response.json()["job_id"]
        status = client.get(f"/jobs/{job_id}").json()
        assert status["status"] == "complete"

        report = client.get(f"/jobs/{job_id}/report").json()
        info = report["patient_info"]
        assert info["effective_obstetric_status"] == "pregnant"
        assert info["gestational_age_weeks"] == 28
        assert info["trimester"] == 3

        # The recovered observation must also be persisted as longitudinal state,
        # otherwise the next visit would forget the GA even though the report shows it.
        patient_id = response.json()["patient_id"]
        patient = client.get(f"/patients/{patient_id}").json()
        assert patient["gestational_age_weeks_now"] == 28

    app.state.executor.shutdown(wait=False)


def test_structured_hemoglobin_is_single_numeric_source_for_kbs():
    """KBS compatibility fact must inherit the structured decimal, not reparse text."""
    item = {
        "item_id": "contract-job-0000",
        "order_index": 0,
        "text": "كمان الهيموغلوبين طلع 9.5",
        "label": "lab",
        "soap_section": "objective",
        "confidence": 0.98,
        "combined_confidence": 0.90,
        "is_low_confidence": False,
        "is_urgent": False,
        "entity_links": [
            {
                "kind": "lab",
                "code": "hemoglobin",
                "assertion": "present",
                "value": 9.5,
                "unit": "g/dL",
                "status": "low",
                "confidence": 1.0,
            },
            {
                "kind": "test",
                "code": "hemoglobin_test",
                "assertion": "present",
                "confidence": 1.0,
            },
        ],
    }
    entities = extract_report(_report(item)).entities
    structured = [e for e in entities if e.kind == "lab" and e.code == "hemoglobin"]
    compat = [e for e in entities if e.kind == "clinical" and e.code == "hemoglobin_g_dl"]
    assert len(structured) == 1 and structured[0].value == 9.5
    assert len(compat) == 1 and compat[0].value == 9.5


def test_explicit_trusted_gdm_condition_suppresses_classifier_only_r10_fact():
    item = {
        "item_id": "contract-job-0000",
        "order_index": 0,
        "text": "هذا الشي بيعني انه هي معها سكر الحمل",
        "label": "pregnancy_risk",
        "soap_section": "assessment",
        "confidence": 0.47,
        "combined_confidence": 0.41,
        "is_low_confidence": True,
        "is_urgent": False,
        "entity_links": [
            {
                "kind": "condition",
                "code": "gdm",
                "assertion": "present",
                "confidence": 1.0,
            }
        ],
    }
    entities = extract_report(_report(item)).entities
    assert any(e.kind == "diagnosis" and e.code == "gdm" for e in entities)
    assert not any(e.kind == "low_conf" and e.code == "pregnancy_risk" for e in entities)


def test_scenario3_colloquial_contract_preserves_gdm_hb_and_negated_dyspnea():
    """Pin the real colloquial Scenario-3 phrases that exposed V4 gaps."""
    from app.core.nlp.extraction import extract_for_item

    raw_items = [
        {
            "item_id": "s3-0002",
            "order_index": 2,
            "text": "بالنسبة لتحليل اختبار السكر بالحمل طلع عندها يام ارتفاع",
            "label": "lab",
            "soap_section": "objective",
            "confidence": 0.98,
            "combined_confidence": 0.85,
            "is_low_confidence": False,
            "is_urgent": False,
        },
        {
            "item_id": "s3-0003",
            "order_index": 3,
            "text": "هذا الشي بيعني انه هي معها سكر الحمل",
            "label": "pregnancy_risk",
            "soap_section": "assessment",
            "confidence": 0.47,
            "combined_confidence": 0.41,
            "is_low_confidence": True,
            "is_urgent": False,
        },
        {
            "item_id": "s3-0004",
            "order_index": 4,
            "text": "كمان الهيموغلوبين طلع 9.5",
            "label": "lab",
            "soap_section": "objective",
            "confidence": 0.98,
            "combined_confidence": 0.85,
            "is_low_confidence": False,
            "is_urgent": False,
        },
        {
            "item_id": "s3-0006",
            "order_index": 6,
            "text": "ما عندها لا دوخة ولا ضيق نفاس",
            "label": "cardiology",
            "soap_section": "subjective",
            "confidence": 0.34,
            "combined_confidence": 0.29,
            "is_low_confidence": True,
            "is_urgent": False,
        },
    ]
    for item in raw_items:
        item["entity_links"] = extract_for_item(item)

    report = {
        "job_id": "s3-contract",
        "patient_info": {"pregnant": True, "gestational_age_weeks": 30, "trimester": 3},
        "soap": {
            "subjective": {"items": [raw_items[3]]},
            "objective": {"items": [raw_items[0], raw_items[2]]},
            "assessment": {"items": [raw_items[1]]},
            "plan": {"items": []},
        },
    }
    entities = extract_report(report).entities

    assert any(e.kind == "diagnosis" and e.code == "gdm" and not e.negated for e in entities)
    assert any(
        e.kind == "lab" and e.code == "gestational_glucose_screen" and e.status == "high"
        for e in entities
    )
    assert any(
        e.kind == "clinical" and e.code == "hemoglobin_g_dl" and e.value == 9.5
        for e in entities
    )
    assert any(
        e.kind == "symptom" and e.code == "shortness_of_breath" and e.negated
        for e in entities
    )
    assert not any(e.kind == "low_conf" and e.code == "pregnancy_risk" for e in entities)


def test_suspected_water_breaking_stays_separate_from_confirmed_rom_rules():
    item = _base_item()
    item.update(
        text="وبتشك انه هو مي الجنين",
        label="diagnosis",
        soap_section="assessment",
        entity_links=[
            {
                "kind": "symptom",
                "code": "water_breaking",
                "assertion": "present",
                "status": "suspected",
                "confidence": 1.0,
            }
        ],
    )
    entities = extract_report(_report(item)).entities
    assert any(e.kind == "clinical" and e.code == "suspected_rupture_of_membranes" for e in entities)
    assert not any(e.kind == "symptom" and e.code == "water_breaking" for e in entities)


def test_structured_cervical_dilation_is_single_source_for_kbs():
    item = _base_item()
    item.update(
        text="وعنق الرحم متوسعة تقريبا 5 سم",
        label="diagnosis",
        soap_section="objective",
        entity_links=[
            {
                "kind": "clinical",
                "code": "cervical_dilation_cm",
                "assertion": "present",
                "value": 5.0,
                "unit": "cm",
            }
        ],
    )
    entities = extract_report(_report(item)).entities
    matches = [e for e in entities if e.kind == "clinical" and e.code == "cervical_dilation_cm"]
    assert len(matches) == 1
    assert matches[0].value == 5.0


def test_low_conf_monitoring_plan_does_not_emit_r10_fact():
    item = _base_item()
    item.update(
        text="ونبض الجنين وحالة الام بشكل مستمر",
        label="pregnancy_risk",
        soap_section="plan",
        is_low_confidence=True,
        is_urgent=False,
        entity_links=[
            {"kind": "test", "code": "ctg", "assertion": "present"},
            {"kind": "context", "code": "plan_continuation", "assertion": "present"},
        ],
    )
    entities = extract_report(_report(item)).entities
    assert not any(e.kind == "low_conf" and e.code == "pregnancy_risk" for e in entities)
