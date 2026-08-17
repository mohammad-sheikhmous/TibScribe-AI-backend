"""Orchestration: report JSON(s) -> facts -> Experta -> explainable suggestions.

The original project pipeline is preserved. The only extension is that the runner
also declares ``ClinicalFact`` objects used by the rules converted from Markdown.
Reports may provide exact facts through an optional ``clinical_facts`` array while
ordinary text extraction continues to work as before.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from . import _compat  # noqa: F401
from .engine import RULES_VERSION, MedicalKBS
from .extraction import Entity, ReportExtraction, extract_report
from .facts import (
    Allergy, ClinicalFact, DiagnosisNoted, EmergencyFlag, FollowUpPlanned,
    LabResult, LowConfidenceCritical, Medication, PatientContext, ReportMeta,
    Symptom, TestDone, Trend, Vital,
)
from .knowledge_graph import KnowledgeGraph
from .ontology import drug_family
from .suggestions import Suggestion, sort_suggestions, to_markdown
from .temporal import compute_trends


def _clinical_provenance(e: Entity) -> dict:
    return {
        "item_id": e.item_id,
        "text": e.text,
        "report_index": e.report_index,
        "confidence": e.confidence,
    }


def _declare_cf(
    engine: MedicalKBS,
    code: str,
    value: Any = True,
    *,
    item_id: str = "",
    text: str = "",
    report_index: int | None = None,
    confidence: float = 1.0,
    status: str | None = None,
    unit: str | None = None,
    negated: bool = False,
) -> None:
    engine.declare(ClinicalFact(
        code=code, value=value, status=status, unit=unit, negated=negated,
        item_id=item_id, text=text, report_index=report_index,
        confidence=confidence,
    ))


def _declare_entity(engine: MedicalKBS, e: Entity) -> None:
    prov = dict(item_id=e.item_id, text=e.text, report_index=e.report_index)
    cprov = _clinical_provenance(e)

    if e.kind == "symptom":
        engine.declare(Symptom(code=e.code, negated=e.negated, confidence=e.confidence, **prov))
        _declare_cf(engine, e.code, not e.negated, negated=e.negated, **cprov)
        # Stable aliases used by the source-derived rules.
        if e.code == "water_breaking" and not e.negated:
            _declare_cf(engine, "rupture_of_membranes", True, **cprov)
        if e.code == "fatigue" and not e.negated:
            _declare_cf(engine, "easy_fatigue", True, **cprov)
        if e.code == "fever" and not e.negated:
            _declare_cf(engine, "fever", True, **cprov)
        if e.code == "foul_vaginal_discharge" and not e.negated:
            # Source rules use both names: D45 expects foul_vaginal_discharge while
            # postpartum early-review rules use the lochia-specific alias.
            _declare_cf(engine, "foul_lochia", True, **cprov)

    elif e.kind == "vital" and e.code == "bp":
        sys_v = float(e.value or 0)
        dia_v = float(e.value2 or 0)
        engine.declare(Vital(kind="bp", sys=sys_v, dia=dia_v, **prov))
        _declare_cf(engine, "systolic_bp", sys_v, unit="mmHg", **cprov)
        _declare_cf(engine, "diastolic_bp", dia_v, unit="mmHg", **cprov)
        if dia_v >= 90:
            _declare_cf(engine, "diastolic_bp_ge_90", True, **cprov)
        if dia_v >= 110:
            _declare_cf(engine, "diastolic_bp_ge_110", True, **cprov)
        if 90 <= dia_v <= 110:
            _declare_cf(engine, "diastolic_bp_90_to_110", True, **cprov)

    elif e.kind == "vital" and e.code == "temp":
        temp = float(e.value or 0)
        engine.declare(Vital(kind="temp", value=temp, **prov))
        _declare_cf(engine, "temperature_c", temp, unit="C", **cprov)
        _declare_cf(engine, "temperature_ge_38", temp >= 38, **cprov)
        _declare_cf(engine, "temperature_gt_38", temp > 38, **cprov)
        _declare_cf(engine, "temperature_not_over_38", temp <= 38, **cprov)
        _declare_cf(engine, "temperature_lt_38", temp < 38, **cprov)
        if temp > 38:
            _declare_cf(engine, "fever", True, **cprov)

    elif e.kind == "lab":
        engine.declare(LabResult(code=e.code, status=e.status or "normal",
                                 value=e.value, **prov))
        _declare_cf(engine, e.code, e.value if e.value is not None else True,
                    status=e.status, unit=e.unit, **cprov)

    elif e.kind == "test_done":
        engine.declare(TestDone(code=e.code, item_id=e.item_id, report_index=e.report_index))
        _declare_cf(engine, f"test_done_{e.code}", True, **cprov)

    elif e.kind == "diagnosis":
        engine.declare(DiagnosisNoted(code=e.code, **prov))
        _declare_cf(engine, e.code, True, **cprov)

    elif e.kind == "medication":
        engine.declare(Medication(drug=e.code, family=drug_family(e.code), **prov))
        _declare_cf(engine, f"medication_{e.code}", True, **cprov)

    elif e.kind == "allergy":
        engine.declare(Allergy(drug=e.code, family=drug_family(e.code), **prov))
        _declare_cf(engine, f"allergy_{e.code}", True, **cprov)

    elif e.kind == "emergency":
        engine.declare(EmergencyFlag(**prov))

    elif e.kind == "followup":
        engine.declare(FollowUpPlanned(item_id=e.item_id, report_index=e.report_index))

    elif e.kind == "low_conf":
        engine.declare(LowConfidenceCritical(label=e.code, **prov))

    elif e.kind == "clinical":
        _declare_cf(
            engine, e.code, e.value if e.value is not None else (not e.negated),
            status=e.status, unit=e.unit, negated=e.negated, **cprov,
        )


def _latest_numeric(exts: list[ReportExtraction], code: str) -> float | None:
    for ext in reversed(exts):
        for e in reversed(ext.entities):
            if e.kind == "clinical" and e.code == code:
                try:
                    return float(e.value)
                except (TypeError, ValueError):
                    return None
    return None


def _present_codes(exts: list[ReportExtraction]) -> set[str]:
    codes: set[str] = set()
    for ext in exts:
        for e in ext.entities:
            if e.kind == "symptom" and not e.negated:
                codes.add(e.code)
            elif e.kind in {"diagnosis", "clinical"}:
                if e.value is True or (e.kind == "diagnosis" and not e.negated):
                    codes.add(e.code)
    return codes


def _declare_derived_facts(
    engine: MedicalKBS,
    exts: list[ReportExtraction],
    *,
    ga_weeks: int,
    trimester: int,
) -> None:
    """Create deterministic threshold and cluster facts used by source-derived rules."""
    def derived(code: str, value: Any = True, text: str = "") -> None:
        _declare_cf(engine, code, value, item_id=f"derived-{code}",
                    text=text or code, report_index=None, confidence=1.0)

    codes = _present_codes(exts)

    # Age and parity thresholds.
    age = _latest_numeric(exts, "maternal_age")
    if age is not None:
        derived("maternal_age_under_14", age < 14, f"maternal_age={age}")
        derived("maternal_age_under_16", age < 16, f"maternal_age={age}")
    parity = _latest_numeric(exts, "parity")
    if parity is not None:
        derived("parity_over_6", parity > 6, f"parity={parity}")

    # Gestational-age operational mappings. Eight months is represented as 32 weeks.
    if ga_weeks:
        derived("early_pregnancy", ga_weeks <= 13, f"ga_weeks={ga_weeks}")
        derived("gestation_less_than_24_weeks", ga_weeks < 24, f"ga_weeks={ga_weeks}")
        derived("gestation_less_than_8_months", ga_weeks < 32, f"ga_weeks={ga_weeks}")
        derived("gestation_8_months_or_more", ga_weeks >= 32, f"ga_weeks={ga_weeks}")
        derived("gestation_more_than_8_months", ga_weeks > 32, f"ga_weeks={ga_weeks}")
        derived("gestation_24_to_34_weeks", 24 <= ga_weeks <= 34, f"ga_weeks={ga_weeks}")
    if trimester:
        derived("trimester_2_or_3", trimester in {2, 3}, f"trimester={trimester}")

    # Count elevated BP readings across supplied visits. A report may also state
    # explicitly that the pressure was measured more than once while documenting
    # only the latest numeric reading, so combine both sources.
    extracted_elevated_readings = sum(
        1 for ext in exts for e in ext.entities
        if e.kind == "vital" and e.code == "bp" and float(e.value2 or 0) >= 90
    )

    documented_readings = _latest_numeric(exts, "bp_readings_count")
    documented_readings_count = (
        int(documented_readings) if documented_readings is not None else 0
    )

    effective_readings = max(
        extracted_elevated_readings,
        documented_readings_count,
    )

    derived(
        "bp_readings_at_least_2",
        effective_readings >= 2,
        (
            f"extracted_elevated_bp_readings={extracted_elevated_readings}; "
            f"documented_bp_readings={documented_readings_count}"
        ),
    )

    # Numeric laboratory thresholds.
    protein = _latest_numeric(exts, "proteinuria_grade")
    if protein is not None:
        derived("proteinuria_ge_3", protein >= 3, f"proteinuria_grade={protein}")
        derived("proteinuria_ge_2", protein >= 2, f"proteinuria_grade={protein}")
        derived("proteinuria_lt_2", protein < 2, f"proteinuria_grade={protein}")
    hb = _latest_numeric(exts, "hemoglobin_g_dl")
    if hb is not None:
        derived("hemoglobin_lt_7", hb < 7, f"hemoglobin={hb}")
        derived("hemoglobin_7_to_11", 7 <= hb <= 11, f"hemoglobin={hb}")
        derived("hemoglobin_gt_11", hb > 11, f"hemoglobin={hb}")
    rr = _latest_numeric(exts, "respiratory_rate")
    if rr is not None:
        derived("respiratory_rate_gt_30", rr > 30, f"respiratory_rate={rr}")

    # Composite danger and respiratory clusters.
    if {"severe_headache", "blurred_vision"}.issubset(codes):
        derived("danger_headache_visual_cluster", True,
                "severe_headache + blurred_vision")
    has_documented_fever = (
        "fever" in codes
        or any(
            e.kind == "vital"
            and e.code == "temp"
            and float(e.value or 0) >= 38
            for ext in exts
            for e in ext.entities
        )
    )

    if has_documented_fever and "unable_to_stand" in codes:
        derived(
            "danger_fever_weakness_cluster",
            True,
            "temperature >= 38 or fever + unable_to_stand",
        )
    pneumonia_count = sum(1 for c in ("temperature_gt_38", "shortness_of_breath", "chest_pain")
                          if c in codes)
    # Temperature thresholds are declared directly and therefore are not necessarily
    # present in Entity codes; inspect measured temperatures too.
    if any(e.kind == "vital" and e.code == "temp" and float(e.value or 0) > 38
           for ext in exts for e in ext.entities):
        pneumonia_count = 1 + sum(1 for c in ("shortness_of_breath", "chest_pain") if c in codes)
    derived("pneumonia_sign_count_ge_2", pneumonia_count >= 2,
            f"pneumonia_sign_count={pneumonia_count}")

    # Labour-specific thresholds and aggregate facts (follow3.py).
    cervical = _latest_numeric(exts, "cervical_dilation_cm")
    irregular = "irregular_contractions" in codes
    if cervical is not None:
        if cervical < 5 or (cervical <= 5 and irregular):
            derived("labour_stage_latent", True, f"cervical_dilation_cm={cervical}")
        if 5 <= cervical < 10:
            derived("labour_stage_active", True, f"cervical_dilation_cm={cervical}")
        if cervical >= 10:
            derived("labour_stage_second", True, f"cervical_dilation_cm={cervical}")
    if codes & {"perineum_bulging", "fetal_head_visible"}:
        derived("labour_stage_second", True, "perineum_bulging or fetal_head_visible")

    if ga_weeks:
        derived("gestation_less_than_32_weeks", ga_weeks < 32, f"ga_weeks={ga_weeks}")

    diastolic_values = [
        float(e.value2 or 0)
        for ext in exts for e in ext.entities
        if e.kind == "vital" and e.code == "bp"
    ]
    if diastolic_values:
        latest_dia = diastolic_values[-1]
        derived("diastolic_bp_gt_90", latest_dia > 90, f"diastolic_bp={latest_dia}")

    fhr = _latest_numeric(exts, "fetal_heart_rate_bpm")
    if fhr is not None:
        derived("fetal_heart_rate_abnormal", fhr < 120 or fhr > 160,
                f"fetal_heart_rate_bpm={fhr}")

    obstructed_signs = {
        "transverse_lie", "continuous_contractions_no_rest",
        "constant_pain_between_contractions", "sudden_severe_abdominal_pain",
        "horizontal_abdominal_ridge", "labour_over_24h",
    }
    if codes & obstructed_signs:
        derived("obstructed_labour_sign", True,
                " + ".join(sorted(codes & obstructed_signs)))

    dehydration_signs = {"thirst", "sunken_eyes", "dry_mouth", "slow_skin_pinch_return"}
    dehydration_count = len(codes & dehydration_signs)
    derived("dehydration_sign_count_ge_2", dehydration_count >= 2,
            f"dehydration_sign_count={dehydration_count}")

    assistance_risks = {
        "genital_warts_or_scars_obstructing", "previous_third_degree_tear",
        "third_trimester_bleeding_history", "previous_cesarean",
        "previous_forceps_delivery", "maternal_age_under_14",
    }
    if codes & assistance_risks:
        derived("delivery_extra_assistance_risk", True,
                " + ".join(sorted(codes & assistance_risks)))

    active_hours = _latest_numeric(exts, "active_stage_duration_hours")
    if active_hours is not None:
        is_primipara = "primipara" in codes
        prolonged = active_hours > (12 if is_primipara else 10)
        derived("active_stage_prolonged", prolonged,
                f"active_stage_hours={active_hours}; primipara={is_primipara}")

    episiotomy_indication_codes = {
        "genital_warts_or_scars_obstructing", "previous_third_degree_tear",
        "breech_presentation", "shoulder_dystocia",
    }
    if codes & episiotomy_indication_codes:
        derived(
            "episiotomy_indication_present",
            True,
            " + ".join(sorted(codes & episiotomy_indication_codes)),
        )

    # Delivery and immediate postpartum derived facts (follow4.py).
    second_stage_hours = _latest_numeric(exts, "second_stage_duration_hours")
    if second_stage_hours is not None:
        derived(
            "second_stage_duration_ge_2h",
            second_stage_hours >= 2,
            f"second_stage_duration_hours={second_stage_hours}",
        )

    postpartum_hours = _latest_numeric(exts, "postpartum_hours_since_birth")
    if postpartum_hours is not None:
        derived("postpartum_first_hour", 0 <= postpartum_hours <= 1,
                f"postpartum_hours={postpartum_hours}")
        derived("postpartum_hours_2_to_4", 2 <= postpartum_hours <= 4,
                f"postpartum_hours={postpartum_hours}")
        derived("postpartum_hours_4_to_24", 4 < postpartum_hours <= 24,
                f"postpartum_hours={postpartum_hours}")

    severe_pe_codes = {"severe_headache", "blurred_vision", "epigastric_pain"}
    derived(
        "severe_preeclampsia_symptom",
        bool(codes & severe_pe_codes),
        " + ".join(sorted(codes & severe_pe_codes)) or "no severe symptom",
    )

    if protein is not None:
        derived("proteinuria_absent", protein < 2,
                f"proteinuria_grade={protein}")

    if "newborn_not_breathing_or_gasping" in codes:
        derived("newborn_resuscitation_needed", True,
                "newborn_not_breathing_or_gasping")

    infection_signs = {
        "chills", "foul_vaginal_discharge", "lower_abdominal_pain",
        "fetal_tachycardia_30min", "rom_duration_gt_18h",
    }
    derived(
        "postpartum_infection_additional_sign",
        bool(codes & infection_signs),
        " + ".join(sorted(codes & infection_signs)) or "no additional infection sign",
    )

    immediate_danger_codes = {
        "pads_2_to_3_in_20_30min", "increasing_bleeding", "convulsions",
        "breathing_difficulty", "fever_with_severe_weakness",
        "severe_abdominal_pain",
    }
    if codes & immediate_danger_codes:
        derived("postpartum_immediate_danger_sign", True,
                " + ".join(sorted(codes & immediate_danger_codes)))

    early_review_codes = {
        "mild_fever", "abdominal_pain", "breast_or_nipple_problem",
        "urinary_leakage", "foul_lochia", "severe_depression",
        "suicidal_ideation",
    }
    if codes & early_review_codes:
        derived("postpartum_early_review_sign", True,
                " + ".join(sorted(codes & early_review_codes)))

    advanced_signs = {
        "weight_loss_visible", "diarrhea_chronic", "cough_over_1_month",
        "shortness_of_breath", "itchy_rash", "herpes_zoster_rib_side",
        "lymphadenopathy", "oral_fissures", "abnormal_vaginal_discharge",
    }
    hiv_risks = {
        "occupational_exposure", "multiple_partners", "iv_drug_use",
        "previous_blood_transfusion", "partner_died_of_aids", "rape_history",
    }
    # Follow2 HIV-PrEP risk alias: explicit high risk or one of the source examples.
    if "high_hiv_exposure_risk" in codes or codes & {"partner_hiv_positive", "multiple_partners"}:
        derived(
            "high_hiv_exposure_risk",
            True,
            "explicit high risk or partner_hiv_positive/multiple_partners",
        )

    advanced_count = len(codes & advanced_signs)
    risk_count = len(codes & hiv_risks)
    derived("advanced_hiv_sign_count", advanced_count,
            f"advanced_hiv_sign_count={advanced_count}")
    derived("advanced_hiv_sign_count_ge_1", advanced_count >= 1)
    derived("advanced_hiv_sign_count_ge_2", advanced_count >= 2)
    derived("hiv_risk_factor_count", risk_count, f"hiv_risk_factor_count={risk_count}")
    derived("hiv_risk_factor_count_ge_1", risk_count >= 1)


def _declare_all(
    engine: MedicalKBS,
    exts: list[ReportExtraction],
    trends: list[dict],
    patient_context: dict | None = None,
) -> dict:
    """Declare the current patient context plus facts from the report series.

    The *current* visit controls obstetric stage. Historical reports contribute
    trends and prior findings, but must never make pregnancy/labour/delivery
    sticky. The application may pass an authoritative ``patient_context`` from
    its longitudinal PatientState store; standalone CLI use falls back to the
    latest extracted report.
    """
    latest = exts[-1] if exts else None
    patient_context = patient_context or {}

    effective_status = str(
        patient_context.get("effective_obstetric_status")
        or patient_context.get("status")
        or ""
    ).lower()
    if effective_status:
        pregnant = effective_status == "pregnant"
        postpartum = effective_status == "postpartum"
    else:
        pregnant = bool(latest.pregnant) if latest else False
        postpartum = bool(latest.postpartum) if latest else False

    labour = bool(getattr(latest, "labour", False)) if latest else False
    labour_stage = getattr(latest, "labour_stage", None) if latest else None
    delivery = bool(getattr(latest, "delivery", False)) if latest else False
    delivery_phase = getattr(latest, "delivery_phase", None) if latest else None

    # Pregnancy state may legitimately be true immediately before a labour/delivery
    # visit. Antenatal rules must nevertheless be inactive once the *current* report
    # is in labour, delivery, newborn, or postpartum care.
    if labour or delivery or postpartum:
        pregnant = False
    ga = patient_context.get("gestational_age_weeks")
    if ga is None:
        ga = patient_context.get("ga_weeks")
    if ga is None:
        ga = next((e.ga_weeks for e in reversed(exts) if e.ga_weeks), 0)
    try:
        ga = int(ga or 0)
    except (TypeError, ValueError):
        ga = 0

    tri = patient_context.get("trimester")
    if tri is None:
        tri = next((e.trimester for e in reversed(exts) if e.trimester), 0)
    try:
        tri = int(tri or 0)
    except (TypeError, ValueError):
        tri = 0

    engine.declare(PatientContext(pregnant=pregnant, postpartum=postpartum,
                                  ga_weeks=ga or 0, trimester=tri or 0))
    _declare_cf(
        engine, "in_labour", labour,
        item_id="context-labour", text=f"clinical_stage=labour; stage={labour_stage or ''}",
        report_index=None, confidence=1.0,
    )
    if labour_stage in {"latent", "active", "second"}:
        _declare_cf(
            engine, f"labour_stage_{labour_stage}", True,
            item_id=f"context-labour-stage-{labour_stage}",
            text=f"labour_stage={labour_stage}", report_index=None,
            confidence=1.0,
        )

    _declare_cf(
        engine, "in_delivery", delivery,
        item_id="context-delivery",
        text=f"clinical_stage=delivery; phase={delivery_phase or ''}",
        report_index=None, confidence=1.0,
    )
    if delivery_phase in {"second", "third", "newborn", "postpartum", "discharge"}:
        _declare_cf(
            engine, f"delivery_phase_{delivery_phase}", True,
            item_id=f"context-delivery-phase-{delivery_phase}",
            text=f"delivery_phase={delivery_phase}", report_index=None,
            confidence=1.0,
        )
    engine.declare(ReportMeta(reports_total=len(exts)))

    # Ordinary clinical rules reason over the CURRENT visit only. Historical
    # reports are deliberately not dumped into the same working memory: doing so can
    # combine an old symptom with a new finding and manufacture a false rule match.
    # Cross-visit reasoning enters through explicit Trend facts below.
    current_exts = [latest] if latest is not None else []
    for ext in current_exts:
        for e in ext.entities:
            _declare_entity(engine, e)
    _declare_derived_facts(engine, current_exts, ga_weeks=ga or 0, trimester=tri or 0)
    for tr in trends:
        engine.declare(Trend(**tr))
    return {"pregnant": pregnant, "postpartum": postpartum,
            "labour": labour, "labour_stage": labour_stage,
            "delivery": delivery, "delivery_phase": delivery_phase,
            "ga_weeks": ga or None, "trimester": tri or None}


def analyze_reports(
    reports: list[dict],
    graph: KnowledgeGraph | None = None,
    *,
    patient_context: dict | None = None,
) -> dict:
    indexed = list(enumerate(reports))
    indexed.sort(key=lambda p: str(p[1].get("created_at") or p[0]))
    exts = [extract_report(r, i) for i, (_, r) in enumerate(indexed)]
    trends = compute_trends(exts)

    engine = MedicalKBS(graph=graph)
    engine.reset()
    context = _declare_all(engine, exts, trends, patient_context=patient_context)
    engine.run()

    suggestions: list[Suggestion] = sort_suggestions(engine.suggestions)
    return {
        "engine": "EXPERTA_MED",
        "rules_version": RULES_VERSION,
        "reports_total": len(reports),
        "job_ids": [e.job_id for e in exts],
        "patient_context": context,
        "trends": trends,
        "suggestions": [s.to_dict() for s in suggestions],
        "audit_trail": engine.fired,
        "extracted_facts_count": sum(len(e.entities) for e in exts),
    }


def _load_json_files(paths: list[str]) -> list[dict]:
    reports: list[dict] = []
    for p in paths:
        path = Path(p)
        files = sorted(path.glob("*.json")) if path.is_dir() else [path]
        for f in files:
            reports.append(json.loads(f.read_text(encoding="utf-8")))
    return reports


def analyze_paths(paths: list[str], graph: KnowledgeGraph | None = None) -> dict:
    return analyze_reports(_load_json_files(paths), graph=graph)


def result_to_markdown(result: dict) -> str:
    ctx = result.get("patient_context", {})
    header = {
        "عدد التقارير": result.get("reports_total"),
        "المرضى/المهام": "، ".join(result.get("job_ids", [])),
        "السياق": ("حامل" if ctx.get("pregnant") else "غير حامل")
                  + (f"، أسبوع {ctx['ga_weeks']}" if ctx.get("ga_weeks") else "")
                  + ("، ما بعد الولادة" if ctx.get("postpartum") else ""),
        "إصدار القواعد": result.get("rules_version"),
        "عدد الاقتراحات": len(result.get("suggestions", [])),
    }
    sugs = [_dict_to_suggestion(d) for d in result.get("suggestions", [])]
    return to_markdown(sugs, header=header)


def _dict_to_suggestion(d: dict) -> Suggestion:
    return Suggestion(
        rule_id=d["rule_id"], rule_version=d["rule_version"], severity=d["severity"],
        title_ar=d["title_ar"], detail_ar=d["detail_ar"], condition=d.get("condition"),
        missing_tests=d.get("missing_tests", []), evidence=d.get("evidence", []),
        chain=d.get("inference_chain", []), references=d.get("references", []),
    )
