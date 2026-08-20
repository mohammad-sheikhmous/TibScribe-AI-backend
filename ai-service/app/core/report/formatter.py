"""Deterministic, fact-first SOAP presentation formatter (P11).

The formatter does not ask a language model to invent a report.  It renders structured
entities/assertions into standard Arabic clinical phrases and falls back to the source
item only when the pipeline has no structured fact for that content.  This keeps the
presentation readable while preserving the exact clinical decision source.
"""
from __future__ import annotations

import re
from typing import Any, Iterable, Mapping

from ..nlp.extraction import display_name
from ..nlp.sections import SOAP_ORDER
from .schema import FormattedSoapSection, ReportItem, ReportSection

_WS = re.compile(r"\s+")
_TRAILING = re.compile(r"[\s،؛;,.]+$")
_AGE_RE = re.compile(r"(?:عمرها|عمره|العمر)\D{0,6}?(\d{1,3})\s*(?:سنة|سنه)")
_GA_RE = re.compile(r"(?:حامل(?:ة)?\s+)?(?:في|ب)?\s*(?:ال)?[اأ]?سبوع\s+(\d{1,2})", re.I)
_VISIT_RESULTS_RE = re.compile(r"(?:نتائج?\s+(?:ال)?(?:تحاليل|تحليل)|نتيجة\s+(?:ال)?(?:تحاليل|تحليل)|شان\s+نتائج?\s+(?:ال)?(?:تحاليل|تحليل))", re.I)
_PREV_DELIVERY_YEARS_RE = re.compile(
    r"(?:الولادة\s+السابقة|ولادة\s+سابقة).{0,40}?(?:قبل|منذ)\D{0,6}?(\d{1,3})\s*(?:سنة|سنين|سنوات)",
    re.I,
)
_FOLLOWUP_RE = re.compile(
    r"(?:موعد(?:نا)?(?:\s+الجاي)?|المتابعة|متابعة|مراجعة).{0,35}?بعد\s+(\d{1,2})\s*(أسبوع|اسبوع|أسابيع|اسابيع|يوم|أيام|ايام)",
    re.I,
)
_FOLLOWUP_TWO_WEEKS_RE = re.compile(r"بعد\s+(?:أسبوعين|اسبوعين|أسبوعان|اسبوعان)", re.I)
_SECOND_PREGNANCY_RE = re.compile(r"(?:هي)?\s*(?:تاني|ثاني)\s+حمل|الحمل\s+الثاني", re.I)
_PREV_NORMAL_DELIVERY_RE = re.compile(
    r"الولادة\s+السابقة.{0,34}?(?:طبيعية|طبيعي).{0,44}?(?:بدون|دون)\s+مضاعفات", re.I
)


def _num(value) -> str:
    if value is None:
        return ""
    try:
        number = float(value)
        if number.is_integer():
            return str(int(number))
        return (f"{number:.2f}").rstrip("0").rstrip(".")
    except (TypeError, ValueError):
        return str(value)


def _join_ar(values: Iterable[str]) -> str:
    values = [v for v in values if v]
    if not values:
        return ""
    if len(values) == 1:
        return values[0]
    if len(values) == 2:
        return f"{values[0]} و{values[1]}"
    return "، ".join(values[:-1]) + f"، و{values[-1]}"


def _clean_clause(text: str) -> str:
    text = _WS.sub(" ", str(text or "")).strip()
    return _TRAILING.sub("", text)


def _links(item: ReportItem) -> list[dict]:
    return [link for link in (item.entity_links or []) if isinstance(link, dict)]


def _dedupe_links(links: Iterable[dict]) -> list[dict]:
    out: list[dict] = []
    seen: set[tuple] = set()
    for link in links:
        key = (
            link.get("kind"), link.get("code"), link.get("assertion", "present"),
            link.get("value"), link.get("value2"), link.get("status"),
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(link)
    return out


class ClinicalSoapFormatter:
    applied = True
    name = "structured-facts-v3"

    def format(
        self,
        soap: Mapping[str, ReportSection],
        *,
        patient_info: Mapping[str, Any] | None = None,
    ) -> dict[str, FormattedSoapSection]:
        """Render doctor-facing SOAP from structured facts.

        Patient identity is supplied separately from ASR so a misspelled spoken name
        never becomes the authoritative identity in the final report. Clinical
        conditions may live inside Objective/Subjective items (because AraBERT labels
        the sentence topic); Assessment therefore also consumes explicit present
        condition entities across all SOAP sections.
        """
        patient_info = dict(patient_info or {})
        warnings = _consistency_warnings(soap, patient_info=patient_info)
        assessment_clauses, assessment_item_ids = _cross_section_assessment_facts(soap)

        formatted: dict[str, FormattedSoapSection] = {}
        for key in SOAP_ORDER:
            if key not in soap:
                continue
            formatted[key] = self.format_section(
                soap[key],
                warnings=warnings if key == "subjective" else [],
                patient_info=patient_info,
                initial_clauses=assessment_clauses if key == "assessment" else (),
                initial_item_ids=assessment_item_ids if key == "assessment" else (),
            )
        return formatted

    def format_section(
        self,
        section: ReportSection,
        *,
        warnings: list[str] | None = None,
        patient_info: Mapping[str, Any] | None = None,
        initial_clauses: Iterable[str] = (),
        initial_item_ids: Iterable[str] = (),
    ) -> FormattedSoapSection:
        clauses: list[str] = []
        item_ids: list[str] = list(initial_item_ids)
        seen: set[str] = set()
        patient_info = dict(patient_info or {})

        for clause in initial_clauses:
            clause = _clean_clause(clause)
            if clause and clause.casefold() not in seen:
                seen.add(clause.casefold())
                clauses.append(clause + ".")

        safety_codes: list[str] = []
        safety_item_ids: list[str] = []
        if section.soap_key == "plan":
            for item in sorted(section.items, key=lambda row: row.order_index):
                for link in _links(item):
                    if link.get("kind") == "symptom" and link.get("assertion") == "hypothetical":
                        name = _doctor_display_name(str(link.get("code")))
                        if name and name not in safety_codes:
                            safety_codes.append(name)
                            safety_item_ids.append(item.item_id)

        for item in sorted(section.items, key=lambda row: row.order_index):
            links = _links(item)

            if (
                section.soap_key == "plan"
                and safety_codes
                and any(
                    isinstance(link, dict)
                    and link.get("kind") == "context"
                    and link.get("code") == "contingency_action"
                    for link in links
                )
            ):
                continue

            if section.soap_key == "plan" and safety_codes:
                if any(
                    link.get("kind") == "symptom"
                    and link.get("assertion") == "hypothetical"
                    for link in links
                ):
                    continue

            item_clauses = self._format_item(
                item,
                section.soap_key,
                patient_info=patient_info,
            )

            for clause in item_clauses:
                clause = _clean_clause(clause)
                if not clause:
                    continue
                if (
                    safety_codes
                    and not links
                    and re.search(
                        r"(?:لازم|يجب).{0,16}(?:تراجع|مراجعة)",
                        item.text,
                    )
                ):
                    continue

                key = clause.casefold()
                if key in seen:
                    continue
                seen.add(key)
                clauses.append(clause + ".")
                item_ids.append(item.item_id)

        if safety_codes:
            safety_clause = (
                "تم شرح علامات الخطر التي تستدعي المراجعة الفورية: "
                f"{_join_ar(safety_codes)}."
            )
            if safety_clause.casefold() not in seen:
                clauses.append(safety_clause)
                item_ids.extend(safety_item_ids)

        return FormattedSoapSection(
            soap_key=section.soap_key,
            title_ar=section.title_ar,
            text=" ".join(clauses),
            item_ids=list(dict.fromkeys(item_ids)),
            warnings=list(warnings or []),
        )

    def _format_item(
        self,
        item: ReportItem,
        section_key: str,
        *,
        patient_info: Mapping[str, Any] | None = None,
    ) -> list[str]:
        if section_key == "objective":
            clauses = _objective_clauses(item)
        elif section_key == "plan":
            clauses = _plan_clauses(item)
        elif section_key == "subjective":
            clauses = _subjective_clauses(item, patient_info=patient_info)
        else:
            clauses = _assessment_clauses(item)

        if clauses:
            return clauses

        if section_key == "subjective":
            demographic = _demographic_clause(
                item.text_raw or item.text,
                patient_info=patient_info,
            )
            if demographic:
                out = [demographic]
                if _VISIT_RESULTS_RE.search(item.text_raw or item.text):
                    out.append("حضرت المريضة لمراجعة نتائج التحاليل")
                return out

        fallback = _clean_clause(item.text)
        return [fallback] if fallback else []

def _subjective_clauses(
    item: ReportItem,
    *,
    patient_info: Mapping[str, Any] | None = None,
) -> list[str]:
    links = _dedupe_links(_links(item))
    symptoms = [x for x in links if x.get("kind") == "symptom"]
    def symptom_name(link: dict) -> str:
        code = str(link.get("code"))
        if code == "edema":
            return "تورم بالقدمين" if re.search(r"قدم|رجل|ساق", item.text) else "وذمة/تورم"
        return _doctor_display_name(code)

    present = [symptom_name(x) for x in symptoms if x.get("assertion", "present") == "present"]
    absent = [symptom_name(x) for x in symptoms if x.get("assertion") == "absent"]
    historical = [_doctor_display_name(str(x.get("code"))) for x in symptoms if x.get("assertion") == "historical"]
    hypothetical = [_doctor_display_name(str(x.get("code"))) for x in symptoms if x.get("assertion") == "hypothetical"]

    clauses: list[str] = []
    if present and absent:
        clauses.append(f"تشكو المريضة من {_join_ar(present)}، دون {_join_ar(absent)}")
    elif present:
        clauses.append(f"تشكو المريضة من {_join_ar(present)}")
    elif absent:
        clauses.append(f"تنفي المريضة وجود {_join_ar(absent)}")
    if historical:
        clauses.append(f"يوجد تاريخ سابق لـ{_join_ar(historical)}")
    if hypothetical:
        clauses.append(f"ذُكرت كعلامات تحذيرية محتملة: {_join_ar(hypothetical)}")

    clinical = [x for x in links if x.get("kind") == "clinical" and x.get("assertion", "present") == "present"]
    for link in clinical:
        if link.get("code") == "fetal_movement_normal":
            clauses.append("حركة الجنين طبيعية ومنتظمة")

    # Obstetric-history wording can be standardized deterministically from explicit
    # phrases without asking a language model to infer missing clinical facts.
    if not clauses:
        demographic = _demographic_clause(
            item.text_raw or item.text,
            patient_info=patient_info,
        )
        if demographic:
            clauses.append(demographic)
            if _VISIT_RESULTS_RE.search(item.text_raw or item.text):
                clauses.append("حضرت المريضة لمراجعة نتائج التحاليل")
        else:
            history = _obstetric_history_clauses(item.text_raw or item.text)
            clauses.extend(history)
    return clauses


def _objective_clauses(item: ReportItem) -> list[str]:
    links = _dedupe_links(_links(item))
    clauses: list[str] = []
    used: set[str] = set()

    for link in sorted(links, key=lambda x: (x.get("char_start") is None, x.get("char_start") or 0)):
        code = str(link.get("code", ""))
        assertion = str(link.get("assertion", "present"))
        if assertion not in {"present", "performed"}:
            continue
        phrase = ""
        if code == "bp":
            phrase = f"ضغط الدم {_num(link.get('value'))}/{_num(link.get('value2'))} مم زئبق"
        elif code == "pulse":
            phrase = f"النبض {_num(link.get('value'))} ض/د"
        elif code == "temp":
            phrase = f"الحرارة {_num(link.get('value'))}°م"
        elif code == "weight_kg":
            phrase = f"الوزن {_num(link.get('value'))} كغ"
        elif code == "fundal_height_cm":
            phrase = f"ارتفاع قاع الرحم {_num(link.get('value'))} سم"
        elif code == "fetal_heart_rate_bpm":
            phrase = f"نبض الجنين {_num(link.get('value'))} ض/د"
        elif code == "fetal_movement_normal":
            phrase = "حركة الجنين طبيعية ومنتظمة"
        elif code == "fetal_presentation" and link.get("status") == "cephalic":
            phrase = "وضعية الجنين رأسية"
        elif code == "hemoglobin":
            suffix = {
                "normal": " (طبيعي)",
                "low": " (منخفض)",
                "high": " (مرتفع)",
            }.get(str(link.get("status") or ""), "")
            phrase = f"الهيموغلوبين {_num(link.get('value'))} غ/دل{suffix}"
        elif code == "gestational_glucose_screen":
            status = str(link.get("status") or "")
            phrase = {
                "high": "اختبار سكر الحمل مرتفع",
                "normal": "اختبار سكر الحمل ضمن الطبيعي",
                "low": "اختبار سكر الحمل منخفض",
            }.get(status, "تم تسجيل نتيجة اختبار سكر الحمل")
        elif code == "blood_glucose":
            phrase = f"سكر الدم {_num(link.get('value'))} ملغ/دل"
        elif code == "urine_protein":
            if link.get("status") == "normal":
                phrase = "بروتين البول سلبي" if re.search(r"سلبي|سلبية|negative", item.text, re.I) else "بروتين البول ضمن الطبيعي"
            elif link.get("status") == "high":
                phrase = "بروتين البول إيجابي"
        elif code == "liver_function" and re.search(r"طبيعي", item.text):
            phrase = "وظائف الكبد طبيعية"
        elif code == "kidney_function" and re.search(r"طبيعي", item.text):
            phrase = "وظائف الكلى طبيعية"

        if phrase and phrase not in used:
            used.add(phrase)
            clauses.append(phrase)

    return clauses


def _assessment_clauses(item: ReportItem) -> list[str]:
    links = _dedupe_links(_links(item))
    clauses: list[str] = []
    absent_symptoms = [
        _doctor_display_name(str(x.get("code"))) for x in links
        if x.get("kind") == "symptom" and x.get("assertion") == "absent"
    ]
    if absent_symptoms:
        rendered_absent = ["انقباضات رحمية" if name == _doctor_display_name("contractions") else name for name in absent_symptoms]
        clauses.append(f"لا توجد {_join_ar(rendered_absent)}")
    for link in links:
        if link.get("code") == "fetal_presentation" and link.get("status") == "cephalic":
            clauses.append("وضعية الجنين رأسية")
    return clauses


def _plan_clauses(item: ReportItem) -> list[str]:
    links = _dedupe_links(_links(item))
    clauses: list[str] = []

    med_links = sorted(
        [x for x in links if x.get("kind") == "medication" and x.get("assertion", "present") in {"present", "planned"}],
        key=lambda x: (x.get("char_start") is None, x.get("char_start") or 0),
    )
    meds = [
        ("الحديد" if str(x.get("code")) == "iron" else _doctor_display_name(str(x.get("code"))))
        for x in med_links
    ]
    if meds:
        clauses.append(f"وُصف {_join_ar(meds)}")

    codes = {str(x.get("code")): x for x in links}
    if "rest" in codes:
        clauses.append("نُصحت المريضة بالراحة")
    if "salt_restriction" in codes:
        clauses.append("نُصحت بتقليل الملح")
    if "home_bp_monitoring" in codes:
        clauses.append("نُصحت بمراقبة ضغط الدم في المنزل")
    if "dietary_plan" in codes and "sugar_restriction" in codes:
        clauses.append("نُصحت المريضة بتنظيم الغذاء وتقليل السكريات")
    elif "dietary_plan" in codes:
        clauses.append("نُصحت المريضة بتنظيم الغذاء")
    elif "sugar_restriction" in codes:
        clauses.append("نُصحت المريضة بتقليل السكريات")
    if "home_glucose_monitoring" in codes:
        clauses.append("نُصحت بمراقبة سكر الدم بانتظام")
    if "repeat_blood_test" in codes:
        timing = " بعد فترة" if re.search(r"بعد\s+فترة", item.text) else ""
        clauses.append(f"سيُعاد تحليل الدم{timing}")
    if "fetal_growth_monitoring" in codes:
        clauses.append("ستتم متابعة نمو الجنين")
    if "obstetric_ultrasound" in codes:
        assertion = str(codes["obstetric_ultrasound"].get("assertion", "present"))
        if assertion in {"planned", "hypothetical"}:
            if re.search(r"(?:النمو|للنمو)", item.text) and re.search(r"السائل\s+(?:الأمينوسي|الامينوسي)", item.text):
                timing = " خلال الأسبوع القادم" if re.search(r"(?:بال|في\s+ال)?[اأ]?سبوع\s+الجاي", item.text) else ""
                clauses.append(f"طُلب سونار لتقييم نمو الجنين والسائل الأمنيوسي{timing}")
            else:
                clauses.append("طُلب سونار توليدي")
        else:
            clauses.append("تم إجراء سونار توليدي")

    if _FOLLOWUP_TWO_WEEKS_RE.search(item.text):
        clauses.append("موعد المتابعة بعد أسبوعين")
        return clauses

    follow = _FOLLOWUP_RE.search(item.text)
    if follow:
        value, unit = follow.group(1), follow.group(2)
        unit = "أسبوع" if "سبوع" in unit else "يوم"
        if value == "2" and unit == "أسبوع":
            clauses.append("موعد المتابعة بعد أسبوعين")
        elif value == "2" and unit == "يوم":
            clauses.append("موعد المتابعة بعد يومين")
        else:
            clauses.append(f"موعد المتابعة بعد {value} {unit}")

    return clauses


def _obstetric_history_clauses(text: str) -> list[str]:
    """Conservative presentation of explicit obstetric-history phrases."""
    cleaned = _clean_clause(text)
    clauses: list[str] = []
    if _SECOND_PREGNANCY_RE.search(cleaned):
        clauses.append("الحمل الحالي هو الحمل الثاني")
    if _PREV_NORMAL_DELIVERY_RE.search(cleaned):
        years = _PREV_DELIVERY_YEARS_RE.search(cleaned)
        if years:
            clauses.append(f"ولادة سابقة طبيعية دون مضاعفات، ذُكر أنها كانت قبل {years.group(1)} سنة")
        else:
            clauses.append("ولادة سابقة طبيعية دون مضاعفات")
    return clauses


def _doctor_display_name(code: str) -> str:
    """Stable Arabic phrases used in the doctor-facing report."""
    overrides = {
        "shortness_of_breath": "ضيق النفس",
        "vaginal_bleeding": "نزيف مهبلي",
        "blurred_vision": "تشوش الرؤية",
        "reduced_fetal_movement": "قلة حركة الجنين",
        "headache": "صداع",
        "dizziness": "دوخة",
        "contractions": "انقباضات",
        "gdm": "سكري الحمل",
        "anemia": "فقر الدم",
    }
    return overrides.get(code, display_name(code))


def _cross_section_assessment_facts(
    soap: Mapping[str, ReportSection],
) -> tuple[list[str], list[str]]:
    """Promote explicit diagnoses/major abnormal structured findings to Assessment.

    The source item is not moved: provenance remains in its AraBERT SOAP section.
    """
    clauses: list[str] = []
    item_ids: list[str] = []
    seen: set[tuple[str, str]] = set()
    condition_codes: set[str] = set()
    low_hb_item_id: str | None = None

    for section in soap.values():
        for item in section.items:
            for link in _links(item):
                kind = str(link.get("kind") or "")
                code = str(link.get("code") or "")
                assertion = str(link.get("assertion", "present"))
                if assertion != "present" or not code:
                    continue

                if kind == "condition":
                    status = str(link.get("status") or "")
                    key = (code, status)
                    if key in seen:
                        continue
                    seen.add(key)
                    condition_codes.add(code)
                    name = _doctor_display_name(code)
                    clauses.append(f"اشتباه {name}" if status == "suspected" else name)
                    item_ids.append(item.item_id)

                if (
                    kind == "lab"
                    and code == "hemoglobin"
                    and str(link.get("status") or "") == "low"
                    and low_hb_item_id is None
                ):
                    low_hb_item_id = item.item_id

    # A low measured haemoglobin is a structured abnormality.  If the clinician
    # explicitly stated anaemia, that condition already appears above; otherwise use
    # the narrower non-diagnostic wording instead of inventing a diagnosis/severity.
    if low_hb_item_id is not None and "anemia" not in condition_codes:
        clauses.append("انخفاض الهيموغلوبين")
        item_ids.append(low_hb_item_id)

    return clauses, item_ids

def _safe_int(value: Any) -> int | None:
    try:
        if value is None or value == "":
            return None
        return int(round(float(value)))
    except (TypeError, ValueError):
        return None


def _demographic_clause(
    text: str,
    *,
    patient_info: Mapping[str, Any] | None = None,
) -> str:
    """Render identity from the patient record, using ASR only for missing age/GA."""
    cleaned = _clean_clause(text)
    patient_info = dict(patient_info or {})

    name = str(patient_info.get("display_name") or "").strip() or None
    age = _safe_int(patient_info.get("age_years"))
    ga = _safe_int(patient_info.get("gestational_age_weeks"))

    if age is None:
        age_match = _AGE_RE.search(cleaned)
        age = int(age_match.group(1)) if age_match else None
    if ga is None:
        ga_match = _GA_RE.search(cleaned)
        ga = int(ga_match.group(1)) if ga_match else None

    if not any((name, age is not None, ga is not None)):
        return ""

    pieces = [f"المريضة {name}" if name else "المريضة"]
    if age is not None:
        pieces.append(f"العمر {age} سنة")
    if ga is not None:
        pieces.append(f"حامل في الأسبوع {ga}")
    return "، ".join(pieces)


def _consistency_warnings(
    soap: Mapping[str, ReportSection],
    *,
    patient_info: Mapping[str, Any] | None = None,
) -> list[str]:
    """Cross-item/longitudinal contradictions are surfaced, never silently corrected."""
    texts = [item.text_raw or item.text for section in soap.values() for item in section.items]
    joined = " ".join(_clean_clause(text) for text in texts)
    age_match = _AGE_RE.search(joined)
    years_match = _PREV_DELIVERY_YEARS_RE.search(joined)
    warnings: list[str] = []

    if age_match and years_match:
        age = int(age_match.group(1))
        years = int(years_match.group(1))
        if years >= age:
            warnings.append(
                f"تعارض زمني محتمل في النص المنسوخ: العمر {age} سنة، بينما ذُكرت "
                f"ولادة سابقة قبل {years} سنة؛ راجع المقطع الصوتي قبل اعتماد المعلومة."
            )

    info = dict(patient_info or {})

    record_age = _safe_int(info.get("age_years"))
    spoken_age = int(age_match.group(1)) if age_match else None
    if (
        record_age is not None
        and spoken_age is not None
        and abs(record_age - spoken_age) > 1
    ):
        warnings.append(
            "العمر المذكور في التسجيل لا يطابق العمر المحسوب من سجل المريضة "
            f"({spoken_age} مقابل {record_age} سنة)؛ يرجى التحقق قبل اعتماد التقرير."
        )

    previous_ga = info.get("gestational_age_weeks_before_visit")
    current_ga = info.get("gestational_age_weeks")
    try:
        if (
            previous_ga is not None
            and current_ga is not None
            and float(previous_ga) > float(current_ga) + 1.0
        ):
            warnings.append(
                "عمر الحمل المذكور في هذه الزيارة أقل من العمر المسجل قبلها "
                f"({int(float(current_ga))} مقابل {int(float(previous_ga))} أسبوعاً)؛ "
                "قد يكون ذلك تصحيحاً لتأريخ الحمل أو عدم اتساق في البيانات، ويرجى مراجعته."
            )
    except (TypeError, ValueError):
        pass

    return warnings
