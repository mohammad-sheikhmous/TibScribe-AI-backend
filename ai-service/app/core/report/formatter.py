"""Deterministic, fact-first SOAP presentation formatter (P11).

The formatter does not ask a language model to invent a report.  It renders structured
entities/assertions into standard Arabic clinical phrases and falls back to the source
item only when the pipeline has no structured fact for that content.  This keeps the
presentation readable while preserving the exact clinical decision source.
"""
from __future__ import annotations

import re
from typing import Iterable, Mapping

from ..nlp.extraction import display_name
from ..nlp.sections import SOAP_ORDER
from .schema import FormattedSoapSection, ReportItem, ReportSection

_WS = re.compile(r"\s+")
_TRAILING = re.compile(r"[\s،؛;,.]+$")
_AGE_RE = re.compile(r"(?:عمرها|عمره|العمر)\D{0,6}?(\d{1,3})\s*(?:سنة|سنه)")
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
    name = "structured-facts-v2"

    def format(self, soap: Mapping[str, ReportSection]) -> dict[str, FormattedSoapSection]:
        warnings = _consistency_warnings(soap)
        formatted: dict[str, FormattedSoapSection] = {}
        for key in SOAP_ORDER:
            if key not in soap:
                continue
            formatted[key] = self.format_section(
                soap[key], warnings=warnings if key == "subjective" else []
            )
        return formatted

    def format_section(
        self, section: ReportSection, *, warnings: list[str] | None = None
    ) -> FormattedSoapSection:
        clauses: list[str] = []
        item_ids: list[str] = []
        seen: set[str] = set()
        safety_codes: list[str] = []
        safety_item_ids: list[str] = []
        if section.soap_key == "plan":
            for item in sorted(section.items, key=lambda row: row.order_index):
                for link in _links(item):
                    if link.get("kind") == "symptom" and link.get("assertion") == "hypothetical":
                        name = display_name(str(link.get("code")))
                        if name and name not in safety_codes:
                            safety_codes.append(name)
                            safety_item_ids.append(item.item_id)

        for item in sorted(section.items, key=lambda row: row.order_index):
            links = _links(item)
            # Safety-netting fragments are often split acoustically across two/three
            # Whisper segments.  Once their hypothetical findings are structured, do
            # not echo the broken raw fragments; a single polished clause is emitted
            # below from the structured facts.
            if section.soap_key == "plan" and safety_codes:
                hypothetical_symptoms = [
                    link for link in links
                    if link.get("kind") == "symptom"
                    and link.get("assertion") == "hypothetical"
                ]
                if hypothetical_symptoms:
                    continue

            item_clauses = self._format_item(item, section.soap_key)
            for clause in item_clauses:
                clause = _clean_clause(clause)
                if not clause:
                    continue
                # A trailing acoustic fragment like "لازم تراجعني" adds no new fact
                # when the section already contains structured safety-netting symptoms.
                if safety_codes and not _links(item) and re.search(r"(?:لازم|يجب).{0,16}(?:تراجع|مراجعة)", item.text):
                    continue
                key = clause.casefold()
                if key in seen:
                    continue
                seen.add(key)
                clauses.append(clause + ".")
                item_ids.append(item.item_id)

        if safety_codes:
            safety_clause = f"تم شرح علامات الخطر التي تستدعي المراجعة الفورية: {_join_ar(safety_codes)}."
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

    def _format_item(self, item: ReportItem, section_key: str) -> list[str]:
        if section_key == "objective":
            clauses = _objective_clauses(item)
        elif section_key == "plan":
            clauses = _plan_clauses(item)
        elif section_key == "subjective":
            clauses = _subjective_clauses(item)
        else:
            clauses = _assessment_clauses(item)

        if clauses:
            return clauses
        fallback = _demographic_clause(item.text) or _clean_clause(item.text)
        return [fallback] if fallback else []


def _subjective_clauses(item: ReportItem) -> list[str]:
    links = _dedupe_links(_links(item))
    symptoms = [x for x in links if x.get("kind") == "symptom"]
    def symptom_name(link: dict) -> str:
        code = str(link.get("code"))
        if code == "edema":
            return "تورم بالقدمين" if re.search(r"قدم|رجل|ساق", item.text) else "وذمة/تورم"
        return display_name(code)

    present = [symptom_name(x) for x in symptoms if x.get("assertion", "present") == "present"]
    absent = [symptom_name(x) for x in symptoms if x.get("assertion") == "absent"]
    historical = [display_name(str(x.get("code"))) for x in symptoms if x.get("assertion") == "historical"]
    hypothetical = [display_name(str(x.get("code"))) for x in symptoms if x.get("assertion") == "hypothetical"]

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
            suffix = " (طبيعي)" if link.get("status") == "normal" else ""
            phrase = f"الهيموغلوبين {_num(link.get('value'))} غ/دل{suffix}"
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
    present_conditions = [
        display_name(str(x.get("code"))) for x in links
        if x.get("kind") == "condition" and x.get("assertion", "present") == "present"
    ]
    absent_symptoms = [
        display_name(str(x.get("code"))) for x in links
        if x.get("kind") == "symptom" and x.get("assertion") == "absent"
    ]
    if present_conditions:
        clauses.append(f"التقييم يتضمن {_join_ar(present_conditions)}")
    if absent_symptoms:
        rendered_absent = ["انقباضات رحمية" if name == display_name("contractions") else name for name in absent_symptoms]
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
        ("الحديد" if str(x.get("code")) == "iron" else display_name(str(x.get("code"))))
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


def _demographic_clause(text: str) -> str:
    cleaned = _clean_clause(text)
    match = re.search(
        r"مريض(?:ة)?\s+(.+?)\s+عمرها\s+(\d{1,3})\s+سنة\s+حامل\s+(?:في|ب)?ال?اسبوع\s+(\d{1,2})",
        cleaned,
        re.I,
    )
    if not match:
        return ""
    name, age, ga = match.groups()
    return f"المريضة {name}، العمر {age} سنة، حامل في الأسبوع {ga}"


def _consistency_warnings(soap: Mapping[str, ReportSection]) -> list[str]:
    """Cross-item contradictions that are impossible, not merely improbable."""
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
                f"تعارض زمني محتمل في النص المنسوخ: العمر {age} سنة، بينما ذُكرت ولادة سابقة قبل {years} سنة؛ راجع المقطع الصوتي قبل اعتماد المعلومة."
            )
    return warnings
