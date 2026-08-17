"""Label metadata: Arabic display names + SOAP grouping.

- MEDICAL_SECTIONS maps each fine-grained classifier label to its Arabic display name.
- SOAP_MAPPING groups those labels into the four canonical SOAP sections used to
  structure the final report. This table is derived from sampling data.jsonl per
  label; it is deliberately isolated and well-commented so a clinician can adjust it
  WITHOUT touching pipeline code.  --> needs domain-expert sign-off.

**Active label set: 20** (was 21). `pregnancy_nutrition` was merged into `nutrition`
on 2026-07-26: only 33% of its sentences mentioned pregnancy at all, and 87 texts
carried both labels — it described the PATIENT's context, not the sentence, so it was
not learnable from text. Pregnancy relevance now comes from the patient record
(PLAN_V2 P2-06). Rationale and evidence: data/LABELING_GUIDE.md §2.

Deprecated labels stay in both tables (see DEPRECATED_LABELS) so reports produced by
an older checkpoint keep rendering in the right SOAP section instead of silently
falling back to `assessment`.
"""
from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

# Labels no longer produced by training, kept only so historical reports still map
# correctly. label -> the label that replaced it.
DEPRECATED_LABELS = {"pregnancy_nutrition": "nutrition"}

# --- fine-grained labels -> Arabic display names ---
MEDICAL_SECTIONS = {
    "symptom": "الأعراض",
    "diagnosis": "التشخيص",
    "medication": "الأدوية والعلاجات",
    "history": "التاريخ المرضي",
    "allergy": "الحساسيات",
    "lab": "التحاليل المخبرية",
    "vital": "العلامات الحيوية",
    "plan": "الخطة العلاجية",
    "procedure": "الإجراءات الطبية",
    "info": "معلومات عامة",
    "treatment": "العلاج",
    "follow_up": "المتابعة",
    "nutrition": "التغذية",
    "pregnancy_nutrition": "التغذية أثناء الحمل",  # deprecated -> nutrition
    "pregnancy_risk": "مخاطر الحمل",
    "cardiology": "القلب",
    "neurology": "الأعصاب",
    "gynecology": "أمراض نسائية",
    "infection": "العدوى",
    "postpartum": "ما بعد الولادة",
    "emergency": "الطوارئ",
}

# --- SOAP section keys + Arabic titles ---
SOAP_SUBJECTIVE = "subjective"
SOAP_OBJECTIVE = "objective"
SOAP_ASSESSMENT = "assessment"
SOAP_PLAN = "plan"

SOAP_ORDER = [SOAP_SUBJECTIVE, SOAP_OBJECTIVE, SOAP_ASSESSMENT, SOAP_PLAN]

SOAP_TITLES_AR = {
    SOAP_SUBJECTIVE: "الشكوى والتاريخ (Subjective)",
    SOAP_OBJECTIVE: "الفحص والقياسات (Objective)",
    SOAP_ASSESSMENT: "التقييم والتشخيص (Assessment)",
    SOAP_PLAN: "الخطة والعلاج (Plan)",
}

# --- 21 labels -> SOAP section (derived from data.jsonl samples; review before final) ---
SOAP_MAPPING = {
    # Subjective: patient-reported complaints / history / context
    "symptom": SOAP_SUBJECTIVE,
    "history": SOAP_SUBJECTIVE,
    "allergy": SOAP_SUBJECTIVE,
    "info": SOAP_SUBJECTIVE,
    "cardiology": SOAP_SUBJECTIVE,
    "neurology": SOAP_SUBJECTIVE,
    "gynecology": SOAP_SUBJECTIVE,
    # Objective: measured / observed / performed
    "vital": SOAP_OBJECTIVE,
    "lab": SOAP_OBJECTIVE,
    "procedure": SOAP_OBJECTIVE,
    # Assessment: diagnostic / risk conclusions
    "diagnosis": SOAP_ASSESSMENT,
    "pregnancy_risk": SOAP_ASSESSMENT,
    "emergency": SOAP_ASSESSMENT,
    "postpartum": SOAP_ASSESSMENT,
    "infection": SOAP_ASSESSMENT,
    # Plan: what to do next
    "plan": SOAP_PLAN,
    "treatment": SOAP_PLAN,
    "medication": SOAP_PLAN,
    "follow_up": SOAP_PLAN,
    "nutrition": SOAP_PLAN,
    "pregnancy_nutrition": SOAP_PLAN,  # deprecated: kept so old reports still land in Plan
}

# The label set a freshly trained model can actually emit (20).
ACTIVE_LABELS = [lbl for lbl in MEDICAL_SECTIONS if lbl not in DEPRECATED_LABELS]


def soap_for_label(label: str) -> str:
    """Return the SOAP section for a fine-grained label.

    Falls back to "assessment" (with a logged warning) for any label missing
    from SOAP_MAPPING, so a future retrain that introduces a new label never
    silently drops items from the report.
    """
    section = SOAP_MAPPING.get(label)
    if section is None:
        logger.warning("Label %r has no SOAP mapping; defaulting to 'assessment'.", label)
        return SOAP_ASSESSMENT
    return section


def label_ar(label: str) -> str:
    """Arabic display name for a label, falling back to the raw label."""
    return MEDICAL_SECTIONS.get(label, label)


# Which section wins when one sentence carries several labels. A sentence like
# "الضغط 150/90 وعندها صداع، سنبدأ بأسبرين" is genuinely Objective + Subjective + Plan;
# it has to live somewhere, and the most actionable section is the safest home — a plan
# item missed because it was filed under Subjective is the costlier error.
SOAP_PRIORITY = [SOAP_PLAN, SOAP_ASSESSMENT, SOAP_OBJECTIVE, SOAP_SUBJECTIVE]


def soap_for_labels(labels: list[str]) -> tuple[str, list[str]]:
    """Primary SOAP section for a multi-label item, plus the other sections it spans.

    Returns (primary, also_in). `also_in` is what lets a reader see that an item was
    filed in one section but belongs to others too, instead of that information being
    silently discarded (P4-09/P4-10).
    """
    if not labels:
        return SOAP_ASSESSMENT, []
    sections = {soap_for_label(label) for label in labels}
    for candidate in SOAP_PRIORITY:
        if candidate in sections:
            primary = candidate
            break
    else:  # pragma: no cover - SOAP_PRIORITY covers every section
        primary = SOAP_ASSESSMENT
    return primary, [s for s in SOAP_ORDER if s in sections and s != primary]


# Explicit action-language routing.  The classifier label is preserved; this helper
# only decides where the sentence belongs in SOAP.  Spoken Arabic often phrases a
# plan as a need/recommendation ("بحاجة لتقييم عاجل") or a monitoring instruction
# ("مراقبة قريبة للأم والجنين"), which can resemble risk/procedure training rows.
_PLAN_MODAL_RE = re.compile(
    r"(?:بحاجة|بيحاجة|بحاجه|تحتاج|يحتاج|لازم|يجب|ينبغي|نوصي|أوصي|اوصي|ننصح|"
    r"نصحنا|يفضل|يفضّل|سوف|رح|سنقوم|سنبدأ|سنكمل)"
)
_PLAN_ACTION_RE = re.compile(
    r"(?:تقييم|تقويم|تقنيم|مراقبة|متابعة|مراجعة|تحويل|إحالة|احالة|دخول|"
    r"المستشفى|مستشفى|فحص|تحاليل|علاج|دواء|إعادة|اعادة)"
)
_PLAN_DIRECT_RE = re.compile(
    r"(?:تقييم|تقنيم|تقويم)\s+(?:(?:طبي|طبيّة|طبية)\s+)?(?:عاجل|سريع|فوري)|"
    r"مراقبة\s+(?:قريبة|مستمرة)|"
    r"تحويل\s+عاجل|إحالة\s+عاجلة|احالة\s+عاجلة"
)
_PAST_PROCEDURE_RE = re.compile(
    r"^\s*(?:تم|تمت|أجرينا|اجرينا|أُجري|اجري|عملنا|قمنا)\b"
)




_PATIENT_REPORTED_RE = re.compile(
    r"(?:حسب\s+(?:كلامها|قولها|إفادتها|افادتها)|تقول|ذكرت|تحكي|تشعر|تحس)"
)
_NORMAL_FETAL_MOVEMENT_RE = re.compile(
    r"حركة\s+الجنين.{0,20}(?:جيدة|جيده|طبيعية|طبيعيه|منيحة|منيحه|موجودة|موجوده)"
)
_EXAM_FINDING_RE = re.compile(r"^\s*(?:بالفحص|عند\s+الفحص|بالفحص\s+السريري)")


def _has_context_code(entity_links, *codes: str) -> bool:
    wanted = set(codes)
    return any(
        isinstance(link, dict)
        and str(link.get("kind", "")) == "context"
        and str(link.get("code", "")) in wanted
        for link in (entity_links or [])
    )


def _has_structured_code(entity_links, *codes: str) -> bool:
    wanted = set(codes)
    return any(
        isinstance(link, dict)
        and str(link.get("code", "")) in wanted
        and str(link.get("assertion", "present")) == "present"
        for link in (entity_links or [])
    )


def _has_present_symptom(entity_links) -> bool:
    return any(
        isinstance(link, dict)
        and str(link.get("kind", "")) == "symptom"
        and str(link.get("assertion", "present")) == "present"
        for link in (entity_links or [])
    )


def _all_structured_clinical_findings_non_actionable(entity_links) -> bool:
    """True when structured findings exist but none describes a current finding.

    This is the guard for a very common OB/GYN sentence: "ما في نزيف ولا ألم شديد".
    AraBERT can correctly recognise the *topic* as emergency while the assertions say
    the danger signs are absent.  SOAP should record that as patient-reported negative
    history, not as an Assessment emergency.
    """
    clinical_kinds = {"symptom", "condition", "vital", "lab", "clinical"}
    links = [
        link for link in (entity_links or [])
        if isinstance(link, dict) and str(link.get("kind", "")) in clinical_kinds
    ]
    return bool(links) and not any(
        str(link.get("assertion", "present")) == "present" for link in links
    )

def text_implies_plan(text: str) -> bool:
    """Return True when the sentence is clearly a next-action instruction.

    This is intentionally conservative: completed procedures stay Objective, while
    explicit needs/recommendations/future monitoring are routed to Plan.  Common ASR
    variants from real recordings (e.g. ``تقنيم`` for ``تقييم``) are included.
    """
    normalized = str(text or "").strip()
    if not normalized or _PAST_PROCEDURE_RE.search(normalized):
        return False
    if _PLAN_DIRECT_RE.search(normalized):
        return True
    # A plan modal must govern an action that comes *after* it in the same local
    # phrase.  The old independent searches misrouted sentences such as
    # "عند الفحص ... لازم نفكر بعدوى" because "فحص" occurred before "لازم".
    for modal in _PLAN_MODAL_RE.finditer(normalized):
        tail = normalized[modal.end(): modal.end() + 56]
        if _PLAN_ACTION_RE.search(tail):
            return True
    return False


def soap_for_item(labels: list[str], text: str, entity_links=None) -> tuple[str, list[str]]:
    """SOAP routing that preserves labels but adds assertion/context safeguards.

    Classifier labels describe semantic topics; SOAP sections describe clinical role.
    Those are not identical.  A negated emergency topic is Subjective history, and a
    patient-reported statement that fetal movement is good is Subjective even when the
    sentence classifier calls it follow-up.
    """
    primary, also_in = soap_for_labels(labels)

    # Discourse-level plan/contingency markers outrank a sentence-level classifier.
    # They describe the role of this fragment inside the clinician's connected plan.
    if _has_context_code(
        entity_links, "plan_continuation", "contingency_condition", "contingency_action"
    ):
        old_primary = primary
        primary = SOAP_PLAN
        also = set(also_in)
        if old_primary != SOAP_PLAN:
            also.add(old_primary)
        also.discard(SOAP_PLAN)
        return primary, [s for s in SOAP_ORDER if s in also]

    # Explicit next-action language has the highest routing priority.
    if text_implies_plan(text) and primary != SOAP_PLAN:
        old_primary = primary
        primary = SOAP_PLAN
        also = set(also_in)
        also.add(old_primary)
        also.discard(SOAP_PLAN)
        also_in = [s for s in SOAP_ORDER if s in also]
        return primary, also_in

    # Examination findings belong in Objective even if AraBERT chooses `diagnosis`.
    # Cervical dilation is the canonical labour example: it is a measured exam value,
    # not the assessment itself.
    if (
        _EXAM_FINDING_RE.search(str(text or ""))
        or _has_context_code(entity_links, "exam_finding")
        or _has_structured_code(entity_links, "cervical_dilation_cm")
    ):
        old_primary = primary
        primary = SOAP_OBJECTIVE
        also = set(also_in)
        if old_primary != SOAP_OBJECTIVE:
            also.add(old_primary)
        also.discard(SOAP_OBJECTIVE)
        return primary, [s for s in SOAP_ORDER if s in also]

    label_set = set(labels or [])
    normalized = str(text or "")

    # "ما في نزيف ..." may be classified emergency by topic, but all structured
    # danger findings are explicitly absent.  File it under Subjective and retain
    # Assessment as provenance rather than creating an apparent diagnosis.
    if "emergency" in label_set and _all_structured_clinical_findings_non_actionable(entity_links):
        old_primary = primary
        primary = SOAP_SUBJECTIVE
        also = set(also_in)
        if old_primary != SOAP_SUBJECTIVE:
            also.add(old_primary)
        also.discard(SOAP_SUBJECTIVE)
        also_in = [s for s in SOAP_ORDER if s in also]
        return primary, also_in

    # Patient-reported postpartum symptoms remain Subjective even when the classifier
    # uses the broad `postpartum` topic label.  Diagnostic interpretation belongs in
    # Assessment; the woman's report of abnormal lochia belongs in her history.
    if (
        "postpartum" in label_set
        and _PATIENT_REPORTED_RE.search(normalized)
        and _has_present_symptom(entity_links)
    ):
        old_primary = primary
        primary = SOAP_SUBJECTIVE
        also = set(also_in)
        if old_primary != SOAP_SUBJECTIVE:
            also.add(old_primary)
        also.discard(SOAP_SUBJECTIVE)
        return primary, [s for s in SOAP_ORDER if s in also]

    # Normal fetal movement stated by the patient is a current subjective finding,
    # not a future follow-up action.  Preserve the model label for auditability.
    if (
        "follow_up" in label_set
        and (_PATIENT_REPORTED_RE.search(normalized) or _NORMAL_FETAL_MOVEMENT_RE.search(normalized))
        and "حركة" in normalized and "الجنين" in normalized
    ):
        old_primary = primary
        primary = SOAP_SUBJECTIVE
        also = set(also_in)
        if old_primary != SOAP_SUBJECTIVE:
            also.add(old_primary)
        also.discard(SOAP_SUBJECTIVE)
        also_in = [s for s in SOAP_ORDER if s in also]

    return primary, also_in
