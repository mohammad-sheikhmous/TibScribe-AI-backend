"""Cross-segment obstetric discourse enrichment.

Whisper segmentation is acoustic, not clinical.  A single clinician statement can be
split into fragments such as::

    تقلصات منتظمة ...
    كل 5 دقايق وصارت أقوى ...

or::

    إذا صار تغير بنبض الجنين ...
    أو توقف تقدم المخاض ...
    وقتها لازم نعيد تقييم الخطة

This module adds only *deterministic discourse facts/context markers*.  It never
changes the AraBERT label or classifier confidence.  That keeps model provenance
intact while allowing SOAP routing and the KBS to understand the connected utterance.
"""
from __future__ import annotations

import re
from typing import Iterable

from ..report.schema import ClassifiedSegment

_CONTRACTION_INTERVAL_RE = re.compile(
    r"كل\s+(\d{1,2}(?:[.,]\d+)?)\s*(?:دقايق|دقائق|دقيقة|دقيقه)", re.I
)
_STRENGTHENING_RE = re.compile(
    r"(?:صار(?:ت)?\s+أ?قوى|صارت\s+اقوى|أ?قوى\s+من\s+قبل|"
    r"زادت\s+(?:قوة|شدة)|عم\s+تقوى|بتقوى)", re.I
)
_AMNIOTIC_RE = re.compile(
    r"(?:مي|ماء)\s+(?:الجنين|الرأس|الراس)|السائل\s+الأ?منيوسي|ماء\s+الأغشية",
    re.I,
)
_SUSPICION_RE = re.compile(
    r"(?:بتشك|تشك|نشك|اشتباه|مشتبه|يمكن|ممكن|غالبا|غالباً)", re.I
)
_LABOUR_ANCHOR_RE = re.compile(
    r"(?:تقلصات|طلق|انقباضات|عنق\s+الرحم|مخاض|مخاط\s+فعال|تقدم\s+المخاط)", re.I
)
_MONITORING_RE = re.compile(
    r"(?:نبض\s+الجنين|حالة\s+الأم|حاله\s+الام|مراقبة|مراقبه|بشكل\s+مستمر|"
    r"تقدم\s+(?:المخاض|المخاط))",
    re.I,
)
_CONDITIONAL_START_RE = re.compile(r"^\s*(?:و)?(?:إذا|اذا|في\s+حال|لو)\b", re.I)
_CONDITIONAL_CONT_RE = re.compile(r"^\s*(?:أو|او|و)\b", re.I)
_CONSEQUENCE_RE = re.compile(
    r"^\s*(?:"
    r"وقتها|عندها|حينها|ساعتها|"
    r"صار\s+(?:في\s+)?(?:أحد|احد|واحد)\s+"
    r"(?:من\s+)?(?:هاي|هذه)?\s*(?:الأعراض|الاعراض)"
    r")\b"
    r"|(?:لازم|يجب|ضروري).{0,20}"
    r"(?:تراجع|مراجعة|طوارئ)",
    re.I,
)

_DANGER_SIGN_CONTINUATION_CODES = {
    "headache",
    "blurred_vision",
    "vaginal_bleeding",
    "reduced_fetal_movement",
    "severe_abdominal_pain",
    "abdominal_pain",
    "shortness_of_breath",
    "convulsions",
}

_EXPLICIT_CURRENT_RE = re.compile(
    r"(?:حاليا|حالياً|الآن|الان|هلا|اليوم|"
    r"بهاللحظة|بهذه\s+اللحظة)",
    re.I,
)

def _tight_acoustic_continuation(
    previous: ClassifiedSegment,
    current: ClassifiedSegment,
) -> bool:
    try:
        gap = float(current.start_sec) - float(previous.end_sec)
    except (TypeError, ValueError):
        return True

    return -0.25 <= gap <= 1.5


def _looks_like_conditional_enumeration(
    segment: ClassifiedSegment,
    text: str,
) -> bool:

    # لا نحول جملة explicitly current إلى hypothetical.
    if _EXPLICIT_CURRENT_RE.search(text):
        return False

    danger_links = [
        link
        for link in (segment.entity_links or [])
        if isinstance(link, dict)
        and str(link.get("kind")) == "symptom"
        and str(link.get("code"))
        in _DANGER_SIGN_CONTINUATION_CODES
        and str(link.get("assertion", "present"))
        in {"present", "planned", "hypothetical"}
    ]

    # قائمة من علامتي خطر أو أكثر داخل condition مفتوحة.
    if len(danger_links) >= 2:
        return True

    # Whisper قد يضع "إذا" في نهاية القطعة.
    if danger_links and re.search(
        r"(?:إذا|اذا|لو)\s*$",
        text,
    ):
        return True

    return False

def _links(segment: ClassifiedSegment) -> list[dict]:
    if segment.entity_links is None:
        segment.entity_links = []
    return segment.entity_links  # type: ignore[return-value]


def _has_code(segment: ClassifiedSegment, code: str, *, actionable_only: bool = False) -> bool:
    for link in segment.entity_links or []:
        if not isinstance(link, dict) or str(link.get("code", "")) != code:
            continue
        if not actionable_only or str(link.get("assertion", "present")) == "present":
            return True
    return False


def _append(segment: ClassifiedSegment, payload: dict) -> None:
    links = _links(segment)
    key = (payload.get("kind"), payload.get("code"), payload.get("value"), payload.get("status"))
    for current in links:
        if not isinstance(current, dict):
            continue
        current_key = (
            current.get("kind"), current.get("code"), current.get("value"), current.get("status")
        )
        if current_key == key:
            return
    links.append(payload)


def _context(segment: ClassifiedSegment, code: str, note: str | None = None) -> None:
    payload = {
        "kind": "context",
        "code": code,
        "assertion": "present",
        "confidence": 1.0,
        "extractor": "discourse",
        "extractor_version": "labour-discourse-1.0",
    }
    if note:
        payload["note"] = note
    _append(segment, payload)


def _segment_text(segment: ClassifiedSegment, use_canonical_text: bool) -> str:
    return str(segment.effective_text if use_canonical_text else segment.text or "")


def _recent_labour_anchor(segments: list[ClassifiedSegment], index: int, use_canonical_text: bool) -> bool:
    for candidate in segments[max(0, index - 2): index + 1]:
        if _LABOUR_ANCHOR_RE.search(_segment_text(candidate, use_canonical_text)):
            return True
        if _has_code(candidate, "contractions", actionable_only=True):
            return True
        if _has_code(candidate, "cervical_dilation_cm", actionable_only=True):
            return True
    return False


def enrich_cross_segment_context(
    segments: Iterable[ClassifiedSegment], *, use_canonical_text: bool = False
) -> list[ClassifiedSegment]:
    """Add cross-segment facts using the same text source as the clinical decision path."""
    ordered = sorted(list(segments), key=lambda s: s.order_index)
    conditional_open = False
    previous_was_plan_monitoring = False
    labour_open = False

    for index, segment in enumerate(ordered):
        text = _segment_text(segment, use_canonical_text).strip()
        if _recent_labour_anchor(ordered, index, use_canonical_text):
            labour_open = True
        labour_context = labour_open

        # 1) Acoustic split: the contraction noun can be in the previous segment while
        #    frequency/strength live here.  Anchor the numeric fact only in labour
        #    context, so unrelated "كل 5 دقائق" statements remain untouched.
        interval = _CONTRACTION_INTERVAL_RE.search(text)
        if interval and labour_context:
            _append(segment, {
                "kind": "clinical",
                "code": "contraction_interval_min",
                "assertion": "present",
                "value": float(interval.group(1).replace(",", ".")),
                "unit": "min",
                "matched_text": interval.group(0),
                "confidence": 1.0,
                "extractor": "discourse",
                "extractor_version": "labour-discourse-1.0",
                "note": "linked to recent contraction mention",
            })
        if _STRENGTHENING_RE.search(text) and labour_context:
            _append(segment, {
                "kind": "clinical",
                "code": "contractions_strengthened",
                "assertion": "present",
                "confidence": 1.0,
                "extractor": "discourse",
                "extractor_version": "labour-discourse-1.0",
                "note": "strength statement linked to recent contractions",
            })

        # 2) Fluid leakage + next-segment suspicion that it is amniotic fluid.  The
        #    SOAP entity is explicit about uncertainty; KBS adapter keeps it separate
        #    from confirmed rupture_of_membranes rules.
        if _AMNIOTIC_RE.search(text) and labour_context:
            recent_fluid = any(
                _has_code(prev, "vaginal_fluid_leak", actionable_only=True)
                for prev in ordered[max(0, index - 2):index]
            )
            if recent_fluid or "سائل" in text:
                suspected = bool(_SUSPICION_RE.search(text))
                _append(segment, {
                    "kind": "symptom",
                    "code": "water_breaking",
                    "assertion": "present",
                    "status": "suspected" if suspected else "confirmed",
                    "matched_text": _AMNIOTIC_RE.search(text).group(0),
                    "confidence": 1.0,
                    "extractor": "discourse",
                    "extractor_version": "labour-discourse-1.0",
                    "note": (
                        "suspected amniotic-fluid leakage across adjacent segments"
                        if suspected else "amniotic-fluid leakage linked across adjacent segments"
                    ),
                })

        # 3) Physical examination/cervical findings are Objective regardless of a
        #    diagnosis-like classifier label.
        if text.startswith(("بالفحص", "بالفحص", "عند الفحص", "بالفحص السريري")):
            _context(segment, "exam_finding")
        if _has_code(segment, "cervical_dilation_cm"):
            _context(segment, "exam_finding", "cervical dilation is an examination finding")

        # 4) Preserve a monitoring plan across an acoustic split beginning with و...
        current_monitoring = bool(_MONITORING_RE.search(text))
        if (
            index > 0
            and previous_was_plan_monitoring
            and text.startswith("و")
            and current_monitoring
        ):
            _context(segment, "plan_continuation", "continues monitoring plan from prior segment")
            # A bare "نبض الجنين" is a physiologic measurement, not automatically a
            # CTG.  Only promote it to the planned CTG test when discourse proves this
            # fragment continues an explicit monitoring plan across the ASR split.
            if "نبض الجنين" in text and not _has_code(segment, "ctg"):
                _append(segment, {
                    "kind": "test",
                    "code": "ctg",
                    "assertion": "planned",
                    "confidence": 1.0,
                    "extractor": "discourse",
                    "extractor_version": "labour-discourse-1.0",
                    "note": "fetal-heart monitoring inherited from prior plan segment",
                })
            # Entity extraction saw this fragment before discourse knew it continued
            # a future monitoring plan.  Re-scope current structured actions/tests as
            # planned so the KBS cannot mistake them for already completed tests.
            for link in segment.entity_links or []:
                if not isinstance(link, dict) or str(link.get("kind")) == "context":
                    continue
                if str(link.get("assertion", "present")) == "present":
                    link["assertion"] = "planned"
                    note = str(link.get("note") or "").strip()
                    link["note"] = (note + "; " if note else "") + "inherits monitoring-plan scope"

        # 5) Conditional scope survives one or more continuation fragments until the
        #    consequence/action segment.  Existing entities in those fragments are
        #    downgraded to hypothetical rather than promoted to current findings.
        if _CONDITIONAL_START_RE.search(text):
            conditional_open = True
            _context(segment, "contingency_condition")
        elif (
            conditional_open
            and _CONSEQUENCE_RE.search(text)
        ):
            _context(segment,"contingency_action")
            conditional_open = False
        elif (
            conditional_open
            and _CONDITIONAL_CONT_RE.search(text)
        ):
            _context(
                segment,
                "contingency_condition",
                "inherits conditional scope from prior segment"
            )
        elif (
            conditional_open
            and index > 0
            and _tight_acoustic_continuation(
                ordered[index - 1],
                segment
            )
            and _looks_like_conditional_enumeration(
                segment,
                text
            )
        ):
            _context(
                segment,
                "contingency_condition",
                "inherits conditional danger-sign enumeration "
                "from prior segment"
            )
        elif (
            conditional_open
            and not text.startswith(
                ("و", "او", "أو")
            )
        ):

            conditional_open = False
        if any(
            isinstance(link, dict) and str(link.get("kind")) == "context"
            and str(link.get("code")) == "contingency_condition"
            for link in segment.entity_links or []
        ):
            for link in segment.entity_links or []:
                if not isinstance(link, dict) or str(link.get("kind")) == "context":
                    continue
                if str(link.get("assertion", "present")) in {"present", "planned"}:
                    link["assertion"] = "hypothetical"
                    note = str(link.get("note") or "").strip()
                    link["note"] = (note + "; " if note else "") + "inherited conditional scope"

        # A segment is a plan anchor if the classifier already called it plan and it
        # describes monitoring, or the discourse marker explicitly says it continues
        # a prior plan.
        previous_was_plan_monitoring = bool(
            (segment.label == "plan" and current_monitoring)
            or any(
                isinstance(link, dict) and str(link.get("code")) == "plan_continuation"
                for link in segment.entity_links or []
            )
        )

    return ordered
