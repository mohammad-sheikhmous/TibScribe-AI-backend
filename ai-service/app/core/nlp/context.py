"""Patient-context detection: pregnancy, postpartum, gestational age (P2-06).

This is the text-side half of the fix for gap ك-٤. Two failures it is built to avoid,
both measured on the real system:

  * `المريضة ليست حامل حالياً` used to mark the patient PREGNANT — the old check was a
    plain substring search with no negation handling. Every lookup here goes through
    `assertion.find_negation_aware`.
  * A general dietary tip used to prove pregnancy, because the classifier label
    `pregnancy_nutrition` was treated as evidence. That label no longer exists (merged
    into `nutrition`, ADR-005) and **labels are not used as context evidence here at
    all**: context comes from what was said, plus the patient record.

The other half of the fix — pregnancy not lasting forever — lives in the state machine
(`app/db/patient_state.py`), because expiry is a property of time, not of text.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable, Optional

from .assertion import find_negation_aware

_ARABIC_INDIC = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")

PREGNANT_MARKERS = ("حامل", "الحامل", "حبلى", "بالحمل", "حملها", "الحمل الحالي")
POSTPARTUM_MARKERS = (
    "بعد الولادة", "ما بعد الولادة", "بعد الوضع", "النفاس", "نفاسية",
    "بعد القيصرية", "بعد العملية القيصرية", "ولدت", "وضعت مولودها",
)

GA_WEEK_RE = re.compile(
    r"(?:بالأسبوع|بالاسبوع|الأسبوع|الاسبوع|أسبوع|اسبوع)\s*(?:ال\s*)?(\d{1,2})"
)
MONTH_RE = re.compile(r"بالشهر\s+(ال\w+)")
TRIMESTER_RE = re.compile(r"الثلث\s+(ال\w+)")

MONTH_TO_WEEKS = {
    "الأول": 4, "الاول": 4, "الثاني": 8, "التاني": 8, "الثالث": 13, "الرابع": 17,
    "الخامس": 22, "السادس": 26, "السابع": 30, "الثامن": 35, "التاسع": 39,
}
TRIMESTER_TO_NUM = {"الأول": 1, "الاول": 1, "الثاني": 2, "التاني": 2, "الثالث": 3}

MAX_GA_WEEKS = 44  # anything beyond this is a misread number, not a gestational age


@dataclass
class ContextObservation:
    """What THIS visit's transcript says about the patient's obstetric context.

    `pregnant` is deliberately tri-state:
      True  — asserted pregnant
      False — explicitly denied ("ليست حامل")
      None  — not mentioned; the previous state stands (silence is not evidence)
    """

    pregnant: Optional[bool] = None
    postpartum: bool = False
    ga_weeks: Optional[int] = None
    trimester: Optional[int] = None
    evidence: list[str] = field(default_factory=list)

    @property
    def has_signal(self) -> bool:
        return (
            self.pregnant is not None
            or self.postpartum
            or self.ga_weeks is not None
            or self.trimester is not None
        )


def normalize_digits(text: str) -> str:
    # Whisper and human typing may preserve the Arabic tatweel in forms such as
    # "الأسبوع الـ 28".  It is typographic, not semantic, so remove it before
    # context parsing and normalize Arabic-Indic digits at the same boundary.
    return str(text).translate(_ARABIC_INDIC).replace("ـ", "")


def extract_ga_weeks(text: str) -> Optional[int]:
    """Gestational age in weeks, from an explicit week or a stated month."""
    text = normalize_digits(text)
    match = GA_WEEK_RE.search(text)
    if match:
        weeks = int(match.group(1))
        if 1 <= weeks <= MAX_GA_WEEKS:
            return weeks
    month = MONTH_RE.search(text)
    if month and month.group(1) in MONTH_TO_WEEKS:
        return MONTH_TO_WEEKS[month.group(1)]
    return None


def extract_trimester(text: str) -> Optional[int]:
    match = TRIMESTER_RE.search(normalize_digits(text))
    if match and match.group(1) in TRIMESTER_TO_NUM:
        return TRIMESTER_TO_NUM[match.group(1)]
    return None


def trimester_from_weeks(weeks: int) -> int:
    return 1 if weeks <= 13 else 2 if weeks <= 27 else 3


def detect_context(texts: Iterable[str]) -> ContextObservation:
    """Read obstetric context out of one visit's sentences.

    Sentence-by-sentence rather than over one concatenated blob: joining the transcript
    lets a negation cue at the end of sentence 3 fall inside the window of a term at the
    start of sentence 4.
    """
    observation = ContextObservation()
    denied_pregnancy = False

    for raw in texts:
        text = normalize_digits(str(raw))
        if not text.strip():
            continue

        found, asserted = find_negation_aware(text, PREGNANT_MARKERS)
        if found and asserted:
            observation.pregnant = True
            observation.evidence.append(text)
        elif found:
            denied_pregnancy = True
            observation.evidence.append(f"(منفي) {text}")

        _, postpartum = find_negation_aware(text, POSTPARTUM_MARKERS)
        if postpartum:
            observation.postpartum = True
            observation.evidence.append(text)

        if observation.ga_weeks is None:
            weeks = extract_ga_weeks(text)
            if weeks is not None:
                observation.ga_weeks = weeks
                observation.evidence.append(text)
        if observation.trimester is None:
            observation.trimester = extract_trimester(text)

    # An explicit denial only counts when nothing else asserted pregnancy.
    if observation.pregnant is None and denied_pregnancy:
        observation.pregnant = False

    # A stated gestational age is itself an assertion of pregnancy.
    if observation.ga_weeks is not None and observation.pregnant is None:
        observation.pregnant = True
    if observation.trimester is None and observation.ga_weeks is not None:
        observation.trimester = trimester_from_weeks(observation.ga_weeks)

    return observation
