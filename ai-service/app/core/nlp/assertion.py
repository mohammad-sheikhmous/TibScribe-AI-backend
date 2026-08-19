"""Assertion detection: is a mentioned finding actually asserted about this patient, now?

A gazetteer or an NER model answers "the word صداع appears here". That is not the same
as "this patient has a headache", and the difference is where the false alarms come
from. Measured on the real system, two whole classes of error traced back to this gap:

  * `صداع لا يوجد عند المريضة` — negation AFTER the term; a prefix-only window scored
    it as a present symptom.
  * `إذا صار عندك صداع أو تشوش بالرؤية راجعينا فوراً` — a doctor's safety-netting
    advice, which fired a pre-eclampsia alert (gap ك-٣). Advice of exactly this shape is
    among the most common things said in an OB/GYN consultation, so it was not a rare
    edge case.

Six categories, because they call for different handling downstream:

| assertion    | example                                   | may fire a clinical rule? |
|--------------|-------------------------------------------|---------------------------|
| present      | تشتكي من صداع شديد                        | yes                       |
| absent       | لا يوجد نزيف · صداع لا يوجد                | no (and can DISPROVE)     |
| hypothetical | إذا صار عندك صداع راجعينا                 | no                        |
| historical   | كانت تشتكي من صداع قبل سنة                | context only              |
| family       | أمها عندها سكري                            | risk factor, not a finding|
| planned      | رح نعطيها سلفات المغنيسيوم                 | it is a plan, not a finding|

Scope is bounded by clause terminators (لكن، بس، أما …): in
`لا يوجد نزيف لكن يوجد صداع` the negation must stop at لكن, or it swallows the صداع.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal, Optional

Assertion = Literal["present", "absent", "hypothetical", "historical", "family", "planned"]

# --- trigger vocabularies (data, so a clinician can extend them without code) --------

PRE_NEGATION = (
    "لا يوجد", "لا توجد", "ليس هناك", "ليست هناك", "ما في", "مافي", "ما فيه", "مافيه",
    "بدون", "بلا", "دون", "لا تشتكي", "لا يشتكي", "ما بتشتكي", "ما بيشتكي",
    "ما عندها", "ما عندهاش", "ما عنده", "ليست", "ليس", "لست", "غير",
    "نفت", "نفى", "لم يعد", "ما عاد", "لم تعد", "لا أعاني", "لا اعاني", "ما بعاني",
    "استُبعد", "استبعد", "نستبعد", "لا يوجد لديها", "لا يوجد لديه", "خالية من",
    "لا تعاني", "لا يعاني", "لا تعانين", "لا تشكو", "لا يشكو",
    "سلبي", "سلبية", "لم تلاحظ", "لم يلاحظ", "ما لاحظت",
)

POST_NEGATION = (
    "لا يوجد", "لا توجد", "غير موجود", "غير موجودة", "منفي", "منفية",
    "غير وارد", "مستبعد", "مستبعدة", "نفته", "نفتها", "ما عادت", "زال", "زالت",
)

# Conditional / counterfactual: the finding has NOT happened.
HYPOTHETICAL = (
    "إذا", "اذا", "لو ", "في حال", "بحال", "إن صار", "ان صار", "لما يصير", "عند حدوث",
    "عند الشعور", "ما إن", "ما ان", "احتمال", "يحتمل", "ممكن يصير", "ممكن تحسي",
    "لا سمح الله", "خوفا من", "خوفاً من", "تحسبا", "تحسباً", "للاحتياط",
    "راقبي", "انتبهي إذا", "انتبهي اذا",
)

# The finding belongs to the past, not to this visit.
HISTORICAL = (
    "سابقا", "سابقاً", "بالماضي", "في الماضي", "قبل سنة", "قبل سنتين", "قبل أشهر",
    "قبل شهور", "قبل فترة", "من زمان", "قديما", "قديماً", "كانت تشتكي", "كان عندها",
    "تاريخ مرضي", "تاريخها المرضي", "سبق أن", "سبق ان", "في حملها السابق",
    "بالحمل السابق", "بالولادة السابقة", "سنة الماضية", "العام الماضي",
)

# The finding belongs to a relative, not the patient.
FAMILY = (
    "أمها", "امها", "والدتها", "أختها", "اختها", "أخوها", "اخوها", "جدتها", "خالتها",
    "عمتها", "بالعائلة", "في العائلة", "تاريخ عائلي", "التاريخ العائلي", "أهلها",
    "زوجها", "بنتها", "ابنتها", "أمه", "والده",
)

# An action to be taken, not a finding that exists.
PLANNED = (
    "رح ", "راح ", "سوف", "سنعطي", "سنصف", "سنطلب", "سنجري", "سنحول", "سنراجع",
    "بدنا نعمل", "بدي أعطيك", "بدي اعطيك", "لازم تاخذي", "لازم نعمل", "منعطيك",
    "الخطة", "ننصح", "يُنصح", "ينصح", "سأصف", "ساصف", "خذي", "تاخذي", "راجعينا",
    "موعد", "المتابعة بعد", "نراجع بعد",
)

# A negation's scope stops here — anything after belongs to a new clause.
TERMINATION = ("لكن", "لكنّ", "بس ", "أما", "اما", "غير أن", "إلا أن", "الا ان",
               "،", ".", "؛", "؟", "!")

# Phrases that merely contain a negation particle without negating anything.
PSEUDO_NEGATION = ("لا شك", "لا بد", "لا سيما", "ليس فقط", "لا غنى", "لا يخفى",
                   "ليس فقط", "غير أن")

PRE_WINDOW = 40
POST_WINDOW = 25

_WS = re.compile(r"\s+")


@dataclass(frozen=True)
class AssertionResult:
    assertion: Assertion
    trigger: Optional[str] = None
    clause: str = ""

    @property
    def is_actionable(self) -> bool:
        """May a clinical rule fire on this mention?

        Only `present` describes something true of this patient now. `absent` is
        informative but negative; the rest are about someone else, some other time, or
        something that has not happened.
        """
        return self.assertion == "present"


def _clean(text: str) -> str:
    return _WS.sub(" ", str(text))


def _is_decimal_point(text: str, position: int) -> bool:
    """Return True when ``.`` is numeric punctuation, not a clause terminator.

    ASR/clinical text frequently contains decimal laboratory values such as ``9.5``.
    Treating the decimal point as sentence punctuation truncated the value to ``9``
    before the lab parser saw it.
    """
    return (
        0 < position < len(text) - 1
        and text[position] == "."
        and text[position - 1].isdigit()
        and text[position + 1].isdigit()
    )


def _previous_termination(text: str, marker: str, before: int) -> int:
    """Find the previous real clause terminator, skipping decimal points."""
    position = text.rfind(marker, 0, before)
    while position >= 0 and marker == "." and _is_decimal_point(text, position):
        position = text.rfind(marker, 0, position)
    return position


def _next_termination(text: str, marker: str, after: int) -> int:
    """Find the next real clause terminator, skipping decimal points."""
    position = text.find(marker, after)
    while position >= 0 and marker == "." and _is_decimal_point(text, position):
        position = text.find(marker, position + 1)
    return position


def clause_bounds(text: str, start: int, end: int) -> tuple[int, int]:
    """Absolute [left, right) of the clause containing [start:end].

    Public because the same boundary matters outside assertion detection: reading a lab
    qualifier must not cross لكن either, or "الهيموغلوبين طبيعي لكن الضغط مرتفع" reports
    a high haemoglobin (see extraction._lab_status). Decimal points inside numbers are
    deliberately *not* clause boundaries (``Hb 9.5`` must remain one clause).
    """
    left = 0
    for marker in TERMINATION:
        position = _previous_termination(text, marker, start)
        if position >= 0:
            left = max(left, position + len(marker))
    right = len(text)
    for marker in TERMINATION:
        position = _next_termination(text, marker, end)
        if position >= 0:
            right = min(right, position)
    return left, right


def _clause_around(text: str, start: int, end: int) -> tuple[str, int]:
    """The clause containing [start:end], plus the offset of the mention inside it.

    Clause boundaries are what stop a trigger governing text it has nothing to do with.
    """
    left, right = clause_bounds(text, start, end)
    return text[left:right], start - left


def _first_hit(haystack: str, needles) -> Optional[str]:
    for needle in needles:
        if needle in haystack:
            return needle.strip()
    return None


def is_negated(text: str, start: int, end: int | None = None) -> bool:
    """Is the mention at [start:end] negated, by a cue on either side?"""
    text = _clean(text)
    end = end if end is not None else start
    clause, offset = _clause_around(text, start, end)
    mention_end = offset + (end - start)

    before = clause[max(0, offset - PRE_WINDOW):offset]
    after = clause[mention_end:mention_end + POST_WINDOW]

    if _first_hit(before, PSEUDO_NEGATION):
        return False
    return bool(_first_hit(before, PRE_NEGATION) or _first_hit(after, POST_NEGATION))


def classify_assertion(text: str, start: int, end: int | None = None) -> AssertionResult:
    """Classify one mention into the six categories.

    Precedence is clinical, not lexical:
      1. **family**  — whose body is this about? Everything else is moot if not hers.
      2. **hypothetical** — did it happen at all? An unhappened event cannot be present
         or absent.
      3. **absent**  — it was explicitly denied.
      4. **historical** — it happened, but not now.
      5. **planned** — it is an intended action, not an existing finding.
      6. **present** — the default, and the only one rules may fire on.
    """
    text = _clean(text)
    end = end if end is not None else start
    clause, offset = _clause_around(text, start, end)
    before = clause[:offset]

    trigger = _first_hit(before, FAMILY)
    if trigger:
        return AssertionResult("family", trigger, clause)

    # A conditional cue anywhere in the clause governs it ("راجعينا إذا صار صداع").
    trigger = _first_hit(clause, HYPOTHETICAL)
    if trigger:
        return AssertionResult("hypothetical", trigger, clause)

    if is_negated(text, start, end):
        cue = _first_hit(clause[max(0, offset - PRE_WINDOW):offset], PRE_NEGATION) or \
            _first_hit(clause[offset + (end - start):], POST_NEGATION)
        return AssertionResult("absent", cue, clause)

    trigger = _first_hit(clause, HISTORICAL)
    if trigger:
        return AssertionResult("historical", trigger, clause)

    trigger = _first_hit(before, PLANNED)
    if trigger:
        return AssertionResult("planned", trigger, clause)

    return AssertionResult("present", None, clause)


def find_negation_aware(text: str, phrases: tuple[str, ...] | list[str]) -> tuple[bool, bool]:
    """Search `phrases` in `text`.

    Returns (found, asserted): `asserted` is True only when at least one occurrence is
    actionable — one negated or hypothetical mention must not cancel a separate
    affirmative one.
    """
    cleaned = _clean(text)
    found = False
    for phrase in phrases:
        start = cleaned.find(phrase)
        while start >= 0:
            found = True
            if classify_assertion(cleaned, start, start + len(phrase)).is_actionable:
                return True, True
            start = cleaned.find(phrase, start + 1)
    return found, False
