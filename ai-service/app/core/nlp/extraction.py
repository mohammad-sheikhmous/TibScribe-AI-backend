"""Shared clinical entity extraction (IMPLEMENTATION.md P5-11 … P5-14).

**The cohesion change.** Extraction used to live inside EXPERTA_MED, which meant the
knowledge-based system re-parsed raw text that the pipeline had already processed —
two NLP layers, two vocabularies, and no way for the API to serve structured entities.
It now runs ONCE in the pipeline, is stored in `entity_links` / the `entities` table,
and the KBS consumes the result (see `EXPERTA_MED/extraction.py`, now an adapter).

Three defects from the measured analysis are fixed here:

* **ك-٢/ك-٣** — every mention goes through `assertion.classify_assertion`, so a denied,
  conditional, historical or family mention is labelled as such instead of counting as
  a present finding.
* **ك-٨ (lab values)** — qualifiers are read from a ±30-character window around the
  term. Scanning the whole sentence made "الهيموغلوبين طبيعي لكن الضغط مرتفع" report a
  HIGH haemoglobin.
* **ك-٨ (drug guessing)** — an unknown Latin token in an allergy sentence is no longer
  invented as a drug code (v1 produced entities named "medications" and "allergic").
  It becomes an explicit `unknown_drug` for review.
"""
from __future__ import annotations

import functools
import logging
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional

import yaml

from .assertion import Assertion, classify_assertion, clause_bounds

logger = logging.getLogger(__name__)

LEXICON_DIR = Path(__file__).parent / "lexicon"
VALUE_WINDOW = 30  # characters scanned around a lab term for its qualifier/value
EXTRACTOR_VERSION = "lexicon-1.0"

_ARABIC_INDIC = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")
_LATIN_TOKEN = re.compile(r"[A-Za-z][A-Za-z\-]{3,}")

# English words that appear in clinical sentences but are never drug names. Without
# this list "allergic to Sulfa medications" yields two phantom drugs (v1 did exactly
# that, producing an entity called "medications").
_LATIN_STOPWORDS = {
    "allergic", "allergy", "drug", "drugs", "medication", "medications", "medicine",
    "tablet", "tablets", "capsule", "capsules", "daily", "twice", "once", "dose",
    "doses", "test", "tests", "scan", "result", "results", "normal", "high", "low",
    "blood", "urine", "patient", "history", "with", "without", "from", "this", "that",
    "after", "before", "review", "follow", "check", "week", "weeks", "month", "months",
}

# Numeric patterns. BP also accepts the Levantine short form ("ضغط 11/7" = 110/70).
BP_RE = re.compile(r"(?:ضغط|الضغط)\D{0,15}?(\d{2,3})\s*(?:/|على)\s*(\d{1,3})")
TEMP_RE = re.compile(r"(?:حرارة|الحرارة)\D{0,12}?((?:3[4-9]|4[0-2])(?:[.,]\d)?)")
PULSE_RE = re.compile(r"(?:نبض|النبض|ضربات القلب)\D{0,12}?(\d{2,3})")
NUMBER_NEAR = re.compile(r"(\d{1,3}(?:[.,]\d{1,2})?)")
CERVICAL_DILATION_RE = re.compile(
    r"(?:اتساع\s+عنق\s+الرحم|عنق\s+الرحم)\D{0,24}?"
    r"(\d{1,2}(?:[.,]\d+)?)\s*(?:سم|سنتيمتر)",
    re.I,
)
VAGINAL_FLUID_LEAK_RE = re.compile(
    r"(?:نزول|نزل|خروج)\s+(?:سائل|سوائل)(?:\s+(?:من\s+المهبل|مهبلي(?:ة)?))?",
    re.I,
)
POSTPARTUM_DAYS_RE = re.compile(
    r"(?:ولدت|الولادة\s+(?:كانت|صارت))\s+(?:منذ\s+|من\s+)?(\d{1,3})\s*(?:يوم|أيام|ايام)",
    re.I,
)
PATIENT_FEVER_RE = re.compile(
    r"(?:بتحس|تحس|تشعر|عندها|معها)\s+(?:ب)?حرارة(?:\s+(?:عالية|مرتفعة))?",
    re.I,
)
LOWER_ABDOMINAL_PAIN_RE = re.compile(
    r"(?:ألم|الم|وجع)\s*(?:ب|في)?\s*أسفل\s+البطن", re.I
)
FOUL_DISCHARGE_RE = re.compile(
    r"(?:إفرازات|افرازات|إفرازات\s+مهبلية|افرازات\s+مهبلية)"
    r".{0,45}?(?:ريح(?:ت|تها|ة)|رائح(?:تها|ة))"
    r".{0,24}?(?:غير\s+طبيعية|غير\s+طبيعيه|كريهة|كريهه|سيئة|سيئه)",
    re.I,
)
QUALITATIVE_TACHYCARDIA_RE = re.compile(
    r"(?:النبض|نبضها|ضربات\s+القلب)\s*.{0,18}?(?:أسرع|اسرع|سريع|سريعة|مرتفع|مرتفعة)"
    r"(?:\s*.{0,12}?(?:من\s+الطبيعي|عن\s+الطبيعي))?",
    re.I,
)
POSTPARTUM_INFECTION_RE = re.compile(
    r"(?:عدوى|عدوه|عدوة|التهاب|إنتان|انتان)\s+(?:بعد\s+الولادة|النفاس|نفاسي(?:ة)?)",
    re.I,
)
POSTPARTUM_SUSPICION_RE = re.compile(
    r"(?:نفكر|نشتبه|اشتباه|مشتبه|ممكن|احتمال|يرجح|نرجح)", re.I
)


@dataclass
class ExtractedEntity:
    """One clinical mention, with everything needed to audit or act on it."""

    kind: str                       # symptom|condition|test|lab|medication|allergy|vital
    code: str
    assertion: Assertion = "present"
    value: Optional[float] = None
    value2: Optional[float] = None  # BP diastolic
    unit: Optional[str] = None
    status: Optional[str] = None    # high|low|normal for labs
    char_start: Optional[int] = None
    char_end: Optional[int] = None
    matched_text: Optional[str] = None
    extractor: str = "lexicon"
    extractor_version: str = EXTRACTOR_VERSION
    confidence: float = 1.0
    note: Optional[str] = None

    @property
    def is_actionable(self) -> bool:
        return self.assertion == "present"

    def to_dict(self) -> dict:
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass
class Lexicon:
    """Loaded gazetteers, pre-sorted so the longest surface form wins."""

    symptoms: list[tuple[str, str]] = field(default_factory=list)
    conditions: list[tuple[str, str]] = field(default_factory=list)
    tests: list[tuple[str, str]] = field(default_factory=list)
    labs: list[tuple[str, str]] = field(default_factory=list)
    drugs: list[tuple[str, str]] = field(default_factory=list)
    procedures: list[tuple[str, str]] = field(default_factory=list)
    nutrition: list[tuple[str, str]] = field(default_factory=list)
    drug_family: dict[str, str] = field(default_factory=dict)
    lab_meta: dict[str, dict] = field(default_factory=dict)
    value_markers: dict[str, list[str]] = field(default_factory=dict)
    display: dict[str, str] = field(default_factory=dict)

    @property
    def size(self) -> dict[str, int]:
        return {
            "symptoms": len({c for c, _ in self.symptoms}),
            "conditions": len({c for c, _ in self.conditions}),
            "tests": len({c for c, _ in self.tests}),
            "labs": len({c for c, _ in self.labs}),
            "drugs": len({c for c, _ in self.drugs}),
            "procedures": len({c for c, _ in self.procedures}),
            "nutrition": len({c for c, _ in self.nutrition}),
            "surface_forms": sum(
                len(x) for x in (self.symptoms, self.conditions, self.tests, self.labs,
                                 self.drugs, self.procedures, self.nutrition)
            ),
        }


def _load_yaml(name: str) -> dict:
    path = LEXICON_DIR / name
    if not path.exists():
        logger.warning("lexicon file missing: %s", path)
        return {}
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _pairs(section: dict, display: dict) -> list[tuple[str, str]]:
    """(code, surface) pairs, longest surface first so specific terms win."""
    out: list[tuple[str, str]] = []
    for code, entry in (section or {}).items():
        if not isinstance(entry, dict):
            continue
        display[code] = entry.get("ar", code)
        for variant in entry.get("variants", []) or []:
            out.append((code, str(variant)))
    return sorted(out, key=lambda pair: len(pair[1]), reverse=True)


@functools.lru_cache(maxsize=1)
def load_lexicon() -> Lexicon:
    """Read the YAML gazetteers once per process."""
    display: dict[str, str] = {}
    symptoms = _pairs(_load_yaml("symptoms.yaml"), display)
    conditions = _pairs(_load_yaml("conditions.yaml"), display)

    tests_labs = _load_yaml("tests_labs.yaml")
    tests = _pairs(tests_labs.get("tests", {}), display)
    labs = _pairs(tests_labs.get("labs", {}), display)
    lab_meta = {code: entry for code, entry in (tests_labs.get("labs") or {}).items()}

    meds = _load_yaml("medications.yaml")
    drugs = _pairs(meds, display)
    families = {
        code: entry.get("family", code)
        for code, entry in meds.items() if isinstance(entry, dict)
    }

    procedures = _pairs(_load_yaml("procedures.yaml"), display)
    nutrition = _pairs(_load_yaml("nutrition.yaml"), display)

    lexicon = Lexicon(
        symptoms=symptoms, conditions=conditions, tests=tests, labs=labs, drugs=drugs,
        procedures=procedures, nutrition=nutrition,
        drug_family=families, lab_meta=lab_meta,
        value_markers=tests_labs.get("value_markers", {}), display=display,
    )
    logger.info("lexicon loaded: %s", lexicon.size)
    return lexicon


def display_name(code: str) -> str:
    return load_lexicon().display.get(code, code)


def drug_family(code: str) -> str:
    """Family for cross-reactivity. Unknown drugs are their own family, never merged."""
    return load_lexicon().drug_family.get(code, code)


def normalize_digits(text: str) -> str:
    return str(text).translate(_ARABIC_INDIC)


# --- matching -----------------------------------------------------------------------


def _context_allows_match(text: str, code: str, kind: str, start: int, end: int) -> bool:
    """Suppress lexicon homonyms when nearby context makes the meaning unambiguous.

    The main real-world collision is ``بروتين``: in a diet sentence it means dietary
    protein, while near ``بول/زلال/تحليل البول`` it is a urine-protein laboratory
    finding.  Keep the rule local to the mention so a later unrelated urine clause
    does not erase genuine nutrition advice.
    """
    if kind == "nutrition" and code == "protein_rich_food":
        window = text[max(0, start - 18): min(len(text), end + 28)]
        urine_markers = ("بول", "البول", "زلال", "تحليل البول", "ألبومين البول")
        if any(marker in window for marker in urine_markers):
            return False

    # ``سكري`` is a disease term, while ``سكريات`` means dietary sugars.  Substring
    # matching used to create a diabetes diagnosis from advice such as
    # "لازم تخفف السكريات".  Suppress only this well-defined morphology collision;
    # explicit disease phrases (سكري، السكري، سكر الحمل...) remain unaffected.
    if kind == "condition" and code == "diabetes":
        word_start = start
        while word_start > 0 and text[word_start - 1].isalpha():
            word_start -= 1
        word_end = end
        while word_end < len(text) and text[word_end].isalpha():
            word_end += 1
        containing_word = text[word_start:word_end]
        if containing_word in {"سكريات", "السكريات"}:
            return False

    # A disease phrase inside the *name of a test* is not itself a diagnosis.  This
    # matters for speech such as "تحليل سكر الحمل طلع مرتفع"; the later explicit
    # sentence "معها سكر الحمل" should establish GDM, not the test name alone.
    if kind == "condition" and code == "gdm":
        before = text[max(0, start - 18):start]
        if any(marker in before for marker in ("تحليل", "اختبار", "فحص")):
            return False

    # ``نزيف غزير`` is not intrinsically menstrual.  In pregnancy/labour speech it
    # usually means vaginal/obstetric bleeding, while the old longest-match lexicon
    # preferred ``heavy_menstrual_bleeding`` and hid the more appropriate
    # ``vaginal_bleeding`` mention.  Require an explicit menstrual context before
    # assigning the menstrual code; otherwise the shorter vaginal-bleeding term is
    # allowed to win on the next pass.
    if kind == "symptom" and code == "heavy_menstrual_bleeding":
        window = text[max(0, start - 18): min(len(text), end + 18)]
        if not any(marker in window for marker in ("دورة", "الدورة", "طمث", "حيض")):
            return False
    return True


def _surface_has_safe_boundaries(text: str, start: int, end: int) -> bool:
    """Reject a surface embedded inside a larger Arabic/Latin token.

    Common Arabic clitic/article prefixes are allowed (``والحديد``, ``الإفرازات``),
    but lexical interiors such as ``حديد`` inside ``تحديد`` are not.
    """
    if end < len(text) and text[end].isalnum():
        return False
    word_start = start
    while word_start > 0 and text[word_start - 1].isalnum():
        word_start -= 1
    prefix = text[word_start:start]
    return prefix in {"", "و", "ف", "ب", "ك", "ل", "ال", "وال", "فال", "بال", "كال", "لل"}


def _find_terms(text: str, pairs: Iterable[tuple[str, str]], kind: str
                ) -> list[ExtractedEntity]:
    """Match gazetteer terms, keeping one entity per code and skipping overlaps.

    Longest-first ordering plus overlap suppression is what stops "ألم" firing inside
    "ألم أسفل البطن" and producing two competing entities for one phrase.
    """
    found: list[ExtractedEntity] = []
    claimed: list[tuple[int, int]] = []
    seen_codes: set[str] = set()

    for code, surface in pairs:
        if code in seen_codes:
            continue
        start = text.find(surface)
        while start >= 0:
            end = start + len(surface)
            if (
                not any(start < c_end and end > c_start for c_start, c_end in claimed)
                and _surface_has_safe_boundaries(text, start, end)
                and _context_allows_match(text, code, kind, start, end)
            ):
                result = classify_assertion(text, start, end)
                found.append(ExtractedEntity(
                    kind=kind, code=code, assertion=result.assertion,
                    char_start=start, char_end=end, matched_text=surface,
                    note=f"trigger: {result.trigger}" if result.trigger else None,
                ))
                claimed.append((start, end))
                seen_codes.add(code)
                break
            start = text.find(surface, start + 1)
    return found


def _lab_status(text: str, start: int, end: int, markers: dict) -> tuple[Optional[str],
                                                                        Optional[float]]:
    """Read a lab's qualifier and value from around the TERM, within its own clause.

    Two bounds, both needed: a character window (a qualifier 200 characters away is not
    about this analyte) AND the clause boundary (in
    "الهيموغلوبين طبيعي لكن الضغط مرتفع" the مرتفع belongs to the blood pressure —
    a window alone still swallows it in a short sentence, which is exactly how v1
    reported a high haemoglobin).
    """
    left_bound, right_bound = clause_bounds(text, start, end)
    window_start = max(left_bound, start - VALUE_WINDOW)
    window_end = min(right_bound, end + VALUE_WINDOW)
    window = text[window_start:window_end]

    status = None
    for level in ("high", "low", "normal"):
        for marker in markers.get(level, []):
            pos = window.find(marker)
            while pos >= 0:
                absolute_start = window_start + pos
                # Qualifiers bind locally to the analyte ("الهيموغلوبين ضمن المقبول").
                # A broad earlier phrase such as "ما فيه مشاكل" must not negate a
                # later qualifier, while a direct "ليس ضمن المقبول" still must.
                local_start = max(left_bound, absolute_start - 12)
                local_text = text[local_start:absolute_start + len(marker)]
                local_marker_start = absolute_start - local_start
                result = classify_assertion(
                    local_text, local_marker_start, local_marker_start + len(marker)
                )
                if result.is_actionable:
                    status = level
                    break
                pos = window.find(marker, pos + 1)
            if status is not None:
                break
        if status is not None:
            break

    value = None
    after = text[end:window_end]
    match = NUMBER_NEAR.search(after)
    if match:
        try:
            value = float(match.group(1).replace(",", "."))
        except ValueError:  # pragma: no cover
            value = None
    return status, value


def _classify_by_range(code: str, value: Optional[float], lexicon: Lexicon
                       ) -> Optional[str]:
    """Prefer the reference range over adjectives when a number is present."""
    meta = lexicon.lab_meta.get(code) or {}
    if value is None or "normal_range" not in meta:
        return None
    low, high = meta["normal_range"]
    if value < low:
        return "low"
    if value > high:
        return "high"
    return "normal"


def _vitals(text: str) -> list[ExtractedEntity]:
    entities: list[ExtractedEntity] = []
    match = BP_RE.search(text)
    if match:
        systolic, diastolic = float(match.group(1)), float(match.group(2))
        if systolic <= 25:  # Levantine short form: "11/7" means 110/70
            systolic, diastolic = systolic * 10, diastolic * 10
        entities.append(ExtractedEntity(
            kind="vital", code="bp", value=systolic, value2=diastolic, unit="mmHg",
            char_start=match.start(), char_end=match.end(), matched_text=match.group(0),
            assertion=classify_assertion(text, match.start(), match.end()).assertion,
        ))
        entities.append(ExtractedEntity(kind="test", code="bp_measurement",
                                        char_start=match.start(), char_end=match.end()))
    match = TEMP_RE.search(text)
    if match:
        entities.append(ExtractedEntity(
            kind="vital", code="temp", value=float(match.group(1).replace(",", ".")),
            unit="C", char_start=match.start(), char_end=match.end(),
            matched_text=match.group(0),
        ))
    match = PULSE_RE.search(text)
    if match:
        entities.append(ExtractedEntity(
            kind="vital", code="pulse", value=float(match.group(1)), unit="bpm",
            char_start=match.start(), char_end=match.end(), matched_text=match.group(0),
        ))
    return entities


def _obstetric_measurements(text: str) -> list[ExtractedEntity]:
    """Structured obstetric findings that must be shared with the KBS.

    These values used to be re-parsed only inside ``EXPERTA_MED``.  That made the
    SOAP report look empty while the rules somehow knew about a 5-cm cervix.  Parse
    them once here so the report, database and KBS all consume the same value.
    """
    entities: list[ExtractedEntity] = []
    match = CERVICAL_DILATION_RE.search(text)
    if match:
        entities.append(ExtractedEntity(
            kind="clinical",
            code="cervical_dilation_cm",
            assertion=classify_assertion(text, match.start(), match.end()).assertion,
            value=float(match.group(1).replace(",", ".")),
            unit="cm",
            char_start=match.start(),
            char_end=match.end(),
            matched_text=match.group(0),
        ))

    # A patient can truthfully report vaginal fluid leakage without knowing whether
    # membranes ruptured.  Keep this generic finding separate; the discourse layer
    # upgrades it to *suspected* water breaking only when a neighbouring segment says
    # the fluid may be amniotic fluid.
    match = VAGINAL_FLUID_LEAK_RE.search(text)
    if match:
        entities.append(ExtractedEntity(
            kind="clinical",
            code="vaginal_fluid_leak",
            assertion=classify_assertion(text, match.start(), match.end()).assertion,
            value=True,
            char_start=match.start(),
            char_end=match.end(),
            matched_text=match.group(0),
        ))
    return entities


def _postpartum_findings(text: str) -> list[ExtractedEntity]:
    """Structured postpartum findings that are reliable without inventing values.

    These patterns deliberately capture *qualitative* findings (foul lochia, a fast
    pulse) without manufacturing numeric measurements.  They also preserve suspected
    infection as suspected rather than silently promoting it to a confirmed diagnosis.
    """
    entities: list[ExtractedEntity] = []

    days = POSTPARTUM_DAYS_RE.search(text)
    if days:
        value = float(days.group(1))
        entities.append(ExtractedEntity(
            kind="clinical", code="postpartum_days_since_birth", assertion="present",
            value=value, unit="days", char_start=days.start(), char_end=days.end(),
            matched_text=days.group(0),
        ))
        entities.append(ExtractedEntity(
            kind="clinical", code="postpartum_hours_since_birth", assertion="present",
            value=value * 24.0, unit="hours", char_start=days.start(), char_end=days.end(),
            matched_text=days.group(0), note="derived from postpartum days",
        ))

    fever = PATIENT_FEVER_RE.search(text)
    if fever:
        entities.append(ExtractedEntity(
            kind="symptom", code="fever",
            assertion=classify_assertion(text, fever.start(), fever.end()).assertion,
            char_start=fever.start(), char_end=fever.end(), matched_text=fever.group(0),
        ))

    lower_pain = LOWER_ABDOMINAL_PAIN_RE.search(text)
    if lower_pain:
        entities.append(ExtractedEntity(
            kind="symptom", code="lower_abdominal_pain",
            assertion=classify_assertion(text, lower_pain.start(), lower_pain.end()).assertion,
            char_start=lower_pain.start(), char_end=lower_pain.end(), matched_text=lower_pain.group(0),
        ))

    foul = FOUL_DISCHARGE_RE.search(text)
    if foul:
        entities.append(ExtractedEntity(
            kind="symptom", code="foul_vaginal_discharge", assertion="present",
            status="foul_smelling", char_start=foul.start(), char_end=foul.end(),
            matched_text=foul.group(0),
        ))

    tachy = QUALITATIVE_TACHYCARDIA_RE.search(text)
    if tachy:
        entities.append(ExtractedEntity(
            kind="clinical", code="maternal_tachycardia", assertion="present",
            value=True, status="high", char_start=tachy.start(), char_end=tachy.end(),
            matched_text=tachy.group(0),
            note="qualitative pulse description; no numeric heart rate inferred",
        ))

    infection = POSTPARTUM_INFECTION_RE.search(text)
    if infection:
        prefix = text[max(0, infection.start() - 28):infection.start()]
        suspected = bool(POSTPARTUM_SUSPICION_RE.search(prefix))
        entities.append(ExtractedEntity(
            kind="condition", code="puerperal_infection", assertion="present",
            status="suspected" if suspected else "confirmed",
            char_start=infection.start(), char_end=infection.end(),
            matched_text=infection.group(0),
            note="clinical suspicion wording" if suspected else None,
        ))

    return entities


def _drugs(text: str, label: str, lexicon: Lexicon) -> list[ExtractedEntity]:
    """Drugs, with allergy vs prescription decided by the sentence's label.

    An unknown Latin token is flagged, not invented as a drug (P5-13).
    """
    kind = "allergy" if label == "allergy" else "medication"
    entities = _find_terms(text, lexicon.drugs, kind)
    known_spans = [(e.char_start, e.char_end) for e in entities]

    if label in {"allergy", "medication", "treatment"}:
        for match in _LATIN_TOKEN.finditer(text):
            if match.group(0).lower() in _LATIN_STOPWORDS:
                continue
            if any(match.start() < end and match.end() > start
                   for start, end in known_spans if start is not None):
                continue
            entities.append(ExtractedEntity(
                kind=kind, code="unknown_drug", char_start=match.start(),
                char_end=match.end(), matched_text=match.group(0), confidence=0.4,
                note=f"unrecognised drug name '{match.group(0)}' — needs review",
                assertion=classify_assertion(text, match.start(), match.end()).assertion,
            ))
    return entities


def extract_entities(text: str, label: str = "", *, lexicon: Optional[Lexicon] = None
                     ) -> list[ExtractedEntity]:
    """Extract every clinical mention from one sentence, with its assertion status."""
    lexicon = lexicon or load_lexicon()
    normalized = normalize_digits(text)
    if not normalized.strip():
        return []

    entities: list[ExtractedEntity] = []
    entities += _find_terms(normalized, lexicon.symptoms, "symptom")
    entities += _find_terms(normalized, lexicon.conditions, "condition")
    entities += _find_terms(normalized, lexicon.tests, "test")
    procedures = _find_terms(normalized, lexicon.procedures, "procedure")
    # A later safety condition must not turn an already scheduled appointment into
    # a hypothetical appointment: "الموعد ... متابعة بعد 4 أسابيع إذا ما صار عرض".
    for entity in procedures:
        if entity.code == "follow_up_visit" and entity.assertion == "hypothetical":
            cond_positions = [
                normalized.find(cue) for cue in ("إذا", "اذا", "في حال", "لو ")
                if normalized.find(cue) >= 0
            ]
            first_cond = min(cond_positions) if cond_positions else -1
            scheduling_prefix = normalized[: first_cond if first_cond >= 0 else len(normalized)]
            if (
                entity.char_end is not None
                and first_cond > entity.char_end
                and any(cue in scheduling_prefix for cue in (
                    "الموعد", "موعد", "المتابعة", "متابعة بعد", "مراجعة بعد", "نشوفك بعد"
                ))
            ):
                entity.assertion = "planned"
                entity.note = "scheduled follow-up; later condition does not scope backward"
    entities += procedures
    entities += _find_terms(normalized, lexicon.nutrition, "nutrition")
    entities += _vitals(normalized)
    entities += _obstetric_measurements(normalized)
    entities += _postpartum_findings(normalized)
    entities += _drugs(normalized, label, lexicon)

    # The generic condition lexicon and the postpartum typo-tolerant regex can both
    # recognise the same infection phrase.  Keep one epistemically safest mention;
    # an explicit suspicion must never coexist with a second "confirmed-looking" link.
    puerperal = [
        e for e in entities if e.kind == "condition" and e.code == "puerperal_infection"
    ]
    if len(puerperal) > 1:
        preferred = next((e for e in puerperal if e.status == "suspected"), puerperal[0])
        entities = [
            e for e in entities
            if not (e.kind == "condition" and e.code == "puerperal_infection")
        ] + [preferred]

    for entity in _find_terms(normalized, lexicon.labs, "lab"):
        status, value = _lab_status(
            normalized, entity.char_start or 0, entity.char_end or 0,
            lexicon.value_markers,
        )
        by_range = _classify_by_range(entity.code, value, lexicon)
        entity.status = by_range or status
        entity.value = value
        entity.unit = (lexicon.lab_meta.get(entity.code) or {}).get("unit")
        # An actionable qualifier/value (e.g. "الهيموغلوبين ضمن المقبول") is direct
        # evidence that the lab result exists.  It must override a broad earlier
        # phrase such as "ما فيه مشاكل مهمة" that otherwise leaks negation forward.
        if entity.status is not None or entity.value is not None:
            entity.assertion = "present"
            if entity.note and entity.note.startswith("trigger:"):
                entity.note = None
        entities.append(entity)
        # A reported result implies the test was performed.
        entities.append(ExtractedEntity(kind="test", code=f"{entity.code}_test",
                                        char_start=entity.char_start,
                                        char_end=entity.char_end))

    return entities


def extract_for_item(item: dict[str, Any]) -> list[dict]:
    """Entities for one report item, ready to store in `entity_links`."""
    return [
        entity.to_dict()
        for entity in extract_entities(item.get("text", ""), item.get("label", ""))
    ]
