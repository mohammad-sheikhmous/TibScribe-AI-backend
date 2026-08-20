"""Clinical Arabic canonicalization between ASR segmentation and downstream NLP.

P11 treats model-generated Arabic correction as a **shadow candidate**. The raw
Whisper/segmentation text remains the clinical decision source for AraBERT, entity
extraction, SOAP routing and the KBS. Candidates are retained only after deterministic
fact-preservation checks, so they can be benchmarked without regressing clinical logic.

Production backend
------------------
``QwenClinicalCanonicalizer`` uses the versioned instruction model
``Qwen/Qwen3-4B-Instruct-2507``. The interface remains model-agnostic so a future
clinical fine-tune can be swapped in without changing the pipeline.

Safety boundary
---------------
The guard is deliberately conservative.  It rejects candidates that change numeric
values/order, flip coarse negation/conditional/history/family scope, lose a clinical
entity that the raw text already expressed reliably, or expand/shrink implausibly.
Rejected/failed candidates fall back to the original ASR text; the job never fails just
because the optional canonicalizer does.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import gc
import logging
import re
from typing import Optional, Protocol, Sequence, runtime_checkable

from .assertion import (
    FAMILY,
    HYPOTHETICAL,
    has_historical_cue,
    has_negation_cue,
)
from .extraction import extract_for_item, normalize_digits

logger = logging.getLogger(__name__)

_WS_RE = re.compile(r"\s+")
_NUMBER_RE = re.compile(r"(?<!\w)[+-]?\d+(?:[\.,]\d+)?")
_ARABIC_RE = re.compile(r"[\u0600-\u06ff]")


@dataclass(frozen=True)
class CanonicalizationCandidate:
    text: str
    confidence: Optional[float] = None


@dataclass(frozen=True)
class SafetyDecision:
    accepted: bool
    reasons: tuple[str, ...] = ()


@dataclass(frozen=True)
class CanonicalizationResult:
    raw_text: str
    canonical_text: Optional[str]
    status: str  # accepted|unchanged|rejected|failed|not_run
    confidence: Optional[float] = None
    model_name: Optional[str] = None
    reasons: tuple[str, ...] = ()

    @property
    def effective_text(self) -> str:
        return self.canonical_text if self.canonical_text else self.raw_text


@runtime_checkable
class CanonicalizationStage(Protocol):
    @property
    def applied(self) -> bool: ...

    @property
    def model_name(self) -> str: ...

    def canonicalize_batch(self, texts: Sequence[str]) -> list[CanonicalizationCandidate]: ...


class NoOpCanonicalizer:
    """Identity fallback used when canonicalization is disabled or unavailable."""

    applied = False
    model_name = "none"

    def canonicalize_batch(self, texts: Sequence[str]) -> list[CanonicalizationCandidate]:
        return [CanonicalizationCandidate(str(text)) for text in texts]


class QwenClinicalCanonicalizer:
    """Instruction-following Arabic clinical corrector used in **shadow mode**.

    The model may propose a cleaner MSA rendering of a Whisper segment, but P11 does
    not let that proposal drive AraBERT, entity extraction, SOAP routing or the KBS.
    It is retained as an auditable candidate for evaluation/display experiments only.
    """

    applied = True

    _SYSTEM_PROMPT = (
        "أنت مصحح محافظ لنصوص التفريغ الصوتي الطبية العربية. مهمتك إعادة صياغة "
        "جملة واحدة من العربية المحكية/المشوشة إلى عربية طبية فصحى طبيعية وموجزة، "
        "من دون تغيير أي حقيقة سريرية. ممنوع إضافة أو استنتاج أو حذف تشخيص أو عرض أو "
        "دواء أو فحص أو قياس أو زمن أو علاقة. حافظ حرفياً على كل الأرقام وترتيبها، "
        "وعلى النفي، والشرط/الافتراض، والماضي، والخطة المستقبلية. لا تحوّل الخبر إلى "
        "سؤال ولا السؤال إلى خبر. لا تصلح رقماً بالمنطق حتى لو بدا مستحيلاً. إذا كانت "
        "كلمة غير واضحة أو تبدو خطأ ASR فلا تخمّنها؛ اتركها كما هي. إذا لم تستطع تحسين "
        "النص بأمان فأعد النص نفسه. أعد الجملة العربية فقط بلا شرح. أمثلة للأسلوب: "
        "«في عندها وجع راس خفيف ورجليها شوي متنفخين» ← «تعاني من صداع خفيف وتورم "
        "بسيط في القدمين». «ما عندها لا تشوش بالرؤية ولا نزيف» ← «لا تعاني من تشوش "
        "في الرؤية أو نزيف». «فحصنا لها ضغطها طلع 128 على 82» ← «ضغط الدم 128 على 82». "
        "«إذا صار عندها تشوش بالرؤية تراجع فوراً» ← «إذا حدث تشوش في الرؤية، تراجع فوراً»."
    )

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-4B-Instruct-2507",
        *,
        revision: Optional[str] = None,
        device: Optional[str] = None,
        max_input_tokens: int = 768,
        max_new_tokens: int = 96,
        batch_size: int = 4,
        local_files_only: bool = False,
    ) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_id = model_name
        self.revision = revision
        self.model_name = f"{model_name}@{revision[:12]}" if revision else model_name
        self.max_input_tokens = max_input_tokens
        self.max_new_tokens = max_new_tokens
        self.batch_size = max(1, batch_size)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, revision=revision, local_files_only=local_files_only
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        dtype = torch.float16 if self.device.startswith("cuda") else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            revision=revision,
            dtype=dtype,
            local_files_only=local_files_only,
        ).to(self.device)
        self.model.eval()

    @classmethod
    def messages_for(cls, text: str) -> list[dict[str, str]]:
        return [
            {
                "role": "system",
                "content": cls._SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": (
                    "صحح النص التالي فقط:\n"
                    f"<TEXT>{_clean_text(text)}</TEXT>"
                ),
            },
        ]

    def canonicalize_batch(self, texts: Sequence[str]) -> list[CanonicalizationCandidate]:
        if not texts:
            return []
        cleaned = [_clean_text(t) for t in texts]
        out: list[CanonicalizationCandidate] = []
        for start in range(0, len(cleaned), self.batch_size):
            out.extend(self._canonicalize_chunk(cleaned[start:start + self.batch_size]))
        return out

    def _canonicalize_chunk(self,texts: Sequence[str]) -> list[CanonicalizationCandidate]:
        import torch

        results: list[CanonicalizationCandidate] = []

        for text in texts:

            messages = self.messages_for(text)

            inputs = self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            )

            inputs = {
                key: value.to(self.device)
                for key, value in inputs.items()
            }

            input_length = inputs["input_ids"].shape[-1]

            # Clinical safety:
            # never silently truncate the patient's text.
            if input_length > self.max_input_tokens:
                logger.warning(
                    "Canonicalizer input too long (%s tokens); "
                    "returning original text instead of truncating.",
                    input_length,
                )

                results.append(
                    CanonicalizationCandidate(
                        text=_clean_text(text),
                        confidence=None,
                    )
                )

                continue

            with torch.inference_mode():

                generated = self.model.generate(
                    **inputs,
                    do_sample=False,
                    max_new_tokens=self.max_new_tokens,
                    repetition_penalty=1.02,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )

            output_ids = generated[0][input_length:]

            decoded = self.tokenizer.decode(
                output_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )

            results.append(
                CanonicalizationCandidate(
                    text=_clean_text(decoded),
                    confidence=None,
                )
            )

        return results

@dataclass
class ClinicalSafetyGuard:
    """Fact-preservation validator for generative canonicalization."""

    min_length_ratio: float = 0.45
    max_length_ratio: float = 2.20
    protected_entity_kinds: set[str] = field(default_factory=lambda: {
        "symptom", "condition", "vital", "lab", "test", "procedure",
        "medication", "allergy", "clinical",
    })

    def validate(self, raw_text: str, candidate_text: str) -> SafetyDecision:
        raw = _clean_text(raw_text)
        candidate = _clean_text(candidate_text)
        reasons: list[str] = []

        if not candidate:
            return SafetyDecision(False, ("empty_candidate",))

        raw_len = max(1, len(raw))
        ratio = len(candidate) / raw_len
        if ratio < self.min_length_ratio or ratio > self.max_length_ratio:
            reasons.append(f"length_ratio_changed:{ratio:.2f}")

        raw_numbers = _numeric_signature(raw)
        candidate_numbers = _numeric_signature(candidate)
        if raw_numbers != candidate_numbers:
            reasons.append("numeric_facts_changed")

        # Units are clinical facts.
        #
        # We protect both directions:
        #
        # source unit disappears  -> reject
        # new unit is invented    -> reject
        #
        # Equivalent surfaces such as:
        # mg -> ملغ
        # remain allowed because both normalize to "mg".
        raw_units = _unit_signature(raw)
        candidate_units = _unit_signature(candidate)

        missing_units = [
            unit
            for unit in raw_units
            if unit not in candidate_units
        ]

        introduced_units = [
            unit
            for unit in candidate_units
            if unit not in raw_units
        ]

        if missing_units:
            reasons.append(
                "clinical_units_changed:"
                + ",".join(
                    dict.fromkeys(missing_units)
                )
            )

        if introduced_units:
            reasons.append(
                "introduced_clinical_unit:"
                + ",".join(
                    dict.fromkeys(
                        introduced_units
                    )
                )
            )

        raw_qualifiers = _qualifier_signature(raw)
        candidate_qualifiers = _qualifier_signature(candidate)
        if raw_qualifiers != candidate_qualifiers:
            reasons.append("clinical_qualifier_changed")

        raw_cues = _scope_signature(raw)
        candidate_cues = _scope_signature(candidate)
        for name in ("negated", "hypothetical", "historical", "family"):
            if raw_cues[name] != candidate_cues[name]:
                reasons.append(f"{name}_scope_changed")

        if _speech_act_signature(raw) != _speech_act_signature(candidate):
            reasons.append("speech_act_changed")

        if _patient_gender_conflict(
            raw,
            candidate,
        ):
            reasons.append(
                "patient_gender_changed"
            )

        reasons.extend(
            _lost_or_changed_entities(
                raw,
                candidate,
                self.protected_entity_kinds,
            )
        )

        reasons.extend(
            _introduced_sensitive_concepts(
                raw,
                candidate,
            )
        )

        return SafetyDecision(not reasons, tuple(dict.fromkeys(reasons)))


def canonicalize_texts(
    texts: Sequence[str],
    stage: CanonicalizationStage,
    guard: Optional[ClinicalSafetyGuard] = None,
) -> list[CanonicalizationResult]:
    """Canonicalize a batch with per-segment fallback; never raises on model failure."""
    guard = guard or ClinicalSafetyGuard()
    raw = [_clean_text(text) for text in texts]
    if not getattr(stage, "applied", False):
        return [
            CanonicalizationResult(
                raw_text=text,
                canonical_text=None,
                status="not_run",
                model_name=getattr(stage, "model_name", "none"),
            )
            for text in raw
        ]

    eligible = [bool(text and _ARABIC_RE.search(text)) for text in raw]
    candidates: list[Optional[CanonicalizationCandidate]] = [None] * len(raw)
    indices = [i for i, ok in enumerate(eligible) if ok]
    try:
        generated = stage.canonicalize_batch([raw[i] for i in indices])
        if len(generated) != len(indices):
            raise RuntimeError(
                f"canonicalizer returned {len(generated)} rows for {len(indices)} inputs"
            )
        for i, candidate in zip(indices, generated):
            candidates[i] = candidate
    except Exception as exc:  # noqa: BLE001 - canonicalization is optional by design
        reason = f"canonicalizer_error:{type(exc).__name__}"
        return [
            CanonicalizationResult(
                raw_text=text,
                canonical_text=None,
                status="failed" if eligible[i] else "not_run",
                model_name=getattr(stage, "model_name", None),
                reasons=(reason,) if eligible[i] else (),
            )
            for i, text in enumerate(raw)
        ]

    results: list[CanonicalizationResult] = []
    for i, text in enumerate(raw):
        if not eligible[i]:
            results.append(CanonicalizationResult(
                raw_text=text, canonical_text=None, status="not_run",
                model_name=getattr(stage, "model_name", None),
            ))
            continue
        candidate = candidates[i]
        assert candidate is not None
        canonical = _clean_text(candidate.text)
        if canonical == text:
            results.append(CanonicalizationResult(
                raw_text=text,
                canonical_text=None,
                status="unchanged",
                confidence=candidate.confidence,
                model_name=getattr(stage, "model_name", None),
            ))
            continue
        decision = guard.validate(text, canonical)
        results.append(CanonicalizationResult(
            raw_text=text,
            canonical_text=canonical if decision.accepted else None,
            status="accepted" if decision.accepted else "rejected",
            confidence=candidate.confidence,
            model_name=getattr(stage, "model_name", None),
            reasons=decision.reasons,
        ))
    return results


def build_canonicalizer(
    *,
    enabled: bool,
    model_name: str,
    revision: Optional[str] = None,
    local_files_only: bool = False,
    required: bool = False,
) -> CanonicalizationStage:
    """Construct the configured shadow corrector with a safe no-op fallback."""
    if not enabled:
        return NoOpCanonicalizer()
    try:
        return QwenClinicalCanonicalizer(
            model_name=model_name,
            revision=revision,
            local_files_only=local_files_only,
        )
    except Exception as exc:
        if required:
            raise
        # Shadow correction must never become an availability dependency. Clean up any
        # partially allocated CUDA state and continue with the production raw-text path.
        logger.warning(
            "Optional clinical corrector unavailable; falling back to NoOp (%s: %s)",
            type(exc).__name__, exc,
        )
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        return NoOpCanonicalizer()


def _clean_text(text: str) -> str:
    return _WS_RE.sub(" ", str(text or "")).strip()


def _speech_act_signature(text: str) -> tuple[bool, int]:
    """Coarse statement/question signature used only as a safety invariant."""
    cleaned = _clean_text(text)
    question_mark = "؟" in cleaned or "?" in cleaned
    hal_count = len(re.findall(r"(?<![\w\u0600-\u06FF])هل(?![\w\u0600-\u06FF])", cleaned))
    return question_mark, hal_count


def _numeric_signature(text: str) -> tuple[str, ...]:
    normalized = normalize_digits(text).replace("٫", ".").replace("٬", "")
    return tuple(match.group(0).replace(",", ".") for match in _NUMBER_RE.finditer(normalized))


_AR_LETTER = "A-Za-z\u0600-\u06FF"
_UNIT_DIACRITICS_RE = re.compile(r"[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED]")


def _bounded_pattern(*surfaces: str) -> re.Pattern[str]:
    """Match a unit/marker as a token, not as a substring of another Arabic word."""
    alternatives = "|".join(sorted((re.escape(s) for s in surfaces), key=len, reverse=True))
    return re.compile(rf"(?<![{_AR_LETTER}])(?:{alternatives})(?![{_AR_LETTER}])", re.I)


_UNIT_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("mcg", _bounded_pattern("ميكروغرام", "ميكروغراما", "مكغ", "mcg", "µg")),
    ("mg", _bounded_pattern("ميليغرام", "ميليغراما", "ميلغرام", "ميلغراما", "مليغرام", "مليغراما", "ملغ", "mg")),
    ("kg", _bounded_pattern("كيلوغرام", "كيلوغراما", "كيلو", "كغ", "kg")),
    ("g", _bounded_pattern("غرام", "غراما", "جرام", "جراما", "g")),
    ("cm", _bounded_pattern("سنتيمتر", "سنتيمترا", "سم", "cm")),
    ("mmhg", _bounded_pattern("مم زئبق", "ملم زئبق", "mmhg")),
    (
        "bpm",
        _bounded_pattern(
            "نبضة في الدقيقة",
            "نبضات في الدقيقة",
            "دورة في الدقيقة",
            "ضربة في الدقيقة",
            "بالدقيقة",
            "في الدقيقة",
            "bpm",
        ),
    ),
    ("mmol_l", re.compile(r"(?<![A-Za-z])(?:mmol\s*/\s*l|مليمول\s*/\s*لتر)(?![A-Za-z])", re.I)),
    ("mg_dl", re.compile(r"(?<![A-Za-z])(?:mg\s*/\s*dl|ملغ\s*/\s*دل)(?![A-Za-z])", re.I)),
    ("week", _bounded_pattern("أسبوع", "اسبوع", "أسبوعين", "اسبوعين", "أسبوعان", "اسبوعان", "أسابيع", "اسابيع")),
    ("day", _bounded_pattern("يوم", "يومين", "يومان", "أيام", "ايام")),
    ("hour", _bounded_pattern("ساعة", "ساعتين", "ساعتان", "ساعات")),
    ("year", _bounded_pattern("سنة", "سنتين", "سنتان", "سنين", "سنوات", "عام", "عامين", "عامان", "عاما", "أعوام", "اعوام")),
)


def _unit_signature(text: str) -> tuple[str, ...]:
    """Normalized clinical/time units explicitly present in text, in source order.

    Arabic case endings/tanween are presentation morphology, not a clinical unit
    change (e.g. ``30 سنتيمترًا`` == ``30 سنتيمتر``).
    """
    normalized = _UNIT_DIACRITICS_RE.sub("", _WS_RE.sub(" ", text))
    hits: list[tuple[int, str]] = []
    for code, pattern in _UNIT_PATTERNS:
        for match in pattern.finditer(normalized):
            hits.append((match.start(), code))
    return tuple(code for _, code in sorted(hits))


def _contains_any(text: str, markers: Sequence[str]) -> bool:
    return any(
        bool(marker) and marker in text
        for marker in markers
    )

_QUALIFIER_GROUPS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("high", (
        "أعلى من الطبيعي", "اعلى من الطبيعي", "فوق الطبيعي",
        "مرتفع", "مرتفعة", "ارتفاع", "عالي", "عالية",
    )),
    ("low", (
        "أقل من الطبيعي", "اقل من الطبيعي", "تحت الطبيعي",
        "منخفض", "منخفضة", "ناقص", "نقص", "هابط",
    )),
    ("normal", (
        "ضمن الحدود الطبيعية", "ضمن الحدود", "ضمن الطبيعي", "ضمن المقبول",
        "طبيعي", "طبيعية", "سليم", "سليمة", "منيح",
    )),
    ("positive", ("إيجابي", "ايجابي")),
    ("negative", ("سلبي", "سلبية")),
)


def _qualifier_signature(text: str) -> tuple[str, ...]:
    """Return non-overlapping clinical qualifiers, preferring the longest surface.

    This avoids interpreting ``فوق الطبيعي`` as both ``high`` and ``normal`` merely
    because the shorter word ``طبيعي`` is contained inside the phrase.
    """
    surfaces = [
        (category, marker)
        for category, markers in _QUALIFIER_GROUPS
        for marker in markers
    ]
    surfaces.sort(key=lambda pair: len(pair[1]), reverse=True)

    claimed: list[tuple[int, int, str]] = []
    for category, marker in surfaces:
        start = text.find(marker)
        while start >= 0:
            end = start + len(marker)
            if not any(start < used_end and end > used_start for used_start, used_end, _ in claimed):
                claimed.append((start, end, category))
            start = text.find(marker, start + 1)

    return tuple(category for _, _, category in sorted(claimed, key=lambda hit: hit[0]))


def _scope_signature(text: str,) -> dict[str, bool]:
    return {
        "negated":
            has_negation_cue(text),
        "hypothetical":
            _contains_any(
                text,
                HYPOTHETICAL
            ),
        "historical":
            has_historical_cue(text),
        "family":
            _contains_any(
                text,
                FAMILY
            ),
    }


def _entity_key(link: dict) -> tuple[str, str]:
    return str(link.get("kind", "")), str(link.get("code", ""))

_ALLOWED_ENTITY_INTRODUCTIONS: dict[
    tuple[str, str],
    set[tuple[str, str]]
] = {

    # Home BP monitoring necessarily implies
    # measuring blood pressure.
    #
    # This is not a newly invented clinical fact.
    (
        "test",
        "bp_measurement",
    ): {
        (
            "procedure",
            "home_bp_monitoring",
        )
    },
}

def _lost_or_changed_entities(
    raw_text: str,
    candidate_text: str,
    protected_kinds: set[str],
) -> list[str]:

    raw_links = extract_for_item({
        "text": raw_text,
        "label": "info",
    })

    candidate_links = extract_for_item({
        "text": candidate_text,
        "label": "info",
    })

    raw_by_key = {
        _entity_key(link): link
        for link in raw_links
        if isinstance(link, dict)
    }

    candidate_by_key = {
        _entity_key(link): link
        for link in candidate_links
        if isinstance(link, dict)
    }

    raw_keys = set(raw_by_key)

    reasons: list[str] = []

    # ------------------------------------------------------------
    # 1. Existing facts may NOT disappear or change
    # ------------------------------------------------------------

    for key, source in raw_by_key.items():

        kind, code = key

        if (
            kind not in protected_kinds
            or not code
        ):
            continue

        target = candidate_by_key.get(key)

        if target is None:

            reasons.append(
                f"protected_entity_lost:"
                f"{kind}:{code}"
            )

            continue

        if (
            str(
                source.get(
                    "assertion",
                    "present",
                )
            )
            != str(
                target.get(
                    "assertion",
                    "present",
                )
            )
        ):
            reasons.append(
                f"assertion_changed:"
                f"{kind}:{code}"
            )

        for field_name in (
            "value",
            "value2",
            "unit",
            "status",
        ):

            if (
                source.get(field_name)
                is not None
                and source.get(field_name)
                != target.get(field_name)
            ):

                reasons.append(
                    f"entity_{field_name}_changed:"
                    f"{kind}:{code}"
                )

    # ------------------------------------------------------------
    # 2. Qwen may NOT invent new protected clinical facts
    # ------------------------------------------------------------

    for key, target in candidate_by_key.items():

        kind, code = key

        if (
            kind not in protected_kinds
            or not code
            or key in raw_keys
        ):
            continue

        allowed_sources = (
            _ALLOWED_ENTITY_INTRODUCTIONS.get(
                key,
                set(),
            )
        )

        if (
            allowed_sources
            and raw_keys.intersection(
                allowed_sources
            )
        ):
            continue

        reasons.append(
            f"introduced_clinical_entity:"
            f"{kind}:{code}"
        )

    return reasons

_FEMALE_PATIENT_RE = re.compile(
    r"(?:^|[\s،؛])"
    r"(?:المريضة|السيدة|عندها|لها|حامل)"
    r"(?:$|[\s،؛])"
)

_MALE_PATIENT_RE = re.compile(
    r"(?:^|[\s،؛])"
    r"(?:المريض|الرجل|عنده)"
    r"(?:$|[\s،؛])"
)


def _patient_gender_conflict(
    raw_text: str,
    candidate_text: str,
) -> bool:

    raw_female = bool(
        _FEMALE_PATIENT_RE.search(
            raw_text
        )
    )

    raw_male = bool(
        _MALE_PATIENT_RE.search(
            raw_text
        )
    )

    candidate_female = bool(
        _FEMALE_PATIENT_RE.search(
            candidate_text
        )
    )

    candidate_male = bool(
        _MALE_PATIENT_RE.search(
            candidate_text
        )
    )

    return (
        raw_female
        and candidate_male
    ) or (
        raw_male
        and candidate_female
    )

_SENSITIVE_CONCEPT_PATTERNS: tuple[
    tuple[str, re.Pattern[str]],
    ...
] = (

    (
        "clinical:fetal_presentation",

        re.compile(
            r"(?:الجنين.{0,20}?)?"
            r"(?:وضعية|وضع)\s+"
            r"(?:"
            r"الرأس|رأس|"
            r"رأسية|راسية|"
            r"رأسي|راسي|"
            r"مقعدية|عرضية"
            r")"
        ),
    ),

)


def _introduced_sensitive_concepts(
    raw_text: str,
    candidate_text: str,
) -> list[str]:

    reasons: list[str] = []

    for code, pattern in (
        _SENSITIVE_CONCEPT_PATTERNS
    ):

        candidate_has = bool(
            pattern.search(candidate_text)
        )

        raw_has = bool(
            pattern.search(raw_text)
        )

        if (
            candidate_has
            and not raw_has
        ):

            reasons.append(
                "introduced_sensitive_concept:"
                + code
            )

    return reasons
