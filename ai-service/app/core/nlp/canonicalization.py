"""Clinical Arabic canonicalization between ASR segmentation and downstream NLP.

The model is allowed to improve *language* (dialect -> MSA, spelling, wording) but it
is not allowed to invent or silently change clinical facts.  The raw Whisper segment is
therefore immutable and every generated candidate passes a deterministic safety gate
before AraBERT/entity extraction can consume it.

Production backend
------------------
``Seq2SeqArabicCanonicalizer`` uses a Hugging Face text-to-text model.  The default
configured model is ``Murhaf/AraT5-MSAizer`` (dialectal Arabic -> MSA).  The interface
is intentionally model-agnostic so a future clinical fine-tune can be swapped in with
one environment variable instead of changing the pipeline.

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
import math
import re
from typing import Optional, Protocol, Sequence, runtime_checkable

from .assertion import FAMILY, HISTORICAL, HYPOTHETICAL, POST_NEGATION, PRE_NEGATION
from .extraction import extract_for_item, normalize_digits


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


class Seq2SeqArabicCanonicalizer:
    """Deterministic local seq2seq backend (no external inference API).

    Imports heavy ML dependencies lazily so unit tests and non-ML tooling can import the
    application without downloading/loading another model.
    """

    applied = True

    def __init__(
        self,
        model_name: str = "Murhaf/AraT5-MSAizer",
        *,
        revision: Optional[str] = None,
        device: Optional[str] = None,
        max_input_tokens: int = 256,
        max_new_tokens: int = 192,
        num_beams: int = 4,
        batch_size: int = 8,
        local_files_only: bool = False,
    ) -> None:
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        self.model_id = model_name
        self.revision = revision
        self.model_name = f"{model_name}@{revision[:12]}" if revision else model_name
        self.max_input_tokens = max_input_tokens
        self.max_new_tokens = max_new_tokens
        self.num_beams = max(1, num_beams)
        self.batch_size = max(1, batch_size)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            revision=revision,
            local_files_only=local_files_only,
            use_fast=False,
            legacy=True
        )
        dtype = torch.float16 if self.device.startswith("cuda") else torch.float32
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            revision=revision,
            torch_dtype=dtype,
            local_files_only=local_files_only,
        ).to(self.device)
        self.model.eval()

    def canonicalize_batch(self, texts: Sequence[str]) -> list[CanonicalizationCandidate]:
        """Canonicalize in bounded batches to avoid VRAM spikes on long visits."""
        if not texts:
            return []

        cleaned = [_clean_text(t) for t in texts]
        out: list[CanonicalizationCandidate] = []
        for start in range(0, len(cleaned), self.batch_size):
            out.extend(self._canonicalize_chunk(cleaned[start:start + self.batch_size]))
        return out

    def _canonicalize_chunk(self, texts: Sequence[str]) -> list[CanonicalizationCandidate]:
        import torch

        enc = self.tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            max_length=self.max_input_tokens,
            return_tensors="pt",
        )
        enc = {k: v.to(self.device) for k, v in enc.items()}
        with torch.inference_mode():
            generated = self.model.generate(
                **enc,
                do_sample=False,
                num_beams=self.num_beams,
                max_new_tokens=self.max_new_tokens,
                no_repeat_ngram_size=3,
                return_dict_in_generate=True,
                output_scores=True,
            )

        decoded = self.tokenizer.batch_decode(
            generated.sequences,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )
        scores = getattr(generated, "sequences_scores", None)
        chunk: list[CanonicalizationCandidate] = []
        for index, text in enumerate(decoded):
            confidence = None
            if scores is not None and index < len(scores):
                # Beam sequence score is a length-normalized log score. exp(score) is
                # useful as an *uncalibrated* generation-quality signal only; safety
                # acceptance never depends on it.
                try:
                    confidence = max(0.0, min(1.0, math.exp(float(scores[index].item()))))
                except (ValueError, OverflowError):
                    confidence = None
            chunk.append(CanonicalizationCandidate(_clean_text(text), confidence))
        return chunk


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

        # Preserve every unit/time-scale explicitly spoken in the source while allowing
        # the canonicalizer to add conventional units that are implicit in context.
        raw_units = _unit_signature(raw)
        candidate_units = _unit_signature(candidate)
        missing_units = [unit for unit in raw_units if unit not in candidate_units]
        if missing_units:
            reasons.append("clinical_units_changed:" + ",".join(missing_units))

        raw_qualifiers = _qualifier_signature(raw)
        candidate_qualifiers = _qualifier_signature(candidate)
        if raw_qualifiers != candidate_qualifiers:
            reasons.append("clinical_qualifier_changed")

        raw_cues = _scope_signature(raw)
        candidate_cues = _scope_signature(candidate)
        for name in ("negated", "hypothetical", "historical", "family"):
            if raw_cues[name] != candidate_cues[name]:
                reasons.append(f"{name}_scope_changed")

        reasons.extend(_lost_or_changed_entities(raw, candidate, self.protected_entity_kinds))
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
    """Construct the configured backend with a safe no-op fallback."""
    if not enabled:
        return NoOpCanonicalizer()
    try:
        return Seq2SeqArabicCanonicalizer(
            model_name=model_name,
            revision=revision,
            local_files_only=local_files_only,
        )
    except Exception:
        if required:
            raise
        # Logging here would introduce a module-level logger just for one branch; the
        # caller logs the selected backend at startup.  Fallback is explicit in meta.
        return NoOpCanonicalizer()


def _clean_text(text: str) -> str:
    return _WS_RE.sub(" ", str(text or "")).strip()


def _numeric_signature(text: str) -> tuple[str, ...]:
    normalized = normalize_digits(text).replace("٫", ".").replace("٬", "")
    return tuple(match.group(0).replace(",", ".") for match in _NUMBER_RE.finditer(normalized))


_AR_LETTER = "A-Za-z\u0600-\u06FF"


def _bounded_pattern(*surfaces: str) -> re.Pattern[str]:
    """Match a unit/marker as a token, not as a substring of another Arabic word."""
    alternatives = "|".join(sorted((re.escape(s) for s in surfaces), key=len, reverse=True))
    return re.compile(rf"(?<![{_AR_LETTER}])(?:{alternatives})(?![{_AR_LETTER}])", re.I)


_UNIT_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("mcg", _bounded_pattern("ميكروغرام", "مكغ", "mcg", "µg")),
    ("mg", _bounded_pattern("ميليغرام", "ميلغرام", "مليغرام", "ملغ", "mg")),
    ("kg", _bounded_pattern("كيلوغرام", "كيلو", "كغ", "kg")),
    ("g", _bounded_pattern("غرام", "جرام", "g")),
    ("cm", _bounded_pattern("سنتيمتر", "سم", "cm")),
    ("mmhg", _bounded_pattern("مم زئبق", "ملم زئبق", "mmhg")),
    ("mmol_l", re.compile(r"(?<![A-Za-z])(?:mmol\s*/\s*l|مليمول\s*/\s*لتر)(?![A-Za-z])", re.I)),
    ("mg_dl", re.compile(r"(?<![A-Za-z])(?:mg\s*/\s*dl|ملغ\s*/\s*دل)(?![A-Za-z])", re.I)),
    ("week", _bounded_pattern("أسبوع", "اسبوع", "أسابيع", "اسابيع")),
    ("day", _bounded_pattern("يوم", "أيام", "ايام")),
    ("hour", _bounded_pattern("ساعة", "ساعات")),
    ("year", _bounded_pattern("سنة", "سنين", "سنوات", "عام", "أعوام", "اعوام")),
)


def _unit_signature(text: str) -> tuple[str, ...]:
    """Normalized clinical/time units explicitly present in text, in source order."""
    normalized = _WS_RE.sub(" ", text)
    hits: list[tuple[int, str]] = []
    for code, pattern in _UNIT_PATTERNS:
        for match in pattern.finditer(normalized):
            hits.append((match.start(), code))
    return tuple(code for _, code in sorted(hits))


def _contains_any(text: str, markers: Sequence[str]) -> bool:
    return any(marker.strip() and marker.strip() in text for marker in markers)


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


def _scope_signature(text: str) -> dict[str, bool]:
    return {
        "negated": _contains_any(text, PRE_NEGATION) or _contains_any(text, POST_NEGATION),
        "hypothetical": _contains_any(text, HYPOTHETICAL),
        "historical": _contains_any(text, HISTORICAL),
        "family": _contains_any(text, FAMILY),
    }


def _entity_key(link: dict) -> tuple[str, str]:
    return str(link.get("kind", "")), str(link.get("code", ""))


def _lost_or_changed_entities(
    raw_text: str,
    candidate_text: str,
    protected_kinds: set[str],
) -> list[str]:
    """Protect facts already known with high precision in the raw transcript.

    Candidate-only entities are allowed: canonicalization is expected to make previously
    dialectal/garbled mentions extractable.  What is forbidden is losing/changing a fact
    the raw text already expressed reliably.
    """
    raw_links = extract_for_item({"text": raw_text, "label": "info"})
    candidate_links = extract_for_item({"text": candidate_text, "label": "info"})
    candidate_by_key = {_entity_key(link): link for link in candidate_links if isinstance(link, dict)}
    reasons: list[str] = []

    for source in raw_links:
        if not isinstance(source, dict):
            continue
        kind = str(source.get("kind", ""))
        code = str(source.get("code", ""))
        if kind not in protected_kinds or not code:
            continue
        target = candidate_by_key.get((kind, code))
        if target is None:
            reasons.append(f"protected_entity_lost:{kind}:{code}")
            continue
        if str(source.get("assertion", "present")) != str(target.get("assertion", "present")):
            reasons.append(f"assertion_changed:{kind}:{code}")
        for field_name in ("value", "value2", "unit", "status"):
            if source.get(field_name) is not None and source.get(field_name) != target.get(field_name):
                reasons.append(f"entity_{field_name}_changed:{kind}:{code}")
    return reasons
