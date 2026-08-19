"""Medical canonicalization stage before AraBERT classification.

Purpose
-------
Whisper may produce dialectal Arabic, spelling errors, or obvious ASR wording
errors. AraBERT, however, was fine-tuned on cleaner/formal Arabic data.

This stage converts each segmented ASR sentence into a medically-safe,
canonical Arabic form BEFORE classification.

Important safety rule:
    The original ASR text is never destroyed.

    Segment.text_raw
        Original Whisper/segmentation text.

    Segment.text
        Canonical text that will be consumed by AraBERT and downstream NLP.

The actual LLM provider is intentionally abstracted behind
`CanonicalizationClient` so this stage is not coupled to OpenAI, Gemini,
or any other provider.
"""

from __future__ import annotations

from collections import Counter
import logging
import re
from typing import Mapping, Protocol, Sequence, runtime_checkable

from ..report.schema import Segment


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Safety helpers
# ---------------------------------------------------------------------------

# Arabic-Indic + Persian digits -> Western digits.
_DIGIT_TRANSLATION = str.maketrans(
    "٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹",
    "01234567890123456789",
)

# Matches:
#   32
#   128
#   36.8
#   11,5
# etc.
_NUMBER_RE = re.compile(
    r"(?<!\w)[+-]?\d+(?:[.,]\d+)?(?!\w)"
)

# Common negation forms in formal and dialectal Arabic.
#
# We intentionally compare NEGATION PRESENCE rather than the exact word:
#
#   "ما عندها نزيف"
#           ↓
#   "لا تعاني من نزيف"
#
# is a valid canonicalization even though "ما" became "لا".
_NEGATION_RE = re.compile(
    r"(?<!\w)"
    r"(?:"
    r"لا|"
    r"لم|"
    r"لن|"
    r"ليس|"
    r"ليست|"
    r"ليسوا|"
    r"ما|"
    r"مش|"
    r"مو|"
    r"بدون|"
    r"دون|"
    r"مافي|"
    r"مافيه"
    r")"
    r"(?!\w)"
)


def _clean_text(text: str) -> str:
    """Collapse unnecessary whitespace without changing medical content."""

    return " ".join(str(text or "").split()).strip()


def _normalize_digits(text: str) -> str:
    """Normalize Arabic/Persian digits for safety comparison."""

    normalized = str(text or "").translate(_DIGIT_TRANSLATION)

    # Arabic decimal separator.
    normalized = normalized.replace("٫", ".")

    # Arabic thousands separator.
    normalized = normalized.replace("٬", "")

    return normalized


def _extract_numbers(text: str) -> list[str]:
    """Return normalized numeric tokens while preserving their values."""

    normalized = _normalize_digits(text)

    values: list[str] = []

    for match in _NUMBER_RE.finditer(normalized):
        value = match.group(0)

        # Treat decimal comma like decimal point for comparison.
        value = value.replace(",", ".")

        values.append(value)

    return values


def _contains_negation(text: str) -> bool:
    """Return True when the sentence contains an explicit negation cue."""

    return bool(_NEGATION_RE.search(str(text or "")))


def _contains_question_mark(text: str) -> bool:
    """Protect explicitly marked questions from being converted to statements."""

    value = str(text or "")
    return "؟" in value or "?" in value


def _numbers_preserved(original: str, corrected: str) -> bool:
    """Require the exact same numeric values before and after the LLM.

    Counter is used instead of simple set equality because repeated values matter.

    Example:

        original:
            الضغط 128 على 82

        corrected:
            ضغط الدم 128/82

    Both contain:

        ["128", "82"]

    so the correction is safe.

    But:

        128/82 -> 138/92

    is rejected.
    """

    original_numbers = Counter(_extract_numbers(original))
    corrected_numbers = Counter(_extract_numbers(corrected))

    return original_numbers == corrected_numbers


def _negation_preserved(original: str, corrected: str) -> bool:
    """Prevent the LLM from adding or removing clinical negation."""

    return _contains_negation(original) == _contains_negation(corrected)


def _question_status_preserved(original: str, corrected: str) -> bool:
    """Do not allow an explicitly marked question to become a statement."""

    if _contains_question_mark(original):
        return _contains_question_mark(corrected)

    return True


def _reasonable_length(original: str, corrected: str) -> bool:
    """Reject extreme expansion or deletion.

    Canonicalization may naturally change sentence length, but a very large
    difference can indicate that the model summarized, invented, or removed
    clinical information.
    """

    original_len = len(_clean_text(original))
    corrected_len = len(_clean_text(corrected))

    if original_len == 0:
        return corrected_len == 0

    if corrected_len == 0:
        return False

    ratio = corrected_len / original_len

    # Deliberately conservative but still allows dialect -> formal Arabic.
    return 0.40 <= ratio <= 2.50


def validate_canonicalization(
    original: str,
    corrected: str,
) -> tuple[bool, str | None]:
    """Validate an LLM correction before allowing it into AraBERT.

    Returns:
        (True, None)
            candidate is safe enough to use.

        (False, reason)
            candidate must be rejected and the original text retained.
    """

    original = _clean_text(original)
    corrected = _clean_text(corrected)

    if not corrected:
        return False, "empty canonicalization result"

    if not _numbers_preserved(original, corrected):
        return False, "numeric values changed"

    if not _negation_preserved(original, corrected):
        return False, "negation polarity changed"

    if not _question_status_preserved(original, corrected):
        return False, "question changed into a statement"

    if not _reasonable_length(original, corrected):
        return False, "canonicalized text changed length excessively"

    return True, None


# ---------------------------------------------------------------------------
# Contracts
# ---------------------------------------------------------------------------


@runtime_checkable
class CanonicalizationClient(Protocol):
    """Provider-independent contract for the medical LLM client.

    The client receives several segments in one request so the LLM has enough
    conversational context while still preserving segment boundaries.

    Expected input example:

        [
            {
                "segment_id": "0",
                "speaker": "patient",
                "text": "في عندها وجع راس خفيف"
            },
            {
                "segment_id": "1",
                "speaker": "doctor",
                "text": "ضغطها طلع 128 على 82"
            }
        ]

    Expected output:

        {
            "0": "تعاني من صداع خفيف.",
            "1": "بلغ ضغط الدم 128/82."
        }
    """

    def canonicalize_batch(
        self,
        items: list[dict[str, str]],
    ) -> Mapping[str, str]:
        """Return mapping: segment_id -> canonical Arabic text."""
        ...


@runtime_checkable
class CanonicalizationStage(Protocol):
    """Contract implemented by pre-AraBERT canonicalization stages."""

    @property
    def applied(self) -> bool:
        """Whether a real canonicalization backend is active."""
        ...

    def canonicalize_segments(
        self,
        segments: Sequence[Segment],
    ) -> list[Segment]:
        """Return canonicalized segments while preserving order."""
        ...


# ---------------------------------------------------------------------------
# Disabled / fallback stage
# ---------------------------------------------------------------------------


class NoOpCanonicalizationStage:
    """Safe identity stage used when LLM canonicalization is disabled.

    Even when disabled, `text_raw` is populated so the text provenance remains
    explicit and the rest of the pipeline can use the same data model.
    """

    applied = False

    def canonicalize_segments(
        self,
        segments: Sequence[Segment],
    ) -> list[Segment]:

        output: list[Segment] = []

        for segment in segments:
            original = segment.text_raw or segment.text

            output.append(
                segment.model_copy(
                    update={
                        "text_raw": original,
                        "text": original,
                    }
                )
            )

        return output


# ---------------------------------------------------------------------------
# LLM-backed medical canonicalization
# ---------------------------------------------------------------------------


class LLMMedicalCanonicalizationStage:
    """Canonicalize dialectal/ASR Arabic before AraBERT classification.

    Responsibilities:

    - preserve original ASR text in `text_raw`;
    - send small batches to the medical LLM;
    - preserve segment boundaries;
    - reject unsafe changes;
    - never allow an LLM/API failure to break the medical pipeline;
    - skip ASR-suspect segments rather than making hallucinated speech look
      medically plausible.
    """

    applied = True

    def __init__(
        self,
        client: CanonicalizationClient,
        *,
        batch_size: int = 12,
        skip_asr_suspect: bool = True,
    ):
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than zero")

        self.client = client
        self.batch_size = batch_size
        self.skip_asr_suspect = skip_asr_suspect

    def canonicalize_segments(
        self,
        segments: Sequence[Segment],
    ) -> list[Segment]:

        if not segments:
            return []

        # First preserve raw text for every segment.
        prepared: list[Segment] = [
            segment.model_copy(
                update={
                    "text_raw": segment.text_raw or segment.text,
                }
            )
            for segment in segments
        ]

        result: list[Segment] = list(prepared)

        # Only safe/non-suspect segments are sent to the LLM.
        candidate_indices = [
            index
            for index, segment in enumerate(prepared)
            if not (
                self.skip_asr_suspect
                and segment.is_asr_suspect
            )
        ]

        for batch_start in range(
            0,
            len(candidate_indices),
            self.batch_size,
        ):
            batch_indices = candidate_indices[
                batch_start : batch_start + self.batch_size
            ]

            batch_segments = [
                prepared[index]
                for index in batch_indices
            ]

            canonicalized = self._canonicalize_batch_safely(
                batch_segments
            )

            for list_index, corrected_segment in zip(
                batch_indices,
                canonicalized,
            ):
                result[list_index] = corrected_segment

        return result

    def _canonicalize_batch_safely(
        self,
        segments: Sequence[Segment],
    ) -> list[Segment]:
        """Call the LLM and safely fall back on any error."""

        payload = [
            {
                "segment_id": str(segment.order_index),
                "speaker": str(segment.speaker or "unknown"),
                "text": segment.text_raw or segment.text,
            }
            for segment in segments
        ]

        try:
            response = self.client.canonicalize_batch(payload)

            # Be tolerant if a provider returns integer-like keys.
            corrected_by_id = {
                str(key): str(value)
                for key, value in response.items()
                if value is not None
            }

        except Exception:  # noqa: BLE001
            # Canonicalization is an enhancement.
            # An LLM/network/provider failure must not fail the medical job.
            logger.exception(
                "Medical canonicalization failed; "
                "falling back to original ASR text"
            )

            return [
                self._original_segment(segment)
                for segment in segments
            ]

        output: list[Segment] = []

        for segment in segments:
            segment_id = str(segment.order_index)
            original = segment.text_raw or segment.text

            candidate = corrected_by_id.get(segment_id)

            if candidate is None:
                logger.warning(
                    "Canonicalizer returned no result for segment %s; "
                    "using original text",
                    segment_id,
                )

                output.append(
                    self._original_segment(segment)
                )
                continue

            candidate = _clean_text(candidate)

            safe, reason = validate_canonicalization(
                original,
                candidate,
            )

            if not safe:
                logger.warning(
                    "Rejected canonicalization for segment %s: %s | "
                    "original=%r | candidate=%r",
                    segment_id,
                    reason,
                    original,
                    candidate,
                )

                output.append(
                    self._original_segment(segment)
                )
                continue

            output.append(
                segment.model_copy(
                    update={
                        "text_raw": original,
                        "text": candidate,
                    }
                )
            )

        return output

    @staticmethod
    def _original_segment(segment: Segment) -> Segment:
        """Return the untouched ASR text as the working text."""

        original = segment.text_raw or segment.text

        return segment.model_copy(
            update={
                "text_raw": original,
                "text": original,
            }
        )
