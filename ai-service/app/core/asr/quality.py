"""ASR quality signals and hallucination detection (IMPLEMENTATION.md P3-01…P3-03).

Whisper reports how sure it was and never used to be asked: `avg_logprob`,
`no_speech_prob` and `compression_ratio` were dropped on the floor, so a hallucinated
sentence entered the report looking exactly like a real one — then got a confident
classifier label on top.

Whisper hallucinates in a characteristic way on silence and noise: it emits fluent,
repetitive text ("شكراً لمشاهدتكم" and friends) with a high `no_speech_prob`, a very low
`avg_logprob`, or an implausibly high compression ratio (repetition compresses well).
Any one of those is weak evidence; together they are strong.

**Suspect segments are flagged, never dropped** (principle P1: no information is lost).
Downstream they are excluded from firing clinical rules and are surfaced first for
review — a doctor decides, not a heuristic.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Optional

# Thresholds follow Whisper's own defaults for its internal fallback logic, which is
# the best available prior; they are settings so a deployment can tune them.
NO_SPEECH_THRESHOLD = 0.6
LOGPROB_THRESHOLD = -1.0
COMPRESSION_RATIO_THRESHOLD = 2.4
MIN_CHARS_FOR_REPETITION = 20
REPETITION_THRESHOLD = 0.5

_WORD_RE = re.compile(r"\S+")


@dataclass(frozen=True)
class SegmentQuality:
    confidence: float                 # 0..1, monotonic in avg_logprob
    is_suspect: bool
    reasons: tuple[str, ...] = ()
    no_speech_prob: Optional[float] = None
    avg_logprob: Optional[float] = None
    compression_ratio: Optional[float] = None


def logprob_to_confidence(avg_logprob: Optional[float]) -> float:
    """Map Whisper's mean token log-probability to a 0..1 score.

    `exp(avg_logprob)` is the geometric-mean token probability — the natural reading,
    and already well spread over the range Whisper produces (about -1.5 .. 0).
    """
    if avg_logprob is None:
        return 1.0  # nothing reported: do not invent doubt
    return max(0.0, min(1.0, math.exp(avg_logprob)))


def repetition_ratio(text: str) -> float:
    """Fraction of words that are repeats. Whisper loops words when it hallucinates."""
    words = _WORD_RE.findall(text)
    if len(words) < 4:
        return 0.0
    return 1.0 - (len(set(words)) / len(words))


def assess_segment(segment: dict[str, Any]) -> SegmentQuality:
    """Score one raw Whisper segment and explain any suspicion in plain terms."""
    text = str(segment.get("text") or "").strip()
    no_speech = segment.get("no_speech_prob")
    avg_logprob = segment.get("avg_logprob")
    compression = segment.get("compression_ratio")

    reasons: list[str] = []
    if no_speech is not None and no_speech > NO_SPEECH_THRESHOLD:
        reasons.append(f"no_speech_prob={no_speech:.2f} (>{NO_SPEECH_THRESHOLD})")
    if avg_logprob is not None and avg_logprob < LOGPROB_THRESHOLD:
        reasons.append(f"avg_logprob={avg_logprob:.2f} (<{LOGPROB_THRESHOLD})")
    if compression is not None and compression > COMPRESSION_RATIO_THRESHOLD:
        reasons.append(f"compression_ratio={compression:.2f} (>{COMPRESSION_RATIO_THRESHOLD})")
    if len(text) >= MIN_CHARS_FOR_REPETITION:
        ratio = repetition_ratio(text)
        if ratio >= REPETITION_THRESHOLD:
            reasons.append(f"repetition_ratio={ratio:.2f} (>={REPETITION_THRESHOLD})")

    return SegmentQuality(
        confidence=round(logprob_to_confidence(avg_logprob), 4),
        is_suspect=bool(reasons),
        reasons=tuple(reasons),
        no_speech_prob=no_speech,
        avg_logprob=avg_logprob,
        compression_ratio=compression,
    )


def transcript_quality(result: dict[str, Any]) -> dict[str, Any]:
    """Whole-file summary, stored on the transcript row for later triage."""
    segments = result.get("segments") or []
    if not segments:
        return {"avg_logprob": None, "no_speech_ratio": None,
                "suspect_segments": 0, "total_segments": 0}

    logprobs = [s["avg_logprob"] for s in segments if s.get("avg_logprob") is not None]
    suspect = sum(1 for s in segments if assess_segment(s).is_suspect)
    return {
        "avg_logprob": round(sum(logprobs) / len(logprobs), 4) if logprobs else None,
        "no_speech_ratio": round(suspect / len(segments), 4),
        "suspect_segments": suspect,
        "total_segments": len(segments),
    }


def combined_confidence(asr_confidence: Optional[float], model_confidence: float) -> float:
    """Fuse "did we hear it right?" with "did we classify it right?".

    A product, not an average: a sentence that was misheard cannot be correctly
    classified, so a low ASR score must dominate rather than be averaged away.
    """
    if asr_confidence is None:
        return round(model_confidence, 4)
    return round(max(0.0, min(1.0, asr_confidence)) * model_confidence, 4)
