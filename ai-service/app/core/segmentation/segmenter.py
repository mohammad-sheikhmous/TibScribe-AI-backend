"""Turn raw Whisper output into sentence-like Segments.

Whisper's own segment boundaries are the PRIMARY unit (spoken dialectal Arabic lacks
reliable punctuation). A single Whisper segment is only re-split when it is unusually
long, using word timestamps (pauses as clause boundaries) when available, or a
proportional char-fraction fallback when they are not.

Two rules added in P3-10:

* **Merge the runts.** A two-word fragment ("طيب"، "آه") classified on its own is noise
  in the report; it belongs to its neighbour. Splitting existed from day one, merging
  did not.
* **Never cross a speaker change.** Merging or splitting across a speaker boundary would
  put the doctor's question and the patient's answer in one item and attach one label to
  both — undoing exactly what diarization is for.
"""
from __future__ import annotations

from typing import Any, Optional

from ..asr.diarization import DiarizationResult
from ..asr.quality import assess_segment
from ..report.schema import Segment

MIN_SEGMENT_CHARS = 20  # below this a segment is a fragment, not a sentence


def _finalize_words(word_list: list[dict[str, Any]]) -> tuple[str, float, float]:
    text = "".join(w.get("word", "") for w in word_list).strip()
    start = float(word_list[0].get("start", 0.0))
    end = float(word_list[-1].get("end", start))
    return text, start, end


def _split_by_words(
    words: list[dict[str, Any]], max_chars: int, pause_gap_sec: float
) -> list[tuple[str, float, float]]:
    """Greedy accumulation; close a chunk on char overflow OR a speech pause."""
    chunks: list[tuple[str, float, float]] = []
    cur: list[dict[str, Any]] = []
    cur_len = 0
    for w in words:
        wtext = w.get("word", "")
        wstart = float(w.get("start", 0.0))
        if cur:
            prev_end = float(cur[-1].get("end", wstart))
            gap = wstart - prev_end
            projected = cur_len + len(wtext)
            if projected > max_chars or gap > pause_gap_sec:
                chunks.append(_finalize_words(cur))
                cur = []
                cur_len = 0
        cur.append(w)
        cur_len += len(wtext)
    if cur:
        chunks.append(_finalize_words(cur))
    return chunks


def _split_by_count(
    text: str, start: float, end: float, max_chars: int
) -> list[tuple[str, float, float]]:
    """Fallback: split by word count, interpolate timestamps by char fraction."""
    words = text.split()
    chunks_text: list[str] = []
    cur: list[str] = []
    cur_len = 0
    for w in words:
        if cur and cur_len + len(w) + 1 > max_chars:
            chunks_text.append(" ".join(cur))
            cur = []
            cur_len = 0
        cur.append(w)
        cur_len += len(w) + 1
    if cur:
        chunks_text.append(" ".join(cur))

    total_chars = sum(len(c) for c in chunks_text) or 1
    duration = max(end - start, 0.0)
    result: list[tuple[str, float, float]] = []
    cum = 0
    for c in chunks_text:
        c_start = start + duration * (cum / total_chars)
        cum += len(c)
        c_end = start + duration * (cum / total_chars)
        result.append((c, c_start, c_end))
    return result


def _merge_short(
    segments: list[Segment], max_chars: int, min_chars: int
) -> list[Segment]:
    """Fold fragments into the neighbour they belong with.

    Guards: never merge across a speaker change, never exceed max_chars, and never
    merge two segments that came from different Whisper segments AND different speakers.
    """
    if not segments:
        return segments

    merged: list[Segment] = [segments[0]]
    for current in segments[1:]:
        previous = merged[-1]
        joined_len = len(previous.text) + 1 + len(current.text)
        same_speaker = previous.speaker == current.speaker
        one_is_fragment = (
            len(previous.text) < min_chars or len(current.text) < min_chars
        )
        if same_speaker and one_is_fragment and joined_len <= max_chars:
            merged[-1] = previous.model_copy(
                update={
                    "text": f"{previous.text} {current.text}".strip(),
                    "end_sec": current.end_sec,
                    # A merged item is only as trustworthy as its weakest half.
                    "asr_confidence": _min_optional(previous.asr_confidence,
                                                    current.asr_confidence),
                    "is_asr_suspect": previous.is_asr_suspect or current.is_asr_suspect,
                }
            )
            continue
        merged.append(current)

    for index, segment in enumerate(merged):
        segment.order_index = index
    return merged


def _min_optional(a: Optional[float], b: Optional[float]) -> Optional[float]:
    values = [v for v in (a, b) if v is not None]
    return min(values) if values else None


def build_segments(
    whisper_segments: list[dict[str, Any]],
    max_chars: int = 140,
    pause_gap_sec: float = 0.5,
    diarization: Optional[DiarizationResult] = None,
    min_chars: int = MIN_SEGMENT_CHARS,
) -> list[Segment]:
    out: list[Segment] = []
    order = 0
    for src_idx, seg in enumerate(whisper_segments):
        text = (seg.get("text") or "").strip()
        if not text:
            continue
        start = float(seg.get("start", 0.0))
        end = float(seg.get("end", start))

        quality = assess_segment(seg)
        speaker = diarization.cluster_for(start, end) if diarization else None

        def make(chunk_text: str, chunk_start: float, chunk_end: float, precision: str):
            nonlocal order
            # Re-resolve the speaker per chunk: a long segment may span a turn change.
            chunk_speaker = (
                diarization.cluster_for(chunk_start, chunk_end) if diarization else speaker
            )
            item = Segment(
                order_index=order,
                text=chunk_text,
                start_sec=chunk_start,
                end_sec=chunk_end,
                timestamp_precision=precision,  # type: ignore[arg-type]
                source_segment_index=src_idx,
                speaker=chunk_speaker,
                asr_confidence=quality.confidence,
                is_asr_suspect=quality.is_suspect,
            )
            order += 1
            return item

        if len(text) <= max_chars:
            out.append(make(text, start, end, "segment"))
            continue

        words = seg.get("words") or []
        if words:
            sub_chunks = _split_by_words(words, max_chars, pause_gap_sec)
            precision = "word"
        else:
            sub_chunks = _split_by_count(text, start, end, max_chars)
            precision = "interpolated"

        for chunk_text, c_start, c_end in sub_chunks:
            if chunk_text:
                out.append(make(chunk_text, c_start, c_end, precision))

    return _merge_short(out, max_chars, min_chars)
