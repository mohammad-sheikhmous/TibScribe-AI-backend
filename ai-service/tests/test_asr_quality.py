"""ASR quality signals, hallucination flagging and speaker-aware segmentation (P3).

Runs against synthetic Whisper output — the signals under test are fields Whisper
already returns, so no audio or model is needed to pin the logic.
"""
import pytest

from app.core.asr.diarization import DiarizationResult, NoOpDiarizer, SpeakerTurn
from app.core.asr.quality import (
    assess_segment,
    combined_confidence,
    logprob_to_confidence,
    repetition_ratio,
    transcript_quality,
)
from app.core.segmentation.segmenter import build_segments


def seg(text="جملة طبية طبيعية وواضحة", start=0.0, end=3.0, **overrides):
    base = {
        "text": text, "start": start, "end": end,
        "avg_logprob": -0.25, "no_speech_prob": 0.05, "compression_ratio": 1.4,
    }
    base.update(overrides)
    return base


# --- confidence ---------------------------------------------------------------------

def test_confidence_is_monotonic_in_logprob():
    assert logprob_to_confidence(-0.1) > logprob_to_confidence(-1.0) > logprob_to_confidence(-3.0)
    assert 0.0 <= logprob_to_confidence(-5.0) <= 1.0


def test_missing_logprob_does_not_invent_doubt():
    assert logprob_to_confidence(None) == 1.0


def test_combined_confidence_lets_bad_audio_dominate():
    """A misheard sentence cannot be correctly classified, however sure the model is."""
    assert combined_confidence(0.2, 0.99) < 0.25
    assert combined_confidence(None, 0.8) == 0.8


# --- hallucination detection -----------------------------------------------------------

def test_clean_segment_is_not_suspect():
    assert assess_segment(seg()).is_suspect is False


@pytest.mark.parametrize(
    "overrides,expected_reason",
    [
        ({"no_speech_prob": 0.9}, "no_speech_prob"),
        ({"avg_logprob": -2.0}, "avg_logprob"),
        ({"compression_ratio": 3.5}, "compression_ratio"),
    ],
)
def test_each_signal_can_flag_a_segment(overrides, expected_reason):
    quality = assess_segment(seg(**overrides))
    assert quality.is_suspect
    assert any(expected_reason in reason for reason in quality.reasons)


def test_repeated_text_is_flagged():
    """Whisper loops phrases when it hallucinates over silence."""
    looping = "شكرا لمشاهدتكم شكرا لمشاهدتكم شكرا لمشاهدتكم شكرا لمشاهدتكم"
    assert repetition_ratio(looping) >= 0.5
    assert assess_segment(seg(text=looping)).is_suspect


def test_short_text_is_not_judged_on_repetition():
    assert repetition_ratio("نعم نعم") == 0.0


def test_transcript_summary_counts_suspect_segments():
    result = {"segments": [seg(), seg(no_speech_prob=0.95), seg(avg_logprob=-2.5)]}
    summary = transcript_quality(result)
    assert summary["total_segments"] == 3 and summary["suspect_segments"] == 2
    assert summary["avg_logprob"] is not None


def test_empty_transcript_summary_is_safe():
    assert transcript_quality({"segments": []})["total_segments"] == 0


# --- segmentation: merging + speaker boundaries ------------------------------------------

def test_short_fragments_are_merged_into_their_neighbour():
    segments = build_segments([
        seg(text="طيب", start=0.0, end=0.5),
        seg(text="تشتكي من صداع شديد منذ يومين مع دوخة", start=0.5, end=4.0),
    ])
    assert len(segments) == 1
    assert segments[0].text.startswith("طيب")
    assert segments[0].end_sec == 4.0


def test_merging_never_crosses_a_speaker_change():
    """The doctor's prompt and the patient's answer must stay separate items."""
    diarization = DiarizationResult(
        turns=[
            SpeakerTurn(0.0, 1.0, "SPEAKER_00"),
            SpeakerTurn(1.0, 5.0, "SPEAKER_01"),
        ],
        backend="test",
    )
    segments = build_segments(
        [seg(text="طيب", start=0.0, end=0.9),
         seg(text="عندي صداع من يومين", start=1.1, end=4.0)],
        diarization=diarization,
    )
    assert len(segments) == 2
    assert segments[0].speaker == "SPEAKER_00"
    assert segments[1].speaker == "SPEAKER_01"


def test_merging_respects_the_length_cap():
    long_a = "أ" * 100
    long_b = "ب" * 100
    segments = build_segments([seg(text=long_a), seg(text=long_b)], max_chars=140)
    assert len(segments) == 2


def test_order_index_is_contiguous_after_merging():
    segments = build_segments([
        seg(text="آه", start=0.0, end=0.3),
        seg(text="نعم", start=0.3, end=0.6),
        seg(text="تشتكي من ألم أسفل البطن منذ ثلاثة أيام", start=0.6, end=4.0),
        seg(text="والفحص يبين التهاب بالحوض يحتاج مضاد حيوي", start=4.0, end=8.0),
    ])
    assert [s.order_index for s in segments] == list(range(len(segments)))


def test_quality_flags_propagate_to_segments():
    segments = build_segments([seg(text="جملة واضحة ومفهومة تماما", no_speech_prob=0.95)])
    assert segments[0].is_asr_suspect is True
    assert segments[0].asr_confidence is not None


def test_merged_segment_keeps_the_weaker_confidence():
    """A merge must not launder a suspect fragment into a clean sentence."""
    segments = build_segments([
        seg(text="آه", start=0.0, end=0.3, no_speech_prob=0.95, avg_logprob=-2.5),
        seg(text="تشتكي من صداع شديد منذ يومين", start=0.3, end=3.0),
    ])
    assert len(segments) == 1
    assert segments[0].is_asr_suspect is True


def test_noop_diarizer_leaves_speakers_unset():
    result = NoOpDiarizer().diarize("x.wav")
    assert result.clusters == []
    segments = build_segments([seg()], diarization=result)
    assert segments[0].speaker is None


def test_cluster_lookup_uses_greatest_overlap_not_the_midpoint():
    """A segment straddling a turn change belongs to whoever said most of it."""
    diarization = DiarizationResult(
        turns=[SpeakerTurn(0.0, 3.5, "A"), SpeakerTurn(3.5, 4.0, "B")], backend="test"
    )
    assert diarization.cluster_for(0.0, 4.0) == "A"
