from __future__ import annotations

from app.core.nlp.canonicalization import (
    LLMMedicalCanonicalizationStage,
    NoOpCanonicalizationStage,
    validate_canonicalization,
)
from app.core.report.builder import build_report
from app.core.report.schema import ClassifiedSegment, Segment


class FakeClient:
    def __init__(self, mapping=None, *, fail=False):
        self.mapping = mapping or {}
        self.fail = fail
        self.calls = []

    def canonicalize_batch(self, items):
        self.calls.append(items)
        if self.fail:
            raise RuntimeError("provider failed")
        return {
            str(item["segment_id"]): self.mapping.get(str(item["segment_id"]), item["text"])
            for item in items
        }


def _seg(index: int, text: str, *, suspect: bool = False) -> Segment:
    return Segment(
        order_index=index,
        text=text,
        start_sec=float(index),
        end_sec=float(index) + 0.5,
        source_segment_index=index,
        is_asr_suspect=suspect,
    )


def test_safe_canonicalization_keeps_numbers_and_negation():
    assert validate_canonicalization("ضغطها 128 على 82", "بلغ ضغط الدم 128/82.")[0]
    assert validate_canonicalization("ما عندها نزيف", "لا تعاني من نزيف.")[0]


def test_unsafe_number_or_negation_change_is_rejected():
    ok, reason = validate_canonicalization("ضغطها 128 على 82", "بلغ ضغط الدم 138/92.")
    assert not ok and reason == "numeric values changed"

    ok, reason = validate_canonicalization("ما عندها نزيف", "تعاني من نزيف.")
    assert not ok and reason == "negation polarity changed"


def test_noop_preserves_raw_text():
    original = _seg(0, "رجليها شوي متنفخين")
    item = NoOpCanonicalizationStage().canonicalize_segments([original])[0]
    assert item.text_raw == original.text
    assert item.text == original.text


def test_llm_stage_accepts_safe_rewrite_rejects_unsafe_and_skips_suspect():
    client = FakeClient({
        "0": "تعاني من صداع خفيف.",
        "1": "بلغ ضغط الدم 138/92.",
        "2": "جملة مصقولة لا يجب اعتمادها",
    })
    stage = LLMMedicalCanonicalizationStage(client, batch_size=2, skip_asr_suspect=True)
    out = stage.canonicalize_segments([
        _seg(0, "في عندها وجع راس خفيف"),
        _seg(1, "ضغطها 128 على 82"),
        _seg(2, "هلوسة محتملة", suspect=True),
    ])

    assert out[0].text == "تعاني من صداع خفيف."
    assert out[0].text_raw == "في عندها وجع راس خفيف"
    assert out[1].text == "ضغطها 128 على 82"
    assert out[2].text == "هلوسة محتملة"
    assert all(
        item["segment_id"] != "2"
        for call in client.calls
        for item in call
    )


def test_provider_failure_falls_back_to_original():
    out = LLMMedicalCanonicalizationStage(FakeClient(fail=True)).canonicalize_segments(
        [_seg(0, "في عندها وجع راس")]
    )[0]
    assert out.text_raw == "في عندها وجع راس"
    assert out.text == "في عندها وجع راس"


def test_report_builder_carries_raw_and_canonical_text():
    seg = ClassifiedSegment(
        order_index=0,
        text_raw="في عندها وجع راس",
        text="تعاني من صداع",
        start_sec=0,
        end_sec=1,
        source_segment_index=0,
        label="symptom",
        confidence=0.9,
    )
    report = build_report("canonical-job", [seg])
    item = next(item for section in report.soap.values() for item in section.items)
    assert item.text_raw == "في عندها وجع راس"
    assert item.text == "تعاني من صداع"
