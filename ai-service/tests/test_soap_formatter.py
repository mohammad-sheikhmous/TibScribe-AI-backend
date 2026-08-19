from app.core.report.builder import build_report
from app.core.report.schema import ClassifiedSegment


def seg(order, text, label, start=None):
    return ClassifiedSegment(
        order_index=order,
        text=text,
        start_sec=float(order if start is None else start),
        end_sec=float(order + 1),
        source_segment_index=order,
        label=label,
        confidence=0.95,
    )


def test_formatter_creates_doctor_facing_sections_without_changing_facts():
    report = build_report("fmt", [
        seg(0, "تعاني من صداع خفيف متقطع", "symptom"),
        seg(1, "لا تعاني من تشوش في الرؤية", "history"),
        seg(2, "ضغط الدم 128 على 82", "vital"),
        seg(3, "الهيموغلوبين 11.5", "lab"),
        seg(4, "وصف الحديد وحمض الفوليك", "medication"),
    ])
    assert report.pipeline_meta.soap_formatter_applied is True
    assert "صداع خفيف" in report.soap_formatted["subjective"].text
    assert "128 على 82" in report.soap_formatted["objective"].text
    assert "11.5" in report.soap_formatted["objective"].text
    assert "الحديد" in report.soap_formatted["plan"].text
    # The formatter is extractive/fact-constrained: it must not manufacture values.
    joined = " ".join(section.text for section in report.soap_formatted.values())
    assert "138" not in joined
    assert "10.5" not in joined


def test_formatter_suppresses_exact_duplicates_only():
    report = build_report("dup", [
        seg(0, "لا يوجد نزيف", "symptom"),
        seg(1, "لا يوجد نزيف", "symptom"),
    ])
    text = report.soap_formatted["subjective"].text
    assert text.count("لا يوجد نزيف") == 1
    assert len(report.soap_formatted["subjective"].item_ids) == 1
