from app.core.report.builder import build_report
from app.core.report.markdown import report_to_markdown
from app.core.report.schema import ClassifiedSegment


def _seg(order, text, label, conf):
    return ClassifiedSegment(
        order_index=order, text=text, start_sec=float(order), end_sec=float(order) + 1,
        source_segment_index=order, label=label, confidence=conf,
    )


def test_markdown_has_all_sections_and_flags():
    report = build_report(
        "job1",
        [
            _seg(0, "صداع شديد", "symptom", 0.95),
            _seg(1, "الضغط مرتفع", "vital", 0.80),
            _seg(2, "حالة طارئة", "emergency", 0.99),
            _seg(3, "متابعة لاحقة", "follow_up", 0.30),
        ],
        low_confidence_threshold=0.5,
    )
    md = report_to_markdown(report)

    # all four SOAP headers present
    for title in ["Subjective", "Objective", "Assessment", "Plan"]:
        assert title in md
    # executive summary + urgent alerts
    assert "ملخّص تنفيذي" in md
    assert "تنبيهات عاجلة" in md
    # flags rendered
    assert "🚨" in md  # emergency item
    assert "🔸" in md  # low-confidence item
    # traceable timestamps
    assert "0.0s" in md
    assert md.strip().endswith("مسؤولية الطبيب.")
