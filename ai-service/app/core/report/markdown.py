"""Render a Report as a comprehensive, human-readable Markdown clinical document.

Another pure consumer of the Report model (no HTTP/model coupling). Adds a doctor-
facing view on top of the raw JSON: an executive summary, an urgent-alerts block,
the four SOAP sections with per-line audio timestamps + confidence + flags, and a
disclaimer footer. Served by `GET /jobs/{id}/report.md`.
"""
from __future__ import annotations

from ..nlp.sections import SOAP_ORDER
from .schema import Report

_SOAP_LETTER = {"subjective": "S", "objective": "O", "assessment": "A", "plan": "P"}


def _t(sec: float) -> str:
    return f"{sec:.1f}s"


def _pct(x: float) -> str:
    return f"{round(x * 100)}%"


def report_to_markdown(report: Report) -> str:
    a, s = report.audio, report.summary
    L: list[str] = []

    # --- Header ---
    L += ["# التقرير الطبي المنظّم (SOAP)", ""]
    L.append(f"- **معرّف المهمة:** `{report.job_id}`")
    if a.filename:
        L.append(f"- **الملف الصوتي:** {a.filename}")
    if a.duration_sec is not None:
        L.append(f"- **مدة التسجيل:** {a.duration_sec:.1f} ثانية")
    L.append(f"- **تاريخ الإنشاء:** {report.created_at:%Y-%m-%d %H:%M} UTC")
    if a.whisper_model:
        L.append(f"- **نموذج التفريغ:** Whisper ({a.whisper_model})")
    if report.pipeline_meta.arabert_model_name:
        L.append(f"- **نموذج التصنيف:** {report.pipeline_meta.arabert_model_name}")
    L += ["", "---", ""]

    # --- Executive summary ---
    L += ["## 📋 ملخّص تنفيذي", ""]
    L.append(f"- **إجمالي الجُمل:** {s.total_segments}")
    L.append(f"- **متوسط الثقة:** {_pct(s.avg_confidence)}")
    L.append(f"- **عناصر عاجلة:** {s.urgent_count}" + (" ⚠️" if s.urgent_count else ""))
    L.append(f"- **عناصر تحتاج مراجعة (ثقة منخفضة):** {s.low_confidence_count}")
    if s.per_label_counts:
        dist = "، ".join(
            f"{k} ({v})" for k, v in sorted(s.per_label_counts.items(), key=lambda x: -x[1])
        )
        L.append(f"- **توزيع الأصناف:** {dist}")
    L.append("")

    # --- Urgent alerts (cross-cutting, before the sections) ---
    urgent = [it for sec in report.soap.values() for it in sec.items if it.is_urgent]
    if urgent:
        L += ["## 🚨 تنبيهات عاجلة", ""]
        for it in urgent:
            L.append(
                f"- **[{_t(it.start_sec)}–{_t(it.end_sec)}]** {it.text}  "
                f"\n  _({it.label_ar}، ثقة {_pct(it.confidence)})_"
            )
        L.append("")
    L += ["---", ""]

    # --- SOAP sections ---
    for key in SOAP_ORDER:
        sec = report.soap.get(key)
        if sec is None:
            continue
        letter = _SOAP_LETTER.get(key, "")
        L += [f"## {letter} — {sec.title_ar}", ""]
        if not sec.items:
            L += ["_لا يوجد._", ""]
            continue
        for it in sec.items:
            flags = ("🚨" if it.is_urgent else "") + ("🔸" if it.is_low_confidence else "")
            flags = f" {flags}" if flags else ""
            L.append(
                f"**[{_t(it.start_sec)}–{_t(it.end_sec)}]** · {it.label_ar} · "
                f"ثقة {_pct(it.confidence)}{flags}"
            )
            L += [f"> {it.text}", ""]

    # --- Footer / disclaimers ---
    L += ["---", "", "## ⚕️ ملاحظات", ""]
    L += [
        "- 🔸 عنصر منخفض الثقة — يحتاج تأكيد الطبيب.",
        "- 🚨 عنصر مُصنَّف طارئاً.",
        "- توزيع الأقسام (SOAP) آلي ويحتاج **اعتماد اختصاصي** قبل الاستخدام السريري.",
        "- هذا التقرير أداة توثيق مساعِدة؛ القرار الطبي النهائي مسؤولية الطبيب.",
        "",
    ]
    return "\n".join(L)


def _main(argv: list[str]) -> int:
    """CLI: render a saved report JSON to Markdown.

        python -m app.core.report.markdown <report.json> [-o out.md]

    With no -o, writes <report>.md next to the input file.
    """
    import argparse
    import json
    import sys
    from pathlib import Path

    from .schema import Report

    p = argparse.ArgumentParser(description="Render a report JSON to Markdown.")
    p.add_argument("json_file", help="path to a data/results/*.json report")
    p.add_argument("-o", "--out", help="output .md path (default: alongside the input)")
    args = p.parse_args(argv)

    data = json.loads(Path(args.json_file).read_text(encoding="utf-8"))
    md = report_to_markdown(Report.model_validate(data))

    out = Path(args.out) if args.out else Path(args.json_file).with_suffix(".md")
    out.write_text(md, encoding="utf-8")
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    import sys

    raise SystemExit(_main(sys.argv[1:]))
