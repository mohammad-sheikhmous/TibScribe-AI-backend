"""Suggestion model + renderers (JSON / Arabic Markdown) — the KBS output.

Explainability contract: every suggestion carries the rule id + version that
produced it, the evidence sentences (with item_ids for audio traceability), and a
human-readable inference chain (fact -> rule -> suggestion).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

SEVERITY_ORDER = {"critical": 0, "high": 1, "medium": 2, "low": 3}
SEVERITY_AR = {"critical": "حرجة 🚨", "high": "عالية ⚠️", "medium": "متوسطة 🔶", "low": "منخفضة ℹ️"}


@dataclass
class Suggestion:
    rule_id: str
    rule_version: str
    severity: str                    # critical | high | medium | low
    title_ar: str
    detail_ar: str
    condition: Optional[str] = None  # suspected condition code (ontology)
    missing_tests: list[str] = field(default_factory=list)
    evidence: list[dict] = field(default_factory=list)   # {item_id, text, report_index}
    chain: list[str] = field(default_factory=list)       # inference chain (Arabic)
    references: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "rule_id": self.rule_id,
            "rule_version": self.rule_version,
            "severity": self.severity,
            "title_ar": self.title_ar,
            "detail_ar": self.detail_ar,
            "condition": self.condition,
            "missing_tests": self.missing_tests,
            "evidence": self.evidence,
            "inference_chain": self.chain,
            "references": self.references,
        }


def _merge_unique(existing: list, incoming: list, *, key_fn) -> list:
    out = list(existing)
    seen = {key_fn(value) for value in out}
    for value in incoming:
        key = key_fn(value)
        if key in seen:
            continue
        seen.add(key)
        out.append(value)
    return out


def merge_suggestions(current: Suggestion, incoming: Suggestion) -> Suggestion:
    """Losslessly merge another firing of the same logical recommendation.

    Used by the engine's dedupe layer: the clinician sees one recommendation card,
    while all independent evidence, missing tests, references and reasoning steps are
    retained for auditability.
    """
    evidence_key = lambda ev: (ev.get("item_id"), ev.get("fact_code"), ev.get("text"))
    current.evidence = _merge_unique([], current.evidence, key_fn=evidence_key)
    current.evidence = _merge_unique(current.evidence, incoming.evidence, key_fn=evidence_key)
    current.missing_tests = _merge_unique([], current.missing_tests, key_fn=lambda x: x)
    current.missing_tests = _merge_unique(
        current.missing_tests, incoming.missing_tests, key_fn=lambda x: x
    )
    current.chain = _merge_unique([], current.chain, key_fn=lambda x: x)
    current.chain = _merge_unique(current.chain, incoming.chain, key_fn=lambda x: x)
    current.references = _merge_unique([], current.references, key_fn=lambda x: x)
    current.references = _merge_unique(
        current.references, incoming.references, key_fn=lambda x: x
    )
    return current


def sort_suggestions(suggestions: list[Suggestion]) -> list[Suggestion]:
    return sorted(suggestions, key=lambda s: (SEVERITY_ORDER.get(s.severity, 9), s.rule_id))


def to_markdown(suggestions: list[Suggestion], header: dict | None = None) -> str:
    from .ontology import ar

    L: list[str] = ["# اقتراحات النظام الخبير (EXPERTA_MED)", ""]
    if header:
        for k, v in header.items():
            L.append(f"- **{k}:** {v}")
        L.append("")
    if not suggestions:
        L += ["_لا توجد اقتراحات — لم تُطلق أي قاعدة._", ""]
    for i, s in enumerate(sort_suggestions(suggestions), 1):
        L.append(f"## {i}. {s.title_ar}")
        L.append(f"- **الخطورة:** {SEVERITY_AR.get(s.severity, s.severity)}")
        L.append(f"- **القاعدة:** `{s.rule_id}` (v{s.rule_version})")
        if s.condition:
            L.append(f"- **الحالة المشتبهة:** {ar(s.condition)}")
        if s.missing_tests:
            L.append("- **فحوصات ناقصة مقترحة:** " + "، ".join(ar(t) for t in s.missing_tests))
        L.append(f"- **التفصيل:** {s.detail_ar}")
        if s.evidence:
            L.append("- **الدليل من التقرير:**")
            for ev in s.evidence:
                L.append(f"  - «{ev.get('text', '')}» _(العنصر `{ev.get('item_id', '')}`)_")
        if s.chain:
            L.append("- **سلسلة الاستدلال:**")
            for step in s.chain:
                L.append(f"  1. {step}")
        if s.references:
            L.append("- **المصادر:**")
            for ref in s.references:
                L.append(f"  - {ref}")
        L.append("")
    L += ["---", "",
          "> ⚕️ هذه اقتراحات آلية **للمراجعة فقط** — القرار السريري النهائي مسؤولية الطبيب،",
          "> والقواعد تحتاج اعتماد اختصاصي توليد قبل الاستخدام الفعلي.", ""]
    return "\n".join(L)
