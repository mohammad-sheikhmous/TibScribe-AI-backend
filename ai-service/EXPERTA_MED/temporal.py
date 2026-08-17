"""Temporal reasoning across a chronological series of reports for one patient.

Produces Trend fact-specs the engine turns into rules R11 (persistent symptom) and
R12 (worsening blood pressure). This is what makes a *series* of reports more than
the sum of single-report analyses.
"""
from __future__ import annotations

from collections import defaultdict

from .extraction import ReportExtraction

PERSISTENT_MIN_REPORTS = 2
BP_WORSENING_DELTA = 10.0


def compute_trends(extractions: list[ReportExtraction]) -> list[dict]:
    trends: list[dict] = []
    ordered = sorted(extractions, key=lambda e: e.index)

    # --- persistent (non-negated) symptoms across >= N distinct reports ---
    sym: dict[str, list[tuple[int, str, str]]] = defaultdict(list)
    for ext in ordered:
        seen: set[str] = set()
        for e in ext.entities:
            if e.kind == "symptom" and not e.negated and e.code not in seen:
                seen.add(e.code)
                sym[e.code].append((ext.index, e.item_id, e.text))
    for code, occ in sym.items():
        if len({r for r, _, _ in occ}) >= PERSISTENT_MIN_REPORTS:
            trends.append({
                "kind": "persistent", "code": code, "count": len(occ),
                "evidence_ids": tuple(i for _, i, _ in occ),
                "evidence_texts": tuple(t for _, _, t in occ),
            })

    # --- worsening systolic BP first -> last ---
    bp: list[tuple[int, float, str, str]] = []
    for ext in ordered:
        for e in ext.entities:
            if e.kind == "vital" and e.code == "bp" and e.value:
                bp.append((ext.index, float(e.value), e.item_id, e.text))
    bp.sort(key=lambda x: x[0])
    if len(bp) >= 2:
        delta = bp[-1][1] - bp[0][1]
        if delta >= BP_WORSENING_DELTA:
            trends.append({
                "kind": "worsening", "code": "bp", "delta": round(delta, 1),
                "evidence_ids": (bp[0][2], bp[-1][2]),
                "evidence_texts": (bp[0][3], bp[-1][3]),
            })
    return trends