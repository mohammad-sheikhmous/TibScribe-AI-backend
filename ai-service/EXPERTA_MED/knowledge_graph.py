"""Domain knowledge graph: typed relations between clinical entities.

Edges are (source, relation, target) triples over ontology codes:
  indicates      symptom/finding -> condition   (raises suspicion)
  confirms       lab finding     -> condition   (supports diagnosis)
  requires_test  condition       -> test        (standard workup)
  treated_by     condition       -> drug/action

The rule engine queries this graph at suggestion time (e.g. "which workup tests
for preeclampsia are still missing?"), so the graph is functional, not decorative.
Curated from standard OB/GYN practice; NEEDS CLINICIAN SIGN-OFF before real use.
"""
from __future__ import annotations

from collections import defaultdict

EDGES: list[tuple[str, str, str]] = [
    # --- preeclampsia ---
    ("headache", "indicates", "preeclampsia"),
    ("blurred_vision", "indicates", "preeclampsia"),
    ("epigastric_pain", "indicates", "preeclampsia"),
    ("edema", "indicates", "preeclampsia"),
    ("bp_high", "indicates", "preeclampsia"),
    ("urine_protein", "confirms", "preeclampsia"),
    # validity_days: how long a result still answers the question. Without it, "the
    # test was done in 2024" wrongly satisfies "the workup is complete" (gap ك-٥).
    ("preeclampsia", "requires_test", "bp_measurement", {"validity_days": 1}),
    ("preeclampsia", "requires_test", "urine_protein_test", {"validity_days": 7}),
    ("preeclampsia", "requires_test", "cbc", {"validity_days": 14}),
    ("preeclampsia", "requires_test", "liver_function", {"validity_days": 14}),
    ("preeclampsia", "treated_by", "magnesium_sulfate"),
    # --- gestational diabetes ---
    ("gdm", "requires_test", "fasting_glucose"),
    ("gdm", "requires_test", "hba1c_test"),
    # --- fetal wellbeing ---
    ("reduced_fetal_movement", "indicates", "preterm_labor"),
    ("preterm_labor", "requires_test", "ctg"),
    ("preterm_labor", "requires_test", "obstetric_ultrasound"),
    # --- bleeding ---
    ("vaginal_bleeding", "indicates", "postpartum_hemorrhage"),
    ("postpartum_hemorrhage", "requires_test", "cbc"),
    ("postpartum_hemorrhage", "requires_test", "obstetric_ultrasound"),
    # --- infection ---
    ("fever", "indicates", "infection"),
    ("temp_high", "indicates", "infection"),
    ("dysuria", "indicates", "uti"),
    ("infection", "requires_test", "cbc"),
    ("infection", "requires_test", "blood_culture"),
    ("uti", "requires_test", "urine_culture"),
    ("puerperal_infection", "requires_test", "cbc"),
    ("puerperal_infection", "requires_test", "wound_exam"),
    ("puerperal_infection", "requires_test", "blood_culture"),
    # --- anemia ---
    ("anemia", "requires_test", "cbc"),
    ("anemia", "treated_by", "iron"),

    # --- test containment -------------------------------------------------------
    # A broader test answers a narrower question. Without these, a rule asking for
    # `urine_protein_test` ignores a doctor who said "عملنا تحليل بول" — and reports a
    # missing test that was in fact performed.
    ("urine_analysis", "satisfies", "urine_protein_test"),
    ("cbc", "satisfies", "hemoglobin_test"),
    ("cbc", "satisfies", "platelets_test"),
    ("ogtt", "satisfies", "fasting_glucose"),
    ("thyroid_function", "satisfies", "tsh_test"),
]


class KnowledgeGraph:
    """Typed relations over ontology codes.

    Edges are 3-tuples, or 4-tuples with an attribute mapping. The attribute that
    matters today is `validity_days` on `requires_test`: a urine-protein result is
    stale within a week, an HbA1c is not, and treating "ever done" as "done" is what let
    a two-year-old test silence a live alert (gap ك-٥).
    """

    def __init__(self, edges: list[tuple] | None = None):
        self.edges = edges or EDGES
        self._by_rel: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
        self._attrs: dict[tuple[str, str, str], dict] = {}
        for edge in self.edges:
            source, relation, target = edge[0], edge[1], edge[2]
            attrs = edge[3] if len(edge) > 3 and isinstance(edge[3], dict) else {}
            self._by_rel[relation][source].append(target)
            self._attrs[(source, relation, target)] = attrs

    def required_tests(self, condition: str) -> list[str]:
        return list(self._by_rel["requires_test"].get(condition, []))

    def test_validity_days(self, condition: str, test: str) -> float | None:
        """How long this test stays valid for this condition, if the edge says."""
        return self._attrs.get((condition, "requires_test", test), {}).get("validity_days")

    def tests_satisfying(self, test: str) -> set[str]:
        """Every test that answers `test`, including itself.

        "عملنا تحليل بول" must count as a urine-protein test; otherwise the workup is
        reported incomplete although it was done.
        """
        satisfying = {test}
        for source, targets in self._by_rel["satisfies"].items():
            if test in targets:
                satisfying.add(source)
        return satisfying

    def indicators(self, condition: str) -> list[str]:
        return [s for s, rel, t in self.edges if rel == "indicates" and t == condition]

    def treatments(self, condition: str) -> list[str]:
        return list(self._by_rel["treated_by"].get(condition, []))

    def to_dict(self) -> dict:
        return {"edges": [{"source": s, "relation": r, "target": t} for s, r, t in self.edges]}
