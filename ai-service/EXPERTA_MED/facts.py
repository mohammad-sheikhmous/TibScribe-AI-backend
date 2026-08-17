"""Experta Fact classes — the working-memory vocabulary of the KBS.

Every fact carries provenance fields (item_id, report_index, text) so a fired
rule can point back to the exact report sentence that justified a suggestion.
"""
from __future__ import annotations

from . import _compat  # noqa: F401  (must run before `import experta` on py3.10+)
from experta import Fact


class ReportMeta(Fact):
    """job_id, index, created_at, reports_total"""


class PatientContext(Fact):
    """pregnant(bool), ga_weeks(int|None), trimester(int|None), postpartum(bool)"""


class Symptom(Fact):
    """code, negated(bool), item_id, text, report_index, confidence"""


class Vital(Fact):
    """kind('bp'|'temp'), sys, dia, value, item_id, text, report_index"""


class LabResult(Fact):
    """code, status('high'|'low'|'normal'), value, item_id, text, report_index"""


class TestDone(Fact):
    """code, item_id, report_index"""


class Medication(Fact):
    """drug, family, item_id, text, report_index"""


class Allergy(Fact):
    """drug, family, item_id, text, report_index"""


class DiagnosisNoted(Fact):
    """code, item_id, text, report_index"""


class EmergencyFlag(Fact):
    """item_id, text, report_index"""


class FollowUpPlanned(Fact):
    """item_id, report_index"""


class LowConfidenceCritical(Fact):
    """label, item_id, text, report_index"""


class Trend(Fact):
    """kind('persistent'|'worsening'), code, count(int), delta(float|None),
    evidence_ids(tuple[str]), evidence_texts(tuple[str])"""


class ClinicalFact(Fact):
    """Generic structured clinical fact used by the MD-derived rule modules.

    Fields:
      code: stable machine-readable name, e.g. ``previous_cesarean``
      value: bool, number, or string
      status/unit: optional qualifiers
      negated: whether the finding is explicitly absent
      item_id/text/report_index/confidence: provenance
    """


class RuleState(Fact):
    """Records that a rule fired, allowing lower-priority exclusion rules.

    Fields: code, category.
    """
