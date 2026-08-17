"""Bridge the medical-scribe Report contract to the final EXPERTA_MED KBS.

The KBS remains a standalone package: it knows nothing about FastAPI or SQLAlchemy.
This module owns the application-specific concerns that must be resolved before a
rule engine may reason safely:

* current longitudinal obstetric state (PatientState is source of truth),
* chronological report history for explicit temporal trends,
* serialization of the pydantic Report into the KBS's plain-dict contract.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from typing import Any, Iterable, Optional

from ..nlp.context import ContextObservation, detect_context, trimester_from_weeks
from ..report.schema import Report
from ...db.patient_state import ResolvedState


@dataclass(frozen=True)
class KBSAnalysis:
    result: dict[str, Any]
    observation: ContextObservation
    patient_context: dict[str, Any]


def report_texts(report: Report) -> list[str]:
    return [item.text for section in report.soap.values() for item in section.items]


def resolve_effective_context(
    prior_state: ResolvedState,
    report: Report,
    *,
    supplemental_texts: Optional[Iterable[str]] = None,
) -> tuple[ContextObservation, dict[str, Any]]:
    """Resolve the current visit without making old pregnancy context sticky.

    SOAP/report items are the primary context source because they have already passed
    segmentation and clinical filtering.  The raw ASR transcript is an optional
    *recovery source* for structured obstetric context (especially gestational age)
    that can otherwise be lost by segmentation/routing.  Supplemental text never
    overrides an explicit clinician state tied to this visit.
    """
    observation = detect_context(report_texts(report))

    # Recovery path: if structured report text did not carry the gestational age,
    # inspect the raw transcript.  Adopt only obstetric context, not arbitrary NLP
    # labels, and require pregnancy evidence when recovering a GA.  This keeps the
    # state machine conservative while preventing a clearly spoken ``week 28`` from
    # disappearing before persistence.
    if supplemental_texts:
        supplemental = detect_context(supplemental_texts)
        if observation.ga_weeks is None and supplemental.ga_weeks is not None:
            if observation.pregnant is True or supplemental.pregnant is True:
                observation.ga_weeks = supplemental.ga_weeks
                observation.trimester = supplemental.trimester or trimester_from_weeks(
                    supplemental.ga_weeks
                )
                if observation.pregnant is None:
                    observation.pregnant = True
                observation.evidence.extend(
                    e for e in supplemental.evidence if e not in observation.evidence
                )
        elif observation.trimester is None and supplemental.trimester is not None:
            if observation.pregnant is True or supplemental.pregnant is True:
                observation.trimester = supplemental.trimester

    # A clinician-stated override tied to THIS clinical visit is authoritative over
    # transcript inference. Older explicit state remains prior context and may be
    # superseded by what a later visit actually says (e.g. delivery/postpartum).
    explicit_current_visit = bool(
        prior_state.source == "explicit"
        and prior_state.effective_at is not None
        and abs(prior_state.effective_at - report.created_at) <= timedelta(minutes=1)
    )
    if explicit_current_visit:
        status = prior_state.status
        ga_weeks = prior_state.ga_weeks_at(report.created_at) if status == "pregnant" else None
    else:
        if observation.postpartum:
            status = "postpartum"
        elif observation.pregnant is True:
            status = "pregnant"
        elif observation.pregnant is False:
            status = "unknown"
        else:
            status = prior_state.status

        ga_weeks = observation.ga_weeks
        if ga_weeks is None and status == "pregnant":
            ga_weeks = prior_state.ga_weeks_at(report.created_at)

    trimester = observation.trimester
    if trimester is None and ga_weeks is not None:
        trimester = trimester_from_weeks(ga_weeks)

    observed_status = (
        "postpartum" if observation.postpartum
        else "pregnant" if observation.pregnant is True
        else "not_pregnant" if observation.pregnant is False
        else "unchanged"
    )

    context = {
        "effective_obstetric_status": status,
        "observed_obstetric_status": observed_status,
        "pregnant": status == "pregnant",
        "postpartum": status == "postpartum",
        "gestational_age_weeks": ga_weeks,
        "trimester": trimester,
    }
    return observation, context


def _to_kbs_dict(report: Report) -> dict[str, Any]:
    return report.model_dump(mode="json")


def analyze_report_with_history(
    report: Report,
    history: Iterable[Report],
    *,
    patient_context: dict[str, Any],
) -> KBSAnalysis:
    """Run final EXPERTA_MED on current report plus chronological history.

    EXPERTA_MED itself restricts ordinary rule matching to the current report and uses
    older reports only to derive explicit temporal Trend facts. This adapter supplies
    the authoritative current patient context separately.
    """
    observation = detect_context(report_texts(report))
    reports = [_to_kbs_dict(r) for r in history]
    reports.append(_to_kbs_dict(report))
    # Lazy import keeps non-runtime tooling importable without loading Experta. The
    # FastAPI lifespan performs an explicit KBS dependency check before readiness.
    from EXPERTA_MED.runner import analyze_reports

    result = analyze_reports(reports, patient_context=patient_context)
    return KBSAnalysis(
        result=result,
        observation=observation,
        patient_context=patient_context,
    )
