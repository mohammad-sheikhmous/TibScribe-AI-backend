"""Assemble a structured SOAP report from classified segments."""
from __future__ import annotations

from datetime import datetime, timezone
import re
from typing import Optional, Sequence

from ..asr.quality import combined_confidence
from ..nlp.sections import SOAP_ORDER, SOAP_TITLES_AR, label_ar, soap_for_item, text_implies_plan
from .formatter import ClinicalSoapFormatter
from .schema import (
    AudioMeta,
    ClassifiedSegment,
    PipelineMeta,
    Report,
    ReportItem,
    ReportSection,
    ReportSummary,
)


HIGH_ENTROPY = 0.75   # near-flat distribution over 20 classes
HIGH_OOD = 0.65       # advisory OOD signal
VERY_HIGH_OOD = 0.95  # extreme OOD with no trusted clinical evidence

_URGENT_ACTION_RE = re.compile(
    r"(?:تقييم|تقويم|تقنيم|رعاية|مراجعة|تحويل|إحالة|احالة)\s+"
    r"(?:(?:طبي|طبيّة|طبية)\s+)?(?:عاجل|عاجلة|سريع|سريعة|فوري|فورية)",
    re.I,
)


def _low_confidence_reasons(seg: ClassifiedSegment, combined: float, threshold: float
                            ) -> list[str]:
    """Say WHY an item needs review, not merely that it does.

    Four independent signals replace the single arbitrary 0.5 cut: a calibrated score
    below the fitted threshold, an unheard sentence, an ambiguous one, and a foreign
    one. A reviewer can act differently on each — re-listen, re-label, or ignore.
    """
    reasons: list[str] = []
    if combined < threshold:
        reasons.append(f"confidence {combined:.2f} < {threshold:.2f}")
    if seg.is_asr_suspect:
        reasons.append("audio segment flagged as a possible ASR hallucination")
    if seg.entropy is not None and seg.entropy > HIGH_ENTROPY:
        reasons.append(f"ambiguous between labels (entropy {seg.entropy:.2f})")
    if seg.ood_score is not None and seg.ood_score > HIGH_OOD:
        # OOD is an important review signal, but dialectal/ASR fragments can be far
        # from the synthetic/formal training distribution while still carrying
        # trustworthy structured clinical evidence (e.g. BP 160/110, "صداع").
        # Do not let OOD ALONE invalidate an otherwise strong item with a confirmed
        # clinical entity.  Extreme OOD without structured evidence still flags it.
        trusted_kinds = {"symptom", "condition", "vital", "lab", "test",
                         "medication", "allergy", "procedure", "clinical"}
        has_trusted_entity = any(
            isinstance(link, dict)
            and str(link.get("assertion", "present")) == "present"
            and str(link.get("kind", "")) in trusted_kinds
            for link in (seg.entity_links or [])
        )
        corroborated = (
            combined < threshold
            or seg.is_asr_suspect
            or (seg.entropy is not None and seg.entropy > HIGH_ENTROPY)
        )
        if corroborated or (seg.ood_score >= VERY_HIGH_OOD and not has_trusted_entity):
            reasons.append(f"unlike the training data (OOD {seg.ood_score:.2f})")
    return reasons




def _emergency_label_is_actionable(seg: ClassifiedSegment) -> bool:
    """Gate an `emergency` topic label through structured assertion evidence.

    The classifier may correctly identify that a sentence *talks about* danger signs
    even when the clinician says they are absent ("ما في نزيف ولا ألم شديد").
    When clinical entities were extracted and none is present, the label must not
    become an urgent flag.  If extraction found no clinical entity at all, retain the
    classifier signal for backwards-compatible safety.
    """
    if seg.label != "emergency":
        return False
    clinical_kinds = {"symptom", "condition", "vital", "lab", "clinical"}
    clinical_links = [
        link for link in (seg.entity_links or [])
        if isinstance(link, dict) and str(link.get("kind", "")) in clinical_kinds
    ]
    if not clinical_links:
        return True
    return any(
        str(link.get("assertion", "present")) == "present"
        for link in clinical_links
    )

def _to_item(
    job_id: str, seg: ClassifiedSegment, low_conf_threshold: float, *, use_canonical_text: bool = False
) -> ReportItem:
    combined = combined_confidence(seg.asr_confidence, seg.confidence)
    reasons = _low_confidence_reasons(seg, combined, low_conf_threshold)
    labels = seg.labels or [seg.label]
    decision_text = seg.effective_text if use_canonical_text else seg.text
    primary_section, also_in = soap_for_item(labels, decision_text, seg.entity_links)

    # Cross-cutting safety flag: severe-range blood pressure or explicit urgent
    # action language must not depend on the classifier choosing the `emergency`
    # class.  This is deterministic evidence, not a probabilistic diagnosis.
    severe_bp = any(
        isinstance(link, dict)
        and str(link.get("kind", "")) == "vital"
        and str(link.get("code", "")) == "bp"
        and str(link.get("assertion", "present")) == "present"
        and (
            float(link.get("value") or 0) >= 160
            or float(link.get("value2") or 0) >= 110
        )
        for link in (seg.entity_links or [])
    )
    urgent_language = bool(
        text_implies_plan(decision_text)
        and (
            _URGENT_ACTION_RE.search(decision_text)
            or any(token in decision_text for token in ("عاجل", "عاجلة", "فوراً", "فورا", "طوارئ"))
        )
    )

    return ReportItem(
        item_id=f"{job_id}-{seg.order_index:04d}",
        order_index=seg.order_index,
        text=decision_text,
        text_raw=seg.text,
        text_canonical=seg.text_canonical,
        canonicalization_status=seg.canonicalization_status,
        canonicalization_confidence=seg.canonicalization_confidence,
        canonicalization_model=seg.canonicalization_model,
        canonicalization_reasons=seg.canonicalization_reasons,
        text_rephrased=None,
        label=seg.label,
        labels=labels,
        label_ar=label_ar(seg.label),
        soap_section=primary_section,  # type: ignore[arg-type]
        also_in_sections=also_in,
        confidence=seg.confidence,
        combined_confidence=combined,
        entropy=seg.entropy,
        ood_score=seg.ood_score,
        review_priority=seg.review_priority,
        is_low_confidence=bool(reasons),
        low_confidence_reasons=reasons,
        is_urgent=(_emergency_label_is_actionable(seg) or severe_bp or urgent_language),
        start_sec=seg.start_sec,
        end_sec=seg.end_sec,
        timestamp_precision=seg.timestamp_precision,
        source_segment_index=seg.source_segment_index,
        entity_links=seg.entity_links,
        speaker=seg.speaker,
        speaker_confidence=seg.speaker_confidence,
        asr_confidence=seg.asr_confidence,
        is_asr_suspect=seg.is_asr_suspect,
    )


def _summarize(items: Sequence[ReportItem]) -> ReportSummary:
    if not items:
        return ReportSummary()
    per_label: dict[str, int] = {}
    for it in items:
        per_label[it.label] = per_label.get(it.label, 0) + 1
    avg = sum(it.confidence for it in items) / len(items)
    return ReportSummary(
        total_segments=len(items),
        avg_confidence=round(avg, 4),
        low_confidence_count=sum(1 for it in items if it.is_low_confidence),
        urgent_count=sum(1 for it in items if it.is_urgent),
        per_label_counts=per_label,
    )


def build_report(
    job_id: str,
    classified_segments: Sequence[ClassifiedSegment],
    *,
    audio_meta: Optional[AudioMeta] = None,
    pipeline_meta: Optional[PipelineMeta] = None,
    formatter: Optional[ClinicalSoapFormatter] = None,
    low_confidence_threshold: float = 0.5,
    created_at: Optional[datetime] = None,
    patient_info: Optional[dict] = None,
    use_canonical_text: bool = False,
) -> Report:
    """Group classified segments into a SOAP-structured Report.

    Items are placed in their SOAP section in chronological (order_index) order.
    A fact-constrained formatter derives doctor-facing section paragraphs after grouping.
    `patient_info` carries identity + prior obstetric context (empty when the upload
    was not linked to a patient).
    """
    ordered = sorted(classified_segments, key=lambda s: s.order_index)
    items = [
        _to_item(job_id, seg, low_confidence_threshold, use_canonical_text=use_canonical_text)
        for seg in ordered
    ]

    # Build the four SOAP sections in fixed order (always present, even if empty).
    soap: dict[str, ReportSection] = {}
    for key in SOAP_ORDER:
        section = ReportSection(
            soap_key=key,
            title_ar=SOAP_TITLES_AR[key],
            items=[it for it in items if it.soap_section == key],
        )
        soap[key] = section

    formatter = formatter or ClinicalSoapFormatter()
    soap_formatted = formatter.format(soap)

    meta = pipeline_meta or PipelineMeta()
    meta.rephrase_applied = False  # deprecated compatibility flag
    meta.soap_formatter_applied = bool(getattr(formatter, "applied", False))

    return Report(
        job_id=job_id,
        created_at=created_at or datetime.now(timezone.utc),
        audio=audio_meta or AudioMeta(),
        pipeline_meta=meta,
        patient_info=patient_info or {},
        soap=soap,
        soap_formatted=soap_formatted,
        summary=_summarize(items),
    )
