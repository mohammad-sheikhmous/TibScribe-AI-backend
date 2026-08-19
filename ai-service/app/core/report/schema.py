
from __future__ import annotations

from datetime import datetime
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


TimestampPrecision = Literal["segment", "word", "interpolated"]
SoapSection = Literal["subjective", "objective", "assessment", "plan"]

Speaker = Literal["doctor", "patient", "unknown"]


class Segment(BaseModel):
    """A sentence-like unit produced by the segmentation stage.

    `text_raw` preserves the original ASR text for audit and traceability.

    `text` is the working text used by the NLP pipeline. A medical
    canonicalization stage may safely modify `text` before it reaches AraBERT,
    while `text_raw` must remain unchanged.
    """

    order_index: int  
    text_raw: Optional[str] = None
    text: str
    start_sec: float
    end_sec: float
    timestamp_precision: TimestampPrecision = "segment"
    source_segment_index: int
    speaker: Optional[str] = None
    speaker_confidence: Optional[float] = None

    # Whisper avg_logprob-derived confidence/provenance value.
    asr_confidence: Optional[float] = None

    is_asr_suspect: bool = False


class ClassifiedSegment(Segment):
    """A Segment after the classifier has assigned label(s) + confidence.

    Because this class inherits from Segment, both `text_raw` and `text`
    automatically remain available after AraBERT classification.
    """

    label: str
    labels: list[str] = Field(default_factory=list)
    confidence: float
    entity_links: Optional[list[Any]] = None

    entropy: Optional[float] = None
    ood_score: Optional[float] = None
    review_priority: Optional[float] = None


class ReportItem(BaseModel):
    """One classified sentence as it appears in the final report.

    Self-describing (carries its own label + soap_section as typed fields) so any
    consumer — frontend or the future knowledge-graph layer — can regroup without
    the server pre-computing duplicate views.

    Text provenance is intentionally preserved:

    `text_raw`
        Original segmented Whisper/ASR text before medical canonicalization.

    `text`
        Canonical/working medical text that was actually consumed by AraBERT
        and downstream NLP stages.

    `text_rephrased`
        Optional future SOAP/report-level stylistic rewrite. This is separate
        from the pre-AraBERT canonicalization layer.
    """

    item_id: str
    order_index: int
    text_raw: Optional[str] = None
    text: str
    text_rephrased: Optional[str] = None
    label: str
    labels: list[str] = Field(default_factory=list)
    label_ar: str

    soap_section: SoapSection
    also_in_sections: list[str] = Field(default_factory=list)

    confidence: float

    combined_confidence: Optional[float] = None

    is_low_confidence: bool = False

    is_urgent: bool = False

    speaker: Optional[str] = None
    speaker_confidence: Optional[float] = None
    asr_confidence: Optional[float] = None
    is_asr_suspect: bool = False

    entropy: Optional[float] = None
    ood_score: Optional[float] = None
    review_priority: Optional[float] = None

    low_confidence_reasons: list[str] = Field(default_factory=list)
    start_sec: float
    end_sec: float
    timestamp_precision: TimestampPrecision = "segment"
    source_segment_index: int
    entity_links: Optional[list[Any]] = None


class ReportSection(BaseModel):
    soap_key: SoapSection
    title_ar: str
    items: list[ReportItem] = Field(default_factory=list)


class AudioMeta(BaseModel):
    filename: Optional[str] = None
    duration_sec: Optional[float] = None
    detected_language: Optional[str] = None
    whisper_model: Optional[str] = None


class PipelineMeta(BaseModel):
    classifier_max_len: Optional[int] = None
    arabert_model_name: Optional[str] = None

    # Existing optional post-SOAP rephrase stage.
    rephrase_applied: bool = False

    # --- ASR stage provenance (P3): what actually ran, so a report can be judged ---
    diarization_backend: str = "none"
    speakers_detected: int = 0
    asr_avg_logprob: Optional[float] = None
    asr_suspect_segments: int = 0


class ReportSummary(BaseModel):
    total_segments: int = 0
    avg_confidence: float = 0.0
    low_confidence_count: int = 0
    urgent_count: int = 0
    per_label_counts: dict[str, int] = Field(default_factory=dict)


class Report(BaseModel):
    schema_version: str = "1.0"
    job_id: str
    created_at: datetime

    audio: AudioMeta = Field(default_factory=AudioMeta)

    pipeline_meta: PipelineMeta = Field(default_factory=PipelineMeta)

    # Placeholder, empty in v1.
    patient_info: dict[str, Any] = Field(default_factory=dict)

    # Keys follow SOAP_ORDER.
    soap: dict[str, ReportSection] = Field(default_factory=dict)

    summary: ReportSummary = Field(default_factory=ReportSummary)