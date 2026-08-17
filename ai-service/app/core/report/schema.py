"""Pydantic data models for the pipeline and the final structured report.

These models are the contract between pipeline stages AND the FastAPI response
body AND the persisted data/results/{job_id}.json. They are deliberately free of
any HTTP/FastAPI coupling so a future app/core/kg/ layer can import and consume
`Report` directly.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

TimestampPrecision = Literal["segment", "word", "interpolated"]
SoapSection = Literal["subjective", "objective", "assessment", "plan"]


Speaker = Literal["doctor", "patient", "unknown"]


class Segment(BaseModel):
    """A sentence-like unit produced by the segmentation stage."""

    order_index: int  # global chronological order across the whole transcript
    text: str
    start_sec: float
    end_sec: float
    timestamp_precision: TimestampPrecision = "segment"
    source_segment_index: int  # id of the parent Whisper segment (for debugging/traceability)

    # --- ASR quality + attribution (P3) ---
    # None = the stage did not run (no diarization backend), NOT "no speaker".
    speaker: Optional[str] = None
    speaker_confidence: Optional[float] = None
    asr_confidence: Optional[float] = None  # from Whisper's avg_logprob
    # Whisper hallucinates fluent text on silence; suspect items are FLAGGED, never
    # dropped, and are excluded from firing clinical rules.
    is_asr_suspect: bool = False


class ClassifiedSegment(Segment):
    """A Segment after the classifier has assigned a label + confidence."""

    label: str
    labels: list[str] = Field(default_factory=list)  # multi-label head (P4-09)
    confidence: float
    entity_links: Optional[list[Any]] = None  # filled by the shared extractor (P5-14)
    entropy: Optional[float] = None
    ood_score: Optional[float] = None
    review_priority: Optional[float] = None


class ReportItem(BaseModel):
    """One classified sentence as it appears in the final report.

    Self-describing (carries its own label + soap_section as typed fields) so any
    consumer — frontend or the future knowledge-graph layer — can regroup without
    the server pre-computing duplicate views.
    """

    item_id: str  # stable, addressable: f"{job_id}-{order_index:04d}"
    order_index: int
    text: str  # raw extractive ASR text — the traceable baseline
    text_rephrased: Optional[str] = None  # populated later by an LLM rephrase stage

    label: str  # fine-grained class id (20 active labels); the primary one
    # Every label above threshold once the multi-label head lands (P4-09). Until then
    # this mirrors `label`, so consumers can be written against it today.
    labels: list[str] = Field(default_factory=list)
    label_ar: str  # Arabic display name
    soap_section: SoapSection
    # Other SOAP sections this item also belongs to — visible instead of discarded.
    also_in_sections: list[str] = Field(default_factory=list)
    confidence: float  # classifier confidence
    combined_confidence: Optional[float] = None  # classifier x ASR confidence
    is_low_confidence: bool = False
    is_urgent: bool = False  # cross-cutting flag (label == "emergency")

    # --- attribution + input quality (P3) ---
    speaker: Optional[str] = None
    speaker_confidence: Optional[float] = None
    asr_confidence: Optional[float] = None
    is_asr_suspect: bool = False

    # --- uncertainty (P4): why an item was flagged, not just that it was ---
    entropy: Optional[float] = None        # ambiguity between known classes
    ood_score: Optional[float] = None      # unlike anything in the training data
    review_priority: Optional[float] = None  # ranks what a reviewer should read first
    low_confidence_reasons: list[str] = Field(default_factory=list)

    # traceability back to the source audio
    start_sec: float
    end_sec: float
    timestamp_precision: TimestampPrecision = "segment"
    source_segment_index: int

    # reserved slot for the future KG entity-linking layer (None in v1)
    entity_links: Optional[list[Any]] = None


class ReportSection(BaseModel):
    soap_key: SoapSection
    title_ar: str
    items: list[ReportItem] = Field(default_factory=list)  # chronological order


class AudioMeta(BaseModel):
    filename: Optional[str] = None
    duration_sec: Optional[float] = None
    detected_language: Optional[str] = None
    whisper_model: Optional[str] = None


class PipelineMeta(BaseModel):
    classifier_max_len: Optional[int] = None
    arabert_model_name: Optional[str] = None
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
    patient_info: dict[str, Any] = Field(default_factory=dict)  # placeholder, empty in v1
    soap: dict[str, ReportSection] = Field(default_factory=dict)  # keys in SOAP_ORDER
    summary: ReportSummary = Field(default_factory=ReportSummary)
