"""End-to-end orchestration: audio -> transcript -> speakers -> segments -> Report.

Plain synchronous, blocking code (Whisper + AraBERT inference), run off the event loop
by the job runner. No web/HTTP coupling, so it can be exercised from a script.

Stage order and why:
  1. **ASR** with word timestamps.
  2. **Diarization** (optional) — who spoke when, as local clusters.
  3. **Role assignment** — which cluster is the doctor, using the voiceprint and the
     language, with abstention when the evidence is thin.
  4. **Segmentation** — sentence units that never cross a speaker change.
  5. **Classification** — one batched forward pass.
  6. **Report** — SOAP grouping, with ASR quality folded into the confidence.

Every stage after ASR degrades to a no-op rather than failing: no diarization backend
means unattributed speakers, not a broken job.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional

from .asr.diarization import DiarizationStage, NoOpDiarizer, cluster_texts
from .asr.quality import transcript_quality
from .asr.voiceprint import VoiceProfile, assign_roles
from .asr.whisper_service import WhisperTranscriber
from .nlp.classifier import MedicalSentenceClassifier
from .nlp.discourse import enrich_cross_segment_context
from .nlp.extraction import extract_for_item
from .report.builder import build_report
from .report.rephrase import NoOpRephraseStage, RephraseStage
from .report.schema import AudioMeta, ClassifiedSegment, PipelineMeta, Report
from .segmentation.segmenter import build_segments

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PipelineArtifacts:
    """All durable outputs produced by one pipeline pass.

    Keeping the raw Whisper result beside the Report lets the job runner persist the
    transcript without shared mutable state on ``MedicalScribePipeline`` (important
    because worker count may be >1).
    """

    report: Report
    transcript: dict[str, Any]
    transcript_quality: dict[str, Any]


class MedicalScribePipeline:
    def __init__(
        self,
        transcriber: WhisperTranscriber,
        classifier: MedicalSentenceClassifier,
        *,
        segment_max_chars: int = 140,
        word_pause_gap_sec: float = 0.5,
        low_confidence_threshold: float = 0.5,
        rephrase_stage: RephraseStage = NoOpRephraseStage(),
        diarizer: Optional[DiarizationStage] = None,
        voice_profile: Optional[VoiceProfile] = None,
        uncertainty_enabled: bool = True,
        mc_passes: int = 8,
        extract_entities: bool = True,
    ):
        self.transcriber = transcriber
        self.classifier = classifier
        self.segment_max_chars = segment_max_chars
        self.word_pause_gap_sec = word_pause_gap_sec
        self.low_confidence_threshold = low_confidence_threshold
        self.rephrase_stage = rephrase_stage
        self.diarizer = diarizer or NoOpDiarizer()
        self.voice_profile = voice_profile
        self.uncertainty_enabled = uncertainty_enabled
        self.mc_passes = mc_passes
        self.extract_entities = extract_entities

    def process(
        self,
        audio_path: str,
        job_id: str,
        filename: Optional[str] = None,
        patient_info: Optional[dict] = None,
    ) -> Report:
        """Backward-compatible report-only entrypoint."""
        return self.process_with_artifacts(
            audio_path, job_id, filename=filename, patient_info=patient_info
        ).report

    def process_with_artifacts(
        self,
        audio_path: str,
        job_id: str,
        filename: Optional[str] = None,
        patient_info: Optional[dict] = None,
    ) -> PipelineArtifacts:
        # 1. Speech-to-text
        result = self.transcriber.transcribe(audio_path)
        whisper_segments = result.get("segments", [])
        quality = transcript_quality(result)
        if quality["suspect_segments"]:
            logger.warning(
                "Job %s: %d/%d segments flagged as possible ASR hallucinations",
                job_id, quality["suspect_segments"], quality["total_segments"],
            )

        # 2. Who spoke when (optional backend)
        diarization = self._diarize(audio_path, job_id)

        # 3. Which cluster is the doctor
        roles = self._assign_roles(diarization, whisper_segments)

        # 4. Segmentation — never merging or splitting across a speaker change
        segments = build_segments(
            whisper_segments,
            max_chars=self.segment_max_chars,
            pause_gap_sec=self.word_pause_gap_sec,
            diarization=diarization,
        )
        for segment in segments:
            assignment = roles.get(segment.speaker) if segment.speaker else None
            if assignment is not None:
                segment.speaker = assignment.role
                segment.speaker_confidence = assignment.confidence

        # 5. Classification — one batched forward pass, order preserved 1:1
        classified = self._classify(segments)

        # 5b. Entity extraction — runs ONCE here and is stored, so the knowledge-based
        #     system consumes structured entities instead of re-parsing raw text.
        if self.extract_entities:
            for segment in classified:
                segment.entity_links = extract_for_item(
                    {"text": segment.text, "label": segment.label}
                )
            # Whisper boundaries are acoustic, so a clinical thought may span two or
            # three segments.  Add deterministic discourse facts/context without
            # changing AraBERT's original label or confidence.
            classified = enrich_cross_segment_context(classified)

        # 6. Structured SOAP report
        duration = max((s.end_sec for s in segments), default=None)
        audio_meta = AudioMeta(
            filename=filename,
            duration_sec=duration,
            detected_language=result.get("language"),
            whisper_model=self.transcriber.model_size,
        )
        pipeline_meta = PipelineMeta(
            classifier_max_len=self.classifier.max_len,
            arabert_model_name=self.classifier.model_name,
            diarization_backend=getattr(self.diarizer, "backend", "none"),
            speakers_detected=len(diarization.clusters) if diarization else 0,
            asr_avg_logprob=quality["avg_logprob"],
            asr_suspect_segments=quality["suspect_segments"],
        )
        report = build_report(
            job_id,
            classified,
            audio_meta=audio_meta,
            pipeline_meta=pipeline_meta,
            rephrase_stage=self.rephrase_stage,
            low_confidence_threshold=self.low_confidence_threshold,
            patient_info=patient_info,
        )
        return PipelineArtifacts(
            report=report, transcript=result, transcript_quality=quality
        )

    def _classify(self, segments) -> list[ClassifiedSegment]:
        """Label every segment, with uncertainty scores when enabled.

        The uncertainty path screens internally (MC-dropout only for the sentences that
        are not already confident), so enabling it costs little on a typical visit while
        giving the report a reason for every "needs review" flag.
        """
        texts = [seg.text for seg in segments]
        if not self.uncertainty_enabled or not hasattr(
            self.classifier, "predict_with_uncertainty"
        ):
            predictions = self.classifier.predict_batch(texts)
            return [
                ClassifiedSegment(**seg.model_dump(), label=label,
                                  confidence=float(confidence))
                for seg, (label, confidence) in zip(segments, predictions)
            ]

        scored = self.classifier.predict_with_uncertainty(texts, mc_passes=self.mc_passes)
        return [
            ClassifiedSegment(
                **seg.model_dump(),
                label=row["label"],
                confidence=float(row["confidence"]),
                entropy=row["uncertainty"].entropy,
                ood_score=row["uncertainty"].ood,
                review_priority=row["uncertainty"].review_priority,
            )
            for seg, row in zip(segments, scored)
        ]

    # --- stages that must never break the job -------------------------------------
    def _diarize(self, audio_path: str, job_id: str):
        try:
            return self.diarizer.diarize(audio_path)
        except Exception:  # noqa: BLE001 - a diarization failure is not a job failure
            logger.exception("Job %s: diarization failed; continuing unattributed", job_id)
            return NoOpDiarizer().diarize(audio_path)

    def _assign_roles(self, diarization, whisper_segments):
        if not diarization or not diarization.clusters:
            return {}
        texts = cluster_texts(diarization, whisper_segments)
        embeddings = {
            turn.cluster: turn.embedding
            for turn in diarization.turns
            if turn.embedding is not None
        }
        try:
            return assign_roles(embeddings, texts, self.voice_profile) if embeddings else \
                assign_roles({c: None for c in diarization.clusters}, texts, None)
        except Exception:  # noqa: BLE001
            logger.exception("role assignment failed; speakers stay unattributed")
            return {}
