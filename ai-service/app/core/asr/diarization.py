"""Speaker diarization (IMPLEMENTATION.md P3-05 / P3-06).

Without it, "هل عندك صداع؟" and "عندي صداع" are the same sentence to the classifier —
a question the doctor asked becomes a symptom the patient reported. This is the single
largest clinical-accuracy gap in the ASR stage.

Shape of the seam: a Protocol plus a NoOp default, so the pipeline is written once and
the real implementation is an optional dependency. `pyannote.audio` needs a Hugging Face
token and an accepted licence, so a deployment without it degrades to unlabelled
speakers rather than failing.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional, Protocol, Sequence

logger = logging.getLogger(__name__)


@dataclass
class SpeakerTurn:
    """One continuous stretch of speech attributed to one cluster."""

    start_sec: float
    end_sec: float
    cluster: str                      # local label ("SPEAKER_00"), not a role
    embedding: Optional[Any] = None   # np.ndarray when the backend provides one

    @property
    def duration(self) -> float:
        return max(0.0, self.end_sec - self.start_sec)

    def overlap(self, start: float, end: float) -> float:
        return max(0.0, min(self.end_sec, end) - max(self.start_sec, start))


@dataclass
class DiarizationResult:
    turns: list[SpeakerTurn] = field(default_factory=list)
    backend: str = "none"

    @property
    def clusters(self) -> list[str]:
        seen: list[str] = []
        for turn in self.turns:
            if turn.cluster not in seen:
                seen.append(turn.cluster)
        return seen

    def cluster_for(self, start: float, end: float) -> Optional[str]:
        """Which cluster owns a time span — by greatest overlap, not by midpoint.

        Midpoint lookup mislabels a segment that straddles a speaker change; overlap
        picks whoever actually said most of it.
        """
        if not self.turns:
            return None
        best = max(self.turns, key=lambda turn: turn.overlap(start, end))
        return best.cluster if best.overlap(start, end) > 0 else None


class DiarizationStage(Protocol):
    def diarize(self, audio_path: str) -> DiarizationResult: ...


class NoOpDiarizer:
    """Default: no speaker separation. Every segment stays unattributed."""

    backend = "none"

    def diarize(self, audio_path: str) -> DiarizationResult:  # noqa: ARG002
        return DiarizationResult(turns=[], backend="none")


class PyannoteDiarizer:
    """pyannote.audio 3.x wrapper.

    The model is gated: it needs `HF_TOKEN` in the environment and the licence accepted
    on the model page. Loading is lazy so importing this module never requires the
    dependency, and construction failures are reported clearly instead of at first use.
    """

    backend = "pyannote"

    def __init__(self, model_name: str = "pyannote/speaker-diarization-3.1",
                 hf_token: Optional[str] = None):
        self.model_name = model_name
        self._pipeline = None
        self._hf_token = hf_token
        self._embedder = None

    def _load(self):
        if self._pipeline is None:
            from pyannote.audio import Pipeline  # imported lazily on purpose

            self._pipeline = Pipeline.from_pretrained(
                self.model_name, use_auth_token=self._hf_token
            )
            logger.info("Loaded diarization pipeline: %s", self.model_name)
        return self._pipeline

    def diarize(self, audio_path: str) -> DiarizationResult:
        pipeline = self._load()
        annotation = pipeline(audio_path)
        turns = [
            SpeakerTurn(start_sec=float(segment.start), end_sec=float(segment.end),
                        cluster=str(label))
            for segment, _, label in annotation.itertracks(yield_label=True)
        ]
        return DiarizationResult(turns=turns, backend=self.backend)


def build_diarizer(enabled: bool, hf_token: Optional[str] = None) -> DiarizationStage:
    """Pick a backend, degrading to NoOp with an explanation rather than crashing."""
    if not enabled:
        return NoOpDiarizer()
    try:
        import pyannote.audio  # noqa: F401
    except Exception:
        logger.warning(
            "Diarization requested but pyannote.audio is unavailable — continuing "
            "without speaker separation. Install it and set HF_TOKEN to enable."
        )
        return NoOpDiarizer()
    if not hf_token:
        logger.warning(
            "Diarization requested but HF_TOKEN is not set; the pyannote model is "
            "gated. Continuing without speaker separation."
        )
        return NoOpDiarizer()
    return PyannoteDiarizer(hf_token=hf_token)


def cluster_texts(result: DiarizationResult, segments: Sequence[dict]) -> dict[str, list[str]]:
    """Group transcript text by speaker cluster — the input to the linguistic scorer."""
    grouped: dict[str, list[str]] = {cluster: [] for cluster in result.clusters}
    for segment in segments:
        cluster = result.cluster_for(
            float(segment.get("start", 0.0)), float(segment.get("end", 0.0))
        )
        if cluster is not None:
            grouped.setdefault(cluster, []).append(str(segment.get("text") or ""))
    return grouped
