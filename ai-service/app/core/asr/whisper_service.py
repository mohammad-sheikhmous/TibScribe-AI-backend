"""Whisper speech-to-text wrapper.

Loads the model ONCE (at FastAPI startup) and exposes .transcribe(path). Returns
the raw Whisper result — segments (with start/end and, when requested, per-word
timestamps) — which the segmentation stage consumes. word_timestamps=True is what
unlocks accurate re-splitting of over-long segments.
"""
from __future__ import annotations

import logging
import shutil
from typing import Any

logger = logging.getLogger(__name__)


class FFmpegMissingError(RuntimeError):
    """Raised when the ffmpeg BINARY is absent (not the `ffmpeg-python` package)."""


def ffmpeg_path() -> str | None:
    """Where the ffmpeg executable is, or None.

    Whisper shells out to ffmpeg to decode audio; without it every transcription fails
    deep inside a subprocess call with an unhelpful message. Checking here turns a
    confusing runtime failure into a named, actionable one — and lets `/ready` report
    the problem before a single recording is accepted.
    """
    return shutil.which("ffmpeg")


def require_ffmpeg() -> str:
    path = ffmpeg_path()
    if path is None:
        raise FFmpegMissingError(
            "ffmpeg is not on PATH. Whisper cannot decode audio without it.\n"
            "  Windows: winget install --id Gyan.FFmpeg --scope user\n"
            "  Linux  : apt install ffmpeg   ·   macOS: brew install ffmpeg\n"
            "Note this is the SYSTEM binary, not the `ffmpeg-python` pip package."
        )
    return path


class WhisperTranscriber:
    def __init__(self, model_size: str = "medium", language: str = "ar"):
        # Validate the external decoder first. This gives callers the precise
        # FFmpegMissingError even on machines where the Python `whisper` package is
        # also absent, and matches the readiness contract.
        binary = require_ffmpeg()
        logger.info("ffmpeg found at %s", binary)

        import whisper  # imported lazily so the module is importable without whisper installed

        self.model_size = model_size
        self.language = language
        logger.info("Loading Whisper model: %s", model_size)
        self.model = whisper.load_model(model_size)

    def transcribe(self, audio_path: str) -> dict[str, Any]:
        """Transcribe an audio file. Returns Whisper's raw result dict.

        Keys of interest: 'text', 'language', and 'segments' (each with 'start', 'end',
        'text', 'words' when word_timestamps=True, plus the quality signals
        'avg_logprob', 'no_speech_prob' and 'compression_ratio' that
        `app/core/asr/quality.py` reads).

        The result is returned whole and stored whole (principle P1) — the fields this
        version ignores are exactly the ones a later version will need.
        """
        result = self.model.transcribe(
            audio_path,
            language=self.language,
            word_timestamps=True,
            verbose=False,
        )
        return result
