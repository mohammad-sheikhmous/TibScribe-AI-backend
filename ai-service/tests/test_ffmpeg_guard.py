"""The ffmpeg dependency is checked explicitly (layer 1 hardening).

Whisper shells out to the ffmpeg BINARY — not the `ffmpeg-python` pip package. When it
is absent, every transcription used to fail deep inside a subprocess with an unhelpful
message, and the job simply ended up `failed`. These tests pin the replacement: a named
error with installation instructions, and a `/ready` that refuses to claim readiness.
"""
import shutil

import pytest

from app.core.asr.whisper_service import (
    FFmpegMissingError,
    ffmpeg_path,
    require_ffmpeg,
)


def test_ffmpeg_is_installed_on_this_machine():
    """Layer 1 cannot work without it — so this is a real environment assertion."""
    assert ffmpeg_path() is not None, (
        "ffmpeg is not on PATH. Install it: winget install --id Gyan.FFmpeg --scope user"
    )


def test_require_ffmpeg_returns_the_path():
    assert require_ffmpeg().lower().endswith(("ffmpeg", "ffmpeg.exe"))


def test_missing_ffmpeg_raises_a_named_actionable_error(monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda _name: None)
    with pytest.raises(FFmpegMissingError) as exc:
        require_ffmpeg()

    message = str(exc.value)
    assert "not on PATH" in message
    assert "winget" in message and "apt install" in message   # how to fix it
    # the classic confusion this note exists to prevent
    assert "ffmpeg-python" in message


def test_transcriber_refuses_to_construct_without_ffmpeg(monkeypatch):
    """Fail at startup with an instruction, not per-job inside a subprocess."""
    monkeypatch.setattr(shutil, "which", lambda _name: None)
    from app.core.asr.whisper_service import WhisperTranscriber

    with pytest.raises(FFmpegMissingError):
        WhisperTranscriber(model_size="tiny")
