"""Audio playback (IMPLEMENTATION.md P1-10).

Every report sentence carries `start_sec`/`end_sec`. Those timestamps are only useful
if the doctor can actually jump to that moment, which needs HTTP **Range** support —
without it a browser must download the whole file before it can seek, and the
traceability the pipeline captures stays theoretical.

Read-only by construction: this module opens archived objects and never writes or
removes them.
"""
from __future__ import annotations

import re
import shutil
import subprocess
from typing import Optional

from pathlib import Path
from urllib.parse import quote

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import Response, StreamingResponse
from sqlalchemy import select

from ..db import repo, session_scope
from ..db.models import ReportItemRow

router = APIRouter(prefix="/jobs", tags=["audio"])

_RANGE_RE = re.compile(r"bytes=(\d*)-(\d*)")

MIME_BY_EXT = {
    ".mp3": "audio/mpeg", ".wav": "audio/wav", ".m4a": "audio/mp4", ".mp4": "audio/mp4",
    ".ogg": "audio/ogg", ".oga": "audio/ogg", ".opus": "audio/opus", ".flac": "audio/flac",
    ".webm": "audio/webm", ".aac": "audio/aac", ".amr": "audio/amr", ".wma": "audio/x-ms-wma",
}

def content_disposition(filename: Optional[str], job_id: str, suffix: str) -> str:
    """Build a Content-Disposition header safe for Unicode filenames."""

    original = Path(filename).name if filename else f"{job_id}{suffix}"

    # Fallback ASCII filename for older clients / HTTP header compatibility.
    fallback_suffix = Path(original).suffix or suffix
    fallback = f"{job_id}{fallback_suffix}"

    # RFC 6266 / RFC 5987 UTF-8 filename.
    encoded = quote(original, safe="")

    return f'inline; filename="{fallback}"; filename*=UTF-8\'\'{encoded}'

def parse_range(header: Optional[str], size: int) -> Optional[tuple[int, int]]:
    """Parse a single-range `bytes=` header into inclusive (start, end).

    Returns None for a missing/unsupported header (caller then serves the whole file).
    Raises ValueError when the range is syntactically fine but unsatisfiable, which
    must become a 416 rather than a silent full-file response.
    """
    if not header:
        return None
    match = _RANGE_RE.fullmatch(header.strip())
    if not match:
        return None
    raw_start, raw_end = match.group(1), match.group(2)
    if not raw_start and not raw_end:
        return None

    if not raw_start:  # suffix form: last N bytes
        length = int(raw_end)
        if length <= 0:
            raise ValueError("empty suffix range")
        start, end = max(0, size - length), size - 1
    else:
        start = int(raw_start)
        end = int(raw_end) if raw_end else size - 1
        end = min(end, size - 1)

    if start >= size or start > end:
        raise ValueError("range not satisfiable")
    return start, end


@router.get("/{job_id}/audio")
async def get_job_audio(job_id: str, request: Request):
    """Stream the original recording, honouring Range so the player can seek."""
    store = request.app.state.audio_store
    with session_scope() as session:
        audio = repo.audio_for_job(session, job_id)
        if audio is None:
            raise HTTPException(404, "No audio for this job.")
        storage_key, filename, mime = audio.storage_key, audio.original_filename, audio.mime

    if not store.exists(storage_key):
        # The archive is append-only, so this means a storage/mount problem, not a
        # deleted recording — say so plainly instead of a bare 404.
        raise HTTPException(
            410, "Archived audio object is not reachable — check the audio store mount."
        )

    size = store.size_of(storage_key)
    suffix = storage_key[storage_key.rfind("."):] if "." in storage_key else ""
    media_type = mime or MIME_BY_EXT.get(suffix.lower(), "application/octet-stream")
    headers = {
        "Accept-Ranges": "bytes",
        "Content-Disposition": content_disposition(filename, job_id, suffix),
    }

    try:
        span = parse_range(request.headers.get("range"), size)
    except ValueError:
        raise HTTPException(416, "Requested range not satisfiable", headers={"Content-Range": f"bytes */{size}"})

    if span is None:
        headers["Content-Length"] = str(size)
        return StreamingResponse(
            store.read_range(storage_key, 0, size - 1),
            media_type=media_type, headers=headers,
        )

    start, end = span
    headers["Content-Range"] = f"bytes {start}-{end}/{size}"
    headers["Content-Length"] = str(end - start + 1)
    return StreamingResponse(
        store.read_range(storage_key, start, end),
        status_code=206, media_type=media_type, headers=headers,
    )


# --- per-sentence clips (P2-05) ------------------------------------------------------

CLIP_PAD_SEC = 0.35  # a little air on both sides; a hard cut clips the first phoneme


def build_ffmpeg_cmd(source: str, start: float, duration: float) -> list[str]:
    """ffmpeg arguments for a lossless-ish clip.

    `-ss` before `-i` seeks by index (fast); `-c copy` avoids re-encoding, so the clip
    is a byte-level excerpt of the original rather than a lossy re-rendering.
    Pure function so it can be tested without ffmpeg installed.
    """
    return [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-ss", f"{max(0.0, start):.3f}",
        "-t", f"{max(0.05, duration):.3f}",
        "-i", source,
        "-c", "copy",
        "-f", "matroska", "pipe:1",
    ]


@router.get("/{job_id}/items/{item_id}/audio")
async def get_item_audio(job_id: str, item_id: str, request: Request, pad: float = CLIP_PAD_SEC):
    """Return just the seconds of audio behind one report sentence.

    This is the traceability loop closing: a doctor reviewing a flagged sentence hears
    exactly what was said, without scrubbing through the whole visit.
    """
    store = request.app.state.audio_store
    with session_scope() as session:
        item = session.scalar(
            select(ReportItemRow).where(
                ReportItemRow.item_id == item_id, ReportItemRow.job_id == job_id
            )
        )
        if item is None:
            raise HTTPException(404, "Unknown report item.")
        audio = repo.audio_for_job(session, job_id)
        if audio is None:
            raise HTTPException(404, "No audio for this job.")
        start, end, storage_key = item.start_sec, item.end_sec, audio.storage_key

    if not store.exists(storage_key):
        raise HTTPException(410, "Archived audio object is not reachable.")

    clip_start = max(0.0, start - pad)
    duration = max(0.05, (end - start) + 2 * pad)
    headers = {
        "X-Clip-Start-Sec": f"{clip_start:.3f}",
        "X-Clip-Duration-Sec": f"{duration:.3f}",
        "Cache-Control": "private, max-age=3600",
    }

    if shutil.which("ffmpeg") is None:
        # Say exactly what is missing and give the zero-dependency alternative, rather
        # than failing with a FileNotFoundError from deep inside subprocess.
        raise HTTPException(
            503,
            {
                "error": "ffmpeg is not installed on the server, so clips cannot be cut.",
                "alternative": f"/jobs/{job_id}/audio#t={clip_start:.2f},{clip_start + duration:.2f}",
                "hint": "Install ffmpeg and put it on PATH (see CLAUDE.md).",
            },
            headers=headers,
        )

    source = str(store.path_for(storage_key))
    try:
        completed = subprocess.run(
            build_ffmpeg_cmd(source, clip_start, duration),
            capture_output=True, timeout=30, check=True,
        )
    except subprocess.TimeoutExpired:
        raise HTTPException(504, "Clip extraction timed out.")
    except subprocess.CalledProcessError as exc:
        raise HTTPException(500, f"Clip extraction failed: {exc.stderr[:200].decode(errors='replace')}")

    headers["Content-Length"] = str(len(completed.stdout))
    return Response(content=completed.stdout, media_type="video/x-matroska", headers=headers)
