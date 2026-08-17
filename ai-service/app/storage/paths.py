"""Filesystem path helpers for per-job audio uploads and result JSON."""
from __future__ import annotations

from pathlib import Path

from ..config import get_settings


def upload_path(job_id: str, ext: str) -> Path:
    s = get_settings()
    s.upload_dir_path.mkdir(parents=True, exist_ok=True)
    if ext and not ext.startswith("."):
        ext = "." + ext
    return s.upload_dir_path / f"{job_id}{ext}"


def result_path(job_id: str) -> Path:
    s = get_settings()
    s.result_dir_path.mkdir(parents=True, exist_ok=True)
    return s.result_dir_path / f"{job_id}.json"
