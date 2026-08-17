"""Operational endpoints (IMPLEMENTATION.md P1-12).

`/admin/storage` exists because the audio archive is never pruned: capacity has to be
*observed* instead of *reclaimed*. It reports size, age and checksum coverage, and
warns past a threshold — it offers no delete action, which is the point.

`/admin/storage/verify` re-hashes archived objects. Bit-rot is the one remaining way a
"never delete" policy can still lose a recording, so it is checked deliberately.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone

from fastapi import APIRouter, Request

from ..config import get_settings
from ..db import repo, session_scope
from ..db.models import AudioFile
from .schemas import StorageStats

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/admin", tags=["admin"])

GB = 1024 ** 3


@router.get("/storage", response_model=StorageStats)
async def storage_stats() -> StorageStats:
    settings = get_settings()
    with session_scope() as session:
        stats = repo.audio_stats(session)
    total_gb = stats["total_bytes"] / GB
    over = total_gb >= settings.audio_warn_gb
    if over:
        logger.warning(
            "audio archive at %.1f GB (threshold %.1f GB) — provision capacity; "
            "retention policy forbids deletion",
            total_gb, settings.audio_warn_gb,
        )
    return StorageStats(
        objects=stats["objects"],
        total_bytes=stats["total_bytes"],
        total_gb=round(total_gb, 3),
        oldest_upload=stats["oldest_upload"],
        never_verified=stats["never_verified"],
        warn_threshold_gb=settings.audio_warn_gb,
        over_threshold=over,
    )


@router.post("/storage/verify")
async def verify_storage(request: Request, limit: int = 50) -> dict:
    """Re-hash up to `limit` least-recently-verified objects and report mismatches."""
    store = request.app.state.audio_store
    checked, ok, failures = 0, 0, []

    with session_scope() as session:
        rows = (
            session.query(AudioFile)
            .order_by(AudioFile.checksum_verified_at.is_(None).desc(),
                      AudioFile.checksum_verified_at.asc())
            .limit(limit)
            .all()
        )
        for row in rows:
            checked += 1
            if store.verify(row.storage_key, row.sha256):
                ok += 1
                row.checksum_verified_at = datetime.now(timezone.utc)
            else:
                failures.append({"audio_id": row.id, "storage_key": row.storage_key})
                logger.error("INTEGRITY FAILURE for archived audio %s (%s)",
                             row.id, row.storage_key)

    return {"checked": checked, "ok": ok, "failed": len(failures), "failures": failures}


@router.get("/metrics")
async def metrics() -> dict:
    """Field metrics, measured on real use rather than on a held-out split (P7-03).

    `confusions` is the honest confusion matrix: which predicted label doctors actually
    change, and to what. A synthetic test set cannot show this, because it is drawn from
    the same distribution the model over-fits.

    The per-rule precision half of this endpoint arrives with P6-12, once suggestions
    are served and can be given feedback.
    """
    with session_scope() as session:
        corrections = repo.correction_stats(session)
        queue_depth = len(repo.review_queue(session, limit=500))
        _, jobs_total = repo.list_jobs(session, limit=1)

    return {
        "jobs_total": jobs_total,
        "corrections": corrections,
        "review_queue_depth": queue_depth,
        "rules": {
            "note": "per-rule precision requires suggestion feedback (P6-12 → P7-02)",
            "available": False,
        },
    }


@router.get("/models")
async def list_models() -> dict:
    """Which model version is live (P4-11 fills this in; the table exists now)."""
    from ..db.models import ModelRegistry

    with session_scope() as session:
        rows = session.query(ModelRegistry).order_by(ModelRegistry.created_at.desc()).all()
        return {
            "models": [
                {
                    "kind": row.kind, "version": row.version, "path": row.path,
                    "is_active": row.is_active, "preprocessing": row.preprocessing,
                    "max_len": row.max_len, "metrics": row.metrics,
                    "trained_at": row.trained_at,
                }
                for row in rows
            ]
        }
