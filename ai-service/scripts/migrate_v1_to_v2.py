"""Import v1 data into the v2 database + audio archive (IMPLEMENTATION.md P1-11).

v1 wrote reports to `data/results/*.json` and left uploads in `data/uploads/`, with no
index tying them together. This walks both, archives every recording under its content
hash, and recreates the jobs/reports rows so old work becomes visible through the v2
API instead of being files nobody can find.

**Nothing is deleted or modified in place.** The original files stay exactly where they
are; this only reads them. Re-running is safe — existing jobs are skipped unless
`--overwrite` is given.

    python scripts/migrate_v1_to_v2.py --dry-run
    python scripts/migrate_v1_to_v2.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import get_settings  # noqa: E402
from app.core.report.schema import Report  # noqa: E402
from app.db import init_db, repo, session_scope  # noqa: E402
from app.db.models import Job  # noqa: E402
from app.storage.audio_store import LocalAudioStore  # noqa: E402


def find_audio(uploads: Path, job_id: str) -> Path | None:
    """v1 named uploads `{job_id}{ext}` — that convention is the only link we have."""
    matches = sorted(uploads.glob(f"{job_id}.*")) if uploads.exists() else []
    return matches[0] if matches else None


def migrate(results_dir: Path, uploads_dir: Path, *, dry_run: bool, overwrite: bool) -> dict:
    settings = get_settings()
    store = LocalAudioStore(settings.audio_dir_path)
    stats = {"reports": 0, "imported": 0, "skipped": 0, "audio_archived": 0,
             "audio_missing": 0, "invalid": 0}

    files = sorted(results_dir.glob("*.json")) if results_dir.exists() else []
    for path in files:
        stats["reports"] += 1
        try:
            report = Report.model_validate(json.loads(path.read_text(encoding="utf-8")))
        except Exception as exc:  # noqa: BLE001 - a malformed file must not abort the run
            print(f"  ! {path.name}: unreadable ({type(exc).__name__}) — skipped")
            stats["invalid"] += 1
            continue

        job_id = report.job_id
        with session_scope() as session:
            if session.get(Job, job_id) is not None and not overwrite:
                stats["skipped"] += 1
                continue

        audio_src = find_audio(uploads_dir, job_id)
        stored = None
        if audio_src is not None:
            if not dry_run:
                stored = store.put(audio_src, audio_src.suffix)
            stats["audio_archived"] += 1
        else:
            stats["audio_missing"] += 1

        print(f"  + {job_id}  items={report.summary.total_segments:3d}  "
              f"audio={'yes' if audio_src else 'MISSING'}")
        if dry_run:
            stats["imported"] += 1
            continue

        with session_scope() as session:
            audio_id = None
            if stored is not None:
                audio = repo.get_or_create_audio(
                    session,
                    sha256=stored.sha256,
                    storage_key=stored.storage_key,
                    original_filename=report.audio.filename or audio_src.name,
                    mime=None,
                    size_bytes=stored.size_bytes,
                )
                audio.duration_sec = report.audio.duration_sec
                audio_id = audio.id

            job = session.get(Job, job_id)
            if job is None:
                job = repo.create_job(session, job_id=job_id, audio_id=audio_id,
                                      status="complete")
            else:
                job.audio_id = audio_id or job.audio_id
                job.status = "complete"
            job.asr_model = report.audio.whisper_model
            job.classifier_version = report.pipeline_meta.arabert_model_name
            job.created_at = report.created_at
            repo.save_report(session, report)
        stats["imported"] += 1

    return stats


def main(argv: list[str] | None = None) -> int:
    settings = get_settings()
    p = argparse.ArgumentParser(description="Import v1 reports/uploads into v2 storage.")
    p.add_argument("--results", default=str(settings.result_dir_path))
    p.add_argument("--uploads", default=str(settings.upload_dir_path))
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--overwrite", action="store_true",
                   help="re-import jobs that already exist in the database")
    args = p.parse_args(argv)

    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # pragma: no cover
        pass

    # The schema is created even for a dry run: deciding what to import requires
    # asking the database what is already there. Creating empty tables writes no data.
    init_db()

    print(f"results: {args.results}")
    print(f"uploads: {args.uploads}")
    print(f"archive: {settings.audio_dir_path}{'  (dry run)' if args.dry_run else ''}\n")

    stats = migrate(Path(args.results), Path(args.uploads),
                    dry_run=args.dry_run, overwrite=args.overwrite)

    print("\n--- summary ---")
    for key, value in stats.items():
        print(f"  {key:16s} {value}")
    if stats["audio_missing"]:
        print(f"\nNOTE: {stats['audio_missing']} report(s) had no matching upload. Their "
              f"reports are imported; the recordings simply were not kept in v1.")
    print("\nOriginal files were not modified or removed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
