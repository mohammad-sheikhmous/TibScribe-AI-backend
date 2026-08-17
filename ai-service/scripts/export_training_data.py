"""Export doctor corrections as training data (IMPLEMENTATION.md P7-07).

This closes the loop the whole plan is built around: sentences the model got wrong, on
real Whisper transcripts, labelled by the clinician who noticed. That is exactly the
distribution the synthetic training set does not cover — and it arrives for free as a
by-product of reviewing reports.

Every row carries its `source`:

    human      a doctor corrected it            -> highest weight, ground truth
    confirmed  a doctor reviewed and kept it    -> also true, and NOT a correction
    pseudo     the model labelled it itself     -> lowest weight, provisional
    synthetic  the original data.jsonl          -> the bulk, but off-distribution

Keeping the source on every row is what makes a bad batch reversible: a pseudo-label
run that hurt the model can be dropped by source, without touching human data.

    python scripts/export_training_data.py --out data/gold/corrections.jsonl
    python scripts/export_training_data.py --dry-run
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.db import init_db, session_scope  # noqa: E402
from app.db.models import Correction, ReportItemRow  # noqa: E402
from app.db.repo import pending_corrections  # noqa: E402


def export(out_path: Path, *, dry_run: bool, include_confirmed: bool,
           mark_exported: bool) -> dict:
    stats: Counter = Counter()
    rows: list[dict] = []

    with session_scope() as session:
        corrections = pending_corrections(session)
        label_corrections = [c for c in corrections if c.field == "label"]

        for correction in label_corrections:
            item = session.get(ReportItemRow, correction.item_id)
            if item is None or not item.text.strip():
                stats["skipped_missing_item"] += 1
                continue
            rows.append({
                "text": item.text,
                "label": correction.new_value,
                "source": "human",
                "was_predicted": correction.old_value,
                "item_id": item.item_id,
                "job_id": item.job_id,
                "actor": correction.actor,
                "corrected_at": correction.created_at.isoformat()
                if correction.created_at else None,
            })
            stats["human"] += 1

        if include_confirmed:
            # Items a doctor read and left alone are ALSO labelled data — the model was
            # right, and confirming that is a training signal, not a non-event.
            confirmed = session.query(ReportItemRow).filter(
                ReportItemRow.is_corrected.is_(False),
                ReportItemRow.review_priority.isnot(None),
                ReportItemRow.review_priority > 0,
            ).limit(2000).all()
            for item in confirmed:
                rows.append({
                    "text": item.text, "label": item.label, "source": "confirmed",
                    "item_id": item.item_id, "job_id": item.job_id,
                })
                stats["confirmed"] += 1

        if mark_exported and not dry_run:
            now = datetime.now(timezone.utc)
            for correction in label_corrections:
                correction.exported_at = now

    if not dry_run and rows:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    return {"rows": len(rows), **stats}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export corrections as training data.")
    parser.add_argument("--out", default="data/gold/corrections.jsonl")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--include-confirmed", action="store_true",
                        help="also export reviewed-and-kept items as `confirmed`")
    parser.add_argument("--keep-pending", action="store_true",
                        help="do not mark corrections as exported (re-exportable)")
    args = parser.parse_args(argv)

    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # pragma: no cover
        pass

    init_db()
    stats = export(Path(args.out), dry_run=args.dry_run,
                   include_confirmed=args.include_confirmed,
                   mark_exported=not args.keep_pending)

    print(f"rows exported : {stats['rows']}")
    for source in ("human", "confirmed"):
        if stats.get(source):
            print(f"  {source:10s}: {stats[source]}")
    if stats.get("skipped_missing_item"):
        print(f"  skipped     : {stats['skipped_missing_item']} (item no longer exists)")

    if args.dry_run:
        print("\n--dry-run: nothing written, nothing marked as exported")
    elif stats["rows"]:
        print(f"\nwrote {args.out}")
        print("Next: python train_arabert.py --extra-data "
              f"{args.out} --extra-weight 3")
    else:
        print("\nNo new corrections to export yet — the loop starts once doctors review "
              "reports (PATCH /jobs/{id}/items/{item_id}).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
