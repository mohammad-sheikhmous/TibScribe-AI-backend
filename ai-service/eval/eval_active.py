"""Does the review queue actually save the doctor's time? (IMPLEMENTATION.md P7-05)

The claim behind active learning is specific and testable: **reading the few
highest-uncertainty sentences catches most of the errors**. If that is false, the queue
is just a different order and the doctor may as well read everything.

Measured against corrections already made: for each report, how many of its actual
errors (= corrected items) sit in the top-K of the uncertainty ranking?

    coverage@K = corrected items in top-K / all corrected items

The P7-05 gate is coverage@3 ≥ 0.60. A random-order baseline is printed alongside,
because a high number means nothing without knowing what chance would give.

    python eval/eval_active.py --k 3
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.db import init_db, session_scope  # noqa: E402
from app.db.models import Correction, ReportItemRow  # noqa: E402

TARGET_COVERAGE = 0.60


def gather() -> dict[str, list[dict]]:
    """Report items grouped by job, marked with whether they were corrected."""
    with session_scope() as session:
        corrected_ids = {
            item_id for (item_id,) in session.query(Correction.item_id).distinct()
        }
        rows = session.query(ReportItemRow).all()
        by_job: dict[str, list[dict]] = defaultdict(list)
        for row in rows:
            by_job[row.job_id].append({
                "item_id": row.item_id,
                "order_index": row.order_index,
                "priority": row.review_priority
                if row.review_priority is not None else (1.0 - (row.confidence or 0.0)),
                "is_error": row.item_id in corrected_ids,
            })
        return by_job


def coverage_at_k(items: list[dict], k: int, *, order: str, seed: int = 0) -> float | None:
    errors = [item for item in items if item["is_error"]]
    if not errors:
        return None  # a report with no known errors tells us nothing either way

    if order == "priority":
        ranked = sorted(items, key=lambda item: -item["priority"])
    elif order == "document":
        ranked = sorted(items, key=lambda item: item["order_index"])
    else:
        ranked = list(items)
        random.Random(seed).shuffle(ranked)

    top = {item["item_id"] for item in ranked[:k]}
    return sum(1 for error in errors if error["item_id"] in top) / len(errors)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Measure review-queue effectiveness.")
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # pragma: no cover
        pass

    init_db()
    by_job = gather()
    scored = {"priority": [], "document": [], "random": []}
    for index, (_, items) in enumerate(by_job.items()):
        for order in scored:
            value = coverage_at_k(items, args.k, order=order, seed=index)
            if value is not None:
                scored[order].append(value)

    n = len(scored["priority"])
    if n == 0:
        print("No corrected reports yet — the measurement needs doctors to have "
              "reviewed some reports first (PATCH /jobs/{id}/items/{item_id}).")
        print("The harness is ready; it will produce a number as soon as there is data.")
        return 0

    print(f"reports with at least one correction: {n}\n")
    results = {}
    for order, values in scored.items():
        mean = sum(values) / len(values)
        results[order] = round(mean, 4)
        label = {"priority": "uncertainty queue", "document": "reading top-down",
                 "random": "random order"}[order]
        print(f"  coverage@{args.k} — {label:<20}: {mean:.1%}")

    passed = results["priority"] >= TARGET_COVERAGE
    print(f"\nP7-05 gate: coverage@{args.k} ≥ {TARGET_COVERAGE:.0%}  ->  "
          f"{'PASS' if passed else 'not met'}")
    if results["priority"] <= results["random"]:
        print("WARNING: the queue is no better than random — the uncertainty signal is "
              "not working, and reviewing by priority buys nothing.")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps({
            "created_at": datetime.now(timezone.utc).isoformat(),
            "k": args.k, "reports": n, "coverage": results, "passed": passed,
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
