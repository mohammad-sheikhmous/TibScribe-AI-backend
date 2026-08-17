"""Label-conflict gate for data.jsonl (IMPLEMENTATION.md P0-04 / P0-05).

A "conflict" is one text that appears with more than one label. The model is then
asked to learn two different answers for the same input, which puts a hard ceiling
on achievable accuracy and makes the confusion matrix unreadable.

Two modes:
  * report  — write every conflict to CSV with the context needed to decide it.
  * gate    — exit code 1 while any conflict remains, so it can guard CI / a merge.

The suggestion column is a *hint*, never an automatic decision: per PLAN_V2 §5.3 the
human resolves the label taxonomy and records the rule in data/LABELING_GUIDE.md.

    python scripts/check_conflicts.py
    python scripts/check_conflicts.py --report reports/label_conflicts.csv
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.nlp.dataset import (  # noqa: E402
    class_distribution,
    dedupe_texts,
    find_conflicts,
    load_jsonl,
)

# Labels where a wrong merge has clinical consequences — never auto-suggested.
CLINICALLY_SENSITIVE = {"emergency", "pregnancy_risk", "allergy", "medication"}


def suggest(labels: list[str]) -> tuple[str, str]:
    """Return (suggested_label, reason). Empty suggestion == needs a human decision."""
    if any(l in CLINICALLY_SENSITIVE for l in labels):
        return "", "clinically sensitive — clinician decision"
    # A specialized label and its base (nutrition vs pregnancy_nutrition): the
    # specialized one carries strictly more information.
    for a in labels:
        for b in labels:
            if a != b and b.endswith(f"_{a}") or (a != b and b == f"pregnancy_{a}"):
                return b, f"'{b}' is a specialization of '{a}'"
    return "", "no rule — human decision"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Detect texts carrying more than one label.")
    p.add_argument("--data", default="data.jsonl")
    p.add_argument("--report", default=None, help="write a CSV of conflicts to this path")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args(argv)

    rows = load_jsonl(args.data)
    conflicts = find_conflicts(rows)
    dedup = dedupe_texts(rows)
    distribution = class_distribution(r["label"] for r in rows)

    if not args.quiet:
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:  # pragma: no cover - non-reconfigurable stream
            pass
        print(f"rows            : {len(rows)}")
        print(f"unique texts    : {len(rows) - dedup.removed}")
        print(f"duplicate rows  : {dedup.removed}")
        print(f"CONFLICTS       : {len(conflicts)}")
        if conflicts:
            pairs: dict[str, int] = {}
            for labels in conflicts.values():
                key = " | ".join(sorted(labels))
                pairs[key] = pairs.get(key, 0) + 1
            print("\nconflicting label pairs (most frequent first):")
            for key, count in sorted(pairs.items(), key=lambda kv: -kv[1]):
                print(f"  {count:4d}  {key}")

    if args.report:
        out = Path(args.report)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8-sig", newline="") as f:
            w = csv.writer(f)
            w.writerow(["text", "labels", "label_counts_in_dataset",
                        "suggestion", "reason", "decision"])
            for text, labels in sorted(conflicts.items()):
                counts = ", ".join(f"{l}={distribution.get(l, 0)}" for l in sorted(labels))
                suggestion, reason = suggest(sorted(labels))
                w.writerow([text, " | ".join(sorted(labels)), counts, suggestion, reason, ""])
        if not args.quiet:
            print(f"\nwrote {out} ({len(conflicts)} rows) — fill the 'decision' column")

    return 1 if conflicts else 0


if __name__ == "__main__":
    raise SystemExit(main())
