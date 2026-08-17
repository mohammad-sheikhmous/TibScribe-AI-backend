"""Export produced reports as a gold-set draft for manual labelling (P0-11 helper).

Takes the reports the pipeline actually produced (`data/results/*.json`) and writes
one line per sentence in `eval/gold` format, pre-filled with the model's own guess.
A human then corrects the `label` column — reviewing is far faster than labelling
from scratch, and the pre-filled guess is never trusted (it is what we are measuring).

    python scripts/export_gold_draft.py data/results --out eval/gold/batch_01.jsonl

Add --uncertain-first to put the model's least-confident sentences at the top: those
are where a human's time buys the most information (the same uncertainty signal that
drives active learning in P7-04).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def iter_items(report: dict):
    for section in (report.get("soap") or {}).values():
        for item in section.get("items", []):
            yield item


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Draft a gold set from produced reports.")
    p.add_argument("inputs", nargs="+", help="report .json file(s) or folder(s)")
    p.add_argument("--out", default="eval/gold/draft.jsonl")
    p.add_argument("--uncertain-first", action="store_true",
                   help="sort by ascending confidence (highest labelling value first)")
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args(argv)

    reports: list[tuple[str, dict]] = []
    for raw in args.inputs:
        path = Path(raw)
        files = sorted(path.glob("*.json")) if path.is_dir() else [path]
        for f in files:
            try:
                reports.append((f.stem, json.loads(f.read_text(encoding="utf-8"))))
            except json.JSONDecodeError:
                print(f"  skipping unreadable {f}")

    rows: list[dict] = []
    for job_id, report in reports:
        source = (report.get("audio") or {}).get("filename") or job_id
        for item in iter_items(report):
            rows.append({
                "text": item.get("text", ""),
                "label": item.get("label", ""),       # model guess — TO BE VERIFIED
                "source": source,
                "start_sec": item.get("start_sec"),
                "item_id": item.get("item_id"),
                "model_confidence": item.get("confidence"),
                "verified": False,                     # flip to true after human review
            })

    if args.uncertain_first:
        rows.sort(key=lambda r: r.get("model_confidence") or 0.0)
    if args.limit:
        rows = rows[:args.limit]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # pragma: no cover
        pass
    print(f"reports : {len(reports)}")
    print(f"sentences: {len(rows)} -> {out}")
    print("Next: verify every `label` by hand, set \"verified\": true, then run "
          "eval/eval_classifier.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
