"""Apply the human label decisions from data/LABELING_GUIDE.md (IMPLEMENTATION.md P0-05).

Two independent mechanisms, usable together:

  --merge OLD=NEW          taxonomy change: every row labelled OLD becomes NEW
  --decisions CONFLICTS.CSV per-text decisions, read from the `decision` column of the
                           CSV produced by scripts/check_conflicts.py

Nothing is destroyed: the previous data.jsonl is copied to data/backups/ with a
timestamp before the file is rewritten (PLAN_V2 principle P1 — no information is lost).
Run scripts/check_conflicts.py afterwards; it must exit 0.
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.nlp.dataset import find_conflicts, load_jsonl  # noqa: E402


def parse_merges(pairs: list[str]) -> dict[str, str]:
    merges: dict[str, str] = {}
    for pair in pairs:
        if "=" not in pair:
            raise SystemExit(f"--merge expects OLD=NEW, got {pair!r}")
        old, new = pair.split("=", 1)
        merges[old.strip()] = new.strip()
    return merges


def read_decisions(path: str) -> dict[str, str]:
    """text -> chosen label, taking only rows where a human filled `decision`."""
    decisions: dict[str, str] = {}
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            choice = (row.get("decision") or "").strip()
            if choice:
                decisions[row["text"]] = choice
    return decisions


def backup(data_path: Path) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    dest_dir = data_path.parent / "data" / "backups"
    if not dest_dir.parent.exists():  # running against a top-level data.jsonl
        dest_dir = data_path.parent / "backups"
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f"{data_path.name}.{stamp}"
    shutil.copy2(data_path, dest)
    return dest


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Apply documented label decisions to a dataset.")
    p.add_argument("--data", default="data.jsonl")
    p.add_argument("--out", default=None, help="default: rewrite --data in place (after backup)")
    p.add_argument("--merge", action="append", default=[], metavar="OLD=NEW")
    p.add_argument("--decisions", default=None, help="CSV from scripts/check_conflicts.py")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args(argv)

    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # pragma: no cover
        pass

    data_path = Path(args.data)
    rows = load_jsonl(data_path)
    merges = parse_merges(args.merge)
    decisions = read_decisions(args.decisions) if args.decisions else {}
    before_conflicts = len(find_conflicts(rows))

    merged = decided = 0
    for row in rows:
        if row["label"] in merges:
            row["label"] = merges[row["label"]]
            merged += 1
        chosen = decisions.get(row["text"])
        if chosen and row["label"] != chosen:
            row["label"] = chosen
            decided += 1

    # De-duplicate again: merging two labels can turn a conflict into an exact duplicate.
    seen: set[tuple[str, str]] = set()
    kept: list[dict] = []
    for row in rows:
        key = (row["text"], row["label"])
        if key in seen:
            continue
        seen.add(key)
        kept.append(row)

    after_conflicts = len(find_conflicts(kept))
    print(f"rows            : {len(rows)} -> {len(kept)}")
    print(f"labels merged   : {merged}")
    print(f"decisions applied: {decided}")
    print(f"conflicts       : {before_conflicts} -> {after_conflicts}")
    if after_conflicts:
        print("WARNING: conflicts remain — fill the 'decision' column for the rest.")

    if args.dry_run:
        print("\n--dry-run: nothing written")
        return 0

    out_path = Path(args.out) if args.out else data_path
    if out_path == data_path:
        print(f"backup          : {backup(data_path)}")
    with open(out_path, "w", encoding="utf-8") as f:
        for row in kept:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote           : {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
