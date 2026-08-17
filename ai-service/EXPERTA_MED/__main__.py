"""CLI: run the KBS over one report JSON, many, or a folder (a patient series).

    python -m EXPERTA_MED <report.json | folder> [more...] [--out DIR] [--name NAME]

Writes to the output dir (default EXPERTA_MED/output/):
    <name>.suggestions.json   full result (context, suggestions, audit trail)
    <name>.suggestions.md     Arabic, human-readable
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .runner import analyze_paths, result_to_markdown

_HERE = Path(__file__).resolve().parent


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="EXPERTA_MED", description="KBS over medical-scribe reports.")
    p.add_argument("inputs", nargs="+", help="report .json file(s) or folder(s) of a patient series")
    p.add_argument("--out", default=str(_HERE / "output"), help="output directory")
    p.add_argument("--name", default="result", help="output basename")
    args = p.parse_args(argv)

    # Windows consoles default to a non-UTF-8 codepage; force UTF-8 so Arabic prints.
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    result = analyze_paths(args.inputs)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    json_path = out / f"{args.name}.suggestions.json"
    md_path = out / f"{args.name}.suggestions.md"
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(result_to_markdown(result), encoding="utf-8")

    print(f"reports analyzed : {result['reports_total']}")
    print(f"facts extracted  : {result['extracted_facts_count']}")
    print(f"suggestions      : {len(result['suggestions'])}")
    for s in result["suggestions"]:
        print(f"  [{s['severity']:>8}] {s['rule_id']}  {s['title_ar']}")
    print(f"\nwrote: {json_path}\n       {md_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
