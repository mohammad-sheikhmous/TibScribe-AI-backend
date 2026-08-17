from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT_PARENT = Path(__file__).resolve().parents[3]
OUTPUT = PROJECT_PARENT / "EXPERTA_MED" / "output"
EXPECTED = json.loads(
    (HERE / "expected_minimum_rules.json").read_text(encoding="utf-8")
)

failed = 0
for report_file, spec in EXPECTED.items():
    stem = Path(report_file).stem
    result_file = OUTPUT / f"{stem}.suggestions.json"

    if not result_file.exists():
        print(f"[MISSING OUTPUT] {result_file}")
        failed += 1
        continue

    result = json.loads(result_file.read_text(encoding="utf-8"))
    fired = {
        item.get("rule_id")
        for item in result.get("suggestions", [])
        if item.get("rule_id")
    }

    missing = [rule for rule in spec.get("must_include", []) if rule not in fired]
    forbidden = [rule for rule in spec.get("must_not_include", []) if rule in fired]

    if not missing and not forbidden:
        print(f"[PASS] {report_file}")
    else:
        print(f"[FAIL] {report_file}")
        for rule in missing:
            print(f"  missing: {rule}")
        for rule in forbidden:
            print(f"  unexpected: {rule}")
        failed += 1

    extra_modern = sorted(
        r for r in fired
        if isinstance(r, str)
        and r.startswith(("FOLLOW1-", "FOLLOW2-", "FOLLOW3-", "FOLLOW4-"))
        and r not in spec.get("must_include", [])
    )
    if extra_modern:
        print("  additional modern rules:", ", ".join(extra_modern))

raise SystemExit(1 if failed else 0)
