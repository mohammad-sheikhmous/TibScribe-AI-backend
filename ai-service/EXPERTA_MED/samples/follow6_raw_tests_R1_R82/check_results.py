from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT_PARENT = Path(__file__).resolve().parents[3]
OUTPUT = PROJECT_PARENT / "EXPERTA_MED" / "output"
EXPECTED = json.loads((HERE / "expected_rules.json").read_text(encoding="utf-8"))

failed = 0
for filename, spec in EXPECTED.items():
    stem = Path(filename).stem
    output = OUTPUT / f"{stem}.suggestions.json"
    if not output.exists():
        print(f"[MISSING] {filename}")
        failed += 1
        continue
    data = json.loads(output.read_text(encoding="utf-8"))
    fired = {x.get("rule_id") for x in data.get("suggestions", [])}
    target = spec["target_rule"]
    if target in fired:
        print(f"[PASS] {filename} -> {target}")
    else:
        print(f"[FAIL] {filename} missing {target}")
        failed += 1

raise SystemExit(1 if failed else 0)
