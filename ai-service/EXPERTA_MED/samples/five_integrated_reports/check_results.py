from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
OUTPUT = ROOT / "EXPERTA_MED" / "output"
EXPECTED = json.loads((HERE / "expected_rules.json").read_text(encoding="utf-8"))

NAME_MAP = {
    "integrated_1_early_pregnancy.json": "integrated_1_early_pregnancy.suggestions.json",
    "integrated_2_high_risk_pregnancy.json": "integrated_2_high_risk_pregnancy.suggestions.json",
    "integrated_3_complicated_labour.json": "integrated_3_complicated_labour.suggestions.json",
    "integrated_4_normal_delivery_newborn.json": "integrated_4_normal_delivery_newborn.suggestions.json",
    "integrated_5_postpartum_discharge.json": "integrated_5_postpartum_discharge.suggestions.json",
}

failed = 0
for report_name, spec in EXPECTED.items():
    result_path = OUTPUT / NAME_MAP[report_name]
    if not result_path.exists():
        print(f"[MISSING OUTPUT] {result_path}")
        failed += 1
        continue

    result = json.loads(result_path.read_text(encoding="utf-8"))
    fired = {x["rule_id"] for x in result.get("suggestions", [])}

    missing = [x for x in spec["must_include"] if x not in fired]
    forbidden = [x for x in spec["must_not_include"] if x in fired]

    if not missing and not forbidden:
        print(f"[PASS] {report_name}")
    else:
        print(f"[FAIL] {report_name}")
        for rule in missing:
            print(f"  missing: {rule}")
        for rule in forbidden:
            print(f"  unexpected: {rule}")
        failed += 1

raise SystemExit(1 if failed else 0)
